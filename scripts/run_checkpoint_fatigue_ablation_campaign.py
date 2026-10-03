#!/usr/bin/env python3
"""Two dedicated CPU workers for the predeclared c120/c140 objective ablation."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_checkpoint_fatigue_ablation import VARIANTS
from scripts.probe_independent_rho_task_reserve import _load_arm
from cocofest.simulation.independent_arms_process import _atomic_json


def main():
    root = ROOT / "asymmetric-sides-r192-rho-bo-20260928"
    output = root / "fatigue-cost-checkpoint-ablation-20261003"
    manifest = output / "campaign.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    if not {16, 17}.issubset(os.sched_getaffinity(0)):
        raise RuntimeError("Dedicated campaign CPUs unavailable")
    jobs = []
    for side, cpu in (("left", 16), ("right", 17)):
        for cycle in (120, 140):
            receipt = root / "task-reserve-rho-instrumented-20260930/results/checkpoints" / f"cycle-{cycle}/receipt.json"
            source, _ = _load_arm(receipt, side)
            source.verify_files()
            for variant in VARIANTS:
                destination = output / f"{side}-c{cycle}-{variant}"
                if destination.exists():
                    raise FileExistsError(destination)
                jobs.append({"side": side, "cpu": cpu, "source_cycle": cycle, "variant": variant,
                    "receipt": str(receipt), "receipt_sha256": sha256(receipt.read_bytes()).hexdigest(),
                    "output": str(destination), "log": str(output / f"{side}-c{cycle}-{variant}.log")})
    script = ROOT / "scripts/run_checkpoint_fatigue_ablation.py"
    document = {"kind": "predeclared_checkpoint_cost_ablation_campaign", "jobs": jobs,
        "horizon": 20, "observed_prefixes": [1, 5, 20], "maximum_parallel_jobs": 2,
        "cpu_affinity": {"left": 16, "right": 17}, "numeric_threads": 1,
        "runner_sha256": sha256(script.read_bytes()).hexdigest(),
        "preflight": "All 4 receipts/models/archives authenticated; left c120 integral_quadratic one-cycle solve independently audited; 5 objective-contract tests passed",
        "promotion": "All 20 cycles audited; independent endpoint work margin improvement >0.01 vs integral_quadratic at both checkpoints and no arm regression. Short fatigue cost decrease is insufficient.",
        "scaling": "10000 times source fatigue factor; terminal scaled by cycle duration; no local-gradient matching"}
    _atomic_json(manifest, document)
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLCONFIGDIR": "/tmp/cocofest-fatigue-ablation-mpl"}

    def run_side(side):
        results = []
        for job in [j for j in jobs if j["side"] == side]:
            if sha256(script.read_bytes()).hexdigest() != document["runner_sha256"]:
                raise RuntimeError("Runner changed during campaign")
            cmd = [sys.executable, str(script), "--receipt", job["receipt"], "--side", side,
                   "--variant", job["variant"], "--output", job["output"], "--cpu", str(job["cpu"])]
            with open(job["log"], "x") as log:
                completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            row = {**job, "returncode": completed.returncode}
            results.append(row)
            _atomic_json(output / f"{side}-progress.json", {"jobs": results})
            print(f"{side} c{job['source_cycle']} {job['variant']}: returncode={completed.returncode}", flush=True)
            if completed.returncode:
                break
        return results

    with ThreadPoolExecutor(max_workers=2) as pool:
        groups = list(pool.map(run_side, ("left", "right")))
    _atomic_json(output / "campaign-completion.json", {"jobs": [job for group in groups for job in group]})


if __name__ == "__main__":
    main()
