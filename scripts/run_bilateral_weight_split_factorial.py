#!/usr/bin/env python3
"""Run the asymmetric 1.92 Nm weight-by-work-split endurance factorial.

Each condition starts from the same cold physical state and uses the same
certified frozen-endpoint protocol.  The script writes its exact inputs before
launching four independent two-arm process coordinators on separate CPU pairs.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.simulation.bilateral_endurance_campaign import assess_summary


SOURCE = ROOT / "asymmetric-sides-r192-rho-bo-20260928"
BO_CAMPAIGN = SOURCE / "bo-fixed-split-50-50-campaign.json"
BO_WINNER = SOURCE / "bo-fixed-split-50-50-results/candidates/64653f690b1599ea/cycles-0240/input.json"
OUTPUT = SOURCE / "factorial-weights-split-20261003"
RUNNER = ROOT / "scripts/run_independent_isokinetic_arms.py"
CONDITIONS = (
    ("unit_fixed", False, False, 8, 9),
    ("bo_fixed", True, False, 10, 11),
    ("unit_adaptive", False, True, 12, 13),
    ("bo_adaptive", True, True, 14, 15),
)
CODE_FILES = (
    ROOT / "cocofest/simulation/independent_arms_process.py",
    ROOT / "cocofest/optimization/independent_arm_rho_pace.py",
    ROOT / "cocofest/simulation/bilateral_endurance_campaign.py",
    ROOT / "examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py",
    ROOT / "cocofest/custom_objectives.py",
    RUNNER,
    Path(__file__),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def build_plan(output: Path, cycles: int) -> dict:
    source = json.loads(BO_CAMPAIGN.read_text(encoding="utf-8"))
    winner = json.loads(BO_WINNER.read_text(encoding="utf-8"))
    base = source["campaign"]["base_payload"]
    expected = {"solver": "ipopt", "formulation": "isokinetic", "cycles_per_window": 1,
                "parallel": True, "stimulations_per_cycle": 30, "parametric_fatigue_weights": True}
    if any(base.get(key) != value for key, value in expected.items()):
        raise ValueError("The reference campaign no longer matches the declared numerical protocol")
    if winner["resistance_pace"]["capacity_feedback"] is not False:
        raise ValueError("The BO winner is not from the strict 50–50 study")
    if winner["endurance_candidate"]["identifier"] != "64653f690b1599ea":
        raise ValueError("The BO winner identifier changed")
    if winner["right_equivalent_mean_torque_nm"] != 0.96 or winner["left_equivalent_mean_torque_nm"] != 0.96:
        raise ValueError("The selected BO reference is not the 1.92 Nm, 50–50 experiment")
    pairs = [cpu for _, _, _, right, left in CONDITIONS for cpu in (right, left)]
    if len(pairs) != len(set(pairs)):
        raise ValueError("CPU pair collision")
    available = os.sched_getaffinity(0)
    if not set(pairs).issubset(available):
        raise RuntimeError(f"Unavailable reserved CPUs: {sorted(set(pairs) - available)}")
    unit = {name: 1.0 for name in ("Delt_ant", "Delt_post", "Biceps", "Triceps")}
    records = []
    for name, bo_weights, feedback, right_cpu, left_cpu in CONDITIONS:
        payload = copy.deepcopy(base)
        payload["cycles"] = cycles
        payload["prepared_restart_checkpoint_cycles"] = [120, 140, 160]
        payload["solver_cpu_affinity"] = {"right": right_cpu, "left": left_cpu}
        payload["right_equivalent_mean_torque_nm"] = 0.96
        payload["left_equivalent_mean_torque_nm"] = 0.96
        payload["muscle_pace"]["adaptation_enabled"] = False
        payload["muscle_pace"]["right_initial_weights"] = (
            copy.deepcopy(winner["muscle_pace"]["right_initial_weights"]) if bo_weights else unit.copy())
        payload["muscle_pace"]["left_initial_weights"] = (
            copy.deepcopy(winner["muscle_pace"]["left_initial_weights"]) if bo_weights else unit.copy())
        payload["muscle_pace"]["initial_weight_basis"] = (
            "Best fixed-split BO candidate 64653f690b1599ea, reused unchanged" if bo_weights
            else "Unit baseline weights; no optimization")
        payload["resistance_pace"]["capacity_feedback"] = feedback
        payload["resistance_pace"]["initial_split_policy"] = "manual"
        payload["resistance_pace"]["initial_right_fraction"] = 0.5
        payload["resistance_pace"]["total_equivalent_mean_torque_nm"] = 1.92
        records.append({"name": name, "weights": "bo_50_50" if bo_weights else "unit",
                        "split": "capacity_feedback" if feedback else "fixed_50_50",
                        "right_cpu": right_cpu, "left_cpu": left_cpu,
                        "input": str(output / name / "input.json"), "payload": payload})
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "source_campaign": str(BO_CAMPAIGN),
            "source_winner": str(BO_WINNER), "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (*CODE_FILES, BO_CAMPAIGN, BO_WINNER)},
            "target_cycles": cycles, "total_equivalent_mean_torque_nm": 1.92,
            "comparison": "2x2 cold-start matched protocol; BO weights selected on earlier fixed-split study",
            "conditions": records}


def run(output: Path, cycles: int, dry_run: bool) -> None:
    if output.exists():
        raise FileExistsError(f"Campaign directory already exists: {output}")
    plan = build_plan(output, cycles)
    if dry_run:
        print(json.dumps({key: value for key, value in plan.items() if key != "conditions"}, indent=2))
        for item in plan["conditions"]:
            print(item["name"], item["split"], item["weights"], item["right_cpu"], item["left_cpu"])
        return
    output.mkdir(parents=True)
    manifest = copy.deepcopy(plan)
    for item in manifest["conditions"]:
        del item["payload"]
    write_json(output / "manifest.json", manifest)
    for item in plan["conditions"]:
        condition_root = output / item["name"]
        condition_root.mkdir()
        write_json(condition_root / "input.json", item["payload"])
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    live = {}
    for item in plan["conditions"]:
        name = item["name"]
        condition_root = output / name
        log = (condition_root / "run.log").open("w", encoding="utf-8")
        command = [sys.executable, str(RUNNER), "--config", str(condition_root / "input.json"),
                   "--output-dir", str(condition_root / "results")]
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        live[name] = (process, log)
        write_json(condition_root / "launch.json", {"command": command, "pid": process.pid,
                   "started_utc": datetime.now(timezone.utc).isoformat()})
        print(f"launched {name} pid={process.pid} cpu={item['right_cpu']},{item['left_cpu']}", flush=True)
    results = {}
    while live:
        time.sleep(5)
        for name, (process, log) in list(live.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            result_root = output / name
            summary_path = result_root / "results/summary.json"
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                observation = assess_summary(summary, cycles).to_dict()
                right = summary["arms"]["right"]["validated_cycles"]
                left = summary["arms"]["left"]["validated_cycles"]
            except Exception as error:
                observation = {"status": "technical_failure", "reason": f"{type(error).__name__}: {error}"}
                right = left = None
            result = {"exit_code": code, "observation": observation,
                      "right_validated_cycles": right, "left_validated_cycles": left,
                      "ended_utc": datetime.now(timezone.utc).isoformat()}
            write_json(result_root / "receipt.json", result)
            results[name] = result
            write_json(output / "status.json", results)
            print(f"finished {name} exit={code} status={observation['status']} "
                  f"common={observation.get('validated_cycles')}", flush=True)
            del live[name]
    write_json(output / "final.json", results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cycles", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.cycles < 240:
        raise ValueError("At least 240 cycles are required for comparison with both BO archives")
    run(args.output.resolve(), args.cycles, args.dry_run)


if __name__ == "__main__":
    main()
