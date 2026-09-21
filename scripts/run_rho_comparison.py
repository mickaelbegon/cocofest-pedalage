#!/usr/bin/env python3
"""Launch the four online IPOPT arms of the 0.2 Nm / 100-cycle comparison.

The iterative FHO is intentionally separate: it has a continuation workflow
and is an offline full-horizon reference rather than an online controller.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "local-results/rho-comparison-0p2-100"
ARMS = {
    "rho-uniform-1": "rho-uniform-1.config.json",
    "rho-pace-uniform-1": "rho-pace-uniform-1.config.json",
    "rho-physio": "rho-physio.config.json",
    "rho-bo": "rho-bo.config.json",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--template-root", type=Path,
                        help="Directory containing the immutable 100-cycle protocol templates.")
    parser.add_argument("--cycles", type=int, default=100,
                        help="Executed-cycle safety ceiling; PACE keeps its 100-cycle prediction horizon.")
    parser.add_argument("--prefix", type=Path,
                        default=Path("/home/mickaelbegon/miniforge3/envs/cocofest-rho32"))
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    template_root = (args.template_root or root).expanduser().resolve()
    prefix = args.prefix.expanduser().resolve()
    if args.cycles < 1 or args.cycles > 2000:
        raise ValueError("--cycles must be in 1..2000")
    sys.path.insert(0, str(ROOT))
    from cocofest.simulation import CapabilityRegistry, SimulationConfig, build_launch_plan

    root.mkdir(parents=True, exist_ok=True)
    unit_weights = json.loads((template_root / "weights-uniform-1.json").read_text(encoding="utf-8"))
    unit_weights["policy"]["max_cycles"] = args.cycles
    # The prediction horizon is intentionally not tied to endurance duration.
    unit_weights["policy"]["projection_horizon_cycles"] = 100
    unit_weights_path = root / "weights-uniform-1.json"
    if unit_weights_path.exists():
        raise FileExistsError(f"Uniform PACE weights already exist: {unit_weights_path}")
    unit_weights_path.write_text(json.dumps(unit_weights, indent=2) + "\n", encoding="utf-8")

    plans = {}
    for arm, filename in ARMS.items():
        source_config_path = template_root / filename
        config_data = json.loads(source_config_path.read_text(encoding="utf-8"))
        config_data["cycles"] = args.cycles
        config_data["output_root"] = str(root / arm)
        if arm == "rho-pace-uniform-1":
            config_data["weights_config"] = str(unit_weights_path)
        config_path = root / filename
        if config_path.exists():
            raise FileExistsError(f"Configuration already exists: {config_path}")
        config_path.write_text(json.dumps(config_data, indent=2) + "\n", encoding="utf-8")
        config = SimulationConfig.from_dict(config_data)
        CapabilityRegistry.require_valid(config)
        plan = build_launch_plan(config, prefix, ROOT)
        if plan.result_json and plan.result_json.exists():
            raise FileExistsError(f"{arm}: result already exists: {plan.result_json}")
        plan.cwd.mkdir(parents=True, exist_ok=True)
        plans[arm] = plan

    manifest = {
        "schema": "rho-comparison-v1",
        "protocol": {
            "resistance_nm": 0.2, "requested_cycles": args.cycles,
            "model": "ding_triceps_alpha_a_x1", "solver": "ipopt-ma57",
            "pace_initial_weights": {"Biceps": 1.0, "Delt_ant": 1.0,
                                     "Delt_post": 1.0, "Triceps": 1.0},
            "pace_projection_horizon_cycles": 100,
            "rho_bo_source_trial": 172,
        },
        "started_at": time.time(),
        "arms": {name: {"config": str(root / filename), "command": list(plan.argv),
                         "result_json": str(plan.result_json), "status": "starting"}
                 for (name, filename), plan in zip(ARMS.items(), plans.values())},
    }
    (root / "online-runtime.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    children: dict[str, subprocess.Popen] = {}
    try:
        for arm, plan in plans.items():
            environment = os.environ.copy()
            environment.update(plan.environment_updates)
            environment["PYTHONPATH"] = str(ROOT)
            log_path = root / arm / "runner.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(plan.argv, cwd=plan.cwd, env=environment,
                                       stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            handle.close()
            children[arm] = process
            manifest["arms"][arm].update(status="running", pid=process.pid, log=str(log_path))
            (root / "online-runtime.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"started {arm}: pid={process.pid}", flush=True)

        while children:
            for arm, process in list(children.items()):
                code = process.poll()
                if code is not None:
                    manifest["arms"][arm].update(status="completed" if code == 0 else "failed",
                                                   returncode=code, ended_at=time.time())
                    del children[arm]
                    print(f"finished {arm}: returncode={code}", flush=True)
            (root / "online-runtime.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            time.sleep(5)
    except KeyboardInterrupt:
        for process in children.values():
            os.killpg(process.pid, signal.SIGTERM)
        raise
    manifest["finished_at"] = time.time()
    (root / "online-runtime.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0 if all(arm["status"] == "completed" for arm in manifest["arms"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
