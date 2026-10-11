#!/usr/bin/env python3
"""Asynchronous fixed-weight BO for a unilateral prescribed-isokinetic RHO.

Each Optuna observation is an independent RHO endurance run.  The score is
the certified RHO prefix length; a failed following window is evidence, not a
certified physiological stop.  Eight subprocesses can run concurrently while
the coordinator keeps the Optuna journal serial and restartable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import optuna


MUSCLES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")


def _certified_cycles(result_path: Path) -> int:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    result = data["results"][0]
    return int(result.get("covered_cycles") or 0)


def _run_trial(number: int, weights: tuple[float, float, float, float], args: argparse.Namespace) -> dict[str, Any]:
    root = args.output / "trials" / f"trial_{number:03d}"
    root.mkdir(parents=True, exist_ok=False)
    result = root / "result.json"
    command = [
        args.python, "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py",
        "--solvers", "ipopt", "--objective", "fatigue", "--objective-shape", "quadratic",
        "--formulation", "isokinetic", "--mechanical-formulation", "reduced",
        "--isokinetic-kinematics", "prescribed", "--isokinetic-omega", str(-2.0 * 3.141592653589793),
        "--energy-equivalent-torque", str(args.resistance), "--load-torque-min", "-4", "--load-torque-max", "4",
        "--stimulations-per-cycle", "30", "--ipopt-profile", "periodic_collocation",
        "--ipopt-collocation-method", "radau", "--ipopt-collocation-degree", "5", "--ipopt-use-sx",
        "--ipopt-linear-solver", "ma57", "--warmup-ipopt-linear-solver", "ma57",
        "--ipopt-hsl-library", args.hsl_library, "--ipopt-max-iter", "4000",
        "--nlp-tolerance", "1e-8", "--primal-feasibility-threshold", "1e-5",
        "--ipopt-enforce-start-constraints", "--cycles-per-window", "1", "--n-windows", str(args.max_cycles),
        "--n-threads", "1", "--compact-rho-output", "--retry-failed-rho-without-advance",
        "--parametric-fatigue-weights", "--fatigue-weight-values", *(f"{item:.12g}" for item in weights),
        "--output-json", str(result),
    ]
    environment = {**os.environ, "PYTHONPATH": str(args.root), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    completed = subprocess.run(["taskset", "-c", args.cpu_set, *command], cwd=args.root, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)
    (root / "driver.log").write_text(completed.stdout, encoding="utf-8")
    cycles = _certified_cycles(result) if result.is_file() else 0
    observation = {"trial": number, "weights": dict(zip(MUSCLES, weights, strict=True)), "returncode": completed.returncode,
                   "certified_cycles": cycles, "result_path": str(result)}
    (root / "observation.json").write_text(json.dumps(observation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hsl-library", required=True)
    parser.add_argument("--resistance", type=float, default=1.3)
    parser.add_argument("--max-cycles", type=int, default=160)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu-set", default="12-19")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--initial-weights", type=float, nargs=4, metavar=MUSCLES)
    args = parser.parse_args()
    args.output = args.output.resolve(); args.root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    study = optuna.create_study(direction="maximize", study_name="rho-bo-isokinetic-unit-1p3",
                                sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=16),
                                storage=f"sqlite:///{args.output / 'study.sqlite3'}")
    if args.initial_weights is not None:
        study.enqueue_trial(dict(zip(MUSCLES, args.initial_weights, strict=True)))
    (args.output / "campaign.json").write_text(json.dumps({"resistance_nm": args.resistance, "trials": args.trials,
        "workers": args.workers, "cpu_affinity": args.cpu_set, "max_cycles": args.max_cycles,
        "objective": "certified_rho_cycles", "weight_order": MUSCLES,
        "initial_weights": args.initial_weights}, indent=2) + "\n", encoding="utf-8")
    active: dict[Any, Any] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while len(study.trials) < args.trials or active:
            while len(active) < args.workers and len(study.trials) < args.trials:
                trial = study.ask()
                weights = tuple(trial.suggest_float(name, 0.0, 1.0) for name in MUSCLES)
                active[pool.submit(_run_trial, trial.number, weights, args)] = trial
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                trial = active.pop(future)
                try:
                    record = future.result()
                    trial.set_user_attr("record", record)
                    study.tell(trial, float(record["certified_cycles"]))
                except Exception as error:
                    trial.set_user_attr("error", f"{type(error).__name__}: {error}")
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                completed = [item for item in study.trials if item.value is not None]
                best = max(completed, key=lambda item: item.value, default=None)
                (args.output / "summary.json").write_text(json.dumps({"completed": len(completed), "best_trial": None if best is None else best.number,
                    "best_cycles": None if best is None else best.value, "best_weights": None if best is None else best.params}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
