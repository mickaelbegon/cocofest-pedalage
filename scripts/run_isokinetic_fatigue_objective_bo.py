#!/usr/bin/env python3
"""Time-boxed BO screens for three fatigue-oriented RHO objective variants.

This is deliberately a screening runner: it ranks candidates by solver-covered
cycles, then terminal reserve. It does not certify the winning candidate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import optuna

MUSCLES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
VARIANTS = {
    "worst_reserve": {"objective": "fatigue", "terminal_reserve_weight": 0.05},
    "fatigue_balance": {"objective": "fatigue", "fatigue_balance_weight": 0.05},
    "fatigue_plus_activation": {"objective": "fatigue,control"},
}


def evaluate(trial_number: int, weights: tuple[float, ...], args: argparse.Namespace) -> dict[str, Any]:
    out = args.output / "trials" / f"trial_{trial_number:04d}"
    out.mkdir(parents=True, exist_ok=False)
    result = out / "result.json"
    variant = VARIANTS[args.variant]
    command = [
        args.python, "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py",
        "--solvers", "ipopt", "--objective", variant["objective"], "--objective-shape", "quadratic",
        "--parametric-fatigue-weights", "--fatigue-weight-values", *(f"{x:.12g}" for x in weights),
        "--formulation", "isokinetic", "--mechanical-formulation", "reduced",
        "--isokinetic-kinematics", "prescribed", "--isokinetic-omega", str(-2.0 * np.pi),
        "--energy-equivalent-torque", str(args.resistance), "--load-torque-min", "-4", "--load-torque-max", "4",
        "--stimulations-per-cycle", "30", "--ipopt-profile", "periodic_collocation",
        "--ipopt-collocation-method", "radau", "--ipopt-collocation-degree", "5", "--ipopt-use-sx",
        "--ipopt-linear-solver", "ma57", "--warmup-ipopt-linear-solver", "ma57",
        "--ipopt-hsl-library", args.hsl_library, "--ipopt-max-iter", "4000",
        "--nlp-tolerance", "1e-8", "--primal-feasibility-threshold", "1e-5",
        "--ipopt-enforce-start-constraints", "--cycles-per-window", "1", "--n-windows", str(args.max_cycles),
        "--n-threads", "1", "--compact-rho-output", "--retry-failed-rho-without-advance",
        "--common-initial-solution", str(args.shared_seed), "--output-json", str(result),
    ]
    for key in ("terminal_reserve_weight", "fatigue_balance_weight"):
        if key in variant:
            command += [f"--{key.replace('_', '-')}", str(variant[key])]
    if args.ding_local:
        command = [args.python, "scripts/benchmark_ding_radau5_local_ab.py", "local", "--", *command[2:]]
    env = {**os.environ, "PYTHONPATH": str(args.root), "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
           "COCOFEST_DEFERRED_RHO_SUMMARY": "1", "DING_ONLINE_JOURNAL": "1"}
    if args.ding_local:
        env["DING_LOCAL_SAVE_INPUTS"] = "0"
    remaining = max(0.0, min(args.deadline - time.monotonic(), args.candidate_timeout_s))
    started = time.monotonic()
    timed_out = remaining <= 0
    returncode = 124 if timed_out else None
    output_text = ""
    if not timed_out:
        try:
            proc = subprocess.run(["taskset", "-c", args.cpu_set, *command], cwd=args.root,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  env=env, timeout=remaining)
            returncode, output_text = proc.returncode, proc.stdout or ""
        except subprocess.TimeoutExpired as exc:
            timed_out, returncode = True, 124
            captured = exc.stdout or ""
            output_text = captured.decode(errors="replace") if isinstance(captured, bytes) else captured
    elapsed = time.monotonic() - started
    (out / "driver.log").write_text(output_text, encoding="utf-8")
    payload = {}
    cycles, reserve = 0, None
    if result.is_file():
        try:
            payload = json.loads(result.read_text(encoding="utf-8"))["results"][0]
            cycles = int(payload.get("bo_screen_cycles_unvalidated", payload.get("covered_cycles")) or 0)
            reserve = payload.get("bo_screen_terminal_min_capacity_ratio_unvalidated")
            reserve = None if reserve is None else float(reserve)
        except (OSError, ValueError, KeyError, TypeError):
            payload = {}
    score = cycles + (0.999 * float(np.clip(reserve, 0.0, 1.0)) if reserve is not None else 0.0)
    row = {"trial": trial_number, "variant": args.variant, "objective": variant,
           "fatigue_weights": dict(zip(MUSCLES, weights, strict=True)), "returncode": returncode,
           "timed_out": timed_out, "cycles_screen_unvalidated": cycles,
           "terminal_min_capacity_unvalidated": reserve, "score_unvalidated": score,
           "wall_time_s": elapsed, "wall_s_per_screen_cycle": elapsed / cycles if cycles else None,
           "result_path": str(result)}
    (out / "observation.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return row


def prepare_seed(args: argparse.Namespace) -> Path:
    """Build one feasible, metadata-compatible one-cycle seed for this OCP."""
    seed_dir = args.output / "shared-seed"
    seed_dir.mkdir()
    seed, result = seed_dir / "one-cycle-unit-fatigue.npz", seed_dir / "result.json"
    variant = VARIANTS[args.variant]
    command = [
        args.python, "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py",
        "--solvers", "ipopt", "--objective", variant["objective"], "--objective-shape", "quadratic",
        "--parametric-fatigue-weights", "--fatigue-weight-values", "1", "1", "1", "1",
        "--formulation", "isokinetic", "--mechanical-formulation", "reduced",
        "--isokinetic-kinematics", "prescribed", "--isokinetic-omega", str(-2.0 * np.pi),
        "--energy-equivalent-torque", str(args.resistance), "--load-torque-min", "-4", "--load-torque-max", "4",
        "--stimulations-per-cycle", "30", "--ipopt-profile", "periodic_collocation",
        "--ipopt-collocation-method", "radau", "--ipopt-collocation-degree", "5", "--ipopt-use-sx",
        "--ipopt-linear-solver", "ma57", "--warmup-ipopt-linear-solver", "ma57",
        "--ipopt-hsl-library", args.hsl_library, "--ipopt-max-iter", "4000",
        "--nlp-tolerance", "1e-8", "--primal-feasibility-threshold", "1e-5",
        "--ipopt-enforce-start-constraints", "--cycles-per-window", "1", "--n-windows", "1",
        "--n-threads", "1", "--compact-rho-output", "--output-json", str(result),
        "--common-initial-solution-output", str(seed), "--allow-primal-feasible-common-initial-solution-output",
    ]
    for key in ("terminal_reserve_weight", "fatigue_balance_weight"):
        if key in variant:
            command += [f"--{key.replace('_', '-')}", str(variant[key])]
    env = {**os.environ, "PYTHONPATH": str(args.root), "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    proc = subprocess.run(["taskset", "-c", args.cpu_set, *command], cwd=args.root,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                          timeout=args.seed_timeout_s)
    (seed_dir / "driver.log").write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0 or not seed.is_file() or not result.is_file():
        raise RuntimeError(f"Compatible one-cycle seed failed (rc={proc.returncode}); see {seed_dir}")
    check = json.loads(result.read_text(encoding="utf-8"))["results"][0]
    if int(check.get("covered_cycles") or 0) < 1:
        raise RuntimeError(f"Compatible seed did not solve one RHO cycle; see {seed_dir}")
    return seed


def validate_best(best: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Audit the selected weights once, with the same OCP and full RHO reporting."""
    out = args.output / "final-validation"
    out.mkdir(exist_ok=False)
    result = out / "result.json"
    variant = VARIANTS[args.variant]
    weights = tuple(float(best["fatigue_weights"][name]) for name in MUSCLES)
    command = [
        args.python, "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py",
        "--solvers", "ipopt", "--objective", variant["objective"], "--objective-shape", "quadratic",
        "--parametric-fatigue-weights", "--fatigue-weight-values", *(f"{x:.12g}" for x in weights),
        "--formulation", "isokinetic", "--mechanical-formulation", "reduced",
        "--isokinetic-kinematics", "prescribed", "--isokinetic-omega", str(-2.0 * np.pi),
        "--energy-equivalent-torque", str(args.resistance), "--load-torque-min", "-4", "--load-torque-max", "4",
        "--stimulations-per-cycle", "30", "--ipopt-profile", "periodic_collocation",
        "--ipopt-collocation-method", "radau", "--ipopt-collocation-degree", "5", "--ipopt-use-sx",
        "--ipopt-linear-solver", "ma57", "--warmup-ipopt-linear-solver", "ma57",
        "--ipopt-hsl-library", args.hsl_library, "--ipopt-max-iter", "4000",
        "--nlp-tolerance", "1e-8", "--primal-feasibility-threshold", "1e-5",
        "--ipopt-enforce-start-constraints", "--cycles-per-window", "1", "--n-windows", str(args.max_cycles),
        "--n-threads", "1", "--compact-rho-output", "--retry-failed-rho-without-advance",
        "--common-initial-solution", str(args.shared_seed), "--output-json", str(result),
    ]
    for key in ("terminal_reserve_weight", "fatigue_balance_weight"):
        if key in variant:
            command += [f"--{key.replace('_', '-')}", str(variant[key])]
    if args.ding_local:
        command = [args.python, "scripts/benchmark_ding_radau5_local_ab.py", "local", "--", *command[2:]]
    env = {**os.environ, "PYTHONPATH": str(args.root), "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
           "COCOFEST_COMPACT_FINAL_SUMMARY": "1", "DING_ONLINE_JOURNAL": "1"}
    if args.ding_local:
        env["DING_LOCAL_SAVE_INPUTS"] = "0"
    started = time.monotonic()
    timed_out = False
    with (out / "driver.log").open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(["taskset", "-c", args.cpu_set, *command], cwd=args.root,
                                  stdout=log, stderr=subprocess.STDOUT, env=env,
                                  timeout=args.validation_timeout_s)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out, returncode = True, 124
    payload: dict[str, Any] = {}
    if result.is_file():
        try:
            payload = json.loads(result.read_text(encoding="utf-8"))["results"][0]
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            pass
    record = {
        "status": "timed_out" if timed_out else ("completed" if returncode == 0 and payload else "failed"),
        "returncode": returncode, "wall_time_s": time.monotonic() - started,
        "source_trial": best["trial"], "fatigue_weights": best["fatigue_weights"],
        "covered_cycles": payload.get("covered_cycles"), "validated_cycles": payload.get("validated_cycles"),
        "physiologically_validated_cycles": payload.get("physiologically_validated_cycles"),
        "solver_success": payload.get("solver_success"), "physical_success": payload.get("physical_success"),
        "first_failed_rho": payload.get("first_failed_rho"), "result_path": str(result),
    }
    (out / "validation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--hsl-library", required=True)
    parser.add_argument("--cpu-set", required=True)
    parser.add_argument("--resistance", type=float, default=0.73)
    parser.add_argument("--max-cycles", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--time-limit-s", type=float, default=600)
    parser.add_argument("--candidate-timeout-s", type=float, default=2400)
    parser.add_argument("--final-validate", action="store_true")
    parser.add_argument("--validation-timeout-s", type=float, default=3600)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--startup-trials", type=int, default=8)
    parser.add_argument("--seed-timeout-s", type=float, default=120)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--warm-start-study", type=Path,
                        help="Import completed Optuna trials from an earlier matching campaign.")
    parser.add_argument("--ding-local", action="store_true",
                        help="Use exact local Ding state elimination with prescribed isokinetic mechanics.")
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    if args.warm_start_study is not None:
        args.warm_start_study = args.warm_start_study.resolve()
        source_campaign = json.loads((args.warm_start_study.parent / "campaign.json").read_text(encoding="utf-8"))
        expected = {"variant": args.variant, "resistance_nm": args.resistance,
                    "max_cycles": args.max_cycles, "ding_local_reduction": args.ding_local,
                    "objective_definition": VARIANTS[args.variant]}
        for key, value in expected.items():
            if source_campaign.get(key) != value:
                raise ValueError(f"Warm-start campaign {key} mismatch: {source_campaign.get(key)!r} != {value!r}")
    args.output.mkdir(parents=True, exist_ok=False)
    seed_start = time.monotonic()
    args.shared_seed = prepare_seed(args)
    seed_wall_s = time.monotonic() - seed_start
    args.deadline = time.monotonic() + args.time_limit_s
    study = optuna.create_study(direction="maximize", study_name=f"fatigue-objective-{args.variant}-{args.output.name}",
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.startup_trials),
        storage=f"sqlite:///{args.output / 'study.sqlite3'}")
    imported_completed = 0
    source_summary: dict[str, Any] = {}
    if args.warm_start_study is not None:
        source_storage = f"sqlite:///{args.warm_start_study}"
        source_studies = optuna.get_all_study_summaries(storage=source_storage)
        if len(source_studies) != 1:
            raise ValueError(f"Expected one source Optuna study, found {len(source_studies)}")
        source_study = optuna.load_study(study_name=source_studies[0].study_name, storage=source_storage)
        source_trials = source_study.get_trials(deepcopy=True)
        if any(trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None for trial in source_trials):
            raise ValueError("Warm-start study contains incomplete or failed trials")
        for source_trial in source_trials:
            study.add_trial(source_trial)
        imported_completed = len(source_trials)
        source_summary = json.loads((args.warm_start_study.parent / "summary.json").read_text(encoding="utf-8"))
        if args.trials <= imported_completed:
            raise ValueError(f"--trials must exceed the {imported_completed} imported trials")
    (args.output / "campaign.json").write_text(json.dumps({
        "variant": args.variant, "objective_definition": VARIANTS[args.variant],
        "objective_note": {
            "worst_reserve": "Fatigue Lagrange term plus smooth-min terminal reserve penalty on each one-cycle RHO window; proxy for worst-muscle fatigue, not literal max integral.",
            "fatigue_balance": "Fatigue Lagrange term plus terminal normalized-fatigue imbalance penalty.",
            "fatigue_plus_activation": "Fatigue and normalized stimulation-charge (PW/PWmax) quadratic terms, equal default objective coefficients.",
        }[args.variant],
        "resistance_nm": args.resistance, "max_cycles": args.max_cycles,
        "workers": args.workers, "cpu_set": args.cpu_set, "time_limit_s": args.time_limit_s,
        "candidate_timeout_s": args.candidate_timeout_s,
        "final_validation_enabled": args.final_validate,
        "ding_local_reduction": args.ding_local,
        "startup_trials": args.startup_trials, "max_trials": args.trials,
        "warm_start_study": None if args.warm_start_study is None else str(args.warm_start_study),
        "imported_completed": imported_completed, "new_trials_requested": args.trials - imported_completed,
        "weight_order": MUSCLES, "common_seed": str(args.shared_seed), "seed_build_wall_time_s": seed_wall_s,
        "search_status": "screening_then_one_final_validation" if args.final_validate else "screening_only_no_final_certification",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    summary_path = args.output / "summary.json"
    summary: dict[str, Any] = {"status": "running", "variant": args.variant, "completed": 0,
                               "imported_completed": imported_completed, "completed_total": imported_completed,
                               "best": source_summary.get("best")}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    active: dict[Any, Any] = {}
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            while len(active) < args.workers and len(study.trials) < args.trials and time.monotonic() < args.deadline:
                trial = study.ask()
                weights = tuple(trial.suggest_float(f"weight_{name}", 0.05, 1.0) for name in MUSCLES)
                active[pool.submit(evaluate, trial.number, weights, args)] = trial
            if not active:
                break
            done, _ = wait(active, timeout=max(0.1, args.deadline-time.monotonic()) if time.monotonic() < args.deadline else None,
                           return_when=FIRST_COMPLETED)
            for future in done:
                trial = active.pop(future)
                try:
                    row = future.result(); rows.append(row); trial.set_user_attr("row", row)
                    if row["returncode"] == 0 and not row["timed_out"] and row["result_path"]:
                        trial.set_user_attr("record", row)
                        study.tell(trial, row["score_unvalidated"])
                    else:
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                except Exception as exc:
                    trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                good = [t for t in study.trials if t.value is not None]
                best = max(good, key=lambda t: t.value, default=None)
                summary.update(completed=len(rows), completed_total=imported_completed + len(rows),
                               best=None if best is None else best.user_attrs.get("record"),
                               elapsed_s=args.time_limit_s-max(0.0, args.deadline-time.monotonic()))
                summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    good = [trial for trial in study.trials if trial.value is not None]
    best_trial = max(good, key=lambda trial: trial.value, default=None)
    summary.update(status="screening_finished", completed=len(rows),
                   completed_total=imported_completed + len(rows),
                   best=None if best_trial is None else best_trial.user_attrs.get("record"),
                   elapsed_s=args.time_limit_s-max(0.0, args.deadline-time.monotonic()))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.final_validate and summary["best"] is not None:
        if args.warm_start_study is not None and summary["best"]["trial"] < imported_completed:
            previous_validation = source_summary.get("final_validation")
            summary["final_validation"] = {
                "status": "reused_previous", "source_summary": str(args.warm_start_study.parent / "summary.json"),
                "record": previous_validation,
            }
        else:
            summary["status"] = "final_validation_running"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            summary["final_validation"] = validate_best(summary["best"], args)
        summary["status"] = "complete"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
