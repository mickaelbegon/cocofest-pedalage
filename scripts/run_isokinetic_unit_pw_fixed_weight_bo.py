#!/usr/bin/env python3
"""Parallel BO of fixed per-muscle PW-squared weights for prescribed RHO.

Candidates use a single shared, one-cycle RHO warm-start and deferred summaries.
The BO scores are explicitly provisional; only the final selected weight vector
gets the complete post-solve audit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import optuna


MUSCLES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")


def provisional_score(result: Path) -> tuple[int, float, float | None, dict[str, Any]]:
    """Rank unvalidated screens by solver-covered cycles, then terminal reserve."""
    payload = json.loads(result.read_text(encoding="utf-8"))["results"][0]
    cycles = int(payload.get("bo_screen_cycles_unvalidated", payload.get("covered_cycles")) or 0)
    reserve = payload.get("bo_screen_terminal_min_capacity_ratio_unvalidated")
    try:
        reserve = float(reserve)
    except (TypeError, ValueError):
        reserve = None
    if reserve is not None and not np.isfinite(reserve):
        reserve = None
    bonus = 0.999 * float(np.clip(reserve, 0.0, 1.0)) if reserve is not None else 0.0
    return cycles, float(cycles + bonus), reserve, payload


def relative_weights(raw: tuple[float, ...]) -> tuple[float, ...]:
    """Remove the irrelevant common multiplier; retain strictly positive ratios."""
    values = np.asarray(raw, dtype=float)
    return tuple((values / np.exp(np.mean(np.log(values)))).tolist())


def _rho_command(args: argparse.Namespace, weights: tuple[float, ...], result: Path,
                 max_cycles: int, *, shared_seed: Path | None = None,
                 seed_output: Path | None = None) -> list[str]:
    command = [
        args.python, "scripts/benchmark_ding_radau5_local_ab.py", "local", "--",
        "--solvers", "ipopt", "--objective", "control", "--objective-shape", "quadratic",
        "--parametric-control-weights", "--control-weight-values", *(f"{value:.12g}" for value in weights),
        "--formulation", "isokinetic", "--mechanical-formulation", "reduced",
        "--isokinetic-kinematics", "prescribed", "--isokinetic-omega", str(-2.0 * np.pi),
        "--energy-equivalent-torque", str(args.resistance), "--load-torque-min", "-4", "--load-torque-max", "4",
        "--stimulations-per-cycle", "30", "--ipopt-profile", "periodic_collocation",
        "--ipopt-collocation-method", "radau", "--ipopt-collocation-degree", "5", "--ipopt-use-sx",
        "--ipopt-linear-solver", "ma57", "--warmup-ipopt-linear-solver", "ma57",
        "--ipopt-hsl-library", args.hsl_library, "--ipopt-max-iter", "4000",
        "--nlp-tolerance", "1e-8", "--primal-feasibility-threshold", "1e-5",
        "--ipopt-enforce-start-constraints", "--cycles-per-window", "1", "--n-windows", str(max_cycles),
        "--n-threads", "1", "--compact-rho-output", "--retry-failed-rho-without-advance",
        "--output-json", str(result),
    ]
    if shared_seed is not None:
        command += ["--common-initial-solution", str(shared_seed)]
    if seed_output is not None:
        command += ["--common-initial-solution-output", str(seed_output)]
    return command


def _base_environment(args: argparse.Namespace, *, deferred: bool) -> dict[str, str]:
    environment = {**os.environ, "PYTHONPATH": str(args.root), "OMP_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "MPLCONFIGDIR": "/tmp/cocofest-bo-mpl"}
    if deferred:
        environment.update(COCOFEST_DEFERRED_RHO_SUMMARY="1", DING_ONLINE_JOURNAL="1")
    return environment


def _prepare_shared_rho_seed(args: argparse.Namespace) -> dict[str, Any]:
    if args.prior_run is not None:
        seed = args.prior_run / "shared-rho-seed" / "one-cycle-unit-weight.npz"
        if not seed.is_file():
            raise FileNotFoundError(f"Prior campaign shared RHO seed is missing: {seed}")
        args.shared_seed = seed
        return {
            "path": str(seed),
            "weights": dict(zip(MUSCLES, (1.0, 1.0, 1.0, 1.0), strict=True)),
            "covered_cycles": 1,
            "method": "reuse the single unit-weight RHO seed from the prior campaign",
        }
    seed_dir = args.output / "shared-rho-seed"
    seed_dir.mkdir(parents=True, exist_ok=False)
    result = seed_dir / "result.json"
    seed = seed_dir / "one-cycle-unit-weight.npz"
    command = _rho_command(args, (1.0, 1.0, 1.0, 1.0), result, 1, seed_output=seed)
    remaining_s = max(0.0, args.deadline_monotonic - time.monotonic())
    if remaining_s <= 0:
        raise TimeoutError("The BO search budget expired before the shared RHO seed was built")
    start = time.monotonic()
    completed = subprocess.run(["taskset", "-c", args.cpu_set, *command], cwd=args.root, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=_base_environment(args, deferred=False), timeout=remaining_s)
    elapsed = time.monotonic() - start
    (seed_dir / "driver.log").write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0 or not seed.is_file() or not result.is_file():
        raise RuntimeError(f"Shared one-cycle RHO seed failed (returncode={completed.returncode}); see {seed_dir}")
    payload = json.loads(result.read_text(encoding="utf-8"))["results"][0]
    if int(payload.get("covered_cycles") or 0) < 1 or payload.get("success") is False:
        raise RuntimeError("Shared one-cycle RHO seed did not produce a successful cycle")
    args.shared_seed = seed
    return {"path": str(seed), "weights": dict(zip(MUSCLES, (1.0, 1.0, 1.0, 1.0), strict=True)),
            "wall_time_s": elapsed, "covered_cycles": int(payload.get("covered_cycles") or 0),
            "method": "one-cycle unit-weight RHO, reused as the identical initial guess for all BO candidates"}


def run_trial(number: int, weights: tuple[float, ...], args: argparse.Namespace) -> dict[str, Any]:
    directory = args.output / "trials" / f"trial_{number:03d}"
    directory.mkdir(parents=True, exist_ok=False)
    result = directory / "result.json"
    command = _rho_command(args, weights, result, args.max_cycles, shared_seed=args.shared_seed)
    environment = _base_environment(args, deferred=True)
    trial_started = time.monotonic()
    remaining_s = max(0.0, args.deadline_monotonic - time.monotonic())
    timed_out = remaining_s <= 0
    output_text = ""
    returncode = 124 if timed_out else None
    if not timed_out:
        try:
            completed = subprocess.run(["taskset", "-c", args.cpu_set, *command], cwd=args.root, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment,
                                       timeout=remaining_s)
            output_text = completed.stdout or ""
            returncode = completed.returncode
        except subprocess.TimeoutExpired as error:
            timed_out = True
            returncode = 124
            captured = error.stdout or ""
            output_text = captured.decode(errors="replace") if isinstance(captured, bytes) else captured
            output_text += "\nBO_TRIAL_STOPPED_AT_GLOBAL_TIME_LIMIT\n"
    elapsed = time.monotonic() - trial_started
    (directory / "driver.log").write_text(output_text, encoding="utf-8")
    cycles, score, min_capacity, payload = provisional_score(result) if result.is_file() else (0, 0.0, None, {})
    timing = payload.get("execution_timing", {}) if isinstance(payload, dict) else {}
    samples = [float(value) for value in timing.get("rho_complete_iteration_wall_time_samples_s", [])
               if isinstance(value, (int, float)) and np.isfinite(value) and value >= 0]
    cycles_per_cycle_wall = elapsed / cycles if cycles else None
    record = {"trial": number, "relative_pw2_weights": dict(zip(MUSCLES, weights, strict=True)),
              "returncode": returncode, "timed_out": timed_out,
              "screen_cycles_unvalidated": cycles, "screen_terminal_min_capacity_ratio_unvalidated": min_capacity,
              "bo_score_unvalidated": score, "screen_result_available": result.is_file(),
              "score_basis": "provisional_solver_covered_cycles_plus_terminal_reserve",
              "trial_wall_time_s": elapsed, "end_to_end_wall_time_per_screen_cycle_s": cycles_per_cycle_wall,
              "rho_cycle_wall_time_sample_count": len(samples),
              "rho_cycle_wall_time_mean_s": float(np.mean(samples)) if samples else None,
              "rho_cycle_wall_time_median_s": float(np.median(samples)) if samples else None,
              "rho_cycle_wall_time_p90_s": float(np.percentile(samples, 90)) if samples else None,
              "rho_solver_loop_wall_time_s": timing.get("rho_solve_loop_wall_time_s"),
              "result_path": str(result)}
    (directory / "observation.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _write_timing_csv(output: Path, records: list[dict[str, Any]]) -> None:
    path = output / "trial-timings.csv"
    fields = ("trial", "screen_cycles_unvalidated", "trial_wall_time_s",
              "end_to_end_wall_time_per_screen_cycle_s", "rho_cycle_wall_time_mean_s",
              "rho_cycle_wall_time_median_s", "rho_cycle_wall_time_p90_s", "timed_out")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in records)


def _run_final_validation(best_record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Run exactly one full-summary validation, after the BO screen is over."""
    directory = args.output / "final-validation"
    directory.mkdir(parents=True, exist_ok=False)
    result = directory / "result.json"
    weights = tuple(float(best_record["relative_pw2_weights"][name]) for name in MUSCLES)
    command = _rho_command(args, weights, result, args.max_cycles, shared_seed=args.shared_seed)
    started = time.monotonic()
    with (directory / "driver.log").open("w", encoding="utf-8") as log:
        validation_environment = _base_environment(args, deferred=False)
        # Keep the exact per-cycle solver and mechanical/isokinetic audits, but
        # skip rebuilding/exporting all state/control traces for the final report.
        validation_environment["COCOFEST_COMPACT_FINAL_SUMMARY"] = "1"
        completed = subprocess.run(["taskset", "-c", args.cpu_set, *command], cwd=args.root,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   env=validation_environment)
    elapsed = time.monotonic() - started
    payload = None
    if result.is_file():
        try:
            payload = json.loads(result.read_text(encoding="utf-8"))["results"][0]
        except (ValueError, KeyError, TypeError):
            payload = None
    validation = {
        "status": "completed" if completed.returncode == 0 and payload is not None else "failed",
        "returncode": completed.returncode,
        "wall_time_s": elapsed,
        "weight_source_trial": best_record["trial"],
        "relative_pw2_weights": dict(zip(MUSCLES, weights, strict=True)),
        "max_cycles": args.max_cycles,
        "result_path": str(result),
        "covered_cycles": None if payload is None else payload.get("covered_cycles"),
        "validated_cycles": None if payload is None else payload.get("validated_cycles"),
        "physiologically_validated_cycles": None if payload is None else payload.get("physiologically_validated_cycles"),
        "success": None if payload is None else payload.get("success"),
        "solver_success": None if payload is None else payload.get("solver_success"),
        "physical_success": None if payload is None else payload.get("physical_success"),
        "execution_timing": None if payload is None else payload.get("execution_timing"),
    }
    (directory / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n",
                                                encoding="utf-8")
    return validation


def _reuse_prior_validation(best_record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    """Reuse an exact matching incumbent's completed full-run validation."""
    if args.prior_run is None:
        return None
    path = args.prior_run / "summary.json"
    try:
        prior = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    validation = prior.get("final_validation") or {}
    if prior.get("status") != "complete" or validation.get("status") != "completed":
        return None
    previous_weights = validation.get("relative_pw2_weights") or {}
    current_weights = best_record.get("relative_pw2_weights") or {}
    if any(
        name not in previous_weights
        or not np.isclose(
            float(current_weights.get(name, float("nan"))),
            float(previous_weights[name]), rtol=0, atol=1e-12
        )
        for name in MUSCLES
    ):
        return None
    return {
        **validation,
        "status": "reused_prior_validation",
        "reused_from": str(args.prior_run / "final-validation" / "result.json"),
        "reused_validation_success": validation.get("success"),
        "note": "Exact same weight vector and model; no duplicate validation rerun.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", required=True); parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hsl-library", required=True); parser.add_argument("--resistance", type=float, default=1.3)
    parser.add_argument("--max-cycles", type=int, default=1500); parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--startup-trials", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8); parser.add_argument("--cpu-set", default="13-20")
    parser.add_argument("--time-limit-s", type=float, default=600.0,
                        help="Hard wall-clock limit for the whole BO campaign, including active evaluations")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--prior-run", type=Path, default=None,
                        help="Import a completed campaign's Optuna observations and reuse its shared RHO seed")
    args = parser.parse_args(); args.output = args.output.resolve(); args.root = args.root.resolve()
    if args.time_limit_s <= 0:
        raise ValueError("--time-limit-s must be positive")
    # A systemd launcher may create only ``driver.log`` before this process
    # starts.  Preserve the no-overwrite contract for any actual BO artifact.
    if args.output.exists():
        existing = [path for path in args.output.iterdir() if path.name != "driver.log"]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing BO output: {args.output}")
    else:
        args.output.mkdir(parents=True)
    start_monotonic = time.monotonic()
    start_utc = datetime.now(timezone.utc)
    args.deadline_monotonic = start_monotonic + args.time_limit_s
    deadline_utc = start_utc + timedelta(seconds=args.time_limit_s)
    study_name = f"rho-bo-pw-isokinetic-unit-{args.resistance:.2f}nm-{args.output.name}"
    study = optuna.create_study(direction="maximize", study_name=study_name,
                                sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.startup_trials),
                                storage=f"sqlite:///{args.output / 'study.sqlite3'}")
    imported_trial_count = 0
    if args.prior_run is not None:
        prior_db = args.prior_run / "study.sqlite3"
        if not prior_db.is_file():
            raise FileNotFoundError(f"Prior campaign Optuna database is missing: {prior_db}")
        prior_study = optuna.load_study(
            study_name=f"rho-bo-pw-isokinetic-unit-{args.resistance:.2f}nm",
            storage=f"sqlite:///{prior_db}",
        )
        prior_trials = prior_study.get_trials(deepcopy=False)
        study.add_trials(prior_trials)
        imported_trial_count = len(prior_trials)
    initial_trial_count = len(study.trials)
    target_study_trial_count = initial_trial_count + args.trials
    (args.output / "campaign.json").write_text(json.dumps({
        "objective": "screen for maximum RHO endurance under fixed per-muscle PW² weights; final candidate fully audited",
        "score_rule": "UNVALIDATED screen_cycles + 0.999*clip(terminal_min_A_capacity_ratio,0,1)",
        "weight_normalization": "geometric_mean_one",
        "weight_order": MUSCLES,
        "resistance_nm": args.resistance,
        "trials": args.trials,
        "new_trials_requested": args.trials,
        "imported_prior_trial_count": imported_trial_count,
        "prior_campaign": None if args.prior_run is None else str(args.prior_run),
        "startup_random_trials": args.startup_trials,
        "workers": args.workers,
        "cpu_affinity": args.cpu_set,
        "time_limit_s": args.time_limit_s,
        "started_at_utc": start_utc.isoformat(),
        "deadline_utc": deadline_utc.isoformat(),
        "weight_search": (
            "TPE refinement initialized with imported prior study observations"
            if imported_trial_count else "cold_start_log_uniform; no prior BO observations or patient-specific weights"
        ),
        "isokinetic_kinematics": "prescribed",
        "ding_state_reduction": "exact_discrete_radau5_local; reconstruct Cn,Tau1,Km; retain F,A; full audit deferred to final candidate",
        "candidate_policy": "COCOFEST_DEFERRED_RHO_SUMMARY; no full retrospective audits for candidate trials",
        "final_policy": "one full validation of the best completed candidate after search",
        "shared_seed_policy": "one unit-weight one-cycle RHO seed, built once and reused for every candidate and final validation",
    }, indent=2) + "\n", encoding="utf-8")
    summary_path = args.output / "summary.json"
    summary = {"status": "preparing_shared_rho_seed", "time_limit_s": args.time_limit_s,
               "started_at_utc": start_utc.isoformat(), "deadline_utc": deadline_utc.isoformat(),
               "max_cycles_per_candidate": args.max_cycles, "startup_random_trials": args.startup_trials,
               "imported_prior_trial_count": imported_trial_count,
               "new_trials_requested": args.trials,
               "new_trials_completed": 0,
               "completed": 0, "best_trial": None, "best_screen_cycles_unvalidated": None,
               "best_score_unvalidated": None}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    try:
        shared_seed_record = _prepare_shared_rho_seed(args)
    except Exception as error:
        summary.update(status="shared_seed_failed", error=f"{type(error).__name__}: {error}")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        raise
    campaign = json.loads((args.output / "campaign.json").read_text(encoding="utf-8"))
    campaign["shared_rho_seed"] = shared_seed_record
    campaign["sampler"] = {"name": "Optuna TPESampler", "startup_random_trials": args.startup_trials,
                           "update_policy": "tell after each completed candidate, then ask the next candidate"}
    (args.output / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n", encoding="utf-8")
    summary["shared_rho_seed"] = shared_seed_record
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    active: dict[Any, Any] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            while (len(active) < args.workers and len(study.trials) < target_study_trial_count
                   and time.monotonic() < args.deadline_monotonic):
                trial = study.ask()
                weights = relative_weights(tuple(trial.suggest_float(name, 0.05, 4.0, log=True) for name in MUSCLES))
                active[pool.submit(run_trial, trial.number, weights, args)] = trial
            if not active:
                break
            remaining_s = max(0.0, args.deadline_monotonic - time.monotonic())
            done, _ = wait(active, timeout=remaining_s if remaining_s > 0 else None,
                           return_when=FIRST_COMPLETED)
            if not done:
                # Workers have the same hard deadline and terminate their OCP
                # subprocesses there; collect their timeout records as they exit.
                continue
            for future in done:
                trial = active.pop(future)
                try:
                    record = future.result(); trial.set_user_attr("record", record)
                    if (record["timed_out"] or record["returncode"] != 0
                            or not record["screen_result_available"]):
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    else:
                        study.tell(trial, float(record["bo_score_unvalidated"]))
                except Exception as error:
                    trial.set_user_attr("error", f"{type(error).__name__}: {error}"); study.tell(trial, state=optuna.trial.TrialState.FAIL)
                completed = [item for item in study.trials if item.value is not None]
                new_completed = [item for item in completed if item.number >= initial_trial_count]
                best = max(completed, key=lambda item: item.value, default=None)
                summary.update(status="candidate_search_running", elapsed_s=time.monotonic() - start_monotonic,
                               completed=len(new_completed), new_trials_completed=len(new_completed),
                               best_trial=None if best is None else best.number,
                               best_screen_cycles_unvalidated=None if best is None else best.user_attrs.get("record", {}).get("screen_cycles_unvalidated"),
                               best_score_unvalidated=None if best is None else best.value,
                               best_raw_parameters=None if best is None else best.params,
                               best_record=None if best is None else best.user_attrs.get("record"))
                summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                records = [item.user_attrs.get("record") for item in study.trials
                           if item.user_attrs.get("record") is not None]
                _write_timing_csv(args.output, records)
    search_elapsed = time.monotonic() - start_monotonic
    new_completed_trials = [item for item in study.trials
                            if item.number >= initial_trial_count and item.value is not None]
    summary.update(status="candidate_search_time_limit_reached" if time.monotonic() >= args.deadline_monotonic else "candidate_search_completed",
                   elapsed_s=time.monotonic() - start_monotonic,
                   completed=len(new_completed_trials),
                   new_trials_completed=len(new_completed_trials),
                   imported_prior_trial_count=imported_trial_count,
                   new_trials_requested=args.trials,
                   total_trials=len(study.trials))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    completed_trials = [item for item in study.trials if item.value is not None]
    best = max(completed_trials, key=lambda item: item.value, default=None)
    if best is None:
        summary.update(status="no_completed_candidate", final_validation=None)
    else:
        best_record = best.user_attrs.get("record")
        summary.update(status="final_validation_running", best_record=best_record,
                       candidate_search_elapsed_s=search_elapsed)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        validation = _reuse_prior_validation(best_record, args)
        if validation is None:
            validation = _run_final_validation(best_record, args)
        summary.update(status="complete", final_validation=validation,
                       total_elapsed_s=time.monotonic() - start_monotonic)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
