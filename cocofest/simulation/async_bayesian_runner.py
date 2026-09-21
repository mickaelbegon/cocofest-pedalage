"""Persistent asynchronous Bayesian campaigns, one isolated process per simulation.

Run with ``python -m cocofest.simulation.async_bayesian_runner --campaign FILE
--prefix ENV``. ``n_trials`` is the total study budget, including previous runs.
Creating ``STOP`` in the campaign directory stops launches and requests process
termination. Completed trials resume; an interrupted simulation starts no checkpoint.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .async_bayesian_model import (
    AsyncBayesianCampaignConfig, SimResult, classify_result, muscle_weight_parameters,
)
from .async_bayesian_study import (
    finish_trial, open_study, recover_interrupted_trials, require_optuna,
    suggest_parameters, write_json, write_summary,
)
from .config import SimulationConfig
from .execution import runtime_helpers
from .launch import ROOT, build_launch_plan


@contextmanager
def campaign_lock(root):
    """Prevent two coordinators from launching the same study budget."""
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".coordinator.lock").open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Un coordinateur est déjà actif dans ce dossier.") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _signal_process(process, sig):
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        elif sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _selected_result(payload, solver):
    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if isinstance(results, list):
        return next((entry for entry in results if isinstance(entry, dict)
                     and str(entry.get("solver", "")).lower() == solver), None)
    return payload if "validated_cycles" in payload else None


def _administrative_evidence(result, requested_cycles):
    """Only a fresh explicitly certified partial result establishes a bound."""
    if not isinstance(result, dict):
        return False
    cycles = result.get("validated_cycles")
    return (type(cycles) is int and 0 <= cycles <= requested_cycles
            and (result.get("administrative_censored") is True
                 or result.get("termination_reason") in ("administrative_stop", "timeout", "stop_requested")))


def write_trial_weights_config(base_config, parameters, trial_root, number):
    """Materialize one immutable named-weight candidate beside its observation.

    The supplied baseline is copied so PACE policy settings remain part of the
    scientific input.  Only the complete, model-validated initial weight map
    is replaced; the candidate and baseline digest are both recorded locally.
    """
    multipliers = muscle_weight_parameters(parameters)
    if not multipliers:
        return None
    if not base_config.weights_config:
        raise ValueError("weights_config absent pour le candidat de poids musculaires")
    source = Path(base_config.weights_config).expanduser()
    if not source.is_absolute():
        source = ROOT / source
    try:
        raw = source.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"weights_config illisible : {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("weights_config doit être un objet JSON")
    basis = payload.get("initial_weight_basis")
    if not isinstance(basis, str) or not basis.strip():
        raise ValueError("weights_config doit documenter initial_weight_basis")
    baseline_weights = payload.get("initial_weights")
    if not isinstance(baseline_weights, dict) or set(baseline_weights) != set(multipliers):
        raise ValueError("weights_config doit contenir exactement les poids initiaux des muscles recherchés")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or value <= 0 for value in baseline_weights.values()):
        raise ValueError("weights_config contient un poids initial non positif ou non fini")
    # The BO domain is deliberately expressed as a dimensionless multiplier.
    # Retaining the baseline's calibrated relative scale matters especially
    # when muscle weights differ by several orders of magnitude.
    effective_weights = {
        name: float(baseline_weights[name]) * multiplier
        for name, multiplier in multipliers.items()
    }
    payload["initial_weights"] = effective_weights
    payload["initial_weight_basis"] = (
        f"{basis}; bayesian_initial_muscle_weight_multipliers_v1; trial={number}; "
        f"baseline_sha256={sha256(raw).hexdigest()}"
    )
    # ``run_rho_pace_benchmark`` intentionally accepts only policy, weights,
    # basis and calibration at the top level.  Keep provenance inside the
    # permitted calibration evidence instead of creating a rejected field.
    calibration = dict(payload.get("calibration") or {})
    calibration["bayesian_optimization"] = {
        "schema": "initial_muscle_weight_multipliers_v1", "trial": int(number),
        "source_weights_config": str(source), "source_weights_sha256": sha256(raw).hexdigest(),
        "parameters": {f"muscle_weight__{name}": value for name, value in multipliers.items()},
        "multipliers": multipliers,
        "baseline_initial_weights": {name: float(value) for name, value in baseline_weights.items()},
        "effective_initial_weights": effective_weights,
    }
    payload["calibration"] = calibration
    destination = trial_root / "weights-config.json"
    write_json(destination, payload)
    return str(destination)


def _cycle1_seed_evidence(result):
    """Return whether an IPOPT result is a usable exact one-cycle seed.

    The benchmark writer already performs the detailed feasibility audit.  The
    coordinator must nevertheless reject a missing, partial, or unsuccessful
    producer result before asking ACADOS to consume its archive.
    """
    return (isinstance(result, dict)
            and result.get("success") is True
            and result.get("validated_cycles") == 1)


def _run_plan(plan, environment, *, root, coordinator_pid, campaign, started, log_path):
    """Run one child command and honour the campaign stop protocol."""
    process, stopped_at, error, returncode = None, None, None, None
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(plan.argv, cwd=plan.cwd, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
        try:
            while process.poll() is None:
                now = time.monotonic()
                timeout = campaign.timeout_s is not None and now - started >= campaign.timeout_s
                orphaned = os.getppid() != coordinator_pid
                if stopped_at is None and (timeout or orphaned or (root / "STOP").exists()):
                    stopped_at = now
                    error = "trial_timeout" if timeout else "coordinator_lost" if orphaned else "campaign_stop"
                    _signal_process(process, signal.SIGTERM)
                if stopped_at is not None and now - stopped_at >= campaign.stop_grace_s:
                    _signal_process(process, signal.SIGKILL)
                time.sleep(.2)
            returncode = process.wait()
        finally:
            if process.poll() is None:
                _signal_process(process, signal.SIGKILL)
                process.wait()
    return returncode, error, stopped_at


def evaluate_trial(campaign_data, parameters, number, prefix, coordinator_pid):
    """Worker entry point; optional scientific dependencies live in the child command."""
    campaign = AsyncBayesianCampaignConfig.from_dict(campaign_data)
    root = Path(campaign.output_root).expanduser().resolve()
    trial_root = root / "trials" / f"trial_{number:06d}"
    trial_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result, error, returncode = None, None, None
    stopped_at = None
    config = None
    try:
        base = SimulationConfig.from_dict(campaign.base_config)
        trial_extras = tuple(base.extra_arguments)
        if base.solver == "acados":
            # ACADOS defaults to a stable generated-code name beneath the
            # example tree. Parallel trials would then load/remove the same
            # shared object. The tag keeps code generation and loading fully
            # trial-local while retaining all scientific options.
            trial_extras += ("--codegen-tag", f"bo-trial-{number:06d}")
        candidate_weights = write_trial_weights_config(base, parameters, trial_root, number)
        simulation_parameters = {
            name: value for name, value in parameters.items() if not name.startswith("muscle_weight__")
        }
        config = replace(base, **simulation_parameters, cycles=campaign.max_cycles,
                         output_root=str(trial_root / "simulation"), numeric_threads=1,
                         extra_arguments=trial_extras,
                         weights_config=candidate_weights or base.weights_config)
        write_json(trial_root / "config.json", config.to_dict())
        write_json(trial_root / "parameters.json", parameters)
        # Honour the explicit environment; only MadNLP uses its separate suite.
        trial_prefix = Path(prefix)
        if config.solver == "madnlp":
            trial_prefix = runtime_helpers().conda_env_prefix("madnlp32") or trial_prefix
        if not (trial_prefix / "bin/python").is_file():
            raise FileNotFoundError(f"Python du solveur absent : {trial_prefix}")
        # Every ACADOS candidate starts from a freshly audited IPOPT/MA57
        # solution of *its own* cycle-1 OCP.  Reusing a seed across controller
        # weights silently changes the initial problem and invalidates the
        # comparison.  The high budget is deliberately separate from the
        # target ACADOS budget and is recorded in seed-launch.json.
        seed_path = trial_root / "seed" / "ipopt-ma57-cycle1.npz"
        seed_entry = None
        if config.solver == "acados":
            seed_extras = tuple(arg for arg in config.extra_arguments
                                if arg != "--common-initial-solution"
                                and not arg.startswith("--common-initial-solution="))
            seed_config = replace(
                config, solver="ipopt", cycles=1, acados_ipopt_cycle1_seed=None,
                ipopt_linear_solver="ma57", output_root=str(trial_root / "seed" / "simulation"),
                extra_arguments=seed_extras + (
                    "--ipopt-max-iter", "10000",
                    "--common-initial-solution-output", str(seed_path),
                ),
            )
            write_json(trial_root / "seed-config.json", seed_config.to_dict())
            seed_plan = build_launch_plan(seed_config, trial_prefix, ROOT)
            seed_plan.cwd.mkdir(parents=True, exist_ok=True)
            seed_environment = runtime_helpers().base_environment(
                trial_prefix, seed_plan.suite, seed_config.threads, 1)
            seed_environment.update(seed_plan.environment_updates)
            write_json(trial_root / "seed-launch.json", {
                "argv": seed_plan.argv, "cwd": str(seed_plan.cwd), "result_json": str(seed_plan.result_json),
                "seed_path": str(seed_path), "ipopt_max_iter": 10000,
            })
            seed_returncode, seed_error, stopped_at = _run_plan(
                seed_plan, seed_environment, root=root, coordinator_pid=coordinator_pid,
                campaign=campaign, started=started, log_path=trial_root / "seed-runner.log")
            seed_result = None
            if seed_plan.result_json is not None and seed_plan.result_json.is_file():
                seed_result = _selected_result(json.loads(seed_plan.result_json.read_text(encoding="utf-8")), "ipopt")
            seed_entry = {"returncode": seed_returncode, "error": seed_error, "result": seed_result,
                          "path": str(seed_path)}
            write_json(trial_root / "seed-observation.json", seed_entry)
            if seed_error is not None:
                error = f"cycle1_seed_{seed_error}"
            elif seed_returncode != 0:
                error = f"cycle1_seed_returncode_{seed_returncode}"
            elif not _cycle1_seed_evidence(seed_result):
                error = "cycle1_seed_not_certified"
            elif not seed_path.is_file():
                error = "cycle1_seed_archive_missing"
            else:
                config = replace(config, acados_ipopt_cycle1_seed=str(seed_path))
                write_json(trial_root / "target-config.json", config.to_dict())
        if error is None:
            plan = build_launch_plan(config, trial_prefix, ROOT)
            plan.cwd.mkdir(parents=True, exist_ok=True)
            if plan.result_json is not None and plan.result_json.exists():
                raise FileExistsError(f"Résultat préexistant : {plan.result_json}")
            environment = runtime_helpers().base_environment(trial_prefix, plan.suite, config.threads, 1)
            environment.update(plan.environment_updates)
            runtime_path = root / "runtime.json"
            runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else None
            write_json(trial_root / "launch.json", {"argv": plan.argv, "cwd": str(plan.cwd),
                       "prefix": str(trial_prefix), "environment_updates": plan.environment_updates,
                       "result_json": str(plan.result_json), "worker_pid": os.getpid(), "runtime": runtime})
            returncode, target_error, stopped_at = _run_plan(
                plan, environment, root=root, coordinator_pid=coordinator_pid, campaign=campaign,
                started=started, log_path=trial_root / "runner.log")
            if target_error is not None:
                error = target_error
            if plan.result_json is not None and plan.result_json.is_file():
                result = _selected_result(json.loads(plan.result_json.read_text(encoding="utf-8")), config.solver)
            if result is None and error is None:
                error = "missing_or_invalid_solver_result"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    observation = classify_result(result, campaign.max_cycles, campaign.metric, error=error,
                                  returncode=returncode,
                                  administrative=stopped_at is not None and
                                  _administrative_evidence(result, campaign.max_cycles))
    entry = {"trial": number, "parameters": parameters, "sim_result": observation.to_dict(),
             "elapsed_s": time.monotonic() - started, "returncode": returncode,
             "error": error, "result": result}
    write_json(trial_root / "observation.json", entry)
    return entry


def run_campaign(campaign, prefix, *, evaluator=evaluate_trial):
    campaign.validate()
    optuna = require_optuna()
    root = Path(campaign.output_root).expanduser().resolve()
    campaign = replace(campaign, output_root=str(root))
    prefix = str(Path(prefix).expanduser().resolve())
    with campaign_lock(root):
        study = open_study(campaign)
        recover_interrupted_trials(study, root)
        try:
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
            write_json(root / "runtime.json", {"git_revision": git.stdout.strip(), "git_dirty": bool(dirty.stdout),
                       "coordinator_python": sys.executable, "solver_prefix": prefix, "pid": os.getpid()})
        except OSError:
            pass
        stopping = False
        previous_handlers = {}

        def request_stop(signum, frame):
            nonlocal stopping
            stopping = True
            (root / "STOP").touch(exist_ok=True)

        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, request_stop)
        try:
            with ProcessPoolExecutor(max_workers=campaign.workers,
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                active = {}
                while True:
                    stopping = stopping or (root / "STOP").exists()
                    started_count = sum(t.state != optuna.trial.TrialState.WAITING
                                        for t in study.get_trials(deepcopy=False))
                    while not stopping and len(active) < campaign.workers and started_count < campaign.n_trials:
                        trial = study.ask()
                        try:
                            parameters = suggest_parameters(trial, campaign.search_space)
                            trial.set_user_attr("trial_directory", str(root / "trials" / f"trial_{trial.number:06d}"))
                            future = pool.submit(evaluator, campaign.to_dict(), parameters,
                                                 trial.number, prefix, os.getpid())
                        except Exception as exc:
                            finish_trial(study, trial, SimResult("technical_failure", reason=str(exc)))
                            raise
                        active[future] = trial
                        started_count += 1
                        print(f"[BO async] lancement essai {trial.number}: {parameters}", flush=True)
                    write_summary(study, root)
                    if not active:
                        break
                    done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        trial = active.pop(future)
                        try:
                            entry = future.result()
                            result = SimResult(**entry["sim_result"])
                        except Exception as exc:
                            result = SimResult("technical_failure", reason=f"worker_error: {type(exc).__name__}: {exc}")
                            trial_root = root / "trials" / f"trial_{trial.number:06d}"
                            trial_root.mkdir(parents=True, exist_ok=True)
                            write_json(trial_root / "observation.json", {"trial": trial.number, "sim_result": result.to_dict()})
                        finish_trial(study, trial, result)
                        print(f"[BO async] essai {trial.number}: {result.status}, score={result.score}, borne={result.lower_bound}", flush=True)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            summary = write_summary(study, root)
        # Valid censoring is a completed scientific evaluation, not a runner error.
        usable = sum(summary["counts"].get(status, 0) for status in ("observed", "horizon_censored"))
        return 130 if stopping else 0 if usable else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--prefix", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = AsyncBayesianCampaignConfig.from_json(args.campaign.read_text(encoding="utf-8"))
    return run_campaign(campaign, args.prefix)


if __name__ == "__main__":
    raise SystemExit(main())
