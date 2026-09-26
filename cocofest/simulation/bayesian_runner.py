"""Child process implementing a restart-safe Bayesian solver campaign."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

from .bayesian_model import BayesianCampaignConfig
from .config import SimulationConfig
from .execution import runtime_helpers
from .launch import ROOT, build_launch_plan


FAILURE_PENALTY = 1_000_000.0


def _metric(result: dict, requested_cycles: int, metric: str) -> float:
    if metric == "continuous_endurance":
        estimate = result.get("continuous_endurance") or {}
        value = estimate.get("continuous_cycle_equivalent")
        if estimate.get("available") is not True or not isinstance(value, (int, float)) or isinstance(value, bool):
            return FAILURE_PENALTY
        # Optimizer minimizes; a higher continuous endurance is better.
        return -float(value)
    if result.get("success") is not True or result.get("validated_cycles") != requested_cycles:
        return FAILURE_PENALTY
    keys = {"hot_time": "hot_solver_time_median_s", "solver_time": "solver_time_per_cycle_s", "p90_time": "hot_solver_time_p90_s"}
    value = result.get(keys[metric])
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        return FAILURE_PENALTY
    return float(value)


def run_campaign(campaign: BayesianCampaignConfig, prefix: Path) -> int:
    """Run GP ask/tell so each observation is written before the next proposal."""
    campaign = campaign.validate()
    try:
        from skopt import Optimizer
        from skopt.space import Categorical, Integer
    except ImportError as error:
        raise RuntimeError("scikit-optimize est requis dans l'environnement solveur (installez scikit-optimize).") from error
    root = Path(campaign.output_root).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=False)
    journal = root / "campaign.jsonl"
    (root / "campaign.json").write_text(campaign.to_json(), encoding="utf-8")
    base = SimulationConfig.from_dict(campaign.base_config)
    space = [Categorical(list(campaign.solvers), name="solver"),
             Integer(campaign.collocation_degree_min, campaign.collocation_degree_max, name="collocation_degree"),
             Categorical(list(campaign.thread_choices), name="threads")]
    optimizer = Optimizer(space, base_estimator="GP", acq_func="EI", n_initial_points=campaign.n_initial_points,
                          random_state=campaign.random_state)
    seen = set()
    best = None
    for index in range(campaign.n_calls):
        point = optimizer.ask()
        key = tuple(point)
        # Tiny discrete spaces can otherwise return a repeated expensive trial.
        if key in seen:
            alternatives = [(s, d, t) for s in campaign.solvers
                            for d in range(campaign.collocation_degree_min, campaign.collocation_degree_max + 1)
                            for t in campaign.thread_choices if (s, d, t) not in seen]
            if alternatives:
                point = list(alternatives[0])
                key = tuple(point)
        seen.add(key)
        solver, degree, threads = point
        trial_root = root / "trials" / f"{index:03d}-{solver}-d{degree}-t{threads}"
        config = replace(base, solver=solver, collocation_degree=int(degree), threads=int(threads), output_root=str(trial_root))
        started = time.monotonic()
        result = None
        error = None
        returncode = None
        try:
            requested_suite = "madnlp32" if solver == "madnlp" else "rho32"
            # IPOPT/Fatrop and MadNLP are commonly installed in distinct conda
            # environments. Prefer the suite's discovered prefix; the GUI
            # prefix remains a useful explicit fallback for custom installs.
            trial_prefix = runtime_helpers().conda_env_prefix(requested_suite) or prefix
            if not (trial_prefix / "bin/python").is_file():
                raise FileNotFoundError(f"Environnement {requested_suite} absent : {trial_prefix}")
            plan = build_launch_plan(config, trial_prefix, ROOT)
            environment = runtime_helpers().base_environment(trial_prefix, plan.suite, config.threads, config.numeric_threads)
            environment.update(plan.environment_updates)
            trial_root.mkdir(parents=True, exist_ok=False)
            plan.save_effective_configuration()
            with (trial_root / "runner.log").open("x", encoding="utf-8") as log:
                completed = subprocess.run(plan.argv, cwd=plan.cwd, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, text=True, check=False)
            returncode = completed.returncode
            payload = json.loads(plan.result_json.read_text(encoding="utf-8"))
            results = payload.get("results") if isinstance(payload, dict) else None
            if isinstance(results, list):
                result = next((item for item in results if isinstance(item, dict) and item.get("solver", "").lower() == solver), None)
            if result is None:
                error = "Résultat solver absent ou JSON invalide"
        except Exception as exc:  # recorded as an observation, never hides a failed configuration
            error = f"{type(exc).__name__}: {exc}"
        score = _metric(result, config.cycles, campaign.metric) if result else FAILURE_PENALTY
        optimizer.tell(point, score)
        entry = {"trial": index, "parameters": {"solver": solver, "collocation_degree": degree, "threads": threads},
                 "score": score, "certified": score < FAILURE_PENALTY, "returncode": returncode,
                 "elapsed_s": time.monotonic() - started, "result": result, "error": error,
                 "config": config.to_dict()}
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
        print(f"[BO] essai {index + 1}/{campaign.n_calls}: {solver}, d={degree}, t={threads} -> {score:g}", flush=True)
        if score < FAILURE_PENALTY and (best is None or score < best["score"]):
            best = entry
    (root / "best.json").write_text(json.dumps(best or {"error": "Aucun essai certifié"}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("[BO] campagne terminée. " + (f"meilleur score: {best['score']:g}" if best else "aucun essai certifié"), flush=True)
    return 0 if best else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--prefix", required=True, type=Path)
    args = parser.parse_args(argv)
    return run_campaign(BayesianCampaignConfig.from_json(args.campaign.read_text(encoding="utf-8")), args.prefix)


if __name__ == "__main__":
    raise SystemExit(main())
