#!/usr/bin/env python3
"""BO fixed-weight RHO study and cross-resistance transfer matrix.

Each study is unilateral, prescribed-isokinetic, Radau-5/SX and uses the
experimental *exact* Ding-local IPOPT adapter.  For every resistance, Optuna
maximises the number of certified RHO windows over fixed four-muscle weights.
The best vector from each study is then replayed at every resistance, yielding
a 5-by-5 transfer matrix of certified cycles.

The local adapter only removes Cn, Tau1 and Km after proving their discrete
affine relation to A.  It reconstructs and audits the original NLP before the
RHO controller accepts a window; it is therefore an acceleration, not a
different fatigue model or acceptance criterion.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import optuna


MUSCLES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")


def _open_study(source: Path) -> tuple[optuna.study.Study, Path]:
    """Load the sole study in a file/directory supplied by a campaign."""
    database = source / "study.sqlite3" if source.is_dir() else source
    if not database.is_file():
        raise FileNotFoundError(f"Missing BO study database: {database}")
    storage = f"sqlite:///{database}"
    summaries = optuna.study.get_all_study_summaries(storage=storage)
    if len(summaries) != 1:
        raise ValueError(
            f"Expected exactly one Optuna study in {database}, found {len(summaries)}."
        )
    return optuna.load_study(storage=storage, study_name=summaries[0].study_name), database


def _import_completed_trials(study: optuna.study.Study, source: Path) -> int:
    """Seed a new TPE study with evaluated weight/cycle observations.

    This is intentionally an *observation* transfer, not merely an enqueue of
    the incumbent point.  TPE therefore conditions its density models on all
    completed previous candidates and immediately refines around promising
    regions once the imported count exceeds ``n_startup_trials``.
    """
    prior, database = _open_study(source)
    distributions = {
        muscle: optuna.distributions.FloatDistribution(0.0, 1.0)
        for muscle in MUSCLES
    }
    imported = 0
    for trial in prior.trials:
        if (
            trial.state != optuna.trial.TrialState.COMPLETE
            or trial.value is None
            or set(trial.params) != set(MUSCLES)
        ):
            continue
        study.add_trial(
            optuna.trial.create_trial(
                params={muscle: float(trial.params[muscle]) for muscle in MUSCLES},
                distributions=distributions,
                value=float(trial.value),
                user_attrs={
                    "imported_from": str(database),
                    "source_trial_number": trial.number,
                },
            )
        )
        imported += 1
    if not imported:
        raise RuntimeError(f"No completed four-weight observations in {database}")
    return imported


def _enqueue_prior_incumbent(study: optuna.study.Study, source: Path) -> str:
    """Re-evaluate a nearby-resistance incumbent at this target resistance.

    Its score is deliberately *not* transferred: only its four coordinates
    are queued.  The next objective evaluation is performed at the current
    resistance, preserving a valid BO data set.
    """
    prior, database = _open_study(source)
    candidates = [
        trial for trial in prior.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and set(trial.params) == set(MUSCLES)
    ]
    if not candidates:
        raise RuntimeError(f"No completed four-weight incumbent in {database}")
    incumbent = max(candidates, key=lambda trial: float(trial.value))
    study.enqueue_trial({muscle: float(incumbent.params[muscle]) for muscle in MUSCLES})
    return str(database)


def _read_result(path: Path) -> tuple[int, bool | None, str | None, float]:
    if not path.is_file():
        return 0, None, "missing result.json", 0.0
    data = json.loads(path.read_text(encoding="utf-8"))
    result = data["results"][0]
    reserve = result.get("terminal_capacity_reserve")
    reserve_value = 0.0
    if isinstance(reserve, dict) and reserve.get("available") is True:
        candidate = reserve.get("minimum_ratio")
        if isinstance(candidate, (float, int)) and 0.0 <= float(candidate) <= 1.0:
            reserve_value = float(candidate)
    return int(result.get("covered_cycles") or 0), result.get("success"), result.get("error"), reserve_value


def _bo_score(record: dict[str, Any], score_mode: str) -> float:
    """Rank certified endurance first and terminal capacity only within a tie.

    The reserve is a ratio in ``[0, 1]`` and the multiplier is strictly below
    one.  Consequently no reserve can make an N-cycle rollout outrank an
    (N+1)-cycle rollout; it merely gives TPE a continuous signal among equal
    integer endurance outcomes.
    """
    cycles = float(record["certified_cycles"])
    if score_mode == "certified_cycles":
        return cycles
    if score_mode == "certified_cycles_then_reserve":
        return cycles + 1e-3 * float(record["terminal_capacity_reserve"])
    raise ValueError(f"Unknown BO score mode: {score_mode}")


def _rho_command(args: argparse.Namespace, resistance: float, weights: tuple[float, ...], result: Path) -> list[str]:
    return [
        args.python, "scripts/benchmark_ding_radau5_local_ab.py", "local", "--",
        "--solvers", "ipopt", "--objective", "fatigue", "--objective-shape", "quadratic",
        "--formulation", "isokinetic", "--mechanical-formulation", "reduced",
        "--isokinetic-kinematics", "prescribed", "--isokinetic-omega", str(-2.0 * 3.141592653589793),
        "--energy-equivalent-torque", str(resistance), "--load-torque-min", "-4", "--load-torque-max", "4",
        "--stimulations-per-cycle", "30", "--ipopt-profile", "periodic_collocation",
        "--ipopt-collocation-method", "radau", "--ipopt-collocation-degree", "5", "--ipopt-use-sx",
        "--ipopt-linear-solver", "ma57", "--warmup-ipopt-linear-solver", "ma57",
        "--ipopt-hsl-library", args.hsl_library, "--ipopt-max-iter", str(args.ipopt_max_iter),
        "--nlp-tolerance", "1e-8", "--primal-feasibility-threshold", "1e-5",
        "--ipopt-enforce-start-constraints", "--enforce-isokinetic-work-per-cycle",
        "--cycles-per-window", "1", "--n-windows", str(args.max_cycles), "--n-threads", "1",
        "--compact-rho-output", "--retry-failed-rho-without-advance",
        "--parametric-fatigue-weights", "--fatigue-weight-values", *(f"{value:.12g}" for value in weights),
        "--output-json", str(result),
    ]


def _environment(args: argparse.Namespace, mpl_dir: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": str(args.root),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MPLCONFIGDIR": str(mpl_dir),
    }


def _run_case(case_id: str, resistance: float, weights: tuple[float, ...], root: Path,
              args: argparse.Namespace) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    result = root / "result.json"
    command = _rho_command(args, resistance, weights, result)
    completed = subprocess.run(
        ["taskset", "-c", args.cpu_set, *command], cwd=args.root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=_environment(args, root / ".matplotlib"),
    )
    (root / "driver.log").write_text(completed.stdout, encoding="utf-8")
    cycles, success, error, terminal_reserve = _read_result(result)
    record = {
        "case": case_id,
        "resistance_nm": resistance,
        "weights": dict(zip(MUSCLES, weights, strict=True)),
        "returncode": completed.returncode,
        "certified_cycles": cycles,
        "solver_success": success,
        "error": error,
        "terminal_capacity_reserve": terminal_reserve,
        "result_path": str(result),
    }
    record["bo_score"] = _bo_score(record, getattr(args, "bo_score", "certified_cycles"))
    (root / "observation.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _run_bo(resistance: float, args: argparse.Namespace) -> dict[str, Any]:
    label = f"{resistance:.2f}".replace(".", "p")
    output = args.output / "bo" / label
    output.mkdir(parents=True, exist_ok=False)
    study = optuna.create_study(
        direction="maximize", study_name=f"rho-ding-local-isokinetic-{label}",
        sampler=optuna.samplers.TPESampler(seed=args.seed + int(round(resistance * 100)),
                                           n_startup_trials=args.startup_trials),
        storage=f"sqlite:///{output / 'study.sqlite3'}",
    )
    warm_start_sources = tuple(getattr(args, "warm_start_study", ()) or ())
    warm_start_observations = 0
    for source in warm_start_sources:
        warm_start_observations += _import_completed_trials(study, Path(source))
    enqueue_sources = tuple(getattr(args, "enqueue_study", ()) or ())
    enqueued_incumbents = [_enqueue_prior_incumbent(study, Path(source)) for source in enqueue_sources]
    target_trials = warm_start_observations + args.trials
    active: dict[Any, Any] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while len(study.trials) < target_trials or active:
            while len(active) < args.workers and len(study.trials) < target_trials:
                trial = study.ask()
                weights = tuple(trial.suggest_float(name, 0.0, 1.0) for name in MUSCLES)
                trial_root = output / "trials" / f"trial_{trial.number:03d}"
                active[pool.submit(_run_case, f"bo-{label}-{trial.number:03d}", resistance, weights,
                                        trial_root, args)] = trial
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                trial = active.pop(future)
                try:
                    record = future.result()
                    trial.set_user_attr("record", record)
                    study.tell(trial, float(record["bo_score"]))
                except Exception as error:  # preserve the remaining BO observations
                    trial.set_user_attr("error", f"{type(error).__name__}: {error}")
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                completed = [item for item in study.trials if item.value is not None]
                best = max(completed, key=lambda item: item.value, default=None)
                (output / "summary.json").write_text(json.dumps({
                    "resistance_nm": resistance, "completed": len(completed),
                    "new_completed": len(completed) - warm_start_observations,
                    "warm_start_observations": warm_start_observations,
                    "warm_start_studies": [str(Path(item).resolve()) for item in warm_start_sources],
                    "re_evaluated_neighbor_incumbents": enqueued_incumbents,
                    "score_mode": getattr(args, "bo_score", "certified_cycles"),
                    "best_trial": None if best is None else best.number,
                    "best_score": None if best is None else best.value,
                    "best_cycles": None if best is None else int(float(best.value)),
                    "best_weights": None if best is None else best.params,
                }, indent=2) + "\n", encoding="utf-8")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary["best_weights"] is None:
        raise RuntimeError(f"No completed BO observation at {resistance} Nm")
    return summary


def _cross_evaluate(best: dict[float, dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    cases: list[tuple[float, float, tuple[float, ...], Path]] = []
    for source, summary in best.items():
        weights = tuple(float(summary["best_weights"][name]) for name in MUSCLES)
        source_label = f"{source:.2f}".replace(".", "p")
        for target in args.resistances:
            target_label = f"{target:.2f}".replace(".", "p")
            cases.append((source, target, weights,
                          args.output / "cross-replay" / f"weights-{source_label}_at-{target_label}"))
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        active = {
            pool.submit(_run_case, f"transfer-{source:.2f}-to-{target:.2f}", target, weights, root, args): (source, target)
            for source, target, weights, root in cases
        }
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                source, target = active.pop(future)
                record = future.result()
                record["source_resistance_nm"] = source
                record["target_resistance_nm"] = target
                records.append(record)
    return sorted(records, key=lambda item: (item["source_resistance_nm"], item["target_resistance_nm"]))


def _write_cross_table(best: dict[float, dict[str, Any]], records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    index = {(item["source_resistance_nm"], item["target_resistance_nm"]): item for item in records}
    csv_path = args.output / "weight-transfer-certified-cycles.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["weights_optimized_at_nm", *(f"evaluated_at_{item:.2f}_nm" for item in args.resistances)])
        for source in args.resistances:
            writer.writerow([f"{source:.2f}", *(index[source, target]["certified_cycles"] for target in args.resistances)])
    lines = ["# Transfert des poids fixes Ding-local", "", "Valeur : nombre de cycles RHO certifiés.", "",
             "| poids optimisés à \\ évalués à | " + " | ".join(f"{item:.2f} Nm" for item in args.resistances) + " |",
             "|---|" + "---|" * len(args.resistances)]
    for source in args.resistances:
        lines.append("| " + f"{source:.2f} Nm" + " | " + " | ".join(
            str(index[source, target]["certified_cycles"]) for target in args.resistances) + " |")
    lines.extend(["", "Les lignes contiennent les quatre poids obtenus par BO à la résistance indiquée; les colonnes sont les résistances de rejeu."])
    (args.output / "weight-transfer-certified-cycles.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output / "best-weights.json").write_text(json.dumps(best, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "weight-transfer-records.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hsl-library", required=True)
    parser.add_argument("--resistances", type=float, nargs="+", default=[1.15, 1.20, 1.25, 1.30, 1.35])
    parser.add_argument("--max-cycles", type=int, default=180)
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu-set", default="0-11")
    parser.add_argument("--ipopt-max-iter", type=int, default=1000)
    parser.add_argument("--startup-trials", type=int, default=16)
    parser.add_argument(
        "--bo-score",
        choices=("certified_cycles", "certified_cycles_then_reserve"),
        default="certified_cycles_then_reserve",
        help="Keep certified cycles primary; optionally break exact ties by terminal capacity reserve.",
    )
    parser.add_argument(
        "--warm-start-study", type=Path, action="append", default=[],
        help=(
            "Existing Optuna study.sqlite3 file or its directory. Completed four-weight "
            "observations are imported before TPE proposes new candidates. Repeatable."
        ),
    )
    parser.add_argument(
        "--enqueue-study", type=Path, action="append", default=[],
        help=(
            "Existing study whose incumbent is re-evaluated at this resistance. "
            "Its objective value is never imported. Repeatable."
        ),
    )
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    args.root = args.root.resolve(); args.output = args.output.resolve()
    if len(args.resistances) != 5:
        raise ValueError("This transfer experiment requires exactly five resistances")
    if len(set(args.resistances)) != len(args.resistances):
        raise ValueError("Resistances must be distinct")
    args.output.mkdir(parents=True, exist_ok=False)
    campaign = vars(args).copy(); campaign["root"] = str(args.root); campaign["output"] = str(args.output)
    campaign["method"] = "Ding-local exact reduction; unilateral prescribed-isokinetic RHO"
    campaign["score"] = "certified RHO cycles; failed following window is not physiological certification"
    (args.output / "campaign.json").write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    best = {resistance: _run_bo(resistance, args) for resistance in args.resistances}
    records = _cross_evaluate(best, args)
    _write_cross_table(best, records, args)


if __name__ == "__main__":
    main()
