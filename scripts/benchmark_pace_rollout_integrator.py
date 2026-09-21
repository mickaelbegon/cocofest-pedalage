#!/usr/bin/env python3
"""Benchmark PACE's compact exponential-midpoint rollout resolution.

PACE does not call ``solve_ivp``.  Its online predictor freezes the three slow
Ding states inside each stimulation interval, propagates the resulting affine
force equation exactly over midpoint-sampled mechanical substeps, and updates
the slow states with the matching exponential force convolution.  This script
compares cheaper substep counts with a refined instance of that same compact
model; it does not certify the compact approximation against the full Ding ODE.

Candidate weights are read from one completed predictive audit in a PACE JSONL
journal.  The source archive must be the certified one-cycle archive used by
the online adapter, so the resistance, state and moment profile are real run
inputs rather than a synthetic fixture.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.models.ding2007.ding2007_with_fatigue_periodic_node import (
    DingModelPulseWidthFrequencyWithFatiguePeriodicNode,
)
from cocofest.optimization.configured_cycling_model import (
    FIELD_ATTRIBUTES,
    resolve_model_config,
)
from cocofest.optimization.rho_adaptive_moment_policy import (
    build_rho_adaptive_moment_policy,
)
from cocofest.optimization.rho_rollout_adapter import select_certified_rho_cycle
from cocofest.optimization.batched_weighted_cycle_prediction import BatchedWeightedCyclePredictor
from cocofest.optimization.weighted_cycle_prediction import WeightedCyclePredictor


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _candidate_weights(journal: Path, cycle_index: int | None) -> tuple[int, list[list[float]]]:
    audits = []
    for line_number, line in enumerate(journal.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid journal JSON on line {line_number}: {error}") from error
        audit = row.get("predictive_audit")
        if audit and audit.get("candidate_rollouts"):
            audits.append((int(row["cycle_index"]), audit))
    if cycle_index is not None:
        audits = [item for item in audits if item[0] == cycle_index]
    if not audits:
        qualifier = "latest" if cycle_index is None else f"cycle {cycle_index}"
        raise ValueError(f"No completed predictive audit found for {qualifier}.")
    selected_cycle, audit = audits[-1]
    weights = [list(map(float, item["weights"])) for item in audit["candidate_rollouts"]]
    if not weights or any(len(item) != len(weights[0]) for item in weights):
        raise ValueError("Predictive audit contains invalid candidate weights.")
    return selected_cycle, weights


def _configured_models(model_config: Path, *, stimulations_per_cycle: int):
    declared = json.loads(model_config.read_text(encoding="utf-8"))
    config = resolve_model_config(declared)
    models = []
    for name, parameters in config["muscles"].items():
        model = DingModelPulseWidthFrequencyWithFatiguePeriodicNode(
            muscle_name=name,
            stim_time=[0.0, 1.0 / stimulations_per_cycle],
            sum_stim_truncation=6,
        )
        for key, value in parameters.items():
            setattr(model, FIELD_ATTRIBUTES[key], value)
        model.a_rest = model.a_scale
        models.append(model)
    return tuple(models), config


def _finite_max(values) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return None if not values.size else float(np.max(np.abs(values)))


def _compare(result, reference, rest) -> dict:
    shared = min(result.completed_intervals, reference.completed_intervals)
    state_delta = result.state_history[: shared + 1] - reference.state_history[: shared + 1]
    slow_error = state_delta[:, :, 2:] / rest[None, :, :]
    pulse_width_error = (result.pulse_widths - reference.pulse_widths) * 1e6
    deficit = result.full_horizon_normalized_deficit
    reference_deficit = reference.full_horizon_normalized_deficit
    reserve = result.terminal_normalized_reserve
    reference_reserve = reference.terminal_normalized_reserve
    return {
        "status_matches_reference": result.status == reference.status,
        "completed_intervals_matches_reference": result.completed_intervals == reference.completed_intervals,
        "shared_intervals": shared,
        "maximum_force_error_n": _finite_max(state_delta[:, :, 1]),
        "maximum_slow_state_error_over_rest": _finite_max(slow_error),
        "maximum_pulse_width_error_us": _finite_max(pulse_width_error),
        "full_horizon_normalized_deficit_absolute_error": (
            None if deficit is None or reference_deficit is None else abs(deficit - reference_deficit)
        ),
        "terminal_normalized_reserve_absolute_error": (
            None if reserve is None or reference_reserve is None else abs(reserve - reference_reserve)
        ),
    }


def _rollout_set(policy, initial, weights, *, cycles, substeps, update_every_cycles, repeats,
                 candidate_backend):
    durations = []
    results = None
    for _ in range(repeats):
        predictor = WeightedCyclePredictor(
            policy.intervals,
            policy.parameters,
            substeps=substeps,
            reference_regularization=1e-3,
        )
        started = perf_counter()
        if candidate_backend == "batch":
            results = BatchedWeightedCyclePredictor(predictor).rollout_many(
                initial, weights, horizon_cycles=cycles, tracking_mode="projected_capacity",
                score_block_cycles=update_every_cycles,
            )
        else:
            results = [
                predictor.rollout(
                    initial, candidate, horizon_cycles=cycles, tracking_mode="projected_capacity",
                    score_block_cycles=update_every_cycles,
                )
                for candidate in weights
            ]
        durations.append(perf_counter() - started)
    return results, durations


def _ranking(results) -> list[int] | None:
    scores = [result.full_horizon_normalized_deficit for result in results]
    if any(score is None or not math.isfinite(score) for score in scores):
        return None
    return sorted(range(len(scores)), key=scores.__getitem__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reduced-profile", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--pace-journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal-cycle", type=int)
    parser.add_argument("--horizon-cycles", type=int, default=100)
    parser.add_argument("--update-every-cycles", type=int, default=20)
    parser.add_argument("--substeps", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--reference-substeps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--candidate-backend", choices=("batch", "scalar"), default="batch",
                        help="batch matches the online PACE worker; scalar is retained for A/B diagnostics")
    args = parser.parse_args(argv)
    positive_integers = {
        "horizon_cycles": args.horizon_cycles,
        "update_every_cycles": args.update_every_cycles,
        "reference_substeps": args.reference_substeps,
        "repeats": args.repeats,
        **{f"substeps[{index}]": value for index, value in enumerate(args.substeps)},
    }
    if any(isinstance(value, bool) or value < 1 for value in positive_integers.values()):
        parser.error("cycle counts, repeats and substeps must be positive integers")
    if args.reference_substeps in args.substeps:
        parser.error("reference-substeps must not also occur in --substeps")

    for name in ("source", "reduced_profile", "model_config", "pace_journal"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output = args.output.expanduser().resolve()
    cycle, _ = select_certified_rho_cycle(args.source, cycle_index=0, cycle_period=1.0)
    models, model_config = _configured_models(
        args.model_config, stimulations_per_cycle=cycle.stimulations_per_cycle
    )
    policy = build_rho_adaptive_moment_policy(
        args.source,
        args.reduced_profile,
        cycle_index=0,
        cycle_period=1.0,
        muscle_models=models,
    )
    journal_cycle, weights = _candidate_weights(args.pace_journal, args.journal_cycle)
    if len(weights[0]) != len(policy.muscle_names):
        raise ValueError("Journal candidate dimension does not match the source muscles.")
    initial = np.asarray(
        [
            [cycle.states[f"{component}_{name}"][cycle.end_column] for component in ("Cn", "F", "A", "Tau1", "Km")]
            for name in policy.muscle_names
        ],
        dtype=float,
    )
    rest = np.asarray([parameter.fatigue.rest_state for parameter in policy.parameters])

    reference, reference_durations = _rollout_set(
        policy,
        initial,
        weights,
        cycles=args.horizon_cycles,
        substeps=args.reference_substeps,
        update_every_cycles=args.update_every_cycles,
        repeats=args.repeats,
        candidate_backend=args.candidate_backend,
    )
    reference_ranking = _ranking(reference)
    report = {
        "schema": "cocofest-pace-rollout-integrator-benchmark-v1",
        "integrator": {
            "online_implementation": "compact_exponential_midpoint",
            "description": (
                "slow Ding states frozen in the force equation per stimulation interval; "
                "piecewise-constant midpoint mechanical/calcium coefficients; exact affine "
                "force flow and exact matching exponential fatigue convolution per substep"
            ),
            "not_used_online": ["Radau", "DOP853", "RK4"],
        },
        "source": {"path": str(args.source), "sha256": _sha256(args.source)},
        "reduced_profile": {"path": str(args.reduced_profile), "sha256": _sha256(args.reduced_profile)},
        "model_config": {
            "path": str(args.model_config),
            "sha256": _sha256(args.model_config),
            "case_id": model_config["case_id"],
            "muscle_parameter_fingerprint": model_config["muscle_parameter_fingerprint"],
        },
        "pace_journal": {"path": str(args.pace_journal), "sha256": _sha256(args.pace_journal)},
        "source_signed_crank_torque_nm": policy.source_signed_crank_torque_nm,
        "journal_cycle": journal_cycle,
        "muscle_names": list(policy.muscle_names),
        "stimulations_per_cycle": len(policy.intervals),
        "horizon_cycles": args.horizon_cycles,
        "update_every_cycles": args.update_every_cycles,
        "candidate_count": len(weights),
        "candidate_weights": weights,
        "candidate_evaluation_backend": args.candidate_backend,
        "runtime": {"python": sys.version, "platform": platform.platform(), "pid": os.getpid()},
        "reference": {
            "substeps": args.reference_substeps,
            "durations_s": reference_durations,
            "median_duration_s": statistics.median(reference_durations),
            "ranking": reference_ranking,
            "statuses": [result.status for result in reference],
        },
        "candidates": [],
        "limitations": [
            "The refined reference is the same compact model, not a full-Ding or RHO certificate.",
            "One archived RHO boundary and one recorded candidate set are timed.",
            "Wall times are host- and load-dependent; rerun locally before fixing a production budget.",
        ],
    }
    print(json.dumps({"reference_substeps": args.reference_substeps, "duration_s": reference_durations}), flush=True)
    for substeps in args.substeps:
        results, durations = _rollout_set(
            policy,
            initial,
            weights,
            cycles=args.horizon_cycles,
            substeps=substeps,
            update_every_cycles=args.update_every_cycles,
            repeats=args.repeats,
            candidate_backend=args.candidate_backend,
        )
        median = statistics.median(durations)
        ranking = _ranking(results)
        comparisons = [_compare(result, ref, rest) for result, ref in zip(results, reference, strict=True)]
        entry = {
            "substeps": substeps,
            "durations_s": durations,
            "median_duration_s": median,
            "amortized_seconds_per_rho_cycle": median / args.update_every_cycles,
            "speedup_vs_reference": statistics.median(reference_durations) / median,
            "statuses": [result.status for result in results],
            "ranking": ranking,
            "ranking_matches_reference": ranking == reference_ranking,
            "candidate_comparisons": comparisons,
            "maximum_force_error_n": max(item["maximum_force_error_n"] or 0.0 for item in comparisons),
            "maximum_slow_state_error_over_rest": max(
                item["maximum_slow_state_error_over_rest"] or 0.0 for item in comparisons
            ),
            "maximum_pulse_width_error_us": max(
                item["maximum_pulse_width_error_us"] or 0.0 for item in comparisons
            ),
            "maximum_deficit_score_absolute_error": max(
                item["full_horizon_normalized_deficit_absolute_error"] or 0.0 for item in comparisons
            ),
            "maximum_terminal_reserve_absolute_error": max(
                item["terminal_normalized_reserve_absolute_error"] or 0.0 for item in comparisons
            ),
        }
        report["candidates"].append(entry)
        print(json.dumps(entry), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {args.output}", flush=True)
    return report


if __name__ == "__main__":
    main()
