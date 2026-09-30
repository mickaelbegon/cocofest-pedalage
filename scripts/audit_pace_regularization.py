#!/usr/bin/env python3
"""Audit PACE weight sensitivity, then prepare an optional regularization pilot.

Historical guard diagnostics use the archived online projections exactly.
The epsilon sweep uses a prepared next-RHO checkpoint as a *local surrogate
anchor*.  That shifted warm-start profile is not the archived certified cycle
used by the online worker; it must not be reported as an exact replay or a
certification of future movement.  Real static RHO continuations are prepared
separately to test whether the surrogate's candidate ranking is useful.

No production default is changed and no OCP is launched by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.endurance_weight_supervisor import (
    EnduranceWeightSupervisor, SupervisorSnapshot, WeightSupervisorConfig,
)
from scripts.validate_pace_decision_fidelity import (
    MUSCLE_NAMES, _candidate_document, _load_journal, _option_value_pairs,
    _read_json, _write_new_json, prepare_validation,
)


FATIGUE_FIELDS = ("full_horizon_mean_squared_fatigue", "first_block_mean_squared_fatigue")


def guard_rejections(candidate: dict, incumbent: dict, config: dict) -> list[str]:
    """Explain the established guard without relaxing any of its inequalities."""
    names = (*FATIGUE_FIELDS, "terminal_minimum_capacity", "full_horizon_normalized_deficit")
    if any(record is None or any(record.get(name) is None for name in names)
           for record in (candidate, incumbent)):
        return ["missing_evidence"]
    reasons = []
    for name in FATIGUE_FIELDS:
        if candidate[name] > incumbent[name] + config["fatigue_noninferiority_tolerance"]:
            reasons.append(name + "_worse")
    if candidate["terminal_minimum_capacity"] + config["capacity_noninferiority_tolerance"] < incumbent["terminal_minimum_capacity"]:
        reasons.append("minimum_capacity_worse")
    if candidate["full_horizon_normalized_deficit"] > incumbent["full_horizon_normalized_deficit"] + 1e-12:
        reasons.append("mechanical_deficit_worse")
    if not any(incumbent[name] - candidate[name] > max(
            config["minimum_absolute_improvement"],
            config["minimum_relative_improvement"] * abs(incumbent[name]))
            for name in (FATIGUE_FIELDS[0], "full_horizon_normalized_deficit")):
        reasons.append("no_material_gain")
    return reasons


def audit_journal(records: list[dict]) -> dict:
    projections, all_reasons = [], Counter()
    for record in records:
        if record.get("event") != "projection_completed" or not record.get("candidate_evaluations"):
            continue
        evaluations = record["candidate_evaluations"]
        incumbent = next(item["result"] for item in evaluations if item["candidate"]["kind"] == "incumbent")
        reasons, candidates = Counter(), []
        for item in evaluations:
            if item["candidate"]["kind"] == "incumbent":
                continue
            rejected = guard_rejections(item.get("result"), incumbent, record["selection_config"])
            reasons.update(rejected)
            candidates.append({"candidate": item["candidate"], "rejection_reasons": rejected})
        spreads = {}
        for field in (*FATIGUE_FIELDS, "terminal_minimum_capacity", "full_horizon_normalized_deficit"):
            values = [item["result"][field] for item in evaluations
                      if item.get("result") is not None and item["result"].get(field) is not None]
            spreads[field] = max(values) - min(values) if values else None
        projections.append({"cycle": record["source_cycle_index"], "candidate_count": len(evaluations),
                            "selected_candidate": record.get("selected_candidate"),
                            "metric_spreads": spreads, "rejection_counts": dict(reasons),
                            "candidates": candidates})
        all_reasons.update(reasons)
    return {"projections": projections, "rejection_counts": dict(all_reasons),
            "classification_scope": "archived_online_candidates; counts may overlap for one candidate"}


def sweep_regularization(*, pace_directory: Path, checkpoint: int, epsilons: list[float],
                         horizon: int) -> dict:
    # Heavy dynamics imports are unnecessary for journal-only helpers/tests.
    from scripts.benchmark_pace_rollout_integrator import _configured_models
    from cocofest.optimization.rho_adaptive_moment_policy import build_rho_adaptive_moment_policy
    from cocofest.optimization.rho_rollout_adapter import select_certified_rho_cycle
    from cocofest.optimization.weighted_cycle_prediction import WeightedCyclePredictor
    from cocofest.optimization.batched_weighted_cycle_prediction import BatchedWeightedCyclePredictor

    if not epsilons or any(not math.isfinite(value) or value <= 0 for value in epsilons):
        raise ValueError("epsilons must be finite and strictly positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    audit = _read_json(pace_directory / "configuration-audit.json")
    arguments = audit["arguments"]
    profile = Path(arguments[arguments.index("--reduced-cycling-profile") + 1])
    source = pace_directory / "checkpoints" / f"cycle-{checkpoint}.npz"
    receipt = _read_json(source.with_suffix(".receipt.json"))
    if receipt["completed_windows"] != checkpoint:
        raise ValueError("Checkpoint receipt does not match the requested boundary")
    records = _load_journal(Path(audit["weights_journal_path"]))
    historical = next(item for item in records if item.get("event") == "projection_completed"
                      and item.get("source_cycle_index") == checkpoint)
    cycle, _ = select_certified_rho_cycle(source, cycle_index=0, cycle_period=1.)
    models, _ = _configured_models(Path(audit["model_config_path"]),
                                   stimulations_per_cycle=cycle.stimulations_per_cycle)
    policy = build_rho_adaptive_moment_policy(source, profile, cycle_index=0,
                                            cycle_period=1., muscle_models=models)
    initial = tuple(tuple(float(cycle.states[f"{state}_{name}"][cycle.end_column])
                          for state in ("Cn", "F", "A", "Tau1", "Km")) for name in policy.muscle_names)
    incumbent = next(item["candidate"]["weights"] for item in historical["candidate_evaluations"]
                     if item["candidate"]["kind"] == "incumbent")
    snapshot = SupervisorSnapshot(
        task_id="pace-regularization-local-audit", context_token=sha256(source.read_bytes()).hexdigest(),
        cycle_index=checkpoint, start_time_s=float(checkpoint), created_at_s=0., horizon_cycles=horizon,
        muscle_names=policy.muscle_names, state_component_names=("Cn", "F", "A", "Tau1", "Km"),
        start_state=initial, incumbent_weights=incumbent)
    supervisor = EnduranceWeightSupervisor(WeightSupervisorConfig(**historical["selection_config"]))
    candidates = supervisor.candidates(snapshot)
    reports = []
    for epsilon in epsilons:
        predictor = WeightedCyclePredictor(policy.intervals, policy.parameters, substeps=16,
                                          reference_regularization=epsilon,
                                          allocation_objective="predicted_ding_fatigue_v1")
        started = perf_counter()
        results = BatchedWeightedCyclePredictor(predictor).rollout_many(
            np.asarray(initial), [item.weights for item in candidates], horizon_cycles=horizon,
            tracking_mode="projected_capacity", score_block_cycles=historical["update_every_cycles"])
        elapsed = perf_counter() - started
        by_weights = {candidate.weights: result for candidate, result in zip(candidates, results, strict=True)}
        proposal = supervisor.evaluate(snapshot, lambda weights, _: by_weights[weights],
                                       budget_seconds=1e6)
        pulse_widths = np.asarray([result.pulse_widths for result in results])
        recruitment = -np.expm1(-(pulse_widths - predictor.pd0[None, None, :, None]) /
                               predictor.pdt[None, None, :, None]) / predictor.maximum_recruitment[None, None, :, None]
        reports.append({
            "epsilon": epsilon, "wall_seconds": elapsed, "candidate_count": len(candidates),
            "maximum_candidate_pw_difference_us": float(np.nanmax(abs(pulse_widths - pulse_widths[0])) * 1e6),
            "incumbent_lower_bound_fraction": float(np.mean(recruitment[0] < 1e-8)),
            "incumbent_upper_bound_fraction": float(np.mean(recruitment[0] > 1. - 1e-8)),
            "selection_basis": proposal.selection_basis,
            "selected_candidate": None if proposal.chosen_candidate is None else asdict(proposal.chosen_candidate),
            "evaluations": [asdict(item) for item in proposal.evaluations],
        })
    return {"source": str(source), "source_sha256": sha256(source.read_bytes()).hexdigest(),
            "checkpoint": checkpoint, "horizon_cycles": horizon, "muscle_names": list(policy.muscle_names),
            "anchor_semantics": "prepared_shifted_next_RHO_primal; not_exact_archived_online_projection",
            "limitations": ["The shifted profile is a warm start, not a newly solved cycle.",
                            "No improved endurance is inferred from surrogate costs.",
                            "These predictions require validation by real paired RHO continuations."],
            "historical_online_audit": audit_journal(records), "sweep": reports}


def prepare_pilot(*, pace_directory: Path, output_directory: Path, python: Path,
                  epsilon: float) -> Path:
    """Clone the actual nominal arm, changing only the slow regularization."""
    audit = _read_json(pace_directory / "configuration-audit.json")
    config = _read_json(Path(audit["weights_config_path"]))
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    config["policy"]["projection_fatigue_reference_regularization"] = epsilon
    config["initial_weight_basis"] += f"; optional_reference_regularization_pilot={epsilon}"
    weights = output_directory / "weights.json"
    _write_new_json(weights, config)
    result = output_directory / "result.json"
    benchmark = _option_value_pairs(audit["arguments"], {
        "--output-json": str(result),
        "--n-windows": str(config["policy"]["max_cycles"]),
        "--common-initial-solution": audit["arguments"][audit["arguments"].index("--common-initial-solution") + 1],
        "--rho-prepared-checkpoint-output-template": str(output_directory / "checkpoints/cycle-{completed_windows}.npz"),
        "--rho-prepared-checkpoint-windows": "60,180,300,420",
    })
    benchmark += ["--common-initial-solution-recenter-first-node-bounds",
                  "--adopt-common-initial-solution-warmup-cycles"]
    command = [str(python), str(ROOT / "scripts/run_configured_cycling_benchmark.py"),
               "--model-config", audit["model_config_path"], "--condition", "rho-pace",
               "--weights-config", str(weights), "--weights-journal", str(output_directory / "weights.jsonl"),
               "--configuration-audit", str(output_directory / "configuration-audit.json"), "--", *benchmark]
    destination = output_directory / "command.json"
    _write_new_json(destination, {"command": command, "cwd": str(ROOT), "launched": False,
                                 "environment": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                                 "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg"},
                                 "source_pace_directory": str(pace_directory),
                                 "only_policy_change": {"projection_fatigue_reference_regularization": epsilon}})
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pace-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1e-12, 1e-8, 1e-6, 1e-5, 1e-4])
    parser.add_argument("--prepare-pilot", action="store_true")
    parser.add_argument("--pilot-epsilon", type=float, default=1e-5)
    parser.add_argument("--bo-weights-config", type=Path,
                        help="Also prepare exact static continuations, including each sweep-selected candidate")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args(argv)
    source = args.pace_directory.resolve(strict=True)
    destination = args.output_directory.resolve()
    if destination.exists():
        raise FileExistsError(f"Output directory must be fresh: {destination}")
    report = sweep_regularization(pace_directory=source, checkpoint=args.checkpoint,
                                  epsilons=args.epsilons, horizon=args.horizon)
    configs = {}
    selected_weights = {}
    for entry in report["sweep"]:
        if entry["selected_candidate"] is None:
            continue
        # Identical static continuations carry no extra decision evidence.
        # Keep epsilon provenance in the audit while preparing each new
        # selected weight vector only once.
        if entry["selected_candidate"]["kind"] == "incumbent":
            entry["continuation_candidate_id"] = "pace_current"
            continue
        name = "epsilon_" + f"{entry['epsilon']:.0e}".replace("-", "m").replace("+", "p")
        key = tuple(round(weight, 12) for weight in entry["selected_candidate"]["weights"])
        if key in selected_weights:
            entry["continuation_candidate_id"] = selected_weights[key]
            continue
        selected_weights[key] = name
        entry["continuation_candidate_id"] = name
        config = destination / "candidates" / f"{name}.json"
        _write_new_json(config, _candidate_document(
            candidate_id=name, weights=tuple(entry["selected_candidate"]["weights"]), checkpoint=args.checkpoint,
            source=f"local_shifted_checkpoint_sweep; epsilon={entry['epsilon']}; not_original_online_proposal", proposal=None))
        configs[name] = config
    _write_new_json(destination / "audit.json", report)
    if args.prepare_pilot:
        label = f"{args.pilot_epsilon:.0e}".replace("-", "m").replace("+", "p")
        prepare_pilot(pace_directory=source, output_directory=destination / f"pilot-epsilon-{label}",
                      python=args.python.resolve(strict=True), epsilon=args.pilot_epsilon)
    if args.bo_weights_config is not None:
        prepare_validation(pace_directory=source, bo_weights_config=args.bo_weights_config,
                           output_directory=destination / "continuations", python=args.python,
                           checkpoints=(args.checkpoint,), horizons=(1, 5, 20), candidate_configs=configs)
    print(f"Prepared audit and optional experiments (no OCP launched): {destination}", flush=True)


if __name__ == "__main__":
    main()
