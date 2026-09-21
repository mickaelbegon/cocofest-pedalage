#!/usr/bin/env python3
"""Compare legacy and guarded PACE from the same certified RHO checkpoint."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from time import monotonic

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_pace_rollout_integrator import _configured_models
from cocofest.optimization.rho_rollout_adapter import select_certified_rho_cycle
from cocofest.optimization.rho_adaptive_moment_policy import build_rho_adaptive_moment_policy
from cocofest.optimization.endurance_weight_supervisor import (
    EnduranceWeightSupervisor, SupervisorSnapshot, WeightSupervisorConfig,
)
from cocofest.optimization.weighted_cycle_prediction import WeightedCyclePredictor
from cocofest.optimization.batched_weighted_cycle_prediction import BatchedWeightedCyclePredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--reduced-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--substeps", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cycle, _ = select_certified_rho_cycle(args.source, cycle_index=0, cycle_period=1.)
    models, config = _configured_models(args.model_config, stimulations_per_cycle=cycle.stimulations_per_cycle)
    task = build_rho_adaptive_moment_policy(args.source, args.reduced_profile, cycle_index=0,
                                           cycle_period=1., muscle_models=models)
    components = ("Cn", "F", "A", "Tau1", "Km")
    initial = tuple(tuple(float(cycle.states[f"{component}_{name}"][cycle.end_column])
                          for component in components) for name in task.muscle_names)
    snapshot = SupervisorSnapshot(
        task_id="guarded-pace-validation", context_token=config["muscle_parameter_fingerprint"],
        cycle_index=0, start_time_s=1., created_at_s=monotonic(), horizon_cycles=args.horizon,
        muscle_names=task.muscle_names, state_component_names=components,
        start_state=initial, incumbent_weights=(1.,) * len(task.muscle_names))
    predictor = WeightedCyclePredictor(task.intervals, task.parameters, substeps=args.substeps)
    output = {"uses_fho_data": False, "source": str(args.source.resolve()),
              "horizon_cycles": args.horizon, "substeps": args.substeps,
              "incumbent": "uniform weights, at the existing checkpoint state", "comparisons": {}}
    for mode in ("full_horizon_deficit", "guarded_fatigue"):
        supervisor = EnduranceWeightSupervisor(WeightSupervisorConfig(
            selection_mode=mode, max_candidate_log_step=math.log(1.1) if mode == "guarded_fatigue" else None))
        candidates = supervisor.candidates(snapshot)
        started = monotonic()
        results = BatchedWeightedCyclePredictor(predictor).rollout_many(
            np.asarray(initial), [c.weights for c in candidates], horizon_cycles=args.horizon,
            tracking_mode="projected_capacity", score_block_cycles=20)
        elapsed = monotonic() - started
        mapping = {c.weights: r for c, r in zip(candidates, results)}
        proposal = supervisor.evaluate(snapshot, lambda weights, _: mapping[weights], budget_seconds=16.)
        output["comparisons"][mode] = {
            "batch_rollout_elapsed_s": elapsed, "config": asdict(supervisor.config),
            "proposal": asdict(proposal),
        }
        print(json.dumps({"mode": mode, "elapsed_s": elapsed, "weights": proposal.weights,
                          "selection_basis": proposal.selection_basis}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
