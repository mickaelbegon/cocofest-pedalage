#!/usr/bin/env python3
"""Audit full-horizon capacity weight selection from a certified RHO cycle."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import monotonic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--reduced-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycle-index", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--budget-seconds", type=float, default=600.)
    args = parser.parse_args()
    for key in ("source", "model_config", "reduced_profile", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    from cocofest.optimization.configured_cycling_model import (
        configured_model_factories, require_seed_fingerprint, resolve_model_config,
    )
    from cocofest.optimization.rho_adaptive_moment_policy import build_rho_adaptive_moment_policy
    from cocofest.optimization.rho_rollout_adapter import select_certified_rho_cycle
    from cocofest.optimization.endurance_weight_supervisor import (
        EnduranceWeightSupervisor, SupervisorSnapshot, WeightSupervisorConfig,
    )
    from cocofest.optimization.weighted_cycle_prediction import WeightedCyclePredictor
    config = resolve_model_config(json.loads(args.model_config.read_text()))
    fingerprint = require_seed_fingerprint(args.source, config["muscle_parameter_fingerprint"])
    with configured_model_factories(config, []):
        task = build_rho_adaptive_moment_policy(args.source, args.reduced_profile,
                                              cycle_index=args.cycle_index, cycle_period=1.)
    cycle, _ = select_certified_rho_cycle(args.source, cycle_index=args.cycle_index, cycle_period=1.)
    components = ("Cn", "F", "A", "Tau1", "Km")
    terminal = tuple(tuple(float(cycle.states[f"{component}_{name}"][cycle.end_column])
                           for component in components) for name in task.muscle_names)
    snapshot = SupervisorSnapshot(
        task_id="long-capacity-validation", context_token=config["muscle_parameter_fingerprint"],
        cycle_index=args.cycle_index, start_time_s=float(args.cycle_index + 1), created_at_s=monotonic(),
        horizon_cycles=args.horizon, muscle_names=task.muscle_names, state_component_names=components,
        start_state=terminal, incumbent_weights=(1.,) * len(task.muscle_names),
    )
    predictor = WeightedCyclePredictor(task.intervals, task.parameters, substeps=args.substeps)
    supervisor = EnduranceWeightSupervisor(WeightSupervisorConfig(
        muscle_count=len(task.muscle_names), selection_mode="full_horizon_deficit"))
    records = []
    def evaluate(weights, context):
        start = monotonic()
        result = predictor.rollout(np.asarray(context.start_state), weights,
                                   horizon_cycles=context.horizon_cycles,
                                   tracking_mode="projected_capacity", score_block_cycles=20)
        record = {"weights": list(weights), "elapsed_s": monotonic() - start,
                  "status": result.status, "completed_cycles": result.completed_cycles,
                  "first_failure": result.first_failure, "metadata": result.metadata}
        records.append(record)
        print(json.dumps({k: record[k] for k in ("weights", "elapsed_s", "status", "completed_cycles")}), flush=True)
        return result
    proposal = supervisor.evaluate(snapshot, evaluate, budget_seconds=args.budget_seconds)
    report = {"schema": "long-capacity-rollout-v1", "source": str(args.source),
              "model_fingerprint": fingerprint, "horizon_cycles": args.horizon,
              "substeps": args.substeps, "source_stimulations_per_cycle": cycle.stimulations_per_cycle,
              "uses_fho_data": False, "endurance_certified": False,
              "proposal": asdict(proposal), "rollouts": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
