#!/usr/bin/env python3
"""Run a zero-objective frozen feasibility probe from an exact prepared RHO state.

This utility is deliberately conservative.  A certified feasible witness proves
that one next cycle exists under the preserved model, work and device bounds.
An unsuccessful IPOPT search is reported as such, never as physiological
exhaustion.
"""

from __future__ import annotations

import argparse
from copy import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.task_load_margin_ocp import (
    TaskLoadMarginObjectiveBinding,
    periodic_ding_conditional_context,
    restore_source_with_inactive_load_binding,
)
from cocofest.optimization.task_reserve_ocp import TaskReserveStateCoordinate
from cocofest.simulation.independent_arms_process import _atomic_json, _configured_payload_model, _driver_arguments
from cocofest.simulation.rho_restart_checkpoint import restore_prepared_checkpoint
from scripts.probe_independent_rho_task_reserve import _load_arm


def _load_margin_binding(source, args, model_path: Path):
    """Construct the same inactive experimental graph used by the refresh RHO."""
    model = json.loads(model_path.read_text())
    layout = tuple(TaskReserveStateCoordinate(f"{key}_{name}", scale=scale)
                   for name, parameters in model["muscles"].items()
                   for key, scale in (("Cn", 1.), ("F", float(parameters["Fmax"])),
                                      ("A", float(parameters["a_scale"])), ("Tau1", .05), ("Km", .1)))
    context = json.loads(source.task_context_json)
    context.update(coordinate_layout=[asdict(item) for item in layout],
                   coordinate_definition="complete_ding_state_normalization_v1")
    # The digest is rebuilt from the restored physical boundary before a
    # binding could be activated. Its value only needs to satisfy graph-build
    # validation here because the frozen probe leaves the parameter at zero.
    placeholder = "0" * 64
    physical_context = {
        "cycle_period_s": abs(2 * 3.141592653589793 / args.isokinetic_omega),
        "cycle_len": args.stimulations_per_cycle, "formulation": "isokinetic",
        "mechanical_formulation": "reduced", "nominal_work_j": context["nominal_work_j"],
        "terminal_half_step_guard": bool(args.reduced_terminal_half_step_velocity_guard),
    }
    return TaskLoadMarginObjectiveBinding(layout, task_context=context,
        model_sha256=source.model_sha256, physical_context=physical_context,
        conditional_context_sha256=placeholder), layout


def run(*, receipt: Path, checkpoint: Path, side: str, output_directory: Path,
        experimental_load_margin_graph: bool, max_iterations: int = 4000) -> dict:
    from cocofest.optimization.configured_cycling_model import configured_model_factories
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver

    source, payload = _load_arm(receipt, side)
    args = copy(_driver_arguments(payload, side))
    args.nlp_ipopt_recovery = args.nlp_ipopt_recovery_ma57_tuned = False
    configured = _configured_payload_model(payload, side)
    if experimental_load_margin_graph:
        binding, layout = _load_margin_binding(source, args, Path(source.model_path))
        args.experimental_task_load_margin_binding = binding
        args.experimental_task_load_margin_model_path = source.model_path
    else:
        binding, layout = None, ()
    with configured_model_factories(configured[1], []):
        runtime = driver.build_unilateral_runtime(args, echo=False)
    program = runtime["nmpc"]
    program._initialize_state_idx_to_cycle({"states": {}})
    checkpoint = Path(checkpoint).resolve(strict=True)
    with __import__("numpy").load(checkpoint, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata__json"].item()))
    completed = int(metadata["producer_completed_windows"])
    if binding is None:
        restored = restore_prepared_checkpoint(checkpoint, program, completed_cycles=completed)
    else:
        restored = restore_source_with_inactive_load_binding(checkpoint, program, completed_cycles=completed)
        signature, _ = periodic_ding_conditional_context(program, layout)
        if signature == "0" * 64:
            raise RuntimeError("Invalid conditional context digest")
    start = perf_counter()
    probe = driver.run_frozen_rho_zero_objective_feasibility_probe(
        program, max_iterations=max_iterations, tolerance=float(args.nlp_tolerance),
        linear_solver="ma57", echo=False)
    report = {
        "schema_version": 1,
        "mode": "frozen_one_cycle_zero_objective_feasibility_probe",
        "source_receipt": str(Path(receipt).resolve()), "checkpoint": str(checkpoint),
        "side": side, "completed_cycles_at_boundary": completed,
        "experimental_load_margin_graph": experimental_load_margin_graph,
        "restored": restored, "probe": probe, "wall_time_s": perf_counter() - start,
        "next_cycle_feasible_witness": bool(probe.get("feasible_witness")),
        "physiological_failure_certified": False,
        "interpretation": (
            "A feasible witness establishes an admissible next cycle. "
            "A failed local feasibility search is not a proof of exhaustion."
        ),
    }
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_directory / "viability-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--experimental-load-margin-graph", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=4000)
    parser.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args()
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("CPU outside allowed affinity")
    os.sched_setaffinity(0, {args.cpu})
    report = run(receipt=args.receipt, checkpoint=args.checkpoint, side=args.side,
                 output_directory=args.output_directory,
                 experimental_load_margin_graph=args.experimental_load_margin_graph,
                 max_iterations=args.max_iterations)
    print(json.dumps({key: report[key] for key in ("next_cycle_feasible_witness", "wall_time_s")}, indent=2))


if __name__ == "__main__":
    main()
