#!/usr/bin/env python3
"""Benchmark a reused load-margin supervisor on successive real RHO boundaries.

This is an observational benchmark: the supervisor does not change the RHO
decision. A feasible local load witness is not a physiological failure proof.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.task_load_margin_oracle import (
    audit_load_solution, configure_load_objective, initial_bound_envelope_gradient,
)
from cocofest.optimization.task_reserve_ocp import TaskReserveStateCoordinate
from cocofest.optimization.task_reserve_probe_adapter import independent_full_nlp_audit
from cocofest.simulation.independent_arms_process import _atomic_json, _configured_payload_model, _driver_arguments
from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint, restore_prepared_checkpoint
from scripts.probe_independent_rho_task_reserve import _load_arm
from scripts.run_local_task_reserve_branch import _advance


def full_ding_layout(model_path: Path) -> tuple[TaskReserveStateCoordinate, ...]:
    model = json.loads(Path(model_path).read_text())
    return tuple(TaskReserveStateCoordinate(f"{key}_{name}", scale=scale)
                 for name, parameters in model["muscles"].items()
                 for key, scale in (("Cn", 1.), ("F", float(parameters["Fmax"])),
                                    ("A", float(parameters["a_scale"])), ("Tau1", .05), ("Km", .1)))


def run(receipt: Path, side: str, output_directory: Path, *, cycles: int = 3,
        upper_bound: float = 3., tolerance: float = 1e-6) -> dict:
    from cocofest.optimization.configured_cycling_model import configured_model_factories
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver

    if cycles < 1 or not np.isfinite(upper_bound) or upper_bound <= 1:
        raise ValueError("Positive cycles and load upper bound >1 required")
    source, payload = _load_arm(Path(receipt), side)
    source.verify_files()
    work = float(json.loads(source.task_context_json)["nominal_work_j"])
    layout = full_ding_layout(Path(source.model_path))
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    args = copy(_driver_arguments(payload, side))
    args.nlp_ipopt_recovery = args.nlp_ipopt_recovery_ma57_tuned = False
    configured = _configured_payload_model(payload, side)

    def build_runtime():
        with configured_model_factories(configured[1], []):
            runtime = driver.build_unilateral_runtime(args, echo=False)
        runtime["nmpc"]._initialize_state_idx_to_cycle({"states": {}})
        return runtime

    start = perf_counter()
    baseline = build_runtime()
    supervisor = build_runtime()
    build_seconds = perf_counter() - start
    baseline_program, baseline_solver = baseline["nmpc"], baseline["solver"]
    supervisor_program, supervisor_solver = supervisor["nmpc"], supervisor["solver"]
    restore = restore_prepared_checkpoint(Path(source.archive_path), baseline_program,
                                          completed_cycles=source.completed_cycles)
    if restore["restored_problem_sha256"] != source.prepared_problem_sha256:
        raise RuntimeError("Baseline checkpoint restoration differs from certified source")
    _atomic_json(output / "protocol.json", {
        "mode": "observational_reused_supervisor_refresh_every_cycle",
        "source_receipt": str(Path(receipt).resolve()), "side": side,
        "source_completed_cycles": source.completed_cycles, "requested_cycles": cycles,
        "nominal_work_j": work, "upper_bound": upper_bound, "tolerance": tolerance,
        "build_seconds_two_programs": build_seconds, "source_restore": restore,
        "interpretation": "No supervisor direction is applied to baseline RHO; no physiological failure certificate",
    })
    records = []
    configured_once = False
    def coordinates(program, states=None):
        return np.asarray([((float(program.nlp[0].x_bounds[c.state_key].min[c.index, 0])
             if states is None else float(np.asarray(states[c.state_key])[c.index, -1]))
             - c.offset) / c.scale for c in layout], dtype=float)
    for offset in range(cycles):
        completed = source.completed_cycles + offset
        checkpoint = output / f"prepared-cycle-{completed}.npz"
        export = export_prepared_checkpoint(checkpoint, baseline_program,
                                             completed_cycles=completed)
        refresh_start = perf_counter()
        restored = restore_prepared_checkpoint(checkpoint, supervisor_program,
                                               completed_cycles=completed)
        if restored["restored_problem_sha256"] != export["prepared_problem_sha256"]:
            raise RuntimeError("Supervisor restored a different physical boundary")
        if not configured_once:
            configure_load_objective(supervisor_program, nominal_work_j=work,
                                     upper_bound=upper_bound)
            configured_once = True
        else:
            bounds = supervisor_program.nlp[0].x_bounds["E_prod"]
            if float(bounds.min[0, 2]) != work or float(bounds.max[0, 2]) != work:
                raise RuntimeError("Restored supervisor work target differs from nominal")
            bounds.min[0, 2] = 0.
            bounds.max[0, 2] = work * upper_bound
        solution = super(driver.RecedingHorizonOptimization, supervisor_program).solve(
            solver=supervisor_solver, warm_start=None)
        center = coordinates(supervisor_program)
        load, audit, kkt = audit_load_solution(solution, supervisor_program,
            nominal_work_j=work, upper_bound=upper_bound, tolerance=tolerance)
        gradient = None
        if audit.passed and all(value is not None and np.isfinite(value) for value in kkt.values()):
            gradient = initial_bound_envelope_gradient(solution, supervisor_program, layout)
        refresh_seconds = perf_counter() - refresh_start
        baseline_start = perf_counter()
        baseline_solution = super(driver.RecedingHorizonOptimization, baseline_program).solve(
            solver=baseline_solver, warm_start=None)
        from bioptim import SolutionMerge
        baseline_terminal = coordinates(baseline_program,
            baseline_solution.decision_states(to_merge=SolutionMerge.NODES))
        baseline_audit = independent_full_nlp_audit(baseline_solution, baseline_program,
            target_work_j=work, tolerance=tolerance)
        baseline_seconds = perf_counter() - baseline_start
        record = {
            "cycle": completed + 1, "supervisor_load_factor": load,
            "supervisor_audit": asdict(audit), "supervisor_kkt": kkt,
            "supervisor_gradient": gradient, "supervisor_seconds": refresh_seconds,
            "supervisor_solver_seconds": float(solution.real_time_to_optimize),
            "baseline_audit": asdict(baseline_audit), "baseline_seconds": baseline_seconds,
            "baseline_solver_seconds": float(baseline_solution.real_time_to_optimize),
            "center_coordinates": center.tolist(),
            "baseline_terminal_coordinates": baseline_terminal.tolist(),
            "baseline_terminal_delta": (baseline_terminal - center).tolist(),
            "supervisor_object_id": id(supervisor_program),
            "supervisor_ocp_solver_id": id(supervisor_program.ocp_solver),
        }
        records.append(record)
        _atomic_json(output / "progress.json", {"cycles": records})
        print(json.dumps({"cycle": record["cycle"], "load_factor": load,
            "supervisor_seconds": refresh_seconds, "baseline_seconds": baseline_seconds,
            "supervisor_passed": audit.passed, "baseline_passed": baseline_audit.passed}), flush=True)
        if not audit.passed or not baseline_audit.passed:
            break
        if offset + 1 < cycles:
            _advance(baseline_program, baseline_solution, work)
    report = {
        "source_completed_cycles": source.completed_cycles,
        "observed_cycles": len(records),
        "certified_baseline_cycles": sum(r["baseline_audit"]["passed"] for r in records),
        "certified_supervisor_refreshes": sum(r["supervisor_audit"]["passed"] for r in records),
        "supervisor_reused": len({r["supervisor_object_id"] for r in records}) == 1,
        "solver_interface_reused": len({r["supervisor_ocp_solver_id"] for r in records}) == 1,
        "mean_supervisor_seconds": float(np.mean([r["supervisor_seconds"] for r in records])),
        "mean_baseline_seconds": float(np.mean([r["baseline_seconds"] for r in records])),
        "warm_supervisor_median_seconds": float(np.median([r["supervisor_seconds"] for r in records[1:]]))
            if len(records) > 1 else None,
        "warm_supervisor_p90_seconds": float(np.quantile([r["supervisor_seconds"] for r in records[1:]], .9))
            if len(records) > 1 else None,
        "cycles": records,
    }
    _atomic_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--upper-bound", type=float, default=3.)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args()
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("CPU outside allowed affinity")
    os.sched_setaffinity(0, {args.cpu})
    report = run(args.receipt, args.side, args.output_directory, cycles=args.cycles,
                 upper_bound=args.upper_bound, tolerance=args.tolerance)
    print(json.dumps({key: value for key, value in report.items() if key != "cycles"}, indent=2))


if __name__ == "__main__":
    main()
