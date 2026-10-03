#!/usr/bin/env python3
"""One-shot CPU-isolated load supervisor from a certified bilateral receipt.

This pilot emits honest load witnesses and KKT diagnostics. Its direction is
deliberately inactive pending independent reoptimized gradient/holdout checks.
"""
from copy import copy
from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import sys
from time import monotonic, perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.task_load_margin import TaskLoadMarginRequest, TaskLoadMarginResult
from cocofest.optimization.task_load_margin_oracle import (
    configure_load_objective, audit_load_solution, reoptimized_central_gradient, initial_bound_envelope_gradient,
)
from cocofest.optimization.task_reserve import TaskReserveCheckpoint, TaskReserveProbe, ProbeEvidence
from cocofest.optimization.task_reserve_coordinates import (
    ding_a_capacity_layout, capacity_coordinate_context, extract_prepared_capacity_coordinates,
)
from cocofest.optimization.task_reserve_probe_adapter import probe_one_cycle, independent_full_nlp_audit
from cocofest.simulation.independent_arms_process import _atomic_json, _driver_arguments, _configured_payload_model
from cocofest.simulation.rho_restart_checkpoint import restore_prepared_checkpoint
from scripts.probe_independent_rho_task_reserve import _load_arm


def run(receipt, side, output_directory, *, upper_bound=3., tolerance=1e-6, finite_difference_step=None):
    from bioptim import SolutionMerge
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver
    from cocofest.optimization.configured_cycling_model import configured_model_factories

    source, payload = _load_arm(receipt, side)
    model = json.loads(Path(source.model_path).read_text())
    layout = ding_a_capacity_layout(Path(source.model_path), muscle_names=tuple(model["muscles"]))
    context = {**json.loads(source.task_context_json), **capacity_coordinate_context(layout)}
    source = TaskReserveCheckpoint.from_archive(Path(source.archive_path), completed_cycles=source.completed_cycles,
        model_path=Path(source.model_path), task_context=context)
    center = extract_prepared_capacity_coordinates(source, layout=layout).values
    request = TaskLoadMarginRequest(f"{side}-c{source.completed_cycles}-{source.archive_sha256[:12]}",
        source, tuple(c.state_key for c in layout), center, (.01,) * len(center), (0.,) * len(center),
        (1.,) * len(center), monotonic(), upper_bound)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "request.json", asdict(request))

    def build_runtime():
        args = copy(_driver_arguments(payload, side))
        args.nlp_ipopt_recovery = args.nlp_ipopt_recovery_ma57_tuned = False
        configured = _configured_payload_model(payload, side)
        with configured_model_factories(configured[1], []):
            return driver.build_unilateral_runtime(args, echo=False)

    def solve(program, solver):
        return super(driver.RecedingHorizonOptimization, program).solve(solver=solver, warm_start=None)

    def save(solution, program, probe, audit):
        path = output / f"witness-{probe.work_scale:.12g}.npz"
        if path.exists():
            path = output / f"witness-load-{probe.work_scale:.12g}.npz"
        if path.exists():
            raise FileExistsError(path)
        temp = path.with_suffix(".partial.npz")
        states = solution.decision_states(to_merge=SolutionMerge.NODES)
        controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
        np.savez_compressed(temp, vector=np.asarray(solution.vector, float),
            lam_g=np.asarray(solution.lam_g, float), lam_x=np.asarray(solution.lam_x, float),
            **{f"states__{k}": np.asarray(v, float) for k, v in states.items()},
            **{f"controls__{k}": np.asarray(v, float) for k, v in controls.items()},
            metadata__json=np.asarray(json.dumps({"request": asdict(probe), "audit": asdict(audit)}, allow_nan=False)))
        os.replace(temp, path)
        return str(path)

    nominal = probe_one_cycle(TaskReserveProbe(source, 1.), build_runtime=build_runtime,
        solve_one_cycle=solve,
        audit=lambda sol, program, target, tol: independent_full_nlp_audit(sol, program, target_work_j=target, tolerance=tol),
        save_witness=save, tolerance=tolerance)
    _atomic_json(output / "nominal.json", asdict(nominal))
    start = perf_counter()
    load_witness, kkt, failure, artifact = None, {}, None, None
    try:
        runtime = build_runtime()
        program = runtime["nmpc"]
        restored = restore_prepared_checkpoint(Path(source.archive_path), program, completed_cycles=source.completed_cycles)
        if restored.get("restored_problem_sha256") != source.prepared_problem_sha256 or restored.get("stimulation_history_complete") is not True:
            raise ValueError("Supervisor restoration differs from source state/history")
        configure_load_objective(program, nominal_work_j=context["nominal_work_j"], upper_bound=upper_bound)
        solution = solve(program, runtime["solver"])
        load, audit, kkt = audit_load_solution(solution, program, nominal_work_j=context["nominal_work_j"],
                                               upper_bound=upper_bound, tolerance=tolerance)
        if audit.passed and load > 0:
            artifact = save(solution, program, TaskReserveProbe(source, load), audit)
        load_witness = ProbeEvidence(load, source.prepared_problem_sha256, source.task_context_sha256,
            source.model_sha256, str(solution.status), audit.maximum_normalized_violation, tolerance,
            bool(audit.passed and artifact), artifact, perf_counter() - start, audit.groups, True,
            None if audit.passed else "complete_discrete_NLP_audit_failed")
        _atomic_json(output / "load-audit.json", {"audit": asdict(audit), "kkt": kkt})
        try:
            envelope = initial_bound_envelope_gradient(solution, program, layout)
            _atomic_json(output / "envelope-gradient.json", envelope)
            from cocofest.optimization.task_reserve_ocp import TaskReserveStateCoordinate
            full_layout = tuple(TaskReserveStateCoordinate(f"{key}_{name}", scale=scale)
                for name, parameters in model["muscles"].items()
                for key, scale in (("Cn", 1.), ("F", float(parameters["Fmax"])),
                                   ("A", float(parameters["a_scale"])), ("Tau1", .05), ("Km", .1)))
            full_envelope = initial_bound_envelope_gradient(solution, program, full_layout)
            _atomic_json(output / "full-state-envelope-gradient.json", {**full_envelope,
                "coordinate_layout": [asdict(item) for item in full_layout],
                "independently_validated": False,
                "note": "Only A-only finite differences have been validated separately; full-state gradient is diagnostic"})
        except (ValueError, RuntimeError, AttributeError) as error:
            _atomic_json(output / "envelope-gradient.json", {"unavailable": f"{type(error).__name__}: {error}"})
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    gradient, method = None, None
    if finite_difference_step is not None and artifact is not None:
        observations = []
        def solve_at(point):
            # Only initial selected slow states move; all other states, bounds,
            # closure and pulse history are restored from the same source.
            if np.any(point < request.domain_lower) or np.any(point > request.domain_upper):
                raise ValueError("Finite difference point leaves physical domain")
            runtime = build_runtime()
            program = runtime["nmpc"]
            restored = restore_prepared_checkpoint(Path(source.archive_path), program, completed_cycles=source.completed_cycles)
            if restored.get("restored_problem_sha256") != source.prepared_problem_sha256 or restored.get("stimulation_history_complete") is not True:
                raise ValueError("Finite difference restoration differs from source")
            for value, coordinate in zip(point, layout):
                physical = value * coordinate.scale + coordinate.offset
                bound = program.nlp[0].x_bounds[coordinate.state_key]
                bound.min[coordinate.index, 0] = bound.max[coordinate.index, 0] = physical
                program.nlp[0].x_init[coordinate.state_key].init[coordinate.index, 0] = physical
            configure_load_objective(program, nominal_work_j=context["nominal_work_j"], upper_bound=upper_bound)
            solution = solve(program, runtime["solver"])
            value, report, diagnostics = audit_load_solution(solution, program,
                nominal_work_j=context["nominal_work_j"], upper_bound=upper_bound, tolerance=tolerance)
            index = len(observations)
            path = output / f"finite-difference-{index}.npz"
            temp = path.with_suffix(".partial.npz")
            np.savez_compressed(temp, vector=np.asarray(solution.vector, float),
                coordinates=np.asarray(point), lam_g=np.asarray(solution.lam_g, float),
                lam_x=np.asarray(solution.lam_x, float))
            os.replace(temp, path)
            observations.append({"coordinates": list(point), "load_factor": value,
                "audit": asdict(report), "kkt": diagnostics, "artifact": str(path)})
            _atomic_json(output / "finite-differences.json", {"observations": observations,
                "source_archive_sha256": source.archive_sha256,
                "coordinate_layout": context["coordinate_layout"]})
            return value if report.passed and value < upper_bound - tolerance else None
        try:
            gradient = tuple(reoptimized_central_gradient(center,
                [finite_difference_step] * len(center), solve_at))
            method = "finite_difference_reoptimized"
        except Exception as error:
            failure = f"Finite difference failure: {type(error).__name__}: {error}"
    result = TaskLoadMarginResult(request.request_id, monotonic(), nominal_witness=nominal,
        load_witness=load_witness, failure_reason=failure, solution_artifact=artifact,
        multipliers_artifact=artifact, gradient=gradient, sensitivity_method=method, **kkt)
    _atomic_json(output / "result.json", {**asdict(result), "direction_enabled": False,
        "activation_blocker": "independent_reoptimized_gradient_and_holdout_validation_missing",
        "interpretation": "local achievable load witness only; no global maximum/failure certificate"})
    return output / "result.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--upper-bound", type=float, default=3.)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--finite-difference-step", type=float,
        help="Optional normalized A/a_scale central differences (2 fresh OCPs per muscle); stays inactive")
    args = parser.parse_args()
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("Requested CPU is outside process affinity")
    os.sched_setaffinity(0, {args.cpu})
    print(run(args.receipt, args.side, args.output_directory, upper_bound=args.upper_bound,
              tolerance=args.tolerance, finite_difference_step=args.finite_difference_step))


if __name__ == "__main__":
    main()
