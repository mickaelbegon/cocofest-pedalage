#!/usr/bin/env python3
"""Reoptimize independent local load-margin checks from immutable source files.

This offline validation never refreshes the oracle timestamps. Passing checks
does not enable a stale result or justify changes in unselected states/history.
"""
from copy import copy
from dataclasses import asdict, replace
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.task_load_margin import TaskLoadMarginPolicy, TaskLoadMarginRequest, TaskLoadMarginResult
from cocofest.optimization.task_load_margin_oracle import configure_load_objective, audit_load_solution, validate_reoptimized_direction
from cocofest.optimization.task_reserve import TaskReserveCheckpoint
from cocofest.optimization.task_reserve_coordinates import ding_a_capacity_layout
from cocofest.optimization.task_reserve_ocp import TaskReserveStateCoordinate
from cocofest.simulation.independent_arms_process import _atomic_json, _driver_arguments, _configured_payload_model
from cocofest.simulation.rho_restart_checkpoint import restore_prepared_checkpoint
from scripts.probe_independent_rho_task_reserve import _load_arm


def run(source_directory, receipt, side, output_directory, *, step=.0005, holdout_radius=.005,
        full_state=False, trust_radius=.01, reuse_gradient_validation=None, periodic_tube=False):
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver
    from cocofest.optimization.configured_cycling_model import configured_model_factories
    source_directory, output = Path(source_directory).resolve(), Path(output_directory).resolve()
    raw_request = json.loads((source_directory / "request.json").read_text())
    raw_result = json.loads((source_directory / "result.json").read_text())
    checkpoint = TaskReserveCheckpoint(**raw_request.pop("checkpoint"))
    request = TaskLoadMarginRequest(checkpoint=checkpoint, **raw_request)
    checkpoint.verify_files()
    actual_source, payload = _load_arm(Path(receipt), side)
    if actual_source.prepared_problem_sha256 != checkpoint.prepared_problem_sha256:
        raise ValueError("Receipt differs from requested complete prepared source")
    model = json.loads(Path(checkpoint.model_path).read_text())
    layout = ding_a_capacity_layout(Path(checkpoint.model_path), muscle_names=tuple(model["muscles"]))
    context = json.loads(checkpoint.task_context_json)
    if context.get("coordinate_layout") != [asdict(item) for item in layout]:
        raise ValueError("Coordinate layout differs from the explicitly supported A normalization")
    if full_state:
        envelope = json.loads((source_directory / "full-state-envelope-gradient.json").read_text())
        layout = tuple(TaskReserveStateCoordinate(**item) for item in envelope["coordinate_layout"])
        center = np.asarray(envelope["normalized_coordinates"])
        domain_upper = tuple(1. if item.state_key.startswith("A_") else max(10., value*2.)
                             for item, value in zip(layout, center))
        radii = np.minimum(trust_radius, .9*center)
        if periodic_tube:
            radii = np.minimum(radii, np.asarray([.001 if c.state_key.startswith("Cn_")
                else .01 if c.state_key.startswith(("A_","Km_")) else trust_radius for c in layout]))
        request = replace(request, center=tuple(center), coordinate_names=tuple(item.state_key for item in layout),
            domain_lower=(0.,)*len(layout), domain_upper=domain_upper,
            trust_radius=tuple(radii))
        raw_result["gradient"] = envelope["gradient"]
    if raw_result.get("gradient") is None:
        raise ValueError("Source has no measured gradient")
    if step <= 0 or holdout_radius <= 0 or trust_radius <= 0 or (not full_state and holdout_radius > min(request.trust_radius)):
        raise ValueError("Positive validation steps within trust region required")
    # Reject overlap with any source training/FD point instead of treating a
    # replay of the same finite difference as an independent gradient check.
    source_points = ([] if full_state else [np.asarray(item["coordinates"]) for item in
        json.loads((source_directory / "finite-differences.json").read_text())["observations"]])
    output.mkdir(parents=True, exist_ok=False)
    observations, started = [], perf_counter()
    cached_gradient_points = []
    if reuse_gradient_validation is not None:
        reused_path=Path(reuse_gradient_validation)
        reused=json.loads(reused_path.read_text())
        if (reused.get("validation_passed") is not True
                or reused.get("source_result") != str(source_directory/"result.json")
                or reused.get("coordinate_layout") != [asdict(item) for item in layout]
                or reused.get("center") != list(request.center)
                or reused.get("gradient") != raw_result["gradient"]
                or reused.get("independent_step") != step):
            raise ValueError("Cached derivative proof belongs to another source, layout, gradient or step")
        cached_gradient_points=json.loads(reused_path.with_name("observations.json").read_text())["observations"][:2*len(layout)]
        if len(cached_gradient_points)!=2*len(layout) or not all(p["passed"] and Path(p["artifact"]).is_file() for p in cached_gradient_points):
            raise ValueError("Cached gradient lacks complete independent trajectory evidence")
    policy = TaskLoadMarginPolicy()
    args = copy(_driver_arguments(payload, side))
    args.nlp_ipopt_recovery = args.nlp_ipopt_recovery_ma57_tuned = False
    configured = _configured_payload_model(payload, side)
    with configured_model_factories(configured[1], []):
        runtime = driver.build_unilateral_runtime(args, echo=False)
    program = runtime["nmpc"]
    restore_prepared_checkpoint(Path(checkpoint.archive_path), program, completed_cycles=checkpoint.completed_cycles)
    configure_load_objective(program, nominal_work_j=context["nominal_work_j"],
                             upper_bound=request.load_factor_upper_bound)

    def solve_at(point):
        if (np.any(point < request.domain_lower) or np.any(point > request.domain_upper)
                or np.any(np.abs(point - request.center) > np.asarray(request.trust_radius)+1e-12)
                or any(np.allclose(point, previous, atol=1e-12, rtol=0) for previous in source_points)):
            raise ValueError("Independent point outside domain/trust or reused training point")
        checkpoint.verify_files()
        cached=next((item for item in cached_gradient_points if np.array_equal(point,np.asarray(item["coordinates"]))),None)
        if cached is not None:
            observations.append({**cached,"reused_immutable_gradient_evidence":str(Path(reuse_gradient_validation).resolve())})
            return cached["load_factor"]
        restored = restore_prepared_checkpoint(Path(checkpoint.archive_path), program,
                                               completed_cycles=checkpoint.completed_cycles)
        if (restored.get("restored_problem_sha256") != checkpoint.prepared_problem_sha256
                or restored.get("stimulation_history_complete") is not True):
            raise ValueError("Source state/history restoration mismatch")
        for value, coordinate in zip(point, layout):
            physical = value * coordinate.scale + coordinate.offset
            bound = program.nlp[0].x_bounds[coordinate.state_key]
            bound.min[coordinate.index, 0] = bound.max[coordinate.index, 0] = physical
            program.nlp[0].x_init[coordinate.state_key].init[coordinate.index, 0] = physical
        # The objective graph stays fixed. Restore all source data first, then
        # relax only the work equality back to the same declared load bounds.
        program.nlp[0].x_bounds["E_prod"].min[0, 2] = 0.
        program.nlp[0].x_bounds["E_prod"].max[0, 2] = context["nominal_work_j"]*request.load_factor_upper_bound
        solution = super(driver.RecedingHorizonOptimization, program).solve(
            solver=runtime["solver"], warm_start=None)
        value, audit, kkt = audit_load_solution(solution, program,
            nominal_work_j=context["nominal_work_j"], upper_bound=request.load_factor_upper_bound,
            tolerance=policy.feasibility_tolerance)
        checkpoint.verify_files()
        path = output / f"independent-{len(observations)}.npz"
        np.savez_compressed(path, vector=np.asarray(solution.vector, float), coordinates=point,
                            lam_g=np.asarray(solution.lam_g, float), lam_x=np.asarray(solution.lam_x, float))
        passed = (audit.passed and value < request.load_factor_upper_bound - policy.feasibility_tolerance
                  and all(v is not None and np.isfinite(v) and 0 <= v <= policy.kkt_tolerance for v in kkt.values()))
        observations.append({"coordinates": point.tolist(), "load_factor": value,
            "audit": asdict(audit), "kkt": kkt, "passed": passed, "artifact": str(path)})
        _atomic_json(output / "observations.json", {"observations": observations,
            "source_directory": str(source_directory), "source_archive_sha256": checkpoint.archive_sha256})
        return value if passed else None

    # Every axis appears with both signs, with simultaneous perturbations;
    # these eight corners are separate from the axis-only derivative probes.
    n = len(layout)
    offsets = []
    for index in range(min(n, 4)):
        delta = np.minimum(holdout_radius,request.trust_radius)*np.ones(n)
        if full_state:
            delta[np.arange(n) % 4 == index] *= -1
        else:
            delta[index] *= -1
        offsets.extend((delta, -delta))
    report, failure = None, None
    try:
        report = validate_reoptimized_direction(request.center, raw_result["gradient"],
            raw_result["load_witness"]["work_scale"], np.full(n, step), offsets, solve_at)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    passed = bool(report is not None and report["gradient_validation_max_abs_error"] <= policy.gradient_error_tolerance
                  and report["local_validation_max_abs_error"] <= policy.local_error_tolerance)
    evidence = {"validation_passed": passed, "failure_reason": failure, "diagnostics": report,
        "source_request": str(source_directory / "request.json"), "source_result": str(source_directory / "result.json"),
        "elapsed_seconds": perf_counter()-started, "coordinates": "full Ding state" if full_state else "A/a_scale only",
        "coordinate_layout": [asdict(item) for item in layout],
        "center": request.center, "gradient": raw_result["gradient"], "trust_radius": request.trust_radius,
        "scope": "local conditional on unselected mechanics and stimulation history",
        "online_activation_authorized": False,
        "activation_blockers": ["original_request_age_must_be_rechecked", "unselected_state_and_history_domain_unvalidated"],
        "independent_step": step, "holdout_radius": holdout_radius,
        "reused_gradient_validation":str(reuse_gradient_validation) if reuse_gradient_validation is not None else None,
        "source_times_not_refreshed": True}
    _atomic_json(output / "validation.json", evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--step", type=float, default=.0005)
    parser.add_argument("--holdout-radius", type=float, default=.005)
    parser.add_argument("--full-state", action="store_true", help="Validate the 20-state KKT envelope using 40 FD OCPs and eight holdouts")
    parser.add_argument("--trust-radius", type=float, default=.01)
    parser.add_argument("--reuse-gradient-validation", type=Path)
    parser.add_argument("--periodic-tube", action="store_true",help="Keep Cn trust at .001 and A/Km at .01; enlarge only force and Tau1")
    args = parser.parse_args()
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("CPU is outside allowed affinity")
    os.sched_setaffinity(0, {args.cpu})
    print(json.dumps(run(args.source_directory, args.receipt, args.side, args.output_directory,
                          step=args.step, holdout_radius=args.holdout_radius, full_state=args.full_state,
                          trust_radius=args.trust_radius,reuse_gradient_validation=args.reuse_gradient_validation,
                          periodic_tube=args.periodic_tube), indent=2))


if __name__ == "__main__":
    main()
