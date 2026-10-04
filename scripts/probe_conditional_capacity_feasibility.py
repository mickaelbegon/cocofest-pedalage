#!/usr/bin/env python3
"""Offline conditional A-capacity directions with a frozen one-cycle deficit NLP.

No production objective or weight is updated, no physical RHO is advanced, and
positive local deficits remain indeterminate. A certified baseline is required
before exploring synthetic capacity states at the same complete condition.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.conditional_capacity_feasibility import (
    ConditionalCapacitySlice, FrozenTaskDeficit, directional_points, mapping_from_program, summarize_slice,
)
from cocofest.optimization.receding_horizon_initial_guess import snapshot_initial_guess
from cocofest.optimization.task_reserve_probe_adapter import independent_full_nlp_audit
from cocofest.simulation.independent_arms_process import _atomic_json
from cocofest.simulation.rho_restart_checkpoint import (
    _runtime_state, prepared_problem_arrays, prepared_problem_digest, restore_prepared_checkpoint,
)
from scripts.probe_independent_rho_task_reserve import _load_arm
from scripts.run_local_task_reserve_branch import _build_runtime, _solve


def _program_digest(program):
    return prepared_problem_digest(snapshot_initial_guess(program), prepared_problem_arrays(program), _runtime_state(program))


def run(receipt, *, side, output_directory, cpu, muscles, offsets, trust_radius=.05,
        max_iterations=200, tolerance=1e-6):
    import casadi as ca
    from bioptim import Solution, SolutionMerge
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver

    campaign_start = perf_counter()
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("Positive auxiliary iteration budget required")
    if not np.isfinite(tolerance) or not 0 < tolerance <= 1e-5:
        raise ValueError("Audit tolerance must lie in (0,1e-5]")
    receipt = Path(receipt).resolve(strict=True)
    source, payload = _load_arm(receipt, side)
    model = json.loads(Path(source.model_path).read_text())
    names = tuple(model["muscles"])
    if not muscles or len(set(muscles)) != len(muscles) or not set(muscles) <= set(names):
        raise ValueError("Name distinct existing muscles to explore")
    slice_ = ConditionalCapacitySlice.from_checkpoint(source, muscle_order=names, trust_radius=trust_radius)
    directions = {name: tuple(float(other == name) for other in names) for name in muscles}
    points = directional_points(slice_, directions, offsets)
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_directory / "plan.json", {"source_receipt": str(receipt),
        "source_receipt_sha256": sha256(receipt.read_bytes()).hexdigest(), "condition_sha256": slice_.condition_sha256,
        "muscle_order": names, "center": slice_.center, "directions": directions, "offsets": offsets,
        "trust_radius": trust_radius, "max_iterations": max_iterations, "audit_tolerance": tolerance,
        "max_cycle_advances": 0, "cpu": cpu})
    construction_start = perf_counter()
    runtime = _build_runtime(payload, side, cpu=cpu)
    construction_seconds = perf_counter() - construction_start
    program, args = runtime["nmpc"], runtime["args"]
    if driver.parse_objectives(args.objective) != {"fatigue"}:
        raise ValueError("Conditional pilot requires the original fatigue-only objective")
    if args.ipopt_linear_solver != "ma57":
        raise ValueError("This pilot retains the requested IPOPT/MA57 solver")
    restored = restore_prepared_checkpoint(Path(source.archive_path), program, completed_cycles=source.completed_cycles)
    if (restored["restored_problem_sha256"] != source.prepared_problem_sha256
            or restored["stimulation_history_complete"] is not True):
        raise ValueError("Source did not restore exactly")
    weights = np.asarray(program.parameter_init["rho_fatigue_weights"].init, float).reshape(-1)
    if not np.allclose(weights, 1., rtol=0, atol=1e-12):
        raise ValueError("Pilot requires unit fatigue weights")
    frozen_digest = _program_digest(program)
    work = json.loads(source.task_context_json)["nominal_work_j"]
    baseline_start = perf_counter()
    baseline = _solve(program, runtime["solver"])
    baseline_seconds = perf_counter() - baseline_start
    audit = independent_full_nlp_audit(baseline, program, target_work_j=work, tolerance=tolerance)
    if not audit.passed:
        _atomic_json(output_directory / "baseline-undetermined.json", {
            "nominal_task_witnessed": False, "audit": dict(audit.detail),
            "physiological_failure_certified": False, "reason": "source_nominal_solve_failed_audit"})
        raise RuntimeError("Nominal frozen source did not supply an independently audited witness")
    if _program_digest(program) != frozen_digest:
        raise RuntimeError("Baseline solve changed prepared source state/history")
    problem = FrozenTaskDeficit(graph=program.ocp_solver.nlp, limits=program.ocp_solver.limits,
        **mapping_from_program(program, slice_), nominal_work_j=work, center=slice_.center)
    # Independent auxiliary options; physical equations and integrator remain
    # the exact source graph. The source solver and production f are untouched.
    auxiliary_options = driver.configure_ipopt_solver(max_iterations=max_iterations,
        linear_solver="ma57", tolerance=float(args.nlp_tolerance), hsl_library=args.ipopt_hsl_library,
        print_level=0, advanced_options=driver._ipopt_advanced_options(args)).as_dict(program.ocp_solver)
    auxiliary_build_start = perf_counter()
    auxiliary = ca.nlpsol("conditional_work_deficit", "ipopt", problem.auxiliary_graph, auxiliary_options)
    auxiliary_construction_seconds = perf_counter() - auxiliary_build_start
    baseline_vector = np.asarray(baseline.vector, float).reshape(-1)
    records = []
    for point in points:
        start = perf_counter()
        record = {"sample_id": point.sample_id, "capacities": point.capacities,
                  "condition_sha256": slice_.condition_sha256, "solver_status": None,
                  "nominal_task_witnessed": False, "physiological_failure_certified": False,
                  "outcome": "indeterminate", "observed_normalized_work_deficit": None}
        try:
            limits = problem.bounds_for(point.capacities, relax_work=True)
            limits["x0"] = baseline_vector.copy()
            limits["x0"][list(problem.indices)] = np.asarray(point.capacities) / problem.coefficients
            result = auxiliary(**limits)
            vector = np.asarray(result["x"], float).reshape(-1)
            record.update(problem.audit(vector, point.capacities, tolerance=tolerance))
            stats = auxiliary.stats()
            record.update(solver_status=stats.get("return_status"), iterations=stats.get("iter_count"))
            solution = Solution.from_vector(program, vector[:, None])
            states = solution.decision_states(to_merge=SolutionMerge.NODES)
            controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
            path = output_directory / f"candidate-{len(records):03d}.npz"
            np.savez_compressed(path, vector=vector, **{f"states__{k}": np.asarray(v, float) for k, v in states.items()},
                **{f"controls__{k}": np.asarray(v, float) for k, v in controls.items()},
                metadata__json=np.asarray(json.dumps(record, sort_keys=True, allow_nan=False)))
            record.update(candidate_path=str(path), candidate_sha256=sha256(path.read_bytes()).hexdigest())
        except (RuntimeError, ValueError, KeyError, TypeError) as error:
            record.update(nominal_task_witnessed=False, outcome="indeterminate",
                          observed_normalized_work_deficit=None, reason=f"{type(error).__name__}: {error}")
        record["elapsed_seconds"] = perf_counter() - start
        records.append(record)
        source.verify_files()
        if _program_digest(program) != frozen_digest:
            raise RuntimeError("Auxiliary probe changed source physical state, parameters or history")
        current_f = sha256(ca.Function("original_unit_objective", [program.ocp_solver.nlp["x"]],
                                       [program.ocp_solver.nlp["f"]]).serialize().encode()).hexdigest()
        if current_f != problem.production_objective_sha256:
            raise RuntimeError("Auxiliary probe changed the original unit-fatigue objective")
        _atomic_json(output_directory / "progress.json", {"completed_points": len(records), "points": len(points),
                                                          "last_record": record})
    summary = summarize_slice(slice_, points, records)
    summary.update(source_receipt=str(receipt), baseline_audit=dict(audit.detail), restored=restored,
                   production_objective_sha256=problem.production_objective_sha256,
                   constraint_graph_sha256=problem.constraint_graph_sha256,
                   auxiliary_objective="normalized_terminal_work_deficit", solver="ipopt-ma57",
                   source_prepared_problem_sha256_after_probes=_program_digest(program),
                   timings={"source_runtime_construction_seconds": construction_seconds,
                            "baseline_solve_seconds": baseline_seconds,
                            "auxiliary_construction_seconds": auxiliary_construction_seconds,
                            "total_wall_seconds": perf_counter() - campaign_start})
    destination = output_directory / "conditional-capacity-report.json"
    _atomic_json(destination, summary)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--muscles", nargs="+", required=True)
    parser.add_argument("--offsets", nargs="+", type=float, default=[-.005, .005])
    parser.add_argument("--trust-radius", type=float, default=.05)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    print(run(**vars(parser.parse_args(argv))))


if __name__ == "__main__":
    main()
