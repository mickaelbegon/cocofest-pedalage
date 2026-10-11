#!/usr/bin/env python3
"""Restore one certified RHO checkpoint and verify the offline front graph only.

This command solves one nominal frozen baseline solely because Bioptim creates
its decision graph lazily at that point.  It advances no RHO cycle and builds
no capacity-frontier IPOPT solver.  It proves that the prepared state can be
restored and bound to the local capacity-frontier auxiliary graph with a
declared capacity box.
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

from cocofest.optimization.capacity_frontier_ocp import CapacityFrontierSupportProblem
from cocofest.optimization.conditional_slow_manifold_adapter import SlowManifoldInitialStateAdapter
from cocofest.simulation.rho_restart_checkpoint import restore_prepared_checkpoint
from scripts.audit_ding_slow_manifold_checkpoint import manifolds_from_model
from scripts.probe_independent_rho_task_reserve import _load_arm
from scripts.run_local_task_reserve_branch import _build_runtime, _solve


def verify(receipt: Path, *, side: str, cpu: int, lower_ratio: float = .2,
           upper_ratio: float = 1., manifold_tolerance: float = 5e-8,
           solve_support: bool = False, validate_independent: bool = False,
           audit_baseline_dop853: bool = False, support=None, support_seed: str = "source_center",
           epsilon_upper=None, fixed_capacity=None, max_iterations: int = 200) -> dict:
    receipt = Path(receipt).resolve(strict=True)
    source, payload = _load_arm(receipt, side)
    started = perf_counter()
    runtime = _build_runtime(payload, side, cpu=cpu)
    runtime_seconds = perf_counter() - started
    program = runtime["nmpc"]
    restored = restore_prepared_checkpoint(Path(source.archive_path), program,
                                           completed_cycles=source.completed_cycles)
    if not restored["stimulation_history_complete"]:
        raise ValueError("Checkpoint stimulation history is incomplete")
    baseline_started = perf_counter()
    baseline = _solve(program, runtime["solver"])
    baseline_seconds = perf_counter() - baseline_started
    if program.ocp_solver is None:
        raise RuntimeError("Nominal source solve did not construct the OCP graph")
    manifolds = manifolds_from_model(Path(source.model_path))
    muscles = tuple(manifolds)
    if support is None:
        support = np.ones(len(muscles))
    support = np.asarray(support, float)
    if support.shape != (len(muscles),) or not np.all(np.isfinite(support)) or np.any(support < 0) or not support.sum() > 0:
        raise ValueError("support must supply one non-negative non-zero value per muscle")
    support = support / support.sum()
    adapter = SlowManifoldInitialStateAdapter.from_program(
        checkpoint=source, program=program, muscle_order=muscles, manifolds=manifolds,
        manifold_tolerance=manifold_tolerance)
    problem = CapacityFrontierSupportProblem(graph=program.ocp_solver.nlp, adapter=adapter,
                                             support=support, epsilon_upper=epsilon_upper,
                                             fixed_capacity=fixed_capacity)
    limits = problem.limits_for_domain(lower=[lower_ratio] * len(muscles),
                                       upper=[upper_ratio] * len(muscles))
    seed_values = {"source_center": np.asarray(adapter.center, float),
                   "domain_midpoint": np.full(len(muscles), (lower_ratio + upper_ratio) / 2.)}
    if support_seed not in seed_values:
        raise ValueError(f"Unknown support seed: {support_seed}")
    if support_seed != "source_center":
        seed_limits = adapter.bounds_for(seed_values[support_seed])
        for indices in adapter.indices:
            limits["x0"][list(indices)] = seed_limits["x0"][list(indices)]
    report = {"schema_version": 1, "receipt": str(receipt),
            "receipt_sha256": sha256(receipt.read_bytes()).hexdigest(), "side": side, "cpu": cpu,
            "restored": restored, "condition_sha256": adapter.condition_sha256,
            "muscle_order": list(muscles), "center_capacity_ratio": list(adapter.center),
            "support": support.tolist(),
            "support_seed": support_seed,
            "epsilon_upper": [None if not np.isfinite(value) else float(value)
                              for value in problem.epsilon_upper],
            "fixed_capacity": [None if not np.isfinite(value) else float(value)
                               for value in problem.fixed_capacity],
            "capacity_domain": [lower_ratio, upper_ratio],
            "baseline_solver_status": str(getattr(baseline, "status", None)),
            "timings_seconds": {"runtime_construction": runtime_seconds, "baseline_solve": baseline_seconds},
            "auxiliary_variables": int(problem.graph["x"].numel()),
            "original_constraints": problem.original_constraint_count,
            "constraints_with_slow_manifold": int(problem.graph["g"].numel()),
            "bounds_dimension": int(limits["lbx"].size),
            "status": "graph_verified_no_capacity_frontier_solver_built_no_cycle_advanced"}
    if audit_baseline_dop853:
        from bioptim import SolutionMerge
        from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver
        baseline_replay_started = perf_counter()
        baseline_replay = driver.high_accuracy_trace_rollout_diagnostics(
            program,
            baseline.decision_states(to_merge=SolutionMerge.NODES),
            baseline.decision_controls(to_merge=SolutionMerge.NODES),
            cycle_count=1,
            capacity_scales={f"A_{name}": manifold.a_rest for name, manifold in manifolds.items()},
            restart_at_cycle_boundaries=True,
        )
        work = json.loads(source.task_context_json)["nominal_work_j"]
        baseline_work = float(baseline_replay["per_cycle_reference_work_j"][-1])
        report["baseline_DOP853"] = {
            "replay": baseline_replay,
            "seconds": perf_counter() - baseline_replay_started,
            "cycle_work_j": baseline_work,
            "relative_work_error": abs(baseline_work - work) / max(1., abs(work)),
            "reported_not_automatically_accepted": True,
        }
    if validate_independent and not solve_support:
        raise ValueError("--validate-independent requires --solve-support")
    if solve_support:
        import casadi as ca
        from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver
        args = runtime["args"]
        options = driver.configure_ipopt_solver(max_iterations=max_iterations, linear_solver="ma57",
            tolerance=float(args.nlp_tolerance), hsl_library=args.ipopt_hsl_library,
            print_level=0, advanced_options=driver._ipopt_advanced_options(args)).as_dict(program.ocp_solver)
        support_build_started = perf_counter()
        solver = ca.nlpsol("capacity_frontier_uniform_support", "ipopt", problem.graph, options)
        support_build_seconds = perf_counter() - support_build_started
        support_solve_started = perf_counter()
        result = solver(**limits)
        support_solve_seconds = perf_counter() - support_solve_started
        report["support_smoke"] = {**problem.audit(np.asarray(result["x"], float).reshape(-1)),
                                   "solver_status": solver.stats().get("return_status"),
                                   "iterations": solver.stats().get("iter_count"),
                                   "construction_seconds": support_build_seconds,
                                   "solve_seconds": support_solve_seconds}
        report["status"] = "auxiliary_support_solve_completed_not_independently_certified"
        if validate_independent:
            from bioptim import Solution, SolutionMerge
            from cocofest.optimization.task_reserve_probe_adapter import independent_full_nlp_audit
            candidate = np.asarray(result["x"], float).reshape(-1)
            fresh_started = perf_counter()
            fresh_runtime = _build_runtime(payload, side, cpu=cpu)
            fresh = fresh_runtime["nmpc"]
            restore_prepared_checkpoint(Path(source.archive_path), fresh,
                                        completed_cycles=source.completed_cycles)
            _solve(fresh, fresh_runtime["solver"])
            fresh_adapter = SlowManifoldInitialStateAdapter.from_program(
                checkpoint=source, program=fresh, muscle_order=muscles,
                manifolds=manifolds, manifold_tolerance=manifold_tolerance)
            fixed = fresh_adapter.bounds_for(report["support_smoke"]["initial_capacity_ratio"])
            fresh_options = driver.configure_ipopt_solver(max_iterations=max_iterations, linear_solver="ma57",
                tolerance=float(fresh_runtime["args"].nlp_tolerance),
                hsl_library=fresh_runtime["args"].ipopt_hsl_library, print_level=0,
                advanced_options=driver._ipopt_advanced_options(fresh_runtime["args"])).as_dict(fresh.ocp_solver)
            independent_solver = ca.nlpsol("capacity_frontier_fresh_feasibility", "ipopt",
                                           fresh.ocp_solver.nlp, fresh_options)
            fixed["x0"] = candidate.copy()
            fresh.ocp_solver.limits = {key: value.copy() for key, value in fixed.items()}
            independent_started = perf_counter()
            independent = independent_solver(**fixed)
            independent_seconds = perf_counter() - independent_started
            independent_solution = Solution.from_vector(fresh, np.asarray(independent["x"], float))
            work = json.loads(source.task_context_json)["nominal_work_j"]
            discrete_audit = independent_full_nlp_audit(independent_solution, fresh,
                                                        target_work_j=work, tolerance=1e-6)
            replay_started = perf_counter()
            try:
                replay = driver.high_accuracy_trace_rollout_diagnostics(
                    fresh,
                    independent_solution.decision_states(to_merge=SolutionMerge.NODES),
                    independent_solution.decision_controls(to_merge=SolutionMerge.NODES),
                    cycle_count=1,
                    capacity_scales={f"A_{name}": manifold.a_rest
                                     for name, manifold in manifolds.items()},
                    restart_at_cycle_boundaries=True,
                )
                replay_status = "complete"
            except (RuntimeError, ValueError, KeyError, TypeError) as error:
                replay = {"reason": f"{type(error).__name__}: {error}"}
                replay_status = "failed"
            replay_work = None
            replay_work_relative_error = None
            if replay_status == "complete" and replay.get("per_cycle_reference_work_j"):
                replay_work = float(replay["per_cycle_reference_work_j"][-1])
                replay_work_relative_error = abs(replay_work - work) / max(1., abs(work))
            report["independent_rebuild"] = {
                "fresh_condition_sha256": fresh_adapter.condition_sha256,
                "solver_status": independent_solver.stats().get("return_status"),
                "iterations": independent_solver.stats().get("iter_count"),
                "solve_seconds": independent_seconds,
                "total_fresh_runtime_seconds": perf_counter() - fresh_started,
                "discrete_audit_passed": discrete_audit.passed,
                "discrete_audit": dict(discrete_audit.detail),
                "continuous_DOP853_replay": replay_status,
                "DOP853": replay,
                "DOP853_seconds": perf_counter() - replay_started,
                "DOP853_cycle_work_j": replay_work,
                "DOP853_relative_work_error": replay_work_relative_error,
                "DOP853_is_reported_not_automatically_accepted": True,
                "certified_feasibility": False,
            }
            report["status"] = (
                "fresh_discrete_rebuild_and_DOP853_completed_not_global_front_certified"
                if replay_status == "complete"
                else "fresh_discrete_rebuild_completed_DOP853_failed_not_global_front_certified"
            )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--lower-ratio", type=float, default=.2)
    parser.add_argument("--upper-ratio", type=float, default=1.)
    parser.add_argument("--manifold-tolerance", type=float, default=5e-8)
    parser.add_argument("--solve-support", action="store_true")
    parser.add_argument("--validate-independent", action="store_true")
    parser.add_argument("--audit-baseline-dop853", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--support", type=float, nargs="+",
                        help="One non-negative support weight per printed muscle")
    parser.add_argument("--support-seed", choices=("source_center", "domain_midpoint"),
                        default="source_center")
    parser.add_argument("--epsilon-upper", type=float, nargs="+",
                        help="Capacity upper bounds; use nan for an unconstrained muscle")
    parser.add_argument("--fixed-capacity", type=float, nargs="+",
                        help="Fixed capacity ratios; use nan for an optimized muscle")
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    output = args.pop("output").resolve()
    report = verify(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
