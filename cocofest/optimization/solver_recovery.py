"""Frozen-window NLP recovery orchestration, independent of solver bindings.

The driver supplies backend and model operations. This module sequences one
recovery solve, its audits and optional primal injection; it never advances RHO.
"""

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
import traceback
from typing import Any, Callable


@dataclass(frozen=True)
class RecoveryOperations:
    """Explicit backend/model boundary for a single prepared recovery window."""

    configure_solver: Callable[[], Any]
    solve: Callable[[Any, Any], Any]
    populate_inf_pr: Callable[[Any, Any], None]
    feasibility_summary: Callable[[Any, float], dict]
    acceptance: Callable[[Any, dict], dict]
    compatibility_summary: Callable[[Any, Any], dict]
    inject_seed: Callable[[Any, Any], Any]


def run_frozen_nlp_recovery(
    recovery_nmpc,
    target_nmpc,
    *,
    operations: RecoveryOperations,
    recovery_solver: str,
    max_iterations: int,
    tolerance: float,
    failed_target_solution,
    target_solver: str,
    mechanical_formulation: str,
    seed_source: str = "prepared_target_rho_primal",
    echo: bool = False,
    clock: Callable[[], float] = perf_counter,
) -> tuple[object | None, dict[str, object]]:
    """Restore a primal using the caller's unchanged acceptance policy.

    Configuration/solve failures produce a diagnostic result. Audit and seed
    injection failures propagate, matching the historical driver's boundary.
    An accepted seed can be provisional; acceptance alone is not permission to
    advance the physical trajectory. The caller owns retry/certification policy.
    """
    if recovery_solver not in {"ipopt", "madnlp"}:
        raise ValueError(
            "Periodic NLP recovery currently supports IPOPT and MadNLP only."
        )

    summary: dict[str, object] = {
        "available": True,
        "solver": recovery_solver,
        "target_solver": target_solver,
        "mechanical_formulation": mechanical_formulation,
        "transcription": "collocation_radau",
        "max_iterations": int(max_iterations),
        "seed_source": seed_source,
        "accepted": False,
        "seed_injected": False,
        "structure": deepcopy(
            getattr(recovery_nmpc, "_cocofest_recovery_structure", None)
        ),
    }
    recovery_start = clock()
    configure_start = clock()
    configure_wall_time_s = None
    solve_start = None
    try:
        solver = operations.configure_solver()
        configure_wall_time_s = clock() - configure_start
        solve_start = clock()
        solution = operations.solve(recovery_nmpc, solver)
        solve_call_wall_time_s = clock() - solve_start
    except Exception as exc:
        summary["timing"] = {
            "configure_solver_wall_time_s": (
                clock() - configure_start
                if configure_wall_time_s is None
                else configure_wall_time_s
            ),
            "solve_call_wall_time_s": (
                None if solve_start is None else clock() - solve_start
            ),
            "total_wall_time_s": clock() - recovery_start,
        }
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        if echo:
            print(
                f"{target_solver}_{recovery_solver}_recovery_error: {summary['error']}"
            )
            print(summary["traceback"], end="")
        return None, summary

    feasibility_start = clock()
    operations.populate_inf_pr(solution, recovery_nmpc)
    feasibility = operations.feasibility_summary(solution, tolerance)
    acceptance = operations.acceptance(solution.status, feasibility)
    feasibility_wall_time_s = clock() - feasibility_start
    accepted = acceptance["accepted"]
    solver_time = getattr(solution, "solver_time_to_optimize", None)
    wall_time = getattr(solution, "real_time_to_optimize", None)
    compatibility_start = clock()
    compatibility = operations.compatibility_summary(failed_target_solution, solution)
    compatibility_wall_time_s = clock() - compatibility_start
    summary.update(
        {
            "status": int(solution.status),
            "solver_time_s": None if solver_time is None else float(solver_time),
            "wall_time_s": None if wall_time is None else float(wall_time),
            "feasibility": feasibility,
            "accepted": accepted,
            "certified_feasibility": acceptance["certified"],
            "provisional": acceptance["provisional"],
            "quality": (
                "converged"
                if acceptance["success"]
                else "feasible_nonconverged" if accepted else "rejected"
            ),
            "compatibility_with_failed_target": compatibility,
        }
    )
    if target_solver == "acados":
        summary["compatibility_with_failed_acados"] = summary[
            "compatibility_with_failed_target"
        ]
    injection_start = clock()
    if accepted:
        operations.inject_seed(target_nmpc, solution)
        summary["seed_injected"] = True
    injection_wall_time_s = clock() - injection_start
    summary["timing"] = {
        "configure_solver_wall_time_s": configure_wall_time_s,
        "solve_call_wall_time_s": solve_call_wall_time_s,
        "feasibility_audit_wall_time_s": feasibility_wall_time_s,
        "compatibility_audit_wall_time_s": compatibility_wall_time_s,
        "seed_injection_wall_time_s": injection_wall_time_s,
        "total_wall_time_s": clock() - recovery_start,
    }
    if echo:
        print(
            f"{target_solver}_{recovery_solver}_recovery: "
            f"status={solution.status} accepted={accepted} "
            f"inf_pr={feasibility['final_inf_pr']} "
            f"quality={summary['quality']} "
            f"solver_time_s={summary['solver_time_s']}"
        )
    return solution, summary
