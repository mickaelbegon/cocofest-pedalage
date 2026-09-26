"""Explicit boundary for invoking an already prepared cyclic NMPC.

The caller owns callback construction, failure budgets, timing and diagnostics.
This module only dispatches the request to the existing FES NMPC implementation.
It imports neither Bioptim nor the cycling example driver.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, kw_only=True)
class RhoSolveRequest:
    """Arguments of one advancing RHO invocation, without implicit defaults.

    Live callbacks, solvers and cyclic options are passed by identity. Freezing
    the request prevents rebinding its fields; it deliberately does not freeze
    or copy those live objects. ``total_cycles`` retains the backend argument's
    historical meaning and is not a count of certified physical cycles.
    """

    update_functions: Any
    solver: Any
    solver_first_iter: Any
    total_cycles: int
    external_force: Any
    cycle_solutions: Any
    get_all_iterations: bool
    cyclic_options: dict
    max_consecutive_failing: int
    compact_solution_output: bool


def run_rho_solve(nmpc: Any, request: RhoSolveRequest) -> Any:
    """Call the prepared backend once, preserving its return value/exceptions.

    Retries and certification belong to the existing NMPC callbacks. In
    particular, this boundary never catches failures or independently retries
    a solve, which could otherwise advance the physical trajectory twice.
    """
    return nmpc.solve_fes_nmpc(
        request.update_functions,
        solver=request.solver,
        solver_first_iter=request.solver_first_iter,
        total_cycles=request.total_cycles,
        external_force=request.external_force,
        cycle_solutions=request.cycle_solutions,
        get_all_iterations=request.get_all_iterations,
        cyclic_options=request.cyclic_options,
        max_consecutive_failing=request.max_consecutive_failing,
        compact_solution_output=request.compact_solution_output,
    )
