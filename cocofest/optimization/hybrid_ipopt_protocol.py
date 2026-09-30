"""Decision rules and provenance for an IPOPT L-BFGS -> exact hand-off.

The limited-memory phase is deliberately a *warm-start generator*.  Its
iterates, including an iterate which is finite but infeasible, are never
reported as FHO results.  Only a subsequent exact-Hessian IPOPT solve may
produce a certified result.

The driver runs IPOPT in small, explicit iteration blocks.  This avoids
depending on IPOPT's human-readable stdout and gives every decision a JSON
audit trail.  A block boundary is also a durable warm-start checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class HybridStoppingPolicy:
    """Conservative criterion for switching L-BFGS to exact Hessians."""

    block_iterations: int = 25
    relative_objective_tolerance: float = 1e-6
    audited_constraint_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if not 20 <= self.block_iterations <= 30:
            raise ValueError("block_iterations must be between 20 and 30.")
        if not 0 < self.relative_objective_tolerance < 1:
            raise ValueError("relative_objective_tolerance must lie in (0, 1).")
        if not 0 < self.audited_constraint_tolerance:
            raise ValueError("audited_constraint_tolerance must be positive.")


@dataclass(frozen=True)
class LbfgsBlockObservation:
    """A finite L-BFGS checkpoint, read from structured solver artifacts."""

    block: int
    cumulative_iterations: int
    objective: float
    audited_constraint_violation: float
    checkpoint_path: str
    native_status: str | None = None

    def __post_init__(self) -> None:
        if self.block < 1 or self.cumulative_iterations < 1:
            raise ValueError("block and cumulative_iterations must be positive.")
        if not isfinite(self.objective):
            raise ValueError("objective must be finite.")
        if not isfinite(self.audited_constraint_violation) or self.audited_constraint_violation < 0:
            raise ValueError("audited_constraint_violation must be finite and non-negative.")
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path is required.")


def relative_objective_change(previous: float, current: float) -> float:
    """Symmetric-scale relative change, well-defined for objectives near zero."""

    if not isfinite(previous) or not isfinite(current):
        raise ValueError("objectives must be finite.")
    return abs(current - previous) / max(1.0, abs(previous), abs(current))


def decide_lbfgs_handoff(
    history: Sequence[LbfgsBlockObservation], policy: HybridStoppingPolicy,
) -> dict[str, Any]:
    """Return a JSON-safe, fail-closed switch decision.

    The two endpoint objectives are exactly one block apart.  Therefore a
    20--30 iteration block implements the requested objective test without
    parsing a formatted IPOPT iteration table.
    """

    if not history:
        return {"switch_to_exact": False, "reason": "no_finite_checkpoint"}
    current = history[-1]
    result: dict[str, Any] = {
        "switch_to_exact": False,
        "reason": "insufficient_objective_history",
        "checkpoint": asdict(current),
        "policy": asdict(policy),
    }
    if len(history) < 2:
        return result
    previous = history[-2]
    iteration_span = current.cumulative_iterations - previous.cumulative_iterations
    result["objective_window_iterations"] = iteration_span
    if not 20 <= iteration_span <= 30:
        result["reason"] = "objective_window_not_20_to_30_iterations"
        return result
    objective_change = relative_objective_change(previous.objective, current.objective)
    result["relative_objective_change"] = objective_change
    result["previous_objective"] = previous.objective
    result["objective_stable"] = objective_change < policy.relative_objective_tolerance
    result["audited_constraint_violation"] = current.audited_constraint_violation
    result["constraint_reasonable"] = (
        current.audited_constraint_violation < policy.audited_constraint_tolerance
    )
    if not result["objective_stable"]:
        result["reason"] = "objective_not_stable"
    elif not result["constraint_reasonable"]:
        result["reason"] = "audited_constraint_violation_too_large"
    else:
        result["switch_to_exact"] = True
        result["reason"] = "objective_stable_and_constraints_reasonable"
    return result


def extract_ipopt_block_observation(
    payload: Mapping[str, Any], *, checkpoint_path: str, block: int,
    cumulative_iterations: int,
) -> LbfgsBlockObservation:
    """Read the endpoint objective and independent audit from result JSON.

    ``audit`` must be produced by the FHO solver/exporter, not inferred from
    IPOPT's ``inf_pr``.  The latter is preserved only as diagnostics because it
    can disagree with the physical constraint audit.
    """

    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], Mapping):
        raise ValueError("Expected one solver result in benchmark JSON.")
    result = results[0]
    solver_stats = result.get("nlp_solver_stats")
    if not isinstance(solver_stats, list) or len(solver_stats) != 1:
        raise ValueError("Missing single-shot IPOPT solver statistics.")
    stats = solver_stats[0]
    diagnostics = stats.get("iteration_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("Missing IPOPT iteration diagnostics.")
    objective = (diagnostics.get("obj") or {}).get("final")
    audit = result.get("uncertified_output_audit") or result.get("single_shot_output_audit")
    # ``build_single_shot_summary`` records the independent feasibility audit
    # on its sole window.  A finite limited-memory output is intentionally not
    # eligible for the normal exported-solution audit, so that top-level field
    # is absent.  The window audit is nevertheless the same recomputed
    # constraint/bound check (not IPOPT's formatted ``inf_pr``) and is the
    # appropriate diagnostic for deciding whether to keep iterating L-BFGS.
    # It remains a hand-off diagnostic only: this function never certifies a
    # limited-memory solution.
    if not isinstance(audit, Mapping):
        windows = result.get("windows")
        if isinstance(windows, list) and len(windows) == 1 and isinstance(windows[0], Mapping):
            candidate = windows[0].get("feasibility")
            if isinstance(candidate, Mapping):
                audit = candidate
    if not isinstance(audit, Mapping):
        raise ValueError("Missing independent output audit; refusing warm-start hand-off.")
    violation = audit.get("effective_primal_infeasibility")
    try:
        objective = float(objective)
        violation = float(violation)
    except (TypeError, ValueError) as error:
        raise ValueError("Non-finite objective or independent constraint audit.") from error
    return LbfgsBlockObservation(
        block=block,
        cumulative_iterations=cumulative_iterations,
        objective=objective,
        audited_constraint_violation=violation,
        checkpoint_path=checkpoint_path,
        native_status=(None if stats.get("return_status") is None else str(stats["return_status"])),
    )
