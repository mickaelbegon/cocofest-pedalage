r"""Differentiable per-phase redistribution of a required total muscle moment.

The allocator is the closed-form solution of an equality-constrained diagonal
QP.  Its diagonal flexibility is a smooth function of the upward/downward
reachable moment reserve.  It enforces the total moment exactly and exposes
bound margins instead of clipping them, so an outer NLP can impose the bounds
as ordinary differentiable constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


def _positive(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return value


@dataclass(frozen=True)
class SmoothMomentAllocationOptions:
    """Temperatures use N.m; ``flexibility_floor`` uses squared N.m."""

    slack_temperature: float = 1e-3
    direction_temperature: float = 1e-3
    bound_penalty_temperature: float = 1e-4
    flexibility_floor: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "slack_temperature",
            "direction_temperature",
            "bound_penalty_temperature",
            "flexibility_floor",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name=name))


@dataclass(frozen=True)
class SmoothMomentAllocation:
    allocated_moments: np.ndarray
    allocation_shares: np.ndarray
    lower_margins: np.ndarray
    upper_margins: np.ndarray
    equality_residual: float
    qp_deviation_cost: float
    smooth_bound_penalty: float

    @property
    def feasible(self) -> bool:
        return bool(np.all(self.lower_margins >= 0.0) and np.all(self.upper_margins >= 0.0))


@dataclass(frozen=True)
class BoundedMomentQpResult:
    """Numerical validation oracle; this piecewise solution is not an NLP objective."""

    status: str
    allocated_moments: np.ndarray | None
    equality_residual: float | None
    lower_active: np.ndarray | None
    upper_active: np.ndarray | None
    message: str


@dataclass(frozen=True)
class SmoothMomentAllocationExpressions:
    allocated_moments: Any
    allocation_shares: Any
    lower_margins: Any
    upper_margins: Any
    equality_residual: Any
    qp_deviation_cost: Any
    smooth_bound_penalty: Any


@dataclass(frozen=True)
class MomentAllocationNlpExpressions:
    """Objective and constraints when muscle moments are outer-NLP variables."""

    seed_moments: Any
    objective: Any
    equality_residual: Any
    lower_margins: Any
    upper_margins: Any


def _vectors(
    reference_moments: Sequence[float] | np.ndarray,
    lower_bounds: Sequence[float] | np.ndarray,
    upper_bounds: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.asarray(reference_moments, dtype=float).reshape(-1)
    lower = np.asarray(lower_bounds, dtype=float).reshape(-1)
    upper = np.asarray(upper_bounds, dtype=float).reshape(-1)
    if reference.size < 1 or lower.shape != reference.shape or upper.shape != reference.shape:
        raise ValueError("reference_moments and bounds must be non-empty vectors of equal size.")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("reference_moments and bounds must be finite.")
    if np.any(lower > upper):
        raise ValueError("Every lower moment bound must be less than or equal to its upper bound.")
    return reference, lower, upper


def _softplus_numpy(value: np.ndarray, temperature: float) -> np.ndarray:
    return temperature * np.logaddexp(0.0, value / temperature)


def _directional_flexibility_numpy(
    reference: np.ndarray,
    required_total_moment: float,
    lower: np.ndarray,
    upper: np.ndarray,
    options: SmoothMomentAllocationOptions,
) -> np.ndarray:
    total_correction = required_total_moment - float(np.sum(reference))
    upward_reserve = _softplus_numpy(upper - reference, options.slack_temperature)
    downward_reserve = _softplus_numpy(reference - lower, options.slack_temperature)
    upward_gate = float(expit(total_correction / options.direction_temperature))
    return (
        upward_gate * upward_reserve**2
        + (1.0 - upward_gate) * downward_reserve**2
        + options.flexibility_floor
    )


def allocate_total_moment_smoothly(
    reference_moments: Sequence[float] | np.ndarray,
    *,
    required_total_moment: float,
    lower_bounds: Sequence[float] | np.ndarray,
    upper_bounds: Sequence[float] | np.ndarray,
    options: SmoothMomentAllocationOptions = SmoothMomentAllocationOptions(),
) -> SmoothMomentAllocation:
    """Solve the smooth equality-only QP and report the unsmoothed bound margins."""

    reference, lower, upper = _vectors(reference_moments, lower_bounds, upper_bounds)
    required_total_moment = float(required_total_moment)
    if not math.isfinite(required_total_moment):
        raise ValueError("required_total_moment must be finite.")
    total_correction = required_total_moment - float(np.sum(reference))
    flexibility = _directional_flexibility_numpy(
        reference, required_total_moment, lower, upper, options
    )
    shares = flexibility / np.sum(flexibility)
    allocated = reference + total_correction * shares
    lower_margins = allocated - lower
    upper_margins = upper - allocated
    lower_violation = _softplus_numpy(-lower_margins, options.bound_penalty_temperature)
    upper_violation = _softplus_numpy(-upper_margins, options.bound_penalty_temperature)
    return SmoothMomentAllocation(
        allocated_moments=allocated,
        allocation_shares=shares,
        lower_margins=lower_margins,
        upper_margins=upper_margins,
        equality_residual=float(np.sum(allocated) - required_total_moment),
        qp_deviation_cost=float(0.5 * np.sum((allocated - reference) ** 2 / flexibility)),
        smooth_bound_penalty=float(np.sum(lower_violation**2 + upper_violation**2)),
    )


def solve_bounded_moment_qp_reference(
    reference_moments: Sequence[float] | np.ndarray,
    *,
    required_total_moment: float,
    lower_bounds: Sequence[float] | np.ndarray,
    upper_bounds: Sequence[float] | np.ndarray,
    options: SmoothMomentAllocationOptions = SmoothMomentAllocationOptions(),
    feasibility_tolerance: float = 1e-10,
) -> BoundedMomentQpResult:
    """Solve the small hard-bounded QP for offline scientific validation.

    The active-set switches make this oracle only piecewise differentiable. It
    must not be called from the RHO objective; the outer-NLP expressions above
    represent the same candidate moments with explicit smooth constraints.
    """

    reference, lower, upper = _vectors(reference_moments, lower_bounds, upper_bounds)
    required_total_moment = float(required_total_moment)
    if not math.isfinite(required_total_moment):
        raise ValueError("required_total_moment must be finite.")
    feasibility_tolerance = _positive(
        feasibility_tolerance, name="feasibility_tolerance"
    )
    minimum_total = float(np.sum(lower))
    maximum_total = float(np.sum(upper))
    if required_total_moment < minimum_total - feasibility_tolerance:
        return BoundedMomentQpResult(
            "infeasible_total_below_bounds",
            None,
            None,
            None,
            None,
            "The requested total moment is below the sum of per-muscle lower bounds.",
        )
    if required_total_moment > maximum_total + feasibility_tolerance:
        return BoundedMomentQpResult(
            "infeasible_total_above_bounds",
            None,
            None,
            None,
            None,
            "The requested total moment is above the sum of per-muscle upper bounds.",
        )
    target = min(max(required_total_moment, minimum_total), maximum_total)
    flexibility = _directional_flexibility_numpy(reference, target, lower, upper, options)
    if abs(target - minimum_total) <= feasibility_tolerance:
        allocated = lower.copy()
    elif abs(target - maximum_total) <= feasibility_tolerance:
        allocated = upper.copy()
    else:
        lower_breakpoints = (lower - reference) / flexibility
        upper_breakpoints = (upper - reference) / flexibility
        left = float(np.min(lower_breakpoints) - 1.0)
        right = float(np.max(upper_breakpoints) + 1.0)

        def candidate(multiplier: float) -> np.ndarray:
            return np.clip(reference + multiplier * flexibility, lower, upper)

        multiplier = brentq(
            lambda value: float(np.sum(candidate(value)) - target),
            left,
            right,
            xtol=1e-14,
            rtol=4.0 * np.finfo(float).eps,
        )
        allocated = candidate(multiplier)
    active_tolerance = max(feasibility_tolerance, 1e-12)
    return BoundedMomentQpResult(
        "ok",
        allocated,
        float(np.sum(allocated) - required_total_moment),
        np.abs(allocated - lower) <= active_tolerance,
        np.abs(allocated - upper) <= active_tolerance,
        "The bounded equality-constrained reference QP is feasible.",
    )


def _softplus_casadi(value, temperature: float):
    import casadi as ca

    scaled = value / temperature
    return temperature * ca.vertcat(
        *(ca.logsumexp(ca.vertcat(0.0, scaled[index])) for index in range(scaled.numel()))
    )


def build_smooth_moment_allocation_expressions(
    reference_moments,
    required_total_moment,
    lower_bounds,
    upper_bounds,
    *,
    options: SmoothMomentAllocationOptions = SmoothMomentAllocationOptions(),
) -> SmoothMomentAllocationExpressions:
    """CasADi equivalent of :func:`allocate_total_moment_smoothly`."""

    import casadi as ca

    reference = ca.vec(reference_moments)
    lower = ca.vec(lower_bounds)
    upper = ca.vec(upper_bounds)
    if reference.numel() < 1 or lower.numel() != reference.numel() or upper.numel() != reference.numel():
        raise ValueError("reference_moments and bounds must be non-empty vectors of equal size.")
    total_correction = required_total_moment - ca.sum1(reference)
    upward_reserve = _softplus_casadi(upper - reference, options.slack_temperature)
    downward_reserve = _softplus_casadi(reference - lower, options.slack_temperature)
    upward_gate = 0.5 * (
        ca.tanh(total_correction / (2.0 * options.direction_temperature)) + 1.0
    )
    flexibility = (
        upward_gate * upward_reserve**2
        + (1.0 - upward_gate) * downward_reserve**2
        + options.flexibility_floor
    )
    shares = flexibility / ca.sum1(flexibility)
    allocated = reference + total_correction * shares
    lower_margins = allocated - lower
    upper_margins = upper - allocated
    lower_violation = _softplus_casadi(
        -lower_margins, options.bound_penalty_temperature
    )
    upper_violation = _softplus_casadi(
        -upper_margins, options.bound_penalty_temperature
    )
    return SmoothMomentAllocationExpressions(
        allocated_moments=allocated,
        allocation_shares=shares,
        lower_margins=lower_margins,
        upper_margins=upper_margins,
        equality_residual=ca.sum1(allocated) - required_total_moment,
        qp_deviation_cost=0.5 * ca.sum1((allocated - reference) ** 2 / flexibility),
        smooth_bound_penalty=ca.sum1(lower_violation**2 + upper_violation**2),
    )


def build_moment_allocation_nlp_expressions(
    candidate_moments,
    reference_moments,
    required_total_moment,
    lower_bounds,
    upper_bounds,
    *,
    options: SmoothMomentAllocationOptions = SmoothMomentAllocationOptions(),
    bound_penalty_weight: float = 1.0,
) -> MomentAllocationNlpExpressions:
    """Return a smooth QP cost plus explicit equality and bound constraints.

    ``candidate_moments`` are decision variables owned by the outer NLP.  The
    closed-form allocator is used only as a seed and to construct smooth
    directional weights.  Feasibility is enforced by ``equality_residual=0``,
    ``lower_margins>=0`` and ``upper_margins>=0``; it never depends on the seed.
    """

    import casadi as ca

    bound_penalty_weight = float(bound_penalty_weight)
    if not math.isfinite(bound_penalty_weight) or bound_penalty_weight < 0.0:
        raise ValueError("bound_penalty_weight must be finite and non-negative.")
    candidate = ca.vec(candidate_moments)
    seed = build_smooth_moment_allocation_expressions(
        reference_moments,
        required_total_moment,
        lower_bounds,
        upper_bounds,
        options=options,
    )
    reference = ca.vec(reference_moments)
    lower = ca.vec(lower_bounds)
    upper = ca.vec(upper_bounds)
    if candidate.numel() != reference.numel():
        raise ValueError("candidate_moments must match the reference muscle count.")
    lower_margins = candidate - lower
    upper_margins = upper - candidate
    lower_violation = _softplus_casadi(
        -lower_margins, options.bound_penalty_temperature
    )
    upper_violation = _softplus_casadi(
        -upper_margins, options.bound_penalty_temperature
    )
    # Shares are strictly positive and sum to one. They therefore define a
    # well-conditioned diagonal QP without any muscle-specific hand tuning.
    deviation = candidate - reference
    qp_cost = 0.5 * ca.sum1(deviation**2 / seed.allocation_shares)
    bound_penalty = ca.sum1(lower_violation**2 + upper_violation**2)
    return MomentAllocationNlpExpressions(
        seed_moments=seed.allocated_moments,
        objective=qp_cost + bound_penalty_weight * bound_penalty,
        equality_residual=ca.sum1(candidate) - required_total_moment,
        lower_margins=lower_margins,
        upper_margins=upper_margins,
    )


def build_moment_allocation_nlp_function(
    muscle_count: int,
    *,
    options: SmoothMomentAllocationOptions = SmoothMomentAllocationOptions(),
    bound_penalty_weight: float = 1.0,
    symbolic_type: str = "SX",
):
    """Build the per-phase subgraph to insert into an outer CasADi NLP."""

    import casadi as ca

    if isinstance(muscle_count, bool) or int(muscle_count) != muscle_count or muscle_count < 1:
        raise ValueError("muscle_count must be a positive integer.")
    if symbolic_type not in {"SX", "MX"}:
        raise ValueError("symbolic_type must be 'SX' or 'MX'.")
    symbol = getattr(ca, symbolic_type).sym
    candidate = symbol("candidate_moments", int(muscle_count))
    reference = symbol("reference_moments", int(muscle_count))
    required = symbol("required_total_moment")
    lower = symbol("lower_moment_bounds", int(muscle_count))
    upper = symbol("upper_moment_bounds", int(muscle_count))
    expressions = build_moment_allocation_nlp_expressions(
        candidate,
        reference,
        required,
        lower,
        upper,
        options=options,
        bound_penalty_weight=bound_penalty_weight,
    )
    return ca.Function(
        "muscle_moment_allocation_nlp_terms",
        [candidate, reference, required, lower, upper],
        [
            expressions.seed_moments,
            expressions.objective,
            expressions.equality_residual,
            expressions.lower_margins,
            expressions.upper_margins,
        ],
        ["candidate", "reference", "required_total", "lower", "upper"],
        ["seed", "objective", "equality_residual", "lower_margins", "upper_margins"],
    )


def build_smooth_moment_allocation_function(
    muscle_count: int,
    *,
    options: SmoothMomentAllocationOptions = SmoothMomentAllocationOptions(),
    symbolic_type: str = "SX",
):
    """Build one reusable and differentiable CasADi allocation function."""

    import casadi as ca

    if isinstance(muscle_count, bool) or int(muscle_count) != muscle_count or muscle_count < 1:
        raise ValueError("muscle_count must be a positive integer.")
    if symbolic_type not in {"SX", "MX"}:
        raise ValueError("symbolic_type must be 'SX' or 'MX'.")
    symbol = getattr(ca, symbolic_type).sym
    reference = symbol("reference_moments", int(muscle_count))
    required = symbol("required_total_moment")
    lower = symbol("lower_moment_bounds", int(muscle_count))
    upper = symbol("upper_moment_bounds", int(muscle_count))
    expressions = build_smooth_moment_allocation_expressions(
        reference,
        required,
        lower,
        upper,
        options=options,
    )
    return ca.Function(
        "smooth_muscle_moment_allocation",
        [reference, required, lower, upper],
        [
            expressions.allocated_moments,
            expressions.allocation_shares,
            expressions.lower_margins,
            expressions.upper_margins,
            expressions.equality_residual,
            expressions.qp_deviation_cost,
            expressions.smooth_bound_penalty,
        ],
        ["reference", "required_total", "lower", "upper"],
        [
            "allocated",
            "shares",
            "lower_margins",
            "upper_margins",
            "equality_residual",
            "qp_deviation_cost",
            "smooth_bound_penalty",
        ],
    )
