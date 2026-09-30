"""Explicit weighted recruitment policy using the compact Ding phase map.

This is an offline surrogate, not a certified future RHO solution. Every phase
of every requested cycle is propagated; no future cycles are skipped. Weights
control recruitment effort, not curvature about an already feasible reference.
The all-ones policy is consequently distinct from the old reference-deviation
allocator. Nothing in this module activates or modifies the public RHO.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .compact_muscle_prediction import CompactMusclePredictor


POLICY_NAME = "weighted_normalized_recruitment_v1"
ALLOCATION_OBJECTIVE_RECRUITMENT = "weighted_recruitment_v1"
ALLOCATION_OBJECTIVE_PREDICTED_DING_FATIGUE = "predicted_ding_fatigue_v1"
_ALLOCATION_OBJECTIVES = {
    ALLOCATION_OBJECTIVE_RECRUITMENT,
    ALLOCATION_OBJECTIVE_PREDICTED_DING_FATIGUE,
}


def _weights(weights, count):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (count,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("weights must be a finite strictly positive vector, one per muscle.")
    # Scale before summing to avoid overflow and preserve common-scale invariance.
    scaled = weights / np.max(weights)
    normalized = scaled / np.mean(scaled)
    if np.any(normalized == 0):
        raise ValueError("weight dynamic range cannot be represented numerically.")
    return normalized


@dataclass(frozen=True)
class WeightedRecruitmentAllocation:
    status: str
    normalized_recruitment: np.ndarray | None
    normalized_weights: np.ndarray
    reference_recruitment: np.ndarray
    total_lower_bound: float
    total_upper_bound: float
    signed_margin: float
    equality_residual: float | None
    objective: float | None
    allocation_objective: str = ALLOCATION_OBJECTIVE_RECRUITMENT
    quadratic_hessian_diagonal: np.ndarray | None = None
    quadratic_linear: np.ndarray | None = None


@dataclass(frozen=True)
class AllocationQuadraticObjective:
    """Diagonal quadratic written as ``.5*x'H*x - q'x + constant``.

    The fatigue mode uses the *endpoint* compact phase map
    ``A_plus(x) = A_intercept + A_slope*x`` and a rectangular phase
    quadrature.  This is deliberately an allocator surrogate: it does not
    certify the full Ding ODE or the following RHO solution.
    """

    mode: str
    hessian_diagonal: np.ndarray
    linear: np.ndarray
    constant: float

    def value(self, recruitment):
        recruitment = np.asarray(recruitment, dtype=float)
        return float(.5 * np.dot(self.hessian_diagonal, recruitment**2)
                     - np.dot(self.linear, recruitment) + self.constant)

    def gradient(self, recruitment):
        recruitment = np.asarray(recruitment, dtype=float)
        return self.hessian_diagonal * recruitment - self.linear


def build_allocation_quadratic_objective(
    normalized_weights, reference_recruitment, *, reference_regularization,
    allocation_objective=ALLOCATION_OBJECTIVE_RECRUITMENT,
    capacity_intercept=None, capacity_slope=None, rest_capacity=None,
    phase_duration=1.0,
):
    """Return the strictly positive diagonal QP curvature and gradient data.

    ``weighted_recruitment_v1`` exactly preserves the historical objective.
    ``predicted_ding_fatigue_v1`` minimizes the rectangular phase integral of
    ``sum_i w_i * (1 - A_plus_i(x_i) / A_rest_i)**2`` plus the existing small
    reference regularizer.  The common 1/2 multiplier is immaterial to the
    minimizer and makes the returned Hessian convention explicit.
    """
    normalized = np.asarray(normalized_weights, dtype=float)
    xref = np.asarray(reference_recruitment, dtype=float)
    if (normalized.ndim != 1 or normalized.size == 0 or xref.shape != normalized.shape
            or not np.all(np.isfinite(normalized)) or np.any(normalized <= 0)
            or not np.all(np.isfinite(xref))):
        raise ValueError("normalized weights and reference recruitment must be finite matching vectors.")
    if allocation_objective not in _ALLOCATION_OBJECTIVES:
        raise ValueError(f"Unknown allocation_objective: {allocation_objective!r}")
    if not np.isfinite(reference_regularization) or reference_regularization <= 0:
        raise ValueError("reference_regularization must be finite and strictly positive.")
    if allocation_objective == ALLOCATION_OBJECTIVE_RECRUITMENT:
        return AllocationQuadraticObjective(
            allocation_objective,
            normalized + reference_regularization,
            reference_regularization * xref,
            float(.5 * reference_regularization * np.dot(xref, xref)),
        )
    intercept = np.asarray(capacity_intercept, dtype=float)
    slope = np.asarray(capacity_slope, dtype=float)
    rest = np.asarray(rest_capacity, dtype=float)
    if (intercept.shape != normalized.shape or slope.shape != normalized.shape
            or rest.shape != normalized.shape or not all(np.all(np.isfinite(value)) for value in
                                                         (intercept, slope, rest))
            or np.any(rest <= 0)):
        raise ValueError("fatigue objective requires finite matching capacity intercept, slope, and positive rest.")
    if not np.isfinite(phase_duration) or phase_duration <= 0:
        raise ValueError("phase_duration must be finite and strictly positive.")
    residual_intercept = 1. - intercept / rest
    residual_slope = -slope / rest
    curvature = phase_duration * normalized * residual_slope**2
    # q is the negative of the conventional linear-gradient coefficient.
    linear = (reference_regularization * xref
              - phase_duration * normalized * residual_intercept * residual_slope)
    return AllocationQuadraticObjective(
        allocation_objective,
        curvature + reference_regularization,
        linear,
        float(.5 * phase_duration * np.dot(normalized, residual_intercept**2)
              + .5 * reference_regularization * np.dot(xref, xref)),
    )


def solve_weighted_recruitment(
    moment_intercept, moment_slope, reference_moments, weights, *,
    reference_regularization=1e-3, moment_tolerance=1e-8,
    allocation_objective=ALLOCATION_OBJECTIVE_RECRUITMENT,
    capacity_intercept=None, capacity_slope=None, rest_capacity=None,
    phase_duration=1.0,
):
    r"""Solve a strictly convex box QP for normalized recruitment ``0 <= x <= 1``.

    ``moment_slope`` is the signed moment gain at maximum allowed recruitment.
    By default, minimize ``0.5 sum(w_normalized*x**2 + epsilon*(x-x_ref)**2)``
    subject to ``sum(moment_intercept + moment_slope*x) == sum(reference_moments)``.
    The opt-in ``predicted_ding_fatigue_v1`` objective instead uses the affine
    compact endpoint prediction for ``A_plus`` and a rectangular phase integral
    of the actual normalized Ding fatigue residual.  Both forms remain strictly
    convex box QPs because the positive reference regularizer is retained.
    ``x_ref`` is clipped per-muscle reference inversion (zero for zero gain).
    Numerical tolerance admits only floating point equality error, not a
    physical tracking band. Negative and zero moment arms are supported.
    """
    intercept, slope, reference = [np.asarray(x, dtype=float) for x in
                                   (moment_intercept, moment_slope, reference_moments)]
    if (intercept.ndim != 1 or intercept.size == 0 or slope.shape != intercept.shape
            or reference.shape != intercept.shape
            or not all(np.all(np.isfinite(x)) for x in (intercept, slope, reference))):
        raise ValueError("moment inputs must be finite nonempty vectors of equal shape.")
    for value, name in ((reference_regularization, "reference_regularization"),
                        (moment_tolerance, "moment_tolerance")):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and strictly positive.")
    normalized = _weights(weights, intercept.size)
    xref = np.zeros_like(slope)
    np.divide(reference - intercept, slope, out=xref, where=slope != 0)
    xref = np.clip(xref, 0., 1.)
    target = float(np.sum(reference))
    lower = float(np.sum(intercept + np.minimum(slope, 0.)))
    upper = float(np.sum(intercept + np.maximum(slope, 0.)))
    margin = min(target - lower, upper - target)

    quadratic = build_allocation_quadratic_objective(
        normalized, xref, reference_regularization=reference_regularization,
        allocation_objective=allocation_objective,
        capacity_intercept=capacity_intercept, capacity_slope=capacity_slope,
        rest_capacity=rest_capacity, phase_duration=phase_duration,
    )

    def result(status, x=None):
        return WeightedRecruitmentAllocation(
            status, x, normalized, xref, lower, upper, margin,
            None if x is None else float(np.sum(intercept + slope * x) - target),
            None if x is None else quadratic.value(x), allocation_objective,
            quadratic.hessian_diagonal, quadratic.linear,
        )

    if target < lower - moment_tolerance:
        return result("infeasible_total_below_bounds")
    if target > upper + moment_tolerance:
        return result("infeasible_total_above_bounds")
    diagonal = quadratic.hessian_diagonal
    linear = quadratic.linear
    active = slope != 0
    x = np.clip(linear / diagonal, 0., 1.)
    if not np.any(active):
        return result("ok", x)
    # At reachable endpoints the equality fixes all nonzero-gain coordinates.
    if target <= lower:
        x[active] = (slope[active] < 0).astype(float)
    elif target >= upper:
        x[active] = (slope[active] > 0).astype(float)
    else:
        # Normalize equality coefficients so dual root tolerances do not depend
        # on moment units. x(lambda)=clip((linear+lambda*a)/diagonal,0,1).
        scale = float(np.max(np.abs(slope)))
        a = slope / scale
        rhs = (target - float(np.sum(intercept))) / scale
        crossings = np.r_[-linear[active] / a[active],
                          (diagonal[active] - linear[active]) / a[active]]

        def candidate(multiplier):
            return np.clip((linear + multiplier * a) / diagonal, 0., 1.)

        multiplier = brentq(
            lambda value: float(np.dot(a, candidate(value)) - rhs),
            float(np.min(crossings)), float(np.max(crossings)),
            xtol=1e-14, rtol=4 * np.finfo(float).eps,
        )
        x = candidate(multiplier)
    if abs(np.sum(intercept + slope * x) - target) > moment_tolerance:
        return result("moment_allocation_residual")
    return result("ok", x)


@dataclass(frozen=True)
class WeightedCycleRolloutResult:
    """Arrays retain requested shapes with NaN tails after the accepted prefix.

    PW, muscle moments and weighted_force_integrals: (cycles,muscles,phases).
    Scalar moment diagnostics: (cycles,phases); state_history:
    (cycles*phases+1,muscles,5), in Cn/F/A/Tau1/Km order. Signed margins
    measure distance from the original signed target to the reachable total
    moment interval and include the first failed phase when available.
    Integrals are exponential fatigue convolutions, not unweighted force dose.
    Metadata contains only JSON-compatible scalars/lists/dictionaries.
    """

    status: str
    requested_cycles: int
    completed_cycles: int
    completed_intervals: int
    completed_duration: float
    pulse_widths: np.ndarray
    allocated_moments: np.ndarray
    achieved_moments: np.ndarray
    original_total_moments: np.ndarray
    signed_moment_errors: np.ndarray
    signed_margins: np.ndarray
    state_history: np.ndarray
    weighted_force_integrals: np.ndarray
    cycle_weighted_force_integrals: np.ndarray
    cycle_fatigue_forcing: np.ndarray
    first_failure: dict | None
    metadata: dict

    @property
    def completed(self):
        return self.status in ("complete", "complete_with_deficit")

    @property
    def full_horizon_normalized_deficit(self):
        return self.metadata.get("full_horizon_normalized_deficit")

    @property
    def terminal_normalized_reserve(self):
        return self.metadata.get("terminal_normalized_reserve")

    @property
    def full_horizon_mean_squared_fatigue(self):
        return self.metadata.get("full_horizon_mean_squared_fatigue")

    @property
    def first_block_mean_squared_fatigue(self):
        return self.metadata.get("first_block_mean_squared_fatigue")

    @property
    def terminal_minimum_capacity(self):
        return self.metadata.get("terminal_minimum_capacity")

    @property
    def minimum_signed_margin(self):
        finite = self.signed_margins[np.isfinite(self.signed_margins)]
        return float(np.min(finite)) if finite.size else float("nan")

    @property
    def phase_weighted_force_integrals(self):
        return self.weighted_force_integrals


def rollout_fatigue_metrics(history, rest_capacity, phase_durations, block_cycles):
    """Unweighted fatigue trapezoids; candidate weights never enter the audit.

    All candidates must cover the same duration. The first block measures the
    period until the next supervisor update; the terminal minimum protects
    the weakest muscle in addition to the across-muscle fatigue average.
    """
    ratios = np.asarray(history)[:, :, 2] / np.asarray(rest_capacity)
    squared_fatigue = np.mean((1. - ratios) ** 2, axis=1)
    durations = np.resize(np.asarray(phase_durations), len(ratios) - 1)
    trapezoids = .5 * (squared_fatigue[:-1] + squared_fatigue[1:])
    block = min(len(trapezoids), int(block_cycles) * len(phase_durations))
    return {
        "full_horizon_mean_squared_fatigue": float(np.average(trapezoids, weights=durations)),
        "first_block_mean_squared_fatigue": float(np.average(trapezoids[:block], weights=durations[:block])),
        "terminal_minimum_capacity": float(np.min(ratios[-1])),
    }


class WeightedCyclePredictor(CompactMusclePredictor):
    """Repeated, fixed-weight allocation with full compact five-state memory.

    A cycle's weighted-force summary composes each phase convolution with its
    remaining slow decay. This gives its actual slow-state affine input along
    the computed trajectory; it is not a state-independent future-cycle map.
    """

    def __init__(self, intervals, parameters, *, substeps=16, reference_regularization=1e-3,
                 allocation_objective=ALLOCATION_OBJECTIVE_RECRUITMENT):
        super().__init__(intervals, parameters, substeps=substeps)
        if not np.isfinite(reference_regularization) or reference_regularization <= 0:
            raise ValueError("reference_regularization must be finite and strictly positive.")
        if allocation_objective not in _ALLOCATION_OBJECTIVES:
            raise ValueError(f"Unknown allocation_objective: {allocation_objective!r}")
        self.reference_regularization = float(reference_regularization)
        self.allocation_objective = allocation_objective
        if np.any(self.maximum_recruitment >= 1.):
            raise ValueError("PW bounds must produce representable recruitment strictly below one.")

    def rollout(self, initial_states, weights, *, horizon_cycles, moment_tolerance=1e-8,
                tracking_mode="exact", score_block_cycles=20):
        """Propagate exact tracking or an explicitly deficit-bearing capacity scenario.

        ``projected_capacity`` continues at the closest reachable total moment
        when the reference cannot be produced. It retains the original demand
        and its deficit. This makes a fixed long-horizon capacity comparison
        possible, but is not a feasible movement beyond the first deficit.
        Numerical or model-domain failures still terminate with unscored tails.
        """
        if tracking_mode not in ("exact", "projected_capacity"):
            raise ValueError("tracking_mode must be exact or projected_capacity.")
        if isinstance(score_block_cycles, bool) or int(score_block_cycles) != score_block_cycles or score_block_cycles < 1:
            raise ValueError("score_block_cycles must be a positive integer.")
        if isinstance(horizon_cycles, bool) or int(horizon_cycles) != horizon_cycles or horizon_cycles < 1:
            raise ValueError("horizon_cycles must be a positive integer.")
        if not np.isfinite(moment_tolerance) or moment_tolerance <= 0:
            raise ValueError("moment_tolerance must be finite and strictly positive.")
        normalized_weights = _weights(weights, self.muscle_count)
        cycles, phases, muscles = int(horizon_cycles), len(self.intervals), self.muscle_count
        current = np.asarray(initial_states, dtype=float).copy()
        if current.shape != (muscles, 5):
            raise ValueError("initial_states must have shape (muscles, 5).")
        shape = (cycles, muscles, phases)
        pws, allocated, achieved, integrals = [np.full(shape, np.nan) for _ in range(4)]
        original = np.tile([sum(p.target_moments) for p in self.intervals], (cycles, 1))
        errors, margins = [np.full((cycles, phases), np.nan) for _ in range(2)]
        history = np.full((cycles * phases + 1, muscles, 5), np.nan)
        cycle_integrals = np.full((cycles, muscles), np.nan)
        completed, duration = 0, 0.
        first_deficit = None
        phase_durations = np.asarray([p.duration for p in self.intervals])
        # One common, reference-only RMS scale: no candidate-dependent
        # denominator, and no singular division at zero-moment phases.
        reference_rms = float(np.sqrt(np.average(original[0]**2, weights=phase_durations)))
        reference_scale = max(reference_rms, moment_tolerance)

        def result(failure=None):
            diagnostics = {}
            if tracking_mode == "projected_capacity":
                diagnostics = {"first_target_deficit": first_deficit,
                               "normalization_reference_rms_nm": reference_rms,
                               "normalization_scale_nm": reference_scale,
                               "score_block_cycles": int(score_block_cycles),
                               "full_horizon_normalized_deficit": None,
                               "terminal_normalized_reserve": None,
                               "cycle_normalized_squared_deficit": [],
                               "block_diagnostics": [],
                               "continuation_is_feasible_movement": False}
                if failure is None:
                    loss = np.average((errors / reference_scale)**2, axis=1, weights=phase_durations)
                    reserve = np.average(margins / reference_scale, axis=1, weights=phase_durations)
                    diagnostics.update({
                        "full_horizon_normalized_deficit": float(np.mean(loss)),
                        "terminal_normalized_reserve": float(np.mean(reserve[-int(score_block_cycles):])),
                        "cycle_normalized_squared_deficit": loss.tolist(),
                        "block_diagnostics": [
                            {"start_cycle": k, "stop_cycle": min(k + int(score_block_cycles), cycles),
                             "mean_squared_deficit": float(np.mean(loss[k:k + int(score_block_cycles)])),
                             "mean_signed_reserve": float(np.mean(reserve[k:k + int(score_block_cycles)]))}
                            for k in range(0, cycles, int(score_block_cycles))],
                    })
                    diagnostics.update(rollout_fatigue_metrics(
                        history, self.rest[:, 0], phase_durations, int(score_block_cycles)))
            return WeightedCycleRolloutResult(
                status=(("complete_with_deficit" if first_deficit else "complete") if failure is None else
                        "numerical_failure" if failure["status"] in (
                            "allocation_numerical_failure", "moment_allocation_residual",
                            "moment_inversion_residual") else "infeasible"),
                requested_cycles=cycles, completed_cycles=completed // phases,
                completed_intervals=completed, completed_duration=duration,
                pulse_widths=pws, allocated_moments=allocated, achieved_moments=achieved,
                original_total_moments=original, signed_moment_errors=errors,
                signed_margins=margins, state_history=history,
                weighted_force_integrals=integrals, cycle_weighted_force_integrals=cycle_integrals,
                cycle_fatigue_forcing=cycle_integrals[:, :, None] * self.alpha[None, :, :],
                first_failure=failure,
                metadata={"policy": POLICY_NAME, "reference_regularization": self.reference_regularization,
                          "normalized_weights": normalized_weights.tolist(),
                          "all_ones_is_old_greedy": False,
                          "force_model": "compact_phase_frozen_slow_states",
                          "moment_constraint_sampling": "phase_endpoints_only",
                          "full_ode_or_rho_feasibility_certified": False,
                          "future_cycles_explicitly_propagated": True,
                          "tracking_mode": ("exact_with_numerical_tolerance" if tracking_mode == "exact"
                                            else "projected_capacity"),
                          "moment_tolerance": float(moment_tolerance),
                          "allocation_objective": self.allocation_objective,
                          "fatigue_objective_approximation": (
                              "endpoint_A_plus_rectangular_phase_quadrature"
                              if self.allocation_objective == ALLOCATION_OBJECTIVE_PREDICTED_DING_FATIGUE
                              else None),
                          "fatigue_objective_full_ding_or_rho_certified": False,
                          "substeps": self.substeps, **diagnostics},
            )

        try:
            self.phase_map(current, 0)
        except ValueError as error:
            return result({"cycle_index": 0, "interval_index": 0,
                           "status": "initial_state_outside_domain", "message": str(error)})
        history[0] = current
        for cycle in range(cycles):
            cycle_integral = np.zeros(muscles)
            for phase, interval in enumerate(self.intervals):
                transition = self.phase_map(current, phase)
                failure = {"cycle_index": cycle, "interval_index": phase}
                # Intersect the PW box with the compact force domain; signed
                # gains can impose a tighter, state-dependent recruitment cap.
                for envelope_state in (transition.intercept,
                                       transition.endpoint(transition.maximum_recruitment)):
                    if (not np.all(np.isfinite(envelope_state)) or np.any(envelope_state[:, :2] < 0.)
                            or np.any(envelope_state[:, 2:] <= 0.)):
                        return result(dict(failure, status="envelope_domain_invalid"))
                coefficients = np.asarray(interval.moment_coefficients)
                moment0 = coefficients * transition.intercept[:, 1]
                slope = coefficients * transition.slope[:, 1] * transition.maximum_recruitment
                try:
                    allocation = solve_weighted_recruitment(
                        moment0, slope, interval.target_moments, normalized_weights,
                        reference_regularization=self.reference_regularization,
                        moment_tolerance=moment_tolerance,
                        allocation_objective=self.allocation_objective,
                        capacity_intercept=transition.intercept[:, 2],
                        capacity_slope=transition.slope[:, 2] * transition.maximum_recruitment,
                        # ``rest`` stores [A_rest, Tau1_rest, Km_rest].  The
                        # fatigue objective is normalized by available force,
                        # never by the resting force-rate constant Km.
                        rest_capacity=self.rest[:, 0], phase_duration=interval.duration,
                    )
                except (ValueError, RuntimeError, FloatingPointError) as error:
                    return result(dict(failure, status="allocation_numerical_failure", message=str(error)))
                margins[cycle, phase] = allocation.signed_margin
                if allocation.normalized_recruitment is None:
                    if tracking_mode != "projected_capacity" or not allocation.status.startswith("infeasible_total_"):
                        return result(dict(failure, status=allocation.status))
                    if first_deficit is None:
                        first_deficit = dict(failure, status=allocation.status,
                                             signed_margin_nm=allocation.signed_margin)
                    # Projection onto a scalar interval is the exact minimum
                    # absolute total-moment deficit over the PW box. At either
                    # endpoint all nonzero-gain recruitment coordinates are
                    # fixed; zero-gain coordinates retain their QP regularizer.
                    x = np.clip(allocation.quadratic_linear / allocation.quadratic_hessian_diagonal, 0., 1.)
                    active = slope != 0
                    upper = original[cycle, phase] > allocation.total_upper_bound
                    x[active] = ((slope[active] > 0) if upper else (slope[active] < 0)).astype(float)
                    recruitment_fraction = x
                else:
                    recruitment_fraction = allocation.normalized_recruitment
                recruitment = recruitment_fraction * transition.maximum_recruitment
                next_states = transition.endpoint(recruitment)
                if (not np.all(np.isfinite(next_states)) or np.any(next_states[:, :2] < 0.)
                        or np.any(next_states[:, 2:] <= 0.)):
                    return result(dict(failure, status="predicted_state_outside_domain"))
                moment = coefficients * next_states[:, 1]
                residual = float(np.sum(moment) - original[cycle, phase])
                if abs(residual) > moment_tolerance and allocation.normalized_recruitment is not None:
                    return result(dict(failure, status="moment_inversion_residual"))
                pws[cycle, :, phase] = np.clip(
                    self.pd0 - self.pdt * np.log1p(-recruitment), self.pd0, self.pulse_width_max)
                allocated[cycle, :, phase] = moment0 + slope * recruitment_fraction
                achieved[cycle, :, phase] = moment
                errors[cycle, phase] = residual
                integral = transition.weighted_force_intercept + transition.weighted_force_slope * recruitment
                integrals[cycle, :, phase] = integral
                cycle_integral = np.exp(-interval.duration / self.tau_fat) * cycle_integral + integral
                current = next_states
                completed += 1
                duration += interval.duration
                history[completed] = current
            cycle_integrals[cycle] = cycle_integral
        return result()
