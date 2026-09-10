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


def solve_weighted_recruitment(
    moment_intercept, moment_slope, reference_moments, weights, *,
    reference_regularization=1e-3, moment_tolerance=1e-8,
):
    r"""Solve a strictly convex box QP for normalized recruitment ``0 <= x <= 1``.

    ``moment_slope`` is the signed moment gain at maximum allowed recruitment.
    Minimize ``0.5 sum(w_normalized*x**2 + epsilon*(x-x_ref)**2)`` subject to
    ``sum(moment_intercept + moment_slope*x) == sum(reference_moments)``.
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

    def result(status, x=None):
        return WeightedRecruitmentAllocation(
            status, x, normalized, xref, lower, upper, margin,
            None if x is None else float(np.sum(intercept + slope * x) - target),
            None if x is None else float(.5 * np.sum(
                normalized * x**2 + reference_regularization * (x - xref)**2)),
        )

    if target < lower - moment_tolerance:
        return result("infeasible_total_below_bounds")
    if target > upper + moment_tolerance:
        return result("infeasible_total_above_bounds")
    diagonal = normalized + reference_regularization
    linear = reference_regularization * xref
    active = slope != 0
    x = linear / diagonal
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
        return self.status == "complete"

    @property
    def minimum_signed_margin(self):
        finite = self.signed_margins[np.isfinite(self.signed_margins)]
        return float(np.min(finite)) if finite.size else float("nan")

    @property
    def phase_weighted_force_integrals(self):
        return self.weighted_force_integrals


class WeightedCyclePredictor(CompactMusclePredictor):
    """Repeated, fixed-weight allocation with full compact five-state memory.

    A cycle's weighted-force summary composes each phase convolution with its
    remaining slow decay. This gives its actual slow-state affine input along
    the computed trajectory; it is not a state-independent future-cycle map.
    """

    def __init__(self, intervals, parameters, *, substeps=16, reference_regularization=1e-3):
        super().__init__(intervals, parameters, substeps=substeps)
        if not np.isfinite(reference_regularization) or reference_regularization <= 0:
            raise ValueError("reference_regularization must be finite and strictly positive.")
        self.reference_regularization = float(reference_regularization)
        if np.any(self.maximum_recruitment >= 1.):
            raise ValueError("PW bounds must produce representable recruitment strictly below one.")

    def rollout(self, initial_states, weights, *, horizon_cycles, moment_tolerance=1e-8):
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

        def result(failure=None):
            return WeightedCycleRolloutResult(
                status=("complete" if failure is None else
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
                          "tracking_mode": "exact_with_numerical_tolerance",
                          "moment_tolerance": float(moment_tolerance),
                          "substeps": self.substeps},
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
                # The full PW box defines the reported reachability envelope;
                # both endpoints must stay in the compact physical domain.
                for envelope_state in (transition.intercept,
                                       transition.endpoint(self.maximum_recruitment)):
                    if (not np.all(np.isfinite(envelope_state)) or np.any(envelope_state[:, :2] < 0.)
                            or np.any(envelope_state[:, 2:] <= 0.)):
                        return result(dict(failure, status="envelope_domain_invalid"))
                coefficients = np.asarray(interval.moment_coefficients)
                moment0 = coefficients * transition.intercept[:, 1]
                slope = coefficients * transition.slope[:, 1] * self.maximum_recruitment
                try:
                    allocation = solve_weighted_recruitment(
                        moment0, slope, interval.target_moments, normalized_weights,
                        reference_regularization=self.reference_regularization,
                        moment_tolerance=moment_tolerance,
                    )
                except (ValueError, RuntimeError, FloatingPointError) as error:
                    return result(dict(failure, status="allocation_numerical_failure", message=str(error)))
                margins[cycle, phase] = allocation.signed_margin
                if allocation.normalized_recruitment is None:
                    return result(dict(failure, status=allocation.status))
                recruitment = allocation.normalized_recruitment * self.maximum_recruitment
                next_states = transition.endpoint(recruitment)
                if (not np.all(np.isfinite(next_states)) or np.any(next_states[:, :2] < 0.)
                        or np.any(next_states[:, 2:] <= 0.)):
                    return result(dict(failure, status="predicted_state_outside_domain"))
                moment = coefficients * next_states[:, 1]
                residual = float(np.sum(moment) - original[cycle, phase])
                if abs(residual) > moment_tolerance:
                    return result(dict(failure, status="moment_inversion_residual"))
                pws[cycle, :, phase] = np.clip(
                    self.pd0 - self.pdt * np.log1p(-recruitment), self.pd0, self.pulse_width_max)
                allocated[cycle, :, phase] = moment0 + slope * allocation.normalized_recruitment
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
