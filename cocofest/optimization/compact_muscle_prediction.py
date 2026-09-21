r"""Small numerical prediction models for an endurance terminal-cost oracle.

These routines run outside the RHO NLP. A phase map freezes the slow states
only in the force equation, integrates its affine recruitment response, and
updates fatigue with the exponentially weighted force integral. Future PW is
then obtained analytically after a small bounded total-moment allocation.
This is an approximation, not a certificate of future physical feasibility.
"""

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from scipy.special import exprel

from .adaptive_moment_rollout import (
    DingPulseWidthParameters,
    MomentTrackingInterval,
    TotalMomentReferenceRolloutResult,
)
from .ding_fatigue_rollout import DingFatigueParameters
from .smooth_muscle_moment_allocation import solve_bounded_moment_qp_reference


def fatigue_memory_coordinates(slow_state, parameters: DingFatigueParameters):
    """Return damage and two independent initial offsets; no projection to rest.

    d = 1 - A/A_rest, e_j = z_j-z_rest-(alpha_j/alpha_A)(A-A_rest).
    Then d_dot=-d/tau-alpha_A*F/A_rest and e_dot=-e/tau exactly.
    """

    state = np.asarray(slow_state, dtype=float)
    if state.shape != (3,) or not np.all(np.isfinite(state)):
        raise ValueError("slow_state must be a finite (A, Tau1, Km) vector.")
    if parameters.alpha_a >= 0.0:
        raise ValueError("Damage coordinates require alpha_a < 0.")
    delta_a = state[0] - parameters.a_rest
    offsets = state[1:] - parameters.rest_state[1:] - (
        parameters.alpha[1:] / parameters.alpha_a
    ) * delta_a
    return float(-delta_a / parameters.a_rest), offsets


def reconstruct_fatigue_memory(damage, offsets, parameters: DingFatigueParameters):
    """Invert the coordinates, including off-manifold initial conditions."""

    offsets = np.asarray(offsets, dtype=float)
    if offsets.shape != (2,) or not np.all(np.isfinite(offsets)) or not np.isfinite(damage):
        raise ValueError("damage and the two offsets must be finite.")
    if parameters.alpha_a >= 0.0:
        raise ValueError("Damage coordinates require alpha_a < 0.")
    delta_a = -parameters.a_rest * float(damage)
    return parameters.rest_state + np.r_[
        delta_a, parameters.alpha[1:] / parameters.alpha_a * delta_a + offsets
    ]


@dataclass(frozen=True)
class AffineRecruitmentMap:
    """Physical five-state endpoint = intercept + slope * recruitment_fraction."""

    intercept: np.ndarray
    slope: np.ndarray
    weighted_force_intercept: np.ndarray
    weighted_force_slope: np.ndarray
    maximum_recruitment: np.ndarray

    def endpoint(self, recruitment):
        recruitment = np.asarray(recruitment, dtype=float)
        if recruitment.shape != self.maximum_recruitment.shape:
            raise ValueError("recruitment must have one entry per muscle.")
        if (not np.all(np.isfinite(recruitment)) or np.any(recruitment < 0.0)
                or np.any(recruitment > self.maximum_recruitment)):
            raise ValueError("recruitment lies outside its PW-derived bounds.")
        return self.intercept + self.slope * recruitment[:, None]


class CompactMusclePredictor:
    """Pre-sampled mechanics and a vectorized exponential force response.

    Fast force/calcium transients remain in the predictor. A, Tau1 and Km are
    frozen over one stimulation interval for force propagation only; all three
    receive the exact fatigue convolution of that approximate force afterward.
    More substeps refine fast quadrature, not the slow-freezing approximation.
    """

    def __init__(self, intervals: Sequence[MomentTrackingInterval],
                 parameters: Sequence[DingPulseWidthParameters], *, substeps=16):
        if isinstance(substeps, bool) or int(substeps) != substeps or substeps < 1:
            raise ValueError("substeps must be a positive integer.")
        self.substeps = int(substeps)
        self.intervals = tuple(intervals)
        self.parameters = tuple(parameters)
        self.muscle_count = len(self.parameters)
        if not self.muscle_count or not self.intervals or any(
            len(interval.target_moments) != self.muscle_count for interval in self.intervals
        ):
            raise ValueError("Intervals and parameters must have matching nonzero dimensions.")
        self.rest = np.asarray([p.fatigue.rest_state for p in self.parameters])
        self.alpha = np.asarray([p.fatigue.alpha for p in self.parameters])
        for key in ("tauc", "tau2", "pd0", "pdt", "pulse_width_max"):
            setattr(self, key, np.asarray([getattr(p, key) for p in self.parameters]))
        self.tau_fat = np.asarray([p.fatigue.tau_fat for p in self.parameters])
        self.maximum_recruitment = -np.expm1(-(self.pulse_width_max - self.pd0) / self.pdt)
        self.gains = np.empty((len(self.intervals), self.substeps, self.muscle_count))
        for k, interval in enumerate(self.intervals):
            for j in range(self.substeps):
                time = (j + 0.5) * interval.duration / self.substeps
                self.gains[k, j] = [gain(time) if callable(gain) else gain
                                    for gain in interval.mechanical_gains]
        if not np.all(np.isfinite(self.gains)):
            raise ValueError("Sampled mechanical gains must be finite.")

    def phase_map(self, states, interval_index: int) -> AffineRecruitmentMap:
        states = np.asarray(states, dtype=float)
        if states.shape != (self.muscle_count, 5) or not np.all(np.isfinite(states)):
            raise ValueError("states must be finite with shape (muscles, 5).")
        if np.any(states[:, :2] < 0.0) or np.any(states[:, 2:] <= 0.0):
            raise ValueError("The predictor requires nonnegative Cn/F and positive A/Tau1/Km.")
        interval = self.intervals[interval_index]
        dt = interval.duration / self.substeps
        amplitude = np.asarray(interval.calcium_amplitudes)
        if np.any(amplitude < 0.0):
            raise ValueError("Calcium amplitudes must be nonnegative.")
        # Both force and its fatigue convolution are affine in recruitment.
        force0 = states[:, 1].copy()
        force1 = np.zeros(self.muscle_count)
        integral0 = np.zeros(self.muscle_count)
        integral1 = np.zeros(self.muscle_count)
        admissible_recruitment = self.maximum_recruitment.copy()
        slow_decay = np.exp(-dt / self.tau_fat)
        integral_constant = -self.tau_fat * np.expm1(-dt / self.tau_fat)
        for j in range(self.substeps):
            time = (j + 0.5) * dt
            cn = np.exp(-time / self.tauc) * (states[:, 0] + amplitude * time / self.tauc)
            activation = cn / (states[:, 4] + cn)
            relaxation = states[:, 3] + self.tau2 * activation
            rate = self.gains[interval_index, j] / relaxation
            equilibrium1 = states[:, 2] * activation * relaxation
            force_decay = np.exp(-rate * dt)
            # Stable even when the fast and slow decay rates coincide.
            kernel = slow_decay * dt * exprel((1.0 / self.tau_fat - rate) * dt)
            integral0 = slow_decay * integral0 + kernel * force0
            integral1 = (slow_decay * integral1 + kernel * force1
                         + equilibrium1 * (integral_constant - kernel))
            force0 = force_decay * force0
            force1 = force_decay * force1 - np.expm1(-rate * dt) * equilibrium1
            # A signed gain is present in the source OCP, so recruitment can
            # reduce force. Intersect the PW bound with F>=0 at every
            # exponential substep instead of clamping the gain or the state.
            force_bound = np.divide(force0, -force1, out=np.full_like(force0, np.inf),
                                    where=force1 < 0.0)
            admissible_recruitment = np.minimum(admissible_recruitment, force_bound)
        cn_end = np.exp(-interval.duration / self.tauc) * (
            states[:, 0] + amplitude * interval.duration / self.tauc
        )
        total_decay = np.exp(-interval.duration / self.tau_fat)
        slow0 = (self.rest + total_decay[:, None] * (states[:, 2:] - self.rest)
                 + self.alpha * integral0[:, None])
        slow1 = self.alpha * integral1[:, None]
        return AffineRecruitmentMap(
            intercept=np.column_stack((cn_end, force0, slow0)),
            slope=np.column_stack((np.zeros(self.muscle_count), force1, slow1)),
            weighted_force_intercept=integral0,
            weighted_force_slope=integral1,
            maximum_recruitment=admissible_recruitment,
        )

    def rollout(self, initial_states, *, horizon_cycles: int,
                moment_tolerance: float = 1e-8) -> TotalMomentReferenceRolloutResult:
        """Numerical rollout with bounded allocation and analytic PW inversion.

        The tiny piecewise QP remains outside any differentiable objective.
        Its result is usable for value-function sampling or candidate ranking.
        """

        if isinstance(horizon_cycles, bool) or int(horizon_cycles) != horizon_cycles or horizon_cycles < 1:
            raise ValueError("horizon_cycles must be a positive integer.")
        if not np.isfinite(moment_tolerance) or moment_tolerance <= 0:
            raise ValueError("moment_tolerance must be finite and positive.")
        horizon_cycles = int(horizon_cycles)
        count = len(self.intervals)
        pws = np.full((horizon_cycles, self.muscle_count, count), np.nan)
        allocated = np.full_like(pws, np.nan)
        achieved = np.full_like(pws, np.nan)
        history = np.full((horizon_cycles * count + 1, self.muscle_count, 5), np.nan)
        current = np.asarray(initial_states, dtype=float).copy()
        self.phase_map(current, 0)  # Validate before assigning to history.
        history[0] = current
        completed = 0

        def result(failure=None):
            return TotalMomentReferenceRolloutResult(
                status="complete" if failure is None else "infeasible",
                requested_cycles=horizon_cycles, completed_cycles=completed // count,
                completed_intervals=completed, pulse_widths=pws,
                allocated_moments=allocated, achieved_moments=achieved,
                state_history=history, first_failure=failure,
                scalar_function_evaluations=0,
            )

        for cycle in range(horizon_cycles):
            for k, interval in enumerate(self.intervals):
                transition = self.phase_map(current, k)
                coefficients = np.asarray(interval.moment_coefficients)
                moment0 = coefficients * transition.intercept[:, 1]
                moment_slope = coefficients * transition.slope[:, 1]
                moment_max = moment0 + moment_slope * transition.maximum_recruitment
                allocation = solve_bounded_moment_qp_reference(
                    interval.target_moments,
                    required_total_moment=float(np.sum(interval.target_moments)),
                    lower_bounds=np.minimum(moment0, moment_max),
                    upper_bounds=np.maximum(moment0, moment_max),
                    feasibility_tolerance=moment_tolerance,
                )
                if allocation.allocated_moments is None:
                    return result({"cycle_index": cycle, "interval_index": k,
                                   "status": allocation.status, "message": allocation.message,
                                   "required_total_moment": float(np.sum(interval.target_moments)),
                                   "minimum_total_moment": float(np.sum(np.minimum(moment0, moment_max))),
                                   "maximum_total_moment": float(np.sum(np.maximum(moment0, moment_max)))})
                recruitment = np.zeros(self.muscle_count)
                np.divide(allocation.allocated_moments - moment0, moment_slope,
                          out=recruitment, where=np.abs(moment_slope) > 1e-14)
                # The allocation is bounded; only floating-point endpoint roundoff
                # can leave the interval. Reject larger excursions explicitly.
                if np.any(recruitment < -1e-12) or np.any(recruitment > transition.maximum_recruitment + 1e-12):
                    return result({"cycle_index": cycle, "interval_index": k,
                                   "status": "recruitment_inversion_outside_bounds"})
                recruitment = np.clip(recruitment, 0.0, transition.maximum_recruitment)
                next_states = transition.endpoint(recruitment)
                if (not np.all(np.isfinite(next_states)) or np.any(next_states[:, :2] < 0.0)
                        or np.any(next_states[:, 2:] <= 0.0)):
                    return result({"cycle_index": cycle, "interval_index": k,
                                   "status": "predicted_state_outside_domain"})
                predicted = coefficients * next_states[:, 1]
                if abs(np.sum(predicted) - np.sum(interval.target_moments)) > moment_tolerance * 1.01:
                    return result({"cycle_index": cycle, "interval_index": k,
                                   "status": "moment_inversion_residual"})
                pws[cycle, :, k] = np.minimum(
                    self.pulse_width_max, self.pd0 - self.pdt * np.log1p(-recruitment)
                )
                allocated[cycle, :, k] = allocation.allocated_moments
                achieved[cycle, :, k] = predicted
                current = next_states
                completed += 1
                history[completed] = current
        return result()
