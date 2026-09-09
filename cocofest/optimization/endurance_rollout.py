r"""Offline endurance diagnostic under a frozen periodic RHO force policy.

This module is intentionally a diagnostic, not an optimal-control problem.  A
certified RHO supplies a periodic force profile and the quantities needed to
invert Ding's force equation.  The profile is repeated for a fixed number of
cycles, while the slow Ding state ``(A, Tau1, Km)`` is propagated exactly under
the declared continuous Fourier or piecewise collocation force interpolant. No full-horizon trajectory,
new decision variable, pulse-width clipping, or muscle-specific weight enters
the calculation.

The slow state used for a recruitment diagnostic is the state at the midpoint
of the same interval: an exact force-convolution half-step is taken, the
diagnostic is evaluated with ``F`` and ``F_dot`` at that midpoint, then the
remaining exact half-step is taken.  The derivative used by recruitment is the
analytical derivative of the same declared periodic interpolant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from cocofest.optimization.ding_fatigue_rollout import (
    DingFatigueParameters,
    propagate_ding_fatigue_from_force_integral,
)
from cocofest.optimization.periodic_force_profile import (
    PeriodicForcePositivityCertificate,
    PeriodicFourierForceProfile,
)
from cocofest.optimization.periodic_collocation_profile import PeriodicCollocationProfile
from cocofest.optimization.recruitment_margin import (
    RecruitmentMarginResult,
    RecruitmentStatus,
    ding_recruitment_margin,
    marginal_capacity_damage_rate,
)


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a strictly positive integer.")
    return int(value)


def _phase_matrix(values, *, name: str, muscle_count: int, interval_count: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    expected_shape = (muscle_count, interval_count)
    if matrix.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape} (muscles, intervals).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix.copy()


@dataclass(frozen=True)
class DingRolloutMuscleParameters:
    """Patient/model parameters needed for one muscle's recruitment diagnostic.

    The slow-fatigue signs and resting-state domains are validated by
    :class:`DingFatigueParameters`.  The remaining parameters are intentionally
    passed to :func:`ding_recruitment_margin` unchanged, so its explicit status
    classification remains available to the diagnostic.
    """

    fatigue: DingFatigueParameters
    tau2: float
    pd0: float
    pdt: float
    pulse_width_max: float

    def __post_init__(self) -> None:
        if not isinstance(self.fatigue, DingFatigueParameters):
            raise TypeError("fatigue must be a DingFatigueParameters instance.")
        values = (self.tau2, self.pd0, self.pdt, self.pulse_width_max)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("tau2, pd0, pdt, and pulse_width_max must be finite.")


@dataclass(frozen=True)
class PeriodicRecruitmentProfile:
    """Force interpolation and midpoint recruitment inputs for all muscles.

    Arrays are deliberately required in ``(muscles, intervals)`` order.  This
    avoids silent broadcasting across muscles and makes muscle permutation a
    straightforward permutation of every first axis.
    """

    force_profile: PeriodicFourierForceProfile | PeriodicCollocationProfile
    interval_count: int
    cn: np.ndarray
    force_length_relationship: np.ndarray
    force_velocity_relationship: np.ndarray
    passive_force_relationship: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.force_profile, (PeriodicFourierForceProfile, PeriodicCollocationProfile)):
            raise TypeError("force_profile must be a periodic Fourier or collocation profile instance.")
        interval_count = _positive_integer(self.interval_count, name="interval_count")
        muscle_count = self.force_profile.signal_count
        object.__setattr__(
            self,
            "cn",
            _phase_matrix(self.cn, name="cn", muscle_count=muscle_count, interval_count=interval_count),
        )
        object.__setattr__(
            self,
            "force_length_relationship",
            _phase_matrix(
                self.force_length_relationship,
                name="force_length_relationship",
                muscle_count=muscle_count,
                interval_count=interval_count,
            ),
        )
        object.__setattr__(
            self,
            "force_velocity_relationship",
            _phase_matrix(
                self.force_velocity_relationship,
                name="force_velocity_relationship",
                muscle_count=muscle_count,
                interval_count=interval_count,
            ),
        )
        object.__setattr__(
            self,
            "passive_force_relationship",
            _phase_matrix(
                self.passive_force_relationship,
                name="passive_force_relationship",
                muscle_count=muscle_count,
                interval_count=interval_count,
            ),
        )
        object.__setattr__(self, "interval_count", interval_count)

    @property
    def muscle_count(self) -> int:
        """Number of force signals/muscles in the frozen policy."""
        return self.force_profile.signal_count

    @property
    def interval_duration(self) -> float:
        """Duration of every equal-width midpoint rollout interval."""
        return self.force_profile.period / self.interval_count

    @property
    def midpoint_times(self) -> np.ndarray:
        """Midpoint times in the base cycle, ordered by phase interval."""
        return (np.arange(self.interval_count, dtype=float) + 0.5) * self.interval_duration

    def force_and_derivative_at_midpoints(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``F`` and analytical ``F_dot`` in ``(muscles, intervals)`` order."""
        return (
            self.force_profile.evaluate(self.midpoint_times),
            self.force_profile.derivative(self.midpoint_times),
        )


@dataclass(frozen=True)
class EnduranceRolloutLocation:
    """A zero-based muscle/phase/cycle location in a periodic rollout."""

    cycle_index: int
    interval_index: int
    muscle_index: int


@dataclass(frozen=True)
class EnduranceRolloutFailure(EnduranceRolloutLocation):
    """First recruitment inversion that was not feasible under ``PW_max``."""

    status: RecruitmentStatus
    utilization: float
    required_pulse_width: float


@dataclass(frozen=True)
class EnduranceRolloutResult:
    """Audit trail for a fixed-policy endurance rollout.

    ``cycle_boundary_states[c]`` is the state at the beginning of cycle ``c``;
    its final row is therefore the state after the complete horizon.
    ``midpoint_states[c, p, m]`` is the state used for the recruitment
    diagnostic at cycle ``c``, phase interval ``p``, muscle ``m``.
    ``marginal_damage_rate[c, p, m]`` is the separate force-driven Ding term
    ``-alpha_a * F / A_rest`` at that same midpoint; it deliberately excludes
    recovery.
    ``diagnostics[c][p][m]`` preserves every individual status without
    aggregating or clipping it.
    """

    cycle_boundary_times: np.ndarray
    cycle_boundary_states: np.ndarray
    midpoint_states: np.ndarray
    midpoint_forces: np.ndarray
    midpoint_force_derivatives: np.ndarray
    force_positivity: PeriodicForcePositivityCertificate
    marginal_damage_rate: np.ndarray
    utilization: np.ndarray
    diagnostics: tuple[tuple[tuple[RecruitmentMarginResult, ...], ...], ...]
    worst_utilization: float
    worst_utilization_location: EnduranceRolloutLocation | None
    first_failure: EnduranceRolloutFailure | None

    @property
    def feasible(self) -> bool:
        """Whether every evaluated muscle-phase sample has a feasible finite PW."""
        return self.first_failure is None


def _initial_states(values, *, muscle_count: int) -> np.ndarray:
    states = np.asarray(values, dtype=float)
    if states.shape != (muscle_count, 3):
        raise ValueError("initial_slow_states must have shape (muscles, 3), ordered as (A, Tau1, Km).")
    if not np.all(np.isfinite(states)):
        raise ValueError("initial_slow_states must contain only finite values.")
    return states.copy()


def _worst_utilization(utilization: np.ndarray) -> tuple[float, EnduranceRolloutLocation | None]:
    finite = np.isfinite(utilization)
    if not np.any(finite):
        return math.nan, None
    masked = np.where(finite, utilization, -math.inf)
    cycle_index, interval_index, muscle_index = np.unravel_index(np.argmax(masked), utilization.shape)
    return float(masked[cycle_index, interval_index, muscle_index]), EnduranceRolloutLocation(
        cycle_index=int(cycle_index),
        interval_index=int(interval_index),
        muscle_index=int(muscle_index),
    )


def rollout_periodic_ding_endurance(
    profile: PeriodicRecruitmentProfile,
    *,
    initial_slow_states,
    muscles: Sequence[DingRolloutMuscleParameters],
    horizon_cycles: int,
) -> EnduranceRolloutResult:
    """Repeat a fixed RHO policy and report its recruitment/PW endurance margin.

    The function has no FHO input and never solves an optimization problem.  A
    negative force reconstructed anywhere in the continuous period is rejected
    explicitly: using it to propagate the physiological Ding fatigue model
    would be a hidden sign change, and it is not clipped to zero.
    """
    if not isinstance(profile, PeriodicRecruitmentProfile):
        raise TypeError("profile must be a PeriodicRecruitmentProfile instance.")
    horizon_cycles = _positive_integer(horizon_cycles, name="horizon_cycles")
    if len(muscles) != profile.muscle_count:
        raise ValueError("muscles must contain exactly one parameter set per force-profile signal.")
    if not all(isinstance(muscle, DingRolloutMuscleParameters) for muscle in muscles):
        raise TypeError("muscles must contain DingRolloutMuscleParameters instances.")

    current = _initial_states(initial_slow_states, muscle_count=profile.muscle_count)
    forces, force_derivatives = profile.force_and_derivative_at_midpoints()
    force_positivity = profile.force_profile.force_positivity_certificate()
    if not np.all(force_positivity.certified_nonnegative):
        negative = np.flatnonzero(~force_positivity.certified_nonnegative).tolist()
        raise ValueError(
            "force_profile is not certified non-negative over the full period "
            f"for signal(s) {negative}; no clipping is applied."
        )

    cycle_boundaries = np.empty((horizon_cycles + 1, profile.muscle_count, 3), dtype=float)
    midpoint_states = np.empty((horizon_cycles, profile.interval_count, profile.muscle_count, 3), dtype=float)
    utilization = np.full((horizon_cycles, profile.interval_count, profile.muscle_count), np.nan, dtype=float)
    marginal_damage_rate = np.empty((horizon_cycles, profile.interval_count, profile.muscle_count), dtype=float)
    cycle_boundaries[0] = current
    diagnostics: list[list[list[RecruitmentMarginResult]]] = []
    first_failure: EnduranceRolloutFailure | None = None
    half_duration = profile.interval_duration / 2.0

    for cycle_index in range(horizon_cycles):
        cycle_diagnostics: list[list[RecruitmentMarginResult]] = []
        for interval_index in range(profile.interval_count):
            interval_diagnostics: list[RecruitmentMarginResult] = []
            interval_start = cycle_index * profile.force_profile.period + interval_index * profile.interval_duration
            marginal_damage_rate[cycle_index, interval_index] = marginal_capacity_damage_rate(
                forces[:, interval_index],
                np.array([muscle.fatigue.alpha_a for muscle in muscles]),
                np.array([muscle.fatigue.a_rest for muscle in muscles]),
            )
            for muscle_index, muscle in enumerate(muscles):
                midpoint = propagate_ding_fatigue_from_force_integral(
                    current[muscle_index],
                    duration=half_duration,
                    exponentially_weighted_force_integral=profile.force_profile.exponentially_weighted_force_integral(
                        start_time=interval_start,
                        duration=half_duration,
                        time_constant=muscle.fatigue.tau_fat,
                    )[muscle_index],
                    parameters=muscle.fatigue,
                )
                midpoint_states[cycle_index, interval_index, muscle_index] = midpoint
                diagnostic = ding_recruitment_margin(
                    cn=profile.cn[muscle_index, interval_index],
                    force=forces[muscle_index, interval_index],
                    force_derivative=force_derivatives[muscle_index, interval_index],
                    capacity=midpoint[0],
                    tau1=midpoint[1],
                    km=midpoint[2],
                    tau2=muscle.tau2,
                    pd0=muscle.pd0,
                    pdt=muscle.pdt,
                    pulse_width_max=muscle.pulse_width_max,
                    force_length_relationship=profile.force_length_relationship[muscle_index, interval_index],
                    force_velocity_relationship=profile.force_velocity_relationship[muscle_index, interval_index],
                    passive_force_relationship=profile.passive_force_relationship[muscle_index, interval_index],
                )
                interval_diagnostics.append(diagnostic)
                utilization[cycle_index, interval_index, muscle_index] = diagnostic.utilization
                if first_failure is None and not diagnostic.feasible:
                    first_failure = EnduranceRolloutFailure(
                        cycle_index=cycle_index,
                        interval_index=interval_index,
                        muscle_index=muscle_index,
                        status=diagnostic.status,
                        utilization=diagnostic.utilization,
                        required_pulse_width=diagnostic.required_pulse_width,
                    )
                current[muscle_index] = propagate_ding_fatigue_from_force_integral(
                    midpoint,
                    duration=half_duration,
                    exponentially_weighted_force_integral=profile.force_profile.exponentially_weighted_force_integral(
                        start_time=interval_start + half_duration,
                        duration=half_duration,
                        time_constant=muscle.fatigue.tau_fat,
                    )[muscle_index],
                    parameters=muscle.fatigue,
                )
            cycle_diagnostics.append(interval_diagnostics)
        diagnostics.append(cycle_diagnostics)
        cycle_boundaries[cycle_index + 1] = current

    worst_utilization, worst_location = _worst_utilization(utilization)
    return EnduranceRolloutResult(
        cycle_boundary_times=np.arange(horizon_cycles + 1, dtype=float) * profile.force_profile.period,
        cycle_boundary_states=cycle_boundaries,
        midpoint_states=midpoint_states,
        midpoint_forces=forces.copy(),
        midpoint_force_derivatives=force_derivatives.copy(),
        force_positivity=force_positivity,
        marginal_damage_rate=marginal_damage_rate,
        utilization=utilization,
        diagnostics=tuple(tuple(tuple(interval) for interval in cycle) for cycle in diagnostics),
        worst_utilization=worst_utilization,
        worst_utilization_location=worst_location,
        first_failure=first_failure,
    )
