r"""Five-cycle Ding viability preview driven by one optimized RHO cycle.

The preview deliberately does not solve another horizon-wide OCP.  One
optimized cycle supplies a phase-dependent muscle-moment reference and a
baseline pulse-width profile.  For each future phase, a small bounded
allocation redistributes the required *total* moment between muscles, then
independent scalar inversions recover the pulse widths that realize that
allocation under the current full five-state Ding state.

This is a compact implicit feedback policy: the future horizon adds no pulse
width decision variables.  The complete ``(Cn, F, A, Tau1, Km)`` state is
rolled forward, so fatigue affects every later inversion.  A fixed-PW replay
is evaluated from the same initial state to make the benefit (or failure) of
the adaptive policy directly measurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from .adaptive_moment_rollout import (
    AdaptiveMomentRolloutResult,
    DingPulseWidthParameters,
    FixedPulseWidthRolloutResult,
    MomentTrackingInterval,
    TotalMomentReferenceRolloutResult,
    rollout_adaptive_moment_policy,
    rollout_bounded_total_moment_reference_policy,
    rollout_fixed_pulse_width_policy,
)
from .smooth_muscle_moment_allocation import SmoothMomentAllocationOptions


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _positive(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _completed_totals(moment_values: np.ndarray) -> np.ndarray:
    """Sum the muscle axis without turning incomplete phases into zero moment."""

    values = np.asarray(moment_values, dtype=float)
    if values.ndim != 3:
        raise ValueError("moment_values must have shape (cycles, muscles, intervals).")
    complete = np.all(np.isfinite(values), axis=1)
    totals = np.sum(np.where(np.isfinite(values), values, 0.0), axis=1)
    totals[~complete] = np.nan
    return totals


def _cache_callable_mechanical_gains(
    intervals: Sequence[MomentTrackingInterval],
) -> tuple[MomentTrackingInterval, ...]:
    """Share exact gain evaluations across inversion candidates and policies.

    A reduced-mechanics gain depends only on interval-local time, yet the
    scalar PW inversion used to evaluate the same callable hundreds of times.
    ``lru_cache`` retains the value for the exact float passed by RK4; it does
    not interpolate, resample, or change signed-gain handling.
    """

    cached_intervals = []
    for interval in intervals:
        gains = tuple(
            lru_cache(maxsize=None)(gain) if callable(gain) else gain
            for gain in interval.mechanical_gains
        )
        cached_intervals.append(
            MomentTrackingInterval(
                duration=interval.duration,
                calcium_amplitudes=interval.calcium_amplitudes,
                mechanical_gains=gains,
                moment_coefficients=interval.moment_coefficients,
                target_moments=interval.target_moments,
            )
        )
    return tuple(cached_intervals)


@dataclass(frozen=True)
class FiveCycleViabilityConfig:
    """Numerical settings for the muscle-only future preview.

    Five cycles is the intended online viability horizon.  ``horizon_cycles``
    remains configurable so short unit tests and horizon sensitivity studies
    use exactly the same implementation.
    """

    horizon_cycles: int = 5
    integration_substeps: int = 8
    moment_tolerance: float = 1e-8
    state_roundoff_tolerance: float = 1e-14
    allocation_options: SmoothMomentAllocationOptions = SmoothMomentAllocationOptions()
    evaluate_fixed_pulse_width_baseline: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "horizon_cycles",
            _positive_integer(self.horizon_cycles, name="horizon_cycles"),
        )
        object.__setattr__(
            self,
            "integration_substeps",
            _positive_integer(self.integration_substeps, name="integration_substeps"),
        )
        object.__setattr__(
            self,
            "moment_tolerance",
            _positive(self.moment_tolerance, name="moment_tolerance"),
        )
        object.__setattr__(
            self,
            "state_roundoff_tolerance",
            _positive(self.state_roundoff_tolerance, name="state_roundoff_tolerance"),
        )
        if not isinstance(self.allocation_options, SmoothMomentAllocationOptions):
            raise TypeError("allocation_options must be SmoothMomentAllocationOptions.")
        if not isinstance(self.evaluate_fixed_pulse_width_baseline, bool):
            raise TypeError("evaluate_fixed_pulse_width_baseline must be a bool.")


@dataclass(frozen=True)
class CompactMomentViabilityPolicy:
    """One-cycle reference used as an implicit future PW feedback policy.

    ``source_pulse_widths`` are not optimized again.  They define the natural
    fixed-policy comparator, while ``intervals[*].target_moments`` define the
    compact reference followed by the adaptive preview.
    """

    intervals: tuple[MomentTrackingInterval, ...]
    parameters: tuple[DingPulseWidthParameters, ...]
    source_pulse_widths: np.ndarray
    muscle_names: tuple[str, ...] = ()
    source_bound_tolerance_s: float = 1e-10
    source_bound_projection_count: int = field(init=False)
    source_bound_projection_maximum_absolute_s: float = field(init=False)

    def __post_init__(self) -> None:
        intervals = tuple(self.intervals)
        parameters = tuple(self.parameters)
        if not intervals or not parameters:
            raise ValueError("intervals and parameters must be non-empty.")
        muscle_count = len(parameters)
        if any(not isinstance(item, MomentTrackingInterval) for item in intervals):
            raise TypeError("intervals must contain MomentTrackingInterval values.")
        if any(not isinstance(item, DingPulseWidthParameters) for item in parameters):
            raise TypeError("parameters must contain DingPulseWidthParameters values.")
        if any(len(item.target_moments) != muscle_count for item in intervals):
            raise ValueError("Every interval must contain one target moment per muscle.")
        pulse_widths = np.asarray(self.source_pulse_widths, dtype=float)
        expected = (muscle_count, len(intervals))
        if pulse_widths.shape != expected or not np.all(np.isfinite(pulse_widths)):
            raise ValueError(f"source_pulse_widths must be finite with shape {expected}.")
        lower = np.asarray([item.pd0 for item in parameters])[:, None]
        upper = np.asarray([item.pulse_width_max for item in parameters])[:, None]
        tolerance = _positive(
            self.source_bound_tolerance_s, name="source_bound_tolerance_s"
        )
        clipped = np.clip(pulse_widths, lower, upper)
        projection = clipped - pulse_widths
        maximum_projection = float(np.max(np.abs(projection)))
        if maximum_projection > tolerance:
            raise ValueError(
                "source_pulse_widths violate a Ding PW bound beyond the declared "
                f"tolerance ({maximum_projection} > {tolerance} s)."
            )
        names = tuple(str(name) for name in self.muscle_names)
        if not names:
            names = tuple(f"muscle_{index}" for index in range(muscle_count))
        if len(names) != muscle_count or len(set(names)) != muscle_count:
            raise ValueError("muscle_names must contain one unique name per muscle.")
        pulse_widths = clipped.copy()
        pulse_widths.setflags(write=False)
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "source_pulse_widths", pulse_widths)
        object.__setattr__(self, "muscle_names", names)
        object.__setattr__(self, "source_bound_tolerance_s", tolerance)
        object.__setattr__(
            self, "source_bound_projection_count", int(np.count_nonzero(projection))
        )
        object.__setattr__(
            self,
            "source_bound_projection_maximum_absolute_s",
            maximum_projection,
        )

    @classmethod
    def from_rho_policy(cls, policy: Any) -> "CompactMomentViabilityPolicy":
        """Build from :class:`RhoAdaptiveMomentPolicy` without a hard import cycle."""

        required = ("intervals", "parameters", "source_pulse_widths", "muscle_names")
        missing = [name for name in required if not hasattr(policy, name)]
        if missing:
            raise TypeError(f"policy is missing required RHO fields: {missing}.")
        return cls(
            intervals=tuple(policy.intervals),
            parameters=tuple(policy.parameters),
            source_pulse_widths=np.asarray(policy.source_pulse_widths),
            muscle_names=tuple(policy.muscle_names),
        )

    @property
    def muscle_count(self) -> int:
        return len(self.parameters)

    @property
    def interval_count(self) -> int:
        return len(self.intervals)

    @property
    def reference_parameter_count(self) -> int:
        """Number of one-cycle muscle-moment values retained by the policy."""

        return self.muscle_count * self.interval_count

    @property
    def target_total_moments(self) -> np.ndarray:
        return np.asarray(
            [sum(interval.target_moments) for interval in self.intervals],
            dtype=float,
        )


@dataclass(frozen=True)
class MechanicalGainAudit:
    """Domain audit for the positive mechanical gain assumed by Ding propagation."""

    minimum_gain: float
    nonpositive_count: int
    nonfinite_count: int
    evaluated_count: int
    first_invalid_location: tuple[int, int, int] | None

    @property
    def valid(self) -> bool:
        return self.nonpositive_count == 0 and self.nonfinite_count == 0

    def summary(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "minimum_gain": self.minimum_gain,
            "nonpositive_count": self.nonpositive_count,
            "nonfinite_count": self.nonfinite_count,
            "evaluated_count": self.evaluated_count,
            "first_invalid_location": (
                None
                if self.first_invalid_location is None
                else list(self.first_invalid_location)
            ),
            "location_order": ["interval", "muscle", "stage"],
        }


class MechanicalGainDomainError(ValueError):
    """Raised when a source profile violates the declared positive-gain model."""

    def __init__(self, audit: MechanicalGainAudit):
        self.audit = audit
        super().__init__(
            "The five-state Ding preview requires finite positive mechanical "
            f"gains; observed minimum={audit.minimum_gain}, "
            f"nonpositive_count={audit.nonpositive_count}, "
            f"nonfinite_count={audit.nonfinite_count}. No clipping was applied."
        )


def audit_policy_mechanical_gains(
    policy: CompactMomentViabilityPolicy,
    *,
    integration_substeps: int,
) -> MechanicalGainAudit:
    """Sample exactly the RK4 gain stages used by the future propagator."""

    if not isinstance(policy, CompactMomentViabilityPolicy):
        raise TypeError("policy must be CompactMomentViabilityPolicy.")
    integration_substeps = _positive_integer(
        integration_substeps, name="integration_substeps"
    )
    minimum = math.inf
    nonpositive_count = 0
    nonfinite_count = 0
    evaluated_count = 0
    first_invalid = None
    for interval_index, interval in enumerate(policy.intervals):
        step = interval.duration / integration_substeps
        # These are the unique RK4 stages over all substeps.
        stage_times = np.arange(2 * integration_substeps + 1, dtype=float) * step / 2.0
        for muscle_index, gain in enumerate(interval.mechanical_gains):
            for stage_index, local_time in enumerate(stage_times):
                value = float(gain(float(local_time)) if callable(gain) else gain)
                evaluated_count += 1
                if math.isfinite(value):
                    minimum = min(minimum, value)
                    invalid = value <= 0.0
                    nonpositive_count += int(invalid)
                else:
                    invalid = True
                    nonfinite_count += 1
                if invalid and first_invalid is None:
                    first_invalid = (interval_index, muscle_index, stage_index)
    return MechanicalGainAudit(
        minimum_gain=float(minimum),
        nonpositive_count=nonpositive_count,
        nonfinite_count=nonfinite_count,
        evaluated_count=evaluated_count,
        first_invalid_location=first_invalid,
    )


@dataclass(frozen=True)
class FiveCycleViabilityResult:
    """Three same-state future policies evaluated with the full Ding model."""

    policy: CompactMomentViabilityPolicy
    config: FiveCycleViabilityConfig
    individual_moment: AdaptiveMomentRolloutResult
    total_moment: TotalMomentReferenceRolloutResult
    fixed: FixedPulseWidthRolloutResult | None
    fixed_failure: str | None
    fixed_elapsed_time_s: float | None
    individual_elapsed_time_s: float
    total_elapsed_time_s: float
    initial_cn_roundoff_projection_count: int
    initial_cn_roundoff_projection_maximum_absolute: float
    mechanical_gain_audit: MechanicalGainAudit

    @property
    def feasible(self) -> bool:
        return self.total_moment.status == "complete"

    @property
    def target_total_moments(self) -> np.ndarray:
        return self.policy.target_total_moments

    @property
    def individual_total_moments(self) -> np.ndarray:
        return _completed_totals(self.individual_moment.achieved_moments)

    @property
    def total_policy_total_moments(self) -> np.ndarray:
        return _completed_totals(self.total_moment.achieved_moments)

    @property
    def fixed_total_moments(self) -> np.ndarray | None:
        if self.fixed is None:
            return None
        return _completed_totals(self.fixed.achieved_moments)

    @property
    def future_pulse_width_decision_count(self) -> int:
        """The implicit feedback introduces no horizon-wide PW variables."""

        return 0

    @property
    def explicit_horizon_pulse_width_count(self) -> int:
        return (
            self.config.horizon_cycles
            * self.policy.muscle_count
            * self.policy.interval_count
        )

    @property
    def maximum_absolute_total_policy_moment_error(self) -> float:
        residual = self.total_policy_total_moments - self.target_total_moments[None, :]
        finite = np.isfinite(residual)
        return float(np.max(np.abs(residual[finite]))) if np.any(finite) else math.nan

    @property
    def maximum_absolute_individual_moment_error(self) -> float:
        target = np.asarray(
            [interval.target_moments for interval in self.policy.intervals],
            dtype=float,
        ).T
        residual = self.individual_moment.achieved_moments - target[None, :, :]
        finite = np.isfinite(residual)
        return float(np.max(np.abs(residual[finite]))) if np.any(finite) else math.nan

    @property
    def maximum_absolute_fixed_moment_error(self) -> float | None:
        observed = self.fixed_total_moments
        if observed is None:
            return None
        residual = observed - self.target_total_moments[None, :]
        finite = np.isfinite(residual)
        return float(np.max(np.abs(residual[finite]))) if np.any(finite) else math.nan

    @property
    def maximum_pulse_width_utilization(self) -> float:
        pulse_widths = np.asarray(self.total_moment.pulse_widths, dtype=float)
        lower = np.asarray([item.pd0 for item in self.policy.parameters])[None, :, None]
        upper = np.asarray(
            [item.pulse_width_max for item in self.policy.parameters]
        )[None, :, None]
        utilization = (pulse_widths - lower) / (upper - lower)
        finite = np.isfinite(utilization)
        return float(np.max(utilization[finite])) if np.any(finite) else math.nan

    @property
    def minimum_capacity_ratio(self) -> float:
        capacity = np.asarray(self.total_moment.state_history, dtype=float)[:, :, 2]
        resting = np.asarray(
            [item.fatigue.a_rest for item in self.policy.parameters], dtype=float
        )[None, :]
        ratio = capacity / resting
        finite = np.isfinite(ratio)
        return float(np.min(ratio[finite])) if np.any(finite) else math.nan

    def summary(self) -> dict[str, Any]:
        """Return JSON-ready metrics for benchmark and launch layers."""

        return {
            "method": "implicit_phase_local_total_moment_feedback_full_ding",
            "status": self.total_moment.status,
            "feasible": self.feasible,
            "requested_cycles": self.config.horizon_cycles,
            "completed_cycles": self.total_moment.completed_cycles,
            "completed_intervals": self.total_moment.completed_intervals,
            "muscle_count": self.policy.muscle_count,
            "interval_count": self.policy.interval_count,
            "one_cycle_reference_parameters": self.policy.reference_parameter_count,
            "source_bound_projection_count": self.policy.source_bound_projection_count,
            "source_bound_projection_maximum_absolute_s": self.policy.source_bound_projection_maximum_absolute_s,
            "future_pw_decision_variables": self.future_pulse_width_decision_count,
            "equivalent_explicit_future_pw_variables": self.explicit_horizon_pulse_width_count,
            "maximum_absolute_total_policy_moment_error_nm": self.maximum_absolute_total_policy_moment_error,
            "maximum_absolute_individual_moment_error_nm": self.maximum_absolute_individual_moment_error,
            "maximum_absolute_fixed_moment_error_nm": self.maximum_absolute_fixed_moment_error,
            "maximum_pw_utilization": self.maximum_pulse_width_utilization,
            "minimum_capacity_ratio": self.minimum_capacity_ratio,
            "fixed_elapsed_time_s": self.fixed_elapsed_time_s,
            "individual_elapsed_time_s": self.individual_elapsed_time_s,
            "total_elapsed_time_s": self.total_elapsed_time_s,
            "initial_cn_roundoff_projection_count": self.initial_cn_roundoff_projection_count,
            "initial_cn_roundoff_projection_maximum_absolute": self.initial_cn_roundoff_projection_maximum_absolute,
            "mechanical_gain_audit": self.mechanical_gain_audit.summary(),
            "individual_scalar_function_evaluations": self.individual_moment.scalar_function_evaluations,
            "total_scalar_function_evaluations": self.total_moment.scalar_function_evaluations,
            "individual_status": self.individual_moment.status,
            "individual_completed_cycles": self.individual_moment.completed_cycles,
            "individual_first_failure": self.individual_moment.first_failure,
            "first_failure": self.total_moment.first_failure,
            "fixed_failure": self.fixed_failure,
        }


def run_five_cycle_viability_preview(
    policy: CompactMomentViabilityPolicy,
    *,
    initial_states: Sequence[Sequence[float]] | np.ndarray,
    config: FiveCycleViabilityConfig = FiveCycleViabilityConfig(),
) -> FiveCycleViabilityResult:
    """Roll out full Ding fatigue while preserving the one-cycle total moment.

    ``initial_states`` normally contains the terminal muscle states of the
    one-cycle OCP that will actually be applied.  The returned future PW values
    are a viability witness only; receding-horizon control still executes the
    optimized first cycle and resolves at the next update.
    """

    if not isinstance(policy, CompactMomentViabilityPolicy):
        raise TypeError("policy must be CompactMomentViabilityPolicy.")
    if not isinstance(config, FiveCycleViabilityConfig):
        raise TypeError("config must be FiveCycleViabilityConfig.")
    initial_states = np.asarray(initial_states, dtype=float)
    expected = (policy.muscle_count, 5)
    if initial_states.shape != expected or not np.all(np.isfinite(initial_states)):
        raise ValueError(f"initial_states must be finite with shape {expected}.")
    initial_states = initial_states.copy()
    negative_cn = initial_states[:, 0] < 0.0
    excessive_cn = initial_states[:, 0] < -config.state_roundoff_tolerance
    if np.any(excessive_cn):
        minimum = float(np.min(initial_states[:, 0]))
        raise ValueError(
            "initial Cn is physiologically negative beyond the declared roundoff "
            f"tolerance ({minimum} < -{config.state_roundoff_tolerance})."
        )
    projection_count = int(np.count_nonzero(negative_cn))
    projection_maximum = (
        float(np.max(np.abs(initial_states[negative_cn, 0])))
        if projection_count
        else 0.0
    )
    initial_states[negative_cn, 0] = 0.0
    gain_audit = audit_policy_mechanical_gains(
        policy, integration_substeps=config.integration_substeps
    )
    if not gain_audit.valid:
        raise MechanicalGainDomainError(gain_audit)
    cached_intervals = _cache_callable_mechanical_gains(policy.intervals)

    individual_started = perf_counter()
    individual = rollout_adaptive_moment_policy(
        initial_states,
        intervals=cached_intervals,
        parameters=policy.parameters,
        horizon_cycles=config.horizon_cycles,
        integration_substeps=config.integration_substeps,
        moment_tolerance=config.moment_tolerance,
    )
    individual_elapsed = perf_counter() - individual_started

    total_started = perf_counter()
    total_moment = rollout_bounded_total_moment_reference_policy(
        initial_states,
        intervals=cached_intervals,
        parameters=policy.parameters,
        horizon_cycles=config.horizon_cycles,
        integration_substeps=config.integration_substeps,
        moment_tolerance=config.moment_tolerance,
        allocation_options=config.allocation_options,
    )
    total_elapsed = perf_counter() - total_started
    fixed = None
    fixed_failure = None
    fixed_elapsed = None
    if config.evaluate_fixed_pulse_width_baseline:
        fixed_started = perf_counter()
        try:
            fixed = rollout_fixed_pulse_width_policy(
                initial_states,
                intervals=cached_intervals,
                parameters=policy.parameters,
                pulse_widths=policy.source_pulse_widths,
                horizon_cycles=config.horizon_cycles,
                integration_substeps=config.integration_substeps,
            )
        except (ValueError, RuntimeError) as error:
            # A failed diagnostic baseline must not hide a valid adaptive
            # viability witness.  The failure remains explicit in the report.
            fixed_failure = f"{type(error).__name__}: {error}"
        fixed_elapsed = perf_counter() - fixed_started
    return FiveCycleViabilityResult(
        policy=policy,
        config=config,
        individual_moment=individual,
        total_moment=total_moment,
        fixed=fixed,
        fixed_failure=fixed_failure,
        fixed_elapsed_time_s=fixed_elapsed,
        individual_elapsed_time_s=individual_elapsed,
        total_elapsed_time_s=total_elapsed,
        initial_cn_roundoff_projection_count=projection_count,
        initial_cn_roundoff_projection_maximum_absolute=projection_maximum,
        mechanical_gain_audit=gain_audit,
    )
