r"""Low-cost pulse-width adaptation for a repeated muscle-moment policy.

The policy does not repeat pulse widths.  For every muscle and stimulation
interval it propagates the five-state Ding 2007 model, then solves one bounded
scalar equation so the interval-end muscle moment matches a prescribed target.
This is a diagnostic/prediction primitive: it uses no FHO data, never clips an
infeasible pulse width, and reports the first target outside the reachable
``[PD0, PW_max]`` envelope.

The calcium forcing matches ``periodic_node``: within an interval of local
time ``s``, the history input is ``H0 exp(-s/tauc)``.  Muscle geometry may vary
inside the interval through a callable mechanical gain.  A fixed-step RK4 map
keeps cost deterministic and makes the numerical fidelity explicit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import numpy as np
from scipy.optimize import brentq

from .ding_fatigue_rollout import (
    DingFatigueParameters,
    ding_fatigue_parameters_from_model,
)
from .smooth_muscle_moment_allocation import (
    SmoothMomentAllocationOptions,
    solve_bounded_moment_qp_reference,
)


FULL_DING_STATE_NAMES = ("Cn", "F", "A", "Tau1", "Km")
MechanicalGain = float | Callable[[float], float]


class MomentTrackingStatus(str, Enum):
    OK = "ok"
    TARGET_FORCE_BELOW_PD0_RESPONSE = "target_force_below_pd0_response"
    TARGET_FORCE_ABOVE_PW_MAX_RESPONSE = "target_force_above_pw_max_response"
    INCOMPATIBLE_MOMENT_DIRECTION = "incompatible_moment_direction"
    NON_MONOTONIC_RESPONSE = "non_monotonic_response"
    DING_DOMAIN_ERROR = "ding_domain_error"


class DingDomainError(ValueError):
    """The fixed-step propagation left the declared Ding domain."""


def _finite(value: float, *, name: str, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.0):
        qualifier = "finite and strictly positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return value


def _state(values: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (5,) or not np.all(np.isfinite(values)):
        raise ValueError(f"state must be finite with order {FULL_DING_STATE_NAMES}.")
    return values.copy()


@dataclass(frozen=True)
class DingPulseWidthParameters:
    fatigue: DingFatigueParameters
    tauc: float
    tau2: float
    pd0: float
    pdt: float
    pulse_width_max: float

    def __post_init__(self) -> None:
        _finite(self.tauc, name="tauc", positive=True)
        _finite(self.tau2, name="tau2")
        _finite(self.pd0, name="pd0")
        _finite(self.pdt, name="pdt", positive=True)
        _finite(self.pulse_width_max, name="pulse_width_max", positive=True)
        if self.tau2 < 0.0 or self.pd0 < 0.0 or self.pulse_width_max <= self.pd0:
            raise ValueError("Ding PW parameters require tau2 >= 0, pd0 >= 0 and PW_max > pd0.")

    @classmethod
    def from_model(cls, model: Any, *, pulse_width_max: float) -> "DingPulseWidthParameters":
        return cls(
            fatigue=ding_fatigue_parameters_from_model(model),
            tauc=float(model.tauc),
            tau2=float(model.tau2),
            pd0=float(model.pd0),
            pdt=float(model.pdt),
            pulse_width_max=pulse_width_max,
        )


@dataclass(frozen=True)
class MomentTrackingInterval:
    """One repeated phase interval for all muscles.

    ``target_moments`` and ``moment_coefficients`` use the reduced-mechanics
    sign convention ``moment_i = coefficient_i * F_i`` at interval end.
    ``mechanical_gains`` enter the Ding force ODE and may vary with local time.
    """

    duration: float
    calcium_amplitudes: tuple[float, ...]
    mechanical_gains: tuple[MechanicalGain, ...]
    moment_coefficients: tuple[float, ...]
    target_moments: tuple[float, ...]

    def __post_init__(self) -> None:
        _finite(self.duration, name="duration", positive=True)
        count = len(self.target_moments)
        if count < 1 or any(
            len(values) != count
            for values in (
                self.calcium_amplitudes,
                self.mechanical_gains,
                self.moment_coefficients,
            )
        ):
            raise ValueError("Every interval field must contain the same non-zero muscle count.")
        for name, values in (
            ("calcium_amplitude", self.calcium_amplitudes),
            ("moment_coefficient", self.moment_coefficients),
            ("target_moment", self.target_moments),
        ):
            for value in values:
                _finite(value, name=name)


@dataclass(frozen=True)
class PulseWidthTrackingResult:
    status: MomentTrackingStatus
    pulse_width: float | None
    achieved_moment: float | None
    target_moment: float
    moment_error: float | None
    next_state: np.ndarray | None
    minimum_reachable_moment: float | None
    maximum_reachable_moment: float | None
    function_evaluations: int
    message: str

    @property
    def feasible(self) -> bool:
        return self.status is MomentTrackingStatus.OK


@dataclass(frozen=True)
class AdaptiveMomentRolloutResult:
    status: str
    requested_cycles: int
    completed_cycles: int
    completed_intervals: int
    pulse_widths: np.ndarray
    achieved_moments: np.ndarray
    state_history: np.ndarray
    first_failure: dict[str, Any] | None
    scalar_function_evaluations: int


@dataclass(frozen=True)
class FixedPulseWidthRolloutResult:
    """Forward prediction obtained by naively repeating one PW cycle."""

    pulse_widths: np.ndarray
    achieved_moments: np.ndarray
    state_history: np.ndarray


@dataclass(frozen=True)
class TotalMomentReferenceRolloutResult:
    """Offline oracle for validating total-moment redistribution."""

    status: str
    requested_cycles: int
    completed_cycles: int
    completed_intervals: int
    pulse_widths: np.ndarray
    allocated_moments: np.ndarray
    achieved_moments: np.ndarray
    state_history: np.ndarray
    first_failure: dict[str, Any] | None
    scalar_function_evaluations: int


def effective_recruitment(capacity: float, pulse_width: float, parameters: DingPulseWidthParameters) -> float:
    """Ding 2007 PW recruitment law, evaluated only inside its bounds."""

    capacity = _finite(capacity, name="capacity", positive=True)
    pulse_width = _finite(pulse_width, name="pulse_width")
    if pulse_width < parameters.pd0 or pulse_width > parameters.pulse_width_max:
        raise ValueError("pulse_width lies outside [pd0, pulse_width_max].")
    return capacity * (-math.expm1(-(pulse_width - parameters.pd0) / parameters.pdt))


def periodic_calcium_state(
    initial_cn: float,
    local_time: float,
    calcium_amplitude: float,
    tauc: float,
) -> float:
    """Exact ``Cn`` solution for exponential periodic-node calcium forcing."""

    initial_cn = _finite(initial_cn, name="initial_cn")
    local_time = _finite(local_time, name="local_time")
    calcium_amplitude = _finite(calcium_amplitude, name="calcium_amplitude")
    tauc = _finite(tauc, name="tauc", positive=True)
    if initial_cn < 0.0 or local_time < 0.0 or calcium_amplitude < 0.0:
        raise ValueError("Cn, local time and calcium amplitude must be non-negative.")
    decay = math.exp(-local_time / tauc)
    return decay * (initial_cn + calcium_amplitude * local_time / tauc)


def _gain_value(gain: MechanicalGain, local_time: float) -> float:
    value = gain(local_time) if callable(gain) else gain
    # Match Ding2003.f_dot_fun exactly. The currently used passive law can
    # make fl*fv+fp negative; this is not the signed muscle moment arm.
    # Preserve that ODE and let the state-domain checks detect invalid force.
    value = _finite(value, name="mechanical_gain")
    return value


def _four_state_rhs(
    local_time: float,
    state: np.ndarray,
    *,
    initial_cn: float,
    pulse_width: float,
    calcium_amplitude: float,
    mechanical_gain: MechanicalGain,
    parameters: DingPulseWidthParameters,
) -> np.ndarray:
    force, capacity, tau1, km = state
    cn = periodic_calcium_state(initial_cn, local_time, calcium_amplitude, parameters.tauc)
    if force < -1e-10 or capacity <= 0.0 or tau1 <= 0.0 or km <= 0.0 or km + cn <= 0.0:
        raise DingDomainError("Ding state left the positive force/capacity/time/calcium domain.")
    activation = cn / (km + cn)
    relaxation = tau1 + parameters.tau2 * activation
    if relaxation <= 0.0:
        raise DingDomainError("Ding relaxation time became non-positive.")
    recruitment = effective_recruitment(capacity, pulse_width, parameters)
    force_dot = _gain_value(mechanical_gain, local_time) * (
        recruitment * activation - force / relaxation
    )
    fatigue = parameters.fatigue
    slow = np.array(
        [
            -(capacity - fatigue.a_rest) / fatigue.tau_fat + fatigue.alpha_a * force,
            -(tau1 - fatigue.tau1_rest) / fatigue.tau_fat + fatigue.alpha_tau1 * force,
            -(km - fatigue.km_rest) / fatigue.tau_fat + fatigue.alpha_km * force,
        ]
    )
    return np.concatenate(([force_dot], slow))


def propagate_ding_pulse_width_interval(
    state: Sequence[float] | np.ndarray,
    *,
    pulse_width: float,
    duration: float,
    calcium_amplitude: float,
    mechanical_gain: MechanicalGain,
    parameters: DingPulseWidthParameters,
    integration_substeps: int = 8,
) -> np.ndarray:
    """Propagate ``(Cn,F,A,Tau1,Km)`` with deterministic fixed-step RK4."""

    state = _state(state)
    duration = _finite(duration, name="duration", positive=True)
    pulse_width = _finite(pulse_width, name="pulse_width")
    calcium_amplitude = _finite(calcium_amplitude, name="calcium_amplitude")
    if calcium_amplitude < 0.0:
        raise ValueError("calcium_amplitude must be non-negative.")
    if isinstance(integration_substeps, bool) or int(integration_substeps) != integration_substeps:
        raise ValueError("integration_substeps must be a positive integer.")
    integration_substeps = int(integration_substeps)
    if integration_substeps < 1:
        raise ValueError("integration_substeps must be a positive integer.")
    # Validate the bound once, including the zero-recruitment endpoint PD0.
    effective_recruitment(state[2], pulse_width, parameters)
    initial_cn = state[0]
    current = state[1:].copy()
    step = duration / integration_substeps
    for index in range(integration_substeps):
        time = index * step
        kwargs = dict(
            initial_cn=initial_cn,
            pulse_width=pulse_width,
            calcium_amplitude=calcium_amplitude,
            mechanical_gain=mechanical_gain,
            parameters=parameters,
        )
        k1 = _four_state_rhs(time, current, **kwargs)
        k2 = _four_state_rhs(time + step / 2.0, current + step * k1 / 2.0, **kwargs)
        k3 = _four_state_rhs(time + step / 2.0, current + step * k2 / 2.0, **kwargs)
        k4 = _four_state_rhs(time + step, current + step * k3, **kwargs)
        current = current + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    final_cn = periodic_calcium_state(initial_cn, duration, calcium_amplitude, parameters.tauc)
    result = np.concatenate(([final_cn], current))
    _four_state_rhs(
        duration,
        current,
        initial_cn=initial_cn,
        pulse_width=pulse_width,
        calcium_amplitude=calcium_amplitude,
        mechanical_gain=mechanical_gain,
        parameters=parameters,
    )
    return result


def _propagate_ding_pulse_width_interval_with_slow_sensitivities(
    state: Sequence[float] | np.ndarray,
    *,
    pulse_width: float,
    duration: float,
    calcium_amplitude: float,
    mechanical_gain: MechanicalGain,
    parameters: DingPulseWidthParameters,
    integration_substeps: int = 8,
    slow_state_sensitivities: np.ndarray | None = None,
    pulse_width_sensitivity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """RK4 Ding map and its tangent map with respect to initial ``A,Tau1,Km``.

    ``slow_state_sensitivities`` is the incoming derivative of ``(F,A,Tau1,Km)``
    with respect to arbitrary local coordinates. ``pulse_width_sensitivity``
    supplies the derivative of this interval's PW in those same coordinates.
    The default is the identity map with respect to the three initial slow
    states. Keeping the tangent alongside the state replaces finite-difference
    replays during offline calibration; it is never inserted into an RHO NLP.
    """
    state = _state(state)
    duration = _finite(duration, name="duration", positive=True)
    pulse_width = _finite(pulse_width, name="pulse_width")
    calcium_amplitude = _finite(calcium_amplitude, name="calcium_amplitude")
    if calcium_amplitude < 0.0:
        raise ValueError("calcium_amplitude must be non-negative.")
    if isinstance(integration_substeps, bool) or int(integration_substeps) != integration_substeps:
        raise ValueError("integration_substeps must be a positive integer.")
    integration_substeps = int(integration_substeps)
    if integration_substeps < 1:
        raise ValueError("integration_substeps must be a positive integer.")
    effective_recruitment(state[2], pulse_width, parameters)
    sensitivity = (np.vstack((np.zeros(3), np.eye(3))) if slow_state_sensitivities is None
                   else np.asarray(slow_state_sensitivities, dtype=float).copy())
    if sensitivity.shape != (4, 3) or not np.all(np.isfinite(sensitivity)):
        if slow_state_sensitivities is None:
            raise AssertionError("Default slow-state sensitivity has an invalid internal shape.")
        if sensitivity.ndim != 2 or sensitivity.shape[0] != 4 or not sensitivity.shape[1] or not np.all(np.isfinite(sensitivity)):
            raise ValueError("slow_state_sensitivities must be finite with shape (4, coordinates).")
    width_sensitivity = (np.zeros(sensitivity.shape[1]) if pulse_width_sensitivity is None
                         else np.asarray(pulse_width_sensitivity, dtype=float).reshape(-1))
    if width_sensitivity.shape != (sensitivity.shape[1],) or not np.all(np.isfinite(width_sensitivity)):
        raise ValueError("pulse_width_sensitivity must be finite with one value per sensitivity coordinate.")
    initial_cn = state[0]
    current = state[1:].copy()
    recruitment_scale = -math.expm1(-(pulse_width - parameters.pd0) / parameters.pdt)
    fatigue = parameters.fatigue

    def rhs(local_time: float, local_state: np.ndarray, local_sensitivity: np.ndarray):
        derivative = _four_state_rhs(
            local_time, local_state, initial_cn=initial_cn, pulse_width=pulse_width,
            calcium_amplitude=calcium_amplitude, mechanical_gain=mechanical_gain, parameters=parameters,
        )
        force, capacity, tau1, km = local_state
        cn = periodic_calcium_state(initial_cn, local_time, calcium_amplitude, parameters.tauc)
        activation = cn / (km + cn)
        activation_km = -cn / (km + cn) ** 2
        relaxation = tau1 + parameters.tau2 * activation
        gain = _gain_value(mechanical_gain, local_time)
        jacobian = np.array([
            [-gain / relaxation, gain * recruitment_scale * activation,
             gain * force / relaxation**2,
             gain * (capacity * recruitment_scale * activation_km
                     + force * parameters.tau2 * activation_km / relaxation**2)],
            [fatigue.alpha_a, -1.0 / fatigue.tau_fat, 0.0, 0.0],
            [fatigue.alpha_tau1, 0.0, -1.0 / fatigue.tau_fat, 0.0],
            [fatigue.alpha_km, 0.0, 0.0, -1.0 / fatigue.tau_fat],
        ])
        direct_pw = np.zeros((4, local_sensitivity.shape[1]))
        direct_pw[0] = gain * capacity * activation * math.exp(
            -(pulse_width - parameters.pd0) / parameters.pdt
        ) / parameters.pdt * width_sensitivity
        return derivative, jacobian @ local_sensitivity + direct_pw

    step = duration / integration_substeps
    for index in range(integration_substeps):
        time = index * step
        k1, s1 = rhs(time, current, sensitivity)
        k2, s2 = rhs(time + step / 2.0, current + step * k1 / 2.0, sensitivity + step * s1 / 2.0)
        k3, s3 = rhs(time + step / 2.0, current + step * k2 / 2.0, sensitivity + step * s2 / 2.0)
        k4, s4 = rhs(time + step, current + step * k3, sensitivity + step * s3)
        current = current + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        sensitivity = sensitivity + step * (s1 + 2.0 * s2 + 2.0 * s3 + s4) / 6.0
    final_cn = periodic_calcium_state(initial_cn, duration, calcium_amplitude, parameters.tauc)
    rhs(duration, current, sensitivity)
    return np.concatenate(([final_cn], current)), sensitivity


def _propagate_ding_pulse_width_interval_batch(
    state: Sequence[float] | np.ndarray,
    *,
    pulse_widths: Sequence[float] | np.ndarray,
    duration: float,
    calcium_amplitude: float,
    mechanical_gain: MechanicalGain,
    parameters: DingPulseWidthParameters,
    integration_substeps: int = 8,
) -> np.ndarray:
    """Vectorized equivalent of :func:`propagate_ding_pulse_width_interval`.

    All rows start from the same state and differ only by pulse width. This is
    the exact situation used by the monotonicity/reachability grid in the
    scalar inversion. Keeping the RK4 stages in one NumPy array avoids nine
    repetitions of the Python integration loop without changing the grid or
    the full five-state Ding equations.
    """

    state = _state(state)
    widths = np.asarray(pulse_widths, dtype=float)
    if widths.ndim != 1 or widths.size < 1 or not np.all(np.isfinite(widths)):
        raise ValueError("pulse_widths must be a non-empty finite vector.")
    duration = _finite(duration, name="duration", positive=True)
    calcium_amplitude = _finite(calcium_amplitude, name="calcium_amplitude")
    if calcium_amplitude < 0.0:
        raise ValueError("calcium_amplitude must be non-negative.")
    if isinstance(integration_substeps, bool) or int(integration_substeps) != integration_substeps:
        raise ValueError("integration_substeps must be a positive integer.")
    integration_substeps = int(integration_substeps)
    if integration_substeps < 1:
        raise ValueError("integration_substeps must be a positive integer.")
    if np.any(widths < parameters.pd0) or np.any(widths > parameters.pulse_width_max):
        raise ValueError("pulse_width lies outside [pd0, pulse_width_max].")

    initial_cn = float(state[0])
    current = np.broadcast_to(state[1:], (widths.size, 4)).copy()
    recruitment_scale = -np.expm1(-(widths - parameters.pd0) / parameters.pdt)
    fatigue = parameters.fatigue

    def rhs(local_time: float, values: np.ndarray) -> np.ndarray:
        force = values[:, 0]
        capacity = values[:, 1]
        tau1 = values[:, 2]
        km = values[:, 3]
        cn = periodic_calcium_state(
            initial_cn, local_time, calcium_amplitude, parameters.tauc
        )
        if (
            np.any(force < -1e-10)
            or np.any(capacity <= 0.0)
            or np.any(tau1 <= 0.0)
            or np.any(km <= 0.0)
            or np.any(km + cn <= 0.0)
        ):
            raise DingDomainError(
                "Ding state left the positive force/capacity/time/calcium domain."
            )
        activation = cn / (km + cn)
        relaxation = tau1 + parameters.tau2 * activation
        if np.any(relaxation <= 0.0):
            raise DingDomainError("Ding relaxation time became non-positive.")
        recruitment = capacity * recruitment_scale
        force_dot = _gain_value(mechanical_gain, local_time) * (
            recruitment * activation - force / relaxation
        )
        capacity_dot = (
            -(capacity - fatigue.a_rest) / fatigue.tau_fat
            + fatigue.alpha_a * force
        )
        tau1_dot = (
            -(tau1 - fatigue.tau1_rest) / fatigue.tau_fat
            + fatigue.alpha_tau1 * force
        )
        km_dot = (
            -(km - fatigue.km_rest) / fatigue.tau_fat
            + fatigue.alpha_km * force
        )
        return np.column_stack((force_dot, capacity_dot, tau1_dot, km_dot))

    step = duration / integration_substeps
    for index in range(integration_substeps):
        time = index * step
        k1 = rhs(time, current)
        k2 = rhs(time + step / 2.0, current + step * k1 / 2.0)
        k3 = rhs(time + step / 2.0, current + step * k2 / 2.0)
        k4 = rhs(time + step, current + step * k3)
        current = current + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    final_cn = periodic_calcium_state(
        initial_cn, duration, calcium_amplitude, parameters.tauc
    )
    rhs(duration, current)
    return np.column_stack((np.full(widths.size, final_cn), current))


def solve_pulse_width_for_target_moment(
    state: Sequence[float] | np.ndarray,
    *,
    target_moment: float,
    moment_coefficient: float,
    duration: float,
    calcium_amplitude: float,
    mechanical_gain: MechanicalGain,
    parameters: DingPulseWidthParameters,
    integration_substeps: int = 8,
    moment_tolerance: float = 1e-8,
    root_tolerance_s: float = 1e-12,
    monotonic_samples: int = 9,
    boundary_states: tuple[np.ndarray, np.ndarray] | None = None,
) -> PulseWidthTrackingResult:
    """Find the bounded PW whose propagated endpoint force gives the target moment."""

    state = _state(state)
    target_moment = _finite(target_moment, name="target_moment")
    moment_coefficient = _finite(moment_coefficient, name="moment_coefficient")
    moment_tolerance = _finite(moment_tolerance, name="moment_tolerance", positive=True)
    root_tolerance_s = _finite(root_tolerance_s, name="root_tolerance_s", positive=True)
    if isinstance(monotonic_samples, bool) or int(monotonic_samples) != monotonic_samples:
        raise ValueError("monotonic_samples must be an integer greater than or equal to 2.")
    monotonic_samples = int(monotonic_samples)
    if monotonic_samples < 2:
        raise ValueError("monotonic_samples must be an integer greater than or equal to 2.")
    if abs(moment_coefficient) <= 1e-12:
        return PulseWidthTrackingResult(
            MomentTrackingStatus.INCOMPATIBLE_MOMENT_DIRECTION, None, None, target_moment, None, None,
            None, None, 0, "Moment coefficient is too small to identify a muscle force target.",
        )
    target_force = target_moment / moment_coefficient
    if target_force < -moment_tolerance / abs(moment_coefficient):
        return PulseWidthTrackingResult(
            MomentTrackingStatus.INCOMPATIBLE_MOMENT_DIRECTION, None, None, target_moment, None, None,
            None, None, 0, "Target moment and muscle moment coefficient imply a negative force.",
        )

    evaluations = 0
    transition_cache: dict[float, np.ndarray] = {}
    if boundary_states is not None:
        if len(boundary_states) != 2:
            raise ValueError("boundary_states must contain the PD0 and PW_max states.")
        transition_cache[float(parameters.pd0)] = _state(boundary_states[0])
        transition_cache[float(parameters.pulse_width_max)] = _state(boundary_states[1])

    def transition(pulse_width: float) -> np.ndarray:
        nonlocal evaluations
        pulse_width = float(pulse_width)
        cached = transition_cache.get(pulse_width)
        if cached is not None:
            return cached
        evaluations += 1
        result = propagate_ding_pulse_width_interval(
            state,
            pulse_width=pulse_width,
            duration=duration,
            calcium_amplitude=calcium_amplitude,
            mechanical_gain=mechanical_gain,
            parameters=parameters,
            integration_substeps=integration_substeps,
        )
        transition_cache[pulse_width] = result
        return result

    try:
        grid = np.linspace(parameters.pd0, parameters.pulse_width_max, monotonic_samples)
        missing = np.asarray(
            [float(value) for value in grid if float(value) not in transition_cache],
            dtype=float,
        )
        if missing.size:
            propagated = _propagate_ding_pulse_width_interval_batch(
                state,
                pulse_widths=missing,
                duration=duration,
                calcium_amplitude=calcium_amplitude,
                mechanical_gain=mechanical_gain,
                parameters=parameters,
                integration_substeps=integration_substeps,
            )
            evaluations += int(missing.size)
            transition_cache.update(
                (float(width), propagated[index])
                for index, width in enumerate(missing)
            )
        states = [transition(float(value)) for value in grid]
    except (DingDomainError, ValueError) as error:
        return PulseWidthTrackingResult(
            MomentTrackingStatus.DING_DOMAIN_ERROR, None, None, target_moment, None, None,
            None, None, evaluations, str(error),
        )
    forces = np.asarray([value[1] for value in states])
    scale = max(1.0, float(np.max(np.abs(forces))))
    if np.any(np.diff(forces) < -1e-10 * scale):
        boundary_moments = moment_coefficient * forces[[0, -1]]
        return PulseWidthTrackingResult(
            MomentTrackingStatus.NON_MONOTONIC_RESPONSE, None, None, target_moment, None, None,
            float(np.min(boundary_moments)), float(np.max(boundary_moments)), evaluations,
            "Endpoint force is not monotone over the bounded PW interval.",
        )
    minimum_force, maximum_force = float(forces[0]), float(forces[-1])
    moment_at_pd0 = moment_coefficient * minimum_force
    moment_at_pw_max = moment_coefficient * maximum_force
    minimum_moment = min(moment_at_pd0, moment_at_pw_max)
    maximum_moment = max(moment_at_pd0, moment_at_pw_max)
    force_tolerance = moment_tolerance / abs(moment_coefficient)
    if target_force < minimum_force - force_tolerance:
        return PulseWidthTrackingResult(
            MomentTrackingStatus.TARGET_FORCE_BELOW_PD0_RESPONSE,
            None, None, target_moment, None, None,
            minimum_moment, maximum_moment, evaluations,
            "Even PD0 produces more endpoint force than the target requests.",
        )
    if target_force > maximum_force + force_tolerance:
        return PulseWidthTrackingResult(
            MomentTrackingStatus.TARGET_FORCE_ABOVE_PW_MAX_RESPONSE,
            None, None, target_moment, None, None,
            minimum_moment, maximum_moment, evaluations,
            "PW_max produces less endpoint force than the target requests.",
        )

    if abs(target_force - minimum_force) <= force_tolerance:
        pulse_width, next_state = parameters.pd0, states[0]
    elif abs(target_force - maximum_force) <= force_tolerance:
        pulse_width, next_state = parameters.pulse_width_max, states[-1]
    else:
        def residual(value: float) -> float:
            return float(transition(value)[1] - target_force)

        try:
            pulse_width = float(
                brentq(
                    residual,
                    parameters.pd0,
                    parameters.pulse_width_max,
                    xtol=root_tolerance_s,
                    rtol=4.0 * np.finfo(float).eps,
                )
            )
            next_state = transition(pulse_width)
        except (DingDomainError, ValueError) as error:
            return PulseWidthTrackingResult(
                MomentTrackingStatus.DING_DOMAIN_ERROR, None, None, target_moment, None, None,
                minimum_moment, maximum_moment, evaluations, str(error),
            )
    achieved = moment_coefficient * float(next_state[1])
    return PulseWidthTrackingResult(
        MomentTrackingStatus.OK,
        pulse_width,
        achieved,
        target_moment,
        achieved - target_moment,
        next_state,
        minimum_moment,
        maximum_moment,
        evaluations,
        "Target moment reached inside the declared PW bounds.",
    )


def rollout_adaptive_moment_policy(
    initial_states: Sequence[Sequence[float]] | np.ndarray,
    *,
    intervals: Sequence[MomentTrackingInterval],
    parameters: Sequence[DingPulseWidthParameters],
    horizon_cycles: int,
    integration_substeps: int = 8,
    moment_tolerance: float = 1e-8,
) -> AdaptiveMomentRolloutResult:
    """Repeat moment targets while adapting every PW to the predicted Ding state."""

    initial_states = np.asarray(initial_states, dtype=float)
    muscle_count = len(parameters)
    if initial_states.shape != (muscle_count, 5) or not np.all(np.isfinite(initial_states)):
        raise ValueError("initial_states must have shape (muscle_count, 5).")
    if isinstance(horizon_cycles, bool) or int(horizon_cycles) != horizon_cycles or horizon_cycles < 1:
        raise ValueError("horizon_cycles must be a positive integer.")
    horizon_cycles = int(horizon_cycles)
    intervals = tuple(intervals)
    if not intervals or any(len(item.target_moments) != muscle_count for item in intervals):
        raise ValueError("intervals must be non-empty and match the parameter muscle count.")

    interval_count = len(intervals)
    pulse_widths = np.full((horizon_cycles, muscle_count, interval_count), np.nan)
    achieved_moments = np.full_like(pulse_widths, np.nan)
    state_history = np.full((horizon_cycles * interval_count + 1, muscle_count, 5), np.nan)
    current = initial_states.copy()
    state_history[0] = current
    completed_intervals = 0
    evaluations = 0
    first_failure = None

    for cycle_index in range(horizon_cycles):
        for interval_index, interval in enumerate(intervals):
            candidates = []
            for muscle_index, muscle_parameters in enumerate(parameters):
                result = solve_pulse_width_for_target_moment(
                    current[muscle_index],
                    target_moment=interval.target_moments[muscle_index],
                    moment_coefficient=interval.moment_coefficients[muscle_index],
                    duration=interval.duration,
                    calcium_amplitude=interval.calcium_amplitudes[muscle_index],
                    mechanical_gain=interval.mechanical_gains[muscle_index],
                    parameters=muscle_parameters,
                    integration_substeps=integration_substeps,
                    moment_tolerance=moment_tolerance,
                )
                evaluations += result.function_evaluations
                candidates.append(result)
                if not result.feasible:
                    first_failure = {
                        "cycle_index": cycle_index,
                        "interval_index": interval_index,
                        "muscle_index": muscle_index,
                        "status": result.status.value,
                        "target_moment": result.target_moment,
                        "minimum_reachable_moment": result.minimum_reachable_moment,
                        "maximum_reachable_moment": result.maximum_reachable_moment,
                        "message": result.message,
                    }
                    return AdaptiveMomentRolloutResult(
                        status="infeasible",
                        requested_cycles=horizon_cycles,
                        completed_cycles=completed_intervals // interval_count,
                        completed_intervals=completed_intervals,
                        pulse_widths=pulse_widths,
                        achieved_moments=achieved_moments,
                        state_history=state_history,
                        first_failure=first_failure,
                        scalar_function_evaluations=evaluations,
                    )
            for muscle_index, result in enumerate(candidates):
                pulse_widths[cycle_index, muscle_index, interval_index] = result.pulse_width
                achieved_moments[cycle_index, muscle_index, interval_index] = result.achieved_moment
                current[muscle_index] = result.next_state
            completed_intervals += 1
            state_history[completed_intervals] = current

    return AdaptiveMomentRolloutResult(
        status="complete",
        requested_cycles=horizon_cycles,
        completed_cycles=horizon_cycles,
        completed_intervals=completed_intervals,
        pulse_widths=pulse_widths,
        achieved_moments=achieved_moments,
        state_history=state_history,
        first_failure=first_failure,
        scalar_function_evaluations=evaluations,
    )


def rollout_fixed_pulse_width_policy(
    initial_states: Sequence[Sequence[float]] | np.ndarray,
    *,
    intervals: Sequence[MomentTrackingInterval],
    parameters: Sequence[DingPulseWidthParameters],
    pulse_widths: Sequence[Sequence[float]] | np.ndarray,
    horizon_cycles: int,
    integration_substeps: int = 8,
) -> FixedPulseWidthRolloutResult:
    """Propagate a fixed one-cycle PW policy as the non-adaptive baseline."""

    initial_states = np.asarray(initial_states, dtype=float)
    muscle_count = len(parameters)
    intervals = tuple(intervals)
    interval_count = len(intervals)
    pulse_widths = np.asarray(pulse_widths, dtype=float)
    if initial_states.shape != (muscle_count, 5) or not np.all(np.isfinite(initial_states)):
        raise ValueError("initial_states must have shape (muscle_count, 5).")
    if not intervals or any(len(item.target_moments) != muscle_count for item in intervals):
        raise ValueError("intervals must be non-empty and match the parameter muscle count.")
    if pulse_widths.shape != (muscle_count, interval_count) or not np.all(np.isfinite(pulse_widths)):
        raise ValueError("pulse_widths must have shape (muscle_count, interval_count).")
    if isinstance(horizon_cycles, bool) or int(horizon_cycles) != horizon_cycles or horizon_cycles < 1:
        raise ValueError("horizon_cycles must be a positive integer.")
    horizon_cycles = int(horizon_cycles)

    moments = np.empty((horizon_cycles, muscle_count, interval_count))
    history = np.empty((horizon_cycles * interval_count + 1, muscle_count, 5))
    current = initial_states.copy()
    history[0] = current
    completed = 0
    for cycle_index in range(horizon_cycles):
        for interval_index, interval in enumerate(intervals):
            for muscle_index, muscle_parameters in enumerate(parameters):
                current[muscle_index] = propagate_ding_pulse_width_interval(
                    current[muscle_index],
                    pulse_width=pulse_widths[muscle_index, interval_index],
                    duration=interval.duration,
                    calcium_amplitude=interval.calcium_amplitudes[muscle_index],
                    mechanical_gain=interval.mechanical_gains[muscle_index],
                    parameters=muscle_parameters,
                    integration_substeps=integration_substeps,
                )
                moments[cycle_index, muscle_index, interval_index] = (
                    interval.moment_coefficients[muscle_index] * current[muscle_index, 1]
                )
            completed += 1
            history[completed] = current
    return FixedPulseWidthRolloutResult(
        pulse_widths=np.broadcast_to(
            pulse_widths[None, :, :], (horizon_cycles, muscle_count, interval_count)
        ).copy(),
        achieved_moments=moments,
        state_history=history,
    )


def rollout_bounded_total_moment_reference_policy(
    initial_states: Sequence[Sequence[float]] | np.ndarray,
    *,
    intervals: Sequence[MomentTrackingInterval],
    parameters: Sequence[DingPulseWidthParameters],
    horizon_cycles: int,
    integration_substeps: int = 8,
    moment_tolerance: float = 1e-8,
    allocation_options: SmoothMomentAllocationOptions = SmoothMomentAllocationOptions(),
) -> TotalMomentReferenceRolloutResult:
    """Validate redistribution with a hard-bounded phase QP outside the NLP.

    This active-set oracle is intentionally separate from the differentiable
    objective.  It establishes whether preserving the total moment can remain
    feasible before adding future moment variables and smooth constraints to
    the RHO graph.
    """

    initial_states = np.asarray(initial_states, dtype=float)
    muscle_count = len(parameters)
    intervals = tuple(intervals)
    if initial_states.shape != (muscle_count, 5) or not np.all(np.isfinite(initial_states)):
        raise ValueError("initial_states must have shape (muscle_count, 5).")
    if not intervals or any(len(item.target_moments) != muscle_count for item in intervals):
        raise ValueError("intervals must be non-empty and match the parameter muscle count.")
    if isinstance(horizon_cycles, bool) or int(horizon_cycles) != horizon_cycles or horizon_cycles < 1:
        raise ValueError("horizon_cycles must be a positive integer.")
    horizon_cycles = int(horizon_cycles)
    interval_count = len(intervals)
    pulse_widths = np.full((horizon_cycles, muscle_count, interval_count), np.nan)
    allocated_moments = np.full_like(pulse_widths, np.nan)
    achieved_moments = np.full_like(pulse_widths, np.nan)
    state_history = np.full((horizon_cycles * interval_count + 1, muscle_count, 5), np.nan)
    current = initial_states.copy()
    state_history[0] = current
    completed_intervals = 0
    evaluations = 0

    def failed(payload: dict[str, Any]) -> TotalMomentReferenceRolloutResult:
        return TotalMomentReferenceRolloutResult(
            status="infeasible",
            requested_cycles=horizon_cycles,
            completed_cycles=completed_intervals // interval_count,
            completed_intervals=completed_intervals,
            pulse_widths=pulse_widths,
            allocated_moments=allocated_moments,
            achieved_moments=achieved_moments,
            state_history=state_history,
            first_failure=payload,
            scalar_function_evaluations=evaluations,
        )

    for cycle_index in range(horizon_cycles):
        for interval_index, interval in enumerate(intervals):
            lower_bounds = np.empty(muscle_count)
            upper_bounds = np.empty(muscle_count)
            boundary_states_by_muscle = []
            for muscle_index, muscle_parameters in enumerate(parameters):
                try:
                    state_at_pd0 = propagate_ding_pulse_width_interval(
                        current[muscle_index],
                        pulse_width=muscle_parameters.pd0,
                        duration=interval.duration,
                        calcium_amplitude=interval.calcium_amplitudes[muscle_index],
                        mechanical_gain=interval.mechanical_gains[muscle_index],
                        parameters=muscle_parameters,
                        integration_substeps=integration_substeps,
                    )
                    state_at_maximum = propagate_ding_pulse_width_interval(
                        current[muscle_index],
                        pulse_width=muscle_parameters.pulse_width_max,
                        duration=interval.duration,
                        calcium_amplitude=interval.calcium_amplitudes[muscle_index],
                        mechanical_gain=interval.mechanical_gains[muscle_index],
                        parameters=muscle_parameters,
                        integration_substeps=integration_substeps,
                    )
                except (DingDomainError, ValueError) as error:
                    return failed(
                        {
                            "cycle_index": cycle_index,
                            "interval_index": interval_index,
                            "muscle_index": muscle_index,
                            "status": "ding_domain_error",
                            "message": str(error),
                        }
                    )
                boundary_moments = interval.moment_coefficients[muscle_index] * np.asarray(
                    [state_at_pd0[1], state_at_maximum[1]]
                )
                lower_bounds[muscle_index] = float(np.min(boundary_moments))
                upper_bounds[muscle_index] = float(np.max(boundary_moments))
                boundary_states_by_muscle.append((state_at_pd0, state_at_maximum))

            reference = np.asarray(interval.target_moments)
            required_total = float(np.sum(reference))
            allocation = solve_bounded_moment_qp_reference(
                reference,
                required_total_moment=required_total,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                options=allocation_options,
                feasibility_tolerance=moment_tolerance,
            )
            if allocation.status != "ok":
                return failed(
                    {
                        "cycle_index": cycle_index,
                        "interval_index": interval_index,
                        "status": allocation.status,
                        "required_total_moment": required_total,
                        "minimum_total_moment": float(np.sum(lower_bounds)),
                        "maximum_total_moment": float(np.sum(upper_bounds)),
                        "message": allocation.message,
                    }
                )

            candidates = []
            for muscle_index, muscle_parameters in enumerate(parameters):
                result = solve_pulse_width_for_target_moment(
                    current[muscle_index],
                    target_moment=allocation.allocated_moments[muscle_index],
                    moment_coefficient=interval.moment_coefficients[muscle_index],
                    duration=interval.duration,
                    calcium_amplitude=interval.calcium_amplitudes[muscle_index],
                    mechanical_gain=interval.mechanical_gains[muscle_index],
                    parameters=muscle_parameters,
                    integration_substeps=integration_substeps,
                    moment_tolerance=moment_tolerance,
                    boundary_states=boundary_states_by_muscle[muscle_index],
                )
                evaluations += result.function_evaluations
                if not result.feasible:
                    return failed(
                        {
                            "cycle_index": cycle_index,
                            "interval_index": interval_index,
                            "muscle_index": muscle_index,
                            "status": result.status.value,
                            "message": result.message,
                        }
                    )
                candidates.append(result)
            for muscle_index, result in enumerate(candidates):
                pulse_widths[cycle_index, muscle_index, interval_index] = result.pulse_width
                allocated_moments[cycle_index, muscle_index, interval_index] = (
                    allocation.allocated_moments[muscle_index]
                )
                achieved_moments[cycle_index, muscle_index, interval_index] = result.achieved_moment
                current[muscle_index] = result.next_state
            completed_intervals += 1
            state_history[completed_intervals] = current

    return TotalMomentReferenceRolloutResult(
        status="complete",
        requested_cycles=horizon_cycles,
        completed_cycles=horizon_cycles,
        completed_intervals=completed_intervals,
        pulse_widths=pulse_widths,
        allocated_moments=allocated_moments,
        achieved_moments=achieved_moments,
        state_history=state_history,
        first_failure=None,
        scalar_function_evaluations=evaluations,
    )
