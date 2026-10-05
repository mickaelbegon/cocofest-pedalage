"""Reproducible pedal-sensor trials from the five-state Ding 2007 model.

This module generates *single-muscle* trials for each of the eight bilateral
muscles.  Geometry is an input: the signed moment arm and optional radial
projection must come from the prescribed cycling model or a measured
calibration.  Radial force is an observation channel, not a second torque.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

import numpy as np

from cocofest.optimization.adaptive_moment_rollout import (
    DingPulseWidthParameters,
    propagate_ding_pulse_width_interval,
)
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters


MUSCLE_NAMES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
BILATERAL_MUSCLE_NAMES = tuple(
    f"{side}_{muscle}" for side in ("right", "left") for muscle in MUSCLE_NAMES
)
_MUSCLE_SCALING = {
    "Delt_ant": (1148.6, -0.14, 445.5),
    "Delt_post": (1234.5, -0.11, 342.7),
    "Biceps": (3314.7, -0.056, 179.6),
    "Triceps": (7036.3, -0.024, 76.2),
}


@dataclass(frozen=True)
class CyclingTrialSpec:
    """Known input and geometry for one stimulated muscle.

    ``moment_arm_m`` is the signed muscle-to-crank moment arm sampled at
    ``time_s``.  ``radial_gain`` is an independently supplied map from muscle
    force to pedal radial force; ``None`` represents a one-axis sensor.
    ``mechanical_gain`` multiplies the Ding force derivative, matching the
    repository's force-length/velocity/passive factor; an array permits the
    prescribed isokinetic geometry to vary through the cycle.
    """

    name: str
    muscle_name: str
    time_s: np.ndarray
    angle_rad: np.ndarray
    pulse_times_s: np.ndarray
    pulse_width_s: np.ndarray
    moment_arm_m: np.ndarray
    crank_radius_m: float
    radial_gain: np.ndarray | None = None
    mechanical_gain: np.ndarray | None = None
    baseline_tangential_n: np.ndarray | None = None
    baseline_radial_n: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.muscle_name not in BILATERAL_MUSCLE_NAMES:
            raise ValueError(f"muscle_name must be one of {BILATERAL_MUSCLE_NAMES}.")
        time = np.asarray(self.time_s, dtype=float)
        if time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time)) or time[0] != 0 or np.any(np.diff(time) <= 0):
            raise ValueError("time_s must start at zero and be finite and strictly increasing.")
        for name in ("angle_rad", "moment_arm_m", "radial_gain", "mechanical_gain", "baseline_tangential_n", "baseline_radial_n"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=float)
            if array.shape != time.shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite with the same shape as time_s.")
        if self.baseline_radial_n is not None and self.radial_gain is None:
            raise ValueError("baseline_radial_n requires radial_gain.")
        if self.mechanical_gain is not None and np.any(np.asarray(self.mechanical_gain) <= 0):
            raise ValueError("mechanical_gain must be positive.")
        pulses = np.asarray(self.pulse_times_s, dtype=float)
        widths = np.asarray(self.pulse_width_s, dtype=float)
        if (pulses.ndim != 1 or pulses.size == 0 or widths.shape != pulses.shape
                or not np.all(np.isfinite(pulses)) or not np.all(np.isfinite(widths))
                or pulses[0] < 0 or pulses[-1] > time[-1] or np.any(np.diff(pulses) <= 0)):
            raise ValueError("pulse times and widths must be finite, aligned, and chronologically inside the trial.")
        if not math.isfinite(self.crank_radius_m) or self.crank_radius_m <= 0:
            raise ValueError("crank_radius_m must be finite and positive.")


@dataclass(frozen=True)
class SensorNoise:
    """Independent Gaussian sensor noise plus a constant channel bias (N)."""

    tangential_std_n: float = 0.0
    radial_std_n: float = 0.0
    tangential_bias_n: float = 0.0
    radial_bias_n: float = 0.0

    def __post_init__(self) -> None:
        values = (self.tangential_std_n, self.radial_std_n, self.tangential_bias_n, self.radial_bias_n)
        if not all(math.isfinite(value) for value in values) or min(values[:2]) < 0:
            raise ValueError("Noise standard deviations must be non-negative and all noise settings finite.")


@dataclass(frozen=True)
class SyntheticTrial:
    """Synthetic sensor readings and noise-free five-state ground truth."""

    spec: CyclingTrialSpec
    observed: np.ndarray  # columns: tangential, optional radial (N)
    noiseless: np.ndarray
    state: np.ndarray  # Cn, F, A, Tau1, Km at sample times
    noise_std: np.ndarray  # one value per observed channel (N)

    @property
    def time_s(self) -> np.ndarray:
        return np.asarray(self.spec.time_s)

    @property
    def torque_tangent_nm(self) -> np.ndarray:
        return self.observed[:, 0] * self.spec.crank_radius_m

    @property
    def force_radial_n(self) -> np.ndarray | None:
        return self.observed[:, 1] if self.observed.shape[1] == 2 else None


def default_ding_parameters(muscle_name: str) -> DingPulseWidthParameters:
    """Repository example scaling, shared between sides only as a starting prior."""

    if muscle_name not in BILATERAL_MUSCLE_NAMES:
        raise ValueError(f"Unknown bilateral muscle: {muscle_name!r}.")
    base_name = muscle_name.split("_", 1)[1]
    a_scale, alpha_a, tau_fat = _MUSCLE_SCALING[base_name]
    return DingPulseWidthParameters(
        fatigue=DingFatigueParameters(
            a_rest=a_scale, tau1_rest=0.060601, km_rest=0.137,
            alpha_a=alpha_a, alpha_tau1=2.1e-5, alpha_km=1.9e-5,
            tau_fat=tau_fat,
        ),
        tauc=0.011, tau2=0.001, pd0=0.000131405,
        pdt=0.000194138, pulse_width_max=0.0006,
    )


def override_ding_parameters(
    base: DingPulseWidthParameters, values: Mapping[str, float]
) -> DingPulseWidthParameters:
    """Apply a candidate's physical parameter values for prediction/design."""

    if "a_scale" in values and "a_rest" in values:
        raise ValueError("Use either a_scale or a_rest for Ding 2007 capacity, not both.")
    values = dict(values)
    if "a_scale" in values:
        values["a_rest"] = values.pop("a_scale")
    fatigue_fields = set(DingFatigueParameters.__dataclass_fields__)
    fast_fields = set(DingPulseWidthParameters.__dataclass_fields__) - {"fatigue"}
    if unknown := set(values) - fatigue_fields - fast_fields:
        raise ValueError(f"Unknown Ding parameter(s): {sorted(unknown)}.")
    fatigue = replace(base.fatigue, **{key: values[key] for key in values if key in fatigue_fields})
    return replace(base, fatigue=fatigue, **{key: values[key] for key in values if key in fast_fields})


def _sampled_gain(spec: CyclingTrialSpec, field: str, time: float, default: float) -> float:
    values = getattr(spec, field)
    return default if values is None else float(np.interp(time, spec.time_s, values))


def predict_trial(
    parameters: Mapping[str, float],
    context: CyclingTrialSpec,
    *,
    base_parameters: DingPulseWidthParameters | None = None,
    integration_step_s: float = 0.002,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict ``(n_samples, n_channels)`` sensor data and five-state trajectory.

    The first calcium pulse begins at its scheduled time.  The history sum
    accumulates variable-frequency Ding ``R_i`` increments and decays exactly
    between pulses; each interval uses the repository's numerical RK4 map.
    """

    if not math.isfinite(integration_step_s) or integration_step_s <= 0:
        raise ValueError("integration_step_s must be finite and positive.")
    spec = context
    params = override_ding_parameters(base_parameters or default_ding_parameters(spec.muscle_name), parameters)
    pulses = np.asarray(spec.pulse_times_s, dtype=float)
    widths = np.asarray(spec.pulse_width_s, dtype=float)
    if np.any(widths < params.pd0) or np.any(widths > params.pulse_width_max):
        raise ValueError("All pulse widths must lie within [pd0, pulse_width_max].")
    time = np.asarray(spec.time_s, dtype=float)
    state = np.empty((time.size, 5), dtype=float)
    current = np.array([0.0, 0.0, params.fatigue.a_rest,
                        params.fatigue.tau1_rest, params.fatigue.km_rest], dtype=float)
    current_time = 0.0
    history = 0.0
    active_width = params.pd0
    last_pulse: float | None = None
    pulse_index = 0

    def advance(target: float) -> None:
        nonlocal current, current_time, history
        duration = target - current_time
        if duration <= 0:
            return
        substeps = max(1, math.ceil(duration / integration_step_s))
        current = propagate_ding_pulse_width_interval(
            current, pulse_width=active_width, duration=duration,
            calcium_amplitude=history,
            mechanical_gain=lambda local: _sampled_gain(spec, "mechanical_gain", current_time + local, 1.0),
            parameters=params, integration_substeps=substeps,
        )
        history *= math.exp(-duration / params.tauc)
        current_time = target

    for sample_index, sample_time in enumerate(time):
        while pulse_index < pulses.size and pulses[pulse_index] <= sample_time:
            event_time = float(pulses[pulse_index])
            advance(event_time)
            if last_pulse is None:
                increment = 1.0
            else:
                decay = math.exp(-(event_time - last_pulse) / params.tauc)
                increment = 1.0 + (params.fatigue.km_rest + 1.04 - 1.0) * decay
            history += increment
            active_width = float(widths[pulse_index])
            last_pulse = event_time
            pulse_index += 1
        advance(float(sample_time))
        state[sample_index] = current

    force = state[:, 1]
    tangent = force * np.asarray(spec.moment_arm_m) / spec.crank_radius_m
    if spec.baseline_tangential_n is not None:
        tangent = tangent + np.asarray(spec.baseline_tangential_n)
    columns = [tangent]
    if spec.radial_gain is not None:
        radial = force * np.asarray(spec.radial_gain)
        if spec.baseline_radial_n is not None:
            radial = radial + np.asarray(spec.baseline_radial_n)
        columns.append(radial)
    return np.column_stack(columns), state


def generate_trial(
    spec: CyclingTrialSpec,
    *,
    parameters: Mapping[str, float] | None = None,
    noise: SensorNoise = SensorNoise(),
    seed: int = 0,
) -> SyntheticTrial:
    """Generate a reproducible, single-muscle sensor trial."""

    noiseless, state = predict_trial(parameters or {}, spec)
    rng = np.random.default_rng(seed)
    standard_deviations = np.array(
        [noise.tangential_std_n] + ([noise.radial_std_n] if noiseless.shape[1] == 2 else []),
        dtype=float,
    )
    biases = np.array(
        [noise.tangential_bias_n] + ([noise.radial_bias_n] if noiseless.shape[1] == 2 else []),
        dtype=float,
    )
    observed = noiseless + biases + rng.normal(size=noiseless.shape) * standard_deviations
    return SyntheticTrial(spec, observed, noiseless, state, standard_deviations)


def simple_mono_muscle_spec(
    muscle_name: str,
    *,
    frequency_hz: float = 30.0,
    pulse_width_s: float = 0.0003,
    duration_s: float = 1.0,
    sample_rate_hz: float = 200.0,
    cadence_rad_s: float = 2 * math.pi / 4,
    moment_arm_m: float = 0.02,
    crank_radius_m: float = 0.17,
    radial_gain: float | None = None,
) -> CyclingTrialSpec:
    """Minimal smoke-test trial; replace the constant projections for studies."""

    if min(frequency_hz, duration_s, sample_rate_hz) <= 0:
        raise ValueError("Frequency, duration, and sample rate must be positive.")
    time = np.linspace(0.0, duration_s, round(duration_s * sample_rate_hz) + 1)
    pulse_times = np.arange(math.ceil(duration_s * frequency_hz)) / frequency_hz
    pulse_times = pulse_times[pulse_times <= duration_s]
    return CyclingTrialSpec(
        name=f"{muscle_name}_{frequency_hz:g}Hz_{pulse_width_s * 1e6:g}us",
        muscle_name=muscle_name, time_s=time, angle_rad=cadence_rad_s * time,
        pulse_times_s=pulse_times,
        pulse_width_s=np.full(pulse_times.shape, pulse_width_s),
        moment_arm_m=np.full(time.shape, moment_arm_m), crank_radius_m=crank_radius_m,
        radial_gain=None if radial_gain is None else np.full(time.shape, radial_gain),
    )
