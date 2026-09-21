r"""Fixed-size differentiable muscle-only horizon for a cycling RHO.

Future pulse widths are outer-NLP variables.  The graph propagates the full
five-state periodic-node Ding model and returns one total-muscle-moment
equality per future phase.  Kinematics and moment arms are fixed numerical
profile parameters, so this adds no future multibody states and uses no nested
QP or FHO data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval


MUSCLE_HORIZON_DOMAIN_MARGIN_NAMES = (
    "Cn",
    "F",
    "A",
    "A_rest_minus_A",
    "Tau1_minus_rest",
    "Km_minus_rest",
    "Tau1",
    "Km_plus_Cn",
    "relaxation_time",
)


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _positive(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return value


@dataclass(frozen=True)
class MuscleHorizonLayout:
    muscle_count: int
    interval_count: int
    horizon_cycles: int = 3
    integration_substeps: int = 1
    smooth_max_temperature: float = 0.02
    allocation_weight: float = 0.05

    def __post_init__(self) -> None:
        for name in ("muscle_count", "interval_count", "horizon_cycles", "integration_substeps"):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "smooth_max_temperature",
            _positive(self.smooth_max_temperature, name="smooth_max_temperature"),
        )
        allocation_weight = float(self.allocation_weight)
        if not math.isfinite(allocation_weight) or allocation_weight < 0.0:
            raise ValueError("allocation_weight must be finite and non-negative.")
        object.__setattr__(self, "allocation_weight", allocation_weight)

    @property
    def phase_size(self) -> int:
        return self.muscle_count * self.interval_count

    @property
    def gain_size(self) -> int:
        return self.phase_size * self.integration_substeps

    @property
    def parameter_size(self) -> int:
        # duration + required total, three muscle-phase fields, and three RK gain fields.
        return 2 * self.interval_count + 3 * self.phase_size + 3 * self.gain_size

    @property
    def future_pulse_width_size(self) -> int:
        return self.horizon_cycles * self.phase_size

    @property
    def initial_state_size(self) -> int:
        return 5 * self.muscle_count

    @property
    def predicted_moment_size(self) -> int:
        return self.future_pulse_width_size

    @property
    def total_moment_constraint_size(self) -> int:
        return self.horizon_cycles * self.interval_count

    @property
    def domain_margin_size(self) -> int:
        return len(MUSCLE_HORIZON_DOMAIN_MARGIN_NAMES) * self.future_pulse_width_size

    def parameter_slices(self) -> dict[str, slice]:
        sizes = {
            "duration": self.interval_count,
            "required_total_moment": self.interval_count,
            "calcium_amplitude": self.phase_size,
            "moment_coefficient": self.phase_size,
            "reference_moment": self.phase_size,
            "gain_start": self.gain_size,
            "gain_midpoint": self.gain_size,
            "gain_endpoint": self.gain_size,
        }
        result: dict[str, slice] = {}
        start = 0
        for name, size in sizes.items():
            result[name] = slice(start, start + size)
            start += size
        if start != self.parameter_size:
            raise RuntimeError("Muscle-horizon parameter packing is inconsistent.")
        return result


@dataclass(frozen=True)
class MuscleHorizonPackedProfile:
    parameters: np.ndarray
    future_pulse_width_seed: np.ndarray


@dataclass(frozen=True)
class MuscleHorizonExpressions:
    objective: Any
    smooth_maximum_utilization: Any
    allocation_deviation: Any
    predicted_moments: Any
    total_moment_residuals: Any
    final_states: Any
    domain_margins: Any


def _matrix(values, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != shape or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return values


def pack_muscle_horizon_profile(
    intervals: Sequence[MomentTrackingInterval],
    source_pulse_widths: Sequence[Sequence[float]] | np.ndarray,
    layout: MuscleHorizonLayout,
) -> MuscleHorizonPackedProfile:
    """Sample fixed mechanics at the exact RK4 stages and pack one base cycle."""

    intervals = tuple(intervals)
    if len(intervals) != layout.interval_count:
        raise ValueError("intervals must match layout.interval_count.")
    source_pulse_widths = _matrix(
        source_pulse_widths,
        shape=(layout.muscle_count, layout.interval_count),
        name="source_pulse_widths",
    )
    duration = np.asarray([interval.duration for interval in intervals])
    amplitude = np.asarray([interval.calcium_amplitudes for interval in intervals]).T
    coefficient = np.asarray([interval.moment_coefficients for interval in intervals]).T
    reference = np.asarray([interval.target_moments for interval in intervals]).T
    required = np.sum(reference, axis=0)
    gain_shape = (layout.muscle_count, layout.interval_count, layout.integration_substeps)
    gain_start = np.empty(gain_shape)
    gain_midpoint = np.empty(gain_shape)
    gain_endpoint = np.empty(gain_shape)
    for interval_index, interval in enumerate(intervals):
        step = interval.duration / layout.integration_substeps
        for muscle_index, gain in enumerate(interval.mechanical_gains):
            evaluate = gain if callable(gain) else lambda _time, value=float(gain): value
            for substep in range(layout.integration_substeps):
                gain_start[muscle_index, interval_index, substep] = evaluate(substep * step)
                gain_midpoint[muscle_index, interval_index, substep] = evaluate(
                    (substep + 0.5) * step
                )
                gain_endpoint[muscle_index, interval_index, substep] = evaluate(
                    (substep + 1.0) * step
                )
    for name, values in (
        ("duration", duration),
        ("calcium_amplitude", amplitude),
        ("moment_coefficient", coefficient),
        ("reference_moment", reference),
        ("required_total_moment", required),
        ("gain_start", gain_start),
        ("gain_midpoint", gain_midpoint),
        ("gain_endpoint", gain_endpoint),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")
    if np.any(duration <= 0.0) or np.any(amplitude < 0.0):
        raise ValueError("Durations must be positive and calcium amplitudes non-negative.")
    if np.any(gain_start <= 0.0) or np.any(gain_midpoint <= 0.0) or np.any(gain_endpoint <= 0.0):
        raise ValueError("Every sampled mechanical gain must be strictly positive.")

    fields = {
        "duration": duration,
        "required_total_moment": required,
        "calcium_amplitude": amplitude,
        "moment_coefficient": coefficient,
        "reference_moment": reference,
        "gain_start": gain_start,
        "gain_midpoint": gain_midpoint,
        "gain_endpoint": gain_endpoint,
    }
    packed = np.empty(layout.parameter_size)
    for name, field_slice in layout.parameter_slices().items():
        packed[field_slice] = np.asarray(fields[name]).reshape(-1, order="C")
    seed = np.tile(source_pulse_widths.reshape(-1, order="C"), layout.horizon_cycles)
    return MuscleHorizonPackedProfile(parameters=packed, future_pulse_width_seed=seed)


def _profile_value(parameters, layout: MuscleHorizonLayout, name: str, *indices: int):
    field_slice = layout.parameter_slices()[name]
    if name in {"duration", "required_total_moment"}:
        flat_index = indices[0]
    elif name in {"calcium_amplitude", "moment_coefficient", "reference_moment"}:
        muscle, interval = indices
        flat_index = muscle * layout.interval_count + interval
    else:
        muscle, interval, substep = indices
        flat_index = (
            (muscle * layout.interval_count + interval) * layout.integration_substeps
            + substep
        )
    return parameters[field_slice.start + flat_index]


def _future_pw_value(future_pulse_widths, layout: MuscleHorizonLayout, cycle: int, muscle: int, interval: int):
    return future_pulse_widths[
        cycle * layout.phase_size + muscle * layout.interval_count + interval
    ]


def _periodic_cn(initial_cn, local_time, amplitude, tauc):
    import casadi as ca

    decay = ca.exp(-local_time / tauc)
    return decay * (initial_cn + amplitude * local_time / tauc)


def _four_state_rhs(
    state,
    *,
    cn,
    pulse_width,
    mechanical_gain,
    muscle: DingPulseWidthParameters,
):
    import casadi as ca

    force, capacity, tau1, km = (state[index] for index in range(4))
    activation = cn / (km + cn)
    relaxation = tau1 + muscle.tau2 * activation
    recruitment = capacity * (1.0 - ca.exp(-(pulse_width - muscle.pd0) / muscle.pdt))
    force_dot = mechanical_gain * (recruitment * activation - force / relaxation)
    fatigue = muscle.fatigue
    return ca.vertcat(
        force_dot,
        -(capacity - fatigue.a_rest) / fatigue.tau_fat + fatigue.alpha_a * force,
        -(tau1 - fatigue.tau1_rest) / fatigue.tau_fat + fatigue.alpha_tau1 * force,
        -(km - fatigue.km_rest) / fatigue.tau_fat + fatigue.alpha_km * force,
    )


def build_muscle_horizon_expressions(
    initial_states,
    profile_parameters,
    future_pulse_widths,
    *,
    muscles: Sequence[DingPulseWidthParameters],
    layout: MuscleHorizonLayout,
) -> MuscleHorizonExpressions:
    """Build the differentiable future Ding rollout and total-moment constraints."""

    import casadi as ca

    states = ca.vec(initial_states)
    profile = ca.vec(profile_parameters)
    pulse_widths = ca.vec(future_pulse_widths)
    if states.numel() != layout.initial_state_size:
        raise ValueError(f"initial_states must contain {layout.initial_state_size} values.")
    if profile.numel() != layout.parameter_size:
        raise ValueError(f"profile_parameters must contain {layout.parameter_size} values.")
    if pulse_widths.numel() != layout.future_pulse_width_size:
        raise ValueError(
            f"future_pulse_widths must contain {layout.future_pulse_width_size} values."
        )
    if len(muscles) != layout.muscle_count:
        raise ValueError("muscles must match layout.muscle_count.")

    current = [states[5 * muscle : 5 * muscle + 5] for muscle in range(layout.muscle_count)]
    predicted_moments = []
    total_residuals = []
    utilizations = []
    normalized_deviations = []
    domain_margins = []
    for cycle_index in range(layout.horizon_cycles):
        for interval_index in range(layout.interval_count):
            interval_moments = []
            reference_scale_squared = 1e-12
            for muscle_index, muscle in enumerate(muscles):
                pulse_width = _future_pw_value(
                    pulse_widths, layout, cycle_index, muscle_index, interval_index
                )
                duration = _profile_value(profile, layout, "duration", interval_index)
                amplitude = _profile_value(
                    profile, layout, "calcium_amplitude", muscle_index, interval_index
                )
                initial_cn = current[muscle_index][0]
                four_state = current[muscle_index][1:]
                step = duration / layout.integration_substeps
                for substep in range(layout.integration_substeps):
                    start_time = substep * step
                    midpoint_time = (substep + 0.5) * step
                    endpoint_time = (substep + 1.0) * step
                    cn_start = _periodic_cn(initial_cn, start_time, amplitude, muscle.tauc)
                    cn_midpoint = _periodic_cn(initial_cn, midpoint_time, amplitude, muscle.tauc)
                    cn_endpoint = _periodic_cn(initial_cn, endpoint_time, amplitude, muscle.tauc)
                    gain_start = _profile_value(
                        profile, layout, "gain_start", muscle_index, interval_index, substep
                    )
                    gain_midpoint = _profile_value(
                        profile, layout, "gain_midpoint", muscle_index, interval_index, substep
                    )
                    gain_endpoint = _profile_value(
                        profile, layout, "gain_endpoint", muscle_index, interval_index, substep
                    )
                    k1 = _four_state_rhs(
                        four_state,
                        cn=cn_start,
                        pulse_width=pulse_width,
                        mechanical_gain=gain_start,
                        muscle=muscle,
                    )
                    k2 = _four_state_rhs(
                        four_state + 0.5 * step * k1,
                        cn=cn_midpoint,
                        pulse_width=pulse_width,
                        mechanical_gain=gain_midpoint,
                        muscle=muscle,
                    )
                    k3 = _four_state_rhs(
                        four_state + 0.5 * step * k2,
                        cn=cn_midpoint,
                        pulse_width=pulse_width,
                        mechanical_gain=gain_midpoint,
                        muscle=muscle,
                    )
                    k4 = _four_state_rhs(
                        four_state + step * k3,
                        cn=cn_endpoint,
                        pulse_width=pulse_width,
                        mechanical_gain=gain_endpoint,
                        muscle=muscle,
                    )
                    four_state = four_state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                final_cn = _periodic_cn(initial_cn, duration, amplitude, muscle.tauc)
                current[muscle_index] = ca.vertcat(final_cn, four_state)
                force, capacity, tau1, km = (four_state[index] for index in range(4))
                coefficient = _profile_value(
                    profile, layout, "moment_coefficient", muscle_index, interval_index
                )
                moment = coefficient * force
                reference = _profile_value(
                    profile, layout, "reference_moment", muscle_index, interval_index
                )
                interval_moments.append(moment)
                predicted_moments.append(moment)
                reference_scale_squared += reference**2
                normalized_deviations.append((moment - reference) ** 2)
                activation = final_cn / (km + final_cn)
                relaxation = tau1 + muscle.tau2 * activation
                domain_margins.extend(
                    (
                        final_cn,
                        force,
                        capacity,
                        muscle.fatigue.a_rest - capacity,
                        tau1 - muscle.fatigue.tau1_rest,
                        km - muscle.fatigue.km_rest,
                        tau1,
                        km + final_cn,
                        relaxation,
                    )
                )
                utilizations.append(
                    (pulse_width - muscle.pd0) / (muscle.pulse_width_max - muscle.pd0)
                )
            required = _profile_value(
                profile, layout, "required_total_moment", interval_index
            )
            total_residuals.append(ca.sum1(ca.vertcat(*interval_moments)) - required)
            # Scale the M squared deviations of this phase by one common,
            # muscle-symmetric reference norm.
            for index in range(layout.muscle_count):
                normalized_deviations[-1 - index] /= reference_scale_squared

    utilization_vector = ca.vertcat(*utilizations)
    temperature = layout.smooth_max_temperature
    smooth_maximum = temperature * (
        ca.logsumexp(utilization_vector / temperature)
        - math.log(layout.future_pulse_width_size)
    )
    allocation_deviation = ca.sum1(ca.vertcat(*normalized_deviations)) / (
        layout.horizon_cycles * layout.interval_count
    )
    return MuscleHorizonExpressions(
        objective=smooth_maximum + layout.allocation_weight * allocation_deviation,
        smooth_maximum_utilization=smooth_maximum,
        allocation_deviation=allocation_deviation,
        predicted_moments=ca.vertcat(*predicted_moments),
        total_moment_residuals=ca.vertcat(*total_residuals),
        final_states=ca.vertcat(*current),
        domain_margins=ca.vertcat(*domain_margins),
    )


def build_muscle_horizon_function(
    *,
    muscles: Sequence[DingPulseWidthParameters],
    layout: MuscleHorizonLayout,
    symbolic_type: str = "SX",
):
    """Return one reusable CasADi function for the muscle-only horizon."""

    import casadi as ca

    if symbolic_type not in {"SX", "MX"}:
        raise ValueError("symbolic_type must be 'SX' or 'MX'.")
    symbol = getattr(ca, symbolic_type).sym
    initial = symbol("muscle_horizon_initial_states", layout.initial_state_size)
    profile = symbol("muscle_horizon_profile", layout.parameter_size)
    pulse_widths = symbol("future_pulse_widths", layout.future_pulse_width_size)
    expressions = build_muscle_horizon_expressions(
        initial,
        profile,
        pulse_widths,
        muscles=muscles,
        layout=layout,
    )
    return ca.Function(
        "rho_muscle_horizon",
        [initial, profile, pulse_widths],
        [
            expressions.objective,
            expressions.smooth_maximum_utilization,
            expressions.allocation_deviation,
            expressions.predicted_moments,
            expressions.total_moment_residuals,
            expressions.final_states,
            expressions.domain_margins,
        ],
        ["initial_states", "profile", "future_pulse_widths"],
        [
            "objective",
            "smooth_maximum_utilization",
            "allocation_deviation",
            "predicted_moments",
            "total_moment_residuals",
            "final_states",
            "domain_margins",
        ],
    )
