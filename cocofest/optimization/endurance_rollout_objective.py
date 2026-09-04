r"""Fixed-size CasADi objective for a frozen-policy Ding rollout.

This module only builds the symbolic, parameterized core required by a future
RHO objective.  The current slow state is symbolic, while every quantity that
describes the previous RHO policy is supplied through one flat runtime
parameter vector.  A CasADi function can therefore be compiled once and called
with a different certified profile without rebuilding its graph.

The objective is the normalized smooth maximum of recruitment utilization
over future cycles, muscles and phase intervals.  It uses no muscle-specific
weight and no FHO datum.  The function also returns every symbolic domain
margin used by the rollout.  A future OCP integration must constrain these
margins to be strictly positive; exposing them prevents an optimizer from
silently exploiting negative recruitment or a singular denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from cocofest.optimization.endurance_rollout import (
    DingRolloutMuscleParameters,
    PeriodicRecruitmentProfile,
)


_PROFILE_FIELDS = (
    "force",
    "force_derivative",
    "cn",
    "force_length",
    "force_velocity",
    "passive_force",
    "half_decay",
    "first_half_force_integral",
    "second_half_force_integral",
)


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a strictly positive integer.")
    return int(value)


def _positive_finite(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return value


@dataclass(frozen=True)
class RolloutObjectiveLayout:
    """Dimensions and deterministic packing contract of the runtime profile."""

    muscle_count: int
    interval_count: int
    horizon_cycles: int
    smooth_max_temperature: float = 0.02

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "muscle_count",
            _positive_integer(self.muscle_count, name="muscle_count"),
        )
        object.__setattr__(
            self,
            "interval_count",
            _positive_integer(self.interval_count, name="interval_count"),
        )
        object.__setattr__(
            self,
            "horizon_cycles",
            _positive_integer(self.horizon_cycles, name="horizon_cycles"),
        )
        object.__setattr__(
            self,
            "smooth_max_temperature",
            _positive_finite(
                self.smooth_max_temperature,
                name="smooth_max_temperature",
            ),
        )

    @property
    def field_size(self) -> int:
        return self.muscle_count * self.interval_count

    @property
    def parameter_size(self) -> int:
        return len(_PROFILE_FIELDS) * self.field_size

    @property
    def initial_state_size(self) -> int:
        return 3 * self.muscle_count

    @property
    def utilization_size(self) -> int:
        return self.horizon_cycles * self.field_size

    def field_slice(self, name: str) -> slice:
        """Return the slice of a field in the flat, row-major parameter vector."""

        try:
            index = _PROFILE_FIELDS.index(name)
        except ValueError as error:
            raise KeyError(f"Unknown rollout profile field: {name}") from error
        start = index * self.field_size
        return slice(start, start + self.field_size)


@dataclass(frozen=True)
class RolloutObjectiveExpressions:
    """Symbolic outputs of the fixed-size utilization rollout."""

    smooth_maximum_utilization: Any
    utilizations: Any
    final_slow_states: Any
    domain_margins: Any


def _profile_matrix(values, *, name: str, layout: RolloutObjectiveLayout) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    expected = (layout.muscle_count, layout.interval_count)
    if matrix.shape != expected or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite array with shape {expected}.")
    return matrix


def pack_rollout_objective_parameters(
    profile: PeriodicRecruitmentProfile,
    muscles: Sequence[DingRolloutMuscleParameters],
    layout: RolloutObjectiveLayout,
) -> np.ndarray:
    """Pack one certified periodic policy for the symbolic rollout.

    The force integrals are exact convolutions of the continuous Fourier force
    with each muscle's Ding recovery kernel over the two half intervals.  Only
    one base-cycle copy is stored because the policy is periodic.
    """

    if not isinstance(profile, PeriodicRecruitmentProfile):
        raise TypeError("profile must be a PeriodicRecruitmentProfile instance.")
    if profile.muscle_count != layout.muscle_count or profile.interval_count != layout.interval_count:
        raise ValueError("profile dimensions do not match the rollout objective layout.")
    if len(muscles) != layout.muscle_count or not all(
        isinstance(muscle, DingRolloutMuscleParameters) for muscle in muscles
    ):
        raise ValueError("muscles must match the rollout objective muscle count.")

    midpoint_times = profile.midpoint_times
    force = profile.force_profile.evaluate(midpoint_times)
    force_derivative = profile.force_profile.derivative(midpoint_times)
    half_duration = profile.interval_duration / 2.0
    first_integral = np.empty_like(force)
    second_integral = np.empty_like(force)
    half_decay = np.empty_like(force)
    for interval_index in range(layout.interval_count):
        interval_start = interval_index * profile.interval_duration
        for muscle_index, muscle in enumerate(muscles):
            half_decay[muscle_index, interval_index] = math.exp(
                -half_duration / muscle.fatigue.tau_fat
            )
            first_integral[muscle_index, interval_index] = (
                profile.force_profile.exponentially_weighted_force_integral(
                    start_time=interval_start,
                    duration=half_duration,
                    time_constant=muscle.fatigue.tau_fat,
                )[muscle_index]
            )
            second_integral[muscle_index, interval_index] = (
                profile.force_profile.exponentially_weighted_force_integral(
                    start_time=interval_start + half_duration,
                    duration=half_duration,
                    time_constant=muscle.fatigue.tau_fat,
                )[muscle_index]
            )

    fields = {
        "force": force,
        "force_derivative": force_derivative,
        "cn": profile.cn,
        "force_length": profile.force_length_relationship,
        "force_velocity": profile.force_velocity_relationship,
        "passive_force": profile.passive_force_relationship,
        "half_decay": half_decay,
        "first_half_force_integral": first_integral,
        "second_half_force_integral": second_integral,
    }
    if np.any(fields["force"] < 0.0):
        raise ValueError("force must be non-negative throughout the packed profile.")
    if np.any(fields["cn"] <= 0.0):
        raise ValueError("cn must be strictly positive throughout the packed profile.")
    mechanical_gain = fields["force_length"] * fields["force_velocity"] + fields["passive_force"]
    if np.any(mechanical_gain <= 0.0):
        raise ValueError("the packed profile must have strictly positive mechanical gain.")
    if np.any(fields["half_decay"] <= 0.0) or np.any(fields["half_decay"] > 1.0):
        raise ValueError("half_decay must lie in (0, 1].")
    if np.any(fields["first_half_force_integral"] < 0.0) or np.any(
        fields["second_half_force_integral"] < 0.0
    ):
        raise ValueError("weighted force integrals must be non-negative.")
    packed = np.empty(layout.parameter_size)
    for name in _PROFILE_FIELDS:
        # Explicit C order makes consecutive phase samples belong to the same
        # muscle. Symbolic unpacking below uses the identical scalar indexing.
        packed[layout.field_slice(name)] = _profile_matrix(
            fields[name], name=name, layout=layout
        ).reshape(-1, order="C")
    return packed


def _symbolic_profile_value(parameters, layout: RolloutObjectiveLayout, name: str, muscle: int, interval: int):
    offset = layout.field_slice(name).start
    return parameters[offset + muscle * layout.interval_count + interval]


def build_rollout_objective_expressions(
    initial_slow_states,
    profile_parameters,
    *,
    muscles: Sequence[DingRolloutMuscleParameters],
    layout: RolloutObjectiveLayout,
) -> RolloutObjectiveExpressions:
    """Build the smooth future-utilization objective without rebuilding an OCP.

    ``initial_slow_states`` is flat in muscle-major ``(A, Tau1, Km)`` order.
    ``profile_parameters`` follows :func:`pack_rollout_objective_parameters`.
    Both may be SX, MX or DM vectors.
    """

    import casadi as ca

    states = ca.vec(initial_slow_states)
    parameters = ca.vec(profile_parameters)
    if states.numel() != layout.initial_state_size:
        raise ValueError(
            f"initial_slow_states must contain {layout.initial_state_size} values."
        )
    if parameters.numel() != layout.parameter_size:
        raise ValueError(
            f"profile_parameters must contain {layout.parameter_size} values."
        )
    if len(muscles) != layout.muscle_count or not all(
        isinstance(muscle, DingRolloutMuscleParameters) for muscle in muscles
    ):
        raise ValueError("muscles must match the rollout objective muscle count.")

    current = [states[3 * muscle : 3 * muscle + 3] for muscle in range(layout.muscle_count)]
    utilizations = []
    domain_margins = []
    for _ in range(layout.horizon_cycles):
        for interval_index in range(layout.interval_count):
            for muscle_index, muscle in enumerate(muscles):
                fatigue = muscle.fatigue
                rest = ca.DM(fatigue.rest_state)
                alpha = ca.DM(fatigue.alpha)
                decay = _symbolic_profile_value(
                    parameters, layout, "half_decay", muscle_index, interval_index
                )
                first_integral = _symbolic_profile_value(
                    parameters,
                    layout,
                    "first_half_force_integral",
                    muscle_index,
                    interval_index,
                )
                midpoint = rest + decay * (current[muscle_index] - rest) + alpha * first_integral

                force = _symbolic_profile_value(
                    parameters, layout, "force", muscle_index, interval_index
                )
                force_derivative = _symbolic_profile_value(
                    parameters,
                    layout,
                    "force_derivative",
                    muscle_index,
                    interval_index,
                )
                cn = _symbolic_profile_value(
                    parameters, layout, "cn", muscle_index, interval_index
                )
                force_length = _symbolic_profile_value(
                    parameters,
                    layout,
                    "force_length",
                    muscle_index,
                    interval_index,
                )
                force_velocity = _symbolic_profile_value(
                    parameters,
                    layout,
                    "force_velocity",
                    muscle_index,
                    interval_index,
                )
                passive_force = _symbolic_profile_value(
                    parameters,
                    layout,
                    "passive_force",
                    muscle_index,
                    interval_index,
                )
                activation = cn / (midpoint[2] + cn)
                relaxation_time = midpoint[1] + muscle.tau2 * activation
                mechanical_gain = force_length * force_velocity + passive_force
                required = (
                    force_derivative / mechanical_gain + force / relaxation_time
                ) / activation
                maximum = midpoint[0] * (
                    1.0
                    - math.exp(
                        -(muscle.pulse_width_max - muscle.pd0) / muscle.pdt
                    )
                )
                domain_margins.extend(
                    (
                        midpoint[0],
                        midpoint[1],
                        midpoint[2] + cn,
                        cn,
                        mechanical_gain,
                        relaxation_time,
                        required,
                        maximum,
                    )
                )
                utilizations.append(required / maximum)

                second_integral = _symbolic_profile_value(
                    parameters,
                    layout,
                    "second_half_force_integral",
                    muscle_index,
                    interval_index,
                )
                current[muscle_index] = (
                    rest + decay * (midpoint - rest) + alpha * second_integral
                )

    utilization_vector = ca.vertcat(*utilizations)
    smooth_maximum = layout.smooth_max_temperature * (
        ca.logsumexp(utilization_vector / layout.smooth_max_temperature)
        - math.log(layout.utilization_size)
    )
    final_states = ca.vertcat(*current)
    domain_margin_vector = ca.vertcat(*domain_margins)
    return RolloutObjectiveExpressions(
        smooth_maximum_utilization=smooth_maximum,
        utilizations=utilization_vector,
        final_slow_states=final_states,
        domain_margins=domain_margin_vector,
    )


def build_rollout_objective_function(
    *,
    muscles: Sequence[DingRolloutMuscleParameters],
    layout: RolloutObjectiveLayout,
    symbolic_type: str = "SX",
):
    """Return one reusable CasADi function of ``(slow_state, profile)``.

    Building this function is analogous to building the objective subgraph
    once. Calling it with another packed profile changes only numerical inputs.
    """

    import casadi as ca

    if symbolic_type not in {"SX", "MX"}:
        raise ValueError("symbolic_type must be 'SX' or 'MX'.")
    symbol = getattr(ca, symbolic_type).sym
    initial_state = symbol("rollout_initial_slow_state", layout.initial_state_size)
    profile = symbol("rollout_profile", layout.parameter_size)
    expressions = build_rollout_objective_expressions(
        initial_state,
        profile,
        muscles=muscles,
        layout=layout,
    )
    return ca.Function(
        "rho_endurance_rollout_objective",
        [initial_state, profile],
        [
            expressions.smooth_maximum_utilization,
            expressions.utilizations,
            expressions.final_slow_states,
            expressions.domain_margins,
        ],
        ["initial_slow_state", "profile"],
        [
            "smooth_maximum_utilization",
            "utilizations",
            "final_slow_states",
            "domain_margins",
        ],
    )
