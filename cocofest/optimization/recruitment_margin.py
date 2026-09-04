r"""Safe diagnostics for Ding fatigue damage and pulse-width recruitment.

This module implements two quantities that follow directly from the Ding model
already used by :mod:`cocofest`:

* the normalized, force-driven contribution to capacity loss,

  ``d = -alpha_a * F / A_rest``;

* the effective recruitment needed to reproduce a force and its derivative,
  followed by the exact inverse of the Ding 2007 pulse-width law.

For the force dynamics implemented in
``DingModelPulseWidthFrequencyWithFatigue`` define

.. math::

    r = \frac{C_n}{K_m+C_n},\qquad
    g = f_l f_v + f_p,\qquad
    T = \tau_1 + \tau_2 r.

The effective recruitment ``a_eff`` appears in

.. math::

    \dot F = g\left(a_{eff}r - \frac{F}{T}\right).

Consequently, when all denominators are in their physical domain,

.. math::

    a_{req} = \frac{\dot F/g + F/T}{r}.

The Ding 2007 pulse-width law and its inverse are

.. math::

    a_{eff}=A\left(1-e^{-(PW-PD_0)/PDT}\right),\qquad
    PW=PD_0-PDT\log\left(1-a_{req}/A\right).

The inversion therefore requires ``F_dot`` in addition to an instantaneous
force state. For a RHO trajectory, the recommended source is the analytical
derivative of the same collocation polynomial (or explicitly selected periodic
interpolant) used to evaluate ``F`` at the queried phase. If a checkpoint stores
only nodal force samples and no interpolation rule, the exact inversion is not
identifiable from that checkpoint alone. In particular, this module does not
silently substitute an arbitrary finite-difference stencil, whose phase and
peak errors could change the inferred feasibility margin.

No finite pulse width can attain ``a_req == A``. The numerical diagnostic below
reports this and every other domain failure explicitly; it never clips
utilization or pulse width.

The functions are independent of a full-horizon optimization and contain no
muscle-specific weighting.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np


class RecruitmentStatus(str, Enum):
    """Domain/feasibility status of a Ding pulse-width inversion."""

    OK = "ok"
    NON_FINITE_INPUT = "non_finite_input"
    NON_FINITE_MUSCLE_RELATIONSHIP = "non_finite_muscle_relationship"
    NEGATIVE_FORCE = "negative_force"
    NEGATIVE_CN = "negative_cn"
    NON_POSITIVE_KM = "non_positive_km"
    NON_POSITIVE_TAU1 = "non_positive_tau1"
    NEGATIVE_TAU2 = "negative_tau2"
    NEGATIVE_PD0 = "negative_pd0"
    NON_POSITIVE_CAPACITY = "non_positive_capacity"
    INVALID_PULSE_WIDTH_TIME_CONSTANT = "invalid_pulse_width_time_constant"
    INVALID_MAXIMUM_PULSE_WIDTH = "invalid_maximum_pulse_width"
    CALCIUM_DENOMINATOR_NEAR_ZERO = "calcium_denominator_near_zero"
    CN_NEAR_ZERO = "cn_near_zero"
    NON_POSITIVE_ACTIVATION = "non_positive_activation"
    RELAXATION_DENOMINATOR_NEAR_ZERO = "relaxation_denominator_near_zero"
    NON_POSITIVE_RELAXATION_TIME = "non_positive_relaxation_time"
    MECHANICAL_GAIN_NEAR_ZERO = "mechanical_gain_near_zero"
    NON_POSITIVE_MECHANICAL_GAIN = "non_positive_mechanical_gain"
    NEGATIVE_REQUIRED_RECRUITMENT = "negative_required_recruitment"
    ZERO_MAXIMUM_RECRUITMENT = "zero_maximum_recruitment"
    RECRUITMENT_AT_CAPACITY_ASYMPTOTE = "recruitment_at_capacity_asymptote"
    RECRUITMENT_EXCEEDS_CAPACITY = "recruitment_exceeds_capacity"
    PULSE_WIDTH_LIMIT_EXCEEDED = "pulse_width_limit_exceeded"


@dataclass(frozen=True)
class RecruitmentMarginResult:
    """Result of :func:`ding_recruitment_margin`.

    ``utilization`` is ``required_effective_recruitment /
    maximum_effective_recruitment``. It is deliberately left above one when
    the requested force dynamics exceed ``pulse_width_max``.
    """

    status: RecruitmentStatus
    activation_fraction: float = math.nan
    relaxation_time: float = math.nan
    mechanical_gain: float = math.nan
    required_effective_recruitment: float = math.nan
    maximum_effective_recruitment: float = math.nan
    utilization: float = math.nan
    required_pulse_width: float = math.nan
    reconstructed_force_derivative: float = math.nan

    @property
    def feasible(self) -> bool:
        """Whether a finite pulse width no larger than the stated maximum exists."""

        return self.status is RecruitmentStatus.OK


def _finite_array(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def marginal_capacity_damage_rate(
    forces: Sequence[float] | np.ndarray,
    alpha_a: Sequence[float] | np.ndarray,
    resting_capacities: Sequence[float] | np.ndarray,
) -> np.ndarray:
    r"""Return ``-alpha_a * F / A_rest`` for each muscle or sample.

    This is the normalized force-driven term in the Ding capacity equation,
    with units of inverse time. It is not the complete ``-A_dot/A_rest``: the
    recovery term ``(A-A_rest)/tau_fat`` is intentionally excluded. Forces
    must be non-negative, identified ``alpha_a`` values non-positive, and
    resting capacities strictly positive for the output to be a damage rate.
    """

    forces = _finite_array(forces, name="forces")
    alpha_a = _finite_array(alpha_a, name="alpha_a")
    resting_capacities = _finite_array(resting_capacities, name="resting_capacities")
    if forces.shape != alpha_a.shape or forces.shape != resting_capacities.shape:
        raise ValueError("forces, alpha_a, and resting_capacities must have the same shape.")
    if np.any(forces < 0.0):
        raise ValueError("forces must be non-negative for a capacity-damage metric.")
    if np.any(alpha_a > 0.0):
        raise ValueError("alpha_a must be non-positive for a capacity-damage metric.")
    if np.any(resting_capacities <= 0.0):
        raise ValueError("resting_capacities must be strictly positive.")
    return -alpha_a * forces / resting_capacities


def effective_recruitment_from_pulse_width(
    capacity: float,
    pulse_width: float,
    pd0: float,
    pdt: float,
) -> float:
    """Evaluate the Ding 2007 pulse-width recruitment law on its physical domain."""

    values = (capacity, pulse_width, pd0, pdt)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("capacity, pulse_width, pd0, and pdt must be finite.")
    capacity = float(capacity)
    pulse_width = float(pulse_width)
    pd0 = float(pd0)
    pdt = float(pdt)
    if capacity <= 0.0:
        raise ValueError("capacity must be strictly positive.")
    if pdt <= 0.0:
        raise ValueError("pdt must be strictly positive.")
    if pd0 < 0.0:
        raise ValueError("pd0 must be non-negative.")
    if pulse_width < pd0:
        raise ValueError("pulse_width must be no smaller than pd0.")
    return capacity * (-math.expm1(-(pulse_width - pd0) / pdt))


def ding_force_derivative(
    *,
    cn: float,
    force: float,
    effective_recruitment: float,
    tau1: float,
    km: float,
    tau2: float,
    force_length_relationship: float = 1.0,
    force_velocity_relationship: float = 1.0,
    passive_force_relationship: float = 0.0,
) -> float:
    """Evaluate the force ODE used by the current Ding implementations.

    This direct evaluator raises for singular or non-physical denominators.
    Use :func:`ding_recruitment_margin` when explicit status reporting is
    required.
    """

    values = (
        cn,
        force,
        effective_recruitment,
        tau1,
        km,
        tau2,
        force_length_relationship,
        force_velocity_relationship,
        passive_force_relationship,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("all Ding force-dynamics inputs must be finite.")
    if float(force) < 0.0:
        raise ValueError("force must be non-negative.")
    if float(cn) < 0.0:
        raise ValueError("cn must be non-negative.")
    if float(km) <= 0.0:
        raise ValueError("km must be strictly positive.")
    if float(tau1) <= 0.0:
        raise ValueError("tau1 must be strictly positive.")
    if float(tau2) < 0.0:
        raise ValueError("tau2 must be non-negative.")
    calcium_denominator = float(km) + float(cn)
    if calcium_denominator == 0.0:
        raise ValueError("km + cn must be non-zero.")
    activation = float(cn) / calcium_denominator
    relaxation_time = float(tau1) + float(tau2) * activation
    if relaxation_time <= 0.0:
        raise ValueError("tau1 + tau2 * cn / (km + cn) must be strictly positive.")
    mechanical_gain = (
        float(force_length_relationship) * float(force_velocity_relationship)
        + float(passive_force_relationship)
    )
    if mechanical_gain <= 0.0:
        raise ValueError("the combined force-length/velocity/passive gain must be strictly positive.")
    return mechanical_gain * (
        float(effective_recruitment) * activation - float(force) / relaxation_time
    )


def _margin_failure(
    status: RecruitmentStatus,
    *,
    activation_fraction: float = math.nan,
    relaxation_time: float = math.nan,
    mechanical_gain: float = math.nan,
    required_effective_recruitment: float = math.nan,
    maximum_effective_recruitment: float = math.nan,
    utilization: float = math.nan,
    required_pulse_width: float = math.nan,
) -> RecruitmentMarginResult:
    return RecruitmentMarginResult(
        status=status,
        activation_fraction=activation_fraction,
        relaxation_time=relaxation_time,
        mechanical_gain=mechanical_gain,
        required_effective_recruitment=required_effective_recruitment,
        maximum_effective_recruitment=maximum_effective_recruitment,
        utilization=utilization,
        required_pulse_width=required_pulse_width,
    )


def ding_recruitment_margin(
    *,
    cn: float,
    force: float,
    force_derivative: float,
    capacity: float,
    tau1: float,
    km: float,
    tau2: float,
    pd0: float,
    pdt: float,
    pulse_width_max: float,
    force_length_relationship: float = 1.0,
    force_velocity_relationship: float = 1.0,
    passive_force_relationship: float = 0.0,
    denominator_tolerance: float = 1e-12,
    activation_tolerance: float = 1e-12,
    recruitment_tolerance: float = 1e-12,
    pulse_width_tolerance: float = 1e-12,
) -> RecruitmentMarginResult:
    """Invert Ding force dynamics and report bounded-PW recruitment margin.

    The calculation is scalar by design so every muscle/phase sample receives
    its own status. Callers can aggregate valid ``utilization`` values without
    hiding which sample first left the model domain.

    ``force_derivative`` must be evaluated analytically from the same RHO
    collocation polynomial, or from the same explicitly declared and validated
    periodic interpolant, as the supplied ``force``. Inferring it from one
    isolated force value is impossible. A caller that only has nodal samples
    must first choose and validate an interpolation policy; an implicit
    ``numpy.gradient``-style fallback is deliberately absent.
    """

    raw_values = (
        cn,
        force,
        force_derivative,
        capacity,
        tau1,
        km,
        tau2,
        pd0,
        pdt,
        pulse_width_max,
        force_length_relationship,
        force_velocity_relationship,
        passive_force_relationship,
        denominator_tolerance,
        activation_tolerance,
        recruitment_tolerance,
        pulse_width_tolerance,
    )
    try:
        values = tuple(float(value) for value in raw_values)
    except (TypeError, ValueError):
        return _margin_failure(RecruitmentStatus.NON_FINITE_INPUT)
    (
        cn,
        force,
        force_derivative,
        capacity,
        tau1,
        km,
        tau2,
        pd0,
        pdt,
        pulse_width_max,
        force_length_relationship,
        force_velocity_relationship,
        passive_force_relationship,
        denominator_tolerance,
        activation_tolerance,
        recruitment_tolerance,
        pulse_width_tolerance,
    ) = values
    muscle_relationships = (
        force_length_relationship,
        force_velocity_relationship,
        passive_force_relationship,
    )
    if not all(math.isfinite(value) for value in muscle_relationships):
        return _margin_failure(RecruitmentStatus.NON_FINITE_MUSCLE_RELATIONSHIP)
    non_relationship_values = values[:10] + values[13:]
    if not all(math.isfinite(value) for value in non_relationship_values):
        return _margin_failure(RecruitmentStatus.NON_FINITE_INPUT)
    if min(denominator_tolerance, activation_tolerance, recruitment_tolerance, pulse_width_tolerance) < 0.0:
        return _margin_failure(RecruitmentStatus.NON_FINITE_INPUT)
    if force < 0.0:
        return _margin_failure(RecruitmentStatus.NEGATIVE_FORCE)
    if capacity <= 0.0:
        return _margin_failure(RecruitmentStatus.NON_POSITIVE_CAPACITY)
    if pdt <= 0.0:
        return _margin_failure(RecruitmentStatus.INVALID_PULSE_WIDTH_TIME_CONSTANT)
    if pd0 < 0.0:
        return _margin_failure(RecruitmentStatus.NEGATIVE_PD0)
    if pulse_width_max < pd0 - pulse_width_tolerance:
        return _margin_failure(RecruitmentStatus.INVALID_MAXIMUM_PULSE_WIDTH)

    calcium_denominator = km + cn
    if abs(calcium_denominator) <= denominator_tolerance:
        return _margin_failure(RecruitmentStatus.CALCIUM_DENOMINATOR_NEAR_ZERO)
    if cn < 0.0:
        return _margin_failure(RecruitmentStatus.NEGATIVE_CN)
    if km <= 0.0:
        return _margin_failure(RecruitmentStatus.NON_POSITIVE_KM)
    activation = cn / calcium_denominator
    if abs(cn) <= activation_tolerance or abs(activation) <= activation_tolerance:
        return _margin_failure(
            RecruitmentStatus.CN_NEAR_ZERO,
            activation_fraction=activation,
        )
    if activation < 0.0:
        return _margin_failure(
            RecruitmentStatus.NON_POSITIVE_ACTIVATION,
            activation_fraction=activation,
        )

    relaxation_time = tau1 + tau2 * activation
    if abs(relaxation_time) <= denominator_tolerance:
        return _margin_failure(
            RecruitmentStatus.RELAXATION_DENOMINATOR_NEAR_ZERO,
            activation_fraction=activation,
            relaxation_time=relaxation_time,
        )
    if tau1 <= 0.0:
        return _margin_failure(
            RecruitmentStatus.NON_POSITIVE_TAU1,
            activation_fraction=activation,
            relaxation_time=relaxation_time,
        )
    if tau2 < 0.0:
        return _margin_failure(
            RecruitmentStatus.NEGATIVE_TAU2,
            activation_fraction=activation,
            relaxation_time=relaxation_time,
        )
    if relaxation_time < 0.0:
        return _margin_failure(
            RecruitmentStatus.NON_POSITIVE_RELAXATION_TIME,
            activation_fraction=activation,
            relaxation_time=relaxation_time,
        )

    mechanical_gain = force_length_relationship * force_velocity_relationship + passive_force_relationship
    common = dict(
        activation_fraction=activation,
        relaxation_time=relaxation_time,
        mechanical_gain=mechanical_gain,
    )
    if abs(mechanical_gain) <= denominator_tolerance:
        return _margin_failure(RecruitmentStatus.MECHANICAL_GAIN_NEAR_ZERO, **common)
    if mechanical_gain < 0.0:
        return _margin_failure(RecruitmentStatus.NON_POSITIVE_MECHANICAL_GAIN, **common)

    required = (force_derivative / mechanical_gain + force / relaxation_time) / activation
    if required < -recruitment_tolerance:
        return _margin_failure(
            RecruitmentStatus.NEGATIVE_REQUIRED_RECRUITMENT,
            required_effective_recruitment=required,
            **common,
        )
    if required < 0.0:
        # Only remove round-off at the physical zero; material negatives retain
        # the explicit status above.
        required = 0.0

    effective_maximum = effective_recruitment_from_pulse_width(
        capacity,
        max(pulse_width_max, pd0),
        pd0,
        pdt,
    )
    if effective_maximum <= recruitment_tolerance:
        return _margin_failure(
            RecruitmentStatus.ZERO_MAXIMUM_RECRUITMENT,
            required_effective_recruitment=required,
            maximum_effective_recruitment=effective_maximum,
            **common,
        )
    utilization = required / effective_maximum
    fraction_of_capacity = required / capacity
    pw_common = dict(
        required_effective_recruitment=required,
        maximum_effective_recruitment=effective_maximum,
        utilization=utilization,
        **common,
    )
    if fraction_of_capacity > 1.0 + recruitment_tolerance:
        return _margin_failure(RecruitmentStatus.RECRUITMENT_EXCEEDS_CAPACITY, **pw_common)
    if fraction_of_capacity >= 1.0 - recruitment_tolerance:
        return _margin_failure(
            RecruitmentStatus.RECRUITMENT_AT_CAPACITY_ASYMPTOTE,
            required_pulse_width=math.inf,
            **pw_common,
        )

    required_pulse_width = pd0 - pdt * math.log1p(-fraction_of_capacity)
    reconstructed_force_derivative = ding_force_derivative(
        cn=cn,
        force=force,
        effective_recruitment=required,
        tau1=tau1,
        km=km,
        tau2=tau2,
        force_length_relationship=force_length_relationship,
        force_velocity_relationship=force_velocity_relationship,
        passive_force_relationship=passive_force_relationship,
    )
    result = RecruitmentMarginResult(
        status=RecruitmentStatus.OK,
        required_pulse_width=required_pulse_width,
        reconstructed_force_derivative=reconstructed_force_derivative,
        **pw_common,
    )
    if required_pulse_width > pulse_width_max + pulse_width_tolerance:
        return RecruitmentMarginResult(
            **{**result.__dict__, "status": RecruitmentStatus.PULSE_WIDTH_LIMIT_EXCEEDED}
        )
    return result


def _casadi_vector(value, *, name: str):
    import casadi as ca

    vector = ca.vec(value)
    if vector.numel() == 0:
        raise ValueError(f"{name} must be non-empty.")
    return vector


def marginal_capacity_damage_rate_casadi(forces, alpha_a, resting_capacities):
    """Differentiable counterpart of :func:`marginal_capacity_damage_rate`.

    Symbolic inputs must be constrained to the physical domains documented by
    the numerical function. Constant resting capacities are checked eagerly.
    """

    import casadi as ca

    forces = _casadi_vector(forces, name="forces")
    alpha_a = _casadi_vector(alpha_a, name="alpha_a")
    resting_capacities = _casadi_vector(resting_capacities, name="resting_capacities")
    if forces.numel() != alpha_a.numel() or forces.numel() != resting_capacities.numel():
        raise ValueError("forces, alpha_a, and resting_capacities must have the same size.")
    if resting_capacities.is_constant():
        scales = np.asarray(ca.DM(resting_capacities), dtype=float).ravel()
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("constant resting_capacities must be finite and strictly positive.")
    return -alpha_a * forces / resting_capacities


def ding_recruitment_expressions_casadi(
    *,
    cn,
    force,
    force_derivative,
    capacity,
    tau1,
    km,
    tau2,
    pd0,
    pdt,
    pulse_width_max,
    force_length_relationship=1.0,
    force_velocity_relationship=1.0,
    passive_force_relationship=0.0,
):
    """Return differentiable ``(a_req, a_max, utilization, PW_req)`` expressions.

    Unlike :func:`ding_recruitment_margin`, symbolic expressions cannot attach
    discrete domain statuses. The caller must constrain ``force >= 0``,
    ``cn >= 0``, ``capacity > 0``, ``tau1 > 0``, ``km > 0``, ``tau2 >= 0``,
    ``pd0 >= 0``, ``pdt > 0``, ``pulse_width_max >= pd0``, finite individual
    muscle relationships, positive non-zero force-dynamics denominators, and
    ``0 <= a_req/capacity < 1``. Use the numerical function for reporting and
    domain audits.
    """

    import casadi as ca

    activation = cn / (km + cn)
    relaxation_time = tau1 + tau2 * activation
    mechanical_gain = force_length_relationship * force_velocity_relationship + passive_force_relationship
    required = (force_derivative / mechanical_gain + force / relaxation_time) / activation
    maximum = capacity * (1.0 - ca.exp(-(pulse_width_max - pd0) / pdt))
    utilization = required / maximum
    required_pulse_width = pd0 - pdt * ca.log(1.0 - required / capacity)
    return required, maximum, utilization, required_pulse_width
