"""Projected local mechanical reserve under a repeated candidate force profile.

Only the slow Ding equations are projected. A local affine model of mechanical
margins is supplied by the caller and refreshed *outside* the NLP. This is a
surrogate, not a certificate of future pulse-width or movement feasibility.
Forces are piecewise constant; their temporal ordering is retained exactly.
No future controls or additional optimization problem are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .ding_fatigue_rollout import DingFatigueParameters


def _finite_array(value, name):
    value = np.array(value, dtype=float, copy=True)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _temperature(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("temperature must be finite and strictly positive.")
    return value


def _projection_constants(durations, parameters, horizons):
    durations = _finite_array(durations, "durations")
    if durations.ndim != 1 or not durations.size or np.any(durations <= 0):
        raise ValueError("durations must be a nonempty vector of strictly positive intervals.")
    parameters = tuple(parameters)
    if not parameters or any(not isinstance(p, DingFatigueParameters) for p in parameters):
        raise ValueError("parameters must contain one DingFatigueParameters per muscle.")
    horizons = tuple(horizons)
    if not horizons or any(isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer))
                           or h < 1 for h in horizons):
        raise ValueError("horizons must be nonempty positive integer cycle counts.")
    if any(b <= a for a, b in zip(horizons, horizons[1:])):
        raise ValueError("horizons must be strictly increasing.")
    period = float(np.sum(durations))
    if not math.isfinite(period):
        raise ValueError("The total cycle duration must be finite.")
    # Contribution of each interval at the *end* of the cycle, without the
    # cancellation in (1-exp(-dt/tau)) for very short intervals.
    remaining = np.cumsum(durations[::-1])[::-1] - durations
    weights = np.array([
        -p.tau_fat * np.expm1(-durations / p.tau_fat) * np.exp(-remaining / p.tau_fat)
        for p in parameters
    ])
    return durations, parameters, horizons, period, weights


def _validate_states(states, parameters, name):
    states = _finite_array(states, name)
    if states.shape != (len(parameters), 3):
        raise ValueError(f"{name} must have shape (muscles, 3), ordered A, Tau1, Km.")
    if np.any(states <= 0):
        raise ValueError(f"{name} must be strictly positive; a depleted A is outside this surrogate's domain.")
    return states


def project_repeated_force_states(initial_states, forces, durations, parameters, horizons):
    """Return endpoint states with shape ``(horizons, muscles, 3)``.

    ``initial_states`` is the state *before* the first candidate cycle;
    ``forces[m,k]`` acts over ``durations[k]``. Horizon 1 therefore applies
    the candidate cycle exactly once. The exact cycle map is raised to each
    requested integer horizon analytically, without unrolling future cycles.
    Durations must span one physical cycle, e.g. sum to 2*pi/abs(omega) for
    isokinetic cycling; no one-second cycle is assumed.
    Numerical projections leaving positive A/Tau1/Km raise ``ValueError``.
    """
    durations, parameters, horizons, period, weights = _projection_constants(durations, parameters, horizons)
    initial_states = _validate_states(initial_states, parameters, "initial_states")
    forces = _finite_array(forces, "forces")
    if forces.shape != (len(parameters), durations.size) or np.any(forces < 0):
        raise ValueError("forces must have shape (muscles, intervals) and be nonnegative.")
    states = np.empty((len(horizons), len(parameters), 3))
    for muscle, p in enumerate(parameters):
        integral = np.dot(weights[muscle], forces[muscle])
        cycle_fraction = -math.expm1(-period / p.tau_fat)
        for index, horizon in enumerate(horizons):
            fraction = -math.expm1(-horizon * period / p.tau_fat)
            states[index, muscle] = (
                p.rest_state + math.exp(-horizon * period / p.tau_fat) * (initial_states[muscle] - p.rest_state)
                + p.alpha * integral * (fraction / cycle_fraction)
            )
    for state in states:
        _validate_states(state, parameters, "projected_states")
    return states


def _casadi_matrix(value, name, shape, *, positive=False, nonnegative=False):
    import casadi as ca

    value = value if isinstance(value, (ca.MX, ca.SX, ca.DM)) else ca.DM(value)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if value.is_constant():
        numeric = np.asarray(ca.evalf(value), dtype=float)
        if not np.all(np.isfinite(numeric)) or (positive and np.any(numeric <= 0)) or (
            nonnegative and np.any(numeric < 0)
        ):
            raise ValueError(f"{name} contains nonfinite or out-of-domain values.")
    return value


def project_repeated_force_states_casadi(initial_states, forces, durations, parameters, horizons):
    """Return one ``(muscles, 3)`` SX/MX/DM matrix per requested horizon.

    Parameters, durations and horizons are numerical constants. The enclosing
    NLP must constrain symbolic force >= 0 and initial/projected A,Tau1,Km > 0;
    symbolic domains cannot be validated here. No clipping hides depletion.
    """
    import casadi as ca

    durations, parameters, horizons, period, weights = _projection_constants(durations, parameters, horizons)
    initial_states = _casadi_matrix(initial_states, "initial_states", (len(parameters), 3), positive=True)
    forces = _casadi_matrix(forces, "forces", (len(parameters), durations.size), nonnegative=True)
    integrals = [ca.mtimes(forces[m, :], ca.DM(weights[m])) for m in range(len(parameters))]
    states = []
    for horizon in horizons:
        rows = []
        for m, p in enumerate(parameters):
            fraction = -math.expm1(-horizon * period / p.tau_fat)
            geometric_sum = fraction / -math.expm1(-period / p.tau_fat)
            rest = ca.DM(p.rest_state).T
            rows.append(rest + math.exp(-horizon * period / p.tau_fat) * (initial_states[m, :] - rest)
                        + ca.DM(p.alpha).T * integrals[m] * geometric_sum)
        states.append(_casadi_matrix(ca.vertcat(*rows), "projected_states", (len(parameters), 3), positive=True))
    return tuple(states)


@dataclass(frozen=True)
class LocalMechanicalMarginModel:
    """Frozen local linearization, to be reconstructed outside the NLP.

    ``reference_states``: (muscles, 3), in physical A/Tau1/Km units.
    ``reference_margins``: (constraints,), where positive means reserve.
    ``state_jacobian``: (constraints, muscles, 3), derivatives of margins with
    respect to physical slow states. Entries may have either sign: in
    particular the mechanical effect of Tau1 is not assumed a priori.
    All margins must use a common scale (e.g. dimensionless torque slack).
    The caller defines the mechanical sign explicitly. With the reduced
    model's moment = b_i*F_i, produced power uses omega*b_i*F_i, not its
    negation, and should be audited against the model's E_prod_dot.
    This object deliberately accepts only numerical coefficients, with no
    differentiation through the procedure that estimates the linearization.
    """

    reference_states: np.ndarray
    reference_margins: np.ndarray
    state_jacobian: np.ndarray

    def __post_init__(self):
        states = _finite_array(self.reference_states, "reference_states")
        margins = _finite_array(self.reference_margins, "reference_margins")
        jacobian = _finite_array(self.state_jacobian, "state_jacobian")
        if states.ndim != 2 or states.shape[1] != 3 or not states.shape[0] or np.any(states <= 0):
            raise ValueError("reference_states must be positive with shape (muscles, 3).")
        if margins.ndim != 1 or not margins.size:
            raise ValueError("reference_margins must be a nonempty vector.")
        if jacobian.shape != (margins.size, *states.shape):
            raise ValueError("state_jacobian must have shape (constraints, muscles, 3).")
        for name, value in (("reference_states", states), ("reference_margins", margins), ("state_jacobian", jacobian)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def evaluate(self, projected_states):
        """Return local margins with shape (horizons, constraints)."""
        states = _finite_array(projected_states, "projected_states")
        if states.ndim != 3 or states.shape[1:] != self.reference_states.shape or not states.shape[0]:
            raise ValueError("projected_states must have shape (horizons, muscles, 3).")
        if np.any(states <= 0):
            raise ValueError("projected_states must be strictly positive.")
        return self.reference_margins + np.einsum("cmj,hmj->hc", self.state_jacobian, states - self.reference_states)

    def evaluate_casadi(self, projected_states):
        """Return an (horizons, constraints) expression from projected matrices."""
        import casadi as ca

        if not len(projected_states):
            raise ValueError("projected_states must be nonempty.")
        rows = []
        jacobian = ca.DM(self.state_jacobian.reshape(self.reference_margins.size, -1))
        for state in projected_states:
            state = _casadi_matrix(state, "projected_states", self.reference_states.shape, positive=True)
            # NumPy flattening is muscle-major; CasADi vec is column-major.
            delta = ca.vec((state - ca.DM(self.reference_states)).T)
            rows.append((ca.DM(self.reference_margins) + ca.mtimes(jacobian, delta)).T)
        return ca.vertcat(*rows)


def soft_maximum(values, *, temperature):
    """Stable unnormalized log-sum-exp, >= hard max and exact for one value."""
    values = _finite_array(values, "values").ravel()
    temperature = _temperature(temperature)
    if not values.size:
        raise ValueError("values must be nonempty.")
    shift = float(np.max(values))
    return float(shift + temperature * np.log(np.sum(np.exp((values - shift) / temperature))))


def soft_minimum(values, *, temperature):
    """Conservative smooth minimum; bias is at most temperature * log(n)."""
    return -soft_maximum(-np.asarray(values, dtype=float), temperature=temperature)


def soft_maximum_casadi(values, *, temperature):
    """Stable CasADi log-sum-exp supporting SX, MX and DM, including ties."""
    import casadi as ca

    temperature = _temperature(temperature)
    values = ca.vec(values)
    if not values.numel():
        raise ValueError("values must be nonempty.")
    _casadi_matrix(values, "values", values.shape)
    shift = ca.mmax(values)
    return shift + temperature * ca.logsumexp((values - shift) / temperature)


def soft_minimum_casadi(values, *, temperature):
    import casadi as ca

    return -soft_maximum_casadi(-ca.vec(values), temperature=temperature)


@dataclass(frozen=True)
class ProjectedMechanicalReserve:
    """Intermediate values for auditing, plus the scalar objective penalty.

    ``horizon_reserves`` is a conservative soft minimum of margins for each
    horizon. ``penalty`` is the soft maximum of their negations; a larger
    penalty means less reserve. The objective is not clipped at zero.
    ``hard_minimum_margin`` audits the weakest affine margin across all
    horizons and constraints, independently of smoothing. Neither value is
    a certificate for the nonlinear mechanical system.
    """

    states: Any
    margins: Any
    horizon_reserves: Any
    penalty: Any
    hard_minimum_margin: Any


def projected_mechanical_reserve(initial_states, forces, durations, parameters, horizons, margin_model,
                                 *, margin_temperature=0.02, horizon_temperature=0.02):
    """Numerical projected reserve using externally supplied local mechanics."""
    states = project_repeated_force_states(initial_states, forces, durations, parameters, horizons)
    margins = margin_model.evaluate(states)
    reserves = np.array([soft_minimum(row, temperature=margin_temperature) for row in margins])
    penalty = soft_maximum(-reserves, temperature=horizon_temperature)
    return ProjectedMechanicalReserve(states, margins, reserves, penalty, float(np.min(margins)))


def projected_mechanical_reserve_casadi(initial_states, forces, durations, parameters, horizons, margin_model,
                                        *, margin_temperature=0.02, horizon_temperature=0.02):
    """Differentiable objective with the same values and sign as NumPy."""
    import casadi as ca

    states = project_repeated_force_states_casadi(initial_states, forces, durations, parameters, horizons)
    margins = margin_model.evaluate_casadi(states)
    reserves = ca.vertcat(*[soft_minimum_casadi(margins[h, :], temperature=margin_temperature)
                           for h in range(margins.size1())])
    penalty = soft_maximum_casadi(-reserves, temperature=horizon_temperature)
    return ProjectedMechanicalReserve(states, margins, reserves, penalty, ca.mmin(margins))
