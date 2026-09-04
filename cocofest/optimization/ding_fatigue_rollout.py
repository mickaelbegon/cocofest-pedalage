r"""Exact, solver-independent propagation of Ding's slow fatigue states.

The fatigue variants of the Ding 2003 and 2007 models both implement, for
``z = (A, Tau1, Km)``, the scalar equations

.. math::

    \dot z_j = -(z_j-z_{j,rest})/\tau_{fat} + \alpha_j F.

For a force which is constant during a sample this module uses the closed-form
flow, rather than an ODE integrator.  It is consequently suitable for a small
look-ahead rollout driven by a *fixed* force profile from a certified RHO.  It
does not model the fast states, pulse widths, or a future force-sharing policy.

The signs below are the signs used in Cocofest's Ding fatigue models: ``A``
decreases with positive force (``alpha_a <= 0``), while ``Tau1`` and ``Km``
increase (their coefficients are non-negative).  These checks are deliberately
performed only for numerical parameters; symbolic callers must impose their
own physiological constraints in the enclosing NLP.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


SLOW_FATIGUE_STATE_NAMES = ("A", "Tau1", "Km")
_STATE_SIZE = len(SLOW_FATIGUE_STATE_NAMES)


def _finite_scalar(value: float, *, name: str, strictly_positive: bool = False) -> float:
    """Return a finite scalar, optionally restricted to the positive half-line."""
    value = float(value)
    if not math.isfinite(value) or (strictly_positive and value <= 0.0):
        qualifier = "finite and strictly positive" if strictly_positive else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return value


def _state_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    """Return one numerical ``(A, Tau1, Km)`` state vector."""
    vector = np.asarray(values, dtype=float)
    if vector.shape != (_STATE_SIZE,):
        raise ValueError(f"{name} must have shape ({_STATE_SIZE},), ordered as {SLOW_FATIGUE_STATE_NAMES}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


@dataclass(frozen=True)
class DingFatigueParameters:
    """Numerical parameters of the three slow Ding fatigue equations.

    ``a_rest`` denotes the resting value appearing in the fatigue equation.
    This is ``model.a_rest`` for Ding 2003 and ``model.a_scale`` for Ding 2007,
    exactly as their respective ``a_dot_fun`` implementations specify.
    """

    a_rest: float
    tau1_rest: float
    km_rest: float
    alpha_a: float
    alpha_tau1: float
    alpha_km: float
    tau_fat: float

    def __post_init__(self) -> None:
        rest = _state_vector(self.rest_state, name="rest_state")
        if np.any(rest <= 0.0):
            raise ValueError("rest_state must be strictly positive componentwise.")
        alpha = _state_vector(self.alpha, name="alpha")
        if alpha[0] > 0.0 or np.any(alpha[1:] < 0.0):
            raise ValueError("Ding fatigue coefficients require alpha_a <= 0 and alpha_tau1, alpha_km >= 0.")
        _finite_scalar(self.tau_fat, name="tau_fat", strictly_positive=True)

    @property
    def rest_state(self) -> np.ndarray:
        """Resting ``(A, Tau1, Km)`` vector in model units."""
        return np.array([self.a_rest, self.tau1_rest, self.km_rest], dtype=float)

    @property
    def alpha(self) -> np.ndarray:
        """Force coefficients ordered as ``(alpha_a, alpha_tau1, alpha_km)``."""
        return np.array([self.alpha_a, self.alpha_tau1, self.alpha_km], dtype=float)


def ding_fatigue_parameters_from_model(model: Any) -> DingFatigueParameters:
    """Extract numerical slow-fatigue parameters from a Cocofest Ding model.

    Ding 2007 inherits an ``a_rest`` attribute from Ding 2003, but its fatigue
    equation explicitly relaxes ``A`` to ``a_scale``.  Presence of ``a_scale``
    therefore selects it before considering ``a_rest``.  The extractor is kept
    duck-typed to avoid importing Bioptim or a concrete model class here.
    """
    a_rest = getattr(model, "a_scale", None)
    if a_rest is None:
        a_rest = getattr(model, "a_rest", None)
    required = {
        "a_rest/a_scale": a_rest,
        "tau1_rest": getattr(model, "tau1_rest", None),
        "km_rest": getattr(model, "km_rest", None),
        "alpha_a": getattr(model, "alpha_a", None),
        "alpha_tau1": getattr(model, "alpha_tau1", None),
        "alpha_km": getattr(model, "alpha_km", None),
        "tau_fat": getattr(model, "tau_fat", None),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"model does not expose Ding fatigue parameter(s): {', '.join(missing)}.")
    return DingFatigueParameters(
        a_rest=a_rest,
        tau1_rest=required["tau1_rest"],
        km_rest=required["km_rest"],
        alpha_a=required["alpha_a"],
        alpha_tau1=required["alpha_tau1"],
        alpha_km=required["alpha_km"],
        tau_fat=required["tau_fat"],
    )


def ding_fatigue_affine_map(
    force: float,
    duration: float,
    parameters: DingFatigueParameters,
) -> tuple[float, np.ndarray]:
    """Return ``(decay, offset)`` for one exact constant-force transition.

    The resulting map is ``z_next = decay * z + offset``.  It is exact for
    the model over ``duration`` if ``force`` is constant on that interval.
    """
    force = _finite_scalar(force, name="force")
    if force < 0.0:
        raise ValueError("force must be non-negative for a physiological Ding rollout.")
    duration = _finite_scalar(duration, name="duration")
    if duration < 0.0:
        raise ValueError("duration must be non-negative.")
    decay = math.exp(-duration / parameters.tau_fat)
    # expm1 preserves the affine input gain when duration << tau_fat.
    input_gain = -parameters.tau_fat * math.expm1(-duration / parameters.tau_fat)
    offset = (input_gain / parameters.tau_fat) * parameters.rest_state + input_gain * parameters.alpha * force
    return decay, offset


def propagate_ding_fatigue(
    state: Sequence[float] | np.ndarray,
    force: float,
    duration: float,
    parameters: DingFatigueParameters,
) -> np.ndarray:
    """Exactly propagate one numerical slow-fatigue state under constant force."""
    state = _state_vector(state, name="state")
    decay, _ = ding_fatigue_affine_map(force, duration, parameters)
    input_gain = -parameters.tau_fat * math.expm1(-duration / parameters.tau_fat)
    return parameters.rest_state + decay * (state - parameters.rest_state) + parameters.alpha * force * input_gain


def propagate_ding_fatigue_from_force_integral(
    state: Sequence[float] | np.ndarray,
    *,
    duration: float,
    exponentially_weighted_force_integral: float,
    parameters: DingFatigueParameters,
) -> np.ndarray:
    r"""Exactly propagate a slow state from ``\int e^{-(dt-s)/tau} F(s) ds``.

    This form is useful when a force is represented by a continuous periodic
    interpolant rather than by a piecewise-constant sample.  The caller must
    supply the integral over the same interval as ``duration``:

    .. math::

        I_F = \int_0^{dt}e^{-(dt-s)/\tau_{fat}}F(t+s)\,ds.

    The exact transition is ``z_next = z_rest + decay * (z-z_rest) + alpha *
    I_F``.  As in :func:`propagate_ding_fatigue`, the force-domain audit is the
    responsibility of the force representation because an integral alone does
    not reveal whether an underlying force became negative.
    """
    state = _state_vector(state, name="state")
    duration = _finite_scalar(duration, name="duration")
    if duration < 0.0:
        raise ValueError("duration must be non-negative.")
    force_integral = _finite_scalar(
        exponentially_weighted_force_integral,
        name="exponentially_weighted_force_integral",
    )
    decay = math.exp(-duration / parameters.tau_fat)
    return parameters.rest_state + decay * (state - parameters.rest_state) + parameters.alpha * force_integral


def propagate_ding_fatigue_piecewise(
    state: Sequence[float] | np.ndarray,
    forces: Sequence[float] | np.ndarray,
    durations: Sequence[float] | np.ndarray,
    parameters: DingFatigueParameters,
) -> np.ndarray:
    """Exactly propagate a numerical state through a piecewise-constant force profile.

    ``forces[k]`` is applied during ``durations[k]``.  Repeating a one-cycle
    profile is intentionally explicit (for example with ``np.tile``), making
    the assumed horizon and interpolation visible to the caller.
    """
    forces = np.asarray(forces, dtype=float)
    durations = np.asarray(durations, dtype=float)
    if forces.ndim != 1 or durations.ndim != 1 or forces.size != durations.size:
        raise ValueError("forces and durations must be one-dimensional vectors with the same length.")
    current = _state_vector(state, name="state")
    for force, duration in zip(forces, durations, strict=True):
        current = propagate_ding_fatigue(current, float(force), float(duration), parameters)
    return current


def physiological_ding_fatigue_state(
    state: Sequence[float] | np.ndarray,
    parameters: DingFatigueParameters,
    *,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Validate the physical slow-state ordering for non-negative force rollouts.

    With Cocofest's signs, a state initialized in this domain satisfies
    ``0 <= A <= A_rest``, ``Tau1 >= Tau1_rest``, and ``Km >= Km_rest``.  The
    latter two have no arbitrary upper bound in the Ding equations.
    """
    tolerance = _finite_scalar(tolerance, name="tolerance")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    state = _state_vector(state, name="state")
    rest = parameters.rest_state
    if state[0] < -tolerance or state[0] > rest[0] + tolerance:
        raise ValueError("A must lie in [0, A_rest] within tolerance.")
    if state[1] < rest[1] - tolerance or state[2] < rest[2] - tolerance:
        raise ValueError("Tau1 and Km must be at least their resting values within tolerance.")
    return state


def _casadi_module():
    """Import CasADi lazily so numerical rollouts have no symbolic dependency."""
    import casadi as ca

    return ca


def _casadi_state_vector(value, *, name: str):
    ca = _casadi_module()
    vector = ca.vec(value)
    if vector.numel() != _STATE_SIZE:
        raise ValueError(f"{name} must contain {_STATE_SIZE} elements ordered as {SLOW_FATIGUE_STATE_NAMES}.")
    return vector


def _casadi_scalar(value, *, name: str, non_negative: bool = False):
    ca = _casadi_module()
    scalar = ca.vec(value)
    if scalar.numel() != 1:
        raise ValueError(f"{name} must be scalar.")
    if scalar.is_constant():
        numeric = float(ca.DM(scalar))
        _finite_scalar(numeric, name=name)
        if non_negative and numeric < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    return scalar


def ding_fatigue_affine_map_casadi(force, duration, parameters: DingFatigueParameters):
    """CasADi version of :func:`ding_fatigue_affine_map` for SX, MX, or DM.

    Symbolic force and duration are accepted.  Their non-negative domains must
    be enforced by the enclosing optimization problem.
    """
    ca = _casadi_module()
    force = _casadi_scalar(force, name="force", non_negative=True)
    duration = _casadi_scalar(duration, name="duration", non_negative=True)
    decay = ca.exp(-duration / parameters.tau_fat)
    rest = ca.DM(parameters.rest_state)
    alpha = ca.DM(parameters.alpha)
    offset = (1.0 - decay) * rest + parameters.tau_fat * (1.0 - decay) * alpha * force
    return decay, offset


def propagate_ding_fatigue_casadi(state, force, duration, parameters: DingFatigueParameters):
    """Differentiable CasADi expression for one exact constant-force transition."""
    state = _casadi_state_vector(state, name="state")
    decay, offset = ding_fatigue_affine_map_casadi(force, duration, parameters)
    return decay * state + offset


def propagate_ding_fatigue_piecewise_casadi(
    state,
    forces: Sequence,
    durations: Sequence,
    parameters: DingFatigueParameters,
):
    """Compose fixed-count exact CasADi transitions for a force profile."""
    if len(forces) != len(durations):
        raise ValueError("forces and durations must have the same length.")
    current = _casadi_state_vector(state, name="state")
    for force, duration in zip(forces, durations, strict=True):
        current = propagate_ding_fatigue_casadi(current, force, duration, parameters)
    return current
