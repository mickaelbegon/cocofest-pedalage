"""Solver-independent mechanics for isokinetic reduced cycling.

This module contains the sign conventions and small numerical operations used
by the isokinetic RHO formulation.  It deliberately has no Bioptim dependency:
the same functions can therefore be used to build seeds and to audit solutions
produced by any NLP backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np


TWO_PI = 2.0 * np.pi
MIN_EXTERNAL_TORQUE_EFFECTIVENESS = 1e-8
SPEED_TOLERANCE_RAD_S = 1e-9
ANGLE_TOLERANCE_RAD = 1e-8
EQUILIBRIUM_TOLERANCE = 1e-6
ENERGY_TOLERANCE_J = 1e-6
TORQUE_BOUND_TOLERANCE_NM = 1e-8
# A continuous high-accuracy replay exposes transcription error rather than
# NLP feasibility.  Two centijoules are below 1.6% of the nominal one-turn
# work and are used as the physical smoke-test threshold; the exact
# transcription work remains certified at ENERGY_TOLERANCE_J.
ROLLOUT_ENERGY_TOLERANCE_J = 2e-2


def _finite_scalar(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


@dataclass(frozen=True)
class IsokineticCyclingConfig:
    """Validated scientific inputs shared by all isokinetic solvers."""

    omega_target_rad_s: float = -TWO_PI
    energy_equivalent_torque_nm: float = 0.2
    load_torque_min_nm: float = -3.0
    load_torque_max_nm: float = 3.0
    number_of_turns: int = 1

    def __post_init__(self):
        omega = _finite_scalar(self.omega_target_rad_s, "omega_target_rad_s")
        equivalent_torque = _finite_scalar(
            self.energy_equivalent_torque_nm,
            "energy_equivalent_torque_nm",
        )
        torque_min = _finite_scalar(
            self.load_torque_min_nm, "load_torque_min_nm"
        )
        torque_max = _finite_scalar(
            self.load_torque_max_nm, "load_torque_max_nm"
        )

        if omega >= 0.0:
            raise ValueError(
                "omega_target_rad_s must be strictly negative for the current "
                "cycling convention."
            )
        if equivalent_torque < 0.0:
            raise ValueError("energy_equivalent_torque_nm cannot be negative.")
        if torque_min >= 0.0 or torque_max <= 0.0:
            raise ValueError(
                "Load-torque bounds must allow both assistance (negative) and "
                "resistance (positive)."
            )
        if torque_min >= torque_max:
            raise ValueError(
                "load_torque_min_nm must be smaller than load_torque_max_nm."
            )
        if equivalent_torque > torque_max:
            raise ValueError(
                "energy_equivalent_torque_nm cannot exceed the maximum "
                "resistive load torque."
            )
        if isinstance(self.number_of_turns, bool) or not isinstance(
            self.number_of_turns, Integral
        ):
            raise ValueError("number_of_turns must be a strictly positive integer.")
        if self.number_of_turns <= 0:
            raise ValueError("number_of_turns must be a strictly positive integer.")

        object.__setattr__(self, "omega_target_rad_s", omega)
        object.__setattr__(
            self, "energy_equivalent_torque_nm", equivalent_torque
        )
        object.__setattr__(self, "load_torque_min_nm", torque_min)
        object.__setattr__(self, "load_torque_max_nm", torque_max)
        object.__setattr__(self, "number_of_turns", int(self.number_of_turns))

    @property
    def energy_target_j(self) -> float:
        """Net work requested over the complete RHO window."""

        return self.energy_equivalent_torque_nm * TWO_PI * self.number_of_turns

    @property
    def angle_change_rad(self) -> float:
        return -TWO_PI * self.number_of_turns

    @property
    def duration_s(self) -> float:
        return abs(self.angle_change_rad / self.omega_target_rad_s)


def produced_mechanical_power(
    load_torque_nm,
    external_torque_effectiveness,
    omega_rad_s,
):
    """Return power produced against the load, ``-tau_load*b_ext*omega``.

    With the project's negative cycling speed, a positive load torque is
    resistive and produces positive work; a negative load torque is assistive
    and produces negative work.
    """

    return (
        -np.asarray(load_torque_nm)
        * np.asarray(external_torque_effectiveness)
        * np.asarray(omega_rad_s)
    )


def mechanical_equilibrium_residual(
    muscle_effectiveness,
    muscle_forces,
    external_torque_effectiveness,
    load_torque_nm,
    projected_gravity,
    projected_velocity_quadratic,
    omega_rad_s,
):
    """Evaluate the inertia-free numerator of the reduced dynamics."""

    return (
        np.dot(np.asarray(muscle_effectiveness), np.asarray(muscle_forces))
        + np.asarray(external_torque_effectiveness) * np.asarray(load_torque_nm)
        - np.asarray(projected_gravity)
        - np.asarray(projected_velocity_quadratic) * np.asarray(omega_rad_s) ** 2
    )


def inverse_load_torque_from_coefficients(
    muscle_effectiveness,
    muscle_forces,
    external_torque_effectiveness,
    projected_gravity,
    projected_velocity_quadratic,
    omega_rad_s,
    *,
    minimum_external_effectiveness: float = MIN_EXTERNAL_TORQUE_EFFECTIVENESS,
) -> float:
    """Solve the isokinetic balance for the external crank torque."""

    b_ext = _finite_scalar(
        external_torque_effectiveness, "external_torque_effectiveness"
    )
    threshold = _finite_scalar(
        minimum_external_effectiveness, "minimum_external_effectiveness"
    )
    if threshold <= 0.0:
        raise ValueError("minimum_external_effectiveness must be positive.")
    if abs(b_ext) < threshold:
        raise ValueError(
            "The reduced profile has an ill-conditioned external torque "
            f"effectiveness b_ext={b_ext:.6g}."
        )

    muscle_effectiveness = np.asarray(muscle_effectiveness, dtype=float).reshape(-1)
    forces = np.asarray(muscle_forces, dtype=float).reshape(-1)
    if muscle_effectiveness.shape != forces.shape:
        raise ValueError(
            "muscle_effectiveness and muscle_forces must have the same size."
        )
    if not np.all(np.isfinite(muscle_effectiveness)) or not np.all(
        np.isfinite(forces)
    ):
        raise ValueError("Mechanical coefficients and muscle forces must be finite.")
    gravity = _finite_scalar(projected_gravity, "projected_gravity")
    velocity = _finite_scalar(
        projected_velocity_quadratic, "projected_velocity_quadratic"
    )
    omega = _finite_scalar(omega_rad_s, "omega_rad_s")
    return float(
        (gravity + velocity * omega**2 - muscle_effectiveness @ forces) / b_ext
    )


def equilibrium_residual(
    reduced_dynamics,
    theta_rad: float,
    omega_rad_s: float,
    muscle_forces: Sequence[float],
    load_torque_nm: float,
) -> float:
    """Evaluate isokinetic balance from a ``ReducedCyclingDynamics`` profile."""

    coefficients = reduced_dynamics.coefficient_values(float(theta_rad))
    return float(
        mechanical_equilibrium_residual(
            coefficients["muscle_effectiveness"],
            muscle_forces,
            coefficients["external_torque_effectiveness"],
            load_torque_nm,
            coefficients["projected_gravity"],
            coefficients["projected_velocity_quadratic"],
            omega_rad_s,
        )
    )


def inverse_load_torque(
    reduced_dynamics,
    theta_rad: float,
    omega_rad_s: float,
    muscle_forces: Sequence[float],
    *,
    minimum_external_effectiveness: float = MIN_EXTERNAL_TORQUE_EFFECTIVENESS,
) -> float:
    """Compute the load torque which makes reduced acceleration exactly zero."""

    coefficients = reduced_dynamics.coefficient_values(float(theta_rad))
    return inverse_load_torque_from_coefficients(
        coefficients["muscle_effectiveness"],
        muscle_forces,
        coefficients["external_torque_effectiveness"],
        coefficients["projected_gravity"],
        coefficients["projected_velocity_quadratic"],
        omega_rad_s,
        minimum_external_effectiveness=minimum_external_effectiveness,
    )


def casadi_equilibrium_residual(
    reduced_dynamics,
    theta_rad,
    omega_rad_s,
    muscle_forces,
    load_torque_nm,
):
    """CasADi expression of the exact reduced-dynamics numerator."""

    from casadi import dot

    values = reduced_dynamics.coefficients.casadi(
        reduced_dynamics.kinematics.progress(theta_rad)
    )
    return (
        dot(values[reduced_dynamics._muscle_slice], muscle_forces)
        + values[reduced_dynamics._external_index] * load_torque_nm
        - values[reduced_dynamics._gravity_index]
        - values[reduced_dynamics._velocity_index] * omega_rad_s**2
    )


def casadi_inverse_load_torque(
    reduced_dynamics,
    theta_rad,
    omega_rad_s,
    muscle_forces,
):
    """CasADi expression for inverse balance.

    The profile must first be checked numerically for small ``b_ext`` values;
    a symbolic expression cannot reliably branch on that condition.
    """

    from casadi import dot

    values = reduced_dynamics.coefficients.casadi(
        reduced_dynamics.kinematics.progress(theta_rad)
    )
    return (
        values[reduced_dynamics._gravity_index]
        + values[reduced_dynamics._velocity_index] * omega_rad_s**2
        - dot(values[reduced_dynamics._muscle_slice], muscle_forces)
    ) / values[reduced_dynamics._external_index]


def external_torque_effectiveness(reduced_dynamics, theta_rad) -> np.ndarray:
    """Sample ``b_ext(theta)`` from a reduced profile."""

    theta = np.asarray(theta_rad, dtype=float)
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta_rad must contain only finite values.")
    values = np.array(
        [
            reduced_dynamics.coefficient_values(value)[
                "external_torque_effectiveness"
            ]
            for value in theta.reshape(-1)
        ],
        dtype=float,
    ).reshape(theta.shape)
    return values


def validate_external_torque_effectiveness(
    reduced_dynamics,
    theta_rad,
    *,
    minimum_external_effectiveness: float = MIN_EXTERNAL_TORQUE_EFFECTIVENESS,
) -> float:
    """Reject a profile whose load projection is singular on sampled angles."""

    threshold = _finite_scalar(
        minimum_external_effectiveness, "minimum_external_effectiveness"
    )
    if threshold <= 0.0:
        raise ValueError("minimum_external_effectiveness must be positive.")
    effectiveness = external_torque_effectiveness(reduced_dynamics, theta_rad)
    minimum = float(np.min(effectiveness))
    if not np.all(np.isfinite(effectiveness)) or minimum < threshold:
        raise ValueError(
            "The reduced profile must have a strictly positive external "
            f"torque effectiveness: min(b_ext)={minimum:.6g}."
        )
    return minimum


def build_energy_seed(
    time_s,
    load_torque_nm,
    external_torque_effectiveness_values,
    omega_target_rad_s: float,
    *,
    initial_energy_j: float = 0.0,
) -> np.ndarray:
    """Integrate produced power by cumulative trapezoidal quadrature.

    ``load_torque_nm`` may contain one nodal value per time point or one
    piecewise-constant value per interval.  ``b_ext`` is always sampled at the
    state nodes, so its variation with crank angle is retained.
    """

    time = np.asarray(time_s, dtype=float).reshape(-1)
    b_ext = np.asarray(
        external_torque_effectiveness_values, dtype=float
    ).reshape(-1)
    torque = np.asarray(load_torque_nm, dtype=float).reshape(-1)
    if torque.size == 1:
        torque = np.full(time.size, torque.item())
    omega = _finite_scalar(omega_target_rad_s, "omega_target_rad_s")
    initial_energy = _finite_scalar(initial_energy_j, "initial_energy_j")
    if time.size < 2:
        raise ValueError("At least two time nodes are required for quadrature.")
    if b_ext.size != time.size:
        raise ValueError("b_ext must contain one value per time node.")
    if torque.size not in (time.size, time.size - 1):
        raise ValueError(
            "load_torque_nm must contain one value per node or per interval."
        )
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(b_ext)) or not np.all(
        np.isfinite(torque)
    ):
        raise ValueError("Quadrature inputs must contain only finite values.")
    delta_time = np.diff(time)
    if np.any(delta_time <= 0.0):
        raise ValueError("time_s must be strictly increasing.")

    if torque.size == time.size:
        nodal_power = produced_mechanical_power(torque, b_ext, omega)
        interval_work = (
            0.5 * (nodal_power[:-1] + nodal_power[1:]) * delta_time
        )
    else:
        interval_work = (
            -torque * omega * 0.5 * (b_ext[:-1] + b_ext[1:]) * delta_time
        )
    return np.concatenate(
        (np.array([initial_energy]), initial_energy + np.cumsum(interval_work))
    )


def build_energy_seed_from_profile(
    reduced_dynamics,
    time_s,
    theta_rad,
    load_torque_nm,
    omega_target_rad_s: float,
    *,
    initial_energy_j: float = 0.0,
) -> np.ndarray:
    """Build an energy seed while sampling ``b_ext(theta)`` from the profile."""

    return build_energy_seed(
        time_s,
        load_torque_nm,
        external_torque_effectiveness(reduced_dynamics, theta_rad),
        omega_target_rad_s,
        initial_energy_j=initial_energy_j,
    )


def audit_isokinetic_trajectory(
    config: IsokineticCyclingConfig,
    reduced_dynamics,
    time_s,
    theta_rad,
    omega_rad_s,
    muscle_forces,
    load_torque_nm,
    energy_prod_j,
) -> dict:
    """Independently audit speed, balance, work, angle and torque bounds."""

    time = np.asarray(time_s, dtype=float).reshape(-1)
    theta = np.asarray(theta_rad, dtype=float).reshape(-1)
    omega = np.asarray(omega_rad_s, dtype=float).reshape(-1)
    energy = np.asarray(energy_prod_j, dtype=float).reshape(-1)
    torque = np.asarray(load_torque_nm, dtype=float).reshape(-1)
    if torque.size == 1:
        torque = np.full(time.size, torque.item())
    forces = np.asarray(muscle_forces, dtype=float)
    if time.size < 2 or theta.size != time.size or omega.size != time.size:
        raise ValueError("time, theta and omega must share at least two nodes.")
    if energy.size != time.size:
        raise ValueError("energy_prod_j must contain one value per state node.")
    if torque.size not in (time.size, time.size - 1):
        raise ValueError("load_torque_nm must be nodal or interval-wise.")
    muscle_count = len(reduced_dynamics.muscle_names)
    if forces.shape != (muscle_count, time.size):
        raise ValueError(
            f"muscle_forces must have shape ({muscle_count}, {time.size})."
        )

    finite = bool(
        np.all(np.isfinite(time))
        and np.all(np.isfinite(theta))
        and np.all(np.isfinite(omega))
        and np.all(np.isfinite(energy))
        and np.all(np.isfinite(torque))
        and np.all(np.isfinite(forces))
    )
    if not finite:
        return {
            "finite": False,
            "maximum_speed_error_rad_s": float("inf"),
            "terminal_angle_error_rad": float("inf"),
            "maximum_equilibrium_residual": float("inf"),
            "minimum_external_torque_effectiveness": float("nan"),
            "minimum_absolute_external_torque_effectiveness": float("nan"),
            "quadrature_energy_j": float("nan"),
            "terminal_energy_target_error_j": float("inf"),
            "quadrature_energy_target_error_j": float("inf"),
            "maximum_energy_state_quadrature_error_j": float("inf"),
            "maximum_load_torque_bound_violation_nm": float("inf"),
            "passes_tolerance": False,
        }

    b_ext = external_torque_effectiveness(reduced_dynamics, theta)
    # The inverse balance requires a positive, orientation-preserving load
    # projection.  An absolute value would silently accept a sign-changing
    # or consistently reversed profile.
    minimum_b_ext = float(np.min(b_ext))
    torque_nodes = torque if torque.size == time.size else torque
    equilibrium_node_count = torque_nodes.size
    residuals = np.array(
        [
            equilibrium_residual(
                reduced_dynamics,
                theta[node],
                omega[node],
                forces[:, node],
                torque_nodes[node],
            )
            for node in range(equilibrium_node_count)
        ]
    )
    energy_quadrature = build_energy_seed(
        time, torque, b_ext, config.omega_target_rad_s
    )
    lower_violation = np.maximum(config.load_torque_min_nm - torque, 0.0)
    upper_violation = np.maximum(torque - config.load_torque_max_nm, 0.0)
    maximum_torque_violation = float(
        max(np.max(lower_violation), np.max(upper_violation))
    )

    maximum_speed_error = float(
        np.max(np.abs(omega - config.omega_target_rad_s))
    )
    terminal_angle_error = float(
        abs(theta[-1] - (theta[0] + config.angle_change_rad))
    )
    maximum_equilibrium_residual = float(np.max(np.abs(residuals)))
    terminal_energy_error = float(abs(energy[-1] - config.energy_target_j))
    quadrature_energy_error = float(
        abs(energy_quadrature[-1] - config.energy_target_j)
    )
    state_quadrature_error = float(np.max(np.abs(energy - energy_quadrature)))
    finite_outputs = bool(
        np.all(np.isfinite(b_ext))
        and np.all(np.isfinite(residuals))
        and np.all(np.isfinite(energy_quadrature))
    )
    passes = bool(
        finite_outputs
        and minimum_b_ext >= MIN_EXTERNAL_TORQUE_EFFECTIVENESS
        and maximum_speed_error <= SPEED_TOLERANCE_RAD_S
        and terminal_angle_error <= ANGLE_TOLERANCE_RAD
        and maximum_equilibrium_residual <= EQUILIBRIUM_TOLERANCE
        and terminal_energy_error <= ENERGY_TOLERANCE_J
        and quadrature_energy_error <= ENERGY_TOLERANCE_J
        and state_quadrature_error <= ENERGY_TOLERANCE_J
        and maximum_torque_violation <= TORQUE_BOUND_TOLERANCE_NM
    )
    return {
        "finite": finite_outputs,
        "maximum_speed_error_rad_s": maximum_speed_error,
        "terminal_angle_error_rad": terminal_angle_error,
        "maximum_equilibrium_residual": maximum_equilibrium_residual,
        "minimum_external_torque_effectiveness": minimum_b_ext,
        # Retained for result-file compatibility.  Since admissible profiles
        # are strictly positive, it is numerically identical to the signed
        # minimum above.
        "minimum_absolute_external_torque_effectiveness": minimum_b_ext,
        "quadrature_energy_j": float(energy_quadrature[-1]),
        "terminal_energy_target_error_j": terminal_energy_error,
        "quadrature_energy_target_error_j": quadrature_energy_error,
        "maximum_energy_state_quadrature_error_j": state_quadrature_error,
        "maximum_load_torque_bound_violation_nm": maximum_torque_violation,
        "passes_tolerance": passes,
    }
