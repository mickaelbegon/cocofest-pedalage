from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.dynamics.reduced_cycling import PeriodicFourierSeries
from cocofest.optimization.isokinetic_cycling import (
    IsokineticCyclingConfig,
    audit_isokinetic_trajectory,
    build_energy_seed,
    equilibrium_residual,
    inverse_load_torque,
    produced_mechanical_power,
    validate_external_torque_effectiveness,
)


class _ReducedProfile:
    """Small profile exposing the public API needed by the scientific module."""

    muscle_names = ("flexor", "extensor")

    @staticmethod
    def coefficient_values(theta):
        return {
            "effective_inertia": 2.0,
            "projected_gravity": 1.1 + 0.1 * np.cos(theta),
            "projected_velocity_quadratic": 0.02,
            "muscle_effectiveness": np.array(
                [0.03 + 0.01 * np.sin(theta), -0.04]
            ),
            "external_torque_effectiveness": 1.0 + 0.2 * np.cos(theta),
        }


def test_configuration_has_one_exact_turn_energy_target():
    config = IsokineticCyclingConfig()

    assert config.energy_target_j == 0.2 * 2.0 * np.pi
    assert config.angle_change_rad == -2.0 * np.pi
    assert config.duration_s == 1.0
    assert IsokineticCyclingConfig(number_of_turns=3).energy_target_j == (
        0.2 * 2.0 * np.pi * 3
    )


def test_power_sign_distinguishes_resistance_and_assistance():
    omega = -2.0 * np.pi
    b_ext = np.array([0.8, 1.2])

    assert np.all(produced_mechanical_power(0.2, b_ext, omega) > 0.0)
    assert np.all(produced_mechanical_power(-0.2, b_ext, omega) < 0.0)


def test_energy_seed_keeps_variable_external_effectiveness():
    time = np.linspace(0.0, 1.0, 1001)
    omega = -2.0 * np.pi
    b_ext = 1.0 + time
    seed = build_energy_seed(time, 0.2, b_ext, omega)

    # Integral_0^1 0.2 * 2*pi * (1+t) dt = 0.3 * 2*pi.
    np.testing.assert_allclose(seed[-1], 0.3 * 2.0 * np.pi, atol=1e-12)
    assert seed[0] == 0.0
    assert np.all(np.diff(seed) > 0.0)


def test_inverse_load_torque_returns_zero_reduced_residual():
    profile = _ReducedProfile()
    theta = -0.7
    omega = -2.0 * np.pi
    forces = np.array([35.0, 12.0])

    torque = inverse_load_torque(profile, theta, omega, forces)

    assert equilibrium_residual(profile, theta, omega, forces, torque) == pytest.approx(
        0.0, abs=1e-14
    )


def test_inverse_load_torque_rejects_small_external_effectiveness():
    profile = _ReducedProfile()
    profile.coefficient_values = lambda theta: {
        **_ReducedProfile.coefficient_values(theta),
        "external_torque_effectiveness": 1e-12,
    }

    with pytest.raises(ValueError, match="ill-conditioned.*b_ext"):
        inverse_load_torque(profile, 0.0, -2.0 * np.pi, [1.0, 2.0])


def test_profile_validation_rejects_negative_external_effectiveness():
    profile = _ReducedProfile()
    profile.coefficient_values = lambda theta: {
        **_ReducedProfile.coefficient_values(theta),
        "external_torque_effectiveness": -1.0,
    }

    with pytest.raises(ValueError, match="strictly positive"):
        validate_external_torque_effectiveness(profile, np.linspace(0.0, -2 * np.pi, 31))


@pytest.mark.parametrize(
    "keywords, message",
    [
        ({"omega_target_rad_s": 0.0}, "strictly negative"),
        ({"omega_target_rad_s": np.nan}, "must be finite"),
        ({"energy_equivalent_torque_nm": -0.1}, "cannot be negative"),
        ({"load_torque_min_nm": 0.0}, "allow both assistance"),
        ({"load_torque_max_nm": 0.0}, "allow both assistance"),
        (
            {"energy_equivalent_torque_nm": 4.0},
            "cannot exceed the maximum resistive",
        ),
        ({"number_of_turns": 0}, "strictly positive integer"),
        ({"number_of_turns": 1.5}, "strictly positive integer"),
    ],
)
def test_configuration_rejects_invalid_inputs(keywords, message):
    with pytest.raises(ValueError, match=message):
        IsokineticCyclingConfig(**keywords)


def test_pure_numerical_audit_certifies_a_consistent_trajectory():
    profile = _ReducedProfile()
    config = IsokineticCyclingConfig(energy_equivalent_torque_nm=1.0)
    time = np.linspace(0.0, config.duration_s, 2001)
    theta = config.omega_target_rad_s * time
    omega = np.full(time.size, config.omega_target_rad_s)
    forces = np.vstack((20.0 + np.sin(theta), 15.0 + np.cos(theta)))
    torque = np.array(
        [
            inverse_load_torque(
                profile, theta_value, config.omega_target_rad_s, forces[:, node]
            )
            for node, theta_value in enumerate(theta)
        ]
    )
    b_ext = np.array(
        [
            profile.coefficient_values(theta_value)[
                "external_torque_effectiveness"
            ]
            for theta_value in theta
        ]
    )
    energy = build_energy_seed(
        time, torque, b_ext, config.omega_target_rad_s
    )
    calibrated_config = IsokineticCyclingConfig(
        energy_equivalent_torque_nm=energy[-1] / (2.0 * np.pi),
        load_torque_min_nm=-10.0,
        load_torque_max_nm=10.0,
    )

    audit = audit_isokinetic_trajectory(
        calibrated_config,
        profile,
        time,
        theta,
        omega,
        forces,
        torque,
        energy,
    )

    assert audit["passes_tolerance"] is True
    assert audit["maximum_equilibrium_residual"] < 1e-12
    assert audit["quadrature_energy_target_error_j"] < 1e-12


def test_casadi_inverse_torque_matches_symbolic_balance():
    casadi = pytest.importorskip("casadi")
    from cocofest.optimization.isokinetic_cycling import (
        casadi_equilibrium_residual,
        casadi_inverse_load_torque,
    )

    phase = np.linspace(0.0, 2.0 * np.pi, 31)
    profile = SimpleNamespace(
        coefficients=PeriodicFourierSeries.fit(
            phase,
            np.vstack(
                (
                    np.full(phase.size, 2.0),
                    1.0 + 0.1 * np.cos(phase),
                    np.full(phase.size, 0.02),
                    0.03 + 0.01 * np.sin(phase),
                    np.full(phase.size, -0.04),
                    1.0 + 0.2 * np.cos(phase),
                )
            ),
            order=2,
        ),
        kinematics=SimpleNamespace(progress=lambda theta: -theta),
        _muscle_slice=slice(3, 5),
        _external_index=5,
        _gravity_index=1,
        _velocity_index=2,
    )
    theta = casadi.MX.sym("theta")
    forces = casadi.MX.sym("forces", 2)
    torque = casadi_inverse_load_torque(profile, theta, -2.0 * np.pi, forces)
    residual = casadi_equilibrium_residual(
        profile, theta, -2.0 * np.pi, forces, torque
    )
    function = casadi.Function("inverse_balance", [theta, forces], [residual])

    assert float(function(-0.4, [30.0, 10.0])) == pytest.approx(0.0, abs=1e-14)
