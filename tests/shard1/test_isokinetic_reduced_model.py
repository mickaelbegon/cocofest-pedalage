from types import SimpleNamespace

import numpy as np
import pytest
from bioptim import ConfigureVariables, DynamicsEvaluation, DynamicsFunctions, OdeSolver
from casadi import DM, vertcat

from cocofest.dynamics.reduced_cycling import (
    PeriodicFourierSeries,
    ReducedCyclingDynamics,
    ReducedCyclingKinematics,
)
from cocofest.models.ding2007.ding2007_with_fatigue import (
    DingModelPulseWidthFrequencyWithFatigue,
)
from cocofest.models.reduced_cycling_model import ReducedFesCyclingModel


MUSCLE_NAMES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")


def _reduced_dynamics():
    zero_series = PeriodicFourierSeries(
        offset=np.zeros(3),
        cosine=np.zeros((3, 1)),
        sine=np.zeros((3, 1)),
    )
    kinematics = ReducedCyclingKinematics(
        theta_origin=0.0,
        direction=-1,
        winding_numbers=np.array([0.0, 0.0, 1.0]),
        periodic_residual=zero_series,
    )
    # M, g, c, four muscle effectiveness coefficients, b_ext.
    coefficients = PeriodicFourierSeries(
        offset=np.array([2.0, 3.0, 0.5, 1.0, -2.0, 0.5, 0.0, 0.25]),
        cosine=np.zeros((8, 1)),
        sine=np.zeros((8, 1)),
    )
    return ReducedCyclingDynamics(
        kinematics=kinematics,
        coefficients=coefficients,
        muscle_names=MUSCLE_NAMES,
        crank_torque_dof_index=2,
    )


def _ding_models():
    return [
        DingModelPulseWidthFrequencyWithFatigue(
            muscle_name=name,
            stim_time=[0.0],
        )
        for name in MUSCLE_NAMES
    ]


def _model(**kwargs):
    return ReducedFesCyclingModel(
        reduced_dynamics=_reduced_dynamics(),
        muscles_model=_ding_models(),
        activate_force_length_relationship=False,
        activate_force_velocity_relationship=False,
        activate_passive_force_relationship=False,
        **kwargs,
    )


def test_isokinetic_mode_adds_energy_state_and_eliminates_load_control(monkeypatch):
    configured = []

    def record_variable(name, elements, *args, **kwargs):
        configured.append((name, elements, kwargs))

    monkeypatch.setattr(
        ConfigureVariables,
        "configure_new_variable",
        staticmethod(record_variable),
    )
    model = _model(isokinetic=True)
    for configure in model.state_configuration_functions:
        configure(None, None)
    state_names = [entry[0] for entry in configured]
    configured.clear()
    for configure in model.control_configuration_functions:
        configure(None, None)
    control_names = [entry[0] for entry in configured]

    assert model.nb_state == 23
    assert state_names[-3:] == ["theta", "omega", "E_prod"]
    assert len(state_names) == 23
    assert "tau_load" not in control_names
    assert len(control_names) == 4

    _, serialized = model.serialize()
    assert serialized["isokinetic"] is True
    assert serialized["isokinetic_omega"] == pytest.approx(-2.0 * np.pi)


def test_default_mode_retains_legacy_dimensions_configuration_and_acceleration(
    monkeypatch,
):
    configured = []

    def record_variable(name, elements, *args, **kwargs):
        configured.append(name)

    monkeypatch.setattr(
        ConfigureVariables,
        "configure_new_variable",
        staticmethod(record_variable),
    )
    model = _model()
    for configure in model.state_configuration_functions:
        configure(None, None)
    state_names = configured.copy()
    configured.clear()
    for configure in model.control_configuration_functions:
        configure(None, None)

    assert model.nb_state == 22
    assert state_names[-2:] == ["theta", "omega"]
    assert "E_prod" not in state_names
    assert "tau_load" not in configured
    assert len(configured) == 4

    dynamics_model = _fake_dynamics_model()
    output = _evaluate_dynamics(
        monkeypatch, dynamics_model, omega=-4.0, tau_load=None
    )
    forces = np.array([5.0, 6.0, 7.0, 8.0])
    expected_acceleration = dynamics_model.reduced_dynamics.acceleration(
        -0.7,
        -4.0,
        forces,
        external_crank_torque=0.0,
    )
    np.testing.assert_array_equal(output[-2:], np.array([-4.0, expected_acceleration]))


def test_isokinetic_dynamics_has_prescribed_speed_and_exact_inverse_work_rate(
    monkeypatch,
):
    model = _fake_dynamics_model(isokinetic=True, isokinetic_omega=-6.0)

    # The omega state is retained for backward-compatible state layouts, but
    # the prescribed speed must govern every mechanical expression.
    result = _evaluate_dynamics(monkeypatch, model, omega=123.0, tau_load=None)
    forces = DM([5.0, 6.0, 7.0, 8.0])
    required_torque = float(model.required_load_torque(-0.7, -6.0, forces))
    expected_power = -required_torque * 0.25 * -6.0

    np.testing.assert_allclose(result[-3:], np.array([-6.0, 0.0, expected_power]))


def test_mechanical_equilibrium_residual_is_acceleration_numerator():
    model = _model(isokinetic=True)
    theta = -0.7
    omega = -2.0
    forces = DM([4.0, 1.0, 2.0, 9.0])
    equilibrium_load = 8.0

    residual = float(
        model.mechanical_equilibrium_residual(
            theta,
            omega,
            forces,
            equilibrium_load,
        )
    )
    acceleration = float(
        model.reduced_dynamics.casadi_acceleration(
            theta,
            omega,
            forces,
            equilibrium_load,
        )
    )

    assert residual == pytest.approx(0.0, abs=1e-14)
    assert residual == pytest.approx(2.0 * acceleration, abs=1e-14)


def test_isokinetic_mode_rejects_ambiguous_constant_external_torque():
    with pytest.raises(ValueError, match="cannot be used together"):
        _model(isokinetic=True, external_crank_torque=-0.2)


@pytest.mark.parametrize("omega", [0.0, 1.0])
def test_isokinetic_mode_rejects_nonnegative_speed(omega):
    with pytest.raises(ValueError, match="strictly negative"):
        _model(isokinetic=True, isokinetic_omega=omega)


class _FakeMuscle:
    name_dof = ["Cn", "F", "A", "Tau1", "Km"]
    nb_state = 5

    def __init__(self, muscle_name):
        self.muscle_name = muscle_name

    def dynamics(self, *args, **kwargs):
        return DynamicsEvaluation(dxdt=vertcat(0.0, 0.0, 0.0, 0.0, 0.0), defects=None)


def _fake_dynamics_model(**kwargs):
    return ReducedFesCyclingModel(
        reduced_dynamics=_reduced_dynamics(),
        muscles_model=[_FakeMuscle(name) for name in MUSCLE_NAMES],
        activate_force_length_relationship=False,
        activate_force_velocity_relationship=False,
        activate_passive_force_relationship=False,
        **kwargs,
    )


def _evaluate_dynamics(monkeypatch, model, *, omega, tau_load):
    monkeypatch.setattr(
        DynamicsFunctions,
        "get",
        staticmethod(lambda variable, vector: vector[variable]),
    )
    state_names = [
        f"{state}_{muscle}"
        for muscle in MUSCLE_NAMES
        for state in _FakeMuscle.name_dof
    ] + ["theta", "omega"]
    if model.isokinetic:
        state_names.append("E_prod")
    control_names = [f"last_pulse_width_{muscle}" for muscle in MUSCLE_NAMES]
    nlp = SimpleNamespace(
        states={name: index for index, name in enumerate(state_names)},
        controls={name: index for index, name in enumerate(control_names)},
        dynamics_type=SimpleNamespace(ode_solver=OdeSolver.RK4()),
    )
    states = np.zeros(len(state_names))
    states[1::5][:4] = np.array([5.0, 6.0, 7.0, 8.0])
    states[-3 if model.isokinetic else -2] = -0.7
    states[-2 if model.isokinetic else -1] = omega
    controls = np.zeros(len(control_names))
    result = model.dynamics(
        time=0.0,
        states=DM(states),
        controls=DM(controls),
        parameters=DM(),
        algebraic_states=DM(),
        numerical_data_timeseries=DM(),
        nlp=nlp,
    )
    return np.asarray(result.dxdt, dtype=float).reshape(-1)
