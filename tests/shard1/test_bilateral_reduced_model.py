"""Bilateral actuator symmetry on the shared reference mechanical profile."""

import numpy as np
import pytest
from types import SimpleNamespace
from bioptim import ConfigureVariables, DynamicsEvaluation, DynamicsFunctions, OdeSolver
from casadi import DM, Function, SX, vertcat

from cocofest.dynamics.reduced_cycling import (
    PeriodicFourierSeries, ReducedCyclingDynamics, ReducedCyclingKinematics,
)
from cocofest.models.ding2007.ding2007_with_fatigue import DingModelPulseWidthFrequencyWithFatigue
from cocofest.models.reduced_cycling_model import (
    ReducedFesCyclingModel, duplicate_bilateral_muscles, make_bilateral_reduced_dynamics,
)
from cocofest.optimization.configured_cycling_model import FIELD_ATTRIBUTES


NAMES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")


def _profile(direction=-1):
    rng = np.random.default_rng(281)
    def series(offset, scale=0.02):
        return PeriodicFourierSeries(
            np.asarray(offset), rng.normal(size=(len(offset), 3)) * scale,
            rng.normal(size=(len(offset), 3)) * scale,
        )
    return ReducedCyclingDynamics(
        kinematics=ReducedCyclingKinematics(
            theta_origin=0.31, direction=direction, winding_numbers=np.array([0., 0., 1.]),
            periodic_residual=series([0., 0., 0.]),
        ),
        coefficients=series([2., 0.3, 0.1, 0.2, -0.1, 0.3, -0.3, 1.]),
        muscle_names=NAMES, crank_torque_dof_index=2,
        muscle_geometry=series([1., 1.1, 0.9, 1.05, 0.02, -0.03, 0.04, -0.01], 0.002),
    )


def _muscles():
    return [DingModelPulseWidthFrequencyWithFatigue(muscle_name=name, stim_time=[0., .1])
            for name in NAMES]


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("offset", [np.pi, 0., 0.71])
def test_phase_shift_is_exact_for_forces_geometry_and_shared_mechanics(direction, offset):
    unilateral = _profile(direction)
    bilateral = make_bilateral_reduced_dynamics(unilateral, phase_offset_rad=offset)
    assert bilateral.muscle_names == tuple(f"{side}_{name}" for side in ("right", "left") for name in NAMES)
    for theta in np.linspace(-8., 3., 9):
        right = unilateral.coefficient_values(theta)
        left = unilateral.coefficient_values(theta + offset)
        both = bilateral.coefficient_values(theta)
        for key in ("effective_inertia", "projected_gravity", "projected_velocity_quadratic",
                    "external_torque_effectiveness"):
            assert both[key] == right[key]
        np.testing.assert_allclose(both["muscle_effectiveness"], np.r_[right["muscle_effectiveness"], left["muscle_effectiveness"]])
        for actual, right_law, left_law in zip(
            bilateral.muscle_relationships(theta, -4.),
            unilateral.muscle_relationships(theta, -4.),
            unilateral.muscle_relationships(theta + offset, -4.),
        ):
            np.testing.assert_allclose(actual, np.r_[right_law, left_law], atol=1e-13)
        forces = np.arange(1., 9.)
        expected = (right["muscle_effectiveness"] @ forces[:4] + left["muscle_effectiveness"] @ forces[4:]
                    + 0.4 * right["external_torque_effectiveness"] - right["projected_gravity"]
                    - 16. * right["projected_velocity_quadratic"]) / right["effective_inertia"]
        assert bilateral.acceleration(theta, -4., forces, external_crank_torque=.4) == pytest.approx(expected)


def test_symbolic_bilateral_dynamics_matches_numpy_and_profile_roundtrip(tmp_path):
    bilateral = make_bilateral_reduced_dynamics(_profile(), phase_offset_rad=.7)
    theta, omega, forces = SX.sym("theta"), SX.sym("omega"), SX.sym("forces", 8)
    fun = Function("bilateral", [theta, omega, forces], [bilateral.casadi_acceleration(theta, omega, forces, .2),
                   *bilateral.casadi_muscle_relationships(theta, omega)])
    actual = fun(-.37, -3., np.arange(8.))
    assert float(actual[0]) == pytest.approx(bilateral.acceleration(-.37, -3., np.arange(8.), external_crank_torque=.2))
    for symbolic, numeric in zip(actual[1:], bilateral.muscle_relationships(-.37, -3.)):
        np.testing.assert_allclose(np.asarray(symbolic).ravel(), numeric, atol=1e-13)
    loaded = ReducedCyclingDynamics.load(bilateral.save(tmp_path / "bilateral.npz"))
    assert loaded.muscle_names == bilateral.muscle_names
    assert loaded.acceleration(-.37, -3., np.arange(8.)) == bilateral.acceleration(-.37, -3., np.arange(8.))


def test_bilateral_ding_copies_keep_all_parameters_but_independent_histories():
    muscles = _muscles()
    muscles[0].a_scale = muscles[0].a_rest = 1234.
    copies = duplicate_bilateral_muscles(muscles)
    assert [muscle.muscle_name for muscle in copies] == list(make_bilateral_reduced_dynamics(_profile()).muscle_names)
    for index, muscle in enumerate(copies):
        assert type(muscle) is type(muscles[index % 4])
        for attribute in (*FIELD_ATTRIBUTES.values(), "a_rest"):
            assert getattr(muscle, attribute) == getattr(muscles[index % 4], attribute)
    copies[0].stim_time.append(.2)
    copies[0].previous_stim["time"].append(-.1)
    assert copies[4].stim_time == muscles[0].stim_time == [0., .1]
    assert copies[4].previous_stim["time"] == muscles[0].previous_stim["time"] == []


@pytest.mark.parametrize("isokinetic", [False, True])
@pytest.mark.parametrize("slew", [None, .0001])
def test_bilateral_dimensions_and_independent_control_names(monkeypatch, isokinetic, slew):
    model = ReducedFesCyclingModel(
        reduced_dynamics=make_bilateral_reduced_dynamics(_profile()),
        muscles_model=duplicate_bilateral_muscles(_muscles()), isokinetic=isokinetic,
        pulse_width_max_step_s=slew, pulse_width_interval_s=.1,
    )
    configured = []
    monkeypatch.setattr(ConfigureVariables, "configure_new_variable", staticmethod(
        lambda name, *args, **kwargs: configured.append(name)))
    for configure in model.state_configuration_functions:
        configure(None, None)
    assert len(configured) == model.nb_state == 42 + int(isokinetic) + (8 if slew else 0)
    assert "F_right_Biceps" in configured and "F_left_Biceps" in configured
    configured.clear()
    for configure in model.control_configuration_functions:
        configure(None, None)
    assert len(configured) == (16 if slew else 8)
    assert "last_pulse_width_right_Biceps" in configured
    assert "last_pulse_width_left_Biceps" in configured
    cls, kwargs = model.serialize()
    assert cls(**kwargs).nb_state == model.nb_state


def test_bilateral_conversion_rejects_second_duplication_and_nonfinite_phase():
    with pytest.raises(ValueError, match="unsided"):
        make_bilateral_reduced_dynamics(make_bilateral_reduced_dynamics(_profile()))
    with pytest.raises(ValueError, match="finite"):
        make_bilateral_reduced_dynamics(_profile(), phase_offset_rad=float("nan"))


@pytest.mark.parametrize("isokinetic", [False, True])
def test_bilateral_rhs_routes_each_muscle_and_combines_shared_mechanical_balance(monkeypatch, isokinetic):
    # Isolate state/control routing from the independently tested Ding RHS.
    class RoutedMuscle(DingModelPulseWidthFrequencyWithFatigue):
        def dynamics(self, time, states, pulse_width, *args, **kwargs):
            return DynamicsEvaluation(dxdt=vertcat(*[pulse_width] * 5), defects=None)
    models = [RoutedMuscle(muscle_name=name, stim_time=[0.]) for name in NAMES]
    model = ReducedFesCyclingModel(
        reduced_dynamics=make_bilateral_reduced_dynamics(_profile()),
        muscles_model=duplicate_bilateral_muscles(models), isokinetic=isokinetic,
        isokinetic_omega=-3.,
    )
    names = model.reduced_dynamics.muscle_names
    state_keys = [f"{state}_{muscle}" for muscle in names for state in models[0].name_dof]
    state_keys += ["theta", "omega"] + (["E_prod"] if isokinetic else [])
    nlp = SimpleNamespace(
        states={name: index for index, name in enumerate(state_keys)},
        controls={f"last_pulse_width_{name}": index for index, name in enumerate(names)},
        dynamics_type=SimpleNamespace(ode_solver=OdeSolver.RK4()),
    )
    monkeypatch.setattr(DynamicsFunctions, "get", staticmethod(lambda index, vector: vector[index]))
    states = np.zeros(model.nb_state)
    forces = np.arange(1., 9.)
    states[1:40:5] = forces
    states[40:42] = [-.37, -3.]
    widths = np.arange(8.) * .0001
    output = model.dynamics(0., DM(states), DM(widths), DM(), DM(), DM(), nlp)
    values = np.asarray(output.dxdt).ravel()
    assert values.size == model.nb_state
    np.testing.assert_allclose(values[:40], np.repeat(widths, 5))
    if isokinetic:
        load = float(model.required_load_torque(-.37, -3., DM(forces)))
        assert float(model.mechanical_equilibrium_residual(-.37, -3., DM(forces), load)) == pytest.approx(0., abs=1e-12)
        power = -load * float(model.external_torque_effectiveness(-.37)) * -3.
        np.testing.assert_allclose(values[40:], [-3., 0., power])
    else:
        acceleration = model.reduced_dynamics.acceleration(-.37, -3., forces)
        np.testing.assert_allclose(values[40:], [-3., acceleration])
