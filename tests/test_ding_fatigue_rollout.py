from types import SimpleNamespace
import shutil
import subprocess

import numpy as np
import pytest

from cocofest.optimization.ding_fatigue_rollout import (
    DingFatigueParameters,
    ding_fatigue_affine_map,
    ding_fatigue_affine_map_casadi,
    ding_fatigue_parameters_from_model,
    physiological_ding_fatigue_state,
    propagate_ding_fatigue,
    propagate_ding_fatigue_casadi,
    propagate_ding_fatigue_piecewise,
    propagate_ding_fatigue_piecewise_casadi,
)


@pytest.fixture
def parameters():
    # Defaults used by the Ding 2003/2007 fatigue implementations in Cocofest.
    return DingFatigueParameters(
        a_rest=4920.0,
        tau1_rest=0.060601,
        km_rest=0.137,
        alpha_a=-4.0e-2,
        alpha_tau1=2.1e-6,
        alpha_km=1.9e-6,
        tau_fat=127.0,
    )


def test_zero_force_recovers_exactly_to_rest(parameters):
    initial = np.array([3700.0, 0.090, 0.170])
    duration = 3.7
    propagated = propagate_ding_fatigue(initial, force=0.0, duration=duration, parameters=parameters)
    decay = np.exp(-duration / parameters.tau_fat)
    expected = parameters.rest_state + decay * (initial - parameters.rest_state)

    np.testing.assert_allclose(propagated, expected, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(
        propagate_ding_fatigue(parameters.rest_state, force=0.0, duration=duration, parameters=parameters),
        parameters.rest_state,
        rtol=0.0,
        atol=1e-13,
    )


def test_affine_map_is_the_documented_closed_form(parameters):
    state = np.array([4500.0, 0.071, 0.149])
    force = 82.0
    duration = 0.35
    decay, offset = ding_fatigue_affine_map(force, duration, parameters)
    expected = (
        parameters.rest_state
        + np.exp(-duration / parameters.tau_fat) * (state - parameters.rest_state)
        + parameters.alpha * force * parameters.tau_fat * (1.0 - np.exp(-duration / parameters.tau_fat))
    )

    np.testing.assert_allclose(decay * state + offset, expected, rtol=0.0, atol=1e-13)


def test_affine_input_gain_is_stable_for_a_duration_much_smaller_than_tau(parameters):
    duration = 1e-12
    force = 90.0
    decay, offset = ding_fatigue_affine_map(force, duration, parameters)
    input_gain = -parameters.tau_fat * np.expm1(-duration / parameters.tau_fat)
    expected_offset = (input_gain / parameters.tau_fat) * parameters.rest_state + input_gain * parameters.alpha * force

    np.testing.assert_allclose(offset, expected_offset, rtol=1e-15, atol=0.0)
    # The previous 1 - exp(-duration/tau) form loses material relative
    # precision at this scale; the expm1 gain remains asymptotic to duration.
    assert input_gain == pytest.approx(duration, rel=1e-12)
    assert decay == pytest.approx(np.exp(-duration / parameters.tau_fat))


def _dop853_piecewise(state, forces, durations, parameters):
    scipy_integrate = pytest.importorskip("scipy.integrate")
    current = np.asarray(state, dtype=float)
    for force, duration in zip(forces, durations, strict=True):
        solution = scipy_integrate.solve_ivp(
            lambda _time, slow_state: -(slow_state - parameters.rest_state) / parameters.tau_fat
            + parameters.alpha * force,
            (0.0, duration),
            current,
            method="DOP853",
            rtol=2e-13,
            atol=2e-14,
        )
        assert solution.success
        current = solution.y[:, -1]
    return current


@pytest.mark.parametrize("cycles", [1, 100])
def test_piecewise_rollout_matches_dop853_for_one_and_one_hundred_cycles(parameters, cycles):
    initial = np.array([4710.0, 0.065, 0.142])
    one_cycle_force = np.array([0.0, 61.0, 138.0, 44.0, 0.0])
    one_cycle_duration = np.array([0.08, 0.17, 0.11, 0.09, 0.15])
    forces = np.tile(one_cycle_force, cycles)
    durations = np.tile(one_cycle_duration, cycles)

    exact = propagate_ding_fatigue_piecewise(initial, forces, durations, parameters)
    integrated = _dop853_piecewise(initial, forces, durations, parameters)
    np.testing.assert_allclose(exact, integrated, rtol=3e-12, atol=3e-12)


def test_piecewise_rollout_stays_in_ding_physiological_domain(parameters):
    state = parameters.rest_state
    forces = np.array([0.0, 100.0, 250.0, 35.0, 0.0])
    durations = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    propagated = propagate_ding_fatigue_piecewise(state, np.tile(forces, 100), np.tile(durations, 100), parameters)

    np.testing.assert_allclose(physiological_ding_fatigue_state(propagated, parameters), propagated)
    assert propagated[0] < parameters.a_rest
    assert propagated[1] > parameters.tau1_rest
    assert propagated[2] > parameters.km_rest


@pytest.mark.parametrize(
    "call",
    [
        lambda parameters: propagate_ding_fatigue([1.0, 2.0], 0.0, 0.1, parameters),
        lambda parameters: propagate_ding_fatigue([1.0, 2.0, 3.0], -1.0, 0.1, parameters),
        lambda parameters: propagate_ding_fatigue([1.0, 2.0, 3.0], 1.0, -0.1, parameters),
        lambda parameters: propagate_ding_fatigue_piecewise([1.0, 2.0, 3.0], [1.0], [0.1, 0.2], parameters),
    ],
)
def test_numerical_rollout_rejects_invalid_domains(parameters, call):
    with pytest.raises(ValueError):
        call(parameters)


def test_parameter_domains_and_ding2007_rest_value_are_not_silently_changed(parameters):
    with pytest.raises(ValueError, match="alpha_a"):
        DingFatigueParameters(1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6, 127.0)
    with pytest.raises(ValueError, match="tau_fat"):
        DingFatigueParameters(1.0, 1.0, 1.0, -1e-6, 1e-6, 1e-6, 0.0)
    with pytest.raises(ValueError, match="A must"):
        physiological_ding_fatigue_state([parameters.a_rest + 1.0, 0.07, 0.15], parameters)

    ding2003_like = SimpleNamespace(
        a_rest=3009.0,
        tau1_rest=0.050957,
        km_rest=0.103,
        alpha_a=-0.04,
        alpha_tau1=2.1e-6,
        alpha_km=1.9e-6,
        tau_fat=127.0,
    )
    ding2007_like = SimpleNamespace(**ding2003_like.__dict__, a_scale=parameters.a_rest)
    assert ding_fatigue_parameters_from_model(ding2003_like).a_rest == pytest.approx(3009.0)
    assert ding_fatigue_parameters_from_model(ding2007_like).a_rest == pytest.approx(parameters.a_rest)


@pytest.mark.parametrize(
    ("model_module", "model_class", "a_rest_attribute"),
    [
        ("cocofest.models.ding2003.ding2003_with_fatigue", "DingModelFrequencyWithFatigue", "a_rest"),
        (
            "cocofest.models.ding2007.ding2007_with_fatigue",
            "DingModelPulseWidthFrequencyWithFatigue",
            "a_scale",
        ),
    ],
)
def test_extracted_parameters_match_the_real_cocofest_dynamics(model_module, model_class, a_rest_attribute):
    module = __import__(model_module, fromlist=[model_class])
    model = getattr(module, model_class)()
    parameters = ding_fatigue_parameters_from_model(model)
    force = 90.0
    state = np.array(
        [
            parameters.a_rest - 100.0,
            parameters.tau1_rest + 0.01,
            parameters.km_rest + 0.01,
        ]
    )
    expected_derivative = np.array(
        [model.a_dot_fun(state[0], force), model.tau1_dot_fun(state[1], force), model.km_dot_fun(state[2], force)]
    )
    rollout_derivative = -(state - parameters.rest_state) / parameters.tau_fat + parameters.alpha * force

    assert parameters.a_rest == pytest.approx(getattr(model, a_rest_attribute))
    np.testing.assert_allclose(rollout_derivative, expected_derivative, rtol=0.0, atol=1e-15)


def test_casadi_mx_value_and_gradients_match_closed_form(parameters):
    ca = pytest.importorskip("casadi")
    state = ca.MX.sym("slow_state", 3)
    force = ca.MX.sym("force")
    duration = 0.23
    expression = propagate_ding_fatigue_casadi(state, force, duration, parameters)
    function = ca.Function(
        "ding_rollout",
        [state, force],
        [expression, ca.jacobian(expression, state), ca.jacobian(expression, force)],
    )
    point = np.array([4700.0, 0.067, 0.145])
    force_value = 116.0
    value, state_jacobian, force_jacobian = function(point, force_value)
    expected = propagate_ding_fatigue(point, force_value, duration, parameters)
    decay = np.exp(-duration / parameters.tau_fat)
    gain = parameters.tau_fat * (1.0 - decay) * parameters.alpha

    np.testing.assert_allclose(np.asarray(value).ravel(), expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(state_jacobian), np.eye(3) * decay, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(force_jacobian).ravel(), gain, rtol=1e-13, atol=1e-13)


def test_casadi_sx_piecewise_and_affine_map_agree_with_numpy(parameters):
    ca = pytest.importorskip("casadi")
    state = ca.SX.sym("slow_state", 3)
    forces = [0.0, 70.0, 125.0]
    durations = [0.15, 0.12, 0.20]
    expression = propagate_ding_fatigue_piecewise_casadi(state, forces, durations, parameters)
    function = ca.Function("ding_piecewise", [state], [expression])
    point = np.array([4610.0, 0.068, 0.147])

    np.testing.assert_allclose(
        np.asarray(function(point)).ravel(),
        propagate_ding_fatigue_piecewise(point, forces, durations, parameters),
        rtol=1e-13,
        atol=1e-13,
    )
    decay, offset = ding_fatigue_affine_map_casadi(70.0, 0.12, parameters)
    np.testing.assert_allclose(
        np.asarray(ca.DM(offset)).ravel(),
        ding_fatigue_affine_map(70.0, 0.12, parameters)[1],
        rtol=1e-13,
        atol=1e-13,
    )
    assert float(decay) == pytest.approx(ding_fatigue_affine_map(70.0, 0.12, parameters)[0])


def test_casadi_rollout_can_be_generated_and_compiled(tmp_path, monkeypatch, parameters):
    ca = pytest.importorskip("casadi")
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("No C compiler available")

    state = ca.MX.sym("slow_state", 3)
    force = ca.MX.sym("force")
    function = ca.Function(
        "compiled_ding_rollout",
        [state, force],
        [propagate_ding_fatigue_casadi(state, force, 0.17, parameters)],
    )
    monkeypatch.chdir(tmp_path)
    function.generate("compiled_ding_rollout.c", {"with_header": True})
    subprocess.run(
        [compiler, "-fPIC", "-shared", "-O2", "compiled_ding_rollout.c", "-o", "compiled_ding_rollout.so"],
        check=True,
        capture_output=True,
        text=True,
    )
    compiled = ca.external("compiled_ding_rollout", str(tmp_path / "compiled_ding_rollout.so"))
    point = np.array([4700.0, 0.067, 0.145])
    np.testing.assert_allclose(
        np.asarray(compiled(point, 100.0)).ravel(),
        propagate_ding_fatigue(point, 100.0, 0.17, parameters),
        rtol=1e-12,
        atol=1e-12,
    )
