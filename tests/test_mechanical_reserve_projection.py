"""Isolated numerical/symbolic tests; no simulator or solver dependencies."""

import numpy as np
import pytest

from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters, propagate_ding_fatigue_piecewise
from cocofest.optimization.mechanical_reserve_projection import (
    LocalMechanicalMarginModel,
    project_repeated_force_states,
    project_repeated_force_states_casadi,
    projected_mechanical_reserve,
    projected_mechanical_reserve_casadi,
    soft_maximum,
    soft_maximum_casadi,
    soft_minimum,
)


@pytest.fixture
def case():
    parameters = (
        DingFatigueParameters(4920., .060601, .137, -.04, 2.1e-6, 1.9e-6, 127.),
        DingFatigueParameters(4100., .070, .12, -.035, 2.8e-6, 1.5e-6, 110.),
    )
    states = np.array([p.rest_state for p in parameters])
    forces = np.array([[40., 110., 80.], [60., 55., 120.]])
    durations = np.array([.07, .19, .37])
    jacobian = np.array([[[1e-4, -1., -.5], [2e-5, -.3, -.1]],
                         [[2e-5, -.2, -.4], [1e-4, -.8, -.6]]])
    margins = LocalMechanicalMarginModel(states, [.5, .6], jacobian)
    return states, forces, durations, parameters, margins


def test_projection_matches_explicit_repetition_including_horizon_one(case):
    states, forces, durations, parameters, _ = case
    horizons = (1, 2, 19, 113)
    projected = project_repeated_force_states(states, forces, durations, parameters, horizons)
    assert projected.shape == (4, 2, 3)
    for h_index, horizon in enumerate(horizons):
        for muscle, p in enumerate(parameters):
            expected = propagate_ding_fatigue_piecewise(
                states[muscle], np.tile(forces[muscle], horizon), np.tile(durations, horizon), p)
            np.testing.assert_allclose(projected[h_index, muscle], expected, rtol=2e-13, atol=1e-11)


def test_projection_preserves_time_order_and_is_invariant_to_interval_subdivision(case):
    states, forces, durations, parameters, _ = case
    base = project_repeated_force_states(states, forces, durations, parameters, [1, 37])
    split = project_repeated_force_states(states, np.repeat(forces, 2, axis=1),
                                          np.repeat(durations / 2, 2), parameters, [1, 37])
    np.testing.assert_allclose(base, split, rtol=2e-15, atol=1e-12)
    reversed_profile = project_repeated_force_states(states, forces[:, ::-1], durations[::-1], parameters, [1, 37])
    assert np.max(np.abs(base - reversed_profile)) > .001


def test_zero_force_recovers_and_duration_is_not_assumed_one_second(case):
    states, forces, durations, parameters, _ = case
    initial = states * [0.8, 1.2, 1.1]
    projected = project_repeated_force_states(initial, np.zeros_like(forces), durations, parameters, [1, 100])
    for index, horizon in enumerate([1, 100]):
        for m, p in enumerate(parameters):
            expected = p.rest_state + np.exp(-horizon * durations.sum() / p.tau_fat) * (initial[m] - p.rest_state)
            np.testing.assert_allclose(projected[index, m], expected, rtol=1e-15)


def test_short_intervals_remain_accurate_and_long_horizons_approach_periodic_equilibrium(case):
    states, forces, _, parameters, _ = case
    constant_force = forces[:, :1]
    tiny = project_repeated_force_states(states, constant_force, [1e-9], parameters, [1, 1_000_000])
    for i, h in enumerate([1, 1_000_000]):
        for m, p in enumerate(parameters):
            expected = p.rest_state + p.alpha * constant_force[m, 0] * (-p.tau_fat * np.expm1(-h * 1e-9 / p.tau_fat))
            np.testing.assert_allclose(tiny[i, m], expected, rtol=1e-15)
    long = project_repeated_force_states(states, constant_force, [.63], parameters, [100_000])
    for m, p in enumerate(parameters):
        np.testing.assert_allclose(long[0, m], p.rest_state + p.alpha * constant_force[m, 0] * p.tau_fat)


def test_single_horizon_and_single_margin_limits_are_exact(case):
    states, forces, durations, parameters, model = case
    result = projected_mechanical_reserve(states, forces, durations, parameters, [1], model)
    assert result.penalty == pytest.approx(-result.horizon_reserves[0], abs=1e-15)
    single = LocalMechanicalMarginModel(states, [.6], model.state_jacobian[:1])
    result = projected_mechanical_reserve(states, forces, durations, parameters, [1], single)
    assert result.penalty == pytest.approx(-result.margins[0, 0], abs=1e-15)


def test_negative_hard_margin_is_never_hidden_by_smoothing(case):
    states, forces, durations, parameters, model = case
    model = LocalMechanicalMarginModel(states, [-.001, 2.], np.zeros_like(model.state_jacobian))
    result = projected_mechanical_reserve(states, forces, durations, parameters, [1, 40], model)
    assert result.hard_minimum_margin == pytest.approx(-.001)
    assert np.all(result.horizon_reserves <= result.hard_minimum_margin)
    assert result.penalty >= -result.hard_minimum_margin


def test_all_three_slow_states_and_external_margin_update_affect_reserve(case):
    states, forces, durations, parameters, model = case
    base = projected_mechanical_reserve(states, forces, durations, parameters, [1, 40], model)
    for column, delta in enumerate([-100., .01, .02]):
        perturbed = states.copy()
        perturbed[0, column] += delta
        result = projected_mechanical_reserve(perturbed, forces, durations, parameters, [1, 40], model)
        assert result.penalty > base.penalty
    updated = LocalMechanicalMarginModel(states, model.reference_margins + .1, model.state_jacobian)
    result = projected_mechanical_reserve(states, forces, durations, parameters, [1, 40], updated)
    assert result.penalty == pytest.approx(base.penalty - .1)
    assert not model.state_jacobian.flags.writeable


def test_more_force_and_more_distant_horizons_reduce_reserve_from_rest(case):
    states, forces, durations, parameters, model = case
    base = projected_mechanical_reserve(states, forces, durations, parameters, [1, 20, 60], model)
    increased = projected_mechanical_reserve(states, forces * 1.3, durations, parameters, [1, 20, 60], model)
    assert increased.penalty > base.penalty
    assert np.all(np.diff(base.horizon_reserves) < 0)
    assert np.all(np.diff(base.states[:, :, 0], axis=0) < 0)
    assert np.all(np.diff(base.states[:, :, 1:], axis=0) > 0)


@pytest.mark.parametrize("temperature", [1e-4, .03, 1.])
def test_stable_conservative_aggregation_and_bias_bounds(temperature):
    values = np.array([-1e6, -1e6 + .1, 1e6])
    minimum = soft_minimum(values, temperature=temperature)
    maximum = soft_maximum(values, temperature=temperature)
    assert np.isfinite(minimum) and np.isfinite(maximum)
    assert 0 <= values.min() - minimum <= temperature * np.log(3) + 1e-9
    assert 0 <= maximum - values.max() <= temperature * np.log(3) + 1e-9
    assert soft_maximum([.4], temperature=temperature) == pytest.approx(.4)
    assert soft_minimum([.4], temperature=temperature) == pytest.approx(.4)


@pytest.mark.parametrize("symbolic_type", ["SX", "MX"])
def test_casadi_value_gradient_and_hessian_match_numpy_finite_difference(case, symbolic_type):
    ca = pytest.importorskip("casadi")
    states, forces, durations, parameters, model = case
    # Normalize variables so each finite difference has a comparable scale.
    point = np.r_[np.ones(6), forces.ravel() / 100.]
    symbol = getattr(ca, symbolic_type).sym("x", point.size)
    initial = ca.reshape(symbol[:6], 3, 2).T * ca.DM(states)
    force = 100. * ca.reshape(symbol[6:], 3, 2).T
    result = projected_mechanical_reserve_casadi(initial, force, durations, parameters, [1, 20, 60], model)
    fun = ca.Function("reserve", [symbol], [result.penalty, ca.gradient(result.penalty, symbol),
                                         ca.hessian(result.penalty, symbol)[0], result.margins])

    def numeric(x):
        return projected_mechanical_reserve(states * x[:6].reshape(2, 3), 100. * x[6:].reshape(2, 3),
                                             durations, parameters, [1, 20, 60], model)

    value, gradient, hessian, margins = fun(point)
    expected = numeric(point)
    assert float(value) == pytest.approx(expected.penalty, abs=1e-13)
    np.testing.assert_allclose(margins, expected.margins, atol=1e-13)
    step = 1e-5
    perturbations = step * np.eye(point.size)
    fd_gradient = np.array([(numeric(point + d).penalty - numeric(point - d).penalty) / (2 * step)
                            for d in perturbations])
    fd_hessian = np.column_stack([(np.asarray(fun(point + d)[1]).ravel() - np.asarray(fun(point - d)[1]).ravel())
                                  / (2 * step) for d in perturbations])
    np.testing.assert_allclose(np.asarray(gradient).ravel(), fd_gradient, rtol=3e-6, atol=2e-9)
    np.testing.assert_allclose(hessian, fd_hessian, rtol=5e-5, atol=2e-8)
    # State sensitivities include A, Tau1, Km for both muscles.
    assert np.all(np.abs(np.asarray(gradient).ravel()[:6]) > 1e-6)


@pytest.mark.parametrize("symbolic_type", ["SX", "MX"])
def test_casadi_stability_and_derivatives_at_ties(symbolic_type):
    ca = pytest.importorskip("casadi")
    x = getattr(ca, symbolic_type).sym("x", 3)
    value = soft_maximum_casadi(x, temperature=.02)
    fun = ca.Function("ties", [x], [value, ca.gradient(value, x), ca.hessian(value, x)[0]])
    for point in ([1e6, 1e6, 1e6], [-1e6, 1e6, 1e6]):
        result = fun(point)
        assert all(np.all(np.isfinite(np.asarray(v))) for v in result)
        assert float(result[0]) == pytest.approx(soft_maximum(point, temperature=.02))
    _, gradient, hessian = fun([0., 0., 0.])
    np.testing.assert_allclose(np.asarray(gradient).ravel(), np.ones(3) / 3)
    np.testing.assert_allclose(hessian, (np.eye(3) / 3 - np.ones((3, 3)) / 9) / .02, atol=1e-12)


@pytest.mark.parametrize("horizons", [[], [0], [-1], [1.5], [True], [2, 1], [1, 1]])
def test_invalid_horizons_rejected(case, horizons):
    states, forces, durations, parameters, _ = case
    with pytest.raises(ValueError, match="horizons"):
        project_repeated_force_states(states, forces, durations, parameters, horizons)


@pytest.mark.parametrize("durations", [[], [0.], [-1.], [np.inf], [np.nan], [[.1]]])
def test_invalid_durations_rejected(case, durations):
    states, forces, _, parameters, _ = case
    with pytest.raises(ValueError, match="durations"):
        project_repeated_force_states(states, forces, durations, parameters, [1])


@pytest.mark.parametrize("field,value", [("forces", -1.), ("forces", np.nan), ("states", 0.), ("states", np.inf)])
def test_input_domains_rejected_by_numpy_and_constant_casadi(case, field, value):
    ca = pytest.importorskip("casadi")
    states, forces, durations, parameters, _ = case
    states, forces = states.copy(), forces.copy()
    (states if field == "states" else forces)[0, 0] = value
    for function in (project_repeated_force_states, project_repeated_force_states_casadi):
        with pytest.raises(ValueError):
            function(states, forces, durations, parameters, [1])
        with pytest.raises(ValueError):
            function(states[:, :2], forces, durations, parameters, [1])


def test_depletion_is_rejected_without_clipping_in_both_backends(case):
    pytest.importorskip("casadi")
    states, forces, durations, parameters, _ = case
    for function in (project_repeated_force_states, project_repeated_force_states_casadi):
        with pytest.raises(ValueError, match="projected_states"):
            function(states, forces * 1000., durations, parameters, [500])


@pytest.mark.parametrize("temperature", [0., -1., np.inf, np.nan])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        soft_maximum([1.], temperature=temperature)


def test_margin_model_shape_finite_and_positive_reference_validation(case):
    states, _, _, _, model = case
    for reference, margins, jacobian in (
        (states, [], np.zeros((0, 2, 3))),
        (states * 0., model.reference_margins, model.state_jacobian),
        (states, [np.nan, .2], model.state_jacobian),
        (states, model.reference_margins, np.zeros((2, 3, 2))),
        (states, model.reference_margins, model.state_jacobian * np.inf),
    ):
        with pytest.raises(ValueError):
            LocalMechanicalMarginModel(reference, margins, jacobian)
