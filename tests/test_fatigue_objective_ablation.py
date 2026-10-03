"""Contracts for the opt-in fatigue-cost ablation (no long OCP solve)."""

from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest
from bioptim import Node

from cocofest import CustomObjective
from cocofest.optimization.fatigue_objective_ablation import (
    terminal_linear_fatigue,
    terminal_parameterized_linear_fatigue,
)
from cocofest.optimization.parametric_fatigue_weights import (
    FATIGUE_WEIGHT_PARAMETER_KEY,
    ParametricFatigueWeightBinding,
)
from examples.fes_multibody.cycling import cycling_pulse_width_mhe as mhe
from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as periodic


def _model():
    return SimpleNamespace(muscles_dynamics_model=[
        SimpleNamespace(muscle_name="first", a_scale=100.),
        SimpleNamespace(muscle_name="second", a_scale=200.),
    ])


def _objective(variant, *, binding=None, duration=1.2, shape="quadratic"):
    return mhe.set_objective_functions(
        _model(), False, True, False, [0., 1., 0.], 0.,
        objective_shape=shape, fatigue_objective_variant=variant,
        fatigue_objective_duration_s=duration,
        fatigue_weight_binding=binding,
        terminal_wheel_regularization_weight=0.,
    )[0][0]


def test_legacy_cost_is_unchanged_and_explicit_modes_have_correct_nodes_and_scale():
    legacy = _objective("legacy")
    assert legacy.custom_function is CustomObjective.minimize_overall_muscle_fatigue
    assert legacy.node == Node.ALL and legacy.quadratic is True
    np.testing.assert_allclose(legacy.weight, [10000.])

    for mode, node, quadratic in (
        ("integral_quadratic", Node.ALL, True),
        ("integral_linear", Node.ALL, False),
        ("terminal_quadratic", Node.END, True),
        ("terminal_linear", Node.END, False),
    ):
        item = _objective(mode)
        assert item.node == node
        assert item.quadratic is quadratic
        np.testing.assert_allclose(item.weight, [12000. if node == Node.END else 10000.])


def test_terminal_linear_uses_weights_not_sqrt_weights_and_has_level_independent_gradient():
    x = ca.SX.sym("x", 2)
    weights = ca.SX.sym("w", 2)
    controller = SimpleNamespace(
        model=_model(),
        states={
            "A_first": SimpleNamespace(cx=x[0]),
            "A_second": SimpleNamespace(cx=x[1]),
        },
        parameters={FATIGUE_WEIGHT_PARAMETER_KEY: SimpleNamespace(cx=weights)},
    )
    weighted = terminal_parameterized_linear_fatigue(controller)
    unweighted = terminal_linear_fatigue(controller)
    quadratic_residual = CustomObjective.minimize_parameterized_overall_muscle_fatigue(controller)
    quadratic = ca.sumsqr(quadratic_residual)
    function = ca.Function("fatigue_linear_ablation", [x, weights],
                           [weighted, ca.gradient(weighted, x), unweighted,
                            quadratic, ca.gradient(quadratic, x)])
    for state in ([80., 120.], [50., 190.]):
        cost, gradient, unit_cost, quadratic_cost, quadratic_gradient = function(state, [2., .5])
        expected = 2. * (1. - state[0] / 100.) + .5 * (1. - state[1] / 200.)
        assert float(cost) == pytest.approx(expected)
        assert float(unit_cost) == pytest.approx(sum(1. - np.array(state) / [100., 200.]))
        np.testing.assert_allclose(np.asarray(gradient).ravel(), [-.02, -.0025])
        fatigue = 1. - np.array(state) / [100., 200.]
        assert float(quadratic_cost) == pytest.approx(np.sum(np.array([2., .5]) * fatigue**2))
        np.testing.assert_allclose(
            np.asarray(quadratic_gradient).ravel(),
            -2. * np.array([2., .5]) * fatigue / [100., 200.],
        )

    bound = _objective("terminal_linear", binding=ParametricFatigueWeightBinding((2., .5)))
    assert bound.custom_function is terminal_parameterized_linear_fatigue
    assert bound.quadratic is False
    assert _objective("terminal_linear").custom_function is terminal_linear_fatigue
    assert _objective("integral_linear", binding=ParametricFatigueWeightBinding((2., .5))).custom_function is (
        terminal_parameterized_linear_fatigue
    )


def test_linear_terminal_cost_is_an_increment_when_start_is_fixed():
    weights = np.array([2., .5])
    starts = np.array([.2, .4])
    end_a = np.array([.3, .45])
    end_b = np.array([.35, .42])
    terminal_difference = weights @ (end_a - end_b)
    increment_difference = weights @ ((end_a - starts) - (end_b - starts))
    assert terminal_difference == pytest.approx(increment_difference)
    # Temporal integral still distinguishes two paths with the same endpoint.
    shallow = np.sum(np.array([0., .1, .4]) ** 2)
    steep = np.sum(np.array([0., .3, .4]) ** 2)
    assert shallow < steep


def test_rejects_invalid_variant_and_duration_before_registering_an_objective():
    with pytest.raises(ValueError, match="fatigue_objective_variant"):
        _objective("terminal_magic")
    with pytest.raises(ValueError, match="duration"):
        _objective("terminal_linear", duration=0.)
    with pytest.raises(ValueError, match="minimize_fatigue"):
        mhe.set_objective_functions(
            _model(), False, False, True, [0., 0., 1.], 0.,
            fatigue_objective_variant="terminal_linear",
        )


def test_cli_variants_have_distinct_compiled_graph_cache_keys():
    parser = periodic.build_argument_parser()
    baseline = parser.parse_args([])
    alternate = parser.parse_args(["--fatigue-objective-variant", "terminal_linear"])
    assert baseline.fatigue_objective_variant == "legacy"
    assert periodic._codegen_signature(baseline) != periodic._codegen_signature(alternate)
