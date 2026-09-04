import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from cocofest import CustomObjective
from examples.fes_multibody.cycling import cycling_pulse_width_mhe as mhe_example
from cocofest.optimization.muscle_reserve import (
    DEFAULT_SMOOTH_MIN_TEMPERATURE,
    capacity_ratios,
    capacity_reserve_metrics,
    physiological_capacity_ratios,
    smooth_minimum_capacity_ratio,
    smooth_minimum_capacity_penalty,
    smooth_minimum_capacity_penalty_casadi,
    smooth_minimum_capacity_ratio_casadi,
    smooth_minimum_optimism_bound,
)


def test_capacity_ratios_are_dimensionless_and_scale_invariant():
    capacities = np.array([80.0, 45.0, 120.0])
    scales = np.array([100.0, 60.0, 150.0])

    np.testing.assert_allclose(capacity_ratios(capacities, scales), [0.8, 0.75, 0.8])
    np.testing.assert_allclose(
        capacity_ratios(17.0 * capacities, 17.0 * scales),
        capacity_ratios(capacities, scales),
    )


def test_smooth_minimum_penalty_is_permutation_invariant_and_monotone():
    capacities = np.array([0.92, 0.63, 0.81])
    scales = np.ones(3)
    baseline = smooth_minimum_capacity_penalty(capacities, scales, temperature=0.03)
    permuted = smooth_minimum_capacity_penalty(capacities[[2, 0, 1]], scales, temperature=0.03)
    improved_weakest = smooth_minimum_capacity_penalty(
        capacities + np.array([0.0, 0.08, 0.0]), scales, temperature=0.03
    )

    assert permuted == pytest.approx(baseline)
    assert improved_weakest < baseline


def test_same_mean_capacity_is_ranked_by_the_bottleneck():
    balanced = np.array([0.70, 0.70, 0.70, 0.70])
    bottlenecked = np.array([0.50, 0.70, 0.70, 0.90])
    assert np.mean(balanced) == pytest.approx(np.mean(bottlenecked))

    balanced_penalty = smooth_minimum_capacity_penalty(
        balanced,
        np.ones(4),
        temperature=0.03,
    )
    bottlenecked_penalty = smooth_minimum_capacity_penalty(
        bottlenecked,
        np.ones(4),
        temperature=0.03,
    )

    assert bottlenecked_penalty > balanced_penalty


def test_equal_muscles_and_zero_temperature_limit_have_expected_reserve():
    equal_ratio = np.array([0.74, 0.74, 0.74])
    assert smooth_minimum_capacity_ratio(equal_ratio, temperature=0.2) == pytest.approx(0.74)

    ratios = np.array([0.83, 0.61, 0.77])
    error = smooth_minimum_capacity_ratio(ratios, temperature=1e-6) - np.min(ratios)
    # The normalized log-sum-exp approaches the hard minimum from above, with
    # a known maximum gap tau * log(number of muscles).
    assert 0.0 <= error <= 1e-6 * np.log(ratios.size) + 1e-12


@pytest.mark.parametrize("size", [1, 2, 4, 17])
@pytest.mark.parametrize("temperature", [1e-4, 0.005, 0.1])
def test_smooth_minimum_has_the_proven_optimism_bound(size, temperature):
    ratios = np.linspace(0.31, 0.93, size)
    gap = smooth_minimum_capacity_ratio(ratios, temperature=temperature) - np.min(ratios)

    assert gap >= -1e-14
    assert gap <= smooth_minimum_optimism_bound(size, temperature) + 1e-14


def test_single_muscle_is_exact_for_every_temperature():
    ratio = 0.63
    for temperature in (1e-4, 0.005, 2.0):
        assert smooth_minimum_capacity_ratio([ratio], temperature=temperature) == pytest.approx(ratio)
        assert smooth_minimum_capacity_penalty([ratio], [1.0], temperature=temperature) == pytest.approx(1 - ratio)


def test_default_temperature_limits_four_muscle_optimism_to_one_percent():
    assert smooth_minimum_optimism_bound(4, DEFAULT_SMOOTH_MIN_TEMPERATURE) < 0.01


def test_metrics_report_hard_minimum_and_smoothing_gap():
    metrics = capacity_reserve_metrics([0.81, 0.62, 0.74], [1.0, 1.0, 1.0], temperature=0.02)
    assert metrics.minimum_ratio == pytest.approx(0.62)
    assert metrics.optimism_gap == pytest.approx(metrics.smooth_minimum_ratio - metrics.minimum_ratio)
    assert 0.0 <= metrics.optimism_gap <= metrics.maximum_optimism_gap
    assert metrics.penalty == pytest.approx(1.0 - metrics.smooth_minimum_ratio)


def test_numpy_implementation_remains_finite_for_extreme_ratios():
    reserve = smooth_minimum_capacity_ratio(np.array([-1e6, 1e6]), temperature=1e-3)
    assert np.isfinite(reserve)
    assert 0.0 <= reserve - (-1e6) <= 1e-3 * np.log(2) + 1e-9


def test_casadi_value_and_gradient_match_finite_difference():
    ca = pytest.importorskip("casadi")
    capacities = ca.MX.sym("A", 3)
    scales = np.array([100.0, 80.0, 120.0])
    penalty = smooth_minimum_capacity_penalty_casadi(capacities, scales, temperature=0.04)
    function = ca.Function("reserve_penalty", [capacities], [penalty, ca.gradient(penalty, capacities)])

    point = np.array([88.0, 52.0, 93.0])
    value, gradient = function(point)
    numeric_value = smooth_minimum_capacity_penalty(point, scales, temperature=0.04)
    np.testing.assert_allclose(float(value), numeric_value, rtol=1e-12, atol=1e-12)

    step = 1e-5
    finite_difference = np.empty(3)
    for index in range(3):
        perturbation = np.zeros(3)
        perturbation[index] = step
        finite_difference[index] = (
            smooth_minimum_capacity_penalty(point + perturbation, scales, temperature=0.04)
            - smooth_minimum_capacity_penalty(point - perturbation, scales, temperature=0.04)
        ) / (2.0 * step)
    np.testing.assert_allclose(np.asarray(gradient).ravel(), finite_difference, rtol=2e-6, atol=2e-8)

    # Increasing any capacity must reduce the penalty. The weakest normalized
    # muscle must have the largest marginal influence in absolute value.
    gradient = np.asarray(gradient).ravel()
    assert np.all(gradient < 0.0)
    weakest = int(np.argmin(point / scales))
    assert np.argmax(np.abs(gradient)) == weakest


def test_casadi_smooth_minimum_matches_numpy():
    ca = pytest.importorskip("casadi")
    ratios = ca.MX.sym("ratios", 3)
    expression = smooth_minimum_capacity_ratio_casadi(ratios, temperature=0.07)
    function = ca.Function("smooth_minimum", [ratios], [expression])
    point = np.array([0.91, 0.68, 0.79])

    assert float(function(point)) == pytest.approx(smooth_minimum_capacity_ratio(point, temperature=0.07))


def test_casadi_sx_and_mx_expressions_agree():
    ca = pytest.importorskip("casadi")
    point = np.array([0.86, 0.71, 0.93])
    scales = np.ones(3)
    values = []
    for symbolic_type in (ca.SX, ca.MX):
        capacities = symbolic_type.sym("A", 3)
        expression = smooth_minimum_capacity_penalty_casadi(
            capacities,
            scales,
            temperature=0.05,
        )
        values.append(float(ca.Function("reserve", [capacities], [expression])(point)))

    assert values[0] == pytest.approx(values[1], rel=1e-13, abs=1e-13)


def test_custom_objective_adapter_uses_capacity_states():
    ca = pytest.importorskip("casadi")
    capacities = ca.MX.sym("A", 3)
    muscle_models = [
        SimpleNamespace(muscle_name="m1", a_scale=100.0),
        SimpleNamespace(muscle_name="m2", a_scale=80.0),
        SimpleNamespace(muscle_name="m3", a_scale=120.0),
    ]
    controller = SimpleNamespace(
        model=SimpleNamespace(muscles_dynamics_model=muscle_models),
        states={
            f"A_{muscle.muscle_name}": SimpleNamespace(cx=capacities[index])
            for index, muscle in enumerate(muscle_models)
        },
    )

    expression = CustomObjective.minimize_terminal_muscle_reserve(
        controller,
        temperature=0.04,
    )
    assert expression.numel() == 1
    gradient = ca.gradient(expression, capacities)
    assert gradient.numel() == capacities.numel()
    function = ca.Function("terminal_reserve", [capacities], [expression, gradient])
    point = np.array([88.0, 52.0, 93.0])

    value, derivative = function(point)
    assert float(value) == pytest.approx(
        smooth_minimum_capacity_penalty(
            point,
            np.array([100.0, 80.0, 120.0]),
            temperature=0.04,
        )
    )
    assert np.all(np.isfinite(np.asarray(derivative, dtype=float)))


def test_bioptim_objective_wiring_omits_zero_reserve_weight():
    """The default reserve weight must not silently modify legacy RHO costs."""
    model = SimpleNamespace(muscles_dynamics_model=[])
    common = dict(
        model=model,
        minimize_force=False,
        minimize_fatigue=True,
        minimize_control=False,
        cost_fun_weight=[0.0, 1.0, 0.0],
        target=2.0 * np.pi,
        terminal_wheel_regularization_weight=0.0,
    )
    without_reserve = mhe_example.set_objective_functions(
        **common,
        terminal_reserve_weight=0.0,
    )
    with_reserve = mhe_example.set_objective_functions(
        **common,
        terminal_reserve_weight=0.25,
        terminal_reserve_temperature=0.031,
    )

    def reserve_entries(objectives):
        return [
            entry
            for phase in objectives
            for entry in phase
            if entry.custom_function is CustomObjective.minimize_terminal_muscle_reserve
        ]

    assert reserve_entries(without_reserve) == []
    entries = reserve_entries(with_reserve)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.node.name == "END"
    assert entry.quadratic is False
    # The public lambda shares the fatigue objective's dimensionless scaling.
    assert float(entry.weight[0]) == pytest.approx(2500.0)
    assert entry.extra_parameters["temperature"] == pytest.approx(0.031)


@pytest.mark.parametrize("capacities, scales", [([1.0], [0.0]), ([1.0, 2.0], [1.0])])
def test_invalid_capacity_vectors_are_rejected(capacities, scales):
    with pytest.raises(ValueError):
        capacity_ratios(capacities, scales)


def test_penalty_is_monotone_in_every_capacity_component():
    capacities = np.array([0.48, 0.67, 0.89])
    baseline = smooth_minimum_capacity_penalty(capacities, np.ones(3), temperature=0.04)
    for index in range(3):
        improved = capacities.copy()
        improved[index] += 0.03
        assert smooth_minimum_capacity_penalty(improved, np.ones(3), temperature=0.04) < baseline


def test_capacity_ratios_are_invariant_to_independent_unit_changes():
    capacities = np.array([80.0, 45.0, 120.0])
    scales = np.array([100.0, 60.0, 150.0])
    unit_factors = np.array([1e-3, 17.0, 1e4])
    np.testing.assert_allclose(
        capacity_ratios(capacities * unit_factors, scales * unit_factors),
        capacity_ratios(capacities, scales),
    )


def test_physiological_capacity_audit_has_explicit_tolerance():
    np.testing.assert_allclose(physiological_capacity_ratios([-1e-10, 1.0 + 1e-10], [1.0, 1.0]), [-1e-10, 1.0 + 1e-10])
    for capacities in ([-2e-9, 0.5], [0.5, 1.0 + 2e-9]):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            physiological_capacity_ratios(capacities, [1.0, 1.0])


@pytest.mark.parametrize("temperature", [0.0, -1.0, np.nan, np.inf])
def test_invalid_temperatures_are_rejected_by_numpy_and_casadi(temperature):
    ca = pytest.importorskip("casadi")
    with pytest.raises(ValueError, match="temperature"):
        smooth_minimum_capacity_ratio([0.4, 0.7], temperature=temperature)
    with pytest.raises(ValueError, match="temperature"):
        smooth_minimum_capacity_ratio_casadi(ca.MX.sym("r", 2), temperature=temperature)


@pytest.mark.parametrize("scales", [[0.0, 1.0], [-1.0, 1.0], [np.nan, 1.0], [np.inf, 1.0]])
def test_invalid_constant_casadi_scales_are_rejected(scales):
    ca = pytest.importorskip("casadi")
    with pytest.raises(ValueError, match="capacity_scales"):
        smooth_minimum_capacity_penalty_casadi(ca.MX.sym("A", 2), scales)


def test_custom_objective_rejects_invalid_muscle_scale():
    ca = pytest.importorskip("casadi")
    muscle = SimpleNamespace(muscle_name="m1", a_scale=0.0)
    controller = SimpleNamespace(
        model=SimpleNamespace(muscles_dynamics_model=[muscle]),
        states={"A_m1": SimpleNamespace(cx=ca.MX.sym("A"))},
    )
    with pytest.raises(ValueError, match="a_scale"):
        CustomObjective.minimize_terminal_muscle_reserve(controller)


def test_equal_capacity_gradient_is_shared_symmetrically():
    ca = pytest.importorskip("casadi")
    capacities = ca.MX.sym("A", 4)
    penalty = smooth_minimum_capacity_penalty_casadi(capacities, np.ones(4), temperature=0.02)
    gradient = ca.Function("equal_gradient", [capacities], [ca.gradient(penalty, capacities)])(
        np.full(4, 0.7)
    )
    np.testing.assert_allclose(np.asarray(gradient).ravel(), np.full(4, -0.25), atol=1e-13)


def test_casadi_implementation_remains_finite_for_extreme_ratios():
    ca = pytest.importorskip("casadi")
    ratios = ca.MX.sym("r", 2)
    function = ca.Function(
        "extreme_reserve",
        [ratios],
        [smooth_minimum_capacity_ratio_casadi(ratios, temperature=1e-3)],
    )
    reserve = float(function(np.array([-1e6, 1e6])))
    assert np.isfinite(reserve)
    assert 0.0 <= reserve + 1e6 <= 1e-3 * np.log(2) + 1e-9


def test_casadi_expression_can_be_compiled_and_evaluated(tmp_path, monkeypatch):
    ca = pytest.importorskip("casadi")
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("No C compiler available")

    capacities = ca.MX.sym("A", 3)
    expression = smooth_minimum_capacity_penalty_casadi(
        capacities,
        np.array([100.0, 80.0, 120.0]),
        temperature=0.01,
    )
    function = ca.Function("compiled_reserve", [capacities], [expression])
    monkeypatch.chdir(tmp_path)
    function.generate("compiled_reserve.c", {"with_header": True})
    subprocess.run(
        [compiler, "-fPIC", "-shared", "-O2", "compiled_reserve.c", "-o", "compiled_reserve.so"],
        check=True,
        capture_output=True,
        text=True,
    )
    compiled = ca.external("compiled_reserve", str(tmp_path / "compiled_reserve.so"))
    point = np.array([88.0, 52.0, 93.0])
    assert float(compiled(point)) == pytest.approx(
        smooth_minimum_capacity_penalty(point, [100.0, 80.0, 120.0], temperature=0.01),
        rel=1e-12,
        abs=1e-12,
    )
