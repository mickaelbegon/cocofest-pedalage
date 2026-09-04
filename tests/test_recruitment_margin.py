import numpy as np
import pytest

from cocofest.models.ding2007.ding2007_with_fatigue import DingModelPulseWidthFrequencyWithFatigue
from cocofest.optimization.recruitment_margin import (
    RecruitmentStatus,
    ding_force_derivative,
    ding_recruitment_expressions_casadi,
    ding_recruitment_margin,
    effective_recruitment_from_pulse_width,
    marginal_capacity_damage_rate,
    marginal_capacity_damage_rate_casadi,
)


BASE = dict(
    cn=0.35,
    force=40.0,
    capacity=3200.0,
    tau1=0.055,
    km=0.12,
    tau2=0.001,
    pd0=0.00013,
    pdt=0.00019,
    pulse_width_max=0.0006,
    force_length_relationship=0.9,
    force_velocity_relationship=0.95,
    passive_force_relationship=0.04,
)


def _force_derivative_for_pulse_width(pulse_width, **overrides):
    parameters = {**BASE, **overrides}
    effective = effective_recruitment_from_pulse_width(
        parameters["capacity"],
        pulse_width,
        parameters["pd0"],
        parameters["pdt"],
    )
    return ding_force_derivative(
        cn=parameters["cn"],
        force=parameters["force"],
        effective_recruitment=effective,
        tau1=parameters["tau1"],
        km=parameters["km"],
        tau2=parameters["tau2"],
        force_length_relationship=parameters["force_length_relationship"],
        force_velocity_relationship=parameters["force_velocity_relationship"],
        passive_force_relationship=parameters["passive_force_relationship"],
    )


def _margin_for_pulse_width(pulse_width, **overrides):
    parameters = {**BASE, **overrides}
    force_derivative = _force_derivative_for_pulse_width(pulse_width, **overrides)
    return ding_recruitment_margin(
        **parameters,
        force_derivative=force_derivative,
    )


def test_marginal_damage_is_the_normalized_force_driven_ding_term():
    forces = np.array([20.0, 50.0, 80.0])
    alpha_a = np.array([-0.04, -0.03, -0.05])
    rested = np.array([3000.0, 2500.0, 4000.0])

    damage = marginal_capacity_damage_rate(forces, alpha_a, rested)

    np.testing.assert_allclose(damage, -alpha_a * forces / rested)
    assert np.all(damage >= 0.0)
    np.testing.assert_allclose(
        marginal_capacity_damage_rate(3.0 * forces, alpha_a, 3.0 * rested),
        damage,
    )
    assert marginal_capacity_damage_rate(2.0 * forces, alpha_a, rested)[1] > damage[1]


def test_marginal_damage_accepts_muscle_by_phase_profiles_without_aggregation_weights():
    forces = np.array([[20.0, 30.0, 40.0], [5.0, 15.0, 25.0]])
    alpha_a = np.full_like(forces, -0.04)
    rested = np.array([[3000.0, 3000.0, 3000.0], [2500.0, 2500.0, 2500.0]])

    damage = marginal_capacity_damage_rate(forces, alpha_a, rested)

    assert damage.shape == forces.shape
    np.testing.assert_allclose(damage, -alpha_a * forces / rested)


@pytest.mark.parametrize(
    "forces,alpha_a,rested",
    [
        ([-1.0], [-0.04], [3000.0]),
        ([1.0], [0.04], [3000.0]),
        ([1.0], [-0.04], [0.0]),
        ([1.0, 2.0], [-0.04], [3000.0]),
    ],
)
def test_marginal_damage_rejects_non_damage_domains(forces, alpha_a, rested):
    with pytest.raises(ValueError):
        marginal_capacity_damage_rate(forces, alpha_a, rested)


def test_exact_pulse_width_and_force_derivative_reconstruction():
    reference_pulse_width = 0.00031
    force_derivative = _force_derivative_for_pulse_width(reference_pulse_width)

    result = ding_recruitment_margin(**BASE, force_derivative=force_derivative)

    assert result.status is RecruitmentStatus.OK
    assert result.feasible
    assert result.required_pulse_width == pytest.approx(reference_pulse_width, abs=1e-15)
    assert result.reconstructed_force_derivative == pytest.approx(force_derivative, rel=1e-13, abs=1e-13)
    assert result.utilization == pytest.approx(
        result.required_effective_recruitment / result.maximum_effective_recruitment
    )


def test_force_equation_matches_the_repository_ding_model():
    model = DingModelPulseWidthFrequencyWithFatigue()
    pulse_width = 0.00031
    effective = float(model.a_calculation(a_scale=BASE["capacity"], pulse_width=pulse_width))
    expected = float(
        model.f_dot_fun(
            BASE["cn"],
            BASE["force"],
            effective,
            BASE["tau1"],
            BASE["km"],
            force_length_relationship=BASE["force_length_relationship"],
            force_velocity_relationship=BASE["force_velocity_relationship"],
            passive_force_relationship=BASE["passive_force_relationship"],
        )
    )

    actual = ding_force_derivative(
        cn=BASE["cn"],
        force=BASE["force"],
        effective_recruitment=effective,
        tau1=BASE["tau1"],
        km=BASE["km"],
        tau2=model.tau2,
        force_length_relationship=BASE["force_length_relationship"],
        force_velocity_relationship=BASE["force_velocity_relationship"],
        passive_force_relationship=BASE["passive_force_relationship"],
    )
    assert actual == pytest.approx(expected, rel=1e-14, abs=1e-14)


def test_utilization_monotonicities_on_the_valid_ding_domain():
    low_force = _margin_for_pulse_width(0.00025)
    high_force = _margin_for_pulse_width(0.00040)
    assert high_force.utilization > low_force.utilization
    assert high_force.required_pulse_width > low_force.required_pulse_width

    smaller_force_state = ding_recruitment_margin(
        **{**BASE, "force": 30.0},
        force_derivative=0.0,
    )
    larger_force_state = ding_recruitment_margin(
        **{**BASE, "force": 50.0},
        force_derivative=0.0,
    )
    assert larger_force_state.required_effective_recruitment > smaller_force_state.required_effective_recruitment
    assert larger_force_state.utilization > smaller_force_state.utilization

    base_force_derivative = _force_derivative_for_pulse_width(0.00031)
    larger_capacity = ding_recruitment_margin(
        **{**BASE, "capacity": 4000.0},
        force_derivative=base_force_derivative,
    )
    baseline = ding_recruitment_margin(**BASE, force_derivative=base_force_derivative)
    assert larger_capacity.utilization < baseline.utilization
    assert larger_capacity.required_pulse_width < baseline.required_pulse_width

    larger_pw_max = ding_recruitment_margin(
        **{**BASE, "pulse_width_max": 0.0008},
        force_derivative=base_force_derivative,
    )
    assert larger_pw_max.utilization < baseline.utilization

    larger_km = ding_recruitment_margin(
        **{**BASE, "km": 0.2},
        force_derivative=base_force_derivative,
    )
    assert larger_km.required_effective_recruitment > baseline.required_effective_recruitment


@pytest.mark.parametrize(
    "overrides,expected_status",
    [
        ({"capacity": 0.0}, RecruitmentStatus.NON_POSITIVE_CAPACITY),
        ({"capacity": -1.0}, RecruitmentStatus.NON_POSITIVE_CAPACITY),
        ({"force": -1.0}, RecruitmentStatus.NEGATIVE_FORCE),
        ({"cn": -0.01}, RecruitmentStatus.NEGATIVE_CN),
        ({"km": 0.0}, RecruitmentStatus.NON_POSITIVE_KM),
        ({"tau1": 0.0}, RecruitmentStatus.NON_POSITIVE_TAU1),
        ({"tau2": -0.0001}, RecruitmentStatus.NEGATIVE_TAU2),
        ({"pd0": -0.0001}, RecruitmentStatus.NEGATIVE_PD0),
        (
            {"force_length_relationship": np.nan},
            RecruitmentStatus.NON_FINITE_MUSCLE_RELATIONSHIP,
        ),
        ({"cn": 1e-15}, RecruitmentStatus.CN_NEAR_ZERO),
        ({"cn": -0.12, "km": 0.12}, RecruitmentStatus.CALCIUM_DENOMINATOR_NEAR_ZERO),
        (
            {"tau1": -(0.001 * (0.35 / (0.12 + 0.35)))},
            RecruitmentStatus.RELAXATION_DENOMINATOR_NEAR_ZERO,
        ),
        (
            {"force_length_relationship": 0.0, "passive_force_relationship": 0.0},
            RecruitmentStatus.MECHANICAL_GAIN_NEAR_ZERO,
        ),
        ({"pulse_width_max": 0.0001}, RecruitmentStatus.INVALID_MAXIMUM_PULSE_WIDTH),
        ({"pdt": 0.0}, RecruitmentStatus.INVALID_PULSE_WIDTH_TIME_CONSTANT),
    ],
)
def test_invalid_and_singular_domains_are_classified(overrides, expected_status):
    result = ding_recruitment_margin(
        **{**BASE, **overrides},
        force_derivative=0.0,
    )
    assert result.status is expected_status
    assert not result.feasible


def test_zero_maximum_recruitment_is_not_reported_as_a_zero_utilization():
    result = ding_recruitment_margin(
        **{**BASE, "pulse_width_max": BASE["pd0"]},
        force_derivative=0.0,
    )
    assert result.status is RecruitmentStatus.ZERO_MAXIMUM_RECRUITMENT
    assert np.isnan(result.utilization)


def test_finite_but_excessive_pulse_width_preserves_utilization_above_one():
    requested_pw = 0.0007
    result = _margin_for_pulse_width(requested_pw)

    assert result.status is RecruitmentStatus.PULSE_WIDTH_LIMIT_EXCEEDED
    assert not result.feasible
    assert result.required_pulse_width == pytest.approx(requested_pw, abs=1e-15)
    assert result.utilization > 1.0


def test_recruitment_at_or_above_capacity_has_no_finite_pulse_width():
    parameters = BASE
    activation = parameters["cn"] / (parameters["km"] + parameters["cn"])
    relaxation = parameters["tau1"] + parameters["tau2"] * activation
    gain = (
        parameters["force_length_relationship"] * parameters["force_velocity_relationship"]
        + parameters["passive_force_relationship"]
    )

    def force_derivative_for(required):
        return gain * (required * activation - parameters["force"] / relaxation)

    at_capacity = ding_recruitment_margin(
        **parameters,
        force_derivative=force_derivative_for(parameters["capacity"]),
    )
    above_capacity = ding_recruitment_margin(
        **parameters,
        force_derivative=force_derivative_for(1.01 * parameters["capacity"]),
    )

    assert at_capacity.status is RecruitmentStatus.RECRUITMENT_AT_CAPACITY_ASYMPTOTE
    assert np.isinf(at_capacity.required_pulse_width)
    assert above_capacity.status is RecruitmentStatus.RECRUITMENT_EXCEEDS_CAPACITY
    assert np.isnan(above_capacity.required_pulse_width)


def test_negative_required_recruitment_is_explicit():
    result = ding_recruitment_margin(**BASE, force_derivative=-1e6)
    assert result.status is RecruitmentStatus.NEGATIVE_REQUIRED_RECRUITMENT


def test_casadi_expressions_match_numeric_value_and_jacobian():
    ca = pytest.importorskip("casadi")
    variables = ca.MX.sym("x", 5)
    required, maximum, utilization, pulse_width = ding_recruitment_expressions_casadi(
        cn=variables[0],
        force=variables[1],
        force_derivative=variables[2],
        capacity=variables[3],
        tau1=variables[4],
        km=BASE["km"],
        tau2=BASE["tau2"],
        pd0=BASE["pd0"],
        pdt=BASE["pdt"],
        pulse_width_max=BASE["pulse_width_max"],
        force_length_relationship=BASE["force_length_relationship"],
        force_velocity_relationship=BASE["force_velocity_relationship"],
        passive_force_relationship=BASE["passive_force_relationship"],
    )
    outputs = ca.vertcat(required, maximum, utilization, pulse_width)
    function = ca.Function("recruitment", [variables], [outputs, ca.jacobian(outputs, variables)])
    reference_pw = 0.00031
    point = np.array(
        [BASE["cn"], BASE["force"], _force_derivative_for_pulse_width(reference_pw), BASE["capacity"], BASE["tau1"]]
    )
    values, jacobian = function(point)
    numeric = ding_recruitment_margin(**BASE, force_derivative=point[2])

    np.testing.assert_allclose(
        np.asarray(values).ravel(),
        [
            numeric.required_effective_recruitment,
            numeric.maximum_effective_recruitment,
            numeric.utilization,
            numeric.required_pulse_width,
        ],
        rtol=1e-12,
        atol=1e-12,
    )

    step = 1e-6
    finite_difference = np.empty((4, 5))
    value_function = ca.Function("recruitment_values", [variables], [outputs])
    for index in range(point.size):
        perturbation = np.zeros_like(point)
        perturbation[index] = step
        finite_difference[:, index] = (
            np.asarray(value_function(point + perturbation)).ravel()
            - np.asarray(value_function(point - perturbation)).ravel()
        ) / (2.0 * step)
    np.testing.assert_allclose(np.asarray(jacobian), finite_difference, rtol=2e-5, atol=2e-6)


def test_casadi_damage_expression_matches_numpy():
    ca = pytest.importorskip("casadi")
    force = ca.MX.sym("force", 2)
    alpha = np.array([-0.04, -0.03])
    rested = np.array([3000.0, 2500.0])
    expression = marginal_capacity_damage_rate_casadi(force, alpha, rested)
    function = ca.Function("damage", [force], [expression])
    point = np.array([25.0, 60.0])

    np.testing.assert_allclose(
        np.asarray(function(point)).ravel(),
        marginal_capacity_damage_rate(point, alpha, rested),
    )
