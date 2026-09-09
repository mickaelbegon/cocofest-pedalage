import numpy as np
import pytest

from cocofest.optimization.smooth_muscle_moment_allocation import (
    SmoothMomentAllocationOptions,
    allocate_total_moment_smoothly,
    build_moment_allocation_nlp_function,
    build_smooth_moment_allocation_function,
    solve_bounded_moment_qp_reference,
)


def _case():
    return {
        "reference_moments": np.array([0.08, 0.04, -0.03, -0.01]),
        "required_total_moment": 0.13,
        "lower_bounds": np.array([0.00, 0.01, -0.10, -0.08]),
        "upper_bounds": np.array([0.20, 0.08, -0.005, -0.002]),
    }


def test_allocation_preserves_required_total_and_prefers_upward_reserve():
    case = _case()
    result = allocate_total_moment_smoothly(**case)

    assert np.sum(result.allocated_moments) == pytest.approx(
        case["required_total_moment"], abs=2e-16
    )
    assert result.equality_residual == pytest.approx(0.0, abs=2e-16)
    assert np.sum(result.allocation_shares) == pytest.approx(1.0, abs=5e-16)
    assert result.allocation_shares[0] > result.allocation_shares[1]
    assert result.allocation_shares[1] > result.allocation_shares[2]
    assert result.feasible


def test_zero_total_correction_returns_the_reference_exactly():
    case = _case()
    case["required_total_moment"] = float(np.sum(case["reference_moments"]))

    result = allocate_total_moment_smoothly(**case)

    np.testing.assert_array_equal(result.allocated_moments, case["reference_moments"])
    assert result.qp_deviation_cost == 0.0


def test_infeasible_total_is_exposed_by_margins_and_never_clipped():
    case = _case()
    case["required_total_moment"] = float(np.sum(case["upper_bounds"]) + 0.05)

    result = allocate_total_moment_smoothly(**case)

    assert np.sum(result.allocated_moments) == pytest.approx(case["required_total_moment"])
    assert np.any(result.upper_margins < 0.0)
    assert not result.feasible
    assert result.smooth_bound_penalty > 0.0


def test_allocation_is_equivariant_to_muscle_permutation():
    case = _case()
    permutation = np.array([2, 0, 3, 1])
    expected = allocate_total_moment_smoothly(**case)
    permuted = allocate_total_moment_smoothly(
        case["reference_moments"][permutation],
        required_total_moment=case["required_total_moment"],
        lower_bounds=case["lower_bounds"][permutation],
        upper_bounds=case["upper_bounds"][permutation],
    )

    np.testing.assert_allclose(
        permuted.allocated_moments,
        expected.allocated_moments[permutation],
        rtol=0.0,
        atol=5e-16,
    )
    np.testing.assert_allclose(
        permuted.allocation_shares,
        expected.allocation_shares[permutation],
        rtol=0.0,
        atol=5e-16,
    )


@pytest.mark.parametrize("symbolic_type", ["SX", "MX"])
def test_casadi_function_matches_numpy_and_has_correct_central_difference_jacobian(
    symbolic_type,
):
    import casadi as ca

    case = _case()
    options = SmoothMomentAllocationOptions(
        slack_temperature=0.004,
        direction_temperature=0.006,
        bound_penalty_temperature=0.002,
    )
    expected = allocate_total_moment_smoothly(**case, options=options)
    function = build_smooth_moment_allocation_function(
        4, options=options, symbolic_type=symbolic_type
    )
    observed = function(
        case["reference_moments"],
        case["required_total_moment"],
        case["lower_bounds"],
        case["upper_bounds"],
    )
    np.testing.assert_allclose(
        np.asarray(observed[0]).reshape(-1), expected.allocated_moments, atol=2e-14
    )
    np.testing.assert_allclose(
        np.asarray(observed[1]).reshape(-1), expected.allocation_shares, atol=2e-14
    )

    inputs = np.concatenate(
        (
            case["reference_moments"],
            [case["required_total_moment"]],
            case["lower_bounds"],
            case["upper_bounds"],
        )
    )
    vector = ca.MX.sym("allocation_inputs", inputs.size)
    allocation = function(vector[:4], vector[4], vector[5:9], vector[9:13])[0]
    jacobian_function = ca.Function("allocation_jacobian", [vector], [ca.jacobian(allocation, vector)])
    analytic = np.asarray(jacobian_function(inputs))
    step = 1e-7
    central = np.empty_like(analytic)
    for column in range(inputs.size):
        plus = inputs.copy()
        minus = inputs.copy()
        plus[column] += step
        minus[column] -= step
        plus_value = np.asarray(function(plus[:4], plus[4], plus[5:9], plus[9:13])[0]).reshape(-1)
        minus_value = np.asarray(function(minus[:4], minus[4], minus[5:9], minus[9:13])[0]).reshape(-1)
        central[:, column] = (plus_value - minus_value) / (2.0 * step)
    np.testing.assert_allclose(analytic, central, rtol=2e-6, atol=2e-8)


def test_outer_nlp_terms_keep_feasibility_in_explicit_constraints():
    case = _case()
    function = build_moment_allocation_nlp_function(4, symbolic_type="SX")
    seed = allocate_total_moment_smoothly(**case).allocated_moments
    observed = function(
        seed,
        case["reference_moments"],
        case["required_total_moment"],
        case["lower_bounds"],
        case["upper_bounds"],
    )

    assert float(observed[2]) == pytest.approx(0.0, abs=2e-16)
    assert np.min(np.asarray(observed[3])) >= 0.0
    assert np.min(np.asarray(observed[4])) >= 0.0

    infeasible_candidate = seed.copy()
    infeasible_candidate[0] = case["upper_bounds"][0] + 0.01
    violated = function(
        infeasible_candidate,
        case["reference_moments"],
        case["required_total_moment"],
        case["lower_bounds"],
        case["upper_bounds"],
    )
    assert np.min(np.asarray(violated[4])) < 0.0


def test_outer_nlp_constraint_jacobians_are_exact():
    import casadi as ca

    function = build_moment_allocation_nlp_function(4, symbolic_type="SX")
    candidate = ca.SX.sym("candidate", 4)
    reference = ca.SX.sym("reference", 4)
    total = ca.SX.sym("total")
    lower = ca.SX.sym("lower", 4)
    upper = ca.SX.sym("upper", 4)
    outputs = function(candidate, reference, total, lower, upper)
    jacobian = ca.Function(
        "allocation_constraint_jacobian",
        [candidate, reference, total, lower, upper],
        [
            ca.jacobian(outputs[2], candidate),
            ca.jacobian(outputs[3], candidate),
            ca.jacobian(outputs[4], candidate),
        ],
    )
    case = _case()
    observed = jacobian(
        case["reference_moments"],
        case["reference_moments"],
        case["required_total_moment"],
        case["lower_bounds"],
        case["upper_bounds"],
    )

    np.testing.assert_array_equal(np.asarray(observed[0]), np.ones((1, 4)))
    np.testing.assert_array_equal(np.asarray(observed[1]), np.eye(4))
    np.testing.assert_array_equal(np.asarray(observed[2]), -np.eye(4))


def test_bounded_qp_oracle_redistributes_an_individually_infeasible_reference():
    reference = np.array([0.23, 0.01, -0.03, -0.01])
    lower = np.array([0.00, 0.01, -0.10, -0.08])
    upper = np.array([0.20, 0.08, -0.005, -0.002])
    result = solve_bounded_moment_qp_reference(
        reference,
        required_total_moment=float(np.sum(reference)),
        lower_bounds=lower,
        upper_bounds=upper,
    )

    assert result.status == "ok"
    assert np.sum(result.allocated_moments) == pytest.approx(np.sum(reference), abs=1e-14)
    assert np.all(result.allocated_moments >= lower)
    assert np.all(result.allocated_moments <= upper)
    assert result.allocated_moments[0] == pytest.approx(upper[0])
    assert np.any(result.allocated_moments[1:] != reference[1:])


@pytest.mark.parametrize(
    ("target", "status"),
    [(-1.0, "infeasible_total_below_bounds"), (1.0, "infeasible_total_above_bounds")],
)
def test_bounded_qp_oracle_reports_infeasible_total_without_projection(target, status):
    case = _case()
    result = solve_bounded_moment_qp_reference(
        case["reference_moments"],
        required_total_moment=target,
        lower_bounds=case["lower_bounds"],
        upper_bounds=case["upper_bounds"],
    )

    assert result.status == status
    assert result.allocated_moments is None
