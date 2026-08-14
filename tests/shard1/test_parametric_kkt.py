import numpy as np
import pytest

from cocofest.optimization.parametric_kkt import (
    assemble_active_kkt_rows,
    kkt_prediction_passes_residual_guard,
    solve_parametric_kkt_sensitivity,
)


def test_active_kkt_rows_include_equalities_and_multiplier_supported_bounds():
    active = assemble_active_kkt_rows(
        variable_values=np.array([0.0, 0.4, 0.8]),
        constraint_values=np.array([1.0, 0.2, 0.7]),
        constraint_jacobian=np.array(
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        variable_lower_bounds=np.array([0.0, 0.0, -np.inf]),
        variable_upper_bounds=np.array([0.0, 1.0, 1.0]),
        constraint_lower_bounds=np.array([1.0, 0.0, -np.inf]),
        constraint_upper_bounds=np.array([1.0, 1.0, 0.9]),
        variable_multipliers=np.array([0.0, 2.0, 0.0]),
        constraint_multipliers=np.array([0.0, 0.0, 3.0]),
    )

    assert active.sources == (
        ("constraint", 0, "equality"),
        ("constraint", 2, "upper"),
        ("variable", 0, "equality"),
        ("variable", 1, "upper"),
    )
    np.testing.assert_allclose(
        active.jacobian,
        [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )


def test_parametric_kkt_sensitivity_matches_quadratic_program_solution():
    # min 1/2 z' H z, subject to A z - p = 0.
    hessian = np.diag([2.0, 4.0])
    jacobian = np.array([[1.0, 1.0]])
    parameter_step = np.array([0.3])

    prediction = solve_parametric_kkt_sensitivity(
        hessian,
        jacobian,
        stationarity_parameter_jacobian=np.zeros((2, 1)),
        constraint_parameter_jacobian=np.array([[-1.0]]),
        parameter_step=parameter_step,
    )

    # H^-1 A' / (A H^-1 A') * dp = [2/3, 1/3] * dp.
    np.testing.assert_allclose(prediction.primal_step, [0.2, 0.1])
    np.testing.assert_allclose(prediction.dual_step, [-0.4])
    assert prediction.linear_residual_inf_norm < 1e-14
    assert prediction.applied_step_scale == 1.0
    assert prediction.used_least_squares is False


def test_parametric_kkt_sensitivity_limits_primal_and_dual_step_together():
    prediction = solve_parametric_kkt_sensitivity(
        np.diag([2.0, 4.0]),
        np.array([[1.0, 1.0]]),
        np.zeros((2, 1)),
        np.array([[-1.0]]),
        np.array([0.3]),
        maximum_primal_step_inf_norm=0.05,
    )

    assert prediction.raw_primal_step_inf_norm == pytest.approx(0.2)
    assert prediction.applied_step_scale == pytest.approx(0.25)
    np.testing.assert_allclose(prediction.primal_step, [0.05, 0.025])
    np.testing.assert_allclose(prediction.dual_step, [-0.1])
    assert prediction.linear_residual_inf_norm < 1e-14


def test_kkt_prediction_residual_guard_rejects_a_worse_initial_guess():
    assert kkt_prediction_passes_residual_guard(1e-3, 8e-4)
    assert not kkt_prediction_passes_residual_guard(1e-3, 1.01e-3)
    assert kkt_prediction_passes_residual_guard(
        0.0, 5e-7, absolute_tolerance=1e-6
    )
    assert not kkt_prediction_passes_residual_guard(np.nan, 0.0)


def test_parametric_kkt_rejects_incompatible_dimensions():
    with pytest.raises(ValueError, match="constraint Jacobian"):
        solve_parametric_kkt_sensitivity(
            np.eye(2),
            np.ones((1, 3)),
            np.zeros((2, 1)),
            np.zeros((1, 1)),
            np.ones(1),
        )
