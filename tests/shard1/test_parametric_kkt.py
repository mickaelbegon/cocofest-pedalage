import numpy as np
import pytest

from cocofest.optimization.parametric_kkt import (
    CanonicalNlpKktEvaluator,
    active_kkt_rhs_from_bound_changes,
    active_kkt_rhs_to_bound_targets,
    apply_active_dual_step,
    assemble_active_kkt_rows,
    assemble_sparse_active_kkt_rows,
    kkt_prediction_passes_residual_guard,
    solve_parametric_kkt_sensitivity,
    solve_sparse_bound_kkt_sensitivity,
    sparse_active_jacobian_from_sources,
)


def test_active_kkt_rhs_tracks_variable_and_constraint_target_motion():
    target_step = active_kkt_rhs_from_bound_changes(
        sources=(
            ("constraint", 0, "equality"),
            ("constraint", 1, "upper"),
            ("variable", 0, "equality"),
            ("variable", 1, "lower"),
        ),
        old_variable_lower_bounds=np.array([0.0, 1.0]),
        old_variable_upper_bounds=np.array([0.0, 3.0]),
        new_variable_lower_bounds=np.array([-2.0, 1.25]),
        new_variable_upper_bounds=np.array([-2.0, 3.0]),
        old_constraint_lower_bounds=np.array([4.0, -np.inf]),
        old_constraint_upper_bounds=np.array([4.0, 5.0]),
        new_constraint_lower_bounds=np.array([3.5, -np.inf]),
        new_constraint_upper_bounds=np.array([3.5, 5.75]),
    )

    np.testing.assert_allclose(target_step, [-0.5, 0.75, -2.0, 0.25])


def test_active_kkt_rhs_rejects_equality_that_opens_into_an_interval():
    with pytest.raises(ValueError, match="equality disagree"):
        active_kkt_rhs_from_bound_changes(
            sources=(("variable", 0, "equality"),),
            old_variable_lower_bounds=np.array([1.0]),
            old_variable_upper_bounds=np.array([1.0]),
            new_variable_lower_bounds=np.array([2.0]),
            new_variable_upper_bounds=np.array([3.0]),
            old_constraint_lower_bounds=np.empty(0),
            old_constraint_upper_bounds=np.empty(0),
            new_constraint_lower_bounds=np.empty(0),
            new_constraint_upper_bounds=np.empty(0),
        )


def test_active_kkt_rhs_to_targets_corrects_a_repeat_candidate():
    residual = active_kkt_rhs_to_bound_targets(
        sources=(
            ("constraint", 0, "equality"),
            ("variable", 1, "lower"),
        ),
        variable_values=np.array([0.0, 0.4]),
        constraint_values=np.array([1.0]),
        variable_lower_bounds=np.array([-np.inf, 0.5]),
        variable_upper_bounds=np.array([np.inf, 1.0]),
        constraint_lower_bounds=np.array([1.3]),
        constraint_upper_bounds=np.array([1.3]),
    )

    np.testing.assert_allclose(residual, [0.3, 0.1])


def test_active_kkt_rows_include_equalities_and_primal_active_bounds():
    active = assemble_active_kkt_rows(
        variable_values=np.array([0.0, 1.0, 0.8]),
        constraint_values=np.array([1.0, 0.2, 0.9]),
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


def test_multiplier_does_not_activate_a_distant_bound():
    active = assemble_sparse_active_kkt_rows(
        variable_values=np.array([0.4]),
        constraint_values=np.array([0.7]),
        constraint_jacobian=np.array([[1.0]]),
        variable_lower_bounds=np.array([0.0]),
        variable_upper_bounds=np.array([1.0]),
        constraint_lower_bounds=np.array([0.0]),
        constraint_upper_bounds=np.array([0.9]),
        variable_multipliers=np.array([2.0]),
        constraint_multipliers=np.array([3.0]),
    )

    assert active.sources == ()
    assert active.jacobian.shape == (0, 1)


def test_sparse_active_rows_drop_duplicate_fixed_variable_direction():
    active = assemble_sparse_active_kkt_rows(
        variable_values=np.array([2.0, 0.0]),
        constraint_values=np.array([4.0]),
        constraint_jacobian=np.array([[2.0, 0.0]]),
        variable_lower_bounds=np.array([2.0, -np.inf]),
        variable_upper_bounds=np.array([2.0, np.inf]),
        constraint_lower_bounds=np.array([4.0]),
        constraint_upper_bounds=np.array([4.0]),
    )

    assert active.sources == (("constraint", 0, "equality"),)
    assert active.jacobian.shape == (1, 2)


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


def test_canonical_sparse_kkt_predicts_moving_equality_solution():
    from casadi import SX, vertcat

    # min x0^2 + 2*x1^2, subject to x0 + x1 = b. At b=1 the
    # optimum and multiplier are [2/3, 1/3] and -4/3.
    x = SX.sym("x", 2)
    evaluator = CanonicalNlpKktEvaluator(
        {"x": x, "f": x[0] ** 2 + 2 * x[1] ** 2, "g": vertcat(x[0] + x[1])}
    )
    old_x = np.array([2.0 / 3.0, 1.0 / 3.0])
    blocks = evaluator.evaluate(old_x, np.array([-4.0 / 3.0]))

    assert blocks.lagrangian_hessian.shape == (2, 2)
    assert blocks.constraint_jacobian.shape == (1, 2)
    assert blocks.lagrangian_hessian.nnz == 2
    np.testing.assert_allclose(blocks.objective_gradient, [4.0 / 3.0, 4.0 / 3.0])
    np.testing.assert_allclose(blocks.constraint_values, [1.0])

    active = assemble_sparse_active_kkt_rows(
        variable_values=old_x,
        constraint_values=blocks.constraint_values,
        constraint_jacobian=blocks.constraint_jacobian,
        variable_lower_bounds=np.full(2, -np.inf),
        variable_upper_bounds=np.full(2, np.inf),
        constraint_lower_bounds=np.array([1.0]),
        constraint_upper_bounds=np.array([1.0]),
        constraint_multipliers=np.array([-4.0 / 3.0]),
    )
    target_step = active_kkt_rhs_from_bound_changes(
        active.sources,
        old_variable_lower_bounds=np.full(2, -np.inf),
        old_variable_upper_bounds=np.full(2, np.inf),
        new_variable_lower_bounds=np.full(2, -np.inf),
        new_variable_upper_bounds=np.full(2, np.inf),
        old_constraint_lower_bounds=np.array([1.0]),
        old_constraint_upper_bounds=np.array([1.0]),
        new_constraint_lower_bounds=np.array([1.3]),
        new_constraint_upper_bounds=np.array([1.3]),
    )
    prediction = solve_sparse_bound_kkt_sensitivity(
        blocks.lagrangian_hessian,
        active.jacobian,
        target_step,
    )

    np.testing.assert_allclose(prediction.primal_step, [0.2, 0.1], atol=1e-13)
    np.testing.assert_allclose(old_x + prediction.primal_step, [13.0 / 15.0, 13.0 / 30.0])
    assert prediction.linear_residual_inf_norm < 1e-13
    assert prediction.used_least_squares is False


def test_sparse_newton_kkt_recovers_new_primal_and_dual_from_repeat():
    from casadi import SX, vertcat

    x = SX.sym("x", 2)
    evaluator = CanonicalNlpKktEvaluator(
        {"x": x, "f": x[0] ** 2 + 2 * x[1] ** 2, "g": vertcat(x[0] + x[1])}
    )
    repeat = np.array([0.8, 0.4])
    old_lam_g = np.array([-4.0 / 3.0])
    old_lam_x = np.zeros(2)
    blocks = evaluator.evaluate(repeat, old_lam_g)
    sources = (("constraint", 0, "equality"),)
    active_jacobian = sparse_active_jacobian_from_sources(
        blocks.constraint_jacobian,
        variable_count=2,
        sources=sources,
    )
    stationarity = (
        blocks.objective_gradient
        + np.asarray(blocks.constraint_jacobian.T @ old_lam_g).reshape(-1)
        + old_lam_x
    )
    prediction = solve_sparse_bound_kkt_sensitivity(
        blocks.lagrangian_hessian,
        active_jacobian,
        active_target_step=np.array([1.3 - blocks.constraint_values[0]]),
        stationarity_residual=stationarity,
    )
    duals = apply_active_dual_step(
        sources,
        old_lam_g,
        old_lam_x,
        prediction.dual_step,
    )

    np.testing.assert_allclose(repeat + prediction.primal_step, [13 / 15, 13 / 30])
    np.testing.assert_allclose(duals.constraint_multipliers, [-26 / 15])
    np.testing.assert_allclose(duals.variable_multipliers, [0.0, 0.0])
    assert duals.passes_sign_guard
    assert prediction.linear_residual_inf_norm < 1e-13


def test_active_dual_prediction_rejects_wrong_bound_sign_without_clipping():
    prediction = apply_active_dual_step(
        sources=(
            ("constraint", 0, "lower"),
            ("variable", 0, "upper"),
        ),
        constraint_multipliers=np.array([-0.2]),
        variable_multipliers=np.array([0.3]),
        dual_step=np.array([0.4, -0.5]),
    )

    np.testing.assert_allclose(prediction.constraint_multipliers, [0.2])
    np.testing.assert_allclose(prediction.variable_multipliers, [-0.2])
    assert not prediction.passes_sign_guard
    assert prediction.sign_violation_count == 2


def test_sparse_active_rows_do_not_materialize_dense_identity():
    from scipy.sparse import csc_matrix, issparse

    active = assemble_sparse_active_kkt_rows(
        variable_values=np.array([0.0, 0.5, 1.0]),
        constraint_values=np.array([0.5]),
        constraint_jacobian=csc_matrix([[0.0, 1.0, 0.0]]),
        variable_lower_bounds=np.array([0.0, -np.inf, -np.inf]),
        variable_upper_bounds=np.array([0.0, np.inf, 1.0]),
        constraint_lower_bounds=np.array([0.5]),
        constraint_upper_bounds=np.array([0.5]),
    )

    assert issparse(active.jacobian)
    assert active.jacobian.shape == (3, 3)
    assert active.jacobian.nnz == 3
    assert active.sources == (
        ("constraint", 0, "equality"),
        ("variable", 0, "equality"),
        ("variable", 2, "upper"),
    )
