from types import SimpleNamespace
import numpy as np
import pytest
import casadi as ca

from cocofest.optimization.task_load_margin_oracle import (
    configure_load_objective, audit_load_solution, reoptimized_central_gradient,
    validate_reoptimized_direction, initial_bound_envelope_gradient,
)


def test_load_objective_replaces_fatigue_without_altering_path_bounds():
    penalty = SimpleNamespace(weight=np.array([[10000.]]), node=(1,))
    bounds = SimpleNamespace(min=np.array([[0., -10., 6.]]), max=np.array([[0., 50., 6.]]))
    updates = []
    program = SimpleNamespace(J=[], nlp=[SimpleNamespace(J=[penalty], x_bounds={"E_prod": bounds})],
                              update_objectives=updates.append)
    configure_load_objective(program, nominal_work_j=6., upper_bound=3.)
    assert updates[0].weight == 0
    assert penalty.weight == 10000
    np.testing.assert_array_equal(bounds.min, [[0., -10., 0.]])
    np.testing.assert_array_equal(bounds.max, [[0., 50., 18.]])
    assert updates[1].quadratic is False


def _linear_problem(*, multiplier=1., value=2.):
    x = ca.MX.sym("x")
    graph = {"x": x, "g": x, "f": -x}
    limits = {"lbg": [0.], "ubg": [3.], "lbx": [0.], "ubx": [2.]}
    program = SimpleNamespace(ocp_solver=SimpleNamespace(nlp=graph, limits=limits))
    solution = SimpleNamespace(vector=np.array([value]), lam_g=np.array([0.]), lam_x=np.array([multiplier]),
        decision_states=lambda **_: {"E_prod": np.array([[0., value]])},
        decision_controls=lambda **_: {"last_pulse_width_Biceps": np.array([[.0002]])})
    return solution, program


def test_complete_audit_and_actual_bound_multiplier_kkt():
    solution, program = _linear_problem()
    load, audit, kkt = audit_load_solution(solution, program, nominal_work_j=1., upper_bound=3., tolerance=1e-6)
    assert load == 2. and audit.passed
    assert kkt == {"stationarity_residual": 0., "complementarity_residual": 0., "dual_feasibility_residual": 0.}
    cached = program._task_load_margin_oracle_cache
    second = audit_load_solution(solution, program, nominal_work_j=1., upper_bound=3., tolerance=1e-6)
    assert second[0] == load and second[1].passed and second[2] == kkt
    assert program._task_load_margin_oracle_cache is cached
    replacement = ca.MX.sym("replacement")
    program.ocp_solver.nlp = {"x": replacement, "g": replacement, "f": -replacement}
    third = audit_load_solution(solution, program, nominal_work_j=1., upper_bound=3., tolerance=1e-6)
    assert third[1].passed and third[2] == kkt
    assert program._task_load_margin_oracle_cache is not cached


def test_fake_success_cannot_hide_infeasibility_or_invalid_kkt():
    solution, program = _linear_problem(value=2.1, multiplier=0.)
    _, audit, kkt = audit_load_solution(solution, program, nominal_work_j=1., upper_bound=3., tolerance=1e-6)
    assert not audit.passed
    assert kkt["stationarity_residual"] == 1.


def test_reoptimized_gradient_and_failure_do_not_invent_direction():
    seen = []
    def oracle(x):
        seen.append(x.copy())
        return x[0] ** 2 + 3 * x[1]
    np.testing.assert_allclose(reoptimized_central_gradient([2., 1.], [.01, .02], oracle), [4., 3.])
    assert len(seen) == 4
    with pytest.raises(ValueError, match="witness missing"):
        reoptimized_central_gradient([1.], [.01], lambda x: None)


def test_independent_local_validation_reveals_gradient_error_and_curvature():
    oracle = lambda x: x[0]**2 + 3*x[1]
    report = validate_reoptimized_direction([2., 1.], [4.2, 3.], 7., [.005, .005],
        [[.01, -.02], [-.01, .02]], oracle)
    assert report["gradient_validation_max_abs_error"] == pytest.approx(.2)
    assert report["local_validation_max_abs_error"] == pytest.approx(.0021)
    assert report["independent_validation_points"] == 2
    with pytest.raises(ValueError, match="witness missing"):
        validate_reoptimized_direction([1.], [2.], 1., [.005], [[.01]], lambda x: None)


def test_missing_holdouts_do_not_validate_a_direction():
    with pytest.raises(ValueError, match="holdouts"):
        validate_reoptimized_direction([1.], [2.], 1., [.005], [], lambda x: x[0]**2)


def test_initial_bound_envelope_respects_global_mapping_scaling_and_sign():
    from cocofest.optimization.task_reserve_ocp import TaskReserveStateCoordinate
    z = ca.SX.sym("z", 3)
    bounds = SimpleNamespace(min=np.array([[6., 0., 0.]]), max=np.array([[6., 10., 10.]]))
    nlp = SimpleNamespace(X_scaled=[z[2]], states={"A": SimpleNamespace(index=[0])},
        x_scaling={"A": SimpleNamespace(scaling=np.array([2.]))}, x_bounds={"A": bounds})
    program = SimpleNamespace(nlp=[nlp], ocp_solver=SimpleNamespace(nlp={"x": z},
        limits={"lbx": [0., 0., 3.], "ubx": [10., 10., 3.]}))
    solution = SimpleNamespace(vector=np.array([1., 1., 3.]), lam_x=np.array([0., 0., .5]))
    report = initial_bound_envelope_gradient(solution, program, [TaskReserveStateCoordinate("A", scale=10.)])
    assert report["gradient"] == [2.5]
    assert report["decision_variable_indices"] == [2]
    assert report["normalized_coordinates"] == pytest.approx([.6])
    cached = program._task_load_margin_mapping_cache
    assert initial_bound_envelope_gradient(solution, program, [TaskReserveStateCoordinate("A", scale=10.)]) == report
    assert program._task_load_margin_mapping_cache is cached
    program.ocp_solver.limits["ubx"][-1] = 4.
    with pytest.raises(ValueError, match="fixed bound"):
        initial_bound_envelope_gradient(solution, program, [TaskReserveStateCoordinate("A", scale=10.)])
