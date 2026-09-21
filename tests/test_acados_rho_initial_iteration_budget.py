"""Deterministic early ACADOS budget without weakening frozen-RHO certification."""

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from examples.fes_multibody.cycling import cycling_fes_solver_comparison as comparison
from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as periodic


def _args():
    return periodic.build_argument_parser().parse_args([
        "--solver", "acados", "--max-acados-iterations", "100",
        "--acados-rho-initial-iteration-budget", "20",
        "--acados-ipopt-recovery", "--retry-failed-rho-without-advance",
    ])


def test_default_budget_is_unset_in_both_entry_points():
    for parser in (periodic.build_argument_parser(), comparison.build_cli()):
        assert parser.parse_args([]).acados_rho_initial_iteration_budget is None
        assert parser.parse_args([
            "--acados-rho-initial-iteration-budget", "20",
        ]).acados_rho_initial_iteration_budget == 20
    assert inspect.signature(comparison.main).parameters[
        "acados_rho_initial_iteration_budget"
    ].default is None
    periodic.validate_acados_rho_initial_iteration_budget(periodic.build_argument_parser().parse_args([]))
    periodic.validate_acados_rho_initial_iteration_budget(_args())


@pytest.mark.parametrize("field,value", [
    ("acados_rho_initial_iteration_budget", 0),
    ("acados_rho_initial_iteration_budget", 100),
    ("solver", "ipopt"),
    ("single_shot", True),
    ("acados_ipopt_recovery", False),
    ("retry_failed_rho_without_advance", False),
    ("max_consecutive_failing", 0),
    ("acados_maxiter_retries", 1),
    ("acados_forced_iteration_cap_rhos", (2,)),
    ("acados_ipopt_recovery_force_first_rho", True),
])
def test_budget_rejects_unsafe_or_contradictory_configuration(field, value):
    args = _args()
    setattr(args, field, value)
    with pytest.raises(ValueError, match="--acados-rho-initial-iteration-budget"):
        periodic.validate_acados_rho_initial_iteration_budget(args)


def test_comparison_forwards_budget_and_keeps_recovery_explicit(monkeypatch):
    captured = {}
    def fake_run(solver_name, args, **_):
        captured[solver_name] = args
        return {}
    monkeypatch.setattr(comparison, "_run_benchmark_case", fake_run)
    monkeypatch.setattr(comparison, "print_solver_overview", lambda _: None)
    comparison.main(
        solvers=("acados",), n_windows=2, mechanical_formulation="full",
        acados_rho_initial_iteration_budget=20, acados_ipopt_recovery=True,
        retry_failed_rho_without_advance=True,
    )
    assert captured["acados"].acados_rho_initial_iteration_budget == 20
    assert captured["acados"].acados_ipopt_recovery is True
    assert captured["acados"].acados_ipopt_fallback_advance is False


def test_only_first_attempt_is_capped_and_failed_rho_is_never_certified():
    budgets, summaries, attempted_rhos = [], [], set()
    interface = SimpleNamespace(status=2, solve=lambda: {"status": 2})
    nmpc = SimpleNamespace(ocp_solver=interface)
    residuals = np.array([1e-2, 2e-3, 0.0, 0.0])
    periodic.install_acados_forced_iteration_cap(
        nmpc, nominal_iterations=100, summaries=summaries, echo=False,
        set_iterations_function=lambda _, value: budgets.append(value) or True,
        diagnostics_function=lambda _: {"sqp_iter": 20, "time_tot": 2.0, "residuals": residuals},
    )
    assert not periodic.arm_acados_rho_initial_iteration_budget(
        nmpc, target_rho=1, budget=20, attempted_rhos=attempted_rhos,
    )
    assert periodic.arm_acados_rho_initial_iteration_budget(
        nmpc, target_rho=2, budget=20, attempted_rhos=attempted_rhos,
    )
    assert interface.solve() == {"status": 2}
    # A budget exhaustion does not authorize advancement, even if primal-feasible.
    assert not periodic._rho_solution_is_certified(2, {"passes_tolerance": True})
    assert not periodic.arm_acados_rho_initial_iteration_budget(
        nmpc, target_rho=2, budget=20, attempted_rhos=attempted_rhos,
    )
    interface.solve()  # Same frozen RHO retry at the nominal budget.
    interface.solve()  # Auxiliary solve remains nominal as well.
    assert budgets == [20, 100]
    assert len(summaries) == 1
    assert summaries[0]["requested_iteration_cap"] == 20
    assert summaries[0]["effective_iteration_cap"] == 20
    assert summaries[0]["iteration_cap_reached"] is True
    assert summaries[0]["unused_nominal_iteration_headroom"] == 80
    assert summaries[0]["nominal_budget_restored"] is True
    assert summaries[0]["estimated_time_saved_s"] is None
    residuals[:] = 0  # The audit owns its copy, independent of later solves.
    assert summaries[0]["residuals_at_return"][0] == 1e-2
    assert periodic.arm_acados_rho_initial_iteration_budget(
        nmpc, target_rho=3, budget=20, attempted_rhos=attempted_rhos,
    )
    interface.solve()
    assert budgets == [20, 100, 20, 100]


def test_budget_restores_nominal_after_native_exception():
    budgets, summaries = [], []
    def fail():
        raise RuntimeError("native exception")
    nmpc = SimpleNamespace(ocp_solver=SimpleNamespace(solve=fail))
    periodic.install_acados_forced_iteration_cap(
        nmpc, nominal_iterations=100, summaries=summaries, echo=False,
        set_iterations_function=lambda _, value: budgets.append(value) or True,
    )
    periodic.arm_acados_rho_initial_iteration_budget(
        nmpc, target_rho=2, budget=20, attempted_rhos=set(),
    )
    with pytest.raises(RuntimeError, match="native exception"):
        nmpc.ocp_solver.solve()
    assert budgets == [20, 100]
    assert nmpc._cocofest_forced_iteration_cap_pending is None
    assert summaries[0]["nominal_budget_restored"] is True
    assert summaries[0]["error"] == "RuntimeError: native exception"


@pytest.mark.parametrize("budget", [None, 20])
def test_budget_configuration_and_recovery_audit_survive_result_json(tmp_path, budget):
    args = _args()
    args.acados_rho_initial_iteration_budget = budget
    result = {"args": args, "window_statuses": [], "covered_cycles": 0}
    if budget is not None:
        result["acados_rho_initial_iteration_budget"] = periodic.summarize_acados_rho_initial_iteration_budget(
            args,
            [{"target_rho": 2, "requested_iteration_cap": 20, "effective_iteration_cap": 20,
              "residuals_at_return": np.array([1e-2, 2e-3, 0.0, 0.0]), "status": 2}],
            [{"target_rho": 2, "attempt_window": 2, "fallback_advanced": True},
             {"target_rho": 3, "attempt_window": 3, "fallback_advanced": False}],
        )
    path = comparison.write_benchmark_summary(tmp_path / "result.json", {"acados": result})
    payload = json.loads(path.read_text())
    assert payload["configurations"]["acados"]["acados_rho_initial_iteration_budget"] == budget
    audit = payload["results"][0]["acados_rho_initial_iteration_budget"]
    if budget is None:
        assert audit is None
    else:
        assert audit["certification_policy"] == "unchanged"
        assert audit["estimated_time_saved_s"] is None
        assert audit["attempts"][0]["residuals_at_return"] == [1e-2, 2e-3, 0.0, 0.0]
        assert audit["attempts"][0]["recovery_attempt_windows"] == [2]
        assert audit["attempts"][0]["fallback_advanced"] is True
