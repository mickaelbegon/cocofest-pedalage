"""Normalized ΔPW costs preserve the hard bound and export as local ACADOS costs."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from casadi import DM, Function, jacobian
from bioptim import ObjectiveFcn, ObjectiveList, SolutionMerge, Solver

from cocofest.optimization.pulse_width_slew import (
    add_slew_regularization, attach_slew_audit, normalized_slew_residual,
    pulse_width_slew_signature, validate_slew_regularization,
)
from cocofest.simulation import CapabilityRegistry, SimulationConfig, build_launch_plan
from cocofest.simulation.gui_model import config_from_form, form_values, scientific_summary
from tests.test_pulse_width_slew import _ocp, _direct_ocp


def _with_regularization(n=10, weight=1.0, *, keep_tracking=False):
    ocp = _ocp(n, collocation=False)
    objectives = ObjectiveList()
    add_slew_regularization(objectives, ocp.nlp[0].model, weight=weight,
                            n_shooting=n, interval_s=1 / n)
    if weight:
        objectives[0][0].list_index = 1 if keep_tracking else 0
        ocp.update_objectives(objectives)
    return ocp


@pytest.mark.parametrize("weight,reference,step", [
    (-1, 100, 100e-6), (float("nan"), 100, 100e-6),
    (float("inf"), 100, 100e-6), (True, 100, 100e-6),
    (1, 0, 100e-6), (1, float("inf"), 100e-6), (1, 100, None),
])
def test_invalid_regularization_rejected(weight, reference, step):
    with pytest.raises(ValueError):
        validate_slew_regularization(weight, reference, max_step_s=step)


def test_zero_weight_does_not_require_a_lift_or_add_an_objective():
    objectives = ObjectiveList()
    add_slew_regularization(objectives, SimpleNamespace(), n_shooting=1, interval_s=1)
    assert not any(objective for phase in objectives for objective in phase)


@pytest.mark.parametrize("delta,expected", [(0, 0), (100e-6, 1), (-100e-6, -1)])
def test_physical_normalized_residual(delta, expected):
    controller = SimpleNamespace(
        model=SimpleNamespace(muscles_dynamics_model=[SimpleNamespace(muscle_name="m")]),
        controls={"pw_slew_delta_pw_m": SimpleNamespace(cx=DM(delta))},
    )
    assert float(normalized_slew_residual(controller, 100e-6)) == pytest.approx(expected)


def test_constant_control_derivative_is_not_an_adjacent_control_difference():
    ocp = _ocp(3, collocation=False)
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="last_pulse_width_m",
                   derivative=True, quadratic=True, weight=1, multi_thread=False)
    ocp.update_objectives(objectives)
    function = ocp.nlp[0].J[0].function[0]
    arguments = [DM.ones(*function.size_in(i)) for i in range(function.n_in())]
    assert float(function(*arguments)) == 0


@pytest.mark.parametrize("n", [30, 50])
def test_direct_regularization_is_nonzero_normalized_and_has_no_final_pair(n):
    ocp = _direct_ocp(n, collocation=True)
    dimensions = (ocp.nlp[0].states.shape, ocp.nlp[0].controls.shape)
    objectives = ObjectiveList()
    add_slew_regularization(objectives, ocp.nlp[0].model, weight=2.5,
                            n_shooting=n, interval_s=1 / n)
    ocp.update_objectives(objectives)
    assert (ocp.nlp[0].states.shape, ocp.nlp[0].controls.shape) == dimensions == (1, 1)
    penalties = ocp.nlp[0].J
    assert len(penalties) == n - 1
    differences = np.linspace(-100e-6, 100e-6, n - 1)
    total = 0.0
    for node, (penalty, difference) in enumerate(zip(penalties, differences)):
        assert penalty.node_idx == [node]
        function = penalty.weighted_function[node]
        arguments = {function.name_in(i): DM.zeros(*function.size_in(i)) for i in range(function.n_in())}
        arguments["dt"] = 1 / n
        arguments["u"] = DM([300e-6, 300e-6 + difference]) / .0025
        arguments["weight"] = float(penalty.weight.evaluate_at(0, 1)[0])
        total += float(function(**arguments)["val"])
    assert total > 0
    assert total == pytest.approx(2.5 * np.mean((differences / 100e-6) ** 2))


def test_direct_regularization_smooths_without_lifting_or_losing_the_bound():
    variations = []
    for weight in (0, 100):
        ocp = _direct_ocp(10, collocation=True)
        objectives = ObjectiveList()
        add_slew_regularization(objectives, ocp.nlp[0].model, weight=weight,
                                n_shooting=10, interval_s=.1)
        for index, objective in enumerate(objectives[0] if weight else []):
            objective.list_index = index + 1
        if weight:
            ocp.update_objectives(objectives)
        solver = Solver.IPOPT(show_online_optim=False)
        solver.set_print_level(0)
        solver.set_tol(1e-9)
        solution = ocp.solve(solver)
        assert solution.status == 0
        controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
        assert set(controls) == {"last_pulse_width_m"}
        delta = np.diff(controls["last_pulse_width_m"][0])
        assert max(abs(delta)) <= 100e-6 + 1e-10
        variations.append(float(np.sum(delta ** 2)))
    assert variations[1] < .9 * variations[0]


@pytest.mark.parametrize("n", [30, 50])
def test_bioptim_weighted_cost_is_the_normalized_mean(n):
    ocp = _with_regularization(n, weight=2.5)
    penalty = ocp.nlp[0].J[0]
    function = penalty.weighted_function[0]
    total = 0.0
    deltas = np.r_[np.linspace(-100e-6, 100e-6, n - 1), 0.0]
    for delta in deltas:
        arguments = {function.name_in(i): DM.zeros(*function.size_in(i)) for i in range(function.n_in())}
        arguments["dt"] = 1 / n
        # Bioptim's penalty input includes both interval ends, even for this
        # stage-local expression; both physical controls are scaled by .0025.
        arguments["u"] = DM([300e-6 / .0025, delta / .0025] * 2)
        arguments["weight"] = float(penalty.weight.evaluate_at(0, 1)[0])
        total += float(function(**arguments)["val"])
    assert total == pytest.approx(2.5 * np.mean((deltas[:-1] / 100e-6) ** 2))


def test_acados_export_is_local_nonzero_and_has_no_terminal_cost():
    from bioptim.interfaces.acados_interface import AcadosInterface
    from examples.fes_multibody.cycling.cycling_pulse_width_mhe_acados_periodic import patch_bioptim_acados_interface
    patch_bioptim_acados_interface()
    ocp = _with_regularization(3, weight=2)
    interface = AcadosInterface(ocp, Solver.ACADOS())
    interface._AcadosInterface__set_costs(ocp)
    model = interface.acados_model
    residual = Function("local_delta_cost", [model.x, model.u], [model.cost_y_expr])
    assert float(residual([.2], [300e-6 / .0025, 100e-6 / .0025])) == pytest.approx(1)
    np.testing.assert_allclose(interface.acados_ocp.cost.W, [[3.]])
    np.testing.assert_allclose(interface.acados_ocp.cost.W_0, [[3.]])
    assert model.cost_y_expr_e.is_zero()
    gradient = Function("delta_jacobian", [model.x, model.u], [jacobian(model.cost_y_expr, model.u)])
    np.testing.assert_allclose(gradient([.2], [.1, 0]), [[0, 25]])


def test_regularization_smooths_controls_and_preserves_the_hard_bound():
    variations = []
    for weight in (0, 100):
        ocp = _with_regularization(10, weight=weight, keep_tracking=True)
        solver = Solver.IPOPT(show_online_optim=False)
        solver.set_print_level(0)
        solver.set_tol(1e-9)
        solution = ocp.solve(solver)
        assert solution.status == 0
        controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
        delta = np.diff(controls["last_pulse_width_m"][0])
        assert max(abs(delta)) <= 100e-6 + 1e-10
        variations.append(float(np.sum(delta ** 2)))
        if weight:
            assert abs(controls["pw_slew_delta_pw_m"][0, -1]) < 1e-10
    assert variations[1] < .9 * variations[0]


def test_audit_separates_intra_window_cost_from_executed_seams():
    summary = {"validated_cycles": 2,
               "control_traces": {"last_pulse_width_m": np.array([200, 250, 300, 500, 450, 400]) * 1e-6}}
    args = SimpleNamespace(stimulations_per_cycle=3, pulse_width_max_step_us=100,
                           pulse_width_slew_weight=2, pulse_width_slew_reference_us=100)
    attach_slew_audit(summary, args)
    audit = summary["pulse_width_slew_regularization_audit"]
    assert audit["mean_squared_normalized_intra_window_change"] == pytest.approx(.25)
    assert audit["mean_weighted_intra_window_cost"] == pytest.approx(.5)
    assert audit["actual_rho_seam_penalized"] is False
    assert summary["pulse_width_slew_audit"]["muscles"]["m"]["maximum_executed_cycle_seam_change_us"] == pytest.approx(200)


def test_direct_audit_does_not_report_nonexistent_auxiliary_variables():
    summary = {"validated_cycles": 1,
               "control_traces": {"last_pulse_width_m": np.array([200, 250, 300]) * 1e-6}}
    args = SimpleNamespace(stimulations_per_cycle=3, pulse_width_max_step_us=100,
                           pulse_width_slew_weight=.1, pulse_width_slew_reference_us=100,
                           pulse_width_slew_formulation="direct_constraints")
    attach_slew_audit(summary, args)
    assert summary["pulse_width_slew_audit"]["auxiliary_control_representation"] is None
    assert summary["pulse_width_slew_regularization_audit"]["final_auxiliary_increment"] is None
    assert summary["pulse_width_slew_regularization_audit"]["reported_cost_excludes_final_auxiliary_increment"] is False


def test_comparison_json_preserves_regularization_audit():
    from examples.fes_multibody.cycling.cycling_fes_solver_comparison import solver_overview_rows
    from tests.shard1.test_periodic_pulse_width import _benchmark_result
    result = _benchmark_result([0, 0, 0], solver_success=True, success=True)
    result["args"].pulse_width_slew_weight = .1
    attach_slew_audit(result, result["args"])
    row = solver_overview_rows({"ipopt": result})[0]
    assert row["pulse_width_slew_regularization_audit"] == result["pulse_width_slew_regularization_audit"]


@pytest.mark.parametrize("solver,mode,model_config", [
    ("ipopt", "rho", None), ("madnlp", "rho", None), ("acados", "rho", None),
    ("ipopt", "rho-physio", None), ("ipopt", "rho", "model.json"),
])
def test_config_gui_and_all_launcher_paths_preserve_regularization(solver, mode, model_config):
    config = SimulationConfig(
        solver=solver, mode=mode, model_config=model_config, weights_config="weights.json",
        integration="irk" if solver == "acados" else "radau", compile_evaluators=False,
        acados_ipopt_cycle1_seed="seed.npz", pulse_width_max_step_us=100,
        pulse_width_slew_weight=.1, pulse_width_slew_reference_us=80,
    )
    assert SimulationConfig.from_json(config.to_json()) == config
    assert config_from_form(form_values(config)) == config
    assert "raccord RHO seulement borné" in scientific_summary(config)
    plan = build_launch_plan(config, Path("/runtime"))
    argv = plan.argv
    if "BENCHMARK_EXTRA_ARGUMENTS_JSON" in plan.environment_updates:
        argv = json.loads(plan.environment_updates["BENCHMARK_EXTRA_ARGUMENTS_JSON"])
    assert argv[argv.index("--pulse-width-slew-weight") + 1] == "0.1"
    assert argv[argv.index("--pulse-width-slew-reference-us") + 1] == "80.0"


@pytest.mark.parametrize("changes", [
    {"pulse_width_slew_weight": 1}, {"pulse_width_slew_weight": -1},
    {"pulse_width_slew_reference_us": 0}, {"pulse_width_slew_reference_us": float("nan")},
    {"pulse_width_slew_weight": True},
    {"pulse_width_max_step_us": 100, "pulse_width_slew_weight": 1, "stimulations_per_cycle": 1},
])
def test_capabilities_reject_invalid_regularization(changes):
    assert not CapabilityRegistry.supports(replace(SimulationConfig(), **changes))


def test_both_cli_parsers_expose_regularization_and_signature_changes():
    from examples.fes_multibody.cycling import cycling_fes_solver_comparison as comparison
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as periodic
    for module in (comparison, periodic):
        parser = module.build_argument_parser()
        defaults = parser.parse_args([])
        assert defaults.pulse_width_slew_weight == 0
        assert defaults.pulse_width_slew_reference_us == 100
        args = parser.parse_args(["--pulse-width-max-step-us", "100", "--pulse-width-slew-weight", ".1"])
        assert pulse_width_slew_signature(args) != pulse_width_slew_signature(defaults)


@pytest.mark.parametrize("solver", ["ipopt", "madnlp"])
def test_comparison_forwards_direct_slew_formulation_to_actual_solver(monkeypatch, solver):
    from examples.fes_multibody.cycling import cycling_fes_solver_comparison as comparison
    captured = {}
    monkeypatch.setattr(comparison, "_run_benchmark_case",
                        lambda name, args, **kwargs: captured.setdefault(name, {"args": args}))
    monkeypatch.setattr(comparison, "print_solver_overview", lambda _: None)
    comparison.main(solvers=(solver,), n_windows=3, stimulations_per_cycle=30,
                    mechanical_formulation="reduced", pulse_width_max_step_us=100,
                    pulse_width_slew_weight=.1, pulse_width_slew_formulation="direct_constraints")
    args = captured[solver]["args"]
    assert args.pulse_width_slew_formulation == "direct_constraints"
    assert args.pulse_width_slew_weight == .1
    assert args.stimulations_per_cycle == 30


@pytest.mark.parametrize("field,value", [
    ("pulse_width_slew_weight", 0), ("pulse_width_slew_reference_us", 80),
    ("pulse_width_slew_normalization", "wrong"),
])
def test_seed_provenance_rejects_a_different_regularization(field, value):
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as periodic
    args = periodic.build_argument_parser().parse_args([
        "--pulse-width-max-step-us", "100", "--pulse-width-slew-weight", ".1",
    ])
    args.terminal_wheel_q_reference_mode = "absolute_initial"
    metadata = periodic._common_initial_solution_metadata(args)
    metadata[field] = value
    with pytest.raises(ValueError, match=field):
        periodic._validate_common_initial_solution_metadata(SimpleNamespace(metadata=metadata), args, Path("seed.npz"))
