import json
import math
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from cocofest.optimization.rho_pace import (
    RhoPaceConfig, RhoPaceController, update_bioptim_fatigue_cost, weighted_fatigue_residual,
)


def controller(tmp_path=None, **kwargs):
    return RhoPaceController(
        ("m0", "m1"), kwargs.pop("initial_weights", (1., 1.)),
        signed_crank_torque_nm=0.22, parameters={"m0": {"a_scale": 100.}, "m1": {"a_scale": 100.}},
        initial_weight_basis="uniform test fixture; no FHO", journal_path=(
            tmp_path / "pace.jsonl" if tmp_path else None), **kwargs)


def apply(weights):
    return {"ocp_cost_updated": True, "weights": weights}


def boundary(policy, index, ratios=(.5, 1.), **kwargs):
    options = dict(certified=True, signed_crank_torque_nm=.22, apply_weights=apply)
    options.update(kwargs)
    return policy.boundary(index, ratios, **options)


def test_slow_updates_preserve_common_scale_and_limit_change(tmp_path):
    policy = controller(tmp_path, config=RhoPaceConfig(update_every_cycles=5))
    assert boundary(policy, 0)["status"] == "applied"
    for cycle in range(1, 5):
        assert boundary(policy, cycle)["status"] == "held"
        assert policy.weights == pytest.approx((1., 1.))
    assert boundary(policy, 5)["status"] == "applied"
    assert policy.weights[0] > policy.weights[1]
    assert math.prod(policy.weights) == pytest.approx(1.)
    assert max(abs(math.log(w)) for w in policy.weights) <= policy.config.max_log_step + 1e-12
    rows = [json.loads(line) for line in (tmp_path / "pace.jsonl").read_text().splitlines()]
    assert rows[0]["uses_fho_data"] is False
    assert rows[-1]["weights_before"] == pytest.approx([1., 1.])
    assert rows[-1]["ocp_cost_connected"] is True
    assert rows[0]["parameters"]["m0"]["a_scale"] == 100


def test_positive_small_physiology_weights_are_projected_and_audited():
    policy = controller(initial_weights=(.00001, 1.))
    assert policy.initial_weights == pytest.approx((.25, 4.))
    assert policy.events[0]["supplied_initial_weights"] == [.00001, 1.]
    assert math.prod(policy.initial_weights) == pytest.approx(1.)


@pytest.mark.parametrize("changes,reason", [
    ({"certified": False}, "uncertified_source_state"),
    ({"signed_crank_torque_nm": .23}, "resistance_changed"),
    ({"ratios": (0., 1.)}, "invalid_capacity_ratios"),
    ({"ratios": (float("nan"), 1.)}, "invalid_capacity_ratios"),
    ({"apply_weights": None}, "ocp_cost_not_integrated"),
])
def test_refusals_never_change_cost(changes, reason):
    policy = controller()
    event = boundary(policy, 0, **changes)
    assert event["status"] == "refused"
    assert reason in event["reasons"]
    assert policy.connected is False
    assert policy.weights == pytest.approx((1., 1.))


def test_missing_or_failed_integration_is_not_an_applied_arm():
    policy = controller()
    with pytest.raises(RuntimeError, match="did not confirm"):
        boundary(policy, 0, apply_weights=lambda _: {})
    assert policy.events[-1]["status"] == "fatal"
    assert not policy.connected
    assert boundary(policy, 5)["reasons"] == ["initial_cost_not_connected"]


def test_duplicate_and_maximum_cycle_are_rejected():
    policy = controller()
    boundary(policy, 0)
    assert "duplicate_or_out_of_order_cycle" in boundary(policy, 0)["reasons"]
    assert "maximum_cycles_reached" in boundary(policy, 100)["reasons"]
    assert RhoPaceConfig(max_cycles=1000).max_cycles == 1000
    assert RhoPaceConfig().update_every_cycles == 20
    assert RhoPaceConfig().projection_horizon_cycles == 500


def test_worker_cpu_ids_are_validated_and_json_friendly():
    assert RhoPaceConfig(projection_worker_cpu_ids=[4, 5]).projection_worker_cpu_ids == (4, 5)
    with pytest.raises(ValueError, match="duplicates"):
        RhoPaceConfig(projection_worker_cpu_ids=[4, 4])
    with pytest.raises(ValueError, match="nonnegative"):
        RhoPaceConfig(projection_worker_cpu_ids=[-1])


def test_static_physio_applies_initial_weights_once_even_at_update_boundaries(tmp_path):
    policy = controller(tmp_path, initial_weights=(.5, 2.),
                        config=RhoPaceConfig(adaptation_enabled=False, update_every_cycles=1))
    calls = []
    def writer(weights):
        calls.append(weights)
        return apply(weights)
    assert boundary(policy, 0, apply_weights=writer)["status"] == "applied"
    initial = policy.weights
    for cycle in (1, 2, 5, 20, 99):
        event = boundary(policy, cycle, ratios=(.01, .9), apply_weights=writer)
        assert event["status"] == "held"
        assert event["reasons"] == ["static_physiological_weights"]
        assert event["arm"] == "RHO-Physio"
        assert policy.weights == initial
    assert calls == [initial]
    assert policy.events[0]["uses_fho_data"] is False


def test_predictive_strategy_applies_only_a_valid_slow_proposal():
    policy = controller(config=RhoPaceConfig(
        adaptation_strategy="predictive_moment", update_every_cycles=2, projection_fatigue_guard=False,
    ))
    assert boundary(policy, 0)["status"] == "applied"
    held = boundary(policy, 1)
    assert held["status"] == "held"
    assert held["reasons"] == ["slow_update_not_due"]
    missing = boundary(policy, 2)
    assert missing["status"] == "held"
    assert missing["reasons"] == ["predictive_proposal_unavailable"]

    # A proposal is normalized and bounded before it reaches the objective.
    applied = boundary(policy, 4, predictive_weights=(4., .25),
                       predictive_audit={"selection_basis": "completed_horizon_margin"})
    assert applied["status"] == "applied"
    assert policy.weights == pytest.approx((4., .25))
    assert applied["predictive_audit"]["selection_basis"] == "completed_horizon_margin"


@pytest.mark.parametrize("strategy", ("", "future_magic"))
def test_unknown_adaptation_strategy_is_rejected(strategy):
    with pytest.raises(ValueError, match="adaptation_strategy"):
        RhoPaceConfig(adaptation_strategy=strategy)


def test_predictive_guard_rejects_an_unscored_or_oversized_step():
    policy = controller(config=RhoPaceConfig(adaptation_strategy="predictive_moment"))
    boundary(policy, 0)
    refused = boundary(policy, 20, predictive_weights=(1.05, 1 / 1.05))
    assert refused["status"] == "held"
    assert policy.weights == (1., 1.)
    refused = boundary(policy, 40, predictive_weights=(2., .5),
                       predictive_audit={"selection_basis": "guarded_fatigue_improvement"})
    assert refused["status"] == "held"
    assert policy.weights == (1., 1.)
    accepted = boundary(policy, 60, predictive_weights=(1.05, 1 / 1.05),
                        predictive_audit={"selection_basis": "guarded_fatigue_improvement"})
    assert accepted["status"] == "applied"
    assert policy.weights == pytest.approx((1.05, 1 / 1.05))


@pytest.mark.parametrize("declared", [
    {"initial_weight_basis": "fixture"},
    {"initial_weights": {"m0": 1.}, "initial_weight_basis": " "},
    {"initial_weights": {"m0": 0.}, "initial_weight_basis": "fixture"},
    {"initial_weights": {"m0": float("nan")}, "initial_weight_basis": "fixture"},
])
def test_static_physio_requires_named_positive_weights_and_provenance(declared):
    from scripts.run_rho_pace_benchmark import validate_weight_configuration
    with pytest.raises(ValueError):
        validate_weight_configuration(declared, RhoPaceConfig(adaptation_enabled=False))


@pytest.mark.parametrize("enabled", [0, 1, "false", None])
def test_adaptation_switch_must_be_boolean(enabled):
    with pytest.raises(ValueError, match="boolean"):
        RhoPaceConfig(adaptation_enabled=enabled)


def test_symbolic_cost_and_gradient_really_use_relative_weights():
    capacity = ca.SX.sym("capacity", 2)
    fake = SimpleNamespace(
        model=SimpleNamespace(muscles_dynamics_model=[
            SimpleNamespace(muscle_name="m0", a_scale=100.),
            SimpleNamespace(muscle_name="m1", a_scale=200.)]),
        states={"A_m0": SimpleNamespace(cx=capacity[0]), "A_m1": SimpleNamespace(cx=capacity[1])})
    residual = weighted_fatigue_residual(fake, (4., .25))
    cost = ca.sumsqr(residual)
    function = ca.Function("pace", [capacity], [cost, ca.gradient(cost, capacity)])
    actual, gradient = function([50., 100.])
    assert float(actual) == pytest.approx(1.0625)
    np.testing.assert_allclose(np.array(gradient).ravel(), [-.04, -.00125])


def test_real_bioptim_objective_replacement_changes_two_successive_solves():
    from bioptim import (BoundsList, ConfigureVariables, DynamicsEvaluation, DynamicsOptions,
                         InitialGuessList, Node, ObjectiveFcn, ObjectiveList,
                         OptimalControlProgram, Solver, StateDynamics)
    from cocofest import CustomObjective

    class CapacityModel(StateDynamics):
        def __init__(self):
            super().__init__()
            self.muscles_dynamics_model = [SimpleNamespace(muscle_name="m0", a_scale=100.)]

        def serialize(self):
            return CapacityModel, {}

        @property
        def name(self):
            return "pace_objective_test"

        @property
        def name_dofs(self):
            return ["dummy"]

        @property
        def state_configuration_functions(self):
            return [lambda ocp, nlp: ConfigureVariables.configure_new_variable(
                "A_m0", ["A_m0"], ocp, nlp, as_states=True)]

        @property
        def control_configuration_functions(self):
            return [lambda ocp, nlp: ConfigureVariables.configure_new_variable(
                "dummy", ["dummy"], ocp, nlp, as_controls=True)]

        @property
        def algebraic_configuration_functions(self):
            return []

        @property
        def extra_configuration_functions(self):
            return []

        def dynamics(self, time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp):
            return DynamicsEvaluation(dxdt=0 * states, defects=None)

    objectives = ObjectiveList()
    objectives.add(CustomObjective.minimize_overall_muscle_fatigue,
                   custom_type=ObjectiveFcn.Lagrange, node=Node.ALL, weight=10000., quadratic=True)
    xb, xi, ub, ui = BoundsList(), InitialGuessList(), BoundsList(), InitialGuessList()
    xb["A_m0"] = [50.], [50.]
    xi["A_m0"] = [50.]
    ub["dummy"] = [-1.], [1.]
    ui["dummy"] = [0.]
    ocp = OptimalControlProgram(CapacityModel(), 2, .1, dynamics=DynamicsOptions(),
                               objective_functions=objectives, x_bounds=xb, x_init=xi,
                               u_bounds=ub, u_init=ui, use_sx=True)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_linear_solver("mumps")
    baseline = ocp.solve(solver)
    receipt = update_bioptim_fatigue_cost(ocp, (1.,))
    assert receipt["ocp_cost_updated"] is True
    first = ocp.solve(solver)
    update_bioptim_fatigue_cost(ocp, (2.,))
    second = ocp.solve(solver)
    assert baseline.status == first.status == second.status == 0
    # A unit-weight PACE attachment must preserve the original objective,
    # including Bioptim's time integration and original global multiplier.
    assert float(first.cost) == pytest.approx(float(baseline.cost), rel=1e-13)
    assert float(second.cost) == pytest.approx(2 * float(first.cost))
    assert len(ocp.nlp[0].J) == 1


def test_launcher_restricts_to_supported_real_dynamic_rho():
    from scripts.run_rho_pace_benchmark import validate_benchmark_arguments
    from examples.fes_multibody.cycling.cycling_fes_solver_comparison import build_cli
    args = build_cli().parse_args(["--solvers", "ipopt", "--mechanical-formulation", "reduced",
                                   "--compact-rho-output", "--signed-crank-torque", ".22",
                                   "--n-windows", "100"])
    assert validate_benchmark_arguments(args, RhoPaceConfig()) == .22
    args.ipopt_c_compile = True
    with pytest.raises(ValueError, match="ipopt_c_compile"):
        validate_benchmark_arguments(args, RhoPaceConfig())


def test_launcher_executes_patched_canonical_cycling_class_and_writes_journal(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from scripts.run_rho_pace_benchmark import main
    from examples.fes_multibody.cycling import cycling_fes_solver_comparison as benchmark
    from examples.fes_multibody.cycling.cycling_pulse_width_mhe import MyCyclicNMPC
    from cocofest.optimization.fes_nmpc_multibody import FesNmpcMsk
    from cocofest.optimization import rho_pace
    original = FesNmpcMsk.solve_fes_nmpc
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"initial_weight_basis": "uniform fixture"}))
    journal = tmp_path / "pace.jsonl"
    model = SimpleNamespace(muscle_name="m0", **dict.fromkeys(
        ("a_scale", "alpha_a", "alpha_tau1", "alpha_km", "tau_fat",
         "tau1_rest", "km_rest", "tauc", "tau2", "pd0", "pdt"), 1.))
    nmpc = object.__new__(MyCyclicNMPC)
    nmpc.nlp = [SimpleNamespace(model=SimpleNamespace(muscles_dynamics_model=[model]),
                               x_bounds={"A_m0": SimpleNamespace(min=np.ones((1, 1)))})]
    monkeypatch.setattr(FesNmpcMsk, "solve_fes_nmpc", lambda self, callback, **kw: callback(self, 0, None))
    patched_original = FesNmpcMsk.solve_fes_nmpc
    monkeypatch.setattr(rho_pace, "update_bioptim_fatigue_cost", lambda ocp, weights: apply(weights))
    backend_directory = tmp_path / "backend"
    backend_directory.mkdir()
    monkeypatch.chdir(tmp_path)

    def benchmark_with_directory_change(**kwargs):
        # The actual cycling main changes cwd to EXAMPLE_DIR before solving.
        os.chdir(backend_directory)
        return nmpc.solve_fes_nmpc(lambda *a: True)

    monkeypatch.setattr(benchmark, "main", benchmark_with_directory_change)
    main(["--pace-config", config.name, "--pace-journal", journal.name, "--", "--solvers", "ipopt",
          "--mechanical-formulation", "reduced", "--compact-rho-output", "--signed-crank-torque", ".22"])
    assert Path.cwd() == tmp_path
    assert not (backend_directory / journal.name).exists()
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert rows[-1]["event"] == "launcher_completed"
    assert any(row.get("status") == "applied" for row in rows)
    assert FesNmpcMsk.solve_fes_nmpc is patched_original


def test_launcher_cannot_succeed_when_benchmark_bypasses_cost_attachment(tmp_path, monkeypatch):
    from scripts.run_rho_pace_benchmark import main
    from examples.fes_multibody.cycling import cycling_fes_solver_comparison as benchmark
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"initial_weight_basis": "uniform fixture"}))
    monkeypatch.setattr(benchmark, "main", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="connected PACE cost and journal"):
        main(["--pace-config", str(config), "--pace-journal", str(tmp_path / "pace.jsonl"), "--",
              "--solvers", "ipopt", "--mechanical-formulation", "reduced",
              "--compact-rho-output", "--signed-crank-torque", ".22"])
