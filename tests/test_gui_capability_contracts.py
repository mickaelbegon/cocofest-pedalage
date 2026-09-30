"""Headless regression checks for GUI options that change the scientific NLP."""
import json

import pytest

from cocofest.simulation import CapabilityRegistry, ConfigurationError, SimulationConfig, build_launch_plan
from cocofest.simulation.gui_model import config_from_form, scientific_summary
from cocofest.simulation.independent_arms_gui_model import (
    IndependentArmsGuiConfig, PROCESS_FACTORY, independent_solver_choices,
)


@pytest.mark.parametrize("change", [
    {"mechanics": "full", "reduced_internal_crank_velocity_guard": "on"},
    {"formulation": "isokinetic", "reduced_internal_crank_velocity_guard": "on"},
    {"reduced_internal_crank_velocity_guard": "off"},
    {"reduced_internal_crank_velocity_guard": "auto"},
])
def test_terminal_prediction_rejects_an_inapplicable_or_inactive_guard(change):
    config = SimulationConfig(reduced_terminal_half_step_velocity_guard=True, **change)
    issues = CapabilityRegistry.validate(config)
    assert any(issue.field == "reduced_terminal_half_step_velocity_guard" for issue in issues)
    with pytest.raises(ConfigurationError):
        config_from_form({"reduced_terminal_half_step_velocity_guard": True}, base=config)


@pytest.mark.parametrize("mode", ["rho", "rho-physio", "rho-pace"])
@pytest.mark.parametrize("configured", [False, True])
def test_terminal_prediction_reaches_each_supported_launcher(tmp_path, mode, configured):
    config = SimulationConfig(
        mode=mode, model_config="model.json" if configured else None,
        weights_config="weights.json" if mode != "rho" else None,
        compile_evaluators=False, reduced_internal_crank_velocity_guard="on",
        reduced_terminal_half_step_velocity_guard=True,
    )
    plan = build_launch_plan(config, tmp_path / "env", tmp_path)
    extras = (plan.argv if mode != "rho" or configured else
              json.loads(plan.environment_updates["BENCHMARK_EXTRA_ARGUMENTS_JSON"]))
    assert "--reduced-terminal-half-step-velocity-guard" in extras
    assert extras[extras.index("--reduced-internal-crank-velocity-guard") + 1] == "on"
    summary = scientific_summary(config)
    assert "Euler" in summary and "ne certifie pas toutes" in summary


@pytest.mark.parametrize("solver,extras", [
    ("acados", ()),
    ("ipopt", ("--common-initial-solution-output", "seed.npz")),
])
def test_auto_terminal_guard_is_effective_for_cycle1_seed_and_consumer(tmp_path, solver, extras):
    config = SimulationConfig(
        solver=solver, integration="irk" if solver == "acados" else "radau",
        acados_ipopt_cycle1_seed="seed.npz" if solver == "acados" else None,
        extra_arguments=extras, reduced_terminal_half_step_velocity_guard=True,
    )
    assert CapabilityRegistry.effective_reduced_velocity_guard(config)
    assert CapabilityRegistry.supports(config)
    plan = build_launch_plan(config, tmp_path / "env", tmp_path)
    argv = plan.argv if solver == "acados" else json.loads(plan.environment_updates["BENCHMARK_EXTRA_ARGUMENTS_JSON"])
    assert argv[argv.index("--reduced-internal-crank-velocity-guard") + 1] == "on"


@pytest.mark.parametrize("solver", ["ipopt", "madnlp"])
@pytest.mark.parametrize("configured", [False, True])
def test_direct_control_constraints_roundtrip_and_forwarding(tmp_path, solver, configured):
    config = config_from_form({"solver": solver, "pulse_width_max_step_us": "100",
                               "pulse_width_slew_formulation": "direct_constraints",
                               "model_config": "model.json" if configured else ""})
    assert SimulationConfig.from_json(config.to_json()) == config
    plan = build_launch_plan(config, tmp_path / "env", tmp_path)
    argv = plan.argv if configured else json.loads(plan.environment_updates["BENCHMARK_EXTRA_ARGUMENTS_JSON"])
    assert argv[argv.index("--pulse-width-slew-formulation") + 1] == "direct_constraints"
    assert plan.resolved_config.to_dict()["effective"]["transcription"]["pulse_width_slew_formulation"] == "direct_constraints"


@pytest.mark.parametrize("solver", ["fatrop", "acados"])
def test_direct_control_constraints_reject_unconnected_solvers(solver):
    issues = CapabilityRegistry.validate(SimulationConfig(
        solver=solver, integration="irk" if solver == "acados" else "radau",
        acados_ipopt_cycle1_seed="seed.npz", pulse_width_slew_formulation="direct_constraints",
    ))
    assert any(issue.field == "pulse_width_slew_formulation" for issue in issues)


def test_default_independent_factory_does_not_advertise_acados():
    assert independent_solver_choices(PROCESS_FACTORY) == ("ipopt",)
    with pytest.raises(ValueError, match="seulement IPOPT"):
        IndependentArmsGuiConfig(factory=PROCESS_FACTORY, solver="acados").validate()
    assert independent_solver_choices("example:build") == ("ipopt", "acados")
    IndependentArmsGuiConfig(factory="example:build", solver="acados").validate()


@pytest.mark.parametrize("solver", ["ipopt", "acados"])
def test_prescribed_isokinetic_kinematics_reaches_solver_and_provenance(tmp_path, solver):
    config = SimulationConfig(
        solver=solver,
        formulation="isokinetic",
        isokinetic_kinematics="prescribed",
        integration="irk" if solver == "acados" else "radau",
        acados_ipopt_cycle1_seed="cycle1.npz" if solver == "acados" else None,
    )
    assert SimulationConfig.from_json(config.to_json()) == config
    plan = build_launch_plan(config, tmp_path / "env", tmp_path)
    if solver == "ipopt":
        assert plan.environment_updates["BENCHMARK_ISOKINETIC_KINEMATICS"] == "prescribed"
        assert "prescribed-kinematics" in str(plan.result_json)
    else:
        assert plan.argv[plan.argv.index("--isokinetic-kinematics") + 1] == "prescribed"
    effective = plan.resolved_config.to_dict()["effective"]
    assert effective["transcription"]["isokinetic_kinematics"] == "prescribed"
    assert "cinématique prescribed" in scientific_summary(config)


@pytest.mark.parametrize("change", [
    {"formulation": "dynamic"},
    {"mechanics": "full"},
    {"formulation": "dynamic", "mechanics": "full"},
])
def test_prescribed_kinematics_rejects_incompatible_mechanics(change):
    config = SimulationConfig(isokinetic_kinematics="prescribed", **change)
    assert any(issue.field == "isokinetic_kinematics" for issue in CapabilityRegistry.validate(config))
    with pytest.raises(ConfigurationError):
        CapabilityRegistry.require_valid(config)


@pytest.mark.parametrize("change,match", [
    ({"cycles_per_window": 2}, "fenêtres d'un cycle"),
    ({"parallel": False}, "exécution parallèle"),
    ({"right_runner_config": "worker.json"}, "configuration worker"),
])
def test_default_independent_factory_fails_before_spawning_unsupported_requests(change, match):
    with pytest.raises(ValueError, match=match):
        IndependentArmsGuiConfig(factory=PROCESS_FACTORY, **change).validate()
