"""Headless GUI behavior and real subprocess lifecycle; no scientific solves."""
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from cocofest.simulation import LaunchPlan, SimulationConfig, build_launch_plan
from cocofest.simulation.execution import SimulationProcess
from cocofest.simulation.gui_model import (
    FORM_FIELDS, config_from_form, form_values, result_summary, scientific_summary,
)
from cocofest.simulation.bayesian_model import BayesianCampaignConfig, campaign_summary
from cocofest.simulation.bayesian_runner import FAILURE_PENALTY, _metric
from cocofest.simulation.gui import muscle_weight_campaign, muscle_weight_search_space

ROOT = Path(__file__).resolve().parents[1]


def test_form_roundtrip_preserves_all_configuration_fields():
    config = SimulationConfig(threads=7, numeric_threads=2, pulse_width_max_step_us=100,
                              extra_arguments=("--standard-warmup-seed", "path with spaces.npz"))
    assert config_from_form(form_values(config), config) == config
    assert set(form_values(config)) == {spec.name for spec in FORM_FIELDS}
    assert config_from_form({"cycles": "17"}, config) == replace(config, cycles=17)


def test_bayesian_campaign_is_json_safe_and_certifies_result_before_scoring(tmp_path):
    campaign = BayesianCampaignConfig(
        base_config=SimulationConfig(compile_evaluators=False).to_dict(), output_root=str(tmp_path / "bo"),
        solvers=("ipopt", "fatrop"), n_calls=4, n_initial_points=2,
        collocation_degree_min=3, collocation_degree_max=3, thread_choices=(1, 2),
    ).validate()
    assert BayesianCampaignConfig.from_json(campaign.to_json()) == campaign
    assert "essais" in campaign_summary(campaign)
    success = {"success": True, "validated_cycles": 100, "solver_time_per_cycle_s": 0.25}
    assert _metric(success, 100, "solver_time") == .25
    assert _metric({**success, "validated_cycles": 99}, 100, "solver_time") == FAILURE_PENALTY
    assert _metric({**success, "solver_time_per_cycle_s": None}, 100, "solver_time") == FAILURE_PENALTY
    assert _metric({"continuous_endurance": {"available": True, "continuous_cycle_equivalent": 42.25}}, 100,
                   "continuous_endurance") == -42.25


def test_bayesian_campaign_rejects_larger_budget_than_its_discrete_space(tmp_path):
    campaign = BayesianCampaignConfig(
        base_config=SimulationConfig().to_dict(), output_root=str(tmp_path / "bo"), solvers=("ipopt",),
        n_calls=3, n_initial_points=1, collocation_degree_min=3, collocation_degree_max=3, thread_choices=(1,),
    )
    with pytest.raises(ValueError, match="configurations distinctes"):
        campaign.validate()


def test_gui_muscle_weight_campaign_has_complete_named_log_domains(tmp_path):
    model = tmp_path / "model.json"
    weights = tmp_path / "weights.json"
    model.write_text(json.dumps({"muscles": {"Delt_ant": {}, "Triceps": {}}}))
    weights.write_text("{}")
    base = SimulationConfig(mode="rho-physio", solver="ipopt", model_config=str(model), weights_config=str(weights))
    campaign = muscle_weight_campaign(
        base, output_root=tmp_path / "bo", study_name="gui-muscles", muscle_names="Delt_ant, Triceps",
        scale_min="0.05", scale_max="20", trials=12, workers=3, startup_trials=4,
        seed=42, max_cycles=500, timeout_s=1200,
    )
    assert campaign.study_kind == "controller"
    assert campaign.metric == "continuous_endurance"
    assert campaign.search_space == {
        "muscle_weight__Delt_ant": {"type": "float", "low": .05, "high": 20., "log": True},
        "muscle_weight__Triceps": {"type": "float", "low": .05, "high": 20., "log": True},
    }
    with pytest.raises(ValueError, match="une seule fois"):
        muscle_weight_search_space("Delt_ant, Delt_ant", .05, 20)
    with pytest.raises(ValueError, match="exactement un poids"):
        muscle_weight_campaign(
            base, output_root=tmp_path / "bo2", study_name="bad", muscle_names="Delt_ant",
            scale_min=.05, scale_max=20, trials=12, workers=3, startup_trials=4,
            seed=42, max_cycles=500, timeout_s=1200,
        )


@pytest.mark.parametrize("name,value", [("cycles", "1.5"), ("threads", ""),
                                        ("pulse_width_max_step_us", "invalid"),
                                        ("extra_arguments", "{not json}"),
                                        ("extra_arguments", "{}"),
                                        ("compile_evaluators", "false")])
def test_invalid_form_values_are_never_coerced_silently(name, value):
    with pytest.raises(ValueError):
        config_from_form({name: value})


def test_optional_slew_and_paths_can_be_cleared():
    config = SimulationConfig(pulse_width_max_step_us=100, weights_config="weights.json")
    result = config_from_form({"pulse_width_max_step_us": "", "weights_config": ""}, config)
    assert result.pulse_width_max_step_us is result.weights_config is None


def test_form_capability_validation_does_not_change_solver_or_compile_choice():
    with pytest.raises(ValueError, match="IRK"):
        config_from_form({"solver": "acados", "acados_ipopt_cycle1_seed": "seed.npz"})
    with pytest.raises(ValueError, match="compile_evaluators"):
        config_from_form({"mode": "rho-physio", "weights_config": "weights.json"})
    with pytest.raises(ValueError, match="FHO is currently connected only"):
        config_from_form({"mode": "fho", "solver": "acados", "integration": "irk",
                          "acados_ipopt_cycle1_seed": "cycle-1.npz"})


def test_form_exposes_bilateral_reduced_selection():
    config = config_from_form({"bilateral_reduced": True})
    assert config.bilateral_reduced is True
    assert "bilatéral" in scientific_summary(config)


def test_summary_states_control_boundary_and_torque_convention():
    summary = scientific_summary(SimulationConfig(pulse_width_max_step_us=100))
    assert "u[k+1]" in summary and "raccord entre RHO" in summary
    assert "positif = résistance" in summary
    assert "cycle 1 obligatoire" in summary


@pytest.mark.parametrize("payload,expected", [
    ({}, "indisponible"),
    ({"results": []}, "indisponible"),
    ({"results": [{"status": 0}]}, "partiel"),
    ({"results": [{"success": True, "validated_cycles": 99}]}, "partiel"),
    ({"results": [{"success": False, "validated_cycles": 100}]}, "échec"),
    ({"results": [{"success": True, "validated_cycles": 100}]}, "réussite déclarée"),
    ({"results": ["invalid"]}, "incomplet"),
])
def test_process_success_is_not_scientific_success(payload, expected):
    assert expected in result_summary(payload, 100)


def wait_finished(process, timeout=5):
    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(.01)
    assert process.poll() is not None
    process.reader.join(timeout=1)


def test_subprocess_passes_arguments_as_data_and_streams_to_disk(tmp_path):
    literal = "$(touch NEVER); a b"
    plan = LaunchPlan((sys.executable, "-c", "import sys; print(sys.argv[1]); sys.exit(3)", literal),
                      tmp_path / "cwd", {}, "rho32", tmp_path / "result.json")
    process = SimulationProcess()
    process.start(plan, dict(os.environ), tmp_path / "log.txt")
    wait_finished(process)
    assert process.returncode == 3
    assert "".join(process.drain()) == literal + "\n"
    assert (tmp_path / "log.txt").read_text() == literal + "\n"
    assert not (tmp_path / "cwd/NEVER").exists()


def test_existing_result_and_log_are_preserved(tmp_path):
    result = tmp_path / "result.json"
    result.write_text("existing scientific result")
    plan = LaunchPlan((sys.executable, "-c", "pass"), tmp_path, {}, "rho32", result)
    process = SimulationProcess()
    with pytest.raises(FileExistsError):
        process.start(plan, dict(os.environ), tmp_path / "new.log")
    assert result.read_text() == "existing scientific result"
    assert not (tmp_path / "new.log").exists()


def test_launch_failure_closes_and_removes_only_its_empty_log(tmp_path):
    plan = LaunchPlan((str(tmp_path / "missing-program"),), tmp_path, {}, "rho32")
    process = SimulationProcess()
    with pytest.raises(FileNotFoundError):
        process.start(plan, dict(os.environ), tmp_path / "log.txt")
    assert not process.running
    assert not (tmp_path / "log.txt").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cancellation")
def test_stop_escalates_when_a_descendant_outlives_the_launcher(tmp_path):
    child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('child-ready',flush=True); time.sleep(60)"
    program = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-u','-c',{child!r}]); time.sleep(60)"
    plan = LaunchPlan((sys.executable, "-u", "-c", program), tmp_path, {}, "rho32")
    process = SimulationProcess()
    process.start(plan, dict(os.environ), tmp_path / "log.txt")
    try:
        deadline = time.monotonic() + 5
        seen = ""
        while "child-ready" not in seen and time.monotonic() < deadline:
            seen += "".join(process.drain())
            time.sleep(.01)
        assert "child-ready" in seen
        process.stop()
        process.process.wait(timeout=5)
        assert process.running  # inherited stdout remains open in the child
        process.stop_requested_at -= 6
        process.poll()
        process.reader.join(timeout=5)
        assert not process.running
    finally:
        try:
            os.killpg(process.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.process.wait(timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cancellation")
def test_stop_terminates_only_owned_process_group(tmp_path):
    plan = LaunchPlan((sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(60)"),
                      tmp_path, {}, "rho32")
    process = SimulationProcess()
    process.start(plan, dict(os.environ), tmp_path / "log.txt")
    try:
        with pytest.raises(RuntimeError, match="déjà"):
            process.start(plan, dict(os.environ), tmp_path / "second.log")
        assert os.getpgid(process.process.pid) == process.process.pid
        assert os.getpgid(process.process.pid) != os.getpgrp()
        process.stop()
        wait_finished(process)
        assert process.returncode == -signal.SIGTERM
    finally:
        if process.running:
            process.process.kill()
            process.process.wait(timeout=5)


def test_dry_run_cli_does_not_require_display_or_create_outputs(tmp_path):
    filename = tmp_path / "simulation.json"
    config = SimulationConfig(output_root=str(tmp_path / "new-output"))
    filename.write_text(config.to_json())
    environment = {**os.environ, "DISPLAY": ""}
    result = subprocess.run([sys.executable, str(ROOT / ".github/scripts/run_simulation_gui.py"),
                             "--config", str(filename), "--prefix", str(tmp_path / "missing-env"),
                             "--dry-run"], cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "run_cycling_benchmark_case.sh" in result.stdout
    assert sorted(path.name for path in tmp_path.iterdir()) == ["simulation.json"]


def test_acados_relative_seed_is_bound_to_project_for_gui(tmp_path):
    config = SimulationConfig(solver="acados", integration="irk", acados_ipopt_cycle1_seed="seeds/cycle1.npz")
    plan = build_launch_plan(config, tmp_path / "env", tmp_path)
    assert plan.argv[plan.argv.index("--common-initial-solution") + 1] == str(tmp_path / "seeds/cycle1.npz")


def test_gui_import_does_not_create_tk_or_import_scientific_stack():
    program = """
import sys
import cocofest.simulation.gui
assert 'tkinter' not in sys.modules
assert not any(m in sys.modules for m in ('casadi', 'bioptim', 'numpy'))
"""
    subprocess.run([sys.executable, "-c", program], cwd=ROOT, check=True)


def local_acados_config(**changes):
    return replace(SimulationConfig(solver="acados", integration="irk",
                   acados_ipopt_cycle1_seed="seeds/cycle1.npz",
                   stimulations_per_cycle=50,
                   acados_ding_local_reduction=True), **changes)


def test_local_ding_option_is_explicit_persisted_and_forwarded(tmp_path):
    defaults = SimulationConfig(solver="acados", integration="irk", acados_ipopt_cycle1_seed="seed.npz",
                                stimulations_per_cycle=50)
    assert form_values(defaults)["acados_ding_local_reduction"] is False
    assert "--acados-ding-local-reduction" not in build_launch_plan(defaults, tmp_path, tmp_path).argv
    config = config_from_form({"acados_ding_local_reduction": True}, defaults)
    restored = SimulationConfig.from_json(config.to_json())
    assert restored == config
    assert build_launch_plan(restored, tmp_path, tmp_path).argv.count("--acados-ding-local-reduction") == 1
    summary = scientific_summary(restored)
    assert "expérimentale activée" in summary and "Gauss-Legendre 4×5" in summary
    assert "multiplicateurs du problème complet non reconstruits" in summary


@pytest.mark.parametrize("changes,expected", [
    ({"solver": "ipopt"}, "solver"),
    ({"mechanics": "full"}, "mechanics"),
    ({"formulation": "isokinetic"}, "formulation"),
    ({"integration": "radau"}, "integration"),
    ({"acados_sim_stages": 5}, "acados_sim_stages"),
    ({"acados_sim_steps": 1}, "acados_sim_steps"),
    ({"stimulations_per_cycle": 30}, "stimulations_per_cycle"),
    ({"pulse_width_max_step_us": 100}, "pulse_width_max_step_us"),
    ({"reduced_internal_crank_velocity_guard": "off"}, "guard"),
    ({"mode": "fho"}, "mode"),
    ({"acados_ding_local_reduction": "false"}, "boolean"),
    ({"extra_arguments": ("--acados-ding-local-reduction",)}, "managed"),
    ({"extra_arguments": ("--acados-initial-irk-rollout",)}, "rollouts"),
    ({"extra_arguments": ("--acados-transfer-irk-rollout",)}, "rollouts"),
    ({"extra_arguments": ("--acados-initial-irk-r",)}, "rollouts"),
    ({"extra_arguments": ("--acados-nlp-solver-type", "SQP_RTI")}, "SQP"),
])
def test_local_ding_rejects_incompatible_configuration_before_launch(changes, expected, tmp_path):
    with pytest.raises(ValueError, match=expected):
        build_launch_plan(local_acados_config(**changes), tmp_path, tmp_path)


def test_local_ding_form_refuses_string_boolean():
    with pytest.raises(ValueError, match="booléen"):
        config_from_form({"acados_ding_local_reduction": "false"}, local_acados_config())


def local_ipopt_config(**changes):
    return replace(
        SimulationConfig(
            solver="ipopt",
            mechanics="reduced",
            formulation="dynamic",
            integration="radau",
            collocation_degree=5,
            cycles_per_window=1,
            stimulations_per_cycle=30,
            pulse_width_max_step_us=None,
            pulse_width_slew_weight=0.0,
            reduced_internal_crank_velocity_guard="off",
            compile_evaluators=False,
            ipopt_ding_local_reduction=True,
        ),
        **changes,
    )


def test_ipopt_local_ding_option_is_explicit_persisted_and_forwarded(tmp_path):
    defaults = replace(SimulationConfig(), compile_evaluators=False,
                       reduced_internal_crank_velocity_guard="off")
    assert form_values(defaults)["ipopt_ding_local_reduction"] is False
    assert "BENCHMARK_IPOPT_DING_LOCAL_REDUCTION" not in build_launch_plan(defaults, tmp_path, tmp_path).environment_updates
    config = config_from_form({"ipopt_ding_local_reduction": True}, defaults)
    restored = SimulationConfig.from_json(config.to_json())
    assert restored == config
    plan = build_launch_plan(restored, tmp_path, tmp_path)
    assert plan.environment_updates["BENCHMARK_IPOPT_DING_LOCAL_REDUCTION"] == "true"
    summary = scientific_summary(restored)
    assert "Réduction Ding IPOPT expérimentale activée" in summary
    assert "NLP complet" in summary


@pytest.mark.parametrize("changes,expected", [
    ({"solver": "madnlp"}, "solver"),
    ({"mechanics": "full"}, "mechanics"),
    ({"formulation": "isokinetic"}, "formulation"),
    ({"integration": "irk"}, "integration"),
    ({"collocation_degree": 3}, "collocation_degree"),
    ({"cycles_per_window": 2}, "cycles_per_window"),
    ({"stimulations_per_cycle": 50}, "stimulations_per_cycle"),
    ({"pulse_width_max_step_us": 100}, "pulse_width_max_step_us"),
    ({"pulse_width_slew_weight": 0.1}, "pulse_width_slew_weight"),
    ({"reduced_internal_crank_velocity_guard": "on"}, "guard vitesse"),
    ({"compile_evaluators": True}, "compile_evaluators"),
    ({"model_config": "muscles.json"}, "model_config"),
])
def test_ipopt_local_ding_rejects_incompatible_configuration_before_launch(changes, expected, tmp_path):
    with pytest.raises(ValueError, match=expected):
        build_launch_plan(local_ipopt_config(**changes), tmp_path, tmp_path)


def test_ipopt_local_ding_bilateral_plan_is_supported(tmp_path):
    plan = build_launch_plan(local_ipopt_config(bilateral_reduced=True), tmp_path, tmp_path)
    extras = json.loads(plan.environment_updates["BENCHMARK_EXTRA_ARGUMENTS_JSON"])
    assert plan.environment_updates["BENCHMARK_IPOPT_DING_LOCAL_REDUCTION"] == "true"
    assert "--bilateral-reduced" in extras


@pytest.mark.parametrize("mode,adaptation", [("rho-physio", False), ("rho-pace", True)])
def test_weighted_bridge_selects_fixed_or_adaptive_policy(monkeypatch, mode, adaptation):
    from cocofest.simulation.weighted_runner import main
    calls = []
    monkeypatch.setitem(sys.modules, "scripts.run_rho_pace_benchmark",
                        SimpleNamespace(main=lambda *args, **kwargs: calls.append((args, kwargs))))
    main(["--mode", mode, "--pace-config", "weights with spaces.json", "--pace-journal", "weights.jsonl",
          "--", "--solvers", "ipopt", "--n-windows", "100"])
    args, kwargs = calls[0]
    assert kwargs == {"adaptation_enabled": adaptation}
    assert args[0] == ["--pace-config", "weights with spaces.json", "--pace-journal", "weights.jsonl",
                       "--", "--solvers", "ipopt", "--n-windows", "100"]
