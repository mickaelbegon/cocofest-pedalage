"""Public configuration boundary: no scientific dependencies or actual solves."""
from dataclasses import fields, replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cocofest.simulation import SimulationConfig, build_launch_plan, resolve_config
from cocofest.simulation.gui_model import config_from_form, form_values
from cocofest.simulation.resolved_config import _GROUPS


def cli(*args):
    return subprocess.run([sys.executable, "-m", "cocofest.simulation.cli", *args],
                          text=True, capture_output=True)


def test_every_managed_field_is_classified():
    classified = [key for names in _GROUPS.values() for key in names.split()]
    assert len(classified) == len(set(classified))
    assert set(classified) == {f.name for f in fields(SimulationConfig)} - {"schema_version", "dry_run"}


@pytest.mark.parametrize("profile", ["K5", "K7", "K9"])
def test_json_gui_cli_campaign_are_the_same_configuration(profile, tmp_path):
    campaign = resolve_config({"cycles": 100}, profile=profile, root=tmp_path)
    config = campaign.config
    gui = resolve_config(config_from_form(form_values(config)), root=tmp_path)
    document = tmp_path / "config.json"
    document.write_text(config.to_json())
    result = cli("--config", str(document), "--root", str(tmp_path))
    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)
    plan = build_launch_plan(config, tmp_path / "env", tmp_path)
    assert campaign.config_hash == gui.config_hash == command["config_hash"] == plan.resolved_config.config_hash
    path = plan.save_effective_configuration()
    assert json.loads(path.read_text())["config_hash"] == campaign.config_hash
    assert plan.save_effective_configuration() == path
    different = build_launch_plan(replace(config, cycles=99), tmp_path / "env", tmp_path)
    with pytest.raises(FileExistsError, match="Different effective"):
        different.save_effective_configuration()


def test_profile_materializes_actual_feature_steps(tmp_path):
    k5, k7, k9 = [resolve_config(profile=p).config for p in ("K5", "K7", "K9")]
    assert not k5.compile_evaluators and not k5.compile_hessian_only and not k5.compact_rho_output
    assert k7.compile_hessian_only and k7.compact_rho_output
    assert k9.pulse_width_max_step_us == 100 and k9.pulse_width_slew_weight == 0.01
    plan = build_launch_plan(k7, tmp_path, tmp_path)
    extras = json.loads(plan.environment_updates["BENCHMARK_EXTRA_ARGUMENTS_JSON"])
    assert extras[extras.index("--ipopt-c-compile-callback") + 1] == "nlp_hess_l"
    assert "--ipopt-c-compiler-flag=-O1" in extras
    assert extras[extras.index("--ipopt-c-cache-dir") + 1].endswith(plan.resolved_config.config_hash[:16])
    assert plan.environment_updates["BENCHMARK_COMPACT_RHO_OUTPUT"] == "true"
    assert build_launch_plan(k5, tmp_path, tmp_path).environment_updates["BENCHMARK_COMPACT_RHO_OUTPUT"] == "false"


@pytest.mark.parametrize("values,profile,match", [
    ({"collocation_degree": 3}, "K7", "conflicts"),
    ({"compile_evaluators": True}, "K5", "conflicts"),
    ({"pulse_width_slew_weight": .1}, "K9", "conflicts"),
    ({"mystery": 1}, None, "Unknown"),
    ({"extra_arguments": ["--unknown-dynamics"]}, None, "Unknown advanced"),
    ({"extra_arguments": ["--ipopt-c-compile-callback", "nlp_hess_l"]}, None, "managed"),
    ({"extra_arguments": ["--ipopt-no-use-sx"]}, "K5", "managed"),
    ({"extra_arguments": ["--ipopt-c-cache-dir", "shared"]}, "K7", "private cache"),
])
def test_conflicts_and_unknowns_fail_before_nlp(values, profile, match):
    with pytest.raises(ValueError, match=match):
        resolve_config(values, profile=profile)


def test_hash_normalizes_paths_numeric_types_and_ignores_preview(tmp_path):
    first = resolve_config({"signed_crank_torque": 1, "output_root": "a/../b"}, root=tmp_path)
    second = resolve_config({"signed_crank_torque": 1.0, "output_root": str(tmp_path / "b"), "dry_run": True}, root=tmp_path)
    assert first.config_hash == second.config_hash
    assert resolve_config({"madnlp_hot_max_wall_time": 20}).config_hash == resolve_config({"madnlp_hot_max_wall_time": 20.0}).config_hash
    assert first.config_hash != resolve_config({"signed_crank_torque": 2}, root=tmp_path).config_hash


def test_effective_seed_guard_and_window_match_launch(tmp_path):
    resolved = resolve_config({"solver": "acados", "integration": "irk", "acados_ipopt_cycle1_seed": "seed.npz"}, root=tmp_path)
    assert resolved.to_dict()["effective"]["transcription"]["reduced_internal_crank_velocity_guard"] == "on"
    assert resolve_config({"mode": "fho", "cycles": 8}).to_dict()["effective"]["transcription"]["cycles_per_window"] == 8


def test_cli_help_and_rejected_unknowns():
    assert cli("--help").returncode == 0
    assert cli("--unknown").returncode == 2
    assert cli("--set", "not_a_field=2").returncode == 2
    assert cli("--profile", "K7", "--set", "collocation_degree=3").returncode == 2
    with pytest.raises(ValueError, match="must be an object"):
        resolve_config([])


def test_form_launch_honors_seed_and_start_constraint(tmp_path):
    config = SimulationConfig(common_initial_solution="seed.npz", ipopt_enforce_start_constraints=False)
    plan = build_launch_plan(config, tmp_path, tmp_path)
    assert plan.environment_updates["BENCHMARK_COMMON_INITIAL_SOLUTION"] == str(tmp_path / "seed.npz")
    assert plan.environment_updates["BENCHMARK_ENFORCE_START_CONSTRAINTS"] == "false"
