"""Pure launcher contracts; no solver imports or executions."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("pycharm_simulation", SCRIPTS / "run_pycharm_simulation.py")
driver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = driver
SPEC.loader.exec_module(driver)
sys.path.remove(str(SCRIPTS))


@pytest.mark.parametrize("solver", ["ipopt", "madnlp", "acados"])
def test_plan_is_pure_and_passes_scientific_options(tmp_path, solver):
    extras = ["--standard-warmup-seed", "a path/$(touch NEVER);x.npz"]
    config = {**driver.CONFIG, "solver": solver, "output_root": "outputs", "extra_arguments": extras,
              "stimulations_per_cycle": 40, "pulse_width_max_step_us": 25,
              "acados_ipopt_cycle1_seed": "ipopt cycle 1/seed.npz",
              "signed_crank_torque": 0.35}
    before = dict(os.environ)
    plan = driver.build_launch_plan(config, tmp_path / "env", tmp_path)
    assert not list(tmp_path.iterdir())
    assert dict(os.environ) == before
    assert plan.environment_updates["IPOPT_LINEAR_SOLVER"] == "ma57"
    if solver == "acados":
        assert "--experimental-reduced-acados" in plan.argv
        assert plan.argv[plan.argv.index("--signed-crank-torque") + 1] == "0.35"
        assert plan.argv[plan.argv.index("--stimulations-per-cycle") + 1] == "40"
        assert plan.argv[plan.argv.index("--common-initial-solution") + 1] == "ipopt cycle 1/seed.npz"
        assert "--adopt-common-initial-solution-warmup-cycles" in plan.argv
        assert "--acados-disable-standard-ipopt-warmup" in plan.argv
        assert list(plan.argv[-(len(extras) + 10):]) == extras + [
            "--reduced-internal-crank-velocity-guard", "on",
            "--acados-qp-solver", "auto",
            "--pulse-width-slew-weight", "0.0",
            "--pulse-width-slew-reference-us", "100.0",
            "--pulse-width-max-step-us", "25.0",
        ]
        assert plan.cwd == tmp_path / "outputs/acados-pycharm/codegen"
    else:
        assert len(plan.argv[2:]) == 14
        assert json.loads(plan.environment_updates["BENCHMARK_EXTRA_ARGUMENTS_JSON"]) == extras + [
            "--reduced-internal-crank-velocity-guard", "auto",
            "--acados-qp-solver", "auto",
            "--pulse-width-slew-weight", "0.0",
            "--pulse-width-slew-reference-us", "100.0",
            "--pulse-width-max-step-us", "25.0"
        ]
        assert plan.environment_updates["BENCHMARK_SIGNED_CRANK_TORQUE"] == "0.35"
        assert plan.environment_updates["BENCHMARK_STIMULATIONS_PER_CYCLE"] == "40"


def test_full_acados_does_not_enable_reduced_path(tmp_path):
    plan = driver.build_launch_plan({
        **driver.CONFIG,
        "solver": "acados",
        "mechanics": "full",
        "acados_ipopt_cycle1_seed": "cycle-1.npz",
    }, tmp_path)
    assert "--experimental-reduced-acados" not in plan.argv


@pytest.mark.parametrize("change", [{"cycles": 0}, {"threads": True}, {"signed_crank_torque": float("nan")},
                                  {"stimulations_per_cycle": 3.5}, {"extra_arguments": ["--solvers=madnlp"]},
                                  {"extra_arguments": ["--pulse-width-max-step-us", "10"]},
                                  {"extra_arguments": "--some-option"},
                                  {"pulse_width_max_step_us": 0},
                                  {"solver": "acados", "acados_ipopt_cycle1_seed": None},
                                  {"mechanics": "full", "pulse_width_max_step_us": 100}])
def test_invalid_configuration_is_rejected(tmp_path, change):
    with pytest.raises(ValueError):
        driver.build_launch_plan({**driver.CONFIG, **change}, tmp_path)


def test_dry_run_does_not_create_environment_or_start_process(monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "CONFIG", {**driver.CONFIG, "output_root": str(tmp_path / "outputs")})
    monkeypatch.setattr(driver, "conda_env_prefix", lambda suite: None)
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run must not create environment or execute")
    monkeypatch.setattr(driver, "base_environment", forbidden)
    monkeypatch.setattr(driver.subprocess, "run", forbidden)
    assert driver.main(["--dry-run"]) == 0
    assert not list(tmp_path.iterdir())


def test_shell_decoder_preserves_arguments_as_data(tmp_path):
    source = (SCRIPTS / "run_cycling_benchmark_case.sh").read_text()
    block = source[source.index('extra_arguments_json='):source.index('stimulations_per_cycle=')]
    extras = ["", "a b", "line\nbreak", "$(touch NEVER)", "'\";$PATH"]
    program = 'set -euo pipefail\npython_executable="$TEST_PYTHON"\n' + block + '\nprintf "%s\\0" "${extra_arguments[@]}"'
    env = {**os.environ, "TEST_PYTHON": sys.executable, "BENCHMARK_EXTRA_ARGUMENTS_JSON": json.dumps(extras)}
    result = subprocess.run(["bash", "-c", program], env=env, cwd=tmp_path, capture_output=True, check=True)
    assert result.stdout.split(b"\0")[:-1] == [x.encode() for x in extras]
    assert not list(tmp_path.iterdir())
    env["BENCHMARK_EXTRA_ARGUMENTS_JSON"] = '{"invalid": true}'
    assert subprocess.run(["bash", "-c", program], env=env, capture_output=True).returncode != 0
