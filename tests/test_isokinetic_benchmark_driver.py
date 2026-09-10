"""Front-end contract tests for the isokinetic benchmark driver."""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / ".github" / "scripts" / "run_benchmarks.py"
CASE_RUNNER = ROOT / ".github" / "scripts" / "run_cycling_benchmark_case.sh"
SPEC = importlib.util.spec_from_file_location("isokinetic_benchmark_driver", DRIVER_PATH)
driver = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = driver
SPEC.loader.exec_module(driver)


def test_isokinetic_cli_defaults_follow_the_documented_crank_convention():
    args = driver.parse_arguments(["--formulation", "isokinetic"])

    assert args.energy_equivalent_torque == pytest.approx(0.2)
    assert args.isokinetic_omega == pytest.approx(-2 * math.pi)
    assert args.load_torque_min == pytest.approx(-3.0)
    assert args.load_torque_max == pytest.approx(3.0)
    assert args.acados_max_iter == 100
    assert args.acados_stationarity_tolerance == pytest.approx(5e-3)
    assert args.acados_control_homotopy_window_growth == pytest.approx(1.25)
    assert args.acados_control_homotopy_window_max_radius == pytest.approx(1e-5)
    assert not args.acados_release_control_homotopy


def test_default_worker_threads_respects_the_process_affinity():
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)

    assert 1 <= driver.default_worker_threads() <= available


@pytest.mark.parametrize(
    "arguments",
    (
        ["--energy-equivalent-torque", "-0.01"],
        ["--energy-equivalent-torque", "nan"],
        ["--load-torque-min", "-inf"],
        ["--load-torque-max", "inf"],
        ["--isokinetic-omega", "0"],
        ["--isokinetic-omega", "1e-3"],
        ["--isokinetic-omega", "nan"],
        ["--load-torque-min", "1", "--load-torque-max", "-1"],
        ["--load-torque-min", "0", "--load-torque-max", "1"],
        ["--load-torque-min", "-1", "--load-torque-max", "0"],
        ["--energy-equivalent-torque", "2", "--load-torque-max", "1"],
        ["--acados-max-iter", "0"],
        ["--acados-stationarity-tolerance", "0"],
        ["--acados-control-homotopy-window-growth", "0.99"],
        ["--acados-control-homotopy-window-max-radius", "0"],
    ),
)
def test_isokinetic_cli_rejects_nonphysical_numeric_values(arguments):
    with pytest.raises(SystemExit, match="2"):
        driver.parse_arguments(arguments)


def test_isokinetic_environment_and_all_backend_commands_share_the_same_options(tmp_path):
    (tmp_path / "libhsl.so").write_bytes(b"test HSL library")
    args = driver.parse_arguments(
        [
            "--formulation", "isokinetic",
            "--energy-equivalent-torque", "0.25",
            "--isokinetic-omega", "-5.5",
            "--load-torque-min", "-0.8",
            "--load-torque-max", "0.9",
            "--ipopt-hsl-library", str(tmp_path / "libhsl.so"),
            "--acados-max-iter", "200",
            "--acados-stationarity-tolerance", "1e-5",
            "--acados-control-homotopy-window-growth", "1.1",
            "--acados-control-homotopy-window-max-radius", "1e-5",
            "--output-root", str(tmp_path / "results"),
        ]
    )
    ipopt = next(case for case in driver.CASES if case.key == "ipopt")
    madnlp = next(case for case in driver.CASES if case.key == "madnlp-mumps")
    acados = next(case for case in driver.CASES if case.key == "acados-irk")

    environment = driver.build_case_environment(ipopt, tmp_path / "rho32", args)
    assert {
        key: environment[key]
        for key in environment
        if key.startswith("BENCHMARK_")
    }.items() >= {
        "BENCHMARK_FORMULATION": "isokinetic",
        "BENCHMARK_ENERGY_EQUIVALENT_TORQUE": "0.25",
        "BENCHMARK_ISOKINETIC_OMEGA": "-5.5",
        "BENCHMARK_LOAD_TORQUE_MIN": "-0.8",
        "BENCHMARK_LOAD_TORQUE_MAX": "0.9",
    }.items()
    assert environment["IPOPT_LINEAR_SOLVER"] == "ma57"
    assert environment["WARMUP_IPOPT_LINEAR_SOLVER"] == "ma57"
    assert environment["IPOPT_HSL_LIBRARY"] == str(tmp_path / "libhsl.so")
    assert environment["BENCHMARK_CONFIGURATION_SLUG"] == (
        "isokinetic-torque-0.25-omega--5.5-load--0.8-to-0.9"
    )
    madnlp_environment = driver.build_case_environment(
        madnlp, tmp_path / "madnlp32", args
    )
    assert madnlp_environment["DUAL_WARM_START"] == "bounds"
    assert madnlp_environment["BENCHMARK_NLP_TOLERANCE"] == "1e-8"

    shell_command, _ = driver.build_command(ipopt, tmp_path / "rho32", args)
    assert shell_command[:2] == ["bash", str(driver.CASE_RUNNER)]
    acados_command, _ = driver.build_command(acados, tmp_path / "rho32", args)
    expected = [
        "--formulation", "isokinetic",
        "--energy-equivalent-torque", "0.25",
        "--isokinetic-omega", "-5.5",
        "--load-torque-min", "-0.8",
        "--load-torque-max", "0.9",
    ]
    for flag, value in zip(expected[::2], expected[1::2]):
        index = acados_command.index(flag)
        assert acados_command[index + 1] == value
    assert acados_command[acados_command.index("--acados-nlp-solver-type") + 1] == (
        "SQP_WITH_FEASIBLE_QP"
    )
    assert acados_command[
        acados_command.index("--acados-search-direction-mode") + 1
    ] == "BYRD_OMOJOKUN"
    assert acados_command[acados_command.index("--acados-max-iter") + 1] == "200"
    assert acados_command[
        acados_command.index("--acados-stationarity-tolerance") + 1
    ] == "1e-05"
    assert acados_command[
        acados_command.index("--acados-control-homotopy-window-growth") + 1
    ] == "1.1"
    assert acados_command[
        acados_command.index("--acados-control-homotopy-window-max-radius") + 1
    ] == "1e-05"
    assert "--acados-control-homotopy-release-final-radius" not in acados_command
    assert "--periodic-ipopt-refinement-each-window" not in acados_command
    assert "--acados-transfer-irk-rollout" not in acados_command
    assert "--acados-ipopt-recovery" in acados_command
    trajectory_index = acados_command.index("--receding-horizon-solution-output")
    assert acados_command[trajectory_index + 1] == str(
        tmp_path
        / "results"
        / "acados-irk-reduced-isokinetic-torque-0.25-omega--5.5-load--0.8-to-0.9"
        / "validated-rho-trajectory.npz"
    )
    assert "--allow-partial-receding-horizon-solution-output" in acados_command
    assert acados_command[acados_command.index("--ipopt-linear-solver") + 1] == "ma57"
    assert acados_command[acados_command.index("--warmup-ipopt-linear-solver") + 1] == "ma57"
    assert acados_command[acados_command.index("--ipopt-hsl-library") + 1] == str(
        tmp_path / "libhsl.so"
    )
    assert acados_command[acados_command.index("--ipopt-profile") + 1] == "scientific-radau5"
    assert acados_command[
        acados_command.index("--periodic-ipopt-refinement-collocation-degree") + 1
    ] == "5"

    assert driver.case_result_dir_name(ipopt, args) == (
        "ipopt-reduced-isokinetic-torque-0.25-omega--5.5-load--0.8-to-0.9"
    )
    assert driver.case_result_dir_name(ipopt, driver.parse_arguments([])) == "ipopt-reduced"


def test_hsl_library_is_validated_before_launching_benchmarks(tmp_path, capsys):
    missing_library = tmp_path / "missing-libhsl.so"
    with pytest.raises(SystemExit, match="2"):
        driver.main([
            "--formulation", "isokinetic", "--cases", "acados-irk",
            "--ipopt-hsl-library", str(missing_library),
        ])
    error = capsys.readouterr().err
    assert str(missing_library) in error
    assert "--ipopt-hsl-library: file not found" in error
    assert "ACADOS case also uses IPOPT/MA57" in error
    assert "Traceback" not in error


def test_isokinetic_uses_hsl_installed_in_the_target_rho_environment(tmp_path):
    args = driver.parse_arguments(["--formulation", "isokinetic"])
    ipopt = next(case for case in driver.CASES if case.key == "ipopt")
    rho_prefix = tmp_path / "rho32"
    installed_hsl = rho_prefix / "lib" / "libhsl.so"
    installed_hsl.parent.mkdir(parents=True)
    installed_hsl.write_bytes(b"test HSL library")

    environment = driver.build_case_environment(ipopt, rho_prefix, args)

    assert environment["IPOPT_HSL_LIBRARY"] == str(installed_hsl)


def test_isokinetic_finds_versioned_hsl_installation_in_target_environment(tmp_path):
    args = driver.parse_arguments(["--formulation", "isokinetic"])
    ipopt = next(case for case in driver.CASES if case.key == "ipopt")
    rho_prefix = tmp_path / "rho32"
    installed_hsl = rho_prefix / "opt" / "libhsl" / "v2025.7.21" / "lib" / "libhsl.so"
    installed_hsl.parent.mkdir(parents=True)
    installed_hsl.write_bytes(b"test HSL library")

    environment = driver.build_case_environment(ipopt, rho_prefix, args)

    assert environment["IPOPT_HSL_LIBRARY"] == str(installed_hsl)


def test_base_environment_exposes_hsl_for_dynamic_rho_child_processes(tmp_path):
    rho_prefix = tmp_path / "cocofest-rho32"
    installed_hsl = rho_prefix / "opt" / "libhsl" / "v2025.7.21" / "lib" / "libhsl.so"
    installed_hsl.parent.mkdir(parents=True)
    installed_hsl.write_bytes(b"test HSL library")

    environment = driver.base_environment(rho_prefix, "rho32", threads=1)

    assert environment["IPOPT_HSL_LIBRARY"] == str(installed_hsl)


def test_hsl_library_rejects_directories(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        driver.parse_arguments(["--ipopt-hsl-library", str(tmp_path)])


def test_hsl_library_relative_path_survives_acados_codegen_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    library = tmp_path / "libhsl.so"
    library.write_bytes(b"test HSL library")
    args = driver.parse_arguments(["--ipopt-hsl-library", "libhsl.so"])
    acados = next(case for case in driver.CASES if case.key == "acados-irk")
    args.output_root = str(tmp_path / "results")
    command, cwd = driver.build_command(acados, tmp_path / "rho32", args)
    assert cwd != tmp_path
    assert command[command.index("--ipopt-hsl-library") + 1] == str(library)


@pytest.mark.parametrize("mode", ["--list", "--report-only"])
def test_read_only_modes_do_not_require_hsl_library(tmp_path, mode):
    args = driver.parse_arguments([
        mode, "--ipopt-hsl-library", str(tmp_path / "missing.so"),
    ])
    assert args.ipopt_hsl_library == tmp_path / "missing.so"


def test_shell_runner_reads_isokinetic_options_from_environment_and_suffixes_results(tmp_path):
    capture = tmp_path / "captured-command.txt"
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "python"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "from pathlib import Path\n"
        "import sys\n"
        "if sys.argv[1:2] == ['-c']:\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        "Path(os.environ['ISOKINETIC_CAPTURE']).write_text('\\n'.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    output_root = tmp_path / "results"
    environment = {
        **os.environ,
        "GITHUB_WORKSPACE": str(ROOT),
        "BENCHMARK_CYCLES": "1",
        "BENCHMARK_CYCLES_PER_WINDOW": "1",
        "BENCHMARK_THREADS": "1",
        "BENCHMARK_ASSISTANCE": "0.00",
        "BENCHMARK_Q_SLACK": "0.002",
        "BENCHMARK_MAX_ITER": "10",
        "BENCHMARK_FORMULATION": "isokinetic",
        "BENCHMARK_ENERGY_EQUIVALENT_TORQUE": "0.20000000000000001",
        "BENCHMARK_ISOKINETIC_OMEGA": "-6.283185307179586",
        "BENCHMARK_LOAD_TORQUE_MIN": "-3.0",
        "BENCHMARK_LOAD_TORQUE_MAX": "3.0",
        "ISOKINETIC_CAPTURE": str(capture),
        "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
    }

    completed = subprocess.run(
        [
            "bash", str(CASE_RUNNER), "ipopt", "ipopt", "reduced", "mumps", "collocation",
            str(output_root), "1", "false", "sx", "none", "3", "periodic_collocation",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    command = capture.read_text(encoding="utf-8").splitlines()
    for flag, value in (
        ("--formulation", "isokinetic"),
        ("--energy-equivalent-torque", "0.20000000000000001"),
        ("--isokinetic-omega", "-6.283185307179586"),
        ("--load-torque-min", "-3.0"),
        ("--load-torque-max", "3.0"),
    ):
        index = command.index(flag)
        assert command[index + 1] == value
    output_index = command.index("--output-json")
    assert command[output_index + 1].endswith(
        "ipopt-reduced-isokinetic-torque-0.2-omega--6.28318530718-load--3-to-3/result.json"
    )


def test_case_result_path_uses_the_same_canonical_float_formatting():
    args = driver.parse_arguments(
        [
            "--formulation", "isokinetic",
            "--energy-equivalent-torque", "0.20000000000000001",
            "--isokinetic-omega", "-6.283185307179586",
            "--load-torque-min", "-3.0",
            "--load-torque-max", "3.0",
        ]
    )
    ipopt = next(case for case in driver.CASES if case.key == "ipopt")

    assert driver.case_result_dir_name(ipopt, args) == (
        "ipopt-reduced-isokinetic-torque-0.2-omega--6.28318530718-load--3-to-3"
    )


def test_existing_result_path_falls_back_to_precanonical_campaign(tmp_path):
    args = driver.parse_arguments(["--formulation", "isokinetic"])
    ipopt_radau5 = next(case for case in driver.CASES if case.key == "ipopt-radau5")
    legacy = (
        tmp_path
        / "ipopt-radau5-reduced-isokinetic-torque-0.2"
        "-omega--6.283185307179586-load--3.0-to-3.0"
        / "result.json"
    )
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"results": [{"success": true}]}', encoding="utf-8")

    assert driver.existing_case_result_path(tmp_path, ipopt_radau5, args) == legacy


def test_shell_runner_rejects_a_nonnegative_isokinetic_speed(tmp_path):
    completed = subprocess.run(
        ["bash", str(CASE_RUNNER), "ipopt", "ipopt", "reduced", "mumps", "collocation"],
        cwd=ROOT,
        env={
            **os.environ,
            "GITHUB_WORKSPACE": str(ROOT),
            "BENCHMARK_CYCLES": "1",
            "BENCHMARK_ISOKINETIC_OMEGA": "0",
        },
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "strictly negative" in completed.stderr
