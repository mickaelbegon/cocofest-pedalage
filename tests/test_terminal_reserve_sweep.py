from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "run_terminal_reserve_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_terminal_reserve_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = sweep
SPEC.loader.exec_module(sweep)


def _write_valid_rho_npz(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata__json.npy", b"metadata")
        archive.writestr("states__A_Biceps.npy", b"states")
        archive.writestr("controls__last_pulse_width_Biceps.npy", b"controls")


def _complete_payload(case: sweep.SweepCase, n_windows: int = 20) -> dict:
    return {
        "configurations": {
            "ipopt": {
                "solver": "ipopt",
                "objective": "fatigue",
                "n_windows": n_windows,
                "mechanical_formulation": "reduced",
                "benchmark_profile": "scientific-radau5",
                "profile_integrity": True,
                "state_scaling": "full",
                "n_threads": 1,
                "constant_crank_torque": 0.2,
                "max_ipopt_iterations": 5000,
                "ipopt_linear_solver": "ma57",
                "ipopt_dual_warm_start_mode": "off",
                "ipopt_c_compile": True,
                "ipopt_ma57_automatic_scaling": True,
                "ipopt_linear_system_scaling": "none",
                "ipopt_ma57_pivot_order": 2,
                "terminal_reserve_weight": case.weight,
                "terminal_reserve_temperature": case.temperature,
            }
        },
        "results": [
            {
                "success": True,
                "solver_success": True,
                "physical_success": True,
                "requested_cycles": n_windows,
                "covered_cycles": n_windows,
                "validated_cycles": n_windows,
                "physically_validated_cycles": n_windows,
                "fatigue_auc_cycles": 0.1,
                "terminal_capacity_reserve": {
                    "available": True,
                    "minimum_ratio": 0.9,
                    "physiological_domain_valid": True,
                },
                "control_saturation": [{"upper_fraction": 0.2}],
                "compiled_nlp_reuse": {
                    "compiled_library_build_count": 1,
                    "compiled_library_reused": True,
                    "graph_rebuild_detected": False,
                },
                "ipopt_runtime": {
                    "kkt_diagnostics": {
                        "available": True,
                        "windows": [
                            {
                                "window": index,
                                "primal_infeasibility": 1e-9,
                                "dual_infeasibility": 2e-9,
                            }
                            for index in range(n_windows)
                        ],
                    }
                },
            }
        ],
    }


def test_grid_has_one_temperature_independent_baseline():
    cases = sweep.build_cases((0.0, 0.01, 0.1), (0.0025, 0.005, 0.01))

    assert cases[0] == sweep.SweepCase(0.0, 0.005)
    assert sum(case.weight == 0.0 for case in cases) == 1
    assert len(cases) == 7


@pytest.mark.parametrize(
    ("raw", "non_negative"),
    [("", True), ("nan", True), ("-0.1", True), ("0", False)],
)
def test_grid_parser_rejects_invalid_values(raw, non_negative):
    with pytest.raises(Exception):
        sweep.parse_float_grid(raw, non_negative=non_negative, option="--grid")


def test_case_command_freezes_the_paired_ma57_protocol(tmp_path):
    args = SimpleNamespace(
        python=Path("/env/bin/python"),
        output_root=tmp_path,
        n_windows=20,
        signed_crank_torque=0.2,
        ipopt_max_iter=5000,
        hsl_library=Path("/opt/libhsl.so"),
        seed=Path("/data/common.npz"),
    )
    command = sweep.build_case_command(args, sweep.SweepCase(0.03, 0.005))

    assert command[command.index("--ipopt-linear-solver") + 1] == "ma57"
    assert command[command.index("--warmup-ipopt-linear-solver") + 1] == "ma57"
    assert command[command.index("--ipopt-dual-warm-start-mode") + 1] == "off"
    assert command[command.index("--common-initial-solution") + 1] == "/data/common.npz"
    assert command[command.index("--terminal-reserve-weight") + 1] == "0.03"
    assert command[command.index("--terminal-reserve-temperature") + 1] == "0.005"
    assert command[command.index("--receding-horizon-solution-output") + 1].endswith(
        "lambda-0p03_tau-0p005/rho_solution.npz"
    )
    assert "--ipopt-c-compile" in command


def test_resume_requires_a_complete_matching_result(tmp_path):
    path = tmp_path / "result.json"
    case = sweep.SweepCase(0.1, 0.005)
    payload = _complete_payload(case)
    path.write_text(json.dumps(payload))
    _write_valid_rho_npz(path.with_name("rho_solution.npz"))

    assert sweep.result_is_complete(path, case, 20)
    payload["results"][0]["compiled_nlp_reuse"]["compiled_library_build_count"] = 2
    path.write_text(json.dumps(payload))
    assert not sweep.result_is_complete(path, case, 20)
    payload["results"][0]["compiled_nlp_reuse"]["compiled_library_build_count"] = 1
    payload["configurations"]["ipopt"]["ipopt_linear_solver"] = "mumps"
    path.write_text(json.dumps(payload))
    assert not sweep.result_is_complete(path, case, 20)


def test_resume_requires_matching_seed_hsl_options_and_kkt(tmp_path):
    path = tmp_path / "result.json"
    case = sweep.SweepCase(0.1, 0.005)
    payload = _complete_payload(case)
    configuration = payload["configurations"]["ipopt"]
    expected = {
        "seed_sha256": "seed-sha",
        "hsl_sha256": "hsl-sha",
        "configuration": dict(configuration),
        "submitted_options": {
            "linear_solver": "ma57",
            "max_iter": 5000,
        },
    }
    result = payload["results"][0]
    result["input_provenance"] = {
        "common_initial_solution": {"sha256": "seed-sha"}
    }
    result["ipopt_runtime"].update(
        {
            "hsl": {"loadable": True, "file": {"sha256": "hsl-sha"}},
            "submitted_options": {"linear_solver": "ma57", "max_iter": 5000},
        }
    )
    path.write_text(json.dumps(payload))
    _write_valid_rho_npz(path.with_name("rho_solution.npz"))

    assert sweep.result_is_complete(path, case, 20, expected=expected)
    result["input_provenance"]["common_initial_solution"]["sha256"] = "other"
    path.write_text(json.dumps(payload))
    assert not sweep.result_is_complete(path, case, 20, expected=expected)
    result["input_provenance"]["common_initial_solution"]["sha256"] = "seed-sha"
    result["ipopt_runtime"]["kkt_diagnostics"]["available"] = False
    path.write_text(json.dumps(payload))
    assert not sweep.result_is_complete(path, case, 20, expected=expected)


def test_invalid_or_empty_rho_npz_is_not_complete(tmp_path):
    path = tmp_path / "result.json"
    case = sweep.SweepCase(0.1, 0.005)
    path.write_text(json.dumps(_complete_payload(case)))
    path.with_name("rho_solution.npz").write_bytes(b"not an npz")

    assert not sweep.result_is_complete(path, case, 20)


def test_manifest_is_immutable_and_rejects_changed_resume_contract(tmp_path):
    path = tmp_path / "manifest.json"
    contract = {"seed": {"sha256": "abc"}, "configuration": {"n_windows": 20}}

    created = sweep.initialize_immutable_manifest(path, contract, resume=False)
    resumed = sweep.initialize_immutable_manifest(path, contract, resume=True)

    assert resumed == created
    with pytest.raises(SystemExit, match="differs"):
        sweep.initialize_immutable_manifest(
            path,
            {"seed": {"sha256": "changed"}, "configuration": {"n_windows": 20}},
            resume=True,
        )
    with pytest.raises(SystemExit, match="already exists"):
        sweep.initialize_immutable_manifest(path, contract, resume=False)


def test_case_artifacts_are_quarantined_before_a_fresh_attempt(tmp_path):
    case_directory = tmp_path / "case"
    case_directory.mkdir()
    for name in ("result.json", "rho_solution.npz", "solver.log"):
        (case_directory / name).write_text(f"old {name}")

    quarantine = sweep.quarantine_case_artifacts(case_directory)

    assert quarantine is not None
    assert not (case_directory / "result.json").exists()
    assert (quarantine / "result.json").read_text() == "old result.json"


def test_failed_subprocess_cannot_reuse_stale_success_artifacts(
    tmp_path, monkeypatch
):
    seed = tmp_path / "seed.npz"
    hsl = tmp_path / "libhsl.so"
    seed.write_bytes(b"seed")
    hsl.write_bytes(b"hsl")
    output = tmp_path / "screen"
    case_directory = output / "lambda-0_tau-0p005"
    case_directory.mkdir(parents=True)
    (case_directory / "result.json").write_text("old result")
    (case_directory / "rho_solution.npz").write_text("old solution")
    monkeypatch.setattr(
        sweep,
        "hsl_preflight",
        lambda path: {
            "file": sweep.source_stamp(path),
            "loadable": True,
            "ma57_symbol": "ma57id_",
            "error": None,
            "abi_scope": "test",
        },
    )
    monkeypatch.setattr(
        sweep.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )

    return_code = sweep.main(
        [
            "--seed",
            str(seed),
            "--hsl-library",
            str(hsl),
            "--output-root",
            str(output),
            "--weights",
            "0,0.01",
            "--temperatures",
            "0.005",
        ]
    )

    assert return_code == 1
    progress = json.loads((output / "progress.json").read_text())
    assert progress["cases"]["lambda-0_tau-0p005"]["status"] == "failed"
    assert not (case_directory / "result.json").exists()
    assert list((case_directory / "previous-attempts").glob("*/result.json"))


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--n-windows", "1"], "at least 2"),
        (["--timeout", "0"], "strictly positive"),
        (["--signed-crank-torque", "nan"], "must be finite"),
        (["--weights", "0.01"], "zero-weight baseline"),
    ],
)
def test_main_rejects_invalid_campaign_inputs(tmp_path, extra_args, message):
    seed = tmp_path / "seed.npz"
    hsl = tmp_path / "libhsl.so"
    seed.write_bytes(b"seed")
    hsl.write_bytes(b"hsl")
    with pytest.raises(SystemExit, match=message):
        sweep.main(
            [
                "--seed",
                str(seed),
                "--hsl-library",
                str(hsl),
                "--output-root",
                str(tmp_path / "screen"),
                *extra_args,
            ]
        )


def test_hsl_preflight_requires_loadability_and_ma57_symbol(tmp_path, monkeypatch):
    hsl = tmp_path / "libhsl.so"
    hsl.write_bytes(b"hsl")
    library = SimpleNamespace(ma57id_=object())
    monkeypatch.setattr(sweep.ctypes, "CDLL", lambda path: library)

    result = sweep.hsl_preflight(hsl)

    assert result["loadable"] is True
    assert result["ma57_symbol"] == "ma57id_"


def test_metric_specific_tolerances_ignore_numerical_noise():
    baseline = {
        "weight": 0.0,
        "complete": True,
        "minimum_capacity_ratio": 0.9,
        "fatigue_auc_cycles": 0.1,
        "maximum_pw_upper_fraction": 0.2,
    }
    noise = {
        "weight": 0.01,
        "complete": True,
        "minimum_capacity_ratio": 0.9 + 1e-7,
        "fatigue_auc_cycles": 0.1 - 1e-7,
        "maximum_pw_upper_fraction": 0.2,
    }

    classified = sweep.classify_screen([baseline, noise])

    assert classified[1]["screen_status"] == "dominated_or_equivalent"


def test_screen_rejects_a_case_that_worsens_any_predeclared_outcome():
    baseline = {
        "weight": 0.0,
        "complete": True,
        "minimum_capacity_ratio": 0.90,
        "fatigue_auc_cycles": 0.10,
        "maximum_pw_upper_fraction": 0.20,
    }
    tradeoff = {
        "weight": 0.03,
        "complete": True,
        "minimum_capacity_ratio": 0.91,
        "fatigue_auc_cycles": 0.11,
        "maximum_pw_upper_fraction": 0.19,
    }

    rows = sweep.classify_screen([baseline, tradeoff])

    assert rows[0]["screen_status"] == "reference"
    assert rows[1]["screen_status"] == "dominated_or_equivalent"


def test_screen_accepts_only_a_non_dominated_improvement():
    baseline = {
        "weight": 0.0,
        "complete": True,
        "minimum_capacity_ratio": 0.90,
        "fatigue_auc_cycles": 0.10,
        "maximum_pw_upper_fraction": 0.20,
    }
    candidate = {
        "weight": 0.01,
        "complete": True,
        "minimum_capacity_ratio": 0.91,
        "fatigue_auc_cycles": 0.09,
        "maximum_pw_upper_fraction": 0.20,
    }

    assert (
        sweep.classify_screen([baseline, candidate])[1]["screen_status"]
        == "pareto_nondominated"
    )
