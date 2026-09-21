from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "summarize_frequency_slew_matrix.py"
SPEC = importlib.util.spec_from_file_location("frequency_slew_matrix", SCRIPT)
matrix = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = matrix
SPEC.loader.exec_module(matrix)


def _write_result(path: Path, *, solver="ipopt", frequency=30, slew_us=None):
    configuration = {
        "mechanical_formulation": "reduced",
        "formulation": "dynamic",
        "constant_crank_torque": 0.1,
        "torque_application": "constant",
        "stimulations_per_cycle": frequency,
        "control_decisions_per_cycle": frequency,
        "calcium_stimulation_interval_s": 1.0 / frequency,
        "cycles_per_window": 1,
        "n_windows": 3,
        "objective": "fatigue",
        "objective_shape": "quadratic",
        "model_formulation": "periodic_node",
        "muscle_names": list(matrix.MUSCLES),
        "n_threads": 4,
        "nlp_tolerance": 1e-6,
        "collocation_method": "radau",
        "collocation_degree": 5 if solver == "ipopt" else 3,
        "transcription_profile": "scientific-radau5" if solver == "ipopt" else "acados-irk-4stage-5step",
        "ipopt_linear_solver": "ma57",
        "acados_integrator_type": "IRK",
        "profile_hash": "profile",
    }
    if slew_us is not None:
        configuration["pulse_width_max_step_us"] = slew_us
    windows = [
        {
            "solver_converged": True,
            "certifier": "target_solver" if index != 2 else "ipopt_recovery",
            "solver_time_s": value,
            "wall_time_s": value + 0.1,
            "iterations": index + 1,
            "native_status": "OK",
            "feasibility": {
                "effective_primal_infeasibility": (index + 1) * 1e-7,
                "acados_stationarity_residual": (index + 1) * 1e-5,
            },
        }
        for index, value in enumerate((9.0, 1.0, 100.0))
    ]
    payload = {
        "configurations": {solver: configuration},
        "runtime": {"python": "3.11", "provenance": {"COCOFEST_BENCHMARK_COMMIT": "abc"}},
        "results": [{
            "solver": solver, "success": True, "solver_success": True,
            "physical_success": True, "requested_cycles": 3, "validated_cycles": 3,
            "executed_fatigue_objective": 12.0, "fatigue_auc_cycles": 0.5,
            "min_A_capacity_ratio": 0.9, "windows": windows,
            "acados_ipopt_recovery": {"attempt_count": 2, "fallback_advanced_count": 1},
            "pulse_width_slew_audit": {
                "enabled": slew_us is not None,
                "limit_us": slew_us,
                "muscles": {
                    "Triceps": {"maximum_adjacent_change_us": 99.5}
                },
            },
            "muscle_fatigue": [
                {
                    "muscle": name,
                    "cumulative_normalized_fatigue_cycles": 0.1,
                    "final_capacity_ratio": 0.95,
                }
                for name in matrix.MUSCLES
            ],
        }],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_classifies_condition_and_uses_only_target_hot_windows(tmp_path):
    path = tmp_path / "result.json"
    _write_result(path, solver="acados", frequency=50, slew_us=100)
    row = matrix.load_rows([path], 1e-5)[0]

    assert row["condition"] == "50hz-dpw100us"
    assert row["solver_cell"] == "acados-irk"
    assert row["pulse_width_max_step_source"] == "pulse_width_max_step_us"
    assert row["observed_max_pulse_width_step_us"] == 99.5
    assert row["pulse_width_max_step_violation_us"] == 0.0
    assert row["n_threads"] == 4
    assert row["common_validated_prefix"] == 3
    assert row["hot_target_sample_count"] == 1
    assert row["hot_solver_mean_s"] == 1.0
    assert row["hot_solver_p90_s"] == 1.0
    assert row["recovery_attempts"] == 2
    assert row["fallback_advanced_count"] == 1
    assert row["fatigue_auc_Triceps"] == 0.1


def test_reads_seconds_slew_alias_and_rejects_conflicts(tmp_path):
    path = tmp_path / "result.json"
    _write_result(path, frequency=30)
    payload = json.loads(path.read_text())
    configuration = payload["configurations"]["ipopt"]
    configuration["pulse_width_slew_limit_s"] = 100e-6
    path.write_text(json.dumps(payload))
    assert matrix.load_rows([path], 1e-5)[0]["condition"] == "30hz-dpw100us"

    configuration["pulse_width_max_step_us"] = 50
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Conflicting"):
        matrix.load_rows([path], 1e-5)


def test_incomplete_matrix_is_explicit_and_cli_can_require_completion(tmp_path):
    result = tmp_path / "result.json"
    output = tmp_path / "summary"
    _write_result(result)

    assert matrix.main([str(result), "--output-dir", str(output)]) == 0
    report = json.loads((output / "frequency-slew-matrix.json").read_text())
    assert report["audit"]["complete"] is False
    assert len(report["audit"]["missing_cells"]) == 7
    markdown = (output / "frequency-slew-matrix.md").read_text()
    assert "Matrice incomplète" in markdown
    assert "50hz-dpw100us/acados-irk" in markdown
    assert matrix.main([str(result), "--output-dir", str(output), "--require-complete"]) == 2


def test_protocol_hash_excludes_frequency_and_max_step(tmp_path):
    baseline = tmp_path / "baseline.json"
    treatment = tmp_path / "treatment.json"
    _write_result(baseline, frequency=30)
    _write_result(treatment, frequency=50, slew_us=100)

    rows = matrix.load_rows([baseline, treatment], 1e-5)
    assert rows[0]["protocol_sha256"] == rows[1]["protocol_sha256"]


def test_percentile_matches_linear_numpy_convention():
    assert matrix._percentile([1.0, 2.0, 3.0, 4.0], 90) == pytest.approx(3.7)
