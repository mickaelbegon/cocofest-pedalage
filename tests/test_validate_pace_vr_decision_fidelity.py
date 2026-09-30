import importlib.util
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_pace_vr_decision_fidelity.py"
SPEC = importlib.util.spec_from_file_location("validate_pace_vr_decision_fidelity", MODULE_PATH)
validation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validation)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fatigue_failure(cycles: int) -> dict:
    arm = {"validated_cycles": cycles, "cycles": [{
        "cycle": cycles + 1, "certified": False,
        "counterfactual_probes": {"fatigue_rest": {"feasible_witness": True}},
    }]}
    return {"success": False, "failure": "next cycle uncertified",
            "arms": {"right": arm, "left": arm}}


def _feasible(horizon: int) -> dict:
    return {"success": True, "arms": {
        "right": {"validated_cycles": horizon, "cycles": []},
        "left": {"validated_cycles": horizon, "cycles": []},
    }}


def _fixture(tmp_path: Path, *, restart: bool = True) -> tuple[Path, dict]:
    names = ("unit", "physio", "bo_fixed", "intermediate")
    if restart:
        configuration = tmp_path / "restart" / "configuration.json"
        _write(configuration, {"torque_nm": 1.92, "cycles": 40})
        arms = {}
        for side in ("right", "left"):
            model = tmp_path / "restart" / f"{side}-model.json"
            _write(model, {"side": side})
            primal = tmp_path / "restart" / f"{side}.npz"
            np.savez(primal, states__A=np.asarray([[1.0, 0.9]]),
                     controls__PW=np.asarray([[0.0002]]),
                     metadata__json=np.asarray(json.dumps({
                         "producer_mode": "rho_replay_checkpoint",
                         "producer_completed_windows": 40,
                     })))
            arms[side] = {"primal_path": primal.name, "sha256": sha256(primal.read_bytes()).hexdigest(),
                          "model_path": model.name, "model_sha256": sha256(model.read_bytes()).hexdigest(),
                          "prepared_problem_sha256": "a" * 64, "restored_problem_sha256": "a" * 64,
                          "completed_cycles": 40, "certified": True,
                          "stimulation_history_complete": True, "replay_roundtrip_exact": True}
        _write(tmp_path / "restart" / "receipt.json", {
            "schema_version": 1, "checkpoint_kind": "certified_bilateral_shifted_primal",
            "exact_bilateral_restart": True, "completed_cycles": 40,
            "configuration_path": configuration.name,
            "configuration_sha256": sha256(configuration.read_bytes()).hexdigest(),
            "arms": arms,
        })
    checkpoint = {"completed_cycles": 40, "horizon_cycles": 30,
                  "restart_receipt": "restart/receipt.json", "selected_candidate_id": "unit",
                  "candidates": [{"id": name, "prediction_file": f"{name}/prediction.json",
                                  "continuation_summary": f"{name}/summary.json"} for name in names]}
    manifest = tmp_path / "manifest.json"
    _write(manifest, {"schema_version": 1, "checkpoints": [checkpoint]})
    return manifest, checkpoint


def _prediction(margin: float) -> dict:
    return {"status": "accepted", "accepted": True, "feasible_prefix_cycles": 30,
            "minimum_task_margin": margin, "terminal_value": margin,
            "work_residual_max": 1e-9, "constraint_violation_max": 1e-9,
            "runtime_s": 0.01, "local_fit": {"rank": 3}}


def test_exact_paired_checkpoint_ranks_surrogate_and_measures_regret(tmp_path):
    manifest, _ = _fixture(tmp_path)
    for name, margin, actual in (
        ("unit", 0.2, 10), ("physio", 0.1, 8),
        ("bo_fixed", 0.4, 20), ("intermediate", 0.3, 15),
    ):
        _write(tmp_path / name / "prediction.json", _prediction(margin))
        _write(tmp_path / name / "summary.json", _fatigue_failure(actual))
    output = validation.summarize(manifest)
    report = json.loads(output.read_text())
    comparison = report["comparisons"][0]
    assert comparison["exact_restart"]["ready"] is True
    assert comparison["predicted_best_candidate_id"] == "bo_fixed"
    assert comparison["pairwise_comparable"] == 6
    assert comparison["pairwise_concordance"] == 1.0
    assert comparison["spearman_exact_endurance"] == 1.0
    assert comparison["selected_regret_exact_cycles"] == 10
    assert comparison["physiological_failure_count"] == 4
    assert comparison["candidate_coverage"]["missing"] == []
    # Running the reader again is safe and updates the same summary atomically.
    assert validation.summarize(manifest) == output


def test_surrogate_failure_is_distinct_from_horizon_censoring_and_technical_stop(tmp_path):
    manifest, checkpoint = _fixture(tmp_path)
    _write(tmp_path / "unit" / "prediction.json", _prediction(0.2))
    _write(tmp_path / "unit" / "summary.json", _fatigue_failure(10))
    _write(tmp_path / "physio" / "prediction.json", {"accepted": False,
        "status": "qp_failed", "reason": "surrogate active set"})
    _write(tmp_path / "physio" / "summary.json", _feasible(30))
    _write(tmp_path / "bo_fixed" / "prediction.json", _prediction(0.4))
    _write(tmp_path / "bo_fixed" / "summary.json", _feasible(30))
    _write(tmp_path / "intermediate" / "prediction.json", _prediction(0.3))
    _write(tmp_path / "intermediate" / "summary.json", {"success": False,
        "failure": "IPOPT restoration failed", "arms": {
            "right": {"validated_cycles": 12, "cycles": [{"cycle": 13, "certified": False}]},
            "left": {"validated_cycles": 12, "cycles": [{"cycle": 13, "certified": False}]},
        }})
    report = json.loads(validation.summarize(manifest).read_text())
    entries = {item["candidate_id"]: item for item in report["entries"]}
    assert entries["physio"]["prediction"]["state"] == "surrogate_failure"
    assert entries["physio"]["continuation"]["state"] == "horizon_censored"
    assert entries["intermediate"]["continuation"]["state"] == "technical_failure"
    comparison = report["comparisons"][0]
    assert comparison["surrogate_failure_count"] == 1
    assert comparison["physiological_failure_count"] == 1
    assert comparison["selected_regret_lower_bound_cycles"] == 20
    assert comparison["selected_regret_exact_cycles"] is None
    assert comparison["spearman_exact_endurance"] is None


def test_missing_exact_bilateral_restart_blocks_causal_ranking(tmp_path):
    manifest, _ = _fixture(tmp_path, restart=False)
    for name in ("unit", "physio", "bo_fixed", "intermediate"):
        _write(tmp_path / name / "prediction.json", _prediction(0.2))
        _write(tmp_path / name / "summary.json", _fatigue_failure(10))
    report = json.loads(validation.summarize(manifest).read_text())
    comparison = report["comparisons"][0]
    assert comparison["exact_restart"] == {"ready": False, "reason": "missing_exact_bilateral_restart"}
    assert comparison["pairwise_comparable"] == 0
    assert comparison["spearman_exact_endurance"] is None
    assert all(entry["continuation"]["state"] == "unpaired" for entry in report["entries"])


def test_rejects_score_outside_declared_work_residual_gate(tmp_path):
    manifest, checkpoint = _fixture(tmp_path)
    checkpoint["maximum_work_residual"] = 1e-10
    _write(manifest, {"schema_version": 1, "checkpoints": [checkpoint]})
    _write(tmp_path / "unit" / "prediction.json", _prediction(0.2))
    report = json.loads(validation.summarize(manifest).read_text())
    unit = next(entry for entry in report["entries"] if entry["candidate_id"] == "unit")
    assert unit["prediction"]["state"] == "surrogate_rejected_by_residual_gate"


def test_rejects_tampered_primal_even_when_receipt_claims_exact_restart(tmp_path):
    manifest, _ = _fixture(tmp_path)
    (tmp_path / "restart" / "right.npz").write_bytes(b"tampered")
    report = json.loads(validation.summarize(manifest).read_text())
    gate = report["comparisons"][0]["exact_restart"]
    assert gate["ready"] is False
    assert gate["reason"] == "invalid_exact_bilateral_restart"
    assert "SHA-256 mismatch" in gate["detail"]


def test_rejects_roundtrip_mismatch_despite_valid_archive_digest(tmp_path):
    manifest, _ = _fixture(tmp_path)
    receipt_path = tmp_path / "restart" / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["arms"]["left"]["restored_problem_sha256"] = "b" * 64
    _write(receipt_path, receipt)
    report = json.loads(validation.summarize(manifest).read_text())
    gate = report["comparisons"][0]["exact_restart"]
    assert gate["ready"] is False
    assert "left receipt lacks certified exact restart audit" in gate["detail"]
