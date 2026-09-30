import importlib.util
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from scripts.validate_pace_vr_decision_fidelity import _restart_gate


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_independent_rho_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("validate_independent_rho_checkpoint", MODULE_PATH)
checkpoint_validation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(checkpoint_validation)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _prepared_export(tmp_path: Path):
    root = tmp_path / "run"
    directory = root / "checkpoints" / "cycle-20"
    directory.mkdir(parents=True)
    configuration = root / "configuration.json"
    _write(configuration, {"cycles": 100, "model_config": "right-model.json"})
    arms = {}
    for side in ("right", "left"):
        model = root / f"{side}-model.json"
        _write(model, {"side": side})
        primal = directory / f"{side}.npz"
        np.savez(primal, states__A=np.asarray([[1.0, 0.9]]),
                 controls__last_pulse_width=np.asarray([[0.0002]]),
                 metadata__json=np.asarray(json.dumps({
                     "producer_mode": "rho_replay_checkpoint", "producer_completed_windows": 20,
                 })))
        arms[side] = {
            "status": "exported", "completed_cycles": 20,
            "primal_path": primal.name, "sha256": sha256(primal.read_bytes()).hexdigest(),
            "model_path": str(model), "model_sha256": sha256(model.read_bytes()).hexdigest(),
            "prepared_primal_signature": side + "-signature",
            "prepared_problem_sha256": "a" * 64,
            "stimulation_history_complete": True,
        }
    export = {
        "schema_version": 1, "checkpoint_kind": "prepared_bilateral_shifted_primal_export",
        "completed_cycles": 20, "configuration_path": "../../configuration.json",
        "configuration_sha256": sha256(configuration.read_bytes()).hexdigest(),
        "exact_bilateral_restart": False, "fresh_worker_replay_verified": False,
        "arms": arms,
    }
    export_path = directory / "prepared-export.json"
    _write(export_path, export)
    return root, export_path, export


def test_final_receipt_requires_fresh_worker_digest_equality_and_passes_gate(tmp_path):
    root, export_path, export = _prepared_export(tmp_path)
    reports = {side: {"restored": {
        "restored_problem_sha256": "a" * 64,
        "restored_primal_signature": side + "-signature",
        "stimulation_history_complete": True,
    }} for side in ("right", "left")}

    receipt = checkpoint_validation._final_receipt(export_path, export, reports)
    receipt_path = export_path.with_name("receipt.json")
    _write(receipt_path, receipt)

    gate = _restart_gate(root / "manifest.json", {"restart_receipt": str(receipt_path)}, 20)
    assert gate["ready"] is True


def test_final_receipt_refuses_mismatched_fresh_worker_digest(tmp_path):
    _, export_path, export = _prepared_export(tmp_path)
    reports = {side: {"restored": {
        "restored_problem_sha256": "b" * 64 if side == "right" else "a" * 64,
        "restored_primal_signature": side + "-signature",
        "stimulation_history_complete": True,
    }} for side in ("right", "left")}

    try:
        checkpoint_validation._final_receipt(export_path, export, reports)
    except RuntimeError as error:
        assert "right" in str(error)
    else:  # pragma: no cover - guards the assertion without extra dependency
        raise AssertionError("mismatched digest must be rejected")
