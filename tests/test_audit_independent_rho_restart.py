import json
from pathlib import Path

import pytest

from scripts.audit_independent_rho_restart import audit_directory


def _result(root: Path, *, left_certified=True):
    root.mkdir()
    (root / "configuration.json").write_text('{"cycles": 100}\n', encoding="utf-8")
    metrics = {"certified": True, "weights_used": [1.0] * 4,
               "pace_vr_terminal_slow": [[1.0, 2.0, 3.0]],
               "pace_vr_snapshot": {"source_cycle": 20}}
    summary = {"architecture": "two-process-unilateral-isokinetic-rho-pace",
               "arms": {side: {"configured_model": None} for side in ("right", "left")},
               "cycles": [{"cycle": 20, "arms": {
                   "right": metrics,
                   "left": {**metrics, "certified": left_certified},
               }}]}
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_projection_snapshots_do_not_count_as_exact_restart(tmp_path):
    root = tmp_path / "result"
    _result(root)
    audit = audit_directory(root, 20)
    assert audit["pair_certified"]
    assert all(item["pace_vr_projection_snapshot_present"] for item in audit["arms"].values())
    assert audit["exact_restart"]["reason"] == "missing_exact_bilateral_restart"
    assert not audit["matched_continuation_ready"]
    assert "stimulation_history_per_arm" in audit["missing_restart_components"]


def test_uncertified_or_absent_pair_is_never_restart_ready(tmp_path):
    root = tmp_path / "result"
    _result(root, left_certified=False)
    assert not audit_directory(root, 20)["pair_certified"]
    with pytest.raises(ValueError, match="absent or repeated"):
        audit_directory(root, 21)


def test_unverified_receipt_is_rejected(tmp_path):
    root = tmp_path / "result"
    _result(root)
    receipt = root / "checkpoints" / "cycle-20" / "receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"exact_bilateral_restart": true}', encoding="utf-8")
    audit = audit_directory(root, 20)
    assert audit["exact_restart"]["reason"] == "invalid_exact_bilateral_restart"
    assert not audit["matched_continuation_ready"]
