"""Fail-closed provenance checks for fatigue-ablation mechanical comparisons."""
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from scripts import probe_fatigue_ablation_endpoint as probe


def test_endpoint_rejects_changed_source_before_loading(tmp_path, monkeypatch):
    receipt = tmp_path / "receipt.json"
    receipt.write_text("changed")
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"source_receipt": str(receipt), "source_receipt_sha256": "wrong"}))
    monkeypatch.setattr(probe, "_load_arm", lambda *_: pytest.fail("Changed receipt must not load"))
    with pytest.raises(ValueError, match="receipt changed"):
        probe.authenticated_endpoint(result, 1)


@pytest.mark.parametrize("rows", [
    [{"offset": 1, "certified": False, "maximum_normalized_violation": 0.}],
    [{"offset": 1, "certified": True, "maximum_normalized_violation": 2e-6}],
    [{"offset": 0, "certified": True, "maximum_normalized_violation": 0.}],
])
def test_endpoint_requires_ordered_independently_audited_prefix(tmp_path, monkeypatch, rows):
    receipt = tmp_path / "receipt.json"
    receipt.write_text("receipt")
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"source_receipt": str(receipt),
        "source_receipt_sha256": sha256(receipt.read_bytes()).hexdigest(), "side": "left",
        "cycles": rows, "audit_tolerance": 1e-6}))
    monkeypatch.setattr(probe, "_load_arm", lambda *_: (SimpleNamespace(completed_cycles=120), {}))
    with pytest.raises(ValueError, match="Uncertified"):
        probe.authenticated_endpoint(result, 1)
