"""Read-only preflight of the calibrated c120/c140 archived checkpoints."""

from __future__ import annotations

import json

import pytest

from cocofest.cycling.slow.causal_transfer import BicepsReservePolicy
from scripts.slow_layer.run_causal_transfer_supervisor import (
    DEFAULT_RECEIPTS,
    _verified_anchor,
    prepare,
    preflight,
)


def _schema_valid_gate(tmp_path):
    """Make a test receipt with the exact model/checkpoint bindings.

    This fixture validates the runner's provenance wiring, not fast-state
    Markov closure.  A production receipt needs the independent validation
    evidence described in the protocol before declaring this schema status.
    """
    anchors = []
    for receipt in DEFAULT_RECEIPTS:
        anchor, _ = _verified_anchor(receipt, BicepsReservePolicy())
        anchors.append({"receipt_sha256": anchor["receipt_sha256"],
                        "model_sha256": {side: anchor["arms"][side]["model_sha256"]
                                         for side in ("left", "right")},
                        "fast_state_gate": True})
    path = tmp_path / "validated-fast-gate.json"
    path.write_text(json.dumps({"schema": "causal-transfer-fast-state-gate-v1",
                                "status": "validated", "anchors": anchors}), encoding="utf-8")
    return path


def test_real_c120_and_c140_preflight_abstains_without_a_fast_state_gate():
    records = preflight(list(DEFAULT_RECEIPTS))
    decisions = {row["anchor"]["cycle"]: row["policy_receipt"]["decision"] for row in records}
    assert decisions[120]["phase"] == "hold"
    assert decisions[120]["reason"] == "fast_state_gate_not_validated"
    assert decisions[140]["phase"] == "hold"
    assert decisions[140]["reason"] == "fast_state_gate_not_validated"


def test_real_c120_holds_and_c140_triggers_with_a_schema_valid_model_bound_fast_gate(tmp_path):
    records = preflight(list(DEFAULT_RECEIPTS), fast_gate_receipt=_schema_valid_gate(tmp_path))
    decisions = {row["anchor"]["cycle"]: row["policy_receipt"]["decision"] for row in records}
    assert decisions[120]["phase"] == "hold"
    assert decisions[120]["reason"] == "left_biceps_reserve_above_activation_band"
    assert decisions[140]["phase"] == "transfer"
    assert decisions[140]["reason"] == "left_biceps_reserve_triggered"
    assert (decisions[140]["left_torque_nm"], decisions[140]["right_torque_nm"]) == (.86, 1.06)


def test_real_c120_holds_and_c140_triggers_under_the_explicit_unvalidated_experimental_override():
    records = preflight(list(DEFAULT_RECEIPTS), experimental_unvalidated_override=True)
    receipts = {row["anchor"]["cycle"]: row["policy_receipt"] for row in records}
    assert receipts[120]["fast_state_gate"]["fast_state_status"] == "unvalidated_experimental_override"
    assert "EXPERIMENTAL ONLY" in receipts[120]["fast_state_gate"]["warning"]
    assert receipts[120]["decision"]["phase"] == "hold"
    assert receipts[120]["decision"]["reason"] == "left_biceps_reserve_above_activation_band"
    assert receipts[140]["decision"]["phase"] == "transfer"
    assert receipts[140]["decision"]["reason"] == "left_biceps_reserve_triggered"


def test_prepare_seals_the_unvalidated_override_as_non_activation_eligible(tmp_path):
    manifest = prepare(tmp_path / "offline", list(DEFAULT_RECEIPTS),
                       experimental_unvalidated_override=True, maximum_cycles=240, replicates=2)
    plan = json.loads(manifest.read_text(encoding="utf-8"))
    assert plan["fast_state_status"] == "unvalidated_experimental_override"
    assert plan["activation_allowed"] is False
    assert "EXPERIMENTAL ONLY" in plan["warning"]
    assert sorted({job["delta_left_nm"] for job in plan["jobs"] if not job["is_control"]}) == pytest.approx([-.1, 0.])
