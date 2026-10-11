"""Narrow paired fast-state gate: robust result is never called closure."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from cocofest.cycling.slow.fast_state_gate import (
    assess_fast_state_pairs,
    canonical_digest,
    make_fast_state_gate_receipt,
    verify_preregistration,
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _plan(tmp_path: Path) -> tuple[dict, Path]:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("certified")
    plan = {
        "schema": "slow-phi-fast-closure-preregistration-v1", "status": "preregistered_not_executed",
        "side": "left", "muscles": ["Biceps", "Triceps"],
        "sources": [{"receipt": str(receipt), "model_sha256": "a" * 64,
                     "artifacts": [{"path": str(receipt), "sha256": _digest(receipt)}]}],
        "actions": {"left_transfer_nm": [-.05, 0., .05]},
        "controls": {"independent_repeats_per_action": 3},
    }
    plan["plan_sha256"] = canonical_digest(plan, field="plan_sha256")
    return plan, receipt


def _audit(tmp_path: Path) -> dict:
    path = tmp_path / "audit.json"
    path.write_text("audited")
    return {"path": str(path), "sha256": _digest(path)}


def _outcome(audit: dict, cycles: int = 12, feasible: bool = True) -> dict:
    return {"endpoint": "frozen_local_endpoint", "audited": True, "accepted_cycles": cycles,
            "feasible": feasible, "weighted_force_integral": {"Biceps": 2., "Triceps": 3.}, "audits": [audit]}


def _evidence(plan: dict, receipt: Path, audit: dict) -> dict:
    receipt_sha = _digest(receipt)
    controls = []
    for action in (-.05, 0., .05):
        for repeat in range(3):
            controls.append({"source_receipt_sha256": receipt_sha, "delta_left_nm": action,
                             "replicate_id": f"r{repeat}", "endpoint": "frozen_local_endpoint",
                             "audited": True, "accepted_cycles": 12, "audits": [audit]})
    identity = {"model_hash": "a" * 64, "task_hash": "b" * 64, "slow_boundary_sha256": "c" * 64,
                "return_policy_sha256": "d" * 64, "stopping_rule_sha256": "e" * 64,
                "discretization_sha256": "f" * 64}
    pair = {"pair_id": "pair-1", "source_receipt_sha256": receipt_sha, "identity": identity,
            "first_fast_state_sha256": "1" * 64, "second_fast_state_sha256": "2" * 64,
            "coverage": {"Cn": "verified_invariant", "F": "paired_different",
                         "PW_history": "paired_different", "mechanics": "verified_invariant"},
            "actions": [{"delta_left_nm": action, "first": _outcome(audit), "second": _outcome(audit)}
                        for action in (-.05, 0., .05)]}
    return {"schema": "slow-phi-fast-pair-evidence-v1", "plan_sha256": plan["plan_sha256"],
            "controls": controls, "pairs": [pair]}


def test_no_pairs_is_explicitly_inconclusive(tmp_path):
    plan, _ = _plan(tmp_path)
    report = assess_fast_state_pairs(plan, {"schema": "slow-phi-fast-pair-evidence-v1", "plan_sha256": plan["plan_sha256"],
                                             "controls": [], "pairs": []})
    assert report["status"] == "inconclusive"
    assert not report["fast_state_gate"] and not report["closure_established"]
    assert any(issue.startswith("missing_matched_pair") for issue in report["issues"])


def test_complete_audited_pairs_issue_local_gate_but_never_closure(tmp_path):
    plan, receipt = _plan(tmp_path)
    report = assess_fast_state_pairs(plan, _evidence(plan, receipt, _audit(tmp_path)))
    assert report["status"] == "validated_local_robustness"
    assert report["fast_state_gate"] and not report["closure_established"]
    gate = make_fast_state_gate_receipt(report, {_digest(receipt): {"left": "a" * 64, "right": "b" * 64}})
    assert gate["status"] == "validated"
    assert gate["closure_established"] is False
    assert "not_markov_closure" in gate["validation_scope"]


def test_resolved_fast_effect_or_ranking_change_rejects_gate(tmp_path):
    plan, receipt = _plan(tmp_path)
    evidence = _evidence(plan, receipt, _audit(tmp_path))
    evidence["pairs"][0]["actions"][0]["second"] = _outcome(evidence["pairs"][0]["actions"][0]["first"]["audits"][0], cycles=14)
    report = assess_fast_state_pairs(plan, evidence)
    assert report["status"] == "rejected_local_robustness"
    assert any("fast_effect_exceeds_resolution" in item for item in report["negative_findings"])
    with pytest.raises(ValueError, match="Only a passed"):
        make_fast_state_gate_receipt(report, {})


def test_censoring_or_missing_coverage_is_inconclusive_not_equivalence(tmp_path):
    plan, receipt = _plan(tmp_path)
    evidence = _evidence(plan, receipt, _audit(tmp_path))
    evidence["pairs"][0]["coverage"].pop("F")
    report = assess_fast_state_pairs(plan, evidence)
    assert report["status"] == "inconclusive"
    assert not report["fast_state_gate"]


def test_tampered_preregistration_refused(tmp_path):
    plan, _ = _plan(tmp_path)
    plan["actions"]["left_transfer_nm"].append(.10)
    with pytest.raises(ValueError, match="changed"):
        verify_preregistration(plan)
