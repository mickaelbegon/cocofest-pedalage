"""Censoring and provenance regression tests; these are synthetic diagnostics."""

from dataclasses import asdict
import hashlib
import json

import pytest

from cocofest.cycling.slow import nhat_validation as validation
from cocofest.cycling.slow.remaining_cycles import ConditionalRemainingCycles
from cocofest.cycling.slow.slow_model import OracleProvenance, SlowState
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters


def label(**changes):
    return {"schema_version": 1, "certified": True, "anchor_cycle": 20, "observed_cycles": 10,
            "protocol_id": "synthetic-only", "audit_reference": "synthetic fixture",
            "censoring": "none", "terminal_classification": "frozen_feasibility_exhaustion",
            "terminal_certified": True, **changes}


@pytest.mark.parametrize("predicted_censored,observed_censored,interval", [
    (False, False, [2, 2]), (False, True, [None, 2]),
    (True, False, [2, None]), (True, True, [None, None]),
])
def test_censoring_never_becomes_an_exact_error(predicted_censored, observed_censored, interval):
    prediction = ConditionalRemainingCycles(12, None if predicted_censored else 13, 30, (), "0"*64)
    observation = label(censoring="right", terminal_classification="administrative_horizon") if observed_censored else label()
    result = validation.compare_label(prediction, observation, anchor_cycle=20)
    assert result["error_interval_cycles"] == interval
    assert result["signed_error_cycles"] == (2 if not (predicted_censored or observed_censored) else None)


@pytest.mark.parametrize("changes", [
    {"certified": False}, {"certified": 1}, {"anchor_cycle": 19}, {"observed_cycles": True},
    {"observed_cycles": -1}, {"protocol_id": ""}, {"audit_reference": None},
    {"terminal_classification": "solver_failed"}, {"terminal_certified": False},
    {"censoring": "left"}, {"censoring": "right", "terminal_classification": "unknown"},
])
def test_rejects_unsupported_or_unaudited_labels(changes):
    prediction = ConditionalRemainingCycles(12, 13, 30, (), "0"*64)
    with pytest.raises(ValueError):
        validation.compare_label(prediction, label(**changes), anchor_cycle=20)


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    # Oracle loading is separately covered by test_slow_oracle; this fixture
    # isolates the manifest/force/label logic and is not a scientific dataset.
    state = SlowState(("Biceps",), (4000.,), (.06,), (.137,),
                      OracleProvenance("synthetic.npz", "0"*64, "synthetic.json", "1"*64, "2"*64, 20))
    monkeypatch.setattr(validation, "_oracle", lambda *_: (state, {"synthetic": True}))
    parameters = DingFatigueParameters(4000., .06, .137, -.4, 2.1e-5, 1.9e-5, 127.)
    audit = tmp_path / "synthetic_audit.json"
    audit.write_text(json.dumps({"synthetic_only": True}))
    documents = {
        "bounds": {"muscle_order": ["Biceps"], "declaration_reference": "synthetic regression only",
                   "lower": [[3900., 0., 0.]], "upper": [[5000., 1., 1.]]},
        "force_profiles": {"muscle_order": ["Biceps"], "model_sha256": "2"*64,
                           "assumption": "repeated_prescribed_force", "source_reference": "synthetic constant 50 N",
                           "muscles": {"Biceps": {"parameters": asdict(parameters), "force_samples": [[50., 50.]],
                                                 "interval_durations": [1.], "normalized_nodes": [0., 1.]}}},
        "continuation_label": label(observed_cycles=5, initial_state_receipt_sha256=state.receipt_sha256,
                                    audit_artifact={"path": audit.name, "sha256": hashlib.sha256(audit.read_bytes()).hexdigest()}),
    }
    case = {"id": "synthetic", "dataset": "T1", "oracle": {}, "horizon_cycles": 30}
    for key, document in documents.items():
        artifact = tmp_path / f"{key}.json"
        artifact.write_text(json.dumps(document))
        case[key] = {"path": artifact.name, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": validation.SCHEMA, "cases": [case]}))
    return path


def test_full_synthetic_diagnostic_has_no_activation_or_validation_claim(manifest):
    result = validation.validate_manifest(manifest)
    # Constant-force analytical threshold: A(n)=4000-2540*(1-exp(-n/127));
    # first A<3900 is n=6, so 5 completed cycles remain inside the bound.
    assert result["mean_absolute_error_cycles"] == 0
    assert result["point_error_coverage"] == 1
    assert result["cases"][0]["prediction"]["first_exceedance_cycle"] == 6
    assert result["controller_enabled"] is False
    assert result["validation_conclusion"].startswith("diagnostics_only")
    json.dumps(result, allow_nan=False)


def test_censored_manifest_has_no_point_error(manifest):
    payload = json.loads(manifest.read_text())
    payload["cases"][0]["horizon_cycles"] = 3
    payload["cases"][0]["dataset"] = "T3"
    manifest.write_text(json.dumps(payload))
    result = validation.validate_manifest(manifest)
    assert result["prediction_censored_count"] == 1
    assert result["mean_absolute_error_cycles"] is None
    assert result["point_error_coverage"] == 0
    assert result["cases"][0]["error_interval_cycles"] == [-2, None]


@pytest.mark.parametrize("field", ["bounds", "continuation_label", "force_profiles"])
def test_missing_or_changed_artifact_is_rejected(manifest, field):
    payload = json.loads(manifest.read_text())
    ref = payload["cases"][0][field]
    artifact = manifest.parent / ref["path"]
    artifact.write_text(artifact.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        validation.validate_manifest(manifest)
    del payload["cases"][0][field]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="content-pinned"):
        validation.validate_manifest(manifest)


@pytest.mark.parametrize("field,key,value", [
    ("bounds", "declaration_reference", ""), ("bounds", "muscle_order", ["Triceps"]),
    ("bounds", "upper", [[float("inf"), 1, 1]]),
    ("force_profiles", "model_sha256", "3"*64),
    ("force_profiles", "assumption", "future_force_prediction"),
    ("continuation_label", "initial_state_receipt_sha256", "3"*64),
    ("continuation_label", "audit_artifact", None),
])
def test_explicit_prescription_contract(manifest, field, key, value):
    payload = json.loads(manifest.read_text())
    ref = payload["cases"][0][field]
    artifact = manifest.parent / ref["path"]
    document = json.loads(artifact.read_text())
    document[key] = value
    artifact.write_text(json.dumps(document))
    ref["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validation.validate_manifest(manifest)
