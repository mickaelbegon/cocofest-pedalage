"""Offline comparison of conditional state-bound Nhat with certified labels.

No physiological bounds, acceptance tolerances, or continuation labels are
inferred here. The manifest explicitly supplies them. This diagnostic never
enables a controller and never treats a failed solver as an endurance label.
An external label is a content-pinned audit attestation, not a new proof of
infeasibility. Exact error is only reported when both endpoints are observed.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.ding_slow_cycle import slow_cycle_map_from_collocation
from .oracle import load_certified_checkpoint, load_certified_cycle
from .remaining_cycles import ConditionalRemainingCycles, SlowStateBounds, remaining_cycles
from .slow_model import OracleProvenance, SlowState


SCHEMA = "conditional_nhat_validation_v1"


def _integer(value, field):
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Explicit {field} is required")
    return value


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _object(data):
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _pinned(base, ref):
    if not isinstance(ref, dict):
        raise ValueError("A content-pinned artifact reference is required")
    path = Path(_text(ref.get("path"), "artifact path"))
    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
    data = path.read_bytes()
    digest = _digest(data)
    if ref.get("sha256") != digest:
        raise ValueError(f"Artifact SHA-256 mismatch: {path}")
    return path, data, {"path": str(path), "sha256": digest}


def compare_label(prediction: ConditionalRemainingCycles, label: dict, *, anchor_cycle: int) -> dict:
    """Compare against an explicitly audited continuation endpoint.

    Labels count *additional completed cycles* after the oracle anchor. An
    exact endpoint needs the external frozen-feasibility protocol's terminal
    attestation. Otherwise the certified observed prefix is right censored.
    Censored intervals use null for the unknown upper endpoint; they do not
    enter mean absolute error or bias.
    """
    anchor_cycle = _integer(anchor_cycle, "anchor_cycle")
    if label.get("schema_version") != 1 or label.get("certified") is not True:
        raise ValueError("An explicitly certified continuation label is required")
    if type(label.get("anchor_cycle")) is not int or label["anchor_cycle"] != anchor_cycle:
        raise ValueError("Label anchor differs from the oracle anchor")
    count = _integer(label.get("observed_cycles"), "observed_cycles")
    _text(label.get("protocol_id"), "continuation protocol_id")
    _text(label.get("audit_reference"), "continuation audit_reference")
    censoring = label.get("censoring")
    if censoring not in ("none", "right"):
        raise ValueError("censoring must be none or right")
    if censoring == "none" and (label.get("terminal_classification") != "frozen_feasibility_exhaustion"
                                 or label.get("terminal_certified") is not True):
        raise ValueError("An exact label requires certified frozen-feasibility exhaustion")
    if censoring == "right" and label.get("terminal_classification") not in (
        "administrative_horizon", "interrupted", "unresolved_solver_failure"
    ):
        raise ValueError("A right-censored label requires an explicit censoring reason")
    predicted = prediction.cycles_within_bounds
    jointly_exact = not prediction.censored and censoring == "none"
    # Intervals for prediction minus observation. A censored prediction or
    # observation provides a lower bound on its own endpoint, not an equality.
    return {
        "predicted_cycles_within_bounds": predicted,
        "prediction_censored": prediction.censored,
        "observed_cycles": count,
        "observation_censored": censoring == "right",
        "jointly_exact": jointly_exact,
        "signed_error_cycles": predicted - count if jointly_exact else None,
        "absolute_error_cycles": abs(predicted - count) if jointly_exact else None,
        "error_interval_cycles": [None if censoring == "right" else predicted - count,
                                   None if prediction.censored else predicted - count],
        "endpoint_interpretation": "conditional_state_bounds_vs_external_continuation_label",
    }


def _oracle(base, spec):
    receipt, _, pin = _pinned(base, spec.get("receipt"))
    kwargs = {"side": spec.get("side"), "muscles": spec.get("muscles")}
    if spec.get("kind") == "checkpoint":
        observed = load_certified_checkpoint(receipt, **kwargs)
    elif spec.get("kind") == "cycle":
        observed = load_certified_cycle(receipt, cycle=spec.get("cycle"), **kwargs)
    else:
        raise ValueError("oracle kind must be checkpoint or cycle")
    archive_path, archive_digest = observed.artifacts[1]
    values = observed.as_array()
    state = SlowState(observed.muscles, tuple(values[:, 0]), tuple(values[:, 1]), tuple(values[:, 2]),
                      OracleProvenance(archive_path, archive_digest, str(receipt), pin["sha256"],
                                       observed.model_sha256, observed.completed_cycles))
    return state, {"receipt": pin, "artifacts": [list(item) for item in observed.artifacts],
                   "source_kind": observed.source_kind, "source_node": observed.source_node}


def validate_manifest(path: str | Path) -> dict:
    """Evaluate T1/T3 cases, returning diagnostics without activation authority.

    Each case references pinned oracle, force-profile, bounds and label files.
    Bounds contain muscle_order/lower/upper/declaration_reference. Profiles
    contain model_sha256/muscle_order/assumption/source_reference/muscles, with
    per-muscle parameters/force_samples/interval_durations/normalized_nodes.
    Labels bind initial_state_receipt_sha256 and a pinned audit_artifact
    (resolved relative to the label file). That audit's terminal conclusion
    remains the supplying protocol's responsibility.
    The literal profile assumption is repeated_prescribed_force. Bounds must
    be finite JSON numbers; unrestricted bounds must be chosen explicitly by
    the study owner rather than being synthesized by this validator.
    """
    path = Path(path).resolve()
    raw = path.read_bytes()
    manifest = _object(raw)
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"manifest schema must be {SCHEMA}")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("At least one explicitly prescribed case is required")
    results, ids = [], set()
    for case in cases:
        case_id = _text(case.get("id"), "case id")
        if case_id in ids:
            raise ValueError("Case ids must be unique")
        ids.add(case_id)
        if case.get("dataset") not in ("T1", "T3"):
            raise ValueError("dataset must explicitly identify T1 or T3")
        horizon = _integer(case.get("horizon_cycles"), "horizon_cycles")
        state, oracle_provenance = _oracle(path.parent, case.get("oracle", {}))
        artifacts = {}
        artifact_paths = {}
        documents = {}
        for field in ("bounds", "force_profiles", "continuation_label"):
            artifact_paths[field], content, artifacts[field] = _pinned(path.parent, case.get(field))
            documents[field] = _object(content)
        bounds_spec, profiles, label = (documents[key] for key in ("bounds", "force_profiles", "continuation_label"))
        if label.get("initial_state_receipt_sha256") != state.receipt_sha256:
            raise ValueError("Continuation label must bind the exact oracle initial-state receipt")
        _, _, artifacts["continuation_audit"] = _pinned(
            artifact_paths["continuation_label"].parent, label.get("audit_artifact"))
        _text(bounds_spec.get("declaration_reference"), "bounds declaration_reference")
        _text(profiles.get("source_reference"), "force source_reference")
        for spec in (bounds_spec, profiles):
            if spec.get("muscle_order") != list(state.muscle_names):
                raise ValueError("Bounds/profile muscle order differs from oracle")
        if profiles.get("model_sha256") != state.provenance.model_sha256:
            raise ValueError("Force-profile model digest differs from oracle")
        if profiles.get("assumption") != "repeated_prescribed_force":
            raise ValueError("The repeated prescribed-force assumption must be explicit")
        bounds = SlowStateBounds(bounds_spec.get("lower"), bounds_spec.get("upper"))
        if not np.isfinite(bounds.lower).all() or not np.isfinite(bounds.upper).all():
            raise ValueError("Manifest bounds must be explicit finite numbers")
        muscle_profiles = profiles.get("muscles", {})
        if set(muscle_profiles) != set(state.muscle_names):
            raise ValueError("Force profiles must cover exactly the observed muscles")
        maps = {}
        for name, profile in muscle_profiles.items():
            maps[name] = slow_cycle_map_from_collocation(
                profile["force_samples"], profile["interval_durations"], profile["normalized_nodes"],
                DingFatigueParameters(**profile["parameters"]))
        prediction = remaining_cycles(state, maps, bounds, horizon=horizon)
        comparison = compare_label(prediction, label, anchor_cycle=state.provenance.cycle)
        results.append({"id": case_id, "dataset": case["dataset"], **comparison,
                        "prediction": asdict(prediction),
                        "provenance": {"oracle": oracle_provenance, **artifacts}})
    exact = [row for row in results if row["jointly_exact"]]
    return {
        "schema": SCHEMA, "manifest": {"path": str(path), "sha256": _digest(raw)},
        "case_count": len(results), "jointly_exact_count": len(exact),
        "point_error_coverage": len(exact) / len(results),
        "prediction_censored_count": sum(row["prediction_censored"] for row in results),
        "observation_censored_count": sum(row["observation_censored"] for row in results),
        "mean_absolute_error_cycles": float(np.mean([r["absolute_error_cycles"] for r in exact])) if exact else None,
        "mean_signed_error_cycles": float(np.mean([r["signed_error_cycles"] for r in exact])) if exact else None,
        "controller_enabled": False,
        "validation_conclusion": "diagnostics_only_no_prescribed_acceptance_criterion",
        "interpretation": "Conditional repeated-force bound crossing is not an endurance certificate; external labels are audit attestations.",
        "cases": results,
    }
