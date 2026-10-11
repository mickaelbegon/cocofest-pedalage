"""Audited paired evidence for a *local robustness* fast-state gate.

This deliberately cannot prove Markov closure.  It can only say that a
preregistered finite set of admissible, slow-matched fast-state pairs did not
change the tested finite action ranking beyond repeated-control resolution.
Missing pairs, censoring, numerical failures, or incomplete coverage remain
inconclusive and cannot open the gate.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from .fast_state_probes import DecisionObservation, decision_pair_effect


REQUIRED_FAST_GROUPS = frozenset(("Cn", "F", "PW_history", "mechanics"))
ACTIONS = (-.05, 0., .05)
FROZEN_ENDPOINT = "frozen_local_endpoint"


def digest(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(document: Mapping[str, Any], *, field: str) -> str:
    copy = dict(document)
    copy.pop(field, None)
    return sha256(json.dumps(copy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _action(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Action must be finite")
    value = float(value)
    if not any(math.isclose(value, declared, abs_tol=1e-12) for declared in ACTIONS):
        raise ValueError("Action is not preregistered")
    return value


def verify_preregistration(plan: Mapping[str, Any]) -> dict[str, dict]:
    """Verify the immutable plan and return its sources keyed by receipt hash."""
    if not isinstance(plan, Mapping) or plan.get("schema") != "slow-phi-fast-closure-preregistration-v1":
        raise ValueError("Unexpected fast-state preregistration schema")
    if plan.get("plan_sha256") != canonical_digest(plan, field="plan_sha256"):
        raise ValueError("Fast-state preregistration changed")
    if plan.get("status") != "preregistered_not_executed":
        raise ValueError("Only an unexecuted preregistration can define this gate")
    controls = plan.get("controls", {})
    repeats = controls.get("independent_repeats_per_action") if isinstance(controls, Mapping) else None
    if type(repeats) is not int or repeats < 2:
        raise ValueError("Preregistration lacks repeated control requirement")
    actions = plan.get("actions", {})
    if not isinstance(actions, Mapping) or tuple(actions.get("left_transfer_nm", ())) != ACTIONS:
        raise ValueError("Preregistration action family differs")
    sources = {}
    for row in plan.get("sources", ()):  # exact source receipt is the experimental stratum
        if not isinstance(row, Mapping):
            raise ValueError("Malformed preregistered source")
        artifacts = row.get("artifacts", ())
        receipt = next((item for item in artifacts if isinstance(item, Mapping) and item.get("path") == row.get("receipt")), None)
        if receipt is None or not _is_digest(receipt.get("sha256")) or not _is_digest(row.get("model_sha256")):
            raise ValueError("Preregistered source lacks immutable receipt/model hashes")
        if receipt["sha256"] in sources:
            raise ValueError("Preregistration repeats a source receipt")
        sources[receipt["sha256"]] = dict(row)
    if not sources:
        raise ValueError("Preregistration has no sources")
    return sources


def _artifacts_valid(items: object, validator: Callable[[str, str], bool]) -> bool:
    if not isinstance(items, list) or not items:
        return False
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not _is_digest(item.get("sha256")):
            return False
        try:
            if not validator(item["path"], item["sha256"]):
                return False
        except (OSError, ValueError):
            return False
    return True


def _outcome(item: object, muscles: tuple[str, ...], validator: Callable[[str, str], bool]) -> dict | None:
    if not isinstance(item, Mapping) or item.get("endpoint") != FROZEN_ENDPOINT or item.get("audited") is not True:
        return None
    cycles = item.get("accepted_cycles")
    if type(cycles) is not int or cycles < 0 or not isinstance(item.get("feasible"), bool):
        return None
    force = item.get("weighted_force_integral")
    if not isinstance(force, Mapping) or set(force) != set(muscles):
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in force.values()):
        return None
    if not _artifacts_valid(item.get("audits"), validator):
        return None
    return dict(item)


def _resolution(controls: list[Mapping[str, Any]], *, sources: Mapping[str, dict], repeats: int,
                validator: Callable[[str, str], bool]) -> tuple[dict[tuple[str, float], float], list[str]]:
    values: dict[tuple[str, float], list[int]] = defaultdict(list)
    ids: dict[tuple[str, float], set[str]] = defaultdict(set)
    failures: list[str] = []
    for row in controls:
        if not isinstance(row, Mapping):
            failures.append("malformed_control")
            continue
        source = row.get("source_receipt_sha256")
        try:
            action = _action(row.get("delta_left_nm"))
        except ValueError:
            failures.append("control_action_not_preregistered")
            continue
        key = (source, action)
        outcome = _outcome(row, tuple(), validator)
        # Controls do not measure weighted force; their audit contract is
        # deliberately smaller than an action-pair outcome.
        if (source not in sources or outcome is not None or row.get("endpoint") != FROZEN_ENDPOINT
                or row.get("audited") is not True or type(row.get("accepted_cycles")) is not int
                or row["accepted_cycles"] < 0 or not _artifacts_valid(row.get("audits"), validator)):
            failures.append(f"invalid_control:{source}:{action}")
            continue
        replicate = row.get("replicate_id")
        if not isinstance(replicate, str) or not replicate or replicate in ids[key]:
            failures.append(f"nonindependent_control:{source}:{action}")
            continue
        ids[key].add(replicate)
        values[key].append(row["accepted_cycles"])
    resolutions = {}
    for source in sources:
        for action in ACTIONS:
            key = (source, action)
            if len(values[key]) < repeats:
                failures.append(f"insufficient_controls:{source}:{action}")
            else:
                resolutions[key] = float(max(values[key]) - min(values[key]))
    return resolutions, failures


def assess_fast_state_pairs(plan: Mapping[str, Any], evidence: Mapping[str, Any], *,
                            artifact_validator: Callable[[str, str], bool] | None = None) -> dict:
    """Apply frozen criteria; returns validation, rejection, or inconclusion.

    ``validated_local_robustness`` is intentionally not called Markov closure.
    It requires finite, audited, exactly slow-matched pairs for every source
    and each declared action, plus repeated-control resolution in every
    stratum.  It is a narrow use gate only.
    """
    sources = verify_preregistration(plan)
    if not isinstance(evidence, Mapping) or evidence.get("schema") != "slow-phi-fast-pair-evidence-v1":
        raise ValueError("Unexpected fast-state pair evidence schema")
    if evidence.get("plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("Evidence is not bound to this preregistration")
    validator = artifact_validator or (lambda path, expected: digest(path) == expected)
    repeats = plan["controls"]["independent_repeats_per_action"]
    resolution, issues = _resolution(evidence.get("controls", []), sources=sources, repeats=repeats, validator=validator)
    results, seen_sources, negative = [], set(), []
    muscles = tuple(plan["muscles"])
    for pair in evidence.get("pairs", []):
        if not isinstance(pair, Mapping):
            issues.append("malformed_pair")
            continue
        source = pair.get("source_receipt_sha256")
        if source not in sources:
            issues.append("unknown_pair_source")
            continue
        seen_sources.add(source)
        identity = pair.get("identity")
        if not isinstance(identity, Mapping) or set(identity) != {"model_hash", "task_hash", "slow_boundary_sha256", "return_policy_sha256", "stopping_rule_sha256", "discretization_sha256"}:
            issues.append(f"incomplete_exact_match_identity:{source}")
            continue
        if identity["model_hash"] != sources[source]["model_sha256"] or any(not _is_digest(identity[name]) for name in identity if name != "model_hash"):
            issues.append(f"identity_not_bound_to_source:{source}")
            continue
        if not _is_digest(pair.get("first_fast_state_sha256")) or not _is_digest(pair.get("second_fast_state_sha256")) or pair["first_fast_state_sha256"] == pair["second_fast_state_sha256"]:
            issues.append(f"fast_states_not_distinct:{source}")
            continue
        coverage = pair.get("coverage")
        if not isinstance(coverage, Mapping) or set(coverage) != REQUIRED_FAST_GROUPS or any(
                coverage[group] not in {"paired_different", "verified_invariant"} for group in REQUIRED_FAST_GROUPS):
            issues.append(f"fast_group_coverage_incomplete:{source}")
            continue
        actions = pair.get("actions")
        if not isinstance(actions, list) or len(actions) != len(ACTIONS):
            issues.append(f"action_family_incomplete:{source}")
            continue
        first_scores, second_scores = {}, {}
        valid_pair = True
        for row in actions:
            try:
                action = _action(row.get("delta_left_nm"))
            except (AttributeError, ValueError):
                valid_pair = False
                break
            if action in first_scores or (source, action) not in resolution:
                valid_pair = False
                break
            first, second = _outcome(row.get("first"), muscles, validator), _outcome(row.get("second"), muscles, validator)
            if first is None or second is None:
                issues.append(f"unresolved_or_unaudited_outcome:{source}:{action}")
                valid_pair = False
                break
            if first["feasible"] != second["feasible"]:
                negative.append(f"feasibility_changed:{source}:{action}")
            effect = decision_pair_effect(
                DecisionObservation(f"{source}:{identity['slow_boundary_sha256']}:{action}", pair["first_fast_state_sha256"], first["accepted_cycles"], FROZEN_ENDPOINT),
                DecisionObservation(f"{source}:{identity['slow_boundary_sha256']}:{action}", pair["second_fast_state_sha256"], second["accepted_cycles"], FROZEN_ENDPOINT),
                resolution_cycles=resolution[(source, action)],
            )
            if effect["classification"] != "within_observed_resolution":
                negative.append(f"fast_effect_exceeds_resolution:{source}:{action}")
            first_scores[action], second_scores[action] = first["accepted_cycles"], second["accepted_cycles"]
            results.append({"source_receipt_sha256": source, "pair_id": pair.get("pair_id"), "action": action, "effect": effect,
                            "weighted_force_integral": {"first": first["weighted_force_integral"], "second": second["weighted_force_integral"]}})
        if not valid_pair or set(first_scores) != set(ACTIONS):
            issues.append(f"action_family_invalid:{source}")
            continue
        for first_action in ACTIONS:
            for second_action in ACTIONS:
                if first_action >= second_action:
                    continue
                margin = max(resolution[(source, first_action)], resolution[(source, second_action)])
                sign_a = (first_scores[first_action] - first_scores[second_action] > margin) - (first_scores[first_action] - first_scores[second_action] < -margin)
                sign_b = (second_scores[first_action] - second_scores[second_action] > margin) - (second_scores[first_action] - second_scores[second_action] < -margin)
                if sign_a != sign_b:
                    negative.append(f"action_ranking_changed:{source}:{first_action}:{second_action}")
    missing = set(sources) - seen_sources
    issues.extend(f"missing_matched_pair:{source}" for source in sorted(missing))
    status = "validated_local_robustness" if not issues and not negative else "rejected_local_robustness" if negative else "inconclusive"
    return {"schema": "slow-phi-fast-state-gate-assessment-v1", "status": status,
            "closure_established": False,
            "fast_state_gate": status == "validated_local_robustness",
            "scope": "finite paired local robustness only; not Markov closure and not clinical activation evidence",
            "plan_sha256": plan["plan_sha256"], "control_resolution_cycles": {f"{source}:{action:+.2f}": value for (source, action), value in resolution.items()},
            "issues": sorted(set(issues)), "negative_findings": sorted(set(negative)), "pair_results": results}


def make_fast_state_gate_receipt(assessment: Mapping[str, Any], anchors: Mapping[str, Mapping[str, str]]) -> dict:
    """Issue the runner-compatible receipt only after the narrow gate passes."""
    if assessment.get("status") != "validated_local_robustness" or assessment.get("closure_established") is not False:
        raise ValueError("Only a passed local-robustness assessment may open the experimental gate")
    records = []
    for receipt, models in sorted(anchors.items()):
        if not _is_digest(receipt) or not isinstance(models, Mapping) or set(models) != {"left", "right"} or not all(_is_digest(models[side]) for side in models):
            raise ValueError("Gate anchors require an exact bilateral receipt/model binding")
        records.append({"receipt_sha256": receipt, "model_sha256": dict(models), "fast_state_gate": True,
                        "scope": "paired local robustness gate only; not Markov closure"})
    if not records:
        raise ValueError("At least one bilateral anchor is required")
    receipt = {"schema": "causal-transfer-fast-state-gate-v1", "status": "validated", "fast_state_gate": True,
               "closure_established": False, "validation_scope": "paired_local_robustness_v1_not_markov_closure",
               "assessment_sha256": sha256(json.dumps(assessment, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
               "anchors": records}
    receipt["receipt_sha256"] = canonical_digest(receipt, field="receipt_sha256")
    return receipt
