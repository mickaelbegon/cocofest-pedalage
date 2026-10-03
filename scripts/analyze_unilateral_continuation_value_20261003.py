#!/usr/bin/env python3
"""Audit continuation labels and directional value fits for one FES arm.

The input is an immutable, predeclared analysis manifest.  A continuation is
an exact label only when its own receipt certifies the terminal outcome.  A
completed RHO prefix followed by an unsuccessful NLP is *not* a failure label.
No model produced here is activated in the RHO.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.configured_cycling_model import resolve_model_config
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.ding_slow_cycle import coupled_slow_offsets

SCHEMA_VERSION = 1
VALUE_KIND = "unilateral_continuation_label_protocol"
PARTITIONS = {"train", "holdout"}
EXACT_OUTCOME = "physiological_failure_certified"
UNIT_FAMILY = "unit"


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _path(owner: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Nonempty artifact path required")
    result = (owner.parent / value).resolve(strict=True)
    if not result.is_file():
        raise ValueError(f"Artifact is not a file: {result}")
    return result


def _digest(path: Path, expected: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Missing SHA-256 for {path}")
    if sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Artifact SHA-256 mismatch: {path}")


def endpoint_coordinates(archive: Path, model_config: dict, *, offset_tolerance: float = 1e-7) -> dict:
    """Use independent Ding capacities; audit, but do not fit, slow offsets.

    The force-driven terms cancel in the Tau1/Km offsets.  On the present
    initial manifold they should remain zero, while measured off-manifold
    states would require a larger model and must not silently be projected.
    Fast states and the complete last-PW history are retained for trust audits.
    """
    resolved = resolve_model_config(model_config)
    names = tuple(resolved["muscles"])
    if not math.isfinite(offset_tolerance) or offset_tolerance <= 0:
        raise ValueError("offset_tolerance must be finite and positive")
    capacity, offsets, fast, history = {}, {}, {}, {}
    with np.load(archive, allow_pickle=False) as prepared:
        for name in names:
            params = DingFatigueParameters(**{
                key: resolved["muscles"][name][source] for key, source in (
                    ("a_rest", "a_scale"), ("tau1_rest", "tau1_rest"),
                    ("km_rest", "km_rest"), ("alpha_a", "alpha_a"),
                    ("alpha_tau1", "alpha_tau1"), ("alpha_km", "alpha_km"),
                    ("tau_fat", "tau_fat"))})
            slow = np.asarray([_first(prepared, f"states__{state}_{name}")
                               for state in ("A", "Tau1", "Km")], float)
            capacity[name] = float(slow[0] / params.a_rest)
            offset = coupled_slow_offsets(slow, params)
            offsets[name] = [float(offset[0]), float(offset[1])]
            fast[name] = [_first(prepared, f"states__{state}_{name}") for state in ("Cn", "F")]
            history[name] = _vector(prepared, f"controls__last_pulse_width_{name}")
        mechanics = {name: _first(prepared, f"states__{name}") for name in ("theta", "omega", "E_prod")}
    if any(not math.isfinite(v) or v <= 0 for v in capacity.values()):
        raise ValueError("Endpoint capacity ratios must be finite and positive")
    normalized_offsets = {name: [abs(offsets[name][0]) / resolved["muscles"][name]["tau1_rest"],
                                 abs(offsets[name][1]) / resolved["muscles"][name]["km_rest"]]
                          for name in names}
    offset_max = max(value for pair in normalized_offsets.values() for value in pair)
    return {"muscle_order": list(names), "capacity_ratios": capacity,
            "slow_offsets": offsets, "maximum_relative_slow_offset": offset_max,
            "slow_manifold_passed": offset_max <= offset_tolerance,
            "fast_states": fast, "last_pulse_width_history": history, "mechanical_states": mechanics}


def _first(archive, key: str) -> float:
    if key not in archive:
        raise ValueError(f"Endpoint archive lacks {key}")
    value = np.asarray(archive[key], dtype=float).reshape(-1)
    if not value.size or not math.isfinite(value[0]):
        raise ValueError(f"Endpoint archive has invalid {key}")
    return float(value[0])


def _vector(archive, key: str) -> list[float]:
    if key not in archive:
        raise ValueError(f"Endpoint archive lacks {key}")
    value = np.asarray(archive[key], dtype=float).reshape(-1)
    if not value.size or not np.isfinite(value).all():
        raise ValueError(f"Endpoint archive has invalid {key}")
    return value.tolist()


def _predeclared_rows(manifest: dict) -> list[dict]:
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("kind") != VALUE_KIND:
        raise ValueError("Unsupported continuation-value manifest")
    if (manifest.get("side") != "left" or manifest.get("return_policy", {}).get("equivalent_mean_torque_nm") != .96
            or manifest.get("return_policy", {}).get("maximum_absolute_cycles") != 240):
        raise ValueError("This protocol requires the left arm at 0.96 Nm")
    expected_policy = sha256(json.dumps(manifest["return_policy"], sort_keys=True, allow_nan=False,
                                        separators=(",", ":")).encode()).hexdigest()
    if manifest.get("return_policy_sha256") != expected_policy:
        raise ValueError("Return policy digest differs from its preregistration")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Manifest must predeclare jobs")
    ids, family_partition, rows = set(), {}, []
    for job in jobs:
        if not isinstance(job, dict) or not isinstance(job.get("id"), str) or job["id"] in ids:
            raise ValueError("Each row needs a unique id")
        ids.add(job["id"])
        action, anchor = job.get("action", {}), job.get("anchor", {})
        partition = action.get("partition")
        if (partition not in PARTITIONS | {"control"} or type(anchor.get("cycle")) is not int
                or not isinstance(action.get("action_group"), str)
                or type(job.get("intervention_cycles")) is not int
                or job["intervention_cycles"] not in (1, 3)):
            raise ValueError("Missing anchor, family, partition or intervention duration")
        key = action["action_group"]
        previous = family_partition.setdefault(key, partition)
        if previous != partition:
            raise ValueError("Shared-prefix family leaks between train and holdout")
        if action["action_group"] == UNIT_FAMILY and (
                partition != "control" or action.get("id") != UNIT_FAMILY
                or any(action.get("weights", {}).get(name) != 1. for name in manifest.get("muscle_order", []))):
            raise ValueError("Unit comparator is not the declared fixed unit policy")
        rows.append({"id": job["id"], "anchor_id": f"c{anchor['cycle']}",
                     "anchor_cycle": anchor["cycle"], "family": action["action_group"],
                     "partition": "train" if partition == "control" else partition,
                     "intervention_cycles": job["intervention_cycles"],
                     "return_policy_sha256": manifest["return_policy_sha256"], "job": job})
    if not all(sum(row["family"] == UNIT_FAMILY and row["anchor_id"] == anchor
                   and row["intervention_cycles"] == duration for row in rows) == 1
               for anchor in {row["anchor_id"] for row in rows} for duration in (1, 3)):
        raise ValueError("Each anchor and duration requires a predeclared unit comparator")
    return rows


def _audited_cycle_rows(rows: list, *, start: int, stop: int, tolerance: float) -> None:
    if not isinstance(rows, list) or [row.get("cycle") for row in rows] != list(range(start, stop + 1)):
        raise ValueError("Certified cycle prefix is noncontiguous")
    for row in rows:
        violation = row.get("maximum_normalized_violation")
        detail = row.get("independent_audit", {})
        if (row.get("certified") is not True or isinstance(violation, bool)
                or not isinstance(violation, (int, float)) or not math.isfinite(violation)
                or violation < 0 or violation > tolerance
                or detail.get("scope") != "complete_discrete_NLP_g_and_x; no continuous-ODE replay"
                or not isinstance(row.get("witness"), str)):
            raise ValueError("Uncertified or incompletely audited cycle")
        witness = Path(row["witness"]).resolve(strict=True)
        _digest(witness, row.get("witness_sha256"))


def _read_completed_job(manifest_path: Path, manifest: dict, row: dict) -> dict:
    """Validate producer/continuation provenance before accepting a label."""
    job = row["job"]
    directory = Path(job["output"]).resolve()
    summary_path = directory / "job-result.json"
    if not summary_path.is_file():
        return {**row, "label_kind": "pending", "exact_label_allowed": False}
    summary = _json(summary_path)
    if summary.get("job_id") != row["id"]:
        raise ValueError("Job summary id differs from manifest")
    if summary.get("phase_failed") is not None:
        return {**row, "label_kind": "numerical_stop", "exact_label_allowed": False,
                "reason": f"phase_failed:{summary['phase_failed']}"}
    intervention_path = directory / "intervention" / "result.json"
    continuation_path = directory / "continuation" / "result.json"
    if (not intervention_path.is_file() or not continuation_path.is_file()
            or Path(summary.get("result", "")).resolve() != continuation_path.resolve()):
        raise ValueError("Job summary does not identify both phase receipts")
    _digest(continuation_path, summary.get("result_sha256"))
    intervention, continuation = _json(intervention_path), _json(continuation_path)
    manifest_digest = sha256(manifest_path.read_bytes()).hexdigest()
    anchor = job["anchor"]
    source_receipt = Path(anchor["receipt"]).resolve(strict=True)
    _digest(source_receipt, anchor["receipt_sha256"])
    for phase, receipt in (("intervention", intervention), ("continuation", continuation)):
        if (receipt.get("schema_version") != 1 or receipt.get("kind") != "unilateral_continuation_phase"
                or receipt.get("phase") != phase or receipt.get("job_id") != row["id"]
                or receipt.get("manifest_sha256") != manifest_digest
                or receipt.get("source_receipt_sha256") != anchor["receipt_sha256"]
                or receipt.get("return_policy_sha256") != manifest["return_policy_sha256"]
                or receipt.get("return_policy") != manifest["return_policy"]
                or receipt.get("anchor_completed_cycles") != anchor["cycle"]
                or receipt.get("intervention_cycles") != job["intervention_cycles"]
                or receipt.get("action") != job["action"]
                or receipt.get("muscle_order") != manifest["muscle_order"]
                or not math.isclose(receipt.get("nominal_work_j", math.nan), 2 * math.pi * .96,
                                    rel_tol=0, abs_tol=1e-10)):
            raise ValueError(f"{phase} receipt differs from predeclared task or policy")
    if (intervention.get("success") is not True or intervention.get("completed") is not True
            or intervention.get("last_certified_cycle") != anchor["cycle"] + job["intervention_cycles"]):
        raise ValueError("Intervention did not certify its full requested prefix")
    tolerance = float(manifest["audit_tolerance"])
    _audited_cycle_rows(intervention.get("cycles"), start=anchor["cycle"] + 1,
                        stop=anchor["cycle"] + job["intervention_cycles"], tolerance=tolerance)
    endpoint = intervention.get("prepared_endpoint", {})
    archive = Path(endpoint.get("primal_path", "")).resolve(strict=True)
    _digest(archive, endpoint.get("sha256"))
    if (endpoint.get("completed_cycles") != anchor["cycle"] + job["intervention_cycles"]
            or continuation.get("endpoint") != endpoint
            or continuation.get("endpoint_receipt_sha256") != sha256(intervention_path.read_bytes()).hexdigest()
            or Path(continuation.get("endpoint_receipt", "")).resolve() != intervention_path.resolve()
            or continuation.get("source_archive_sha256") != endpoint["sha256"]
            or continuation.get("source_restore", {}).get("restored_problem_sha256")
            != endpoint.get("prepared_problem_sha256")
            or continuation.get("source_restore", {}).get("stimulation_history_complete") is not True
            or continuation.get("fresh_worker_replay_verified") is not True
            or continuation.get("producer_pid") == intervention.get("producer_pid")):
        raise ValueError("Fresh continuation did not restore the certified intervention endpoint")
    last = continuation.get("last_certified_cycle")
    if type(last) is not int or last < endpoint["completed_cycles"] or last > 240:
        raise ValueError("Continuation last certified cycle invalid")
    _audited_cycle_rows(continuation.get("cycles"), start=endpoint["completed_cycles"] + 1,
                        stop=last, tolerance=tolerance)
    if (continuation.get("continuation_cycles") != last - endpoint["completed_cycles"]
            or continuation.get("total_cycles_after_anchor") != last - anchor["cycle"]
            or continuation.get("completed") is not True):
        raise ValueError("Continuation cycle accounting differs from its audited prefix")
    status = continuation.get("status")
    if status not in (EXACT_OUTCOME, "right_censored", "numerical_stop") or summary.get("status") != status:
        raise ValueError("Continuation classification missing or inconsistent")
    if status == EXACT_OUTCOME:
        terminal = continuation.get("terminal_attempt") or {}
        zero = terminal.get("zero_objective_feasibility_probe") or {}
        fatigue = (terminal.get("counterfactual_probes") or {}).get("fatigue_rest") or {}
        if (continuation.get("exact_label_allowed") is not True or summary.get("exact_label_allowed") is not True
                or terminal.get("cycle") != last + 1 or zero.get("available") is not True
                or zero.get("constraints_unchanged") is not True or zero.get("feasible_witness") is not False
                or (zero.get("solver_stats") or {}).get("return_status") != "Infeasible_Problem_Detected"
                or fatigue.get("feasible_witness") is not True
                or fatigue.get("physical_rho_advanced") is not False):
            raise ValueError("Exact label lacks the declared local frozen endpoint certificate")
    elif continuation.get("exact_label_allowed") is not False or summary.get("exact_label_allowed") is not False:
        raise ValueError("Censored or numerical continuation cannot be an exact label")
    source, _ = _load_verified_arm(source_receipt)
    if (source.completed_cycles != anchor["cycle"] or source.archive_sha256 != anchor["archive_sha256"]
            or source.prepared_problem_sha256 != anchor["prepared_problem_sha256"]):
        raise ValueError("Anchor changed since preregistration")
    model_path = Path(source.model_path)
    coordinates = endpoint_coordinates(archive, _json(model_path))
    if not coordinates["slow_manifold_passed"]:
        status = "off_slow_manifold"
    return {**row, "label_kind": status, "exact_label_allowed": status == EXACT_OUTCOME,
            "last_certified_cycle": last, "model_sha256": source.model_sha256,
            "endpoint_archive_sha256": endpoint["sha256"], **coordinates}


def _load_verified_arm(receipt: Path):
    from scripts.probe_independent_rho_task_reserve import _load_arm
    return _load_arm(receipt, "left")


def _block(row: dict) -> tuple[str, int, str]:
    return row["anchor_id"], row["intervention_cycles"], row["return_policy_sha256"]


def _nuisance_vector(row: dict) -> np.ndarray | None:
    if not all(key in row for key in ("fast_states", "last_pulse_width_history", "mechanical_states")):
        return None
    values = []
    for name in row["muscle_order"]:
        values.extend(row["fast_states"][name])
        values.extend(row["last_pulse_width_history"][name])
    values.extend(row["mechanical_states"][name] for name in ("theta", "omega", "E_prod"))
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Endpoint fast state/history is nonfinite")
    return result


def _exact_labels(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Center every action on its same-anchor, same-duration unit comparator."""
    baseline = {}
    for row in rows:
        if row["family"] == UNIT_FAMILY and row.get("label_kind") == EXACT_OUTCOME:
            key = _block(row)
            if key in baseline:
                raise ValueError("Two exact unit baselines in one block")
            baseline[key] = row
    labelled, excluded = [], []
    for row in rows:
        if row["family"] == UNIT_FAMILY:
            continue
        unit = baseline.get(_block(row))
        if row.get("label_kind") != EXACT_OUTCOME or unit is None:
            excluded.append({"id": row["id"], "reason": "no_exact_label_or_matched_unit_baseline"})
            continue
        if row["muscle_order"] != unit["muscle_order"]:
            raise ValueError("Muscle order differs within a matched block")
        x = np.asarray([row["capacity_ratios"][name] for name in row["muscle_order"]], float)
        x0 = np.asarray([unit["capacity_ratios"][name] for name in unit["muscle_order"]], float)
        fast, fast0 = _nuisance_vector(row), _nuisance_vector(unit)
        if (fast is None) != (fast0 is None):
            raise ValueError("Endpoint nuisance coordinates missing from only one matched receipt")
        labelled.append({**row, "baseline_id": unit["id"], "delta_capacity": (x - x0).tolist(),
                         "delta_fast_history": None if fast is None else (fast - fast0).tolist(),
                         "advantage_cycles": row["last_certified_cycle"] - unit["last_certified_cycle"]})
    return labelled, excluded


def fit_reachable_direction(rows: list[dict], *, max_condition: float = 1e5,
                            minimum_rank: int = 1, minimum_holdout_pairs: int = 2) -> dict:
    """Fit only observed capacity directions; publish no full-state costate.

    One endpoint per family enters the rank/fit.  Other durations are retained
    as consistency checks and never counted as independent samples.  Held-out
    families are excluded from the SVD, scale and coefficients.
    """
    if len({row["anchor_id"] for row in rows}) != 1:
        raise ValueError("A local costate fit must use exactly one anchor")
    labelled, excluded = _exact_labels(rows)
    chosen = {}
    for row in labelled:
        key = (row["anchor_id"], row["family"])
        if key not in chosen or row["intervention_cycles"] > chosen[key]["intervention_cycles"]:
            chosen[key] = row
    train = [row for row in chosen.values() if row["partition"] == "train"]
    holdout = [row for row in chosen.values() if row["partition"] == "holdout"]
    reasons = []
    if not train:
        reasons.append("no_exact_training_continuations")
    if len(holdout) < minimum_holdout_pairs:
        reasons.append("insufficient_independent_exact_holdout_families")
    if reasons:
        return {"accepted": False, "reasons": reasons, "excluded": excluded,
                "exact_train_families": len(train), "exact_holdout_families": len(holdout)}
    order = train[0]["muscle_order"]
    if any(row["muscle_order"] != order for row in train + holdout):
        raise ValueError("Muscle orders differ across continuation receipts")
    x_train = np.asarray([row["delta_capacity"] for row in train], float)
    y_train = np.asarray([row["advantage_cycles"] for row in train], float)
    x_hold = np.asarray([row["delta_capacity"] for row in holdout], float)
    y_hold = np.asarray([row["advantage_cycles"] for row in holdout], float)
    scale = np.maximum(np.max(np.abs(x_train), axis=0), 1e-8)
    design = x_train / scale
    u, singular, vt = np.linalg.svd(design, full_matrices=False)
    rank = int(np.linalg.matrix_rank(design))
    condition = float(singular[0] / singular[rank - 1]) if rank else math.inf
    if rank < minimum_rank:
        reasons.append("reachable_capacity_design_rank_too_low")
    if condition > max_condition:
        reasons.append("reachable_capacity_design_ill_conditioned")
    if rank == 0 or reasons:
        return {"accepted": False, "reasons": reasons, "reachable_rank": rank,
                "reachable_condition": condition, "excluded": excluded,
                "exact_train_families": len(train), "exact_holdout_families": len(holdout)}
    basis = vt[:rank].T
    reduced_train = design @ basis
    coeff = np.linalg.lstsq(reduced_train, y_train, rcond=None)[0]
    predicted = (x_hold / scale) @ basis @ coeff
    projected = (x_hold / scale) @ basis @ basis.T
    off_span = np.max(np.abs(x_hold / scale - projected), axis=1)
    # A model fitted to contrasts cannot claim validity on unseen capacity
    # directions.  This predeclared geometric gate is stricter than residuals.
    projected_train = design @ basis
    projected_hold = (x_hold / scale) @ basis
    observed_span = np.max(np.abs(projected_train), axis=0)
    along_span = np.all(np.abs(projected_hold) <= observed_span + 1e-10, axis=1)
    trusted = (off_span <= 0.1) & along_span
    nuisance_available = all(row.get("delta_fast_history") is not None for row in train + holdout)
    if nuisance_available:
        nuisance_train = np.asarray([row["delta_fast_history"] for row in train], float)
        nuisance_hold = np.asarray([row["delta_fast_history"] for row in holdout], float)
        if nuisance_train.shape[1] != nuisance_hold.shape[1]:
            raise ValueError("Fast/history endpoint coordinate dimension changed")
        nuisance_limit = np.max(np.abs(nuisance_train), axis=0) + 1e-9
        nuisance_inside = np.all(np.abs(nuisance_hold) <= nuisance_limit, axis=1)
    else:
        nuisance_inside = np.full(len(holdout), False)
    pair_total = pair_correct = sign_total = sign_correct = 0
    for index, observed in enumerate(y_hold):
        if trusted[index] and observed != 0:
            sign_total += 1
            sign_correct += int(np.sign(predicted[index]) == np.sign(observed))
        for later in range(index + 1, len(y_hold)):
            if trusted[index] and trusted[later] and observed != y_hold[later]:
                pair_total += 1
                pair_correct += int(np.sign(predicted[index] - predicted[later])
                                    == np.sign(observed - y_hold[later]))
    if not trusted.all():
        reasons.append("holdout_outside_reachable_training_span")
    if nuisance_available and not nuisance_inside.all():
        reasons.append("unmodeled_fast_history_outside_training_envelope")
    if sign_total < 2 or pair_total < 1:
        reasons.append("insufficient_non_tied_holdout_contrasts")
    if sign_total and sign_correct / sign_total < 0.8:
        reasons.append("heldout_sign_accuracy_below_0p8")
    if pair_total and pair_correct / pair_total < 0.75:
        reasons.append("heldout_pair_concordance_below_0p75")
    return {"accepted": not reasons, "reasons": reasons, "muscle_order": order,
            "reachable_rank": rank, "reachable_condition": condition,
            "capacity_scale": scale.tolist(), "reachable_basis": basis.tolist(),
            "projected_gradient_cycles_per_capacity_ratio": ((basis @ coeff) / scale).tolist(),
            "fast_history_envelope_available": nuisance_available,
            "exact_train_families": len(train), "exact_holdout_families": len(holdout),
            "holdout": [{"id": row["id"], "observed_advantage_cycles": row["advantage_cycles"],
                         "predicted_advantage_cycles": float(predicted[index]),
                         "off_span_distance": float(off_span[index]),
                         "inside_training_range": bool(along_span[index]),
                         "fast_history_inside_training_envelope": bool(nuisance_inside[index]),
                         "trusted": bool(trusted[index])}
                        for index, row in enumerate(holdout)],
            "heldout_sign_correct": sign_correct, "heldout_sign_total": sign_total,
            "heldout_pair_correct": pair_correct, "heldout_pair_total": pair_total,
            "excluded": excluded, "activation_allowed": False}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = args.manifest.resolve(strict=True)
    plan = _json(manifest)
    planned = _predeclared_rows(plan)
    rows = [_read_completed_job(manifest, plan, row) for row in planned]
    controls = [row for row in rows if row["family"] == UNIT_FAMILY]
    control_gate = (len(controls) == 4 and all(row.get("label_kind") == EXACT_OUTCOME
                    and row.get("last_certified_cycle") == plan["control_gate"]["expected_last_certified_cycle"]
                    for row in controls))
    by_anchor = {}
    for anchor in sorted({row["anchor_id"] for row in rows}):
        subset = [row for row in rows if row["anchor_id"] == anchor]
        by_anchor[anchor] = (fit_reachable_direction(subset) if control_gate else
                             {"accepted": False, "reasons": ["control_replay_gate_pending_or_failed"]})
    observed, _ = _exact_labels(rows)
    status_counts = {status: sum(row.get("label_kind") == status for row in rows)
                     for status in ("pending", EXACT_OUTCOME, "right_censored", "numerical_stop",
                                    "off_slow_manifold")}
    report = {"schema_version": 1, "kind": "unilateral_continuation_value_directional_validation",
              "manifest": str(manifest), "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
              "side": "left", "nominal_torque_nm": .96,
              "return_policy_sha256": plan["return_policy_sha256"],
              "label_interpretation": "locally certified frozen endpoint under the declared reference RHO, not global infeasibility",
              "control_replay_gate_passed": control_gate, "planned_jobs": len(rows),
              "status_counts": status_counts, "anchors": by_anchor,
              "maximum_relative_slow_offset_observed": max(
                  (row["maximum_relative_slow_offset"] for row in rows
                   if "maximum_relative_slow_offset" in row), default=None),
              "observed_exact_action_advantages": [
                  {"id": row["id"], "anchor_id": row["anchor_id"],
                   "family": row["family"], "partition": row["partition"],
                   "intervention_cycles": row["intervention_cycles"],
                   "advantage_cycles": row["advantage_cycles"],
                   "delta_capacity_ratios": dict(zip(row["muscle_order"], row["delta_capacity"]))}
                  for row in observed],
              "all_anchors_pass_directional_holdout": control_gate and all(v.get("accepted") for v in by_anchor.values()),
              "activation_allowed": False,
              "unmodeled_endpoint_covariates": ["Cn", "F", "last_pulse_width_history", "phase and mechanical state"],
              "next_gate": "matched +g/-g/0 causal RHO branches before any endurance claim"}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
