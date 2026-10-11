#!/usr/bin/env python3
"""Analyze predeclared, audited, paired bilateral torque-transfer branches.

This consumes a normalized export from the RHO runner. It does not certify RHO
receipts itself; ``audit_certified`` and terminal classifications must be set
only after independent receipt verification by the exporter.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path


ACTIONS = ("minus", "hold", "plus")
TERMINALS = {"exact": "frozen_feasibility_exhaustion",
             "censored": "administrative_horizon",
             "numerical": "numerical_unresolved"}


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _validated(manifest: dict) -> tuple[list[dict], list[dict]]:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("kind") != "torque_transfer_t3_pilot":
        raise ValueError("Expected torque_transfer_t3_pilot schema version 1")
    if manifest.get("intervention_cycles") != 10:
        raise ValueError("This protocol requires a 10-cycle maintained intervention")
    anchors, jobs = manifest.get("anchors"), manifest.get("jobs")
    if not isinstance(anchors, list) or not anchors or not isinstance(jobs, list) or not jobs:
        raise ValueError("Nonempty anchors and jobs are required")
    by_id = {}
    for anchor in anchors:
        if not isinstance(anchor, dict) or not isinstance(anchor.get("id"), str) or not anchor["id"] or anchor["id"] in by_id:
            raise ValueError("Anchor ids must be unique nonempty strings")
        _integer(anchor.get("cycle"), "anchor cycle")
        if anchor.get("chosen_action") is not None and anchor["chosen_action"] not in ACTIONS:
            raise ValueError("chosen_action must be minus, hold, or plus")
        by_id[anchor["id"]] = anchor
    seen = set()
    for job in jobs:
        if not isinstance(job, dict) or job.get("anchor_id") not in by_id:
            raise ValueError("Job refers to an unknown anchor")
        if not isinstance(job.get("replicate_id"), str) or not job["replicate_id"] or job.get("action") not in ACTIONS:
            raise ValueError("Job needs replicate_id and a declared action")
        key = (job["anchor_id"], job["replicate_id"], job["action"])
        if key in seen:
            raise ValueError(f"Duplicate paired branch: {key}")
        seen.add(key)
        status = job.get("status")
        if status not in TERMINALS or job.get("terminal_classification") != TERMINALS[status]:
            raise ValueError(f"Inconsistent terminal classification: {key}")
        if job.get("audit_certified") is not True:
            raise ValueError(f"Unaudited certified prefix: {key}")
        start = by_id[job["anchor_id"]]["cycle"]
        stop = _integer(job.get("last_certified_cycle"), "last_certified_cycle", minimum=start)
        series = job.get("certified_work_j_by_cycle")
        if not isinstance(series, list) or len(series) != stop - start:
            raise ValueError(f"Work series must cover every certified cycle: {key}")
        for index, row in enumerate(series, start + 1):
            if (not isinstance(row, dict) or row.get("cycle") != index
                    or isinstance(row.get("work_j"), bool)
                    or not isinstance(row.get("work_j"), (int, float))
                    or not math.isfinite(row["work_j"]) or row["work_j"] < 0):
                raise ValueError(f"Invalid certified cycle work at {index}: {key}")
    return anchors, jobs


def _work_to(job: dict, anchor_cycle: int, prefix_cycle: int) -> float:
    return math.fsum(row["work_j"] for row in job["certified_work_j_by_cycle"][:prefix_cycle - anchor_cycle])


def _spread(values: list[float]) -> float | None:
    return max(values) - min(values) if len(values) >= 2 else None


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _verified_path(base: Path, value: str, expected: str) -> Path:
    if not isinstance(value, str) or not value or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Missing artifact path or SHA-256")
    path = Path(value)
    path = (path if path.is_absolute() else base / path).resolve(strict=True)
    if sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Artifact digest mismatch: {path}")
    return path


def _verified_phase(path: Path, plan: dict, job: dict, side: str, phase: str,
                    manifest_digest: str) -> tuple[dict, list[dict]]:
    receipt = _read_json(path)
    anchor = job["anchor"]["cycle"]
    start = anchor if phase == "intervention" else anchor + 10
    torque = .96 + (job["delta_left_nm"] if side == "left" else -job["delta_left_nm"]) if phase == "intervention" else .96
    expected_work = 2 * math.pi * torque
    if (receipt.get("job_id") != job["id"] or receipt.get("side") != side or receipt.get("phase") != phase
            or receipt.get("manifest_sha256") != manifest_digest
            or receipt.get("source_receipt_sha256") != job["anchor"]["receipt_sha256"]
            or receipt.get("anchor_completed_cycles") != anchor or receipt.get("phase_start_cycle") != start
            or receipt.get("weights") != plan["weights"]
            or receipt.get("continuation_policy") != plan["continuation_policy"]
            or not math.isclose(receipt.get("torque_nm", math.nan), torque, abs_tol=1e-12)
            or not math.isclose(receipt.get("target_work_j", math.nan), expected_work, abs_tol=1e-10)
            or receipt.get("source_restore", {}).get("stimulation_history_complete") is not True):
        raise ValueError(f"Phase receipt differs from the preregistered problem: {path}")
    if phase == "intervention":
        if (receipt["source_restore"].get("restored_problem_sha256")
                != job["anchor"]["arms"][side]["prepared_problem_sha256"]
                or receipt.get("source_archive_sha256") != job["anchor"]["arms"][side]["archive_sha256"]):
            raise ValueError("Intervention did not restore the anchor problem")
    else:
        predecessor = _verified_path(path.parent, receipt.get("predecessor"), receipt.get("predecessor_sha256"))
        previous = _read_json(predecessor)
        if (previous.get("job_id") != job["id"] or previous.get("side") != side
                or previous.get("phase") != "intervention" or previous.get("phase_complete") is not True
                or previous.get("prepared_endpoint", {}).get("completed_cycles") != start
                or receipt["source_restore"].get("restored_problem_sha256")
                != previous["prepared_endpoint"].get("prepared_problem_sha256")
                or receipt.get("source_archive_sha256") != previous["prepared_endpoint"].get("sha256")
                or receipt.get("producer_pid") == previous.get("producer_pid")):
            raise ValueError("Continuation lacks its independently restored intervention endpoint")
        _verified_path(predecessor.parent, previous["prepared_endpoint"].get("primal_path"),
                       previous["prepared_endpoint"].get("sha256"))
    stop = receipt.get("last_certified_cycle")
    _integer(stop, "phase last_certified_cycle", minimum=start)
    rows = receipt.get("cycles")
    if not isinstance(rows, list) or len(rows) != stop - start:
        raise ValueError("Noncontiguous certified phase prefix")
    verified_rows = []
    for expected_cycle, row in enumerate(rows, start + 1):
        violation = row.get("maximum_normalized_violation") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or row.get("cycle") != expected_cycle or row.get("certified") is not True
                or isinstance(violation, bool) or not isinstance(violation, (int, float))
                or not math.isfinite(violation) or violation < 0 or violation > plan["audit_tolerance"]
                or row.get("independent_audit", {}).get("scope")
                != "complete_discrete_NLP_g_and_x; no continuous-ODE replay"
                or not math.isclose(row.get("torque_nm", math.nan), torque, abs_tol=1e-12)
                or not math.isclose(row.get("target_work_j", math.nan), expected_work, abs_tol=1e-10)):
            raise ValueError(f"Invalid independently audited cycle {expected_cycle}")
        _verified_path(path.parent, row.get("witness"), row.get("witness_sha256"))
        achieved = row.get("achieved_work_j")
        if isinstance(achieved, bool) or not isinstance(achieved, (int, float)) or not math.isfinite(achieved) or achieved < 0:
            raise ValueError(f"Invalid achieved work in cycle {expected_cycle}")
        if abs(achieved - expected_work) > plan["audit_tolerance"] * max(1., expected_work):
            raise ValueError(f"Achieved work differs from audited target at cycle {expected_cycle}")
        verified_rows.append(row)
    if phase == "intervention" and receipt.get("phase_complete") is True and stop != anchor + 10:
        raise ValueError("Completed intervention did not certify ten cycles")
    if receipt.get("status") == "physiological_failure_certified":
        terminal = receipt.get("terminal_attempt") or {}
        zero = terminal.get("zero_objective_feasibility_probe") or {}
        fatigue = (terminal.get("counterfactual_probes") or {}).get("fatigue_rest") or {}
        if (receipt.get("exact_label_allowed") is not True or terminal.get("cycle") != stop + 1
                or zero.get("available") is not True or zero.get("constraints_unchanged") is not True
                or zero.get("feasible_witness") is not False
                or (zero.get("solver_stats") or {}).get("return_status") != "Infeasible_Problem_Detected"
                or fatigue.get("feasible_witness") is not True
                or fatigue.get("physical_rho_advanced") is not False):
            raise ValueError("Exact phase lacks its frozen feasibility and fatigue-rest certificate")
    return receipt, verified_rows


def load_runner_campaign(path: Path) -> dict:
    """Authenticate the pilot runner's campaign receipts and normalize them."""
    path = Path(path).resolve(strict=True)
    base = path.parent
    completion = _read_json(path)
    manifest_path = base / "manifest.json"
    seal = _read_json(base / "manifest-seal.json")
    manifest_digest = sha256(manifest_path.read_bytes()).hexdigest()
    if seal.get("manifest_sha256") != manifest_digest:
        raise ValueError("Pilot manifest seal mismatch")
    plan = _read_json(manifest_path)
    if plan.get("schema_version") != 1 or plan.get("kind") != "paired_bilateral_causal_torque_pilot":
        raise ValueError("Unsupported pilot runner manifest")
    source_jobs = {job["id"]: job for job in plan["jobs"]}
    if len(source_jobs) != len(plan["jobs"]):
        raise ValueError("Duplicate preregistered job ids")
    anchors = {}
    for job in plan["jobs"]:
        anchor = job["anchor"]
        _verified_path(base, anchor["receipt"], anchor["receipt_sha256"])
        anchors[f"c{anchor['cycle']}"] = {"id": f"c{anchor['cycle']}", "cycle": anchor["cycle"]}
    normalized = []
    seen = set()
    for result in completion.get("jobs", []):
        job_id = result.get("job_id")
        if job_id not in source_jobs or job_id in seen:
            raise ValueError("Unknown or duplicated completed job")
        seen.add(job_id)
        job = source_jobs[job_id]
        anchor_cycle = job["anchor"]["cycle"]
        if result.get("anchor_cycle") != anchor_cycle or result.get("delta_left_nm") != job["delta_left_nm"]:
            raise ValueError("Campaign job differs from preregistration")
        side_rows: dict[str, dict[int, dict]] = {}
        side_stops = []
        for side in ("left", "right"):
            arm = result.get("arms", {}).get(side, {})
            rows = {}
            receipts = []
            for phase in ("intervention", "continuation"):
                receipt_path = base / job_id / side / phase / "result.json"
                if not receipt_path.is_file():
                    continue
                receipt, verified_rows = _verified_phase(receipt_path, plan, job, side, phase, manifest_digest)
                receipts.append((phase, receipt_path, receipt))
                rows.update({row["cycle"]: row for row in verified_rows})
            if receipts:
                latest_phase, latest_path, latest = receipts[-1]
                if arm.get("result") is None:
                    if (arm.get("status") != "numerical_stop" or arm.get("failed_phase") != "continuation"
                            or latest_phase != "intervention"):
                        raise ValueError("Missing final arm receipt without documented failed continuation")
                elif (Path(arm["result"]).resolve() != latest_path.resolve()
                      or arm.get("result_sha256") != sha256(latest_path.read_bytes()).hexdigest()
                      or arm.get("status") != latest.get("status")
                      or arm.get("last_certified_cycle") != latest.get("last_certified_cycle")):
                    raise ValueError("Campaign arm summary differs from phase receipt")
                if latest_phase == "continuation" and (len(receipts) != 2 or receipts[0][2].get("phase_complete") is not True):
                    raise ValueError("Continuation without a complete intervention")
                side_stops.append(latest["last_certified_cycle"])
            else:
                if arm.get("status") != "numerical_stop":
                    raise ValueError("Missing phase receipts for non-numerical arm")
                side_stops.append(anchor_cycle)
            side_rows[side] = rows
        common = min(side_stops)
        if result.get("last_certified_cycle") is not None and result["last_certified_cycle"] != common:
            raise ValueError("Bilateral certified prefix differs from arm receipts")
        status = {"physiological_failure_certified": "exact", "right_censored": "censored",
                  "numerical_stop": "numerical"}.get(result.get("status"))
        if status is None:
            raise ValueError("Unknown bilateral terminal status")
        if status == "exact" and (result.get("exact_label_allowed") is not True or
                                  not any(arm.get("status") == "physiological_failure_certified" and
                                          arm.get("last_certified_cycle") == common
                                          for arm in result.get("arms", {}).values())):
            raise ValueError("Exact bilateral endpoint lacks a limiting frozen feasibility certificate")
        if status == "censored" and (any(arm.get("status") != "right_censored"
                                         for arm in result.get("arms", {}).values()) or
                                     common != plan["maximum_absolute_cycles"]):
            raise ValueError("Censored bilateral endpoint differs from horizon")
        bilateral_work = []
        for current in range(anchor_cycle + 1, common + 1):
            if current not in side_rows["left"] or current not in side_rows["right"]:
                raise ValueError("Missing audited bilateral cycle in certified prefix")
            bilateral_work.append({"cycle": current,
                                   "work_j": side_rows["left"][current]["achieved_work_j"]
                                   + side_rows["right"][current]["achieved_work_j"]})
        delta = job["delta_left_nm"]
        action = "hold" if delta == 0 else "plus" if delta > 0 else "minus"
        normalized.append({"job_id": job_id, "anchor_id": f"c{anchor_cycle}", "replicate_id": job_id,
                           "action": action, "status": status,
                           "terminal_classification": TERMINALS[status], "audit_certified": True,
                           "last_certified_cycle": common, "certified_work_j_by_cycle": bilateral_work})
    return {"schema_version": 1, "kind": "torque_transfer_t3_pilot", "intervention_cycles": 10,
            "anchors": list(anchors.values()), "jobs": normalized,
            "runner_manifest_sha256": manifest_digest}


def analyze_pilot(manifest: dict) -> dict:
    """Return conservative T3 comparisons; never turn censoring into failure."""
    anchors, jobs = _validated(manifest)
    reports = []
    for anchor in anchors:
        anchor_jobs = [job for job in jobs if job["anchor_id"] == anchor["id"]]
        if not anchor_jobs:
            reports.append({"anchor_id": anchor["id"], "gate": "inconclusive", "reason": "no_jobs"})
            continue
        cycle = anchor["cycle"]
        common = min(job["last_certified_cycle"] for job in anchor_jobs)
        controls = [job for job in anchor_jobs if job["action"] == "hold" and job["status"] == "exact"]
        endpoint_resolution = _spread([job["last_certified_cycle"] for job in controls])
        work_resolution = _spread([_work_to(job, cycle, common) for job in controls])
        grouped: dict[str, list[dict]] = {action: [] for action in ACTIONS}
        branch_rows = []
        for job in anchor_jobs:
            grouped[job["action"]].append(job)
            branch_rows.append({"job_id": job.get("job_id", f"{job['action']}:{job['replicate_id']}"),
                                "replicate_id": job["replicate_id"], "action": job["action"],
                                "status": job["status"], "last_certified_cycle": job["last_certified_cycle"],
                                "work_at_common_prefix_j": _work_to(job, cycle, common),
                                "work_through_certified_prefix_j": _work_to(job, cycle, job["last_certified_cycle"])})
        branch_rows.sort(key=lambda row: (row["replicate_id"], ACTIONS.index(row["action"])))
        pairs = []
        regrets = []
        for action in ("minus", "plus"):
            for branch, hold in product(grouped[action], grouped["hold"]):
                exact = (hold["status"] == branch["status"] == "exact"
                         and min(hold["last_certified_cycle"], branch["last_certified_cycle"]) >= cycle + 10)
                delta = branch["last_certified_cycle"] - hold["last_certified_cycle"] if exact else None
                pairs.append({"action_job_id": branch.get("job_id", f"{action}:{branch['replicate_id']}"),
                              "control_job_id": hold.get("job_id", f"hold:{hold['replicate_id']}"),
                              "action": action, "status": "exact" if exact else "incomplete_intervention_or_non_exact",
                              "component_statuses": [hold["status"], branch["status"]], "delta_cycles": delta,
                              "work_delta_at_common_prefix_j": _work_to(branch, cycle, common) - _work_to(hold, cycle, common),
                              "resolves_control_variation": bool(endpoint_resolution is not None and delta is not None
                                                                 and abs(delta) > endpoint_resolution)})
        full = all(grouped.values()) and all(
            job["status"] == "exact" and job["last_certified_cycle"] >= cycle + 10 for job in anchor_jobs)
        advantages = {}
        for action in ("minus", "plus"):
            rows = [row for row in pairs if row["action"] == action]
            advantages[action] = bool(full and endpoint_resolution is not None and rows
                                      and all(row["delta_cycles"] > endpoint_resolution for row in rows))
        direction = None
        if full and endpoint_resolution is not None:
            direct = [plus["last_certified_cycle"] - minus["last_certified_cycle"]
                      for plus, minus in product(grouped["plus"], grouped["minus"])]
            if advantages["plus"] and all(value > endpoint_resolution for value in direct):
                direction = "plus"
            elif advantages["minus"] and all(value < -endpoint_resolution for value in direct):
                direction = "minus"
        chosen = anchor.get("chosen_action")
        if chosen is not None:
            if full:
                for minus, hold, plus in product(grouped["minus"], grouped["hold"], grouped["plus"]):
                    combination = {"minus": minus, "hold": hold, "plus": plus}
                    best = max(job["last_certified_cycle"] for job in combination.values())
                    regret = best - combination[chosen]["last_certified_cycle"]
                    regrets.append({"job_ids": {action: job.get("job_id", f"{action}:{job['replicate_id']}")
                                                 for action, job in combination.items()},
                                    "status": "exact", "chosen_action": chosen, "regret_cycles": regret,
                                    "resolves_control_variation": bool(endpoint_resolution is not None and regret > endpoint_resolution)})
            else:
                regrets.append({"status": "unavailable_non_exact_or_missing_action", "regret_cycles": None})
        reason = ("insufficient_exact_repeated_controls" if endpoint_resolution is None else
                  "missing_or_non_exact_branch" if not full else
                  "contrast_within_control_resolution_or_inconsistent" if direction is None else
                  "direction_resolved_in_all_replicates")
        reports.append({"anchor_id": anchor["id"], "anchor_cycle": cycle,
                        "common_prefix_cycle": common, "certified_cycles_at_common_prefix": common - cycle,
                        "control_repeats_exact": len(controls),
                        "control_resolution_cycles": endpoint_resolution,
                        "control_resolution_work_j_at_common_prefix": work_resolution,
                        "branches": branch_rows, "paired_advantages": pairs,
                        "beneficial_actions_resolved": advantages, "chosen_action_regret": regrets,
                        "chosen_action_regret_status": "not_predeclared" if chosen is None else
                        "exact" if full else "unavailable_non_exact_or_missing_action",
                        "direction": direction, "gate": "resolved" if direction else "inconclusive",
                        "reason": reason})
    return {"schema_version": 1, "kind": "torque_transfer_t3_analysis",
            "source_kind": manifest["kind"], "anchors": reports,
            "activation_allowed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Normalized audited pilot JSON")
    parser.add_argument("--output", type=Path, help="Write report JSON (stdout by default)")
    args = parser.parse_args()
    source = _read_json(args.manifest)
    normalized = load_runner_campaign(args.manifest) if source.get("kind") != "torque_transfer_t3_pilot" else source
    report = analyze_pilot(normalized)
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
