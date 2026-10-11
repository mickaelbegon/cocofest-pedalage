"""Synthetic T3 pilot contrasts; no RHO simulation is run by these tests."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.analyze_torque_transfer_pilot import analyze_pilot, load_runner_campaign


def _job(replicate: str, action: str, last: int, *, work: float = 10.0,
         status: str = "exact") -> dict:
    terminal = {"exact": "frozen_feasibility_exhaustion",
                "censored": "administrative_horizon",
                "numerical": "numerical_unresolved"}[status]
    return {"anchor_id": "early", "replicate_id": replicate, "action": action,
            "status": status, "terminal_classification": terminal,
            "audit_certified": True, "last_certified_cycle": last,
            "certified_work_j_by_cycle": [{"cycle": cycle, "work_j": work}
                                          for cycle in range(101, last + 1)]}


def _manifest() -> dict:
    return {"schema_version": 1, "kind": "torque_transfer_t3_pilot",
            "intervention_cycles": 10,
            "anchors": [{"id": "early", "cycle": 100, "chosen_action": "hold"}],
            "jobs": [_job("r1", "hold", 120), _job("r1", "minus", 118, work=9),
                     _job("r1", "plus", 124, work=11),
                     _job("r2", "hold", 121), _job("r2", "minus", 119, work=9),
                     _job("r2", "plus", 125, work=11)]}


def test_resolved_direction_uses_control_repeat_resolution_and_common_prefix():
    result = analyze_pilot(_manifest())
    anchor = result["anchors"][0]
    assert anchor["control_resolution_cycles"] == 1
    assert anchor["common_prefix_cycle"] == 118
    assert anchor["control_resolution_work_j_at_common_prefix"] == 0
    assert anchor["gate"] == "resolved" and anchor["direction"] == "plus"
    assert anchor["beneficial_actions_resolved"] == {"minus": False, "plus": True}
    plus = next(row for row in anchor["paired_advantages"] if row["action_job_id"] == "plus:r1"
                and row["control_job_id"] == "hold:r1")
    assert plus["delta_cycles"] == 4
    assert plus["work_delta_at_common_prefix_j"] == 18
    assert {row["regret_cycles"] for row in anchor["chosen_action_regret"]} == {3, 4, 5}
    assert result["activation_allowed"] is False


def test_contrast_equal_to_control_spread_is_inconclusive():
    manifest = _manifest()
    for job in manifest["jobs"]:
        if job["action"] == "plus":
            job["last_certified_cycle"] -= 3
            job["certified_work_j_by_cycle"] = job["certified_work_j_by_cycle"][:-3]
    anchor = analyze_pilot(manifest)["anchors"][0]
    assert anchor["control_resolution_cycles"] == 1
    assert anchor["direction"] is None
    assert anchor["gate"] == "inconclusive"
    assert not anchor["beneficial_actions_resolved"]["plus"]


def test_runner_shape_two_controls_one_of_each_treatment_does_not_invent_repeats():
    manifest = _manifest()
    manifest["anchors"][0].pop("chosen_action")
    manifest["jobs"] = [row for row in manifest["jobs"] if row["action"] == "hold" or row["replicate_id"] == "r1"]
    anchor = analyze_pilot(manifest)["anchors"][0]
    assert anchor["gate"] == "resolved" and anchor["direction"] == "plus"
    assert len(anchor["paired_advantages"]) == 4
    assert anchor["chosen_action_regret"] == []
    assert anchor["chosen_action_regret_status"] == "not_predeclared"


@pytest.mark.parametrize("status", ["censored", "numerical"])
def test_non_exact_branch_cannot_establish_direction_or_regret(status):
    manifest = _manifest()
    job = next(row for row in manifest["jobs"] if row["replicate_id"] == "r2" and row["action"] == "plus")
    job["status"] = status
    job["terminal_classification"] = {"censored": "administrative_horizon",
                                      "numerical": "numerical_unresolved"}[status]
    anchor = analyze_pilot(manifest)["anchors"][0]
    assert anchor["direction"] is None
    assert anchor["gate"] == "inconclusive"
    assert any(row["delta_cycles"] is None for row in anchor["paired_advantages"])
    assert anchor["chosen_action_regret"][0]["regret_cycles"] is None
    assert anchor["common_prefix_cycle"] == 118


def test_one_control_and_partial_intervention_are_inconclusive():
    manifest = _manifest()
    manifest["jobs"] = [row for row in manifest["jobs"] if row["replicate_id"] == "r1"]
    anchor = analyze_pilot(manifest)["anchors"][0]
    assert anchor["control_resolution_cycles"] is None
    assert anchor["gate"] == "inconclusive"
    manifest = _manifest()
    row = next(job for job in manifest["jobs"] if job["replicate_id"] == "r1" and job["action"] == "minus")
    row["last_certified_cycle"] = 109
    row["certified_work_j_by_cycle"] = row["certified_work_j_by_cycle"][:9]
    anchor = analyze_pilot(manifest)["anchors"][0]
    assert anchor["direction"] is None
    assert anchor["chosen_action_regret"][0]["regret_cycles"] is None


@pytest.mark.parametrize("change", [
    lambda m: m["jobs"][0].update(audit_certified=False),
    lambda m: m["jobs"][0].update(terminal_classification="solver_failed"),
    lambda m: m["jobs"][0]["certified_work_j_by_cycle"].pop(),
    lambda m: m["jobs"][0]["certified_work_j_by_cycle"][0].update(cycle=102),
    lambda m: m["jobs"].append(deepcopy(m["jobs"][0])),
])
def test_invalid_or_unaudited_inputs_are_rejected(change):
    manifest = _manifest()
    change(manifest)
    with pytest.raises(ValueError):
        analyze_pilot(manifest)


def test_cli_writes_analysis_report(tmp_path: Path):
    source = tmp_path / "pilot.json"
    output = tmp_path / "analysis.json"
    source.write_text(json.dumps(_manifest()), encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "analyze_torque_transfer_pilot.py"
    subprocess.run([sys.executable, str(script), str(source), "--output", str(output)],
                   check=True, capture_output=True, text=True)
    assert json.loads(output.read_text(encoding="utf-8"))["anchors"][0]["direction"] == "plus"


def _save(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return sha256(path.read_bytes()).hexdigest()


def _runner_campaign(tmp_path: Path) -> Path:
    anchor_path = tmp_path / "anchor.json"
    anchor_sha = _save(anchor_path, {"certified": True})
    source_shas = {}
    for side in ("left", "right"):
        source_shas[side] = _save(tmp_path / f"{side}-source.json", {"side": side, "source": True})
    anchor = {"cycle": 100, "receipt": str(anchor_path), "receipt_sha256": anchor_sha,
              "arms": {side: {"prepared_problem_sha256": "a" * 64, "archive_sha256": source_shas[side]}
                       for side in ("left", "right")}}
    job = {"id": "c100-control-1", "anchor": anchor, "delta_left_nm": 0., "is_control": True,
           "intervention_cycles": 10}
    policy = {"kind": "fixed_equal_split"}
    plan = {"schema_version": 1, "kind": "paired_bilateral_causal_torque_pilot",
            "jobs": [job], "weights": {"m": 1.}, "continuation_policy": policy,
            "audit_tolerance": 1e-5, "maximum_absolute_cycles": 240}
    manifest = tmp_path / "manifest.json"
    manifest_sha = _save(manifest, plan)
    _save(tmp_path / "manifest-seal.json", {"manifest_sha256": manifest_sha})
    work = 2 * math.pi * .96
    arms = {}
    for side in ("left", "right"):
        witness = tmp_path / f"{side}-witness.json"
        witness_sha = _save(witness, {"audit": True})
        endpoint_path = tmp_path / f"{side}-endpoint.json"
        endpoint_sha = _save(endpoint_path, {"side": side, "endpoint": True})
        phase_files = {}
        for phase, start, stop in (("intervention", 100, 110), ("continuation", 110, 112)):
            path = tmp_path / job["id"] / side / phase / "result.json"
            rows = [{"cycle": cycle, "certified": True, "maximum_normalized_violation": 0.,
                     "independent_audit": {"scope": "complete_discrete_NLP_g_and_x; no continuous-ODE replay"},
                     "torque_nm": .96, "target_work_j": work, "achieved_work_j": work,
                     "witness": str(witness), "witness_sha256": witness_sha}
                    for cycle in range(start + 1, stop + 1)]
            receipt = {"job_id": job["id"], "side": side, "phase": phase,
                       "manifest_sha256": manifest_sha, "source_receipt_sha256": anchor_sha,
                       "anchor_completed_cycles": 100, "phase_start_cycle": start,
                       "weights": plan["weights"], "continuation_policy": policy,
                       "torque_nm": .96, "target_work_j": work,
                       "source_archive_sha256": source_shas[side] if phase == "intervention" else endpoint_sha,
                       "source_restore": {"stimulation_history_complete": True,
                                          "restored_problem_sha256": "a" * 64 if phase == "intervention" else "b" * 64},
                       "producer_pid": 1 if phase == "intervention" else 2,
                       "cycles": rows, "last_certified_cycle": stop,
                       "status": "intervention_complete" if phase == "intervention" else "physiological_failure_certified",
                       "phase_complete": phase == "intervention", "exact_label_allowed": phase == "continuation",
                       "prepared_endpoint": {"completed_cycles": 110, "prepared_problem_sha256": "b" * 64,
                                             "primal_path": str(endpoint_path), "sha256": endpoint_sha}
                       if phase == "intervention" else None}
            if phase == "continuation":
                receipt.update(predecessor=str(phase_files["intervention"]),
                               predecessor_sha256=sha256(phase_files["intervention"].read_bytes()).hexdigest(),
                               terminal_attempt={"cycle": 113,
                                   "zero_objective_feasibility_probe": {"available": True,
                                       "constraints_unchanged": True, "feasible_witness": False,
                                       "solver_stats": {"return_status": "Infeasible_Problem_Detected"}},
                                   "counterfactual_probes": {"fatigue_rest": {"feasible_witness": True,
                                       "physical_rho_advanced": False}}})
            digest = _save(path, receipt)
            phase_files[phase] = path
        arms[side] = {"status": "physiological_failure_certified", "last_certified_cycle": 112,
                      "result": str(phase_files["continuation"]), "result_sha256": digest}
    _save(tmp_path / "campaign-completion.json", {"completed": True,
        "jobs": [{"job_id": job["id"], "anchor_cycle": 100, "delta_left_nm": 0.,
                  "arms": arms, "status": "physiological_failure_certified",
                  "last_certified_cycle": 112, "exact_label_allowed": True}]})
    return tmp_path / "campaign-completion.json"


def test_runner_campaign_adapter_checks_receipts_and_bilateral_work(tmp_path: Path):
    campaign = _runner_campaign(tmp_path)
    normalized = load_runner_campaign(campaign)
    assert normalized["jobs"][0]["status"] == "exact"
    assert normalized["jobs"][0]["last_certified_cycle"] == 112
    assert len(normalized["jobs"][0]["certified_work_j_by_cycle"]) == 12
    assert normalized["jobs"][0]["certified_work_j_by_cycle"][0]["work_j"] == pytest.approx(2 * 2 * math.pi * .96)
    assert analyze_pilot(normalized)["anchors"][0]["gate"] == "inconclusive"
    (tmp_path / "left-witness.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_runner_campaign(campaign)
