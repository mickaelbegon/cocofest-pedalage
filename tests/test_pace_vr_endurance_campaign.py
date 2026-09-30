import json
from pathlib import Path

import pytest

from scripts.run_pace_vr_endurance_campaign import (
    CONDITIONS,
    REFERENCE,
    _complete_state,
    analyze,
    prepare,
    run,
)


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _summary(*, cycles=3, successful=False):
    return {
        "completed_rho_cycles": cycles,
        "success": successful,
        "failure": None if successful else f"At least one arm failed certification after cycle {cycles}",
        "arms": {
            "right": {"validated_cycles": cycles, "cycles": [
                {"cycle": index, "certified": True, "solver_time_s": float(index),
                 "capacity_ratios": {"Biceps": .9}, "weights_used": [1, 1, 1, 1]}
                for index in range(1, cycles + 1)]},
            "left": {"validated_cycles": cycles if successful else cycles - 1, "cycles": [
                {"cycle": index, "certified": successful or index < cycles,
                 "solver_time_s": float(index), "capacity_ratios": {"Biceps": .8},
                 "weights_used": [1, 1, 1, 1]}
                for index in range(1, cycles + 1)]},
        },
    }


def _vr_template(tmp_path):
    path = tmp_path / "pace-vr-template.json"
    payload = json.loads((REFERENCE / "unit.json").read_text())
    payload["experimental_pace_vr"] = {
        "enabled": True, "asynchronous": True,
        "application_mode": "derived_fatigue_weights_experimental",
        "snapshot_at_block_boundary": True,
        "retain_last_valid_value": True,
        "deadline_seconds": 10, "horizon_cycles": 100,
        "update_every_cycles": 20,
    }
    _write(path, payload)
    return path


def test_prepared_three_condition_manifest_has_matched_science_and_distinct_cores(tmp_path):
    path = prepare(
        tmp_path / "campaign",
        {"unit": REFERENCE / "unit.json", "pace": REFERENCE / "pace.json",
         "pace_vr": _vr_template(tmp_path)},
        {"unit": [12, 13], "pace": [14, 15], "pace_vr": [16, 17]},
        max_cycles=300, python=Path(__import__("sys").executable), pace_vr_cpu_ids=[18, 19],
    )
    manifest = json.loads(path.read_text())
    assert set(manifest["conditions"]) == set(CONDITIONS)
    assert manifest["protocol"]["collocation"] == "Radau5"
    assert manifest["protocol"]["nlp_solver"] == "IPOPT/MA57"
    assert manifest["protocol"]["pace_vr_supervisor_cpu_ids"] == [18, 19]
    assert manifest["protocol"]["pace_vr_horizon_cycles"] == 20
    assert manifest["protocol"]["pace_vr_horizon_ladder_cycles"] == [20]
    for name in CONDITIONS:
        config = json.loads(Path(manifest["conditions"][name]["config"]).read_text())
        assert config["cycles"] == 300
        assert config["solver_cpu_affinity"] == manifest["conditions"][name]["cpu_affinity"]
    assert json.loads(Path(manifest["conditions"]["pace_vr"]["config"]).read_text())[
        "experimental_pace_vr"]["supervisor_cpu_ids"] == [18, 19]
    actual_template = REFERENCE.parent / "pace-vr-async-h100-k20-template.json"
    actual = prepare(tmp_path / "actual-template",
                     {"unit": REFERENCE / "unit.json", "pace": REFERENCE / "pace.json",
                      "pace_vr": actual_template},
                     {"unit": [12, 13], "pace": [14, 15], "pace_vr": [16, 17]},
                     max_cycles=300, python=Path(__import__("sys").executable), pace_vr_cpu_ids=[18, 19])
    assert actual.is_file()
    with pytest.raises(ValueError, match="distinct"):
        prepare(tmp_path / "bad", {"unit": REFERENCE / "unit.json", "pace": REFERENCE / "pace.json",
                                       "pace_vr": _vr_template(tmp_path)},
                {"unit": [12, 13], "pace": [13, 14], "pace_vr": [16, 17]},
                max_cycles=300, python=Path(__import__("sys").executable), pace_vr_cpu_ids=[18, 19])
    with pytest.raises(ValueError, match="supervisor CPU"):
        prepare(tmp_path / "bad-supervisor", {"unit": REFERENCE / "unit.json", "pace": REFERENCE / "pace.json",
                                                  "pace_vr": _vr_template(tmp_path)},
                {"unit": [12, 13], "pace": [14, 15], "pace_vr": [16, 17]},
                max_cycles=300, python=Path(__import__("sys").executable), pace_vr_cpu_ids=[17, 19])
    h50 = prepare(tmp_path / "h50", {"unit": REFERENCE / "unit.json", "pace": REFERENCE / "pace.json",
                                      "pace_vr": _vr_template(tmp_path)},
                  {"unit": [12, 13], "pace": [14, 15], "pace_vr": [16, 17]},
                  max_cycles=300, python=Path(__import__("sys").executable), pace_vr_cpu_ids=[18, 19],
                  pace_vr_horizon=50)
    h50_config = json.loads(Path(json.loads(h50.read_text())["conditions"]["pace_vr"]["config"]).read_text())
    assert h50_config["experimental_pace_vr"]["horizon_cycles"] == 50

    ladder = prepare(tmp_path / "ladder", {"unit": REFERENCE / "unit.json", "pace": REFERENCE / "pace.json",
                                             "pace_vr": _vr_template(tmp_path)},
                     {"unit": [12, 13], "pace": [14, 15], "pace_vr": [16, 17]},
                     max_cycles=300, python=Path(__import__("sys").executable),
                     pace_vr_cpu_ids=list(range(18, 28)), pace_vr_horizons=(100, 50, 25, 20, 5))
    ladder_config = json.loads(Path(json.loads(ladder.read_text())["conditions"]["pace_vr"]["config"]).read_text())
    assert ladder_config["experimental_pace_vr"]["horizon_ladder_cycles"] == [100, 50, 25, 20, 5]


def test_failure_is_not_called_physiological_or_restarted_without_separate_review(tmp_path):
    output = tmp_path / "result"
    _write(output / "summary.json", _summary())
    assert _complete_state(output, 300) == "stopped_unreviewed"
    evidence = tmp_path / "feasibility-certificate.json"
    _write(evidence, {"status": "reviewed"})
    _write(output / "physiological_review.json", {
        "verdict": "physiological_infeasibility", "completed_rho_cycles": 3,
        "evidence_path": str(evidence),
    })
    assert _complete_state(output, 300) == "reviewed_physiological_failure"
    (output / "summary.json").unlink()
    assert _complete_state(output, 300) == "partial_unreviewed"


def test_analysis_counts_only_certified_pairs_and_reports_timings(tmp_path):
    output = tmp_path / "unit"
    _write(output / "summary.json", _summary())
    manifest = tmp_path / "manifest.json"
    _write(manifest, {"campaign": "test", "requested_cycles": 300,
                      "conditions": {name: {"output": str(output if name == "unit" else tmp_path / name)}
                                     for name in CONDITIONS}})
    report = analyze(manifest)
    unit = report["conditions"]["unit"]
    assert unit["state"] == "stopped_unreviewed"
    assert unit["certified_pair_cycles"] == 2
    assert unit["right"]["solver_timing"]["mean_s"] == pytest.approx(2)
    assert unit["left"]["solver_timing"]["p95_s"] == pytest.approx(2)
    assert report["conditions"]["pace"]["state"] == "pending"


def test_async_supervisor_receipts_report_runtime_lag_and_stale_decisions(tmp_path):
    output = tmp_path / "pace-vr"
    summary = _summary(cycles=3, successful=True)
    summary["arms"]["right"]["pace_vr_events"] = [
        {"request_id": "r1", "source_cycle": 1, "applied_cycle": 2,
         "runtime_s": 3.0, "status": "applied", "deadline_exceeded": False},
        {"request_id": "r2", "source_cycle": 2, "runtime_s": 12.0,
         "status": "stale_discarded", "stale": True, "deadline_exceeded": True},
    ]
    summary["preparations"] = [
        {"completed_cycles": 1, "arms": {"right": {"prepare_time_s": .01}}},
        {"completed_cycles": 20, "arms": {"right": {"prepare_time_s": .02}}},
    ]
    _write(output / "summary.json", summary)
    manifest = tmp_path / "manifest.json"
    _write(manifest, {"campaign": "test", "requested_cycles": 3,
                      "conditions": {name: {"output": str(output if name == "pace_vr" else tmp_path / name)}
                                     for name in CONDITIONS}})
    row = analyze(manifest)["conditions"]["pace_vr"]["right"]
    assert row["pace_vr_supervisor"]["supervisor_runtime_s"]["mean_s"] == pytest.approx(7.5)
    assert row["pace_vr_supervisor"]["application_lag_cycles"]["mean_cycles"] == 1
    assert row["pace_vr_supervisor"]["stale_count"] == 1
    assert row["pace_vr_supervisor"]["deadline_miss_count"] == 1
    assert row["prepare_at_slow_update_s"]["mean_s"] == pytest.approx(.02)
    assert row["pace_vr_candidate_screen"]["audit_available"] is False


def test_analysis_reports_candidate_screen_by_horizon_without_double_counting_selected_arm(tmp_path):
    output = tmp_path / "pace-vr"
    summary = _summary(cycles=3, successful=True)
    summary["pace_vr_supervisor"] = [{
        "status": "proposed", "applied_cycle": 3, "applied_arms": ["right"],
        "selected_horizon_cycles": {"right": 20},
        "arms": {"right": {"candidate_screen": {"chosen": "half_step"}}},
        "attempts": {
            "50": {"right": {"source_cycle": 1, "status": "complete", "runtime_s": 2.0,
                             "candidate_screen": {"chosen": "incumbent", "predicted_improvement": 0.0,
                                                  "reason": "no_feasible_candidate_exceeds_improvement_threshold"}}},
            "20": {"right": {"source_cycle": 1, "status": "complete", "runtime_s": 1.0,
                             "candidate_screen": {"chosen": "half_step", "predicted_improvement": .25}}},
        },
    }]
    _write(output / "summary.json", summary)
    manifest = tmp_path / "manifest.json"
    _write(manifest, {"campaign": "test", "requested_cycles": 3,
                      "conditions": {name: {"output": str(output if name == "pace_vr" else tmp_path / name)}
                                     for name in CONDITIONS}})
    report = analyze(manifest)["conditions"]["pace_vr"]
    right = report["right"]["pace_vr_candidate_screen"]
    assert right["screened_attempt_count"] == 2
    assert right["proposal_count"] == 1
    assert right["incumbent_retained_count"] == 1
    assert right["selected_candidate_counts"] == {"half_step": 1}
    assert right["selected_horizon_counts"] == {"20": 1}
    assert right["predicted_improvement"]["mean_score"] == pytest.approx(.25)
    assert right["rollout_runtime_s"]["mean_s"] == pytest.approx(1.5)
    assert right["attempts"][1]["selected_by_supervisor"] is True
    assert right["attempts"][1]["command_issued"] is True
    assert report["left"]["pace_vr_candidate_screen"]["audit_available"] is False


def test_resume_rejects_partial_artifacts_before_launching_any_condition(tmp_path):
    path = prepare(tmp_path / "campaign",
                   {"unit": REFERENCE / "unit.json", "pace": REFERENCE / "pace.json",
                    "pace_vr": _vr_template(tmp_path)},
                   {"unit": [12, 13], "pace": [14, 15], "pace_vr": [16, 17]},
                   max_cycles=300, python=Path(__import__("sys").executable), pace_vr_cpu_ids=[18, 19])
    manifest = json.loads(path.read_text())
    Path(manifest["conditions"]["unit"]["output"]).mkdir(parents=True)
    (Path(manifest["conditions"]["unit"]["output"]) / "weights.jsonl").touch()
    with pytest.raises(ValueError, match="Partial or unreviewed"):
        run(path)
    assert not (path.parent / "logs").exists()


def test_legacy_async_key_is_rejected_instead_of_silently_ignored(tmp_path):
    vr_template = _vr_template(tmp_path)
    payload = json.loads(vr_template.read_text())
    payload["experimental_pace_vr"]["async"] = True
    _write(vr_template, payload)
    with pytest.raises(ValueError, match="asynchronous snapshot"):
        prepare(tmp_path / "campaign",
                {"unit": REFERENCE / "unit.json", "pace": REFERENCE / "pace.json",
                 "pace_vr": vr_template},
                {"unit": [12, 13], "pace": [14, 15], "pace_vr": [16, 17]},
                max_cycles=300, python=Path(__import__("sys").executable), pace_vr_cpu_ids=[18, 19])
