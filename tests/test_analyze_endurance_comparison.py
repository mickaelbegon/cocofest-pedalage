"""Synthetic contract tests for the conservative cross-campaign reader."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_endurance_comparison import build_report, collect_source, render_markdown


def _cycle(cycle: int, *, certified: bool = True, work: float = 5.0) -> dict:
    return {"cycle": cycle, "pair_solve_and_transfer_wall_time_s": 1.0 + cycle,
            "arms": {side: {"certified": certified, "achieved_work_j_per_cycle": work,
                             "solver_time_s": .3} for side in ("left", "right")}}


def test_summary_is_a_certified_prefix_not_an_inferred_endpoint(tmp_path: Path):
    source = tmp_path / "summary.json"
    source.write_text(json.dumps({"success": False, "cycles": [_cycle(1), _cycle(2), _cycle(3, certified=False)],
                                  "pair_cycle_wall_timing_s": {"count": 2, "median": 1.5}}))
    record = collect_source("unit", source)[0]
    assert record["last_shared_certified_cycle"] == 2
    assert record["endpoint"]["certified"] is False
    assert record["bilateral_work_j"] == 20.0
    assert record["rho_timing"]["median"] == 1.5
    assert "1.50" in render_markdown({"records": [record]})


def test_live_continuous_progress_keeps_endpoint_unavailable_and_reports_trigger(tmp_path: Path):
    root = tmp_path / "continuous"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"kind": "continuous_causal_left_to_right_transfer_rho_benchmark",
                                                      "fast_state_status": "unvalidated_experimental_override",
                                                      "payload": {"muscle_pace": {"adaptation_enabled": False}}}))
    (root / "progress.json").write_text(json.dumps({"cycles": [_cycle(1), _cycle(2)],
                                                      "decisions": [{"phase": "hold"}, {"completed_cycles": 2, "phase": "transfer", "left_torque_nm": .86, "right_torque_nm": 1.06}]}))
    record = collect_source("continuous", root)[0]
    assert record["state"] == "in_progress"
    assert record["endpoint"]["classification"] == "not_available"
    assert record["trigger_events"][0]["completed_cycles"] == 2
    assert record["fast_state_status"] == "unvalidated_experimental_override"


def test_missing_source_and_checkpoint_branch_remain_explicit(tmp_path: Path):
    missing = collect_source("missing", tmp_path / "none")[0]
    assert missing["state"] == "missing"
    completion = tmp_path / "campaign-completion.json"
    completion.write_text(json.dumps({"completed": True, "jobs": [{"job_id": "c140-causal", "anchor_cycle": 140,
        "delta_left_nm": -.1, "status": "physiological_failure_certified", "exact_label_allowed": True,
        "last_certified_cycle": 174, "limiting_sides": ["left"], "arms": {}}]}))
    record = collect_source("pilot", completion)[0]
    assert record["scope"] == "checkpoint_continuation"
    assert record["endpoint"] == {"classification": "frozen_feasibility_exhaustion", "certified": True,
                                  "cycle": 174, "limiting_sides": ["left"]}
    report = build_report({"missing": tmp_path / "none", "pilot": completion})
    assert "Reprises depuis checkpoint" in render_markdown(report)


def test_completion_needing_exact_label_does_not_be_upgraded(tmp_path: Path):
    root = tmp_path / "continuous"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"kind": "continuous_causal_left_to_right_transfer_rho_benchmark", "payload": {}}))
    (root / "campaign-completion.json").write_text(json.dumps({"status": "numerical_stop", "exact_label_allowed": False,
        "last_certified_cycle": 12, "cycles": [_cycle(1)], "terminal": {"arms": {}}}))
    record = collect_source("continuous", root)[0]
    assert record["endpoint"]["certified"] is False
    assert record["endpoint"]["cycle"] is None
