from dataclasses import asdict
import json

import numpy as np
import pytest

from cocofest.optimization.pace_vr import create_pace_vr_snapshot
from scripts.benchmark_pace_vr_candidates import benchmark, _candidate_set
from tests.test_benchmark_pace_vr_rollout import _case


def test_log_symmetric_candidate_set_and_duplicate_guard():
    incumbent = np.array([1., 1.])
    proposal = np.array([1.44, .64])
    candidates = _candidate_set(incumbent, proposal)
    np.testing.assert_allclose(candidates["half_step"], [1.2, .8])
    np.testing.assert_allclose(candidates["opposite_step"], [1 / 1.44, 1 / .64])
    with pytest.raises(ValueError, match="duplicate"):
        _candidate_set(incumbent, proposal, {"proposal": [1., 1.]})


def test_matched_certified_snapshot_scores_candidates_without_rho_claim(tmp_path):
    supervisor, states, widths = _case()
    snapshot = asdict(create_pace_vr_snapshot(supervisor, states, widths,
        request_id="right:20", source_cycle=20, deadline_seconds=60.,
        certified=True, weights=[1., 1.]))
    source = tmp_path / "result.json"
    source.write_text(json.dumps({"cycles": [{"cycle": 20, "certified": True,
                                              "pace_vr_snapshot": snapshot}]}))
    journal = tmp_path / "pace_vr.jsonl"
    journal.write_text(json.dumps({"status": "applied", "source_cycle": 20,
                                   "weights_after": [1.2, .8]}) + "\n")
    report = benchmark(source, 20, 2, proposal_journal=journal,
                       extra_candidates={"unit_scaled": [2., 2.]})
    assert report["source_context_digest"] == snapshot["context_digest"]
    assert report["full_rho_restarted"] is False
    assert report["physiological_failure_certified"] is False
    assert report["source_sha256"]
    assert report["best_versus_incumbent"]["feasible_prefix_gain_cycles"] >= 0
    entries = {entry["candidate_id"]: entry for entry in report["entries"]}
    assert set(entries) == {"incumbent", "proposal", "half_step", "opposite_step", "unit_scaled"}
    assert entries["incumbent"]["feasible_prefix_cycles"] == 2
    np.testing.assert_allclose(entries["incumbent"]["normalized_weights_used"],
                               entries["unit_scaled"]["normalized_weights_used"])
    assert all(entry["work_residual_max_j"] >= 0 for entry in report["entries"])
    json.dumps(report, allow_nan=False)
