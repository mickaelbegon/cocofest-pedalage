import json
from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.endurance_weight_supervisor import (
    CandidateEvaluation,
    CandidateRolloutResult,
    WeightCandidate,
)
from scripts import validate_multilevel_endurance as validation


def _candidate():
    return WeightCandidate(
        index=0,
        kind="incumbent",
        muscle_name=None,
        log_step=0.0,
        weights=(1.0, 1.0, 1.0, 1.0),
    )


def test_candidate_report_uses_comparable_not_merely_parsed_result():
    candidate = _candidate()
    parsed = CandidateRolloutResult(
        completed_duration=7.0,
        completed=False,
        feasible=False,
        minimum_signed_margin=-0.2,
        status="envelope_domain_invalid",
        prefix_comparable=False,
    )
    evaluated = CandidateEvaluation(candidate, parsed, elapsed_after_s=0.3)
    actual = SimpleNamespace(
        status="infeasible",
        completed_cycles=7,
        completed_intervals=7,
        completed_duration=7.0,
        minimum_signed_margin=-0.2,
        signed_moment_errors=np.full((2, 4), np.nan),
        first_failure={"status": "envelope_domain_invalid"},
    )

    record = validation._candidate_report(
        evaluated,
        {candidate.weights: actual},
        {candidate.weights: 0.25},
        baseline=None,
        phases=4,
    )

    assert record["supervisor_result_parsed"] is True
    assert record["supervisor_accepted_evidence"] is False
    assert record["first_failure"]["status"] == "envelope_domain_invalid"
    assert record["maximum_tracking_error_nm"] is None


def test_candidate_report_survives_callback_failure_without_cache_entry():
    candidate = _candidate()
    evaluated = CandidateEvaluation(
        candidate,
        result=None,
        elapsed_after_s=0.4,
        error_type="RuntimeError",
        error_message="predictor failed",
    )

    record = validation._candidate_report(
        evaluated,
        cached={},
        timings={candidate.weights: 0.4},
        baseline=None,
        phases=4,
    )

    assert record["status"] == "callback_result_unavailable"
    assert record["callback_result_cached"] is False
    assert record["supervisor_accepted_evidence"] is False
    assert record["supervisor_error_type"] == "RuntimeError"
    assert record["completed_duration_s"] is None


def test_prefix_replay_labels_its_limited_validation_scope(monkeypatch):
    initial = np.asarray([[0.1, 10.0, 5000.0, 0.06, 0.14]])
    replay_history = np.repeat(initial[None, :, :], 3, axis=0)
    replay_moments = np.ones((2, 1))
    monkeypatch.setattr(
        validation,
        "_full_ding_replay",
        lambda initial, task, pulse_widths, steps, substeps: (
            replay_history,
            replay_moments,
        ),
    )
    task = SimpleNamespace(
        intervals=(SimpleNamespace(duration=1.0),),
        parameters=(SimpleNamespace(fatigue=SimpleNamespace(rest_state=(5000.0, 0.06, 0.14))),),
    )
    result = SimpleNamespace(
        completed_intervals=3,
        completed=False,
        pulse_widths=np.ones((3, 1, 1)),
        original_total_moments=np.ones((3, 1)),
        state_history=np.repeat(initial[None, :, :], 4, axis=0),
    )
    arrays = {}

    report = validation._prefix_replay(
        initial,
        task,
        result,
        maximum_intervals=2,
        substeps=4,
        arrays=arrays,
        key="candidate",
    )

    assert report["status"] == "completed_prefix_replay"
    assert report["validation_scope"] == "explicit_completed_prefix_only"
    assert report["entire_requested_horizon_replayed"] is False
    assert report["predictor_horizon_completed"] is False
    assert report["does_not_certify_unreplayed_intervals"] is True


def test_json_finite_replaces_nonfinite_values_with_audited_nulls():
    converted, paths = validation._json_finite(
        {"array": np.asarray([1.0, np.nan]), "nested": {"value": np.inf}}
    )

    assert converted == {"array": [1.0, None], "nested": {"value": None}}
    assert paths == ["$.array[1]", "$.nested.value"]
    json.dumps(converted, allow_nan=False)


def test_budget_audit_reports_whole_candidate_noninterrupting_semantics():
    proposal = SimpleNamespace(
        evaluations=(object(), object()),
        budget_seconds=1.0,
        budget_exhausted=True,
        hard_timeout_enforced=False,
        budget_overrun_s=0.4,
    )

    audit = validation._budget_audit(proposal, declared_count=10)

    assert audit["candidate_count_evaluated"] == 2
    assert audit["candidate_count_not_started"] == 8
    assert audit["budget_exhausted_between_candidates"] is True
    assert audit["whole_candidate_callbacks_are_not_interrupted"] is True
    assert audit["measured_budget_overrun_s"] == pytest.approx(0.4)


@pytest.mark.parametrize("budget", (0.0, -1.0, np.inf, np.nan))
def test_run_case_rejects_an_unbounded_or_nonpositive_budget_before_setup(budget):
    with pytest.raises(ValueError, match="budget_seconds"):
        validation.run_case(
            SimpleNamespace(budget_seconds=budget),
            anchor=0,
            horizon=1,
            arrays={},
            source_sha="source",
            profile_sha="profile",
        )
