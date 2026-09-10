from dataclasses import FrozenInstanceError, replace
import math

import pytest

from cocofest.optimization.endurance_weight_supervisor import (
    CandidateRolloutResult,
    EnduranceWeightSupervisor,
    SupervisorSnapshot,
    WeightSupervisorConfig,
    check_proposal_eligibility,
)


MUSCLES = ("anterior", "posterior", "biceps", "triceps")
COMPONENTS = ("Cn", "F", "A", "Tau1", "Km")


def _snapshot(**changes):
    values = {
        "task_id": "rider-7",
        "context_token": "rho-policy-v1",
        "cycle_index": 20,
        "start_time_s": 20.0,
        "created_at_s": 100.0,
        "horizon_cycles": 120,
        "muscle_names": MUSCLES,
        "state_component_names": COMPONENTS,
        "start_state": tuple(
            (0.1 + index, 10.0 + index, 5000.0 - index, 0.06, 0.14)
            for index in range(4)
        ),
        "incumbent_weights": (1.0, 1.0, 1.0, 1.0),
        "at_cycle_boundary": True,
    }
    values.update(changes)
    return SupervisorSnapshot(**values)


def _result(
    *,
    duration=120.0,
    completed=True,
    feasible=True,
    margin=0.1,
    status="complete",
    prefix_comparable=False,
):
    return CandidateRolloutResult(
        completed_duration=duration,
        completed=completed,
        feasible=feasible,
        minimum_signed_margin=margin,
        status=status,
        prefix_comparable=prefix_comparable,
    )


def test_completed_candidate_with_better_declared_margin_changes_weights():
    supervisor = EnduranceWeightSupervisor()
    snapshot = _snapshot()

    def evaluator(weights, _snapshot):
        margin = 0.4 if weights[0] > weights[1] else 0.1
        return _result(margin=margin)

    proposal = supervisor.evaluate(snapshot, evaluator, budget_seconds=10.0, clock=lambda: 0.0)

    assert proposal.selection_basis == "completed_horizon_margin"
    assert proposal.chosen_candidate.kind == "increase"
    assert proposal.chosen_candidate.muscle_name == "anterior"
    assert proposal.weights[0] > proposal.weights[1]
    assert proposal.chosen_evidence.minimum_signed_margin == pytest.approx(0.4)


def test_exact_tie_preserves_incumbent_before_small_change_tiebreaks():
    supervisor = EnduranceWeightSupervisor()
    snapshot = _snapshot(incumbent_weights=(1.2, 0.9, 1.1, 0.85))

    proposal = supervisor.evaluate(
        snapshot,
        lambda weights, context: _result(),
        budget_seconds=10.0,
        clock=lambda: 5.0,
    )

    assert proposal.chosen_candidate.kind == "incumbent"
    assert proposal.weights == snapshot.incumbent_weights


def test_all_incomplete_rollouts_select_only_longest_explicit_prefix():
    supervisor = EnduranceWeightSupervisor()
    snapshot = _snapshot()

    def evaluator(weights, _snapshot):
        if weights[2] > weights[0]:
            return _result(
                duration=17.0,
                completed=False,
                feasible=False,
                margin=-4.0,
                status="pulse_width_limit",
                prefix_comparable=True,
            )
        return _result(
            duration=8.0,
            completed=False,
            feasible=False,
            margin=1e9,
            status="envelope_domain_invalid",
            prefix_comparable=False,
        )

    proposal = supervisor.evaluate(snapshot, evaluator, budget_seconds=10.0, clock=lambda: 0.0)

    assert proposal.selection_basis == "longest_explicit_prefix"
    assert proposal.chosen_evidence.completed is False
    assert proposal.chosen_evidence.status == "pulse_width_limit"
    assert proposal.chosen_evidence.completed_duration == pytest.approx(17.0)
    assert proposal.chosen_candidate.muscle_name == "biceps"


def test_completed_feasible_evidence_is_not_compared_with_failed_rollout_margin():
    supervisor = EnduranceWeightSupervisor()
    snapshot = _snapshot()
    call_index = 0

    def evaluator(weights, _snapshot):
        nonlocal call_index
        call_index += 1
        if call_index == 1:
            return _result(duration=2.0, margin=-0.5)
        return _result(
            duration=1000.0,
            completed=False,
            feasible=False,
            margin=1e100,
            status="failed_after_prefix",
            prefix_comparable=True,
        )

    proposal = supervisor.evaluate(snapshot, evaluator, budget_seconds=10.0, clock=lambda: 0.0)

    assert proposal.chosen_candidate.kind == "incumbent"
    assert proposal.selection_basis == "completed_horizon_margin"


def test_invalid_and_exceptional_results_are_recorded_not_scored():
    supervisor = EnduranceWeightSupervisor()
    snapshot = _snapshot()
    call_index = 0

    def evaluator(weights, _snapshot):
        nonlocal call_index
        call_index += 1
        if call_index == 1:
            return _result(margin=0.2)
        if call_index == 2:
            return {
                "completed_duration": math.nan,
                "completed": True,
                "feasible": True,
                "minimum_signed_margin": 1e12,
                "status": "complete",
            }
        raise RuntimeError("predictor refused this envelope")

    proposal = supervisor.evaluate(snapshot, evaluator, budget_seconds=10.0, clock=lambda: 0.0)

    assert proposal.chosen_candidate.kind == "incumbent"
    assert sum(record.valid for record in proposal.evaluations) == 1
    assert proposal.evaluations[1].error_type == "ValueError"
    assert proposal.evaluations[2].error_type == "RuntimeError"


def test_predictor_failure_status_is_preserved_and_domain_prefix_fails_closed():
    supervisor = EnduranceWeightSupervisor()
    snapshot = _snapshot()
    call_index = 0

    def evaluator(weights, _snapshot):
        nonlocal call_index
        call_index += 1
        physical_failure = call_index == 1
        return {
            "completed_duration": 5.0 if physical_failure else 100.0,
            "completed": False,
            "minimum_signed_margin": -0.1,
            "status": "infeasible",
            "first_failure": {
                "status": (
                    "infeasible_total_above_bounds"
                    if physical_failure
                    else "envelope_domain_invalid"
                )
            },
        }

    proposal = supervisor.evaluate(snapshot, evaluator, budget_seconds=10.0, clock=lambda: 0.0)

    assert proposal.chosen_candidate.kind == "incumbent"
    assert proposal.chosen_evidence.status == "infeasible_total_above_bounds"
    assert proposal.evaluations[1].result.status == "envelope_domain_invalid"
    assert proposal.evaluations[1].comparable is False


def test_candidates_are_bounded_geometric_mean_one_and_include_references():
    supervisor = EnduranceWeightSupervisor(
        WeightSupervisorConfig(min_weight=0.7, max_weight=1.6, adjustment_factor=1.5)
    )
    snapshot = _snapshot(incumbent_weights=(1.5, 1.0, 0.9, 0.8))

    candidates = supervisor.candidates(snapshot)

    assert candidates[0].kind == "incumbent"
    assert any(candidate.kind == "all_ones_reference" for candidate in candidates)
    assert len(candidates) <= 10
    for candidate in candidates:
        assert all(0.7 <= weight <= 1.6 for weight in candidate.weights)
        geometric_mean = math.exp(
            math.fsum(math.log(weight) for weight in candidate.weights) / len(candidate.weights)
        )
        assert geometric_mean == pytest.approx(1.0, abs=2e-14)


def test_snapshot_normalizes_common_weight_scale_and_is_deeply_immutable():
    snapshot = _snapshot(incumbent_weights=(2.0, 2.0, 2.0, 2.0))

    assert snapshot.incumbent_weights == pytest.approx((1.0, 1.0, 1.0, 1.0))
    assert isinstance(snapshot.start_state, tuple)
    assert isinstance(snapshot.start_state[0], tuple)
    with pytest.raises(FrozenInstanceError):
        snapshot.cycle_index = 21


def test_eligibility_uses_age_cycle_boundary_and_per_component_state_tolerances():
    supervisor = EnduranceWeightSupervisor()
    source = _snapshot()
    proposal = supervisor.evaluate(
        source,
        lambda weights, context: _result(),
        budget_seconds=10.0,
        clock=lambda: 101.0,
    )
    moved_state = tuple(
        (row[0] + 0.01, row[1] + 0.2, row[2] + 2.0, row[3] + 0.001, row[4] + 0.002)
        for row in source.start_state
    )
    current = replace(
        source,
        cycle_index=21,
        start_time_s=21.0,
        created_at_s=102.0,
        start_state=moved_state,
    )
    tolerances = {"Cn": 0.02, "F": 0.3, "A": 3.0, "Tau1": 0.002, "Km": 0.003}

    eligible = check_proposal_eligibility(
        proposal,
        current,
        now_s=102.0,
        max_age_s=2.0,
        max_cycle_lag=1,
        start_time_tolerance_s=1.0,
        state_component_tolerances=tolerances,
        log_weight_tolerance=0.0,
    )
    stale = check_proposal_eligibility(
        proposal,
        current,
        now_s=104.0,
        max_age_s=2.0,
        max_cycle_lag=1,
        start_time_tolerance_s=1.0,
        state_component_tolerances=tolerances,
        log_weight_tolerance=0.0,
    )
    off_boundary = check_proposal_eligibility(
        proposal,
        replace(current, at_cycle_boundary=False),
        now_s=102.0,
        max_age_s=2.0,
        max_cycle_lag=1,
        start_time_tolerance_s=1.0,
        state_component_tolerances=tolerances,
        log_weight_tolerance=0.0,
    )

    assert eligible.eligible
    assert stale.eligible is False
    assert "proposal_too_old" in stale.reasons
    assert "not_at_cycle_boundary" in off_boundary.reasons
    newly_issued_from_old_source = check_proposal_eligibility(
        replace(proposal, issued_at_s=200.0),
        current,
        now_s=201.0,
        max_age_s=2.0,
        max_cycle_lag=1,
        start_time_tolerance_s=1.0,
        state_component_tolerances=tolerances,
        log_weight_tolerance=0.0,
    )
    assert newly_issued_from_old_source.eligible is False
    assert "source_snapshot_too_old" in newly_issued_from_old_source.reasons
    assert "proposal_too_old" not in newly_issued_from_old_source.reasons
    with pytest.raises(ValueError, match="max_age_s must be finite"):
        check_proposal_eligibility(
            proposal,
            current,
            now_s=102.0,
            max_age_s=math.inf,
            max_cycle_lag=1,
            start_time_tolerance_s=1.0,
            state_component_tolerances=tolerances,
            log_weight_tolerance=0.0,
        )


def test_budget_is_checked_between_whole_evaluations_not_as_a_hard_timeout():
    supervisor = EnduranceWeightSupervisor()
    snapshot = _snapshot()
    clock_values = iter((0.0, 0.6, 1.4))
    calls = []

    proposal = supervisor.evaluate(
        snapshot,
        lambda weights, context: calls.append(weights) or _result(),
        budget_seconds=1.0,
        clock=lambda: next(clock_values),
    )

    assert len(calls) == 2
    assert len(proposal.evaluations) == 2
    assert proposal.budget_exhausted
    assert proposal.elapsed_s == pytest.approx(1.4)
    assert proposal.budget_overrun_s == pytest.approx(0.4)
    assert proposal.hard_timeout_enforced is False
