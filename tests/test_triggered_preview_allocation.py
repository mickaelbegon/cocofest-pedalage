from dataclasses import replace
import json

import numpy as np
import pytest

from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval
from cocofest.optimization.compact_muscle_prediction import AffineRecruitmentMap, CompactMusclePredictor
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.preview_muscle_allocation import PreviewAllocationPlan, PreviewMuscleAllocation
from cocofest.optimization.triggered_preview_allocation import TriggeredPreviewMuscleAllocation


class ToyPredictor:
    muscle_count = 2
    maximum_recruitment = np.array([.95, .95])
    pd0 = np.array([.0001, .0001])
    pdt = np.array([.0002, .0002])
    pulse_width_max = pd0 - pdt * np.log1p(-maximum_recruitment)

    def __init__(self, *, next_coefficients=(1., 0.), next_target=.1, decay=.9):
        self.decay = decay
        self.intervals = (
            MomentTrackingInterval(.1, (1., 1.), (1., 1.), (1., 1.), (.5, .5)),
            MomentTrackingInterval(.1, (1., 1.), (1., 1.), next_coefficients, (next_target, 0.)),
        )

    def phase_map(self, state, phase):
        intercept = np.asarray(state, float).copy()
        intercept[:, 1] *= self.decay
        slope = np.zeros_like(intercept)
        slope[:, 1] = 1.
        return AffineRecruitmentMap(intercept, slope, np.zeros(2), np.zeros(2), self.maximum_recruitment)


def initial_state():
    return np.array([[.1, 0., 1., .1, .1]] * 2)


def test_safe_hybrid_matches_greedy_trajectory_and_never_solves_qp():
    predictor = ToyPredictor(next_coefficients=(1., 1.), next_target=1.)
    greedy = PreviewMuscleAllocation(predictor, preview_phases=1).rollout(initial_state(), horizon_cycles=1)
    hybrid = TriggeredPreviewMuscleAllocation(predictor, moment_scale=1.)
    result = hybrid.rollout(initial_state(), horizon_cycles=1)
    assert result.status == greedy.status == "complete"
    for name in ("state_history", "pulse_widths", "allocated_moments", "achieved_moments"):
        np.testing.assert_array_equal(getattr(result, name), getattr(greedy, name))
    assert result.screenings == 1 and result.triggers == result.qp_solves == 0
    assert result.greedy_fast_path_steps == 2 and result.greedy_fallback_steps == 0
    assert result.screen_failures == 0
    assert result.screening_time_s > 0 and result.triggered_preview_time_s == 0
    assert result.total_time_s >= result.screening_time_s


@pytest.mark.parametrize("fraction", [0., .01])
def test_residual_trap_triggers_from_greedy_endpoint_and_recovers_exact_tracking(fraction):
    predictor = ToyPredictor()
    result = TriggeredPreviewMuscleAllocation(predictor, moment_scale=1., trigger_margin_fraction=fraction).rollout(
        initial_state(), horizon_cycles=1)
    assert result.status == "complete", result.first_failure
    screen = result.diagnostics[0]
    assert screen["next_lower_total_moment_nm"] == pytest.approx(.45)
    assert screen["next_signed_margin_nm"] == pytest.approx(-.35)
    assert result.screenings == result.triggers == result.qp_solves == 1
    assert result.greedy_fast_path_steps == 1 and result.greedy_fallback_steps == 0
    assert result.triggered_preview_time_s > 0
    np.testing.assert_allclose(result.achieved_moments.sum(axis=1), [[1., .1]], atol=1e-8)
    assert np.all(result.pulse_widths >= predictor.pd0[None, :, None])
    assert np.all(result.pulse_widths <= predictor.pulse_width_max[None, :, None])


def test_trigger_uses_less_equal_at_fixed_threshold_without_tracking_tolerance():
    predictor = ToyPredictor(next_target=.5, decay=.5)
    # Next minimum=.25, target=.5, margin=.25 = .01 * 25 exactly.
    boundary = TriggeredPreviewMuscleAllocation(predictor, moment_scale=25., trigger_margin_fraction=.01).plan(initial_state(), 0)
    assert boundary.diagnostics["next_signed_margin_nm"] == .25
    assert boundary.diagnostics["trigger_threshold_nm"] == .25
    assert boundary.diagnostics["triggered"]
    smaller = TriggeredPreviewMuscleAllocation(predictor, moment_scale=25., trigger_margin_fraction=.009).plan(initial_state(), 0)
    assert not smaller.diagnostics["triggered"]
    assert smaller.diagnostics["allocation_mode"] == "greedy"
    assert smaller.diagnostics["equality_residual_max_nm"] < 1e-8


def test_signed_antagonist_and_zero_coefficient_envelopes():
    signed = TriggeredPreviewMuscleAllocation(ToyPredictor(next_coefficients=(1., -1.), next_target=-.1),
                                               moment_scale=2., trigger_margin_fraction=0.).plan(initial_state(), 0)
    assert signed.accepted and not signed.diagnostics["triggered"]
    assert signed.diagnostics["next_lower_total_moment_nm"] == pytest.approx(-.95)
    assert signed.diagnostics["next_upper_total_moment_nm"] == pytest.approx(.95)
    assert signed.diagnostics["next_signed_margin_nm"] == pytest.approx(.85)
    assert signed.diagnostics["next_normalized_signed_margin"] == pytest.approx(.425)
    zero = TriggeredPreviewMuscleAllocation(ToyPredictor(next_coefficients=(0., 0.), next_target=0.),
                                             moment_scale=1., trigger_margin_fraction=0.).plan(initial_state(), 0)
    assert zero.accepted and zero.diagnostics["triggered"]
    assert zero.diagnostics["next_signed_margin_nm"] == 0.


@pytest.mark.parametrize("corruption", ["nan", "negative_capacity"])
def test_invalid_future_screen_fails_closed_without_calling_preview(monkeypatch, corruption):
    predictor = ToyPredictor()
    original = predictor.phase_map
    def invalid(state, phase):
        transition = original(state, phase)
        if phase == 1:
            intercept = transition.intercept.copy()
            intercept[0, 2] = np.nan if corruption == "nan" else -1.
            return replace(transition, intercept=intercept)
        return transition
    monkeypatch.setattr(predictor, "phase_map", invalid)
    result = TriggeredPreviewMuscleAllocation(predictor, moment_scale=1.).rollout(initial_state(), horizon_cycles=1)
    assert result.status == "infeasible" and result.completed_intervals == 0
    assert result.screenings == result.screen_failures == 1
    assert result.triggers == result.qp_solves == result.greedy_fallback_steps == 0
    assert np.all(np.isnan(result.pulse_widths))
    assert result.diagnostics[0]["screen_reason"] == "invalid_next_envelope"


def test_rejected_triggered_preview_never_reverts_to_greedy(monkeypatch):
    def reject(self, states, phase, *, depth):
        return PreviewAllocationPlan(False, None, None, {
            "accepted": False, "reason": "deliberate_qp_rejection", "qp_solved": True,
            "optimizer_success": False, "optimizer_status": 9, "optimizer_iterations": 1,
            "nominal_projected_steps": 0, "fallback_reason": None,
        })
    monkeypatch.setattr(PreviewMuscleAllocation, "plan", reject)
    result = TriggeredPreviewMuscleAllocation(ToyPredictor(), moment_scale=1.).rollout(initial_state(), horizon_cycles=1)
    assert result.status == "infeasible" and result.completed_intervals == 0
    assert result.triggers == result.qp_failures == 1
    assert result.greedy_fallback_steps == result.greedy_fast_path_steps == 0
    assert result.first_failure["status"] == "deliberate_qp_rejection"


def test_last_phase_never_looks_outside_declared_horizon(monkeypatch):
    predictor = ToyPredictor()
    original = predictor.phase_map
    def prohibit_next(state, phase):
        if phase != 0:
            pytest.fail("A terminal greedy phase must not access the following phase.")
        return original(state, phase)
    monkeypatch.setattr(predictor, "phase_map", prohibit_next)
    plan = TriggeredPreviewMuscleAllocation(predictor, moment_scale=1.).plan(initial_state(), 0, depth=1)
    assert plan.accepted and not plan.diagnostics["screened"] and not plan.diagnostics["triggered"]
    assert plan.diagnostics["next_phase_index"] is None


def test_threshold_is_fixed_and_metadata_is_an_independent_json_snapshot():
    policy = TriggeredPreviewMuscleAllocation(ToyPredictor(), moment_scale=2., trigger_margin_fraction=.01)
    metadata = policy.allocation_policy_metadata
    assert metadata["screening_depth"] == 1 and metadata["preview_phases"] == 2
    assert metadata["trigger_margin_fraction"] == .01
    json.dumps(metadata, allow_nan=False)
    metadata["trigger_margin_fraction"] = 0.
    assert policy.allocation_policy_metadata["trigger_margin_fraction"] == .01
    for setting in ("trigger_margin_fraction", "moment_scale", "screening_depth"):
        with pytest.raises(AttributeError):
            setattr(policy, setting, 0)


def test_nonfinite_normalized_screen_is_rejected_and_json_remains_valid():
    result = TriggeredPreviewMuscleAllocation(ToyPredictor(), moment_scale=5e-324,
                                               trigger_margin_fraction=0.).rollout(initial_state(), horizon_cycles=1)
    assert result.status == "infeasible" and result.screen_failures == 1
    assert result.triggers == 0
    json.dumps(result.diagnostics, allow_nan=False)


@pytest.mark.parametrize("keyword,value", [
    ("moment_scale", 0.), ("moment_scale", -1.), ("moment_scale", np.nan), ("moment_scale", True),
    ("moment_scale", "1"), ("trigger_margin_fraction", -1.), ("trigger_margin_fraction", np.inf),
    ("trigger_margin_fraction", False), ("preview_phases", 3), ("preview_phases", 2.),
    ("fallback_to_greedy", True), ("fallback_to_greedy", "False"),
])
def test_invalid_policy_configuration_is_rejected(keyword, value):
    options = {"moment_scale": 1., keyword: value}
    with pytest.raises(ValueError):
        TriggeredPreviewMuscleAllocation(ToyPredictor(), **options)


def test_real_compact_safe_path_retains_greedy_pw_and_states():
    parameter = DingPulseWidthParameters(
        DingFatigueParameters(1200., .060601, .137, -1.4, 2.1e-5, 1.9e-5, 445.5),
        tauc=.011, tau2=.001, pd0=.000131405, pdt=.000194138, pulse_width_max=.0006)
    phase = MomentTrackingInterval(1 / 30, (1.0597, 1.0597), (.95, .95), (.05, -.03), (.7, -.2))
    predictor = CompactMusclePredictor((phase,), (parameter, parameter))
    initial = np.array([[.16298, 20., 1150., .078, .15]] * 2)
    direct = predictor.rollout(initial, horizon_cycles=5)
    result = TriggeredPreviewMuscleAllocation(predictor, moment_scale=1., trigger_margin_fraction=0.).rollout(initial, horizon_cycles=5)
    assert result.status == direct.status == "complete"
    assert result.triggers == 0
    np.testing.assert_array_equal(result.pulse_widths, direct.pulse_widths)
    np.testing.assert_array_equal(result.state_history, direct.state_history)
