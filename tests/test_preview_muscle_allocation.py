from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval
from cocofest.optimization.compact_muscle_prediction import AffineRecruitmentMap, CompactMusclePredictor
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.preview_muscle_allocation import PreviewMuscleAllocation
import cocofest.optimization.preview_muscle_allocation as preview_module


def real_predictor():
    p = DingPulseWidthParameters(
        DingFatigueParameters(1200., .060601, .137, -1.4, 2.1e-5, 1.9e-5, 445.5),
        tauc=.011, tau2=.001, pd0=.000131405, pdt=.000194138, pulse_width_max=.0006)
    intervals = tuple(MomentTrackingInterval(1 / 30, (1.0597, 1.0597), (.95, .95),
                                             (.05, -.03), (.7, -.2)) for _ in range(3))
    return CompactMusclePredictor(intervals, (p, p), substeps=16)


class ResidualTrapPredictor:
    """Exact toy map F_next=.9 F+r; mechanics change at the next phase."""
    muscle_count = 2
    maximum_recruitment = np.array([.95, .95])
    pd0 = np.array([.0001, .0001])
    pdt = np.array([.0002, .0002])
    pulse_width_max = pd0 - pdt * np.log1p(-maximum_recruitment)

    def __init__(self, *, impossible=False, signed=False, invisible=False):
        self.intervals = (
            MomentTrackingInterval(.1, (1., 1.), (1., 1.), (1., -1.) if signed else (1., 1.),
                                   (.5, -.5) if signed else (.5, .5)),
            MomentTrackingInterval(.1, (1., 1.), (1., 1.), (0., 0.) if invisible else (1., 0.),
                                   ((-.1 if impossible else 0.) if invisible else (-.1 if impossible else .1), 0.)),
        )

    def phase_map(self, states, interval_index):
        state = np.asarray(states, float)
        intercept = state.copy()
        intercept[:, 1] *= .9
        slope = np.zeros_like(state)
        slope[:, 1] = 1.
        return AffineRecruitmentMap(intercept, slope, np.zeros(2), np.zeros(2), self.maximum_recruitment)


def toy_state():
    return np.array([[.1, 0., 1., .1, .1]] * 2)


def test_depth_one_matches_original_compact_strict_policy():
    predictor = real_predictor()
    initial = np.array([[.16298, 20., 1150., .078, .15]] * 2)
    direct = predictor.rollout(initial, horizon_cycles=5)
    result = PreviewMuscleAllocation(predictor, preview_phases=1).rollout(initial, horizon_cycles=5)
    assert result.status == direct.status == "complete"
    for name in ("state_history", "pulse_widths", "allocated_moments", "achieved_moments"):
        np.testing.assert_allclose(getattr(result, name), getattr(direct, name), atol=1e-13, rtol=1e-13)
    assert result.qp_solves == result.qp_failures == result.greedy_fallback_steps == 0


def test_causal_force_coupling_and_first_phase_map_are_exact():
    predictor = ResidualTrapPredictor()
    policy = PreviewMuscleAllocation(predictor, preview_phases=2)
    problem = policy.build_preview_qp(toy_state(), 0)
    np.testing.assert_array_equal(problem.force_jacobians[0, :, 2:], 0.)
    np.testing.assert_allclose(problem.force_jacobians[1, :, :2], .9 * np.eye(2))
    np.testing.assert_allclose(problem.force_jacobians[1, :, 2:], np.eye(2))
    controls = np.array([.1, .8, .01, .0])
    first = predictor.phase_map(toy_state(), 0).endpoint(controls[:2])
    np.testing.assert_allclose(problem.force_offsets[0] + problem.force_jacobians[0] @ controls, first[:, 1])
    second = predictor.phase_map(first, 1).endpoint(controls[2:])
    np.testing.assert_allclose(problem.force_offsets[1] + problem.force_jacobians[1] @ controls, second[:, 1])


def test_preview_prevents_residual_force_trap_without_changing_any_target():
    predictor = ResidualTrapPredictor()
    greedy = PreviewMuscleAllocation(predictor, preview_phases=1).rollout(toy_state(), horizon_cycles=1)
    preview = PreviewMuscleAllocation(predictor, preview_phases=2).rollout(toy_state(), horizon_cycles=1)
    assert greedy.status == "infeasible" and greedy.completed_intervals == 1
    assert preview.status == "complete" and preview.completed_intervals == 2
    np.testing.assert_allclose(preview.achieved_moments.sum(axis=1), [[1., .1]], atol=1e-8)
    assert preview.state_history[1, 0, 1] < .12
    assert preview.seed_projected_steps > 0
    assert preview.qp_solves == 1 and preview.greedy_fallback_steps == 0
    assert [d["preview_depth"] for d in preview.diagnostics] == [2, 1]
    assert preview.diagnostics[0]["kkt_stationarity_max"] <= 1e-6
    assert np.all(preview.pulse_widths >= predictor.pd0[None, :, None])
    assert np.all(preview.pulse_widths <= predictor.pulse_width_max[None, :, None])


def test_antagonists_and_zero_effect_equalities_preserve_signed_targets():
    for predictor in (ResidualTrapPredictor(signed=True), ResidualTrapPredictor(invisible=True)):
        result = PreviewMuscleAllocation(predictor).rollout(toy_state(), horizon_cycles=1)
        assert result.status == "complete", result.first_failure
        target = np.array([[sum(p.target_moments) for p in predictor.intervals]])
        np.testing.assert_allclose(result.achieved_moments.sum(axis=1), target, atol=1e-8)


def test_invisible_incompatible_equality_fails_honestly():
    predictor = ResidualTrapPredictor(impossible=True, invisible=True)
    result = PreviewMuscleAllocation(predictor).rollout(toy_state(), horizon_cycles=1)
    assert result.status == "infeasible" and result.completed_intervals == 0
    assert result.first_failure["status"] == "inconsistent_affine_equalities"
    assert np.all(np.isnan(result.pulse_widths))


def test_exact_greedy_fallback_is_explicit_and_does_not_hide_later_failure():
    predictor = ResidualTrapPredictor(impossible=True)
    strict = PreviewMuscleAllocation(predictor).rollout(toy_state(), horizon_cycles=1)
    fallback = PreviewMuscleAllocation(predictor, fallback_to_greedy=True).rollout(toy_state(), horizon_cycles=1)
    assert strict.status == fallback.status == "infeasible"
    assert strict.completed_intervals == 0 and fallback.completed_intervals == 1
    assert fallback.greedy_fallback_steps == 1
    assert fallback.diagnostics[0]["fallback_reason"] is not None


def test_false_optimizer_success_is_rejected_by_independent_optimality_audit(monkeypatch):
    predictor = ResidualTrapPredictor(invisible=True)
    def fake_minimize(fun, x0, **kwargs):
        # Phase-zero .1/.9 is feasible but worse than equal sharing .5/.5.
        return SimpleNamespace(x=np.array([.1, .9, .0, .0]), success=True, status=0, nit=1, message="claimed success")
    monkeypatch.setattr(preview_module, "minimize", fake_minimize)
    plan = PreviewMuscleAllocation(predictor).plan(toy_state(), 0)
    assert not plan.accepted and plan.recruitment is None
    assert plan.diagnostics["optimizer_success"]
    assert plan.diagnostics["equality_residual_max_nm"] < 1e-8
    assert plan.diagnostics["kkt_stationarity_max"] > 1e-6


def test_nonfinite_claimed_optimizer_success_fails_closed(monkeypatch):
    monkeypatch.setattr(preview_module, "minimize", lambda *args, **kwargs:
                        SimpleNamespace(x=np.full(4, np.nan), success=True, status=0, nit=1, message="false success"))
    plan = PreviewMuscleAllocation(ResidualTrapPredictor()).plan(toy_state(), 0)
    assert not plan.accepted and plan.recruitment is None
    assert plan.diagnostics["reason"] == "nonfinite_or_invalid_qp_solution"


def test_roundoff_recruitment_is_clipped_consistently_for_state_and_pw(monkeypatch):
    predictor = ResidualTrapPredictor(invisible=True)
    predictor.intervals = (predictor.intervals[1], predictor.intervals[1])
    monkeypatch.setattr(preview_module, "minimize", lambda *args, **kwargs:
                        SimpleNamespace(x=np.array([-5e-13, 0., 0., 0.]), success=True, status=0, nit=1, message="roundoff"))
    result = PreviewMuscleAllocation(predictor).rollout(toy_state(), horizon_cycles=1)
    assert result.status == "complete", result.first_failure
    np.testing.assert_array_equal(result.pulse_widths[:, :, 0], predictor.pd0[None, :])
    np.testing.assert_array_equal(result.state_history[1], predictor.phase_map(toy_state(), 0).endpoint(np.zeros(2)))
    assert result.diagnostics[0]["kkt_complementarity_max"] <= 1e-6


def test_preview_is_truncated_at_declared_horizon():
    predictor = ResidualTrapPredictor()
    result = PreviewMuscleAllocation(predictor, preview_phases=3).rollout(toy_state(), horizon_cycles=1)
    assert result.status == "complete"
    assert [d["preview_depth"] for d in result.diagnostics] == [2, 1]


@pytest.mark.parametrize("keyword,value", [("preview_phases", 4), ("max_iterations", 0),
                                          ("moment_tolerance", 0), ("max_solve_time_s", -1)])
def test_invalid_options(keyword, value):
    with pytest.raises(ValueError):
        PreviewMuscleAllocation(real_predictor(), **{keyword: value})
