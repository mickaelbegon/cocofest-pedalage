from types import SimpleNamespace
import json

import numpy as np
import pytest

from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from scripts.diagnose_compact_rollout_failure import compose_signed_envelope, diagnose_failure, full_ding_phase_envelope
import scripts.diagnose_compact_rollout_failure as diagnostic


def example_policy():
    parameters = DingPulseWidthParameters(
        DingFatigueParameters(1200., .060601, .137, -1.4, 2.1e-5, 1.9e-5, 445.5),
        tauc=.011, tau2=.001, pd0=.000131405, pdt=.000194138, pulse_width_max=.0006)
    first = MomentTrackingInterval(1 / 30, (1.0597, 1.0597), (.95, .95), (.05, -.03), (.7, -.2))
    second = MomentTrackingInterval(1 / 30, (1.0597, 1.0597), (.95, .95), (.05, -.03), (100., 0.))
    return SimpleNamespace(intervals=(first, second), parameters=(parameters, parameters))


def test_signed_lower_envelope_uses_antagonist_maximum_force():
    result = compose_signed_envelope([[2., 3., 1.], [3., 4., 2.], [4., 5., 3.]], [.1, -.2, 0.])
    assert result["sampled_monotonicity_passed"]
    assert result["lower_bound_nm"] == pytest.approx(.1 * 2 - .2 * 5)
    assert result["upper_bound_nm"] == pytest.approx(.1 * 4 - .2 * 3)
    assert result["lower_uses_pw_max"] == [False, True, False]


def test_nonmonotonic_force_grid_cannot_claim_endpoint_envelope():
    result = compose_signed_envelope([[1., 2.], [3., 3.], [2., 4.]], [.1, -.1])
    assert not result["sampled_monotonicity_passed"]
    assert result["lower_bound_nm"] is None and result["upper_bound_nm"] is None


def test_full_ding_grid_includes_exact_minmax_and_monotone_intermediates():
    policy = example_policy()
    state = np.array([[.16298, 20., 1150., .078, .15]] * 2)
    result, endpoints, widths = full_ding_phase_envelope(state, policy.intervals[0], policy.parameters,
                                                       substeps=16, pw_samples=5)
    assert result["sampled_monotonicity_passed"]
    np.testing.assert_array_equal(widths[0], [p.pd0 for p in policy.parameters])
    np.testing.assert_allclose(widths[-1], [p.pulse_width_max for p in policy.parameters], atol=1e-18)
    assert endpoints.shape == (5, 2, 5)
    assert result["lower_signed_margin_nm"] == pytest.approx(.5 - result["lower_bound_nm"])


def test_diagnostic_replays_successful_prefix_from_exact_state_then_refines_failure():
    policy = example_policy()
    initial = np.array([[.16295396, 20., 1150., .078, .15]] * 2)
    report, arrays = diagnose_failure(initial, policy, horizon_cycles=1, refinements=(16, 32, 64),
                                      maximum_substeps=128, pw_samples=5)
    assert report["compact_completed_intervals"] == 1
    assert report["critical_phase_zero_based"] == 1
    assert report["convergence_passed"]
    assert report["conditional_critical_target_outside_refined_signed_envelope"]
    assert not report["complete_horizon_validated"]
    for substeps in (16, 32, 64):
        history = arrays[f"rk4_{substeps}__prefix_state_history"]
        np.testing.assert_array_equal(history[0], initial)
        assert history.shape == (2, 2, 5)
    assert len(report["refinements"]) == 4  # Automatically refine an unresolved critical envelope.


def test_invalid_refinement_schedule_is_rejected_before_prediction():
    with pytest.raises(ValueError, match="refinements"):
        diagnose_failure(np.ones((2, 5)), example_policy(), refinements=(16, 8), maximum_substeps=64)


def test_nonmonotonic_critical_grid_is_reported_as_rejected_with_strict_json(monkeypatch):
    original = diagnostic.full_ding_phase_envelope
    def nonmonotonic(*args, **kwargs):
        result, history, widths = original(*args, **kwargs)
        result.update(sampled_monotonicity_passed=False, lower_bound_nm=None,
                      upper_bound_nm=None, lower_signed_margin_nm=None, upper_signed_margin_nm=None)
        return result, history, widths
    monkeypatch.setattr(diagnostic, "full_ding_phase_envelope", nonmonotonic)
    initial = np.array([[.16295396, 20., 1150., .078, .15]] * 2)
    report, _ = diagnose_failure(initial, example_policy(), horizon_cycles=1,
                                refinements=(16, 32), maximum_substeps=32, pw_samples=3)
    assert not report["convergence_passed"]
    assert report["conditional_critical_target_outside_refined_signed_envelope"] is None
    assert report["refinements"][-1]["refinement"]["maximum_critical_signed_bound_change_nm"] is None
    json.dumps(report, allow_nan=False)


def test_same_compact_boundary_comparison_must_also_converge(monkeypatch):
    original = diagnostic.full_ding_phase_envelope
    calls = 0
    def altered_same_boundary(*args, **kwargs):
        nonlocal calls
        calls += 1
        result, history, widths = original(*args, **kwargs)
        if calls % 2 == 0:
            # Emulate unresolved full-Ding integration on the second boundary.
            result["lower_bound_nm"] += .01 * kwargs["substeps"]
        return result, history, widths
    monkeypatch.setattr(diagnostic, "full_ding_phase_envelope", altered_same_boundary)
    initial = np.array([[.16295396, 20., 1150., .078, .15]] * 2)
    report, _ = diagnose_failure(initial, example_policy(), horizon_cycles=1,
                                refinements=(16, 32), maximum_substeps=32, pw_samples=3,
                                convergence_absolute_nm=.001)
    refinement = report["refinements"][-1]["refinement"]
    assert refinement["maximum_critical_signed_bound_change_nm"] < .001
    assert refinement["maximum_compact_boundary_signed_bound_change_nm"] > .1
    assert not report["convergence_passed"]


@pytest.mark.parametrize("samples", [0, 2, 2.5, True])
def test_invalid_pw_grid_is_rejected_before_prediction(samples):
    with pytest.raises(ValueError, match="pw_samples"):
        diagnose_failure(np.ones((2, 5)), example_policy(), pw_samples=samples)
