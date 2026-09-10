"""The preview adapter must preserve value semantics and reject stale policy settings."""

from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval
from cocofest.optimization.compact_muscle_prediction import CompactMusclePredictor
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.local_endurance_value import (
    CompactEnduranceValueOracle, LocalEnduranceCoordinates, fit_local_endurance_value,
)
from cocofest.optimization.local_endurance_value_ocp import LocalEnduranceValueBinding
from cocofest.optimization.preview_endurance_value import PreviewEnduranceValueOracle


def _problem():
    p = DingPulseWidthParameters(
        DingFatigueParameters(1200., .060601, .137, -1.4, 2.1e-5, 1.9e-5, 445.5),
        tauc=.011, tau2=.001, pd0=.000131405, pdt=.000194138, pulse_width_max=.0006,
    )
    phase = MomentTrackingInterval(1/30, (1.0597355478,)*2, (.95,)*2, (.05, -.03), (.7, -.2))
    predictor = CompactMusclePredictor((phase,), (p, p))
    coordinates = LocalEnduranceCoordinates.from_state(
        [[.16295396, 20., 1150., .078, .15]]*2, (p, p), force_scale=100.,
    )
    return predictor, coordinates


def _stub_policy(predictor, **options):
    return SimpleNamespace(predictor=predictor, preview_phases=2, moment_tolerance=1e-8,
                           max_iterations=60, fallback_to_greedy=False,
                           optimality_tolerance=1e-6, recruitment_regularization=1e-8,
                           max_solve_time_s=.5, **options)


def test_one_phase_preview_value_matches_original_scalar_value():
    from cocofest.optimization.preview_muscle_allocation import PreviewMuscleAllocation

    predictor, coordinates = _problem()
    policy = PreviewMuscleAllocation(predictor, preview_phases=1)
    scalar = CompactEnduranceValueOracle(predictor, coordinates, horizon_cycles=3, moment_scale=1.)
    preview = PreviewEnduranceValueOracle(policy, coordinates, horizon_cycles=3, moment_scale=1.)
    for point in coordinates.anchor + np.random.default_rng(4).uniform(-1e-4, 1e-4, (3, 4)):
        actual, expected = preview.evaluate(point), scalar.evaluate(point)
        assert actual.status == expected.status == "complete"
        assert actual.value == pytest.approx(expected.value, abs=1e-13)
        assert actual.minimum_signed_margin == pytest.approx(expected.minimum_signed_margin, abs=1e-13)
    assert preview.value_context_metadata["task_sha256"] == scalar.value_context_metadata["task_sha256"]
    assert preview.value_context_metadata["allocation_policy"]["preview_phases"] == 1
    assert preview.value_context_metadata["tracking_band_nm"] == 0.


def test_incomplete_policy_never_produces_a_partial_value():
    predictor, coordinates = _problem()
    calls = []

    def failed_rollout(state, *, horizon_cycles):
        calls.append((state.copy(), horizon_cycles))
        return SimpleNamespace(status="infeasible", completed_intervals=2, first_failure={"reason": "QP failed"})

    policy = _stub_policy(predictor, rollout=failed_rollout)
    oracle = PreviewEnduranceValueOracle(policy, coordinates, horizon_cycles=10, moment_scale=1.)
    result = oracle.evaluate(coordinates.anchor)
    assert result.status == "policy_failed" and result.value is None
    assert result.completed_intervals == 2 and calls[0][1] == 10
    np.testing.assert_array_equal(calls[0][0], coordinates.decode(coordinates.anchor))


@pytest.mark.parametrize("setting,value", [
    ("preview_phases", 3), ("optimality_tolerance", 1e-5),
    ("recruitment_regularization", 2e-8), ("max_solve_time_s", 1.),
])
def test_changed_policy_is_rejected_before_rollout_and_tolerances_must_agree(setting, value):
    predictor, coordinates = _problem()
    policy = _stub_policy(predictor)
    with pytest.raises(ValueError, match="same moment tolerance"):
        PreviewEnduranceValueOracle(policy, coordinates, moment_scale=1., moment_tolerance=1e-7)
    oracle = PreviewEnduranceValueOracle(policy, coordinates, moment_scale=1.)
    setattr(policy, setting, value)
    result = oracle.evaluate(coordinates.anchor)
    assert result.status == "domain_invalid" and result.value is None
    assert "settings changed" in result.message
    assert oracle.value_context_metadata["allocation_policy"]["preview_phases"] == 2


def test_two_phase_oracle_can_fit_audited_polynomial_without_future_nlp_variables():
    from cocofest.optimization.preview_muscle_allocation import PreviewMuscleAllocation

    predictor, coordinates = _problem()
    policy = PreviewMuscleAllocation(predictor, preview_phases=2)
    oracle = PreviewEnduranceValueOracle(policy, coordinates, horizon_cycles=3, moment_scale=1.)
    # A small synthetic local box tests the adapter, not clinical latency or
    # an endurance benefit. Its informative-pair threshold is explicit.
    fit = fit_local_endurance_value(oracle, trust_radius=.0001, ranking_tolerance=1e-10)
    assert fit.accepted, fit.audit.reason
    binding = LocalEnduranceValueBinding(fit, coordinates, ["m0", "m1"])
    assert binding.parameter_size == 30
    assert binding.audit_terminal_state(coordinates.decode(coordinates.anchor))["valid"]
    assert fit.metadata["value_context"]["allocation_policy"]["preview_phases"] == 2
    assert fit.metadata["evaluation_backend"] == "scalar"
