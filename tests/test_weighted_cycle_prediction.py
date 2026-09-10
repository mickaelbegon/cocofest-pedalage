from dataclasses import replace
from itertools import product
import json

import numpy as np
import pytest

from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval
from cocofest.optimization.compact_muscle_prediction import (
    CompactMusclePredictor, fatigue_memory_coordinates,
)
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.weighted_cycle_prediction import (
    POLICY_NAME, WeightedCyclePredictor, solve_weighted_recruitment,
)


def parameters():
    return DingPulseWidthParameters(
        DingFatigueParameters(1200., .060601, .137, -1.4, 2.1e-5, 1.9e-5, 445.5),
        tauc=.011, tau2=.001, pd0=.000131405, pdt=.000194138, pulse_width_max=.0006,
    )


def predictor(coefficients=(.05, .05), targets=(.9, .3), phases=2, parameters_override=None):
    p = parameters() if parameters_override is None else parameters_override
    intervals = tuple(MomentTrackingInterval(
        1 / 30, (1.0597355478,) * len(targets), (.95,) * len(targets), coefficients, targets,
    ) for _ in range(phases))
    return WeightedCyclePredictor(intervals, (p,) * len(targets))


def states(count=2):
    return np.tile([.24, 3., 1130., .081, .153], (count, 1))


def independent_active_set_optimum(intercept, slope, reference, weights, epsilon):
    """Enumerate every active set, solving its equality KKT system directly."""
    xref = np.clip(np.divide(reference - intercept, slope,
                            out=np.zeros_like(slope), where=slope != 0), 0, 1)
    h = weights / weights.mean() + epsilon
    g = -epsilon * xref
    rhs = reference.sum() - intercept.sum()
    best, best_cost = None, np.inf
    for flags in product((-1, 0, 1), repeat=len(slope)):
        flags = np.asarray(flags)
        free = flags == -1
        x = np.maximum(flags, 0).astype(float)
        if np.any(free):
            a = slope[free]
            kkt = np.block([[np.diag(h[free]), a[:, None]], [a[None, :], np.zeros((1, 1))]])
            vector = np.r_[-g[free], rhs - np.dot(slope[~free], x[~free])]
            solution, _, _, _ = np.linalg.lstsq(kkt, vector, rcond=None)
            if np.max(np.abs(kkt @ solution - vector)) > 1e-9:
                continue
            x[free] = solution[:-1]
        if np.any(x < -1e-9) or np.any(x > 1 + 1e-9) or abs(slope @ x - rhs) > 1e-9:
            continue
        cost = .5 * np.dot(h, x*x) + np.dot(g, x)
        if cost < best_cost:
            best, best_cost = x, cost
    return best


def test_signed_weighted_qp_matches_independent_convex_optimum_including_active_bounds():
    rng = np.random.default_rng(61)
    for _ in range(30):
        intercept = rng.uniform(-.2, .2, 4)
        slope = rng.uniform(-2., 2., 4)
        slope[-1] = 0.
        reference = intercept + slope * rng.uniform(0, 1, 4)
        weights = np.exp(rng.uniform(-2., 2., 4))
        actual = solve_weighted_recruitment(intercept, slope, reference, weights)
        expected = independent_active_set_optimum(intercept, slope, reference, weights, 1e-3)
        assert actual.status == "ok"
        np.testing.assert_allclose(actual.normalized_recruitment, expected, atol=2e-9)
        assert abs(actual.equality_residual) < 1e-10


@pytest.mark.parametrize("scale", [1e-200, 1e-6, 1., 1e150])
def test_common_weight_scale_invariance_including_reference_regularization(scale):
    args = ([.1, -.1, 0.], [2., -1., 0.], [.8, -.1, 0.])
    original = solve_weighted_recruitment(*args, [1., 4., 2.])
    scaled = solve_weighted_recruitment(*args, np.array([1., 4., 2.]) * scale)
    np.testing.assert_allclose(scaled.normalized_recruitment, original.normalized_recruitment, atol=1e-14)
    assert scaled.objective == pytest.approx(original.objective)


def test_weights_change_force_and_pw_when_reference_itself_is_feasible():
    p = predictor(phases=1)
    initial = states()
    ones = p.rollout(initial, [1., 1.], horizon_cycles=1)
    biased = p.rollout(initial, [9., 1.], horizon_cycles=1)
    old = CompactMusclePredictor(p.intervals, p.parameters).rollout(initial, horizon_cycles=1)
    assert ones.completed and biased.completed and old.status == "complete"
    np.testing.assert_allclose(old.achieved_moments[0, :, 0], [.9, .3], atol=1e-8)
    assert biased.pulse_widths[0, 0, 0] < ones.pulse_widths[0, 0, 0]
    assert biased.achieved_moments[0, 0, 0] < ones.achieved_moments[0, 0, 0]
    assert biased.achieved_moments[0, 1, 0] > ones.achieved_moments[0, 1, 0]
    assert not np.allclose(ones.achieved_moments, old.achieved_moments)
    assert ones.metadata["policy"] == POLICY_NAME
    assert not ones.metadata["all_ones_is_old_greedy"]
    json.dumps(ones.metadata, allow_nan=False)


@pytest.mark.parametrize("coefficients,targets", [((.05, -.03), (.7, -.2)),
                                                ((-.05, .03), (-.7, .2)),
                                                ((.05, 0.), (.5, 0.)),
                                                ((0., 0.), (0., 0.))])
def test_original_signed_constraint_exact_pw_bounds_and_zero_arms(coefficients, targets):
    p = predictor(coefficients, targets)
    result = p.rollout(states(), [2., 1.], horizon_cycles=3)
    assert result.completed
    assert result.pulse_widths.shape == (3, 2, 2)
    np.testing.assert_allclose(result.achieved_moments.sum(axis=1), sum(targets), atol=1e-8)
    np.testing.assert_allclose(result.original_total_moments, sum(targets))
    np.testing.assert_allclose(result.signed_moment_errors, 0., atol=1e-8)
    assert np.all(result.pulse_widths >= p.pd0[None, :, None])
    assert np.all(result.pulse_widths <= p.pulse_width_max[None, :, None])
    assert np.all(result.weighted_force_integrals >= 0)
    if coefficients[1] == 0:
        np.testing.assert_allclose(result.pulse_widths[:, 1], p.pd0[1])
        assert result.state_history[1, 1, 1] < result.state_history[0, 1, 1]
        assert result.state_history[1, 1, 1] > 0.


def test_cycle_summary_and_phase_map_preserve_actual_cn_and_off_manifold_offsets():
    p = predictor()
    initial = states()
    result = p.rollout(initial, [1., 3.], horizon_cycles=4)
    assert result.completed
    cycle_duration = sum(i.duration for i in p.intervals)
    for cycle in range(4):
        initial_slow = result.state_history[cycle * 2, :, 2:]
        final_slow = result.state_history[(cycle + 1) * 2, :, 2:]
        expected = p.rest + np.exp(-cycle_duration / p.tau_fat)[:, None] * (initial_slow - p.rest)
        expected += result.cycle_fatigue_forcing[cycle]
        np.testing.assert_allclose(final_slow, expected, atol=1e-12)
    _, initial_offsets = fatigue_memory_coordinates(initial[0, 2:], p.parameters[0].fatigue)
    _, final_offsets = fatigue_memory_coordinates(result.state_history[-1, 0, 2:], p.parameters[0].fatigue)
    assert np.max(np.abs(initial_offsets)) > 1e-3
    np.testing.assert_allclose(final_offsets, initial_offsets * np.exp(-4 * cycle_duration / p.tau_fat[0]),
                               atol=1e-14)
    expected_cn = initial[:, 0].copy()
    for _ in range(4):
        for phase in p.intervals:
            expected_cn = np.exp(-phase.duration / p.tauc) * (
                expected_cn + np.asarray(phase.calcium_amplitudes) * phase.duration / p.tauc)
    np.testing.assert_allclose(result.state_history[-1, :, 0], expected_cn)
    assert np.all(expected_cn > 0.)


def test_unreachable_second_phase_keeps_only_verified_prefix_and_negative_margin():
    p = predictor()
    p = WeightedCyclePredictor((p.intervals[0], replace(p.intervals[1], target_moments=(100., 100.))),
                               p.parameters)
    result = p.rollout(states(), [1., 1.], horizon_cycles=4)
    assert result.status == "infeasible"
    assert result.completed_intervals == 1 and result.completed_cycles == 0
    assert result.completed_duration == pytest.approx(1 / 30)
    assert result.first_failure["interval_index"] == 1
    assert result.minimum_signed_margin < 0
    assert np.all(np.isfinite(result.state_history[:2]))
    assert np.all(np.isnan(result.state_history[2:]))
    assert np.all(np.isnan(result.pulse_widths[:, :, 1]))
    assert np.all(np.isnan(result.cycle_weighted_force_integrals))


def test_nonphysical_initial_or_predicted_states_refused_without_clipping():
    p = predictor()
    invalid = states()
    invalid[0, 2] = -1.
    result = p.rollout(invalid, [1., 1.], horizon_cycles=3)
    assert result.first_failure["status"] == "initial_state_outside_domain"
    assert result.completed_intervals == 0
    bad = replace(parameters(), fatigue=replace(parameters().fatigue, alpha_a=-1e7))
    result = predictor(parameters_override=bad).rollout(states(), [1., 1.], horizon_cycles=3)
    assert result.first_failure["status"] == "envelope_domain_invalid"
    assert result.completed_intervals == 0
    assert np.isnan(result.minimum_signed_margin)
    assert np.all(np.isnan(result.state_history[1:]))


@pytest.mark.parametrize("weights", [[0., 1.], [-1., 1.], [np.nan, 1.], [1.]])
def test_invalid_weights_refused(weights):
    with pytest.raises(ValueError, match="weights"):
        predictor().rollout(states(), weights, horizon_cycles=1)


def test_numerical_solver_failure_is_not_reported_as_physiological_infeasibility(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate root failure")

    monkeypatch.setattr("cocofest.optimization.weighted_cycle_prediction.brentq", fail)
    result = predictor().rollout(states(), [1., 1.], horizon_cycles=2)
    assert result.status == "numerical_failure"
    assert result.first_failure["status"] == "allocation_numerical_failure"
    assert result.completed_intervals == 0
    assert np.all(np.isnan(result.pulse_widths))


@pytest.mark.parametrize("target,expected", [(-1., [0., 1., 0.]), (2., [1., 0., 0.])])
def test_signed_exact_reachable_endpoints(target, expected):
    result = solve_weighted_recruitment([0., 0., 0.], [2., -1., 0.], [target, 0., 0.], [1., 2., 3.])
    assert result.status == "ok"
    np.testing.assert_array_equal(result.normalized_recruitment, expected)
    assert result.signed_margin == 0.
