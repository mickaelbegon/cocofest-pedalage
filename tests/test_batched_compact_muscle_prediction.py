import numpy as np
import pytest

from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval
from cocofest.optimization.batched_compact_muscle_prediction import (
    BatchedCompactMusclePredictor, solve_bounded_moment_qp_many,
)
from cocofest.optimization.compact_muscle_prediction import CompactMusclePredictor
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.smooth_muscle_moment_allocation import solve_bounded_moment_qp_reference


def predictor(coefficients=(.05, -.03), targets=(.7, -.2), substeps=16):
    p = DingPulseWidthParameters(
        DingFatigueParameters(1200., .060601, .137, -1.4, 2.1e-5, 1.9e-5, 445.5),
        tauc=.011, tau2=.001, pd0=.000131405, pdt=.000194138, pulse_width_max=.0006,
    )
    intervals = tuple(MomentTrackingInterval(1 / 30, (1.0597355478,) * len(targets),
                                            (.95,) * len(targets), coefficients, targets) for _ in range(3))
    return CompactMusclePredictor(intervals, (p,) * len(targets), substeps=substeps)


@pytest.mark.parametrize("scale", [1e-9, 1e-3, 1., 1e6])
def test_batched_sorted_qp_matches_scalar_signed_random_and_degenerate_bounds(scale):
    rng = np.random.default_rng(910)
    lower = scale * rng.uniform(-2., 1., (101, 4))
    upper = lower + scale * rng.uniform(0., 2., lower.shape)
    upper[::3, 0] = lower[::3, 0]
    upper[0] = lower[0]
    reference = scale * rng.normal(size=lower.shape)
    target = lower.sum(axis=1) + rng.uniform(size=101) * (upper - lower).sum(axis=1)
    tolerance = max(1e-15, scale * 1e-10)
    batch = solve_bounded_moment_qp_many(reference, required_total_moment=target,
                                        lower_bounds=lower, upper_bounds=upper,
                                        feasibility_tolerance=tolerance)
    assert set(batch.statuses) == {"ok"}
    assert np.max(np.abs(batch.allocated_moments.sum(axis=1) - target)) <= tolerance
    for row in range(len(target)):
        scalar = solve_bounded_moment_qp_reference(reference[row], required_total_moment=target[row],
                                                   lower_bounds=lower[row], upper_bounds=upper[row],
                                                   feasibility_tolerance=tolerance)
        assert scalar.status == "ok"
        np.testing.assert_allclose(batch.allocated_moments[row], scalar.allocated_moments,
                                   atol=max(2e-15, scale * 2e-9), rtol=1e-8)


def test_qp_tracking_band_projects_only_outside_and_never_changes_feasible_target():
    lower, upper = np.array([[-1., 0.]] * 5), np.array([[0., 2.]] * 5)
    target = np.array([.4, -1.00005, 2.00005, -1.0002, 2.0002])
    result = solve_bounded_moment_qp_many([-.2, .5], required_total_moment=target,
                                         lower_bounds=lower, upper_bounds=upper, tracking_band_nm=.0001)
    assert result.statuses == ("ok", "ok", "ok", "infeasible_total_below_bounds", "infeasible_total_above_bounds")
    np.testing.assert_allclose(result.effective_total_targets[:3], [.4, -1., 2.])
    np.testing.assert_array_equal(result.relaxed, [False, True, True, False, False])
    np.testing.assert_allclose(result.allocated_moments[:3].sum(axis=1), [.4, -1., 2.], atol=1e-12)
    assert np.all(np.isnan(result.allocated_moments[3:]))
    strict = solve_bounded_moment_qp_many([-.2, .5], required_total_moment=target,
                                         lower_bounds=lower, upper_bounds=upper)
    assert strict.statuses[1] == "infeasible_total_below_bounds"


def test_qp_invalid_rows_do_not_kill_valid_rows():
    result = solve_bounded_moment_qp_many([0., 0.], required_total_moment=[1., 1., np.nan],
                                         lower_bounds=[[0., 0.], [2., 0.], [0., 0.]],
                                         upper_bounds=[[1., 1.], [1., 1.], [1., 1.]])
    assert result.statuses == ("ok", "invalid_bounds_or_target", "invalid_bounds_or_target")
    np.testing.assert_allclose(result.allocated_moments[0], [.5, .5])


def test_scalar_tolerance_parity_at_endpoints_is_distinct_from_physical_band():
    tolerance = 1e-8
    targets = np.array([-.5 * tolerance, .5 * tolerance, 1 + .5 * tolerance,
                        1 - .5 * tolerance, 1 + 2 * tolerance])
    lower, upper = np.zeros((5, 2)), np.full((5, 2), .5)
    result = solve_bounded_moment_qp_many([.2, .3], required_total_moment=targets,
                                         lower_bounds=lower, upper_bounds=upper,
                                         feasibility_tolerance=tolerance)
    assert result.statuses == ("ok", "ok", "ok", "ok", "infeasible_total_above_bounds")
    assert not np.any(result.relaxed)
    for row in range(4):
        reference = solve_bounded_moment_qp_reference(
            [.2, .3], required_total_moment=targets[row], lower_bounds=lower[row],
            upper_bounds=upper[row], feasibility_tolerance=tolerance)
        np.testing.assert_array_equal(result.allocated_moments[row], reference.allocated_moments)
    band = solve_bounded_moment_qp_many(
        [.2, .3], required_total_moment=[1 + 1e-4 + .5 * tolerance, 1 + 1e-4 + 2 * tolerance],
        lower_bounds=lower[:2], upper_bounds=upper[:2], tracking_band_nm=1e-4,
        feasibility_tolerance=tolerance)
    assert band.statuses == ("ok", "infeasible_total_above_bounds")
    assert band.relaxed[0]


@pytest.mark.parametrize("substeps", [1, 4, 16])
def test_batched_phase_maps_match_every_scalar_field(substeps):
    scalar = predictor(substeps=substeps)
    batch = BatchedCompactMusclePredictor(scalar)
    rng = np.random.default_rng(11)
    states = np.tile([.16298, 20., 1150., .078, .15], (17, 2, 1)) * rng.uniform(.9, 1.1, (17, 2, 5))
    maps = batch.phase_map_many(states, 1)
    for row in range(len(states)):
        direct = scalar.phase_map(states[row], 1)
        for name in ("intercept", "slope", "weighted_force_intercept", "weighted_force_slope"):
            np.testing.assert_allclose(getattr(maps, name)[row], getattr(direct, name), rtol=1e-14, atol=1e-13)


@pytest.mark.parametrize("coefficients,targets", [((.05, -.03), (.7, -.2)), ((-.05, .03), (-.7, .2)),
                                                ((.05, 0.), (.8, 0.)), ((0., 0.), (0., 0.))])
def test_batched_rollouts_match_scalar_and_expose_original_signed_moment_diagnostics(coefficients, targets):
    scalar = predictor(coefficients, targets)
    batch = BatchedCompactMusclePredictor(scalar)
    states = np.tile([.16298, 20., 1150., .078, .15], (7, 2, 1))
    states[:, :, 2] += np.linspace(-20., 20., 7)[:, None]
    states[:, :, 1] += np.linspace(-1., 1., 7)[:, None]
    results = batch.rollout_many(states, horizon_cycles=5)
    for row, result in enumerate(results):
        direct = scalar.rollout(states[row], horizon_cycles=5)
        assert result.status == direct.status == "complete"
        assert result.completed_intervals == direct.completed_intervals == 15
        for name in ("state_history", "pulse_widths", "allocated_moments", "achieved_moments"):
            np.testing.assert_allclose(getattr(result, name), getattr(direct, name), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(result.signed_moment_errors,
                                   result.achieved_moments.sum(axis=1) - sum(targets), atol=1e-15)
        np.testing.assert_allclose(result.original_total_moments, sum(targets))
        assert result.relaxed_steps == 0
        assert result.tracking_mode == "exact_with_numerical_tolerance"
        assert np.all(result.envelope_domain_valid)


def test_failed_candidates_remain_failed_while_other_candidates_finish():
    scalar = predictor()
    states = np.tile([.16298, 20., 1150., .078, .15], (4, 2, 1))
    states[0, 0, 2] = -1.
    states[1, :, 1] = [1000., 0.]
    results = BatchedCompactMusclePredictor(scalar).rollout_many(states, horizon_cycles=3)
    assert results[0].first_failure["status"] == "initial_state_outside_domain"
    assert results[1].first_failure["status"] == "infeasible_total_below_bounds"
    assert results[0].completed_intervals == results[1].completed_intervals == 0
    assert np.all(np.isnan(results[1].state_history[1:]))
    assert np.all(np.isnan(results[0].pulse_widths))
    assert results[2].status == results[3].status == "complete"


def test_physical_tracking_band_reports_original_error_and_relaxed_count():
    scalar = predictor((.05,), (0.,))
    state = np.array([[[.16298, .001, 1150., .078, .15]]])
    phase = scalar.phase_map(state[0], 0)
    minimum = .05 * phase.intercept[0, 1]
    batch = BatchedCompactMusclePredictor(scalar)
    strict = batch.rollout_many(state, horizon_cycles=1)[0]
    assert strict.status == "infeasible"
    relaxed = batch.rollout_many(state, horizon_cycles=1, tracking_band_nm=minimum * 1.001)[0]
    assert relaxed.status == "complete"
    assert relaxed.relaxed_steps == 3
    assert relaxed.tracking_mode == "bounded_tracking"
    assert np.all(relaxed.relaxed_step_mask)
    assert np.max(relaxed.signed_moment_errors) == pytest.approx(minimum)
    assert np.all(relaxed.original_total_moments == 0.)
    np.testing.assert_allclose(relaxed.effective_total_targets, relaxed.total_lower_bounds)


@pytest.mark.parametrize("option,value", [("horizon_cycles", 0), ("moment_tolerance", 0.),
                                         ("tracking_band_nm", -1.), ("tracking_band_nm", np.nan)])
def test_invalid_rollout_options(option, value):
    batch = BatchedCompactMusclePredictor(predictor())
    kwargs = {"horizon_cycles": 1, option: value}
    with pytest.raises(ValueError):
        batch.rollout_many(np.ones((1, 2, 5)), **kwargs)
