import json

import numpy as np
import pytest

from scripts.benchmark_pace_rollout_integrator import (
    _candidate_weights,
    _compare,
    _ranking,
)
from cocofest.optimization.adaptive_moment_rollout import (
    DingPulseWidthParameters,
    MomentTrackingInterval,
)
from cocofest.optimization.batched_weighted_cycle_prediction import (
    BatchedWeightedCyclePredictor,
    solve_weighted_recruitment_many,
)
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.weighted_cycle_prediction import (
    WeightedCyclePredictor,
    solve_weighted_recruitment,
)


class _Result:
    def __init__(self, score=0.1, reserve=0.2):
        self.status = "complete"
        self.completed_intervals = 2
        self.state_history = np.asarray(
            [[[0.2, 1.0, 100.0, 0.06, 0.14]], [[0.1, 2.0, 90.0, 0.07, 0.15]], [[0.1, 3.0, 80.0, 0.08, 0.16]]]
        )
        self.pulse_widths = np.asarray([[[0.0002]], [[0.0003]]])
        self.full_horizon_normalized_deficit = score
        self.terminal_normalized_reserve = reserve


def test_candidate_weights_select_latest_or_requested_cycle(tmp_path):
    path = tmp_path / "pace.jsonl"
    rows = [
        {"event": "boundary", "cycle_index": 20, "predictive_audit": {
            "candidate_rollouts": [{"weights": [1, 2]}]}},
        {"event": "completed_window", "cycle_index": 20},
        {"event": "boundary", "cycle_index": 40, "predictive_audit": {
            "candidate_rollouts": [{"weights": [3, 4]}, {"weights": [5, 6]}]}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert _candidate_weights(path, None) == (40, [[3.0, 4.0], [5.0, 6.0]])
    assert _candidate_weights(path, 20) == (20, [[1.0, 2.0]])
    with pytest.raises(ValueError, match="cycle 60"):
        _candidate_weights(path, 60)


def test_comparison_metrics_and_score_ranking_are_explicit():
    reference = _Result(score=0.1, reserve=0.2)
    candidate = _Result(score=0.1001, reserve=0.21)
    candidate.state_history[-1, 0, 1] += 0.5
    candidate.state_history[-1, 0, 2] += 1.0
    candidate.pulse_widths[-1, 0, 0] += 2e-6
    comparison = _compare(candidate, reference, np.asarray([[100.0, 0.1, 0.2]]))
    assert comparison["maximum_force_error_n"] == pytest.approx(0.5)
    assert comparison["maximum_slow_state_error_over_rest"] == pytest.approx(0.01)
    assert comparison["maximum_pulse_width_error_us"] == pytest.approx(2.0)
    assert comparison["full_horizon_normalized_deficit_absolute_error"] == pytest.approx(1e-4)
    assert comparison["terminal_normalized_reserve_absolute_error"] == pytest.approx(0.01)
    assert _ranking([_Result(0.3), _Result(0.1), _Result(0.2)]) == [1, 2, 0]


def test_batched_weighted_allocator_matches_scalar_including_projection():
    rng = np.random.default_rng(921)
    batch, muscles = 51, 4
    intercept = rng.uniform(-0.2, 0.2, (batch, muscles))
    slope = rng.uniform(-2.0, 2.0, (batch, muscles))
    slope[::5, -1] = 0.0
    reference = intercept + slope * rng.uniform(-0.2, 1.2, (batch, muscles))
    weights = np.exp(rng.uniform(-2.0, 2.0, (batch, muscles)))
    actual = solve_weighted_recruitment_many(
        intercept, slope, reference, weights, project_infeasible=True
    )
    for row in range(batch):
        expected = solve_weighted_recruitment(
            intercept[row], slope[row], reference[row], weights[row]
        )
        assert actual.statuses[row] == expected.status
        if expected.normalized_recruitment is not None:
            np.testing.assert_allclose(
                actual.normalized_recruitment[row], expected.normalized_recruitment,
                atol=2e-11, rtol=1e-11,
            )


def _weighted_predictor(substeps=4):
    parameter = DingPulseWidthParameters(
        DingFatigueParameters(1200.0, 0.060601, 0.137, -1.4, 2.1e-5, 1.9e-5, 445.5),
        tauc=0.011, tau2=0.001, pd0=0.000131405, pdt=0.000194138,
        pulse_width_max=0.0006,
    )
    intervals = tuple(
        MomentTrackingInterval(
            1 / 30, (1.0597355478, 1.0597355478), (0.95, 0.95),
            (0.05, 0.05), targets,
        )
        for targets in ((0.9, 0.3), (100.0, 100.0))
    )
    return WeightedCyclePredictor(intervals, (parameter, parameter), substeps=substeps)


@pytest.mark.parametrize("tracking_mode", ["exact", "projected_capacity"])
def test_candidate_batch_matches_scalar_rollouts(tracking_mode):
    predictor = _weighted_predictor()
    initial = np.tile([0.24, 3.0, 1130.0, 0.081, 0.153], (2, 1))
    weights = np.asarray([[1.0, 1.0], [4.0, 0.25], [0.5, 2.0]])
    actual = BatchedWeightedCyclePredictor(predictor).rollout_many(
        initial, weights, horizon_cycles=5, tracking_mode=tracking_mode,
    )
    for row, candidate in enumerate(weights):
        expected = predictor.rollout(
            initial, candidate, horizon_cycles=5, tracking_mode=tracking_mode,
        )
        assert actual[row].status == expected.status
        assert actual[row].completed_intervals == expected.completed_intervals
        for name in (
            "state_history", "pulse_widths", "allocated_moments", "achieved_moments",
            "signed_moment_errors", "signed_margins", "weighted_force_integrals",
        ):
            np.testing.assert_allclose(
                getattr(actual[row], name), getattr(expected, name),
                atol=3e-10, rtol=2e-11, equal_nan=True,
            )
        assert actual[row].full_horizon_normalized_deficit == pytest.approx(
            expected.full_horizon_normalized_deficit, abs=1e-14
        )
        for name in ("full_horizon_mean_squared_fatigue", "first_block_mean_squared_fatigue",
                     "terminal_minimum_capacity"):
            assert getattr(actual[row], name) == pytest.approx(getattr(expected, name), abs=1e-12)
