import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.benchmark_cycle_preview_controller import (
    add_certified_speed_comparison,
    baseline_window_solver_times,
    compare_future_fatigue,
    preview_anchor_indices,
    rho_future_fatigue_boundaries,
    sanitize_initial_calcium,
    summarize_blocks,
)


def test_failed_prefix_never_receives_speedup_or_deadline_claim():
    record = {"preview_certified": False, "baseline_block_solver_time_s": 10.,
              "proposed_block_online_kernel_time_s": .1, "cycle_period_s": 1.}
    add_certified_speed_comparison(record)
    assert record["block_speedup"] is None
    assert record["online_time_saved_s"] is None
    assert record["first_cycle_deadline_passed"] is False
    record["preview_certified"] = True
    add_certified_speed_comparison(record)
    assert record["block_speedup"] == 100.
    assert record["first_cycle_deadline_passed"] is True


def test_baseline_times_require_validated_finite_windows():
    document = {"results": [{"windows": [
        {"validated": True, "solver_time_s": 1.0},
        {"validated": True, "solver_time_s": 2.0},
    ]}]}
    assert np.array_equal(baseline_window_solver_times(document), [1.0, 2.0])
    document["results"][0]["windows"][1]["validated"] = False
    with pytest.raises(ValueError, match="not validated"):
        baseline_window_solver_times(document)


def test_preview_blocks_are_disjoint_and_fit_horizon():
    assert preview_anchor_indices(17, 5, 20) == [0, 5, 10]
    assert preview_anchor_indices(17, 5, 20, anchor_stride=1) == list(range(13))
    assert preview_anchor_indices(17, 5, 3, anchor_stride=2) == [0, 2, 4]
    assert preview_anchor_indices(10, 5, 1) == [0]
    assert preview_anchor_indices(1, 5, 1) == [0]
    with pytest.raises(ValueError, match="fewer cycles"):
        preview_anchor_indices(4, 5, 1)
    with pytest.raises(ValueError, match="anchor_stride"):
        preview_anchor_indices(10, 5, 1, anchor_stride=0)


def test_only_cn_roundoff_is_sanitized():
    state = np.array([[-1e-12, 1., 2., 3., 4.]])
    assert sanitize_initial_calcium(state)[0, 0] == 0.
    with pytest.raises(ValueError, match="materially negative"):
        sanitize_initial_calcium(np.array([[-1e-6, 1., 2., 3., 4.]]))


def test_summary_excludes_uncertified_preview_from_speed_claim():
    blocks = [
        {"preview_certified": True, "baseline_block_solver_time_s": 5.0,
         "proposed_block_online_kernel_time_s": 1.0,
         "proposed_block_end_to_end_emulation_time_s": 1.1, "preview_online_overhead_s": .1,
         "block_speedup": 5.0, "first_cycle_deadline_passed": True,
         "full_ding_state_audit": {"minimum_capacity_ratio": .9},
         "maximum_total_crank_moment_error_nm": 1e-5,
         "maximum_individual_muscle_moment_error_nm": .2},
        {"preview_certified": False, "baseline_block_solver_time_s": 6.0,
         "proposed_block_online_kernel_time_s": 2.0,
         "proposed_block_end_to_end_emulation_time_s": 2.2, "preview_online_overhead_s": .2,
         "block_speedup": 3.0, "first_cycle_deadline_passed": False},
    ]
    summary = summarize_blocks(blocks)
    assert summary["preview_numerically_certified_blocks"] == 1
    assert summary["five_cycle_speedup"]["mean_s"] == 5.0
    assert summary["baseline_five_cycle_solver_time"]["mean_s"] == 5.5


def test_fatigue_comparison_uses_shared_boundary_and_preserves_three_indicators():
    parameters = (
        SimpleNamespace(fatigue=SimpleNamespace(rest_state=np.array([100., 2., 3.]))),
    )
    # Cn/F are irrelevant here; A, Tau1 and Km form the slow-state comparison.
    full = np.array([
        [[0., 0., 100., 2., 3.]],
        [[0., 0., 99., 2.1, 3.1]],
        [[0., 0., 98., 2.2, 3.2]],
    ])
    normal = np.array([
        [[100., 2., 3.]],
        [[99.5, 2.05, 3.05]],
        [[99., 2.1, 3.1]],
    ])
    report, arrays = compare_future_fatigue(full, normal, parameters, intervals_per_cycle=1)
    assert report["initial_boundary_maximum_absolute_normalized_difference"] == 0.
    terminal = report["terminal_preview_minus_normal_rho"]
    assert terminal["mean_capacity_ratio"] == pytest.approx(-.01)
    assert terminal["mean_tau1_ratio"] == pytest.approx(2.2 / 2 - 2.1 / 2)
    assert terminal["mean_km_ratio"] == pytest.approx(3.2 / 3 - 3.1 / 3)
    assert terminal["muscles_with_any_worse_terminal_fatigue_indicator"] == 1
    assert arrays["preview_minus_normal_rho_fatigue_ratio"].shape == (3, 1, 3)


def test_future_rho_boundaries_start_at_anchor_end_then_advance_whole_cycles():
    states = {}
    for name, offset in (("A_m", 0.), ("Tau1_m", 100.), ("Km_m", 200.)):
        states[name] = np.arange(13, dtype=float) + offset
    cycle = SimpleNamespace(
        end_column=4,
        stimulations_per_cycle=2,
        state_columns_per_interval=2,
        states=states,
    )
    values = rho_future_fatigue_boundaries(cycle, ("m",), horizon_cycles=2)
    np.testing.assert_array_equal(values[:, 0, 0], [4., 8., 12.])
    np.testing.assert_array_equal(values[:, 0, 1], [104., 108., 112.])
