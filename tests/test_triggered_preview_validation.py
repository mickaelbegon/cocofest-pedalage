from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validate_triggered_preview_allocation import (
    CODE_PATHS,
    ROOT,
    _trajectory_difference,
    _trigger_summary,
    _validate_batch_history,
)


def _compatible_reports():
    local_cases = {}
    batch_cases = []
    expected_contexts = {}
    for anchor, scale in ((0, 0.6), (112, 0.54)):
        radius = [0.01] * 8
        context = {
            "task_sha256": f"task-{anchor}",
            "horizon_cycles": 10,
            "moment_scale": scale,
            "moment_tolerance": 1e-8,
            "softmin_temperature": 0.05,
            "margin_target": 0.1,
            "penalty_temperature": 0.05,
            "tracking_band_nm": 0.0,
            "tracking_penalty_weight": 1.0,
            "tracking_cost_definition": "duration_mean_squared_original_error_over_task_scale_v1",
        }
        expected_contexts[anchor] = context.copy()
        local_cases[anchor] = {
            "moment_scale_nm": scale,
            "fit_attempts": [{"accepted": True, "trust_radius": radius}],
        }
        batch_cases.append({
            "anchor_zero_based": anchor,
            "horizon_cycles": 10,
            "timed_point_count": 41,
            "value_context": context,
            "fit_attempts": [{
                "batch_accepted": True,
                "trust_radius": radius.copy(),
                "elapsed_s": 0.64,
            }],
        })
    report = {
        "schema": "cocofest-batched-endurance-value-validation-v1",
        "uses_fho_data": False,
        "source": {"sha256": "source"},
        "reduced_profile": {"sha256": "profile"},
        "baseline_report": {"sha256": "local"},
        "strict_batch_scalar": batch_cases,
    }
    return report, local_cases, expected_contexts


def test_historical_batch_report_requires_matching_radii_and_context():
    report, local_cases, expected_contexts = _compatible_reports()
    history = _validate_batch_history(
        report,
        report_sha256="batch",
        local_baseline_sha256="local",
        source_sha256="source",
        profile_sha256="profile",
        local_cases=local_cases,
        expected_value_contexts=expected_contexts,
    )
    assert history[0]["elapsed_s"] == 0.64
    assert history[112]["point_count"] == 41


@pytest.mark.parametrize(
    "mutation", ("source", "radius", "point_count", "tracking_band", "task_sha", "elapsed")
)
def test_historical_batch_report_rejects_incompatible_input(mutation):
    report, local_cases, expected_contexts = _compatible_reports()
    if mutation == "source":
        report["source"]["sha256"] = "wrong"
    elif mutation == "radius":
        report["strict_batch_scalar"][0]["fit_attempts"][0]["trust_radius"][0] = 0.02
    elif mutation == "point_count":
        report["strict_batch_scalar"][0]["timed_point_count"] = 40
    elif mutation == "tracking_band":
        report["strict_batch_scalar"][0]["value_context"]["tracking_band_nm"] = 1e-4
    elif mutation == "task_sha":
        report["strict_batch_scalar"][0]["value_context"]["task_sha256"] = "wrong"
    else:
        report["strict_batch_scalar"][0]["fit_attempts"][0]["elapsed_s"] = np.inf
    with pytest.raises(ValueError, match="Incompatible batched historical report"):
        _validate_batch_history(
            report,
            report_sha256="batch",
            local_baseline_sha256="local",
            source_sha256="source",
            profile_sha256="profile",
            local_cases=local_cases,
            expected_value_contexts=expected_contexts,
        )


def test_trigger_summary_counts_trigger_paths_and_uses_signed_margin():
    result = SimpleNamespace(
        diagnostics=(
            {"screen_reason": "next_margin_above_threshold", "next_signed_margin_nm": 0.02},
            {"screen_reason": "next_margin_at_or_below_threshold", "next_signed_margin_nm": -0.01},
        ),
        screenings=2,
        triggers=1,
        greedy_fast_path_steps=1,
        screen_failures=0,
        greedy_fallback_steps=0,
        screening_time_s=0.001,
        triggered_preview_time_s=0.01,
        total_time_s=0.02,
    )
    summary = _trigger_summary(result)
    assert summary["triggers"] == 1
    assert summary["minimum_screen_margin_nm"] == -0.01
    assert summary["screen_reason_counts"]["next_margin_at_or_below_threshold"] == 1


def test_trajectory_difference_rejects_nonfinite_common_prefix():
    finite = SimpleNamespace(
        completed_intervals=1,
        pulse_widths=np.array([[[0.2]]]),
        state_history=np.ones((2, 1, 5)),
    )
    nonfinite = SimpleNamespace(
        completed_intervals=1,
        pulse_widths=np.array([[[np.inf]]]),
        state_history=np.ones((2, 1, 5)),
    )
    parameter = SimpleNamespace(fatigue=SimpleNamespace(rest_state=np.ones(3)))
    with pytest.raises(ValueError, match="Nonfinite"):
        _trajectory_difference(nonfinite, finite, (parameter,))


def test_trajectory_difference_reports_separate_state_scales():
    reference = SimpleNamespace(
        completed_intervals=1,
        pulse_widths=np.array([[[0.2]]]),
        state_history=np.ones((2, 1, 5)),
    )
    candidate = SimpleNamespace(
        completed_intervals=1,
        pulse_widths=np.array([[[0.3]]]),
        state_history=np.array([[[2., 3., 5., 7., 9.]], [[2., 3., 5., 7., 9.]]]),
    )
    parameter = SimpleNamespace(fatigue=SimpleNamespace(rest_state=np.array([2., 3., 4.])))
    difference = _trajectory_difference(candidate, reference, (parameter,))
    assert difference["maximum_absolute_pw_difference_s"] == pytest.approx(0.1)
    assert difference["maximum_absolute_cn_difference"] == 1.0
    assert difference["maximum_absolute_force_difference_n"] == 2.0
    assert difference["maximum_normalized_slow_state_difference"] == 2.0


def test_trajectory_difference_flattens_cycle_phase_before_muscle_and_ignores_suffix():
    reference = SimpleNamespace(
        completed_intervals=3,
        pulse_widths=np.zeros((2, 2, 2)),
        state_history=np.ones((4, 2, 5)),
    )
    candidate = SimpleNamespace(
        completed_intervals=3,
        pulse_widths=np.array([
            [[0.1, 0.2], [0.11, 0.21]],
            [[0.3, np.nan], [0.31, np.nan]],
        ]),
        state_history=np.ones((4, 2, 5)),
    )
    parameter = SimpleNamespace(fatigue=SimpleNamespace(rest_state=np.ones(3)))
    difference = _trajectory_difference(candidate, reference, (parameter, parameter))
    assert difference["common_completed_intervals"] == 3
    assert difference["maximum_absolute_pw_difference_s"] == 0.31


def test_all_hashed_code_paths_exist():
    missing = [name for name in CODE_PATHS if not (ROOT / name).is_file()]
    assert missing == []
