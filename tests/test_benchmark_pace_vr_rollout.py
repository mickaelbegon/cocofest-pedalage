from dataclasses import asdict
import json

import numpy as np
import pytest

from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.pace_vr import PaceVrConfig, PaceVrSupervisor, create_pace_vr_snapshot
from scripts.benchmark_pace_vr_rollout import benchmark, load_case


def _case():
    parameters = tuple(DingPulseWidthParameters(
        DingFatigueParameters(1200., .060601, .137, alpha, 2.1e-5, 1.9e-5, 445.5),
        .011, .001, .000131405, .000194138, .0006) for alpha in (-1.4, -2.0))
    intervals = (MomentTrackingInterval(1 / 30, (1.0597355478,) * 2, (.95, .85), (.05, .03), (0., 0.)),) * 3
    supervisor = PaceVrSupervisor(intervals=intervals, pulse_width_parameters=parameters,
        angular_velocity_rad_s=1., required_work_j=.07,
        config=PaceVrConfig(horizon_cycles=2, phase_knots=2, fit_max_samples=1))
    states = np.array([[.16298215835, 10., 1150., .067, .15], [.16298215835, 10., 1100., .068, .151]])
    widths = np.full((2, 3), .00030)
    return supervisor, states, widths


def test_archived_snapshot_requires_same_digest_and_certified_cycle(tmp_path):
    supervisor, states, widths = _case()
    snapshot = asdict(create_pace_vr_snapshot(supervisor, states, widths, request_id="test",
        source_cycle=20, deadline_seconds=10., certified=True))
    path = tmp_path / "result.json"
    document = {"cycles": [{"cycle": 20, "certified": True, "pace_vr_snapshot": snapshot}]}
    path.write_text(json.dumps(document))
    restored, payload, checked = load_case(path, 20)
    assert restored.config.horizon_cycles == 2
    np.testing.assert_array_equal(payload["initial_states"], states)
    assert checked["context_digest"] == snapshot["context_digest"]
    document["cycles"][0]["certified"] = False
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="certified"):
        load_case(path, 20)
    snapshot["context_digest"] = "wrong"
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="digest"):
        load_case(path, 20)


def test_benchmark_compares_same_schedules_and_independent_dop853():
    supervisor, states, widths = _case()
    report, outcomes = benchmark(supervisor, {"initial_states": states, "pulse_widths": widths},
                                 dop853_cycles=2)
    assert report["prefixes"] == {"reference": 2, "optimized": 2}
    assert report["timing_comparison_valid"]
    assert report["fits"]["optimized"]["accepted"]
    assert not report["physiology_failure_certified"]
    assert report["same_pw_force_substep_error_max"] < 1e-10
    assert report["same_pw_work_error_max"] < 1e-12
    assert report["dop853"]["force_substep_samples"] == 2 * 3 * 8
    assert report["dop853"]["force_substep_peak_normalized_error"] < .01
    assert report["dop853"]["required_work_relative_error_max"] < .01
    json.dumps(report, allow_nan=False)


def test_no_speedup_claim_when_no_projected_cycle_is_validated():
    supervisor, states, widths = _case()
    supervisor.required_work_j = 100.
    report, _ = benchmark(supervisor, {"initial_states": states, "pulse_widths": widths}, fit=False)
    assert report["prefixes"] == {"reference": 0, "optimized": 0}
    assert not report["timing_comparison_valid"]
    assert report["speedup"] is None
    assert not report["trajectory_comparison_available"]
