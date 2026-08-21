import json

import numpy as np

from examples.fes_multibody.cycling.build_terminal_set_profile import (
    FULL_TURN,
    build_terminal_set_profile,
)


def _write_prefix(
    path,
    *,
    load_nm,
    omega_offset=0.0,
    include_scales=True,
    start_cycle=0,
):
    stimulations = 2
    cycles = 3
    nodes = cycles * stimulations + 1
    theta = -FULL_TURN * start_cycle - np.pi * np.arange(nodes, dtype=float)
    theta[-1] += 1e-3
    metadata = {
        "producer_mode": "receding_horizon_concatenation",
        "producer_solver": "ipopt",
        "producer_collocation_degree": 5,
        "cycles_per_window": cycles,
        "stimulations_per_cycle": stimulations,
        "constant_crank_torque": load_nm,
        "absolute_wheel_q_origin_reference": 0.0,
        "absolute_wheel_q_start_cycle_index": start_cycle,
    }
    if include_scales:
        metadata["fatigue_capacity_scales"] = {
            "A_Biceps": 100.0,
            "A_Triceps": 80.0,
        }
    np.savez(
        path,
        states__theta=theta[None, :],
        states__omega=np.linspace(-6.2, -6.4, nodes)[None, :] + omega_offset,
        states__A_Biceps=np.linspace(100.0, 76.0, nodes)[None, :],
        states__A_Triceps=np.linspace(80.0, 72.0, nodes)[None, :],
        controls__pulse_width=np.full((1, nodes - 1), 2e-4),
        metadata__json=np.asarray(json.dumps(metadata)),
    )


def test_terminal_set_profile_uses_absolute_angle_and_reports_coverage(tmp_path):
    first = tmp_path / "load-0.npz"
    second = tmp_path / "load-015.npz"
    _write_prefix(first, load_nm=0.0)
    _write_prefix(second, load_nm=0.15, omega_offset=-0.1)

    profile = build_terminal_set_profile([first, second])

    assert profile["absolute_angle_target"] == "theta_initial_minus_2pi_times_cycle"
    assert profile["coverage"]["distinct_loads_nm"] == [0.0, 0.15]
    assert profile["coverage"]["boundary_sample_count"] == 8
    assert profile["coverage"]["certification_ready"] is False
    assert profile["coverage"]["legacy_capacity_sources"] == []
    assert profile["coverage"]["legacy_angle_sources"] == []
    assert profile["status"] == "diagnostic_only"
    assert np.isclose(
        max(
            row["theta_phase_error_rad"]["observed_max"]
            for row in profile["bins"]
        ),
        1e-3,
    )


def test_legacy_capacity_normalization_cannot_certify_terminal_set(tmp_path):
    legacy = tmp_path / "legacy.npz"
    _write_prefix(legacy, load_nm=0.0, include_scales=False)

    profile = build_terminal_set_profile([legacy])

    assert profile["status"] == "diagnostic_only"
    assert profile["coverage"]["certification_ready"] is False
    assert profile["coverage"]["legacy_capacity_sources"] == [
        str(legacy.resolve())
    ]
    assert (
        profile["sources"][0]["capacity_normalization"]
        == "legacy_source_initial_diagnostic_only"
    )


def test_replay_uses_global_cycle_index_for_absolute_angle(tmp_path):
    replay = tmp_path / "replay.npz"
    _write_prefix(replay, load_nm=0.15, start_cycle=7)

    profile = build_terminal_set_profile([replay])

    assert profile["sources"][0]["angle_reference"] == "global_absolute_cycle"
    assert np.isclose(
        max(
            row["theta_phase_error_rad"]["observed_max"]
            for row in profile["bins"]
        ),
        1e-3,
    )
