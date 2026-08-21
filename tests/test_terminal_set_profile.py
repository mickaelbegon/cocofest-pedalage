import json

import numpy as np

from examples.fes_multibody.cycling.build_terminal_set_profile import (
    build_terminal_set_profile,
)


def _write_prefix(path, *, load_nm, omega_offset=0.0):
    stimulations = 2
    cycles = 3
    nodes = cycles * stimulations + 1
    theta = -np.pi * np.arange(nodes, dtype=float)
    theta[-1] += 1e-3
    metadata = {
        "producer_mode": "receding_horizon_concatenation",
        "producer_solver": "ipopt",
        "producer_collocation_degree": 5,
        "cycles_per_window": cycles,
        "stimulations_per_cycle": stimulations,
        "constant_crank_torque": load_nm,
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
    assert profile["status"] == "diagnostic_only"
    assert np.isclose(
        max(
            row["theta_phase_error_rad"]["observed_max"]
            for row in profile["bins"]
        ),
        1e-3,
    )
