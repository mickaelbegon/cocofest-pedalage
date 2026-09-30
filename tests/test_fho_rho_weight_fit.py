import json
import threading
import numpy as np
import pytest

from cocofest.optimization.fho_rho_weight_fit import (
    TrajectoryArchive, fit_window, trajectory_tracking_loss, window_archive, write_archive,
)


def _archive(tmp_path, name, offset=0.0):
    path = tmp_path / name
    states = {"A_Biceps": np.array([[1., 2., 3., 4., 5.]]), "theta": np.array([[0., 1., 2., 3., 4.]])}
    controls = {"last_pulse_width_Biceps": np.array([[.0, .0001, .0002, .0003]])}
    if offset:
        states = {key: value + offset for key, value in states.items()}
    np.savez(path, **{f"states__{key}": value for key, value in states.items()},
             **{f"controls__{key}": value for key, value in controls.items()},
             metadata__json=np.asarray(json.dumps({})))
    return TrajectoryArchive.load(path)


def test_tracking_loss_is_zero_for_identical_archives(tmp_path):
    archive = _archive(tmp_path, "a.npz")
    result = trajectory_tracking_loss(archive, archive)
    assert result["objective"] == 0.0
    assert result["initial_state_within_tolerance"]


def test_window_archive_uses_one_based_inclusive_cycles(tmp_path):
    archive = _archive(tmp_path, "a.npz")
    # Four controls, configured as two controls/cycle for this small fixture.
    window = window_archive(archive, 2, 2, controls_per_cycle=2)
    assert window.controls["last_pulse_width_Biceps"].shape[-1] == 2
    assert window.states["A_Biceps"].shape[-1] == 3
    written = write_archive(window, tmp_path / "window.npz")
    assert TrajectoryArchive.load(written).controls["last_pulse_width_Biceps"].shape[-1] == 2


def test_fit_window_runs_independent_candidates_in_parallel(tmp_path):
    archive = _archive(tmp_path, "a.npz")
    barrier = threading.Barrier(2)

    def solve(_weights, _index):
        barrier.wait(timeout=2)
        return archive.path

    result = fit_window(
        archive,
        solve,
        window_index=0,
        evaluations=6,
        parallel_workers=2,
    )
    assert result["parallel_workers"] == 2
    assert [attempt["evaluation"] for attempt in result["evaluations"]] == list(range(6))
    assert result["weight_constraint"] == "nonnegative_l1_simplex_sum_equals_1"
    for attempt in result["evaluations"]:
        assert sum(attempt["weights"].values()) == pytest.approx(1.0)
