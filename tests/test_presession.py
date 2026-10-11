"""P1/P2/P7 contracts, including the archived Ding calcium equation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cocofest.cycling.presession import PresessionConfig, ding_cn_on_stimulation_grid, prepare_presession


def _model(path: Path, *, a_scale: float) -> None:
    path.write_text(json.dumps({"schema_version": 1, "muscles": {
        "Biceps": {"a_scale": a_scale, "km_rest": 0.2, "tauc": 0.02},
        "Triceps": {"a_scale": a_scale * 2},
    }}), encoding="utf-8")


def test_cn_exact_single_pulse_and_pw_independence() -> None:
    tau = 0.02
    frequency = 30
    actual = ding_cn_on_stimulation_grid(pulse_count=1, stimulation_hz=frequency, tauc_s=tau, km_rest=0.137)
    delta = 1 / frequency
    np.testing.assert_allclose(actual, [0, (delta / tau) * math.exp(-delta / tau)], rtol=1e-14)
    # The API has no PW input because DingModelPulseWidthFrequency.system_dynamics
    # sends it only to a_calculation, while calculate_cn_dot uses fixed km_rest.


def test_presession_bundle_is_content_addressed_and_portable(tmp_path: Path) -> None:
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    _model(left, a_scale=1000)
    _model(right, a_scale=2000)
    config = PresessionConfig(left, right, horizon_cycles=2)
    directory = prepare_presession(config, tmp_path / "output")
    assert directory.name.startswith("presession-")
    assert prepare_presession(config, tmp_path / "output") == directory
    manifest = json.loads((directory / "manifest.json").read_text())
    budget = json.loads((directory / "budget.json").read_text())
    assert manifest["prepared_phases"] == ["P1", "P2", "P7"]
    assert manifest["p2_passive_torque"]["status"] == "not_measured"
    assert not (directory / "passive_torque.npz").exists()
    assert (directory / "model_left.json").is_file()
    assert (directory / "model_right.json").is_file()
    assert budget["seconds"]["total"] >= budget["seconds"]["P1_cn"]
    with np.load(directory / "cn_timeseries.npz", allow_pickle=False) as data:
        assert data["time_s"].shape == (61,)
        assert data["left__Biceps"].shape == (61,)
        assert data["right__Triceps"].shape == (61,)
        assert data["time_s"][-1] == pytest.approx(2.0)
    copied = tmp_path / "copied"
    copied.mkdir()
    for artifact in directory.iterdir():
        (copied / artifact.name).write_bytes(artifact.read_bytes())
    assert json.loads((copied / "config.json").read_text())["model_left"] == "model_left.json"
    # PW-relevant model changes invalidate the bundle, even while Cn stays identical.
    _model(left, a_scale=1001)
    updated = prepare_presession(config, tmp_path / "output")
    assert updated != directory
    with np.load(updated / "cn_timeseries.npz", allow_pickle=False) as new, np.load(directory / "cn_timeseries.npz", allow_pickle=False) as old:
        np.testing.assert_array_equal(new["left__Biceps"], old["left__Biceps"])


def test_changed_artifact_is_rejected(tmp_path: Path) -> None:
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    _model(left, a_scale=1000)
    _model(right, a_scale=2000)
    config = PresessionConfig(left, right, horizon_cycles=1)
    directory = prepare_presession(config, tmp_path / "output")
    (directory / "passive_torque.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        prepare_presession(config, tmp_path / "output")


def test_passive_torque_from_reduced_profile(tmp_path: Path) -> None:
    from cocofest.dynamics.reduced_cycling import (
        PeriodicFourierSeries, ReducedCyclingDynamics, ReducedCyclingKinematics,
    )

    left, right = tmp_path / "left.json", tmp_path / "right.json"
    _model(left, a_scale=1000)
    _model(right, a_scale=2000)
    kinematics = ReducedCyclingKinematics(
        theta_origin=0.0, direction=-1, winding_numbers=np.array([1.0]),
        periodic_residual=PeriodicFourierSeries(np.array([0.0]), np.zeros((1, 1)), np.zeros((1, 1))),
    )
    # M=1, gravity=2, quadratic velocity coefficient=3, one muscle, b_ext=1.
    coefficients = PeriodicFourierSeries(
        np.array([1.0, 2.0, 3.0, 0.0, 1.0]), np.zeros((5, 1)), np.zeros((5, 1)),
    )
    profile = tmp_path / "reduced.npz"
    ReducedCyclingDynamics(kinematics, coefficients, ("Biceps",), 0).save(profile)
    directory = prepare_presession(PresessionConfig(left, right, 1, reduced_profile=profile), tmp_path / "output")
    with np.load(directory / "passive_torque.npz", allow_pickle=False) as data:
        np.testing.assert_allclose(data["mechanical_passive_nm"], 2 + 3 * (2 * math.pi) ** 2)
    assert json.loads((directory / "passive_torque.json").read_text())["status"] == "computed_from_reduced_profile"
