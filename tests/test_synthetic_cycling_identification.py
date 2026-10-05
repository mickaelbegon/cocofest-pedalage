"""Numerical and sensor-contract checks for synthetic Ding cycling trials."""

import numpy as np
import pytest

from cocofest.identification.synthetic_cycling import (
    BILATERAL_MUSCLE_NAMES,
    CyclingTrialSpec,
    SensorNoise,
    generate_trial,
    predict_trial,
    simple_mono_muscle_spec,
)


def test_bilateral_template_has_eight_muscles_and_nonzero_ding_response():
    assert len(BILATERAL_MUSCLE_NAMES) == 8
    for muscle in BILATERAL_MUSCLE_NAMES:
        spec = simple_mono_muscle_spec(muscle, duration_s=0.2, sample_rate_hz=100)
        predicted, states = predict_trial({}, spec)
        assert predicted.shape == (21, 1)
        assert states.shape == (21, 5)
        assert np.all(np.isfinite(states))
        assert np.max(predicted[:, 0]) > 0


def test_torque_and_two_axis_sensor_channels_use_declared_geometry():
    spec = simple_mono_muscle_spec(
        "left_Biceps", duration_s=0.2, sample_rate_hz=100,
        moment_arm_m=-0.03, crank_radius_m=0.15, radial_gain=0.07,
    )
    trial = generate_trial(spec)
    assert trial.observed.shape == (21, 2)
    np.testing.assert_allclose(trial.torque_tangent_nm, -0.03 * trial.state[:, 1])
    np.testing.assert_allclose(trial.force_radial_n, 0.07 * trial.state[:, 1])
    assert trial.observed[-1, 0] < 0


def test_seeded_noise_and_bias_are_reproducible_and_keep_truth():
    spec = simple_mono_muscle_spec("right_Triceps", duration_s=0.2, sample_rate_hz=100, radial_gain=0.1)
    noise = SensorNoise(tangential_std_n=0.2, radial_std_n=0.4,
                        tangential_bias_n=0.5, radial_bias_n=-0.25)
    first = generate_trial(spec, noise=noise, seed=42)
    second = generate_trial(spec, noise=noise, seed=42)
    other = generate_trial(spec, noise=noise, seed=43)
    np.testing.assert_array_equal(first.observed, second.observed)
    assert not np.array_equal(first.observed, other.observed)
    np.testing.assert_array_equal(first.noise_std, [0.2, 0.4])
    np.testing.assert_allclose(first.noiseless, predict_trial({}, spec)[0])


def test_prediction_changes_with_force_and_fatigue_parameters():
    spec = simple_mono_muscle_spec("right_Biceps", duration_s=0.6, sample_rate_hz=100)
    baseline, _ = predict_trial({}, spec)
    larger, _ = predict_trial({"a_rest": 4000.0}, spec)
    assert larger[-1, 0] > baseline[-1, 0]
    slow_recovery, _ = predict_trial({"tau_fat": 40.0, "alpha_a": -0.3}, spec)
    assert not np.allclose(slow_recovery, baseline)


def test_invalid_stimulation_and_geometry_are_rejected():
    spec = simple_mono_muscle_spec("left_Delt_ant", duration_s=0.2, sample_rate_hz=100)
    with pytest.raises(ValueError, match="within"):
        predict_trial({}, CyclingTrialSpec(**{**vars(spec), "pulse_width_s": np.full_like(spec.pulse_width_s, 0.00005)}))
    with pytest.raises(ValueError, match="radial_gain"):
        CyclingTrialSpec(**{**vars(spec), "baseline_radial_n": np.zeros_like(spec.time_s)})
    with pytest.raises(ValueError, match="Unknown Ding"):
        predict_trial({"not_a_parameter": 1}, spec)
