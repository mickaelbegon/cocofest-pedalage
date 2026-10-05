"""Sensor geometry must retain the OCP's sign, side and velocity laws."""

from dataclasses import replace

import numpy as np
import pytest

from cocofest.dynamics.reduced_cycling import (
    PeriodicFourierSeries, ReducedCyclingDynamics, ReducedCyclingKinematics,
)
from cocofest.identification.cycling_geometry import ReducedGeometryAdapter
from cocofest.identification.synthetic_cycling import generate_trial, simple_mono_muscle_spec


def profile(names=("right_Biceps", "left_Biceps")):
    def series(offset, cosine=None):
        return PeriodicFourierSeries(np.array(offset), np.zeros((len(offset), 1)) if cosine is None else cosine,
                                     np.zeros((len(offset), 1)))
    kinematics = ReducedCyclingKinematics(0., 1, np.array([1.]), series([0.]))
    return ReducedCyclingDynamics(
        kinematics, series([1., 0., 0., 0., -0.03, 1.],
                           np.array([[0.], [0.], [0.], [0.02], [0.], [0.]])),
        names, 0, series([1., 1.1, 0.1, -0.2]), "test-source-hash",
    )


def test_profile_roundtrip_reuses_exact_coefficients_and_speed(tmp_path):
    reduced = profile()
    path = reduced.save(tmp_path / "profile.npz")
    adapter = ReducedGeometryAdapter.load(path)
    spec = simple_mono_muscle_spec("right_Biceps", duration_s=.2)
    velocity = np.linspace(-2, -5, len(spec.time_s))
    mapped = adapter.apply(spec, omega_rad_s=velocity)
    for i, (angle, speed) in enumerate(zip(spec.angle_rad, velocity)):
        fl, fv, fp = reduced.muscle_relationships(angle, speed)
        assert mapped.mechanical_gain[i] == pytest.approx((fl * fv + fp)[0])
        assert mapped.moment_arm_m[i] == pytest.approx(reduced.coefficient_values(angle)["muscle_effectiveness"][0])
    assert not np.allclose(mapped.mechanical_gain, adapter.apply(spec, omega_rad_s=0).mechanical_gain)
    assert adapter.provenance["source_model_sha256"] == "test-source-hash"
    trial = generate_trial(mapped)
    np.testing.assert_allclose(trial.torque_tangent_nm, mapped.moment_arm_m * trial.state[:, 1])


def test_left_side_sign_and_radial_require_independent_mapping():
    adapter = ReducedGeometryAdapter(profile())
    spec = simple_mono_muscle_spec("left_Biceps", duration_s=.2, radial_gain=.1)
    with pytest.raises(ValueError, match="radial_gain"):
        adapter.apply(spec, omega_rad_s=-2)
    mapped = adapter.apply(spec, omega_rad_s=-2, radial_gain=np.full(spec.time_s.shape, .17))
    np.testing.assert_allclose(mapped.moment_arm_m, -.03)
    trial = generate_trial(mapped)
    np.testing.assert_allclose(trial.force_radial_n, .17 * trial.state[:, 1])
    assert trial.observed[-1, 0] < 0


def test_unilateral_mapping_must_be_explicit_and_offset_is_respected():
    reduced = profile(("Biceps", "Triceps"))
    with pytest.raises(ValueError, match="muscle_map"):
        ReducedGeometryAdapter(reduced)
    adapter = ReducedGeometryAdapter(reduced, {"right_Biceps": "Biceps"}, np.pi)
    spec = simple_mono_muscle_spec("right_Biceps", duration_s=.2)
    mapped = adapter.apply(spec, omega_rad_s=-2)
    np.testing.assert_allclose(mapped.moment_arm_m, -.02 * np.cos(spec.angle_rad))
    with pytest.raises(ValueError, match="mapping"):
        adapter.apply(replace(spec, muscle_name="left_Biceps"), omega_rad_s=-2)


@pytest.mark.parametrize("velocity", [np.nan, [1., 2.]])
def test_invalid_velocity_rejected(velocity):
    with pytest.raises(ValueError, match="omega_rad_s"):
        ReducedGeometryAdapter(profile()).apply(simple_mono_muscle_spec("right_Biceps"), omega_rad_s=velocity)


@pytest.mark.parametrize("fault", ["inertia", "nan", "negative_gain", "missing_geometry"])
def test_corrupt_or_nonphysical_profile_rejected(fault):
    reduced = profile()
    if fault == "missing_geometry":
        reduced = replace(reduced, muscle_geometry=None)
    elif fault == "inertia":
        reduced.coefficients.offset[0] = -1
    elif fault == "nan":
        reduced.coefficients.offset[3] = np.nan
    else:
        reduced.muscle_geometry.offset[2] = 1e8
    with pytest.raises(ValueError, match="geometry|gain"):
        ReducedGeometryAdapter(reduced).apply(simple_mono_muscle_spec("right_Biceps"), omega_rad_s=-2)
