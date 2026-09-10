import numpy as np
import pytest

from cocofest.optimization.physiological_muscle_weights import (
    calculate_physiological_muscle_weights,
)
from cocofest.optimization.physiological_weight_cases import (
    MUSCLE_NAMES,
    PUBLISHED_CODE_BASELINE_ID,
    get_baseline,
)
from scripts import physiological_weight_geometry as geometry_module


@pytest.fixture(scope="module")
def published_geometry():
    """Build the 120-shooting source geometry exactly once for this module."""

    parameters = get_baseline(PUBLISHED_CODE_BASELINE_ID).as_parameter_dict()
    geometry = geometry_module.build_geometry(MUSCLE_NAMES, n_shooting=120)
    profiles, profile_audit = geometry.profiles(parameters)
    result = calculate_physiological_muscle_weights(
        geometry.theta,
        profiles,
        parameters,
        muscle_names=MUSCLE_NAMES,
        case_id="published_code_nominal_geometry_audit",
        target_cycles=1500,
        rho=0.8,
        pre_risk_width_deg=90.0,
        task_torque_threshold=0.20,
    )
    return geometry, parameters, profiles, profile_audit, result


def test_published_nominal_geometry_recovers_source_rounded_factors(published_geometry):
    _, _, _, _, result = published_geometry

    assert result["status"] == "ok"
    comparisons = (
        (
            tuple(result["legacy_normalized_weights"].values()),
            (1.0, 0.0943, 0.389, 0.0),
            (1e-12, 5e-5, 5e-4, 1e-12),
        ),
        (
            tuple(result["fatigability"].values()),
            (6.53e-4, 3.04e-4, 1.52e-4, 3.62e-5),
            (5e-7, 5e-7, 5e-7, 5e-8),
        ),
        (
            tuple(result["mechanical_contribution"].values()),
            (0.0815, 0.0354, 0.582, 0.0),
            (5e-5, 5.1e-5, 5e-4, 1e-12),
        ),
    )
    for actual, rounded_source, rounding_tolerance in comparisons:
        assert np.all(
            np.abs(np.asarray(actual) - np.asarray(rounded_source))
            <= np.asarray(rounding_tolerance)
        )


def _direct_native_profiles(geometry, parameters, sample_indices, function_suffix):
    """Project torques from a separate native model, not the adapter graph."""

    import biorbd_casadi as biorbd
    import casadi as ca

    model = biorbd.Model(str(geometry_module.MODEL))
    for muscle_index, muscle_name in enumerate(MUSCLE_NAMES):
        model.muscle(muscle_index).setForceIsoMax(parameters[muscle_name]["Fmax"])
    q = ca.MX.sym(f"direct_q_{function_suffix}", model.nbQ())
    qdot = ca.MX.sym(f"direct_qdot_{function_suffix}", model.nbQdot())
    torque_functions = []
    for active_index in range(len(MUSCLE_NAMES)):
        states = model.stateSet()
        for muscle_index, state in enumerate(states):
            active = float(muscle_index == active_index)
            state.setExcitation(active)
            state.setActivation(active)
        torque_functions.append(
            ca.Function(
                f"direct_native_torque_{function_suffix}_{active_index}",
                [q, qdot],
                [model.muscularJointTorque(states, q, qdot).to_mx()],
            )
        )

    projected = np.empty((len(MUSCLE_NAMES), len(sample_indices)))
    for output_index, sample_index in enumerate(sample_indices):
        for muscle_index, torque_function in enumerate(torque_functions):
            joint_torque = np.asarray(
                torque_function(
                    geometry.q[:, sample_index], geometry.qdot[:, sample_index]
                ),
                dtype=float,
            ).ravel()
            projected[muscle_index, output_index] = (
                geometry.projectors[:, sample_index] @ joint_torque
            )
    return projected


def test_profiles_recompute_fmax_and_retain_cross_muscle_passive_torque(published_geometry):
    geometry, parameters, nominal_profiles, nominal_audit, _ = published_geometry
    sample_indices = (0, 31, 73, 120)
    doubled_triceps = {
        muscle_name: dict(values) for muscle_name, values in parameters.items()
    }
    doubled_triceps["Triceps"]["Fmax"] *= 2.0
    changed_profiles, changed_audit = geometry.profiles(doubled_triceps)

    direct_nominal = _direct_native_profiles(
        geometry, parameters, sample_indices, "nominal"
    )
    direct_changed = _direct_native_profiles(
        geometry, doubled_triceps, sample_indices, "triceps_x2"
    )

    np.testing.assert_allclose(
        nominal_profiles[:, sample_indices], direct_nominal, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        changed_profiles[:, sample_indices], direct_changed, rtol=0.0, atol=1e-12
    )
    # Delt_ant is active in row zero; changing inactive Triceps Fmax still
    # changes that row because the source calculation retains passive forces.
    assert np.max(np.abs(changed_profiles[0] - nominal_profiles[0])) > 0.05
    assert nominal_audit["maximum_absolute_all_inactive_crank_torque_nm"] > 0.1
    assert nominal_audit["passive_torque_subtracted"] is False
    assert changed_audit["profiles_recomputed_for_case_Fmax"] is True
    assert changed_audit["Fmax"]["Triceps"] == 2.0 * parameters["Triceps"]["Fmax"]


def test_wu_bounds_parser_finds_the_three_active_rotations():
    names, lower, upper = geometry_module._wu_bounds(geometry_module.IK_MODEL)

    assert names == ("humerus_RotZ", "ulna_RotZ", "wheel_rotation_RotZ")
    np.testing.assert_allclose(
        lower, (-1.570796326795, -1.0, -40.0 * np.pi), rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        upper, (1.570796326795, 3.0, 40.0 * np.pi), rtol=0.0, atol=1e-14
    )
    assert np.all(lower < upper)


@pytest.mark.parametrize(
    "expression",
    (
        "1 + 2",
        "pi / 2",
        "2 ** 3",
        "__import__('os')",
        "unknown",
        "[1]",
    ),
)
def test_range_scalar_refuses_code_and_unsupported_arithmetic(expression):
    with pytest.raises(ValueError, match="Unsupported Wu model range expression"):
        geometry_module._range_scalar(expression)


def test_range_scalar_accepts_only_the_declared_numeric_pi_product_grammar():
    assert geometry_module._range_scalar("-40*pi") == pytest.approx(-40.0 * np.pi)
    assert geometry_module._range_scalar("+1.5") == pytest.approx(1.5)


def test_ik_audit_reports_source_marker_misfit_without_exactness_claim(published_geometry):
    geometry, _, _, _, _ = published_geometry
    audit = geometry.metadata["ik_audit"]

    assert audit["maximum_marker_error_m"] == pytest.approx(0.20147, abs=5e-6)
    assert audit["maximum_xy_marker_error_m"] == pytest.approx(0.009706, abs=5e-6)
    assert audit["maximum_marker_error_m"] > 0.2
    assert audit["maximum_xy_marker_error_m"] > 0.009
    assert audit["requested_marker_geometry_within_10um"] is False
    assert audit["source_least_squares_result_retained_without_exact_geometry_claim"] is True
    assert audit["bounds_source"] == "parsed_active_bioMod_ranges"
    assert audit["solver"] == "scipy_trf_source_settings_with_casadi_marker_jacobian"
