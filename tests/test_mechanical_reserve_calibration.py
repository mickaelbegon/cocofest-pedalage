"""Calibration contracts independent of NLP backends and full-horizon data."""

import math

import numpy as np
import pytest

from cocofest.optimization.isokinetic_cycling import (
    inverse_load_torque_from_coefficients, produced_mechanical_power,
)
from cocofest.optimization.mechanical_reserve_calibration import (
    calibrate_isokinetic_ding_margin_model, calibrate_isokinetic_ding_pulse_width_force_map,
    calibrate_local_mechanical_margin_model, clip_numerical_pulse_width_bound_violations,
)


def synthetic_case():
    states = np.array([[900., .065, .14], [760., .075, .16]])
    profile = np.array([[.7, 1.2, .9], [1.1, .8, 1.3]])

    def envelope(slow):
        return (slow[:, 0] * slow[:, 1] / (slow[:, 2] + .2))[:, None] * profile

    return dict(reference_states=states, force_envelope=envelope,
                moment_coefficients=np.array([[-.02, .01, -.03], [.005, -.025, -.015]]),
                phase_durations=np.array([.1, .2, .2]), angular_velocity_rad_s=-4 * math.pi,
                required_power_w=np.array([4., 5., 3.]), power_scale_w=10.)


@pytest.mark.parametrize("direction", [-1, 1])
def test_power_sign_matches_eprod_dot_including_gravity_and_velocity(direction):
    args = synthetic_case()
    omega = direction * abs(args["angular_velocity_rad_s"])
    args["angular_velocity_rad_s"] = omega
    gravity, velocity = np.array([.1, -.2, .05]), np.array([.001, .002, -.001])
    args["nonmuscle_power_w"] = -omega * (gravity + velocity * omega**2)
    result = calibrate_local_mechanical_margin_model(**args)
    np.testing.assert_allclose(result.power_coefficients, omega * args["moment_coefficients"])
    for phase in range(3):
        external = 1. + .1 * phase
        torque = inverse_load_torque_from_coefficients(
            args["moment_coefficients"][:, phase], result.reference_envelope_force[:, phase],
            external, gravity[phase], velocity[phase], omega,
        )
        # This is E_prod_dot in ReducedFesCyclingModel: -tau_load*b_ext*omega.
        expected_eprod_dot = produced_mechanical_power(torque, external, omega)
        assert result.reference_net_power_w[phase] == pytest.approx(expected_eprod_dot)
    muscle_power = result.power_coefficients * result.reference_envelope_force
    assert np.any(muscle_power < 0)  # Antagonistic contributions are retained.


def test_cycle_duration_normalization_and_proxy_labels():
    args = synthetic_case()
    args["required_work_j"] = 9.
    result = calibrate_local_mechanical_margin_model(**args)
    assert result.cycle_duration_s == pytest.approx(.5)
    power = result.reference_net_power_w
    expected = np.r_[(power - args["required_power_w"]) / 10.,
                     (power @ args["phase_durations"] - 9.) / 5.]
    np.testing.assert_allclose(result.margin_model.reference_margins, expected)
    assert not result.attainable_work_certified
    assert not result.endurance_prediction
    assert "proxy" in result.calibration_kind
    assert not result.power_coefficients.flags.writeable


def test_normalized_finite_differences_match_analytic_all_three_state_derivatives():
    args = synthetic_case()
    args["state_scales"] = np.array([[1000., .06, .13], [1000., .06, .13]])
    result = calibrate_local_mechanical_margin_model(**args)
    states = args["reference_states"]
    force = result.reference_envelope_force
    derivatives = np.stack((force / states[:, 0, None], force / states[:, 1, None],
                            -force / (states[:, 2, None] + .2)), axis=-1)
    phase_jacobian = np.transpose(result.power_coefficients[:, :, None] * derivatives / 10., (1, 0, 2))
    work_jacobian = np.einsum("kmj,k->mj", phase_jacobian, args["phase_durations"]) / .5
    expected = np.concatenate((phase_jacobian, work_jacobian[None]))
    np.testing.assert_allclose(result.margin_model.state_jacobian, expected, rtol=3e-9, atol=2e-10)
    np.testing.assert_allclose(result.normalized_state_jacobian, expected * args["state_scales"], rtol=3e-9)
    np.testing.assert_allclose(result.difference_steps, 1e-4 * args["state_scales"])
    assert np.all(np.abs(expected) > 0)


def test_direct_force_tangent_matches_the_finite_difference_calibration():
    args = synthetic_case()
    profile = np.array([[.7, 1.2, .9], [1.1, .8, 1.3]])

    def tangent(states):
        result = np.zeros((2, 3, 2, 3))
        for muscle, (capacity, tau1, km) in enumerate(states):
            scale = profile[muscle]
            result[muscle, :, muscle, :] = np.column_stack((
                tau1 / (km + .2) * scale,
                capacity / (km + .2) * scale,
                -capacity * tau1 / (km + .2) ** 2 * scale,
            ))
        return result

    finite_difference = calibrate_local_mechanical_margin_model(**args)
    direct = calibrate_local_mechanical_margin_model(
        **args, force_envelope_jacobian=tangent,
    )
    np.testing.assert_allclose(direct.margin_model.reference_margins,
                               finite_difference.margin_model.reference_margins)
    np.testing.assert_allclose(direct.margin_model.state_jacobian,
                               finite_difference.margin_model.state_jacobian,
                               rtol=3e-9, atol=2e-10)


def test_muscle_permutation_equivariance():
    args = synthetic_case()
    baseline = calibrate_local_mechanical_margin_model(**args)
    permutation = np.array([1, 0])
    original_envelope = args["force_envelope"]
    args["reference_states"] = args["reference_states"][permutation]
    args["moment_coefficients"] = args["moment_coefficients"][permutation]
    args["force_envelope"] = lambda state: original_envelope(state[permutation])[permutation]
    permuted = calibrate_local_mechanical_margin_model(**args)
    np.testing.assert_allclose(permuted.margin_model.reference_margins, baseline.margin_model.reference_margins)
    np.testing.assert_allclose(permuted.margin_model.state_jacobian,
                               baseline.margin_model.state_jacobian[:, permutation])


def ding_case():
    from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval
    from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters

    parameters = tuple(DingPulseWidthParameters(
        DingFatigueParameters(1000., .06, .13, -.2 * factor, 2e-5, 2e-5, 100.),
        .011, .001, .0001, .0002, .0006,
    ) for factor in (1., 2.))
    intervals = (
        MomentTrackingInterval(.02, (1., 1.1), (.95, .9), (-.02, .01), (0., 0.)),
        MomentTrackingInterval(.03, (1.2, .9), (.92, .88), (.025, -.02), (0., 0.)),
    )
    return dict(terminal_states=np.array([[.12, 18., 900., .063, .14], [.16, 24., 800., .065, .15]]),
                intervals=intervals, pulse_width_parameters=parameters, angular_velocity_rad_s=-40 * math.pi,
                required_power_w=np.array([1., 2.]), power_scale_w=10., integration_substeps=64)


def test_full_ding_replay_preserves_fast_states_and_recomputes_slow_sensitivities():
    from cocofest.optimization.adaptive_moment_rollout import propagate_ding_pulse_width_interval

    args = ding_case()
    original = args["terminal_states"].copy()
    result = calibrate_isokinetic_ding_margin_model(**args)
    current = original.copy()
    for phase, interval in enumerate(args["intervals"]):
        for muscle, parameter in enumerate(args["pulse_width_parameters"]):
            current[muscle] = propagate_ding_pulse_width_interval(
                current[muscle], pulse_width=parameter.pulse_width_max, duration=interval.duration,
                calcium_amplitude=interval.calcium_amplitudes[muscle],
                mechanical_gain=interval.mechanical_gains[muscle], parameters=parameter,
                integration_substeps=args["integration_substeps"],
            )
        np.testing.assert_allclose(result.reference_envelope_force[:, phase], current[:, 1])
    np.testing.assert_array_equal(args["terminal_states"], original)
    for component in range(3):
        plus, minus = original.copy(), original.copy()
        delta = original[0, component + 2] * 3e-5
        plus[0, component + 2] += delta
        minus[0, component + 2] -= delta
        margin_plus = calibrate_isokinetic_ding_margin_model(**{**args, "terminal_states": plus}).margin_model.reference_margins
        margin_minus = calibrate_isokinetic_ding_margin_model(**{**args, "terminal_states": minus}).margin_model.reference_margins
        np.testing.assert_allclose(result.margin_model.state_jacobian[:, 0, component],
                                   (margin_plus - margin_minus) / (2 * delta), rtol=3e-7, atol=1e-8)
    assert np.all(np.max(np.abs(result.margin_model.state_jacobian), axis=0) > 0)
    altered_fast = original.copy()
    altered_fast[:, :2] *= .2
    altered = calibrate_isokinetic_ding_margin_model(**{**args, "terminal_states": altered_fast})
    assert not np.allclose(result.reference_envelope_force, altered.reference_envelope_force)


def test_ding_tangent_map_matches_central_differences():
    from cocofest.optimization.adaptive_moment_rollout import (
        _propagate_ding_pulse_width_interval_with_slow_sensitivities,
        propagate_ding_pulse_width_interval,
    )

    args = ding_case()
    state = args["terminal_states"][0]
    interval, parameter = args["intervals"][0], args["pulse_width_parameters"][0]
    output, tangent = _propagate_ding_pulse_width_interval_with_slow_sensitivities(
        state, pulse_width=parameter.pulse_width_max, duration=interval.duration,
        calcium_amplitude=interval.calcium_amplitudes[0], mechanical_gain=interval.mechanical_gains[0],
        parameters=parameter, integration_substeps=args["integration_substeps"],
    )
    expected = propagate_ding_pulse_width_interval(
        state, pulse_width=parameter.pulse_width_max, duration=interval.duration,
        calcium_amplitude=interval.calcium_amplitudes[0], mechanical_gain=interval.mechanical_gains[0],
        parameters=parameter, integration_substeps=args["integration_substeps"],
    )
    np.testing.assert_allclose(output, expected, rtol=1e-14, atol=1e-14)
    for component in range(3):
        delta = state[component + 2] * 3e-5
        plus, minus = state.copy(), state.copy()
        plus[component + 2] += delta
        minus[component + 2] -= delta
        expected_derivative = (
            propagate_ding_pulse_width_interval(
                plus, pulse_width=parameter.pulse_width_max, duration=interval.duration,
                calcium_amplitude=interval.calcium_amplitudes[0], mechanical_gain=interval.mechanical_gains[0],
                parameters=parameter, integration_substeps=args["integration_substeps"],
            )[1:]
            - propagate_ding_pulse_width_interval(
                minus, pulse_width=parameter.pulse_width_max, duration=interval.duration,
                calcium_amplitude=interval.calcium_amplitudes[0], mechanical_gain=interval.mechanical_gains[0],
                parameters=parameter, integration_substeps=args["integration_substeps"],
            )[1:]
        ) / (2 * delta)
        # The reference is itself a finite difference; RK4's tangent map is
        # compared at a tolerance appropriate for its O(delta**2) truncation.
        np.testing.assert_allclose(tangent[:, component], expected_derivative, rtol=1e-6, atol=2e-10)


def test_ding_tangent_calibration_matches_finite_difference_reference():
    args = ding_case()
    tangent = calibrate_isokinetic_ding_margin_model(**args, sensitivity_mode="tangent")
    finite_difference = calibrate_isokinetic_ding_margin_model(
        **args, sensitivity_mode="finite_difference",
    )
    np.testing.assert_allclose(tangent.reference_envelope_force,
                               finite_difference.reference_envelope_force, rtol=0, atol=0)
    np.testing.assert_allclose(tangent.margin_model.reference_margins,
                               finite_difference.margin_model.reference_margins, rtol=0, atol=0)
    np.testing.assert_allclose(tangent.margin_model.state_jacobian,
                               finite_difference.margin_model.state_jacobian,
                               rtol=4e-7, atol=2e-9)


def test_ding_pulse_width_force_tangent_is_causal_and_matches_finite_differences():
    from cocofest.optimization.adaptive_moment_rollout import propagate_ding_pulse_width_interval

    args = ding_case()
    parameters, intervals = args["pulse_width_parameters"], args["intervals"]
    widths = np.array([
        [.5 * (parameters[0].pd0 + parameters[0].pulse_width_max),
         .75 * parameters[0].pulse_width_max + .25 * parameters[0].pd0],
        [.75 * parameters[1].pulse_width_max + .25 * parameters[1].pd0,
         .5 * (parameters[1].pd0 + parameters[1].pulse_width_max)],
    ])
    affine = calibrate_isokinetic_ding_pulse_width_force_map(
        terminal_states=args["terminal_states"], intervals=intervals,
        pulse_width_parameters=parameters, pulse_widths=widths,
        integration_substeps=args["integration_substeps"],
    )
    assert affine.force_jacobian.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(affine.force_jacobian[0, :, 1], 0.0)
    np.testing.assert_array_equal(affine.force_jacobian[1, :, 0], 0.0)
    assert affine.force_jacobian[0, 0, 0, 1] == pytest.approx(0.0, abs=1e-15)
    assert affine.force_jacobian[1, 0, 1, 1] == pytest.approx(0.0, abs=1e-15)

    def replay(schedule):
        values = np.empty_like(schedule)
        for muscle, parameter in enumerate(parameters):
            current = args["terminal_states"][muscle].copy()
            for phase, interval in enumerate(intervals):
                current = propagate_ding_pulse_width_interval(
                    current, pulse_width=schedule[muscle, phase], duration=interval.duration,
                    calcium_amplitude=interval.calcium_amplitudes[muscle],
                    mechanical_gain=interval.mechanical_gains[muscle], parameters=parameter,
                    integration_substeps=args["integration_substeps"],
                )
                values[muscle, phase] = current[1]
        return values

    np.testing.assert_allclose(affine.reference_forces, replay(widths), rtol=1e-14, atol=1e-14)
    for muscle, phase in np.ndindex(widths.shape):
        delta = 2e-8
        plus, minus = widths.copy(), widths.copy()
        plus[muscle, phase] += delta
        minus[muscle, phase] -= delta
        finite_difference = (replay(plus) - replay(minus)) / (2 * delta)
        np.testing.assert_allclose(affine.force_jacobian[:, :, muscle, phase], finite_difference,
                                   rtol=3e-6, atol=2e-5)


def test_certified_pw_replay_clips_only_explicit_sub_tolerance_bound_roundoff():
    args = ding_case()
    parameters = args["pulse_width_parameters"]
    tolerance = 1e-8
    raw = np.array([
        [parameters[0].pd0 - .5 * tolerance, parameters[0].pulse_width_max + .25 * tolerance],
        [parameters[1].pd0, parameters[1].pulse_width_max],
    ])
    clipped, audit = clip_numerical_pulse_width_bound_violations(
        raw, parameters, tolerance_s=tolerance, certified_source=True,
    )
    np.testing.assert_array_equal(clipped, np.array([
        [parameters[0].pd0, parameters[0].pulse_width_max],
        [parameters[1].pd0, parameters[1].pulse_width_max],
    ]))
    assert audit == {
        "units": "s",
        "source_certified": True,
        "numerical_bound_tolerance_s_by_muscle": [tolerance, tolerance],
        "raw_min_s": pytest.approx(parameters[0].pd0 - .5 * tolerance),
        "raw_max_s": pytest.approx(parameters[0].pulse_width_max + .25 * tolerance),
        "corrected": True,
        "corrected_entries": 2,
        "corrected_below_entries": 1,
        "corrected_above_entries": 1,
        "max_abs_correction_s": pytest.approx(.5 * tolerance),
        "max_abs_correction_us": pytest.approx(.5 * tolerance * 1e6),
    }


def test_certified_pw_replay_refuses_material_bound_violation_without_clipping():
    args = ding_case()
    parameters = args["pulse_width_parameters"]
    raw = np.array([
        [parameters[0].pd0 - 1.01e-8, parameters[0].pulse_width_max],
        [parameters[1].pd0, parameters[1].pulse_width_max],
    ])
    with pytest.raises(ValueError, match=r"exceed physical Ding bounds.*no clipping was applied"):
        clip_numerical_pulse_width_bound_violations(
            raw, parameters, tolerance_s=1e-8, certified_source=True,
        )


def test_numerical_pw_clip_requires_an_explicitly_certified_source():
    args = ding_case()
    parameter = args["pulse_width_parameters"][0]
    raw = np.array([[parameter.pd0 - 1e-12, parameter.pulse_width_max]])
    with pytest.raises(ValueError, match="requires a certified NLP source"):
        clip_numerical_pulse_width_bound_violations(
            raw, args["pulse_width_parameters"][:1], tolerance_s=5e-12,
        )


@pytest.mark.parametrize("field,value", [
    ("mechanics_mode", "dynamic"), ("angular_velocity_rad_s", 0.),
    ("angular_velocity_rad_s", [-10., -11., -12.]), ("angular_velocity_rad_s", np.nan),
    ("phase_durations", [.1, .2, .3]), ("phase_durations", [.1, 0., .4]),
    ("phase_durations", []), ("power_scale_w", 0.), ("power_scale_w", np.inf),
    ("relative_step", 1.), ("relative_step", 1e-30), ("relative_step", -1.),
    ("reference_states", [[1., 0., 1.]]), ("reference_states", [[1., 1.]]),
    ("state_scales", np.ones((2, 3)) * 1e8), ("state_scales", [[1., 1., -1.]]),
    ("required_work_j", np.nan), ("required_power_w", [1., 2.]),
    ("moment_coefficients", np.zeros((3, 2))), ("nonmuscle_power_w", [0., np.inf, 0.]),
    ("force_envelope", np.ones((2, 3))), ("force_envelope", lambda states: -np.ones((2, 3))),
    ("force_envelope", lambda states: np.full((2, 3), np.nan)),
])
def test_invalid_domains_and_nonisokinetic_configuration_rejected(field, value):
    args = synthetic_case()
    args[field] = value
    with pytest.raises(ValueError):
        calibrate_local_mechanical_margin_model(**args)


@pytest.mark.parametrize("field,value", [
    ("terminal_states", [[1., 1., 1.]]), ("terminal_states", np.zeros((2, 5))),
    ("integration_substeps", True), ("integration_substeps", 0),
    ("integration_substeps", 1.5), ("mechanics_mode", "dynamic"),
])
def test_ding_domain_and_configuration_rejection(field, value):
    args = ding_case()
    args[field] = value
    with pytest.raises(ValueError):
        calibrate_isokinetic_ding_margin_model(**args)
