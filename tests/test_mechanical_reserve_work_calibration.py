"""Numerical falsification tests for a signed cycle-work reserve proxy."""

import math

import casadi as ca
import numpy as np
import pytest

from cocofest.optimization.adaptive_moment_rollout import (
    DingPulseWidthParameters, MomentTrackingInterval, propagate_ding_pulse_width_interval,
)
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.mechanical_reserve_calibration import (
    calibrate_isokinetic_ding_margin_model, calibrate_local_mechanical_margin_model,
)
from cocofest.optimization.mechanical_reserve_projection import projected_mechanical_reserve_casadi


def signed_case():
    fatigue = DingFatigueParameters(1000., .06, .13, -.2, 2e-5, 2e-5, 100.)
    parameter = DingPulseWidthParameters(fatigue, .011, .001, .0001, .0002, .0006)
    # A predominantly antagonistic muscle illustrates why PWmax is not a
    # mechanical capacity envelope. Its residual initial force is preserved.
    intervals = tuple(MomentTrackingInterval(
        .025, (1., 1.), (1., 1.), (-.02, coefficient), (0., 0.),
    ) for coefficient in (.01, .02, .02, -.003))
    return dict(terminal_states=np.array([[.2, 12., 900., .065, .14],
                                          [.2, 12., 900., .065, .14]]),
                intervals=intervals, pulse_width_parameters=(parameter, parameter),
                angular_velocity_rad_s=-20*math.pi, required_power_w=np.full(4, 3.),
                power_scale_w=3., integration_substeps=24)


def test_negative_phase_power_is_not_negative_cycle_work():
    args = dict(reference_states=np.array([[100., .06, .13]]),
                force_envelope=lambda states: np.ones((1, 2)) * states[0, 0],
                moment_coefficients=np.array([[.001, -.01]]),
                phase_durations=np.array([.5, .5]), angular_velocity_rad_s=-2*math.pi,
                required_power_w=np.ones(2), power_scale_w=1.)
    legacy = calibrate_local_mechanical_margin_model(**args)
    work = calibrate_local_mechanical_margin_model(**args, margin_aggregation="cycle_work")
    assert legacy.margin_model.reference_margins.min() < 0
    assert work.margin_model.reference_margins.shape == (1,)
    assert work.margin_model.reference_margins[0] > 0
    np.testing.assert_allclose(work.margin_model.state_jacobian,
                               legacy.margin_model.state_jacobian[-1:])


def test_gated_work_dominates_allmax_with_complete_bounded_replays():
    args = signed_case()
    legacy = calibrate_isokinetic_ding_margin_model(**args)
    gated = calibrate_isokinetic_ding_margin_model(**args, calibration_policy="cycle_work_gated_v1")
    durations = np.array([i.duration for i in args["intervals"]])
    assert gated.reference_net_power_w @ durations > legacy.reference_net_power_w @ durations
    assert gated.margin_aggregation == "cycle_work"
    assert gated.margin_model.reference_margins.shape == (1,)
    assert not gated.attainable_work_certified
    for muscle, parameter in enumerate(args["pulse_width_parameters"]):
        widths = gated.selected_pulse_widths[muscle]
        assert np.all((widths >= parameter.pd0) & (widths <= parameter.pulse_width_max))
        state = args["terminal_states"][muscle].copy()
        for phase, interval in enumerate(args["intervals"]):
            state = propagate_ding_pulse_width_interval(
                state, pulse_width=widths[phase], duration=interval.duration,
                calcium_amplitude=interval.calcium_amplitudes[muscle],
                mechanical_gain=interval.mechanical_gains[muscle], parameters=parameter,
                integration_substeps=args["integration_substeps"],
            )
            assert gated.reference_envelope_force[muscle, phase] == pytest.approx(state[1])


def test_work_margin_and_jacobian_invariant_to_consistent_mechanical_unit_scaling():
    args = signed_case()
    base = calibrate_isokinetic_ding_margin_model(**args, calibration_policy="cycle_work_gated_v1")
    scale = 7.3
    args["intervals"] = tuple(MomentTrackingInterval(
        i.duration, i.calcium_amplitudes, i.mechanical_gains,
        tuple(scale * b for b in i.moment_coefficients), i.target_moments,
    ) for i in args["intervals"])
    args["required_power_w"] *= scale
    args["power_scale_w"] *= scale
    other = calibrate_isokinetic_ding_margin_model(**args, calibration_policy="cycle_work_gated_v1")
    np.testing.assert_allclose(base.margin_model.reference_margins, other.margin_model.reference_margins)
    np.testing.assert_allclose(base.normalized_state_jacobian, other.normalized_state_jacobian, rtol=1e-8)
    np.testing.assert_array_equal(base.selected_pulse_widths, other.selected_pulse_widths)


def test_gated_jacobian_matches_independent_finite_differences_with_same_schedule():
    args = signed_case()
    base = calibrate_isokinetic_ding_margin_model(**args, calibration_policy="cycle_work_gated_v1")
    for muscle, component in np.ndindex(2, 3):
        delta = args["terminal_states"][muscle, component+2] * 2e-5
        values = []
        for sign in (-1, 1):
            states = args["terminal_states"].copy()
            states[muscle, component+2] += sign*delta
            probe = calibrate_isokinetic_ding_margin_model(
                **{**args, "terminal_states": states}, calibration_policy="cycle_work_gated_v1")
            # Away from selection switches, the independently reselected
            # schedule agrees; near switches this comparison is invalid.
            np.testing.assert_array_equal(probe.selected_pulse_widths, base.selected_pulse_widths)
            values.append(probe.margin_model.reference_margins[0])
        assert base.margin_model.state_jacobian[0, muscle, component] == pytest.approx(
            (values[1]-values[0])/(2*delta), rel=3e-6, abs=1e-8)


def test_recalibrated_cost_has_a_nonzero_correct_force_gradient_and_ranks_allocations():
    # Two equal-work productive muscles: preferentially fatigue the one with
    # lower reserve-loss-per-unit-work. This is a sensitivity/decision screen,
    # not a complete constrained RHO/endurance result.
    states = np.array([[1000., .06, .13], [1000., .06, .13]])
    calibrated = calibrate_local_mechanical_margin_model(
        reference_states=states, force_envelope=lambda s: (s[:, 0] / 10)[:, None],
        moment_coefficients=np.array([[-.01], [-.01]]), phase_durations=np.array([1.]),
        angular_velocity_rad_s=-2*math.pi, required_power_w=np.array([6.]), power_scale_w=6.,
        margin_aggregation="cycle_work",
    )
    parameters = tuple(DingFatigueParameters(1000., .06, .13, alpha, 0., 0., 100.)
                       for alpha in (-.1, -.4))
    allocation = ca.SX.sym("allocation")
    force = ca.vertcat(allocation, 100-allocation)
    result = projected_mechanical_reserve_casadi(states, force, [1.], parameters, [20],
                                                calibrated.margin_model)
    fun = ca.Function("work_reserve_decision", [allocation], [result.penalty, ca.gradient(result.penalty, allocation)])
    value, gradient = map(float, fun(50.))
    finite = (float(fun(50.001)[0])-float(fun(49.999)[0]))/.002
    assert gradient == pytest.approx(finite, rel=1e-8)
    assert gradient < -1e-4
    assert float(fun(90.)[0]) < value < float(fun(10.)[0])


def test_unknown_policy_rejected():
    with pytest.raises(ValueError, match="calibration_policy"):
        calibrate_isokinetic_ding_margin_model(**signed_case(), calibration_policy="unsupported")
