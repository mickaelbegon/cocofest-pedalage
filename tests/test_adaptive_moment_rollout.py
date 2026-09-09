import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cocofest.optimization.adaptive_moment_rollout import (
    DingPulseWidthParameters,
    MomentTrackingInterval,
    MomentTrackingStatus,
    effective_recruitment,
    periodic_calcium_state,
    propagate_ding_pulse_width_interval,
    rollout_adaptive_moment_policy,
    rollout_fixed_pulse_width_policy,
    solve_pulse_width_for_target_moment,
)
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters


def _parameters(capacity=4920.0):
    return DingPulseWidthParameters(
        fatigue=DingFatigueParameters(
            a_rest=capacity,
            tau1_rest=0.060601,
            km_rest=0.137,
            alpha_a=-0.4,
            alpha_tau1=2.1e-5,
            alpha_km=1.9e-5,
            tau_fat=127.0,
        ),
        tauc=0.011,
        tau2=0.001,
        pd0=0.000131405,
        pdt=0.000194138,
        pulse_width_max=0.0006,
    )


def _initial(parameters=None, force=35.0):
    parameters = _parameters() if parameters is None else parameters
    return np.array(
        [
            0.1629821583533315,
            force,
            0.98 * parameters.fatigue.a_rest,
            1.02 * parameters.fatigue.tau1_rest,
            1.02 * parameters.fatigue.km_rest,
        ]
    )


def test_fixed_rk4_transition_matches_independent_dop853_reference():
    parameters = _parameters()
    initial = _initial(parameters)
    duration = 1.0 / 30.0
    amplitude = 1.0597355478114694
    pulse_width = 0.00037

    def gain(time):
        return 0.92 + 0.04 * math.cos(2.0 * math.pi * time / duration)

    def rhs(time, state):
        cn, force, capacity, tau1, km = state
        history = amplitude * math.exp(-time / parameters.tauc)
        activation = cn / (km + cn)
        recruitment = effective_recruitment(capacity, pulse_width, parameters)
        fatigue = parameters.fatigue
        return [
            (history - cn) / parameters.tauc,
            gain(time)
            * (recruitment * activation - force / (tau1 + parameters.tau2 * activation)),
            -(capacity - fatigue.a_rest) / fatigue.tau_fat + fatigue.alpha_a * force,
            -(tau1 - fatigue.tau1_rest) / fatigue.tau_fat + fatigue.alpha_tau1 * force,
            -(km - fatigue.km_rest) / fatigue.tau_fat + fatigue.alpha_km * force,
        ]

    reference = solve_ivp(
        rhs,
        (0.0, duration),
        initial,
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
    ).y[:, -1]
    observed = propagate_ding_pulse_width_interval(
        initial,
        pulse_width=pulse_width,
        duration=duration,
        calcium_amplitude=amplitude,
        mechanical_gain=gain,
        parameters=parameters,
        integration_substeps=128,
    )

    np.testing.assert_allclose(observed, reference, rtol=2e-8, atol=2e-9)
    assert observed[0] == periodic_calcium_state(
        initial[0], duration, amplitude, parameters.tauc
    )


def test_scalar_inverse_recovers_a_known_pw_and_signed_muscle_moment():
    parameters = _parameters()
    initial = _initial(parameters)
    duration = 1.0 / 30.0
    amplitude = 1.0597355478114694
    known_pw = 0.000355
    coefficient = -0.047
    expected = propagate_ding_pulse_width_interval(
        initial,
        pulse_width=known_pw,
        duration=duration,
        calcium_amplitude=amplitude,
        mechanical_gain=0.96,
        parameters=parameters,
        integration_substeps=16,
    )

    result = solve_pulse_width_for_target_moment(
        initial,
        target_moment=coefficient * expected[1],
        moment_coefficient=coefficient,
        duration=duration,
        calcium_amplitude=amplitude,
        mechanical_gain=0.96,
        parameters=parameters,
        integration_substeps=16,
    )

    assert result.status is MomentTrackingStatus.OK
    assert result.pulse_width == pytest.approx(known_pw, abs=2e-12)
    assert result.moment_error == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_allclose(result.next_state, expected, rtol=2e-10, atol=2e-10)


def test_unreachable_targets_are_reported_without_boundary_clipping():
    parameters = _parameters()
    initial = _initial(parameters)
    kwargs = dict(
        state=initial,
        moment_coefficient=0.05,
        duration=1.0 / 30.0,
        calcium_amplitude=1.0597355478114694,
        mechanical_gain=0.96,
        parameters=parameters,
        integration_substeps=12,
    )
    low_state = propagate_ding_pulse_width_interval(
        initial,
        pulse_width=parameters.pd0,
        duration=kwargs["duration"],
        calcium_amplitude=kwargs["calcium_amplitude"],
        mechanical_gain=kwargs["mechanical_gain"],
        parameters=parameters,
        integration_substeps=kwargs["integration_substeps"],
    )
    high_state = propagate_ding_pulse_width_interval(
        initial,
        pulse_width=parameters.pulse_width_max,
        duration=kwargs["duration"],
        calcium_amplitude=kwargs["calcium_amplitude"],
        mechanical_gain=kwargs["mechanical_gain"],
        parameters=parameters,
        integration_substeps=kwargs["integration_substeps"],
    )

    below = solve_pulse_width_for_target_moment(
        target_moment=0.05 * low_state[1] - 0.1, **kwargs
    )
    above = solve_pulse_width_for_target_moment(
        target_moment=0.05 * high_state[1] + 0.1, **kwargs
    )

    assert below.status is MomentTrackingStatus.TARGET_FORCE_BELOW_PD0_RESPONSE
    assert above.status is MomentTrackingStatus.TARGET_FORCE_ABOVE_PW_MAX_RESPONSE
    assert below.pulse_width is above.pulse_width is None
    assert below.next_state is above.next_state is None


def test_multicycle_rollout_adapts_pw_and_preserves_each_target_moment():
    parameters = (_parameters(), _parameters(capacity=4200.0))
    initial = np.vstack((_initial(parameters[0], force=25.0), _initial(parameters[1], force=22.0)))
    duration = 1.0 / 30.0
    amplitude = (1.0597355478114694, 1.0597355478114694)
    coefficients = (0.05, -0.04)
    target_forces = (25.0, 22.0)
    interval = MomentTrackingInterval(
        duration=duration,
        calcium_amplitudes=amplitude,
        mechanical_gains=(0.96, 0.90),
        moment_coefficients=coefficients,
        target_moments=tuple(c * force for c, force in zip(coefficients, target_forces, strict=True)),
    )

    result = rollout_adaptive_moment_policy(
        initial,
        intervals=(interval,),
        parameters=parameters,
        horizon_cycles=8,
        integration_substeps=16,
    )

    assert result.status == "complete"
    assert result.completed_cycles == 8
    expected_moments = np.broadcast_to(
        np.asarray(interval.target_moments)[None, :, None],
        result.achieved_moments.shape,
    )
    np.testing.assert_allclose(result.achieved_moments, expected_moments, atol=1e-8)
    # Recovery and fatigue compete, so a monotone increase is not a valid
    # physiological invariant.  The required property is that the policy
    # adapts instead of blindly repeating the first pulse width.
    assert np.ptp(result.pulse_widths[:, 0, 0]) > 1e-9
    assert np.ptp(result.pulse_widths[:, 1, 0]) > 1e-9
    for muscle_index, muscle_parameters in enumerate(parameters):
        assert np.all(result.pulse_widths[:, muscle_index, :] >= muscle_parameters.pd0)
        assert np.all(
            result.pulse_widths[:, muscle_index, :] <= muscle_parameters.pulse_width_max
        )
    assert result.scalar_function_evaluations > 0

    # Repeating the first PW under the same Ding dynamics does not preserve
    # the requested moment, which is precisely the drift this policy removes.
    fixed = rollout_fixed_pulse_width_policy(
        initial,
        intervals=(interval,),
        parameters=parameters,
        pulse_widths=result.pulse_widths[0],
        horizon_cycles=result.requested_cycles,
        integration_substeps=16,
    )
    assert abs(fixed.achieved_moments[-1, 0, 0] - interval.target_moments[0]) > 1e-5


def test_rollout_stops_at_first_infeasible_target_without_advancing_partial_interval():
    parameters = (_parameters(),)
    initial = _initial(parameters[0], force=20.0)[None, :]
    interval = MomentTrackingInterval(
        duration=1.0 / 30.0,
        calcium_amplitudes=(1.0597355478114694,),
        mechanical_gains=(0.96,),
        moment_coefficients=(0.05,),
        target_moments=(1000.0,),
    )

    result = rollout_adaptive_moment_policy(
        initial,
        intervals=(interval,),
        parameters=parameters,
        horizon_cycles=5,
    )

    assert result.status == "infeasible"
    assert result.completed_cycles == result.completed_intervals == 0
    assert result.first_failure["status"] == "target_force_above_pw_max_response"
    assert np.isnan(result.pulse_widths).all()
    np.testing.assert_array_equal(result.state_history[0], initial)
