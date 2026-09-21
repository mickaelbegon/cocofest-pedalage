from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cocofest.optimization.adaptive_moment_rollout import (
    DingPulseWidthParameters, MomentTrackingInterval, propagate_ding_pulse_width_interval,
)
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.compact_muscle_prediction import (
    CompactMusclePredictor, fatigue_memory_coordinates, reconstruct_fatigue_memory,
)


def muscle():
    return DingPulseWidthParameters(
        DingFatigueParameters(1200., .060601, .137, -1.4, 2.1e-5, 1.9e-5, 445.5),
        tauc=.011, tau2=.001, pd0=.000131405, pdt=.000194138, pulse_width_max=.0006,
    )


def interval(targets=(1.,), coefficients=(.05,)):
    return MomentTrackingInterval(1 / 30, (1.0597355478,) * len(targets),
                                  (.95,) * len(targets), coefficients, targets)


def test_one_forced_memory_preserves_nonzero_initial_offsets_exactly():
    params = muscle().fatigue
    initial = np.array([1100., .078, .141])
    damage, offsets = fatigue_memory_coordinates(initial, params)
    np.testing.assert_allclose(reconstruct_fatigue_memory(damage, offsets, params), initial)
    duration = 10.
    force = lambda t: 20. + 8. * np.sin(2 * np.pi * t)
    full = solve_ivp(
        lambda t, z: -(z - params.rest_state) / params.tau_fat + params.alpha * force(t),
        (0, duration), initial, method="DOP853", rtol=1e-12, atol=1e-13,
    ).y[:, -1]
    reduced = solve_ivp(
        lambda t, d: -d / params.tau_fat - params.alpha_a * force(t) / params.a_rest,
        (0, duration), [damage], method="DOP853", rtol=1e-12, atol=1e-13,
    ).y[0, -1]
    reconstructed = reconstruct_fatigue_memory(
        reduced, offsets * np.exp(-duration / params.tau_fat), params
    )
    np.testing.assert_allclose(reconstructed, full, rtol=1e-10, atol=1e-10)
    assert np.max(np.abs(offsets)) > .001  # Test cannot pass by assuming rest manifold.


def test_affine_recruitment_map_converges_to_full_ding_when_slow_states_are_constant():
    base = muscle()
    params = replace(base, fatigue=replace(base.fatigue, alpha_a=0., alpha_tau1=0., alpha_km=0.))
    state = np.r_[.16298215835, 20., params.fatigue.rest_state][None, :]
    pw = .00035
    recruitment = [-np.expm1(-(pw - params.pd0) / params.pdt)]
    truth = propagate_ding_pulse_width_interval(
        state[0], pulse_width=pw, duration=1 / 30, calcium_amplitude=1.0597355478,
        mechanical_gain=.95, parameters=params, integration_substeps=256,
    )
    errors = []
    for substeps in (16, 32, 64, 128):
        predicted = CompactMusclePredictor((interval(),), (params,), substeps=substeps)
        errors.append(abs(predicted.phase_map(state, 0).endpoint(recruitment)[0, 1] - truth[1]))
    assert all(b < .3 * a for a, b in zip(errors, errors[1:]))  # midpoint order two
    assert errors[-1] < .001


def test_weighted_force_convolution_matches_direct_integral_for_constant_coefficients():
    params = muscle()
    state = np.array([[.4, 20., 1100., .075, .15]])
    # One midpoint makes the force coefficients constant by construction.
    predictor = CompactMusclePredictor((interval(),), (params,), substeps=1)
    transition = predictor.phase_map(state, 0)
    recruitment = .6
    time = (1 / 30) / 2
    cn = np.exp(-time / params.tauc) * (state[0, 0] + 1.0597355478 * time / params.tauc)
    q = cn / (state[0, 4] + cn)
    k = .95 / (state[0, 3] + params.tau2 * q)
    forcing = .95 * state[0, 2] * q * recruitment
    truth = solve_ivp(
        lambda t, x: [forcing - k * x[0], -x[1] / params.fatigue.tau_fat + x[0]],
        (0, 1 / 30), [20., 0.], method="DOP853", rtol=1e-12, atol=1e-13,
    ).y[:, -1]
    actual = transition.endpoint([recruitment])
    integral = transition.weighted_force_intercept[0] + recruitment * transition.weighted_force_slope[0]
    np.testing.assert_allclose([actual[0, 1], integral], truth, rtol=1e-10, atol=1e-11)
    expected_slow = (params.fatigue.rest_state + np.exp(-(1 / 30) / params.fatigue.tau_fat)
                     * (state[0, 2:] - params.fatigue.rest_state) + params.fatigue.alpha * integral)
    np.testing.assert_allclose(actual[0, 2:], expected_slow, atol=1e-12)


def test_recruitment_rollout_preserves_signed_total_moment_with_antagonists():
    params = (muscle(), muscle())
    state = np.array([[.16298, 20., 1150., .067, .15]] * 2)
    predictor = CompactMusclePredictor((interval((.7, -.2), (.05, -.03)),), params)
    result = predictor.rollout(state, horizon_cycles=10)
    assert result.status == "complete"
    np.testing.assert_allclose(np.sum(result.achieved_moments, axis=1), .5, atol=1e-8)
    assert np.all(result.pulse_widths >= params[0].pd0)
    assert np.all(result.pulse_widths <= params[0].pulse_width_max)


def test_unreachable_task_is_reported_without_inventing_successful_cycles():
    params = muscle()
    state = np.array([[.16298, 20., 1150., .067, .15]])
    predictor = CompactMusclePredictor((interval((100.,)),), (params,))
    result = predictor.rollout(state, horizon_cycles=5)
    assert result.status == "infeasible"
    assert result.completed_cycles == 0
    assert result.first_failure["interval_index"] == 0
    assert np.all(np.isnan(result.pulse_widths))


def test_invalid_domains_and_zero_fatigue_coefficient_are_explicit():
    with pytest.raises(ValueError, match="alpha_a"):
        fatigue_memory_coordinates([1200., .06, .137], replace(muscle().fatigue, alpha_a=0.))
    predictor = CompactMusclePredictor((interval(),), (muscle(),))
    with pytest.raises(ValueError, match="positive"):
        predictor.phase_map([[.16, 20., -1., .06, .137]], 0)


@pytest.mark.parametrize("gain", [0.0, -0.0186])
def test_signed_source_gain_matches_full_ding_without_changing_ode(gain):
    base = muscle()
    params = replace(base, fatigue=replace(base.fatigue, alpha_a=0., alpha_tau1=0., alpha_km=0.))
    state = np.r_[.16298, 20., params.fatigue.rest_state][None, :]
    phase = replace(interval(), mechanical_gains=(gain,))
    pw = .00035
    truth = propagate_ding_pulse_width_interval(
        state[0], pulse_width=pw, duration=phase.duration,
        calcium_amplitude=phase.calcium_amplitudes[0], mechanical_gain=gain,
        parameters=params, integration_substeps=256,
    )
    predictor = CompactMusclePredictor((phase,), (params,), substeps=256)
    response = predictor.phase_map(state, 0)
    recruitment = [-np.expm1(-(pw - params.pd0) / params.pdt)]
    np.testing.assert_allclose(response.endpoint(recruitment)[0], truth, atol=1e-6, rtol=1e-6)
    if gain < 0.0:
        assert response.slope[0, 1] < 0.0


def test_signed_gain_recruitment_envelope_preserves_force_domain():
    params = muscle()
    state = np.r_[.16298, .005, params.fatigue.rest_state][None, :]
    phase = replace(interval(), mechanical_gains=(-.0186,))
    predictor = CompactMusclePredictor((phase,), (params,), substeps=128)
    response = predictor.phase_map(state, 0)
    assert 0 < response.maximum_recruitment[0] < predictor.maximum_recruitment[0]
    feasible = response.endpoint(response.maximum_recruitment * .99)
    assert feasible[0, 1] > 0.0
    with pytest.raises(ValueError, match="bounds"):
        response.endpoint(predictor.maximum_recruitment)
    # Verify a feasible interior command independently with all fatigue states.
    pw = params.pd0 - params.pdt * np.log1p(-response.maximum_recruitment[0] * .99)
    truth = propagate_ding_pulse_width_interval(
        state[0], pulse_width=pw, duration=phase.duration,
        calcium_amplitude=phase.calcium_amplitudes[0], mechanical_gain=-.0186,
        parameters=params, integration_substeps=256,
    )
    assert truth[1] > 0.0
