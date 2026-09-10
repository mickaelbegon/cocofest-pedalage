from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.fixed_force_cycle_map import FixedForceCycleMap


def parameters():
    return DingFatigueParameters(1200., .060601, .137, -1.4, 2.1e-5, 1.9e-5, 445.5)


def test_repeated_cycle_matches_independent_phase_integration_and_preserves_offsets():
    params = (parameters(), replace(parameters(), tau_fat=320.))
    durations = np.array([.2, .5, .3])
    forces = np.array([[3., 8.], [9., 2.], [4., 5.]])
    initial = np.array([[1150., .078, .15], [1120., .072, .16]])
    model = FixedForceCycleMap.from_piecewise_constant_forces(params, durations, forces)
    current = initial.copy()
    minima = initial.copy()
    for _ in range(120):
        for dt, force in zip(durations, forces):
            for muscle, p in enumerate(params):
                current[muscle] = solve_ivp(
                    lambda t, z: -(z - p.rest_state) / p.tau_fat + p.alpha * force[muscle],
                    (0., dt), current[muscle], method="DOP853", rtol=1e-11, atol=1e-12,
                ).y[:, -1]
            minima = np.minimum(minima, current)
    result = model.project(initial, 120)
    np.testing.assert_allclose(result.slow_states, current, rtol=2e-11, atol=2e-10)
    np.testing.assert_allclose(result.minimum_slow_states_by_muscle, minima, rtol=2e-11, atol=2e-10)
    assert result.slow_domain_valid_at_phase_endpoints
    assert result.force_and_pw_feasibility_checked is False
    for m, p in enumerate(params):
        old_offset = initial[m, 1:] - p.rest_state[1:] - p.alpha[1:] / p.alpha_a * (initial[m, 0] - p.a_rest)
        new_offset = result.slow_states[m, 1:] - p.rest_state[1:] - p.alpha[1:] / p.alpha_a * (result.slow_states[m, 0] - p.a_rest)
        np.testing.assert_allclose(new_offset, old_offset * np.exp(-120. / p.tau_fat), atol=1e-13)


def test_force_phase_order_matters_even_when_mean_force_matches():
    p = replace(parameters(), tau_fat=1.)
    early = FixedForceCycleMap.from_piecewise_constant_forces((p,), [.5, .5], [[10.], [0.]])
    late = FixedForceCycleMap.from_piecewise_constant_forces((p,), [.5, .5], [[0.], [10.]])
    a = early.project(p.rest_state[None, :], 1)
    b = late.project(p.rest_state[None, :], 1)
    assert a.slow_states[0, 0] > b.slow_states[0, 0]


def test_long_low_decay_ratio_is_stable_and_zero_cycles_is_identity():
    p = replace(parameters(), tau_fat=1e15)
    model = FixedForceCycleMap.from_piecewise_constant_forces((p,), [.01], [[2.]])
    initial = p.rest_state[None, :]
    np.testing.assert_array_equal(model.project(initial, 0).slow_states, initial)
    result = model.project(initial, 300)
    np.testing.assert_allclose(result.slow_states[0], p.rest_state + p.alpha * 6., rtol=1e-14)


def test_invalid_domain_is_reported_without_claiming_pw_feasibility():
    p = parameters()
    model = FixedForceCycleMap.from_piecewise_constant_forces((p,), [1.], [[100.]])
    result = model.project(p.rest_state[None, :], 300)
    assert not result.slow_domain_valid_at_phase_endpoints
    assert result.slow_states[0, 0] < 0
    assert not result.force_and_pw_feasibility_checked


def test_phase_domain_failure_cannot_be_hidden_by_recovery_at_cycle_boundary():
    p = replace(parameters(), tau_fat=.5)
    model = FixedForceCycleMap.from_piecewise_constant_forces((p,), [.02, 2.], [[1e5], [0.]])
    result = model.project(p.rest_state[None, :], 3)
    assert np.all(result.slow_states > 0)
    assert result.minimum_slow_states_by_muscle[0, 0] < 0
    assert not result.slow_domain_valid_at_phase_endpoints


@pytest.mark.parametrize("cycles", [-1, 1.2, True])
def test_invalid_repetition_counts_rejected(cycles):
    p = parameters()
    model = FixedForceCycleMap.from_piecewise_constant_forces((p,), [1.], [[2.]])
    with pytest.raises(ValueError):
        model.project(p.rest_state[None, :], cycles)


@pytest.mark.parametrize("durations,doses", [([0.], [[1.]]), ([np.nan], [[1.]]), ([1.], [[-1.]]), ([1.], [[np.inf]]), ([1.], [[1., 2.]])])
def test_invalid_phase_inputs_rejected(durations, doses):
    with pytest.raises(ValueError):
        FixedForceCycleMap((parameters(),), durations, doses)
