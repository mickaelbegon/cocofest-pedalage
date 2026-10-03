import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.ding_slow_cycle import (
    _exponential_moments, coupled_slow_offsets, slow_cycle_map_from_collocation,
)


def parameters(tau=127.0):
    return DingFatigueParameters(4000, .06, .137, -.4, 2.1e-5, 1.9e-5, tau)


@pytest.mark.parametrize("ratio", [0, 1e-12, .001, .9, 4, 4.01, 20, 1000])
def test_exponential_moments_against_independent_quadrature(ratio):
    actual = _exponential_moments(5, ratio)
    reference = [quad(lambda x: np.exp(-ratio * (1-x)) * x**k, 0, 1, epsabs=1e-14)[0]
                 for k in range(6)]
    np.testing.assert_allclose(actual, reference, rtol=2e-11, atol=1e-14)


def test_constant_force_and_polynomial_force_against_dop853():
    p = parameters()
    nodes = np.array([0, .04, .20, .48, .80, 1])
    durations = np.array([.13, .06, .41])
    functions = [lambda u: 45 + 0*u, lambda u: 10 + 120*u - 90*u**2,
                 lambda u: 55 + 6*u**3 - 12*u**4 + 18*u**5]
    cycle = slow_cycle_map_from_collocation(np.array([f(nodes) for f in functions]), durations, nodes, p)
    initial = np.array([3250, .097, .15])
    exact = initial.copy()
    for duration, force in zip(durations, functions):
        result = solve_ivp(lambda t, z: -(z-p.rest_state)/p.tau_fat+p.alpha*force(t/duration),
                           (0, duration), exact, method="DOP853", rtol=1e-12, atol=1e-13)
        assert result.success
        exact = result.y[:, -1]
    np.testing.assert_allclose(cycle.propagate(initial), exact, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(coupled_slow_offsets(cycle.propagate(initial), p),
                               cycle.decay*coupled_slow_offsets(initial, p), rtol=1e-13, atol=1e-14)


def test_force_timing_changes_exponential_convolution_even_with_same_mean():
    p = parameters(tau=1)
    nodes = np.linspace(0, 1, 6)
    early = slow_cycle_map_from_collocation(np.array([100*(1-nodes)]), [1], nodes, p)
    late = slow_cycle_map_from_collocation(np.array([100*nodes]), [1], nodes, p)
    assert early.force_integral == pytest.approx(late.force_integral)
    assert late.weighted_force_integral > early.weighted_force_integral
    assert late.propagate(p.rest_state)[0] < early.propagate(p.rest_state)[0]
    np.testing.assert_allclose(early.propagate_with_cycle_average(p.rest_state),
                               late.propagate_with_cycle_average(p.rest_state))


def test_constant_time_repeated_map_preserves_arbitrary_initial_offsets():
    p = parameters()
    nodes = np.linspace(0, 1, 6)
    cycle = slow_cycle_map_from_collocation(np.array([30+60*nodes**2]), [1], nodes, p)
    initial = np.array([3800, .073, .142])
    current = initial.copy()
    for _ in range(500):
        current = cycle.propagate(current)
    np.testing.assert_allclose(cycle.propagate(initial, 500), current, rtol=1e-13, atol=3e-10)
    np.testing.assert_array_equal(cycle.propagate(initial, 0), initial)


@pytest.mark.parametrize("nodes,durations,forces", [
    ([0,.5,.5,1], [1], [[1,1,1,1]]),
    ([0,.5,1], [0], [[1,1,1]]),
    ([0,.5,1], [1], [[1,np.nan,1]]),
    ([0,.5,1], [1,1], [[1,1,1]]),
])
def test_invalid_archived_profiles_are_rejected(nodes, durations, forces):
    with pytest.raises(ValueError):
        slow_cycle_map_from_collocation(forces, durations, nodes, parameters())
