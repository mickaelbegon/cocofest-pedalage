import numpy as np

from scripts.validate_radau_coupled_mechanics import (
    collocation_coefficients,
    periodic_node_history_amplitude,
    radau_collocation_step,
    radau_nodes,
)


def test_lagrange_coefficients_integrate_constants_and_end_at_one():
    nodes = radau_nodes(5)
    derivative, endpoint = collocation_coefficients(nodes)
    assert np.allclose(np.sum(derivative, axis=0), 0.0, atol=1e-12)
    assert np.isclose(np.sum(endpoint), 1.0, atol=1e-12)
    assert np.allclose(endpoint, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_higher_radau_degree_improves_fixed_rhs_endpoint_accuracy():
    exact = np.array([np.exp(-0.6)])
    rhs = lambda _time, state: -2.0 * state
    end3, _, _ = radau_collocation_step(rhs, np.array([1.0]), duration=0.3, degree=3)
    end5, _, _ = radau_collocation_step(rhs, np.array([1.0]), duration=0.3, degree=5)
    assert abs(end5[0] - exact[0]) < abs(end3[0] - exact[0])
    assert abs(end5[0] - exact[0]) < 1e-9


def test_periodic_history_amplitude_matches_the_documented_truncated_sum():
    interval, tauc, km_rest, truncation = 0.02, 0.011, 0.137, 6
    decay = np.exp(-interval / tauc)
    increment = 1.0 + (km_rest + 1.04 - 1.0) * decay
    expected = decay**5 + increment * sum(decay**age for age in range(5))
    assert periodic_node_history_amplitude(
        interval_s=interval, tauc=tauc, km_rest=km_rest, truncation=truncation
    ) == expected
