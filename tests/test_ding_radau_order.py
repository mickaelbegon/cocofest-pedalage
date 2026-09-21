import math

import numpy as np

from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from scripts.validate_ding_fixed_pulse_width import FixedPulseWidthDingCase, radau_collocation_step
from scripts.validate_ding_radau_order import propagate, radau_step


def case():
    parameters = DingPulseWidthParameters(
        fatigue=DingFatigueParameters(a_rest=4920., tau1_rest=.060601, km_rest=.137,
                                     alpha_a=-.04, alpha_tau1=2.1e-6,
                                     alpha_km=1.9e-6, tau_fat=127.),
        tauc=.011, tau2=.001, pd0=.000131405, pdt=.000194138, pulse_width_max=.0006,
    )
    return FixedPulseWidthDingCase("test", np.array([.25, 20., 4920., .060601, .137]),
                                  np.array([.00035, .0004]), 1/30, 1.05, parameters)


def test_butcher_map_matches_existing_collocation_equations():
    data = case()
    kwargs = dict(degree=5, pulse_width=data.pulse_widths[0], duration_s=data.duration_s,
                  calcium_amplitude=data.calcium_amplitude, parameters=data.parameters)
    butcher, _ = radau_step(data.initial_state, **kwargs)
    direct, _ = radau_collocation_step(data.initial_state, **kwargs)
    np.testing.assert_allclose(butcher, direct, rtol=1e-10, atol=1e-10)


def test_subdivision_preserves_pulse_count_and_exact_calcium_history():
    data = case()
    result, residual = propagate(data, degree=5, subdivisions=8)
    # Exact double-pole calcium solution, continuous Cn across each pulse.
    exact = data.initial_state[0]
    for _ in data.pulse_widths:
        exact = math.exp(-data.duration_s / data.parameters.tauc) * (
            exact + data.calcium_amplitude * data.duration_s / data.parameters.tauc)
    assert result.shape == (3, 5)
    assert residual < 1e-10
    np.testing.assert_allclose(result[-1, 0], exact, atol=1e-11, rtol=1e-10)
