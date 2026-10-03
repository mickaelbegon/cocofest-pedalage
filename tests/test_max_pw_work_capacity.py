import math
import time

import casadi as ca
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.max_pw_work_capacity import (
    MaxPwWorkCapacityLayout, build_max_pw_work_capacity_function,
    max_pw_work_domain_lower_bounds, pack_max_pw_work_profile,
)


def muscle():
    return DingPulseWidthParameters(DingFatigueParameters(
        a_rest=4920., tau1_rest=.060601, km_rest=.137, alpha_a=-.04,
        alpha_tau1=2.1e-6, alpha_km=1.9e-6, tau_fat=127.),
        tauc=.011, tau2=.001, pd0=.000131405, pdt=.000194138, pulse_width_max=.0006)


def example(frequency=30, muscles=1, substeps=16, policy="all_intervals_pw_max"):
    parameters = (muscle(),) * muscles
    layout = MaxPwWorkCapacityLayout(muscles, frequency, substeps)
    intervals, coefficients = [], []
    for k in range(frequency):
        gains = tuple(lambda t, k=k, m=m: .95 + .05*np.cos(2*np.pi*(k/frequency+t)+m)
                      for m in range(muscles))
        bs = tuple(lambda t, k=k, m=m: -.04*np.sin(2*np.pi*(k/frequency+t)+m)
                   for m in range(muscles))
        coefficients.append(bs)
        intervals.append(MomentTrackingInterval(1/frequency, (1.06,)*muscles, gains,
                         tuple(b(.5/frequency) for b in bs), (0.,)*muscles))
    profile = pack_max_pw_work_profile(intervals, layout, angular_velocity_rad_s=-2*np.pi,
                                      moment_coefficient_functions=coefficients,
                                      stimulation_policy=policy)
    state = np.tile([.16, 20., 4400., .065, .15], muscles)
    return parameters, layout, intervals, coefficients, profile, state


@pytest.mark.parametrize("frequency", [30, 50])
@pytest.mark.parametrize("policy", ["all_intervals_pw_max", "selected_propulsive_intervals"])
def test_full_cycle_dop853_reference(frequency, policy):
    parameters, layout, intervals, coefficients, profile, state = example(frequency, policy=policy)
    function = build_max_pw_work_capacity_function(muscles=parameters, layout=layout)
    actual = function(state, profile)
    p = parameters[0]
    expected = np.r_[state, 0.]
    mask = profile[layout.slices()["stimulation_mask"]]
    for k, interval in enumerate(intervals):
        pw = p.pd0 + mask[k]*(p.pulse_width_max-p.pd0)
        def rhs(t, x):
            cn, force, a, tau1, km, work = x
            activation = cn/(km+cn)
            recruitment = a*(1-np.exp(-(pw-p.pd0)/p.pdt))
            return [(interval.calcium_amplitudes[0]*np.exp(-t/p.tauc)-cn)/p.tauc,
                    interval.mechanical_gains[0](t)*(recruitment*activation-force/(tau1+p.tau2*activation)),
                    -(a-p.fatigue.a_rest)/p.fatigue.tau_fat+p.fatigue.alpha_a*force,
                    -(tau1-p.fatigue.tau1_rest)/p.fatigue.tau_fat+p.fatigue.alpha_tau1*force,
                    -(km-p.fatigue.km_rest)/p.fatigue.tau_fat+p.fatigue.alpha_km*force,
                    max(-2*np.pi*coefficients[k][0](t), 0)*force]
        solved = solve_ivp(rhs, [0, interval.duration], expected, method="DOP853", rtol=1e-11, atol=1e-12)
        assert solved.success
        expected = solved.y[:, -1]
    # 0.003% relative-error gate is stricter than the desired work proxy
    # precision; it is not an NLP feasibility tolerance.
    np.testing.assert_allclose(np.asarray(actual[2]).ravel(), expected[:5], rtol=3e-5, atol=1e-8)
    np.testing.assert_allclose(float(actual[0]), expected[5], rtol=3e-5, atol=1e-8)
    print(f"DOP853 {frequency}Hz {policy}: force_relative_error={abs(float(actual[2][1])-expected[1])/expected[1]:.3g} work_relative_error={abs(float(actual[0])-expected[5])/expected[5]:.3g}")
    assert np.all(np.asarray(actual[3]).ravel() >= max_pw_work_domain_lower_bounds(layout))


@pytest.mark.parametrize("symbolic_type", ["SX", "MX"])
def test_terminal_state_gradient_matches_finite_difference(symbolic_type):
    parameters, layout, _, _, profile, state = example(muscles=2)
    f = build_max_pw_work_capacity_function(muscles=parameters, layout=layout, symbolic_type=symbolic_type)
    x = getattr(ca, symbolic_type).sym("state", state.size)
    gradient = ca.Function("gradient", [x], [ca.gradient(f(x, profile)[0], x)])
    observed = np.asarray(gradient(state)).ravel()
    expected = []
    for j, value in enumerate(state):
        h = 1e-5*max(abs(value), .01)
        displacement = np.zeros(state.size)
        displacement[j] = h
        expected.append((float(f(state+displacement, profile)[0])-float(f(state-displacement, profile)[0]))/(2*h))
    np.testing.assert_allclose(observed, expected, rtol=2e-5, atol=1e-7)
    assert observed[2] > 0  # More capacity preserves more projected work.


def test_sign_mask_and_sum_are_physical():
    parameters, layout, intervals, _, _, state = example(muscles=2)
    all_negative = [[-.04, -.02] for _ in intervals]
    function = build_max_pw_work_capacity_function(muscles=parameters, layout=layout)
    propulsive = pack_max_pw_work_profile(intervals, layout, angular_velocity_rad_s=-2*np.pi,
                                        moment_coefficient_functions=all_negative)
    braking = pack_max_pw_work_profile(intervals, layout, angular_velocity_rad_s=2*np.pi,
                                     moment_coefficient_functions=all_negative)
    result = function(state, propulsive)
    assert float(result[0]) > 0
    assert float(result[0]) == pytest.approx(np.sum(result[1]))
    assert float(function(state, braking)[0]) == 0
    selected = pack_max_pw_work_profile(intervals, layout, angular_velocity_rad_s=2*np.pi,
                                       moment_coefficient_functions=all_negative,
                                       stimulation_policy="selected_propulsive_intervals")
    assert np.all(selected[layout.slices()["stimulation_mask"]] == 0)
    assert float(function(state, selected)[0]) == 0


def test_policies_differ_and_reuse_one_graph():
    parameters, layout, _, _, profile, state = example(muscles=2)
    selected = example(muscles=2, policy="selected_propulsive_intervals")[4]
    f = build_max_pw_work_capacity_function(muscles=parameters, layout=layout)
    assert np.all(profile[layout.slices()["stimulation_mask"]] == 1)
    assert set(selected[layout.slices()["stimulation_mask"]]) == {0, 1}
    initial_nodes = f.n_nodes()
    assert not np.isclose(float(f(state, profile)[0]), float(f(state, selected)[0]))
    assert f.n_nodes() == initial_nodes
    # Fully fatigued but physically valid state has less predicted work.
    depleted = state.copy()
    depleted[2] *= .5
    assert float(f(depleted, profile)[0]) < float(f(state, profile)[0])


def test_invalid_profile_and_layout_fail_closed():
    with pytest.raises(ValueError):
        MaxPwWorkCapacityLayout(1, 30, 0)
    parameters, layout, intervals, _, _, _ = example()
    with pytest.raises(ValueError, match="binary"):
        pack_max_pw_work_profile(intervals, layout, angular_velocity_rad_s=-1,
                                 stimulation_mask=np.full((1,30), .5))
    with pytest.raises(ValueError, match="nonzero"):
        pack_max_pw_work_profile(intervals, layout, angular_velocity_rad_s=0)


def test_four_muscle_microbenchmark_reports_cost(capsys):
    parameters, layout, _, _, profile, state = example(muscles=4)
    started = time.perf_counter()
    f = build_max_pw_work_capacity_function(muscles=parameters, layout=layout)
    build_s = time.perf_counter()-started
    x = ca.SX.sym("x", state.size)
    objective = f(x, profile)[0]
    g = ca.Function("gradient", [x], [ca.gradient(objective, x)])
    f(state, profile)
    started = time.perf_counter()
    for _ in range(20):
        f(state, profile)
    value_ms = 1000*(time.perf_counter()-started)/20
    started = time.perf_counter()
    for _ in range(20):
        g(state)
    gradient_ms = 1000*(time.perf_counter()-started)/20
    with capsys.disabled():
        print(f"max-PW work proxy: build={build_s:.3f}s value={value_ms:.3f}ms gradient={gradient_ms:.3f}ms nodes={f.n_nodes()}")
    assert np.isfinite(float(f(state, profile)[0]))
