import casadi as ca
import numpy as np
import pytest
from scipy.integrate import quad

from cocofest.optimization.periodic_collocation_profile import PeriodicCollocationProfile


def _profile():
    coefficients = np.array([
        [[3., .3, -.2, .07, -.01, .001], [2., -.2, .1, -.03, .01, -.002]],
        [[8., -.2, .3, -.1, .02, -.001], [7., .1, -.2, .06, -.02, .003]],
    ])
    return PeriodicCollocationProfile(0.8, coefficients)


def test_declared_radau_polynomials_values_derivatives_and_periodic_repetition():
    original = _profile()
    nodes = np.r_[0., ca.collocation_points(5, "radau")]
    samples = np.einsum("sik,nk->sin", original.coefficients, nodes[:, None] ** np.arange(6))
    profile = PeriodicCollocationProfile.from_samples(period=0.8, nodes=nodes, values=samples)
    query = np.array([.031, .217, .431, .703])
    np.testing.assert_allclose(profile.coefficients, original.coefficients, atol=5e-12)
    for index, time in enumerate(query):
        interval = int(time / .4)
        local = (time - interval * .4) / .4
        for signal in range(2):
            coefficients = original.coefficients[signal, interval]
            assert profile.evaluate(query)[signal, index] == pytest.approx(
                np.polynomial.polynomial.polyval(local, coefficients), abs=3e-14)
            assert profile.derivative(query)[signal, index] == pytest.approx(
                np.polynomial.polynomial.polyval(local, np.polynomial.polynomial.polyder(coefficients)) / .4,
                abs=3e-13)
    np.testing.assert_allclose(profile.evaluate(query - 2.4), profile.evaluate(query + 4.), atol=2e-14)


@pytest.mark.parametrize("start,duration,tau", [
    (-.13, 2.13, 127.), (.037, .077, 1e9), (.037, .077, .001),
    (.8, .4, 127.), (.031, 1e-10, 127.), (.031, 0., 127.),
])
def test_analytical_convolution_matches_independent_adaptive_quadrature(start, duration, tau):
    profile = _profile()
    actual = profile.exponentially_weighted_force_integral(start_time=start, duration=duration, time_constant=tau)
    boundaries = np.arange(np.floor(start / .4) + 1, np.ceil((start + duration) / .4)) * .4
    expected = []
    for signal in range(2):
        expected.append(quad(
            lambda time: np.exp(-(start + duration - time) / tau) * profile.evaluate(time)[signal, 0],
            start, start + duration, points=boundaries.tolist(), epsabs=1e-12, epsrel=1e-12,
        )[0])
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-13)


def test_positivity_audit_finds_narrow_negative_region_and_never_clips():
    # The negative trough lies between a 1000-point grid's samples.
    location = .3712345
    depth = 1e-10
    profile = PeriodicCollocationProfile(1., np.array([[[location**2 - depth, -2*location, 1.]]]))
    assert np.min(profile.evaluate(np.linspace(0., 1., 1000))) > 0.
    certificate = profile.force_positivity_certificate(tolerance=1e-12)
    assert not certificate.certified_nonnegative[0]
    assert certificate.minimum_force[0] == pytest.approx(-depth, abs=1e-16)
    assert certificate.minimum_time[0] == pytest.approx(location)
    assert profile.evaluate(location)[0, 0] < 0.


def test_knots_keep_one_sided_values_and_derivatives_with_explicit_jumps():
    profile = PeriodicCollocationProfile(1., np.array([[[1., 2.], [4., -1.]]]))
    assert profile.evaluate(.5)[0, 0] == 4.
    assert profile.derivative(.5)[0, 0] == -2.
    assert profile.evaluate(1.)[0, 0] == 1.
    audit = profile.seam_audit()
    assert audit["value_jump"] == [[1., -2.]]
    assert audit["time_derivative_jump"] == [[-6., 6.]]


def test_integral_is_linear_in_fixed_mesh_coefficients():
    first = _profile()
    second = PeriodicCollocationProfile(first.period, first.coefficients * .2 + .1)
    combined = PeriodicCollocationProfile(first.period, 2. * first.coefficients - .3 * second.coefficients)
    kwargs = dict(start_time=.117, duration=1.92, time_constant=127.)
    np.testing.assert_allclose(combined.exponentially_weighted_force_integral(**kwargs),
        2. * first.exponentially_weighted_force_integral(**kwargs) - .3 * second.exponentially_weighted_force_integral(**kwargs),
        rtol=2e-15, atol=2e-14)


def test_packed_collocation_policy_matches_numpy_casadi_and_reuses_a_differentiable_graph():
    from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
    from cocofest.optimization.endurance_rollout import (
        DingRolloutMuscleParameters, PeriodicRecruitmentProfile, rollout_periodic_ding_endurance,
    )
    from cocofest.optimization.endurance_rollout_objective import (
        RolloutObjectiveLayout, build_rollout_objective_function, pack_rollout_objective_parameters,
    )

    muscles = [DingRolloutMuscleParameters(
        fatigue=DingFatigueParameters(a_rest=1500., tau1_rest=.060601, km_rest=.137,
            alpha_a=-.2, alpha_tau1=2.1e-5, alpha_km=1.9e-5, tau_fat=127.),
        tau2=.001, pd0=.000131405, pdt=.000194138, pulse_width_max=.0006,
    )]
    layout = RolloutObjectiveLayout(1, 2, 3)
    function = build_rollout_objective_function(muscles=muscles, layout=layout, symbolic_type="MX")
    initial = np.array([[1450., .065, .142]])
    results = []
    for offset in (0., 2.):
        coefficients = np.array([[[12. + offset, .4, -.4], [12. + offset, -.2, .2]]])
        profile = PeriodicRecruitmentProfile(
            force_profile=PeriodicCollocationProfile(.8, coefficients), interval_count=2,
            cn=np.full((1, 2), .42), force_length_relationship=np.full((1, 2), .95),
            force_velocity_relationship=np.full((1, 2), .98), passive_force_relationship=np.full((1, 2), .02),
        )
        packed = pack_rollout_objective_parameters(profile, muscles, layout)
        symbolic = function(initial.ravel(), packed)
        numerical = rollout_periodic_ding_endurance(profile, initial_slow_states=initial,
            muscles=muscles, horizon_cycles=3)
        np.testing.assert_allclose(np.asarray(symbolic[1]).ravel(), numerical.utilization.ravel(), rtol=2e-13)
        np.testing.assert_allclose(np.asarray(symbolic[2]).ravel(), numerical.cycle_boundary_states[-1].ravel(), rtol=2e-13)
        results.append(float(symbolic[0]))
    assert results[1] > results[0]
    state = ca.MX.sym("state", layout.initial_state_size)
    parameters = ca.MX.sym("profile", layout.parameter_size)
    gradient = ca.Function("collocation_gradient", [state, parameters],
        [ca.gradient(function(state, parameters)[0], parameters)])
    evaluated = np.asarray(gradient(initial.ravel(), packed)).ravel()
    assert np.all(np.isfinite(evaluated))
    delta = np.zeros_like(packed)
    delta[0] = 1e-4
    finite_difference = (float(function(initial.ravel(), packed + delta)[0])
        - float(function(initial.ravel(), packed - delta)[0])) / 2e-4
    assert evaluated[0] == pytest.approx(finite_difference, rel=1e-7, abs=1e-10)
