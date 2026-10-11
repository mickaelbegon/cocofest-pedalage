import numpy as np
import pytest

from cocofest.optimization.ding_slow_manifold import DingSlowFatigueManifold
from cocofest.models.ding2007.ding2007_with_fatigue import DingModelPulseWidthFrequencyWithFatigue


@pytest.fixture
def manifold():
    # alpha_a is negative in the Ding fatigue model; the other slow-state
    # coefficients are positive.  This checks the sign-sensitive mapping.
    return DingSlowFatigueManifold(4920., .060, .137, -.4, 2.1e-6, 1.9e-6)


def test_capacity_round_trip_stays_on_exact_slow_manifold(manifold):
    state = manifold.states_from_capacity_ratio(.73)
    assert manifold.capacity_ratio_from_states(*state) == pytest.approx(.73)
    assert manifold.invariant_residual(*state) <= 1e-10


def test_off_manifold_A_only_perturbation_is_rejected(manifold):
    a, tau1, km = manifold.states_from_capacity_ratio(.8)
    with pytest.raises(ValueError, match="off the exact"):
        manifold.capacity_ratio_from_states(a * .99, tau1, km)


def test_capacity_interval_intersects_all_three_sign_sensitive_bounds(manifold):
    lower_state = manifold.states_from_capacity_ratio(.6)
    upper_state = manifold.states_from_capacity_ratio(.9)
    lo, hi = manifold.capacity_interval(np.minimum(lower_state, upper_state), np.maximum(lower_state, upper_state))
    assert lo == pytest.approx(.6)
    assert hi == pytest.approx(.9)


def test_nonintersecting_bounds_fail(manifold):
    state_a = manifold.states_from_capacity_ratio(.4)
    state_b = manifold.states_from_capacity_ratio(.8)
    lower, upper = np.minimum(state_a, state_b), np.maximum(state_a, state_b)
    lower[0], upper[0] = manifold.states_from_capacity_ratio(.95)[0], manifold.states_from_capacity_ratio(1.)[0]
    with pytest.raises(ValueError, match="no intersection"):
        manifold.capacity_interval(lower, upper)


def test_repository_ding_derivatives_preserve_the_manifold():
    model = DingModelPulseWidthFrequencyWithFatigue(muscle_name="Biceps")
    manifold = DingSlowFatigueManifold.from_model(model)
    a, tau1, km = manifold.states_from_capacity_ratio(.71)
    force = 71.3
    q_dot = np.asarray((
        float(model.a_dot_fun(a, force)) / model.alpha_a,
        float(model.tau1_dot_fun(tau1, force)) / model.alpha_tau1,
        float(model.km_dot_fun(km, force)) / model.alpha_km,
    ))
    assert q_dot[0] == pytest.approx(q_dot[1], abs=1e-12)
    assert q_dot[0] == pytest.approx(q_dot[2], abs=1e-12)
