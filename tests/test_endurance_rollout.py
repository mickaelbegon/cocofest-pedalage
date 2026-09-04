import numpy as np
import pytest

from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters, propagate_ding_fatigue
from cocofest.optimization.endurance_rollout import (
    DingRolloutMuscleParameters,
    PeriodicRecruitmentProfile,
    rollout_periodic_ding_endurance,
)
from cocofest.optimization.periodic_force_profile import PeriodicFourierForceProfile
from cocofest.optimization.recruitment_margin import RecruitmentStatus, ding_recruitment_margin


def _fatigue(a_rest=4920.0):
    return DingFatigueParameters(
        a_rest=a_rest,
        tau1_rest=0.060601,
        km_rest=0.137,
        alpha_a=-4.0e-2,
        alpha_tau1=2.1e-6,
        alpha_km=1.9e-6,
        tau_fat=127.0,
    )


def _muscle(a_rest=4920.0, pulse_width_max=0.0006):
    return DingRolloutMuscleParameters(
        fatigue=_fatigue(a_rest),
        tau2=0.001,
        pd0=0.000131405,
        pdt=0.000194138,
        pulse_width_max=pulse_width_max,
    )


def _profile(mean, *, sine=None, interval_count=4):
    mean = np.asarray(mean, dtype=float)
    muscle_count = mean.size
    if sine is None:
        sine = np.zeros((muscle_count, 1))
    force_profile = PeriodicFourierForceProfile(
        period=0.8,
        mean=mean,
        cosine=np.zeros((muscle_count, 1)),
        sine=np.asarray(sine, dtype=float),
    )
    shape = (muscle_count, interval_count)
    return PeriodicRecruitmentProfile(
        force_profile=force_profile,
        interval_count=interval_count,
        cn=np.full(shape, 0.40),
        force_length_relationship=np.full(shape, 0.92),
        force_velocity_relationship=np.full(shape, 0.95),
        passive_force_relationship=np.full(shape, 0.03),
    )


def test_midpoint_state_and_cycle_boundaries_use_two_exact_fourier_half_steps():
    profile = _profile([20.0], interval_count=2)
    muscle = _muscle()
    initial = np.array([[4800.0, 0.065, 0.145]])

    result = rollout_periodic_ding_endurance(
        profile,
        initial_slow_states=initial,
        muscles=[muscle],
        horizon_cycles=2,
    )

    half_step = profile.interval_duration / 2.0
    expected_midpoint = propagate_ding_fatigue(initial[0], 20.0, half_step, muscle.fatigue)
    expected_after_interval = propagate_ding_fatigue(expected_midpoint, 20.0, half_step, muscle.fatigue)
    expected_after_cycle = propagate_ding_fatigue(
        propagate_ding_fatigue(initial[0], 20.0, profile.interval_duration, muscle.fatigue),
        20.0,
        profile.interval_duration,
        muscle.fatigue,
    )

    np.testing.assert_allclose(result.midpoint_states[0, 0, 0], expected_midpoint, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(result.midpoint_states[0, 1, 0], propagate_ding_fatigue(expected_after_interval, 20.0, half_step, muscle.fatigue), atol=1e-13)
    np.testing.assert_allclose(result.cycle_boundary_states[0], initial, atol=0.0)
    np.testing.assert_allclose(result.cycle_boundary_states[1, 0], expected_after_cycle, atol=1e-13)
    np.testing.assert_allclose(result.cycle_boundary_times, [0.0, 0.8, 1.6])
    assert result.feasible


def test_rollout_uses_analytic_force_derivative_and_retains_every_status():
    profile = _profile([22.0], sine=[[4.0]], interval_count=4)
    muscle = _muscle()
    initial = np.array([[4800.0, 0.065, 0.145]])
    result = rollout_periodic_ding_endurance(
        profile,
        initial_slow_states=initial,
        muscles=[muscle],
        horizon_cycles=1,
    )

    phase = 0
    midpoint = result.midpoint_states[0, phase, 0]
    direct = ding_recruitment_margin(
        cn=profile.cn[0, phase],
        force=result.midpoint_forces[0, phase],
        force_derivative=result.midpoint_force_derivatives[0, phase],
        capacity=midpoint[0],
        tau1=midpoint[1],
        km=midpoint[2],
        tau2=muscle.tau2,
        pd0=muscle.pd0,
        pdt=muscle.pdt,
        pulse_width_max=muscle.pulse_width_max,
        force_length_relationship=profile.force_length_relationship[0, phase],
        force_velocity_relationship=profile.force_velocity_relationship[0, phase],
        passive_force_relationship=profile.passive_force_relationship[0, phase],
    )

    assert result.midpoint_force_derivatives[0, phase] != 0.0
    assert result.diagnostics[0][phase][0] == direct
    assert result.diagnostics[0][phase][0].status is RecruitmentStatus.OK
    assert result.utilization[0, phase, 0] == pytest.approx(direct.utilization)


def test_marginal_damage_rate_is_the_force_driven_ding_term_not_recovery():
    profile = _profile([20.0, 35.0], interval_count=2)
    muscles = [_muscle(a_rest=3000.0), _muscle(a_rest=4000.0)]
    result = rollout_periodic_ding_endurance(
        profile,
        initial_slow_states=[[2900.0, 0.065, 0.145], [3500.0, 0.070, 0.150]],
        muscles=muscles,
        horizon_cycles=3,
    )
    expected = np.array([0.04 * 20.0 / 3000.0, 0.04 * 35.0 / 4000.0])

    np.testing.assert_allclose(result.marginal_damage_rate[0, 0], expected)
    np.testing.assert_allclose(
        result.marginal_damage_rate,
        np.broadcast_to(result.marginal_damage_rate[0], result.marginal_damage_rate.shape),
    )
    # The result depends on alpha_a, F, and A_rest only; changing the current
    # capacity does not introduce the separate recovery term into this metric.
    assert result.marginal_damage_rate[0, 0, 0] == pytest.approx(expected[0])


def test_first_failure_and_worst_utilization_are_explicit_and_unclipped():
    profile = _profile([45.0, 10.0], interval_count=3)
    muscles = [_muscle(a_rest=120.0), _muscle()]
    initial = np.array([[120.0, 0.065, 0.145], [4900.0, 0.065, 0.145]])

    result = rollout_periodic_ding_endurance(
        profile,
        initial_slow_states=initial,
        muscles=muscles,
        horizon_cycles=2,
    )

    failure = result.first_failure
    assert failure is not None
    assert (failure.cycle_index, failure.interval_index, failure.muscle_index) == (0, 0, 0)
    assert failure.status in {
        RecruitmentStatus.RECRUITMENT_EXCEEDS_CAPACITY,
        RecruitmentStatus.PULSE_WIDTH_LIMIT_EXCEEDED,
        RecruitmentStatus.RECRUITMENT_AT_CAPACITY_ASYMPTOTE,
    }
    assert result.utilization[0, 0, 0] > 1.0
    assert result.worst_utilization > 1.0
    assert result.worst_utilization_location is not None
    assert not result.feasible


def test_permuting_muscles_only_permutes_the_rollout_result():
    profile = _profile([18.0, 24.0], sine=[[2.0], [1.0]], interval_count=4)
    muscles = [_muscle(), _muscle()]
    initial = np.array([[4800.0, 0.065, 0.145], [4600.0, 0.070, 0.150]])
    baseline = rollout_periodic_ding_endurance(
        profile,
        initial_slow_states=initial,
        muscles=muscles,
        horizon_cycles=3,
    )
    permutation = np.array([1, 0])
    permuted_profile = PeriodicRecruitmentProfile(
        force_profile=PeriodicFourierForceProfile(
            period=profile.force_profile.period,
            mean=profile.force_profile.mean[permutation],
            cosine=profile.force_profile.cosine[permutation],
            sine=profile.force_profile.sine[permutation],
        ),
        interval_count=profile.interval_count,
        cn=profile.cn[permutation],
        force_length_relationship=profile.force_length_relationship[permutation],
        force_velocity_relationship=profile.force_velocity_relationship[permutation],
        passive_force_relationship=profile.passive_force_relationship[permutation],
    )
    permuted = rollout_periodic_ding_endurance(
        permuted_profile,
        initial_slow_states=initial[permutation],
        muscles=[muscles[index] for index in permutation],
        horizon_cycles=3,
    )

    np.testing.assert_allclose(permuted.cycle_boundary_states, baseline.cycle_boundary_states[:, permutation])
    np.testing.assert_allclose(permuted.midpoint_states, baseline.midpoint_states[:, :, permutation])
    np.testing.assert_allclose(permuted.utilization, baseline.utilization[:, :, permutation], equal_nan=True)
    np.testing.assert_allclose(permuted.marginal_damage_rate, baseline.marginal_damage_rate[:, :, permutation])
    np.testing.assert_allclose(permuted.midpoint_forces, baseline.midpoint_forces[permutation])
    np.testing.assert_allclose(permuted.midpoint_force_derivatives, baseline.midpoint_force_derivatives[permutation])


def test_negative_reconstructed_force_is_rejected_instead_of_clipped():
    profile = _profile([-1.0], interval_count=2)
    with pytest.raises(ValueError, match="not certified non-negative"):
        rollout_periodic_ding_endurance(
            profile,
            initial_slow_states=[[4800.0, 0.065, 0.145]],
            muscles=[_muscle()],
            horizon_cycles=1,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon_cycles": 0},
        {"horizon_cycles": True},
        {"horizon_cycles": 1, "initial_slow_states": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]},
        {"horizon_cycles": 1, "muscles": []},
    ],
)
def test_rollout_rejects_incompatible_fixed_policy_inputs(kwargs):
    profile = _profile([20.0], interval_count=2)
    arguments = dict(
        initial_slow_states=[[4800.0, 0.065, 0.145]],
        muscles=[_muscle()],
        horizon_cycles=1,
    )
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        rollout_periodic_ding_endurance(profile, **arguments)


@pytest.mark.parametrize("cycles", [1, 100])
def test_continuous_fourier_slow_rollout_matches_dop853_at_midpoints_and_boundaries(cycles):
    solve_ivp = pytest.importorskip("scipy.integrate").solve_ivp
    profile = _profile([22.0], sine=[[4.0]], interval_count=4)
    muscle = _muscle()
    initial = np.array([[4800.0, 0.065, 0.145]])
    result = rollout_periodic_ding_endurance(
        profile,
        initial_slow_states=initial,
        muscles=[muscle],
        horizon_cycles=cycles,
    )

    def rhs(time, state):
        force = profile.force_profile.evaluate(time)[0, 0]
        return -(state - muscle.fatigue.rest_state) / muscle.fatigue.tau_fat + muscle.fatigue.alpha * force

    solution = solve_ivp(
        rhs,
        (0.0, cycles * profile.force_profile.period),
        initial[0],
        method="DOP853",
        rtol=2e-12,
        atol=2e-13,
        dense_output=True,
    )
    assert solution.success
    boundary_reference = solution.sol(result.cycle_boundary_times).T
    midpoint_times = (
        np.arange(cycles, dtype=float)[:, None] * profile.force_profile.period + profile.midpoint_times[None, :]
    ).ravel()
    midpoint_reference = solution.sol(midpoint_times).T.reshape(cycles, profile.interval_count, 3)

    np.testing.assert_allclose(result.cycle_boundary_states[:, 0], boundary_reference, rtol=5e-11, atol=5e-11)
    np.testing.assert_allclose(result.midpoint_states[:, :, 0], midpoint_reference, rtol=5e-11, atol=5e-11)
