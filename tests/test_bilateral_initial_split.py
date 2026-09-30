import numpy as np
import pytest

from cocofest.optimization.bilateral_initial_split import (
    capacity_fatigability_measurement,
    recommend_capacity_fatigability_split,
)


def measurement(*, work=2.0, power=(3.0, 1.0), force=(2.0, 1.0), alpha=(-.1, -.1)):
    return capacity_fatigability_measurement(
        certified=True, reference_work_j=work,
        available_positive_power=np.asarray(power, dtype=float)[:, None], phase_durations=[1.0],
        current_capacity=[10.0, 10.0], rest_capacity=[10.0, 10.0], alpha_a=alpha, tau_fat=[100., 100.],
        reference_force=np.asarray(force, dtype=float)[:, None],
    )


def test_identical_certified_references_produce_an_equal_split():
    right = measurement()
    decision = recommend_capacity_fatigability_split(right, right, total_equivalent_mean_torque_nm=.4)
    assert right["status"] == "measured"
    assert decision["status"] == "proposed"
    assert decision["right_fraction"] == pytest.approx(.5)
    assert decision["right_equivalent_mean_torque_nm"] + decision["left_equivalent_mean_torque_nm"] == pytest.approx(.4)


def test_higher_force_induced_damage_reduces_the_assigned_work():
    right = measurement(alpha=(-.2, -.2))
    left = measurement(alpha=(-.05, -.05))
    decision = recommend_capacity_fatigability_split(right, left, total_equivalent_mean_torque_nm=.4)
    assert decision["right_fraction"] < .5


def test_reference_work_and_damage_scaling_leave_the_score_invariant():
    first = measurement(work=2., force=(2., 1.))
    doubled = measurement(work=4., force=(4., 2.))
    assert doubled["weighted_force_induced_damage"] == pytest.approx(2 * first["weighted_force_induced_damage"], rel=.02)
    assert doubled["endurance_work_score_j"] == pytest.approx(first["endurance_work_score_j"], rel=.02)


def test_zero_damage_and_uncertified_references_hold_instead_of_using_an_epsilon():
    zero = measurement(alpha=(0., 0.))
    assert zero["status"] == "held"
    held = recommend_capacity_fatigability_split(zero, measurement(), total_equivalent_mean_torque_nm=.4)
    assert held["status"] == "held"
    assert held["reason"] == "measurement_unavailable"


def test_projection_preserves_total_and_declared_minimum_load():
    high = measurement(alpha=(-.01, -.01))
    low = measurement(alpha=(-.5, -.5))
    decision = recommend_capacity_fatigability_split(high, low, total_equivalent_mean_torque_nm=.4,
                                                      minimum_arm_equivalent_mean_torque_nm=.15)
    assert decision["right_equivalent_mean_torque_nm"] >= .15
    assert decision["left_equivalent_mean_torque_nm"] >= .15
