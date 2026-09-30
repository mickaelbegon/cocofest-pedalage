import math

import pytest
import numpy as np

from cocofest.optimization.independent_arm_rho_pace import (
    BilateralArmPaceConfig, BilateralArmPaceController,
    IndependentArmResistancePace, IndependentArmRhoPaceConfig,
)
from cocofest.optimization.rho_pace import RhoPaceController


def policy(side="right", **kwargs):
    return BilateralArmPaceController(side, ("m0", "m1", "m2", "m3"), (1.,) * 4,
        equivalent_mean_torque_nm=.1, initial_weight_basis="uniform test fixture", **kwargs)


def apply(weights):
    return {"ocp_cost_updated": True, "weights": weights}


def boundary(controller, cycles, *, ratios=(.4, .6, .8, 1.), torque=.1, certified=True, writer=apply):
    return controller.boundary(cycles, ratios, certified=certified,
        equivalent_mean_torque_nm=torque, apply_weights=writer)


def test_two_local_four_weight_vectors_change_only_at_ten_certified_windows():
    right, left = policy("right"), policy("left")
    written = {"right": [], "left": []}
    for index in range(22):
        torque = .1 if index < 10 else .13
        for controller in (right, left):
            def writer(weights):
                written[controller.side].append(index)
                return apply(weights)
            boundary(controller, index, torque=torque,
                     ratios=(.4, .6, .8, 1.) if controller.side == "right" else (1.,) * 4,
                     writer=writer)
    assert written == {"right": [0, 10, 20], "left": [0, 10, 20]}
    assert len(right.weights) == len(left.weights) == 4
    assert right.weights[0] > right.weights[-1]
    assert left.weights == pytest.approx((1.,) * 4)
    assert math.prod(right.weights) == pytest.approx(1.)
    for event in right.events:
        if event.get("status") == "applied":
            assert max(abs(math.log(a / b)) for a, b in zip(event["weights"], event["weights_before"])) <= right.config.max_log_step + 1e-12


def test_certification_failure_does_not_consume_a_cycle_or_change_weights_or_work():
    controller = policy()
    boundary(controller, 0)
    event = boundary(controller, 1, certified=False)
    assert event["status"] == "refused"
    assert controller.completed_cycles == 0
    assert controller.weights == (1.,) * 4
    assert boundary(controller, 1)["status"] == "held"
    assert boundary(controller, 3)["reasons"] == ["nonsequential_certified_cycle"]


def test_work_updates_are_rejected_within_a_block_and_historical_guard_stays_intact():
    controller = policy()
    boundary(controller, 0)
    event = boundary(controller, 1, torque=.2)
    assert "work_change_outside_block_boundary" in event["reasons"]
    assert controller.equivalent_mean_torque_nm == .1
    old = RhoPaceController(("a", "b"), (1., 1.), signed_crank_torque_nm=.1,
        parameters={"a": {}, "b": {}}, initial_weight_basis="fixture")
    assert old.boundary(0, (1., 1.), certified=True, signed_crank_torque_nm=.2,
                        apply_weights=apply)["reasons"] == ["resistance_changed"]


def test_unconnected_writer_and_invalid_dimensions_fail_closed():
    controller = policy()
    assert boundary(controller, 0, writer=None)["status"] == "refused"
    assert not controller.connected
    with pytest.raises(RuntimeError, match="did not confirm"):
        boundary(controller, 0, writer=lambda _: {})
    with pytest.raises(ValueError, match="four"):
        BilateralArmPaceController("right", tuple(f"m{i}" for i in range(8)), (1.,) * 8,
            equivalent_mean_torque_nm=.1, initial_weight_basis="fixture")


def test_physio_update_strategy_applies_only_a_certified_due_state_conditioned_proposal():
    controller = policy(config=BilateralArmPaceConfig(
        adaptation_strategy="physio_update", update_every_cycles=10,
    ))
    assert boundary(controller, 0)["status"] == "applied"
    assert boundary(controller, 1)["reasons"] == ["slow_update_not_due"]
    for cycle in range(2, 10):
        assert boundary(controller, cycle)["status"] == "held"
    inputs = {
        "current_capacity": np.full(4, 800.), "rest_capacity": np.full(4, 1000.),
        "alpha_a": np.array([-.1, -.2, -.3, -.4]), "tau_fat": np.full(4, 100.),
        "reference_force": np.full((4, 2), 50.),
        "available_positive_power": np.array([[.2, .2], [.3, .3], [.4, .4], [.5, .5]]),
        "phase_durations": np.full(2, .5), "required_active_work": .5,
    }
    event = controller.boundary(10, (.8, .8, .8, .8), certified=True,
                                equivalent_mean_torque_nm=.1, apply_weights=apply,
                                physio_update_inputs=inputs)
    assert event["status"] == "applied"
    assert event["physio_update"]["status"] == "proposed"
    assert math.prod(controller.weights) == pytest.approx(1.)


def test_physio_update_strategy_holds_fail_closed_when_live_envelope_is_missing():
    controller = policy(config=BilateralArmPaceConfig(
        adaptation_strategy="physio_update", update_every_cycles=10,
    ))
    boundary(controller, 0)
    for cycle in range(1, 10):
        boundary(controller, cycle)
    event = boundary(controller, 10)
    assert event["status"] == "held"
    assert event["reasons"] == ["physio_update_inputs_unavailable"]


def test_mechanical_sensitivity_strategy_is_a_distinct_bounded_ablation():
    controller = policy(config=BilateralArmPaceConfig(
        adaptation_strategy="mechanical_sensitivity", update_every_cycles=10,
    ))
    assert boundary(controller, 0)["status"] == "applied"
    for cycle in range(1, 10):
        assert boundary(controller, cycle)["status"] == "held"
    inputs = {
        "available_positive_power": np.array([[.2, .2], [.3, .3], [.4, .4], [.5, .5]]),
        "phase_durations": np.full(2, .5), "required_active_work": .5,
    }
    event = controller.boundary(10, (.8, .8, .8, .8), certified=True,
                                equivalent_mean_torque_nm=.1, apply_weights=apply,
                                physio_update_inputs=inputs)
    assert event["status"] == "applied"
    assert event["physio_update"]["policy"] == "mechanical_sensitivity_squared_v1_experimental"
    assert math.prod(controller.weights) == pytest.approx(1.)


def test_resistance_supervisor_requires_both_certifications_and_contiguous_cycles():
    controller = IndependentArmResistancePace(IndependentArmRhoPaceConfig(
        total_equivalent_mean_torque_nm=.3, update_every_cycles=2,
        initial_right_fraction=1 / 3, smoothing=1., max_fraction_step=1.))
    good = {"certified": True, "minimum_capacity_ratio": .9}
    bad = {"certified": False, "minimum_capacity_ratio": .3}
    assert controller.observe(0, good, bad)["status"] == "refused"
    assert controller.certified_cycles == 0
    assert controller.observe(1, good, good)["reason"] == "nonsequential_cycle"
    assert controller.observe(0, good, good)["status"] == "held"
    event = controller.observe(1, good, {**good, "minimum_capacity_ratio": .3})
    assert event["status"] == "updated"
    assert sum(event["next_equivalent_mean_torque_nm"].values()) == pytest.approx(.3)


def test_calibrated_initial_split_is_a_distinct_audited_event_before_feedback():
    controller = IndependentArmResistancePace(IndependentArmRhoPaceConfig(
        total_equivalent_mean_torque_nm=.3,
        initial_split_policy="capacity_fatigability_after_first_cycle",
    ))
    event = controller.apply_initial_split_decision({
        "status": "proposed", "right_fraction": .7,
        "policy": "capacity_fatigability_initial_split_v1",
    })
    assert event["event_type"] == "initial_split"
    assert event["status"] == "applied"
    assert controller.equivalent_mean_torques == pytest.approx({"right": .21, "left": .09})
    controller.observe(0, {"certified": True, "minimum_capacity_ratio": .8},
                       {"certified": True, "minimum_capacity_ratio": .8})
    with pytest.raises(RuntimeError, match="precede"):
        controller.apply_initial_split_decision({"status": "proposed", "right_fraction": .5})


def test_unknown_initial_split_policy_is_rejected():
    with pytest.raises(ValueError, match="initial_split_policy"):
        IndependentArmRhoPaceConfig(total_equivalent_mean_torque_nm=.3, initial_split_policy="unknown")


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_cadence_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        BilateralArmPaceConfig(update_every_cycles=value)
