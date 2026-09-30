import json
import math
import threading

import pytest

from cocofest.simulation.independent_arms import (
    ArmRunResult,
    IndependentArmConfig,
    IndependentArmCoordinator,
)
from cocofest.optimization.independent_arm_rho_pace import (
    IndependentArmResistancePace,
    IndependentArmRhoPaceConfig,
)


class FakeCompiledArm:
    def __init__(self, name):
        self.name = name
        self.handle_identity = object()
        self.targets = []
        self.states = []
        self.thread_ids = []

    def set_isokinetic_work_target(self, work, torque):
        self.targets.append((work, torque))

    def solve_rho(self, rho):
        self.states.append(rho)
        self.thread_ids.append(threading.get_ident())
        return ArmRunResult(rho_state=(rho or 0) + 1, metrics={"arm": self.name, "success": True, "certified": True})


def test_independent_arms_keep_separate_rho_and_update_work_targets_without_rebuilding(tmp_path):
    right, left = FakeCompiledArm("right"), FakeCompiledArm("left")
    runner = IndependentArmCoordinator(
        right, left, right_rho_state=3, left_rho_state=11,
        right_equivalent_mean_torque_nm=.2, left_equivalent_mean_torque_nm=.35,
    )
    original_handles = (right.handle_identity, left.handle_identity)
    summary = runner.run_to_directory(tmp_path)

    assert summary["architecture"] == "two-independent-unilateral-isokinetic-ocps"
    assert summary["arms"]["right"]["target_work_j_per_cycle"] == pytest.approx(.2 * 2 * math.pi)
    assert summary["arms"]["left"]["target_work_j_per_cycle"] == pytest.approx(.35 * 2 * math.pi)
    assert right.states == [3]
    assert left.states == [11]
    assert json.loads((tmp_path / "summary.json").read_text())["arms"]["right"]["rho_state"] == 4

    runner.set_equivalent_mean_torques(right_nm=.4, left_nm=.1)
    runner.run()
    assert (right.handle_identity, left.handle_identity) == original_handles
    assert right.targets[-1] == pytest.approx((.4 * 2 * math.pi, .4))
    assert left.targets[-1] == pytest.approx((.1 * 2 * math.pi, .1))
    assert right.states[-1] == 4
    assert left.states[-1] == 12


def test_only_isokinetic_and_non_negative_work_targets_are_accepted():
    with pytest.raises(ValueError, match="only"):
        IndependentArmConfig(formulation="dynamic")
    with pytest.raises(ValueError, match="non-negative"):
        IndependentArmCoordinator(FakeCompiledArm("r"), FakeCompiledArm("l"), right_equivalent_mean_torque_nm=-.1)


def test_distinct_solver_instances_are_required():
    shared = FakeCompiledArm("shared")
    with pytest.raises(ValueError, match="distinct"):
        IndependentArmCoordinator(shared, shared)


class CapacityArm(FakeCompiledArm):
    def __init__(self, name, capacity):
        super().__init__(name)
        self.capacity = capacity

    def solve_rho(self, rho):
        result = super().solve_rho(rho)
        return ArmRunResult(rho_state=result.rho_state,
                            metrics={**result.metrics, "minimum_capacity_ratio": self.capacity})


def test_resistance_pace_keeps_total_work_and_waits_for_both_arms_each_cycle():
    right, left = CapacityArm("right", .9), CapacityArm("left", .3)
    runner = IndependentArmCoordinator(right, left, config=IndependentArmConfig(parallel=False))
    pace = IndependentArmResistancePace(IndependentArmRhoPaceConfig(
        total_equivalent_mean_torque_nm=.6, initial_right_fraction=.5,
        update_every_cycles=1, smoothing=1., max_fraction_step=1., capacity_gain=1.,
    ))
    summary = runner.run_with_resistance_pace(pace, cycles=2)

    events = summary["resistance_pace"]["events"]
    assert summary["cycle_synchronous"] is True
    assert len(events) == 2
    assert events[0]["allocation_used_equivalent_mean_torque_nm"] == pytest.approx({"right": .3, "left": .3})
    # The high-reserve right arm receives 0.9/(0.9+0.3) = 75% next cycle.
    assert events[0]["next_equivalent_mean_torque_nm"] == pytest.approx({"right": .45, "left": .15})
    assert right.targets[-1] == pytest.approx((.45 * 2 * math.pi, .45))
    assert left.targets[-1] == pytest.approx((.15 * 2 * math.pi, .15))
    for event in events:
        allocation = event["allocation_used_equivalent_mean_torque_nm"]
        assert allocation["right"] + allocation["left"] == pytest.approx(.6)


def test_resistance_pace_holds_last_safe_allocation_when_reserve_is_missing():
    pace = IndependentArmResistancePace(IndependentArmRhoPaceConfig(
        total_equivalent_mean_torque_nm=.5, initial_right_fraction=.4,
        update_every_cycles=1,
    ))
    event = pace.observe(0, {"certified": True}, {"certified": True})
    assert event["status"] == "held"
    assert event["reason"] == "capacity_measurement_unavailable"
    assert pace.equivalent_mean_torques == pytest.approx({"right": .2, "left": .3})


def test_resistance_pace_default_hard_rate_limit_updates_after_ten_certified_cycles():
    pace = IndependentArmResistancePace(IndependentArmRhoPaceConfig(
        total_equivalent_mean_torque_nm=.6, initial_right_fraction=.5,
        smoothing=1., max_fraction_step=1., capacity_gain=1.,
    ))
    for index in range(9):
        event = pace.observe(index, {"minimum_capacity_ratio": .9, "certified": True},
                             {"minimum_capacity_ratio": .3, "certified": True})
        assert event["status"] == "held"
        assert event["reason"] == "update_not_due"
        assert pace.equivalent_mean_torques == pytest.approx({"right": .3, "left": .3})
    event = pace.observe(9, {"minimum_capacity_ratio": .9, "certified": True},
                         {"minimum_capacity_ratio": .3, "certified": True})
    assert event["status"] == "updated"
    assert pace.equivalent_mean_torques == pytest.approx({"right": .45, "left": .15})
