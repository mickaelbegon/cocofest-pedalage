import json
import math
import threading

import pytest

from cocofest.simulation.independent_arms import (
    ArmRunResult,
    IndependentArmConfig,
    IndependentArmCoordinator,
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
        return ArmRunResult(rho_state=(rho or 0) + 1, metrics={"arm": self.name, "success": True})


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
