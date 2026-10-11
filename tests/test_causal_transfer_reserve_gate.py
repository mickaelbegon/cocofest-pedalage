"""The G->D reserve gate is state based, conservative and one shot."""

from __future__ import annotations

from hashlib import sha256

import pytest

from cocofest.cycling.slow.causal_transfer import (
    BicepsReservePolicy,
    CausalTransferSupervisor,
    TransferConfig,
    biceps_reserve_observation,
)
from cocofest.cycling.slow.oracle import OracleSlowState


def state(tmp_path, side: str, cycle: int, biceps_fraction: float) -> OracleSlowState:
    artifact = tmp_path / f"{side}-{cycle}-{biceps_fraction}.json"
    artifact.write_text("certified boundary", encoding="utf-8")
    # The non-Biceps rows are deliberately present: the adapter must select by
    # name, rather than assuming a fixed order in the four-muscle model.
    values = ((1000.0, .06, .137), (biceps_fraction * 1000.0, .10, .175),
              (3000.0, .20, .30))
    return OracleSlowState(side, cycle, ("Delt_ant", "Biceps", "Triceps"), values,
                           "certified_rho_trajectory", "last_node",
                           ((str(artifact), sha256(artifact.read_bytes()).hexdigest()),), "a" * 64)


def observation(tmp_path, *, left: float, right: float, fast: bool = True, cycle: int = 140):
    return biceps_reserve_observation(
        state(tmp_path, "left", cycle, left), state(tmp_path, "right", cycle, right),
        left_a_scale=1000.0, right_a_scale=1000.0, fast_state_gate=fast,
    )


def test_calibrated_c120_and_c140_states_differ_without_using_cycle_number(tmp_path) -> None:
    policy = BicepsReservePolicy()
    c120 = observation(tmp_path, left=.699, right=.849, cycle=120)
    c140 = observation(tmp_path, left=.590, right=.813, cycle=140)
    assert policy.evaluate(c120) == (False, "left_biceps_reserve_above_activation_band", pytest.approx(.699))
    assert policy.evaluate(c140) == (True, "left_biceps_reserve_triggered", pytest.approx(.590))


def test_transition_missing_fast_gate_and_weak_recipient_abstain(tmp_path) -> None:
    policy = BicepsReservePolicy()
    assert policy.evaluate(observation(tmp_path, left=.64, right=.81))[1] == "left_biceps_reserve_transition_band"
    assert policy.evaluate(observation(tmp_path, left=.59, right=.81, fast=False))[1] == "fast_state_gate_not_validated"
    assert policy.evaluate(observation(tmp_path, left=.59, right=.77))[1] == "right_recipient_reserve_below_minimum"


def test_one_shot_action_is_ten_cycles_then_an_explicit_baseline_return(tmp_path) -> None:
    supervisor = CausalTransferSupervisor(BicepsReservePolicy(), TransferConfig())
    trigger = observation(tmp_path, left=.59, right=.81)
    decisions = [supervisor.decide(trigger) for _ in range(11)]
    assert [decision.phase for decision in decisions] == ["transfer"] * 10 + ["return"]
    assert all((decision.left_torque_nm, decision.right_torque_nm) == (.86, 1.06)
               for decision in decisions[:10])
    assert (decisions[-1].left_torque_nm, decisions[-1].right_torque_nm) == (.96, .96)
    assert supervisor.decide(trigger).phase == "return"  # one intervention per task by default


def test_adapter_rejects_mixed_boundary_or_invalid_policy_band(tmp_path) -> None:
    with pytest.raises(ValueError, match="share one certified boundary"):
        biceps_reserve_observation(state(tmp_path, "left", 120, .6), state(tmp_path, "right", 121, .8),
                                   left_a_scale=1000., right_a_scale=1000.)
    with pytest.raises(ValueError, match="abstention band"):
        BicepsReservePolicy(left_activate_at_or_below=.67, left_no_action_at_or_above=.62)
