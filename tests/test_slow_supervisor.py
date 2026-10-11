"""Conservative oracle supervisor behavior without a live RHO process."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from cocofest.cycling.slow.oracle import OracleSlowState
from cocofest.cycling.slow.remaining_cycles import ConditionalRemainingCycles
from cocofest.cycling.slow.split_rule import SplitComfort
from cocofest.cycling.slow.supervisor import (
    BoundConditionalEstimate, SplitProposal, SupervisorLimits,
    equalizing_split_rule, oracle_state_sha256, supervise_cycle,
)


def _inputs(tmp_path: Path, *, cycle: int = 10, censored: bool = False):
    states = {}
    estimates = {}
    for side in ("left", "right"):
        artifact = tmp_path / f"{side}.receipt"
        artifact.write_text(f"certified {side} cycle {cycle}", encoding="utf-8")
        state = OracleSlowState(
            side=side, completed_cycles=cycle, muscles=("Biceps", "Triceps"),
            values=((1000.0, 0.06, 0.137), (2000.0, 0.06, 0.137)),
            source_kind="certified_rho_trajectory", source_node="last_node",
            artifacts=((str(artifact), sha256(artifact.read_bytes()).hexdigest()),),
            model_sha256="a" * 64,
        )
        states[side] = state
        estimate = ConditionalRemainingCycles(
            cycles_within_bounds=20 if side == "left" else 30,
            first_exceedance_cycle=None if censored and side == "right" else (21 if side == "left" else 31),
            horizon=40, limiting_muscles=("Biceps",), initial_receipt_sha256="b" * 64,
        )
        estimates[side] = BoundConditionalEstimate(estimate, oracle_state_sha256(state))
    return states, estimates


def _rule(estimates, previous):
    assert estimates["left"].result.cycles_within_bounds == 20
    return SplitProposal(0.90, 1.02, (2.0, 0.5), (1.0, 1.0))


def test_supervisor_applies_conditional_rule_and_receipt_is_immutable(tmp_path: Path) -> None:
    states, estimates = _inputs(tmp_path)
    limits = SupervisorLimits(max_step_nm=0.1, max_asymmetry_nm=0.3, prescribed_total_nm=1.92)
    decision, receipt = supervise_cycle(
        completed_cycles=10, states=states, estimates=estimates,
        previous_torque_nm=(0.96, 0.96), rule=_rule, limits=limits,
    )
    assert decision.status == "proposed"
    assert decision.feasibility == "not_assessed"
    assert decision.target_torque_nm == (0.90, 1.02)
    assert decision.weights == ((2.0, 0.5), (1.0, 1.0))
    assert receipt.reason == "rule_applied_conditionally"
    assert len(receipt.sha256) == 64
    assert receipt.sha256 == receipt.sha256
    try:
        receipt.reason = "edited"
    except AttributeError:
        pass
    else:
        raise AssertionError("receipt must be immutable")


def test_supervisor_holds_when_not_due_or_estimate_censored(tmp_path: Path) -> None:
    states, estimates = _inputs(tmp_path, censored=True)
    decision, receipt = supervise_cycle(
        completed_cycles=10, states=states, estimates=estimates,
        previous_torque_nm=(0.96, 0.96), rule=_rule,
    )
    assert (decision.status, receipt.reason) == ("hold", "censored_estimate_right")
    assert decision.target_torque_nm == (0.96, 0.96)
    assert decision.weights == ((1.0, 1.0), (1.0, 1.0))
    decision, receipt = supervise_cycle(
        completed_cycles=11, states=states, estimates=estimates,
        previous_torque_nm=(0.96, 0.96), rule=_rule,
    )
    assert (decision.status, receipt.reason) == ("hold", "update_not_due")


def test_supervisor_holds_on_stale_or_tampered_oracle(tmp_path: Path) -> None:
    states, estimates = _inputs(tmp_path)
    estimates["left"] = BoundConditionalEstimate(estimates["left"].result, "0" * 64)
    decision, receipt = supervise_cycle(
        completed_cycles=10, states=states, estimates=estimates,
        previous_torque_nm=(0.96, 0.96), rule=_rule,
    )
    assert receipt.reason == "stale_estimate_left"
    estimates["left"] = BoundConditionalEstimate(estimates["left"].result, oracle_state_sha256(states["left"]))
    (tmp_path / "left.receipt").write_text("tampered", encoding="utf-8")
    decision, receipt = supervise_cycle(
        completed_cycles=10, states=states, estimates=estimates,
        previous_torque_nm=(0.96, 0.96), rule=_rule,
    )
    assert receipt.reason == "uncertified_state_left"


def test_supervisor_holds_on_invalid_rule_output(tmp_path: Path) -> None:
    states, estimates = _inputs(tmp_path)
    for rule, reason in (
        (lambda _estimates, _previous: SplitProposal(1.30, 0.62), "rule_violates_comfort_bounds"),
        (lambda _estimates, _previous: SplitProposal(0.96, 0.96, (2.0, 2.0)), "invalid_rule_weights"),
        (lambda _estimates, _previous: None, "invalid_rule_proposal"),
    ):
        decision, receipt = supervise_cycle(
            completed_cycles=10, states=states, estimates=estimates,
            previous_torque_nm=(0.96, 0.96), rule=rule,
            limits=SupervisorLimits(max_step_nm=0.1, max_asymmetry_nm=0.3, prescribed_total_nm=1.92),
        )
        assert decision.status == "hold"
        assert receipt.reason == reason
        assert decision.weights == ((1.0, 1.0), (1.0, 1.0))


def test_existing_equalization_rule_receipt_is_embedded(tmp_path: Path) -> None:
    states, estimates = _inputs(tmp_path)
    rule = equalizing_split_rule(SplitComfort((0.6, 0.6), (1.2, 1.2)))
    decision, receipt = supervise_cycle(
        completed_cycles=10, states=states, estimates=estimates,
        previous_torque_nm=(0.96, 0.96), rule=rule,
        limits=SupervisorLimits(max_step_nm=0.05, max_asymmetry_nm=0.3, prescribed_total_nm=1.92),
    )
    assert decision.status == "proposed"
    assert decision.target_torque_nm == (0.95, 0.97)
    assert receipt.rule_reason == "adjusted"
    assert receipt.rule_receipt_sha256 is not None and len(receipt.rule_receipt_sha256) == 64


def test_malformed_conditional_estimate_holds_before_rule(tmp_path: Path) -> None:
    states, estimates = _inputs(tmp_path)
    malformed = ConditionalRemainingCycles(20, 35, 40, ("Biceps",), "b" * 64)
    estimates["left"] = BoundConditionalEstimate(malformed, oracle_state_sha256(states["left"]))
    called = False

    def rule(_estimates, _previous):
        nonlocal called
        called = True
        return SplitProposal(0.90, 1.02)

    decision, receipt = supervise_cycle(
        completed_cycles=10, states=states, estimates=estimates,
        previous_torque_nm=(0.96, 0.96), rule=rule,
    )
    assert decision.status == "hold"
    assert receipt.reason == "invalid_estimate_left"
    assert not called
