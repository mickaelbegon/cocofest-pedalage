import pytest

from cocofest.cycling.slow.causal_transfer import (
    CausalTransferSupervisor,
    ThresholdPolicy,
    TransferConfig,
    TransferObservation,
)
from scripts.slow_layer import run_causal_transfer_supervisor as runner
from scripts.slow_layer.run_causal_transfer_supervisor import make_jobs


def observation(cycle=100, value=.25):
    return TransferObservation(
        cycle,
        {"left": {"A_Triceps": value}, "right": {"A_Triceps": .8}},
        True,
        "a" * 64,
    )


def test_threshold_starts_exactly_the_declared_number_of_transfer_windows_then_returns():
    controller = CausalTransferSupervisor(ThresholdPolicy("A_Triceps", .3), TransferConfig(action_cycles=3))
    decisions = [controller.decide(observation(100 + index)) for index in range(5)]
    assert [decision.phase for decision in decisions] == ["transfer", "transfer", "transfer", "return", "return"]
    assert [(d.left_torque_nm, d.right_torque_nm) for d in decisions[:3]] == [(pytest.approx(.86), pytest.approx(1.06))] * 3
    assert (decisions[3].left_torque_nm, decisions[3].right_torque_nm) == pytest.approx((.96, .96))
    assert decisions[0].reason == "threshold_met"
    assert decisions[3].reason == "explicit_return_to_baseline"


def test_threshold_hold_is_baseline_and_has_a_receipt_bound_to_the_observation():
    controller = CausalTransferSupervisor(ThresholdPolicy("A_Triceps", .1))
    decision = controller.decide(observation(value=.25))
    assert decision.phase == "hold"
    assert (decision.left_torque_nm, decision.right_torque_nm) == pytest.approx((.96, .96))
    assert decision.reason == "threshold_not_met"
    assert len(decision.receipt_sha256) == 64


def test_missing_metric_abstains_instead_of_guessing_a_fatigue_signal():
    controller = CausalTransferSupervisor(ThresholdPolicy("A_Biceps", .1))
    decision = controller.decide(observation())
    assert decision.phase == "hold"
    assert decision.reason == "metric_unavailable"


def test_invalid_uncertified_observation_is_refused_at_the_interface():
    with pytest.raises(ValueError, match="certified"):
        TransferObservation(2, {"left": {"A_Triceps": .2}, "right": {"A_Triceps": .8}}, False, "a" * 64)


def test_policy_jobs_pair_independent_controls_and_preserve_total_torque():
    anchor = {"cycle": 120, "receipt": "/receipt", "receipt_sha256": "a" * 64, "arms": {}}
    decision = {"decision": {"phase": "transfer", "left_torque_nm": .86, "right_torque_nm": 1.06}}
    jobs = make_jobs([anchor], [decision], replicates=2)
    assert len(jobs) == 4
    assert sum(job["is_control"] for job in jobs) == 2
    causal = [job for job in jobs if not job["is_control"]]
    assert [job["delta_left_nm"] for job in causal] == pytest.approx([-.1, -.1])
    assert all(job["intervention_cycles"] == 10 for job in causal)


def test_held_policy_branch_is_kept_as_a_noncontrol_branch_after_the_gate():
    anchor = {"cycle": 140, "receipt": "/receipt", "receipt_sha256": "a" * 64, "arms": {}}
    decision = {"decision": {"phase": "hold", "left_torque_nm": .96, "right_torque_nm": .96}}
    jobs = make_jobs([anchor], [decision], replicates=2)
    assert sum(job["is_control"] for job in jobs) == 2
    assert {job["delta_left_nm"] for job in jobs if not job["is_control"]} == {0.}


def test_job_builder_refuses_single_replicate_or_an_unknown_initial_phase():
    anchor = {"cycle": 120}
    with pytest.raises(ValueError, match="two independent"):
        make_jobs([anchor], [{"decision": {"phase": "hold", "left_torque_nm": .96, "right_torque_nm": .96}}], replicates=1)
    with pytest.raises(ValueError, match="transfer or hold"):
        make_jobs([anchor], [{"decision": {"phase": "return", "left_torque_nm": .96, "right_torque_nm": .96}}], replicates=2)


def test_campaign_replay_does_not_accept_or_require_an_unsealed_runtime_threshold(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(runner.runner, "campaign", lambda *args: seen.setdefault("args", args))
    monkeypatch.setattr("sys.argv", ["causal-transfer", "campaign", "--manifest", str(tmp_path / "manifest.json"),
                                      "--cpus", "0", "1"])
    runner.main()
    assert seen["args"] == (tmp_path / "manifest.json", [0, 1], 7200, False)
