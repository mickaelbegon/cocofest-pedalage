import json

import pytest

from cocofest.cycling.slow.causal_transfer import BicepsReservePolicy, CausalTransferSupervisor
from cocofest.optimization.independent_arm_rho_pace import BilateralArmPaceConfig, BilateralArmPaceController
from scripts.slow_layer import run_continuous_causal_transfer as runner


def report(left, right):
    return {"left": {"certified": True, "capacity_ratios": {"Biceps": left}, "solver_time_s": .2},
            "right": {"certified": True, "capacity_ratios": {"Biceps": right}, "solver_time_s": .2}}


def test_live_observation_is_state_based_and_binds_complete_boundary_reports():
    first = runner.live_biceps_observation(report(.7, .9), 120, fast_state_gate=True)
    changed = runner.live_biceps_observation(report(.7, .9), 121, fast_state_gate=True)
    assert first.metrics["left"]["biceps_A_fraction"] == .7
    assert first.provenance_sha256 != changed.provenance_sha256
    controller = CausalTransferSupervisor(BicepsReservePolicy())
    assert controller.decide(first).phase == "hold"
    action = controller.decide(runner.live_biceps_observation(report(.61, .9), 121, fast_state_gate=True))
    assert action.phase == "transfer"
    assert (action.left_torque_nm, action.right_torque_nm) == pytest.approx((.86, 1.06))


def test_live_observation_refuses_an_uncertified_or_incomplete_boundary():
    bad = report(.61, .9)
    bad["left"]["certified"] = False
    with pytest.raises(ValueError, match="certified"):
        runner.live_biceps_observation(bad, 1, fast_state_gate=True)
    with pytest.raises(ValueError, match="capacity ratios"):
        runner.live_biceps_observation({"left": {"certified": True}, "right": {"certified": True}}, 1,
                                       fast_state_gate=True)


def test_terminal_aggregation_never_labels_a_bare_solver_stop_as_physiological_failure():
    outcome = runner.aggregate_terminal(last_certified_cycle=12, terminal_reports={
        "left": {"native_solver_status": "Maximum_Iterations_Exceeded"},
        "right": {"native_solver_status": "Solve_Succeeded"},
    }, censored=False)
    assert outcome["status"] == "numerical_stop"
    assert outcome["exact_label_allowed"] is False


def test_prepare_seals_cold_reference_and_safe_default_abstention(tmp_path):
    manifest_path = runner.prepare(tmp_path / "continuous", maximum_cycles=20, right_cpu=0, left_cpu=1)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["payload"]["cycles"] == 20
    assert manifest["fast_state_status"] == "not_validated_safe_abstention"
    assert manifest["activation_allowed"] is False
    assert runner.authenticated(manifest_path)["kind"] == "continuous_causal_left_to_right_transfer_rho_benchmark"


def test_prepare_records_explicit_override_as_experimental_only(tmp_path):
    manifest_path = runner.prepare(tmp_path / "continuous", maximum_cycles=20, right_cpu=0, left_cpu=1,
                                   experimental_unvalidated_fast_state_override=True)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["fast_state_status"] == "unvalidated_experimental_override"
    assert "EXPERIMENTAL ONLY" in manifest["warning"]


def test_trigger_crossing_uses_next_complete_rho_windows_for_exactly_ten_cycles_then_returns():
    controller = CausalTransferSupervisor(BicepsReservePolicy())
    # 135 is intentionally not a multiple of the historical 20-cycle PACE
    # block.  The continuous adapter admits the numerical work update at this
    # completed one-cycle RHO boundary, not mid-window.
    first = controller.decide(runner.live_biceps_observation(report(.61, .9), 135, fast_state_gate=True))
    following = [controller.decide(runner.live_biceps_observation(report(.60, .9), cycle, fast_state_gate=True))
                 for cycle in range(136, 146)]
    assert first.phase == "transfer"
    assert [row.phase for row in following[:9]] == ["transfer"] * 9
    assert following[9].phase == "return"
    assert (following[9].left_torque_nm, following[9].right_torque_nm) == pytest.approx((.96, 1.06 - .10))


def test_continuous_payload_uses_one_rho_boundary_adapter_without_enabling_feedback_or_weight_adaptation(tmp_path):
    manifest_path = runner.prepare(tmp_path / "continuous", maximum_cycles=20, right_cpu=0, left_cpu=1)
    manifest = json.loads(manifest_path.read_text())
    payload = manifest["payload"]
    assert payload["muscle_pace"]["update_every_cycles"] == 1
    assert payload["resistance_pace"]["update_every_cycles"] == 1
    assert payload["muscle_pace"]["adaptation_enabled"] is False
    assert payload["resistance_pace"]["capacity_feedback"] is False
    assert payload["causal_work_target_adapter"]["compiled_nlp_reused"] is True


def test_compiled_work_target_adapter_admits_non_block_trigger_and_exact_return_without_weight_update():
    controller = BilateralArmPaceController(
        "left", ("Delt_ant", "Delt_post", "Biceps", "Triceps"), (1., 1., 1., 1.),
        equivalent_mean_torque_nm=.96, initial_weight_basis="continuous causal test",
        config=BilateralArmPaceConfig(update_every_cycles=1, adaptation_enabled=False),
    )
    # In the persistent worker `connected` is set after its graph/parameter
    # binding is built.  This test advances genuine RHO boundaries only; it
    # never needs a graph/objective writer to accept a target change.
    controller.connected = True
    for completed in range(135):
        event = controller.boundary(completed, (1., 1., 1., 1.), certified=True,
                                    equivalent_mean_torque_nm=.96, apply_weights=None)
        assert event["status"] == "held"
    action = []
    for completed in range(135, 145):
        action.append(controller.boundary(completed, (1., 1., 1., 1.), certified=True,
                                          equivalent_mean_torque_nm=.86, apply_weights=None))
    returned = controller.boundary(145, (1., 1., 1., 1.), certified=True,
                                   equivalent_mean_torque_nm=.96, apply_weights=None)
    assert all(event["status"] == "held" for event in action)
    assert "work_change_outside_block_boundary" not in action[0]["reasons"]
    assert returned["status"] == "held"
    assert controller.equivalent_mean_torque_nm == pytest.approx(.96)


def test_checkpoint_export_at_cycle_120_requires_and_uses_runtime_configuration(tmp_path, monkeypatch):
    manifest_path = runner.prepare(tmp_path / "continuous", maximum_cycles=121, right_cpu=0, left_cpu=1)
    manifest = runner.authenticated(manifest_path)
    root = manifest_path.parent
    runner._write_runtime_configuration(root, manifest_path, manifest)
    seen = {}
    def publish(actual_root, cycle, prepared):
        assert (actual_root / "configuration.json").is_file()
        seen.update(cycle=cycle, prepared=prepared)
        return {"status": "exported"}
    monkeypatch.setattr(runner.process, "_publish_prepared_pair_export", publish)
    assert runner._publish_prepared_checkpoint(root, 120, {"left": {}, "right": {}}) == {"status": "exported"}
    assert seen["cycle"] == 120
