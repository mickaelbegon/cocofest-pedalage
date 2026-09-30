import os
import time

import pytest

from cocofest.optimization.independent_arm_rho_pace import (
    IndependentArmResistancePace, IndependentArmRhoPaceConfig,
)
from cocofest.simulation.independent_arms_process import (
    IndependentArmProcessCoordinator,
    _can_preserve_static_unit_objective,
    _configured_payload_model,
    _resolve_hsl_library,
)
from cocofest.optimization.independent_arm_rho_pace import BilateralArmPaceConfig, BilateralArmPaceController


def fake_worker(connection, side, payload, output_root):
    """Use real spawned processes and IPC, without importing native solvers."""
    history = []
    prepared = []
    completed = 0
    try:
        for completed in range(payload["cycles"] + 1):
            good = not (side == "left" and completed == payload.get("fail_at"))
            metrics = {"certified": good, "minimum_capacity_ratio": .8 if side == "right" else .4}
            if completed == 1:
                measurement = payload.get("initial_split_measurements", {}).get(side)
                if measurement is not None:
                    metrics["capacity_fatigability_initial_split"] = measurement
            connection.send({"kind": "boundary", "completed_cycles": completed, "metrics": metrics})
            command = connection.recv()
            if command["kind"] == "stop":
                break
            assert command["kind"] == "prepare"
            prepared.append(completed)
            connection.send({"kind": "prepared", "completed_cycles": completed,
                             "weight_event": {"status": "applied" if completed % 10 == 0 else "held"},
                             "prepare_time_s": 0.})
            command = connection.recv()
            if command["kind"] == "stop":
                break
            assert command == {"kind": "solve", "completed_cycles": completed}
            history.append({"cycle": completed + 1, "start": time.monotonic(), "pid": os.getpid()})
            time.sleep(.02)
        connection.send({"kind": "finished", "completed_cycles": completed,
                         "result": {"pid": os.getpid(), "history": history, "prepared": prepared}})
    finally:
        connection.close()


def run(tmp_path, **payload):
    cycles = payload.pop("cycles", 11)
    coordinator = IndependentArmProcessCoordinator(payload, worker_target=fake_worker)
    pace_overrides = payload.pop("pace_overrides", {})
    pace = IndependentArmResistancePace(IndependentArmRhoPaceConfig(
        total_equivalent_mean_torque_nm=.3, initial_right_fraction=1 / 3, **pace_overrides))
    return coordinator.run_with_resistance_pace_to_directory(tmp_path, pace, cycles=cycles)


def test_real_process_barrier_updates_work_only_after_ten_completed_windows(tmp_path):
    summary = run(tmp_path)
    assert summary["success"]
    assert summary["completed_rho_cycles"] == 11
    assert summary["arms"]["right"]["pid"] != summary["arms"]["left"]["pid"]
    events = summary["resistance_pace"]["events"]
    assert [event["cycle_index"] for event in events if event["status"] == "updated"] == [9]
    assert len(summary["preparations"]) == 11
    for cycle in range(11):
        right = summary["arms"]["right"]["history"][cycle]
        left = summary["arms"]["left"]["history"][cycle]
        assert abs(right["start"] - left["start"]) < .5
        assert summary["pair_cycle_wall_timing_s"]["min"] >= .019


def test_one_arm_failure_stops_both_before_next_work_or_weight_update(tmp_path):
    summary = run(tmp_path, fail_at=10)
    assert not summary["success"]
    assert summary["completed_rho_cycles"] == 10
    assert not any(event["status"] == "updated" for event in summary["resistance_pace"]["events"])
    for side in ("right", "left"):
        assert len(summary["arms"][side]["history"]) == 10
        assert summary["arms"][side]["prepared"] == list(range(10))


def test_first_certified_pair_can_install_a_task_normalized_initial_split(tmp_path):
    measured = {"status": "measured", "endurance_work_score_j": 4.0}
    summary = run(
        tmp_path, cycles=2,
        pace_overrides={"initial_split_policy": "capacity_fatigability_after_first_cycle", "capacity_feedback": False},
        initial_split_measurements={"right": measured, "left": {**measured, "endurance_work_score_j": 1.0}},
    )
    assert summary["initial_split_events"][0]["status"] == "applied"
    assert summary["resistance_pace"]["next_equivalent_mean_torque_nm"] == pytest.approx({"right": .24, "left": .06})


def test_mismatched_cadences_rejected_before_workers_are_started(tmp_path):
    with pytest.raises(ValueError, match="same block cadence"):
        run(tmp_path, muscle_pace={"update_every_cycles": 5})


def test_hsl_library_is_discovered_from_the_selected_runtime_prefix(tmp_path, monkeypatch):
    monkeypatch.delenv("IPOPT_HSL_LIBRARY", raising=False)
    library = tmp_path / "opt" / "libhsl" / "v-test" / "lib" / "libhsl.so"
    library.parent.mkdir(parents=True)
    library.touch()
    assert _resolve_hsl_library({"runtime_prefix": str(tmp_path)}) == str(library.resolve())


def test_side_specific_model_config_overrides_the_common_configuration(tmp_path):
    base = {
        "schema_version": 1, "case_id": "test", "provenance": "test only",
        "muscles": {name: {"Fmax": 1., "a_scale": 1., "alpha_a": -.1, "tau_fat": 1.}
                    for name in ("Delt_ant", "Delt_post", "Biceps", "Triceps")},
    }
    common = tmp_path / "common.json"
    left = tmp_path / "left.json"
    common.write_text(__import__("json").dumps(base), encoding="utf-8")
    altered = {**base, "case_id": "left", "muscles": {**base["muscles"],
               "Biceps": {**base["muscles"]["Biceps"], "Fmax": 2.}}}
    left.write_text(__import__("json").dumps(altered), encoding="utf-8")
    payload = {"model_config": str(common), "left_model_config": str(left)}
    assert _configured_payload_model(payload, "right")[1]["case_id"] == "test"
    assert _configured_payload_model(payload, "left")[1]["case_id"] == "left"


def test_compilation_guard_requires_a_static_unit_weight_objective():
    static = BilateralArmPaceController(
        "right", ("a", "b", "c", "d"), (1., 1., 1., 1.), equivalent_mean_torque_nm=.1,
        initial_weight_basis="unit test", config=BilateralArmPaceConfig(adaptation_enabled=False),
    )
    adaptive = BilateralArmPaceController(
        "right", ("a", "b", "c", "d"), (1., 1., 1., 1.), equivalent_mean_torque_nm=.1,
        initial_weight_basis="unit test", config=BilateralArmPaceConfig(adaptation_enabled=True),
    )
    asymmetric = BilateralArmPaceController(
        "right", ("a", "b", "c", "d"), (1., 2., 1., 1.), equivalent_mean_torque_nm=.1,
        initial_weight_basis="unit test", config=BilateralArmPaceConfig(adaptation_enabled=False),
    )
    assert _can_preserve_static_unit_objective(static)
    assert not _can_preserve_static_unit_objective(adaptive)
    assert not _can_preserve_static_unit_objective(asymmetric)


@pytest.mark.parametrize("payload", [{"parallel": False}, {"left_runner_config": "unread.json"}, {"solver": "acados"}])
def test_unsupported_runtime_requests_are_not_silently_ignored(payload):
    with pytest.raises(ValueError):
        IndependentArmProcessCoordinator(payload)
