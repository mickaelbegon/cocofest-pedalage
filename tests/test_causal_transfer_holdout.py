"""Static contract checks for the no-recalibration causal-transfer holdouts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cocofest.cycling.slow.causal_transfer import TransferConfig
from cocofest.cycling.slow.causal_transfer_holdout import baseline_payload, standard_holdouts, validate_baseline_runtime
from scripts.slow_layer.prepare_causal_transfer_holdout import _write_baseline, _authenticated_baseline, _select
from scripts.slow_layer.run_causal_torque_pilot import job_action_schedule
from scripts.slow_layer.run_causal_transfer_supervisor import make_jobs


def test_preregistered_holdouts_change_context_but_not_policy_constants():
    conditions = standard_holdouts(Path.cwd())
    assert set(conditions) == {"higher-load-208", "higher-load-200", "reversed-ding-sides-192"}
    higher, reversed_sides = conditions["higher-load-208"], conditions["reversed-ding-sides-192"]
    assert higher.total_torque_nm == pytest.approx(2.08)
    assert reversed_sides.total_torque_nm == pytest.approx(1.92)
    for condition in conditions.values():
        assert condition.left_activate_at_or_below == pytest.approx(.62)
        assert condition.right_recipient_at_or_above == pytest.approx(.78)
        assert condition.left_relief_nm == pytest.approx(.10)
        assert condition.action_cycles == 10
        payload = baseline_payload(condition)
        assert payload["stimulations_per_cycle"] == 30
        assert payload["resistance_pace"]["capacity_feedback"] is False
        assert payload["muscle_pace"]["adaptation_enabled"] is False
        assert payload["left_equivalent_mean_torque_nm"] == pytest.approx(condition.total_torque_nm / 2)
        assert payload["right_equivalent_mean_torque_nm"] == pytest.approx(condition.total_torque_nm / 2)
        runtime = validate_baseline_runtime(payload)
        assert Path(runtime["runtime_prefix"]).is_dir()
        assert Path(runtime["ipopt_hsl_library"]).is_file()
        assert Path(runtime["ipopt_hsl_library"]).is_absolute()
    assert higher.left_model == reversed_sides.right_model
    assert higher.right_model == reversed_sides.left_model


def test_higher_load_200_is_a_new_dense_preregistration_with_unchanged_action():
    condition = standard_holdouts(Path.cwd())["higher-load-200"]
    assert condition.total_torque_nm == pytest.approx(2.00)
    assert condition.baseline_per_arm_nm == pytest.approx(1.00)
    assert condition.canonical_dict()["action_left_nm"] == pytest.approx(.90)
    assert condition.canonical_dict()["action_right_nm"] == pytest.approx(1.10)
    assert condition.checkpoint_cycles == (40, 60, 80, 90, 100, 110, 120, 130, 140, 150)
    payload = baseline_payload(condition)
    assert payload["prepared_restart_checkpoint_cycles"] == list(condition.checkpoint_cycles)
    assert payload["ipopt_hsl_library_sha256"] == validate_baseline_runtime(payload)["ipopt_hsl_library_sha256"]


def test_holdout_baseline_preflight_rejects_missing_or_changed_hsl_library():
    payload = baseline_payload(standard_holdouts(Path.cwd())["higher-load-208"])
    payload["ipopt_hsl_library"] = "/definitely/missing/libhsl.so"
    with pytest.raises(ValueError, match="HSL library"):
        validate_baseline_runtime(payload)


def test_baseline_plan_is_sealed_and_detects_input_tampering(tmp_path):
    condition = standard_holdouts(Path.cwd())["higher-load-208"]
    path = _write_baseline(tmp_path / "source", condition)
    plan = json.loads((tmp_path / "source" / "baseline-plan.json").read_text())
    assert path.name == "baseline-input.json"
    assert plan["condition"]["action_left_nm"] == pytest.approx(.94)
    assert "fresh-worker" in plan["checkpoint_admission"]["required"]
    assert plan["checkpoint_admission"]["rejected"].startswith("prepared-export")
    assert _authenticated_baseline(tmp_path / "source", condition.identifier)[0] == condition
    path.write_text("{}")
    with pytest.raises(ValueError, match="input changed"):
        _authenticated_baseline(tmp_path / "source", condition.identifier)


def test_holdout_schedule_returns_to_its_own_equal_split_not_calibration_split():
    config = TransferConfig(baseline_left_nm=1.04, baseline_right_nm=1.04, left_relief_nm=.10, action_cycles=10)
    anchor = {"cycle": 80}
    decision = {"decision": {"phase": "transfer", "left_torque_nm": .94, "right_torque_nm": 1.14}}
    jobs = make_jobs([anchor], [decision], replicates=2, transfer_config=config)
    causal = next(job for job in jobs if not job["is_control"])
    assert job_action_schedule(causal, "intervention") == pytest.approx({"left": .94, "right": 1.14})
    assert job_action_schedule(causal, "continuation") == pytest.approx({"left": 1.04, "right": 1.04})


def test_selection_is_policy_state_based_and_keeps_hold_only_falsification():
    hold60 = {"cycle": 60, "decision": {"phase": "hold"}}
    hold80 = {"cycle": 80, "decision": {"phase": "hold"}}
    trigger100 = {"cycle": 100, "decision": {"phase": "transfer"}}
    assert _select([hold60, hold80, trigger100]) == [hold80, trigger100]
    assert _select([hold60, hold80]) == [hold80]
