from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.simulation.rho_restart_checkpoint import (
    export_prepared_checkpoint,
    prepared_problem_arrays,
    prepared_problem_digest,
    restore_prepared_checkpoint,
    verify_prepared_checkpoint,
)
from cocofest.simulation.independent_arms_process import (
    _publish_prepared_pair_export,
    _requested_restart_checkpoint_cycles,
)


class _Program:
    def __init__(self):
        self.nlp = [SimpleNamespace(
            x_init={"A_Biceps": SimpleNamespace(init=np.array([[1.0, 0.9]]))},
            u_init={"last_pulse_width_Biceps": SimpleNamespace(init=np.array([[0.0002]]))},
            x_bounds={"A_Biceps": SimpleNamespace(min=np.array([[0.0, 0.0]]),
                                                      max=np.array([[2.0, 2.0]]))},
            u_bounds={"last_pulse_width_Biceps": SimpleNamespace(min=np.array([[5e-5]]),
                                                                     max=np.array([[6e-4]]))},
        )]
        self.parameter_init = {"fatigue_weights": SimpleNamespace(init=np.ones((4, 1)))}
        self.parameter_bounds = {"fatigue_weights": SimpleNamespace(min=np.ones((4, 1)),
                                                                        max=np.ones((4, 1)))}
        self.absolute_wheel_q_cycle_index = 20
        self.absolute_wheel_q_reference = 1.5
        self.absolute_wheel_q_cycle_shift = -2 * np.pi


def test_export_roundtrips_shifted_primal_bounds_and_fixed_parameters(tmp_path):
    program = _Program()
    model = tmp_path / "model.json"
    model.write_text('{"case":"test"}\n', encoding="utf-8")
    target = tmp_path / "cycle-20" / "right.npz"

    receipt = export_prepared_checkpoint(target, program, completed_cycles=20, model_path=model)
    verified = verify_prepared_checkpoint(target, completed_cycles=20)

    assert receipt["status"] == "exported"
    assert receipt["serialization_roundtrip_exact"]
    assert not receipt["fresh_worker_replay_verified"]
    assert receipt["stimulation_history_complete"]
    assert verified["prepared_primal_signature"] == receipt["prepared_primal_signature"]
    assert verified["prepared_problem_sha256"] == receipt["prepared_problem_sha256"]
    with np.load(target, allow_pickle=False) as archive:
        assert "states__A_Biceps" in archive.files
        assert "controls__last_pulse_width_Biceps" in archive.files
        assert "problem__parameter_init:fatigue_weights" in archive.files


def test_checkpoint_digest_changes_when_active_bounds_or_fixed_parameters_change(tmp_path):
    first, second = _Program(), _Program()
    second.nlp[0].x_bounds["A_Biceps"].max[0, 1] = 1.5
    second.parameter_init["fatigue_weights"].init[0, 0] = 1.2
    first_receipt = export_prepared_checkpoint(tmp_path / "first.npz", first, completed_cycles=4)
    second_receipt = export_prepared_checkpoint(tmp_path / "second.npz", second, completed_cycles=4)

    assert first_receipt["prepared_problem_sha256"] != second_receipt["prepared_problem_sha256"]


def test_restore_requires_and_recovers_the_entire_prepared_problem(tmp_path):
    source, rebuilt = _Program(), _Program()
    source.nlp[0].x_init["A_Biceps"].init[:] = [[1.3, 0.8]]
    source.nlp[0].u_init["last_pulse_width_Biceps"].init[:] = [[0.0004]]
    source.nlp[0].x_bounds["A_Biceps"].min[0, 1] = 0.2
    source.parameter_init["fatigue_weights"].init[:] = [[1.1], [0.9], [1.0], [1.2]]
    receipt = export_prepared_checkpoint(tmp_path / "prepared.npz", source, completed_cycles=8)

    rebuilt.nlp[0].x_init["A_Biceps"].init[:] = -1.0
    rebuilt.nlp[0].u_init["last_pulse_width_Biceps"].init[:] = 0.0
    rebuilt.nlp[0].x_bounds["A_Biceps"].min[:] = -3.0
    rebuilt.parameter_init["fatigue_weights"].init[:] = 5.0
    rebuilt.absolute_wheel_q_cycle_index = 0
    rebuilt.absolute_wheel_q_reference = 0.0
    restored = restore_prepared_checkpoint(tmp_path / "prepared.npz", rebuilt, completed_cycles=8)

    assert restored["restored_problem_sha256"] == receipt["prepared_problem_sha256"]
    np.testing.assert_array_equal(rebuilt.nlp[0].x_init["A_Biceps"].init,
                                  source.nlp[0].x_init["A_Biceps"].init)
    np.testing.assert_array_equal(rebuilt.nlp[0].x_bounds["A_Biceps"].min,
                                  source.nlp[0].x_bounds["A_Biceps"].min)
    np.testing.assert_array_equal(rebuilt.parameter_init["fatigue_weights"].init,
                                  source.parameter_init["fatigue_weights"].init)
    assert rebuilt.absolute_wheel_q_cycle_index == source.absolute_wheel_q_cycle_index
    assert rebuilt.absolute_wheel_q_reference == source.absolute_wheel_q_reference


def test_export_never_overwrites_a_checkpoint(tmp_path):
    program = _Program()
    target = tmp_path / "checkpoint.npz"
    export_prepared_checkpoint(target, program, completed_cycles=1)
    with pytest.raises(FileExistsError):
        export_prepared_checkpoint(target, program, completed_cycles=1)


def test_digest_is_deterministic_for_the_live_problem_arrays():
    program = _Program()
    snapshot = {"states": {"A_Biceps": program.nlp[0].x_init["A_Biceps"].init.copy()},
                "controls": {"last_pulse_width_Biceps": program.nlp[0].u_init[
                    "last_pulse_width_Biceps"].init.copy()}}
    assert prepared_problem_digest(snapshot, prepared_problem_arrays(program)) == (
        prepared_problem_digest(snapshot, prepared_problem_arrays(program))
    )


def test_bilateral_export_requires_two_same_boundary_arm_exports(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "configuration.json").write_text('{"cycles":100}\n', encoding="utf-8")
    directory = root / "checkpoints" / "cycle-20"
    directory.mkdir(parents=True)
    prepared = {}
    for side in ("right", "left"):
        archive = directory / f"{side}.npz"
        archive.write_bytes(side.encode("ascii"))
        prepared[side] = {"checkpoint_export": {
            "status": "exported", "primal_path": str(archive), "sha256": side * 32,
            "completed_cycles": 20, "prepared_primal_signature": side,
            "prepared_problem_sha256": "a" * 64, "serialization_roundtrip_exact": True,
            "fresh_worker_replay_verified": False, "stimulation_history_complete": True,
            "model_path": None, "model_sha256": None,
        }}
    report = _publish_prepared_pair_export(root, 20, prepared)
    assert report["status"] == "exported_pending_fresh_worker_replay"
    document = __import__("json").loads((directory / "prepared-export.json").read_text())
    assert document["exact_bilateral_restart"] is False
    assert document["arms"]["right"]["primal_path"] == "right.npz"

    prepared["left"]["checkpoint_export"] = None
    with pytest.raises(RuntimeError, match="only one arm"):
        _publish_prepared_pair_export(root, 20, prepared)


@pytest.mark.parametrize("raw", [[20, 40], (), None])
def test_explicit_checkpoint_cycle_request_is_validated(raw):
    assert _requested_restart_checkpoint_cycles({"prepared_restart_checkpoint_cycles": raw}) == frozenset(raw or ())


@pytest.mark.parametrize("raw", [[0], [20, 20], "20"])
def test_invalid_checkpoint_cycle_request_is_refused(raw):
    with pytest.raises(ValueError):
        _requested_restart_checkpoint_cycles({"prepared_restart_checkpoint_cycles": raw})
