"""Oracle measurements must come from certified physical boundaries."""

from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.cycling.slow.oracle import AUDIT_SCOPE, load_certified_checkpoint, load_certified_cycle
from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path, document):
    path.write_text(json.dumps(document))


@pytest.fixture
def checkpoint(tmp_path):
    keys = {"A_Biceps": 3000., "Tau1_Biceps": .07, "Km_Biceps": .14}
    program = SimpleNamespace(nlp=[SimpleNamespace(
        x_init={k: SimpleNamespace(init=np.array([[v+1., v+2.]])) for k, v in keys.items()},
        u_init={"last_pulse_width_Biceps": SimpleNamespace(init=np.array([[.0002]]))},
        x_bounds={k: SimpleNamespace(min=np.array([[v, 0., 0.]]), max=np.array([[v, 1e4, 1e4]]))
                  for k, v in keys.items()},
        u_bounds={"last_pulse_width_Biceps": SimpleNamespace(min=np.array([[0.]]), max=np.array([[.0006]]))},
    )], parameter_init={}, parameter_bounds={})
    model = tmp_path / "model.json"
    save(model, {"muscles": {"Biceps": {}}})
    configuration = tmp_path / "configuration.json"
    save(configuration, {"left_model_config": str(model)})
    archive = tmp_path / "prepared.npz"
    info = export_prepared_checkpoint(archive, program, completed_cycles=20, model_path=model)
    arm = {**info, "certified": True, "replay_roundtrip_exact": True,
           "restored_problem_sha256": info["prepared_problem_sha256"]}
    receipt = tmp_path / "receipt.json"
    save(receipt, {"schema_version": 1, "checkpoint_kind": "certified_bilateral_shifted_primal",
                   "exact_bilateral_restart": True, "completed_cycles": 20,
                   "configuration_path": str(configuration), "configuration_sha256": digest(configuration),
                   "arms": {"left": arm}})
    return receipt


@pytest.fixture
def cycle_receipt(checkpoint):
    audit = {"scope": AUDIT_SCOPE, "constraint_normalized_violation": 1e-9,
             "decision_normalized_violation": 2e-9, "terminal_work_normalized_violation": 0.}
    witness = checkpoint.parent / "witness.npz"
    terminal = {"A_Biceps": 2900., "Tau1_Biceps": .08, "Km_Biceps": .15}
    np.savez_compressed(witness, **{f"states__{key}": np.array([[value+1., value]])
                                    for key, value in terminal.items()},
                        metadata__json=np.asarray(json.dumps({"independent_audit": audit})))
    receipt = checkpoint.parent / "cycle.json"
    save(receipt, {"schema_version": 1, "kind": "unilateral_continuation_phase",
                   "return_policy": {"side": "left"}, "muscle_order": ["Biceps"],
                   "anchor_completed_cycles": 20, "source_receipt": str(checkpoint),
                   "source_receipt_sha256": digest(checkpoint),
                   "source_archive_sha256": json.loads(checkpoint.read_text())["arms"]["left"]["sha256"],
                   "audit_tolerance": 1e-5,
                   "cycles": [{"cycle": 21, "certified": True, "maximum_normalized_violation": 2e-9,
                               "independent_audit": audit, "witness": str(witness),
                               "witness_sha256": digest(witness), "terminal_slow_states": terminal}]})
    return receipt


def test_checkpoint_reads_frozen_boundary_not_shifted_guess(checkpoint):
    state = load_certified_checkpoint(checkpoint, side="left", muscles=["Biceps"])
    np.testing.assert_array_equal(state.as_array(), [[3000., .07, .14]])
    assert state.completed_cycles == 20
    assert state.source_node == "frozen_initial_equality_of_next_cycle"
    state.as_array()[0, 0] = -1
    assert state.values[0][0] == 3000.
    state.verify_files()


def test_witness_reads_last_node_and_preserves_provenance(cycle_receipt):
    state = load_certified_cycle(cycle_receipt, cycle=21, side="left", muscles=["Biceps"])
    np.testing.assert_array_equal(state.as_array(), [[2900., .08, .15]])
    assert state.source_node == "last_node"
    assert state.completed_cycles == 21
    assert len(state.artifacts) == 6
    state.verify_files()
    cycle_receipt.write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        state.verify_files()


@pytest.mark.parametrize("mutation", ["certified", "replay", "digest", "cycle", "history"])
def test_uncertified_or_changed_restart_is_rejected(checkpoint, mutation):
    document = json.loads(checkpoint.read_text())
    key, value = {"certified": ("certified", False), "replay": ("replay_roundtrip_exact", False),
                  "digest": ("sha256", "0"*64), "cycle": ("completed_cycles", 19),
                  "history": ("stimulation_history_complete", False)}[mutation]
    document["arms"]["left"][key] = value
    save(checkpoint, document)
    with pytest.raises(ValueError):
        load_certified_checkpoint(checkpoint, side="left", muscles=["Biceps"])


@pytest.mark.parametrize("mutation", ["certified", "scope", "violation", "nan", "digest", "terminal", "side", "duplicate"])
def test_incomplete_or_changed_witness_is_rejected(cycle_receipt, mutation):
    document = json.loads(cycle_receipt.read_text())
    row = document["cycles"][0]
    if mutation == "certified":
        row["certified"] = False
    elif mutation == "scope":
        row["independent_audit"]["scope"] = "solver_status_only"
    elif mutation in ("violation", "nan"):
        row["maximum_normalized_violation"] = .01 if mutation == "violation" else float("nan")
    elif mutation == "digest":
        row["witness_sha256"] = "0"*64
    elif mutation == "terminal":
        row["terminal_slow_states"]["A_Biceps"] = 2800.
    elif mutation == "side":
        document["return_policy"]["side"] = "right"
    else:
        document["cycles"].append(row)
    save(cycle_receipt, document)
    with pytest.raises(ValueError):
        load_certified_cycle(cycle_receipt, cycle=21, side="left", muscles=["Biceps"])


def test_missing_muscle_is_rejected(checkpoint):
    with pytest.raises(ValueError, match="Missing slow state"):
        load_certified_checkpoint(checkpoint, side="left", muscles=["Triceps"])
