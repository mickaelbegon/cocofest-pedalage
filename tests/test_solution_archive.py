"""Reproducible RHO/FHO exports use the existing independent replay reader."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.solution_archive import (
    ARCHIVE_SCHEMA, model_parameter_metadata, physical_archive_arrays,
)
from cocofest.optimization.configured_cycling_model import FIELD_ATTRIBUTES
from cocofest.optimization.solver_cross_rollout import load_source, evaluate_rollout
from tests.test_solver_cross_rollout import Profile, PARAMETERS, make_source


def archive_payload(cycles=2, stride=1):
    source = make_source(cycles)
    model = SimpleNamespace(muscles_dynamics_model=[SimpleNamespace(
        muscle_name="test", **{attribute: PARAMETERS[name] for name, attribute in FIELD_ATTRIBUTES.items()})])
    metadata = dict(source.metadata, cycle_duration_s=source.duration, producer_solver="ipopt",
                    producer_mode="full_horizon" if stride > 1 else "receding_horizon_concatenation",
                    producer_collocation_method="radau", producer_collocation_degree=5,
                    **model_parameter_metadata(model))
    # Repeated values are sufficient to check stride extraction; integration
    # uses the actual initial state and physical commands, never observations.
    states = {name: np.repeat(row[:-1], stride).tolist() + [row[-1]]
              for name, row in zip(source.state_names, source.shooting_states)}
    controls = {"last_pulse_width_test": source.controls}
    extra, metadata = physical_archive_arrays(states, controls, metadata)
    payload = {f"states__{name}": np.asarray(value)[None, :] for name, value in states.items()}
    payload.update({f"controls__{name}": value for name, value in controls.items()})
    payload.update(extra)
    return payload, metadata, source


@pytest.mark.parametrize("stride", [1, 6])
def test_new_rho_fho_archive_replays_without_legacy_overrides(tmp_path, stride):
    payload, metadata, reference = archive_payload(stride=stride)
    path = tmp_path / "trajectory.npz"
    np.savez(path, **payload, metadata__json=np.asarray(json.dumps(metadata)))
    loaded = load_source(path, Profile(), cycles=2)
    assert loaded.provenance["archive_schema"] == ARCHIVE_SCHEMA
    np.testing.assert_array_equal(loaded.initial_state, reference.initial_state)
    np.testing.assert_array_equal(loaded.controls, reference.controls)
    score, _ = evaluate_rollout(loaded, Profile())
    expected, _ = evaluate_rollout(reference, Profile())
    assert score["common_score"] == pytest.approx(expected["common_score"], rel=1e-12)
    second = load_source(path, Profile(), cycle_start=1)
    np.testing.assert_array_equal(second.initial_state, reference.shooting_states[:, 2])


@pytest.mark.parametrize("field", ["physical__initial_state", "physical__shooting_time_s"])
def test_new_archive_refuses_corrupted_initial_condition_or_grid(tmp_path, field):
    payload, metadata, _ = archive_payload()
    payload[field][0] += .1
    path = tmp_path / "corrupt.npz"
    np.savez(path, **payload, metadata__json=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError, match="inconsistent physical__"):
        load_source(path, Profile())


def test_new_archive_cannot_fall_back_to_legacy_defaults(tmp_path):
    payload, metadata, _ = archive_payload()
    del metadata["configured_muscle_parameters"]
    path = tmp_path / "incomplete.npz"
    np.savez(path, **payload, metadata__json=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError, match="missing configured_muscle_parameters"):
        load_source(path, Profile())


def test_effective_parameters_are_copied_from_model_after_configuration():
    muscle = SimpleNamespace(muscle_name="test", **{attr: PARAMETERS[key] for key, attr in FIELD_ATTRIBUTES.items()})
    muscle.alpha_a = -.004
    metadata = model_parameter_metadata(SimpleNamespace(muscles_dynamics_model=[muscle]))
    assert metadata["configured_muscle_parameters"]["test"]["alpha_a"] == -.004
    muscle.alpha_a = -.006
    assert metadata["configured_muscle_parameters"]["test"]["alpha_a"] == -.004


def test_production_writer_adds_physical_contract_to_fho_or_rho_seed(tmp_path):
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver

    payload, metadata, reference = archive_payload()
    states = {key[8:]: value for key, value in payload.items() if key.startswith("states__")}
    controls = {key[10:]: value for key, value in payload.items() if key.startswith("controls__")}
    path = tmp_path / "standard-export.npz"
    driver._save_warmup_cache(path, driver._WarmupSolutionAdapter(states, controls), metadata)
    loaded = load_source(path, Profile(), cycles=2)
    assert loaded.provenance["archive_schema"] == ARCHIVE_SCHEMA
    np.testing.assert_array_equal(loaded.initial_state, reference.initial_state)


def test_matrix_refuses_a_different_mechanical_profile(tmp_path, monkeypatch):
    from cocofest.optimization import solver_cross_rollout as replay

    payload, metadata, _ = archive_payload()
    metadata["reduced_profile"] = {"sha256": "different"}
    path = tmp_path / "trajectory.npz"
    np.savez(path, **payload, metadata__json=np.asarray(json.dumps(metadata)))
    profile = tmp_path / "profile.npz"
    profile.write_bytes(b"profile fixture")
    monkeypatch.setattr(replay.ReducedCyclingDynamics, "load", lambda path: Profile())
    with pytest.raises(ValueError, match="Reduced profile differs"):
        replay.run_matrix([path], profile, tmp_path / "replay", evaluators=["dop853"])
    assert not (tmp_path / "replay").exists()
