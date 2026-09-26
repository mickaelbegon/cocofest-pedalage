"""Package archive interfaces stay usable without importing solver examples."""
import builtins
import json
from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.configured_cycling_model import FINGERPRINT_KEY
from cocofest.optimization.configured_rho_checkpoints import configured_checkpoint_writer
from cocofest.optimization.trajectory_io import (
    WarmupSolutionAdapter, load_trajectory, save_trajectory,
    checkpoint_publication_hook, save_rho_replay_checkpoint,
)


def test_legacy_npz_roundtrip_and_example_alias(tmp_path):
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver

    states, controls = {"F_test": np.array([[1., 2.]])}, {"pw": np.array([[.001]])}
    path = tmp_path / "legacy.npz"
    save_trajectory(path, states, controls, {"old": True}, {"pw": controls["pw"]})
    loaded = driver._load_warmup_cache(path)
    assert driver._WarmupSolutionAdapter is WarmupSolutionAdapter
    assert loaded.metadata == {"old": True}
    np.testing.assert_array_equal(loaded.decision_states()["F_test"], states["F_test"])
    with np.load(path) as archive:
        np.testing.assert_array_equal(archive["applied_pulse_widths__pw"], controls["pw"])
    second = tmp_path / "wrapper.npz"
    driver._save_warmup_cache(second, loaded, loaded.metadata)
    np.testing.assert_array_equal(load_trajectory(second).decision_controls()["pw"], controls["pw"])


def test_configured_checkpoint_uses_package_hook_without_example_import(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "examples" or name.startswith("examples."):
            raise AssertionError("Package checkpoint publication must not import examples")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    nmpc = SimpleNamespace(nlp=[SimpleNamespace(
        x_init={"F_test": SimpleNamespace(init=np.array([[1., 2.]]))},
        u_init={"pw": SimpleNamespace(init=np.array([[.001]]))})])
    config = {FINGERPRINT_KEY: "fixture", "muscles": {"test": {}}, "case_id": "test"}
    directory = tmp_path / "checkpoint"
    with configured_checkpoint_writer(directory, config, condition="rho", arguments=[]):
        save_rho_replay_checkpoint(directory / "cycle-1.npz", nmpc, None,
                                   completed_windows=1, metadata_factory=lambda _: {})
    receipt = json.loads((directory / "latest.json").read_text())
    assert receipt["target_rho"] == 2
    assert load_trajectory(directory / "cycle-1.npz").metadata[FINGERPRINT_KEY] == "fixture"
    # The publication scope has ended: an ordinary checkpoint has no receipt.
    save_rho_replay_checkpoint(tmp_path / "ordinary.npz", nmpc, None,
                               completed_windows=2, metadata_factory=lambda _: {})
    assert not (tmp_path / "ordinary.receipt.json").exists()


def test_checkpoint_scope_restores_outer_policy_after_failure(tmp_path):
    events = []
    def outer(write, *args, **kwargs):
        events.append("outer")
    def inner(write, *args, **kwargs):
        raise RuntimeError("fixture")
    with checkpoint_publication_hook(outer):
        with pytest.raises(RuntimeError, match="fixture"):
            with checkpoint_publication_hook(inner):
                save_rho_replay_checkpoint(tmp_path / "x", None, None,
                                           completed_windows=1, metadata_factory=lambda _: {})
        save_rho_replay_checkpoint(tmp_path / "x", None, None,
                                   completed_windows=1, metadata_factory=lambda _: {})
    assert events == ["outer"]
