import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "examples/fes_multibody/cycling/extract_rho_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("extract_rho_checkpoint", MODULE_PATH)
checkpoint_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(checkpoint_module)


def certified_prefix(path: Path) -> None:
    metadata = {
        "schema": "cocofest-common-periodic-initial-solution-v2",
        "producer_mode": "receding_horizon_concatenation",
        "cycles_per_window": 4,
        "stimulations_per_cycle": 2,
    }
    np.savez(
        path,
        states__theta=np.arange(9.0).reshape((1, 9)),
        states__omega=(100.0 + np.arange(9.0)).reshape((1, 9)),
        controls__pw=np.arange(8.0).reshape((1, 8)),
        metadata__json=np.asarray(json.dumps(metadata)),
    )


def test_extracts_exact_one_and_two_cycle_checkpoints(tmp_path):
    source = tmp_path / "prefix.npz"
    certified_prefix(source)

    one = tmp_path / "one.npz"
    metadata = checkpoint_module.extract_rho_checkpoint(
        source, one, completed_windows=2
    )
    with np.load(one, allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["states__theta"], [[4.0, 5.0, 6.0]])
        np.testing.assert_allclose(archive["controls__pw"], [[4.0, 5.0]])
    assert metadata["cycles_per_window"] == 1
    assert metadata["producer_completed_windows"] == 2

    two = tmp_path / "two.npz"
    metadata = checkpoint_module.extract_rho_checkpoint(
        source, two, completed_windows=1, cycles_per_window=2
    )
    with np.load(two, allow_pickle=False) as archive:
        np.testing.assert_allclose(
            archive["states__theta"], [[2.0, 3.0, 4.0, 5.0, 6.0]]
        )
        np.testing.assert_allclose(archive["controls__pw"], [[2.0, 3.0, 4.0, 5.0]])
    assert metadata["cycles_per_window"] == 2


def test_rejects_a_checkpoint_beyond_certified_prefix(tmp_path):
    source = tmp_path / "prefix.npz"
    certified_prefix(source)

    with pytest.raises(ValueError, match="exceeds the certified prefix"):
        checkpoint_module.extract_rho_checkpoint(
            source,
            tmp_path / "invalid.npz",
            completed_windows=3,
            cycles_per_window=2,
        )


def test_repeat_checkpoint_starts_from_last_certified_terminal(tmp_path):
    source = tmp_path / "prefix.npz"
    certified_prefix(source)
    output = tmp_path / "repeat.npz"

    metadata = checkpoint_module.extract_rho_checkpoint(
        source,
        output,
        completed_windows=4,
        repeat_last_certified=True,
    )

    with np.load(output, allow_pickle=False) as archive:
        # The measured signed cycle shift is +2 in this synthetic trace.
        np.testing.assert_allclose(archive["states__theta"], [[8.0, 9.0, 10.0]])
        # Every non-angle state starts at the exact certified terminal; the
        # remaining nodes retain the previous phase-aligned cycle as a seed.
        np.testing.assert_allclose(archive["states__omega"], [[108.0, 107.0, 108.0]])
        np.testing.assert_allclose(archive["controls__pw"], [[6.0, 7.0]])
    assert metadata["producer_mode"] == "certified_prefix_repeat_checkpoint"
    assert metadata["producer_repeat_last_certified"] is True
