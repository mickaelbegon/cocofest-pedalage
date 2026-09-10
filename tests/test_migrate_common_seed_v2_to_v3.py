import json

import numpy as np
import pytest

from scripts.migrate_common_seed_v2_to_v3 import migrate_seed, upgrade_metadata


def _v2_metadata():
    return {
        "schema": "cocofest-common-periodic-initial-solution-v2",
        "torque_application": "constant",
        "mechanical_formulation": "reduced",
        "constant_crank_torque": .1,
        "signed_crank_torque_nm": .1,
    }


def test_upgrade_adds_only_the_explicit_dynamic_defaults():
    upgraded = upgrade_metadata(_v2_metadata(), signed_crank_torque=.1)
    assert upgraded["schema"] == "cocofest-common-periodic-initial-solution-v3"
    assert upgraded["formulation"] == "dynamic"
    assert upgraded["terminal_reserve_effective_weight"] == 0.


def test_upgrade_rejects_a_torque_mismatch_or_nonconstant_seed():
    with pytest.raises(ValueError, match="signed torque"):
        upgrade_metadata(_v2_metadata(), signed_crank_torque=.2)
    metadata = _v2_metadata()
    metadata["torque_application"] = "isokinetic"
    with pytest.raises(ValueError, match="constant-torque"):
        upgrade_metadata(metadata, signed_crank_torque=.1)


def test_migration_preserves_all_numeric_arrays_and_writes_a_new_archive(tmp_path):
    source, destination = tmp_path / "v2.npz", tmp_path / "v3.npz"
    state = np.arange(6.).reshape(2, 3)
    np.savez_compressed(source, state=state, metadata__json=np.asarray(json.dumps(_v2_metadata())))
    migrated = migrate_seed(source, destination, signed_crank_torque=.1)
    assert migrated["schema"].endswith("v3")
    with np.load(source, allow_pickle=False) as original, np.load(destination, allow_pickle=False) as upgraded:
        np.testing.assert_array_equal(upgraded["state"], original["state"])
        assert json.loads(str(upgraded["metadata__json"].item()))["schema"] == migrated["schema"]
