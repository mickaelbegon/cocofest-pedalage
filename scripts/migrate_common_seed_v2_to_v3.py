"""Create an auditable v3 copy of a dynamic, constant-torque v2 RHO seed.

This preserves every numerical trajectory array.  It only supplies fields
introduced by schema v3 when their values are the historical dynamic defaults.
It does not certify the seed, the RHO, or a future FHO solve.
"""

import argparse
import json
from pathlib import Path

import numpy as np


V2_SCHEMA = "cocofest-common-periodic-initial-solution-v2"
V3_SCHEMA = "cocofest-common-periodic-initial-solution-v3"
V3_DYNAMIC_DEFAULTS = {
    "formulation": "dynamic",
    "isokinetic_omega": -float(2 * np.pi),
    "energy_equivalent_torque": 0.2,
    "load_torque_min": -3.0,
    "load_torque_max": 3.0,
    "terminal_reserve_weight": 0.0,
    "terminal_reserve_temperature": 0.005,
    "terminal_reserve_effective_weight": 0.0,
}


def upgrade_metadata(metadata, *, signed_crank_torque):
    """Upgrade only a physically explicit v2 dynamic seed metadata mapping."""
    upgraded = dict(metadata)
    if upgraded.get("schema") != V2_SCHEMA:
        raise ValueError(f"Expected schema {V2_SCHEMA!r}, got {upgraded.get('schema')!r}.")
    if upgraded.get("torque_application") != "constant":
        raise ValueError("Only constant-torque v2 seeds can be migrated.")
    if upgraded.get("mechanical_formulation") not in ("reduced", "full"):
        raise ValueError("The v2 seed must declare reduced or full mechanics.")
    for field in ("constant_crank_torque", "signed_crank_torque_nm"):
        if not np.isclose(float(upgraded.get(field, np.nan)), signed_crank_torque, atol=1e-12):
            raise ValueError(f"Seed {field} does not match the declared signed torque.")
    for field, default in V3_DYNAMIC_DEFAULTS.items():
        if field in upgraded and upgraded[field] != default:
            raise ValueError(f"Seed {field} is incompatible with the v3 dynamic default.")
        upgraded[field] = default
    upgraded["schema"] = V3_SCHEMA
    return upgraded


def migrate_seed(source, destination, *, signed_crank_torque):
    """Write a new NPZ; never modify the source archive in place."""
    source, destination = Path(source), Path(destination)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files if name != "metadata__json"}
        metadata = json.loads(str(archive["metadata__json"].item()))
    upgraded = upgrade_metadata(metadata, signed_crank_torque=signed_crank_torque)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays, metadata__json=np.asarray(json.dumps(upgraded, sort_keys=True)))
    return upgraded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--signed-crank-torque", type=float, required=True)
    args = parser.parse_args(argv)
    metadata = migrate_seed(args.source, args.destination, signed_crank_torque=args.signed_crank_torque)
    print(json.dumps({"destination": str(args.destination), "schema": metadata["schema"],
                      "signed_crank_torque_nm": metadata["signed_crank_torque_nm"]}))


if __name__ == "__main__":
    main()
