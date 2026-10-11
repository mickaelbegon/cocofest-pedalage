#!/usr/bin/env python3
"""Convert a concatenated one-cycle RHO E_prod trace into FHO's gauge.

RHO deliberately resets its accounting integral at every cycle.  An FHO has
one continuous integral, so a RHO-BO prefix must be rebased before it can be
used as an FHO initial guess.  Physical states and controls are untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, required=True)
    parser.add_argument("--work-per-cycle-j", type=float, required=True)
    args = parser.parse_args()
    if args.cycles < 1 or not np.isfinite(args.work_per_cycle_j):
        raise ValueError("cycles and work-per-cycle-j must be finite and positive.")

    with np.load(args.input, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    energy = np.asarray(payload["states__E_prod"], dtype=float)
    if energy.ndim != 2 or energy.shape[0] != 1:
        raise ValueError("Expected scalar E_prod with one row.")
    stride, remainder = divmod(energy.shape[1] - 1, args.cycles)
    if remainder or stride < 1:
        raise ValueError("E_prod nodes are incompatible with the requested cycle count.")
    cycle_index = np.minimum(np.arange(energy.shape[1]) // stride, args.cycles - 1)
    payload["states__E_prod"] = energy + (
        cycle_index * args.work_per_cycle_j
    )[np.newaxis, :]
    metadata = json.loads(str(payload["metadata__json"]))
    metadata["producer_e_prod_rebased_from_local_rho"] = True
    metadata["producer_e_prod_rebase_cycles"] = int(args.cycles)
    metadata["producer_e_prod_work_per_cycle_j"] = float(args.work_per_cycle_j)
    payload["metadata__json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)


if __name__ == "__main__":
    main()
