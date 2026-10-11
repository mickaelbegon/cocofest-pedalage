#!/usr/bin/env python3
"""Truncate an exported homogeneous-cycle FHO/RHO seed at a cycle boundary."""

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
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    metadata = json.loads(str(data["metadata__json"]))
    source_cycles = int(metadata["cycles_per_window"])
    if not 1 <= args.cycles <= source_cycles:
        raise ValueError(f"--cycles must be in [1, {source_cycles}].")
    for key, value in tuple(data.items()):
        if key.startswith("states__"):
            stride, remainder = divmod(value.shape[-1] - 1, source_cycles)
            if remainder:
                raise ValueError(f"State archive {key} is not cycle-aligned.")
            data[key] = value[..., : args.cycles * stride + 1]
        elif key.startswith(("controls__", "applied_pulse_widths__")):
            stride, remainder = divmod(value.shape[-1], source_cycles)
            if remainder:
                raise ValueError(f"Control archive {key} is not cycle-aligned.")
            data[key] = value[..., : args.cycles * stride]
        elif key == "physical__shooting_time_s":
            stride, remainder = divmod(value.shape[-1] - 1, source_cycles)
            if remainder:
                raise ValueError("Physical time archive is not cycle-aligned.")
            data[key] = value[..., : args.cycles * stride + 1]
    for key in ("cycles_per_window", "physical_cycles", "producer_cycles_per_window", "producer_requested_cycles"):
        if key in metadata:
            metadata[key] = args.cycles
    metadata["truncated_from_cycles"] = source_cycles
    data["metadata__json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **data)


if __name__ == "__main__":
    main()
