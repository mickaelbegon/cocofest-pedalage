"""Extract a solver-independent RHO checkpoint from a certified prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def extract_rho_checkpoint(
    source: Path,
    output: Path,
    *,
    completed_windows: int,
    cycles_per_window: int = 1,
) -> dict:
    """Slice the next complete OCP window after ``completed_windows`` cycles.

    The source must be a certified, concatenated one-cycle RHO trajectory.
    States contain both endpoints, while controls contain one value per
    shooting interval.  No interpolation is performed.
    """

    if completed_windows < 0:
        raise ValueError("completed_windows must be non-negative.")
    if cycles_per_window < 1:
        raise ValueError("cycles_per_window must be positive.")

    with np.load(source, allow_pickle=False) as archive:
        if "metadata__json" not in archive.files:
            raise ValueError(f"{source} has no certified trajectory metadata.")
        metadata = json.loads(str(archive["metadata__json"].item()))
        state_keys = sorted(k for k in archive.files if k.startswith("states__"))
        control_keys = sorted(k for k in archive.files if k.startswith("controls__"))
        if not state_keys or not control_keys:
            raise ValueError(f"{source} has no state/control trajectory.")
        if metadata.get("producer_mode") != "receding_horizon_concatenation":
            raise ValueError(
                "The source must be a certified receding-horizon concatenation."
            )
        source_cycles = int(metadata["cycles_per_window"])
        stimulations_per_cycle = int(metadata["stimulations_per_cycle"])
        first_control = np.asarray(archive[control_keys[0]])
        expected_controls = source_cycles * stimulations_per_cycle
        if first_control.shape[1] != expected_controls:
            raise ValueError(
                "Control trace length is inconsistent with trajectory metadata: "
                f"{first_control.shape[1]} != {expected_controls}."
            )

        first_control_node = completed_windows * stimulations_per_cycle
        last_control_node = (
            completed_windows + cycles_per_window
        ) * stimulations_per_cycle
        if last_control_node > expected_controls:
            raise ValueError(
                "The requested checkpoint exceeds the certified prefix: "
                f"cycles {completed_windows}:{completed_windows + cycles_per_window} "
                f"of {source_cycles}."
            )

        payload = {}
        for key in state_keys:
            values = np.asarray(archive[key])
            if values.shape[1] != expected_controls + 1:
                raise ValueError(
                    f"State trace {key} has {values.shape[1]} nodes; "
                    f"expected {expected_controls + 1}."
                )
            payload[key] = values[:, first_control_node : last_control_node + 1]
        for key in control_keys:
            values = np.asarray(archive[key])
            if values.shape[1] != expected_controls:
                raise ValueError(
                    f"Control trace {key} has {values.shape[1]} nodes; "
                    f"expected {expected_controls}."
                )
            payload[key] = values[:, first_control_node:last_control_node]

    checkpoint_metadata = dict(metadata)
    checkpoint_metadata.update(
        {
            "cycles_per_window": int(cycles_per_window),
            "producer_mode": "certified_prefix_checkpoint",
            "producer_completed_windows": int(completed_windows),
            "producer_checkpoint_cycles": int(cycles_per_window),
            "producer_source_cycles": int(source_cycles),
        }
    )
    payload["metadata__json"] = np.asarray(
        json.dumps(checkpoint_metadata, sort_keys=True, separators=(",", ":"))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    return checkpoint_metadata


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-windows", type=int, required=True)
    parser.add_argument("--cycles-per-window", type=int, default=1)
    return parser


def main() -> None:
    args = build_cli().parse_args()
    metadata = extract_rho_checkpoint(
        args.source,
        args.output,
        completed_windows=args.completed_windows,
        cycles_per_window=args.cycles_per_window,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
