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
    repeat_last_certified: bool = False,
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
    if repeat_last_certified and cycles_per_window != 1:
        raise ValueError("repeat_last_certified currently requires one cycle per window.")

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

        if repeat_last_certified:
            if completed_windows < 1:
                raise ValueError(
                    "repeat_last_certified requires at least one certified cycle."
                )
            first_control_node = (completed_windows - 1) * stimulations_per_cycle
            last_control_node = completed_windows * stimulations_per_cycle
        else:
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
            payload[key] = values[
                :, first_control_node : last_control_node + 1
            ].copy()
        for key in control_keys:
            values = np.asarray(archive[key])
            if values.shape[1] != expected_controls:
                raise ValueError(
                    f"Control trace {key} has {values.shape[1]} nodes; "
                    f"expected {expected_controls}."
                )
            payload[key] = values[:, first_control_node:last_control_node].copy()

    if repeat_last_certified:
        theta_key = "states__theta"
        if theta_key not in payload:
            raise ValueError(
                "repeat_last_certified requires the reduced mechanical state 'theta'."
            )
        theta = payload[theta_key]
        signed_cycle_shift = theta[:, -1] - theta[:, 0]
        if not np.all(np.isfinite(signed_cycle_shift)):
            raise ValueError("The certified crank-angle shift is not finite.")
        payload[theta_key] = theta + signed_cycle_shift[:, np.newaxis]
        for key in state_keys:
            if key == theta_key:
                continue
            # The new RHO starts at the exact terminal state of the last
            # certified RHO. Interior nodes remain the phase-aligned previous
            # cycle and are only an initial guess for the target solver.
            payload[key][:, 0] = payload[key][:, -1]

    checkpoint_metadata = dict(metadata)
    checkpoint_metadata.update(
        {
            "cycles_per_window": int(cycles_per_window),
            "producer_mode": (
                "certified_prefix_repeat_checkpoint"
                if repeat_last_certified
                else "certified_prefix_checkpoint"
            ),
            "producer_completed_windows": int(completed_windows),
            "producer_checkpoint_cycles": int(cycles_per_window),
            "producer_source_cycles": int(source_cycles),
            "producer_repeat_last_certified": bool(repeat_last_certified),
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
    parser.add_argument("--repeat-last-certified", action="store_true")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    metadata = extract_rho_checkpoint(
        args.source,
        args.output,
        completed_windows=args.completed_windows,
        cycles_per_window=args.cycles_per_window,
        repeat_last_certified=args.repeat_last_certified,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
