#!/usr/bin/env python3
"""Freeze a direct-PW continuation-value protocol from three exact RHO anchors."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.continuation_value_pw_action import direct_pw_continuation_plan
from scripts.probe_independent_rho_task_reserve import _load_arm


def paired_directions(muscle_names: tuple[str, ...]) -> dict[str, list[float]]:
    """Predeclared shoulder/elbow exchanges and an independent mixed holdout."""
    required = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
    if tuple(muscle_names) != required:
        raise ValueError(f"Built-in PW directions require muscle order {required}")
    shoulder = np.array([1., -1., 0., 0.]) / np.sqrt(2.)
    elbow = np.array([0., 0., 1., -1.]) / np.sqrt(2.)
    mixed = (shoulder + elbow) / np.linalg.norm(shoulder + elbow)
    return {"shoulder_exchange": shoulder.tolist(), "elbow_exchange": elbow.tolist(), "mixed_holdout": mixed.tolist()}


def parse_anchor(value: str) -> tuple[str, Path]:
    try:
        identifier, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Anchor must be ID=/absolute/or/relative/receipt.json") from error
    if not identifier:
        raise argparse.ArgumentTypeError("Anchor id cannot be empty")
    return identifier, Path(raw_path).resolve()


def plan(anchor_specs: list[tuple[str, Path]], *, side: str, amplitude_fraction: float,
         maximum_absolute_cycles: int) -> dict:
    if side != "left":
        raise ValueError("The first direct-PW protocol is preregistered for the left-arm reference only")
    anchors, names, model = [], None, None
    for identifier, receipt_path in anchor_specs:
        receipt_path = Path(receipt_path).resolve(strict=True)
        source, payload = _load_arm(receipt_path, side)
        current_names = tuple(json.loads(Path(source.model_path).read_text())["muscles"])
        if names is None:
            names, model = current_names, source.model_sha256
        if current_names != names or source.model_sha256 != model:
            raise ValueError("All direct-PW anchors must use the same ordered model")
        anchors.append({"id": identifier, "completed_cycles": source.completed_cycles,
                        "source_receipt": str(receipt_path), "source_receipt_sha256": sha256(receipt_path.read_bytes()).hexdigest(),
                        "prepared_problem_sha256": source.prepared_problem_sha256})
    if names is None:
        raise ValueError("At least one anchor is required")
    return direct_pw_continuation_plan(anchors, muscle_names=names, directions=paired_directions(names),
                                       amplitude_fraction=amplitude_fraction,
                                       maximum_absolute_cycles=maximum_absolute_cycles)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", action="append", required=True, type=parse_anchor,
                        help="Repeat exactly three times: c120=/path/to/receipt.json")
    parser.add_argument("--side", choices=("left",), default="left")
    parser.add_argument("--amplitude-fraction", type=float, default=.05)
    parser.add_argument("--maximum-absolute-cycles", type=int, default=240)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    result = plan(args.anchor, side=args.side, amplitude_fraction=args.amplitude_fraction,
                  maximum_absolute_cycles=args.maximum_absolute_cycles)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
