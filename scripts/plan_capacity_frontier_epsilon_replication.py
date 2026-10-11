#!/usr/bin/env python3
"""Write, but never run, missing local ε-cut jobs for checkpoint replication."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MUSCLES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")


def plan(*, receipts: dict[int, Path], output_directory: Path, epsilon_values: tuple[float, ...],
         lower_ratio: float, upper_ratio: float, max_iterations: int) -> dict:
    if not receipts or any(type(cycle) is not int or cycle < 0 for cycle in receipts):
        raise ValueError("At least one non-negative integer checkpoint is required")
    if not (0 <= lower_ratio < upper_ratio <= 1):
        raise ValueError("capacity domain must be a physical ordered ratio box")
    if not epsilon_values or any(not lower_ratio < value < upper_ratio for value in epsilon_values):
        raise ValueError("every epsilon value must lie strictly inside the capacity domain")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    jobs = []
    for checkpoint, receipt in sorted(receipts.items()):
        receipt = Path(receipt).resolve(strict=True)
        for muscle_index, muscle in enumerate(MUSCLES[:-1]):
            for epsilon in epsilon_values:
                bounds = [None] * len(MUSCLES)
                bounds[muscle_index] = epsilon
                identifier = f"c{checkpoint}-left-epsilon-{muscle.lower()}-{epsilon:.3f}".replace(".", "p")
                jobs.append({
                    "id": identifier,
                    "checkpoint_cycle": checkpoint,
                    "side": "left",
                    "receipt": str(receipt),
                    "receipt_sha256": sha256(receipt.read_bytes()).hexdigest(),
                    "support": [0., 0., 0., 1.],
                    "epsilon_upper": bounds,
                    "capacity_domain": [lower_ratio, upper_ratio],
                    "max_iterations": max_iterations,
                    "required_validation": ["fresh_discrete_rebuild", "DOP853", "epsilon_constraint"],
                    "output": str(output_directory.resolve() / f"capacity-frontier-{identifier}.json"),
                })
    body = {
        "schema_version": 1,
        "kind": "capacity_frontier_checkpoint_epsilon_replication_plan",
        "interpretation": "offline local-frontier replication only; does not enable an RHO policy",
        "muscle_order": list(MUSCLES),
        "jobs": jobs,
        "jobs_must_run_only_on_idle_cores": True,
        "completion_gate": "all raw receipts independently rebuilt and DOP853 replayed before archive/export",
    }
    return {**body, "plan_sha256": sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-c120", type=Path, help="Legacy c120 receipt input.")
    parser.add_argument("--receipt-c160", type=Path, help="Legacy c160 receipt input.")
    parser.add_argument("--checkpoint", action="append", metavar="CYCLE=RECEIPT",
                        help="Generic checkpoint input; repeat as needed. Cannot mix with legacy flags.")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, nargs="+", default=(.4, .8))
    parser.add_argument("--lower-ratio", type=float, default=.1)
    parser.add_argument("--upper-ratio", type=float, default=1.)
    parser.add_argument("--max-iterations", type=int, default=800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.checkpoint:
        if args.receipt_c120 or args.receipt_c160:
            parser.error("--checkpoint cannot be mixed with --receipt-c120/--receipt-c160")
        receipts = {}
        for item in args.checkpoint:
            try:
                cycle, path = item.split("=", 1)
                cycle = int(cycle)
            except ValueError:
                parser.error("--checkpoint must use CYCLE=RECEIPT")
            if cycle in receipts:
                parser.error("Each --checkpoint cycle must be unique")
            receipts[cycle] = Path(path)
    elif args.receipt_c120 and args.receipt_c160:
        receipts = {120: args.receipt_c120, 160: args.receipt_c160}
    else:
        parser.error("Supply --checkpoint CYCLE=RECEIPT or both legacy c120/c160 receipts")
    document = plan(receipts=receipts,
                    output_directory=args.output_directory, epsilon_values=tuple(args.epsilon),
                    lower_ratio=args.lower_ratio, upper_ratio=args.upper_ratio, max_iterations=args.max_iterations)
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(destination)


if __name__ == "__main__":
    main()
