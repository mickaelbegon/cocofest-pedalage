#!/usr/bin/env python3
"""Export an immutable exact-state triceps-exchange slice from frontier evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.capacity_frontier_archive import (  # noqa: E402
    serialize_slice,
    slice_from_raw_reports,
    slice_from_summary,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-cycle", type=int, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--report", type=Path, action="append",
                        help="Raw triceps-support/epsilon receipts; repeat four times.")
    source.add_argument("--summary", type=Path,
                        help="Immutable pilot summary only when raw receipts are unavailable.")
    parser.add_argument("--capacity-domain", type=float, nargs=2, metavar=("LOWER", "UPPER"),
                        help="Required with --summary; recorded physical capacity box.")
    parser.add_argument("--epsilon-upper", type=float,
                        help="Required with --summary; common non-triceps epsilon bound.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report:
        if args.capacity_domain is not None or args.epsilon_upper is not None:
            parser.error("--capacity-domain and --epsilon-upper apply only to --summary")
        result = slice_from_raw_reports(args.report, checkpoint_cycle=args.checkpoint_cycle)
    else:
        if args.capacity_domain is None or args.epsilon_upper is None:
            parser.error("--summary requires --capacity-domain and --epsilon-upper")
        result = slice_from_summary(args.summary, checkpoint_cycle=args.checkpoint_cycle,
                                    capacity_domain=tuple(args.capacity_domain), epsilon_upper=args.epsilon_upper)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(serialize_slice(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
