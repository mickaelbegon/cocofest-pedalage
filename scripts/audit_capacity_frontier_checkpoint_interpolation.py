#!/usr/bin/env python3
"""Perform a strict c120/c140/c160 held-out local exchange interpolation audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.capacity_frontier_archive import deserialize_slice  # noqa: E402
from cocofest.optimization.capacity_frontier_interpolation import audit_linear_checkpoint_interpolation  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--upper", type=Path, required=True)
    parser.add_argument("--maximum-relative-error", type=float, default=.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = audit_linear_checkpoint_interpolation(
        deserialize_slice(args.lower), deserialize_slice(args.holdout), deserialize_slice(args.upper),
        maximum_relative_error=args.maximum_relative_error,
    )
    audit["slice_paths"] = {"lower": str(args.lower.resolve()), "holdout": str(args.holdout.resolve()),
                            "upper": str(args.upper.resolve())}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
