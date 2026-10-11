#!/usr/bin/env python3
"""Run one matched equal-weight prescribed-isokinetic RHO to endurance."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_isokinetic_ding_local_weight_transfer_study import _run_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resistance", type=float, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hsl-library", required=True)
    parser.add_argument("--max-cycles", type=int, default=1500)
    parser.add_argument("--cpu-set", default="22-23")
    parser.add_argument("--ipopt-max-iter", type=int, default=1000)
    args = parser.parse_args()
    args.output = args.output.resolve(); args.root = args.root.resolve()
    _run_case(f"unit-{args.resistance:.2f}", args.resistance, (1., 1., 1., 1.), args.output, args)


if __name__ == "__main__":
    main()
