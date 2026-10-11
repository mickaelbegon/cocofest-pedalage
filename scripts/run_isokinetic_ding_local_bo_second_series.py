#!/usr/bin/env python3
"""Second independent 48-candidate Ding-local fixed-weight BO series."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_isokinetic_ding_local_weight_transfer_study import _run_bo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hsl-library", required=True)
    parser.add_argument("--resistances", type=float, nargs="+", required=True)
    parser.add_argument("--max-cycles", type=int, default=1500)
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu-set", default="0-11")
    parser.add_argument("--ipopt-max-iter", type=int, default=1000)
    parser.add_argument("--startup-trials", type=int, default=16)
    parser.add_argument(
        "--warm-start-study", type=Path, action="append", default=[],
        help="Completed Optuna study database/directory to import before TPE refinement. Repeatable.",
    )
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    args.output = args.output.resolve(); args.root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for resistance in args.resistances:
        summaries.append(_run_bo(resistance, args))
        (args.output / "series-summary.json").write_text(
            json.dumps({"series": 2, "trials_per_resistance": args.trials,
                        "completed_resistances": summaries}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
