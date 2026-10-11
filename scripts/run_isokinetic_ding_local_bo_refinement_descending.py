#!/usr/bin/env python3
"""Refine fixed-weight Ding-local BO studies from high to low resistance.

Every target resistance imports only its own prior Optuna observations.  This
keeps the statistical objective coherent: a 1.35 Nm outcome is not presented
to TPE as if it were an observation at 1.30 Nm.  One process handles the
resistances sequentially, while each BO keeps its usual parallel candidate
batch on its assigned CPU set.
"""
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
    parser.add_argument(
        "--source", nargs=2, action="append", required=True,
        metavar=("RESISTANCE_NM", "STUDY"),
        help="Prior Optuna study directory/database for this exact resistance. Repeatable.",
    )
    parser.add_argument(
        "--seed-source", nargs=2, action="append", default=[],
        metavar=("TARGET_RESISTANCE_NM", "STUDY"),
        help=(
            "Study whose incumbent is re-evaluated at this target resistance; "
            "its objective value is not transferred. Repeatable."
        ),
    )
    parser.add_argument("--max-cycles", type=int, default=1500)
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu-set", default="12-23")
    parser.add_argument("--ipopt-max-iter", type=int, default=1000)
    parser.add_argument("--startup-trials", type=int, default=16)
    parser.add_argument(
        "--bo-score",
        choices=("certified_cycles", "certified_cycles_then_reserve"),
        default="certified_cycles_then_reserve",
        help="Use terminal capacity reserve only to break equal certified-cycle scores.",
    )
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.root = args.root.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    source_by_resistance = {round(float(resistance), 4): Path(source) for resistance, source in args.source}
    seed_by_resistance = {round(float(resistance), 4): Path(source) for resistance, source in args.seed_source}
    expected = {round(value, 4) for value in args.resistances}
    if set(source_by_resistance) != expected:
        raise ValueError("--source must supply exactly one prior study for every requested resistance.")
    args.output.mkdir(parents=True)
    summaries = []
    for resistance in sorted(args.resistances, reverse=True):
        source = source_by_resistance[round(resistance, 4)]
        args.warm_start_study = [source]
        args.enqueue_study = [seed_by_resistance[round(resistance, 4)]] if round(resistance, 4) in seed_by_resistance else []
        summary = _run_bo(resistance, args)
        summaries.append(summary)
        (args.output / "series-summary.json").write_text(
            json.dumps(
                {
                    "order": "descending_resistance",
                    "trials_per_resistance": args.trials,
                    "completed_resistances": summaries,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
