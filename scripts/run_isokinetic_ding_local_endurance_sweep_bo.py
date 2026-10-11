#!/usr/bin/env python3
"""Descending-resistance, fixed-weight BO sweep for a target RHO endurance.

At every 0.10 Nm step, a fresh 48-observation BO maximises certified cycles
for a unilateral prescribed-isokinetic RHO using exact Ding-local reduction.
The sweep stops after the first resistance whose best fixed-weight policy
reaches the requested lower target.  Results above max_cycles are right-
censored and reported explicitly rather than treated as physiological stops.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_isokinetic_ding_local_weight_transfer_study import _run_bo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hsl-library", required=True)
    parser.add_argument("--start-resistance", type=float, default=1.00)
    parser.add_argument("--step", type=float, default=0.10)
    parser.add_argument("--minimum-resistance", type=float, default=0.30)
    parser.add_argument("--target-cycles", type=int, default=1100)
    parser.add_argument("--max-cycles", type=int, default=1500)
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu-set", default="0-11")
    parser.add_argument("--ipopt-max-iter", type=int, default=1000)
    parser.add_argument("--startup-trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    args.root = args.root.resolve(); args.output = args.output.resolve()
    if args.step <= 0:
        raise ValueError("step must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    resistance = args.start_resistance
    while resistance >= args.minimum_resistance - 1e-12:
        resistance = round(resistance, 2)
        summary = _run_bo(resistance, args)
        summaries.append(summary)
        (args.output / "sweep-summary.json").write_text(
            json.dumps({"target_cycles": args.target_cycles, "max_cycles": args.max_cycles,
                        "steps": summaries}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if float(summary["best_cycles"]) >= args.target_cycles:
            break
        resistance = round(resistance - args.step, 2)


if __name__ == "__main__":
    main()
