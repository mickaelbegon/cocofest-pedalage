#!/usr/bin/env python3
"""Write a new Phi fast-state closure plan; performs no simulations."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cocofest.cycling.slow.fast_state_probes import build_fast_probe_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--muscles", nargs="+", required=True)
    parser.add_argument("--control-repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_fast_probe_plan(args.checkpoint, side=args.side, muscles=args.muscles,
                                 control_repeats=args.control_repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "status": plan["status"],
                      "plan_sha256": plan["plan_sha256"]}))


if __name__ == "__main__":
    main()
