#!/usr/bin/env python3
"""Write an immutable offline certified-capacity-frontier plan; solve nothing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.capacity_frontier import (
    build_capacity_feasibility_frontier_plan,
    build_capacity_frontier_plan,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-sha256", required=True)
    parser.add_argument("--fast-condition-sha256",
                        help="Required for the local cycle-feasibility atlas")
    parser.add_argument("--muscle", action="append", required=True, dest="muscles")
    parser.add_argument("--maximum-cycles", type=int,
                        help="Required only for the continuation-policy Pareto plan")
    parser.add_argument("--random-directions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--endpoint-tolerance-cycles", type=int, default=0)
    parser.add_argument("--mode", choices=("feasibility", "continuation"), default="feasibility")
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    output = args.pop("output").resolve()
    mode = args.pop("mode")
    fast_condition_sha256 = args.pop("fast_condition_sha256")
    if mode == "feasibility":
        if fast_condition_sha256 is None:
            parser.error("--fast-condition-sha256 is required in feasibility mode")
        args.pop("maximum_cycles")
        args.pop("endpoint_tolerance_cycles")
        plan = build_capacity_feasibility_frontier_plan(fast_condition_sha256=fast_condition_sha256, **args)
    else:
        if fast_condition_sha256 is not None:
            parser.error("--fast-condition-sha256 applies only to feasibility mode")
        if args["maximum_cycles"] is None:
            parser.error("--maximum-cycles is required in continuation mode")
        plan = build_capacity_frontier_plan(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
