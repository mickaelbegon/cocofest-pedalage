"""Synchronize experimental compiled unilateral RHO solves at each NLP call.

Each worker still propagates its own solution. The file barrier aligns the two
solver calls and measures pair latency independently of subsequent report I/O.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from scripts import benchmark_ding_local_compiled as compiled
from scripts import benchmark_ding_radau5_local_ab as local


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--barrier-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    options, arguments = parser.parse_known_args(argv)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if "--output-json" not in arguments:
        parser.error("Pass compiled benchmark arguments including --output-json")
    output = Path(arguments[arguments.index("--output-json") + 1]).resolve().parent
    timing_file = output / "symbolic-audit" / "pair_timing.jsonl"
    timing_file.parent.mkdir(parents=True, exist_ok=True)
    options.barrier_dir.mkdir(parents=True, exist_ok=True)
    original_call = local.LocalSolver.call

    def synchronized_call(self, original):
        cycle = self.calls
        ready = options.barrier_dir / f"cycle-{cycle:04d}-{options.side}.ready"
        peer = options.barrier_dir / f"cycle-{cycle:04d}-{'right' if options.side == 'left' else 'left'}.ready"
        ready.touch(exist_ok=False)
        deadline = time.monotonic() + options.timeout_s
        while not peer.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Peer did not reach paired NLP cycle {cycle}")
            time.sleep(0.001)
        release_ns = time.monotonic_ns()
        try:
            result = original_call(self, original)
        finally:
            done_ns = time.monotonic_ns()
            with timing_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"cycle": cycle, "side": options.side,
                    "release_ns": release_ns, "done_ns": done_ns}) + "\n")
        return result

    local.LocalSolver.call = synchronized_call
    compiled.main(arguments)


if __name__ == "__main__":
    main()
