"""Small explicit bridge selecting fixed Physio versus adaptive PACE weights."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("rho-physio", "rho-pace"))
    parser.add_argument("--pace-config", required=True)
    parser.add_argument("--pace-journal", required=True)
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from scripts.run_rho_pace_benchmark import main as run_weighted
    benchmark = args.benchmark_args[1:] if args.benchmark_args[:1] == ["--"] else args.benchmark_args
    return run_weighted(["--pace-config", args.pace_config, "--pace-journal", args.pace_journal,
                         "--", *benchmark], adaptation_enabled=args.mode == "rho-pace")


if __name__ == "__main__":
    main()
