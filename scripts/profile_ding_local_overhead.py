"""Profile only repeated local adapter calls, excluding graph construction.

Pass the usual ``benchmark_ding_local_compiled.py`` arguments after ``--``.
The profile includes each complete adapter call (preparation, native solve,
reconstruction/certificates and synchronous research exports).
"""
from __future__ import annotations

import argparse
import cProfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_ding_local_compiled as compiled


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--scope", choices=("adapter", "rho"), default="adapter")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    profile_path = args.profile.resolve()
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    profile = cProfile.Profile()
    if args.scope == "rho":
        from bioptim.optimization.receding_horizon_optimization import RecedingHorizonOptimization
        owner, method = RecedingHorizonOptimization, "solve"
    else:
        owner, method = compiled.local.LocalSolver, "call"
    original = getattr(owner, method)

    def call(self, *call_args, **call_kwargs):
        return profile.runcall(original, self, *call_args, **call_kwargs)

    setattr(owner, method, call)
    try:
        compiled.main(arguments)
    finally:
        setattr(owner, method, original)
        profile.dump_stats(str(profile_path))


if __name__ == "__main__":
    main()
