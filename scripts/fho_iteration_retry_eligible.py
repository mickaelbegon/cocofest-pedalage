#!/usr/bin/env python3
"""Decide whether a capped FHO solve deserves one longer retry.

The criterion deliberately requires a stable final iterate, rather than merely
having visited a feasible point earlier in the run.  This prevents a 1,500-
iteration retry from being spent on the oscillatory numerical failures seen at
FHO93/FHO94.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and float(value) == float(value) and abs(float(value)) != float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--iteration-budget", type=int, required=True)
    args = parser.parse_args()
    try:
        result = json.loads(args.result.read_text(encoding="utf-8"))["results"][0]
        stats = (result.get("nlp_solver_stats") or [{}])[0]
        diagnostics = stats.get("iteration_diagnostics") or {}
        primal = diagnostics.get("inf_pr") or {}
        dual = diagnostics.get("inf_du") or {}
        objective = diagnostics.get("obj") or {}
        eligible = (
            stats.get("return_status") == "Maximum_Iterations_Exceeded"
            and int(stats.get("iter_count", -1)) >= args.iteration_budget
            and all(finite(primal.get(key)) for key in ("initial", "final"))
            and finite(dual.get("final"))
            and all(finite(objective.get(key)) for key in ("initial", "final"))
            # The current iterate must be both substantially closer to
            # feasibility and numerically stable enough to warrant extension.
            and float(primal["final"]) <= min(1e-2, float(primal["initial"]) * 1e-4)
            and float(dual["final"]) <= 1e3
            and float(objective["final"]) <= 1.25 * float(objective["initial"])
        )
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        eligible = False
    raise SystemExit(0 if eligible else 1)


if __name__ == "__main__":
    main()
