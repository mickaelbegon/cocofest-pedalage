#!/usr/bin/env python3
"""Hold out c140 while interpolating conservative *local* ε exchange slopes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.capacity_frontier_interpolation import diagnostic_frontier_weight_hint  # noqa: E402


def _load(path: Path) -> dict:
    value = json.loads(Path(path).resolve(strict=True).read_text())
    if value.get("kind") != "local_epsilon_exchange_replication_summary":
        raise ValueError("Expected an epsilon replication summary")
    if not value.get("all_monotonicity_checks_passed"):
        raise ValueError("All monotonicity checks must pass before a slope audit")
    return value


def _local_interval(row: dict) -> tuple[float, float]:
    cuts = row["cuts"]
    eps = [float(item["epsilon_upper"]) for item in cuts]
    tri = [float(item["capacity_ratio"][-1]) for item in cuts]
    if len(cuts) != 3 or eps != sorted(eps) or not all(math.isfinite(value) for value in eps + tri):
        raise ValueError("Each curve needs exactly three ordered finite epsilon cuts")
    secants = [(left - right) / (right_eps - left_eps)
                for left, right, left_eps, right_eps in zip(tri, tri[1:], eps, eps[1:])]
    if any(value <= 0 for value in secants):
        raise ValueError("Local triceps exchange slope must be positive")
    return min(secants), max(secants)


def audit(lower: dict, holdout: dict, upper: dict, *, maximum_midpoint_relative_error=.10) -> dict:
    order = tuple(holdout["muscle_order"])
    if tuple(lower["muscle_order"]) != order or tuple(upper["muscle_order"]) != order:
        raise ValueError("Muscle order differs across epsilon summaries")
    rows = [{row["muscle"]: row for row in value["rows"]} for value in (lower, holdout, upper)]
    if any(set(item) != set(order) - {"Triceps"} for item in rows):
        raise ValueError("Each summary requires one curve per non-triceps muscle")
    report, conservative = {}, {}
    for muscle in order:
        if muscle == "Triceps":
            continue
        low, observed, high = (_local_interval(item[muscle]) for item in rows)
        predicted = tuple((left + right) / 2. for left, right in zip(low, high))
        predicted_midpoint, observed_midpoint = sum(predicted) / 2., sum(observed) / 2.
        error = abs(predicted_midpoint - observed_midpoint) / observed_midpoint
        overlap = max(predicted[0], observed[0]) <= min(predicted[1], observed[1])
        report[muscle] = {"predicted_local_slope_interval": list(predicted),
                          "observed_local_slope_interval": list(observed),
                          "intervals_overlap": overlap,
                          "midpoint_relative_error": error}
        conservative[muscle] = predicted[0]
    passed = all(row["intervals_overlap"] and row["midpoint_relative_error"] <= maximum_midpoint_relative_error
                 for row in report.values())
    return {"schema_version": 1, "kind": "offline_local_epsilon_slope_holdout",
            "lower_checkpoint_cycle": 120, "holdout_checkpoint_cycle": 140, "upper_checkpoint_cycle": 160,
            "muscle_order": list(order),
            "locality": "two adjacent epsilon secants spanning epsilon=0.6; conservative value is their lower slope",
            "maximum_midpoint_relative_error": maximum_midpoint_relative_error, "muscles": report,
            "predicted_conservative_exchange_slopes": conservative,
            "diagnostic_fatigue_weights": diagnostic_frontier_weight_hint(conservative),
            "passed": passed, "online_use": "forbidden_pending_causal_RHO_holdout"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c120", type=Path, required=True)
    parser.add_argument("--c140", type=Path, required=True)
    parser.add_argument("--c160", type=Path, required=True)
    parser.add_argument("--maximum-midpoint-relative-error", type=float, default=.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = audit(_load(args.c120), _load(args.c140), _load(args.c160),
                  maximum_midpoint_relative_error=args.maximum_midpoint_relative_error)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
