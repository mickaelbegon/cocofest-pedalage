#!/usr/bin/env python3
"""Summarize registered ε cuts as local tradeoff curves, not a global front."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.capacity_frontier_archive import slice_from_raw_reports  # noqa: E402


def _load(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text())
    if value.get("status") != "fresh_discrete_rebuild_and_DOP853_completed_not_global_front_certified":
        raise ValueError(f"{path}: incomplete independent rebuild/replay")
    rebuild = value.get("independent_rebuild", {})
    if not rebuild.get("discrete_audit_passed") or rebuild.get("continuous_DOP853_replay") != "complete":
        raise ValueError(f"{path}: numerical gate unavailable")
    return value


def summarize(*, minimum_reports: dict[int, Path], epsilon_reports: list[Path], maximum_work_error=.001,
              maximum_endpoint_error=.0015, epsilon_tolerance=1e-6) -> dict:
    if not minimum_reports or any(type(cycle) is not int or cycle < 0 for cycle in minimum_reports):
        raise ValueError("At least one non-negative integer minimum-support checkpoint is required")
    grouped: dict[int, dict[str, list[tuple[float, tuple[float, ...], str]]]] = {
        cycle: {} for cycle in minimum_reports
    }
    contexts, order = {}, None
    for cycle, path in minimum_reports.items():
        value = _load(path)
        contexts[cycle] = value["condition_sha256"]
        if order is None:
            order = tuple(value["muscle_order"])
    if order is None:
        raise ValueError("No muscle order")
    tri = order.index("Triceps")
    minima = {}
    for cycle, path in minimum_reports.items():
        value = _load(path)
        if tuple(value["muscle_order"]) != order:
            raise ValueError("Minimum reports disagree on muscle order")
        minima[cycle] = tuple(float(x) for x in value["support_smoke"]["initial_capacity_ratio"])
    for path in epsilon_reports:
        value = _load(path)
        receipt = Path(value["receipt"])
        name = receipt.parent.name
        if not name.startswith("cycle-"):
            raise ValueError(f"{path}: cannot recover checkpoint from receipt")
        cycle = int(name.split("-", 1)[1])
        if cycle not in grouped or value["condition_sha256"] != contexts[cycle] or tuple(value["muscle_order"]) != order:
            raise ValueError(f"{path}: report context differs")
        upper = value["epsilon_upper"]
        active = [i for i, item in enumerate(upper) if item is not None]
        if len(active) != 1 or active[0] == tri:
            raise ValueError(f"{path}: expected one non-triceps epsilon bound")
        index = active[0]
        capacity = tuple(float(x) for x in value["support_smoke"]["initial_capacity_ratio"])
        epsilon = float(upper[index])
        rebuild = value["independent_rebuild"]
        if (rebuild["DOP853_relative_work_error"] > maximum_work_error
                or rebuild["DOP853"]["maximum_scaled_endpoint_error"] > maximum_endpoint_error
                or value["support_smoke"].get("epsilon_capacity_max_violation", np.inf) > epsilon_tolerance):
            raise ValueError(f"{path}: registered numerical gate failed")
        grouped[cycle].setdefault(order[index], []).append((epsilon, capacity, str(Path(path).resolve())))
    rows = []
    for cycle, per_muscle in grouped.items():
        for muscle in order:
            if muscle == "Triceps":
                continue
            points = sorted(per_muscle.get(muscle, ()))
            if len(points) < 3 or len({point[0] for point in points}) != len(points):
                raise ValueError(f"c{cycle} {muscle}: need three distinct epsilon cuts")
            baseline = minima[cycle]
            entries = []
            for epsilon, capacity, path in points:
                gain = capacity[tri] - baseline[tri]
                relaxation = baseline[order.index(muscle)] - capacity[order.index(muscle)]
                entries.append({"epsilon_upper": epsilon, "capacity_ratio": list(capacity),
                                "triceps_gain": gain, "relaxed_capacity": relaxation,
                                "exchange_slope": gain / relaxation, "path": path})
            # As the upper bound becomes looser, the minimized triceps capacity
            # must not increase.  This is a numerical consistency check, not a
            # convexity or global Pareto claim.
            monotone = all(b["capacity_ratio"][tri] <= a["capacity_ratio"][tri] + 1e-6
                           for a, b in zip(entries, entries[1:]))
            rows.append({"checkpoint_cycle": cycle, "condition_sha256": contexts[cycle], "muscle": muscle,
                         "minimum_triceps_capacity": baseline[tri], "cuts": entries,
                         "triceps_monotone_with_relaxed_epsilon": monotone})
    return {"schema_version": 1, "kind": "local_epsilon_exchange_replication_summary",
            "muscle_order": list(order), "rows": rows,
            "interpretation": "conditional local exchange curves with frozen fast state/history; not global Pareto curves or an online RHO policy",
            "all_monotonicity_checks_passed": all(row["triceps_monotone_with_relaxed_epsilon"] for row in rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-c120", type=Path, help="Legacy c120 triceps-minimum receipt.")
    parser.add_argument("--minimum-c160", type=Path, help="Legacy c160 triceps-minimum receipt.")
    parser.add_argument("--minimum", action="append", metavar="CYCLE=REPORT",
                        help="Generic triceps-minimum input; repeat as needed.")
    parser.add_argument("--epsilon-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.minimum:
        if args.minimum_c120 or args.minimum_c160:
            parser.error("--minimum cannot be mixed with legacy minimum flags")
        minima = {}
        for item in args.minimum:
            try:
                cycle, path = item.split("=", 1)
                cycle = int(cycle)
            except ValueError:
                parser.error("--minimum must use CYCLE=REPORT")
            if cycle in minima:
                parser.error("Each --minimum cycle must be unique")
            minima[cycle] = Path(path)
    elif args.minimum_c120 and args.minimum_c160:
        minima = {120: args.minimum_c120, 160: args.minimum_c160}
    else:
        parser.error("Supply --minimum CYCLE=REPORT or both legacy c120/c160 minimum reports")
    value = summarize(minimum_reports=minima,
                      epsilon_reports=args.epsilon_report)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
