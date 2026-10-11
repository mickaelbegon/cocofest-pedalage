#!/usr/bin/env python3
"""Summarize pre-registered capacity-support witnesses without overstating a front."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text())
    if value.get("status") != "fresh_discrete_rebuild_and_DOP853_completed_not_global_front_certified":
        raise ValueError(f"{path}: independent rebuild/DOP853 status is incomplete")
    return value


def summarize(paths, *, maximum_relative_work_error: float, maximum_scaled_endpoint_error: float,
              maximum_epsilon_capacity_violation: float = 1e-6,
              dominance_tolerance: float = 1e-5) -> dict:
    if not all(np.isfinite(value) and value >= 0 for value in
               (maximum_relative_work_error, maximum_scaled_endpoint_error,
                maximum_epsilon_capacity_violation, dominance_tolerance)):
        raise ValueError("All thresholds must be finite and non-negative")
    rows, seen = [], set()
    context = None
    for path in map(Path, paths):
        value = _load(path)
        if str(path.resolve()) in seen:
            raise ValueError("Duplicate report path")
        seen.add(str(path.resolve()))
        if context is None:
            context = (value["condition_sha256"], tuple(value["muscle_order"]))
        elif context != (value["condition_sha256"], tuple(value["muscle_order"])):
            raise ValueError("Pilot reports mix conditioned physical contexts or muscle order")
        support_result = value["support_smoke"]
        rebuild = value["independent_rebuild"]
        capacity = np.asarray(support_result["initial_capacity_ratio"], float)
        if capacity.shape != (len(context[1]),) or not np.all(np.isfinite(capacity)):
            raise ValueError("Invalid capacity vector")
        work_error = rebuild.get("DOP853_relative_work_error")
        endpoint_error = rebuild.get("DOP853", {}).get("maximum_scaled_endpoint_error")
        epsilon_violation = support_result.get("epsilon_capacity_max_violation", 0.)
        numeric_gate = bool(rebuild.get("discrete_audit_passed") is True
                            and rebuild.get("continuous_DOP853_replay") == "complete"
                            and np.isfinite(work_error) and np.isfinite(endpoint_error)
                            and np.isfinite(epsilon_violation)
                            and work_error <= maximum_relative_work_error
                            and endpoint_error <= maximum_scaled_endpoint_error
                            and epsilon_violation <= maximum_epsilon_capacity_violation)
        support = value.get("support")
        support_inferred_uniform = support is None
        if support_inferred_uniform:
            # The first uniform smoke predates the explicit support field.
            # Its filename and protocol document the uniform support; retain
            # that provenance rather than silently dropping a validated run.
            support = (np.ones(len(context[1])) / len(context[1])).tolist()
        rows.append({"id": path.stem, "path": str(path.resolve()),
                     "sha256": sha256(path.read_bytes()).hexdigest(), "support": support,
                     "support_inferred_uniform_from_legacy_smoke": support_inferred_uniform,
                     "capacity_ratio": capacity.tolist(), "support_value": support_result["support_value"],
                     "numeric_gate_passed": numeric_gate, "DOP853_relative_work_error": work_error,
                     "DOP853_scaled_endpoint_error": endpoint_error,
                     "epsilon_upper": value.get("epsilon_upper"),
                     "epsilon_capacity_max_violation": epsilon_violation,
                     "discrete_audit": rebuild.get("discrete_audit")})
    if not rows:
        raise ValueError("At least one report is required")
    accepted = [row for row in rows if row["numeric_gate_passed"]]
    for candidate in accepted:
        vector = np.asarray(candidate["capacity_ratio"])
        candidate["provisionally_dominated_by"] = []
        for other in accepted:
            if other is candidate:
                continue
            comparison = np.asarray(other["capacity_ratio"])
            if np.all(comparison <= vector + dominance_tolerance) and np.any(comparison < vector - dominance_tolerance):
                candidate["provisionally_dominated_by"].append(other["id"])
        candidate["provisional_nondominated"] = not candidate["provisionally_dominated_by"]
    return {"schema_version": 2, "kind": "provisional_local_capacity_support_pilot_summary",
            "condition_sha256": context[0], "muscle_order": list(context[1]),
            "numeric_gates": {"maximum_relative_DOP853_work_error": maximum_relative_work_error,
                              "maximum_scaled_DOP853_endpoint_error": maximum_scaled_endpoint_error,
                              "maximum_epsilon_capacity_violation": maximum_epsilon_capacity_violation,
                              "discrete_audit_required": True, "dominance_tolerance": dominance_tolerance},
            "reports": rows, "accepted_count": len(accepted),
            "provisional_nondominated_count": sum(row.get("provisional_nondominated", False) for row in accepted),
            "interpretation": "support witnesses that pass numerical gates; not a global Pareto proof and not an online RHO policy",
            "remaining_gates": ["epsilon_constraint_coverage_of_non-supported_front_branches",
                                "additional_multistart_or_globality_evidence",
                                "independent_replication_at_other_fast_state_conditions",
                                "holdout_interpolation_and_RHO_ablation"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--maximum-relative-work-error", type=float, required=True)
    parser.add_argument("--maximum-scaled-endpoint-error", type=float, required=True)
    parser.add_argument("--maximum-epsilon-capacity-violation", type=float, default=1e-6)
    parser.add_argument("--dominance-tolerance", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    output = args.pop("output").resolve()
    report = summarize(args.pop("report"), **args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
