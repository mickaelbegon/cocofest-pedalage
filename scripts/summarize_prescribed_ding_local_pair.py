"""Summarize a synchronized two-process K5-I1 local Ding benchmark."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("No values")
    position = (len(ordered) - 1) * fraction
    low = int(position)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)


def read_side(root: Path, side: str):
    result = json.loads((root / side / "result.json").read_text())["results"][0]
    metrics = json.loads((root / side / "symbolic-audit" / "nlp0_metrics.json").read_text())
    timing = [json.loads(line) for line in (root / side / "symbolic-audit" / "pair_timing.jsonl").read_text().splitlines()]
    if len(metrics) != len(timing) or len(metrics) != result["validated_cycles"]:
        raise ValueError(f"Incomplete {side} result/metric/timing records")
    return result, metrics, timing


def summarize(root: Path):
    (start, end, left_status, right_status, left_cpu, right_cpu) = (
        (root / "process-timing.txt").read_text().split()
    )
    if left_status != "0" or right_status != "0":
        raise ValueError("A paired worker failed")
    data = {side: read_side(root, side) for side in ("left", "right")}
    cycles = min(len(data[side][2]) for side in data)
    side_summary = {}
    for side, (result, metrics, timing) in data.items():
        audits = result["isokinetic_audits"]
        side_summary[side] = {
            "validated_cycles": result["validated_cycles"],
            "physical_success": result["physical_success"],
            "isokinetic_scientific_success": result["isokinetic_scientific_success"],
            "hot_ipopt_median_s": statistics.median(m["timings"]["t_wall_total"] for m in metrics[1:]),
            "hot_complete_iteration_median_s": result["hot_complete_iteration_wall_time_median_s"],
            "end_to_end_wall_s": result["end_to_end_wall_time_s"],
            "max_original_nlp_violation": max(m["original_max_bound_constraint_violation"] for m in metrics),
            "max_radau_certificate": max(m["discrete_equivalence_certificate_max"] for m in metrics),
            "max_work_target_error_j": max(abs(a["certification_energy_target_error_j"]) for a in audits),
            "max_dop853_work_target_error_j": max(abs(a["high_accuracy_energy_target_error_j"]) for a in audits),
            "max_load_torque_bound_violation_nm": max(a["maximum_load_torque_bound_violation_nm"] for a in audits),
            "executed_fatigue_objective": result["executed_fatigue_objective"],
            "min_A_capacity_ratio": result["min_A_capacity_ratio"],
            "all_ipopt_success": all(m["success"] for m in metrics),
            "all_work_audits_pass": all(a["passes_tolerance"] for a in audits),
            "cpu": int(left_cpu if side == "left" else right_cpu),
        }
    if cycles != side_summary["left"]["validated_cycles"] or cycles != side_summary["right"]["validated_cycles"]:
        raise ValueError("Paired sides did not certify the same number of cycles")
    pair_latencies = []
    release_skews = []
    for index in range(cycles):
        left = data["left"][2][index]
        right = data["right"][2][index]
        if left["cycle"] != index or right["cycle"] != index:
            raise ValueError("Paired cycle indices disagree")
        pair_latencies.append((max(left["done_ns"], right["done_ns"]) -
                               min(left["release_ns"], right["release_ns"])) / 1e9)
        release_skews.append(abs(left["release_ns"] - right["release_ns"]) / 1e3)
    hot = pair_latencies[1:]
    return {
        "protocol": "two identical unilateral prescribed isokinetic K5-I1 models, independent compiled local Ding process per CPU; file barrier before each NLP; each candidate propagates its own state",
        "cycles": cycles,
        "process_wall_s": (int(end) - int(start)) / 1e9,
        "pair_nlp_adapter_wall_median_s": statistics.median(hot),
        "pair_nlp_adapter_wall_p90_s": percentile(hot, .9),
        "pair_nlp_adapter_wall_max_s": max(pair_latencies),
        "barrier_release_skew_median_us": statistics.median(release_skews),
        "barrier_release_skew_max_us": max(release_skews),
        "sides": side_summary,
        "scope_note": "Barrier aligns native NLP calls; pair latency includes local mapping, solve, reconstruction, audit and archive write, but excludes inter-window control preparation and final export. Identical side models establish compute latency only, not a heterogeneous bilateral physiology claim.",
    }


def main():
    root = Path(sys.argv[1])
    summary = summarize(root)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
