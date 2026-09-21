"""Recompute Test 0 from archived symbolic A/B native CasADi timings.

No solves or formulation changes. The residual timer is total minus all
instrumented NLP evaluation timers; it is not a MA57-only measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median


def analyze(path):
    raw = path.read_bytes()
    entries = json.loads(raw)
    hot = [entry for entry in entries if entry["call"] > 0]
    if not hot or not all(entry["success"] for entry in hot):
        raise ValueError("Expected successful hot windows after call 0")
    rows = []
    for entry in hot:
        timings = entry["timings"]
        row = {"call": entry["call"], "iterations": entry["iterations"]}
        for clock in ("wall", "proc"):
            prefix = f"t_{clock}_"
            callbacks = {key[len(prefix):]: value for key, value in timings.items()
                         if key.startswith(prefix + "nlp_")}
            total = timings[prefix + "total"]
            evaluation = sum(callbacks.values())
            if not 0 <= evaluation <= total:
                raise ValueError("Inconsistent total and evaluation timers")
            row[clock] = {"total_s": total, "nlp_evaluation_s": evaluation,
                          "outside_evaluations_s": total - evaluation,
                          "evaluation_fraction": evaluation / total,
                          "total_per_iteration_s": total / entry["iterations"],
                          "callbacks_s": callbacks}
        rows.append(row)
    summary = {
        "hot_windows": len(rows),
        "iterations": [r["iterations"] for r in rows],
        "mean_iterations": mean(r["iterations"] for r in rows),
        "median_iterations": median(r["iterations"] for r in rows),
        "variables": hot[0]["solver_nx"], "constraints": hot[0]["solver_ng"],
    }
    for clock in ("wall", "proc"):
        total = sum(r[clock]["total_s"] for r in rows)
        evaluation = sum(r[clock]["nlp_evaluation_s"] for r in rows)
        fraction = evaluation / total
        summary[clock] = {
            "median_total_s": median(r[clock]["total_s"] for r in rows),
            "total_s": total, "nlp_evaluation_s": evaluation,
            "outside_evaluations_s": total - evaluation,
            "evaluation_fraction": fraction,
            "weighted_total_per_iteration_s": total / sum(r["iterations"] for r in rows),
            "median_total_per_iteration_s": median(r[clock]["total_per_iteration_s"] for r in rows),
            "callback_fractions": {
                name: sum(r[clock]["callbacks_s"][name] for r in rows) / total
                for name in rows[0][clock]["callbacks_s"]
            },
            "fixed_iteration_speedup_if_evaluations_4x": 1 / (1 - fraction + fraction / 4),
            "fixed_iteration_speedup_if_evaluations_free": 1 / (1 - fraction),
        }
    return {"source": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "method": "sum all native t_{wall,proc}_nlp_* timers; residual=total-sum; exclude call0",
            "summary": summary, "windows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("ding-radau5-symbolic-ab-20260912"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = {mode: analyze(args.root / f"{mode}10/symbolic-audit/nlp0_metrics.json")
               for mode in ("baseline", "condensed")}
    a, b = [reports[mode]["summary"] for mode in ("baseline", "condensed")]
    reports["comparison"] = {
        "hot_mean_iterations_ratio": b["mean_iterations"] / a["mean_iterations"],
        "hot_weighted_per_iteration_ratio": b["wall"]["weighted_total_per_iteration_s"] / a["wall"]["weighted_total_per_iteration_s"],
        "hot_median_total_ratio": b["wall"]["median_total_s"] / a["wall"]["median_total_s"],
        "hot_total_ratio": b["wall"]["total_s"] / a["wall"]["total_s"],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps({mode: report["summary"] for mode, report in reports.items() if mode != "comparison"}, indent=2))
    print(json.dumps(reports["comparison"], indent=2))
