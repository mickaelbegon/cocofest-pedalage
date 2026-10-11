"""Compare frozen-input interpreted and native-callback local Ding runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(interpreted: Path, compiled: Path) -> dict:
    paths = {"interpreted": interpreted, "compiled": compiled}
    metrics = {name: json.loads((path / "symbolic-audit/nlp0_metrics.json").read_text())
               for name, path in paths.items()}
    count = len(metrics["interpreted"])
    if count < 2 or len(metrics["compiled"]) != count:
        raise ValueError("Both runs need the same number of calls, including hot calls")
    inputs = ("x0", "lbx", "ubx", "lbg", "ubg", "lam_x0", "lam_g0")
    errors = {key: 0.0 for key in (*inputs, "x", "g", "lam_x", "lam_g")}
    for call in range(count):
        with np.load(interpreted / f"symbolic-audit/nlp0_call{call}.npz") as a, \
                np.load(compiled / f"symbolic-audit/nlp0_call{call}.npz") as b:
            if set(a.files) != set(b.files):
                raise ValueError(f"Recorded field mismatch at call {call}")
            for field in errors:
                if field in a:
                    errors[field] = max(errors[field], float(np.max(np.abs(a[field] - b[field]), initial=0)))
    if any(errors[field] != 0 for field in inputs):
        raise ValueError("Original-space inputs differ; this is not a frozen A/B")

    def cohort(name):
        rows, hot = metrics[name], metrics[name][1:]
        result = json.loads((paths[name] / "result.json").read_text())["results"][0]
        return {
            "count": len(rows), "success_count": sum(r["success"] is True for r in rows),
            "hot_ipopt_median_s": float(np.median([r["timings"]["t_wall_total"] for r in hot])),
            "hot_ipopt_p90_s": float(np.percentile([r["timings"]["t_wall_total"] for r in hot], 90)),
            "hot_hessian_total_median_s": float(np.median([r["timings"]["t_wall_nlp_hess_l"] for r in hot])),
            "hot_iterations_median": float(np.median([r["iterations"] for r in hot])),
            "hot_complete_iteration_median_s": result["hot_complete_iteration_wall_time_median_s"],
            "hot_complete_iteration_p90_s": result["hot_complete_iteration_wall_time_p90_s"],
            "max_full_nlp_violation": max(r["original_max_bound_constraint_violation"] for r in rows),
            "max_discrete_certificate": max(r["discrete_equivalence_certificate_max"] for r in rows),
            "solver_nx": rows[0]["solver_nx"], "solver_ng": rows[0]["solver_ng"],
        }

    a, b = cohort("interpreted"), cohort("compiled")
    return {
        "protocol": "identical frozen original-space IPOPT inputs; local Ding reconstruction; native Hessian after substitution",
        "interpreted_directory": str(interpreted.resolve()), "compiled_directory": str(compiled.resolve()),
        "interpreted": a, "compiled": b,
        "native_compilation": json.loads((compiled / "symbolic-audit/nlp0_compilation.json").read_text()),
        "hot_solver_median_relative_change": b["hot_ipopt_median_s"] / a["hot_ipopt_median_s"] - 1,
        "hot_complete_iteration_median_relative_change": b["hot_complete_iteration_median_s"] / a["hot_complete_iteration_median_s"] - 1,
        "max_absolute_errors": errors,
        "max_candidate_objective_abs_difference": max(abs(x["objective"] - y["objective"])
                                                         for x, y in zip(metrics["interpreted"], metrics["compiled"])),
        "different_iteration_counts": sum(x["iterations"] != y["iterations"]
                                           for x, y in zip(metrics["interpreted"], metrics["compiled"])),
        "limitation": "Outer RHO consumes the recorded baseline trajectory; this is not independently propagated compiled-controller endurance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interpreted", type=Path)
    parser.add_argument("compiled", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.interpreted, args.compiled)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "native_compilation"}, indent=2))


if __name__ == "__main__":
    main()
