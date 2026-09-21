"""Summarize sequential real-cycling baseline/condensed audits as JSON."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(root, windows):
    modes = {}
    values = {}
    for mode in ("baseline", "condensed"):
        folder = root / f"{mode}{windows}"
        metrics = []
        vectors = []
        for metric_path in sorted((folder / "symbolic-audit").glob("nlp*_metrics.json")):
            entries = json.loads(metric_path.read_text())
            for entry in entries:
                metrics.append(entry)
                vectors.append(np.load(metric_path.parent / f"nlp{entry['nlp_index']}_call{entry['call']}.npz"))
        if len(metrics) != windows:
            raise ValueError(f"{mode}: expected {windows} windows, found {len(metrics)}")
        times = np.array([m["timings"]["t_wall_total"] for m in metrics])
        iterations = np.array([m["iterations"] for m in metrics])
        modes[mode] = {
            "successful_windows": sum(m["success"] for m in metrics),
            "original_nx": metrics[0]["original_nx"], "solver_nx": metrics[0]["solver_nx"],
            "original_ng": metrics[0]["original_ng"], "solver_ng": metrics[0]["solver_ng"],
            "condensation_s": metrics[0]["condensation_s"],
            "solver_build_s": metrics[0]["solver_build_s"],
            "first_solve_s": float(times[0]), "total_solve_s": float(sum(times)),
            "hot_median_s": float(np.median(times[1:])) if windows > 1 else None,
            "hot_p90_s": float(np.percentile(times[1:], 90)) if windows > 1 else None,
            "hot_mean_iterations": float(np.mean(iterations[1:])) if windows > 1 else None,
            "times_s": times.tolist(), "iterations": iterations.tolist(),
            "max_original_violation": max(m["original_max_bound_constraint_violation"] for m in metrics),
            "mean_hessian_evaluation_s": sum(m["timings"]["t_wall_nlp_hess_l"] for m in metrics)
                / sum(m["timings"]["n_call_nlp_hess_l"] for m in metrics),
            "mean_jacobian_evaluation_s": sum(m["timings"]["t_wall_nlp_jac_g"] for m in metrics)
                / sum(m["timings"]["n_call_nlp_jac_g"] for m in metrics),
        }
        values[mode] = (metrics, vectors)
    am, av = values["baseline"]
    bm, bv = values["condensed"]
    comparison = {
        "same_initial_seed": bool(np.array_equal(av[0]["x0"], bv[0]["x0"])),
        "max_scaled_decision_difference": max(float(np.max(np.abs(a["x"] - b["x"]))) for a, b in zip(av, bv)),
        "max_original_constraint_difference": max(float(np.max(np.abs(a["g"] - b["g"]))) for a, b in zip(av, bv)),
        "max_objective_difference": max(abs(a["objective"] - b["objective"]) for a, b in zip(am, bm)),
        "per_window_scaled_decision_difference": [float(np.max(np.abs(a["x"] - b["x"]))) for a, b in zip(av, bv)],
        "total_solve_ratio_condensed_over_baseline": modes["condensed"]["total_solve_s"] / modes["baseline"]["total_solve_s"],
        "hot_median_ratio_condensed_over_baseline": modes["condensed"]["hot_median_s"] / modes["baseline"]["hot_median_s"]
            if windows > 1 else None,
    }
    return {"windows": windows, "modes": modes, "comparison": comparison}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(summarize(args.root, args.windows), indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report)
