"""Summarize current-code closed-loop local Ding A/B; no precision claims."""
import json
from pathlib import Path
import sys

import numpy as np


def summarize(root, windows=10):
    output = {}
    for mode in ("baseline", "local"):
        folder = root / f"{mode}{windows}" / "symbolic-audit"
        metrics = json.loads((folder / "nlp0_metrics.json").read_text())
        hot = metrics[1:]
        times = [m["timings"]["t_wall_total"] for m in hot]
        def total(key):
            return sum(m["timings"].get(key, 0) for m in hot)
        evaluations = sum(total("t_wall_"+f) for f in ("nlp_f", "nlp_g", "nlp_grad_f", "nlp_jac_g", "nlp_hess_l"))
        output[mode] = dict(
            windows=len(metrics), successes=sum(bool(m["success"]) for m in metrics),
            solver_nx=metrics[0]["solver_nx"], solver_ng=metrics[0]["solver_ng"], nnz=metrics[0]["nnz"],
            first_s=metrics[0]["timings"]["t_wall_total"], first_iterations=metrics[0]["iterations"],
            median_hot_s=float(np.median(times)), p90_hot_s=float(np.quantile(times,.9)),
            hot_iterations=[m["iterations"] for m in hot],
            mean_hot_iterations=float(np.mean([m["iterations"] for m in hot])),
            hot_evaluations_s=evaluations, hot_outside_evaluations_s=sum(times)-evaluations,
            hot_per_iteration_s=sum(times)/sum(m["iterations"] for m in hot),
            hot_hessian_per_eval_s=total("t_wall_nlp_hess_l")/total("n_call_nlp_hess_l"),
            hot_jacobian_per_eval_s=total("t_wall_nlp_jac_g")/total("n_call_nlp_jac_g"),
            max_original_violation=max(m["original_max_bound_constraint_violation"] for m in metrics),
            objective_sum=sum(m["objective"] for m in metrics),
            certificate_max=max((m.get("discrete_equivalence_certificate_max") or 0) for m in metrics),
            mapping_s=metrics[0]["mapping_s"], solver_build_s=metrics[0]["solver_build_s"],
            hot_mapping_median_s=float(np.median([m["input_mapping_s"] for m in hot])))
    trajectories = []
    for i in range(windows):
        a,b = [np.load(root/f"{mode}{windows}"/"symbolic-audit"/f"nlp0_call{i}.npz")
               for mode in ("baseline","local")]
        trajectories.append(dict(window=i, max_scaled_x_difference=float(np.max(np.abs(a["x"]-b["x"]))),
            identical_initial_guess=bool(np.array_equal(a["x0"],b["x0"])),
            identical_bounds=all(np.array_equal(a[k],b[k]) for k in ("lbx","ubx","lbg","ubg"))))
    output["comparison"] = dict(
        median_ratio=output["local"]["median_hot_s"]/output["baseline"]["median_hot_s"],
        max_scaled_x_difference=max(t["max_scaled_x_difference"] for t in trajectories),
        objective_sum_ratio=output["local"]["objective_sum"]/output["baseline"]["objective_sum"],
        per_window=trajectories,
        limitation="Closed-loop trajectories diverge; this is not a frozen-window replay.")
    return output


if __name__ == "__main__":
    root = Path(sys.argv[1])
    output = summarize(root)
    (root/"comparison10.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2))
