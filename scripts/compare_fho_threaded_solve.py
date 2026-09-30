"""Compare bounded FHO native-Hessian solve reports and saved numeric probes."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", type=Path)
    parser.add_argument("threaded", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [json.loads((path / "report.json").read_text()) for path in (args.serial, args.threaded)]
    if any(report["stats"]["n_call_nlp_hess_l"] <= 0 for report in reports):
        raise ValueError("Cannot compare runs without Hessian evaluations")
    if any(report["certificate_valid"] for report in reports):
        raise ValueError("Expected uncertified bounded benchmark reports")
    a, b = reports
    initial = [np.load(path / "initial-probe.npz") for path in (args.serial, args.threaded)]
    final = [np.load(path / "final-probe-uncertified.npz") for path in (args.serial, args.threaded)]

    def sorted_rows(probe):
        rows = np.column_stack([probe[k].ravel() for k in ("lbg", "ubg", "g0")])
        return rows[np.lexsort(rows.T[::-1])]

    preparation = [r["application_preparation_s"] + r["nlpsol_construction_s"] for r in reports]
    solve_saving = a["solve_elapsed_s"] - b["solve_elapsed_s"]
    iterations = a["stats"]["iter_count"]
    comparison = {
        "paths": [str(args.serial), str(args.threaded)],
        "initial_x_identical": bool(np.array_equal(initial[0]["x0"], initial[1]["x0"])),
        "initial_constraint_bound_triplets_identical_after_sort": bool(np.array_equal(
            sorted_rows(initial[0]), sorted_rows(initial[1]))),
        "initial_objective_identical": a["initial_objective"] == b["initial_objective"],
        "variable_bounds_identical": all(a[k] == b[k] for k in ("lbx_sha256", "ubx_sha256")),
        "hessian_sparsity_identical": a["hessian_sparsity_sha256"] == b["hessian_sparsity_sha256"],
        "structure_dimensions_identical": all(a[k] == b[k] for k in ("nx", "ng", "hessian_nnz_upper", "jacobian_nnz")),
        "ipopt_objective_history_identical": a["stats"]["iterations"]["obj"] == b["stats"]["iterations"]["obj"],
        "final_x_identical": bool(np.array_equal(final[0]["x"], final[1]["x"])),
        "final_x_max_absolute_difference": float(np.max(np.abs(final[0]["x"] - final[1]["x"]))),
        "preparation_plus_nlpsol_s": preparation,
        "solve_wall_s": [r["solve_elapsed_s"] for r in reports],
        "total_wall_s": [r["benchmark_elapsed_s"] for r in reports],
        "solve_time_reduction_fraction": 1 - b["solve_elapsed_s"] / a["solve_elapsed_s"],
        "total_time_reduction_fraction": 1 - b["benchmark_elapsed_s"] / a["benchmark_elapsed_s"],
        "hessian_time_reduction_fraction": 1 - b["stats"]["t_wall_nlp_hess_l"] / a["stats"]["t_wall_nlp_hess_l"],
        "jacobian_time_reduction_fraction": 1 - b["stats"]["t_wall_nlp_jac_g"] / a["stats"]["t_wall_nlp_jac_g"],
        "linear_projection_break_even_iterations": ((preparation[1] - preparation[0]) / (solve_saving / iterations)
                                                     if solve_saving > 0 else None),
        "projection_note": "Budget-fixed, local linear extrapolation only; no convergence-time claim.",
        "certified": False,
    }
    content = json.dumps(comparison, indent=2)
    print(content)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(content + "\n")


if __name__ == "__main__":
    main()
