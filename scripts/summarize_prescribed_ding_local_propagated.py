"""Compare independently propagated full and compiled-local K5-I1 RHO runs."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

import numpy as np

from scripts.summarize_prescribed_ding_local_ab import control_indices


def read_run(root: Path):
    result = json.loads((root / "result.json").read_text())
    metrics = json.loads((root / "symbolic-audit" / "nlp0_metrics.json").read_text())
    return result, result["results"][0], metrics


def summarize(full_root: Path, local_root: Path):
    full_json, full, full_metrics = read_run(full_root)
    local_json, local, local_metrics = read_run(local_root)
    count = len(full_metrics)
    if len(local_metrics) != count or count < 2:
        raise ValueError("Call counts differ or are too short")
    scale = full_json["configurations"]["ipopt"]["pulse_width_scaling"]
    first = full["stimulation_patterns"]["cycle_1"]
    with np.load(full_root / "symbolic-audit" / "nlp0_call0.npz") as record:
        pw_indices = control_indices(record["x"], first, scale)
    pw_differences = []
    for call in range(count):
        with np.load(full_root / "symbolic-audit" / f"nlp0_call{call}.npz") as left, \
             np.load(local_root / "symbolic-audit" / f"nlp0_call{call}.npz") as right:
            pw_differences.extend((1e6 * scale * np.abs(left["x"][pw_indices] -
                right["x"][pw_indices])).tolist())
    state_errors = {}
    for full_audit, local_audit in zip(full["isokinetic_audits"], local["isokinetic_audits"]):
        full_state = full_audit["high_accuracy_rollout"]["final_reference_state"]
        local_state = local_audit["high_accuracy_rollout"]["final_reference_state"]
        if full_state.keys() != local_state.keys():
            raise ValueError("DOP853 state keys differ")
        for key in full_state:
            difference = abs(float(full_state[key][0]) - float(local_state[key][0]))
            state_errors[key] = max(state_errors.get(key, 0.0), difference)
    def cohort(result, metrics):
        audits = result["isokinetic_audits"]
        return {
            "validated_cycles": result["validated_cycles"],
            "physical_success": result["physical_success"],
            "isokinetic_scientific_success": result["isokinetic_scientific_success"],
            "all_ipopt_success": all(m["success"] for m in metrics),
            "all_work_audits_pass": all(a["passes_tolerance"] for a in audits),
            "hot_ipopt_median_s": statistics.median(m["timings"]["t_wall_total"] for m in metrics[1:]),
            "hot_complete_iteration_median_s": result["hot_complete_iteration_wall_time_median_s"],
            "hot_complete_iteration_p90_s": result["hot_complete_iteration_wall_time_p90_s"],
            "rho_solve_loop_wall_s": result["wall_time_s"],
            "end_to_end_wall_s": result["end_to_end_wall_time_s"],
            "max_original_nlp_violation": max(m["original_max_bound_constraint_violation"] for m in metrics),
            "max_radau_certificate": max((m["discrete_equivalence_certificate_max"] or 0) for m in metrics),
            "max_collocation_work_target_error_j": max(abs(a["certification_energy_target_error_j"]) for a in audits),
            "max_dop853_work_target_error_j": max(abs(a["high_accuracy_energy_target_error_j"]) for a in audits),
            "max_dop853_scaled_endpoint_error": max(a["high_accuracy_rollout"]["maximum_scaled_endpoint_error"] for a in audits),
            "max_dop853_cn_endpoint_error": max(a["high_accuracy_rollout"]["maximum_absolute_endpoint_error_by_state"]["Cn_Biceps"] for a in audits),
            "max_dop853_triceps_force_endpoint_error_n": max(a["high_accuracy_rollout"]["maximum_absolute_endpoint_error_by_state"]["F_Triceps"] for a in audits),
            "max_load_torque_bound_violation_nm": max(a["maximum_load_torque_bound_violation_nm"] for a in audits),
            "executed_fatigue_objective": result["executed_fatigue_objective"],
            "fatigue_auc_cycles": result["fatigue_auc_cycles"],
            "min_A_capacity_ratio": result["min_A_capacity_ratio"],
        }
    full_work = [a["high_accuracy_energy_j"] for a in full["isokinetic_audits"]]
    local_work = [a["high_accuracy_energy_j"] for a in local["isokinetic_audits"]]
    return {
        "protocol": "K5-I1 unilateral prescribed isokinetic 30 Hz Radau-5 unit-fatigue RHO; original full and compiled local Ding separately propagate their own candidates for 100 cycles",
        "full": cohort(full, full_metrics),
        "compiled_local": cohort(local, local_metrics),
        "max_pw_difference_us": max(pw_differences),
        "median_pw_difference_us": statistics.median(pw_differences),
        "max_dop853_work_difference_j_per_cycle": max(abs(x-y) for x,y in zip(full_work, local_work)),
        "max_dop853_boundary_state_difference_by_key": state_errors,
        "interpretation": "Candidates can converge to distinct local minima; numerical feasibility and closed-loop fatigue are compared without claiming solution identity. An IPOPT failure, if one occurs, is a solver outcome and not a physiological infeasibility certificate.",
    }


def main():
    full_root, local_root, output = map(Path, sys.argv[1:4])
    summary = summarize(full_root, local_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
