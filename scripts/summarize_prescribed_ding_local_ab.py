"""Summarize a frozen-input unilateral prescribed-kinematics Ding A/B.

The local run returns recorded baseline solutions to the outer RHO loop. Its
native candidate, full-NLP audit, and timing are in ``symbolic-audit``; the
outer trajectory is intentionally identical and is not evidence of candidate
PW equivalence.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


def control_indices(first_x: np.ndarray, pattern: dict, scale: float) -> np.ndarray:
    """Identify controls from the saved, named first-cycle PW trace.

    The full NLP export has no control-name metadata. Require every one of the
    30 known PW values to match an equally spaced index sequence; this fails
    closed if Bioptim changes its NLP ordering.
    """
    indices = []
    for name, muscle in pattern["muscles"].items():
        expected = np.asarray(muscle["pulse_width_s"], dtype=float) / scale
        if len(expected) < 3:
            raise ValueError(f"Too few PW samples for {name}")
        starts = np.flatnonzero(np.isclose(first_x, expected[0], rtol=1e-10, atol=1e-11))
        seconds = np.flatnonzero(np.isclose(first_x, expected[1], rtol=1e-10, atol=1e-11))
        matched = None
        for start in starts:
            for second in seconds[seconds > start]:
                candidate = start + (second - start) * np.arange(len(expected))
                if candidate[-1] < len(first_x) and np.allclose(
                    first_x[candidate], expected, rtol=1e-10, atol=1e-11
                ):
                    matched = candidate
                    break
            if matched is not None:
                break
        if matched is None:
            raise ValueError(f"Could not identify all PW controls for {name}")
        indices.extend(matched)
    if len(indices) != len(set(indices)):
        raise ValueError("PW controls overlap between muscles")
    return np.asarray(indices, dtype=int)


def summarize(root: Path) -> dict:
    paths = {mode: root / mode / "symbolic-audit" for mode in ("baseline", "local")}
    metrics = {mode: json.loads((path / "nlp0_metrics.json").read_text())
               for mode, path in paths.items()}
    count = len(metrics["baseline"])
    if count == 0 or len(metrics["local"]) != count:
        raise ValueError("Baseline/local call counts differ or are empty")
    fields = ("x0", "lbx", "ubx", "lbg", "ubg", "lam_x0", "lam_g0")
    max_input_error = 0.0
    max_solution_error = 0.0
    baseline_result = json.loads((root / "baseline" / "result.json").read_text())
    scale = baseline_result["configurations"]["ipopt"]["pulse_width_scaling"]
    pattern = baseline_result["results"][0]["stimulation_patterns"]["cycle_1"]
    with np.load(paths["baseline"] / "nlp0_call0.npz") as first:
        pw_indices = control_indices(first["x"], pattern, scale)
    max_candidate_pw_error_us = 0.0
    for call in range(count):
        with np.load(paths["baseline"] / f"nlp0_call{call}.npz") as b, \
             np.load(paths["local"] / f"nlp0_call{call}.npz") as l:
            if set(b.files) != set(l.files):
                raise ValueError(f"Archived fields differ at call {call}")
            for key in fields:
                if key in b:
                    max_input_error = max(max_input_error,
                                          float(np.max(np.abs(b[key] - l[key]), initial=0)))
            max_solution_error = max(max_solution_error,
                                     float(np.max(np.abs(b["x"] - l["x"]), initial=0)))
            max_candidate_pw_error_us = max(max_candidate_pw_error_us,
                float(1e6 * scale * np.max(np.abs(b["x"][pw_indices] - l["x"][pw_indices]), initial=0)))
    if max_input_error != 0:
        raise ValueError(f"Frozen NLP input mismatch: {max_input_error}")

    def cohort(values: list[dict]) -> dict:
        hot = values[1:]
        return {
            "count": len(values),
            "success_count": sum(m["success"] is True for m in values),
            "original_nx": values[0]["original_nx"],
            "original_ng": values[0]["original_ng"],
            "solver_nx": values[0]["solver_nx"],
            "solver_ng": values[0]["solver_ng"],
            "mapping_s": values[0]["mapping_s"],
            "solver_build_s": values[0]["solver_build_s"],
            "hot_ipopt_median_s": float(np.median([m["timings"]["t_wall_total"] for m in hot])) if hot else None,
            "hot_ipopt_p90_s": float(np.percentile([m["timings"]["t_wall_total"] for m in hot], 90)) if hot else None,
            "hot_adapter_wall_median_s": float(np.median([m["call_wall_s"] for m in hot])) if hot else None,
            "hot_iterations_median": float(np.median([m["iterations"] for m in hot])) if hot else None,
            "max_full_nlp_violation": max(m["original_max_bound_constraint_violation"] for m in values),
            "max_discrete_certificate": max((m["discrete_equivalence_certificate_max"] or 0)
                                            for m in values),
        }

    baseline, local = metrics["baseline"], metrics["local"]
    bstats, lstats = cohort(baseline), cohort(local)
    for mode, stats in (("baseline", bstats), ("local", lstats)):
        result = json.loads((root / mode / "result.json").read_text())["results"][0]
        stats["validated_outer_cycles"] = result["validated_cycles"]
        stats["hot_complete_iteration_median_s"] = result["hot_complete_iteration_wall_time_median_s"]
        stats["hot_complete_iteration_p90_s"] = result["hot_complete_iteration_wall_time_p90_s"]
    btime, ltime = bstats["hot_ipopt_median_s"], lstats["hot_ipopt_median_s"]
    return {
        "protocol": "unilateral prescribed isokinetic K5-I1; 4 Ding muscles; unit fatigue; 30 Hz; Radau-5 SX; IPOPT/MA57; one BLAS thread; baseline states frozen per call",
        "baseline": bstats, "local": lstats,
        "hot_ipopt_median_speedup": (btime / ltime) if btime and ltime else None,
        "max_frozen_input_abs_difference": max_input_error,
        "max_candidate_full_x_abs_difference": max_solution_error,
        "max_candidate_pw_abs_difference_us": max_candidate_pw_error_us,
        "max_candidate_objective_abs_difference": max(abs(b["objective"] - l["objective"])
                                                  for b, l in zip(baseline, local)),
        "mean_candidate_objective_relative_difference": float(np.mean([
            (l["objective"] - b["objective"]) / max(abs(b["objective"]), 1e-12)
            for b, l in zip(baseline, local)])),
        "candidate_pw_note": "Control indices were identified from all 30 named first-cycle PW samples per muscle and checked against the original-space NLP archive.",
        "outer_trajectory_note": "Local returns recorded baseline solutions to RHO solely to preserve the next frozen window; candidate solutions and objective differ.",
    }


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    report = summarize(root)
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
