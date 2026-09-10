"""Recalculate source physiological weights for wide, explicitly named cases.

This is a source-calibration sensitivity experiment, not a closed-loop
RHO/FHO endurance comparison. Published and current-repository parameter
families remain separate. Invalid constant-force fatigue experiments are
reported, never repaired by clipping or reused as controller weights.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.physiological_muscle_weights import (
    SOURCE_COMMIT, calculate_physiological_muscle_weights,
)
from cocofest.optimization.physiological_weight_cases import (
    MUSCLE_NAMES, OFAT_FACTORS, list_cases, nominal_baselines,
)
from scripts.physiological_weight_geometry import build_geometry


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _case_report(case, result, geometry_audit, elapsed):
    raw = result["raw_weights"]
    other_minimum = min(value for name, value in raw.items() if name != "Triceps")
    admissible = result["usable_for_controller"]
    return {
        "case": asdict(case), "parameters": case.as_parameter_dict(),
        "status": result["status"], "domain_valid": result["domain_valid"],
        "calibration_admissible": admissible,
        "actual_rho_controller_validated": False,
        "weights": result["normalized_weights"],
        "legacy_weights_audit_only": result["legacy_normalized_weights"],
        "raw_weights": raw,
        "fatigability": result["fatigability"],
        "mechanical_contribution": result["mechanical_contribution"],
        "pre_risk_contribution": result["support_in_pre_risk"],
        "unique_contribution": result["unique_support"],
        "duty_cycles": result["duty_cycles"],
        "minimum_capacity_ratio": float(min(np.min(result["ratios"]), np.min(result["active_ratios"]))),
        "first_invalid": result["first_invalid"],
        "triceps_raw_gap_to_other_minimum": raw["Triceps"] - other_minimum,
        "triceps_positive_weight": bool(admissible and result["normalized_weights"]["Triceps"] > 0.),
        "raw_minimum_muscles": [name for name, value in raw.items() if value == min(raw.values())],
        "risk_fraction_first_last": [float(result["risk_mask"][i].mean()) for i in (0, -1)],
        "weight_calculation_s": elapsed, "geometry_audit": geometry_audit,
        "weights_recomputed_for_this_case": True, "context": result["context"],
    }


def _comparison_candidates(records):
    """Select clinically legible panel representatives, never by an FHO result."""
    valid = [record for record in records if record["calibration_admissible"]]
    positive = max((record for record in valid if record["triceps_positive_weight"]),
                   key=lambda record: record["weights"]["Triceps"], default=None)
    boundary = min(valid, key=lambda record: record["minimum_capacity_ratio"], default=None)
    return {
        "nominal": records[0]["case"]["case_id"],
        "largest_admissible_triceps_weight": None if positive is None else positive["case"]["case_id"],
        "smallest_positive_calibration_capacity_margin": None if boundary is None else boundary["case"]["case_id"],
        "selection_uses_fho": False,
        "must_validate_actual_rho_before_fho_comparison": True,
    }


def _plot_panel(records, output, baseline_id):
    import matplotlib.pyplot as plt
    names = MUSCLE_NAMES
    matrix = np.asarray([[np.nan if row["weights"] is None else row["weights"][name]
                          for name in names] for row in records])
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), constrained_layout=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#c6c6c6")
    plot = axes[0].imshow(np.ma.masked_invalid(matrix.T), aspect="auto", vmin=0, vmax=1, cmap=cmap)
    axes[0].set_yticks(range(4), names)
    axes[0].set_ylabel("Muscle")
    fig.colorbar(plot, ax=axes[0], label="Recalculated weight / maximum raw weight")
    axes[1].plot([row["minimum_capacity_ratio"] for row in records], ".-", markersize=3)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set(xlabel="Declared case index (0 = nominal)", ylabel="Minimum A / A_rest")
    axes[1].grid(alpha=.2)
    fig.suptitle(f"{baseline_id}: independent weights per case; gray = inadmissible calibration")
    fig.savefig(output / f"{baseline_id}_panel.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for column, field in enumerate(("alpha_a", "tau_fat", "a_scale", "Fmax")):
        selected = [records[0]] + [record for record in records
            if record["case"]["control_family"] == "one_factor_at_a_time"
            and tuple(record["case"]["changed_fields"]) == (f"Triceps.{field}",)]
        selected.sort(key=lambda row: row["case"]["factor"])
        for name in names:
            axes[0, column].plot([row["case"]["factor"] for row in selected],
                [np.nan if row["weights"] is None else row["weights"][name] for row in selected], "o-", label=name)
        axes[1, column].plot([row["case"]["factor"] for row in selected],
                            [row["mechanical_contribution"]["Triceps"] if row["calibration_admissible"] else np.nan
                             for row in selected], "o-", label="Criticality")
        for row in range(2):
            axes[row, column].set_xscale("log", base=2)
            axes[row, column].set_xlim(.23, 4.3)
            axes[row, column].set_xticks([.25, .5, 1., 2., 4.], ["0.25", "0.5", "1", "2", "4"])
            axes[row, column].grid(alpha=.2)
            axes[row, column].set_xlabel(f"Triceps {field} multiplier")
        axes[0, column].set_ylim(-.03, 1.03)
        if not any(row["calibration_admissible"] for row in selected):
            axes[0, column].text(.5, .5, "No admissible calibration", transform=axes[0, column].transAxes,
                                 ha="center", va="center", fontsize=9)
    axes[0, 0].set_ylabel("Admissible weights / maximum raw weight")
    axes[1, 0].set_ylabel("Triceps mechanical contribution")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle(f"{baseline_id}: wide triceps OFAT sensitivity (no weight floor)")
    fig.savefig(output / f"{baseline_id}_triceps.png", dpi=150)
    plt.close(fig)


def _plot_positive_triceps(records, arrays, theta, output):
    import matplotlib.pyplot as plt
    changed = max((record for record in records if record["triceps_positive_weight"]),
                  key=lambda record: record["weights"]["Triceps"], default=None)
    if changed is None:
        return
    nominal = records[0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    x = np.arange(len(MUSCLE_NAMES))
    for offset, record, label in ((-.18, nominal, "Nominal"), (.18, changed, "Largest positive-triceps case")):
        for axis, key in zip(axes[:2], ("weights", "mechanical_contribution")):
            axis.bar(x + offset, [record[key][name] for name in MUSCLE_NAMES], width=.36, label=label)
            axis.set_xticks(x, MUSCLE_NAMES, rotation=25)
        case_id = record["case"]["case_id"]
        profiles = arrays[f"{case_id}__torque_profiles"]
        ratios = arrays[f"{case_id}__capacity_ratios"][:, -1]
        total_positive = np.maximum(profiles * ratios[:, None], 0).sum(axis=0)
        axes[2].plot(np.degrees(theta), total_positive, label=label)
    axes[0].set(title="Recalculated weights", ylabel="Weight / maximum raw weight")
    axes[0].annotate(f"Triceps = {changed['weights']['Triceps']:.5f}",
                     xy=(3.18, changed["weights"]["Triceps"]), xytext=(1.6, .45),
                     arrowprops={"arrowstyle": "->"})
    axes[1].set(title="Mechanical contribution", ylabel="Positive support integral")
    axes[2].axhline(.2, color="k", linestyle=":", label="Source risk threshold")
    axes[2].set(title="After 1500 prescribed-force cycles", xlabel="Sorted angle (degrees)", ylabel="Sum of positive torques (Nm)")
    axes[2].legend(fontsize=8)
    fig.suptitle(changed["case"]["name"] + " — calibration sensitivity, not an endurance result")
    fig.savefig(output / f"{changed['case']['baseline_id']}_positive_triceps.png", dpi=150)
    plt.close(fig)


def _plot_reference(result, theta, profiles, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for axis, key, title in zip(axes[:3], ("fatigability", "mechanical_contribution", "normalized_weights"),
                               ("Fatigability", "Mechanical criticality", "Weight / maximum raw weight")):
        axis.bar(MUSCLE_NAMES, [result[key][name] for name in MUSCLE_NAMES])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
    for name, profile in zip(MUSCLE_NAMES, profiles):
        axes[3].plot(np.degrees(theta), profile, label=name)
    axes[3].axhline(0, color="k", linewidth=.6)
    axes[3].set(title="Source single-active torque", xlabel="Sorted crank angle (degrees)", ylabel="Nm")
    axes[3].legend(fontsize=7)
    fig.suptitle("Published-code nominal reference — formula reproduction, not closed-loop validation")
    fig.savefig(output / "published_reference.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/physiological-weight-sensitivity")
    parser.add_argument("--n-shooting", type=int, default=120)
    parser.add_argument("--target-cycles", type=int, default=1500)
    parser.add_argument("--refinement-shooting", type=int, default=240)
    parser.add_argument("--baselines", nargs="+", choices=[b.baseline_id for b in nominal_baselines()],
                        default=[b.baseline_id for b in nominal_baselines()])
    args = parser.parse_args(argv)
    if args.target_cycles < 2 or len(set(args.baselines)) != len(args.baselines):
        parser.error("target-cycles must be >=2 and baselines must be distinct")
    if args.refinement_shooting and args.refinement_shooting <= args.n_shooting:
        parser.error("refinement-shooting must exceed n-shooting, or be zero to disable")
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    geometry = build_geometry(MUSCLE_NAMES, n_shooting=args.n_shooting)
    report = {
        "schema": "physiological-weight-sensitivity-v1", "source_commit": SOURCE_COMMIT,
        "uses_fho_data": False, "public_rho_modified": False,
        "actual_controller_comparison_run": False,
        "settings": {"target_cycles": args.target_cycles, "rho": .8, "pre_risk_width_deg": 90.,
                     "risk_threshold_nm": .2, "normalization": "max", "factors": OFAT_FACTORS,
                     "case_families_frozen_before_results": True},
        "geometry": geometry.metadata, "panels": [],
    }
    arrays = {"theta": geometry.theta, "q": geometry.q, "qdot": geometry.qdot}
    nominal_reference = None
    for baseline_id in args.baselines:
        records = []
        for index, case in enumerate(list_cases(baseline_id)):
            profiles, profile_audit = geometry.profiles(case.as_parameter_dict())
            step_start = perf_counter()
            result = calculate_physiological_muscle_weights(
                geometry.theta, profiles, case.as_parameter_dict(), muscle_names=MUSCLE_NAMES,
                case_id=case.case_id, target_cycles=args.target_cycles,
            )
            records.append(_case_report(case, result, profile_audit, perf_counter() - step_start))
            arrays[f"{case.case_id}__capacity_ratios"] = result["ratios"]
            arrays[f"{case.case_id}__active_capacity_ratios"] = result["active_ratios"]
            arrays[f"{case.case_id}__torque_profiles"] = profiles
            if baseline_id == "published_code" and index == 0:
                nominal_reference = result
                _plot_reference(result, geometry.theta, profiles, args.output)
        valid = [row for row in records if row["calibration_admissible"]]
        panel = {"baseline_id": baseline_id, "case_count": len(records),
                 "admissible_case_count": len(valid),
                 "positive_triceps_case_count": sum(row["triceps_positive_weight"] for row in records),
                 "comparison_candidates": _comparison_candidates(records), "cases": records}
        report["panels"].append(panel)
        print(json.dumps({key: panel[key] for key in ("baseline_id", "case_count", "admissible_case_count",
                         "positive_triceps_case_count", "comparison_candidates")}), flush=True)
        _plot_panel(records, args.output, baseline_id)
        _plot_positive_triceps(records, arrays, geometry.theta, args.output)
    if nominal_reference is not None and args.target_cycles == 1500 and args.n_shooting == 120:
        expected = np.asarray([1., .0943, .389, 0.])
        actual = np.asarray([nominal_reference["legacy_normalized_weights"][name] for name in MUSCLE_NAMES])
        report["published_reference_check"] = {
            "actual_weights": actual, "rounded_article_weights": expected,
            "within_publication_rounding": bool(np.all(np.abs(actual - expected) <= [1e-12, 5e-5, 5e-4, 1e-12])),
            "not_a_claim_of_exact_marker_geometry_or_controller_feasibility": True,
        }
    if args.refinement_shooting:
        refined = build_geometry(MUSCLE_NAMES, n_shooting=args.refinement_shooting)
        refinement = {"geometry": refined.metadata, "cases": [],
                      "purpose": "angular_sampling_sensitivity_not_controller_validation"}
        for panel in report["panels"]:
            chosen = set(value for value in panel["comparison_candidates"].values() if isinstance(value, str))
            for row in panel["cases"]:
                if row["case"]["case_id"] not in chosen or not row["calibration_admissible"]:
                    continue
                profiles, _ = refined.profiles(row["parameters"])
                result = calculate_physiological_muscle_weights(
                    refined.theta, profiles, row["parameters"], muscle_names=MUSCLE_NAMES,
                    case_id=row["case"]["case_id"], target_cycles=args.target_cycles,
                )
                weights = result["normalized_weights"]
                refinement["cases"].append({
                    "case_id": row["case"]["case_id"], "coarse_weights": row["weights"],
                    "refined_status": result["status"], "refined_weights": weights,
                    "maximum_absolute_weight_change": None if weights is None else
                        max(abs(weights[name] - row["weights"][name]) for name in MUSCLE_NAMES),
                    "triceps_positive_on_refined_grid": bool(weights is not None and weights["Triceps"] > 0.),
                    "minimum_refined_capacity_ratio": float(min(np.min(result["ratios"]), np.min(result["active_ratios"]))),
                })
        report["angular_refinement"] = refinement
    report["elapsed_s_including_geometry_plots_and_cases"] = perf_counter() - started
    paths = ["scripts/physiological_weight_geometry.py", "scripts/validate_physiological_weights.py",
             "cocofest/optimization/physiological_muscle_weights.py", "cocofest/optimization/physiological_weight_cases.py"]
    report["code_sha256"] = {path: sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}
    (args.output / "report.json").write_text(json.dumps(report, default=_jsonable, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    print(f"Report: {args.output / 'report.json'}", flush=True)
    return report


if __name__ == "__main__":
    main()
