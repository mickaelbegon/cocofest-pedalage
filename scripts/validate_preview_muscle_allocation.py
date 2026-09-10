"""Compare greedy and 2/3-phase preview muscle allocation on real RHO boundaries.

All policies run outside the RHO NLP, use exact original total-moment targets,
and use no FHO trajectory. Completed compact PW sequences are replayed through
full Ding dynamics. This is numerical policy validation, not evidence of an
endurance improvement.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.compact_muscle_prediction import CompactMusclePredictor
from cocofest.optimization.local_endurance_value import LocalEnduranceCoordinates
from cocofest.optimization.preview_muscle_allocation import PreviewMuscleAllocation
from cocofest.optimization.rho_adaptive_moment_policy import build_rho_adaptive_moment_policy
from cocofest.optimization.rho_rollout_adapter import select_certified_rho_cycle
from scripts.validate_batched_endurance_value import (
    DEFAULT_BASELINE,
    _validate_baseline_case,
    _validate_baseline_report,
)
from scripts.validate_local_endurance_value import (
    DEFAULT_PROFILE,
    DEFAULT_SOURCE,
    _full_ding_replay,
    _source_terminal_states,
)


DEFAULT_OUTPUT = Path(".cache/preview-muscle-allocation-validation")
PREVIEW_DEPTHS = (1, 2, 3)
HORIZONS = (10, 30)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _diagnostic_summary(records):
    records = tuple(records)
    reasons = Counter(record.get("reason", "missing") for record in records)
    qp = [record for record in records if record.get("qp_solved")]
    accepted = [record for record in records if record.get("accepted")]

    def maximum(name):
        values = [record.get(name) for record in records]
        values = [float(value) for value in values if value is not None and np.isfinite(value)]
        return max(values) if values else None

    def total(name):
        values = [record.get(name) for record in records]
        values = [float(value) for value in values if value is not None and np.isfinite(value)]
        return sum(values) if values else None

    return {
        "plan_attempts": len(records),
        "accepted_plans": len(accepted),
        "rejected_plans": len(records) - len(accepted),
        "reason_counts": dict(sorted(reasons.items())),
        "qp_solved_plans": len(qp),
        "optimizer_unsuccessful_but_full_audit_accepted": sum(
            bool(record.get("accepted")) and record.get("qp_solved")
            and not bool(record.get("optimizer_success")) for record in records
        ),
        "greedy_fallback_steps": sum(bool(record.get("greedy_fallback")) for record in records),
        "maximum_plan_time_s": maximum("elapsed_s"),
        "sum_plan_time_s": total("elapsed_s"),
        "maximum_equality_residual_nm": maximum("equality_residual_max_nm"),
        "maximum_bound_violation": maximum("bound_violation_max"),
        "maximum_kkt_stationarity": maximum("kkt_stationarity_max"),
        "maximum_kkt_complementarity": maximum("kkt_complementarity_max"),
        "sum_optimizer_objective": total("optimizer_objective"),
        "sum_moment_deviation_objective": total("moment_deviation_objective"),
        "sum_recruitment_regularization_objective": total("recruitment_regularization_objective"),
    }


def _pw_and_state_audit(result, parameters):
    widths = np.asarray(result.pulse_widths)
    finite = np.isfinite(widths)
    expected = result.completed_intervals * len(parameters)
    lower = np.asarray([parameter.pd0 for parameter in parameters])[None, :, None]
    upper = np.asarray([parameter.pulse_width_max for parameter in parameters])[None, :, None]
    lower_violation = np.maximum(lower - widths, 0.0)
    upper_violation = np.maximum(widths - upper, 0.0)
    history = np.asarray(result.state_history[: result.completed_intervals + 1])
    return {
        "finite_pw_count": int(finite.sum()),
        "expected_finite_pw_count": int(expected),
        "maximum_pw_bound_violation_s": float(
            max(np.nanmax(lower_violation, initial=0.0), np.nanmax(upper_violation, initial=0.0))
        ),
        "minimum_cn": float(np.min(history[:, :, 0])),
        "minimum_force_n": float(np.min(history[:, :, 1])),
        "minimum_capacity": float(np.min(history[:, :, 2])),
        "minimum_tau1_s": float(np.min(history[:, :, 3])),
        "minimum_km": float(np.min(history[:, :, 4])),
    }


def _compact_tracking_audit(result):
    errors = np.asarray(result.signed_moment_errors)
    finite = np.isfinite(errors)
    return {
        "successful_steps": int(finite.sum()),
        "maximum_absolute_original_target_error_nm": (
            float(np.max(np.abs(errors[finite]))) if np.any(finite) else None
        ),
        "signed_error_sum_nm": float(np.sum(errors[finite])),
        "absolute_error_sum_nm": float(np.sum(np.abs(errors[finite]))),
    }


def _full_ding_audit(initial, policy, result, integration_substeps):
    started = perf_counter()
    try:
        history, moments = _full_ding_replay(
            initial, policy, result.pulse_widths, result.completed_intervals, integration_substeps
        )
    except (ValueError, FloatingPointError, OverflowError) as error:
        return {"status": "failed", "elapsed_s": perf_counter() - started, "message": str(error)}, {}
    elapsed = perf_counter() - started
    target = np.asarray(result.original_total_moments).ravel()[: result.completed_intervals]
    moment_error = moments.sum(axis=1) - target
    rest = np.asarray([parameter.fatigue.rest_state for parameter in policy.parameters])
    compact_history = result.state_history[: result.completed_intervals + 1]
    slow_error = (compact_history[:, :, 2:] - history[:, :, 2:]) / rest
    maximum_moment = float(np.max(np.abs(moment_error)))
    maximum_slow = float(np.max(np.abs(slow_error)))
    report = {
        "status": "complete",
        "elapsed_s": elapsed,
        "integration_substeps": integration_substeps,
        "maximum_absolute_original_target_error_nm": maximum_moment,
        "rmse_original_target_error_nm": float(np.sqrt(np.mean(moment_error**2))),
        "maximum_normalized_slow_state_error": maximum_slow,
        "endpoint_numeric_gate_passed": bool(maximum_moment <= 1e-3 and maximum_slow <= 1e-3),
    }
    return report, {
        "full_ding_moment_error_nm": moment_error,
        "maximum_normalized_slow_error_by_node": np.max(np.abs(slow_error), axis=(1, 2)),
        "full_ding_state_history": history,
    }


def _policy_options(args, depth):
    return {
        "preview_phases": depth,
        "moment_tolerance": args.moment_tolerance,
        "max_iterations": args.max_iterations,
        "fallback_to_greedy": False,
        "optimality_tolerance": args.optimality_tolerance,
        "recruitment_regularization": args.recruitment_regularization,
        "max_solve_time_s": args.max_qp_objective_time_s,
    }


def _run_anchor(args, anchor, baseline_case, arrays):
    policy = build_rho_adaptive_moment_policy(
        args.source, args.reduced_profile, cycle_index=anchor, cycle_period=1.0
    )
    cycle, _ = select_certified_rho_cycle(args.source, cycle_index=anchor, cycle_period=1.0)
    initial = _source_terminal_states(cycle, policy.muscle_names)
    coordinates = LocalEnduranceCoordinates.from_state(
        initial, policy.parameters, force_scale=np.maximum(initial[:, 1], 1.0)
    )
    started = perf_counter()
    predictor = CompactMusclePredictor(policy.intervals, policy.parameters, substeps=args.substeps)
    predictor_setup = perf_counter() - started
    moment_scale = float(np.max(np.abs([sum(item.target_moments) for item in policy.intervals])))
    _validate_baseline_case(
        baseline_case, anchor=anchor, horizon=10, coordinates=coordinates,
        moment_scale=moment_scale, substeps=args.substeps,
    )
    policies = {depth: PreviewMuscleAllocation(predictor, **_policy_options(args, depth))
                for depth in PREVIEW_DEPTHS}
    prescreen = {}
    for depth, allocation in policies.items():
        started = perf_counter()
        result = allocation.rollout(initial, horizon_cycles=1)
        elapsed = perf_counter() - started
        projected = elapsed * max(HORIZONS)
        prescreen[depth] = {
            "status": result.status,
            "completed_intervals": result.completed_intervals,
            "elapsed_s": elapsed,
            "linear_projection_to_30_cycles_s": projected,
            "within_runtime_budget": projected <= args.maximum_projected_rollout_time_s,
            "first_failure": result.first_failure,
        }
    cases = []
    for horizon in HORIZONS:
        for depth, allocation in policies.items():
            key = f"anchor{anchor}_h{horizon}_preview{depth}"
            screen = prescreen[depth]
            if not screen["within_runtime_budget"]:
                cases.append(
                    {
                        "case": key,
                        "anchor_zero_based": anchor,
                        "horizon_cycles": horizon,
                        "preview_phases": depth,
                        "status": "prescreen_runtime_stop",
                        "prescreen": screen,
                    }
                )
                continue
            started = perf_counter()
            result = allocation.rollout(initial, horizon_cycles=horizon)
            elapsed = perf_counter() - started
            diagnostic = _diagnostic_summary(result.diagnostics)
            entry = {
                "case": key,
                "anchor_zero_based": anchor,
                "horizon_cycles": horizon,
                "preview_phases": depth,
                "policy_name": "greedy_H1" if depth == 1 else f"receding_preview_{depth}_phase",
                "status": result.status,
                "completed_intervals": result.completed_intervals,
                "completed_cycles": result.completed_cycles,
                "requested_intervals": horizon * len(policy.intervals),
                "first_failure": result.first_failure,
                "rollout_time_s": elapsed,
                "prescreen": screen,
                "qp_solves": result.qp_solves,
                "qp_iterations": result.qp_iterations,
                "qp_failures": result.qp_failures,
                "greedy_fallback_steps": result.greedy_fallback_steps,
                "seed_projected_steps": result.seed_projected_steps,
                "diagnostic_summary": diagnostic,
                "failure_preview_diagnostic": (
                    None if result.status == "complete" else _jsonable(result.diagnostics[-1])
                ),
                "compact_tracking": _compact_tracking_audit(result),
                "pw_and_state_audit": _pw_and_state_audit(result, policy.parameters),
                "full_ding_replay": None,
            }
            arrays[f"{key}__pulse_widths_s"] = result.pulse_widths
            arrays[f"{key}__state_history"] = result.state_history
            arrays[f"{key}__achieved_moments_nm"] = result.achieved_moments
            arrays[f"{key}__original_total_moment_nm"] = result.original_total_moments
            arrays[f"{key}__signed_moment_error_nm"] = result.signed_moment_errors
            arrays[f"{key}__total_lower_bound_nm"] = result.total_lower_bounds
            arrays[f"{key}__total_upper_bound_nm"] = result.total_upper_bounds
            arrays[f"{key}__plan_elapsed_s"] = np.asarray(
                [item.get("elapsed_s", np.nan) for item in result.diagnostics]
            )
            arrays[f"{key}__optimizer_iterations"] = np.asarray(
                [item.get("optimizer_iterations", 0) for item in result.diagnostics]
            )
            if result.status == "complete":
                replay, replay_arrays = _full_ding_audit(
                    initial, policy, result, args.full_ding_substeps
                )
                entry["full_ding_replay"] = replay
                for suffix, values in replay_arrays.items():
                    arrays[f"{key}__{suffix}"] = values
            cases.append(entry)
    return {
        "anchor_zero_based": anchor,
        "predictor_setup_s": predictor_setup,
        "coordinate_context_sha256": coordinates.context_signature,
        "moment_scale_nm": moment_scale,
        "source_terminal_states": initial.tolist(),
        "prescreen_by_preview_depth": {str(key): value for key, value in prescreen.items()},
        "cases": cases,
    }


def _plot(report, output):
    import matplotlib.pyplot as plt

    cases = [case for anchor in report["anchors"] for case in anchor["cases"]
             if case["status"] != "prescreen_runtime_stop"]
    labels = [case["case"].replace("anchor", "a").replace("_preview", " p") for case in cases]
    x = np.arange(len(cases))
    colors = [f"C{case['preview_phases'] - 1}" for case in cases]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].bar(x, [case["completed_intervals"] for case in cases], color=colors)
    axes[0, 0].scatter(x, [case["requested_intervals"] for case in cases], marker="_", color="k")
    axes[0, 0].set(title="Compact policy progress", ylabel="completed phases")
    axes[0, 1].bar(x, [case["rollout_time_s"] for case in cases], color=colors)
    axes[0, 1].set(title="Allocation cost outside NLP", ylabel="rollout time (s)", yscale="log")
    moment = [
        np.nan if case["full_ding_replay"] is None
        else case["full_ding_replay"].get("maximum_absolute_original_target_error_nm", np.nan)
        for case in cases
    ]
    slow = [
        np.nan if case["full_ding_replay"] is None
        else case["full_ding_replay"].get("maximum_normalized_slow_state_error", np.nan)
        for case in cases
    ]
    axes[1, 0].bar(x, np.asarray(moment) * 1000, color=colors)
    axes[1, 0].axhline(1.0, color="k", linestyle=":", label="1 mN·m gate")
    axes[1, 0].set(title="Full Ding endpoint tracking", ylabel="maximum error (mN·m)")
    axes[1, 1].bar(x, np.asarray(slow) * 100, color=colors)
    axes[1, 1].axhline(0.1, color="k", linestyle=":", label="0.1% gate")
    axes[1, 1].set(title="Compact vs full Ding slow states", ylabel="maximum error (% rest)")
    for axis, values in ((axes[1, 0], moment), (axes[1, 1], slow)):
        for index, value in enumerate(values):
            if not np.isfinite(value):
                axis.text(index, 0.025, "NR", ha="center", va="bottom", fontweight="bold")
        axis.text(
            0.99, 0.92, "NR = incomplete trajectory, not replayed",
            transform=axis.transAxes, ha="right", va="top", fontsize=8,
        )
    for axis in axes.ravel():
        axis.set_xticks(x, labels, rotation=55, ha="right", fontsize=7)
        axis.grid(alpha=0.2)
        handles, legend_labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    fig.suptitle("Greedy vs short preview allocation — numerical validation only; no FHO")
    fig.savefig(output / "validation.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reduced-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--full-ding-substeps", type=int, default=16)
    parser.add_argument("--moment-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=60)
    parser.add_argument("--optimality-tolerance", type=float, default=1e-6)
    parser.add_argument("--recruitment-regularization", type=float, default=1e-8)
    parser.add_argument("--max-qp-objective-time-s", type=float, default=0.5)
    parser.add_argument("--maximum-projected-rollout-time-s", type=float, default=15.0)
    args = parser.parse_args(argv)
    baseline = json.loads(args.baseline_report.read_text())
    source_sha = sha256(args.source.read_bytes()).hexdigest()
    profile_sha = sha256(args.reduced_profile.read_bytes()).hexdigest()
    try:
        baseline_cases = _validate_baseline_report(
            baseline,
            source_sha256=source_sha,
            profile_sha256=profile_sha,
            substeps=args.substeps,
            absolute_tolerance=0.002,
            relative_tolerance=0.05,
            ranking_tolerance=1e-5,
        )
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {}
    started = perf_counter()
    report = {
        "schema": "cocofest-preview-muscle-allocation-validation-v1",
        "source": {"path": str(args.source.resolve()), "sha256": source_sha},
        "reduced_profile": {"path": str(args.reduced_profile.resolve()), "sha256": profile_sha},
        "baseline_report": {
            "path": str(args.baseline_report.resolve()),
            "sha256": sha256(args.baseline_report.read_bytes()).hexdigest(),
        },
        "uses_fho_data": False,
        "adds_future_decision_variables_to_rho": 0,
        "controller_validation": "not_run",
        "endurance_improvement": "not_established",
        "settings_predeclared": {
            "preview_phases": list(PREVIEW_DEPTHS),
            "horizon_cycles": list(HORIZONS),
            "fallback_to_greedy": False,
            "substeps": args.substeps,
            "full_ding_substeps": args.full_ding_substeps,
            "moment_tolerance": args.moment_tolerance,
            "max_iterations": args.max_iterations,
            "optimality_tolerance": args.optimality_tolerance,
            "recruitment_regularization": args.recruitment_regularization,
            "max_qp_objective_time_s": args.max_qp_objective_time_s,
            "maximum_linear_prescreen_projection_s": args.maximum_projected_rollout_time_s,
        },
        "numeric_gates": {
            "maximum_full_ding_endpoint_moment_error_nm": 1e-3,
            "maximum_normalized_slow_state_error": 1e-3,
        },
        "timing_protocol": "nonconcurrent one-pass rollouts after one-cycle linear runtime prescreen",
        "anchors": [],
    }
    for anchor in (0, 112):
        result = _run_anchor(args, anchor, baseline_cases[anchor], arrays)
        report["anchors"].append(result)
        print(
            json.dumps(
                {
                    "anchor": anchor,
                    "prescreen": result["prescreen_by_preview_depth"],
                    "cases": [
                        {"case": case["case"], "status": case["status"],
                         "completed": case.get("completed_intervals"),
                         "rollout_time_s": case.get("rollout_time_s")}
                        for case in result["cases"]
                    ],
                }
            ), flush=True,
        )
    report["total_elapsed_s"] = perf_counter() - started
    report["code_sha256"] = {
        name: sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "cocofest/optimization/compact_muscle_prediction.py",
            "cocofest/optimization/preview_muscle_allocation.py",
            "cocofest/optimization/rho_adaptive_moment_policy.py",
            "cocofest/optimization/rho_rollout_adapter.py",
            "scripts/validate_local_endurance_value.py",
            "scripts/validate_batched_endurance_value.py",
            "scripts/validate_preview_muscle_allocation.py",
        )
    }
    report = _jsonable(report)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output / "validation_arrays.npz", **arrays)
    _plot(report, args.output)
    print(f"Report: {args.output / 'report.json'}", flush=True)
    return report


if __name__ == "__main__":
    main()
