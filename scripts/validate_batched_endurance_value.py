"""Audit batched compact endurance values and bounded tracking on real RHO states.

The experiment uses no FHO data and adds no variables to the RHO NLP. Strict
batch results are compared with the scalar implementation and the previously
accepted physical-box fits. Nonzero tracking bands change the predicted task
and are reported separately from numerical and policy feasibility.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.batched_compact_muscle_prediction import BatchedCompactMusclePredictor
from cocofest.optimization.batched_endurance_value import BatchedEnduranceValueOracle
from cocofest.optimization.compact_muscle_prediction import CompactMusclePredictor
from cocofest.optimization.local_endurance_value import (
    CompactEnduranceValueOracle,
    LocalEnduranceCoordinates,
    fit_local_endurance_value,
)
from cocofest.optimization.rho_adaptive_moment_policy import build_rho_adaptive_moment_policy
from cocofest.optimization.rho_rollout_adapter import select_certified_rho_cycle
from scripts.validate_local_endurance_value import (
    DEFAULT_PROFILE,
    DEFAULT_SOURCE,
    _full_ding_replay,
    _source_terminal_states,
)


DEFAULT_BASELINE = Path(".cache/local-endurance-value-validation/physical-box/report.json")
DEFAULT_OUTPUT = Path(".cache/batched-endurance-value-validation")
TRACKING_BANDS_NM = (0.0, 1e-4, 5e-4)


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _finite_max_difference(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if (left.shape != right.shape or np.any(np.isinf(left)) or np.any(np.isinf(right))
            or not np.array_equal(np.isnan(left), np.isnan(right))):
        return None
    finite = np.isfinite(left) & np.isfinite(right)
    return float(np.max(np.abs(left[finite] - right[finite]))) if np.any(finite) else 0.0


def _tracking_metrics(result, intervals, horizon):
    durations = np.tile([interval.duration for interval in intervals], horizon).reshape(
        horizon, len(intervals)
    )
    errors = np.asarray(result.signed_moment_errors)
    mask = np.isfinite(errors)
    return {
        "successful_tracking_steps": int(mask.sum()),
        "relaxed_steps": int(result.relaxed_steps),
        "maximum_absolute_error_nm": float(np.max(np.abs(errors[mask]))) if np.any(mask) else None,
        "signed_error_quadrature_nm_s": float(np.sum(durations[mask] * errors[mask])),
        "absolute_error_quadrature_nm_s": float(np.sum(durations[mask] * np.abs(errors[mask]))),
        "tracking_mode": result.tracking_mode,
    }


def _fit_points(baseline_case):
    accepted = next(attempt for attempt in baseline_case["fit_attempts"] if attempt["accepted"])
    training = np.asarray([record["coordinates"] for record in accepted["training_records"]])
    heldout = np.asarray([record["coordinates"] for record in accepted["heldout_records"]])
    if len(training) != 17 or len(heldout) != 24:
        raise ValueError("Baseline must contain the expected 17 training and 24 held-out points.")
    if any(np.any(np.all(np.isclose(training, point, rtol=0, atol=1e-13), axis=1)) for point in heldout):
        raise ValueError("Baseline held-out points overlap training points.")
    return np.r_[training, heldout], accepted


def _validate_baseline_report(
    baseline, *, source_sha256, profile_sha256, substeps,
    absolute_tolerance, relative_tolerance, ranking_tolerance,
):
    problems = []
    if baseline.get("schema") != "cocofest-local-endurance-value-validation-v1":
        problems.append("schema")
    if baseline.get("uses_fho_data") is not False:
        problems.append("uses_fho_data")
    if baseline.get("source", {}).get("sha256") != source_sha256:
        problems.append("source SHA-256")
    if baseline.get("reduced_profile", {}).get("sha256") != profile_sha256:
        problems.append("reduced-profile SHA-256")
    if baseline.get("fit_protocol", {}).get("kind") != "diagonal_quadratic":
        problems.append("fit kind")
    expected_thresholds = {
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "ranking_tolerance": ranking_tolerance,
    }
    thresholds = baseline.get("thresholds", {})
    for name, expected in expected_thresholds.items():
        actual = thresholds.get(name)
        if actual is None or not np.isclose(actual, expected, rtol=0, atol=1e-15):
            problems.append(name)
    h10 = [case for case in baseline.get("cases", []) if case.get("horizon_cycles") == 10]
    if len(h10) != 2 or sorted(case.get("source_cycle_index_zero_based") for case in h10) != [0, 112]:
        problems.append("unique H10 anchors 0 and 112")
    if any(case.get("predictor_substeps") != substeps for case in h10):
        problems.append("predictor substeps")
    if problems:
        raise ValueError("Incompatible scalar baseline: " + ", ".join(problems) + ".")
    return {case["source_cycle_index_zero_based"]: case for case in h10}


def _validate_baseline_case(baseline_case, *, anchor, horizon, coordinates, moment_scale, substeps):
    problems = []
    expected = {
        "source_cycle_index_zero_based": anchor,
        "horizon_cycles": horizon,
        "predictor_substeps": substeps,
        "fit_kind": "diagonal_quadratic",
        "fit_accepted": True,
        "coordinate_context_sha256": coordinates.context_signature,
    }
    for name, value in expected.items():
        if baseline_case.get(name) != value:
            problems.append(name)
    if not np.isclose(baseline_case.get("moment_scale_nm", np.nan), moment_scale, rtol=0, atol=1e-15):
        problems.append("moment_scale_nm")
    center = np.asarray(baseline_case.get("coordinate_center", []), dtype=float)
    if center.shape != coordinates.anchor.shape or not np.array_equal(center, coordinates.anchor):
        problems.append("coordinate_center")
    if problems:
        raise ValueError(
            f"Incompatible scalar baseline for anchor {anchor}, H{horizon}: "
            + ", ".join(problems) + "."
        )


def _context(source, profile, anchor_index, substeps):
    policy = build_rho_adaptive_moment_policy(
        source, profile, cycle_index=anchor_index, cycle_period=1.0
    )
    cycle, _ = select_certified_rho_cycle(source, cycle_index=anchor_index, cycle_period=1.0)
    initial = _source_terminal_states(cycle, policy.muscle_names)
    coordinates = LocalEnduranceCoordinates.from_state(
        initial, policy.parameters, force_scale=np.maximum(initial[:, 1], 1.0)
    )
    start = perf_counter()
    predictor = CompactMusclePredictor(policy.intervals, policy.parameters, substeps=substeps)
    predictor_setup = perf_counter() - start
    start = perf_counter()
    batch_predictor = BatchedCompactMusclePredictor(predictor)
    batch_wrapper_setup = perf_counter() - start
    moment_scale = float(
        np.max(np.abs([sum(interval.target_moments) for interval in policy.intervals]))
    )
    return policy, initial, coordinates, predictor, batch_predictor, moment_scale, {
        "compact_predictor_setup_s": predictor_setup,
        "batch_wrapper_setup_s": batch_wrapper_setup,
    }


def _strict_anchor_case(args, baseline_case, arrays):
    anchor = baseline_case["source_cycle_index_zero_based"]
    horizon = 10
    policy, initial, coordinates, predictor, batch_predictor, moment_scale, setup = _context(
        args.source, args.reduced_profile, anchor, args.substeps
    )
    _validate_baseline_case(
        baseline_case,
        anchor=anchor,
        horizon=horizon,
        coordinates=coordinates,
        moment_scale=moment_scale,
        substeps=args.substeps,
    )
    scalar_oracle = CompactEnduranceValueOracle(
        predictor, coordinates, horizon_cycles=horizon, moment_scale=moment_scale
    )
    batch_oracle = BatchedEnduranceValueOracle(
        predictor, coordinates, horizon_cycles=horizon, moment_scale=moment_scale,
        tracking_band_nm=0.0, tracking_penalty_weight=1.0,
    )
    points, accepted_baseline = _fit_points(baseline_case)

    # Fixed timing order and a single pass per backend; parity rollouts below
    # are explicitly outside this timing comparison.
    batch_start = perf_counter()
    batch_values = batch_oracle.evaluate_many(points)
    batch_value_time = perf_counter() - batch_start
    scalar_start = perf_counter()
    scalar_values = tuple(scalar_oracle.evaluate(point) for point in points)
    scalar_value_time = perf_counter() - scalar_start
    value_status_parity = [a.status for a in batch_values] == [a.status for a in scalar_values]
    value_differences = [
        abs(a.value - b.value)
        for a, b in zip(batch_values, scalar_values)
        if a.value is not None and b.value is not None
    ]

    selected_indices = (0, 1, 8, 17, 40)
    selected_states = np.asarray([coordinates.decode(points[index]) for index in selected_indices])
    batched_rollouts = batch_predictor.rollout_many(selected_states, horizon_cycles=horizon)
    scalar_rollouts = tuple(predictor.rollout(state, horizon_cycles=horizon) for state in selected_states)
    rollout_records = []
    for point_index, batch, scalar in zip(selected_indices, batched_rollouts, scalar_rollouts):
        rollout_records.append(
            {
                "point_index": point_index,
                "status_equal": batch.status == scalar.status,
                "completed_intervals_equal": batch.completed_intervals == scalar.completed_intervals,
                "maximum_state_difference": _finite_max_difference(batch.state_history, scalar.state_history),
                "maximum_pw_difference_s": _finite_max_difference(batch.pulse_widths, scalar.pulse_widths),
                "maximum_achieved_moment_difference_nm": _finite_max_difference(
                    batch.achieved_moments, scalar.achieved_moments
                ),
                "maximum_allocated_moment_difference_nm": _finite_max_difference(
                    batch.allocated_moments, scalar.allocated_moments
                ),
            }
        )

    fit_records = []
    accepted_fit = None
    for baseline_attempt in baseline_case["fit_attempts"]:
        radius = np.asarray(baseline_attempt["trust_radius"])
        start = perf_counter()
        fit = fit_local_endurance_value(
            batch_oracle,
            trust_radius=radius,
            kind="diagonal_quadratic",
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
            ranking_tolerance=args.ranking_tolerance,
        )
        elapsed = perf_counter() - start
        fit_records.append(
            {
                "baseline_accepted": baseline_attempt["accepted"],
                "batch_accepted": fit.accepted,
                "acceptance_equal": fit.accepted == baseline_attempt["accepted"],
                "elapsed_s": elapsed,
                "baseline_elapsed_s": baseline_attempt["elapsed_s"],
                "speedup_vs_recorded_scalar": baseline_attempt["elapsed_s"] / elapsed,
                "trust_radius": radius.tolist(),
                "maximum_absolute_error": fit.audit.maximum_absolute_error,
                "maximum_scaled_error": fit.audit.maximum_scaled_error,
                "ranking_pairs": fit.audit.ranking_pairs,
                "ranking_failures": fit.audit.ranking_failures,
            }
        )
        if fit.accepted:
            accepted_fit = fit
            break

    coefficient_difference = None
    if accepted_fit is not None:
        coefficient_difference = float(
            np.max(
                np.abs(
                    accepted_fit.model.coefficients
                    - np.asarray(baseline_case["model"]["coefficients"])
                )
            )
        )
    key = f"anchor{anchor}_h10"
    arrays[f"{key}__strict_points"] = points
    arrays[f"{key}__batch_values"] = np.asarray(
        [np.nan if value.value is None else value.value for value in batch_values]
    )
    arrays[f"{key}__scalar_values"] = np.asarray(
        [np.nan if value.value is None else value.value for value in scalar_values]
    )
    return {
        "case": key,
        "anchor_zero_based": anchor,
        "horizon_cycles": horizon,
        "setup_timings": setup,
        "timed_point_count": len(points),
        "timing_order": "batch_once_then_scalar_once",
        "batch_value_time_s": batch_value_time,
        "sequential_scalar_value_time_s": scalar_value_time,
        "value_evaluation_speedup": scalar_value_time / batch_value_time,
        "value_status_parity": value_status_parity,
        "maximum_value_difference": max(value_differences, default=0.0),
        "rollout_parity": rollout_records,
        "fit_attempts": fit_records,
        "fit_acceptance_sequence_equal": all(item["acceptance_equal"] for item in fit_records),
        "accepted_model_coefficient_maximum_difference": coefficient_difference,
        "batch_accepted": accepted_fit is not None,
        "accepted_ranking_pairs": None if accepted_fit is None else accepted_fit.audit.ranking_pairs,
        "accepted_ranking_failures": None if accepted_fit is None else accepted_fit.audit.ranking_failures,
        "value_context": batch_oracle.value_context_metadata,
    }


def _band_case(args, arrays):
    anchor, horizon = 112, 30
    policy, initial, coordinates, predictor, batch_predictor, moment_scale, setup = _context(
        args.source, args.reduced_profile, anchor, args.substeps
    )
    cases = []
    for band in TRACKING_BANDS_NM:
        start = perf_counter()
        rollout = batch_predictor.rollout_many(
            initial[None], horizon_cycles=horizon, tracking_band_nm=band
        )[0]
        elapsed = perf_counter() - start
        metrics = _tracking_metrics(rollout, policy.intervals, horizon)
        failure_context = None
        if rollout.status != "complete" and rollout.first_failure is not None:
            cycle = rollout.first_failure["cycle_index"]
            interval = rollout.first_failure["interval_index"]
            original = rollout.original_total_moments[cycle, interval]
            lower = rollout.total_lower_bounds[cycle, interval]
            upper = rollout.total_upper_bounds[cycle, interval]
            failure_context = {
                "cycle_index_zero_based": cycle,
                "interval_index_zero_based": interval,
                "original_target_nm": original,
                "lower_bound_nm": lower,
                "upper_bound_nm": upper,
                "original_minus_lower_nm": original - lower,
                "upper_minus_original_nm": upper - original,
            }
        oracle = BatchedEnduranceValueOracle(
            predictor,
            coordinates,
            horizon_cycles=horizon,
            moment_scale=moment_scale,
            tracking_band_nm=band,
            tracking_penalty_weight=1.0,
        )
        oracle_result = oracle.evaluate(coordinates.anchor)
        entry = {
            "tracking_band_nm": band,
            "rollout_status": rollout.status,
            "completed_intervals": rollout.completed_intervals,
            "requested_intervals": horizon * len(policy.intervals),
            "rollout_time_s": elapsed,
            "policy_failure": rollout.first_failure,
            "failure_context": failure_context,
            "tracking": metrics,
            "oracle": _jsonable(oracle_result),
            "strict_feasible": bool(band == 0 and rollout.status == "complete"),
            "band_policy_feasible": bool(band > 0 and rollout.status == "complete"),
            "full_ding_replay": None,
        }
        label = f"band_{band:.0e}".replace("-", "m")
        for suffix, values in (
            ("signed_error_nm", rollout.signed_moment_errors),
            ("original_target_nm", rollout.original_total_moments),
            ("effective_target_nm", rollout.effective_total_targets),
            ("relaxed_step_mask", rollout.relaxed_step_mask),
            ("lower_bound_nm", rollout.total_lower_bounds),
            ("upper_bound_nm", rollout.total_upper_bounds),
        ):
            arrays[f"anchor112_h30__{label}__{suffix}"] = values
        if rollout.status == "complete":
            replay_start = perf_counter()
            try:
                history, moments = _full_ding_replay(
                    initial,
                    policy,
                    rollout.pulse_widths,
                    rollout.completed_intervals,
                    args.full_ding_substeps,
                )
                replay_elapsed = perf_counter() - replay_start
                target = rollout.original_total_moments.ravel()
                full_error = moments.sum(axis=1) - target
                durations = np.tile([interval.duration for interval in policy.intervals], horizon)
                full_metrics = {
                    "status": "complete",
                    "elapsed_s": replay_elapsed,
                    "integration_substeps": args.full_ding_substeps,
                    "maximum_absolute_error_nm": float(np.max(np.abs(full_error))),
                    "signed_error_quadrature_nm_s": float(durations @ full_error),
                    "absolute_error_quadrature_nm_s": float(durations @ np.abs(full_error)),
                    "within_1e-3_nm_endpoint_gate": bool(np.max(np.abs(full_error)) <= 1e-3),
                }
                entry["full_ding_replay"] = full_metrics
                arrays[f"anchor112_h30__{label}__full_ding_error_nm"] = full_error
                arrays[f"anchor112_h30__{label}__full_ding_state_history"] = history
            except (ValueError, FloatingPointError, OverflowError) as error:
                entry["full_ding_replay"] = {
                    "status": "failed",
                    "elapsed_s": perf_counter() - replay_start,
                    "message": str(error),
                }
        cases.append(entry)
    return {"case": "anchor112_h30_tracking_band_screen", "setup_timings": setup, "bands": cases}


def _plot(report, arrays, output):
    import matplotlib.pyplot as plt

    strict = report["strict_batch_scalar"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for case in strict:
        key = case["case"]
        axes[0, 0].scatter(
            arrays[f"{key}__scalar_values"], arrays[f"{key}__batch_values"],
            s=22, alpha=0.8, label=key,
        )
    values = np.concatenate(
        [arrays[f"{case['case']}__scalar_values"] for case in strict]
        + [arrays[f"{case['case']}__batch_values"] for case in strict]
    )
    lo, hi = np.nanmin(values), np.nanmax(values)
    axes[0, 0].plot([lo, hi], [lo, hi], "k:")
    axes[0, 0].set(title="Strict value parity (41 fixed points)", xlabel="Scalar", ylabel="Batch")

    x = np.arange(len(strict))
    width = 0.36
    axes[0, 1].bar(
        x - width / 2, [case["sequential_scalar_value_time_s"] for case in strict],
        width, label="sequential scalar",
    )
    axes[0, 1].bar(
        x + width / 2, [case["batch_value_time_s"] for case in strict],
        width, label="batch",
    )
    axes[0, 1].set(
        title="One-pass value timing", xticks=x,
        xticklabels=[case["case"] for case in strict], ylabel="seconds",
    )

    for case in strict:
        axes[1, 0].plot(
            range(1, len(case["fit_attempts"]) + 1),
            [attempt["elapsed_s"] for attempt in case["fit_attempts"]],
            marker="o", label=f"batch {case['case']}",
        )
        axes[1, 0].plot(
            range(1, len(case["fit_attempts"]) + 1),
            [attempt["baseline_elapsed_s"] for attempt in case["fit_attempts"]],
            marker="x", linestyle=":", label=f"scalar baseline {case['case']}",
        )
    axes[1, 0].set(title="Fit attempts at identical radii", xlabel="attempt", ylabel="seconds")

    band_cases = report["tracking_band_screen"]["bands"]
    labels = [f"{case['tracking_band_nm']:.0e}" for case in band_cases]
    completed = [case["completed_intervals"] for case in band_cases]
    bars = axes[1, 1].bar(labels, completed)
    for bar, case in zip(bars, band_cases):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4,
            f"relaxed={case['tracking']['relaxed_steps']}",
            ha="center", fontsize=8,
        )
    axes[1, 1].axhline(900, color="k", linestyle=":", label="requested 900 phases")
    axes[1, 1].set(
        title="H30 policy progress; band changes the task",
        xlabel="tracking band (N·m)", ylabel="completed phases", ylim=(0, 970),
    )
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
        handles, labels_ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    fig.suptitle("Batched local endurance value — numerical validation only; no FHO")
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
    parser.add_argument("--absolute-tolerance", type=float, default=0.002)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--ranking-tolerance", type=float, default=1e-5)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(args.baseline_report.read_text())
    source_sha256 = sha256(args.source.read_bytes()).hexdigest()
    profile_sha256 = sha256(args.reduced_profile.read_bytes()).hexdigest()
    try:
        baseline_cases = _validate_baseline_report(
            baseline,
            source_sha256=source_sha256,
            profile_sha256=profile_sha256,
            substeps=args.substeps,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
            ranking_tolerance=args.ranking_tolerance,
        )
    except ValueError as error:
        parser.error(str(error))
    arrays = {}
    start = perf_counter()
    report = {
        "schema": "cocofest-batched-endurance-value-validation-v1",
        "source": {"path": str(args.source.resolve()), "sha256": source_sha256},
        "reduced_profile": {
            "path": str(args.reduced_profile.resolve()),
            "sha256": profile_sha256,
        },
        "baseline_report": {
            "path": str(args.baseline_report.resolve()),
            "sha256": sha256(args.baseline_report.read_bytes()).hexdigest(),
        },
        "uses_fho_data": False,
        "adds_future_decision_variables_to_rho": 0,
        "controller_validation": "not_run",
        "endurance_improvement": "not_established",
        "timing_protocol": "single fixed-order calls; no concurrent timing and no repeated median",
        "tracking_bands_nm_predeclared": list(TRACKING_BANDS_NM),
        "tracking_penalty_weight": 1.0,
        "strict_batch_scalar": [],
    }
    for anchor in (0, 112):
        case = _strict_anchor_case(args, baseline_cases[anchor], arrays)
        report["strict_batch_scalar"].append(case)
        print(
            json.dumps(
                {
                    "case": case["case"],
                    "value_speedup": case["value_evaluation_speedup"],
                    "max_value_difference": case["maximum_value_difference"],
                    "fit_sequence_equal": case["fit_acceptance_sequence_equal"],
                }
            ), flush=True,
        )
    report["tracking_band_screen"] = _band_case(args, arrays)
    report["total_elapsed_s"] = perf_counter() - start
    report["code_sha256"] = {
        name: sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "cocofest/optimization/compact_muscle_prediction.py",
            "cocofest/optimization/batched_compact_muscle_prediction.py",
            "cocofest/optimization/local_endurance_value.py",
            "cocofest/optimization/batched_endurance_value.py",
            "cocofest/optimization/rho_adaptive_moment_policy.py",
            "cocofest/optimization/rho_rollout_adapter.py",
            "scripts/validate_local_endurance_value.py",
            "scripts/validate_batched_endurance_value.py",
        )
    }
    report = _jsonable(report)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output / "validation_arrays.npz", **arrays)
    _plot(report, arrays, args.output)
    print(
        json.dumps(
            {
                "tracking_bands": [
                    {
                        "band": case["tracking_band_nm"],
                        "status": case["rollout_status"],
                        "completed": case["completed_intervals"],
                        "relaxed": case["tracking"]["relaxed_steps"],
                    }
                    for case in report["tracking_band_screen"]["bands"]
                ]
            }
        ), flush=True,
    )
    print(f"Report: {args.output / 'report.json'}", flush=True)
    return report


if __name__ == "__main__":
    main()
