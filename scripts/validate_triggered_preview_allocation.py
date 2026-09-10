"""Validate margin-triggered two-phase muscle allocation on real RHO boundaries.

The trigger and all preview QPs run outside the RHO NLP.  The experiment uses
only the archived RHO trajectory and its reduced cycling profile, never FHO
data.  Results are numerical policy and local-fit checks, not evidence of an
endurance or clinical improvement.
"""

from __future__ import annotations

import argparse
from collections import Counter
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

from cocofest.optimization.compact_muscle_prediction import CompactMusclePredictor
from cocofest.optimization.local_endurance_value import (
    CompactEnduranceValueOracle,
    LocalEnduranceCoordinates,
    fit_local_endurance_value,
)
from cocofest.optimization.preview_endurance_value import PreviewEnduranceValueOracle
from cocofest.optimization.preview_muscle_allocation import PreviewMuscleAllocation
from cocofest.optimization.rho_adaptive_moment_policy import build_rho_adaptive_moment_policy
from cocofest.optimization.rho_rollout_adapter import select_certified_rho_cycle
from cocofest.optimization.triggered_preview_allocation import (
    TriggeredPreviewMuscleAllocation,
)
from scripts.validate_batched_endurance_value import (
    DEFAULT_BASELINE as DEFAULT_LOCAL_BASELINE,
    _validate_baseline_case,
    _validate_baseline_report,
)
from scripts.validate_local_endurance_value import (
    DEFAULT_PROFILE,
    DEFAULT_SOURCE,
    _source_terminal_states,
)
from scripts.validate_preview_muscle_allocation import (
    _compact_tracking_audit,
    _full_ding_audit,
    _pw_and_state_audit,
)


DEFAULT_BATCH_BASELINE = Path(".cache/batched-endurance-value-validation/report.json")
DEFAULT_OUTPUT = Path(".cache/triggered-preview-allocation-validation")
ANCHORS = (0, 112)
HORIZONS = (10, 30)
TRIGGER_FRACTIONS = (0.0, 0.01)
FAVORITE_TRIGGER_FRACTION = 0.01
FIT_ABSOLUTE_TOLERANCE = 0.002
FIT_RELATIVE_TOLERANCE = 0.05
FIT_RANKING_TOLERANCE = 1e-5
CODE_PATHS = (
    "cocofest/optimization/compact_muscle_prediction.py",
    "cocofest/optimization/preview_muscle_allocation.py",
    "cocofest/optimization/triggered_preview_allocation.py",
    "cocofest/optimization/local_endurance_value.py",
    "cocofest/optimization/preview_endurance_value.py",
    "cocofest/optimization/rho_adaptive_moment_policy.py",
    "cocofest/optimization/rho_rollout_adapter.py",
    "scripts/validate_local_endurance_value.py",
    "scripts/validate_batched_endurance_value.py",
    "scripts/validate_preview_muscle_allocation.py",
    "scripts/validate_triggered_preview_allocation.py",
)


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


def _accepted_attempt(case):
    accepted = [attempt for attempt in case.get("fit_attempts", ()) if attempt.get("accepted")]
    if len(accepted) != 1:
        raise ValueError("Each local baseline case must contain exactly one accepted fit attempt.")
    return accepted[0]


def _validate_batch_history(
    report,
    *,
    report_sha256,
    local_baseline_sha256,
    source_sha256,
    profile_sha256,
    local_cases,
    expected_value_contexts,
):
    """Return historical H10 greedy-batch fit timings only after strict compatibility."""
    problems = []
    if report.get("schema") != "cocofest-batched-endurance-value-validation-v1":
        problems.append("schema")
    if report.get("uses_fho_data") is not False:
        problems.append("uses_fho_data")
    if report.get("source", {}).get("sha256") != source_sha256:
        problems.append("source SHA-256")
    if report.get("reduced_profile", {}).get("sha256") != profile_sha256:
        problems.append("reduced-profile SHA-256")
    if report.get("baseline_report", {}).get("sha256") != local_baseline_sha256:
        problems.append("local-baseline SHA-256")
    cases = report.get("strict_batch_scalar", ())
    by_anchor = {case.get("anchor_zero_based"): case for case in cases}
    if set(by_anchor) != set(ANCHORS) or len(cases) != len(ANCHORS):
        problems.append("unique H10 anchors 0 and 112")
    history = {}
    for anchor in ANCHORS:
        case = by_anchor.get(anchor, {})
        local_case = local_cases[anchor]
        accepted = _accepted_attempt(local_case)
        attempts = case.get("fit_attempts", ())
        batch_accepted = [attempt for attempt in attempts if attempt.get("batch_accepted")]
        context = case.get("value_context", {})
        if case.get("horizon_cycles") != 10:
            problems.append(f"anchor {anchor} horizon")
        if case.get("timed_point_count") != 41:
            problems.append(f"anchor {anchor} point count")
        if len(batch_accepted) != 1:
            problems.append(f"anchor {anchor} accepted batch fit")
            continue
        prior = batch_accepted[0]
        if not np.array_equal(np.asarray(prior.get("trust_radius")), np.asarray(accepted["trust_radius"])):
            problems.append(f"anchor {anchor} trust radius")
        if context != expected_value_contexts.get(anchor):
            problems.append(f"anchor {anchor} value context")
        elapsed = prior.get("elapsed_s")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) \
                or not np.isfinite(elapsed) or elapsed <= 0:
            problems.append(f"anchor {anchor} elapsed_s")
        history[anchor] = {
            "elapsed_s": elapsed,
            "point_count": case.get("timed_point_count"),
            "trust_radius": prior.get("trust_radius"),
            "source_report_sha256": report_sha256,
            "comparison_kind": "historical_nonconcurrent_batch_greedy_fit",
            "value_context": context,
        }
    if problems:
        raise ValueError("Incompatible batched historical report: " + ", ".join(problems) + ".")
    return history


def _expected_historical_value_contexts(args, local_cases):
    """Rebuild the full strict-batch value context without evaluating the oracle."""
    contexts = {}
    for anchor in ANCHORS:
        task = build_rho_adaptive_moment_policy(
            args.source, args.reduced_profile, cycle_index=anchor, cycle_period=1.0
        )
        cycle, _ = select_certified_rho_cycle(
            args.source, cycle_index=anchor, cycle_period=1.0
        )
        initial = _source_terminal_states(cycle, task.muscle_names)
        coordinates = LocalEnduranceCoordinates.from_state(
            initial, task.parameters, force_scale=np.maximum(initial[:, 1], 1.0)
        )
        predictor = CompactMusclePredictor(task.intervals, task.parameters, substeps=args.substeps)
        moment_scale = float(
            np.max(np.abs([sum(item.target_moments) for item in task.intervals]))
        )
        _validate_baseline_case(
            local_cases[anchor], anchor=anchor, horizon=10, coordinates=coordinates,
            moment_scale=moment_scale, substeps=args.substeps,
        )
        context = CompactEnduranceValueOracle(
            predictor, coordinates, horizon_cycles=10, moment_scale=moment_scale,
            moment_tolerance=args.moment_tolerance,
        ).value_context_metadata
        # These two batch-adapter fields were frozen in the historical strict
        # report; at zero band the nonzero weight contributes no tracking cost.
        context["tracking_penalty_weight"] = 1.0
        context["tracking_cost_definition"] = (
            "duration_mean_squared_original_error_over_task_scale_v1"
        )
        contexts[anchor] = context
    return contexts


def _preview_options(args):
    return {
        "moment_tolerance": args.moment_tolerance,
        "max_iterations": args.max_iterations,
        "fallback_to_greedy": False,
        "optimality_tolerance": args.optimality_tolerance,
        "recruitment_regularization": args.recruitment_regularization,
        "max_solve_time_s": args.max_qp_objective_time_s,
    }


def _make_policies(predictor, moment_scale, args):
    options = _preview_options(args)
    return {
        "greedy_k1": PreviewMuscleAllocation(predictor, preview_phases=1, **options),
        "systematic_k2": PreviewMuscleAllocation(predictor, preview_phases=2, **options),
        "hybrid_0pct": TriggeredPreviewMuscleAllocation(
            predictor, moment_scale=moment_scale, trigger_margin_fraction=0.0, **options
        ),
        "hybrid_1pct": TriggeredPreviewMuscleAllocation(
            predictor, moment_scale=moment_scale,
            trigger_margin_fraction=FAVORITE_TRIGGER_FRACTION, **options,
        ),
    }


def _trigger_summary(result):
    records = tuple(result.diagnostics)
    margins = [record.get("next_signed_margin_nm") for record in records]
    margins = [float(value) for value in margins if value is not None and np.isfinite(value)]
    reasons = Counter(record.get("screen_reason", "not_screened") for record in records)
    return {
        "screenings": int(getattr(result, "screenings", 0)),
        "triggers": int(getattr(result, "triggers", 0)),
        "greedy_fast_path_steps": int(getattr(result, "greedy_fast_path_steps", 0)),
        "screen_failures": int(getattr(result, "screen_failures", 0)),
        "greedy_fallback_steps": int(getattr(result, "greedy_fallback_steps", 0)),
        "screening_time_s": float(getattr(result, "screening_time_s", 0.0)),
        "triggered_preview_time_s": float(getattr(result, "triggered_preview_time_s", 0.0)),
        "policy_reported_total_time_s": float(getattr(result, "total_time_s", 0.0)),
        "minimum_screen_margin_nm": min(margins) if margins else None,
        "screen_reason_counts": dict(sorted(reasons.items())),
    }


def _trajectory_difference(candidate, reference, parameters):
    """Compare their common completed prefix without hiding nonfinite values."""
    steps = min(candidate.completed_intervals, reference.completed_intervals)
    candidate_pw = np.asarray(candidate.pulse_widths).transpose(0, 2, 1).reshape(
        -1, len(parameters)
    )[:steps]
    reference_pw = np.asarray(reference.pulse_widths).transpose(0, 2, 1).reshape(
        -1, len(parameters)
    )[:steps]
    candidate_state = np.asarray(candidate.state_history)[: steps + 1]
    reference_state = np.asarray(reference.state_history)[: steps + 1]
    if not all(np.all(np.isfinite(item)) for item in (
        candidate_pw, reference_pw, candidate_state, reference_state
    )):
        raise ValueError("Nonfinite value in a common completed trajectory prefix.")
    rest = np.asarray([parameter.fatigue.rest_state for parameter in parameters])
    return {
        "common_completed_intervals": steps,
        "maximum_absolute_pw_difference_s": float(
            np.max(np.abs(candidate_pw - reference_pw), initial=0.0)
        ),
        "maximum_absolute_cn_difference": float(
            np.max(np.abs(candidate_state[:, :, 0] - reference_state[:, :, 0]), initial=0.0)
        ),
        "maximum_absolute_force_difference_n": float(
            np.max(np.abs(candidate_state[:, :, 1] - reference_state[:, :, 1]), initial=0.0)
        ),
        "maximum_normalized_slow_state_difference": float(
            np.max(
                np.abs(candidate_state[:, :, 2:] - reference_state[:, :, 2:]) / rest,
                initial=0.0,
            )
        ),
    }


def _case_entry(
    name, anchor, horizon, result, elapsed, allocation, task, initial, parameters, arrays, args
):
    key = f"anchor{anchor}_h{horizon}_{name}"
    entry = {
        "case": key,
        "anchor_zero_based": anchor,
        "horizon_cycles": horizon,
        "policy": name,
        "status": result.status,
        "completed_intervals": result.completed_intervals,
        "requested_intervals": horizon * len(allocation.predictor.intervals),
        "first_failure": result.first_failure,
        "rollout_time_s": elapsed,
        "qp_solves": result.qp_solves,
        "qp_iterations": result.qp_iterations,
        "qp_failures": result.qp_failures,
        "seed_projected_steps": result.seed_projected_steps,
        "compact_tracking": _compact_tracking_audit(result),
        "pw_and_state_audit": _pw_and_state_audit(result, parameters),
        "trigger": _trigger_summary(result) if name.startswith("hybrid") else None,
        "full_ding_replay": None,
        "comparison_to_greedy": None,
        "comparison_to_systematic_k2": None,
    }
    arrays[f"{key}__pulse_widths_s"] = result.pulse_widths
    arrays[f"{key}__state_history"] = result.state_history
    arrays[f"{key}__achieved_moments_nm"] = result.achieved_moments
    arrays[f"{key}__original_total_moment_nm"] = result.original_total_moments
    arrays[f"{key}__signed_moment_error_nm"] = result.signed_moment_errors
    if name.startswith("hybrid") and result.status == "complete":
        replay, replay_arrays = _full_ding_audit(initial, task, result, args.full_ding_substeps)
        entry["full_ding_replay"] = replay
        for suffix, values in replay_arrays.items():
            arrays[f"{key}__{suffix}"] = values
    return entry


def _fit_favorite(args, anchor, predictor, initial, coordinates, moment_scale, radius, historical):
    policy = TriggeredPreviewMuscleAllocation(
        predictor,
        moment_scale=moment_scale,
        trigger_margin_fraction=FAVORITE_TRIGGER_FRACTION,
        **_preview_options(args),
    )
    oracle = PreviewEnduranceValueOracle(
        policy, coordinates, horizon_cycles=10, moment_scale=moment_scale,
        moment_tolerance=args.moment_tolerance,
    )
    started = perf_counter()
    fit = fit_local_endurance_value(
        oracle,
        trust_radius=radius,
        kind="diagonal_quadratic",
        absolute_tolerance=FIT_ABSOLUTE_TOLERANCE,
        relative_tolerance=FIT_RELATIVE_TOLERANCE,
        ranking_tolerance=FIT_RANKING_TOLERANCE,
    )
    elapsed = perf_counter() - started
    model = None if fit.model is None else {
        "center": fit.model.center,
        "constant": fit.model.constant,
        "gradient": fit.model.gradient,
        "diagonal_hessian": fit.model.diagonal_hessian,
        "coefficients": fit.model.coefficients,
        "lower_bounds": fit.model.lower_bounds,
        "upper_bounds": fit.model.upper_bounds,
        "finite_difference_step": fit.model.finite_difference_step,
    }
    return {
        "anchor_zero_based": anchor,
        "horizon_cycles": 10,
        "favorite_policy": "hybrid_1pct",
        "trust_radius": np.asarray(radius).tolist(),
        "elapsed_s": elapsed,
        "accepted": fit.accepted,
        "reason": fit.audit.reason,
        "training_evaluations": fit.metadata.get("training_evaluations", 0),
        "heldout_evaluations": fit.audit.heldout_count,
        "total_oracle_evaluations": (
            fit.metadata.get("training_evaluations", 0) + fit.audit.heldout_count
        ),
        "maximum_absolute_error": fit.audit.maximum_absolute_error,
        "maximum_scaled_error": fit.audit.maximum_scaled_error,
        "ranking_pairs": fit.audit.ranking_pairs,
        "ranking_failures": fit.audit.ranking_failures,
        "finite_difference_step": fit.metadata.get("finite_difference_step"),
        "training_records": fit.metadata.get("training_records", ()),
        "heldout_records": fit.audit.records,
        "model": model,
        "fit_metadata": fit.metadata,
        "value_context": fit.metadata.get("value_context"),
        "historical_batch_greedy_fit": historical,
        "historical_comparison_warning": (
            "Historical compatible report only; not rerun in this timing campaign."
        ),
    }


def _run_anchor(args, anchor, local_case, historical, arrays):
    task = build_rho_adaptive_moment_policy(
        args.source, args.reduced_profile, cycle_index=anchor, cycle_period=1.0
    )
    cycle, _ = select_certified_rho_cycle(args.source, cycle_index=anchor, cycle_period=1.0)
    initial = _source_terminal_states(cycle, task.muscle_names)
    coordinates = LocalEnduranceCoordinates.from_state(
        initial, task.parameters, force_scale=np.maximum(initial[:, 1], 1.0)
    )
    started = perf_counter()
    predictor = CompactMusclePredictor(task.intervals, task.parameters, substeps=args.substeps)
    setup_elapsed = perf_counter() - started
    moment_scale = float(np.max(np.abs([sum(item.target_moments) for item in task.intervals])))
    _validate_baseline_case(
        local_case, anchor=anchor, horizon=10, coordinates=coordinates,
        moment_scale=moment_scale, substeps=args.substeps,
    )
    policies = _make_policies(predictor, moment_scale, args)
    cases = []
    results = {}
    for horizon in HORIZONS:
        horizon_entries = {}
        for name, policy in policies.items():
            started = perf_counter()
            result = policy.rollout(initial, horizon_cycles=horizon)
            elapsed = perf_counter() - started
            results[(horizon, name)] = result
            entry = _case_entry(
                name, anchor, horizon, result, elapsed, policy, task, initial,
                task.parameters, arrays, args,
            )
            horizon_entries[name] = entry
            cases.append(entry)
        greedy = results[(horizon, "greedy_k1")]
        systematic = results[(horizon, "systematic_k2")]
        for name in ("hybrid_0pct", "hybrid_1pct"):
            entry = horizon_entries[name]
            result = results[(horizon, name)]
            entry["comparison_to_greedy"] = _trajectory_difference(
                result, greedy, task.parameters
            )
            entry["comparison_to_systematic_k2"] = _trajectory_difference(
                result, systematic, task.parameters
            )
    favorite_h10 = next(case for case in cases if case["policy"] == "hybrid_1pct"
                        and case["horizon_cycles"] == 10)
    gate_passed = (
        favorite_h10["status"] == "complete"
        and favorite_h10["full_ding_replay"] is not None
        and favorite_h10["full_ding_replay"].get("endpoint_numeric_gate_passed") is True
    )
    radius = np.asarray(_accepted_attempt(local_case)["trust_radius"])
    fit = (
        _fit_favorite(
            args, anchor, predictor, initial, coordinates, moment_scale, radius, historical
        )
        if gate_passed
        else {
            "anchor_zero_based": anchor,
            "horizon_cycles": 10,
            "favorite_policy": "hybrid_1pct",
            "status": "not_run_after_failed_rollout_or_ding_gate",
            "historical_batch_greedy_fit": historical,
        }
    )
    return {
        "anchor_zero_based": anchor,
        "predictor_setup_s": setup_elapsed,
        "moment_scale_nm": moment_scale,
        "coordinate_context_sha256": coordinates.context_signature,
        "source_terminal_states": initial.tolist(),
        "cases": cases,
        "favorite_h10_fit": fit,
    }


def _plot(report, output):
    import matplotlib.pyplot as plt

    cases = [case for anchor in report["anchors"] for case in anchor["cases"]]
    labels = [
        f"a{case['anchor_zero_based']} H{case['horizon_cycles']}\n{case['policy']}"
        for case in cases
    ]
    x = np.arange(len(cases))
    colors = {
        "greedy_k1": "C0", "systematic_k2": "C1",
        "hybrid_0pct": "C2", "hybrid_1pct": "C3",
    }
    bar_colors = [colors[case["policy"]] for case in cases]
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
    axes[0, 0].bar(x, [case["completed_intervals"] for case in cases], color=bar_colors)
    axes[0, 0].scatter(x, [case["requested_intervals"] for case in cases], marker="_", color="k")
    axes[0, 0].set(title="Compact policy progress", ylabel="completed phases")
    axes[0, 1].bar(x, [case["rollout_time_s"] for case in cases], color=bar_colors)
    axes[0, 1].set(title="Allocation cost outside NLP (one pass)", ylabel="seconds", yscale="log")
    moment = [
        np.nan if case["full_ding_replay"] is None else
        case["full_ding_replay"].get("maximum_absolute_original_target_error_nm", np.nan)
        for case in cases
    ]
    slow = [
        np.nan if case["full_ding_replay"] is None else
        case["full_ding_replay"].get("maximum_normalized_slow_state_error", np.nan)
        for case in cases
    ]
    axes[1, 0].bar(x, np.asarray(moment) * 1000.0, color=bar_colors)
    axes[1, 0].axhline(1.0, color="k", linestyle=":", label="1 mN·m gate")
    axes[1, 0].set(title="Full Ding endpoint error (hybrids only)", ylabel="mN·m")
    axes[1, 1].bar(x, np.asarray(slow) * 100.0, color=bar_colors)
    axes[1, 1].axhline(0.1, color="k", linestyle=":", label="0.1% gate")
    axes[1, 1].set(title="Compact vs full Ding slow states", ylabel="% rest")
    for axis, values in ((axes[1, 0], moment), (axes[1, 1], slow)):
        for index, (case, value) in enumerate(zip(cases, values)):
            if not np.isfinite(value):
                replay = case["full_ding_replay"]
                label = "FAIL" if replay is not None and replay.get("status") == "failed" else "NR"
                axis.text(index, 0.02, label, ha="center", va="bottom", fontsize=7)
        axis.text(
            0.99, 0.94, "NR = not replayed; FAIL = replay attempted but failed",
            transform=axis.transAxes, ha="right", va="top", fontsize=8,
        )
    for axis in axes.ravel():
        axis.set_xticks(x, labels, rotation=60, ha="right", fontsize=7)
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    fig.suptitle("Triggered two-phase allocation — numerical validation only; no FHO")
    fig.savefig(output / "validation.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reduced-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--local-baseline", type=Path, default=DEFAULT_LOCAL_BASELINE)
    parser.add_argument("--batch-baseline", type=Path, default=DEFAULT_BATCH_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--full-ding-substeps", type=int, default=16)
    parser.add_argument("--moment-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=60)
    parser.add_argument("--optimality-tolerance", type=float, default=1e-6)
    parser.add_argument("--recruitment-regularization", type=float, default=1e-8)
    parser.add_argument("--max-qp-objective-time-s", type=float, default=0.5)
    args = parser.parse_args(argv)
    source_sha = sha256(args.source.read_bytes()).hexdigest()
    profile_sha = sha256(args.reduced_profile.read_bytes()).hexdigest()
    local_sha = sha256(args.local_baseline.read_bytes()).hexdigest()
    batch_sha = sha256(args.batch_baseline.read_bytes()).hexdigest()
    local_report = json.loads(args.local_baseline.read_text())
    batch_report = json.loads(args.batch_baseline.read_text())
    try:
        local_cases = _validate_baseline_report(
            local_report,
            source_sha256=source_sha,
            profile_sha256=profile_sha,
            substeps=args.substeps,
            absolute_tolerance=FIT_ABSOLUTE_TOLERANCE,
            relative_tolerance=FIT_RELATIVE_TOLERANCE,
            ranking_tolerance=FIT_RANKING_TOLERANCE,
        )
        expected_contexts = _expected_historical_value_contexts(args, local_cases)
        historical = _validate_batch_history(
            batch_report,
            report_sha256=batch_sha,
            local_baseline_sha256=local_sha,
            source_sha256=source_sha,
            profile_sha256=profile_sha,
            local_cases=local_cases,
            expected_value_contexts=expected_contexts,
        )
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {}
    started = perf_counter()
    report = {
        "schema": "cocofest-triggered-preview-allocation-validation-v1",
        "source": {"path": str(args.source.resolve()), "sha256": source_sha},
        "reduced_profile": {"path": str(args.reduced_profile.resolve()), "sha256": profile_sha},
        "input_reports": {
            "local_physical_box": {"path": str(args.local_baseline.resolve()), "sha256": local_sha},
            "historical_batch_greedy": {"path": str(args.batch_baseline.resolve()), "sha256": batch_sha},
        },
        "uses_fho_data": False,
        "adds_future_decision_variables_to_rho": 0,
        "endurance_improvement": "not_established",
        "controller_validation": "not_run",
        "timing_protocol": "sequential one-pass policy rollouts and fits; no concurrent timing or median",
        "settings_predeclared": {
            "anchors_zero_based": list(ANCHORS),
            "horizon_cycles": list(HORIZONS),
            "witness_policies": ["greedy_k1", "systematic_k2"],
            "trigger_margin_fractions": list(TRIGGER_FRACTIONS),
            "favorite_before_computation": "hybrid_1pct",
            "moment_scale": "maximum absolute total task demand at the fixed phase profile",
            "preview_phases_when_triggered": 2,
            "screening_future_phases": 1,
            "fallback_to_greedy": False,
            "substeps": args.substeps,
            "full_ding_substeps": args.full_ding_substeps,
            "moment_tolerance_nm": args.moment_tolerance,
            "fit_kind": "diagonal_quadratic",
            "fit_points_if_complete": "17 training plus 24 deterministic held-out",
            "fit_radii": "accepted H10 radii from compatible physical-box scalar baseline",
            "fit_absolute_tolerance": FIT_ABSOLUTE_TOLERANCE,
            "fit_relative_tolerance": FIT_RELATIVE_TOLERANCE,
            "fit_ranking_tolerance": FIT_RANKING_TOLERANCE,
            "no_h30_fit": True,
        },
        "numeric_gates": {
            "maximum_full_ding_endpoint_moment_error_nm": 1e-3,
            "maximum_normalized_slow_state_error": 1e-3,
        },
        "anchors": [],
    }
    for anchor in ANCHORS:
        result = _run_anchor(args, anchor, local_cases[anchor], historical[anchor], arrays)
        report["anchors"].append(result)
        print(json.dumps({
            "anchor": anchor,
            "cases": [
                {"case": case["case"], "status": case["status"],
                 "completed": case["completed_intervals"], "time_s": case["rollout_time_s"],
                 "triggers": None if case["trigger"] is None else case["trigger"]["triggers"]}
                for case in result["cases"]
            ],
            "fit": result["favorite_h10_fit"],
        }, default=_jsonable), flush=True)
    report["total_elapsed_s"] = perf_counter() - started
    report["code_sha256"] = {
        name: sha256((ROOT / name).read_bytes()).hexdigest() for name in CODE_PATHS
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
