"""Audit slow weight selection on compact RHO-derived tasks, without FHO data.

The recruitment-cost policy is an engineering witness, not a reproduction of
the manuscript's integrated weighted RMS capacity-loss objective. No weights
are applied to a public RHO. All phases are explicitly propagated for ranking;
the exact prescribed-force cycle map is measured separately and never used to
claim that an unverified force waveform remains attainable.
"""

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys
from time import monotonic, perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.endurance_weight_supervisor import (
    EnduranceWeightSupervisor, SupervisorSnapshot, WeightSupervisorConfig,
)
from cocofest.optimization.fixed_force_cycle_map import FixedForceCycleMap
from cocofest.optimization.rho_adaptive_moment_policy import build_rho_adaptive_moment_policy
from cocofest.optimization.rho_rollout_adapter import select_certified_rho_cycle
from cocofest.optimization.weighted_cycle_prediction import WeightedCyclePredictor
from scripts.validate_local_endurance_value import (
    DEFAULT_SOURCE, DEFAULT_PROFILE, _full_ding_replay, _source_terminal_states,
)

CODE_PATHS = (
    "cocofest/optimization/weighted_cycle_prediction.py",
    "cocofest/optimization/compact_muscle_prediction.py",
    "cocofest/optimization/ding_fatigue_rollout.py",
    "cocofest/optimization/fixed_force_cycle_map.py",
    "cocofest/optimization/endurance_weight_supervisor.py",
    "cocofest/optimization/rho_adaptive_moment_policy.py",
    "cocofest/optimization/rho_rollout_adapter.py",
    "scripts/validate_local_endurance_value.py",
    "scripts/validate_multilevel_endurance.py",
)


def _finite_or_none(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _first_cycle_difference(candidate, baseline, phases):
    if candidate is None or baseline is None:
        return {"status": "not_run", "reason": "candidate_or_baseline_result_unavailable"}
    if min(candidate.completed_intervals, baseline.completed_intervals) < phases:
        return {"status": "no_common_complete_first_cycle"}
    maximum_pw = float(np.max(np.abs(candidate.pulse_widths[0] - baseline.pulse_widths[0])))
    maximum_force = float(np.max(np.abs(
        candidate.state_history[:phases + 1, :, 1]
        - baseline.state_history[:phases + 1, :, 1]
    )))
    if not np.isfinite(maximum_pw) or not np.isfinite(maximum_force):
        return {"status": "invalid_nonfinite_common_first_cycle"}
    return {
        "status": "compared",
        "maximum_absolute_pw_difference_s": maximum_pw,
        "maximum_absolute_force_difference_n": maximum_force,
    }


def _prefix_replay(initial, task, result, *, maximum_intervals, substeps, arrays, key):
    steps = min(result.completed_intervals, maximum_intervals)
    if steps == 0:
        return {"status": "not_run", "reason": "no_completed_prefix"}
    started = perf_counter()
    history, moments = _full_ding_replay(initial, task, result.pulse_widths, steps, substeps)
    target = result.original_total_moments.ravel()[:steps]
    moment_error = moments.sum(axis=1) - target
    rest = np.asarray([p.fatigue.rest_state for p in task.parameters])
    slow_error = (history[:, :, 2:] - result.state_history[:steps + 1, :, 2:]) / rest
    maximum_moment = float(np.max(np.abs(moment_error)))
    maximum_slow = float(np.max(np.abs(slow_error)))
    finite = bool(np.isfinite(maximum_moment) and np.isfinite(maximum_slow))
    arrays[f"{key}__ding_prefix_states"] = history
    arrays[f"{key}__ding_prefix_moment_error_nm"] = moment_error
    return {
        "status": "completed_prefix_replay" if finite else "nonfinite_prefix_replay",
        "validation_scope": "explicit_completed_prefix_only",
        "does_not_certify_unreplayed_intervals": True,
        "replayed_intervals": steps,
        "predictor_horizon_completed": bool(result.completed),
        "entire_requested_horizon_replayed": bool(
            result.completed and steps == result.completed_intervals
        ),
        "substeps": substeps, "elapsed_s": perf_counter() - started,
        "maximum_moment_error_nm": _finite_or_none(maximum_moment),
        "maximum_normalized_slow_error": _finite_or_none(maximum_slow),
        "endpoint_gate_passed": bool(
            finite and maximum_moment <= 1e-3 and maximum_slow <= 1e-3
        ),
    }


def _fixed_cycle_audit(initial, task, baseline, horizons):
    if baseline is None:
        return {"status": "not_run", "reason": "baseline_callback_result_unavailable"}
    phases = len(task.intervals)
    if baseline.completed_intervals < phases:
        return {"status": "not_run", "reason": "no_complete_force_cycle"}
    started = perf_counter()
    model = FixedForceCycleMap(
        [p.fatigue for p in task.parameters], [it.duration for it in task.intervals],
        baseline.weighted_force_integrals[0].T,
    )
    construction_s = perf_counter() - started
    first = model.project(initial[:, 2:], 1)
    difference = float(np.max(np.abs(first.slow_states - baseline.state_history[phases, :, 2:])))
    finite_difference = bool(np.isfinite(difference))
    projections = []
    for cycles in horizons:
        started = perf_counter()
        outcome = model.project(initial[:, 2:], cycles)
        elapsed = perf_counter() - started
        projections.append({
            "cycles": cycles, "elapsed_s": elapsed,
            "slow_states": outcome.slow_states,
            "slow_states_finite": bool(np.all(np.isfinite(outcome.slow_states))),
            "slow_domain_valid_at_phase_endpoints": outcome.slow_domain_valid_at_phase_endpoints,
            "force_and_pw_feasibility_checked": outcome.force_and_pw_feasibility_checked,
        })
    return {
        "status": "conditional_algebraic_projection", "construction_s": construction_s,
        "first_cycle_absolute_state_reconstruction_error": _finite_or_none(difference),
        "first_cycle_reconstruction_passed": bool(finite_difference and difference <= 1e-10),
        "force_waveform": "first weighted-all-ones cycle repeated identically",
        "used_for_policy_ranking": False, "projections": projections,
    }


def _candidate_report(evaluated, cached, timings, baseline, phases):
    """Report an attempted callback without assuming that it returned or parsed."""

    candidate = evaluated.candidate
    result = cached.get(candidate.weights)
    if result is None:
        return {
            "index": candidate.index,
            "kind": candidate.kind,
            "muscle": candidate.muscle_name,
            "weights": candidate.weights,
            "status": "callback_result_unavailable",
            "completed_cycles": None,
            "completed_intervals": None,
            "completed_duration_s": None,
            "minimum_signed_margin_nm": None,
            "maximum_tracking_error_nm": None,
            "first_failure": None,
            "elapsed_s": _finite_or_none(timings.get(candidate.weights)),
            "callback_result_cached": False,
            "supervisor_result_parsed": evaluated.valid,
            "supervisor_accepted_evidence": evaluated.comparable,
            "supervisor_error_type": evaluated.error_type,
            "supervisor_error": evaluated.error_message,
            "first_cycle_change_from_weighted_all_ones": _first_cycle_difference(
                None, baseline, phases
            ),
        }

    errors = result.signed_moment_errors[np.isfinite(result.signed_moment_errors)]
    return {
        "index": candidate.index,
        "kind": candidate.kind,
        "muscle": candidate.muscle_name,
        "weights": candidate.weights,
        "status": result.status,
        "completed_cycles": result.completed_cycles,
        "completed_intervals": result.completed_intervals,
        "completed_duration_s": _finite_or_none(result.completed_duration),
        "minimum_signed_margin_nm": _finite_or_none(result.minimum_signed_margin),
        "maximum_tracking_error_nm": (
            _finite_or_none(np.max(np.abs(errors))) if errors.size else None
        ),
        "first_failure": result.first_failure,
        "elapsed_s": _finite_or_none(timings.get(candidate.weights)),
        "callback_result_cached": True,
        "supervisor_result_parsed": evaluated.valid,
        "supervisor_accepted_evidence": evaluated.comparable,
        "supervisor_error_type": evaluated.error_type,
        "supervisor_error": evaluated.error_message,
        "first_cycle_change_from_weighted_all_ones": _first_cycle_difference(
            result, baseline, phases
        ),
    }


def _budget_audit(proposal, declared_count):
    evaluated_count = len(proposal.evaluations)
    if evaluated_count > declared_count:
        raise ValueError("Supervisor evaluated more candidates than were declared.")
    return {
        "budget_seconds": proposal.budget_seconds,
        "candidate_count_declared": declared_count,
        "candidate_count_evaluated": evaluated_count,
        "candidate_count_not_started": declared_count - evaluated_count,
        "budget_exhausted_between_candidates": proposal.budget_exhausted,
        "whole_candidate_callbacks_are_not_interrupted": not proposal.hard_timeout_enforced,
        "measured_budget_overrun_s": proposal.budget_overrun_s,
        "semantics": "budget checked only between completed candidate callbacks",
    }


def _json_finite(value, path="$"):
    """Return strict-JSON data and paths where non-finite numbers became null."""

    if is_dataclass(value):
        return _json_finite(asdict(value), path)
    if isinstance(value, Path):
        return str(value), []
    if isinstance(value, np.ndarray):
        return _json_finite(value.tolist(), path)
    if isinstance(value, np.generic):
        return _json_finite(value.item(), path)
    if isinstance(value, dict):
        converted = {}
        replacements = []
        for key, item in value.items():
            child, child_replacements = _json_finite(item, f"{path}.{key}")
            converted[str(key)] = child
            replacements.extend(child_replacements)
        return converted, replacements
    if isinstance(value, (tuple, list)):
        converted = []
        replacements = []
        for index, item in enumerate(value):
            child, child_replacements = _json_finite(item, f"{path}[{index}]")
            converted.append(child)
            replacements.extend(child_replacements)
        return converted, replacements
    if isinstance(value, float) and not np.isfinite(value):
        return None, [path]
    return value, []


def run_case(args, anchor, horizon, arrays, source_sha, profile_sha):
    if not np.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        raise ValueError("budget_seconds must be finite and strictly positive.")
    task = build_rho_adaptive_moment_policy(args.source, args.reduced_profile, cycle_index=anchor, cycle_period=1.)
    cycle, _ = select_certified_rho_cycle(args.source, cycle_index=anchor, cycle_period=1.)
    initial = _source_terminal_states(cycle, task.muscle_names)
    started = perf_counter()
    predictor = WeightedCyclePredictor(task.intervals, task.parameters, substeps=args.substeps,
                                       reference_regularization=1e-3)
    setup_s = perf_counter() - started
    task_context = {
        "source_sha256": source_sha, "profile_sha256": profile_sha,
        "source_anchor": anchor, "policy": "weighted_normalized_recruitment_v1",
        "substeps": args.substeps, "reference_regularization": 1e-3, "moment_tolerance": 1e-8,
    }
    snapshot = SupervisorSnapshot(
        task_id="rho-derived-signed-moment", context_token=sha256(json.dumps(task_context, sort_keys=True).encode()).hexdigest(),
        cycle_index=anchor + 1, start_time_s=float(anchor + 1), created_at_s=monotonic(),
        horizon_cycles=horizon, muscle_names=task.muscle_names,
        state_component_names=("Cn", "F", "A", "Tau1", "Km"),
        start_state=initial, incumbent_weights=np.ones(len(task.parameters)),
    )
    supervisor = EnduranceWeightSupervisor(WeightSupervisorConfig(muscle_count=len(task.parameters)))
    cached = {}
    timings = {}

    def evaluate(weights, context):
        started = perf_counter()
        try:
            result = predictor.rollout(
                np.asarray(context.start_state),
                weights,
                horizon_cycles=context.horizon_cycles,
            )
        finally:
            timings[weights] = perf_counter() - started
        cached[weights] = result
        return result

    proposal = supervisor.evaluate(snapshot, evaluate, budget_seconds=args.budget_seconds)
    declared_count = len(supervisor.candidates(snapshot))
    incumbent_evaluation = next(
        (
            evaluated
            for evaluated in proposal.evaluations
            if evaluated.candidate.kind == "incumbent"
        ),
        None,
    )
    baseline = cached.get(snapshot.incumbent_weights)
    records = []
    key = f"anchor{anchor}_h{horizon}"
    for evaluated in proposal.evaluations:
        records.append(
            _candidate_report(
                evaluated,
                cached,
                timings,
                baseline,
                len(task.intervals),
            )
        )
    selected = (
        cached.get(proposal.chosen_candidate.weights)
        if proposal.chosen_candidate is not None
        else None
    )
    replays = {
        "baseline": {"status": "not_run", "reason": "baseline_callback_result_unavailable"},
        "selected": {"status": "not_run", "reason": "no_comparable_selected_evidence"},
    }
    for label, result in (("baseline", baseline), ("selected", selected)):
        if result is None:
            continue
        arrays[f"{key}_{label}__states"] = result.state_history[:result.completed_intervals + 1]
        arrays[f"{key}_{label}__pw_s"] = result.pulse_widths
        arrays[f"{key}_{label}__signed_margins_nm"] = result.signed_margins
        if label == "selected" and selected is baseline:
            replays[label] = {**replays["baseline"], "reused_identical_baseline_candidate": True}
        else:
            replays[label] = _prefix_replay(initial, task, result, maximum_intervals=10 * len(task.intervals),
                                            substeps=args.ding_substeps, arrays=arrays, key=f"{key}_{label}")
    return {
        "anchor_zero_based": anchor, "horizon_cycles": horizon, "cycle_period_s": 1.,
        "muscle_names": task.muscle_names, "initial_states": initial, "predictor_setup_s": setup_s,
        "task_context": task_context,
        "policy_metadata": None if baseline is None else baseline.metadata,
        "baseline_callback_succeeded": baseline is not None,
        "baseline_supervisor_evidence_comparable": bool(
            incumbent_evaluation is not None and incumbent_evaluation.comparable
        ),
        "candidate_count_declared": declared_count,
        "candidates": records,
        "proposal": asdict(proposal), "baseline_and_selected_ding_prefix": replays,
        "fixed_force_cycle_map": _fixed_cycle_audit(initial, task, baseline, (120, 300)),
        "budget_audit": _budget_audit(proposal, declared_count),
        "array_semantics": (
            "NPZ rollout arrays retain requested shapes and explicit NaN tails after "
            "the completed prefix; JSON non-finite values are replaced by audited nulls"
        ),
        "actual_rho_response_to_weights": "not_tested",
        "paper_weighted_fatigue_objective_reproduced": False,
    }


def _plot(report, output):
    import matplotlib.pyplot as plt
    cases = report["cases"]
    fig, axes = plt.subplots(2, len(cases), figsize=(4.5 * len(cases), 8), squeeze=False, constrained_layout=True)
    for column, case in enumerate(cases):
        entries = case["candidates"]
        x = np.arange(len(entries))
        selected = case["proposal"]["chosen_candidate"]
        colors = ["C1" if selected is not None and e["index"] == selected["index"] else "C0" for e in entries]
        durations = [
            np.nan if e["completed_duration_s"] is None else e["completed_duration_s"]
            for e in entries
        ]
        elapsed = [np.nan if e["elapsed_s"] is None else e["elapsed_s"] for e in entries]
        axes[0, column].bar(x, durations, color=colors)
        axes[0, column].axhline(case["horizon_cycles"], color="k", linestyle=":")
        axes[0, column].set(title=f"Anchor {case['anchor_zero_based']} / {case['horizon_cycles']} cycles",
                            ylabel="Explicit compact prefix (s)")
        axes[1, column].bar(x, elapsed, color=colors)
        axes[1, column].set(ylabel="Candidate rollout time (s)", xlabel="Candidate index (0 = incumbent)")
        for row in range(2):
            axes[row, column].set_xticks(x)
            axes[row, column].grid(axis="y", alpha=.2)
    fig.suptitle("Slow weight supervisor — recruitment-cost witness, not manuscript fatigue-cost or RHO validation")
    fig.savefig(output / "validation.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reduced-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=Path(".cache/multilevel-endurance-validation"))
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--ding-substeps", type=int, default=16)
    parser.add_argument("--budget-seconds", type=float, default=60.)
    args = parser.parse_args(argv)
    if not np.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        parser.error("budget-seconds must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=True)
    source_sha = sha256(args.source.read_bytes()).hexdigest()
    profile_sha = sha256(args.reduced_profile.read_bytes()).hexdigest()
    arrays = {}
    started = perf_counter()
    report = {
        "schema": "cocofest-multilevel-endurance-validation-v1", "uses_fho_data": False,
        "settings_predeclared": {"anchors": [0, 112], "horizons": [120, 300], "weight_factor": 1.5,
                                  "weight_bounds": [.25, 4.], "initial_weights": [1., 1., 1., 1.],
                                  "reference_regularization": 1e-3, "moment_tolerance": 1e-8,
                                  "budget_seconds_per_case": args.budget_seconds,
                                  "full_ding_prefix_limit_cycles": 10},
        "source": {"path": str(args.source.resolve()), "sha256": source_sha},
        "profile": {"path": str(args.reduced_profile.resolve()), "sha256": profile_sha},
        "timing_protocol": "one sequential pass; fixed-force algebraic timing not a feasible long rollout",
        "controller_endurance_gain": "not_established", "public_rho_modified": False,
        "cases": [],
    }
    for anchor in (0, 112):
        for horizon in (120, 300):
            case = run_case(args, anchor, horizon, arrays, source_sha, profile_sha)
            report["cases"].append(case)
            progress, progress_nonfinite = _json_finite({
                "anchor": anchor,
                "horizon": horizon,
                "selected_weights": case["proposal"]["weights"],
                "elapsed_s": case["proposal"]["elapsed_s"],
                "selection_basis": case["proposal"]["selection_basis"],
                "durations": [c["completed_duration_s"] for c in case["candidates"]],
            })
            if progress_nonfinite:
                progress["nonfinite_values_replaced_with_null"] = progress_nonfinite
            print(json.dumps(progress, allow_nan=False), flush=True)
    report["total_elapsed_s"] = perf_counter() - started
    report["code_sha256"] = {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in CODE_PATHS}
    strict_report, replacements = _json_finite(report)
    strict_report["json_serialization_audit"] = {
        "nonfinite_values_replaced_with_null_count": len(replacements),
        "nonfinite_value_paths": replacements,
    }
    (args.output / "report.json").write_text(
        json.dumps(strict_report, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(args.output / "validation_arrays.npz", **arrays)
    _plot(strict_report, args.output)
    print(f"Report: {args.output / 'report.json'}", flush=True)
    return strict_report


if __name__ == "__main__":
    main()
