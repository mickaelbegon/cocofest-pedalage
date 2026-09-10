"""Replay a strict compact-policy prefix independently before its failed phase.

No FHO data are used. Repeated full Ding replays start from the exact archived
terminal state and use identical compact prefix PW. Signed critical envelopes
are compared at both compact and independently replayed boundaries. This is a
conditional-policy diagnostic, not evidence of overall task infeasibility.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.adaptive_moment_rollout import propagate_ding_pulse_width_interval
from cocofest.optimization.compact_muscle_prediction import CompactMusclePredictor
from cocofest.optimization.rho_adaptive_moment_policy import build_rho_adaptive_moment_policy
from cocofest.optimization.rho_rollout_adapter import select_certified_rho_cycle
from scripts.validate_local_endurance_value import (
    DEFAULT_PROFILE, DEFAULT_SOURCE, _full_ding_replay, _jsonable, _source_terminal_states,
)


def compose_signed_envelope(force_samples, moment_coefficients, *, monotonicity_tolerance_n=1e-8):
    """Check a PW-ordered force grid before composing signed endpoint bounds.

    Finite-grid monotonicity is a numerical check, not an analytic certificate
    between samples. A nonmonotonic grid has no accepted endpoint envelope.
    """
    force = np.asarray(force_samples, float)
    coefficients = np.asarray(moment_coefficients, float)
    if (force.ndim != 2 or force.shape[0] < 3 or coefficients.shape != force.shape[1:]
            or not np.all(np.isfinite(force)) or not np.all(np.isfinite(coefficients))
            or np.any(force < 0)):
        raise ValueError("Finite nonnegative force samples require (PW samples>=3, muscles) shape.")
    if not np.isfinite(monotonicity_tolerance_n) or monotonicity_tolerance_n < 0:
        raise ValueError("monotonicity_tolerance_n must be finite and nonnegative.")
    increments = np.diff(force, axis=0)
    monotonic = np.all(increments >= -monotonicity_tolerance_n, axis=0)
    moment_min_pw, moment_max_pw = force[0] * coefficients, force[-1] * coefficients
    lower_muscle = np.where(coefficients >= 0, moment_min_pw, moment_max_pw)
    upper_muscle = np.where(coefficients >= 0, moment_max_pw, moment_min_pw)
    accepted = bool(np.all(monotonic))
    return {
        "sampled_monotonicity_passed": accepted,
        "monotonic_muscles": monotonic.tolist(),
        "minimum_force_increment_by_muscle_n": increments.min(axis=0).tolist(),
        "lower_bound_nm": float(lower_muscle.sum()) if accepted else None,
        "upper_bound_nm": float(upper_muscle.sum()) if accepted else None,
        "lower_muscle_moments_nm": lower_muscle.tolist(),
        "upper_muscle_moments_nm": upper_muscle.tolist(),
        "lower_uses_pw_max": (coefficients < 0).tolist(),
    }


def full_ding_phase_envelope(states, interval, parameters, *, substeps, pw_samples=9):
    if isinstance(pw_samples, bool) or int(pw_samples) != pw_samples or pw_samples < 3:
        raise ValueError("pw_samples must be an integer >=3.")
    states = np.asarray(states, float)
    if states.shape != (len(parameters), 5):
        raise ValueError("states must have shape (muscles, 5).")
    fractions = np.linspace(0., 1., pw_samples)
    history = np.empty((pw_samples, len(parameters), 5))
    widths = np.empty((pw_samples, len(parameters)))
    for sample, fraction in enumerate(fractions):
        for muscle, parameter in enumerate(parameters):
            widths[sample, muscle] = parameter.pd0 + fraction * (parameter.pulse_width_max - parameter.pd0)
            history[sample, muscle] = propagate_ding_pulse_width_interval(
                states[muscle], pulse_width=widths[sample, muscle], duration=interval.duration,
                calcium_amplitude=interval.calcium_amplitudes[muscle],
                mechanical_gain=interval.mechanical_gains[muscle], parameters=parameter,
                integration_substeps=substeps)
    result = compose_signed_envelope(history[:, :, 1], interval.moment_coefficients)
    required = float(np.sum(interval.target_moments))
    result.update({"original_target_nm": required, "pw_samples": pw_samples,
                   "lower_signed_margin_nm": None if result["lower_bound_nm"] is None else required - result["lower_bound_nm"],
                   "upper_signed_margin_nm": None if result["upper_bound_nm"] is None else result["upper_bound_nm"] - required})
    return result, history, widths


def diagnose_failure(initial, policy, *, horizon_cycles=30, compact_substeps=16,
                     refinements=(16, 32, 64), maximum_substeps=256, pw_samples=9,
                     convergence_absolute_nm=1e-6, convergence_deficit_fraction=.01):
    """Return JSON-ready evidence and arrays; refine until a preset gate passes."""
    if isinstance(pw_samples, bool) or int(pw_samples) != pw_samples or pw_samples < 3:
        raise ValueError("pw_samples must be an integer >=3.")
    levels = list(refinements)
    if (len(levels) < 2 or any(isinstance(n, bool) or int(n) != n or n < 1 for n in levels)
            or any(b <= a for a, b in zip(levels, levels[1:])) or maximum_substeps < levels[-1]):
        raise ValueError("refinements must be increasing positive integers, within maximum_substeps.")
    if any(not np.isfinite(v) or v <= 0 for v in (convergence_absolute_nm, convergence_deficit_fraction)):
        raise ValueError("Convergence thresholds must be finite and positive.")
    predictor = CompactMusclePredictor(policy.intervals, policy.parameters, substeps=compact_substeps)
    compact = predictor.rollout(initial, horizon_cycles=horizon_cycles)
    if compact.status == "complete" or compact.first_failure is None:
        raise ValueError("The strict compact rollout completed; there is no failed phase to diagnose.")
    completed = compact.completed_intervals
    count = len(policy.intervals)
    cycle, phase = divmod(completed, count)
    interval = policy.intervals[phase]
    compact_boundary = compact.state_history[completed].copy()
    transition = predictor.phase_map(compact_boundary, phase)
    force_samples = np.array([transition.endpoint(transition.maximum_recruitment * fraction)[:, 1]
                              for fraction in np.linspace(0., 1., pw_samples)])
    compact_envelope = compose_signed_envelope(force_samples, interval.moment_coefficients)
    required = float(np.sum(interval.target_moments))
    compact_envelope.update({"original_target_nm": required,
                             "lower_signed_margin_nm": required - compact_envelope["lower_bound_nm"],
                             "upper_signed_margin_nm": compact_envelope["upper_bound_nm"] - required})
    deficit = max(-compact_envelope["lower_signed_margin_nm"], -compact_envelope["upper_signed_margin_nm"], 0.)
    threshold = min(convergence_absolute_nm, convergence_deficit_fraction * deficit) if deficit else convergence_absolute_nm
    targets = np.array([sum(policy.intervals[k % count].target_moments) for k in range(completed)])
    rest = np.array([parameter.fatigue.rest_state for parameter in policy.parameters])
    arrays = {"initial_states": np.asarray(initial), "compact_prefix_pulse_widths_s": compact.pulse_widths,
              "compact_prefix_state_history": compact.state_history[:completed + 1],
              "original_prefix_targets_nm": targets,
              "critical_moment_coefficients": np.asarray(interval.moment_coefficients)}
    records, previous = [], None
    index = 0
    while index < len(levels):
        substeps = levels[index]
        started = perf_counter()
        history, moments = _full_ding_replay(initial, policy, compact.pulse_widths, completed, substeps)
        replay_envelope, replay_endpoints, widths = full_ding_phase_envelope(
            history[-1], interval, policy.parameters, substeps=substeps, pw_samples=pw_samples)
        same_boundary_envelope, same_boundary_endpoints, _ = full_ding_phase_envelope(
            compact_boundary, interval, policy.parameters, substeps=substeps, pw_samples=pw_samples)
        errors = moments.sum(axis=1) - targets
        record = {"full_ding_substeps": substeps, "elapsed_s": perf_counter() - started,
                  "prefix_completed_intervals": completed,
                  "prefix_maximum_original_target_error_nm": float(np.max(np.abs(errors))) if completed else 0.,
                  "prefix_maximum_normalized_slow_difference_from_compact": float(np.max(np.abs(
                      (history[:, :, 2:] - compact.state_history[:completed + 1, :, 2:]) / rest))),
                  "replayed_boundary_envelope": replay_envelope,
                  "compact_boundary_full_ding_envelope": same_boundary_envelope,
                  "refinement": None}
        current = (history, moments, replay_endpoints, same_boundary_endpoints, replay_envelope, same_boundary_envelope)
        if previous is not None:
            coefficient = np.asarray(interval.moment_coefficients)
            envelope_change = max(abs(replay_envelope[key] - previous[4][key])
                                  for key in ("lower_bound_nm", "upper_bound_nm")) if (
                replay_envelope["sampled_monotonicity_passed"] and previous[4]["sampled_monotonicity_passed"]) else None
            same_boundary_change = max(abs(same_boundary_envelope[key] - previous[5][key])
                                       for key in ("lower_bound_nm", "upper_bound_nm")) if (
                same_boundary_envelope["sampled_monotonicity_passed"] and previous[5]["sampled_monotonicity_passed"]) else None
            prefix_change = float(np.max(np.abs((moments - previous[1]).sum(axis=1)))) if completed else 0.
            endpoint_change = float(np.max(np.abs((replay_endpoints[:, :, 1] - previous[2][:, :, 1]) * coefficient)))
            same_boundary_endpoint_change = float(np.max(np.abs(
                (same_boundary_endpoints[:, :, 1] - previous[3][:, :, 1]) * coefficient)))
            record["refinement"] = {
                "previous_substeps": records[-1]["full_ding_substeps"],
                "maximum_prefix_total_moment_change_nm": prefix_change,
                "maximum_critical_signed_bound_change_nm": envelope_change,
                "maximum_critical_sample_muscle_moment_change_nm": endpoint_change,
                "maximum_compact_boundary_signed_bound_change_nm": same_boundary_change,
                "maximum_compact_boundary_sample_muscle_moment_change_nm": same_boundary_endpoint_change,
                "maximum_normalized_slow_state_change": float(np.max(np.abs((history[:, :, 2:] - previous[0][:, :, 2:]) / rest))),
                "threshold_nm": threshold,
                "passed": bool(envelope_change is not None and same_boundary_change is not None and
                               max(prefix_change, envelope_change, endpoint_change, same_boundary_change,
                                   same_boundary_endpoint_change) <= threshold)}
        records.append(record)
        arrays.update({f"rk4_{substeps}__prefix_state_history": history,
                       f"rk4_{substeps}__prefix_muscle_moments_nm": moments,
                       f"rk4_{substeps}__prefix_original_target_errors_nm": errors,
                       f"rk4_{substeps}__replayed_boundary_critical_pw_states": replay_endpoints,
                       f"rk4_{substeps}__compact_boundary_critical_pw_states": same_boundary_endpoints})
        previous = current
        index += 1
        if index == len(levels) and not (record["refinement"] and record["refinement"]["passed"]):
            if substeps * 2 <= maximum_substeps:
                levels.append(substeps * 2)
    arrays["critical_pw_samples_s"] = widths
    finest = records[-1]
    envelope = finest["replayed_boundary_envelope"]
    monotonic = all(record["replayed_boundary_envelope"]["sampled_monotonicity_passed"] and
                    record["compact_boundary_full_ding_envelope"]["sampled_monotonicity_passed"] for record in records)
    converged = bool(finest["refinement"] and finest["refinement"]["passed"] and monotonic)
    outside = bool(monotonic and min(envelope["lower_signed_margin_nm"], envelope["upper_signed_margin_nm"]) < -threshold)
    report = {"no_fho_data": True, "prediction_start": "exact_archived_source_terminal_state",
              "requested_horizon_cycles": horizon_cycles, "compact_substeps": compact_substeps,
              "compact_status": compact.status, "compact_completed_intervals": completed,
              "first_failure": compact.first_failure, "critical_cycle_zero_based": cycle,
              "critical_phase_zero_based": phase, "compact_critical_envelope": compact_envelope,
              "compact_deficit_nm": deficit, "convergence_threshold_nm": threshold,
              "convergence_passed": converged, "refinements": records,
              "conditional_critical_target_outside_refined_signed_envelope": outside if converged else None,
              "complete_horizon_validated": False,
              "conclusion": ("The original target remains outside the sampled-monotone full-Ding signed envelope after independently replaying the compact policy prefix."
                             if converged and outside else "The refined diagnostic does not establish the same conditional failure."),
              "limitations": ["The prefix replays fixed compact PW; it is not a newly optimized full-Ding policy.",
                              "Prefix tracking error against the original task is reported; exact tracking is not presumed.",
                              "Monotonicity is checked on a finite PW grid, not proven continuously.",
                              "Convergence is estimated by refinement, not a rigorous integration error bound.",
                              "No global infeasibility, complete H30 success, or endurance improvement is claimed."]}
    return report, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reduced-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=Path(".cache/compact-rollout-failure-diagnostic"))
    parser.add_argument("--anchor-index", type=int, default=112)
    parser.add_argument("--horizon-cycles", type=int, default=30)
    parser.add_argument("--cycle-period", type=float, default=1.)
    parser.add_argument("--compact-substeps", type=int, default=16)
    parser.add_argument("--refinements", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--maximum-substeps", type=int, default=256)
    parser.add_argument("--pw-samples", type=int, default=9)
    args = parser.parse_args(argv)
    policy = build_rho_adaptive_moment_policy(args.source, args.reduced_profile,
                                             cycle_index=args.anchor_index, cycle_period=args.cycle_period)
    cycle, _ = select_certified_rho_cycle(args.source, cycle_index=args.anchor_index, cycle_period=args.cycle_period)
    initial = _source_terminal_states(cycle, policy.muscle_names)
    report, arrays = diagnose_failure(initial, policy, horizon_cycles=args.horizon_cycles,
                                      compact_substeps=args.compact_substeps, refinements=args.refinements,
                                      maximum_substeps=args.maximum_substeps, pw_samples=args.pw_samples)
    report.update({"source_cycle_index_zero_based": args.anchor_index, "muscle_names": list(policy.muscle_names),
                   "cycle_period_s": args.cycle_period, "source_hashes_sha256": {}})
    sources = [args.source, args.reduced_profile, Path(__file__),
               ROOT / "scripts/validate_local_endurance_value.py",
               ROOT / "cocofest/optimization/compact_muscle_prediction.py",
               ROOT / "cocofest/optimization/adaptive_moment_rollout.py",
               ROOT / "cocofest/optimization/rho_adaptive_moment_policy.py",
               ROOT / "cocofest/optimization/rho_rollout_adapter.py",
               ROOT / "cocofest/optimization/smooth_muscle_moment_allocation.py"]
    for source in sources:
        report["source_hashes_sha256"][str(source.resolve())] = sha256(source.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "diagnostic_arrays.npz", **arrays)
    (args.output / "diagnostic_report.json").write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "completed_prefix": report["compact_completed_intervals"],
                      "convergence_passed": report["convergence_passed"],
                      "compact_deficit_nm": report["compact_deficit_nm"],
                      "finest_replayed_boundary_envelope": report["refinements"][-1]["replayed_boundary_envelope"]}, indent=2))


if __name__ == "__main__":
    main()
