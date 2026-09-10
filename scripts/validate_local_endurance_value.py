"""Validate a local compact-policy endurance value on real RHO boundaries.

This experiment uses no FHO trajectory.  It fits outside the RHO NLP, audits
deterministic held-out points, and independently replays selected compact PW
sequences with the full Ding equations.  The result is numerical model
validation, not evidence that a controller improves endurance.
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

from cocofest.optimization.adaptive_moment_rollout import propagate_ding_pulse_width_interval
from cocofest.optimization.compact_muscle_prediction import CompactMusclePredictor
from cocofest.optimization.local_endurance_value import (
    CompactEnduranceValueOracle,
    LocalEnduranceCoordinates,
    fit_local_endurance_value,
)
from cocofest.optimization.rho_adaptive_moment_policy import build_rho_adaptive_moment_policy
from cocofest.optimization.rho_rollout_adapter import select_certified_rho_cycle


DEFAULT_SOURCE = Path(
    "ipopt-linear-solver-150-20260904/resistance-0p10Nm/"
    "ipopt-sx-radau5-ma57-150-max2000-reduced/validated-rho-trajectory.npz"
)
DEFAULT_PROFILE = Path("benchmark-seed/reduced-cycling-fourier12.npz")
DEFAULT_OUTPUT = Path(".cache/local-endurance-value-validation/physical-box")
STATE_KEYS = ("Cn", "F", "A", "Tau1", "Km")


def _case(value: str) -> tuple[int, int]:
    try:
        anchor, horizon = (int(item) for item in value.split(":"))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("cases must have form zero_based_anchor:horizon") from None
    if anchor < 0 or horizon < 1:
        raise argparse.ArgumentTypeError("anchor must be nonnegative and horizon positive")
    return anchor, horizon


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
    if isinstance(value, Path):
        return str(value)
    return value


def _source_terminal_states(cycle, muscle_names) -> np.ndarray:
    """Use the actual archived final column, including its archived Cn value."""

    return np.asarray(
        [
            [cycle.states[f"{key}_{muscle}"][cycle.end_column] for key in STATE_KEYS]
            for muscle in muscle_names
        ],
        dtype=float,
    )


def _full_ding_replay(initial, policy, pulse_widths, completed_intervals, substeps):
    interval_count = len(policy.intervals)
    history = np.full((completed_intervals + 1, len(policy.parameters), 5), np.nan)
    moments = np.full((completed_intervals, len(policy.parameters)), np.nan)
    history[0] = initial
    current = initial.copy()
    for step in range(completed_intervals):
        cycle_index, interval_index = divmod(step, interval_count)
        interval = policy.intervals[interval_index]
        for muscle_index, parameters in enumerate(policy.parameters):
            current[muscle_index] = propagate_ding_pulse_width_interval(
                current[muscle_index],
                pulse_width=pulse_widths[cycle_index, muscle_index, interval_index],
                duration=interval.duration,
                calcium_amplitude=interval.calcium_amplitudes[muscle_index],
                mechanical_gain=interval.mechanical_gains[muscle_index],
                parameters=parameters,
                integration_substeps=substeps,
            )
        moments[step] = current[:, 1] * np.asarray(interval.moment_coefficients)
        history[step + 1] = current
    return history, moments


def _status_counts(attempts) -> dict[str, int]:
    statuses = Counter()
    for attempt in attempts:
        for record in attempt["training_records"] + attempt["heldout_records"]:
            statuses[record["status"]] += 1
    return dict(sorted(statuses.items()))


def _fit_attempt_record(fit, elapsed, radius):
    training = fit.metadata.get("training_records", [])
    heldout = list(fit.audit.records)
    return {
        "accepted": fit.accepted,
        "elapsed_s": elapsed,
        "trust_radius": radius.tolist(),
        "finite_difference_step": (radius / 4).tolist(),
        "reason": fit.audit.reason,
        "training_evaluations_observed": len(training),
        "training_records": training,
        "heldout_evaluations_requested": fit.metadata.get("heldout_evaluations_requested", 0),
        "heldout_evaluations_observed": len(heldout),
        "heldout_records": heldout,
        "maximum_absolute_error": fit.audit.maximum_absolute_error,
        "maximum_scaled_error": fit.audit.maximum_scaled_error,
        "ranking_pairs": fit.audit.ranking_pairs,
        "ranking_failures": fit.audit.ranking_failures,
    }


def _observed_center_status(fit):
    records = fit.metadata.get("training_records", [])
    return records[0].get("status") if records else None


def _initial_trust_radius(
    center, muscle_count, *, damage_radius, force_radius, maximum_damage_fraction,
    maximum_force_fraction,
):
    center = np.asarray(center, dtype=float)
    damage = center[:muscle_count]
    force = center[muscle_count:]
    if np.any(damage <= 0) or np.any(force <= 0):
        raise ValueError(
            "A symmetric physiological box requires strictly positive anchor damage and force; "
            "use another parameterization at a zero boundary."
        )
    radius = np.r_[
        np.minimum(damage_radius, maximum_damage_fraction * damage),
        np.minimum(force_radius, maximum_force_fraction * force),
    ]
    if np.any(center - radius < 0):
        raise AssertionError("Trust-radius construction crossed a nonnegative physiological bound.")
    return radius


def _replay_point(label, point, *, coordinates, predictor, policy, horizon, substeps):
    initial = coordinates.decode(point)
    compact_start = perf_counter()
    compact = predictor.rollout(initial, horizon_cycles=horizon)
    compact_time = perf_counter() - compact_start
    entry = {
        "label": label,
        "coordinates": point.tolist(),
        "compact_status": compact.status,
        "compact_completed_intervals": compact.completed_intervals,
        "compact_time_s": compact_time,
        "full_ding_integration_substeps": substeps,
    }
    arrays = {}
    if compact.status != "complete":
        entry["full_ding_replay_status"] = "not_run"
        entry["first_failure"] = compact.first_failure
        return entry, arrays
    replay_start = perf_counter()
    try:
        history, moments = _full_ding_replay(
            initial, policy, compact.pulse_widths, compact.completed_intervals, substeps
        )
    except (ValueError, FloatingPointError, OverflowError) as error:
        entry.update(
            {
                "full_ding_replay_status": "failed",
                "full_ding_replay_time_s": perf_counter() - replay_start,
                "full_ding_replay_error": str(error),
            }
        )
        return entry, arrays
    replay_time = perf_counter() - replay_start
    target = np.tile(
        [sum(interval.target_moments) for interval in policy.intervals], horizon
    )[: compact.completed_intervals]
    moment_error = moments.sum(axis=1) - target
    rest = np.asarray([parameters.fatigue.rest_state for parameters in policy.parameters])
    slow_error = (compact.state_history[: compact.completed_intervals + 1, :, 2:] - history[:, :, 2:]) / rest
    entry.update(
        {
            "full_ding_replay_status": "complete",
            "full_ding_replay_time_s": replay_time,
            "maximum_absolute_moment_error_nm": float(np.max(np.abs(moment_error))),
            "rmse_moment_error_nm": float(np.sqrt(np.mean(moment_error**2))),
            "maximum_normalized_slow_state_error": float(np.max(np.abs(slow_error))),
            "numeric_gate_passed": bool(
                np.max(np.abs(moment_error)) <= 1e-3
                and np.max(np.abs(slow_error)) <= 1e-3
            ),
        }
    )
    arrays = {
        "moment_error_nm": moment_error,
        "maximum_normalized_slow_error_by_node": np.max(np.abs(slow_error), axis=(1, 2)),
        "compact_pulse_widths_s": compact.pulse_widths,
        "compact_state_history": compact.state_history,
        "full_ding_state_history": history,
    }
    return entry, arrays


def _run_case(args, anchor_index, horizon, arrays):
    policy = build_rho_adaptive_moment_policy(
        args.source, args.reduced_profile, cycle_index=anchor_index, cycle_period=args.cycle_period
    )
    cycle, _ = select_certified_rho_cycle(
        args.source, cycle_index=anchor_index, cycle_period=args.cycle_period
    )
    initial = _source_terminal_states(cycle, policy.muscle_names)
    # A one-newton floor keeps the normalization readable without magnifying
    # millinewton residual forces. Trust radii still limit every force change
    # to at most a fixed fraction of its actual positive anchor value.
    force_scale = np.maximum(initial[:, 1], 1.0)
    coordinates = LocalEnduranceCoordinates.from_state(
        initial, policy.parameters, force_scale=force_scale
    )
    preparation_start = perf_counter()
    predictor = CompactMusclePredictor(policy.intervals, policy.parameters, substeps=args.substeps)
    preparation_time = perf_counter() - preparation_start
    required = np.asarray([sum(interval.target_moments) for interval in policy.intervals])
    moment_scale = float(np.max(np.abs(required)))
    oracle = CompactEnduranceValueOracle(
        predictor,
        coordinates,
        horizon_cycles=horizon,
        moment_scale=moment_scale,
        softmin_temperature=args.softmin_temperature,
        margin_target=args.margin_target,
        penalty_temperature=args.penalty_temperature,
    )
    base_start = perf_counter()
    base = oracle.evaluate(coordinates.anchor)
    base_time = perf_counter() - base_start
    initial_radius = _initial_trust_radius(
        coordinates.anchor,
        len(policy.parameters),
        damage_radius=args.damage_radius,
        force_radius=args.force_radius,
        maximum_damage_fraction=args.maximum_damage_fraction,
        maximum_force_fraction=args.maximum_force_fraction,
    )
    attempts = []
    accepted_fit = None
    for attempt_index in range(args.maximum_fit_attempts):
        radius = initial_radius * args.radius_shrink**attempt_index
        start = perf_counter()
        fit = fit_local_endurance_value(
            oracle,
            trust_radius=radius,
            kind="diagonal_quadratic",
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
            ranking_tolerance=args.ranking_tolerance,
        )
        attempts.append(_fit_attempt_record(fit, perf_counter() - start, radius))
        if fit.accepted:
            accepted_fit = fit
            break
        # A failed center is invariant under shrinking the box. Repeating it
        # would inflate timing and sample counts without adding evidence.
        if _observed_center_status(fit) not in (None, "complete"):
            break

    key = f"anchor{anchor_index}_h{horizon}"
    entry = {
        "case": key,
        "source_cycle_index_zero_based": anchor_index,
        "prediction_start": "source_cycle_final_boundary",
        "horizon_cycles": horizon,
        "muscle_names": list(policy.muscle_names),
        "source_terminal_states": initial.tolist(),
        "fixed_source_terminal_cn": coordinates.fixed_cn.tolist(),
        "preserved_fatigue_offsets": coordinates.offsets.tolist(),
        "force_scale_n": force_scale.tolist(),
        "coordinate_center": coordinates.anchor.tolist(),
        "coordinate_context_sha256": coordinates.context_signature,
        "moment_scale_nm": moment_scale,
        "predictor_substeps": args.substeps,
        "predictor_preparation_time_s": preparation_time,
        "base_oracle_time_s": base_time,
        "base_oracle": _jsonable(base),
        "fit_kind": "diagonal_quadratic",
        "fit_attempts": attempts,
        "fit_accepted": accepted_fit is not None,
        "accepted_attempt_zero_based": None if accepted_fit is None else len(attempts) - 1,
        "oracle_status_counts_observed_during_fits": _status_counts(attempts),
        "fit_rejections_before_acceptance_or_stop": sum(not item["accepted"] for item in attempts),
        "trust_domain_outside_point_rejected": None,
        "full_ding_replays": [],
    }
    arrays[f"{key}__source_terminal_states"] = initial
    arrays[f"{key}__coordinate_center"] = coordinates.anchor
    arrays[f"{key}__force_scale_n"] = force_scale
    if accepted_fit is not None:
        model = accepted_fit.model
        radius = np.asarray(attempts[-1]["trust_radius"])
        entry["model"] = {
            "constant": model.constant,
            "gradient": model.gradient.tolist(),
            "diagonal_hessian": model.diagonal_hessian.tolist(),
            "lower_bounds": model.lower_bounds.tolist(),
            "upper_bounds": model.upper_bounds.tolist(),
            "coefficients": model.coefficients.tolist(),
        }
        arrays[f"{key}__model_coefficients"] = model.coefficients
        arrays[f"{key}__trust_radius"] = radius
        records = accepted_fit.audit.records
        arrays[f"{key}__heldout_coordinates"] = np.asarray([item["coordinates"] for item in records])
        arrays[f"{key}__heldout_oracle_value"] = np.asarray([item["value"] for item in records])
        arrays[f"{key}__heldout_polynomial_value"] = np.asarray([item["prediction"] for item in records])
        outside = model.upper_bounds.copy()
        outside[0] += 0.01 * radius[0]
        try:
            model.evaluate(outside)
        except ValueError:
            entry["trust_domain_outside_point_rejected"] = True
        else:
            entry["trust_domain_outside_point_rejected"] = False
        if horizon == 10:
            descent_corner = model.center - radius * np.sign(model.gradient)
            for label, point in (("anchor", model.center), ("predicted_descent_box_corner", descent_corner)):
                replay, replay_arrays = _replay_point(
                    label,
                    point,
                    coordinates=coordinates,
                    predictor=predictor,
                    policy=policy,
                    horizon=horizon,
                    substeps=args.full_ding_substeps,
                )
                entry["full_ding_replays"].append(replay)
                for suffix, values in replay_arrays.items():
                    arrays[f"{key}__replay_{label}__{suffix}"] = values
    return entry


def _plot(report, arrays, output):
    import matplotlib.pyplot as plt

    cases = report["cases"]
    accepted = [case for case in cases if case["fit_accepted"]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(accepted), 1)))
    all_values = []
    for color, case in zip(colors, accepted):
        key = case["case"]
        actual = arrays[f"{key}__heldout_oracle_value"]
        predicted = arrays[f"{key}__heldout_polynomial_value"]
        axes[0, 0].scatter(actual, predicted, s=23, alpha=0.8, color=color, label=key)
        all_values.extend(actual.tolist() + predicted.tolist())
        error = np.abs(predicted - actual)
        axes[0, 1].plot(np.arange(len(error)), error, marker=".", color=color, label=key)
        radius = arrays[f"{key}__trust_radius"]
        gradient = np.asarray(case["model"]["gradient"])
        axes[1, 0].plot(
            np.arange(len(gradient)), gradient * radius, marker="o", color=color, label=key
        )
    if all_values:
        low, high = min(all_values), max(all_values)
        padding = max((high - low) * 0.05, 1e-7)
        axes[0, 0].plot([low - padding, high + padding], [low - padding, high + padding], "k:")
    axes[0, 0].set(title="Held-out value: polynomial vs compact oracle", xlabel="Oracle", ylabel="Polynomial")
    axes[0, 1].axhline(report["thresholds"]["absolute_tolerance"], color="k", linestyle=":",
                       label="absolute tolerance")
    axes[0, 1].set(title="Absolute held-out residual", xlabel="Deterministic held-out index", ylabel="|error|")
    coordinate_labels = [f"d{i + 1}" for i in range(4)] + [f"F{i + 1}" for i in range(4)]
    axes[1, 0].axhline(0, color="k", linewidth=0.7)
    axes[1, 0].set(
        title="First-order value change over accepted radius",
        xticks=np.arange(8), xticklabels=coordinate_labels, ylabel="gradient × radius",
    )
    for case in cases:
        for replay in case["full_ding_replays"]:
            if replay.get("full_ding_replay_status") != "complete":
                continue
            key = case["case"]
            error = arrays[f"{key}__replay_{replay['label']}__moment_error_nm"]
            interval_count = len(error) / case["horizon_cycles"]
            axes[1, 1].plot(
                np.arange(1, len(error) + 1) / interval_count,
                error * 1000,
                label=f"{key}: {replay['label']}",
            )
    axes[1, 1].axhline(1, color="k", linestyle=":")
    axes[1, 1].axhline(-1, color="k", linestyle=":")
    axes[1, 1].set(
        title="Selected compact PW replayed with full Ding",
        xlabel="Future cycles", ylabel="Total-moment error (mN·m)",
    )
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    fig.suptitle("Local endurance value — numerical model validation only; no FHO")
    fig.savefig(output / "validation.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reduced-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", type=_case, action="append", dest="cases")
    parser.add_argument("--cycle-period", type=float, default=1.0)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--full-ding-substeps", type=int, default=16)
    parser.add_argument("--damage-radius", type=float, default=0.0025)
    parser.add_argument("--force-radius", type=float, default=0.02)
    parser.add_argument("--maximum-damage-fraction", type=float, default=0.25)
    parser.add_argument("--maximum-force-fraction", type=float, default=0.25)
    parser.add_argument("--maximum-fit-attempts", type=int, default=5)
    parser.add_argument("--radius-shrink", type=float, default=0.5)
    parser.add_argument("--absolute-tolerance", type=float, default=0.002)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--ranking-tolerance", type=float, default=1e-5)
    parser.add_argument("--softmin-temperature", type=float, default=0.05)
    parser.add_argument("--margin-target", type=float, default=0.1)
    parser.add_argument("--penalty-temperature", type=float, default=0.05)
    args = parser.parse_args(argv)
    args.cases = args.cases or [(0, 10), (112, 10), (112, 30)]
    positive = (
        "cycle_period", "substeps", "full_ding_substeps", "damage_radius", "force_radius",
        "maximum_damage_fraction", "maximum_force_fraction", "maximum_fit_attempts", "radius_shrink", "absolute_tolerance",
        "ranking_tolerance", "softmin_temperature", "penalty_temperature",
    )
    for name in positive:
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.relative_tolerance < 0 or not np.isfinite(args.relative_tolerance):
        parser.error("relative-tolerance must be finite and nonnegative")
    if args.radius_shrink >= 1:
        parser.error("radius-shrink must be below one")
    if int(args.maximum_fit_attempts) != args.maximum_fit_attempts:
        parser.error("maximum-fit-attempts must be an integer")
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {}
    report = {
        "schema": "cocofest-local-endurance-value-validation-v1",
        "source": {"path": str(args.source.resolve()), "sha256": sha256(args.source.read_bytes()).hexdigest()},
        "reduced_profile": {
            "path": str(args.reduced_profile.resolve()),
            "sha256": sha256(args.reduced_profile.read_bytes()).hexdigest(),
        },
        "uses_fho_data": False,
        "adds_future_decision_variables_to_rho": 0,
        "controller_validation": "not_run",
        "endurance_improvement": "not_established",
        "coordinate_order": "all four damage coordinates, then all four F/force_scale coordinates",
        "fit_protocol": {
            "deterministic": True,
            "kind": "diagonal_quadratic",
            "initial_damage_radius": args.damage_radius,
            "initial_force_radius": args.force_radius,
            "maximum_damage_fraction_of_anchor": args.maximum_damage_fraction,
            "maximum_force_fraction_of_anchor": args.maximum_force_fraction,
            "finite_difference_fraction_of_radius": 0.25,
            "radius_shrink_after_rejection": args.radius_shrink,
            "maximum_fit_attempts": args.maximum_fit_attempts,
            "failed_anchor_stops_without_redundant_refits": True,
        },
        "thresholds": {
            "absolute_tolerance": args.absolute_tolerance,
            "relative_tolerance": args.relative_tolerance,
            "ranking_tolerance": args.ranking_tolerance,
            "maximum_full_ding_replay_moment_error_nm": 1e-3,
            "maximum_normalized_slow_state_error": 1e-3,
        },
        "assumptions": [
            "repeat source-cycle kinematics and total-moment task",
            "preserve archived terminal Cn and exact Tau1/Km fatigue offsets",
            "restrict sampled initial damage and force to nonnegative values",
            "this simple physiological box is not proof of OCP terminal-state reachability",
            "adapt PW by the compact bounded-allocation policy outside the RHO NLP",
            "local value is conditional on this policy and fixed task context",
        ],
        "cases": [],
    }
    start = perf_counter()
    for anchor, horizon in args.cases:
        case = _run_case(args, anchor, horizon, arrays)
        report["cases"].append(case)
        print(
            json.dumps(
                {
                    "case": case["case"],
                    "base_status": case["base_oracle"]["status"],
                    "fit_accepted": case["fit_accepted"],
                    "attempts": len(case["fit_attempts"]),
                    "status_counts": case["oracle_status_counts_observed_during_fits"],
                }
            ),
            flush=True,
        )
    report["total_elapsed_s"] = perf_counter() - start
    report["code_sha256"] = {
        name: sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "cocofest/optimization/compact_muscle_prediction.py",
            "cocofest/optimization/local_endurance_value.py",
            "cocofest/optimization/adaptive_moment_rollout.py",
            "cocofest/optimization/rho_adaptive_moment_policy.py",
            "scripts/validate_local_endurance_value.py",
        )
    }
    report = _jsonable(report)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output / "validation_arrays.npz", **arrays)
    _plot(report, arrays, args.output)
    print(f"Report: {args.output / 'report.json'}", flush=True)
    return report


if __name__ == "__main__":
    main()
