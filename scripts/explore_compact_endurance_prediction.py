"""Compare compact future PW prediction against the full Ding rollout, without FHO.

This numerical experiment does not run or modify any RHO controller. The
approximate PW sequence is independently replayed through full Ding dynamics;
its own zero total-moment residual is not used as a physical validation.
"""

from __future__ import annotations

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

from cocofest.optimization.adaptive_moment_rollout import (
    propagate_ding_pulse_width_interval, rollout_bounded_total_moment_reference_policy,
)
from cocofest.optimization.compact_muscle_prediction import (
    CompactMusclePredictor, fatigue_memory_coordinates,
)
from cocofest.optimization.rho_adaptive_moment_policy import build_rho_adaptive_moment_policy
from cocofest.optimization.rho_rollout_adapter import select_certified_rho_cycle


def replay_full_ding(initial, policy, widths, completed_intervals, substeps):
    count = len(policy.intervals)
    history = np.empty((completed_intervals + 1, len(policy.parameters), 5))
    moments = np.empty((completed_intervals, len(policy.parameters)))
    history[0] = initial
    current = initial.copy()
    for step in range(completed_intervals):
        cycle, k = divmod(step, count)
        interval = policy.intervals[k]
        for m, params in enumerate(policy.parameters):
            current[m] = propagate_ding_pulse_width_interval(
                current[m], pulse_width=widths[cycle, m, k], duration=interval.duration,
                calcium_amplitude=interval.calcium_amplitudes[m],
                mechanical_gain=interval.mechanical_gains[m], parameters=params,
                integration_substeps=substeps,
            )
        moments[step] = current[:, 1] * interval.moment_coefficients
        history[step + 1] = current
    return history, moments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("reduced_profile", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cycle-index", type=int, default=0)
    parser.add_argument("--cycle-period", type=float, default=1.)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--substeps", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--reference-substeps", type=int, default=16)
    parser.add_argument("--maximum-moment-error", type=float, default=1e-3)
    parser.add_argument("--maximum-normalized-slow-error", type=float, default=1e-3)
    args = parser.parse_args(argv)
    if args.cycles < 1 or args.reference_substeps < 1:
        parser.error("cycles and reference-substeps must be positive.")
    for value in (args.maximum_moment_error, args.maximum_normalized_slow_error):
        if not np.isfinite(value) or value <= 0:
            parser.error("Validation tolerances must be finite and positive.")
    policy = build_rho_adaptive_moment_policy(
        args.source, args.reduced_profile, cycle_index=args.cycle_index,
        cycle_period=args.cycle_period,
    )
    cycle, _ = select_certified_rho_cycle(
        args.source, cycle_index=args.cycle_index, cycle_period=args.cycle_period
    )
    initial = np.asarray([[cycle.states[f"{key}_{name}"][cycle.end_column]
                           for key in ("Cn", "F", "A", "Tau1", "Km")]
                          for name in policy.muscle_names])
    rest = np.asarray([p.fatigue.rest_state for p in policy.parameters])
    start = perf_counter()
    reference = rollout_bounded_total_moment_reference_policy(
        initial, intervals=policy.intervals, parameters=policy.parameters,
        horizon_cycles=args.cycles, integration_substeps=args.reference_substeps,
        moment_tolerance=1e-8,
    )
    reference_time = perf_counter() - start
    count = len(policy.intervals)
    target = np.asarray([np.sum(it.target_moments) for it in policy.intervals])
    report = {
        "schema": "cocofest-compact-endurance-exploration-v1",
        "source": {"path": str(args.source.resolve()), "sha256": sha256(args.source.read_bytes()).hexdigest()},
        "reduced_profile": {"path": str(args.reduced_profile.resolve()),
                            "sha256": sha256(args.reduced_profile.read_bytes()).hexdigest()},
        "source_cycle_index": args.cycle_index, "prediction_start": "source_cycle_terminal_boundary",
        "uses_fho_data": False, "requested_cycles": args.cycles,
        "model_parameters": [asdict(p) for p in policy.parameters],
        "muscle_names": policy.muscle_names,
        "source_context": {"signed_crank_torque_nm": cycle.metadata.get("signed_crank_torque_nm"),
                           "formulation": cycle.metadata.get("formulation"),
                           "mechanical_formulation": cycle.metadata.get("mechanical_formulation")},
        "assumptions": ["repeat source kinematics and total-moment target",
                        "reallocate muscle contributions every future phase",
                        "freeze slow states within each phase force equation",
                        "QP is a numerical policy outside the RHO NLP"],
        "thresholds": {"maximum_full_ding_replay_moment_error_nm": args.maximum_moment_error,
                       "maximum_normalized_slow_state_error": args.maximum_normalized_slow_error},
        "reference": {"status": reference.status, "completed_intervals": reference.completed_intervals,
                      "time_s": reference_time, "integration_substeps": args.reference_substeps,
                      "first_failure": reference.first_failure},
        "fatigue_coordinates": [], "candidates": [],
        "controller_validation": "not_run", "endurance_improvement": "not_established",
    }
    report["code_sha256"] = {
        name: sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("cocofest/optimization/compact_muscle_prediction.py",
                     "cocofest/optimization/adaptive_moment_rollout.py",
                     "cocofest/optimization/rho_adaptive_moment_policy.py",
                     "cocofest/optimization/smooth_muscle_moment_allocation.py",
                     "scripts/explore_compact_endurance_prediction.py")
    }
    for m, params in enumerate(policy.parameters):
        damage, offsets = fatigue_memory_coordinates(initial[m, 2:], params.fatigue)
        report["fatigue_coordinates"].append({"muscle": policy.muscle_names[m],
            "damage": damage, "offsets": offsets.tolist(),
            "normalized_offsets": (offsets / params.fatigue.rest_state[1:]).tolist()})
    plot_data = []
    for steps in args.substeps:
        start = perf_counter()
        predictor = CompactMusclePredictor(policy.intervals, policy.parameters, substeps=steps)
        preparation = perf_counter() - start
        start = perf_counter()
        prediction = predictor.rollout(initial, horizon_cycles=args.cycles)
        elapsed = perf_counter() - start
        n = prediction.completed_intervals
        replay_start = perf_counter()
        history, moments = replay_full_ding(
            initial, policy, prediction.pulse_widths, n, args.reference_substeps
        )
        replay_time = perf_counter() - replay_start
        error = np.sum(moments, axis=1) - np.tile(target, args.cycles)[:n]
        slow_error = (prediction.state_history[:n + 1, :, 2:] - history[:, :, 2:]) / rest
        maximum_error = float(np.max(np.abs(error))) if n else None
        maximum_slow_error = float(np.max(np.abs(slow_error))) if n else None
        entry = {"substeps": steps, "status": prediction.status,
                 "completed_intervals": n, "completed_cycles": prediction.completed_cycles,
                 "first_failure": prediction.first_failure,
                 "preparation_time_s": preparation, "prediction_time_s": elapsed,
                 "reference_time_ratio": reference_time / elapsed,
                 "ratio_comparable_full_horizons": reference.status == prediction.status == "complete",
                 "replay_time_s": replay_time, "maximum_replay_moment_error_nm": maximum_error,
                 "rmse_replay_moment_error_nm": float(np.sqrt(np.mean(error**2))) if n else None,
                 "maximum_normalized_slow_error": maximum_slow_error,
                 "numeric_gate_passed": bool(n == args.cycles * count and
                    maximum_error <= args.maximum_moment_error and
                    maximum_slow_error <= args.maximum_normalized_slow_error),
                 "adds_future_decision_variables_to_rho": 0}
        report["candidates"].append(entry)
        plot_data.append((steps, np.arange(1, n + 1) / count, error,
                          np.max(np.abs(slow_error), axis=(1, 2)), elapsed,
                          prediction.pulse_widths))
        print(json.dumps(entry), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    arrays = {"initial_states": initial, "reference_pulse_widths": reference.pulse_widths,
              "reference_state_history": reference.state_history}
    for steps, x, error, slow_error, _, widths in plot_data:
        for name, values in (("cycle_time", x), ("replay_moment_error", error),
                             ("normalized_slow_error", slow_error), ("pulse_widths", widths)):
            arrays[f"substeps_{steps}__{name}"] = values
    np.savez_compressed(args.output / "predictions.npz", **arrays)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for steps, x, error, slow_error, _, widths in plot_data:
        axes[0, 0].plot(x, error * 1000., label=f"{steps} sous-pas")
        axes[0, 1].plot(np.arange(len(slow_error)) / count, slow_error * 100., label=f"{steps} sous-pas")
        width_changes = np.abs(widths - widths[0:1]).transpose(0, 2, 1)
        axes[1, 1].plot(x, width_changes.reshape(-1, len(policy.parameters))[:len(x)].max(axis=1) * 1e6,
                        label=f"{steps} sous-pas")
    axes[0, 0].axhline(args.maximum_moment_error * 1000., color="k", linestyle=":")
    axes[0, 0].axhline(-args.maximum_moment_error * 1000., color="k", linestyle=":")
    axes[0, 0].set(ylabel="Erreur de moment total (mN·m)", title="PW prédites rejouées dans Ding complet")
    axes[0, 1].set(ylabel="Erreur lente max (% du repos)", title="Écart carte compacte / rejeu indépendant")
    axes[1, 0].bar(["Ding + inversions"] + [f"Compact S={x[0]}" for x in plot_data],
                   [reference_time] + [x[4] for x in plot_data])
    axes[1, 0].set(yscale="log", ylabel="Temps de prédiction (s, échelle log)",
                   title=f"Horizon demandé : {args.cycles} cycles")
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 1].set(ylabel="Écart PW max au 1er cycle prédit (µs)",
                   title="Adaptation des PW à phase identique")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.set_xlabel("Cycles futurs après le cycle source")
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Exploration numérique uniquement — aucun RHO prospectif ni FHO")
    fig.savefig(args.output / "comparison.png", dpi=170)
    plt.close(fig)
    print(f"Report: {args.output / 'report.json'}", flush=True)
    return report


if __name__ == "__main__":
    main()
