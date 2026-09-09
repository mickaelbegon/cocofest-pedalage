#!/usr/bin/env python3
"""Predict future PWs that preserve one RHO cycle's individual muscle moments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cocofest-matplotlib")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from cocofest.optimization.adaptive_moment_rollout import (
    rollout_adaptive_moment_policy,
    rollout_fixed_pulse_width_policy,
)
from cocofest.optimization.rho_adaptive_moment_policy import (
    build_rho_adaptive_moment_policy,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use one certified RHO cycle as an individual-muscle moment policy, "
            "then predict bounded PWs with the full Ding state over many cycles."
        )
    )
    parser.add_argument("source", type=Path, help="Certified RHO trajectory NPZ.")
    parser.add_argument("reduced_profile", type=Path, help="Reduced cycling profile NPZ.")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--cycles", type=int, default=150)
    parser.add_argument("--source-cycle-index", type=int, default=0)
    parser.add_argument(
        "--cycle-period",
        type=float,
        default=None,
        help="Cycle duration in seconds; required when absent from source metadata.",
    )
    parser.add_argument("--integration-substeps", type=int, default=8)
    parser.add_argument("--moment-tolerance", type=float, default=1e-4)
    parser.add_argument("--model-path", type=Path, default=None)
    return parser


def _cycle_rmse(values: np.ndarray, targets: np.ndarray) -> list[float | None]:
    errors = values - targets[None, :, :]
    return [
        None if not np.any(np.isfinite(cycle)) else float(np.sqrt(np.nanmean(cycle**2)))
        for cycle in errors
    ]


def _plot_pulse_widths(output: Path, names: tuple[str, ...], adaptive, source) -> None:
    cycles = np.arange(1, adaptive.shape[0] + 1)
    figure, axes = plt.subplots(len(names), 1, figsize=(10, 2.5 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    for muscle_index, (name, axis) in enumerate(zip(names, axes, strict=True)):
        values = adaptive[:, muscle_index, :] * 1e6
        valid = np.any(np.isfinite(values), axis=1)
        axis.fill_between(
            cycles[valid],
            np.nanmin(values[valid], axis=1),
            np.nanmax(values[valid], axis=1),
            alpha=0.2,
        )
        axis.plot(
            cycles[valid],
            np.nanmean(values[valid], axis=1),
            label="PW adaptative moyenne",
        )
        axis.axhline(np.mean(source[muscle_index]) * 1e6, color="black", ls="--", label="RHO répété")
        axis.set_ylabel(f"{name}\nPW (µs)")
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Cycle prédit")
    figure.tight_layout()
    figure.savefig(output / "pulse-width-evolution.png", dpi=180)
    plt.close(figure)


def _plot_tracking_error(output: Path, targets: np.ndarray, adaptive, fixed) -> None:
    adaptive_rmse = np.asarray(
        [np.nan if value is None else value for value in _cycle_rmse(adaptive, targets)]
    )
    fixed_rmse = np.sqrt(np.mean((fixed - targets[None, :, :]) ** 2, axis=(1, 2)))
    cycles = np.arange(1, fixed.shape[0] + 1)
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.semilogy(cycles, np.maximum(adaptive_rmse, 1e-15), label="PW adaptatives")
    axis.semilogy(cycles, np.maximum(fixed_rmse, 1e-15), label="PW RHO répétées")
    axis.set(xlabel="Cycle prédit", ylabel="RMSE du moment musculaire (N·m)")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "moment-tracking-error.png", dpi=180)
    plt.close(figure)


def _plot_capacity(output: Path, names: tuple[str, ...], policy, adaptive, fixed) -> None:
    interval_count = len(policy.intervals)
    cycle_nodes = np.arange(0, adaptive.shape[0], interval_count)
    cycles = np.arange(cycle_nodes.size)
    figure, axes = plt.subplots(len(names), 1, figsize=(10, 2.5 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    for muscle_index, (name, axis) in enumerate(zip(names, axes, strict=True)):
        rest = policy.parameters[muscle_index].fatigue.a_rest
        axis.plot(cycles, adaptive[cycle_nodes, muscle_index, 2] / rest, label="PW adaptatives")
        axis.plot(cycles, fixed[cycle_nodes, muscle_index, 2] / rest, ls="--", label="PW RHO répétées")
        axis.set_ylabel(f"{name}\nA/A_rest")
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Début du cycle")
    figure.tight_layout()
    figure.savefig(output / "capacity-evolution.png", dpi=180)
    plt.close(figure)


def main() -> int:
    arguments = _parser().parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    policy = build_rho_adaptive_moment_policy(
        arguments.source,
        arguments.reduced_profile,
        cycle_index=arguments.source_cycle_index,
        cycle_period=arguments.cycle_period,
        model_path=arguments.model_path,
    )

    started = perf_counter()
    adaptive = rollout_adaptive_moment_policy(
        policy.initial_states,
        intervals=policy.intervals,
        parameters=policy.parameters,
        horizon_cycles=arguments.cycles,
        integration_substeps=arguments.integration_substeps,
        moment_tolerance=arguments.moment_tolerance,
    )
    adaptive_seconds = perf_counter() - started
    started = perf_counter()
    bounded_source_pulse_widths = policy.source_pulse_widths.copy()
    for muscle_index, parameters in enumerate(policy.parameters):
        bounded_source_pulse_widths[muscle_index] = np.clip(
            bounded_source_pulse_widths[muscle_index],
            parameters.pd0,
            parameters.pulse_width_max,
        )
    baseline_projection = bounded_source_pulse_widths - policy.source_pulse_widths
    if np.max(np.abs(baseline_projection)) > 1e-10:
        raise ValueError(
            "The source RHO pulse widths violate the Ding bounds by more than 1e-10 s."
        )
    fixed = rollout_fixed_pulse_width_policy(
        policy.initial_states,
        intervals=policy.intervals,
        parameters=policy.parameters,
        pulse_widths=bounded_source_pulse_widths,
        horizon_cycles=arguments.cycles,
        integration_substeps=arguments.integration_substeps,
    )
    fixed_seconds = perf_counter() - started

    first_cycle_error_us = (adaptive.pulse_widths[0] - policy.source_pulse_widths) * 1e6
    finite_first_cycle = np.isfinite(first_cycle_error_us)
    report = {
        "schema": "cocofest-adaptive-muscle-moment-policy-v1",
        "method": "bounded_scalar_pw_inverse_full_ding_periodic_node",
        "uses_fho_data": False,
        "clips_infeasible_pw": False,
        "fixed_baseline_source_bound_projection_count": int(
            np.count_nonzero(baseline_projection)
        ),
        "fixed_baseline_source_bound_projection_maximum_absolute_s": float(
            np.max(np.abs(baseline_projection))
        ),
        "source": str(arguments.source.resolve()),
        "reduced_profile": str(arguments.reduced_profile.resolve()),
        "source_cycle_index": policy.source_cycle_index,
        "source_cycle_count": policy.source_cycle_count,
        "certification_basis": policy.certification_basis,
        "period_s": policy.period,
        "period_basis": policy.period_basis,
        "muscle_names": list(policy.muscle_names),
        "requested_cycles": arguments.cycles,
        "completed_cycles": adaptive.completed_cycles,
        "completed_intervals": adaptive.completed_intervals,
        "status": adaptive.status,
        "first_failure": adaptive.first_failure,
        "integration_substeps": arguments.integration_substeps,
        "moment_tolerance_nm": arguments.moment_tolerance,
        "scalar_function_evaluations": adaptive.scalar_function_evaluations,
        "adaptive_runtime_s": adaptive_seconds,
        "fixed_runtime_s": fixed_seconds,
        "first_cycle_pw_reconstruction_rmse_us": (
            float(np.sqrt(np.mean(first_cycle_error_us[finite_first_cycle] ** 2)))
            if np.any(finite_first_cycle)
            else None
        ),
        "first_cycle_pw_reconstruction_maximum_absolute_error_us": (
            float(np.max(np.abs(first_cycle_error_us[finite_first_cycle])))
            if np.any(finite_first_cycle)
            else None
        ),
        "first_cycle_pw_reconstruction_sample_count": int(
            np.count_nonzero(finite_first_cycle)
        ),
        "adaptive_moment_rmse_nm_by_cycle": _cycle_rmse(
            adaptive.achieved_moments, policy.target_moments
        ),
        "fixed_pw_moment_rmse_nm_by_cycle": _cycle_rmse(
            fixed.achieved_moments, policy.target_moments
        ),
    }
    (arguments.output_directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        arguments.output_directory / "rollout.npz",
        muscle_names=np.asarray(policy.muscle_names),
        source_pulse_widths=policy.source_pulse_widths,
        fixed_bounded_source_pulse_widths=bounded_source_pulse_widths,
        target_moments=policy.target_moments,
        adaptive_pulse_widths=adaptive.pulse_widths,
        adaptive_achieved_moments=adaptive.achieved_moments,
        adaptive_state_history=adaptive.state_history,
        fixed_pulse_widths=fixed.pulse_widths,
        fixed_achieved_moments=fixed.achieved_moments,
        fixed_state_history=fixed.state_history,
        metadata__json=np.asarray(json.dumps(report, sort_keys=True)),
    )
    _plot_pulse_widths(
        arguments.output_directory,
        policy.muscle_names,
        adaptive.pulse_widths,
        policy.source_pulse_widths,
    )
    _plot_tracking_error(
        arguments.output_directory,
        policy.target_moments,
        adaptive.achieved_moments,
        fixed.achieved_moments,
    )
    _plot_capacity(
        arguments.output_directory,
        policy.muscle_names,
        policy,
        adaptive.state_history,
        fixed.state_history,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if adaptive.status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
