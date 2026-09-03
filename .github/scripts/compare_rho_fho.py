#!/usr/bin/env python3
"""Compare a concatenated RHO trajectory with a monolithic FHO solution.

The post-processor consumes only the JSON/NPZ artifacts produced by
``run_full_horizon_benchmark.py``.  It does not import Cocofest or Bioptim, so
it can run after a campaign in either solver environment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from trajectory_comparison_common import (  # noqa: E402
    MUSCLES,
    aligned_blocks as _aligned_blocks,
    artifact_path as _artifact_path,
    capacity_scale as _capacity_scale,
    cycle_blocks as _cycle_blocks,
    load_json as _load_json,
    load_trajectory,
    resample_blocks as _resample_blocks,
    short_variable_name as _short_variable_name,
    state_boundaries as _state_boundaries,
    style_axis as _style_axis,
)


COLORS = {"rho": "#3366cc", "fho": "#dc3912"}
LINESTYLES = {"rho": "-", "fho": "--"}


def _label(mode: str, cycles: int) -> str:
    return f"{cycles} RHO" if mode == "rho" else f"FHO {cycles}"


def discover_artifacts(
    report_path: Path, cycles: int
) -> tuple[Path, Path, dict, dict | None]:
    """Find the RHO seed and accepted FHO solution for ``cycles``."""

    report = _load_json(report_path)
    rho_record = report.get("rho") or {}
    if not rho_record.get("seed_path"):
        raise ValueError(f"{report_path} does not identify the concatenated RHO seed.")

    attempts = [
        attempt
        for attempt in report.get("full_horizon_attempts", [])
        if int(attempt.get("cycles") or 0) == cycles
        and attempt.get("success") is True
        and attempt.get("accepted_for_continuation", True) is True
        and attempt.get("solution_path")
    ]
    if not attempts:
        raise ValueError(
            f"No accepted FHO_{cycles} is recorded in {report_path}. "
            "Complete or resume the continuation first."
        )
    attempt = attempts[-1]
    return (
        _artifact_path(rho_record["seed_path"], report_path),
        _artifact_path(attempt["solution_path"], report_path),
        report,
        attempt,
    )


def _required_keys() -> set[str]:
    keys = {"states__theta", "states__omega"}
    for muscle in MUSCLES:
        keys.update(
            {
                f"states__A_{muscle}",
                f"states__F_{muscle}",
                f"controls__last_pulse_width_{muscle}",
            }
        )
    return keys


def validate_compatibility(
    rho: dict[str, np.ndarray],
    fho: dict[str, np.ndarray],
    rho_metadata: dict,
    fho_metadata: dict,
    cycles: int,
) -> tuple[int, int, list[str]]:
    rho_cycles = int(rho_metadata.get("cycles_per_window") or 0)
    fho_cycles = int(fho_metadata.get("cycles_per_window") or 0)
    if rho_cycles < cycles or fho_cycles < cycles:
        raise ValueError(
            f"Requested {cycles} cycles, but RHO/FHO contain "
            f"{rho_cycles}/{fho_cycles}."
        )
    missing_rho = sorted(_required_keys() - set(rho))
    missing_fho = sorted(_required_keys() - set(fho))
    if missing_rho or missing_fho:
        raise ValueError(
            f"Missing required variables (RHO={missing_rho}, FHO={missing_fho})."
        )
    for field in ("mechanical_formulation", "stimulations_per_cycle"):
        if rho_metadata.get(field) != fho_metadata.get(field):
            raise ValueError(
                f"RHO and FHO metadata disagree for {field}: "
                f"{rho_metadata.get(field)!r} != {fho_metadata.get(field)!r}."
            )
    common_keys = sorted(
        key
        for key in set(rho) & set(fho)
        if key.startswith(("states__", "controls__"))
    )
    return rho_cycles, fho_cycles, common_keys


def compute_comparison(
    rho: dict[str, np.ndarray],
    fho: dict[str, np.ndarray],
    rho_metadata: dict,
    fho_metadata: dict,
    cycles: int,
) -> tuple[dict, dict]:
    rho_cycles, fho_cycles, common_keys = validate_compatibility(
        rho, fho, rho_metadata, fho_metadata, cycles
    )
    per_variable = {}
    per_cycle = {}
    for key in common_keys:
        rho_blocks, fho_blocks = _aligned_blocks(
            rho, fho, rho_cycles, fho_cycles, cycles, key
        )
        delta = fho_blocks - rho_blocks
        axes = tuple(range(1, delta.ndim))
        cycle_rmse = np.sqrt(np.mean(delta**2, axis=axes))
        reference_span = float(np.ptp(rho_blocks))
        reference_rms = float(np.sqrt(np.mean(rho_blocks**2)))
        scale = max(reference_span, reference_rms, np.finfo(float).eps)
        row = {
            "rmse": float(np.sqrt(np.mean(delta**2))),
            "mae": float(np.mean(np.abs(delta))),
            "maximum_absolute_error": float(np.max(np.abs(delta))),
            "normalized_rmse": float(np.sqrt(np.mean(delta**2)) / scale),
            "reference_scale": scale,
        }
        if key.startswith("states__"):
            row["maximum_terminal_absolute_error"] = float(
                np.max(np.abs(delta[..., -1]))
            )
        per_variable[key] = row
        per_cycle[key] = cycle_rmse

    capacity = {}
    pulse_width = {}
    force = {}
    for muscle in MUSCLES:
        scale = _capacity_scale(rho_metadata, muscle)
        capacity_key = f"states__A_{muscle}"
        capacity[muscle] = {
            "rho": _state_boundaries(rho[capacity_key], rho_cycles, cycles) / scale,
            "fho": _state_boundaries(fho[capacity_key], fho_cycles, cycles) / scale,
        }
        pw_key = f"controls__last_pulse_width_{muscle}"
        rho_pw, fho_pw = _aligned_blocks(
            rho, fho, rho_cycles, fho_cycles, cycles, pw_key
        )
        pulse_width[muscle] = {
            "rho": rho_pw.reshape(cycles, -1) * 1e6,
            "fho": fho_pw.reshape(cycles, -1) * 1e6,
        }
        force_key = f"states__F_{muscle}"
        rho_force, fho_force = _aligned_blocks(
            rho, fho, rho_cycles, fho_cycles, cycles, force_key
        )
        force[muscle] = {
            "rho": rho_force.reshape(cycles, -1),
            "fho": fho_force.reshape(cycles, -1),
        }

    mechanics = {}
    for variable in ("theta", "omega"):
        key = f"states__{variable}"
        rho_values, fho_values = _aligned_blocks(
            rho, fho, rho_cycles, fho_cycles, cycles, key
        )
        mechanics[variable] = {
            "rho": rho_values.reshape(cycles, -1),
            "fho": fho_values.reshape(cycles, -1),
        }

    final_capacity_deltas = {
        muscle: float(values["fho"][-1] - values["rho"][-1])
        for muscle, values in capacity.items()
    }
    pw_delta = np.concatenate(
        [
            pulse_width[muscle]["fho"] - pulse_width[muscle]["rho"]
            for muscle in MUSCLES
        ],
        axis=1,
    )
    summary = {
        "cycles_compared": cycles,
        "rho_exported_cycles": rho_cycles,
        "fho_exported_cycles": fho_cycles,
        "common_variable_count": len(common_keys),
        "per_variable": per_variable,
        "headline": {
            "maximum_final_capacity_ratio_difference": float(
                max(abs(value) for value in final_capacity_deltas.values())
            ),
            "pulse_width_rmse_us": float(np.sqrt(np.mean(pw_delta**2))),
            "pulse_width_maximum_absolute_error_us": float(np.max(np.abs(pw_delta))),
            "omega_rmse_rad_s": per_variable["states__omega"]["rmse"],
            "theta_rmse_rad": per_variable["states__theta"]["rmse"],
        },
        "final_capacity_ratio_difference_fho_minus_rho": final_capacity_deltas,
    }
    plot_data = {
        "capacity": capacity,
        "pulse_width": pulse_width,
        "force": force,
        "mechanics": mechanics,
        "per_cycle": per_cycle,
        "per_variable": per_variable,
        "common_keys": common_keys,
    }
    return summary, plot_data


def plot_capacities(data: dict, cycles: int, output: Path) -> None:
    x = np.arange(cycles + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, muscle in zip(axes.flat, MUSCLES):
        for mode in ("rho", "fho"):
            ax.plot(
                x,
                data["capacity"][muscle][mode],
                color=COLORS[mode],
                linestyle=LINESTYLES[mode],
                linewidth=1.5,
                label=_label(mode, cycles),
            )
        ax.set_title(muscle)
        ax.set_ylabel(r"Capacité $A/A_{scale}$")
        _style_axis(ax)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Évolution de la capacité musculaire")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pulse_widths(data: dict, cycles: int, output: Path) -> None:
    x = np.arange(1, cycles + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, muscle in zip(axes.flat, MUSCLES):
        for mode in ("rho", "fho"):
            values = data["pulse_width"][muscle][mode]
            ax.plot(
                x,
                np.mean(values, axis=1),
                color=COLORS[mode],
                linestyle=LINESTYLES[mode],
                linewidth=1.4,
                label=_label(mode, cycles),
            )
            ax.fill_between(
                x,
                np.min(values, axis=1),
                np.max(values, axis=1),
                color=COLORS[mode],
                alpha=0.10,
            )
        ax.set_title(muscle)
        ax.set_ylabel("PW moyenne [min, max] (µs)")
        _style_axis(ax)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Largeurs d’impulsion au fil des cycles")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_forces(data: dict, cycles: int, output: Path) -> None:
    x = np.arange(1, cycles + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, muscle in zip(axes.flat, MUSCLES):
        for mode in ("rho", "fho"):
            values = data["force"][muscle][mode]
            ax.plot(
                x,
                np.mean(values, axis=1),
                color=COLORS[mode],
                linestyle=LINESTYLES[mode],
                linewidth=1.4,
                label=_label(mode, cycles),
            )
            ax.fill_between(
                x,
                np.min(values, axis=1),
                np.max(values, axis=1),
                color=COLORS[mode],
                alpha=0.10,
            )
        ax.set_title(muscle)
        ax.set_ylabel("Force moyenne [min, max] (N)")
        _style_axis(ax)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Forces musculaires au fil des cycles")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mechanics(data: dict, cycles: int, output: Path) -> None:
    x = np.arange(1, cycles + 1)
    theta_rho = data["mechanics"]["theta"]["rho"]
    theta_fho = data["mechanics"]["theta"]["fho"]
    omega_rho = data["mechanics"]["omega"]["rho"]
    omega_fho = data["mechanics"]["omega"]["fho"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for mode, values in (("rho", omega_rho), ("fho", omega_fho)):
        axes[0, 0].plot(
            x,
            np.mean(values, axis=1),
            color=COLORS[mode],
            linestyle=LINESTYLES[mode],
            label=_label(mode, cycles),
        )
        axes[0, 1].plot(
            x,
            values[:, -1],
            color=COLORS[mode],
            linestyle=LINESTYLES[mode],
            label=_label(mode, cycles),
        )
    axes[0, 0].set_title("Cadence moyenne")
    axes[0, 0].set_ylabel("ω (rad/s)")
    axes[0, 1].set_title("Cadence en fin de cycle")
    axes[0, 1].set_ylabel("ω terminal (rad/s)")
    axes[1, 0].plot(x, np.sqrt(np.mean((theta_fho - theta_rho) ** 2, axis=1)))
    axes[1, 0].set_title("Écart angulaire RHO–FHO")
    axes[1, 0].set_ylabel("RMSE θ (rad)")
    axes[1, 1].plot(x, np.sqrt(np.mean((omega_fho - omega_rho) ** 2, axis=1)))
    axes[1, 1].set_title("Écart de cadence RHO–FHO")
    axes[1, 1].set_ylabel("RMSE ω (rad/s)")
    for ax in axes.flat:
        _style_axis(ax)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Comparaison de la mécanique réduite")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_error_heatmap(data: dict, cycles: int, output: Path) -> None:
    keys = data["common_keys"]
    matrix = np.vstack(
        [
            data["per_cycle"][key] / data["per_variable"][key]["reference_scale"]
            for key in keys
        ]
    )
    log_matrix = np.log10(np.maximum(matrix, 1e-16))
    fig_height = max(7.0, 0.30 * len(keys))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    image = ax.imshow(log_matrix, origin="upper", aspect="auto", cmap="magma")
    ax.set_yticks(np.arange(len(keys)), [_short_variable_name(key) for key in keys])
    ticks = np.unique(np.linspace(0, cycles - 1, min(cycles, 11), dtype=int))
    ax.set_xticks(ticks, ticks + 1)
    ax.set_xlabel("Cycle")
    ax.set_title("Écart RHO–FHO par variable et par cycle")
    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("log10(RMSE normalisée)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_selected_pw_profiles(data: dict, cycles: int, output: Path) -> None:
    selected = sorted(
        set(
            max(1, min(cycles, int(round(value))))
            for value in np.linspace(1, cycles, min(cycles, 5))
        )
    )
    fig, axes = plt.subplots(
        len(MUSCLES), len(selected), figsize=(3.2 * len(selected), 10), sharex=True, sharey=True
    )
    axes = np.asarray(axes, dtype=object).reshape(len(MUSCLES), len(selected))
    for row, muscle in enumerate(MUSCLES):
        for column, cycle in enumerate(selected):
            ax = axes[row, column]
            count = data["pulse_width"][muscle]["rho"].shape[1]
            phase = np.arange(count) * 360.0 / count
            for mode in ("rho", "fho"):
                ax.step(
                    phase,
                    data["pulse_width"][muscle][mode][cycle - 1],
                    where="post",
                    color=COLORS[mode],
                    linestyle=LINESTYLES[mode],
                    linewidth=1.15,
                    label=_label(mode, cycles),
                )
            if row == 0:
                ax.set_title(f"Cycle {cycle}")
            if column == 0:
                ax.set_ylabel(f"{muscle}\nPW (µs)")
            if row == len(MUSCLES) - 1:
                ax.set_xlabel("Phase (°)")
            ax.grid(alpha=0.18)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Profils de stimulation à plusieurs jalons")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def timing_metrics(report: dict | None, target_attempt: dict | None, cycles: int) -> dict | None:
    if report is None or target_attempt is None:
        return None
    rho_elapsed = (report.get("rho") or {}).get("elapsed_s")
    target_elapsed = target_attempt.get("elapsed_s")
    attempts = [
        attempt
        for attempt in report.get("full_horizon_attempts", [])
        if int(attempt.get("cycles") or 0) <= cycles
    ]
    extensions = [
        attempt
        for attempt in report.get("extension_rho_attempts", [])
        if int(attempt.get("target_cycle") or 0) <= cycles
    ]
    if rho_elapsed is None or target_elapsed is None:
        return None
    return {
        "rho_reference_elapsed_s": float(rho_elapsed),
        "fho_target_solve_elapsed_s": float(target_elapsed),
        "iterative_fho_construction_elapsed_s": float(
            sum(float(row.get("elapsed_s") or 0.0) for row in (*attempts, *extensions))
        ),
        "full_horizon_attempt_count": len(attempts),
        "extension_rho_attempt_count": len(extensions),
    }


def plot_timing(timing: dict, cycles: int, output: Path) -> None:
    labels = [f"Chaîne {cycles} RHO", f"Solve FHO_{cycles}", "Construction FHO itérative"]
    values = [
        timing["rho_reference_elapsed_s"],
        timing["fho_target_solve_elapsed_s"],
        timing["iterative_fho_construction_elapsed_s"],
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=(COLORS["rho"], COLORS["fho"], "#7b4ab5"))
    ax.bar_label(bars, fmt="%.1f s", padding=3)
    ax.set_ylabel("Temps mur-à-mur (s)")
    ax.set_title("Coût calculatoire observé")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, summary: dict, figure_names: list[str]) -> None:
    headline = summary["headline"]
    lines = [
        f"# Comparaison {summary['cycles_compared']} RHO vs FHO_{summary['cycles_compared']}",
        "",
        (
            "Cette comparaison porte sur les trajectoires primales exportées, "
            "alignées cycle par cycle."
        ),
        "",
        (
            "- Écart maximal de capacité finale : "
            f"`{100 * headline['maximum_final_capacity_ratio_difference']:.6g}` "
            "points de pourcentage"
        ),
        f"- RMSE globale des largeurs d’impulsion : `{headline['pulse_width_rmse_us']:.6g} µs`",
        (
            "- Écart maximal des largeurs d’impulsion : "
            f"`{headline['pulse_width_maximum_absolute_error_us']:.6g} µs`"
        ),
        f"- RMSE globale de cadence : `{headline['omega_rmse_rad_s']:.6g} rad/s`",
        f"- RMSE globale d’angle : `{headline['theta_rmse_rad']:.6g} rad`",
        "",
        "## Figures",
        "",
    ]
    for name in figure_names:
        title = name.removesuffix(".png").replace("_", " ").title()
        lines.extend([f"### {title}", "", f"![{name}]({name})", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("full-horizon-results"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--rho-solution", type=Path)
    parser.add_argument("--fho-solution", type=Path)
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cycles < 1:
        raise ValueError("--cycles must be strictly positive.")
    report_path = (args.report or args.results_dir / "full-horizon-report.json").resolve()
    report = None
    target_attempt = None
    if args.rho_solution is None or args.fho_solution is None:
        rho_path, fho_path, report, target_attempt = discover_artifacts(
            report_path, args.cycles
        )
        rho_path = args.rho_solution.resolve() if args.rho_solution else rho_path
        fho_path = args.fho_solution.resolve() if args.fho_solution else fho_path
    else:
        rho_path = args.rho_solution.resolve()
        fho_path = args.fho_solution.resolve()
        if report_path.is_file():
            report = _load_json(report_path)
            matching = [
                row
                for row in report.get("full_horizon_attempts", [])
                if int(row.get("cycles") or 0) == args.cycles and row.get("success")
            ]
            target_attempt = matching[-1] if matching else None

    output_dir = (
        args.output_dir
        or args.results_dir / f"rho-vs-fho-{args.cycles:04d}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rho, rho_metadata = load_trajectory(rho_path)
    fho, fho_metadata = load_trajectory(fho_path)
    summary, plot_data = compute_comparison(
        rho, fho, rho_metadata, fho_metadata, args.cycles
    )
    summary.update(
        {
            "schema": "cocofest-rho-fho-comparison-v1",
            "rho_solution": str(rho_path),
            "fho_solution": str(fho_path),
            "rho_metadata": rho_metadata,
            "fho_metadata": fho_metadata,
        }
    )
    timing = timing_metrics(report, target_attempt, args.cycles)
    summary["timing"] = timing

    figures = {
        "capacity_over_cycles.png": lambda path: plot_capacities(plot_data, args.cycles, path),
        "pulse_width_over_cycles.png": lambda path: plot_pulse_widths(
            plot_data, args.cycles, path
        ),
        "force_over_cycles.png": lambda path: plot_forces(plot_data, args.cycles, path),
        "mechanics_over_cycles.png": lambda path: plot_mechanics(plot_data, args.cycles, path),
        "normalized_error_heatmap.png": lambda path: plot_error_heatmap(
            plot_data, args.cycles, path
        ),
        "selected_pulse_width_profiles.png": lambda path: plot_selected_pw_profiles(
            plot_data, args.cycles, path
        ),
    }
    if timing is not None:
        figures["timing_comparison.png"] = lambda path: plot_timing(timing, args.cycles, path)
    for name, renderer in figures.items():
        renderer(output_dir / name)

    summary_path = output_dir / "comparison.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "comparison.md", summary, list(figures))
    print(f"RHO solution : {rho_path}")
    print(f"FHO solution : {fho_path}")
    print(f"Cycles       : {args.cycles}")
    print(f"Figures      : {len(figures)} in {output_dir}")
    print(f"Summary      : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
