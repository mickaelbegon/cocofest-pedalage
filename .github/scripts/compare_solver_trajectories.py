#!/usr/bin/env python3
"""Compare IPOPT, MadNLP, and ACADOS RHO states and controls.

The defaults point to the local 500-RHO campaign, so the file can be run
directly from PyCharm. Command-line options remain available for other runs.
Only NPZ artifacts are consumed; Cocofest, Bioptim, and the solvers are not
imported.
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
    align_many_blocks,
    capacity_scale,
    cycle_blocks,
    error_metrics,
    load_trajectory,
    short_variable_name,
    state_boundaries,
    style_axis,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPOSITORY_ROOT / "radau5-500-20260901"
DEFAULT_PATHS = {
    "IPOPT R5": DEFAULT_RESULTS / "ipopt-radau5-500-reduced" / "validated-rho-trajectory.npz",
    "MadNLP R5": (
        DEFAULT_RESULTS
        / "madnlp-radau5-fatigue-endurance-500-reduced"
        / "validated-rho-trajectory.npz"
    ),
    "ACADOS IRK": (
        DEFAULT_RESULTS
        / "acados-sqp-irk-full-phase-one-mechanical"
        / "validated-rho-trajectory.npz"
    ),
}
COLORS = {"IPOPT R5": "#3366cc", "MadNLP R5": "#dc3912", "ACADOS IRK": "#109618"}
LINESTYLES = {"IPOPT R5": "-", "MadNLP R5": "--", "ACADOS IRK": "-."}
REFERENCE = "IPOPT R5"
COMPARATORS = ("MadNLP R5", "ACADOS IRK")
DING_STATES = ("Cn", "Tau1", "Km")


def _configuration_note(metadata: dict[str, dict]) -> str:
    """Describe the mechanics and transcription recorded by each artifact."""

    descriptions = []
    for name, values in metadata.items():
        mechanics = values.get("mechanical_formulation", "unknown mechanics")
        profile = values.get("producer_transcription_profile") or values.get(
            "ode_solver", "unknown transcription"
        )
        descriptions.append(f"{name}: {mechanics}/{profile}")
    return "; ".join(descriptions) + "."


def canonicalize_trajectory(
    arrays: dict[str, np.ndarray], metadata: dict
) -> dict[str, np.ndarray]:
    """Expose common FES variables and a canonical wheel angle/speed pair."""

    canonical = {
        key: np.asarray(values, dtype=float)
        for key, values in arrays.items()
        if key.startswith("controls__") or (key.startswith("states__") and key not in {"states__q", "states__qdot"})
    }
    if "states__theta" in arrays and "states__omega" in arrays:
        canonical["states__wheel_angle"] = np.asarray(arrays["states__theta"], dtype=float)
        canonical["states__wheel_speed"] = np.asarray(arrays["states__omega"], dtype=float)
    elif "states__q" in arrays and "states__qdot" in arrays:
        q = np.atleast_2d(np.asarray(arrays["states__q"], dtype=float))
        qdot = np.atleast_2d(np.asarray(arrays["states__qdot"], dtype=float))
        if q.shape[0] < 3 or qdot.shape[0] < 3:
            raise ValueError("Full mechanics must export the wheel as component 2 of q/qdot.")
        canonical["states__wheel_angle"] = q[2:3, :]
        canonical["states__wheel_speed"] = qdot[2:3, :]
    else:
        formulation = metadata.get("mechanical_formulation")
        raise ValueError(f"Cannot identify wheel states for {formulation!r} mechanics.")
    canonical.pop("states__theta", None)
    canonical.pop("states__omega", None)
    return canonical


def _required_keys() -> set[str]:
    keys = {"states__wheel_angle", "states__wheel_speed"}
    for muscle in MUSCLES:
        keys.add(f"controls__last_pulse_width_{muscle}")
        for state in ("Cn", "F", "A", "Tau1", "Km"):
            keys.add(f"states__{state}_{muscle}")
    return keys


def validate_compatibility(
    trajectories: dict[str, dict[str, np.ndarray]], metadata: dict[str, dict], cycles: int
) -> tuple[dict[str, int], list[str]]:
    exported_cycles = {
        name: int(values.get("cycles_per_window") or 0) for name, values in metadata.items()
    }
    unavailable = {name: count for name, count in exported_cycles.items() if count < cycles}
    if unavailable:
        raise ValueError(f"Requested {cycles} cycles, unavailable trajectory lengths: {unavailable}.")
    stimulation_counts = {
        int(values.get("stimulations_per_cycle") or 0) for values in metadata.values()
    }
    if len(stimulation_counts) != 1 or next(iter(stimulation_counts)) < 1:
        raise ValueError(f"Incompatible stimulations_per_cycle values: {stimulation_counts}.")
    required = _required_keys()
    missing = {name: sorted(required - set(arrays)) for name, arrays in trajectories.items()}
    missing = {name: keys for name, keys in missing.items() if keys}
    if missing:
        raise ValueError(f"Missing required variables: {missing}.")
    common_keys = sorted(set.intersection(*(set(arrays) for arrays in trajectories.values())))
    common_keys = [key for key in common_keys if key.startswith(("states__", "controls__"))]
    return exported_cycles, common_keys


def _cycle_matrix(blocks: np.ndarray) -> np.ndarray:
    return np.asarray(blocks, dtype=float).reshape(blocks.shape[0], -1)


def _comparison_values(
    key: str, blocks: dict[str, np.ndarray], metadata: dict[str, dict]
) -> tuple[dict[str, np.ndarray], str]:
    values = {name: _cycle_matrix(item) for name, item in blocks.items()}
    if key == "states__wheel_angle":
        values = {name: item - item[:, :1] for name, item in values.items()}
        return values, "rad (angle relatif au début du cycle)"
    if key.startswith("controls__"):
        return {name: item * 1e6 for name, item in values.items()}, "µs"
    if key.startswith("states__A_"):
        muscle = key.removeprefix("states__A_")
        return {
            name: item / capacity_scale(metadata[name], muscle) for name, item in values.items()
        }, "A/A_scale"
    units = "rad/s" if key == "states__wheel_speed" else "unité native"
    return values, units


def compute_comparison(
    raw_trajectories: dict[str, dict[str, np.ndarray]], metadata: dict[str, dict], cycles: int
) -> tuple[dict, dict]:
    trajectories = {
        name: canonicalize_trajectory(arrays, metadata[name])
        for name, arrays in raw_trajectories.items()
    }
    exported_cycles, common_keys = validate_compatibility(trajectories, metadata, cycles)
    aligned = {
        key: align_many_blocks(trajectories, exported_cycles, cycles, key) for key in common_keys
    }
    pairwise = {}
    per_cycle_normalized = {name: {} for name in COMPARATORS}
    for comparator in COMPARATORS:
        rows = {}
        for key in common_keys:
            values, units = _comparison_values(key, aligned[key], metadata)
            row = error_metrics(values[REFERENCE], values[comparator])
            row["unit"] = units
            if key.startswith("states__"):
                row["maximum_terminal_absolute_error"] = float(
                    np.max(np.abs(values[comparator][:, -1] - values[REFERENCE][:, -1]))
                )
            rows[key] = row
            delta = values[comparator] - values[REFERENCE]
            per_cycle_normalized[comparator][key] = (
                np.sqrt(np.mean(delta**2, axis=1)) / row["reference_scale"]
            )
        pairwise[f"{comparator} vs {REFERENCE}"] = rows

    headline = {}
    for comparator in COMPARATORS:
        control_deltas = []
        final_capacity_deltas = []
        for muscle in MUSCLES:
            pw_key = f"controls__last_pulse_width_{muscle}"
            pw, _ = _comparison_values(pw_key, aligned[pw_key], metadata)
            control_deltas.append(pw[comparator] - pw[REFERENCE])
            capacity_key = f"states__A_{muscle}"
            capacity, _ = _comparison_values(capacity_key, aligned[capacity_key], metadata)
            final_capacity_deltas.append(capacity[comparator][-1, -1] - capacity[REFERENCE][-1, -1])
        all_control_deltas = np.concatenate(control_deltas, axis=1)
        speed, _ = _comparison_values("states__wheel_speed", aligned["states__wheel_speed"], metadata)
        headline[comparator] = {
            "pulse_width_rmse_us": float(np.sqrt(np.mean(all_control_deltas**2))),
            "pulse_width_maximum_absolute_error_us": float(np.max(np.abs(all_control_deltas))),
            "maximum_final_capacity_ratio_difference": float(
                np.max(np.abs(final_capacity_deltas))
            ),
            "wheel_speed_rmse_rad_s": float(
                np.sqrt(np.mean((speed[comparator] - speed[REFERENCE]) ** 2))
            ),
        }

    summary = {
        "schema": "cocofest-three-solver-trajectory-comparison-v1",
        "cycles_compared": cycles,
        "reference": REFERENCE,
        "common_variables": common_keys,
        "common_variable_count": len(common_keys),
        "headline": headline,
        "pairwise": pairwise,
        "metadata": metadata,
        "comparison_note": _configuration_note(metadata),
    }
    plot_data = {
        "trajectories": trajectories,
        "exported_cycles": exported_cycles,
        "aligned": aligned,
        "common_keys": common_keys,
        "per_cycle_normalized": per_cycle_normalized,
        "metadata": metadata,
    }
    return summary, plot_data


def plot_capacities(data: dict, cycles: int, output: Path) -> None:
    x = np.arange(cycles + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, muscle in zip(axes.flat, MUSCLES):
        key = f"states__A_{muscle}"
        for name, arrays in data["trajectories"].items():
            values = state_boundaries(arrays[key], data["exported_cycles"][name], cycles)
            values = values / capacity_scale(data["metadata"][name], muscle)
            axis.plot(x, values, color=COLORS[name], linestyle=LINESTYLES[name], label=name)
        axis.set_title(muscle)
        axis.set_ylabel(r"Capacité $A/A_{scale}$")
        style_axis(axis)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("États de capacité musculaire aux frontières des cycles")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_forces(data: dict, cycles: int, output: Path) -> None:
    x = np.arange(1, cycles + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, muscle in zip(axes.flat, MUSCLES):
        key = f"states__F_{muscle}"
        for name, blocks in data["aligned"][key].items():
            values = _cycle_matrix(blocks)
            axis.plot(
                x, np.mean(values, axis=1), color=COLORS[name], linestyle=LINESTYLES[name], label=name
            )
            axis.fill_between(
                x, np.min(values, axis=1), np.max(values, axis=1), color=COLORS[name], alpha=0.08
            )
        axis.set_title(muscle)
        axis.set_ylabel("Force moyenne [min, max] (N)")
        style_axis(axis)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("États de force musculaire au fil des cycles")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_other_ding_states(data: dict, cycles: int, output: Path) -> None:
    x = np.arange(cycles + 1)
    fig, axes = plt.subplots(len(DING_STATES), len(MUSCLES), figsize=(16, 10), sharex=True)
    for row, state in enumerate(DING_STATES):
        for column, muscle in enumerate(MUSCLES):
            axis = axes[row, column]
            key = f"states__{state}_{muscle}"
            for name, arrays in data["trajectories"].items():
                values = state_boundaries(arrays[key], data["exported_cycles"][name], cycles)
                axis.plot(x, values, color=COLORS[name], linestyle=LINESTYLES[name], label=name)
            if row == 0:
                axis.set_title(muscle)
            if column == 0:
                axis.set_ylabel(state)
            style_axis(axis)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Autres états Ding aux frontières des cycles")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pulse_widths(data: dict, cycles: int, output: Path) -> None:
    x = np.arange(1, cycles + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, muscle in zip(axes.flat, MUSCLES):
        key = f"controls__last_pulse_width_{muscle}"
        for name, blocks in data["aligned"][key].items():
            values = _cycle_matrix(blocks) * 1e6
            axis.plot(
                x, np.mean(values, axis=1), color=COLORS[name], linestyle=LINESTYLES[name], label=name
            )
            axis.fill_between(
                x, np.min(values, axis=1), np.max(values, axis=1), color=COLORS[name], alpha=0.08
            )
        axis.set_title(muscle)
        axis.set_ylabel("PW moyenne [min, max] (µs)")
        style_axis(axis)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Contrôles de stimulation au fil des cycles")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_control_differences(data: dict, cycles: int, output: Path) -> None:
    fig, axes = plt.subplots(len(MUSCLES), len(COMPARATORS), figsize=(13, 10), sharex=True)
    for row, muscle in enumerate(MUSCLES):
        key = f"controls__last_pulse_width_{muscle}"
        values = {name: _cycle_matrix(blocks) * 1e6 for name, blocks in data["aligned"][key].items()}
        maximum = max(np.max(np.abs(values[name] - values[REFERENCE])) for name in COMPARATORS)
        maximum = max(float(maximum), 1.0)
        for column, name in enumerate(COMPARATORS):
            delta = values[name] - values[REFERENCE]
            image = axes[row, column].imshow(
                delta,
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                vmin=-maximum,
                vmax=maximum,
                extent=(0, 360, 1, cycles),
            )
            if row == 0:
                axes[row, column].set_title(f"{name} − {REFERENCE}")
            if column == 0:
                axes[row, column].set_ylabel(f"{muscle}\nCycle")
            if row == len(MUSCLES) - 1:
                axes[row, column].set_xlabel("Phase du pédalier (°)")
            fig.colorbar(image, ax=axes[row, column], label="ΔPW (µs)", pad=0.01)
    fig.suptitle("Écarts des contrôles par cycle et par phase")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mechanics(data: dict, cycles: int, output: Path) -> None:
    x = np.arange(1, cycles + 1)
    speed, _ = _comparison_values(
        "states__wheel_speed", data["aligned"]["states__wheel_speed"], data["metadata"]
    )
    angle, _ = _comparison_values(
        "states__wheel_angle", data["aligned"]["states__wheel_angle"], data["metadata"]
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for name in COLORS:
        axes[0, 0].plot(
            x, np.mean(speed[name], axis=1), color=COLORS[name], linestyle=LINESTYLES[name], label=name
        )
        axes[0, 1].plot(
            x, speed[name][:, -1], color=COLORS[name], linestyle=LINESTYLES[name], label=name
        )
        angle_closure_error = angle[name][:, -1] + 2.0 * np.pi
        axes[1, 0].plot(
            x, angle_closure_error, color=COLORS[name], linestyle=LINESTYLES[name], label=name
        )
    for name in COMPARATORS:
        rmse = np.sqrt(np.mean((speed[name] - speed[REFERENCE]) ** 2, axis=1))
        axes[1, 1].plot(x, rmse, color=COLORS[name], linestyle=LINESTYLES[name], label=name)
    axes[0, 0].set_title("Vitesse angulaire moyenne")
    axes[0, 0].set_ylabel("ω (rad/s)")
    axes[0, 1].set_title("Vitesse angulaire terminale")
    axes[0, 1].set_ylabel("ω terminale (rad/s)")
    axes[1, 0].set_title("Erreur de fermeture angulaire")
    axes[1, 0].set_ylabel(r"Δθ + 2π (rad)")
    axes[1, 1].set_title(f"Écart de vitesse par rapport à {REFERENCE}")
    axes[1, 1].set_ylabel("RMSE ω (rad/s)")
    for axis in axes.flat:
        style_axis(axis)
    axes[0, 0].legend(frameon=False)
    axes[1, 1].legend(frameon=False)
    fig.suptitle("États mécaniques communs après projection sur la roue")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_error_heatmaps(data: dict, cycles: int, output: Path) -> None:
    keys = data["common_keys"]
    fig_height = max(8.0, 0.30 * len(keys))
    fig, axes = plt.subplots(1, len(COMPARATORS), figsize=(16, fig_height), sharey=True)
    image = None
    for axis, name in zip(np.atleast_1d(axes), COMPARATORS):
        matrix = np.vstack([data["per_cycle_normalized"][name][key] for key in keys])
        image = axis.imshow(
            np.log10(np.maximum(matrix, 1e-16)), origin="upper", aspect="auto", cmap="magma"
        )
        axis.set_title(f"{name} vs {REFERENCE}")
        ticks = np.unique(np.linspace(0, cycles - 1, min(cycles, 11), dtype=int))
        axis.set_xticks(ticks, ticks + 1)
        axis.set_xlabel("Cycle")
    axes = np.atleast_1d(axes)
    axes[0].set_yticks(np.arange(len(keys)), [short_variable_name(key) for key in keys])
    if image is not None:
        colorbar = fig.colorbar(image, ax=list(axes), pad=0.01)
        colorbar.set_label("log10(RMSE normalisée)")
    fig.suptitle("Écarts des états et contrôles communs au fil des cycles")
    fig.subplots_adjust(left=0.20, right=0.90, bottom=0.08, top=0.94, wspace=0.08)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_selected_pulse_width_profiles(data: dict, cycles: int, output: Path) -> None:
    selected = sorted(
        set(max(1, min(cycles, int(round(value)))) for value in np.linspace(1, cycles, 4))
    )
    fig, axes = plt.subplots(
        len(MUSCLES), len(selected), figsize=(3.4 * len(selected), 10), sharex=True, sharey=True
    )
    axes = np.asarray(axes, dtype=object).reshape(len(MUSCLES), len(selected))
    for row, muscle in enumerate(MUSCLES):
        key = f"controls__last_pulse_width_{muscle}"
        for column, cycle in enumerate(selected):
            axis = axes[row, column]
            for name, blocks in data["aligned"][key].items():
                values = _cycle_matrix(blocks) * 1e6
                phase = np.arange(values.shape[1]) * 360.0 / values.shape[1]
                axis.step(
                    phase,
                    values[cycle - 1],
                    where="post",
                    color=COLORS[name],
                    linestyle=LINESTYLES[name],
                    linewidth=1.1,
                    label=name,
                )
            if row == 0:
                axis.set_title(f"Cycle {cycle}")
            if column == 0:
                axis.set_ylabel(f"{muscle}\nPW (µs)")
            if row == len(MUSCLES) - 1:
                axis.set_xlabel("Phase (°)")
            axis.grid(alpha=0.18)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Profils de contrôle à plusieurs jalons")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, summary: dict, figure_names: list[str]) -> None:
    lines = [
        "# Comparaison IPOPT–MadNLP–ACADOS",
        "",
        f"Comparaison de `{summary['cycles_compared']}` cycles exportés.",
        "",
        f"> {summary['comparison_note']}",
        "> Les états mécaniques sont comparés uniquement après projection sur l’angle et la vitesse de la roue.",
        "",
        "## Indicateurs principaux par rapport à IPOPT",
        "",
        "| Méthode | RMSE PW (µs) | Écart PW max (µs) | Écart capacité finale max | RMSE vitesse (rad/s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in summary["headline"].items():
        lines.append(
            f"| {name} | {row['pulse_width_rmse_us']:.6g} | "
            f"{row['pulse_width_maximum_absolute_error_us']:.6g} | "
            f"{row['maximum_final_capacity_ratio_difference']:.6g} | "
            f"{row['wheel_speed_rmse_rad_s']:.6g} |"
        )
    lines.extend(["", "## Figures", ""])
    for name in figure_names:
        title = name.removesuffix(".png").replace("_", " ").title()
        lines.extend([f"### {title}", "", f"![{name}]({name})", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--ipopt-trajectory", type=Path, default=DEFAULT_PATHS["IPOPT R5"])
    parser.add_argument("--madnlp-trajectory", type=Path, default=DEFAULT_PATHS["MadNLP R5"])
    parser.add_argument("--acados-trajectory", type=Path, default=DEFAULT_PATHS["ACADOS IRK"])
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RESULTS / "three-solver-trajectory-comparison"
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cycles < 1:
        raise ValueError("--cycles must be strictly positive.")
    paths = {
        "IPOPT R5": args.ipopt_trajectory.resolve(),
        "MadNLP R5": args.madnlp_trajectory.resolve(),
        "ACADOS IRK": args.acados_trajectory.resolve(),
    }
    raw_trajectories = {}
    metadata = {}
    for name, path in paths.items():
        raw_trajectories[name], metadata[name] = load_trajectory(path)
    summary, plot_data = compute_comparison(raw_trajectories, metadata, args.cycles)
    summary["trajectories"] = {name: str(path) for name, path in paths.items()}

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "capacity_states_over_cycles.png": plot_capacities,
        "force_states_over_cycles.png": plot_forces,
        "other_ding_states_over_cycles.png": plot_other_ding_states,
        "pulse_width_controls_over_cycles.png": plot_pulse_widths,
        "pulse_width_control_differences.png": plot_control_differences,
        "mechanical_states_over_cycles.png": plot_mechanics,
        "normalized_state_control_errors.png": plot_normalized_error_heatmaps,
        "selected_pulse_width_profiles.png": plot_selected_pulse_width_profiles,
    }
    for filename, renderer in figures.items():
        renderer(plot_data, args.cycles, output_dir / filename)
    summary_path = output_dir / "comparison.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_markdown(output_dir / "comparison.md", summary, list(figures))
    print(f"Cycles  : {args.cycles}")
    print(f"Figures : {len(figures)} in {output_dir}")
    print(f"Summary : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
