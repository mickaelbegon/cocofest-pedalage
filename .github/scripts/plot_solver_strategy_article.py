#!/usr/bin/env python3
"""Article-style comparison of exported IPOPT/MA57 and Acados trajectories.

Run directly in PyCharm (edit DEFAULT_* below), or use --help. Requires only
NumPy and Matplotlib, never imports Cocofest/Bioptim or a solver. Each PW sector
is one original zero-order-held control interval; controls are not interpolated.
The polar angle is elapsed cycle phase, clockwise from the top, not anatomical
crank angle. Outputs include vector SVG, PNG, numerical CSV and provenance JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from trajectory_comparison_common import capacity_scale, cycle_blocks, load_trajectory, state_boundaries


ROOT = SCRIPT_DIR.parents[1]
DEFAULT_IPOPT = ROOT / "solver-strategy-comparison-100-20260909/ipopt-radau5-reduced-isokinetic-torque-0.2-omega--6.28318530718-load--3-to-3/validated-rho-trajectory.npz"
DEFAULT_ACADOS = ROOT / "acados-optimality-fast-100-20260909/acados-irk-reduced-isokinetic-torque-0.2-omega--6.28318530718-load--3-to-3/validated-rho-trajectory.npz"
DEFAULT_OUTPUT = ROOT / "solver-strategy-article-figures"
DEFAULT_SELECTED_CYCLES = (1, 25, 50, 100)  # One-based; None selects four evenly spaced cycles.
DEFAULT_THRESHOLD_US = 150.0  # Descriptive threshold, not the model's activation threshold.
MUSCLES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
MUSCLE_LABELS = ("Anterior deltoid", "Posterior deltoid", "Biceps", "Triceps")
SOLVERS = ("IPOPT / MA57", "Acados")
COLORS = ("#2463a0", "#d06027")


def load_comparison(ipopt: Path, acados: Path, cycles: int | None) -> dict:
    """Validate and stack original PW bins and normalized capacity boundaries."""
    loaded = [load_trajectory(path) for path in (ipopt, acados)]
    available = [int(meta["cycles_per_window"]) for _, meta in loaded]
    cycles = min(available) if cycles is None else cycles
    if cycles < 1 or cycles > min(available):
        raise ValueError(f"cycles must be in [1, {min(available)}].")
    pw, capacity = [], []
    for (arrays, meta), exported in zip(loaded, available):
        if int(meta.get("stimulations_per_cycle", 0)) != 30:
            raise ValueError("This figure requires exactly 30 controls per cycle.")
        muscle_pw, muscle_capacity = [], []
        for muscle in MUSCLES:
            blocks = cycle_blocks(arrays[f"controls__last_pulse_width_{muscle}"],
                                  exported_cycles=exported, cycles=cycles, state=False)
            if blocks.shape != (cycles, 1, 30):
                raise ValueError(f"Unexpected PW shape for {muscle}: {blocks.shape}.")
            muscle_pw.append(blocks[:, 0, :] * 1e6)
            scale = capacity_scale(meta, muscle)
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid A_scale for {muscle}: {scale}.")
            muscle_capacity.append(state_boundaries(arrays[f"states__A_{muscle}"],
                                                   exported, cycles) / scale)
        pw.append(np.stack(muscle_pw, axis=1))
        capacity.append(np.stack(muscle_capacity, axis=1))
    pw, capacity = np.stack(pw), np.stack(capacity)
    if not np.all(np.isfinite(pw)) or not np.all(np.isfinite(capacity)):
        raise ValueError("Input trajectories contain non-finite values.")
    if np.min(pw) < 0:
        raise ValueError("Pulse widths must be nonnegative.")
    # Exact cycle-phase comparison requires the same isokinetic operating point.
    for key in ("formulation", "mechanical_formulation", "isokinetic_omega", "energy_equivalent_torque",
                "load_torque_min", "load_torque_max", "absolute_wheel_q_origin_reference",
                "absolute_wheel_q_start_cycle_index"):
        left, right = loaded[0][1].get(key), loaded[1][1].get(key)
        same = np.isclose(left, right) if isinstance(left, (int, float)) and isinstance(right, (int, float)) else left == right
        if not same:
            raise ValueError(f"Incompatible operating point: {key}: {left!r} vs {right!r}.")
    return {"pw": pw, "capacity": capacity, "cycles": cycles,
            "metadata": [meta for _, meta in loaded], "available_cycles": available}


def polar_style(ax) -> None:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]), ["0°", "90°", "180°", "270°"])
    ax.tick_params(axis="x", pad=3)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)


def save_figure(fig, output: Path, stem: str, dpi: int) -> list[str]:
    paths = []
    for extension in ("png", "svg"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_paired_rings(data: dict, selected: list[int], output: Path, dpi: int) -> list[str]:
    fig, axes = plt.subplots(2, len(selected), figsize=(3.8 * len(selected), 8.1),
                             subplot_kw={"projection": "polar"}, squeeze=False)
    fig.subplots_adjust(top=.80, bottom=.22, left=.055, right=.97, hspace=.40, wspace=.28)
    vmax = max(600.0, float(data["pw"].max()))
    norm = Normalize(0, vmax)
    cmap = plt.get_cmap("viridis")
    theta = np.linspace(0, 2 * np.pi, 31)
    for solver, row in enumerate(axes):
        for column, ax in enumerate(row):
            polar_style(ax)
            for muscle in range(4):
                ax.bar(theta[:-1], height=.72, width=np.diff(theta), bottom=1 + muscle,
                       align="edge", color=cmap(norm(data["pw"][solver, selected[column] - 1, muscle])),
                       linewidth=0, antialiased=False)
            ax.set_ylim(0, 4.85)
            ax.set_yticks([])
            if solver == 0:
                ax.set_title(f"Cycle {selected[column]}", pad=17, fontsize=11)
            if column == 0:
                ax.text(-.25, .5, SOLVERS[solver], transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=13, fontweight="bold")
    color_ax = fig.add_axes([.30, .105, .40, .025])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_ax, orientation="horizontal",
                 label="Pulse width (µs) — shared scale")
    fig.suptitle("Paired pulse-width strategies", y=.97, fontsize=18)
    fig.text(.5, .915, "Elapsed cycle phase, clockwise from 0°; each sector = one 12° control interval",
             ha="center", fontsize=11)
    fig.text(.5, .025, "Rings, centre → outside: anterior deltoid · posterior deltoid · biceps · triceps",
             ha="center", fontsize=10)
    return save_figure(fig, output, "01_paired_pw_rings", dpi)


def plot_delta_map(data: dict, output: Path, dpi: int) -> list[str]:
    delta = data["pw"][1] - data["pw"][0]
    limit = max(float(np.max(np.abs(delta))), 1e-9)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10.5), subplot_kw={"projection": "polar"})
    fig.subplots_adjust(left=.06, right=.94, bottom=.16, top=.81, hspace=.48, wspace=.30)
    theta = np.linspace(0, 2 * np.pi, 31)
    hole = max(3.0, data["cycles"] * .08)
    radius = hole + np.arange(data["cycles"] + 1)
    for muscle, ax in enumerate(axes.flat):
        polar_style(ax)
        ax.pcolormesh(theta, radius, delta[:, muscle], cmap="RdBu_r", norm=norm,
                      shading="flat", edgecolors="none", antialiased=False)
        ticks = np.unique(np.linspace(1, data["cycles"], 4).round().astype(int))
        ax.set_yticks(hole + ticks - .5, [str(cycle) for cycle in ticks], fontsize=8)
        ax.set_rlabel_position(45)
        for label in ax.get_yticklabels():
            label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": .4})
        ax.set_ylim(0, radius[-1])
        ax.set_title(MUSCLE_LABELS[muscle], pad=19)
    color_ax = fig.add_axes([.25, .075, .50, .023])
    fig.colorbar(ScalarMappable(norm=norm, cmap="RdBu_r"), cax=color_ax, orientation="horizontal",
                 label="Δ pulse width = Acados − IPOPT / MA57 (µs)")
    fig.suptitle("Evolution of pulse-width differences", y=.97, fontsize=18)
    fig.text(.5, .925, "Angle: elapsed cycle phase (clockwise) · radius: cycle, earliest inside",
             ha="center", fontsize=11)
    fig.text(.5, .015, "30 original ZOH sectors per cycle · one shared symmetric colour scale · no clipping",
             ha="center", fontsize=10)
    return save_figure(fig, output, "02_delta_pw_cycle_phase_polar", dpi)


def plot_capacity(data: dict, output: Path, dpi: int) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True, layout="constrained")
    x = np.arange(data["cycles"] + 1)
    for muscle, ax in enumerate(axes.flat):
        for solver in range(2):
            ax.plot(x, data["capacity"][solver, :, muscle], color=COLORS[solver],
                    ls=("-", "--")[solver], lw=1.8, label=SOLVERS[solver])
        ax.set_title(MUSCLE_LABELS[muscle])
        ax.grid(alpha=.2)
        ax.set_xlim(0, data["cycles"])
    axes[0, 0].legend(frameon=False)
    fig.supxlabel("Completed cycles (0 = initial state)")
    fig.supylabel("Normalized capacity, A / A_scale")
    fig.suptitle("Fatigue capacity at cycle boundaries — shared vertical scale", fontsize=16)
    return save_figure(fig, output, "03_normalized_capacity", dpi)


def activation_metrics(pw: np.ndarray, threshold: float) -> dict[str, float]:
    """Summarize a circular ZOH signal without bridging disjoint active windows.

    Onset and offset delimit the longest contiguous arc above the descriptive
    threshold; ties choose the first onset in [0, 360). Arc duration sums all
    active bins. Empty/full cycles have undefined onset and offset (NaN).
    """
    active = np.asarray(pw) > threshold
    n = active.size
    starts = np.flatnonzero(active & ~np.roll(active, 1))
    onset, offset = np.nan, np.nan
    longest = 360.0 if active.all() else 0.0
    if starts.size:
        lengths = [next((k for k in range(1, n + 1) if not active[(start + k) % n]), n)
                   for start in starts]
        best = int(np.argmax(lengths))
        onset = float(starts[best] * 360 / n)
        offset = float((starts[best] + lengths[best]) % n * 360 / n)
        longest = float(lengths[best] * 360 / n)
    return {"onset_deg": onset, "offset_deg": offset,
            "total_arc_deg": float(active.sum() * 360 / n),
            "longest_arc_deg": longest, "arc_count": int(starts.size) if not active.all() else 1,
            "mean_pw_us": float(np.mean(pw))}


def plot_decomposition(data: dict, threshold: float, output: Path, dpi: int) -> tuple[list[str], list[dict]]:
    keys = ("onset_deg", "offset_deg", "total_arc_deg", "mean_pw_us")
    titles = ("Onset, longest arc (°)", "Offset, longest arc (°)", "Total arc above threshold (°)", "Mean PW, whole cycle (µs)")
    metrics = np.full((2, data["cycles"], 4, 4), np.nan)
    rows = []
    for solver in range(2):
        for cycle in range(data["cycles"]):
            for muscle in range(4):
                result = activation_metrics(data["pw"][solver, cycle, muscle], threshold)
                metrics[solver, cycle, muscle] = [result[key] for key in keys]
                rows.append({"solver": SOLVERS[solver], "cycle": cycle + 1,
                             "muscle": MUSCLES[muscle], "threshold_us": threshold, **result})
    fig, axes = plt.subplots(4, 4, figsize=(15, 10), sharex=True)
    fig.subplots_adjust(left=.085, right=.98, bottom=.115, top=.86, hspace=.35, wspace=.28)
    for muscle in range(4):
        for metric, key in enumerate(keys):
            ax = axes[muscle, metric]
            for solver in range(2):
                # Markers preserve circular discontinuities without drawing a false wrap line.
                ax.plot(np.arange(1, data["cycles"] + 1), metrics[solver, :, muscle, metric],
                        color=COLORS[solver], ls="none" if metric < 2 else ("-", "--")[solver],
                        marker="." if metric < 2 else None, ms=2.6, lw=1.4, label=SOLVERS[solver])
            ax.grid(alpha=.18)
            ax.set_xlim(.5, data["cycles"] + .5)
            if metric < 3:
                ax.set_ylim(-12, 372)
                ax.set_yticks([0, 180, 360])
            else:
                ax.set_ylim(0, max(1, float(np.max(metrics[..., metric]))) * 1.05)
            if muscle == 0:
                ax.set_title(titles[metric], fontsize=11)
            if not np.isfinite(metrics[:, :, muscle, metric]).any():
                ax.set_axis_off()
        position = axes[muscle, 0].get_position()
        fig.text(.035, (position.y0 + position.y1) / 2, MUSCLE_LABELS[muscle],
                 ha="center", va="center", rotation=90)
    handles, labels = axes[0, 3].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .945), ncol=2, frameon=False)
    fig.suptitle(f"Descriptive stimulation-window decomposition · PW > {threshold:g} µs", y=.99, fontsize=17)
    fig.text(.5, .055, "Cycle", ha="center", fontsize=12)
    fig.text(.5, .013, "Onset/offset: longest circular arc; blank panels = undefined (empty/full cycles). Total arc sums all windows.\n"
             "Threshold defines a descriptive window, not physiological activation. Resolution: 12°; PW mean includes all 30 bins.",
             ha="center", fontsize=9)
    return save_figure(fig, output, "04_pw_window_decomposition", dpi), rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ipopt", type=Path, default=DEFAULT_IPOPT)
    parser.add_argument("--acados", type=Path, default=DEFAULT_ACADOS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cycles", type=int, default=None, help="Compare first N cycles; default: all common cycles.")
    parser.add_argument("--selected-cycles", type=int, nargs="+", default=None, help="One-based cycles for paired rings.")
    parser.add_argument("--threshold-us", type=float, default=DEFAULT_THRESHOLD_US)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args(argv)
    if not np.isfinite(args.threshold_us) or args.threshold_us < 0 or args.dpi < 1:
        parser.error("threshold-us must be finite and nonnegative; dpi must be positive.")
    data = load_comparison(args.ipopt.expanduser(), args.acados.expanduser(), args.cycles)
    selected = args.selected_cycles
    if selected is None:
        selected = [c for c in (DEFAULT_SELECTED_CYCLES or ()) if c <= data["cycles"]]
        if not selected or data["cycles"] < max(DEFAULT_SELECTED_CYCLES or (0,)):
            selected = np.unique(np.linspace(1, data["cycles"], min(4, data["cycles"])).round().astype(int)).tolist()
    if any(c < 1 or c > data["cycles"] for c in selected):
        parser.error(f"selected-cycles must be in [1, {data['cycles']}].")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    paths = plot_paired_rings(data, selected, output, args.dpi)
    paths += plot_delta_map(data, output, args.dpi)
    paths += plot_capacity(data, output, args.dpi)
    decomposition_paths, rows = plot_decomposition(data, args.threshold_us, output, args.dpi)
    paths += decomposition_paths
    csv_path = output / "pw_window_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: ("" if isinstance(value, float) and np.isnan(value) else value)
                         for key, value in row.items()} for row in rows)
    delta = data["pw"][1] - data["pw"][0]
    report = {"sources": [str(args.ipopt.expanduser().resolve()), str(args.acados.expanduser().resolve())],
              "labels": SOLVERS, "cycles": data["cycles"], "selected_cycles": selected,
              "controls_per_cycle": 30, "control_representation": "ZOH, original bins, no interpolation",
              "phase_convention": "elapsed cycle phase, clockwise from top; not anatomical crank angle",
              "capacity_normalization": "A / per-artifact, per-muscle A_scale, at cycle boundaries",
              "threshold_us": args.threshold_us, "threshold_interpretation": "descriptive, not physiological activation",
              "pw_delta_acados_minus_ipopt": {muscle: {"rmse_us": float(np.sqrt(np.mean(delta[:, i] ** 2))),
                                                       "maximum_absolute_us": float(np.max(np.abs(delta[:, i])))}
                                               for i, muscle in enumerate(MUSCLES)},
              "metadata": data["metadata"], "figures": paths, "metrics_csv": str(csv_path),
              "interpretation": "Exported trajectory comparison; different transcriptions and solver settings do not isolate a solver-only effect."}
    report_path = output / "figure_provenance.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "figures": paths, "provenance": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
