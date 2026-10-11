#!/usr/bin/env python3
"""Render auditable projections of a local capacity-feasibility pilot.

The input is the JSON summary written by ``summarize_capacity_frontier_pilot``.
Only witnesses that pass every numerical gate are rendered.  The resulting
plots deliberately say *local pilot*, rather than Pareto front: a bounded,
non-convex support campaign is not a proof of global optimality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_COLOURS = {
    "uniform": "#202020",
    "axis": "#0072B2",
    "pair": "#009E73",
    "simplex": "#CC79A7",
    "epsilon": "#D55E00",
}
_MARKERS = {"uniform": "*", "axis": "o", "pair": "s", "simplex": "^", "epsilon": "D"}


def _kind(row: dict) -> str:
    identifier = row["id"]
    if "epsilon" in identifier:
        return "epsilon"
    if "axis" in identifier:
        return "axis"
    if "pair" in identifier:
        return "pair"
    if "simplex" in identifier:
        return "simplex"
    return "uniform"


def _accepted_unique(summary: dict, *, tolerance: float = 1e-5) -> list[dict]:
    """Return valid, display-unique rows while retaining source reports in JSON."""
    accepted = [row for row in summary["reports"] if row["numeric_gate_passed"]]
    displayed = []
    for row in accepted:
        capacity = np.asarray(row["capacity_ratio"], float)
        if not any(np.max(np.abs(capacity - np.asarray(other["capacity_ratio"], float))) <= tolerance
                   and _kind(row) == _kind(other) for other in displayed):
            displayed.append(row)
    return displayed


def _epsilon_index(row: dict) -> int | None:
    values = row.get("epsilon_upper")
    if values is None:
        return None
    finite = [index for index, value in enumerate(values) if value is not None]
    return finite[0] if len(finite) == 1 else None


def _style_axis(axis):
    axis.set_xlim(.15, 1.03)
    axis.set_ylim(.15, 1.03)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=.22, linewidth=.6)
    axis.axvline(.2, color="0.55", linestyle=":", linewidth=1, zorder=0)
    axis.axhline(.2, color="0.55", linestyle=":", linewidth=1, zorder=0)


def _scatter_tradeoffs(rows: list[dict], muscles: list[str], output: Path) -> Path:
    triceps = muscles.index("Triceps") if "Triceps" in muscles else len(muscles) - 1
    companions = [index for index in range(len(muscles)) if index != triceps]
    figure, axes = plt.subplots(1, len(companions), figsize=(4.4 * len(companions), 4.85))
    if len(companions) == 1:
        axes = [axes]
    legend = {}
    for axis, companion in zip(axes, companions):
        for row in rows:
            capacity = np.asarray(row["capacity_ratio"], float)
            kind = _kind(row)
            handle = axis.scatter(capacity[companion], capacity[triceps], s=50 if kind != "uniform" else 110,
                                  marker=_MARKERS[kind], color=_COLOURS[kind], edgecolor="white",
                                  linewidth=.6, alpha=.9, zorder=3, label=kind)
            legend.setdefault(kind, handle)
        _style_axis(axis)
        axis.set_xlabel(f"Capacité initiale {muscles[companion]} / repos")
        axis.set_ylabel(f"Capacité initiale {muscles[triceps]} / repos")
        axis.set_title(f"{muscles[companion]}–{muscles[triceps]}")
    # Epsilon paths are their own slices, not inferred Pareto edges.
    for constrained in companions:
        path = [row for row in rows if _kind(row) == "epsilon" and _epsilon_index(row) == constrained]
        if path:
            path.sort(key=lambda row: row["capacity_ratio"][constrained])
            capacities = np.asarray([row["capacity_ratio"] for row in path], float)
            axes[companions.index(constrained)].plot(capacities[:, constrained], capacities[:, triceps],
                                                     color=_COLOURS["epsilon"], linewidth=1.2, zorder=2)
    figure.subplots_adjust(top=.83, bottom=.27, wspace=.2)
    figure.legend(legend.values(), [name for name in legend], loc="lower center", bbox_to_anchor=(.5, .105),
                  ncol=len(legend), frameon=False)
    figure.suptitle("Pilot local c140, bras gauche — capacités initiales faisables pour un cycle", y=.965)
    figure.text(.5, .025, "Lignes pointillées : borne explorée A/Arest = 0.2. Les coupes ε sont des tranches, pas une preuve de front global.",
                ha="center", fontsize=8.5)
    path = output / "capacity_frontier_tradeoffs.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def _parallel_coordinates(rows: list[dict], muscles: list[str], output: Path) -> Path:
    figure, axis = plt.subplots(figsize=(9.2, 5.2), layout="constrained")
    x = np.arange(len(muscles))
    legend = {}
    for row in rows:
        kind = _kind(row)
        capacity = np.asarray(row["capacity_ratio"], float)
        line, = axis.plot(x, capacity, color=_COLOURS[kind], marker=_MARKERS[kind], linewidth=1.6,
                          markersize=5.5, alpha=.8 if kind != "uniform" else 1., label=kind)
        legend.setdefault(kind, line)
    axis.axhline(.2, color="0.4", linestyle=":", linewidth=1, label="borne explorée")
    axis.set_xticks(x, muscles)
    axis.set_ylim(.15, 1.03)
    axis.set_ylabel("Capacité initiale A / Arest")
    axis.grid(axis="y", alpha=.22, linewidth=.6)
    axis.set_title("Pilot local c140, bras gauche — profils de capacité témoins")
    axis.legend(legend.values(), [name for name in legend], loc="lower right", frameon=False)
    path = output / "capacity_frontier_parallel_coordinates.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def _epsilon_slices(rows: list[dict], muscles: list[str], output: Path) -> Path | None:
    epsilon = [row for row in rows if _kind(row) == "epsilon" and _epsilon_index(row) is not None]
    groups = sorted({_epsilon_index(row) for row in epsilon})
    if not groups:
        return None
    triceps = muscles.index("Triceps") if "Triceps" in muscles else len(muscles) - 1
    figure, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.8), layout="constrained")
    if len(groups) == 1:
        axes = [axes]
    for axis, index in zip(axes, groups):
        group = sorted((row for row in epsilon if _epsilon_index(row) == index),
                       key=lambda row: row["capacity_ratio"][index])
        constrained = np.asarray([row["capacity_ratio"][index] for row in group])
        tri = np.asarray([row["capacity_ratio"][triceps] for row in group])
        axis.plot(constrained, tri, marker="D", color=_COLOURS["epsilon"], linewidth=1.8)
        axis.set_xlabel(f"Borne/résultat {muscles[index]} / repos")
        axis.set_ylabel("Triceps / repos")
        axis.set_xlim(.35, 1.03)
        axis.set_ylim(.28, .45)
        axis.grid(alpha=.22, linewidth=.6)
        axis.set_title(f"Coupe ε sur {muscles[index]}")
    figure.suptitle("Tranches ε : compromis local lorsque le triceps est minimisé", y=1.03)
    path = output / "capacity_frontier_epsilon_slices.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def render(summary: dict, output_directory: Path) -> dict:
    if summary.get("kind") != "provisional_local_capacity_support_pilot_summary":
        raise ValueError("Expected a capacity-frontier pilot summary")
    rows = _accepted_unique(summary)
    if not rows:
        raise ValueError("Summary contains no valid witness to plot")
    output_directory.mkdir(parents=True, exist_ok=True)
    muscles = list(summary["muscle_order"])
    figures = [_scatter_tradeoffs(rows, muscles, output_directory),
               _parallel_coordinates(rows, muscles, output_directory)]
    epsilon = _epsilon_slices(rows, muscles, output_directory)
    if epsilon is not None:
        figures.append(epsilon)
    manifest = {
        "kind": "local_capacity_frontier_figure_manifest",
        "summary_kind": summary["kind"],
        "condition_sha256": summary["condition_sha256"],
        "rendered_witness_count": len(rows),
        "accepted_witness_count": summary["accepted_count"],
        "interpretation": "local numerical pilot, not a global Pareto proof or an online policy",
        "figures": [str(path.resolve()) for path in figures],
    }
    manifest_path = output_directory / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = json.loads(args.summary.resolve(strict=True).read_text(encoding="utf-8"))
    print(json.dumps(render(summary, args.output_directory.resolve()), indent=2))


if __name__ == "__main__":
    main()
