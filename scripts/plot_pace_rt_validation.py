#!/usr/bin/env python3
"""Render auditable figures from ``run_pace_rt_validation_campaign.py analyze``.

The figures describe numerical behaviour and certified prefixes.  They do
not label a stopped run as physiological fatigue unless a separate frozen-RHO
review has supplied that verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MUSCLES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")


def _conditions(document):
    return {row["condition"]: row for row in document["conditions"] if isinstance(row, dict)}


def _save(figure, path: Path):
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render(document: dict, output_directory: Path) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    conditions = _conditions(document)
    created = []

    names = list(conditions)
    values = [conditions[name].get("certified_pair_cycles") or 0 for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(names, values, color="#4c78a8")
    axes[0].set_ylabel("Certified bilateral RHO cycles")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("Certified prefix (review still required after a stop)")
    supervisor = [conditions[name].get("supervisor", {}).get("per_arm_compact_time", {}) or {} for name in names]
    timing = [item.get("mean_s", 0.) for item in supervisor]
    axes[1].bar(names, timing, color="#f58518")
    axes[1].set_ylabel("Mean compact rollout time per arm [s]")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_title("Asynchronous supervisor time")
    path = output_directory / "certified_cycles_and_supervisor_time.png"
    _save(fig, path); created.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, side in zip(axes, ("right", "left"), strict=True):
        for name, row in conditions.items():
            trace = row.get(side, {}).get("rho_solver_trace", [])
            if trace:
                axis.plot([x["cycle"] for x in trace], [x["solver_time_s"] for x in trace], label=name, linewidth=1.1)
        axis.set_title(f"{side.capitalize()} RHO")
        axis.set_xlabel("Cycle")
        axis.grid(alpha=.25)
    axes[0].set_ylabel("IPOPT RHO solve time [s]")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, fontsize=8)
    path = output_directory / "rho_solver_time_by_cycle.png"
    _save(fig, path); created.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, (name, row) in zip(axes.flat, conditions.items()):
        any_weight = False
        for side, style in (("right", "-"), ("left", "--")):
            trace = row.get(side, {}).get("weights", [])
            for muscle in MUSCLES:
                x = [item["cycle"] for item in trace if isinstance(item.get("weights"), list) and len(item["weights"]) == 4]
                y = [item["weights"][MUSCLES.index(muscle)] for item in trace if isinstance(item.get("weights"), list) and len(item["weights"]) == 4]
                if x:
                    axis.plot(x, y, linestyle=style, linewidth=1., label=f"{muscle} ({side[0].upper()})")
                    any_weight = True
        axis.set_title(name)
        axis.set_yscale("log")
        axis.grid(alpha=.25)
        if any_weight:
            axis.legend(fontsize=6, ncol=2)
    for axis in axes[-1]:
        axis.set_xlabel("Cycle")
    for axis in axes[:, 0]:
        axis.set_ylabel("Fatigue weight (log scale)")
    path = output_directory / "fatigue_weights_by_cycle.png"
    _save(fig, path); created.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    any_reserve = False
    for side, color in (("right", "#54a24b"), ("left", "#e45756")):
        trace = conditions.get("pace_rt", {}).get(side, {}).get("terminal_reserve_local_validation", [])
        if trace:
            cycles = [x["cycle"] for x in trace]
            axes[0].plot(cycles, [x.get("prediction_at_observed_terminal") for x in trace], color=color, label=side)
            axes[1].plot(cycles, [x.get("trust_fraction") for x in trace], color=color, label=side)
            any_reserve = True
    axes[0].set_title("PACE-RT local prediction at solved terminal")
    axes[0].set_ylabel("Compact local reserve score")
    axes[1].set_title("PACE-RT terminal-model domain check")
    axes[1].axhline(1., color="black", linestyle=":", linewidth=1, label="trust radius")
    axes[1].set_ylabel("Maximum normalized trust displacement")
    for axis in axes:
        axis.set_xlabel("Cycle")
        axis.grid(alpha=.25)
        if any_reserve:
            axis.legend()
    path = output_directory / "pace_rt_terminal_reserve_validation.png"
    _save(fig, path); created.append(path)
    return created


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args(argv)
    document = json.loads(args.comparison.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("conditions"), list):
        raise ValueError("comparison must be a PACE-RT validation analysis JSON")
    for path in render(document, args.output_directory):
        print(path)


if __name__ == "__main__":
    main()
