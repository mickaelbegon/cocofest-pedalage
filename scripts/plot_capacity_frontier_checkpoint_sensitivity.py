#!/usr/bin/env python3
"""Plot the same local capacity-support directions at several checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> tuple[int, dict]:
    path = path.resolve(strict=True)
    match = re.search(r"-c(\d+)-", path.name)
    if match is None:
        raise ValueError(f"Cannot infer checkpoint cycle from {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "fresh_discrete_rebuild_and_DOP853_completed_not_global_front_certified":
        raise ValueError(f"{path}: independent reconstruction/DOP853 incomplete")
    rebuild = value["independent_rebuild"]
    if not (rebuild.get("discrete_audit_passed") and rebuild.get("continuous_DOP853_replay") == "complete"):
        raise ValueError(f"{path}: numerical gate incomplete")
    return int(match.group(1)), value


def render(paths: list[Path], output: Path) -> Path:
    runs = [_load(path) for path in paths]
    muscles = runs[0][1]["muscle_order"]
    if any(value["muscle_order"] != muscles for _, value in runs):
        raise ValueError("Muscle orders differ")
    groups: dict[str, list[tuple[int, np.ndarray]]] = {"uniform": [], "triceps": []}
    for cycle, value in runs:
        support = np.asarray(value["support"], float)
        if np.allclose(support, .25):
            group = "uniform"
        elif np.argmax(support) == len(muscles) - 1 and np.count_nonzero(support) == 1:
            group = "triceps"
        else:
            raise ValueError("Expected only uniform or triceps-axis supports")
        groups[group].append((cycle, np.asarray(value["support_smoke"]["initial_capacity_ratio"], float)))
    if any(len(group) < 2 for group in groups.values()):
        raise ValueError("At least two checkpoints are required for both directions")
    figure, axes = plt.subplots(1, 2, figsize=(10.3, 4.1), sharey=True, layout="constrained")
    palette = plt.get_cmap("tab10").colors
    for axis, (group, title) in zip(axes, (("uniform", "Support uniforme"),
                                           ("triceps", "Minimisation du triceps"))):
        ordered = sorted(groups[group])
        cycles = np.asarray([item[0] for item in ordered])
        capacity = np.asarray([item[1] for item in ordered])
        for index, muscle in enumerate(muscles):
            axis.plot(cycles, capacity[:, index], marker="o", linewidth=1.8,
                      color=palette[index], label=muscle)
        axis.set_title(title)
        axis.set_xlabel("Cycle du checkpoint")
        axis.set_xticks(cycles)
        axis.set_ylim(.05, 1.03)
        axis.grid(alpha=.25, linewidth=.6)
        axis.axhline(.1, color="0.35", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Capacité minimale initiale A / Arest")
    axes[1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, .5))
    figure.suptitle("Réplication multi-checkpoints de deux témoins locaux (bras gauche)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(render(args.report, args.output.resolve()))


if __name__ == "__main__":
    main()
