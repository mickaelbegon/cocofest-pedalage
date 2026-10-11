#!/usr/bin/env python3
"""Plot all exported certified endurance fatigue trajectories."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cocofest")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "local-results/isokinetic-prescribed-unit-1p3-pw2-20261001/analysis"
PER_CYCLE_FHO_DIR = ROOT / "local-results/isokinetic-prescribed-unit-1p3-fho-from-rho-bo-per-cycle-radau5-20261002/fho-continuation"


def latest_operational_fho(directory: Path) -> tuple[int, Path]:
    """Return the highest promoted FHO, never a partial attempt in progress."""
    candidates: list[tuple[int, Path]] = []
    for provenance in directory.glob("fho-????-continuation-provenance.json"):
        try:
            record = json.loads(provenance.read_text(encoding="utf-8"))
            cycles = int(record["fho_cycles"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        solution = directory / f"fho-{cycles:04d}-solution.npz"
        if record.get("operational_continuation_accepted") and solution.is_file():
            candidates.append((cycles, solution))
    if not candidates:
        raise FileNotFoundError(f"No promoted operational FHO found in {directory}.")
    return max(candidates)


PER_CYCLE_FHO_CYCLES, PER_CYCLE_FHO = latest_operational_fho(PER_CYCLE_FHO_DIR)
SOURCES = {
    "RHO min-fatigue": (ROOT / "local-results/isokinetic-prescribed-unit-1p3-fho-bo64-20260930/rho-seed/validated-rho-trajectory.npz", "#4c78a8", "--"),
    "RHO_BO min-fatigue": (OUT / "replay-rho-bo-min-fatigue/trajectory.npz", "#1f4e79", "-"),
    f"FHO depuis RHO_BO (par cycle, {PER_CYCLE_FHO_CYCLES})": (PER_CYCLE_FHO, "#2ca02c", "-"),
    "RHO min-PW²": (ROOT / "local-results/isokinetic-prescribed-unit-1p3-pw2-20261001/rho-unit-pw/validated-rho-trajectory.npz", "#e45756", "--"),
    "RHO_BO min-PW²": (OUT / "replay-rho-bo-pw/trajectory.npz", "#b23a48", "-"),
}
MUSCLES = ("Biceps", "Delt_ant", "Delt_post", "Triceps")


def boundary(path: Path, muscle: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        info = json.loads(str(data["metadata__json"]))
        cycles = int(info["cycles_per_window"])
        scale = float(info.get("fatigue_capacity_scales", {}).get(
            f"A_{muscle}", info["configured_muscle_parameters"][muscle]["a_scale"]
        ))
        states = np.asarray(data[f"states__A_{muscle}"], dtype=float).reshape(-1)
    step, remainder = divmod(states.size - 1, cycles)
    if remainder:
        raise ValueError(f"Invalid A-state layout: {path}")
    return np.arange(cycles + 1), 1.0 - states[::step] / scale


def main() -> None:
    missing = [str(path) for path, _, _ in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing trajectory archive(s): " + ", ".join(missing))
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.5), sharex=True, sharey=True, constrained_layout=True)
    summary: dict[str, object] = {"sources": {name: str(path) for name, (path, _, _) in SOURCES.items()}, "muscles": {}}
    for axis, muscle in zip(axes.flat, MUSCLES):
        terminal: dict[str, float] = {}
        for name, (path, color, style) in SOURCES.items():
            x, fatigue = boundary(path, muscle)
            axis.plot(x, 100 * fatigue, label=name, color=color, linestyle=style, linewidth=1.65)
            terminal[name] = float(100 * fatigue[-1])
        axis.set_title(muscle); axis.grid(True, alpha=0.25)
        summary["muscles"][muscle] = terminal
    axes[0, 0].set_ylabel("fatigue normalisée (%)"); axes[1, 0].set_ylabel("fatigue normalisée (%)")
    axes[1, 0].set_xlabel("cycle"); axes[1, 1].set_xlabel("cycle")
    axes[0, 0].legend(fontsize=7.2, loc="upper left")
    fig.suptitle("Profils de fatigue : RHO, RHO_BO et FHO à travail par cycle", fontsize=13)
    fig.text(0.5, -0.02, "F = 1 − A/a_scale aux frontières de cycle. Les FHO sans contrainte de travail à chaque cycle sont exclus.", ha="center", fontsize=8)
    fig.savefig(OUT / "endurance-fatigue-profiles-rho-bo-fho.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    (OUT / "endurance-fatigue-profiles-rho-bo-fho.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
