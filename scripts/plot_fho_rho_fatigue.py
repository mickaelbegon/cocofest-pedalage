#!/usr/bin/env python3
"""Compare capacity-normalized fatigue trajectories between RHO and FHO."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cocofest")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "local-results/isokinetic-prescribed-unit-1p3-fho-bo64-20260930"
RHO = BASE / "rho-seed/validated-rho-trajectory.npz"
FHO_DIR = BASE / "fho-continuation"
OUT = FHO_DIR / "analysis"
MUSCLES = ("Biceps", "Delt_ant", "Delt_post", "Triceps")


def metadata(data: np.lib.npyio.NpzFile) -> dict:
    return json.loads(str(data["metadata__json"]))


def latest_fho() -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in FHO_DIR.glob("fho-????-solution.npz"):
        try:
            with np.load(path, allow_pickle=False) as data:
                candidates.append((int(metadata(data)["cycles_per_window"]), path))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not candidates:
        raise RuntimeError("No readable FHO solution is available.")
    return max(candidates)[1]


def boundary_fatigue(path: Path, muscle: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        info = metadata(data)
        cycles = int(info["cycles_per_window"])
        capacity = float(info.get("fatigue_capacity_scales", {}).get(
            f"A_{muscle}", info["configured_muscle_parameters"][muscle]["a_scale"]
        ))
        values = np.asarray(data[f"states__A_{muscle}"], dtype=float).reshape(-1)
        intervals, remainder = divmod(values.size - 1, cycles)
        if remainder:
            raise ValueError(f"Invalid state layout for {path}.")
        boundary = values[::intervals]
        return np.arange(cycles + 1), 1.0 - boundary / capacity


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fho = latest_fho()
    with np.load(fho, allow_pickle=False) as data:
        fho_cycles = int(metadata(data)["cycles_per_window"])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True, constrained_layout=True)
    summary: dict[str, object] = {"fho_solution": str(fho), "cycles": fho_cycles, "muscles": {}}
    for axis, muscle in zip(axes.flat, MUSCLES):
        rho_x, rho_y = boundary_fatigue(RHO, muscle)
        fho_x, fho_y = boundary_fatigue(fho, muscle)
        rho_mask = rho_x <= fho_cycles
        axis.plot(rho_x[rho_mask], 100 * rho_y[rho_mask], "--", color="#4c78a8", linewidth=1.8, label="RHO reference")
        axis.plot(fho_x, 100 * fho_y, "-", color="#f58518", linewidth=1.8, label=f"FHO_{fho_cycles}")
        axis.set_title(muscle)
        axis.grid(True, alpha=0.25)
        summary["muscles"][muscle] = {
            "rho_terminal_fatigue_percent": float(100 * rho_y[fho_cycles]),
            "fho_terminal_fatigue_percent": float(100 * fho_y[-1]),
            "fho_minus_rho_percentage_points": float(100 * (fho_y[-1] - rho_y[fho_cycles])),
        }
    axes[0, 0].set_ylabel("fatigue normalisée (%)")
    axes[1, 0].set_ylabel("fatigue normalisée (%)")
    axes[1, 0].set_xlabel("cycle")
    axes[1, 1].set_xlabel("cycle")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Fatigue musculaire normalisée : RHO vs FHO_{fho_cycles}", fontsize=13)
    fig.text(0.5, -0.02, "F = 1 − A / a_scale, évaluée à chaque frontière de cycle. FHO : certificat opérationnel (tolérance énergétique 0,1 %).", ha="center", fontsize=8)
    fig.savefig(OUT / "rho-fho-muscle-fatigue-comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    (OUT / "rho-fho-muscle-fatigue-comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
