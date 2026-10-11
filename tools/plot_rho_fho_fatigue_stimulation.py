#!/usr/bin/env python3
"""Compare a certified FHO horizon with the matching prefix of the RHO run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MUSCLES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
LABELS = {"Delt_ant": "Deltoïde ant.", "Delt_post": "Deltoïde post.", "Biceps": "Biceps", "Triceps": "Triceps"}
COLORS = {"Delt_ant": "#3976af", "Delt_post": "#d07a29", "Biceps": "#4c9a62", "Triceps": "#a84d68"}


def load_solution(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].squeeze() for key in archive.files if key != "metadata__json"}
        metadata = json.loads(str(archive["metadata__json"]))
    return arrays, metadata


def cycle_trace(values: np.ndarray, cycles: int, nodes_per_cycle: int) -> np.ndarray:
    indices = np.arange(cycles + 1) * nodes_per_cycle
    return values[indices]


def polar_stimulation(ax: plt.Axes, controls: dict[str, np.ndarray], cycles: int, label: str) -> None:
    n_controls = len(controls["controls__last_pulse_width_Biceps"])
    per_cycle = n_controls // cycles
    theta = np.linspace(0, 2 * np.pi, per_cycle, endpoint=False)
    width = 2 * np.pi / per_cycle
    max_pw = max(float(np.max(controls[f"controls__last_pulse_width_{muscle}"])) for muscle in MUSCLES)
    max_pw = max(max_pw, 1.0)
    for index, muscle in enumerate(MUSCLES):
        values = controls[f"controls__last_pulse_width_{muscle}"][-per_cycle:] / max_pw
        bottom = 0.22 + index * 0.21
        radii = bottom + values * 0.18
        closed_theta = np.append(theta, theta[0])
        closed_radii = np.append(radii, radii[0])
        ax.fill_between(closed_theta, bottom, closed_radii, color=COLORS[muscle], alpha=0.48)
        ax.plot(closed_theta, closed_radii, color=COLORS[muscle], linewidth=1.6)
        ax.text(np.deg2rad(8), bottom + 0.09, LABELS[muscle], fontsize=8, va="center")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad((0, 90, 180, 270)), labels=("0°", "90°", "180°", "270°"))
    ax.set_yticks([])
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Stimulations normalisées — cycle {cycles}\n{label}", pad=18)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=Path, required=True)
    parser.add_argument("--fho", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rho, rho_meta = load_solution(args.rho)
    fho, fho_meta = load_solution(args.fho)
    rho_cycles = int(rho_meta["cycles_per_window"])
    fho_cycles = int(fho_meta["cycles_per_window"])
    common_cycles = min(rho_cycles, fho_cycles)
    rho_nodes = (len(rho["states__A_Biceps"]) - 1) // rho_cycles
    fho_nodes = (len(fho["states__A_Biceps"]) - 1) // fho_cycles
    capacities = rho_meta["fatigue_capacity_scales"]

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12})
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    mosaic = fig.subplot_mosaic([["fatigue", "rho_polar"], ["min_capacity", "fho_polar"]], subplot_kw={"projection": None})
    # Recreate polar axes in the right-hand slots.
    for key in ("rho_polar", "fho_polar"):
        spec = mosaic[key].get_subplotspec()
        mosaic[key].remove()
        mosaic[key] = fig.add_subplot(spec, projection="polar")

    cycles = np.arange(common_cycles + 1)
    min_rho, min_fho = [], []
    for muscle in MUSCLES:
        capacity = capacities[f"A_{muscle}"]
        rho_relative = cycle_trace(rho[f"states__A_{muscle}"], common_cycles, rho_nodes) / capacity
        fho_relative = cycle_trace(fho[f"states__A_{muscle}"], common_cycles, fho_nodes) / capacity
        min_rho.append(rho_relative)
        min_fho.append(fho_relative)
        mosaic["fatigue"].plot(cycles, 1 - rho_relative, color=COLORS[muscle], label=f"{LABELS[muscle]} — RHO")
        mosaic["fatigue"].plot(cycles, 1 - fho_relative, color=COLORS[muscle], linestyle="--", label=f"{LABELS[muscle]} — FHO")
    mosaic["fatigue"].set(title="Évolution de la fatigue normalisée", xlabel="Cycles", ylabel="F = 1 − A / A₀")
    mosaic["fatigue"].grid(alpha=0.25)
    mosaic["fatigue"].legend(ncol=2, fontsize=8, loc="upper left")
    rho_min = np.min(np.vstack(min_rho), axis=0)
    fho_min = np.min(np.vstack(min_fho), axis=0)
    mosaic["min_capacity"].plot(cycles, rho_min, color="#222222", label="RHO")
    mosaic["min_capacity"].plot(cycles, fho_min, color="#222222", linestyle="--", label="FHO")
    mosaic["min_capacity"].set(title="Capacité résiduelle du muscle limitant", xlabel="Cycles", ylabel="min(A / A₀)")
    mosaic["min_capacity"].grid(alpha=0.25)
    mosaic["min_capacity"].legend()
    polar_stimulation(mosaic["rho_polar"], rho, common_cycles, "préfixe de référence")
    polar_stimulation(mosaic["fho_polar"], fho, common_cycles, "solution monolithique certifiée")
    fig.suptitle(f"RHO vs FHO au même horizon certifié : {common_cycles} cycles", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    print(args.output)
    print(json.dumps({"cycles": common_cycles, "min_capacity_rho": float(rho_min[-1]), "min_capacity_fho": float(fho_min[-1]), "fatigue_rho": float(1-rho_min[-1]), "fatigue_fho": float(1-fho_min[-1])}))


if __name__ == "__main__":
    main()
