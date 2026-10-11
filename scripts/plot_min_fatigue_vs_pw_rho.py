#!/usr/bin/env python3
"""Compare certified RHO trajectories under fatigue and PW² objectives."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cocofest")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FATIGUE = ROOT / "local-results/isokinetic-prescribed-unit-1p3-fho-bo64-20260930/rho-seed/validated-rho-trajectory.npz"
PW = ROOT / "local-results/isokinetic-prescribed-unit-1p3-pw2-20261001/rho-unit-pw/validated-rho-trajectory.npz"
OUT = PW.parent.parent / "analysis"
MUSCLES = ("Biceps", "Delt_ant", "Delt_post", "Triceps")
FATIGUE_BO = ROOT / "local-results/isokinetic-prescribed-unit-1p3-fho-bo64-20260930/rho-bo/summary.json"
FATIGUE_FHO = ROOT / "local-results/isokinetic-prescribed-unit-1p3-fho-bo64-20260930/fho-continuation/fho-0078-result.json"
PW_BO = ROOT / "local-results/isokinetic-prescribed-unit-1p3-pw2-20261001/rho-bo-pw/summary.json"


def metadata(data: np.lib.npyio.NpzFile) -> dict:
    return json.loads(str(data["metadata__json"]))


def boundaries(path: Path, muscle: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        info = metadata(data)
        cycles = int(info["cycles_per_window"])
        scale = float(info.get("fatigue_capacity_scales", {}).get(
            f"A_{muscle}", info["configured_muscle_parameters"][muscle]["a_scale"]
        ))
        values = np.asarray(data[f"states__A_{muscle}"], dtype=float).reshape(-1)
    step, remainder = divmod(values.size - 1, cycles)
    if remainder:
        raise ValueError(f"Invalid state layout in {path}")
    return np.arange(cycles + 1), 1.0 - values[::step] / scale


def cycles(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        return int(metadata(data)["cycles_per_window"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fatigue_cycles, pw_cycles = cycles(FATIGUE), cycles(PW)
    overlap = min(fatigue_cycles, pw_cycles)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.1), sharex=True, sharey=True, constrained_layout=True)
    summary: dict[str, object] = {"fatigue_objective_cycles": fatigue_cycles, "pw_squared_objective_cycles": pw_cycles,
                                  "comparison_cycles": overlap, "muscles": {}}
    for axis, muscle in zip(axes.flat, MUSCLES):
        x_f, f_f = boundaries(FATIGUE, muscle)
        x_p, f_p = boundaries(PW, muscle)
        axis.plot(x_f, 100 * f_f, color="#4c78a8", linewidth=1.8, label="RHO min-fatigue")
        axis.plot(x_p, 100 * f_p, color="#e45756", linewidth=1.8, label="RHO min-PW²")
        axis.axvline(overlap, color="0.55", linestyle=":", linewidth=1)
        axis.set_title(muscle); axis.grid(True, alpha=0.25)
        summary["muscles"][muscle] = {
            "fatigue_objective_terminal_percent": float(100 * f_f[-1]),
            "pw_squared_terminal_percent": float(100 * f_p[-1]),
            "fatigue_objective_at_common_horizon_percent": float(100 * f_f[overlap]),
            "pw_squared_at_common_horizon_percent": float(100 * f_p[overlap]),
            "pw_minus_fatigue_common_horizon_pp": float(100 * (f_p[overlap] - f_f[overlap])),
        }
    axes[0, 0].set_ylabel("fatigue normalisée (%)"); axes[1, 0].set_ylabel("fatigue normalisée (%)")
    axes[1, 0].set_xlabel("cycle RHO"); axes[1, 1].set_xlabel("cycle RHO")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("RHO unitaire : objectif min-fatigue vs somme quadratique des PW", fontsize=13)
    fig.text(0.5, -0.02, "F = 1 − A/a_scale aux frontières de cycle. Même modèle prescrit isocinétique, résistance 1,3 Nm; les horizons diffèrent.", ha="center", fontsize=8)
    figure = OUT / "rho-min-fatigue-vs-pw2-muscle-fatigue.png"
    fig.savefig(figure, dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    bars = axis.bar(["RHO min-fatigue", "RHO min-PW²"], [fatigue_cycles, pw_cycles], color=["#4c78a8", "#e45756"])
    for bar, value in zip(bars, (fatigue_cycles, pw_cycles)):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.8, str(value), ha="center", va="bottom")
    axis.set(ylabel="cycles certifiés", title="Endurance RHO certifiée par objectif")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "rho-min-fatigue-vs-pw2-certified-cycles.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fatigue_bo = json.loads(FATIGUE_BO.read_text(encoding="utf-8"))["best_cycles"]
    fatigue_fho_result = json.loads(FATIGUE_FHO.read_text(encoding="utf-8"))["results"][0]
    fatigue_fho = int(fatigue_fho_result["covered_cycles"])
    pw_bo = json.loads(PW_BO.read_text(encoding="utf-8")) if PW_BO.is_file() else None
    pw_bo_cycles = None if pw_bo is None else pw_bo.get("best_cycles")
    labels = ("RHO", "RHO_BO", "FHO")
    fatigue_values = (fatigue_cycles, fatigue_bo, fatigue_fho)
    pw_values = (pw_cycles, pw_bo_cycles or 0.0, 0.0)
    fig, axis = plt.subplots(figsize=(9.2, 4.7), constrained_layout=True)
    index = np.arange(len(labels)); width = 0.36
    left = axis.bar(index - width / 2, fatigue_values, width, color="#4c78a8", label="objectif min-fatigue")
    right = axis.bar(index + width / 2, pw_values, width, color="#e45756", label="objectif min-PW²")
    for bar, value in zip(left, fatigue_values):
        suffix = " opérationnel" if bar is left[2] and not fatigue_fho_result.get("success") else ""
        axis.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:g}{suffix}", ha="center", fontsize=8)
    for number, (bar, value) in enumerate(zip(right, pw_values)):
        text = "en cours" if number == 2 else (f"{value:g} certifié" if number == 0 else f"{value:g} certifié BO")
        axis.text(bar.get_x() + bar.get_width() / 2, max(value, 1.0) + 1.0, text, ha="center", fontsize=8)
    axis.set_xticks(index, labels); axis.set_ylabel("cycles certifiés")
    axis.set_title("Endurance : RHO unitaire, RHO_BO et FHO")
    axis.grid(axis="y", alpha=0.25); axis.legend(fontsize=8)
    fig.text(0.5, -0.02, "FHO min-fatigue : continuation opérationnelle (audit énergétique 0,1 %), pas un certificat scientifique strict. RHO_BO PW² : meilleur résultat à date.", ha="center", fontsize=8)
    fig.savefig(OUT / "endurance-rho-bo-fho-min-fatigue-vs-pw2.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    summary["endurance"] = {"rho_min_fatigue": fatigue_cycles, "rho_bo_min_fatigue": fatigue_bo,
                             "fho_min_fatigue_operational": fatigue_fho, "rho_pw2": pw_cycles,
                             "rho_bo_pw2_best_so_far": pw_bo_cycles, "fho_pw2": None}
    (OUT / "rho-min-fatigue-vs-pw2-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
