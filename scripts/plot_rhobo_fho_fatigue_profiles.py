#!/usr/bin/env python3
"""Plot normalized muscle-fatigue trajectories for matched RHO-BO/FHO runs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = {
    1.30: ROOT / "local-results/isokinetic-ding-local-rhobo-fho-parallel-run2-20261006/1p300",
    1.20: ROOT / "local-results/isokinetic-ding-local-rhobo-fho-parallel-run2-20261006/1p200",
    1.15: ROOT / "local-results/isokinetic-ding-local-rhobo-fho-parallel-run2-20261006/1p150",
    1.00: ROOT / "local-results/isokinetic-ding-local-rhobo-fho-1p00-20261007/1p000",
}
OUT = ROOT / "local-results/isokinetic-ding-local-rhobo-fho-parallel-run2-20261006/analysis"
MUSCLES = ("Biceps", "Delt_ant", "Delt_post", "Triceps")
COLORS = {1.00: "#7b2cbf", 1.15: "#2a9d8f", 1.20: "#e76f51", 1.30: "#457b9d"}


def _certified_fho(campaign: Path) -> tuple[int, Path] | None:
    status_path = campaign / "campaign-status.json"
    if not status_path.is_file():
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    cycles = status.get("fho_certified_cycles")
    if not isinstance(cycles, int) or cycles < 1:
        return None
    result = campaign / f"fho-{cycles:04d}-result.json"
    solution = campaign / f"fho-{cycles:04d}-solution.npz"
    if not result.is_file() or not solution.is_file():
        return None
    data = json.loads(result.read_text(encoding="utf-8"))["results"][0]
    if not data.get("success") or int(data.get("validated_cycles") or 0) < cycles:
        return None
    return cycles, solution


def _profile(path: Path, muscle: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata__json"]))
        cycles = int(metadata["cycles_per_window"])
        scale = float(metadata.get("fatigue_capacity_scales", {}).get(
            f"A_{muscle}", metadata["configured_muscle_parameters"][muscle]["a_scale"]
        ))
        capacity = np.asarray(archive[f"states__A_{muscle}"], dtype=float).reshape(-1)
    step, remainder = divmod(capacity.size - 1, cycles)
    if remainder:
        raise ValueError(f"Cycle boundaries do not divide the state trace: {path}")
    x = np.arange(cycles + 1)
    fatigue = 100.0 * (1.0 - capacity[::step] / scale)
    return x, fatigue


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharey=True, constrained_layout=True)

    for resistance, campaign in sorted(CAMPAIGNS.items()):
        rho = campaign / "rho-bo-trajectory.npz"
        status_path = campaign / "campaign-status.json"
        if not rho.is_file() or not status_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        rho_cycles = int(status.get("rho_certified_cycles") or 0)
        if rho_cycles < 1:
            continue
        fho = _certified_fho(campaign)
        label_rho = f"RHO-BO {resistance:.2f} Nm ({rho_cycles} c.)"
        records[f"{resistance:.2f}"] = {
            "rho_cycles": rho_cycles,
            "rho_archive": str(rho),
            "fho_cycles": fho[0] if fho else None,
            "fho_archive": str(fho[1]) if fho else None,
            "rho_terminal_fatigue_percent": {},
            "fho_terminal_fatigue_percent": {},
        }
        for axis, muscle in zip(axes.flat, MUSCLES, strict=True):
            x, fatigue = _profile(rho, muscle)
            axis.plot(x, fatigue, color=COLORS[resistance], linestyle="--", linewidth=1.55,
                      label=label_rho)
            records[f"{resistance:.2f}"]["rho_terminal_fatigue_percent"][muscle] = float(fatigue[-1])
            if fho:
                x_fho, fatigue_fho = _profile(fho[1], muscle)
                axis.plot(x_fho, fatigue_fho, color=COLORS[resistance], linestyle="-", linewidth=1.75,
                          label=f"FHO {resistance:.2f} Nm ({fho[0]} c.)")
                records[f"{resistance:.2f}"]["fho_terminal_fatigue_percent"][muscle] = float(fatigue_fho[-1])

    for axis, muscle in zip(axes.flat, MUSCLES, strict=True):
        axis.set_title(muscle)
        axis.set_xlabel("Cycle")
        axis.set_ylabel("Fatigue normalisée, 1 − A/a_scale (%)")
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, fontsize=7.2, frameon=False, loc="best")
    fig.suptitle("Profils de fatigue par cycle : RHO-BO et FHO", fontsize=14)
    fig.text(0.5, -0.015,
             "Tirets = RHO-BO reconstruit avec poids BO fixes; trait plein = dernier FHO certifié. "
             "Les horizons diffèrent; comparer les courbes à cycle commun, pas seulement leurs extrémités.",
             ha="center", fontsize=8)
    figure = OUT / "rhobo-vs-fho-fatigue-per-cycle.png"
    fig.savefig(figure, dpi=180, bbox_inches="tight")
    plt.close(fig)
    (OUT / "rhobo-vs-fho-fatigue-per-cycle.json").write_text(
        json.dumps({"definition": "100*(1-A/a_scale) at physical cycle boundaries",
                    "campaigns": records}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
