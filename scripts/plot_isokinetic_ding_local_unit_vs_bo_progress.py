#!/usr/bin/env python3
"""Render a provisional unit-weight versus BO endurance comparison."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


RESISTANCES = (0.75, 0.80, 0.90, 1.00, 1.15, 1.20, 1.25, 1.30, 1.35)
ROOT = Path("/home/mickaelbegon/Documents/Kevin/cocofest-pedalage")
UNIT = ROOT / "local-results/isokinetic-ding-local-unit-vs-bo-0p75-to-1p35-20261005"
LOW = ROOT / "local-results/isokinetic-ding-local-endurance-sweep-bo-20261004"
HIGH = ROOT / "local-results/isokinetic-ding-local-weight-transfer-5r-48x8-20261004"
INTERIM = ROOT / "local-results/isokinetic-ding-local-bo-0p75-20261005"


def label(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def main() -> None:
    unit = {}
    for path in UNIT.glob("unit/*/observation.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        unit[float(record["resistance_nm"])] = float(record["certified_cycles"])
    bo, bo_final = [], []
    for resistance in RESISTANCES:
        if resistance == 0.75:
            path = INTERIM / "bo/0p75/summary.json"
        elif resistance <= 1.00:
            path = LOW / "bo" / label(resistance) / "summary.json"
        else:
            path = HIGH / "bo" / label(resistance) / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        bo.append(float(summary["best_cycles"])); bo_final.append(int(summary["completed"]) == 48)
    unit_x = [value for value in RESISTANCES if value in unit]
    fig, axis = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    axis.plot(unit_x, [unit[value] for value in unit_x], "o-", color="#355c9a", linewidth=2,
              label="RHO unitaire — poids égaux")
    final_x = [value for value, final in zip(RESISTANCES, bo_final) if final]
    final_y = [value for value, final in zip(bo, bo_final) if final]
    axis.plot(final_x, final_y, "s-", color="#d0782c", linewidth=2,
              label="RHO-BO — 48 essais")
    pending_x = [value for value, final in zip(RESISTANCES, bo_final) if not final]
    pending_y = [value for value, final in zip(bo, bo_final) if not final]
    if pending_x:
        axis.scatter(pending_x, pending_y, marker="s", s=60, facecolors="none", edgecolors="#d0782c",
                     linewidths=2, label="RHO-BO provisoire")
    axis.set(xlabel="Résistance équivalente (Nm)", ylabel="Cycles RHO certifiés",
             title="RHO unitaire et RHO-BO — isocinétique Ding-local")
    axis.set_xticks(RESISTANCES); axis.grid(True, alpha=0.25); axis.legend()
    fig.savefig(UNIT / "unit-vs-bo-certified-cycles-progress.png", dpi=180)


if __name__ == "__main__":
    main()
