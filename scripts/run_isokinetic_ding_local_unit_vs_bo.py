#!/usr/bin/env python3
"""Run matched unit-weight RHO sweeps and plot them against fixed-weight BO."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_isokinetic_ding_local_weight_transfer_study import _run_case


RESISTANCES = (0.75, 0.80, 0.90, 1.00, 1.15, 1.20, 1.25, 1.30, 1.35)
UNIT_WEIGHTS = (1.0, 1.0, 1.0, 1.0)


def _label(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _bo_values(low: Path, high: Path, interim: Path) -> tuple[list[float], list[bool]]:
    values, provisional = [], []
    for resistance in RESISTANCES:
        if resistance == 0.75:
            summary_path = interim / "bo" / "0p75" / "summary.json"
        elif resistance <= 1.00:
            summary_path = low / "bo" / _label(resistance) / "summary.json"
        else:
            summary_path = high / "bo" / _label(resistance) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        values.append(float(summary["best_cycles"]))
        provisional.append(int(summary["completed"]) < 48)
    return values, provisional


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hsl-library", required=True)
    parser.add_argument("--low-bo", type=Path, required=True)
    parser.add_argument("--high-bo", type=Path, required=True)
    parser.add_argument("--interim-bo", type=Path, required=True)
    parser.add_argument("--max-cycles", type=int, default=1500)
    parser.add_argument("--cpu-set", default="12-23")
    parser.add_argument("--ipopt-max-iter", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output = args.output.resolve(); args.root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_case, f"unit-{resistance:.2f}", resistance, UNIT_WEIGHTS,
                        args.output / "unit" / _label(resistance), args): resistance
            for resistance in RESISTANCES
        }
        records = []
        for future in as_completed(futures):
            resistance = futures[future]
            record = future.result(); record["resistance_nm"] = resistance
            records.append(record)
    records.sort(key=lambda item: item["resistance_nm"])
    units = {item["resistance_nm"]: item["certified_cycles"] for item in records}
    bo, provisional = _bo_values(args.low_bo.resolve(), args.high_bo.resolve(), args.interim_bo.resolve())
    with (args.output / "unit-vs-bo-certified-cycles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["resistance_nm", "rho_unit_cycles", "rho_bo_cycles", "rho_bo_provisional"])
        for resistance, bo_cycles, is_provisional in zip(RESISTANCES, bo, provisional, strict=True):
            writer.writerow([resistance, units[resistance], bo_cycles, is_provisional])
    fig, axis = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    axis.plot(RESISTANCES, [units[item] for item in RESISTANCES], "o-", color="#355c9a", linewidth=2,
              label="RHO unitaire — poids égaux")
    confirmed = [not item for item in provisional]
    axis.plot([item for item, keep in zip(RESISTANCES, confirmed) if keep],
              [value for value, keep in zip(bo, confirmed) if keep], "s-", color="#d0782c", linewidth=2,
              label="RHO-BO — meilleur poids fixe, 48 essais")
    if any(provisional):
        axis.scatter([item for item, keep in zip(RESISTANCES, provisional) if keep],
                     [value for value, keep in zip(bo, provisional) if keep], marker="s", s=60,
                     facecolors="none", edgecolors="#d0782c", linewidths=2,
                     label="RHO-BO provisoire")
    axis.set(xlabel="Résistance équivalente (Nm)", ylabel="Cycles RHO certifiés",
             title="RHO unitaire et RHO-BO — isocinétique Ding-local")
    axis.set_xticks(RESISTANCES); axis.grid(True, alpha=0.25); axis.legend()
    fig.savefig(args.output / "unit-vs-bo-certified-cycles.png", dpi=180)
    plt.close(fig)
    (args.output / "unit-records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
