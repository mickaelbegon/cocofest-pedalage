#!/usr/bin/env python3
"""Complete and plot the 0.90--1.35 Nm Ding-local BO weight-transfer matrix."""
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

from scripts.run_isokinetic_ding_local_weight_transfer_study import MUSCLES, _run_case


RESISTANCES = (0.73, 0.75, 0.80, 0.90, 1.00, 1.15, 1.20, 1.25, 1.30, 1.35)


def _key(source: float, target: float) -> tuple[float, float]:
    return round(source, 2), round(target, 2)


def _read_records(path: Path) -> dict[tuple[float, float], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {_key(item["source_resistance_nm"], item["target_resistance_nm"]): item for item in data}


def _label(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _plot(records: dict[tuple[float, float], dict], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9.2, 5.5), constrained_layout=True)
    for source in RESISTANCES:
        values = [records[_key(source, target)]["certified_cycles"] for target in RESISTANCES]
        is_invalid = source == 1.35
        axis.plot(RESISTANCES, values, "o--" if is_invalid else "o-", linewidth=1.8,
                  alpha=0.6 if is_invalid else 0.9,
                  label=f"poids BO {source:.2f} Nm" + (" (aucun cycle à la source)" if is_invalid else ""))
    axis.set(xlabel="Résistance équivalente (Nm)", ylabel="Cycles RHO certifiés",
             title="Transfert des poids BO fixes — RHO isocinétique Ding-local")
    axis.set_xticks(RESISTANCES)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right", fontsize=8, ncol=2)
    fig.savefig(output / "weight-transfer-cycles-vs-resistance.png", dpi=180)
    plt.close(fig)


def _write(records: dict[tuple[float, float], dict], output: Path) -> None:
    with (output / "weight-transfer-certified-cycles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["weights_optimized_at_nm", *(f"evaluated_at_{target:.2f}_nm" for target in RESISTANCES)])
        for source in RESISTANCES:
            writer.writerow([f"{source:.2f}", *(records[_key(source, target)]["certified_cycles"] for target in RESISTANCES)])
    lines = ["# Matrice complète de transfert des poids Ding-local", "", "Valeur : cycles RHO certifiés.", "",
             "| poids optimisés à \\ évalués à | " + " | ".join(f"{target:.2f} Nm" for target in RESISTANCES) + " |",
             "|---|" + "---|" * len(RESISTANCES)]
    for source in RESISTANCES:
        suffix = " †" if source == 1.35 else ""
        lines.append("| " + f"{source:.2f} Nm{suffix}" + " | " + " | ".join(
            str(records[_key(source, target)]["certified_cycles"]) for target in RESISTANCES) + " |")
    lines.extend(["", "† À 1.35 Nm, aucun des 48 essais n’a certifié le premier cycle; cette ligne ne représente donc pas un optimum BO physiologiquement interprétable."])
    (output / "weight-transfer-certified-cycles.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ordered = [records[_key(source, target)] for source in RESISTANCES for target in RESISTANCES]
    (output / "weight-transfer-records.json").write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
    _plot(records, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low-campaign", type=Path, required=True)
    parser.add_argument("--high-campaign", type=Path, required=True)
    parser.add_argument("--base-records", type=Path, required=True,
                        help="Previously completed matrix records to reuse instead of rerunning cells.")
    parser.add_argument("--additional-bo", type=str, nargs=2, action="append", default=[],
                        metavar=("RESISTANCE", "CAMPAIGN"),
                        help="Completed BO campaign to add, for example: 0.73 /path/to/campaign.")
    parser.add_argument("--python", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hsl-library", required=True)
    parser.add_argument("--max-cycles", type=int, default=1500)
    parser.add_argument("--cpu-set", default="12-23")
    parser.add_argument("--ipopt-max-iter", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output = args.output.resolve(); args.root = args.root.resolve()
    args.low_campaign = args.low_campaign.resolve(); args.high_campaign = args.high_campaign.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    best: dict[float, tuple[float, ...]] = {}
    for source in (0.80, 0.90, 1.00):
        summary = json.loads((args.low_campaign / "bo" / _label(source) / "summary.json").read_text(encoding="utf-8"))
        best[source] = tuple(float(summary["best_weights"][muscle]) for muscle in MUSCLES)
    for raw_source, raw_campaign in args.additional_bo:
        source = float(raw_source)
        campaign = Path(raw_campaign).resolve()
        summary = json.loads((campaign / "bo" / _label(source) / "summary.json").read_text(encoding="utf-8"))
        if int(summary["completed"]) != 48:
            raise ValueError(f"BO {source:.2f} Nm is incomplete")
        best[source] = tuple(float(summary["best_weights"][muscle]) for muscle in MUSCLES)
    for raw_source, summary in json.loads((args.high_campaign / "best-weights.json").read_text(encoding="utf-8")).items():
        source = float(raw_source)
        best[source] = tuple(float(summary["best_weights"][muscle]) for muscle in MUSCLES)
    records = _read_records(args.base_records.resolve())
    missing = [(source, target) for source in RESISTANCES for target in RESISTANCES if _key(source, target) not in records]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_case, f"transfer-{source:.2f}-to-{target:.2f}", target, best[source],
                        args.output / "replays" / f"weights-{_label(source)}_at-{_label(target)}", args): (source, target)
            for source, target in missing
        }
        for future in as_completed(futures):
            source, target = futures[future]
            record = future.result()
            record["source_resistance_nm"] = source
            record["target_resistance_nm"] = target
            records[_key(source, target)] = record
    _write(records, args.output)


if __name__ == "__main__":
    main()
