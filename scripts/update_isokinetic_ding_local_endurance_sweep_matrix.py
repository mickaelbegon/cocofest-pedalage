#!/usr/bin/env python3
"""Replay completed BO-optimal weights into the current endurance sweep matrix."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import sys

# The script may be called by systemd through an absolute path, in which case
# Python only places ``scripts/`` on sys.path.  Keep the project root explicit.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_isokinetic_ding_local_weight_transfer_study import MUSCLES, _run_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hsl-library", required=True)
    parser.add_argument("--resistances", type=float, nargs="+", required=True)
    parser.add_argument("--max-cycles", type=int, default=1500)
    parser.add_argument("--cpu-set", default="12-23")
    parser.add_argument("--ipopt-max-iter", type=int, default=1000)
    args = parser.parse_args()
    args.campaign = args.campaign.resolve(); args.root = args.root.resolve()
    best: dict[float, dict[str, float]] = {}
    for source in args.resistances:
        label = f"{source:.2f}".replace(".", "p")
        summary = json.loads((args.campaign / "bo" / label / "summary.json").read_text(encoding="utf-8"))
        if summary["completed"] != 48:
            raise ValueError(f"BO {source:.2f} Nm is not complete")
        best[source] = {name: float(summary["best_weights"][name]) for name in MUSCLES}
    cases = []
    for source, weights_by_muscle in best.items():
        weights = tuple(weights_by_muscle[name] for name in MUSCLES)
        for target in args.resistances:
            source_label, target_label = (f"{value:.2f}".replace(".", "p") for value in (source, target))
            root = args.campaign / "cross-replay" / f"weights-{source_label}_at-{target_label}"
            cases.append((source, target, weights, root))
    with ThreadPoolExecutor(max_workers=len(cases)) as pool:
        futures = {pool.submit(_run_case, f"transfer-{source:.2f}-to-{target:.2f}", target, weights, root, args): (source, target)
                   for source, target, weights, root in cases}
        records = []
        for future, (source, target) in futures.items():
            record = future.result()
            record["source_resistance_nm"] = source
            record["target_resistance_nm"] = target
            records.append(record)
    records.sort(key=lambda item: (item["source_resistance_nm"], item["target_resistance_nm"]))
    lookup = {(item["source_resistance_nm"], item["target_resistance_nm"]): item for item in records}
    with (args.campaign / "current-weight-transfer-certified-cycles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["weights_optimized_at_nm", *(f"evaluated_at_{target:.2f}_nm" for target in args.resistances)])
        for source in args.resistances:
            writer.writerow([f"{source:.2f}", *(lookup[source, target]["certified_cycles"] for target in args.resistances)])
    lines = ["# Matrice courante de transfert des poids Ding-local", "", "Cycles RHO certifiés; chaque ligne utilise le meilleur poids fixe parmi les 48 essais BO.", "",
             "| poids optimisés à \\ évalués à | " + " | ".join(f"{target:.2f} Nm" for target in args.resistances) + " |",
             "|---|" + "---|" * len(args.resistances)]
    for source in args.resistances:
        lines.append("| " + f"{source:.2f} Nm" + " | " + " | ".join(
            str(lookup[source, target]["certified_cycles"]) for target in args.resistances) + " |")
    (args.campaign / "current-weight-transfer-certified-cycles.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.campaign / "current-weight-transfer-records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
