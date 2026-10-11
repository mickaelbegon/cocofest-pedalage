#!/usr/bin/env python3
"""Cross-evaluate refined Ding-local BO weights and plot their evolution."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import json
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_isokinetic_ding_local_weight_transfer_study import MUSCLES, _run_case


def label(resistance: float) -> str:
    return f"{resistance:.2f}".replace(".", "p")


def refined_summaries(campaign: Path, resistances: list[float], *, require_complete: bool) -> dict[float, dict]:
    result: dict[float, dict] = {}
    for resistance in resistances:
        path = campaign / "bo" / label(resistance) / "summary.json"
        if not path.is_file():
            if require_complete:
                raise FileNotFoundError(f"Missing refined BO summary: {path}")
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not summary.get("best_weights"):
            if require_complete:
                raise RuntimeError(f"No incumbent in {path}")
            continue
        if require_complete and summary.get("new_completed") != 48:
            raise RuntimeError(f"Refinement not complete for {resistance:.2f} Nm: {path}")
        result[resistance] = summary
    return result


def write_weight_figure(best: dict[float, dict], output: Path, *, partial: bool) -> None:
    ordered = sorted(best)
    fig, axis = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
    for muscle in MUSCLES:
        axis.plot(
            ordered,
            [float(best[value]["best_weights"][muscle]) for value in ordered],
            marker="o",
            linewidth=1.8,
            label=muscle,
        )
    # Make the evidence behind each BO incumbent explicit: counts can differ
    # after follow-up refinements (e.g., the 1.00-Nm 96-candidate extension).
    axis.set_xticks(
        ordered,
        [f"{value:.2f}\n(n={int(best[value].get('completed') or 0)})" for value in ordered],
    )
    axis.set(xlabel="Résistance (Nm)\n(n = évaluations BO disponibles)", ylabel="Poids fixe optimisé", ylim=(-0.03, 1.03))
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    axis.set_title(
        "Pondérations BO raffinées par résistance"
        + (" — état partiel" if partial else "")
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hsl-library", required=True)
    parser.add_argument("--resistances", type=float, nargs="+", required=True)
    parser.add_argument(
        "--summary-source",
        action="append",
        nargs=2,
        metavar=("RESISTANCE_NM", "SUMMARY_JSON"),
        default=[],
        help="Use this completed BO summary for a resistance outside source-campaign.",
    )
    parser.add_argument("--max-cycles", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu-set", default="12-23")
    parser.add_argument("--ipopt-max-iter", type=int, default=1000)
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--existing-replay",
        action="append",
        type=Path,
        default=[],
        help="Reuse certified observation.json files from this prior cross-replay directory.",
    )
    parser.add_argument(
        "--only-source-resistance",
        type=float,
        nargs="+",
        help="Replay only these source-weight rows against every requested target resistance.",
    )
    args = parser.parse_args()
    args.source_campaign = args.source_campaign.resolve()
    args.output = args.output.resolve()
    args.root = args.root.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    external_resistances = {float(resistance) for resistance, _ in args.summary_source}
    best = refined_summaries(
        args.source_campaign,
        [value for value in args.resistances if value not in external_resistances],
        require_complete=not args.allow_partial,
    )
    for resistance_text, summary_path in args.summary_source:
        resistance = float(resistance_text)
        path = Path(summary_path).resolve()
        if resistance not in args.resistances:
            raise ValueError(f"External summary resistance not requested: {resistance:.2f} Nm")
        if not path.is_file():
            raise FileNotFoundError(f"Missing external BO summary: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not summary.get("best_weights"):
            raise RuntimeError(f"No incumbent in external summary: {path}")
        # A refinement can legitimately contain more than the baseline 48 new
        # observations (for example a 96/192-candidate follow-up).  Reject an
        # incomplete study, not a more thoroughly sampled one.
        if not args.allow_partial and int(summary.get("new_completed") or 0) < 48:
            raise RuntimeError(f"External refinement is incomplete: {path}")
        if abs(float(summary.get("resistance_nm")) - resistance) > 1e-12:
            raise ValueError(f"External summary resistance mismatch: {path}")
        best[resistance] = summary
    if not best:
        raise RuntimeError("No refined BO incumbent is available.")
    args.output.mkdir(parents=True)
    (args.output / "refined-best-weights.json").write_text(
        json.dumps(best, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_weight_figure(best, args.output / "weights-vs-resistance.png", partial=args.allow_partial)
    if args.weights_only:
        return
    if len(best) != len(args.resistances):
        raise RuntimeError("Cross matrix requires every requested refined BO result.")
    ordered = sorted(best)
    sources = ordered if args.only_source_resistance is None else sorted(args.only_source_resistance)
    if any(source not in best for source in sources):
        raise ValueError("Every --only-source-resistance value must have a BO incumbent.")
    cases = []
    for source in sources:
        weights = tuple(float(best[source]["best_weights"][muscle]) for muscle in MUSCLES)
        for target in sorted(best):
            root = args.output / "cross-replay" / f"weights-{label(source)}_at-{label(target)}"
            cases.append((source, target, weights, root))
    # Reuse already certified cells so completing a matrix never wastes a
    # potentially multi-hour low-resistance replay.  Later directories win,
    # allowing a newer BO incumbent (notably 1.00 Nm) to supersede an older
    # source row.
    cached: dict[tuple[float, float], dict] = {}
    for replay in args.existing_replay:
        for observation in replay.resolve().glob("weights-*_at-*/observation.json"):
            match = re.fullmatch(r"weights-(\d+p\d+)_at-(\d+p\d+)", observation.parent.name)
            if not match:
                continue
            source = float(match.group(1).replace("p", "."))
            target = float(match.group(2).replace("p", "."))
            if source not in sources or target not in best:
                continue
            row = json.loads(observation.read_text(encoding="utf-8"))
            if row.get("error") is not None or not isinstance(row.get("certified_cycles"), (int, float)):
                continue
            row["source_resistance_nm"] = source
            row["target_resistance_nm"] = target
            cached[source, target] = row
    records = list(cached.values())
    # _run_case builds an OCP through a subprocess.  Some imported numerical
    # backends keep Tk-related process-global state, which is not thread-safe:
    # concurrent ThreadPool workers can therefore abort during interpreter
    # cleanup.  A process pool retains parallelism while fully isolating those
    # per-case runtimes.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        active = {
        pool.submit(_run_case, f"refined-transfer-{source:.2f}-to-{target:.2f}", target, weights, root, args): (source, target)
            for source, target, weights, root in cases if (source, target) not in cached
        }
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                source, target = active.pop(future)
                record = future.result()
                record["source_resistance_nm"] = source
                record["target_resistance_nm"] = target
                records.append(record)
    index = {(row["source_resistance_nm"], row["target_resistance_nm"]): row for row in records}
    with (args.output / "weight-transfer-certified-cycles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["weights_optimized_at_nm", *(f"evaluated_at_{item:.2f}_nm" for item in ordered)])
        for source in sources:
            writer.writerow([source, *(index[source, target]["certified_cycles"] for target in ordered)])
    lines = ["# Matrice croisée — poids BO raffinés", "", "Valeur : cycles RHO certifiés.", ""]
    lines += [
        "| poids optimisés à \\ évalués à | " + " | ".join(f"{item:.2f} Nm" for item in ordered) + " |",
        "|---|" + "---|" * len(ordered),
    ]
    for source in sources:
        lines.append("| " + f"{source:.2f} Nm" + " | " + " | ".join(
            str(index[source, target]["certified_cycles"]) for target in ordered
        ) + " |")
    (args.output / "weight-transfer-certified-cycles.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output / "weight-transfer-records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
