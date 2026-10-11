#!/usr/bin/env python3
"""Plot current unilateral Ding-local RHO-unit and refined-BO endurance."""
from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "local-results/isokinetic-ding-local-bo-refined-figures-20261006"
FHO_CAMPAIGN = ROOT / "local-results/isokinetic-ding-local-rhobo-fho-parallel-run2-20261006"
RESISTANCES = (0.73, 0.75, 0.80, 0.90, 1.00, 1.15, 1.20, 1.25, 1.30)
MUSCLES = ("Biceps", "Delt_ant", "Delt_post", "Triceps")


def _label(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _unit_cycles(resistance: float) -> int:
    if resistance == 0.73:
        path = ROOT / "local-results/isokinetic-ding-local-unit-0p73-20261005/result.json"
    else:
        path = ROOT / "local-results/isokinetic-ding-local-unit-vs-bo-0p75-to-1p35-20261005/unit" / _label(resistance) / "result.json"
    data = json.loads(path.read_text(encoding="utf-8"))["results"][0]
    return int(data["validated_cycles"])


def _bo_summary(resistance: float) -> tuple[dict, bool]:
    if resistance == 0.73:
        path = ROOT / "local-results/isokinetic-ding-local-bo-refine-0p73-20261005/bo/0p73/summary.json"
        return json.loads(path.read_text(encoding="utf-8")), False
    if resistance == 1.00:
        path = ROOT / "local-results/isokinetic-ding-local-bo-refine-1p00-plus96-20261006/bo/1p00/summary.json"
        return json.loads(path.read_text(encoding="utf-8")), False
    if resistance == 0.90:
        path = ROOT / "local-results/isokinetic-ding-local-bo-refine-0p90-plus96-round2-20261006/bo/0p90/summary.json"
        return json.loads(path.read_text(encoding="utf-8")), False
    if resistance == 1.20:
        path = ROOT / "local-results/isokinetic-ding-local-bo-refine-1p20-plus96-round2-20261006/bo/1p20/summary.json"
        return json.loads(path.read_text(encoding="utf-8")), False
    if resistance == 1.30:
        path = ROOT / "local-results/isokinetic-ding-local-bo-refine-1p30-plus192-lexicographic-20261006/bo/1p30/summary.json"
        return json.loads(path.read_text(encoding="utf-8")), True
    path = ROOT / "local-results/isokinetic-ding-local-bo-refine-descending-20261005/bo" / _label(resistance) / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")), False


def _crossed_cycles() -> list[tuple[float, float, int]]:
    """Return certified transfer replays from the curated campaigns.

    The 0.73 campaign contains the completed validated matrix.  The second
    directory supersedes its 1.00-Nm source row with the newer 1.00 BO
    weights whenever a replay has completed.
    """
    campaigns = (
        ROOT / "local-results/isokinetic-ding-local-crossed-0p73-validated-20261005/cross-replay",
        ROOT / "local-results/isokinetic-ding-local-crossed-1p00-plus96-rerun-20261006/cross-replay",
        ROOT / "local-results/isokinetic-ding-local-crossed-complete-20261006/cross-replay",
        ROOT / "local-results/isokinetic-ding-local-crossed-0p90-1p20-round2-20261006/cross-replay",
    )
    values: dict[tuple[float, float], int] = {}
    pattern = re.compile(r"weights-(\d+p\d+)_at-(\d+p\d+)")
    for campaign in campaigns:
        for path in campaign.glob("weights-*_at-*/observation.json"):
            match = pattern.fullmatch(path.parent.name)
            if not match:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            cycles = data.get("certified_cycles")
            if data.get("error") is not None or not isinstance(cycles, (int, float)):
                continue
            source = float(match.group(1).replace("p", "."))
            target = float(match.group(2).replace("p", "."))
            values[source, target] = int(cycles)
    return [(source, target, cycles) for (source, target), cycles in values.items()]


def _fho_cycles() -> dict[float, int]:
    """Read promoted FHO horizons only; ignore partial attempts and seeds."""
    values: dict[float, int] = {}
    for resistance, folder in ((1.15, "1p150"), (1.20, "1p200"), (1.30, "1p300")):
        path = FHO_CAMPAIGN / folder / "campaign-status.json"
        if not path.is_file():
            continue
        status = json.loads(path.read_text(encoding="utf-8"))
        cycles = status.get("fho_certified_cycles")
        if isinstance(cycles, int) and cycles > 0:
            values[resistance] = cycles
    return values


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    unit = [_unit_cycles(value) for value in RESISTANCES]
    summaries = [_bo_summary(value) for value in RESISTANCES]
    bo = [float(summary["best_cycles"]) for summary, _ in summaries]
    fho = _fho_cycles()
    pending = [value for value, (_, is_pending) in zip(RESISTANCES, summaries) if is_pending]
    crossed = _crossed_cycles()

    fig, axis = plt.subplots(figsize=(9.2, 5.0), layout="constrained")
    axis.plot(RESISTANCES, unit, "o-", linewidth=2, color="#355c9a", label="RHO unitaire")
    axis.plot(RESISTANCES, bo, "s-", linewidth=2, color="#d0782c", label="RHO-BO raffiné")
    if fho:
        axis.plot(sorted(fho), [fho[x] for x in sorted(fho)], "*--", markersize=13,
                  linewidth=1.5, color="#7b2cbf", label="FHO certifié (depuis RHO-BO)")
    if crossed:
        targets = [target for _, target, _ in crossed]
        cycles = [candidate for _, _, candidate in crossed]
        axis.scatter(
            targets, cycles, marker="x", s=30, linewidths=1.0, color="#6c757d", alpha=0.55,
            label="Rejeux croisés des poids BO",
        )
    for value in pending:
        index = RESISTANCES.index(value)
        axis.scatter(value, bo[index], marker="s", s=75, facecolors="none", edgecolors="#d0782c", linewidths=2.2,
                     label="BO 1,30 Nm en cours")
    axis.set(xlabel="Résistance équivalente (Nm)", ylabel="Cycles RHO certifiés",
             title="Endurance unilatérale isocinétique Ding-local")
    axis.set_xticks(RESISTANCES); axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), frameon=False)
    fig.savefig(OUT / "unit-vs-refined-bo-certified-cycles.png", dpi=180)
    plt.close(fig)

    gains = [100.0 * (candidate / reference - 1.0) for candidate, reference in zip(bo, unit)]
    fig, axis = plt.subplots(figsize=(9.2, 4.7), layout="constrained")
    colors = ["#d0782c" if value not in pending else "#f0b36d" for value in RESISTANCES]
    axis.bar([str(value) for value in RESISTANCES], gains, color=colors)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel="Résistance équivalente (Nm)", ylabel="Gain vs RHO unitaire (%)",
             title="Gain des poids BO fixes")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "refined-bo-gain-vs-unit.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.2, 4.9), layout="constrained")
    for muscle in MUSCLES:
        axis.plot(
            RESISTANCES,
            [float(summary["best_weights"][muscle]) for summary, _ in summaries],
            marker="o",
            linewidth=1.8,
            label=muscle,
        )
    counts = [int(summary.get("completed") or 0) for summary, _ in summaries]
    axis.set_xticks(RESISTANCES, [f"{value:.2f}\n(n={count})" for value, count in zip(RESISTANCES, counts)])
    axis.set(
        xlabel="Résistance équivalente (Nm)\n(n = évaluations BO disponibles)",
        ylabel="Poids fixe optimisé",
        ylim=(-0.03, 1.03),
        title="Pondérations BO courantes par résistance",
    )
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    fig.savefig(OUT / "refined-bo-weights-vs-resistance.png", dpi=180)
    plt.close(fig)

    # Compare transfer quality independently at each target resistance.  The
    # RHO-unit reference is 0% and the best BO replay is 100%, so this shows
    # directly how much of the available BO gain each transferred weight gets.
    target_max = {
        target: max(cycles for _, candidate, cycles in crossed if candidate == target)
        for target in RESISTANCES
        if any(candidate == target for _, candidate, _ in crossed)
    }
    def relative_bo_gain(target: float, cycles: float) -> float:
        reference = unit[RESISTANCES.index(target)]
        return 100.0 * (cycles - reference) / (target_max[target] - reference)

    fig, axis = plt.subplots(figsize=(10.8, 6.1), layout="constrained")
    palette = plt.get_cmap("tab10")
    markers = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
    plotted_values = []
    for index, source in enumerate(RESISTANCES):
        points = sorted((target, cycles) for candidate, target, cycles in crossed if candidate == source and target in target_max)
        if not points:
            continue
        gains = [relative_bo_gain(target, cycles) for target, cycles in points]
        plotted_values.extend(gains)
        axis.plot(
            [target for target, _ in points],
            gains,
            marker=markers[index], markersize=5.8, linewidth=1.15, alpha=0.82, color=palette(index),
            markerfacecolor="white", markeredgewidth=1.1,
            label=f"Poids BO appris à {source:.2f} Nm",
        )
    unit_x = [value for value in RESISTANCES if value in target_max]
    axis.plot(unit_x, [0.0] * len(unit_x), "o-", linewidth=2, color="#355c9a", label="RHO unitaire (0 %)")
    axis.axhline(100.0, color="#d0782c", linewidth=1.3, linestyle="--", label="Meilleur rejeu BO (100 %)")
    fho_gains: list[float] = []
    fho_x: list[float] = []
    for target, cycles in sorted(fho.items()):
        if target not in target_max:
            continue
        value = relative_bo_gain(target, cycles)
        fho_x.append(target)
        fho_gains.append(value)
        axis.scatter([target], [value], marker="*", s=190, color="#7b2cbf", edgecolors="white",
                     linewidths=0.8, zorder=6,
                     label="FHO certifié" if len(fho_x) == 1 else None)
        axis.annotate(f"{cycles} cycles\n{value:.0f} %", (target, value), xytext=(5, 7),
                      textcoords="offset points", fontsize=8, color="#572080")
    lower = min(-5.0, min(plotted_values, default=0.0) - 5.0)
    upper = max(105.0, max(fho_gains, default=100.0) + 25.0)
    axis.set(
        xlabel="Résistance cible (Nm)", ylabel="Gain BO relatif (% de l'écart RHO unitaire → meilleur BO)",
        title="Rejeux croisés BO et FHO certifiés (référence BO = 100 %)",
        ylim=(lower, upper),
    )
    axis.set_xticks(RESISTANCES)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, loc="lower center", ncol=3, fontsize=8.2)
    fig.savefig(OUT / "crossed-replays-relative-to-target-maximum.png", dpi=180)
    plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps({
        "resistances_nm": RESISTANCES, "unit_cycles": unit, "bo_cycles": bo,
        "bo_evaluations": counts,
        "crossed_target_max_cycles": target_max,
        "fho_certified_cycles": {str(resistance): cycles for resistance, cycles in sorted(fho.items())},
        "fho_relative_to_unit_to_best_bo_percent": {
            str(resistance): relative_bo_gain(resistance, cycles)
            for resistance, cycles in sorted(fho.items()) if resistance in target_max
        },
        "crossed_replays": [
            {"source_resistance_nm": source, "target_resistance_nm": target, "certified_cycles": cycles}
            for source, target, cycles in crossed
        ],
        "bo_1p30_is_in_progress": True,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
