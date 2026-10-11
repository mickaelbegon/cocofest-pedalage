#!/usr/bin/env python3
"""Plot FHO convergence timing from solver JSON artifacts.

Only FHO campaigns enforcing the prescribed work at every cycle are shown.
Earlier total-work-only FHO results are deliberately excluded: they are not
comparable to the current feasibility protocol.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cocofest")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PER_CYCLE = ROOT / "local-results/isokinetic-prescribed-unit-1p3-fho-from-rho-bo-per-cycle-radau5-20261002/fho-continuation"
OUTPUT_DIR = PER_CYCLE / "analysis"


def load_result(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["results"][0]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def timing_record(label: str, cycles: int, path: Path, accepted: bool) -> dict | None:
    result = load_result(path)
    if result is None:
        return None
    wall = result.get("wall_time_s")
    solver = result.get("solver_time_s")
    if not isinstance(wall, (int, float)) or wall <= 0:
        return None
    stats = (result.get("nlp_solver_stats") or [{}])[0]
    return {
        "campaign": label,
        "cycles": cycles,
        "wall_time_s": float(wall),
        "solver_time_s": float(solver) if isinstance(solver, (int, float)) else None,
        "iterations": stats.get("iter_count"),
        "hessian_wall_s": stats.get("t_wall_nlp_hess_l"),
        "accepted": accepted,
        "strict_success": bool(result.get("success")),
        "source": str(path),
    }


def per_cycle_records() -> list[dict]:
    records: list[dict] = []
    for result_path in sorted(PER_CYCLE.glob("fho-????-result.json")):
        cycles = int(result_path.name.split("-")[1])
        record = timing_record(
            "Prescribed 1.3 Nm, per-cycle work, Radau-5",
            cycles,
            result_path,
            accepted=True,
        )
        if record:
            records.append(record)
    return records


def plot(records: list[dict], output: Path) -> None:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["campaign"], []).append(record)
    for rows in groups.values():
        rows.sort(key=lambda row: row["cycles"])

    fig, (total, per_cycle) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    styles = {
        "Prescribed 1.3 Nm, per-cycle work, Radau-5": {"color": "#54a24b", "marker": "^"},
    }
    for label, rows in groups.items():
        style = styles[label]
        x = [row["cycles"] for row in rows]
        y = [row["wall_time_s"] for row in rows]
        total.plot(x, y, label=label, linewidth=1.8, markersize=5, **style)
        per_cycle.plot(x, [wall / cycles for wall, cycles in zip(y, x)], label=label, linewidth=1.8, markersize=5, **style)

    total.set(title="FHO wall-clock convergence time", xlabel="FHO horizon (cycles)", ylabel="wall time (s)")
    per_cycle.set(title="Wall-clock time normalized by horizon", xlabel="FHO horizon (cycles)", ylabel="wall time / cycle (s)")
    for axis in (total, per_cycle):
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("FHO convergence timing — work constrained at every cycle", fontsize=13)
    fig.text(
        0.5,
        -0.02,
        "Only certified FHO results with prescribed work constrained at every cycle are included.",
        ha="center",
        fontsize=8,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    records = per_cycle_records()
    if not records:
        raise RuntimeError("No readable FHO timing records found.")
    csv_path = OUTPUT_DIR / "fho-convergence-time-comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plot(records, OUTPUT_DIR / "fho-convergence-time-comparison.png")


if __name__ == "__main__":
    main()
