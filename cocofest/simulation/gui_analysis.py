"""Read existing simulation artifacts for GUI monitoring and scientific figures.

This module imports no solver or plotting backend until a figure is requested.
Journals are read incrementally: polling never rereads the full simulation log.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def artifact_files(root: Path, pattern: str) -> list[Path]:
    """Discover only the run and its two immediate children, never other campaigns."""
    root = Path(root)
    return sorted({*root.glob(pattern), *root.glob(f"*/{pattern}")})


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class LiveArtifacts:
    """Bounded incremental JSONL reader tolerant of a writer's unfinished last line."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.offsets = {}
        self.latest = {}
        self.result_stamps = {}

    def update(self):
        for path in artifact_files(self.root, "*.jsonl"):
            offset = self.offsets.get(path, 0)
            try:
                if path.stat().st_size < offset:
                    offset = 0
                with path.open("rb") as stream:
                    stream.seek(offset)
                    data = stream.read(256 * 1024)
                complete = data.rfind(b"\n") + 1
                self.offsets[path] = offset + complete
                for line in data[:complete].splitlines():
                    try:
                        item = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(item, dict):
                        self.latest[path] = item
            except OSError:
                continue
        for path in artifact_files(self.root, "result.json"):
            try:
                stamp = path.stat().st_mtime_ns
                if self.result_stamps.get(path) == stamp:
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows = payload.get("cycles") if isinstance(payload, dict) else None
                if isinstance(rows, list) and rows and isinstance(rows[-1], dict):
                    self.latest[path] = rows[-1]
                    for journal in list(self.latest):
                        if journal.parent == path.parent and journal.suffix == ".jsonl":
                            del self.latest[journal]
                self.result_stamps[path] = stamp
            except (OSError, ValueError):
                continue
        return self.text()

    def text(self):
        rows = []
        for path, item in sorted(self.latest.items()):
            if path.suffix == ".jsonl" and any(final.parent == path.parent for final in self.result_stamps):
                continue
            fields = []
            cycle = item.get("physical_cycle", item.get("completed_cycles", item.get("cycle")))
            if _finite(cycle):
                fields.append(f"cycle {cycle:g}")
            for key, label, unit in (("solver_time_s", "solveur", " s"),
                                     ("minimum_capacity_ratio", "capacité min.", ""),
                                     ("target_work_j_per_cycle", "travail cible", " J")):
                if _finite(item.get(key)):
                    fields.append(f"{label} {item[key]:.4g}{unit}")
            ratios = item.get("capacity_ratios")
            values = list(ratios.values()) if isinstance(ratios, dict) else ratios
            if isinstance(values, list) and values and all(_finite(value) for value in values):
                if "minimum_capacity_ratio" not in item:
                    fields.append(f"capacité min. {100 * min(values):.2f} %")
            if isinstance(item.get("certified"), bool):
                fields.append("certifié" if item["certified"] else "non certifié")
            if fields:
                rows.append(f"{path.relative_to(self.root)} : " + " · ".join(fields))
        return "\n".join(rows) or "Initialisation / résolution en cours ; aucune métrique publiée pour le moment."


def cycle_series(root: Path):
    """Extract explicit per-cycle metrics; summary statistics are never expanded."""
    series = {}
    for path in artifact_files(root, "result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        records = payload.get("cycles") if isinstance(payload, dict) else None
        if isinstance(records, list):
            rows = [row for row in records if isinstance(row, dict)]
            if rows:
                series[str(path.relative_to(root))] = rows
    # Attempts are useful while a run is active, but failures must stay visible.
    for path in artifact_files(root, "attempts.jsonl"):
        label = str(path.parent.relative_to(root) / "result.json")
        if label in series:
            continue
        try:
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
            if rows:
                series[label] = rows
        except OSError:
            continue
    return series


def generate_analysis_figures(root: Path) -> list[Path]:
    """Write PNG diagnostics using saved metrics and canonical trajectory NPZ keys.

    These are descriptive figures, not a dynamics certification or a cross-solver
    equivalence test. Physical units are only assigned to documented exports.
    """
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    root = Path(root)
    output = root / "gui-analysis"
    figures = []
    series = cycle_series(root)
    specs = (("solver_time_s", "Temps solveur (s)"),
             ("minimum_capacity_ratio", "Capacité minimale (ratio)"),
             ("achieved_work_j_per_cycle", "Travail produit (J/cycle)"))
    present = [spec for spec in specs if any(any(_finite(row.get(spec[0])) for row in rows) for rows in series.values())]
    if present:
        figure = Figure(figsize=(10, 3 * len(present)), constrained_layout=True)
        FigureCanvasAgg(figure)
        for index, (key, ylabel) in enumerate(present, 1):
            axis = figure.add_subplot(len(present), 1, index)
            for label, rows in series.items():
                xy = [(row.get("physical_cycle", row.get("cycle", i + 1)), row.get(key), row.get("certified"))
                      for i, row in enumerate(rows) if _finite(row.get(key))]
                xy = [(x, y, certified) for x, y, certified in xy if _finite(x)]
                if xy:
                    axis.plot([v[0] for v in xy], [v[1] for v in xy], label=label)
                    failed = [v for v in xy if v[2] is False]
                    if failed:
                        axis.scatter([v[0] for v in failed], [v[1] for v in failed], color="red", marker="x", label=f"{label} non certifié")
            axis.set(xlabel="Cycle physique", ylabel=ylabel)
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
        output.mkdir(parents=True, exist_ok=True)
        path = output / "cycle-metrics.png"
        figure.savefig(path, dpi=150)
        figures.append(path)
    for source in artifact_files(root, "*trajectory.npz"):
        with np.load(source, allow_pickle=False) as archive:
            groups = [("PW (µs)", [key for key in archive.files if key.startswith("controls__last_pulse_width_")], 1e6),
                      ("Force (N)", [key for key in archive.files if key.startswith("states__F_")], 1),
                      ("État de capacité A (unité du modèle)", [key for key in archive.files if key.startswith("states__A_")], 1)]
            groups = [group for group in groups if group[1]]
            if not groups:
                continue
            figure = Figure(figsize=(10, 3 * len(groups)), constrained_layout=True)
            FigureCanvasAgg(figure)
            for index, (ylabel, keys, scale) in enumerate(groups, 1):
                axis = figure.add_subplot(len(groups), 1, index)
                for key in sorted(keys):
                    values = np.asarray(archive[key], dtype=float)
                    for component, row in enumerate(np.atleast_2d(values)):
                        if row.ndim != 1:
                            continue
                        axis.plot(np.arange(row.size), row * scale, label=key.split("__", 1)[1] + (f"[{component}]" if values.ndim > 1 and values.shape[0] > 1 else ""))
                axis.set(xlabel="Échantillon exporté (grille propre à chaque variable)", ylabel=ylabel)
                axis.grid(alpha=.25)
                axis.legend(fontsize=7, ncol=2)
            output.mkdir(parents=True, exist_ok=True)
            relative = source.relative_to(root).with_suffix("")
            path = output / ("-".join(relative.parts) + ".png")
            figure.savefig(path, dpi=150)
            figures.append(path)
    if not figures:
        raise ValueError("Aucune métrique par cycle ni trajectoire NPZ exploitable dans ce dossier.")
    return figures
