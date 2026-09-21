#!/usr/bin/env python3
"""Summarize a multi-model RHO / RHO-Physio / RHO-PACE campaign.

The analysis is deliberately read-only with respect to the campaign itself:
missing, malformed and failed solver artifacts are reported explicitly and are
never converted into zero-duration observations or physiological failures.
It supports a live campaign directory, so it can be re-run while independent
model-condition arms are still pending.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cocofest-matplotlib")

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("rho", "rho-physio", "rho-pace")
LABELS = {"rho": "RHO", "rho-physio": "RHO-Physio", "rho-pace": "RHO-PACE"}
COLORS = {"rho": "#4c78a8", "rho-physio": "#f58518", "rho-pace": "#54a24b"}
MUSCLE_COLORS = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#72b7b2")


def _finite(value: Any) -> float | None:
    """Return a finite number, or None for absent/non-numeric values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _condition(job: dict[str, Any]) -> str | None:
    condition = job.get("condition")
    if condition in CONDITIONS:
        return condition
    # The older manifest named the unweighted RHO arm baseline_rho.
    return "rho" if job.get("arm_id") == "baseline_rho" or job.get("id", "").endswith("/baseline_rho") else None


def _model_id(job: dict[str, Any]) -> str:
    if isinstance(job.get("model_id"), str):
        return job["model_id"]
    job_id = job.get("id")
    return job_id.rsplit("/", 1)[0] if isinstance(job_id, str) and "/" in job_id else "unlabeled-model"


def _window_cycle(window: dict[str, Any], fallback: int) -> int:
    for key in ("rho", "cycle_index"):
        value = window.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return value
    value = window.get("window")
    return value + 1 if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else fallback


def _windows(job: dict[str, Any], result: dict[str, Any], issues: list[str]) -> list[dict[str, Any]]:
    rows = result.get("results")
    if not isinstance(rows, list) or not rows:
        issues.append("result.results absent or empty")
        return []
    if len(rows) > 1:
        issues.append("multiple solver result rows: retaining each reported window")
    output = []
    for solver_index, solver_row in enumerate(rows):
        if not isinstance(solver_row, dict):
            issues.append(f"result.results[{solver_index}] is not an object")
            continue
        windows = solver_row.get("windows")
        if not isinstance(windows, list):
            issues.append(f"result.results[{solver_index}].windows absent")
            continue
        for fallback, window in enumerate(windows, start=1):
            if not isinstance(window, dict):
                issues.append(f"non-object window in solver row {solver_index}")
                continue
            cycle = _window_cycle(window, fallback)
            output.append({
                "model_id": _model_id(job), "condition": _condition(job), "cycle": cycle,
                "solver": solver_row.get("solver"), "solver_time_s": _finite(window.get("solver_time_s")),
                "wall_time_s": _finite(window.get("wall_time_s")),
                "effective_wall_time_s": _finite(window.get("effective_wall_time_s")),
                "iterations": _finite(window.get("iterations")),
                "solver_converged": window.get("solver_converged") is True,
                "validated": window.get("validated") is True,
                "primal_feasible": window.get("primal_feasible") is True,
                "status": window.get("status"),
            })
    return sorted(output, key=lambda row: (row["model_id"], row["condition"], row["cycle"]))


def _journal_weights(job: dict[str, Any], issues: list[str]) -> list[dict[str, Any]]:
    path_text = job.get("weights_journal_path")
    if not path_text:
        return []
    path = Path(path_text)
    if not path.is_file():
        issues.append(f"weights journal pending: {path}")
        return []
    rows: list[dict[str, Any]] = []
    muscle_names: list[str] | None = None
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError(f"line {line_number} is not an object")
            if event.get("event") == "configuration" and isinstance(event.get("muscle_names"), list):
                muscle_names = [str(name) for name in event["muscle_names"]]
            if event.get("event") != "boundary":
                continue
            cycle = event.get("cycle_index")
            weights = event.get("weights")
            if (isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 0
                    or not isinstance(weights, list) or not muscle_names or len(weights) != len(muscle_names)):
                issues.append(f"ignored malformed weight boundary at {path}:{line_number}")
                continue
            for muscle, weight in zip(muscle_names, weights, strict=True):
                number = _finite(weight)
                if number is not None:
                    rows.append({"model_id": _model_id(job), "condition": _condition(job),
                                 "cycle": cycle + 1, "muscle": muscle, "weight": number,
                                 "status": event.get("status")})
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        issues.append(f"unreadable weights journal {path}: {error}")
    return rows


def collect(manifest_path: Path) -> dict[str, Any]:
    """Read available artifacts referenced by a manifest; never raise for partial arms."""
    manifest = _read_json(manifest_path)
    jobs = manifest.get("arms")
    if not isinstance(jobs, list):
        raise ValueError("Manifest needs an arms list")
    accepted = [job for job in jobs if isinstance(job, dict) and _condition(job) in CONDITIONS]
    if not accepted:
        raise ValueError("Manifest has no RHO, RHO-Physio, or RHO-PACE arms")
    data: dict[str, Any] = {"schema_version": 1, "manifest": str(manifest_path.resolve()),
                            "campaign_id": manifest.get("campaign_id"), "resistance_nm": manifest.get("resistance_nm"),
                            "requested_cycles": manifest.get("cycles"), "conditions": list(CONDITIONS),
                            "models": sorted({_model_id(job) for job in accepted}), "arms": [],
                            "windows": [], "weights": [], "diagnostics": []}
    for job in accepted:
        issues: list[str] = []
        result_path = Path(job["result_path"]) if isinstance(job.get("result_path"), str) else None
        arm = {"id": job.get("id"), "model_id": _model_id(job), "condition": _condition(job),
               "result_path": None if result_path is None else str(result_path), "state": "pending"}
        if result_path is None:
            issues.append("missing result_path in manifest")
        elif not result_path.is_file():
            issues.append(f"result pending: {result_path}")
        else:
            try:
                result = _read_json(result_path)
                rows = _windows(job, result, issues)
                data["windows"].extend(rows)
                arm["state"] = "available" if rows else "unusable_result"
                arm["reported_error"] = next((row.get("error") for row in result.get("results", [])
                                               if isinstance(row, dict) and row.get("error")), None)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                issues.append(f"unreadable result {result_path}: {error}")
                arm["state"] = "unusable_result"
        weight_rows = _journal_weights(job, issues)
        data["weights"].extend(weight_rows)
        arm["window_count"] = sum(1 for row in data["windows"] if row["model_id"] == arm["model_id"] and row["condition"] == arm["condition"])
        arm["issues"] = issues
        data["arms"].append(arm)
        data["diagnostics"].extend(f"{arm['id']}: {issue}" for issue in issues)
    return data


def _rolling(values: list[dict[str, Any]], window: int, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_cycle: dict[int, list[float]] = {}
    for row in values:
        value = _finite(row.get(metric))
        if value is not None:
            by_cycle.setdefault(int(row["cycle"]), []).append(value)
    xs, medians, lows, highs = [], [], [], []
    for cycle in sorted(by_cycle):
        population = [value for other, entries in by_cycle.items() if abs(other - cycle) <= window // 2 for value in entries]
        if population:
            xs.append(cycle)
            lows.append(float(np.quantile(population, .25)))
            medians.append(float(np.median(population)))
            highs.append(float(np.quantile(population, .75)))
    return np.asarray(xs), np.asarray(lows), np.asarray(medians), np.asarray(highs)


def _plot_timing(data: dict[str, Any], output: Path, rolling_window: int) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8), layout="constrained")
    metrics = (("solver_time_s", "Solver time per RHO [s]"), ("effective_wall_time_s", "Effective wall time per RHO [s]"))
    for axis, (metric, ylabel) in zip(axes, metrics, strict=True):
        failure_label_added = False
        for condition in CONDITIONS:
            rows = [row for row in data["windows"] if row["condition"] == condition]
            for model in data["models"]:
                series = sorted((row for row in rows if row["model_id"] == model and _finite(row.get(metric)) is not None), key=lambda row: row["cycle"])
                if series:
                    axis.plot([row["cycle"] for row in series], [row[metric] for row in series],
                              color=COLORS[condition], alpha=.25, linewidth=.9, marker=".", markersize=3)
                    failed = [row for row in series if not (row["solver_converged"] and row["validated"])]
                    if failed:
                        axis.scatter([row["cycle"] for row in failed], [row[metric] for row in failed],
                                     marker="x", color="#c43c39", s=18, linewidths=.9,
                                     label="unvalidated/failed attempt" if not failure_label_added else None)
                        failure_label_added = True
            x, low, median, high = _rolling(rows, rolling_window, metric)
            if x.size:
                axis.fill_between(x, low, high, color=COLORS[condition], alpha=.16)
                axis.plot(x, median, color=COLORS[condition], linewidth=2, label=f"{LABELS[condition]} median (IQR)")
        axis.set_ylabel(ylabel)
        # Individual IPOPT iterations can have rare expensive restorations;
        # a logarithmic scale retains their visibility without flattening the
        # clinically relevant 0.5--2 s band.
        axis.set_yscale("log")
        axis.grid(alpha=.25)
        axis.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel(f"Cycle RHO (rolling window: {rolling_window} cycles; thin lines: model trajectories)")
    fig.suptitle("Convergence timing over RHO cycles")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_weights(data: dict[str, Any], output: Path) -> bool:
    rows = data["weights"]
    if not rows:
        return False
    muscles = sorted({row["muscle"] for row in rows})
    fig, axes = plt.subplots(len(muscles), 1, sharex=True, figsize=(10, max(3, 2.1 * len(muscles))), layout="constrained")
    axes = np.atleast_1d(axes)
    for axis, muscle, color in zip(axes, muscles, MUSCLE_COLORS, strict=False):
        for condition, style in (("rho-physio", "--"), ("rho-pace", "-")):
            for model in data["models"]:
                series = sorted((row for row in rows if row["muscle"] == muscle and row["condition"] == condition and row["model_id"] == model), key=lambda row: row["cycle"])
                if series:
                    axis.step([row["cycle"] for row in series], [row["weight"] for row in series],
                              where="post", color=color, linestyle=style, alpha=.55, linewidth=1.2)
        axis.axhline(1., color="black", linewidth=.7, alpha=.35)
        axis.set_ylabel(f"{muscle}\nrelative weight")
        axis.grid(alpha=.25)
    axes[0].plot([], [], color="black", linestyle="--", label="RHO-Physio (fixed)")
    axes[0].plot([], [], color="black", linestyle="-", label="RHO-PACE (adaptive)")
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Cycle RHO (weight applied at its boundary)")
    fig.suptitle("Physiological and adaptive cost weights over cycles")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return True


def analyze(manifest: Path, output_dir: Path, rolling_window: int) -> dict[str, Any]:
    if rolling_window < 1 or rolling_window % 2 == 0:
        raise ValueError("rolling_window must be a positive odd integer")
    data = collect(manifest.expanduser().resolve())
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_path = output_dir / "rho-convergence-timing.png"
    _plot_timing(data, timing_path, rolling_window)
    weights_path = output_dir / "rho-weight-evolution.png"
    data["figures"] = {"timing": str(timing_path), "weights": str(weights_path) if _plot_weights(data, weights_path) else None}
    data["summary"] = {
        "arms_available": sum(arm["state"] == "available" for arm in data["arms"]),
        "arms_pending_or_unusable": sum(arm["state"] != "available" for arm in data["arms"]),
        "observed_windows": len(data["windows"]),
        "validated_windows": sum(row["validated"] for row in data["windows"]),
        "solver_converged_windows": sum(row["solver_converged"] for row in data["windows"]),
        "weight_observations": len(data["weights"]),
        "rolling_window_cycles": rolling_window,
    }
    _atomic_json(output_dir / "rho-condition-analysis.json", data)
    return data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rolling-window", type=int, default=51)
    args = parser.parse_args(argv)
    report = analyze(args.manifest, args.output_dir, args.rolling_window)
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
