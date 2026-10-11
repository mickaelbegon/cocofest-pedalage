#!/usr/bin/env python3
"""Repeated, CPU-pinned A/B of Bioptim numeric-vector dispatch.

Both conditions use the same unilateral prescribed K6-I1-DL RHO and retain
the exact local-Ding solver archives.  The online process writes only a JSONL
call journal and a deliberately non-certifying completion record.  This
script materializes the journals and performs all trajectory comparisons only
after each RHO has completed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.materialize_rho_online_journal import load_journal, materialize


def _percentile(values: list[float], percentile: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values, float), percentile))


def _read_run(path: Path) -> dict:
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))["results"][0]
    metrics = load_journal(path / "symbolic-audit" / "nlp0_metrics.jsonl")
    samples = result["execution_timing"]["rho_complete_iteration_wall_time_samples_s"]
    if len(metrics) != len(samples):
        raise ValueError(f"Metric/completion-count mismatch in {path}")
    if (result.get("mode") != "rho_deferred_summary"
            or result.get("execution_timing", {}).get("offline_finalizer_required") is not True):
        raise ValueError(f"{path} was not run with the deferred-summary contract")
    return {"result": result, "metrics": metrics, "samples": [float(value) for value in samples]}


def _exact_archive_difference(left: Path, right: Path, calls: int) -> dict:
    maximum = {field: 0.0 for field in ("x", "g", "lam_x", "lam_g")}
    for call in range(calls):
        left_path = left / "symbolic-audit" / f"nlp0_call{call}.npz"
        right_path = right / "symbolic-audit" / f"nlp0_call{call}.npz"
        with np.load(left_path) as a, np.load(right_path) as b:
            if set(a.files) != set(b.files):
                raise ValueError(f"Archive field mismatch at call {call}")
            for field in maximum:
                if not np.array_equal(a[field], b[field]):
                    maximum[field] = max(
                        maximum[field],
                        float(np.max(np.abs(np.asarray(a[field]) - np.asarray(b[field])))),
                    )
    return maximum


def summarize(root: Path, replicates: int) -> dict:
    root = Path(root).resolve(strict=True)
    rows, scalar_times, cached_times, scalar_ipopt, cached_ipopt = [], [], [], [], []
    for index in range(1, replicates + 1):
        replica = root / f"replicate-{index:02d}"
        scalar, cached = _read_run(replica / "scalar"), _read_run(replica / "cached")
        if len(scalar["metrics"]) != len(cached["metrics"]):
            raise ValueError(f"RHO call counts differ in replicate {index}")
        errors = _exact_archive_difference(replica / "scalar", replica / "cached", len(scalar["metrics"]))
        if any(value != 0.0 for value in errors.values()):
            raise ValueError(f"Numeric dispatch changed an archived solver vector in replicate {index}: {errors}")
        sh, ch = scalar["samples"][1:], cached["samples"][1:]
        si = [float(row["timings"]["t_wall_total"]) for row in scalar["metrics"][1:]]
        ci = [float(row["timings"]["t_wall_total"]) for row in cached["metrics"][1:]]
        rows.append({
            "replicate": index,
            "calls": len(scalar["metrics"]),
            "exact_archive_max_abs_error": errors,
            "scalar_hot_complete_median_s": float(np.median(sh)),
            "cached_hot_complete_median_s": float(np.median(ch)),
            "scalar_hot_ipopt_median_s": float(np.median(si)),
            "cached_hot_ipopt_median_s": float(np.median(ci)),
        })
        scalar_times.extend(sh); cached_times.extend(ch); scalar_ipopt.extend(si); cached_ipopt.extend(ci)
    return {
        "protocol": (
            "Sequential CPU-pinned unilateral K6-I1-DL RHO A/B; 30 Hz, Radau-5, "
            "unit fatigue, IPOPT/MA57, one numeric thread. Cache stores geometry only; "
            "values/scaling are reread every call."
        ),
        "replicates": rows,
        "pooled": {
            "hot_complete_scalar_median_s": float(np.median(scalar_times)),
            "hot_complete_cached_median_s": float(np.median(cached_times)),
            "hot_complete_cached_p90_s": _percentile(cached_times, 90),
            "hot_complete_relative_change": float(np.median(cached_times) / np.median(scalar_times) - 1.0),
            "hot_ipopt_scalar_median_s": float(np.median(scalar_ipopt)),
            "hot_ipopt_cached_median_s": float(np.median(cached_ipopt)),
            "hot_ipopt_relative_change": float(np.median(cached_ipopt) / np.median(scalar_ipopt) - 1.0),
        },
        "interpretation": (
            "Exact archive equality validates the dispatch implementation for this condition. "
            "Deferred records are not mechanical or physiological certificates; run the offline finalizer "
            "and the intended independent audits before any scientific/endurance conclusion."
        ),
    }


def _run(command: list[str], *, environment: dict[str, str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def run_campaign(root: Path, *, replicates: int, cycles: int, cpu: int, python: str) -> dict:
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to reuse a campaign directory: {root}")
    if cycles < 2 or replicates < 1:
        raise ValueError("At least two cycles and one replicate are required")
    root.mkdir(parents=True)
    base_environment = {
        **os.environ,
        "PYTHON": python,
        "MPLCONFIGDIR": "/tmp/mpl-cocofest",
        "COCOFEST_DEFERRED_RHO_SUMMARY": "1",
        "DING_ONLINE_JOURNAL": "1",
    }
    for index in range(1, replicates + 1):
        replica = root / f"replicate-{index:02d}"
        replica.mkdir(parents=True, exist_ok=False)
        for mode, enabled in (("scalar", "0"), ("cached", "1")):
            output = replica / mode
            environment = {**base_environment, "BIOPTIM_NUMERIC_VECTOR_DISPATCH": enabled}
            _run(
                ["taskset", "-c", str(cpu), "bash", "scripts/run_prescribed_ding_local_propagated.sh",
                 "compiled", str(output), str(cycles), str(output / "cache")],
                environment=environment,
                log=output.parent / f"{mode}.log",
            )
            materialize(output / "symbolic-audit" / "nlp0_metrics.jsonl")
    report = summarize(root, replicates)
    (root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        report = summarize(args.output, args.replicates)
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    report = run_campaign(args.output, replicates=args.replicates, cycles=args.cycles,
                          cpu=args.cpu, python=args.python)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
