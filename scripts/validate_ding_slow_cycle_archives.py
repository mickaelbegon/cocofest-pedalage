#!/usr/bin/env python3
"""Read-only slow-Ding propagation audit of certified witness-cycle archives.

The witness directory must contain a summary.json with explicit certified cycle
records. Shifted prepared-primal exports are deliberately not treated as solved
trajectories. Output is restricted to a new directory selected by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.ding_slow_cycle import coupled_slow_offsets, slow_cycle_map_from_collocation


def validate(witness_directory: Path, model_result: Path, *, intervals: int, duration: float, degree: int) -> dict:
    from casadi import collocation_points

    source_summary = json.loads((witness_directory / "summary.json").read_text())
    certified = {int(row["cycle"]) for row in source_summary["cycles"] if row.get("certified") is True}
    model_document = json.loads(model_result.read_text())
    configured = model_document["configured_model"]["configured_muscle_parameters"]
    protocol_path = witness_directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    source_model = protocol["request"]["checkpoint"]
    model_path = Path(model_document["configured_model"]["model_config_path"])
    if model_path.resolve() != Path(source_model["model_path"]).resolve():
        raise ValueError("Witness and parameter result come from different model configurations")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != source_model["model_sha256"]:
        raise ValueError("Witness model configuration has changed since the archived experiment")
    # Results supply the defaults as actually configured, while the model JSON
    # may contain only overrides. Every override must agree with the result.
    for name, overrides in json.loads(model_path.read_text())["muscles"].items():
        for key, value in overrides.items():
            if key in configured[name] and configured[name][key] != value:
                raise ValueError(f"Archived parameter mismatch for {name}.{key}")
    parameters = {name: DingFatigueParameters(
        a_rest=value["a_scale"], tau1_rest=value["tau1_rest"], km_rest=value["km_rest"],
        alpha_a=value["alpha_a"], alpha_tau1=value["alpha_tau1"], alpha_km=value["alpha_km"],
        tau_fat=value["tau_fat"],
    ) for name, value in configured.items()}
    nodes = np.asarray([0.0, *collocation_points(degree, "radau")])
    stride = degree + 1
    rows, provenance, map_times = [], [], []
    chained, averaged, last_end = {}, {}, {}
    previous_cycle = None
    for cycle in sorted(certified):
        path = witness_directory / f"witness-cycle-{cycle}.npz"
        if not path.exists():
            continue
        provenance.append({"cycle": cycle, "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        with np.load(path, allow_pickle=False) as archive:
            for name, p in parameters.items():
                force = np.asarray(archive[f"states__F_{name}"], dtype=float).reshape(-1)
                slow = np.stack([np.asarray(archive[f"states__{state}_{name}"], dtype=float).reshape(-1)
                                 for state in ("A", "Tau1", "Km")])
                if force.size != intervals*stride+1 or slow.shape != (3, force.size):
                    raise ValueError(f"{path}: grid incompatible with supplied intervals/degree")
                begin, end = slow[:, 0], slow[:, -1]
                if previous_cycle is None or previous_cycle != cycle-1:
                    chained[name] = begin.copy()
                    averaged[name] = begin.copy()
                    continuity = np.zeros(3)
                else:
                    continuity = (begin - last_end[name]) / p.rest_state
                    if np.max(np.abs(continuity)) > 1e-5:
                        raise ValueError(f"{path}: cycle {cycle} is not continuous for {name}: {continuity}")
                started = time.perf_counter()
                cycle_map = slow_cycle_map_from_collocation(force[:-1].reshape(intervals, stride),
                    np.full(intervals, duration/intervals), nodes, p)
                prediction = cycle_map.propagate(begin)
                chained[name] = cycle_map.propagate(chained[name])
                averaged[name] = cycle_map.propagate_with_cycle_average(averaged[name])
                map_times.append(time.perf_counter()-started)
                offset_defect = coupled_slow_offsets(end, p)-cycle_map.decay*coupled_slow_offsets(begin, p)
                rows.append({"cycle": cycle, "muscle": name, "state_names": ["A", "Tau1", "Km"],
                    "observed_start": begin.tolist(), "observed_end": end.tolist(),
                    "predicted_end": prediction.tolist(), "chained_prediction": chained[name].tolist(),
                    "cycle_average_chained_prediction": averaged[name].tolist(),
                    "one_cycle_error_relative_to_rest": ((prediction-end)/p.rest_state).tolist(),
                    "chained_error_relative_to_rest": ((chained[name]-end)/p.rest_state).tolist(),
                    "cycle_average_chained_error_relative_to_rest": ((averaged[name]-end)/p.rest_state).tolist(),
                    "continuity_error_relative_to_rest": continuity.tolist(),
                    "coupled_offset_defect_relative_to_rest": (offset_defect/p.rest_state[1:]).tolist(),
                    "mean_force_n": cycle_map.force_integral/duration,
                    "weighted_force_integral_n_s": cycle_map.weighted_force_integral,
                    "minimum_saved_force_n": float(force.min()),
                })
                last_end[name] = end
        previous_cycle = cycle
    if not rows:
        raise ValueError("No certified witness-cycle archives available")
    metrics = {}
    for metric in ("one_cycle_error_relative_to_rest", "chained_error_relative_to_rest",
                   "cycle_average_chained_error_relative_to_rest", "continuity_error_relative_to_rest",
                   "coupled_offset_defect_relative_to_rest"):
        values = np.asarray([row[metric] for row in rows])
        metrics[metric] = {"max_abs": np.max(np.abs(values), axis=0).tolist(),
                           "rms": np.sqrt(np.mean(values**2, axis=0)).tolist()}
    return {"schema": "ding-slow-polynomial-force-archive-validation-v1",
        "interpretation": "Measured conditional replay errors on certified discrete NLP trajectories; not an endurance forecast or independent full-Ding integration validation.",
        "assumptions": {"collocation": "Radau", "degree": degree, "intervals": intervals,
                        "cycle_duration_seconds": duration, "force_interpolant": "degree-d polynomial on d+1 archived nodes",
                        "cycle_mean_ablation": "same ordinary polynomial force integral, constant over whole cycle",
                        "state_error_normalization": "each state's resting value; not percent fatigue increment",
                        "future_force_policy": "observed force from each successive archived cycle"},
        "model_result_path": str(model_result.resolve()),
        "model_result_sha256": hashlib.sha256(model_result.read_bytes()).hexdigest(),
        "model_parameter_fingerprint": model_document["configured_model"]["muscle_parameter_fingerprint"],
        "source_model_sha256": source_model["model_sha256"],
        "source_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "witness_directory": str(witness_directory.resolve()),
        "certified_archives_count": len(provenance), "muscle_cycle_count": len(rows),
        "source_archives": provenance, "metrics": metrics,
        "map_timing_seconds_per_muscle_cycle_including_build": {"median": float(np.median(map_times)),
            "p95": float(np.quantile(map_times,.95)), "sum": float(sum(map_times))},
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness-directory", required=True, type=Path)
    parser.add_argument("--model-result", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--intervals", default=30, type=int)
    parser.add_argument("--duration", default=1.0, type=float)
    parser.add_argument("--degree", default=5, type=int)
    args = parser.parse_args()
    report = validate(args.witness_directory, args.model_result, intervals=args.intervals,
                      duration=args.duration, degree=args.degree)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    (args.output_directory/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("certified_archives_count", "metrics", "map_timing_seconds_per_muscle_cycle_including_build")}, indent=2))


if __name__ == "__main__":
    main()
