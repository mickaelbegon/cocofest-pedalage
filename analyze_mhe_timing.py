from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def _load_data(path: Path) -> dict:
    if path.suffix == ".pkl":
        with open(path, "rb") as file:
            return pickle.load(file)
    if path.suffix == ".npz":
        loaded = np.load(path, allow_pickle=True)
        data = {}
        for key in loaded.files:
            value = loaded[key]
            if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
                data[key] = value.item()
            else:
                data[key] = value.tolist() if hasattr(value, "tolist") else value
        return data
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _stats(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _speed_stats(times_s: list[float]) -> dict:
    arr = np.array(times_s, dtype=float)
    cycles_per_second = 1.0 / arr
    return {
        "seconds_per_cycle_mean": float(arr.mean()),
        "seconds_per_cycle_median": float(np.median(arr)),
        "cycles_per_second_mean": float(cycles_per_second.mean()),
        "cycles_per_second_median": float(np.median(cycles_per_second)),
    }


def _slice_1_based(values: list, start_cycle: int, end_cycle: int) -> list:
    start_idx = max(start_cycle - 1, 0)
    end_idx = end_cycle
    return values[start_idx:end_idx]


def main():
    parser = argparse.ArgumentParser(description="Analyze MHE timing over a selected cycle/window range.")
    parser.add_argument("result_file", type=Path, help="Path to the saved .npz or .pkl result file.")
    parser.add_argument("--start-cycle", type=int, default=4, help="First kept cycle/window to include (1-based).")
    parser.add_argument("--end-cycle", type=int, default=10, help="Last kept cycle/window to include (1-based).")
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path to save the summary as JSON.")
    args = parser.parse_args()

    data = _load_data(args.result_file)

    solver_times = list(data.get("solving_time_per_ocp", []))
    wall_times = list(data.get("real_time_per_ocp", solver_times))
    iterations = list(data.get("iter_per_ocp", []))
    objectives = list(data.get("objective_values_per_ocp", []))
    statuses = list(data.get("convergence_status", []))

    selected_solver_times = _slice_1_based(solver_times, args.start_cycle, args.end_cycle)
    selected_wall_times = _slice_1_based(wall_times, args.start_cycle, args.end_cycle)
    selected_iterations = _slice_1_based(iterations, args.start_cycle, args.end_cycle)
    selected_objectives = _slice_1_based(objectives, args.start_cycle, args.end_cycle)
    selected_statuses = _slice_1_based(statuses, args.start_cycle, args.end_cycle)

    summary = {
        "result_file": str(args.result_file),
        "solver_config": data.get("solver_config"),
        "selected_cycles_1_based": [args.start_cycle, args.end_cycle],
        "n_selected_windows": len(selected_wall_times),
        "selected_window_indices_0_based": [args.start_cycle - 1, args.end_cycle - 1],
        "all_windows": len(wall_times),
        "wall_time_stats_s": _stats(selected_wall_times) if selected_wall_times else None,
        "solver_time_stats_s": _stats(selected_solver_times) if selected_solver_times else None,
        "iteration_stats": _stats(selected_iterations) if selected_iterations else None,
        "objective_stats": _stats(selected_objectives) if selected_objectives else None,
        "speed_stats": _speed_stats(selected_wall_times) if selected_wall_times else None,
        "n_converged_windows": int(sum(1 for status in selected_statuses if status == 0)),
        "n_failed_windows": int(sum(1 for status in selected_statuses if status != 0)),
    }

    phase_records = data.get("solver_phase_records")
    if phase_records:
        if isinstance(phase_records, np.ndarray):
            phase_records = phase_records.tolist()
        normalized_phase_records = []
        for record in phase_records:
            if isinstance(record, np.ndarray) and record.shape == ():
                record = record.item()
            normalized_phase_records.append(record)

        selected_phase_records = [
            record
            for record in normalized_phase_records
            if args.start_cycle - 1 <= int(record["window"]) <= args.end_cycle - 1
        ]
        phase_summary = {}
        for phase_name in sorted({record["phase"] for record in selected_phase_records}):
            phase_subset = [record for record in selected_phase_records if record["phase"] == phase_name]
            phase_summary[phase_name] = {
                "wall_time_stats_s": _stats([record["wall_time_s"] for record in phase_subset]),
                "solver_time_stats_s": _stats([record["solver_time_s"] for record in phase_subset]),
                "iteration_stats": _stats([record["iterations"] for record in phase_subset]),
                "n_converged_windows": int(sum(1 for record in phase_subset if record["status"] == 0)),
            }
        summary["phase_summary"] = phase_summary

    print("MHE timing summary")
    print(json.dumps(summary, indent=2))

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
