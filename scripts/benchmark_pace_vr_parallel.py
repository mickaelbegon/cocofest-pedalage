#!/usr/bin/env python3
"""Compare current threaded PACE-VR dispatch with isolated numerical processes.

This is an offline timing experiment on archived certified snapshots. It does
not change the RHO's transport, process lifecycle, or application policy.
"""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_pace_vr_rollout import load_case
from cocofest.optimization.pace_vr import run_pace_vr_snapshot
from cocofest.optimization.pace_vr_async import evaluate_bilateral_pace_vr


def _process_arm(task):
    side, snapshot, cpu = task
    os.sched_setaffinity(0, {cpu})
    return side, run_pace_vr_snapshot(snapshot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--right-source", type=Path, required=True)
    parser.add_argument("--left-source", type=Path, required=True)
    parser.add_argument("--source-cycle", type=int, default=1)
    parser.add_argument("--cpu-ids", type=int, nargs=2, required=True)
    parser.add_argument("--budget-seconds", type=float, default=16.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.cpu_ids)) != 2 or not set(args.cpu_ids) <= os.sched_getaffinity(0):
        parser.error("Two distinct allowed CPUs are required.")
    if args.budget_seconds <= 0 or not math.isfinite(args.budget_seconds):
        parser.error("budget-seconds must be finite and positive")
    os.sched_setaffinity(0, set(args.cpu_ids))
    snapshots, weights = {}, {}
    for side, source in (("right", args.right_source), ("left", args.left_source)):
        supervisor, payload, snapshot = load_case(source, args.source_cycle)
        snapshots[side] = snapshot
        weights[side] = payload.get("weights") or [1.] * supervisor.muscles
    report = {"source_cycle": args.source_cycle, "cpu_ids": args.cpu_ids,
              "budget_seconds": args.budget_seconds, "modes": {},
              "same_snapshot_digests": {side: snapshot["context_digest"] for side, snapshot in snapshots.items()},
              "rho_transport_changed": False}
    for mode in ("current_threads", "isolated_spawn_processes"):
        started = time.perf_counter()
        deadline = time.monotonic() + args.budget_seconds
        docs = {side: {**snapshot, "deadline_monotonic": deadline} for side, snapshot in snapshots.items()}
        if mode == "current_threads":
            _, audit = evaluate_bilateral_pace_vr(dict(
                snapshots=docs, incumbent_weights=weights, budget_seconds=args.budget_seconds,
                maximum_log_step=math.log(1.25), supervisor_cpu_ids=args.cpu_ids))
            outcomes = audit["arms"]
        else:
            tasks = [(side, docs[side], cpu) for side, cpu in zip(("right", "left"), args.cpu_ids)]
            with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
                outcomes = dict(pool.map(_process_arm, tasks))
        report["modes"][mode] = dict(wall_seconds=time.perf_counter() - started,
            includes_pool_startup_and_shutdown=True,
            arms={side: {key: outcome[key] for key in (
                "runtime_s", "deadline_met", "accepted", "feasible_prefix_cycles", "status", "local_fit")}
                for side, outcome in outcomes.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({mode: {"wall_seconds": value["wall_seconds"], "arms": {
        side: {key: arm[key] for key in ("runtime_s", "deadline_met", "accepted", "feasible_prefix_cycles")}
        for side, arm in value["arms"].items()}} for mode, value in report["modes"].items()}, indent=2))


if __name__ == "__main__":
    main()
