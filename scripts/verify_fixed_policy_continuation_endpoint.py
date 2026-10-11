#!/usr/bin/env python3
"""Freshly restore a fixed-policy continuation endpoint and re-solve one nominal cycle.

The solve is diagnostic only: it never advances or overwrites the continuation
state.  A failed re-solve is not a physiological-failure certificate.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.task_reserve_probe_adapter import independent_full_nlp_audit
from cocofest.simulation.independent_arms_process import _atomic_json
from cocofest.simulation.rho_restart_checkpoint import restore_prepared_checkpoint
from scripts.run_local_task_reserve_branch import _build_runtime, _solve, load_branch_endpoint


def run(result_path: Path, *, cpu: int, output: Path, tolerance: float = 1e-5) -> Path:
    result_path = Path(result_path).resolve(strict=True)
    result = json.loads(result_path.read_text())
    if result.get("kind") != "fixed_policy_branch_continuation" or not result.get("completed"):
        raise ValueError("Expected a completed fixed-policy continuation result")
    if cpu not in os.sched_getaffinity(0):
        raise ValueError("Requested CPU is outside process affinity")
    endpoint = result["last_certified_endpoint"]
    archive = Path(endpoint["primal_path"]).resolve(strict=True)
    if sha256(archive.read_bytes()).hexdigest() != endpoint["sha256"]:
        raise ValueError("Last certified endpoint archive changed")
    source_receipt = Path(result["source_endpoint_receipt"]).resolve(strict=True)
    if sha256(source_receipt.read_bytes()).hexdigest() != result["source_endpoint_receipt_sha256"]:
        raise ValueError("Source endpoint receipt changed")
    _, payload = load_branch_endpoint(source_receipt)
    os.sched_setaffinity(0, {cpu})
    runtime = _build_runtime(payload, "left", cpu=cpu)
    program = runtime["nmpc"]
    restored = restore_prepared_checkpoint(archive, program, completed_cycles=endpoint["completed_cycles"])
    if restored.get("restored_problem_sha256") != endpoint["prepared_problem_sha256"]:
        raise RuntimeError("Fresh endpoint restore signature differs")
    actual = np.asarray(program.parameter_init["rho_fatigue_weights"].init, float).reshape(-1)
    expected = np.asarray(result["weights"], float)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-12):
        raise RuntimeError("Fresh endpoint restore changed fatigue weights")
    work = float(result["nominal_work_j"])
    started = perf_counter()
    solution = _solve(program, runtime["solver"])
    audit = independent_full_nlp_audit(solution, program, target_work_j=work, tolerance=tolerance)
    report = {"schema_version": 1, "kind": "fresh_fixed_policy_continuation_endpoint_check",
              "source_result": str(result_path), "source_result_sha256": sha256(result_path.read_bytes()).hexdigest(),
              "endpoint": endpoint, "restored": restored, "weights": actual.tolist(), "cpu": cpu,
              "next_nominal_cycle_solver_status": str(getattr(solution, "status", None)),
              "next_nominal_cycle_independent_audit": dict(audit.detail),
              "next_nominal_cycle_feasible_witness": bool(audit.passed),
              "solve_seconds": perf_counter() - started,
              "interpretation": "a failed fresh solve remains indeterminate; no state was advanced"}
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, report)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args(argv)
    print(run(args.result, cpu=args.cpu, output=args.output, tolerance=args.tolerance))


if __name__ == "__main__":
    main()
