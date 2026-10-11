#!/usr/bin/env python3
"""Fresh-process mechanical-margin audit of an authenticated ablation endpoint."""
from dataclasses import asdict
from hashlib import sha256
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cocofest.optimization.task_load_margin_oracle import configure_load_objective, audit_load_solution
from cocofest.simulation.rho_restart_checkpoint import restore_prepared_checkpoint
from cocofest.simulation.independent_arms_process import _atomic_json
from scripts.probe_independent_rho_task_reserve import _load_arm
from scripts.run_checkpoint_fatigue_ablation import build_runtime
from scripts.run_local_task_reserve_branch import _solve


def authenticated_endpoint(result_path, offset):
    result = json.loads(Path(result_path).read_text())
    receipt = Path(result["source_receipt"])
    if sha256(receipt.read_bytes()).hexdigest() != result["source_receipt_sha256"]:
        raise ValueError("Source receipt changed")
    source, payload = _load_arm(receipt, result["side"])
    prefix = [r for r in result["cycles"] if r["offset"] <= offset]
    if ([r["offset"] for r in prefix] != list(range(1, offset + 1))
            or not all(r["certified"] and r["maximum_normalized_violation"] is not None
                       and 0 <= r["maximum_normalized_violation"] <= result["audit_tolerance"] for r in prefix)):
        raise ValueError("Uncertified endpoint prefix")
    for row in prefix:
        if sha256(Path(row["witness"]).read_bytes()).hexdigest() != row["witness_sha256"]:
            raise ValueError("Trajectory witness changed")
    endpoint, = [e for e in result["prepared_endpoints"] if e["offset"] == offset]
    if sha256(Path(endpoint["primal_path"]).read_bytes()).hexdigest() != endpoint["sha256"]:
        raise ValueError("Endpoint archive changed")
    if endpoint["completed_cycles"] != source.completed_cycles + offset:
        raise ValueError("Endpoint cycle mismatch")
    return result, source, payload, endpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--offset", type=int, default=20)
    parser.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args()
    result, source, payload, endpoint = authenticated_endpoint(args.result, args.offset)
    output = args.result.parent / f"margin-cycle-{args.offset}.json"
    if output.exists():
        raise FileExistsError(output)
    os.sched_setaffinity(0, {args.cpu})
    runtime, _ = build_runtime(payload, result["side"], "integral_quadratic")
    program = runtime["nmpc"]
    restored = restore_prepared_checkpoint(Path(endpoint["primal_path"]), program, completed_cycles=endpoint["completed_cycles"])
    if restored["restored_problem_sha256"] != endpoint["prepared_problem_sha256"] or not restored["stimulation_history_complete"]:
        raise ValueError("Fresh endpoint restoration differs")
    work = result["nominal_work_j"]
    configure_load_objective(program, nominal_work_j=work, upper_bound=3.)
    start = time.perf_counter()
    solution = _solve(program, runtime["solver"])
    load, audit, kkt = audit_load_solution(solution, program, nominal_work_j=work, upper_bound=3., tolerance=1e-6)
    witness = args.result.parent / f"margin-cycle-{args.offset}-witness.npz"
    np.savez_compressed(witness, vector=np.asarray(solution.vector, float), lam_g=np.asarray(solution.lam_g, float), lam_x=np.asarray(solution.lam_x, float))
    report = {"kind": "independent_fresh_ablation_endpoint_margin", "source_result": str(args.result.resolve()),
        "source_result_sha256": sha256(args.result.read_bytes()).hexdigest(), "endpoint": endpoint,
        "restored": restored, "load_factor": load, "margin": load - 1., "audit": asdict(audit),
        "kkt": kkt, "cpu": args.cpu, "solve_wall_seconds": time.perf_counter() - start,
        "witness": str(witness.resolve()), "witness_sha256": sha256(witness.read_bytes()).hexdigest(),
        "interpretation": "Local feasible load witness; no global optimum or physiological failure certificate"}
    _atomic_json(output, report)
    print(json.dumps({"output": str(output), "load": load, "passed": audit.passed}), flush=True)


if __name__ == "__main__":
    main()
