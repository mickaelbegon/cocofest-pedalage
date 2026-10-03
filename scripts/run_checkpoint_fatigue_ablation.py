#!/usr/bin/env python3
"""Audited fixed-task fatigue-cost branches from immutable bilateral receipts.

One process runs one arm/policy. Observations at 1, 5 and 20 cycles are nested
prefixes of the same deterministic branch, not independent replicates.
"""
from __future__ import annotations

import argparse
from copy import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.task_reserve_probe_adapter import independent_full_nlp_audit
from cocofest.simulation.independent_arms_process import (
    _atomic_json, _configured_payload_model, _driver_arguments,
)
from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint, restore_prepared_checkpoint
from scripts.probe_independent_rho_task_reserve import _load_arm
from scripts.run_local_task_reserve_branch import _advance, _save_witness, _solve

VARIANTS = ("integral_quadratic", "terminal_quadratic", "integral_linear", "terminal_linear")


def build_runtime(payload, side, variant):
    from cocofest.optimization.configured_cycling_model import configured_model_factories
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver
    args = copy(_driver_arguments(payload, side))
    args.nlp_ipopt_recovery = False
    args.nlp_ipopt_recovery_ma57_tuned = False
    args.fatigue_objective_variant = variant
    if not args.parametric_fatigue_weights or args.rho_pulse_width_transfer_mode != "repeat":
        raise ValueError("Requires existing parametric weights and exact repeat-PW checkpoint")
    if payload.get("muscle_pace", {}).get("adaptation_enabled", False):
        raise ValueError("Fixed weights required")
    if payload.get("resistance_pace", {}).get("capacity_feedback", False):
        raise ValueError("Fixed mechanical split required")
    configured = _configured_payload_model(payload, side)
    with configured_model_factories(configured[1], []):
        runtime = driver.build_unilateral_runtime(args, echo=False)
    runtime["nmpc"]._initialize_state_idx_to_cycle({"states": {}})
    return runtime, args


def run(receipt, side, variant, output, cpu, *, horizon=20, tolerance=1e-6):
    from bioptim import SolutionMerge
    if variant not in VARIANTS or horizon not in (1, 5, 20):
        raise ValueError("Use a predeclared fatigue variant and horizon 1/5/20")
    if cpu not in os.sched_getaffinity(0):
        raise ValueError("Requested dedicated CPU unavailable")
    source, payload = _load_arm(receipt, side)
    if output.exists():
        raise FileExistsError(output)
    os.sched_setaffinity(0, {cpu})
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema_version": 1, "kind": "checkpoint_fatigue_objective_ablation",
        "source_receipt": str(receipt.resolve()),
        "source_receipt_sha256": sha256(receipt.read_bytes()).hexdigest(),
        "source_completed_cycles": source.completed_cycles,
        "source_archive_sha256": sha256(Path(source.archive_path).read_bytes()).hexdigest(),
        "source_prepared_problem_sha256": source.prepared_problem_sha256,
        "model_path": source.model_path, "model_sha256": source.model_sha256,
        "source_configuration": payload, "side": side, "variant": variant,
        "cpu": cpu, "numeric_threads": 1, "endpoints": [n for n in (1, 5, 20) if n <= horizon],
        "audit_tolerance": tolerance,
        "objective_scale": "10000 times source fatigue coefficient; terminal multiplied by window duration; same unit muscle weights; no slope matching",
        "predeclared_promotion": "Require all 20 cycles audited, then independently reoptimized endpoint work margin exceeds integral_quadratic by >0.01 at both c120 and c140 with no arm regression. No promotion from fatigue score alone.",
        "failure_interpretation": "Unsuccessful branch solve is indeterminate, not physiological failure",
        "solver_recovery": "disabled for these matched short branches",
    }
    _atomic_json(output / "protocol.json", provenance)
    runtime, args = build_runtime(payload, side, variant)
    program, solver = runtime["nmpc"], runtime["solver"]
    restored = restore_prepared_checkpoint(Path(source.archive_path), program, completed_cycles=source.completed_cycles)
    if restored["restored_problem_sha256"] != source.prepared_problem_sha256 or not restored["stimulation_history_complete"]:
        raise RuntimeError("Exact numerical checkpoint restoration failed")
    nlp = program.nlp[0]
    work = float(nlp.x_bounds["E_prod"].min[0, 2])
    if work != float(nlp.x_bounds["E_prod"].max[0, 2]):
        raise RuntimeError("Work equality required")
    weights = np.asarray(program.parameter_init["rho_fatigue_weights"].init).ravel()
    if not np.array_equal(weights, np.ones_like(weights)):
        raise RuntimeError("Unit-weight checkpoint required")
    provenance.update(source_restore=restored, effective_driver_arguments=vars(args), nominal_work_j=work)
    # Namespace contains Paths; the serialized argument record must be portable JSON.
    provenance = json.loads(json.dumps(provenance, default=str))
    _atomic_json(output / "protocol.json", provenance)
    cycles, endpoints = [], []
    compiled = None
    for offset in range(1, horizon + 1):
        program.total_optimization_run = offset - 1
        start = time.perf_counter()
        solution = _solve(program, solver)
        elapsed = time.perf_counter() - start
        current = getattr(getattr(program, "ocp_solver", None), "shaked_ocp_solver", None)
        if current is None or (compiled is not None and current is not compiled) or program.nlp[0] is not nlp:
            raise RuntimeError("Compiled NLP was not reused")
        compiled = current
        audit = independent_full_nlp_audit(solution, program, target_work_j=work, tolerance=tolerance)
        states = solution.decision_states(to_merge=SolutionMerge.NODES)
        models = program.nlp[0].model.muscles_dynamics_model
        row = {"offset": offset, "completed_cycles": source.completed_cycles + offset,
               "certified": bool(audit.passed), "status": str(solution.status),
               "solve_wall_seconds": elapsed, "maximum_normalized_violation": audit.maximum_normalized_violation,
               "audit": dict(audit.detail),
               "terminal_normalized_A": {m.muscle_name: float(states[f"A_{m.muscle_name}"][0, -1] / m.a_scale) for m in models},
               "terminal_slow_states": {key: float(value[0, -1]) for key, value in states.items() if key.startswith(("A_", "Tau1_", "Km_"))}}
        cycles.append(row)
        if audit.passed:
            row["witness"] = _save_witness(solution, program, output, offset, audit)
            row["witness_sha256"] = sha256(Path(row["witness"]).read_bytes()).hexdigest()
            _advance(program, solution, work)
            if offset in provenance["endpoints"]:
                endpoints.append({"offset": offset, **export_prepared_checkpoint(output / f"cycle-{offset}.npz", program,
                    completed_cycles=source.completed_cycles + offset, model_path=Path(source.model_path))})
        _atomic_json(output / "result.json", {**provenance, "cycles": cycles, "prepared_endpoints": endpoints,
            "completed": offset == horizon or not audit.passed, "success": offset == horizon and all(r["certified"] for r in cycles),
            "promotion_allowed": False, "promotion_reason": "Independent endpoint margin comparison still required"})
        print(f"{side} {variant} c{source.completed_cycles}+{offset}: certified={audit.passed} elapsed={elapsed:.3f}", flush=True)
        if not audit.passed:
            break
        source.verify_files()
    return output / "result.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--horizon", type=int, choices=(1, 5, 20), default=20)
    args = parser.parse_args()
    print(run(args.receipt, args.side, args.variant, args.output, args.cpu, horizon=args.horizon))


if __name__ == "__main__":
    main()
