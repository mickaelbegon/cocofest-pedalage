#!/usr/bin/env python3
"""Continue one certified fixed-weight unilateral branch without changing its RHO graph.

An unsuccessful NLP is recorded as numerical/indeterminate unless the native
frozen zero-objective and fatigue-rest counterfactual probes supply the narrow
local failure evidence specified in the output receipt.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.independent_arm_backends import set_terminal_eprod_target
from cocofest.optimization.task_reserve_probe_adapter import independent_full_nlp_audit
from cocofest.simulation.independent_arms_process import _FrozenRhoRetry, _atomic_json
from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint, restore_prepared_checkpoint
from scripts.run_local_task_reserve_branch import _build_runtime, _solve, load_branch_endpoint
from scripts.run_direct_pw_continuation_branch import load_direct_pw_endpoint


def classify_terminal(last: dict | None, *, censored: bool) -> tuple[str, str]:
    if censored:
        return "right_censored", "absolute_cycle_budget"
    last = last or {}
    zero = last.get("zero_objective_feasibility_probe") or {}
    rest = (last.get("counterfactual_probes") or {}).get("fatigue_rest") or {}
    native = (zero.get("solver_stats") or {}).get("return_status")
    if (zero.get("available") and zero.get("constraints_unchanged")
            and zero.get("feasible_witness") is False and native == "Infeasible_Problem_Detected"
            and rest.get("feasible_witness") is True and rest.get("physical_rho_advanced") is False):
        return "local_frozen_physiological_failure", "frozen_zero_objective_infeasible_with_fatigue_rest_witness"
    return "numerical_or_physical_indeterminate", "no_local_frozen_failure_certificate"


def run(endpoint_receipt: Path, *, output_directory: Path, cpu: int, maximum_absolute_cycles: int,
        audit_tolerance: float = 1e-5) -> Path:
    if type(maximum_absolute_cycles) is not int or maximum_absolute_cycles < 1:
        raise ValueError("maximum_absolute_cycles must be a positive integer")
    if not math.isfinite(audit_tolerance) or not 0 < audit_tolerance <= 1e-5:
        raise ValueError("audit_tolerance must lie in (0, 1e-5]")
    if cpu not in os.sched_getaffinity(0):
        raise ValueError("Requested CPU is outside process affinity")
    endpoint_receipt = Path(endpoint_receipt).resolve(strict=True)
    source_receipt = json.loads(endpoint_receipt.read_text())
    if source_receipt.get("kind") == "certified_direct_pw_continuation_value_endpoint":
        checkpoint, payload = load_direct_pw_endpoint(endpoint_receipt)
    else:
        checkpoint, payload = load_branch_endpoint(endpoint_receipt)
    if checkpoint.completed_cycles >= maximum_absolute_cycles:
        raise ValueError("Maximum cycle budget must exceed the certified branch endpoint")
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {cpu})
    runtime = _build_runtime(payload, "left", cpu=cpu)
    program, solver, args = runtime["nmpc"], runtime["solver"], runtime["args"]
    restored = restore_prepared_checkpoint(Path(checkpoint.archive_path), program,
                                           completed_cycles=checkpoint.completed_cycles)
    if (restored.get("restored_problem_sha256") != checkpoint.prepared_problem_sha256
            or restored.get("stimulation_history_complete") is not True):
        raise RuntimeError("Certified branch endpoint did not restore exactly")
    weights = np.asarray(program.parameter_init["rho_fatigue_weights"].init, float).reshape(-1)
    policy = source_receipt["policy"]
    muscle_order = tuple(policy.get("muscle_order", ()))
    if not muscle_order:
        # Branch endpoint receipts retain the order separately from the policy
        # mapping, whose JSON serialization is deliberately sorted for hashes.
        muscle_order = tuple(json.loads(endpoint_receipt.read_text())["muscle_order"])
    expected = np.asarray([policy["absolute_fatigue_weights_from_unit"][name] for name in muscle_order], float)
    if weights.shape != expected.shape or not np.allclose(weights, expected, rtol=0, atol=1e-12):
        raise RuntimeError("Endpoint fatigue weights differ from its certified branch policy")
    work = float(program.nlp[0].x_bounds["E_prod"].min[0, 2])
    if not math.isfinite(work) or work <= 0 or program.nlp[0].x_bounds["E_prod"].max[0, 2] != work:
        raise RuntimeError("Continuation task work is not a positive equality")
    receipt = {"schema_version": 1, "kind": "fixed_policy_branch_continuation",
               "source_endpoint_receipt": str(endpoint_receipt),
               "source_endpoint_receipt_sha256": sha256(endpoint_receipt.read_bytes()).hexdigest(),
               "source_completed_cycles": checkpoint.completed_cycles, "source_restore": restored,
               "policy": policy, "weights": weights.tolist(), "cpu": cpu,
               "maximum_absolute_cycles": maximum_absolute_cycles, "audit_tolerance": audit_tolerance,
               "nominal_work_j": work, "activation_allowed": False,
               "failure_interpretation": "only local frozen-failure evidence can attribute a terminal stop to fatigue"}
    _atomic_json(output_directory / "protocol.json", receipt)
    from bioptim import SolutionMerge
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver
    retry = _FrozenRhoRetry(program, driver, tolerance=driver._window_feasibility_tolerance(args),
                            max_attempts=args.max_consecutive_failing, recovery_args=args)
    retry.completed = checkpoint.completed_cycles
    cycles, attempts, terminal = [], [], None
    while retry.completed < maximum_absolute_cycles:
        retry.capture()
        while True:
            program.total_optimization_run = retry.completed - checkpoint.completed_cycles
            started = time.perf_counter()
            solution = _solve(program, solver)
            elapsed = time.perf_counter() - started
            audit = independent_full_nlp_audit(solution, program, target_work_j=work, tolerance=audit_tolerance)
            feasibility = driver._solution_feasibility_summary(solution, driver._window_feasibility_tolerance(args))
            native = driver._rho_solution_is_certified(solution.status, feasibility)
            row = {"cycle": retry.completed + 1, "certified": bool(native and audit.passed),
                   "solver_status": int(solution.status), "native_solver_status": driver.snapshot_nlp_solver_stats(program).get("return_status"),
                   "independent_audit": dict(audit.detail), "maximum_normalized_violation": audit.maximum_normalized_violation,
                   "solve_wall_seconds": elapsed, "feasibility": feasibility, "weights": weights.tolist()}
            if native and not audit.passed:
                terminal = {**row, "failure_reason": "independent_audit_disagrees_with_native_certification"}
                break
            if row["certified"]:
                states = solution.decision_states(to_merge=SolutionMerge.NODES)
                row["terminal_normalized_A"] = {
                    model.muscle_name: float(states[f"A_{model.muscle_name}"][0, -1] / model.a_scale)
                    for model in program.nlp[0].model.muscles_dynamics_model}
                retry._advance(program, solution, n_cycles_simultaneous=program.n_cycles_simultaneous)
                if not solution._cocofest_advanced_physical_rho:
                    raise RuntimeError("Certified continuation solution was not physically advanced")
                set_terminal_eprod_target(program, work)
                program.all_models.clear()
                cycles.append(row)
                _atomic_json(output_directory / "progress.json", {**receipt, "cycles": cycles, "attempts": attempts,
                                                                      "last_certified_cycle": retry.completed, "completed": False})
                break
            for field in ("frozen_retry", "zero_objective_feasibility_probe", "counterfactual_probes"):
                value = getattr(solution, f"_cocofest_{field}", None)
                if value is not None:
                    row[field] = value
            attempts.append(row)
            if not program._cocofest_retry_same_rho_pending:
                terminal = row
                break
        if terminal:
            break
        checkpoint.verify_files()
    censored = retry.completed == maximum_absolute_cycles
    status, reason = classify_terminal(terminal, censored=censored)
    endpoint = export_prepared_checkpoint(output_directory / "last-certified-endpoint.npz", program,
                                          completed_cycles=retry.completed, model_path=Path(checkpoint.model_path))
    result = {**receipt, "cycles": cycles, "attempts": attempts, "terminal_attempt": terminal,
              "last_certified_cycle": retry.completed, "certified_cycles": len(cycles), "censored": censored,
              "terminal_status": status, "terminal_reason": reason, "last_certified_endpoint": endpoint,
              "fresh_endpoint_replay_required": True, "completed": True}
    destination = output_directory / "continuation-result.json"
    _atomic_json(destination, result)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-receipt", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--maximum-absolute-cycles", type=int, default=260)
    parser.add_argument("--audit-tolerance", type=float, default=1e-5)
    args = parser.parse_args(argv)
    print(run(args.endpoint_receipt, output_directory=args.output_directory, cpu=args.cpu,
              maximum_absolute_cycles=args.maximum_absolute_cycles, audit_tolerance=args.audit_tolerance))


if __name__ == "__main__":
    main()
