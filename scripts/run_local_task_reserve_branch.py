#!/usr/bin/env python3
"""Run a predeclared short nominal RHO branch from an exact bilateral anchor.

Each policy runs in its own spawned worker. Endpoint archives are restored in
separate fresh workers before publishing unilateral receipts. This command
never fits or enables a reserve cost, and never labels an NLP failure as a
physiological failure.
"""

from __future__ import annotations

import argparse
from copy import copy
from hashlib import sha256
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.task_reserve import TaskReserveCheckpoint
from cocofest.optimization.task_reserve_branch import (
    certified_endpoint_receipt, execute_short_branch, json_digest, physical_branch_context, validate_policy,
)
from cocofest.optimization.task_reserve_probe_adapter import independent_full_nlp_audit
from cocofest.simulation.independent_arms_process import (
    _apply_worker_solver_affinity, _atomic_json, _configured_payload_model, _driver_arguments,
)
from cocofest.simulation.rho_restart_checkpoint import restore_prepared_checkpoint
from scripts.probe_independent_rho_task_reserve import _load_arm
from scripts.validate_independent_rho_checkpoint import _object, _source_path


def _build_runtime(payload, side):
    from cocofest.optimization.configured_cycling_model import configured_model_factories
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver

    args = copy(_driver_arguments(payload, side))
    args.nlp_ipopt_recovery = False
    args.nlp_ipopt_recovery_ma57_tuned = False
    if not args.parametric_fatigue_weights:
        raise ValueError("Anchor configuration must already enable parametric fatigue weights")
    if args.rho_pulse_width_transfer_mode != "repeat":
        raise ValueError("Non-repeat PW predictors need additional historical memory not present in this archive schema")
    if payload.get("muscle_pace", {}).get("adaptation_enabled", False):
        raise ValueError("Branch anchor must have fixed weights")
    configured = _configured_payload_model(payload, side)
    if configured is None:
        raise ValueError("Branch requires an explicitly configured muscle model")
    with configured_model_factories(configured[1], []):
        runtime = driver.build_unilateral_runtime(args, echo=False)
    # Native RHO.solve normally initializes this before its loop. Here the
    # prepared bounds are restored afterwards; never run _set_cyclic_bound on
    # the restored problem, which could replace source terminal constraints.
    runtime["nmpc"]._initialize_state_idx_to_cycle({"states": {}})
    _apply_worker_solver_affinity(payload, side)
    return runtime


def _solve(program, solver):
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver
    return super(driver.RecedingHorizonOptimization, program).solve(solver=solver, warm_start=None)


def _advance(program, solution, work):
    from cocofest.optimization.independent_arm_backends import set_terminal_eprod_target
    # Exactly the same native advance as the independent-arm worker. Auditing
    # has already occurred, so driver recovery wrappers must not advance again.
    type(program).advance_window(program, solution, n_cycles_simultaneous=program.n_cycles_simultaneous)
    set_terminal_eprod_target(program, work)
    program.all_models.clear()


def _save_witness(solution, program, directory, offset, audit):
    from bioptim import SolutionMerge
    path = directory / f"witness-cycle-{offset}.npz"
    states = solution.decision_states(to_merge=SolutionMerge.NODES)
    controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
    np.savez_compressed(path, vector=np.asarray(solution.vector, float),
        **{f"states__{key}": np.asarray(value, float) for key, value in states.items()},
        **{f"controls__{key}": np.asarray(value, float) for key, value in controls.items()},
        metadata__json=np.asarray(json.dumps({"independent_audit": dict(audit.detail)}, sort_keys=True)))
    return str(path)


def _worker(connection, spec, verify_endpoint=None):
    try:
        source, payload = _load_arm(Path(spec["source_receipt"]), spec["side"])
        if sha256(Path(spec["source_receipt"]).read_bytes()).hexdigest() != spec["source_receipt_sha256"]:
            raise ValueError("Source receipt changed before worker execution")
        if verify_endpoint is None:
            result = execute_short_branch(source, policy=spec["policy"], muscle_names=spec["muscle_order"],
                output_directory=Path(spec["output_directory"]),
                build_runtime=lambda: _build_runtime(payload, spec["side"]), solve_one_cycle=_solve,
                advance_one_cycle=_advance,
                audit=lambda solution, program, target, tol: independent_full_nlp_audit(
                    solution, program, target_work_j=target, tolerance=tol),
                save_witness=_save_witness, tolerance=spec["tolerance"])
        else:
            runtime = _build_runtime(payload, spec["side"])
            result = restore_prepared_checkpoint(Path(verify_endpoint["primal_path"]), runtime["nmpc"],
                                                 completed_cycles=verify_endpoint["completed_cycles"])
        connection.send({"kind": "result", "pid": os.getpid(), "result": result})
    except BaseException as error:
        connection.send({"kind": "error", "error": f"{type(error).__name__}: {error}",
                         "traceback": traceback.format_exc()})
    finally:
        connection.close()


def _spawn(spec, *, timeout_seconds, verify_endpoint=None):
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(child, spec, verify_endpoint))
    try:
        process.start()
        child.close()
        if not parent.poll(timeout_seconds):
            raise TimeoutError("Short reserve branch worker timed out")
        result = parent.recv()
        if result.get("kind") != "result":
            raise RuntimeError(f"Branch worker failed: {result}")
        return result
    finally:
        parent.close()
        child.close()
        if process.pid is not None:
            process.join(timeout=2.)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.)


def load_branch_endpoint(path: Path) -> tuple[TaskReserveCheckpoint, dict]:
    """Strict separate loader; existing bilateral _load_arm stays unchanged."""
    path = Path(path).resolve(strict=True)
    receipt = _object(path)
    if (receipt.get("schema_version") != 1
            or receipt.get("kind") != "certified_unilateral_task_reserve_branch_endpoint"
            or receipt.get("fresh_worker_replay_verified") is not True
            or receipt.get("exact_bilateral_restart") is not False):
        raise ValueError("A certified unilateral branch endpoint receipt is required")
    source_receipt = _source_path(path, receipt.get("source_receipt"))
    if sha256(source_receipt.read_bytes()).hexdigest() != receipt.get("source_receipt_sha256"):
        raise ValueError("Branch source receipt changed")
    source, payload = _load_arm(source_receipt, receipt["side"])
    plan_path = _source_path(path, receipt.get("plan_path"))
    if sha256(plan_path.read_bytes()).hexdigest() != receipt.get("plan_sha256"):
        raise ValueError("Branch experiment plan changed")
    plan = _object(plan_path)
    planned = [row for row in plan["reachable_branch_design"]["branches"]
               if row["id"] == receipt["policy"]["id"]]
    anchors = [row for row in plan["local_neighborhoods"] if row["anchor_id"] == receipt["anchor_id"]]
    if (planned != [receipt["policy"]] or len(anchors) != 1 or plan["side"] != receipt["side"]
            or _source_path(plan_path, anchors[0]["anchor_receipt"]) != source_receipt
            or receipt["muscle_order"] != plan["reachable_branch_design"]["muscle_order"]):
        raise ValueError("Branch provenance differs from its predeclared plan")
    policy, weights = validate_policy(receipt["policy"], receipt["muscle_order"])
    endpoint = receipt["endpoint"]
    if (receipt["branch_cycle"] not in policy["endpoint_cycles"]
            or receipt["completed_cycles"] != source.completed_cycles + receipt["branch_cycle"]
            or endpoint["completed_cycles"] != receipt["completed_cycles"]
            or receipt["source_restore"].get("restored_problem_sha256") != source.prepared_problem_sha256):
        raise ValueError("Branch endpoint cycle or source provenance mismatch")
    # Re-run the publication checks as part of reading an artifact.
    certified_endpoint_receipt(receipt, endpoint, receipt["restored"], provenance={},
        producer_pid=receipt["producer_pid"], verifier_pid=receipt["verifier_pid"])
    for cycle in receipt["cycles"]:
        witness = _source_path(path, cycle["witness"])
        if sha256(witness.read_bytes()).hexdigest() != cycle.get("witness_sha256"):
            raise ValueError("Branch witness bytes changed")
    source_work = json.loads(source.task_context_json)["nominal_work_j"]
    context = physical_branch_context(payload, side=receipt["side"], nominal_work_j=source_work,
                                      model_sha256=source.model_sha256)
    if context != receipt.get("physical_task_context") or json_digest(context) != receipt.get("physical_task_context_sha256"):
        raise ValueError("Branch physical context differs from source configuration")
    archive = _source_path(path, endpoint["primal_path"])
    if sha256(archive.read_bytes()).hexdigest() != endpoint["sha256"]:
        raise ValueError("Branch endpoint archive bytes changed")
    with np.load(archive, allow_pickle=False) as prepared:
        for suffix in ("parameter_init:rho_fatigue_weights", "parameter_bounds:rho_fatigue_weights:min",
                       "parameter_bounds:rho_fatigue_weights:max"):
            if not np.allclose(np.asarray(prepared[f"problem__{suffix}"]).reshape(-1), weights,
                               rtol=0, atol=1e-12):
                raise ValueError("Branch archive policy differs from predeclared weights")
        for suffix in ("min", "max"):
            if float(prepared[f"problem__x_bounds:E_prod:{suffix}"][0, 2]) != source_work:
                raise ValueError("Branch changed nominal task work")
    result = TaskReserveCheckpoint.from_archive(archive, completed_cycles=receipt["completed_cycles"],
        model_path=Path(source.model_path), task_context=context)
    if result.prepared_problem_sha256 != receipt["restored"]["restored_problem_sha256"]:
        raise ValueError("Branch archive differs from fresh restore")
    return result, payload


def run(plan_path: Path, *, anchor_id: str, branch_id: str, output_directory: Path,
        tolerance: float = 1e-5, timeout_seconds: float = 900.) -> Path:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Worker timeout must be finite and positive")
    plan_path = Path(plan_path).resolve(strict=True)
    plan = _object(plan_path)
    if plan.get("schema_version") != 1 or plan.get("kind") != "local_task_reserve_experiment_plan":
        raise ValueError("Unsupported local task-reserve experiment plan")
    anchors = [row for row in plan["local_neighborhoods"] if row["anchor_id"] == anchor_id]
    branches = [row for row in plan["reachable_branch_design"]["branches"] if row["id"] == branch_id]
    if len(anchors) != 1 or len(branches) != 1:
        raise ValueError("Exactly one predeclared anchor and branch are required")
    anchor = _source_path(plan_path, anchors[0]["anchor_receipt"])
    source, payload = _load_arm(anchor, plan["side"])
    names = plan["reachable_branch_design"]["muscle_order"]
    policy, _ = validate_policy(branches[0], names)
    configured = _configured_payload_model(payload, plan["side"])
    if set(configured[1]["muscles"]) != set(names):
        raise ValueError("Plan muscles differ from the configured model")
    work = json.loads(source.task_context_json)["nominal_work_j"]
    physical = physical_branch_context(payload, side=plan["side"], nominal_work_j=work,
                                       model_sha256=source.model_sha256)
    provenance = {"plan_path": str(plan_path), "plan_sha256": sha256(plan_path.read_bytes()).hexdigest(),
        "anchor_id": anchor_id, "side": plan["side"], "source_receipt": str(anchor),
        "source_receipt_sha256": sha256(anchor.read_bytes()).hexdigest(),
        "physical_task_context": physical, "physical_task_context_sha256": json_digest(physical)}
    spec = {**provenance, "policy": policy, "muscle_order": names,
            "output_directory": str(Path(output_directory).resolve()), "tolerance": tolerance}
    produced = _spawn(spec, timeout_seconds=timeout_seconds)
    export = produced["result"]
    directory = Path(spec["output_directory"])
    _atomic_json(directory / "prepared-branch-export.json", {**export, **provenance})
    receipts = []
    for endpoint in export["endpoints"]:
        verified = _spawn(spec, timeout_seconds=timeout_seconds, verify_endpoint=endpoint)
        receipt = certified_endpoint_receipt(export, endpoint, verified["result"], provenance=provenance,
            producer_pid=produced["pid"], verifier_pid=verified["pid"])
        destination = directory / f"cycle-{endpoint['branch_cycle']}-receipt.json"
        _atomic_json(destination, receipt)
        load_branch_endpoint(destination)
        receipts.append(str(destination))
    destination = directory / "branch-result.json"
    _atomic_json(destination, {**provenance, "success": export["success"], "endpoint_receipts": receipts,
                              "cycles": export["cycles"], "activation_allowed": False})
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--timeout-seconds", type=float, default=900.)
    args = parser.parse_args(argv)
    print(run(args.plan, anchor_id=args.anchor, branch_id=args.branch,
              output_directory=args.output_directory, tolerance=args.tolerance,
              timeout_seconds=args.timeout_seconds))


if __name__ == "__main__":
    main()
