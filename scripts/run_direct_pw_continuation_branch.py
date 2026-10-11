#!/usr/bin/env python3
"""Run one predeclared direct-PW continuation-value intervention.

This is intentionally an offline experimental runner.  It produces one
audited, freshly restored endpoint, but does not fit or activate a terminal
cost.  Its output is consumed by a separate fixed-policy continuation runner.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.continuation_value_pw_action import DirectPulseWidthAction, execute_direct_pw_branch
from cocofest.optimization.task_reserve import TaskReserveCheckpoint
from cocofest.optimization.task_reserve_branch import json_digest, physical_branch_context
from cocofest.simulation.independent_arms_process import _atomic_json, _configured_payload_model
from cocofest.simulation.rho_restart_checkpoint import restore_prepared_checkpoint
from scripts.probe_independent_rho_task_reserve import _load_arm
from scripts.run_local_task_reserve_branch import _advance, _build_runtime, _save_witness, _solve
from cocofest.optimization.task_reserve_probe_adapter import independent_full_nlp_audit


def _object(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def resolve_job(plan_path: Path, job_id: str) -> tuple[dict, dict, dict]:
    """Validate enough immutable plan structure to launch exactly one job."""
    plan_path = Path(plan_path).resolve(strict=True)
    plan = _object(plan_path)
    if (plan.get("schema_version") != 1 or plan.get("kind") != "direct_pw_continuation_value_protocol"
            or plan.get("fatigue_weight_updates_allowed") is not False
            or plan.get("actions_are_direct_pw_constraints") is not True
            or plan.get("activation_allowed") is not False):
        raise ValueError("Unsupported direct-PW continuation plan")
    names = tuple(plan.get("muscle_order", ()))
    anchors = {item.get("id"): item for item in plan.get("anchors", [])}
    actions = {item.get("id"): item for item in plan.get("actions", [])}
    matches = [item for item in plan.get("jobs", []) if item.get("id") == job_id]
    if len(matches) != 1:
        raise ValueError("Exactly one predeclared PW job is required")
    job = matches[0]
    anchor = anchors.get(job.get("anchor_id"))
    action = job.get("action")
    if (not names or anchor is None or not isinstance(action, dict) or actions.get(action.get("id")) != action
            or job.get("intervention_cycles") != 1 or job.get("continuation_policy") != "unit_fatigue_reference"):
        raise ValueError("PW job, anchor, action or continuation policy differs from its plan")
    if action.get("kind") == "direct_pulse_width_trace_bias_v1":
        DirectPulseWidthAction(names, tuple(action.get("direction", ())), action.get("amplitude_fraction"))
    elif action.get("kind") != "no_pw_constraint":
        raise ValueError("Unsupported PW action kind")
    return plan, anchor, job


def _worker(connection, spec, *, verify_endpoint=None):
    try:
        source, payload = _load_arm(Path(spec["source_receipt"]), spec["side"])
        if sha256(Path(spec["source_receipt"]).read_bytes()).hexdigest() != spec["source_receipt_sha256"]:
            raise ValueError("Source receipt changed before direct-PW branch")
        if verify_endpoint is not None:
            runtime = _build_runtime(payload, spec["side"], cpu=spec["cpu"])
            result = restore_prepared_checkpoint(Path(verify_endpoint["primal_path"]), runtime["nmpc"],
                                                 completed_cycles=verify_endpoint["completed_cycles"])
        else:
            action_spec = spec["action"]
            action = None if action_spec["kind"] == "no_pw_constraint" else DirectPulseWidthAction(
                tuple(spec["muscle_order"]), tuple(action_spec["direction"]), action_spec["amplitude_fraction"])
            result = execute_direct_pw_branch(
                source, action=action, muscle_names=spec["muscle_order"], output_directory=Path(spec["output_directory"]),
                build_runtime=lambda: _build_runtime(payload, spec["side"], cpu=spec["cpu"]), solve_one_cycle=_solve,
                advance_one_cycle=_advance,
                audit=lambda solution, program, target, tolerance: independent_full_nlp_audit(
                    solution, program, target_work_j=target, tolerance=tolerance),
                save_witness=_save_witness, tolerance=spec["audit_tolerance"],
            )
        connection.send({"kind": "result", "pid": os.getpid(), "result": result})
    except BaseException as error:
        connection.send({"kind": "error", "error": f"{type(error).__name__}: {error}",
                         "traceback": traceback.format_exc()})
    finally:
        connection.close()


def _spawn(spec, *, timeout_seconds: float, verify_endpoint=None) -> dict:
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(child, spec), kwargs={"verify_endpoint": verify_endpoint})
    try:
        process.start()
        child.close()
        if not parent.poll(timeout_seconds):
            raise TimeoutError("Direct-PW branch worker timed out")
        message = parent.recv()
        if message.get("kind") != "result":
            raise RuntimeError(f"Direct-PW branch worker failed: {message}")
        return message
    finally:
        parent.close()
        child.close()
        if process.pid is not None:
            process.join(timeout=2.)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.)


def run(plan_path: Path, *, job_id: str, output_directory: Path, cpu: int,
        timeout_seconds: float = 900., audit_tolerance: float = 1e-5) -> Path:
    if cpu not in os.sched_getaffinity(0):
        raise ValueError("Requested CPU is outside process affinity")
    if timeout_seconds <= 0. or audit_tolerance <= 0. or audit_tolerance > 1e-5:
        raise ValueError("Invalid worker timeout or audit tolerance")
    plan_path = Path(plan_path).resolve(strict=True)
    plan, anchor, job = resolve_job(plan_path, job_id)
    source_path = Path(anchor["source_receipt"]).resolve(strict=True)
    if sha256(source_path.read_bytes()).hexdigest() != anchor["source_receipt_sha256"]:
        raise ValueError("Anchor receipt digest differs from direct-PW plan")
    source, payload = _load_arm(source_path, "left")
    if (source.completed_cycles != anchor["completed_cycles"]
            or source.prepared_problem_sha256 != anchor["prepared_problem_sha256"]):
        raise ValueError("Anchor checkpoint differs from direct-PW plan")
    configured = _configured_payload_model(payload, "left")
    if configured is None or tuple(configured[1]["muscles"]) != tuple(plan["muscle_order"]):
        raise ValueError("Configured muscle layout differs from direct-PW plan")
    context = physical_branch_context(payload, side="left", nominal_work_j=json.loads(source.task_context_json)["nominal_work_j"],
                                      model_sha256=source.model_sha256)
    output_directory = Path(output_directory).resolve()
    spec = {"side": "left", "source_receipt": str(source_path), "source_receipt_sha256": anchor["source_receipt_sha256"],
            "source_prepared_problem_sha256": source.prepared_problem_sha256, "plan_path": str(plan_path),
            "plan_sha256": sha256(plan_path.read_bytes()).hexdigest(), "job": job, "anchor": anchor,
            "action": job["action"], "muscle_order": plan["muscle_order"], "cpu": cpu,
            "output_directory": str(output_directory), "audit_tolerance": audit_tolerance,
            "physical_task_context": context, "physical_task_context_sha256": json_digest(context)}
    produced = _spawn(spec, timeout_seconds=timeout_seconds)
    export = produced["result"]
    output_directory.mkdir(parents=True, exist_ok=False) if not output_directory.exists() else None
    _atomic_json(output_directory / "prepared-direct-pw-export.json", {**export, **spec})
    endpoint = export.get("endpoint")
    policy = {"id": "unit_fatigue_reference",
              "absolute_fatigue_weights_from_unit": dict.fromkeys(plan["muscle_order"], 1.)}
    receipt = {**spec, "schema_version": 1, "kind": "certified_direct_pw_continuation_value_endpoint",
               "producer_pid": produced["pid"], "success": bool(export.get("success")),
               "cycle": export.get("cycle"), "endpoint": endpoint,
               "policy": policy,
               "fresh_worker_replay_verified": False, "activation_allowed": False}
    if endpoint is not None and export.get("success"):
        verified = _spawn(spec, timeout_seconds=timeout_seconds, verify_endpoint=endpoint)
        restored = verified["result"]
        if (verified["pid"] == produced["pid"] or restored.get("restored_problem_sha256") != endpoint["prepared_problem_sha256"]
                or restored.get("stimulation_history_complete") is not True):
            raise RuntimeError("Direct-PW endpoint failed fresh-worker restoration")
        receipt.update(restored=restored, verifier_pid=verified["pid"], fresh_worker_replay_verified=True)
    destination = output_directory / "endpoint-receipt.json"
    _atomic_json(destination, receipt)
    return destination


def load_direct_pw_endpoint(path: Path) -> tuple[TaskReserveCheckpoint, dict]:
    """Read a freshly restored direct-PW endpoint for nominal continuation."""
    path = Path(path).resolve(strict=True)
    receipt = _object(path)
    if (receipt.get("schema_version") != 1 or receipt.get("kind") != "certified_direct_pw_continuation_value_endpoint"
            or receipt.get("success") is not True or receipt.get("fresh_worker_replay_verified") is not True
            or receipt.get("activation_allowed") is not False):
        raise ValueError("A successful, freshly restored direct-PW endpoint is required")
    plan_path = Path(receipt.get("plan_path", "")).resolve(strict=True)
    if sha256(plan_path.read_bytes()).hexdigest() != receipt.get("plan_sha256"):
        raise ValueError("Direct-PW plan changed after endpoint production")
    plan, anchor, job = resolve_job(plan_path, receipt.get("job", {}).get("id", ""))
    if (job != receipt.get("job") or anchor != receipt.get("anchor") or job.get("action") != receipt.get("action")
            or plan.get("muscle_order") != receipt.get("muscle_order")
            or receipt.get("cycle", {}).get("certified") is not True
            or receipt["cycle"].get("action", {}).get("fatigue_weight_update") is not False
            or receipt["cycle"].get("nominal_problem_restored_before_advance") is not True):
        raise ValueError("Direct-PW endpoint differs from its predeclared action")
    source_path = Path(receipt.get("source_receipt", "")).resolve(strict=True)
    if sha256(source_path.read_bytes()).hexdigest() != receipt.get("source_receipt_sha256"):
        raise ValueError("Direct-PW source receipt changed")
    source, payload = _load_arm(source_path, "left")
    endpoint = receipt.get("endpoint", {})
    archive = Path(endpoint.get("primal_path", "")).resolve(strict=True)
    if (sha256(archive.read_bytes()).hexdigest() != endpoint.get("sha256")
            or endpoint.get("completed_cycles") != source.completed_cycles + 1
            or receipt.get("restored", {}).get("restored_problem_sha256") != endpoint.get("prepared_problem_sha256")
            or receipt.get("restored", {}).get("stimulation_history_complete") is not True
            or receipt.get("producer_pid") == receipt.get("verifier_pid")):
        raise ValueError("Direct-PW endpoint archive or fresh restoration is invalid")
    work = json.loads(source.task_context_json)["nominal_work_j"]
    context = physical_branch_context(payload, side="left", nominal_work_j=work, model_sha256=source.model_sha256)
    if context != receipt.get("physical_task_context") or json_digest(context) != receipt.get("physical_task_context_sha256"):
        raise ValueError("Direct-PW physical task context differs from source")
    checkpoint = TaskReserveCheckpoint.from_archive(archive, completed_cycles=endpoint["completed_cycles"],
        model_path=Path(source.model_path), task_context=context)
    if checkpoint.prepared_problem_sha256 != receipt["restored"]["restored_problem_sha256"]:
        raise ValueError("Direct-PW endpoint archive differs from fresh restored problem")
    return checkpoint, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.)
    parser.add_argument("--audit-tolerance", type=float, default=1e-5)
    args = parser.parse_args(argv)
    print(run(args.plan, job_id=args.job, output_directory=args.output_directory, cpu=args.cpu,
              timeout_seconds=args.timeout_seconds, audit_tolerance=args.audit_tolerance))


if __name__ == "__main__":
    main()
