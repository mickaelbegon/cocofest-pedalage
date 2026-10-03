"""Reachable short-RHO branches for local reserve experiments.

A branch is a real nominal-task trajectory, not an edited fatigue state. Its
new schema preserves the bilateral source receipt while describing only the
executed arm. No artifact from this module claims a bilateral continuation.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from cocofest.optimization.parametric_fatigue_weights import FATIGUE_WEIGHT_PARAMETER_KEY
from cocofest.optimization.task_reserve import DEFAULT_CONSTRAINT_GROUPS, TaskReserveCheckpoint
from cocofest.optimization.task_reserve_probe_adapter import IndependentAudit
from cocofest.optimization.receding_horizon_initial_guess import snapshot_initial_guess, initial_guess_signature
from cocofest.simulation.rho_restart_checkpoint import (
    export_prepared_checkpoint, prepared_problem_arrays, restore_prepared_checkpoint,
)


def json_digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def physical_branch_context(payload: Mapping, *, side: str, nominal_work_j: float,
                            model_sha256: str) -> dict:
    """Retain every source configuration field except the two policy vectors.

    Deliberately conservative: even output scheduling is retained. In particular
    no solver, tolerance, recovery, load, cadence, model or integrator setting is
    dropped. Branch policies are separate and do not rewrite this configuration.
    """
    if side not in ("left", "right") or not math.isfinite(nominal_work_j) or nominal_work_j <= 0:
        raise ValueError("Invalid physical arm/work context")
    configuration = deepcopy(dict(payload))
    pace = configuration.get("muscle_pace", {})
    if not isinstance(pace, dict):
        raise ValueError("muscle_pace must be an object")
    for arm in ("left", "right"):
        pace.pop(f"{arm}_initial_weights", None)
    return {"schema_version": 1, "kind": "nominal_unilateral_task_for_reachable_reserve_branch",
            "source_configuration_except_policy_weights": configuration,
            "side": side, "nominal_work_j": nominal_work_j, "model_sha256": model_sha256,
            "excluded_configuration_fields": ["muscle_pace.left_initial_weights",
                                                "muscle_pace.right_initial_weights"]}


def validate_policy(policy: Mapping, muscle_names) -> tuple[dict, tuple[float, ...]]:
    policy = deepcopy(dict(policy))
    names = tuple(muscle_names)
    if (not names or len(set(names)) != len(names)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(policy.get("id", "")))):
        raise ValueError("Branch requires explicit unique muscles and a safe id")
    endpoints = policy.get("endpoint_cycles")
    if (not isinstance(endpoints, list) or not endpoints
            or any(type(n) is not int or n < 1 or n > 4 for n in endpoints)
            or endpoints != sorted(set(endpoints))):
        raise ValueError("Short branch endpoints must be distinct increasing integers in [1,4]")
    expected = [1, 3] if policy.get("partition") == "train" else [2, 4]
    if policy.get("partition") not in ("train", "holdout") or endpoints != expected:
        raise ValueError("Predeclared train/holdout durations differ from the reachable experiment")
    if policy.get("nominal_work_scale") != 1. or policy.get("starting_checkpoint") != "exact_anchor_receipt":
        raise ValueError("Branch must start at the exact anchor and retain nominal work")
    weights = policy.get("absolute_fatigue_weights_from_unit")
    if not isinstance(weights, dict) or set(weights) != set(names):
        raise ValueError("Policy weights must identify every model muscle exactly")
    if any(isinstance(weights[name], bool) for name in names):
        raise ValueError("Policy weights must be finite positive numbers")
    values = tuple(float(weights[name]) for name in names)
    if (any(not math.isfinite(x) or x <= 0 for x in values)
            or not math.isclose(sum(values) / len(values), 1., abs_tol=1e-12, rel_tol=0.)):
        raise ValueError("Policy weights must be positive and have arithmetic mean one")
    return policy, values


def _unchanged_numerical_problem(before, after):
    allowed = {f"parameter_bounds:{FATIGUE_WEIGHT_PARAMETER_KEY}:min",
               f"parameter_bounds:{FATIGUE_WEIGHT_PARAMETER_KEY}:max",
               f"parameter_init:{FATIGUE_WEIGHT_PARAMETER_KEY}"}
    if before.keys() != after.keys() or any(not np.array_equal(before[key], after[key])
                                           for key in before if key not in allowed):
        raise RuntimeError("Policy update changed physical bounds or non-policy parameters")


def execute_short_branch(source: TaskReserveCheckpoint, *, policy: Mapping, muscle_names,
                         output_directory: Path, build_runtime, solve_one_cycle, advance_one_cycle,
                         audit, save_witness, tolerance: float) -> dict:
    """One build, exact restore, parameter update, then audited native advances.

    Callers run this in a spawned process. A failed solve never advances the
    physical RHO and cannot become a certified endpoint. Every reached endpoint
    must subsequently restore in another fresh worker before publication.
    """
    policy, values = validate_policy(policy, muscle_names)
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Audit tolerance must be finite and positive")
    output_directory = Path(output_directory).resolve()
    if output_directory.exists():
        raise FileExistsError(f"Branch output already exists: {output_directory}")
    source.verify_files()
    runtime = build_runtime()
    program, solver = runtime["nmpc"], runtime["solver"]
    actual_names = tuple(model.muscle_name for model in program.nlp[0].model.muscles_dynamics_model)
    if actual_names != tuple(muscle_names):
        raise ValueError("Plan muscle order differs from the actual compiled model")
    binding = getattr(program, "fatigue_weight_binding", None)
    if binding is None:
        raise ValueError("Source RHO must already contain the fixed fatigue parameter graph")
    restored = restore_prepared_checkpoint(Path(source.archive_path), program,
                                           completed_cycles=source.completed_cycles)
    if (restored.get("restored_problem_sha256") != source.prepared_problem_sha256
            or restored.get("stimulation_history_complete") is not True):
        raise RuntimeError("Branch source did not restore exactly")
    nlp = program.nlp[0]
    initial = initial_guess_signature(snapshot_initial_guess(program))
    before = prepared_problem_arrays(program)
    weight_keys = (f"parameter_init:{FATIGUE_WEIGHT_PARAMETER_KEY}",
                   f"parameter_bounds:{FATIGUE_WEIGHT_PARAMETER_KEY}:min",
                   f"parameter_bounds:{FATIGUE_WEIGHT_PARAMETER_KEY}:max")
    if any(key not in before or not np.allclose(before[key], 1., atol=1e-12, rtol=0) for key in weight_keys):
        raise ValueError("This experiment requires the predeclared unit-weight anchor")
    work = float(nlp.x_bounds["E_prod"].min[0, 2])
    if not math.isfinite(work) or work <= 0 or nlp.x_bounds["E_prod"].max[0, 2] != work:
        raise ValueError("Source must prescribe a positive nominal work equality")
    update = binding.update(program, values)
    _unchanged_numerical_problem(before, prepared_problem_arrays(program))
    if initial_guess_signature(snapshot_initial_guess(program)) != initial or program.nlp[0] is not nlp:
        raise RuntimeError("Policy update changed the primal or rebuilt the NLP")
    output_directory.mkdir(parents=True, exist_ok=False)
    endpoints, cycles, compiled_solver = [], [], None
    for offset in range(1, max(policy["endpoint_cycles"]) + 1):
        program.total_optimization_run = offset - 1
        solution = solve_one_cycle(program, solver)
        if program.nlp[0] is not nlp:
            raise RuntimeError("Branch rebuilt its NLP between physical cycles")
        live_solver = getattr(getattr(program, "ocp_solver", None), "shaked_ocp_solver", None)
        if live_solver is None:
            raise RuntimeError("Cannot audit compiled solver reuse")
        if compiled_solver is not None and live_solver is not compiled_solver:
            raise RuntimeError("Branch rebuilt its compiled solver between physical cycles")
        compiled_solver = live_solver
        report = audit(solution, program, work, tolerance)
        if not isinstance(report, IndependentAudit):
            raise TypeError("Independent audit returned an invalid report")
        passed = (isinstance(report, IndependentAudit) and report.passed
                  and report.maximum_normalized_violation is not None
                  and math.isfinite(report.maximum_normalized_violation)
                  and 0 <= report.maximum_normalized_violation <= tolerance
                  and set(DEFAULT_CONSTRAINT_GROUPS).issubset(report.groups))
        row = {"branch_cycle": offset, "completed_cycles": source.completed_cycles + offset,
               "solver_status": str(getattr(solution, "status", "missing")), "certified": bool(passed),
               "independent_audit": dict(report.detail),
               "audit_tolerance": tolerance, "validated_constraint_groups": list(report.groups),
               "maximum_normalized_violation": report.maximum_normalized_violation}
        cycles.append(row)
        if not passed:
            break
        row["witness"] = save_witness(solution, program, output_directory, offset, report)
        if not isinstance(row["witness"], str) or not Path(row["witness"]).is_file():
            raise RuntimeError("Branch trajectory witness was not persisted")
        row["witness_sha256"] = sha256(Path(row["witness"]).read_bytes()).hexdigest()
        advance_one_cycle(program, solution, work)
        current_weights = np.asarray(program.parameter_init[FATIGUE_WEIGHT_PARAMETER_KEY].init).reshape(-1)
        if not np.allclose(current_weights, values, atol=1e-12, rtol=0):
            raise RuntimeError("Native advance changed the requested policy weights")
        if offset in policy["endpoint_cycles"]:
            exported = export_prepared_checkpoint(output_directory / f"cycle-{offset}.npz", program,
                completed_cycles=source.completed_cycles + offset, model_path=Path(source.model_path))
            endpoints.append({"branch_cycle": offset, **exported})
        source.verify_files()
    return {"schema_version": 1, "kind": "prepared_unilateral_reserve_branch_export",
            "policy": policy, "muscle_order": list(muscle_names), "source_restore": restored,
            "source_prepared_problem_sha256": source.prepared_problem_sha256,
            "source_completed_cycles": source.completed_cycles,
            "nominal_work_j": work, "weight_update": update, "cycles": cycles,
            "endpoints": endpoints, "compiled_solver_reused": True,
            "success": len(cycles) == max(policy["endpoint_cycles"]) and all(row["certified"] for row in cycles),
            "fresh_worker_replay_verified": False, "exact_bilateral_restart": False,
            "failure_interpretation": "failed nominal branch solves are indeterminate"}


def certified_endpoint_receipt(export: Mapping, endpoint: Mapping, restored: Mapping, *,
                               provenance: Mapping, producer_pid: int, verifier_pid: int) -> dict:
    if (type(producer_pid) is not int or type(verifier_pid) is not int or producer_pid == verifier_pid
            or min(producer_pid, verifier_pid) <= 0):
        raise ValueError("Endpoint requires a different fresh verifier process")
    if (restored.get("restored_problem_sha256") != endpoint.get("prepared_problem_sha256")
            or restored.get("restored_primal_signature") != endpoint.get("prepared_primal_signature")
            or restored.get("stimulation_history_complete") is not True):
        raise ValueError("Endpoint fresh-worker restoration mismatch")
    offset = endpoint["branch_cycle"]
    rows = [row for row in export["cycles"] if row["branch_cycle"] <= offset]
    if (len(rows) != offset or [row["branch_cycle"] for row in rows] != list(range(1, offset + 1))
            or not all(row["certified"] and row["maximum_normalized_violation"] is not None
                       and 0 <= row["maximum_normalized_violation"] <= row["audit_tolerance"]
                       and set(DEFAULT_CONSTRAINT_GROUPS).issubset(row["validated_constraint_groups"])
                       for row in rows)):
        raise ValueError("Endpoint has an uncertified trajectory prefix")
    return {"schema_version": 1, "kind": "certified_unilateral_task_reserve_branch_endpoint",
            **deepcopy(dict(provenance)), "policy": deepcopy(export["policy"]),
            "muscle_order": list(export["muscle_order"]), "branch_cycle": offset,
            "completed_cycles": endpoint["completed_cycles"], "endpoint": dict(endpoint),
            "source_restore": dict(export["source_restore"]), "cycles": deepcopy(rows),
            "compiled_solver_reused": export["compiled_solver_reused"],
            "fresh_worker_replay_verified": True, "exact_bilateral_restart": False,
            "restored": dict(restored), "producer_pid": producer_pid, "verifier_pid": verifier_pid,
            "audit_scope": "complete discrete NLP, no continuous ODE replay"}
