#!/usr/bin/env python3
"""Evaluate the compiled soft PW-tracking objective on a certified c140 RHO anchor.

This is a one-cycle mechanics/solver pilot, not an endurance estimate. It solves
an inactive reference and three equal-activation targets (center, shoulder +,
shoulder -) from the identical migrated c140 checkpoint with one compiled NLP.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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

from cocofest.optimization.continuation_value_pw_action import DirectPulseWidthAction
from cocofest.optimization.task_reserve_probe_adapter import independent_full_nlp_audit
from cocofest.optimization.parametric_pw_tracking import migrate_prepared_checkpoint
from cocofest.simulation.rho_restart_checkpoint import restore_prepared_checkpoint
from scripts.probe_independent_rho_task_reserve import _load_arm
from scripts.run_local_task_reserve_branch import _build_runtime, _save_witness, _solve


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def run(*, source_receipt: Path, extended_archive: Path, output_directory: Path,
        cpu: int, amplitude_fraction: float = .05, activation: float = .1,
        scale_s: float = 1e-4, tolerance: float = 1e-5) -> Path:
    source_receipt, extended_archive = source_receipt.resolve(strict=True), extended_archive.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    if cpu not in os.sched_getaffinity(0):
        raise ValueError("Requested pilot CPU is outside the current process affinity")
    source, original_payload = _load_arm(source_receipt, "left")
    if source.completed_cycles != 140:
        raise ValueError("This controlled pilot is predeclared for the c140 anchor")
    payload = deepcopy(original_payload)
    payload["experimental_pw_tracking"] = {"scale_s": scale_s}
    context = json.loads(source.task_context_json)
    compile_start = time.perf_counter()
    runtime = _build_runtime(payload, "left", cpu=cpu)
    compile_seconds = time.perf_counter() - compile_start
    migrated = migrate_prepared_checkpoint(source.archive_path, extended_archive, runtime["nmpc"],
                                           completed_cycles=140) if not extended_archive.exists() else None
    # A separately produced migration is accepted only after its byte digest is checked.
    migration_receipt = extended_archive.with_suffix(".migration.json")
    migration = migrated or json.loads(migration_receipt.read_text(encoding="utf-8"))
    if (migration.get("kind") != "pw_tracking_checkpoint_migration"
            or migration.get("completed_cycles") != 140
            or migration.get("migration_checks", {}).get("source_projection_digest_matches") is not True
            or not np.isclose(migration.get("pw_graph_signature", {}).get("scale_s", np.nan), scale_s)
            or sha256(extended_archive.read_bytes()).hexdigest() != migration.get("enriched_archive_sha256")):
        raise ValueError("PW-tracking migration evidence does not authenticate this c140 archive")

    from cocofest.optimization.task_reserve import TaskReserveCheckpoint
    checkpoint = TaskReserveCheckpoint.from_archive(
        extended_archive, completed_cycles=140, model_path=Path(source.model_path), task_context=context)
    program, solver = runtime["nmpc"], runtime["solver"]
    binding = getattr(program, "pw_tracking_binding", None)
    if binding is None or binding.summary().get("graph_version") != migration["pw_graph_signature"].get("graph_version"):
        raise RuntimeError("Fresh runtime did not rebuild the exact migrated PW-tracking graph")
    restored = restore_prepared_checkpoint(extended_archive, program, completed_cycles=140)
    if (restored.get("restored_problem_sha256") != checkpoint.prepared_problem_sha256
            or restored.get("stimulation_history_complete") is not True):
        raise RuntimeError("Fresh pilot worker failed exact c140 checkpoint restoration")

    names = tuple(binding.muscle_names)
    from cocofest.optimization.continuation_value_pw_action import PW_PREFIX, _phase_zero
    controls = _phase_zero(program.nlp[0].u_init)
    seed = np.vstack([np.asarray(controls[PW_PREFIX + name].init, float).reshape(1, -1) for name in names])
    work = float(program.nlp[0].x_bounds["E_prod"].min[0, 2])
    if not np.isfinite(work) or work <= 0.:
        raise ValueError("c140 archive does not prescribe positive cycle work")
    output_directory.mkdir(parents=True, exist_ok=False)
    direction = np.array([1., -1., 0., 0.]) / np.sqrt(2.)
    conditions = [
        ("reference_inactive", None, 0.),
        ("target_center_active", DirectPulseWidthAction(names, (0.,) * len(names), amplitude_fraction, activation), activation),
        ("shoulder_plus", DirectPulseWidthAction(names, tuple(direction), amplitude_fraction, activation), activation),
        ("shoulder_minus", DirectPulseWidthAction(names, tuple(-direction), amplitude_fraction, activation), activation),
    ]
    rows = []
    solver_identity, nlp_identity = id(solver), id(program.nlp[0])
    for name, action, alpha in conditions:
        restore = restore_prepared_checkpoint(extended_archive, program, completed_cycles=140)
        if restore.get("restored_problem_sha256") != checkpoint.prepared_problem_sha256:
            raise RuntimeError(f"Condition {name} did not start from the exact common c140 source")
        reference = seed
        if name != "reference_inactive":
            # The optimized, inactive reference is the center target for all
            # active cases, making activation and graph identical in contrasts.
            reference = optimized_reference
            target, action_receipt = action.target_from_reference(program, reference)
        else:
            target, action_receipt = seed, {"kind": "inactive_reference", "tracking_activation": 0.}
        binding.update_nmpc(program, target=target, activation=alpha)
        start = time.perf_counter()
        solution = _solve(program, solver)
        solve_seconds = time.perf_counter() - start
        audit = independent_full_nlp_audit(solution, program, target_work_j=work, tolerance=tolerance)
        from bioptim import SolutionMerge
        realized_dict = solution.decision_controls(to_merge=SolutionMerge.NODES)
        realized = np.vstack([np.asarray(realized_dict[PW_PREFIX + muscle], float).reshape(1, -1)
                              for muscle in names])
        row = {"condition": name, "tracking_activation": alpha, "solver_status": str(getattr(solution, "status", "missing")),
               "solve_seconds": solve_seconds, "independent_audit_passed": bool(audit.passed),
               "maximum_normalized_violation": audit.maximum_normalized_violation,
               "audit_groups": list(audit.groups), "audit_detail": dict(audit.detail),
               "action": action_receipt, "target_pw": None if action is None else np.asarray(target).tolist(),
               "realized_pw": realized.tolist(),
               "max_target_error_s": None if action is None else float(np.max(np.abs(realized - target))),
               "reference_seed_pw": seed.tolist(),
               "fatigue_states_final": {key: float(np.asarray(value)[:, -1].mean())
                    for key, value in solution.decision_states(to_merge=SolutionMerge.NODES).items()
                    if key.startswith("A_") or key.startswith("F_")}}
        rows.append(_jsonable(row))
        if not audit.passed:
            raise RuntimeError(f"Independent full-NLP audit failed for {name}; pilot stops without labels")
        if name == "reference_inactive":
            optimized_reference = realized.copy()
        _save_witness(solution, program, output_directory, len(rows), audit)
        if id(solver) != solver_identity or id(program.nlp[0]) != nlp_identity:
            raise RuntimeError("The NLP or solver object was rebuilt between pilot conditions")

    result = {"schema_version": 1, "kind": "pw_tracking_c140_one_cycle_pilot", "completed_cycles_before": 140,
              "source_receipt": str(source_receipt), "source_archive_sha256": sha256(extended_archive.read_bytes()).hexdigest(),
              "migrated_prepared_problem_sha256": checkpoint.prepared_problem_sha256,
              "fresh_worker_restoration": restored, "physical_task_context": context,
              "muscle_order": list(names), "nominal_work_j": work, "solver": "same compiled IPOPT OCP solver",
              "compile_and_runtime_build_seconds": compile_seconds,
              "reused_nlp_identity": nlp_identity, "reused_solver_identity": solver_identity,
              "scale_s": binding.scale_s, "amplitude_fraction": amplitude_fraction,
              "active_tracking_activation": activation, "conditions": rows,
              "interpretation": "one-cycle local PW-tracking/feasibility pilot only; not a remaining-endurance label"}
    destination = output_directory / "pilot.json"
    destination.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--extended-archive", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument("--amplitude-fraction", type=float, default=.05)
    parser.add_argument("--activation", type=float, default=.1)
    parser.add_argument("--scale-s", type=float, default=1e-4)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args(argv)
    print(run(source_receipt=args.source_receipt, extended_archive=args.extended_archive,
              output_directory=args.output_directory, cpu=args.cpu,
              amplitude_fraction=args.amplitude_fraction, activation=args.activation, scale_s=args.scale_s,
              tolerance=args.tolerance))


if __name__ == "__main__":
    main()
