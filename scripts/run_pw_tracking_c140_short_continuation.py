#!/usr/bin/env python3
"""Compare short unit-RHO continuations under causal PW-target policies.

One compiled one-cycle IPOPT NLP is reused within each condition. Every
condition starts from the same migrated c140 checkpoint. By default, a PW
intervention is active for cycle 1 only; ``--repeat-action-every-cycle`` instead
refreshes its target from the previous audited cycle, with no extra reference
solve. This is a local mechanism pilot, not a physiological failure or
endurance label.
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

from cocofest.optimization.continuation_value_pw_action import DirectPulseWidthAction, PW_PREFIX
from cocofest.optimization.parametric_pw_tracking import migrate_prepared_checkpoint
from cocofest.optimization.task_reserve import TaskReserveCheckpoint
from cocofest.optimization.task_reserve_probe_adapter import independent_full_nlp_audit
from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint, restore_prepared_checkpoint
from scripts.probe_independent_rho_task_reserve import _load_arm
from scripts.run_local_task_reserve_branch import _advance, _build_runtime, _save_witness, _solve


def _jsonable(value):
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)): return value.item()
    if isinstance(value, Path): return str(value)
    return value


def _state_summary(solution):
    from bioptim import SolutionMerge
    states = solution.decision_states(to_merge=SolutionMerge.NODES)
    result = {}
    for prefix, label in (("A_", "fatigue_scaling_A"), ("F_", "force_F"), ("Cn_", "calcium_Cn"),
                          ("Tau1_", "tau1"), ("Km_", "km")):
        values = {key[len(prefix):]: float(np.asarray(value, float)[0, -1])
                  for key, value in states.items() if key.startswith(prefix)}
        if values:
            result[label] = values
    return result


def _initial_pw_target(program, muscle_names):
    """Read the prepared direct-PW warm start without performing a RHO solve."""
    from cocofest.optimization.continuation_value_pw_action import _phase_zero
    initial = _phase_zero(program.nlp[0].u_init)
    return np.vstack([np.asarray(initial[PW_PREFIX + muscle].init, float).reshape(1, -1)
                      for muscle in muscle_names])


def run(*, source_receipt: Path, extended_archive: Path, output_directory: Path, cpu: int,
        side: str = "left",
        cycles: int = 10, scale_s: float = 1e-5, amplitude_fraction: float = .05,
        activation: float = 1., tolerance: float = 1e-5,
        repeat_action_every_cycle: bool = False, paired_policy_only: bool = False) -> Path:
    source_receipt = source_receipt.resolve(strict=True)
    extended_archive = extended_archive.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists(): raise FileExistsError(output_directory)
    if type(cycles) is not int or cycles < 2: raise ValueError("Continuation length must be at least two cycles")
    if cpu not in os.sched_getaffinity(0): raise ValueError("Requested CPU is outside process affinity")
    if side not in ("left", "right"): raise ValueError("Side must be 'left' or 'right'")
    if paired_policy_only and not repeat_action_every_cycle:
        raise ValueError("Paired policy-only mode requires repeated-action mode")

    source, base_payload = _load_arm(source_receipt, side)
    if source.completed_cycles != 140: raise ValueError("Only the certified c140 anchor is supported")
    payload = deepcopy(base_payload)
    payload["experimental_pw_tracking"] = {"scale_s": scale_s}
    compile_start = time.perf_counter()
    runtime = _build_runtime(payload, side, cpu=cpu)
    compile_seconds = time.perf_counter() - compile_start
    if not extended_archive.exists():
        migrate_prepared_checkpoint(source.archive_path, extended_archive, runtime["nmpc"], completed_cycles=140)
    migration_path = extended_archive.with_suffix(".migration.json")
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    if (migration.get("kind") != "pw_tracking_checkpoint_migration"
            or migration.get("migration_checks", {}).get("source_projection_digest_matches") is not True
            or not np.isclose(migration.get("pw_graph_signature", {}).get("scale_s", np.nan), scale_s)
            or sha256(extended_archive.read_bytes()).hexdigest() != migration.get("enriched_archive_sha256")):
        raise ValueError("Migrated checkpoint does not authenticate the requested PW objective graph")

    context = json.loads(source.task_context_json)
    checkpoint = TaskReserveCheckpoint.from_archive(extended_archive, completed_cycles=140,
        model_path=Path(source.model_path), task_context=context)
    program, solver = runtime["nmpc"], runtime["solver"]
    binding = getattr(program, "pw_tracking_binding", None)
    if binding is None or binding.summary().get("graph_version") != migration["pw_graph_signature"].get("graph_version"):
        raise RuntimeError("Built runtime and migrated checkpoint use different objective graphs")
    names = tuple(binding.muscle_names)
    if len(names) != 4:
        raise ValueError("This shoulder pilot expects the four Ding muscle groups")
    work = float(program.nlp[0].x_bounds["E_prod"].min[0, 2])
    if not np.isfinite(work) or work <= 0.: raise ValueError("Anchor lacks positive prescribed cycle work")
    output_directory.mkdir(parents=True, exist_ok=False)
    solver_id, nlp_id = id(solver), id(program.nlp[0])

    # The ordinary pilot obtains a shared optimized reference with one extra
    # solve. The repeated-action pilot avoids that solve and starts causally
    # from the prepared checkpoint's direct-PW warm start.
    restore = restore_prepared_checkpoint(extended_archive, program, completed_cycles=140)
    if restore.get("restored_problem_sha256") != checkpoint.prepared_problem_sha256:
        raise RuntimeError("Initial pilot restoration failed the migrated checkpoint digest")
    from bioptim import SolutionMerge
    reference_seconds, reference_audit = 0., None
    if repeat_action_every_cycle:
        reference = _initial_pw_target(program, names)
        reference_strategy = "prepared c140 direct-PW initial guess; then previous audited cycle PW"
    else:
        binding.update_nmpc(program, target=binding.target, activation=0.)
        start = time.perf_counter()
        reference_solution = _solve(program, solver)
        reference_seconds = time.perf_counter() - start
        reference_audit = independent_full_nlp_audit(reference_solution, program, target_work_j=work, tolerance=tolerance)
        if not reference_audit.passed:
            raise RuntimeError("Nominal reference RHO cycle failed the independent audit")
        optimized = reference_solution.decision_controls(to_merge=SolutionMerge.NODES)
        reference = np.vstack([np.asarray(optimized[PW_PREFIX + muscle], float).reshape(1, -1) for muscle in names])
        reference_strategy = "extra optimized inactive RHO cycle at c140"
    direction = np.array([1., -1., 0., 0.]) / np.sqrt(2.)
    actions = [("rho_unit", None, 0.)]
    if not repeat_action_every_cycle:
        actions.append(("pw_center_active", DirectPulseWidthAction(names, (0.,) * 4, amplitude_fraction, activation), activation))
    else:
        # A zero-direction target is a causal inter-cycle PW smoothness term:
        # it tracks the previous realized cycle without accumulating a bias.
        actions.append(("pw_previous_cycle", DirectPulseWidthAction(names, (0.,) * 4, amplitude_fraction, activation), activation))
    if not paired_policy_only:
        actions.extend([
            ("pw_shoulder_plus", DirectPulseWidthAction(names, tuple(direction), amplitude_fraction, activation), activation),
            ("pw_shoulder_minus", DirectPulseWidthAction(names, tuple(-direction), amplitude_fraction, activation), activation),
        ])
    rows = []
    for condition, action, alpha in actions:
        (output_directory / condition).mkdir()
        checkpoint_directory = output_directory / condition / "checkpoints"
        checkpoint_directory.mkdir()
        restore = restore_prepared_checkpoint(extended_archive, program, completed_cycles=140)
        if restore.get("restored_problem_sha256") != checkpoint.prepared_problem_sha256:
            raise RuntimeError(f"{condition} did not restart from the common c140 checkpoint")
        fatigue_weights = np.asarray(program.parameter_init["rho_fatigue_weights"].init, dtype=float).reshape(-1)
        if not np.array_equal(fatigue_weights, np.ones(len(names))):
            raise RuntimeError(f"{condition} source does not carry the declared unit fatigue weights")
        if action is None:
            target, action_detail = reference, {"kind": "ordinary_rho", "tracking_activation": 0.}
        else:
            target, action_detail = action.target_from_reference(program, reference)
        binding.update_nmpc(program, target=target, activation=alpha)
        previous_pw = reference.copy()
        condition_rows = []
        cycle_solution = None
        for local_cycle in range(1, cycles + 1):
            if repeat_action_every_cycle:
                if action is None:
                    target, action_detail = previous_pw.copy(), {"kind": "ordinary_rho", "tracking_activation": 0.}
                else:
                    target, action_detail = action.target_from_reference(program, previous_pw)
                    action_detail["interpretation"] = "soft PW target refreshed every cycle from the previous audited PW trace"
                binding.update_nmpc(program, target=target, activation=alpha)
            started = time.perf_counter()
            try:
                cycle_solution = _solve(program, solver)
            except Exception as error:
                condition_rows.append({"local_cycle": local_cycle, "absolute_cycle": 140 + local_cycle,
                    "solver_exception": f"{type(error).__name__}: {error}", "audit_passed": False,
                    "failure_interpretation": "indeterminate numerical/solver stop; not physiological failure"})
                break
            solve_seconds = time.perf_counter() - started
            audit = independent_full_nlp_audit(cycle_solution, program, target_work_j=work, tolerance=tolerance)
            cycle_row = {"local_cycle": local_cycle, "absolute_cycle": 140 + local_cycle,
                "solver_status": str(getattr(cycle_solution, "status", "missing")),
                "solve_seconds": solve_seconds, "audit_passed": bool(audit.passed),
                "maximum_normalized_violation": audit.maximum_normalized_violation,
                "audit_detail": dict(audit.detail), "ding_state_endpoints": _state_summary(cycle_solution),
                "failure_interpretation": None if audit.passed else
                    "indeterminate solver/discrete-NLP audit stop; not physiological failure"}
            realized_dict = cycle_solution.decision_controls(to_merge=SolutionMerge.NODES)
            realized = np.vstack([np.asarray(realized_dict[PW_PREFIX + muscle], float).reshape(1, -1)
                                  for muscle in names])
            if repeat_action_every_cycle or local_cycle == 1:
                cycle_row.update(action=action_detail, target_pw=target.tolist(), realized_pw=realized.tolist(),
                    pw_rms_from_nominal_reference_s=float(np.sqrt(np.mean((realized - reference) ** 2))),
                    pw_rms_error_to_target_s=None if action is None else float(np.sqrt(np.mean((realized - target) ** 2))))
            if local_cycle == 1 or local_cycle == cycles:
                _save_witness(cycle_solution, program, output_directory / condition, local_cycle, audit)
            condition_rows.append(cycle_row)
            if not audit.passed:
                break
            previous_pw = realized.copy()
            if local_cycle < cycles:
                # End the experimental action after cycle 1; continue with
                # the unchanged fatigue objective and constraints thereafter.
                if local_cycle == 1 and not repeat_action_every_cycle:
                    binding.update_nmpc(program, target=reference, activation=0.)
                _advance(program, cycle_solution, work)
                checkpoint_path = checkpoint_directory / f"next-cycle-after-{140 + local_cycle}.npz"
                exported = export_prepared_checkpoint(checkpoint_path, program,
                    completed_cycles=140 + local_cycle, model_path=Path(source.model_path))
                cycle_row["next_cycle_checkpoint"] = exported
            if id(solver) != solver_id or id(program.nlp[0]) != nlp_id:
                raise RuntimeError("RHO advance rebuilt the compiled NLP/solver during the pilot")
        rows.append({"condition": condition, "initial_tracking_activation": alpha,
                     "continuation_policy": ("one compiled RHO solve per cycle; target is the previous audited PW trace"
                         if repeat_action_every_cycle else "ordinary unit-weight RHO after first cycle"),
                     "completed_local_cycles": len(condition_rows), "all_cycles_audited": all(x["audit_passed"] for x in condition_rows),
                     "cycles": condition_rows})

    result = {"schema_version": 1, "kind": "pw_tracking_c140_short_continuation",
        "source_receipt": str(source_receipt), "source_completed_cycles": 140,
        "migrated_archive": str(extended_archive), "migrated_archive_sha256": sha256(extended_archive.read_bytes()).hexdigest(),
        "migrated_prepared_problem_sha256": checkpoint.prepared_problem_sha256,
        "fresh_worker_checkpoint_restore": restore, "physical_task_context": context,
        "muscle_order": list(names), "nominal_work_j": work, "scale_s": scale_s,
        "compile_and_runtime_build_seconds": compile_seconds, "inactive_reference_solve_seconds": reference_seconds,
        "reference_strategy": reference_strategy, "one_solve_per_cycle": bool(repeat_action_every_cycle),
        "reference_cycle_audit": None if reference_audit is None else dict(reference_audit.detail), "compiled_solver_reused": True,
        "reused_nlp_identity": nlp_id, "reused_solver_identity": solver_id,
        "conditions": rows, "interpretation": ("exploratory causal PW-persistence policy; not a demonstrated endurance optimizer or failure label"
            if repeat_action_every_cycle else "short local state-response pilot only; not an endurance or failure label")}
    destination = output_directory / "continuation.json"
    destination.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--extended-archive", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--scale-s", type=float, default=1e-5)
    parser.add_argument("--amplitude-fraction", type=float, default=.05)
    parser.add_argument("--activation", type=float, default=1.)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--repeat-action-every-cycle", action="store_true",
                        help="Update PW target from the previous cycle; no extra reference RHO solve.")
    parser.add_argument("--paired-policy-only", action="store_true",
                        help="Run only unit RHO and previous-cycle PW tracking; requires repeated-action mode.")
    args = parser.parse_args(argv)
    print(run(source_receipt=args.source_receipt, extended_archive=args.extended_archive,
        output_directory=args.output_directory, cpu=args.cpu, side=args.side, cycles=args.cycles, scale_s=args.scale_s,
        amplitude_fraction=args.amplitude_fraction, activation=args.activation, tolerance=args.tolerance,
        repeat_action_every_cycle=args.repeat_action_every_cycle, paired_policy_only=args.paired_policy_only))


if __name__ == "__main__": main()
