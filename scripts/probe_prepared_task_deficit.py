#!/usr/bin/env python3
"""Minimize normalized next-cycle task-work deficit from an exact frozen RHO state.

The prepared state/history/model/load are restored exactly. Pulse widths and
the one-cycle trajectory remain decision variables; terminal work alone is
relaxed to [0, requested work] and its normalized shortfall is minimized.
This is a local, nonconvex deficit estimate, not a global infeasibility proof.
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

from cocofest.optimization.conditional_capacity_feasibility import (
    ConditionalCapacitySlice, FrozenTaskDeficit, mapping_from_program,
)
from cocofest.optimization.task_reserve import TaskReserveCheckpoint
from cocofest.optimization.task_reserve_probe_adapter import independent_full_nlp_audit
from cocofest.simulation.rho_restart_checkpoint import (
    _runtime_state, prepared_problem_arrays, prepared_problem_digest, restore_prepared_checkpoint,
)
from cocofest.optimization.receding_horizon_initial_guess import snapshot_initial_guess
from scripts.probe_independent_rho_task_reserve import _load_arm
from scripts.run_local_task_reserve_branch import _build_runtime, _solve


def run(*, receipt: Path, checkpoint: Path, side: str, output_directory: Path, cpu: int,
        pw_tracking_scale_s: float, max_iterations: int = 4000, tolerance: float = 1e-6) -> Path:
    import casadi as ca
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver

    receipt, checkpoint = Path(receipt).resolve(strict=True), Path(checkpoint).resolve(strict=True)
    output_directory = Path(output_directory).resolve()
    if output_directory.exists(): raise FileExistsError(output_directory)
    if cpu not in os.sched_getaffinity(0): raise ValueError("Requested CPU is outside current process affinity")
    if (not np.isfinite(pw_tracking_scale_s) or pw_tracking_scale_s <= 0.
            or type(max_iterations) is not int or max_iterations < 1
            or not np.isfinite(tolerance) or not 0. < tolerance <= 1e-5):
        raise ValueError("Invalid scale, iteration budget, or tolerance")
    os.sched_setaffinity(0, {cpu})

    source_at_c140, payload = _load_arm(receipt, side)
    payload["experimental_pw_tracking"] = {"scale_s": float(pw_tracking_scale_s)}
    with np.load(checkpoint, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata__json"].item()))
    completed = metadata.get("producer_completed_windows")
    if type(completed) is not int or completed <= source_at_c140.completed_cycles:
        raise ValueError("Expected a later prepared checkpoint from the branch continuation")
    source = TaskReserveCheckpoint.from_archive(checkpoint, completed_cycles=completed,
        model_path=Path(source_at_c140.model_path), task_context=json.loads(source_at_c140.task_context_json))
    source.verify_files()

    build_start = perf_counter()
    runtime = _build_runtime(payload, side, cpu=cpu)
    construction_seconds = perf_counter() - build_start
    program, args = runtime["nmpc"], runtime["args"]
    binding = getattr(program, "pw_tracking_binding", None)
    if binding is None or not np.isclose(binding.scale_s, pw_tracking_scale_s, rtol=0., atol=0.):
        raise RuntimeError("Rebuilt RHO does not match the migrated PW-tracking graph")
    restored = restore_prepared_checkpoint(checkpoint, program, completed_cycles=completed)
    if (restored.get("restored_problem_sha256") != source.prepared_problem_sha256
            or restored.get("stimulation_history_complete") is not True):
        raise RuntimeError("Fresh worker did not restore the exact frozen state/history checkpoint")
    weights = np.asarray(program.parameter_init["rho_fatigue_weights"].init, float).reshape(-1)
    if not np.array_equal(weights, np.ones(len(binding.muscle_names))):
        raise ValueError("Prepared endpoint is not on the declared unit-fatigue RHO policy")

    # Instantiate the exact source OCP graph. This full-work trial is recorded
    # for context but need not pass; the auxiliary objective below quantifies
    # the best locally achievable normalized task-work deficit explicitly.
    restored_digest = prepared_problem_digest(snapshot_initial_guess(program), prepared_problem_arrays(program),
                                              _runtime_state(program))
    nominal_work = float(json.loads(source.task_context_json)["nominal_work_j"])
    baseline_start = perf_counter()
    try:
        baseline_solution = _solve(program, runtime["solver"])
        baseline_seconds = perf_counter() - baseline_start
        baseline_audit = independent_full_nlp_audit(baseline_solution, program,
            target_work_j=nominal_work, tolerance=tolerance)
        baseline = {"status": str(getattr(baseline_solution, "status", "missing")),
            "independent_audit_passed": bool(baseline_audit.passed),
            "independent_audit": dict(baseline_audit.detail), "solve_seconds": baseline_seconds}
    except Exception as error:
        baseline_seconds = perf_counter() - baseline_start
        baseline = {"exception": f"{type(error).__name__}: {error}",
            "independent_audit_passed": False, "solve_seconds": baseline_seconds}
    current_digest = prepared_problem_digest(snapshot_initial_guess(program), prepared_problem_arrays(program),
                                             _runtime_state(program))
    if current_digest != restored_digest:
        raise RuntimeError("Nominal diagnostic changed prepared state/history before deficit minimization")
    if program.ocp_solver is None:
        raise RuntimeError("Nominal trial did not expose the NLP graph required by the deficit objective")

    model_data = json.loads(Path(source.model_path).read_text(encoding="utf-8"))
    muscle_names = tuple(model_data["muscles"])
    slice_ = ConditionalCapacitySlice.from_checkpoint(source, muscle_order=muscle_names, trust_radius=.05)
    problem = FrozenTaskDeficit(graph=program.ocp_solver.nlp, limits=program.ocp_solver.limits,
        **mapping_from_program(program, slice_), nominal_work_j=float(json.loads(source.task_context_json)["nominal_work_j"]),
        center=slice_.center)
    limits = problem.bounds_for(slice_.center, relax_work=True)
    limits["x0"] = problem.limits["x0"].copy()
    objective_hash_before = problem.production_objective_sha256
    constraint_hash_before = problem.constraint_graph_sha256
    solver_options = driver.configure_ipopt_solver(max_iterations=max_iterations,
        linear_solver="ma57", tolerance=float(args.nlp_tolerance), hsl_library=args.ipopt_hsl_library,
        print_level=0, advanced_options=driver._ipopt_advanced_options(args)).as_dict(program.ocp_solver)
    solver_start = perf_counter()
    solver = ca.nlpsol("frozen_normalized_work_deficit", "ipopt", problem.auxiliary_graph, solver_options)
    solver_build_seconds = perf_counter() - solver_start
    solve_start = perf_counter()
    result = solver(**limits)
    solve_seconds = perf_counter() - solve_start
    vector = np.asarray(result["x"], float).reshape(-1)
    stats = solver.stats()
    audit = problem.audit(vector, slice_.center, tolerance=tolerance)
    final_work = float(vector[problem.work_index] * problem.work_coefficient)
    report = {"schema_version": 1, "kind": "frozen_state_normalized_next_cycle_work_deficit",
        "source_receipt": str(receipt), "source_anchor_cycles": source_at_c140.completed_cycles,
        "frozen_checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "prepared_completed_cycles": completed, "prepared_problem_sha256": source.prepared_problem_sha256,
        "fresh_worker_restore": restored, "preserved": ["all initial Ding states", "mechanical initial state",
            "pulse-width/stimulation history", "model and resistance", "cycle closure and path constraints"],
        "free_variables": ["next-cycle trajectory", "PW controls within physical bounds"],
        "only_relaxed_task_condition": "terminal work in [0, nominal_work]",
        "objective": "minimize (nominal_work - achieved_work) / nominal_work",
        "nominal_work_j": problem.nominal_work_j, "achieved_work_j": final_work,
        "normalized_work_deficit": float(audit["observed_normalized_work_deficit"] or 0.),
        "reported_task_witness": bool(audit["nominal_task_witnessed"]),
        "trajectory_constraints_validated": bool(audit["phase_one_trajectory_validated"]),
        "audit": audit, "solver": {"name": "IPOPT-MA57", "return_status": stats.get("return_status"),
            "success": bool(stats.get("success")), "iterations": stats.get("iter_count"),
            "max_iterations": max_iterations, "tolerance": tolerance},
        "nominal_full_work_trial": baseline,
        "timings_s": {"fresh_runtime_construction": construction_seconds,
            "nominal_trial": baseline_seconds,
            "auxiliary_solver_construction": solver_build_seconds, "auxiliary_solve": solve_seconds},
        "capacity_center": list(slice_.center), "capacity_muscle_order": list(muscle_names),
        "capacity_condition_sha256": slice_.condition_sha256,
        "production_objective_sha256": objective_hash_before, "constraint_graph_sha256": constraint_hash_before,
        "production_objective_unchanged": True, "physiological_failure_certified": False,
        "interpretation": "A feasible nominal-work witness proves one admissible cycle. A positive deficit is the shortfall found by this local auxiliary search, not proof that no other PW solution exists."}

    output_directory.mkdir(parents=True, exist_ok=False)
    candidate = output_directory / "minimum-deficit-candidate.npz"
    np.savez_compressed(candidate, vector=vector, metadata__json=np.asarray(json.dumps(report, sort_keys=True, allow_nan=False)))
    report["candidate_path"] = str(candidate)
    report["candidate_sha256"] = sha256(candidate.read_bytes()).hexdigest()
    destination = output_directory / "deficit-report.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--pw-tracking-scale-s", type=float, default=1e-5)
    parser.add_argument("--max-iterations", type=int, default=4000)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    print(run(**vars(parser.parse_args(argv))))


if __name__ == "__main__": main()
