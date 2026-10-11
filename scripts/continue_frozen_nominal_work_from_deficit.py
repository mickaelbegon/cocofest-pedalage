#!/usr/bin/env python3
"""Try to recover the nominal next-cycle RHO from a frozen deficit candidate.

The source state, model, resistance, stimulation history and trajectory NLP
are restored exactly. First, the ordinary full-work problem is warm-started
from the minimum-deficit candidate. If that fails its independent audit, an
optional terminal-work homotopy is attempted on the same NLP and same frozen
initial state. A found nominal witness disproves a physiological-stop claim;
failure to find one remains inconclusive.
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


def _sha(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _shift_pw_seed(program, graph, seed, limits, direction, *, fraction: float, muscle_names):
    """Perturb only PW rows using Bioptim's authoritative flattened layout."""
    from cocofest.optimization.continuation_value_pw_action import PW_PREFIX

    layout = program.vector_layout
    vector = np.asarray(seed, dtype=float).reshape(-1, 1).copy()
    lower_data = layout.unstack(np.asarray(limits["lbx"], dtype=float).reshape(-1, 1))
    upper_data = layout.unstack(np.asarray(limits["ubx"], dtype=float).reshape(-1, 1))
    shifted_data = layout.unstack(vector)
    row_indices = {name: np.asarray(program.nlp[0].controls.key_index(PW_PREFIX + name), dtype=int)
                   for name in muscle_names}
    for key, (sl, _columns) in layout.index_map.items():
        if len(key) != 3 or key[0] != 0 or key[1] != "controls":
            continue
        controls = shifted_data[key]
        for name, coefficient in zip(muscle_names, direction, strict=True):
            if coefficient == 0.:
                continue
            rows = row_indices[name]
            lower = np.asarray(lower_data[key], dtype=float)[rows, :]
            upper = np.asarray(upper_data[key], dtype=float)[rows, :]
            current = np.asarray(controls, dtype=float)[rows, :]
            controls[rows, :] = np.clip(current + fraction * float(coefficient) * (upper - lower),
                                         lower, upper)
        shifted_data[key] = controls
    shifted = np.empty_like(vector)
    for key, (sl, _columns) in layout.index_map.items():
        shifted[sl, 0] = np.asarray(shifted_data[key], dtype=float).reshape(-1, order="F")
    return shifted.reshape(-1)


def run(*, receipt: Path, checkpoint: Path, candidate: Path, side: str, output_directory: Path,
        cpu: int, pw_tracking_scale_s: float = 1e-5, max_iterations: int = 4000,
        tolerance: float = 1e-6, homotopy_steps: int = 4,
        hessian_approximation: str = "exact") -> Path:
    import casadi as ca
    from bioptim import Solution
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver

    receipt = Path(receipt).resolve(strict=True)
    checkpoint = Path(checkpoint).resolve(strict=True)
    candidate = Path(candidate).resolve(strict=True)
    output_directory = Path(output_directory).resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    if cpu not in os.sched_getaffinity(0):
        raise ValueError("Requested CPU is outside current process affinity")
    if (not np.isfinite(pw_tracking_scale_s) or pw_tracking_scale_s <= 0.
            or type(max_iterations) is not int or max_iterations < 1
            or not np.isfinite(tolerance) or not 0. < tolerance <= 1e-5
            or type(homotopy_steps) is not int or homotopy_steps < 1
            or hessian_approximation not in ("exact", "limited-memory")):
        raise ValueError("Invalid solver settings")
    os.sched_setaffinity(0, {cpu})

    source_at_anchor, payload = _load_arm(receipt, side)
    payload["experimental_pw_tracking"] = {"scale_s": float(pw_tracking_scale_s)}
    with np.load(checkpoint, allow_pickle=False) as archive:
        checkpoint_meta = json.loads(str(archive["metadata__json"].item()))
    completed = checkpoint_meta.get("producer_completed_windows")
    if type(completed) is not int or completed <= source_at_anchor.completed_cycles:
        raise ValueError("Frozen checkpoint must be later than the source receipt")
    source = TaskReserveCheckpoint.from_archive(checkpoint, completed_cycles=completed,
        model_path=Path(source_at_anchor.model_path), task_context=json.loads(source_at_anchor.task_context_json))
    source.verify_files()

    with np.load(candidate, allow_pickle=False) as archive:
        seed = np.asarray(archive["vector"], dtype=float).reshape(-1)
        candidate_report = json.loads(str(archive["metadata__json"].item()))
    if (candidate_report.get("prepared_completed_cycles") != completed
            or candidate_report.get("prepared_problem_sha256") != source.prepared_problem_sha256
            or candidate_report.get("frozen_checkpoint") != str(checkpoint)
            or candidate_report.get("checkpoint_sha256") != _sha(checkpoint)):
        raise ValueError("Deficit candidate provenance does not match the exact frozen checkpoint")
    if candidate_report.get("trajectory_constraints_validated") is not True:
        raise ValueError("Only an independently validated trajectory candidate can seed recovery")

    runtime = _build_runtime(payload, side, cpu=cpu)
    program, args = runtime["nmpc"], runtime["args"]
    restored = restore_prepared_checkpoint(checkpoint, program, completed_cycles=completed)
    if (restored.get("restored_problem_sha256") != source.prepared_problem_sha256
            or restored.get("stimulation_history_complete") is not True):
        raise RuntimeError("Exact frozen state/history restoration failed")
    frozen_digest = prepared_problem_digest(snapshot_initial_guess(program), prepared_problem_arrays(program),
                                            _runtime_state(program))
    # Bioptim lazily materializes the flattened NLP graph on its first solve.
    # This diagnostic solve does not advance the physical cycle; guard the
    # complete prepared state/history digest around it.
    initialization_start = perf_counter()
    try:
        initialization_solution = _solve(program, runtime["solver"])
        initialization_audit = independent_full_nlp_audit(initialization_solution, program,
            target_work_j=float(json.loads(source.task_context_json)["nominal_work_j"]), tolerance=tolerance)
        initialization_trial = {"solver_status": str(getattr(initialization_solution, "status", "missing")),
            "audit_passed": bool(initialization_audit.passed), "audit": dict(initialization_audit.detail),
            "solve_seconds": perf_counter() - initialization_start}
    except Exception as error:
        initialization_trial = {"solver_exception": f"{type(error).__name__}: {error}",
            "audit_passed": False, "solve_seconds": perf_counter() - initialization_start}
    if prepared_problem_digest(snapshot_initial_guess(program), prepared_problem_arrays(program),
                               _runtime_state(program)) != frozen_digest:
        raise RuntimeError("NLP graph initialization changed the exact frozen state/history")
    if initialization_trial.get("audit_passed"):
        output_directory.mkdir(parents=True, exist_ok=False)
        init_vector = np.asarray(initialization_solution.vector, dtype=float).reshape(-1)
        witness_path = output_directory / "nominal-witness.npz"
        np.savez_compressed(witness_path, vector=init_vector,
            metadata__json=np.asarray(json.dumps({"completed_cycles": completed, "target_work_j": nominal_work,
                "source_prepared_problem_sha256": source.prepared_problem_sha256}, sort_keys=True)))
        report = {"schema_version": 1, "kind": "frozen_nominal_work_recovery_from_deficit_candidate",
            "source_receipt": str(receipt), "side": side, "completed_cycles": completed,
            "frozen_checkpoint": str(checkpoint), "frozen_checkpoint_sha256": _sha(checkpoint),
            "candidate": str(candidate), "candidate_sha256": _sha(candidate),
            "candidate_deficit": candidate_report.get("normalized_work_deficit"),
            "prepared_problem_sha256": source.prepared_problem_sha256,
            "fresh_worker_restore": restored, "nominal_work_j": nominal_work,
            "candidate_work_j": initial_work, "candidate_gap_closed_by_homotopy": homotopy_steps,
            "solver": {"name": "IPOPT-MA57", "hessian_approximation": hessian_approximation,
                "max_iterations": max_iterations, "tolerance": tolerance},
            "solver_build_seconds": 0., "initialization_nominal_trial": initialization_trial,
            "attempts": [], "nominal_task_witnessed": True, "physiological_failure_certified": False,
            "witness_path": str(witness_path), "witness_sha256": _sha(witness_path),
            "interpretation": "An independently audited nominal-work witness exists at the frozen checkpoint."}
        destination = output_directory / "recovery-report.json"
        _dump(destination, report)
        return destination
    if program.ocp_solver is None:
        raise RuntimeError("Restored RHO did not expose its complete NLP graph")
    graph, base_limits = program.ocp_solver.nlp, program.ocp_solver.limits
    n_decision = int(graph["x"].numel())
    if seed.size != n_decision:
        raise ValueError(f"Candidate dimension {seed.size} != frozen RHO dimension {n_decision}")

    nominal_work = float(json.loads(source.task_context_json)["nominal_work_j"])
    model_data = json.loads(Path(source.model_path).read_text(encoding="utf-8"))
    muscle_names = tuple(model_data["muscles"])
    slice_ = ConditionalCapacitySlice.from_checkpoint(source, muscle_order=muscle_names, trust_radius=.05)
    deficit_problem = FrozenTaskDeficit(graph=graph, limits=base_limits,
        **mapping_from_program(program, slice_), nominal_work_j=nominal_work, center=slice_.center)
    initial_work = float(seed[deficit_problem.work_index] * deficit_problem.work_coefficient)
    if not 0. <= initial_work < nominal_work:
        raise ValueError("Deficit seed must have sub-nominal terminal work")

    args.ipopt_hessian_approximation = hessian_approximation
    options = driver.configure_ipopt_solver(max_iterations=max_iterations, linear_solver="ma57",
        tolerance=float(tolerance), hsl_library=args.ipopt_hsl_library, print_level=0,
        advanced_options=driver._ipopt_advanced_options(args)).as_dict(program.ocp_solver)
    solver_build_start = perf_counter()
    solver = ca.nlpsol("frozen_nominal_work_recovery", "ipopt", graph, options)
    build_seconds = perf_counter() - solver_build_start

    limits = {key: np.asarray(value).copy() if hasattr(value, "shape") else value
              for key, value in base_limits.items()}
    records = []
    warm = seed.copy()
    targets = [nominal_work]
    phase = "direct_nominal_warm_start"
    for index, target_work in enumerate(targets):
        limits["x0"] = warm.copy()
        if target_work == nominal_work:
            # Keep the true source task bound untouched for the direct attempt.
            limits["lbx"] = np.asarray(base_limits["lbx"], dtype=float).copy()
            limits["ubx"] = np.asarray(base_limits["ubx"], dtype=float).copy()
        start = perf_counter()
        try:
            result = solver(**limits)
            vector = np.asarray(result["x"], dtype=float).reshape(-1)
            stats = solver.stats()
            solution = Solution.from_vector(program, vector[:, None])
            audit = independent_full_nlp_audit(solution, program, target_work_j=target_work, tolerance=tolerance)
            row = {"phase": phase, "target_work_j": target_work,
                "solver_status": stats.get("return_status"), "solver_success": bool(stats.get("success")),
                "iterations": stats.get("iter_count"), "solve_seconds": perf_counter() - start,
                "audit_passed": bool(audit.passed), "maximum_normalized_violation": audit.maximum_normalized_violation,
                "audit": dict(audit.detail)}
            records.append(row)
            if audit.passed:
                final = vector
                break
            # No failed optimizer iterate is promoted to a witness. Retain the
            # independently feasible deficit candidate for the work homotopy.
            warm = seed.copy()
        except Exception as error:
            records.append({"phase": phase, "target_work_j": target_work,
                "solver_exception": f"{type(error).__name__}: {error}", "audit_passed": False,
                "solve_seconds": perf_counter() - start})
            warm = seed.copy()
        else:
            final = None

    nominal_found = bool(records and records[-1].get("audit_passed"))
    pw_multistart = []
    if not nominal_found:
        if "Biceps" not in muscle_names or "Delt_ant" not in muscle_names:
            raise ValueError("Predeclared c169 multistart requires Biceps and Delt_ant PW controls")
        direction = np.zeros(len(muscle_names), dtype=float)
        direction[muscle_names.index("Biceps")] = 1. / np.sqrt(2.)
        direction[muscle_names.index("Delt_ant")] = -1. / np.sqrt(2.)
        for sign, suffix in ((1., "shoulder_plus"), (-1., "shoulder_minus")):
            shifted_seed = _shift_pw_seed(program, graph, seed, base_limits, sign * direction,
                                          fraction=.05, muscle_names=muscle_names)
            shifted_limits = {key: np.asarray(value).copy() if hasattr(value, "shape") else value
                              for key, value in base_limits.items()}
            shifted_limits["x0"] = shifted_seed
            start = perf_counter()
            try:
                result = solver(**shifted_limits)
                vector = np.asarray(result["x"], dtype=float).reshape(-1)
                stats = solver.stats()
                solution = Solution.from_vector(program, vector[:, None])
                audit = independent_full_nlp_audit(solution, program, target_work_j=nominal_work,
                                                    tolerance=tolerance)
                row = {"phase": f"pw_multistart_{suffix}", "seed": "deficit_candidate_with_5pct_bound_range_pw_shift",
                    "solver_status": stats.get("return_status"), "solver_success": bool(stats.get("success")),
                    "iterations": stats.get("iter_count"), "solve_seconds": perf_counter() - start,
                    "audit_passed": bool(audit.passed), "maximum_normalized_violation": audit.maximum_normalized_violation,
                    "audit": dict(audit.detail)}
                records.append(row)
                pw_multistart.append(row)
                if audit.passed:
                    nominal_found, final = True, vector
                    break
            except Exception as error:
                row = {"phase": f"pw_multistart_{suffix}", "seed": "deficit_candidate_with_5pct_bound_range_pw_shift",
                    "solver_exception": f"{type(error).__name__}: {error}", "audit_passed": False,
                    "solve_seconds": perf_counter() - start}
                records.append(row)
                pw_multistart.append(row)
    if not nominal_found and homotopy_steps > 0:
        # Keep all non-work bounds and constraints at their exact source values;
        # relax only the terminal-work variable, then close the observed gap.
        for step in range(0, homotopy_steps + 1):
            fraction = step / homotopy_steps
            target_work = initial_work + fraction * (nominal_work - initial_work)
            limits = {key: np.asarray(value).copy() if hasattr(value, "shape") else value
                      for key, value in base_limits.items()}
            limits["x0"] = warm.copy()
            target_scaled = target_work / deficit_problem.work_coefficient
            limits["lbx"][deficit_problem.work_index] = target_scaled
            limits["ubx"][deficit_problem.work_index] = target_scaled
            start = perf_counter()
            phase = f"work_homotopy_{step}_of_{homotopy_steps}"
            try:
                result = solver(**limits)
                vector = np.asarray(result["x"], dtype=float).reshape(-1)
                stats = solver.stats()
                # Intermediate homotopy targets differ from the nominal task
                # equality stored in the restored Bioptim object. Audit these
                # stages against the exact frozen path constraints plus the
                # explicit intermediate-work equality. The final stage uses
                # the normal independent production-NLP audit.
                if step == homotopy_steps:
                    solution = Solution.from_vector(program, vector[:, None])
                    audit = independent_full_nlp_audit(solution, program,
                        target_work_j=nominal_work, tolerance=tolerance)
                    stage_audit = {"audit_passed": bool(audit.passed),
                        "maximum_normalized_violation": audit.maximum_normalized_violation,
                        "audit": dict(audit.detail)}
                else:
                    stage_audit = deficit_problem.audit(vector, slice_.center, tolerance=tolerance)
                    target_error = abs(vector[deficit_problem.work_index] * deficit_problem.work_coefficient
                                        - target_work) / nominal_work
                    stage_audit = {**stage_audit, "target_work_normalized_error": target_error,
                        "audit_passed": bool(stage_audit["phase_one_trajectory_validated"]
                                              and target_error <= tolerance),
                        "maximum_normalized_violation": max(
                            float(stage_audit["constraint_normalized_violation"] or 0.), target_error)}
                row = {"phase": phase, "target_work_j": target_work,
                    "solver_status": stats.get("return_status"), "solver_success": bool(stats.get("success")),
                    "iterations": stats.get("iter_count"), "solve_seconds": perf_counter() - start,
                    **stage_audit}
                records.append(row)
                if stage_audit["audit_passed"]:
                    warm = vector.copy()
                    if step == homotopy_steps:
                        nominal_found, final = True, vector
                        break
                else:
                    warm = seed.copy()
                    break
            except Exception as error:
                records.append({"phase": phase, "target_work_j": target_work,
                    "solver_exception": f"{type(error).__name__}: {error}", "audit_passed": False,
                    "solve_seconds": perf_counter() - start})
                break

    feasibility_restoration = None
    if not nominal_found:
        # Change only the objective, not the physical NLP: minimize distance
        # to the valid deficit candidate while enforcing the full nominal
        # work equality and all original RHO constraints exactly.
        lower = np.asarray(base_limits["lbx"], dtype=float).reshape(-1)
        upper = np.asarray(base_limits["ubx"], dtype=float).reshape(-1)
        finite_span = np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
        scale = np.ones_like(seed)
        scale[finite_span] = upper[finite_span] - lower[finite_span]
        x = graph["x"]
        proximity_objective = ca.sumsqr((x - ca.DM(seed)) / ca.DM(scale)) / float(n_decision)
        restoration_graph = {"x": x, "f": proximity_objective, "g": graph["g"]}
        restoration_build_start = perf_counter()
        restoration_solver = ca.nlpsol("frozen_nominal_feasibility_restoration", "ipopt",
                                       restoration_graph, options)
        restoration_build_seconds = perf_counter() - restoration_build_start
        restoration_limits = {key: np.asarray(value).copy() if hasattr(value, "shape") else value
                              for key, value in base_limits.items()}
        restoration_limits["x0"] = seed.copy()
        restoration_start = perf_counter()
        try:
            result = restoration_solver(**restoration_limits)
            vector = np.asarray(result["x"], dtype=float).reshape(-1)
            stats = restoration_solver.stats()
            solution = Solution.from_vector(program, vector[:, None])
            audit = independent_full_nlp_audit(solution, program, target_work_j=nominal_work, tolerance=tolerance)
            feasibility_restoration = {"solver_status": stats.get("return_status"),
                "solver_success": bool(stats.get("success")), "iterations": stats.get("iter_count"),
                "solver_build_seconds": restoration_build_seconds,
                "solve_seconds": perf_counter() - restoration_start,
                "audit_passed": bool(audit.passed), "maximum_normalized_violation": audit.maximum_normalized_violation,
                "audit": dict(audit.detail),
                "objective": "normalized squared distance from deficit candidate",
                "physical_constraints_and_nominal_work_unchanged": True}
            if audit.passed:
                nominal_found, final = True, vector
        except Exception as error:
            feasibility_restoration = {"solver_exception": f"{type(error).__name__}: {error}",
                "solver_build_seconds": restoration_build_seconds,
                "solve_seconds": perf_counter() - restoration_start,
                "audit_passed": False,
                "objective": "normalized squared distance from deficit candidate",
                "physical_constraints_and_nominal_work_unchanged": True}

    output_directory.mkdir(parents=True, exist_ok=False)
    if nominal_found:
        np.savez_compressed(output_directory / "nominal-witness.npz", vector=final,
            metadata__json=np.asarray(json.dumps({"completed_cycles": completed, "target_work_j": nominal_work,
                "source_prepared_problem_sha256": source.prepared_problem_sha256}, sort_keys=True)))
    report = {"schema_version": 1, "kind": "frozen_nominal_work_recovery_from_deficit_candidate",
        "source_receipt": str(receipt), "side": side, "completed_cycles": completed,
        "frozen_checkpoint": str(checkpoint), "frozen_checkpoint_sha256": _sha(checkpoint),
        "candidate": str(candidate), "candidate_sha256": _sha(candidate),
        "candidate_deficit": candidate_report.get("normalized_work_deficit"),
        "prepared_problem_sha256": source.prepared_problem_sha256,
        "fresh_worker_restore": restored, "nominal_work_j": nominal_work,
        "candidate_work_j": initial_work, "candidate_gap_closed_by_homotopy": homotopy_steps,
        "solver": {"name": "IPOPT-MA57", "hessian_approximation": hessian_approximation,
            "max_iterations": max_iterations, "tolerance": tolerance},
        "solver_build_seconds": build_seconds, "initialization_nominal_trial": initialization_trial,
        "attempts": records, "pw_multistart": pw_multistart,
        "feasibility_restoration": feasibility_restoration,
        "nominal_task_witnessed": nominal_found, "physiological_failure_certified": False,
        "witness_path": str(output_directory / "nominal-witness.npz") if nominal_found else None,
        "interpretation": ("An independently audited nominal-work witness exists at the frozen checkpoint."
            if nominal_found else "No nominal witness was recovered; this is inconclusive and is not a physiological-failure certificate.")}
    destination = output_directory / "recovery-report.json"
    _dump(destination, report)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--pw-tracking-scale-s", type=float, default=1e-5)
    parser.add_argument("--max-iterations", type=int, default=4000)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--homotopy-steps", type=int, default=4)
    parser.add_argument("--hessian-approximation", choices=("exact", "limited-memory"), default="exact")
    print(run(**vars(parser.parse_args(argv))))


if __name__ == "__main__":
    main()
