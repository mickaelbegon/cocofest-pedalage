#!/usr/bin/env python3
"""Predeclared reachable interventions followed by a fixed-policy continuation.

The intervention and its continuation execute in distinct Python processes.
Only a restored, independently audited endpoint can supply a continuation label.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cocofest.optimization.task_reserve_branch import _unchanged_numerical_problem, json_digest
from cocofest.optimization.task_reserve_probe_adapter import independent_full_nlp_audit
from cocofest.simulation.independent_arms_process import _atomic_json, _driver_arguments, _FrozenRhoRetry
from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint, prepared_problem_arrays, restore_prepared_checkpoint
from scripts.probe_independent_rho_task_reserve import _load_arm
from scripts.run_checkpoint_fatigue_ablation import build_runtime
from scripts.run_local_task_reserve_branch import _save_witness, _solve

CAMPAIGN = ROOT / "asymmetric-sides-r192-rho-bo-20260928/unilateral-continuation-labels-20261003"
NAMES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")


def action_design():
    """Orthonormal Helmert contrasts; diagonal actions are untouched holdouts."""
    basis = np.asarray([[1., -1., 0., 0.], [1., 1., -2., 0.], [1., 1., 1., -3.]])
    basis /= np.linalg.norm(basis, axis=1)[:, None]
    directions = [("unit", "control", np.zeros(4))]
    for i, vector in enumerate(basis, 1):
        for sign, label in ((1., "plus"), (-1., "minus")):
            directions.append((f"contrast-{i}-{label}", "train", sign * vector))
    mixed = basis.sum(axis=0) / np.sqrt(3.)
    directions += [("mixed-plus", "holdout", mixed), ("mixed-minus", "holdout", -mixed)]
    actions = []
    for name, partition, direction in directions:
        weights = np.exp(.35 * direction)
        weights /= weights.mean()
        actions.append({"id": name, "partition": partition, "action_group": name,
                        "centered_log_direction": direction.tolist(), "amplitude": .35,
                        "weights": dict(zip(NAMES, weights.tolist()))})
    return actions


def prepare(output=CAMPAIGN):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    cpus = [12, 13, 14, 15]
    if not set(cpus) <= os.sched_getaffinity(0):
        raise RuntimeError("Dedicated CPU set unavailable")
    anchors = []
    for cycle in (120, 140):
        receipt = ROOT / "asymmetric-sides-r192-rho-bo-20260928/factorial-weights-split-20261003/unit_fixed/results/checkpoints" / f"cycle-{cycle}/receipt.json"
        source, payload = _load_arm(receipt, "left")
        source.verify_files()
        args = _driver_arguments(payload, "left")
        if (args.collocation_degree != 5 or args.stimulations_per_cycle != 30
                or args.ipopt_linear_solver != "ma57" or payload["resistance_pace"]["capacity_feedback"]
                or payload["muscle_pace"]["adaptation_enabled"]
                or not math.isclose(payload["left_equivalent_mean_torque_nm"], .96)):
            raise ValueError("Reference task mismatch")
        anchors.append({"cycle": cycle, "receipt": str(receipt),
                        "receipt_sha256": sha256(receipt.read_bytes()).hexdigest(),
                        "archive_sha256": sha256(Path(source.archive_path).read_bytes()).hexdigest(),
                        "prepared_problem_sha256": source.prepared_problem_sha256})
    policy = {"kind": "unit_fatigue_integral_quadratic", "muscle_order": list(NAMES),
              "weights": dict.fromkeys(NAMES, 1.), "side": "left", "equivalent_mean_torque_nm": .96,
              "configuration_source": "authenticated anchor payload unchanged except process affinity and output",
              "maximum_absolute_cycles": 240, "recovery": "same source _FrozenRhoRetry and counterfactual probes"}
    jobs = []
    for anchor in anchors:
        for action in action_design():
            for duration in (1, 3):
                name = f"left-c{anchor['cycle']}-{action['id']}-d{duration}"
                jobs.append({"id": name, "anchor": anchor, "action": action,
                             "intervention_cycles": duration, "output": str(output / name)})
    manifest = {"schema_version": 1, "kind": "unilateral_continuation_label_protocol", "side": "left",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "runner_path": str(Path(__file__).resolve()), "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "cpu_ids": cpus, "maximum_parallel_jobs": len(cpus), "numeric_threads": 1,
                "cpu_preflight": "Host FHO PID 4004377 uses 0-11; preserved. Reserve 12-15; leave siblings 28-31 unused.",
                "anchors": anchors, "muscle_order": list(NAMES), "actions": action_design(), "jobs": jobs,
                "return_policy": policy, "return_policy_sha256": json_digest(policy),
                "audit_tolerance": 1e-5, "control_gate": {"expected_last_certified_cycle": 169, "allowed_difference_cycles": 0},
                "grouping": "Both durations and anchors of one action remain in the same train/holdout partition.",
                "classification": {"physiological_failure_certified": "No zero-objective feasible witness, explicit local infeasible status, and feasible frozen fatigue-rest counterfactual. Protocol-conditioned endpoint, not proof of global infeasibility.",
                                   "right_censored": "All cycles certified through absolute cycle budget.",
                                   "numerical_stop": "Any other unsuccessful or unverified continuation."},
                "activation_allowed": False}
    output.mkdir(parents=True)
    _atomic_json(output / "manifest.json", manifest)
    return output / "manifest.json"


def authenticated_job(manifest_path, job_id):
    manifest = json.loads(Path(manifest_path).read_text())
    if sha256(Path(manifest["runner_path"]).read_bytes()).hexdigest() != manifest["runner_sha256"]:
        raise ValueError("Runner changed after preregistration")
    job, = [j for j in manifest["jobs"] if j["id"] == job_id]
    receipt = Path(job["anchor"]["receipt"])
    if sha256(receipt.read_bytes()).hexdigest() != job["anchor"]["receipt_sha256"]:
        raise ValueError("Anchor receipt changed")
    source, payload = _load_arm(receipt, "left")
    source.verify_files()
    if json_digest(manifest["return_policy"]) != manifest["return_policy_sha256"]:
        raise ValueError("Return policy changed")
    return manifest, job, source, payload


def classify_terminal(last, *, censored):
    if censored:
        return "right_censored", "absolute_cycle_budget"
    zero = last.get("zero_objective_feasibility_probe") or {}
    fatigue = (last.get("counterfactual_probes") or {}).get("fatigue_rest") or {}
    native = (zero.get("solver_stats") or {}).get("return_status")
    if (zero.get("available") and zero.get("constraints_unchanged")
            and zero.get("feasible_witness") is False and native == "Infeasible_Problem_Detected"
            and fatigue.get("feasible_witness") is True and fatigue.get("physical_rho_advanced") is False):
        return "physiological_failure_certified", "local_frozen_infeasibility_with_fatigue_rest_witness"
    return "numerical_stop", "continuation_endpoint_not_attributed_to_fatigue"


def run_phase(manifest_path, job_id, phase, cpu):
    from bioptim import SolutionMerge
    from cocofest.optimization.independent_arm_backends import set_terminal_eprod_target
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver
    manifest, job, source, payload = authenticated_job(manifest_path, job_id)
    if cpu not in manifest["cpu_ids"] or cpu not in os.sched_getaffinity(0):
        raise ValueError("Unreserved CPU")
    os.sched_setaffinity(0, {cpu})
    directory = Path(job["output"])
    output = directory / phase
    output.mkdir(parents=True, exist_ok=False)
    args = _driver_arguments(payload, "left")
    runtime, _ = build_runtime(payload, "left", "integral_quadratic")
    program, solver = runtime["nmpc"], runtime["solver"]
    archive, completed = Path(source.archive_path), source.completed_cycles
    expected_digest = source.prepared_problem_sha256
    intervention = None
    if phase == "continuation":
        intervention_file = directory / "intervention/result.json"
        intervention = json.loads(intervention_file.read_text())
        if not intervention["success"] or intervention["manifest_sha256"] != sha256(Path(manifest_path).read_bytes()).hexdigest():
            raise ValueError("Intervention endpoint is not certified")
        for row in intervention["cycles"]:
            if not row["certified"] or sha256(Path(row["witness"]).read_bytes()).hexdigest() != row["witness_sha256"]:
                raise ValueError("Intervention trajectory changed")
        endpoint = intervention["prepared_endpoint"]
        archive, completed = Path(endpoint["primal_path"]), endpoint["completed_cycles"]
        if sha256(archive.read_bytes()).hexdigest() != endpoint["sha256"]:
            raise ValueError("Endpoint archive changed")
        expected_digest = endpoint["prepared_problem_sha256"]
    restored = restore_prepared_checkpoint(archive, program, completed_cycles=completed)
    if restored["restored_problem_sha256"] != expected_digest or not restored["stimulation_history_complete"]:
        raise ValueError("Exact restoration failed")
    nlp = program.nlp[0]
    if tuple(m.muscle_name for m in nlp.model.muscles_dynamics_model) != NAMES:
        raise ValueError("Muscle order mismatch")
    work = float(nlp.x_bounds["E_prod"].min[0, 2])
    if not math.isclose(work, 2 * math.pi * .96, abs_tol=1e-12):
        raise ValueError("Task work changed")
    weights = [job["action"]["weights"][name] if phase == "intervention" else 1. for name in NAMES]
    before = prepared_problem_arrays(program)
    program.fatigue_weight_binding.update(program, weights)
    _unchanged_numerical_problem(before, prepared_problem_arrays(program))
    receipt = {"schema_version": 1, "kind": "unilateral_continuation_phase", "phase": phase, "job_id": job_id,
               "manifest": str(Path(manifest_path).resolve()), "manifest_sha256": sha256(Path(manifest_path).read_bytes()).hexdigest(),
               "source_receipt": str(Path(job["anchor"]["receipt"])), "source_receipt_sha256": job["anchor"]["receipt_sha256"],
               "source_archive_sha256": sha256(archive.read_bytes()).hexdigest(), "source_restore": restored,
               "source_completed_cycles": completed, "anchor_completed_cycles": source.completed_cycles,
               "producer_pid": os.getpid(), "cpu": cpu, "effective_affinity": sorted(os.sched_getaffinity(0)),
               "action": job["action"], "intervention_cycles": job["intervention_cycles"],
               "nominal_work_j": work, "muscle_order": list(NAMES), "effective_weights": weights,
               "return_policy": manifest["return_policy"], "return_policy_sha256": manifest["return_policy_sha256"],
               "effective_driver_arguments": json.loads(json.dumps(vars(args), default=str)),
               "audit_tolerance": manifest["audit_tolerance"], "exact_bilateral_restart": False,
               "maximum_absolute_cycles": manifest["return_policy"]["maximum_absolute_cycles"], "activation_allowed": False}
    if intervention:
        receipt.update(endpoint_receipt=str(intervention_file), endpoint_receipt_sha256=sha256(intervention_file.read_bytes()).hexdigest(),
                       endpoint=intervention["prepared_endpoint"], fresh_worker_replay_verified=intervention["producer_pid"] != os.getpid())
        if not receipt["fresh_worker_replay_verified"]:
            raise ValueError("Continuation must execute in a fresh worker")
    _atomic_json(output / "protocol.json", receipt)
    retry = _FrozenRhoRetry(program, driver, tolerance=driver._window_feasibility_tolerance(args),
                            max_attempts=args.max_consecutive_failing, recovery_args=args)
    retry.completed = completed
    stop = completed + job["intervention_cycles"] if phase == "intervention" else receipt["maximum_absolute_cycles"]
    rows, attempts = [], []
    terminal = None
    while retry.completed < stop:
        retry.capture()
        while True:
            program.total_optimization_run = retry.completed - source.completed_cycles
            start = time.perf_counter()
            solution = _solve(program, solver)
            elapsed = time.perf_counter() - start
            audit = independent_full_nlp_audit(solution, program, target_work_j=work, tolerance=manifest["audit_tolerance"])
            feasibility = driver._solution_feasibility_summary(solution, driver._window_feasibility_tolerance(args))
            native_certified = driver._rho_solution_is_certified(solution.status, feasibility)
            states = solution.decision_states(to_merge=SolutionMerge.NODES)
            row = {"cycle": retry.completed + 1, "offset": retry.completed + 1 - source.completed_cycles,
                   "certified": bool(audit.passed and native_certified), "solver_status": int(solution.status),
                   "native_solver_status": driver.snapshot_nlp_solver_stats(program).get("return_status"),
                   "independent_audit": dict(audit.detail), "maximum_normalized_violation": audit.maximum_normalized_violation,
                   "feasibility": feasibility, "solve_wall_seconds": elapsed,
                   "attempt": retry.failed_attempts + 1, "weights": weights,
                   "terminal_slow_states": {key: float(value[0, -1]) for key, value in states.items() if key.startswith(("A_", "Tau1_", "Km_"))},
                   "terminal_normalized_A": {m.muscle_name: float(states[f"A_{m.muscle_name}"][0, -1] / m.a_scale) for m in nlp.model.muscles_dynamics_model}}
            if native_certified and not audit.passed:
                terminal = {**row, "failure_reason": "independent_audit_disagrees_with_native_certification"}
                break
            if row["certified"]:
                row["witness"] = _save_witness(solution, program, output, row["cycle"], audit)
                row["witness_sha256"] = sha256(Path(row["witness"]).read_bytes()).hexdigest()
            retry._advance(program, solution, n_cycles_simultaneous=program.n_cycles_simultaneous)
            if row["certified"]:
                if not solution._cocofest_advanced_physical_rho:
                    raise RuntimeError("Certified solve was not advanced")
                set_terminal_eprod_target(program, work)
                program.all_models.clear()
                rows.append(row)
                print(f"{job_id} {phase} c{row['cycle']}: certified=True solve={elapsed:.3f}s", flush=True)
                _atomic_json(output / "progress.json", {**receipt, "cycles": rows, "attempts": attempts,
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
        source.verify_files()
    receipt.update(cycles=rows, attempts=attempts, terminal_attempt=terminal,
                   last_certified_cycle=retry.completed, certified_cycles=len(rows), completed=True)
    if phase == "intervention":
        receipt["success"] = retry.completed == stop
        if receipt["success"]:
            receipt["prepared_endpoint"] = export_prepared_checkpoint(output / "endpoint.npz", program,
                completed_cycles=retry.completed, model_path=Path(source.model_path))
            receipt["endpoint_terminal_slow_states"] = rows[-1]["terminal_slow_states"]
    else:
        status, reason = classify_terminal(terminal or {}, censored=retry.completed == stop)
        receipt.update(status=status, status_reason=reason, success=status == "physiological_failure_certified",
                       continuation_cycles=retry.completed - completed,
                       total_cycles_after_anchor=retry.completed - source.completed_cycles,
                       exact_label_allowed=status == "physiological_failure_certified",
                       label_scope="Local numerical frozen-feasibility/counterfactual protocol; no global infeasibility proof")
    _atomic_json(output / "result.json", receipt)
    return output / "result.json"


def campaign(manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1", "MPLCONFIGDIR": "/tmp/cocofest-continuation-labels-mpl"}
    def execute(job, cpu):
        target = Path(job["output"])
        target.mkdir(exist_ok=False)
        launch = {"job_id": job["id"], "cpu": cpu, "reserved_affinity": [cpu], "numeric_threads": 1, "phases": []}
        for phase in ("intervention", "continuation"):
            command = [sys.executable, str(Path(__file__).resolve()), "worker", "--manifest", str(manifest_path),
                       "--job-id", job["id"], "--phase", phase, "--cpu", str(cpu)]
            with (target / f"{phase}.log").open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                launch["phases"].append({"phase": phase, "pid": child.pid, "command": command})
                _atomic_json(target / "launch.json", launch)
                try:
                    code = child.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=30)
                    code = -15
                launch["phases"][-1]["returncode"] = code
            result = target / phase / "result.json"
            if code or not result.exists() or (phase == "intervention" and not json.loads(result.read_text())["success"]):
                _atomic_json(target / "job-result.json", {**launch, "status": "numerical_stop", "phase_failed": phase, "exact_label_allowed": False})
                return {"job_id": job["id"], "status": "numerical_stop"}
        final = json.loads(result.read_text())
        summary = {**launch, "status": final["status"], "last_certified_cycle": final["last_certified_cycle"],
                   "continuation_cycles": final["continuation_cycles"], "result": str(result),
                   "result_sha256": sha256(result.read_bytes()).hexdigest(), "exact_label_allowed": final["exact_label_allowed"]}
        _atomic_json(target / "job-result.json", summary)
        return summary
    # Four matched controls cover both anchors and intervention durations.
    controls = [j for j in manifest["jobs"] if j["action"]["partition"] == "control"]
    with ThreadPoolExecutor(max_workers=len(manifest["cpu_ids"])) as pool:
        gates = list(pool.map(lambda pair: execute(*pair), zip(controls, manifest["cpu_ids"])))
    gate_passed = all(r.get("last_certified_cycle") == manifest["control_gate"]["expected_last_certified_cycle"]
                      and r.get("status") == "physiological_failure_certified" for r in gates)
    _atomic_json(base / "control-gate.json", {"passed": gate_passed, "controls": gates,
        "criterion": manifest["control_gate"]})
    if not gate_passed:
        _atomic_json(base / "campaign-completion.json", {"completed": False, "halted": "control_replay_gate_failed", "jobs": gates})
        return
    remaining = [j for j in manifest["jobs"] if j["action"]["partition"] != "control"]
    def lane(index):
        results = []
        for job in remaining[index::len(manifest["cpu_ids"])]:
            results.append(execute(job, manifest["cpu_ids"][index]))
            _atomic_json(base / f"lane-{index}.json", {"jobs": results})
        return results
    with ThreadPoolExecutor(max_workers=len(manifest["cpu_ids"])) as pool:
        groups = list(pool.map(lane, range(len(manifest["cpu_ids"]))))
    _atomic_json(base / "campaign-completion.json", {"completed": True, "jobs": gates + [r for group in groups for r in group]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "campaign", "worker"))
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN / "manifest.json")
    parser.add_argument("--job-id")
    parser.add_argument("--phase", choices=("intervention", "continuation"))
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(prepare(args.manifest.parent))
    elif args.mode == "campaign":
        campaign(args.manifest)
    else:
        try:
            print(run_phase(args.manifest, args.job_id, args.phase, args.cpu), flush=True)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
