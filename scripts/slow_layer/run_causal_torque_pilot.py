#!/usr/bin/env python3
"""Preregister and run audited paired bilateral torque-transfer experiments.

Each independent arm is solved in a fresh process per phase. Only the shared
certified prefix is a bilateral trajectory; later unilateral rows are explicitly
counterfactual. This is a simulation protocol, not a clinical comfort envelope.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
NAMES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
DEFAULT_CHECKPOINTS = ROOT / "asymmetric-sides-r192-rho-bo-20260928/factorial-weights-split-20261003/unit_fixed/results/checkpoints"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    from cocofest.simulation.independent_arms_process import _atomic_json
    _atomic_json(Path(path), value)


def action_schedule(delta, phase):
    if delta not in (-.05, 0., .05) or phase not in ("intervention", "continuation"):
        raise ValueError("Undeclared action or phase")
    effective = delta if phase == "intervention" else 0.
    return {"left": .96 + effective, "right": .96 - effective}


def job_action_schedule(job, phase):
    """Return a preregistered bilateral schedule for either pilot schema.

    ``action_schedule`` remains deliberately narrow because it documents the
    original +/- 0.05 Nm, ten-cycle pilot.  The dose--duration protocol uses
    the same audited worker but declares its larger action set in its sealed
    manifest.  Keeping the generic validation here prevents a worker from
    silently accepting an arbitrary torque transfer supplied at runtime.
    """
    if phase not in ("intervention", "continuation"):
        raise ValueError("Unknown phase")
    delta = job.get("delta_left_nm")
    duration = job.get("intervention_cycles")
    if (isinstance(delta, bool) or not isinstance(delta, (int, float))
            or not math.isfinite(delta) or isinstance(duration, bool)
            or not isinstance(duration, int) or duration <= 0):
        raise ValueError("Malformed sealed intervention")
    baseline_left = job.get("baseline_left_nm", .96)
    baseline_right = job.get("baseline_right_nm", .96)
    if (isinstance(baseline_left, bool) or not isinstance(baseline_left, (int, float))
            or isinstance(baseline_right, bool) or not isinstance(baseline_right, (int, float))
            or not math.isfinite(baseline_left) or not math.isfinite(baseline_right)
            or baseline_left <= 0 or baseline_right <= 0):
        raise ValueError("Malformed sealed baseline torque")
    if phase == "continuation":
        return {"left": float(baseline_left), "right": float(baseline_right)}
    left, right = float(baseline_left) + float(delta), float(baseline_right) - float(delta)
    if min(left, right) <= 0 or not math.isclose(left + right, baseline_left + baseline_right, abs_tol=1e-12):
        raise ValueError("Intervention does not preserve the sealed positive total torque")
    return {"left": left, "right": right}


def aggregate_arms(arms):
    """A failure is attributable only if both arms certify the preceding prefix."""
    if set(arms) != {"left", "right"} or any("last_certified_cycle" not in r for r in arms.values()):
        return {"status": "numerical_stop", "exact_label_allowed": False}
    common = min(r["last_certified_cycle"] for r in arms.values())
    limiting = [side for side, row in arms.items() if row["last_certified_cycle"] == common]
    statuses = [arms[side]["status"] for side in limiting]
    if "physiological_failure_certified" in statuses:
        status = "physiological_failure_certified"
    elif all(row["status"] == "right_censored" for row in arms.values()):
        status = "right_censored"
    else:
        status = "numerical_stop"
    return {"status": status, "last_certified_cycle": common, "limiting_sides": limiting,
            "exact_label_allowed": status == "physiological_failure_certified",
            "unilateral_rows_after_common_prefix": "counterfactual, excluded from bilateral work/endurance"}


def prepare(output, receipts, maximum_cycles=240, control_repeats=2):
    from scripts.probe_independent_rho_task_reserve import _load_arm
    from cocofest.simulation.independent_arms_process import _driver_arguments
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if control_repeats < 2:
        raise ValueError("At least two repeated controls are required")
    anchors = []
    for receipt in receipts:
        receipt = Path(receipt).resolve(strict=True)
        record = {"receipt": str(receipt), "receipt_sha256": digest(receipt), "arms": {}}
        for side in ("left", "right"):
            source, payload = _load_arm(receipt, side)
            source.verify_files()
            args = _driver_arguments(payload, side)
            unit = payload["muscle_pace"].get(f"{side}_initial_weights")
            if (args.collocation_degree != 5 or args.stimulations_per_cycle != 30
                    or args.ipopt_linear_solver != "ma57" or payload["resistance_pace"]["capacity_feedback"]
                    or payload["muscle_pace"]["adaptation_enabled"] or unit != dict.fromkeys(NAMES, 1.)
                    or not math.isclose(payload[f"{side}_equivalent_mean_torque_nm"], .96)):
                raise ValueError("Anchor differs from declared fixed unit-weight baseline")
            record["cycle"] = source.completed_cycles
            record["arms"][side] = {"archive_sha256": source.archive_sha256,
                                    "prepared_problem_sha256": source.prepared_problem_sha256}
        if maximum_cycles <= record["cycle"] + 10:
            raise ValueError("Budget must exceed the ten-cycle intervention at every anchor")
        anchors.append(record)
    if not anchors or len({a["cycle"] for a in anchors}) != len(anchors):
        raise ValueError("Distinct certified checkpoint cycles required")
    jobs = []
    for anchor in anchors:
        actions = [(f"control-{i + 1}", 0.) for i in range(control_repeats)] + [("left-minus", -.05), ("left-plus", .05)]
        for name, delta in actions:
            jobs.append({"id": f"c{anchor['cycle']}-{name}", "anchor": anchor,
                         "delta_left_nm": delta, "is_control": delta == 0., "intervention_cycles": 10})
    sources = sorted(set(ROOT.glob("cocofest/**/*.py")) | {
        Path(__file__).resolve(), ROOT / "scripts/run_unilateral_continuation_labels.py",
        ROOT / "scripts/run_checkpoint_fatigue_ablation.py", ROOT / "scripts/run_local_task_reserve_branch.py",
        ROOT / "scripts/probe_independent_rho_task_reserve.py", ROOT / "scripts/validate_independent_rho_checkpoint.py",
        ROOT / "examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py"})
    plan = {"schema_version": 1, "kind": "paired_bilateral_causal_torque_pilot", "jobs": jobs,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_sha256": {str(p): digest(p) for p in sources}, "maximum_absolute_cycles": maximum_cycles,
            "numeric_threads": 1, "audit_tolerance": 1e-5, "weights": dict.fromkeys(NAMES, 1.),
            "intervention": "10 complete cycles at left=0.96+delta, right=0.96-delta; total 1.92 Nm",
            "continuation_policy": {"kind": "fixed_equal_split", "left_nm": .96, "right_nm": .96,
                "switch": "Immediately after ten certified intervention cycles; change <=0.05 Nm per arm",
                "update_period_cycles": 10, "maximum_asymmetry_nm": .3},
            "endpoint_protocol": "Inherited frozen retries, zero-objective feasibility and fatigue-rest counterfactual; local attribution only",
            "control_gate": "All repeated controls complete with attributable frozen endpoint; no historical endpoint target imposed",
            "resolution": "Per anchor, max(control endpoints)-min(control endpoints). Differences <= this spread are ties; repetitions are numerical reproducibility, not biological replicates.",
            "activation_allowed": False}
    output.mkdir(parents=True)
    path = output / "manifest.json"
    write_json(path, plan)
    write_json(output / "manifest-seal.json", {"manifest_sha256": digest(path)})
    return path


def authenticated(manifest_path, job_id=None):
    path = Path(manifest_path).resolve(strict=True)
    seal = json.loads((path.parent / "manifest-seal.json").read_text())
    if digest(path) != seal["manifest_sha256"]:
        raise ValueError("Manifest changed after preregistration")
    plan = json.loads(path.read_text())
    for filename, expected in plan["code_sha256"].items():
        if digest(filename) != expected:
            raise ValueError(f"Protocol code changed: {filename}")
    for job in plan["jobs"]:
        if digest(job["anchor"]["receipt"]) != job["anchor"]["receipt_sha256"]:
            raise ValueError("Anchor receipt changed")
    if job_id is None:
        return plan
    job, = [j for j in plan["jobs"] if j["id"] == job_id]
    return plan, job


def worker(manifest_path, job_id, side, phase, cpu):
    from bioptim import SolutionMerge
    from cocofest.optimization.independent_arm_backends import set_terminal_eprod_target
    from cocofest.optimization.task_reserve_probe_adapter import independent_full_nlp_audit
    from cocofest.simulation.independent_arms_process import _driver_arguments, _FrozenRhoRetry
    from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint, restore_prepared_checkpoint
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver
    from scripts.probe_independent_rho_task_reserve import _load_arm
    from scripts.run_checkpoint_fatigue_ablation import build_runtime
    from scripts.run_local_task_reserve_branch import _save_witness, _solve
    from scripts.run_unilateral_continuation_labels import classify_terminal
    plan, job = authenticated(manifest_path, job_id)
    if cpu not in os.sched_getaffinity(0):
        raise ValueError("CPU unavailable")
    os.sched_setaffinity(0, {cpu})
    source, payload = _load_arm(Path(job["anchor"]["receipt"]), side)
    source.verify_files()
    expected = job["anchor"]["arms"][side]
    if source.archive_sha256 != expected["archive_sha256"]:
        raise ValueError("Anchor archive changed")
    output = Path(manifest_path).parent / job_id / side / phase
    output.mkdir(parents=True, exist_ok=False)
    runtime, _ = build_runtime(payload, side, "integral_quadratic")
    program, solver = runtime["nmpc"], runtime["solver"]
    args = _driver_arguments(payload, side)
    archive, completed, prepared_digest = Path(source.archive_path), source.completed_cycles, source.prepared_problem_sha256
    predecessor = None
    if phase == "continuation":
        predecessor = output.parent / "intervention/result.json"
        previous = json.loads(predecessor.read_text())
        if not previous["phase_complete"] or previous["manifest_sha256"] != digest(manifest_path):
            raise ValueError("Intervention not certified against this manifest")
        if previous["producer_pid"] == os.getpid():
            raise ValueError("Fresh phase process required")
        for row in previous["cycles"]:
            if not row["certified"] or digest(row["witness"]) != row["witness_sha256"]:
                raise ValueError("Intervention witness changed")
        endpoint = previous["prepared_endpoint"]
        archive, completed = Path(endpoint["primal_path"]), endpoint["completed_cycles"]
        if digest(archive) != endpoint["sha256"]:
            raise ValueError("Intervention endpoint changed")
        prepared_digest = endpoint["prepared_problem_sha256"]
    restored = restore_prepared_checkpoint(archive, program, completed_cycles=completed)
    if restored["restored_problem_sha256"] != prepared_digest or not restored["stimulation_history_complete"]:
        raise ValueError("Exact phase restoration failed")
    torque = job_action_schedule(job, phase)[side]
    work = 2 * math.pi * torque
    program.fatigue_weight_binding.update(program, [1.] * len(NAMES))
    set_terminal_eprod_target(program, work)
    retry = _FrozenRhoRetry(program, driver, tolerance=driver._window_feasibility_tolerance(args),
                            max_attempts=args.max_consecutive_failing, recovery_args=args)
    retry.completed = completed
    stop = (source.completed_cycles + job["intervention_cycles"]
            if phase == "intervention" else plan["maximum_absolute_cycles"])
    receipt = {"job_id": job_id, "side": side, "phase": phase, "producer_pid": os.getpid(),
               "manifest_sha256": digest(manifest_path), "source_restore": restored, "cpu": cpu,
               "source_archive_sha256": digest(archive), "source_receipt_sha256": job["anchor"]["receipt_sha256"],
               "anchor_completed_cycles": source.completed_cycles, "phase_start_cycle": completed,
               "torque_nm": torque, "target_work_j": work, "weights": plan["weights"],
               "model_path": source.model_path, "model_sha256": source.model_sha256,
               "audit_tolerance": plan["audit_tolerance"],
               "label_scope": "Protocol-conditioned local frozen infeasibility with fatigue-rest witness; no global infeasibility proof",
               "intervention_cycles": job["intervention_cycles"],
               "continuation_policy": plan["continuation_policy"], "exact_bilateral_restart": False,
               "bilateral_scope": "paired independent arm schedule; use only shared certified prefix",
               "effective_recovery_arguments": json.loads(json.dumps(vars(args), default=str))}
    if predecessor:
        receipt.update(predecessor=str(predecessor), predecessor_sha256=digest(predecessor))
    write_json(output / "protocol.json", receipt)
    rows, attempts, terminal = [], [], None
    while retry.completed < stop and terminal is None:
        retry.capture()
        while True:
            program.total_optimization_run = retry.completed - source.completed_cycles
            start = time.perf_counter()
            solution = _solve(program, solver)
            audit = independent_full_nlp_audit(solution, program, target_work_j=work, tolerance=plan["audit_tolerance"])
            feasibility = driver._solution_feasibility_summary(solution, driver._window_feasibility_tolerance(args))
            native = driver._rho_solution_is_certified(solution.status, feasibility)
            row = {"cycle": retry.completed + 1, "certified": bool(native and audit.passed),
                   "solver_status": int(solution.status), "independent_audit": dict(audit.detail),
                   "native_solver_status": driver.snapshot_nlp_solver_stats(program).get("return_status"),
                   "maximum_normalized_violation": audit.maximum_normalized_violation,
                   "feasibility": feasibility, "solve_wall_seconds": time.perf_counter() - start,
                   "attempt": retry.failed_attempts + 1, "torque_nm": torque}
            if native and not audit.passed:
                terminal = {**row, "failure_reason": "independent_audit_disagrees"}
                break
            if row["certified"]:
                row["witness"] = _save_witness(solution, program, output, row["cycle"], audit)
                row["witness_sha256"] = digest(row["witness"])
                states = solution.decision_states(to_merge=SolutionMerge.NODES)
                row["target_work_j"] = work
                row["achieved_work_j"] = float(states["E_prod"][0, -1] - states["E_prod"][0, 0])
                row["terminal_slow_states"] = {k: float(v[0, -1]) for k, v in states.items() if k.startswith(("A_", "Tau1_", "Km_"))}
            retry._advance(program, solution, n_cycles_simultaneous=program.n_cycles_simultaneous)
            if row["certified"]:
                if not solution._cocofest_advanced_physical_rho:
                    raise RuntimeError("Certified solution did not advance")
                set_terminal_eprod_target(program, work)
                program.all_models.clear()
                rows.append(row)
                write_json(output / "progress.json", {**receipt, "cycles": rows, "last_certified_cycle": retry.completed})
                print(f"{job_id} {side} {phase} c{retry.completed} certified", flush=True)
                break
            for field in ("frozen_retry", "zero_objective_feasibility_probe", "counterfactual_probes"):
                value = getattr(solution, f"_cocofest_{field}", None)
                if value is not None:
                    row[field] = value
            attempts.append(row)
            if not program._cocofest_retry_same_rho_pending:
                terminal = row
                break
    complete = retry.completed == stop
    status, reason = classify_terminal(terminal or {}, censored=complete)
    if complete and phase == "intervention":
        status, reason = "intervention_complete", f"{job['intervention_cycles']}_audited_cycles"
        receipt["prepared_endpoint"] = export_prepared_checkpoint(output / "endpoint.npz", program,
                completed_cycles=retry.completed, model_path=Path(source.model_path))
    receipt.update(cycles=rows, attempts=attempts, terminal_attempt=terminal, phase_complete=complete,
                   last_certified_cycle=retry.completed, status=status, status_reason=reason,
                   exact_label_allowed=status == "physiological_failure_certified")
    source.verify_files()
    write_json(output / "result.json", receipt)


def campaign(manifest_path, cpus, timeout_seconds=7200, controls_only=False):
    manifest_path = Path(manifest_path).resolve(strict=True)
    plan = authenticated(manifest_path)
    if not cpus or len(set(cpus)) != len(cpus) or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("Supply distinct currently available dedicated CPU IDs")
    if timeout_seconds <= 0:
        raise ValueError("Positive worker timeout required")
    base = manifest_path.parent
    launch_path = base / "campaign-launch.json"
    if launch_path.exists():
        raise FileExistsError("Campaign already launched; use a new immutable plan")
    write_json(launch_path, {"cpus": cpus, "numeric_threads": 1, "timeout_seconds": timeout_seconds,
                           "controls_only": controls_only, "manifest_sha256": digest(manifest_path)})
    env = {**os.environ, **dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")}

    def execute(job, cpu):
        arms = {}
        for side in ("left", "right"):
            target = base / job["id"] / side
            target.mkdir(parents=True, exist_ok=False)
            for phase in ("intervention", "continuation"):
                command = [sys.executable, str(Path(__file__).resolve()), "worker", "--manifest", str(manifest_path),
                           "--job-id", job["id"], "--side", side, "--phase", phase, "--cpu", str(cpu)]
                with (target / f"{phase}.log").open("x") as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    write_json(target / f"{phase}-launch.json", {"command": command, "pid": child.pid, "cpu": cpu})
                    try:
                        code = child.wait(timeout=timeout_seconds)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        code = -15
                result = target / phase / "result.json"
                if code or not result.exists():
                    arms[side] = {"status": "numerical_stop", "exit_code": code, "failed_phase": phase}
                    break
                final = json.loads(result.read_text())
                arms[side] = {"status": final["status"], "last_certified_cycle": final["last_certified_cycle"],
                              "result": str(result), "result_sha256": digest(result)}
                if phase == "intervention" and not final["phase_complete"]:
                    break
        answer = {"job_id": job["id"], "anchor_cycle": job["anchor"]["cycle"],
                  "delta_left_nm": job["delta_left_nm"], "arms": arms, **aggregate_arms(arms)}
        write_json(base / job["id"] / "result.json", answer)
        return answer

    def batch(jobs):
        def lane(index):
            return [execute(job, cpus[index]) for job in jobs[index::len(cpus)]]
        with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
            return [row for lane_rows in pool.map(lane, range(len(cpus))) for row in lane_rows]

    controls = batch([j for j in plan["jobs"] if j["is_control"]])
    passed = all(row["status"] == "physiological_failure_certified" for row in controls)
    resolution = {}
    for cycle in sorted({row["anchor_cycle"] for row in controls}):
        rows = [r for r in controls if r["anchor_cycle"] == cycle]
        values = [r["last_certified_cycle"] for r in rows if r["status"] == "physiological_failure_certified"]
        resolution[str(cycle)] = {"endpoints": values, "spread_cycles": max(values) - min(values) if len(values) == len(rows) else None}
    write_json(base / "control-gate.json", {"passed": passed, "resolution": resolution, "controls": controls})
    results = controls
    if passed and not controls_only:
        results += batch([j for j in plan["jobs"] if not j["is_control"]])
    write_json(base / "campaign-completion.json", {"completed": passed and not controls_only,
               "halted": "controls_only_requested" if controls_only else None if passed else "control_replay_gate_failed",
               "jobs": results, "resolution": resolution})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "campaign", "worker"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipts", nargs="+", type=Path)
    parser.add_argument("--maximum-cycles", type=int, default=240)
    parser.add_argument("--control-repeats", type=int, default=2)
    parser.add_argument("--cpus", nargs="+", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--job-id")
    parser.add_argument("--side", choices=("left", "right"))
    parser.add_argument("--phase", choices=("intervention", "continuation"))
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.manifest.name != "manifest.json":
            parser.error("Manifest filename must be manifest.json")
        receipts = args.receipts or [DEFAULT_CHECKPOINTS / f"cycle-{c}/receipt.json" for c in (120, 140, 160)]
        print(prepare(args.manifest.parent, receipts, args.maximum_cycles, args.control_repeats))
    elif args.mode == "campaign":
        campaign(args.manifest, args.cpus, args.timeout_seconds, args.controls_only)
    else:
        if None in (args.job_id, args.side, args.phase, args.cpu):
            parser.error("Worker requires job ID, side, phase and CPU")
        worker(args.manifest, args.job_id, args.side, args.phase, args.cpu)


if __name__ == "__main__":
    main()
