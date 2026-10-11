#!/usr/bin/env python3
"""Run the causal G->D transfer policy through one uninterrupted RHO session.

This is deliberately different from the checkpoint pilots.  Two persistent,
compiled unilateral IPOPT/MA57 RHO workers start from the cold state and meet
at every certified cycle boundary.  The parent observes the returned Ding
state at that boundary, asks :class:`BicepsReservePolicy` for the *next*
cycle's total-preserving work allocation, and records the immutable decision
receipt before either worker begins that cycle.

The policy is one shot: it may relieve the left arm by 0.10 Nm for ten
complete cycles and then explicitly returns to 0.96/0.96 Nm.  There is no
cycle-number trigger.  The default is safe abstention because the fast/history
state gate has not yet been validated; ``--experimental-unvalidated-fast-state
-override`` exists only to reproduce the offline c140 pilot in a live run.
It never makes this protocol clinical or activation eligible.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.cycling.slow.causal_transfer import (
    BicepsReservePolicy,
    CausalTransferSupervisor,
    TransferConfig,
    TransferObservation,
)
from cocofest.simulation import independent_arms_process as process
from scripts.run_unilateral_continuation_labels import classify_terminal


DEFAULT_INPUT = (ROOT / "asymmetric-sides-r192-rho-bo-20260928/factorial-weights-split-20261003"
                 / "unit_fixed/input.json")
NAMES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")


def digest(path: Path | str) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _json_digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def live_biceps_observation(reports: Mapping[str, Mapping[str, object]], completed_cycles: int,
                            *, fast_state_gate: bool) -> TransferObservation:
    """Adapt two live, certified process-boundary reports for the policy.

    The digest binds the readable Biceps fractions to the complete compact
    worker reports, including solver/feasibility receipts and fast-history
    diagnostics.  It is intentionally not an oracle checkpoint digest: these
    are live persistent-process states, so they are labelled separately in
    the manifest and result rather than being misrepresented as a fresh-worker
    checkpoint restoration.
    """
    if type(completed_cycles) is not int or completed_cycles < 0:
        raise ValueError("completed_cycles must be a nonnegative integer")
    if not isinstance(fast_state_gate, bool):
        raise TypeError("fast_state_gate must be boolean")
    if set(reports) != {"left", "right"}:
        raise ValueError("Live reports must contain exactly left and right")
    metrics: dict[str, dict[str, float]] = {}
    provenance = {"completed_cycles": completed_cycles, "reports": {}}
    for side in ("left", "right"):
        report = reports[side]
        if not isinstance(report, Mapping) or report.get("certified") is not True:
            raise ValueError("A causal policy may observe only a certified live boundary")
        ratios = report.get("capacity_ratios")
        if not isinstance(ratios, Mapping):
            raise ValueError("Certified live boundary lacks capacity ratios")
        biceps = _finite(ratios.get("Biceps"), f"{side}.Biceps capacity ratio")
        metrics[side] = {"biceps_A_fraction": biceps, "fast_state_gate": float(fast_state_gate)}
        provenance["reports"][side] = report
    return TransferObservation(completed_cycles, metrics, True, _json_digest(provenance))


def _status_from_attempt(metrics: Mapping[str, object] | None, *, censored: bool) -> tuple[str, str]:
    if censored:
        return "right_censored", "absolute_cycle_budget"
    if not metrics:
        return "numerical_stop", "missing_terminal_attempt"
    return classify_terminal(dict(metrics), censored=False)


def aggregate_terminal(*, last_certified_cycle: int, terminal_reports: Mapping[str, Mapping[str, object]] | None,
                       censored: bool) -> dict:
    """Apply the existing frozen-endpoint classifier to both arm reports."""
    arms = {}
    for side in ("left", "right"):
        metrics = (terminal_reports or {}).get(side)
        status, reason = _status_from_attempt(metrics, censored=censored)
        arms[side] = {"status": status, "status_reason": reason,
                      "terminal_metrics": metrics, "last_certified_cycle": last_certified_cycle}
    statuses = {row["status"] for row in arms.values()}
    if censored:
        status = "right_censored"
    elif "physiological_failure_certified" in statuses:
        status = "physiological_failure_certified"
    else:
        status = "numerical_stop"
    return {"status": status, "last_certified_cycle": last_certified_cycle,
            "arms": arms, "exact_label_allowed": status == "physiological_failure_certified",
            "bilateral_scope": "Only the shared live certified prefix is bilateral; the failed attempt is a synchronized boundary observation."}


def _validate_reference(payload: Mapping[str, object], *, require_causal_work_target_adapter: bool = False) -> None:
    """Refuse a silent change of the cold unit-weight Radau-5 reference."""
    if payload.get("solver") != "ipopt" or payload.get("formulation") != "isokinetic":
        raise ValueError("Continuous causal benchmark requires IPOPT isokinetic RHO")
    if payload.get("cycles_per_window") != 1 or payload.get("stimulations_per_cycle") != 30:
        raise ValueError("Continuous causal benchmark requires one-cycle 30 Hz RHO windows")
    if payload.get("parametric_fatigue_weights") is not True:
        raise ValueError("Compiled unit fatigue-weight parameter binding is required")
    pace = payload.get("muscle_pace")
    resistance = payload.get("resistance_pace")
    if not isinstance(pace, Mapping) or not isinstance(resistance, Mapping):
        raise ValueError("Missing muscle/resistance PACE configuration")
    if pace.get("adaptation_enabled") is not False or resistance.get("capacity_feedback") is not False:
        raise ValueError("Reference must have fixed unit weights and fixed equal split")
    if require_causal_work_target_adapter:
        if pace.get("update_every_cycles") != 1 or resistance.get("update_every_cycles") != 1:
            raise ValueError("Continuous causal work target requires an admitted one-RHO-boundary parameter adapter")
        adapter = payload.get("causal_work_target_adapter")
        if not isinstance(adapter, Mapping) or adapter.get("kind") != "compiled_terminal_eprod_parameter":
            raise ValueError("Continuous causal work target adapter receipt is missing")
    for side in ("left", "right"):
        if not math.isclose(_finite(payload.get(f"{side}_equivalent_mean_torque_nm"), side), .96, abs_tol=1e-12):
            raise ValueError("Reference cold start must use 0.96 Nm on both arms")
        if pace.get(f"{side}_initial_weights") != dict.fromkeys(NAMES, 1.):
            raise ValueError("Reference cold start must use unit muscle weights")
        args = process._driver_arguments(payload, side)
        if args.collocation_degree != 5 or args.ipopt_linear_solver != "ma57":
            raise ValueError("Reference must use Radau-5 and IPOPT MA57")
        if not args.retry_failed_rho_without_advance or not args.ipopt_frozen_rho_feasibility_probe:
            raise ValueError("Reference must use the frozen-feasibility retry/probe protocol")


def _configure_causal_work_target_adapter(payload: Mapping[str, object]) -> dict:
    """Open only the compiled terminal-work parameter at every RHO boundary.

    The source unit RHO has a 20-cycle *slow PACE* cadence.  That cadence is
    safe for weight adaptation but cannot represent an event whose state
    crossing occurs at c135 and whose declared duration is ten cycles: the
    stock worker correctly rejects its target change as outside the block.
    Here weights remain fixed and capacity feedback remains disabled.  We
    change only the admissibility cadence of the already-compiled terminal
    ``E_prod`` parameter to one RHO window.  Therefore the mechanical target
    itself stays exactly 0.96/0.96 outside the one action and its explicit
    return; no graph/objective rebuild or mid-window mechanics are involved.
    """
    result = json.loads(json.dumps(payload))
    for key in ("muscle_pace", "resistance_pace"):
        if not isinstance(result.get(key), dict):
            raise ValueError(f"Missing {key} configuration")
        result[key]["update_every_cycles"] = 1
    result["causal_work_target_adapter"] = {
        "kind": "compiled_terminal_eprod_parameter",
        "boundary_cadence_cycles": 1,
        "weights_fixed": True,
        "capacity_feedback": False,
        "mechanical_scope": "A target may change only between complete one-cycle RHO windows; it is reasserted unchanged otherwise.",
        "compiled_nlp_reused": True,
    }
    return result


def prepare(output: Path, *, source_input: Path = DEFAULT_INPUT, maximum_cycles: int = 240,
            right_cpu: int = 0, left_cpu: int = 1,
            experimental_unvalidated_fast_state_override: bool = False) -> Path:
    """Seal a cold-start continuous protocol without spawning a solver."""
    output = Path(output).resolve()
    source_input = Path(source_input).resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    if type(maximum_cycles) is not int or maximum_cycles < 1:
        raise ValueError("maximum_cycles must be a positive integer")
    if type(right_cpu) is not int or type(left_cpu) is not int or right_cpu < 0 or left_cpu < 0 or right_cpu == left_cpu:
        raise ValueError("Supply two distinct nonnegative CPU IDs")
    payload = json.loads(source_input.read_text(encoding="utf-8"))
    _validate_reference(payload)
    payload = _configure_causal_work_target_adapter(payload)
    payload["cycles"] = maximum_cycles
    payload["solver_cpu_affinity"] = {"right": right_cpu, "left": left_cpu}
    _validate_reference(payload, require_causal_work_target_adapter=True)
    policy, config = BicepsReservePolicy(), TransferConfig()
    code = [Path(__file__).resolve(), ROOT / "cocofest/cycling/slow/causal_transfer.py",
            ROOT / "cocofest/simulation/independent_arms_process.py",
            ROOT / "scripts/run_unilateral_continuation_labels.py",
            ROOT / "examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py"]
    gate_status = ("unvalidated_experimental_override" if experimental_unvalidated_fast_state_override
                   else "not_validated_safe_abstention")
    manifest = {
        "schema_version": 1,
        "kind": "continuous_causal_left_to_right_transfer_rho_benchmark",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_input": str(source_input), "source_input_sha256": digest(source_input),
        "payload": payload, "code_sha256": {str(path): digest(path) for path in code},
        "maximum_cycles": maximum_cycles, "numeric_threads": 1,
        "policy": {"kind": "biceps_reserve_v1", **asdict(policy)},
        "transfer_config": asdict(config),
        "fast_state_status": gate_status,
        "experimental_unvalidated_fast_state_override": experimental_unvalidated_fast_state_override,
        "policy_observation": "At every certified live bilateral process boundary, using Biceps A/a_scale from both workers; no cycle-number trigger.",
        "action": "One shot only: left=0.86 Nm, right=1.06 Nm for ten complete next-cycle windows, then explicit 0.96/0.96 Nm return.",
        "work_target_adapter": payload["causal_work_target_adapter"],
        "endpoint_protocol": "Per-arm frozen retry, zero-objective feasibility probe and fatigue-rest counterfactual; bare IPOPT failure is never an endurance endpoint.",
        "compiled_nlp_contract": "One persistent worker per arm builds once; all work-target changes are numerical terminal E_prod updates in that same compiled NLP.",
        "activation_allowed": False,
    }
    if experimental_unvalidated_fast_state_override:
        manifest["warning"] = "EXPERIMENTAL ONLY: fast/history-state closure is not validated; output cannot support clinical activation."
    output.mkdir(parents=True)
    process._atomic_json(output / "manifest.json", manifest)
    process._atomic_json(output / "manifest-seal.json", {"manifest_sha256": digest(output / "manifest.json")})
    return output / "manifest.json"


def authenticated(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = json.loads((manifest_path.parent / "manifest-seal.json").read_text(encoding="utf-8"))
    if digest(manifest_path) != seal.get("manifest_sha256"):
        raise ValueError("Manifest changed after sealing")
    if digest(manifest["source_input"]) != manifest["source_input_sha256"]:
        raise ValueError("Cold-start source input changed")
    for path, expected in manifest["code_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Protocol code changed after sealing: {path}")
    _validate_reference(manifest["payload"], require_causal_work_target_adapter=True)
    return manifest


def _write_runtime_configuration(root: Path, manifest_path: Path, manifest: Mapping[str, object]) -> Path:
    """Publish the configuration receipt required by prepared checkpoint export.

    ``_publish_prepared_pair_export`` content-addresses ``configuration.json``
    when a requested cycle is reached.  The continuous runner originally
    created only its own manifest, which made a perfectly healthy trajectory
    fail at its first requested export (c120 in the reference payload).  This
    is an immutable runtime receipt, not an unsealed replacement for the
    protocol manifest.
    """
    path = Path(root) / "configuration.json"
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite runtime configuration: {path}")
    document = {
        "schema_version": 1,
        "kind": "continuous_causal_transfer_runtime_configuration",
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": digest(manifest_path),
        "payload": manifest["payload"],
        "compiled_nlp_contract": manifest["compiled_nlp_contract"],
        "fast_state_status": manifest["fast_state_status"],
        "activation_allowed": False,
    }
    process._atomic_json(path, document)
    return path


def _publish_prepared_checkpoint(root: Path, completed_cycles: int, prepared: Mapping[str, Mapping[str, object]]):
    """Require the pinned runtime receipt before delegating a checkpoint export."""
    configuration = Path(root) / "configuration.json"
    if not configuration.is_file():
        raise RuntimeError("Prepared checkpoint export requires the pinned runtime configuration.json")
    return process._publish_prepared_pair_export(Path(root), completed_cycles, prepared)


def run(manifest_path: Path, *, timeout_seconds: float = 7200.) -> dict:
    """Execute one live, cold-start two-arm RHO trajectory to its endpoint."""
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = authenticated(manifest_path)
    root = manifest_path.parent
    launch = root / "campaign-launch.json"
    if launch.exists():
        raise FileExistsError("This immutable continuous protocol has already been launched")
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    cpus = manifest["payload"]["solver_cpu_affinity"]
    if not set(cpus.values()) <= os.sched_getaffinity(0):
        raise ValueError("A requested dedicated worker CPU is unavailable")
    _write_runtime_configuration(root, manifest_path, manifest)
    process._atomic_json(launch, {"manifest_sha256": digest(manifest_path), "timeout_seconds": timeout_seconds,
                                  "cpus": cpus, "numeric_threads": 1})
    env = {**os.environ, **dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")}
    # Spawned workers receive their environment through the parent process.
    saved_thread_environment = {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    os.environ.update({key: value for key, value in env.items() if key in saved_thread_environment})
    parents: dict[str, object] = {}
    workers: dict[str, mp.Process] = {}
    context = mp.get_context("spawn")
    payload = manifest["payload"]
    policy = BicepsReservePolicy(**{key: value for key, value in manifest["policy"].items() if key != "kind"})
    supervisor = CausalTransferSupervisor(policy, TransferConfig(**manifest["transfer_config"]))
    fast_gate = bool(manifest["experimental_unvalidated_fast_state_override"])
    history, preparations, decisions = [], [], []
    terminal_reports = None
    completed = 0
    started = time.perf_counter()
    solve_started = started
    try:
        for side in ("right", "left"):
            parent, child = context.Pipe()
            worker = context.Process(target=process._arm_worker, args=(child, side, payload, str(root)),
                                     name=f"continuous-causal-{side}-rho")
            worker.start()
            child.close()
            parents[side], workers[side] = parent, worker
        for completed in range(int(manifest["maximum_cycles"]) + 1):
            reports = process._receive_pair(parents, kind="boundary", completed_cycles=completed,
                                            timeout_seconds=float(timeout_seconds), processes=workers)
            certified = all(reports[side]["metrics"].get("certified") is True for side in ("left", "right"))
            if completed and certified:
                history.append({"cycle": completed, "arms": {side: reports[side]["metrics"] for side in ("left", "right")},
                                "pair_solve_and_transfer_wall_time_s": time.perf_counter() - solve_started})
            if not certified:
                terminal_reports = {side: reports[side]["metrics"] for side in ("left", "right")}
                for connection in parents.values():
                    connection.send({"kind": "stop"})
                break
            if completed == int(manifest["maximum_cycles"]):
                for connection in parents.values():
                    connection.send({"kind": "stop"})
                break
            boundary_start = time.perf_counter()
            metrics = {side: reports[side]["metrics"] for side in ("left", "right")}
            observation = live_biceps_observation(metrics, completed, fast_state_gate=fast_gate)
            decision = supervisor.decide(observation)
            decision_row = {**asdict(decision), "receipt_sha256": decision.receipt_sha256,
                            "observation_metrics": observation.metrics,
                            "policy_wall_seconds": time.perf_counter() - boundary_start,
                            "live_boundary": True,
                            "fast_state_status": manifest["fast_state_status"]}
            decisions.append(decision_row)
            process._atomic_json(root / "decisions.json", {"decisions": decisions})
            previous_torque = {"right": float(payload["right_equivalent_mean_torque_nm"]),
                               "left": float(payload["left_equivalent_mean_torque_nm"])} if not decisions[:-1] else {
                "right": decisions[-2]["right_torque_nm"], "left": decisions[-2]["left_torque_nm"]}
            for side, connection in parents.items():
                connection.send({"kind": "prepare", "completed_cycles": completed,
                                 "equivalent_mean_torque_nm": getattr(decision, f"{side}_torque_nm"),
                                 "previous_equivalent_mean_torque_nm": previous_torque[side], "pace_vr_decision": None})
            prepared = process._receive_pair(parents, kind="prepared", completed_cycles=completed,
                                              timeout_seconds=float(timeout_seconds), processes=workers)
            checkpoint_export = _publish_prepared_checkpoint(root, completed, prepared)
            preparations.append({"completed_cycles": completed, "arms": prepared, "checkpoint_export": checkpoint_export})
            process._atomic_json(root / "progress.json", {"last_certified_cycle": completed, "cycles": history,
                                                            "decisions": decisions, "preparations": preparations})
            solve_started = time.perf_counter()
            for connection in parents.values():
                connection.send({"kind": "solve", "completed_cycles": completed})
        finished = process._receive_pair(parents, kind="finished", completed_cycles=completed,
                                         timeout_seconds=float(timeout_seconds), processes=workers)
        censored = completed == int(manifest["maximum_cycles"])
        terminal = aggregate_terminal(last_certified_cycle=len(history), terminal_reports=terminal_reports, censored=censored)
        result = {"manifest_sha256": digest(manifest_path), "status": terminal["status"],
                  "last_certified_cycle": terminal["last_certified_cycle"], "exact_label_allowed": terminal["exact_label_allowed"],
                  "terminal": terminal, "completed_rho_cycles": len(history), "requested_cycles": manifest["maximum_cycles"],
                  "wall_time_s": time.perf_counter() - started, "arms": {side: message["result"] for side, message in finished.items()},
                  "cycles": history, "decisions": decisions, "preparations": preparations,
                  "decision_timing_s": process._timing_summary(row["policy_wall_seconds"] for row in decisions),
                  "pair_cycle_wall_timing_s": process._timing_summary(row["pair_solve_and_transfer_wall_time_s"] for row in history),
                  "compiled_nlp_contract": manifest["compiled_nlp_contract"], "fast_state_status": manifest["fast_state_status"],
                  "activation_allowed": False}
        process._atomic_json(root / "campaign-completion.json", result)
        return result
    finally:
        for key, value in saved_thread_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for connection in parents.values():
            try:
                connection.send({"kind": "stop"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        for worker in workers.values():
            worker.join(timeout=2.)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2.)
        for connection in parents.values():
            connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--maximum-cycles", type=int, default=240)
    parser.add_argument("--right-cpu", type=int, default=0)
    parser.add_argument("--left-cpu", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=7200.)
    parser.add_argument("--experimental-unvalidated-fast-state-override", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.manifest.name != "manifest.json":
            parser.error("The sealed manifest must be named manifest.json")
        print(prepare(args.manifest.parent, source_input=args.source_input, maximum_cycles=args.maximum_cycles,
                      right_cpu=args.right_cpu, left_cpu=args.left_cpu,
                      experimental_unvalidated_fast_state_override=args.experimental_unvalidated_fast_state_override))
    else:
        print(json.dumps(run(args.manifest, timeout_seconds=args.timeout_seconds), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
