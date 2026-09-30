#!/usr/bin/env python3
"""Prepare, run and inspect a matched three-condition bilateral endurance campaign.

The existing process RHO does not restore a partially completed physical
trajectory. ``run`` resumes at the *condition* level only: it skips a complete
cap-limited result, or a failed result accompanied by a separate physiological
review. Any partial directory or unreviewed stopping result is left untouched.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from tempfile import NamedTemporaryFile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "asymmetric-sides-r192-rho-bo-20260928" / "local-pw-comparative-20260928"
CONDITIONS = ("unit", "pace", "pace_vr")
MUSCLES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cpu_pair(value: str) -> list[int]:
    pieces = value.split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Use exactly two CPU ids: right,left")
    try:
        ids = [int(piece) for piece in pieces]
    except ValueError as error:
        raise argparse.ArgumentTypeError("CPU ids must be integers") from error
    if any(cpu < 0 for cpu in ids) or ids[0] == ids[1]:
        raise argparse.ArgumentTypeError("CPU ids must be distinct and nonnegative")
    return ids


def _cpu_ids(value: str) -> list[int]:
    try:
        ids = [int(piece) for piece in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("CPU ids must be integers") from error
    if not ids or any(cpu < 0 for cpu in ids) or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("CPU ids must be distinct and nonnegative")
    return ids


def _horizon_ladder(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(int(piece) for piece in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("PACE-VR horizons must be integers") from error
    if (not horizons or any(horizon < 1 for horizon in horizons)
            or tuple(sorted(set(horizons), reverse=True)) != horizons):
        raise argparse.ArgumentTypeError("PACE-VR horizons must be distinct positive values in decreasing order")
    return horizons


def _validate_template(name: str, payload: dict[str, Any]) -> None:
    required = {
        "factory": "cocofest.simulation.independent_arms_process:build_process_independent_arms",
        "formulation": "isokinetic", "solver": "ipopt", "cycles_per_window": 1,
        "stimulations_per_cycle": 30, "parallel": True,
        "right_equivalent_mean_torque_nm": .96,
        "left_equivalent_mean_torque_nm": .96,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"{name}: {key} must be {expected!r}")
    if not math.isclose(float(payload.get("omega_rad_s", 0)), -2 * math.pi, abs_tol=1e-10):
        raise ValueError(f"{name}: expected one turn per second at negative angular velocity")
    split = payload.get("resistance_pace", {})
    if not isinstance(split, dict) or split.get("total_equivalent_mean_torque_nm") != 1.92 or split.get("initial_right_fraction") != .5 or split.get("capacity_feedback") is not False or split.get("initial_split_policy") != "manual":
        raise ValueError(f"{name}: expected a fixed 0.96 + 0.96 Nm work split")
    for side in ("right", "left"):
        model = Path(payload.get(f"{side}_model_config", ""))
        if not model.is_file():
            raise ValueError(f"{name}: missing {side} Ding model {model}")
        arguments = payload.get(f"{side}_driver_arguments", [])
        if not isinstance(arguments, list) or any(
            flag in arguments for flag in ("--collocation-degree", "--collocation-method", "--ipopt-linear-solver")
        ):
            raise ValueError(f"{name}: driver arguments may not override Radau5 or MA57")
        if "--ipopt-frozen-rho-feasibility-probe" not in arguments:
            raise ValueError(f"{name}: frozen RHO feasibility probe is required")
    if name == "unit" and payload.get("muscle_pace", {}).get("adaptation_enabled") is not False:
        raise ValueError("unit: weight adaptation must be disabled")
    if name == "pace" and payload.get("muscle_pace", {}).get("adaptation_strategy") != "capacity_feedback":
        raise ValueError("pace: expected the existing capacity feedback policy")


def prepare(run_directory: Path, templates: dict[str, Path], cpu_pairs: dict[str, list[int]],
            *, max_cycles: int, python: Path, pace_vr_cpu_ids: list[int],
            pace_vr_horizon: int = 20, pace_vr_horizons: tuple[int, ...] | None = None) -> Path:
    if run_directory.exists():
        raise FileExistsError(f"Use a fresh campaign directory: {run_directory}")
    if not 1 <= max_cycles <= 3000:
        raise ValueError("max_cycles must be in 1..3000")
    horizons = (pace_vr_horizon,) if pace_vr_horizons is None else tuple(pace_vr_horizons)
    if (not horizons or any(type(horizon) is not int or horizon < 1 for horizon in horizons)
            or tuple(sorted(set(horizons), reverse=True)) != horizons):
        raise ValueError("PACE-VR horizons must be distinct positive values in decreasing order.")
    if set(templates) != set(CONDITIONS) or set(cpu_pairs) != set(CONDITIONS):
        raise ValueError("All three conditions require a template and CPU pair")
    cpu_ids = [cpu for pair in cpu_pairs.values() for cpu in pair]
    if len(cpu_ids) != len(set(cpu_ids)):
        raise ValueError("CPU ids must be distinct across concurrent conditions")
    if (len(pace_vr_cpu_ids) != 2 * len(horizons) or len(set(pace_vr_cpu_ids)) != len(pace_vr_cpu_ids)
            or any(cpu < 0 for cpu in pace_vr_cpu_ids)):
        raise ValueError("PACE-VR supervisor CPUs must provide two distinct nonnegative ids per horizon")
    if set(pace_vr_cpu_ids) & set(cpu_ids):
        raise ValueError("PACE-VR supervisor CPUs must be distinct from all RHO CPUs")
    if not python.is_file():
        raise FileNotFoundError(python)
    configs: dict[str, dict[str, Any]] = {}
    for name in CONDITIONS:
        payload = _read(templates[name].resolve(strict=True))
        _validate_template(name, payload)
        payload["cycles"] = max_cycles
        payload["solver_cpu_affinity"] = dict(zip(("right", "left"), cpu_pairs[name], strict=True))
        if name == "pace_vr":
            vr = payload.get("experimental_pace_vr")
            if not isinstance(vr, dict) or vr.get("enabled") is not True:
                raise ValueError("pace_vr: template must enable experimental_pace_vr")
            if "async" in vr or vr.get("asynchronous") is not True or vr.get("snapshot_at_block_boundary") is not True or vr.get("retain_last_valid_value") is not True:
                raise ValueError("pace_vr: asynchronous snapshot and last-valid-value policy are required")
            if not isinstance(vr.get("horizon_cycles"), int) or vr.get("horizon_cycles") < 1 or vr.get("update_every_cycles") != 20:
                raise ValueError("pace_vr: template must use a positive horizon and update every 20 cycles")
            deadline = _finite(vr.get("deadline_seconds"))
            if deadline is None or not 0 < deadline <= 20:
                raise ValueError("pace_vr: deadline_seconds must be in (0, 20] for the 20-cycle cadence")
            vr["supervisor_cpu_ids"] = list(pace_vr_cpu_ids)
            # Retain the scalar for older audit readers; validation uses both.
            vr["supervisor_cpu_id"] = pace_vr_cpu_ids[0]
            vr["horizon_cycles"] = horizons[0]
            if len(horizons) > 1:
                vr["horizon_ladder_cycles"] = list(horizons)
            else:
                vr.pop("horizon_ladder_cycles", None)
        configs[name] = payload
    reference = configs["unit"]
    for name in CONDITIONS[1:]:
        for key in ("right_model_config", "left_model_config", "omega_rad_s", "stimulations_per_cycle", "runtime_prefix"):
            if configs[name].get(key) != reference.get(key):
                raise ValueError(f"{name}: {key} differs from the unit reference")
    manifest = {"schema_version": 1, "campaign": "pace_vr_asymmetric_1p92nm",
                "requested_cycles": max_cycles, "resistance_nm": 1.92,
                "protocol": {"frequency_hz": 30, "collocation": "Radau5", "nlp_solver": "IPOPT/MA57",
                             "physical_failure_requires_separate_review": True,
                             "partial_cycle_resume_supported": False,
                             "pace_vr_supervisor_cpu_ids": list(pace_vr_cpu_ids),
                             "pace_vr_horizon_cycles": horizons[0],
                             "pace_vr_horizon_ladder_cycles": list(horizons),
                             "pace_vr_async": True,
                             "pace_vr_snapshot_at_block_boundary": True,
                             "pace_vr_retain_last_valid_value": True},
                "python": str(python.resolve()), "conditions": {}}
    for name in CONDITIONS:
        path = run_directory / "configs" / f"{name}.json"
        _write_new(path, configs[name])
        manifest["conditions"][name] = {
            "template": str(templates[name].resolve()), "config": str(path),
            "config_sha256": _hash(path), "output": str(run_directory / "results" / name),
            "cpu_affinity": configs[name]["solver_cpu_affinity"],
        }
    manifest_path = run_directory / "manifest.json"
    _write_new(manifest_path, manifest)
    return manifest_path


def _complete_state(result_root: Path, requested_cycles: int) -> str:
    summary_path = result_root / "summary.json"
    if not summary_path.is_file():
        return "partial_unreviewed" if result_root.exists() and any(result_root.iterdir()) else "pending"
    try:
        summary = _read(summary_path)
        count = summary["completed_rho_cycles"]
        if type(count) is not int or count < 0 or count > requested_cycles:
            return "unreviewed_result"
        arms = summary["arms"]
        if summary.get("success") is True and count == requested_cycles:
            if any(arms[side]["validated_cycles"] != count for side in ("right", "left")):
                return "unreviewed_result"
            return "completed_cap"
        if summary.get("success") is False and isinstance(summary.get("failure"), str):
            # The coordinator records the failed attempted boundary as cycle
            # ``count``. Thus the certified prefix has exactly count - 1 pairs.
            if count < 1 or any(arms[side]["validated_cycles"] < count - 1 for side in ("right", "left")):
                return "unreviewed_result"
            review_path = result_root / "physiological_review.json"
            if not review_path.is_file():
                return "stopped_unreviewed"
            review = _read(review_path)
            if review.get("verdict") == "physiological_infeasibility" and review.get("completed_rho_cycles") == count:
                evidence = Path(review.get("evidence_path", ""))
                if evidence.is_file():
                    return "reviewed_physiological_failure"
            return "stopped_unreviewed"
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return "unreviewed_result"
    return "unreviewed_result"


def run(manifest_path: Path, *, max_parallel: int = 3) -> dict[str, str]:
    if not 1 <= max_parallel <= 3:
        raise ValueError("max_parallel must be in 1..3")
    manifest = _read(manifest_path)
    requested = manifest["requested_cycles"]
    states = {}
    for name in CONDITIONS:
        job = manifest["conditions"][name]
        config = Path(job["config"])
        if _hash(config) != job["config_sha256"]:
            raise ValueError(f"{name}: prepared configuration was modified")
        states[name] = _complete_state(Path(job["output"]), requested)
    blocked = {name: state for name, state in states.items()
               if state not in {"pending", "completed_cap", "reviewed_physiological_failure"}}
    for name, state in states.items():
        if state == "pending" and (manifest_path.parent / "logs" / f"{name}.log").exists():
            blocked[name] = "launcher_log_without_result"
    if blocked:
        raise ValueError(f"Partial or unreviewed results require attention; no new job launched: {blocked}")
    jobs = [name for name, state in states.items() if state == "pending"]

    def launch(name: str) -> str:
        job = manifest["conditions"][name]
        output = Path(job["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        log = manifest_path.parent / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [manifest["python"], str(ROOT / "scripts" / "run_independent_isokinetic_arms.py"),
                   "--config", job["config"], "--output-dir", str(output),
                   "--prefix", _read(Path(job["config"]))["runtime_prefix"]]
        environment = os.environ.copy()
        environment.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                     check=False)
        return f"exit={process.returncode}; result={_complete_state(output, requested)}"

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(launch, name): name for name in jobs}
        for future in as_completed(futures):
            name = futures[future]
            states[name] = future.result()
            print(f"{name}: {states[name]}", flush=True)
    return states


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _timing(values: list[float], *, unit: str = "s") -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"count": len(values), f"mean_{unit}": statistics.fmean(values),
            f"p95_{unit}": ordered[math.ceil(.95 * len(values)) - 1], f"max_{unit}": ordered[-1]}


def _vr_events(arm: dict[str, Any]) -> list[dict[str, Any]]:
    direct = arm.get("pace_vr_events")
    if isinstance(direct, list):
        return [event for event in direct if isinstance(event, dict)]
    cycles = arm.get("cycles", [])
    return [event for cycle in cycles if isinstance(cycle, dict)
            for event in (cycle.get("pace_vr"),)
            if isinstance(event, dict)]


def _vr_event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    runtimes = []
    lags = []
    stale = deadline = applied = 0
    compact = []
    for event in events:
        status = str(event.get("status", ""))
        runtime = _finite(event.get("runtime_s", event.get("supervisor_runtime_s")))
        if runtime is not None:
            runtimes.append(runtime)
        source, application = event.get("source_cycle"), event.get("applied_cycle")
        lag = application - source if type(source) is int and type(application) is int else None
        if lag is not None:
            lags.append(lag)
        stale += event.get("stale") is True or "stale" in status
        deadline += event.get("deadline_exceeded") is True or "deadline" in status
        applied += application is not None or status == "applied"
        compact.append({key: event.get(key) for key in (
            "request_id", "context_digest", "source_cycle", "applied_cycle",
            "status", "runtime_s", "deadline_exceeded", "stale") if key in event})
    return {"event_count": len(events), "applied_count": applied,
            "stale_count": stale, "deadline_miss_count": deadline,
            "supervisor_runtime_s": _timing(runtimes),
            "application_lag_cycles": _timing(lags, unit="cycles"), "events": compact}


def _vr_candidate_screen_summary(events: list[dict[str, Any]], side: str) -> dict[str, Any]:
    """Audit each horizon attempt once; legacy runs explicitly lack screen data."""
    attempts = []
    selected_candidates: dict[str, int] = {}
    selected_horizons: dict[str, int] = {}
    runtimes = []
    improvements = []
    retained = proposed = screened = 0
    for event in events:
        raw_attempts = event.get("attempts")
        if isinstance(raw_attempts, dict):
            horizon_attempts = [(key, value.get(side)) for key, value in raw_attempts.items()
                                if isinstance(value, dict)]
        else:
            arms = event.get("arms")
            arm = arms.get(side) if isinstance(arms, dict) else None
            horizon_attempts = [(arm.get("rollout_horizon_cycles"), arm)] if isinstance(arm, dict) else []
        chosen_horizon = event.get("selected_horizon_cycles")
        chosen_horizon = chosen_horizon.get(side) if isinstance(chosen_horizon, dict) else None
        for horizon, arm in horizon_attempts:
            if not isinstance(arm, dict):
                continue
            screen = arm.get("candidate_screen")
            if not isinstance(screen, dict):
                continue
            screened += 1
            chosen = screen.get("chosen")
            if chosen == "incumbent":
                retained += 1
            elif isinstance(chosen, str) and chosen:
                proposed += 1
            runtime = _finite(arm.get("runtime_s"))
            if runtime is not None:
                runtimes.append(runtime)
            gain = _finite(screen.get("predicted_improvement"))
            if gain is not None and chosen != "incumbent":
                improvements.append(gain)
            selected = chosen_horizon is not None and str(horizon) == str(chosen_horizon)
            issued = selected and side in event.get("applied_arms", [])
            if selected:
                key = str(chosen)
                selected_candidates[key] = selected_candidates.get(key, 0) + 1
                selected_horizons[str(horizon)] = selected_horizons.get(str(horizon), 0) + 1
            attempts.append({"source_cycle": arm.get("source_cycle", event.get("source_cycle")),
                             "applied_cycle": event.get("applied_cycle"), "horizon_cycles": horizon,
                             "rollout_status": arm.get("status"), "chosen_candidate": chosen,
                             "predicted_improvement": gain, "runtime_s": runtime,
                             "selected_by_supervisor": selected, "command_issued": issued,
                             "incumbent_value": _finite(screen.get("incumbent_value")),
                             "candidate_scores": screen.get("candidates"),
                             "reason": screen.get("reason")})
    return {"screened_attempt_count": screened, "proposal_count": proposed,
            "incumbent_retained_count": retained,
            "selected_candidate_counts": selected_candidates,
            "selected_horizon_counts": selected_horizons,
            "rollout_runtime_s": _timing(runtimes),
            "predicted_improvement": _timing(improvements, unit="score"),
            "audit_available": screened > 0, "attempts": attempts}


def analyze(manifest_path: Path) -> dict[str, Any]:
    manifest = _read(manifest_path)
    report: dict[str, Any] = {"campaign": manifest["campaign"], "requested_cycles": manifest["requested_cycles"],
                              "conditions": {}}
    for name in CONDITIONS:
        output = Path(manifest["conditions"][name]["output"])
        state = _complete_state(output, manifest["requested_cycles"])
        row: dict[str, Any] = {"state": state, "output": str(output)}
        report["conditions"][name] = row
        if not (output / "summary.json").is_file():
            continue
        try:
            summary = _read(output / "summary.json")
            row.update({"attempted_boundary_count_reported": summary.get("completed_rho_cycles"),
                        "solver_stop": summary.get("failure"),
                        "pair_wall_timing_s": summary.get("pair_cycle_wall_timing_s")})
            preparations = summary.get("preparations", [])
            certified_counts = []
            for side in ("right", "left"):
                arm = summary["arms"][side]
                cycles = arm["cycles"]
                certified = [cycle for cycle in cycles if cycle.get("certified") is True]
                certified_counts.append(len(certified))
                times = [value for cycle in certified if (value := _finite(cycle.get("solver_time_s"))) is not None]
                last = certified[-1] if certified else {}
                row[side] = {"certified_cycles": len(certified), "solver_timing": _timing(times),
                             "last_certified_cycle": last.get("cycle"),
                             "final_capacity_ratios": last.get("capacity_ratios"),
                             "terminal_slow_states": last.get("terminal_slow_states"),
                             "final_weights_used": last.get("weights_used"),
                             "last_certified_reserve_cost": last.get("mechanical_reserve_cost")}
                prepare_times = [(item.get("completed_cycles"),
                                  _finite(item.get("arms", {}).get(side, {}).get("prepare_time_s")))
                                 for item in preparations if isinstance(item, dict)]
                row[side]["prepare_timing_s"] = _timing([
                    timing for _, timing in prepare_times if timing is not None])
                row[side]["prepare_at_slow_update_s"] = _timing([
                    timing for boundary, timing in prepare_times
                    if type(boundary) is int and boundary > 0 and boundary % 20 == 0 and timing is not None])
                journal = output / side / "weights.jsonl"
                if journal.is_file():
                    applied = []
                    for line in journal.read_text(encoding="utf-8").splitlines():
                        event = json.loads(line)
                        if event.get("event") == "boundary" and event.get("status") == "applied":
                            applied.append({"completed_cycles": event.get("completed_cycles"),
                                            "weights": event.get("weights")})
                    row[side]["weight_updates"] = applied
                # Preserve future-value receipts without assuming the PACE-VR
                # implementation's eventual field names.
                vr_keys = ("pace_vr", "pace_vr_value", "future_value", "viability_value")
                row[side]["future_value_events"] = [
                    {"cycle": cycle.get("cycle"), **{key: cycle[key] for key in vr_keys if key in cycle}}
                    for cycle in cycles if any(key in cycle for key in vr_keys)
                ]
                if name == "pace_vr":
                    row[side]["pace_vr_supervisor"] = _vr_event_summary(_vr_events(arm))
                    row[side]["pace_vr_candidate_screen"] = _vr_candidate_screen_summary(
                        summary.get("pace_vr_supervisor") or [], side)
            row["certified_pair_cycles"] = min(certified_counts)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as error:
            row["analysis_error"] = f"{type(error).__name__}: {error}"
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--unit-config", type=Path, default=REFERENCE / "unit.json")
    parser.add_argument("--pace-config", type=Path, default=REFERENCE / "pace.json")
    parser.add_argument("--pace-vr-config", type=Path)
    parser.add_argument("--unit-cpus", type=_cpu_pair)
    parser.add_argument("--pace-cpus", type=_cpu_pair)
    parser.add_argument("--pace-vr-cpus", type=_cpu_pair)
    parser.add_argument("--pace-vr-supervisor-cpus", type=_cpu_ids,
                        help="two dedicated CPUs per asynchronous bilateral PACE-VR rollout horizon")
    parser.add_argument("--max-cycles", type=int, default=600)
    parser.add_argument("--pace-vr-horizon", type=int, default=20,
                        help="single-horizon PACE-VR rollout (default: 20)")
    parser.add_argument("--pace-vr-horizons", type=_horizon_ladder,
                        help="parallel descending horizon ladder, e.g. 100,50,25,20,5")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--report", type=Path, help="analysis JSON (default: <run-directory>/comparison.json)")
    args = parser.parse_args(argv)
    manifest = args.run_directory.resolve() / "manifest.json"
    if args.command == "prepare":
        if any(value is None for value in (args.pace_vr_config, args.unit_cpus, args.pace_cpus,
                                           args.pace_vr_cpus, args.pace_vr_supervisor_cpus)):
            parser.error("prepare requires --pace-vr-config, all three RHO --*-cpus pairs, and --pace-vr-supervisor-cpus")
        manifest = prepare(args.run_directory.resolve(),
                           {"unit": args.unit_config, "pace": args.pace_config, "pace_vr": args.pace_vr_config},
                           {"unit": args.unit_cpus, "pace": args.pace_cpus, "pace_vr": args.pace_vr_cpus},
                           max_cycles=args.max_cycles, python=args.python,
                           pace_vr_cpu_ids=args.pace_vr_supervisor_cpus,
                           pace_vr_horizon=args.pace_vr_horizon, pace_vr_horizons=args.pace_vr_horizons)
        print(manifest)
    elif args.command == "run":
        print(json.dumps(run(manifest, max_parallel=args.max_parallel), indent=2))
    else:
        result = analyze(manifest)
        report = args.report or (manifest.parent / "comparison.json")
        report.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=report.parent, delete=False) as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
            temporary = Path(stream.name)
        temporary.replace(report)
        print(report)


if __name__ == "__main__":
    main()
