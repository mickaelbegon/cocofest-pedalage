#!/usr/bin/env python3
"""Prepare, run and analyse the controlled four-method PACE-RT validation.

The primary comparison is ``unit``, fixed physiological weights, causal
``PACE`` and asynchronous direct-terminal ``PACE-RT``.  It uses the same two
Ding models and imposed 1.92 Nm total task in every condition.  Optional PACE
RT ablations are run *after* that comparison on the PACE-RT CPU pair: no
physiological fatigue term (only a documented numerical tie-breaker), no
terminal-proximity term and no terminal-target term.

Stopped runs are intentionally never relabelled as fatigue failures here. The
independent-arm worker stores its frozen-state feasibility evidence; a separate
review remains necessary before making a physiological endurance claim.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
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
PACE_RT_TEMPLATE = ROOT / "asymmetric-sides-r192-rho-bo-20260928" / "pace-rt-async-h3-k10-template.json"
PRIMARY = ("unit", "physio", "pace", "pace_rt")
ABLATIONS = ("pace_rt_no_fatigue", "pace_rt_no_proximal", "pace_rt_no_target")
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
    try:
        answer = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("CPU ids must be integers") from error
    if len(answer) != 2 or len(set(answer)) != 2 or any(item < 0 for item in answer):
        raise argparse.ArgumentTypeError("Use two distinct nonnegative CPU ids: right,left")
    return answer


def _cpu_ids(value: str) -> list[int]:
    try:
        answer = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("CPU ids must be integers") from error
    if not answer or len(answer) != len(set(answer)) or any(item < 0 for item in answer):
        raise argparse.ArgumentTypeError("Use distinct nonnegative CPU ids")
    return answer


def _finite_summary(values: list[float]) -> dict[str, float | int] | None:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return None
    return {"count": len(values), "mean_s": statistics.fmean(values),
            "median_s": statistics.median(values),
            "p95_s": values[math.ceil(.95 * len(values)) - 1], "max_s": values[-1]}


def _require_common(name: str, payload: dict[str, Any]) -> None:
    expected = {"factory": "cocofest.simulation.independent_arms_process:build_process_independent_arms",
                "solver": "ipopt", "formulation": "isokinetic", "cycles_per_window": 1,
                "parallel": True, "stimulations_per_cycle": 30, "parametric_fatigue_weights": True}
    for key, wanted in expected.items():
        if payload.get(key) != wanted:
            raise ValueError(f"{name}: {key} must be {wanted!r}")
    if not math.isclose(float(payload.get("omega_rad_s", math.nan)), -2 * math.pi, abs_tol=1e-10):
        raise ValueError(f"{name}: angular velocity must be -2*pi rad/s")
    split = payload.get("resistance_pace")
    if not isinstance(split, dict) or not math.isclose(float(split.get("total_equivalent_mean_torque_nm", math.nan)), 1.92):
        raise ValueError(f"{name}: expected total 1.92 Nm task")
    if (not math.isclose(float(payload.get("right_equivalent_mean_torque_nm", math.nan)), .96)
            or not math.isclose(float(payload.get("left_equivalent_mean_torque_nm", math.nan)), .96)):
        raise ValueError(f"{name}: expected fixed 0.96 + 0.96 Nm split")
    for side in ("right", "left"):
        if not Path(payload.get(f"{side}_model_config", "")).is_file():
            raise ValueError(f"{name}: {side} Ding model is missing")
        arguments = payload.get(f"{side}_driver_arguments")
        if not isinstance(arguments, list) or "--ipopt-frozen-rho-feasibility-probe" not in arguments:
            raise ValueError(f"{name}: frozen feasibility probe is mandatory")
        if any(flag in arguments for flag in ("--collocation-degree", "--collocation-method", "--ipopt-linear-solver")):
            raise ValueError(f"{name}: Radau-5/MA57 may not be overridden")


def _configure_pace_rt(payload: dict[str, Any], supervisor_cpus: list[int]) -> None:
    vr = payload.get("experimental_pace_vr")
    if not isinstance(vr, dict) or vr.get("enabled") is not True:
        raise ValueError("pace_rt: experimental_pace_vr must be enabled")
    if (vr.get("application_mode") != "terminal_reserve_target_experimental"
            or vr.get("asynchronous") is not True or vr.get("update_every_cycles") != 10):
        raise ValueError("pace_rt: expected asynchronous direct terminal reserve every 10 cycles")
    if len(supervisor_cpus) != 2:
        raise ValueError("The bilateral PACE-RT rollout needs two dedicated CPUs (one arm each).")
    if not 0 < float(vr.get("deadline_seconds", 0)) <= 10:
        raise ValueError("pace_rt: deadline must fit inside the 10-cycle update cadence")
    vr["supervisor_cpu_ids"] = list(supervisor_cpus)
    vr["supervisor_cpu_id"] = supervisor_cpus[0]  # compatibility with older result readers


def _ablation_payload(base: dict[str, Any], name: str) -> dict[str, Any]:
    payload = copy.deepcopy(base)
    vr = payload["experimental_pace_vr"]
    if name == "pace_rt_no_fatigue":
        # With both PACE-RT and fatigue inactive at the first boundary, an
        # identically-zero NLP objective triggers IPOPT's Invalid_Number path
        # even though the frozen feasibility probe is valid.  This is a
        # numerical tie-breaker (eight orders below the unit fatigue term),
        # not a physiological fatigue cost; it keeps the ablation solvable
        # until the first asynchronous PACE-RT target arrives.
        payload["fatigue_weight_values"] = [1e-8] * len(MUSCLES)
        payload["fatigue_ablation_tie_breaker"] = 1e-8
    elif name == "pace_rt_no_proximal":
        vr["terminal_proximal_weight"] = 0.
    elif name == "pace_rt_no_target":
        vr["terminal_shortage_weight"] = 0.
    else:
        raise ValueError(f"Unknown ablation: {name}")
    payload["muscle_pace"]["initial_weight_basis"] += f"; ablation={name}"
    return payload


def prepare(run_directory: Path, templates: dict[str, Path], cpu_pairs: dict[str, list[int]], *,
            supervisor_cpus: list[int], max_cycles: int, python: Path, include_ablations: bool) -> Path:
    if run_directory.exists():
        raise FileExistsError(f"Use a fresh campaign directory: {run_directory}")
    if not 1 <= max_cycles <= 3000:
        raise ValueError("max_cycles must be in 1..3000")
    if set(templates) != set(PRIMARY) or set(cpu_pairs) != set(PRIMARY):
        raise ValueError("Templates and CPU pairs are required for unit, physio, pace and pace_rt")
    all_rho_cpus = [cpu for pair in cpu_pairs.values() for cpu in pair]
    if len(set(all_rho_cpus)) != len(all_rho_cpus):
        raise ValueError("Primary conditions need distinct RHO CPUs for concurrent execution")
    if set(all_rho_cpus) & set(supervisor_cpus):
        raise ValueError("PACE-RT supervisor CPUs must be distinct from all concurrent RHO CPUs")
    if not python.is_file():
        raise FileNotFoundError(python)

    configurations: dict[str, dict[str, Any]] = {}
    for name in PRIMARY:
        payload = _read(templates[name].resolve(strict=True))
        _require_common(name, payload)
        payload["cycles"] = max_cycles
        payload["solver_cpu_affinity"] = dict(zip(("right", "left"), cpu_pairs[name], strict=True))
        if name == "pace_rt":
            _configure_pace_rt(payload, supervisor_cpus)
        configurations[name] = payload
    reference = configurations["unit"]
    for name, payload in configurations.items():
        for key in ("right_model_config", "left_model_config", "runtime_prefix", "omega_rad_s", "stimulations_per_cycle"):
            if payload.get(key) != reference.get(key):
                raise ValueError(f"{name}: {key} differs from the matched unit condition")
    if configurations["unit"].get("muscle_pace", {}).get("adaptation_enabled") is not False:
        raise ValueError("unit must have fixed weights")
    if configurations["physio"].get("muscle_pace", {}).get("adaptation_enabled") is not False:
        raise ValueError("physio must have fixed initial physiological weights")
    if configurations["pace"].get("muscle_pace", {}).get("adaptation_strategy") != "capacity_feedback":
        raise ValueError("pace must use the existing causal capacity feedback")

    groups = {name: "primary" for name in PRIMARY}
    if include_ablations:
        for name in ABLATIONS:
            configurations[name] = _ablation_payload(configurations["pace_rt"], name)
            groups[name] = "ablation"

    manifest: dict[str, Any] = {
        "schema_version": 1, "campaign": "pace_rt_four_condition_validation",
        "requested_cycles": max_cycles, "resistance_nm": 1.92,
        "protocol": {"frequency_hz": 30, "collocation": "Radau5", "nlp_solver": "IPOPT/MA57",
                     "frozen_feasibility_review_required": True, "pace_rt_update_every_cycles": 10,
                     "pace_rt_horizon_cycles": configurations["pace_rt"]["experimental_pace_vr"]["horizon_cycles"],
                     "pace_rt_supervisor_cpu_ids": list(supervisor_cpus),
                     "asynchronous_supervisor_isolated": True},
        "python": str(python.resolve()), "conditions": {},
    }
    for name, payload in configurations.items():
        path = run_directory / "configs" / f"{name}.json"
        _write_new(path, payload)
        manifest["conditions"][name] = {"config": str(path), "config_sha256": _hash(path),
                                        "output": str(run_directory / "results" / name),
                                        "cpu_affinity": payload["solver_cpu_affinity"],
                                        "execution_group": groups[name]}
    manifest_path = run_directory / "manifest.json"
    _write_new(manifest_path, manifest)
    return manifest_path


def _state(output: Path, requested: int) -> str:
    summary = output / "summary.json"
    if not summary.is_file():
        return "partial" if output.exists() and any(output.iterdir()) else "pending"
    try:
        document = _read(summary)
        completed = document["completed_rho_cycles"]
        if document.get("success") is True and completed == requested:
            return "completed_cap"
        if document.get("success") is False:
            return "stopped_requires_frozen_review"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        pass
    return "unreviewed"


def run(manifest_path: Path, *, max_parallel: int = 4) -> dict[str, str]:
    if not 1 <= max_parallel <= 4:
        raise ValueError("max_parallel must be in 1..4")
    manifest = _read(manifest_path)
    requested = int(manifest["requested_cycles"])
    states = {name: _state(Path(job["output"]), requested) for name, job in manifest["conditions"].items()}
    blocked = {name: state for name, state in states.items() if state not in {"pending", "completed_cap"}}
    if blocked:
        raise ValueError(f"No result is overwritten or restarted: {blocked}")

    def launch(name: str) -> tuple[str, str]:
        job = manifest["conditions"][name]
        output = Path(job["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        log = manifest_path.parent / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [manifest["python"], str(ROOT / "scripts" / "run_independent_isokinetic_arms.py"),
                   "--config", job["config"], "--output-dir", str(output), "--prefix",
                   _read(Path(job["config"]))["runtime_prefix"]]
        environment = os.environ.copy()
        environment.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                     env=environment, check=False)
        return name, f"exit={process.returncode}; {_state(output, requested)}"

    for group in ("primary", "ablation"):
        pending = [name for name, job in manifest["conditions"].items()
                   if job["execution_group"] == group and states[name] == "pending"]
        # Ablations reuse the PACE-RT CPU pair and are intentionally serial.
        workers = min(max_parallel, len(pending)) if group == "primary" else min(1, len(pending))
        if not workers:
            continue
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(launch, name) for name in pending]
            for future in as_completed(futures):
                name, state = future.result()
                states[name] = state
                print(f"{name}: {state}", flush=True)
    return states


def _supervisor_events(output: Path) -> list[dict[str, Any]]:
    path = output / "pace_vr_supervisor.json"
    if not path.is_file():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return [item for item in doc if isinstance(item, dict)] if isinstance(doc, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _summarise_condition(name: str, job: dict[str, Any]) -> dict[str, Any]:
    output = Path(job["output"])
    row: dict[str, Any] = {"condition": name, "output": str(output), "status": "missing"}
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        return row
    summary = _read(summary_path)
    row.update(success=summary.get("success"), completed_rho_cycles=summary.get("completed_rho_cycles"),
               requested_cycles=summary.get("requested_cycles"), wall_time_s=summary.get("wall_time_s"))
    arms = summary.get("arms", {})
    certified = []
    for side in ("right", "left"):
        arm = arms.get(side, {})
        cycles = arm.get("cycles", []) if isinstance(arm, dict) else []
        solver = [float(c["solver_time_s"]) for c in cycles if isinstance(c, dict)
                  and isinstance(c.get("solver_time_s"), (int, float))]
        solver_trace = [{"cycle": c.get("cycle"), "solver_time_s": c.get("solver_time_s")}
                        for c in cycles if isinstance(c, dict)
                        and isinstance(c.get("solver_time_s"), (int, float))]
        weights = [{"cycle": c.get("cycle"),
                    "weights": c.get("fatigue_objective_weights", c.get("weights_used")),
                    "policy_weights": c.get("weights_used")}
                   for c in cycles if isinstance(c, dict)
                   and isinstance(c.get("fatigue_objective_weights", c.get("weights_used")), list)]
        reserve = [{"cycle": c.get("cycle"), "prediction_at_observed_terminal": c.get("pace_rt_terminal", {}).get("predicted_terminal_value"),
                    "trust_fraction": c.get("pace_rt_terminal", {}).get("maximum_trust_fraction"),
                    "trust_valid": c.get("pace_rt_terminal", {}).get("terminal_trust_validated")}
                   for c in cycles if isinstance(c, dict) and isinstance(c.get("pace_rt_terminal"), dict)]
        model_updates = [event for event in arm.get("pace_vr_events", []) if isinstance(event, dict)
                         and isinstance(event.get("parameter_update", {}).get("last_model"), dict)]
        row[side] = {"certified_cycles": arm.get("validated_cycles"), "rho_solver_time": _finite_summary(solver),
                     "rho_solver_trace": solver_trace,
                     "weights": weights, "terminal_reserve_local_validation": reserve}
        if model_updates:
            row[side]["terminal_reserve_models"] = [{
                "source_cycle": event.get("source_cycle"), "applied_cycle": event.get("applied_cycle"),
                **event["parameter_update"]["last_model"],
            } for event in model_updates]
        if isinstance(arm.get("validated_cycles"), int):
            certified.append(arm["validated_cycles"])
    row["certified_pair_cycles"] = min(certified) if len(certified) == 2 else None
    supervisor = _supervisor_events(output)
    runtimes = [float(event["runtime_s"]) for event in supervisor if isinstance(event.get("runtime_s"), (int, float))]
    # Each accepted compact result has one runtime per arm.  Preserve both so
    # the report can distinguish total supervisor wall time from arm rollouts.
    arm_runtimes = [float(arm["runtime_s"]) for event in supervisor for arm in
                    (event.get("arms", {}).values() if isinstance(event.get("arms"), dict) else [])
                    if isinstance(arm, dict) and isinstance(arm.get("runtime_s"), (int, float))]
    arm_trace = [{"source_cycle": event.get("source_cycle"), "side": side,
                  "runtime_s": arm.get("runtime_s"), "terminal_value": arm.get("terminal_value")}
                 for event in supervisor if isinstance(event.get("arms"), dict)
                 for side, arm in event["arms"].items() if isinstance(arm, dict)
                 and isinstance(arm.get("runtime_s"), (int, float))]
    row["supervisor"] = {"events": len(supervisor), "wall_time": _finite_summary(runtimes),
                         "per_arm_compact_time": _finite_summary(arm_runtimes), "per_arm_trace": arm_trace}
    row["status"] = "cap_reached" if summary.get("success") is True else "stopped_requires_frozen_review"
    return row


def analyze(manifest_path: Path) -> dict[str, Any]:
    manifest = _read(manifest_path)
    return {"schema_version": 1, "campaign": manifest["campaign"], "protocol": manifest["protocol"],
            "conditions": [_summarise_condition(name, job) for name, job in manifest["conditions"].items()],
            "interpretation": {
                "certified_cycles": "common certified RHO prefix, not automatically a physiological failure label",
                "terminal_reserve_local_validation": "local-model prediction evaluated at the solved endpoint; it is not an independent long-horizon realised-reserve certificate",
            }}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--unit-config", type=Path, default=REFERENCE / "unit.json")
    parser.add_argument("--physio-config", type=Path, default=REFERENCE / "physio.json")
    parser.add_argument("--pace-config", type=Path, default=REFERENCE / "pace.json")
    parser.add_argument("--pace-rt-config", type=Path, default=PACE_RT_TEMPLATE)
    parser.add_argument("--unit-cpus", type=_cpu_pair)
    parser.add_argument("--physio-cpus", type=_cpu_pair)
    parser.add_argument("--pace-cpus", type=_cpu_pair)
    parser.add_argument("--pace-rt-cpus", type=_cpu_pair)
    parser.add_argument("--pace-rt-supervisor-cpus", type=_cpu_ids)
    parser.add_argument("--max-cycles", type=int, default=2000)
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    manifest = args.run_directory.resolve() / "manifest.json"
    if args.command == "prepare":
        values = (args.unit_cpus, args.physio_cpus, args.pace_cpus, args.pace_rt_cpus,
                  args.pace_rt_supervisor_cpus)
        if any(value is None for value in values):
            parser.error("prepare requires all four RHO CPU pairs and --pace-rt-supervisor-cpus")
        manifest = prepare(args.run_directory.resolve(),
                           {"unit": args.unit_config, "physio": args.physio_config, "pace": args.pace_config,
                            "pace_rt": args.pace_rt_config},
                           {"unit": args.unit_cpus, "physio": args.physio_cpus, "pace": args.pace_cpus,
                            "pace_rt": args.pace_rt_cpus}, supervisor_cpus=args.pace_rt_supervisor_cpus,
                           max_cycles=args.max_cycles, python=args.python, include_ablations=args.include_ablations)
        print(manifest)
    elif args.command == "run":
        print(json.dumps(run(manifest, max_parallel=args.max_parallel), indent=2))
    else:
        report = args.report or (manifest.parent / "comparison.json")
        result = analyze(manifest)
        report.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=report.parent, delete=False) as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
            temporary = Path(stream.name)
        temporary.replace(report)
        print(report)


if __name__ == "__main__":
    main()
