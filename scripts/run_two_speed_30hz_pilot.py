#!/usr/bin/env python3
"""Paired 30 Hz RHO/PACE pilot and separate slow projection audit, without FHO.

The closed-loop slow arm uses the existing causal capacity-feedback PACE
controller. The projected-horizon weight selector is audited separately: its
proposals do not drive either OCP. Radau degree five is the OCP transcription;
the benchmark's independent continuous replay uses DOP853.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_manifest(*, python, model_config, reduced_profile, hsl_library,
                   output_directory, cycles=6, update_every_cycles=2,
                   projection_horizon=3, resistance_nm=0.1, max_iterations=4000):
    for name, value in (("cycles", cycles), ("update_every_cycles", update_every_cycles),
                        ("projection_horizon", projection_horizon),
                        ("max_iterations", max_iterations)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if cycles > 100 or cycles <= update_every_cycles:
        raise ValueError("Require update_every_cycles < cycles <= 100 to observe an update")
    if not math.isfinite(resistance_nm) or resistance_nm <= 0:
        raise ValueError("resistance_nm must be finite and positive")
    paths = {key: str(Path(value).resolve(strict=True)) for key, value in {
        "python": python, "model_config": model_config, "reduced_profile": reduced_profile,
        "hsl_library": hsl_library}.items()}
    model = json.loads(Path(paths["model_config"]).read_text())
    weights = {
        "initial_weight_basis": "uniform common initial cost; causal capacity feedback; no FHO data",
        "initial_weights": dict.fromkeys(model["muscles"], 1.0),
        "policy": {"update_every_cycles": update_every_cycles, "smoothing": 0.2,
                   "capacity_gain": 1.0, "min_relative_weight": 0.25,
                   "max_relative_weight": 4.0, "max_log_step": math.log(1.1),
                   "max_cycles": cycles},
    }
    output = Path(output_directory).resolve()
    seed = output / "preparation" / "common-seed.npz"
    weights_path = output / "slow-weights.json"
    common = [
        "--solvers", "ipopt", "--objective", "fatigue", "--objective-shape", "quadratic",
        "--formulation", "dynamic", "--mechanical-formulation", "reduced",
        "--signed-crank-torque", str(resistance_nm),
        "--reduced-cycling-profile", paths["reduced_profile"],
        "--cycles-per-window", "1", "--compact-rho-output",
        "--ipopt-profile", "periodic_collocation", "--ipopt-collocation-method", "radau",
        "--ipopt-collocation-degree", "5", "--stimulations-per-cycle", "30", "--ipopt-use-sx",
        "--ipopt-linear-solver", "ma57", "--ipopt-hsl-library", paths["hsl_library"],
        "--ipopt-max-iter", str(max_iterations), "--ipopt-print-level", "0",
        "--nlp-tolerance", "1e-8", "--n-threads", "1", "--max-consecutive-failing", "1",
        "--ipopt-disable-historical-initial-guess", "--validate-integrator-maps",
        "--high-accuracy-trace-max-cycles", str(cycles),
        "--high-accuracy-trace-cycle-milestones", ",".join(map(str, range(1, cycles + 1))),
    ]
    cases = []
    for name, condition, windows in (("preparation", "rho", 1), ("rho", "rho", cycles),
                                      ("slow-pace", "rho-pace", cycles)):
        directory = output / name
        argv = [paths["python"], str(ROOT / "scripts/run_configured_cycling_benchmark.py"),
                "--model-config", paths["model_config"], "--condition", condition]
        if condition == "rho-pace":
            argv += ["--weights-config", str(weights_path), "--weights-journal", str(directory / "weights.jsonl")]
        argv += ["--", *common, "--n-windows", str(windows),
                 "--output-json", str(directory / "result.json")]
        if name == "preparation":
            argv += ["--common-initial-solution-output", str(seed)]
        else:
            argv += ["--common-initial-solution", str(seed),
                     "--common-initial-solution-recenter-first-node-bounds",
                     "--adopt-common-initial-solution-warmup-cycles",
                     "--receding-horizon-solution-output", str(directory / "trajectory.npz")]
        cases.append({"name": name, "kind": "ocp", "argv": argv, "directory": str(directory),
                      "result": str(directory / "result.json"), "requested_cycles": windows})
    directory = output / "projection"
    cases.append({"name": "projection", "kind": "offline_projection", "directory": str(directory),
                  "result": str(directory / "report.json"), "argv": [paths["python"], str(Path(__file__).resolve()),
                  "audit-projection", "--source", str(output / "rho" / "trajectory.npz"),
                  "--model-config", paths["model_config"],
                  "--reduced-profile", paths["reduced_profile"], "--output-directory", str(directory),
                  "--horizon", str(projection_horizon)]})
    return {
        "schema": "cocofest-two-speed-30hz-pilot-v1", "uses_fho_data": False,
        "frequency_hz": 30, "cycle_period_s": 1.0, "control_decisions_per_muscle_per_cycle": 30,
        "fast_ocp_period_cycles": 1, "slow_update_period_cycles": update_every_cycles,
        "ocp_solver": "IPOPT/MA57", "ocp_transcription": "Radau collocation degree 5",
        "continuous_reference_audit": "DOP853; distinct from Radau degree-5 transcription",
        "closed_loop_slow_policy": "causal_capacity_feedback_v1",
        "projection_drives_ocp": False, "endurance_improvement_established": False,
        "criteria": {"required_cycles_per_arm": cycles, "complete_initial_state_equal": True,
                     "all_ocp_windows_certified": True, "slow_update_after_cycle_zero_required": True,
                     # Exploratory acceptance bound agreed for the 30 Hz
                     # coupled pilot. It is an observed-model discrepancy,
                     # not a claim of continuous-time equivalence.
                     "maximum_theta_replay_error_rad": 0.06,
                     "projection_endpoint_moment_error_nm": 0.001,
                     "projection_endpoint_normalized_slow_state_error": 0.001},
        "inputs": {key: {"path": value, "sha256": sha256(Path(value).read_bytes()).hexdigest()}
                   for key, value in paths.items() if key != "python"},
        "weights_path": str(weights_path), "weights": weights, "cases": cases,
    }


def summarize(manifest):
    rows = []
    for case in manifest["cases"]:
        path = Path(case["result"])
        row = {"name": case["name"], "result": str(path), "available": path.is_file()}
        if path.is_file() and case["kind"] == "ocp":
            document = json.loads(path.read_text())
            result = next(r for r in document["results"] if r.get("solver") == "ipopt")
            config = document["configurations"]["ipopt"]
            replay = result.get("high_accuracy_trace_rollout") or {}
            theta_error = replay.get("maximum_absolute_endpoint_error_by_state", {}).get("theta")
            first_states = result.get("state_boundary_snapshots", {}).get("cycle_1", {}).get("states", {})
            row.update({key: result.get(key) for key in (
                "solver_success", "physical_success", "nlp_validated_cycles", "solver_time_s",
                "physically_validated_cycles", "min_A_capacity_ratio", "fatigue_auc_cycles",
                "fatigue_endurance_outcome", "hot_solver_time_median_s")})
            row.update(initial_state={key: state["start"] for key, state in first_states.items()},
                       frequency_hz=1.0 / config["calcium_stimulation_interval_s"],
                       collocation_degree=config["collocation_degree"],
                       max_theta_replay_error_rad=theta_error,
                       replay_available=bool(replay.get("available")),
                       replay_covered_cycles=replay.get("cycle_count"),
                       ipopt_linear_solver=config.get("ipopt_linear_solver"))
            row["replay_local_cycle_checks"] = [{
                "cycle": item.get("absolute_cycle"), "available": item.get("available"),
                "local_reset": item.get("local_reset_at_cycle_start"),
                "max_theta_error_rad": item.get("maximum_absolute_endpoint_error_by_state", {}).get("theta"),
                "max_omega_error_rad_s": item.get("maximum_absolute_endpoint_error_by_state", {}).get("omega"),
            } for item in result.get("high_accuracy_cycle_milestones", [])]
            jumps = result.get("state_boundary_jumps", {}).get("by_state", {})
            row["maximum_state_boundary_jumps"] = {
                name: value.get("maximum_absolute_jump") for name, value in jumps.items()}
            row["numerical_gate_passed"] = bool(
                row["solver_success"] is True and row["physical_success"] is True
                and (row["nlp_validated_cycles"] or 0) >= case["requested_cycles"]
                and row["frequency_hz"] == 30.0 and row["collocation_degree"] == 5
                and row["ipopt_linear_solver"] == "ma57"
                and (row["replay_covered_cycles"] or 0) >= case["requested_cycles"]
                and row["replay_available"] and theta_error is not None
                and math.isfinite(theta_error)
                and theta_error <= manifest["criteria"]["maximum_theta_replay_error_rad"])
        elif path.is_file() and case["kind"] == "offline_projection":
            document = json.loads(path.read_text())
            projected = document["case"]
            row.update(proposal_weights=projected["proposal"]["weights"],
                       selection_basis=projected["proposal"]["selection_basis"],
                       selection_elapsed_s=projected["proposal"]["elapsed_s"],
                       candidate_count_evaluated=len(projected["proposal"]["evaluations"]),
                       prefix_replays=projected["baseline_and_selected_ding_prefix"],
                       actual_model_parameters_verified=bool(document.get("actual_model_builds")),
                       projection_drives_ocp=False)
        rows.append(row)
    arms = [r for r in rows if r["name"] in ("rho", "slow-pace")]
    required_states = {"theta", "omega"} | {
        f"{component}_{muscle}" for component in ("Cn", "F", "A", "Tau1", "Km")
        for muscle in manifest["weights"]["initial_weights"]}
    state_equal = bool(len(arms) == 2 and required_states <= set(arms[0].get("initial_state", {}))
                       and arms[0].get("initial_state") == arms[1].get("initial_state"))
    pace = next(c for c in manifest["cases"] if c["name"] == "slow-pace")
    journal = Path(pace["directory"]) / "weights.jsonl"
    events = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    updates = [e for e in events if e.get("event") == "boundary" and e.get("status") == "applied"
               and e.get("cycle_index", 0) > 0]
    return {"schema": manifest["schema"], "uses_fho_data": False, "rows": rows,
            "paired_complete_initial_state_equal": state_equal,
            "slow_updates": updates,
            "closed_loop_pilot_gate_passed": bool(state_equal and updates
                and all(r.get("numerical_gate_passed", False) for r in arms)),
            "projection_drives_ocp": False, "endurance_improvement_established": False}


def audit_projection(args):
    # Reuse the existing supervisor's bounded candidate set and censoring rules.
    # The source at this stage contains RHO only; no PACE or FHO future enters selection.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cocofest-two-speed-mpl")
    import numpy as np
    from scripts.validate_multilevel_endurance import run_case, _json_finite
    from cocofest.optimization.rho_rollout_adapter import select_certified_rho_cycle
    from cocofest.optimization.configured_cycling_model import (
        configured_model_factories, require_seed_fingerprint, resolve_model_config,
    )
    config = resolve_model_config(json.loads(args.model_config.read_text()))
    fingerprint_audit = require_seed_fingerprint(args.source, config["muscle_parameter_fingerprint"])
    cycle, _ = select_certified_rho_cycle(args.source, cycle_index=0, cycle_period=1.)
    # The adapter validates period and stimulation count; interval metadata is
    # optional in concatenated exports and must not be confused with frequency.
    frequency = cycle.stimulations_per_cycle / cycle.period
    if (not np.isclose(frequency, 30., rtol=0., atol=1e-12) or cycle.collocation_degree != 5
            or cycle.metadata.get("producer_collocation_method") != "radau"
            or cycle.metadata.get("producer_solver") != "ipopt"
            or cycle.metadata.get("configured_condition") != "rho"):
        raise ValueError("Projection audit requires a configured certified IPOPT Radau-5 30 Hz RHO source")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    args.substeps = args.ding_substeps = 16
    arrays = {}
    model_records = []
    # _build_actual_muscle_models imports this patched factory at call time.
    # Keep the resolved variant active throughout policy construction and rollout.
    with configured_model_factories(config, model_records):
        case = run_case(args, 0, args.horizon, arrays,
                        sha256(args.source.read_bytes()).hexdigest(),
                        sha256(args.reduced_profile.read_bytes()).hexdigest())
    if not model_records or any(record["actual_parameters"] != config["muscles"] for record in model_records):
        raise RuntimeError("Projection did not instantiate the exact configured muscle parameters")
    report, replacements = _json_finite({"schema": "two-speed-30hz-projection-audit-v1",
        "uses_fho_data": False, "projection_drives_ocp": False,
        "source_fingerprint_audit": fingerprint_audit, "actual_model_builds": model_records,
        "source_anchor": 0, "case": case,
        "reference": "certified RHO with Radau collocation degree 5",
        "prefix_replay": "full Ding RK4 with 16 substeps per stimulation interval; prescribed kinematics",
        "closed_loop_endurance_gain_established": False})
    report["nonfinite_values_replaced_with_null"] = replacements
    (args.output_directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output_directory / "arrays.npz", **arrays)
    print(f"Projection audit: {args.output_directory / 'report.json'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("prepare", help="Write a reproducible manifest; --run executes it")
    for name in ("model-config", "reduced-profile", "hsl-library", "output-directory"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--python", default=sys.executable)
    prepare.add_argument("--cycles", type=int, default=6)
    prepare.add_argument("--update-every-cycles", type=int, default=2)
    prepare.add_argument("--projection-horizon", type=int, default=3)
    prepare.add_argument("--resistance-nm", type=float, default=0.1)
    prepare.add_argument("--max-iterations", type=int, default=4000)
    prepare.add_argument("--run", action="store_true")
    run = sub.add_parser("run", help="Execute a prepared manifest in fresh case directories")
    run.add_argument("manifest", type=Path)
    audit = sub.add_parser("audit-projection")
    for name in ("source", "model-config", "reduced-profile", "output-directory"):
        audit.add_argument("--" + name, type=Path, required=True)
    audit.add_argument("--horizon", type=int, default=3)
    audit.add_argument("--budget-seconds", type=float, default=60.)
    summary = sub.add_parser("summarize")
    summary.add_argument("manifest", type=Path)
    summary.add_argument("--output", type=Path, help="Write refreshed evidence to a fresh JSON file")
    args = parser.parse_args(argv)
    if args.mode == "audit-projection":
        audit_projection(args)
        return 0
    if args.mode == "prepare":
        values = vars(args).copy()
        values.pop("mode")
        execute = values.pop("run")
        manifest = build_manifest(**values)
        output = args.output_directory.resolve()
        output.mkdir(parents=True, exist_ok=False)
        Path(manifest["weights_path"]).write_text(json.dumps(manifest["weights"], indent=2) + "\n")
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Prepared {manifest_path}", flush=True)
        if not execute:
            return 0
    else:
        manifest_path = args.manifest.resolve()
        manifest = json.loads(manifest_path.read_text())
        if args.mode == "summarize":
            encoded = json.dumps(summarize(manifest), indent=2, allow_nan=False)
            if args.output:
                with args.output.open("x") as stream:
                    stream.write(encoded + "\n")
                print(f"Summary: {args.output}")
            else:
                print(encoded)
            return 0
    environment = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR="/tmp/cocofest-two-speed-mpl",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for label, source in manifest["inputs"].items():
        if sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Manifest input changed after preparation: {label}")
    if json.loads(Path(manifest["weights_path"]).read_text()) != manifest["weights"]:
        raise ValueError("The saved slow-weight configuration differs from the manifest")
    execution = []
    for case in manifest["cases"]:
        directory = Path(case["directory"])
        # The audit creates its own fresh directory, so keep launcher logs beside it.
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite prior case: {directory}")
        if case["kind"] == "ocp":
            directory.mkdir(parents=True)
        print(f"Starting {case['name']}", flush=True)
        started = perf_counter()
        with (manifest_path.parent / (case["name"] + ".log")).open("x") as log:
            result = subprocess.run(case["argv"], cwd=ROOT, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        execution.append({"case": case["name"], "returncode": result.returncode,
                          "elapsed_s": perf_counter() - started})
        if result.returncode:
            break
    report = summarize(manifest)
    report["execution"] = execution
    (manifest_path.parent / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Summary: {manifest_path.parent / 'summary.json'}", flush=True)
    return 0 if (len(execution) == len(manifest["cases"])
                 and all(e["returncode"] == 0 for e in execution)
                 and report["closed_loop_pilot_gate_passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
