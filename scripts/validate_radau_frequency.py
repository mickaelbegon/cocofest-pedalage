"""Paired one-cycle Radau-3/5 validation against continuous DOP853 replay.

The dynamic cycling benchmark fixes cycle duration to one second. Thus 50 Hz
means 50 independent PW decisions per muscle for BOTH degrees. A degree-5
preparation supplies their common complete initial state and stimulation model.
This pilot measures transcription error; it does not certify task endurance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_cases(*, python, model_config, reduced_profile, hsl_library,
                output_directory, frequency_hz=50, resistance_nm=0.22,
                max_iterations=4000):
    if isinstance(frequency_hz, bool) or not isinstance(frequency_hz, int) or frequency_hz < 1:
        raise ValueError("frequency_hz must be a positive integer for a one-second cycle")
    if not math.isfinite(resistance_nm) or resistance_nm <= 0:
        raise ValueError("resistance_nm must be finite and positive")
    output = Path(output_directory).resolve()
    seed = output / "preparation-radau5" / "common-seed.npz"
    common = [
        str(Path(python).resolve()), str(ROOT / "scripts/run_configured_cycling_benchmark.py"),
        "--model-config", str(Path(model_config).resolve(strict=True)), "--condition", "rho", "--",
        "--solvers", "ipopt", "--objective", "fatigue", "--objective-shape", "quadratic",
        "--formulation", "dynamic", "--mechanical-formulation", "reduced",
        "--signed-crank-torque", str(resistance_nm),
        "--reduced-cycling-profile", str(Path(reduced_profile).resolve(strict=True)),
        "--cycles-per-window", "1", "--n-windows", "1", "--compact-rho-output",
        "--ipopt-profile", "periodic_collocation", "--ipopt-collocation-method", "radau",
        "--stimulations-per-cycle", str(frequency_hz), "--ipopt-use-sx",
        "--ipopt-linear-solver", "ma57", "--ipopt-hsl-library", str(Path(hsl_library).resolve(strict=True)),
        "--ipopt-max-iter", str(max_iterations), "--ipopt-print-level", "0",
        "--nlp-tolerance", "1e-8", "--n-threads", "1", "--max-consecutive-failing", "1",
        "--ipopt-disable-historical-initial-guess", "--validate-integrator-maps",
        "--high-accuracy-trace-max-cycles", "1",
    ]
    cases = []
    for name, degree in (("preparation-radau5", 5), ("radau3", 3), ("radau5", 5)):
        directory = output / name
        argv = common + ["--ipopt-collocation-degree", str(degree),
                         "--output-json", str(directory / "result.json")]
        if name.startswith("preparation"):
            argv += ["--common-initial-solution-output", str(seed)]
        else:
            argv += ["--common-initial-solution", str(seed),
                     "--common-initial-solution-recenter-first-node-bounds",
                     "--adopt-common-initial-solution-warmup-cycles"]
        cases.append({"name": name, "degree": degree, "argv": argv,
                      "directory": str(directory), "result": str(directory / "result.json")})
    return cases


def summarize_result(document, *, phase_error_budget_rad=0.0002):
    """Score replay accuracy separately from feasibility and solver success.

    The phase budget reserves 10% of the current 0.002 rad task tolerance for
    transcription error. This is an explicit engineering accuracy target, not
    a physiological threshold. A failed or absent replay can never pass.
    """
    result = next(item for item in document["results"] if item.get("solver") == "ipopt")
    config = document["configurations"]["ipopt"]
    audit = result.get("high_accuracy_trace_rollout") or {}
    row = {"degree": config["collocation_degree"],
           "frequency_hz": 1.0 / config["calcium_stimulation_interval_s"],
           "pw_decisions_per_muscle": config["control_decisions_per_cycle"],
           "solver_success": result.get("solver_success"),
           "nlp_validated_cycles": result.get("nlp_validated_cycles"),
           "solver_time_s": result.get("solver_time_s"),
           "solver_attempts": [{"return_status": item.get("return_status"),
                                "iterations": item.get("iter_count")}
                               for item in result.get("nlp_solver_stats", [])],
           "maximum_primal_violation": max(
               (item.get("feasibility", {}).get("maximum_bound_violation", 0.0)
                for item in result.get("windows", [])), default=None),
           "replay_available": bool(audit.get("available")),
           "phase_error_budget_rad": phase_error_budget_rad,
           "transcription_accuracy_pass": False,
           "task_closure_pass": False}
    if not audit.get("available"):
        row["reason"] = audit.get("reason", result.get("error") or "continuous_replay_unavailable")
        return row
    errors = audit["maximum_absolute_endpoint_error_by_state"]
    snapshots = result.get("state_boundary_snapshots", {}).get("cycle_1", {}).get("states", {})
    start = snapshots.get("theta", {}).get("start", [None])[0]
    final_theta = audit.get("final_reference_state", {}).get("theta", [None])[0]
    closure_error = None if start is None or final_theta is None else abs(final_theta - start + 2 * math.pi)
    angle_tolerance = result.get("physical_crank_diagnostics", {}).get("absolute_cycle_tolerance", 0.00201)
    row.update(maximum_endpoint_error_by_state=errors,
               max_theta_error_rad=errors.get("theta"), max_omega_error_rad_s=errors.get("omega"),
               maximum_scaled_endpoint_error=audit.get("maximum_scaled_endpoint_error"),
               dop853_cycle_closure_error_rad=closure_error,
               task_closure_tolerance_rad=angle_tolerance,
               dop853_rtol=audit.get("relative_tolerance"), dop853_atol=audit.get("absolute_tolerance"),
               initial_state={key: values["start"] for key, values in snapshots.items()},
               reference_evaluations=audit.get("reference_evaluations"))
    error = errors.get("theta")
    row["transcription_accuracy_pass"] = bool(
        result.get("solver_success") and error is not None and math.isfinite(error)
        and error <= phase_error_budget_rad)
    row["task_closure_pass"] = bool(closure_error is not None and math.isfinite(closure_error)
                                     and closure_error <= angle_tolerance)
    row["interpretation"] = "phase_accuracy_pilot_only; requires_dense_bounds_and_reference_convergence"
    return row


def summarize_cases(cases):
    rows = []
    for case in cases:
        path = Path(case["result"])
        row = {"name": case["name"], "result": str(path)}
        row.update(summarize_result(json.loads(path.read_text())) if path.exists()
                   else {"replay_available": False, "reason": "result_missing"})
        rows.append(row)
    paired = [row for row in rows if row["name"] in ("radau3", "radau5")]
    comparable = (len(paired) == 2 and all(row.get("initial_state") for row in paired)
                  and paired[0]["frequency_hz"] == paired[1]["frequency_hz"]
                  and paired[0]["pw_decisions_per_muscle"] == paired[1]["pw_decisions_per_muscle"]
                  and paired[0]["initial_state"] == paired[1]["initial_state"])
    return {"schema": "radau-frequency-validation-v1", "rows": rows,
            "paired_complete_initial_state_and_frequency_equal": comparable,
            "scope": "one_cycle_optimized_controls; not_an_endurance_certificate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--reduced-profile", type=Path, required=True)
    parser.add_argument("--hsl-library", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--frequency-hz", type=int, default=50)
    parser.add_argument("--resistance-nm", type=float, default=0.22)
    parser.add_argument("--max-iterations", type=int, default=4000)
    parser.add_argument("--run", action="store_true", help="Execute the three sequential one-cycle OCPs")
    parser.add_argument("--summarize-only", action="store_true")
    args = vars(parser.parse_args())
    run, summarize = args.pop("run"), args.pop("summarize_only")
    output = args["output_directory"].resolve()
    cases = build_cases(**args)
    if summarize:
        print(json.dumps(summarize_cases(cases), indent=2, allow_nan=False))
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.json").open("x") as handle:
        json.dump({"schema": "radau-frequency-validation-v1", "cases": cases}, handle, indent=2)
    if not run:
        print(f"Prepared {output / 'manifest.json'}; add --run with a fresh output directory to execute")
        return
    environment = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR="/tmp/cocofest-radau-frequency-mpl",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for case in cases:
        directory = Path(case["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Starting {case['name']}", flush=True)
        with (directory / "launcher.log").open("x") as handle:
            completed = subprocess.run(case["argv"], cwd=ROOT, env=environment,
                                       stdout=handle, stderr=subprocess.STDOUT, check=False)
        print(f"Completed {case['name']}: returncode={completed.returncode}", flush=True)
        if completed.returncode:
            break
    summary = summarize_cases(cases)
    with (output / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    for row in summary["rows"]:
        print(f"{row['name']}: NLP success={row.get('solver_success')}, "
              f"DOP853 available={row.get('replay_available')}, "
              f"max phase error={row.get('max_theta_error_rad')} rad, "
              f"closure error={row.get('dop853_cycle_closure_error_rad')} rad, "
              f"accuracy pass={row.get('transcription_accuracy_pass')}")
    print(f"Full numerical audit: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
