#!/usr/bin/env python3
"""Reproducible short IPOPT/MA57 RHO A/B for direct PW versus PW-rate state."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.fes_multibody.cycling import (  # noqa: E402
    cycling_pulse_width_mhe_acados_periodic as rho,
)


def build_arm_cli(
    *,
    mode: str,
    reduced_profile: Path,
    solution_output: Path,
    n_windows: int,
    stimulations_per_cycle: int,
    maximum_rate_us_per_s: float,
    maximum_iterations: int,
    n_threads: int,
    hsl_library: Path | None,
) -> list[str]:
    """Build one arm while keeping every non-structural option identical."""

    cli = [
        "--solver", "ipopt",
        "--ipopt-linear-solver", "ma57",
        "--warmup-ipopt-linear-solver", "ma57",
        "--n-windows", str(n_windows),
        "--cycles-per-window", "1",
        "--stimulations-per-cycle", str(stimulations_per_cycle),
        "--model-formulation", "periodic_node",
        "--mechanical-formulation", "reduced",
        "--formulation", "dynamic",
        "--ode-solver", "collocation",
        "--collocation-degree", "5",
        "--collocation-method", "radau",
        "--objective", "fatigue",
        "--objective-shape", "quadratic",
        "--crank-assistance", "signed:0.1",
        "--compact-rho-output",
        "--use-sx",
        "--initial-guess-diagnostics",
        "--disable-historical-ipopt-initial-guess",
        "--disable-periodic-fes-warmup-projection",
        "--preserve-warmup-pulse-width-seed",
        "--max-ipopt-iterations", str(maximum_iterations),
        "--nlp-tolerance", "1e-6",
        "--primal-feasibility-threshold", "1e-5",
        "--n-threads", str(n_threads),
        "--reduced-cycling-profile", str(reduced_profile),
        "--receding-horizon-solution-output", str(solution_output),
        "--pulse-width-control-mode", mode,
    ]
    if mode == rho.PW_RATE_MODE:
        cli.extend(
            ["--pulse-width-max-rate-us-per-s", str(maximum_rate_us_per_s)]
        )
    else:
        maximum_step_us = maximum_rate_us_per_s / stimulations_per_cycle
        cli.extend(["--pulse-width-max-step-us", str(maximum_step_us)])
    if hsl_library is not None:
        cli.extend(["--ipopt-hsl-library", str(hsl_library)])
    return cli


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def summarize_arm(summary: dict, elapsed_s: float, cli: list[str]) -> dict:
    return _json_safe(
        {
            "success": bool(summary.get("success")),
            "status": summary.get("status"),
            "objective": summary.get("objective"),
            "solver_time_s": summary.get("solver_time_s"),
            "reported_wall_time_s": summary.get("wall_time_s"),
            "end_to_end_wall_time_s": elapsed_s,
            "window_iterations": summary.get("window_iterations"),
            "window_statuses": summary.get("window_statuses"),
            "window_feasibility": summary.get("window_feasibility"),
            "covered_cycles": summary.get("covered_cycles"),
            "pulse_width_command": summary.get("pulse_width_command"),
            "pulse_width_slew_audit": summary.get("pulse_width_slew_audit"),
            "pulse_width_slew_regularization_audit": summary.get(
                "pulse_width_slew_regularization_audit"
            ),
            "initial_applied_pulse_width_traces": summary.get(
                "initial_applied_pulse_width_traces"
            ),
            "initial_guess_audits": summary.get("initial_guess_audits"),
            "execution_timing": summary.get("execution_timing"),
            "solution_output_error": summary.get(
                "receding_horizon_solution_output_error"
            ),
            "cli": cli,
        }
    )


def _median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(finite)) if finite else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reduced-profile",
        type=Path,
        help="Reuse one reduced-mechanics profile in both arms.",
    )
    parser.add_argument("--n-windows", type=int, default=2)
    parser.add_argument("--stimulations-per-cycle", type=int, default=30)
    parser.add_argument("--maximum-rate-us-per-s", type=float, default=3000.0)
    parser.add_argument("--maximum-iterations", type=int, default=1000)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--order",
        choices=("direct-first", "rate-first"),
        default="direct-first",
    )
    parser.add_argument(
        "--ipopt-hsl-library",
        type=Path,
        default=(
            Path(os.environ["IPOPT_HSL_LIBRARY"])
            if os.environ.get("IPOPT_HSL_LIBRARY")
            else None
        ),
    )
    args = parser.parse_args(argv)
    if args.n_windows < 1:
        parser.error("--n-windows must be positive.")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive.")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    hsl_library = (
        args.ipopt_hsl_library.expanduser().resolve()
        if args.ipopt_hsl_library is not None
        else None
    )
    if hsl_library is not None and not hsl_library.is_file():
        parser.error(f"IPOPT HSL library does not exist: {hsl_library}")
    reduced_profile = (
        args.reduced_profile.expanduser().resolve()
        if args.reduced_profile is not None
        else output_dir / "reduced-profile.npz"
    )
    if args.reduced_profile is not None and not reduced_profile.is_file():
        parser.error(f"Reduced profile does not exist: {reduced_profile}")

    order = (
        ("direct", rho.PW_RATE_MODE)
        if args.order == "direct-first"
        else (rho.PW_RATE_MODE, "direct")
    )
    runs = []
    for repetition in range(args.repetitions):
        for mode in order:
            arm_dir = output_dir / f"repeat-{repetition + 1:02d}-{mode}"
            arm_dir.mkdir()
            cli = build_arm_cli(
                mode=mode,
                reduced_profile=reduced_profile,
                solution_output=arm_dir / "trajectory.npz",
                n_windows=args.n_windows,
                stimulations_per_cycle=args.stimulations_per_cycle,
                maximum_rate_us_per_s=args.maximum_rate_us_per_s,
                maximum_iterations=args.maximum_iterations,
                n_threads=args.n_threads,
                hsl_library=hsl_library,
            )
            # Keep every generated solver artifact isolated, while the reduced
            # profile is intentionally shared by both arms.
            previous_cwd = Path.cwd()
            started = perf_counter()
            summary = None
            error = None
            log_path = arm_dir / "runner.log"
            try:
                with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                    os.chdir(arm_dir)
                    parsed = rho.build_argument_parser().parse_args(cli)
                    summary = rho.solve_case(parsed, echo=True)
            except Exception as exc:  # Preserve both arms and their diagnostics.
                error = f"{type(exc).__name__}: {exc}"
                with log_path.open("a", encoding="utf-8") as log:
                    traceback.print_exc(file=log)
            finally:
                os.chdir(previous_cwd)
            arm = (
                summarize_arm(summary, perf_counter() - started, cli)
                if summary is not None
                else {
                    "success": False,
                    "error": error,
                    "end_to_end_wall_time_s": perf_counter() - started,
                    "cli": cli,
                }
            )
            arm.update({"mode": mode, "repetition": repetition + 1})
            arm["log"] = str(log_path)
            runs.append(arm)
            print(
                f"arm={mode} success={arm['success']} "
                f"solver_time_s={arm.get('solver_time_s')} log={log_path}",
                flush=True,
            )

    by_mode = {mode: [run for run in runs if run["mode"] == mode] for mode in order}
    initial_seed_hashes = {
        mode: sorted(
            {
                run.get("pulse_width_command", {}).get(
                    "initial_applied_trace_sha256"
                )
                for run in mode_runs
                if run.get("pulse_width_command", {}).get(
                    "initial_applied_trace_sha256"
                )
            }
        )
        for mode, mode_runs in by_mode.items()
    }
    direct_hashes = initial_seed_hashes.get("direct", [])
    rate_hashes = initial_seed_hashes.get(rho.PW_RATE_MODE, [])
    same_seed = bool(direct_hashes) and direct_hashes == rate_hashes
    aggregate = {
        mode: {
            "successful_runs": sum(bool(run["success"]) for run in mode_runs),
            "run_count": len(mode_runs),
            "median_solver_time_s": _median(
                [run.get("solver_time_s") for run in mode_runs]
            ),
            "median_end_to_end_wall_time_s": _median(
                [run.get("end_to_end_wall_time_s") for run in mode_runs]
            ),
            "median_total_iterations": _median(
                [
                    sum(value for value in (run.get("window_iterations") or []) if value is not None)
                    for run in mode_runs
                ]
            ),
        }
        for mode, mode_runs in by_mode.items()
    }
    document = {
        "schema": "cocofest-pulse-width-rate-state-ab-v1",
        "protocol": {
            "solver": "ipopt",
            "linear_solver": "ma57",
            "objective": "fatigue_only",
            "same_physical_initial_seed_required": True,
            "n_windows": args.n_windows,
            "stimulations_per_cycle": args.stimulations_per_cycle,
            "maximum_rate_us_per_s": args.maximum_rate_us_per_s,
            "repetitions": args.repetitions,
            "order": args.order,
            "hsl_library": str(hsl_library) if hsl_library is not None else None,
        },
        "same_physical_initial_seed": same_seed,
        "initial_applied_trace_sha256": initial_seed_hashes,
        "aggregate": aggregate,
        "runs": runs,
    }
    output_path = output_dir / "benchmark.json"
    output_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"benchmark_json: {output_path}")
    print(f"same_physical_initial_seed: {same_seed}")
    return 0 if same_seed and all(run["success"] for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
