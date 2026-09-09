#!/usr/bin/env python3
"""Run an immutable paired RHO-only prospective endurance matrix.

The currently executable policies are historical fatigue and terminal reserve.
The predeclared H=5/10/20 rollout-objective arms remain explicitly censored
until their scientific source-policy gate and reviewed CLI contract exist.  No
full-horizon result is consumed by this protocol.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SWEEP_SCRIPT = Path(__file__).resolve().with_name("run_terminal_reserve_sweep.py")
_SPEC = importlib.util.spec_from_file_location("_cocofest_terminal_sweep", SWEEP_SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_sweep = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _sweep
_SPEC.loader.exec_module(_sweep)

ROLLOUT_HORIZONS = (5, 10, 20)
ROLLOUT_ACTIVATION_REQUIREMENT = (
    "A reviewed fixed-size symbolic rollout objective must pass the source-policy "
    "reproducibility gate and expose a frozen CLI contract. Option names are "
    "intentionally unspecified until that review is complete."
)


@dataclass(frozen=True)
class MatrixCase:
    slug: str
    formulation: str
    terminal_reserve_weight: float = 0.0
    terminal_reserve_temperature: float = 0.005
    rollout_horizon: int | None = None
    executable: bool = True


def build_cases(weight: float, temperature: float) -> tuple[MatrixCase, ...]:
    return (
        MatrixCase("historical", "historical_fatigue"),
        MatrixCase(
            "terminal-reserve",
            "terminal_reserve",
            terminal_reserve_weight=float(weight),
            terminal_reserve_temperature=float(temperature),
        ),
        *(
            MatrixCase(
                f"rollout-h{horizon}",
                "rollout_objective",
                rollout_horizon=horizon,
                executable=False,
            )
            for horizon in ROLLOUT_HORIZONS
        ),
    )


def _source_stamp(path: Path) -> dict[str, Any]:
    return _sweep.source_stamp(path)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(REPOSITORY_ROOT),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": "/tmp/cocofest-ma57-probe-mpl",
            "OMP_NUM_THREADS": "1",
            "OMP_THREAD_LIMIT": "1",
            "OMP_DYNAMIC": "FALSE",
            "OPENBLAS_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment


def build_case_command(args: argparse.Namespace, case: MatrixCase) -> list[str] | None:
    """Build one executable paired policy; unavailable rollout arms have no command."""

    if not case.executable:
        return None
    directory = args.output_root / case.slug
    return [
        str(args.python),
        "-m",
        "examples.fes_multibody.cycling.cycling_fes_solver_comparison",
        "--solvers",
        "ipopt",
        "--objective",
        "fatigue",
        "--n-windows",
        str(args.n_windows),
        "--cycles-per-window",
        "1",
        "--stimulations-per-cycle",
        str(args.stimulations_per_cycle),
        "--mechanical-formulation",
        "reduced",
        "--signed-crank-torque",
        str(args.signed_crank_torque),
        "--isokinetic-omega",
        str(args.isokinetic_omega),
        "--ipopt-profile",
        "scientific-radau5",
        "--state-scaling",
        "full",
        "--n-threads",
        "1",
        "--ipopt-max-iter",
        str(args.ipopt_max_iter),
        "--ipopt-linear-solver",
        "ma57",
        "--warmup-ipopt-linear-solver",
        "ma57",
        "--ipopt-hsl-library",
        str(args.hsl_library),
        "--ipopt-ma57-automatic-scaling",
        "--ipopt-linear-system-scaling",
        "none",
        "--ipopt-ma57-pivot-order",
        "2",
        "--ipopt-dual-warm-start-mode",
        "off",
        "--ipopt-c-compile",
        "--compact-rho-output",
        "--common-initial-solution",
        str(args.seed),
        "--adopt-common-initial-solution-warmup-cycles",
        "--terminal-reserve-weight",
        repr(case.terminal_reserve_weight),
        "--terminal-reserve-temperature",
        repr(case.terminal_reserve_temperature),
        "--codegen-tag",
        f"prospective_{case.slug}",
        "--output-json",
        str(directory / "result.json"),
        "--receding-horizon-solution-output",
        str(directory / "rho_solution.npz"),
    ]


def _contract(args: argparse.Namespace, cases: tuple[MatrixCase, ...], probe: dict) -> dict:
    source_paths = (
        Path(__file__).resolve(),
        SWEEP_SCRIPT,
        SWEEP_SCRIPT.with_name("probe_ipopt_ma57.py"),
        REPOSITORY_ROOT / "cocofest" / "optimization" / "solver_backends.py",
        REPOSITORY_ROOT / "cocofest" / "optimization" / "muscle_reserve.py",
        REPOSITORY_ROOT
        / "examples"
        / "fes_multibody"
        / "cycling"
        / "cycling_fes_solver_comparison.py",
    )
    return {
        "protocol": "paired-rho-only; no FHO data",
        "seed": _source_stamp(args.seed),
        "hsl_library": _source_stamp(args.hsl_library),
        "python": _source_stamp(args.python),
        "ma57_runtime_probe": _sweep.stable_runtime_probe_identity(probe),
        "sources": [_source_stamp(path) for path in source_paths],
        "configuration": {
            "n_windows": args.n_windows,
            "cycles_per_window": 1,
            "stimulations_per_cycle": args.stimulations_per_cycle,
            "signed_crank_torque_nm": args.signed_crank_torque,
            "isokinetic_omega_rad_s": args.isokinetic_omega,
            "ipopt_max_iter": args.ipopt_max_iter,
            "ipopt_profile": "scientific-radau5",
            "ipopt_linear_solver": "ma57",
            "ipopt_c_compile": True,
            "dual_warm_start_mode": "off",
            "n_threads": 1,
            "sequential_execution": True,
            "allow_experimental_ma57_runtime": bool(
                args.allow_experimental_ma57_runtime
            ),
        },
        "cases": [asdict(case) for case in cases],
        "rollout_activation_requirement": ROLLOUT_ACTIVATION_REQUIREMENT,
    }


def _expected(args: argparse.Namespace, case: MatrixCase) -> dict[str, Any]:
    return {
        "seed_sha256": _source_stamp(args.seed)["sha256"],
        "hsl_sha256": _source_stamp(args.hsl_library)["sha256"],
        "configuration": {
            "solver": "ipopt",
            "objective": "fatigue",
            "n_windows": args.n_windows,
            "cycles_per_window": 1,
            "stimulations_per_cycle": args.stimulations_per_cycle,
            "mechanical_formulation": "reduced",
            "benchmark_profile": "scientific-radau5",
            "profile_integrity": True,
            "state_scaling": "full",
            "n_threads": 1,
            "constant_crank_torque": args.signed_crank_torque,
            "isokinetic_omega": args.isokinetic_omega,
            "max_ipopt_iterations": args.ipopt_max_iter,
            "ipopt_linear_solver": "ma57",
            "warmup_ipopt_linear_solver": "ma57",
            "ipopt_dual_warm_start_mode": "off",
            "ipopt_c_compile": True,
            "ipopt_ma57_automatic_scaling": True,
            "ipopt_linear_system_scaling": "none",
            "ipopt_ma57_pivot_order": 2,
            "terminal_reserve_weight": case.terminal_reserve_weight,
            "terminal_reserve_temperature": case.terminal_reserve_temperature,
        },
        "submitted_options": {
            "max_iter": args.ipopt_max_iter,
            "linear_solver": "ma57",
            "hsllib": str(args.hsl_library),
            "ma57_automatic_scaling": "yes",
            "linear_system_scaling": "none",
            "ma57_pivot_order": 2,
        },
    }


def result_is_complete(args: argparse.Namespace, case: MatrixCase) -> bool:
    if not case.executable:
        return False
    return _sweep.result_is_complete(
        args.output_root / case.slug / "result.json",
        _sweep.SweepCase(
            case.terminal_reserve_weight, case.terminal_reserve_temperature
        ),
        args.n_windows,
        expected=_expected(args, case),
    )


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def summarize_case(
    args: argparse.Namespace,
    case: MatrixCase,
    execution_status: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "slug": case.slug,
        "formulation": case.formulation,
        "rollout_horizon": case.rollout_horizon,
        "terminal_reserve_weight": case.terminal_reserve_weight,
        "terminal_reserve_temperature": case.terminal_reserve_temperature,
        "execution_status": execution_status,
        "complete": False,
        "censored": True,
        "censor_reason": None,
        "covered_cycles": 0,
        "minimum_capacity_ratio": None,
        "fatigue_auc_cycles": None,
        "maximum_pw_upper_fraction": None,
        "solver_time_per_cycle_s": None,
    }
    if not case.executable:
        row["censor_reason"] = "formulation_unavailable"
        row["activation_requirement"] = ROLLOUT_ACTIVATION_REQUIREMENT
        return row
    path = args.output_root / case.slug / "result.json"
    try:
        result = json.loads(path.read_text())["results"][0]
    except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError):
        row["censor_reason"] = execution_status
        return row
    reserve = result.get("terminal_capacity_reserve") or {}
    saturation = result.get("control_saturation") or []
    upper = [
        value
        for item in saturation
        if (value := _finite_or_none(item.get("upper_fraction"))) is not None
    ]
    complete = result_is_complete(args, case)
    row.update(
        {
            "complete": complete,
            "censored": not complete,
            "censor_reason": None if complete else execution_status,
            "covered_cycles": result.get("covered_cycles", 0),
            "minimum_capacity_ratio": _finite_or_none(
                reserve.get("minimum_ratio", result.get("min_A_capacity_ratio"))
            ),
            "fatigue_auc_cycles": _finite_or_none(result.get("fatigue_auc_cycles")),
            "maximum_pw_upper_fraction": max(upper) if upper else None,
            "solver_time_per_cycle_s": _finite_or_none(
                result.get("solver_time_per_cycle_s")
            ),
            "stop_kind": (result.get("stop") or {}).get("kind"),
        }
    )
    return row


def write_summary(
    args: argparse.Namespace,
    cases: tuple[MatrixCase, ...],
    statuses: dict[str, str],
) -> list[dict[str, Any]]:
    rows = [summarize_case(args, case, statuses[case.slug]) for case in cases]
    _sweep.write_manifest(
        args.output_root / "summary.json",
        {
            "schema": "cocofest-prospective-rho-matrix-summary-v1",
            "interpretation": (
                "Paired RHO-only prospective protocol. Incomplete and unavailable "
                "arms are censored; they are not interpreted as endurance failures."
            ),
            "rows": rows,
        },
    )
    fieldnames = sorted({key for row in rows for key in row})
    with (args.output_root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--hsl-library", type=Path, default=_sweep.default_hsl_library())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--n-windows", type=int, default=150)
    parser.add_argument("--stimulations-per-cycle", type=int, default=30)
    parser.add_argument("--signed-crank-torque", type=float, default=0.2)
    parser.add_argument("--isokinetic-omega", type=float, default=-2.0 * math.pi)
    parser.add_argument("--terminal-reserve-weight", type=float, default=0.03)
    parser.add_argument("--terminal-reserve-temperature", type=float, default=0.005)
    parser.add_argument("--ipopt-max-iter", type=int, default=5000)
    parser.add_argument("--ma57-probe-timeout", type=float, default=60.0)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run both long RHO arms; without it only commands are printed.",
    )
    parser.add_argument(
        "--allow-experimental-ma57-runtime",
        action="store_true",
        help="Allow a functionally successful but ABI-warning MA57 stack (never clinical).",
    )
    return parser


def _normalize_and_validate(args: argparse.Namespace) -> None:
    if args.hsl_library is None:
        try:
            args.hsl_library = _sweep.discover_conda_hsl_library()
        except _sweep.HslLibraryDiscoveryError as error:
            raise SystemExit(str(error)) from error
    for name in ("seed", "hsl_library", "output_root", "python"):
        value = getattr(args, name)
        setattr(args, name, None if value is None else value.expanduser().resolve())
    if args.seed is None or not args.seed.is_file():
        raise SystemExit("--seed must name an existing common RHO NPZ seed.")
    if args.hsl_library is None or not args.hsl_library.is_file():
        raise SystemExit("--hsl-library must name an existing CoinHSL library.")
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise SystemExit("--python must point to an executable file.")
    if args.output_root.exists() and not args.output_root.is_dir():
        raise SystemExit("--output-root must be a directory.")
    if args.n_windows < 2:
        raise SystemExit("--n-windows must be at least 2 to prove compiled NLP reuse.")
    if args.stimulations_per_cycle < 1:
        raise SystemExit("--stimulations-per-cycle must be positive.")
    finite = (
        args.signed_crank_torque,
        args.isokinetic_omega,
        args.terminal_reserve_weight,
        args.terminal_reserve_temperature,
        args.ma57_probe_timeout,
    )
    if not all(math.isfinite(value) for value in finite):
        raise SystemExit("Every numerical protocol value must be finite.")
    if args.isokinetic_omega >= 0.0:
        raise SystemExit("--isokinetic-omega must be strictly negative.")
    if args.terminal_reserve_weight <= 0.0:
        raise SystemExit("--terminal-reserve-weight must be strictly positive.")
    if args.terminal_reserve_temperature <= 0.0:
        raise SystemExit("--terminal-reserve-temperature must be strictly positive.")
    if args.ipopt_max_iter < 1 or args.ma57_probe_timeout <= 0.0:
        raise SystemExit("IPOPT iteration and MA57 probe limits must be positive.")
    if args.timeout is not None and (
        not math.isfinite(args.timeout) or args.timeout <= 0.0
    ):
        raise SystemExit("--timeout must be finite and strictly positive.")
    if args.execute and args.dry_run:
        raise SystemExit("--execute and --dry-run are mutually exclusive.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _normalize_and_validate(args)
    cases = build_cases(
        args.terminal_reserve_weight, args.terminal_reserve_temperature
    )
    preflight = _sweep.hsl_preflight(args.hsl_library)
    if not preflight["static_inspection_success"] or preflight["ma57_symbol"] is None:
        raise SystemExit("CoinHSL failed its static MA57-symbol preflight.")
    environment = _environment()
    probe = _sweep.run_ma57_runtime_probe(
        args.python,
        args.hsl_library,
        timeout=args.ma57_probe_timeout,
        environment=environment,
    )
    if probe.get("functional_success") is not True:
        raise SystemExit(
            "The mandatory IPOPT/MA57 solve probe failed: "
            f"{probe.get('error') or probe.get('failure')}."
        )
    if probe.get("production_ready") is not True and not args.allow_experimental_ma57_runtime:
        raise SystemExit(
            "The IPOPT/MA57 solve works but the native stack is not production-ready: "
            f"{probe.get('production_readiness_reasons')}. Rebuild CoinHSL or use "
            "--allow-experimental-ma57-runtime for non-clinical diagnostics."
        )
    contract = _contract(args, cases, probe)
    manifest = _sweep.initialize_immutable_manifest(
        args.output_root / "manifest.json", contract, resume=args.resume
    )
    if not (args.output_root / "ma57-runtime-probe.json").exists():
        _sweep.write_manifest(args.output_root / "ma57-runtime-probe.json", probe)
    _sweep.write_manifest(args.output_root / "ma57-runtime-probe.latest.json", probe)
    progress: dict[str, Any] = {
        "schema": "cocofest-prospective-rho-matrix-progress-v1",
        "campaign_id": manifest["campaign_id"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "cases": {},
    }
    statuses: dict[str, str] = {}
    exit_code = 0
    for case in cases:
        command = build_case_command(args, case)
        record: dict[str, Any] = {
            "case": asdict(case),
            "command": command,
        }
        progress["cases"][case.slug] = record
        if not case.executable:
            record["status"] = statuses[case.slug] = "formulation_unavailable"
            record["activation_requirement"] = ROLLOUT_ACTIVATION_REQUIREMENT
            continue
        if args.resume and result_is_complete(args, case):
            record["status"] = statuses[case.slug] = "reused"
            continue
        if not args.execute:
            record["status"] = statuses[case.slug] = "dry_run"
            print(shlex.join(command or []))
            continue
        directory = args.output_root / case.slug
        directory.mkdir(parents=True, exist_ok=True)
        quarantine = _sweep.quarantine_case_artifacts(directory)
        if quarantine is not None:
            record["quarantined_previous_attempt"] = str(quarantine)
        record["started_at"] = datetime.now(timezone.utc).isoformat()
        _sweep.write_manifest(args.output_root / "progress.json", progress)
        with (directory / "solver.log").open("w") as log:
            try:
                completed = subprocess.run(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                record["return_code"] = completed.returncode
                if completed.returncode != 0:
                    status = "censored_solver_failure"
                elif result_is_complete(args, case):
                    status = "complete"
                else:
                    status = "censored_invalid_result"
            except subprocess.TimeoutExpired:
                record["return_code"] = None
                status = "censored_timeout"
            except OSError as error:
                record["return_code"] = None
                record["error"] = f"{type(error).__name__}: {error}"
                status = "censored_process_error"
        record["status"] = statuses[case.slug] = status
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        if status != "complete":
            exit_code = 1
    progress["updated_at"] = datetime.now(timezone.utc).isoformat()
    _sweep.write_manifest(args.output_root / "progress.json", progress)
    write_summary(args, cases, statuses)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
