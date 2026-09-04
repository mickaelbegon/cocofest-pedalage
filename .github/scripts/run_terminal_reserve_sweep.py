#!/usr/bin/env python3
"""Run a paired terminal-reserve RHO screen with IPOPT/MA57.

The screen deliberately runs cases sequentially. Concurrent NLP compilation or
factorization would make wall-time and memory comparisons scientifically
ambiguous. Every case starts from the same solver-neutral primal seed.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Iterable
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = (0.0, 0.01, 0.03, 0.1, 1.0)
DEFAULT_TEMPERATURES = (0.0025, 0.005, 0.01)
DEFAULT_BASELINE_TEMPERATURE = 0.005
DEFAULT_METRIC_TOLERANCES = {
    "minimum_capacity_ratio": 1e-5,
    "fatigue_auc_cycles": 1e-4,
    "maximum_pw_upper_fraction": 1e-3,
}
CAMPAIGN_SOURCE_PATHS = (
    Path(__file__).resolve(),
    REPOSITORY_ROOT
    / "cocofest"
    / "optimization"
    / "solver_backends.py",
    REPOSITORY_ROOT
    / "cocofest"
    / "optimization"
    / "muscle_reserve.py",
    REPOSITORY_ROOT
    / "examples"
    / "fes_multibody"
    / "cycling"
    / "cycling_fes_solver_comparison.py",
    REPOSITORY_ROOT
    / "examples"
    / "fes_multibody"
    / "cycling"
    / "cycling_pulse_width_mhe_acados_periodic.py",
)


@dataclass(frozen=True)
class SweepCase:
    weight: float
    temperature: float

    @property
    def slug(self) -> str:
        def encode(value: float) -> str:
            return f"{value:.8g}".replace("-", "m").replace(".", "p")

        return f"lambda-{encode(self.weight)}_tau-{encode(self.temperature)}"


def parse_float_grid(raw: str, *, non_negative: bool, option: str) -> tuple[float, ...]:
    """Parse, validate and de-duplicate a comma-separated numerical grid."""

    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"{option} contains {item!r}, which is not a number."
            ) from error
        valid_sign = value >= 0.0 if non_negative else value > 0.0
        if not math.isfinite(value) or not valid_sign:
            relation = "non-negative" if non_negative else "strictly positive"
            raise argparse.ArgumentTypeError(f"{option} values must be finite and {relation}.")
        if value not in values:
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError(f"{option} must contain at least one value.")
    return tuple(values)


def build_cases(
    weights: Iterable[float],
    temperatures: Iterable[float],
    *,
    baseline_temperature: float = DEFAULT_BASELINE_TEMPERATURE,
) -> tuple[SweepCase, ...]:
    """Return the predeclared grid with one temperature-independent baseline."""

    weights = tuple(weights)
    temperatures = tuple(temperatures)
    cases: list[SweepCase] = []
    if 0.0 in weights:
        cases.append(SweepCase(0.0, float(baseline_temperature)))
    cases.extend(
        SweepCase(float(weight), float(temperature))
        for weight in weights
        if weight > 0.0
        for temperature in temperatures
    )
    return tuple(cases)


def source_stamp(path: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def hsl_preflight(path: Path) -> dict:
    """Fail early unless the selected library loads and exports MA57."""

    result = {
        "file": source_stamp(path),
        "loadable": False,
        "ma57_symbol": None,
        "error": None,
        "abi_scope": (
            "dlopen and symbol availability only; a short IPOPT/MA57 solve is "
            "still required to validate integer, BLAS and Fortran ABI compatibility"
        ),
    }
    try:
        library = ctypes.CDLL(str(path))
    except OSError as error:
        result["error"] = f"{type(error).__name__}: {error}"
        return result
    result["loadable"] = True
    for symbol in ("ma57id_", "ma57id", "MA57ID"):
        if hasattr(library, symbol):
            result["ma57_symbol"] = symbol
            break
    return result


def metric_tolerances(args: argparse.Namespace) -> dict[str, float]:
    return {
        "minimum_capacity_ratio": float(args.capacity_ratio_tolerance),
        "fatigue_auc_cycles": float(args.fatigue_auc_tolerance),
        "maximum_pw_upper_fraction": float(args.pw_upper_fraction_tolerance),
    }


def campaign_contract(
    args: argparse.Namespace,
    weights: tuple[float, ...],
    temperatures: tuple[float, ...],
    preflight: dict,
) -> dict:
    """Build the immutable identity shared by every case in the screen."""

    return {
        "seed": source_stamp(args.seed),
        "hsl_library": source_stamp(args.hsl_library),
        "hsl_preflight": preflight,
        "python": source_stamp(args.python),
        "sources": [source_stamp(path) for path in CAMPAIGN_SOURCE_PATHS],
        "configuration": {
            "n_windows": int(args.n_windows),
            "signed_crank_torque_nm": float(args.signed_crank_torque),
            "ipopt_max_iter": int(args.ipopt_max_iter),
            "linear_solver": "ma57",
            "dual_warm_start_mode": "off",
            "ipopt_profile": "scientific-radau5",
            "mechanical_formulation": "reduced",
            "state_scaling": "full",
            "n_threads": 1,
            "ipopt_c_compile": True,
            "ma57_automatic_scaling": True,
            "linear_system_scaling": "none",
            "ma57_pivot_order": 2,
            "sequential_execution": True,
            "weights": list(weights),
            "temperatures": list(temperatures),
            "baseline_temperature": float(args.baseline_temperature),
            "metric_tolerances": metric_tolerances(args),
        },
    }


def contract_digest(contract: dict) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def expected_case_contract(
    contract: dict, case: SweepCase
) -> dict:
    """Return the exact result fields required to reuse one case."""

    configuration = contract["configuration"]
    return {
        "seed_sha256": contract["seed"]["sha256"],
        "hsl_sha256": contract["hsl_library"]["sha256"],
        "configuration": {
            "solver": "ipopt",
            "objective": "fatigue",
            "n_windows": configuration["n_windows"],
            "mechanical_formulation": configuration["mechanical_formulation"],
            "benchmark_profile": configuration["ipopt_profile"],
            "profile_integrity": True,
            "state_scaling": configuration["state_scaling"],
            "n_threads": configuration["n_threads"],
            "constant_crank_torque": configuration["signed_crank_torque_nm"],
            "max_ipopt_iterations": configuration["ipopt_max_iter"],
            "ipopt_linear_solver": configuration["linear_solver"],
            "ipopt_dual_warm_start_mode": configuration["dual_warm_start_mode"],
            "ipopt_c_compile": configuration["ipopt_c_compile"],
            "ipopt_ma57_automatic_scaling": configuration[
                "ma57_automatic_scaling"
            ],
            "ipopt_linear_system_scaling": configuration[
                "linear_system_scaling"
            ],
            "ipopt_ma57_pivot_order": configuration["ma57_pivot_order"],
            "terminal_reserve_weight": case.weight,
            "terminal_reserve_temperature": case.temperature,
        },
        "submitted_options": {
            "max_iter": configuration["ipopt_max_iter"],
            "linear_solver": "ma57",
            "hsllib": contract["hsl_library"]["path"],
            "ma57_automatic_scaling": "yes",
            "linear_system_scaling": "none",
            "ma57_pivot_order": 2,
        },
    }


def _values_match(observed, expected) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-15)
        except (TypeError, ValueError):
            return False
    return observed == expected


def valid_rho_solution(path: Path) -> bool:
    """Check that the continuation artefact is a non-empty structured NPZ."""

    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            return bool(
                archive.testzip() is None
                and "metadata__json.npy" in names
                and any(name.startswith("states__") for name in names)
                and any(name.startswith("controls__") for name in names)
            )
    except (OSError, zipfile.BadZipFile):
        return False


def build_case_command(args: argparse.Namespace, case: SweepCase) -> list[str]:
    case_directory = args.output_root / case.slug
    output_json = case_directory / "result.json"
    rho_solution = case_directory / "rho_solution.npz"
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
        "--mechanical-formulation",
        "reduced",
        "--signed-crank-torque",
        str(args.signed_crank_torque),
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
        repr(case.weight),
        "--terminal-reserve-temperature",
        repr(case.temperature),
        "--codegen-tag",
        f"terminal_reserve_{case.slug}",
        "--output-json",
        str(output_json),
        "--receding-horizon-solution-output",
        str(rho_solution),
    ]


def result_is_complete(
    path: Path,
    case: SweepCase,
    n_windows: int,
    *,
    expected: dict | None = None,
) -> bool:
    """Accept a case only when its complete contract and artefacts validate."""

    try:
        payload = json.loads(path.read_text())
        configuration = payload["configurations"]["ipopt"]
        result = payload["results"][0]
        reuse = result["compiled_nlp_reuse"]
    except (OSError, KeyError, IndexError, TypeError, json.JSONDecodeError):
        return False
    reserve = result.get("terminal_capacity_reserve") or {}
    saturation = result.get("control_saturation") or []
    upper_fractions = [
        item.get("upper_fraction")
        for item in saturation
        if item.get("upper_fraction") is not None
    ]
    metrics = (
        reserve.get("minimum_ratio"),
        result.get("fatigue_auc_cycles"),
        max(upper_fractions) if upper_fractions else None,
    )
    kkt = ((result.get("ipopt_runtime") or {}).get("kkt_diagnostics") or {})
    kkt_windows = kkt.get("windows") or []
    kkt_finite = len(kkt_windows) >= n_windows and all(
        all(
            isinstance(row.get(key), (int, float))
            and math.isfinite(float(row[key]))
            for key in ("primal_infeasibility", "dual_infeasibility")
        )
        for row in kkt_windows[:n_windows]
    )
    complete = bool(
        valid_rho_solution(path.with_name("rho_solution.npz"))
        and result.get("success")
        and result.get("solver_success")
        and result.get("physical_success")
        and result.get("requested_cycles") == n_windows
        and result.get("covered_cycles") == n_windows
        and result.get("validated_cycles") == n_windows
        and result.get("physically_validated_cycles") == n_windows
        and configuration.get("ipopt_linear_solver") == "ma57"
        and configuration.get("ipopt_dual_warm_start_mode") == "off"
        and configuration.get("ipopt_c_compile") is True
        and reuse.get("compiled_library_build_count") == 1
        and reuse.get("compiled_library_reused") is True
        and reuse.get("graph_rebuild_detected") is False
        and math.isclose(configuration.get("terminal_reserve_weight", math.nan), case.weight)
        and math.isclose(
            configuration.get("terminal_reserve_temperature", math.nan),
            case.temperature,
        )
        and reserve.get("available") is True
        and reserve.get("physiological_domain_valid") is True
        and all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in metrics
        )
        and kkt.get("available") is True
        and kkt_finite
    )
    if not complete or expected is None:
        return complete

    for key, expected_value in expected["configuration"].items():
        if not _values_match(configuration.get(key), expected_value):
            return False
    input_seed = (result.get("input_provenance") or {}).get(
        "common_initial_solution"
    ) or {}
    hsl = ((result.get("ipopt_runtime") or {}).get("hsl") or {}).get("file") or {}
    if input_seed.get("sha256") != expected["seed_sha256"]:
        return False
    if hsl.get("sha256") != expected["hsl_sha256"]:
        return False
    if (result.get("ipopt_runtime") or {}).get("hsl", {}).get("loadable") is not True:
        return False
    submitted = (result.get("ipopt_runtime") or {}).get(
        "submitted_options"
    ) or (result.get("ipopt_runtime") or {}).get("effective_options") or {}
    return all(
        _values_match(submitted.get(key), expected_value)
        for key, expected_value in expected["submitted_options"].items()
    )


def extract_case_metrics(
    path: Path,
    case: SweepCase,
    n_windows: int,
    *,
    expected: dict | None = None,
) -> dict:
    """Extract the predeclared screening outcomes from one compact result."""

    row = {
        "slug": case.slug,
        "weight": case.weight,
        "temperature": case.temperature,
        "complete": False,
        "covered_cycles": 0,
        "minimum_capacity_ratio": None,
        "fatigue_auc_cycles": None,
        "maximum_pw_upper_fraction": None,
        "solver_time_per_cycle_s": None,
        "compiled_library_build_count": None,
        "compiled_library_reused": None,
        "graph_rebuild_detected": None,
    }
    try:
        payload = json.loads(path.read_text())
        result = payload["results"][0]
    except (OSError, KeyError, IndexError, TypeError, json.JSONDecodeError):
        return row
    reserve = result.get("terminal_capacity_reserve") or {}
    saturation = result.get("control_saturation") or []
    reuse = result.get("compiled_nlp_reuse") or {}
    upper_fractions = [
        entry.get("upper_fraction")
        for entry in saturation
        if entry.get("upper_fraction") is not None
    ]
    row.update(
        {
            "complete": result_is_complete(
                path, case, n_windows, expected=expected
            ),
            "covered_cycles": result.get("covered_cycles", 0),
            "minimum_capacity_ratio": reserve.get("minimum_ratio"),
            "fatigue_auc_cycles": result.get("fatigue_auc_cycles"),
            "maximum_pw_upper_fraction": max(upper_fractions) if upper_fractions else None,
            "solver_time_per_cycle_s": result.get("solver_time_per_cycle_s"),
            "compiled_library_build_count": reuse.get("compiled_library_build_count"),
            "compiled_library_reused": reuse.get("compiled_library_reused"),
            "graph_rebuild_detected": reuse.get("graph_rebuild_detected"),
        }
    )
    return row


def classify_screen(
    rows: list[dict],
    *,
    tolerances: dict[str, float] | None = None,
) -> list[dict]:
    """Classify Pareto-nondominated candidates without claiming a net benefit.

    A candidate must be no worse than the baseline for capacity, fatigue AUC
    and pulse-width saturation, and strictly improve at least one outcome.
    """

    tolerances = dict(DEFAULT_METRIC_TOLERANCES if tolerances is None else tolerances)
    baseline = next(
        (row for row in rows if row["weight"] == 0.0 and row["complete"]), None
    )
    outcomes = (
        ("minimum_capacity_ratio", 1.0),
        ("fatigue_auc_cycles", -1.0),
        ("maximum_pw_upper_fraction", -1.0),
    )
    classified: list[dict] = []
    for source in rows:
        row = dict(source)
        if not row["complete"]:
            row["screen_status"] = "incomplete"
        elif row["weight"] == 0.0:
            row["screen_status"] = "reference"
        elif baseline is None:
            row["screen_status"] = "missing_reference"
        elif any(row[key] is None or baseline[key] is None for key, _ in outcomes):
            row["screen_status"] = "missing_metric"
        else:
            oriented_deltas = {
                key: direction * (float(row[key]) - float(baseline[key]))
                for key, direction in outcomes
            }
            row["delta_vs_baseline"] = {
                key: float(row[key]) - float(baseline[key]) for key, _ in outcomes
            }
            worse = any(
                delta < -tolerances[key]
                for key, delta in oriented_deltas.items()
            )
            improved = any(
                delta > tolerances[key]
                for key, delta in oriented_deltas.items()
            )
            row["screen_status"] = (
                "pareto_nondominated"
                if not worse and improved
                else "dominated_or_equivalent"
            )
        classified.append(row)
    return classified


def write_screen_summary(
    output_root: Path,
    cases: Iterable[SweepCase],
    n_windows: int,
    *,
    expected_by_slug: dict[str, dict] | None = None,
    tolerances: dict[str, float] | None = None,
) -> list[dict]:
    expected_by_slug = expected_by_slug or {}
    tolerances = dict(DEFAULT_METRIC_TOLERANCES if tolerances is None else tolerances)
    rows = classify_screen(
        [
            extract_case_metrics(
                output_root / case.slug / "result.json",
                case,
                n_windows,
                expected=expected_by_slug.get(case.slug),
            )
            for case in cases
        ],
        tolerances=tolerances,
    )
    write_manifest(
        output_root / "summary.json",
        {
            "schema": "cocofest-terminal-reserve-screen-summary-v1",
            "interpretation": (
                "Numerical Pareto screen only; nondominance does not imply "
                "improved endurance."
            ),
            "metric_tolerances": tolerances,
            "rows": rows,
        },
    )
    fieldnames = sorted({key for row in rows for key in row if key != "delta_vs_baseline"})
    with (output_root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key in fieldnames}
            for row in rows
        )
    return rows


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def initialize_immutable_manifest(
    path: Path,
    contract: dict,
    *,
    resume: bool,
) -> dict:
    """Create the campaign identity once or validate it before a resume."""

    campaign_id = contract_digest(contract)
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"Cannot read existing campaign manifest: {error}")
        if not resume:
            raise SystemExit(
                f"Campaign manifest already exists at {path}; use --resume or "
                "choose a new --output-root."
            )
        if (
            existing.get("campaign_id") != campaign_id
            or existing.get("contract") != contract
        ):
            raise SystemExit(
                "The existing campaign contract differs from the requested "
                "resume (seed, HSL, code, Python or numerical options changed)."
            )
        return existing
    if resume:
        raise SystemExit(f"Cannot resume: campaign manifest does not exist at {path}.")
    manifest = {
        "schema": "cocofest-terminal-reserve-sweep-v2",
        "campaign_id": campaign_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
    }
    write_manifest(path, manifest)
    return manifest


def quarantine_case_artifacts(case_directory: Path) -> Path | None:
    """Move known stale outputs aside before a fresh subprocess attempt."""

    stale = [
        path
        for path in (
            case_directory / "result.json",
            case_directory / "rho_solution.npz",
            case_directory / "solver.log",
        )
        if path.exists()
    ]
    if not stale:
        return None
    quarantine = (
        case_directory
        / "previous-attempts"
        / f"attempt-{time.time_ns()}"
    )
    quarantine.mkdir(parents=True, exist_ok=False)
    for path in stale:
        path.replace(quarantine / path.name)
    return quarantine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--hsl-library", type=Path, default=os.environ.get("IPOPT_HSL_LIBRARY"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--weights", default=",".join(map(str, DEFAULT_WEIGHTS)))
    parser.add_argument("--temperatures", default=",".join(map(str, DEFAULT_TEMPERATURES)))
    parser.add_argument("--baseline-temperature", type=float, default=DEFAULT_BASELINE_TEMPERATURE)
    parser.add_argument("--n-windows", type=int, default=20)
    parser.add_argument("--signed-crank-torque", type=float, default=0.2)
    parser.add_argument("--ipopt-max-iter", type=int, default=5000)
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds per case.",
    )
    parser.add_argument(
        "--capacity-ratio-tolerance",
        type=float,
        default=DEFAULT_METRIC_TOLERANCES["minimum_capacity_ratio"],
    )
    parser.add_argument(
        "--fatigue-auc-tolerance",
        type=float,
        default=DEFAULT_METRIC_TOLERANCES["fatigue_auc_cycles"],
    )
    parser.add_argument(
        "--pw-upper-fraction-tolerance",
        type=float,
        default=DEFAULT_METRIC_TOLERANCES["maximum_pw_upper_fraction"],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(cli_args: list[str] | None = None) -> int:
    args = build_parser().parse_args(cli_args)
    args.seed = args.seed.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    args.hsl_library = (
        None
        if args.hsl_library is None
        else args.hsl_library.expanduser().resolve()
    )
    if not args.seed.is_file():
        raise SystemExit(f"Common initial solution does not exist: {args.seed}")
    if args.hsl_library is None or not args.hsl_library.is_file():
        raise SystemExit(
            "MA57 requires --hsl-library pointing to an existing CoinHSL "
            "shared library."
        )
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise SystemExit("--python must point to an executable file.")
    if args.output_root.exists() and not args.output_root.is_dir():
        raise SystemExit("--output-root must be a directory.")
    if args.n_windows < 2:
        raise SystemExit("--n-windows must be at least 2 to audit compiled NLP reuse.")
    if args.ipopt_max_iter < 1:
        raise SystemExit("--ipopt-max-iter must be a positive integer.")
    if not math.isfinite(args.signed_crank_torque):
        raise SystemExit("--signed-crank-torque must be finite.")
    if args.timeout is not None and (
        not math.isfinite(args.timeout) or args.timeout <= 0.0
    ):
        raise SystemExit("--timeout must be finite and strictly positive.")
    if not math.isfinite(args.baseline_temperature) or args.baseline_temperature <= 0.0:
        raise SystemExit("--baseline-temperature must be finite and strictly positive.")
    tolerances = metric_tolerances(args)
    if any(not math.isfinite(value) or value < 0.0 for value in tolerances.values()):
        raise SystemExit("Every metric tolerance must be finite and non-negative.")

    weights = parse_float_grid(args.weights, non_negative=True, option="--weights")
    temperatures = parse_float_grid(
        args.temperatures,
        non_negative=False,
        option="--temperatures",
    )
    if 0.0 not in weights:
        raise SystemExit("A paired screen requires a zero-weight baseline.")
    cases = build_cases(weights, temperatures, baseline_temperature=args.baseline_temperature)
    slugs = [case.slug for case in cases]
    if len(slugs) != len(set(slugs)):
        raise SystemExit(
            "The numerical grid contains values that collide after slug encoding."
        )
    preflight = hsl_preflight(args.hsl_library)
    if not preflight["loadable"]:
        raise SystemExit(
            "CoinHSL failed the parent-process load preflight: "
            f"{preflight['error']}"
        )
    if preflight["ma57_symbol"] is None:
        raise SystemExit(
            "The selected HSL library loads but exports no recognized MA57 symbol."
        )

    manifest_path = args.output_root / "manifest.json"
    contract = campaign_contract(args, weights, temperatures, preflight)
    manifest = initialize_immutable_manifest(
        manifest_path, contract, resume=args.resume
    )
    progress_path = args.output_root / "progress.json"
    progress = {
        "schema": "cocofest-terminal-reserve-sweep-progress-v1",
        "campaign_id": manifest["campaign_id"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "cases": {},
    }
    expected_by_slug = {
        case.slug: expected_case_contract(contract, case) for case in cases
    }

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(REPOSITORY_ROOT),
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "1",
            "OMP_THREAD_LIMIT": "1",
            "OMP_DYNAMIC": "FALSE",
            "OPENBLAS_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    exit_code = 0
    for case in cases:
        case_directory = args.output_root / case.slug
        result_path = case_directory / "result.json"
        command = build_case_command(args, case)
        record = {
            "case": asdict(case),
            "command": command,
            "result": str(result_path),
            "rho_solution": str(case_directory / "rho_solution.npz"),
        }
        progress["cases"][case.slug] = record
        expected = expected_by_slug[case.slug]
        if args.resume and result_is_complete(
            result_path, case, args.n_windows, expected=expected
        ):
            record["status"] = "reused"
            continue
        if args.dry_run:
            record["status"] = "dry_run"
            print(shlex.join(command))
            continue

        case_directory.mkdir(parents=True, exist_ok=True)
        quarantine = quarantine_case_artifacts(case_directory)
        if quarantine is not None:
            record["quarantined_previous_attempt"] = str(quarantine)
        record["started_at"] = datetime.now(timezone.utc).isoformat()
        progress["updated_at"] = record["started_at"]
        write_manifest(progress_path, progress)
        with (case_directory / "solver.log").open("w") as log:
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
                complete = completed.returncode == 0 and result_is_complete(
                    result_path,
                    case,
                    args.n_windows,
                    expected=expected,
                )
                record["status"] = "complete" if complete else "failed"
            except subprocess.TimeoutExpired:
                record["return_code"] = None
                record["status"] = "timeout"
            except OSError as error:
                record["return_code"] = None
                record["status"] = "launch_error"
                record["error"] = f"{type(error).__name__}: {error}"
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        progress["updated_at"] = record["finished_at"]
        write_manifest(progress_path, progress)
        if record["status"] != "complete":
            exit_code = 1
            break

    progress["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(progress_path, progress)
    write_screen_summary(
        args.output_root,
        cases,
        args.n_windows,
        expected_by_slug=expected_by_slug,
        tolerances=tolerances,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
