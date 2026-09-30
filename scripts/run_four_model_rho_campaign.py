#!/usr/bin/env python3
"""Prepare and execute a four-model RHO endurance comparison.

Each independent model chain creates and certifies its one-cycle IPOPT seed at
the declared resistance, then runs uniform RHO, fixed physiological weights,
and predictive RHO-PACE.  The manifest is intentionally compatible with
``analyze_rho_condition_campaign.py`` while jobs are still running.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.physiological_muscle_weights import (
    SOURCE_COMMIT,
    calculate_physiological_muscle_weights,
)
from cocofest.optimization.physiological_weight_cases import (
    MUSCLE_NAMES,
    REPOSITORY_CURRENT_BASELINE_ID,
    list_cases,
)
from scripts.physiological_weight_geometry import build_geometry


DEFAULT_FACTORS = (0.25, 0.5, 1.0, 2.0)
RHO_CONDITIONS = ("rho", "rho-physio", "rho-pace")


def _factor_label(factor: float) -> str:
    return f"x{factor:g}".replace(".", "p")


def _write_new_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _case_for_factor(factor: float):
    for case in list_cases(REPOSITORY_CURRENT_BASELINE_ID, include_targeted_cross=False):
        if math.isclose(factor, 1.0, rel_tol=0, abs_tol=1e-12) and case.control_family == "nominal":
            return case
        if (len(case.factors) == 1
                and case.factors[0].muscle_name == "Triceps"
                and case.factors[0].parameter_name == "alpha_a"
                and math.isclose(case.factors[0].factor, factor, rel_tol=0, abs_tol=1e-12)):
            return case
    raise ValueError(f"No repository-current Triceps alpha_a OFAT case for x{factor:g}")


def prepare_campaign(run_directory: Path, *, reduced_profile: Path, resistance_nm: float,
                     cycles: int, factors: tuple[float, ...]) -> Path:
    """Create immutable model/weight inputs and a live-readable manifest."""
    run_directory = run_directory.expanduser().resolve()
    reduced_profile = reduced_profile.expanduser().resolve(strict=True)
    if run_directory.exists():
        raise FileExistsError(f"Campaign directory must be fresh: {run_directory}")
    if cycles < 1 or cycles > 3000:
        raise ValueError("cycles must be in 1..3000")
    if resistance_nm <= 0 or not math.isfinite(resistance_nm):
        raise ValueError("resistance-nm must be finite and positive")
    if tuple(sorted(set(factors))) != DEFAULT_FACTORS:
        raise ValueError("This controlled panel is exactly Triceps alpha_a x0.25, x0.5, x1, x2")

    # One geometry avoids case-dependent numerical geometry changes; each
    # model nevertheless receives freshly evaluated profiles and weights.
    geometry = build_geometry(MUSCLE_NAMES, n_shooting=960)
    arms, models = [], []
    for index, factor in enumerate(factors):
        case = _case_for_factor(factor)
        model_id = f"ding_triceps_alpha_a_{_factor_label(factor)}"
        model_directory = run_directory / model_id
        parameters = case.as_parameter_dict()
        profiles, geometry_audit = geometry.profiles(parameters)
        weights = calculate_physiological_muscle_weights(
            geometry.theta, profiles, parameters, muscle_names=MUSCLE_NAMES,
            case_id=case.case_id, target_cycles=45, rho=0.8,
            task_torque_threshold=0.2, pre_risk_width_deg=90.0, normalization="max",
        )
        if weights.get("status") != "ok" or not isinstance(weights.get("normalized_weights"), dict):
            raise ValueError(f"Physiological weight calculation is inadmissible for {model_id}: {weights.get('status')}")
        model_path = model_directory / "model.json"
        weight_path = model_directory / "weights.json"
        _write_new_json(model_path, {
            "schema_version": 1, "case_id": case.case_id,
            "provenance": (
                f"repository_current independent OFAT; Triceps alpha_a x{factor:g}; "
                "experimental Ding sensitivity, not clinical calibration"
            ), "muscles": parameters,
        })
        _write_new_json(weight_path, {
            "initial_weight_basis": (
                f"article_formula_raw_max; case={case.case_id}; source_commit={SOURCE_COMMIT}; "
                "source_geometry_n_shooting=960; calibration_cycles=45; rho=0.8; "
                "task_torque_threshold_nm=0.2; pre_risk_width_deg=90; no_FHO_data; "
                "experimental_source_calibration"
            ), "initial_weights": weights["normalized_weights"],
            "policy": {
                "update_every_cycles": 20, "smoothing": 0.2, "capacity_gain": 1.0,
                "min_relative_weight": 0.001, "max_relative_weight": 64.0,
                "max_log_step": math.log(1.1), "max_cycles": cycles,
                "adaptation_strategy": "predictive_moment", "projection_horizon_cycles": 100,
                "projection_substeps": 16, "projection_budget_seconds": 16.0,
                "projection_async": True, "projection_fatigue_guard": True,
            },
        })
        seed = model_directory / "seed" / "seed.npz"
        record = {"model_id": model_id, "factor": factor, "model_config": str(model_path),
                  "weights_config": str(weight_path), "seed": str(seed), "cpu_id": index}
        models.append(record)
        for condition in ("rho", "rho-physio", "rho-pace"):
            target = model_directory / condition
            arms.append({"id": f"{model_id}/{condition}", "model_id": model_id,
                         "condition": condition, "result_path": str(target / "result.json"),
                         "weights_journal_path": (str(target / "weights.jsonl")
                                                  if condition != "rho" else None),
                         "configuration_audit_path": str(target / "configuration-audit.json"),
                         "checkpoint_directory": str(target / "checkpoints"), "state": "pending"})
    manifest = {"schema_version": 1, "campaign_id": run_directory.name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "resistance_nm": resistance_nm, "cycles": cycles, "models": models, "arms": arms,
                "protocol": {"formulation": "dynamic_reduced", "stimulations_per_cycle": 30,
                             "integrator": "Radau-5", "solver": "IPOPT/MA57",
                             "uniform_rho_weights": [1.0] * 4,
                             "weight_calibration": "article formula, recalculated per Ding variant",
                             "fho_used": False, "failure_interpretation": "solver failure is not physiological proof"}}
    manifest_path = run_directory / "manifest.json"
    _write_new_json(manifest_path, manifest)
    return manifest_path


def _base_benchmark_args(*, profile: Path, resistance_nm: float, cycles: int, hsl_library: Path,
                         output: Path, seed: Path | None = None,
                         replay_checkpoint: Path | None = None) -> list[str]:
    args = ["--solvers", "ipopt", "--objective", "fatigue", "--objective-shape", "quadratic",
            "--formulation", "dynamic", "--mechanical-formulation", "reduced",
            "--stimulations-per-cycle", "30", "--signed-crank-torque", str(resistance_nm),
            "--reduced-cycling-profile", str(profile), "--ipopt-profile", "periodic_collocation",
            "--ipopt-collocation-method", "radau", "--ipopt-collocation-degree", "5", "--ipopt-use-sx",
            "--ipopt-linear-solver", "ma57", "--ipopt-hsl-library", str(hsl_library),
            "--ipopt-max-iter", "4000", "--ipopt-print-level", "0", "--nlp-tolerance", "1e-8",
            "--n-threads", "1", "--compact-rho-output", "--cycles-per-window", "1",
            "--n-windows", str(cycles), "--retry-failed-rho-without-advance", "--output-json", str(output)]
    if seed is not None:
        args += ["--common-initial-solution", str(seed), "--common-initial-solution-recenter-first-node-bounds",
                 "--adopt-common-initial-solution-warmup-cycles"]
    if replay_checkpoint is not None:
        # This file is overwritten only after a certified physical advance.
        # Following a later failure it is therefore the exact frozen state and
        # PW history needed by probe_rho_endurance_viability.py, never the
        # failed IPOPT iterate.
        args += ["--rho-replay-checkpoint-output", str(replay_checkpoint)]
    return args


def _execute(argv: list[str], *, environment: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        completed = subprocess.run(argv, cwd=ROOT, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}); inspect {log}")


def _archive_failed_weighted_attempt(arm: dict[str, Any]) -> None:
    """Preserve, rather than overwrite, an incomplete weighted attempt."""
    result = Path(arm["result_path"])
    audit = Path(arm["configuration_audit_path"])
    log = result.parent / "launcher-output.log"
    journal = Path(arm.get("weights_journal_path") or result.parent / "weights.jsonl")
    checkpoints = Path(arm["checkpoint_directory"])
    sources = (audit, log, journal, checkpoints)
    if result.exists() or not any(path.exists() for path in sources):
        return
    if audit.is_file():
        try:
            failed = json.loads(audit.read_text(encoding="utf-8")).get("status") == "failed"
        except (OSError, json.JSONDecodeError):
            failed = False
    else:
        # A preflight failure can create a log/checkpoint receipt before the
        # runner creates its audit.  With no result it is safe to preserve it
        # as an incomplete attempt and start in a fresh checkpoint directory.
        failed = True
    if not failed:
        raise RuntimeError(f"Refusing to replace non-failed weighted artifacts: {result.parent}")
    number = 1
    while (result.parent / f"failed-attempt-{number}").exists():
        number += 1
    attempt = result.parent / f"failed-attempt-{number}"
    attempt.mkdir()
    for path in sources:
        if path.exists():
            shutil.move(str(path), attempt / path.name)


def run_model(model: dict[str, Any], *, manifest: dict[str, Any], python: Path, profile: Path,
              hsl_library: Path, conditions: tuple[str, ...] = RHO_CONDITIONS) -> str:
    """Run one model chain; completed result files are retained on service restart."""
    model_id, resistance, cycles = model["model_id"], manifest["resistance_nm"], manifest["cycles"]
    model_dir = Path(model["model_config"]).parent
    seed = Path(model["seed"])
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "IPOPT_HSL_LIBRARY": str(hsl_library)})
    prefix = ["taskset", "--cpu-list", str(model["cpu_id"]), str(python), "scripts/run_configured_cycling_benchmark.py"]
    if not seed.is_file():
        seed_result = model_dir / "seed" / "result.json"
        seed_audit = model_dir / "seed" / "configuration-audit.json"
        seed_args = _base_benchmark_args(profile=profile, resistance_nm=resistance, cycles=1,
                                         hsl_library=hsl_library, output=seed_result)
        seed_args += ["--common-initial-solution-output", str(seed)]
        _execute([*prefix, "--model-config", model["model_config"], "--condition", "rho",
                  "--configuration-audit", str(seed_audit), "--", *seed_args], environment=env,
                 log=model_dir / "seed" / "launcher-output.log")
    for condition in conditions:
        arm = next(item for item in manifest["arms"] if item["id"] == f"{model_id}/{condition}")
        result = Path(arm["result_path"])
        if result.is_file():
            continue
        if condition != "rho":
            _archive_failed_weighted_attempt(arm)
        args = _base_benchmark_args(profile=profile, resistance_nm=resistance, cycles=cycles,
                                    hsl_library=hsl_library, output=result, seed=seed,
                                    replay_checkpoint=result.parent / "last-certified-rho-replay.npz")
        command = [*prefix, "--model-config", model["model_config"], "--condition", condition,
                   "--configuration-audit", arm["configuration_audit_path"], "--checkpoint-every", "20",
                   "--checkpoint-directory", arm["checkpoint_directory"]]
        if condition != "rho":
            command += ["--weights-config", model["weights_config"], "--weights-journal", arm["weights_journal_path"]]
        _execute([*command, "--", *args], environment=env, log=result.parent / "launcher-output.log")
    return model_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--reduced-profile", required=True, type=Path)
    parser.add_argument("--hsl-library", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--resistance-nm", type=float, default=.15)
    parser.add_argument("--cycles", type=int, default=3000)
    parser.add_argument("--max-parallel-models", type=int, default=4)
    parser.add_argument(
        "--conditions",
        default=",".join(RHO_CONDITIONS),
        help=(
            "Comma-separated subset of rho,rho-physio,rho-pace to execute. "
            "This permits a matched baseline/physiology campaign before a "
            "separately configured PACE policy."
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an already prepared campaign without overwriting any artifact")
    args = parser.parse_args(argv)
    if args.max_parallel_models < 1 or args.max_parallel_models > 4:
        parser.error("max-parallel-models must be in 1..4")
    conditions = tuple(item.strip() for item in args.conditions.split(",") if item.strip())
    if not conditions or len(set(conditions)) != len(conditions) or any(item not in RHO_CONDITIONS for item in conditions):
        parser.error("--conditions must be a non-empty, non-repeated subset of rho,rho-physio,rho-pace")
    manifest_path = args.run_directory.expanduser().resolve() / "manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            parser.error(f"--resume requires an existing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("resistance_nm") != args.resistance_nm or manifest.get("cycles") != args.cycles):
            parser.error("--resume resistance-nm and cycles must match the manifest exactly")
        print(f"Resuming: {manifest_path}", flush=True)
    else:
        manifest_path = prepare_campaign(args.run_directory, reduced_profile=args.reduced_profile,
                                         resistance_nm=args.resistance_nm, cycles=args.cycles,
                                         factors=DEFAULT_FACTORS)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"Prepared: {manifest_path}", flush=True)
    if args.prepare_only:
        return
    failures = []
    with ThreadPoolExecutor(max_workers=args.max_parallel_models) as pool:
        futures = {pool.submit(run_model, model, manifest=manifest, python=args.python.resolve(),
                               profile=args.reduced_profile.resolve(), hsl_library=args.hsl_library.resolve(),
                               conditions=conditions): model["model_id"]
                   for model in manifest["models"]}
        for future in as_completed(futures):
            model = futures[future]
            try:
                print(f"Completed chain: {future.result()}", flush=True)
            except BaseException as error:
                failures.append(f"{model}: {type(error).__name__}: {error}")
                print(failures[-1], flush=True)
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
