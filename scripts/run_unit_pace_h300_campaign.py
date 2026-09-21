#!/usr/bin/env python3
"""Run RHO-PACE from a unit prior with an actually budgeted H=300 rollout.

The baseline RHO and fixed-physiology artifacts are referenced read-only from
an earlier matched campaign.  Only the new PACE arms are launched here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_four_model_rho_campaign import _base_benchmark_args, _execute


def _new_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prepare(*, source_manifest: Path, run_directory: Path, slow_update_cycles: int,
            projection_budget_fraction: float, worker_cpu_offset: int) -> Path:
    source_manifest = source_manifest.expanduser().resolve(strict=True)
    run_directory = run_directory.expanduser().resolve()
    if run_directory.exists():
        raise FileExistsError(f"Run directory must be fresh: {run_directory}")
    if slow_update_cycles < 1:
        raise ValueError("slow_update_cycles must be positive")
    if not 0 < projection_budget_fraction <= 1:
        raise ValueError("projection_budget_fraction must lie in (0, 1]")
    if worker_cpu_offset < 4:
        raise ValueError("worker_cpu_offset must not overlap the four fast RHO CPU ids 0..3")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source.get("cycles") != 3000 or source.get("resistance_nm") != .15:
        raise ValueError("This matched experiment requires the 3000-cycle, 0.15-Nm source campaign")
    models = source.get("models")
    if not isinstance(models, list) or len(models) != 4:
        raise ValueError("Source manifest must define exactly four models")
    source_arms = {arm["id"]: arm for arm in source.get("arms", [])}
    arms, prepared_models = [], []
    for index, model in enumerate(models):
        model_id = model["model_id"]
        for condition in ("rho", "rho-physio"):
            arm = source_arms.get(f"{model_id}/{condition}")
            if arm is None or not Path(arm["result_path"]).is_file():
                raise ValueError(f"Missing matched source result for {model_id}/{condition}")
            arms.append({**arm, "source": "four-model-r015-3000-20260913b"})
        directory = run_directory / model_id / "rho-pace-unit-h300"
        weights = run_directory / model_id / "weights-unit-h300.json"
        _new_json(weights, {
            "initial_weight_basis": (
                "uniform_unit_prior; matched RHO baseline; no_FHO_data; "
                "PACE predictive rollout H=300; experimental comparison"
            ),
            "initial_weights": {name: 1.0 for name in ("Delt_ant", "Delt_post", "Biceps", "Triceps")},
            "policy": {
                # The fast OCP is still one RHO per cycle.  This cadence only
                # governs independent supervisory proposals: 60 x 0.9 = 54 s
                # is deliberately large enough for the measured H=300 batch.
                "update_every_cycles": slow_update_cycles, "max_cycles": 3000,
                "min_relative_weight": .25, "max_relative_weight": 4.0,
                "max_log_step": math.log(1.1), "adaptation_strategy": "predictive_moment",
                "projection_horizon_cycles": 300, "projection_substeps": 16,
                "projection_budget_seconds": slow_update_cycles * projection_budget_fraction,
                "projection_budget_fraction": projection_budget_fraction,
                "projection_async": True, "projection_worker_cpu_ids": [worker_cpu_offset + index],
                "projection_fatigue_guard": True,
            },
        })
        prepared = {**model, "cpu_id": index, "weights_config": str(weights)}
        prepared_models.append(prepared)
        arms.append({"id": f"{model_id}/rho-pace-unit-h300", "model_id": model_id,
                     "condition": "rho-pace",
                     "label": f"RHO-PACE, unit prior, H=300, K={slow_update_cycles}",
                     "result_path": str(directory / "result.json"),
                     "weights_journal_path": str(directory / "weights.jsonl"),
                     "configuration_audit_path": str(directory / "configuration-audit.json"),
                     "checkpoint_directory": str(directory / "checkpoints"), "source": "new"})
    manifest = {"schema_version": 1, "campaign_id": run_directory.name, "resistance_nm": .15,
                "cycles": 3000, "models": prepared_models, "arms": arms,
                "protocol": {"only_new_arms": "RHO-PACE unit prior H300", "same_seed": True,
                             "same_ding_parameters": True, "same_solver": "IPOPT/MA57",
                             "no_FHO_data": True, "slow_update_cycles": slow_update_cycles,
                             "projection_budget_fraction": projection_budget_fraction,
                             "fast_rho_cpu_ids": [0, 1, 2, 3],
                             "projection_worker_cpu_ids": list(range(worker_cpu_offset, worker_cpu_offset + 4))}}
    path = run_directory / "manifest.json"
    _new_json(path, manifest)
    return path


def _run_model(model: dict, *, manifest: dict, python: Path, profile: Path, hsl_library: Path) -> str:
    arm = next(item for item in manifest["arms"] if item["id"] == f"{model['model_id']}/rho-pace-unit-h300")
    result = Path(arm["result_path"])
    if result.is_file():
        return model["model_id"]
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "IPOPT_HSL_LIBRARY": str(hsl_library)})
    args = _base_benchmark_args(profile=profile, resistance_nm=.15, cycles=3000,
                                hsl_library=hsl_library, output=result, seed=Path(model["seed"]))
    command = ["taskset", "--cpu-list", str(model["cpu_id"]), str(python),
               "scripts/run_configured_cycling_benchmark.py", "--model-config", model["model_config"],
               "--condition", "rho-pace", "--weights-config", model["weights_config"],
               "--weights-journal", arm["weights_journal_path"], "--configuration-audit",
               arm["configuration_audit_path"], "--checkpoint-every", "20", "--checkpoint-directory",
               arm["checkpoint_directory"], "--", *args]
    _execute(command, environment=env, log=result.parent / "launcher-output.log")
    return model["model_id"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--reduced-profile", required=True, type=Path)
    parser.add_argument("--hsl-library", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--slow-update-cycles", type=int, default=60)
    parser.add_argument("--projection-budget-fraction", type=float, default=.9)
    parser.add_argument("--worker-cpu-offset", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    manifest_path = args.run_directory.expanduser().resolve() / "manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            parser.error(f"--resume requires {manifest_path}")
    else:
        manifest_path = prepare(
            source_manifest=args.source_manifest, run_directory=args.run_directory,
            slow_update_cycles=args.slow_update_cycles,
            projection_budget_fraction=args.projection_budget_fraction,
            worker_cpu_offset=args.worker_cpu_offset,
        )
    print(f"Campaign: {manifest_path}", flush=True)
    if args.prepare_only:
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(_run_model, model, manifest=manifest, python=args.python.resolve(),
                            profile=args.reduced_profile.resolve(), hsl_library=args.hsl_library.resolve()): model["model_id"]
                for model in manifest["models"]}
        for job in as_completed(jobs):
            try:
                print(f"Completed: {job.result()}", flush=True)
            except BaseException as error:
                errors.append(f"{jobs[job]}: {type(error).__name__}: {error}")
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
