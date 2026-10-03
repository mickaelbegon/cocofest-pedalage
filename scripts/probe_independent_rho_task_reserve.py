#!/usr/bin/env python3
"""Probe witnessed one-cycle work reserve from a certified arm checkpoint.

Each work factor rebuilds a fresh full RHO NLP, restores the same prepared
state/history, changes only terminal E_prod, and solves exactly one cycle.
Failures remain indeterminate; the reported reserve is an observed lower
bound from independently audited trajectories, never a maximum-load proof.
"""

from __future__ import annotations

import argparse
from copy import copy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.task_reserve import TaskReserveCheckpoint, evaluate_work_reserve
from cocofest.optimization.task_reserve_probe_adapter import (
    independent_full_nlp_audit, probe_one_cycle,
)
from cocofest.simulation.independent_arms_process import (
    _atomic_json, _configured_payload_model, _driver_arguments,
)
from scripts.validate_independent_rho_checkpoint import _object, _source_path


def _load_arm(receipt_path: Path, side: str) -> tuple[TaskReserveCheckpoint, dict[str, Any]]:
    receipt_path = Path(receipt_path).resolve(strict=True)
    receipt = _object(receipt_path)
    if (receipt.get("schema_version") != 1 or receipt.get("checkpoint_kind") != "certified_bilateral_shifted_primal"
            or receipt.get("exact_bilateral_restart") is not True):
        raise ValueError("A certified exact bilateral restart receipt is required")
    cycle = receipt.get("completed_cycles")
    if type(cycle) is not int or cycle < 1:
        raise ValueError("Receipt lacks a positive completed cycle count")
    configuration = _source_path(receipt_path, receipt.get("configuration_path"))
    if sha256(configuration.read_bytes()).hexdigest() != receipt.get("configuration_sha256"):
        raise ValueError("Receipt configuration digest mismatch")
    arm = receipt.get("arms", {}).get(side)
    if not isinstance(arm, dict) or arm.get("certified") is not True or arm.get("replay_roundtrip_exact") is not True:
        raise ValueError(f"Receipt lacks a certified exact {side} restart")
    archive = _source_path(receipt_path, arm.get("primal_path"))
    model = _source_path(receipt_path, arm.get("model_path"))
    if (sha256(archive.read_bytes()).hexdigest() != arm.get("sha256")
            or sha256(model.read_bytes()).hexdigest() != arm.get("model_sha256")):
        raise ValueError("Receipt archive/model byte digest mismatch")
    payload = _object(configuration)
    configured = _configured_payload_model(payload, side)
    if configured is None or configured[0] != model:
        raise ValueError("Receipt model is different from the runner configuration")
    args = _driver_arguments(payload, side)
    if args.solver != "ipopt" or args.formulation != "isokinetic" or args.cycles_per_window != 1:
        raise ValueError("Task-reserve probe requires the same one-cycle IPOPT isokinetic RHO")
    with np.load(archive, allow_pickle=False) as prepared:
        bounds_min = np.asarray(prepared["problem__x_bounds:E_prod:min"], dtype=float)
        bounds_max = np.asarray(prepared["problem__x_bounds:E_prod:max"], dtype=float)
        if bounds_min.shape != bounds_max.shape or bounds_min.shape[1] < 3:
            raise ValueError("Prepared E_prod bounds are incomplete")
        nominal_work = float(bounds_min[0, 2])
        if not np.isfinite(nominal_work) or nominal_work <= 0 or bounds_max[0, 2] != nominal_work:
            raise ValueError("Prepared E_prod terminal work equality is invalid")
    context = {
        "configuration_sha256": receipt["configuration_sha256"], "side": side,
        "nominal_work_j": nominal_work,
        "solver": args.solver, "ode_solver": args.ode_solver,
        "collocation_degree": args.collocation_degree,
        "stimulations_per_cycle": args.stimulations_per_cycle,
        "isokinetic_omega": args.isokinetic_omega,
    }
    source = TaskReserveCheckpoint.from_archive(archive, completed_cycles=cycle,
                                                model_path=model, task_context=context)
    if (source.prepared_problem_sha256 != arm.get("prepared_problem_sha256")
            or source.prepared_problem_sha256 != arm.get("restored_problem_sha256")):
        raise ValueError("Receipt prepared/restored problem digests differ")
    return source, payload


def run(receipt_path: Path, *, side: str, work_scales: list[float], output_directory: Path,
        tolerance: float = 1e-5) -> Path:
    source, payload = _load_arm(receipt_path, side)
    return run_from_checkpoint(source, payload, side=side, work_scales=work_scales,
                               output_directory=output_directory, tolerance=tolerance)


def run_from_checkpoint(source: TaskReserveCheckpoint, payload: dict[str, Any], *, side: str,
                        work_scales: list[float], output_directory: Path,
                        tolerance: float = 1e-5) -> Path:
    """Probe a checkpoint already authenticated by its source-specific loader."""
    if json.loads(source.task_context_json).get("side") != side:
        raise ValueError("Probe checkpoint side differs from the requested side")
    source.verify_files()
    output_directory = Path(output_directory).resolve()
    result_path = output_directory / f"{side}-task-reserve.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing probe result: {result_path}")

    def build_runtime():
        from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver
        from cocofest.optimization.configured_cycling_model import configured_model_factories

        args = _driver_arguments(payload, side)
        args = copy(args)
        args.nlp_ipopt_recovery = False
        args.nlp_ipopt_recovery_ma57_tuned = False
        configured = _configured_payload_model(payload, side)
        if configured is None or configured[0] != Path(source.model_path):
            raise ValueError("Configured model changed before fresh build")
        with configured_model_factories(configured[1], []):
            return driver.build_unilateral_runtime(args, echo=False)

    def solve_one_cycle(program, solver):
        from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver
        # Bypass NMPC's advancing loop: this is one frozen, prepared OCP.
        return super(driver.RecedingHorizonOptimization, program).solve(
            solver=solver, warm_start=None)

    def save_witness(solution, program, request, audit):
        from bioptim import SolutionMerge
        factor = f"{request.work_scale:.16g}".replace(".", "p").replace("+", "plus")
        path = output_directory / f"{side}-work-scale-{factor}.npz"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing witness: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        states = solution.decision_states(to_merge=SolutionMerge.NODES)
        controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
        metadata = {"source_archive_sha256": request.checkpoint.archive_sha256,
                    "prepared_problem_sha256": request.checkpoint.prepared_problem_sha256,
                    "work_scale": request.work_scale, "solver_status": str(solution.status),
                    "independent_audit": dict(audit.detail)}
        np.savez_compressed(path, **{f"states__{key}": np.asarray(value, float) for key, value in states.items()},
                            **{f"controls__{key}": np.asarray(value, float) for key, value in controls.items()},
                            vector=np.asarray(solution.vector, float),
                            metadata__json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
        return str(path)

    estimate = evaluate_work_reserve(source, work_scales,
        lambda request: probe_one_cycle(request, build_runtime=build_runtime,
            solve_one_cycle=solve_one_cycle,
            audit=lambda solution, program, target, tol: independent_full_nlp_audit(
                solution, program, target_work_j=target, tolerance=tol),
            save_witness=save_witness, tolerance=tolerance))
    _atomic_json(result_path, {**asdict(estimate),
        "interpretation": "observed feasible one-cycle work factors only; failed solves are indeterminate",
        "independent_audit_scope": "complete discrete NLP constraint/decision vectors; no continuous-ODE replay"})
    return result_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--side", choices=("right", "left"), required=True)
    parser.add_argument("--work-scales", type=float, nargs="+", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args(argv)
    print(run(args.receipt, side=args.side, work_scales=args.work_scales,
              output_directory=args.output_directory, tolerance=args.tolerance))


if __name__ == "__main__":
    main()
