#!/usr/bin/env python3
"""Certify a bilateral prepared RHO checkpoint in two fresh solver workers.

The input is the provisional ``prepared-export.json`` written by the
independent-arm coordinator.  Each worker rebuilds its unilateral NLP, restores
the physical primal, bounds and fixed parameters, and recomputes the prepared
problem digest.  A final ``receipt.json`` is published only after both
independent restorations match their source archive.

No OCP is solved here.  This is deliberately a pre-continuation provenance
gate: a subsequent experiment may apply a candidate weight vector only after
this command succeeds.
"""

from __future__ import annotations

import argparse
from copy import copy
from hashlib import sha256
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.simulation.independent_arms_process import ARMS, _atomic_json, _configured_payload_model, _driver_arguments
from cocofest.simulation.rho_restart_checkpoint import restore_prepared_checkpoint


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _source_path(reference: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Checkpoint receipt has an empty path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else reference.parent / path).resolve()


def _load_prepared_export(run_directory: Path, completed_cycles: int) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    run_directory = Path(run_directory).resolve(strict=True)
    checkpoint_directory = run_directory / "checkpoints" / f"cycle-{completed_cycles}"
    export_path = checkpoint_directory / "prepared-export.json"
    export = _object(export_path)
    if (export.get("schema_version") != 1
            or export.get("checkpoint_kind") != "prepared_bilateral_shifted_primal_export"
            or export.get("completed_cycles") != completed_cycles
            or export.get("exact_bilateral_restart") is not False
            or export.get("fresh_worker_replay_verified") is not False):
        raise ValueError("Prepared export has the wrong kind, cycle, or verification state")
    configuration = _source_path(export_path, export.get("configuration_path"))
    if sha256(configuration.read_bytes()).hexdigest() != export.get("configuration_sha256"):
        raise ValueError("Prepared export configuration digest mismatch")
    arms = export.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("Prepared export must contain exactly right and left arms")
    for side in ARMS:
        arm = arms[side]
        if not isinstance(arm, dict) or arm.get("status") != "exported":
            raise ValueError(f"Prepared export has no usable {side} archive")
        archive = _source_path(export_path, arm.get("primal_path"))
        if not archive.is_file() or sha256(archive.read_bytes()).hexdigest() != arm.get("sha256"):
            raise ValueError(f"Prepared export {side} archive digest mismatch")
        model = _source_path(export_path, arm.get("model_path")) if arm.get("model_path") else None
        if model is None or not model.is_file() or sha256(model.read_bytes()).hexdigest() != arm.get("model_sha256"):
            raise ValueError(f"Prepared export {side} lacks a model-fingerprinted source")
    payload = _object(configuration)
    return export_path, export, payload


def _restore_worker(connection, side: str, payload: Mapping[str, Any], archive_path: str,
                    completed_cycles: int) -> None:
    """Build and restore in a spawned process; no live parent NLP is reused."""
    try:
        from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver

        args = _driver_arguments(payload, side)
        build_args = copy(args)
        build_args.nlp_ipopt_recovery = False
        build_args.nlp_ipopt_recovery_ma57_tuned = False
        configured = _configured_payload_model(payload, side)
        if configured is None:
            raise ValueError("Fresh checkpoint validation requires a configured model input")
        _model_path, model_config = configured
        from cocofest.optimization.configured_cycling_model import configured_model_factories

        builds = []
        with configured_model_factories(model_config, builds):
            runtime = driver.build_unilateral_runtime(build_args, echo=False)
        restored = restore_prepared_checkpoint(
            Path(archive_path), runtime["nmpc"], completed_cycles=completed_cycles
        )
        connection.send({"kind": "restored", "side": side, "restored": restored,
                         "model_config_path": str(_model_path), "model_builds": builds})
    except BaseException as error:
        connection.send({"kind": "error", "side": side,
                         "error": f"{type(error).__name__}: {error}",
                         "traceback": traceback.format_exc()})
    finally:
        connection.close()


def _receive_restores(connections, processes, *, timeout_seconds: float) -> dict[str, dict[str, Any]]:
    pending = dict(connections)
    reports = {}
    deadline = time.monotonic() + timeout_seconds
    while pending:
        ready = wait(list(pending.values()), timeout=max(0., deadline - time.monotonic()))
        if not ready:
            raise TimeoutError(f"Fresh-worker checkpoint validation timed out for {sorted(pending)}")
        for connection in ready:
            side = next(key for key, value in pending.items() if value == connection)
            try:
                report = connection.recv()
            except EOFError as error:
                raise RuntimeError(f"{side} worker exited without a restore report") from error
            del pending[side]
            if report.get("kind") != "restored":
                raise RuntimeError(f"{side} fresh restore failed: {report.get('error', report)}")
            reports[side] = report
    return reports


def _final_receipt(export_path: Path, export: Mapping[str, Any], reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Build the exact-restart receipt after independently restored digests match."""
    arms = {}
    for side in ARMS:
        source, report = export["arms"][side], reports[side]
        restored = report["restored"]
        if (restored.get("restored_problem_sha256") != source.get("prepared_problem_sha256")
                or restored.get("restored_primal_signature") != source.get("prepared_primal_signature")
                or restored.get("stimulation_history_complete") is not True):
            raise RuntimeError(f"{side} fresh worker did not reproduce its prepared RHO")
        model_path = _source_path(export_path, source["model_path"])
        arms[side] = {
            "completed_cycles": export["completed_cycles"], "certified": True,
            "stimulation_history_complete": True, "replay_roundtrip_exact": True,
            "primal_path": source["primal_path"], "sha256": source["sha256"],
            "model_path": str(Path(os.path.relpath(model_path, export_path.parent))),
            "model_sha256": source["model_sha256"],
            "prepared_problem_sha256": source["prepared_problem_sha256"],
            "restored_problem_sha256": restored["restored_problem_sha256"],
            "prepared_primal_signature": source["prepared_primal_signature"],
            "restored_primal_signature": restored["restored_primal_signature"],
            "fresh_worker_pid_verified": True,
        }
    return {
        "schema_version": 1,
        "checkpoint_kind": "certified_bilateral_shifted_primal",
        "exact_bilateral_restart": True,
        "completed_cycles": export["completed_cycles"],
        "configuration_path": export["configuration_path"],
        "configuration_sha256": export["configuration_sha256"],
        "arms": arms,
        "validation": {"mode": "two_spawned_fresh_workers", "source_export": "prepared-export.json"},
    }


def validate(run_directory: Path, completed_cycles: int, *, timeout_seconds: float = 900.) -> Path:
    export_path, export, payload = _load_prepared_export(run_directory, completed_cycles)
    receipt_path = export_path.with_name("receipt.json")
    if receipt_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing exact restart receipt: {receipt_path}")
    context = mp.get_context("spawn")
    parents, processes = {}, {}
    try:
        for side in ARMS:
            parent, child = context.Pipe(duplex=False)
            archive = _source_path(export_path, export["arms"][side]["primal_path"])
            process = context.Process(target=_restore_worker,
                args=(child, side, payload, str(archive), completed_cycles), name=f"cocofest-{side}-checkpoint-restore")
            process.start()
            child.close()
            parents[side], processes[side] = parent, process
        reports = _receive_restores(parents, processes, timeout_seconds=timeout_seconds)
        receipt = _final_receipt(export_path, export, reports)
        _atomic_json(receipt_path, receipt)
        return receipt_path
    finally:
        for connection in parents.values():
            connection.close()
        for process in processes.values():
            process.join(timeout=2.)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--completed-cycles", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.)
    args = parser.parse_args(argv)
    if args.completed_cycles < 1 or args.timeout_seconds <= 0:
        parser.error("completed-cycles and timeout-seconds must be positive")
    print(validate(args.run_directory, args.completed_cycles, timeout_seconds=args.timeout_seconds))


if __name__ == "__main__":
    main()
