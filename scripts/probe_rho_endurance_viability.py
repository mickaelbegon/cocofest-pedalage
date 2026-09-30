#!/usr/bin/env python3
"""Run a frozen-state one-cycle feasibility probe after an RHO endpoint.

The input checkpoint must be the shifted primal written *after* the final
certified RHO, not the failed solver iterate.  The probe keeps the load,
model, fatigue state, mechanical state and PW history fixed at that boundary;
only the next cycle's pulse-width allocation is optimized.

It never calls a failed local solve proof of muscle exhaustion.  See the JSON
report's ``scientific_scope`` before using it in an endurance comparison.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.endurance_viability import assess_frozen_state_one_cycle_probe


_VALUE_OPTIONS = {
    "--output-json",
    "--common-initial-solution",
    "--common-initial-solution-output",
    "--receding-horizon-solution-output",
    "--rho-replay-checkpoint-output",
    "--rho-prepared-checkpoint-output-template",
    "--rho-prepared-checkpoint-windows",
    "--n-windows",
    "--cycles-per-window",
    "--max-consecutive-failing",
}
_FLAG_OPTIONS = {
    "--common-initial-solution-recenter-first-node-bounds",
    "--common-initial-solution-feasibility-probe",
    "--allow-partial-receding-horizon-solution-output",
}


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return document


def _sole_solver_row(document: dict[str, Any]) -> dict[str, Any]:
    rows = document.get("results") or []
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("A viability probe requires exactly one solver result.")
    return rows[0]


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        if "metadata__json" not in archive.files:
            raise ValueError("Checkpoint has no metadata__json.")
        return json.loads(str(archive["metadata__json"].item()))


def build_probe_benchmark_arguments(
    source_arguments: list[str], *, checkpoint: Path, output_json: Path,
) -> list[str]:
    """Preserve source physics while replacing all trajectory/output controls."""

    cleaned: list[str] = []
    index = 0
    while index < len(source_arguments):
        token = source_arguments[index]
        if token in _VALUE_OPTIONS:
            if index + 1 >= len(source_arguments):
                raise ValueError(f"Source argument {token} has no value.")
            index += 2
            continue
        if token in _FLAG_OPTIONS:
            index += 1
            continue
        cleaned.append(token)
        index += 1
    return [
        *cleaned,
        "--n-windows", "1",
        "--cycles-per-window", "1",
        "--max-consecutive-failing", "1",
        "--common-initial-solution", str(checkpoint),
        "--common-initial-solution-recenter-first-node-bounds",
        "--common-initial-solution-feasibility-probe",
        "--output-json", str(output_json),
    ]


def run_probe(*, python: Path, model_config: Path, source_result: Path,
              source_configuration_audit: Path, checkpoint: Path,
              output_directory: Path) -> dict[str, Any]:
    """Execute a single configured RHO and return its non-overclaiming report."""

    for path in (python, model_config, source_result, source_configuration_audit, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_directory.exists():
        raise FileExistsError(f"Probe output directory must be fresh: {output_directory}")
    source_document = _read_json(source_result)
    source_row = _sole_solver_row(source_document)
    audit = _read_json(source_configuration_audit)
    source_arguments = audit.get("arguments")
    if not isinstance(source_arguments, list) or not all(isinstance(x, str) for x in source_arguments):
        raise ValueError("Source configuration audit has no reproducible argument list.")
    metadata = _checkpoint_metadata(checkpoint)
    completed = metadata.get("producer_completed_windows")
    if isinstance(completed, bool) or not isinstance(completed, int):
        raise ValueError("Checkpoint lacks an integer producer_completed_windows field.")
    if metadata.get("producer_mode") != "rho_replay_checkpoint":
        raise ValueError("Probe requires a rho_replay_checkpoint, not an arbitrary warm start.")

    output_directory.mkdir(parents=True)
    result_path = output_directory / "probe-result.json"
    audit_path = output_directory / "probe-configuration-audit.json"
    log_path = output_directory / "probe-launcher-output.log"
    benchmark_args = build_probe_benchmark_arguments(
        source_arguments, checkpoint=checkpoint.resolve(), output_json=result_path,
    )
    command = [
        str(python.resolve()), str(ROOT / "scripts" / "run_configured_cycling_benchmark.py"),
        "--model-config", str(model_config.resolve()), "--condition", "rho",
        "--configuration-audit", str(audit_path), "--", *benchmark_args,
    ]
    with log_path.open("x", encoding="utf-8") as stream:
        completed_process = subprocess.run(command, cwd=ROOT, stdout=stream,
                                           stderr=subprocess.STDOUT, text=True,
                                           check=False)
    probe_row = None
    if result_path.is_file():
        try:
            probe_row = _sole_solver_row(_read_json(result_path))
        except (OSError, ValueError, json.JSONDecodeError):
            probe_row = None
    report = assess_frozen_state_one_cycle_probe(
        source_row, probe_row, checkpoint_completed_windows=completed,
        launcher_returncode=completed_process.returncode,
    )
    report.update(
        source_result={"path": str(source_result.resolve()), "sha256": sha256(source_result.read_bytes()).hexdigest()},
        source_configuration_audit={"path": str(source_configuration_audit.resolve()), "sha256": sha256(source_configuration_audit.read_bytes()).hexdigest()},
        checkpoint={"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint.read_bytes()).hexdigest()},
        model_config={"path": str(model_config.resolve()), "sha256": sha256(model_config.read_bytes()).hexdigest()},
        probe_result_path=str(result_path),
        probe_configuration_audit_path=str(audit_path),
        launcher_log_path=str(log_path),
        command=command,
    )
    report_path = output_directory / "viability-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-configuration-audit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_probe(**vars(args))
    print(json.dumps({key: report.get(key) for key in ("status", "certificate_valid", "next_cycle_viable", "normalized_constraint_deficit")}, indent=2))


if __name__ == "__main__":
    main()
