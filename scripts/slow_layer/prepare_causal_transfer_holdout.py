#!/usr/bin/env python3
"""Preregister source RHO and no-recalibration causal-transfer holdouts.

``baseline`` writes (but never launches) the fixed unit-weight source RHO for
one out-of-distribution condition.  After that RHO has produced certified
prepared checkpoints, ``preflight`` exposes the inherited policy decisions
and ``seal-branches`` selects a predeclared hold/trigger pair without changing
the threshold (.62), dose (.10 Nm), or duration (10 cycles).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cocofest.cycling.slow.causal_transfer import TransferConfig
from cocofest.cycling.slow.causal_transfer_holdout import baseline_payload, standard_holdouts, validate_baseline_runtime
from scripts.slow_layer import run_causal_torque_pilot as runner
from scripts.slow_layer import run_causal_transfer_supervisor as supervisor


def _condition(identifier: str):
    try:
        return standard_holdouts(ROOT)[identifier]
    except KeyError as error:
        raise ValueError(f"Unknown holdout {identifier!r}; choose one of {sorted(standard_holdouts(ROOT))}") from error


def _sources(condition) -> dict[str, str]:
    paths = set(ROOT.glob("cocofest/**/*.py")) | {Path(__file__).resolve(), Path(runner.__file__).resolve(), Path(supervisor.__file__).resolve(),
             ROOT / "scripts/run_independent_isokinetic_arms.py", ROOT / "scripts/validate_independent_rho_checkpoint.py",
             Path(condition.left_model), Path(condition.right_model)}
    return {str(path): runner.digest(path) for path in sorted(paths)}


def _write_baseline(output: Path, condition) -> Path:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    condition.validate()
    output.mkdir(parents=True)
    payload = baseline_payload(condition)
    runtime = validate_baseline_runtime(payload)
    source = {"schema": "causal-transfer-holdout-source-v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "condition": condition.canonical_dict(), "code_sha256": _sources(condition),
              "numerical_runtime": runtime,
              "baseline_output_directory": str(output / "results"),
              "baseline_command": [sys.executable, str(ROOT / "scripts/run_independent_isokinetic_arms.py"),
                                   "--config", str(output / "baseline-input.json"), "--output", str(output / "results")],
              "checkpoint_admission": {
                  "required": "certified_bilateral_shifted_primal receipt.json after two fresh-worker restores",
                  "rejected": "prepared-export.json alone; any mismatched model/configuration/runtime digest",
                  "validator": str(ROOT / "scripts/validate_independent_rho_checkpoint.py"),
              },
              "endpoint_requirement": "The source RHO must preserve frozen retries, zero-objective feasibility and fatigue-rest counterfactuals. A raw IPOPT status is never an endpoint.",
              "selection_rule": "Use the last available certified hold checkpoint plus the first available certified trigger checkpoint. If no trigger occurs, retain one hold-only falsification branch; do not tune threshold/dose/duration."}
    runner.write_json(output / "baseline-input.json", payload)
    runner.write_json(output / "baseline-plan.json", source)
    runner.write_json(output / "baseline-plan-seal.json", {"baseline_plan_sha256": runner.digest(output / "baseline-plan.json"),
                                                              "baseline_input_sha256": runner.digest(output / "baseline-input.json")})
    return output / "baseline-input.json"


def _authenticated_baseline(directory: Path, identifier: str):
    directory = Path(directory).resolve(strict=True)
    plan_path, seal_path = directory / "baseline-plan.json", directory / "baseline-plan-seal.json"
    plan, seal = json.loads(plan_path.read_text()), json.loads(seal_path.read_text())
    if plan.get("schema") != "causal-transfer-holdout-source-v1" or seal.get("baseline_plan_sha256") != runner.digest(plan_path):
        raise ValueError("Source holdout plan is absent or changed after preregistration")
    condition = _condition(identifier)
    if plan.get("condition") != condition.canonical_dict():
        raise ValueError("Requested holdout differs from its preregistered source condition")
    if seal.get("baseline_input_sha256") != runner.digest(directory / "baseline-input.json"):
        raise ValueError("Baseline input changed after preregistration")
    payload = json.loads((directory / "baseline-input.json").read_text())
    if plan.get("numerical_runtime") != validate_baseline_runtime(payload):
        raise ValueError("Pinned MA57 runtime differs from the preregistered holdout source")
    for path, expected in plan["code_sha256"].items():
        if runner.digest(path) != expected:
            raise ValueError(f"Source holdout dependency changed: {path}")
    return condition, plan


def _receipts(directory: Path, condition) -> list[Path]:
    root = Path(directory) / "results" / "checkpoints"
    receipts = [root / f"cycle-{cycle}" / "receipt.json" for cycle in condition.checkpoint_cycles]
    return [path for path in receipts if path.is_file()]


def _config(condition) -> TransferConfig:
    # The equal split changes only because the total work is a holdout.  The
    # policy's trigger, signed dose, and duration remain the calibration values.
    return TransferConfig(baseline_left_nm=condition.baseline_per_arm_nm,
                          baseline_right_nm=condition.baseline_per_arm_nm,
                          left_relief_nm=condition.left_relief_nm,
                          action_cycles=condition.action_cycles)


def inspect(directory: Path, identifier: str, *, fast_gate_receipt: Path | None = None,
            experimental_override: bool = False) -> list[dict]:
    condition, _ = _authenticated_baseline(directory, identifier)
    receipts = _receipts(directory, condition)
    if not receipts:
        raise FileNotFoundError("No certified planned checkpoint receipt yet; run the sealed baseline RHO first")
    rows = supervisor.preflight(receipts, transfer_config=_config(condition), fast_gate_receipt=fast_gate_receipt,
                               experimental_unvalidated_override=experimental_override)
    return [{"cycle": row["anchor"]["cycle"], "receipt": row["anchor"]["receipt"],
             "decision": row["policy_receipt"]["decision"], "observation": row["policy_receipt"]["observation"],
             "fast_state_gate": row["policy_receipt"]["fast_state_gate"]} for row in rows]


def _select(rows: list[dict]) -> list[dict]:
    """Select, without peeking at endpoints, a bracketing hold/trigger pair."""
    holds = [row for row in rows if row["decision"]["phase"] == "hold"]
    transfers = [row for row in rows if row["decision"]["phase"] == "transfer"]
    if transfers:
        selected = [transfers[0]]
        before = [row for row in holds if row["cycle"] < transfers[0]["cycle"]]
        if before:
            selected.insert(0, before[-1])
        return selected
    if not holds:
        raise ValueError("No valid inherited-policy hold or trigger decision in certified checkpoints")
    # A non-trigger holdout is scientifically useful: it tests that an action
    # calibrated for a weak left arm does not appear merely by cycle number.
    return [holds[-1]]


def seal_branches(directory: Path, identifier: str, output: Path, *, fast_gate_receipt: Path | None = None,
                  experimental_override: bool = False, replicates: int = 2) -> Path:
    if not fast_gate_receipt and not experimental_override:
        raise ValueError("A validated fast-state receipt or explicit experimental override is required")
    condition, baseline_plan = _authenticated_baseline(directory, identifier)
    rows = inspect(directory, identifier, fast_gate_receipt=fast_gate_receipt, experimental_override=experimental_override)
    selected = _select(rows)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    selection = {"schema": "causal-transfer-holdout-selection-v1", "source_baseline_plan": str(Path(directory).resolve() / "baseline-plan.json"),
                 "source_baseline_plan_sha256": runner.digest(Path(directory) / "baseline-plan.json"),
                 "condition": condition.canonical_dict(), "transfer_config": asdict(_config(condition)),
                 "threshold_policy": {"left_activate_at_or_below": .62, "right_recipient_at_or_above": .78},
                 "selected": selected, "selection_rule": baseline_plan["selection_rule"],
                 "activation_allowed": False}
    runner.write_json(output / "selection.json", selection)
    manifest = supervisor.prepare(output / "branches", [Path(row["receipt"]) for row in selected],
                                  transfer_config=_config(condition), fast_gate_receipt=fast_gate_receipt,
                                  experimental_unvalidated_override=experimental_override,
                                  maximum_cycles=condition.maximum_cycles, replicates=replicates)
    runner.write_json(output / "holdout-seal.json", {"selection_sha256": runner.digest(output / "selection.json"),
                                                       "branches_manifest_sha256": runner.digest(manifest)})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "preflight", "seal-branches"))
    parser.add_argument("--condition", required=True, choices=sorted(standard_holdouts(ROOT)))
    parser.add_argument("--directory", type=Path, required=True,
                        help="Baseline directory for baseline, then the same sealed source directory for preflight/seal-branches.")
    parser.add_argument("--output", type=Path, help="New branch output directory, required by seal-branches.")
    parser.add_argument("--fast-state-gate-receipt", type=Path)
    parser.add_argument("--experimental-unvalidated-fast-state-override", action="store_true")
    parser.add_argument("--replicates", type=int, default=2)
    args = parser.parse_args()
    if args.mode == "baseline":
        print(_write_baseline(args.directory, _condition(args.condition)))
    elif args.mode == "preflight":
        print(json.dumps(inspect(args.directory, args.condition, fast_gate_receipt=args.fast_state_gate_receipt,
                                 experimental_override=args.experimental_unvalidated_fast_state_override), indent=2, sort_keys=True))
    else:
        if args.output is None:
            parser.error("--output is required for seal-branches")
        print(seal_branches(args.directory, args.condition, args.output,
                            fast_gate_receipt=args.fast_state_gate_receipt,
                            experimental_override=args.experimental_unvalidated_fast_state_override,
                            replicates=args.replicates))


if __name__ == "__main__":
    main()
