#!/usr/bin/env python3
"""Prepare and run an optional causal G->D torque-transfer RHO benchmark.

This is a deliberately conservative *branch* benchmark.  Each replication
starts from a byte-pinned, certified bilateral RHO checkpoint, obtains a
policy decision from its archived boundary state, applies either the declared
left relief for a fixed number of cycles or a hold, then returns explicitly to
the equal split and continues to the frozen-feasibility endpoint.  The
baseline and policy branches are independent fresh processes.

It reuses the audited worker of ``run_causal_torque_pilot``: Radau-5, 30 Hz,
IPOPT/MA57, unit muscle weights, complete discrete-NLP audits, frozen retries,
and fatigue-rest counterfactual endpoint labels are therefore not duplicated
here.  This protocol is experimental and has no clinical activation claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cocofest.cycling.slow.causal_transfer import (
    BicepsReservePolicy,
    CausalTransferSupervisor,
    TransferConfig,
    biceps_reserve_observation,
)
from cocofest.cycling.slow.oracle import load_certified_checkpoint
from scripts.slow_layer import run_causal_torque_pilot as runner


DEFAULT_RECEIPTS = (
    ROOT / "asymmetric-sides-r192-rho-bo-20260928/factorial-weights-split-20261003/unit_fixed/results/checkpoints/cycle-120/receipt.json",
    ROOT / "asymmetric-sides-r192-rho-bo-20260928/factorial-weights-split-20261003/unit_fixed/results/checkpoints/cycle-140/receipt.json",
)


def _biceps_a_scale(model_path: Path, expected_sha256: str) -> float:
    """Read the declared Biceps scale from the byte-pinned model config."""
    model_path = Path(model_path).resolve(strict=True)
    if runner.digest(model_path) != expected_sha256:
        raise ValueError("Model configuration differs from the certified checkpoint")
    try:
        value = json.loads(model_path.read_text(encoding="utf-8"))["muscles"]["Biceps"]["a_scale"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Certified model configuration lacks Biceps.a_scale") from error
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Certified Biceps.a_scale must be finite and positive")
    return float(value)


def _fast_state_gate(path: Path | None, anchor: dict, *, experimental_unvalidated_override: bool = False) -> dict:
    """Return a checkpoint/model-bound fast-state gate declaration.

    An absent receipt is a valid *abstention* state.  A gate is normally never
    enabled by a command-line boolean: a receipt identifies this exact
    bilateral checkpoint and the two model hashes.  The sole exception is an
    explicit offline experimental override.  It is labelled as unvalidated in
    every output and can never enable clinical activation.
    """
    if path is not None and experimental_unvalidated_override:
        raise ValueError("Choose a validated fast-state receipt or the experimental override, not both")
    if experimental_unvalidated_override:
        return {"policy_gate_open": True, "fast_state_status": "unvalidated_experimental_override",
                "warning": "EXPERIMENTAL ONLY: fast/history-state closure has not been validated; this permits an offline branch benchmark only."}
    if path is None:
        return {"policy_gate_open": False, "fast_state_status": "not_validated",
                "reason": "fast_state_gate_receipt_not_supplied"}
    path = Path(path).resolve(strict=True)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Fast-state gate receipt is not JSON") from error
    if document.get("schema") != "causal-transfer-fast-state-gate-v1" or document.get("status") != "validated":
        raise ValueError("Fast-state gate receipt lacks validated v1 schema")
    rows = document.get("anchors")
    if not isinstance(rows, list):
        raise ValueError("Fast-state gate receipt lacks anchor records")
    matches = [row for row in rows if isinstance(row, dict) and row.get("receipt_sha256") == anchor["receipt_sha256"]]
    if len(matches) != 1:
        raise ValueError("Fast-state gate receipt lacks this exact checkpoint")
    row = matches[0]
    expected_models = {side: anchor["arms"][side]["model_sha256"] for side in ("left", "right")}
    if row.get("model_sha256") != expected_models or row.get("fast_state_gate") is not True:
        raise ValueError("Fast-state gate record differs from the certified bilateral model pair")
    # A v1 receipt can certify only the narrowly preregistered paired
    # robustness check.  It must never be re-described as Markov closure.
    paired_local = document.get("validation_scope") == "paired_local_robustness_v1_not_markov_closure"
    return {"policy_gate_open": True,
            "fast_state_status": ("validated_paired_local_robustness_not_closure" if paired_local
                                  else "validated_evidence"),
            "receipt": str(path), "receipt_sha256": runner.digest(path), "record": row,
            "closure_established": False if paired_local else None}


def _verified_anchor(receipt: Path, policy: BicepsReservePolicy, *, transfer_config: TransferConfig = TransferConfig(),
                     fast_gate_receipt: Path | None = None,
                     experimental_unvalidated_override: bool = False) -> tuple[dict, dict]:
    """Load only a certified checkpoint and bind a policy receipt to it."""
    from cocofest.simulation.independent_arms_process import _driver_arguments
    from scripts.probe_independent_rho_task_reserve import _load_arm

    receipt = Path(receipt).resolve(strict=True)
    anchor = {"receipt": str(receipt), "receipt_sha256": runner.digest(receipt), "arms": {}}
    states, biceps_scales = {}, {}
    for side in ("left", "right"):
        source, payload = _load_arm(receipt, side)
        source.verify_files()
        args = _driver_arguments(payload, side)
        weights = payload["muscle_pace"].get(f"{side}_initial_weights")
        if (args.collocation_degree != 5 or args.stimulations_per_cycle != 30
                or args.ipopt_linear_solver != "ma57" or payload["resistance_pace"]["capacity_feedback"]
                or payload["muscle_pace"]["adaptation_enabled"]
                or weights != dict.fromkeys(runner.NAMES, 1.)
                or not math.isclose(payload[f"{side}_equivalent_mean_torque_nm"],
                                    getattr(transfer_config, f"baseline_{side}_nm"))):
            raise ValueError("Anchor differs from the fixed unit-weight Radau-5/30 Hz IPOPT-MA57 baseline")
        state = load_certified_checkpoint(receipt, side=side, muscles=runner.NAMES)
        states[side] = state
        biceps_scales[side] = _biceps_a_scale(Path(source.model_path), source.model_sha256)
        anchor["cycle"] = source.completed_cycles
        anchor["arms"][side] = {"archive_sha256": source.archive_sha256,
                                "prepared_problem_sha256": source.prepared_problem_sha256,
                                "model_sha256": source.model_sha256,
                                "model_path": source.model_path,
                                "biceps_a_scale": biceps_scales[side]}
    if len({state.completed_cycles for state in states.values()}) != 1:
        raise ValueError("A causal decision requires synchronized bilateral checkpoints")
    gate = _fast_state_gate(fast_gate_receipt, anchor,
                            experimental_unvalidated_override=experimental_unvalidated_override)
    observation = biceps_reserve_observation(states["left"], states["right"],
                                             left_a_scale=biceps_scales["left"],
                                             right_a_scale=biceps_scales["right"],
                                             fast_state_gate=gate["policy_gate_open"])
    supervisor = CausalTransferSupervisor(policy, transfer_config)
    decision = supervisor.decide(observation)
    return anchor, {"policy": {"kind": "biceps_reserve_v1", **asdict(policy)},
                    "fast_state_gate": gate,
                    "observation": {"completed_cycles": observation.completed_cycles,
                                    "provenance_sha256": observation.provenance_sha256,
                    "metrics": observation.metrics},
                    "decision": {**asdict(decision), "receipt_sha256": decision.receipt_sha256}}


def make_jobs(anchors: list[dict], decisions: list[dict], *, replicates: int,
              transfer_config: TransferConfig = TransferConfig()) -> list[dict]:
    """Pair fresh equal-split controls with an independently sealed policy arm."""
    if type(replicates) is not int or replicates < 2:
        raise ValueError("At least two independent control/policy replications are required")
    if len(anchors) != len(decisions):
        raise ValueError("Every anchor must have exactly one policy decision")
    jobs = []
    for anchor, record in zip(anchors, decisions, strict=True):
        decision = record["decision"]
        if decision["phase"] not in {"transfer", "hold"}:
            raise ValueError("Initial policy decision must be transfer or hold")
        delta = float(decision["left_torque_nm"]) - transfer_config.baseline_left_nm
        if not math.isclose(float(decision["right_torque_nm"]), transfer_config.baseline_right_nm - delta,
                              abs_tol=1e-12):
            raise ValueError("Policy decision does not preserve the declared total torque")
        for replicate in range(1, replicates + 1):
            prefix = f"c{anchor['cycle']}-rep{replicate}"
            jobs.append({"id": f"{prefix}-baseline", "anchor": anchor, "is_control": True,
                         "delta_left_nm": 0., "intervention_cycles": 10,
                         "baseline_left_nm": transfer_config.baseline_left_nm,
                         "baseline_right_nm": transfer_config.baseline_right_nm,
                         "policy_decision": {"phase": "baseline", "reason": "paired_equal_split_control"}})
            jobs.append({"id": f"{prefix}-causal", "anchor": anchor, "is_control": False,
                         "delta_left_nm": delta, "intervention_cycles": 10,
                         "baseline_left_nm": transfer_config.baseline_left_nm,
                         "baseline_right_nm": transfer_config.baseline_right_nm,
                         "policy_decision": record})
    return jobs


def preflight(receipts: list[Path], *, transfer_config: TransferConfig = TransferConfig(), fast_gate_receipt: Path | None = None,
              experimental_unvalidated_override: bool = False) -> list[dict]:
    """Read certified c-states and report action/abstention without solving."""
    policy = BicepsReservePolicy()
    return [{"anchor": anchor, "policy_receipt": decision}
            for anchor, decision in (_verified_anchor(receipt, policy, transfer_config=transfer_config,
                                                       fast_gate_receipt=fast_gate_receipt,
                                                       experimental_unvalidated_override=experimental_unvalidated_override)
            for receipt in receipts)]


def prepare(output: Path, receipts: list[Path], *, transfer_config: TransferConfig = TransferConfig(), fast_gate_receipt: Path | None = None,
            experimental_unvalidated_override: bool = False, maximum_cycles: int = 240,
            replicates: int = 2) -> Path:
    """Seal the fixed Biceps-reserve benchmark without launching it."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if maximum_cycles <= 10:
        raise ValueError("maximum_cycles must exceed the intervention duration")
    policy = BicepsReservePolicy()
    loaded = [_verified_anchor(receipt, policy, transfer_config=transfer_config, fast_gate_receipt=fast_gate_receipt,
                               experimental_unvalidated_override=experimental_unvalidated_override)
              for receipt in receipts]
    anchors, decisions = map(list, zip(*loaded, strict=True))
    if len({a["cycle"] for a in anchors}) != len(anchors):
        raise ValueError("Distinct certified checkpoint cycles required")
    if any(maximum_cycles <= a["cycle"] + 10 for a in anchors):
        raise ValueError("Endpoint must exceed each intervention window")
    statuses = {decision["fast_state_gate"]["fast_state_status"] for decision in decisions}
    if len(statuses) != 1:
        raise ValueError("Every branch must use one unambiguous fast-state gate status")
    fast_state_status, = statuses
    jobs = make_jobs(anchors, decisions, replicates=replicates, transfer_config=transfer_config)
    sources = sorted(set(ROOT.glob("cocofest/**/*.py")) | {
        Path(__file__).resolve(), Path(runner.__file__).resolve(),
        ROOT / "scripts/run_unilateral_continuation_labels.py",
        ROOT / "scripts/run_checkpoint_fatigue_ablation.py",
        ROOT / "scripts/run_local_task_reserve_branch.py",
        ROOT / "scripts/probe_independent_rho_task_reserve.py",
        ROOT / "scripts/validate_independent_rho_checkpoint.py",
        ROOT / "examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py",
    })
    plan = {
        "schema_version": 1,
        "kind": "optional_causal_left_to_right_transfer_rho_benchmark",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha256": {str(path): runner.digest(path) for path in sources},
        "jobs": jobs,
        "policy_receipts": decisions,
        "policy": {"kind": "biceps_reserve_v1", **asdict(policy)},
        "fast_state_status": fast_state_status,
        "maximum_absolute_cycles": maximum_cycles,
        "numeric_threads": 1,
        "audit_tolerance": 1e-5,
        "weights": dict.fromkeys(runner.NAMES, 1.),
        "transfer_config": asdict(transfer_config),
        "intervention": ("If and only if the sealed Biceps A/a_scale policy triggers (left <= 0.62, right >= 0.78 and validated fast-state gate): "
                         f"left={transfer_config.transfer[0]:.12g} Nm, right={transfer_config.transfer[1]:.12g} Nm for "
                         f"{transfer_config.action_cycles} complete cycles; total remains {sum(transfer_config.baseline):.12g} Nm."),
        "continuation_policy": {"kind": "fixed_equal_split", "left_nm": transfer_config.baseline_left_nm,
            "right_nm": transfer_config.baseline_right_nm,
            "switch": (f"Immediately after the ten-cycle action, explicitly return to "
                       f"{transfer_config.baseline_left_nm:.12g}/{transfer_config.baseline_right_nm:.12g} Nm until endpoint."),
            "scope": "experimental branch benchmark; no clinical ramp/comfort claim"},
        "endpoint_protocol": "Inherited frozen retries, zero-objective feasibility and fatigue-rest counterfactual; a solver status alone is not an endpoint.",
        "control_gate": "Every paired fresh equal-split control must have an attributable frozen endpoint before policy branches launch.",
        "resolution": "The per-anchor repeated-control endpoint spread is the numerical resolution; equal-or-smaller treatment differences are ties.",
        "activation_allowed": False,
    }
    if fast_state_status == "unvalidated_experimental_override":
        plan["warning"] = "EXPERIMENTAL ONLY: fast/history-state closure has not been validated. Offline benchmark results cannot support clinical activation."
    output.mkdir(parents=True)
    manifest = output / "manifest.json"
    runner.write_json(manifest, plan)
    runner.write_json(output / "manifest-seal.json", {"manifest_sha256": runner.digest(manifest)})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "prepare", "campaign"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--receipts", nargs="+", type=Path, default=list(DEFAULT_RECEIPTS))
    parser.add_argument("--fast-state-gate-receipt", type=Path,
                        help="Validated v1 gate receipt bound to every requested checkpoint/model pair.")
    parser.add_argument("--experimental-unvalidated-fast-state-override", action="store_true",
                        help="EXPERIMENTAL ONLY: open the fast-state gate for an offline benchmark; recorded as unvalidated and never activation-eligible.")
    parser.add_argument("--maximum-cycles", type=int, default=240)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--baseline-left-nm", type=float, default=.96,
                        help="Sealed equal-split left torque at every source checkpoint.")
    parser.add_argument("--baseline-right-nm", type=float, default=.96,
                        help="Sealed equal-split right torque at every source checkpoint.")
    parser.add_argument("--left-relief-nm", type=float, default=.10,
                        help="Sealed one-shot left-to-right transfer magnitude.")
    parser.add_argument("--action-cycles", type=int, default=10,
                        help="Sealed duration of the one-shot transfer.")
    parser.add_argument("--cpus", nargs="+", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()
    transfer_config = TransferConfig(baseline_left_nm=args.baseline_left_nm,
                                     baseline_right_nm=args.baseline_right_nm,
                                     left_relief_nm=args.left_relief_nm,
                                     action_cycles=args.action_cycles)
    if args.mode == "prepare":
        if args.manifest is None:
            parser.error("--manifest is required for prepare")
        if args.manifest.name != "manifest.json":
            parser.error("Manifest filename must be manifest.json")
        if args.experimental_unvalidated_fast_state_override:
            print("WARNING: creating an unvalidated experimental override; no clinical activation is allowed.",
                  file=sys.stderr)
        print(prepare(args.manifest.parent, args.receipts, transfer_config=transfer_config,
                      fast_gate_receipt=args.fast_state_gate_receipt,
                      experimental_unvalidated_override=args.experimental_unvalidated_fast_state_override,
                      maximum_cycles=args.maximum_cycles, replicates=args.replicates))
    elif args.mode == "campaign":
        if args.manifest is None:
            parser.error("--manifest is required for campaign")
        runner.campaign(args.manifest, args.cpus, args.timeout_seconds, args.controls_only)
    else:
        if args.experimental_unvalidated_fast_state_override:
            print("WARNING: preflight uses an unvalidated experimental fast-state override.", file=sys.stderr)
        print(json.dumps(preflight(args.receipts, transfer_config=transfer_config,
                                   fast_gate_receipt=args.fast_state_gate_receipt,
                                   experimental_unvalidated_override=args.experimental_unvalidated_fast_state_override), indent=2,
                         sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
