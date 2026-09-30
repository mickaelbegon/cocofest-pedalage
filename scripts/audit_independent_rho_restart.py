#!/usr/bin/env python3
"""Audit whether a bilateral RHO result can support matched continuations.

The ordinary PACE-VR snapshot is a *projection* input.  In particular it is
not a native RHO restart checkpoint.  This audit reports what is present at a
certified pair boundary and refuses to infer a restart from terminal states.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ARMS = ("right", "left")
REQUIRED_RESTART_COMPONENTS = (
    "shifted_one_cycle_primal_per_arm",
    "first_node_and_terminal_bounds_per_arm",
    "stimulation_history_per_arm",
    "fixed_parameters_and_work_target_per_arm",
    "roundtrip_prepared_problem_digest_per_arm",
)


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def audit_directory(root: Path, completed_cycles: int) -> dict:
    """Inspect existing files only; never manufacture an exact-restart claim."""
    if type(completed_cycles) is not int or completed_cycles < 1:
        raise ValueError("completed_cycles must be a positive integer")
    root = Path(root).resolve(strict=True)
    configuration_path = root / "configuration.json"
    summary_path = root / "summary.json"
    _object(configuration_path)
    summary = _object(summary_path)
    if summary.get("architecture") != "two-process-unilateral-isokinetic-rho-pace":
        raise ValueError("Expected the two-process unilateral isokinetic RHO summary")
    rows = summary.get("cycles")
    if not isinstance(rows, list):
        raise ValueError("Summary has no cycle list")
    matches = [row for row in rows if row.get("cycle") == completed_cycles]
    if len(matches) != 1:
        raise ValueError(f"Cycle {completed_cycles} is absent or repeated")
    arms = matches[0].get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("The boundary has no exact right/left pair")
    per_arm = {}
    for side in ARMS:
        metrics = arms[side]
        if not isinstance(metrics, dict):
            raise ValueError(f"Missing {side} metrics")
        snapshot = metrics.get("pace_vr_snapshot")
        per_arm[side] = {
            "certified": metrics.get("certified") is True,
            "terminal_slow_state_present": isinstance(metrics.get("pace_vr_terminal_slow"), list),
            "pace_vr_projection_snapshot_present": isinstance(snapshot, dict),
            "weights_used": metrics.get("weights_used"),
            "equivalent_mean_torque_nm": metrics.get("equivalent_mean_torque_nm"),
            "configured_model_receipt_present": isinstance(
                summary.get("arms", {}).get(side, {}).get("configured_model"), dict),
        }
    pair_certified = all(per_arm[side]["certified"] for side in ARMS)
    # A future producer may publish a receipt. Reuse the same content and
    # provenance gate as the ranking validator, never trust a Boolean here.
    receipt_path = root / "checkpoints" / f"cycle-{completed_cycles}" / "receipt.json"
    prepared_export_path = root / "checkpoints" / f"cycle-{completed_cycles}" / "prepared-export.json"
    if receipt_path.is_file():
        from scripts.validate_pace_vr_decision_fidelity import _restart_gate
        gate = _restart_gate(root / "manifest.json", {
            "restart_receipt": str(receipt_path),
        }, completed_cycles)
    else:
        gate = {"ready": False, "reason": "missing_exact_bilateral_restart"}
    prepared_export = None
    if prepared_export_path.is_file():
        try:
            prepared_export = _object(prepared_export_path)
            if (prepared_export.get("completed_cycles") != completed_cycles
                    or prepared_export.get("checkpoint_kind")
                    != "prepared_bilateral_shifted_primal_export"):
                raise ValueError("wrong checkpoint kind or cycle")
            prepared_export = {
                "status": "available_pending_fresh_worker_replay",
                "path": str(prepared_export_path),
                "serialization_roundtrip_exact": prepared_export.get("serialization_roundtrip_exact") is True,
                "fresh_worker_replay_verified": prepared_export.get("fresh_worker_replay_verified") is True,
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            prepared_export = {"status": "invalid", "detail": f"{type(error).__name__}: {error}"}
    return {
        "schema_version": 1,
        "source_directory": str(root),
        "completed_cycles": completed_cycles,
        "configuration_sha256": sha256(configuration_path.read_bytes()).hexdigest(),
        "pair_certified": pair_certified,
        "arms": per_arm,
        "exact_restart": gate,
        "prepared_export": prepared_export,
        "matched_continuation_ready": bool(pair_certified and gate["ready"]),
        "missing_restart_components": [] if gate["ready"] else list(REQUIRED_RESTART_COMPONENTS),
        "minimum_producer_change": (
            None if gate["ready"] else
            "At one certified bilateral boundary, each live worker must export its "
            "shifted primal, active bounds, Ding stimulation history, fixed parameters "
            "and work target; reload into a fresh worker and verify an identical "
            "prepared-problem digest before publishing the joint receipt."
        ),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--completed-cycles", required=True, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_directory(args.run_directory, args.completed_cycles)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
