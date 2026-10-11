#!/usr/bin/env python3
"""Preregister an audited dose--duration bilateral torque-transfer pilot.

The computational worker and frozen-feasibility endpoint protocol are shared
with :mod:`run_causal_torque_pilot`.  This script only creates a new immutable
design: it never alters or reuses a previous causal-pilot output directory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.slow_layer import run_causal_torque_pilot as runner


DEFAULT_CHECKPOINTS = (
    ROOT / "asymmetric-sides-r192-rho-bo-20260928/factorial-weights-split-20261003/unit_fixed/results/checkpoints/cycle-120/receipt.json",
    ROOT / "asymmetric-sides-r192-rho-bo-20260928/factorial-weights-split-20261003/unit_fixed/results/checkpoints/cycle-140/receipt.json",
)
NEGATIVE_AMPLITUDES_NM = (.025, .05, .10)
DURATIONS_CYCLES = (5, 10)
CONTROL_REPEATS_PER_DURATION = 5


def declared_design() -> dict:
    """Return the fixed, scientifically preregistered 36-branch design."""
    return {"name": "fixed_36_branch_v1", "durations_cycles": list(DURATIONS_CYCLES),
            "negative_amplitudes_nm": list(NEGATIVE_AMPLITUDES_NM),
            "positive_amplitudes_nm": [.05],
            "control_repeats_per_duration": CONTROL_REPEATS_PER_DURATION}


def make_jobs(anchors: list[dict], design: dict, control_repeats: int) -> list[dict]:
    """Create duration-matched controls and every predeclared transfer branch."""
    if control_repeats != design["control_repeats_per_duration"]:
        raise ValueError("The fixed protocol requires five controls per duration")
    jobs = []
    for anchor in anchors:
        for duration in design["durations_cycles"]:
            for replicate in range(1, control_repeats + 1):
                jobs.append({"id": f"c{anchor['cycle']}-d{duration}-control-{replicate}",
                             "anchor": anchor, "delta_left_nm": 0., "is_control": True,
                             "action": "control", "intervention_cycles": duration})
            for amplitude in design["negative_amplitudes_nm"]:
                jobs.append({"id": f"c{anchor['cycle']}-d{duration}-left-minus-{amplitude:.3f}",
                             "anchor": anchor, "delta_left_nm": -amplitude,
                             "is_control": False, "action": "left-minus",
                             "intervention_cycles": duration})
            for amplitude in design["positive_amplitudes_nm"]:
                jobs.append({"id": f"c{anchor['cycle']}-d{duration}-left-plus-{amplitude:.3f}",
                             "anchor": anchor, "delta_left_nm": amplitude,
                             "is_control": False, "action": "left-plus",
                             "intervention_cycles": duration})
    return jobs


def _load_verified_anchor(receipt: Path) -> dict:
    """Authenticate a fixed-unit-weight RHO checkpoint before sealing it."""
    from cocofest.simulation.independent_arms_process import _driver_arguments
    from scripts.probe_independent_rho_task_reserve import _load_arm

    receipt = Path(receipt).resolve(strict=True)
    record = {"receipt": str(receipt), "receipt_sha256": runner.digest(receipt), "arms": {}}
    for side in ("left", "right"):
        source, payload = _load_arm(receipt, side)
        source.verify_files()
        args = _driver_arguments(payload, side)
        unit = payload["muscle_pace"].get(f"{side}_initial_weights")
        if (args.collocation_degree != 5 or args.stimulations_per_cycle != 30
                or args.ipopt_linear_solver != "ma57" or payload["resistance_pace"]["capacity_feedback"]
                or payload["muscle_pace"]["adaptation_enabled"]
                or unit != dict.fromkeys(runner.NAMES, 1.)
                or not math.isclose(payload[f"{side}_equivalent_mean_torque_nm"], .96)):
            raise ValueError("Anchor differs from the fixed unit-weight Radau-5/30 Hz IPOPT-MA57 baseline")
        record["cycle"] = source.completed_cycles
        record["arms"][side] = {"archive_sha256": source.archive_sha256,
                                "prepared_problem_sha256": source.prepared_problem_sha256}
    return record


def prepare(output: Path, receipts: list[Path], *, maximum_cycles: int = 240,
            control_repeats: int = CONTROL_REPEATS_PER_DURATION) -> Path:
    """Write a sealed, non-overwritable dose--duration campaign manifest."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if maximum_cycles <= 150:
        raise ValueError("The absolute endpoint must exceed both anchor windows")
    design = declared_design()
    if control_repeats != design["control_repeats_per_duration"]:
        raise ValueError("This preregistration fixes two matched controls per duration")
    anchors = [_load_verified_anchor(receipt) for receipt in receipts]
    cycles = sorted(anchor["cycle"] for anchor in anchors)
    if cycles != [120, 140] or len({anchor["cycle"] for anchor in anchors}) != 2:
        raise ValueError("Dose--duration pilot requires distinct certified checkpoints at cycles 120 and 140")
    if any(maximum_cycles <= anchor["cycle"] + max(design["durations_cycles"]) for anchor in anchors):
        raise ValueError("Endpoint must exceed every intervention window")
    jobs = make_jobs(anchors, design, control_repeats)
    sources = sorted(set(ROOT.glob("cocofest/**/*.py")) | {
        Path(__file__).resolve(), Path(runner.__file__).resolve(),
        ROOT / "scripts/run_unilateral_continuation_labels.py",
        ROOT / "scripts/run_checkpoint_fatigue_ablation.py",
        ROOT / "scripts/run_local_task_reserve_branch.py",
        ROOT / "scripts/probe_independent_rho_task_reserve.py",
        ROOT / "scripts/validate_independent_rho_checkpoint.py",
        ROOT / "examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py"})
    plan = {
        "schema_version": 1,
        "kind": "paired_bilateral_torque_transfer_dose_duration",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "jobs": jobs,
        "intervention_design": design,
        "code_sha256": {str(path): runner.digest(path) for path in sources},
        "maximum_absolute_cycles": maximum_cycles,
        "numeric_threads": 1,
        "audit_tolerance": 1e-5,
        "weights": dict.fromkeys(runner.NAMES, 1.),
        "intervention": "At c120/c140, preserve 1.92 Nm total; transfer -0.025/-0.05/-0.10 Nm or +0.05 Nm for 5 or 10 complete cycles.",
        "continuation_policy": {
            "kind": "fixed_equal_split", "left_nm": .96, "right_nm": .96,
            "switch": "Immediately after the sealed intervention duration; return explicitly to 0.96/0.96 Nm until endpoint.",
            "update_period_cycles": 10, "maximum_asymmetry_nm": .20,
            "scope": "simulation protocol, not a clinical comfort/ramp envelope"},
        "endpoint_protocol": "Inherited frozen retries, zero-objective feasibility and fatigue-rest counterfactual; local attribution only",
        "control_gate": "Every duration-matched repeated control must have an attributable frozen endpoint before any transfer branch launches.",
        "resolution": "For each anchor/duration, the repeated-control endpoint spread is the numerical resolution; a treatment difference no larger than it is a tie.",
        "activation_allowed": False,
    }
    output.mkdir(parents=True)
    path = output / "manifest.json"
    runner.write_json(path, plan)
    runner.write_json(output / "manifest-seal.json", {"manifest_sha256": runner.digest(path)})
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "campaign"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipts", nargs="+", type=Path, default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument("--maximum-cycles", type=int, default=240)
    parser.add_argument("--control-repeats", type=int, default=CONTROL_REPEATS_PER_DURATION)
    parser.add_argument("--cpus", nargs="+", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.manifest.name != "manifest.json":
            parser.error("Manifest filename must be manifest.json")
        print(prepare(args.manifest.parent, args.receipts, maximum_cycles=args.maximum_cycles,
                      control_repeats=args.control_repeats))
    else:
        runner.campaign(args.manifest, args.cpus, args.timeout_seconds, args.controls_only)


if __name__ == "__main__":
    main()
