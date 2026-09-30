#!/usr/bin/env python3
"""Run two prebuilt independent unilateral isokinetic arm solvers.

The supplied factory has the form ``package.module:factory`` and receives the
decoded JSON configuration.  It must construct *two distinct already-built*
unilateral solver handles and return an
``IndependentArmCoordinator``.  This intentionally separates one-off model
construction/code generation from repeated work-target updates.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.simulation.independent_arms import IndependentArmCoordinator
from cocofest.optimization.independent_arm_rho_pace import (
    IndependentArmResistancePace,
    IndependentArmRhoPaceConfig,
)


def _factory(specification: str):
    try:
        module_name, attribute = specification.split(":", 1)
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ValueError, ImportError, AttributeError) as exc:
        raise ValueError("--factory must be an importable 'package.module:function'.") from exc
    if not callable(factory):
        raise ValueError("--factory must name a callable.")
    return factory


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON runtime configuration")
    parser.add_argument("--factory", help="module:function that builds compiled arm handles (or config key factory)")
    parser.add_argument("--output-dir", "--output", dest="output", required=True, type=Path,
                        help="output directory for summary.json")
    # Kept in the public invocation shape used by the GUI.  A factory can use
    # it to locate its Python/ACADOS runtime; the coordinator itself never
    # rebuilds a solver from this value.
    parser.add_argument("--prefix", type=Path, default=None, help="optional prepared solver runtime prefix")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The independent-arm configuration must be a JSON object.")
    specification = args.factory or payload.get("factory")
    if not specification:
        raise ValueError("Provide --factory or a 'factory' key in the configuration.")
    if args.prefix is not None:
        payload.setdefault("runtime_prefix", str(args.prefix))
    coordinator = _factory(specification)(payload)
    required = ("run_to_directory", "run_with_resistance_pace_to_directory")
    if not isinstance(coordinator, IndependentArmCoordinator) and not all(
        callable(getattr(coordinator, name, None)) for name in required
    ):
        raise TypeError("The factory must return an independent-arm coordinator with the public run methods.")
    cycles = payload.get("cycles", 1)
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
        raise ValueError("'cycles' must be a strictly positive integer.")
    pace_payload = payload.get("resistance_pace")
    if pace_payload is None:
        summary = coordinator.run_to_directory(args.output, cycles=cycles)
    else:
        if not isinstance(pace_payload, dict):
            raise ValueError("'resistance_pace' must be an object.")
        # The total is deliberately explicit: it must not silently drift when
        # an operator changes one side in the GUI.
        pace = IndependentArmResistancePace(IndependentArmRhoPaceConfig(**pace_payload))
        summary = coordinator.run_with_resistance_pace_to_directory(args.output, pace, cycles=cycles)
    # A GUI may pre-plan a small load-balancing sequence.  It is deliberately
    # handled in this one process: the coordinator retains both compiled
    # handles, so each target update is an E_prod bound update rather than a
    # second code-generation/compilation event.
    adjustments = payload.get("adjustments", [])
    if not isinstance(adjustments, list):
        raise ValueError("'adjustments' must be a list of torque-update objects.")
    adjustment_summaries = []
    if adjustments and not callable(getattr(coordinator, "set_equivalent_mean_torques", None)):
        raise ValueError("Preplanned adjustments are unsupported by the process-synchronised coordinator.")
    for index, adjustment in enumerate(adjustments, start=1):
        if not isinstance(adjustment, dict):
            raise ValueError("Every adjustment must be an object.")
        allowed = {"right_equivalent_mean_torque_nm", "left_equivalent_mean_torque_nm"}
        unknown = set(adjustment) - allowed
        if unknown:
            raise ValueError(f"Unknown adjustment fields: {sorted(unknown)}")
        coordinator.set_equivalent_mean_torques(
            right_nm=adjustment.get("right_equivalent_mean_torque_nm"),
            left_nm=adjustment.get("left_equivalent_mean_torque_nm"),
        )
        adjustment_summaries.append(coordinator.run_to_directory(
            args.output / "adjustments" / f"{index:03d}"
        ))
    if adjustment_summaries:
        (args.output / "adjustments.json").write_text(
            json.dumps({"initial": summary, "adjustments": adjustment_summaries}, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
