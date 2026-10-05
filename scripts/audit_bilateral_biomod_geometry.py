"""Audit bilateral bioMod fibre lengths before Ding identification.

The script writes the exact muscles whose paths yield nonpositive fibre
lengths along the saved reduced kinematic profile.  It does not alter model
geometry or silently rectify a length.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cocofest.identification.cycling_geometry import PlanarEndpointForceAdapter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduced-profile", type=Path, required=True)
    parser.add_argument("--biomod", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=181)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-positive", action="store_true",
                        help="Return 2 when any fibre length is nonpositive.")
    args = parser.parse_args(argv)
    if args.samples < 5:
        parser.error("--samples must be at least 5")
    if args.output.exists():
        parser.error("--output must be new; an earlier audit is never overwritten")

    # Both adapters carry the same full bilateral profile; the right one is
    # enough to audit all model muscles and keeps the constructor unambiguous.
    adapter = PlanarEndpointForceAdapter.bilateral_arm_from_paths(
        args.reduced_profile, args.biomod, side="right",
    )
    kinematics = adapter.geometry.profile.kinematics
    angles = kinematics.theta_origin + kinematics.direction * np.linspace(0.0, 2 * np.pi, args.samples)
    report = adapter.audit_fibre_lengths(angles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if report["all_fibre_lengths_strictly_positive"]:
        print("PASS: all bilateral bioMod fibre lengths are strictly positive")
        return 0
    print("FAIL: nonpositive fibre lengths: " + ", ".join(report["invalid_muscles"]))
    return 2 if args.require_positive else 0


if __name__ == "__main__":
    raise SystemExit(main())
