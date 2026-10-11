"""Prepare the independent P1/P2/P7 inputs for a simulated slow-layer session."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cocofest.cycling.presession import PresessionConfig, prepare_presession


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-left", type=Path, required=True)
    parser.add_argument("--model-right", type=Path, required=True)
    parser.add_argument("--horizon-cycles", type=int, required=True)
    parser.add_argument("--mode", choices=("prescribed_load", "max_work"), default="prescribed_load")
    parser.add_argument("--cycle-duration-s", type=float, default=1.0)
    parser.add_argument("--stimulation-hz", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reduced-profile", type=Path)
    parser.add_argument("--theta0-rad", type=float, default=0.0)
    parser.add_argument("--omega-rad-s", type=float, default=-6.283185307179586)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = PresessionConfig(
        model_left=args.model_left, model_right=args.model_right,
        horizon_cycles=args.horizon_cycles, mode=args.mode,
        cycle_duration_s=args.cycle_duration_s, stimulation_hz=args.stimulation_hz,
        seed=args.seed, reduced_profile=args.reduced_profile,
        theta0_rad=args.theta0_rad, omega_rad_s=args.omega_rad_s,
    )
    print(prepare_presession(config, args.output_root))


if __name__ == "__main__":
    main()
