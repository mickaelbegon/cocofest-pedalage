"""Run a compact synthetic Ding force-then-fatigue identification example.

Example:
    python scripts/demo_ding_identification.py --two-axis --output /tmp/ding-demo.json

The default mechanical projection is constant and illustrative. Supply a saved
reduced cycling profile to exercise task-dependent geometry. Supplying its
source bioMod derives the two-axis endpoint-force map by inverse dynamics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cocofest.identification.demo import run_demo
from cocofest.identification.synthetic_cycling import BILATERAL_MUSCLE_NAMES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muscle", choices=BILATERAL_MUSCLE_NAMES, default="right_Biceps")
    parser.add_argument("--two-axis", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reduced-profile", type=Path,
                        help="Saved reduced cycling geometry profile (.npz)")
    parser.add_argument("--profile-muscle-name",
                        help="Explicit name in a unilateral profile, e.g. Biceps")
    parser.add_argument("--biomod", type=Path,
                        help="Source bioMod matching --reduced-profile; enables endpoint-force geometry")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must be a new path, so earlier reports are never overwritten")
    result = run_demo(
        muscle=args.muscle, two_axis=args.two_axis, seed=args.seed,
        profile_path=args.reduced_profile, profile_muscle_name=args.profile_muscle_name,
        model_path=args.biomod,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote identification report to {args.output}")
    for stage in ("force", "fatigue"):
        fit = result[f"{stage}_fit"]
        print(f"{stage}: selected {[row['name'] for row in result[f'{stage}_design']['selected']]}; "
              f"Fisher rank {fit['fisher']['rank']}/{fit['fisher']['n_parameters']}; "
              f"holdout {fit['holdout_metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
