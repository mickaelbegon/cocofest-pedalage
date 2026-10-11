"""Print a pinned A0/A1 protocol as JSON; never launch a simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cocofest.cycling.slow.campaign import SlowCampaignInputs, build_slow_campaign_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("model-left", "model-right", "rho-config", "baseline-source", "split-config"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--horizon-cycles", type=int, required=True)
    parser.add_argument("--update-every-cycles", type=int, default=10)
    args = parser.parse_args()
    try:
        plan = build_slow_campaign_plan(SlowCampaignInputs(**vars(args)))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
