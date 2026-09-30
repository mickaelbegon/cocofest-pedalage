#!/usr/bin/env python3
"""Make variant-local feedback weights for baseline RHO, separate from fixed BO."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for case in manifest["cases"]:
        directory = root / case["id"]
        source = json.loads((directory / "unit-weights.json").read_text())
        target = directory / "rho-feedback-weights-retry4.json"
        if target.exists():
            raise FileExistsError(target)
        source["policy"]["adaptation_strategy"] = "capacity_feedback"
        source["policy"]["update_every_cycles"] = 20
        target.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
        print(target)


if __name__ == "__main__":
    main()
