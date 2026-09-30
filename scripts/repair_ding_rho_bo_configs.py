#!/usr/bin/env python3
"""Create fresh, valid RHO-BO retry configurations for a sensitivity campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tag", default="retry3")
    args = parser.parse_args()
    root = args.campaign.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        case_dir = root / case["id"]
        source = json.loads((case_dir / "rho-bo.json").read_text(encoding="utf-8"))
        target = case_dir / f"rho-bo-{args.tag}.json"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        config = source
        base = config["base_config"]
        base["compile_evaluators"] = False
        base["reduced_internal_crank_velocity_guard"] = "on"
        base["reduced_terminal_half_step_velocity_guard"] = True
        base["extra_arguments"] = []
        base["weights_config"] = str(case_dir / "unit-weights-retry3.json")
        config["output_root"] = str(case_dir / f"rho-bo-{args.tag}")
        base["output_root"] = config["output_root"]
        config["study_name"] = f"rho-weight-bo-{case['id']}-{args.tag}"
        config["workers"] = args.workers
        weights = json.loads((case_dir / "unit-weights.json").read_text(encoding="utf-8"))
        weights["policy"]["adaptation_strategy"] = "capacity_feedback"
        weights_path = Path(base["weights_config"])
        if weights_path.exists():
            raise FileExistsError(f"Refusing to overwrite {weights_path}")
        weights_path.write_text(json.dumps(weights, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(target)


if __name__ == "__main__":
    main()
