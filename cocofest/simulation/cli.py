"""Resolve reproducible GUI/campaign inputs without importing scientific engines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .launch import ROOT, build_launch_plan
from .resolved_config import PROFILES, resolve_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, help="SimulationConfig JSON (explicit fields)")
    parser.add_argument("--profile", choices=tuple(PROFILES), help="Cumulative IPOPT ablation profile")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=JSON",
                        help='Explicit field, e.g. cycles=100 or solver=\"ipopt\"; repeatable')
    parser.add_argument("--prefix", type=Path, help="Optional runtime prefix to include a launch plan")
    parser.add_argument("--root", type=Path, default=ROOT, help="Root for managed relative paths")
    parser.add_argument("--output", type=Path, help="Write a new resolved JSON document instead of stdout")
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        if not isinstance(data, dict):
            raise ValueError("Configuration JSON must be an object")
        for assignment in args.set:
            name, separator, encoded = assignment.partition("=")
            if not separator:
                raise ValueError(f"Expected FIELD=JSON, got {assignment!r}")
            value = json.loads(encoded)
            if name in data and data[name] != value:
                raise ValueError(f"Conflicting explicit values for {name}")
            data[name] = value
        resolved = resolve_config(data, profile=args.profile, root=args.root)
        document = resolved.to_dict()
        # Keep the editable input separate from the resolved audit document.
        document["simulation_config"] = resolved.config.to_dict()
        if args.prefix:
            plan = build_launch_plan(resolved.config, args.prefix, args.root)
            document["launch"] = {"argv": list(plan.argv), "cwd": str(plan.cwd),
                                  "environment_updates": plan.environment_updates}
        text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(text)
        else:
            print(text, end="")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
