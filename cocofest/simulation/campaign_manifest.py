"""Read-only planning for declarative campaigns; never starts a process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Mapping

from .launch import ROOT, build_launch_plan
from .resolved_config import resolve_config

SCHEMA_VERSION = 1
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _object(value, allowed, required, context):
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if set(value) - allowed or required - set(value):
        raise ValueError(f"{context}: unknown or missing fields")
    return value


def _name(value, context):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"{context} must be a safe, non-empty identifier")
    return value


def _cpus(value, context, *, empty=False):
    if not isinstance(value, list) or (not value and not empty):
        raise ValueError(f"{context} must be a list of CPU indices")
    if any(type(cpu) is not int or cpu < 0 for cpu in value) or len(set(value)) != len(
        value
    ):
        raise ValueError(f"{context} requires unique non-negative integer CPU indices")
    return tuple(sorted(value))


def _path(value, root, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _artifact_status(output: Path, result: Path, config_hash: str):
    """Reuse means inputs match, not that scientific validity was certified."""
    if not output.exists():
        return "ready", None
    if not output.is_dir():
        return "conflict", "Case output path is not a directory"
    if not any(output.iterdir()):
        return "ready", None
    effective = result.parent / "effective-configuration.json"
    try:
        record = json.loads(effective.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("config_hash") != config_hash:
            return "conflict", "Existing effective configuration hash differs"
        document = json.loads(result.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not document:
            return "conflict", "Existing result must be a non-empty JSON object"
    except (OSError, ValueError) as error:
        return "incomplete", f"Existing artifacts need review: {error}"
    return "recorded", None


def plan_campaign(manifest: Mapping, *, root: Path = ROOT) -> dict:
    """Validate and plan independent CPU lanes with ordered cases.

    Fields are strict and cases have unique output/cache roots. Within a lane,
    every case implicitly depends on its predecessor. Cross-lane dependencies
    must reference an earlier case, preventing cycles. This function only reads
    artifact files. ``recorded`` never implies convergence or feasibility.
    """
    root = Path(root).expanduser().resolve()
    manifest = _object(
        manifest,
        {
            "schema_version",
            "campaign_id",
            "output_root",
            "reserved_cpus",
            "lanes",
            "cases",
        },
        {
            "schema_version",
            "campaign_id",
            "output_root",
            "reserved_cpus",
            "lanes",
            "cases",
        },
        "Manifest",
    )
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("Unsupported campaign schema_version")
    campaign_id = _name(manifest["campaign_id"], "campaign_id")
    campaign_root = _path(manifest["output_root"], root, "output_root") / campaign_id
    reserved = set(_cpus(manifest["reserved_cpus"], "reserved_cpus", empty=True))
    if not isinstance(manifest["lanes"], dict) or not manifest["lanes"]:
        raise ValueError("lanes must be a non-empty object")
    lanes = {}
    allocated = set()
    for name, lane in manifest["lanes"].items():
        _name(name, "Lane name")
        lane = _object(
            lane, {"cpus", "runtime_prefix"}, {"cpus", "runtime_prefix"}, f"Lane {name}"
        )
        cpus = _cpus(lane["cpus"], f"Lane {name} cpus")
        if set(cpus) & (reserved | allocated):
            raise ValueError(f"Lane {name} overlaps reserved CPUs or another lane")
        allocated.update(cpus)
        lanes[name] = (
            cpus,
            _path(lane["runtime_prefix"], root, f"Lane {name} runtime_prefix"),
        )
    if not isinstance(manifest["cases"], list) or not manifest["cases"]:
        raise ValueError("cases must be a non-empty list")
    cases, seen, previous = [], {}, {}
    for case in manifest["cases"]:
        case = _object(
            case,
            {"id", "lane", "profile", "config", "depends_on"},
            {"id", "lane", "config"},
            "Case",
        )
        case_id = _name(case["id"], "Case id")
        if case_id in seen:
            raise ValueError(f"Duplicate case id: {case_id}")
        lane = case["lane"]
        if not isinstance(lane, str) or lane not in lanes:
            raise ValueError(f"Unknown lane for {case_id}")
        dependencies = case.get("depends_on", [])
        if not isinstance(dependencies, list) or any(
            not isinstance(item, str) for item in dependencies
        ):
            raise ValueError(f"depends_on for {case_id} must be a list of case ids")
        if len(set(dependencies)) != len(dependencies) or any(
            item not in seen for item in dependencies
        ):
            raise ValueError(
                f"Dependencies for {case_id} must be unique earlier case ids"
            )
        dependencies = list(dependencies)
        if lane in previous and previous[lane] not in dependencies:
            dependencies.append(previous[lane])
        if not isinstance(case["config"], dict):
            raise ValueError(f"Configuration for {case_id} must be an object")
        config = dict(case["config"])
        forbidden = {"output_root", "numeric_threads", "dry_run"} & set(config)
        if forbidden:
            raise ValueError(
                f"Case {case_id} overrides campaign-managed fields: {sorted(forbidden)}"
            )
        output = campaign_root / case_id
        config.update(output_root=str(output), numeric_threads=1, dry_run=True)
        resolved = resolve_config(config, profile=case.get("profile"), root=root)
        cpus, prefix = lanes[lane]
        plan = build_launch_plan(resolved.config, prefix, root)
        if resolved.config.threads > len(cpus):
            raise ValueError(
                f"Case {case_id} uses more worker threads than allocated CPUs"
            )
        if plan.result_json is None or not plan.result_json.is_relative_to(output):
            raise ValueError(f"Case {case_id} result escapes its output directory")
        status, reason = _artifact_status(
            output, plan.result_json, resolved.config_hash
        )
        waiting = [
            dependency
            for dependency in dependencies
            if seen[dependency]["status"] != "recorded"
        ]
        if status == "ready" and waiting:
            status, reason = (
                "waiting",
                f"Awaiting recorded results: {', '.join(waiting)}",
            )
        record = {
            "id": case_id,
            "lane": lane,
            "cpus": list(cpus),
            "depends_on": dependencies,
            "status": status,
            "reason": reason,
            "config_hash": resolved.config_hash,
            "resolved_configuration": resolved.to_dict(),
            "output_root": str(output),
            "result_json": str(plan.result_json),
            "cwd": str(plan.cwd),
            "argv": ["taskset", "--cpu-list", ",".join(map(str, cpus)), *plan.argv],
            "environment_updates": plan.environment_updates,
        }
        cases.append(record)
        seen[case_id] = record
        previous[lane] = case_id
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "dry_run": True,
        "reserved_cpus": sorted(reserved),
        "cases": cases,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        print(
            json.dumps(
                plan_campaign(manifest, root=args.root),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
