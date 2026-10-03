#!/usr/bin/env python3
"""Plan local reachable-state reserve experiments from a completed calibration.

Reads and verifies existing checkpoints and witnesses. Writes only a new
experiment plan; does not launch simulations, fit a model or activate a cost.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.task_reserve import LocalReserveSample
from cocofest.optimization.task_reserve_experiment_design import (
    ReserveDesignSample, common_followup_grid, local_neighborhood, reachable_branch_design,
)
from scripts.calibrate_independent_rho_task_reserve import (
    _load_arm, _object, checkpoint_coordinates, load_manifest, load_probe_result,
)


def plan(audit_path: Path, *, anchor_ids=None, coordinate_radius=.05,
         refinement_step=.025, exploration_ceiling=1.5, extension_step=.1,
         log_amplitude=.25) -> dict:
    audit_path = Path(audit_path).resolve(strict=True)
    audit = _object(audit_path)
    if audit.get("schema_version") != 1:
        raise ValueError("Unsupported calibration audit schema")
    spec = load_manifest(audit["manifest_path"])
    if spec["sha256"] != audit.get("manifest_sha256") or spec["side"] != audit.get("side"):
        raise ValueError("Calibration audit and source manifest differ")
    row_by_id = {row["id"]: row for row in audit["samples"]}
    if len(row_by_id) != len(audit["samples"]) or set(row_by_id) != {row["id"] for row in spec["samples"]}:
        raise ValueError("Calibration audit sample ids differ from manifest")
    samples = []
    for entry in spec["samples"]:
        row = row_by_id[entry["id"]]
        if sha256(entry["receipt"].read_bytes()).hexdigest() != row["receipt_sha256"]:
            raise ValueError("Source receipt changed after calibration")
        source, _ = _load_arm(entry["receipt"], spec["side"])
        result = Path(row["probe"]["path"])
        if sha256(result.read_bytes()).hexdigest() != row["probe"]["sha256"]:
            raise ValueError("Source probe changed after calibration")
        estimate, checked_probe = load_probe_result(result, source, spec["work_scales"], spec["tolerance"])
        if checked_probe != row["probe"]:
            raise ValueError("Witness artifact changed after calibration")
        coordinates, coordinate_audit = checkpoint_coordinates(source, spec["coordinates"])
        if coordinate_audit != row["coordinates"] or entry["partition"] != row["partition"]:
            raise ValueError("Discovery coordinates or partition disagree with audit")
        samples.append(ReserveDesignSample(entry["id"], LocalReserveSample(coordinates, estimate),
                                           entry["partition"], str(entry["receipt"])))
    if anchor_ids is None:
        anchor_ids = [item.sample_id for item in sorted(samples,
            key=lambda item: item.sample.estimate.checkpoint.completed_cycles)[-3:]]
    if not anchor_ids or len(set(anchor_ids)) != len(anchor_ids):
        raise ValueError("Distinct anchor ids are required")
    muscle_names = []
    for coordinate in spec["coordinates"]:
        if not coordinate.state_key.startswith("A_") or coordinate.index != 0:
            raise ValueError("Initial experiment supports an explicit A capacity coordinate per muscle")
        muscle_names.append(coordinate.state_key[2:])
    neighborhoods = [local_neighborhood(samples, anchor_id=anchor,
                         coordinate_radius=[coordinate_radius] * len(spec["coordinates"]))
                     for anchor in anchor_ids]
    reports = [row["grid"] for neighborhood in neighborhoods for row in neighborhood["selected"]]
    grid = common_followup_grid(reports, refinement_step=refinement_step,
                               exploration_ceiling=exploration_ceiling, extension_step=extension_step)
    branch_design = reachable_branch_design(muscle_names, log_amplitude=log_amplitude)
    return {"schema_version": 1, "kind": "local_task_reserve_experiment_plan", "side": spec["side"],
            "source_audit": str(audit_path), "source_audit_sha256": sha256(audit_path.read_bytes()).hexdigest(),
            "source_manifest": str(spec["path"]), "source_manifest_sha256": spec["sha256"],
            "discovery_only": True, "activation_allowed": False,
            "local_neighborhoods": neighborhoods, "common_followup_grid": grid,
            "reachable_branch_design": branch_design,
            "estimated_one_cycle_probe_solves": (len(anchor_ids) * branch_design["endpoint_count_per_anchor"]
                                                  * len(grid["work_scales"])),
            "execution_stages": [
                "Repeat nominal checkpoint solve to audit restoration before branch work.",
                "Run all short branch policies at the same nominal task and certify every reached endpoint.",
                "Audit the reachable-state rank and trust box before spending the load-probe budget.",
                "Evaluate the same full grid at every retained endpoint; retain failed factors as undetermined.",
                "Fit on training branches only; freeze coefficients and assess all holdout branches.",
                "Repeat selected near-boundary factors with alternative starts to audit numerical sensitivity.",
                "Check omitted Ding/history variation; reject A-only fit if near-identical A predicts different reserves.",
                "Validate accepted cost by matched real-RHO continuations before an endurance comparison."],
            "executor_implemented_by_this_plan": False,
            "scientific_scope": "empirical witnessed reserve on reachable local states, not a maximum load boundary"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor", action="append", dest="anchor_ids")
    parser.add_argument("--coordinate-radius", type=float, default=.05)
    parser.add_argument("--refinement-step", type=float, default=.025)
    parser.add_argument("--exploration-ceiling", type=float, default=1.5)
    parser.add_argument("--extension-step", type=float, default=.1)
    parser.add_argument("--log-amplitude", type=float, default=.25)
    args = vars(parser.parse_args(argv))
    output = args.pop("output").resolve()
    value = plan(args.pop("calibration_audit"), **args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
