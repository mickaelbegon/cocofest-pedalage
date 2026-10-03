#!/usr/bin/env python3
"""Audit and fit a local observed-work-reserve model from exact RHO checkpoints.

A manifest declares one arm, one work grid, explicit normalized state
coordinates, and a train/holdout partition *before* evaluating the probes.
Existing probe artifacts are revalidated, never overwritten. Missing probes
are only launched with --run-missing. This is a local discrete-NLP witness
calibration, not a maximum-load or endurance certificate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.task_reserve import (
    LocalReserveSample, ProbeEvidence, TaskReserveProbe, WORK_RESERVE_METRIC,
    evaluate_work_reserve, fit_local_reserve,
)
from cocofest.optimization.task_reserve_ocp import TaskReserveStateCoordinate
from scripts.probe_independent_rho_task_reserve import (
    _load_arm, run as run_probe, run_from_checkpoint,
)
from scripts.run_local_task_reserve_branch import load_branch_endpoint


BILATERAL_SOURCE = "certified_bilateral_restart"
BRANCH_SOURCE = "certified_unilateral_branch_endpoint"


def _object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _path(parent, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Artifact paths must be nonempty strings")
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else parent / candidate).resolve()


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def load_manifest(path):
    """Validate an explicit, reproducible partition without loading any NLP."""
    path = Path(path).resolve(strict=True)
    manifest = _object(path)
    if manifest.get("schema_version") != 1 or manifest.get("side") not in ("left", "right"):
        raise ValueError("Manifest requires schema_version=1 and one explicit side")
    source_kind = manifest.get("source_kind", BILATERAL_SOURCE)
    if source_kind not in (BILATERAL_SOURCE, BRANCH_SOURCE):
        raise ValueError("Unknown calibration source_kind")
    build_context = manifest.get("build_context", {})
    if not isinstance(build_context, dict):
        raise ValueError("build_context must be an object when provided")
    try:
        build_context = json.loads(json.dumps(build_context, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError("build_context must be finite JSON data") from error
    plan_path = None
    if source_kind == BRANCH_SOURCE:
        plan_path = _path(path.parent, manifest.get("experiment_plan"))
        plan = _object(plan_path)
        if (plan.get("schema_version") != 1 or plan.get("kind") != "local_task_reserve_experiment_plan"
                or plan.get("side") != manifest["side"]):
            raise ValueError("Branch calibration needs a matching predeclared experiment plan")
    elif "experiment_plan" in manifest:
        raise ValueError("Bilateral calibration cannot use a branch experiment plan")
    factors = tuple(_positive(v, "work_scale") for v in manifest.get("work_scales", ()))
    if len(set(factors)) != len(factors) or 1. not in factors:
        raise ValueError("Distinct work_scales must explicitly include nominal 1")
    factors = (1.,) + tuple(factor for factor in factors if factor != 1.)
    coordinates = tuple(TaskReserveStateCoordinate(**value)
                        for value in manifest.get("coordinates", ()))
    if not coordinates or len({(c.state_key, c.index) for c in coordinates}) != len(coordinates):
        raise ValueError("Distinct explicit normalized state coordinates are required")
    fit = manifest.get("fit", {})
    if not isinstance(fit, dict):
        raise ValueError("fit must be an object")
    center, trust = np.asarray(fit.get("center", ()), float), np.asarray(fit.get("trust_radius", ()), float)
    if (center.shape != (len(coordinates),) or trust.shape != center.shape
            or not np.all(np.isfinite(center)) or not np.all(np.isfinite(trust)) or np.any(trust <= 0)):
        raise ValueError("fit center/trust_radius must match the finite coordinate layout")
    minimum_train = fit.get("minimum_train_samples", len(coordinates) + 1)
    minimum_holdout = fit.get("minimum_holdout_samples", 1)
    if (type(minimum_train) is not int or minimum_train < len(coordinates) + 1
            or type(minimum_holdout) is not int or minimum_holdout < 1):
        raise ValueError("Minimum samples must include dimension+1 train and one independent holdout")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Manifest needs checkpoint samples")
    ids, receipts, parsed = set(), set(), []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Each sample must be an object")
        name = sample.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name) or name in ids:
            raise ValueError("Sample ids must be unique safe path components")
        partition = sample.get("partition")
        if partition not in ("train", "holdout"):
            raise ValueError("Each sample must explicitly declare train or holdout")
        receipt = _path(path.parent, sample.get("receipt"))
        if receipt in receipts:
            raise ValueError("A receipt cannot appear twice or leak between train and holdout")
        ids.add(name)
        receipts.add(receipt)
        parsed.append({"id": name, "partition": partition, "receipt": receipt,
                       "probe_result": (_path(path.parent, sample["probe_result"])
                                        if "probe_result" in sample else None)})
    return {"path": path, "sha256": sha256(path.read_bytes()).hexdigest(),
            "side": manifest["side"], "source_kind": source_kind, "plan_path": plan_path,
            "plan_sha256": sha256(plan_path.read_bytes()).hexdigest() if plan_path else None,
            "build_context": build_context,
            "work_scales": factors,
            "coordinates": coordinates, "samples": parsed,
            "center": tuple(center), "trust_radius": tuple(trust),
            "minimum_train_samples": minimum_train, "minimum_holdout_samples": minimum_holdout,
            "maximum_error": _positive(fit.get("maximum_error", .01), "maximum_error"),
            "maximum_condition_number": _positive(fit.get("maximum_condition_number", 1e6),
                                                   "maximum_condition_number"),
            "tolerance": _positive(manifest.get("tolerance", 1e-5), "tolerance")}


def checkpoint_coordinates(source, coordinates):
    """Read the actual frozen initial equality, not the shifted warm start."""
    source.verify_files()
    values, detail = [], []
    with np.load(source.archive_path, allow_pickle=False) as archive:
        for coordinate in coordinates:
            key = f"problem__x_bounds:{coordinate.state_key}"
            lower, upper = np.asarray(archive[key + ":min"], float), np.asarray(archive[key + ":max"], float)
            if (lower.ndim != 2 or lower.shape != upper.shape or lower.shape[1] < 1
                    or coordinate.index >= lower.shape[0]):
                raise ValueError(f"Incomplete prepared bounds for {coordinate.state_key}")
            raw = float(lower[coordinate.index, 0])
            if not math.isfinite(raw) or raw != float(upper[coordinate.index, 0]):
                raise ValueError(f"Coordinate {coordinate.state_key} is not a frozen initial equality")
            value = (raw - coordinate.offset) / coordinate.scale
            values.append(value)
            detail.append({**asdict(coordinate), "raw_value": raw, "normalized_value": value,
                           "source": key, "column": 0})
    return tuple(values), detail


def _load_source(sample, spec):
    """Authenticate either receipt type without broadening the bilateral loader."""
    if spec["source_kind"] == BILATERAL_SOURCE:
        source, payload = _load_arm(sample["receipt"], spec["side"])
        return source, payload, None
    source, payload = load_branch_endpoint(sample["receipt"])
    receipt = _object(sample["receipt"])
    if receipt.get("side") != spec["side"] or receipt.get("plan_sha256") != spec["plan_sha256"]:
        raise ValueError("Branch endpoint differs from the declared plan or side")
    if _path(sample["receipt"].parent, receipt.get("plan_path")) != spec["plan_path"]:
        raise ValueError("Branch endpoint refers to another experiment plan")
    plan = _object(spec["plan_path"])
    policies = plan["reachable_branch_design"]["branches"]
    matching = [policy for policy in policies if policy["id"] == receipt["policy"]["id"]]
    if len(matching) != 1 or matching[0] != receipt["policy"]:
        raise ValueError("Branch policy differs from the predeclared plan")
    if matching[0]["partition"] != sample["partition"]:
        raise ValueError("Branch train/holdout partition differs from its plan policy")
    anchors = [row for row in plan["local_neighborhoods"] if row["anchor_id"] == receipt["anchor_id"]]
    if len(anchors) != 1:
        raise ValueError("Branch anchor differs from the predeclared plan")
    provenance = {"anchor_id": receipt["anchor_id"], "policy_id": receipt["policy"]["id"],
                  "policy_partition": matching[0]["partition"], "branch_cycle": receipt["branch_cycle"],
                  "source_receipt": receipt["source_receipt"],
                  "source_receipt_sha256": receipt["source_receipt_sha256"],
                  "plan_sha256": receipt["plan_sha256"]}
    return source, payload, provenance


def load_probe_result(path, source, work_scales, tolerance):
    """Recompute witness feasibility and summaries; distrust stored success flags."""
    path = Path(path).resolve(strict=True)
    value = _object(path)
    expected = json.loads(json.dumps(asdict(source)))
    if value.get("checkpoint") != expected:
        raise ValueError("Probe result provenance differs from its certified receipt")
    if value.get("metric") != WORK_RESERVE_METRIC:
        raise ValueError("Probe result uses a different reserve metric")
    if value.get("global_upper_bound") is not None or value.get("physiological_failure_certified") is not False:
        raise ValueError("Probe result makes an unsupported maximum/failure claim")
    reports = []
    witnesses = []
    for raw in value.get("evidence", ()):
        report = ProbeEvidence(**{**raw, "validated_constraint_groups": tuple(raw.get("validated_constraint_groups", ()))})
        if report.tolerance > tolerance:
            raise ValueError("Probe tolerance exceeds the calibration tolerance")
        request = TaskReserveProbe(source, report.work_scale)
        if report.witness_is_valid(request):
            witness = _path(path.parent, report.witness_id).resolve(strict=True)
            with np.load(witness, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata__json"]))
                if (metadata.get("source_archive_sha256") != source.archive_sha256
                        or metadata.get("prepared_problem_sha256") != source.prepared_problem_sha256
                        or metadata.get("work_scale") != report.work_scale):
                    raise ValueError("Witness trajectory has different checkpoint/load provenance")
                vector = np.asarray(archive["vector"], float)
                trajectories = [np.asarray(archive[key], float) for key in archive.files
                                if key.startswith(("states__", "controls__"))]
                if (not vector.size or not np.all(np.isfinite(vector)) or not trajectories
                        or not any(key.startswith("states__") for key in archive.files)
                        or not any(key.startswith("controls__") for key in archive.files)
                        or any(not item.size or not np.all(np.isfinite(item)) for item in trajectories)):
                    raise ValueError("Witness trajectory is missing or nonfinite")
            witnesses.append({"path": str(witness), "sha256": sha256(witness.read_bytes()).hexdigest(),
                              "work_scale": report.work_scale})
        reports.append(report)
    if len(reports) != len(work_scales) or {r.work_scale for r in reports} != set(work_scales):
        raise ValueError("Every checkpoint must use the same complete work-scale grid")
    by_factor = {report.work_scale: report for report in reports}
    estimate = evaluate_work_reserve(source, work_scales, lambda request: by_factor[request.work_scale])
    for name in ("feasible_work_scales", "nominal_task_witnessed", "work_scale_lower_bound",
                 "reserve_lower_bound", "status"):
        actual = getattr(estimate, name)
        if value.get(name) != (list(actual) if isinstance(actual, tuple) else actual):
            raise ValueError(f"Probe summary disagrees with independently checked evidence: {name}")
    return estimate, {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest(), "witnesses": witnesses}


def _json_finite(value):
    """Rejected rank-deficient fits contain infinities; preserve valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_finite(item) for item in value]
    return value


def _trust_box_violations(rows, spec):
    """Describe every usable checkpoint outside the declared fit domain."""
    violations = []
    for row in rows:
        if not row["usable_for_fit"]:
            continue
        for index, coordinate in enumerate(row["coordinates"]):
            value = coordinate["normalized_value"]
            center, radius = spec["center"][index], spec["trust_radius"][index]
            if abs(value - center) > radius + 1e-14:
                violations.append({"sample_id": row["id"], "partition": row["partition"],
                                   "state_key": coordinate["state_key"], "index": coordinate["index"],
                                   "normalized_value": value, "center": center,
                                   "trust_radius": radius, "lower_bound": center - radius,
                                   "upper_bound": center + radius})
    return violations


def run(manifest_path, *, output_directory, run_missing=False):
    spec = load_manifest(manifest_path)
    output = Path(output_directory).resolve()
    audit_path, model_path = output / "calibration-audit.json", output / "local-reserve-model.json"
    if audit_path.exists() or model_path.exists():
        raise FileExistsError("Refusing to overwrite a calibration audit/model; choose a fresh output directory")
    train, holdout, rows, seen, branch_families = [], [], [], set(), set()
    source_context, source_model, prepared = None, None, []
    # Finish every receipt, partition, physical-context and coordinate check
    # before a potentially expensive/mutating missing-probe evaluation starts.
    for sample in spec["samples"]:
        source, payload, policy_provenance = _load_source(sample, spec)
        if source.prepared_problem_sha256 in seen:
            raise ValueError("A prepared checkpoint cannot be repeated or leak into holdout")
        seen.add(source.prepared_problem_sha256)
        if policy_provenance is not None:
            family = (policy_provenance["source_receipt_sha256"], policy_provenance["policy_id"])
            if family in branch_families:
                raise ValueError("Shared-prefix branch endpoints cannot count as independent samples")
            branch_families.add(family)
        if source_context is None:
            source_context, source_model = source.task_context_sha256, source.model_sha256
            physical_context = json.loads(source.task_context_json)
        elif source.task_context_sha256 != source_context or source.model_sha256 != source_model:
            raise ValueError("Calibration mixes physical task contexts or source models")
        coordinates, coordinate_audit = checkpoint_coordinates(source, spec["coordinates"])
        prepared.append((sample, source, payload, policy_provenance, coordinates, coordinate_audit))
    for sample, source, payload, policy_provenance, coordinates, coordinate_audit in prepared:
        result = sample["probe_result"] or output / "probes" / sample["id"] / f"{spec['side']}-task-reserve.json"
        if not result.exists():
            if not run_missing:
                raise FileNotFoundError(f"Missing probe {result}; use --run-missing to evaluate it")
            if sample["probe_result"] is not None:
                raise FileNotFoundError(f"Explicit read-only probe artifact is missing: {result}")
            if result.parent.exists() and any(result.parent.iterdir()):
                raise FileExistsError(f"Partial probe directory requires inspection before retry: {result.parent}")
            if spec["source_kind"] == BRANCH_SOURCE:
                result = run_from_checkpoint(source, payload, side=spec["side"],
                    work_scales=list(spec["work_scales"]), output_directory=result.parent,
                    tolerance=spec["tolerance"])
            else:
                result = run_probe(sample["receipt"], side=spec["side"], work_scales=list(spec["work_scales"]),
                                   output_directory=result.parent, tolerance=spec["tolerance"])
        estimate, probe_audit = load_probe_result(result, source, spec["work_scales"], spec["tolerance"])
        usable = estimate.usable_for_local_fit
        if usable:
            (train if sample["partition"] == "train" else holdout).append(LocalReserveSample(coordinates, estimate))
        rows.append({"id": sample["id"], "partition": sample["partition"], "receipt": str(sample["receipt"]),
                     "source_kind": spec["source_kind"], "policy_provenance": policy_provenance,
                     "receipt_sha256": sha256(sample["receipt"].read_bytes()).hexdigest(),
                     "completed_cycles": source.completed_cycles, "archive_sha256": source.archive_sha256,
                     "prepared_problem_sha256": source.prepared_problem_sha256,
                     "coordinates": coordinate_audit, "probe": probe_audit,
                     "usable_for_fit": usable, "excluded_reason": None if usable else "nominal_task_undetermined",
                     "reserve_lower_bound": estimate.reserve_lower_bound,
                     "largest_grid_factor_witnessed": max(spec["work_scales"]) in estimate.feasible_work_scales})
    conflicting_build_context = {
        key: value for key, value in spec["build_context"].items()
        if key in physical_context and physical_context[key] != value
    }
    if conflicting_build_context:
        raise ValueError("build_context conflicts with the certified physical task context")
    # These graph-structural values are declared before fitting and are later
    # compared against the actual compiled RHO by TaskReserveObjectiveBinding.
    # They do not alter the full-NLP witnesses or their source hashes.
    context = {**physical_context, **spec["build_context"],
               "coordinate_layout": [asdict(c) for c in spec["coordinates"]]}
    reasons = []
    if len(train) < spec["minimum_train_samples"]:
        reasons.append("insufficient_independent_training_samples")
    if len(holdout) < spec["minimum_holdout_samples"]:
        reasons.append("insufficient_independent_holdout_samples")
    trust_box_violations = _trust_box_violations(rows, spec)
    for partition in ("train", "holdout"):
        if any(item["partition"] == partition for item in trust_box_violations):
            reasons.append(f"{partition}_sample_outside_trust_box")
    fitted = None
    if not reasons:
        fitted = fit_local_reserve(train, center=spec["center"], trust_radius=spec["trust_radius"],
                                   holdout=holdout, maximum_error=spec["maximum_error"],
                                   maximum_condition_number=spec["maximum_condition_number"])
        reasons.extend(fitted.reasons)
        if np.ptp([sample.estimate.reserve_lower_bound for sample in train]) <= 1e-12:
            reasons.append("no_observed_reserve_contrast_for_directional_cost")
        fitted = replace(fitted, task_context_sha256=_hash(context),
                         accepted=not reasons, reasons=tuple(reasons))
    audit = {"schema_version": 1, "manifest_path": str(spec["path"]), "manifest_sha256": spec["sha256"],
             "side": spec["side"], "source_kind": spec["source_kind"],
             "experiment_plan": str(spec["plan_path"]) if spec["plan_path"] else None,
             "experiment_plan_sha256": spec["plan_sha256"],
             "accepted": fitted is not None and fitted.accepted, "reasons": reasons,
             "physical_task_context_sha256": source_context, "task_context": context,
             "task_context_sha256": _hash(context), "model_sha256": source_model,
             "work_scales": spec["work_scales"], "tolerance": spec["tolerance"],
             "sample_counts": {"train": len(train), "holdout": len(holdout), "manifest": len(rows)},
             "trust_box_violations": trust_box_violations,
             "fit": None if fitted is None else asdict(fitted), "samples": rows,
             "interpretation": "affine fit of observed feasible one-cycle work reserve; not maximum load or endurance",
             "globally_certified": False, "continuous_ode_replay_performed": False,
             "model_path": str(model_path) if fitted is not None and fitted.accepted else None}
    output.mkdir(parents=True, exist_ok=True)
    with audit_path.open("x", encoding="utf-8") as stream:
        json.dump(_json_finite(audit), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    if fitted is not None and fitted.accepted:
        with model_path.open("x", encoding="utf-8") as stream:
            json.dump({"schema_version": 1, "model": asdict(fitted), "task_context": context,
                       "calibration_audit": str(audit_path),
                       "calibration_audit_sha256": sha256(audit_path.read_bytes()).hexdigest()},
                      stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return audit_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--run-missing", action="store_true", help="Run missing probes, sequentially, from exact receipts")
    args = parser.parse_args(argv)
    print(run(args.manifest, output_directory=args.output_directory, run_missing=args.run_missing))


if __name__ == "__main__":
    main()
