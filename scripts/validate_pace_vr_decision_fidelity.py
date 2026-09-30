#!/usr/bin/env python3
"""Audit PACE-VR rankings against matched, certified bilateral RHO continuations.

This reader is intentionally resumable: rerun it as score and continuation JSON
files arrive. It never calls a surrogate, starts an OCP, or converts an IPOPT
failure into a physiological endpoint. A paired comparison requires a receipt
for an exact bilateral restart at the stated certified checkpoint.

Manifest schema (paths are relative to the manifest)::

  {"schema_version": 1, "checkpoints": [
    {"completed_cycles": 40, "horizon_cycles": 100,
     "restart_receipt": "cycle-40/receipt.json", "selected_candidate_id": "unit",
     "candidates": [
       {"id": "unit", "prediction_file": "cycle-40/unit/score.json",
        "continuation_summary": "cycle-40/unit/summary.json"}, ...]}
  ]}

The restart receipt must say ``exact_bilateral_restart: true``, identify the
same completed cycle, bind the configuration and each arm's model/prepared
problem to SHA-256 digests, and name content-hashed right and left primal
archives. Each arm must attest an exact replay round trip, including
stimulation history. This is a provenance gate, not an assertion that any
particular OCP is feasible.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import Any
from zipfile import BadZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.simulation.bilateral_endurance_campaign import assess_summary


DEFAULT_COMPARATORS = ("unit", "physio", "bo_fixed", "intermediate")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _source_path(manifest: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_absolute() else manifest.parent / path


def _sha256(value: Any) -> str | None:
    if (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)):
        return value
    return None


def _archive_metadata(path: Path, cycle: int) -> None:
    """Check that the hashed archive is a shifted one-cycle primal, not any NPZ."""
    with np.load(path, allow_pickle=False) as archive:
        if not any(key.startswith("states__") for key in archive.files):
            raise ValueError("checkpoint archive has no states")
        if not any(key.startswith("controls__") for key in archive.files):
            raise ValueError("checkpoint archive has no controls")
        if "metadata__json" not in archive.files:
            raise ValueError("checkpoint archive has no metadata")
        metadata = json.loads(str(archive["metadata__json"].item()))
        if not isinstance(metadata, dict) or metadata.get("producer_completed_windows") != cycle:
            raise ValueError("checkpoint archive has wrong source cycle")
        if metadata.get("producer_mode") != "rho_replay_checkpoint":
            raise ValueError("checkpoint archive is not a shifted RHO primal")


def _restart_gate(manifest: Path, checkpoint: dict[str, Any], cycle: int) -> dict[str, Any]:
    path = _source_path(manifest, checkpoint.get("restart_receipt"))
    if path is None or not path.is_file():
        return {"ready": False, "reason": "missing_exact_bilateral_restart"}
    try:
        receipt = _read_object(path)
        arms = receipt.get("arms")
        if (receipt.get("schema_version") != 1
                or receipt.get("checkpoint_kind") != "certified_bilateral_shifted_primal"
                or receipt.get("exact_bilateral_restart") is not True
                or type(receipt.get("completed_cycles")) is not int
                or receipt["completed_cycles"] != cycle
                or _sha256(receipt.get("configuration_sha256")) is None
                or not isinstance(arms, dict)
                or set(arms) != {"right", "left"}):
            raise ValueError("receipt does not certify the same exact bilateral boundary")
        configuration = _source_path(path, receipt.get("configuration_path"))
        if configuration is None or not configuration.is_file():
            raise ValueError("receipt has no source configuration file")
        if sha256(configuration.read_bytes()).hexdigest() != receipt["configuration_sha256"]:
            raise ValueError("source configuration SHA-256 mismatch")
        primal_files = {}
        for side in ("right", "left"):
            arm = arms.get(side)
            if not isinstance(arm, dict) or type(arm.get("completed_cycles")) is not int or arm["completed_cycles"] != cycle:
                raise ValueError(f"{side} receipt has wrong certified cycle")
            if (arm.get("certified") is not True or arm.get("stimulation_history_complete") is not True
                    or arm.get("replay_roundtrip_exact") is not True
                    or _sha256(arm.get("model_sha256")) is None
                    or _sha256(arm.get("prepared_problem_sha256")) is None
                    or arm.get("prepared_problem_sha256") != arm.get("restored_problem_sha256")):
                raise ValueError(f"{side} receipt lacks certified exact restart audit")
            model = _source_path(path, arm.get("model_path"))
            if model is None or not model.is_file():
                raise ValueError(f"{side} receipt has no source model file")
            if sha256(model.read_bytes()).hexdigest() != arm["model_sha256"]:
                raise ValueError(f"{side} model SHA-256 mismatch")
            expected_digest = _sha256(arm.get("sha256"))
            if expected_digest is None:
                raise ValueError(f"{side} archive has no SHA-256 digest")
            raw = arm.get("primal_path")
            primal = _source_path(path, raw)
            if primal is None or not primal.is_file():
                raise ValueError(f"missing {side} certified shifted primal")
            if sha256(primal.read_bytes()).hexdigest() != expected_digest:
                raise ValueError(f"{side} primal SHA-256 mismatch")
            _archive_metadata(primal, cycle)
            primal_files[side] = str(primal.resolve())
        return {"ready": True, "receipt": str(path.resolve()), "primal_files": primal_files}
    except (OSError, ValueError, TypeError, KeyError, BadZipFile, json.JSONDecodeError) as error:
        return {"ready": False, "reason": "invalid_exact_bilateral_restart",
                "detail": f"{type(error).__name__}: {error}"}


def _prediction(manifest: Path, candidate: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    path = _source_path(manifest, candidate.get("prediction_file"))
    if path is None or not path.is_file():
        return {"state": "pending"}
    try:
        prediction = _read_object(path)
        # The rollout can fail independently of the physical RHO continuation.
        if prediction.get("accepted") is not True:
            return {"state": "surrogate_failure", "status": prediction.get("status"),
                    "reason": prediction.get("reason"), "source": str(path)}
        prefix = prediction.get("feasible_prefix_cycles")
        margin = _finite(prediction.get("minimum_task_margin"))
        value = _finite(prediction.get("terminal_value"))
        work = _finite(prediction.get("work_residual_max"))
        violation = _finite(prediction.get("constraint_violation_max"))
        if (type(prefix) is not int or prefix < 0 or margin is None or value is None
                or work is None or work < 0 or violation is None or violation < 0):
            raise ValueError("accepted score lacks finite prefix, margin, value, or residual diagnostics")
        max_work = _finite(checkpoint.get("maximum_work_residual"))
        max_violation = _finite(checkpoint.get("maximum_constraint_violation"))
        if max_work is not None and work > max_work or max_violation is not None and violation > max_violation:
            return {"state": "surrogate_rejected_by_residual_gate", "source": str(path),
                    "work_residual_max": work, "constraint_violation_max": violation}
        sign = 1.0 if checkpoint.get("terminal_value_higher_is_better", True) else -1.0
        return {"state": "valid", "source": str(path), "status": prediction.get("status"),
                "feasible_prefix_cycles": prefix, "minimum_task_margin": margin,
                "terminal_value": value, "work_residual_max": work,
                "constraint_violation_max": violation,
                "rank_key": [prefix, margin, sign * value],
                "runtime_s": _finite(prediction.get("runtime_s")),
                "reallocation": prediction.get("reallocation"), "local_fit": prediction.get("local_fit")}
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return {"state": "unreadable", "source": str(path),
                "detail": f"{type(error).__name__}: {error}"}


def _continuation(manifest: Path, candidate: dict[str, Any], horizon: int,
                  restart_ready: bool) -> dict[str, Any]:
    path = _source_path(manifest, candidate.get("continuation_summary"))
    if not restart_ready:
        return {"state": "unpaired", "reason": "missing_exact_bilateral_restart",
                "source": str(path) if path is not None else None}
    if path is None or not path.is_file():
        return {"state": "pending"}
    try:
        summary = _read_object(path)
        observation = assess_summary(summary, horizon)
        completed = observation.validated_cycles
        if observation.status == "feasible":
            state, interval = "horizon_censored", [horizon, None]
        elif (observation.status == "infeasible"
              and (str(observation.reason).startswith("fatigue_counterfactual_certified")
                   or summary.get("failure_kind") == "fatigue_limit")):
            state, interval = "physiological_failure_certified", [completed, completed]
        elif observation.status == "infeasible":
            state, interval = "other_limit_certified", None
        else:
            state, interval = observation.status, None
        return {"state": state, "source": str(path), "certified_common_cycles": completed,
                "remaining_cycle_interval": interval, "endpoint_reason": observation.reason,
                "assessment_status": observation.status}
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return {"state": "unreadable", "source": str(path),
                "detail": f"{type(error).__name__}: {error}"}


def _average_ranks(values: list[float]) -> list[float]:
    indices = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indices):
        end = position + 1
        while end < len(indices) and values[indices[end]] == values[indices[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in indices[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def _spearman(predicted: list[float], observed: list[float]) -> float | None:
    if len(predicted) < 3:
        return None
    first, second = _average_ranks(predicted), _average_ranks(observed)
    mid_first, mid_second = sum(first) / len(first), sum(second) / len(second)
    numerator = sum((a - mid_first) * (b - mid_second) for a, b in zip(first, second))
    denominator = math.sqrt(sum((a - mid_first) ** 2 for a in first)
                            * sum((b - mid_second) ** 2 for b in second))
    return numerator / denominator if denominator else None


def _actual_order(a: dict[str, Any], b: dict[str, Any]) -> int | None:
    """Return +1 if a outlasts b, -1 if b outlasts a, else unknown."""
    first, second = a.get("remaining_cycle_interval"), b.get("remaining_cycle_interval")
    if first is None or second is None:
        return None
    if second[1] is not None and first[0] > second[1]:
        return 1
    if first[1] is not None and second[0] > first[1]:
        return -1
    return None


def _checkpoint_comparison(checkpoint: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [entry for entry in entries if entry["prediction"]["state"] == "valid"]
    ordered = sorted(valid, key=lambda entry: tuple(entry["prediction"]["rank_key"]), reverse=True)
    position = 0
    while position < len(ordered):
        end = position + 1
        key = ordered[position]["prediction"]["rank_key"]
        while end < len(ordered) and ordered[end]["prediction"]["rank_key"] == key:
            end += 1
        rank = (position + 1 + end) / 2.0
        for entry in ordered[position:end]:
            entry["predicted_rank"] = rank
        position = end
    selected_id = checkpoint.get("selected_candidate_id")
    selected = next((entry for entry in entries if entry["candidate_id"] == selected_id), None)
    comparable, concordant, discordant = 0, 0, 0
    for index, first in enumerate(valid):
        for second in valid[index + 1:]:
            actual = _actual_order(first["continuation"], second["continuation"])
            a, b = tuple(first["prediction"]["rank_key"]), tuple(second["prediction"]["rank_key"])
            predicted = (a > b) - (a < b)
            if actual is None or predicted == 0:
                continue
            comparable += 1
            concordant += actual == predicted
            discordant += actual != predicted
    exact = [entry for entry in valid if entry["continuation"]["state"] == "physiological_failure_certified"]
    all_exact = len(exact) == len(valid) and len(valid) >= 3
    spearman = None
    if all_exact:
        predicted_ranks = [entry["predicted_rank"] for entry in valid]
        observed_cycles = [entry["continuation"]["certified_common_cycles"] for entry in valid]
        spearman = _spearman([-rank for rank in predicted_ranks], observed_cycles)
    regret_lower_bound = regret_exact = None
    if selected is not None and selected["continuation"]["state"] == "physiological_failure_certified":
        selected_cycles = selected["continuation"]["certified_common_cycles"]
        intervals = [entry["continuation"].get("remaining_cycle_interval") for entry in entries]
        available = [interval for interval in intervals if interval is not None]
        if available:
            regret_lower_bound = max(0, max(interval[0] for interval in available) - selected_cycles)
            if len(available) == len(entries) and all(interval[1] is not None for interval in available):
                regret_exact = regret_lower_bound
    required = tuple(checkpoint.get("required_candidates", DEFAULT_COMPARATORS))
    return {
        "completed_cycles": checkpoint["completed_cycles"], "horizon_cycles": checkpoint["horizon_cycles"],
        "selected_candidate_id": selected_id,
        "predicted_best_candidate_id": ordered[0]["candidate_id"] if ordered else None,
        "candidate_coverage": {"required": list(required),
                               "missing": sorted(set(required) - {entry["candidate_id"] for entry in entries}),
                               "valid_surrogate_scores": len(valid), "total": len(entries)},
        "pairwise_comparable": comparable, "pairwise_concordant": concordant,
        "pairwise_discordant": discordant,
        "pairwise_concordance": concordant / comparable if comparable else None,
        "spearman_exact_endurance": spearman,
        "selected_regret_lower_bound_cycles": regret_lower_bound,
        "selected_regret_exact_cycles": regret_exact,
        "surrogate_failure_count": sum(entry["prediction"]["state"] not in {"valid", "pending"}
                                       for entry in entries),
        "physiological_failure_count": sum(entry["continuation"]["state"] == "physiological_failure_certified"
                                           for entry in entries),
    }


def summarize(manifest_path: Path, output_path: Path | None = None) -> Path:
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    manifest = _read_object(manifest_path)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("checkpoints"), list):
        raise ValueError("PACE-VR validation manifest requires schema_version=1 and checkpoints list")
    output_path = output_path or manifest_path.parent / "pace-vr-validation-summary.json"
    comparisons, all_entries = [], []
    for checkpoint in manifest["checkpoints"]:
        if not isinstance(checkpoint, dict):
            raise ValueError("Each checkpoint must be an object")
        cycle, horizon = checkpoint.get("completed_cycles"), checkpoint.get("horizon_cycles")
        if type(cycle) is not int or cycle < 1 or type(horizon) is not int or horizon < 1:
            raise ValueError("Checkpoint completed_cycles and horizon_cycles must be positive integers")
        candidates = checkpoint.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Each checkpoint needs candidates")
        ids = [item.get("id") for item in candidates if isinstance(item, dict)]
        if (len(ids) != len(candidates)
                or any(not isinstance(item, str) or not item for item in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("Candidate IDs must be distinct nonempty strings")
        if checkpoint.get("selected_candidate_id") not in ids:
            raise ValueError("selected_candidate_id must identify a checkpoint candidate")
        gate = _restart_gate(manifest_path, checkpoint, cycle)
        entries = [{"checkpoint_completed_cycles": cycle, "candidate_id": item["id"],
                    "prediction": _prediction(manifest_path, item, checkpoint),
                    "continuation": _continuation(manifest_path, item, horizon, gate["ready"])}
                   for item in candidates]
        comparison = _checkpoint_comparison(checkpoint, entries)
        comparison["exact_restart"] = gate
        comparisons.append(comparison)
        all_entries.extend(entries)
    document = {"schema_version": 1, "kind": "pace_vr_decision_fidelity",
                "manifest": str(manifest_path), "comparisons": comparisons, "entries": all_entries,
                "interpretation": {
                    "surrogate_score": "higher feasible prefix, then higher minimum task margin, then higher terminal value",
                    "pairwise_concordance": "counted only where certified cycle intervals establish strict order",
                    "spearman": "reported only when every valid prediction has an exact fatigue-certified endpoint",
                    "regret": "remaining certified cycles versus best observed candidate; censored results provide a lower bound",
                    "missing_restart": "unpaired continuations are never ranked as causal checkpoint interventions",
                }}
    _atomic_json(output_path, document)
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    print(summarize(args.manifest, args.output), flush=True)


if __name__ == "__main__":
    main()
