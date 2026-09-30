#!/usr/bin/env python3
"""Score alternative muscle weights from one certified PACE-VR snapshot.

This is a matched *surrogate* comparison: each candidate starts from the same
certified terminal Ding state and PW history, with the same sampled geometry
and work demand. It does not restart the full RHO or certify endurance.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_pace_vr_rollout import load_case


def _weights(raw, label, muscles):
    weights = np.asarray(raw, dtype=float)
    if weights.shape != (muscles,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError(f"{label} requires {muscles} finite positive muscle weights")
    return weights


def _proposal_at_source(path: Path, source_cycle: int, muscles: int):
    matches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("status") == "applied" and event.get("source_cycle") == source_cycle:
            matches.append(event)
    if len(matches) != 1:
        raise ValueError(f"Expected one applied PACE-VR proposal from cycle {source_cycle}; found {len(matches)}")
    return _weights(matches[0]["weights_after"], "applied proposal", muscles), matches[0]


def _candidate_set(incumbent, proposal=None, extras=None):
    candidates = {"incumbent": incumbent}
    if proposal is not None:
        change = np.log(proposal / incumbent)
        candidates.update(proposal=proposal,
                          half_step=incumbent * np.exp(0.5 * change),
                          opposite_step=incumbent * np.exp(-change))
    for name, weights in (extras or {}).items():
        if not isinstance(name, str) or not name or name in candidates:
            raise ValueError(f"Invalid or duplicate candidate name: {name!r}")
        candidates[name] = _weights(weights, name, len(incumbent))
    return candidates


def _score(supervisor, states, widths, candidates):
    entries = []
    for name, weights in candidates.items():
        result = supervisor.evaluate(states, widths, certified=True,
                                     weights=weights, candidate_name=name)
        prefix = result["feasible_prefix_cycles"]
        margin = result["minimum_task_margin"]
        value = result["terminal_value"]
        accepted = bool(result["accepted"] and margin is not None
                        and math.isfinite(margin) and math.isfinite(value))
        entries.append({
            "candidate_id": name,
            "weights": np.asarray(weights, dtype=float).tolist(),
            "normalized_weights_used": result["candidate"]["weights"],
            "surrogate_accepted_full_horizon": accepted,
            "surrogate_status": result["status"],
            "feasible_prefix_cycles": prefix,
            "minimum_task_margin": margin,
            "terminal_value": value,
            "work_residual_max_j": result["work_residual_max"],
            "constraint_violation_max": result["constraint_violation_max"],
            "runtime_s": result["runtime_s"],
            "first_failure": result["first_failure"],
            "qp_backends": sorted(set(result["reallocation"]["qp_backends"])),
        })
    # An incomplete projection still has a useful *surrogate* prefix. Margin
    # and terminal score break only equal-prefix ties, never override length.
    entries.sort(key=lambda item: (item["feasible_prefix_cycles"],
                                   item["minimum_task_margin"] if item["minimum_task_margin"] is not None else -math.inf,
                                   -item["terminal_value"]), reverse=True)
    for position, entry in enumerate(entries, 1):
        entry["surrogate_rank"] = position
    return entries


def benchmark(source: Path, cycle: int, horizon: int, *, proposal_journal=None,
              extra_candidates=None, margin_tie_tolerance=1e-5):
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not math.isfinite(margin_tie_tolerance) or margin_tie_tolerance < 0:
        raise ValueError("margin tie tolerance must be finite and nonnegative")
    supervisor, payload, snapshot = load_case(source, cycle)
    supervisor.config = replace(supervisor.config, horizon_cycles=horizon)
    incumbent = _weights(payload.get("weights"), "snapshot incumbent", supervisor.muscles)
    proposal, receipt = (None, None)
    if proposal_journal is not None:
        proposal, receipt = _proposal_at_source(Path(proposal_journal), cycle, supervisor.muscles)
    candidates = _candidate_set(incumbent, proposal, extra_candidates)
    entries = _score(supervisor, payload["initial_states"], payload["pulse_widths"], candidates)
    best = entries[0]
    incumbent_score = next(entry for entry in entries if entry["candidate_id"] == "incumbent")
    same_prefix = best["feasible_prefix_cycles"] == incumbent_score["feasible_prefix_cycles"]
    margin_gain = (None if best["minimum_task_margin"] is None or incumbent_score["minimum_task_margin"] is None
                   else best["minimum_task_margin"] - incumbent_score["minimum_task_margin"])
    return {
        "schema_version": 1,
        "kind": "pace_vr_matched_surrogate_candidate_ranking",
        "source": str(Path(source).resolve()),
        "source_sha256": sha256(Path(source).read_bytes()).hexdigest(),
        "source_cycle": cycle,
        "source_context_digest": snapshot["context_digest"],
        "horizon_cycles": horizon,
        "proposal_receipt": receipt,
        "entries": entries,
        "surrogate_best_candidate_id": entries[0]["candidate_id"],
        "best_versus_incumbent": {
            "feasible_prefix_gain_cycles": best["feasible_prefix_cycles"] - incumbent_score["feasible_prefix_cycles"],
            "minimum_task_margin_gain": margin_gain,
            "minimum_task_margin_tie_tolerance": margin_tie_tolerance,
            "margin_difference_exceeds_tolerance": bool(same_prefix and margin_gain is not None
                                                         and margin_gain > margin_tie_tolerance),
        },
        "scope": "same certified Ding state, PW history, geometry and cycle work; compact PACE-VR rollout only",
        "full_rho_restarted": False,
        "physiological_failure_certified": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="One arm's result.json")
    parser.add_argument("--source-cycle", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--margin-tie-tolerance", type=float, default=1e-5)
    parser.add_argument("--proposal-journal", type=Path, help="One arm's pace_vr.jsonl")
    parser.add_argument("--candidates", type=Path, help='JSON object {"candidate_id": [weights, ...]}')
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    extras = json.loads(args.candidates.read_text()) if args.candidates else None
    if extras is not None and not isinstance(extras, dict):
        parser.error("--candidates must be a JSON object")
    report = benchmark(args.source, args.source_cycle, args.horizon,
                       proposal_journal=args.proposal_journal, extra_candidates=extras,
                       margin_tie_tolerance=args.margin_tie_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"source_cycle": args.source_cycle, "horizon": args.horizon,
                      "ranked": [(entry["candidate_id"], entry["feasible_prefix_cycles"])
                                 for entry in report["entries"]]}))


if __name__ == "__main__":
    main()
