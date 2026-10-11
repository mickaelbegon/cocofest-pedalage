#!/usr/bin/env python3
"""Materialize a diagnostic RHO weight proposal from a passing frontier holdout."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.capacity_frontier_interpolation import diagnostic_frontier_weight_hint  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpolation-audit", type=Path, required=True)
    parser.add_argument("--triceps-multiplier", type=float, default=2.)
    parser.add_argument("--lower", type=float, default=.25)
    parser.add_argument("--upper", type=float, default=4.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit_path = args.interpolation_audit.resolve(strict=True)
    audit = json.loads(audit_path.read_text())
    if not audit.get("passed"):
        raise ValueError("A passing offline checkpoint interpolation audit is required")
    if audit.get("kind") == "offline_checkpoint_exchange_interpolation_holdout":
        slopes = audit.get("conservative_slope_interval", {})
        conservative = {name: float(interval[0]) for name, interval in slopes.items()}
        mapping = "inverse_conservative_triceps_exchange_slope; triceps amplified then mean-normalized"
    elif audit.get("kind") == "offline_local_epsilon_slope_holdout":
        conservative = {name: float(value) for name, value in audit.get("predicted_conservative_exchange_slopes", {}).items()}
        mapping = "inverse lower local-epsilon exchange slope; triceps amplified then mean-normalized"
    else:
        raise ValueError("Unsupported frontier interpolation-audit kind")
    weights = diagnostic_frontier_weight_hint(conservative, triceps_multiplier=args.triceps_multiplier,
                                               lower=args.lower, upper=args.upper)
    order = tuple(audit["muscle_order"])
    if set(order) != set(weights):
        raise ValueError("Interpolation muscle order does not match the inferred fatigue weights")
    body = {
        "schema_version": 1,
        "kind": "capacity_frontier_diagnostic_rho_weight_proposal",
        "source_audit": str(audit_path),
        "source_audit_sha256": sha256(audit_path.read_bytes()).hexdigest(),
        "fatigue_weights": {name: weights[name] for name in order},
        "mapping": mapping,
        "status": "offline_diagnostic_only_requires_causal_RHO_ablation",
        "not_a_feasibility_certificate": True,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump({**body, "proposal_sha256": sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
                  stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
