#!/usr/bin/env python3
"""Add one audited capacity-frontier diagnostic policy to a short RHO branch plan.

This is intentionally a *short* paired-branch registration.  A subsequent
continuation-to-frozen-feasibility campaign is allowed only if this branch
preserves the nominal task and passes the complete independent NLP audit.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.task_reserve_branch import json_digest, validate_policy  # noqa: E402


def build(plan_path: Path, proposal_path: Path, *, branch_id: str = "frontier-diagnostic") -> dict:
    plan_path, proposal_path = Path(plan_path).resolve(strict=True), Path(proposal_path).resolve(strict=True)
    plan, proposal = json.loads(plan_path.read_text()), json.loads(proposal_path.read_text())
    if plan.get("kind") != "local_task_reserve_experiment_plan" or plan.get("activation_allowed") is not False:
        raise ValueError("Expected a non-activated local task-reserve branch plan")
    if proposal.get("kind") != "capacity_frontier_diagnostic_rho_weight_proposal":
        raise ValueError("Expected a capacity-frontier diagnostic weight proposal")
    if proposal.get("status") != "offline_diagnostic_only_requires_causal_RHO_ablation":
        raise ValueError("Only explicitly diagnostic frontier proposals can be registered")
    design = deepcopy(plan["reachable_branch_design"])
    names = tuple(design["muscle_order"])
    weights = proposal.get("fatigue_weights", {})
    if set(weights) != set(names):
        raise ValueError("Proposal muscle names differ from the existing RHO plan")
    values = np.asarray([float(weights[name]) for name in names])
    if not np.all(np.isfinite(values)) or np.any(values <= 0) or not math.isclose(float(values.mean()), 1., abs_tol=1e-12):
        raise ValueError("Proposal fatigue weights must be finite, positive and mean-normalized")
    if any(row["id"] == branch_id for row in design["branches"]):
        raise ValueError("Branch id is already present")
    direction = np.log(values)
    direction -= direction.mean()
    candidate = {
        "id": branch_id,
        "partition": "train",
        "endpoint_cycles": [1, 3],
        "relative_log_direction": direction.tolist(),
        "absolute_fatigue_weights_from_unit": {name: float(weights[name]) for name in names},
        "starting_checkpoint": "exact_anchor_receipt",
        "nominal_work_scale": 1.,
        "origin": "capacity_frontier_checkpoint_interpolation_diagnostic",
        "source_proposal": str(proposal_path),
        "source_proposal_sha256": sha256(proposal_path.read_bytes()).hexdigest(),
    }
    validate_policy(candidate, names)
    design["branches"].append(candidate)
    design["estimated_nominal_rho_cycles_per_anchor"] = sum(max(row["endpoint_cycles"]) for row in design["branches"])
    design["endpoint_count_per_anchor"] = sum(len(row["endpoint_cycles"]) for row in design["branches"])
    result = deepcopy(plan)
    result["reachable_branch_design"] = design
    result["capacity_frontier_policy_ablation"] = {
        "source_plan": str(plan_path), "source_plan_sha256": sha256(plan_path.read_bytes()).hexdigest(),
        "source_proposal": str(proposal_path), "source_proposal_sha256": sha256(proposal_path.read_bytes()).hexdigest(),
        "candidate_branch": branch_id,
        "paired_control_branch": "unit",
        "required_next_gate": "paired short branches must independently audit nominal work before any long continuation",
        "long_continuation_endpoint": "last_certified_cycle_then_frozen_feasibility_with_alternative_starts",
        "activation_allowed": False,
    }
    # Existing external hashes still refer to the original calibration, while
    # this digest seals the newly registered policy design.
    body = deepcopy(result)
    return {**body, "capacity_frontier_branch_plan_sha256": json_digest(body)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--branch-id", default="frontier-diagnostic")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = build(args.plan, args.proposal, branch_id=args.branch_id)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
