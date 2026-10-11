import json
from hashlib import sha256

import pytest

from scripts.add_capacity_frontier_policy_branch import build


def test_frontier_proposal_is_registered_as_a_short_paired_diagnostic_branch(tmp_path):
    plan = {
        "kind": "local_task_reserve_experiment_plan", "activation_allowed": False,
        "reachable_branch_design": {"muscle_order": ["A", "B", "Triceps"], "branches": [{
            "id": "unit", "partition": "train", "endpoint_cycles": [1, 3],
            "relative_log_direction": [0., 0., 0.],
            "absolute_fatigue_weights_from_unit": {"A": 1., "B": 1., "Triceps": 1.},
            "starting_checkpoint": "exact_anchor_receipt", "nominal_work_scale": 1.,
        }]},
    }
    proposal = {
        "kind": "capacity_frontier_diagnostic_rho_weight_proposal",
        "status": "offline_diagnostic_only_requires_causal_RHO_ablation",
        "fatigue_weights": {"A": .75, "B": .9, "Triceps": 1.35},
    }
    plan_path, proposal_path = tmp_path / "plan.json", tmp_path / "proposal.json"
    plan_path.write_text(json.dumps(plan)); proposal_path.write_text(json.dumps(proposal))
    value = build(plan_path, proposal_path)
    branch = value["reachable_branch_design"]["branches"][-1]
    assert branch["id"] == "frontier-diagnostic"
    assert branch["endpoint_cycles"] == [1, 3]
    assert sum(branch["absolute_fatigue_weights_from_unit"].values()) / 3 == pytest.approx(1.)
    assert value["capacity_frontier_policy_ablation"]["activation_allowed"] is False
    assert len(value["capacity_frontier_branch_plan_sha256"]) == 64


def test_frontier_branch_refuses_a_proposal_without_the_diagnostic_status(tmp_path):
    plan = {"kind": "local_task_reserve_experiment_plan", "activation_allowed": False,
            "reachable_branch_design": {"muscle_order": ["A", "Triceps"], "branches": []}}
    proposal = {"kind": "capacity_frontier_diagnostic_rho_weight_proposal", "status": "active",
                "fatigue_weights": {"A": 1., "Triceps": 1.}}
    plan_path, proposal_path = tmp_path / "plan.json", tmp_path / "proposal.json"
    plan_path.write_text(json.dumps(plan)); proposal_path.write_text(json.dumps(proposal))
    with pytest.raises(ValueError, match="diagnostic"):
        build(plan_path, proposal_path)
