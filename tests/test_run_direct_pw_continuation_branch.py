import json

import numpy as np
import pytest

from cocofest.optimization.continuation_value_pw_action import direct_pw_continuation_plan
from scripts.run_direct_pw_continuation_branch import resolve_job
from scripts.plan_direct_pw_continuation_value import paired_directions


def _plan(tmp_path):
    digest = "a" * 64
    anchors = [{"id": f"c{cycle}", "completed_cycles": cycle, "source_receipt": f"/tmp/c{cycle}.json",
                "source_receipt_sha256": digest, "prepared_problem_sha256": digest}
               for cycle in (120, 140, 160)]
    directions = {"d1": [1 / np.sqrt(2), -1 / np.sqrt(2)],
                  "d2": [-1 / np.sqrt(2), 1 / np.sqrt(2)],
                  "mixed": [1 / np.sqrt(2), 1 / np.sqrt(2)]}
    value = direct_pw_continuation_plan(anchors, muscle_names=("Biceps", "Triceps"), directions=directions,
                                        amplitude_fraction=.05)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(value))
    return path, value


def test_resolve_job_requires_predeclared_direct_action(tmp_path):
    path, plan = _plan(tmp_path)
    job = plan["jobs"][1]
    loaded, anchor, resolved = resolve_job(path, job["id"])
    assert loaded["kind"] == "direct_pw_continuation_value_protocol"
    assert anchor["id"] == "c120"
    assert resolved == job


def test_resolve_job_refuses_weight_activation_or_unplanned_action(tmp_path):
    path, plan = _plan(tmp_path)
    plan["fatigue_weight_updates_allowed"] = True
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_job(path, plan["jobs"][0]["id"])
    plan["fatigue_weight_updates_allowed"] = False
    plan["jobs"][0]["action"] = {"id": "invented"}
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="PW job"):
        resolve_job(path, plan["jobs"][0]["id"])


def test_fixed_policy_continuation_recognizes_direct_pw_endpoint_kind():
    # The full endpoint loader remains responsible for provenance and archive
    # validation; this is a dispatch contract only.
    import inspect
    from scripts import run_fixed_policy_branch_continuation as continuation
    source = inspect.getsource(continuation.run)
    assert "certified_direct_pw_continuation_value_endpoint" in source
    assert "load_direct_pw_endpoint" in source


def test_default_paired_directions_are_orthonormal_with_a_mixed_holdout():
    directions = paired_directions(("Delt_ant", "Delt_post", "Biceps", "Triceps"))
    values = list(map(np.asarray, directions.values()))
    assert np.isclose(np.dot(values[0], values[1]), 0.)
    assert np.isclose(np.linalg.norm(values[2]), 1.)
    with pytest.raises(ValueError, match="require muscle order"):
        paired_directions(("Biceps", "Triceps"))
