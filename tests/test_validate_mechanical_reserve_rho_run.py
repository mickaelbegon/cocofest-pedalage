"""Regression tests for the runtime reserve RHO audit."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_mechanical_reserve_rho_run.py"
spec = importlib.util.spec_from_file_location("validate_mechanical_reserve_rho_run", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def _summary():
    arm = {
        "cycles": [
            {"cycle": 1, "certified": True,
             "mechanical_reserve_cost": {"activation": 0.0, "attainable_work_certified": False},
             "mechanical_reserve_solver": {"solver_observed": True, "compiled_solver_reuse_verified": False,
                                           "solver_identity": 42}},
            {"cycle": 2, "certified": True,
             "mechanical_reserve_cost": {"activation": 1.0, "attainable_work_certified": False},
             "mechanical_reserve_solver": {"solver_observed": True, "compiled_solver_reuse_verified": True,
                                           "solver_identity": 42}},
        ],
        "mechanical_reserve_binding": {
            "objective_graph_build_count": 1, "objective_graph_rebuild_required": False,
            "nlp_attached": True, "compiled_solver_reuse_verified": True,
            "compiled_solver_observation_count": 2,
        },
        "mechanical_reserve_events": [{
            "status": "updated", "source_cycle": 1, "application_cycle": 2,
            "attainable_work_certified": False, "endurance_prediction": False,
            "parameter_update": {"nlp_reused": True, "objective_graph_rebuild_required": False,
                                 "objective_graph_build_count": 1, "parameter_update_count": 1},
        }],
    }
    return {"requested_cycles": 2, "completed_rho_cycles": 2, "success": True,
            "failure": None, "arms": {"right": copy.deepcopy(arm), "left": copy.deepcopy(arm)}}


def test_valid_two_arm_run():
    report = module.validate_summary(_summary())
    assert report["passed"] is True
    assert report["arms"]["right"]["activation_first_two"] == [0.0, 1.0]


def test_rejects_uncertified_cycle_and_false_future_claim():
    summary = _summary()
    summary["arms"]["left"]["cycles"][1]["certified"] = False
    summary["arms"]["right"]["mechanical_reserve_events"][0]["endurance_prediction"] = True
    report = module.validate_summary(summary)
    assert report["passed"] is False
    assert any("consecutively certified" in failure for failure in report["failures"])
    assert any("endurance prediction" in failure for failure in report["failures"])


def test_rejects_graph_rebuild_or_solver_replacement():
    summary = _summary()
    summary["arms"]["right"]["mechanical_reserve_binding"]["objective_graph_build_count"] = 2
    summary["arms"]["left"]["cycles"][1]["mechanical_reserve_solver"]["solver_identity"] = 99
    report = module.validate_summary(summary)
    assert report["passed"] is False
    assert any("built exactly once" in failure for failure in report["failures"])
    assert any("identity changed" in failure for failure in report["failures"])


def test_rejects_late_activation_and_missing_boundary_update():
    summary = _summary()
    summary["arms"]["right"]["cycles"][1]["mechanical_reserve_cost"]["activation"] = 0.0
    summary["arms"]["left"]["mechanical_reserve_events"][0]["application_cycle"] = 3
    report = module.validate_summary(summary)
    assert report["passed"] is False
    assert any("expected activation" in failure for failure in report["failures"])
    assert any("cycle 1 to cycle 2 update" in failure for failure in report["failures"])


def test_rejects_future_endurance_claim_in_later_update():
    summary = _summary()
    summary["arms"]["right"]["mechanical_reserve_events"].append({
        "status": "updated", "source_cycle": 20, "application_cycle": 21,
        "attainable_work_certified": False, "endurance_prediction": True,
    })
    report = module.validate_summary(summary)
    assert report["passed"] is False
    assert any("event 2 mislabels" in failure for failure in report["failures"])
