import json

import pytest

from cocofest.simulation.independent_arms_gui_model import (
    IndependentArmsGuiConfig,
    independent_arms_summary,
)


def test_work_and_equivalent_torque_are_the_same_full_turn_target():
    config = IndependentArmsGuiConfig(
        right_work_j_per_cycle=2.0 * 3.141592653589793 * .2,
        right_equivalent_mean_torque_nm=.2,
        left_equivalent_mean_torque_nm=.1,
        factory="example:build",
    ).validate()
    assert config.target_work_j("right") == pytest.approx(2.0 * 3.141592653589793 * .2)
    runtime = config.to_runtime_dict()
    assert runtime["formulation"] == "isokinetic"
    assert runtime["omega_rad_s"] < 0
    assert runtime["right_equivalent_mean_torque_nm"] == pytest.approx(.2)
    assert runtime["right_target_work_j_per_cycle"] == pytest.approx(2.0 * 3.141592653589793 * .2)
    assert json.loads(config.to_json())["factory"] == "example:build"


def test_rejects_dynamic_or_inconsistent_work_torque_pair():
    with pytest.raises(ValueError, match="isocinétique"):
        IndependentArmsGuiConfig(formulation="dynamic").validate()
    with pytest.raises(ValueError, match="ne concordent pas"):
        IndependentArmsGuiConfig(right_work_j_per_cycle=1, right_equivalent_mean_torque_nm=.1).validate()


def test_synthesis_understands_coordinator_arms_layout_and_never_infers_success():
    payload = {
        "arms": {
            "right": {"equivalent_mean_torque_nm": .2, "target_work_j_per_cycle": 1.256,
                      "metrics": {"success": True, "validated_cycles": 100, "solver_time_s": .5}},
            "left": {"equivalent_mean_torque_nm": .1, "target_work_j_per_cycle": .628,
                     "metrics": {"success": False, "validated_cycles": 4}},
        }
    }
    text = independent_arms_summary(payload, 100)
    assert "Bras droit : réussite déclarée" in text
    assert "Bras gauche : échec déclaré" in text
    assert "τeq 0.2 N.m" in text
