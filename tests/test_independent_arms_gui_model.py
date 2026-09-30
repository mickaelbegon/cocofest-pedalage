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


def test_capacity_feedback_exports_one_fixed_total_work_budget():
    config = IndependentArmsGuiConfig(
        factory="example:build", right_equivalent_mean_torque_nm=.2,
        left_equivalent_mean_torque_nm=.1, resistance_pace_policy="capacity_feedback",
        cycles_per_window=1, resistance_pace_minimum_arm_torque_nm=.05,
    ).validate()
    pace = config.to_runtime_dict()["resistance_pace"]
    assert pace["total_equivalent_mean_torque_nm"] == pytest.approx(.3)
    assert pace["initial_right_fraction"] == pytest.approx(2 / 3)
    assert pace["minimum_arm_equivalent_mean_torque_nm"] == pytest.approx(.05)
    assert IndependentArmsGuiConfig(
        factory="example:build", parametric_fatigue_weights=True,
    ).to_runtime_dict()["parametric_fatigue_weights"] is True
    with pytest.raises(ValueError, match="fenêtre d'un cycle"):
        IndependentArmsGuiConfig(
            factory="example:build", cycles_per_window=2,
            resistance_pace_policy="capacity_feedback",
        ).to_runtime_dict()


def test_calibrated_initial_split_exports_a_synchronous_pace_with_fixed_follow_up():
    config = IndependentArmsGuiConfig(
        cycles=2, right_equivalent_mean_torque_nm=.1, left_equivalent_mean_torque_nm=.1,
        resistance_pace_policy="fixed",
        resistance_pace_initial_split_policy="capacity_fatigability_after_first_cycle",
    )
    pace = config.to_runtime_dict()["resistance_pace"]
    assert pace["capacity_feedback"] is False
    assert pace["initial_split_policy"] == "capacity_fatigability_after_first_cycle"


def test_calibrated_initial_split_requires_a_following_rho_cycle():
    with pytest.raises(ValueError, match="au moins deux cycles"):
        IndependentArmsGuiConfig(
            cycles=1, resistance_pace_initial_split_policy="capacity_fatigability_after_first_cycle",
        ).validate()


def test_physio_u_exports_parametric_cycle_synchronous_weight_controller():
    config = IndependentArmsGuiConfig(
        factory="example:build", parametric_fatigue_weights=True,
        muscle_weight_policy="physio_u", physio_update_every_cycles=20,
        resistance_pace_policy="capacity_feedback",
        resistance_pace_update_every_cycles=20,
    ).validate()
    runtime = config.to_runtime_dict()
    pace = runtime["muscle_pace"]
    assert runtime["parametric_fatigue_weights"] is True
    assert pace["adaptation_strategy"] == "physio_update"
    assert pace["right_initial_weights"] == {"Delt_ant": 1., "Delt_post": 1., "Biceps": 1., "Triceps": 1.}
    assert pace["left_initial_weights"] == pace["right_initial_weights"]
    with pytest.raises(ValueError, match="paramétriques"):
        IndependentArmsGuiConfig(factory="example:build", muscle_weight_policy="physio_u").validate()
    with pytest.raises(ValueError, match="même cadence"):
        IndependentArmsGuiConfig(
            factory="example:build", parametric_fatigue_weights=True, muscle_weight_policy="physio_u",
            physio_update_every_cycles=20, resistance_pace_policy="capacity_feedback",
            resistance_pace_update_every_cycles=10,
        ).validate()


def test_mechanics_only_ablation_exports_a_distinct_parametric_controller():
    config = IndependentArmsGuiConfig(
        factory="example:build", parametric_fatigue_weights=True,
        muscle_weight_policy="mechanical_sensitivity_squared_v1_experimental",
    ).validate()
    pace = config.to_runtime_dict()["muscle_pace"]
    assert pace["adaptation_strategy"] == "mechanical_sensitivity"
    assert "mechanics-only" in pace["initial_weight_basis"]
    with pytest.raises(ValueError, match="paramétriques"):
        IndependentArmsGuiConfig(
            factory="example:build", muscle_weight_policy="mechanical_sensitivity_squared_v1_experimental",
        ).validate()


def test_solver_cpu_pair_is_exported_only_when_both_distinct_cpus_are_selected():
    config = IndependentArmsGuiConfig(factory="example:build", right_solver_cpu=30, left_solver_cpu=31).validate()
    assert config.to_runtime_dict()["solver_cpu_affinity"] == {"right": 30, "left": 31}
    with pytest.raises(ValueError, match="deux CPU"):
        IndependentArmsGuiConfig(factory="example:build", right_solver_cpu=30).validate()
    with pytest.raises(ValueError, match="distincts"):
        IndependentArmsGuiConfig(factory="example:build", right_solver_cpu=30, left_solver_cpu=30).validate()


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
