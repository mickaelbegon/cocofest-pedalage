import math

import pytest

from cocofest.simulation.bilateral_endurance_campaign import (
    BilateralEnduranceCampaignConfig,
    BilateralEnduranceCandidate,
    BilateralEnduranceObservation,
    BilateralTorqueBracketConfig,
    assess_summary,
)
from cocofest.simulation.bilateral_endurance_runner import run_torque_bracket


def candidate():
    return BilateralEnduranceCandidate(.4, .6,
        {"r0": 1., "r1": 2., "r2": .5, "r3": 1.},
        {"l0": 1., "l1": .5, "l2": 2., "l3": 1.})


def config(tmp_path):
    return BilateralEnduranceCampaignConfig(
        base_payload={"factory": "cocofest.simulation.independent_arms_process:build_process_independent_arms",
                      "solver": "ipopt", "formulation": "isokinetic", "cycles_per_window": 1,
                      "parallel": True, "omega_rad_s": -2 * math.pi}, output_root=str(tmp_path)).validate()


def test_candidate_payload_preserves_constant_total_work_and_normalizes_weights(tmp_path):
    payload = config(tmp_path).payload_for(candidate(), 120)
    pace = payload["resistance_pace"]
    assert payload["right_equivalent_mean_torque_nm"] + payload["left_equivalent_mean_torque_nm"] == pytest.approx(.4)
    assert pace["total_equivalent_mean_torque_nm"] == pytest.approx(.4)
    for side in ("right", "left"):
        weights = payload["muscle_pace"][f"{side}_initial_weights"]
        assert math.prod(weights.values()) == pytest.approx(1.)
    assert payload["cycles"] == 120


def test_candidate_payload_keeps_base_pace_safety_limits(tmp_path):
    protocol = config(tmp_path)
    protocol = BilateralEnduranceCampaignConfig(
        **{**protocol.to_dict(), "base_payload": {**protocol.base_payload,
            "muscle_pace": {"max_relative_weight": 2.},
            "resistance_pace": {"minimum_arm_equivalent_mean_torque_nm": .02}}}).validate()
    payload = protocol.payload_for(candidate(), 30)
    assert payload["muscle_pace"]["max_relative_weight"] == 2.
    assert payload["resistance_pace"]["minimum_arm_equivalent_mean_torque_nm"] == .02


def test_candidate_payload_can_hold_a_strict_half_half_work_split(tmp_path):
    protocol = BilateralEnduranceCampaignConfig(
        **{**config(tmp_path).to_dict(), "resistance_capacity_feedback": False}).validate()
    half_half = BilateralEnduranceCandidate(.4, .5, candidate().right_weights, candidate().left_weights)
    payload = protocol.payload_for(half_half, 30)
    assert payload["resistance_pace"]["capacity_feedback"] is False
    assert payload["resistance_pace"]["initial_right_fraction"] == pytest.approx(.5)
    assert payload["right_equivalent_mean_torque_nm"] == pytest.approx(.2)
    assert payload["left_equivalent_mean_torque_nm"] == pytest.approx(.2)


def test_campaign_rejects_non_certification_protocol(tmp_path):
    with pytest.raises(ValueError, match="solver"):
        BilateralEnduranceCampaignConfig(
            base_payload={"factory": "x:y", "solver": "acados", "formulation": "isokinetic",
                          "cycles_per_window": 1, "parallel": True, "omega_rad_s": -1},
            output_root=str(tmp_path)).validate()
    with pytest.raises(ValueError, match="end at target"):
        BilateralEnduranceCampaignConfig(
            base_payload={"factory": "x:y", "solver": "ipopt", "formulation": "isokinetic",
                          "cycles_per_window": 1, "parallel": True, "omega_rad_s": -1},
            output_root=str(tmp_path), fidelities=(30, 120)).validate()


def test_summary_classification_keeps_technical_failure_distinct_from_infeasibility():
    good = {"success": True, "arms": {"right": {"validated_cycles": 30}, "left": {"validated_cycles": 30}}}
    assert assess_summary(good, 30).status == "feasible"
    stopped = {"success": False, "failure": "At least one arm failed certification after cycle 8",
               "arms": {"right": {"validated_cycles": 8}, "left": {"validated_cycles": 8}}}
    assert assess_summary(stopped, 30).status == "technical_failure"
    physiological = {**stopped, "failure_kind": "fatigue_limit"}
    assert assess_summary(physiological, 30).status == "infeasible"
    assert assess_summary({}, 30).status == "technical_failure"


def test_summary_labels_timeout_primal_infeasibility_as_numerical_and_uses_common_prefix_reserve():
    failed = {"success": False, "failure": "At least one arm failed certification after cycle 3", "arms": {
        "right": {"validated_cycles": 2, "cycles": [
            {"cycle": 1, "certified": True, "minimum_capacity_ratio": .7},
            {"cycle": 2, "certified": True, "minimum_capacity_ratio": .6},
            {"cycle": 3, "certified": True, "minimum_capacity_ratio": .1}]},
        "left": {"validated_cycles": 2, "cycles": [
            {"cycle": 1, "certified": True, "minimum_capacity_ratio": .8},
            {"cycle": 2, "certified": True, "minimum_capacity_ratio": .65},
            {"cycle": 3, "certified": False, "minimum_capacity_ratio": .2, "solver_time_s": 60.,
             "feasibility": {"failure_reason": "primal_infeasibility_above_threshold"}}]}}}
    observation = assess_summary(failed, 600)
    assert observation.status == "numerical_failure"
    assert observation.validated_cycles == 2
    assert observation.minimum_capacity_ratio == pytest.approx(.6)


def test_summary_classifies_certified_fatigue_counterfactual_as_physiological():
    summary = {"success": False, "failure": "uncertified", "arms": {
        "right": {"validated_cycles": 5, "cycles": [{"cycle": 6, "certified": True}]},
        "left": {"validated_cycles": 5, "cycles": [{
            "cycle": 6, "certified": False,
            "counterfactual_probes": {
                "work_relief": [{"feasible_witness": True}],
                "pw_upper_relief": {"feasible_witness": True},
                "fatigue_rest": {"feasible_witness": True},
            },
        }]},
    }}
    observation = assess_summary(summary, 600)
    assert observation.status == "infeasible"
    assert observation.reason == "fatigue_counterfactual_certified_with_pw_saturation"


def test_summary_keeps_work_relief_without_physiological_counterfactual_unclassified():
    summary = {"success": False, "failure": "uncertified", "arms": {
        "right": {"validated_cycles": 5, "cycles": []},
        "left": {"validated_cycles": 5, "cycles": [{
            "cycle": 6, "certified": False,
            "counterfactual_probes": {"work_relief": [{"feasible_witness": True}]},
        }]},
    }}
    observation = assess_summary(summary, 600)
    assert observation.status == "technical_failure"
    assert observation.reason == "near_load_boundary_work_relief_certified"


def test_torque_bracket_converges_only_from_explicitly_classified_failures(tmp_path):
    def evaluate(_campaign, point):
        status = "feasible" if point.total_equivalent_mean_torque_nm <= .35 else "infeasible"
        return [BilateralEnduranceObservation(status, 600, 600 if status == "feasible" else 50, .8)]

    result = run_torque_bracket(config(tmp_path), candidate(),
        BilateralTorqueBracketConfig(.3, .4, .0125, 8), evaluate=evaluate)
    assert result["status"] == "converged"
    assert .3375 <= result["certified_lower_torque_nm"] <= .35
    assert result["infeasible_upper_torque_nm"] > .35


def test_torque_bracket_refuses_to_call_technical_failure_a_load_limit(tmp_path):
    def evaluate(_campaign, _point):
        return [BilateralEnduranceObservation("technical_failure", 30, 0, None, "ipopt")]

    result = run_torque_bracket(config(tmp_path), candidate(),
        BilateralTorqueBracketConfig(.3, .4, .01, 4), evaluate=evaluate)
    assert result["status"] == "invalid_lower_bound"
