from dataclasses import replace
import json

import numpy as np
import pytest

from cocofest.cycling.slow.fast_state_probes import DecisionObservation, _inventory, build_fast_probe_plan, decision_pair_effect
from tests.test_slow_oracle import checkpoint


def observation(**kwargs):
    return replace(DecisionObservation("same-slow-action-contract", "history-a", 30, "frozen_local_endpoint"), **kwargs)


def test_resolved_effect_is_not_a_closure_certificate():
    result = decision_pair_effect(observation(), observation(fast_state_key="history-b", accepted_cycles=34),
                                  resolution_cycles=2)
    assert result["effect_over_resolution"] == 2
    assert result["classification"] == "resolved_effect"
    assert result["pair_kind"] == "fast_probe"
    assert not result["closure_established"] and not result["certified"]


def test_difference_equal_to_resolution_is_tie():
    result = decision_pair_effect(observation(), observation(accepted_cycles=32), resolution_cycles=2)
    assert result["classification"] == "within_observed_resolution"
    assert result["pair_kind"] == "repeat_control"


@pytest.mark.parametrize("endpoint", ["right_censored", "numerical_unresolved"])
def test_censored_lower_bounds_do_not_prove_fast_insensitivity(endpoint):
    result = decision_pair_effect(observation(), observation(endpoint=endpoint), resolution_cycles=0)
    assert result["classification"] == "inconclusive"
    assert result["cycle_difference"] is None and result["effect_over_resolution"] is None


def test_zero_control_spread_keeps_absolute_effect_without_infinite_ratio():
    result = decision_pair_effect(observation(), observation(accepted_cycles=31), resolution_cycles=0)
    assert result["classification"] == "resolved_effect"
    assert result["effect_over_resolution"] is None


@pytest.mark.parametrize("resolution", [-1, float("nan"), float("inf"), True])
def test_invalid_resolution_is_rejected(resolution):
    with pytest.raises(ValueError, match="resolution"):
        decision_pair_effect(observation(), observation(), resolution_cycles=resolution)


def test_nearby_slow_states_cannot_be_used_as_matched_fast_pair():
    with pytest.raises(ValueError, match="contract"):
        decision_pair_effect(observation(), observation(experiment_key="different-slow-state"), resolution_cycles=1)


def test_plan_pins_certified_sources_and_remains_unassessed(checkpoint):
    plan = build_fast_probe_plan([checkpoint], side="left", muscles=["Biceps"])
    assert plan["sources"][0]["slow_state"]["A_Biceps"] == 3000
    assert plan["sources"][0]["fast_inventory"]["PW_history"][0]["values"] == [[.0002]]
    assert plan["execution"]["online_activation"] is False
    assert plan["gate"]["status"] == "not_assessed"
    assert plan == build_fast_probe_plan([checkpoint], side="left", muscles=["Biceps"])


def test_plan_rejects_modified_source_and_duplicate_sources(checkpoint):
    with pytest.raises(ValueError, match="distinct"):
        build_fast_probe_plan([checkpoint, checkpoint], side="left", muscles=["Biceps"])
    document = json.loads(checkpoint.read_text())
    document["arms"]["left"]["sha256"] = "0" * 64
    checkpoint.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="digest"):
        build_fast_probe_plan([checkpoint], side="left", muscles=["Biceps"])


def test_inventory_preserves_mechanical_runtime_and_unrecognized_states(tmp_path):
    path = tmp_path / "inventory.npz"
    arrays = {f"problem__x_bounds:{state}:{bound}": np.array([[value, 999.]])
              for state, value in (("theta", 0.), ("omega", 6.28), ("E_prod", 0.),
                                   ("Cn_Biceps", .4), ("extra_memory", .5))
              for bound in ("min", "max")}
    np.savez(path, **arrays, states__Cn_Biceps=np.array([[100., 100.]]),
             metadata__json=np.asarray(json.dumps({"runtime_state": {"absolute_wheel_q_cycle_index": 10}})))
    inventory = _inventory(str(path))
    assert len(inventory["mechanics"]) == 3
    assert inventory["Cn"][0]["initial_lower"] == [.4]
    assert inventory["other_state"][0]["initial_lower"] == [.5]
    assert inventory["runtime"][0]["value"] == 10
