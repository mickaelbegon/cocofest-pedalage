import pytest

from scripts.slow_layer.run_causal_torque_pilot import action_schedule, aggregate_arms, authenticated


def test_ten_cycle_action_returns_to_explicit_common_policy():
    for delta in (-.05, 0., .05):
        action = action_schedule(delta, "intervention")
        common = action_schedule(delta, "continuation")
        assert sum(action.values()) == pytest.approx(1.92)
        assert common == {"left": .96, "right": .96}
        assert abs(action["left"] - action["right"]) <= .3
        assert all(abs(action[s] - common[s]) <= .050000001 for s in action)
    with pytest.raises(ValueError):
        action_schedule(.1, "intervention")


def test_numerical_limiting_arm_prevents_later_physiological_label():
    arms = {"left": {"status": "numerical_stop", "last_certified_cycle": 150},
            "right": {"status": "physiological_failure_certified", "last_certified_cycle": 160}}
    assert aggregate_arms(arms)["status"] == "numerical_stop"
    arms["left"]["last_certified_cycle"] = 170
    result = aggregate_arms(arms)
    assert result["status"] == "physiological_failure_certified"
    assert result["last_certified_cycle"] == 160


def test_common_budget_is_censored_not_exact_label():
    result = aggregate_arms({s: {"status": "right_censored", "last_certified_cycle": 240} for s in ("left", "right")})
    assert result["status"] == "right_censored"
    assert not result["exact_label_allowed"]
    assert aggregate_arms({"left": {"status": "numerical_stop"}})["status"] == "numerical_stop"


def test_changed_manifest_fails_before_runtime_import(tmp_path):
    (tmp_path / "manifest.json").write_text('{}')
    (tmp_path / "manifest-seal.json").write_text('{"manifest_sha256": "wrong"}')
    with pytest.raises(ValueError, match="Manifest changed"):
        authenticated(tmp_path / "manifest.json")
