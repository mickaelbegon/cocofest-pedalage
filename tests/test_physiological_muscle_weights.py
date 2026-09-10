import math

import numpy as np
import pytest

from cocofest.optimization.physiological_muscle_weights import (
    build_pre_risk_mask, calculate_physiological_muscle_weights, simulate_fatigue_ratios,
)


def case():
    names = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
    params = {name: {"Fmax": 10. + i, "a_scale": 100. + 3 * i,
                     "alpha_a": -.01 - i * .002, "tau_fat": 50. + 5 * i}
              for i, name in enumerate(names)}
    theta = np.linspace(0., 2 * np.pi, 32, endpoint=False)
    profiles = np.full((4, 32), -.1)
    for i in range(4):
        profiles[i, i*8:i*8+5] = .4
    return theta, profiles, params, names


def direct_source_loop(theta, profiles, params, names, cycles, rho=.8, width=90., threshold=.2):
    """Independent literal scalar recurrence and angular loops from source."""
    ratios = np.zeros((len(names), cycles))
    for muscle, name in enumerate(names):
        p = params[name]
        duty = np.mean(profiles[muscle] > 1e-9)
        ea, er = math.exp(-duty / p["tau_fat"]), math.exp(-(1. - duty) / p["tau_fat"])
        beta = p["alpha_a"] * rho * p["Fmax"] * p["tau_fat"]
        a = p["a_scale"]
        for cycle in range(cycles):
            active = (p["a_scale"] + beta) * (1 - ea) + ea * a
            a = p["a_scale"] * (1 - er) + er * active
            ratios[muscle, cycle] = a / p["a_scale"]
    support, unique = np.zeros((cycles, len(names))), np.zeros((cycles, len(names)))
    risk_masks, pre_masks = [], []
    for cycle in range(cycles):
        positive = np.maximum(profiles * ratios[:, cycle, None], 0.)
        redundancy = (positive > 1e-9).sum(axis=0)
        risk = positive.sum(axis=0) < threshold
        onsets = risk & ~np.roll(risk, 1)
        pre = np.zeros_like(risk)
        for onset in np.degrees(theta)[onsets]:
            distance = (onset - np.degrees(theta)) % 360.
            pre |= (distance > 0.) & (distance <= width)
        pre &= ~risk
        risk_masks.append(risk)
        pre_masks.append(pre)
        for muscle in range(len(names)):
            support[cycle, muscle] = np.trapezoid(positive[muscle] * pre, theta)
            unique[cycle, muscle] = np.trapezoid(np.where(redundancy == 1, positive[muscle], 0.), theta)
    fatigue = (ratios[:, 0] - ratios[:, -1]) / cycles
    raw = (support.mean(axis=0) + unique.mean(axis=0)) * fatigue**2
    normalized = (raw - raw.min()) / (raw.max() - raw.min()) if raw.max() != raw.min() else np.zeros_like(raw)
    return ratios, support, unique, np.array(risk_masks), np.array(pre_masks), raw, normalized


def test_1500_cycle_calculation_matches_independent_published_source_loop():
    theta, profiles, params, names = case()
    actual = calculate_physiological_muscle_weights(theta, profiles, params,
                                                   muscle_names=names, case_id="reference")
    ratios, support, unique, risk, pre, raw, normalized = direct_source_loop(theta, profiles, params, names, 1500)
    assert actual["status"] == "ok" and actual["usable_for_controller"]
    np.testing.assert_allclose(actual["ratios"], ratios, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual["support_in_pre_risk_by_cycle"], support, atol=1e-12)
    np.testing.assert_allclose(actual["unique_support_by_cycle"], unique, atol=1e-12)
    np.testing.assert_array_equal(actual["risk_mask"], risk)
    np.testing.assert_array_equal(actual["pre_risk_mask"], pre)
    np.testing.assert_allclose(list(actual["raw_weights"].values()), raw, rtol=1e-10)
    np.testing.assert_allclose(list(actual["legacy_normalized_weights"].values()), normalized, atol=1e-10)
    np.testing.assert_allclose(list(actual["normalized_weights"].values()), raw / raw.max(), atol=1e-10)
    assert actual["context"]["cycle_duration_seconds"] == 1.
    assert not actual["context"]["published_normalization"]


def test_max_normalization_preserves_all_strictly_positive_raw_weights():
    theta, profiles, params, names = case()
    result = calculate_physiological_muscle_weights(theta, profiles, params, muscle_names=names, case_id="positive")
    assert result["min_raw_weight"] > 0.
    assert min(result["normalized_weights"].values()) > 0.
    variant = calculate_physiological_muscle_weights(theta, profiles, params, muscle_names=names,
                                                    case_id="positive", normalization="minmax")
    assert min(variant["normalized_weights"].values()) == 0.
    assert variant["context"]["published_normalization"]
    assert variant["legacy_normalized_weights"] == result["legacy_normalized_weights"]


def test_duplicate_nonuniform_angular_samples_use_source_trapezoid_without_closure():
    theta, profiles, params, names = case()
    indices = [0, 0, 2, 4, 7, 8, 9, 13, 16, 20, 24, 28, 30]
    theta, profiles = theta[indices], profiles[:, indices]
    result = calculate_physiological_muscle_weights(theta, profiles, params, muscle_names=names,
                                                   case_id="nonuniform", target_cycles=27)
    expected = direct_source_loop(theta, profiles, params, names, 27)
    np.testing.assert_allclose(result["support_in_pre_risk_by_cycle"], expected[1], atol=1e-12)
    np.testing.assert_allclose(result["unique_support_by_cycle"], expected[2], atol=1e-12)
    np.testing.assert_array_equal(result["pre_risk_mask"], expected[4])
    np.testing.assert_allclose(list(result["raw_weights"].values()), expected[5], rtol=1e-10)


def test_triceps_positive_raw_score_is_not_zeroed_by_normalization():
    theta, profiles, params, names = case()
    params["Triceps"]["alpha_a"] = -.001
    initial = calculate_physiological_muscle_weights(theta, profiles, params, muscle_names=names, case_id="weak_fatigue")
    assert initial["raw_weights"]["Triceps"] > 0.
    assert initial["normalized_weights"]["Triceps"] > 0.
    params["Triceps"]["alpha_a"] = -.05
    changed = calculate_physiological_muscle_weights(theta, profiles, params, muscle_names=names, case_id="strong_fatigue")
    assert changed["normalized_weights"]["Triceps"] > 0.
    assert min(changed["normalized_weights"].values()) > 0.


def test_muscle_permutation_keeps_named_results_and_explicit_case_parameters():
    theta, profiles, params, names = case()
    original = calculate_physiological_muscle_weights(theta, profiles, params, muscle_names=names, case_id="one")
    permutation = [3, 1, 0, 2]
    moved = tuple(names[i] for i in permutation)
    permuted = calculate_physiological_muscle_weights(theta, profiles[permutation], params,
                                                     muscle_names=moved, case_id="one")
    for key in ("normalized_weights", "raw_weights", "mechanical_contribution", "fatigability", "duty_cycles"):
        for name in names:
            assert permuted[key][name] == pytest.approx(original[key][name])
    altered = {name: dict(values) for name, values in params.items()}
    altered["Triceps"]["alpha_a"] *= 2.
    second = calculate_physiological_muscle_weights(theta, profiles, altered, muscle_names=names, case_id="two")
    assert second["context"]["case_id"] == "two"
    assert second["fatigability"]["Triceps"] > original["fatigability"]["Triceps"]
    assert second["normalized_weights"] != original["normalized_weights"]
    assert original["context"]["parameters"]["Triceps"]["alpha_a"] == params["Triceps"]["alpha_a"]


def test_fatigue_duty_zero_and_one_and_force_capacity_joint_scaling():
    _, _, params, names = case()
    zero = simulate_fatigue_ratios(np.zeros(4), params, muscle_names=names)
    np.testing.assert_array_equal(zero["ratios"], np.ones((4, 1500)))
    np.testing.assert_array_equal(zero["active_ratios"], np.ones((4, 1500)))
    full = simulate_fatigue_ratios(np.ones(4), params, muscle_names=names)
    np.testing.assert_allclose(full["ratios"], full["active_ratios"], atol=1e-15)
    for i, name in enumerate(names):
        p = params[name]
        expected = 1 + p["alpha_a"] * .8 * p["Fmax"] * p["tau_fat"] / p["a_scale"] * (
            1 - np.exp(-np.arange(1, 1501) / p["tau_fat"]))
        np.testing.assert_allclose(full["ratios"][i], expected, atol=1e-15)
    scaled = {name: {**p, "Fmax": p["Fmax"] * 3, "a_scale": p["a_scale"] * 3} for name, p in params.items()}
    invariant = simulate_fatigue_ratios(np.ones(4), scaled, muscle_names=names)
    np.testing.assert_allclose(invariant["ratios"], full["ratios"], atol=1e-15)


def test_wrapped_pre_risk_onset_excludes_risk_and_all_risk_has_no_onset():
    degrees = np.arange(0., 360., 45.)
    risk = np.array([True, False, False, False, False, False, False, False])
    np.testing.assert_array_equal(build_pre_risk_mask(degrees, risk),
                                   [False, False, False, False, False, False, True, True])
    assert not np.any(build_pre_risk_mask(degrees, np.ones(8, bool)))
    assert not np.any(build_pre_risk_mask(degrees, np.zeros(8, bool)))


def test_invalid_active_capacity_even_after_positive_recovery_is_not_controller_usable():
    theta = np.linspace(0., 2*np.pi, 10, endpoint=False)
    profiles = np.full((1, 10), -.1)
    profiles[0, 0] = .4  # 0.1 s active, then 0.9 s recovery.
    params = {"m": {"Fmax": 1., "a_scale": 1., "alpha_a": -20., "tau_fat": .1}}
    result = calculate_physiological_muscle_weights(theta, profiles, params, muscle_names=["m"],
                                                   case_id="active_failure", rho=1.)
    assert result["status"] == "invalid_fatigue_domain"
    assert result["first_invalid"]["phase"] == "active"
    assert result["active_ratios"][0, 0] < 0.
    assert result["ratios"][0, 0] > 0.
    assert result["normalized_weights"] is None
    assert not result["usable_for_controller"]
    assert "legacy_normalized_weights" in result


def test_nonpositive_post_rest_ratio_retained_for_legacy_audit_not_clipped():
    theta, profiles, params, names = case()
    params[names[0]]["alpha_a"] = -100.
    result = calculate_physiological_muscle_weights(theta, profiles, params, muscle_names=names, case_id="bad")
    assert result["status"] == "invalid_fatigue_domain"
    assert result["ratios"].min() < 0.
    assert result["normalized_weights"] is None
    assert not result["usable_for_controller"]
    expected = direct_source_loop(theta, profiles, params, names, 1500)
    np.testing.assert_allclose(list(result["legacy_normalized_weights"].values()), expected[-1], atol=1e-10)


def test_no_fatigue_gives_degenerate_zero_weights_without_controller_eligibility():
    theta, profiles, params, names = case()
    result = calculate_physiological_muscle_weights(theta, profiles, params, muscle_names=names,
                                                   case_id="no_force", rho=0.)
    assert result["domain_valid"]
    assert result["status"] == "degenerate_weights"
    assert set(result["legacy_normalized_weights"].values()) == {0.}
    assert result["normalized_weights"] is None


@pytest.mark.parametrize("key,value", [("Fmax", 0.), ("a_scale", -1.), ("alpha_a", .1), ("tau_fat", np.nan)])
def test_invalid_muscle_parameters_are_refused(key, value):
    theta, profiles, params, names = case()
    params[names[0]][key] = value
    with pytest.raises(ValueError):
        calculate_physiological_muscle_weights(theta, profiles, params, muscle_names=names, case_id="bad")


@pytest.mark.parametrize("options", [{"rho": -1.}, {"rho": 1.1}, {"target_cycles": 0},
                                      {"target_cycles": 1.5}, {"target_cycles": True},
                                      {"pre_risk_width_deg": -1.}, {"normalization": "unknown"}])
def test_invalid_calculation_options_are_refused(options):
    theta, profiles, params, names = case()
    with pytest.raises(ValueError):
        calculate_physiological_muscle_weights(theta, profiles, params, muscle_names=names, case_id="bad", **options)
