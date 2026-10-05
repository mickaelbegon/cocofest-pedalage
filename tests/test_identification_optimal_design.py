import numpy as np
import pytest

from cocofest.identification.optimal_design import (
    TrialInformation,
    fisher_information,
    information_diagnostics,
    scaled_sensitivities,
    select_trials,
)


def test_scaled_whitened_derivatives_match_linear_model_with_correlated_sensors():
    model = np.array([[2.0, 3.0], [1.0, -4.0], [0.5, 2.0]])
    scales = np.array([0.01, 100.0])
    covariance = np.array([[2.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 3.0]])
    jac = scaled_sensitivities(lambda p: model @ p, np.array([-0.1, 20.0]), scales, covariance=covariance, step=1e-3)
    expected = np.diag(scales) @ model.T @ np.linalg.solve(covariance, model) @ np.diag(scales)
    np.testing.assert_allclose(fisher_information(jac), expected, rtol=1e-8, atol=1e-8)


def test_unknown_offset_removes_confounding_not_identifiability_failure():
    jac = np.column_stack((np.ones(10), np.arange(10)))
    info = fisher_information(jac, np.ones((10, 1)))
    report = information_diagnostics(info)
    assert report["rank"] == 1
    assert report["scaled_crlb_std"] is None
    assert abs(report["weakest_direction"][0]) == pytest.approx(1)


def test_complementary_excitation_is_chosen_over_redundant_high_amplitude():
    candidates = [
        TrialInformation("force_gain", np.diag([100.0, 0.0]), 1),
        TrialInformation("same_gain", np.diag([90.0, 0.0]), 1),
        TrialInformation("frequency_transient", np.diag([0.0, 2.0]), 1),
    ]
    report = select_trials(candidates, 2)
    assert [item["name"] for item in report["selected"]] == ["force_gain", "frequency_transient"]
    assert report["data_diagnostics"][0]["full_rank"]


def test_prior_never_masquerades_as_data_identifiability():
    report = select_trials([TrialInformation("gain", np.diag([1.0, 0.0]), 1)], 1, prior_precision=np.eye(2))
    assert report["data_diagnostics"][0]["rank"] == 1
    assert report["uses_prior"]


def test_budget_and_admissibility_are_hard_constraints():
    candidates = [
        TrialInformation("inadmissible", np.eye(2) * 10000, 1, admissible=False),
        TrialInformation("too_fatiguing", np.eye(2) * 1000, 1, fatigue_cost=2),
        TrialInformation("allowed", np.eye(2), 2, fatigue_cost=0.3, charge_cost=0.4),
        TrialInformation("too_long", np.eye(2) * 10, 20),
    ]
    report = select_trials(candidates, 4, duration_budget_s=3, fatigue_budget=1, charge_budget=1)
    assert [item["name"] for item in report["selected"]] == ["allowed"]
    assert report["spent"] == {"duration_s": 2, "fatigue_cost": 0.3, "charge_cost": 0.4}


def test_ensemble_design_avoids_nominal_only_best_choice():
    candidates = [
        TrialInformation("nominal", np.array([np.eye(2) * 100, np.zeros((2, 2))]), 1),
        TrialInformation("robust", np.array([np.eye(2) * 2, np.eye(2) * 2]), 1),
    ]
    report = select_trials(candidates, 1)
    assert report["selected"][0]["name"] == "robust"
    assert len(report["data_diagnostics"]) == 2


def test_information_rank_is_invariant_to_overall_units():
    for scale in (1e-12, 1, 1e12):
        assert information_diagnostics(np.diag([2.0, 1.0]) * scale)["rank"] == 2


@pytest.mark.parametrize("information", [np.array([[1, 2], [0, 1]]), np.diag([1, -1]), np.array([[np.nan]])])
def test_invalid_information_rejected(information):
    with pytest.raises(ValueError):
        information_diagnostics(information)


def test_zero_noise_rejected():
    with pytest.raises(ValueError):
        scaled_sensitivities(lambda p: p, np.ones(1), np.ones(1), noise_std=0)
