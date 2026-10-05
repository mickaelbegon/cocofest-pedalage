"""Scientific guards for the dependency-light force/fatigue estimator."""

import json

import numpy as np
import pytest

from cocofest.identification.sequential import (
    ParameterSpec, TrialObservation, fisher_diagnostics, fit_force_then_fatigue, fit_parameters,
)


def exponential_predictor(parameters, context):
    time, fatigued = context
    return parameters["gain"] * (1 - np.exp(-time / parameters["tau"])) * (
        np.exp(-parameters["decay"] * time) if fatigued else 1.
    )


def test_staged_fit_recovers_parameters_and_freezes_fresh_force():
    true = {"gain": 110., "tau": .065, "decay": .013}
    initial = {"gain": 80., "tau": .1, "decay": .02}
    def trial(name, time, fatigued):
        context = (time, fatigued)
        return TrialObservation(name, exponential_predictor(true, context), .5, context)
    force = trial("force", np.linspace(.005, .6, 70), False)
    fatigue = trial("fatigue", np.linspace(.005, 60., 100), True)
    force_holdout = trial("force_holdout", np.linspace(.01, .9, 90), False)
    fatigue_holdout = trial("fatigue_holdout", np.linspace(.01, 90., 80), True)
    result = fit_force_then_fatigue(
        exponential_predictor, [force], [fatigue],
        [ParameterSpec("gain", initial["gain"], 20., 200., 100.),
         ParameterSpec("tau", initial["tau"], .01, .3, .1)],
        [ParameterSpec("decay", initial["decay"], .001, .1, .01)],
        force_holdout=[force_holdout], fatigue_holdout=[fatigue_holdout],
    )
    assert result["force"].converged and result["fatigue"].converged
    for name, value in true.items():
        assert result["fatigue"].parameters[name] == pytest.approx(value, rel=1e-5)
    assert result["fatigue"].parameters["gain"] == result["force"].parameters["gain"]
    assert result["fatigue"].parameters["tau"] == result["force"].parameters["tau"]
    assert result["force"].fisher["rank"] == 2
    assert result["fatigue"].fisher["rank"] == 1
    assert result["fatigue"].holdout_metrics["fatigue_holdout"]["noise_normalized_rmse"] < 1e-6
    json.dumps(result["fatigue"].to_dict(), allow_nan=False)


def test_prior_does_not_hide_unidentifiable_data():
    def predictor(parameters, context):
        return np.full(20, parameters["gain"] * parameters["mechanical_gain"])
    specs = [ParameterSpec(name, 1., .1, 5., 1., prior_mean=1., prior_std=.01)
             for name in ("gain", "mechanical_gain")]
    result = fit_parameters(predictor, [TrialObservation("confounded", np.ones(20), 1.)], specs)
    assert result.converged
    assert result.fisher["rank"] == 1
    assert not result.fisher["full_rank"]
    assert result.fisher["scaled_covariance_if_interior"] is None
    assert result.fisher["fisher_condition_number"] is None


def test_holdout_never_affects_fit_and_reveals_bias():
    predictor = lambda parameters, context: np.full(10, parameters["gain"])
    specs = [ParameterSpec("gain", 1., 0., 20., 1.)]
    training = [TrialObservation("train", np.full(10, 2.), 1.)]
    holdout = [TrialObservation("holdout", np.full(10, 15.), 1.)]
    result = fit_parameters(predictor, training, specs, holdout=holdout)
    assert result.parameters["gain"] == pytest.approx(2.)
    assert result.holdout_metrics["holdout"]["rmse_by_channel"] == pytest.approx([13.])
    with pytest.raises(ValueError, match="unique and disjoint"):
        fit_parameters(predictor, training, specs, holdout=training)


def test_two_sensor_channels_are_whitened_by_noise():
    predictor = lambda parameters, context: np.full((5, 2), parameters["gain"])
    trial = TrialObservation("two_dof", np.tile([1., 3.], (5, 1)), [1., 2.])
    result = fit_parameters(predictor, [trial], [ParameterSpec("gain", 0., -10., 10., 1.)])
    # Independent Gaussian channels give inverse-variance weighting.
    assert result.parameters["gain"] == pytest.approx((1. + 3. / 4.) / 1.25)
    assert result.fisher["n_observations"] == 10


def test_underdetermined_fisher_retains_null_directions():
    result = fisher_diagnostics(np.array([[1., 2., 3.]]))
    assert result["rank"] == 1
    assert len(result["singular_values"]) == 3
    assert result["scaled_covariance_if_interior"] is None
    assert np.dot([1., 2., 3.], result["weakest_scaled_parameter_direction"]) == pytest.approx(0., abs=1e-12)


def test_bound_solution_has_no_unconstrained_covariance():
    result = fit_parameters(lambda parameters, context: np.full(5, parameters["gain"]),
                            [TrialObservation("beyond_bound", np.full(5, 10.), 1.)],
                            [ParameterSpec("gain", 1., 0., 2., 1.)])
    assert result.parameters["gain"] == pytest.approx(2.)
    assert result.fisher["full_rank"]
    assert result.fisher["at_parameter_bound"]
    assert result.fisher["scaled_covariance_if_interior"] is None


def test_nonconverged_force_stage_stops_fatigue():
    with pytest.raises(RuntimeError, match="fatigue fit not started"):
        fit_force_then_fatigue(
            exponential_predictor,
            [TrialObservation("force", np.ones(10), 1., (np.linspace(.01, 1., 10), False))],
            [TrialObservation("fatigue", np.ones(10), 1., (np.linspace(.01, 1., 10), True))],
            [ParameterSpec("gain", 80., 20., 200., 100.), ParameterSpec("tau", .1, .01, .3, .1)],
            [ParameterSpec("decay", .02, .001, .1, .01)], max_nfev=1,
        )


@pytest.mark.parametrize("std", [0., -1., np.nan])
def test_invalid_noise_is_rejected(std):
    with pytest.raises(ValueError):
        TrialObservation("bad", np.ones(3), std)


def test_fixed_parameters_cannot_silently_be_overwritten():
    with pytest.raises(ValueError, match="simultaneously fitted and fixed"):
        fit_parameters(lambda params, context: np.ones(2), [TrialObservation("x", np.ones(2), 1.)],
                       [ParameterSpec("gain", 1., 0., 2., 1.)], fixed_parameters={"gain": 1.})


def test_prediction_shape_and_finiteness_are_enforced():
    for prediction in (np.ones(4), np.array([1., np.nan])):
        with pytest.raises(ValueError, match="prediction"):
            fit_parameters(lambda params, context: prediction, [TrialObservation("x", np.ones(2), 1.)],
                           [ParameterSpec("gain", 1., 0., 2., 1.)])
