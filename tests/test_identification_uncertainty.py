"""Whole-acquisition bootstrap guards and cross-stage uncertainty propagation."""

import json

import numpy as np
import pytest

from cocofest.identification.sequential import ParameterSpec, TrialObservation
from cocofest.identification.uncertainty import bootstrap_force_then_fatigue, fatigue_design_ensemble


def predictor(parameters, context):
    # Gain and decay are coupled in fatigue observations; force identifies gain.
    time, fatigued = context
    return parameters["gain"] * np.exp(-parameters["decay"] * time * fatigued)


def data():
    force, fatigue = [], []
    for index, gain in enumerate((8., 9., 11., 12.)):
        context = (np.linspace(0, 1, 5), False)
        force.append(TrialObservation(f"force{index}", predictor({"gain": gain, "decay": .2}, context), 1., context))
        context = (np.linspace(1, 5, 8), True)
        fatigue.append(TrialObservation(f"fatigue{index}", predictor({"gain": 10., "decay": .2}, context), 1., context))
    return force, fatigue, [ParameterSpec("gain", 10., 1., 20., 10.)], [ParameterSpec("decay", .1, .001, 1., .1)]


def test_force_variation_propagates_to_fatigue_even_with_identical_fatigue_trials():
    report = bootstrap_force_then_fatigue(predictor, *data(), n_resamples=20, seed=42)
    assert report["successful_resamples"] == 20
    draws = report["replicates"]
    gains = [draw["parameters"]["gain"] for draw in draws]
    decays = [draw["parameters"]["decay"] for draw in draws]
    assert np.ptp(gains) > 1.
    assert np.ptp(decays) > .02
    assert np.corrcoef(gains, decays)[0, 1] > .99
    assert report["intervals"]["decay"]["lower"] < .2 < report["intervals"]["decay"]["upper"]
    assert all(draw["force_parameters"]["gain"] == draw["parameters"]["gain"] for draw in draws)
    json.dumps(report, allow_nan=False)


def test_resampling_preserves_complete_context_and_stratum_allocation():
    force, fatigue, force_specs, fatigue_specs = data()
    contexts = {id(trial.context): trial.observed.shape for trial in (*force, *fatigue)}

    def checked_predictor(parameters, context):
        assert id(context) in contexts
        prediction = predictor(parameters, context)
        assert prediction.shape == contexts[id(context)]
        return prediction

    labels = {trial.name: "low" if i < 2 else "high" for i, trial in enumerate(force)}
    report = bootstrap_force_then_fatigue(
        checked_predictor, force, fatigue, force_specs, fatigue_specs,
        force_strata=labels, n_resamples=4, seed=4,
    )
    for draw in report["replicates"]:
        assert sum(labels[name] == "low" for name in draw["force_sources"]) == 2
        assert sum(labels[name] == "high" for name in draw["force_sources"]) == 2
    repeated = bootstrap_force_then_fatigue(
        checked_predictor, force, fatigue, force_specs, fatigue_specs,
        force_strata=labels, n_resamples=4, seed=4,
    )
    assert report == repeated


def test_fatigue_design_preserves_force_draws_and_rejects_overwriting():
    report = bootstrap_force_then_fatigue(predictor, *data(), n_resamples=4)
    ensemble = fatigue_design_ensemble(report, [{"decay": .1}, {"decay": .3}])
    assert len(ensemble) == 8
    assert [p["gain"] for p in ensemble[:4]] == [r["parameters"]["gain"] for r in report["replicates"]]
    assert [p["gain"] for p in ensemble[4:]] == [p["gain"] for p in ensemble[:4]]
    with pytest.raises(ValueError, match="exactly"):
        fatigue_design_ensemble(report, [{"decay": .2, "gain": 50.}])
    with pytest.raises(ValueError, match="sufficient"):
        fatigue_design_ensemble({**report, "adequate_success_fraction": False}, [{"decay": .2}])


def test_single_acquisition_or_singleton_strata_cannot_fake_uncertainty():
    force, fatigue, fs, ds = data()
    with pytest.raises(ValueError, match="two"):
        bootstrap_force_then_fatigue(predictor, force[:1], fatigue, fs, ds)
    with pytest.raises(ValueError, match="replicated"):
        bootstrap_force_then_fatigue(predictor, force, fatigue, fs, ds,
                                    force_strata={trial.name: trial.name for trial in force})
    with pytest.raises(ValueError, match="every"):
        bootstrap_force_then_fatigue(predictor, force, fatigue, fs, ds, force_strata={"force0": "a"})


def test_failed_refits_are_reported_and_intervals_withheld(monkeypatch):
    import cocofest.identification.uncertainty as module

    original = module.fit_force_then_fatigue
    calls = 0

    def sometimes_failed(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 2:
            raise RuntimeError("synthetic refit failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "fit_force_then_fatigue", sometimes_failed)
    report = module.bootstrap_force_then_fatigue(predictor, *data(), n_resamples=4)
    assert report["successful_resamples"] == 1
    assert len(report["failed_replicates"]) == 3
    assert report["intervals"] is None
    assert not report["adequate_success_fraction"]


@pytest.mark.parametrize("kwargs", [{"n_resamples": 1}, {"confidence_level": 1}, {"minimum_success_fraction": 0}])
def test_invalid_bootstrap_configuration(kwargs):
    with pytest.raises(ValueError):
        bootstrap_force_then_fatigue(predictor, *data(), **kwargs)
