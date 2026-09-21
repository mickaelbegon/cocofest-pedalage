"""Warmup activity bounds can use the supplied PW bridge without a pickle seed."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.models.ding2007.ding2007 import DingModelPulseWidthFrequency
from examples.fes_multibody.cycling.cycling_pulse_width_mhe import set_u_bounds_and_init


KEY = "last_pulse_width_Biceps"


@pytest.fixture
def model():
    return SimpleNamespace(
        muscles_dynamics_model=[DingModelPulseWidthFrequency(muscle_name="Biceps")]
    )


def test_warmup_reference_initializes_and_projects_without_file(model):
    pd0 = model.muscles_dynamics_model[0].pd0
    reference = np.array([[0.0, pd0 + 1e-6, 0.0002, 0.0008]])
    original = reference.copy()
    with pytest.warns(RuntimeWarning, match="violates the physical Ding bounds"):
        bounds, guesses, _ = set_u_bounds_and_init(
            model, 4, None, active_set_mode="warmup", active_margin=0,
            active_reference={KEY: reference},
        )
    np.testing.assert_allclose(guesses[KEY].init, [[pd0, pd0, 0.0002, 0.0006]])
    np.testing.assert_allclose(bounds[KEY].min, [[pd0] * 4])
    np.testing.assert_allclose(bounds[KEY].max, [[pd0, pd0, 0.0006, 0.0006]])
    np.testing.assert_array_equal(reference, original)


@pytest.mark.parametrize(
    "reference,error,message",
    [
        (None, ValueError, "requires warmup controls"),
        ({}, KeyError, "missing from warmup active-set reference"),
        ({KEY: [0.0002, 0.0003]}, ValueError, "2 warmup reference controls; expected 3"),
        ({KEY: [0.0002, np.nan, 0.0003]}, ValueError, "non-finite"),
    ],
)
def test_warmup_without_file_rejects_invalid_reference(model, reference, error, message):
    with pytest.raises(error, match=message):
        set_u_bounds_and_init(model, 3, None, active_set_mode="warmup", active_reference=reference)


def test_historical_mask_still_requires_file(model):
    with pytest.raises(ValueError, match="requires an initial-guess file"):
        set_u_bounds_and_init(
            model, 3, None, active_set_mode="historical",
            active_reference={KEY: [0.0002] * 3},
        )


def test_unrestricted_without_file_keeps_pd0_seed(model):
    pd0 = model.muscles_dynamics_model[0].pd0
    bounds, guesses, _ = set_u_bounds_and_init(
        model, 3, None, active_reference={KEY: [0.0004] * 3},
    )
    np.testing.assert_allclose(guesses[KEY].init, [[pd0] * 3])
    assert np.all(bounds[KEY].min == pd0)
    assert np.all(bounds[KEY].max == 0.0006)


@pytest.mark.parametrize("active_set_mode", ["none", "historical", "warmup"])
def test_file_seed_precedence_is_preserved(model, tmp_path, active_set_mode):
    pd0 = model.muscles_dynamics_model[0].pd0
    values = np.array([[pd0, 0.0003, pd0]])
    seed_path = tmp_path / "seed.pkl"
    with seed_path.open("wb") as stream:
        pickle.dump({KEY: values}, stream)
    _, guesses, _ = set_u_bounds_and_init(
        model, 3, seed_path, active_set_mode=active_set_mode, active_margin=0,
        active_reference={KEY: [[pd0, 0.0005, pd0]]},
    )
    np.testing.assert_array_equal(guesses[KEY].init, values)
