from dataclasses import replace

import numpy as np

from cocofest.identification.cycling_design import design_trial_catalogue, mono_muscle_catalogue
from cocofest.identification.synthetic_cycling import simple_mono_muscle_spec


def test_force_design_uses_complementary_pw_to_identify_recruitment():
    trials = [simple_mono_muscle_spec(
        "right_Biceps", pulse_width_s=width, duration_s=0.12, sample_rate_hz=50,
    ) for width in (0.0002, 0.0005)]
    report = design_trial_catalogue(
        trials, ("pd0", "pdt"), [{"pd0": 0.000131405, "pdt": 0.000194138}],
        [0.0001, 0.0002], count=2,
    )
    assert len(report["selected"]) == 2
    assert report["data_diagnostics"][0]["rank"] == 2
    assert report["noise_assumption"].startswith("independent")


def test_fatigue_design_holds_fitted_force_parameters_and_tracks_recovery():
    spec = simple_mono_muscle_spec("left_Triceps", duration_s=0.3, sample_rate_hz=30)
    spec = replace(spec, pulse_times_s=spec.pulse_times_s[:3], pulse_width_s=spec.pulse_width_s[:3])
    report = design_trial_catalogue(
        [spec], ("alpha_a", "tau_fat"),
        [{"alpha_a": -0.024, "tau_fat": 76.2, "pdt": 0.00021}],
        [0.024, 76.2], count=1, noise_std_n=0.1,
    )
    assert report["selected"][0]["name"] == spec.name
    assert 0 <= report["candidate_capacity_loss"][0] < 1
    assert np.all(np.isfinite(report["data_information"]))


def test_catalogue_contains_recovery_and_identical_probes():
    force = mono_muscle_catalogue("right_Biceps", "force")
    fatigue = mono_muscle_catalogue("left_Triceps", "fatigue", radial_gain=0.3)
    assert len(force) == 9 and len(fatigue) == 4
    assert all(trial.pulse_times_s[-1] < trial.time_s[-1] - 0.2 for trial in force)
    assert all(np.max(np.diff(trial.pulse_times_s)) > 9 for trial in fatigue)
    assert all(trial.pulse_width_s[-1] == 0.00035 for trial in fatigue)
    assert all(trial.radial_gain is not None for trial in fatigue)
