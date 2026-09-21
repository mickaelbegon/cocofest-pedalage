import json

import numpy as np

from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from scripts.validate_ding_fixed_pulse_width import (
    FixedPulseWidthDingCase,
    _configured_parameters_from_file,
    periodic_post_stimulation_amplitude,
    validate_case,
)


def _parameters():
    return DingPulseWidthParameters(
        fatigue=DingFatigueParameters(
            a_rest=4920.0, tau1_rest=0.060601, km_rest=0.137,
            alpha_a=-0.04, alpha_tau1=2.1e-6, alpha_km=1.9e-6, tau_fat=127.0,
        ),
        tauc=0.011, tau2=0.001, pd0=0.000131405, pdt=0.000194138,
        pulse_width_max=0.0006,
    )


def test_fixed_pw_validation_compares_all_five_ding_states():
    parameters = _parameters()
    report = validate_case(
        FixedPulseWidthDingCase(
            name="test", initial_state=np.array([0.25, 20.0, 4920.0, 0.060601, 0.137]),
            pulse_widths=np.array([0.00035, 0.0004, 0.00032, 0.00045]), duration_s=0.02,
            calcium_amplitude=periodic_post_stimulation_amplitude(
                duration_s=0.02, tauc=parameters.tauc, km_rest=parameters.fatigue.km_rest,
                retained_stimulations=6,
            ),
            parameters=parameters,
        )
    )
    assert report["interval_count"] == 4
    for degree in ("3", "5"):
        result = report["degrees"][degree]
        assert set(result["maximum_absolute_endpoint_error_by_state"]) == {"Cn", "F", "A", "Tau1", "Km"}
        assert result["maximum_collocation_residual"] < 1e-8
        assert np.isfinite(result["maximum_absolute_endpoint_error"])
    assert (report["degrees"]["5"]["maximum_absolute_endpoint_error"]
            < report["degrees"]["3"]["maximum_absolute_endpoint_error"])


def test_calcium_amplitude_reproduces_the_truncated_periodic_history():
    # The value is independently useful as a regression guard for the exact
    # periodic-node forcing at 50 Hz, six retained stimulations.
    amplitude = periodic_post_stimulation_amplitude(
        duration_s=0.02, tauc=0.011, km_rest=0.137, retained_stimulations=6
    )
    np.testing.assert_approx_equal(amplitude, 1.2280464736, significant=10)


def test_legacy_seed_parameter_recovery_requires_an_explicit_model_config(tmp_path):
    """The audit must record a declared configuration, never use defaults."""

    declared = {
        "schema_version": 1,
        "case_id": "test-case",
        "provenance": "unit test",
        "muscles": {
            "Triceps": {
                "Fmax": 617.0,
                "a_scale": 7036.3,
                "alpha_a": -0.12,
                "tau_fat": 76.2,
            }
        },
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(declared))
    parameters, provenance = _configured_parameters_from_file(path)
    assert parameters["Triceps"]["a_scale"] == 7036.3
    assert provenance["source"] == "explicit_model_config"
    assert provenance["path"] == str(path.resolve())
