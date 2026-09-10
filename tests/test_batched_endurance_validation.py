from types import SimpleNamespace

import numpy as np

from scripts.validate_batched_endurance_value import (
    _finite_max_difference,
    _tracking_metrics,
    _validate_baseline_case,
    _validate_baseline_report,
)


def test_finite_difference_requires_matching_nan_layout():
    assert _finite_max_difference([1.0, np.nan, 3.0], [1.5, np.nan, 2.0]) == 1.0
    assert _finite_max_difference([1.0, np.nan], [1.0, 2.0]) is None
    assert _finite_max_difference([1.0, np.inf], [1.0, np.inf]) is None
    assert _finite_max_difference([1.0, np.inf], [1.0, 2.0]) is None


def test_tracking_metrics_use_phase_durations_and_original_signed_error():
    intervals = (SimpleNamespace(duration=0.25), SimpleNamespace(duration=0.75))
    result = SimpleNamespace(
        signed_moment_errors=np.array([[0.2, -0.1], [np.nan, np.nan]]),
        relaxed_steps=2,
        tracking_mode="bounded_tracking",
    )
    metrics = _tracking_metrics(result, intervals, horizon=2)
    assert metrics["successful_tracking_steps"] == 2
    assert metrics["maximum_absolute_error_nm"] == 0.2
    np.testing.assert_allclose(metrics["signed_error_quadrature_nm_s"], -0.025)
    np.testing.assert_allclose(metrics["absolute_error_quadrature_nm_s"], 0.125)
    assert metrics["relaxed_steps"] == 2


def _baseline():
    return {
        "schema": "cocofest-local-endurance-value-validation-v1",
        "uses_fho_data": False,
        "source": {"sha256": "source"},
        "reduced_profile": {"sha256": "profile"},
        "fit_protocol": {"kind": "diagonal_quadratic"},
        "thresholds": {
            "absolute_tolerance": 0.002,
            "relative_tolerance": 0.05,
            "ranking_tolerance": 1e-5,
        },
        "cases": [
            {"source_cycle_index_zero_based": 0, "horizon_cycles": 10, "predictor_substeps": 16},
            {"source_cycle_index_zero_based": 112, "horizon_cycles": 10, "predictor_substeps": 16},
        ],
    }


def test_baseline_report_rejects_foreign_source_threshold_and_substeps():
    kwargs = dict(
        source_sha256="source", profile_sha256="profile", substeps=16,
        absolute_tolerance=0.002, relative_tolerance=0.05, ranking_tolerance=1e-5,
    )
    assert set(_validate_baseline_report(_baseline(), **kwargs)) == {0, 112}
    for mutation in (
        lambda value: value["source"].update(sha256="foreign"),
        lambda value: value["thresholds"].update(ranking_tolerance=2e-5),
        lambda value: value["cases"][0].update(predictor_substeps=8),
    ):
        baseline = _baseline()
        mutation(baseline)
        with np.testing.assert_raises_regex(ValueError, "Incompatible scalar baseline"):
            _validate_baseline_report(baseline, **kwargs)


def test_baseline_case_binds_coordinate_context_horizon_and_scale():
    coordinates = SimpleNamespace(anchor=np.array([0.1, 0.2]), context_signature="context")
    case = {
        "source_cycle_index_zero_based": 112,
        "horizon_cycles": 10,
        "predictor_substeps": 16,
        "fit_kind": "diagonal_quadratic",
        "fit_accepted": True,
        "coordinate_context_sha256": "context",
        "moment_scale_nm": 0.5,
        "coordinate_center": [0.1, 0.2],
    }
    _validate_baseline_case(
        case, anchor=112, horizon=10, coordinates=coordinates, moment_scale=0.5, substeps=16
    )
    for name, value in (
        ("coordinate_context_sha256", "foreign"),
        ("horizon_cycles", 20),
        ("moment_scale_nm", 0.6),
    ):
        incompatible = {**case, name: value}
        with np.testing.assert_raises_regex(ValueError, "Incompatible scalar baseline for anchor"):
            _validate_baseline_case(
                incompatible,
                anchor=112,
                horizon=10,
                coordinates=coordinates,
                moment_scale=0.5,
                substeps=16,
            )
