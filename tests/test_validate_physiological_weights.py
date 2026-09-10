from dataclasses import asdict
import json

import numpy as np

from cocofest.optimization.physiological_muscle_weights import calculate_physiological_muscle_weights
from cocofest.optimization.physiological_weight_cases import MUSCLE_NAMES, list_cases
from scripts.validate_physiological_weights import _case_report, _comparison_candidates, _jsonable


def test_comparison_selection_uses_largest_admissible_triceps_weight_only():
    def record(name, valid, positive, margin, triceps_weight=0.):
        return {"case": {"case_id": name}, "calibration_admissible": valid,
                "triceps_positive_weight": positive, "minimum_capacity_ratio": margin,
                "weights": {"Triceps": triceps_weight}}
    rows = [record("nominal", True, False, .5), record("invalid", False, True, -8.),
            record("first", True, True, .3, .01), record("later", True, True, .1, .02)]
    result = _comparison_candidates(rows)
    assert result["largest_admissible_triceps_weight"] == "later"
    assert result["smallest_positive_calibration_capacity_margin"] == "later"
    assert result["selection_uses_fho"] is False
    assert result["must_validate_actual_rho_before_fho_comparison"] is True


def test_all_invalid_panel_has_no_positive_or_boundary_candidate():
    result = _comparison_candidates([{"case": {"case_id": "nominal"},
                                      "calibration_admissible": False}])
    assert result["largest_admissible_triceps_weight"] is None
    assert result["smallest_positive_calibration_capacity_margin"] is None


def test_case_report_keeps_explicit_parameters_and_never_promotes_legacy_invalid_weights():
    case = list_cases("repository_current")[0]
    angles = np.linspace(0, 2 * np.pi, 16)
    profiles = np.ones((4, len(angles)))
    result = calculate_physiological_muscle_weights(
        angles, profiles, case.as_parameter_dict(), muscle_names=MUSCLE_NAMES,
        case_id=case.case_id,
    )
    report = _case_report(case, result, {"profiles_recomputed_for_case_Fmax": True}, .01)
    assert report["status"] == "invalid_fatigue_domain"
    assert report["weights"] is None
    assert report["triceps_positive_weight"] is False
    assert report["calibration_admissible"] is False
    assert report["actual_rho_controller_validated"] is False
    assert report["case"] == asdict(case)
    assert report["context"]["parameters"] == case.as_parameter_dict()
    json.dumps(report, default=_jsonable, allow_nan=False)


def test_distinct_parameter_cases_are_passed_to_separate_calculations():
    cases = list_cases("published_code")[:2]
    angles = np.linspace(0, 2 * np.pi, 12)
    profiles = np.vstack([.3 + .4 * np.cos(angles + phase) for phase in np.arange(4)])
    results = [calculate_physiological_muscle_weights(
        angles, profiles, case.as_parameter_dict(), muscle_names=MUSCLE_NAMES,
        case_id=case.case_id, target_cycles=40,
    ) for case in cases]
    assert results[0]["context"]["case_id"] != results[1]["context"]["case_id"]
    assert results[0]["context"]["parameters"] != results[1]["context"]["parameters"]
    assert not np.array_equal(results[0]["ratios"], results[1]["ratios"])
