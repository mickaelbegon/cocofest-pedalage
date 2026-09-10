from types import SimpleNamespace

import numpy as np

from scripts.validate_preview_muscle_allocation import (
    _compact_tracking_audit,
    _diagnostic_summary,
    _pw_and_state_audit,
)


def test_diagnostic_summary_counts_rejections_fallbacks_and_audited_optimizer_status():
    records = (
        {"reason": "accepted", "accepted": True, "qp_solved": True, "optimizer_success": False,
         "greedy_fallback": False, "elapsed_s": 0.01, "equality_residual_max_nm": 1e-10},
        {"reason": "kkt_failed", "accepted": False, "qp_solved": True, "optimizer_success": False,
         "greedy_fallback": True, "elapsed_s": 0.02, "equality_residual_max_nm": 1e-5},
    )
    summary = _diagnostic_summary(records)
    assert summary["accepted_plans"] == 1
    assert summary["rejected_plans"] == 1
    assert summary["optimizer_unsuccessful_but_full_audit_accepted"] == 1
    assert summary["greedy_fallback_steps"] == 1
    assert summary["maximum_equality_residual_nm"] == 1e-5


def test_compact_tracking_audit_ignores_only_uncompleted_nan_suffix():
    result = SimpleNamespace(signed_moment_errors=np.array([[1e-9, -2e-9], [np.nan, np.nan]]))
    audit = _compact_tracking_audit(result)
    assert audit["successful_steps"] == 2
    assert audit["maximum_absolute_original_target_error_nm"] == 2e-9
    np.testing.assert_allclose(audit["signed_error_sum_nm"], -1e-9)
    np.testing.assert_allclose(audit["absolute_error_sum_nm"], 3e-9)


def test_pw_and_state_audit_checks_completed_count_and_bounds():
    parameters = (SimpleNamespace(pd0=0.1, pulse_width_max=0.5),)
    result = SimpleNamespace(
        completed_intervals=2,
        pulse_widths=np.array([[[0.1, 0.5]], [[np.nan, np.nan]]]),
        state_history=np.array(
            [
                [[0.2, 1.0, 10.0, 0.05, 0.1]],
                [[0.1, 2.0, 9.0, 0.04, 0.09]],
                [[0.05, 3.0, 8.0, 0.03, 0.08]],
            ]
        ),
    )
    audit = _pw_and_state_audit(result, parameters)
    assert audit["finite_pw_count"] == audit["expected_finite_pw_count"] == 2
    assert audit["maximum_pw_bound_violation_s"] == 0.0
    assert audit["minimum_capacity"] == 8.0
