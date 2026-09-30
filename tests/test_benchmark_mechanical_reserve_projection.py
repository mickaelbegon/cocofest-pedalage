"""Focused checks for the standalone RHO-Réserve projection experiment."""

import json
import math

import numpy as np
import pytest

from scripts.benchmark_mechanical_reserve_projection import (
    GATES,
    benchmark_case,
    gradient_finite_difference,
    main,
    margin_score,
    project_analytic,
    project_reference,
    synthetic_case,
)


@pytest.mark.parametrize("case_name", ["constant", "piecewise"])
@pytest.mark.parametrize("horizon", [1, 20, 100])
def test_analytic_ding_states_and_affine_margins_match_dop853(case_name, horizon):
    case = synthetic_case(case_name)
    analytic = project_analytic(case, horizon)
    reference = project_reference(case, horizon)
    np.testing.assert_allclose(analytic, reference, atol=2e-10, rtol=2e-11)
    analytic_margin, analytic_score, _ = margin_score(case, analytic)
    reference_margin, reference_score, _ = margin_score(case, reference)
    np.testing.assert_allclose(analytic_margin, reference_margin, atol=2e-11, rtol=2e-11)
    assert analytic_score == pytest.approx(reference_score, abs=2e-11)


def test_zero_force_relaxes_to_rest_and_signed_gains_survive():
    case = synthetic_case("piecewise")
    case["forces"][:] = 0
    states = project_analytic(case, 20)
    elapsed = 20 * float(np.sum(case["durations"]))
    expected = case["rest"] + np.exp(-elapsed / case["tau_fat"]) * (case["initial"] - case["rest"])
    np.testing.assert_allclose(states[-1, -1], expected, atol=2e-10, rtol=1e-12)
    assert np.any(case["gain"][:, :, 0] > 0)
    assert np.any(case["gain"][:, :, 0] < 0)


def test_initial_state_gradient_and_softmin_bound():
    case = synthetic_case("piecewise")
    margins, score, gradient = margin_score(case, project_analytic(case, 20), gradient=True)
    finite_difference = gradient_finite_difference(case, 20)
    np.testing.assert_allclose(gradient, finite_difference, atol=GATES["max_gradient_absolute_error"], rtol=GATES["max_gradient_relative_error"])
    assert score <= float(np.min(margins))
    assert float(np.min(margins)) - score <= .001 * math.log(margins.size) + 1e-15


def test_normalized_softmin_is_retained_as_negative_control():
    row = benchmark_case(synthetic_case("piecewise"), 100, analytic_repeats=2, reference_repeats=1)
    assert row["minimum_margin"] < 0
    assert row["smooth_minimum_margin"] < 0
    assert row["normalized_smooth_minimum_diagnostic"] > 0
    assert row["negative_control_normalized_false_safe"] is True
    assert row["risk_sign_gate_pass"] is True


def test_cli_writes_machine_readable_report_without_ocp(tmp_path, capsys):
    path = tmp_path / "benchmark.json"
    status = main(["--horizons", "1", "--analytic-repeats", "2", "--reference-repeats", "1", "--output", str(path)])
    report = json.loads(path.read_text())
    assert status == 0
    assert json.loads(capsys.readouterr().out) == report
    assert report["schema"] == "cocofest-mechanical-reserve-projection-benchmark-v1"
    assert report["rho_integration_authorized_by_this_benchmark"] is False
    assert len(report["cases"]) == 2
    assert all(row["accuracy_gate_pass"] for row in report["cases"])
    assert report["registered_gates"] == GATES


def test_invalid_force_domain_is_rejected():
    case = synthetic_case("constant")
    case["forces"][0, 0] = -.01
    with pytest.raises(ValueError, match="physiological domain"):
        project_analytic(case, 1)
