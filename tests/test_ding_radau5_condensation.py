"""Scientific guards for the isolated prototype, not production OCP tests."""

import casadi as ca
import numpy as np
import pytest

from scripts.validate_ding_radau5_condensation import (
    Parameters,
    SCALE,
    dop853_step,
    full_rhs,
    radau5_step,
)


@pytest.mark.parametrize("hz", [30, 50])
@pytest.mark.parametrize("pw", [0.000131405, 0.00037, 0.0006])
def test_condensation_preserves_all_original_radau5_stage_equations(hz, pw):
    # Nonzero calcium/force and independent fatigue offsets matter at an RHO
    # boundary: assuming the rested fatigue manifold would fail this test.
    x0 = np.array([0.12, 12.0, 2952.0, 0.0848414, 0.1507])
    h, amplitude = 1 / hz, 1.2
    endpoint, stages, _ = radau5_step(x0, pw, h, amplitude)
    full_endpoint, full_stages, _ = radau5_step(x0, pw, h, amplitude, kind="full")
    np.testing.assert_allclose(stages / SCALE, full_stages / SCALE, rtol=0, atol=1e-9)
    np.testing.assert_allclose(endpoint / SCALE, full_endpoint / SCALE, rtol=0, atol=1e-9)
    # Independently assemble the original differentiation-form defects.
    points = ca.collocation_points(5, "radau")
    coefficients, _, _ = ca.collocation_coeff(points)
    lhs = np.array(coefficients).T @ np.vstack((x0, stages))
    rhs = h * np.array([full_rhs(c * h, x, pw, amplitude, Parameters()) for c, x in zip(points, stages)])
    np.testing.assert_allclose((lhs - rhs) / SCALE, 0, atol=1e-9)


def test_full_and_condensed_endpoint_sensitivities_match():
    # Numerical derivative guard for the map equivalence, not a claim of
    # implemented CasADi implicit sensitivities/Hessians.
    v = np.array([0.16, 35.0, 3690.0, 0.07575125, 0.17125, 0.00037, 1 / 30, 1.06])
    input_scale = np.r_[SCALE, 0.0006, 1 / 30, 1.0]
    derivatives = {}
    for kind in ("full", "condensed"):
        jac = np.empty((5, 8))
        for i in range(8):
            shift = np.zeros(8)
            shift[i] = 2e-5 * input_scale[i]
            plus, minus = v + shift, v - shift
            yp = radau5_step(plus[:5], *plus[5:], kind=kind)[0]
            ym = radau5_step(minus[:5], *minus[5:], kind=kind)[0]
            jac[:, i] = (yp - ym) / (4e-5 * SCALE)
        derivatives[kind] = jac
    np.testing.assert_allclose(derivatives["condensed"], derivatives["full"], rtol=1e-6, atol=1e-7)


def test_analytic_map_preserves_nonrest_fatigue_offsets_during_recovery():
    p = Parameters()
    x0 = np.array([0.12, 0.0, 2952.0, 0.0848414, 0.1507])
    h = 1 / 30
    endpoint, _, _ = radau5_step(x0, p.pd0, h, 1.06, kind="analytic")
    expected_slow = p.rest + np.exp(-h / p.tau_fat) * (x0[2:] - p.rest)
    np.testing.assert_allclose(endpoint[2:], expected_slow, rtol=1e-13)
    assert abs(endpoint[1]) < 1e-12
    expected_cn = np.exp(-h / p.tauc) * (x0[0] + 1.06 * h / p.tauc)
    assert endpoint[0] == pytest.approx(expected_cn, rel=1e-13)


def test_both_radau_maps_are_compared_to_independent_high_accuracy_integration():
    x0 = np.array([0.16, 35.0, 4920.0, 0.060601, 0.137])
    args = (x0, 0.0006, 1 / 30, 1.06)
    reference = dop853_step(*args)
    errors = {}
    for kind in ("full", "condensed", "analytic"):
        endpoint = radau5_step(*args, kind=kind)[0]
        errors[kind] = np.max(np.abs(endpoint - reference) / SCALE)
        # 0.1 N equivalent scaled bound is a regression guard for this case,
        # not the acceptance tolerance of a cycling optimization benchmark.
        assert errors[kind] < 1e-3
        half_args = (x0, args[1], args[2] / 2, args[3])
        half_error = np.max(np.abs(radau5_step(*half_args, kind=kind)[0] - dop853_step(*half_args)) / SCALE)
        assert half_error < errors[kind] / 2
    # No expectation that the analytically transformed force discretization
    # is identical to or always more accurate than the original Radau-5.


def test_reject_zero_initial_fatigue_and_invalid_duration():
    with pytest.raises(ValueError, match="fatigue"):
        radau5_step(np.zeros(5), 0.00037, 1 / 30, 1.06)
    with pytest.raises(ValueError, match="interval"):
        radau5_step([0.16, 35, 4920, 0.060601, 0.137], 0.00037, 0, 1.06)
