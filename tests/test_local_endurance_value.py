from dataclasses import replace

import numpy as np
import pytest

from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval
from cocofest.optimization.compact_muscle_prediction import CompactMusclePredictor, fatigue_memory_coordinates
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.local_endurance_value import (
    CompactEnduranceValueOracle, LocalEnduranceCoordinates, OracleEvaluation,
    audit_local_endurance_value, fit_local_endurance_value,
)


def setup_oracle(targets=(.7, -.2), coefficients=(.05, -.03), horizon=3):
    parameter = DingPulseWidthParameters(
        DingFatigueParameters(1200., .060601, .137, -1.4, 2.1e-5, 1.9e-5, 445.5),
        tauc=.011, tau2=.001, pd0=.000131405, pdt=.000194138, pulse_width_max=.0006,
    )
    parameters = (parameter,) * len(targets)
    phase = MomentTrackingInterval(1 / 30, (1.0597355478,) * len(targets),
                                   (.95,) * len(targets), coefficients, targets)
    predictor = CompactMusclePredictor((phase,), parameters, substeps=16)
    state = np.array([[.16298, 20., 1150., .078, .15]] * len(targets))
    coordinates = LocalEnduranceCoordinates.from_state(state, parameters, force_scale=100.)
    return CompactEnduranceValueOracle(predictor, coordinates, horizon_cycles=horizon, moment_scale=1.)


def test_coordinates_exact_nonzero_offsets_fixed_calcium_and_normalized_force():
    oracle = setup_oracle()
    coordinates = oracle.coordinates
    state = coordinates.decode(coordinates.anchor)
    np.testing.assert_allclose(coordinates.encode(state), coordinates.anchor)
    assert np.max(np.abs(coordinates.offsets)) > .01
    changed = coordinates.anchor + [.001, -.001, .02, -.01]
    reconstructed = coordinates.decode(changed)
    np.testing.assert_array_equal(reconstructed[:, 0], coordinates.fixed_cn)
    np.testing.assert_allclose(reconstructed[:, 1], [22., 19.])
    for i in range(2):
        damage, offsets = fatigue_memory_coordinates(reconstructed[i, 2:], coordinates.parameters[i].fatigue)
        assert damage == pytest.approx(changed[i])
        np.testing.assert_allclose(offsets, coordinates.offsets[i], atol=1e-15)
    state[0, 0] += 1e-5
    with pytest.raises(ValueError, match="context"):
        coordinates.encode(state)
    state = coordinates.decode(coordinates.anchor)
    state[1, 3] += .001
    with pytest.raises(ValueError, match="context"):
        coordinates.encode(state)


def test_context_digest_snapshots_calcium_offsets_force_scale_parameters_and_anchor():
    oracle = setup_oracle()
    coordinates = oracle.coordinates
    baseline = coordinates.context_signature
    fit = fit_local_endurance_value(oracle, trust_radius=.0001)
    assert fit.metadata["coordinate_context_sha256"] == baseline
    for field in ("fixed_cn", "offsets", "force_scale", "anchor"):
        altered = replace(coordinates, **{field: getattr(coordinates, field) + .0001})
        assert altered.context_signature != baseline
    parameters = tuple(replace(p, tauc=p.tauc + .0001) for p in coordinates.parameters)
    assert replace(coordinates, parameters=parameters).context_signature != baseline
    coordinates.fixed_cn[0] += .0001
    assert coordinates.context_signature != baseline
    assert fit.metadata["coordinate_context_sha256"] == baseline


@pytest.mark.parametrize("targets,coefficients", [((.7, -.2), (.05, -.03)), ((-.7, .2), (-.05, .03))])
def test_signed_margin_matches_attainable_total_envelope_for_either_target_direction(targets, coefficients):
    oracle = setup_oracle(targets, coefficients, horizon=1)
    state = oracle.coordinates.decode(oracle.coordinates.anchor)
    transition = oracle.predictor.phase_map(state, 0)
    moments0 = np.asarray(coefficients) * transition.intercept[:, 1]
    moments1 = np.asarray(coefficients) * transition.endpoint(transition.maximum_recruitment)[:, 1]
    margin = min(sum(targets) - np.minimum(moments0, moments1).sum(),
                 np.maximum(moments0, moments1).sum() - sum(targets))
    result = oracle.evaluate(oracle.coordinates.anchor)
    assert result.status == "complete"
    assert result.minimum_signed_margin == pytest.approx(margin)
    assert result.soft_minimum_margin <= margin
    assert np.isfinite(result.value)


def test_zero_target_and_tiny_softening_are_finite():
    oracle = setup_oracle((.2, -.2), (.05, -.05), horizon=1)
    oracle.softmin_temperature = 1e-12
    oracle.penalty_temperature = 1e-12
    result = oracle.evaluate(oracle.coordinates.anchor)
    assert result.status == "complete"
    assert np.isfinite(result.value)


def test_inadmissible_maximal_recruitment_does_not_count_as_attainable_reserve():
    base = setup_oracle((.0001,), (.05,), horizon=1)
    parameter = base.predictor.parameters[0]
    parameter = replace(parameter, fatigue=replace(parameter.fatigue, alpha_a=-10000.))
    state = base.coordinates.decode(base.coordinates.anchor)
    state[:, 1] = .001
    coordinates = LocalEnduranceCoordinates.from_state(state, (parameter,), force_scale=100.)
    predictor = CompactMusclePredictor(base.predictor.intervals, (parameter,))
    assert predictor.rollout(state, horizon_cycles=1).status == "complete"
    oracle = CompactEnduranceValueOracle(predictor, coordinates, moment_scale=1., horizon_cycles=1)
    result = oracle.evaluate(coordinates.anchor)
    assert result.status == "envelope_domain_invalid"
    assert result.value is None


def test_impossible_policy_and_invalid_domain_never_produce_fit_values():
    oracle = setup_oracle((100., 100.))
    result = oracle.evaluate(oracle.coordinates.anchor)
    assert result.status == "policy_failed"
    assert result.value is None
    fit = fit_local_endurance_value(oracle, trust_radius=1e-3)
    assert not fit.accepted and fit.model is None
    assert fit.metadata["training_evaluations"] == 1
    oracle = setup_oracle()
    fit = fit_local_endurance_value(oracle, trust_radius=[.001, .001, .3, .001])
    assert not fit.accepted and fit.model is None
    assert "Trust domain invalid" in fit.audit.reason
    assert fit.metadata["training_evaluations"] == 0


def test_real_policy_linear_fit_has_independent_directional_audit_and_fixed_size():
    oracle = setup_oracle()
    fit = fit_local_endurance_value(oracle, trust_radius=[.0002, .0002, .0001, .0001],
                                   absolute_tolerance=1e-5, ranking_tolerance=1e-10)
    assert fit.accepted, fit.audit
    model = fit.model
    assert model.coefficients.size == 1 + 4 * oracle.predictor.muscle_count
    assert np.all(model.diagonal_hessian == 0)
    assert fit.audit.ranking_pairs > 0 and fit.audit.ranking_failures == 0
    assert fit.audit.maximum_scaled_error < 1
    assert len(fit.audit.records) == fit.audit.heldout_count
    assert fit.metadata["training_evaluations"] == 1 + 2 * model.center.size
    assert model.evaluate(model.center) == pytest.approx(oracle.evaluate(model.center).value)
    with pytest.raises(ValueError, match="trust box"):
        model.evaluate(model.upper_bounds + .001)


class PolynomialOracle:
    """Known smooth value for isolating the finite-difference and audit gates."""
    def __init__(self, function):
        real = setup_oracle()
        self.__dict__.update(real.__dict__)
        self.function = function

    def evaluate(self, xi):
        x = np.asarray(xi) - self.coordinates.anchor
        return OracleEvaluation("complete", float(self.function(x)), 1., 1., 3)


def test_diagonal_quadratic_derivatives_match_known_value_on_independent_points():
    gradient = np.array([1., -2., 3., -4.])
    curvature = np.array([.2, -.3, .4, .5])
    oracle = PolynomialOracle(lambda x: 2 + gradient @ x + .5 * curvature @ (x * x))
    fit = fit_local_endurance_value(oracle, trust_radius=.001, kind="diagonal_quadratic",
                                   absolute_tolerance=1e-10, relative_tolerance=0)
    assert fit.accepted, fit.audit
    np.testing.assert_allclose(fit.model.gradient, gradient, atol=1e-10)
    np.testing.assert_allclose(fit.model.diagonal_hessian, curvature, atol=2e-8)
    assert fit.audit.maximum_absolute_error < 1e-10


def test_independent_audit_rejects_training_points_bad_direction_and_flat_values():
    oracle = PolynomialOracle(lambda x: 1 + x[0] - 2 * x[1])
    fit = fit_local_endurance_value(oracle, trust_radius=.001)
    assert fit.accepted
    model = fit.model
    overlap = audit_local_endurance_value(model, oracle, [model.center, model.upper_bounds])
    assert not overlap.accepted and "fitted sample" in overlap.reason
    reversed_model = replace(model, gradient=-model.gradient)
    bad = audit_local_endurance_value(reversed_model, oracle, [model.lower_bounds, model.upper_bounds],
                                      absolute_tolerance=1.)
    assert not bad.accepted and bad.ranking_failures > 0
    flat = fit_local_endurance_value(PolynomialOracle(lambda x: 1.), trust_radius=.001)
    assert not flat.accepted and flat.audit.ranking_pairs == 0


def test_central_difference_fit_cannot_hide_cross_terms_from_mixed_heldouts():
    oracle = PolynomialOracle(lambda x: 1 + x[0] + 1e6 * x[0] * x[1])
    fit = fit_local_endurance_value(oracle, trust_radius=.001, kind="diagonal_quadratic",
                                   absolute_tolerance=1e-4, relative_tolerance=0.)
    assert not fit.accepted and fit.model is None
    assert fit.audit.maximum_absolute_error >= .9


def test_infeasible_heldout_fails_closed_even_when_all_fitted_samples_complete():
    oracle = PolynomialOracle(lambda x: 1 + x[0] - 2 * x[1])
    original = oracle.evaluate
    def evaluate(xi):
        if abs(xi[0] - oracle.coordinates.anchor[0]) > .0009:
            return OracleEvaluation("policy_failed", None, None, None, 0, "chosen policy failed")
        return original(xi)
    oracle.evaluate = evaluate
    fit = fit_local_endurance_value(oracle, trust_radius=.001)
    assert not fit.accepted and fit.model is None
    assert fit.metadata["training_evaluations"] == 9
    assert "Held-out oracle policy_failed" in fit.audit.reason


@pytest.mark.parametrize("keyword,value", [("moment_scale", 0), ("softmin_temperature", -1),
                                          ("penalty_temperature", np.nan), ("horizon_cycles", 0)])
def test_invalid_oracle_options_are_explicit(keyword, value):
    oracle = setup_oracle()
    options = {"moment_scale": 1., keyword: value}
    with pytest.raises(ValueError):
        CompactEnduranceValueOracle(oracle.predictor, oracle.coordinates, **options)
