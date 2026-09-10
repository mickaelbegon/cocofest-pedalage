from dataclasses import replace

import numpy as np
import pytest

from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval
from cocofest.optimization.compact_muscle_prediction import CompactMusclePredictor
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.local_endurance_value import (
    CompactEnduranceValueOracle, LocalEnduranceCoordinates, fit_local_endurance_value,
)
from cocofest.optimization.batched_endurance_value import BatchedEnduranceValueOracle
from cocofest.optimization.local_endurance_value_ocp import LocalEnduranceValueBinding


def _problem():
    p = DingPulseWidthParameters(
        DingFatigueParameters(1200., .060601, .137, -1.4, 2.1e-5, 1.9e-5, 445.5),
        tauc=.011, tau2=.001, pd0=.000131405, pdt=.000194138, pulse_width_max=.0006,
    )
    phase = MomentTrackingInterval(1/30, (1.0597355478,)*2, (.95,)*2, (.05, -.03), (.7, -.2))
    predictor = CompactMusclePredictor((phase,), (p, p))
    coordinates = LocalEnduranceCoordinates.from_state(
        [[.16295396, 20., 1150., .078, .15]]*2, (p, p), force_scale=100.)
    return predictor, coordinates


def test_batched_values_and_audited_fit_match_scalar_exact_mode():
    predictor, coordinates = _problem()
    options = dict(moment_scale=1., horizon_cycles=3)
    scalar = CompactEnduranceValueOracle(predictor, coordinates, **options)
    batch = BatchedEnduranceValueOracle(predictor, coordinates, **options)
    points = coordinates.anchor + np.random.default_rng(9).uniform(-1e-4, 1e-4, (11, 4))
    results = batch.evaluate_many(points)
    for point, result in zip(points, results):
        expected = scalar.evaluate(point)
        assert result.status == expected.status == "complete"
        assert result.value == pytest.approx(expected.value, abs=1e-13)
        assert result.minimum_signed_margin == pytest.approx(expected.minimum_signed_margin, abs=1e-13)
        assert result.tracking_mode == "exact_with_numerical_tolerance"
        assert result.tracking_cost == 0.
    settings = dict(trust_radius=[.0002, .0002, .0001, .0001], kind="diagonal_quadratic",
                    absolute_tolerance=1e-5, ranking_tolerance=1e-10)
    first, second = (fit_local_endurance_value(oracle, **settings) for oracle in (scalar, batch))
    assert first.accepted and second.accepted
    np.testing.assert_allclose(second.model.gradient, first.model.gradient, atol=1e-10)
    np.testing.assert_allclose(second.model.diagonal_hessian, first.model.diagonal_hessian, atol=1e-5)
    assert second.audit.ranking_failures == first.audit.ranking_failures == 0
    assert second.audit.ranking_pairs == first.audit.ranking_pairs
    assert second.metadata["evaluation_backend"] == "batch"
    assert second.metadata["training_evaluations"] == 9
    assert second.metadata["value_context"]["task_sha256"] == first.metadata["value_context"]["task_sha256"]
    binding = LocalEnduranceValueBinding(first, coordinates, ["m0", "m1"])
    # Backend choice alone must not force a new compiled NLP.
    assert binding._pack(second, coordinates, 1.).shape == (30,)


def test_invalid_rows_and_failed_center_do_not_hide_successful_rows_or_add_fit_values():
    predictor, coordinates = _problem()
    oracle = BatchedEnduranceValueOracle(predictor, coordinates, moment_scale=1., horizon_cycles=3)
    points = np.tile(coordinates.anchor, (3, 1))
    points[1, 2] = -1.
    results = oracle.evaluate_many(points)
    assert [r.status for r in results] == ["complete", "domain_invalid", "complete"]
    assert results[1].value is None
    assert oracle.evaluate_many(np.empty((0, 4))) == ()
    impossible = replace(predictor.intervals[0], target_moments=(100., 100.))
    failed = BatchedEnduranceValueOracle(CompactMusclePredictor((impossible,), predictor.parameters),
                                       coordinates, moment_scale=1., horizon_cycles=3)
    fit = fit_local_endurance_value(failed, trust_radius=.0001)
    assert not fit.accepted and fit.model is None
    assert fit.metadata["training_evaluations"] == 1


def test_physical_band_keeps_original_target_margin_error_and_fixed_penalty_scale():
    predictor, coordinates = _problem()
    phase = replace(predictor.intervals[0], moment_coefficients=(.05, .03))
    first = CompactMusclePredictor((phase,), predictor.parameters)
    transition = first.phase_map(coordinates.decode(coordinates.anchor), 0)
    lower = np.dot(phase.moment_coefficients, transition.intercept[:, 1])
    target = lower - 1e-4
    phase = replace(phase, target_moments=(target/2, target/2))
    first = CompactMusclePredictor((phase,), predictor.parameters)
    strict = BatchedEnduranceValueOracle(first, coordinates, moment_scale=2., horizon_cycles=1)
    assert strict.evaluate(coordinates.anchor).status == "policy_failed"
    band = BatchedEnduranceValueOracle(first, coordinates, moment_scale=2., horizon_cycles=1,
                                       tracking_band_nm=2e-4, tracking_penalty_weight=3.)
    result = band.evaluate(coordinates.anchor)
    assert result.status == "complete" and result.tracking_mode == "bounded_tracking"
    assert result.relaxed_steps == 1
    assert result.minimum_signed_margin == pytest.approx(-1e-4/2, abs=1e-12)
    assert result.max_abs_moment_error_nm == pytest.approx(1e-4, abs=1e-12)
    assert result.signed_error_quadrature_nm_s == pytest.approx(1e-4/30, abs=1e-12)
    assert result.absolute_error_quadrature_nm_s == pytest.approx(1e-4/30, abs=1e-12)
    assert result.tracking_cost == pytest.approx(3*(1e-4/2)**2, abs=1e-15)
    assert result.value == pytest.approx(result.margin_cost + result.tracking_cost)
    assert band.value_context_metadata["tracking_band_nm"] == 2e-4
    assert band.value_context_metadata["task_sha256"] == strict.value_context_metadata["task_sha256"]


def test_fit_dispatches_batches_and_binding_requires_explicit_band_opt_in():
    predictor, coordinates = _problem()
    oracle = BatchedEnduranceValueOracle(predictor, coordinates, moment_scale=1., horizon_cycles=3,
                                         tracking_band_nm=1e-4)
    sizes = []
    original = oracle.evaluate_many
    def observe(points):
        sizes.append(len(points))
        return original(points)
    oracle.evaluate_many = observe
    fit = fit_local_endurance_value(oracle, trust_radius=.0001, ranking_tolerance=1e-10)
    assert fit.accepted
    assert sizes[0] == 1 and sizes[1] == 8 and sizes[2] == fit.audit.heldout_count
    with pytest.raises(ValueError, match="allow_tracking_band"):
        LocalEnduranceValueBinding(fit, coordinates, ["m0", "m1"])
    binding = LocalEnduranceValueBinding(fit, coordinates, ["m0", "m1"], allow_tracking_band=True)
    changed_context = {**fit.metadata["value_context"], "tracking_band_nm": 2e-4}
    changed = replace(fit, metadata={**fit.metadata, "value_context": changed_context})
    with pytest.raises(ValueError, match="tracking task"):
        binding._pack(changed, coordinates, 1.)
    assert binding.parameter_size == 30


@pytest.mark.parametrize("name,value", [("tracking_band_nm", -1.), ("tracking_penalty_weight", np.nan)])
def test_invalid_band_settings_are_rejected(name, value):
    predictor, coordinates = _problem()
    with pytest.raises(ValueError):
        BatchedEnduranceValueOracle(predictor, coordinates, moment_scale=1., **{name: value})
