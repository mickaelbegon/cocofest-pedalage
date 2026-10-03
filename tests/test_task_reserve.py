"""Scientific contracts: a failed NLP never becomes an infeasibility proof."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.task_reserve import (
    DEFAULT_CONSTRAINT_GROUPS, LocalReserveSample, ProbeEvidence,
    TaskReserveCheckpoint, TaskReserveProbe, evaluate_work_reserve,
    fit_local_reserve, reserve_penalty_expression,
)
from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint


@pytest.fixture
def checkpoint(tmp_path):
    program = SimpleNamespace(
        nlp=[SimpleNamespace(
            x_init={"A_Biceps": SimpleNamespace(init=np.array([[1., .9]]))},
            u_init={"last_pulse_width_Biceps": SimpleNamespace(init=np.array([[.0002]]))},
            x_bounds={"A_Biceps": SimpleNamespace(min=np.array([[1., 0.]]), max=np.array([[1., 2.]]))},
            u_bounds={"last_pulse_width_Biceps": SimpleNamespace(min=np.array([[0.]]), max=np.array([[.0006]]))},
        )],
        parameter_init={}, parameter_bounds={}, absolute_wheel_q_cycle_index=20,
    )
    model = tmp_path / "model.json"
    model.write_text('{"synthetic_model":true}', encoding="utf-8")
    archive = tmp_path / "prepared.npz"
    export_prepared_checkpoint(archive, program, completed_cycles=20, model_path=model)
    return TaskReserveCheckpoint.from_archive(
        archive, completed_cycles=20, model_path=model,
        task_context={"nominal_work_j": 1., "arm": "right", "coordinate_names": ["A/A_rest"]})


def evidence(request, feasible=True, **overrides):
    source = request.checkpoint
    report = ProbeEvidence(request.work_scale, source.prepared_problem_sha256,
        source.task_context_sha256, source.model_sha256,
        "Solve_Succeeded" if feasible else "Infeasible_Problem_Detected",
        1e-10 if feasible else .1, 1e-8, feasible, "synthetic:test-trajectory" if feasible else None,
        .01, DEFAULT_CONSTRAINT_GROUPS, True)
    return replace(report, **overrides)


def estimate_at(checkpoint, factor):
    return evaluate_work_reserve(checkpoint, [1., factor], lambda request: evidence(request))


def test_restoration_archive_pins_all_arrays_and_model(checkpoint):
    checkpoint.verify_files()
    model = __import__("pathlib").Path(checkpoint.model_path)
    model.write_text('{"different_model":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="model changed"):
        checkpoint.verify_files()


def test_nonmonotone_witnesses_and_solver_failure_never_prove_a_maximum(checkpoint):
    requests = []

    def oracle(request):
        requests.append(request)
        # A nonconvex local failure at 1.1 must not stop the 1.2 test.
        return evidence(request, request.work_scale != 1.1)

    result = evaluate_work_reserve(checkpoint, [1.1, 1., 1.2], oracle)
    assert [request.work_scale for request in requests] == [1., 1.1, 1.2]
    assert all(request.checkpoint is checkpoint for request in requests)
    assert result.work_scale_lower_bound == 1.2
    assert result.reserve_lower_bound == pytest.approx(.2)
    assert result.nominal_task_witnessed
    assert result.status == "nominal_and_reserve_witnessed"
    assert result.global_upper_bound is None
    assert not result.physiological_failure_certified


def test_failed_nominal_with_lighter_work_is_undetermined_not_negative_proof(checkpoint):
    result = evaluate_work_reserve(checkpoint, [1., .98],
                                   lambda request: evidence(request, request.work_scale < 1.))
    assert result.reserve_lower_bound == pytest.approx(-.02)
    assert result.status == "nominal_undetermined"
    assert not result.usable_for_local_fit
    assert not result.physiological_failure_certified


def test_larger_work_witness_does_not_implicitly_certify_nominal_equality(checkpoint):
    result = evaluate_work_reserve(checkpoint, [1., 1.1],
                                   lambda request: evidence(request, request.work_scale > 1.))
    assert result.reserve_lower_bound == pytest.approx(.1)
    assert not result.nominal_task_witnessed
    assert result.status == "nominal_undetermined"


def test_status_alone_neither_accepts_nor_rejects_a_numerical_witness(checkpoint):
    request = TaskReserveProbe(checkpoint, 1.)
    feasible_nonoptimal = evidence(request, solver_status="Maximum_Iterations_Exceeded")
    assert feasible_nonoptimal.witness_is_valid(request)
    false_success = evidence(request, False, solver_status="Solve_Succeeded")
    assert not false_success.witness_is_valid(request)


@pytest.mark.parametrize("change", [
    {"maximum_normalized_constraint_violation": float("nan")},
    {"maximum_normalized_constraint_violation": -1.},
    {"maximum_normalized_constraint_violation": 1e-3},
    {"source_state_and_history_restored": False},
    {"validated_constraint_groups": ("dynamics",)},
    {"witness_id": None},
])
def test_incomplete_or_nonfinite_witness_is_refused(checkpoint, change):
    request = TaskReserveProbe(checkpoint, 1.)
    assert not evidence(request, **change).witness_is_valid(request)


@pytest.mark.parametrize("change", [
    {"prepared_problem_sha256": "different"}, {"task_context_sha256": "different"},
    {"model_sha256": "different"}, {"work_scale": 1.2},
])
def test_mismatched_source_is_an_error_not_a_failed_physiology(checkpoint, change):
    request = TaskReserveProbe(checkpoint, 1.)
    with pytest.raises(ValueError, match="different checkpoint"):
        evidence(request, **change).witness_is_valid(request)


@pytest.mark.parametrize("factors", [[], [1.1], [1., 1.], [1., float("nan")], [1., 0.]])
def test_work_factors_must_include_explicit_nominal_once(checkpoint, factors):
    with pytest.raises(ValueError):
        evaluate_work_reserve(checkpoint, factors, evidence)


def affine_samples(checkpoint):
    def sample(x, y=0., error=0.):
        # Analytic synthetic reserve R=.3 + .4*x - .2*y.
        factor = 1.3 + .4 * x - .2 * y + error
        result = estimate_at(checkpoint, factor)
        # Synthetic distinct prepared problems for the numerical fit fixture.
        result = replace(result, checkpoint=replace(checkpoint,
                         prepared_problem_sha256=f"synthetic-coordinate:{x}:{y}"))
        return LocalReserveSample((x, y), result)
    return sample, [sample(0., 0.), sample(.1, 0.), sample(-.1, 0.), sample(0., .1), sample(0., -.1)]


def test_local_fit_recovers_analytic_gradient_and_parameter_graph(checkpoint):
    sample, samples = affine_samples(checkpoint)
    model = fit_local_reserve(samples, center=(0., 0.), trust_radius=(.2, .2),
                             holdout=[sample(.03, -.04), sample(-.06, .05)])
    assert model.accepted
    np.testing.assert_allclose(model.gradient, [.4, -.2], atol=1e-13)
    assert model.margin((.05, .1)) == pytest.approx(.3)
    assert not model.globally_certified
    with pytest.raises(ValueError, match="trust box"):
        model.margin((.3, .0))

    ca = pytest.importorskip("casadi")
    state, parameters = ca.SX.sym("state", 2), ca.SX.sym("parameters", 5)
    expression = reserve_penalty_expression(state, parameters, dimension=2, target=.4, sqrt=ca.sqrt)
    function = ca.Function("task_reserve_penalty", [state, parameters], [expression, ca.gradient(expression, state)])
    p = model.parameter_vector()
    point = np.array([.02, -.05])
    value, gradient = function(point, p)
    expected = reserve_penalty_expression(point, p, dimension=2, target=.4)
    assert float(value) == pytest.approx(expected)
    finite_difference = []
    for dimension in range(2):
        perturbation = np.eye(2)[dimension] * 1e-6
        plus, _ = function(point + perturbation, p)
        minus, _ = function(point - perturbation, p)
        finite_difference.append(float(plus - minus) / 2e-6)
    np.testing.assert_allclose(np.asarray(gradient).ravel(), finite_difference, rtol=1e-6)
    changed = p.copy()
    changed[0] += .02
    changed_value, _ = function(point, changed)
    assert float(changed_value) < float(value)  # same compiled function, higher reserve


def test_gradient_pointing_to_higher_reserve_reduces_shortage_cost(checkpoint):
    sample, samples = affine_samples(checkpoint)
    fit = fit_local_reserve(samples, center=(0., 0.), trust_radius=(.2, .2), holdout=[sample(.05, .05)])
    parameters = fit.parameter_vector()
    current = reserve_penalty_expression([0., 0.], parameters, dimension=2, target=.5)
    better = reserve_penalty_expression([.05, -.05], parameters, dimension=2, target=.5)
    assert better < current


def test_fit_refuses_untested_or_rank_deficient_or_inaccurate_surrogates(checkpoint):
    sample, samples = affine_samples(checkpoint)
    no_holdout = fit_local_reserve(samples, center=(0., 0.), trust_radius=(.2, .2), holdout=[])
    assert not no_holdout.accepted
    assert "no_independent_holdout" in no_holdout.reasons
    deficient = fit_local_reserve(samples[:3], center=(0., 0.), trust_radius=(.2, .2), holdout=[sample(.04, .04)])
    assert not deficient.accepted
    assert "rank_deficient" in deficient.reasons
    bad = fit_local_reserve(samples, center=(0., 0.), trust_radius=(.2, .2), holdout=[sample(.04, .04, error=.1)])
    assert not bad.accepted
    assert "holdout_error_exceeds_tolerance" in bad.reasons
    with pytest.raises(ValueError, match="rejected"):
        bad.parameter_vector()


def test_holdout_and_provenance_requirements_prevent_self_validation(checkpoint):
    sample, samples = affine_samples(checkpoint)
    with pytest.raises(ValueError, match="independent"):
        fit_local_reserve(samples, center=(0., 0.), trust_radius=(.2, .2), holdout=samples[:1])
    altered = replace(sample(.03, .03), estimate=replace(sample(.03, .03).estimate,
                        checkpoint=replace(checkpoint, task_context_sha256="other-task")))
    with pytest.raises(ValueError, match="mix physical tasks"):
        fit_local_reserve(samples, center=(0., 0.), trust_radius=(.2, .2), holdout=[altered])


def test_coordinate_changes_need_distinct_prepared_checkpoints(checkpoint):
    baseline = estimate_at(checkpoint, 1.2)
    samples = [LocalReserveSample((point,), baseline) for point in (-.1, 0., .1)]
    with pytest.raises(ValueError, match="same prepared checkpoint"):
        fit_local_reserve(samples, center=(0.,), trust_radius=(.2,), holdout=[])


def test_fit_accuracy_is_rechecked_after_empirical_bias_correction(checkpoint):
    sample, samples = affine_samples(checkpoint)
    # Each unshifted holdout residual is <.01, but pessimistic bias from the
    # first increases the second residual beyond tolerance. Reject publication.
    fit = fit_local_reserve(samples, center=(0., 0.), trust_radius=(.2, .2),
                            holdout=[sample(.03, .03, error=-.008), sample(-.03, -.03, error=.008)],
                            maximum_error=.01)
    assert fit.empirical_bias_correction == pytest.approx(.008)
    assert fit.holdout_maximum_error == pytest.approx(.016)
    assert not fit.accepted
