"""Synthetic oracle tests of asynchronous acceptance, signs and provenance."""
from dataclasses import replace

import numpy as np
import pytest

from cocofest.optimization.task_reserve import DEFAULT_CONSTRAINT_GROUPS, ProbeEvidence, TaskReserveCheckpoint
from cocofest.optimization.task_load_margin import (
    TaskLoadMarginRequest, TaskLoadMarginResult, TaskLoadMarginPolicy,
    load_gradient_from_kkt, select_task_load_margin, validate_task_load_margin,
)


@pytest.fixture
def case():
    checkpoint = TaskReserveCheckpoint("test-only.npz", "a" * 64, "b" * 64, 100,
        "test-only.model", "c" * 64, '{}', "d" * 64)
    request = TaskLoadMarginRequest("request-100", checkpoint, ("A", "Km"), (1., 1.),
                                   (.1, .1), (.1, .1), (2., 2.), 200., 3.)
    witness = ProbeEvidence(1., checkpoint.prepared_problem_sha256, checkpoint.task_context_sha256,
        checkpoint.model_sha256, "Solve_Succeeded", 1e-9, 1e-8, True, "trajectory.npz", .5,
        DEFAULT_CONSTRAINT_GROUPS, True)
    result = TaskLoadMarginResult("request-100", 201., witness, replace(witness, work_scale=1.2),
        (.5, -.2), "kkt_envelope", True, 1e-8, 1e-8, 1e-8, .001, .001, 3,
        solution_artifact="solution.npz", multipliers_artifact="multipliers.npz")
    context = dict(policy=TaskLoadMarginPolicy(), completed_cycles=102, now_monotonic_seconds=203.,
                   coordinates=(1., 1.), model_sha256=checkpoint.model_sha256,
                   task_context_sha256=checkpoint.task_context_sha256, coordinate_names=("A", "Km"))
    return request, result, context


def test_accepted_affine_sign_and_fixed_graph_parameters(case):
    request, result, context = case
    selection = select_task_load_margin([(request, result)], **context)
    assert selection.action == "activate"
    assert not selection.physiological_failure_certified
    direction = selection.direction
    assert direction.value((1.04, .98)) == pytest.approx(.224)
    p = direction.parameter_vector(20.)
    np.testing.assert_allclose(p, [20., 1., 1., .5, -.2])
    assert -p[0] * np.dot(p[3:], np.array([1.04, .98]) - p[1:3]) < 0
    with pytest.raises(ValueError, match="trust_region"):
        direction.value((1.2, 1.))


def test_worker_contract_copies_caller_owned_vectors(case):
    request, result, _ = case
    center, gradient = [1., 1.], [.5, -.2]
    copied_request = replace(request, center=center)
    copied_result = replace(result, gradient=gradient)
    center[0], gradient[0] = 99., 99.
    assert copied_request.center == (1., 1.)
    assert copied_result.gradient == (.5, -.2)


@pytest.mark.parametrize("changes,reason", [
    ({"request_id": "other"}, "request_id_mismatch"),
    ({"completed_monotonic_seconds": 199.}, "time_age_invalid"),
    ({"completed_monotonic_seconds": 205.}, "time_age_invalid"),
    ({"completed_monotonic_seconds": float("nan")}, "time_age_invalid"),
    ({"failure_reason": "solver timeout"}, "oracle_reported_failure"),
    ({"nominal_witness": None}, "missing_nominal_witness"),
    ({"load_witness": None}, "missing_load_witness"),
    ({"gradient": None}, "invalid_gradient"),
    ({"gradient": (1.,)}, "invalid_gradient"),
    ({"gradient": (1., float("inf"))}, "invalid_gradient"),
    ({"sensitivity_regular": False}, "irregular_sensitivity"),
    ({"sensitivity_method": "guessed"}, "unsupported_sensitivity_method"),
    ({"stationarity_residual": .1}, "invalid_stationarity_residual"),
    ({"complementarity_residual": None}, "invalid_complementarity_residual"),
    ({"dual_feasibility_residual": -1.}, "invalid_dual_feasibility_residual"),
    ({"gradient_validation_max_abs_error": .1}, "invalid_gradient_validation_max_abs_error"),
    ({"local_validation_max_abs_error": float("nan")}, "invalid_local_validation_max_abs_error"),
    ({"independent_validation_points": 0}, "missing_independent_local_validation"),
    ({"solution_artifact": None}, "missing_solution_artifact"),
])
def test_result_fail_closed(case, changes, reason):
    request, result, context = case
    selection = select_task_load_margin([(request, replace(result, **changes))], **context)
    assert selection.direction is None
    assert selection.action == "deactivate"
    assert reason in selection.rejected[0][1]
    assert not selection.physiological_failure_certified


@pytest.mark.parametrize("changes,reason", [
    ({"completed_cycles": 99}, "cycle_age_invalid"),
    ({"completed_cycles": 121}, "cycle_age_invalid"),
    ({"now_monotonic_seconds": 221.}, "time_age_invalid"),
    ({"model_sha256": "different"}, "context_mismatch"),
    ({"task_context_sha256": "different"}, "context_mismatch"),
    ({"coordinate_names": ("Km", "A")}, "context_mismatch"),
    ({"coordinates": (1.11, 1.)}, "outside_trust_region"),
    ({"coordinates": (0., 1.)}, "outside_physical_domain"),
    ({"coordinates": (float("nan"), 1.)}, "invalid_coordinates"),
])
def test_application_context_fail_closed(case, changes, reason):
    request, result, context = case
    assert reason in validate_task_load_margin(request, result, **(context | changes))


@pytest.mark.parametrize("changes", [
    {"prepared_problem_sha256": "different"},
    {"validated_constraint_groups": ("dynamics",)},
    {"source_state_and_history_restored": False},
    {"independent_validation_passed": False},
    {"maximum_normalized_constraint_violation": 1e-3, "tolerance": .01},
])
def test_witness_requires_full_constraints_and_application_tolerance(case, changes):
    request, result, context = case
    bad = replace(result, load_witness=replace(result.load_witness, **changes))
    assert "invalid_load_witness" in validate_task_load_margin(request, bad, **context)


def test_status_does_not_replace_primal_or_kkt_diagnostics(case):
    request, result, context = case
    stopped = replace(result, load_witness=replace(result.load_witness,
                                                  solver_status="Maximum_Iterations_Exceeded"))
    assert validate_task_load_margin(request, stopped, **context) == ()
    assert "invalid_stationarity_residual" in validate_task_load_margin(
        request, replace(stopped, stationarity_residual=.1), **context)


@pytest.mark.parametrize("scale,reason", [(3., "artificial_load_cap_active"),
                                         (3.1, "artificial_load_cap_active"),
                                         (.99, "load_below_explicit_nominal_witness")])
def test_load_cap_and_dominated_optimum_are_rejected(case, scale, reason):
    request, result, context = case
    report = replace(result, load_witness=replace(result.load_witness, work_scale=scale))
    assert reason in validate_task_load_margin(request, report, **context)


def test_failed_new_result_keeps_previous_until_it_expires(case):
    request, result, context = case
    previous = select_task_load_margin([(request, result)], **context).direction
    failed_request = replace(request, request_id="new", issued_monotonic_seconds=202.)
    failed_result = TaskLoadMarginResult("new", 203., failure_reason="timeout")
    selected = select_task_load_margin([(failed_request, failed_result)], previous=previous, **context)
    assert selected.direction == previous
    assert selected.action == "keep_previous"
    assert selected.direction.result.load_witness.work_scale == 1.2
    stale = select_task_load_margin([], previous=previous, **(context | {"completed_cycles": 121}))
    assert stale.direction is None


def test_out_of_order_completion_selects_newest_source_not_largest_margin(case):
    request, result, context = case
    newer = replace(request, request_id="newer", checkpoint=replace(request.checkpoint, completed_cycles=101),
                    issued_monotonic_seconds=202.)
    newer_result = replace(result, request_id="newer", completed_monotonic_seconds=202.5,
                           load_witness=replace(result.load_witness, work_scale=1.1))
    selected = select_task_load_margin([(newer, newer_result), (request, result)], **context)
    assert selected.direction.request.request_id == "newer"
    assert selected.direction.value((1., 1.)) == pytest.approx(.1)


def test_all_failures_remain_unknown(case):
    request, _, context = case
    report = TaskLoadMarginResult(request.request_id, 201., failure_reason="Infeasible_Problem_Detected")
    selected = select_task_load_margin([(request, report)], **context)
    assert selected.direction is None
    assert report.load_witness is None
    assert not selected.physiological_failure_certified


def test_kkt_envelope_sign_units_and_parameter_dependent_bounds():
    # max lambda with lambda <= 2*p1 - 3*p2 gives d lambda/d x=(20,-.3)
    gradient = load_gradient_from_kkt(objective_parameter_gradient=[0., 0.],
        constraint_parameter_jacobian=[[-2., 0.]], constraint_multipliers=[1.],
        bound_parameter_contribution=[0., 3.], coordinate_scales=[10., .1])
    np.testing.assert_allclose(gradient, [20., -.3])


@pytest.mark.parametrize("changes", [
    {"coordinate_scales": [0., 1.]},
    {"coordinate_scales": [float("nan"), 1.]},
    {"bound_parameter_contribution": [0.]},
    {"constraint_parameter_jacobian": [[1.]]},
    {"constraint_multipliers": [float("inf")]},
])
def test_invalid_kkt_inputs_rejected(changes):
    data = dict(objective_parameter_gradient=[0., 0.], constraint_parameter_jacobian=[[1., 2.]],
                constraint_multipliers=[1.], bound_parameter_contribution=[0., 0.], coordinate_scales=[1., 1.])
    with pytest.raises(ValueError):
        load_gradient_from_kkt(**(data | changes))


@pytest.mark.parametrize("changes", [
    {"trust_radius": (0., .1)}, {"domain_upper": (.9, 2.)},
    {"center": (1.,)}, {"coordinate_names": ("A", "A")},
    {"issued_monotonic_seconds": float("nan")}, {"load_factor_upper_bound": 1.},
])
def test_invalid_request_rejected(case, changes):
    with pytest.raises(ValueError):
        replace(case[0], **changes)
