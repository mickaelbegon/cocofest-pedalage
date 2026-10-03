"""Fixed graph, provenance and coexistence tests for experimental task reserve."""

from dataclasses import asdict, replace
from hashlib import sha256
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from cocofest import CustomObjective
from cocofest.optimization.parametric_fatigue_weights import (
    FATIGUE_WEIGHT_PARAMETER_KEY, ParametricFatigueWeightBinding,
)
from cocofest.optimization.task_reserve import LocalReserveModel, reserve_penalty_expression
from cocofest.optimization.task_reserve_ocp import (
    TASK_RESERVE_PARAMETER_KEY, TaskReserveObjectiveBinding,
    TaskReserveStateCoordinate, task_reserve_parameter_options,
)


@pytest.fixture
def case(tmp_path):
    coordinates = (TaskReserveStateCoordinate("A_Biceps", scale=100.),
                   TaskReserveStateCoordinate("Tau1_Biceps", scale=.1, offset=.05))
    source = tmp_path / "model.json"
    source.write_text('{"model":"synthetic"}')
    digest = sha256(source.read_bytes()).hexdigest()
    context = {"coordinate_layout": [asdict(c) for c in coordinates],
               "nominal_work_j": 6., "formulation": "isokinetic"}
    binding = TaskReserveObjectiveBinding(coordinates, task_context=context,
        model_sha256=digest, target=.2, maximum_age_cycles=20)
    local = LocalReserveModel(center=(.8, .2), gradient=(.4, -.3), intercept=.1,
        trust_radius=(.1, .1), accepted=True, reasons=(), sample_count=5,
        holdout_count=2, rank=3, condition_number=2., training_maximum_error=1e-7,
        holdout_maximum_error=1e-7, empirical_bias_correction=0.,
        task_context_sha256=binding.task_context_sha256, model_sha256=digest)
    return binding, local, source


def program(binding, fatigue=None):
    from bioptim import OptimalControlProgram

    options = task_reserve_parameter_options(binding, fatigue_weight_binding=fatigue)
    ocp = SimpleNamespace(nlp=[SimpleNamespace(update_init=lambda *args: None)], n_phases=1,
                          compiled_solver=object(), **options)
    # Use the real Bioptim merge implementation, not a mock that replaces a
    # parameter list. This catches updates dropping the coexisting objective.
    ocp.update_initial_guess = MethodType(OptimalControlProgram.update_initial_guess, ocp)
    return ocp


def publish(binding, ocp, local, **overrides):
    return binding.update(ocp, local, **{
        "weight": 2., "source_completed_cycles": 20, "completed_cycles": 21,
        "evaluation_coordinates": (.8, .2), **overrides,
    })


@pytest.mark.parametrize("symbol_type", ["SX", "MX"])
def test_same_symbolic_function_changes_value_and_gradient_with_parameters(case, symbol_type):
    import casadi as ca

    binding, local, _ = case
    symbolic = getattr(ca, symbol_type)
    states, parameters = symbolic.sym("x", 2), symbolic.sym("p", binding.values.size)
    controller = SimpleNamespace(
        states={"A_Biceps": SimpleNamespace(cx=states[0]),
                "Tau1_Biceps": SimpleNamespace(cx=states[1])},
        parameters={TASK_RESERVE_PARAMETER_KEY: SimpleNamespace(cx=parameters)},
    )
    expr = CustomObjective.minimize_terminal_task_reserve(controller, binding)
    function = ca.Function("reserve", [states, parameters], [expr, ca.gradient(expr, states)])
    physical = np.array([80., .07])
    value, gradient = function(physical, binding.values)
    assert float(value) == 0.
    np.testing.assert_array_equal(gradient, np.zeros((2, 1)))
    params = np.r_[2., local.parameter_vector()]
    value, gradient = function(physical, params)
    expected = 2. * reserve_penalty_expression([.8, .2], params[1:], dimension=2, target=.2)
    assert float(value) == pytest.approx(expected)
    finite_difference = []
    for index in range(2):
        dx = np.eye(2)[index] * 1e-6
        plus, _ = function(physical + dx, params)
        minus, _ = function(physical - dx, params)
        finite_difference.append(float(plus - minus) / 2e-6)
    np.testing.assert_allclose(np.asarray(gradient).ravel(), finite_difference, rtol=1e-6)
    altered = params.copy()
    altered[-2:] *= -1.
    changed, changed_gradient = function(physical + [.5, .001], altered)
    original, _ = function(physical + [.5, .001], params)
    assert float(changed) != pytest.approx(float(original))
    np.testing.assert_allclose(np.sign(changed_gradient), -np.sign(gradient))


def test_updates_coexist_with_fatigue_and_preserve_real_bioptim_initial_guesses(case):
    binding, local, _ = case
    fatigue = ParametricFatigueWeightBinding((1., 2.))
    ocp = program(binding, fatigue)
    graph, solver = ocp.nlp[0], ocp.compiled_solver
    signature = binding.graph_signature_sha256
    assert set(ocp.parameters.keys()) == {TASK_RESERVE_PARAMETER_KEY, FATIGUE_WEIGHT_PARAMETER_KEY}
    fatigue_before = ocp.parameter_bounds[FATIGUE_WEIGHT_PARAMETER_KEY].min.copy()
    result = publish(binding, ocp, local)
    assert result["weight"] == 2.
    assert result["requires_terminal_trust_validation"] is True
    assert result["objective_graph_rebuild_required"] is False
    np.testing.assert_array_equal(ocp.parameter_bounds[FATIGUE_WEIGHT_PARAMETER_KEY].min, fatigue_before)
    reserve_before = ocp.parameter_bounds[TASK_RESERVE_PARAMETER_KEY].min.copy()
    fatigue.update(ocp, (.5, 3.))
    np.testing.assert_array_equal(ocp.parameter_bounds[TASK_RESERVE_PARAMETER_KEY].min, reserve_before)
    np.testing.assert_array_equal(ocp.parameter_init[TASK_RESERVE_PARAMETER_KEY].init, reserve_before)
    np.testing.assert_array_equal(ocp.parameter_init[FATIGUE_WEIGHT_PARAMETER_KEY].init, [[.5], [3.]])
    publish(binding, ocp, replace(local, intercept=.15))
    assert ocp.nlp[0] is graph and ocp.compiled_solver is solver
    assert binding.graph_signature_sha256 == signature
    bound = ocp.parameter_bounds[TASK_RESERVE_PARAMETER_KEY]
    np.testing.assert_array_equal(bound.min, bound.max)
    np.testing.assert_array_equal(bound.min, ocp.parameter_init[TASK_RESERVE_PARAMETER_KEY].init)
    binding.deactivate(ocp)
    assert binding.values[0] == 0.
    np.testing.assert_array_equal(ocp.parameter_bounds[FATIGUE_WEIGHT_PARAMETER_KEY].min, [[.5], [3.]])


@pytest.mark.parametrize("change, message", [
    ({"accepted": False, "reasons": ("rank_deficient",)}, "rejected"),
    ({"task_context_sha256": "different"}, "different task"),
    ({"model_sha256": "different"}, "different task"),
    ({"gradient": (.1,)}, "dimensions"),
    ({"gradient": (.1, float("nan"))}, "nonfinite"),
    ({"trust_radius": (-1., .1)}, "trust radii"),
    ({"rank": 1}, "full-rank"),
    ({"holdout_count": 0}, "holdout"),
    ({"holdout_maximum_error": float("nan")}, "diagnostics"),
])
def test_bad_local_models_do_not_change_parameters(case, change, message):
    binding, local, _ = case
    ocp = program(binding)
    before = ocp.parameter_bounds[TASK_RESERVE_PARAMETER_KEY].min.copy()
    with pytest.raises(ValueError, match=message):
        publish(binding, ocp, replace(local, **change))
    np.testing.assert_array_equal(ocp.parameter_bounds[TASK_RESERVE_PARAMETER_KEY].min, before)
    assert binding.update_count == 0


@pytest.mark.parametrize("arguments, message", [
    ({"completed_cycles": 41}, "stale"),
    ({"completed_cycles": 19}, "future"),
    ({"evaluation_coordinates": (.95, .2)}, "trust box"),
    ({"evaluation_coordinates": (.8,)}, "dimension"),
    ({"weight": 0.}, "positive weight"),
    ({"weight": float("nan")}, "positive weight"),
])
def test_stale_outside_domain_or_invalid_activation_is_refused(case, arguments, message):
    binding, local, _ = case
    ocp = program(binding)
    with pytest.raises(ValueError, match=message):
        publish(binding, ocp, local, **arguments)
    assert binding.values[0] == 0.


def test_terminal_domain_is_checked_separately_from_publication(case):
    binding, local, _ = case
    ocp = program(binding)
    publish(binding, ocp, local)
    assert binding.validate_terminal_point((.82, .2), completed_cycles=22)["terminal_trust_validated"]
    with pytest.raises(ValueError, match="trust box"):
        binding.validate_terminal_point((.91, .2), completed_cycles=22)
    with pytest.raises(ValueError, match="stale"):
        binding.validate_terminal_point((.82, .2), completed_cycles=41)


def test_build_context_verifies_actual_model_bytes_and_physical_task(case):
    binding, _, source = case
    binding.validate_build_context(model_path=source, nominal_work_j=6., formulation="isokinetic")
    with pytest.raises(ValueError, match="nominal_work_j"):
        binding.validate_build_context(model_path=source, nominal_work_j=6.1)
    with pytest.raises(ValueError, match="required"):
        binding.validate_build_context(model_path=None)
    source.write_text("changed model")
    with pytest.raises(ValueError, match="model digest"):
        binding.validate_build_context(model_path=source)


def test_failed_initial_guess_update_does_not_mutate_active_policy_or_bounds(case):
    binding, local, _ = case
    ocp = program(binding)
    publish(binding, ocp, local)
    before = binding.values.copy()
    def fail(**kwargs):
        raise RuntimeError("rejected numerical initial guess")
    ocp.update_initial_guess = fail
    with pytest.raises(RuntimeError, match="rejected numerical"):
        publish(binding, ocp, replace(local, intercept=.05))
    np.testing.assert_array_equal(binding.values, before)
    np.testing.assert_array_equal(ocp.parameter_bounds[TASK_RESERVE_PARAMETER_KEY].min[:, 0], before)


def test_rebuilt_nlp_and_unsupported_parameter_merge_are_refused(case):
    binding, local, _ = case
    ocp = program(binding)
    publish(binding, ocp, local)
    ocp.nlp = [object()]
    with pytest.raises(RuntimeError, match="rebuilt"):
        publish(binding, ocp, local)
    with pytest.raises(TypeError, match="Only ParametricFatigue"):
        task_reserve_parameter_options(binding, fatigue_weight_binding=object())


def test_objective_registration_is_terminal_nonquadratic_and_keeps_fatigue(case):
    from bioptim import Node
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe as mhe

    binding, _, _ = case
    fatigue = ParametricFatigueWeightBinding((1., 1.))
    objectives = mhe.set_objective_functions(
        SimpleNamespace(muscles_dynamics_model=[object(), object()]),
        False, True, False, [0., 1., 0.], 0., task_reserve_binding=binding,
        fatigue_weight_binding=fatigue, terminal_wheel_regularization_weight=0.,
    )
    reserve, regularization = objectives[0][0], objectives[0][1]
    assert reserve.custom_function is CustomObjective.minimize_terminal_task_reserve
    assert reserve.node == Node.END and reserve.quadratic is False
    np.testing.assert_array_equal(reserve.weight, [1.])
    assert regularization.custom_function is CustomObjective.minimize_parameterized_overall_muscle_fatigue


def test_coordinate_normalization_is_pinned_in_task_context(case):
    binding, _, _ = case
    changed = dict(binding.task_context, coordinate_layout=[])
    with pytest.raises(ValueError, match="coordinate_layout"):
        TaskReserveObjectiveBinding(binding.coordinates, task_context=changed,
            model_sha256=binding.model_sha256, target=.2)


def test_fixed_graph_configuration_has_its_own_codegen_signature(case):
    binding, _, _ = case
    other = TaskReserveObjectiveBinding(binding.coordinates, task_context=binding.task_context,
        model_sha256=binding.model_sha256, target=.3)
    assert other.graph_signature_sha256 != binding.graph_signature_sha256
