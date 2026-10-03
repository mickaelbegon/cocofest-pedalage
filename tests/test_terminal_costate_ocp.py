"""Numerical contract for an opt-in terminal remaining-endurance costate."""

from dataclasses import asdict
from hashlib import sha256
from types import MethodType, SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from cocofest.optimization.parametric_fatigue_weights import (
    FATIGUE_WEIGHT_PARAMETER_KEY, ParametricFatigueWeightBinding,
)
from cocofest.optimization.task_reserve_ocp import (
    TASK_RESERVE_PARAMETER_KEY, TaskReserveStateCoordinate, task_reserve_parameter_options,
)
from cocofest.optimization.terminal_costate_ocp import TerminalCostateObjectiveBinding
from cocofest.simulation.independent_arms_process import (
    _driver_arguments, _guard_untrusted_terminal_value, _observe_local_terminal_value,
)


def _binding(tmp_path):
    source = tmp_path / "model.json"
    source.write_text('{"model":"synthetic"}')
    coordinates = (TaskReserveStateCoordinate("A_right", scale=100.),
                   TaskReserveStateCoordinate("A_left", scale=200.))
    binding = TerminalCostateObjectiveBinding(
        coordinates, task_context={"coordinate_layout": [asdict(c) for c in coordinates]},
        model_sha256=sha256(source.read_bytes()).hexdigest(), maximum_age_cycles=10,
    )
    return binding, source


def _program(binding, fatigue=None):
    from bioptim import OptimalControlProgram

    options = task_reserve_parameter_options(binding, fatigue_weight_binding=fatigue)
    program = SimpleNamespace(nlp=[SimpleNamespace(update_init=lambda *args: None)], n_phases=1,
                              compiled_solver=object(), **options)
    program.update_initial_guess = MethodType(OptimalControlProgram.update_initial_guess, program)
    return program


def _function(binding):
    state = ca.SX.sym("x", 2)
    parameter = ca.SX.sym("p", binding.values.size)
    controller = SimpleNamespace(
        states={"A_right": SimpleNamespace(cx=state[0]),
                "A_left": SimpleNamespace(cx=state[1])},
        parameters={TASK_RESERVE_PARAMETER_KEY: SimpleNamespace(cx=parameter)},
    )
    cost = binding.objective(controller)
    return ca.Function("remaining_cycles_cost", [state, parameter], [cost, ca.gradient(cost, state)])


def test_inactive_default_and_remaining_endurance_sign(tmp_path):
    binding, _ = _binding(tmp_path)
    function = _function(binding)
    state = np.array([80., 120.])
    cost, gradient = function(state, binding.values)
    assert float(cost) == 0.
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros((2, 1)))

    # One extra normalized unit of capacity raises V by 4 cycles; the
    # minimised cost therefore decreases by 4. Physical dJ/dA = -4/100.
    program = _program(binding)
    binding.update_from_remaining_cycles_value(
        program, remaining_cycles=20., center=[.8, .6], gradient=[4., -2.],
        trust_radius=[.2, .2], evaluation_coordinates=[.8, .6],
        source_completed_cycles=5, completed_cycles=5,
    )
    cost, gradient = function(state, binding.values)
    assert float(cost) == pytest.approx(-20.)
    np.testing.assert_allclose(np.asarray(gradient).ravel(), [-.04, .01])
    higher_right = float(function([90., 120.], binding.values)[0])
    assert higher_right < float(cost)


def test_numeric_update_preserves_graph_solver_and_fatigue_secondary_parameter(tmp_path):
    binding, source = _binding(tmp_path)
    fatigue = ParametricFatigueWeightBinding((1., 3.))
    program = _program(binding, fatigue)
    binding.validate_build_context(model_path=source)
    graph, solver = program.nlp[0], program.compiled_solver
    signature = binding.graph_signature_sha256
    fatigue_before = program.parameter_bounds[FATIGUE_WEIGHT_PARAMETER_KEY].min.copy()
    assert set(program.parameters.keys()) == {TASK_RESERVE_PARAMETER_KEY, FATIGUE_WEIGHT_PARAMETER_KEY}

    receipt = binding.update_from_remaining_cycles_value(
        program, remaining_cycles=20., center=[.8, .6], gradient=[4., -2.],
        trust_radius=[.2, .2], evaluation_coordinates=[.8, .6],
        source_completed_cycles=5, completed_cycles=6,
    )
    assert receipt["gradient_scientifically_validated"] is False
    assert receipt["objective_graph_rebuild_required"] is False
    np.testing.assert_array_equal(program.parameter_bounds[FATIGUE_WEIGHT_PARAMETER_KEY].min, fatigue_before)
    np.testing.assert_allclose(program.parameter_bounds[TASK_RESERVE_PARAMETER_KEY].min.ravel(),
                               [1., 20., .8, .6, 4., -2.])
    assert binding.validate_terminal_point([.85, .6], completed_cycles=6)["terminal_trust_validated"]
    assert not binding.validate_terminal_point([1.05, .6], completed_cycles=6)["terminal_trust_validated"]

    binding.deactivate(program)
    assert binding.values[0] == 0.
    assert not binding.validate_terminal_point([.8, .6], completed_cycles=6)["active"]
    np.testing.assert_array_equal(program.parameter_bounds[FATIGUE_WEIGHT_PARAMETER_KEY].min, fatigue_before)
    assert program.nlp[0] is graph and program.compiled_solver is solver
    assert binding.graph_signature_sha256 == signature


def test_rejects_stale_or_untrusted_gradient_before_parameter_mutation(tmp_path):
    binding, _ = _binding(tmp_path)
    program = _program(binding)
    kwargs = dict(remaining_cycles=20., center=[.8, .6], gradient=[4., -2.],
                  trust_radius=[.1, .1], evaluation_coordinates=[.8, .6],
                  source_completed_cycles=5, completed_cycles=6)
    with pytest.raises(ValueError, match="outside"):
        binding.update_from_remaining_cycles_value(program, **{**kwargs, "evaluation_coordinates": [1., .6]})
    with pytest.raises(ValueError, match="stale"):
        binding.update_from_remaining_cycles_value(program, **{**kwargs, "completed_cycles": 16})
    with pytest.raises(ValueError, match="finite"):
        binding.update_from_remaining_cycles_value(program, **{**kwargs, "gradient": [np.nan, -2.]})
    assert binding.update_count == 0
    np.testing.assert_array_equal(program.parameter_bounds[TASK_RESERVE_PARAMETER_KEY].min,
                                  np.zeros((binding.values.size, 1)))


def test_costate_post_solve_audit_deactivates_untrusted_next_window(tmp_path):
    binding, _ = _binding(tmp_path)
    program = _program(binding)
    program.task_reserve_binding = binding
    states = {"A_right": np.array([[80., 85.]]), "A_left": np.array([[120., 120.]])}
    key, inactive = _observe_local_terminal_value(program, states, completed_cycles=6)
    assert key == "terminal_costate" and inactive == {"active": False, "terminal_trust_validated": False}
    assert _guard_untrusted_terminal_value(program, inactive, completed_cycles=6) is None
    assert binding.update_count == 0

    binding.update_from_remaining_cycles_value(
        program, remaining_cycles=20., center=[.8, .6], gradient=[4., -2.],
        trust_radius=[.1, .1], evaluation_coordinates=[.8, .6],
        source_completed_cycles=5, completed_cycles=5,
    )
    graph, solver = program.nlp[0], program.compiled_solver
    key, trusted = _observe_local_terminal_value(program, states, completed_cycles=6)
    assert key == "terminal_costate" and trusted["terminal_trust_validated"]
    assert trusted["predicted_remaining_cycles"] == pytest.approx(20.2)
    assert _guard_untrusted_terminal_value(program, trusted, completed_cycles=6) is None

    states["A_right"][0, -1] = 95.
    key, untrusted = _observe_local_terminal_value(program, states, completed_cycles=6)
    assert key == "terminal_costate" and untrusted["active"]
    assert not untrusted["terminal_trust_validated"]
    event = _guard_untrusted_terminal_value(program, untrusted, completed_cycles=6)
    assert event["mode"] == "terminal_costate"
    assert event["kind"] == "terminal_trust_guard"
    assert event["compiled_nlp_reused"]
    assert binding.values[0] == 0.
    assert program.nlp[0] is graph and program.compiled_solver is solver
    assert not _observe_local_terminal_value(program, states, completed_cycles=7)[1]["active"]


def test_pace_rt_dispatch_and_fresh_replacement_are_unchanged():
    from cocofest.optimization.pace_rt_ocp import PaceRtObjectiveBinding

    coordinates = (TaskReserveStateCoordinate("A_right", scale=100.),
                   TaskReserveStateCoordinate("A_left", scale=200.))
    binding = PaceRtObjectiveBinding(
        coordinates, task_context={"coordinate_layout": [asdict(c) for c in coordinates]},
        model_sha256="a" * 64,
    )
    program = _program(binding)
    program.task_reserve_binding = binding
    fit = {"accepted": True, "constant": .4,
           "gradient": [[-1., 0., 0.], [.5, 0., 0.]],
           "reference_state": [[80., 1., 1.], [120., 1., 1.]],
           "state_scales": [[100., 1., 1.], [200., 1., 1.]],
           "trust_bounds": [-.01, .01]}
    binding.update_from_rollout(program, local_fit=fit, source_completed_cycles=5,
                                completed_cycles=5, proximal_weight=.03)
    states = {"A_right": np.array([[80., 90.]]), "A_left": np.array([[120., 120.]])}
    key, audit = _observe_local_terminal_value(program, states, completed_cycles=6)
    assert key == "pace_rt_terminal" and not audit["terminal_trust_validated"]
    assert _guard_untrusted_terminal_value(program, audit, completed_cycles=6,
                                           incoming_pace_rt=True) is None
    assert binding.values[0] == 1.
    event = _guard_untrusted_terminal_value(program, audit, completed_cycles=6)
    assert event["mode"] == "pace_rt" and binding.values[0] == 0.


def test_costate_runner_config_is_explicit_and_inactive_until_numeric_update():
    payload = {"cycles": 2, "right_equivalent_mean_torque_nm": .96,
               "right_driver_arguments": ["--retry-failed-rho-without-advance"],
               "experimental_terminal_costate": {"maximum_age_cycles": 8}}
    args = _driver_arguments(payload, "right")
    assert args.experimental_terminal_costate_config == {"maximum_age_cycles": 8}
    assert args.experimental_pace_rt_config is None
    with pytest.raises(ValueError, match="nonnegative maximum_age_cycles"):
        _driver_arguments({**payload, "experimental_terminal_costate": {"maximum_age_cycles": -1}}, "right")
    with pytest.raises(ValueError, match="same numerical parameter channel"):
        _driver_arguments({**payload, "experimental_pace_vr": {
            "application_mode": "terminal_reserve_target_experimental"}}, "right")
