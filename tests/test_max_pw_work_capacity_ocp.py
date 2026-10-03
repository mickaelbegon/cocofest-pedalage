"""Numerical contract for the experimental terminal max-PW work objective."""

from types import MethodType, SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from cocofest import CustomObjective
from cocofest.optimization.adaptive_moment_rollout import (
    DingPulseWidthParameters,
    MomentTrackingInterval,
)
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.max_pw_work_capacity import (
    MaxPwWorkCapacityLayout,
    build_max_pw_work_capacity_function,
    max_pw_work_domain_lower_bounds,
    pack_max_pw_work_profile,
)
from cocofest.optimization.max_pw_work_capacity_ocp import (
    ACTIVATION_KEY,
    LINEAR_GRADIENT_KEY,
    LINEAR_REFERENCE_KEY,
    MaxPwWorkCapacityBinding,
    max_pw_work_boundary,
    max_pw_work_parameter_options,
)
from cocofest.optimization.parametric_fatigue_weights import (
    FATIGUE_WEIGHT_PARAMETER_KEY,
    ParametricFatigueWeightBinding,
)


def _case(gradient_filter="full"):
    fatigue = DingFatigueParameters(
        a_rest=4920., tau1_rest=.060601, km_rest=.137,
        alpha_a=-.04, alpha_tau1=2.1e-6, alpha_km=1.9e-6,
        tau_fat=127.,
    )
    muscles = tuple(
        DingPulseWidthParameters(
            fatigue, tauc=.011, tau2=.001, pd0=.000131405,
            pdt=.000194138, pulse_width_max=.0006,
        )
        for _ in range(2)
    )
    layout = MaxPwWorkCapacityLayout(2, 30, 2)
    intervals = tuple(
        MomentTrackingInterval(
            1 / 30, (1.06, 1.12), (1., .9), (-.04, -.02), (0., 0.),
        )
        for _ in range(30)
    )
    profile = pack_max_pw_work_profile(
        intervals, layout, angular_velocity_rad_s=-2 * np.pi,
    )
    function = build_max_pw_work_capacity_function(muscles=muscles, layout=layout)
    state = np.array([.16, 20., 4400., .065, .15,
                      .18, 30., 4500., .07, .14])
    binding = MaxPwWorkCapacityBinding(
        function=function, profile=profile, reference_work_j=8., weight=.3,
        layout=layout, muscle_names=("Biceps", "Triceps"), gradient_filter=gradient_filter,
    )
    return binding, layout, state


def _program(binding, fatigue=None):
    from bioptim import OptimalControlProgram

    options = max_pw_work_parameter_options(
        binding, fatigue_weight_binding=fatigue,
    )
    nmpc = SimpleNamespace(
        nlp=[SimpleNamespace(update_init=lambda *args: None)],
        n_phases=1, compiled_solver=object(), **options,
    )
    nmpc.update_initial_guess = MethodType(
        OptimalControlProgram.update_initial_guess, nmpc,
    )
    return nmpc


@pytest.mark.parametrize("bad_profile,bad_reference,bad_weight", [
    ([], 8., .3),
    ([np.nan], 8., .3),
    ([1.], 0., .3),
    ([1.], 8., 0.),
    ([1.], 8., np.inf),
])
def test_binding_rejects_invalid_parameter_values(bad_profile, bad_reference, bad_weight):
    with pytest.raises(ValueError):
        MaxPwWorkCapacityBinding(
            function=object(), profile=bad_profile,
            reference_work_j=bad_reference, weight=bad_weight,
        )


def test_gradient_filter_is_strict_and_full_preserves_exact_gradient():
    binding, _, physical = _case()
    nmpc = _program(binding)
    with pytest.raises(ValueError, match="gradient_filter"):
        _case("unvalidated_filter")
    direct = np.asarray(binding._gradient_function(physical), float).reshape(-1)
    binding.update(nmpc, terminal_state=physical)
    np.testing.assert_allclose(binding.gradient, direct)
    assert binding.summary()["gradient_filter"] == "full"
    assert binding.summary()["last_domain_audit"]["gradient_l2_raw"] == pytest.approx(
        np.linalg.norm(direct))


def test_slow_filter_zeroes_fast_sensitivities_and_reports_exact_audit():
    binding, _, physical = _case("slow_fatigue_states")
    nmpc = _program(binding)
    full = np.asarray(binding._gradient_function(physical), float).reshape(-1)
    audit = binding.audit(physical)
    assert audit["valid"]
    assert set(audit["work_by_muscle_j"]) == {"Biceps", "Triceps"}
    assert sum(audit["work_by_muscle_j"].values()) == pytest.approx(audit["capacity_j"])
    for index, name in enumerate(binding.muscle_names):
        part = full[5*index:5*(index+1)]
        item = audit["gradient_by_muscle"][name]
        assert item["fast_cn_force_l2_raw"] == pytest.approx(np.linalg.norm(part[:2]))
        assert item["slow_a_tau1_km_l2_raw"] == pytest.approx(np.linalg.norm(part[2:]))
    binding.update(nmpc, terminal_state=physical)
    np.testing.assert_array_equal(binding.gradient.reshape(-1, 5)[:, :2], np.zeros((2, 2)))
    np.testing.assert_allclose(binding.gradient.reshape(-1, 5)[:, 2:], full.reshape(-1, 5)[:, 2:])
    np.testing.assert_allclose(nmpc.parameter_bounds[LINEAR_GRADIENT_KEY].min.ravel(), binding.gradient)
    assert binding.summary()["gradient_filter"] == "slow_fatigue_states"
    assert audit["filtered_gradient_l2_raw"] == pytest.approx(np.linalg.norm(binding.gradient))


def test_taylor_audit_uses_previous_injected_gradient_and_exact_new_work():
    binding, _, physical = _case("slow_fatigue_states")
    nmpc = _program(binding)
    assert binding.audit(physical)["taylor_from_previous_checkpoint"] is None
    binding.update(nmpc, terminal_state=physical)
    previous_work, previous_gradient = binding.reference_capacity_j, binding.gradient.copy()
    moved = physical.copy()
    moved[0] += .001
    moved[2] -= 1.
    moved[7] -= 2.
    audit = binding.audit(moved)
    assert audit["valid"]
    taylor = audit["taylor_from_previous_checkpoint"]
    predicted = previous_work + float(np.dot(previous_gradient, moved-physical))
    assert taylor["prediction_j"] == pytest.approx(predicted)
    assert taylor["error_j"] == pytest.approx(audit["capacity_j"]-predicted)
    contributions = taylor["directional_contributions_by_muscle_j"]
    assert contributions["Biceps"]["fast_cn_force_j"] == 0.
    assert sum(sum(row.values()) for row in contributions.values()) == pytest.approx(predicted-previous_work)
    binding.update(nmpc, terminal_state=moved)
    assert binding.summary()["last_domain_audit"]["taylor_from_previous_checkpoint"] == taylor


def test_independent_arm_driver_accepts_explicit_gradient_ablation_option():
    from cocofest.simulation.independent_arms_process import _driver_arguments

    payload = {"cycles": 2, "right_equivalent_mean_torque_nm": .96,
               "experimental_max_pw_work_weight": 1.,
               "right_driver_arguments": ["--retry-failed-rho-without-advance"],
               "experimental_max_pw_work_gradient_filter": "slow_fatigue_states"}
    args = _driver_arguments(payload, "right")
    assert args.experimental_max_pw_work_gradient_filter == "slow_fatigue_states"
    del payload["experimental_max_pw_work_gradient_filter"]
    assert _driver_arguments(payload, "right").experimental_max_pw_work_gradient_filter == "full"


def test_fixed_profile_starts_inactive_and_activation_has_fixed_numerical_bounds():
    binding, layout, _ = _case()
    options = binding.parameter_options(use_sx=True)
    assert set(options["parameters"].keys()) == {
        ACTIVATION_KEY, LINEAR_REFERENCE_KEY, LINEAR_GRADIENT_KEY,
    }
    assert binding.profile.size == layout.parameter_size
    assert not binding.profile.flags.writeable
    np.testing.assert_array_equal(options["parameter_bounds"][ACTIVATION_KEY].min, [[0.]])
    np.testing.assert_array_equal(options["parameter_bounds"][ACTIVATION_KEY].max, [[0.]])
    np.testing.assert_array_equal(options["parameter_init"][ACTIVATION_KEY].init, [[0.]])
    np.testing.assert_array_equal(
        options["parameter_init"][LINEAR_REFERENCE_KEY].init, np.zeros((10, 1)),
    )
    np.testing.assert_array_equal(
        options["parameter_init"][LINEAR_GRADIENT_KEY].init, np.zeros((10, 1)),
    )


def test_terminal_callback_preserves_muscle_and_ding_state_order():
    binding, _, physical = _case()
    state = ca.SX.sym("terminal", 10)
    activation = ca.SX.sym("activation")
    reference = ca.SX.sym("reference", 10)
    gradient = ca.SX.sym("gradient", 10)
    muscle_names = ("Biceps", "Triceps")
    states = {
        f"{key}_{name}": SimpleNamespace(cx=state[5 * m + k])
        for m, name in enumerate(muscle_names)
        for k, key in enumerate(("Cn", "F", "A", "Tau1", "Km"))
    }
    controller = SimpleNamespace(
        states=states,
        model=SimpleNamespace(muscles_dynamics_model=[
            SimpleNamespace(muscle_name=name) for name in muscle_names
        ]),
        parameters={
            ACTIVATION_KEY: SimpleNamespace(cx=ca.vertcat(activation)),
            LINEAR_REFERENCE_KEY: SimpleNamespace(cx=reference),
            LINEAR_GRADIENT_KEY: SimpleNamespace(cx=gradient),
        },
    )
    expression = CustomObjective.minimize_terminal_max_pw_work_capacity(
        controller, binding,
    )
    objective = ca.Function(
        "terminal_max_pw_cost", [state, activation, reference, gradient],
        [expression, ca.gradient(expression, state)],
    )
    zero = np.zeros(10)
    inactive_value, inactive_gradient = objective(physical, 0., zero, zero)
    assert float(inactive_value) == 0.
    np.testing.assert_array_equal(np.asarray(inactive_gradient), np.zeros((10, 1)))

    direct_gradient = ca.Function(
        "direct_max_pw_gradient", [state],
        [ca.gradient(binding.function(state, binding.profile)[0], state)],
    )
    direct = np.asarray(direct_gradient(physical)).ravel()
    active_value, active_gradient = objective(physical, 1., physical, direct)
    # The constant W+(x_ref) is deliberately omitted: it does not affect the
    # NLP minimizer.  At the expansion point the affine term is therefore 0.
    assert float(active_value) == pytest.approx(0.)
    assert float(active_gradient[2]) < 0.
    assert float(active_gradient[7]) < 0.
    np.testing.assert_allclose(
        np.asarray(active_gradient).ravel(),
        -binding.weight / binding.reference_work_j
        * direct,
        rtol=1e-11, atol=1e-11,
    )


def test_domain_margins_are_exposed_for_an_independent_terminal_audit():
    binding, layout, physical = _case()
    output = binding.function(physical, binding.profile)
    margins = np.asarray(output[3]).ravel()
    assert margins.size == layout.domain_margin_size
    assert np.all(margins >= max_pw_work_domain_lower_bounds(layout))
    invalid = physical.copy()
    invalid[2] = -1.
    invalid_margins = np.asarray(binding.function(invalid, binding.profile)[3]).ravel()
    assert np.any(invalid_margins < max_pw_work_domain_lower_bounds(layout))


def test_binding_refuses_a_different_nlp_identity():
    binding, _, _ = _case()
    first = SimpleNamespace(nlp=[object()])
    binding.attach(first)
    binding.attach(first)
    with pytest.raises(RuntimeError, match="rebuilt"):
        binding.attach(SimpleNamespace(nlp=[object()]))
    with pytest.raises(ValueError, match="one RHO phase"):
        MaxPwWorkCapacityBinding(
            function=binding.function, profile=binding.profile,
            reference_work_j=8., weight=.3,
        ).attach(SimpleNamespace(nlp=[object(), object()]))


def test_muscle_name_permutation_is_rejected_before_objective_assembly():
    binding, _, _ = _case()
    binding.validate_muscle_names(("Biceps", "Triceps"))
    with pytest.raises(ValueError, match="muscle order"):
        binding.validate_muscle_names(("Triceps", "Biceps"))
    with pytest.raises(ValueError, match="muscle order"):
        binding.validate_muscle_names(("Biceps", "Other"))
    swapped_controller = SimpleNamespace(
        model=SimpleNamespace(muscles_dynamics_model=[
            SimpleNamespace(muscle_name="Triceps"),
            SimpleNamespace(muscle_name="Biceps"),
        ]),
        states={}, parameters={},
    )
    with pytest.raises(ValueError, match="muscle order"):
        CustomObjective.minimize_terminal_max_pw_work_capacity(
            swapped_controller, binding,
        )


def test_activation_and_fatigue_updates_keep_one_parameter_graph():
    binding, _, physical = _case()
    fatigue = ParametricFatigueWeightBinding((1., 2.))
    nmpc = _program(binding, fatigue)
    graph, solver = nmpc.nlp[0], nmpc.compiled_solver
    assert set(nmpc.parameters.keys()) == {
        ACTIVATION_KEY, LINEAR_REFERENCE_KEY, LINEAR_GRADIENT_KEY,
        FATIGUE_WEIGHT_PARAMETER_KEY,
    }
    fatigue_before = nmpc.parameter_bounds[FATIGUE_WEIGHT_PARAMETER_KEY].min.copy()
    profile_before = binding.profile.copy()
    outcome = binding.update(nmpc, terminal_state=physical)
    assert outcome["activation"] == 1.
    assert outcome["objective_graph_rebuild_required"] is False
    np.testing.assert_array_equal(
        nmpc.parameter_bounds[FATIGUE_WEIGHT_PARAMETER_KEY].min,
        fatigue_before,
    )
    np.testing.assert_array_equal(binding.profile, profile_before)
    np.testing.assert_array_equal(nmpc.parameter_bounds[ACTIVATION_KEY].min, [[1.]])
    np.testing.assert_array_equal(nmpc.parameter_init[ACTIVATION_KEY].init, [[1.]])
    np.testing.assert_allclose(
        nmpc.parameter_bounds[LINEAR_REFERENCE_KEY].min.ravel(), physical,
    )
    np.testing.assert_allclose(
        nmpc.parameter_bounds[LINEAR_GRADIENT_KEY].min.ravel(), binding.gradient,
    )

    fatigue.update(nmpc, (.5, 3.))
    np.testing.assert_array_equal(binding.profile, profile_before)
    np.testing.assert_array_equal(nmpc.parameter_bounds[ACTIVATION_KEY].min, [[1.]])
    binding.deactivate(nmpc)
    np.testing.assert_array_equal(nmpc.parameter_bounds[ACTIVATION_KEY].min, [[0.]])
    np.testing.assert_array_equal(nmpc.parameter_bounds[FATIGUE_WEIGHT_PARAMETER_KEY].min, [[.5], [3.]])
    assert nmpc.nlp[0] is graph and nmpc.compiled_solver is solver
    assert binding.update_count == 2


def test_domain_failure_refuses_activation_then_requests_hold_if_active():
    binding, _, physical = _case()
    nmpc = _program(binding)
    invalid = physical.copy()
    invalid[2] = -1.
    assert not binding.audit(invalid)["valid"]
    with pytest.raises(ValueError, match="invalid rollout domain"):
        binding.update(nmpc, terminal_state=invalid)
    np.testing.assert_array_equal(nmpc.parameter_bounds[ACTIVATION_KEY].min, [[0.]])
    assert binding.update_count == 0

    binding.update(nmpc, terminal_state=physical)
    status = binding.validate_terminal_point(invalid)
    assert status["active"] and status["hold_required"]
    assert not status["valid"]
    assert status["physiological_feasibility_certified"] is False


def test_failed_initial_guess_write_does_not_mutate_active_parameters():
    binding, _, physical = _case()
    nmpc = _program(binding)
    binding.update(nmpc, terminal_state=physical)
    activation_before = nmpc.parameter_bounds[ACTIVATION_KEY].min.copy()
    profile_before = binding.profile.copy()
    reference_before = binding.reference_state.copy()
    gradient_before = binding.gradient.copy()
    capacity_before = binding.reference_capacity_j

    def reject(**_):
        raise RuntimeError("numerical update rejected")

    nmpc.update_initial_guess = reject
    with pytest.raises(RuntimeError, match="numerical update rejected"):
        binding.deactivate(nmpc)
    changed = physical.copy()
    changed[2] -= 1.
    with pytest.raises(RuntimeError, match="numerical update rejected"):
        binding.update(nmpc, terminal_state=changed)
    np.testing.assert_array_equal(nmpc.parameter_bounds[ACTIVATION_KEY].min, activation_before)
    np.testing.assert_array_equal(binding.profile, profile_before)
    np.testing.assert_array_equal(binding.reference_state, reference_before)
    np.testing.assert_array_equal(binding.gradient, gradient_before)
    assert binding.reference_capacity_j == capacity_before
    assert binding.activation == 1.
    assert binding.update_count == 1


def test_changed_fixed_profile_requires_graph_rebuild_before_any_parameter_write():
    binding, layout, physical = _case()
    nmpc = _program(binding)
    changed = binding.profile.copy()
    changed[layout.slices()["power_start"].start] *= 1.01
    with pytest.raises(ValueError, match="graph rebuild"):
        binding.update(nmpc, terminal_state=physical, profile=changed)
    np.testing.assert_array_equal(nmpc.parameter_bounds[ACTIVATION_KEY].min, [[0.]])
    assert binding.update_count == 0


def test_boundary_activates_only_after_certified_and_valid_cycle():
    binding, _, physical = _case()
    nmpc = _program(binding)
    nmpc.max_pw_work_binding = binding

    def solution(state):
        keys = ("Cn", "F", "A", "Tau1", "Km")
        values = {
            f"{key}_{name}": np.array([0., state[5 * muscle_index + key_index]])
            for muscle_index, name in enumerate(binding.muscle_names)
            for key_index, key in enumerate(keys)
        }
        return SimpleNamespace(decision_states=lambda **_: values)

    candidate = solution(physical)
    refused = max_pw_work_boundary(nmpc, candidate, certified=False)
    assert not refused["accepted"]
    assert binding.activation == 0.
    accepted = max_pw_work_boundary(nmpc, candidate, certified=True)
    assert accepted["accepted"] and accepted["activation_next"] == 1.
    assert binding.update_count == 1

    invalid = physical.copy()
    invalid[2] = -1.
    held = max_pw_work_boundary(nmpc, solution(invalid), certified=True)
    assert not held["accepted"] and held["hold_required"]
    assert held["reason"] == "invalid_capacity_domain"
    assert binding.update_count == 1


def test_pretransfer_guard_preserves_physical_boundary_on_invalid_proxy(monkeypatch):
    from cocofest.optimization.fes_nmpc_multibody import FesNmpcMsk
    from examples.fes_multibody.cycling.cycling_pulse_width_mhe import MyCyclicNMPC

    binding, _, physical = _case()
    nmpc = object.__new__(MyCyclicNMPC)
    nmpc.max_pw_work_binding = binding
    calls = []

    def super_advance(_, solution, steps=0, **options):
        calls.append((solution, steps, options))
        return "advanced"

    monkeypatch.setattr(FesNmpcMsk, "advance_window", super_advance)

    def solution(state):
        names = binding.muscle_names
        keys = ("Cn", "F", "A", "Tau1", "Km")
        values = {
            f"{key}_{name}": np.array([0., state[5 * muscle_index + key_index]])
            for muscle_index, name in enumerate(names)
            for key_index, key in enumerate(keys)
        }
        return SimpleNamespace(decision_states=lambda **_: values)

    invalid = physical.copy()
    invalid[2] = -1.
    assert nmpc.advance_window(solution(invalid), steps=1) is None
    assert not nmpc.last_max_pw_work_pretransfer_audit["valid"]
    assert calls == []
    valid_solution = solution(physical)
    assert nmpc.advance_window(valid_solution, steps=1) == "advanced"
    assert nmpc.last_max_pw_work_pretransfer_audit["valid"]
    assert calls == [(valid_solution, 1, {})]
