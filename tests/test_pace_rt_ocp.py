from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from cocofest.optimization.pace_rt_ocp import PaceRtObjectiveBinding, PACE_RT_MODE
from cocofest.optimization.task_reserve_ocp import TaskReserveStateCoordinate


def _binding():
    coordinates = (TaskReserveStateCoordinate("A_m0", scale=100.),
                   TaskReserveStateCoordinate("A_m1", scale=200.))
    context = {"coordinate_layout": [dict(state_key="A_m0", index=0, scale=100., offset=0.),
                                     dict(state_key="A_m1", index=0, scale=200., offset=0.)]}
    return PaceRtObjectiveBinding(coordinates, task_context=context, model_sha256="a" * 64,
                                  target_fraction=.25, maximum_age_cycles=20)


def test_pace_rt_shortage_is_targeted_and_proximal():
    binding = _binding()
    x = ca.SX.sym("x", 2)
    p = ca.SX.sym("p", binding.values.size)
    controller = SimpleNamespace(states={"A_m0": SimpleNamespace(cx=x[0]), "A_m1": SimpleNamespace(cx=x[1])},
                                 parameters={"rho_task_reserve": SimpleNamespace(cx=p)})
    fun = ca.Function("pace_rt", [x, p], [binding.objective(controller)])
    values = np.r_[1., .2, .3, [.8, .8], [-1., 0.], [.1, .1], .01, 1., 1., 0.]
    # Below target: only a small proximity term remains. Above target: the
    # smooth shortage must increase, but no affine reward drives it unbounded.
    good = float(fun([90., 160.], values))
    bad = float(fun([70., 160.], values))
    assert good < bad


def test_pace_rt_inactive_bootstrap_is_finite():
    binding = _binding()
    x = ca.SX.sym("x", 2)
    p = ca.SX.sym("p", binding.values.size)
    controller = SimpleNamespace(states={"A_m0": SimpleNamespace(cx=x[0]), "A_m1": SimpleNamespace(cx=x[1])},
                                 parameters={"rho_task_reserve": SimpleNamespace(cx=p)})
    value = float(ca.Function("pace_rt_inactive", [x, p], [binding.objective(controller)])(
        [80., 160.], binding.values))
    assert np.isfinite(value)
    assert value == 0.


def test_pace_rt_rollout_update_reuses_numeric_parameter_channel():
    binding = _binding()
    written = {}
    nmpc = SimpleNamespace(nlp=[object()], parameter_bounds={"rho_task_reserve": SimpleNamespace(
        min=np.zeros((binding.values.size, 1)), max=np.zeros((binding.values.size, 1)))},
        update_initial_guess=lambda **kwargs: written.update(kwargs))
    fit = {"accepted": True, "constant": .4, "gradient": [[-1., 0., 0.], [.5, 0., 0.]],
           "reference_state": [[80., 1., 1.], [160., 1., 1.]],
           "state_scales": [[100., 1., 1.], [200., 1., 1.]],
           "trust_bounds": [-.01, .01], "fit_rank": 2, "sample_count": 2}
    summary = binding.update_from_rollout(nmpc, local_fit=fit, source_completed_cycles=10,
                                          completed_cycles=11, proximal_weight=.03)
    assert summary["mode"] == PACE_RT_MODE
    assert binding.values[0] == 1.
    assert binding.values[1] == pytest.approx(.4 - .25 * .015)
    assert "parameter_init" in written
    assert binding.validate_terminal_point([.8, .8], completed_cycles=11)["terminal_trust_validated"]
    assert not binding.validate_terminal_point([.82, .8], completed_cycles=11)["terminal_trust_validated"]


def test_pace_rt_can_ablate_the_shortage_term_without_rebuilding_the_graph():
    binding = _binding()
    written = {}
    nmpc = SimpleNamespace(nlp=[object()], parameter_bounds={"rho_task_reserve": SimpleNamespace(
        min=np.zeros((binding.values.size, 1)), max=np.zeros((binding.values.size, 1)))},
        update_initial_guess=lambda **kwargs: written.update(kwargs))
    fit = {"accepted": True, "constant": .4, "gradient": [[-1., 0., 0.], [.5, 0., 0.]],
           "reference_state": [[80., 1., 1.], [160., 1., 1.]],
           "state_scales": [[100., 1., 1.], [200., 1., 1.]],
           "trust_bounds": [-.01, .01]}
    binding.update_from_rollout(nmpc, local_fit=fit, source_completed_cycles=10,
                                completed_cycles=11, proximal_weight=.03, shortage_weight=0.)
    assert binding.values[binding._slices["shortage"]] == 0.
    assert binding.values[binding._slices["proximal"]] == pytest.approx(.03)
    assert "parameter_init" in written


def test_pace_rt_normalizes_shortage_by_local_attainable_decrease():
    coordinates = (TaskReserveStateCoordinate("A_m0", scale=100.),
                   TaskReserveStateCoordinate("A_m1", scale=200.))
    context = {"coordinate_layout": [dict(state_key="A_m0", index=0, scale=100., offset=0.),
                                     dict(state_key="A_m1", index=0, scale=200., offset=0.)]}
    binding = PaceRtObjectiveBinding(coordinates, task_context=context, model_sha256="a" * 64,
                                     target_fraction=.25, smoothing=1e-3,
                                     normalize_shortage=True)
    written = {}
    nmpc = SimpleNamespace(nlp=[object()], parameter_bounds={"rho_task_reserve": SimpleNamespace(
        min=np.zeros((binding.values.size, 1)), max=np.zeros((binding.values.size, 1)))},
        update_initial_guess=lambda **kwargs: written.update(kwargs))
    fit = {"accepted": True, "constant": .4, "gradient": [[-1., 0., 0.], [.5, 0., 0.]],
           "reference_state": [[80., 1., 1.], [160., 1., 1.]],
           "state_scales": [[100., 1., 1.], [200., 1., 1.]], "trust_bounds": [-.01, .01]}
    binding.update_from_rollout(nmpc, local_fit=fit, source_completed_cycles=10,
                                completed_cycles=11, proximal_weight=.03)
    assert binding.values[binding._slices["shortage_scale"]] == pytest.approx(.015)
    assert binding.last_model["shortage_normalized"] is True
    assert binding.summary()["shortage_normalized_by_local_attainable_decrease"] is True


def test_pace_rt_can_compile_an_independent_terminal_target_constraint():
    coordinates = (TaskReserveStateCoordinate("A_m0", scale=100.),
                   TaskReserveStateCoordinate("A_m1", scale=200.))
    context = {"coordinate_layout": [dict(state_key="A_m0", index=0, scale=100., offset=0.),
                                     dict(state_key="A_m1", index=0, scale=200., offset=0.)]}
    binding = PaceRtObjectiveBinding(coordinates, task_context=context, model_sha256="a" * 64,
                                     enforce_target_constraint=True)
    x = ca.SX.sym("x", 2)
    p = ca.SX.sym("p", binding.values.size)
    controller = SimpleNamespace(states={"A_m0": SimpleNamespace(cx=x[0]), "A_m1": SimpleNamespace(cx=x[1])},
                                 parameters={"rho_task_reserve": SimpleNamespace(cx=p)})
    residual = ca.Function("pace_rt_target", [x, p], [binding.target_constraint(controller)])
    values = np.r_[0., .2, .3, [.8, .8], [-1., 0.], [.1, .1], .01, 1., 1., 1.]
    assert float(residual([100., 160.], values)) < 0.
    assert float(residual([100., 160.], np.r_[values[:-1], 0.])) == 0.
    assert binding.summary()["terminal_target_constraint_compiled"] is True


def test_pace_rt_can_enter_fatigue_secondary_stage_without_rebuilding_graph():
    coordinates = (TaskReserveStateCoordinate("A_m0", scale=100.),
                   TaskReserveStateCoordinate("A_m1", scale=200.))
    context = {"coordinate_layout": [dict(state_key="A_m0", index=0, scale=100., offset=0.),
                                     dict(state_key="A_m1", index=0, scale=200., offset=0.)]}
    binding = PaceRtObjectiveBinding(coordinates, task_context=context, model_sha256="a" * 64,
                                     enforce_target_constraint=True)
    written = {}
    nmpc = SimpleNamespace(nlp=[object()], parameter_bounds={"rho_task_reserve": SimpleNamespace(
        min=np.zeros((binding.values.size, 1)), max=np.zeros((binding.values.size, 1)))},
        update_initial_guess=lambda **kwargs: written.update(kwargs))
    fit = {"accepted": True, "constant": .4, "gradient": [[-1., 0., 0.], [.5, 0., 0.]],
           "reference_state": [[80., 1., 1.], [160., 1., 1.]],
           "state_scales": [[100., 1., 1.], [200., 1., 1.]], "trust_bounds": [-.01, .01]}
    binding.update_from_rollout(nmpc, local_fit=fit, source_completed_cycles=10,
                                completed_cycles=11, proximal_weight=.03)
    graph = nmpc.nlp[0]
    receipt = binding.activate_fatigue_secondary_stage(nmpc, reserve_target=.37)
    assert nmpc.nlp[0] is graph
    assert binding.values[binding._slices["activation"]] == 0.
    assert binding.values[binding._slices["constraint_activation"]] == 1.
    assert binding.values[binding._slices["target"]] == pytest.approx(.37)
    assert receipt["priority_stage"] == "fatigue_secondary"


def test_pace_rt_rejects_unidentified_memory_gradient():
    binding = _binding()
    nmpc = SimpleNamespace(nlp=[object()], parameter_bounds={"rho_task_reserve": SimpleNamespace(
        min=np.zeros((binding.values.size, 1)), max=np.zeros((binding.values.size, 1)))},
        update_initial_guess=lambda **kwargs: None)
    fit = {"accepted": True, "constant": .4, "gradient": [[-1., .1, 0.], [.5, 0., 0.]],
           "reference_state": [[80., 1., 1.], [160., 1., 1.]],
           "state_scales": [[100., 1., 1.], [200., 1., 1.]], "trust_bounds": [-.01, .01]}
    with pytest.raises(ValueError, match="A-state gradient"):
        binding.update_from_rollout(nmpc, local_fit=fit, source_completed_cycles=10,
                                    completed_cycles=11, proximal_weight=.03)
