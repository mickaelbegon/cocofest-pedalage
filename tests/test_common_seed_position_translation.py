"""A common IPOPT seed must not be clipped against an earlier absolute cycle."""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as example
from examples.fes_multibody.cycling.cycling_pulse_width_mhe import MyCyclicNMPC


def common_seed_target(shift, full=False):
    position_key, velocity_key = ("q", "qdot") if full else ("theta", "omega")
    wheel_index = 2 if full else 0
    position = np.full((3 if full else 1, 31), .25)
    position[wheel_index] = np.linspace(.1, .1 - 2 * np.pi, 31)
    velocity = np.full_like(position, -2 * np.pi)
    force = np.linspace(20., 30., 31)[None, :]
    states = {position_key: position.copy(), velocity_key: velocity.copy(), "F_m": force.copy()}
    states[position_key][wheel_index] += shift
    controls = {"last_pulse_width_m": np.full((1, 30), 300e-6)}
    seed = example._WarmupSolutionAdapter(states, controls,
                                          metadata={"producer_solver": "ipopt", "cycles_per_window": 1})
    low, high = np.full((position.shape[0], 3), -.5), np.full((position.shape[0], 3), .5)
    low[wheel_index] = [.05, -2 * np.pi - .2, .05 - 2 * np.pi]
    high[wheel_index] = [.15, .3, .15 - 2 * np.pi]
    x_bounds = {position_key: SimpleNamespace(min=low, max=high),
                velocity_key: SimpleNamespace(min=np.full((velocity.shape[0], 3), -10.), max=np.full((velocity.shape[0], 3), 0.)),
                "F_m": SimpleNamespace(min=np.zeros((1, 3)), max=np.full((1, 3), 100.))}
    target = SimpleNamespace(position_state_key=position_key, wheel_state_index=wheel_index,
                             nlp=[SimpleNamespace(
                                 model=SimpleNamespace(muscles_dynamics_model=[]),
                                 x_init={position_key: SimpleNamespace(init=position.copy()),
                                         velocity_key: SimpleNamespace(init=velocity.copy()),
                                         "F_m": SimpleNamespace(init=force.copy())},
                                 u_init={key: SimpleNamespace(init=value.copy()) for key, value in controls.items()},
                                 x_bounds=x_bounds,
                                 u_bounds={"last_pulse_width_m": SimpleNamespace(min=np.full((1, 3), 131e-6), max=np.full((1, 3), 600e-6))},
                             )], _sync_acados_state_bounds=lambda: None)
    target._correct_init_guess_to_fit_bounds = MethodType(MyCyclicNMPC._correct_init_guess_to_fit_bounds, target)
    return seed, target


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("shift", [-2 * np.pi, 0.])
def test_exact_common_seed_translates_absolute_bounds_without_clipping_or_widening(full, shift):
    seed, target = common_seed_target(shift, full)
    nlp = target.nlp[0]
    original_bounds = {key: (bounds.min.copy(), bounds.max.copy()) for key, bounds in nlp.x_bounds.items()}
    original_states = {key: values.copy() for key, values in seed.decision_states().items()}
    example.apply_solution_directly_to_periodic_nmpc_initial_guess(
        target, seed, translate_absolute_position_bounds=True,
        # The old position recenter option must not widen PATH or independently
        # change END on the exact common-seed translation path.
        recenter_position_bounds=True,
    )
    for key, bounds in nlp.x_bounds.items():
        low, high = original_bounds[key]
        expected_low, expected_high = low.copy(), high.copy()
        if key == target.position_state_key:
            expected_low[target.wheel_state_index] += shift
            expected_high[target.wheel_state_index] += shift
        np.testing.assert_allclose(bounds.min, expected_low, atol=1e-14)
        np.testing.assert_allclose(bounds.max, expected_high, atol=1e-14)
        np.testing.assert_allclose(bounds.max - bounds.min, high - low, atol=1e-14)
        np.testing.assert_array_equal(nlp.x_init[key].init, original_states[key])
        np.testing.assert_array_equal(seed.decision_states()[key], original_states[key])
        if shift == 0:
            np.testing.assert_array_equal(bounds.min, low)
            np.testing.assert_array_equal(bounds.max, high)


def test_exact_translation_keeps_position_width_even_when_other_first_states_are_recentered():
    seed, target = common_seed_target(-2 * np.pi)
    original_width = target.nlp[0].x_bounds["theta"].max - target.nlp[0].x_bounds["theta"].min
    example.apply_solution_directly_to_periodic_nmpc_initial_guess(
        target, seed, translate_absolute_position_bounds=True, recenter_first_node_bounds=True,
    )
    np.testing.assert_allclose(target.nlp[0].x_bounds["theta"].max - target.nlp[0].x_bounds["theta"].min,
                               original_width, atol=1e-14)


def test_unrelated_direct_refinement_keeps_its_existing_bounds_policy():
    seed, target = common_seed_target(-2 * np.pi)
    original_low = target.nlp[0].x_bounds["theta"].min.copy()
    example.apply_solution_directly_to_periodic_nmpc_initial_guess(target, seed)
    np.testing.assert_array_equal(target.nlp[0].x_bounds["theta"].min, original_low)
    assert not np.array_equal(target.nlp[0].x_init["theta"].init, seed.decision_states()["theta"])


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("shift", [-2 * np.pi, 0.])
def test_default_exact_common_seed_pairs_first_physical_states_before_clipping(full, shift):
    args = example.build_argument_parser().parse_args([
        "--solver", "acados", "--formulation", "dynamic", "--common-initial-solution", "ipopt-cycle-1.npz",
    ])
    assert args.common_initial_solution_recenter_first_node_bounds is False
    seed, target = common_seed_target(shift, full)
    nlp = target.nlp[0]
    # The fresh OCP starts unactivated, whereas cycle 1 carries accumulated FES
    # force. Without automatic pairing the real clipping method erases F(0).
    nlp.x_bounds["F_m"].min[:, 0] = 0.
    nlp.x_bounds["F_m"].max[:, 0] = 0.
    originals = {key: (b.min.copy(), b.max.copy()) for key, b in nlp.x_bounds.items()}
    original_states = {key: values.copy() for key, values in seed.decision_states().items()}
    translate = example._common_seed_uses_absolute_position_translation(args, seed, mechanical_bridge=False)
    kinematic, position, first = example._common_initial_solution_recenter_modes(args, False, translate)
    assert (kinematic, position, first) == (False, True, True)
    example.apply_solution_directly_to_periodic_nmpc_initial_guess(
        target, seed, recenter_kinematic_bounds=kinematic, recenter_position_bounds=position,
        recenter_first_node_bounds=first, translate_absolute_position_bounds=translate,
    )
    for key, values in original_states.items():
        np.testing.assert_array_equal(nlp.x_init[key].init, values)
        np.testing.assert_array_equal(seed.decision_states()[key], values)
        low, high = originals[key]
        expected_low, expected_high = low.copy(), high.copy()
        expected_low[:, 0] = expected_high[:, 0] = values[:, 0]
        if key == target.position_state_key:
            row = target.wheel_state_index
            expected_low[row, :] = low[row, :] + shift
            expected_high[row, :] = high[row, :] + shift
            np.testing.assert_allclose(nlp.x_bounds[key].max[row] - nlp.x_bounds[key].min[row],
                                       high[row] - low[row], atol=1e-14)
        np.testing.assert_allclose(nlp.x_bounds[key].min, expected_low, atol=1e-14)
        np.testing.assert_allclose(nlp.x_bounds[key].max, expected_high, atol=1e-14)


@pytest.mark.parametrize("change,metadata,bridge,expected", [
    ({}, {"producer_solver": "ipopt", "cycles_per_window": 1}, False, True),
    ({"solver": "ipopt"}, {"producer_solver": "ipopt", "cycles_per_window": 1}, False, False),
    ({"formulation": "isokinetic"}, {"producer_solver": "ipopt", "cycles_per_window": 1}, False, False),
    ({"common_initial_solution_feasibility_probe": True}, {"producer_solver": "ipopt", "cycles_per_window": 1}, False, False),
    ({}, {"producer_solver": "acados", "cycles_per_window": 1}, False, False),
    ({}, {"producer_solver": "ipopt", "cycles_per_window": 2}, False, False),
    ({}, {"producer_solver": "ipopt", "cycles_per_window": 1}, True, False),
])
def test_translation_is_limited_to_exact_same_mechanics_common_ipopt_cycle_one(change, metadata, bridge, expected):
    args = SimpleNamespace(**({"solver": "acados", "formulation": "dynamic",
                               "common_initial_solution_feasibility_probe": False,
                               "common_initial_solution_recenter_first_node_bounds": False} | change))
    translated = example._common_seed_uses_absolute_position_translation(args, SimpleNamespace(metadata=metadata), bridge)
    assert translated is expected
    kinematic, position, first = example._common_initial_solution_recenter_modes(args, bridge, translated)
    assert first is expected
    assert position is expected
    assert kinematic is bridge
