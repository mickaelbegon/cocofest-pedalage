"""Regression checks for an IPOPT ΔPW seed consumed by ACADOS or recovery."""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from bioptim import BoundsList, InitialGuessList, VariableScalingList, OdeSolver, InterpolationType, Solver, SolutionMerge
from casadi import collocation_points

from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as example
from examples.fes_multibody.cycling.cycling_pulse_width_mhe import MyCyclicNMPC
from cocofest.optimization.pulse_width_slew import add_auxiliary_bounds_and_guesses


PW = "last_pulse_width_m"
CARRIER = "pw_slew_carrier_m"
DELTA = "pw_slew_delta_pw_m"


def lifted_seed_target(stride=1):
    controls = np.array([[160., 240., 320.]]) * 1e-6
    delta = np.array([[80., 80., 40.]]) * 1e-6
    carrier = np.array([[160., 240., 320., 360.]]) * 1e-6
    integrated = np.concatenate((np.zeros((1, 1)), np.cumsum(controls / 3, axis=1)), axis=1)
    source_grid = np.arange(4)
    dense_grid = np.linspace(0., 3., 3 * stride + 1)
    states = {"F_m": np.interp(dense_grid, source_grid, integrated[0])[None, :],
              CARRIER: np.interp(dense_grid, source_grid, carrier[0])[None, :]}
    seed = example._WarmupSolutionAdapter(states, {PW: controls.copy(), DELTA: delta.copy()})
    target = SimpleNamespace(nlp=[SimpleNamespace(
        model=SimpleNamespace(pulse_width_max_step_s=100e-6,
                              muscles_dynamics_model=[SimpleNamespace(muscle_name="m", pd0=131.405e-6)]),
        x_init={"F_m": SimpleNamespace(init=np.zeros((1, 4))),
                CARRIER: SimpleNamespace(init=np.full((1, 4), 300e-6))},
        u_init={PW: SimpleNamespace(init=np.full((1, 3), 300e-6)),
                DELTA: SimpleNamespace(init=np.zeros((1, 3)))},
    )])
    return seed, target, controls, delta, carrier, integrated


@pytest.mark.parametrize("stride", [1, 6])
def test_certified_slew_seed_keeps_its_physical_controls_and_lift(stride):
    seed, target, controls, delta, carrier, integrated = lifted_seed_target(stride)
    adapted = example._adapt_warmup_solution_to_periodic_nodes(target, seed)
    actual_u = adapted.decision_controls()
    actual_x = adapted.decision_states()
    np.testing.assert_array_equal(actual_u[PW], controls)
    np.testing.assert_array_equal(actual_u[DELTA], delta)
    np.testing.assert_array_equal(actual_x[CARRIER], carrier)
    np.testing.assert_array_equal(actual_x["F_m"], integrated)
    # An integrator xdot=u makes the physical inconsistency independently
    # observable: replacing u by a constant destroys a zero-defect seed.
    np.testing.assert_allclose(np.diff(actual_x["F_m"], axis=1), actual_u[PW] / 3, atol=1e-18)
    np.testing.assert_allclose(actual_x[CARRIER][:, :-1], actual_u[PW], atol=1e-18)
    np.testing.assert_allclose(np.diff(actual_x[CARRIER], axis=1), actual_u[DELTA], atol=1e-18)
    # The adapter must not modify either its source or the target container.
    np.testing.assert_array_equal(seed.decision_controls()[PW], controls)
    np.testing.assert_array_equal(target.nlp[0].u_init[PW].init, np.full((1, 3), 300e-6))


def test_generic_unlifted_warmup_retains_the_existing_constant_auxiliary_fallback():
    seed, target, *_ = lifted_seed_target()
    generic = example._WarmupSolutionAdapter({"F_m": seed.decision_states()["F_m"]},
                                             {PW: seed.decision_controls()[PW]})
    adapted = example._adapt_warmup_solution_to_periodic_nodes(target, generic)
    np.testing.assert_array_equal(adapted.decision_controls()[PW], target.nlp[0].u_init[PW].init)
    np.testing.assert_array_equal(adapted.decision_controls()[DELTA], target.nlp[0].u_init[DELTA].init)
    np.testing.assert_array_equal(adapted.decision_states()[CARRIER], target.nlp[0].x_init[CARRIER].init)


def test_legacy_next_pw_seed_is_rejected_explicitly():
    seed, target, physical, delta, *_ = lifted_seed_target()
    legacy = example._WarmupSolutionAdapter(
        seed.decision_states(), {PW: physical, "pw_slew_next_m": physical + delta}
    )
    with pytest.raises(ValueError, match="Legacy PW-slew seed.*delta_pw_v1"):
        example._adapt_warmup_solution_to_periodic_nodes(target, legacy)


@pytest.mark.parametrize("representation", [None, "next_pw_v1"])
def test_common_seed_rejects_missing_or_legacy_increment_signature(representation):
    args = example.build_argument_parser().parse_args([])
    args.pulse_width_max_step_us = 100.
    args.terminal_wheel_q_reference_mode = "absolute_initial"
    metadata = example._common_initial_solution_metadata(args)
    assert metadata["pulse_width_slew_control_representation"] == "delta_pw_v1"
    metadata["pulse_width_slew_control_representation"] = representation
    with pytest.raises(ValueError, match="pulse_width_slew_control_representation"):
        example._validate_common_initial_solution_metadata(
            SimpleNamespace(metadata=metadata), args, "legacy.npz"
        )


@pytest.mark.parametrize("missing", [CARRIER, DELTA])
def test_partially_lifted_seed_is_rejected_instead_of_mixed_with_defaults(missing):
    seed, target, *_ = lifted_seed_target()
    states = dict(seed.decision_states())
    controls = dict(seed.decision_controls())
    (states if missing == CARRIER else controls).pop(missing)
    incomplete = example._WarmupSolutionAdapter(states, controls)
    with pytest.raises(KeyError, match=missing):
        example._adapt_warmup_solution_to_periodic_nodes(target, incomplete)


@pytest.mark.parametrize("collocation", [False, True])
def test_auxiliary_guesses_use_the_same_explicit_grid_as_physical_states(collocation):
    model = SimpleNamespace(muscles_dynamics_model=[SimpleNamespace(muscle_name="m")],
                            pulse_width_max_step_s=100e-6)
    xb, xi, xs = BoundsList(), InitialGuessList(), VariableScalingList()
    ub, ui, us = BoundsList(), InitialGuessList(), VariableScalingList()
    ub.add(PW, min_bound=[131.405e-6], max_bound=[600e-6])
    ui.add(PW, initial_guess=[300e-6])
    ode = OdeSolver.COLLOCATION(polynomial_degree=5, method="radau") if collocation else OdeSolver.RK4()
    add_auxiliary_bounds_and_guesses(model, xb, xi, xs, ub, ui, us,
                                     n_shooting=3, scale=.0025, ode_solver=ode)
    assert xi[CARRIER].init.shape == (1, 19 if collocation else 4)
    assert xi[CARRIER].init.type == (InterpolationType.ALL_POINTS if collocation else InterpolationType.EACH_FRAME)
    assert ui[DELTA].init.shape == (1, 3)
    np.testing.assert_array_equal(ui[DELTA].init, np.zeros((1, 3)))
    np.testing.assert_allclose(ub[DELTA].min, -100e-6)
    np.testing.assert_allclose(ub[DELTA].max, 100e-6)


@pytest.mark.parametrize("mode", ["repeat", "extrapolate", "lag2"])
@pytest.mark.parametrize("collocation", [False, True])
def test_window_transfer_rebuilds_lift_from_the_actual_physical_predictor(mode, collocation):
    seed, target, physical, *_ = lifted_seed_target()
    nlp = target.nlp[0]
    ode = OdeSolver.COLLOCATION(polynomial_degree=5, method="radau") if collocation else OdeSolver.RK4()
    nlp.dynamics_type = SimpleNamespace(ode_solver=ode)
    state_stride = 6 if collocation else 1
    nlp.x_init[CARRIER].init = np.full((1, 3 * state_stride + 1), 555e-6)
    previous = physical[0] - np.array([20., 30., 40.]) * 1e-6
    target.control_nodes_per_cycle = 3
    target.debugg_bounds = False
    target.pulse_width_transfer_mode = mode
    target.pulse_width_extrapolation_factor = 1.
    target._previous_pulse_width_cycle = {PW: previous.copy()}
    target._pulse_width_transfer_candidates = {}
    target._correct_init_guess_to_fit_bounds = lambda **kw: None
    target.set_init_cyclical_controls = MethodType(MyCyclicNMPC.set_init_cyclical_controls, target)
    target._rebuild_pulse_width_slew_initial_guess = MethodType(MyCyclicNMPC._rebuild_pulse_width_slew_initial_guess, target)
    MyCyclicNMPC.advance_window_initial_guess_controls(target, seed)
    expected = physical[0] if mode == "repeat" else previous if mode == "lag2" else 2 * physical[0] - previous
    np.testing.assert_allclose(nlp.u_init[PW].init[0], expected, atol=1e-18)
    following = np.r_[expected[1:], expected[-1]]
    np.testing.assert_allclose(nlp.u_init[DELTA].init[0], following - expected, atol=1e-18)
    carrier = nlp.x_init[CARRIER].init[0]
    np.testing.assert_allclose(carrier[::state_stride], np.r_[expected, expected[-1]], atol=1e-18)
    fractions = np.r_[0., collocation_points(5, "radau")] if collocation else np.array([0.])
    for stage in range(3):
        np.testing.assert_allclose(carrier[stage * state_stride:(stage + 1) * state_stride],
                                   expected[stage] + fractions * (following[stage] - expected[stage]), atol=1e-18)
    assert carrier[-1] == expected[-1]
    assert carrier[-1] != expected[0]  # no artificial periodic closure


def test_ipopt_npz_common_seed_reaches_pre_solve_state_and_control_packing_unchanged(tmp_path):
    """A real small IPOPT/Radau solution crosses the actual NPZ/adapter path.

    The target is the same lifted OCP on a shooting grid. We inspect Bioptim's
    actual pre-solve ACADOS state packing without building or running ACADOS.
    """
    from tests.test_pulse_width_slew import _ocp
    from bioptim.interfaces.acados_interface import _scaled_state_initial_guess

    source = _ocp(intervals=6, collocation=True)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_tol(1e-9)
    solved = source.solve(solver)
    assert solved.status == 0
    source_u = solved.decision_controls(to_merge=SolutionMerge.NODES)
    source_x = solved.decision_states(to_merge=SolutionMerge.NODES)
    assert np.ptp(source_u[PW]) > 100e-6
    seed_path = tmp_path / "ipopt-cycle-1.npz"
    target_args = SimpleNamespace(
        solver="acados",
        mechanical_formulation="reduced",
        formulation="dynamic",
        reduced_internal_crank_velocity_guard="on",
        stimulations_per_cycle=6,
        wheel_qdot_regularization_target=-2.0 * np.pi,
        wheel_qdot_bound_margin=3.0,
        acados_wheel_qdot_fast_bound_margin=None,
        acados_wheel_qdot_slow_bound_margin=None,
    )
    example._save_warmup_cache(
        seed_path,
        solved,
        metadata={
            "producer_solver": "ipopt",
            "cycles_per_window": 1,
            **example.reduced_internal_crank_velocity_guard_signature(target_args),
        },
    )
    seed = example._load_warmup_cache(seed_path)
    example.validate_acados_ipopt_common_seed_provenance(
        seed, target_args, seed_path
    )
    target = _ocp(intervals=6, collocation=False)
    # The cycling constructor already supplies EACH_FRAME physical controls;
    # make the toy's otherwise scalar control guess follow that same layout.
    target_controls = InitialGuessList()
    for key in (PW, DELTA):
        target_controls.add(key, initial_guess=np.full((1, 6), 300e-6 if key == PW else 0.), interpolation=InterpolationType.EACH_FRAME)
    target.update_initial_guess(u_init=target_controls)
    target._correct_init_guess_to_fit_bounds = MethodType(MyCyclicNMPC._correct_init_guess_to_fit_bounds, target)
    target._sync_acados_state_bounds = MethodType(MyCyclicNMPC._sync_acados_state_bounds, target)
    example.apply_solution_directly_to_periodic_nmpc_initial_guess(target, seed)
    nlp = target.nlp[0]
    np.testing.assert_allclose(nlp.u_init[PW].init, source_u[PW], atol=1e-10)
    np.testing.assert_allclose(nlp.u_init[DELTA].init, source_u[DELTA], atol=1e-10)
    np.testing.assert_allclose(nlp.x_init[CARRIER].init, source_x[CARRIER][:, ::6], atol=1e-10)
    carrier_scale = np.asarray(nlp.x_scaling[CARRIER].scaling[:, 0]).reshape(-1)
    for stage in range(7):
        packed = _scaled_state_initial_guess(nlp, stage)
        np.testing.assert_allclose(packed * carrier_scale, source_x[CARRIER][:, stage * 6].reshape(-1), atol=1e-10)
    # All native-preparation reads leave the seed untouched.
    np.testing.assert_array_equal(seed.decision_controls()[PW], source_u[PW])
