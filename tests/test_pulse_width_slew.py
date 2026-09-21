"""Small exact OCP checks; no cycling campaign or fitted mechanics required."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from casadi import vertcat, Function, DM
from bioptim import (BoundsList, ConfigureVariables, ConstraintList, DynamicsEvaluation,
                     DynamicsFunctions,
                     DynamicsOptions, InitialGuessList, ObjectiveFcn, ObjectiveList,
                     OdeSolver, OptimalControlProgram, SolutionMerge, Solver, StateDynamics,
                     VariableScalingList)

from cocofest.optimization.pulse_width_slew import (
    DIRECT_CONSTRAINTS_FORMULATION, add_auxiliary_bounds_and_guesses,
    add_direct_slew_constraints, add_slew_constraints, adjacent_pulse_width_difference,
    auxiliary_configuration,
    auxiliary_rhs, auxiliary_keys, validate_max_step,
    advance_auxiliary_bounds, advance_direct_slew_bounds, validate_slew_formulation,
)


class _SlewToy(StateDynamics):
    def __init__(self, intervals=30):
        super().__init__()
        self.muscles_dynamics_model = [SimpleNamespace(muscle_name="m")]
        self.pulse_width_interval_s = 1 / intervals
        self.pulse_width_max_step_s = 100e-6

    @property
    def name(self):
        return "pulse_width_slew_test"

    @property
    def name_dofs(self):
        return ["toy"]

    @property
    def state_configuration_functions(self):
        return auxiliary_configuration(self.muscles_dynamics_model, states=True)

    @property
    def control_configuration_functions(self):
        return [lambda ocp, nlp: ConfigureVariables.configure_new_variable(
            "last_pulse_width_m", ["last_pulse_width_m"], ocp, nlp, as_controls=True
        )] + auxiliary_configuration(self.muscles_dynamics_model, states=False)

    @property
    def algebraic_configuration_functions(self):
        return []

    @property
    def extra_configuration_functions(self):
        return []

    def serialize(self):
        return _SlewToy, {"intervals": round(1 / self.pulse_width_interval_s)}

    def dynamics(self, time, states, controls, parameters, algebraic_states, numerical_data_timeseries, nlp):
        rhs = auxiliary_rhs(self, states, controls, nlp)
        defects = nlp.states_dot.scaled.cx * nlp.dt - rhs * nlp.dt if nlp.dynamics_type.ode_solver.is_direct_collocation else None
        return DynamicsEvaluation(dxdt=rhs, defects=defects)


def _ocp(intervals=30, *, collocation=True, odd_interpolation=False):
    model = _SlewToy(intervals)
    xb, xi, xs = BoundsList(), InitialGuessList(), VariableScalingList()
    ub, ui, us = BoundsList(), InitialGuessList(), VariableScalingList()
    ub.add("last_pulse_width_m", min_bound=[131.405e-6], max_bound=[600e-6])
    ui.add("last_pulse_width_m", initial_guess=[300e-6])
    us.add("last_pulse_width_m", scaling=[.0025])
    add_auxiliary_bounds_and_guesses(model, xb, xi, xs, ub, ui, us,
                                     n_shooting=intervals, scale=.0025)
    constraints = ConstraintList()
    add_slew_constraints(constraints, model.muscles_dynamics_model, max_step_s=100e-6, scale=.0025)
    objective = ObjectiveList()
    # A discontinuous desired pattern excites the hard bound, including the seam.
    target = np.full((1, intervals), 131.405e-6)
    target[0, :intervals // 2] = 600e-6
    objective.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="last_pulse_width_m",
                  target=target, weight=1e10, quadratic=True)
    ode = OdeSolver.COLLOCATION(polynomial_degree=5, method="radau") if collocation else OdeSolver.RK4(n_integration_steps=1)
    if odd_interpolation:
        from cocofest.optimization.pulse_width_interpolation import odd_interpolation_constraints
        odd_interpolation_constraints(model.muscles_dynamics_model, scale=.0025, constraints=constraints)
    return OptimalControlProgram(model, intervals, 1., dynamics=DynamicsOptions(ode_solver=ode),
                                 constraints=constraints, objective_functions=objective,
                                 x_bounds=xb, x_init=xi, x_scaling=xs,
                                 u_bounds=ub, u_init=ui, u_scaling=us, use_sx=True, n_threads=1)


class _DirectSlewToy(StateDynamics):
    """One-state OCP used to validate the sparse two-control-node binding."""
    def __init__(self):
        super().__init__()
        self.muscles_dynamics_model = [SimpleNamespace(muscle_name="m")]
        self.pulse_width_max_step_s = 100e-6
        self.pulse_width_slew_formulation = DIRECT_CONSTRAINTS_FORMULATION

    @property
    def name(self):
        return "direct_pulse_width_slew_test"

    @property
    def name_dofs(self):
        return ["dummy"]

    @property
    def state_configuration_functions(self):
        return [lambda ocp, nlp: ConfigureVariables.configure_new_variable(
            "dummy", ["dummy"], ocp, nlp, as_states=True
        )]

    @property
    def control_configuration_functions(self):
        return [lambda ocp, nlp: ConfigureVariables.configure_new_variable(
            "last_pulse_width_m", ["last_pulse_width_m"], ocp, nlp, as_controls=True
        )]

    @property
    def algebraic_configuration_functions(self):
        return []

    @property
    def extra_configuration_functions(self):
        return []

    def serialize(self):
        return _DirectSlewToy, {}

    def dynamics(self, time, states, controls, parameters, algebraic_states, numerical_data_timeseries, nlp):
        # Keep the physical PW in the NLP graph without adding an unrelated
        # mechanical restriction to the adjacent-control test.
        pw = DynamicsFunctions.get(nlp.controls["last_pulse_width_m"], controls)
        rhs = vertcat(pw)
        defects = (
            nlp.states_dot.scaled.cx * nlp.dt - rhs * nlp.dt
            if nlp.dynamics_type.ode_solver.is_direct_collocation
            else None
        )
        return DynamicsEvaluation(dxdt=rhs, defects=defects)


def _direct_ocp(intervals=30, *, collocation=True):
    model = _DirectSlewToy()
    xb, xi, xs = BoundsList(), InitialGuessList(), VariableScalingList()
    ub, ui, us = BoundsList(), InitialGuessList(), VariableScalingList()
    xb.add("dummy", min_bound=[-1.0], max_bound=[1.0])
    xi.add("dummy", initial_guess=[0.0])
    xs.add("dummy", scaling=[1.0])
    ub.add("last_pulse_width_m", min_bound=[131.405e-6], max_bound=[600e-6])
    ui.add("last_pulse_width_m", initial_guess=[300e-6])
    us.add("last_pulse_width_m", scaling=[.0025])
    constraints = ConstraintList()
    add_direct_slew_constraints(
        constraints, model.muscles_dynamics_model,
        max_step_s=100e-6, scale=.0025, n_shooting=intervals,
    )
    objective = ObjectiveList()
    target = np.full((1, intervals), 131.405e-6)
    target[0, :intervals // 2] = 600e-6
    objective.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="last_pulse_width_m",
                  target=target, weight=1e10, quadratic=True)
    ode = OdeSolver.COLLOCATION(polynomial_degree=5, method="radau") if collocation else OdeSolver.RK4()
    return OptimalControlProgram(
        model, intervals, 1., dynamics=DynamicsOptions(ode_solver=ode),
        constraints=constraints,
        objective_functions=objective, x_bounds=xb, x_init=xi, x_scaling=xs,
        u_bounds=ub, u_init=ui, u_scaling=us, use_sx=True, n_threads=1,
    )


def test_radau5_odd_interpolation_is_exact_with_increment_lift():
    ocp = _ocp(50, odd_interpolation=True)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_tol(1e-9)
    solution = ocp.solve(solver)
    assert solution.status == 0
    controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
    pw = np.asarray(controls["last_pulse_width_m"]).reshape(-1)
    delta = np.asarray(controls["pw_slew_delta_pw_m"]).reshape(-1)
    np.testing.assert_allclose(pw[1:48:2], .5 * (pw[0:47:2] + pw[2:49:2]), atol=1e-10)
    np.testing.assert_allclose(delta[:-1], np.diff(pw), atol=1e-10)
    assert np.max(np.abs(np.diff(pw))) <= 100e-6 + 1e-10
    # The endpoint controls differ: no hidden periodic closure is introduced.
    assert abs(pw[-1] - pw[0]) > 100e-6


@pytest.mark.parametrize("intervals", [30, 50])
def test_radau5_exact_successive_control_slew_bound(intervals):
    ocp = _ocp(intervals)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_tol(1e-9)
    solution = ocp.solve(solver)
    assert solution.status == 0
    u = solution.decision_controls(to_merge=SolutionMerge.NODES)
    pw = np.asarray(u["last_pulse_width_m"]).reshape(-1)
    delta = np.asarray(u["pw_slew_delta_pw_m"]).reshape(-1)
    assert pw.size == intervals
    np.testing.assert_allclose(delta[:-1], np.diff(pw), atol=1e-10)
    assert np.max(np.abs(delta)) <= 100e-6 + 1e-10
    differences = np.diff(pw)
    assert np.max(np.abs(differences)) <= 100e-6 + 1e-10
    assert np.max(np.abs(differences)) > 99e-6


@pytest.mark.parametrize("collocation", [False, True])
def test_direct_constraints_enforce_the_same_intra_window_pw_bound(collocation):
    ocp = _direct_ocp(12, collocation=collocation)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_tol(1e-9)
    solution = ocp.solve(solver)
    assert solution.status == 0
    pw = np.asarray(solution.decision_controls(to_merge=SolutionMerge.NODES)["last_pulse_width_m"]).reshape(-1)
    assert pw.size == 12
    assert np.max(np.abs(np.diff(pw))) <= 100e-6 + 1e-10
    # The discontinuous reference makes the hard adjacent bound active.
    assert np.max(np.abs(np.diff(pw))) > 99e-6


def test_direct_constraint_is_the_scaled_difference_of_two_physical_commands():
    controller = SimpleNamespace(controls={"last_pulse_width_m": SimpleNamespace(
        cx_start=200e-6, cx_end=275e-6,
    )})
    assert adjacent_pulse_width_difference(controller, muscle="m", scale=.0025) == pytest.approx(.03)


@pytest.mark.parametrize("value", [None, "lifting", "direct_constraints"])
def test_slew_formulation_validation(value):
    assert validate_slew_formulation(value) in {"lifting", "direct_constraints"}


def test_direct_slew_seam_bounds_follow_the_last_executed_command():
    model = SimpleNamespace(
        pulse_width_max_step_s=100e-6,
        pulse_width_slew_formulation=DIRECT_CONSTRAINTS_FORMULATION,
        muscles_dynamics_model=[SimpleNamespace(muscle_name="m")],
    )
    bounds = {"last_pulse_width_m": SimpleNamespace(
        min=np.full((1, 3), 131.405e-6),
        max=np.full((1, 3), 600e-6),
    )}
    nmpc = SimpleNamespace(nlp=[SimpleNamespace(model=model, u_bounds=bounds)], time_idx_to_cycle=30)
    solution = SimpleNamespace(decision_controls=lambda **kwargs: {"last_pulse_width_m": np.full((1, 30), 300e-6)})
    advance_direct_slew_bounds(nmpc, solution)
    assert bounds["last_pulse_width_m"].min[0, 0] == pytest.approx(200e-6)
    assert bounds["last_pulse_width_m"].max[0, 0] == pytest.approx(400e-6)


def test_acados_stage_local_constraint_export():
    from bioptim.interfaces.acados_interface import AcadosInterface
    from examples.fes_multibody.cycling.cycling_pulse_width_mhe_acados_periodic import patch_bioptim_acados_interface
    patch_bioptim_acados_interface()
    ocp = _ocp(50, collocation=False)
    opts = Solver.ACADOS()
    opts.set_integrator_type("IRK")
    interface = AcadosInterface(ocp, opts)
    interface._AcadosInterface__set_constraints(ocp)
    assert interface.acados_model.con_h_expr.shape == (1, 1)
    assert interface.acados_model.con_h_expr_0.shape == (1, 1)
    assert interface.acados_model.con_h_expr_e.shape == (0, 0)
    np.testing.assert_allclose(interface.acados_ocp.constraints.lh, [0])
    np.testing.assert_allclose(interface.acados_ocp.constraints.uh, [0])
    np.testing.assert_allclose(interface.acados_ocp.constraints.lbu, [131.405e-6 / .0025, -.04])
    np.testing.assert_allclose(interface.acados_ocp.constraints.ubu, [600e-6 / .0025, .04])
    # IRK sees a constant auxiliary derivative: z_next=z+delta exactly for any dt.
    model = interface.acados_model
    rhs = Function("slew_rhs", [model.x, model.u], [model.f_expl_expr])
    scaled_state = np.array([.0002 / .0025])
    scaled_control = np.array([.0002 / .0025, .0001 / .0025])
    endpoint = scaled_state + np.asarray(rhs(scaled_state, scaled_control)).reshape(-1) / 50
    np.testing.assert_allclose(endpoint, [.0003 / .0025], atol=1e-13)


def test_terminal_increment_does_not_close_the_physical_horizon():
    ocp = _ocp(6, collocation=False)
    nlp = ocp.nlp[0]
    carrier, delta = auxiliary_keys("m")
    # Every physical sequence satisfying adjacent bounds admits delta[-1]=0
    # and carrier[-1]=physical[-1], regardless of its first/last separation.
    physical = np.array([200., 280., 360., 440., 520., 600.]) * 1e-6
    increments = np.r_[np.diff(physical), 0.]
    endpoints = np.r_[physical, physical[-1]]
    np.testing.assert_allclose(np.diff(endpoints), increments, atol=1e-18)
    assert abs(physical[-1] - physical[0]) > 100e-6
    assert np.max(np.abs(increments)) <= 100e-6
    assert nlp.x_bounds[carrier].min[0, -1] <= endpoints[-1] <= nlp.x_bounds[carrier].max[0, -1]
    # Keep uniform increment bounds, including the unused final increment,
    # because native ACADOS control bounds are currently stage-invariant.
    np.testing.assert_allclose(nlp.u_bounds[delta].min, -100e-6)
    np.testing.assert_allclose(nlp.u_bounds[delta].max, 100e-6)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_step_rejected(value):
    with pytest.raises(ValueError, match="strictly positive"):
        validate_max_step(value)


@pytest.mark.parametrize("nmpc", [
    SimpleNamespace(),
    SimpleNamespace(nlp=None),
    SimpleNamespace(nlp=[]),
    SimpleNamespace(nlp=[SimpleNamespace()]),
    SimpleNamespace(nlp=[SimpleNamespace(model=None)]),
    SimpleNamespace(nlp=[SimpleNamespace(model=SimpleNamespace())]),
])
def test_advance_auxiliary_bounds_is_noop_without_slew_model(nmpc):
    # The historical minimal callback does not expose NLP or solution data.
    # No-op detection must precede accessing either the controls or bounds.
    advance_auxiliary_bounds(nmpc, SimpleNamespace())


def test_actual_rho_seam_bounds_do_not_freeze_first_pw():
    model = _SlewToy()
    carrier, _ = auxiliary_keys("m")
    bounds = {carrier: SimpleNamespace(
        min=np.full((1, 3), 131.405e-6),
        max=np.full((1, 3), 600e-6),
    )}
    nmpc = SimpleNamespace(nlp=[SimpleNamespace(model=model, x_bounds=bounds)], time_idx_to_cycle=30)
    solution = SimpleNamespace(decision_controls=lambda **kwargs: {"last_pulse_width_m": np.full((1, 30), 300e-6)})
    advance_auxiliary_bounds(nmpc, solution)
    assert bounds[carrier].min[0, 0] == pytest.approx(200e-6)
    assert bounds[carrier].max[0, 0] == pytest.approx(400e-6)
    # The next seam replaces the previous one, using physical limits at the
    # interior node; old seam bounds must not accumulate across RHO windows.
    solution = SimpleNamespace(decision_controls=lambda **kwargs: {"last_pulse_width_m": np.full((1, 30), 550e-6)})
    advance_auxiliary_bounds(nmpc, solution)
    assert bounds[carrier].min[0, 0] == pytest.approx(450e-6)
    assert bounds[carrier].max[0, 0] == pytest.approx(600e-6)


@pytest.mark.parametrize("isokinetic", [False, True])
def test_reduced_slew_dimensions_and_configuration(isokinetic, monkeypatch):
    from tests.shard1.test_isokinetic_reduced_model import _model

    model = _model(isokinetic=isokinetic, pulse_width_max_step_s=100e-6,
                   pulse_width_interval_s=1 / 30)
    configured = []
    monkeypatch.setattr(ConfigureVariables, "configure_new_variable",
                        staticmethod(lambda name, *args, **kwargs: configured.append(name)))
    for configure in model.state_configuration_functions:
        configure(None, None)
    assert len(configured) == model.nb_state == (27 if isokinetic else 26)
    configured.clear()
    for configure in model.control_configuration_functions:
        configure(None, None)
    assert len(configured) == 8
    assert sum(key.startswith("pw_slew_delta_pw_") for key in configured) == 4


@pytest.mark.parametrize("isokinetic, expected_states", [(False, 22), (True, 23)])
def test_direct_slew_constraints_do_not_add_carrier_states_or_delta_controls(
    isokinetic, expected_states, monkeypatch,
):
    from tests.shard1.test_isokinetic_reduced_model import _model

    model = _model(
        isokinetic=isokinetic,
        pulse_width_max_step_s=100e-6,
        pulse_width_interval_s=1 / 30,
        pulse_width_slew_formulation=DIRECT_CONSTRAINTS_FORMULATION,
    )
    configured = []
    monkeypatch.setattr(
        ConfigureVariables, "configure_new_variable",
        staticmethod(lambda name, *args, **kwargs: configured.append(name)),
    )
    for configure in model.state_configuration_functions:
        configure(None, None)
    assert model.nb_state == expected_states
    assert not any(name.startswith("pw_slew_") for name in configured)
    configured.clear()
    for configure in model.control_configuration_functions:
        configure(None, None)
    assert len(configured) == 4
    assert not any(name.startswith("pw_slew_") for name in configured)


@pytest.mark.parametrize("isokinetic", [False, True])
def test_reduced_physical_rhs_is_independent_of_carrier_and_increment(isokinetic, monkeypatch):
    from tests.shard1.test_isokinetic_reduced_model import _model

    monkeypatch.setattr(DynamicsFunctions, "get", staticmethod(lambda variable, vector: vector[variable]))
    plain = _model(isokinetic=isokinetic)
    lifted = _model(isokinetic=isokinetic, pulse_width_max_step_s=100e-6,
                    pulse_width_interval_s=1 / 30)
    # Record the exact PW passed to each muscle; its derivative deliberately
    # depends on PW to expose any accidental substitution of the carrier.
    for model in (plain, lifted):
        for muscle in model.muscles_dynamics_model:
            muscle.dynamics = lambda time, states, controls, *args, **kwargs: DynamicsEvaluation(
                dxdt=vertcat(controls, 2 * controls, 3 * controls, 4 * controls, 5 * controls)
            )
    names = [f"{state}_{muscle.muscle_name}" for muscle in plain.muscles_dynamics_model
             for state in muscle.name_dof] + ["theta", "omega"] + (["E_prod"] if isokinetic else [])
    control_names = [f"last_pulse_width_{m.muscle_name}" for m in plain.muscles_dynamics_model]
    nlp = SimpleNamespace(states={name: i for i, name in enumerate(names)},
                          controls={name: i for i, name in enumerate(control_names)},
                          dynamics_type=SimpleNamespace(ode_solver=OdeSolver.RK4()))
    physical_x = np.linspace(.1, .9, len(names))
    physical_u = np.array([200., 250., 300., 350.]) * 1e-6
    def rhs(model, x, u):
        return np.asarray(model.dynamics(0., DM(x), DM(u), DM(), DM(), DM(), nlp).dxdt).reshape(-1)
    expected = rhs(plain, physical_x, physical_u)
    for i, muscle in enumerate(lifted.muscles_dynamics_model):
        carrier, increment = auxiliary_keys(muscle.muscle_name)
        nlp.states[carrier] = len(names) + i
        nlp.controls[increment] = 4 + i
    for sign in (-1, 1):
        increments = sign * np.array([20., 40., 60., 80.]) * 1e-6
        actual = rhs(lifted, np.r_[physical_x, sign * np.ones(4)], np.r_[physical_u, increments])
        np.testing.assert_array_equal(actual[:len(names)], expected)
        np.testing.assert_allclose(actual[len(names):], increments * 30, atol=1e-18)


def test_cli_frequency_and_slew_are_independent():
    from examples.fes_multibody.cycling.cycling_pulse_width_mhe_acados_periodic import build_argument_parser
    parser = build_argument_parser()
    for frequency, step in ((50, None), (30, 100.), (50, 100.)):
        argv = ["--stimulations-per-cycle", str(frequency)]
        if step is not None:
            argv += ["--pulse-width-max-step-us", str(step)]
        args = parser.parse_args(argv)
        assert args.stimulations_per_cycle == frequency
        assert args.pulse_width_max_step_us == step


def test_acados_cannot_bypass_ipopt_cycle_one_without_a_certified_seed():
    from examples.fes_multibody.cycling.cycling_pulse_width_mhe_acados_periodic import (
        validate_acados_ipopt_initialization_policy,
    )

    args = SimpleNamespace(
        solver="acados",
        formulation="dynamic",
        disable_standard_ipopt_warmup=True,
        common_initial_solution=None,
    )
    with pytest.raises(ValueError, match="certified IPOPT solution of cycle 1"):
        validate_acados_ipopt_initialization_policy(args)

    args.common_initial_solution = Path("cycle-1-ipopt.npz")
    validate_acados_ipopt_initialization_policy(args)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"producer_solver": "acados", "cycles_per_window": 1}, "produced by IPOPT"),
        ({"producer_solver": "ipopt", "cycles_per_window": 2}, "exactly the IPOPT solution of cycle 1"),
    ],
)
def test_acados_external_seed_requires_ipopt_cycle_one_provenance(metadata, message):
    from examples.fes_multibody.cycling.cycling_pulse_width_mhe_acados_periodic import (
        reduced_internal_crank_velocity_guard_signature,
        validate_acados_ipopt_common_seed_provenance,
    )

    args = SimpleNamespace(
        solver="acados",
        mechanical_formulation="reduced",
        formulation="dynamic",
        reduced_internal_crank_velocity_guard="on",
        stimulations_per_cycle=30,
        wheel_qdot_regularization_target=-2.0 * np.pi,
        wheel_qdot_bound_margin=3.0,
        acados_wheel_qdot_fast_bound_margin=None,
        acados_wheel_qdot_slow_bound_margin=None,
    )
    seed = SimpleNamespace(metadata=metadata)
    with pytest.raises(ValueError, match=message):
        validate_acados_ipopt_common_seed_provenance(
            seed, args, Path("common.npz")
        )

    seed.metadata = {
        "producer_solver": "ipopt",
        "cycles_per_window": 1,
        **reduced_internal_crank_velocity_guard_signature(args),
    }
    validate_acados_ipopt_common_seed_provenance(
        seed, args, Path("common.npz")
    )

    seed.metadata.pop("reduced_internal_crank_velocity_guard")
    with pytest.raises(ValueError, match="exact ACADOS target guard"):
        validate_acados_ipopt_common_seed_provenance(
            seed, args, Path("common.npz")
        )


def test_common_seed_rejects_a_different_successive_control_bound():
    from examples.fes_multibody.cycling import (
        cycling_pulse_width_mhe_acados_periodic as periodic,
    )

    args = periodic.build_argument_parser().parse_args([])
    args.pulse_width_max_step_us = 100.0
    args.terminal_wheel_q_reference_mode = "absolute_initial"
    metadata = periodic._common_initial_solution_metadata(args)
    metadata["pulse_width_max_step_us"] = None
    seed = SimpleNamespace(metadata=metadata)
    with pytest.raises(ValueError, match="pulse_width_max_step_us"):
        periodic._validate_common_initial_solution_metadata(
            seed, args, Path("unconstrained-cycle-1.npz")
        )
