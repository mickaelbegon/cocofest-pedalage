"""Physical and numerical contract for PW-state/rate-control RHO."""
from types import SimpleNamespace

import numpy as np
import pytest
from casadi import DM
from bioptim import (BoundsList, ConfigureVariables, DynamicsEvaluation,
                     DynamicsFunctions, DynamicsOptions, InitialGuessList,
                     ObjectiveFcn, ObjectiveList, OdeSolver, OptimalControlProgram,
                     SolutionMerge, Solver, StateDynamics, VariableScalingList, Node)

from cocofest.optimization.pulse_width_rate import (
    physical_pulse_controls, promote_pulse_width_to_state, rate_configuration,
    rate_rhs, sampled_pulse_width, validate_rate_mode,
)


class _RateToy(StateDynamics):
    def __init__(self, intervals):
        super().__init__()
        self.muscles_dynamics_model = [SimpleNamespace(muscle_name="m")]
        self.pulse_width_interval_s = 1 / intervals
        self.pulse_width_max_rate_s_per_s = 100e-6 * intervals

    @property
    def name(self):
        return "rate_toy"

    @property
    def name_dofs(self):
        return ["pw"]

    @property
    def state_configuration_functions(self):
        return rate_configuration(self.muscles_dynamics_model, states=True)

    @property
    def control_configuration_functions(self):
        return rate_configuration(self.muscles_dynamics_model, states=False)

    @property
    def algebraic_configuration_functions(self):
        return []

    @property
    def extra_configuration_functions(self):
        return []

    def serialize(self):
        return _RateToy, {"intervals": round(1 / self.pulse_width_interval_s)}

    def dynamics(self, time, states, controls, parameters, algebraic_states, numerical_data_timeseries, nlp):
        rhs = rate_rhs(self, controls, nlp)
        return DynamicsEvaluation(dxdt=rhs, defects=nlp.states_dot.scaled.cx * nlp.dt - rhs * nlp.dt)


@pytest.mark.parametrize("intervals", [30, 50])
def test_rate_state_radau5_bounds_and_integral(intervals):
    model = _RateToy(intervals)
    xb, xi, xs = BoundsList(), InitialGuessList(), VariableScalingList()
    ub, ui, us = BoundsList(), InitialGuessList(), VariableScalingList()
    ub.add("last_pulse_width_m", min_bound=[131e-6], max_bound=[600e-6])
    ui.add("last_pulse_width_m", initial_guess=[300e-6])
    us.add("last_pulse_width_m", scaling=[.0025])
    ode = OdeSolver.COLLOCATION(polynomial_degree=5, method="radau")
    ub, ui, us = promote_pulse_width_to_state(
        model, xb, xi, xs, ub, ui, us, n_shooting=intervals, scale=.0025, ode_solver=ode
    )
    # Nontrivial first PW also represents an actual inter-window carry.
    xb["last_pulse_width_m"].min[:, 0] = 300e-6
    xb["last_pulse_width_m"].max[:, 0] = 300e-6
    target = np.full((1, intervals), 131e-6)
    target[:, :intervals // 2] = 600e-6
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="last_pulse_width_m",
                   target=target, node=Node.ALL_SHOOTING, weight=1e10, quadratic=True)
    ocp = OptimalControlProgram(model, intervals, 1., dynamics=DynamicsOptions(ode_solver=ode),
        x_bounds=xb, x_init=xi, x_scaling=xs, u_bounds=ub, u_init=ui, u_scaling=us,
        objective_functions=objectives, use_sx=True, n_threads=1)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_tol(1e-9)
    solution = ocp.solve(solver)
    assert solution.status == 0
    controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
    states = solution.decision_states(to_merge=SolutionMerge.NODES)
    pw = np.asarray(states["last_pulse_width_m"])[0, ::6]
    rate = np.asarray(controls["pulse_width_rate_m"])[0]
    assert set(controls) == {"pulse_width_rate_m"}
    assert pw[0] == pytest.approx(300e-6, abs=1e-12)
    np.testing.assert_allclose(np.diff(pw), rate / intervals, atol=1e-10)
    # IPOPT's default bound relaxation is expressed in scaled variables.
    assert max(abs(rate)) <= model.pulse_width_max_rate_s_per_s + 2e-9
    assert max(abs(np.diff(pw))) > 99e-6
    assert np.min(pw) >= 131e-6 - 1e-10
    assert np.max(pw) <= 600e-6 + 1e-10
    # The applied commands are sampled at interval starts, excluding terminal.
    traces = physical_pulse_controls(states, controls, shooting_stride=6)
    np.testing.assert_array_equal(traces["last_pulse_width_m"], pw[None, :-1])


def _reduced_rate_model(mode):
    from tests.shard1.test_isokinetic_reduced_model import _reduced_dynamics, MUSCLE_NAMES
    from cocofest.models.ding2007.ding2007_with_fatigue_periodic_node import (
        DingModelPulseWidthFrequencyWithFatiguePeriodicNode,
    )
    from cocofest.models.reduced_cycling_model import ReducedFesCyclingModel
    muscles = [DingModelPulseWidthFrequencyWithFatiguePeriodicNode(
        muscle_name=name, stim_time=[0., 1 / 30], stim_interval=1 / 30,
    ) for name in MUSCLE_NAMES]
    return ReducedFesCyclingModel(
        reduced_dynamics=_reduced_dynamics(), muscles_model=muscles,
        activate_force_length_relationship=False, activate_force_velocity_relationship=False,
        activate_passive_force_relationship=False, pulse_width_interval_s=1 / 30,
        pulse_width_control_mode=mode,
        pulse_width_max_rate_s_per_s=.003 if mode == "rate_state" else None,
    )


def test_reduced_rate_rhs_preserves_actual_ding_calcium_force_and_fatigue(monkeypatch):
    monkeypatch.setattr(DynamicsFunctions, "get", staticmethod(lambda variable, vector: vector[variable]))
    plain, rate_model = _reduced_rate_model("direct"), _reduced_rate_model("rate_state")
    names = [f"{key}_{m.muscle_name}" for m in plain.muscles_dynamics_model for key in m.name_dof]
    names += ["theta", "omega"]
    pulse_keys = [f"last_pulse_width_{m.muscle_name}" for m in plain.muscles_dynamics_model]
    state = np.r_[np.concatenate([[.3, 10., m.a_scale * .7, m.tau1_rest, m.km_rest]
                                  for m in plain.muscles_dynamics_model]), -.2, -6.28]
    pw = np.array([200., 250., 300., 350.]) * 1e-6
    rate = np.array([.003, -.002, .001, -.001])
    base_nlp = SimpleNamespace(states={key: i for i, key in enumerate(names)},
        controls={key: i for i, key in enumerate(pulse_keys)},
        dynamics_type=SimpleNamespace(ode_solver=OdeSolver.RK4()))
    rate_nlp = SimpleNamespace(states={key: i for i, key in enumerate(names + pulse_keys)},
        controls={f"pulse_width_rate_{m.muscle_name}": i for i, m in enumerate(plain.muscles_dynamics_model)},
        dynamics_type=SimpleNamespace(ode_solver=OdeSolver.RK4()))
    start = .7
    for elapsed in [0., .001, .02, 1 / 30]:
        time = DM(start + elapsed)
        data = DM([1.2, start])
        direct = np.asarray(plain.dynamics(time, DM(state), DM(pw), DM(), DM(), data, base_nlp).dxdt).ravel()
        actual = np.asarray(rate_model.dynamics(time, DM(np.r_[state, pw + elapsed * rate]),
                            DM(rate), DM(), DM(), data, rate_nlp).dxdt).ravel()
        np.testing.assert_allclose(actual[:22], direct, rtol=1e-13, atol=1e-12)
        np.testing.assert_array_equal(actual[22:], rate)


def test_rate_configuration_contains_pw_states_and_only_rate_controls(monkeypatch):
    model = _reduced_rate_model("rate_state")
    configured = []
    monkeypatch.setattr(ConfigureVariables, "configure_new_variable",
        staticmethod(lambda name, *args, **kwargs: configured.append(name)))
    for configure in model.state_configuration_functions:
        configure(None, None)
    assert len(configured) == model.nb_state == 26
    assert sum(k.startswith("last_pulse_width_") for k in configured) == 4
    configured.clear()
    for configure in model.control_configuration_functions:
        configure(None, None)
    assert len(configured) == 4
    assert all(k.startswith("pulse_width_rate_") for k in configured)
    constructor, kwargs = model.serialize()
    assert constructor(**kwargs).pulse_width_max_rate_s_per_s == .003


@pytest.mark.parametrize("invalid", [None, True, 0, -1, float("inf"), float("nan")])
def test_invalid_rate_bounds(invalid):
    with pytest.raises(ValueError):
        validate_rate_mode("rate_state", invalid)


def test_rate_rejects_missing_interval_local_ding_timing():
    from tests.shard1.test_isokinetic_reduced_model import _model
    with pytest.raises(ValueError, match="periodic-node"):
        _model(pulse_width_control_mode="rate_state", pulse_width_max_rate_s_per_s=.003,
               pulse_width_interval_s=1 / 30)


def test_right_endpoint_sampling_does_not_select_next_pulse():
    # Radau-5 evaluates exactly at t[k+1]; the old interval data must still be used.
    assert sampled_pulse_width(400e-6, .003, 1 / 30, 0.) == pytest.approx(300e-6)
