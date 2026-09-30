"""Independent numerical equivalence checks for prescribed isokinetic motion."""

from types import SimpleNamespace

import numpy as np
from bioptim import DynamicsEvaluation, DynamicsFunctions, OdeSolver
from casadi import DM, Function, SX, jacobian, vertcat
from scipy.integrate import solve_ivp

from cocofest.dynamics.reduced_cycling import (
    PeriodicFourierSeries,
    ReducedCyclingDynamics,
    ReducedCyclingKinematics,
)
from cocofest.models.reduced_cycling_model import ReducedFesCyclingModel


NAMES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
OMEGA = -2.0 * np.pi
THETA0 = 0.37
TIME0 = 0.2


def _profile():
    kinematics = ReducedCyclingKinematics(
        theta_origin=0.0,
        direction=-1,
        winding_numbers=np.array([0.0, 0.0, 1.0]),
        periodic_residual=PeriodicFourierSeries(
            offset=np.zeros(3), cosine=np.zeros((3, 1)), sine=np.zeros((3, 1))
        ),
    )
    # M, g, c, four signed muscle coefficients and b_ext. Nonconstant
    # coefficients detect a wrong phase at an interior collocation stage.
    coefficients = PeriodicFourierSeries(
        offset=np.array([2.0, 3.0, 0.5, 1.0, -2.0, 0.5, 0.7, 0.25]),
        cosine=np.array([[0.0], [0.4], [0.1], [0.2], [-0.3], [0.15], [0.08], [0.02]]),
        sine=np.array([[0.0], [0.15], [0.05], [-0.1], [0.2], [0.07], [-0.05], [0.01]]),
    )
    geometry = PeriodicFourierSeries(
        offset=np.array([1.0, 1.1, 0.9, 1.05, 0.12, -0.08, 0.06, -0.1]),
        cosine=np.array([[0.06], [-0.03], [0.05], [0.02], [0.02], [0.01], [-0.01], [0.02]]),
        sine=np.array([[0.02], [0.04], [-0.02], [0.03], [0.01], [-0.01], [0.02], [-0.02]]),
    )
    return ReducedCyclingDynamics(
        kinematics=kinematics,
        coefficients=coefficients,
        muscle_geometry=geometry,
        muscle_names=NAMES,
        crank_torque_dof_index=2,
    )


class _CoupledMuscle:
    """Small smooth proxy that exposes every Hill input in its state rates."""

    name_dof = ["Cn", "F", "A", "Tau1", "Km"]
    nb_state = 5

    def __init__(self, name):
        self.muscle_name = name

    def dynamics(self, time, states, pulse_width, *args, **kwargs):
        fl = kwargs["force_length_relationship"]
        fv = kwargs["force_velocity_relationship"]
        passive = kwargs["passive_force_relationship"]
        cn, force, fatigue, tau, km = (states[i] for i in range(5))
        return DynamicsEvaluation(
            dxdt=vertcat(
                -0.8 * cn + 1000.0 * pulse_width,
                -0.2 * force + 0.3 * cn * fl * fv + 0.15 * passive,
                -0.1 * fatigue + 0.05 * force,
                -0.05 * tau + 0.01 * fatigue,
                -0.07 * km + 0.02 * tau,
            ),
            defects=None,
        )


def _model(mode):
    return ReducedFesCyclingModel(
        reduced_dynamics=_profile(),
        muscles_model=[_CoupledMuscle(name) for name in NAMES],
        isokinetic=True,
        isokinetic_omega=OMEGA,
        isokinetic_kinematics=mode,
        isokinetic_theta0=THETA0,
        isokinetic_time_origin=TIME0,
    )


def _nlp(mode):
    names = [f"{state}_{muscle}" for muscle in NAMES for state in _CoupledMuscle.name_dof]
    if mode == "states":
        names += ["theta", "omega"]
    names += ["E_prod"]
    controls = [f"last_pulse_width_{muscle}" for muscle in NAMES]
    return SimpleNamespace(
        states={name: i for i, name in enumerate(names)},
        controls={name: i for i, name in enumerate(controls)},
        dynamics_type=SimpleNamespace(ode_solver=OdeSolver.RK4()),
    )


def _rhs(model, mode, t, x, u):
    return model.dynamics(
        time=t,
        states=x,
        controls=u,
        parameters=DM(),
        algebraic_states=DM(),
        numerical_data_timeseries=DM(),
        nlp=_nlp(mode),
    ).dxdt


def _functions(monkeypatch):
    monkeypatch.setattr(DynamicsFunctions, "get", staticmethod(lambda index, vector: vector[index]))
    old = _model("states")
    new = _model("prescribed")
    t = SX.sym("t")
    x = SX.sym("x", 21)
    u = SX.sym("u", 4)
    theta = THETA0 + OMEGA * (t - TIME0)
    reconstructed = vertcat(x[:20], theta, OMEGA, x[20])
    old_rates = _rhs(old, "states", t, reconstructed, u)
    retained_rates = vertcat(old_rates[:20], old_rates[22])
    new_rates = _rhs(new, "prescribed", t, x, u)
    old_fun = Function("old_iso_rates", [t, x, u], [retained_rates])
    new_fun = Function("new_iso_rates", [t, x, u], [new_rates])
    wrt = vertcat(t, x, u)
    old_jac = Function("old_iso_jac", [t, x, u], [jacobian(retained_rates, wrt)])
    new_jac = Function("new_iso_jac", [t, x, u], [jacobian(new_rates, wrt)])
    return old_fun, new_fun, old_jac, new_jac


def _initial_state():
    x = np.zeros(21)
    for muscle_index in range(4):
        x[5 * muscle_index : 5 * muscle_index + 5] = [0.2, 4.0 + muscle_index, 0.8, 0.7, 0.9]
    return x


def test_prescribed_rhs_and_jacobian_match_reconstructed_states_at_stages(monkeypatch):
    old_fun, new_fun, old_jac, new_jac = _functions(monkeypatch)
    x = _initial_state()
    u = np.array([0.00017, 0.00023, 0.00031, 0.00019])
    for time in (TIME0, TIME0 + 0.007, TIME0 + 0.137, TIME0 + 0.629):
        np.testing.assert_allclose(np.asarray(new_fun(time, x, u)), np.asarray(old_fun(time, x, u)), rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(np.asarray(new_jac(time, x, u)), np.asarray(old_jac(time, x, u)), rtol=1e-10, atol=1e-10)


def test_prescribed_dense_cycle_matches_state_formulation(monkeypatch):
    _, new_fun, _, _ = _functions(monkeypatch)
    x0 = _initial_state()
    u = np.array([0.00017, 0.00023, 0.00031, 0.00019])
    old_x0 = np.concatenate((x0[:20], [THETA0, OMEGA], x0[20:]))
    times = np.linspace(TIME0, TIME0 + 1.0, 31)

    # The reference integrates its own theta/omega states, including their
    # numerical error; the prescribed variant does not carry these states.
    old_model = _model("states")
    monkeypatch.setattr(DynamicsFunctions, "get", staticmethod(lambda index, vector: vector[index]))
    t_sym = SX.sym("old_t")
    x_sym = SX.sym("old_x", 23)
    old_full = Function("old_full_iso_rhs", [t_sym, x_sym], [_rhs(old_model, "states", t_sym, x_sym, DM(u))])
    reference = solve_ivp(
        lambda t, x: np.asarray(old_full(t, x)).reshape(-1),
        (times[0], times[-1]), old_x0, t_eval=times, rtol=1e-10, atol=1e-11,
    )
    prescribed = solve_ivp(
        lambda t, x: np.asarray(new_fun(t, x, u)).reshape(-1),
        (times[0], times[-1]), x0, t_eval=times, rtol=1e-10, atol=1e-11,
    )
    assert reference.success and prescribed.success
    np.testing.assert_allclose(prescribed.y, reference.y[np.r_[:20, 22]], rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(reference.y[20], THETA0 + OMEGA * (times - TIME0), atol=2e-10)
    np.testing.assert_allclose(reference.y[21], OMEGA, atol=2e-10)
