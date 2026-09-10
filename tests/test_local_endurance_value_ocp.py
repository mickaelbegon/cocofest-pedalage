"""Symbolic/binding checks; the tiny dynamics are not a physiological trial."""

from dataclasses import replace
from pathlib import Path
import os
import sys

import numpy as np
import pytest
from casadi import Function, SX, gradient, hessian
from bioptim import (
    BoundsList, ConfigureVariables, ConstraintList, DynamicsEvaluation, DynamicsOptions,
    InitialGuessList, ObjectiveFcn, ObjectiveList, OptimalControlProgram, SolutionMerge,
    Solver, StateDynamics,
)

from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.local_endurance_value import (
    LocalEnduranceCoordinates, OracleEvaluation, fit_local_endurance_value,
)
from cocofest.optimization.local_endurance_value_ocp import (
    LOCAL_ENDURANCE_PARAMETER_KEY, LocalEnduranceValueBinding,
)


def _coordinates(count=1):
    parameter = DingPulseWidthParameters(
        DingFatigueParameters(1200., .060601, .137, -1.4, 2.1e-5, 1.9e-5, 445.5),
        tauc=.011, tau2=.001, pd0=.000131405, pdt=.000194138, pulse_width_max=.0006,
    )
    return LocalEnduranceCoordinates.from_state(
        np.asarray([[.16295396, 20., 1150., .078, .15]] * count), (parameter,) * count,
        force_scale=100.,
    )


class _AnalyticValueOracle:
    """Known linear value used solely to audit the binding/compiled updates."""
    horizon_cycles = 10
    moment_scale = 1.
    softmin_temperature = .05
    margin_target = .1
    penalty_temperature = .05
    moment_tolerance = 1e-8

    def __init__(self, coordinates, direction):
        self.coordinates = coordinates
        self.direction = direction

    def evaluate(self, xi):
        self.coordinates.decode(xi)
        value = 2 + self.direction * np.sum(xi[len(xi)//2:] - self.coordinates.anchor[len(xi)//2:])
        return OracleEvaluation("complete", float(value), 1., 1., 1)


def _fit(coordinates, direction=1.):
    m = len(coordinates.parameters)
    fit = fit_local_endurance_value(
        _AnalyticValueOracle(coordinates, direction), trust_radius=np.r_[np.full(m, .005), np.full(m, .1)],
        absolute_tolerance=1e-9, relative_tolerance=0., ranking_tolerance=1e-8,
    )
    assert fit.accepted, fit.audit.reason
    return fit


@pytest.mark.parametrize("use_sx", [True, False])
def test_symbolic_polynomial_derivatives_context_and_constant_dimension(use_sx):
    coordinates = _coordinates(4)
    fit = _fit(coordinates)
    binding = LocalEnduranceValueBinding(fit, coordinates, [f"m{i}" for i in range(4)],
                                         weight=3., use_sx=use_sx)
    state = coordinates.decode(coordinates.anchor)
    value, trust, context, xi = binding.function(state.ravel(), binding.values)
    assert float(value) == pytest.approx(3 * fit.model.evaluate(coordinates.anchor))
    assert binding.parameter_size == 58
    assert np.asarray(trust).size == 16
    assert np.asarray(context).size == 12
    np.testing.assert_allclose(np.asarray(context), 0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(xi).ravel(), coordinates.anchor)
    x = SX.sym("x", 20)
    expr = binding.function(x, binding.values)[0]
    derivatives = Function("local_derivatives", [x], [gradient(expr, x), hessian(expr, x)[0]])
    jac, hes = derivatives(state.ravel())
    expected = np.zeros(20)
    expected[1::5] = .03
    np.testing.assert_allclose(np.asarray(jac).ravel(), expected, atol=1e-12)
    np.testing.assert_allclose(hes, 0, atol=1e-12)
    assert binding.audit_terminal_state(state)["valid"]
    changed = state.copy()
    changed[0, 0] += 1e-5
    assert not binding.audit_terminal_state(changed)["valid"]
    changed = state.copy()
    changed[0, 1] = 35.
    assert not binding.audit_terminal_state(changed)["valid"]


def test_rejected_fit_and_changed_coordinate_context_are_never_bound():
    coordinates = _coordinates()
    fit = _fit(coordinates)
    with pytest.raises(ValueError, match="accepted"):
        LocalEnduranceValueBinding(replace(fit, accepted=False), coordinates, ["m0"])
    state = coordinates.decode(coordinates.anchor)
    state[0, 0] += 1e-5
    changed = LocalEnduranceCoordinates.from_state(state, coordinates.parameters, force_scale=100.)
    np.testing.assert_allclose(changed.anchor, coordinates.anchor)
    with pytest.raises(ValueError, match="context"):
        LocalEnduranceValueBinding(fit, changed, ["m0"])


class _ForceIntegrator(StateDynamics):
    """F'=u, other states constant: makes the value's effect exactly testable."""
    def __init__(self):
        super().__init__()

    def serialize(self):
        return _ForceIntegrator, {}

    @property
    def name(self):
        return "synthetic_local_value_binding_test"

    @property
    def name_dofs(self):
        return ["dummy"]

    @property
    def state_configuration_functions(self):
        def configure(ocp, nlp):
            for key in ("Cn", "F", "A", "Tau1", "Km"):
                ConfigureVariables.configure_new_variable(f"{key}_m0", [f"{key}_m0"],
                                                          ocp, nlp, as_states=True)
        return [configure]

    @property
    def control_configuration_functions(self):
        return [lambda ocp, nlp: ConfigureVariables.configure_new_variable(
            "drive", ["drive"], ocp, nlp, as_controls=True)]

    @property
    def algebraic_configuration_functions(self):
        return []

    @property
    def extra_configuration_functions(self):
        return []

    def dynamics(self, time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp):
        from casadi import vertcat
        return DynamicsEvaluation(dxdt=vertcat(0*states[0], controls[0], 0*states[2:]), defects=None)


def _tiny_ocp(binding, coordinates):
    initial = coordinates.decode(coordinates.anchor)[0]
    x_bounds, x_init = BoundsList(), InitialGuessList()
    for key, value in zip(("Cn", "F", "A", "Tau1", "Km"), initial):
        lo, hi = (.5*value, 1.5*value)
        x_bounds[f"{key}_m0"] = [[value, lo, lo]], [[value, hi, hi]]
        x_init[f"{key}_m0"] = [value]
    u_bounds, u_init = BoundsList(), InitialGuessList()
    u_bounds["drive"] = [-50.], [50.]
    u_init["drive"] = [0.]
    objectives, constraints = ObjectiveList(), ConstraintList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="drive", weight=.01)
    binding.add_penalties(objectives, constraints)
    ocp = OptimalControlProgram(
        _ForceIntegrator(), 2, 1., dynamics=DynamicsOptions(), objective_functions=objectives,
        constraints=constraints, x_bounds=x_bounds, x_init=x_init, u_bounds=u_bounds, u_init=u_init,
        use_sx=True, n_threads=1, **binding.parameter_options(),
    )
    binding.attach(ocp)
    return ocp


def _ma57_solver():
    explicit = os.environ.get("IPOPT_HSL_LIBRARY")
    if explicit:
        library = Path(explicit)
    else:
        candidates = [p for p in [Path(sys.prefix)/"lib/libhsl.so",
                                  *sorted((Path(sys.prefix)/"opt/libhsl").glob("*/lib/libhsl.so"))]
                      if p.is_file()]
        if len(candidates) != 1:
            pytest.skip("Set IPOPT_HSL_LIBRARY to a working MA57 library for the compiled integration test.")
        library = candidates[0]
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_linear_solver("ma57")
    solver.set_option_unsafe(str(library), "hsllib")
    solver.set_print_level(0)
    solver.set_c_compile(True)
    return solver


def test_compiled_ma57_reuse_updates_objective_and_optimal_terminal_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    coordinates = _coordinates()
    fit = _fit(coordinates, 1.)
    binding = LocalEnduranceValueBinding(fit, coordinates, ["m0"], weight=10.)
    ocp = _tiny_ocp(binding, coordinates)
    solver = _ma57_solver()
    first = ocp.solve(solver=solver)
    assert first.status == 0
    binding.record_solution(ocp, first)
    compiled, nlp = ocp.ocp_solver.shaked_ocp_solver, ocp.nlp[0]
    c_before = (tmp_path/"nlp.c").stat().st_mtime_ns
    buffer = ocp.parameter_bounds[LOCAL_ENDURANCE_PARAMETER_KEY].min
    previous = binding.values.copy()
    with pytest.raises(ValueError, match="accepted"):
        binding.update(ocp, replace(fit, accepted=False), coordinates, weight=10.)
    np.testing.assert_array_equal(binding.values, previous)
    np.testing.assert_array_equal(buffer.ravel(), previous)
    binding.update(ocp, _fit(coordinates, -1.), coordinates, weight=10.)
    second = ocp.solve(solver=solver)
    assert second.status == 0
    binding.record_solution(ocp, second)
    first_force = first.decision_states(to_merge=SolutionMerge.NODES)["F_m0"][0, -1]
    second_force = second.decision_states(to_merge=SolutionMerge.NODES)["F_m0"][0, -1]
    assert first_force == pytest.approx(15., abs=2e-4)
    assert second_force == pytest.approx(25., abs=2e-4)
    assert ocp.nlp[0] is nlp
    assert ocp.ocp_solver.shaked_ocp_solver is compiled
    assert ocp.parameter_bounds[LOCAL_ENDURANCE_PARAMETER_KEY].min is buffer
    assert (tmp_path/"nlp.c").stat().st_mtime_ns == c_before
    assert binding.summary()["same_compiled_solver_verified"]
    assert binding.summary()["numeric_update_count"] == 1
    with pytest.raises(ValueError, match="same solution"):
        binding.record_solution(ocp, second)
    # Numerical context must also remain dynamic, not just polynomial slopes.
    # Shift initial Cn/Tau1 consistently with the constant synthetic dynamics.
    changed_state = coordinates.decode(coordinates.anchor)
    changed_state[0, 0] += 1e-5
    changed_state[0, 3] += .001
    changed_coordinates = LocalEnduranceCoordinates.from_state(
        changed_state, coordinates.parameters, force_scale=200.,
    )
    x_init = InitialGuessList()
    for key, value in zip(("Cn", "F", "A", "Tau1", "Km"), changed_state[0]):
        bounds = ocp.nlp[0].x_bounds[f"{key}_m0"]
        bounds.min[0, 0] = bounds.max[0, 0] = value
        x_init[f"{key}_m0"] = [value]
    ocp.update_initial_guess(x_init=x_init)
    binding.update(ocp, _fit(changed_coordinates, -1.), changed_coordinates)
    third = ocp.solve(solver=solver)
    assert third.status == 0
    audit = binding.record_solution(ocp, third)
    assert audit["maximum_context_residual"] < 1e-9
    third_force = third.decision_states(to_merge=SolutionMerge.NODES)["F_m0"][0, -1]
    assert third_force == pytest.approx(22.5, abs=2e-4)
    assert ocp.nlp[0] is nlp
    assert ocp.ocp_solver.shaked_ocp_solver is compiled
    assert ocp.parameter_bounds[LOCAL_ENDURANCE_PARAMETER_KEY].min is buffer
    assert (tmp_path/"nlp.c").stat().st_mtime_ns == c_before
    assert binding.summary()["successful_compiled_solves"] == 3
    assert binding.summary()["numeric_update_count"] == 2
