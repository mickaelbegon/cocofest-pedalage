"""Counterexamples and physical-condition contracts for the offline oracle."""

from copy import deepcopy
import json
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest
from scipy.optimize import linprog

from cocofest.optimization.conditional_capacity_feasibility import (
    ConditionalCapacitySlice, FrozenTaskDeficit, directional_points, mapping_from_program, summarize_slice,
)
from cocofest.optimization.task_reserve import TaskReserveCheckpoint
from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint


def _source(directory, *, force=.1, tau=.05, a=.8, history=.0002):
    directory.mkdir()
    model = directory / "model.json"
    model.write_text(json.dumps({"schema_version": 1, "muscles": {"Biceps": {"a_scale": 100.}}}))
    initial = {"Cn_Biceps": .3, "F_Biceps": force, "A_Biceps": 100*a,
               "Tau1_Biceps": tau, "Km_Biceps": .1, "theta": 4., "E_prod": 0.}
    states = {k: SimpleNamespace(init=np.array([[v, v, v]])) for k, v in initial.items()}
    bounds = {k: SimpleNamespace(min=np.array([[v, 0., 0.]]), max=np.array([[v, 100., 100.]]))
              for k, v in initial.items()}
    bounds["E_prod"].min[0, 2] = bounds["E_prod"].max[0, 2] = 1.
    program = SimpleNamespace(nlp=[SimpleNamespace(x_init=states, x_bounds=bounds,
        u_init={"last_pulse_width_Biceps": SimpleNamespace(init=np.array([[history]]))},
        u_bounds={"last_pulse_width_Biceps": SimpleNamespace(min=np.array([[.0001]]), max=np.array([[.0006]]))})],
        parameter_init={}, parameter_bounds={}, absolute_wheel_q_cycle_index=20)
    archive = directory / "prepared.npz"
    export_prepared_checkpoint(archive, program, completed_cycles=20, model_path=model)
    return TaskReserveCheckpoint.from_archive(archive, completed_cycles=20, model_path=model,
        task_context={"nominal_work_j": 1., "side": "left", "solver": "ipopt", "collocation_degree": 5})


def _slice(source):
    return ConditionalCapacitySlice.from_checkpoint(source, muscle_order=("Biceps",))


def test_same_A_different_fast_slow_or_history_is_a_different_condition(tmp_path):
    base = _slice(_source(tmp_path / "base"))
    for name, kwargs in (("fast", {"force": .2}), ("slow", {"tau": .06}), ("history", {"history": .0003})):
        alternative = _slice(_source(tmp_path / name, **kwargs))
        assert alternative.center == base.center
        assert alternative.condition_sha256 != base.condition_sha256
    altered_A = _slice(_source(tmp_path / "only_A", a=.79))
    assert altered_A.condition_sha256 == base.condition_sha256
    assert altered_A.checkpoint.prepared_problem_sha256 != base.checkpoint.prepared_problem_sha256
    assert json.loads(base.condition_json)["synthetic_capacity_states_are_reachable"] is False


def test_explicit_full_grid_keeps_both_sides_without_clipping_or_binary_search(tmp_path):
    slice_ = _slice(_source(tmp_path / "source"))
    points = directional_points(slice_, {"Biceps": [1.]}, [-.05, -.01, .01, .05])
    assert [p.offset for p in points] == [0., -.05, -.01, .01, .05]
    assert points[1].capacities == pytest.approx((.75,))
    assert points[-1].capacities == pytest.approx((.85,))
    with pytest.raises(ValueError, match="domain"):
        directional_points(slice_, {"Biceps": [1.]}, [.1])
    with pytest.raises(ValueError, match="infinity norm"):
        directional_points(slice_, {"Biceps": [.5]}, [.01])


def _toy_problem(*, initial_force=0., capacity=1., target=.6, control_min=0., tau=1.):
    """Exact one-period force-filter quadrature: F'=(A*u-F)/tau.

    W=F0*tau*(1-exp(-T/tau))+A*u*(T-tau*(1-exp(-T/tau))).
    Unit fatigue cost remains an irrelevant source objective for this test.
    """
    x = ca.SX.sym("x", 4)  # A, F0, stimulation amplitude, integrated force
    memory = tau*(1-np.exp(-1/tau))
    production = 1-memory
    graph = {"x": x, "g": x[3] - memory*x[1] - production*x[0]*x[2],
             "f": (1-x[0])**2}
    limits = {"lbx": [capacity, initial_force, control_min, target],
              "ubx": [capacity, initial_force, 1., target],
              "lbg": [0.], "ubg": [0.], "x0": [capacity, initial_force, .8, target]}
    original = deepcopy(limits)
    problem = FrozenTaskDeficit(graph=graph, limits=limits, capacity_indices=[0], capacity_coefficients=[1.],
        work_index=3, work_coefficient=1., nominal_work_j=target, center=[capacity])
    return problem, graph, limits, original, memory, production


def _exact_filter_lp(problem, force, memory, production, capacity, control_min=0.):
    solution = linprog([0., -1. / problem.nominal_work_j],
        A_eq=[[-production*capacity, 1.]], b_eq=[memory*force],
        bounds=[(control_min, 1.), (0., problem.nominal_work_j)], method="highs")
    if not solution.success:
        return None
    return np.array([capacity, force, *solution.x])


def test_numerical_counterexample_same_A_has_different_task_feasibility():
    deficits = []
    for force in (0., .8):
        problem, _, _, _, memory, production = _toy_problem(initial_force=force)
        vector = _exact_filter_lp(problem, force, memory, production, 1.)
        audit = problem.audit(vector, [1.], tolerance=1e-8)
        assert audit["phase_one_trajectory_validated"]
        deficits.append(audit)
    assert deficits[0]["observed_normalized_work_deficit"] == pytest.approx(1-(np.exp(-1))/.6)
    assert not deficits[0]["nominal_task_witnessed"]
    assert deficits[1]["nominal_task_witnessed"]
    assert deficits[1]["observed_normalized_work_deficit"] == pytest.approx(0.)
    assert all(not row["physiological_failure_certified"] for row in deficits)


def test_nonmonotone_feasibility_when_nonzero_minimum_stimulation_meets_exact_work():
    # Fixed fast state and exact work: A too small cannot supply the task;
    # A too large cannot reduce force enough under a positive device minimum.
    outcomes = []
    for capacity in (.2, .5, .9):
        problem, _, _, _, memory, production = _toy_problem(capacity=capacity, target=.4, control_min=.7, tau=.1)
        vector = _exact_filter_lp(problem, 0., memory, production, capacity, control_min=.7)
        outcomes.append(False if vector is None else problem.audit(vector, [capacity], tolerance=1e-8)["nominal_task_witnessed"])
    assert outcomes == [False, True, False]


def test_positive_locally_valid_deficit_never_proves_infeasibility():
    problem, _, _, _, memory, production = _toy_problem(initial_force=.8)
    # Deliberately supply a suboptimal feasible iterate although nominal work
    # IS achievable; status strings cannot fix this absence of a global bound.
    vector = [.9999999999999999, .8, 0., .8*memory]
    audit = problem.audit(vector, [1.], tolerance=1e-8)
    assert audit["phase_one_trajectory_validated"] and audit["observed_normalized_work_deficit"] > 0
    assert audit["outcome"] == "indeterminate"
    assert not audit["positive_deficit_is_infeasibility_proof"]


def test_auxiliary_copies_bounds_preserves_fatigue_cost_and_conditions_everything_else():
    problem, graph, limits, original, _, _ = _toy_problem(capacity=.8)
    varied = problem.bounds_for([.78], relax_work=True)
    assert limits == original
    assert varied["lbx"] == pytest.approx([.78, 0., 0., 0.])
    assert varied["ubx"] == pytest.approx([.78, 0., 1., .6])
    source_cost = ca.Function("source_f", [graph["x"]], [graph["f"]])
    phase_one = ca.Function("phase_one_f", [graph["x"]], [problem.auxiliary_graph["f"]])
    assert float(source_cost([.8, 0., 1., .3])) == pytest.approx(.04)
    assert float(phase_one([.8, 0., 1., .3])) == pytest.approx(.5)
    assert problem.auxiliary_graph["g"] is graph["g"]


def test_nominal_claim_requires_all_constraints_and_normalized_work():
    problem, _, _, _, _, _ = _toy_problem()
    audit = problem.audit([1., 0., 1., .6], [1.], tolerance=1e-8)
    assert audit["constraint_normalized_violation"] > .1
    assert audit["observed_normalized_work_deficit"] is None
    assert not audit["nominal_task_witnessed"]


def test_small_task_work_is_audited_relative_to_target_not_one_unit_floor():
    problem, _, _, _, memory, production = _toy_problem(capacity=.8, target=1e-5)
    work = .99e-5
    vector = [.8, 0., work / (.8*production), work]
    audit = problem.audit(vector, [.8], tolerance=1e-6)
    assert audit["phase_one_trajectory_validated"]
    assert audit["nominal_decision_normalized_violation"] < 1e-6
    assert audit["observed_normalized_work_deficit"] == pytest.approx(.01)
    assert not audit["nominal_task_witnessed"]


def test_symbolic_mapping_finds_scaled_global_offsets(tmp_path):
    slice_ = _slice(_source(tmp_path / "source"))
    x = ca.SX.sym("x", 6)
    nlp = SimpleNamespace(X_scaled=[ca.vertcat(x[4], x[0]), ca.vertcat(x[5], x[3])],
        states={"A_Biceps": SimpleNamespace(index=[0]), "E_prod": SimpleNamespace(index=[1])},
        x_scaling={"A_Biceps": SimpleNamespace(scaling=np.array([200.])),
                   "E_prod": SimpleNamespace(scaling=np.array([3.]))})
    program = SimpleNamespace(nlp=[nlp], ocp_solver=SimpleNamespace(nlp={"x": x}))
    assert mapping_from_program(program, slice_) == {
        "capacity_indices": [4], "capacity_coefficients": [2.], "work_index": 3, "work_coefficient": 3.}


def test_summary_does_not_convert_unknown_holes_to_a_boundary(tmp_path):
    slice_ = _slice(_source(tmp_path / "source"))
    points = directional_points(slice_, {"Biceps": [1.]}, [-.04, -.02, .02, .04])
    records = [{"sample_id": p.sample_id, "capacities": p.capacities, "condition_sha256": slice_.condition_sha256,
                "nominal_task_witnessed": i in (0, 1, 4), "physiological_failure_certified": False}
               for i, p in enumerate(points)]
    report = summarize_slice(slice_, points, records)
    assert report["directions"]["Biceps"]["unknown_between_witnesses"]
    assert report["directions"]["Biceps"]["boundary_estimate"] is None
    assert not report["activation_allowed"]
    records[2]["condition_sha256"] = "a_different_F_Tau1_Km_or_history"
    with pytest.raises(ValueError, match="full-state condition"):
        summarize_slice(slice_, points, records)
