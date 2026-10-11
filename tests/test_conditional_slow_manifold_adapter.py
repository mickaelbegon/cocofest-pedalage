"""The offline slow-state binding must never build synthetic A-only states."""

import json
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from cocofest.optimization.conditional_slow_manifold_adapter import SlowManifoldInitialStateAdapter
from cocofest.optimization.ding_slow_manifold import DingSlowFatigueManifold
from cocofest.optimization.task_reserve import TaskReserveCheckpoint
from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint


def _prepared(tmp_path, *, force=.2, off_manifold=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifold = DingSlowFatigueManifold(100., .05, .1, -.4, 2.1e-6, 1.9e-6)
    a, tau1, km = manifold.states_from_capacity_ratio(.8)
    if off_manifold:
        tau1 += .001
    source = {"A_Biceps": a, "Tau1_Biceps": tau1, "Km_Biceps": km,
              "Cn_Biceps": .3, "F_Biceps": force, "E_prod": 0.}
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"schema_version": 1, "muscles": {"Biceps": {
        "a_scale": 100., "alpha_a": -.4, "alpha_tau1": 2.1e-6,
        "alpha_km": 1.9e-6, "tau1_rest": .05, "km_rest": .1}}}))
    states = {key: SimpleNamespace(init=np.array([[value, value, value]])) for key, value in source.items()}
    bounds = {key: SimpleNamespace(min=np.array([[value, 0., 0.]]),
                                   max=np.array([[value, 100., 100.]])) for key, value in source.items()}
    bounds["E_prod"].min[0, 2] = bounds["E_prod"].max[0, 2] = 1.
    nlp = SimpleNamespace(x_init=states, x_bounds=bounds,
                          u_init={"last_pulse_width_Biceps": SimpleNamespace(init=np.array([[.0002]]))},
                          u_bounds={"last_pulse_width_Biceps": SimpleNamespace(
                              min=np.array([[.0001]]), max=np.array([[.0006]]))})
    program = SimpleNamespace(nlp=[nlp], parameter_init={}, parameter_bounds={},
                              absolute_wheel_q_cycle_index=20)
    archive = tmp_path / "prepared.npz"
    export_prepared_checkpoint(archive, program, completed_cycles=20, model_path=model)
    checkpoint = TaskReserveCheckpoint.from_archive(archive, completed_cycles=20, model_path=model,
        task_context={"nominal_work_j": 1., "side": "left", "solver": "ipopt", "collocation_degree": 5})

    x = ca.SX.sym("x", 8)
    # The symbolic decision ordering differs from the physiological ordering.
    nlp.X_scaled = [ca.vertcat(x[4], x[0], x[6], x[1], x[2], x[3])]
    nlp.states = {key: SimpleNamespace(index=[index]) for index, key in enumerate(source)}
    scales = {"A_Biceps": 100., "Tau1_Biceps": .05, "Km_Biceps": .1,
              "Cn_Biceps": 1., "F_Biceps": 1., "E_prod": 1.}
    nlp.x_scaling = {key: SimpleNamespace(scaling=np.array([value])) for key, value in scales.items()}
    indices = [4, 0, 6, 1, 2, 3]
    lbx = np.full(8, -10., dtype=float)
    ubx = np.full(8, 10., dtype=float)
    x0 = np.zeros(8, dtype=float)
    for (key, value), index in zip(source.items(), indices):
        lbx[index] = ubx[index] = x0[index] = value / scales[key]
    program.ocp_solver = SimpleNamespace(nlp={"x": x, "g": x[7] - x[1], "f": x[4] ** 2},
        limits={"lbx": lbx, "ubx": ubx, "lbg": np.array([0.]),
                "ubg": np.array([0.]), "x0": x0})
    return checkpoint, program, {"Biceps": manifold}


def test_coupled_initial_states_and_all_other_bounds_unchanged(tmp_path):
    checkpoint, program, manifolds = _prepared(tmp_path)
    binding = SlowManifoldInitialStateAdapter.from_program(
        checkpoint=checkpoint, program=program, muscle_order=("Biceps",), manifolds=manifolds)
    assert binding.indices == ((4, 0, 6),)
    assert binding.center == pytest.approx((.8,))
    candidate = binding.bounds_for((.7,))
    expected = manifolds["Biceps"].states_from_capacity_ratio(.7)
    for physical, index, scale in zip(expected, binding.indices[0], binding.coefficients[0]):
        assert candidate["lbx"][index] * scale == pytest.approx(physical)
        assert candidate["ubx"][index] * scale == pytest.approx(physical)
        assert candidate["x0"][index] * scale == pytest.approx(physical)
    unchanged = set(range(8)) - set(binding.indices[0])
    for key in ("lbx", "ubx", "x0"):
        np.testing.assert_array_equal(candidate[key][list(unchanged)], program.ocp_solver.limits[key][list(unchanged)])
    for key in ("lbg", "ubg"):
        np.testing.assert_array_equal(candidate[key], program.ocp_solver.limits[key])
    assert candidate["lbx"] is not binding.limits["lbx"]
    assert json.loads(binding.condition_json)["fast_states_mechanics_history_and_other_bounds_conditioned"]


def test_off_manifold_source_is_rejected(tmp_path):
    checkpoint, program, manifolds = _prepared(tmp_path, off_manifold=True)
    with pytest.raises(ValueError, match="off the exact fatigue manifold"):
        SlowManifoldInitialStateAdapter.from_program(
            checkpoint=checkpoint, program=program, muscle_order=("Biceps",), manifolds=manifolds)


def test_mismatched_model_or_binding_is_rejected(tmp_path):
    checkpoint, program, manifolds = _prepared(tmp_path)
    wrong = {"Biceps": DingSlowFatigueManifold(100., .05, .1, -.41, 2.1e-6, 1.9e-6)}
    with pytest.raises(ValueError, match="disagrees with declared model"):
        SlowManifoldInitialStateAdapter.from_program(
            checkpoint=checkpoint, program=program, muscle_order=("Biceps",), manifolds=wrong)
    program.ocp_solver.limits["lbx"][4] = .79
    with pytest.raises(ValueError, match="not the mapped fixed source"):
        SlowManifoldInitialStateAdapter.from_program(
            checkpoint=checkpoint, program=program, muscle_order=("Biceps",), manifolds=manifolds)


def test_candidate_domain_and_source_context_are_enforced(tmp_path):
    checkpoint, program, manifolds = _prepared(tmp_path)
    binding = SlowManifoldInitialStateAdapter.from_program(
        checkpoint=checkpoint, program=program, muscle_order=("Biceps",), manifolds=manifolds)
    for candidate in ((-0.1,), (1.1,), (float("nan"),), ()):
        with pytest.raises(ValueError, match="physical capacity"):
            binding.bounds_for(candidate)
    checkpoint_2, program_2, manifolds_2 = _prepared(tmp_path / "other", force=.3)
    binding_2 = SlowManifoldInitialStateAdapter.from_program(
        checkpoint=checkpoint_2, program=program_2, muscle_order=("Biceps",), manifolds=manifolds_2)
    assert binding.condition_sha256 != binding_2.condition_sha256
