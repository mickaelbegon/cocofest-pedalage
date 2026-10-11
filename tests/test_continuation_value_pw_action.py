from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.continuation_value_pw_action import (
    DirectPulseWidthAction,
    direct_pw_continuation_plan,
    execute_direct_pw_branch,
    predeclared_pw_actions,
)
from cocofest.optimization.task_reserve import DEFAULT_CONSTRAINT_GROUPS, TaskReserveCheckpoint
from cocofest.optimization.task_reserve_probe_adapter import IndependentAudit
from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint


def _bound(lower, upper):
    return SimpleNamespace(min=np.asarray(lower, dtype=float), max=np.asarray(upper, dtype=float))


def _program(*, auxiliary=False):
    controls = {
        "last_pulse_width_Biceps": SimpleNamespace(init=np.array([[.2, .3, .4]])),
        "last_pulse_width_Triceps": SimpleNamespace(init=np.array([[.5, .4, .3]])),
    }
    if auxiliary:
        controls["pulse_width_rate_Biceps"] = SimpleNamespace(init=np.zeros((1, 3)))
    return SimpleNamespace(nlp=[SimpleNamespace(
        u_init=controls,
        u_bounds={
            "last_pulse_width_Biceps": _bound([[.1, .1, .1]], [[.6, .6, .6]]),
            "last_pulse_width_Triceps": _bound([[.2, .2, .2]], [[.8, .8, .8]]),
        },
    )])


def test_direct_pw_action_creates_soft_offset_without_changing_bounds():
    program = _program()
    action = DirectPulseWidthAction(("Biceps", "Triceps"), (1 / np.sqrt(2), -1 / np.sqrt(2)), .1)
    original = {key: (value.min.copy(), value.max.copy(), program.nlp[0].u_init[key].init.copy())
                for key, value in program.nlp[0].u_bounds.items()}
    reference = np.vstack([program.nlp[0].u_init[f"last_pulse_width_{name}"].init[0]
                           for name in ("Biceps", "Triceps")])
    target, receipt = action.target_from_reference(program, reference)
    assert receipt["task_constraints_unchanged"] and receipt["tracking_activation"] == .1
    assert receipt["controls"]["Biceps"]["constant_offset_s"] > 0.
    assert receipt["controls"]["Triceps"]["constant_offset_s"] < 0.
    assert target.shape == reference.shape
    assert np.all(np.isfinite(target))
    for key, values in program.nlp[0].u_bounds.items():
        lower, upper, initial = original[key]
        np.testing.assert_allclose(program.nlp[0].u_bounds[key].min, lower)
        np.testing.assert_allclose(program.nlp[0].u_bounds[key].max, upper)
        np.testing.assert_allclose(program.nlp[0].u_init[key].init, initial)


def test_direct_pw_action_refuses_auxiliary_pw_representation():
    with pytest.raises(ValueError, match="auxiliary"):
        DirectPulseWidthAction(("Biceps", "Triceps"), (1 / np.sqrt(2), -1 / np.sqrt(2)), .1).target_from_reference(
            _program(auxiliary=True), np.array([[.2, .3, .4], [.5, .4, .3]]))


def test_direct_pw_action_accepts_one_phase_bioptim_style_containers():
    program = _program()
    program.nlp[0].u_init = [program.nlp[0].u_init]
    program.nlp[0].u_bounds = [program.nlp[0].u_bounds]
    action = DirectPulseWidthAction(("Biceps", "Triceps"), (1 / np.sqrt(2), -1 / np.sqrt(2)), .05)
    reference = np.vstack([program.nlp[0].u_init[0][f"last_pulse_width_{name}"].init[0]
                           for name in ("Biceps", "Triceps")])
    target, receipt = action.target_from_reference(program, reference)
    assert receipt["controls"]["Biceps"]["maximum_absolute_target_change_s"] > 0.
    assert target.shape == reference.shape
    assert np.any(program.nlp[0].u_bounds[0]["last_pulse_width_Biceps"].min
                  != program.nlp[0].u_bounds[0]["last_pulse_width_Biceps"].max)


def test_direct_pw_action_refuses_nonunit_direction():
    with pytest.raises(ValueError, match="unit Euclidean"):
        DirectPulseWidthAction(("Biceps", "Triceps"), (1., 1.), .1)


def test_direct_pw_target_accepts_compact_bounds_without_rewriting_them():
    program = _program()
    for key in program.nlp[0].u_init:
        program.nlp[0].u_init[key].init = np.tile(program.nlp[0].u_init[key].init, (1, 10))
    for key in program.nlp[0].u_bounds:
        program.nlp[0].u_bounds[key] = _bound([[.1, .1, .1]], [[.6, .6, .6]])
    action = DirectPulseWidthAction(("Biceps", "Triceps"), (1 / np.sqrt(2), -1 / np.sqrt(2)), .05)
    reference = np.vstack([program.nlp[0].u_init[f"last_pulse_width_{name}"].init[0]
                           for name in ("Biceps", "Triceps")])
    target, _ = action.target_from_reference(program, reference)
    assert target.shape == reference.shape
    assert program.nlp[0].u_bounds["last_pulse_width_Biceps"].min.shape == (1, 3)


def test_action_table_separates_train_holdout_and_never_updates_fatigue_weights():
    directions = {"d1": [1 / np.sqrt(2), -1 / np.sqrt(2)],
                  "d2": [-1 / np.sqrt(2), 1 / np.sqrt(2)],
                  "mixed": [1 / np.sqrt(2), 1 / np.sqrt(2)]}
    actions = predeclared_pw_actions(("Biceps", "Triceps"), directions, amplitude_fraction=.05)
    assert actions[0]["kind"] == "no_pw_constraint"
    assert [item["partition"] for item in actions[1:5]] == ["train"] * 4
    assert [item["partition"] for item in actions[5:]] == ["holdout"] * 2
    assert all(item.get("fatigue_weight_update") is False for item in actions[1:])


def test_action_table_requires_holdout_direction():
    with pytest.raises(ValueError, match="held-out"):
        predeclared_pw_actions(("Biceps", "Triceps"), {"d1": [1., 0.], "d2": [0., 1.]},
                               amplitude_fraction=.05)


def test_predeclared_direct_pw_plan_has_21_nonweight_jobs():
    digest = "a" * 64
    anchors = [{"id": f"c{cycle}", "completed_cycles": cycle, "source_receipt": f"/tmp/c{cycle}.json",
                "source_receipt_sha256": digest, "prepared_problem_sha256": digest}
               for cycle in (120, 140, 160)]
    directions = {"d1": [1 / np.sqrt(2), -1 / np.sqrt(2)],
                  "d2": [-1 / np.sqrt(2), 1 / np.sqrt(2)],
                  "mixed": [1 / np.sqrt(2), 1 / np.sqrt(2)]}
    plan = direct_pw_continuation_plan(anchors, muscle_names=("Biceps", "Triceps"), directions=directions,
                                       amplitude_fraction=.05)
    assert len(plan["jobs"]) == 21
    assert plan["fatigue_weight_updates_allowed"] is False
    assert {job["action"]["partition"] for job in plan["jobs"]} == {"control", "train", "holdout"}
    assert all(job["intervention_cycles"] == 1 for job in plan["jobs"])


def test_predeclared_direct_pw_plan_requires_exactly_three_anchors():
    anchor = {"id": "c120", "completed_cycles": 120, "source_receipt": "/tmp/c120.json",
              "source_receipt_sha256": "a" * 64, "prepared_problem_sha256": "a" * 64}
    directions = {"d1": [1., 0.], "d2": [0., 1.], "mixed": [1 / np.sqrt(2), 1 / np.sqrt(2)]}
    with pytest.raises(ValueError, match="three"):
        direct_pw_continuation_plan([anchor], muscle_names=("Biceps", "Triceps"), directions=directions,
                                    amplitude_fraction=.05)


def _branch_program():
    program = _program()
    program.nlp[0].model = SimpleNamespace(muscles_dynamics_model=[
        SimpleNamespace(muscle_name="Biceps"), SimpleNamespace(muscle_name="Triceps")])
    program.nlp[0].x_init = {"E_prod": SimpleNamespace(init=np.array([[0., 4.]]))}
    program.nlp[0].x_bounds = {"E_prod": _bound([[0., 0., 4.]], [[0., 8., 4.]])}
    program.parameter_init = {}
    program.parameter_bounds = {}
    program.absolute_wheel_q_cycle_index = 20
    class Binding:
        target = np.zeros((2, 3))
        activation = 0.
        def update_nmpc(self, nmpc, *, target, activation):
            self.target, self.activation = np.asarray(target).copy(), float(activation)
    program.pw_tracking_binding = Binding()
    return program


def test_direct_pw_branch_restores_nominal_problem_before_export(tmp_path):
    model = tmp_path / "model.json"
    model.write_text('{"muscles": ["Biceps", "Triceps"]}')
    source_program = _branch_program()
    source_path = tmp_path / "source.npz"
    export_prepared_checkpoint(source_path, source_program, completed_cycles=20, model_path=model)
    source = TaskReserveCheckpoint.from_archive(source_path, completed_cycles=20, model_path=model,
                                                task_context={"nominal_work_j": 4., "side": "left"})
    built = []
    def build():
        program = _branch_program()
        built.append(program)
        return {"nmpc": program, "solver": object()}
    def solve(program, _solver):
        for key in ("last_pulse_width_Biceps", "last_pulse_width_Triceps"):
            np.testing.assert_array_equal(program.nlp[0].u_bounds[key].min,
                                          source_program.nlp[0].u_bounds[key].min)
        return SimpleNamespace(status=0, decision_controls=lambda **_: {
            "last_pulse_width_Biceps": np.array([[.2, .3, .4]]),
            "last_pulse_width_Triceps": np.array([[.5, .4, .3]])})
    def advance(program, _solution, _work):
        np.testing.assert_array_equal(program.nlp[0].u_bounds["last_pulse_width_Biceps"].min,
                                      source_program.nlp[0].u_bounds["last_pulse_width_Biceps"].min)
        program.absolute_wheel_q_cycle_index += 1
    def witness(_solution, _program, directory, _offset, _report):
        path = directory / "witness.npz"
        np.savez(path, x=np.array([1.]))
        return str(path)
    action = DirectPulseWidthAction(("Biceps", "Triceps"), (1 / np.sqrt(2), -1 / np.sqrt(2)), .05)
    result = execute_direct_pw_branch(
        source, action=action, muscle_names=("Biceps", "Triceps"), output_directory=tmp_path / "result",
        build_runtime=build, solve_one_cycle=solve, advance_one_cycle=advance,
        audit=lambda *_: IndependentAudit(True, 0., DEFAULT_CONSTRAINT_GROUPS, {}),
        save_witness=witness, tolerance=1e-5,
    )
    assert result["success"] and result["endpoint"] is not None
    assert result["cycle"]["action"]["fatigue_weight_update"] is False
