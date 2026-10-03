"""The full-NLP reserve probe must never turn a solver failure into fatigue."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.task_reserve import (
    DEFAULT_CONSTRAINT_GROUPS, TaskReserveCheckpoint, TaskReserveProbe,
    evaluate_work_reserve,
)
from cocofest.optimization.task_reserve_probe_adapter import (
    IndependentAudit, _normalized_bound_violation, probe_one_cycle,
)
from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint


def _program():
    def variable(values):
        return SimpleNamespace(init=np.asarray(values, float))

    def bounds(lower, upper):
        return SimpleNamespace(min=np.asarray(lower, float), max=np.asarray(upper, float))

    return SimpleNamespace(nlp=[SimpleNamespace(
        x_init={"A_Biceps": variable([[.9, .8]]), "E_prod": variable([[0., 1.]])},
        u_init={"last_pulse_width_Biceps": variable([[.0002]])},
        x_bounds={"A_Biceps": bounds([[.9, 0., 0.]], [[.9, 2., 2.]]),
                  "E_prod": bounds([[0., 0., 1.]], [[0., 2., 1.]])},
        u_bounds={"last_pulse_width_Biceps": bounds([[0.]], [[.0006]])},
    )], parameter_init={}, parameter_bounds={}, absolute_wheel_q_cycle_index=20)


@pytest.fixture
def source(tmp_path):
    model = tmp_path / "model.json"
    model.write_text('{"case":"probe"}', encoding="utf-8")
    archive = tmp_path / "source.npz"
    export_prepared_checkpoint(archive, _program(), completed_cycles=20, model_path=model)
    return TaskReserveCheckpoint.from_archive(archive, completed_cycles=20,
        model_path=model, task_context={"nominal_work_j": 1., "side": "right"})


def test_each_scale_restores_fresh_checkpoint_before_target_update(source):
    built, targets = [], []

    def build():
        program = _program()
        # A fresh NLP starts with unrelated active data. Restoration must fix it.
        program.nlp[0].x_init["A_Biceps"].init[:] = 0.
        built.append(program)
        return {"nmpc": program, "solver": object()}

    def solve(program, solver):
        assert program.nlp[0].x_init["A_Biceps"].init[0, 0] == .9
        target = program.nlp[0].x_bounds["E_prod"].min[0, 2]
        targets.append(target)
        return SimpleNamespace(status=3)  # status does not decide witness validity

    def audit(solution, program, target, tolerance):
        return IndependentAudit(True, 0., DEFAULT_CONSTRAINT_GROUPS, {})

    estimate = evaluate_work_reserve(source, [1.2, 1.],
        lambda request: probe_one_cycle(request, build_runtime=build,
            solve_one_cycle=solve, audit=audit,
            save_witness=lambda *args: "synthetic:validated-trajectory"))
    assert len(built) == 2 and built[0] is not built[1]
    assert targets == pytest.approx([1., 1.2])
    assert estimate.feasible_work_scales == (1., 1.2)
    assert all(item.solver_status == "3" for item in estimate.evidence)


@pytest.mark.parametrize("failure", ["restore", "solve", "audit"])
def test_failures_are_indeterminate_not_physiological(source, monkeypatch, failure):
    if failure == "restore":
        monkeypatch.setattr("cocofest.optimization.task_reserve_probe_adapter.restore_prepared_checkpoint",
                            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("restore failure")))

    def solve(*args):
        if failure == "solve":
            raise RuntimeError("solver failure")
        return SimpleNamespace(status=0)

    def audit(*args):
        if failure == "audit":
            raise RuntimeError("audit unavailable")
        return IndependentAudit(True, 0., DEFAULT_CONSTRAINT_GROUPS, {})

    result = probe_one_cycle(TaskReserveProbe(source, 1.),
        build_runtime=lambda: {"nmpc": _program(), "solver": object()},
        solve_one_cycle=solve, audit=audit,
        save_witness=lambda *args: "synthetic:trajectory")
    assert result.independent_validation_passed is False
    assert result.witness_id is None
    assert not result.witness_is_valid(TaskReserveProbe(source, 1.))
    assert "failure" in result.reason or "unavailable" in result.reason


def test_incomplete_constraint_coverage_cannot_become_witness(source):
    result = probe_one_cycle(TaskReserveProbe(source, 1.),
        build_runtime=lambda: {"nmpc": _program(), "solver": object()},
        solve_one_cycle=lambda *args: SimpleNamespace(status=0),
        audit=lambda *args: IndependentAudit(True, 0., ("task_work",), {}),
        save_witness=lambda *args: "synthetic:trajectory")
    assert not result.independent_validation_passed
    assert result.witness_id is None


def test_nonfinite_audit_result_cannot_save_a_witness(source):
    saved = []
    result = probe_one_cycle(TaskReserveProbe(source, 1.),
        build_runtime=lambda: {"nmpc": _program(), "solver": object()},
        solve_one_cycle=lambda *args: SimpleNamespace(status=0),
        audit=lambda *args: IndependentAudit(True, float("nan"), DEFAULT_CONSTRAINT_GROUPS, {}),
        save_witness=lambda *args: saved.append(True))
    assert not result.independent_validation_passed
    assert result.witness_id is None
    assert not saved


def test_normalized_bound_audit_refuses_missing_rows_or_bounds():
    assert _normalized_bound_violation([2.], [1.], [1.]) == pytest.approx(1.)
    with pytest.raises(ValueError, match="absent"):
        _normalized_bound_violation([1.], [], [])
    with pytest.raises(ValueError, match="nonfinite"):
        _normalized_bound_violation([float("nan")], [0.], [1.])
