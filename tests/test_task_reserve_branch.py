"""Physical branch causality, graph reuse, and endpoint provenance contracts."""

from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.task_reserve import DEFAULT_CONSTRAINT_GROUPS, TaskReserveCheckpoint
from cocofest.optimization.task_reserve_branch import (
    certified_endpoint_receipt, execute_short_branch, json_digest, physical_branch_context, validate_policy,
)
from cocofest.optimization.task_reserve_experiment_design import reachable_branch_design
from cocofest.optimization.task_reserve_probe_adapter import IndependentAudit
from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint, restore_prepared_checkpoint


NAMES = ("Biceps", "Triceps")
KEY = "rho_fatigue_weights"


def _program():
    def var(v):
        return SimpleNamespace(init=np.array(v, float))
    def bound(a, b):
        return SimpleNamespace(min=np.array(a, float), max=np.array(b, float))
    program = SimpleNamespace(nlp=[SimpleNamespace(
        model=SimpleNamespace(muscles_dynamics_model=[SimpleNamespace(muscle_name=n) for n in NAMES]),
        x_init={"A_Biceps": var([[.8, .7]]), "E_prod": var([[0., 4.]])},
        u_init={"last_pulse_width_Biceps": var([[.0002]])},
        x_bounds={"A_Biceps": bound([[.8, 0., 0.]], [[.8, 2., 2.]]),
                  "E_prod": bound([[0., 0., 4.]], [[0., 8., 4.]])},
        u_bounds={"last_pulse_width_Biceps": bound([[.0001]], [[.0006]])},
    )], parameter_init={KEY: var([[1.], [1.]])},
        parameter_bounds={KEY: bound([[1.], [1.]], [[1.], [1.]])},
        absolute_wheel_q_cycle_index=20, ocp_solver=SimpleNamespace(shaked_ocp_solver=object()))

    class Binding:
        def update(self, p, weights):
            assert p.absolute_wheel_q_cycle_index == 20
            assert p.nlp[0].x_init["A_Biceps"].init[0, 0] == .8
            values = np.asarray(weights)[:, None]
            p.parameter_init[KEY].init[:] = values
            p.parameter_bounds[KEY].min[:] = values
            p.parameter_bounds[KEY].max[:] = values
            return {"integration": "fixed_parameter_bounds"}
    program.fatigue_weight_binding = Binding()
    return program


@pytest.fixture
def branch(tmp_path):
    model = tmp_path / "model.json"
    model.write_text('{"muscles": ["Biceps", "Triceps"]}')
    source_path = tmp_path / "source.npz"
    export_prepared_checkpoint(source_path, _program(), completed_cycles=20, model_path=model)
    source = TaskReserveCheckpoint.from_archive(source_path, completed_cycles=20, model_path=model,
                                                task_context={"nominal_work_j": 4., "side": "left"})
    policy = reachable_branch_design(NAMES)["branches"][1]
    output = tmp_path / "branch"
    return source, policy, output


def _execute(branch, *, fail_at=None, rebuild_at=None, mutate_update=False):
    source, policy, output = branch
    built, solved, advanced = [], [], []

    def build():
        program = _program()
        program.nlp[0].x_init["A_Biceps"].init[:] = 0.
        program.absolute_wheel_q_cycle_index = 0
        if mutate_update:
            update = program.fatigue_weight_binding.update
            def bad_update(p, w):
                result = update(p, w)
                p.nlp[0].u_bounds["last_pulse_width_Biceps"].max[:] = .01
                return result
            program.fatigue_weight_binding.update = bad_update
        built.append(program)
        return {"nmpc": program, "solver": object()}

    def solve(program, solver):
        solved.append(id(program))
        if rebuild_at == len(solved):
            program.ocp_solver.shaked_ocp_solver = object()
        return SimpleNamespace(status=3)

    def advance(program, solution, work):
        advanced.append(program.absolute_wheel_q_cycle_index)
        program.absolute_wheel_q_cycle_index += 1
        program.nlp[0].x_init["A_Biceps"].init -= .01
        program.nlp[0].x_bounds["A_Biceps"].min[0, 0] -= .01
        program.nlp[0].x_bounds["A_Biceps"].max[0, 0] -= .01

    def witness(solution, program, directory, offset, audit):
        path = directory / f"witness-{offset}.npz"
        np.savez(path, verified_state=program.nlp[0].x_init["A_Biceps"].init)
        return str(path)

    result = execute_short_branch(source, policy=policy, muscle_names=NAMES, output_directory=output,
        build_runtime=build, solve_one_cycle=solve, advance_one_cycle=advance,
        audit=lambda *args: IndependentAudit(len(solved) != fail_at, 0. if len(solved) != fail_at else .1,
                                            DEFAULT_CONSTRAINT_GROUPS, {}),
        save_witness=witness, tolerance=1e-5)
    return result, built, solved, advanced


def test_restores_before_weight_update_reuses_one_solver_and_keeps_source_unchanged(branch):
    source, policy, _ = branch
    original = deepcopy(policy)
    result, built, solved, advanced = _execute(branch)
    assert len(built) == 1 and len(set(solved)) == 1
    assert advanced == [20, 21, 22]
    assert result["success"] and result["compiled_solver_reused"]
    assert [row["completed_cycles"] for row in result["endpoints"]] == [21, 23]
    assert not result["fresh_worker_replay_verified"] and not result["exact_bilateral_restart"]
    assert policy == original
    source.verify_files()
    endpoint = result["endpoints"][-1]
    restored = restore_prepared_checkpoint(endpoint["primal_path"], _program(), completed_cycles=23)
    receipt = certified_endpoint_receipt(result, endpoint, restored, provenance={"source_receipt": "anchor"},
                                         producer_pid=100, verifier_pid=200)
    assert receipt["fresh_worker_replay_verified"]
    assert not receipt["exact_bilateral_restart"]
    assert len(receipt["cycles"]) == 3


def test_failed_audit_stops_without_advancing_or_exporting_uncertified_endpoint(branch):
    result, _, solved, advanced = _execute(branch, fail_at=2)
    assert len(solved) == 2 and advanced == [20]
    assert not result["success"]
    assert len(result["endpoints"]) == 1
    assert not result["cycles"][-1]["certified"]


def test_refuses_solver_rebuild(branch):
    with pytest.raises(RuntimeError, match="compiled solver"):
        _execute(branch, rebuild_at=2)


def test_refuses_physical_change_in_weight_update(branch):
    with pytest.raises(RuntimeError, match="physical bounds"):
        _execute(branch, mutate_update=True)


@pytest.mark.parametrize("key,value", [("nominal_work_scale", 1.1), ("endpoint_cycles", [1, 4]),
                                      ("partition", "other"), ("id", "../escape")])
def test_policy_validation_preserves_predeclared_partition_and_task(branch, key, value):
    _, policy, _ = branch
    policy[key] = value
    with pytest.raises(ValueError):
        validate_policy(policy, NAMES)


def test_context_excludes_only_weight_vector_and_never_mutates_input():
    payload = {"solver": "ipopt", "collocation_degree": 5, "torque": .96,
               "muscle_pace": {"left_initial_weights": {"Biceps": 1.}, "adaptation_enabled": False}}
    original = deepcopy(payload)
    context = physical_branch_context(payload, side="left", nominal_work_j=4., model_sha256="abc")
    changed_weights = deepcopy(payload)
    changed_weights["muscle_pace"]["left_initial_weights"] = {"Biceps": 2.}
    assert context == physical_branch_context(changed_weights, side="left", nominal_work_j=4., model_sha256="abc")
    changed_weights["collocation_degree"] = 3
    assert json_digest(context) != json_digest(physical_branch_context(
        changed_weights, side="left", nominal_work_j=4., model_sha256="abc"))
    assert payload == original


def test_receipt_refuses_same_process_or_mismatched_restoration(branch):
    result, _, _, _ = _execute(branch)
    endpoint = result["endpoints"][0]
    restored = restore_prepared_checkpoint(endpoint["primal_path"], _program(), completed_cycles=21)
    with pytest.raises(ValueError, match="different fresh"):
        certified_endpoint_receipt(result, endpoint, restored, provenance={}, producer_pid=2, verifier_pid=2)
    restored["restored_problem_sha256"] = "changed"
    with pytest.raises(ValueError, match="restoration mismatch"):
        certified_endpoint_receipt(result, endpoint, restored, provenance={}, producer_pid=2, verifier_pid=3)


def _persist_receipt(branch, monkeypatch):
    from scripts import run_local_task_reserve_branch as cli
    source, policy, output = branch
    export, _, _, _ = _execute(branch)
    endpoint = export["endpoints"][0]
    restored = restore_prepared_checkpoint(endpoint["primal_path"], _program(), completed_cycles=21)
    anchor = output.parent / "anchor.json"
    anchor.write_text('{"source": "mocked_strict_bilateral_receipt"}')
    plan = {"side": "left", "local_neighborhoods": [{"anchor_id": "c20", "anchor_receipt": str(anchor)}],
            "reachable_branch_design": {"muscle_order": list(NAMES), "branches": [policy]}}
    plan_path = output.parent / "plan.json"
    plan_path.write_text(json.dumps(plan))
    payload = {"solver": "ipopt", "torque": .96, "muscle_pace": {"left_initial_weights": {n: 1. for n in NAMES}}}
    monkeypatch.setattr(cli, "_load_arm", lambda *_: (source, payload))
    context = physical_branch_context(payload, side="left", nominal_work_j=4., model_sha256=source.model_sha256)
    provenance = {"side": "left", "source_receipt": str(anchor),
                  "source_receipt_sha256": sha256(anchor.read_bytes()).hexdigest(),
                  "plan_path": str(plan_path), "plan_sha256": sha256(plan_path.read_bytes()).hexdigest(),
                  "anchor_id": "c20", "physical_task_context": context,
                  "physical_task_context_sha256": json_digest(context)}
    receipt = certified_endpoint_receipt(export, endpoint, restored, provenance=provenance,
                                        producer_pid=2, verifier_pid=3)
    path = output / "receipt.json"
    path.write_text(json.dumps(receipt))
    return cli, path, receipt, payload


def test_endpoint_loader_preserves_physical_context_and_complete_history(branch, monkeypatch):
    cli, path, receipt, payload = _persist_receipt(branch, monkeypatch)
    endpoint, returned = cli.load_branch_endpoint(path)
    assert returned == payload
    assert endpoint.completed_cycles == 21
    assert endpoint.task_context_sha256 == receipt["physical_task_context_sha256"]
    assert endpoint.prepared_problem_sha256 == receipt["restored"]["restored_problem_sha256"]


@pytest.mark.parametrize("tamper", ["context", "archive", "witness", "policy", "plan"])
def test_endpoint_loader_refuses_provenance_and_artifact_tampering(branch, monkeypatch, tamper):
    from pathlib import Path
    cli, path, receipt, payload = _persist_receipt(branch, monkeypatch)
    if tamper == "context":
        payload["solver"] = "different_solver"
    elif tamper == "archive":
        with Path(receipt["endpoint"]["primal_path"]).open("ab") as stream:
            stream.write(b"tampered")
    elif tamper == "witness":
        with Path(receipt["cycles"][0]["witness"]).open("ab") as stream:
            stream.write(b"tampered")
    elif tamper == "policy":
        receipt["policy"]["absolute_fatigue_weights_from_unit"] = dict.fromkeys(NAMES, 1.)
        path.write_text(json.dumps(receipt))
    else:
        with Path(receipt["plan_path"]).open("a") as stream:
            stream.write(" ")
    with pytest.raises(ValueError):
        cli.load_branch_endpoint(path)
