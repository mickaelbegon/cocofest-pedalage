from dataclasses import asdict, replace
import json
from types import SimpleNamespace, MethodType
import casadi as ca
import numpy as np
import pytest
from cocofest.optimization.task_load_margin import TaskLoadMarginRequest, TaskLoadMarginResult, TaskLoadMarginDirection
from cocofest.optimization.task_load_margin_ocp import TaskLoadMarginObjectiveBinding
from cocofest.optimization.task_reserve import TaskReserveCheckpoint, ProbeEvidence, DEFAULT_CONSTRAINT_GROUPS
from cocofest.optimization.task_reserve_ocp import TaskReserveStateCoordinate, TASK_RESERVE_PARAMETER_KEY, task_reserve_parameter_options
from cocofest.optimization.parametric_fatigue_weights import ParametricFatigueWeightBinding, FATIGUE_WEIGHT_PARAMETER_KEY


@pytest.fixture
def case(tmp_path, monkeypatch):
    # Synthetic complete evidence tests acceptance semantics, not physiology.
    monkeypatch.setattr(TaskReserveCheckpoint, "verify_files", lambda _: None)
    layout = (TaskReserveStateCoordinate("A", scale=10.), TaskReserveStateCoordinate("Km", scale=.1))
    context = {"coordinate_layout": [asdict(item) for item in layout]}
    binding = TaskLoadMarginObjectiveBinding(layout, task_context=context, model_sha256="a"*64,
        physical_context={}, conditional_context_sha256="b"*64)
    checkpoint = TaskReserveCheckpoint("unused.npz", "c"*64, "d"*64, 100,
        "unused.json", "a"*64, json.dumps(context), binding.task_context_sha256)
    request = TaskLoadMarginRequest("test", checkpoint, ("A", "Km"), (.8, 1.), (.1,.1),
                                   (0.,0.), (1.,10.), 200., 3.)
    artifact = tmp_path/"solution.npz"
    artifact.write_bytes(b"synthetic test fixture")
    witness = ProbeEvidence(1., checkpoint.prepared_problem_sha256, checkpoint.task_context_sha256,
        checkpoint.model_sha256, "0", 1e-9, 1e-6, True, str(artifact), .1,
        DEFAULT_CONSTRAINT_GROUPS, True)
    result = TaskLoadMarginResult("test", 201., witness, replace(witness, work_scale=1.2),
        (.4,-.2), "kkt_envelope", True, 1e-9,1e-9,1e-9,.001,.001,4,
        solution_artifact=str(artifact))
    from bioptim import OptimalControlProgram
    program = SimpleNamespace(nlp=[SimpleNamespace(update_init=lambda *args: None)], n_phases=1,
        **task_reserve_parameter_options(binding, fatigue_weight_binding=ParametricFatigueWeightBinding((1.,2.))))
    program.update_initial_guess = MethodType(OptimalControlProgram.update_initial_guess, program)
    validation = dict(coordinates=(.8,1.), completed_cycles=101, now_monotonic_seconds=202.,
                      conditional_context_sha256="b"*64)
    return binding, TaskLoadMarginDirection(request,result), program, validation, artifact


@pytest.mark.parametrize("kind", ["SX", "MX"])
def test_affine_terminal_gradient_and_inactive_cost(case, kind):
    binding, direction, _, _, _ = case
    symbol = getattr(ca, kind)
    x, p = symbol.sym("x",2), symbol.sym("p",binding.values.size)
    controller = SimpleNamespace(states={"A":SimpleNamespace(cx=x[0]),"Km":SimpleNamespace(cx=x[1])},
        parameters={TASK_RESERVE_PARAMETER_KEY:SimpleNamespace(cx=p)})
    cost = binding.objective(controller)
    fun = ca.Function("load_margin", [x,p], [cost,ca.gradient(cost,x)])
    value, gradient = fun([8.,.1], binding.values)
    assert float(value) == 0
    np.testing.assert_array_equal(gradient,[[0.],[0.]])
    value, gradient = fun([8.1,.101], np.r_[100.,0.,direction.request.center,direction.result.gradient])
    assert float(value) == pytest.approx(-.2)
    np.testing.assert_allclose(np.asarray(gradient).ravel(),[-4.,200.])


def test_update_keeps_fatigue_and_parameter_graph_then_terminal_gate_requires_retry(case):
    binding, direction, program, validation, _ = case
    graph = program.nlp[0]
    assert binding.update(program,direction,weight=100.,**validation)["accepted"]
    np.testing.assert_array_equal(program.parameter_bounds[FATIGUE_WEIGHT_PARAMETER_KEY].min,[[1.],[2.]])
    assert program.nlp[0] is graph
    terminal = dict(validation); terminal.pop("coordinates")
    assert binding.validate_terminal_point((.82,1.01),**terminal)["accepted"]
    bad = binding.validate_terminal_point((.95,1.01),**terminal)
    assert not bad["accepted"] and bad["fallback_resolve_required"]
    assert not bad["physiological_failure_certified"]


@pytest.mark.parametrize("changes,reason", [
    ({"now_monotonic_seconds":222.},"time_age_invalid"),
    ({"completed_cycles":125},"cycle_age_invalid"),
    ({"conditional_context_sha256":"c"*64},"conditional_context_mismatch"),
    ({"coordinates":(.99,1.)},"outside_trust_region"),
])
def test_bad_application_context_falls_back_without_rebuilding(case,changes,reason):
    binding,direction,program,validation,_ = case
    binding.update(program,direction,weight=100.,**validation)
    report = binding.update(program,direction,weight=100.,**(validation|changes))
    assert not report["accepted"] and reason in report["reasons"]
    assert binding.values[0] == 0


def test_changed_solution_provenance_is_rejected_before_transfer(case):
    binding,direction,program,validation,artifact = case
    binding.update(program,direction,weight=100.,**validation)
    artifact.write_bytes(b"changed")
    report = binding.validate_terminal_point(**validation)
    assert "solution_artifact_changed" in report["reasons"]
    assert report["fallback_resolve_required"]
