import ast
from dataclasses import replace
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import pytest

from cocofest.optimization.solver_recovery import (
    RecoveryOperations,
    run_frozen_nlp_recovery,
)


def recovery_case(*, accepted=True, certified=True, success=True):
    events = []
    solution = SimpleNamespace(status=0 if success else 1, solver_time_to_optimize=0.25)
    recovery = SimpleNamespace(_cocofest_recovery_structure={"states": ["force"]})
    target = SimpleNamespace()

    def configure():
        events.append("configure")
        return "backend"

    def solve(nmpc, backend):
        assert nmpc is recovery and backend == "backend"
        events.append("solve")
        return solution

    def populate(sol, nmpc):
        assert sol is solution and nmpc is recovery
        events.append("populate")
        sol.inf_pr = 1e-9

    def feasibility(sol, tolerance):
        assert sol is solution and tolerance == 1e-5
        events.append("feasibility")
        return {"final_inf_pr": sol.inf_pr}

    def acceptance(status, feasibility):
        events.append("acceptance")
        return dict(
            accepted=accepted,
            certified=certified,
            success=success,
            provisional=not success and certified,
        )

    def compatibility(failed, sol):
        assert failed == "failed" and sol is solution
        events.append("compatibility")
        return {"available": True}

    def inject(nmpc, sol):
        assert nmpc is target and sol is solution
        events.append("inject")
        nmpc.seed = sol

    operations = RecoveryOperations(
        configure, solve, populate, feasibility, acceptance, compatibility, inject
    )
    arguments = dict(
        recovery_nmpc=recovery,
        target_nmpc=target,
        operations=operations,
        recovery_solver="ipopt",
        max_iterations=20,
        tolerance=1e-5,
        failed_target_solution="failed",
        target_solver="acados",
        mechanical_formulation="reduced",
        clock=lambda: next(ticks),
    )
    ticks = count()
    return arguments, events, solution


@pytest.mark.parametrize(
    "success,certified,accepted,quality",
    [
        (True, True, True, "converged"),
        (False, True, True, "feasible_nonconverged"),
        (False, False, False, "rejected"),
    ],
)
def test_recovery_audits_before_injection_and_preserves_certification(
    success, certified, accepted, quality
):
    arguments, events, solution = recovery_case(
        accepted=accepted, certified=certified, success=success
    )
    returned, summary = run_frozen_nlp_recovery(**arguments)
    assert returned is solution
    assert events == [
        "configure",
        "solve",
        "populate",
        "feasibility",
        "acceptance",
        "compatibility",
    ] + (["inject"] if accepted else [])
    assert summary["accepted"] is accepted
    assert summary["seed_injected"] is accepted
    assert hasattr(arguments["target_nmpc"], "seed") is accepted
    assert summary["certified_feasibility"] is certified
    assert summary["provisional"] is (not success and certified)
    assert summary["quality"] == quality
    assert summary["wall_time_s"] is None
    assert summary["solver_time_s"] == 0.25
    assert (
        summary["compatibility_with_failed_acados"]
        == summary["compatibility_with_failed_target"]
    )
    assert summary["timing"] == {
        "configure_solver_wall_time_s": 1,
        "solve_call_wall_time_s": 1,
        "feasibility_audit_wall_time_s": 1,
        "compatibility_audit_wall_time_s": 1,
        "seed_injection_wall_time_s": 1,
        "total_wall_time_s": 11,
    }
    arguments["recovery_nmpc"]._cocofest_recovery_structure["states"].append("calcium")
    assert summary["structure"] == {"states": ["force"]}


@pytest.mark.parametrize("failure_stage", ["configure_solver", "solve"])
def test_recovery_failures_report_phase_timing_without_injection(failure_stage, capsys):
    arguments, events, _ = recovery_case()

    def fail(*args):
        raise RuntimeError("backend unavailable")

    arguments["operations"] = replace(arguments["operations"], **{failure_stage: fail})
    solution, summary = run_frozen_nlp_recovery(**arguments, echo=True)
    assert solution is None
    assert not summary["accepted"] and not summary["seed_injected"]
    assert summary["error"] == "RuntimeError: backend unavailable"
    assert "RuntimeError: backend unavailable" in summary["traceback"]
    assert (summary["timing"]["solve_call_wall_time_s"] is None) == (
        failure_stage == "configure_solver"
    )
    assert "acados_ipopt_recovery_error:" in capsys.readouterr().out
    assert not hasattr(arguments["target_nmpc"], "seed")
    assert "feasibility" not in events


@pytest.mark.parametrize(
    "operation", ["populate_inf_pr", "feasibility_summary", "acceptance", "inject_seed"]
)
def test_audit_or_injection_failure_remains_fatal(operation):
    arguments, _, _ = recovery_case()

    def fail(*args):
        raise ValueError("invalid physical state")

    arguments["operations"] = replace(arguments["operations"], **{operation: fail})
    with pytest.raises(ValueError, match="invalid physical state"):
        run_frozen_nlp_recovery(**arguments)


def test_unsupported_backend_rejected_before_configuration():
    arguments, events, _ = recovery_case()
    arguments["recovery_solver"] = "acados"
    with pytest.raises(ValueError, match="IPOPT and MadNLP only"):
        run_frozen_nlp_recovery(**arguments)
    assert not events


def test_madnlp_recovery_retains_backend_identity_without_acados_alias():
    arguments, _, _ = recovery_case()
    arguments.update(recovery_solver="madnlp", target_solver="ipopt")
    _, summary = run_frozen_nlp_recovery(**arguments)
    assert summary["solver"] == "madnlp"
    assert summary["target_solver"] == "ipopt"
    assert "compatibility_with_failed_acados" not in summary


@pytest.mark.parametrize("backend", ["ipopt", "madnlp"])
def test_historical_driver_wrapper_preserves_solver_options_and_frozen_solve(backend):
    # Execute only the compatibility function so this guard also runs in the
    # lightweight CI environment without loading Bioptim or native solvers.
    driver = (
        Path(__file__).resolve().parents[1]
        / "examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py"
    )
    module = ast.parse(driver.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_periodic_nlp_recovery"
    )
    arguments, events, solution = recovery_case()
    configured = []
    solved = []

    class FrozenProblem:
        def solve(self, *, solver, warm_start):
            solved.append((solver, warm_start))
            return solution

    class RhoProblem(FrozenProblem):
        def solve(self, **kwargs):
            raise AssertionError("recovery must not enter the advancing RHO solve")

    def configure(*args, **options):
        configured.append((args, options))
        return "backend"

    ops = arguments["operations"]
    namespace = dict(
        RecoveryOperations=RecoveryOperations,
        run_frozen_nlp_recovery=run_frozen_nlp_recovery,
        RecedingHorizonOptimization=RhoProblem,
        configure_ipopt_solver=configure,
        configure_nlp_solver=configure,
        populate_solution_inf_pr_from_solver_stats=lambda *args: None,
        _solution_feasibility_summary=lambda *args: {"final_inf_pr": 1e-9},
        periodic_refinement_acceptance=ops.acceptance,
        solution_trace_compatibility_summary=ops.compatibility_summary,
        apply_solution_directly_to_periodic_nmpc_initial_guess=ops.inject_seed,
        perf_counter=arguments["clock"],
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(driver), "exec"),
        namespace,
    )
    arguments.pop("operations")
    arguments.pop("clock")
    arguments.update(
        recovery_nmpc=RhoProblem(),
        recovery_solver=backend,
        linear_solver=None,
        c_compile=True,
        max_wall_time=5.0,
        ipopt_advanced_options={"mu_strategy": "adaptive"},
    )
    result, summary = namespace["run_periodic_nlp_recovery"](**arguments)
    assert result is solution and summary["seed_injected"]
    assert solved == [("backend", None)]
    assert configured == (
        [
            (
                (),
                dict(
                    max_iterations=20,
                    linear_solver="mumps",
                    tolerance=1e-5,
                    c_compile=True,
                    advanced_options={"mu_strategy": "adaptive"},
                ),
            )
        ]
        if backend == "ipopt"
        else [
            (
                ("madnlp",),
                dict(
                    max_iterations=20,
                    tolerance=1e-5,
                    madnlp_linear_solver=None,
                    madnlp_c_compile=True,
                    madnlp_max_wall_time=5.0,
                ),
            )
        ]
    )
