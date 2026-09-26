import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from cocofest.optimization.rho_solve import RhoSolveRequest, run_rho_solve


def request_case(**changes):
    options = dict(
        update_functions=object(),
        solver=object(),
        solver_first_iter=object(),
        total_cycles=100,
        external_force={"torque": [0.2]},
        cycle_solutions=object(),
        get_all_iterations=True,
        cyclic_options={"states": {}},
        max_consecutive_failing=3,
        compact_solution_output=False,
    )
    options.update(changes)
    return RhoSolveRequest(**options)


@pytest.mark.parametrize("compact", [False, True])
def test_invocation_preserves_live_inputs_and_backend_return_by_identity(compact):
    request = request_case(compact_solution_output=compact)
    solution = (object(), [object()], [object()])
    calls = []

    def solve(*args, **kwargs):
        calls.append((args, kwargs))
        # Backend mutations of the live options must remain visible to callers.
        kwargs["cyclic_options"]["states"]["backend"] = True
        return solution

    result = run_rho_solve(SimpleNamespace(solve_fes_nmpc=solve), request)

    assert result is solution
    assert len(calls) == 1
    positional, options = calls[0]
    assert positional == (request.update_functions,)
    for name in (
        "solver",
        "solver_first_iter",
        "external_force",
        "cycle_solutions",
        "cyclic_options",
    ):
        assert options[name] is getattr(request, name)
    assert options == dict(
        solver=request.solver,
        solver_first_iter=request.solver_first_iter,
        total_cycles=100,
        external_force=request.external_force,
        cycle_solutions=request.cycle_solutions,
        get_all_iterations=True,
        cyclic_options={"states": {"backend": True}},
        max_consecutive_failing=3,
        compact_solution_output=compact,
    )


@pytest.mark.parametrize(
    "error",
    [RuntimeError("did not produce a valid solution"), ValueError("invalid request")],
)
def test_invocation_propagates_exact_exception_without_retry(error):
    calls = []

    def solve(*args, **kwargs):
        calls.append((args, kwargs))
        raise error

    with pytest.raises(type(error)) as raised:
        run_rho_solve(SimpleNamespace(solve_fes_nmpc=solve), request_case())
    assert raised.value is error
    assert len(calls) == 1


@pytest.mark.parametrize(
    "changes,recovery,expected_budget",
    [
        ({}, False, 3),
        ({}, True, 4),
        ({"retry_failed_rho_without_advance": False}, True, 3),
        ({"ipopt_failed_rho_pw_micro_retry": True}, False, 4),
        ({"nlp_failed_rho_phase_one_recovery": True}, False, 4),
        ({"acados_failed_rho_phase_one_recovery": True}, False, 4),
        ({"acados_ipopt_fallback_advance": True}, False, 301),
        ({"nlp_ipopt_fallback_advance": True}, False, 301),
        ({"ipopt_madnlp_fallback_advance": True}, False, 301),
    ],
)
def test_periodic_driver_dispatch_preserves_legacy_kwargs_and_failure_budgets(
    changes, recovery, expected_budget
):
    # Execute the real dispatch and budget helper without importing scientific
    # bindings. This checks the production call site, including recovery flags.
    driver = (
        Path(__file__).resolve().parents[1]
        / "examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py"
    )
    module = ast.parse(driver.read_text())
    budget = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "receding_horizon_solver_failure_budget"
    )
    calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "run_rho_solve"
    ]
    assert len(calls) == 1
    options = dict(
        n_windows=100,
        max_consecutive_failing=3,
        retry_failed_rho_without_advance=True,
        acados_ipopt_fallback_advance=False,
        compact_rho_output=True,
    )
    options.update(changes)
    received = []
    result = object()

    def solve(*args, **kwargs):
        received.append((args, kwargs))
        return result

    callback, solver, first_solver, cycles = object(), object(), object(), object()
    namespace = dict(
        RhoSolveRequest=RhoSolveRequest,
        run_rho_solve=run_rho_solve,
        nmpc=SimpleNamespace(solve_fes_nmpc=solve),
        update_functions=callback,
        solver=solver,
        solver_first_iter=first_solver,
        cycling_info={"resistive_torque": {"torque": [0.2]}},
        MultiCyclicCycleSolutions=SimpleNamespace(ALL_CYCLES=cycles),
        args=SimpleNamespace(**options),
        nlp_recovery_enabled=recovery,
        requested_window_solves=100,
    )
    exec(
        compile(ast.Module(body=[budget, calls[0]], type_ignores=[]), str(driver), "exec"),
        namespace,
    )
    assert namespace["sol"] is result
    assert received == [
        (
            (callback,),
            dict(
                solver=solver,
                solver_first_iter=first_solver,
                total_cycles=100,
                external_force={"torque": [0.2]},
                cycle_solutions=cycles,
                get_all_iterations=True,
                cyclic_options={"states": {}},
                max_consecutive_failing=expected_budget,
                compact_solution_output=True,
            ),
        )
    ]
