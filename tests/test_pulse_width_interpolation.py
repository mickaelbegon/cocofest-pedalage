from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.pulse_width_interpolation import (
    validate_odd_interpolation, attach_odd_interpolation_audit,
    odd_interpolation_constraints,
)


@pytest.mark.parametrize("change", [
    {"max_step_s": None}, {"mechanical_formulation": "full"},
    {"cycles_per_window": 2}, {"stimulations_per_cycle": 30}, {"solver": "acados"},
])
def test_unsupported_odd_interpolation_is_rejected(change):
    options = dict(max_step_s=100e-6, mechanical_formulation="reduced",
                   cycles_per_window=1, stimulations_per_cycle=50, solver="ipopt")
    options.update(change)
    with pytest.raises(ValueError, match="requires"):
        validate_odd_interpolation(True, **options)
    assert validate_odd_interpolation(False, **options) is False


def test_no_last_control_or_cycle_wrap_constraint():
    constraints = odd_interpolation_constraints([SimpleNamespace(muscle_name="m")], scale=.0025)
    # Each entry couples increments k and k+1, thus physical controls k..k+2.
    entries = [entry for group in constraints.options for entry in group if entry]
    assert len(entries) == 24
    assert [entry.node for entry in entries] == list(range(0, 48, 2))


def test_audit_reports_physical_residual_and_ignores_last_control():
    pw = np.linspace(200., 400., 50)
    pw[-1] = 600.
    summary = {"control_traces": {"last_pulse_width_m": pw * 1e-6}}
    args = SimpleNamespace(pulse_width_odd_interpolation=True)
    attach_odd_interpolation_audit(summary, args)
    audit = summary["pulse_width_odd_interpolation_audit"]
    assert audit["muscles"]["m"]["maximum_abs_residual_us"] < 1e-10
    pw[1] += 5
    summary["control_traces"]["last_pulse_width_m"] = pw * 1e-6
    attach_odd_interpolation_audit(summary, args)
    assert summary["pulse_width_odd_interpolation_audit"]["muscles"]["m"]["maximum_abs_residual_us"] == pytest.approx(5)


def test_cli_signature_distinguishes_opt_in():
    import argparse
    from cocofest.optimization.pulse_width_slew import add_pulse_width_slew_cli, pulse_width_slew_signature
    parser = argparse.ArgumentParser()
    add_pulse_width_slew_cli(parser)
    assert pulse_width_slew_signature(parser.parse_args([]))["pulse_width_odd_interpolation"] is False
    assert pulse_width_slew_signature(parser.parse_args(["--pulse-width-odd-interpolation"]))["pulse_width_odd_interpolation"] is True


@pytest.mark.parametrize("enabled", [False, True])
def test_common_seed_rejects_the_other_interpolation_contract(tmp_path, enabled):
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as runner
    args = runner.build_argument_parser().parse_args(["--solver", "ipopt", "--mechanical-formulation", "reduced"])
    args.terminal_wheel_q_reference_mode = "absolute_initial"
    args.pulse_width_odd_interpolation = enabled
    metadata = runner._common_initial_solution_metadata(args)
    assert metadata["pulse_width_odd_interpolation"] is enabled
    metadata["pulse_width_odd_interpolation"] = not enabled
    seed = runner._WarmupSolutionAdapter({}, {}, metadata=metadata)
    with pytest.raises(ValueError, match="pulse_width_odd_interpolation"):
        runner._validate_common_initial_solution_metadata(seed, args, tmp_path / "seed.npz")


def test_generic_warmup_disables_target_interpolation():
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as runner
    conditions = {"pulse_width_odd_interpolation": True, "pulse_width_max_step_s": 100e-6}
    bridge = runner._target_independent_warmup_conditions(conditions)
    assert bridge["pulse_width_odd_interpolation"] is False
    assert conditions["pulse_width_odd_interpolation"] is True


@pytest.mark.parametrize("enabled", [False, True])
def test_comparison_cli_forwards_interpolation_to_solver(monkeypatch, enabled):
    import ast
    from pathlib import Path
    import sys
    from examples.fes_multibody.cycling import cycling_fes_solver_comparison as runner
    arguments = ["compare", "--solvers", "ipopt", "--benchmark-profile", "scientific-radau5",
                 "--mechanical-formulation", "reduced", "--stimulations-per-cycle", "50",
                 "--pulse-width-max-step-us", "100"]
    if enabled:
        arguments.append("--pulse-width-odd-interpolation")
    monkeypatch.setattr(sys, "argv", arguments)
    captured = {}

    class ReachedSolver(Exception):
        pass

    def capture(name, args, **kwargs):
        captured["enabled"] = args.pulse_width_odd_interpolation
        raise ReachedSolver()

    monkeypatch.setattr(runner, "_run_benchmark_case", capture)
    # Execute the real __main__ call with the imported functions. This checks
    # parser -> main kwargs -> solver Namespace, without constructing an OCP.
    tree = ast.parse(Path(runner.__file__).read_text())
    entrypoint = tree.body[-1]
    assert isinstance(entrypoint, ast.If)
    code = compile(ast.Module(body=entrypoint.body, type_ignores=[]), runner.__file__, "exec")
    with pytest.raises(ReachedSolver):
        exec(code, vars(runner).copy())
    assert captured["enabled"] is enabled
