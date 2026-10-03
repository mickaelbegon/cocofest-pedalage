#!/usr/bin/env python3
"""Audited unilateral terminal-costate plumbing branches from one exact receipt.

The injected gradient is artificial. Results test parameter transport, objective
sign, trust rollback and timing; they are not endurance-value measurements.
"""

from __future__ import annotations

import argparse
from copy import copy
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.task_reserve_probe_adapter import independent_full_nlp_audit
from cocofest.optimization.terminal_costate_ocp import TerminalCostateObjectiveBinding
from cocofest.simulation.independent_arms_process import (
    _atomic_json, _configured_payload_model, _driver_arguments,
    _guard_untrusted_terminal_value, _observe_local_terminal_value,
)
from cocofest.simulation.rho_restart_checkpoint import (
    _read_prepared_checkpoint, _restore_array, _restore_container,
    _restore_runtime_state, _runtime_state, prepared_problem_arrays,
    prepared_problem_digest, export_prepared_checkpoint, restore_prepared_checkpoint,
)
from cocofest.optimization.receding_horizon_initial_guess import (
    initial_guess_signature, snapshot_initial_guess,
)
from scripts.probe_independent_rho_task_reserve import _load_arm
from scripts.run_local_task_reserve_branch import _advance, _solve, _save_witness


VARIANTS = {"inactive": 0., "plus": 1., "minus": -1., "half": .5}
PARAMETER_KEY = "rho_task_reserve"


def restore_source_with_inactive_costate(source, program):
    """Restore all source data and independently verify its original digest.

    The extra zero-valued costate parameter cannot be present in the archived
    checkpoint. All archived variables, bounds, parameters and runtime state
    must nevertheless match byte-for-byte at the numerical-array level.
    """
    metadata, snapshot, arrays = _read_prepared_checkpoint(
        Path(source.archive_path), completed_cycles=source.completed_cycles)
    nlp = program.nlp[0]
    _restore_container(nlp.x_init, snapshot["states"], "state initial guess")
    _restore_container(nlp.u_init, snapshot["controls"], "control initial guess")
    for prefix, container in (("x_bounds", nlp.x_bounds), ("u_bounds", nlp.u_bounds),
                              ("parameter_bounds", program.parameter_bounds)):
        archived_keys = {name.split(":")[1] for name in arrays if name.startswith(prefix + ":")}
        extra_keys = {PARAMETER_KEY} if prefix == "parameter_bounds" else set()
        if set(container.keys()) != archived_keys | extra_keys:
            raise ValueError(f"{prefix} keys differ from exact anchor plus the costate parameter")
        for key in archived_keys:
            for field in ("min", "max"):
                name = f"{prefix}:{key}:{field}"
                _restore_array(getattr(container[key], field), arrays[name], name)
    archived_parameter_keys = {name.split(":", 1)[1] for name in arrays if name.startswith("parameter_init:")}
    if set(program.parameter_init.keys()) != archived_parameter_keys | {PARAMETER_KEY}:
        raise ValueError("Fixed parameter keys differ from exact anchor plus costate")
    for key in archived_parameter_keys:
        _restore_array(program.parameter_init[key].init, arrays[f"parameter_init:{key}"], f"parameter_init:{key}")
    _restore_runtime_state(program, metadata.get("runtime_state", {}))
    restored_snapshot = snapshot_initial_guess(program)
    restored_arrays = {key: value for key, value in prepared_problem_arrays(program).items()
                       if key not in {f"parameter_bounds:{PARAMETER_KEY}:min",
                                      f"parameter_bounds:{PARAMETER_KEY}:max",
                                      f"parameter_init:{PARAMETER_KEY}"}}
    if initial_guess_signature(restored_snapshot) != metadata["prepared_primal_signature"]:
        raise RuntimeError("Physical primal differs from exact source checkpoint")
    digest = prepared_problem_digest(restored_snapshot, restored_arrays, _runtime_state(program))
    if digest != source.prepared_problem_sha256 or digest != metadata["prepared_problem_sha256"]:
        raise RuntimeError("Physical problem differs from exact source checkpoint")
    if not metadata.get("prepared_stimulation_history_complete"):
        raise RuntimeError("Source stimulation history is incomplete")
    if np.any(program.parameter_init[PARAMETER_KEY].init) or np.any(program.parameter_bounds[PARAMETER_KEY].min):
        raise RuntimeError("New costate parameter must start inactive")
    return {"source_prepared_problem_sha256": digest,
            "prepared_primal_signature": metadata["prepared_primal_signature"],
            "costate_extra_parameter_initially_zero": True,
            "stimulation_history_complete": True}


def _terminal_coordinates(binding, states):
    return np.asarray([float((np.asarray(states[c.state_key])[c.index, -1] - c.offset) / c.scale)
                       for c in binding.coordinates])


def run(receipt: Path, output: Path, *, cpu: int, variant: str, horizon: int,
        gradient_amplitude: float = 10000., trust_radius: float = .02,
        tolerance: float = 1e-6):
    from bioptim import SolutionMerge
    from cocofest.optimization.configured_cycling_model import configured_model_factories
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver

    if variant not in VARIANTS or horizon not in (1, 5, 20):
        raise ValueError("Variant must be inactive/plus/minus/half, horizon 1/5/20")
    if cpu not in os.sched_getaffinity(0):
        raise ValueError(f"Dedicated CPU {cpu} is outside this process affinity")
    if output.exists():
        raise FileExistsError(output)
    source, payload = _load_arm(receipt, "left")
    args = copy(_driver_arguments(payload, "left"))
    if (source.completed_cycles != 140 or payload["left_equivalent_mean_torque_nm"] != .96
            or payload["right_equivalent_mean_torque_nm"] != .96
            or payload.get("resistance_pace", {}).get("capacity_feedback") is not False
            or payload.get("muscle_pace", {}).get("adaptation_enabled") is not False
            or args.solver != "ipopt" or args.ipopt_linear_solver != "ma57"
            or args.ode_solver != "collocation" or args.collocation_degree != 5
            or args.collocation_method != "radau" or args.stimulations_per_cycle != 30
            or args.mechanical_formulation != "reduced" or args.n_threads != 1):
        raise ValueError("Fixed-task 30 Hz Radau-5 IPOPT/MA57 left c140 protocol mismatch")
    args.nlp_ipopt_recovery = False
    args.nlp_ipopt_recovery_ma57_tuned = False
    args.experimental_terminal_costate_config = {"maximum_age_cycles": 20}
    os.sched_setaffinity(0, {cpu})
    output.mkdir(parents=True, exist_ok=False)
    configured = _configured_payload_model(payload, "left")
    start_build = perf_counter()
    with configured_model_factories(configured[1], []):
        runtime = driver.build_unilateral_runtime(args, echo=False)
    build_seconds = perf_counter() - start_build
    program, solver = runtime["nmpc"], runtime["solver"]
    program._initialize_state_idx_to_cycle({"states": {}})
    binding = program.task_reserve_binding
    if not isinstance(binding, TerminalCostateObjectiveBinding):
        raise RuntimeError("Runtime did not construct the terminal costate graph")
    restored = restore_source_with_inactive_costate(source, program)
    nlp = program.nlp[0]
    source_point = np.asarray([(nlp.x_bounds[c.state_key].min[c.index, 0] - c.offset) / c.scale
                               for c in binding.coordinates], dtype=float)
    for c in binding.coordinates:
        if nlp.x_bounds[c.state_key].min[c.index, 0] != nlp.x_bounds[c.state_key].max[c.index, 0]:
            raise RuntimeError("A source state is not fixed at the exact checkpoint")
    # Artificial direction in compiled normalized A coordinates. It prices
    # Biceps upward and Triceps downward; no physiological claim follows.
    muscle_names = [c.state_key.removeprefix("A_") for c in binding.coordinates]
    if set(("Biceps", "Triceps")) - set(muscle_names):
        raise RuntimeError("Predeclared artificial muscle direction unavailable")
    direction = np.asarray([1. if name == "Biceps" else -1. if name == "Triceps" else 0.
                            for name in muscle_names])
    gradient = VARIANTS[variant] * gradient_amplitude * direction
    work = float(nlp.x_bounds["E_prod"].min[0, 2])
    if work != float(nlp.x_bounds["E_prod"].max[0, 2]):
        raise RuntimeError("Terminal work is not an equality")
    protocol = {"kind": "artificial_terminal_costate_plumbing_branch", "schema_version": 1,
                "source_receipt": str(receipt.resolve()), "source_receipt_sha256": sha256(receipt.read_bytes()).hexdigest(),
                "source_completed_cycles": source.completed_cycles, "source_archive_sha256": sha256(Path(source.archive_path).read_bytes()).hexdigest(),
                "source_restore": restored, "side": "left", "fixed_torque_nm": .96,
                "variant": variant, "horizon": horizon, "cpu": cpu, "numeric_threads": 1,
                "solver": "IPOPT/MA57", "integrator": "Radau-5", "stimulations_per_cycle": 30,
                "build_wall_seconds": build_seconds, "coordinate_layout": [asdict(c) for c in binding.coordinates],
                "source_point": source_point.tolist(), "gradient_artificial": gradient.tolist(),
                "gradient_amplitude": gradient_amplitude, "trust_radius": trust_radius,
                "nominal_work_j": work, "audit_tolerance": tolerance,
                "scientific_interpretation": "Numerical control only; gradient has no validated endurance meaning"}
    _atomic_json(output / "protocol.json", protocol)
    cycles = []
    compiled_solver = None
    for offset in range(1, horizon + 1):
        completed_before = source.completed_cycles + offset - 1
        program.total_optimization_run = offset - 1
        parameter_seconds = 0.
        if variant != "inactive" and binding.values[0] == 0. and offset == 1:
            tick = perf_counter()
            binding.update_from_remaining_cycles_value(
                program, remaining_cycles=20., center=source_point, gradient=gradient,
                trust_radius=np.full(binding.dimension, trust_radius),
                evaluation_coordinates=source_point,
                source_completed_cycles=source.completed_cycles,
                completed_cycles=completed_before)
            parameter_seconds = perf_counter() - tick
        active_before = bool(binding.values[0])
        rollback = export_prepared_checkpoint(output / f"prepared-before-{offset}.npz", program,
                                               completed_cycles=completed_before, model_path=Path(source.model_path))
        tick = perf_counter()
        solution = _solve(program, solver)
        solve_seconds = perf_counter() - tick
        current_compiled = getattr(getattr(program, "ocp_solver", None), "shaked_ocp_solver", None)
        if current_compiled is None or (compiled_solver is not None and current_compiled is not compiled_solver) or program.nlp[0] is not nlp:
            raise RuntimeError("Compiled RHO graph/solver not reused")
        compiled_solver = current_compiled
        audit = independent_full_nlp_audit(solution, program, target_work_j=work, tolerance=tolerance)
        states = solution.decision_states(to_merge=SolutionMerge.NODES)
        controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
        key, trust = _observe_local_terminal_value(program, states, completed_cycles=completed_before + 1)
        if key != "terminal_costate":
            raise RuntimeError("Costate audit was not selected")
        rejected_attempt = None
        if active_before and not trust["terminal_trust_validated"]:
            # No transfer or advance from the extrapolated decision. Restore
            # the exact prepared problem and solve the same cycle again.
            rejected_attempt = {"audit": dict(audit.detail), "trust": trust,
                                "solve_wall_seconds": solve_seconds, "status": str(solution.status)}
            replay = restore_prepared_checkpoint(Path(rollback["primal_path"]), program,
                                                 completed_cycles=completed_before)
            if replay["restored_problem_sha256"] != rollback["prepared_problem_sha256"]:
                raise RuntimeError("Pre-solve rollback was not exact")
            guard = _guard_untrusted_terminal_value(program, trust,
                                                    completed_cycles=completed_before + 1)
            if guard is None or binding.values[0] != 0.:
                raise RuntimeError("Trust guard did not deactivate the costate")
            tick = perf_counter()
            solution = _solve(program, solver)
            solve_seconds += perf_counter() - tick
            if getattr(getattr(program, "ocp_solver", None), "shaked_ocp_solver", None) is not compiled_solver:
                raise RuntimeError("Rollback reconstructed the compiled solver")
            audit = independent_full_nlp_audit(solution, program, target_work_j=work, tolerance=tolerance)
            states = solution.decision_states(to_merge=SolutionMerge.NODES)
            controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
            _, trust = _observe_local_terminal_value(program, states, completed_cycles=completed_before + 1)
            rejected_attempt["guard"] = guard
        terminal = _terminal_coordinates(binding, states)
        row = {"offset": offset, "completed_cycles": completed_before + 1,
               "certified": bool(audit.passed), "status": str(solution.status),
               "solve_wall_seconds": solve_seconds, "parameter_update_seconds": parameter_seconds,
               "maximum_normalized_violation": audit.maximum_normalized_violation,
               "audit": dict(audit.detail), "costate_active_before": active_before,
               "costate_active_after": bool(binding.values[0]), "terminal_trust": trust,
               "terminal_normalized_A": dict(zip(muscle_names, terminal.tolist())),
               "artificial_direction_projection": float(direction @ (terminal - source_point)),
               "pulse_width_controls_s": {name.removeprefix("last_pulse_width_"): np.asarray(value, float).tolist()
                                          for name, value in controls.items() if name.startswith("last_pulse_width_")},
               "rejected_untrusted_attempt": rejected_attempt}
        if audit.passed:
            row["witness"] = _save_witness(solution, program, output, offset, audit)
            _advance(program, solution, work)
        cycles.append(row)
        _atomic_json(output / "result.json", {**protocol, "cycles": cycles,
            "completed": offset == horizon or not audit.passed,
            "success": offset == horizon and all(c["certified"] for c in cycles),
            "compiled_nlp_reused": True})
        print(f"{variant} c{completed_before + 1}: certified={audit.passed} active={active_before} "
              f"trusted={trust.get('terminal_trust_validated')} solve_s={solve_seconds:.3f}", flush=True)
        if not audit.passed:
            break
    return output / "result.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=16)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--horizon", type=int, choices=(1, 5, 20), default=20)
    parser.add_argument("--gradient-amplitude", type=float, default=10000.)
    parser.add_argument("--trust-radius", type=float, default=.02)
    args = parser.parse_args()
    print(run(args.receipt, args.output, cpu=args.cpu, variant=args.variant,
              horizon=args.horizon, gradient_amplitude=args.gradient_amplitude,
              trust_radius=args.trust_radius))


if __name__ == "__main__":
    main()
