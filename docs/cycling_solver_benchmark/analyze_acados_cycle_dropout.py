#!/usr/bin/env python3
"""Replay an intermittent ACADOS dropout with a physically continuous policy.

The failed NLP state trajectory must never be shifted into the next RHO.  This
tool instead starts from the last certified state, applies either the optimized
PW pattern or the previous cycle's PW pattern, and integrates the exact reduced
model with DOP853.  It therefore measures the cost of a simple, implementable
degraded mode while keeping state continuity by construction.

The future controls are intentionally *not* reoptimized during the propagation
test.  The resulting multi-cycle error is a sensitivity test, not a prediction
of a closed-loop NMPC equipped with state feedback.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.integrate import solve_ivp

from cocofest import ReducedCyclingDynamics
from examples.fes_multibody.cycling.cycling_pulse_width_mhe import set_fes_model


MUSCLE_STATE_NAMES = ("Cn", "F", "A", "Tau1", "Km")


@dataclass(frozen=True)
class ReplayContext:
    muscle_models: tuple
    reduced_dynamics: ReducedCyclingDynamics
    external_crank_torque: float
    stimulations_per_cycle: int
    terminal_angle_tolerance: float
    velocity_lower: float
    velocity_upper: float


def parse_cycle_indices(value: str) -> tuple[int, ...]:
    cycles = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    if not cycles or any(cycle < 2 for cycle in cycles):
        raise argparse.ArgumentTypeError(
            "Dropout cycles must contain comma-separated integers >= 2."
        )
    return cycles


def _configuration(payload: dict) -> dict:
    configurations = payload.get("configurations") or {}
    if "acados" in configurations:
        return configurations["acados"]
    if len(configurations) == 1:
        return next(iter(configurations.values()))
    raise ValueError("The result JSON does not contain one identifiable ACADOS configuration.")


def _result(payload: dict) -> dict:
    results = payload.get("results") or []
    matches = [result for result in results if result.get("solver") == "acados"]
    if len(matches) == 1:
        return matches[0]
    if len(results) == 1:
        return results[0]
    raise ValueError("The result JSON does not contain one identifiable ACADOS result.")


def build_context(
    configuration: dict,
    *,
    repository_root: Path,
    reduced_profile: Path | None = None,
) -> ReplayContext:
    stimulations = int(configuration["stimulations_per_cycle"])
    model_path = (
        repository_root
        / "examples/msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod"
    )
    profile_path = reduced_profile or (
        repository_root
        / "examples/fes_multibody/cycling/result/cache/reduced_cycling_fourier12.npz"
    )
    fes_model = set_fes_model(
        str(model_path),
        list(np.linspace(0.0, 1.0, stimulations, endpoint=False)),
        periodic_node_forcing=True,
    )
    muscle_models = tuple(fes_model.muscles_dynamics_model)
    reduced_dynamics = ReducedCyclingDynamics.load(profile_path)
    expected_names = tuple(model.muscle_name for model in muscle_models)
    if expected_names != reduced_dynamics.muscle_names:
        raise ValueError(
            f"Muscle ordering differs between Ding and reduced models: "
            f"{expected_names} versus {reduced_dynamics.muscle_names}."
        )
    margin = float(configuration.get("wheel_qdot_bound_margin", 3.0))
    target_velocity = float(
        configuration.get("wheel_qdot_regularization_target", -2.0 * np.pi)
    )
    return ReplayContext(
        muscle_models=muscle_models,
        reduced_dynamics=reduced_dynamics,
        external_crank_torque=float(configuration.get("constant_crank_torque", 0.0)),
        stimulations_per_cycle=stimulations,
        terminal_angle_tolerance=float(
            configuration.get("acados_terminal_wheel_q_slack", 0.002)
        ),
        velocity_lower=target_velocity - margin,
        velocity_upper=target_velocity + margin,
    )


def state_keys(context: ReplayContext) -> tuple[str, ...]:
    return tuple(
        f"{state}_{model.muscle_name}"
        for model in context.muscle_models
        for state in MUSCLE_STATE_NAMES
    ) + ("theta", "omega")


def load_trajectory(path: Path, context: ReplayContext) -> tuple[dict, dict, int]:
    with np.load(path, allow_pickle=False) as data:
        states = {
            key: np.asarray(data[f"states__{key}"], dtype=float).reshape(-1)
            for key in state_keys(context)
        }
        controls = {
            model.muscle_name: np.asarray(
                data[f"controls__last_pulse_width_{model.muscle_name}"],
                dtype=float,
            ).reshape(-1)
            for model in context.muscle_models
        }
    node_counts = {values.size for values in states.values()}
    control_counts = {values.size for values in controls.values()}
    if len(node_counts) != 1 or len(control_counts) != 1:
        raise ValueError("All exported state and control traces must share their layouts.")
    interval_count = next(iter(control_counts))
    if next(iter(node_counts)) != interval_count + 1:
        raise ValueError("The dropout replay requires shooting-node state traces.")
    cycle_count, remainder = divmod(interval_count, context.stimulations_per_cycle)
    if remainder:
        raise ValueError("The exported controls do not contain complete crank cycles.")
    return states, controls, cycle_count


def _state_vector(states: dict[str, np.ndarray], keys: tuple[str, ...], node: int) -> np.ndarray:
    return np.asarray([states[key][node] for key in keys], dtype=float)


def _control_cycle(
    controls: dict[str, np.ndarray],
    context: ReplayContext,
    cycle: int,
) -> np.ndarray:
    first = (cycle - 1) * context.stimulations_per_cycle
    final = cycle * context.stimulations_per_cycle
    return np.column_stack(
        [controls[model.muscle_name][first:final] for model in context.muscle_models]
    )


def integrate_controls(
    initial_state: np.ndarray,
    control_cycles: Iterable[np.ndarray],
    context: ReplayContext,
    *,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
) -> dict:
    controls = tuple(np.asarray(values, dtype=float) for values in control_cycles)
    interval_count = len(controls) * context.stimulations_per_cycle
    dt = 1.0 / context.stimulations_per_cycle
    n_states = initial_state.size
    n_muscles = len(context.muscle_models)
    augmented = np.concatenate((np.asarray(initial_state, dtype=float), np.zeros(2 * n_muscles)))
    omega_minimum = float(initial_state[-1])
    omega_maximum = float(initial_state[-1])
    function_evaluations = 0

    for interval in range(interval_count):
        cycle_offset, local_node = divmod(interval, context.stimulations_per_cycle)
        control = controls[cycle_offset][local_node]
        interval_start = local_node * dt

        def rhs(time, values):
            state = values[:n_states]
            theta, omega = state[-2:]
            force_length, force_velocity, passive_force = (
                context.reduced_dynamics.muscle_relationships(theta, omega)
            )
            derivatives = []
            normalized_fatigue = []
            for muscle_index, model in enumerate(context.muscle_models):
                first = 5 * muscle_index
                muscle_state = state[first : first + 5]
                muscle_derivative = model.system_dynamics(
                    states=muscle_state,
                    controls=np.asarray([control[muscle_index]]),
                    time=np.asarray([time]),
                    numerical_timeseries=np.asarray(
                        [model.post_stimulation_amplitude(), interval_start]
                    ),
                    force_length_relationship=force_length[muscle_index],
                    force_velocity_relationship=force_velocity[muscle_index],
                    passive_force_relationship=passive_force[muscle_index],
                )
                derivatives.extend(np.asarray(muscle_derivative, dtype=float).reshape(-1))
                normalized_fatigue.append(1.0 - muscle_state[2] / float(model.a_scale))
            forces = [state[5 * index + 1] for index in range(n_muscles)]
            derivatives.extend(
                (
                    omega,
                    context.reduced_dynamics.acceleration(
                        theta,
                        omega,
                        forces,
                        external_crank_torque=context.external_crank_torque,
                    ),
                )
            )
            normalized_fatigue = np.asarray(normalized_fatigue)
            return np.concatenate(
                (np.asarray(derivatives), normalized_fatigue, normalized_fatigue**2)
            )

        solution = solve_ivp(
            rhs,
            (interval_start, interval_start + dt),
            augmented,
            method="DOP853",
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        if not solution.success or not np.all(np.isfinite(solution.y)):
            raise RuntimeError(
                f"DOP853 failed at interval {interval}: {solution.message}"
            )
        augmented = solution.y[:, -1]
        omega_minimum = min(omega_minimum, float(np.min(solution.y[n_states - 1])))
        omega_maximum = max(omega_maximum, float(np.max(solution.y[n_states - 1])))
        function_evaluations += int(solution.nfev)

    return {
        "final_state": augmented[:n_states],
        "fatigue_auc_by_muscle": augmented[n_states : n_states + n_muscles],
        "fatigue_objective_by_muscle": 10_000.0 * augmented[n_states + n_muscles :],
        "omega_minimum": omega_minimum,
        "omega_maximum": omega_maximum,
        "function_evaluations": function_evaluations,
    }


def _capacity_ratios(state: np.ndarray, context: ReplayContext) -> dict[str, float]:
    return {
        model.muscle_name: float(state[5 * index + 2] / model.a_scale)
        for index, model in enumerate(context.muscle_models)
    }


def _policy_metrics(
    rollout: dict,
    *,
    initial_theta: float,
    cycle_count: int,
    context: ReplayContext,
) -> dict:
    expected_theta = initial_theta - cycle_count * 2.0 * np.pi
    angle_error = float(rollout["final_state"][-2] - expected_theta)
    fast_violation = max(0.0, context.velocity_lower - rollout["omega_minimum"])
    slow_violation = max(0.0, rollout["omega_maximum"] - context.velocity_upper)
    return {
        "terminal_angle_error_rad": angle_error,
        "omega_minimum_rad_s": rollout["omega_minimum"],
        "omega_maximum_rad_s": rollout["omega_maximum"],
        "maximum_velocity_violation_rad_s": max(fast_violation, slow_violation),
        "passes_terminal_angle": abs(angle_error) <= context.terminal_angle_tolerance,
        "passes_velocity_bounds": max(fast_violation, slow_violation) <= 1e-8,
        "capacity_ratios": _capacity_ratios(rollout["final_state"], context),
        "fatigue_auc": float(np.sum(rollout["fatigue_auc_by_muscle"])),
        "fatigue_objective": float(np.sum(rollout["fatigue_objective_by_muscle"])),
        "function_evaluations": rollout["function_evaluations"],
    }


def replay_dropout_cycle(
    cycle: int,
    *,
    states: dict[str, np.ndarray],
    controls: dict[str, np.ndarray],
    context: ReplayContext,
    propagation_cycles: int,
) -> dict:
    keys = state_keys(context)
    first_node = (cycle - 1) * context.stimulations_per_cycle
    initial_state = _state_vector(states, keys, first_node)
    available_cycles = next(iter(controls.values())).size // context.stimulations_per_cycle
    horizon = min(propagation_cycles, available_cycles - cycle + 1)
    optimized_controls = [
        _control_cycle(controls, context, target)
        for target in range(cycle, cycle + horizon)
    ]
    degraded_controls = [
        _control_cycle(controls, context, cycle - 1), *optimized_controls[1:]
    ]
    baseline = integrate_controls(initial_state, optimized_controls, context)
    degraded = integrate_controls(initial_state, degraded_controls, context)
    baseline_metrics = _policy_metrics(
        baseline,
        initial_theta=float(initial_state[-2]),
        cycle_count=horizon,
        context=context,
    )
    degraded_metrics = _policy_metrics(
        degraded,
        initial_theta=float(initial_state[-2]),
        cycle_count=horizon,
        context=context,
    )
    optimized_terminal = _state_vector(
        states,
        keys,
        (cycle - 1 + horizon) * context.stimulations_per_cycle,
    )
    return {
        "cycle": cycle,
        "propagation_cycles": horizon,
        "baseline": baseline_metrics,
        "hold_previous": degraded_metrics,
        "baseline_transcription_terminal_theta_error_rad": float(
            baseline["final_state"][-2] - optimized_terminal[-2]
        ),
        "baseline_transcription_terminal_omega_error_rad_s": float(
            baseline["final_state"][-1] - optimized_terminal[-1]
        ),
        "hold_minus_baseline_terminal_theta_rad": float(
            degraded["final_state"][-2] - baseline["final_state"][-2]
        ),
        "hold_minus_baseline_terminal_omega_rad_s": float(
            degraded["final_state"][-1] - baseline["final_state"][-1]
        ),
        "hold_minus_baseline_fatigue_auc": (
            degraded_metrics["fatigue_auc"] - baseline_metrics["fatigue_auc"]
        ),
        "hold_minus_baseline_fatigue_objective": (
            degraded_metrics["fatigue_objective"]
            - baseline_metrics["fatigue_objective"]
        ),
        "hold_minus_baseline_capacity_ratio": {
            muscle: (
                degraded_metrics["capacity_ratios"][muscle]
                - baseline_metrics["capacity_ratios"][muscle]
            )
            for muscle in baseline_metrics["capacity_ratios"]
        },
    }


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"cycle_count": 0}

    def distribution(values):
        array = np.asarray(tuple(values), dtype=float)
        return {
            "minimum": float(np.min(array)),
            "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90)),
            "maximum": float(np.max(array)),
        }

    muscles = tuple(rows[0]["hold_minus_baseline_capacity_ratio"])
    return {
        "cycle_count": len(rows),
        "baseline_physical_count": sum(
            row["baseline"]["passes_terminal_angle"]
            and row["baseline"]["passes_velocity_bounds"]
            for row in rows
        ),
        "hold_previous_physical_count": sum(
            row["hold_previous"]["passes_terminal_angle"]
            and row["hold_previous"]["passes_velocity_bounds"]
            for row in rows
        ),
        "absolute_hold_minus_baseline_terminal_theta_rad": distribution(
            abs(row["hold_minus_baseline_terminal_theta_rad"]) for row in rows
        ),
        "absolute_hold_minus_baseline_terminal_omega_rad_s": distribution(
            abs(row["hold_minus_baseline_terminal_omega_rad_s"]) for row in rows
        ),
        "hold_minus_baseline_fatigue_auc": distribution(
            row["hold_minus_baseline_fatigue_auc"] for row in rows
        ),
        "hold_minus_baseline_fatigue_objective": distribution(
            row["hold_minus_baseline_fatigue_objective"] for row in rows
        ),
        "hold_minus_baseline_capacity_ratio": {
            muscle: distribution(
                row["hold_minus_baseline_capacity_ratio"][muscle] for row in rows
            )
            for muscle in muscles
        },
    }


def analyze_solver_attempts(result: dict, budgets: tuple[int, ...] = (3, 5, 10, 20, 30)) -> dict:
    """Quantify which iteration caps would interrupt historically successful RHO."""

    accounting = result.get("solver_attempt_accounting") or {}
    attempts = accounting.get("attempts") or []
    native_successes = [
        attempt
        for attempt in attempts
        if attempt.get("certifier") == "target_solver"
        and attempt.get("advanced") is True
        and attempt.get("status") == 0
        and attempt.get("iterations") is not None
    ]
    successful_iterations = np.asarray(
        [attempt["iterations"] for attempt in native_successes], dtype=float
    )
    failed_native = [
        attempt
        for attempt in attempts
        if attempt.get("certifier") == "target_solver"
        and attempt.get("advanced") is False
    ]
    infeasibilities = np.asarray(
        [
            (attempt.get("feasibility") or {}).get(
                "effective_primal_infeasibility", np.nan
            )
            for attempt in failed_native
        ],
        dtype=float,
    )
    infeasibilities = infeasibilities[np.isfinite(infeasibilities)]

    by_rho = {}
    for attempt in attempts:
        by_rho.setdefault(int(attempt["target_rho"]), []).append(attempt)
    first_native_successes = [
        rho_attempts[0]
        for rho_attempts in by_rho.values()
        if rho_attempts[0].get("certifier") == "target_solver"
        and rho_attempts[0].get("advanced") is True
        and rho_attempts[0].get("status") == 0
    ]
    first_successful_iterations = np.asarray(
        [attempt["iterations"] for attempt in first_native_successes], dtype=float
    )
    fallback_rhos = {
        rho
        for rho, rho_attempts in by_rho.items()
        if any(
            attempt.get("certifier") == "ipopt_radau"
            and attempt.get("advanced") is True
            for attempt in rho_attempts
        )
    }

    def next_first_attempts(source_rhos):
        return [
            by_rho[rho + 1][0]
            for rho in source_rhos
            if rho + 1 in by_rho
        ]

    ordinary_rhos = {
        rho
        for rho, rho_attempts in by_rho.items()
        if len(rho_attempts) == 1
        and rho_attempts[0].get("certifier") == "target_solver"
        and rho_attempts[0].get("advanced") is True
    }
    after_fallback = next_first_attempts(fallback_rhos)
    after_ordinary = next_first_attempts(ordinary_rhos)

    def transition_summary(rows):
        if not rows:
            return {"count": 0, "first_attempt_failure_fraction": None}
        failed = sum(attempt.get("advanced") is not True for attempt in rows)
        return {
            "count": len(rows),
            "first_attempt_failure_count": failed,
            "first_attempt_failure_fraction": failed / len(rows),
            "median_iterations": float(
                np.median([attempt["iterations"] for attempt in rows])
            ),
        }

    return {
        "first_attempt_native_success_count": int(first_successful_iterations.size),
        "first_attempt_native_success_iteration_distribution": (
            None
            if first_successful_iterations.size == 0
            else {
                "median": float(np.median(first_successful_iterations)),
                "p90": float(np.percentile(first_successful_iterations, 90)),
                "p95": float(np.percentile(first_successful_iterations, 95)),
                "p99": float(np.percentile(first_successful_iterations, 99)),
                "maximum": int(np.max(first_successful_iterations)),
            }
        ),
        "first_attempt_historical_success_fraction_within_budget": {
            str(budget): (
                None
                if first_successful_iterations.size == 0
                else float(np.mean(first_successful_iterations <= budget))
            )
            for budget in budgets
        },
        "native_success_count": int(successful_iterations.size),
        "native_success_iteration_distribution": (
            None
            if successful_iterations.size == 0
            else {
                "median": float(np.median(successful_iterations)),
                "p90": float(np.percentile(successful_iterations, 90)),
                "p95": float(np.percentile(successful_iterations, 95)),
                "p99": float(np.percentile(successful_iterations, 99)),
                "maximum": int(np.max(successful_iterations)),
            }
        ),
        "historical_success_fraction_within_budget": {
            str(budget): (
                None
                if successful_iterations.size == 0
                else float(np.mean(successful_iterations <= budget))
            )
            for budget in budgets
        },
        "failed_native_attempt_count": len(failed_native),
        "failed_native_primal_infeasibility_distribution": (
            None
            if infeasibilities.size == 0
            else {
                "minimum": float(np.min(infeasibilities)),
                "median": float(np.median(infeasibilities)),
                "p90": float(np.percentile(infeasibilities, 90)),
                "maximum": float(np.max(infeasibilities)),
                "strict_1e-5_count": int(np.sum(infeasibilities <= 1e-5)),
                "relaxed_1e-3_count": int(np.sum(infeasibilities <= 1e-3)),
            }
        ),
        "fallback_rho_count": len(fallback_rhos),
        "next_rho_after_fallback": transition_summary(after_fallback),
        "next_rho_after_ordinary_native_success": transition_summary(after_ordinary),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("result_json", type=Path)
    parser.add_argument(
        "--cycles",
        type=parse_cycle_indices,
        default=(10, 97, 150, 346, 450, 657),
    )
    parser.add_argument("--propagation-cycles", type=int, default=1)
    parser.add_argument("--reduced-profile", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.propagation_cycles < 1:
        raise ValueError("--propagation-cycles must be positive.")
    repository_root = Path(__file__).resolve().parents[2]
    payload = json.loads(args.result_json.read_text(encoding="utf-8"))
    configuration = _configuration(payload)
    context = build_context(
        configuration,
        repository_root=repository_root,
        reduced_profile=args.reduced_profile,
    )
    states, controls, available_cycles = load_trajectory(args.trajectory, context)
    invalid = [cycle for cycle in args.cycles if cycle > available_cycles]
    if invalid:
        raise ValueError(
            f"Requested cycles exceed the {available_cycles}-cycle trajectory: {invalid}."
        )
    rows = [
        replay_dropout_cycle(
            cycle,
            states=states,
            controls=controls,
            context=context,
            propagation_cycles=args.propagation_cycles,
        )
        for cycle in args.cycles
    ]
    output = {
        "policy": "hold_previous_controls_and_roll_out_from_last_certified_state",
        "interpretation": (
            "Open-loop sensitivity test; future PW are not reoptimized from the altered state."
        ),
        "configuration": {
            "external_crank_torque_nm": context.external_crank_torque,
            "terminal_angle_tolerance_rad": context.terminal_angle_tolerance,
            "velocity_bounds_rad_s": [context.velocity_lower, context.velocity_upper],
            "stimulations_per_cycle": context.stimulations_per_cycle,
            "available_cycles": available_cycles,
        },
        "summary": summarize_rows(rows),
        "solver_attempt_analysis": analyze_solver_attempts(_result(payload)),
        "cycles": rows,
    }
    rendered = json.dumps(output, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
