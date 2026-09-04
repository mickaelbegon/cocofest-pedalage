"""Scientific controls for the terminal-reserve proxy on a toy RHO.

This deliberately simple global grid search does not validate clinical
endurance. It checks the qualitative claims that may legitimately be made at
Level 1 and contains a negative control showing why a mechanical rollout is
still required.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from cocofest.optimization.muscle_reserve import (
    capacity_reserve_metrics,
    physiological_capacity_ratios,
)


@dataclass(frozen=True)
class ToyRhoStep:
    force: np.ndarray
    capacity: np.ndarray
    task_residual: float
    hard_minimum: float
    smooth_minimum: float
    smoothing_gap: float
    smoothing_bound: float
    reserve_penalty: float


def _globally_solve_two_muscle_rho(
    capacity: np.ndarray,
    mechanical_effectiveness: np.ndarray,
    damage_per_force: np.ndarray,
    demand: float,
    reserve_lambda: float,
    *,
    temperature: float = 0.005,
    grid_size: int = 4001,
) -> ToyRhoStep | None:
    """Solve a transparent normalized one-step allocation problem on a grid."""
    capacity = np.asarray(capacity, dtype=float)
    mechanical_effectiveness = np.asarray(mechanical_effectiveness, dtype=float)
    damage_per_force = np.asarray(damage_per_force, dtype=float)
    force_0 = np.linspace(0.0, capacity[0], grid_size)
    force_1 = (demand - mechanical_effectiveness[0] * force_0) / mechanical_effectiveness[1]
    forces = np.column_stack((force_0, force_1))
    next_capacity = capacity - forces * damage_per_force
    feasible = (
        (force_1 >= 0.0)
        & (force_1 <= capacity[1])
        & np.all(next_capacity >= 0.0, axis=1)
    )
    if not np.any(feasible):
        return None

    forces = forces[feasible]
    next_capacity = next_capacity[feasible]
    smooth_minimum = -temperature * (
        np.logaddexp(
            -next_capacity[:, 0] / temperature,
            -next_capacity[:, 1] / temperature,
        )
        - np.log(2.0)
    )
    historical_cost = np.sum(np.square(1.0 - next_capacity), axis=1)
    total_cost = historical_cost + reserve_lambda * (1.0 - smooth_minimum)
    optimum = int(np.argmin(total_cost))
    selected_force = forces[optimum]
    selected_capacity = next_capacity[optimum]
    metrics = capacity_reserve_metrics(
        selected_capacity,
        np.ones(2),
        temperature=temperature,
    )
    return ToyRhoStep(
        force=selected_force,
        capacity=selected_capacity,
        task_residual=float(
            mechanical_effectiveness @ selected_force - demand
        ),
        hard_minimum=metrics.minimum_ratio,
        smooth_minimum=metrics.smooth_minimum_ratio,
        smoothing_gap=metrics.optimism_gap,
        smoothing_bound=metrics.maximum_optimism_gap,
        reserve_penalty=metrics.penalty,
    )


def _rollout_toy_rho(
    capacity: np.ndarray,
    mechanical_effectiveness: np.ndarray,
    damage_per_force: np.ndarray,
    demand: float,
    reserve_lambda: float,
    *,
    maximum_cycles: int = 200,
) -> list[ToyRhoStep]:
    steps = []
    current_capacity = np.asarray(capacity, dtype=float)
    for _ in range(maximum_cycles):
        step = _globally_solve_two_muscle_rho(
            current_capacity,
            mechanical_effectiveness,
            damage_per_force,
            demand,
            reserve_lambda,
        )
        if step is None:
            break
        physiological_capacity_ratios(step.capacity, np.ones(2))
        steps.append(step)
        current_capacity = step.capacity
    return steps


def test_symmetric_toy_rho_preserves_symmetry_and_task():
    step = _globally_solve_two_muscle_rho(
        capacity=np.ones(2),
        mechanical_effectiveness=np.ones(2),
        damage_per_force=np.full(2, 0.03),
        demand=0.4,
        reserve_lambda=1.0,
    )

    assert step is not None
    np.testing.assert_allclose(step.force, [0.2, 0.2], atol=1e-12)
    assert abs(step.task_residual) < 1e-12
    assert step.smoothing_gap <= step.smoothing_bound + 1e-14


def test_more_terminal_weight_cannot_worsen_its_global_grid_optimum():
    parameters = {
        "capacity": np.array([0.60, 0.90]),
        "mechanical_effectiveness": np.ones(2),
        "damage_per_force": np.array([0.03, 0.05]),
        "demand": 0.4,
    }
    steps = [
        _globally_solve_two_muscle_rho(**parameters, reserve_lambda=value)
        for value in (0.0, 0.1, 1.0, 10.0)
    ]

    assert all(step is not None for step in steps)
    penalties = np.array([step.reserve_penalty for step in steps])
    assert np.all(np.diff(penalties) <= 1e-10)
    assert all(abs(step.task_residual) < 1e-12 for step in steps)


def test_toy_rho_responds_to_identified_damage_parameters_without_muscle_weights():
    common = {
        "capacity": np.full(2, 0.8),
        "mechanical_effectiveness": np.ones(2),
        "demand": 0.4,
        "reserve_lambda": 1.0,
    }
    equal_damage = _globally_solve_two_muscle_rho(
        **common,
        damage_per_force=np.full(2, 0.04),
    )
    unequal_damage = _globally_solve_two_muscle_rho(
        **common,
        damage_per_force=np.array([0.02, 0.08]),
    )

    assert equal_damage is not None and unequal_damage is not None
    np.testing.assert_allclose(equal_damage.force, [0.2, 0.2], atol=1e-12)
    assert unequal_damage.force[0] > unequal_damage.force[1]


def test_negative_control_terminal_reserve_is_not_an_endurance_predictor():
    """A low-reserve, low-effectiveness muscle can make the proxy myopic.

    Protecting it transfers damage to the mechanically useful muscle. This
    intentionally demonstrates a case where a large terminal weight completes
    fewer cycles, motivating the force/PW rollout in Levels 3 and 4.
    """
    parameters = {
        "capacity": np.array([0.4, 1.0]),
        "mechanical_effectiveness": np.array([0.1, 1.0]),
        "damage_per_force": np.array([0.001, 0.05]),
        "demand": 0.4,
    }
    baseline = _rollout_toy_rho(**parameters, reserve_lambda=0.0)
    terminal_reserve = _rollout_toy_rho(**parameters, reserve_lambda=10.0)

    assert len(baseline) == 35
    assert len(terminal_reserve) == 33
    assert len(terminal_reserve) < len(baseline)
    assert all(abs(step.task_residual) < 1e-12 for step in baseline + terminal_reserve)


@pytest.mark.parametrize("reserve_lambda", [0.0, 0.1, 1.0])
def test_toy_rho_is_invariant_to_muscle_permutation(reserve_lambda):
    capacity = np.array([0.64, 0.83])
    effectiveness = np.array([0.7, 1.1])
    damage = np.array([0.02, 0.06])
    direct = _globally_solve_two_muscle_rho(
        capacity,
        effectiveness,
        damage,
        demand=0.4,
        reserve_lambda=reserve_lambda,
    )
    reverse = _globally_solve_two_muscle_rho(
        capacity[::-1],
        effectiveness[::-1],
        damage[::-1],
        demand=0.4,
        reserve_lambda=reserve_lambda,
    )

    assert direct is not None and reverse is not None
    np.testing.assert_allclose(direct.force, reverse.force[::-1], atol=3e-4)
    np.testing.assert_allclose(direct.capacity, reverse.capacity[::-1], atol=1e-5)
    # The grid is parameterized by the first muscle capacity, so reversing
    # unequal muscles changes its physical spacing slightly. The tolerance is
    # below one grid interval and tests permutation invariance up to that known
    # discretization error.
    assert direct.reserve_penalty == pytest.approx(reverse.reserve_penalty, abs=5e-6)
