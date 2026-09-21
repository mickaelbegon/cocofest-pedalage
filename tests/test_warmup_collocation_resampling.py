"""Unit tests for physical-time transfer between Radau warm-start grids."""

from types import SimpleNamespace

import numpy as np

import examples.fes_multibody.cycling.cycling_pulse_width_mhe_acados_periodic as periodic


def test_radau5_to_radau3_warmup_resampling_uses_physical_stage_times():
    """A linear physical trace must stay linear after a 5 -> 3 transfer.

    This deliberately uses non-uniform Radau stages.  Interpolating against
    normalized column indices (the former behavior) would not reproduce this
    trace at the target stages.
    """

    interval_count = 2
    source_time = periodic._collocation_state_time_grid(
        state_length=interval_count * (5 + 1) + 1,
        interval_count=interval_count,
    )
    target_time = periodic._collocation_state_time_grid(
        state_length=interval_count * (3 + 1) + 1,
        interval_count=interval_count,
    )
    source_values = np.vstack((source_time, 1.0 - 3.0 * source_time))

    transferred = periodic._resample_warmup_data(
        source_values,
        target_len=target_time.size,
        has_terminal_node=True,
        source_interval_count=interval_count,
        target_interval_count=interval_count,
    )

    np.testing.assert_allclose(transferred[0], target_time, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        transferred[1], 1.0 - 3.0 * target_time, rtol=0.0, atol=1e-14
    )


def test_collocation_resampling_retains_right_shooting_value_at_duplicate_boundary():
    """The terminal shooting column wins over the same-time Radau stage."""

    interval_count = 2
    source_time = periodic._collocation_state_time_grid(
        state_length=interval_count * (5 + 1) + 1,
        interval_count=interval_count,
    )
    target_time = periodic._collocation_state_time_grid(
        state_length=interval_count * (3 + 1) + 1,
        interval_count=interval_count,
    )
    source_values = source_time[np.newaxis, :].copy()
    duplicate_right_index = np.flatnonzero(np.isclose(source_time, 0.5))[-1]
    source_values[0, duplicate_right_index - 1] = -100.0
    source_values[0, duplicate_right_index] = 7.0

    transferred = periodic._resample_warmup_data(
        source_values,
        target_len=target_time.size,
        has_terminal_node=True,
        source_interval_count=interval_count,
        target_interval_count=interval_count,
    )

    np.testing.assert_allclose(transferred[0, np.isclose(target_time, 0.5)], 7.0)


def test_warmup_adapter_passes_collocation_topology_to_state_resampling():
    """The public warmup adapter must use the Radau path, not its legacy fallback."""

    interval_count = 2
    source_time = periodic._collocation_state_time_grid(
        state_length=interval_count * (5 + 1) + 1,
        interval_count=interval_count,
    )
    target_time = periodic._collocation_state_time_grid(
        state_length=interval_count * (3 + 1) + 1,
        interval_count=interval_count,
    )
    seed = periodic._WarmupSolutionAdapter(
        states={"F_m": source_time[np.newaxis, :]},
        controls={"pw_m": np.zeros((1, interval_count))},
        metadata={"producer_collocation_method": "radau"},
    )
    target = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                model=SimpleNamespace(muscles_dynamics_model=[]),
                dynamics_type=SimpleNamespace(
                    ode_solver=SimpleNamespace(method="radau")
                ),
                x_init={"F_m": SimpleNamespace(init=np.zeros((1, target_time.size)))},
                u_init={"pw_m": SimpleNamespace(init=np.zeros((1, interval_count)))},
            )
        ]
    )

    adapted = periodic._adapt_warmup_solution_to_periodic_nodes(target, seed)

    np.testing.assert_allclose(adapted.decision_states()["F_m"][0], target_time)
