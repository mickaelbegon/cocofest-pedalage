from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.phase_locked_pw_preview import (
    PhaseLockedPreviewResult,
    PhaseTablePulseWidthPreview,
    StimulusTickAudit,
)
from scripts.validate_phase_feedback_playback import (
    evaluate_playback_gates,
    extract_initial_ding_states,
    extract_phase_ordered_commands,
)


def synthetic_source():
    muscles = ("second", "first")
    state_names = (
        "omega",
        "F_first",
        "Cn_second",
        "A_second",
        "theta",
        "F_second",
        "Tau1_second",
        "Km_second",
        "Cn_first",
        "A_first",
        "Tau1_first",
        "Km_first",
    )
    initial = np.array([-6.0, 21.0, 0.2, 1100.0, -0.1, 12.0, 0.07, 0.15, 0.3, 1200.0, 0.08, 0.16])
    return SimpleNamespace(
        muscles=muscles,
        state_names=state_names,
        initial_state=initial,
        controls=np.array([[0.00020, 0.00021, 0.00022], [0.00030, 0.00031, 0.00032]]),
        intervals_per_cycle=3,
        cycles=1,
    )


def test_extraction_preserves_profile_muscle_then_cn_force_slow_state_order():
    states = extract_initial_ding_states(synthetic_source())
    np.testing.assert_array_equal(
        states,
        [[0.2, 12.0, 1100.0, 0.07, 0.15], [0.3, 21.0, 1200.0, 0.08, 0.16]],
    )


def test_archived_muscle_by_phase_controls_become_phase_by_muscle_without_permutation():
    commands = extract_phase_ordered_commands(synthetic_source())
    np.testing.assert_array_equal(
        commands,
        [[0.00020, 0.00030], [0.00021, 0.00031], [0.00022, 0.00032]],
    )
    preview = PhaseTablePulseWidthPreview(commands)
    for phase in range(3):
        observation = SimpleNamespace(sector_index=phase)
        np.testing.assert_array_equal(preview(observation), commands[phase])


def test_gate_failure_reason_is_deterministic_and_slew_uses_applied_tick_order():
    period = 1 / 30
    widths = ([0.0002], [0.00025], [0.00045])
    ticks = tuple(
        StimulusTickAudit(
            tick_index=index,
            time_s=index * period,
            theta_rad=-index,
            unwrapped_phase_rad=float(index),
            wrapped_phase_rad=float(index),
            sector_index=index,
            omega_rad_s=-6.0,
            forces_n=np.array([10.0 + index]),
            applied_pulse_widths_s=np.array(width),
            source_event_id=index,
        )
        for index, width in enumerate(widths)
    )
    result = PhaseLockedPreviewResult(
        stimulation_period_s=period,
        phase_count=3,
        final_theta_rad=-2.0,
        final_omega_rad_s=-6.0,
        final_ding_states=np.ones((1, 5)),
        phase_events=(),
        stimulus_ticks=ticks,
    )
    parameter = SimpleNamespace(pd0=0.0001, pulse_width_max=0.0006)
    inputs = SimpleNamespace(
        source=SimpleNamespace(intervals_per_cycle=3),
        profile=SimpleNamespace(kinematics=SimpleNamespace(direction=-1)),
        parameters=(parameter,),
    )
    gates, reason = evaluate_playback_gates(
        result,
        inputs,
        expected_tick_count=3,
        clock_tolerance_s=1e-12,
        maximum_terminal_phase_error_rad=10.0,
        minimum_directed_omega_rad_s=0.1,
        maximum_directed_omega_rad_s=10.0,
        maximum_pw_slew_s=0.0001,
    )
    assert gates["fixed_stimulation_clock"]["passed"]
    assert gates["pulse_width_bounds"]["passed"]
    assert not gates["pulse_width_slew"]["passed"]
    assert gates["pulse_width_slew"]["maximum_observed_step_s"] == pytest.approx(0.0002)
    assert reason == "gate_failed:pulse_width_slew"
