import json
from types import SimpleNamespace

import numpy as np
import pytest

import cocofest.optimization.phase_locked_pw_preview as phase_preview_module
from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.phase_locked_pw_preview import (
    PhaseLockedPulseWidthPreviewPlant,
    PhaseTablePulseWidthPreview,
)


class ConstantSpeedReducedDynamics:
    muscle_names = ("flexor",)
    kinematics = SimpleNamespace(direction=-1, theta_origin=0.0)

    def __init__(self):
        self.acceleration_calls = 0

    def acceleration(self, theta, omega, muscle_forces, *, external_crank_torque=0.0):
        del theta, omega, external_crank_torque
        assert np.asarray(muscle_forces).shape == (1,)
        self.acceleration_calls += 1
        return 0.0


def ding_parameters():
    return DingPulseWidthParameters(
        DingFatigueParameters(1200.0, 0.060601, 0.137, -1.4, 2.1e-5, 1.9e-5, 445.5),
        tauc=0.011,
        tau2=0.001,
        pd0=0.000131405,
        pdt=0.000194138,
        pulse_width_max=0.0006,
    )


def make_plant(reduced=None):
    reduced = ConstantSpeedReducedDynamics() if reduced is None else reduced
    return reduced, PhaseLockedPulseWidthPreviewPlant(
        reduced,
        (ding_parameters(),),
        calcium_amplitudes=(1.0597,),
        stimulation_frequency_hz=30.0,
        ding_integration_substeps=4,
        mechanics_integration_substeps=2,
    )


def initial_ding_state():
    return np.array([[0.16298, 20.0, 1150.0, 0.078, 0.15]])


def test_phase_crossing_refreshes_preview_but_never_moves_the_30hz_stimulus_clock(monkeypatch):
    reduced, plant = make_plant()
    preview = PhaseTablePulseWidthPreview([[0.0002], [0.0003], [0.0004], [0.0005]])
    ding_intervals = []
    real_propagate = phase_preview_module.propagate_ding_pulse_width_interval

    def counted_propagate(*args, **kwargs):
        ding_intervals.append(kwargs["duration"])
        return real_propagate(*args, **kwargs)

    monkeypatch.setattr(
        phase_preview_module,
        "propagate_ding_pulse_width_interval",
        counted_propagate,
    )

    result = plant.run(
        initial_theta_rad=0.0,
        initial_omega_rad_s=-20.0,
        initial_ding_states=initial_ding_state(),
        preview=preview,
        phase_count=preview.phase_count,
        stimulation_ticks=6,
    )

    tick_times = np.array([tick.time_s for tick in result.stimulus_ticks])
    np.testing.assert_allclose(tick_times, np.arange(6) / 30.0, atol=0.0, rtol=0.0)
    crossing = result.phase_events[1]
    assert crossing.kind == "phase_crossing"
    assert crossing.observation.unwrapped_phase_rad == pytest.approx(np.pi / 2)
    assert 2 / 30 < crossing.observation.time_s < 3 / 30
    assert not np.any(np.isclose(tick_times, crossing.observation.time_s, atol=1e-12))

    # The old PW remains active through tick 2.  The phase-refreshed PW is
    # latched at tick 3, without an extra Ding/stimulation interval.
    widths = [tick.applied_pulse_widths_s[0] for tick in result.stimulus_ticks]
    assert widths[:3] == pytest.approx([0.0002] * 3)
    assert widths[3:5] == pytest.approx([0.0003] * 2)
    assert widths[5] == pytest.approx(0.0004)
    assert result.stimulus_ticks[3].source_event_id == crossing.event_id
    # Five intervals connect six 30 Hz ticks; the between-tick phase crossing
    # did not split or add a periodic-node Ding interval.
    assert ding_intervals == pytest.approx([1 / 30] * 5)
    assert reduced.acceleration_calls > 0


def test_audit_is_json_safe_and_contains_measured_phase_omega_force_and_both_event_clocks():
    _, plant = make_plant()
    preview = PhaseTablePulseWidthPreview([[0.0002], [0.0003], [0.0004], [0.0005]])
    result = plant.run(
        initial_theta_rad=0.0,
        initial_omega_rad_s=-20.0,
        initial_ding_states=initial_ding_state(),
        preview=preview,
        phase_count=preview.phase_count,
        stimulation_ticks=5,
    )

    audit = result.audit()
    json.dumps(audit, allow_nan=False)
    assert audit["timing"] == {
        "stimulation_period_s": pytest.approx(1 / 30),
        "maximum_stimulation_tick_error_s": 0.0,
        "pulse_width_refresh_clock": "measured_phase_crossing",
        "pulse_width_application_clock": "fixed_stimulation_tick",
        "phase_crossings_create_stimuli": False,
    }
    assert len(audit["stimulus_ticks"]) == 5
    for row in audit["stimulus_ticks"] + audit["phase_command_events"]:
        assert np.isfinite(row["wrapped_phase_rad"])
        assert np.isfinite(row["omega_rad_s"])
        assert len(row["forces_n"]) == 1 and np.isfinite(row["forces_n"][0])
    assert result.final_ding_states[0, 1] != initial_ding_state()[0, 1]


def test_multiple_phase_events_inside_one_interval_keep_only_latest_pending_command_for_next_tick():
    _, plant = make_plant()
    preview = PhaseTablePulseWidthPreview([[0.0002], [0.0003], [0.0004], [0.0005]])
    result = plant.run(
        initial_theta_rad=0.0,
        initial_omega_rad_s=-120.0,
        initial_ding_states=initial_ding_state(),
        preview=preview,
        phase_count=preview.phase_count,
        stimulation_ticks=2,
    )

    crossings = result.phase_events[1:]
    assert [event.observation.sector_index for event in crossings] == [1, 2]
    assert result.stimulus_ticks[1].source_event_id == crossings[-1].event_id
    assert result.stimulus_ticks[1].applied_pulse_widths_s[0] == pytest.approx(0.0004)
    assert result.stimulus_ticks[0].time_s == 0.0
    assert result.stimulus_ticks[1].time_s == pytest.approx(1 / 30)


def test_invalid_preview_pw_and_phase_reversal_fail_closed():
    _, plant = make_plant()
    with pytest.raises(ValueError, match="outside its Ding"):
        plant.run(
            initial_theta_rad=0.0,
            initial_omega_rad_s=-20.0,
            initial_ding_states=initial_ding_state(),
            preview=lambda _observation: [0.001],
            phase_count=4,
            stimulation_ticks=2,
        )


    with pytest.raises(ValueError, match="kinematic direction"):
        plant.run(
            initial_theta_rad=0.0,
            initial_omega_rad_s=1.0,
            initial_ding_states=initial_ding_state(),
            preview=lambda _observation: [0.0002],
            phase_count=4,
            stimulation_ticks=2,
        )


def test_tick_refresh_observes_actual_state_without_extra_stimulation(monkeypatch):
    _, plant = make_plant()
    observed = []
    intervals = []
    propagate = phase_preview_module.propagate_ding_pulse_width_interval

    def policy(observation):
        observed.append(observation)
        return [0.0002]

    def track_interval(*args, **kwargs):
        intervals.append(kwargs["duration"])
        return propagate(*args, **kwargs)

    monkeypatch.setattr(phase_preview_module, "propagate_ding_pulse_width_interval", track_interval)
    result = plant.run(initial_theta_rad=0.0, initial_omega_rad_s=-120.0,
                       initial_ding_states=initial_ding_state(), preview=policy,
                       phase_count=4, stimulation_ticks=3, refresh_on_stimulation_tick=True)
    assert [event.kind for event in result.phase_events] == ["initialization", "stimulus_refresh", "stimulus_refresh"]
    np.testing.assert_allclose([observation.time_s for observation in observed], np.arange(3)/30)
    assert intervals == pytest.approx([1/30, 1/30])
    assert result.audit()["timing"]["pulse_width_refresh_clock"] == "fixed_stimulation_tick"
