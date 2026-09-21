"""Deterministic isolated phase playback screen; no OCP/RHO fallback is run."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.phase_feedback_correction import SourcePhasePulseWidthFeedback
from cocofest.optimization.phase_locked_pw_preview import PhaseLockedPulseWidthPreviewPlant, PhaseTablePulseWidthPreview
from scripts.validate_phase_feedback_playback import (
    DEFAULT_SOURCE, DEFAULT_PROFILE, DEFAULT_MODEL_CONFIG, load_playback_inputs,
    _mechanical_gain_provider, evaluate_playback_gates,
)


def run_candidate(inputs, *, name, phase_gain=0.0, speed_gain=0.0, cycles=1,
                  old_mapping=False, time_replay=False, ding_substeps=16, mechanics_substeps=8):
    source, profile = inputs.source, inputs.profile
    phase = profile.kinematics.direction * (source.shooting_states[-2] - profile.kinematics.theta_origin)
    if time_replay:
        policy = lambda observation: inputs.phase_commands[
            int(np.floor(observation.time_s/source.dt+1e-10)) % source.intervals_per_cycle].copy()
    elif old_mapping:
        policy = PhaseTablePulseWidthPreview(inputs.phase_commands)
    else:
        policy = SourcePhasePulseWidthFeedback(
            phase_nodes=phase, directed_omega_nodes=profile.kinematics.direction * source.shooting_states[-1],
            commands=inputs.phase_commands, period_s=source.duration,
            direction=profile.kinematics.direction,
            effectiveness=lambda theta: profile.coefficient_values(theta)["muscle_effectiveness"],
            lower_bounds=[p.pd0 for p in inputs.parameters],
            upper_bounds=[p.pulse_width_max for p in inputs.parameters],
            maximum_slew_s=500e-6, phase_gain_s_per_rad=phase_gain,
            speed_gain_s2_per_rad=speed_gain,
        )
    plant = PhaseLockedPulseWidthPreviewPlant(
        profile, inputs.parameters, calcium_amplitudes=inputs.calcium_amplitudes,
        stimulation_frequency_hz=1/source.dt, ding_integration_substeps=ding_substeps,
        mechanics_integration_substeps=mechanics_substeps,
        external_crank_torque=inputs.external_crank_torque,
        mechanical_gain_provider=_mechanical_gain_provider(source, profile),
    )
    started = time.perf_counter()
    result = {"name": name, "phase_gain_s_per_rad": phase_gain,
              "speed_gain_s2_per_rad": speed_gain, "cycles": cycles,
              "ding_substeps": ding_substeps, "mechanics_substeps": mechanics_substeps}
    try:
        trajectory = plant.run(
            initial_theta_rad=source.initial_state[-2], initial_omega_rad_s=source.initial_state[-1],
            initial_ding_states=inputs.ding_states, preview=policy,
            phase_count=source.intervals_per_cycle, stimulation_ticks=source.intervals_per_cycle*cycles+1,
            refresh_on_stimulation_tick=not old_mapping,
        )
        gates, reason = evaluate_playback_gates(
            trajectory, inputs, expected_tick_count=source.intervals_per_cycle*cycles+1,
            clock_tolerance_s=1e-12, maximum_terminal_phase_error_rad=0.5,
            minimum_directed_omega_rad_s=0.1, maximum_directed_omega_rad_s=15,
            maximum_pw_slew_s=500e-6,
        )
        result.update(status="passed" if reason is None else "rejected", gates=gates, failure_reason=reason)
    except (RuntimeError, ValueError, FloatingPointError) as error:
        result.update(status="rejected", failure_reason=str(error))
    result["wall_s"] = time.perf_counter()-started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--cycle-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(".cache/phase-feedback-screen/report.json"))
    args = parser.parse_args()
    inputs = load_playback_inputs(DEFAULT_SOURCE, DEFAULT_PROFILE, model_config=DEFAULT_MODEL_CONFIG,
                                 legacy_formulation="dynamic", cycle_duration=1.0, cycle_index=args.cycle_index)
    candidates = [("uniform_phase_baseline", 0, 0, True), ("source_phase_mapping", 0, 0, False)]
    candidates += [(f"kp{kp:g}_kd{kd:g}", kp, kd, False)
                   for kp, kd in [(20e-6, 5e-6), (50e-6, 10e-6), (100e-6, 20e-6), (200e-6, 40e-6)]]
    rows = []
    for name, kp, kd, old in candidates:
        row = run_candidate(inputs, name=name, phase_gain=kp, speed_gain=kd,
                            cycles=args.cycles, old_mapping=old)
        rows.append(row)
        print(json.dumps(row), flush=True)
    rows.append(run_candidate(inputs, name="source_phase_mapping_fine_integration", cycles=args.cycles,
                              ding_substeps=32, mechanics_substeps=16))
    rows.append(run_candidate(inputs, name="archived_time_replay", cycles=args.cycles, time_replay=True))
    rows.append(run_candidate(inputs, name="archived_time_replay_fine_integration", cycles=args.cycles,
                              time_replay=True, ding_substeps=32, mechanics_substeps=16))
    report = {"scope": "phase_feedback_screen_not_rho_or_fatigue_preview_validation",
              "source_provenance": inputs.source.provenance, "source_cycle_index": args.cycle_index,
              "settings": {"maximum_phase_error_rad": 0.5, "omega_bounds_rad_s": [0.1,15],
                           "maximum_pw_slew_s":500e-6, "maximum_correction_s":100e-6},
              "candidates": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
