"""Validate fixed-clock phase-feedback PW playback on a reduced Ding plant.

This script consumes one archived unilateral RHO cycle as a phase-indexed PW
table.  It is a phase-feedback *playback*, not a RHO fallback and not a new
RHO solve.  Measured phase-boundary events refresh a pending command while the
periodic-node Ding model remains on its declared fixed stimulation clock.

Legacy archives are loaded only through ``solver_cross_rollout.load_source``:
an absent formulation therefore needs an explicit legacy declaration, and an
archive without embedded parameters needs an explicitly provenanced model
config.  Both declarations and their file hashes are retained in the report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.dynamics.reduced_cycling import ReducedCyclingDynamics
from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.phase_locked_pw_preview import (
    PhaseLockedPreviewResult,
    PhaseLockedPulseWidthPreviewPlant,
    PhaseTablePulseWidthPreview,
)
from cocofest.optimization.solver_cross_rollout import RolloutSource, load_source


DEFAULT_SOURCE = Path(
    "ipopt-linear-solver-150-20260904/resistance-0p10Nm/"
    "ipopt-sx-radau5-ma57-150-max2000-reduced/validated-rho-trajectory.npz"
)
DEFAULT_PROFILE = Path("benchmark-seed/reduced-cycling-fourier12.npz")
DEFAULT_MODEL_CONFIG = Path(
    ".github/benchmark-seeds/ipopt-radau5-20260904-ding-config.json"
)
DEFAULT_OUTPUT = Path(".cache/phase-feedback-playback-one-block")
DING_STATES = ("Cn", "F", "A", "Tau1", "Km")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def extract_initial_ding_states(source: RolloutSource) -> np.ndarray:
    """Extract ``(muscle, Cn/F/A/Tau1/Km)`` in reduced-profile order."""

    lookup = {name: index for index, name in enumerate(source.state_names)}
    try:
        values = [
            [source.initial_state[lookup[f"{state}_{muscle}"]] for state in DING_STATES]
            for muscle in source.muscles
        ]
    except KeyError as error:
        raise ValueError(f"Source is missing physical Ding state {error.args[0]}.") from error
    result = np.asarray(values, dtype=float)
    if result.shape != (len(source.muscles), len(DING_STATES)) or not np.all(np.isfinite(result)):
        raise ValueError("Extracted Ding state matrix is not finite and complete.")
    return result


def extract_phase_ordered_commands(source: RolloutSource) -> np.ndarray:
    """Return archived controls as ``(phase, muscle)`` without reordering."""

    controls = np.asarray(source.controls, dtype=float)
    expected = (len(source.muscles), source.intervals_per_cycle * source.cycles)
    if controls.shape != expected or not np.all(np.isfinite(controls)):
        raise ValueError(f"Source controls must have shape {expected} in profile-muscle order.")
    if source.cycles != 1:
        raise ValueError("Phase-feedback playback extraction requires exactly one source cycle.")
    return controls.T.copy()


def _ding_parameters(source: RolloutSource) -> tuple[DingPulseWidthParameters, ...]:
    pulse_width_max = float(source.metadata.get("pulse_width_maximum_s", np.nan))
    if not np.isfinite(pulse_width_max):
        raise ValueError("Source must declare pulse_width_maximum_s.")
    result = []
    for muscle in source.muscles:
        values = source.parameters[muscle]
        result.append(
            DingPulseWidthParameters(
                fatigue=DingFatigueParameters(
                    a_rest=values["a_scale"],
                    tau1_rest=values["tau1_rest"],
                    km_rest=values["km_rest"],
                    alpha_a=values["alpha_a"],
                    alpha_tau1=values["alpha_tau1"],
                    alpha_km=values["alpha_km"],
                    tau_fat=values["tau_fat"],
                ),
                tauc=values["tauc"],
                tau2=values["tau2"],
                pd0=values["pd0"],
                pdt=values["pdt"],
                pulse_width_max=pulse_width_max,
            )
        )
    return tuple(result)


def _calcium_amplitudes(source: RolloutSource) -> np.ndarray:
    """Use the exact periodic-node history amplitude from cross-rollout."""

    truncation = source.metadata.get("ding_sum_stim_truncation")
    if isinstance(truncation, bool) or not isinstance(truncation, (int, float)):
        raise ValueError("Source must declare integer ding_sum_stim_truncation.")
    truncation = int(truncation)
    if truncation < 1:
        raise ValueError("ding_sum_stim_truncation must be positive.")
    result = []
    for muscle in source.muscles:
        values = source.parameters[muscle]
        decay = np.exp(-source.dt / values["tauc"])
        history = sum(decay**index for index in range(truncation - 1))
        result.append(
            decay ** (truncation - 1)
            + (1.0 + (values["km_rest"] + 0.04) * decay) * history
        )
    return np.asarray(result, dtype=float)


def _mechanical_gain_provider(source: RolloutSource, profile: ReducedCyclingDynamics):
    flags = {
        key: source.metadata[key]
        for key in (
            "activate_force_length_relationship",
            "activate_force_velocity_relationship",
            "activate_passive_force_relationship",
        )
    }

    def provider(theta: float, omega: float) -> np.ndarray:
        if not any(flags.values()):
            return np.ones(len(source.muscles))
        force_length, force_velocity, passive = profile.muscle_relationships(theta, omega)
        if not flags["activate_force_length_relationship"]:
            force_length = np.ones(len(source.muscles))
        if not flags["activate_force_velocity_relationship"]:
            force_velocity = np.ones(len(source.muscles))
        if not flags["activate_passive_force_relationship"]:
            passive = np.zeros(len(source.muscles))
        return force_length * force_velocity + passive

    return provider


@dataclass(frozen=True)
class PlaybackInputs:
    source: RolloutSource
    profile: ReducedCyclingDynamics
    ding_states: np.ndarray
    phase_commands: np.ndarray
    parameters: tuple[DingPulseWidthParameters, ...]
    calcium_amplitudes: np.ndarray
    external_crank_torque: float


def load_playback_inputs(
    source_path: Path,
    profile_path: Path,
    *,
    model_config: Path,
    legacy_formulation: str,
    cycle_duration: float,
    cycle_index: int,
) -> PlaybackInputs:
    """Load one real cycle through the strict legacy provenance adapter."""

    profile = ReducedCyclingDynamics.load(profile_path)
    source = load_source(
        source_path,
        profile,
        model_config=model_config,
        cycle_start=cycle_index,
        cycles=1,
        cycle_duration=cycle_duration,
        formulation_override=legacy_formulation,
    )
    if source.metadata["formulation"] != "dynamic":
        raise ValueError("This coupled playback prototype requires a dynamic source.")
    torque = float(
        source.metadata.get(
            "signed_crank_torque_nm",
            source.metadata.get("constant_crank_torque", np.nan),
        )
    )
    if not np.isfinite(torque):
        raise ValueError("Source must declare a finite signed crank torque.")
    return PlaybackInputs(
        source=source,
        profile=profile,
        ding_states=extract_initial_ding_states(source),
        phase_commands=extract_phase_ordered_commands(source),
        parameters=_ding_parameters(source),
        calcium_amplitudes=_calcium_amplitudes(source),
        external_crank_torque=torque,
    )


def evaluate_playback_gates(
    result: PhaseLockedPreviewResult,
    inputs: PlaybackInputs,
    *,
    expected_tick_count: int,
    clock_tolerance_s: float,
    maximum_terminal_phase_error_rad: float,
    minimum_directed_omega_rad_s: float,
    maximum_directed_omega_rad_s: float,
    maximum_pw_slew_s: float,
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Evaluate deterministic clock, plant and command safety gates."""

    ticks = result.stimulus_ticks
    times = np.asarray([tick.time_s for tick in ticks])
    phases = np.asarray([tick.unwrapped_phase_rad for tick in ticks])
    omega = np.asarray([tick.omega_rad_s for tick in ticks])
    pulse_widths = np.stack([tick.applied_pulse_widths_s for tick in ticks])
    period = result.stimulation_period_s
    clock_error = float(np.max(np.abs(times - np.arange(times.size) * period)))
    expected_advance = (
        2.0 * np.pi * (expected_tick_count - 1) / inputs.source.intervals_per_cycle
    )
    terminal_phase_error = float(abs((phases[-1] - phases[0]) - expected_advance))
    directed_omega = inputs.profile.kinematics.direction * omega
    phase_steps = np.diff(phases)
    pd0 = np.asarray([item.pd0 for item in inputs.parameters])
    pwmax = np.asarray([item.pulse_width_max for item in inputs.parameters])
    bound_violation = max(
        float(np.max(pd0[None, :] - pulse_widths)),
        float(np.max(pulse_widths - pwmax[None, :])),
        0.0,
    )
    slew = np.abs(np.diff(pulse_widths, axis=0))
    maximum_slew = float(np.max(slew)) if slew.size else 0.0
    gates = {
        "fixed_stimulation_clock": {
            "passed": bool(len(ticks) == expected_tick_count and clock_error <= clock_tolerance_s),
            "expected_tick_count": expected_tick_count,
            "observed_tick_count": len(ticks),
            "maximum_tick_error_s": clock_error,
            "tolerance_s": clock_tolerance_s,
        },
        "phase_and_omega": {
            "passed": bool(
                np.all(np.isfinite(phases))
                and np.all(np.isfinite(omega))
                and np.all(phase_steps > 0.0)
                and np.min(directed_omega) >= minimum_directed_omega_rad_s
                and np.max(directed_omega) <= maximum_directed_omega_rad_s
                and terminal_phase_error <= maximum_terminal_phase_error_rad
            ),
            "terminal_phase_error_rad": terminal_phase_error,
            "maximum_terminal_phase_error_rad": maximum_terminal_phase_error_rad,
            "minimum_directed_omega_rad_s": float(np.min(directed_omega)),
            "maximum_directed_omega_rad_s": float(np.max(directed_omega)),
            "directed_omega_bounds_rad_s": [
                minimum_directed_omega_rad_s,
                maximum_directed_omega_rad_s,
            ],
        },
        "pulse_width_bounds": {
            "passed": bool(bound_violation <= 1e-12),
            "maximum_violation_s": bound_violation,
            "tolerance_s": 1e-12,
        },
        "pulse_width_slew": {
            "passed": bool(maximum_slew <= maximum_pw_slew_s + 1e-12),
            "maximum_observed_step_s": maximum_slew,
            "maximum_allowed_step_s": maximum_pw_slew_s,
            "threshold_basis": "explicit_validation_setting",
        },
    }
    failure_reason = next(
        (f"gate_failed:{name}" for name, gate in gates.items() if not gate["passed"]),
        None,
    )
    return gates, failure_reason


def _result_arrays(result: PhaseLockedPreviewResult) -> dict[str, np.ndarray]:
    ticks = result.stimulus_ticks
    events = result.phase_events
    return {
        "tick_time_s": np.asarray([tick.time_s for tick in ticks]),
        "tick_theta_rad": np.asarray([tick.theta_rad for tick in ticks]),
        "tick_phase_rad": np.asarray([tick.unwrapped_phase_rad for tick in ticks]),
        "tick_omega_rad_s": np.asarray([tick.omega_rad_s for tick in ticks]),
        "tick_forces_n": np.stack([tick.forces_n for tick in ticks]),
        "tick_pulse_widths_s": np.stack([tick.applied_pulse_widths_s for tick in ticks]),
        "tick_source_event_id": np.asarray([tick.source_event_id for tick in ticks], dtype=int),
        "event_time_s": np.asarray([event.observation.time_s for event in events]),
        "event_phase_rad": np.asarray([event.observation.unwrapped_phase_rad for event in events]),
        "event_omega_rad_s": np.asarray([event.observation.omega_rad_s for event in events]),
        "event_forces_n": np.stack([event.observation.forces_n for event in events]),
        "event_requested_pulse_widths_s": np.stack(
            [event.requested_pulse_widths_s for event in events]
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reduced-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--legacy-formulation", choices=("dynamic", "isokinetic"), default="dynamic")
    parser.add_argument("--cycle-duration", type=float, default=1.0)
    parser.add_argument("--cycle-index", type=int, default=0)
    parser.add_argument("--stimulation-ticks", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ding-substeps", type=int, default=16)
    parser.add_argument("--mechanics-substeps", type=int, default=8)
    parser.add_argument("--clock-tolerance-s", type=float, default=1e-12)
    parser.add_argument("--maximum-terminal-phase-error-rad", type=float, default=0.5)
    parser.add_argument("--minimum-directed-omega-rad-s", type=float, default=0.1)
    parser.add_argument("--maximum-directed-omega-rad-s", type=float, default=15.0)
    parser.add_argument("--maximum-pw-slew-us", type=float, default=500.0)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema": "cocofest-phase-feedback-playback-v1",
        "scope": "phase_feedback_playback_not_rho_fallback",
        "rho_fallback_claimed": False,
        "source": str(args.source.resolve()),
        "reduced_profile": str(args.reduced_profile.resolve()),
        "model_config": str(args.model_config.resolve()),
        "legacy_formulation_declaration": args.legacy_formulation,
        "cycle_duration_declaration_s": args.cycle_duration,
        "cycle_index": args.cycle_index,
    }
    arrays: dict[str, np.ndarray] = {}
    try:
        inputs = load_playback_inputs(
            args.source,
            args.reduced_profile,
            model_config=args.model_config,
            legacy_formulation=args.legacy_formulation,
            cycle_duration=args.cycle_duration,
            cycle_index=args.cycle_index,
        )
        tick_count = (
            inputs.source.intervals_per_cycle + 1
            if args.stimulation_ticks is None
            else args.stimulation_ticks
        )
        preview = PhaseTablePulseWidthPreview(inputs.phase_commands)
        plant = PhaseLockedPulseWidthPreviewPlant(
            inputs.profile,
            inputs.parameters,
            calcium_amplitudes=inputs.calcium_amplitudes,
            stimulation_frequency_hz=1.0 / inputs.source.dt,
            ding_integration_substeps=args.ding_substeps,
            mechanics_integration_substeps=args.mechanics_substeps,
            external_crank_torque=inputs.external_crank_torque,
            mechanical_gain_provider=_mechanical_gain_provider(inputs.source, inputs.profile),
        )
        result = plant.run(
            initial_theta_rad=inputs.source.initial_state[-2],
            initial_omega_rad_s=inputs.source.initial_state[-1],
            initial_ding_states=inputs.ding_states,
            preview=preview,
            phase_count=preview.phase_count,
            stimulation_ticks=tick_count,
        )
        gates, failure_reason = evaluate_playback_gates(
            result,
            inputs,
            expected_tick_count=tick_count,
            clock_tolerance_s=args.clock_tolerance_s,
            maximum_terminal_phase_error_rad=args.maximum_terminal_phase_error_rad,
            minimum_directed_omega_rad_s=args.minimum_directed_omega_rad_s,
            maximum_directed_omega_rad_s=args.maximum_directed_omega_rad_s,
            maximum_pw_slew_s=args.maximum_pw_slew_us * 1e-6,
        )
        report.update(
            status="passed" if failure_reason is None else "rejected",
            failure_reason=failure_reason,
            gates=gates,
            settings={
                "stimulation_frequency_hz": 1.0 / inputs.source.dt,
                "stimulation_ticks": tick_count,
                "phase_count": preview.phase_count,
                "ding_substeps": args.ding_substeps,
                "mechanics_substeps": args.mechanics_substeps,
                "maximum_pw_slew_us": args.maximum_pw_slew_us,
            },
            provenance=inputs.source.provenance,
            audit=result.audit(),
        )
        arrays = _result_arrays(result)
        arrays["source_phase_commands_s"] = inputs.phase_commands
        arrays["final_ding_states"] = result.final_ding_states
    except (ValueError, RuntimeError, FloatingPointError) as error:
        report.update(
            status="failed",
            failure_reason=f"{type(error).__name__}: {error}",
            gates={},
        )

    report_path = args.output / "report.json"
    arrays_path = args.output / "arrays.npz"
    report_path.write_text(
        json.dumps(_jsonable(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(arrays_path, **arrays)
    print(json.dumps({"status": report["status"], "failure_reason": report["failure_reason"]}))
    print(f"Report: {report_path}")
    print(f"Arrays: {arrays_path}")
    return report


if __name__ == "__main__":
    main()
