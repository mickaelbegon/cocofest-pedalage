"""Isolated phase-feedback prototype for pulse-width preview playback.

The controller deliberately keeps two different clocks:

* crossing a measured crank-phase boundary refreshes the pending preview;
* the pending pulse width is latched only on the next fixed stimulation tick.

This distinction is essential for the periodic-node Ding formulation: a phase
crossing must never move a stimulation time or change the declared frequency.
The plant is intentionally independent from the RHO and GUI adapters.  It uses
the public numerical Ding interval map and ``ReducedCyclingDynamics.acceleration``
through small, duck-typed interfaces so it can also be exercised with a
synthetic reduced plant.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .adaptive_moment_rollout import (
    DingPulseWidthParameters,
    propagate_ding_pulse_width_interval,
)


TWO_PI = 2.0 * math.pi


def _finite_scalar(value: float, name: str, *, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.0):
        qualifier = "finite and strictly positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return value


def _json_vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


@dataclass(frozen=True)
class PhaseObservation:
    """Measured plant state at an initialisation or phase-crossing event."""

    time_s: float
    theta_rad: float
    unwrapped_phase_rad: float
    wrapped_phase_rad: float
    sector_index: int
    omega_rad_s: float
    forces_n: np.ndarray


@dataclass(frozen=True)
class PhaseCommandEvent:
    """A preview refresh; its command remains pending until a stimulus tick."""

    event_id: int
    kind: str
    observation: PhaseObservation
    requested_pulse_widths_s: np.ndarray


@dataclass(frozen=True)
class StimulusTickAudit:
    """Command and measured state at one fixed-frequency stimulation tick."""

    tick_index: int
    time_s: float
    theta_rad: float
    unwrapped_phase_rad: float
    wrapped_phase_rad: float
    sector_index: int
    omega_rad_s: float
    forces_n: np.ndarray
    applied_pulse_widths_s: np.ndarray
    source_event_id: int


@dataclass(frozen=True)
class PhaseLockedPreviewResult:
    """Auditable result of a fixed-clock, phase-feedback preview playback."""

    stimulation_period_s: float
    phase_count: int
    final_theta_rad: float
    final_omega_rad_s: float
    final_ding_states: np.ndarray
    phase_events: tuple[PhaseCommandEvent, ...]
    stimulus_ticks: tuple[StimulusTickAudit, ...]

    def audit(self) -> dict[str, Any]:
        """Return a JSON-compatible phase/omega/force and timing audit."""

        times = np.asarray([tick.time_s for tick in self.stimulus_ticks])
        expected = np.arange(times.size, dtype=float) * self.stimulation_period_s
        tick_error = float(np.max(np.abs(times - expected))) if times.size else 0.0
        event_boundary_error = 0.0
        crossings = [event for event in self.phase_events if event.kind == "phase_crossing"]
        if crossings:
            width = TWO_PI / self.phase_count
            event_boundary_error = max(
                abs(
                    event.observation.unwrapped_phase_rad / width
                    - round(event.observation.unwrapped_phase_rad / width)
                )
                * width
                for event in crossings
            )
        return {
            "schema_version": 1,
            "timing": {
                "stimulation_period_s": float(self.stimulation_period_s),
                "maximum_stimulation_tick_error_s": tick_error,
                "pulse_width_refresh_clock": (
                    "fixed_stimulation_tick" if any(event.kind == "stimulus_refresh" for event in self.phase_events)
                    else "measured_phase_crossing"
                ),
                "pulse_width_application_clock": "fixed_stimulation_tick",
                "phase_crossings_create_stimuli": False,
            },
            "maximum_phase_boundary_error_rad": event_boundary_error,
            "stimulus_ticks": [
                {
                    "tick_index": tick.tick_index,
                    "time_s": tick.time_s,
                    "theta_rad": tick.theta_rad,
                    "unwrapped_phase_rad": tick.unwrapped_phase_rad,
                    "wrapped_phase_rad": tick.wrapped_phase_rad,
                    "sector_index": tick.sector_index,
                    "omega_rad_s": tick.omega_rad_s,
                    "forces_n": _json_vector(tick.forces_n),
                    "applied_pulse_widths_s": _json_vector(tick.applied_pulse_widths_s),
                    "source_event_id": tick.source_event_id,
                }
                for tick in self.stimulus_ticks
            ],
            "phase_command_events": [
                {
                    "event_id": event.event_id,
                    "kind": event.kind,
                    "time_s": event.observation.time_s,
                    "theta_rad": event.observation.theta_rad,
                    "unwrapped_phase_rad": event.observation.unwrapped_phase_rad,
                    "wrapped_phase_rad": event.observation.wrapped_phase_rad,
                    "sector_index": event.observation.sector_index,
                    "omega_rad_s": event.observation.omega_rad_s,
                    "forces_n": _json_vector(event.observation.forces_n),
                    "requested_pulse_widths_s": _json_vector(event.requested_pulse_widths_s),
                }
                for event in self.phase_events
            ],
        }


class PhaseTablePulseWidthPreview:
    """Minimal preview policy indexed by the measured crank-phase sector."""

    def __init__(self, pulse_widths_s: Sequence[Sequence[float]] | np.ndarray):
        values = np.asarray(pulse_widths_s, dtype=float)
        if values.ndim != 2 or min(values.shape) < 1 or not np.all(np.isfinite(values)):
            raise ValueError("pulse_widths_s must be a non-empty finite (phase, muscle) matrix.")
        self._values = values.copy()

    @property
    def phase_count(self) -> int:
        return int(self._values.shape[0])

    @property
    def muscle_count(self) -> int:
        return int(self._values.shape[1])

    def __call__(self, observation: PhaseObservation) -> np.ndarray:
        return self._values[observation.sector_index].copy()


class PhaseLockedPulseWidthPreviewPlant:
    """Run reduced mechanics and full Ding states under phase-refreshed PW.

    ``reduced_dynamics`` must expose ``muscle_names``, a kinematics object with
    ``direction`` and ``theta_origin``, and the numerical ``acceleration``
    method of :class:`~cocofest.dynamics.reduced_cycling.ReducedCyclingDynamics`.
    No solver, RHO, or GUI object is accessed.
    """

    def __init__(
        self,
        reduced_dynamics,
        ding_parameters: Sequence[DingPulseWidthParameters],
        *,
        calcium_amplitudes: Sequence[float],
        stimulation_frequency_hz: float = 30.0,
        ding_integration_substeps: int = 8,
        mechanics_integration_substeps: int = 4,
        external_crank_torque: float = 0.0,
        mechanical_gains: Sequence[float | Callable[[float], float]] | None = None,
        mechanical_gain_provider: Callable[[float, float], Sequence[float]] | None = None,
    ):
        self.reduced_dynamics = reduced_dynamics
        self.parameters = tuple(ding_parameters)
        self.muscle_count = len(tuple(reduced_dynamics.muscle_names))
        if len(self.parameters) != self.muscle_count or self.muscle_count < 1:
            raise ValueError("One Ding parameter set is required per reduced muscle.")
        amplitudes = np.asarray(calcium_amplitudes, dtype=float)
        if amplitudes.shape != (self.muscle_count,) or not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < 0.0):
            raise ValueError("calcium_amplitudes must be one finite non-negative value per muscle.")
        self.calcium_amplitudes = amplitudes.copy()
        self.stimulation_frequency_hz = _finite_scalar(
            stimulation_frequency_hz, "stimulation_frequency_hz", positive=True
        )
        self.stimulation_period_s = 1.0 / self.stimulation_frequency_hz
        for name, value in (
            ("ding_integration_substeps", ding_integration_substeps),
            ("mechanics_integration_substeps", mechanics_integration_substeps),
        ):
            if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.ding_integration_substeps = int(ding_integration_substeps)
        self.mechanics_integration_substeps = int(mechanics_integration_substeps)
        self.external_crank_torque = _finite_scalar(external_crank_torque, "external_crank_torque")
        gains = (1.0,) * self.muscle_count if mechanical_gains is None else tuple(mechanical_gains)
        if len(gains) != self.muscle_count:
            raise ValueError("One mechanical gain is required per muscle.")
        self.mechanical_gains = gains
        if mechanical_gain_provider is not None and not callable(mechanical_gain_provider):
            raise ValueError("mechanical_gain_provider must be callable when provided.")
        self.mechanical_gain_provider = mechanical_gain_provider
        direction = int(reduced_dynamics.kinematics.direction)
        if direction not in (-1, 1):
            raise ValueError("Reduced kinematic direction must be -1 or 1.")
        self.direction = direction
        self.theta_origin = _finite_scalar(reduced_dynamics.kinematics.theta_origin, "theta_origin")

    def _validate_command(self, command) -> np.ndarray:
        command = np.asarray(command, dtype=float)
        if command.shape != (self.muscle_count,) or not np.all(np.isfinite(command)):
            raise ValueError("The preview command must contain one finite pulse width per muscle.")
        for index, (value, parameters) in enumerate(zip(command, self.parameters, strict=True)):
            if value < parameters.pd0 or value > parameters.pulse_width_max:
                raise ValueError(
                    f"Preview pulse width {index} lies outside its Ding [pd0, PW_max] bounds."
                )
        return command.copy()

    def _phase(self, theta: float) -> float:
        return self.direction * (float(theta) - self.theta_origin)

    def _propagate_ding(
        self,
        states: np.ndarray,
        pulse_widths: np.ndarray,
        theta: float,
        omega: float,
    ) -> np.ndarray:
        # This is intentionally one indivisible fixed-frequency Ding interval.
        # Phase events are observed after it and never split/reset calcium.
        gains = self.mechanical_gains
        if self.mechanical_gain_provider is not None:
            values = np.asarray(self.mechanical_gain_provider(theta, omega), dtype=float)
            if values.shape != (self.muscle_count,) or not np.all(np.isfinite(values)):
                raise ValueError(
                    "mechanical_gain_provider must return one finite value per muscle."
                )
            gains = tuple(float(value) for value in values)
        return np.vstack(
            [
                propagate_ding_pulse_width_interval(
                    state,
                    pulse_width=float(width),
                    duration=self.stimulation_period_s,
                    calcium_amplitude=float(amplitude),
                    mechanical_gain=gain,
                    parameters=parameters,
                    integration_substeps=self.ding_integration_substeps,
                )
                for state, width, amplitude, gain, parameters in zip(
                    states,
                    pulse_widths,
                    self.calcium_amplitudes,
                    gains,
                    self.parameters,
                    strict=True,
                )
            ]
        )

    def _propagate_mechanics(
        self,
        theta: float,
        omega: float,
        start_forces: np.ndarray,
        end_forces: np.ndarray,
    ) -> tuple[float, float]:
        step = self.stimulation_period_s / self.mechanics_integration_substeps

        def rhs(local_time: float, state: np.ndarray) -> np.ndarray:
            fraction = min(max(local_time / self.stimulation_period_s, 0.0), 1.0)
            forces = start_forces + fraction * (end_forces - start_forces)
            acceleration = self.reduced_dynamics.acceleration(
                float(state[0]),
                float(state[1]),
                forces,
                external_crank_torque=self.external_crank_torque,
            )
            return np.array([state[1], acceleration], dtype=float)

        state = np.array([theta, omega], dtype=float)
        for index in range(self.mechanics_integration_substeps):
            local_time = index * step
            k1 = rhs(local_time, state)
            k2 = rhs(local_time + step / 2.0, state + step * k1 / 2.0)
            k3 = rhs(local_time + step / 2.0, state + step * k2 / 2.0)
            k4 = rhs(local_time + step, state + step * k3)
            state += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("Reduced mechanics produced a non-finite state.")
        return float(state[0]), float(state[1])

    def run(
        self,
        *,
        initial_theta_rad: float,
        initial_omega_rad_s: float,
        initial_ding_states: Sequence[Sequence[float]] | np.ndarray,
        preview: Callable[[PhaseObservation], Sequence[float] | np.ndarray],
        phase_count: int,
        stimulation_ticks: int,
        refresh_on_stimulation_tick: bool = False,
    ) -> PhaseLockedPreviewResult:
        """Run ``stimulation_ticks`` fixed ticks with phase-event refreshes."""

        if isinstance(phase_count, (bool, np.bool_)) or int(phase_count) != phase_count or phase_count < 1:
            raise ValueError("phase_count must be a positive integer.")
        if (
            isinstance(stimulation_ticks, (bool, np.bool_))
            or int(stimulation_ticks) != stimulation_ticks
            or stimulation_ticks < 2
        ):
            raise ValueError("stimulation_ticks must be an integer of at least two.")
        phase_count, stimulation_ticks = int(phase_count), int(stimulation_ticks)
        theta = _finite_scalar(initial_theta_rad, "initial_theta_rad")
        omega = _finite_scalar(initial_omega_rad_s, "initial_omega_rad_s")
        if self.direction * omega <= 0.0:
            raise ValueError("initial_omega_rad_s must advance in the reduced kinematic direction.")
        states = np.asarray(initial_ding_states, dtype=float)
        if states.shape != (self.muscle_count, 5) or not np.all(np.isfinite(states)):
            raise ValueError("initial_ding_states must be finite with shape (muscles, 5).")
        states = states.copy()
        phase_width = TWO_PI / phase_count
        progress = self._phase(theta)
        if progress < 0.0:
            raise ValueError("initial_theta_rad must not precede the reduced theta origin.")

        def observation(time_s, event_theta, event_progress, event_omega, forces):
            wrapped = float(event_progress % TWO_PI)
            sector = int(math.floor((wrapped + 1e-14) / phase_width)) % phase_count
            return PhaseObservation(
                time_s=float(time_s),
                theta_rad=float(event_theta),
                unwrapped_phase_rad=float(event_progress),
                wrapped_phase_rad=wrapped,
                sector_index=sector,
                omega_rad_s=float(event_omega),
                forces_n=np.asarray(forces, dtype=float).copy(),
            )

        initial_observation = observation(0.0, theta, progress, omega, states[:, 1])
        command = self._validate_command(preview(initial_observation))
        events = [PhaseCommandEvent(0, "initialization", initial_observation, command.copy())]
        ticks = [
            StimulusTickAudit(
                0,
                0.0,
                theta,
                progress,
                initial_observation.wrapped_phase_rad,
                initial_observation.sector_index,
                omega,
                states[:, 1].copy(),
                command.copy(),
                0,
            )
        ]
        source_event_id = 0

        for tick_index in range(1, stimulation_ticks):
            start_theta, start_omega, start_progress = theta, omega, progress
            start_states = states.copy()
            states = self._propagate_ding(states, command, theta, omega)
            theta, omega = self._propagate_mechanics(
                theta, omega, start_states[:, 1], states[:, 1]
            )
            progress = self._phase(theta)
            if progress <= start_progress or self.direction * omega <= 0.0:
                raise RuntimeError("The actual crank phase must remain strictly monotonic.")

            first_boundary = math.floor(start_progress / phase_width) + 1
            last_boundary = math.floor((progress + 1e-13) / phase_width)
            for boundary_index in (() if refresh_on_stimulation_tick else range(first_boundary, last_boundary + 1)):
                boundary = boundary_index * phase_width
                if boundary > progress + 1e-12:
                    continue
                fraction = (boundary - start_progress) / (progress - start_progress)
                event_time = (tick_index - 1 + fraction) * self.stimulation_period_s
                event_theta = self.theta_origin + self.direction * boundary
                event_omega = start_omega + fraction * (omega - start_omega)
                event_forces = start_states[:, 1] + fraction * (states[:, 1] - start_states[:, 1])
                current_observation = observation(
                    event_time, event_theta, boundary, event_omega, event_forces
                )
                pending = self._validate_command(preview(current_observation))
                source_event_id += 1
                events.append(
                    PhaseCommandEvent(
                        source_event_id,
                        "phase_crossing",
                        current_observation,
                        pending.copy(),
                    )
                )
                # Multiple boundaries within one 30 Hz interval are audited;
                # only the latest pending preview is latched at the next tick.
                command = pending

            tick_time = tick_index * self.stimulation_period_s
            tick_observation = observation(tick_time, theta, progress, omega, states[:, 1])
            if refresh_on_stimulation_tick:
                command = self._validate_command(preview(tick_observation))
                source_event_id += 1
                events.append(PhaseCommandEvent(source_event_id, "stimulus_refresh", tick_observation, command.copy()))
            ticks.append(
                StimulusTickAudit(
                    tick_index,
                    tick_time,
                    theta,
                    progress,
                    tick_observation.wrapped_phase_rad,
                    tick_observation.sector_index,
                    omega,
                    states[:, 1].copy(),
                    command.copy(),
                    source_event_id,
                )
            )

        return PhaseLockedPreviewResult(
            stimulation_period_s=self.stimulation_period_s,
            phase_count=phase_count,
            final_theta_rad=theta,
            final_omega_rad_s=omega,
            final_ding_states=states.copy(),
            phase_events=tuple(events),
            stimulus_ticks=tuple(ticks),
        )
