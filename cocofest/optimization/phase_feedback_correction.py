"""Fixed-clock source-phase playback with optional bounded mechanical feedback.

Archived shooting controls are indexed in time, not uniformly spaced angle.
This policy maps measured phase to the actual archived shooting phase knots.
Feedback is a local proportional PW correction in the direction of each
muscle's crank effectiveness; it is not an inverse dynamics or fatigue preview.
"""
from __future__ import annotations

import numpy as np


class SourcePhasePulseWidthFeedback:
    def __init__(self, *, phase_nodes, directed_omega_nodes, commands, period_s,
                 direction, effectiveness, lower_bounds, upper_bounds,
                 maximum_slew_s, phase_gain_s_per_rad=0.0,
                 speed_gain_s2_per_rad=0.0, maximum_correction_s=100e-6):
        self.phase_nodes = np.asarray(phase_nodes, dtype=float)
        self.omega_nodes = np.asarray(directed_omega_nodes, dtype=float)
        self.commands = np.asarray(commands, dtype=float)
        self.lower = np.asarray(lower_bounds, dtype=float)
        self.upper = np.asarray(upper_bounds, dtype=float)
        self.period = float(period_s)
        self.direction = int(direction)
        self.effectiveness = effectiveness
        self.slew = float(maximum_slew_s)
        self.kp = float(phase_gain_s_per_rad)
        self.kd = float(speed_gain_s2_per_rad)
        self.maximum_correction = float(maximum_correction_s)
        n = self.commands.shape[0]
        if (self.commands.ndim != 2 or n < 1 or self.phase_nodes.shape != (n + 1,)
                or self.omega_nodes.shape != (n + 1,)
                or self.lower.shape != self.commands.shape[1:] or self.upper.shape != self.lower.shape
                or np.any(np.diff(self.phase_nodes) <= 0) or np.any(self.lower > self.upper)
                or self.direction not in (-1, 1) or self.period <= 0
                or min(self.slew, self.kp, self.kd, self.maximum_correction) < 0
                or not all(np.all(np.isfinite(x)) for x in
                           (self.phase_nodes, self.omega_nodes, self.commands, self.lower, self.upper,
                            [self.period, self.slew, self.kp, self.kd, self.maximum_correction]))):
            raise ValueError("Invalid source phase, PW bounds or feedback settings.")
        self.span = float(self.phase_nodes[-1] - self.phase_nodes[0])
        self.previous = None

    def __call__(self, observation):
        # One repeated revolution uses the physical 2 pi span, preserving the
        # source's actual nonuniform knots and initial angular offset.
        relative = (observation.unwrapped_phase_rad - self.phase_nodes[0]) % (2 * np.pi)
        phase = self.phase_nodes[0] + relative
        index = int(np.clip(np.searchsorted(self.phase_nodes, phase, side="right") - 1,
                            0, self.commands.shape[0] - 1))
        base = self.commands[index]
        cycle_time = observation.time_s % self.period
        cycle_number = np.floor(observation.time_s / self.period)
        reference_phase = np.interp(cycle_time,
                                   np.linspace(0, self.period, self.phase_nodes.size),
                                   self.phase_nodes) + cycle_number * 2 * np.pi
        reference_speed = np.interp(phase, self.phase_nodes, self.omega_nodes)
        error = (self.kp * (reference_phase - observation.unwrapped_phase_rad)
                 + self.kd * (reference_speed - self.direction * observation.omega_rad_s))
        coefficients = self.direction * np.asarray(self.effectiveness(observation.theta_rad), dtype=float)
        scale = float(np.max(np.abs(coefficients)))
        correction = np.zeros_like(base) if scale == 0 else np.clip(error, -self.maximum_correction,
                                                                   self.maximum_correction) * coefficients / scale
        command = np.clip(base + correction, self.lower, self.upper)
        if self.previous is not None:
            command = np.clip(command, self.previous - self.slew, self.previous + self.slew)
        self.previous = command.copy()
        return command
