"""Exact slow Ding updates conditional on a prescribed repeated force waveform.

Each phase input is the exponentially weighted force integral from that phase,
not its endpoint force or arithmetic mean. This module does not solve force
generation, invert PW, or establish that the prescribed force remains attainable.
It is a reduction primitive for a multilevel predictor, not a long-horizon
controller or a replacement for phase-resolved feasibility checks.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .ding_fatigue_rollout import DingFatigueParameters


@dataclass(frozen=True)
class FixedForceCycleProjection:
    slow_states: np.ndarray
    cycles: int
    slow_domain_valid_at_phase_endpoints: bool
    minimum_slow_states_by_muscle: np.ndarray
    force_and_pw_feasibility_checked: bool = False


class FixedForceCycleMap:
    """Compose phase doses, then repeat the same force cycle algebraically.

    ``phase_force_integrals`` has shape (phases, muscles). Integral (k,m) is
    integral exp(-(duration_k-t)/tau_fat_m)*F_m(t) dt. Known fatigue offsets
    are preserved by propagating all three original slow states.
    """

    def __init__(self, parameters, phase_durations, phase_force_integrals):
        parameters = tuple(parameters)
        if not parameters or any(not isinstance(p, DingFatigueParameters) for p in parameters):
            raise ValueError("parameters must be a nonempty sequence of DingFatigueParameters.")
        durations = np.array(phase_durations, dtype=float, copy=True)
        doses = np.array(phase_force_integrals, dtype=float, copy=True)
        if (durations.ndim != 1 or durations.size == 0 or not np.all(np.isfinite(durations))
                or np.any(durations <= 0)):
            raise ValueError("phase_durations must be a nonempty positive finite vector.")
        if (doses.shape != (len(durations), len(parameters)) or not np.all(np.isfinite(doses))
                or np.any(doses < 0)):
            raise ValueError("phase_force_integrals must be finite nonnegative (phases, muscles) data.")
        self.parameters = parameters
        self.phase_durations = durations
        self.phase_force_integrals = doses
        self.rest = np.asarray([p.rest_state for p in parameters])
        self.alpha = np.asarray([p.alpha for p in parameters])
        self.tau_fat = np.asarray([p.tau_fat for p in parameters])
        self.period = float(durations.sum())
        if not np.isfinite(self.period):
            raise ValueError("Total cycle duration must be finite.")
        self.phase_decay = np.exp(-durations[:, None] / self.tau_fat)
        self.cycle_force_integral = np.zeros(len(parameters))
        for decay, dose in zip(self.phase_decay, doses):
            self.cycle_force_integral = decay * self.cycle_force_integral + dose
        for value in (self.phase_durations, self.phase_force_integrals, self.rest,
                      self.alpha, self.tau_fat, self.phase_decay, self.cycle_force_integral):
            value.setflags(write=False)

    @classmethod
    def from_piecewise_constant_forces(cls, parameters, phase_durations, phase_forces):
        parameters = tuple(parameters)
        durations = np.asarray(phase_durations, float)
        forces = np.asarray(phase_forces, float)
        if (durations.ndim != 1 or not parameters or
                forces.shape != (len(durations), len(parameters)) or
                not np.all(np.isfinite(forces)) or np.any(forces < 0)):
            raise ValueError("phase_forces must be finite nonnegative (phases, muscles) data.")
        tau = np.asarray([p.tau_fat for p in parameters])
        doses = forces * (-tau * np.expm1(-durations[:, None] / tau))
        return cls(parameters, durations, doses)

    def _advance(self, initial, cycles):
        if cycles == 0:
            return initial.copy()
        exponent = self.period / self.tau_fat
        decay = np.exp(-cycles * exponent)
        # exprel-like ratio avoids subtracting nearly equal exponentials.
        ratio = -np.expm1(-cycles * exponent) / -np.expm1(-exponent)
        return (self.rest + decay[:, None] * (initial - self.rest)
                + self.alpha * (self.cycle_force_integral * ratio)[:, None])

    def project(self, initial_slow_states, cycles):
        """Return the conditional algebraic projection, including domain failure.

        At a fixed phase, each slow component is monotone with cycle number
        under exactly repeated force. Thus first/last phase endpoints bound
        all intervening cycle endpoints. This does not check between phases,
        nor force/PW attainability. Invalid slow states remain explicitly
        reported as invalid; they are never treated as a realizable rollout.
        """
        if isinstance(cycles, (bool, np.bool_)) or not isinstance(cycles, Integral) or cycles < 0:
            raise ValueError("cycles must be a nonnegative integer.")
        initial = np.asarray(initial_slow_states, float)
        if (initial.shape != self.rest.shape or not np.all(np.isfinite(initial)) or np.any(initial <= 0)):
            raise ValueError("initial_slow_states must be finite positive (muscles, 3) data.")
        final = self._advance(initial, int(cycles))
        minimum = initial.copy()
        if cycles:
            for start in (initial.copy(), self._advance(initial, int(cycles) - 1)):
                current = start
                minimum = np.minimum(minimum, current)
                for decay, dose in zip(self.phase_decay, self.phase_force_integrals):
                    current = (self.rest + decay[:, None] * (current - self.rest)
                               + self.alpha * dose[:, None])
                    minimum = np.minimum(minimum, current)
        valid = bool(np.all(np.isfinite(final)) and np.all(np.isfinite(minimum)) and np.all(minimum > 0))
        return FixedForceCycleProjection(final, int(cycles), valid, minimum)
