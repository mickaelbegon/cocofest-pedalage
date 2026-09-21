"""Auditable actual-state transfer for a one-cycle, fixed-clock RHO restart.

This module prepares a solver request; it does not replay an archived control
sequence or certify convergence. A caller supplies a real OCP builder/solver
through ``solve``. The builder must consume all returned bounds and clock data.
The existing whole-cycle ``advance_window`` hook is deliberately not used:
an interior stimulation tick is a valid restart, but not a whole-cycle advance.

Only full five-state Ding, dynamic reduced mechanics are supported. A compact
Ding OCP needs its own exact state reconstruction adapter before it can use this
interface. Arrays use named physical state rows, independent of Bioptim order.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable, Mapping

import numpy as np


def config_fingerprint(config: Mapping) -> str:
    """Fingerprint an explicit complete dynamics/configuration declaration."""
    return sha256(json.dumps(dict(config), sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class FallbackContract:
    muscles: tuple[str, ...]
    frequency_hz: float
    intervals_per_cycle: int
    phase_origin_rad: float
    time_origin_s: float
    cycle_displacement_rad: float
    dynamics_fingerprint: str
    max_delta_pw_s: float | None = None
    mechanics: str = "reduced"
    formulation: str = "dynamic"
    ding_states: str = "full_five_state"
    calcium_forcing: str = "exact_exponential_periodic_node"

    @property
    def state_names(self):
        return ("theta", "omega") + tuple(
            f"{component}_{muscle}" for muscle in self.muscles
            for component in ("Cn", "F", "A", "Tau1", "Km")
        )

    @property
    def duration_s(self):
        return self.intervals_per_cycle / self.frequency_hz

    def validate(self):
        if not self.muscles or len(set(self.muscles)) != len(self.muscles):
            raise ValueError("Unique nonempty muscle names are required")
        if any(not isinstance(m, str) or not m for m in self.muscles):
            raise ValueError("Unique nonempty muscle names are required")
        if (self.mechanics, self.formulation, self.ding_states, self.calcium_forcing) != (
            "reduced", "dynamic", "full_five_state", "exact_exponential_periodic_node"
        ):
            raise ValueError("Fallback requires dynamic reduced mechanics and full five-state periodic-node Ding")
        if not np.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive and finite")
        if (isinstance(self.intervals_per_cycle, bool)
                or not isinstance(self.intervals_per_cycle, (int, np.integer))
                or self.intervals_per_cycle < 2):
            raise ValueError("intervals_per_cycle must be an integer >= 2")
        if not np.isfinite([self.phase_origin_rad, self.time_origin_s, self.cycle_displacement_rad]).all():
            raise ValueError("Clock and global phase origin must be finite")
        if not np.isclose(abs(self.cycle_displacement_rad), 2 * np.pi, rtol=0, atol=1e-12):
            raise ValueError("A one-cycle fallback must represent one signed revolution")
        if (not isinstance(self.dynamics_fingerprint, str) or len(self.dynamics_fingerprint) != 64
                or any(c not in "0123456789abcdef" for c in self.dynamics_fingerprint)):
            raise ValueError("An explicit dynamics configuration SHA256 fingerprint is required")
        if self.max_delta_pw_s is not None and (
            not np.isfinite(self.max_delta_pw_s) or self.max_delta_pw_s <= 0
        ):
            raise ValueError("max_delta_pw_s must be positive and finite")


@dataclass(frozen=True)
class TickMeasurement:
    tick_index: int
    time_s: float
    states: Mapping[str, float]
    last_pw_s: Mapping[str, float]
    contract: FallbackContract


@dataclass(frozen=True)
class OneCycleCandidate:
    """A window already positioned on the global reference, before transfer.

    State bounds have START / path / END columns. Physical state bounds are
    supplied independently: old START equalities are not physical limits.
    PW bounds are stage-specific, including any known inactive-muscle mask.
    The dynamics fingerprint must cover profile, muscle parameters, external
    load, relationship flags and stimulation options (not just a model name).
    """

    contract: FallbackContract
    start_tick: int
    state_names: tuple[str, ...]
    state_times_s: np.ndarray
    states: np.ndarray
    state_lower: np.ndarray
    state_upper: np.ndarray
    physical_state_lower: np.ndarray
    physical_state_upper: np.ndarray
    pw_s: np.ndarray
    pw_lower_s: np.ndarray
    pw_upper_s: np.ndarray
    physical_pw_lower_s: np.ndarray
    physical_pw_upper_s: np.ndarray
    reference_start_theta: float
    reference_end_theta: float
    source_provenance: Mapping


@dataclass(frozen=True)
class PreparedFallback:
    candidate: OneCycleCandidate
    measurement: TickMeasurement
    # When enabled, these are shooting-node carrier guesses and bounds and
    # interval increment guesses. A collocation builder interpolates carriers
    # linearly onto its internal nodes; the auxiliary derivative is constant.
    carrier_s: np.ndarray | None
    carrier_lower_s: np.ndarray | None
    carrier_upper_s: np.ndarray | None
    increments_s: np.ndarray | None
    provenance: Mapping


def _array(value, shape, name):
    result = np.asarray(value, dtype=float).copy()
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return result


def prepare_tick_fallback(candidate: OneCycleCandidate, measurement: TickMeasurement) -> PreparedFallback:
    """Copy a candidate and impose the exact observed state and previous PW seam.

    No angle wrapping, state clipping, cycle-index increment, fatigue reset or
    movement of stimulation times occurs. Preparation never mutates the input.
    """
    contract = candidate.contract
    contract.validate()
    measurement.contract.validate()
    if contract != measurement.contract:
        raise ValueError("Measurement and candidate source/config/frequency/delta-PW contracts differ")
    tick = measurement.tick_index
    if isinstance(tick, bool) or not isinstance(tick, (int, np.integer)) or tick < 0:
        raise ValueError("tick_index must be a nonnegative integer")
    expected_time = contract.time_origin_s + tick / contract.frequency_hz
    if not np.isfinite(measurement.time_s) or abs(measurement.time_s - expected_time) > 1e-10:
        raise ValueError("Fallback starts only on a fixed stimulation tick")
    if candidate.start_tick != tick:
        raise ValueError("Candidate window must be positioned on the measured tick")
    names = candidate.state_names
    if len(names) != len(set(names)) or set(names) != set(contract.state_names):
        raise ValueError("Candidate must contain exactly theta, omega and all five Ding states")
    if set(measurement.states) != set(names):
        raise ValueError("Measurement must contain exactly the complete named physical state")
    if set(measurement.last_pw_s) != set(contract.muscles):
        raise ValueError("Last applied PW is required for every muscle")
    if not candidate.source_provenance:
        raise ValueError("Explicit candidate source provenance is required")
    json.dumps(dict(candidate.source_provenance), allow_nan=False)

    rows, muscles, intervals = len(names), len(contract.muscles), contract.intervals_per_cycle
    times = np.asarray(candidate.state_times_s, dtype=float)
    if (times.ndim != 1 or len(times) < intervals + 1 or not np.isfinite(times).all()
            or times[0] != 0 or np.any(np.diff(times) <= 0)
            or not np.isclose(times[-1], contract.duration_s, rtol=0, atol=1e-12)):
        raise ValueError("State times must cover exactly one cycle in increasing local time")
    states = _array(candidate.states, (rows, len(times)), "states")
    lower = _array(candidate.state_lower, (rows, 3), "state_lower")
    upper = _array(candidate.state_upper, (rows, 3), "state_upper")
    physical_lower = _array(candidate.physical_state_lower, (rows,), "physical_state_lower")
    physical_upper = _array(candidate.physical_state_upper, (rows,), "physical_state_upper")
    pw = _array(candidate.pw_s, (muscles, intervals), "pw_s")
    pw_lower = _array(candidate.pw_lower_s, pw.shape, "pw_lower_s")
    pw_upper = _array(candidate.pw_upper_s, pw.shape, "pw_upper_s")
    physical_pw_lower = _array(candidate.physical_pw_lower_s, (muscles,), "physical_pw_lower_s")
    physical_pw_upper = _array(candidate.physical_pw_upper_s, (muscles,), "physical_pw_upper_s")
    measured = _array([measurement.states[name] for name in names], (rows,), "measured state")
    last_pw = _array([measurement.last_pw_s[name] for name in contract.muscles], (muscles,), "last PW")
    if (np.any(lower > upper) or np.any(physical_lower > physical_upper)
            or np.any(pw_lower > pw_upper) or np.any(physical_pw_lower < 0)
            or np.any(physical_pw_lower > physical_pw_upper)):
        raise ValueError("Physical/transcribed bounds are inconsistent")
    if (np.any(lower[:, 1:] < physical_lower[:, None])
            or np.any(upper[:, 1:] > physical_upper[:, None])
            or np.any(pw_lower < physical_pw_lower[:, None])
            or np.any(pw_upper > physical_pw_upper[:, None])):
        raise ValueError("Transcribed path/control bounds must respect physical limits")
    if np.any(measured < physical_lower) or np.any(measured > physical_upper):
        raise ValueError("Measured state lies outside physical bounds; it must not be clipped")
    for muscle in contract.muscles:
        if any(measurement.states[f"{key}_{muscle}"] < 0 for key in ("Cn", "F")) or any(
            measurement.states[f"{key}_{muscle}"] <= 0 for key in ("A", "Tau1", "Km")
        ):
            raise ValueError("Measured Ding state is outside the physiological domain")
    # The phase target belongs to the original clock, even when theta has drifted.
    phase_start = contract.phase_origin_rad + tick / intervals * contract.cycle_displacement_rad
    if not np.allclose([candidate.reference_start_theta, candidate.reference_end_theta],
                       [phase_start, phase_start + contract.cycle_displacement_rad], rtol=0, atol=1e-10):
        raise ValueError("Candidate must preserve the original unwrapped global phase reference")
    theta_row = names.index("theta")
    if not lower[theta_row, 2] <= candidate.reference_end_theta <= upper[theta_row, 2]:
        raise ValueError("Terminal angle bounds conflict with the global phase target")
    if np.any(last_pw < physical_pw_lower) or np.any(last_pw > physical_pw_upper):
        raise ValueError("Last applied PW lies outside the physical PW envelope")

    states[:, 0] = measured
    lower[:, 0] = measured
    upper[:, 0] = measured
    carrier = carrier_lower = carrier_upper = increments = None
    step = contract.max_delta_pw_s
    if step is not None:
        pw_lower[:, 0] = np.maximum(pw_lower[:, 0], last_pw - step)
        pw_upper[:, 0] = np.minimum(pw_upper[:, 0], last_pw + step)
        if np.any(pw_lower[:, 0] > pw_upper[:, 0]):
            raise ValueError("Last applied PW has no feasible delta-PW seam with first-stage bounds")
        # Project the warm start only; this does not certify muscle dynamics.
        previous = last_pw
        for stage in range(intervals):
            lo = np.maximum(pw_lower[:, stage], previous - step)
            hi = np.minimum(pw_upper[:, stage], previous + step)
            if np.any(lo > hi):
                raise ValueError("Candidate PW warm start cannot satisfy the successive delta-PW bounds")
            pw[:, stage] = np.clip(pw[:, stage], lo, hi)
            previous = pw[:, stage]
        carrier = np.column_stack((pw, pw[:, -1]))
        carrier_lower = np.column_stack((pw_lower, pw_lower[:, -1]))
        carrier_upper = np.column_stack((pw_upper, pw_upper[:, -1]))
        increments = np.diff(carrier, axis=1)
    else:
        pw = np.clip(pw, pw_lower, pw_upper)

    from dataclasses import replace
    updated = replace(candidate, states=states, state_times_s=times.copy(), state_lower=lower,
                      state_upper=upper, physical_state_lower=physical_lower,
                      physical_state_upper=physical_upper, pw_s=pw, pw_lower_s=pw_lower,
                      pw_upper_s=pw_upper, physical_pw_lower_s=physical_pw_lower,
                      physical_pw_upper_s=physical_pw_upper,
                      source_provenance=dict(candidate.source_provenance))
    provenance = {
        "method": "actual_state_fixed_tick_one_cycle_rho_transfer_v1",
        "scope": "prepared_solver_request_not_convergence_certificate",
        "start_tick": int(tick), "start_time_s": float(measurement.time_s),
        "frequency_hz": contract.frequency_hz, "duration_s": contract.duration_s,
        "global_phase_origin_rad": contract.phase_origin_rad,
        "dynamics_fingerprint": contract.dynamics_fingerprint,
        "source": dict(candidate.source_provenance),
        "max_delta_pw_s": step, "last_applied_pw_s": last_pw.tolist(),
        "physical_state_names": list(names), "measured_initial_state": measured.tolist(),
    }
    return PreparedFallback(updated, measurement, carrier, carrier_lower, carrier_upper, increments, provenance)


def solve_tick_fallback(candidate: OneCycleCandidate, measurement: TickMeasurement,
                        *, solve: Callable[[PreparedFallback], object]):
    """Invoke an injected actual OCP solve after all transfer checks succeed.

    The callback owns backend construction, execution and convergence/defect
    certification. Its return is passed through without calling it successful.
    Backend exceptions propagate; no archived solution substitutes for failure.
    """
    prepared = prepare_tick_fallback(candidate, measurement)
    return solve(prepared)
