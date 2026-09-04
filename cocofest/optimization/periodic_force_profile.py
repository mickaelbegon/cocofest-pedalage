"""Auditable periodic Fourier representation of force profiles from a RHO.

The representation supplies both force and its analytical time derivative.
This makes the interpolation assumption required by the recruitment inversion
explicit. It does not clip negative reconstruction overshoot.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PeriodicForcePositivityCertificate:
    """Numerical extrema audit for non-negative periodic muscle forces.

    ``certified_nonnegative`` is true only when the unit-circle stationary
    points of each trigonometric polynomial were resolved and their force
    values are non-negative within ``tolerance``.  It is a property of the
    continuous Fourier interpolant, not merely of an evaluation grid.
    """

    minimum_force: np.ndarray
    minimum_time: np.ndarray
    certified_nonnegative: np.ndarray
    stationary_point_count: np.ndarray
    tolerance: float


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return value


@dataclass(frozen=True)
class PeriodicFourierForceProfile:
    """Least-squares Fourier fit shared by one or more muscle-force signals."""

    period: float
    mean: np.ndarray
    cosine: np.ndarray
    sine: np.ndarray

    def __post_init__(self) -> None:
        period = _positive_finite(self.period, "period")
        mean = np.asarray(self.mean, dtype=float)
        cosine = np.asarray(self.cosine, dtype=float)
        sine = np.asarray(self.sine, dtype=float)
        if mean.ndim != 1 or cosine.ndim != 2 or sine.shape != cosine.shape:
            raise ValueError("mean must be 1-D and cosine/sine must be equal 2-D arrays.")
        if cosine.shape[0] != mean.size or cosine.shape[1] < 1:
            raise ValueError("Fourier coefficients must contain at least one harmonic per signal.")
        if not all(np.all(np.isfinite(values)) for values in (mean, cosine, sine)):
            raise ValueError("Fourier coefficients must be finite.")
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(self, "cosine", cosine.copy())
        object.__setattr__(self, "sine", sine.copy())

    @property
    def signal_count(self) -> int:
        return int(self.mean.size)

    @property
    def harmonic_count(self) -> int:
        return int(self.cosine.shape[1])

    def evaluate(self, time) -> np.ndarray:
        """Evaluate force at scalar or vector times; output shape is signals × times."""

        time = np.atleast_1d(np.asarray(time, dtype=float))
        if time.ndim != 1 or not np.all(np.isfinite(time)):
            raise ValueError("time must contain finite scalar/vector values.")
        phase = 2.0 * np.pi * np.mod(time, self.period) / self.period
        harmonics = np.arange(1, self.harmonic_count + 1, dtype=float)[:, None]
        angles = harmonics * phase[None, :]
        return (
            self.mean[:, None]
            + self.cosine @ np.cos(angles)
            + self.sine @ np.sin(angles)
        )

    def derivative(self, time) -> np.ndarray:
        """Evaluate the analytical derivative dF/dt at scalar or vector times."""

        time = np.atleast_1d(np.asarray(time, dtype=float))
        if time.ndim != 1 or not np.all(np.isfinite(time)):
            raise ValueError("time must contain finite scalar/vector values.")
        phase = 2.0 * np.pi * np.mod(time, self.period) / self.period
        harmonic_numbers = np.arange(1, self.harmonic_count + 1, dtype=float)
        angular_frequencies = 2.0 * np.pi * harmonic_numbers / self.period
        angles = harmonic_numbers[:, None] * phase[None, :]
        return (
            -(self.cosine * angular_frequencies[None, :]) @ np.sin(angles)
            + (self.sine * angular_frequencies[None, :]) @ np.cos(angles)
        )

    def piecewise_midpoints(self, interval_count: int) -> tuple[np.ndarray, np.ndarray]:
        """Return midpoint forces and equal durations for a piecewise-constant rollout."""

        if isinstance(interval_count, bool) or int(interval_count) != interval_count or interval_count < 1:
            raise ValueError("interval_count must be a positive integer.")
        duration = self.period / int(interval_count)
        time = (np.arange(int(interval_count), dtype=float) + 0.5) * duration
        return self.evaluate(time), np.full(int(interval_count), duration)

    def exponentially_weighted_force_integral(
        self,
        *,
        start_time: float,
        duration: float,
        time_constant: float,
    ) -> np.ndarray:
        r"""Return ``\int_0^dt exp(-(dt-s)/tau) F(start+s) ds`` exactly.

        The integral is evaluated analytically for every Fourier harmonic and
        is stable for short durations through ``expm1``.  It is the forcing
        term required by the exact affine Ding slow-state map under the
        *continuous* force interpolant.
        """
        start_time = float(start_time)
        duration = float(duration)
        time_constant = float(time_constant)
        if not all(math.isfinite(value) for value in (start_time, duration, time_constant)):
            raise ValueError("start_time, duration, and time_constant must be finite.")
        if duration < 0.0 or time_constant <= 0.0:
            raise ValueError("duration must be non-negative and time_constant strictly positive.")
        if duration == 0.0:
            return np.zeros(self.signal_count)

        decay_integral = -time_constant * math.expm1(-duration / time_constant)
        harmonic_numbers = np.arange(1, self.harmonic_count + 1, dtype=float)
        angular_frequencies = 2.0 * np.pi * harmonic_numbers / self.period
        denominator = 1.0 / time_constant + 1j * angular_frequencies
        end_phase = np.mod(start_time + duration, self.period)
        numerator = -np.expm1(-denominator * duration)
        complex_integrals = np.exp(1j * angular_frequencies * end_phase) * numerator / denominator
        return (
            self.mean * decay_integral
            + self.cosine @ complex_integrals.real
            + self.sine @ complex_integrals.imag
        )

    def force_positivity_certificate(self, *, tolerance: float = 1e-10) -> PeriodicForcePositivityCertificate:
        """Find each signal's continuous-period minimum from derivative roots.

        For a trigonometric polynomial, all extrema occur at the periodic
        boundary or at roots of its derivative.  Substitution ``z=exp(i phi)``
        transforms that derivative into a finite ordinary polynomial.  The
        auxiliary roots off the unit circle have no real phase and are
        discarded; the unit-circle roots are evaluated directly instead of
        relying on a dense grid that could miss a narrow negative interval.
        """
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        minimum_force = np.empty(self.signal_count)
        minimum_time = np.empty(self.signal_count)
        certified = np.zeros(self.signal_count, dtype=bool)
        stationary_count = np.zeros(self.signal_count, dtype=int)
        root_tolerance = 1e-7

        for signal_index in range(self.signal_count):
            coefficients = np.zeros(2 * self.harmonic_count + 1, dtype=complex)
            for harmonic in range(1, self.harmonic_count + 1):
                cosine = self.cosine[signal_index, harmonic - 1]
                sine = self.sine[signal_index, harmonic - 1]
                coefficients[self.harmonic_count + harmonic] = harmonic * (sine + 1j * cosine) / 2.0
                coefficients[self.harmonic_count - harmonic] = harmonic * (sine - 1j * cosine) / 2.0

            while coefficients.size > 1 and coefficients[-1] == 0.0:
                coefficients = coefficients[:-1]
            while coefficients.size > 1 and coefficients[0] == 0.0:
                coefficients = coefficients[1:]
            if np.all(coefficients == 0.0):
                candidate_phase = np.array([0.0])
                roots_complete = True
            else:
                roots = np.polynomial.polynomial.polyroots(coefficients)
                unit_roots = roots[np.abs(np.abs(roots) - 1.0) <= root_tolerance]
                # Roots off the unit circle are valid roots of the auxiliary
                # polynomial but do not correspond to real phases. Every real
                # stationary phase is represented by a unit-circle root.
                roots_complete = True
                candidate_phase = np.concatenate(([0.0], np.mod(np.angle(unit_roots), 2.0 * np.pi)))

            candidate_time = candidate_phase * self.period / (2.0 * np.pi)
            values = self.evaluate(candidate_time)[signal_index]
            minimum_index = int(np.argmin(values))
            minimum_force[signal_index] = values[minimum_index]
            minimum_time[signal_index] = candidate_time[minimum_index]
            stationary_count[signal_index] = candidate_time.size - 1
            certified[signal_index] = roots_complete and minimum_force[signal_index] >= -tolerance

        return PeriodicForcePositivityCertificate(
            minimum_force=minimum_force,
            minimum_time=minimum_time,
            certified_nonnegative=certified,
            stationary_point_count=stationary_count,
            tolerance=tolerance,
        )


def fit_periodic_fourier_force_profile(
    samples,
    *,
    period: float,
    harmonic_count: int,
    endpoint_included: bool = False,
) -> PeriodicFourierForceProfile:
    """Fit uniformly sampled periodic forces without changing their units.

    Samples are shaped ``(signals, nodes)``; a one-dimensional input represents
    one signal. If the periodic endpoint is present, it is checked against the
    first node and removed before fitting.
    """

    period = _positive_finite(period, "period")
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2 or samples.shape[0] < 1 or samples.shape[1] < 3:
        raise ValueError("samples must have shape (signals, nodes) with at least three nodes.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must be finite.")
    if endpoint_included:
        if samples.shape[1] < 4:
            raise ValueError("endpoint-included samples require at least four nodes.")
        scale = np.maximum(1.0, np.max(np.abs(samples), axis=1))
        if np.any(np.abs(samples[:, -1] - samples[:, 0]) > 1e-8 * scale):
            raise ValueError("the declared periodic endpoint does not match the first sample.")
        samples = samples[:, :-1]
    if isinstance(harmonic_count, bool) or int(harmonic_count) != harmonic_count or harmonic_count < 1:
        raise ValueError("harmonic_count must be a positive integer.")
    harmonic_count = int(harmonic_count)
    if 2 * harmonic_count + 1 > samples.shape[1]:
        raise ValueError("the requested Fourier order is underdetermined by the samples.")

    phase = 2.0 * np.pi * np.arange(samples.shape[1], dtype=float) / samples.shape[1]
    columns = [np.ones_like(phase)]
    columns.extend(np.cos(harmonic * phase) for harmonic in range(1, harmonic_count + 1))
    columns.extend(np.sin(harmonic * phase) for harmonic in range(1, harmonic_count + 1))
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, samples.T, rcond=None)
    return PeriodicFourierForceProfile(
        period=period,
        mean=coefficients[0],
        cosine=coefficients[1 : harmonic_count + 1].T,
        sine=coefficients[harmonic_count + 1 :].T,
    )


def periodic_force_profile_audit(
    profile: PeriodicFourierForceProfile,
    samples,
    *,
    endpoint_included: bool = False,
) -> dict[str, float | int | bool]:
    """Report reconstruction error and negative Fourier overshoot explicitly."""

    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[None, :]
    if endpoint_included:
        samples = samples[:, :-1]
    if samples.ndim != 2 or samples.shape[0] != profile.signal_count:
        raise ValueError("samples are incompatible with the fitted profile.")
    times = np.arange(samples.shape[1], dtype=float) * profile.period / samples.shape[1]
    reconstructed = profile.evaluate(times)
    if reconstructed.shape != samples.shape:
        raise ValueError("samples are incompatible with the fitted profile.")
    error = reconstructed - samples
    dense = profile.evaluate(np.linspace(0.0, profile.period, 10 * samples.shape[1], endpoint=False))
    return {
        "signal_count": profile.signal_count,
        "sample_count": int(samples.shape[1]),
        "harmonic_count": profile.harmonic_count,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "minimum_reconstructed_force": float(np.min(dense)),
        "negative_reconstruction": bool(np.any(dense < 0.0)),
    }
