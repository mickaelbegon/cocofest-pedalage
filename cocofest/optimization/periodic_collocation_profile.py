"""Periodic, piecewise polynomial policies retaining the source collocation mesh.

Coefficients are in ascending powers of the local normalized interval time.
Values, time derivatives and exponentially weighted integrals are linear in
these fixed-size coefficients.  Phase derivatives are one-sided at mesh knots:
control switches need not give a C1 force trajectory.  No smoothing, endpoint
projection or clipping is performed; discontinuities must be audited by the
caller before interpreting the repeated source cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import hyp1f1

from cocofest.optimization.periodic_force_profile import PeriodicForcePositivityCertificate


def _unit_interval_roots(coefficients):
    """Isolate real roots recursively between derivative extrema.

    A companion-matrix solver can lose a small root when interpolation leaves
    an almost-zero highest coefficient (and hence another root near infinity).
    Bracketing on [0,1] avoids that conditioning problem without discarding or
    modifying coefficients. Repeated roots at critical points are retained.
    """
    coefficients = np.trim_zeros(np.asarray(coefficients, dtype=float), "b")
    if coefficients.size < 2:
        return np.empty(0)
    if coefficients.size == 2:
        root = -coefficients[0] / coefficients[1]
        return np.array([root]) if 0.0 < root < 1.0 else np.empty(0)
    critical = _unit_interval_roots(np.polynomial.polynomial.polyder(coefficients))
    boundaries = np.r_[0., critical, 1.]
    values = np.polynomial.polynomial.polyval(boundaries, coefficients)
    roundoff = 32. * np.finfo(float).eps * np.sum(np.abs(coefficients))
    roots = list(critical[np.abs(values[1:-1]) <= roundoff])
    for left, right, f_left, f_right in zip(boundaries[:-1], boundaries[1:], values[:-1], values[1:]):
        if np.signbit(f_left) != np.signbit(f_right) and f_left != 0.0 and f_right != 0.0:
            roots.append(brentq(lambda x: np.polynomial.polynomial.polyval(x, coefficients),
                left, right, xtol=5e-15, rtol=1e-14))
    return np.unique(roots)


@dataclass(frozen=True)
class PeriodicCollocationProfile:
    """A fixed mesh of polynomials shaped ``(signals, intervals, degree+1)``."""

    period: float
    coefficients: np.ndarray

    def __post_init__(self):
        period = float(self.period)
        coefficients = np.asarray(self.coefficients, dtype=float)
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError("period must be finite and strictly positive.")
        if coefficients.ndim != 3 or min(coefficients.shape) < 1:
            raise ValueError("coefficients must have shape (signals, intervals, degree+1).")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients must be finite.")
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "coefficients", coefficients.copy())

    @classmethod
    def from_samples(cls, *, period, nodes, values):
        """Interpolate the shooting node and actual declared collocation stages."""
        nodes = np.asarray(nodes, dtype=float)
        values = np.asarray(values, dtype=float)
        if (nodes.ndim != 1 or not np.all(np.isfinite(nodes))
                or np.unique(nodes).size != nodes.size or nodes.size < 2
                or np.any(nodes < 0.0) or np.any(nodes > 1.0)):
            raise ValueError("nodes must be distinct finite normalized collocation times.")
        if values.ndim != 3 or values.shape[-1] != nodes.size:
            raise ValueError("values must have shape (signals, intervals, nodes).")
        design = np.polynomial.polynomial.polyvander(nodes, nodes.size - 1)
        coefficients = np.linalg.solve(design, values.reshape(-1, nodes.size).T).T
        return cls(period=period, coefficients=coefficients.reshape(values.shape))

    @property
    def signal_count(self):
        return self.coefficients.shape[0]

    @property
    def interval_count(self):
        return self.coefficients.shape[1]

    @property
    def degree(self):
        return self.coefficients.shape[2] - 1

    @property
    def interval_duration(self):
        return self.period / self.interval_count

    def _local_times(self, time):
        time = np.atleast_1d(np.asarray(time, dtype=float))
        if time.ndim != 1 or not np.all(np.isfinite(time)):
            raise ValueError("time must contain finite scalar/vector values.")
        coordinates = np.mod(time, self.period) / self.interval_duration
        # Snap only roundoff in the time coordinate, never a state value.
        nearest = np.rint(coordinates)
        coordinates = np.where(np.abs(coordinates - nearest) < 1e-12, nearest, coordinates)
        interval = np.floor(coordinates).astype(int)
        return interval % self.interval_count, coordinates - interval

    def evaluate(self, time):
        interval, local = self._local_times(time)
        powers = local[:, None] ** np.arange(self.degree + 1)
        return np.einsum("stk,tk->st", self.coefficients[:, interval, :], powers)

    def derivative(self, time):
        interval, local = self._local_times(time)
        powers = local[:, None] ** np.arange(self.degree)
        return np.einsum(
            "stk,tk->st", self.coefficients[:, interval, 1:] * np.arange(1, self.degree + 1), powers
        ) / self.interval_duration

    def piecewise_midpoints(self, interval_count):
        if isinstance(interval_count, bool) or int(interval_count) != interval_count or interval_count < 1:
            raise ValueError("interval_count must be a positive integer.")
        duration = self.period / int(interval_count)
        return self.evaluate((np.arange(interval_count) + 0.5) * duration), np.full(interval_count, duration)

    def seam_audit(self):
        """Record C0 jumps and one-sided derivative jumps, including the wrap."""
        left_value = np.sum(self.coefficients, axis=-1)
        right_value = np.roll(self.coefficients[:, :, 0], -1, axis=1)
        left_derivative = np.sum(
            self.coefficients[:, :, 1:] * np.arange(1, self.degree + 1), axis=-1
        ) / self.interval_duration
        right_derivative = (
            np.roll(self.coefficients[:, :, 1], -1, axis=1) / self.interval_duration
            if self.degree else np.zeros_like(left_derivative)
        )
        return {
            "value_jump": (right_value - left_value).tolist(),
            "periodic_value_jump": (right_value - left_value)[:, -1].tolist(),
            "maximum_internal_absolute_value_jump": (
                np.max(np.abs(right_value - left_value)[:, :-1], axis=1)
                if self.interval_count > 1 else np.zeros(self.signal_count)
            ).tolist(),
            "maximum_absolute_value_jump": np.max(np.abs(right_value - left_value), axis=1).tolist(),
            "time_derivative_jump": (right_derivative - left_derivative).tolist(),
            "periodic_time_derivative_jump": (right_derivative - left_derivative)[:, -1].tolist(),
            "maximum_absolute_time_derivative_jump": np.max(
                np.abs(right_derivative - left_derivative), axis=1
            ).tolist(),
            "boundary_evaluation": "right_sided; no endpoint projection",
            "phase_differentiability": "piecewise; derivative jumps retained at control switches",
            "coefficient_differentiability": "linear at all fixed rollout phases",
        }

    def force_positivity_certificate(self, *, tolerance=1e-10):
        """Find minima over every closed polynomial interval via derivative roots."""
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        minima = np.full(self.signal_count, np.inf)
        minimum_times = np.zeros(self.signal_count)
        counts = np.zeros(self.signal_count, dtype=int)
        for signal, pieces in enumerate(self.coefficients):
            for interval, coefficients in enumerate(pieces):
                derivative = np.polynomial.polynomial.polyder(coefficients)
                real = _unit_interval_roots(derivative)
                candidates = np.concatenate(([0.0, 1.0], real))
                values = np.polynomial.polynomial.polyval(candidates, coefficients)
                index = np.argmin(values)
                if values[index] < minima[signal]:
                    minima[signal] = values[index]
                    minimum_times[signal] = (interval + candidates[index]) * self.interval_duration
                counts[signal] += real.size
        return PeriodicForcePositivityCertificate(
            minimum_force=minima, minimum_time=minimum_times,
            certified_nonnegative=minima >= -tolerance,
            stationary_point_count=counts, tolerance=tolerance,
        )

    def exponentially_weighted_force_integral(self, *, start_time, duration, time_constant):
        r"""Analytical polynomial convolution, split exactly at mesh boundaries.

        For a shifted monomial v**k, the stable moment is
        ``L**(k+1)/(k+1) * hyp1f1(1,k+2,-h*L/tau)``.  This avoids the severe
        cancellation of the integration-by-parts recurrence when h/tau is
        small.  Each segment and their composition are linear in coefficients.
        """
        start_time, duration, time_constant = map(float, (start_time, duration, time_constant))
        if not all(math.isfinite(v) for v in (start_time, duration, time_constant)):
            raise ValueError("start_time, duration, and time_constant must be finite.")
        if duration < 0.0 or time_constant <= 0.0:
            raise ValueError("duration must be non-negative and time_constant positive.")
        result = np.zeros(self.signal_count)
        interval, local = self._local_times(start_time)
        interval, local = int(interval[0]), float(local[0])
        remaining = duration
        powers = np.arange(self.degree + 1)
        while remaining > 0.0:
            length_s = min(remaining, (1.0 - local) * self.interval_duration)
            length = length_s / self.interval_duration
            shifted = np.zeros((self.signal_count, self.degree + 1))
            for power in range(self.degree + 1):
                for source_power in range(power, self.degree + 1):
                    shifted[:, power] += (self.coefficients[:, interval, source_power]
                        * math.comb(source_power, power) * local ** (source_power - power))
            moments = (self.interval_duration * length ** (powers + 1) / (powers + 1)
                * hyp1f1(1.0, powers + 2.0, -length_s / time_constant))
            result = math.exp(-length_s / time_constant) * result + shifted @ moments
            remaining = max(0.0, remaining - length_s)
            interval, local = (interval + 1) % self.interval_count, 0.0
        return result
