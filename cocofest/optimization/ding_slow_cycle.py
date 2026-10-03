"""Exact slow-state cycle maps conditional on archived polynomial force profiles.

This module does not predict force or feasibility. It integrates the forcing of
the existing Ding slow equations, using the force polynomial represented by a
collocation interval. The integration is exact for that polynomial (up to
floating-point error), not for an unknown continuous force between saved nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters


def _exponential_moments(degree: int, ratio: float) -> np.ndarray:
    """Integral of exp(-ratio*(1-u))*u**k on [0, 1], k=0..degree."""
    if ratio < 0 or not math.isfinite(ratio):
        raise ValueError("duration/tau_fat must be finite and non-negative")
    if ratio <= 4.0:
        # beta-integral series: term_n = (-ratio)^n k! / (k+n+1)!.
        # It avoids subtracting nearly equal numbers at clinical dt/tau << 1.
        result = []
        for power in range(degree + 1):
            term = 1.0 / (power + 1)
            terms = [term]
            for index in range(1, 256):
                term *= -ratio / (power + index + 1)
                terms.append(term)
                if abs(term) < 1e-17 * abs(math.fsum(terms)):
                    break
            result.append(math.fsum(terms))
        return np.asarray(result)
    moments = [-math.expm1(-ratio) / ratio]
    for power in range(1, degree + 1):
        moments.append((1.0 - power * moments[-1]) / ratio)
    return np.asarray(moments)


@lru_cache(maxsize=128)
def _weights(nodes: tuple[float, ...], ratio: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(nodes, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all():
        raise ValueError("nodes must contain at least two finite entries")
    if grid[0] != 0 or grid[-1] != 1 or np.any(np.diff(grid) <= 0):
        raise ValueError("nodes must increase strictly from 0 to 1")
    vandermonde = np.polynomial.polynomial.polyvander(grid, grid.size - 1)
    exponential = np.linalg.solve(vandermonde.T, _exponential_moments(grid.size - 1, ratio))
    ordinary = np.linalg.solve(vandermonde.T, 1.0 / np.arange(1, grid.size + 1))
    return exponential, ordinary


@dataclass(frozen=True)
class DingSlowCycleMap:
    """One muscle's affine flow under a known within-cycle force profile."""

    duration: float
    parameters: DingFatigueParameters
    weighted_force_integral: float
    force_integral: float

    def __post_init__(self):
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("cycle duration must be finite and strictly positive")
        if not math.isfinite(self.weighted_force_integral) or not math.isfinite(self.force_integral):
            raise ValueError("force integrals must be finite")

    @property
    def decay(self) -> float:
        return math.exp(-self.duration / self.parameters.tau_fat)

    def propagate(self, state: np.ndarray, repetitions: int = 1) -> np.ndarray:
        """Propagate a full slow state, preserving off-manifold initial offsets.

        Repetitions > 1 assume this same force profile remains realizable and
        unchanged; this assumption is not a forecast of a future RHO policy.
        """
        state = np.asarray(state, dtype=float)
        if state.shape != (3,) or not np.isfinite(state).all():
            raise ValueError("state must be a finite (A, Tau1, Km) vector")
        if isinstance(repetitions, bool) or not isinstance(repetitions, (int, np.integer)) or repetitions < 0:
            raise ValueError("repetitions must be a non-negative integer")
        if repetitions == 0:
            return state.copy()
        exponent = -self.duration / self.parameters.tau_fat
        total_decay = math.exp(repetitions * exponent)
        accumulated_gain = -math.expm1(repetitions * exponent) / -math.expm1(exponent)
        return (self.parameters.rest_state + total_decay * (state - self.parameters.rest_state)
                + self.parameters.alpha * self.weighted_force_integral * accumulated_gain)

    def propagate_with_cycle_average(self, state: np.ndarray) -> np.ndarray:
        """Ablation replacing force by its ordinary cycle average."""
        gain = -self.parameters.tau_fat * math.expm1(-self.duration / self.parameters.tau_fat)
        return (self.parameters.rest_state + self.decay * (np.asarray(state) - self.parameters.rest_state)
                + self.parameters.alpha * (self.force_integral / self.duration) * gain)


def slow_cycle_map_from_collocation(
    force_samples: np.ndarray,
    interval_durations: np.ndarray,
    normalized_nodes: np.ndarray,
    parameters: DingFatigueParameters,
) -> DingSlowCycleMap:
    """Compose exponential convolutions from each force collocation polynomial.

    ``force_samples`` has shape (intervals, nodes). The first node of each row
    is the shooting state; the remaining nodes are the collocation stages.
    Nonuniform interval durations are supported. Negative archived samples are
    not silently clipped: signed force is mathematically meaningful to this
    diagnostic, and callers should separately audit physiological admissibility.
    """
    samples = np.asarray(force_samples, dtype=float)
    durations = np.asarray(interval_durations, dtype=float)
    nodes = tuple(np.asarray(normalized_nodes, dtype=float))
    if samples.ndim != 2 or samples.shape != (durations.size, len(nodes)) or durations.ndim != 1:
        raise ValueError("force_samples must have shape (intervals, normalized_nodes)")
    if not samples.size or not np.isfinite(samples).all() or not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("forces must be finite and interval durations strictly positive")
    weighted = ordinary = 0.0
    for row, duration in zip(samples, durations, strict=True):
        ratio = float(duration / parameters.tau_fat)
        exponential_weights, ordinary_weights = _weights(nodes, ratio)
        weighted = math.exp(-ratio) * weighted + duration * float(exponential_weights @ row)
        ordinary += duration * float(ordinary_weights @ row)
    return DingSlowCycleMap(float(durations.sum()), parameters, weighted, ordinary)


def coupled_slow_offsets(state: np.ndarray, parameters: DingFatigueParameters) -> np.ndarray:
    """Two offsets whose exact flow is homogeneous exponential recovery.

    Tau1 and Km are determined by A only if these initial offsets are zero.
    Restarting from arbitrary measured/perturbed slow states must retain them.
    """
    if parameters.alpha_a == 0:
        raise ValueError("offset coordinates require nonzero alpha_a")
    delta = np.asarray(state, dtype=float) - parameters.rest_state
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise ValueError("state must be a finite (A, Tau1, Km) vector")
    return delta[1:] - parameters.alpha[1:] / parameters.alpha_a * delta[0]
