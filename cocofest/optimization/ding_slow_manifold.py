"""Exact slow-state manifold of the Ding 2007 fatigue model.

For the fatigue model used in this repository, ``A``, ``Tau1`` and ``Km``
share both their relaxation time and their force input.  A state started on
the manifold represented here stays on it under the continuous equations.
This is useful for offline *capacity-feasibility* studies: varying only ``A``
while keeping ``Tau1`` and ``Km`` fixed creates a synthetic, generally
unreachable state.

The module is numerical and intentionally independent of CasADi/Bioptim.  It
does not assert that a complete cycling state is reachable: ``Cn``, ``F`` and
the stimulation history must still be conditioned and recorded separately.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


def _finite(value: float, name: str, *, nonzero: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (nonzero and value == 0.):
        raise ValueError(f"{name} must be finite" + (" and non-zero" if nonzero else ""))
    return value


@dataclass(frozen=True)
class DingSlowFatigueManifold:
    """Map normalized capacity ``A/A_rest`` to compatible slow Ding states."""

    a_rest: float
    tau1_rest: float
    km_rest: float
    alpha_a: float
    alpha_tau1: float
    alpha_km: float

    def __post_init__(self) -> None:
        if _finite(self.a_rest, "a_rest") <= 0:
            raise ValueError("a_rest must be strictly positive")
        for name in ("tau1_rest", "km_rest"):
            _finite(getattr(self, name), name)
        for name in ("alpha_a", "alpha_tau1", "alpha_km"):
            _finite(getattr(self, name), name, nonzero=True)

    @classmethod
    def from_model(cls, model) -> "DingSlowFatigueManifold":
        """Extract the six immutable slow-state coefficients from a Ding model."""
        return cls(model.a_scale, model.tau1_rest, model.km_rest,
                   model.alpha_a, model.alpha_tau1, model.alpha_km)

    def states_from_capacity_ratio(self, ratio: float) -> tuple[float, float, float]:
        """Return compatible ``(A, Tau1, Km)`` for a normalized capacity."""
        ratio = _finite(ratio, "capacity ratio")
        q = (ratio * self.a_rest - self.a_rest) / self.alpha_a
        return (ratio * self.a_rest,
                self.tau1_rest + self.alpha_tau1 * q,
                self.km_rest + self.alpha_km * q)

    def capacity_ratio_from_states(self, a: float, tau1: float, km: float, *,
                                   tolerance: float = 1e-10) -> float:
        """Recover capacity only after checking the slow-manifold invariant."""
        tolerance = _finite(tolerance, "tolerance")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        a, tau1, km = (_finite(a, "A"), _finite(tau1, "Tau1"), _finite(km, "Km"))
        q_a = (a - self.a_rest) / self.alpha_a
        q_tau1 = (tau1 - self.tau1_rest) / self.alpha_tau1
        q_km = (km - self.km_rest) / self.alpha_km
        if max(abs(q_a - q_tau1), abs(q_a - q_km)) > tolerance:
            raise ValueError("Slow Ding state is off the exact fatigue manifold")
        return a / self.a_rest

    def invariant_residual(self, a: float, tau1: float, km: float) -> float:
        """Largest absolute disagreement among the three shared coordinates."""
        q = np.asarray(((float(a) - self.a_rest) / self.alpha_a,
                        (float(tau1) - self.tau1_rest) / self.alpha_tau1,
                        (float(km) - self.km_rest) / self.alpha_km))
        if not np.all(np.isfinite(q)):
            return math.inf
        return float(np.max(np.abs(q - q[0])))

    def capacity_interval(self, slow_lower: Sequence[float], slow_upper: Sequence[float]) -> tuple[float, float]:
        """Intersect A/Tau1/Km box bounds with the one-dimensional manifold."""
        lower = np.asarray(slow_lower, float)
        upper = np.asarray(slow_upper, float)
        if lower.shape != (3,) or upper.shape != (3,) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("slow bounds must be finite vectors ordered as A, Tau1, Km")
        if np.any(lower > upper):
            raise ValueError("Slow lower bounds must not exceed upper bounds")
        # Each state is affine in r.  Transform its interval while respecting
        # coefficient signs, then intersect all three r intervals.
        reference = np.asarray((self.a_rest, self.tau1_rest, self.km_rest))
        slopes = np.asarray((self.a_rest,
                             self.alpha_tau1 * self.a_rest / self.alpha_a,
                             self.alpha_km * self.a_rest / self.alpha_a))
        intercepts = reference - slopes
        candidates = np.column_stack(((lower - intercepts) / slopes, (upper - intercepts) / slopes))
        ratios_lower, ratios_upper = np.min(candidates, axis=1), np.max(candidates, axis=1)
        lo, hi = float(np.max(ratios_lower)), float(np.min(ratios_upper))
        if lo > hi:
            raise ValueError("Slow-state bounds have no intersection with the fatigue manifold")
        return lo, hi
