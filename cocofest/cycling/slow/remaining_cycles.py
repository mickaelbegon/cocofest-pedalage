"""Conditional time-to-threshold under a prescribed repeated force profile.

The thresholds are declared state bounds, not a learned feasibility boundary.
The result is never a certificate of realizable RHO cycles or endurance. Using
it for load decisions requires a separate validation against continuations.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from cocofest.optimization.ding_slow_cycle import DingSlowCycleMap
from .slow_model import SlowState, propagate_prescribed_force


@dataclass(frozen=True)
class SlowStateBounds:
    """Inclusive bounds (muscles, 3), ordered as the input SlowState."""

    lower: tuple[tuple[float, float, float], ...]
    upper: tuple[tuple[float, float, float], ...]

    def __post_init__(self):
        low, high = np.asarray(self.lower, dtype=float), np.asarray(self.upper, dtype=float)
        if low.ndim != 2 or low.shape[1] != 3 or high.shape != low.shape or not low.shape[0]:
            raise ValueError("bounds must have shape (muscles, 3)")
        if np.isnan(low).any() or np.isnan(high).any() or np.any(low > high):
            raise ValueError("bounds must be ordered and must not contain NaN")
        object.__setattr__(self, "lower", tuple(tuple(float(v) for v in row) for row in low))
        object.__setattr__(self, "upper", tuple(tuple(float(v) for v in row) for row in high))


@dataclass(frozen=True)
class ConditionalRemainingCycles:
    cycles_within_bounds: int
    first_exceedance_cycle: int | None
    horizon: int
    limiting_muscles: tuple[str, ...]
    initial_receipt_sha256: str
    interpretation: str = "conditional_state_threshold_only_not_force_or_feasibility_prediction"

    @property
    def censored(self) -> bool:
        return self.first_exceedance_cycle is None


def remaining_cycles(
    state: SlowState, maps: Mapping[str, DingSlowCycleMap], bounds: SlowStateBounds, *, horizon: int,
) -> ConditionalRemainingCycles:
    """Find the first sampled-cycle bound violation using exact affine powers.

    Under a repeated profile each scalar state is monotonic toward its affine
    equilibrium, so an initially inside trajectory cannot leave then reenter.
    Binary search uses O(log(horizon)) evaluations, without time integration.
    ``None`` means no crossing observed through the horizon, not infinite life.
    A crossing at cycle n gives n-1 completed in-bound cycles; an initially
    outside state reports crossing 0 and zero in-bound cycles.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 0:
        raise ValueError("horizon must be a non-negative integer")
    horizon = int(horizon)
    low, high = np.asarray(bounds.lower), np.asarray(bounds.upper)
    if low.shape != state.matrix.shape:
        raise ValueError("bounds must have one row per observed muscle")

    def violated(n):
        values = propagate_prescribed_force(state, maps, n)
        return np.any((values < low) | (values > high), axis=1)

    initial = violated(0)
    if np.any(initial):
        crossing, violations = 0, initial
    elif not np.any(violated(horizon)):
        return ConditionalRemainingCycles(horizon, None, horizon, (), state.receipt_sha256)
    else:
        left, right = 0, horizon
        while right - left > 1:
            middle = (left + right) // 2
            if np.any(violated(middle)):
                right = middle
            else:
                left = middle
        crossing, violations = right, violated(right)
    limiting = tuple(name for name, flag in zip(state.muscle_names, violations, strict=True) if flag)
    return ConditionalRemainingCycles(max(0, crossing - 1), crossing, horizon, limiting, state.receipt_sha256)
