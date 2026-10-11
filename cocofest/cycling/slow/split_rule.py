"""Auditable conditional load sharing, subject only to declared comfort bounds.

This heuristic transfers load toward the arm with the larger uncensored
conditional threshold estimate. It does not certify RHO feasibility and must
not be enabled in a session before its inputs pass continuation validation.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from .remaining_cycles import ConditionalRemainingCycles


@dataclass(frozen=True)
class SplitComfort:
    """Per-arm Nm bounds and per-update slew/asymmetry bounds (left, right)."""

    minimum: tuple[float, float]
    maximum: tuple[float, float]
    delta_max: float = .05
    asymmetry_max: float = .3

    def __post_init__(self):
        low, high = np.asarray(self.minimum, dtype=float), np.asarray(self.maximum, dtype=float)
        if low.shape != (2,) or high.shape != (2,) or not np.isfinite([low, high]).all():
            raise ValueError("comfort bounds require two finite values per arm")
        if np.any(low < 0) or np.any(low > high):
            raise ValueError("comfort bounds must be nonnegative and ordered")
        for name in ("delta_max", "asymmetry_max"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "minimum", tuple(float(v) for v in low))
        object.__setattr__(self, "maximum", tuple(float(v) for v in high))


@dataclass(frozen=True)
class SplitDecision:
    previous_torque: tuple[float, float]
    torque: tuple[float, float]
    proposed_transfer_to_left: float
    admissible_left_interval: tuple[float, float]
    reason: str
    comfort: SplitComfort
    conditional_estimates: tuple[int | None, int | None]
    initial_state_receipts: tuple[str | None, str | None]
    interpretation: str = "conditional_sharing_heuristic_not_feasibility_certificate"

    @property
    def receipt_sha256(self) -> str:
        """Content digest of inputs and decision, not a digital signature."""
        content = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(content.encode()).hexdigest()


def _valid_estimate(value) -> bool:
    if not isinstance(value, ConditionalRemainingCycles):
        return False
    for number in (value.cycles_within_bounds, value.horizon):
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            return False
    if value.cycles_within_bounds > value.horizon:
        return False
    crossing = value.first_exceedance_cycle
    if crossing is not None:
        if isinstance(crossing, bool) or not isinstance(crossing, int) or crossing < 0 or crossing > value.horizon:
            return False
        if value.cycles_within_bounds != max(0, crossing-1):
            return False
    elif value.cycles_within_bounds != value.horizon:
        return False
    digest = value.initial_receipt_sha256
    return (value.interpretation == "conditional_state_threshold_only_not_force_or_feasibility_prediction"
            and isinstance(digest, str) and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest))


def equalize_remaining_cycles(
    current_torque: tuple[float, float],
    estimates: Mapping[str, ConditionalRemainingCycles | None],
    comfort: SplitComfort,
) -> SplitDecision:
    """One slow update at fixed total torque, with conservative input handling.

    The raw transfer is delta_max*(N_left-N_right)/(N_left+N_right). Projection
    onto the exact one-dimensional comfort interval enforces both arms' torque
    and slew bounds and the asymmetry constraint. Equal estimates hold the
    current split. Missing, malformed, zero-life or censored inputs also hold.
    An already-invalid current split raises: holding it would violate comfort,
    and automatically changing it would need a separate recovery policy.
    """
    current = np.asarray(current_torque, dtype=float)
    if current.shape != (2,) or not np.isfinite(current).all():
        raise ValueError("current_torque must contain finite left/right values")
    if (np.any(current < comfort.minimum) or np.any(current > comfort.maximum)
            or abs(current[0]-current[1]) > comfort.asymmetry_max + 1e-12):
        raise ValueError("current split violates comfort; a recovery policy is required")
    left, right = (float(v) for v in current)
    total = left + right
    lower = max(comfort.minimum[0], total-comfort.maximum[1], left-comfort.delta_max,
                total-right-comfort.delta_max, (total-comfort.asymmetry_max)/2)
    upper = min(comfort.maximum[0], total-comfort.minimum[1], left+comfort.delta_max,
                total-right+comfort.delta_max, (total+comfort.asymmetry_max)/2)
    values = (estimates.get("left"), estimates.get("right"))
    valid = tuple(_valid_estimate(value) for value in values)
    counts = tuple(value.cycles_within_bounds if ok else None for value, ok in zip(values, valid))
    receipts = tuple(value.initial_receipt_sha256 if ok else None for value, ok in zip(values, valid))
    transfer, new_left, reason = 0.0, left, "hold_invalid_estimate"
    if all(valid):
        if any(value.censored for value in values):
            reason = "hold_censored_estimate"
        elif min(counts) == 0:
            reason = "hold_zero_remaining_cycles"
        elif counts[0] == counts[1]:
            reason = "hold_equal_estimates"
        else:
            transfer = comfort.delta_max * (counts[0]-counts[1]) / (counts[0]+counts[1])
            new_left = min(upper, max(lower, left+transfer))
            reason = "adjusted" if new_left != left else "hold_comfort_boundary"
    # Preserve the exact original pair when holding (including roundoff).
    new_pair = (new_left, total-new_left) if new_left != left else (left, right)
    return SplitDecision((left, right), new_pair, transfer, (lower, upper), reason, comfort, counts, receipts)
