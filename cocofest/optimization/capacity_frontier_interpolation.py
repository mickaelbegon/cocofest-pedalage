"""Conservative, offline-only local exchange models from capacity witnesses.

The model deliberately keys every estimate to an exact conditioned state hash.
It never interpolates across a different fast Ding/mechanical state, and it is
not an online RHO objective or a feasibility certificate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TricepsExchangeSlice:
    """One exact fast-state slice with a triceps-minimum witness and ε cuts."""

    condition_sha256: str
    checkpoint_cycle: int
    muscle_order: tuple[str, ...]
    capacity_domain: tuple[float, float]
    minimum_triceps_capacity: tuple[float, ...]
    epsilon_capacity: Mapping[str, tuple[float, ...]]
    epsilon_upper: Mapping[str, float]
    witness_sha256: Mapping[str, str]

    def validate(self) -> None:
        if not isinstance(self.condition_sha256, str) or len(self.condition_sha256) != 64:
            raise ValueError("condition_sha256 must be a SHA-256 digest")
        if type(self.checkpoint_cycle) is not int or self.checkpoint_cycle < 0:
            raise ValueError("checkpoint_cycle must be non-negative")
        names = tuple(self.muscle_order)
        if "Triceps" not in names or len(names) < 2 or len(set(names)) != len(names):
            raise ValueError("A distinct Triceps coordinate and other muscles are required")
        domain = np.asarray(self.capacity_domain, float)
        if domain.shape != (2,) or not np.all(np.isfinite(domain)) or not (0 <= domain[0] < domain[1] <= 1):
            raise ValueError("capacity_domain must be one ordered physical box")
        baseline = np.asarray(self.minimum_triceps_capacity, float)
        if baseline.shape != (len(names),) or not np.all(np.isfinite(baseline)) or np.any((baseline < 0) | (baseline > 1)):
            raise ValueError("minimum_triceps_capacity must be physical")
        expected = set(names) - {"Triceps"}
        if set(self.epsilon_capacity) != expected or set(self.epsilon_upper) != expected:
            raise ValueError("One ε capacity witness and upper bound are required for every non-triceps muscle")
        if set(self.witness_sha256) != expected | {"minimum_triceps"} or any(
                not isinstance(value, str) or len(value) != 64 for value in self.witness_sha256.values()):
            raise ValueError("A SHA-256 witness provenance is required for every support solve")
        tri = names.index("Triceps")
        for muscle, capacity in self.epsilon_capacity.items():
            value = np.asarray(capacity, float)
            index = names.index(muscle)
            upper = float(self.epsilon_upper[muscle])
            if (not math.isfinite(upper) or not domain[0] <= upper <= domain[1]
                    or value.shape != baseline.shape or not np.all(np.isfinite(value)) or np.any((value < 0) | (value > 1))
                    or not np.isclose(value[index], upper, atol=1e-4, rtol=0)
                    or not value[index] < baseline[index] or not value[tri] > baseline[tri]):
                raise ValueError("Each ε witness must trade reduced non-triceps reserve for more triceps reserve")

    def exchange_slopes(self) -> dict[str, float]:
        """Conservative positive gain in triceps reserve per relaxed reserve."""
        self.validate()
        names, baseline = self.muscle_order, np.asarray(self.minimum_triceps_capacity, float)
        tri = names.index("Triceps")
        return {muscle: float((np.asarray(value)[tri] - baseline[tri]) /
                              (baseline[names.index(muscle)] - np.asarray(value)[names.index(muscle)]))
                for muscle, value in self.epsilon_capacity.items()}


def build_exact_conditioned_exchange_model(slices: Sequence[TricepsExchangeSlice]) -> dict:
    """Seal exact-state exchange slopes; unknown conditions must be rejected."""
    rows = tuple(slices)
    if not rows:
        raise ValueError("At least one exact conditioned slice is required")
    seen, serial = set(), []
    order = rows[0].muscle_order
    for item in rows:
        item.validate()
        if item.muscle_order != order or item.capacity_domain != rows[0].capacity_domain or item.condition_sha256 in seen:
            raise ValueError("Slices must share muscle order, atlas domain and distinct exact condition hashes")
        seen.add(item.condition_sha256)
        serial.append({**asdict(item), "exchange_slopes": item.exchange_slopes()})
    value = {"schema_version": 1, "kind": "exact_conditioned_triceps_exchange_model",
             "muscle_order": list(order), "slices": serial,
             "capacity_domain": list(rows[0].capacity_domain),
             "domain_rule": "exact prepared-problem condition_sha256 match only; no interpolation across fast states",
             "online_use": "offline diagnostic only; not a feasibility certificate or RHO cost"}
    return {**value, "model_sha256": sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def lookup_exact_exchange(model: Mapping, condition_sha256: str) -> dict[str, float] | None:
    """Return a slope only for a measured fast-state condition; otherwise None."""
    if model.get("kind") != "exact_conditioned_triceps_exchange_model":
        raise ValueError("Wrong exchange-model kind")
    for item in model.get("slices", []):
        if item.get("condition_sha256") == condition_sha256:
            return dict(item["exchange_slopes"])
    return None


def audit_linear_checkpoint_interpolation(
        lower: TricepsExchangeSlice, holdout: TricepsExchangeSlice, upper: TricepsExchangeSlice, *,
        maximum_relative_error: float = 0.10) -> dict:
    """Audit, but do not deploy, interpolation between two fast-state slices.

    The cycle index is merely an ordering coordinate.  It is not asserted to
    capture all fast Ding or mechanical states.  Consequently a passing audit
    produces an *offline hypothesis* with an uncertainty band; it never
    authorizes direct use in a RHO cost.  This makes the c120/c140/c160
    triplet a genuine held-out test rather than fitting a curve through all
    three slices.
    """
    rows = (lower, holdout, upper)
    for row in rows:
        row.validate()
    if (lower.muscle_order != holdout.muscle_order or upper.muscle_order != holdout.muscle_order
            or lower.capacity_domain != holdout.capacity_domain
            or upper.capacity_domain != holdout.capacity_domain):
        raise ValueError("Interpolation slices must share muscle order and capacity domain")
    if not (lower.checkpoint_cycle < holdout.checkpoint_cycle < upper.checkpoint_cycle):
        raise ValueError("Holdout checkpoint must lie strictly between interpolation anchors")
    tolerance = float(maximum_relative_error)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("maximum_relative_error must be finite and positive")
    fraction = ((holdout.checkpoint_cycle - lower.checkpoint_cycle)
                / (upper.checkpoint_cycle - lower.checkpoint_cycle))
    low, observed, high = (row.exchange_slopes() for row in rows)
    predicted = {muscle: float((1. - fraction) * low[muscle] + fraction * high[muscle])
                 for muscle in holdout.epsilon_capacity}
    relative_error = {
        muscle: float(abs(predicted[muscle] - observed[muscle]) / max(abs(observed[muscle]), 1e-12))
        for muscle in predicted
    }
    # The observed holdout discrepancy is retained as a symmetric uncertainty
    # interval.  Later code may only use the lower confidence slope when it
    # has a separate causal validation; optimism is deliberately unavailable.
    lower_band = {muscle: max(0., predicted[muscle] - abs(predicted[muscle] - observed[muscle]))
                  for muscle in predicted}
    upper_band = {muscle: predicted[muscle] + abs(predicted[muscle] - observed[muscle])
                  for muscle in predicted}
    return {
        "schema_version": 1,
        "kind": "offline_checkpoint_exchange_interpolation_holdout",
        "muscle_order": list(holdout.muscle_order),
        "capacity_domain": list(holdout.capacity_domain),
        "anchors": {
            "lower": {"checkpoint_cycle": lower.checkpoint_cycle,
                      "condition_sha256": lower.condition_sha256},
            "upper": {"checkpoint_cycle": upper.checkpoint_cycle,
                      "condition_sha256": upper.condition_sha256},
        },
        "holdout": {"checkpoint_cycle": holdout.checkpoint_cycle,
                    "condition_sha256": holdout.condition_sha256},
        "fraction": float(fraction),
        "predicted_exchange_slopes": predicted,
        "observed_exchange_slopes": observed,
        "relative_error": relative_error,
        "conservative_slope_interval": {
            muscle: [lower_band[muscle], upper_band[muscle]] for muscle in predicted
        },
        "maximum_relative_error": tolerance,
        "passed": bool(all(value <= tolerance for value in relative_error.values())),
        "online_use": "forbidden_pending_causal_RHO_holdout",
    }


def diagnostic_frontier_weight_hint(exchange_slopes: Mapping[str, float], *, triceps_multiplier: float = 2.,
                                    lower: float = .25, upper: float = 4.) -> dict[str, float]:
    """Map locally measured exchanges to a *diagnostic* fatigue-weight hint.

    A larger slope means that allowing fatigue in that non-triceps muscle buys
    more triceps capacity.  Its fatigue penalty is therefore smaller.  The
    mapping is rank preserving, normalized to mean one, and intentionally
    does not claim to be an optimal policy or a feasibility constraint.
    """
    if not exchange_slopes:
        raise ValueError("At least one non-triceps exchange slope is required")
    slopes = {str(name): float(value) for name, value in exchange_slopes.items()}
    if (any(not name or not math.isfinite(value) or value <= 0 for name, value in slopes.items())
            or "Triceps" in slopes):
        raise ValueError("Exchange slopes must be finite positive non-triceps values")
    if not all(math.isfinite(value) and value > 0 for value in (triceps_multiplier, lower, upper)) or lower > upper:
        raise ValueError("Weight multiplier and bounds must be ordered positive finite values")
    reference = float(np.median(list(slopes.values())))
    weights = {name: float(np.clip(reference / slope, lower, upper)) for name, slope in slopes.items()}
    weights["Triceps"] = float(np.clip(triceps_multiplier * max(weights.values()), lower, upper))
    mean = float(np.mean(list(weights.values())))
    return {name: value / mean for name, value in weights.items()}
