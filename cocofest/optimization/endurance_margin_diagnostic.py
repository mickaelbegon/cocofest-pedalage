"""Offline diagnostics of a local task margin and its rate of consumption.

A feasible one-cycle work witness is a lower bound, not a maximum. A local
affine model is conditional on the source mechanics/history and has a finite
trust region. Neither a secant extrapolation nor repeating a saved force
profile turns those facts into a physiological exhaustion certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .ding_slow_cycle import DingSlowCycleMap


def consumption_time_proxy(margin: float, margin_decrement_per_cycle: float) -> dict:
    """Return a heuristic zero-crossing time; non-depletion stays unknown.

    Zero/negative local margin does not prove failure. In particular this
    function must never be used to label a failed numerical solve as fatigue.
    """
    if not math.isfinite(margin) or not math.isfinite(margin_decrement_per_cycle):
        raise ValueError("Margin and decrement must be finite")
    if margin <= 0:
        return {"cycles": None, "status": "no_positive_margin_witness"}
    if margin_decrement_per_cycle <= 0:
        return {"cycles": None, "status": "no_positive_depletion_measured"}
    return {"cycles": margin / margin_decrement_per_cycle,
            "status": "conditional_linear_extrapolation"}


@dataclass(frozen=True)
class LocalMarginDiagnostic:
    """An archived affine load-factor model, with explicit coordinate scales."""

    names: tuple[str, ...]
    center: tuple[float, ...]
    scales: tuple[float, ...]
    offsets: tuple[float, ...]
    gradient: tuple[float, ...]
    trust_radius: tuple[float, ...]
    load_factor: float

    def __post_init__(self):
        size = len(self.names)
        if not size or len(set(self.names)) != size:
            raise ValueError("Distinct nonempty state names required")
        for field in ("center", "scales", "offsets", "gradient", "trust_radius"):
            value = np.asarray(getattr(self, field), float)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {field}")
            object.__setattr__(self, field, tuple(value))
        if np.any(np.asarray(self.scales) <= 0) or np.any(np.asarray(self.trust_radius) <= 0):
            raise ValueError("Scales and trust radii must be positive")
        if not math.isfinite(self.load_factor) or self.load_factor < 0:
            raise ValueError("Nonnegative finite load factor required")
        object.__setattr__(self, "names", tuple(self.names))

    def coordinates(self, physical: Mapping[str, float]) -> np.ndarray:
        values = np.asarray([physical[name] for name in self.names], float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite physical state")
        return (values - self.offsets) / self.scales

    def evaluate(self, coordinates) -> dict:
        values = np.asarray(coordinates, float)
        if values.shape != (len(self.names),) or not np.isfinite(values).all():
            raise ValueError("Invalid normalized coordinates")
        difference = values - self.center
        fraction = float(np.max(np.abs(difference) / self.trust_radius))
        return {"affine_margin": float(self.load_factor - 1 + np.dot(self.gradient, difference)),
                "trust_fraction": fraction, "inside_coordinate_trust_region": fraction <= 1 + 1e-12}


def repeated_force_margin_path(model: LocalMarginDiagnostic,
                               initial_physical: Mapping[str, float],
                               maps: Mapping[str, DingSlowCycleMap], *, horizon: int) -> dict:
    """Repeat observed force while freezing Cn/F endpoint and all other data.

    This produces a *conditional diagnostic*, not a forecast of realizable PW.
    Fast endpoint states are not predicted. A first affine margin crossing is
    deliberately kept separate from the first trust-region exit.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    required = {f"{state}_{muscle}" for muscle in maps for state in ("A", "Tau1", "Km")}
    selected_slow = {name for name in model.names if name.startswith(("A_", "Tau1_", "Km_"))}
    if not maps or required != selected_slow:
        raise ValueError("Exactly one slow cycle map per selected muscle is required")
    current = dict(initial_physical)
    initial = {muscle: np.asarray([current[f"{state}_{muscle}"] for state in ("A", "Tau1", "Km")])
               for muscle in maps}
    rows = []
    for cycle in range(horizon + 1):
        for muscle, cycle_map in maps.items():
            end = cycle_map.propagate(initial[muscle], repetitions=cycle)
            current.update({f"{state}_{muscle}": float(value)
                            for state, value in zip(("A", "Tau1", "Km"), end)})
        rows.append({"future_cycle": cycle, **model.evaluate(model.coordinates(current))})
    first_exit = next((row["future_cycle"] for row in rows if not row["inside_coordinate_trust_region"]), None)
    first_crossing = next((row["future_cycle"] for row in rows if row["affine_margin"] <= 0), None)
    decrement = rows[0]["affine_margin"] - rows[1]["affine_margin"]
    return {"rows": rows, "first_coordinate_trust_exit": first_exit,
            "first_affine_zero_crossing": first_crossing,
            "zero_crossing_inside_trust": first_crossing is not None and
                (first_exit is None or first_crossing < first_exit),
            "one_cycle_margin_decrement": decrement,
            "linear_consumption_time": consumption_time_proxy(rows[0]["affine_margin"], decrement),
            "future_force_realizability_validated": False,
            "physiological_failure_certified": False,
            "scope": "fixed observed force polynomial; fast endpoint states frozen; local margin extrapolation"}


def witness_grid_consumption(rows: list[dict]) -> list[dict]:
    """Audit coarse observed feasible-work lower bounds without inventing maxima.

    Required keys: cycle, load_lower_bound, tested_maximum. A decrease between
    lower bounds is not a rigorous lower/upper bound on the true consumption
    rate. Successful points at the highest tested load are right-censored.
    """
    result = []
    previous = None
    for row in sorted(rows, key=lambda item: item["cycle"]):
        cycle, load, maximum = row["cycle"], row["load_lower_bound"], row["tested_maximum"]
        if (isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 0 or
                not math.isfinite(load) or not math.isfinite(maximum) or load > maximum + 1e-12):
            raise ValueError("Invalid witnessed grid row")
        entry = dict(row, reserve_lower_bound=load-1, at_tested_load_cap=abs(load-maximum) <= 1e-12)
        if previous is None:
            entry.update(secant_decrement_per_cycle=None, time_proxy={"cycles": None, "status": "no_previous_checkpoint"})
        else:
            if cycle <= previous["cycle"]:
                raise ValueError("Checkpoint cycles must be distinct")
            decrement = (previous["load_lower_bound"] - load) / (cycle - previous["cycle"])
            entry.update(secant_decrement_per_cycle=decrement,
                         time_proxy=consumption_time_proxy(load - 1, decrement))
        result.append(entry)
        previous = row
    return result
