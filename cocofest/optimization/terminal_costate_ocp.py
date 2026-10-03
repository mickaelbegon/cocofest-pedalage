"""Experimental affine terminal value for a one-cycle compiled RHO.

``V`` is remaining feasible cycles to *maximise*.  The minimised terminal
cost is ``-activation * (V0 + grad(V) dot (z_terminal - z0))``.  Gradients must
come from a separately validated slow model; this binding only transports a
numerical local approximation into an unchanged NLP graph.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
from typing import Mapping, Sequence

import numpy as np

from .task_reserve_ocp import TaskReserveObjectiveBinding, TaskReserveStateCoordinate


TERMINAL_COSTATE_MODE = "terminal_remaining_cycles_costate_experimental"


def _vector(value, dimension: int, name: str, *, positive: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (dimension,) or not np.all(np.isfinite(result)) or (positive and np.any(result <= 0.)):
        raise ValueError(f"{name} must contain {dimension} finite" + (" positive" if positive else "") + " values.")
    return result


class TerminalCostateObjectiveBinding(TaskReserveObjectiveBinding):
    """Opt-in fixed-graph affine remaining-endurance value.

    The inherited ``rho_task_reserve`` numerical channel also permits a
    separate fatigue-weight parameter.  This mode cannot coexist with the
    PACE-RT objective because both own that same parameter key.
    """

    def __init__(self, coordinates: Sequence[TaskReserveStateCoordinate], *,
                 task_context: Mapping, model_sha256: str, maximum_age_cycles: int = 20):
        super().__init__(coordinates, task_context=task_context, model_sha256=model_sha256,
                         target=0., smoothing=1e-3, maximum_age_cycles=maximum_age_cycles)
        self.graph_signature_sha256 = sha256(json.dumps({
            "version": "terminal-remaining-cycles-affine-v1-experimental",
            "coordinate_layout": [asdict(c) for c in self.coordinates],
            "parameter_size": int(self.values.size),
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.last_model = None

    def objective(self, controller):
        from casadi import vertcat

        point = vertcat(*[
            (controller.states[c.state_key].cx[c.index] - c.offset) / c.scale
            for c in self.coordinates
        ])
        parameters = controller.parameters["rho_task_reserve"].cx
        if int(parameters.numel()) != self.values.size:
            raise ValueError("Terminal costate parameter vector has the wrong dimension.")
        n = self.dimension
        remaining_cycles = parameters[1]
        for index in range(n):
            remaining_cycles += parameters[2 + n + index] * (point[index] - parameters[2 + index])
        return -parameters[0] * remaining_cycles

    def update_from_remaining_cycles_value(self, nmpc, *, remaining_cycles: float,
                                           center: Sequence[float], gradient: Sequence[float],
                                           trust_radius: Sequence[float],
                                           evaluation_coordinates: Sequence[float],
                                           source_completed_cycles: int, completed_cycles: int,
                                           activation: float = 1.) -> dict:
        """Install a local gradient in *remaining cycles per normalised state*.

        This method checks numerical shape, provenance age and the current
        point's trust box.  Acceptance of the slow-model gradient remains a
        separate scientific validation step.
        """
        if type(source_completed_cycles) is not int or type(completed_cycles) is not int:
            raise ValueError("Costate cycle indices must be integers.")
        if (source_completed_cycles < 0 or completed_cycles < source_completed_cycles
                or completed_cycles - source_completed_cycles > self.maximum_age_cycles):
            raise ValueError("Terminal costate is stale or belongs to a future cycle.")
        remaining_cycles, activation = float(remaining_cycles), float(activation)
        if not math.isfinite(remaining_cycles) or remaining_cycles < 0.:
            raise ValueError("Remaining cycles must be finite and nonnegative.")
        if not math.isfinite(activation) or not 0. <= activation <= 1.:
            raise ValueError("Costate activation must lie in [0, 1].")
        center = _vector(center, self.dimension, "center")
        gradient = _vector(gradient, self.dimension, "gradient")
        radius = _vector(trust_radius, self.dimension, "trust_radius", positive=True)
        point = _vector(evaluation_coordinates, self.dimension, "evaluation_coordinates")
        if np.any(np.abs(point - center) > radius):
            raise ValueError("Current terminal coordinate lies outside the costate trust box.")
        self._write(nmpc, np.r_[activation, remaining_cycles, center, gradient])
        self.source_completed_cycles = source_completed_cycles
        self.last_model = {"remaining_cycles": remaining_cycles, "center": center.tolist(),
                           "gradient_remaining_cycles": gradient.tolist(),
                           "trust_radius": radius.tolist(), "activation": activation,
                           "source_completed_cycles": source_completed_cycles}
        return self.summary()

    def update(self, *args, **kwargs):
        raise TypeError("Use update_from_remaining_cycles_value with an explicit remaining-endurance gradient.")

    def validate_terminal_point(self, coordinates, *, completed_cycles: int):
        if self.last_model is None or self.values[0] == 0.:
            return {"active": False, "terminal_trust_validated": False}
        point = _vector(coordinates, self.dimension, "terminal coordinates")
        center = np.asarray(self.last_model["center"])
        radius = np.asarray(self.last_model["trust_radius"])
        age = int(completed_cycles) - int(self.source_completed_cycles)
        fraction = np.abs(point - center) / radius
        return {"active": True,
                "terminal_trust_validated": bool(0 <= age <= self.maximum_age_cycles and np.all(fraction <= 1.)),
                "maximum_trust_fraction": float(np.max(fraction)), "source_age_cycles": age,
                "predicted_remaining_cycles": float(self.last_model["remaining_cycles"] +
                                                    np.dot(self.last_model["gradient_remaining_cycles"], point - center))}

    def summary(self):
        return {**super().summary(), "mode": TERMINAL_COSTATE_MODE,
                "value_units": "remaining feasible cycles",
                "objective_sign": "minimise negative remaining-cycle value",
                "gradient_scientifically_validated": False,
                "last_model": self.last_model}
