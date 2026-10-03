"""Fixed per-muscle weights for the direct pulse-width quadratic objective."""

from __future__ import annotations

from typing import Iterable

import numpy as np


CONTROL_WEIGHT_PARAMETER_KEY = "rho_control_weights"


def _validated_weights(weights: Iterable[float], size: int | None = None) -> np.ndarray:
    values = np.asarray(tuple(float(weight) for weight in weights), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Control weights must be a non-empty one-dimensional sequence.")
    if size is not None and values.size != size:
        raise ValueError("Control-weight dimensions cannot change between RHO windows.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Control weights must be finite and strictly positive.")
    return values


class ParametricControlWeightBinding:
    """Own fixed NLP parameters for ``sum_i w_i * (PW_i/PWmax_i)^2``."""

    def __init__(self, weights: Iterable[float]):
        self.weights = _validated_weights(weights)
        self._nlp = None

    @property
    def size(self) -> int:
        return int(self.weights.size)

    def parameter_options(self, *, use_sx: bool = True) -> dict:
        from bioptim import BoundsList, InitialGuessList, InterpolationType, ParameterList, VariableScaling

        parameters = ParameterList(use_sx=use_sx)
        parameters.add(name=CONTROL_WEIGHT_PARAMETER_KEY, function=None, size=self.size,
                       scaling=VariableScaling(CONTROL_WEIGHT_PARAMETER_KEY, np.ones(self.size)))
        values = self.weights[:, np.newaxis]
        bounds, initial = BoundsList(), InitialGuessList()
        bounds.add(CONTROL_WEIGHT_PARAMETER_KEY, min_bound=values, max_bound=values,
                   interpolation=InterpolationType.CONSTANT)
        initial.add(CONTROL_WEIGHT_PARAMETER_KEY, initial_guess=values)
        return {"parameters": parameters, "parameter_bounds": bounds, "parameter_init": initial}

    def attach(self, nmpc) -> None:
        if self._nlp is not None and self._nlp is not nmpc.nlp[0]:
            raise RuntimeError("Control-weight binding cannot be attached to a rebuilt NLP.")
        self._nlp = nmpc.nlp[0]

    def summary(self) -> dict:
        return {"parameter_key": CONTROL_WEIGHT_PARAMETER_KEY, "parameter_size": self.size,
                "weights": self.weights.tolist(), "objective_graph_rebuild_required": False}
