"""Fixed numerical fatigue weights that keep the RHO objective graph static.

The fatigue PACE policy changes four positive relative weights between RHO
windows.  Replacing the objective via ``ocp.update_objectives`` rebuilds the
symbolic NLP and invalidates CasADi's compiled callbacks.  This binding makes
those weights fixed *parameters* of one unchanged graph instead.  They are
not states and have no dynamics; their lower and upper bounds are equal.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


FATIGUE_WEIGHT_PARAMETER_KEY = "rho_fatigue_weights"


def _validated_weights(
    weights: Iterable[float], size: int | None = None, *, allow_zero: bool = False
) -> np.ndarray:
    values = np.asarray(tuple(float(weight) for weight in weights), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Fatigue weights must be a non-empty one-dimensional sequence.")
    if size is not None and values.size != size:
        raise ValueError("Fatigue weight dimensions cannot change between RHO windows.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0 if allow_zero else values <= 0.0):
        qualifier = "non-negative" if allow_zero else "strictly positive"
        raise ValueError(f"Fatigue weights must be finite and {qualifier}.")
    return values


class ParametricFatigueWeightBinding:
    """Own the fixed parameter buffers of the compiled fatigue objective."""

    def __init__(self, weights: Iterable[float], *, allow_zero: bool = False):
        self.allow_zero = bool(allow_zero)
        self.weights = _validated_weights(weights, allow_zero=self.allow_zero)
        self.update_count = 0
        # A primary ACADOS OCP and its separate IPOPT recovery OCP share the
        # same policy vector but have independent compiled NLPs.
        self._attached_nlps: dict[int, tuple[object, object]] = {}

    @property
    def size(self) -> int:
        return int(self.weights.size)

    def parameter_options(self, *, use_sx: bool = True) -> dict:
        from bioptim import BoundsList, InitialGuessList, InterpolationType, ParameterList, VariableScaling

        parameters = ParameterList(use_sx=use_sx)
        parameters.add(
            name=FATIGUE_WEIGHT_PARAMETER_KEY,
            function=None,
            size=self.size,
            scaling=VariableScaling(FATIGUE_WEIGHT_PARAMETER_KEY, np.ones(self.size)),
        )
        values = self.weights[:, np.newaxis]
        bounds, initial = BoundsList(), InitialGuessList()
        bounds.add(
            FATIGUE_WEIGHT_PARAMETER_KEY,
            min_bound=values,
            max_bound=values,
            interpolation=InterpolationType.CONSTANT,
        )
        initial.add(FATIGUE_WEIGHT_PARAMETER_KEY, initial_guess=values)
        return {"parameters": parameters, "parameter_bounds": bounds, "parameter_init": initial}

    def attach(self, nmpc) -> None:
        nlp = nmpc.nlp[0]
        previous = self._attached_nlps.get(id(nmpc))
        if previous is not None and previous[1] is not nlp:
            raise RuntimeError("Fatigue-weight binding cannot be attached to a rebuilt NLP.")
        self._attached_nlps[id(nmpc)] = (nmpc, nlp)

    def update(self, nmpc, weights: Iterable[float]) -> dict:
        """Update numerical equality bounds without modifying OCP objectives."""
        values = _validated_weights(weights, self.size, allow_zero=self.allow_zero)[:, np.newaxis]
        from bioptim import InitialGuessList
        self.attach(nmpc)
        # Update every OCP that consumes this binding (e.g. ACADOS plus its
        # separate IPOPT recovery transcription) without rebuilding either.
        for attached_nmpc, attached_nlp in self._attached_nlps.values():
            if attached_nmpc.nlp[0] is not attached_nlp:
                raise RuntimeError("Fatigue-weight binding cannot update a rebuilt NLP.")
            bounds = attached_nmpc.parameter_bounds[FATIGUE_WEIGHT_PARAMETER_KEY]
            if bounds.min.shape != values.shape or bounds.max.shape != values.shape:
                raise ValueError("Fatigue-weight parameter bounds have an unexpected shape.")
            bounds.min[...] = values
            bounds.max[...] = values
            parameter_init = InitialGuessList()
            parameter_init.add(FATIGUE_WEIGHT_PARAMETER_KEY, initial_guess=values.copy())
            attached_nmpc.update_initial_guess(parameter_init=parameter_init)

            # Bioptim's ACADOS backend represents OCP parameters as constant
            # augmented states. Its interface snapshots their bounds at
            # construction, so synchronize that snapshot too. The next
            # ACADOS window applies these arrays at every stage.
            acados_interface = getattr(attached_nmpc, "ocp_solver", None)
            if acados_interface is not None and hasattr(acados_interface, "x_bound_min"):
                parameter = attached_nmpc.parameters[FATIGUE_WEIGHT_PARAMETER_KEY]
                parameter_indices = np.asarray(parameter.index, dtype=int).reshape(-1)
                scaled_bounds = bounds.scale(parameter.scaling.scaling)
                if parameter_indices.size != self.size:
                    raise ValueError("ACADOS fatigue-weight state indices have an unexpected shape.")
                if np.max(parameter_indices, initial=-1) >= acados_interface.x_bound_min.shape[0]:
                    raise ValueError("ACADOS fatigue-weight state indices exceed the augmented state bounds.")
                acados_interface.x_bound_min[parameter_indices, :] = scaled_bounds.min[:, :1]
                acados_interface.x_bound_max[parameter_indices, :] = scaled_bounds.max[:, :1]
        self.weights = values[:, 0].copy()
        self.update_count += 1
        return {
            "ocp_cost_updated": True,
            "integration": "fixed_parameter_bounds_and_acados_constant_states",
            "parameter_key": FATIGUE_WEIGHT_PARAMETER_KEY,
            "weights": self.weights.tolist(),
            "attached_nlp_count": len(self._attached_nlps),
            "allow_zero": self.allow_zero,
            "update_count": self.update_count,
        }

    def summary(self) -> dict:
        return {
            "parameter_key": FATIGUE_WEIGHT_PARAMETER_KEY,
            "parameter_size": self.size,
            "weights": self.weights.tolist(),
            "parameter_update_count": self.update_count,
            "attached_nlp_count": len(self._attached_nlps),
            "objective_graph_rebuild_required": False,
        }
