"""Experimental fixed-graph terminal cost for an audited local task reserve.

The binding starts disabled. A supervisor must supply an accepted local fit,
check its age and trust box before each RHO solve, and check the *solved terminal
point* before accepting that decision. The affine model is not a feasibility
certificate and its domain is not an extra physical constraint on the RHO.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .parametric_fatigue_weights import FATIGUE_WEIGHT_PARAMETER_KEY, ParametricFatigueWeightBinding
from .task_reserve import LocalReserveModel, reserve_penalty_expression


TASK_RESERVE_PARAMETER_KEY = "rho_task_reserve"


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _cycle(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class TaskReserveStateCoordinate:
    """One dimensionless terminal coordinate, ``(state[index]-offset)/scale``."""

    state_key: str
    index: int = 0
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self):
        if not isinstance(self.state_key, str) or not self.state_key:
            raise ValueError("state_key must be a nonempty string")
        _cycle(self.index, "coordinate index")
        if not math.isfinite(self.scale) or self.scale <= 0 or not math.isfinite(self.offset):
            raise ValueError("Coordinate scale must be finite positive and offset finite")


class TaskReserveObjectiveBinding:
    """Fixed parameters ``[weight, intercept, center(n), gradient(n)]``.

    ``weight`` is the actual multiplier of the smooth squared reserve shortage,
    with no implicit factor 10000. Zero disables this cost exactly. Target,
    smoothing and coordinate normalization are graph structure, so changing
    them requires a new binding and compiled problem.
    """

    def __init__(self, coordinates: Sequence[TaskReserveStateCoordinate], *,
                 task_context: Mapping, model_sha256: str, target: float,
                 smoothing: float = 1e-3, maximum_age_cycles: int = 20):
        self.coordinates = tuple(coordinates)
        if (not self.coordinates or any(not isinstance(c, TaskReserveStateCoordinate)
                                       for c in self.coordinates)
                or len({(c.state_key, c.index) for c in self.coordinates}) != len(self.coordinates)):
            raise ValueError("Distinct explicit task-reserve state coordinates are required")
        self.task_context = json.loads(json.dumps(dict(task_context), allow_nan=False))
        if self.task_context.get("coordinate_layout") != [asdict(c) for c in self.coordinates]:
            raise ValueError("task_context coordinate_layout must exactly describe the normalized coordinates")
        if (not isinstance(model_sha256, str) or len(model_sha256) != 64
                or any(c not in "0123456789abcdef" for c in model_sha256)):
            raise ValueError("model_sha256 must be a lowercase SHA-256 digest")
        self.task_context_sha256 = _hash(self.task_context)
        self.model_sha256 = model_sha256
        self.target, self.smoothing = float(target), float(smoothing)
        if not math.isfinite(self.target) or self.target < 0:
            raise ValueError("target reserve must be finite and nonnegative")
        if not math.isfinite(self.smoothing) or self.smoothing <= 0:
            raise ValueError("smoothing must be finite and positive")
        self.maximum_age_cycles = _cycle(maximum_age_cycles, "maximum_age_cycles")
        self.values = np.zeros(2 + 2 * len(self.coordinates))
        self.graph_signature_sha256 = _hash({
            "version": "task-reserve-terminal-shortage-v1-experimental",
            "coordinate_layout": [asdict(c) for c in self.coordinates],
            "target": self.target, "smoothing": self.smoothing,
            "parameter_key": TASK_RESERVE_PARAMETER_KEY,
            "parameter_size": self.values.size,
        })
        self.update_count = 0
        self._nlp = None
        self.local_model = None
        self.source_completed_cycles = None

    @property
    def dimension(self):
        return len(self.coordinates)

    def validate_build_context(self, *, model_path, **actual_context):
        """Check the source file and the actual RHO configuration at graph build."""
        if model_path is None:
            raise ValueError("task_reserve_model_path is required to verify source-model provenance")
        if sha256(Path(model_path).read_bytes()).hexdigest() != self.model_sha256:
            raise ValueError("Task-reserve model digest differs from the RHO source model")
        for key, value in actual_context.items():
            if key not in self.task_context or self.task_context[key] != value:
                raise ValueError(f"Task-reserve context does not match the RHO {key}")

    def parameter_options(self, *, use_sx=True):
        return task_reserve_parameter_options(self, use_sx=use_sx)

    def attach(self, nmpc):
        if len(nmpc.nlp) != 1:
            raise ValueError("Experimental task reserve requires one RHO phase")
        if self._nlp is not None and self._nlp is not nmpc.nlp[0]:
            raise RuntimeError("Task-reserve binding cannot be attached to a rebuilt NLP")
        self._nlp = nmpc.nlp[0]

    def objective(self, controller):
        from casadi import sqrt, vertcat

        coordinates = []
        for coordinate in self.coordinates:
            state = controller.states[coordinate.state_key].cx
            if coordinate.index >= int(state.numel()):
                raise ValueError(f"Task-reserve coordinate index outside state {coordinate.state_key}")
            coordinates.append((state[coordinate.index] - coordinate.offset) / coordinate.scale)
        parameters = controller.parameters[TASK_RESERVE_PARAMETER_KEY].cx
        if int(parameters.numel()) != self.values.size:
            raise ValueError("Task-reserve parameter vector has the wrong dimension")
        return parameters[0] * reserve_penalty_expression(
            vertcat(*coordinates), parameters[1:], dimension=self.dimension,
            target=self.target, smoothing=self.smoothing, sqrt=sqrt,
        )

    def _validate_fit(self, model, *, source_completed_cycles, completed_cycles,
                      evaluation_coordinates):
        if not isinstance(model, LocalReserveModel):
            raise TypeError("Task reserve requires a LocalReserveModel")
        source = _cycle(source_completed_cycles, "source_completed_cycles")
        current = _cycle(completed_cycles, "completed_cycles")
        if current < source or current - source > self.maximum_age_cycles:
            raise ValueError("Local task reserve is stale or belongs to a future cycle")
        if (model.task_context_sha256 != self.task_context_sha256
                or model.model_sha256 != self.model_sha256):
            raise ValueError("Local task reserve belongs to a different task context or model")
        parameters = np.asarray(model.parameter_vector(), dtype=float)
        if parameters.shape != (1 + 2 * self.dimension,) or not np.all(np.isfinite(parameters)):
            raise ValueError("Local task reserve has incompatible dimensions or nonfinite coefficients")
        diagnostics = (model.condition_number, model.training_maximum_error,
                       model.holdout_maximum_error, model.empirical_bias_correction)
        if (model.rank != self.dimension + 1 or model.sample_count < self.dimension + 1
                or model.holdout_count < 1 or any(not math.isfinite(v) or v < 0 for v in diagnostics)):
            raise ValueError("Local task reserve requires finite full-rank training and independent holdout diagnostics")
        trust = np.asarray(model.trust_radius, dtype=float)
        if trust.shape != (self.dimension,) or not np.all(np.isfinite(trust)) or np.any(trust <= 0):
            raise ValueError("Local task reserve has invalid trust radii")
        model.margin(evaluation_coordinates)
        return parameters

    def validate_terminal_point(self, coordinates, *, completed_cycles):
        """Required after a solve: refuse extrapolated/stale terminal decisions."""
        if self.local_model is None or self.values[0] == 0:
            return {"active": False, "terminal_trust_validated": False}
        self._validate_fit(self.local_model, source_completed_cycles=self.source_completed_cycles,
                           completed_cycles=completed_cycles, evaluation_coordinates=coordinates)
        return {"active": True, "terminal_trust_validated": True,
                "observed_reserve_model_value": self.local_model.margin(coordinates)}

    def _write(self, nmpc, values):
        self.attach(nmpc)
        from bioptim import InitialGuessList

        bound = nmpc.parameter_bounds[TASK_RESERVE_PARAMETER_KEY]
        column = np.asarray(values, dtype=float).reshape((-1, 1))
        if bound.min.shape != column.shape or bound.max.shape != column.shape:
            raise ValueError("Task-reserve parameter bounds have an unexpected shape")
        initial = InitialGuessList()
        initial.add(TASK_RESERVE_PARAMETER_KEY, initial_guess=column.copy())
        # Update the initial guess first: a failed API call must not alter the
        # equality bounds of the previously valid policy.
        nmpc.update_initial_guess(parameter_init=initial)
        bound.min[...] = column
        bound.max[...] = column
        self.values = column[:, 0].copy()
        self.update_count += 1

    def update(self, nmpc, model: LocalReserveModel, *, weight: float,
               source_completed_cycles: int, completed_cycles: int,
               evaluation_coordinates: Sequence[float]):
        weight = float(weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("An accepted task-reserve update needs a finite positive weight; use deactivate for zero")
        parameters = self._validate_fit(model, source_completed_cycles=source_completed_cycles,
                                       completed_cycles=completed_cycles,
                                       evaluation_coordinates=evaluation_coordinates)
        self._write(nmpc, np.r_[weight, parameters])
        self.local_model = model
        self.source_completed_cycles = source_completed_cycles
        return self.summary()

    def deactivate(self, nmpc):
        values = self.values.copy()
        values[0] = 0.
        self._write(nmpc, values)
        return self.summary()

    def summary(self):
        return {"parameter_key": TASK_RESERVE_PARAMETER_KEY,
                "parameter_size": self.values.size, "weight": float(self.values[0]),
                "parameter_update_count": self.update_count,
                "objective_graph_rebuild_required": False,
                "experimental": True, "globally_certified": False,
                "requires_terminal_trust_validation": True,
                "graph_signature_sha256": self.graph_signature_sha256,
                "task_context_sha256": self.task_context_sha256,
                "model_sha256": self.model_sha256,
                "source_completed_cycles": self.source_completed_cycles,
                "maximum_age_cycles": self.maximum_age_cycles}


def task_reserve_parameter_options(binding, *, fatigue_weight_binding=None, use_sx=True):
    """Merge the explicitly supported reserve+fatigue pair into one ParameterList.

    Both objectives have independent keys, fixed equality bounds and unit
    scaling. Other objective bindings are deliberately not merged here.
    """
    from bioptim import BoundsList, InitialGuessList, InterpolationType, ParameterList, VariableScaling

    if not isinstance(binding, TaskReserveObjectiveBinding):
        raise TypeError("binding must be a TaskReserveObjectiveBinding")
    entries = [(TASK_RESERVE_PARAMETER_KEY, binding.values)]
    if fatigue_weight_binding is not None:
        if not isinstance(fatigue_weight_binding, ParametricFatigueWeightBinding):
            raise TypeError("Only ParametricFatigueWeightBinding can coexist with task reserve")
        entries.append((FATIGUE_WEIGHT_PARAMETER_KEY, fatigue_weight_binding.weights))
    parameters, bounds, initial = ParameterList(use_sx=use_sx), BoundsList(), InitialGuessList()
    for key, vector in entries:
        column = np.asarray(vector, dtype=float).reshape((-1, 1))
        parameters.add(name=key, function=None, size=column.size,
                       scaling=VariableScaling(key, np.ones(column.size)))
        bounds.add(key, min_bound=column.copy(), max_bound=column.copy(),
                   interpolation=InterpolationType.CONSTANT)
        initial.add(key, initial_guess=column.copy())
    return {"parameters": parameters, "parameter_bounds": bounds, "parameter_init": initial}
