"""Audited Ding capacity coordinates from a prepared one-cycle RHO archive.

``A/a_scale`` is a dimensionless ratio of the current force-generation scale
to that muscle's rested scale (both in N/s). It is not a scalar fatigue state:
Tau1, Km, calcium, force, mechanics and stimulation history remain in the
checkpoint used by the full-NLP reserve probe. The source value is the fixed
initial bound of the *next* prepared RHO window, not a later warm-start node.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from cocofest.optimization.task_reserve import TaskReserveCheckpoint
from cocofest.optimization.task_reserve_ocp import TaskReserveStateCoordinate
from cocofest.simulation.rho_restart_checkpoint import verify_prepared_checkpoint


COORDINATE_DEFINITION = "ding2007_resting_A_capacity_ratio_v1"
COORDINATE_UNITS = "dimensionless"


@dataclass(frozen=True)
class PreparedCapacityCoordinates:
    values: tuple[float, ...]
    coordinate_layout: tuple[TaskReserveStateCoordinate, ...]
    prepared_problem_sha256: str
    model_sha256: str
    task_context_sha256: str
    completed_cycles: int
    definition: str = COORDINATE_DEFINITION
    units: str = COORDINATE_UNITS


def _model_a_scales(model_path: Path) -> dict[str, float]:
    document = json.loads(Path(model_path).read_text(encoding="utf-8"))
    muscles = document.get("muscles") if isinstance(document, dict) else None
    if (not isinstance(document, dict) or document.get("schema_version") != 1
            or not isinstance(muscles, dict) or not muscles):
        raise ValueError("Ding model must declare schema_version 1 and a nonempty muscles mapping")
    scales = {}
    for name, parameters in muscles.items():
        if (not isinstance(name, str) or not name or name.startswith("_") or "__" in name
                or not isinstance(parameters, dict)):
            raise ValueError("Ding model has an invalid muscle declaration")
        scale = parameters.get("a_scale")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Ding muscle {name} needs a finite positive a_scale in N/s")
        scales[name] = float(scale)
    return scales


def ding_a_capacity_layout(model_path: Path, *, muscle_names: Sequence[str]) -> tuple[TaskReserveStateCoordinate, ...]:
    """Build an explicit, ordered graph layout from named muscles and rested A.

    All muscles must be named by the caller. Sorting or silently omitting a
    muscle would change the meaning of fitted coefficients, so neither occurs.
    """
    scales = _model_a_scales(model_path)
    names = tuple(muscle_names)
    if (not names or len(set(names)) != len(names) or set(names) != set(scales)
            or any(not isinstance(name, str) for name in names)):
        raise ValueError("muscle_names must explicitly list every model muscle exactly once")
    return tuple(TaskReserveStateCoordinate(f"A_{name}", index=0, scale=scales[name], offset=0.)
                 for name in names)


def capacity_coordinate_context(layout: Sequence[TaskReserveStateCoordinate]) -> dict:
    """Fields that must be pinned in the probe and objective task contexts."""
    coordinates = tuple(layout)
    if not coordinates or any(not isinstance(value, TaskReserveStateCoordinate) for value in coordinates):
        raise ValueError("An explicit capacity-coordinate layout is required")
    return {
        "coordinate_definition": COORDINATE_DEFINITION,
        "coordinate_units": COORDINATE_UNITS,
        "coordinate_layout": [asdict(value) for value in coordinates],
    }


def extract_prepared_capacity_coordinates(
    checkpoint: TaskReserveCheckpoint,
    *,
    layout: Sequence[TaskReserveStateCoordinate],
) -> PreparedCapacityCoordinates:
    """Extract fixed source A/a_scale, rejecting implicit layouts and drift.

    This function does not interpret arbitrary normalized states as Ding
    capacity. The model's ``a_scale``, task-context layout, prepared-problem
    digest, initial equality bounds and shifted initial guess must all agree.
    """
    if not isinstance(checkpoint, TaskReserveCheckpoint):
        raise TypeError("checkpoint must be a TaskReserveCheckpoint")
    coordinates = tuple(layout)
    checkpoint.verify_files()
    expected_context = capacity_coordinate_context(coordinates)
    actual_context = json.loads(checkpoint.task_context_json)
    if any(actual_context.get(key) != value for key, value in expected_context.items()):
        raise ValueError("Task context does not declare the exact capacity-coordinate layout and units")
    scales = _model_a_scales(Path(checkpoint.model_path))
    names = []
    for coordinate in coordinates:
        if not coordinate.state_key.startswith("A_"):
            raise ValueError("Only Ding A/a_scale capacity coordinates are supported")
        name = coordinate.state_key[2:]
        names.append(name)
        if (name not in scales or coordinate.index != 0 or coordinate.offset != 0.
                or coordinate.scale != scales[name]):
            raise ValueError(f"Capacity coordinate {coordinate.state_key} disagrees with model a_scale")
    if len(set(names)) != len(names) or set(names) != set(scales):
        raise ValueError("Capacity layout must include each model muscle exactly once")

    verified = verify_prepared_checkpoint(Path(checkpoint.archive_path),
                                          completed_cycles=checkpoint.completed_cycles)
    if verified["prepared_problem_sha256"] != checkpoint.prepared_problem_sha256:
        raise ValueError("Prepared RHO digest differs from checkpoint provenance")
    if verified["metadata"].get("prepared_stimulation_history_complete") is not True:
        raise ValueError("Prepared checkpoint lacks complete stimulation history")
    values = []
    with np.load(checkpoint.archive_path, allow_pickle=False) as archive:
        for coordinate in coordinates:
            key = coordinate.state_key
            muscle = key[2:]
            if any(f"states__{prefix}_{muscle}" not in archive.files
                   for prefix in ("Cn", "F", "A", "Tau1", "Km")):
                raise ValueError(f"Prepared checkpoint lacks complete Ding states for {muscle}")
            try:
                initial = np.asarray(archive[f"states__{key}"], dtype=float)
                lower = np.asarray(archive[f"problem__x_bounds:{key}:min"], dtype=float)
                upper = np.asarray(archive[f"problem__x_bounds:{key}:max"], dtype=float)
            except KeyError as exc:
                raise ValueError(f"Prepared checkpoint lacks state or initial bound for {key}") from exc
            if (initial.ndim != 2 or initial.shape[0] != 1 or initial.shape[1] < 2
                    or lower.shape != (1, 3) or upper.shape != (1, 3)
                    or not np.all(np.isfinite(initial)) or not np.all(np.isfinite(lower))
                    or not np.all(np.isfinite(upper))):
                raise ValueError(f"Prepared state/bounds have invalid shape or nonfinite values for {key}")
            source = float(lower[0, 0])
            if source != float(upper[0, 0]) or not np.isclose(source, initial[0, 0], rtol=0, atol=1e-9):
                raise ValueError(f"Prepared initial {key} must agree with its fixed source-state bounds")
            ratio = source / coordinate.scale
            if ratio < -1e-10 or ratio > 1 + 1e-10:
                raise ValueError(f"Prepared {key}/a_scale is outside the Ding physical bounds")
            values.append(ratio)
    checkpoint.verify_files()
    return PreparedCapacityCoordinates(tuple(values), coordinates, checkpoint.prepared_problem_sha256,
                                       checkpoint.model_sha256, checkpoint.task_context_sha256,
                                       checkpoint.completed_cycles)
