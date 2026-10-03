"""Physical source-coordinate contract for the experimental reserve fit."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.task_reserve import TaskReserveCheckpoint
from cocofest.optimization.task_reserve_coordinates import (
    COORDINATE_DEFINITION, capacity_coordinate_context, ding_a_capacity_layout,
    extract_prepared_capacity_coordinates,
)
from cocofest.optimization.task_reserve_ocp import TaskReserveStateCoordinate
from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint


def _source(tmp_path, *, fixed=True):
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"schema_version": 1, "muscles": {
        "Biceps": {"a_scale": 100.}, "Triceps": {"a_scale": 200.},
    }}), encoding="utf-8")
    states, bounds, controls, control_bounds = {}, {}, {}, {}
    for muscle, scale, a in (("Biceps", 100., 80.), ("Triceps", 200., 100.)):
        for prefix in ("Cn", "F", "A", "Tau1", "Km"):
            key = f"{prefix}_{muscle}"
            value = a if prefix == "A" else .1
            states[key] = SimpleNamespace(init=np.array([[value, value, value]]))
            bounds[key] = SimpleNamespace(
                min=np.array([[value, 0., 0.]]),
                max=np.array([[value if fixed or prefix != "A" else value + 1., scale, scale]]),
            )
        key = f"last_pulse_width_{muscle}"
        controls[key] = SimpleNamespace(init=np.array([[.0002]]))
        control_bounds[key] = SimpleNamespace(min=np.array([[.0001]]), max=np.array([[.0005]]))
    program = SimpleNamespace(nlp=[SimpleNamespace(x_init=states, u_init=controls,
        x_bounds=bounds, u_bounds=control_bounds)], parameter_init={}, parameter_bounds={})
    archive = tmp_path / "prepared.npz"
    export_prepared_checkpoint(archive, program, completed_cycles=20, model_path=model)
    layout = ding_a_capacity_layout(model, muscle_names=("Biceps", "Triceps"))
    context = {"side": "right", "nominal_work_j": 4., **capacity_coordinate_context(layout)}
    checkpoint = TaskReserveCheckpoint.from_archive(archive, completed_cycles=20,
                                                      model_path=model, task_context=context)
    return checkpoint, layout


def test_extracts_explicit_dimensionless_resting_a_ratios_from_fixed_source(tmp_path):
    checkpoint, layout = _source(tmp_path)
    observation = extract_prepared_capacity_coordinates(checkpoint, layout=layout)
    assert observation.values == pytest.approx((.8, .5))
    assert observation.units == "dimensionless"
    assert observation.definition == COORDINATE_DEFINITION
    assert observation.prepared_problem_sha256 == checkpoint.prepared_problem_sha256
    assert observation.task_context_sha256 == checkpoint.task_context_sha256


def test_refuses_implicit_or_different_normalization(tmp_path):
    checkpoint, layout = _source(tmp_path)
    no_layout = json.loads(checkpoint.task_context_json)
    no_layout.pop("coordinate_layout")
    implicit = replace(checkpoint, task_context_json=json.dumps(no_layout))
    with pytest.raises(ValueError, match="context digest"):
        extract_prepared_capacity_coordinates(implicit, layout=layout)
    with pytest.raises(ValueError, match="coordinate layout and units"):
        extract_prepared_capacity_coordinates(checkpoint, layout=layout[::-1])
    wrong = (TaskReserveStateCoordinate("A_Biceps", scale=80.), layout[1])
    with pytest.raises(ValueError, match="coordinate layout and units"):
        extract_prepared_capacity_coordinates(checkpoint, layout=wrong)
    wrong_context = {"side": "right", "nominal_work_j": 4., **capacity_coordinate_context(wrong)}
    wrong_checkpoint = TaskReserveCheckpoint.from_archive(Path(checkpoint.archive_path),
        completed_cycles=20, model_path=Path(checkpoint.model_path), task_context=wrong_context)
    with pytest.raises(ValueError, match="disagrees with model a_scale"):
        extract_prepared_capacity_coordinates(wrong_checkpoint, layout=wrong)


def test_refuses_unfixed_prepared_state_or_changed_model_bytes(tmp_path):
    checkpoint, layout = _source(tmp_path, fixed=False)
    with pytest.raises(ValueError, match="fixed source-state bounds"):
        extract_prepared_capacity_coordinates(checkpoint, layout=layout)
    model = Path(checkpoint.model_path)
    model.write_text(model.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model changed"):
        extract_prepared_capacity_coordinates(checkpoint, layout=layout)


def test_requires_all_muscles_in_explicit_order(tmp_path):
    checkpoint, layout = _source(tmp_path)
    with pytest.raises(ValueError, match="every model muscle"):
        ding_a_capacity_layout(checkpoint.model_path, muscle_names=("Biceps",))
    reversed_layout = ding_a_capacity_layout(checkpoint.model_path,
                                             muscle_names=("Triceps", "Biceps"))
    assert reversed_layout == layout[::-1]
