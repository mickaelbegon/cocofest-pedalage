"""Versioned physical trajectory contract shared by FHO and concatenated RHO.

The existing states__/controls__ keys remain the seed format. Additional
physical arrays describe the replay initial value problem without guessing
defaults from the installed model. Internal warmup caches remain unversioned.
"""

from hashlib import sha256
import json

import numpy as np

from cocofest.optimization.configured_cycling_model import FIELD_ATTRIBUTES

ARCHIVE_SCHEMA = "cocofest-physical-solution-v1"
COMPONENTS = ("Cn", "F", "A", "Tau1", "Km")


def model_parameter_metadata(model):
    """Capture effective parameters from the constructed model, including overrides."""
    parameters = {
        muscle.muscle_name: {
            name: float(getattr(muscle, attribute))
            for name, attribute in FIELD_ATTRIBUTES.items()
        }
        for muscle in model.muscles_dynamics_model
    }
    encoded = json.dumps({"schema_version": 1, "muscles": parameters},
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"configured_muscle_parameters": parameters,
            "muscle_parameter_fingerprint": sha256(encoded.encode()).hexdigest()}


def physical_archive_arrays(states, controls, metadata):
    """Add explicit shooting times and entering physical state to a supported export.

    Dense collocation state arrays are retained unchanged; times here describe
    shooting nodes only. Unsupported transcriptions stay explicitly legacy.
    """
    if not (metadata.get("model_formulation") == "periodic_node"
            and metadata.get("mechanical_formulation") == "reduced"
            and not metadata.get("bilateral_reduced", False)
            and metadata.get("configured_muscle_parameters")
            and metadata.get("cycle_duration_s") is not None):
        return {}, dict(metadata)
    muscles = tuple(sorted(metadata["configured_muscle_parameters"]))
    names = tuple(f"{part}_{muscle}" for muscle in muscles for part in COMPONENTS) + ("theta", "omega")
    if not all(name in states for name in names):
        return {}, dict(metadata, physical_archive_unavailable_reason="physical states not reconstructed")
    # The rate-state formulation needs its own continuous-control replay adapter.
    if not all(f"last_pulse_width_{m}" in controls for m in muscles):
        return {}, dict(metadata, physical_archive_unavailable_reason="unsupported control representation")
    command_rows = [np.asarray(controls[f"last_pulse_width_{m}"]).reshape(-1) for m in muscles]
    intervals = len(command_rows[0])
    count = int(metadata["stimulations_per_cycle"])
    duration = float(metadata["cycle_duration_s"])
    if count < 1 or intervals < 1 or intervals % count or not np.isfinite(duration) or duration <= 0:
        raise ValueError("Physical archive requires positive duration and complete stimulation cycles")
    if any(len(row) != intervals or not np.isfinite(row).all() for row in command_rows):
        raise ValueError("Physical archive controls must be finite with a common length")
    rows = [np.asarray(states[name]).reshape(-1) for name in names]
    stride, remainder = divmod(len(rows[0]) - 1, intervals)
    if remainder or stride < 1 or any(len(row) != intervals * stride + 1 for row in rows):
        raise ValueError("Physical archive states must share a shooting-node layout")
    if any(not np.isfinite(row).all() for row in rows):
        raise ValueError("Physical archive states must be finite")
    if stride != 1 and (metadata.get("producer_collocation_method") != "radau"
                        or stride != int(metadata.get("producer_collocation_degree") or 0) + 1):
        raise ValueError("Physical archive dense states require declared Radau degree")
    updated = dict(metadata, physical_archive_schema=ARCHIVE_SCHEMA,
                   physical_state_names=list(names), physical_state_stride=stride,
                   physical_time_origin="archive_start", physical_control_interpolation="zero_order_hold",
                   physical_cycles=intervals // count)
    return {"physical__initial_state": np.array([row[0] for row in rows]),
            "physical__shooting_time_s": np.arange(intervals + 1) * duration / count}, updated


def validate_physical_archive(data, metadata):
    """Validate new contracts strictly; absence alone selects the legacy reader."""
    schema = metadata.get("physical_archive_schema")
    if schema is None:
        return
    if schema != ARCHIVE_SCHEMA:
        raise ValueError(f"Unsupported physical archive schema: {schema!r}")
    required = ("configured_muscle_parameters", "muscle_parameter_fingerprint", "cycle_duration_s",
                "physical_state_names", "physical_state_stride", "producer_solver", "formulation")
    for key in required:
        if metadata.get(key) is None:
            raise ValueError(f"Physical archive missing {key}")
    states = {key[8:]: data[key] for key in data.files if key.startswith("states__")}
    controls = {key[10:]: data[key] for key in data.files if key.startswith("controls__")}
    expected, reconstructed = physical_archive_arrays(states, controls, metadata)
    if not expected:
        raise ValueError("Declared physical archive has an unsupported model/control layout")
    for key in ("physical_state_names", "physical_state_stride", "physical_cycles",
                "physical_time_origin", "physical_control_interpolation"):
        if metadata.get(key) != reconstructed[key]:
            raise ValueError(f"Physical archive inconsistent {key}")
    for key, value in expected.items():
        if key not in data or np.asarray(data[key]).shape != value.shape or not np.allclose(data[key], value, rtol=1e-12, atol=1e-14):
            raise ValueError(f"Physical archive inconsistent {key}")
