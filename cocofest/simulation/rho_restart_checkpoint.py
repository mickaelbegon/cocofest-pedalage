"""Portable export of an already-prepared one-cycle RHO problem.

This module deliberately distinguishes two things which are often conflated:
an exact *serialization* of the live next RHO window, and an exact replay in a
new solver process.  The former is implemented here.  The latter must rebuild
the same model and restore this archive before an experiment can call two
continuations causally matched.

The archive contains physical (unscaled) initial guesses, active bounds and
fixed parameters.  It is independent of Bioptim and CasADi objects so it can
be inspected before a costly solver rebuild.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from cocofest.optimization.receding_horizon_initial_guess import (
    initial_guess_signature,
    snapshot_initial_guess,
)


def prepared_problem_arrays(program) -> dict[str, np.ndarray]:
    """Copy every numerical quantity defining the next prepared RHO window."""
    nlp = program.nlp[0]
    arrays: dict[str, np.ndarray] = {}
    for prefix, container in (
        ("x_bounds", nlp.x_bounds),
        ("u_bounds", nlp.u_bounds),
        ("parameter_bounds", getattr(program, "parameter_bounds", {})),
    ):
        for key in container.keys():
            for field in ("min", "max"):
                arrays[f"{prefix}:{key}:{field}"] = np.asarray(
                    getattr(container[key], field), dtype=float
                ).copy()
    for key in getattr(program, "parameter_init", {}).keys():
        arrays[f"parameter_init:{key}"] = np.asarray(
            program.parameter_init[key].init, dtype=float
        ).copy()
    return arrays


def prepared_problem_digest(snapshot: dict[str, dict[str, np.ndarray]],
                            arrays: dict[str, np.ndarray], runtime_state: dict[str, int | float] | None = None) -> str:
    """Return a content digest for a shifted primal plus active numerical data."""
    digest = sha256()
    digest.update(b"cocofest-prepared-rho-v1")
    for category in ("states", "controls"):
        for key in sorted(snapshot[category]):
            _digest_array(digest, f"{category}:{key}", snapshot[category][key])
    for key in sorted(arrays):
        _digest_array(digest, f"problem:{key}", arrays[key])
    for key, value in sorted((runtime_state or {}).items()):
        digest.update(f"runtime:{key}".encode("utf-8"))
        digest.update(json.dumps(value, allow_nan=False, separators=(",", ":")).encode("ascii"))
    return digest.hexdigest()


def export_prepared_checkpoint(path: Path, program, *, completed_cycles: int,
                               model_path: Path | None = None) -> dict[str, Any]:
    """Atomically persist a prepared window and verify its serialized content.

    This function does *not* mark the checkpoint as a fresh-worker replay.
    That claim requires rebuilding the problem in a separate process and is
    intentionally made by a later validation step only.
    """
    if type(completed_cycles) is not int or completed_cycles < 0:
        raise ValueError("completed_cycles must be a non-negative integer")
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite prepared RHO checkpoint: {path}")
    snapshot = snapshot_initial_guess(program)
    if not snapshot["states"] or not snapshot["controls"]:
        raise RuntimeError("Prepared RHO checkpoint requires state and control initial guesses")
    arrays = prepared_problem_arrays(program)
    signature = initial_guess_signature(snapshot)
    runtime_state = _runtime_state(program)
    digest = prepared_problem_digest(snapshot, arrays, runtime_state)
    stimulation_keys = sorted(
        key for key in snapshot["controls"] if key.startswith("last_pulse_width_")
    )
    metadata = {
        "schema_version": 1,
        "producer_mode": "rho_replay_checkpoint",
        "producer_completed_windows": completed_cycles,
        "prepared_primal_signature": signature,
        "prepared_problem_sha256": digest,
        "runtime_state": runtime_state,
        "stimulation_history_control_keys": stimulation_keys,
        # This is complete for the prepared NLP (the state carries Ding's
        # memory); it is not a statement about a new worker having restored it.
        "prepared_stimulation_history_complete": bool(stimulation_keys),
        "fresh_worker_replay_verified": False,
    }
    payload = {
        **{f"states__{key}": value for key, value in snapshot["states"].items()},
        **{f"controls__{key}": value for key, value in snapshot["controls"].items()},
        **{f"problem__{key}": value for key, value in arrays.items()},
        "metadata__json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".prepared-", suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
        np.savez_compressed(temporary, **payload)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    verification = verify_prepared_checkpoint(path, completed_cycles=completed_cycles)
    if verification["prepared_problem_sha256"] != digest:
        raise RuntimeError("Prepared RHO checkpoint digest changed after serialization")
    model = None if model_path is None else Path(model_path).resolve()
    return {
        "status": "exported",
        "primal_path": str(path),
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "completed_cycles": completed_cycles,
        "prepared_primal_signature": signature,
        "prepared_problem_sha256": digest,
        "serialization_roundtrip_exact": True,
        "fresh_worker_replay_verified": False,
        "stimulation_history_complete": metadata["prepared_stimulation_history_complete"],
        "model_path": str(model) if model is not None else None,
        "model_sha256": sha256(model.read_bytes()).hexdigest() if model is not None and model.is_file() else None,
    }


def verify_prepared_checkpoint(path: Path, *, completed_cycles: int) -> dict[str, Any]:
    """Verify archive bytes reconstruct the same primal signature and digest."""
    metadata, snapshot, arrays = _read_prepared_checkpoint(path, completed_cycles=completed_cycles)
    signature = initial_guess_signature(snapshot)
    digest = prepared_problem_digest(snapshot, arrays, metadata.get("runtime_state", {}))
    if signature != metadata.get("prepared_primal_signature"):
        raise ValueError("archive primal signature mismatch")
    if digest != metadata.get("prepared_problem_sha256"):
        raise ValueError("archive prepared-problem digest mismatch")
    return {"prepared_primal_signature": signature, "prepared_problem_sha256": digest,
            "metadata": metadata}


def restore_prepared_checkpoint(path: Path, program, *, completed_cycles: int) -> dict[str, Any]:
    """Restore a portable checkpoint into an already rebuilt compatible NLP.

    The caller is responsible for creating ``program`` in a new worker from
    the model/configuration named by the bilateral receipt.  This routine
    refuses missing keys or changed array shapes rather than partially
    restoring a different problem.
    """
    metadata, snapshot, arrays = _read_prepared_checkpoint(path, completed_cycles=completed_cycles)
    nlp = program.nlp[0]
    _restore_container(nlp.x_init, snapshot["states"], "state initial guess")
    _restore_container(nlp.u_init, snapshot["controls"], "control initial guess")
    bounds, parameters = _split_problem_arrays(arrays)
    for prefix, container in (("x_bounds", nlp.x_bounds), ("u_bounds", nlp.u_bounds),
                              ("parameter_bounds", getattr(program, "parameter_bounds", {}))):
        for key in container.keys():
            for field in ("min", "max"):
                name = f"{prefix}:{key}:{field}"
                if name not in bounds:
                    raise ValueError(f"checkpoint lacks {name}")
                _restore_array(getattr(container[key], field), bounds.pop(name), name)
    if bounds:
        raise ValueError(f"checkpoint contains unknown bounds: {sorted(bounds)}")
    _restore_container(getattr(program, "parameter_init", {}), parameters, "fixed parameter", attribute="init")
    _restore_runtime_state(program, metadata.get("runtime_state", {}))
    restored_snapshot = snapshot_initial_guess(program)
    restored_arrays = prepared_problem_arrays(program)
    signature = initial_guess_signature(restored_snapshot)
    digest = prepared_problem_digest(restored_snapshot, restored_arrays, _runtime_state(program))
    if signature != metadata["prepared_primal_signature"] or digest != metadata["prepared_problem_sha256"]:
        raise RuntimeError("restored RHO differs from the serialized prepared problem")
    return {"restored_primal_signature": signature, "restored_problem_sha256": digest,
            "stimulation_history_complete": metadata["prepared_stimulation_history_complete"] is True}


def _read_prepared_checkpoint(path: Path, *, completed_cycles: int):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata__json"].item()))
        if metadata.get("producer_mode") != "rho_replay_checkpoint":
            raise ValueError("archive is not a prepared RHO replay checkpoint")
        if metadata.get("producer_completed_windows") != completed_cycles:
            raise ValueError("archive source cycle differs from requested checkpoint")
        snapshot = {
            "states": {key.split("__", 1)[1]: np.asarray(archive[key]) for key in archive.files
                       if key.startswith("states__")},
            "controls": {key.split("__", 1)[1]: np.asarray(archive[key]) for key in archive.files
                         if key.startswith("controls__")},
        }
        arrays = {key.split("__", 1)[1]: np.asarray(archive[key]) for key in archive.files
                  if key.startswith("problem__")}
    return metadata, snapshot, arrays


def _split_problem_arrays(arrays: dict[str, np.ndarray]):
    bounds, parameters = {}, {}
    for name, values in arrays.items():
        if name.startswith("parameter_init:"):
            parameters[name.split(":", 1)[1]] = values
        else:
            bounds[name] = values
    return bounds, parameters


def _restore_container(container, values: dict[str, np.ndarray], label: str, *, attribute="init") -> None:
    expected, provided = set(container.keys()), set(values)
    if expected != provided:
        raise ValueError(f"checkpoint {label} keys differ: expected {sorted(expected)}, got {sorted(provided)}")
    for key in expected:
        _restore_array(getattr(container[key], attribute), values[key], f"{label}:{key}")


def _restore_array(target, source: np.ndarray, label: str) -> None:
    source = np.asarray(source, dtype=float)
    if target.shape != source.shape:
        raise ValueError(f"checkpoint {label} shape differs: expected {target.shape}, got {source.shape}")
    target[:, :] = source


_RUNTIME_STATE_FIELDS = (
    "absolute_wheel_q_cycle_index",
    "absolute_wheel_q_reference",
    "absolute_wheel_q_cycle_shift",
    "_cocofest_terminal_wheel_q_center",
)


def _runtime_state(program) -> dict[str, int | float]:
    """Persist the small absolute-clock state not represented by NLP arrays."""
    state: dict[str, int | float] = {}
    for name in _RUNTIME_STATE_FIELDS:
        value = getattr(program, name, None)
        if value is None:
            continue
        if name == "absolute_wheel_q_cycle_index":
            if type(value) is not int or value < 0:
                raise ValueError("absolute crank cycle index must be a non-negative integer")
            state[name] = value
        else:
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            state[name] = value
    return state


def _restore_runtime_state(program, state: Any) -> None:
    if not isinstance(state, dict):
        raise ValueError("checkpoint runtime state must be an object")
    unknown = set(state) - set(_RUNTIME_STATE_FIELDS)
    if unknown:
        raise ValueError(f"checkpoint contains unknown runtime state: {sorted(unknown)}")
    for name, value in state.items():
        if name == "absolute_wheel_q_cycle_index":
            if type(value) is not int or value < 0:
                raise ValueError("checkpoint absolute crank cycle index is invalid")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"checkpoint {name} is invalid")
        setattr(program, name, value)


def _digest_array(digest, name: str, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values, dtype=np.float64)
    digest.update(name.encode("utf-8"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
