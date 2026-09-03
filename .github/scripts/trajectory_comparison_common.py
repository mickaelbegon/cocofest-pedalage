#!/usr/bin/env python3
"""Shared helpers for comparing exported cycling benchmark trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


MUSCLES = ("Biceps", "Triceps", "Delt_ant", "Delt_post")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_path(raw_path: str | Path, report_path: Path) -> Path:
    """Resolve an artifact path relative to its report when possible."""

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    relative_to_report = report_path.parent / path
    return relative_to_report if relative_to_report.exists() else path.resolve()


def load_trajectory(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load numerical arrays and JSON metadata from a benchmark NPZ."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if "metadata__json" not in archive.files:
            raise ValueError(f"{path} has no metadata__json entry.")
        metadata = json.loads(str(archive["metadata__json"].item()))
        arrays = {
            key: np.asarray(archive[key], dtype=float)
            for key in archive.files
            if key != "metadata__json"
        }
    return arrays, metadata


def cycle_blocks(
    values: np.ndarray, *, exported_cycles: int, cycles: int, state: bool
) -> np.ndarray:
    """Split a concatenated trajectory into complete per-cycle blocks."""

    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if exported_cycles < cycles:
        raise ValueError(
            f"Only {exported_cycles} cycles are available; {cycles} were requested."
        )
    sample_count = values.shape[-1] - (1 if state else 0)
    samples_per_cycle, remainder = divmod(sample_count, exported_cycles)
    if remainder or samples_per_cycle < 1:
        kind = "state" if state else "control"
        raise ValueError(f"The {kind} trajectory cannot be split into complete cycles.")
    if state:
        return np.stack(
            [
                values[..., cycle * samples_per_cycle : (cycle + 1) * samples_per_cycle + 1]
                for cycle in range(cycles)
            ]
        )
    trimmed = values[..., : cycles * samples_per_cycle]
    reshaped = trimmed.reshape(*values.shape[:-1], cycles, samples_per_cycle)
    return np.moveaxis(reshaped, -2, 0)


def resample_blocks(blocks: np.ndarray, samples: int) -> np.ndarray:
    """Interpolate cycle blocks on a shared normalized phase grid."""

    if blocks.shape[-1] == samples:
        return blocks
    source_phase = np.linspace(0.0, 1.0, blocks.shape[-1])
    target_phase = np.linspace(0.0, 1.0, samples)
    flattened = blocks.reshape(-1, blocks.shape[-1])
    interpolated = np.vstack(
        [np.interp(target_phase, source_phase, row) for row in flattened]
    )
    return interpolated.reshape(*blocks.shape[:-1], samples)


def aligned_blocks(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    left_cycles: int,
    right_cycles: int,
    cycles: int,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two variables aligned cycle-by-cycle on the denser phase grid."""

    state = key.startswith("states__")
    left_blocks = cycle_blocks(
        left[key], exported_cycles=left_cycles, cycles=cycles, state=state
    )
    right_blocks = cycle_blocks(
        right[key], exported_cycles=right_cycles, cycles=cycles, state=state
    )
    samples = max(left_blocks.shape[-1], right_blocks.shape[-1])
    return resample_blocks(left_blocks, samples), resample_blocks(right_blocks, samples)


def align_many_blocks(
    trajectories: dict[str, dict[str, np.ndarray]],
    exported_cycles: dict[str, int],
    cycles: int,
    key: str,
) -> dict[str, np.ndarray]:
    """Align one variable from several trajectories on a common phase grid."""

    state = key.startswith("states__")
    blocks = {
        name: cycle_blocks(
            arrays[key], exported_cycles=exported_cycles[name], cycles=cycles, state=state
        )
        for name, arrays in trajectories.items()
    }
    samples = max(values.shape[-1] for values in blocks.values())
    return {name: resample_blocks(values, samples) for name, values in blocks.items()}


def capacity_scale(metadata: dict, muscle: str) -> float:
    scales = metadata.get("fatigue_capacity_scales") or {}
    value = scales.get(f"A_{muscle}")
    if value is None:
        raise ValueError(f"Missing fatigue capacity scale for {muscle}.")
    return float(value)


def state_boundaries(values: np.ndarray, exported_cycles: int, cycles: int) -> np.ndarray:
    blocks = cycle_blocks(
        values, exported_cycles=exported_cycles, cycles=cycles, state=True
    )
    initial = blocks[0, ..., 0].reshape(-1)
    terminals = blocks[..., -1].reshape(cycles, -1)
    return np.concatenate(([float(np.mean(initial))], np.mean(terminals, axis=1)))


def error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Compute finite aggregate errors, normalized against the reference scale."""

    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    delta = candidate - reference
    reference_span = float(np.ptp(reference))
    reference_rms = float(np.sqrt(np.mean(reference**2)))
    scale = max(reference_span, reference_rms, np.finfo(float).eps)
    correlation = None
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    if np.std(reference_flat) > 0.0 and np.std(candidate_flat) > 0.0:
        correlation = float(np.corrcoef(reference_flat, candidate_flat)[0, 1])
    return {
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mae": float(np.mean(np.abs(delta))),
        "maximum_absolute_error": float(np.max(np.abs(delta))),
        "normalized_rmse": float(np.sqrt(np.mean(delta**2)) / scale),
        "reference_scale": scale,
        "correlation": correlation,
    }


def style_axis(axis, *, xlabel: str = "Cycle") -> None:
    axis.set_xlabel(xlabel)
    axis.grid(alpha=0.2)


def short_variable_name(key: str) -> str:
    return (
        key.replace("states__", "")
        .replace("controls__last_pulse_width_", "PW_")
        .replace("wheel_angle", "angle roue")
        .replace("wheel_speed", "vitesse roue")
    )
