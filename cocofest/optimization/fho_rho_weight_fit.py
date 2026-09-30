"""Fit local RHO fatigue weights to a certified FHO trajectory.

This module deliberately keeps the optimisation *outside* the running RHO/FHO
campaigns.  A caller supplies a function which solves one RHO window for a
candidate four-vector of weights and returns its standard ``.npz`` archive.
The resulting JSON is consequently both reproducible and suitable for a GUI:
it records every candidate, its archive, and the normalised tracking loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


MUSCLE_NAMES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
PULSE_WIDTH_LIMIT_S = 6e-4
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class TrajectoryArchive:
    """Numerical view of a standard compact RHO/FHO ``.npz`` archive."""

    states: Mapping[str, np.ndarray]
    controls: Mapping[str, np.ndarray]
    metadata: Mapping[str, object]
    path: Path

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryArchive":
        path = Path(path).expanduser().resolve()
        with np.load(path, allow_pickle=False) as data:
            states = {key.removeprefix("states__"): np.asarray(data[key], dtype=float)
                      for key in data.files if key.startswith("states__")}
            controls = {key.removeprefix("controls__"): np.asarray(data[key], dtype=float)
                        for key in data.files if key.startswith("controls__")}
            if "metadata__json" not in data:
                raise ValueError(f"Archive {path} has no metadata__json.")
            metadata = json.loads(str(data["metadata__json"]))
        if not states or not controls:
            raise ValueError(f"Archive {path} must contain states and controls.")
        return cls(states=states, controls=controls, metadata=metadata, path=path)

    def cycles(self, *, controls_per_cycle: int) -> int:
        sizes = {np.asarray(value).shape[-1] for value in self.controls.values()}
        if len(sizes) != 1:
            raise ValueError(f"Control dimensions disagree in {self.path}: {sizes}.")
        count = sizes.pop()
        cycles, remainder = divmod(count, controls_per_cycle)
        if remainder:
            raise ValueError(
                f"Control count {count} is not divisible by {controls_per_cycle} in {self.path}."
            )
        return cycles


def _matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.ndim == 1:
        return value[np.newaxis, :]
    if value.ndim != 2:
        raise ValueError(f"Trajectory values must be one or two dimensional, got {value.shape}.")
    return value


def _resample(values: np.ndarray, columns: int) -> np.ndarray:
    """Linearly sample node values on the same normalised-cycle grid."""
    values = _matrix(values)
    if values.shape[-1] == columns:
        return values
    source = np.linspace(0.0, 1.0, values.shape[-1])
    target = np.linspace(0.0, 1.0, columns)
    return np.vstack([np.interp(target, source, row) for row in values])


def _scale(values: np.ndarray, *, floor: float) -> float:
    values = np.asarray(values, dtype=float)
    p05, p95 = np.nanpercentile(values, (5.0, 95.0))
    amplitude = max(abs(float(p05)), abs(float(p95)), float(np.nanmax(np.abs(values))))
    return max(float(p95 - p05), 0.05 * amplitude, floor)


def trajectory_tracking_loss(
    reference: TrajectoryArchive,
    candidate: TrajectoryArchive,
    *,
    state_weight: float = 0.5,
    control_weight: float = 0.5,
    alignment_rtol: float = 1e-5,
    alignment_atol: float = 1e-8,
) -> dict:
    """Return equal-per-variable, normalised node and stimulation errors.

    States are scaled independently from the *reference window* by its robust
    5--95 percentile span (with a 5 % amplitude guard).  Pulse-width controls
    use their physical 0.6 ms range.  Thus neither high-capacity muscles nor
    the larger number of state rows can dominate the fit.
    """
    if state_weight < 0 or control_weight < 0 or state_weight + control_weight <= 0:
        raise ValueError("state_weight and control_weight must have a positive sum.")
    state_names = sorted(set(reference.states) & set(candidate.states))
    control_names = sorted(set(reference.controls) & set(candidate.controls))
    if not state_names or not control_names:
        raise ValueError("RHO and FHO archives must share at least one state and control.")

    # A local fit is valid only if its RHO starts at the FHO state.  Reporting
    # the diagnostic here prevents weights from compensating an initialisation
    # mismatch rather than identifying the FHO's local policy.
    start_errors = []
    for name in state_names:
        ref, got = _matrix(reference.states[name]), _matrix(candidate.states[name])
        if ref.shape[0] != got.shape[0]:
            raise ValueError(f"State row count differs for {name}: {ref.shape} vs {got.shape}.")
        start_errors.append(float(np.max(np.abs(ref[:, 0] - got[:, 0]))))
    max_start_error = max(start_errors)
    if not np.isfinite(max_start_error):
        raise ValueError("Initial-state comparison contains non-finite values.")

    state_losses, control_losses = {}, {}
    for name in state_names:
        ref = _matrix(reference.states[name])
        got = _resample(candidate.states[name], ref.shape[-1])
        if ref.shape[0] != got.shape[0]:
            raise ValueError(f"State row count differs for {name}: {ref.shape} vs {got.shape}.")
        state_losses[name] = float(np.mean(((got - ref) / _scale(ref, floor=1e-8)) ** 2))
    for name in control_names:
        ref = _matrix(reference.controls[name])
        got = _resample(candidate.controls[name], ref.shape[-1])
        if ref.shape[0] != got.shape[0]:
            raise ValueError(f"Control row count differs for {name}: {ref.shape} vs {got.shape}.")
        control_losses[name] = float(np.mean(((got - ref) / PULSE_WIDTH_LIMIT_S) ** 2))
    state_mean = float(np.mean(list(state_losses.values())))
    control_mean = float(np.mean(list(control_losses.values())))
    total = (state_weight * state_mean + control_weight * control_mean) / (state_weight + control_weight)
    return {
        "objective": float(total), "state_mse_normalized": state_mean,
        "control_mse_normalized": control_mean, "state_by_name": state_losses,
        "control_by_name": control_losses, "max_initial_state_abs_error": max_start_error,
        "initial_state_within_tolerance": bool(np.allclose(
            [0.0], [max_start_error], rtol=alignment_rtol, atol=alignment_atol)),
    }


def window_archive(reference: TrajectoryArchive, start_cycle: int, end_cycle: int,
                   *, controls_per_cycle: int = 30) -> TrajectoryArchive:
    """Extract inclusive 1-based cycle interval while retaining standard keys."""
    total_cycles = reference.cycles(controls_per_cycle=controls_per_cycle)
    if not 1 <= start_cycle <= end_cycle <= total_cycles:
        raise ValueError(f"Invalid window {start_cycle}-{end_cycle}; reference has {total_cycles} cycles.")
    states = {}
    for name, values in reference.states.items():
        columns = _matrix(values).shape[-1]
        state_intervals, remainder = divmod(columns - 1, total_cycles)
        if remainder:
            raise ValueError(
                f"State '{name}' has {columns - 1} intervals, incompatible with "
                f"the {total_cycles} control-derived cycles in {reference.path}."
            )
        left, right = (start_cycle - 1) * state_intervals, end_cycle * state_intervals
        states[name] = _matrix(values)[:, left:right + 1].copy()
    left, right = (start_cycle - 1) * controls_per_cycle, end_cycle * controls_per_cycle
    controls = {name: _matrix(values)[:, left:right].copy() for name, values in reference.controls.items()}
    metadata = dict(reference.metadata)
    window_cycles = end_cycle - start_cycle + 1
    # The extracted trace is used as the initial primal of a same-length RHO.
    # Keep its physical provenance but make the horizon metadata consistent
    # with that receiving OCP, otherwise strict seed validation rejects an
    # FHO_65 prefix when fitting an RHO_10 window.
    metadata.update({"fho_weight_fit_window_start_cycle": start_cycle,
                     "fho_weight_fit_window_end_cycle": end_cycle,
                     "cycles_per_window": window_cycles,
                     "n_windows": window_cycles})
    return TrajectoryArchive(states, controls, metadata, reference.path)


def write_archive(archive: TrajectoryArchive, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"states__{name}": value for name, value in archive.states.items()}
    payload.update({f"controls__{name}": value for name, value in archive.controls.items()})
    payload["metadata__json"] = np.asarray(json.dumps(archive.metadata, sort_keys=True))
    np.savez(path, **payload)
    return path


def _simplex_latin_hypercube(count: int, rng: np.random.Generator) -> np.ndarray:
    """Generate well-spread, non-negative four-muscle proportions.

    A Latin hypercube is made in four exponential coordinates and then
    normalised.  The transformation maps every row exactly onto the L1
    simplex, without requiring SciPy or a projection after sampling.
    """
    if count == 0:
        return np.empty((0, len(MUSCLE_NAMES)))
    uniforms = np.empty((count, len(MUSCLE_NAMES)))
    for column in range(len(MUSCLE_NAMES)):
        uniforms[:, column] = (rng.permutation(count) + rng.random(count)) / count
    exponential = -np.log(np.clip(uniforms, np.finfo(float).tiny, 1.0))
    return exponential / exponential.sum(axis=1, keepdims=True)


def _simplex_local_candidates(
    centre: np.ndarray, count: int, rng: np.random.Generator, *, concentration: float
) -> np.ndarray:
    """Sample local simplex perturbations around a successful proportion vector."""
    if count == 0:
        return np.empty((0, len(MUSCLE_NAMES)))
    centre = np.asarray(centre, dtype=float)
    if centre.shape != (len(MUSCLE_NAMES),) or np.any(centre < 0) or not np.isclose(centre.sum(), 1.0):
        raise ValueError("Local refinement centre must be a non-negative four-weight simplex vector.")
    # A small floor keeps a boundary optimum explorable rather than permanently
    # fixing zero-weight muscles during refinement.
    alpha = np.maximum(centre * concentration, 0.05)
    return rng.dirichlet(alpha, size=count)


def fit_window(
    reference: TrajectoryArchive,
    solve_rho: Callable[[np.ndarray, int], str | Path],
    *, window_index: int, evaluations: int = 32,
    seed: int = 0,
    parallel_workers: int = 1,
    refine_best: int = 0,
    refine_evaluations: int = 0,
    refine_concentration: float = 30.0,
) -> dict:
    """Fit interpretable muscle proportions with global design and refinement.

    The RHO solves have no data dependency once their common FHO window has
    been materialised.  Results are still assembled in evaluation order, so a
    parallel run has the same report as a serial one for a deterministic RHO
    command.  The caller is responsible for assigning each RHO process a
    non-overlapping CPU allocation in its command template.
    """
    if evaluations < 5:
        raise ValueError("At least five global evaluations are required (four vertices plus uniform).")
    if parallel_workers < 1:
        raise ValueError("parallel_workers must be at least one.")
    if refine_best < 0 or refine_evaluations < 0 or refine_concentration <= 0:
        raise ValueError("Refinement counts must be non-negative and concentration must be positive.")
    rng = np.random.default_rng(seed + window_index)
    # The vertices make sparse policies observable; the uniform vector anchors
    # the equal-sharing policy.  All remaining points are L1-normalised.
    candidates = [np.full(len(MUSCLE_NAMES), 1.0 / len(MUSCLE_NAMES))]
    candidates.extend(np.eye(len(MUSCLE_NAMES)))
    candidates.extend(_simplex_latin_hypercube(evaluations - len(candidates), rng))

    def evaluate(index: int, weights: np.ndarray, phase: str) -> dict:
        weights = np.asarray(weights, dtype=float)
        try:
            output = Path(solve_rho(weights, index)).expanduser().resolve()
            score = trajectory_tracking_loss(reference, TrajectoryArchive.load(output))
            attempt = {"evaluation": index, "weights": dict(zip(MUSCLE_NAMES, map(float, weights))),
                       "rho_solution": str(output), "status": "ok", "phase": phase, **score}
            if not score["initial_state_within_tolerance"]:
                attempt["status"] = "rejected_initial_state_mismatch"
        except Exception as exc:  # RHO infeasibility is an observation, not a crashed campaign.
            attempt = {"evaluation": index, "weights": dict(zip(MUSCLE_NAMES, map(float, weights))),
                       "status": "rho_failed", "phase": phase, "error": f"{type(exc).__name__}: {exc}"}
        return attempt

    def evaluate_batch(batch: Sequence[np.ndarray], *, offset: int, phase: str) -> list[dict]:
        attempts_by_index: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=min(parallel_workers, len(batch))) as executor:
            futures = {
                executor.submit(evaluate, index, weights, phase): index
                for index, weights in enumerate(batch, start=offset)
            }
            for future in as_completed(futures):
                attempt = future.result()
                attempts_by_index[attempt["evaluation"]] = attempt
        return [attempts_by_index[index] for index in range(offset, offset + len(batch))]

    attempts = evaluate_batch(candidates, offset=0, phase="global_latin_hypercube")
    valid_global = sorted(
        (attempt for attempt in attempts if attempt["status"] == "ok"),
        key=lambda attempt: attempt["objective"],
    )
    refinement_centres = valid_global[:refine_best]
    refinement = []
    for centre in refinement_centres:
        weights = np.array([centre["weights"][name] for name in MUSCLE_NAMES], dtype=float)
        refinement.extend(_simplex_local_candidates(
            weights, refine_evaluations, rng, concentration=refine_concentration,
        ))
    if refinement:
        attempts.extend(evaluate_batch(refinement, offset=len(attempts), phase="local_dirichlet_refinement"))
    best = min(
        (attempt for attempt in attempts if attempt["status"] == "ok"),
        key=lambda attempt: attempt["objective"],
        default=None,
    )
    return {"window_index": window_index, "evaluation_count": len(attempts),
            "global_evaluations": evaluations, "refinement_evaluations": len(refinement),
            "weight_constraint": "nonnegative_l1_simplex_sum_equals_1",
            "parallel_workers": min(parallel_workers, len(candidates)),
            "best": best, "evaluations": attempts,
            "status": "ok" if best is not None else "no_valid_rho_candidate"}
