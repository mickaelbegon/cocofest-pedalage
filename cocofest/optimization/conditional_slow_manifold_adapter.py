"""Offline initial-state binding for a Ding slow-manifold capacity atlas.

This only prepares copied NLP bounds.  It does not solve, certify feasibility,
or change the production RHO graph.  Fast states, mechanics, controls, history,
terminal conditions and every graph constraint remain as prepared.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .ding_slow_manifold import DingSlowFatigueManifold
from .task_reserve import TaskReserveCheckpoint
from .task_reserve_coordinates import ding_a_capacity_layout


STATE_ORDER = ("A", "Tau1", "Km")
SCHEMA = "conditional-ding-slow-initial-state-v1"


def _array_digest(value: np.ndarray) -> dict:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return {"shape": list(array.shape), "sha256": sha256(array.tobytes()).hexdigest()}


@dataclass(frozen=True)
class SlowManifoldInitialStateAdapter:
    """Map one capacity ratio per muscle to three copied initial equalities."""

    checkpoint: TaskReserveCheckpoint
    muscle_order: tuple[str, ...]
    manifolds: tuple[DingSlowFatigueManifold, ...]
    indices: tuple[tuple[int, int, int], ...]
    coefficients: tuple[tuple[float, float, float], ...]
    center: tuple[float, ...]
    limits: Mapping[str, np.ndarray]
    condition_sha256: str
    condition_json: str

    @classmethod
    def from_program(cls, *, checkpoint: TaskReserveCheckpoint, program,
                     muscle_order: Sequence[str],
                     manifolds: Mapping[str, DingSlowFatigueManifold],
                     manifold_tolerance: float = 5e-8) -> "SlowManifoldInitialStateAdapter":
        import casadi as ca
        from cocofest.optimization.receding_horizon_initial_guess import snapshot_initial_guess
        from cocofest.simulation.rho_restart_checkpoint import (
            _runtime_state, prepared_problem_arrays, prepared_problem_digest, verify_prepared_checkpoint,
        )

        if not isinstance(checkpoint, TaskReserveCheckpoint):
            raise TypeError("checkpoint must be a TaskReserveCheckpoint")
        checkpoint.verify_files()
        checked = verify_prepared_checkpoint(Path(checkpoint.archive_path),
                                             completed_cycles=checkpoint.completed_cycles)
        if (checked["prepared_problem_sha256"] != checkpoint.prepared_problem_sha256
                or checked["metadata"].get("prepared_stimulation_history_complete") is not True):
            raise ValueError("Prepared checkpoint or stimulation history disagrees with source")
        live_digest = prepared_problem_digest(snapshot_initial_guess(program),
                                              prepared_problem_arrays(program), _runtime_state(program))
        if live_digest != checkpoint.prepared_problem_sha256:
            raise ValueError("Live program differs from the complete prepared source")
        if not math.isfinite(manifold_tolerance) or manifold_tolerance < 0:
            raise ValueError("manifold_tolerance must be finite and non-negative")
        names = tuple(muscle_order)
        layout = ding_a_capacity_layout(Path(checkpoint.model_path), muscle_names=names)
        if set(manifolds) != set(names) or any(not isinstance(manifolds[name], DingSlowFatigueManifold) for name in names):
            raise ValueError("One explicit Ding slow manifold is required for every muscle")
        ordered = tuple(manifolds[name] for name in names)
        model = json.loads(Path(checkpoint.model_path).read_text())
        for name, coordinate, manifold in zip(names, layout, ordered):
            parameters = model["muscles"][name]
            if manifold.a_rest != coordinate.scale or any(
                getattr(manifold, field) != float(parameters[field])
                for field in ("alpha_a", "alpha_tau1", "alpha_km", "tau1_rest", "km_rest")
                if field in parameters
            ):
                raise ValueError(f"Slow manifold disagrees with declared model parameters for {name}")

        nlp, graph = program.nlp[0], program.ocp_solver.nlp
        x = graph["x"]
        expressions = []
        for name in names:
            for state in STATE_ORDER:
                key = f"{state}_{name}"
                expressions.append(nlp.X_scaled[0][nlp.states[key].index[0], 0]
                                   * float(np.asarray(nlp.x_scaling[key].scaling).reshape(-1)[0]))
        selected = ca.vertcat(*expressions)
        jacobian = ca.jacobian(selected, x)
        if ca.depends_on(jacobian, x):
            raise ValueError("Initial slow states must be affine decision coordinates")
        map_fn = ca.Function("ding_slow_initial_map", [x], [jacobian, selected])
        matrix, intercept = map_fn(np.zeros(int(x.numel())))
        if not np.allclose(np.asarray(intercept, float), 0., rtol=0, atol=1e-12):
            raise ValueError("Initial slow-state coordinates have an unexpected offset")
        indices, coefficients = [], []
        for row in np.asarray(matrix, float):
            nonzero = np.flatnonzero(abs(row) > 1e-14)
            if nonzero.size != 1 or not np.isfinite(row[nonzero[0]]) or row[nonzero[0]] <= 0:
                raise ValueError("Each slow state must map to one positive scaled decision variable")
            indices.append(int(nonzero[0]))
            coefficients.append(float(row[nonzero[0]]))
        if len(set(indices)) != len(indices):
            raise ValueError("Slow-state decision coordinates overlap")

        limits = {key: np.asarray(program.ocp_solver.limits[key], float).reshape(-1).copy()
                  for key in ("lbx", "ubx", "lbg", "ubg", "x0")}
        if any(limits[key].size != int(x.numel()) for key in ("lbx", "ubx", "x0")):
            raise ValueError("Incomplete source decision vectors")
        if (limits["lbg"].size != int(graph["g"].numel())
                or limits["ubg"].size != limits["lbg"].size
                or np.any(limits["lbx"] > limits["ubx"])
                or np.any(limits["lbg"] > limits["ubg"])
                or np.any(np.isnan(limits["lbx"])) or np.any(np.isnan(limits["ubx"]))
                or not np.all(np.isfinite(limits["x0"]))):
            raise ValueError("Invalid source NLP vectors")

        centers = []
        with np.load(checkpoint.archive_path, allow_pickle=False) as archive:
            for muscle_index, (name, manifold) in enumerate(zip(names, ordered)):
                source_state = []
                for component_index, state in enumerate(STATE_ORDER):
                    key = f"{state}_{name}"
                    lower = np.asarray(archive[f"problem__x_bounds:{key}:min"], float)
                    upper = np.asarray(archive[f"problem__x_bounds:{key}:max"], float)
                    guess = np.asarray(archive[f"states__{key}"], float)
                    if (lower.ndim != 2 or upper.shape != lower.shape or guess.ndim != 2
                            or lower.shape[0] != 1 or guess.shape[0] != 1 or guess.shape[1] < 1):
                        raise ValueError(f"Invalid prepared slow-state layout for {key}")
                    value = float(lower[0, 0])
                    index = indices[3 * muscle_index + component_index]
                    coefficient = coefficients[3 * muscle_index + component_index]
                    if (not np.isfinite(value) or value != float(upper[0, 0])
                            or not np.isclose(value, guess[0, 0], rtol=0, atol=1e-8)
                            or not np.isclose(value, limits["lbx"][index] * coefficient, rtol=0, atol=1e-8)
                            or not np.isclose(value, limits["ubx"][index] * coefficient, rtol=0, atol=1e-8)):
                        raise ValueError(f"Prepared initial {key} is not the mapped fixed source state")
                    source_state.append(value)
                ratio = manifold.capacity_ratio_from_states(*source_state, tolerance=manifold_tolerance)
                if not 0 <= ratio <= 1:
                    raise ValueError(f"Prepared capacity for {name} is outside [0,1]")
                centers.append(ratio)

        graph_hash = sha256(ca.Function("ding_slow_source_graph", [x],
                                        [graph["g"], graph["f"]]).serialize().encode()).hexdigest()
        condition = {"schema": SCHEMA, "archive_sha256": checkpoint.archive_sha256,
                     "prepared_problem_sha256": checkpoint.prepared_problem_sha256,
                     "model_sha256": checkpoint.model_sha256,
                     "task_context_sha256": checkpoint.task_context_sha256,
                     "muscle_order": names, "manifolds": [asdict(item) for item in ordered],
                     "state_order": STATE_ORDER, "indices": indices, "coefficients": coefficients,
                     "source_limits": {key: _array_digest(value) for key, value in limits.items()},
                     "constraint_and_objective_sha256": graph_hash,
                     "fast_states_mechanics_history_and_other_bounds_conditioned": True,
                     "complete_state_reachability_asserted": False}
        condition_json = json.dumps(condition, sort_keys=True, separators=(",", ":"), allow_nan=False)
        checkpoint.verify_files()
        return cls(checkpoint, names, ordered,
                   tuple(tuple(indices[3*i:3*i+3]) for i in range(len(names))),
                   tuple(tuple(coefficients[3*i:3*i+3]) for i in range(len(names))),
                   tuple(centers), limits, sha256(condition_json.encode()).hexdigest(), condition_json)

    def bounds_for(self, capacity_ratios: Sequence[float]) -> dict[str, np.ndarray]:
        """Return an independent source-bound copy with compatible slow initial states."""
        self.verify()
        ratios = np.asarray(capacity_ratios, float)
        if (ratios.shape != (len(self.muscle_order),) or not np.all(np.isfinite(ratios))
                or np.any(ratios < 0) or np.any(ratios > 1)):
            raise ValueError("One physical capacity ratio in [0,1] is required per muscle")
        limits = {key: value.copy() for key, value in self.limits.items()}
        for ratio, manifold, indices, coefficients in zip(ratios, self.manifolds, self.indices, self.coefficients):
            states = manifold.states_from_capacity_ratio(float(ratio))
            if any(value <= 0 or not np.isfinite(value) for value in states):
                raise ValueError("Candidate leaves the positive Ding slow-state domain")
            for value, index, coefficient in zip(states, indices, coefficients):
                scaled = value / coefficient
                for key in ("lbx", "ubx", "x0"):
                    limits[key][index] = scaled
        return limits

    def verify(self) -> None:
        """Detect source-file or copied-bound mutation before producing a probe."""
        self.checkpoint.verify_files()
        condition = json.loads(self.condition_json)
        if (sha256(self.condition_json.encode()).hexdigest() != self.condition_sha256
                or condition["source_limits"] != {key: _array_digest(value) for key, value in self.limits.items()}):
            raise ValueError("Slow-manifold condition or copied source bounds changed")
