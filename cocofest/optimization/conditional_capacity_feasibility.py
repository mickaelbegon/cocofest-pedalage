"""Offline conditional slices of one-cycle task feasibility.

Only initial A/a_scale varies. Cn, F, Tau1, Km, mechanics, device bounds and
stimulation history remain conditioned on the exact prepared source. These
counterfactual states are NOT asserted to be reachable by a fatigue trajectory.

The auxiliary solver minimizes normalized missing work while keeping all other
discrete constraints. A positive local deficit is an upper bound on the best
deficit, never a lower bound proving infeasibility. No production cost changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .task_reserve import TaskReserveCheckpoint
from .task_reserve_coordinates import (
    capacity_coordinate_context, ding_a_capacity_layout, extract_prepared_capacity_coordinates,
)
from .task_reserve_probe_adapter import _normalized_bound_violation


SCHEMA = "conditional-capacity-feasibility-v1"


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ConditionalCapacitySlice:
    checkpoint: TaskReserveCheckpoint
    muscle_order: tuple[str, ...]
    center: tuple[float, ...]
    a_scales: tuple[float, ...]
    trust_radius: float
    condition_sha256: str
    condition_json: str

    @classmethod
    def from_checkpoint(cls, checkpoint, *, muscle_order, trust_radius=.05):
        if isinstance(trust_radius, bool) or not math.isfinite(trust_radius) or not 0 < trust_radius <= .2:
            raise ValueError("Local normalized capacity radius must lie in (0,.2]")
        names = tuple(muscle_order)
        layout = ding_a_capacity_layout(Path(checkpoint.model_path), muscle_names=names)
        context = {**json.loads(checkpoint.task_context_json), **capacity_coordinate_context(layout)}
        normalized = TaskReserveCheckpoint.from_archive(Path(checkpoint.archive_path),
            completed_cycles=checkpoint.completed_cycles, model_path=Path(checkpoint.model_path), task_context=context)
        observed = extract_prepared_capacity_coordinates(normalized, layout=layout)
        conditioned, intervals = {}, {}
        with np.load(checkpoint.archive_path, allow_pickle=False) as archive:
            for key in sorted(archive.files):
                if key == "metadata__json":
                    metadata = json.loads(str(archive[key].item()))
                    conditioned["runtime_state"] = metadata.get("runtime_state", {})
                    continue
                value = np.asarray(archive[key], float).copy()
                is_capacity = any(key == f"states__A_{name}" or key.startswith(f"problem__x_bounds:A_{name}:")
                                  for name in names)
                if is_capacity:
                    # Omit the varied initial capacity and its trajectory guess;
                    # retain path/terminal bounds. All non-A data remain pinned.
                    if key.startswith("states__"):
                        continue
                    value[:, 0] = 0.
                conditioned[key] = {"shape": list(value.shape),
                                    "sha256": sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
            for key in sorted(k.split("__", 1)[1] for k in archive.files if k.startswith("states__")):
                if key in {f"A_{name}" for name in names}:
                    continue
                lower, upper = [np.asarray(archive[f"problem__x_bounds:{key}:{field}"], float)[:, 0]
                                for field in ("min", "max")]
                if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
                    raise ValueError("All initial conditioned state intervals must be finite")
                intervals[key] = {"lower": lower.tolist(), "upper": upper.tolist(),
                                  "fixed_equality": bool(np.array_equal(lower, upper))}
        # Require every unvaried Ding state explicitly, including the slow
        # Tau1/Km that an A-only fatigue description would otherwise omit.
        if not {f"{prefix}_{name}" for name in names for prefix in ("Cn", "F", "Tau1", "Km")} <= intervals.keys():
            raise ValueError("Missing complete conditioned Ding state")
        condition = {"schema": SCHEMA, "model_sha256": checkpoint.model_sha256,
            "task_context_sha256": checkpoint.task_context_sha256,
            "muscle_order": list(names), "a_scales": [item.scale for item in layout],
            "unchanged_prepared_data": conditioned, "conditioned_initial_state_intervals": intervals,
            "synthetic_capacity_states_are_reachable": False,
            "conditioning_includes_warm_start_and_full_prepared_history": True}
        checkpoint.verify_files()
        return cls(checkpoint, names, observed.values, tuple(item.scale for item in layout), float(trust_radius),
                   _digest(condition), json.dumps(condition, sort_keys=True, allow_nan=False))

    def validate_point(self, values):
        values = np.asarray(values, float)
        if (values.shape != (len(self.center),) or not np.all(np.isfinite(values))
                or np.any(values < 0) or np.any(values > 1)
                or np.any(np.abs(values - self.center) > self.trust_radius + 1e-12)):
            raise ValueError("Capacity point is outside its declared physical/local domain")
        return tuple(float(x) for x in values)

    def verify(self):
        self.checkpoint.verify_files()
        if _digest(json.loads(self.condition_json)) != self.condition_sha256:
            raise ValueError("Conditional slice digest mismatch")


@dataclass(frozen=True)
class CapacityPoint:
    sample_id: str
    direction: str
    offset: float
    capacities: tuple[float, ...]


def directional_points(slice_: ConditionalCapacitySlice, directions: Mapping[str, Sequence[float]], offsets):
    """Evaluate the declared full grid; never binary-search a presumed boundary."""
    offsets = tuple(float(value) for value in offsets)
    if (not offsets or len(set(offsets)) != len(offsets) or any(not math.isfinite(v) or v == 0 for v in offsets)
            or not directions):
        raise ValueError("Distinct finite nonzero offsets and explicit directions are required")
    points = [CapacityPoint("center", "center", 0., slice_.center)]
    for name, direction in directions.items():
        vector = np.asarray(direction, float)
        if (not isinstance(name, str) or not name or name == "center" or vector.shape != (len(slice_.center),)
                or not np.all(np.isfinite(vector)) or not np.isclose(np.max(np.abs(vector)), 1., atol=1e-12, rtol=0)):
            raise ValueError("Every direction must have explicit matching dimension and infinity norm one")
        for index, offset in enumerate(offsets):
            capacities = slice_.validate_point(np.asarray(slice_.center) + offset * vector)
            points.append(CapacityPoint(f"{name}-{index}", name, offset, capacities))
    return tuple(points)


class FrozenTaskDeficit:
    """Own an auxiliary CasADi NLP without editing the source OCP or objective.

    Decision-space coordinate mappings are explicit and independently checked.
    All graph constraints and all bounds except initial A and terminal E_prod
    are copied unchanged. Terminal work is allowed in [0, nominal]; its linear
    normalized deficit is the sole *offline auxiliary* objective. The linear
    form avoids a vanishing gradient near a zero deficit.
    """

    def __init__(self, *, graph, limits, capacity_indices, capacity_coefficients,
                 work_index, work_coefficient, nominal_work_j, center):
        import casadi as ca

        self.graph = {key: graph[key] for key in ("x", "g", "f")}
        self.limits = {key: np.asarray(limits[key], float).reshape(-1).copy()
                       for key in ("lbx", "ubx", "lbg", "ubg", "x0")}
        size = int(self.graph["x"].numel())
        if (any(self.limits[key].size != size for key in ("lbx", "ubx", "x0"))
                or self.limits["lbg"].size != self.graph["g"].numel()
                or self.limits["ubg"].shape != self.limits["lbg"].shape):
            raise ValueError("Incomplete copied NLP vectors")
        for lower, upper in (("lbx", "ubx"), ("lbg", "ubg")):
            if (np.any(np.isnan(self.limits[lower])) or np.any(np.isnan(self.limits[upper]))
                    or np.any(self.limits[lower] > self.limits[upper])):
                raise ValueError("Invalid copied NLP bounds")
        if not np.all(np.isfinite(self.limits["x0"])):
            raise ValueError("Nonfinite source initial guess")
        self.indices = tuple(capacity_indices)
        self.coefficients = np.asarray(capacity_coefficients, float).copy()
        self.center = np.asarray(center, float).copy()
        if (not self.indices or len(set(self.indices)) != len(self.indices)
                or any(type(index) is not int or not 0 <= index < size for index in self.indices)
                or self.coefficients.shape != self.center.shape or self.center.shape != (len(self.indices),)
                or not np.all(np.isfinite(self.coefficients)) or np.any(self.coefficients <= 0)
                or not np.all(np.isfinite(self.center)) or np.any(self.center < 0) or np.any(self.center > 1)):
            raise ValueError("Invalid normalized initial-capacity mapping")
        if (type(work_index) is not int or not 0 <= work_index < size or work_index in self.indices
                or not math.isfinite(work_coefficient) or work_coefficient <= 0
                or not math.isfinite(nominal_work_j) or nominal_work_j <= 0):
            raise ValueError("Invalid terminal-work mapping")
        self.work_index, self.work_coefficient, self.nominal_work_j = work_index, work_coefficient, nominal_work_j
        for key in ("lbx", "ubx"):
            if not np.allclose(self.limits[key][list(self.indices)] * self.coefficients,
                               self.center, rtol=0, atol=1e-10):
                raise ValueError("Mapped capacity source is not a fixed initial equality")
            if not math.isclose(self.limits[key][work_index] * work_coefficient,
                                nominal_work_j, rel_tol=0, abs_tol=1e-10):
                raise ValueError("Mapped nominal work is not a fixed terminal equality")
        work = self.graph["x"][work_index] * work_coefficient
        self.auxiliary_graph = {"x": self.graph["x"], "g": self.graph["g"],
                                "f": (nominal_work_j - work) / nominal_work_j}
        self.evaluate_constraints = ca.Function("conditional_capacity_constraints", [self.graph["x"]], [self.graph["g"]])
        self.production_objective_sha256 = sha256(ca.Function("original_unit_objective", [self.graph["x"]],
                                                             [self.graph["f"]]).serialize().encode()).hexdigest()
        self.constraint_graph_sha256 = sha256(self.evaluate_constraints.serialize().encode()).hexdigest()

    def bounds_for(self, capacities, *, relax_work: bool):
        values = np.asarray(capacities, float)
        if values.shape != self.center.shape or not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise ValueError("Finite physical capacity vector required")
        limits = {key: value.copy() for key, value in self.limits.items()}
        for key in ("lbx", "ubx", "x0"):
            limits[key][list(self.indices)] = values / self.coefficients
        if relax_work:
            limits["lbx"][self.work_index] = 0.
        return limits

    def audit(self, vector, capacities, *, tolerance):
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Positive finite audit tolerance required")
        vector = np.asarray(vector, float).reshape(-1)
        constraints = np.asarray(self.evaluate_constraints(vector), float).reshape(-1)
        relaxed = self.bounds_for(capacities, relax_work=True)
        nominal = self.bounds_for(capacities, relax_work=False)
        constraint_residual = _normalized_bound_violation(constraints, relaxed["lbg"], relaxed["ubg"])
        relaxed_bound_residual = _normalized_bound_violation(vector, relaxed["lbx"], relaxed["ubx"])
        nominal_bound_residual = _normalized_bound_violation(vector, nominal["lbx"], nominal["ubx"])
        work = float(vector[self.work_index] * self.work_coefficient)
        deficit = (self.nominal_work_j - work) / self.nominal_work_j
        phase_one_valid = max(constraint_residual, relaxed_bound_residual) <= tolerance
        # Independent nominal audit, including work normalized by requested work
        # rather than the arbitrary numerical scaling of its NLP variable.
        nominal_valid = phase_one_valid and max(nominal_bound_residual, abs(deficit)) <= tolerance
        return {"phase_one_trajectory_validated": phase_one_valid,
                "nominal_task_witnessed": nominal_valid,
                "outcome": "nominal_feasible_witness" if nominal_valid else "indeterminate",
                "observed_normalized_work_deficit": max(0., deficit) if phase_one_valid else None,
                "achieved_work_j": work if phase_one_valid else None,
                "constraint_normalized_violation": constraint_residual,
                "relaxed_decision_normalized_violation": relaxed_bound_residual,
                "nominal_decision_normalized_violation": nominal_bound_residual,
                "tolerance": tolerance, "physiological_failure_certified": False,
                "positive_deficit_is_infeasibility_proof": False,
                "scope": "complete_discrete_constraints_and_bounds_no_continuous_replay"}


def mapping_from_program(program, slice_):
    """Derive selected state locations symbolically; never assume NLP offsets."""
    import casadi as ca

    nlp, graph = program.nlp[0], program.ocp_solver.nlp
    expressions = []
    for name, scale in zip(slice_.muscle_order, slice_.a_scales):
        key = f"A_{name}"
        expressions.append(nlp.X_scaled[0][nlp.states[key].index[0], 0]
                          * float(np.asarray(nlp.x_scaling[key].scaling).reshape(-1)[0]) / scale)
    expressions.append(nlp.X_scaled[-1][nlp.states["E_prod"].index[0], 0]
                      * float(np.asarray(nlp.x_scaling["E_prod"].scaling).reshape(-1)[0]))
    selected = ca.vertcat(*expressions)
    jacobian = ca.jacobian(selected, graph["x"])
    if ca.depends_on(jacobian, graph["x"]):
        raise ValueError("Selected states are not direct affine decision variables")
    mapping = ca.Function("capacity_state_mapping", [graph["x"]], [jacobian, selected])
    matrix, intercept = mapping(np.zeros(int(graph["x"].numel())))
    if np.any(np.asarray(intercept, float) != 0):
        raise ValueError("Unexpected offset in normalized coordinate mapping")
    indices, coefficients = [], []
    for row in np.asarray(matrix, float):
        nonzero = np.flatnonzero(row)
        if nonzero.size != 1 or row[nonzero[0]] <= 0:
            raise ValueError("Each selected state must map to one positive scaled decision variable")
        indices.append(int(nonzero[0]))
        coefficients.append(float(row[nonzero[0]]))
    return {"capacity_indices": indices[:-1], "capacity_coefficients": coefficients[:-1],
            "work_index": indices[-1], "work_coefficient": coefficients[-1]}


def summarize_slice(slice_, points, records):
    """Report witnessed samples only; no invented separating hyperplane."""
    slice_.verify()
    if len(points) != len(records) or len({point.sample_id for point in points}) != len(points):
        raise ValueError("Every predeclared point requires exactly one result")
    for point, record in zip(points, records):
        slice_.validate_point(point.capacities)
        if (record.get("sample_id") != point.sample_id or tuple(record.get("capacities", ())) != point.capacities
                or record.get("condition_sha256") != slice_.condition_sha256):
            raise ValueError("Result has a different point or full-state condition")
        if record.get("physiological_failure_certified") is not False:
            raise ValueError("A local deficit result cannot certify physiological failure")
    directions = {}
    for name in sorted({point.direction for point in points} - {"center"}):
        rows = sorted([(point.offset, record) for point, record in zip(points, records)
                       if point.direction in (name, "center")], key=lambda item: item[0])
        witnessed = [offset for offset, record in rows if record.get("nominal_task_witnessed") is True]
        unknown = [offset for offset, record in rows if record.get("nominal_task_witnessed") is not True]
        directions[name] = {"witnessed_offsets": witnessed, "undetermined_offsets": unknown,
            "unknown_between_witnesses": any(min(witnessed) < value < max(witnessed) for value in unknown)
                                        if witnessed else False,
            "feasibility_monotonicity_assumed": False, "boundary_estimate": None}
    return {"schema": SCHEMA, "source": asdict(slice_.checkpoint),
            "condition": json.loads(slice_.condition_json), "condition_sha256": slice_.condition_sha256,
            "center": list(slice_.center), "trust_radius": slice_.trust_radius,
            "points": [asdict(point) for point in points], "records": list(records), "directions": directions,
            "activation_allowed": False, "endurance_prediction_validated": False,
            "production_objective_modified": False, "physical_cycles_advanced": 0,
            "interpretation": "Conditional feasibility witnesses at sampled counterfactual capacities; no fatigue boundary proof"}
