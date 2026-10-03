"""Pure acceptance contract for an asynchronous one-cycle load supervisor.

The external oracle minimizes ``-load_factor`` with the complete RHO mechanics,
muscle dynamics, history and closure constraints. Its feasible local solution
is a lower bound on achievable load, never a global maximum certificate. This
module does not solve that OCP or communicate with a worker.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .task_reserve import ProbeEvidence, TaskReserveCheckpoint, TaskReserveProbe


CONTRACT_VERSION = "task_load_margin_v1"


def _vector(value, name, size=None):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not result.size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    if size is not None and result.size != size:
        raise ValueError(f"{name} has incompatible dimension")
    return result


def _nonnegative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class TaskLoadMarginRequest:
    """Pinned request; all coordinates below are dimensionless and ordered.

    Normalizations belong in checkpoint.task_context.coordinate_layout. All
    unselected states and stimulation history are fixed by the checkpoint.
    ``issued_monotonic_seconds`` uses the same host clock as selection.
    """

    request_id: str
    checkpoint: TaskReserveCheckpoint
    coordinate_names: tuple[str, ...]
    center: tuple[float, ...]
    trust_radius: tuple[float, ...]
    domain_lower: tuple[float, ...]
    domain_upper: tuple[float, ...]
    issued_monotonic_seconds: float
    load_factor_upper_bound: float

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be nonempty")
        if not isinstance(self.checkpoint, TaskReserveCheckpoint):
            raise TypeError("checkpoint must be a TaskReserveCheckpoint")
        if type(self.checkpoint.completed_cycles) is not int or self.checkpoint.completed_cycles < 0:
            raise ValueError("checkpoint completed_cycles must be a nonnegative integer")
        names = self.coordinate_names
        if (not names or len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name for name in names)):
            raise ValueError("coordinate_names must be distinct nonempty strings")
        center = _vector(self.center, "center", len(names))
        radius = _vector(self.trust_radius, "trust_radius", len(names))
        lower = _vector(self.domain_lower, "domain_lower", len(names))
        upper = _vector(self.domain_upper, "domain_upper", len(names))
        if np.any(radius <= 0) or np.any(lower > center) or np.any(upper < center):
            raise ValueError("center must be inside domain and trust radii positive")
        _nonnegative(self.issued_monotonic_seconds, "issued_monotonic_seconds")
        if not math.isfinite(self.load_factor_upper_bound) or self.load_factor_upper_bound <= 1:
            raise ValueError("load_factor_upper_bound must be finite and greater than nominal 1")
        # Frozen dataclasses alone do not freeze caller-owned list/array inputs.
        object.__setattr__(self, "coordinate_names", tuple(names))
        for name, value in (("center", center), ("trust_radius", radius),
                            ("domain_lower", lower), ("domain_upper", upper)):
            object.__setattr__(self, name, tuple(float(item) for item in value))


@dataclass(frozen=True)
class TaskLoadMarginResult:
    """Oracle evidence; unknown/failed fields remain None, never synthetic zero.

    ``gradient`` is d(load_factor)/d(normalized coordinates), not the gradient
    of minimized -load_factor. Local validation must use reoptimized held-out
    complete OCPs, not a replay of the affine expression itself. KKT diagnostics
    include the load variable, all bounds and the complete constraint system.
    A feasible witness alone can be reported but cannot activate a direction.
    """

    request_id: str
    completed_monotonic_seconds: float
    nominal_witness: ProbeEvidence | None = None
    load_witness: ProbeEvidence | None = None
    gradient: tuple[float, ...] | None = None
    sensitivity_method: str | None = None
    sensitivity_regular: bool = False
    stationarity_residual: float | None = None
    complementarity_residual: float | None = None
    dual_feasibility_residual: float | None = None
    gradient_validation_max_abs_error: float | None = None
    local_validation_max_abs_error: float | None = None
    independent_validation_points: int = 0
    failure_reason: str | None = None
    solution_artifact: str | None = None
    multipliers_artifact: str | None = None

    def __post_init__(self):
        if self.gradient is not None:
            object.__setattr__(self, "gradient", tuple(self.gradient))


@dataclass(frozen=True)
class TaskLoadMarginPolicy:
    maximum_age_cycles: int = 20
    maximum_age_seconds: float = 20.
    feasibility_tolerance: float = 1e-6
    kkt_tolerance: float = 1e-5
    gradient_error_tolerance: float = .05
    local_error_tolerance: float = .01

    def __post_init__(self):
        if type(self.maximum_age_cycles) is not int or self.maximum_age_cycles < 0:
            raise ValueError("maximum_age_cycles must be a nonnegative integer")
        for name in ("maximum_age_seconds", "feasibility_tolerance", "kkt_tolerance",
                     "gradient_error_tolerance", "local_error_tolerance"):
            _nonnegative(getattr(self, name), name)


@dataclass(frozen=True)
class TaskLoadMarginDirection:
    """Accepted local affine model; not a feasibility or endurance certificate."""

    request: TaskLoadMarginRequest
    result: TaskLoadMarginResult

    def value(self, coordinates: Sequence[float]) -> float:
        reasons = _point_reasons(self.request, coordinates)
        if reasons:
            raise ValueError("Invalid local load-margin point: " + ", ".join(reasons))
        return float(self.result.load_witness.work_scale - 1.
                     + np.dot(self.result.gradient, np.asarray(coordinates) - self.request.center))

    def parameter_vector(self, weight: float) -> np.ndarray:
        """Fixed graph layout [weight, center(n), gradient(n)] for -w*g.(x-x0)."""
        _nonnegative(weight, "weight")
        return np.r_[weight, self.request.center, self.result.gradient]


@dataclass(frozen=True)
class TaskLoadMarginSelection:
    direction: TaskLoadMarginDirection | None
    action: str
    rejected: tuple[tuple[str, tuple[str, ...]], ...]
    physiological_failure_certified: bool = False


def _point_reasons(request, coordinates):
    try:
        point = _vector(coordinates, "coordinates", len(request.center))
    except (ValueError, TypeError):
        return ["invalid_coordinates"]
    reasons = []
    if np.any(point < request.domain_lower) or np.any(point > request.domain_upper):
        reasons.append("outside_physical_domain")
    if np.any(np.abs(point - request.center) > np.asarray(request.trust_radius) + 1e-12):
        reasons.append("outside_trust_region")
    return reasons


def validate_task_load_margin(request, result, *, policy, completed_cycles,
                              now_monotonic_seconds, coordinates,
                              model_sha256, task_context_sha256, coordinate_names):
    """Return deterministic rejection reasons; never infer physiological failure."""
    if type(completed_cycles) is not int or completed_cycles < 0:
        raise ValueError("completed_cycles must be a nonnegative integer")
    _nonnegative(now_monotonic_seconds, "now_monotonic_seconds")
    reasons = []
    if result.request_id != request.request_id:
        reasons.append("request_id_mismatch")
    if (request.checkpoint.model_sha256 != model_sha256
            or request.checkpoint.task_context_sha256 != task_context_sha256
            or tuple(coordinate_names) != request.coordinate_names):
        reasons.append("context_mismatch")
    age = completed_cycles - request.checkpoint.completed_cycles
    if age < 0 or age > policy.maximum_age_cycles:
        reasons.append("cycle_age_invalid")
    issued, finished = request.issued_monotonic_seconds, result.completed_monotonic_seconds
    if (not math.isfinite(finished) or finished < issued or finished > now_monotonic_seconds
            or now_monotonic_seconds - issued > policy.maximum_age_seconds):
        reasons.append("time_age_invalid")
    reasons.extend(_point_reasons(request, coordinates))
    if result.failure_reason is not None:
        reasons.append("oracle_reported_failure")
    for name, witness, scale in (("nominal", result.nominal_witness, 1.),
                                  ("load", result.load_witness,
                                   None if result.load_witness is None else result.load_witness.work_scale)):
        if witness is None:
            reasons.append(f"missing_{name}_witness")
            continue
        try:
            valid = witness.witness_is_valid(TaskReserveProbe(request.checkpoint, scale))
            valid = valid and witness.maximum_normalized_constraint_violation <= policy.feasibility_tolerance
        except (ValueError, TypeError):
            valid = False
        if not valid:
            reasons.append(f"invalid_{name}_witness")
    if result.load_witness is not None:
        if result.load_witness.work_scale >= request.load_factor_upper_bound - policy.feasibility_tolerance:
            reasons.append("artificial_load_cap_active")
        if result.load_witness.work_scale < 1.:
            reasons.append("load_below_explicit_nominal_witness")
    try:
        _vector(result.gradient, "gradient", len(request.center))
    except (ValueError, TypeError):
        reasons.append("invalid_gradient")
    if result.sensitivity_method not in ("kkt_envelope", "finite_difference_reoptimized"):
        reasons.append("unsupported_sensitivity_method")
    if result.sensitivity_regular is not True:
        reasons.append("irregular_sensitivity")
    for name, limit in (("stationarity_residual", policy.kkt_tolerance),
                        ("complementarity_residual", policy.kkt_tolerance),
                        ("dual_feasibility_residual", policy.kkt_tolerance),
                        ("gradient_validation_max_abs_error", policy.gradient_error_tolerance),
                        ("local_validation_max_abs_error", policy.local_error_tolerance)):
        value = getattr(result, name)
        if value is None or not math.isfinite(value) or value < 0 or value > limit:
            reasons.append(f"invalid_{name}")
    if type(result.independent_validation_points) is not int or result.independent_validation_points < 1:
        reasons.append("missing_independent_local_validation")
    if not result.solution_artifact:
        reasons.append("missing_solution_artifact")
    return tuple(reasons)


def select_task_load_margin(candidates, *, previous=None, **validation):
    """Choose newest usable source; failed/new stale jobs cannot erase a valid old one.

    The caller polls nonblocking, passes already completed (request,result)
    pairs, and uses ``None`` to deactivate the extra objective. Revalidate at
    the solved terminal point using this same API before transferring a cycle.
    """
    pairs = list(candidates)
    if previous is not None:
        pairs.append((previous.request, previous.result))
    accepted, rejected = [], []
    for request, result in pairs:
        reasons = validate_task_load_margin(request, result, **validation)
        if reasons:
            rejected.append((request.request_id, reasons))
        else:
            accepted.append(TaskLoadMarginDirection(request, result))
    if not accepted:
        return TaskLoadMarginSelection(None, "deactivate", tuple(rejected))
    selected = max(accepted, key=lambda item: (item.request.checkpoint.completed_cycles,
                   item.request.issued_monotonic_seconds, item.result.completed_monotonic_seconds))
    action = "keep_previous" if previous is not None and selected == previous else "activate"
    return TaskLoadMarginSelection(selected, action, tuple(rejected))


def load_gradient_from_kkt(*, objective_parameter_gradient,
                           constraint_parameter_jacobian, constraint_multipliers,
                           bound_parameter_contribution, coordinate_scales):
    """Envelope gradient for min -lambda, with L=f+mu.T*c convention.

    Inputs differentiate w.r.t. physical checkpoint coordinates p. If normalized
    x=(p-offset)/scale, d(lambda)/dx=-scale*(f_p+c_p.T*mu+bound_p).
    ``bound_parameter_contribution`` is required explicitly (zeros only when
    bounds are parameter-independent). Inequality and bound multiplier signs
    must first be converted from the solver convention to this Lagrangian.
    Regularity, residuals and independent derivative checks are caller duties.
    """
    objective = _vector(objective_parameter_gradient, "objective_parameter_gradient")
    scale = _vector(coordinate_scales, "coordinate_scales", objective.size)
    bound = _vector(bound_parameter_contribution, "bound_parameter_contribution", objective.size)
    multipliers = _vector(constraint_multipliers, "constraint_multipliers")
    jacobian = np.asarray(constraint_parameter_jacobian, dtype=float)
    if jacobian.shape != (multipliers.size, objective.size) or not np.all(np.isfinite(jacobian)):
        raise ValueError("constraint_parameter_jacobian has incompatible shape or nonfinite values")
    if np.any(scale <= 0):
        raise ValueError("coordinate_scales must be positive")
    return -scale * (objective + jacobian.T @ multipliers + bound)
