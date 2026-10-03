"""Experimental one-cycle task reserve measured with the full RHO problem.

The oracle supplied to :func:`evaluate_work_reserve` must restore the *same*
prepared checkpoint before every solve, change only the prescribed work, and
check the returned trajectory against the complete RHO constraints.  This
module deliberately has no reduced Ding/geometry model and no solver-status
test masquerading as a physiological impossibility certificate.

A feasible trajectory at work factor ``lambda`` witnesses that the maximum
achievable factor is at least ``lambda``. Its reserve ``lambda - 1`` is a
lower bound, not the maximum itself. A failed NLP gives neither an upper bound
nor a proof of fatigue failure. We also do not assume feasibility is monotone
in work: cycle closure, power bounds and stimulation history can invalidate
that assumption. The nominal factor 1 must therefore be tested explicitly.

Local fits approximate the *observed witness reserve*, not the unknown global
value function. They require full-rank samples and independent holdout checks.
The resulting coefficients can be injected as numerical parameters into a
fixed objective graph through the opt-in ``task_reserve_ocp`` binding. That
experimental binding explicitly supports coexistence with fatigue weights;
calling the standalone functions here does not alter any production RHO.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


WORK_RESERVE_METRIC = "one_cycle_work_scale"
DEFAULT_CONSTRAINT_GROUPS = (
    "dynamics", "initial_state", "state_bounds", "control_bounds", "parameter_bounds",
    "path_constraints", "task_work", "cycle_closure", "stimulation_history",
)


def _digest_json(value) -> str:
    return sha256(json.dumps(value, allow_nan=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class TaskReserveCheckpoint:
    """Immutable provenance of a complete prepared one-cycle RHO problem.

    ``task_context`` identifies the physical problem (including nominal work,
    cadence, arm, half-step constraint, solver tolerance and state-coordinate
    normalization), excluding the particular fatigue state being sampled.
    Model and task hashes must agree between the samples of a local fit.
    The archive digest additionally pins the actual initial state and history.
    """

    archive_path: str
    archive_sha256: str
    prepared_problem_sha256: str
    completed_cycles: int
    model_path: str
    model_sha256: str
    task_context_json: str
    task_context_sha256: str
    required_constraint_groups: tuple[str, ...] = DEFAULT_CONSTRAINT_GROUPS

    @classmethod
    def from_archive(cls, archive_path: Path, *, completed_cycles: int,
                     model_path: Path, task_context: Mapping,
                     required_constraint_groups: Sequence[str] = DEFAULT_CONSTRAINT_GROUPS):
        from cocofest.simulation.rho_restart_checkpoint import verify_prepared_checkpoint

        if type(completed_cycles) is not int or completed_cycles < 0:
            raise ValueError("completed_cycles must be a nonnegative integer")
        archive_path, model_path = Path(archive_path).resolve(strict=True), Path(model_path).resolve(strict=True)
        checked = verify_prepared_checkpoint(archive_path, completed_cycles=completed_cycles)
        if checked["metadata"].get("prepared_stimulation_history_complete") is not True:
            raise ValueError("Task reserve needs a complete stimulation-history checkpoint")
        if not isinstance(task_context, Mapping) or not task_context:
            raise ValueError("task_context must describe the fixed physical task")
        context = json.dumps(dict(task_context), allow_nan=False, sort_keys=True, separators=(",", ":"))
        groups = tuple(required_constraint_groups)
        if (not groups or len(set(groups)) != len(groups)
                or any(not isinstance(group, str) or not group for group in groups)
                or not set(DEFAULT_CONSTRAINT_GROUPS).issubset(groups)):
            raise ValueError("required_constraint_groups must include every default RHO constraint group")
        return cls(str(archive_path), sha256(archive_path.read_bytes()).hexdigest(),
                   checked["prepared_problem_sha256"], completed_cycles, str(model_path),
                   sha256(model_path.read_bytes()).hexdigest(), context, _digest_json(dict(task_context)), groups)

    def verify_files(self) -> None:
        """Fail if source bytes changed after a request was constructed."""
        if sha256(Path(self.archive_path).read_bytes()).hexdigest() != self.archive_sha256:
            raise ValueError("Task-reserve checkpoint archive changed")
        if sha256(Path(self.model_path).read_bytes()).hexdigest() != self.model_sha256:
            raise ValueError("Task-reserve model changed")
        if _digest_json(json.loads(self.task_context_json)) != self.task_context_sha256:
            raise ValueError("Task-reserve context digest differs")


@dataclass(frozen=True)
class TaskReserveProbe:
    checkpoint: TaskReserveCheckpoint
    work_scale: float
    metric: str = WORK_RESERVE_METRIC

    def __post_init__(self):
        _positive(self.work_scale, "work_scale")
        if self.metric != WORK_RESERVE_METRIC:
            raise ValueError("Only one_cycle_work_scale is implemented; PW saturation is a future audit")


@dataclass(frozen=True)
class ProbeEvidence:
    """Numerical witness report returned by the *full RHO* oracle.

    Residuals must include dynamics, every active bound, initial state/history,
    closure and target work, normalized by the oracle's documented physical
    scales. A solver exit string is recorded but does not decide feasibility.
    ``witness_id`` should identify the saved trajectory/solution artifact.
    """

    work_scale: float
    prepared_problem_sha256: str
    task_context_sha256: str
    model_sha256: str
    solver_status: str
    maximum_normalized_constraint_violation: float | None
    tolerance: float
    independent_validation_passed: bool
    witness_id: str | None
    elapsed_seconds: float
    validated_constraint_groups: tuple[str, ...] = ()
    source_state_and_history_restored: bool = False
    reason: str | None = None

    def witness_is_valid(self, request: TaskReserveProbe) -> bool:
        source = request.checkpoint
        if (self.work_scale != request.work_scale
                or self.prepared_problem_sha256 != source.prepared_problem_sha256
                or self.task_context_sha256 != source.task_context_sha256
                or self.model_sha256 != source.model_sha256):
            raise ValueError("Probe evidence belongs to a different checkpoint, task, model or load")
        _positive(self.tolerance, "tolerance")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        residual = self.maximum_normalized_constraint_violation
        finite_residual = (residual is not None and math.isfinite(residual) and residual >= 0.)
        return bool(self.independent_validation_passed is True
                    and self.source_state_and_history_restored is True
                    and isinstance(self.witness_id, str) and self.witness_id
                    and finite_residual and residual <= self.tolerance
                    and set(source.required_constraint_groups).issubset(self.validated_constraint_groups))


@dataclass(frozen=True)
class ReserveEstimate:
    """Best *observed* feasible reserve; unknown global upper bound stays None."""

    checkpoint: TaskReserveCheckpoint
    evidence: tuple[ProbeEvidence, ...]
    feasible_work_scales: tuple[float, ...]
    nominal_task_witnessed: bool
    work_scale_lower_bound: float | None
    reserve_lower_bound: float | None
    status: str
    metric: str = WORK_RESERVE_METRIC
    global_upper_bound: None = None
    physiological_failure_certified: bool = False

    @property
    def usable_for_local_fit(self) -> bool:
        return self.nominal_task_witnessed and self.reserve_lower_bound is not None


def evaluate_work_reserve(checkpoint: TaskReserveCheckpoint, work_scales: Sequence[float],
                          evaluator: Callable[[TaskReserveProbe], ProbeEvidence]) -> ReserveEstimate:
    """Evaluate explicit factors from one frozen checkpoint, nominal first.

    No bisection or stopping at the first failed solve: neither assumes a
    nonconvex optimizer is a feasibility oracle. Exceptions from the evaluator
    propagate so a programming/provenance error cannot become fatigue failure.
    The evaluator owns fresh restoration and any deadline/CPU isolation.
    """
    factors = tuple(_positive(value, "work_scale") for value in work_scales)
    if not factors or len(set(factors)) != len(factors) or 1. not in factors:
        raise ValueError("work_scales must be distinct and explicitly include nominal factor 1")
    factors = (1.,) + tuple(value for value in factors if value != 1.)
    evidence, feasible = [], []
    for factor in factors:
        checkpoint.verify_files()
        request = TaskReserveProbe(checkpoint, factor)
        report = evaluator(request)
        if not isinstance(report, ProbeEvidence):
            raise TypeError("The full-RHO evaluator must return ProbeEvidence")
        checkpoint.verify_files()
        if report.witness_is_valid(request):
            feasible.append(factor)
        evidence.append(report)
    best = max(feasible) if feasible else None
    nominal = 1. in feasible
    status = ("nominal_and_reserve_witnessed" if nominal and best > 1.
              else "nominal_witnessed" if nominal
              else "nominal_undetermined")
    return ReserveEstimate(checkpoint, tuple(evidence), tuple(feasible), nominal,
                           best, None if best is None else best - 1., status)


@dataclass(frozen=True)
class LocalReserveSample:
    coordinates: tuple[float, ...]
    estimate: ReserveEstimate


@dataclass(frozen=True)
class LocalReserveModel:
    """Empirical affine reserve, valid only in a normalized-state trust box.

    The bias correction removes the largest observed optimistic residual in
    training/holdout data. This is an empirical guard, not a certified lower
    envelope between samples. A full-RHO continuation must validate any cost
    using this model before an endurance claim is made.
    """

    center: tuple[float, ...]
    gradient: tuple[float, ...]
    intercept: float
    trust_radius: tuple[float, ...]
    accepted: bool
    reasons: tuple[str, ...]
    sample_count: int
    holdout_count: int
    rank: int
    condition_number: float
    training_maximum_error: float
    holdout_maximum_error: float
    empirical_bias_correction: float
    task_context_sha256: str
    model_sha256: str
    globally_certified: bool = False

    def parameter_vector(self) -> np.ndarray:
        """Fixed layout: [intercept, center(n), gradient(n)], for graph reuse."""
        if not self.accepted:
            raise ValueError(f"Local reserve model was rejected: {', '.join(self.reasons)}")
        return np.r_[self.intercept, self.center, self.gradient]

    def margin(self, coordinates: Sequence[float]) -> float:
        values = np.asarray(coordinates, dtype=float)
        if values.shape != (len(self.center),) or not np.all(np.isfinite(values)):
            raise ValueError("coordinates have the wrong dimension or contain non-finite values")
        if np.any(np.abs(values - self.center) > np.asarray(self.trust_radius) + 1e-14):
            raise ValueError("Local reserve model used outside its normalized-state trust box")
        self.parameter_vector()
        return float(self.intercept + np.asarray(self.gradient) @ (values - self.center))


def fit_local_reserve(samples: Sequence[LocalReserveSample], *, center: Sequence[float],
                      trust_radius: Sequence[float], holdout: Sequence[LocalReserveSample],
                      maximum_error: float = .01, maximum_condition_number: float = 1e6) -> LocalReserveModel:
    """Fit and independently audit an affine observed-reserve surrogate.

    All coordinates must have the same declared nondimensional convention.
    The full state/history for each coordinate is in that sample's checkpoint.
    This routine never fabricates a sample by perturbing fatigue while silently
    resetting calcium, force, previous PW or mechanical states.
    """
    center, trust = np.asarray(center, float), np.asarray(trust_radius, float)
    if (center.ndim != 1 or not center.size or trust.shape != center.shape
            or not np.all(np.isfinite(center)) or not np.all(np.isfinite(trust)) or np.any(trust <= 0)):
        raise ValueError("center/trust_radius must be finite same-sized vectors with positive trust radii")
    maximum_error = _positive(maximum_error, "maximum_error")
    maximum_condition_number = _positive(maximum_condition_number, "maximum_condition_number")
    samples, holdout = tuple(samples), tuple(holdout)
    if not samples:
        raise ValueError("At least one observed reserve sample is needed")
    source = samples[0].estimate.checkpoint
    context, model = source.task_context_sha256, source.model_sha256
    reasons = []
    checkpoint_coordinates = {}

    def rows(items):
        design, values = [], []
        for sample in items:
            estimate, point = sample.estimate, np.asarray(sample.coordinates, float)
            if (estimate.checkpoint.task_context_sha256 != context
                    or estimate.checkpoint.model_sha256 != model or estimate.metric != WORK_RESERVE_METRIC):
                raise ValueError("Local fit samples mix physical tasks, models or reserve metrics")
            if point.shape != center.shape or not np.all(np.isfinite(point)):
                raise ValueError("Sample coordinates differ from the normalization dimension")
            if np.any(np.abs(point - center) > trust + 1e-14):
                raise ValueError("Sample lies outside the requested trust box")
            if not estimate.usable_for_local_fit:
                raise ValueError("Every fit/holdout sample needs an independently witnessed nominal task")
            digest = estimate.checkpoint.prepared_problem_sha256
            previous = checkpoint_coordinates.setdefault(digest, tuple(point))
            if previous != tuple(point):
                raise ValueError("Different state coordinates cannot refer to the same prepared checkpoint")
            design.append(np.r_[1., (point - center) / trust])
            values.append(estimate.reserve_lower_bound)
        return np.asarray(design).reshape((-1, center.size + 1)), np.asarray(values)

    matrix, values = rows(samples)
    test_matrix, test_values = rows(holdout)
    train_points = {tuple(sample.coordinates) for sample in samples}
    if any(tuple(sample.coordinates) in train_points for sample in holdout):
        raise ValueError("Holdout coordinates must be independent of the training coordinates")
    if len({tuple(sample.coordinates) for sample in holdout}) != len(holdout):
        raise ValueError("Holdout coordinates must not be repeated")
    coefficients, _, rank, singular = np.linalg.lstsq(matrix, values, rcond=None)
    condition = float(singular[0] / singular[-1]) if rank == center.size + 1 else math.inf
    if rank != center.size + 1:
        reasons.append("rank_deficient")
    if condition > maximum_condition_number:
        reasons.append("ill_conditioned")
    # The design must identify an intercept and every requested state direction.
    training_error = matrix @ coefficients - values
    holdout_error = test_matrix @ coefficients - test_values
    bias = max(0., float(np.max(training_error)), float(np.max(holdout_error)) if holdout else 0.)
    # Audit the function actually published, including the empirical shift.
    train_max = float(np.max(np.abs(training_error - bias)))
    test_max = float(np.max(np.abs(holdout_error - bias))) if holdout else math.inf
    if not holdout:
        reasons.append("no_independent_holdout")
    if train_max > maximum_error:
        reasons.append("training_error_exceeds_tolerance")
    if test_max > maximum_error:
        reasons.append("holdout_error_exceeds_tolerance")
    return LocalReserveModel(tuple(center), tuple(coefficients[1:] / trust),
                             float(coefficients[0] - bias), tuple(trust), not reasons,
                             tuple(reasons), len(samples), len(holdout), int(rank), condition,
                             train_max, test_max, bias, context, model)


def reserve_penalty_expression(coordinates, parameters, *, dimension: int, target: float,
                               smoothing: float = 1e-3, sqrt=None):
    """Smooth squared shortage cost for NumPy or CasADi scalar expressions.

    Use a CasADi symbolic parameter vector of length ``1 + 2 * dimension``
    and ``sqrt=casadi.sqrt`` to compile once and update only numerical values.
    Trust/age checking belongs to the supervisor, outside the RHO graph.
    No production RHO objective is bound by calling this standalone helper.
    """
    if type(dimension) is not int or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if not math.isfinite(float(target)):
        raise ValueError("target must be finite")
    smoothing = _positive(smoothing, "smoothing")
    sqrt = np.sqrt if sqrt is None else sqrt
    margin = parameters[0]
    for index in range(dimension):
        margin = margin + parameters[1 + dimension + index] * (coordinates[index] - parameters[1 + index])
    shortage = float(target) - margin
    smooth_positive_part = .5 * (shortage + sqrt(shortage * shortage + smoothing * smoothing))
    return smooth_positive_part * smooth_positive_part
