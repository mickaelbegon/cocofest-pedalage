r"""Bounded slow supervisor for relative muscle weights.

The fast controller is deliberately outside this module.  A caller takes an
immutable :class:`SupervisorSnapshot`, evaluates the small, predeclared set of
weight candidates in a slow worker, and hands the resulting immutable
:class:`WeightProposal` back to the fast loop.  The fast loop may continue to
use its incumbent weights while evaluation is in progress and must call
:func:`check_proposal_eligibility` before applying a returned proposal.

The callback boundary is predictor independent.  In particular, this module
does not import or run a full-horizon optimization.  An incomplete rollout is
evidence only for its explicitly completed prefix; it is never assigned a
finite surrogate penalty that could make it look like a completed horizon.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time
from typing import Any


def _strict_nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a non-negative integer.") from error
    if integer != value or integer < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return integer


def _strict_positive_integer(value: Any, *, name: str) -> int:
    integer = _strict_nonnegative_integer(value, name=name)
    if integer == 0:
        raise ValueError(f"{name} must be a strictly positive integer.")
    return integer


def _finite(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _nonnegative_finite(value: Any, *, name: str) -> float:
    number = _finite(value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return number


def _positive_finite(value: Any, *, name: str) -> float:
    number = _finite(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    return number


def _tuple_of_strings(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or any(not value for value in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique, non-empty names.")
    return result


def normalize_relative_weights(weights: Sequence[float]) -> tuple[float, ...]:
    """Remove the unidentifiable common scale using the geometric mean."""

    values = tuple(_positive_finite(value, name="weight") for value in weights)
    if not values:
        raise ValueError("weights must not be empty.")
    mean_log = math.fsum(math.log(value) for value in values) / len(values)
    try:
        normalized = tuple(math.exp(math.log(value) - mean_log) for value in values)
    except OverflowError as error:
        raise ValueError("weights have an unrepresentable relative dynamic range.") from error
    if any(not math.isfinite(value) or value <= 0.0 for value in normalized):
        raise ValueError("weights have an unrepresentable relative dynamic range.")
    return normalized


@dataclass(frozen=True)
class SupervisorSnapshot:
    """Immutable context captured at a cycle boundary for a slow evaluation."""

    task_id: str
    context_token: str
    cycle_index: int
    start_time_s: float
    created_at_s: float
    horizon_cycles: int
    muscle_names: tuple[str, ...]
    state_component_names: tuple[str, ...]
    start_state: tuple[tuple[float, ...], ...]
    incumbent_weights: tuple[float, ...]
    at_cycle_boundary: bool = True

    def __post_init__(self) -> None:
        task_id = str(self.task_id)
        context_token = str(self.context_token)
        if not task_id:
            raise ValueError("task_id must not be empty.")
        if not context_token:
            raise ValueError("context_token must not be empty.")
        muscle_names = _tuple_of_strings(self.muscle_names, name="muscle_names")
        component_names = _tuple_of_strings(
            self.state_component_names, name="state_component_names"
        )
        state = tuple(
            tuple(_finite(value, name="start_state value") for value in row)
            for row in self.start_state
        )
        if len(state) != len(muscle_names):
            raise ValueError("start_state must have one row per muscle.")
        if any(len(row) != len(component_names) for row in state):
            raise ValueError("start_state rows must match state_component_names.")
        incumbent = normalize_relative_weights(self.incumbent_weights)
        if len(incumbent) != len(muscle_names):
            raise ValueError("incumbent_weights must have one entry per muscle.")
        if not isinstance(self.at_cycle_boundary, bool):
            raise ValueError("at_cycle_boundary must be boolean.")

        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "context_token", context_token)
        object.__setattr__(
            self, "cycle_index", _strict_nonnegative_integer(self.cycle_index, name="cycle_index")
        )
        object.__setattr__(self, "start_time_s", _finite(self.start_time_s, name="start_time_s"))
        object.__setattr__(self, "created_at_s", _finite(self.created_at_s, name="created_at_s"))
        object.__setattr__(
            self,
            "horizon_cycles",
            _strict_positive_integer(self.horizon_cycles, name="horizon_cycles"),
        )
        object.__setattr__(self, "muscle_names", muscle_names)
        object.__setattr__(self, "state_component_names", component_names)
        object.__setattr__(self, "start_state", state)
        object.__setattr__(self, "incumbent_weights", incumbent)

    @property
    def fingerprint(self) -> str:
        """Stable digest covering every source-context field."""

        payload = {
            "task_id": self.task_id,
            "context_token": self.context_token,
            "cycle_index": self.cycle_index,
            "start_time_s": self.start_time_s,
            "created_at_s": self.created_at_s,
            "horizon_cycles": self.horizon_cycles,
            "muscle_names": self.muscle_names,
            "state_component_names": self.state_component_names,
            "start_state": self.start_state,
            "incumbent_weights": self.incumbent_weights,
            "at_cycle_boundary": self.at_cycle_boundary,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WeightSupervisorConfig:
    """Declared candidate set and admissible relative-weight box."""

    min_weight: float = 0.25
    max_weight: float = 4.0
    adjustment_factor: float = 1.5
    muscle_count: int = 4
    selection_mode: str = "exact_prefix"
    max_candidate_log_step: float | None = None
    minimum_relative_improvement: float = 0.01
    minimum_absolute_improvement: float = 1e-6
    fatigue_noninferiority_tolerance: float = 1e-10
    capacity_noninferiority_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        minimum = _positive_finite(self.min_weight, name="min_weight")
        maximum = _positive_finite(self.max_weight, name="max_weight")
        factor = _positive_finite(self.adjustment_factor, name="adjustment_factor")
        count = _strict_positive_integer(self.muscle_count, name="muscle_count")
        if self.selection_mode not in ("exact_prefix", "full_horizon_deficit", "guarded_fatigue"):
            raise ValueError("selection_mode must be exact_prefix, full_horizon_deficit or guarded_fatigue.")
        if self.max_candidate_log_step is not None:
            _positive_finite(self.max_candidate_log_step, name="max_candidate_log_step")
        for name in ("minimum_relative_improvement", "minimum_absolute_improvement",
                     "fatigue_noninferiority_tolerance", "capacity_noninferiority_tolerance"):
            _nonnegative_finite(getattr(self, name), name=name)
        if minimum > 1.0 or maximum < 1.0 or minimum > maximum:
            raise ValueError("Relative-weight bounds must contain 1.0.")
        if factor <= 1.0:
            raise ValueError("adjustment_factor must be greater than 1.0.")
        object.__setattr__(self, "min_weight", minimum)
        object.__setattr__(self, "max_weight", maximum)
        object.__setattr__(self, "adjustment_factor", factor)
        object.__setattr__(self, "muscle_count", count)


@dataclass(frozen=True)
class WeightCandidate:
    index: int
    kind: str
    muscle_name: str | None
    log_step: float
    weights: tuple[float, ...]


@dataclass(frozen=True)
class CandidateRolloutResult:
    """Only the evidence the supervisor is permitted to compare."""

    completed_duration: float
    completed: bool
    feasible: bool
    minimum_signed_margin: float | None
    status: str
    prefix_comparable: bool = False
    full_horizon_normalized_deficit: float | None = None
    terminal_normalized_reserve: float | None = None
    full_horizon_mean_squared_fatigue: float | None = None
    first_block_mean_squared_fatigue: float | None = None
    terminal_minimum_capacity: float | None = None

    def __post_init__(self) -> None:
        duration = _nonnegative_finite(self.completed_duration, name="completed_duration")
        if (
            not isinstance(self.completed, bool)
            or not isinstance(self.feasible, bool)
            or not isinstance(self.prefix_comparable, bool)
        ):
            raise ValueError("completed, feasible, and prefix_comparable must be boolean.")
        if self.feasible and not self.completed:
            raise ValueError("A feasible horizon must be completed.")
        margin = self.minimum_signed_margin
        if margin is not None:
            margin = _finite(margin, name="minimum_signed_margin")
        if self.feasible and margin is None:
            raise ValueError("A feasible completed horizon requires a signed margin.")
        score = self.full_horizon_normalized_deficit
        reserve = self.terminal_normalized_reserve
        if score is not None:
            score = _nonnegative_finite(score, name="full_horizon_normalized_deficit")
            if not self.completed or reserve is None:
                raise ValueError("A long-horizon score requires a completed simulation and terminal reserve.")
            reserve = _finite(reserve, name="terminal_normalized_reserve")
        status = str(self.status)
        if not status:
            raise ValueError("status must not be empty.")
        object.__setattr__(self, "completed_duration", duration)
        object.__setattr__(self, "minimum_signed_margin", margin)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "full_horizon_normalized_deficit", score)
        object.__setattr__(self, "terminal_normalized_reserve", reserve)
        for name in ("full_horizon_mean_squared_fatigue", "first_block_mean_squared_fatigue",
                     "terminal_minimum_capacity"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonnegative_finite(value, name=name))


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: WeightCandidate
    result: CandidateRolloutResult | None
    elapsed_after_s: float
    error_type: str | None = None
    error_message: str | None = None

    @property
    def valid(self) -> bool:
        return self.result is not None

    @property
    def comparable(self) -> bool:
        return self.result is not None and (
            (self.result.completed and self.result.feasible) or self.result.prefix_comparable
            or self.result.full_horizon_normalized_deficit is not None
        )


@dataclass(frozen=True)
class WeightProposal:
    """Atomic, immutable handoff payload; it does not apply itself."""

    source_snapshot: SupervisorSnapshot
    source_fingerprint: str
    weights: tuple[float, ...]
    chosen_candidate: WeightCandidate | None
    chosen_evidence: CandidateRolloutResult | None
    evaluations: tuple[CandidateEvaluation, ...]
    issued_at_s: float
    elapsed_s: float
    budget_seconds: float
    budget_exhausted: bool
    budget_overrun_s: float
    hard_timeout_enforced: bool
    selection_basis: str
    has_actionable_evidence: bool


@dataclass(frozen=True)
class ProposalEligibility:
    eligible: bool
    reasons: tuple[str, ...]
    age_s: float
    source_age_s: float
    cycle_lag: int
    maximum_state_ratios: tuple[float, ...]
    maximum_log_weight_difference: float


def _project_centered_logs(
    logs: Sequence[float], *, lower: float, upper: float
) -> tuple[float, ...]:
    """Euclidean projection onto ``sum(log(w))=0`` and box constraints."""

    values = tuple(float(value) for value in logs)
    low_lambda = min(value - upper for value in values)
    high_lambda = max(value - lower for value in values)
    for _ in range(100):
        midpoint = (low_lambda + high_lambda) / 2.0
        total = math.fsum(min(upper, max(lower, value - midpoint)) for value in values)
        if total > 0.0:
            low_lambda = midpoint
        else:
            high_lambda = midpoint
    shift = (low_lambda + high_lambda) / 2.0
    projected = [min(upper, max(lower, value - shift)) for value in values]
    # Remove the final floating-point residual without changing declaration order.
    residual = math.fsum(projected)
    if residual:
        for index, value in enumerate(projected):
            room = value - lower if residual > 0.0 else upper - value
            correction = math.copysign(min(abs(residual), room), residual)
            projected[index] -= correction
            residual -= correction
            if residual == 0.0:
                break
    return tuple(projected)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _coerce_rollout_result(value: Any) -> CandidateRolloutResult:
    if isinstance(value, CandidateRolloutResult):
        return value
    duration = _field(value, "completed_duration", _field(value, "completed_duration_s"))
    if duration is None:
        raise ValueError("Evaluator result is missing completed_duration.")
    raw_status = _field(value, "status", "unspecified")
    status = str(getattr(raw_status, "value", raw_status))
    completed_value = _field(value, "completed")
    completed = status.lower() == "complete" if completed_value is None else completed_value
    feasible_value = _field(value, "feasible")
    feasible = (bool(completed) and status.lower() == "complete") if feasible_value is None else feasible_value
    margin = _field(value, "minimum_signed_margin", _field(value, "min_signed_margin"))
    first_failure = _field(value, "first_failure")
    failure_status = _field(first_failure, "status") if first_failure is not None else None
    if failure_status is not None:
        status = str(failure_status)
    prefix_comparable = _field(value, "prefix_comparable")
    if prefix_comparable is None:
        # The weighted-cycle predictor reserves these statuses for an actual
        # reachability-bound failure.  Numerical and model-domain refusals are
        # deliberately fail-closed unless an adapter explicitly declares its
        # own prefix semantics.
        prefix_comparable = bool(
            not completed
            and failure_status is not None
            and str(failure_status).startswith("infeasible_total_")
        )
    return CandidateRolloutResult(
        completed_duration=duration,
        completed=completed,
        feasible=feasible,
        minimum_signed_margin=margin,
        status=status,
        prefix_comparable=prefix_comparable,
        full_horizon_normalized_deficit=_field(value, "full_horizon_normalized_deficit"),
        terminal_normalized_reserve=_field(value, "terminal_normalized_reserve"),
        full_horizon_mean_squared_fatigue=_field(value, "full_horizon_mean_squared_fatigue"),
        first_block_mean_squared_fatigue=_field(value, "first_block_mean_squared_fatigue"),
        terminal_minimum_capacity=_field(value, "terminal_minimum_capacity"),
    )


class EnduranceWeightSupervisor:
    """Evaluate a fixed, bounded set of relative-weight candidates."""

    def __init__(self, config: WeightSupervisorConfig | None = None):
        self.config = WeightSupervisorConfig() if config is None else config

    def candidates(self, snapshot: SupervisorSnapshot) -> tuple[WeightCandidate, ...]:
        if len(snapshot.muscle_names) != self.config.muscle_count:
            raise ValueError(
                f"Expected {self.config.muscle_count} muscles, got {len(snapshot.muscle_names)}."
            )
        tolerance = 32.0 * math.ulp(1.0)
        if any(
            weight < self.config.min_weight - tolerance
            or weight > self.config.max_weight + tolerance
            for weight in snapshot.incumbent_weights
        ):
            raise ValueError("incumbent_weights are outside the declared weight bounds.")

        candidates: list[WeightCandidate] = []
        seen: set[tuple[float, ...]] = set()

        def append(kind: str, muscle_name: str | None, log_step: float, weights: Sequence[float]):
            if kind != "incumbent" and self.config.max_candidate_log_step is not None:
                # Bound BEFORE evaluating: applying an unevaluated interpolation
                # afterward would invalidate every claimed rollout improvement.
                delta = tuple(math.log(w / base) for w, base in zip(weights, snapshot.incumbent_weights))
                scale = min(1., self.config.max_candidate_log_step / max(max(map(abs, delta)), 1e-300))
                weights = tuple(base * math.exp(scale * d)
                                for base, d in zip(snapshot.incumbent_weights, delta))
            key = tuple(round(float(value), 14) for value in weights)
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                WeightCandidate(
                    index=len(candidates),
                    kind=kind,
                    muscle_name=muscle_name,
                    log_step=float(log_step),
                    weights=tuple(float(value) for value in weights),
                )
            )

        incumbent = snapshot.incumbent_weights
        append("incumbent", None, 0.0, incumbent)
        append("all_ones_reference", None, 0.0, (1.0,) * self.config.muscle_count)

        lower = math.log(self.config.min_weight)
        upper = math.log(self.config.max_weight)
        base_logs = tuple(math.log(value) for value in incumbent)
        step = math.log(self.config.adjustment_factor)
        for muscle_index, muscle_name in enumerate(snapshot.muscle_names):
            for sign, kind in ((1.0, "increase"), (-1.0, "decrease")):
                moved = list(base_logs)
                moved[muscle_index] += sign * step
                projected = _project_centered_logs(moved, lower=lower, upper=upper)
                append(
                    kind,
                    muscle_name,
                    sign * step,
                    tuple(math.exp(value) for value in projected),
                )
        return tuple(candidates)

    def evaluate(
        self,
        snapshot: SupervisorSnapshot,
        evaluator: Callable[[tuple[float, ...], SupervisorSnapshot], Any],
        *,
        budget_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> WeightProposal:
        """Evaluate whole candidates, checking the budget only between calls.

        The callback is not interrupted.  Consequently ``budget_overrun_s`` may
        be positive and ``hard_timeout_enforced`` is always false.
        """

        budget = _nonnegative_finite(budget_seconds, name="budget_seconds")
        declared = self.candidates(snapshot)
        started_at = _finite(clock(), name="clock value")
        last_time = started_at
        evaluations: list[CandidateEvaluation] = []
        budget_exhausted = False

        for candidate in declared:
            if evaluations and last_time - started_at >= budget:
                budget_exhausted = True
                break
            try:
                result = _coerce_rollout_result(evaluator(candidate.weights, snapshot))
                error_type = None
                error_message = None
            except Exception as error:  # one contaminated candidate must not contaminate ranking
                result = None
                error_type = type(error).__name__
                error_message = str(error)
            evaluation_end = _finite(clock(), name="clock value")
            if evaluation_end < last_time:
                raise ValueError("clock must be monotonic during evaluation.")
            last_time = evaluation_end
            evaluations.append(
                CandidateEvaluation(
                    candidate=candidate,
                    result=result,
                    elapsed_after_s=last_time - started_at,
                    error_type=error_type,
                    error_message=error_message,
                )
            )

        comparable = [record for record in evaluations if record.comparable]
        # Compare long capacity simulations only over the same fully evaluated
        # duration as the incumbent. An uncomputed tail has no score.
        incumbent_record = next((r for r in evaluations if r.candidate.kind == "incumbent"), None)
        guarded = self.config.selection_mode == "guarded_fatigue"
        long_mode = (self.config.selection_mode in ("full_horizon_deficit", "guarded_fatigue") or
                     any(r.result is not None and r.result.full_horizon_normalized_deficit is not None
                         for r in evaluations))
        if long_mode:
            incumbent_result = None if incumbent_record is None else incumbent_record.result
            if incumbent_result is None or incumbent_result.full_horizon_normalized_deficit is None:
                comparable = []
            else:
                comparable = [r for r in comparable if r.result.full_horizon_normalized_deficit is not None
                              and math.isclose(r.result.completed_duration, incumbent_result.completed_duration,
                                               rel_tol=1e-10, abs_tol=1e-10)]
        if guarded and comparable:
            comparable = [record for record in comparable if self._passes_fatigue_guard(
                record, incumbent_record)]
        if comparable:
            def rank(record: CandidateEvaluation) -> tuple[float, ...]:
                assert record.result is not None
                result = record.result
                full = result.completed and result.feasible
                margin = result.minimum_signed_margin if full else 0.0
                duration = 0.0 if full else result.completed_duration
                assert margin is not None
                change = math.fsum(
                    (math.log(weight / incumbent)) ** 2
                    for weight, incumbent in zip(
                        record.candidate.weights, snapshot.incumbent_weights, strict=True
                    )
                )
                if long_mode:
                    if guarded:
                        return (-round(result.full_horizon_normalized_deficit, 12),
                                -round(result.full_horizon_mean_squared_fatigue, 12),
                                float(record.candidate.kind == "incumbent"), -change,
                                -float(record.candidate.index))
                    return (-round(result.full_horizon_normalized_deficit, 12),
                            round(result.terminal_normalized_reserve, 12),
                            float(record.candidate.kind == "incumbent"), -change,
                            -float(record.candidate.index))
                return (
                    float(full),
                    duration,
                    margin,
                    float(record.candidate.kind == "incumbent"),
                    -change,
                    -float(record.candidate.index),
                )

            chosen_record = max(comparable, key=rank)
            chosen = chosen_record.candidate
            evidence = chosen_record.result
            assert evidence is not None
            full_horizon = evidence.completed and evidence.feasible
            selection_basis = (
                ("guarded_hold_incumbent" if chosen.kind == "incumbent" else "guarded_fatigue_improvement")
                if guarded else
                "full_horizon_normalized_moment_deficit" if long_mode else
                "completed_horizon_margin" if full_horizon else "longest_explicit_prefix"
            )
            actionable = True
        else:
            chosen = None
            evidence = None
            selection_basis = "no_valid_evaluation"
            actionable = False

        elapsed = last_time - started_at
        return WeightProposal(
            source_snapshot=snapshot,
            source_fingerprint=snapshot.fingerprint,
            weights=snapshot.incumbent_weights if chosen is None else chosen.weights,
            chosen_candidate=chosen,
            chosen_evidence=evidence,
            evaluations=tuple(evaluations),
            issued_at_s=last_time,
            elapsed_s=elapsed,
            budget_seconds=budget,
            budget_exhausted=budget_exhausted,
            budget_overrun_s=max(0.0, elapsed - budget),
            hard_timeout_enforced=False,
            selection_basis=selection_basis,
            has_actionable_evidence=actionable,
        )

    def _passes_fatigue_guard(self, record, incumbent_record):
        """Require material gain with no predicted damage to fatigue or reserve.

        Reserve-only improvements are insufficient: a tiny numerical deficit
        change or a better torque envelope must not buy increased fatigue.
        This is a surrogate safeguard, not a real-system endurance proof.
        """
        result, incumbent = record.result, incumbent_record.result
        names = ("full_horizon_mean_squared_fatigue", "first_block_mean_squared_fatigue",
                 "terminal_minimum_capacity")
        if any(getattr(value, name) is None for value in (result, incumbent) for name in names):
            return False
        if record.candidate.kind == "incumbent":
            return True
        cfg = self.config
        if any(getattr(result, name) > getattr(incumbent, name) + cfg.fatigue_noninferiority_tolerance
               for name in names[:2]):
            return False
        if result.terminal_minimum_capacity + cfg.capacity_noninferiority_tolerance < incumbent.terminal_minimum_capacity:
            return False
        if result.full_horizon_normalized_deficit > incumbent.full_horizon_normalized_deficit + 1e-12:
            return False
        def material_gain(name):
            previous, proposed = getattr(incumbent, name), getattr(result, name)
            return previous - proposed > max(cfg.minimum_absolute_improvement,
                                              cfg.minimum_relative_improvement * abs(previous))
        return material_gain("full_horizon_normalized_deficit") or material_gain("full_horizon_mean_squared_fatigue")


def _declared_tolerances(
    values: Mapping[str, float] | Sequence[float], names: tuple[str, ...]
) -> tuple[float, ...]:
    if isinstance(values, Mapping):
        missing = set(names) - set(values)
        unknown = set(values) - set(names)
        if missing or unknown:
            raise ValueError(
                f"state_component_tolerances names differ: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}."
            )
        ordered = tuple(values[name] for name in names)
    else:
        ordered = tuple(values)
        if len(ordered) != len(names):
            raise ValueError("state_component_tolerances must have one value per component.")
    return tuple(
        _nonnegative_finite(value, name=f"state tolerance for {name}")
        for name, value in zip(names, ordered, strict=True)
    )


def check_proposal_eligibility(
    proposal: WeightProposal,
    current_snapshot: SupervisorSnapshot,
    *,
    now_s: float,
    max_age_s: float,
    max_cycle_lag: int,
    start_time_tolerance_s: float,
    state_component_tolerances: Mapping[str, float] | Sequence[float],
    log_weight_tolerance: float,
) -> ProposalEligibility:
    """Check whether an asynchronous proposal still matches current context.

    ``source_snapshot.created_at_s``, ``proposal.issued_at_s``, and ``now_s``
    must share one monotonic clock origin.  ``start_time_s`` remains the
    separately checked task/simulation time.  All tolerances are supplied by
    the caller; this module has no hidden physiological acceptance threshold.
    """

    now = _finite(now_s, name="now_s")
    maximum_age = _nonnegative_finite(max_age_s, name="max_age_s")
    allowed_lag = _strict_nonnegative_integer(max_cycle_lag, name="max_cycle_lag")
    time_tolerance = _nonnegative_finite(
        start_time_tolerance_s, name="start_time_tolerance_s"
    )
    weight_tolerance = _nonnegative_finite(
        log_weight_tolerance, name="log_weight_tolerance"
    )
    source = proposal.source_snapshot
    tolerances = _declared_tolerances(state_component_tolerances, source.state_component_names)

    reasons: list[str] = []
    age = now - proposal.issued_at_s
    source_age = now - source.created_at_s
    cycle_lag = current_snapshot.cycle_index - source.cycle_index
    if proposal.source_fingerprint != source.fingerprint:
        reasons.append("source_fingerprint_mismatch")
    if not proposal.has_actionable_evidence:
        reasons.append("no_actionable_evidence")
    if not source.at_cycle_boundary or not current_snapshot.at_cycle_boundary:
        reasons.append("not_at_cycle_boundary")
    if source.task_id != current_snapshot.task_id:
        reasons.append("task_id_mismatch")
    if source.context_token != current_snapshot.context_token:
        reasons.append("context_token_mismatch")
    if source.horizon_cycles != current_snapshot.horizon_cycles:
        reasons.append("horizon_mismatch")
    if source.muscle_names != current_snapshot.muscle_names:
        reasons.append("muscle_names_mismatch")
    if source.state_component_names != current_snapshot.state_component_names:
        reasons.append("state_component_names_mismatch")
    if cycle_lag < 0:
        reasons.append("current_cycle_precedes_source")
    elif cycle_lag > allowed_lag:
        reasons.append("cycle_lag_exceeded")
    if age < 0.0:
        reasons.append("proposal_from_future")
    elif age > maximum_age:
        reasons.append("proposal_too_old")
    if source_age < 0.0:
        reasons.append("source_snapshot_from_future")
    elif source_age > maximum_age:
        reasons.append("source_snapshot_too_old")
    if abs(current_snapshot.start_time_s - source.start_time_s) > time_tolerance:
        reasons.append("start_time_mismatch")

    ratios: list[float] = []
    if (
        source.muscle_names == current_snapshot.muscle_names
        and source.state_component_names == current_snapshot.state_component_names
    ):
        for component_index, tolerance in enumerate(tolerances):
            maximum_difference = max(
                abs(current_snapshot.start_state[muscle_index][component_index] - row[component_index])
                for muscle_index, row in enumerate(source.start_state)
            )
            if tolerance == 0.0:
                ratio = 0.0 if maximum_difference == 0.0 else math.inf
            else:
                ratio = maximum_difference / tolerance
            ratios.append(ratio)
            if maximum_difference > tolerance:
                reasons.append(f"state_component_mismatch:{source.state_component_names[component_index]}")
    else:
        ratios = [math.inf] * len(source.state_component_names)

    if len(source.incumbent_weights) == len(current_snapshot.incumbent_weights):
        maximum_log_weight_difference = max(
            abs(math.log(current / original))
            for current, original in zip(
                current_snapshot.incumbent_weights, source.incumbent_weights, strict=True
            )
        )
        if maximum_log_weight_difference > weight_tolerance:
            reasons.append("incumbent_weights_mismatch")
    else:
        maximum_log_weight_difference = math.inf
        reasons.append("incumbent_weight_shape_mismatch")

    return ProposalEligibility(
        eligible=not reasons,
        reasons=tuple(reasons),
        age_s=age,
        source_age_s=source_age,
        cycle_lag=cycle_lag,
        maximum_state_ratios=tuple(ratios),
        maximum_log_weight_difference=maximum_log_weight_difference,
    )


__all__ = [
    "CandidateEvaluation",
    "CandidateRolloutResult",
    "EnduranceWeightSupervisor",
    "ProposalEligibility",
    "SupervisorSnapshot",
    "WeightCandidate",
    "WeightProposal",
    "WeightSupervisorConfig",
    "check_proposal_eligibility",
    "normalize_relative_weights",
]
