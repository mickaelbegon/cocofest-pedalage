"""Conservative, solver-independent oracle supervisor at block boundaries.

The supervisor verifies observed RHO-state provenance and the binding of each
conditional estimate to that observation.  It can accept a proposed split, but
neither the estimate nor this decision certifies feasibility of a future cycle.
No RHO graph, process coordinator, or online force predictor is used here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math

import numpy as np

from .oracle import OracleSlowState
from .remaining_cycles import ConditionalRemainingCycles


SIDES = ("left", "right")


def oracle_state_sha256(state: OracleSlowState) -> str:
    """Content address of the state and its frozen provenance declarations."""
    payload = json.dumps(asdict(state), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BoundConditionalEstimate:
    """Conditional threshold count explicitly associated with one oracle state.

    Binding is a caller declaration. The supervisor detects stale/mixed state
    inputs; it cannot certify how the caller computed a conditional count.
    """

    result: ConditionalRemainingCycles
    oracle_state_sha256: str


@dataclass(frozen=True)
class SplitProposal:
    left_torque_nm: float
    right_torque_nm: float
    left_weights: tuple[float, ...] | None = None
    right_weights: tuple[float, ...] | None = None
    rule_reason: str | None = None
    rule_receipt_sha256: str | None = None


@dataclass(frozen=True)
class SupervisorLimits:
    update_every_cycles: int = 10
    min_torque_nm: float = 0.0
    max_torque_nm: float = math.inf
    max_step_nm: float = math.inf
    max_asymmetry_nm: float = math.inf
    prescribed_total_nm: float | None = None

    def __post_init__(self) -> None:
        if type(self.update_every_cycles) is not int or self.update_every_cycles <= 0:
            raise ValueError("update_every_cycles must be a positive integer")
        for name in ("min_torque_nm", "max_torque_nm", "max_step_nm", "max_asymmetry_nm"):
            value = getattr(self, name)
            if isinstance(value, bool) or math.isnan(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and not NaN")
        if self.min_torque_nm > self.max_torque_nm:
            raise ValueError("torque bounds are reversed")
        if self.prescribed_total_nm is not None and (
            isinstance(self.prescribed_total_nm, bool) or not math.isfinite(self.prescribed_total_nm)
            or self.prescribed_total_nm < 0
        ):
            raise ValueError("prescribed_total_nm must be finite and nonnegative")


@dataclass(frozen=True)
class SupervisorDecision:
    completed_cycles: int
    target_torque_nm: tuple[float, float]
    weights: tuple[tuple[float, ...], tuple[float, ...]]
    status: str
    reason: str
    feasibility: str = "not_assessed"


@dataclass(frozen=True)
class SupervisorReceipt:
    """Immutable record of inputs and decision; it is not a solver certificate."""

    completed_cycles: int
    update_due: bool
    status: str
    reason: str
    oracle_state_sha256: tuple[str | None, str | None]
    estimate_first_exceedance: tuple[int | None, int | None]
    previous_torque_nm: tuple[float, float]
    target_torque_nm: tuple[float, float]
    weights: tuple[tuple[float, ...], tuple[float, ...]]
    rule_name: str | None
    rule_reason: str | None = None
    rule_receipt_sha256: str | None = None
    feasibility: str = "not_assessed"

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(payload.encode("utf-8")).hexdigest()


SplitRule = Callable[[Mapping[str, BoundConditionalEstimate], tuple[float, float]], SplitProposal]


def equalizing_split_rule(comfort) -> SplitRule:
    """Adapt the existing conditional equalization rule to this supervisor."""
    from .split_rule import SplitComfort, equalize_remaining_cycles

    if not isinstance(comfort, SplitComfort):
        raise TypeError("comfort must be SplitComfort")

    def equalize(estimates: Mapping[str, BoundConditionalEstimate], previous: tuple[float, float]) -> SplitProposal:
        result = equalize_remaining_cycles(previous, {side: estimates[side].result for side in SIDES}, comfort)
        return SplitProposal(*result.torque, rule_reason=result.reason, rule_receipt_sha256=result.receipt_sha256)

    return equalize


def _valid_weights(values: tuple[float, ...] | None, expected: int) -> tuple[float, ...] | None:
    if values is None:
        return tuple(1.0 for _ in range(expected))
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if len(result) != expected or any(not math.isfinite(value) or value <= 0 for value in result):
        return None
    if not math.isclose(math.fsum(math.log(value) for value in result), 0.0, abs_tol=1e-8):
        return None
    return result


def _valid_conditional_estimate(result: ConditionalRemainingCycles) -> bool:
    if (result.interpretation != "conditional_state_threshold_only_not_force_or_feasibility_prediction"
            or type(result.horizon) is not int or result.horizon < 0
            or type(result.cycles_within_bounds) is not int
            or not 0 <= result.cycles_within_bounds <= result.horizon):
        return False
    crossing = result.first_exceedance_cycle
    if (type(crossing) is not int or not 0 <= crossing <= result.horizon
            or result.cycles_within_bounds != max(0, crossing - 1)):
        return False
    digest = result.initial_receipt_sha256
    return isinstance(digest, str) and len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def supervise_cycle(
    *,
    completed_cycles: int,
    states: Mapping[str, OracleSlowState | None],
    estimates: Mapping[str, BoundConditionalEstimate | None],
    previous_torque_nm: tuple[float, float],
    rule: SplitRule,
    limits: SupervisorLimits = SupervisorLimits(),
) -> tuple[SupervisorDecision, SupervisorReceipt]:
    """Apply a split rule every M cycles or return a reasoned hold decision.

    A hold repeats the last torque targets with unit relative muscle weights.
    The split rule is an injected pure policy; validation here checks declared
    comfort bounds, stale inputs, and the shape of proposed weights.
    """
    if type(completed_cycles) is not int or completed_cycles < 0:
        raise ValueError("completed_cycles must be a nonnegative integer")
    if set(states) != set(SIDES) or set(estimates) != set(SIDES):
        raise ValueError("states and estimates must contain exactly left and right")
    if len(previous_torque_nm) != 2 or any(not math.isfinite(v) or v < 0 for v in previous_torque_nm):
        raise ValueError("previous_torque_nm must have two finite nonnegative values")
    previous = tuple(float(value) for value in previous_torque_nm)
    due = completed_cycles % limits.update_every_cycles == 0
    digests: list[str | None] = [None, None]
    crossings: list[int | None] = [None, None]
    sizes = [len(state.muscles) if isinstance(state, OracleSlowState) else 0 for state in (states["left"], states["right"])]

    def done(status: str, reason: str, targets: tuple[float, float], weights, name: str | None,
             rule_reason: str | None = None, rule_receipt_sha256: str | None = None):
        decision = SupervisorDecision(completed_cycles, targets, weights, status, reason)
        receipt = SupervisorReceipt(completed_cycles, due, status, reason, tuple(digests), tuple(crossings),
                                    previous, targets, weights, name, rule_reason, rule_receipt_sha256)
        return decision, receipt

    unit = (tuple(1.0 for _ in range(sizes[0])), tuple(1.0 for _ in range(sizes[1])))
    if not due:
        return done("hold", "update_not_due", previous, unit, None)
    for index, side in enumerate(SIDES):
        state = states[side]
        if not isinstance(state, OracleSlowState) or state.side != side or state.completed_cycles != completed_cycles:
            return done("hold", f"uncertified_state_{side}", previous, unit, None)
        if (state.source_kind not in {"certified_bilateral_restart", "certified_rho_trajectory"}
                or state.source_node != ({"certified_bilateral_restart": "frozen_initial_equality_of_next_cycle",
                                          "certified_rho_trajectory": "last_node"}.get(state.source_kind))
                or not state.artifacts or len(state.muscles) != len(state.values)
                or np.asarray(state.values).shape != (len(state.muscles), 3)
                or not np.isfinite(state.as_array()).all() or np.any(state.as_array() <= 0)):
            return done("hold", f"uncertified_state_{side}", previous, unit, None)
        try:
            state.verify_files()
        except (OSError, ValueError):
            return done("hold", f"uncertified_state_{side}", previous, unit, None)
        digests[index] = oracle_state_sha256(state)
        estimate = estimates[side]
        if not isinstance(estimate, BoundConditionalEstimate) or not isinstance(estimate.result, ConditionalRemainingCycles):
            return done("hold", f"missing_estimate_{side}", previous, unit, None)
        if estimate.oracle_state_sha256 != digests[index]:
            return done("hold", f"stale_estimate_{side}", previous, unit, None)
        result = estimate.result
        crossings[index] = result.first_exceedance_cycle
        if result.censored:
            return done("hold", f"censored_estimate_{side}", previous, unit, None)
        if not _valid_conditional_estimate(result):
            return done("hold", f"invalid_estimate_{side}", previous, unit, None)

    name = getattr(rule, "__name__", type(rule).__name__)
    try:
        proposal = rule(estimates, previous)
    except Exception as error:
        return done("hold", f"rule_error:{type(error).__name__}", previous, unit, name)
    if not isinstance(proposal, SplitProposal):
        return done("hold", "invalid_rule_proposal", previous, unit, name)
    targets = (proposal.left_torque_nm, proposal.right_torque_nm)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
           or not limits.min_torque_nm <= value <= limits.max_torque_nm for value in targets):
        return done("hold", "invalid_rule_torque", previous, unit, name)
    if (any(abs(target - old) > limits.max_step_nm + 1e-12 for target, old in zip(targets, previous))
            or abs(targets[0] - targets[1]) > limits.max_asymmetry_nm + 1e-12
            or (limits.prescribed_total_nm is not None
                and not math.isclose(math.fsum(targets), limits.prescribed_total_nm, abs_tol=1e-9))):
        return done("hold", "rule_violates_comfort_bounds", previous, unit, name)
    weights = (_valid_weights(proposal.left_weights, sizes[0]), _valid_weights(proposal.right_weights, sizes[1]))
    if any(value is None for value in weights):
        return done("hold", "invalid_rule_weights", previous, unit, name)
    targets = tuple(float(value) for value in targets)
    if targets == previous and weights == unit:
        return done("hold", "rule_held_conditionally", previous, unit, name,
                    proposal.rule_reason, proposal.rule_receipt_sha256)
    return done("proposed", "rule_applied_conditionally", targets, weights, name,
                proposal.rule_reason, proposal.rule_receipt_sha256)
