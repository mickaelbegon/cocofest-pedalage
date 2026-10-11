"""Optional causal left-to-right torque-transfer policy.

This module is intentionally small and solver agnostic.  It decides only an
allocation for the *next* certified RHO window.  It neither estimates
feasibility nor modifies an RHO NLP.  A caller must provide an auditable,
certified observation at the window boundary and must execute/audit the two
independent RHO problems separately.

The experimental policy is based on the paired transfer pilot: temporarily
reduce the left target torque by 0.10 Nm, increase the right target by the
same amount for ten complete cycles, then return to the equal split.  The
number of cycles and all thresholds are configuration, never hidden cycle
numbers in the decision kernel.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid making the small decision kernel import solver code
    from .oracle import OracleSlowState


SIDES = ("left", "right")


@dataclass(frozen=True)
class TransferObservation:
    """One certified boundary observation supplied to a transfer policy."""

    completed_cycles: int
    metrics: Mapping[str, Mapping[str, float]]
    certified: bool
    provenance_sha256: str

    def __post_init__(self) -> None:
        if type(self.completed_cycles) is not int or self.completed_cycles < 0:
            raise ValueError("completed_cycles must be a nonnegative integer")
        if self.certified is not True:
            raise ValueError("A causal transfer observation must be certified")
        if (not isinstance(self.provenance_sha256, str) or len(self.provenance_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.provenance_sha256)):
            raise ValueError("provenance_sha256 must be a SHA-256 digest")
        if set(self.metrics) != set(SIDES):
            raise ValueError("metrics must contain exactly left and right")
        for side in SIDES:
            values = self.metrics[side]
            if not isinstance(values, Mapping) or not values:
                raise ValueError(f"{side} metrics must be a nonempty mapping")
            if any(not isinstance(k, str) or not k or isinstance(v, bool)
                   or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for k, v in values.items()):
                raise ValueError("metrics must be finite named scalar values")


@dataclass(frozen=True)
class TransferConfig:
    """Bounded total-preserving action; experimental defaults are explicit."""

    baseline_left_nm: float = 0.96
    baseline_right_nm: float = 0.96
    left_relief_nm: float = 0.10
    action_cycles: int = 10
    allow_retrigger: bool = False

    def __post_init__(self) -> None:
        for name in ("baseline_left_nm", "baseline_right_nm", "left_relief_nm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.baseline_left_nm <= 0 or self.baseline_right_nm <= 0:
            raise ValueError("baseline torques must be positive")
        if not 0 < self.left_relief_nm < self.baseline_left_nm:
            raise ValueError("left_relief_nm must be positive and keep left torque positive")
        if type(self.action_cycles) is not int or self.action_cycles <= 0:
            raise ValueError("action_cycles must be a positive integer")
        if not isinstance(self.allow_retrigger, bool):
            raise ValueError("allow_retrigger must be boolean")

    @property
    def baseline(self) -> tuple[float, float]:
        return self.baseline_left_nm, self.baseline_right_nm

    @property
    def transfer(self) -> tuple[float, float]:
        return self.baseline_left_nm - self.left_relief_nm, self.baseline_right_nm + self.left_relief_nm


@dataclass(frozen=True)
class ThresholdPolicy:
    """A declared scalar gate, rather than a presumed fatigue estimator.

    The metric is read from the left certified observation.  Values are only a
    trigger for an experimental intervention; neither direction infers a
    physiological capacity or an imminent feasibility loss.
    """

    metric: str
    threshold: float
    direction: str = "below"

    def __post_init__(self) -> None:
        if not isinstance(self.metric, str) or not self.metric:
            raise ValueError("metric must be a nonempty string")
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)) or not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if self.direction not in {"below", "above"}:
            raise ValueError("direction must be below or above")

    def evaluate(self, observation: TransferObservation) -> tuple[bool, str, float | None]:
        value = observation.metrics["left"].get(self.metric)
        if value is None:
            return False, "metric_unavailable", None
        active = value <= self.threshold if self.direction == "below" else value >= self.threshold
        return active, "threshold_met" if active else "threshold_not_met", float(value)


@dataclass(frozen=True)
class BicepsReservePolicy:
    """A one-shot, state-based gate for the paired G->D transfer pilot.

    ``A/a_scale`` is deliberately treated as an *observed Ding amplitude
    proxy*, not as a certified force reserve or a prediction of infeasibility.
    The two thresholds make a no-decision band: a state in that band has no
    automatic action.  This prevents a decision boundary fitted to the c120
    and c140 calibration snapshots from silently becoming a cycle counter.

    ``fast_state_gate`` has to be supplied explicitly by an independently
    validated fast/history-state check.  Slow ``A, Tau1, Km`` states alone do
    not establish that the decision is Markovian, so the safe default is to
    abstain.
    """

    left_metric: str = "biceps_A_fraction"
    right_metric: str = "biceps_A_fraction"
    fast_state_metric: str = "fast_state_gate"
    left_activate_at_or_below: float = 0.62
    left_no_action_at_or_above: float = 0.67
    right_recipient_at_or_above: float = 0.78

    def __post_init__(self) -> None:
        for name in ("left_metric", "right_metric", "fast_state_metric"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("left_activate_at_or_below", "left_no_action_at_or_above",
                     "right_recipient_at_or_above"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.left_activate_at_or_below < self.left_no_action_at_or_above:
            raise ValueError("left thresholds must form a positive abstention band")
        if self.right_recipient_at_or_above <= 0:
            raise ValueError("right recipient threshold must be positive")

    def evaluate(self, observation: TransferObservation) -> tuple[bool, str, float | None]:
        left = observation.metrics["left"].get(self.left_metric)
        right = observation.metrics["right"].get(self.right_metric)
        fast = observation.metrics["left"].get(self.fast_state_metric)
        right_fast = observation.metrics["right"].get(self.fast_state_metric)
        if left is None or right is None:
            return False, "biceps_reserve_unavailable", None
        if fast != 1.0 or right_fast != 1.0:
            return False, "fast_state_gate_not_validated", float(left)
        if right < self.right_recipient_at_or_above:
            return False, "right_recipient_reserve_below_minimum", float(left)
        if left <= self.left_activate_at_or_below:
            return True, "left_biceps_reserve_triggered", float(left)
        if left >= self.left_no_action_at_or_above:
            return False, "left_biceps_reserve_above_activation_band", float(left)
        return False, "left_biceps_reserve_transition_band", float(left)


def biceps_reserve_observation(
    left: "OracleSlowState",
    right: "OracleSlowState",
    *,
    left_a_scale: float,
    right_a_scale: float,
    biceps_name: str = "Biceps",
    fast_state_gate: bool = False,
) -> TransferObservation:
    """Bind a current certified bilateral state to a reserve-gate observation.

    This is intentionally a measurement adapter, not a rollout.  It rejects
    different cycle boundaries and unverified artifacts, and exposes the
    companion ``Tau1``/``Km`` values for audit without using them as additional
    post-hoc trigger dimensions.  ``fast_state_gate=False`` is the conservative
    default until the dedicated fast-state validation has passed.
    """
    if isinstance(fast_state_gate, bool) is False:
        raise TypeError("fast_state_gate must be boolean")
    for name, value in (("left_a_scale", left_a_scale), ("right_a_scale", right_a_scale)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(biceps_name, str) or not biceps_name:
        raise ValueError("biceps_name must be a nonempty string")
    if getattr(left, "side", None) != "left" or getattr(right, "side", None) != "right":
        raise ValueError("states must be supplied left then right")
    if left.completed_cycles != right.completed_cycles:
        raise ValueError("bilateral states must share one certified boundary")
    left.verify_files()
    right.verify_files()

    def values(state, scale):
        try:
            index = state.muscles.index(biceps_name)
        except ValueError as error:
            raise ValueError(f"Missing {biceps_name} slow state") from error
        A, tau1, km = state.values[index]
        if not all(math.isfinite(value) and value > 0 for value in (A, tau1, km)):
            raise ValueError("Biceps slow state must be finite and positive")
        return {"biceps_A_fraction": float(A / scale), "biceps_Tau1": float(tau1),
                "biceps_Km": float(km), "fast_state_gate": float(fast_state_gate)}

    # Include the complete frozen state/provenance and scaling constants.  The
    # scalar metrics in TransferObservation remain readable, but cannot be
    # detached from their actual certified boundary by a copied digest.
    provenance = json.dumps({"left": asdict(left), "right": asdict(right),
                             "left_a_scale": left_a_scale, "right_a_scale": right_a_scale,
                             "biceps_name": biceps_name, "fast_state_gate": fast_state_gate},
                            sort_keys=True, separators=(",", ":"), allow_nan=False)
    return TransferObservation(left.completed_cycles, {"left": values(left, left_a_scale),
                                                        "right": values(right, right_a_scale)},
                               True, sha256(provenance.encode("utf-8")).hexdigest())


Policy = Callable[[TransferObservation], tuple[bool, str, float | None]]


@dataclass(frozen=True)
class TransferDecision:
    """Allocation for the next window with an immutable, auditable reason."""

    completed_cycles: int
    phase: str
    left_torque_nm: float
    right_torque_nm: float
    action_cycles_remaining: int
    reason: str
    metric_value: float | None
    observation_sha256: str

    @property
    def receipt_sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(payload.encode("utf-8")).hexdigest()


def observation_sha256(observation: TransferObservation) -> str:
    """Content address used to bind a decision to an RHO boundary state."""
    payload = json.dumps(asdict(observation), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


class CausalTransferSupervisor:
    """One temporary left-relief action, then an explicit equal-split return."""

    def __init__(self, policy: Policy | ThresholdPolicy | BicepsReservePolicy,
                 config: TransferConfig = TransferConfig()):
        if isinstance(policy, (ThresholdPolicy, BicepsReservePolicy)):
            policy = policy.evaluate
        if not callable(policy):
            raise TypeError("policy must be callable, ThresholdPolicy or BicepsReservePolicy")
        self.policy = policy
        self.config = config
        self._remaining = 0
        self._has_acted = False
        self.receipts: list[TransferDecision] = []

    def decide(self, observation: TransferObservation) -> TransferDecision:
        """Choose the next allocation from a certified observation only.

        The first decision that passes ``policy`` starts ``action_cycles``
        transfer windows.  The next decision after that block is an explicit
        baseline return.  Re-triggering is off by default so that a single
        pilot action cannot silently become chronic redistribution.
        """
        digest = observation_sha256(observation)
        metric = None
        if self._remaining > 0:
            self._remaining -= 1
            phase, torques, reason = "transfer", self.config.transfer, "active_transfer_block"
        elif self._has_acted:
            phase, torques, reason = "return", self.config.baseline, "explicit_return_to_baseline"
            if self.config.allow_retrigger:
                active, reason, metric = self.policy(observation)
                if active:
                    self._has_acted = True
                    self._remaining = self.config.action_cycles - 1
                    phase, torques = "transfer", self.config.transfer
        else:
            active, reason, metric = self.policy(observation)
            if active:
                self._has_acted = True
                self._remaining = self.config.action_cycles - 1
                phase, torques = "transfer", self.config.transfer
            else:
                phase, torques = "hold", self.config.baseline
        result = TransferDecision(observation.completed_cycles, phase, *torques, self._remaining,
                                  reason, metric, digest)
        self.receipts.append(result)
        return result
