"""Cycle-synchronous resistance allocation for two independent arm RHO runs.

The two unilateral NLPs remain independent.  This controller only chooses the
two *next* terminal-work targets between certified RHO windows.  With one
turn per window, the invariant is

``W_R + W_L = 2 pi tau_total``  and therefore ``tau_R + tau_L = tau_total``.

It is deliberately a supervisory feedback law, rather than a coupled OCP:
the targets for cycle ``k+1`` depend solely on terminal reserve measurements
from the completed, synchronized cycle ``k``.  Consequently it does not add
cross-arm states, controls, or constraints to either compiled NLP.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from cocofest.optimization.endurance_weight_supervisor import (
    _project_centered_logs,
    normalize_relative_weights,
)


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return value


def _jsonable(value):
    """Convert NumPy audit values before the append-only controller journal."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class IndependentArmRhoPaceConfig:
    """Bounded causal allocation of one fixed bilateral work budget.

    If ``capacity_feedback`` is enabled, the next right fraction is the
    clipped and smoothed reserve proportion

    ``p_R^* = r_R**gain / (r_R**gain + r_L**gain)``,

    where ``r_i`` is the minimum terminal ``A/A_scale`` reported for arm i.
    A fresher arm is thus assigned more of the *next* cycle's work.  The
    projection and the maximum change protect the individual arm problems
    from an abrupt infeasible target change.
    """

    total_equivalent_mean_torque_nm: float
    initial_right_fraction: float = 0.5
    minimum_arm_equivalent_mean_torque_nm: float = 0.0
    capacity_feedback: bool = True
    update_every_cycles: int = 10
    capacity_gain: float = 1.0
    smoothing: float = 0.25
    max_fraction_step: float = 0.10
    initial_split_policy: str = "manual"

    def __post_init__(self) -> None:
        total = _finite_nonnegative(self.total_equivalent_mean_torque_nm,
                                    "total_equivalent_mean_torque_nm")
        minimum = _finite_nonnegative(self.minimum_arm_equivalent_mean_torque_nm,
                                      "minimum_arm_equivalent_mean_torque_nm")
        if 2.0 * minimum > total + 1e-12:
            raise ValueError("twice the minimum arm torque must not exceed the total torque.")
        if (isinstance(self.update_every_cycles, bool) or not isinstance(self.update_every_cycles, int)
                or self.update_every_cycles < 1):
            raise ValueError("update_every_cycles must be a strictly positive integer.")
        for name in ("initial_right_fraction", "capacity_gain", "smoothing", "max_fraction_step"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not 0.0 <= self.initial_right_fraction <= 1.0:
            raise ValueError("initial_right_fraction must lie in [0, 1].")
        if self.capacity_gain <= 0:
            raise ValueError("capacity_gain must be positive.")
        if not 0.0 <= self.smoothing <= 1.0:
            raise ValueError("smoothing must lie in [0, 1].")
        if not 0.0 <= self.max_fraction_step <= 1.0:
            raise ValueError("max_fraction_step must lie in [0, 1].")
        if not isinstance(self.capacity_feedback, bool):
            raise ValueError("capacity_feedback must be boolean.")
        if self.initial_split_policy not in {"manual", "capacity_fatigability_after_first_cycle"}:
            raise ValueError("initial_split_policy must be manual or capacity_fatigability_after_first_cycle.")


def _minimum_capacity_ratio(metrics: Mapping[str, Any]) -> float | None:
    """Read one arm reserve from a deliberately small, explicit schema."""
    values: list[float] = []
    for key in ("minimum_capacity_ratio", "final_capacity_ratio"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values.append(float(value))
    reserve = metrics.get("terminal_capacity_reserve")
    if isinstance(reserve, Mapping):
        value = reserve.get("minimum_ratio")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values.append(float(value))
    muscles = metrics.get("muscle_fatigue")
    if isinstance(muscles, (list, tuple)):
        for muscle in muscles:
            if isinstance(muscle, Mapping):
                value = muscle.get("final_capacity_ratio")
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    values.append(float(value))
    if not values or min(values) < 0.0:
        return None
    return min(values)


class IndependentArmResistancePace:
    """Stateful, auditable supervisor for the resistance split."""

    def __init__(self, config: IndependentArmRhoPaceConfig):
        self.config = config
        self._right_fraction = self._project(float(config.initial_right_fraction))
        self.events: list[dict[str, Any]] = []
        self.certified_cycles = 0

    def _bounds(self) -> tuple[float, float]:
        total = self.config.total_equivalent_mean_torque_nm
        if total == 0.0:
            return 0.0, 1.0
        lower = self.config.minimum_arm_equivalent_mean_torque_nm / total
        return lower, 1.0 - lower

    def _project(self, fraction: float) -> float:
        lower, upper = self._bounds()
        return min(max(float(fraction), lower), upper)

    @property
    def equivalent_mean_torques(self) -> dict[str, float]:
        right = self.config.total_equivalent_mean_torque_nm * self._right_fraction
        return {"right": right, "left": self.config.total_equivalent_mean_torque_nm - right}

    def apply_initial_split_decision(self, decision: Mapping[str, Any]) -> dict[str, Any]:
        """Install one calibrated split before the regular causal feedback.

        The decision is expected after a first certified reference cycle.  It
        is deliberately separate from :meth:`observe`: it must not be logged
        as a reserve-feedback update, and it cannot be a feasibility verdict.
        """
        event = {"event_type": "initial_split", **_jsonable(dict(decision))}
        if self.certified_cycles != 0:
            raise RuntimeError("The calibrated initial split must precede reserve-feedback observations.")
        if self.config.initial_split_policy != "capacity_fatigability_after_first_cycle":
            event.update({"status": "held", "reason": "initial_split_policy_disabled"})
        elif decision.get("status") != "proposed":
            event.update({"status": "held", "reason": decision.get("reason", "measurement_unavailable")})
        else:
            fraction = self._project(float(decision["right_fraction"]))
            self._right_fraction = fraction
            event.update({"status": "applied", "next_right_fraction": fraction,
                          "next_equivalent_mean_torque_nm": self.equivalent_mean_torques})
        self.events.append(event)
        return event

    def observe(self, cycle_index: int, right_metrics: Mapping[str, Any],
                left_metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Certify cycle ``cycle_index`` and choose its successor allocation."""
        if isinstance(cycle_index, bool) or not isinstance(cycle_index, int) or cycle_index < 0:
            raise ValueError("cycle_index must be a non-negative integer.")
        right_before = self._right_fraction
        right_reserve = _minimum_capacity_ratio(right_metrics)
        left_reserve = _minimum_capacity_ratio(left_metrics)
        event: dict[str, Any] = {
            "cycle_index": cycle_index,
            "allocation_used_equivalent_mean_torque_nm": self.equivalent_mean_torques,
            "right_minimum_capacity_ratio": right_reserve,
            "left_minimum_capacity_ratio": left_reserve,
            "capacity_feedback_enabled": self.config.capacity_feedback,
        }
        if cycle_index != self.certified_cycles:
            event.update({"status": "refused", "reason": "nonsequential_cycle"})
        elif any(metrics.get("certified") is not True for metrics in (right_metrics, left_metrics)):
            event.update({"status": "refused", "reason": "uncertified_source_state"})
        elif not self.config.capacity_feedback:
            event.update({"status": "held", "reason": "capacity_feedback_disabled"})
        elif (cycle_index + 1) % self.config.update_every_cycles:
            # This is a hard update cadence: all targets stay fixed for a
            # complete block of certified cycles, independently of the
            # measured reserves within that block.
            event.update({"status": "held", "reason": "update_not_due"})
        elif right_reserve is None or left_reserve is None:
            # Do not invent an endurance observation. The last certified split
            # is safe to retain if the backend did not expose reserve data.
            event.update({"status": "held", "reason": "capacity_measurement_unavailable"})
        else:
            denominator = right_reserve ** self.config.capacity_gain + left_reserve ** self.config.capacity_gain
            proposed = right_before if denominator <= 0.0 else right_reserve ** self.config.capacity_gain / denominator
            smoothed = (1.0 - self.config.smoothing) * right_before + self.config.smoothing * proposed
            limited = min(max(smoothed, right_before - self.config.max_fraction_step),
                          right_before + self.config.max_fraction_step)
            self._right_fraction = self._project(limited)
            event.update({"status": "updated", "proposed_right_fraction": proposed,
                          "next_right_fraction": self._right_fraction})
        event["next_equivalent_mean_torque_nm"] = self.equivalent_mean_torques
        if event["status"] != "refused":
            self.certified_cycles += 1
        event["certified_cycles"] = self.certified_cycles
        self.events.append(event)
        return event

    def audit(self) -> dict[str, Any]:
        return {"policy": "independent_arm_resistance_pace_v1", "config": asdict(self.config),
                "events": list(self.events), "next_equivalent_mean_torque_nm": self.equivalent_mean_torques}


@dataclass(frozen=True)
class BilateralArmPaceConfig:
    """Local four-muscle feedback with a hard shared block cadence.

    This is intentionally separate from the historical ``RhoPaceConfig``:
    the bilateral experiment permits changing local work while preserving
    the bilateral total. The unilateral comparison's fixed-load guard stays
    in ``RhoPaceController`` without any exception or bypass.
    """

    update_every_cycles: int = 10
    adaptation_enabled: bool = True
    smoothing: float = 0.2
    capacity_gain: float = 1.0
    min_relative_weight: float = 0.25
    max_relative_weight: float = 4.0
    max_log_step: float = math.log(1.1)
    adaptation_strategy: str = "capacity_feedback"
    physio_deadband_log: float = math.log(1.01)

    def __post_init__(self):
        if type(self.update_every_cycles) is not int or self.update_every_cycles < 1:
            raise ValueError("update_every_cycles must be a positive integer")
        if type(self.adaptation_enabled) is not bool:
            raise ValueError("adaptation_enabled must be boolean")
        for name in ("smoothing", "capacity_gain", "min_relative_weight", "max_relative_weight", "max_log_step",
                     "physio_deadband_log"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.smoothing > 1:
            raise ValueError("smoothing must not exceed one")
        if not self.min_relative_weight <= 1 <= self.max_relative_weight:
            raise ValueError("weight bounds must contain one")
        if self.adaptation_strategy not in {"capacity_feedback", "physio_update", "mechanical_sensitivity"}:
            raise ValueError("adaptation_strategy must be capacity_feedback, physio_update, or mechanical_sensitivity")


class BilateralArmPaceController:
    """One arm's four weights; no opposite-arm state enters this policy.

    Call ``boundary(0, ...)`` to install the initial cost. Subsequently call
    once after *each* certified window, using the number of completed cycles.
    Only boundaries K, 2K, ... can change weights or local work. The caller
    must provide the total-work-conserving target selected by the bilateral
    supervisor after both arms have passed certification.
    """

    def __init__(self, side, muscle_names, initial_weights, *,
                 equivalent_mean_torque_nm, initial_weight_basis,
                 config=None, journal_path=None):
        if side not in {"right", "left"}:
            raise ValueError("side must be right or left")
        self.side = side
        self.muscle_names = tuple(muscle_names)
        if (len(self.muscle_names) != 4 or len(set(self.muscle_names)) != 4
                or any(not isinstance(name, str) or not name for name in self.muscle_names)):
            raise ValueError("Each independent arm requires exactly four unique muscle names")
        if not isinstance(initial_weight_basis, str) or not initial_weight_basis.strip():
            raise ValueError("initial_weight_basis must describe the initial weights")
        self.config = config or BilateralArmPaceConfig()
        normalized = normalize_relative_weights(initial_weights)
        if len(normalized) != 4:
            raise ValueError("Each independent arm requires exactly four weights")
        self.initial_weights = tuple(math.exp(v) for v in _project_centered_logs(
            [math.log(w) for w in normalized],
            lower=math.log(self.config.min_relative_weight), upper=math.log(self.config.max_relative_weight)))
        self.weights = self.initial_weights
        self.equivalent_mean_torque_nm = _finite_nonnegative(equivalent_mean_torque_nm, "equivalent_mean_torque_nm")
        self.connected = False
        self.completed_cycles = -1
        self.events = []
        self.journal_path = Path(journal_path) if journal_path is not None else None
        if self.journal_path is not None:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            self.journal_path.touch(exist_ok=False)
        self._record({"event": "configuration", "policy": (
                      "bilateral_arm_capacity_feedback_v1"
                      if self.config.adaptation_strategy == "capacity_feedback"
                      else ("bilateral_arm_physio_update_work_v1_experimental"
                            if self.config.adaptation_strategy == "physio_update"
                            else "bilateral_arm_mechanical_sensitivity_squared_v1_experimental")),
                      "config": asdict(self.config), "muscle_names": self.muscle_names,
                      "initial_weights": self.initial_weights, "initial_weight_basis": initial_weight_basis,
                      "mechanical_formulation": "isokinetic", "uses_opposite_arm_capacities": False,
                      "uses_fho_data": False, "endurance_improvement_validated": False})

    def _record(self, event):
        event = {"side": self.side, "weights": self.weights,
                 "equivalent_mean_torque_nm": self.equivalent_mean_torque_nm,
                 "target_work_j_per_cycle": 2 * math.pi * self.equivalent_mean_torque_nm,
                 "ocp_cost_connected": self.connected, **event}
        encoded = json.dumps(event, allow_nan=False, sort_keys=True)
        self.events.append(json.loads(encoded))
        if self.journal_path is not None:
            with self.journal_path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
        return self.events[-1]

    def boundary(self, completed_cycles, capacity_ratios, *, certified,
                 equivalent_mean_torque_nm, apply_weights: Callable | None,
                 physio_update_inputs=None):
        event = {"event": "boundary", "completed_cycles": completed_cycles,
                 "weights_before": self.weights, "reasons": []}
        reasons = event["reasons"]
        if type(completed_cycles) is not int or completed_cycles != self.completed_cycles + 1:
            reasons.append("nonsequential_certified_cycle")
        if certified is not True:
            reasons.append("uncertified_source_state")
        try:
            torque = _finite_nonnegative(equivalent_mean_torque_nm, "equivalent_mean_torque_nm")
        except (ValueError, TypeError):
            torque = None
            reasons.append("invalid_work_target")
        due = type(completed_cycles) is int and completed_cycles % self.config.update_every_cycles == 0
        if torque != self.equivalent_mean_torque_nm and not due:
            reasons.append("work_change_outside_block_boundary")
        try:
            ratios = tuple(float(value) for value in capacity_ratios)
            if len(ratios) != 4 or any(not math.isfinite(value) or value <= 0 for value in ratios):
                raise ValueError("invalid ratios")
        except (ValueError, TypeError):
            ratios = None
            reasons.append("invalid_capacity_ratios")
        event["capacity_ratios"] = ratios
        if reasons:
            return self._record({**event, "status": "refused"})
        if self.connected and (not due or not self.config.adaptation_enabled):
            self.completed_cycles = completed_cycles
            self.equivalent_mean_torque_nm = torque
            return self._record({**event, "status": "held", "reasons": [
                "slow_update_not_due" if not due else "static_weights"]})
        proposed = self.initial_weights
        if self.connected:
            if self.config.adaptation_strategy == "capacity_feedback":
                target = _project_centered_logs(
                    [math.log(weight) - self.config.capacity_gain * math.log(ratio)
                     for weight, ratio in zip(self.initial_weights, ratios)],
                    lower=math.log(self.config.min_relative_weight), upper=math.log(self.config.max_relative_weight))
                logs = tuple(math.log(weight) for weight in self.weights)
                step = tuple(self.config.smoothing * (value - previous) for value, previous in zip(target, logs))
                scale = min(1.0, self.config.max_log_step / max(max(map(abs, step)), 1e-300))
                proposed = tuple(math.exp(previous + scale * change) for previous, change in zip(logs, step))
            else:
                if physio_update_inputs is None:
                    self.completed_cycles = completed_cycles
                    self.equivalent_mean_torque_nm = torque
                    return self._record({**event, "status": "held", "reasons": [
                        "physio_update_inputs_unavailable"]})
                from cocofest.optimization.physio_update import (
                    PhysioUpdateConfig, propose_mechanical_sensitivity_weights, propose_physio_update,
                )
                try:
                    proposal_config = PhysioUpdateConfig(
                        update_every_cycles=self.config.update_every_cycles,
                        smoothing=self.config.smoothing,
                        max_log_step=self.config.max_log_step,
                        deadband_log=self.config.physio_deadband_log,
                        min_relative_weight=self.config.min_relative_weight,
                        max_relative_weight=self.config.max_relative_weight,
                    )
                    if self.config.adaptation_strategy == "physio_update":
                        proposal = propose_physio_update(
                            completed_cycles=completed_cycles, certified=True,
                            incumbent_weights=self.weights, config=proposal_config,
                            **physio_update_inputs,
                        )
                    else:
                        proposal = propose_mechanical_sensitivity_weights(
                            completed_cycles=completed_cycles, certified=True,
                            incumbent_weights=self.weights, config=proposal_config,
                            available_positive_power=physio_update_inputs["available_positive_power"],
                            phase_durations=physio_update_inputs["phase_durations"],
                            required_active_work=physio_update_inputs["required_active_work"],
                        )
                except Exception as error:
                    self.completed_cycles = completed_cycles
                    self.equivalent_mean_torque_nm = torque
                    return self._record({**event, "status": "held", "reasons": [
                        "physio_update_input_rejected"], "error": f"{type(error).__name__}: {error}"})
                event["physio_update"] = _jsonable(proposal)
                if proposal["status"] != "proposed":
                    self.completed_cycles = completed_cycles
                    self.equivalent_mean_torque_nm = torque
                    return self._record({**event, "status": "held", "reasons": [proposal["reason"]]})
                proposed = tuple(float(value) for value in proposal["weights"])
        if apply_weights is None:
            return self._record({**event, "status": "refused", "reasons": ["ocp_cost_not_integrated"]})
        try:
            receipt = apply_weights(proposed)
            if not isinstance(receipt, dict) or receipt.get("ocp_cost_updated") is not True:
                raise RuntimeError("OCP writer did not confirm the cost update")
        except Exception as error:
            self._record({**event, "status": "fatal", "reasons": ["ocp_update_failed"], "error": str(error)})
            raise
        self.weights, self.connected = proposed, True
        self.completed_cycles = completed_cycles
        self.equivalent_mean_torque_nm = torque
        return self._record({**event, "status": "applied", "ocp_update": receipt})
