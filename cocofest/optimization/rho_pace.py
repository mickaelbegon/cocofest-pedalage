"""Causal, slowly varying physiological fatigue cost for dynamic RHO.

PACE is an experimental feedback heuristic, not an endurance certificate:
every K certified cycles its target relative weights are proportional to
``initial_weight * (A/A_scale)**(-capacity_gain)``. A bounded log-space step
approaches that target. The geometric mean and the overall fatigue weight
remain fixed. No FHO reference or future trajectory enters the policy.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable

from cocofest.optimization.endurance_weight_supervisor import (
    _project_centered_logs,
    normalize_relative_weights,
)


@dataclass(frozen=True)
class RhoPaceConfig:
    update_every_cycles: int = 5
    smoothing: float = 0.2
    capacity_gain: float = 1.0
    min_relative_weight: float = 0.25
    max_relative_weight: float = 4.0
    max_log_step: float = math.log(1.1)
    max_cycles: int = 100

    def __post_init__(self):
        for name in ("update_every_cycles", "max_cycles"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_cycles > 100:
            raise ValueError("RHO-PACE campaign is limited to 100 cycles")
        for name in ("smoothing", "capacity_gain", "min_relative_weight",
                     "max_relative_weight", "max_log_step"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.smoothing > 1:
            raise ValueError("smoothing must not exceed 1")
        if not self.min_relative_weight <= 1 <= self.max_relative_weight:
            raise ValueError("Relative weight bounds must contain 1")


class RhoPaceController:
    """Transactional policy: weights change only after the OCP writer succeeds.

    ``apply_weights`` must actually update the OCP and return an audit mapping
    containing ``ocp_cost_updated=True``. Missing integration is a refusal.
    Exceptions propagate: a partially failed external write must stop the run.
    """

    def __init__(self, muscle_names, initial_weights, *, signed_crank_torque_nm,
                 parameters, initial_weight_basis, config=None, journal_path=None):
        self.config = config or RhoPaceConfig()
        self.muscle_names = tuple(muscle_names)
        if (not self.muscle_names or len(set(self.muscle_names)) != len(self.muscle_names)
                or any(not isinstance(n, str) or not n for n in self.muscle_names)):
            raise ValueError("muscle_names must be unique nonempty strings")
        supplied_weights = tuple(float(w) for w in initial_weights)
        normalized_initial = normalize_relative_weights(supplied_weights)
        if len(normalized_initial) != len(self.muscle_names):
            raise ValueError("Expected one initial weight per muscle")
        self.initial_weights = tuple(math.exp(v) for v in _project_centered_logs(
            [math.log(w) for w in normalized_initial],
            lower=math.log(self.config.min_relative_weight),
            upper=math.log(self.config.max_relative_weight)))
        self.signed_crank_torque_nm = float(signed_crank_torque_nm)
        if not math.isfinite(self.signed_crank_torque_nm) or self.signed_crank_torque_nm <= 0:
            raise ValueError("Negative crank rotation requires positive constant resistive torque")
        if not initial_weight_basis:
            raise ValueError("An explicit initial_weight_basis is required")
        if set(parameters) != set(self.muscle_names):
            raise ValueError("parameters must identify every muscle")
        self.parameters = json.loads(json.dumps(parameters, allow_nan=False))
        self.weights = self.initial_weights
        self.connected = False
        self.last_cycle = -1
        self.events = []
        self.journal_path = None if journal_path is None else Path(journal_path).expanduser().resolve()
        if self.journal_path:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            # An existing journal belongs to a previous run: never mix arms.
            with self.journal_path.open("x", encoding="utf-8"):
                pass
        self._record({"event": "configuration", "arm": "RHO-PACE",
                      "policy": "causal_capacity_feedback_v1", "config": asdict(self.config),
                      "muscle_names": self.muscle_names, "parameters": self.parameters,
                      "initial_weights": self.initial_weights,
                      "supplied_initial_weights": supplied_weights,
                      "normalized_initial_weights_before_projection": normalized_initial,
                      "initial_weight_normalization": "geometric_mean_one_then_log_box_projection",
                      "applied_weights_are_article_raw_max": False,
                      "initial_projection_changed_ratios": any(
                          abs(math.log(a / b)) > 1e-12
                          for a, b in zip(self.initial_weights, normalized_initial)),
                      "initial_weight_basis": initial_weight_basis,
                      "uses_fho_data": False, "mechanical_formulation": "dynamic",
                      "endurance_improvement_validated": False})

    def _record(self, event):
        event = {"signed_crank_torque_nm": self.signed_crank_torque_nm,
                 "weights": self.weights, "ocp_cost_connected": self.connected, **event}
        encoded = json.dumps(event, allow_nan=False, sort_keys=True)
        self.events.append(json.loads(encoded))
        if self.journal_path:
            with self.journal_path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
        return self.events[-1]

    def boundary(self, cycle_index, capacity_ratios, *, certified,
                 signed_crank_torque_nm, apply_weights: Callable | None):
        """Prepare window ``cycle_index``; zero is the certified common seed.

        ``certified`` refers to the previous completed solution for later
        windows. A rejected input leaves the incumbent cost untouched.
        """
        before = self.weights
        event = {"event": "boundary", "cycle_index": cycle_index,
                 "weights_before": before, "proposed_weights": None,
                 "previous_solution_certified": bool(certified), "reasons": []}
        reasons = event["reasons"]
        if isinstance(cycle_index, bool) or not isinstance(cycle_index, int) or cycle_index < 0:
            reasons.append("invalid_cycle_index")
        elif cycle_index >= self.config.max_cycles:
            reasons.append("maximum_cycles_reached")
        elif cycle_index <= self.last_cycle:
            reasons.append("duplicate_or_out_of_order_cycle")
        if signed_crank_torque_nm != self.signed_crank_torque_nm:
            reasons.append("resistance_changed")
        if not certified:
            reasons.append("uncertified_source_state")
        try:
            ratios = tuple(float(v) for v in capacity_ratios)
            valid = len(ratios) == len(self.weights) and all(
                math.isfinite(v) and v > 0 for v in ratios)
        except (TypeError, ValueError):
            ratios, valid = (), False
        event["capacity_ratios"] = ratios if valid else None
        if not valid:
            reasons.append("invalid_capacity_ratios")
        if reasons:
            return self._record({**event, "status": "refused"})
        if self.connected and cycle_index % self.config.update_every_cycles:
            self.last_cycle = cycle_index
            return self._record({**event, "status": "held", "reasons": ["slow_update_not_due"]})
        if not self.connected and cycle_index != 0:
            return self._record({**event, "status": "refused", "reasons": ["initial_cost_not_connected"]})
        proposed = self.initial_weights
        if self.connected:
            target = _project_centered_logs(
                [math.log(w) - self.config.capacity_gain * math.log(r)
                 for w, r in zip(self.initial_weights, ratios)],
                lower=math.log(self.config.min_relative_weight),
                upper=math.log(self.config.max_relative_weight))
            logs = tuple(math.log(w) for w in self.weights)
            delta = tuple(self.config.smoothing * (t - w) for t, w in zip(target, logs))
            scale = min(1.0, self.config.max_log_step / max(max(map(abs, delta)), 1e-300))
            proposed = tuple(math.exp(w + scale * d) for w, d in zip(logs, delta))
        event["proposed_weights"] = proposed
        if apply_weights is None:
            return self._record({**event, "status": "refused", "reasons": ["ocp_cost_not_integrated"]})
        try:
            receipt = apply_weights(proposed)
            if not isinstance(receipt, dict) or receipt.get("ocp_cost_updated") is not True:
                raise RuntimeError("OCP writer did not confirm the cost update")
        except Exception as error:
            self._record({**event, "status": "fatal", "reasons": ["ocp_update_failed"],
                          "error": f"{type(error).__name__}: {error}"})
            raise
        self.weights, self.connected, self.last_cycle = proposed, True, cycle_index
        return self._record({**event, "status": "applied", "ocp_update": receipt})


def weighted_fatigue_residual(controller, muscle_weights):
    """sqrt(w)*(1-A/A_scale), so Bioptim's quadratic cost is sum(w*f²)."""
    from casadi import vertcat
    models = controller.model.muscles_dynamics_model
    if len(models) != len(muscle_weights):
        raise ValueError("Fatigue objective weight/model dimension mismatch")
    return vertcat(*[
        math.sqrt(weight) * (1 - controller.states[f"A_{model.muscle_name}"].cx / model.a_scale)
        for model, weight in zip(models, muscle_weights)
    ])


def update_bioptim_fatigue_cost(ocp, weights):
    """Replace the existing quadratic fatigue objective in-place via public API.

    The caller must disable compiled IPOPT caching. This updates the symbolic
    OCP; validation of the resulting physical RHO remains the benchmark's job.
    """
    from bioptim import Node, Objective
    from cocofest import CustomObjective
    weights = tuple(float(w) for w in weights)
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Fatigue weights must be finite and positive")
    matches = [(phase, objective) for phase, nlp in enumerate(ocp.nlp) for objective in nlp.J
               if objective and objective.custom_function in (
                   CustomObjective.minimize_overall_muscle_fatigue, weighted_fatigue_residual)]
    if len(matches) != 1:
        raise ValueError("Expected exactly one existing fatigue objective")
    phase, previous = matches[0]
    if previous.quadratic is not True or previous.target is not None:
        raise ValueError("RHO-PACE requires the untargeted quadratic fatigue objective")
    replacement = Objective(
        weighted_fatigue_residual, custom_type=type(previous.type), phase=phase,
        node=Node.ALL, weight=previous.weight, quadratic=True,
        integration_rule=previous.integration_rule, list_index=previous.list_index,
        muscle_weights=weights,
    )
    ocp.update_objectives(replacement)
    return {"ocp_cost_updated": True, "objective_index": previous.list_index,
            "phase": phase, "weights": weights, "integration": "bioptim.update_objectives"}
