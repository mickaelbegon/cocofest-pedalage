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
    adaptation_enabled: bool = True
    update_every_cycles: int = 20
    smoothing: float = 0.2
    capacity_gain: float = 1.0
    min_relative_weight: float = 0.25
    max_relative_weight: float = 4.0
    max_log_step: float = math.log(1.1)
    max_cycles: int = 100
    adaptation_strategy: str = "capacity_feedback"
    projection_horizon_cycles: int = 500
    projection_substeps: int = 16
    projection_budget_seconds: float = 600.0
    projection_adjustment_factor: float = 1.5
    projection_tracking_mode: str = "projected_capacity"
    projection_async: bool = True
    # Keep the slow rollout off the fast RHO core when the campaign uses
    # taskset.  JSON lists are accepted and normalized to an immutable tuple.
    projection_worker_cpu_ids: tuple[int, ...] | list[int] | None = None
    target_cycle_seconds: float = 1.0
    projection_budget_fraction: float = 0.8
    projection_fatigue_guard: bool = True
    projection_minimum_relative_improvement: float = 0.01
    projection_minimum_absolute_improvement: float = 1e-6
    # The default exactly preserves the historic recruitment-space QP.  The
    # fatigue-aligned mode is experimental and only affects the slow compact
    # rollout allocator, never the differentiable RHO objective itself.
    projection_allocation_objective: str = "weighted_recruitment_v1"
    # The fatigue Hessian scales with (dA / A_rest)**2.  It is many orders of
    # magnitude smaller than the historical recruitment Hessian, so the old
    # 1e-3 tie-breaker would mask every fatigue-weight effect.
    projection_fatigue_reference_regularization: float = 1e-12

    def __post_init__(self):
        if not isinstance(self.adaptation_enabled, bool):
            raise ValueError("adaptation_enabled must be boolean")
        if not isinstance(self.projection_async, bool):
            raise ValueError("projection_async must be boolean")
        if self.projection_worker_cpu_ids is not None:
            try:
                cpu_ids = tuple(self.projection_worker_cpu_ids)
            except TypeError as error:
                raise ValueError("projection_worker_cpu_ids must be a sequence of CPU ids") from error
            if not cpu_ids or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0
                                  for cpu in cpu_ids):
                raise ValueError("projection_worker_cpu_ids must contain nonnegative integer CPU ids")
            if len(set(cpu_ids)) != len(cpu_ids):
                raise ValueError("projection_worker_cpu_ids must not contain duplicates")
            object.__setattr__(self, "projection_worker_cpu_ids", cpu_ids)
        if not isinstance(self.projection_fatigue_guard, bool):
            raise ValueError("projection_fatigue_guard must be boolean")
        for name in ("projection_minimum_relative_improvement", "projection_minimum_absolute_improvement"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("update_every_cycles", "max_cycles", "projection_horizon_cycles", "projection_substeps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.projection_tracking_mode not in {"exact", "projected_capacity"}:
            raise ValueError("projection_tracking_mode must be exact or projected_capacity")
        if self.projection_adjustment_factor <= 1:
            raise ValueError("projection_adjustment_factor must be greater than one")
        if self.adaptation_strategy not in {"capacity_feedback", "predictive_moment"}:
            raise ValueError("adaptation_strategy must be 'capacity_feedback' or 'predictive_moment'")
        if self.projection_allocation_objective not in {
                "weighted_recruitment_v1", "predicted_ding_fatigue_v1"}:
            raise ValueError("projection_allocation_objective must be weighted_recruitment_v1 or "
                             "predicted_ding_fatigue_v1")
        for name in ("smoothing", "capacity_gain", "min_relative_weight",
                     "max_relative_weight", "max_log_step", "projection_budget_seconds",
                     "projection_adjustment_factor", "target_cycle_seconds", "projection_budget_fraction",
                     "projection_fatigue_reference_regularization"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.smoothing > 1:
            raise ValueError("smoothing must not exceed 1")
        if self.projection_budget_fraction > 1:
            raise ValueError("projection_budget_fraction must not exceed 1")
        if not self.min_relative_weight <= 1 <= self.max_relative_weight:
            raise ValueError("Relative weight bounds must contain 1")

    @property
    def effective_projection_budget_seconds(self):
        """A slow update must finish within its next real-time call period."""
        return min(self.projection_budget_seconds, self.update_every_cycles
                   * self.target_cycle_seconds * self.projection_budget_fraction)


class RhoPaceController:
    """Transactional policy: weights change only after the OCP writer succeeds.

    ``apply_weights`` must actually update the OCP and return an audit mapping
    containing ``ocp_cost_updated=True``. Missing integration is a refusal.
    Exceptions propagate: a partially failed external write must stop the run.
    """

    def __init__(self, muscle_names, initial_weights, *, signed_crank_torque_nm,
                 parameters, initial_weight_basis, config=None, journal_path=None):
        self.config = config or RhoPaceConfig()
        self.arm = "RHO-PACE" if self.config.adaptation_enabled else "RHO-Physio"
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
        if not isinstance(initial_weight_basis, str) or not initial_weight_basis.strip():
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
        self._record({"event": "configuration", "arm": self.arm,
                      "policy": (
                          f"{self.config.adaptation_strategy}_v1"
                          if self.config.adaptation_enabled
                          else "static_initial_physiological_cost_v1"
                      ), "config": asdict(self.config),
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
        event = {"arm": self.arm, "signed_crank_torque_nm": self.signed_crank_torque_nm,
                 "weights": self.weights, "ocp_cost_connected": self.connected, **event}
        encoded = json.dumps(event, allow_nan=False, sort_keys=True)
        self.events.append(json.loads(encoded))
        if self.journal_path:
            with self.journal_path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
        return self.events[-1]

    def boundary(self, cycle_index, capacity_ratios, *, certified,
                 signed_crank_torque_nm, apply_weights: Callable | None,
                 predictive_weights=None, predictive_audit=None):
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
        if self.connected and not self.config.adaptation_enabled:
            self.last_cycle = cycle_index
            return self._record({**event, "status": "held", "reasons": ["static_physiological_weights"]})
        async_result_ready = (self.config.projection_async
                              and self.config.adaptation_strategy == "predictive_moment"
                              and predictive_weights is not None)
        if self.connected and cycle_index % self.config.update_every_cycles and not async_result_ready:
            self.last_cycle = cycle_index
            return self._record({**event, "status": "held", "reasons": ["slow_update_not_due"]})
        if not self.connected and cycle_index != 0:
            return self._record({**event, "status": "refused", "reasons": ["initial_cost_not_connected"]})
        proposed = self.initial_weights
        if self.connected:
            if self.config.adaptation_strategy == "predictive_moment":
                if predictive_weights is None:
                    self.last_cycle = cycle_index
                    return self._record({**event, "status": "held", "reasons": [
                        "predictive_proposal_unavailable"], "predictive_audit": predictive_audit})
                try:
                    normalized = normalize_relative_weights(predictive_weights)
                    if len(normalized) != len(self.weights):
                        raise ValueError("wrong dimension")
                    proposed = tuple(math.exp(value) for value in _project_centered_logs(
                        [math.log(value) for value in normalized],
                        lower=math.log(self.config.min_relative_weight),
                        upper=math.log(self.config.max_relative_weight)))
                    if self.config.projection_fatigue_guard and self.config.projection_tracking_mode == "projected_capacity":
                        if not predictive_audit or predictive_audit.get("selection_basis") not in {
                                "guarded_hold_incumbent", "guarded_fatigue_improvement"}:
                            raise ValueError("missing guarded rollout evidence")
                        if any(abs(math.log(w / old)) > self.config.max_log_step + 1e-12
                               for w, old in zip(proposed, self.weights)):
                            raise ValueError("predictive proposal exceeds evaluated trust region")
                except (TypeError, ValueError, OverflowError) as error:
                    self.last_cycle = cycle_index
                    return self._record({**event, "status": "held", "reasons": [
                        "invalid_predictive_proposal"], "predictive_audit": predictive_audit,
                        "error": f"{type(error).__name__}: {error}"})
            else:
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
        if predictive_audit is not None:
            event["predictive_audit"] = predictive_audit
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
