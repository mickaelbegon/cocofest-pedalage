"""Guarded fixed-graph affine load-margin terminal cost, experimental.

The complete request/result contract is rechecked both before solve and before
transfer. Conditional context (unselected states and stimulation regime) must
have an explicit matching digest. Unknown or stale evidence deactivates the
objective; a failed terminal gate asks the caller to re-solve the same boundary.
"""
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .task_load_margin import TaskLoadMarginDirection, TaskLoadMarginPolicy, validate_task_load_margin
from .task_reserve_ocp import TaskReserveObjectiveBinding, TASK_RESERVE_PARAMETER_KEY


class TaskLoadMarginObjectiveBinding(TaskReserveObjectiveBinding):
    """Use the existing fixed task-reserve parameter channel with affine cost.

    Layout is [weight, unused_zero, center(n), gradient(n)]. The unused slot
    preserves compatibility with the already supported fatigue/reserve merge.
    It cannot coexist with another task-reserve objective in the same NLP.
    """
    def __init__(self, coordinates, *, task_context, model_sha256, physical_context,
                 conditional_context_sha256, policy=None):
        self.policy = policy or TaskLoadMarginPolicy()
        super().__init__(coordinates, task_context=task_context, model_sha256=model_sha256,
                         target=0., maximum_age_cycles=self.policy.maximum_age_cycles)
        if (not isinstance(conditional_context_sha256, str) or len(conditional_context_sha256) != 64
                or any(c not in "0123456789abcdef" for c in conditional_context_sha256)):
            raise ValueError("Explicit SHA256 of the conditioned context required")
        self.conditional_context_sha256 = conditional_context_sha256
        self.physical_context = json.loads(json.dumps(physical_context, allow_nan=False))
        self.direction = None
        self.solution_sha256 = None
        self.last_rejection = ()
        self.graph_signature_sha256 = sha256(json.dumps({
            "version": "task_load_margin_affine_v1_experimental",
            "base_graph": self.graph_signature_sha256,
        }, sort_keys=True).encode()).hexdigest()

    def validate_build_context(self, *, model_path, **actual_context):
        if model_path is None or sha256(Path(model_path).read_bytes()).hexdigest() != self.model_sha256:
            raise ValueError("Load-margin source model differs from the RHO model")
        for key, value in actual_context.items():
            if self.physical_context.get(key) != value:
                raise ValueError(f"Load-margin physical context differs for {key}")

    def objective(self, controller):
        from casadi import vertcat, dot
        point = vertcat(*[(controller.states[c.state_key].cx[c.index]-c.offset)/c.scale
                          for c in self.coordinates])
        p = controller.parameters[TASK_RESERVE_PARAMETER_KEY].cx
        n = self.dimension
        return -p[0]*dot(p[2+n:2+2*n], point-p[2:2+n])

    def _reasons(self, direction, *, coordinates, completed_cycles, now_monotonic_seconds,
                 conditional_context_sha256):
        if not isinstance(direction, TaskLoadMarginDirection):
            return ("missing_validated_direction",)
        reasons = list(validate_task_load_margin(direction.request, direction.result,
            policy=self.policy, completed_cycles=completed_cycles,
            now_monotonic_seconds=now_monotonic_seconds, coordinates=coordinates,
            model_sha256=self.model_sha256, task_context_sha256=self.task_context_sha256,
            coordinate_names=tuple(c.state_key for c in self.coordinates)))
        if conditional_context_sha256 != self.conditional_context_sha256:
            reasons.append("conditional_context_mismatch")
        try:
            direction.request.checkpoint.verify_files()
            digest = sha256(Path(direction.result.solution_artifact).read_bytes()).hexdigest()
            if self.direction is direction and self.solution_sha256 != digest:
                reasons.append("solution_artifact_changed")
        except (OSError, ValueError, TypeError):
            reasons.append("source_or_solution_artifact_invalid")
        return tuple(reasons)

    def update(self, nmpc, direction, *, weight, **validation):
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Finite positive load-margin weight required")
        reasons = self._reasons(direction, **validation)
        self.last_rejection = reasons
        if reasons:
            self.deactivate(nmpc)
            return {**self.summary(), "accepted": False, "reasons": reasons}
        digest = sha256(Path(direction.result.solution_artifact).read_bytes()).hexdigest()
        self._write(nmpc, np.r_[weight, 0., direction.request.center, direction.result.gradient])
        self.direction, self.solution_sha256 = direction, digest
        self.source_completed_cycles = direction.request.checkpoint.completed_cycles
        return {**self.summary(), "accepted": True, "reasons": ()}

    def validate_terminal_point(self, coordinates, **validation):
        if not self.values[0]:
            return {"active": False, "accepted": True, "fallback_resolve_required": False}
        reasons = self._reasons(self.direction, coordinates=coordinates, **validation)
        self.last_rejection = reasons
        return {"active": True, "accepted": not reasons, "reasons": reasons,
                "fallback_resolve_required": bool(reasons),
                "predicted_local_load_margin": (None if reasons else self.direction.value(coordinates)),
                "physiological_failure_certified": False}

    def summary(self):
        return {**super().summary(), "objective": "negative_affine_task_load_margin",
                "maximum_age_seconds": self.policy.maximum_age_seconds,
                "conditional_context_sha256": self.conditional_context_sha256,
                "last_rejection": self.last_rejection}


def physical_problem_view_without_load_binding(program):
    """Expose physical checkpoint arrays without the objective-only parameter.

    This view is for exact physical-state/history transfer to a baseline load
    oracle. It does not imply that the two objective graphs are identical.
    """
    binding = getattr(program, "task_reserve_binding", None)
    if not isinstance(binding, TaskLoadMarginObjectiveBinding):
        raise ValueError("A load-margin binding is required")
    key = TASK_RESERVE_PARAMETER_KEY
    class SourceView:
        def __init__(self):
            object.__setattr__(self, "parameter_bounds", {k: program.parameter_bounds[k]
                for k in program.parameter_bounds.keys() if k != key})
            object.__setattr__(self, "parameter_init", {k: program.parameter_init[k]
                for k in program.parameter_init.keys() if k != key})
        def __getattr__(self, name):
            return getattr(program, name)
        def __setattr__(self, name, value):
            setattr(program, name, value)
    return SourceView()


def restore_source_with_inactive_load_binding(path, program, *, completed_cycles):
    """Restore an exact baseline archive while preserving a new zero-weight channel.

    Only this explicitly identified objective-only parameter is omitted from
    the source digest. All physical arrays, original parameters, history and
    runtime state are verified by the original strict restoration routine.
    The experimental program as a whole has a different parameter layout.
    """
    from cocofest.simulation.rho_restart_checkpoint import restore_prepared_checkpoint
    binding = getattr(program, "task_reserve_binding", None)
    if not isinstance(binding, TaskLoadMarginObjectiveBinding) or binding.values[0] != 0:
        raise ValueError("Source restoration requires an inactive load-margin binding")
    key = TASK_RESERVE_PARAMETER_KEY
    for vector in (program.parameter_bounds[key].min, program.parameter_bounds[key].max,
                   program.parameter_init[key].init):
        if not np.all(np.asarray(vector) == 0):
            raise ValueError("Only a fresh all-zero objective-only parameter may be added to the source")
    result = restore_prepared_checkpoint(path, physical_problem_view_without_load_binding(program),
                                         completed_cycles=completed_cycles)
    return {**result, "experimental_objective_parameter_added": key,
            "source_physical_problem_restored": True, "whole_augmented_program_identical": False}


def periodic_ding_conditional_context(program, coordinates, *, states=None):
    """Audit the unselected boundary state for this specific periodic model.

    All five Ding states per muscle must be modeled. PW enters recruitment,
    not the fixed periodic calcium forcing, so it carries no additional past
    PW state in this formulation. The work accumulator is reset each cycle;
    crank phase is identified modulo 2*pi. Other model types are refused.
    """
    from cocofest.models.ding2007.ding2007_with_fatigue_periodic_node import DingModelPulseWidthFrequencyWithFatiguePeriodicNode
    nlp = program.nlp[0]
    models = nlp.model.muscles_dynamics_model
    expected = {f"{key}_{m.muscle_name}" for m in models for key in ("Cn", "F", "A", "Tau1", "Km")}
    if {c.state_key for c in coordinates} != expected or any(c.index != 0 for c in coordinates):
        raise ValueError("Conditional periodic context requires every Ding state")
    if set(nlp.states.keys()) != expected | {"theta", "omega", "E_prod"}:
        raise ValueError("Additional unmodeled states invalidate the periodic context")
    if any(not isinstance(m, DingModelPulseWidthFrequencyWithFatiguePeriodicNode) for m in models):
        raise ValueError("Only the fixed-history periodic-node Ding dynamics is supported")
    def value(key):
        if states is not None:
            return float(np.asarray(states[key]).reshape(-1)[-1])
        lower, upper = nlp.x_bounds[key].min[0, 0], nlp.x_bounds[key].max[0, 0]
        if lower != upper:
            raise ValueError(f"Boundary {key} must be fixed")
        return float(lower)
    theta, omega = value("theta"), value("omega")
    if not math.isfinite(theta) or not math.isfinite(omega):
        raise ValueError("Nonfinite mechanical boundary")
    context = {"version": "periodic_ding_complete_state_context_v1",
        "cos_theta": round(math.cos(theta), 8), "sin_theta": round(math.sin(theta), 8),
        "omega": round(omega, 8), "next_cycle_eprod_initial": 0.,
        "muscles": [{"name": m.muscle_name, "period_s": float(m._stim_interval),
            "tauc": float(m.tauc), "truncation": int(m.sum_stim_truncation),
            "post_stimulation_amplitude": m.post_stimulation_amplitude()} for m in models]}
    if abs(context["sin_theta"]) < 5e-9:
        context["sin_theta"] = 0.
    digest = sha256(json.dumps(context, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return digest, context
