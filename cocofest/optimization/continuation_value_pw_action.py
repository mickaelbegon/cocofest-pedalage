"""Direct, reversible pulse-width actions for continuation-value experiments.

The purpose of this module is deliberately narrow. It creates an offline
*soft PW-target intervention*: a parametric tracking term biases the direct
pulse-width trace for one RHO cycle while physical PW bounds and nominal task
constraints remain unchanged. Thus an endpoint is produced by a real RHO
solve, rather than by editing fatigue states or fatigue weights.

It is not a real-time controller and it is not a numerical recovery trick.
An unsuccessful branch is consequently an indeterminate experimental result,
never an endurance label.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


PW_PREFIX = "last_pulse_width_"
_AUXILIARY_PREFIXES = ("pw_slew_increment_", "pulse_width_rate_")


def _trajectory_bounds(bounds, columns: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-control-node bounds without silently broadcasting a mismatch."""
    lower = np.asarray(bounds.min, dtype=float)
    upper = np.asarray(bounds.max, dtype=float)
    if lower.shape != upper.shape or lower.ndim != 2 or lower.shape[0] != 1:
        raise ValueError("Direct PW bounds must be one-row arrays of matching shape.")
    if lower.shape[1] == columns:
        return lower.copy(), upper.copy()
    if lower.shape[1] == 1:
        return np.repeat(lower, columns, axis=1), np.repeat(upper, columns, axis=1)
    if lower.shape[1] == 3 and columns > 2:
        # This is Bioptim's start/interior/end compact bounds layout.  It
        # cannot encode equality to a nonconstant PW trace and must not be
        # expanded then written back as if it could.  A parameterised tracking
        # objective is the correct intervention for this formulation.
        raise ValueError(
            "Compressed start/interior/end PW bounds cannot lock a nonconstant direct trace; "
            "use the parametric PW-tracking objective instead."
        )
    raise ValueError("Direct PW bounds do not match the pulse-width trace.")


def _write_bounds(bounds, lower: np.ndarray, upper: np.ndarray) -> None:
    """Write only a shape-preserving bound representation."""
    if bounds.min.shape[1] == lower.shape[1]:
        bounds.min[:, :] = lower
        bounds.max[:, :] = upper
    elif bounds.min.shape[1] == 1:
        if not (np.allclose(lower, lower[:, :1], rtol=0., atol=0.)
                and np.allclose(upper, upper[:, :1], rtol=0., atol=0.)):
            raise ValueError("A nodewise PW intervention cannot be stored in scalar bounds.")
        bounds.min[:, :] = lower[:, :1]
        bounds.max[:, :] = upper[:, :1]
    else:  # guarded by _trajectory_bounds, kept explicit for future representations.
        raise ValueError("Direct PW bound representation changed unexpectedly.")


def _phase_zero(values):
    """Accept lightweight mapping tests and Bioptim's one-phase containers."""
    if isinstance(values, Mapping) and all(isinstance(key, str) for key in values.keys()):
        return values
    try:
        if len(values) == 1:
            return values[0]
    except TypeError:
        pass
    raise ValueError("Direct PW action requires exactly one accessible NLP phase.")


def _names(nmpc, muscle_names: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(muscle_names)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Muscle names must be nonempty and unique.")
    nlp = nmpc.nlp[0]
    controls = set(_phase_zero(nlp.u_init).keys())
    auxiliary = sorted(key for key in controls if key.startswith(_AUXILIARY_PREFIXES))
    if auxiliary:
        raise ValueError(f"PW action requires direct controls, not auxiliary controls: {auxiliary}")
    missing = [PW_PREFIX + name for name in requested if PW_PREFIX + name not in controls]
    if missing:
        raise ValueError(f"RHO lacks direct PW controls: {missing}")
    return requested


@dataclass
class DirectPulseWidthAction:
    """Soft one-cycle PW target used only to create reachable pilot endpoints."""

    muscle_names: tuple[str, ...]
    direction: tuple[float, ...]
    amplitude_fraction: float
    tracking_activation: float = .1

    def __post_init__(self):
        if len(self.muscle_names) != len(self.direction) or not self.muscle_names:
            raise ValueError("PW action direction must match a nonempty muscle layout.")
        vector = np.asarray(self.direction, dtype=float)
        if (not np.all(np.isfinite(vector))
                or not math.isfinite(float(self.amplitude_fraction))
                or not 0. <= float(self.amplitude_fraction) <= 1.
                or not math.isfinite(float(self.tracking_activation))
                or not 0. < float(self.tracking_activation) <= 1.):
            raise ValueError("PW action requires a finite direction, amplitude in [0, 1], and tracking activation in (0, 1].")
        if np.linalg.norm(vector) > 0. and not math.isclose(float(np.linalg.norm(vector)), 1., rel_tol=0., abs_tol=1e-12):
            raise ValueError("PW action direction must have unit Euclidean norm.")

    def target_from_reference(self, nmpc, reference: Sequence[Sequence[float]]) -> tuple[np.ndarray, dict]:
        """Create a bounded constant offset target around a nominal PW trace."""
        names = _names(nmpc, self.muscle_names)
        nlp = nmpc.nlp[0]
        initial_guesses, bounds_by_key = _phase_zero(nlp.u_init), _phase_zero(nlp.u_bounds)
        base = np.asarray(reference, dtype=float)
        reference_nodes = np.asarray(initial_guesses[PW_PREFIX + names[0]].init).shape[1]
        if base.shape != (len(names), reference_nodes) or not np.all(np.isfinite(base)):
            raise ValueError("PW action reference shape differs from the compiled direct-control grid.")
        targets, applied = [], {}
        for name, coefficient in zip(names, self.direction, strict=True):
            key = PW_PREFIX + name
            trace = np.array(initial_guesses[key].init, dtype=float, copy=True)
            if trace.ndim != 2 or trace.shape[0] != 1 or not trace.shape[1] or not np.all(np.isfinite(trace)):
                raise ValueError(f"Direct PW seed is invalid for {name}.")
            compact = bounds_by_key[key]
            lower = np.asarray(compact.min, dtype=float)
            upper = np.asarray(compact.max, dtype=float)
            if (lower.shape != upper.shape or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
                    or np.any(upper < lower) or not np.all(np.isfinite(base[len(targets)]))):
                raise ValueError(f"Direct PW seed or bounds are invalid for {name}.")
            physical_lower, physical_upper = float(np.min(lower)), float(np.max(upper))
            span = physical_upper - physical_lower
            reference_trace = base[len(targets)]
            bound_roundoff_tolerance = max(1e-12, 2e-5 * span)
            if (np.any(reference_trace < physical_lower - bound_roundoff_tolerance)
                    or np.any(reference_trace > physical_upper + bound_roundoff_tolerance)):
                raise ValueError(f"PW tracking reference escapes physical bounds for {name}.")
            # IPOPT can return bound residuals of a few nanoseconds at this
            # scale. Canonicalize only this numerical roundoff before making
            # the bounded target; the realized control is still audited.
            reference_trace = np.clip(reference_trace, physical_lower, physical_upper)
            requested_offset = float(self.amplitude_fraction) * float(coefficient) * span
            # Preserve the requested muscle-direction contrast while tapering
            # its nodewise magnitude wherever the optimized reference is near
            # a physical bound. The applied action is therefore always logged,
            # rather than silently clipping a solver control or rejecting an
            # otherwise informative direction because one node is saturated.
            if requested_offset > 0.:
                available = np.maximum(0., physical_upper - reference_trace)
                applied_offset = np.minimum(requested_offset, available)
            elif requested_offset < 0.:
                available = np.maximum(0., reference_trace - physical_lower)
                applied_offset = -np.minimum(-requested_offset, available)
            else:
                applied_offset = np.zeros_like(reference_trace)
            shifted = reference_trace + applied_offset
            targets.append(shifted.copy())
            applied[name] = {
                "constant_offset_s": requested_offset,
                "requested_offset_s": requested_offset,
                "physical_range_s": span,
                "maximum_absolute_target_change_s": float(np.max(np.abs(shifted - reference_trace))),
                "mean_absolute_target_change_s": float(np.mean(np.abs(shifted - reference_trace))),
                "active_node_fraction": float(np.mean(np.abs(shifted - reference_trace) > 1e-14)),
            }
        return np.asarray(targets), {"kind": "parametric_pw_tracking_target_v1", "muscle_order": list(names),
                "direction": list(self.direction), "amplitude_fraction": float(self.amplitude_fraction),
                "tracking_activation": float(self.tracking_activation), "controls": applied,
                "task_constraints_unchanged": True,
                "interpretation": "soft one-cycle PW target; realized action must be audited"}


def predeclared_pw_actions(muscle_names: Sequence[str], directions: Mapping[str, Sequence[float]], *,
                           amplitude_fraction: float) -> list[dict]:
    """Create a train/holdout action table without any fatigue-weight action.

    The first two named directions are training contrasts; every remaining
    direction is a holdout contrast.  A zero direct-PW control is included.
    Keeping this small is intentional: the local continuation value is not
    identifiable from a high-dimensional collection of one-cycle endpoints.
    """
    names = tuple(muscle_names)
    if len(directions) < 3:
        raise ValueError("Require two training PW directions and at least one held-out direction.")
    actions = [{"id": "pw-unit", "partition": "control", "kind": "no_pw_constraint"}]
    for index, (label, vector) in enumerate(directions.items()):
        action = DirectPulseWidthAction(names, tuple(vector), amplitude_fraction)
        partition = "train" if index < 2 else "holdout"
        for sign, suffix in ((1., "plus"), (-1., "minus")):
            actions.append({"id": f"pw-{label}-{suffix}", "partition": partition,
                            "kind": "direct_pulse_width_trace_bias_v1",
                            "direction": (sign * np.asarray(action.direction)).tolist(),
                            "amplitude_fraction": action.amplitude_fraction,
                            "fatigue_weight_update": False})
    return actions


def direct_pw_continuation_plan(anchors: Sequence[Mapping], *, muscle_names: Sequence[str],
                                directions: Mapping[str, Sequence[float]], amplitude_fraction: float,
                                maximum_absolute_cycles: int = 240) -> dict:
    """Freeze the 21-branch direct-PW continuation design before execution.

    ``anchors`` must already have been independently authenticated by the
    calling runner.  This pure function does not resolve paths or inspect a
    numerical OCP; its role is to make the experimental partition and the
    non-weight action unambiguous and testable.
    """
    names = tuple(muscle_names)
    if type(maximum_absolute_cycles) is not int or maximum_absolute_cycles < 2:
        raise ValueError("Maximum absolute cycle count must be an integer >= 2.")
    rows = [dict(anchor) for anchor in anchors]
    ids = [row.get("id") for row in rows]
    required = {"id", "completed_cycles", "source_receipt", "source_receipt_sha256", "prepared_problem_sha256"}
    if (len(rows) != 3 or len(set(ids)) != len(ids) or any(not isinstance(value, str) or not value for value in ids)
            or any(set(row) != required for row in rows)):
        raise ValueError("Exactly three complete, uniquely identified PW anchors are required.")
    if any(type(row["completed_cycles"]) is not int or row["completed_cycles"] < 0 for row in rows):
        raise ValueError("PW anchor cycle indices must be nonnegative integers.")
    if any(not isinstance(row["source_receipt_sha256"], str) or len(row["source_receipt_sha256"]) != 64
           or not isinstance(row["prepared_problem_sha256"], str) or len(row["prepared_problem_sha256"]) != 64
           or not isinstance(row["source_receipt"], str) or not row["source_receipt"] for row in rows):
        raise ValueError("PW anchors require receipt paths and SHA-256 provenance.")
    actions = predeclared_pw_actions(names, directions, amplitude_fraction=amplitude_fraction)
    jobs = [{"id": f"{anchor['id']}-{action['id']}", "anchor_id": anchor["id"], "action": action,
             "intervention_cycles": 1, "continuation_policy": "unit_fatigue_reference"}
            for anchor in rows for action in actions]
    return {"schema_version": 1, "kind": "direct_pw_continuation_value_protocol",
            "muscle_order": list(names), "anchors": rows, "actions": actions, "jobs": jobs,
            "amplitude_fraction": float(amplitude_fraction),
            "return_policy": {"id": "unit_fatigue_reference", "fatigue_weights": dict.fromkeys(names, 1.),
                              "must_be_restored_after_intervention": True},
            "maximum_absolute_cycles": maximum_absolute_cycles,
            "actions_are_direct_pw_constraints": True, "fatigue_weight_updates_allowed": False,
            "activation_allowed": False,
            "label": "certified remaining cycles under fixed continuation policy",
            "failure_interpretation": "solver failure is indeterminate; use frozen feasibility and fatigue-rest counterfactual"}


def execute_direct_pw_branch(source, *, action: DirectPulseWidthAction | None, muscle_names: Sequence[str],
                             output_directory, build_runtime, solve_one_cycle, advance_one_cycle, audit,
                             save_witness, tolerance: float, reference_pw=None) -> dict:
    """Execute one certified RHO cycle with an optional direct-PW intervention.

    The tracking target/activation are changed only for the solve and reset
    before the native RHO advance and checkpoint export. A later fresh
    continuation receives the nominal physical problem plus the *real* state
    and PW history generated by the action. This prevents an experimental
    objective parameter from masquerading as a new physical constraint.

    Dependencies are passed in to keep this small primitive testable without a
    Bioptim runtime.  The caller owns process isolation and fresh-worker
    endpoint restoration.
    """
    from pathlib import Path

    from cocofest.optimization.task_reserve import DEFAULT_CONSTRAINT_GROUPS
    from cocofest.optimization.task_reserve_probe_adapter import IndependentAudit
    from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint, prepared_problem_arrays, restore_prepared_checkpoint

    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance <= 0.:
        raise ValueError("Audit tolerance must be finite and positive.")
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    source.verify_files()
    runtime = build_runtime()
    program, solver = runtime["nmpc"], runtime["solver"]
    names = _names(program, tuple(muscle_names))
    actual_names = tuple(model.muscle_name for model in program.nlp[0].model.muscles_dynamics_model)
    if actual_names != names:
        raise ValueError("PW action layout differs from the compiled muscle model.")
    restored = restore_prepared_checkpoint(Path(source.archive_path), program,
                                           completed_cycles=source.completed_cycles)
    if (restored.get("restored_problem_sha256") != source.prepared_problem_sha256
            or restored.get("stimulation_history_complete") is not True):
        raise RuntimeError("PW branch source did not restore exactly.")
    nlp = program.nlp[0]
    work = float(nlp.x_bounds["E_prod"].min[0, 2])
    if not math.isfinite(work) or work <= 0. or nlp.x_bounds["E_prod"].max[0, 2] != work:
        raise ValueError("PW branch source must prescribe a positive nominal work equality.")
    before = prepared_problem_arrays(program)
    binding = getattr(program, "pw_tracking_binding", None)
    if binding is None:
        raise ValueError("Direct PW continuation branch requires the compiled parametric PW-tracking graph.")
    reference = (np.vstack([np.asarray(_phase_zero(nlp.u_init)[PW_PREFIX + name].init, dtype=float).reshape(1, -1)
                            for name in names]) if reference_pw is None else np.asarray(reference_pw, dtype=float))
    if reference.shape != binding.target.shape or not np.all(np.isfinite(reference)):
        raise ValueError("Direct PW reference trace differs from the compiled binding layout.")
    action_receipt = {"kind": "pw_tracking_disabled", "task_constraints_unchanged": True,
                      "fatigue_weight_update": False, "tracking_activation": 0.}
    if action is not None:
        if action.muscle_names != names:
            raise ValueError("PW action and branch muscle layouts differ.")
        target, action_receipt = action.target_from_reference(program, reference)
        binding.update_nmpc(program, target=target, activation=action.tracking_activation)
        action_receipt["fatigue_weight_update"] = False
    try:
        solution = solve_one_cycle(program, solver)
        report = audit(solution, program, work, tolerance)
        if not isinstance(report, IndependentAudit):
            raise TypeError("Independent audit returned an invalid report.")
    finally:
        binding.update_nmpc(program, target=reference, activation=0.)
    after = prepared_problem_arrays(program)
    protected_prefixes = ("x_bounds:", "u_bounds:", "parameter_bounds:rho_fatigue_weights:",
                          "parameter_init:rho_fatigue_weights")
    for key in before:
        if key.startswith(protected_prefixes) and not np.array_equal(before[key], after[key]):
            raise RuntimeError(f"Direct PW tracking changed protected physical/fatigue data: {key}")
    passed = (report.passed and report.maximum_normalized_violation is not None
              and math.isfinite(report.maximum_normalized_violation)
              and 0. <= report.maximum_normalized_violation <= tolerance
              and set(DEFAULT_CONSTRAINT_GROUPS).issubset(report.groups))
    destination.mkdir(parents=True, exist_ok=False)
    row = {"branch_cycle": 1, "completed_cycles": source.completed_cycles + 1,
           "solver_status": str(getattr(solution, "status", "missing")), "certified": bool(passed),
           "independent_audit": dict(report.detail), "audit_tolerance": tolerance,
           "validated_constraint_groups": list(report.groups),
           "maximum_normalized_violation": report.maximum_normalized_violation,
           "action": action_receipt, "nominal_task_work_j": work,
           "nominal_problem_restored_before_advance": True}
    from bioptim import SolutionMerge
    achieved_controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
    realized = np.vstack([np.asarray(achieved_controls[PW_PREFIX + name], dtype=float).reshape(1, -1)
                          for name in names])
    row["reference_pw"] = reference.tolist()
    row["realized_pw"] = realized.tolist()
    row["realized_pw_tracking_error_s"] = float(np.max(np.abs(realized - reference)))
    row["realized_pw_action_from_reference_s"] = (realized - reference).tolist()
    endpoint = None
    if passed:
        row["witness"] = save_witness(solution, program, destination, 1, report)
        if not isinstance(row["witness"], str) or not Path(row["witness"]).is_file():
            raise RuntimeError("PW branch trajectory witness was not persisted.")
        row["witness_sha256"] = __import__("hashlib").sha256(Path(row["witness"]).read_bytes()).hexdigest()
        advance_one_cycle(program, solution, work)
        endpoint = export_prepared_checkpoint(destination / "cycle-1.npz", program,
            completed_cycles=source.completed_cycles + 1, model_path=Path(source.model_path))
    return {"schema_version": 1, "kind": "prepared_direct_pw_continuation_value_branch",
            "source_restore": restored, "source_completed_cycles": source.completed_cycles,
            "muscle_order": list(names), "nominal_work_j": work, "cycle": row,
            "endpoint": endpoint, "success": bool(passed), "compiled_solver_reused": True,
            "fresh_worker_replay_verified": False, "activation_allowed": False,
            "failure_interpretation": "failed nominal action solve is indeterminate"}
