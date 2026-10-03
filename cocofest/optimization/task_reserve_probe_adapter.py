"""Conservative one-cycle, full-NLP task-reserve probe.

Every call to :func:`probe_one_cycle` constructs a new program and restores
the prepared checkpoint before changing its terminal work equality.  An NLP
exit code is diagnostic only.  The independent audit recomputes the complete
constraint vector and checks every constraint and decision bound; missing
vectors, bounds, or a complete prepared history give an indeterminate result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import math

import numpy as np

from cocofest.optimization.independent_arm_backends import set_terminal_eprod_target
from cocofest.optimization.task_reserve import DEFAULT_CONSTRAINT_GROUPS, ProbeEvidence, TaskReserveProbe
from cocofest.simulation.rho_restart_checkpoint import restore_prepared_checkpoint


@dataclass(frozen=True)
class IndependentAudit:
    """All numerical RHO constraints were reevaluated from the solved primal."""

    passed: bool
    maximum_normalized_violation: float | None
    groups: tuple[str, ...]
    detail: Mapping[str, Any]


def _normalized_bound_violation(values: Any, lower: Any, upper: Any) -> float:
    """Use each row's bound magnitude, with a one-unit floor, as its scale."""
    value = np.asarray(values, dtype=float).reshape(-1)
    lb = np.asarray(lower, dtype=float).reshape(-1)
    ub = np.asarray(upper, dtype=float).reshape(-1)
    if (not value.size or value.size != lb.size or value.size != ub.size
            or not np.all(np.isfinite(value)) or np.any(np.isnan(lb))
            or np.any(np.isnan(ub)) or np.any(lb > ub)):
        raise ValueError("NLP values/bounds are absent, nonfinite, or have incompatible lengths")
    scale = np.maximum(1., np.maximum(np.where(np.isfinite(lb), np.abs(lb), 0.),
                                      np.where(np.isfinite(ub), np.abs(ub), 0.)))
    return float(np.max(np.maximum.reduce((lb - value, value - ub, np.zeros(value.size))) / scale))


def independent_full_nlp_audit(solution: Any, program: Any, *, target_work_j: float,
                               tolerance: float) -> IndependentAudit:
    """Recompute g(x), inspect all live bounds, and check explicit work/inputs.

    The complete g and x vectors cover collocation dynamics, custom/path
    constraints, bounds and any closure rows in this particular compiled NLP.
    This is discrete transcription feasibility, not independent DOP853 replay.
    It does not use IPOPT's ``inf_pr`` or its exit code.
    """
    try:
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        from casadi import Function
        from bioptim import SolutionMerge

        interface = getattr(program, "ocp_solver", None)
        graph = getattr(interface, "nlp", None)
        limits = getattr(interface, "limits", None)
        if not isinstance(graph, dict) or not all(key in graph for key in ("x", "g")):
            raise ValueError("compiled NLP constraint graph is unavailable")
        if not isinstance(limits, dict) or not all(key in limits for key in ("lbg", "ubg", "lbx", "ubx")):
            raise ValueError("complete live NLP bounds are unavailable")
        vector = np.asarray(solution.vector, dtype=float).reshape(-1)
        evaluator = Function("task_reserve_full_constraint_audit", [graph["x"]], [graph["g"]])
        constraints = np.asarray(evaluator(vector), dtype=float).reshape(-1)
        constraint_violation = _normalized_bound_violation(constraints, limits["lbg"], limits["ubg"])
        decision_violation = _normalized_bound_violation(vector, limits["lbx"], limits["ubx"])
        states = solution.decision_states(to_merge=SolutionMerge.NODES)
        controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
        if not states or not controls or not all(np.all(np.isfinite(np.asarray(v, float)))
                                                   for group in (states, controls) for v in group.values()):
            raise ValueError("state/control trajectories are missing or nonfinite")
        nlp = program.nlp[0]
        work_bounds = nlp.x_bounds["E_prod"]
        if not (math.isclose(float(work_bounds.min[0, 2]), target_work_j, rel_tol=0., abs_tol=1e-12)
                and math.isclose(float(work_bounds.max[0, 2]), target_work_j, rel_tol=0., abs_tol=1e-12)):
            raise ValueError("terminal work equality was not the requested target")
        work_terminal = float(np.asarray(states["E_prod"], dtype=float).reshape(-1)[-1])
        work_violation = abs(work_terminal - target_work_j) / max(1., abs(target_work_j))
        if not any(key.startswith("last_pulse_width_") for key in controls):
            raise ValueError("direct pulse-width controls and their history are missing")
        # No row is excluded: every compiled constraint and decision bound is
        # evaluated. These groups describe coverage by the complete graph;
        # the explicit E_prod check also guards a stale live-bound update.
        maximum = max(constraint_violation, decision_violation, work_violation)
        detail = {"constraint_rows": int(constraints.size), "decision_variables": int(vector.size),
                  "constraint_normalized_violation": constraint_violation,
                  "decision_normalized_violation": decision_violation,
                  "terminal_work_normalized_violation": work_violation,
                  "normalization": "row_bound_magnitude_with_one_unit_floor",
                  "scope": "complete_discrete_NLP_g_and_x; no continuous-ODE replay"}
        return IndependentAudit(maximum <= tolerance, maximum, DEFAULT_CONSTRAINT_GROUPS, detail)
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as error:
        return IndependentAudit(False, None, (), {"reason": f"{type(error).__name__}: {error}"})


def probe_one_cycle(request: TaskReserveProbe, *, build_runtime: Callable[[], Mapping[str, Any]],
                    solve_one_cycle: Callable[[Any, Any], Any],
                    audit: Callable[[Any, Any, float, float], IndependentAudit],
                    save_witness: Callable[[Any, Any, TaskReserveProbe, IndependentAudit], str],
                    tolerance: float = 1e-5) -> ProbeEvidence:
    """Restore an independent program, change only work, solve, and audit.

    Runtime/solver/audit failures are indeterminate. A valid witness remains
    possible after a nonzero solver status if every independent check passes.
    """
    start = perf_counter()
    source = request.checkpoint
    status, reason, residual, groups, witness = "not_run", None, None, (), None
    restored_source, passed = False, False
    try:
        source.verify_files()
        runtime = build_runtime()
        program, solver = runtime["nmpc"], runtime["solver"]
        restored = restore_prepared_checkpoint(Path(source.archive_path), program,
                                               completed_cycles=source.completed_cycles)
        restored_source = bool(restored.get("restored_problem_sha256") == source.prepared_problem_sha256
                               and restored.get("stimulation_history_complete") is True)
        if not restored_source:
            raise ValueError("fresh restoration did not reproduce the prepared state/history")
        original_work = float(program.nlp[0].x_bounds["E_prod"].min[0, 2])
        if not math.isfinite(original_work) or original_work <= 0:
            raise ValueError("nominal prepared E_prod target is not positive")
        target = original_work * request.work_scale
        set_terminal_eprod_target(program, target)
        solution = solve_one_cycle(program, solver)
        status = str(getattr(solution, "status", "missing"))
        report = audit(solution, program, target, tolerance)
        if not isinstance(report, IndependentAudit):
            raise TypeError("independent audit returned the wrong report type")
        residual, groups = report.maximum_normalized_violation, report.groups
        passed = bool(report.passed and residual is not None and math.isfinite(residual)
                      and 0. <= residual <= tolerance
                      and set(source.required_constraint_groups).issubset(groups))
        if passed:
            witness = save_witness(solution, program, request, report)
            if not isinstance(witness, str) or not witness:
                raise ValueError("validated trajectory was not saved")
        else:
            reason = str(report.detail.get("reason", "independent_full_nlp_audit_failed"))
    except Exception as error:
        passed, witness = False, None
        reason = f"{type(error).__name__}: {error}"
    return ProbeEvidence(request.work_scale, source.prepared_problem_sha256,
                         source.task_context_sha256, source.model_sha256, status,
                         residual, tolerance, bool(passed), witness, perf_counter() - start,
                         tuple(groups), restored_source, reason)
