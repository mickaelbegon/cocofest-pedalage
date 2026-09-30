"""Interpret a frozen-state one-cycle endurance feasibility probe.

The RHO solver is non-convex.  A failed solve is consequently *not* a proof
that no stimulation is able to complete the next cycle.  This module keeps
that distinction explicit for the post-failure probe: it reports a certified
admissible cycle when one is found and otherwise only reports the normalized
constraint residual of the failed numerical search.
"""

from __future__ import annotations

from math import isfinite
from typing import Any


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if isfinite(value) and value >= 0.0 else None


def _failed_attempt(result: dict[str, Any]) -> dict[str, Any] | None:
    """Return the last explicitly non-advanced solver attempt, when present."""

    attempts = (result.get("solver_attempt_accounting") or {}).get("attempts") or []
    for attempt in reversed(attempts):
        if isinstance(attempt, dict) and attempt.get("advanced") is False:
            return attempt
    return None


def normalized_constraint_deficit(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize a failed probe's primal residual by its declared tolerance.

    This is a numerical *constraint* deficit, not a biomechanical realization
    deficit nor an optimality gap.  It remains useful for comparing retries
    only because every probe preserves the same physical state and bounds.
    """

    attempt = _failed_attempt(result)
    if attempt is None:
        return {
            "available": False,
            "reason": "no_nonadvanced_solver_attempt",
            "kind": "normalized_constraint_residual",
        }
    feasibility = attempt.get("feasibility") or {}
    residual = _finite_nonnegative(feasibility.get("effective_primal_infeasibility"))
    tolerance = _finite_nonnegative(feasibility.get("feasibility_threshold"))
    if residual is None or tolerance is None or tolerance == 0.0:
        return {
            "available": False,
            "reason": "missing_finite_primal_residual_or_tolerance",
            "kind": "normalized_constraint_residual",
        }
    return {
        "available": True,
        "kind": "normalized_constraint_residual",
        "value": residual / tolerance,
        "raw_primal_residual": residual,
        "primal_feasibility_tolerance": tolerance,
        "interpretation": (
            "value <= 1 is necessary for the configured primal-feasibility gate; "
            "it is not a task-moment deficit or a proof of global infeasibility"
        ),
    }


def assess_frozen_state_one_cycle_probe(
    source_result: dict[str, Any], probe_result: dict[str, Any] | None,
    *, checkpoint_completed_windows: int | None = None,
    launcher_returncode: int | None = None,
) -> dict[str, Any]:
    """Build the normalized, deliberately conservative viability verdict.

    ``source_result`` is the row from a completed endurance run and
    ``probe_result`` is the row from a new one-cycle RHO initialized at the
    exact shifted primal after its final certified cycle.  The only positive
    conclusion is that a closed, admissible cycle was found and independently
    certified by the configured target solver.

    A negative outcome intentionally has no label such as ``fatigue_limit``:
    one local IPOPT failure cannot establish the non-existence of a feasible
    pulse-width trajectory in a non-convex OCP.
    """

    validated_cycles = int(source_result.get("validated_cycles") or 0)
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "frozen_state_closed_cycle_rho_probe_v1",
        "source_validated_cycles": validated_cycles,
        "checkpoint_completed_windows": checkpoint_completed_windows,
        "preserved_quantities": [
            "Ding fatigue and force states",
            "reduced mechanical state",
            "previous pulse-width/control history",
            "load, motion constraints and model parameters",
        ],
        "scientific_scope": (
            "A certified feasible probe establishes existence of one admissible "
            "next closed cycle for the configured NLP. A failed probe is only a "
            "local numerical non-find; it does not establish physiological "
            "exhaustion or global infeasibility."
        ),
        "global_infeasibility_proven": False,
    }
    if checkpoint_completed_windows is not None and checkpoint_completed_windows != validated_cycles:
        report.update(
            status="invalid_checkpoint_source_mismatch",
            certificate_valid=False,
            reason="checkpoint_does_not_follow_the_last_certified_source_cycle",
        )
        return report
    if validated_cycles < 1:
        report.update(
            status="source_has_no_certified_cycle",
            certificate_valid=False,
        )
        return report
    if probe_result is None:
        report.update(
            status="probe_execution_failed",
            certificate_valid=False,
            launcher_returncode=launcher_returncode,
        )
        return report

    certified = bool(probe_result.get("physical_success")) and int(
        probe_result.get("validated_cycles") or 0
    ) >= 1
    report.update(
        launcher_returncode=launcher_returncode,
        probe_solver=probe_result.get("solver"),
        probe_status=probe_result.get("status"),
        probe_validated_cycles=int(probe_result.get("validated_cycles") or 0),
        normalized_constraint_deficit=normalized_constraint_deficit(probe_result),
    )
    if certified:
        report.update(
            status="feasible_closed_cycle_found",
            certificate_valid=True,
            next_cycle_viable=True,
            conclusion=(
                "At least one admissible closed next cycle was found from the "
                "frozen final certified state."
            ),
        )
    else:
        report.update(
            status="no_feasible_closed_cycle_found_by_local_probe",
            certificate_valid=False,
            next_cycle_viable=None,
            conclusion=(
                "The configured local NLP probe did not certify a next cycle; "
                "this remains an unconfirmed endurance stop."
            ),
        )
    return report
