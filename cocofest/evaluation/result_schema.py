"""Additive benchmark result metadata, independent of solver objects.

Existing scalar result fields remain readable.  This module makes their units,
sampling populations and diagnostic availability explicit for schema v4.
"""

from __future__ import annotations

import math


RESULT_SCHEMA_VERSION = 4


def timing_populations(rows: list[dict], validated_prefix: int) -> dict:
    """Describe exactly which zero-based windows enter warm timing metrics.

    A window is one NLP solve, even for a monolithic FHO containing many cycles.
    The first window is excluded from warm samples; hybrid certifiers remain in
    the combined population and are excluded from the target-only population.
    """

    validated = rows[:validated_prefix]
    hot = [row for row in validated if row["window"] > 0]
    target = [row for row in hot if row.get("certifier") == "target_solver"]

    def population(selected: list[dict]) -> dict:
        output = {"window_indices": [row["window"] for row in selected]}
        for key in ("solver_time_s", "wall_time_s", "iterations"):
            output[key] = {
                "unit": "iterations" if key == "iterations" else "s/window",
                "sample_count": sum(
                    row.get(key) is not None and math.isfinite(float(row[key]))
                    for row in selected
                ),
            }
        return output

    return {
        "index_base": 0,
        "sample_unit": "optimization_window",
        "first_window_excluded": True,
        "validated_prefix_windows": int(validated_prefix),
        "hot": population(hot),
        "target_solver_only_hot": population(target),
        "per_cycle_time": {
            "definition": "sum of validated-prefix solve times / physically validated cycles",
            "unit": "s/cycle",
            "includes_first_window": True,
        },
    }


def audit_registry(result: dict) -> dict:
    """Index diagnostic evidence without conflating missing with passing.

    The detailed legacy fields remain the source of measured values.  Some
    diagnostics are informative, while others participate in certification.
    Their semantics are explicitly declared here, rather than deriving an
    overall scientific pass from the mere presence of a DOP853 report.
    """

    definitions = {
        "window_feasibility": ("certification", "optimization_window"),
        "mechanical_equivalence_audit": ("certification", "trajectory"),
        "isokinetic_audits": ("certification", "optimization_window"),
        "high_accuracy_trace_rollout": ("diagnostic", "open_loop_prefix"),
        "high_accuracy_cycle_milestones": ("diagnostic", "local_cycle_reset"),
        "integrator_map_initial_guess": ("diagnostic", "local_interval_reset"),
        "integrator_map_final_solution": ("diagnostic", "local_interval_reset"),
        "parametric_kkt_audits": ("diagnostic", "optimization_window"),
    }
    output = {}
    for field, (role, scope) in definitions.items():
        value = result.get(field)
        rows = value if isinstance(value, list) else [value] if value is not None else []
        populated = [row for row in rows if isinstance(row, dict)]
        unavailable = sum(row.get("available") is False for row in populated)
        output[field] = {
            "role": role,
            "scope": scope,
            "status": (
                "not_recorded" if not populated else
                "unavailable" if unavailable == len(populated) else
                "partial" if unavailable else "recorded"
            ),
            "record_count": len(populated),
            "unavailable_count": unavailable,
            "passes_tolerance": (
                all(bool(row["passes_tolerance"]) for row in populated)
                if populated and all("passes_tolerance" in row for row in populated)
                else None
            ),
        }
    return output
