"""Task-normalized initial work split for two independent isokinetic arms.

This is a supervisor initialization aid, not a bilateral OCP and not a
feasibility certificate.  It compares one certified reference cycle from each
arm.  The available PW-max work is explicitly retained as an *opportunity*;
it is used to normalize the utilization of the reference, never as a proof
that a larger target is feasible.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .physio_update import prescribed_force_fatigue_challenge


def _vector(value, name: str, *, positive: bool = False, nonnegative: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not result.size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite nonempty vector.")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must be strictly positive.")
    if nonnegative and np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative.")
    return result


def capacity_fatigability_measurement(
    *, certified: bool, reference_work_j: float, available_positive_power,
    phase_durations, current_capacity, rest_capacity, alpha_a, tau_fat,
    reference_force,
) -> dict[str, Any]:
    """Measure one arm's task-normalized damage from a certified reference.

    ``D`` is the capacity-opportunity-weighted force-induced decrement.  The
    normalized rate ``gamma=D/(W_ref/C)`` removes the bias that would otherwise
    favour an arm simply because its reference was run at a lower fraction of
    its PW-max opportunity.  It follows that the score ``C/gamma`` equals
    ``W_ref/D``.  Keeping both expressions in the audit makes that cancellation
    visible rather than hiding the role of the capacity opportunity.
    """
    result: dict[str, Any] = {
        "policy": "capacity_fatigability_reference_v1",
        "status": "held",
        "certified": certified is True,
        "opportunity_is_not_feasibility_certificate": True,
    }
    if certified is not True:
        return {**result, "reason": "uncertified_reference"}
    work = float(reference_work_j)
    if not math.isfinite(work) or work <= 0:
        return {**result, "reason": "invalid_reference_work"}
    power = np.asarray(available_positive_power, dtype=float)
    if power.ndim != 2 or not power.shape[0] or not power.shape[1] or not np.all(np.isfinite(power)) or np.any(power < 0):
        return {**result, "reason": "invalid_available_positive_power"}
    duration = _vector(phase_durations, "phase_durations", positive=True)
    if power.shape[1] != duration.size:
        return {**result, "reason": "incompatible_phase_count"}
    opportunity_by_muscle = power @ duration
    opportunity = float(opportunity_by_muscle.sum())
    if not math.isfinite(opportunity) or opportunity <= 0:
        return {**result, "reason": "zero_mechanical_opportunity"}
    try:
        challenge = prescribed_force_fatigue_challenge(
            current_capacity, rest_capacity, alpha_a, tau_fat, reference_force, duration,
        )
    except (TypeError, ValueError) as error:
        return {**result, "reason": "invalid_ding_challenge", "detail": str(error)}
    if not challenge["domain_valid"]:
        return {**result, "reason": "ding_challenge_outside_domain", "challenge": challenge}
    decrement = _vector(
        challenge["stimulus_capacity_decrement_ratio"], "stimulus_capacity_decrement_ratio", nonnegative=True,
    )
    if decrement.shape != opportunity_by_muscle.shape:
        return {**result, "reason": "incompatible_muscle_count"}
    contribution = opportunity_by_muscle / opportunity
    damage = float(np.dot(contribution, decrement))
    if not math.isfinite(damage) or damage <= 0:
        # Never replace a zero with a small arbitrary number: that would send
        # the complete bilateral target to this arm.
        return {**result, "reason": "zero_or_invalid_force_induced_damage", "challenge": challenge,
                "mechanical_opportunity_j_by_muscle": opportunity_by_muscle.tolist()}
    utilization = work / opportunity
    if not math.isfinite(utilization) or utilization <= 0:
        return {**result, "reason": "invalid_relative_utilization"}
    damage_per_relative_utilization = damage / utilization
    score = opportunity / damage_per_relative_utilization
    return {
        **result,
        "status": "measured",
        "reason": "certified_reference",
        "reference_work_j": work,
        "mechanical_opportunity_j": opportunity,
        "mechanical_opportunity_j_by_muscle": opportunity_by_muscle.tolist(),
        "mechanical_contribution": contribution.tolist(),
        "stimulus_capacity_decrement_ratio": decrement.tolist(),
        "weighted_force_induced_damage": damage,
        "relative_utilization": utilization,
        "damage_per_relative_utilization": damage_per_relative_utilization,
        "endurance_work_score_j": score,
        "challenge": challenge,
    }


def recommend_capacity_fatigability_split(
    right: Mapping[str, Any], left: Mapping[str, Any], *,
    total_equivalent_mean_torque_nm: float,
    minimum_arm_equivalent_mean_torque_nm: float = 0.0,
) -> dict[str, Any]:
    """Return a bounded next split, or an auditable hold decision.

    The projection uses only the declared minimum load.  In particular, it
    does *not* treat a PW-max opportunity as a hard arm-capacity bound.
    """
    total = float(total_equivalent_mean_torque_nm)
    minimum = float(minimum_arm_equivalent_mean_torque_nm)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("total_equivalent_mean_torque_nm must be finite and positive.")
    if not math.isfinite(minimum) or minimum < 0 or 2 * minimum > total:
        raise ValueError("minimum_arm_equivalent_mean_torque_nm is incompatible with the total.")
    result: dict[str, Any] = {
        "policy": "capacity_fatigability_initial_split_v1",
        "status": "held",
        "total_equivalent_mean_torque_nm": total,
        "minimum_arm_equivalent_mean_torque_nm": minimum,
        "opportunity_is_not_feasibility_certificate": True,
    }
    if right.get("status") != "measured" or left.get("status") != "measured":
        return {**result, "reason": "measurement_unavailable", "right_measurement": dict(right),
                "left_measurement": dict(left)}
    try:
        right_score = float(right["endurance_work_score_j"])
        left_score = float(left["endurance_work_score_j"])
    except (KeyError, TypeError, ValueError):
        return {**result, "reason": "score_unavailable"}
    if not (math.isfinite(right_score) and math.isfinite(left_score) and right_score > 0 and left_score > 0):
        return {**result, "reason": "zero_or_invalid_score"}
    proposed = right_score / (right_score + left_score)
    lower = minimum / total
    fraction = min(max(proposed, lower), 1.0 - lower)
    return {
        **result,
        "status": "proposed",
        "reason": "task_normalized_capacity_fatigability",
        "right_endurance_work_score_j": right_score,
        "left_endurance_work_score_j": left_score,
        "proposed_right_fraction": proposed,
        "right_fraction": fraction,
        "right_equivalent_mean_torque_nm": total * fraction,
        "left_equivalent_mean_torque_nm": total * (1.0 - fraction),
    }
