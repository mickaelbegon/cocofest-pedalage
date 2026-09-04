r"""Retrospective validation of frozen-policy endurance rollouts using RHO only.

For an anchor cycle ``k``, the source policy is the certified RHO cycle ``k``
and the forecast origin is its end boundary.  A horizon ``H`` is compared with
the observed end boundary of RHO cycle ``k + H``.  The module delegates policy
adaptation and Ding propagation to :mod:`rho_rollout_adapter`; it does not
reimplement either model and never reads a full-horizon trajectory.

Only anchors whose source-policy gate is complete produce error metrics.
Unavailable future cycles are represented as censored records.  Failed gates,
malformed targets, and out-of-domain observed pulse widths remain explicit
invalid records with ``metrics=null``.
"""

from __future__ import annotations

from collections.abc import Sequence
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from cocofest.optimization.rho_rollout_adapter import (
    DEFAULT_HORIZONS,
    build_rho_endurance_rollout_report,
    select_certified_rho_cycle,
)


REPORT_SCHEMA = "cocofest-rho-retrospective-validation-v2"
DEFAULT_ANCHOR_STRIDE = 10
DEFAULT_SATURATION_THRESHOLD = 0.95
DEFAULT_OBSERVED_PULSE_WIDTH_TOLERANCE_S = 1e-10
DEFAULT_SCIENTIFIC_CRITERIA = {
    "minimum_valid_anchors_per_horizon": 3,
    "maximum_A_rest_normalized_rmse": 0.05,
    "maximum_Tau1_rest_normalized_rmse": 0.05,
    "maximum_Km_rest_normalized_rmse": 0.05,
    "maximum_utilization_rmse": 0.10,
    "maximum_policy_drift_normalized_pw_rmse": 0.10,
    "maximum_policy_drift_normalized_pw_p95": 0.20,
    "minimum_critical_muscle_match_fraction": 0.75,
}


def _scientific_criteria(values: dict[str, Any] | None) -> dict[str, float | int]:
    criteria = dict(DEFAULT_SCIENTIFIC_CRITERIA)
    supplied = dict(values or {})
    unknown = set(supplied) - set(criteria)
    if unknown:
        raise ValueError(f"Unknown scientific criteria: {sorted(unknown)}.")
    criteria.update(supplied)
    criteria["minimum_valid_anchors_per_horizon"] = _positive_integer(
        criteria["minimum_valid_anchors_per_horizon"],
        name="minimum_valid_anchors_per_horizon",
    )
    for name in (
        "maximum_A_rest_normalized_rmse",
        "maximum_Tau1_rest_normalized_rmse",
        "maximum_Km_rest_normalized_rmse",
        "maximum_utilization_rmse",
        "maximum_policy_drift_normalized_pw_rmse",
        "maximum_policy_drift_normalized_pw_p95",
    ):
        value = float(criteria[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        criteria[name] = value
    match = float(criteria["minimum_critical_muscle_match_fraction"])
    if not math.isfinite(match) or not 0.0 <= match <= 1.0:
        raise ValueError("minimum_critical_muscle_match_fraction must be in [0, 1].")
    criteria["minimum_critical_muscle_match_fraction"] = match
    return criteria


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a strictly positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a strictly positive integer.") from error
    if integer < 1 or integer != value:
        raise ValueError(f"{name} must be a strictly positive integer.")
    return integer


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a non-negative integer.") from error
    if integer < 0 or integer != value:
        raise ValueError(f"{name} must be a non-negative integer.")
    return integer


def _file_stamp(path: Path) -> dict[str, Any]:
    digest = sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "size_bytes": size}


def _finite_number(value: Any) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _json_configuration(value: Any) -> Any:
    """Keep production options auditable while making injected tests serializable."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("adapter_options must not contain non-finite values.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_configuration(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_configuration(item) for item in value]
    return {"injected_object_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _actual_cycle_observation(
    cycle,
    *,
    muscle_names: Sequence[str],
    model_parameters: Sequence[dict[str, Any]],
    saturation_threshold: float,
    pulse_width_tolerance_s: float,
) -> dict[str, Any]:
    """Extract target state and observed recruitment utilization without clipping."""

    by_muscle = {str(row["muscle"]): row for row in model_parameters}
    if set(by_muscle) != set(muscle_names):
        raise ValueError("Target muscle names do not match the adapted model parameters.")

    final_states = np.empty((len(muscle_names), 3), dtype=float)
    utilization = np.empty((cycle.stimulations_per_cycle, len(muscle_names)), dtype=float)
    normalized_pulse_width = np.empty_like(utilization)
    domain_violations: list[dict[str, Any]] = []
    tolerated_bound_excursions: list[dict[str, Any]] = []
    first_control = cycle.cycle_index * cycle.stimulations_per_cycle
    last_control = first_control + cycle.stimulations_per_cycle
    for muscle_index, muscle in enumerate(muscle_names):
        for state_index, state_name in enumerate(("A", "Tau1", "Km")):
            key = f"{state_name}_{muscle}"
            if key not in cycle.states:
                raise ValueError(f"Target RHO cycle is missing state {key!r}.")
            final_states[muscle_index, state_index] = cycle.states[key][cycle.end_column]

        key = f"last_pulse_width_{muscle}"
        if key not in cycle.controls:
            raise ValueError(f"Target RHO cycle is missing control {key!r}.")
        pulse_widths = np.asarray(cycle.controls[key][first_control:last_control], dtype=float)
        parameter = by_muscle[muscle]
        pd0 = float(parameter["pd0"])
        pdt = float(parameter["pdt"])
        maximum = float(parameter["pulse_width_max"])
        if pdt <= 0.0 or maximum <= pd0:
            raise ValueError(f"Invalid pulse-width model parameters for muscle {muscle!r}.")
        invalid = (~np.isfinite(pulse_widths)) | (
            pulse_widths < pd0 - pulse_width_tolerance_s
        ) | (pulse_widths > maximum + pulse_width_tolerance_s)
        for interval_index in np.flatnonzero(invalid):
            domain_violations.append(
                {
                    "muscle": muscle,
                    "muscle_index": muscle_index,
                    "interval_index": int(interval_index),
                    "pulse_width_s": _finite_number(pulse_widths[interval_index]),
                    "minimum_s": pd0,
                    "maximum_s": maximum,
                    "tolerance_s": pulse_width_tolerance_s,
                }
            )
        tolerated = np.isfinite(pulse_widths) & (~invalid) & (
            (pulse_widths < pd0) | (pulse_widths > maximum)
        )
        for interval_index in np.flatnonzero(tolerated):
            value = float(pulse_widths[interval_index])
            tolerated_bound_excursions.append(
                {
                    "muscle": muscle,
                    "muscle_index": muscle_index,
                    "interval_index": int(interval_index),
                    "pulse_width_s": value,
                    "signed_bound_excursion_s": value - (pd0 if value < pd0 else maximum),
                }
            )
        denominator = -math.expm1(-(maximum - pd0) / pdt)
        utilization[:, muscle_index] = -np.expm1(-(pulse_widths - pd0) / pdt) / denominator
        normalized_pulse_width[:, muscle_index] = (pulse_widths - pd0) / (maximum - pd0)

    if domain_violations:
        return {
            "valid": False,
            "reason": "observed_pulse_width_out_of_domain",
            "domain_violations": domain_violations,
            "pulse_width_bound_tolerance_s": pulse_width_tolerance_s,
            "clipping_applied": False,
        }

    slow_state_violations = []
    for muscle_index, muscle in enumerate(muscle_names):
        parameter = by_muscle[muscle]
        rest = np.asarray(
            [parameter["A_rest"], parameter["Tau1_rest"], parameter["Km_rest"]],
            dtype=float,
        )
        state = final_states[muscle_index]
        if (
            not np.all(np.isfinite(state))
            or state[0] < -1e-9
            or state[0] > rest[0] + 1e-9
            or state[1] < rest[1] - 1e-9
            or state[2] < rest[2] - 1e-9
        ):
            slow_state_violations.append(
                {
                    "muscle": muscle,
                    "muscle_index": muscle_index,
                    "state": [_finite_number(value) for value in state],
                    "rest_state": rest.tolist(),
                    "tolerance": 1e-9,
                }
            )
    if slow_state_violations:
        return {
            "valid": False,
            "reason": "non_physiological_observed_slow_state",
            "slow_state_violations": slow_state_violations,
        }

    critical_flat = int(np.argmax(utilization))
    interval_index, muscle_index = np.unravel_index(critical_flat, utilization.shape)
    saturated = utilization >= saturation_threshold
    resting = np.asarray([by_muscle[name]["A_rest"] for name in muscle_names], dtype=float)
    return {
        "valid": True,
        "final_slow_states": final_states,
        "final_capacity_ratios": final_states[:, 0] / resting,
        "normalized_pulse_width": normalized_pulse_width,
        "pulse_width_audit": {
            "bound_tolerance_s": pulse_width_tolerance_s,
            "tolerated_bound_excursion_count": len(tolerated_bound_excursions),
            "tolerated_bound_excursions": tolerated_bound_excursions,
            "clipping_applied": False,
        },
        "recruitment": {
            "maximum_utilization": float(utilization[interval_index, muscle_index]),
            "critical_location": {
                "interval_index": int(interval_index),
                "muscle_index": int(muscle_index),
                "muscle": muscle_names[muscle_index],
            },
            "saturation_threshold": float(saturation_threshold),
            "saturated_sample_count": int(np.count_nonzero(saturated)),
            "saturated_sample_fraction": float(np.mean(saturated)),
            "utilization_definition": (
                "effective_recruitment(PW)/effective_recruitment(PW_max); capacity cancels"
            ),
        },
    }


def _state_error_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    model_parameters: Sequence[dict[str, Any]],
    muscle_names: Sequence[str],
) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    expected = (len(muscle_names), 3)
    if predicted.shape != expected or observed.shape != expected:
        raise ValueError(f"Slow-state arrays must both have shape {expected}.")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(observed)):
        raise ValueError("Slow-state comparison requires finite values.")
    normalizers = np.asarray(
        [
            [row["A_rest"], row["Tau1_rest"], row["Km_rest"]]
            for row in model_parameters
        ],
        dtype=float,
    )
    if normalizers.shape != expected or np.any(normalizers <= 0.0):
        raise ValueError("Resting slow-state normalizers must be finite and positive.")
    errors = predicted - observed
    result: dict[str, Any] = {}
    for state_index, state_name in enumerate(("A", "Tau1", "Km")):
        values = errors[:, state_index]
        normalized = values / normalizers[:, state_index]
        result[state_name] = {
            "rmse": float(np.sqrt(np.mean(values**2))),
            "mean_absolute_error": float(np.mean(np.abs(values))),
            "maximum_absolute_error": float(np.max(np.abs(values))),
            "rest_normalized_rmse": float(np.sqrt(np.mean(normalized**2))),
            "per_muscle": [
                {
                    "muscle": muscle,
                    "predicted": float(predicted[index, state_index]),
                    "observed": float(observed[index, state_index]),
                    "error": float(values[index]),
                    "rest_normalized_error": float(normalized[index]),
                }
                for index, muscle in enumerate(muscle_names)
            ],
        }
    return result


def _forecast_outcome_audit(
    forecast: dict[str, Any],
    *,
    model_parameters: Sequence[dict[str, Any]],
    expected_utilization_samples: int,
    physiological_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Gate non-finite or physiologically inadmissible forecast outcomes."""

    reasons: list[dict[str, Any]] = []
    states = np.asarray(forecast.get("final_slow_states"), dtype=float)
    expected_shape = (len(model_parameters), 3)
    if states.shape != expected_shape or not np.all(np.isfinite(states)):
        reasons.append(
            {
                "code": "non_finite_or_malformed_final_slow_states",
                "observed_shape": list(states.shape),
                "expected_shape": list(expected_shape),
            }
        )
    else:
        for index, (state, parameter) in enumerate(zip(states, model_parameters, strict=True)):
            rest = np.asarray(
                [parameter["A_rest"], parameter["Tau1_rest"], parameter["Km_rest"]],
                dtype=float,
            )
            if (
                state[0] < -physiological_tolerance
                or state[0] > rest[0] + physiological_tolerance
                or state[1] < rest[1] - physiological_tolerance
                or state[2] < rest[2] - physiological_tolerance
            ):
                reasons.append(
                    {
                        "code": "non_physiological_final_slow_state",
                        "muscle": parameter["muscle"],
                        "muscle_index": index,
                        "state": state.tolist(),
                        "rest_state": rest.tolist(),
                        "tolerance": physiological_tolerance,
                    }
                )
    recruitment = forecast.get("last_cycle_recruitment") or {}
    maximum = recruitment.get("maximum_finite_utilization")
    saturation_fraction = recruitment.get("saturated_sample_fraction")
    finite_count = int(recruitment.get("finite_sample_count", 0))
    if (
        maximum is None
        or not math.isfinite(float(maximum))
        or saturation_fraction is None
        or not math.isfinite(float(saturation_fraction))
        or finite_count != expected_utilization_samples
        or recruitment.get("critical_location") is None
    ):
        reasons.append(
            {
                "code": "non_finite_or_incomplete_predicted_utilization",
                "finite_sample_count": finite_count,
                "expected_sample_count": expected_utilization_samples,
                "maximum_finite_utilization": maximum,
                "saturated_sample_fraction": saturation_fraction,
            }
        )
    if forecast.get("feasible") is not True:
        reasons.append(
            {
                "code": "physiologically_infeasible_recruitment_forecast",
                "first_failure": forecast.get("first_failure"),
            }
        )
    return {"valid": not reasons, "reasons": reasons}


def _policy_drift_metrics(
    anchor_observation: dict[str, Any],
    target_observation: dict[str, Any],
    *,
    interval_count: int,
    muscle_names: Sequence[str],
) -> dict[str, Any]:
    anchor = np.asarray(anchor_observation["normalized_pulse_width"], dtype=float)
    target = np.asarray(target_observation["normalized_pulse_width"], dtype=float)
    if anchor.shape != target.shape or anchor.shape[0] != interval_count:
        raise ValueError("Anchor and target normalized pulse-width policies must share phase layout.")
    if not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(target)):
        raise ValueError("Policy-drift metrics require finite normalized pulse widths.")
    difference = target - anchor
    absolute = np.abs(difference)
    anchor_index = np.unravel_index(np.argmax(anchor), anchor.shape)
    target_index = np.unravel_index(np.argmax(target), target.shape)
    raw_phase_error = abs(int(anchor_index[0]) - int(target_index[0]))
    circular_phase_error = min(raw_phase_error, interval_count - raw_phase_error)
    return {
        "definition": "(PW-PD0)/(PW_max-PD0); target minus anchor at matched muscle/phase",
        "sample_count": int(difference.size),
        "bias": float(np.mean(difference)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "p95_absolute": float(np.percentile(absolute, 95.0)),
        "maximum_absolute": float(np.max(absolute)),
        "critical_policy_location_anchor": {
            "interval_index": int(anchor_index[0]),
            "muscle_index": int(anchor_index[1]),
            "muscle": muscle_names[int(anchor_index[1])],
        },
        "critical_policy_location_target": {
            "interval_index": int(target_index[0]),
            "muscle_index": int(target_index[1]),
            "muscle": muscle_names[int(target_index[1])],
        },
        "critical_muscle_changed": int(anchor_index[1]) != int(target_index[1]),
        "critical_phase_circular_change_intervals": int(circular_phase_error),
    }


def _comparison_metrics(
    forecast: dict[str, Any],
    observation: dict[str, Any],
    *,
    muscle_names: Sequence[str],
    model_parameters: Sequence[dict[str, Any]],
    interval_count: int,
    anchor_observation: dict[str, Any],
) -> dict[str, Any]:
    predicted_states = np.asarray(forecast["final_slow_states"], dtype=float)
    observed_states = np.asarray(observation["final_slow_states"], dtype=float)
    state_errors = _state_error_metrics(
        predicted_states, observed_states, model_parameters, muscle_names
    )
    predicted_recruitment = forecast["last_cycle_recruitment"]
    observed_recruitment = observation["recruitment"]
    predicted_location = predicted_recruitment["critical_location"]
    observed_location = observed_recruitment["critical_location"]
    if interval_count <= 0:
        raise ValueError("interval_count must be strictly positive.")
    raw_phase_error = abs(
        int(predicted_location["interval_index"])
        - int(observed_location["interval_index"])
    )
    circular_phase_error = min(raw_phase_error, interval_count - raw_phase_error)
    predicted_max = predicted_recruitment["maximum_finite_utilization"]
    observed_max = observed_recruitment["maximum_utilization"]
    predicted_fraction = predicted_recruitment["saturated_sample_fraction"]
    observed_fraction = observed_recruitment["saturated_sample_fraction"]
    resting = np.asarray([row["A_rest"] for row in model_parameters], dtype=float)
    predicted_capacity = predicted_states[:, 0] / resting
    observed_capacity = observed_states[:, 0] / resting
    return {
        "slow_state_errors": state_errors,
        "capacity_ratio": {
            "predicted": predicted_capacity.tolist(),
            "observed": observed_capacity.tolist(),
            "error": (predicted_capacity - observed_capacity).tolist(),
            "minimum_predicted": float(np.min(predicted_capacity)),
            "minimum_observed": float(np.min(observed_capacity)),
            "minimum_error": float(np.min(predicted_capacity) - np.min(observed_capacity)),
        },
        "recruitment": {
            "maximum_utilization_predicted": predicted_max,
            "maximum_utilization_observed": observed_max,
            "maximum_utilization_error": float(predicted_max - observed_max),
            "saturated_sample_fraction_predicted": predicted_fraction,
            "saturated_sample_fraction_observed": observed_fraction,
            "saturated_sample_fraction_error": float(predicted_fraction - observed_fraction),
            "critical_location_predicted": predicted_location,
            "critical_location_observed": observed_location,
            "critical_muscle_match": predicted_location["muscle"] == observed_location["muscle"],
            "critical_phase_circular_error_intervals": int(circular_phase_error),
        },
        "observed_policy_drift": _policy_drift_metrics(
            anchor_observation,
            observation,
            interval_count=interval_count,
            muscle_names=muscle_names,
        ),
    }


def _aggregate_by_horizon(
    records: Sequence[dict[str, Any]],
    horizons: Sequence[int],
    scientific_criteria: dict[str, float | int],
) -> list[dict[str, Any]]:
    """Aggregate only valid comparisons; invalid/censored rows stay counted."""

    summaries = []
    for horizon in horizons:
        selected = [record for record in records if record["horizon_cycles"] == horizon]
        valid = [record for record in selected if record["status"] == "valid"]
        status_counts = {
            status: sum(record["status"] == status for record in selected)
            for status in sorted({record["status"] for record in selected})
        }
        if not valid:
            summaries.append(
                {
                    "horizon_cycles": horizon,
                    "processing_status": "no_valid_records",
                    "scientific_validation_status": "not_evaluable",
                    "record_counts": status_counts,
                    "metrics": None,
                    "criteria_results": [],
                }
            )
            continue
        state_metrics = {}
        for state in ("A", "Tau1", "Km"):
            rows = [
                muscle_row
                for record in valid
                for muscle_row in record["metrics"]["slow_state_errors"][state]["per_muscle"]
            ]
            errors = np.asarray([row["error"] for row in rows], dtype=float)
            normalized = np.asarray([row["rest_normalized_error"] for row in rows], dtype=float)
            state_metrics[state] = {
                "sample_count": int(errors.size),
                "bias": float(np.mean(errors)),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mean_absolute_error": float(np.mean(np.abs(errors))),
                "rest_normalized_rmse": float(np.sqrt(np.mean(normalized**2))),
            }
        recruitment_rows = [record["metrics"]["recruitment"] for record in valid]
        utilization_errors = np.asarray(
            [row["maximum_utilization_error"] for row in recruitment_rows], dtype=float
        )
        saturation_errors = np.asarray(
            [row["saturated_sample_fraction_error"] for row in recruitment_rows], dtype=float
        )
        capacity_errors = np.asarray(
            [record["metrics"]["capacity_ratio"]["minimum_error"] for record in valid],
            dtype=float,
        )
        policy_drift_rows = [record["metrics"]["observed_policy_drift"] for record in valid]
        policy_rmse = np.asarray([row["rmse"] for row in policy_drift_rows], dtype=float)
        policy_p95 = np.asarray([row["p95_absolute"] for row in policy_drift_rows], dtype=float)
        metrics = {
            "valid_anchor_count": len(valid),
            "slow_state_errors": state_metrics,
            "minimum_capacity_ratio": {
                "bias": float(np.mean(capacity_errors)),
                "rmse": float(np.sqrt(np.mean(capacity_errors**2))),
            },
            "maximum_utilization": {
                "bias": float(np.mean(utilization_errors)),
                "rmse": float(np.sqrt(np.mean(utilization_errors**2))),
            },
            "saturated_sample_fraction": {
                "bias": float(np.mean(saturation_errors)),
                "rmse": float(np.sqrt(np.mean(saturation_errors**2))),
            },
            "observed_policy_drift": {
                "normalized_pw_rmse_across_anchors": float(
                    np.sqrt(np.mean(policy_rmse**2))
                ),
                "normalized_pw_p95_worst_anchor": float(np.max(policy_p95)),
                "critical_muscle_change_fraction": float(
                    np.mean([row["critical_muscle_changed"] for row in policy_drift_rows])
                ),
                "critical_phase_mean_absolute_change_intervals": float(
                    np.mean(
                        [
                            row["critical_phase_circular_change_intervals"]
                            for row in policy_drift_rows
                        ]
                    )
                ),
            },
            "critical_muscle_match_fraction": float(
                np.mean([row["critical_muscle_match"] for row in recruitment_rows])
            ),
            "critical_phase_mean_absolute_error_intervals": float(
                np.mean(
                    [
                        row["critical_phase_circular_error_intervals"]
                        for row in recruitment_rows
                    ]
                )
            ),
            "forecast_feasible_fraction": float(
                np.mean([record["forecast_feasible"] for record in valid])
            ),
        }
        criteria_results = [
            {
                "criterion": "minimum_valid_anchors_per_horizon",
                "observed": len(valid),
                "operator": ">=",
                "threshold": scientific_criteria["minimum_valid_anchors_per_horizon"],
                "passed": len(valid)
                >= scientific_criteria["minimum_valid_anchors_per_horizon"],
            },
            *[
                {
                    "criterion": f"maximum_{state}_rest_normalized_rmse",
                    "observed": state_metrics[state]["rest_normalized_rmse"],
                    "operator": "<=",
                    "threshold": scientific_criteria[
                        f"maximum_{state}_rest_normalized_rmse"
                    ],
                    "passed": state_metrics[state]["rest_normalized_rmse"]
                    <= scientific_criteria[f"maximum_{state}_rest_normalized_rmse"],
                }
                for state in ("A", "Tau1", "Km")
            ],
            {
                "criterion": "maximum_utilization_rmse",
                "observed": metrics["maximum_utilization"]["rmse"],
                "operator": "<=",
                "threshold": scientific_criteria["maximum_utilization_rmse"],
                "passed": metrics["maximum_utilization"]["rmse"]
                <= scientific_criteria["maximum_utilization_rmse"],
            },
            {
                "criterion": "maximum_policy_drift_normalized_pw_rmse",
                "observed": metrics["observed_policy_drift"][
                    "normalized_pw_rmse_across_anchors"
                ],
                "operator": "<=",
                "threshold": scientific_criteria[
                    "maximum_policy_drift_normalized_pw_rmse"
                ],
                "passed": metrics["observed_policy_drift"][
                    "normalized_pw_rmse_across_anchors"
                ]
                <= scientific_criteria["maximum_policy_drift_normalized_pw_rmse"],
            },
            {
                "criterion": "maximum_policy_drift_normalized_pw_p95",
                "observed": metrics["observed_policy_drift"][
                    "normalized_pw_p95_worst_anchor"
                ],
                "operator": "<=",
                "threshold": scientific_criteria[
                    "maximum_policy_drift_normalized_pw_p95"
                ],
                "passed": metrics["observed_policy_drift"][
                    "normalized_pw_p95_worst_anchor"
                ]
                <= scientific_criteria["maximum_policy_drift_normalized_pw_p95"],
            },
            {
                "criterion": "minimum_critical_muscle_match_fraction",
                "observed": metrics["critical_muscle_match_fraction"],
                "operator": ">=",
                "threshold": scientific_criteria[
                    "minimum_critical_muscle_match_fraction"
                ],
                "passed": metrics["critical_muscle_match_fraction"]
                >= scientific_criteria["minimum_critical_muscle_match_fraction"],
            },
        ]
        summaries.append(
            {
                "horizon_cycles": horizon,
                "processing_status": "complete",
                "scientific_validation_status": (
                    "passed" if all(row["passed"] for row in criteria_results) else "failed"
                ),
                "record_counts": status_counts,
                "metrics": metrics,
                "criteria_results": criteria_results,
            }
        )
    return summaries


def build_rho_retrospective_validation_report(
    source: str | Path,
    reduced_profile: str | Path,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    anchors: Sequence[int] | str | None = None,
    anchor_stride: int = DEFAULT_ANCHOR_STRIDE,
    saturation_threshold: float = DEFAULT_SATURATION_THRESHOLD,
    observed_pulse_width_tolerance_s: float = DEFAULT_OBSERVED_PULSE_WIDTH_TOLERANCE_S,
    scientific_criteria: dict[str, Any] | None = None,
    adapter_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate frozen-policy predictions against later cycles of one RHO archive."""

    source = Path(source)
    reduced_profile = Path(reduced_profile)
    horizons = tuple(_positive_integer(value, name="horizon") for value in horizons)
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must contain unique positive integers.")
    anchor_stride = _positive_integer(anchor_stride, name="anchor_stride")
    saturation_threshold = float(saturation_threshold)
    if not math.isfinite(saturation_threshold) or not 0.0 < saturation_threshold <= 1.0:
        raise ValueError("saturation_threshold must be in (0, 1].")
    observed_pulse_width_tolerance_s = float(observed_pulse_width_tolerance_s)
    if (
        not math.isfinite(observed_pulse_width_tolerance_s)
        or observed_pulse_width_tolerance_s < 0.0
        or observed_pulse_width_tolerance_s > 1e-8
    ):
        raise ValueError(
            "observed_pulse_width_tolerance_s must be finite and in [0, 1e-8] s; "
            "it is only for solver-scale numerical noise."
        )
    criteria = _scientific_criteria(scientific_criteria)
    adapter_options = dict(adapter_options or {})
    forbidden = {
        "source_cycle_index",
        "horizons",
        "saturation_utilization_threshold",
    }.intersection(adapter_options)
    if forbidden:
        raise ValueError(f"adapter_options may not override {sorted(forbidden)}.")
    adapter_options["saturation_utilization_threshold"] = saturation_threshold

    first_cycle, period_basis = select_certified_rho_cycle(
        source,
        cycle_index=0,
        cycle_period=adapter_options.get("cycle_period"),
    )
    cycle_count = first_cycle.cycle_count
    anchor_policy = "stride"
    if isinstance(anchors, str):
        if anchors != "all":
            raise ValueError("The only string anchor policy is 'all'.")
        anchors = tuple(range(cycle_count))
        anchor_policy = "all"
    elif anchors is None:
        anchors = tuple(range(0, cycle_count, anchor_stride))
    else:
        anchors = tuple(_nonnegative_integer(value, name="anchor") for value in anchors)
        anchor_policy = "explicit"
        if len(set(anchors)) != len(anchors):
            raise ValueError("anchors must be unique.")
    if not anchors:
        raise ValueError("At least one anchor is required.")

    records: list[dict[str, Any]] = []
    for anchor in anchors:
        if anchor >= cycle_count:
            for horizon in horizons:
                records.append(
                    {
                        "anchor_cycle_index": anchor,
                        "horizon_cycles": horizon,
                        "target_cycle_index": anchor + horizon,
                        "status": "censored_anchor_unavailable",
                        "censoring": {
                            "declared_cycle_count": cycle_count,
                            "last_available_cycle_index": cycle_count - 1,
                        },
                        "metrics": None,
                    }
                )
            continue
        available = tuple(horizon for horizon in horizons if anchor + horizon < cycle_count)
        censored = tuple(horizon for horizon in horizons if anchor + horizon >= cycle_count)
        for horizon in censored:
            records.append(
                {
                    "anchor_cycle_index": anchor,
                    "horizon_cycles": horizon,
                    "target_cycle_index": anchor + horizon,
                    "status": "censored_target_unavailable",
                    "censoring": {
                        "declared_cycle_count": cycle_count,
                        "last_available_cycle_index": cycle_count - 1,
                    },
                    "metrics": None,
                }
            )
        if not available:
            continue
        try:
            adapter_report = build_rho_endurance_rollout_report(
                source,
                reduced_profile,
                horizons=available,
                source_cycle_index=anchor,
                **adapter_options,
            )
        except Exception as error:
            for horizon in available:
                records.append(
                    {
                        "anchor_cycle_index": anchor,
                        "horizon_cycles": horizon,
                        "target_cycle_index": anchor + horizon,
                        "status": "anchor_evaluation_error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "metrics": None,
                    }
                )
            continue

        if adapter_report["status"] != "complete":
            gate = {
                "adapter_status": adapter_report["status"],
                "rejection_reasons": adapter_report.get("rejection_reasons", []),
                "adapted_policy_fidelity": adapter_report.get("adapted_policy_fidelity"),
                "source_transcription": adapter_report.get("audits", {}).get(
                    "source_transcription"
                ),
                "source_midpoint_interpolation": adapter_report.get("audits", {}).get(
                    "source_midpoint_interpolation"
                ),
            }
            for horizon in available:
                records.append(
                    {
                        "anchor_cycle_index": anchor,
                        "horizon_cycles": horizon,
                        "target_cycle_index": anchor + horizon,
                        "status": "invalid_anchor_policy",
                        "gate": gate,
                        "metrics": None,
                    }
                )
            continue

        muscle_names = tuple(adapter_report["muscle_names"])
        model_parameters = adapter_report["model_parameters"]
        forecasts = {row["horizon_cycles"]: row for row in adapter_report["horizons"]}
        try:
            anchor_cycle, _ = select_certified_rho_cycle(
                source, cycle_index=anchor, cycle_period=first_cycle.period
            )
            anchor_observation = _actual_cycle_observation(
                anchor_cycle,
                muscle_names=muscle_names,
                model_parameters=model_parameters,
                saturation_threshold=saturation_threshold,
                pulse_width_tolerance_s=observed_pulse_width_tolerance_s,
            )
        except Exception as error:
            for horizon in available:
                records.append(
                    {
                        "anchor_cycle_index": anchor,
                        "horizon_cycles": horizon,
                        "target_cycle_index": anchor + horizon,
                        "status": "anchor_observation_error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "metrics": None,
                    }
                )
            continue
        if not anchor_observation["valid"]:
            for horizon in available:
                records.append(
                    {
                        "anchor_cycle_index": anchor,
                        "horizon_cycles": horizon,
                        "target_cycle_index": anchor + horizon,
                        "status": "invalid_anchor_observation",
                        "anchor_audit": anchor_observation,
                        "metrics": None,
                    }
                )
            continue
        for horizon in available:
            target_index = anchor + horizon
            try:
                target_cycle, _ = select_certified_rho_cycle(
                    source, cycle_index=target_index, cycle_period=first_cycle.period
                )
                observation = _actual_cycle_observation(
                    target_cycle,
                    muscle_names=muscle_names,
                    model_parameters=model_parameters,
                    saturation_threshold=saturation_threshold,
                    pulse_width_tolerance_s=observed_pulse_width_tolerance_s,
                )
                if not observation["valid"]:
                    records.append(
                        {
                            "anchor_cycle_index": anchor,
                            "horizon_cycles": horizon,
                            "target_cycle_index": target_index,
                            "status": "invalid_target_observation",
                            "target_audit": observation,
                            "metrics": None,
                        }
                    )
                    continue
                forecast = forecasts[horizon]
                forecast_audit = _forecast_outcome_audit(
                    forecast,
                    model_parameters=model_parameters,
                    expected_utilization_samples=(
                        target_cycle.stimulations_per_cycle * len(muscle_names)
                    ),
                )
                if not forecast_audit["valid"]:
                    records.append(
                        {
                            "anchor_cycle_index": anchor,
                            "horizon_cycles": horizon,
                            "target_cycle_index": target_index,
                            "status": "invalid_forecast_outcome",
                            "forecast_audit": forecast_audit,
                            "metrics": None,
                        }
                    )
                    continue
                metrics = _comparison_metrics(
                    forecast,
                    observation,
                    muscle_names=muscle_names,
                    model_parameters=model_parameters,
                    interval_count=target_cycle.stimulations_per_cycle,
                    anchor_observation=anchor_observation,
                )
            except Exception as error:
                records.append(
                    {
                        "anchor_cycle_index": anchor,
                        "horizon_cycles": horizon,
                        "target_cycle_index": target_index,
                        "status": "target_evaluation_error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "metrics": None,
                    }
                )
                continue
            records.append(
                {
                    "anchor_cycle_index": anchor,
                    "horizon_cycles": horizon,
                    "target_cycle_index": target_index,
                    "status": "valid",
                    "forecast_feasible": bool(forecast["feasible"]),
                    "observed_pulse_width_audits": {
                        "anchor": anchor_observation["pulse_width_audit"],
                        "target": observation["pulse_width_audit"],
                    },
                    "metrics": metrics,
                }
            )

    records.sort(key=lambda row: (row["anchor_cycle_index"], row["horizon_cycles"]))
    counts = {
        status: sum(record["status"] == status for record in records)
        for status in sorted({record["status"] for record in records})
    }
    valid_count = counts.get("valid", 0)
    invalid_count = sum(
        count
        for status, count in counts.items()
        if status
        not in {"valid", "censored_anchor_unavailable", "censored_target_unavailable"}
    )
    if valid_count == 0:
        processing_status = "fully_censored" if invalid_count == 0 else "no_valid_comparisons"
    elif invalid_count:
        processing_status = "partial_invalid"
    elif counts.get("censored_target_unavailable", 0) or counts.get(
        "censored_anchor_unavailable", 0
    ):
        processing_status = "complete_with_censoring"
    else:
        processing_status = "complete"
    aggregate = _aggregate_by_horizon(records, horizons, criteria)
    horizon_scientific_statuses = {
        row["scientific_validation_status"] for row in aggregate
    }
    if "failed" in horizon_scientific_statuses:
        scientific_validation_status = "failed"
    elif horizon_scientific_statuses == {"passed"}:
        scientific_validation_status = "passed"
    elif "passed" in horizon_scientific_statuses:
        scientific_validation_status = "incomplete"
    else:
        scientific_validation_status = "not_evaluable"
    return {
        "schema": REPORT_SCHEMA,
        "processing_status": processing_status,
        "scientific_validation_status": scientific_validation_status,
        "data_policy": "rho_only_no_full_horizon_data",
        "source": _file_stamp(source),
        "reduced_profile": _file_stamp(reduced_profile),
        "selection": {
            "certified": first_cycle.certified,
            "certification_basis": first_cycle.certification_basis,
            "declared_cycle_count": cycle_count,
            "producer_requested_cycle_count": int(
                first_cycle.metadata.get("producer_requested_cycles", cycle_count)
            ),
            "source_truncated_before_requested_count": max(
                0,
                int(first_cycle.metadata.get("producer_requested_cycles", cycle_count))
                - cycle_count,
            ),
            "cycle_period_s": first_cycle.period,
            "cycle_period_basis": period_basis,
            "anchors": list(anchors),
            "anchor_policy": anchor_policy,
            "anchor_stride": anchor_stride if anchor_policy == "stride" else None,
            "horizons": list(horizons),
            "indexing": "zero_based; forecast origin=end(k); target=end(k+H)",
        },
        "configuration": {
            "saturation_threshold": saturation_threshold,
            "observed_pulse_width_bound_tolerance_s": (
                observed_pulse_width_tolerance_s
            ),
            "observed_pulse_width_clipping_applied": False,
            "scientific_criteria": criteria,
            "scientific_criteria_policy": (
                "configured_before metric evaluation; every criterion must pass per horizon"
            ),
            "adapter_options": _json_configuration(adapter_options),
        },
        "record_counts": counts,
        "aggregate_metrics_by_horizon": aggregate,
        "records": records,
    }


CSV_FIELDS = (
    "anchor_cycle_index",
    "horizon_cycles",
    "target_cycle_index",
    "status",
    "forecast_feasible",
    "A_rmse",
    "A_rest_normalized_rmse",
    "Tau1_rmse",
    "Tau1_rest_normalized_rmse",
    "Km_rmse",
    "Km_rest_normalized_rmse",
    "minimum_capacity_ratio_error",
    "maximum_utilization_error",
    "saturated_sample_fraction_error",
    "critical_muscle_match",
    "critical_phase_circular_error_intervals",
    "policy_drift_normalized_pw_rmse",
    "policy_drift_normalized_pw_p95_absolute",
    "policy_drift_critical_muscle_changed",
    "policy_drift_critical_phase_change_intervals",
)


def _csv_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    for field in ("anchor_cycle_index", "horizon_cycles", "target_cycle_index", "status"):
        row[field] = record.get(field, "")
    if record["status"] != "valid":
        return row
    row["forecast_feasible"] = record["forecast_feasible"]
    metrics = record["metrics"]
    for state in ("A", "Tau1", "Km"):
        row[f"{state}_rmse"] = metrics["slow_state_errors"][state]["rmse"]
        row[f"{state}_rest_normalized_rmse"] = metrics["slow_state_errors"][state][
            "rest_normalized_rmse"
        ]
    row["minimum_capacity_ratio_error"] = metrics["capacity_ratio"]["minimum_error"]
    recruitment = metrics["recruitment"]
    row["maximum_utilization_error"] = recruitment["maximum_utilization_error"]
    row["saturated_sample_fraction_error"] = recruitment["saturated_sample_fraction_error"]
    row["critical_muscle_match"] = recruitment["critical_muscle_match"]
    row["critical_phase_circular_error_intervals"] = recruitment[
        "critical_phase_circular_error_intervals"
    ]
    drift = metrics["observed_policy_drift"]
    row["policy_drift_normalized_pw_rmse"] = drift["rmse"]
    row["policy_drift_normalized_pw_p95_absolute"] = drift["p95_absolute"]
    row["policy_drift_critical_muscle_changed"] = drift["critical_muscle_changed"]
    row["policy_drift_critical_phase_change_intervals"] = drift[
        "critical_phase_circular_change_intervals"
    ]
    return row


def write_rho_retrospective_validation(
    json_output: str | Path,
    csv_output: str | Path,
    report: dict[str, Any],
) -> tuple[Path, Path]:
    """Atomically write standards-compliant JSON and a flat audit table."""

    json_output = Path(json_output)
    csv_output = Path(csv_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_temporary = json_output.with_suffix(json_output.suffix + ".tmp")
    csv_temporary = csv_output.with_suffix(csv_output.suffix + ".tmp")
    json_temporary.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    with csv_temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_row(record) for record in report["records"])
    json_temporary.replace(json_output)
    csv_temporary.replace(csv_output)
    return json_output, csv_output
