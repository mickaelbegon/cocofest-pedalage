#!/usr/bin/env python3
"""Build a read-only comparison matrix from cycling benchmark result.json files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


CONDITIONS = (
    "30hz-unconstrained",
    "50hz-unconstrained",
    "30hz-dpw100us",
    "50hz-dpw100us",
)
SOLVERS = ("ipopt-ma57-radau5", "acados-irk")
MUSCLES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")

# New producers must write ``pulse_width_max_step_us``. The aliases are
# accepted only to read development/legacy artefacts; the selected source is
# exported so that use of a non-canonical field remains visible.
MAX_STEP_US_KEYS = (
    "pulse_width_max_step_us",
    "pulse_width_slew_limit_us",
    "pulse_width_change_limit_us",
    "maximum_pulse_width_change_us",
    "max_pulse_width_change_us",
    "pulse_width_delta_max_us",
    "inter_cycle_pulse_width_slew_limit_us",
)
LEGACY_MAX_STEP_S_KEYS = tuple(
    key.removesuffix("_us") + "_s" for key in MAX_STEP_US_KEYS[1:]
)


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _percentile(values: Iterable[Any], percentile: float) -> float | None:
    finite = sorted(value for raw in values if (value := _finite(raw)) is not None)
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def _mean(values: Iterable[Any]) -> float | None:
    finite = [value for raw in values if (value := _finite(raw)) is not None]
    return statistics.fmean(finite) if finite else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _frequency_hz(configuration: dict) -> tuple[float | None, str | None]:
    interval = _finite(configuration.get("calcium_stimulation_interval_s"))
    if interval is not None and interval > 0.0:
        return 1.0 / interval, "calcium_stimulation_interval_s"
    value = _finite(configuration.get("stimulation_frequency_hz"))
    if value is not None:
        return value, "stimulation_frequency_hz"
    value = _finite(configuration.get("stimulations_per_cycle"))
    if value is not None:
        return value, "stimulations_per_cycle_assuming_one_second_cycle"
    return None, None


def _max_step_us(configuration: dict) -> tuple[float | None, str | None]:
    found: list[tuple[float, str]] = []
    for key in MAX_STEP_US_KEYS:
        value = _finite(configuration.get(key))
        if value is not None:
            found.append((value, key))
    for key in LEGACY_MAX_STEP_S_KEYS:
        value = _finite(configuration.get(key))
        if value is not None:
            found.append((value * 1e6, key))
    nested = configuration.get("pulse_width_slew")
    if isinstance(nested, dict):
        value = _finite(nested.get("maximum_change_us"))
        if value is not None:
            found.append((value, "pulse_width_slew.maximum_change_us"))
        value = _finite(nested.get("maximum_change_s"))
        if value is not None:
            found.append((value * 1e6, "pulse_width_slew.maximum_change_s"))
    if not found:
        return None, None
    reference = found[0][0]
    if any(not math.isclose(value, reference, rel_tol=1e-9, abs_tol=1e-9) for value, _ in found[1:]):
        raise ValueError(f"Conflicting pulse-width maximum steps: {found}")
    return reference, ",".join(key for _, key in found)


def _condition(frequency_hz: float | None, slew_us: float | None) -> str:
    frequency = next(
        (
            candidate
            for candidate in (30, 50)
            if frequency_hz is not None
            and math.isclose(frequency_hz, candidate, abs_tol=1e-6)
        ),
        None,
    )
    if frequency is None:
        return "unclassified"
    if slew_us is None:
        return f"{frequency}hz-unconstrained"
    if math.isclose(slew_us, 100.0, rel_tol=1e-9, abs_tol=1e-6):
        return f"{frequency}hz-dpw100us"
    return "unclassified"


def _solver_label(solver: str, configuration: dict) -> str:
    if solver == "ipopt":
        linear = str(configuration.get("ipopt_linear_solver") or "unknown").lower()
        degree = configuration.get("collocation_degree")
        method = str(configuration.get("collocation_method") or "").lower()
        if linear == "ma57" and degree == 5 and method == "radau":
            return "ipopt-ma57-radau5"
        return f"ipopt-{linear}-{method}{degree}"
    if solver == "acados":
        integrator = str(configuration.get("acados_integrator_type") or "").upper()
        return "acados-irk" if integrator == "IRK" else "acados-other"
    return solver or "unknown"


def _hot_target_rows(windows: list[dict]) -> list[dict]:
    return [
        window
        for index, window in enumerate(windows)
        if index > 0
        and window.get("certifier", "target_solver") == "target_solver"
        and bool(window.get("solver_converged", window.get("status") == 0))
    ]


def _common_prefix(windows: list[dict], threshold: float) -> int:
    prefix = 0
    for window in windows:
        infeasibility = _finite(
            (window.get("feasibility") or {}).get(
                "effective_primal_infeasibility"
            )
        )
        converged = bool(window.get("solver_converged", window.get("status") == 0))
        if infeasibility is None or infeasibility > threshold or not converged:
            break
        prefix += 1
    return prefix


def _recovery_total(result: dict, field: str) -> float:
    return float(
        sum(
            _finite((result.get(key) or {}).get(field)) or 0.0
            for key in (
                "acados_ipopt_recovery",
                "ipopt_madnlp_recovery",
                "nlp_ipopt_recovery",
            )
        )
    )


def _muscle_values(result: dict) -> dict[str, dict]:
    return {
        str(item.get("muscle")): item
        for item in result.get("muscle_fatigue") or []
        if isinstance(item, dict) and item.get("muscle")
    }


def _observed_max_step_us(result: dict) -> tuple[float | None, dict[str, float]]:
    # Do not reuse ``pulse_width_cycle_variation`` here: that diagnostic
    # compares the same phase across successive RHO cycles, whereas the hard
    # constraint concerns consecutive stimulation intervals (plus seams).
    audit = result.get("pulse_width_slew_audit") or {}
    explicit = _finite(audit.get("maximum_adjacent_change_us"))
    by_muscle = {}
    for muscle, item in (audit.get("muscles") or {}).items():
        if not isinstance(item, dict):
            continue
        value = _finite(item.get("maximum_adjacent_change_us"))
        if value is not None:
            by_muscle[str(muscle)] = value
    candidates = [
        value
        for value in (explicit, *by_muscle.values())
        if value is not None
    ]
    return (max(candidates) if candidates else None), by_muscle


def _provenance_shas(value: Any, prefix: str = "") -> dict[str, Any]:
    found = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if "sha" in str(key).lower() and child is not None:
                found[path] = child
            found.update(_provenance_shas(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(_provenance_shas(child, f"{prefix}[{index}]"))
    return found


def summarize_result(path: Path, payload: dict, result: dict, threshold: float) -> dict:
    solver = str(result.get("solver") or "").lower()
    configuration = (payload.get("configurations") or {}).get(solver) or {}
    windows = [
        window
        for window in result.get("windows") or []
        if isinstance(window, dict)
    ]
    hot = _hot_target_rows(windows)
    frequency, frequency_source = _frequency_hz(configuration)
    max_step, max_step_source = _max_step_us(configuration)
    observed_max_step, observed_max_step_by_muscle = _observed_max_step_us(result)
    muscle = _muscle_values(result)
    infeasibilities = [
        _finite((window.get("feasibility") or {}).get("effective_primal_infeasibility"))
        for window in windows
    ]
    stationarity = [
        _finite((window.get("feasibility") or {}).get("acados_stationarity_residual"))
        for window in windows
    ]
    runtime = payload.get("runtime") or {}
    runtime_provenance = runtime.get("provenance") or {}
    protocol = {
        "formulation": configuration.get("formulation") or "dynamic",
        "mechanical_formulation": configuration.get("mechanical_formulation"),
        "constant_crank_torque_nm": _finite(
            configuration.get("constant_crank_torque")
        ),
        "torque_application": configuration.get("torque_application"),
        "requested_cycles": int(
            result.get("requested_cycles") or configuration.get("n_windows") or 0
        ),
        "cycles_per_window": configuration.get("cycles_per_window"),
        "objective": configuration.get("objective"),
        "objective_shape": configuration.get("objective_shape"),
        "model_formulation": configuration.get("model_formulation"),
        "muscle_names": configuration.get("muscle_names"),
        "force_length": configuration.get("activate_force_length_relationship"),
        "force_velocity": configuration.get("activate_force_velocity_relationship"),
        "passive_force": configuration.get("activate_passive_force_relationship"),
        "enforce_start_constraints": configuration.get(
            "enforce_start_constraints"
        ),
        "full_contact_constraints_all_nodes": configuration.get(
            "full_contact_constraints_all_nodes"
        ),
        "full_contact_constraints_terminal": configuration.get(
            "full_contact_constraints_terminal"
        ),
        "wheel_qdot_bound_margin": configuration.get("wheel_qdot_bound_margin"),
        "terminal_wheel_qdot_bound_margin": configuration.get(
            "terminal_wheel_qdot_bound_margin"
        ),
        "pulse_width_active_set": configuration.get("pulse_width_active_set"),
        "state_scaling": configuration.get("state_scaling"),
        "ding_sum_stim_truncation": configuration.get(
            "ding_sum_stim_truncation"
        ),
        "terminal_reserve_weight": configuration.get("terminal_reserve_weight"),
        "terminal_reserve_temperature": configuration.get(
            "terminal_reserve_temperature"
        ),
        "common_initial_solution": configuration.get("common_initial_solution"),
        "n_threads": configuration.get("n_threads"),
    }
    row = {
        "condition": _condition(frequency, max_step),
        "solver_cell": _solver_label(solver, configuration),
        "solver": solver,
        "linear_solver": (
            configuration.get("ipopt_linear_solver")
            if solver == "ipopt"
            else configuration.get("acados_qp_solver")
        ),
        "transcription_profile": configuration.get("transcription_profile"),
        "collocation_method": configuration.get("collocation_method"),
        "collocation_degree": configuration.get("collocation_degree"),
        "acados_integrator_type": configuration.get("acados_integrator_type"),
        "acados_sim_stages": configuration.get("acados_sim_stages"),
        "acados_sim_steps": configuration.get("acados_sim_steps"),
        "frequency_hz": frequency,
        "frequency_source": frequency_source,
        "stimulations_per_cycle": configuration.get("stimulations_per_cycle"),
        "control_decisions_per_cycle": configuration.get("control_decisions_per_cycle"),
        "pulse_width_max_step_us": max_step,
        "pulse_width_max_step_source": max_step_source,
        "observed_max_pulse_width_step_us": observed_max_step,
        "observed_max_pulse_width_step_by_muscle_us": observed_max_step_by_muscle,
        "pulse_width_max_step_violation_us": (
            max(0.0, observed_max_step - max_step)
            if observed_max_step is not None and max_step is not None
            else None
        ),
        **protocol,
        "protocol_sha256": _stable_hash(protocol),
        "configured_tolerance": _finite(configuration.get("nlp_tolerance")),
        "primal_feasibility_threshold": _finite(
            configuration.get("primal_feasibility_threshold")
        ),
        "success": bool(result.get("success")),
        "solver_success": bool(result.get("solver_success")),
        "physical_success": bool(result.get("physical_success")),
        "scientific_success": bool(
            result.get(
                "isokinetic_scientific_success", result.get("physical_success")
            )
        ),
        "validated_cycles": int(result.get("validated_cycles") or 0),
        "attempted_windows": int(result.get("attempted_windows") or len(windows)),
        "common_validated_prefix": _common_prefix(windows, threshold),
        "primal_feasible_windows": int(result.get("primal_feasible_windows") or 0),
        "status_zero_windows": int(result.get("status_zero_windows") or 0),
        "maximum_effective_infeasibility": max(
            (value for value in infeasibilities if value is not None), default=None
        ),
        "maximum_stationarity_residual": max(
            (value for value in stationarity if value is not None), default=None
        ),
        "objective_reported": _finite(result.get("objective")),
        "window_objective_sum": _finite(result.get("window_objective_sum")),
        "validated_prefix_window_objective_sum": _finite(
            result.get("validated_prefix_window_objective_sum")
        ),
        "executed_fatigue_objective": _finite(
            result.get("executed_fatigue_objective")
        ),
        "fatigue_auc_cycles": _finite(result.get("fatigue_auc_cycles")),
        "minimum_capacity_ratio": _finite(result.get("min_A_capacity_ratio")),
        "maximum_mean_normalized_fatigue": _finite(
            result.get("max_mean_normalized_fatigue")
        ),
        "hot_target_sample_count": len(hot),
        "hot_solver_mean_s": _mean(row.get("solver_time_s") for row in hot),
        "hot_solver_median_s": _percentile(
            (row.get("solver_time_s") for row in hot), 50
        ),
        "hot_solver_p90_s": _percentile(
            (row.get("solver_time_s") for row in hot), 90
        ),
        "hot_wall_mean_s": _mean(row.get("wall_time_s") for row in hot),
        "hot_wall_median_s": _percentile((row.get("wall_time_s") for row in hot), 50),
        "hot_wall_p90_s": _percentile((row.get("wall_time_s") for row in hot), 90),
        "hot_iterations_mean": _mean(row.get("iterations") for row in hot),
        "hot_iterations_median": _percentile((row.get("iterations") for row in hot), 50),
        "hot_iterations_p90": _percentile((row.get("iterations") for row in hot), 90),
        "validated_solver_time_s": _finite(result.get("validated_solver_time_s")),
        "validated_wall_time_s": _finite(result.get("validated_wall_time_s")),
        "end_to_end_wall_time_s": _finite(result.get("end_to_end_wall_time_s")),
        "stop_label": (result.get("stop") or {}).get("label"),
        "first_failed_rho": result.get("first_failed_rho"),
        "last_native_status": (
            windows[-1].get("native_status", windows[-1].get("status"))
            if windows
            else result.get("native_solver_status")
        ),
        "recovery_attempts": int(_recovery_total(result, "attempt_count")),
        "recovery_seed_only_attempts": int(
            _recovery_total(result, "seed_only_attempt_count")
        ),
        "recovery_certifying_attempts": int(
            _recovery_total(result, "certifying_attempt_count")
        ),
        "recovery_wall_time_s": _recovery_total(result, "recovery_wall_time_s"),
        "fallback_advanced_count": int(
            _recovery_total(result, "fallback_advanced_count")
        ),
        "solver_attempt_count": int(
            (result.get("solver_attempt_accounting") or {}).get("attempt_count") or 0
        ),
        "certifier_counts": {
            str(certifier): sum(
                1
                for window in windows
                if window.get("certifier", "target_solver") == certifier
            )
            for certifier in sorted(
                {
                    window.get("certifier", "target_solver")
                    for window in windows
                }
            )
        },
        "feasibility_restoration_attempts": int(
            (result.get("feasibility_restoration") or {}).get("attempt_count") or 0
        ),
        "maximum_consecutive_failures": result.get("maximum_consecutive_failures"),
        "mechanical_audit_passes": (
            result.get("mechanical_equivalence_audit") or {}
        ).get("passes_tolerance"),
        "error": result.get("error"),
        "profile_hash": configuration.get("profile_hash"),
        "profile_version": configuration.get("profile_version"),
        "configuration_sha256": _stable_hash(configuration),
        "configuration": configuration,
        "common_initial_solution": configuration.get("common_initial_solution"),
        "runtime_python": runtime.get("python"),
        "runtime_casadi": runtime.get("casadi"),
        "runtime_bioptim": runtime.get("bioptim"),
        "runtime_platform": runtime.get("platform"),
        "logical_cpu_count": runtime.get("logical_cpu_count"),
        "thread_environment": runtime.get("thread_environment"),
        "runtime_provenance": runtime_provenance,
        "input_provenance": result.get("input_provenance"),
        "provenance_shas": _provenance_shas(
            {
                "runtime": runtime,
                "input": result.get("input_provenance"),
                "ipopt": result.get("ipopt_runtime"),
            }
        ),
        "source": str(path),
        "source_result_sha256": _sha256(path),
    }
    for name in MUSCLES:
        item = muscle.get(name) or {}
        row[f"fatigue_auc_{name}"] = _finite(
            item.get("cumulative_normalized_fatigue_cycles")
        )
        row[f"final_capacity_ratio_{name}"] = _finite(
            item.get("final_capacity_ratio")
        )
    return row


def load_rows(paths: list[Path], threshold: float) -> list[dict]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("results") or []
        if not results:
            raise ValueError(f"{path} contains no solver result")
        rows.extend(
            summarize_result(path, payload, result, threshold)
            for result in results
            if isinstance(result, dict)
        )
    return sorted(
        rows,
        key=lambda row: (
            CONDITIONS.index(row["condition"])
            if row["condition"] in CONDITIONS
            else len(CONDITIONS),
            SOLVERS.index(row["solver_cell"])
            if row["solver_cell"] in SOLVERS
            else len(SOLVERS),
            row["source"],
        ),
    )


def matrix_audit(rows: list[dict]) -> dict:
    cells: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row["condition"] in CONDITIONS and row["solver_cell"] in SOLVERS:
            cells.setdefault((row["condition"], row["solver_cell"]), []).append(row)
    expected = [
        (condition, solver) for condition in CONDITIONS for solver in SOLVERS
    ]
    missing = [
        f"{condition}/{solver}"
        for condition, solver in expected
        if not cells.get((condition, solver))
    ]
    duplicates = [
        f"{condition}/{solver}"
        for (condition, solver), values in cells.items()
        if len(values) > 1
    ]
    protocol_hashes = sorted(
        {
            row["protocol_sha256"]
            for values in cells.values()
            for row in values
        }
    )
    protocol_fields = (
        "formulation",
        "mechanical_formulation",
        "constant_crank_torque_nm",
        "torque_application",
        "requested_cycles",
        "cycles_per_window",
        "objective",
        "objective_shape",
        "model_formulation",
        "muscle_names",
        "force_length",
        "force_velocity",
        "passive_force",
        "enforce_start_constraints",
        "full_contact_constraints_all_nodes",
        "full_contact_constraints_terminal",
        "wheel_qdot_bound_margin",
        "terminal_wheel_qdot_bound_margin",
        "pulse_width_active_set",
        "state_scaling",
        "ding_sum_stim_truncation",
        "terminal_reserve_weight",
        "terminal_reserve_temperature",
        "common_initial_solution",
        "n_threads",
    )
    protocol_differences = {}
    for field in protocol_fields:
        values = {}
        for row in rows:
            encoded = json.dumps(row.get(field), sort_keys=True)
            values.setdefault(encoded, []).append(row["source"])
        if len(values) > 1:
            protocol_differences[field] = [
                {"value": json.loads(encoded), "sources": sources}
                for encoded, sources in sorted(values.items())
            ]
    return {
        "complete": not missing and not duplicates and len(rows) == len(expected),
        "missing_cells": missing,
        "duplicate_cells": duplicates,
        "unclassified_sources": [
            row["source"]
            for row in rows
            if row["condition"] not in CONDITIONS
            or row["solver_cell"] not in SOLVERS
        ],
        "common_protocol": len(protocol_hashes) == 1,
        "protocol_hashes": protocol_hashes,
        "protocol_differences": protocol_differences,
    }


CSV_FIELDS = (
    "condition",
    "solver_cell",
    "frequency_hz",
    "pulse_width_max_step_us",
    "observed_max_pulse_width_step_us",
    "pulse_width_max_step_violation_us",
    "constant_crank_torque_nm",
    "n_threads",
    "mechanical_formulation",
    "formulation",
    "requested_cycles",
    "validated_cycles",
    "common_validated_prefix",
    "success",
    "maximum_effective_infeasibility",
    "maximum_stationarity_residual",
    "executed_fatigue_objective",
    "fatigue_auc_cycles",
    "minimum_capacity_ratio",
    *(f"fatigue_auc_{name}" for name in MUSCLES),
    *(f"final_capacity_ratio_{name}" for name in MUSCLES),
    "hot_target_sample_count",
    "hot_solver_mean_s",
    "hot_solver_median_s",
    "hot_solver_p90_s",
    "hot_wall_mean_s",
    "hot_wall_median_s",
    "hot_wall_p90_s",
    "hot_iterations_mean",
    "hot_iterations_median",
    "hot_iterations_p90",
    "recovery_attempts",
    "recovery_seed_only_attempts",
    "recovery_certifying_attempts",
    "recovery_wall_time_s",
    "fallback_advanced_count",
    "solver_attempt_count",
    "stop_label",
    "profile_hash",
    "configuration_sha256",
    "protocol_sha256",
    "source_result_sha256",
    "source",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    value = _finite(value)
    return "—" if value is None else f"{value:.{digits}g}"


def render_markdown(rows: list[dict], audit: dict, threshold: float) -> str:
    by_cell = {
        (row["condition"], row["solver_cell"]): row
        for row in rows
        if row["condition"] in CONDITIONS and row["solver_cell"] in SOLVERS
    }
    lines = [
        "# Matrice fréquence × contrainte ΔPW × solveur",
        "",
        f"Seuil de faisabilité commun : `{threshold:.1e}`. Les statistiques "
        "chaudes excluent la première fenêtre et toute fenêtre certifiée par "
        "un solveur de récupération.",
        "",
    ]
    if not audit["complete"]:
        lines.extend(
            [
                "**Matrice incomplète — cellules absentes : "
                f"{', '.join(audit['missing_cells']) or 'aucune'}; doublons : "
                f"{', '.join(audit['duplicate_cells']) or 'aucun'}.**",
                "",
            ]
        )
    if not audit["common_protocol"]:
        differing = ", ".join(audit["protocol_differences"])
        lines.extend(
            [
                "**Les champs de protocole fixes ne sont pas identiques entre "
                "toutes les lignes; ne pas interpréter les écarts comme un "
                "effet isolé de fréquence/ΔPW. Champs divergents : "
                f"{differing}.**",
                "",
            ]
        )
    lines.extend(
        [
            "| Condition | Solveur | Cycles valides | ΔPW max/limite (us) | Infais. max | Coût "
            "fatigue exécuté | AUC fatigue | Capacité min. | Solveur chaud "
            "moy./méd./P90 (s) | Itérations moy./méd./P90 | "
            "Recoveries/fallbacks |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        for solver in SOLVERS:
            row = by_cell.get((condition, solver))
            if row is None:
                lines.append(
                    f"| `{condition}` | `{solver}` | **absent** | — | — | — | — "
                    "| — | — | — | — |"
                )
                continue
            requested = row["requested_cycles"]
            lines.append(
                f"| `{condition}` | `{solver}` | {row['common_validated_prefix']}/{requested} | "
                f"{_fmt(row['observed_max_pulse_width_step_us'])}/"
                f"{_fmt(row['pulse_width_max_step_us'])} | "
                f"{_fmt(row['maximum_effective_infeasibility'])} | "
                f"{_fmt(row['executed_fatigue_objective'])} | "
                f"{_fmt(row['fatigue_auc_cycles'])} | {_fmt(row['minimum_capacity_ratio'])} | "
                f"{_fmt(row['hot_solver_mean_s'])}/"
                f"{_fmt(row['hot_solver_median_s'])}/"
                f"{_fmt(row['hot_solver_p90_s'])} | "
                f"{_fmt(row['hot_iterations_mean'])}/"
                f"{_fmt(row['hot_iterations_median'])}/"
                f"{_fmt(row['hot_iterations_p90'])} | "
                f"{row['recovery_attempts']}/{row['fallback_advanced_count']} |"
            )
    lines.extend(
        [
            "",
            "Chaque ligne JSON conserve la configuration, les SHA disponibles, "
            "l’environnement de threads et la provenance complète. La CSV est "
            "volontairement aplatie pour l’analyse statistique.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_files", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--common-feasibility-threshold", type=float, default=1e-5)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    if args.common_feasibility_threshold <= 0:
        parser.error("--common-feasibility-threshold must be positive")
    rows = load_rows(args.json_files, args.common_feasibility_threshold)
    audit = matrix_audit(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "frequency-slew-matrix.csv", rows)
    (args.output_dir / "frequency-slew-matrix.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "common_feasibility_threshold": args.common_feasibility_threshold,
                "audit": audit,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "frequency-slew-matrix.md").write_text(
        render_markdown(rows, audit, args.common_feasibility_threshold),
        encoding="utf-8",
    )
    if args.require_complete and not audit["complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
