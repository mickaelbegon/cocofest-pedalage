"""Validate the isolated periodic-node Ding dynamics at fixed pulse widths.

This deliberately contains no mechanics, OCP, or optimization.  Starting from
the same complete five-state Ding state, it applies the pulse-width sequence
stored in a 50 Hz common-initial-solution archive and compares the direct
Radau transcription maps to a high-accuracy DOP853 integration.  It therefore
measures the numerical error of the Ding equations themselves, including the
calcium state, rather than differences between two optimized PW policies.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

# Scripts are invoked as ``python scripts/name.py`` in the campaign commands.
# Put the repository root on the import path before importing project modules.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.adaptive_moment_rollout import (
    DingPulseWidthParameters,
    FULL_DING_STATE_NAMES,
)
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters


DEFAULT_R0_KM_RELATIONSHIP = 1.04


@dataclass(frozen=True)
class FixedPulseWidthDingCase:
    """One muscle and its imposed stimulation sequence."""

    name: str
    initial_state: np.ndarray
    pulse_widths: np.ndarray
    duration_s: float
    calcium_amplitude: float
    parameters: DingPulseWidthParameters

    def __post_init__(self) -> None:
        state = np.asarray(self.initial_state, dtype=float)
        widths = np.asarray(self.pulse_widths, dtype=float)
        if state.shape != (5,) or not np.all(np.isfinite(state)):
            raise ValueError(f"{self.name}: initial state must be five finite values.")
        if widths.ndim != 1 or widths.size < 1 or not np.all(np.isfinite(widths)):
            raise ValueError(f"{self.name}: pulse widths must be a non-empty finite vector.")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError(f"{self.name}: duration_s must be finite and positive.")
        if not math.isfinite(self.calcium_amplitude) or self.calcium_amplitude < 0.0:
            raise ValueError(f"{self.name}: calcium_amplitude must be finite and non-negative.")


def periodic_post_stimulation_amplitude(
    *,
    duration_s: float,
    tauc: float,
    km_rest: float,
    retained_stimulations: int,
    r0_km_relationship: float = DEFAULT_R0_KM_RELATIONSHIP,
) -> float:
    """Reproduce ``periodic_node.post_stimulation_amplitude`` numerically.

    The historical code computes ``r0 = Km_rest + 1.04``.  The relationship is
    an explicit argument because common-seed v3 archives do not store it.
    """

    if duration_s <= 0.0 or tauc <= 0.0 or retained_stimulations < 1:
        raise ValueError("duration_s, tauc and retained_stimulations must be positive.")
    if not all(math.isfinite(value) for value in (km_rest, r0_km_relationship)):
        raise ValueError("calcium-history parameters must be finite.")
    decay = math.exp(-duration_s / tauc)
    if retained_stimulations == 1:
        return 1.0
    stimulation_increment = 1.0 + (km_rest + r0_km_relationship - 1.0) * decay
    recent_sum = sum(decay**age for age in range(retained_stimulations - 1))
    return decay ** (retained_stimulations - 1) + stimulation_increment * recent_sum


def ding_rhs(
    local_time_s: float,
    state: np.ndarray,
    *,
    pulse_width: float,
    calcium_amplitude: float,
    parameters: DingPulseWidthParameters,
) -> np.ndarray:
    """Five-state periodic-node Ding RHS, with neutral mechanical gain exactly one."""

    cn, force, capacity, tau1, km = np.asarray(state, dtype=float)
    if cn < -1e-10 or force < -1e-10 or capacity <= 0.0 or tau1 <= 0.0 or km <= 0.0:
        raise ValueError("Ding state left its positive numerical domain.")
    calcium_history = calcium_amplitude * math.exp(-local_time_s / parameters.tauc)
    cn_dot = (calcium_history - cn) / parameters.tauc
    activation = cn / (km + cn)
    relaxation = tau1 + parameters.tau2 * activation
    recruitment = capacity * (-math.expm1(-(pulse_width - parameters.pd0) / parameters.pdt))
    force_dot = recruitment * activation - force / relaxation
    fatigue = parameters.fatigue
    return np.array(
        [
            cn_dot,
            force_dot,
            -(capacity - fatigue.a_rest) / fatigue.tau_fat + fatigue.alpha_a * force,
            -(tau1 - fatigue.tau1_rest) / fatigue.tau_fat + fatigue.alpha_tau1 * force,
            -(km - fatigue.km_rest) / fatigue.tau_fat + fatigue.alpha_km * force,
        ],
        dtype=float,
    )


def dop853_step(
    state: np.ndarray,
    *,
    pulse_width: float,
    duration_s: float,
    calcium_amplitude: float,
    parameters: DingPulseWidthParameters,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, int]:
    solution = solve_ivp(
        lambda time, values: ding_rhs(
            time, values, pulse_width=pulse_width, calcium_amplitude=calcium_amplitude, parameters=parameters
        ),
        (0.0, duration_s),
        np.asarray(state, dtype=float),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"DOP853 failed: {solution.message}")
    return solution.y[:, -1], int(solution.nfev)


def radau_collocation_step(
    state: np.ndarray,
    *,
    degree: int,
    pulse_width: float,
    duration_s: float,
    calcium_amplitude: float,
    parameters: DingPulseWidthParameters,
    nonlinear_tolerance: float = 1e-11,
) -> tuple[np.ndarray, float]:
    """Solve the direct Radau collocation equations for one fixed-PW interval."""

    if degree < 1:
        raise ValueError("degree must be positive.")
    points = np.asarray(ca.collocation_points(degree, "radau"), dtype=float)
    coefficients, continuity, _ = ca.collocation_coeff(points)
    coefficients = np.asarray(coefficients, dtype=float)
    continuity = np.asarray(continuity, dtype=float).reshape(-1)
    initial = np.asarray(state, dtype=float)
    # A constant initial state is deliberately method-neutral: this is not an
    # OCP warm-start and cannot hide a transfer/interpolation error.
    guess = np.tile(initial, (degree, 1)).reshape(-1)

    def residual(flat_stages: np.ndarray) -> np.ndarray:
        stages = np.asarray(flat_stages, dtype=float).reshape(degree, 5)
        all_states = np.vstack((initial, stages))
        defects = []
        for stage_index, point in enumerate(points):
            derivative = coefficients[:, stage_index] @ all_states / duration_s
            rhs = ding_rhs(
                point * duration_s,
                stages[stage_index],
                pulse_width=pulse_width,
                calcium_amplitude=calcium_amplitude,
                parameters=parameters,
            )
            defects.append(derivative - rhs)
        return np.concatenate(defects)

    solution = root(residual, guess, method="hybr", options={"xtol": nonlinear_tolerance})
    residual_norm = float(np.max(np.abs(residual(solution.x))))
    if not solution.success or not math.isfinite(residual_norm) or residual_norm > 1e-8:
        raise RuntimeError(
            f"Radau-{degree} stage solve failed (success={solution.success}, residual={residual_norm:.3e}): "
            f"{solution.message}"
        )
    stages = solution.x.reshape(degree, 5)
    return continuity @ np.vstack((initial, stages)), residual_norm


def _scales(reference: np.ndarray, initial_state: np.ndarray) -> np.ndarray:
    """Statewise scale retained in output so normalized errors are auditable."""

    return np.maximum(1.0, np.maximum(np.max(np.abs(reference), axis=0), np.abs(initial_state)))


def validate_case(
    case: FixedPulseWidthDingCase,
    *,
    degrees: Sequence[int] = (3, 5),
    dop853_rtol: float = 1e-11,
    dop853_atol: float = 1e-13,
) -> dict:
    """Return accumulated endpoint errors for each requested Radau degree."""

    if not degrees or len(set(degrees)) != len(degrees):
        raise ValueError("degrees must be a non-empty sequence without duplicates.")
    reference = [np.asarray(case.initial_state, dtype=float)]
    nfev = 0
    current = reference[0]
    for pulse_width in case.pulse_widths:
        current, evaluations = dop853_step(
            current,
            pulse_width=float(pulse_width),
            duration_s=case.duration_s,
            calcium_amplitude=case.calcium_amplitude,
            parameters=case.parameters,
            rtol=dop853_rtol,
            atol=dop853_atol,
        )
        reference.append(current)
        nfev += evaluations
    reference = np.asarray(reference)
    report = {
        "initial_state": dict(zip(FULL_DING_STATE_NAMES, case.initial_state.tolist())),
        "interval_count": int(case.pulse_widths.size),
        "duration_per_interval_s": case.duration_s,
        "calcium_post_stimulation_amplitude": case.calcium_amplitude,
        "dop853": {"rtol": dop853_rtol, "atol": dop853_atol, "function_evaluations": nfev},
        "degrees": {},
    }
    scales = _scales(reference, case.initial_state)
    for degree in degrees:
        approximate = [np.asarray(case.initial_state, dtype=float)]
        residuals = []
        current = approximate[0]
        for pulse_width in case.pulse_widths:
            current, residual = radau_collocation_step(
                current,
                degree=int(degree),
                pulse_width=float(pulse_width),
                duration_s=case.duration_s,
                calcium_amplitude=case.calcium_amplitude,
                parameters=case.parameters,
            )
            approximate.append(current)
            residuals.append(residual)
        error = np.abs(np.asarray(approximate) - reference)
        report["degrees"][str(degree)] = {
            "maximum_collocation_residual": max(residuals),
            "state_scales": dict(zip(FULL_DING_STATE_NAMES, scales.tolist())),
            "maximum_absolute_endpoint_error_by_state": dict(zip(FULL_DING_STATE_NAMES, np.max(error, axis=0).tolist())),
            "terminal_absolute_error_by_state": dict(zip(FULL_DING_STATE_NAMES, error[-1].tolist())),
            "maximum_scaled_endpoint_error_by_state": dict(zip(FULL_DING_STATE_NAMES, (np.max(error, axis=0) / scales).tolist())),
            "maximum_absolute_endpoint_error": float(np.max(error)),
            "reference_terminal_state": dict(zip(FULL_DING_STATE_NAMES, reference[-1].tolist())),
            "radau_terminal_state": dict(zip(FULL_DING_STATE_NAMES, approximate[-1].tolist())),
        }
    return report


def _parameters(values: Mapping[str, float], *, pulse_width_max: float) -> DingPulseWidthParameters:
    required = ("a_scale", "tau1_rest", "km_rest", "alpha_a", "alpha_tau1", "alpha_km", "tau_fat", "tauc", "tau2", "pd0", "pdt")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"configured_muscle_parameters is missing {', '.join(missing)}.")
    return DingPulseWidthParameters(
        fatigue=DingFatigueParameters(
            a_rest=float(values["a_scale"]), tau1_rest=float(values["tau1_rest"]),
            km_rest=float(values["km_rest"]), alpha_a=float(values["alpha_a"]),
            alpha_tau1=float(values["alpha_tau1"]), alpha_km=float(values["alpha_km"]),
            tau_fat=float(values["tau_fat"]),
        ),
        tauc=float(values["tauc"]), tau2=float(values["tau2"]), pd0=float(values["pd0"]),
        pdt=float(values["pdt"]), pulse_width_max=float(pulse_width_max),
    )


def _configured_parameters_from_file(path: str | Path) -> tuple[dict, dict]:
    """Resolve a declared model config for a legacy archive without parameters.

    Old common seeds predate the effective-parameter metadata.  Reconstructing
    their parameters is acceptable only when the caller explicitly names the
    versioned model config that produced the run; defaults are never assumed.
    """

    from cocofest.optimization.configured_cycling_model import resolve_model_config

    config_path = Path(path).resolve(strict=True)
    config = resolve_model_config(json.loads(config_path.read_text()))
    return config["muscles"], {
        "source": "explicit_model_config",
        "path": str(config_path),
        "case_id": config["case_id"],
        "fingerprint": config["muscle_parameter_fingerprint"],
    }


def load_common_seed(
    path: str | Path,
    *,
    expected_frequency_hz: int | None = 50,
    r0_km_relationship: float = DEFAULT_R0_KM_RELATIONSHIP,
    model_config: str | Path | None = None,
) -> tuple[list[FixedPulseWidthDingCase], dict, dict]:
    """Extract comparable isolated-Ding cases from a v3 common seed archive."""

    path = Path(path).resolve(strict=True)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata__json"].item()))
        count = int(metadata["stimulations_per_cycle"])
        if expected_frequency_hz is not None and count != expected_frequency_hz:
            raise ValueError(f"Expected {expected_frequency_hz} stimulations per one-second cycle, got {count}.")
        if metadata.get("calcium_forcing_formulation") != "exact_exponential_periodic_node":
            raise ValueError("The archive does not use exact_exponential_periodic_node calcium forcing.")
        if metadata.get("model_formulation") != "periodic_node":
            raise ValueError("The archive does not use periodic_node Ding states.")
        values_by_muscle = metadata.get("configured_muscle_parameters", {})
        parameter_provenance = {"source": "seed_metadata"}
        if not values_by_muscle:
            if model_config is None:
                raise ValueError(
                    "The archive has no configured_muscle_parameters metadata. "
                    "Pass --model-config with the explicit configuration that "
                    "produced this legacy seed."
                )
            values_by_muscle, parameter_provenance = _configured_parameters_from_file(
                model_config
            )
        duration_s = 1.0 / count
        pulse_width_max = float(metadata["pulse_width_maximum_s"])
        truncation = int(metadata.get("ding_sum_stim_truncation", 6))
        cases = []
        for name, values in values_by_muscle.items():
            state_keys = [f"states__{state}_{name}" for state in FULL_DING_STATE_NAMES]
            control_key = f"controls__last_pulse_width_{name}"
            missing = [key for key in (*state_keys, control_key) if key not in archive.files]
            if missing:
                raise ValueError(f"{name}: archive is missing {', '.join(missing)}.")
            initial = np.array([float(np.asarray(archive[key]).reshape(-1)[0]) for key in state_keys])
            widths = np.asarray(archive[control_key], dtype=float).reshape(-1)
            if widths.size != count:
                raise ValueError(f"{name}: expected {count} PW values, got {widths.size}.")
            parameters = _parameters(values, pulse_width_max=pulse_width_max)
            amplitude = periodic_post_stimulation_amplitude(
                duration_s=duration_s, tauc=parameters.tauc, km_rest=parameters.fatigue.km_rest,
                retained_stimulations=truncation, r0_km_relationship=r0_km_relationship,
            )
            cases.append(FixedPulseWidthDingCase(name, initial, widths, duration_s, amplitude, parameters))
    return cases, metadata, parameter_provenance


def validate_seed(
    path: str | Path,
    *,
    expected_frequency_hz: int | None = 50,
    r0_km_relationship: float = DEFAULT_R0_KM_RELATIONSHIP,
    degrees: Sequence[int] = (3, 5),
    model_config: str | Path | None = None,
) -> dict:
    cases, metadata, parameter_provenance = load_common_seed(
        path, expected_frequency_hz=expected_frequency_hz,
        r0_km_relationship=r0_km_relationship,
        model_config=model_config,
    )
    return {
        "schema": "ding-fixed-pulse-width-validation-v1",
        "scope": {
            "same_initial_state_and_fixed_pw_sequence": True,
            "mechanical_gain": 1.0,
            "includes": list(FULL_DING_STATE_NAMES),
            "excludes": ["mechanics", "OCP", "PW_optimization", "warm-start_transfer"],
            "r0_km_relationship": r0_km_relationship,
        },
        "source": {
            "common_seed": str(Path(path).resolve()),
            "stimulations_per_cycle": metadata["stimulations_per_cycle"],
            "parameter_provenance": parameter_provenance,
        },
        "muscles": {case.name: validate_case(case, degrees=degrees) for case in cases},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-seed", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-frequency-hz", type=int, default=50)
    parser.add_argument(
        "--model-config",
        type=Path,
        help="Required only for a legacy seed without configured_muscle_parameters metadata.",
    )
    parser.add_argument("--r0-km-relationship", type=float, default=DEFAULT_R0_KM_RELATIONSHIP)
    parser.add_argument("--degrees", type=int, nargs="+", default=(3, 5))
    args = parser.parse_args()
    report = validate_seed(
        args.common_seed, expected_frequency_hz=args.expected_frequency_hz,
        r0_km_relationship=args.r0_km_relationship, degrees=tuple(args.degrees),
        model_config=args.model_config,
    )
    output = args.output_json.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    for name, muscle in report["muscles"].items():
        for degree, result in muscle["degrees"].items():
            print(
                f"{name} Radau-{degree}: max Ding endpoint error="
                f"{result['maximum_absolute_endpoint_error']:.3e}, "
                f"collocation residual={result['maximum_collocation_residual']:.3e}",
                flush=True,
            )
    print(f"Fixed-PW Ding validation: {output}")


if __name__ == "__main__":
    main()
