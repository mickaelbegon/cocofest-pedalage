"""Observed Radau convergence at fixed periodic-node Ding states and PW.

Refines only the integration mesh, never stimulation frequency or PW decisions.
Each subelement sees the decayed history of its original stimulation interval,
including its terminal Radau stage. Reuses the existing fixed-PW model oracle.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import casadi as ca
import numpy as np
from scipy.optimize import root

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.validate_ding_fixed_pulse_width import (
    FULL_DING_STATE_NAMES, ding_rhs, dop853_step, load_common_seed,
)


def radau_step(state, *, degree, pulse_width, duration_s, calcium_amplitude, parameters):
    """Equivalent Butcher equations avoid dividing cancellation errors by h."""
    points = np.asarray(ca.collocation_points(degree, "radau"))
    coefficients, _, _ = ca.collocation_coeff(points)
    butcher = np.linalg.inv(np.asarray(coefficients)[1:, :].T)
    scales = np.maximum(np.abs(state), np.array([1., 100., 1000., .1, .1]))

    def residual(z):
        stages = z.reshape(degree, 5) * scales
        derivatives = np.array([
            ding_rhs(point * duration_s, stage, pulse_width=pulse_width,
                     calcium_amplitude=calcium_amplitude, parameters=parameters)
            for point, stage in zip(points, stages)
        ])
        return ((stages-state-duration_s * butcher @ derivatives) / scales).ravel()

    answer = root(residual, np.tile(state / scales, degree), options={"xtol": 1e-11})
    norm = float(np.max(np.abs(residual(answer.x))))
    if not math.isfinite(norm) or norm > 1e-10:
        raise RuntimeError(f"Radau-{degree} failed with normalized stage residual {norm}: {answer.message}")
    # Stiff accuracy: final stage is the step endpoint.
    return answer.x.reshape(degree, 5)[-1] * scales, norm


def propagate(case, *, degree=None, subdivisions=1, rtol=1e-13, atol=1e-15):
    current = case.initial_state.copy()
    trajectory = [current.copy()]
    maximum_residual = 0.0
    for width in case.pulse_widths:
        if degree is None:
            current, _ = dop853_step(
                current, pulse_width=float(width), duration_s=case.duration_s,
                calcium_amplitude=case.calcium_amplitude, parameters=case.parameters,
                rtol=rtol, atol=atol,
            )
        else:
            step = case.duration_s / subdivisions
            for index in range(subdivisions):
                # No new pulse at a subelement boundary; rebase exponential only.
                amplitude = case.calcium_amplitude * math.exp(-index * step / case.parameters.tauc)
                current, residual = radau_step(
                    current, degree=degree, pulse_width=float(width), duration_s=step,
                    calcium_amplitude=amplitude, parameters=case.parameters,
                )
                maximum_residual = max(maximum_residual, residual)
        trajectory.append(current.copy())
    return np.asarray(trajectory), maximum_residual


def validate(case, subdivisions):
    reference, _ = propagate(case)
    loose, _ = propagate(case, rtol=1e-12, atol=1e-14)
    force_scale = max(float(np.max(np.abs(reference[:, 1]))), 1e-12)
    rows = []
    for degree in (3, 5):
        previous = None
        for count in subdivisions:
            trajectory, residual = propagate(case, degree=degree, subdivisions=count)
            error = np.abs(trajectory - reference)
            terminal_force_relative = float(error[-1, 1] / max(abs(reference[-1, 1]), 1e-12))
            order = None if previous is None or terminal_force_relative == 0 else math.log2(previous / terminal_force_relative)
            rows.append({
                "degree": degree, "subdivisions_per_stimulation": count,
                "terminal_force_relative_error": terminal_force_relative,
                "observed_terminal_force_order": order,
                "max_force_error_relative_to_peak": float(np.max(error[:, 1]) / force_scale),
                "max_absolute_endpoint_error": dict(zip(FULL_DING_STATE_NAMES, np.max(error, axis=0).tolist())),
                "maximum_normalized_stage_residual": residual,
            })
            previous = terminal_force_relative
    return {
        "initial_state": case.initial_state.tolist(), "stimulations": len(case.pulse_widths),
        "interval_s": case.duration_s, "reference_force_peak_N": force_scale,
        "reference_convergence_max_absolute": dict(zip(FULL_DING_STATE_NAMES, np.max(np.abs(reference-loose), axis=0).tolist())),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-seed", type=Path, required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = parser.parse_args()
    if args.subdivisions != [2**i for i in range(len(args.subdivisions))]:
        parser.error("subdivisions must be 1, 2, 4, ... to report log2 orders")
    cases, metadata, provenance = load_common_seed(args.common_seed, expected_frequency_hz=None, model_config=args.model_config)
    report = {
        "schema": "ding-radau-observed-order-v1",
        "scope": "fixed_PW_isolated_Ding_neutral_mechanics; not_an_OCP_or_physiological_certificate",
        "source_seed": str(args.common_seed.resolve()), "parameter_provenance": provenance,
        "frequency_hz": metadata["stimulations_per_cycle"],
        "dop853": {"rtol": 1e-13, "atol": 1e-15, "comparison_rtol": 1e-12, "comparison_atol": 1e-14},
        "terminal_stage_convention": "history from interval start decays through terminal stage; new pulse only in next interval",
        "muscles": {},
    }
    for case in cases:
        report["muscles"][case.name] = validate(case, args.subdivisions)
        print(f"Completed {case.name}", flush=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
