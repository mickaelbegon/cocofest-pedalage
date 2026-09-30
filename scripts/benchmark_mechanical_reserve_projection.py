#!/usr/bin/env python3
"""Reproducible, solver-free microbenchmark for a proposed RHO-Réserve projection.

This is a *local affine margin surrogate*, not a certified torque capacity or
an OCP experiment.  A candidate's periodic force profile is frozen.  Ding's
three slow states are propagated exactly for each constant-force interval,
then signed local mechanical sensitivities map their normalized displacement
from a fixed reference state to one margin per muscle and phase.  A smooth
minimum reports the most vulnerable future margin.  Its unnormalized form is
conservative: it cannot report a positive margin when a raw margin is negative.
No force, PW, or
recruitment policy is optimized in this benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp


SCHEMA = "cocofest-mechanical-reserve-projection-benchmark-v1"
REFERENCE_RTOL = 1e-11
REFERENCE_ATOL = 1e-13
TEMPERATURE = 0.001
NEGATIVE_CONTROL_TEMPERATURE = 0.01
# Registered before benchmarking.  Timing is a screening criterion, not an
# OCP integration certificate; noisy shared hosts may need an isolated rerun.
GATES = {
    "max_normalized_state_error": 2e-9,
    "max_margin_error": 2e-9,
    "max_score_error": 2e-9,
    "max_gradient_absolute_error": 2e-7,
    "max_gradient_relative_error": 2e-5,
    "gradient_relative_floor": 1e-6,
    "h100_min_speedup_vs_dop853": 5.0,
    "h100_max_analytic_call_p90_ms": 20.0,
    "allow_false_safe_smooth_margin": False,
}


def synthetic_case(name: str) -> dict[str, np.ndarray | float | str]:
    """Four known Ding-like muscles, with phase-varying signed margin gains."""
    if name not in {"constant", "piecewise"}:
        raise ValueError("case must be constant or piecewise")
    phases = 8
    angle = 2 * np.pi * np.arange(phases) / phases
    rest = np.array([
        [1200.0, .060601, .137], [1180.0, .064, .142],
        [1240.0, .058, .131], [1160.0, .067, .146],
    ])
    alpha = np.array([
        [-1.4, 2.1e-5, 1.9e-5], [-1.2, 1.8e-5, 2.0e-5],
        [-1.5, 2.2e-5, 1.7e-5], [-1.1, 1.9e-5, 2.1e-5],
    ])
    initial = rest * np.array([[.94, 1.08, 1.06], [.91, 1.04, 1.08],
                               [.96, 1.06, 1.03], [.92, 1.10, 1.07]])
    baseline_force = np.array([1.8, 2.1, 1.5, 2.4])
    if name == "constant":
        forces = np.broadcast_to(baseline_force, (phases, 4)).copy()
    else:
        modulation = 1 + .45 * np.sin(angle[:, None] + np.array([0., .8, 1.7, 2.6]))
        forces = baseline_force[None, :] * modulation
    # Gains operate on dimensionless slow-state deviations.  The negative A
    # coefficients in muscles 1 and 3 emulate opposing mechanical leverage;
    # Tau1/Km signs are mixed deliberately, since their effect is not universal.
    gain = np.empty((phases, 4, 3))
    gain[:, :, 0] = np.array([.24, -.13, .19, -.08])[None, :] * (1 + .15 * np.cos(angle[:, None]))
    gain[:, :, 1] = np.array([-.04, .025, -.03, .02])[None, :]
    gain[:, :, 2] = np.array([-.06, .03, .04, -.025])[None, :]
    reference_margin = .055 + .012 * np.cos(angle[:, None] + np.arange(4)[None, :])
    return {
        "name": name,
        "rest": rest,
        "alpha": alpha,
        "initial": initial,
        "forces": forces,
        "durations": np.full(phases, 1 / phases),
        "tau_fat": 445.5,
        "gain": gain,
        "reference_margin": reference_margin,
    }


def _validate(case: dict, horizon: int) -> None:
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    rest, alpha, initial = (np.asarray(case[key], dtype=float) for key in ("rest", "alpha", "initial"))
    forces = np.asarray(case["forces"], dtype=float)
    durations = np.asarray(case["durations"], dtype=float)
    gain = np.asarray(case["gain"], dtype=float)
    margin = np.asarray(case["reference_margin"], dtype=float)
    if rest.ndim != 2 or rest.shape[1] != 3 or initial.shape != rest.shape or alpha.shape != rest.shape:
        raise ValueError("rest, alpha, initial must share shape (muscles, 3)")
    phases, muscles = forces.shape
    if muscles != rest.shape[0] or durations.shape != (phases,) or gain.shape != (phases, muscles, 3) or margin.shape != (phases, muscles):
        raise ValueError("force, duration, gain, margin dimensions disagree")
    if not all(np.all(np.isfinite(np.asarray(value))) for value in case.values() if isinstance(value, np.ndarray)):
        raise ValueError("case arrays must be finite")
    if np.any(rest <= 0) or np.any(forces < 0) or np.any(durations <= 0) or float(case["tau_fat"]) <= 0:
        raise ValueError("invalid physiological domain")
    if np.any(alpha[:, 0] > 0) or np.any(alpha[:, 1:] < 0):
        raise ValueError("Ding fatigue coefficient signs are invalid")


def project_analytic(case: dict, horizon: int) -> np.ndarray:
    """States at all future phase ends, shape (cycles, phases, muscles, 3)."""
    _validate(case, horizon)
    rest, alpha = case["rest"], case["alpha"]
    forces, durations = case["forces"], case["durations"]
    tau = float(case["tau_fat"])
    state = np.array(case["initial"], copy=True)
    states = np.empty((horizon, len(durations), *rest.shape))
    for cycle in range(horizon):
        for phase, duration in enumerate(durations):
            decay = math.exp(-float(duration) / tau)
            input_gain = -tau * math.expm1(-float(duration) / tau)
            state = rest + decay * (state - rest) + input_gain * forces[phase, :, None] * alpha
            states[cycle, phase] = state
    return states


def project_reference(case: dict, horizon: int) -> np.ndarray:
    """Independent DOP853 solution, restarted at each force discontinuity."""
    _validate(case, horizon)
    rest, alpha = case["rest"], case["alpha"]
    state = np.array(case["initial"], copy=True)
    states = np.empty((horizon, len(case["durations"]), *rest.shape))
    for cycle in range(horizon):
        for phase, duration in enumerate(case["durations"]):
            force = case["forces"][phase, :, None]

            def rhs(_time, flat_state):
                values = flat_state.reshape(rest.shape)
                return (-(values - rest) / case["tau_fat"] + alpha * force).ravel()

            solution = solve_ivp(rhs, (0., float(duration)), state.ravel(), method="DOP853",
                                  rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)
            if not solution.success:
                raise RuntimeError(solution.message)
            state = solution.y[:, -1].reshape(rest.shape)
            states[cycle, phase] = state
    return states


def margin_score(case: dict, states: np.ndarray, *, gradient: bool = False) -> tuple[np.ndarray, float, np.ndarray | None]:
    """Local affine margins and conservative smooth minimum over future samples."""
    horizon, phases = states.shape[:2]
    gain = np.asarray(case["gain"])
    rest = np.asarray(case["rest"])
    margins = case["reference_margin"][None, :, :] + np.sum(
        gain[None, :, :, :] * ((states - rest[None, None, :, :]) / rest[None, None, :, :]), axis=-1)
    flat = margins.ravel()
    anchor = float(np.min(flat))
    weights = np.exp(-(flat - anchor) / TEMPERATURE)
    total = float(np.sum(weights))
    score = anchor - TEMPERATURE * math.log(total)
    if not gradient:
        return margins, score, None
    elapsed = np.cumsum(np.tile(case["durations"], horizon)).reshape(horizon, phases)
    decay = np.exp(-elapsed / float(case["tau_fat"]))
    probabilities = (weights / total).reshape(margins.shape)
    derivative = np.sum(
        probabilities[:, :, :, None] * gain[None, :, :, :] / rest[None, None, :, :] * decay[:, :, None, None],
        axis=(0, 1))
    return margins, score, derivative


def normalized_diagnostic_minimum(margins: np.ndarray) -> float:
    """Optimistic T=0.01 control retained to expose false-safe classifications."""
    flat = margins.ravel()
    anchor = float(np.min(flat))
    terms = np.exp(-(flat - anchor) / NEGATIVE_CONTROL_TEMPERATURE)
    return anchor - NEGATIVE_CONTROL_TEMPERATURE * math.log(float(np.mean(terms)))


def gradient_finite_difference(case: dict, horizon: int) -> np.ndarray:
    """Independent central difference of the complete analytic score map."""
    gradient = np.empty_like(case["initial"])
    for muscle in range(gradient.shape[0]):
        for component in range(gradient.shape[1]):
            step = 1e-5 * float(case["rest"][muscle, component])
            plus = {**case, "initial": np.array(case["initial"], copy=True)}
            minus = {**case, "initial": np.array(case["initial"], copy=True)}
            plus["initial"][muscle, component] += step
            minus["initial"][muscle, component] -= step
            score_plus = margin_score(plus, project_analytic(plus, horizon))[1]
            score_minus = margin_score(minus, project_analytic(minus, horizon))[1]
            gradient[muscle, component] = (score_plus - score_minus) / (2 * step)
    return gradient


def _timed(function, repeats: int) -> tuple[np.ndarray, list[float]]:
    durations = []
    result = None
    for _ in range(repeats):
        started = perf_counter()
        result = function()
        durations.append((perf_counter() - started) * 1e3)
    assert result is not None
    return result, durations


def project_and_score(case: dict, horizon: int, projector, *, gradient: bool = False):
    states = projector(case, horizon)
    margins, score, derivative = margin_score(case, states, gradient=gradient)
    return states, margins, score, derivative


def benchmark_case(case: dict, horizon: int, *, analytic_repeats: int, reference_repeats: int) -> dict:
    (analytic_states, analytic_margins, analytic_score, _), analytic_times = _timed(
        lambda: project_and_score(case, horizon, project_analytic), analytic_repeats)
    (reference_states, reference_margins, reference_score, _), reference_times = _timed(
        lambda: project_and_score(case, horizon, project_reference), reference_repeats)
    (_, _, _, gradient), gradient_times = _timed(
        lambda: project_and_score(case, horizon, project_analytic, gradient=True), analytic_repeats)
    finite_difference = gradient_finite_difference(case, horizon)
    assert gradient is not None
    normalized_state_error = float(np.max(np.abs((analytic_states - reference_states) / case["rest"][None, None])))
    margin_error = float(np.max(np.abs(analytic_margins - reference_margins)))
    gradient_abs_error = float(np.max(np.abs(gradient - finite_difference)))
    gradient_rel_error = float(np.max(np.abs(gradient - finite_difference) /
                                      np.maximum(np.abs(finite_difference), GATES["gradient_relative_floor"])))
    analytic_ms = float(np.median(analytic_times))
    reference_ms = float(np.median(reference_times))
    accuracy_pass = (
        normalized_state_error <= GATES["max_normalized_state_error"]
        and margin_error <= GATES["max_margin_error"]
        and abs(analytic_score - reference_score) <= GATES["max_score_error"]
        and gradient_abs_error <= GATES["max_gradient_absolute_error"]
        and gradient_rel_error <= GATES["max_gradient_relative_error"]
    )
    speed_pass = (horizon != 100 or (
        reference_ms / analytic_ms >= GATES["h100_min_speedup_vs_dop853"]
        and float(np.percentile(analytic_times, 90)) <= GATES["h100_max_analytic_call_p90_ms"]
    ))
    hard_minimum = float(np.min(analytic_margins))
    normalized_smooth = normalized_diagnostic_minimum(analytic_margins)
    false_safe = hard_minimum < 0 <= analytic_score
    normalized_false_safe = hard_minimum < 0 <= normalized_smooth
    return {
        "case": case["name"], "horizon_cycles": horizon,
        "phase_count_per_cycle": len(case["durations"]), "muscle_count": len(case["rest"]),
        "minimum_margin": hard_minimum,
        "smooth_minimum_margin": analytic_score,
        "normalized_smooth_minimum_diagnostic": normalized_smooth,
        "softmin_conservatism": hard_minimum - analytic_score,
        "normalized_softmin_optimism": normalized_smooth - hard_minimum,
        "softmin_gap_bound": TEMPERATURE * math.log(analytic_margins.size),
        "normalized_diagnostic_gap_bound": NEGATIVE_CONTROL_TEMPERATURE * math.log(analytic_margins.size),
        "false_safe_smooth_margin": bool(false_safe),
        "negative_control_normalized_false_safe": bool(normalized_false_safe),
        "errors": {
            "normalized_state_max": normalized_state_error,
            "margin_max": margin_error,
            "score_absolute": abs(analytic_score - reference_score),
            "gradient_initial_state_absolute_max": gradient_abs_error,
            "gradient_initial_state_relative_max": gradient_rel_error,
        },
        "cost_ms_per_projection": {
            "analytic_median": analytic_ms, "analytic_p90": float(np.percentile(analytic_times, 90)),
            "dop853_median": reference_ms, "dop853_p90": float(np.percentile(reference_times, 90)),
            "speedup_dop853_over_analytic": reference_ms / analytic_ms,
            "analytic_with_gradient_median": float(np.median(gradient_times)),
        },
        "accuracy_gate_pass": bool(accuracy_pass), "screening_speed_gate_pass": bool(speed_pass),
        "risk_sign_gate_pass": bool(not false_safe),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON file; report is always printed")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 20, 100])
    parser.add_argument("--analytic-repeats", type=int, default=31)
    parser.add_argument("--reference-repeats", type=int, default=1)
    args = parser.parse_args(argv)
    if any(value < 1 for value in (*args.horizons, args.analytic_repeats, args.reference_repeats)):
        parser.error("horizons and repeat counts must be positive")
    if len(set(args.horizons)) != len(args.horizons):
        parser.error("horizons must be unique")
    rows = [benchmark_case(synthetic_case(case_name), horizon,
                           analytic_repeats=args.analytic_repeats,
                           reference_repeats=args.reference_repeats)
            for case_name in ("constant", "piecewise") for horizon in args.horizons]
    report = {
        "schema": SCHEMA,
        "contract": {
            "force_policy": "fixed synthetic profile repeated each cycle; no future recruitment optimization",
            "slow_states": ["A", "Tau1", "Km"],
            "fatigue_equation": "dz/dt = -(z-rest)/tau_fat + alpha*force",
            "margin_equation": "m_ref[phase,muscle] + sum_j gain[phase,muscle,j]*(z_j-rest_j)/rest_j",
            "margin_interpretation": "local affine sensitivity surrogate with signed gains; not a certified attainable moment envelope",
            "aggregation": "conservative unnormalized smooth minimum over all future phase-end muscle margins; normalized optimistic variant is diagnostic only",
            "gradient": "score derivative with respect to the 12 initial slow states, compared with central differences",
        },
        "reference": {"integrator": "DOP853", "rtol": REFERENCE_RTOL, "atol": REFERENCE_ATOL,
                      "restart_at_each_force_discontinuity": True},
        "temperature": TEMPERATURE,
        "negative_control_normalized_temperature": NEGATIVE_CONTROL_TEMPERATURE,
        "registered_gates": GATES,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": __import__("scipy").__version__},
        "settings": {"horizons": args.horizons, "analytic_repeats": args.analytic_repeats,
                     "reference_repeats": args.reference_repeats},
        "cases": rows,
        "all_accuracy_gates_pass": all(row["accuracy_gate_pass"] for row in rows),
        "all_screening_speed_gates_pass": all(row["screening_speed_gate_pass"] for row in rows),
        "all_risk_sign_gates_pass": all(row["risk_sign_gate_pass"] for row in rows),
        "rho_integration_authorized_by_this_benchmark": False,
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if (report["all_accuracy_gates_pass"] and report["all_screening_speed_gates_pass"]
                 and report["all_risk_sign_gates_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
