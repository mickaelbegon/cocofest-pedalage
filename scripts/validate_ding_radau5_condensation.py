"""Isolated Ding Radau-5 stage-condensation audit; no production OCP changes.

Run with the benchmark environment.  ``condensed`` preserves the original
five-state Radau equations exactly; ``analytic`` changes their discretization
and therefore has a separate DOP853 accuracy check.  This script measures
single-muscle maps, not IPOPT/MA57 performance or mechanical feasibility.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root


STATE_NAMES = ("Cn", "F", "A", "Tau1", "Km")
SCALE = np.array([1.0, 100.0, 4920.0, 0.060601, 0.137])


@dataclass(frozen=True)
class Parameters:
    """Declared Ding-2007 defaults; no implicit model/seed parameter recovery."""

    tauc: float = 0.011
    tau2: float = 0.001
    pd0: float = 0.000131405
    pdt: float = 0.000194138
    tau_fat: float = 127.0
    rest: tuple = (4920.0, 0.060601, 0.137)
    alpha: tuple = (-0.4, 2.1e-5, 1.9e-5)


def radau5_tableau():
    """Integral tableau equivalent to CasADi's five-stage Radau collocation."""
    points = np.array(ca.collocation_points(5, "radau"))
    coefficients, _, _ = ca.collocation_coeff(points.tolist())
    # C[1:, :].T Y + C[0, :] x0 = h f(Y).
    a = np.linalg.solve(np.array(coefficients)[1:, :].T, np.eye(5))
    return points, a


def _force_rhs(states, pw, p):
    cn, force, capacity, tau1, km = np.asarray(states).T
    activation = cn / (km + cn)
    recruitment = capacity * (-np.expm1(-(pw - p.pd0) / p.pdt))
    return recruitment * activation - force / (tau1 + p.tau2 * activation)


def full_rhs(t, x, pw, amplitude, p):
    return np.r_[
        (amplitude * np.exp(-t / p.tauc) - x[0]) / p.tauc,
        _force_rhs(x, pw, p),
        -(x[2:] - p.rest) / p.tau_fat + np.array(p.alpha) * x[1],
    ]


def radau5_step(x0, pw, h, amplitude, *, kind="condensed", p=Parameters()):
    """Return endpoint, physical stages and scaled nonlinear residual.

    Every call accepts arbitrary positive fatigue initial states.  It does not
    project them onto a rest-generated one-dimensional fatigue manifold.
    """
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (5,) or not np.all(np.isfinite(x0)):
        raise ValueError("x0 must contain five finite states")
    if np.any(x0 < 0) or np.any(x0[2:] <= 0):
        raise ValueError("initial states must be nonnegative; fatigue states strictly positive")
    if not np.isfinite([pw, h, amplitude]).all() or h <= 0 or amplitude < 0 or pw < p.pd0:
        raise ValueError("invalid interval, pulse width or calcium amplitude")
    c, a = radau5_tableau()
    t = c * h
    rest, alpha = np.array(p.rest), np.array(p.alpha)

    if kind == "full":
        guess = np.tile(x0 / SCALE, (5, 1)).ravel()

        def reconstruct(z):
            return z.reshape(5, 5) * SCALE

        def residual(z):
            stages = reconstruct(z)
            rhs = np.array([full_rhs(ti, xi, pw, amplitude, p) for ti, xi in zip(t, stages)])
            return ((stages - x0 - h * a @ rhs) / SCALE).ravel()

    elif kind == "condensed":
        # Eliminate four linear stage systems without approximating the Radau
        # equations.  The fatigue LU matrix is shared by A, Tau1 and Km.
        cn = np.linalg.solve(
            np.eye(5) + h * a / p.tauc,
            np.full(5, x0[0]) + h * a @ (amplitude * np.exp(-t / p.tauc) / p.tauc),
        )
        fatigue_inverse = np.linalg.solve(np.eye(5) + h * a / p.tau_fat, np.eye(5))
        fatigue_initial = fatigue_inverse @ np.tile(x0[2:] - rest, (5, 1))
        fatigue_force = fatigue_inverse @ (h * a)
        guess = np.full(5, x0[1] / SCALE[1])

        def reconstruct(z):
            force = z * SCALE[1]
            slow = rest + fatigue_initial + (fatigue_force @ force)[:, None] * alpha
            return np.column_stack((cn, force, slow))

        def residual(z):
            states = reconstruct(z)
            return (states[:, 1] - x0[1] - h * a @ _force_rhs(states, pw, p)) / SCALE[1]

    elif kind == "analytic":
        # J' = exp(t/tau_fat) F, J(0)=0.  A/Tau1/Km are reconstructed
        # analytically given J; F and J alone receive Radau-5 collocation.
        cn = np.exp(-t / p.tauc) * (x0[0] + amplitude * t / p.tauc)
        reduced_scale = np.array([SCALE[1], SCALE[1] * h])
        guess = np.column_stack((np.full(5, x0[1]), t * x0[1])) / reduced_scale
        guess = guess.ravel()

        def reconstruct(z):
            force, integral = (z.reshape(5, 2) * reduced_scale).T
            slow = rest + np.exp(-t[:, None] / p.tau_fat) * (
                x0[2:] - rest + integral[:, None] * alpha
            )
            return np.column_stack((cn, force, slow))

        def residual(z):
            reduced = z.reshape(5, 2) * reduced_scale
            states = reconstruct(z)
            rhs = np.column_stack((_force_rhs(states, pw, p), np.exp(t / p.tau_fat) * states[:, 1]))
            return ((reduced - [x0[1], 0.0] - h * a @ rhs) / reduced_scale).ravel()

    else:
        raise ValueError(f"Unknown map kind: {kind}")

    solution = root(residual, guess, method="hybr", options={"xtol": 1e-11})
    defect = float(np.max(np.abs(residual(solution.x))))
    # MINPACK can report stagnation at machine precision.  Accept only the
    # independently checked finite residual, never success status alone.
    if not np.isfinite(defect) or defect > 2e-10:
        raise RuntimeError(f"{kind} Radau-5 failed: {solution.message}; residual={defect}")
    stages = reconstruct(solution.x)
    if not np.isfinite(stages).all() or np.any(stages[:, 2:] <= 0):
        raise RuntimeError(f"{kind} Radau-5 produced invalid physical stages")
    return stages[-1].copy(), stages, defect


def dop853_step(x0, pw, h, amplitude, p=Parameters()):
    result = solve_ivp(
        lambda t, x: full_rhs(t, x, pw, amplitude, p),
        (0, h), x0, method="DOP853", rtol=2e-13, atol=2e-14 * SCALE,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[:, -1]


def audit(*, intervals=30):
    """Independent 30/50-Hz trajectories with low/mid/high pulse widths."""
    p = Parameters()
    # The last state deliberately violates the rest-generated fatigue ratios.
    initial_states = (
        np.array([0.16, 35.0, *p.rest]),
        np.array([0.22, 55.0, 0.75 * p.rest[0], 1.25 * p.rest[1], 1.25 * p.rest[2]]),
        np.array([0.12, 12.0, 0.6 * p.rest[0], 1.4 * p.rest[1], 1.1 * p.rest[2]]),
    )
    cases = []
    for hz in (30, 50):
        h = 1 / hz
        decay = np.exp(-h / p.tauc)
        amplitude = decay**5 + (1 + (p.rest[2] + 1.04 - 1) * decay) * sum(decay**k for k in range(5))
        for state_index, x0 in enumerate(initial_states):
            for pw in (p.pd0, 0.00037, 0.0006):
                currents = {kind: x0.copy() for kind in ("full", "condensed", "analytic", "dop853")}
                errors = {kind: np.zeros(5) for kind in ("full", "condensed", "analytic")}
                equivalence = np.zeros(5)
                max_stage_error = 0.0
                residuals = {kind: 0.0 for kind in errors}
                elapsed = {kind: 0.0 for kind in currents}
                for _ in range(intervals):
                    stages = {}
                    for kind in currents:
                        start = time.perf_counter()
                        if kind == "dop853":
                            currents[kind] = dop853_step(currents[kind], pw, h, amplitude, p)
                        else:
                            currents[kind], stages[kind], defect = radau5_step(
                                currents[kind], pw, h, amplitude, kind=kind, p=p
                            )
                            residuals[kind] = max(residuals[kind], defect)
                        elapsed[kind] += time.perf_counter() - start
                    for kind in errors:
                        errors[kind] = np.maximum(errors[kind], np.abs(currents[kind] - currents["dop853"]))
                    equivalence = np.maximum(equivalence, np.abs(currents["full"] - currents["condensed"]))
                    max_stage_error = max(max_stage_error, float(np.max(np.abs(stages["full"] - stages["condensed"]) / SCALE)))
                cases.append({
                    "frequency_hz": hz, "state_case": state_index, "initial_state": x0.tolist(), "pulse_width_s": pw,
                    "calcium_amplitude": amplitude, "intervals": intervals,
                    "maximum_endpoint_error_vs_dop853": {k: dict(zip(STATE_NAMES, v.tolist())) for k, v in errors.items()},
                    "maximum_scaled_error_vs_dop853": {k: float(np.max(v / SCALE)) for k, v in errors.items()},
                    "maximum_full_vs_condensed_scaled_endpoint_error": float(np.max(equivalence / SCALE)),
                    "maximum_full_vs_condensed_scaled_stage_error": max_stage_error,
                    "maximum_scaled_residual": residuals,
                    "python_map_elapsed_s_not_solver_benchmark": elapsed,
                })
    return {
        "scope": "isolated single-muscle periodic-node Ding; neutral mechanics; no IPOPT/MA57 benchmark",
        "radau_stages": 5, "radau_order": 9, "nonlinear_stage_unknowns": {"full": 25, "condensed": 5, "analytic": 10},
        "parameters": asdict(p), "state_scales": dict(zip(STATE_NAMES, SCALE.tolist())),
        "dop853_rtol": 2e-13, "dop853_atol": (2e-14 * SCALE).tolist(),
        "cases": cases,
        "condensation_equivalence_pass": all(c["maximum_full_vs_condensed_scaled_stage_error"] < 1e-8 for c in cases),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", type=int, default=30)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.intervals < 1:
        parser.error("--intervals must be positive")
    report = audit(intervals=args.intervals)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload)
        print(f"Saved {len(report['cases'])} cases to {args.output_json}")
    else:
        print(payload, end="")
    if not report["condensation_equivalence_pass"]:
        raise SystemExit("Full/condensed Radau-5 equivalence failed")


if __name__ == "__main__":
    main()
