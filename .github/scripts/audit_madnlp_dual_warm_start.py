#!/usr/bin/env python3
"""Detect whether a CasADi MadNLP runtime consumes lam_g0/lam_x0.

This is intentionally a diagnostic, not a pass/fail assertion: older libMad
runtimes silently ignore the inputs. Use --require-consumed only when gating a
runtime that claims effective dual initialization.
"""

import argparse
import json
from pathlib import Path

import casadi as ca
import numpy as np


def _flat(value) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def run_audit(perturbation: float = 1e6, tolerance: float = 1e-10) -> dict:
    x = ca.MX.sym("x", 2)
    problem = {
        "x": x,
        "f": (x[0] - 1) ** 4 + (x[1] - 2) ** 4,
        "g": x[0] + x[1],
    }
    cases = {
        "zero": ([0.0], [0.0, 0.0]),
        "large": ([perturbation], [perturbation, -perturbation]),
        "reversed": ([-perturbation], [-perturbation, perturbation]),
    }
    option_cases = {"default": {}, "dual_initialized": {"dual_initialized": True}}
    outputs = {}
    maximum_difference = 0.0
    for option_name, extra_options in option_cases.items():
        solver = ca.nlpsol(
            f"madnlp_dual_audit_{option_name}",
            "madnlp",
            problem,
            {
                "print_time": False,
                "madnlp": {
                    "print_level": 6,
                    "tol": 1e-10,
                    "max_iter": 1,
                    **extra_options,
                },
            },
        )
        option_outputs = {}
        for case_name, (lam_g0, lam_x0) in cases.items():
            solution = solver(
                x0=[4.0, -1.0],
                lbx=[0.0, 0.0],
                ubx=[10.0, 10.0],
                lbg=[3.0],
                ubg=[3.0],
                lam_g0=lam_g0,
                lam_x0=lam_x0,
            )
            option_outputs[case_name] = {
                "x_after_one_iteration": _flat(solution["x"]).tolist(),
                "lam_g": _flat(solution["lam_g"]).tolist(),
                "lam_x": _flat(solution["lam_x"]).tolist(),
                "iteration_count": int(solver.stats().get("iter_count", 0)),
                "madnlp": solver.stats().get("madnlp", {}),
            }
        reference = option_outputs["zero"]
        for case_name in ("large", "reversed"):
            candidate = option_outputs[case_name]
            for key in ("x_after_one_iteration", "lam_g", "lam_x"):
                maximum_difference = max(
                    maximum_difference,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(candidate[key])
                                - np.asarray(reference[key])
                            )
                        )
                    ),
                )
        outputs[option_name] = option_outputs

    return {
        "schema": "cocofest-madnlp-dual-consumption-audit-v1",
        "casadi_version": ca.CasadiMeta.version(),
        "perturbation": float(perturbation),
        "detection_tolerance": float(tolerance),
        "maximum_first_step_difference": maximum_difference,
        "dual_inputs_consumed": bool(maximum_difference > tolerance),
        "cases": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--perturbation", type=float, default=1e6)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--require-consumed", action="store_true")
    args = parser.parse_args()
    report = run_audit(args.perturbation, args.tolerance)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    return int(args.require_consumed and not report["dual_inputs_consumed"])


if __name__ == "__main__":
    raise SystemExit(main())
