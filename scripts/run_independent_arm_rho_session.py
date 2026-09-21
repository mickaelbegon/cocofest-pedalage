#!/usr/bin/env python3
"""Run one persistent unilateral IPOPT isokinetic RHO session.

Used by the process-isolated parallel launcher: each process owns CasADi,
IPOPT and MA57 state, avoiding their unsupported concurrent use from Python
threads while retaining a single compiled NLP over all requested RHO windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver
from cocofest.optimization.independent_arm_backends import BioptimIndependentArmSolver


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--equivalent-mean-torque", type=float, default=.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cycles < 1:
        raise ValueError("--cycles must be positive")
    options = driver.build_argument_parser().parse_args([
        "--solver", "ipopt", "--mechanical-formulation", "reduced",
        "--formulation", "isokinetic", "--ode-solver", "collocation",
        "--collocation-degree", "5", "--collocation-method", "radau",
        "--use-sx", "--cycles-per-window", "1", "--n-windows", str(args.cycles),
        "--stimulations-per-cycle", "30", "--n-threads", "1", "--compact-rho-output",
        "--energy-equivalent-torque", str(args.equivalent_mean_torque),
        "--disable-historical-ipopt-initial-guess",
    ])
    runtime = driver.build_unilateral_runtime(options, echo=False)
    arm = BioptimIndependentArmSolver(
        runtime["nmpc"], runtime["solver"], external_force=runtime["external_force"]
    )
    arm.set_isokinetic_work_target(args.equivalent_mean_torque * 2.0 * 3.141592653589793,
                                   args.equivalent_mean_torque)
    result = arm.run_rho_cycles(args.cycles)
    payload = {"cycles": args.cycles, "equivalent_mean_torque_nm": args.equivalent_mean_torque,
               "metrics": result.metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
