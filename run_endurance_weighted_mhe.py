from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run cycling MHE with fixed endurance weights.")
    parser.add_argument("--cycles", type=int, default=20, help="Total simulated cycles.")
    parser.add_argument("--simultaneous", type=int, default=3, help="Number of simultaneous cycles in the MHE window.")
    parser.add_argument("--frequency", type=int, default=30, help="Stimulation frequency in Hz.")
    parser.add_argument("--torque", type=float, default=-0.20, help="Resistive crank torque in Nm.")
    parser.add_argument(
        "--objective",
        type=str,
        default="minimize_endurance_1500_weighted_fatigue",
        choices=[
            "minimize_endurance_1500_weighted_fatigue",
            "minimize_endurance_fixed_weight_risk_to_failure",
            "minimize_endurance_adaptive_weight_risk_to_failure",
        ],
        help="Custom cycling endurance objective to use in the MHE.",
    )
    parser.add_argument("--linear-solver", type=str, default="ma57", help="IPOPT linear solver, e.g. ma57 or mumps.")
    parser.add_argument("--max-iter", type=int, default=6000, help="Maximum IPOPT iterations per window.")
    parser.add_argument("--hsllib", type=str, default=None, help="Optional path to an HSL shared library for IPOPT.")
    parser.add_argument("--save", action="store_true", help="Save the solution pickle.")
    parser.add_argument("--with-init-guess", action="store_true", help="Generate initial guesses if needed.")
    args = parser.parse_args()

    # Keep matplotlib headless in this environment.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache")

    repo_root = Path(__file__).resolve().parent / "cocofest"
    cycling_dir = repo_root / "examples" / "fes_multibody" / "cycling"
    os.chdir(cycling_dir)

    from examples.fes_multibody.cycling.cycling_pulse_width_mhe import main as run_main

    run_main(
        stimulation_frequency=args.frequency,
        n_total_cycle=args.cycles,
        n_cycles_simultaneous=[args.simultaneous],
        resistive_torque=args.torque,
        cost_fun_dict={"optimized_function": [[args.objective]]},
        init_guess=args.with_init_guess,
        save=args.save,
        ipopt_linear_solver=args.linear_solver,
        ipopt_max_iter=args.max_iter,
        ipopt_hsllib=args.hsllib,
    )


if __name__ == "__main__":
    main()
