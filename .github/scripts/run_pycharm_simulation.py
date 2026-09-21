#!/usr/bin/env python
"""Edit CONFIG and run in PyCharm; --dry-run previews without creating files.

Extra benchmark CLI arguments are passed as individual strings, never shell
code. For example, extra_arguments=["--pulse-width-max-step-us", "25"].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess

from run_benchmarks import base_environment, conda_env_prefix, default_worker_threads

ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    "solver": "ipopt",                 # ipopt, madnlp, fatrop, acados
    "cycles": 100,
    "mechanics": "reduced",             # reduced or full
    "bilateral_reduced": False,          # bilateral bioMod, reduced theta/omega OCP
    "ipopt_linear_solver": "ma57",       # requires HSL in the solver environment
    "collocation_degree": 5,
    "stimulations_per_cycle": 30,        # count, not independently adjustable Hz
    "pulse_width_max_step_us": None,      # None or e.g. 100 for a hard adjacent bound
    "reduced_internal_crank_velocity_guard": "auto",  # auto, on or off
    "acados_qp_solver": "auto",          # auto, PARTIAL/FULL HPIPM or QPOASES
    "acados_ipopt_cycle1_seed": None,     # required for ACADOS; exact target IPOPT cycle 1
    "threads": default_worker_threads(),
    "signed_crank_torque": 0.1,          # N.m; positive resists negative crank velocity
    "terminal_q_slack": 0.002,
    "output_root": "pycharm-results",
    "compile_evaluators": True,
    "madnlp_hot_max_iterations": 100,
    "madnlp_hot_max_wall_time": 20,
    "madnlp_recovery": True,
    "dry_run": False,
    "extra_arguments": [],
}


# Shared pure configuration and capability checks; importing this package does
# not import Bioptim, CasADi, Qt or a solver.
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cocofest.simulation import LaunchPlan, build_launch_plan

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    suite = "madnlp32" if CONFIG["solver"] == "madnlp" else "rho32"
    # Read-only discovery; dry-run also works when the target runtime is absent.
    prefix = conda_env_prefix(suite)
    candidate = prefix or Path.home() / "miniforge3/envs" / f"cocofest-{suite}"
    plan = build_launch_plan(CONFIG, candidate)
    print("Command:\n" + shlex.join(plan.argv), flush=True)
    print("Working directory:", plan.cwd)
    print("Environment overrides:\n" + json.dumps(plan.environment_updates, indent=2))
    if args.dry_run or CONFIG["dry_run"]:
        return 0
    if prefix is None:
        raise RuntimeError(f"Missing solver environment: {candidate}")
    env = base_environment(prefix, plan.suite, CONFIG["threads"])
    env.update(plan.environment_updates)
    plan.cwd.mkdir(parents=True, exist_ok=True)
    return subprocess.run(plan.argv, cwd=plan.cwd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
