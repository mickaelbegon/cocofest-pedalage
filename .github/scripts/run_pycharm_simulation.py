#!/usr/bin/env python
"""Editable PyCharm entry point for a single cycling simulation.

Change only ``CONFIG`` below, then run this file in PyCharm.  No command-line
arguments or shell activation are needed.  ``extra_arguments`` accepts every
option exposed by ``cycling_fes_solver_comparison.py`` as a flat list, e.g.
``["--terminal-wheel-qdot-bound-margin", "0.3"]``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from run_benchmarks import default_worker_threads

ROOT = Path(__file__).resolve().parents[2]

# ---- Edit this block for each PyCharm run configuration -------------------
CONFIG = {
    "solver": "madnlp",             # "ipopt", "madnlp", or "acados"
    "cycles": 100,
    "mechanics": "reduced",         # "reduced" or "full"
    "collocation_degree": 5,
    "threads": default_worker_threads(),
    "assistance": "0.00",
    "terminal_q_slack": "0.002",
    "output_root": "pycharm-results",
    "compile_evaluators": True,
    # MadNLP only.  Setting recovery=True enables IPOPT seed recovery and the
    # certified IPOPT fallback on the same frozen RHO.
    "madnlp_hot_max_iterations": 100,
    "madnlp_hot_max_wall_time": 20,
    "madnlp_recovery": True,
    # Add any solver-native flags here.  They are appended verbatim.
    "extra_arguments": [],
}


def env_for(suite: str) -> dict[str, str]:
    prefix = Path.home() / "miniforge3" / "envs" / f"cocofest-{suite}"
    if not (prefix / "bin" / "python").exists():
        raise RuntimeError(f"Missing environment {prefix}")
    env = os.environ.copy()
    env.update({
        "CONDA_PREFIX": str(prefix), "PATH": f"{prefix / 'bin'}:{env.get('PATH', '')}",
        "GITHUB_WORKSPACE": str(ROOT), "PYTHONPATH": f"{ROOT}:{env.get('PYTHONPATH', '')}",
        "BENCHMARK_THREADS": str(CONFIG["threads"]), "BENCHMARK_CYCLES_PER_WINDOW": "1",
        "BENCHMARK_ASSISTANCE": str(CONFIG["assistance"]),
        "BENCHMARK_Q_SLACK": str(CONFIG["terminal_q_slack"]), "BENCHMARK_MAX_ITER": "2000",
        "OMP_NUM_THREADS": "1", "OMP_THREAD_LIMIT": "1", "OMP_DYNAMIC": "FALSE",
        "OPENBLAS_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "JULIA_NUM_THREADS": "1", "MPLBACKEND": "Agg",
    })
    if suite == "madnlp32":
        cache = ROOT / ".cache" / "madnlp-mumps"
        env["CASADI_CXX_ABI"] = "1"
        env["LD_LIBRARY_PATH"] = f"{cache / 'lib'}:{cache / 'share/julia/lib'}:{env.get('LD_LIBRARY_PATH', '')}"
        env["MADNLP_FAST_MAX_ITERATIONS"] = str(CONFIG["madnlp_hot_max_iterations"])
        env["MADNLP_FAST_MAX_WALL_TIME"] = str(CONFIG["madnlp_hot_max_wall_time"])
        env["MADNLP_FIRST_MAX_ITERATIONS"] = "2000"
    else:
        env["CASADI_CXX_ABI"] = "0"
        env["LD_LIBRARY_PATH"] = f"{prefix / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def main() -> int:
    solver = CONFIG["solver"]
    if solver not in {"ipopt", "madnlp", "acados"}:
        raise ValueError("CONFIG['solver'] must be ipopt, madnlp, or acados")
    suite = "madnlp32" if solver == "madnlp" else "rho32"
    env = env_for(suite)
    output = str(CONFIG["output_root"])
    if solver != "acados":
        slug = f"{solver}-pycharm"
        if solver == "madnlp" and CONFIG["madnlp_recovery"]:
            slug += "-fatigue-endurance"  # activates IPOPT recovery in runner
        command = ["bash", str(ROOT / ".github/scripts/run_cycling_benchmark_case.sh"),
            slug, solver, CONFIG["mechanics"], "mumps", "collocation", output,
            str(CONFIG["cycles"]), str(bool(CONFIG["compile_evaluators"])).lower(),
            "sx", "none", str(CONFIG["collocation_degree"]), "periodic_collocation", "auto", "auto"]
    else:
        # Start from the stable reduced ACADOS base; put full/Phase-I choices
        # and every experimental switch in extra_arguments.
        case = ROOT / output / "acados-pycharm"
        case.mkdir(parents=True, exist_ok=True)
        command = [str(Path(env["CONDA_PREFIX"]) / "bin/python"),
            str(ROOT / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"),
            "--solvers", "acados", "--objective", "fatigue", "--ipopt-use-sx",
            "--cycles-per-window", "1", "--n-windows", str(CONFIG["cycles"]),
            "--n-threads", str(CONFIG["threads"]), "--crank-assistance", str(CONFIG["assistance"]),
            "--terminal-wheel-q-slack", str(CONFIG["terminal_q_slack"]), "--acados-dir", env["CONDA_PREFIX"],
            "--acados-nlp-solver-type", "SQP", "--acados-integrator-type", "IRK",
            "--acados-sim-stages", "4", "--acados-sim-steps", "5", "--acados-max-iter", "100",
            "--mechanical-formulation", CONFIG["mechanics"], "--compact-rho-output",
            "--output-json", str(case / "result.json")]
    command += list(CONFIG["extra_arguments"])
    print("Running:\n", " ".join(map(str, command)), flush=True)
    return subprocess.run(command, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
