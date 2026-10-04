"""Bounded single-phase FHO smoke for exact local Ding elimination.

Solve the original and reduced two-cycle OCP with identical original-space
inputs. This is a single-shot horizon, not two successive RHO resolutions.
The smoke deliberately retains the existing periodic-node stimulation model.
It does not establish equivalence with transient-calcium or multi-phase FHO.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, choices=(2, 3), default=2)
    parser.add_argument("--cpu", type=int, default=14)
    parser.add_argument("--timeout", type=float, default=180.)
    parser.add_argument("--modes", nargs="+", choices=("baseline", "local-interpreted", "local"),
                        default=("baseline", "local-interpreted", "local"))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hsl = Path(sys.executable).resolve().parent.parent / "opt/libhsl/v2025.7.21/lib/libhsl.so"
    if not hsl.is_file():
        parser.error(f"Missing MA57 library: {hsl}")
    if hasattr(os, "sched_getaffinity") and args.cpu not in os.sched_getaffinity(0):
        parser.error(f"CPU {args.cpu} is not in the available affinity mask")
    common = [
        "--solvers", "ipopt", "--objective", "fatigue", "--ipopt-profile", "scientific-radau5",
        "--ipopt-use-sx", "--mechanical-formulation", "reduced", "--formulation", "isokinetic",
        "--isokinetic-kinematics", "prescribed", "--isokinetic-omega", "-6.283185307179586",
        "--energy-equivalent-torque", "0.2", "--load-torque-min", "-3", "--load-torque-max", "3",
        "--ipopt-enforce-start-constraints", "--cycles-per-window", str(args.cycles),
        "--single-shot", "--n-windows", str(args.cycles), "--stimulations-per-cycle", "30",
        "--n-threads", "1", "--ipopt-linear-solver", "ma57", "--ipopt-hsl-library", str(hsl),
        "--nlp-tolerance", "1e-6", "--primal-feasibility-threshold", "1e-5",
        "--reduced-cycling-profile", str(root / "benchmark-seed/reduced-cycling-fourier12.npz"),
        "--state-scaling", "full", "--compact-rho-output", "--rho-pulse-width-transfer-mode", "repeat",
        "--rho-pulse-width-extrapolation-factor", "1.0", "--reduced-internal-crank-velocity-guard", "off",
        "--no-optional-nlp-periodic-ipopt-hot-start",
    ]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", MPLCONFIGDIR="/tmp/mpl-cocofest", PYTHONPATH=str(root))
    for key in ("DING_FROZEN_INPUT_DIR", "DING_FROZEN_RETURN_RECORDED", "DING_FROZEN_CALL_OFFSET"):
        env.pop(key, None)
    for mode in args.modes:
        destination = output / mode
        destination.mkdir(exist_ok=True)
        if (destination / "result.json").exists():
            parser.error(f"Refusing to overwrite completed result: {destination}")
        if mode in ("baseline", "local-interpreted"):
            entry = [str(root / "scripts/benchmark_ding_radau5_local_ab.py"),
                     "baseline" if mode == "baseline" else "local", "--"]
        else:
            entry = [str(root / "scripts/benchmark_ding_local_compiled.py"), "--callbacks", "hessian",
                     "--cache-dir", str(output / "cache"), "local", "--"]
        if mode != "baseline":
            archive = output / "baseline/symbolic-audit"
            if not (archive / "nlp0_call0.npz").is_file():
                parser.error(f"Missing baseline original-space input: {archive}")
            env["DING_FROZEN_INPUT_DIR"] = str(archive)
        else:
            env.pop("DING_FROZEN_INPUT_DIR", None)
        command = ["taskset", "-c", str(args.cpu), sys.executable, *entry, *common,
                   "--output-json", str(destination / "result.json")]
        (destination / "command.json").write_text(json.dumps(command, indent=2)+"\n")
        with (destination / "solver.log").open("w") as log:
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=args.timeout)
        print(f"Completed {mode} {args.cycles}-cycle single-shot FHO", flush=True)


if __name__ == "__main__":
    main()
