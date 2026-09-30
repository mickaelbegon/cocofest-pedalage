"""Isolated direct/implicit reduced-mechanics kernel experiment (not an RHO benchmark)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import casadi as ca
import numpy as np

from cocofest.dynamics.reduced_cycling import ReducedCyclingDynamics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = ReducedCyclingDynamics.load(args.profile)
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260925)
    count = 4096
    # Identical physical states, acceleration and load for all formulations.
    z = ca.SX.sym("z", 4 + len(profile.muscle_names))
    theta, omega, alpha, load = z[0], z[1], z[2], z[3]
    forces = z[4:]
    coefficients = profile.coefficients.casadi(profile.kinematics.progress(theta))
    inertia = coefficients[0]
    numerator = ca.dot(coefficients[profile._muscle_slice], forces) + coefficients[profile._external_index] * load - coefficients[1] - coefficients[2] * omega**2
    phases = np.linspace(-2 * np.pi, 0, count)
    inertias = profile.coefficients.evaluate(profile.kinematics.progress(phases))[0]
    reference_inertia = float(np.median(inertias))
    expressions = {
        "direct": alpha - numerator / inertia,
        "implicit_inverse": (inertia * alpha - numerator) / reference_inertia,
    }
    samples = np.vstack((rng.uniform(-2 * np.pi, 0, count), rng.uniform(-9, -3, count), rng.uniform(-30, 30, count), rng.uniform(-3, 3, count), rng.uniform(0, 400, (len(profile.muscle_names), count))))
    results = {}
    functions = {}
    natives = {}
    for name, residual in expressions.items():
        jacobian = ca.jacobian(residual, z)
        hessian = ca.hessian(residual, z)[0]
        f = ca.Function(name, [z], [residual, jacobian, hessian])
        functions[name] = f
        started = perf_counter()
        generator = ca.CodeGenerator(f"{name}.c")
        generator.add(f)
        generator.generate(str(args.output.resolve()) + "/")
        subprocess.run(["gcc", "-O3", "-fPIC", "-shared", str(args.output / f"{name}.c"), "-o", str(args.output / f"{name}.so"), "-lm"], check=True)
        compile_time = perf_counter() - started
        native = ca.external(name, str((args.output / f"{name}.so").resolve())).map(count, "serial")
        native(samples)
        natives[name] = native
        results[name] = {"instructions": f.n_instructions(), "jacobian_nnz": jacobian.sparsity().nnz(), "hessian_nnz": hessian.sparsity().nnz(), "compile_s": compile_time}
    durations = {name: [] for name in natives}
    for repetition in range(21):
        for name in (list(natives) if repetition % 2 == 0 else list(natives)[::-1]):
            started = perf_counter()
            natives[name](samples)
            durations[name].append((perf_counter() - started) / count * 1e6)
    for name, timings in durations.items():
        results[name].update(microseconds_per_residual_jacobian_hessian_median=float(np.median(timings)), microseconds_per_residual_jacobian_hessian_p90=float(np.percentile(timings, 90)))
    # Algebraic identity away from feasible points, root identity on feasible points.
    evaluate_inertia = ca.Function("inertia", [z], [inertia]).map(count)(samples)
    direct = np.asarray(functions["direct"].map(count)(samples)[0]).ravel()
    inverse = np.asarray(functions["implicit_inverse"].map(count)(samples)[0]).ravel()
    identity_error = float(np.max(np.abs(inverse - np.asarray(evaluate_inertia).ravel() / reference_inertia * direct)))
    samples[2] -= direct
    root_error = float(np.max(np.abs(np.asarray(functions["implicit_inverse"].map(count)(samples)[0]))))
    record = {"kind": "mechanical_kernel_only_not_RHO", "casadi": ca.__version__, "profile": str(args.profile.resolve()), "cpu_affinity": sorted(os.sched_getaffinity(0)), "samples": count, "repetitions": 21, "timing_order": "alternating", "state_ranges": {"theta_rad": [-2*np.pi, 0], "omega_rad_s": [-9, -3], "alpha_rad_s2": [-30, 30], "load_nm": [-3, 3], "muscle_forces_n": [0, 400]}, "inertia_min": float(np.min(inertias)), "inertia_max": float(np.max(inertias)), "inertia_median": reference_inertia, "scaled_residual_identity_max": identity_error, "inverse_at_direct_root_max": root_error, "cases": results}
    (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
