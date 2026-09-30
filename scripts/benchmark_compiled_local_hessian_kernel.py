"""Benchmark a compiled *local exact-Hessian* kernel without solving an NLP.

This deliberately does not compile an IPOPT/FHO callback.  It measures the
building block needed by such a callback: a fixed-dimension local Lagrangian
Hessian is generated as C, compiled once, loaded as a CasADi external
function, mapped over stages and scatter-added at shared state boundaries.

The global FHO can consequently remain MX and uncompiled.  The experiment is
only evidence for the cost of this evaluator architecture; it does not claim
that an external primal function can be differentiated by CasADi.  In a real
implementation the compiled function below would be called directly from an
explicit ``hess_l`` callback, alongside separately audited objective,
constraint and Jacobian callbacks.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.sparse import coo_matrix


STATE_DIM = 4
CONTROL_DIM = 4
LOCAL_DIM = 2 * STATE_DIM + CONTROL_DIM
N_CONSTRAINTS = 4


def stage_hessian_function() -> ca.Function:
    """Return the exact local Lagrangian Hessian as a first-class output."""
    y = ca.MX.sym("y", LOCAL_DIM)
    multipliers = ca.MX.sym("multipliers", N_CONSTRAINTS)
    sigma = ca.MX.sym("sigma")
    left = y[:STATE_DIM]
    right = y[STATE_DIM : 2 * STATE_DIM]
    control = y[2 * STATE_DIM :]
    objective = ca.sumsqr(right - left) + 0.1 * ca.sumsqr(control)
    constraints = right - left
    for j in range(32):
        i = j % STATE_DIM
        r = (j + 1) % STATE_DIM
        a = left[i] * control[r] + 0.4 * right[r]
        b = right[i] * control[i] - 0.2 * left[r]
        objective += 0.05 * ca.sin(a) ** 2 + 0.03 * ca.exp(0.05 * b)
        constraints[i] += 0.01 * ca.sin(a + b)
    lagrangian = sigma * objective + ca.dot(multipliers, constraints)
    return ca.Function(
        "local_exact_hessian",
        [y, multipliers, sigma],
        [ca.reshape(ca.hessian(lagrangian, y)[0], LOCAL_DIM**2, 1)],
        ["y", "lam", "sigma"],
        ["hess"],
    )


def stage_indices(n_stages: int) -> np.ndarray:
    state_count = (n_stages + 1) * STATE_DIM
    indices = np.empty((LOCAL_DIM, n_stages), dtype=np.int64)
    for stage in range(n_stages):
        indices[:, stage] = [
            *range(stage * STATE_DIM, (stage + 1) * STATE_DIM),
            *range((stage + 1) * STATE_DIM, (stage + 2) * STATE_DIM),
            *range(state_count + stage * CONTROL_DIM, state_count + (stage + 1) * CONTROL_DIM),
        ]
    return indices


def timings(call, repeats: int) -> dict[str, float]:
    call()  # lazy loading / allocation out of the measurement
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started)
    return {
        "median_s": float(statistics.median(samples)),
        "minimum_s": float(min(samples)),
        "p90_s": float(np.percentile(samples, 90)),
    }


def mapped(function: ca.Function, stages: int, workers: int) -> ca.Function:
    return function.map(stages, "serial" if workers == 1 else "thread", workers)


def evaluate_and_assemble(
    function: ca.Function,
    local_x: np.ndarray,
    lambdas: np.ndarray,
    sigmas: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    n_variables: int,
):
    values = np.asarray(function(local_x, lambdas, sigmas)).reshape(-1, order="F")
    return coo_matrix((values, (rows, cols)), shape=(n_variables, n_variables)).tocsr()


def compile_external(function: ca.Function, build_dir: Path) -> tuple[ca.Function, float, Path]:
    build_dir.mkdir(parents=True, exist_ok=True)
    source = build_dir / "local_exact_hessian.c"
    library = build_dir / "local_exact_hessian.so"
    started = time.perf_counter()
    generator = ca.CodeGenerator(source.name)
    generator.add(function)
    generator.generate(str(build_dir) + "/")
    subprocess.run(
        ["gcc", "-O3", "-fPIC", "-shared", str(source), "-o", str(library), "-lm"],
        check=True,
        capture_output=True,
        text=True,
    )
    compile_s = time.perf_counter() - started
    return ca.external(function.name(), str(library)), compile_s, library


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", type=int, default=180)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 8, 12])
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stages < 1 or any(worker < 1 for worker in args.workers):
        parser.error("stages and workers must be positive")
    if shutil.which("gcc") is None:
        raise RuntimeError("gcc is required for the compiled local-kernel experiment")

    rng = np.random.default_rng(20260927)
    indices = stage_indices(args.stages)
    n_variables = (args.stages + 1) * STATE_DIM + args.stages * CONTROL_DIM
    x = rng.normal(0, 0.25, n_variables)
    local_x = x[indices]
    lambdas = rng.normal(0, 0.1, (N_CONSTRAINTS, args.stages))
    sigmas = np.ones((1, args.stages))
    rows = np.broadcast_to(indices[:, None, :], (LOCAL_DIM, LOCAL_DIM, args.stages)).reshape(-1, order="F")
    cols = np.broadcast_to(indices[None, :, :], (LOCAL_DIM, LOCAL_DIM, args.stages)).reshape(-1, order="F")
    started = time.perf_counter()
    vm = stage_hessian_function()
    vm_build_s = time.perf_counter() - started
    compiled, compile_s, library = compile_external(vm, args.output.parent / (args.output.stem + "_codegen"))

    result: dict = {
        "kind": "local_exact_hessian_kernel_only_not_an_ipopt_or_fho_solve",
        "casadi": ca.__version__,
        "python": platform.python_version(),
        "stages": args.stages,
        "local_dimension": LOCAL_DIM,
        "local_hessian_entries": LOCAL_DIM**2,
        "global_variables": n_variables,
        "global_hessian_triplets_before_coalescing": int(rows.size),
        "vm_symbolic_build_s": vm_build_s,
        "c_generate_and_compile_s": compile_s,
        "generated_library_bytes": library.stat().st_size,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "numeric_threads_env": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "workers": {},
        "limitations": [
            "The output is an already differentiated local exact Hessian, not an external primal function differentiated by CasADi.",
            "Scatter-add is Python/SciPy here; a production hess_l callback needs native sparse assembly and an IPOPT sparsity/order audit.",
            "No IPOPT solve, feasibility certificate, or FHO graph construction is performed.",
        ],
    }
    for workers in args.workers:
        vm_map = mapped(vm, args.stages, workers)
        c_map = mapped(compiled, args.stages, workers)
        vm_matrix = evaluate_and_assemble(vm_map, local_x, lambdas, sigmas, rows, cols, n_variables)
        c_matrix = evaluate_and_assemble(c_map, local_x, lambdas, sigmas, rows, cols, n_variables)
        max_error = float(np.max(np.abs((vm_matrix - c_matrix).data))) if (vm_matrix != c_matrix).nnz else 0.0
        vm_map_time = timings(lambda: vm_map(local_x, lambdas, sigmas), args.repeats)
        c_map_time = timings(lambda: c_map(local_x, lambdas, sigmas), args.repeats)
        vm_full_time = timings(lambda: evaluate_and_assemble(vm_map, local_x, lambdas, sigmas, rows, cols, n_variables), args.repeats)
        c_full_time = timings(lambda: evaluate_and_assemble(c_map, local_x, lambdas, sigmas, rows, cols, n_variables), args.repeats)
        result["workers"][str(workers)] = {
            "map_backend": vm_map.class_name(),
            "matrix_nnz": int(vm_matrix.nnz),
            "max_abs_error_compiled_vs_vm": max_error,
            "vm_map_only": vm_map_time,
            "compiled_map_only": c_map_time,
            "vm_map_and_assemble": vm_full_time,
            "compiled_map_and_assemble": c_full_time,
            "map_only_speedup": vm_map_time["median_s"] / c_map_time["median_s"],
            "end_to_end_speedup": vm_full_time["median_s"] / c_full_time["median_s"],
            "compiled_break_even_map_evaluations": compile_s / max(vm_map_time["median_s"] - c_map_time["median_s"], 1e-15),
            "compiled_break_even_end_to_end_evaluations": compile_s / max(vm_full_time["median_s"] - c_full_time["median_s"], 1e-15),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
