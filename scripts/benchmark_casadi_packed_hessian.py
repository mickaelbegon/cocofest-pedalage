"""Small, reproducible MX Hessian mapping experiment (no NLP solve).

The global variables contain shared state boundaries and independent controls.
Each stage contributes an objective and local equality constraints to the
Lagrangian. The experiment compares CasADi's Hessian of a mapped global sum
with a map of stage Hessians followed by sparse scatter-add assembly.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import casadi as ca
import numpy as np
import scipy
from scipy.sparse import coo_matrix


STATE_DIM = 4
CONTROL_DIM = 4
LOCAL_DIM = 2 * STATE_DIM + CONTROL_DIM
N_CONSTRAINTS = 4


def stage_functions() -> tuple[ca.Function, ca.Function]:
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
    stage_lag = ca.Function("stage_lag", [y, multipliers, sigma], [lagrangian])
    local_hessian = ca.hessian(lagrangian, y)[0]
    stage_hess = ca.Function(
        "stage_hess", [y, multipliers, sigma], [ca.reshape(local_hessian, LOCAL_DIM**2, 1)]
    )
    return stage_lag, stage_hess


def stage_indices(n_stages: int) -> np.ndarray:
    state_count = (n_stages + 1) * STATE_DIM
    out = np.empty((LOCAL_DIM, n_stages), dtype=np.int64)
    for k in range(n_stages):
        out[:, k] = [
            *range(k * STATE_DIM, (k + 1) * STATE_DIM),
            *range((k + 1) * STATE_DIM, (k + 2) * STATE_DIM),
            *range(state_count + k * CONTROL_DIM, state_count + (k + 1) * CONTROL_DIM),
        ]
    return out


def packet_indices(n_stages: int, packet_size: int) -> np.ndarray:
    if n_stages % packet_size:
        raise ValueError("n_stages must be divisible by packet_size")
    n_packets = n_stages // packet_size
    state_count = (n_stages + 1) * STATE_DIM
    packet_dim = (packet_size + 1) * STATE_DIM + packet_size * CONTROL_DIM
    out = np.empty((packet_dim, n_packets), dtype=np.int64)
    for packet in range(n_packets):
        start = packet * packet_size
        out[:, packet] = [
            *range(start * STATE_DIM, (start + packet_size + 1) * STATE_DIM),
            *range(state_count + start * CONTROL_DIM, state_count + (start + packet_size) * CONTROL_DIM),
        ]
    return out


def packet_hessian(stage_lag: ca.Function, packet_size: int) -> ca.Function:
    packet_dim = (packet_size + 1) * STATE_DIM + packet_size * CONTROL_DIM
    states_len = (packet_size + 1) * STATE_DIM
    y = ca.MX.sym("packet_y", packet_dim)
    multipliers = ca.MX.sym("packet_multipliers", N_CONSTRAINTS * packet_size)
    sigma = ca.MX.sym("packet_sigma")
    total = 0
    for stage in range(packet_size):
        stage_y = ca.vertcat(
            y[stage * STATE_DIM : (stage + 1) * STATE_DIM],
            y[(stage + 1) * STATE_DIM : (stage + 2) * STATE_DIM],
            y[states_len + stage * CONTROL_DIM : states_len + (stage + 1) * CONTROL_DIM],
        )
        total += stage_lag(
            stage_y, multipliers[stage * N_CONSTRAINTS : (stage + 1) * N_CONSTRAINTS], sigma
        )
    hessian = ca.hessian(total, y)[0]
    return ca.Function(
        f"packet_hess_{packet_size}",
        [y, multipliers, sigma],
        [ca.reshape(hessian, packet_dim**2, 1)],
    )


def build_global_hessian(stage_lag: ca.Function, indices: np.ndarray, backend: str, workers: int):
    n_stages = indices.shape[1]
    n_variables = (n_stages + 1) * STATE_DIM + n_stages * CONTROL_DIM
    x = ca.MX.sym("x", n_variables)
    multipliers = ca.MX.sym("multipliers", N_CONSTRAINTS, n_stages)
    sigma = ca.MX.sym("sigma")
    columns = [x[indices[:, k].tolist()] for k in range(n_stages)]
    x_local = ca.horzcat(*columns)
    mapped = stage_lag.map(n_stages, backend, workers)
    total = ca.sum2(mapped(x_local, multipliers, ca.repmat(sigma, 1, n_stages)))
    hessian = ca.hessian(total, x)[0]
    return ca.Function(f"global_hess_{backend}", [x, multipliers, sigma], [hessian])


def timed(call, repeats: int) -> dict[str, float]:
    call()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        call()
        samples.append(time.perf_counter() - t0)
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "p90_s": float(np.percentile(samples, 90)),
    }


def run_case(n_stages: int, workers: int, repeats: int, packet_sizes: list[int]) -> dict:
    indices = stage_indices(n_stages)
    n_variables = (n_stages + 1) * STATE_DIM + n_stages * CONTROL_DIM
    rng = np.random.default_rng(4400 + n_stages)
    x_val = rng.normal(0, 0.25, n_variables)
    lam_val = rng.normal(0, 0.1, (N_CONSTRAINTS, n_stages))
    sigma_val = 1.0
    local_x = x_val[indices]
    stage_lag, stage_hess = stage_functions()
    results: dict = {"n_stages": n_stages, "n_variables": n_variables, "workers": workers}

    global_functions = {}
    for backend in ("serial", "thread", "openmp"):
        t0 = time.perf_counter()
        global_functions[backend] = build_global_hessian(stage_lag, indices, backend, workers)
        results[f"global_{backend}_build_s"] = time.perf_counter() - t0
        results[f"global_{backend}_nnz"] = global_functions[backend].sparsity_out(0).nnz()

    expected = np.asarray(global_functions["serial"](x_val, lam_val, sigma_val))
    for backend, fun in global_functions.items():
        actual = np.asarray(fun(x_val, lam_val, sigma_val))
        results[f"global_{backend}_max_abs_error"] = float(np.max(np.abs(actual - expected)))
        results[f"global_{backend}_eval"] = timed(lambda: fun(x_val, lam_val, sigma_val), repeats)
        results[f"global_{backend}_children"] = [child.class_name() for child in fun.find_functions()]

    # Scatter-add shared boundary variables; this is the exact global Hessian.
    rows = np.broadcast_to(indices[:, None, :], (LOCAL_DIM, LOCAL_DIM, n_stages)).reshape(-1, order="F")
    cols = np.broadcast_to(indices[None, :, :], (LOCAL_DIM, LOCAL_DIM, n_stages)).reshape(-1, order="F")
    for backend in ("serial", "thread", "openmp"):
        mapped = stage_hess.map(n_stages, backend, workers)

        def evaluate():
            values = np.asarray(mapped(local_x, lam_val, np.ones((1, n_stages))))
            return coo_matrix((values.reshape(-1, order="F"), (rows, cols)), shape=(n_variables, n_variables)).tocsr()

        actual = evaluate().toarray()
        results[f"local_{backend}_max_abs_error"] = float(np.max(np.abs(actual - expected)))
        results[f"local_{backend}_map_only"] = timed(
            lambda: mapped(local_x, lam_val, np.ones((1, n_stages))), repeats
        )
        results[f"local_{backend}_eval_assemble"] = timed(evaluate, repeats)
        results[f"local_{backend}_map_class"] = mapped.class_name()

    results["packets"] = {}
    for packet_size in packet_sizes:
        if n_stages % packet_size:
            continue
        packet_idx = packet_indices(n_stages, packet_size)
        packet_dim, n_packets = packet_idx.shape
        packet_x = x_val[packet_idx]
        packet_lam = lam_val.reshape((N_CONSTRAINTS * packet_size, n_packets), order="F")
        packet_rows = np.broadcast_to(
            packet_idx[:, None, :], (packet_dim, packet_dim, n_packets)
        ).reshape(-1, order="F")
        packet_cols = np.broadcast_to(
            packet_idx[None, :, :], (packet_dim, packet_dim, n_packets)
        ).reshape(-1, order="F")
        t0 = time.perf_counter()
        packet_fn = packet_hessian(stage_lag, packet_size)
        item: dict = {
            "n_packets": n_packets,
            "packet_dim": packet_dim,
            "build_s": time.perf_counter() - t0,
        }
        for backend in ("serial", "thread"):
            mapped = packet_fn.map(n_packets, backend, workers)

            def map_only():
                return mapped(packet_x, packet_lam, np.ones((1, n_packets)))

            def evaluate():
                values = np.asarray(map_only())
                return coo_matrix(
                    (values.reshape(-1, order="F"), (packet_rows, packet_cols)),
                    shape=(n_variables, n_variables),
                ).tocsr()

            item[f"{backend}_max_abs_error"] = float(np.max(np.abs(evaluate().toarray() - expected)))
            item[f"{backend}_map_only"] = timed(map_only, repeats)
            item[f"{backend}_eval_assemble"] = timed(evaluate, repeats)
        results["packets"][str(packet_size)] = item

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", type=int, nargs="+", default=[20, 60])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--packet-sizes", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "casadi": ca.__version__,
        "casadi_path": ca.__file__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "numeric_threads_env": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        )},
        "cases": [run_case(n, args.workers, args.repeats, args.packet_sizes) for n in args.stages],
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
