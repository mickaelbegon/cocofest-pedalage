"""Audit exact local Hessian terms from a post-shake MX FHO without solving.

This is deliberately a *term-extraction audit*, not a production callback:
Bioptim currently exposes the final aggregate ``f,g`` graph but not ownership
of its local penalty terms.  We nevertheless recover exact local constraint
terms from Jacobian row dependencies, then compare each embedded local
Hessian to CasADi's native ``nlp_hess_l`` at the identical post-shake point.
It gives a bounded, reproducible gate before attempting a packet callback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import statistics
import subprocess
import sys
import time

import casadi as ca
import numpy as np


class AuditFinished(BaseException):
    pass


def _command_from_log(path: Path) -> list[str]:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("command: "):
            return shlex.split(line.removeprefix("command: "))
    raise ValueError(f"No 'command: ' line in {path}")


def _replace_or_append(cli: list[str], flag: str, value: str) -> None:
    if flag in cli:
        cli[cli.index(flag) + 1] = value
    else:
        cli.extend((flag, value))


def _local_constraint_kernel(
    x: ca.MX,
    p: ca.MX,
    g_row: ca.MX,
    variables: tuple[int, ...],
    name: str,
) -> tuple[ca.Function, ca.Function, ca.Sparsity]:
    """Return primal and exact local-Hessian evaluators for one constraint."""
    z = ca.MX.sym(f"z_{name}", len(variables))
    # CasADi MX permits vector substitution (not scalar indexing).  Expanding
    # immediately afterwards is essential: it removes the unused global-zero
    # entries before local code generation.
    embedded = ca.MX.zeros(x.numel(), 1)
    embedded[list(variables)] = z
    local_g = ca.substitute(g_row, x, embedded)
    primal = ca.Function(f"{name}_primal", [z, p], [local_g]).expand()
    sigma = ca.MX.sym(f"sigma_{name}")
    lam = ca.MX.sym(f"lambda_{name}")
    # sigma is retained for the IPOPT hess_lag ABI even though f_local=0.
    hessian = ca.triu(ca.hessian(lam * primal(z, p), z)[0])
    values = ca.vertcat(*[hessian.nz[index] for index in range(hessian.nnz())])
    local_hessian = ca.Function(f"{name}_hessian", [z, p, sigma, lam], [values])
    return primal, local_hessian, hessian.sparsity()


def _embed_upper(values: np.ndarray, variables: tuple[int, ...], sparsity: ca.Sparsity) -> dict[tuple[int, int], float]:
    rows, cols = sparsity.get_triplet()
    embedded: dict[tuple[int, int], float] = {}
    for value, row, col in zip(values.reshape(-1), rows, cols):
        key = variables[row], variables[col]
        embedded[key] = embedded.get(key, 0.0) + float(value)
    return embedded


def _compile_external(function: ca.Function, directory: Path) -> tuple[ca.Function, float]:
    """Generate one already-differentiated local Hessian, never the MX FHO."""
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / f"{function.name()}.c"
    library = directory / f"{function.name()}.so"
    started = time.perf_counter()
    generator = ca.CodeGenerator(source.name)
    generator.add(function)
    generator.generate(str(directory) + "/")
    subprocess.run(
        ["gcc", "-O3", "-fPIC", "-shared", str(source), "-o", str(library), "-lm"],
        check=True, capture_output=True, text=True,
    )
    return ca.external(function.name(), str(library)), time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-rows", type=int, default=8)
    parser.add_argument("--max-local-variables", type=int, default=128)
    parser.add_argument("--hsl-library", type=Path, required=True)
    parser.add_argument("--compile-local", action="store_true",
                        help="Generate C only for sampled, already differentiated local Hessians")
    args = parser.parse_args()
    if args.sample_rows < 1 or args.max_local_variables < 1:
        parser.error("sample-rows and max-local-variables must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = _command_from_log(args.command_log.resolve())
    script, cli = Path(command[1]).resolve(), command[2:]
    _replace_or_append(cli, "--ipopt-hsl-library", str(args.hsl_library.resolve()))
    _replace_or_append(cli, "--output-json", str(output / "unused-result.json"))
    (output / "command.json").write_text(json.dumps([command[0], str(script), *cli], indent=2) + "\n")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import bioptim.interfaces.interface_utils as iu
    import bioptim.interfaces.ipopt_interface as ii

    original_solve, original_nlpsol = ii.generic_solve, iu.nlpsol
    active: dict[str, object] = {}
    started = time.perf_counter()

    def solve(interface, expand_during_shake_tree=False):
        if getattr(interface.opts, "c_compile", False) or getattr(interface.opts, "function_transform", False):
            raise ValueError("Packet audit requires the uncompiled MX FHO")
        active["interface"] = interface
        return original_solve(interface, expand_during_shake_tree)

    def nlpsol(name, plugin, nlp, options):
        if plugin != "ipopt" or not isinstance(nlp["x"], ca.MX):
            raise ValueError("Expected final MX IPOPT NLP")
        interface = active["interface"]
        solver = original_nlpsol(name, plugin, nlp, options)
        hess = solver.get_function("nlp_hess_l")
        jac = solver.get_function("nlp_jac_g")
        nx, ng = nlp["x"].numel(), nlp["g"].numel()
        x0 = ca.DM(interface.limits["x0"])
        p0 = ca.DM.zeros(hess.size1_in(1), hess.size2_in(1))
        rows, columns = jac.sparsity_out(1).get_triplet()
        row_variables: dict[int, list[int]] = {}
        for row, column in zip(rows, columns):
            row_variables.setdefault(row, []).append(column)
        eligible = [(row, tuple(sorted(set(columns)))) for row, columns in row_variables.items()
                    if 0 < len(set(columns)) <= args.max_local_variables]
        # Uniformly span the eligible post-shake rows; this avoids selecting
        # only the first collocation interval.
        positions = np.linspace(0, len(eligible) - 1, min(args.sample_rows, len(eligible)), dtype=int)
        selected = [eligible[position] for position in positions]
        report: dict[str, object] = {
            "mode": "post_shake_constraint_term_exactness_only_no_solve",
            "nx": nx, "ng": ng, "hessian_upper_nnz": hess.sparsity_out(0).nnz(),
            "jacobian_nnz": jac.sparsity_out(1).nnz(),
            "casadi_version": ca.__version__,
            "affinity": sorted(os.sched_getaffinity(0)),
            "numeric_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "eligible_row_count": len(eligible), "selected_row_count": len(selected),
            "selected_rows": [], "build_and_audit_elapsed_s": None,
        }
        for ordinal, (row, variables) in enumerate(selected):
            primal, local, upper_sparsity = _local_constraint_kernel(
                nlp["x"], nlp.get("p", ca.MX.zeros(0, 1)), nlp["g"][row], variables, f"row_{row}"
            )
            z0 = x0[list(variables)]
            tic = time.perf_counter()
            local_value = np.asarray(local(z0, p0, 0.0, 1.0)).reshape(-1)
            local_s = time.perf_counter() - tic
            lambda_row = ca.DM.zeros(ng, 1)
            lambda_row[row] = 1.0
            native = hess(x0, p0, 0.0, lambda_row)
            embedded_local = _embed_upper(local_value, variables, upper_sparsity)
            native_rows, native_cols = native.sparsity().get_triplet()
            embedded_native = {
                (native_row, native_col): float(value)
                for value, native_row, native_col in zip(
                    np.asarray(native.nonzeros()).reshape(-1), native_rows, native_cols
                )
            }
            keys = set(embedded_local) | set(embedded_native)
            error = max((abs(embedded_local.get(key, 0.0) - embedded_native.get(key, 0.0)) for key in keys), default=0.0)
            local_set = set(variables)
            outside = sum(
                1 for (native_row, native_col), value in embedded_native.items()
                if value and (native_row not in local_set or native_col not in local_set)
            )
            entry = {
                "row": row, "local_variable_count": len(variables),
                "local_hessian_nnz": upper_sparsity.nnz(),
                "local_evaluation_s": local_s,
                "max_abs_error_vs_native": error,
                "native_nonzero_outside_local_support": outside,
                "local_primal_at_x0": float(primal(z0, p0)),
            }
            if args.compile_local:
                compiled, compile_s = _compile_external(local, output / "compiled-local-terms")
                tic = time.perf_counter()
                compiled_value = np.asarray(compiled(z0, p0, 0.0, 1.0)).reshape(-1)
                compiled_s = time.perf_counter() - tic
                entry.update({
                    "local_codegen_compile_s": compile_s,
                    "compiled_local_evaluation_s": compiled_s,
                    "compiled_max_abs_error": float(np.max(np.abs(compiled_value - local_value), initial=0.0)),
                    "compiled_local_speedup": local_s / max(compiled_s, 1e-15),
                })
            report["selected_rows"].append(entry)
        entries = report["selected_rows"]
        report["max_abs_error_vs_native"] = max((entry["max_abs_error_vs_native"] for entry in entries), default=0.0)
        report["median_local_evaluation_s"] = statistics.median([entry["local_evaluation_s"] for entry in entries]) if entries else None
        report["x0_sha256"] = hashlib.sha256(np.asarray(x0).tobytes()).hexdigest()
        report["build_and_audit_elapsed_s"] = time.perf_counter() - started
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("POST_SHAKE_PACKET_AUDIT " + json.dumps(report), flush=True)
        raise AuditFinished()

    ii.generic_solve, iu.nlpsol = solve, nlpsol
    sys.argv = [str(script), *cli]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    except AuditFinished:
        return
    raise RuntimeError("The final MX NLP was not intercepted")


if __name__ == "__main__":
    main()
