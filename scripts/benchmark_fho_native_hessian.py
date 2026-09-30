"""Isolated native FHO Hessian construction/evaluation audit; never solves.

Uses the final post-shake NLP intercepted immediately before CasADi nlpsol.
Run one worker count at a time, on a fixed affinity with OMP/BLAS=1.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import hashlib
import os
from pathlib import Path
import resource
import runpy
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import casadi as ca
import numpy as np


class AuditFinished(BaseException):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-command", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.base_command.read_text())
    cli = base[2:]
    for flag, value in (("--n-threads", str(args.threads)), ("--ipopt-linear-solver", "ma57")):
        cli[cli.index(flag) + 1] = value
    cli += ["--common-initial-solution", str(args.seed.resolve()),
            "--output-json", str(output / "unused-result.json")]
    (output / "command.json").write_text(json.dumps([*base[:2], *cli], indent=2))
    import bioptim.interfaces.interface_utils as iu
    import bioptim.interfaces.ipopt_interface as ii

    original_solve, original_nlpsol = ii.generic_solve, iu.nlpsol
    active = {}
    started = time.perf_counter()

    def solve(interface, expand_during_shake_tree=False):
        if (getattr(interface.opts, "c_compile", False)
                or getattr(interface.opts, "function_transform", False)):
            raise ValueError("Audit requires uncompiled, untransformed MX")
        active["interface"] = interface
        active["dispatch_started"] = time.perf_counter()
        return original_solve(interface, expand_during_shake_tree)

    def nlpsol(name, plugin, nlp, options):
        interface = active["interface"]
        if plugin != "ipopt" or not isinstance(nlp["x"], ca.MX):
            raise ValueError("Expected IPOPT MX post-shake NLP")
        before = time.perf_counter()
        report = {
            "casadi_version": ca.__version__, "bioptim_path": str(Path(iu.__file__).resolve()),
            "threads": args.threads, "affinity": sorted(os.sched_getaffinity(0)),
            "numeric_environment": {k: os.environ.get(k) for k in
                                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "nx": nlp["x"].numel(), "ng": nlp["g"].numel(),
            "application_preparation_s": before - started,
            "dispatch_and_shake_s": before - active["dispatch_started"],
            "mode": "native_exact_hessian_evaluation_only", "solve_started": False,
            "packet_callback_installed": False,
        }
        (output / "progress.json").write_text(json.dumps(report, indent=2))
        print("NATIVE_HESSIAN_BUILD " + json.dumps(report), flush=True)
        solver = original_nlpsol(name, plugin, nlp, options)
        report["nlpsol_construction_s"] = time.perf_counter() - before
        hessian = solver.get_function("nlp_hess_l")
        jacobian = solver.get_function("nlp_jac_g")
        report["hessian_signature"] = str(hessian)
        report["hessian_nnz_upper"] = hessian.sparsity_out(0).nnz()
        report["jacobian_nnz"] = jacobian.sparsity_out(1).nnz()
        children = hessian.find_functions()
        report["hessian_children"] = [{"name": f.name(), "class": f.class_name()} for f in children]
        report["hessian_child_classes"] = dict(Counter(f.class_name() for f in children))
        x = ca.DM(interface.limits["x0"])
        # Constraint ordering changes between serial/threaded Bioptim dispatch.
        # Ones weights the same sum of constraints regardless of row order.
        lam = ca.DM.ones(report["ng"])
        report["lambda_probe"] = "ones_invariant_to_constraint_row_permutation"
        parameters = ca.DM.zeros(hessian.size1_in(1), hessian.size2_in(1))
        tic = time.perf_counter()
        value = hessian(x, parameters, 1.0, lam)
        report["cold_evaluation_s"] = time.perf_counter() - tic
        samples = []
        for _ in range(args.repeats):
            tic = time.perf_counter()
            value = hessian(x, parameters, 1.0, lam)
            samples.append(time.perf_counter() - tic)
        report["evaluation_samples_s"] = samples
        report["evaluation_median_s"] = statistics.median(samples)
        values = np.array(value.nonzeros())
        objective_values = np.array(hessian(x, parameters, 1.0, ca.DM.zeros(report["ng"])).nonzeros())
        rows, columns = hessian.sparsity_out(0).get_triplet()
        report["x0_sha256"] = hashlib.sha256(np.array(x).tobytes()).hexdigest()
        report["hessian_sparsity_sha256"] = hashlib.sha256(
            np.array([rows, columns], dtype=np.int64).tobytes()).hexdigest()
        report["finite"] = bool(np.isfinite(values).all())
        report["max_abs_hessian"] = float(np.max(np.abs(values), initial=0))
        report["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        report["audit_elapsed_s"] = time.perf_counter() - started
        np.savez_compressed(output / "hessian-values.npz", values=values,
                            objective_only=objective_values, x0=np.array(x))
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print("NATIVE_HESSIAN_RESULT " + json.dumps({k: v for k, v in report.items()
                                                    if k != "hessian_children"}), flush=True)
        raise AuditFinished()

    ii.generic_solve, iu.nlpsol = solve, nlpsol
    script = Path(base[1]).resolve()
    sys.argv = [str(script), *cli]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    except AuditFinished:
        return
    raise RuntimeError("The expected final MX NLP was not intercepted")


if __name__ == "__main__":
    main()
