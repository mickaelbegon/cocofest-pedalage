"""Isolated bounded IPOPT/MA57 exact FHO solve; never emits a certificate.

Intercepts the final post-shake NLP, records construction and solve statistics,
and exits before application result handling. Run worker counts sequentially
with identical seed, CPU affinity, OMP/BLAS=1, and iteration cap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import casadi as ca
import numpy as np


class BenchmarkFinished(BaseException):
    pass


def digest(value):
    return hashlib.sha256(np.asarray(value, dtype=float).tobytes()).hexdigest()


def violation(value, lower, upper):
    arrays = [np.asarray(v, dtype=float).reshape(-1) for v in (value, lower, upper)]
    return float(np.maximum(np.maximum(arrays[1] - arrays[0], arrays[0] - arrays[2]), 0).max(initial=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-command", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--hsl-library", type=Path, required=True)
    parser.add_argument("--map-first-node-only", action="store_true",
                        help="Experimental Bioptim ThreadMap construction audit")
    args = parser.parse_args()
    hsl_library = args.hsl_library.resolve(strict=True)
    # Fail before constructing a large NLP if HSL cannot load in this environment.
    z = ca.MX.sym("z")
    preflight = ca.nlpsol("ma57_preflight", "ipopt", {"x": z, "f": (z - 1)**2},
                         {"ipopt.linear_solver": "ma57", "ipopt.hsllib": str(hsl_library),
                          "ipopt.print_level": 0, "print_time": False})
    preflight(x0=0)
    if not preflight.stats()["success"]:
        raise RuntimeError("MA57 preflight failed: " + preflight.stats()["return_status"])
    del preflight
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.base_command.read_text())
    cli = base[2:]
    for flag, value in (("--n-threads", str(args.threads)), ("--ipopt-linear-solver", "ma57"),
                        ("--output-json", str(output / "unused-result.json"))):
        if flag in cli:
            cli[cli.index(flag) + 1] = value
        else:
            cli.extend([flag, value])
    (output / "command.json").write_text(json.dumps([*base[:2], *cli], indent=2))
    import bioptim.interfaces.interface_utils as iu
    import bioptim.interfaces.ipopt_interface as ii

    if args.map_first_node_only:
        from bioptim.limits.penalty_option import PenaltyOption

        original_set_penalty_function = PenaltyOption._set_penalty_function

        def set_penalty_function_first_node_only(self, controllers, fcn):
            controller = controllers[-1] if isinstance(controllers, list) else controllers
            skip = bool(self.multi_thread and len(self.node_idx) > 1
                        and controller.node_index != self.node_idx[0])
            if not skip:
                return original_set_penalty_function(self, controllers, fcn)
            self.multi_thread = False
            try:
                return original_set_penalty_function(self, controllers, fcn)
            finally:
                self.multi_thread = True

        PenaltyOption._set_penalty_function = set_penalty_function_first_node_only

    original_solve, original_nlpsol = ii.generic_solve, iu.nlpsol
    active = {}
    started = time.perf_counter()

    def solve(interface, expand_during_shake_tree=False):
        if (getattr(interface.opts, "c_compile", False)
                or getattr(interface.opts, "function_transform", False)):
            raise ValueError("Benchmark requires uncompiled, untransformed MX")
        active["interface"] = interface
        active["dispatch_started"] = time.perf_counter()
        return original_solve(interface, expand_during_shake_tree)

    def nlpsol(name, plugin, nlp, options):
        interface = active["interface"]
        if plugin != "ipopt" or not isinstance(nlp["x"], ca.MX):
            raise ValueError("Expected IPOPT MX post-shake NLP")
        before = time.perf_counter()
        options = dict(options)
        options.update({"ipopt.max_iter": args.iterations,
                        "ipopt.hessian_approximation": "exact",
                        "ipopt.hsllib": str(hsl_library),
                        "ipopt.linear_solver": "ma57", "ipopt.print_level": 5,
                        "ipopt.print_timing_statistics": "yes", "record_time": True})
        options.pop("iteration_callback", None)
        report = {
            "casadi_version": ca.__version__, "bioptim_path": str(Path(iu.__file__).resolve()),
            "threads": args.threads, "affinity": sorted(os.sched_getaffinity(0)),
            "map_first_node_only": args.map_first_node_only,
            "numeric_environment": {k: os.environ.get(k) for k in
                                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "nx": nlp["x"].numel(), "ng": nlp["g"].numel(),
            "application_preparation_s": before - started,
            "dispatch_and_shake_s": before - active["dispatch_started"],
            "mode": "native_exact_bounded_solve", "certificate_valid": False,
            "iteration_cap": args.iterations,
            "ipopt_options": {k: v for k, v in options.items() if k.startswith("ipopt.")},
        }
        (output / "progress.json").write_text(json.dumps(report, indent=2))
        print("BUILD " + json.dumps(report), flush=True)
        solver = original_nlpsol(name, plugin, nlp, options)
        report["nlpsol_construction_s"] = time.perf_counter() - before
        hessian = solver.get_function("nlp_hess_l")
        jacobian = solver.get_function("nlp_jac_g")
        report["hessian_nnz_upper"] = hessian.sparsity_out(0).nnz()
        report["jacobian_nnz"] = jacobian.sparsity_out(1).nnz()
        report["hessian_sparsity_sha256"] = digest(hessian.sparsity_out(0).get_triplet())
        limits = {k: ca.DM(v) for k, v in interface.limits.items()}
        limits["lam_x0"] = ca.DM.zeros(report["nx"])
        limits["lam_g0"] = ca.DM.zeros(report["ng"])
        report["x0_sha256"] = digest(limits["x0"])
        report["lbx_sha256"], report["ubx_sha256"] = digest(limits["lbx"]), digest(limits["ubx"])
        tic = time.perf_counter()
        parameters = ca.DM.zeros(solver.size1_in("p"), solver.size2_in("p"))
        f0 = solver.get_function("nlp_f")(limits["x0"], parameters)
        g0 = solver.get_function("nlp_g")(limits["x0"], parameters)
        report["initial_audit_s"] = time.perf_counter() - tic
        report["initial_objective"] = float(f0)
        report["initial_constraint_violation"] = violation(g0, limits["lbg"], limits["ubg"])
        np.savez_compressed(output / "initial-probe.npz", x0=np.asarray(limits["x0"]),
                            g0=np.asarray(g0), lbg=np.asarray(limits["lbg"]), ubg=np.asarray(limits["ubg"]))
        report["pre_solve_elapsed_s"] = time.perf_counter() - started
        (output / "progress.json").write_text(json.dumps(report, indent=2))
        print("SOLVE_START " + json.dumps(report), flush=True)
        tic = time.perf_counter()
        solution = solver(**limits)
        report["solve_elapsed_s"] = time.perf_counter() - tic
        stats = solver.stats()
        report["stats"] = stats
        report["final_objective"] = float(solution["f"])
        report["final_constraint_violation"] = violation(solution["g"], limits["lbg"], limits["ubg"])
        report["final_variable_violation"] = violation(solution["x"], limits["lbx"], limits["ubx"])
        report["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        report["benchmark_elapsed_s"] = time.perf_counter() - started
        np.savez_compressed(output / "final-probe-uncertified.npz", x=np.asarray(solution["x"]),
                            g=np.asarray(solution["g"]))
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print("BOUNDED_SOLVE_RESULT " + json.dumps(report), flush=True)
        raise BenchmarkFinished()

    ii.generic_solve, iu.nlpsol = solve, nlpsol
    script = Path(base[1]).resolve()
    sys.argv = [str(script), *cli]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    except BenchmarkFinished:
        return
    raise RuntimeError("The expected final MX NLP was not intercepted")


if __name__ == "__main__":
    main()
