"""Opt-in experimental adapter for strict Ding Radau stage condensation.

Run as ``python scripts/benchmark_ding_radau5_symbolic_ab.py MODE -- ARGS``;
ARGS are the usual cycling_fes_solver_comparison.py arguments. The adapter is
process-local and leaves the production Bioptim and Cocofest sources intact.
"""
from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys
import time

import casadi as ca
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu


def flat(value):
    return np.asarray(value, dtype=float).reshape(-1)


def eliminate_affine_stages(nlp, eliminated, limits):
    """Exact affine elimination; retain every original variable bound.

    Only equalities with five candidate entries qualify: each is a Radau-5
    differentiation row. Continuities (one candidate) are deliberately retained.
    All selected rows must be affine in the complete NLP vector. The resulting
    coefficient blocks must be square nonsingular 5-by-5 systems.
    """
    x, g = nlp["x"], nlp["g"]
    eliminated = np.asarray(sorted(eliminated), dtype=int)
    retained = np.setdiff1d(np.arange(x.numel()), eliminated)
    jz = ca.jacobian(g, x[eliminated.tolist()])
    row, col = jz.sparsity().get_triplet()
    counts = np.bincount(row, minlength=g.numel())
    nonlinear = np.array(ca.which_depends(g, x, 2, True), dtype=bool)
    lo, hi = flat(limits["lbg"]), flat(limits["ubg"])
    equations = np.flatnonzero((counts == 5) & ~nonlinear & (lo == 0) & (hi == 0))
    if len(equations) != len(eliminated):
        raise ValueError(f"Expected {len(eliminated)} affine Radau rows, found {len(equations)}; "
                         f"five-candidate rows={sum(counts == 5)}, nonlinear={sum(nonlinear)}")
    selected_jac = jz[equations.tolist(), :]
    if ca.symvar(selected_jac):
        raise ValueError("Elimination coefficients are not constant; variable interval unsupported")
    coefficients = ca.Function("linear_coefficients", [], [selected_jac])()["o0"].sparse().tocsc()
    factor = splu(coefficients)
    # Column adjacency isolates each muscle/state/interval block independently.
    nblocks, labels = connected_components(coefficients.T @ coefficients, directed=False)
    zero_eliminated = ca.substitute(g[equations.tolist()], x[eliminated.tolist()], ca.SX.zeros(len(eliminated)))
    reconstruction = ca.SX.zeros(len(eliminated))
    block_sizes = []
    for block in range(nblocks):
        columns = np.flatnonzero(labels == block)
        rows = np.unique(coefficients[:, columns].nonzero()[0])
        block_sizes.append((len(rows), len(columns)))
        if (len(rows), len(columns)) != (5, 5):
            raise ValueError(f"Unexpected elimination block {(len(rows), len(columns))}")
        inverse = np.linalg.inv(coefficients[rows, :][:, columns].toarray())
        reconstruction[columns.tolist()] = -ca.DM(inverse) @ zero_eliminated[rows.tolist()]
    y = ca.SX.sym("retained", len(retained))
    full_x = ca.SX.zeros(x.numel())
    full_x[retained.tolist()] = y
    full_x[eliminated.tolist()] = ca.substitute(reconstruction, x[retained.tolist()], y)
    kept_rows = np.setdiff1d(np.arange(g.numel()), equations)
    f_reduced = ca.substitute(nlp["f"], x, full_x)
    g_reduced = ca.substitute(g[kept_rows.tolist()], x, full_x)
    # Eliminated variable bounds become constraints on their reconstruction.
    reduced_nlp = {"x": y, "f": f_reduced, "g": ca.vertcat(g_reduced, full_x[eliminated.tolist()])}
    reconstruct = ca.Function("reconstruct_full_solution", [y], [full_x, ca.substitute(g, x, full_x)])
    eliminated_residual = ca.substitute(g[equations.tolist()], x, full_x)
    residual_jacobian = ca.jacobian(eliminated_residual, y)
    audit = ca.Function("audit_eliminated_equations", [y], [eliminated_residual, residual_jacobian])
    # Original eliminated-coordinate stationarity reconstructs equality duals.
    lm = ca.SX.sym("lambda_remaining", len(kept_rows))
    stationarity = ca.gradient(nlp["f"] + ca.dot(lm, g[kept_rows.tolist()]), x)[eliminated.tolist()]
    stationarity_fn = ca.Function("original_eliminated_stationarity", [x, lm], [stationarity])
    return dict(nlp=reduced_nlp, retained=retained, eliminated=eliminated, equations=equations,
                kept_rows=kept_rows, reconstruct=reconstruct, audit=audit, factor=factor,
                stationarity=stationarity_fn, block_sizes=block_sizes)


class AuditedSolver:
    def __init__(self, factory, name, plugin, nlp, options, interface, mode, output, index):
        tic = time.perf_counter()
        self.original = nlp
        self.output, self.index, self.mode = output, index, mode
        self.mapping = None
        if mode == "condensed":
            if not isinstance(nlp["x"], ca.SX):
                raise ValueError("This first adapter requires SX")
            names = {nlp["x"][i].name(): i for i in range(nlp["x"].numel())}
            eliminated = []
            for phase in interface.ocp.nlp:
                if (getattr(phase.ode_solver, "method", None) != "radau"
                        or getattr(phase.ode_solver, "polynomial_degree", None) != 5):
                    raise ValueError("Only five-stage Radau collocation is supported")
                if any(matrix.shape[1] != 6 for matrix in phase.X_scaled[:-1]):
                    raise ValueError("Expected exactly five Radau stages plus interval initial node")
                states = [i for key in phase.states.keys() if key.startswith(("Cn_", "A_", "Tau1_", "Km_"))
                          for i in phase.states[key].index]
                for matrix in phase.X_scaled[:-1]:
                    eliminated.extend(names[matrix[i, j].name()] for j in range(1, 6) for i in states)
            self.mapping = eliminate_affine_stages(nlp, eliminated, interface.limits)
            nlp = self.mapping["nlp"]
        self.condensation_time = time.perf_counter() - tic
        tic = time.perf_counter()
        self.native = factory(name, plugin, nlp, options)
        self.solver_build_time = time.perf_counter() - tic
        self.calls = 0
        self.metrics = []
        self.original_fg = ca.Function("original_fg", [self.original["x"]], [self.original["f"], self.original["g"]])
        print(f"DING_AB {mode}: x {self.original['x'].numel()} -> {nlp['x'].numel()}, "
              f"g {self.original['g'].numel()} -> {nlp['g'].numel()}, "
              f"condensation {self.condensation_time:.3f}s, solver build {self.solver_build_time:.3f}s", flush=True)

    def stats(self):
        return self.native.stats()

    def call(self, limits):
        preparation_tic = time.perf_counter()
        original_limits = limits
        mapping = self.mapping
        if mapping:
            r, z, kept = (mapping[k] for k in ("retained", "eliminated", "kept_rows"))
            for key in ("lbg", "ubg"):
                if np.any(flat(original_limits[key])[mapping["equations"]] != 0):
                    raise ValueError("A removed Radau equation changed its zero equality bound")
            limits = {k: v for k, v in limits.items() if k not in ("x0", "lbx", "ubx", "lbg", "ubg", "lam_x0", "lam_g0")}
            for key in ("x0", "lbx", "ubx"):
                limits[key] = flat(original_limits[key])[r]
            for key, bound in (("lbg", "lbx"), ("ubg", "ubx")):
                limits[key] = np.r_[flat(original_limits[key])[kept], flat(original_limits[bound])[z]]
            if "lam_x0" in original_limits or "lam_g0" in original_limits:
                lx = flat(original_limits.get("lam_x0", np.zeros(self.original["x"].numel())))
                lg = flat(original_limits.get("lam_g0", np.zeros(self.original["g"].numel())))
                limits["lam_x0"], limits["lam_g0"] = lx[r], np.r_[lg[kept], lx[z]]
            defects, defect_jac = mapping["audit"](limits["x0"])
            # Inspect sparse nonzeros, not a dense 2400-by-1703 allocation.
            if (np.max(np.abs(defects.nonzeros()), initial=0) > 1e-8
                    or np.max(np.abs(defect_jac.nonzeros()), initial=0) > 1e-8):
                raise ValueError("Eliminated equations or sensitivities failed the exactness audit")
        preparation_s = time.perf_counter() - preparation_tic
        tic = time.perf_counter()
        sol = self.native.call(limits)
        wall = time.perf_counter() - tic
        reconstruction_tic = time.perf_counter()
        if mapping:
            full_x, full_g = mapping["reconstruct"](sol["x"])
            lx = np.zeros(self.original["x"].numel())
            lg = np.zeros(self.original["g"].numel())
            lx[r], lx[z] = flat(sol["lam_x"]), flat(sol["lam_g"])[len(kept):]
            lg[kept] = flat(sol["lam_g"])[:len(kept)]
            stationarity = flat(mapping["stationarity"](full_x, lg[kept])) + lx[z]
            lg[mapping["equations"]] = mapping["factor"].solve(-stationarity, trans="T")
            sol.update(x=full_x, g=full_g, lam_x=ca.DM(lx), lam_g=ca.DM(lg))
        f, g = self.original_fg(sol["x"])
        violations = [np.max(np.maximum(flat(original_limits[lo]) - flat(value), 0), initial=0)
                      for value, lo in ((sol["x"], "lbx"), (g, "lbg"))]
        violations += [np.max(np.maximum(flat(value) - flat(original_limits[hi]), 0), initial=0)
                       for value, hi in ((sol["x"], "ubx"), (g, "ubg"))]
        stats = self.native.stats()
        metric = {"mode": self.mode, "nlp_index": self.index, "call": self.calls,
                  "original_nx": self.original["x"].numel(), "original_ng": self.original["g"].numel(),
                  "solver_nx": len(flat(limits["x0"])), "solver_ng": len(flat(limits["lbg"])),
                  "condensation_s": self.condensation_time, "solver_build_s": self.solver_build_time,
                  "input_mapping_audit_s": preparation_s,
                  "call_wall_s": wall, "reconstruction_audit_s": time.perf_counter() - reconstruction_tic,
                  "original_max_bound_constraint_violation": max(violations), "objective": float(f),
                  "success": stats.get("success"), "return_status": stats.get("return_status"),
                  "iterations": stats.get("iter_count"),
                  "timings": {k: v for k, v in stats.items() if k.startswith(("t_wall_", "t_proc_", "n_call_"))}}
        self.output.mkdir(parents=True, exist_ok=True)
        np.savez(self.output / f"nlp{self.index}_call{self.calls}.npz", x=flat(sol["x"]), g=flat(g),
                 lam_x=flat(sol["lam_x"]), lam_g=flat(sol["lam_g"]),
                 **{key: flat(value) for key, value in original_limits.items()
                    if key in ("x0", "lbx", "ubx", "lbg", "ubg", "lam_x0", "lam_g0")})
        self.metrics.append(metric)
        (self.output / f"nlp{self.index}_metrics.json").write_text(json.dumps(self.metrics, indent=2) + "\n")
        print("DING_AB_RESULT " + json.dumps(metric), flush=True)
        self.calls += 1
        return sol


def main():
    mode = sys.argv[1]
    if mode not in ("baseline", "condensed", "inspect"):
        raise ValueError("mode must be baseline, condensed or inspect")
    args = sys.argv[2:]
    if args and args[0] == "--":
        args = args[1:]
    import bioptim.interfaces.interface_utils as iu
    import bioptim.interfaces.ipopt_interface as ii

    original_solve, original_nlpsol = ii.generic_solve, iu.nlpsol
    active = {}
    output = Path(args[args.index("--output-json") + 1]).resolve().parent / "symbolic-audit"
    solver_count = 0

    def solve(interface, expand_during_shake_tree=False):
        if (getattr(interface.opts, "c_compile", False)
                or getattr(interface.opts, "function_transform", False)):
            raise ValueError("The experimental adapter requires the standard interpreted CasADi solver path")
        active["interface"] = interface
        return original_solve(interface, expand_during_shake_tree)

    def nlpsol(name, plugin, nlp, options):
        nonlocal solver_count
        interface = active["interface"]
        if plugin != "ipopt":
            raise ValueError("This experimental adapter only supports IPOPT")
        if mode == "inspect":
            print("SYMBOLIC_NLP", nlp["x"].shape, nlp["g"].shape, type(nlp["x"]), flush=True)
            for phase in interface.ocp.nlp:
                print("STATES", list(phase.states.keys()), flush=True)
                print("X_SHAPES", [x.shape for x in phase.X_scaled[:3]], flush=True)
                print("X_SYMBOLS", phase.X_scaled[0], flush=True)
            print("VECTOR", nlp["x"][:10], flush=True)
            raise RuntimeError("Inspection complete")
        result = AuditedSolver(original_nlpsol, name, plugin, nlp, options, interface, mode, output, solver_count)
        solver_count += 1
        return result

    ii.generic_solve, iu.nlpsol = solve, nlpsol
    script = Path(__file__).resolve().parents[1] / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"
    sys.argv = [str(script), *args]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
