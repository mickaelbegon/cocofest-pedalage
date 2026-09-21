"""Process-local Ding reduction retaining F,A at every Radau node/stage.

Offsets are computed with the original discrete Radau operator, not continuous
exponentials. This preserves the discrete NLP up to floating point roundoff.
Usage: python scripts/benchmark_ding_radau5_local_ab.py baseline|local -- ARGS
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.sparse import vstack, csc_matrix
from scipy.sparse.linalg import splu

try:
    from scripts import benchmark_ding_radau5_symbolic_ab as base
except ModuleNotFoundError:
    import benchmark_ding_radau5_symbolic_ab as base

flat = base.flat


def frozen_limits(original, archive_dir, nlp_index, call, offset=0):
    """Replace an OCP call's incoming data by a recorded original-space call.

    The saved archives are emitted by this adapter after reconstruction, so they
    have the baseline NLP dimensions even when the native local solver is
    reduced.  This makes a fixed-input A/B possible without altering the RHO
    controller or production sources.
    """
    if archive_dir is None:
        return original, None
    path = archive_dir / f"nlp{nlp_index}_call{call + offset}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen IPOPT input: {path}")
    with np.load(path) as saved:
        copied = dict(original)
        for key in ("x0", "lbx", "ubx", "lbg", "ubg", "lam_x0", "lam_g0"):
            if key in saved:
                copied[key] = saved[key]
    return copied, path


def recorded_outer_solution(solution, archive_path, original_fg):
    """Return the recorded solution only to the RHO envelope during a replay.

    A fixed-input benchmark must not let the candidate's different local
    minimizer change the next simulated state.  Native IPOPT timings and the
    audited candidate solution are recorded before this substitution; this is
    solely the state passed back to Bioptim's outer RHO loop.
    """
    with np.load(archive_path) as saved:
        returned = dict(solution)
        for key in ("x", "g", "lam_x", "lam_g"):
            if key in saved:
                returned[key] = ca.DM(saved[key])
    f, g = original_fg(returned["x"])
    returned["f"], returned["g"] = f, g
    return returned


def intersect_affine_bounds(lower, upper, target, coefficient, offset, zlo, zhi):
    """Intersect local bounds, including either coefficient sign and constants."""
    lower, upper = np.array(lower, copy=True), np.array(upper, copy=True)
    for j, c, d, lo, hi in zip(target, coefficient, offset, zlo, zhi):
        if c == 0:
            if d < lo - 1e-9 or d > hi + 1e-9:
                raise ValueError("Fixed reconstructed state violates its original bounds")
            continue
        a, b = (lo - d) / c, (hi - d) / c
        lower[j], upper[j] = max(lower[j], min(a, b)), min(upper[j], max(a, b))
    if np.any(lower > upper + 1e-10):
        raise ValueError("Infeasible intersection of A,Tau1,Km bounds")
    upper = np.maximum(upper, lower)  # Roundoff only, checked above.
    return lower, upper


def build_local_map(nlp, interface):
    x, g = nlp["x"], nlp["g"]
    if not isinstance(x, ca.SX):
        raise ValueError("Local adapter requires SX")
    names = {x[i].name(): i for i in range(x.numel())}
    all_linear, initial, z, a_index, coefficient = [], [], [], [], []
    for phase in interface.ocp.nlp:
        ode_solver = phase.dynamics_type.ode_solver
        if (ode_solver.method != "radau" or ode_solver.polynomial_degree != 5):
            raise ValueError("Local adapter requires Radau-5")
        for muscle in phase.model.muscles_dynamics_model:
            keys = {k: f"{k}_{muscle.muscle_name}" for k in ("Cn", "A", "Tau1", "Km")}
            scale = {k: float(flat(phase.x_scaling[key].scaling)[0]) for k, key in keys.items()}
            row = {k: phase.states[key].index[0] for k, key in keys.items()}
            ratios = {"Cn": 0., "Tau1": muscle.alpha_tau1 / muscle.alpha_a,
                      "Km": muscle.alpha_km / muscle.alpha_a}
            for node, matrix in enumerate(phase.X_scaled):
                for stage in range(matrix.shape[1]):
                    indices = {k: names[matrix[i, stage].name()] for k, i in row.items()}
                    all_linear.extend(indices.values())
                    if node == 0 and stage == 0:
                        initial.extend(indices.values())
                    for k in ("Cn", "Tau1", "Km"):
                        z.append(indices[k]); a_index.append(indices["A"])
                        coefficient.append(ratios[k] * scale["A"] / scale[k])
    all_linear, initial, z, a_index = map(lambda v: np.asarray(v, dtype=int),
                                         (all_linear, initial, z, a_index))
    coefficient = np.asarray(coefficient)
    r = np.setdiff1d(np.arange(x.numel()), z)
    inverse_r = {int(v): i for i, v in enumerate(r)}
    target = np.array([inverse_r[int(i)] for i in a_index])
    lo, hi = flat(interface.limits["lbg"]), flat(interface.limits["ubg"])
    affine = ~np.asarray(ca.which_depends(g, x, 2, True), dtype=bool)
    def rows_for(columns):
        rows = np.unique(ca.jacobian(g, x[columns.tolist()]).sparsity().get_triplet()[0])
        return rows[affine[rows] & (lo[rows] == 0) & (hi[rows] == 0)]
    linear_rows, removed = rows_for(all_linear), rows_for(z)
    linear_jac = ca.jacobian(g[linear_rows.tolist()], x[all_linear.tolist()])
    if ca.symvar(linear_jac):
        raise ValueError("Linear Ding operator depends on decisions")
    matrix = ca.Function("linear_matrix", [], [linear_jac])()["o0"].sparse().tocsc()
    positions = {int(v): i for i, v in enumerate(all_linear)}
    boundary = csc_matrix((np.ones(len(initial)), (np.arange(len(initial)),
                           [positions[int(i)] for i in initial])), shape=(len(initial), len(all_linear)))
    operator = vstack([matrix, boundary]).tocsc()
    if operator.shape[0] != operator.shape[1]:
        raise ValueError(f"Unexpected affine state operator {operator.shape}")
    factor = splu(operator)
    constant = flat(ca.Function("linear_constant", [x], [g[linear_rows.tolist()]])(np.zeros(x.numel())))
    kept = np.setdiff1d(np.arange(g.numel()), removed)
    y, p = ca.SX.sym("local_states", len(r)), ca.SX.sym("radau_offsets", len(z))
    full = ca.SX.zeros(x.numel()); full[r.tolist()] = y
    full[z.tolist()] = ca.DM(coefficient) * y[target.tolist()] + p
    reduced = {"x": y, "p": p, "f": ca.substitute(nlp["f"], x, full),
               "g": ca.substitute(g[kept.tolist()], x, full)}
    # Prove that dropped dynamics are implied by retained A dynamics. Their
    # residuals need not vanish at an infeasible IPOPT iterate: G_z = B G_A.
    projection = ca.Function("local_projection", [], [ca.jacobian(full, y)])()["o0"].sparse()
    affine_jac = ca.Function("affine_jacobian", [],
        [ca.jacobian(g[linear_rows.tolist()], x)])()["o0"].sparse()
    transformed = (affine_jac @ projection).tocsr()
    transformed.sort_indices()
    linear_pos = {int(v): i for i,v in enumerate(linear_rows)}
    retained_affine = np.setdiff1d(linear_rows, removed)
    def row_values(row):
        values = transformed.getrow(linear_pos[int(row)])
        mask = np.abs(values.data) > 1e-10 * np.max(np.abs(values.data), initial=0)
        return values.indices[mask], values.data[mask]
    candidates = {}
    for row in retained_affine:
        columns, values = row_values(row)
        candidates.setdefault(tuple(columns), []).append((int(row), values))
    links, weights = [], []
    for row in removed:
        columns, values = row_values(row)
        if not len(columns):
            links.append(0); weights.append(0.); continue
        for counterpart, reference in candidates.get(tuple(columns), []):
            ratio = values[0] / reference[0]
            if np.max(np.abs(values-ratio*reference), initial=0) < 1e-9:
                links.append(counterpart); weights.append(ratio); break
        else:
            raise ValueError("Removed dynamics are not implied locally by retained A dynamics")
    certificate = ca.Function("local_equivalence_certificate", [x],
        [g[removed.tolist()] - ca.DM(weights)*g[links]])
    # Recover original removed-equation duals and fixed initial-state duals.
    initial_z = np.array([i for i in initial if i in set(z)], dtype=int)
    jz = ca.Function("removed_matrix", [], [ca.jacobian(g[removed.tolist()], x[z.tolist()])])()["o0"].sparse()
    zpositions = {int(v): i for i, v in enumerate(z)}
    bz = csc_matrix((np.ones(len(initial_z)), (np.arange(len(initial_z)),
                    [zpositions[int(i)] for i in initial_z])), shape=(len(initial_z), len(z)))
    dual_factor = splu(vstack([jz, bz]).tocsc())
    lm = ca.SX.sym("remaining_duals", len(kept))
    stationarity = ca.Function("z_stationarity", [x, lm],
        [ca.gradient(nlp["f"] + ca.dot(lm, g[kept.tolist()]), x)[z.tolist()]])
    return dict(nlp=reduced, r=r, z=z, target=target, coefficient=coefficient, initial=initial,
                removed=removed, kept=kept, factor=factor, constant=constant,
                zpos=np.array([positions[int(i)] for i in z]),
                apos=np.array([positions[int(i)] for i in a_index]),
                reconstruct=ca.Function("local_full", [y, p], [full]),
                dual_factor=dual_factor, initial_z=initial_z, stationarity=stationarity,
                certificate=certificate)


class LocalSolver:
    def __init__(self, factory, name, plugin, nlp, options, interface, mode, output, index):
        self.mode = "local" if mode == "condensed" else "baseline"
        self.original, self.output, self.index = nlp, output, index
        tic = time.perf_counter()
        self.mapping = build_local_map(nlp, interface) if self.mode == "local" else None
        self.mapping_s = time.perf_counter() - tic
        effective = self.mapping["nlp"] if self.mapping else nlp
        lm = ca.SX.sym("metrics_dual", effective["g"].numel())
        self.nnz = dict(jacobian=ca.jacobian(effective["g"], effective["x"]).nnz(),
            hessian=ca.hessian(effective["f"] + ca.dot(lm, effective["g"]), effective["x"])[0].nnz())
        tic = time.perf_counter(); self.native = factory(name, plugin, effective, options)
        self.build_s = time.perf_counter() - tic
        self.fg = ca.Function("original_audit", [nlp["x"]], [nlp["f"], nlp["g"]])
        self.metrics, self.calls = [], 0
        frozen = os.environ.get("DING_FROZEN_INPUT_DIR")
        self.frozen_input_dir = Path(frozen).resolve() if frozen else None
        self.frozen_input_offset = int(os.environ.get("DING_FROZEN_CALL_OFFSET", "0"))
        self.frozen_return_recorded = os.environ.get("DING_FROZEN_RETURN_RECORDED") == "1"
        print("DING_LOCAL_BUILD", self.mode, self.nnz, flush=True)

    def stats(self):
        return self.native.stats()

    def call(self, original):
        original, frozen_path = frozen_limits(
            original, self.frozen_input_dir, self.index, self.calls, self.frozen_input_offset
        )
        tic = time.perf_counter(); m = self.mapping
        limits = dict(original)
        if m:
            r, z, kept, target, c = (m[k] for k in ("r", "z", "kept", "target", "coefficient"))
            lo, hi = flat(original["lbx"]), flat(original["ubx"])
            if not np.allclose(lo[m["initial"]], hi[m["initial"]], atol=1e-12, rtol=0):
                raise ValueError("All incoming linear Ding states must be fixed")
            zero_force = m["factor"].solve(np.r_[-m["constant"], lo[m["initial"]]])
            offsets = zero_force[m["zpos"]] - c * zero_force[m["apos"]]
            limits["p"] = offsets
            limits["x0"] = flat(original["x0"])[r]
            certificate_error = np.max(np.abs(flat(m["certificate"](
                m["reconstruct"](limits["x0"], offsets)))), initial=0)
            if certificate_error > 1e-8:
                raise ValueError(f"Discrete local reconstruction certificate failed: {certificate_error}")
            limits["lbx"], limits["ubx"] = intersect_affine_bounds(lo[r], hi[r], target, c, offsets, lo[z], hi[z])
            for key in ("lbg", "ubg"):
                limits[key] = flat(original[key])[kept]
            if "lam_x0" in original:
                old_lx = flat(original["lam_x0"])
                lx = old_lx[r].copy(); np.add.at(lx, target, c * old_lx[z])
                limits["lam_x0"] = lx
            if "lam_g0" in original:
                limits["lam_g0"] = flat(original["lam_g0"])[kept]
        preparation = time.perf_counter() - tic
        tic = time.perf_counter(); sol = self.native.call(limits); wall = time.perf_counter() - tic
        if m:
            full = m["reconstruct"](sol["x"], offsets)
            lx, lg = np.zeros(self.original["x"].numel()), np.zeros(self.original["g"].numel())
            lx[r], lg[kept] = flat(sol["lam_x"]), flat(sol["lam_g"])
            # Assign each effective A-bound multiplier to the physical bound
            # that actually generated it, so repeated bound warm starts agree.
            for i in np.unique(target):
                candidates = np.flatnonzero((target == i) & (c != 0))
                direction = "lbx" if lx[r[i]] < 0 else "ubx"
                if abs(float(limits[direction][i]) - flat(original[direction])[r[i]]) < 1e-10:
                    continue
                for q in candidates:
                    bound = lo[z[q]] if (direction == "lbx") == (c[q] > 0) else hi[z[q]]
                    if abs((bound - offsets[q]) / c[q] - limits[direction][i]) < 1e-9:
                        lx[z[q]], lx[r[i]] = lx[r[i]] / c[q], 0.; break
            rhs = -flat(m["stationarity"](full, lg[kept])) - lx[z]
            dual = m["dual_factor"].solve(rhs, trans="T")
            lg[m["removed"]] = dual[:len(m["removed"])]; lx[m["initial_z"]] += dual[len(m["removed"]):]
            sol.update(x=full, lam_x=ca.DM(lx), lam_g=ca.DM(lg))
        f, g = self.fg(sol["x"]); sol["g"] = g
        violation = max(np.max(np.maximum(flat(original[lo]) - flat(value), 0), initial=0)
            for value, lo in ((sol["x"], "lbx"), (g, "lbg")))
        violation = max(violation, *(np.max(np.maximum(flat(value) - flat(original[hi]), 0), initial=0)
            for value, hi in ((sol["x"], "ubx"), (g, "ubg"))))
        stats = self.native.stats()
        metric = dict(mode=self.mode, call=self.calls, original_nx=self.original["x"].numel(),
            original_ng=self.original["g"].numel(), solver_nx=len(flat(limits["x0"])),
            solver_ng=len(flat(limits["lbg"])), nnz=self.nnz, mapping_s=self.mapping_s,
            solver_build_s=self.build_s, input_mapping_s=preparation, call_wall_s=wall,
            original_max_bound_constraint_violation=violation, objective=float(f),
            discrete_equivalence_certificate_max=certificate_error if m else None,
            frozen_original_input=str(frozen_path) if frozen_path else None,
            success=stats.get("success"), return_status=stats.get("return_status"),
            iterations=stats.get("iter_count"), timings={k:v for k,v in stats.items()
                if k.startswith(("t_wall_", "t_proc_", "n_call_"))})
        self.output.mkdir(parents=True, exist_ok=True)
        np.savez(self.output / f"nlp{self.index}_call{self.calls}.npz", x=flat(sol["x"]), g=flat(g),
                 lam_x=flat(sol["lam_x"]), lam_g=flat(sol["lam_g"]),
                 **{k:flat(v) for k,v in original.items() if k in ("x0", "lbx", "ubx", "lbg", "ubg", "lam_x0", "lam_g0")})
        self.metrics.append(metric)
        (self.output / f"nlp{self.index}_metrics.json").write_text(json.dumps(self.metrics, indent=2)+"\n")
        print("DING_LOCAL_RESULT " + json.dumps(metric), flush=True)
        self.calls += 1
        if frozen_path and self.frozen_return_recorded:
            return recorded_outer_solution(sol, frozen_path, self.fg)
        return sol


if __name__ == "__main__":
    if sys.argv[1] == "local":
        sys.argv[1] = "condensed"
    base.AuditedSolver = LocalSolver
    base.main()
