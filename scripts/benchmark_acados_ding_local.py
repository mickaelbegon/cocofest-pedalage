"""Run the ordinary benchmark, then independently audit full native IRK maps.

The audit happens after benchmark timing/JSON finalization. It generates
independent full/reduced simulators and checks the last solved window, including
mechanics, on the incoming affine manifold. This is not a performance test.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
import tempfile

import casadi as ca
import numpy as np
from acados_template import AcadosSim, AcadosSimSolver
from cocofest.optimization.acados_ding_local_reduction import DingLocalAcadosSolver


def audit_capsule(capsule):
    mapping = capsule.mapping
    root = Path(tempfile.mkdtemp(prefix="acados-ding-coupled-audit-"))
    simulators = []
    options = capsule.native_ocp.solver_options
    for label, source in (("full", capsule.acados_ocp), ("local", capsule.native_ocp)):
        sim = AcadosSim()
        sim.model = deepcopy(source.model)
        sim.model.name = "ding_coupled_audit_" + label
        sim.code_gen_options.code_export_directory = str(root / label)
        sim.parameter_values = np.asarray(source.parameter_values).copy()
        sim.solver_options.T = mapping.duration
        sim.solver_options.integrator_type = "IRK"
        sim.solver_options.num_stages = mapping.stages
        sim.solver_options.num_steps = mapping.substeps
        sim.solver_options.collocation_type = options.collocation_type
        sim.solver_options.newton_iter = options.sim_method_newton_iter
        sim.solver_options.newton_tol = options.sim_method_newton_tol
        sim.solver_options.sim_method_jac_reuse = int(np.asarray(options.sim_method_jac_reuse).reshape(-1)[0])
        simulators.append(AcadosSimSolver(sim, json_file=str(root / (label + ".json")), verbose=False))
    full, local = simulators
    tangent = np.zeros((mapping.nx, mapping.keep.size))
    tangent[mapping.keep, np.arange(mapping.keep.size)] = 1
    for muscle in mapping.muscles:
        tangent[[muscle.tau1, muscle.km], mapping.inverse[muscle.a]] = (
            muscle.ratios * mapping.scaling[muscle.a] / mapping.scaling[[muscle.tau1, muscle.km]])
    nu = int(capsule.acados_ocp.model.u.numel())
    incoming = np.zeros((mapping.nx + nu, mapping.keep.size + nu))
    incoming[:mapping.nx, :mapping.keep.size] = tangent
    incoming[mapping.nx:, mapping.keep.size:] = np.eye(nu)
    map_error, sensitivity_error, original_defect = 0.0, 0.0, 0.0
    cost_error, constraint_error, original_bounds_error = 0.0, 0.0, 0.0
    for node in range(mapping.horizon):
        x, u, p = capsule.get(node, "x"), capsule.get(node, "u"), capsule.get(node, "p")
        pp = np.concatenate((p, mapping.profiles[node]))
        full.set("t0", 0.0)
        local.set("t0", 0.0)
        fx = full.simulate(x=x, u=u, p=p, xdot=np.zeros(mapping.nx))
        lx = local.simulate(x=x[mapping.keep], u=u, p=pp, xdot=np.zeros(mapping.keep.size))
        lifted = mapping.lift(lx, node + 1)
        map_error = max(map_error, float(np.max(np.abs(fx - lifted))))
        original_defect = max(original_defect, float(np.max(np.abs(fx - capsule.get(node + 1, "x")))))
        full_sens = full.get("S_forw") @ incoming
        local_sens = tangent @ local.get("S_forw")
        sensitivity_error = max(sensitivity_error, float(np.max(np.abs(full_sens - local_sens) / np.maximum(1, np.abs(full_sens)))))
    # Compare native exported costs and generic constraints, including terminal.
    for node in range(mapping.horizon + 1):
        x = capsule.get(node, "x")
        u = capsule.get(min(node, mapping.horizon - 1), "u")
        p = capsule.get(node, "p")
        pp = np.concatenate((p, mapping.profiles[node]))
        suffix = "_0" if node == 0 else "_e" if node == mapping.horizon else ""
        for name, is_cost in (("cost_y_expr", True), ("con_h_expr", False)):
            fmodel, lmodel = capsule.acados_ocp.model, capsule.native_ocp.model
            fexpr, lexpr = getattr(fmodel, name + suffix), getattr(lmodel, name + suffix)
            if not isinstance(fexpr, ca.SX) or not fexpr.numel():
                continue
            ffun = ca.Function("audit_full", [fmodel.x, fmodel.u, fmodel.p], [fexpr])
            lfun = ca.Function("audit_local", [lmodel.x, lmodel.u, lmodel.p], [lexpr])
            error = float(np.max(np.abs(np.asarray(ffun(x, u, p)) - np.asarray(lfun(x[mapping.keep], u, pp)))))
            if is_cost:
                cost_error = max(cost_error, error)
            else:
                constraint_error = max(constraint_error, error)
        original_bounds_error = max(original_bounds_error, float(np.max(np.maximum(capsule._lower[node] - x, x - capsule._upper[node]))))
    result = {**capsule.summary, "native_status": int(capsule.status),
              "max_scaled_map_error": map_error, "max_scaled_tangent_sensitivity_error": sensitivity_error,
              "max_original_full_irk_defect": original_defect,
              "max_original_scaled_bound_violation": original_bounds_error,
              "max_cost_residual_substitution_error": cost_error,
              "max_constraint_substitution_error": constraint_error,
              "audit_directory": str(root), "audited_shooting_intervals": mapping.horizon,
              "t0_per_interval": 0.0}
    if map_error > 1e-8 or sensitivity_error > 1e-7 or cost_error > 1e-10 or constraint_error > 1e-10:
        raise AssertionError(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    destination = Path(sys.argv[1]).resolve()
    if sys.argv[2] != "--":
        raise ValueError("Usage: benchmark_acados_ding_local.py AUDIT.json -- benchmark arguments")
    sys.argv = ["cycling_fes_solver_comparison.py", *sys.argv[3:]]
    expected_local = "--acados-ding-local-reduction" in sys.argv
    captured = []
    original_init = DingLocalAcadosSolver.__init__
    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured.append(self)
    DingLocalAcadosSolver.__init__ = recording_init
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/fes_multibody/cycling"))
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"), run_name="__main__")
    if expected_local and not captured:
        raise RuntimeError("The requested local capsule was never initialized")
    result = [audit_capsule(capsule) for capsule in captured]
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
