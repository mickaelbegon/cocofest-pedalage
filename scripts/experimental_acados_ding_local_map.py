"""Isolated ACADOS IRK map experiment, not a Bioptim/RHO solver adapter.

Retain F,A and precompute Cn,Tau1-c_tau*A,Km-c_km*A at the *same*
IRK abscissae. Polynomial forcing is valid only with this fixed tableau,
substep count and interval length. Defaults reproduce GL4 x 5 substeps.
Generated native files are placed in a fresh temporary directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver
from cocofest.models.ding2007.ding2007_with_fatigue_periodic_node import (
    DingModelPulseWidthFrequencyWithFatiguePeriodicNode,
)


def tableau(stages, family):
    nodes = np.array(ca.collocation_points(stages, family))
    polys = []
    for j, node in enumerate(nodes):
        poly = np.poly1d([1.0])
        for k, other in enumerate(nodes):
            if j != k:
                poly *= np.poly1d([1.0, -other]) / (node - other)
        polys.append(poly)
    integrals = [np.polyint(poly) for poly in polys]
    matrix = np.array([[poly(c) - poly(0) for poly in integrals] for c in nodes])
    weights = np.array([poly(1) - poly(0) for poly in integrals])
    return nodes, matrix, weights


def offsets(model, initial, duration, substeps, nodes, matrix, weights):
    """Exact elimination of the affine stage equations, modulo roundoff."""
    ratios = np.array([model.alpha_tau1, model.alpha_km]) / model.alpha_a
    value = np.array([initial[0], *(initial[3:5] - ratios * initial[2])])
    rest = np.array([0.0, *(np.array([model.tau1_rest, model.km_rest]) - ratios * model.a_scale)])
    tau = np.array([model.tauc, model.tau_fat, model.tau_fat])
    step = duration / substeps
    history = model.post_stimulation_amplitude()
    stage_values = []
    for index in range(substeps):
        forcing = np.repeat((rest / tau)[:, None], len(nodes), axis=1)
        forcing[0] = history * np.exp(-(index + nodes) * step / model.tauc) / model.tauc
        stages = np.array([
            np.linalg.solve(np.eye(len(nodes)) + step / tau[k] * matrix,
                            np.full(len(nodes), value[k]) + step * matrix @ forcing[k])
            for k in range(3)
        ])
        value += step * ((forcing - stages / tau[:, None]) @ weights)
        stage_values.append(stages)
    return np.array(stage_values), value, ratios


def interpolate_fixed_stages(time, parameters, duration, substeps, nodes):
    """Deliver stage-specific constants through ACADOS's supported t,p API."""
    step = duration / substeps
    pieces = []
    cursor = 0
    for index in range(substeps):
        local = time / step - index
        values = []
        for _ in range(3):
            value = 0
            for j, node in enumerate(nodes):
                basis = 1
                for k, other in enumerate(nodes):
                    if j != k:
                        basis *= (local - other) / (node - other)
                value += parameters[cursor + j] * basis
            cursor += len(nodes)
            values.append(value)
        pieces.append(ca.vertcat(*values))
    result = pieces[-1]
    for index in reversed(range(substeps - 1)):
        result = ca.if_else(time <= (index + 1) * step, pieces[index], result)
    return result


def simulator(model, directory, duration, substeps, stages, family, parameter_values):
    sim = AcadosSim()
    sim.model = model
    sim.code_export_directory = str(directory / model.name)
    sim.parameter_values = parameter_values
    sim.solver_options.T = duration
    sim.solver_options.integrator_type = "IRK"
    sim.solver_options.num_stages = stages
    sim.solver_options.num_steps = substeps
    sim.solver_options.collocation_type = {"radau": "GAUSS_RADAU_IIA", "legendre": "GAUSS_LEGENDRE"}[family]
    sim.solver_options.newton_iter = 12
    sim.solver_options.newton_tol = 1e-12
    sim.solver_options.sim_method_jac_reuse = 0
    return AcadosSimSolver(sim, json_file=str(directory / (model.name + ".json")), verbose=False)


def run(args):
    model = DingModelPulseWidthFrequencyWithFatiguePeriodicNode(stim_interval=args.duration)
    nodes, matrix, weights = tableau(args.stages, args.family)
    initial = np.array([0.35, 30.0, model.a_scale * 0.8, model.tau1_rest * 1.12, model.km_rest * 1.07])
    values, endpoint, ratios = offsets(model, initial, args.duration, args.substeps, nodes, matrix, weights)
    time = ca.SX.sym("time")
    control = ca.SX.sym("pw")
    full = AcadosModel()
    full.name = "ding_local_probe_full"
    full.x = ca.SX.sym("x", 5)
    full.xdot = ca.SX.sym("xdot", 5)
    full.u, full.t = control, time
    full.p = ca.SX.sym("empty", 0)
    full.f_expl_expr = model.system_dynamics(states=full.x, controls=control, time=time,
        numerical_timeseries=ca.DM([model.post_stimulation_amplitude(), 0]))
    full.f_impl_expr = full.xdot - full.f_expl_expr
    reduced = AcadosModel()
    reduced.name = "ding_local_probe_reduced"
    reduced.x = ca.SX.sym("retained", 2)
    reduced.xdot = ca.SX.sym("retained_dot", 2)
    reduced.u, reduced.t = control, time
    reduced.p = ca.SX.sym("stage_offsets", values.size)
    eliminated = interpolate_fixed_stages(time, reduced.p, args.duration, args.substeps, nodes)
    lift = ca.vertcat(eliminated[0], reduced.x[0], reduced.x[1],
        ratios[0] * reduced.x[1] + eliminated[1], ratios[1] * reduced.x[1] + eliminated[2])
    reduced.f_expl_expr = ca.substitute(full.f_expl_expr, full.x, lift)[[1, 2]]
    reduced.f_impl_expr = reduced.xdot - reduced.f_expl_expr
    directory = Path(tempfile.mkdtemp(prefix="acados-ding-local-map-"))
    reference = simulator(full, directory, args.duration, args.substeps, args.stages, args.family, np.empty(0))
    candidate = simulator(reduced, directory, args.duration, args.substeps, args.stages, args.family, values.reshape(-1))
    pw = np.array([0.0003])
    original = reference.simulate(x=initial, u=pw, xdot=np.zeros(5))
    local = candidate.simulate(x=initial[[1, 2]], u=pw, p=values.reshape(-1), xdot=np.zeros(2))
    reconstructed = np.array([endpoint[0], *local, ratios[0]*local[1]+endpoint[1], ratios[1]*local[1]+endpoint[2]])
    # Differentiate on the affine incoming manifold: dTau1=c_tau*dA, dKm=c_km*dA.
    incoming = np.zeros((6, 3))
    incoming[1, 0], incoming[2, 1], incoming[5, 2] = 1, 1, 1
    incoming[3:5, 1] = ratios
    original_sensitivity = reference.get("S_forw") @ incoming
    outgoing = incoming[:5, :2]
    reconstructed_sensitivity = outgoing @ candidate.get("S_forw")
    scale = np.maximum(np.abs(original), 1.0)
    value_error = float(np.max(np.abs(original-reconstructed)/scale))
    sensitivity_error = float(np.max(np.abs(original_sensitivity-reconstructed_sensitivity) /
                                    np.maximum(np.abs(original_sensitivity), 1.0)))
    result = {"family": args.family, "stages": args.stages, "substeps": args.substeps,
        "nx_full": 5, "nx_reduced": 2, "stage_parameter_count": values.size,
        "full_endpoint": original.tolist(), "reconstructed_endpoint": reconstructed.tolist(),
        "max_scaled_endpoint_error": value_error, "max_scaled_sensitivity_error": sensitivity_error,
        "generated_directory": str(directory),
        "scope": "single-muscle native integration map; no NLP, mechanics, bounds, or RHO timing certificate"}
    print(json.dumps(result, indent=2))
    if value_error > 1e-9 or sensitivity_error > 1e-8:
        raise AssertionError("Native map or tangent sensitivity mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("radau", "legendre"), default="legendre")
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--substeps", type=int, default=5)
    parser.add_argument("--duration", type=float, default=0.02)
    arguments = parser.parse_args()
    if arguments.stages < 1 or arguments.substeps < 1 or arguments.duration <= 0:
        parser.error("stages, substeps and duration must be positive")
    run(arguments)
