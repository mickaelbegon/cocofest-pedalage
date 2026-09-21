"""One actual IPOPT restart at a fixed interior tick; no closed-loop claim.

Uses the production reduced full-Ding builder and a code-matched legacy model
declaration. The archive supplies a numerical warm start and a stand-in tick
measurement, not evidence of a newly propagated plant trajectory.
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples/fes_multibody/cycling"))

from cocofest.optimization.rho_tick_fallback import (
    FallbackContract, OneCycleCandidate, TickMeasurement,
    config_fingerprint, solve_tick_fallback,
)


def run(args):
    from bioptim import OdeSolver, Solver, SolutionMerge
    from bioptim.optimization.receding_horizon_optimization import RecedingHorizonOptimization
    from cycling_pulse_width_mhe import prepare_nmpc, set_fes_model, state_initial_guess_time_grid
    from cocofest.dynamics.reduced_cycling import ReducedCyclingDynamics
    from cocofest.optimization.solver_cross_rollout import load_source, PARAMETERS, file_stamp

    profile = ReducedCyclingDynamics.load(args.profile)
    source = load_source(args.source, profile, model_config=args.model_config,
                         cycle_start=0, cycles=3, cycle_duration=1., formulation_override="dynamic")
    n = source.intervals_per_cycle
    tick = args.tick
    if tick < 1 or tick + n > source.controls.shape[1]:
        raise ValueError("Tick must leave one complete cycle in the three-cycle smoke source")
    model = set_fes_model(str(args.biomod), list(np.arange(n) * source.dt), periodic_node_forcing=True)
    effective = {m.muscle_name: {key: float(getattr(m, "fmax" if key == "Fmax" else key))
                               for key in PARAMETERS} for m in model.muscles_dynamics_model}
    if any(not np.isclose(effective[m][k], source.parameters[m][k], rtol=1e-14, atol=0.)
           for m in source.muscles for k in PARAMETERS):
        raise ValueError("Production builder differs from the explicit archive-matched effective parameters")
    for flag in ("activate_force_length_relationship", "activate_force_velocity_relationship",
                 "activate_passive_force_relationship"):
        if getattr(model, flag) != source.metadata[flag]:
            raise ValueError(f"Builder differs from archive relationship flag: {flag}")
    ode = OdeSolver.COLLOCATION(polynomial_degree=5, method="radau")
    started = perf_counter()
    ocp = prepare_nmpc(model, {
        "cycle_duration": 1., "cycle_len": n, "n_cycles_to_advance": 1,
        "n_cycles_simultaneous": 1, "ode_solver": ode, "use_sx": True, "n_threads": 1,
    }, {"turn_number": 1, "pedal_config": {"x_center": .35, "y_center": 0., "radius": .1},
        "constant_crank_torque": float(source.metadata["constant_crank_torque"]),
        "enforce_start_constraints": True, "periodic_cn_sum_approximation": True}, {
        "minimize_force": False, "minimize_fatigue": True, "minimize_control": False,
        "cost_fun_weight": [0, 1, 0], "init_guess_file_path": None,
        "mechanical_formulation": "reduced", "reduced_cycling_dynamics": profile,
        "terminal_wheel_regularization_weight": 0.,
    })
    build_s = perf_counter() - started
    nlp = ocp.nlp[0]
    names = tuple(source.state_names)
    lo = np.vstack([np.asarray(nlp.x_bounds[k].min, dtype=float) for k in names])
    hi = np.vstack([np.asarray(nlp.x_bounds[k].max, dtype=float) for k in names])
    # The original absolute phase clock includes the warmup revolution.
    origin = float(source.metadata["absolute_wheel_q_origin_reference"]) - 2*np.pi*int(
        source.metadata["absolute_wheel_q_start_cycle_index"])
    reference_start = origin - 2*np.pi*tick/n
    reference_end = reference_start - 2*np.pi
    theta = names.index("theta")
    lo[theta] = [reference_start, reference_end-2., reference_end-.002]
    hi[theta] = [reference_start, reference_start+2., reference_end+.002]
    physical_lo, physical_hi = lo[:, 1].copy(), hi[:, 1].copy()
    physical_lo[theta], physical_hi[theta] = -1e6, 1e6
    lower_pw = np.asarray([source.parameters[m]["pd0"] for m in source.muscles])
    upper_pw = np.full(len(source.muscles), float(source.metadata["pulse_width_maximum_s"]))
    contract = FallbackContract(source.muscles, 1/source.dt, n, origin, 0., -2*np.pi,
        config_fingerprint({"parameters": effective, "profile": file_stamp(args.profile),
                            "biomod": file_stamp(args.biomod), "metadata": source.metadata}))
    measured = dict(zip(names, source.shooting_states[:, tick]))
    measured["omega"] += args.omega_perturbation
    candidate = OneCycleCandidate(contract, tick, names, np.linspace(0, 1, n+1),
        source.shooting_states[:, tick:tick+n+1], lo, hi, physical_lo, physical_hi,
        source.controls[:, tick:tick+n], np.repeat(lower_pw[:, None], n, axis=1),
        np.repeat(upper_pw[:, None], n, axis=1), lower_pw, upper_pw,
        reference_start, reference_end, source.provenance)
    measurement = TickMeasurement(tick, tick*source.dt, measured,
        dict(zip(source.muscles, source.controls[:, tick-1])), contract)
    report = {"scope": "single_actual_ocp_solve_from_archive_tick_measurement",
              "plant_feedback_benchmark": False, "build_s": build_s,
              "backend": "IPOPT-MA57", "transcription": "SX Radau5 full Ding reduced mechanics",
              "compiled_nlp": False, "omega_perturbation_rad_s": args.omega_perturbation,
              "effective_builder_parameters_match": True, "solver_invocations": 0}

    def solve(prepared):
        c = prepared.candidate
        grid = state_initial_guess_time_grid(n_shooting=n, turn_number=1, ode_solver=ode)
        for row, name in enumerate(names):
            nlp.x_bounds[name].min[:] = c.state_lower[row:row+1]
            nlp.x_bounds[name].max[:] = c.state_upper[row:row+1]
            nlp.x_init[name].init[:] = np.interp(grid, c.state_times_s, c.states[row])[None, :]
        for row, muscle in enumerate(source.muscles):
            name = "last_pulse_width_"+muscle
            nlp.u_init[name].init[:] = c.pw_s[row:row+1]
            # This smoke has constant physical PW bounds and no delta-PW lift.
            nlp.u_bounds[name].min[:] = c.pw_lower_s[row, 0]
            nlp.u_bounds[name].max[:] = c.pw_upper_s[row, 0]
        ocp.update_bounds(x_bounds=nlp.x_bounds, u_bounds=nlp.u_bounds)
        ocp.update_initial_guess(x_init=nlp.x_init, u_init=nlp.u_init)
        solver = Solver.IPOPT(show_online_optim=False)
        solver.set_linear_solver("ma57")
        solver.set_maximum_iterations(args.max_iterations)
        solver.set_tol(1e-8)
        solver.set_constr_viol_tol(1e-8)
        solver.set_print_level(0)
        if args.hsl_library:
            solver.set_option_unsafe(str(args.hsl_library), "hsllib")
        report["prepared_request"] = dict(prepared.provenance)
        report["reference_terminal_theta"] = c.reference_end_theta
        report["solver_invocations"] += 1
        start = perf_counter()
        solution = super(RecedingHorizonOptimization, ocp).solve(solver)
        report["solve_wall_s"] = perf_counter()-start
        report["status"] = int(solution.status)
        report["iterations"] = int(solution.iterations)
        states = solution.decision_states(to_merge=SolutionMerge.NODES)
        report["initial_transfer_max_abs_error"] = max(abs(float(states[k][0, 0])-measured[k]) for k in names)
        report["terminal_theta"] = float(states["theta"][0, -1])
        report["terminal_reference_error_rad"] = abs(report["terminal_theta"]-c.reference_end_theta)
        report["initial_values"] = {k: float(states[k][0, 0]) for k in names}
        bound_violations = []
        for row, key in enumerate(names):
            values = np.asarray(states[key], dtype=float).ravel()
            lower = np.full(values.shape, c.state_lower[row, 1])
            upper = np.full(values.shape, c.state_upper[row, 1])
            lower[[0, -1]] = c.state_lower[row, [0, 2]]
            upper[[0, -1]] = c.state_upper[row, [0, 2]]
            bound_violations.extend((np.max(lower-values), np.max(values-upper)))
        report["maximum_state_bound_violation"] = float(max(0., *bound_violations))
        controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
        report["maximum_pw_bound_violation_s"] = float(max(0., *[
            violation for row, muscle in enumerate(source.muscles)
            for violation in (
                np.max(c.physical_pw_lower_s[row]-np.asarray(controls["last_pulse_width_"+muscle])),
                np.max(np.asarray(controls["last_pulse_width_"+muscle])-c.physical_pw_upper_s[row]),
            )]))
        # IPOPT reports the primal infeasibility in its own backend stats.
        stats = ocp.ocp_solver.shaked_ocp_solver.stats()
        report["solver_core_wall_s"] = stats.get("t_wall_total")
        report["ipopt_return_status"] = stats.get("return_status")
        iterations = stats.get("iterations", {})
        inf_pr = iterations.get("inf_pr", [])
        report["final_ipopt_inf_pr"] = float(inf_pr[-1]) if len(inf_pr) else None
        report["maximum_raw_equality_residual"] = float(np.max(np.abs(np.asarray(solution.constraints))))
        report["passed"] = bool(report["status"] == 0 and report["initial_transfer_max_abs_error"] < 1e-6
                                and report["terminal_reference_error_rad"] <= .002001
                                and report["final_ipopt_inf_pr"] is not None and report["final_ipopt_inf_pr"] <= 1e-6
                                and report["maximum_raw_equality_residual"] <= 1e-6)
        report["passed"] &= report["maximum_state_bound_violation"] <= 1e-6 and report["maximum_pw_bound_violation_s"] <= 1e-10
        return solution

    try:
        solve_tick_fallback(candidate, measurement, solve=solve)
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("prepared_request", "initial_values")}, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT / "ipopt-linear-solver-150-20260904/resistance-0p10Nm/ipopt-sx-radau5-ma57-150-max2000-reduced/validated-rho-trajectory.npz")
    p.add_argument("--profile", type=Path, default=ROOT / "benchmark-seed/reduced-cycling-fourier12.npz")
    p.add_argument("--model-config", type=Path, default=ROOT / ".github/benchmark-seeds/ipopt-radau5-20260904-ding-config.json")
    p.add_argument("--biomod", type=Path, default=ROOT / "examples/msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod")
    p.add_argument("--tick", type=int, default=35)
    p.add_argument("--omega-perturbation", type=float, default=0.)
    p.add_argument("--max-iterations", type=int, default=200)
    p.add_argument("--hsl-library", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "two-timescale-tick-fallback-20260920/smoke.json")
    run(p.parse_args())
