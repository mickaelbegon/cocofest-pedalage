#!/usr/bin/env python3
"""Run the existing dynamic IPOPT RHO benchmark with a PACE cost adapter.

Example: python scripts/run_rho_pace_benchmark.py --pace-journal run/pace.jsonl
--pace-config run/pace-config.json -- [cycling_fes_solver_comparison options]

The config has optional ``policy`` (RhoPaceConfig fields), ``initial_weights``
(mapping by model muscle name), and mandatory ``initial_weight_basis``. Omit
initial_weights for a uniform start; a physiological calibration must provide
strictly positive weights explicitly and record its provenance in the basis.
The standard benchmark result and this sidecar journal together describe the
arm. The benchmark's unweighted fatigue metrics remain comparison metrics.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLBACKEND", "Agg")


def validate_benchmark_arguments(args, config):
    """Refuse modes whose objective integration is not verified by this adapter."""
    requirements = {
        "solvers": ("ipopt",), "single_shot": False, "objective": "fatigue",
        "objective_shape": "quadratic", "formulation": "dynamic",
        "mechanical_formulation": "reduced", "cycles_per_window": 1,
        "ipopt_c_compile": False, "compact_rho_output": True,
    }
    for name, expected in requirements.items():
        if getattr(args, name, None) != expected:
            raise ValueError(f"RHO-PACE requires {name}={expected!r}")
    if not 1 <= args.n_windows <= config.max_cycles:
        raise ValueError("n_windows must be within the PACE campaign cycle limit")
    torque = args.resistive_torque
    if torque is None or not 0 < torque < float("inf"):
        raise ValueError("An explicit finite positive --signed-crank-torque is required")
    return torque


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pace-journal", type=Path, required=True)
    parser.add_argument("--pace-config", type=Path, required=True)
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    benchmark_argv = args.benchmark_args
    if benchmark_argv[:1] == ["--"]:
        benchmark_argv = benchmark_argv[1:]

    from bioptim import SolutionMerge
    from cocofest.optimization.fes_nmpc_multibody import FesNmpcMsk
    from cocofest.optimization.rho_pace import (
        RhoPaceConfig, RhoPaceController, update_bioptim_fatigue_cost,
    )
    from examples.fes_multibody.cycling import cycling_fes_solver_comparison as benchmark
    from examples.fes_multibody.cycling.cycling_pulse_width_mhe_acados_periodic import (
        _rho_solution_is_certified,
    )

    declared = json.loads(args.pace_config.read_text(encoding="utf-8"))
    unexpected = set(declared) - {"policy", "initial_weights", "initial_weight_basis"}
    if unexpected:
        raise ValueError(f"Unknown PACE configuration fields: {sorted(unexpected)}")
    if not declared.get("initial_weight_basis"):
        raise ValueError("PACE configuration needs initial_weight_basis")
    config = RhoPaceConfig(**declared.get("policy", {}))
    parsed = benchmark.build_cli().parse_args(benchmark_argv)
    torque = validate_benchmark_arguments(parsed, config)
    original = FesNmpcMsk.solve_fes_nmpc
    controllers = []

    def solve_with_pace(nmpc, update_functions, *positional, **kwargs):
        if controllers:
            raise RuntimeError("Multiple RHO sessions require separate PACE journals")
        models = nmpc.nlp[0].model.muscles_dynamics_model
        names = tuple(model.muscle_name for model in models)
        weights = declared.get("initial_weights", dict.fromkeys(names, 1.0))
        if set(weights) != set(names):
            raise ValueError("initial_weights must match actual OCP muscle names exactly")
        parameter_names = ("a_scale", "alpha_a", "alpha_tau1", "alpha_km", "tau_fat",
                           "tau1_rest", "km_rest", "tauc", "tau2", "pd0", "pdt")
        parameters = {m.muscle_name: {key: float(getattr(m, key)) for key in parameter_names}
                      for m in models}
        controller = RhoPaceController(
            names, [weights[name] for name in names], signed_crank_torque_nm=torque,
            parameters=parameters, initial_weight_basis=declared["initial_weight_basis"],
            config=config, journal_path=args.pace_journal,
        )
        controller._record({"event": "benchmark_arguments", "arguments": benchmark_argv,
                            "resolved_benchmark_arguments": json.loads(json.dumps(vars(parsed), default=str)),
                            "policy": asdict(config), "initial_seed_certification":
                            "delegated_to_original_benchmark", "comparison_metrics":
                            "original_unweighted_fatigue_metrics"})
        controllers.append(controller)

        def callback(ocp, cycle_index, solution, **extra):
            continue_solving = update_functions(ocp, cycle_index, solution, **extra)
            if solution is not None:
                solution = getattr(solution, "_cocofest_fallback_solution", None) or solution
                feasibility = getattr(solution, "_cocofest_feasibility_summary", None)
                controller._record({"event": "completed_window", "cycle_index": int(cycle_index) - 1,
                                    "solver_status": int(solution.status),
                                    "physical_tolerance_passed": bool(
                                        feasibility and feasibility.get("passes_tolerance", False)),
                                    "certified": _rho_solution_is_certified(solution.status, feasibility)})
            if not continue_solving or cycle_index >= config.max_cycles:
                controller._record({"event": "stopped", "cycle_index": cycle_index,
                                    "reason": "benchmark_stopped" if not continue_solving
                                    else "maximum_cycles_reached"})
                return False
            if solution is not None:
                certified = _rho_solution_is_certified(
                    solution.status, getattr(solution, "_cocofest_feasibility_summary", None))
                states = solution.decision_states(to_merge=SolutionMerge.NODES)
                ratios = [float(states[f"A_{m.muscle_name}"][0, -1]) / m.a_scale for m in models]
            else:
                # First solve has no completed source; original seed validation
                # and all original initial bounds are retained.
                certified = True
                ratios = [float(ocp.nlp[0].x_bounds[f"A_{m.muscle_name}"].min[0, 0]) / m.a_scale
                          for m in models]
            event = controller.boundary(
                int(cycle_index), ratios, certified=certified,
                signed_crank_torque_nm=torque,
                apply_weights=lambda values: update_bioptim_fatigue_cost(ocp, values),
            )
            if event["status"] == "refused":
                # A stopped arm must not silently become baseline RHO.
                return False
            return True

        return original(nmpc, callback, *positional, **kwargs)

    old_argv = sys.argv
    FesNmpcMsk.solve_fes_nmpc = solve_with_pace
    sys.argv = [str(REPO / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"),
                *benchmark_argv]
    try:
        runpy.run_module("examples.fes_multibody.cycling.cycling_fes_solver_comparison",
                         run_name="__main__")
        if not controllers:
            raise RuntimeError("RHO benchmark never reached the PACE integration")
        for controller in controllers:
            controller._record({"event": "launcher_completed", "ocp_cost_connected": controller.connected,
                                "physical_outcome": "see_original_benchmark_result"})
    except BaseException as error:
        for controller in controllers:
            controller._record({"event": "launcher_failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        sys.argv = old_argv
        FesNmpcMsk.solve_fes_nmpc = original


if __name__ == "__main__":
    main()
