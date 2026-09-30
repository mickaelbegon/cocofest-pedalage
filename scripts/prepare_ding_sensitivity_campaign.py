#!/usr/bin/env python3
"""Prepare reproducible four-muscle Ding sensitivity campaigns.

The four cases perturb *all* Ding muscles coherently.  A more-endurant case
increases force reserve (Fmax/a_scale) and reduces both force-induced fatigue
and its time constant; the less-endurant case makes the inverse change.  The
script writes immutable inputs only: RHO and RHO-BO launchers consume them,
while FHO is deliberately gated on a certified variant-specific RHO prefix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE = {
    "Biceps": {"Fmax": 149.0, "a_scale": 3314.7, "alpha_a": -5.6e-2, "tau_fat": 179.6},
    "Triceps": {"Fmax": 617.0, "a_scale": 7036.3, "alpha_a": -2.4e-2, "tau_fat": 76.2},
    "Delt_ant": {"Fmax": 48.0, "a_scale": 1148.6, "alpha_a": -1.4e-1, "tau_fat": 445.5},
    "Delt_post": {"Fmax": 51.0, "a_scale": 1234.5, "alpha_a": -1.1e-1, "tau_fat": 342.7},
}
CASES = (("more_endurant_05", 1.05), ("more_endurant_10", 1.10),
         ("less_endurant_05", 0.95), ("less_endurant_10", 0.90))
MUSCLES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")


def write_new(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite campaign input: {path}")
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def parameters(factor: float) -> dict[str, dict[str, float]]:
    # Reserve tracks factor.  Fatigue and recovery change in the opposite
    # direction for a more-endurant phenotype (factor > 1).
    inverse = 2.0 - factor
    return {name: {
        "Fmax": values["Fmax"] * factor,
        "a_scale": values["a_scale"] * factor,
        "alpha_a": values["alpha_a"] * inverse,
        "tau_fat": values["tau_fat"] * inverse,
    } for name, values in BASE.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resistance-nm", type=float, default=.30)
    parser.add_argument("--rho-cycles", type=int, default=120)
    parser.add_argument("--bo-trials", type=int, default=24)
    parser.add_argument("--bo-cycles", type=int, default=140,
                        help="BO safety ceiling; it is never accepted as a physiological endpoint.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Campaign directory must be fresh: {args.output}")
    if args.resistance_nm <= 0 or args.rho_cycles < 3 or args.bo_trials < 2 or args.bo_cycles < args.rho_cycles:
        raise ValueError("positive resistance, rho-cycles >= 3, bo-trials >= 2 and bo-cycles >= rho-cycles are required")

    root = args.output.resolve()
    manifest = {"schema_version": 1, "resistance_nm": args.resistance_nm,
                "rho_cycles": args.rho_cycles, "bo_cycles": args.bo_cycles, "bo_trials": args.bo_trials,
                "protocol": {
                    "rho": "IPOPT/MA57, SX, Radau-5, one-cycle RHO, 30 stimulations/cycle",
                    "rho_bo": "TPE over three centred log relative weights; Biceps is the gauge; 24 trials in batches of 8; RHO only",
                    "fho": "IPOPT/MA57, MX, exact Hessian; enabled only after its own certified RHO prefix",
                    "validation": "solver success plus feasibility audit; BO never certifies an FHO",
                }, "cases": []}
    for label, factor in CASES:
        case = root / label
        model = case / "model.json"
        weights = case / "unit-weights.json"
        bo = case / "rho-bo.json"
        values = parameters(factor)
        direction = "more endurant" if factor > 1 else "less endurant"
        write_new(model, {"schema_version": 1, "case_id": f"ding_global_{label}_v1",
                          "provenance": (
                              f"Global all-muscle Ding sensitivity ({direction}, {abs(factor - 1) * 100:g}%). "
                              "Fmax/a_scale scaled together; |alpha_a| and tau_fat changed coherently. "
                              "Experimental sensitivity, not a clinical calibration."),
                          "muscles": values})
        write_new(weights, {"initial_weight_basis": "unit reference; no FHO trajectory or BO data",
                            "initial_weights": {name: 1.0 for name in MUSCLES},
                            # The runner requires a named policy, but the BO compares
                            # static weight vectors.  Scheduling its first update beyond
                            # the safety horizon keeps every candidate strictly fixed.
                            "policy": {"update_every_cycles": args.bo_cycles + 1, "smoothing": .2, "capacity_gain": 1.0,
                                       "min_relative_weight": .25, "max_relative_weight": 4.0,
                                       "max_log_step": .0953101798043249, "max_cycles": args.bo_cycles,
                                       "adaptation_strategy": "capacity_feedback"}})
        base = {"mode": "rho-physio", "solver": "ipopt", "mechanics": "reduced",
                "formulation": "dynamic", "cycles": args.bo_cycles, "cycles_per_window": 1,
                "stimulations_per_cycle": 30, "signed_crank_torque": args.resistance_nm,
                "integration": "radau", "collocation_degree": 5, "ipopt_enforce_start_constraints": True,
                "ipopt_linear_solver": "ma57", "threads": 1, "numeric_threads": 1,
                # rho-physio uses SX evaluators; its managed validation guards
                # are incompatible with the generic compiled evaluator switch.
                "compile_evaluators": False, "compact_rho_output": True,
                "reduced_internal_crank_velocity_guard": "on",
                "reduced_terminal_half_step_velocity_guard": True,
                "model_config": str(model), "weights_config": str(weights),
                "output_root": str(case / "rho-bo"),
                "extra_arguments": []}
        search = {f"muscle_weight_coordinate__{name}": {"type": "float", "low": -1.0, "high": 1.0}
                  for name in MUSCLES if name != "Biceps"}
        baseline = {name: .0 for name in search}
        write_new(bo, {"base_config": base, "output_root": str(case / "rho-bo"),
                       "study_name": f"rho-weight-bo-{label}", "study_kind": "controller",
                       "phase": "screening", "max_cycles": args.bo_cycles,
                       "metric": "continuous_endurance", "n_trials": args.bo_trials, "workers": 8,
                       "n_startup_trials": min(4, args.bo_trials - 1), "seed": 20260929,
                       "timeout_s": 21600, "stop_grace_s": 120.0, "search_space": search,
                       "enqueue_trials": [baseline]})
        manifest["cases"].append({"id": label, "factor": factor, "model_config": str(model),
                                  "unit_weights": str(weights), "rho_bo": str(bo),
                                  "rho_bootstrap": str(case / "rho-bootstrap"),
                                  "fho_status": "blocked_on_variant_specific_rho_prefix"})
    write_new(root / "manifest.json", manifest)
    print(root / "manifest.json")


if __name__ == "__main__":
    main()
