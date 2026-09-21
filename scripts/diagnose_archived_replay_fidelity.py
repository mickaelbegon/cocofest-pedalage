"""Separate local transcription defects from accumulated open-loop replay drift.

Local one-interval checks restart from each archived shooting state solely to
diagnose the archive. Continuous runs never reset states. All integrators use
the same simultaneously coupled Ding/mechanical vector field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from cocofest.dynamics.reduced_cycling import ReducedCyclingDynamics
from cocofest.optimization.solver_cross_rollout import (
    _physical_rhs, collocation_tableau, evaluate_rollout, file_stamp, implicit_step, load_source,
)


def infer_archive_alpha_a(source):
    """Infer alpha_A from saved Radau stages, conditional on A_rest/tau_fat.

    This is a consistency diagnostic, not permission to substitute a parameter.
    The linear fatigue equation permits an exact collocation quadrature check.
    """
    if source.provenance.get("state_stride") != 6:
        return {"supported": False, "reason": "Requires five-stage Radau archive."}
    _, _, weights, _ = collocation_tableau("radau5")
    result = {"supported": True, "basis": "conditional_on_declared_a_scale_and_tau_fat", "muscles": {}}
    with np.load(source.provenance["source"]["path"], allow_pickle=False) as archive:
        start = source.cycle_start*source.intervals_per_cycle
        intervals = np.arange(start, start+source.controls.shape[1])
        columns = intervals[:, None]*6 + np.arange(1, 6)[None, :]
        for muscle in source.muscles:
            capacity = np.asarray(archive[f"states__A_{muscle}"]).reshape(-1)
            force = np.asarray(archive[f"states__F_{muscle}"]).reshape(-1)
            p = source.parameters[muscle]
            integrated_force = source.dt*(force[columns] @ weights)
            integrated_capacity = source.dt*(capacity[columns] @ weights)
            lhs = (capacity[(intervals+1)*6]-capacity[intervals*6]
                   +(integrated_capacity-p["a_scale"]*source.dt)/p["tau_fat"])
            denominator = float(integrated_force @ integrated_force)
            inferred = None if denominator <= 1e-20 else float(integrated_force @ lhs/denominator)
            result["muscles"][muscle] = {
                "declared_alpha_a": p["alpha_a"], "inferred_alpha_a": inferred,
                "maximum_capacity_equation_defect_declared": float(np.max(np.abs(lhs-p["alpha_a"]*integrated_force))),
                "maximum_capacity_equation_defect_inferred": None if inferred is None else float(np.max(np.abs(lhs-inferred*integrated_force))),
            }
    return result


def diagnose(source, profile, *, continuous=True):
    scales = np.asarray([
        value for name in source.muscles
        for value in (1., source.parameters[name]["Fmax"], source.parameters[name]["a_scale"],
                      source.parameters[name]["tau1_rest"], source.parameters[name]["km_rest"])
    ] + [2*np.pi, 2*np.pi])
    local = {method: [] for method in ("dop853", "radau5", "radau5_half_step")}
    for index, control in enumerate(source.controls.T):
        rhs = _physical_rhs(source, profile, control)
        initial = source.shooting_states[:, index]
        expected = source.shooting_states[:, index+1]
        dop = solve_ivp(rhs, (0., source.dt), initial, method="DOP853",
                        rtol=1e-11, atol=1e-13*scales)
        if not dop.success:
            raise RuntimeError(dop.message)
        local["dop853"].append(dop.y[:, -1]-expected)
        radau, *_ = implicit_step(rhs, initial, 0., source.dt, "radau5", scales)
        local["radau5"].append(radau-expected)
        half, *_ = implicit_step(rhs, initial, 0., source.dt/2, "radau5", scales)
        half, *_ = implicit_step(rhs, half, source.dt/2, source.dt/2, "radau5", scales)
        local["radau5_half_step"].append(half-expected)
    result = {
        "scope": "archive_fidelity_diagnostic_not_closed_loop_validation",
        "provenance": source.provenance,
        "cycle_start": source.cycle_start,
        "cycles": source.cycles,
        "duration_per_cycle_s": source.duration,
        "local_one_interval_defects": {},
    }
    for method, rows in local.items():
        errors = np.asarray(rows)
        result["local_one_interval_defects"][method] = {
            "maximum_absolute_by_state": dict(zip(source.state_names, np.abs(errors).max(axis=0).tolist())),
            "rms_by_state": dict(zip(source.state_names, np.sqrt(np.mean(errors**2, axis=0)).tolist())),
            "maximum_scaled": float(np.max(np.abs(errors)/scales)),
        }
    differences = np.asarray(local["radau5_half_step"])-np.asarray(local["dop853"])
    result["local_half_step_radau_vs_dop853"] = dict(zip(source.state_names, np.abs(differences).max(axis=0).tolist()))
    if continuous:
        result["continuous_open_loop"] = {}
        for method in ("dop853", "radau5"):
            metrics, _ = evaluate_rollout(source, profile, evaluator=method, samples_per_interval=3)
            result["continuous_open_loop"][method] = metrics
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reduced-profile", type=Path, required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--legacy-formulation", choices=("dynamic", "isokinetic"))
    parser.add_argument("--cycle-start", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--cycle-duration", type=float)
    parser.add_argument("--archive-equations-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = ReducedCyclingDynamics.load(args.reduced_profile)
    source = load_source(args.source, profile, model_config=args.model_config,
                         formulation_override=args.legacy_formulation,
                         cycle_duration=args.cycle_duration,
                         cycle_start=args.cycle_start, cycles=args.cycles)
    report = ({"scope": "archive_fatigue_equation_diagnostic", "provenance": source.provenance}
              if args.archive_equations_only else diagnose(source, profile))
    report["archive_alpha_a_consistency"] = infer_archive_alpha_a(source)
    report["reduced_profile"] = file_stamp(args.reduced_profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
