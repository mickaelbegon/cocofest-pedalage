#!/usr/bin/env python3
"""Compare PACE-VR acceleration on an archived, certified numerical snapshot.

The reference uses the original sequential exponential map and all force
constraints. Both paths keep the same eight substeps, QP and work tolerance.
An optional independent DOP853 replay uses the *same optimized PW schedules*
and the same sampled geometry; it measures existing surrogate error as well
as checking that acceleration has not hidden it. No RHO or FHO is launched.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import MethodType

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.adaptive_moment_rollout import (
    DingPulseWidthParameters, MomentTrackingInterval, _four_state_rhs, periodic_calcium_state,
)
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.pace_vr import PaceVrConfig, PaceVrSupervisor, _SampledGain


def load_case(path, source_cycle):
    document = json.loads(Path(path).read_text())
    if "cycles" in document:
        matches = [c for c in document["cycles"] if c.get("cycle") == source_cycle]
        if len(matches) != 1 or matches[0].get("certified") is not True:
            raise ValueError("The selected cycle must be uniquely identified and certified.")
        snapshot = matches[0].get("pace_vr_snapshot")
    else:
        snapshot = document
    if not snapshot or "payload_json" not in snapshot:
        raise ValueError("No PACE-VR snapshot at the selected cycle.")
    payload = json.loads(snapshot["payload_json"])
    context = payload["context"]
    digest = sha256(json.dumps(context, allow_nan=False, sort_keys=True).encode()).hexdigest()
    if digest != snapshot["context_digest"]:
        raise ValueError("Snapshot digest mismatch.")
    parameters = tuple(DingPulseWidthParameters(
        **{**entry, "fatigue": DingFatigueParameters(**entry["fatigue"])})
        for entry in context["pulse_width_parameters"])
    intervals = tuple(MomentTrackingInterval(
        entry["duration"], tuple(entry["calcium_amplitudes"]),
        tuple(_SampledGain(tuple(samples), entry["duration"])
              for samples in entry["mechanical_gain_samples"]),
        tuple(entry["moment_coefficients"]), tuple(entry["target_moments"]))
        for entry in context["intervals"])
    supervisor = PaceVrSupervisor(
        intervals=intervals, pulse_width_parameters=parameters,
        config=PaceVrConfig(**context["config"]),
        **{key: context[key] for key in ("angular_velocity_rad_s", "required_work_j",
                                        "nonmuscle_power_w", "power_lower_w", "power_upper_w")})
    return supervisor, payload, snapshot


def state_errors(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    error = np.max(np.abs(actual - reference), axis=tuple(range(actual.ndim - 1)))
    scale = np.max(np.abs(reference), axis=tuple(range(reference.ndim - 1)))
    return {name: {"absolute_max": float(err), "normalized_by_reference_peak": float(err / max(s, 1e-15))}
            for name, err, s in zip(("Cn", "F", "A", "Tau1", "Km"), error, scale)}


def fixed_schedule_replay(supervisor, states, schedules, *, reference=False, dop853=False):
    state = np.asarray(states, float).copy()
    history, work, force_trace = [state.copy()], [], []
    started = perf_counter()
    for widths in np.asarray(schedules):
        if not dop853:
            rec = -np.expm1(-(widths - supervisor.predictor.pd0[:, None]) / supervisor.predictor.pdt[:, None])
            model = (supervisor._cycle_reference if reference else supervisor._cycle)(state, rec, linearize=False)
            state = model["state"]
            cycle_work = model["work"]
            force_trace.extend(force for force, _ in model["force_rows"])
        else:
            cycle_work = 0.
            for phase, interval in enumerate(supervisor.intervals):
                times = np.arange(1, supervisor.predictor.substeps + 1) * interval.duration / supervisor.predictor.substeps
                phase_forces = np.empty((supervisor.predictor.substeps, supervisor.muscles))
                for muscle, parameter in enumerate(supervisor.parameters):
                    cn0 = state[muscle, 0]

                    def rhs(time, value):
                        return np.r_[_four_state_rhs(
                            time, value[:4], initial_cn=cn0,
                            pulse_width=widths[muscle, phase],
                            calcium_amplitude=interval.calcium_amplitudes[muscle],
                            mechanical_gain=interval.mechanical_gains[muscle], parameters=parameter), value[0]]

                    integrated = solve_ivp(rhs, (0., interval.duration), np.r_[state[muscle, 1:], 0.],
                                           method="DOP853", rtol=1e-10, atol=1e-11, t_eval=times)
                    if not integrated.success:
                        raise RuntimeError(integrated.message)
                    endpoint = integrated.y[:, -1]
                    phase_forces[:, muscle] = integrated.y[0]
                    state[muscle, 0] = periodic_calcium_state(
                        cn0, interval.duration, interval.calcium_amplitudes[muscle], parameter.tauc)
                    state[muscle, 1:] = endpoint[:4]
                    cycle_work += supervisor.power_coefficients[phase, muscle] * endpoint[4]
                cycle_work += supervisor.nonmuscle_power[phase] * interval.duration
                force_trace.extend(phase_forces)
        work.append(float(cycle_work))
        history.append(state.copy())
    return {"state_history": np.asarray(history), "force_substep_trace": np.asarray(force_trace),
            "work_per_cycle_j": np.asarray(work),
            "runtime_s": perf_counter() - started}


def benchmark(supervisor, payload, *, repeats=1, fit=True, dop853_cycles=0):
    states, widths = np.asarray(payload["initial_states"]), np.asarray(payload["pulse_widths"])
    config = supervisor.config
    outcomes, timing = {}, {}
    for mode in ("reference", "optimized"):
        supervisor.config = replace(config, reduce_redundant_force_constraints=mode == "optimized")
        supervisor._cycle = MethodType(
            PaceVrSupervisor._cycle_reference if mode == "reference" else PaceVrSupervisor._cycle,
            supervisor)
        durations = []
        for _ in range(repeats):
            outcome = (supervisor.fit_terminal if fit else supervisor.evaluate)(
                states, widths, certified=True, weights=payload.get("weights"))
            durations.append(outcome["runtime_s"])
        outcomes[mode], timing[mode] = outcome, durations
    supervisor.config = config
    supervisor._cycle = MethodType(PaceVrSupervisor._cycle, supervisor)
    reference, optimized = outcomes["reference"], outcomes["optimized"]
    common = min(reference["feasible_prefix_cycles"], optimized["feasible_prefix_cycles"])
    comparable = (common > 0 and reference["feasible_prefix_cycles"] == optimized["feasible_prefix_cycles"]
                  and reference["status"] == optimized["status"]
                  and (reference["local_fit"] or {}).get("sample_count")
                      == (optimized["local_fit"] or {}).get("sample_count"))
    # Fixed-schedule replay isolates floating-point arithmetic error from any
    # active-set/tolerance-dependent changes in the reoptimized QP schedules.
    same_pw_reference = fixed_schedule_replay(supervisor, states, optimized["pulse_widths"], reference=True)
    same_pw_fast = fixed_schedule_replay(supervisor, states, optimized["pulse_widths"])
    report = dict(
        benchmark="PACE-VR identical discretization acceleration", fit_included=fit,
        horizon_cycles=config.horizon_cycles, muscles=supervisor.muscles, phases=supervisor.phases,
        substeps=config.integration_substeps, cpus=sorted(os.sched_getaffinity(0)),
        timing_seconds=timing,
        speedup=float(np.median(timing["reference"]) / np.median(timing["optimized"])) if comparable else None,
        timing_comparison_valid=comparable, common_feasible_prefix_cycles=common,
        trajectory_comparison_available=common > 0,
        reference_scope="Sequential map and all F>=0 rows; same current SQP/LP driver as optimized",
        prefixes={mode: outcome["feasible_prefix_cycles"] for mode, outcome in outcomes.items()},
        statuses={mode: outcome["status"] for mode, outcome in outcomes.items()},
        first_failures={mode: outcome["first_failure"] for mode, outcome in outcomes.items()},
        reallocation={mode: outcome["reallocation"] for mode, outcome in outcomes.items()},
        fits={mode: outcome["local_fit"] for mode, outcome in outcomes.items()},
        work_residual_max={mode: outcome["work_residual_max"] for mode, outcome in outcomes.items()},
        constraint_violation_max={mode: outcome["constraint_violation_max"] for mode, outcome in outcomes.items()},
        task_margins={mode: outcome["minimum_task_margin"] for mode, outcome in outcomes.items()},
        reoptimized_state_error=state_errors(np.asarray(optimized["state_history"])[:common + 1],
                                             np.asarray(reference["state_history"])[:common + 1]),
        same_pw_state_error=state_errors(same_pw_fast["state_history"], same_pw_reference["state_history"]),
        same_pw_work_error_max=float(np.max(np.abs(same_pw_fast["work_per_cycle_j"]
                                                   - same_pw_reference["work_per_cycle_j"]), initial=0.)),
        same_pw_force_substep_error_max=float(np.max(np.abs(same_pw_fast["force_substep_trace"]
                                                            - same_pw_reference["force_substep_trace"]), initial=0.)),
        physiology_failure_certified=False,
    )
    if dop853_cycles:
        count = min(dop853_cycles, optimized["feasible_prefix_cycles"])
        dop = fixed_schedule_replay(supervisor, states, optimized["pulse_widths"][:count], dop853=True)
        trace_count = count * supervisor.phases * supervisor.predictor.substeps
        force_error = same_pw_fast["force_substep_trace"][:trace_count] - dop["force_substep_trace"]
        force_peak = float(np.max(np.abs(dop["force_substep_trace"]), initial=0.))
        force_by_muscle = np.max(np.abs(force_error), axis=0, initial=0.)
        peak_by_muscle = np.max(np.abs(dop["force_substep_trace"]), axis=0, initial=0.)
        report["dop853"] = dict(
            cycles=count, runtime_s=dop["runtime_s"], geometry="same sampled geometry as PACE-VR",
            state_error=state_errors(same_pw_fast["state_history"][:count + 1], dop["state_history"]),
            state_sampling="cycle boundaries; force additionally checked at every exponential substep",
            force_substep_samples=trace_count,
            force_substep_absolute_error_max=float(np.max(np.abs(force_error), initial=0.)),
            force_substep_peak_normalized_error=float(np.max(np.abs(force_error), initial=0.) / max(force_peak, 1e-15)),
            force_substep_error_by_muscle=[dict(index=index, absolute_max=float(error),
                normalized_by_own_peak=float(error / max(peak, 1e-15)))
                for index, (error, peak) in enumerate(zip(force_by_muscle, peak_by_muscle))],
            state_error_by_muscle=[state_errors(same_pw_fast["state_history"][:count + 1, muscle:muscle + 1],
                                                dop["state_history"][:, muscle:muscle + 1])
                                   for muscle in range(supervisor.muscles)],
            work_error_max_j=float(np.max(np.abs(same_pw_fast["work_per_cycle_j"][:count]
                                                  - dop["work_per_cycle_j"]), initial=0.)),
            required_work_relative_error_max=float(np.max(
                np.abs(dop["work_per_cycle_j"] - supervisor.required_work_j), initial=0.) / supervisor.required_work_j))
    return report, outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Snapshot JSON or one arm's result.json")
    parser.add_argument("--source-cycle", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--dop853-cycles", type=int, default=0)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if args.repeats < 1 or args.dop853_cycles < 0:
        parser.error("repeats must be positive and dop853-cycles nonnegative")
    if args.cpu is not None:
        if args.cpu not in os.sched_getaffinity(0):
            parser.error("Requested CPU is not available in this process's affinity")
        os.sched_setaffinity(0, {args.cpu})
    supervisor, payload, snapshot = load_case(args.source, args.source_cycle)
    report, outcomes = benchmark(supervisor, payload, repeats=args.repeats,
                                fit=not args.rollout_only, dop853_cycles=args.dop853_cycles)
    report.update(source=str(args.source.resolve()), source_cycle=snapshot["source_cycle"],
                  source_context_digest=snapshot["context_digest"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("timing_seconds", "speedup", "prefixes", "statuses",
                                                   "work_residual_max", "same_pw_state_error")}, indent=2))


if __name__ == "__main__":
    main()
