#!/usr/bin/env python3
"""Run a synthetic numerical/latency audit of the experimental Physio-U kernel.

This is not an OCP, RHO, or endurance campaign. The generated envelopes are
synthetic and must not be used as evidence of physiological efficacy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.physio_update import (
    PhysioUpdateConfig,
    build_isokinetic_max_pw_envelope,
    propose_physio_update,
)


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def audit(repeats=200):
    phases = 30
    angles = np.linspace(0, 2 * np.pi, phases, endpoint=False)
    # Explicitly synthetic force and work envelopes, no biological fit.
    envelope = 40 * (1.2 + np.sin(angles[None, :] + np.arange(4)[:, None] * np.pi / 2))
    rest = np.array([1000., 1200., 1800., 900.])
    args = dict(certified=True, rest_capacity=rest,
                alpha_a=np.array([-.1, -.12, -.08, -.06]), tau_fat=np.array([100., 110., 130., 90.]),
                phase_durations=np.full(phases, 1 / phases), required_active_work=1.)
    campaigns = {}
    for cadence in (10, 20):
        config = PhysioUpdateConfig(update_every_cycles=cadence)
        weights = np.ones(4)
        events = []
        for cycle in range(41):
            ratios = 1 - cycle * np.array([.008, .004, .006, .01])
            available = envelope * ratios[:, None]
            event = propose_physio_update(
                **args, completed_cycles=cycle, incumbent_weights=weights,
                current_capacity=rest * ratios, reference_force=.8 * available,
                available_positive_power=.01 * available, config=config)
            if event["status"] == "proposed":
                weights = event["weights"]
            events.append(event)
        campaigns[f"update_{cadence}"] = {"config": asdict(config), "events": events}
    latencies = []
    for _ in range(repeats):
        start = perf_counter()
        propose_physio_update(
            **args, completed_cycles=20, incumbent_weights=np.ones(4), current_capacity=rest * .7,
            reference_force=.8 * envelope, available_positive_power=.01 * envelope)
        latencies.append(perf_counter() - start)
    # This is deliberately a synthetic latency-only input.  It measures the
    # full five-state PW-max envelope plus the proposal, but not an OCP solve
    # or any claim about a muscle's clinical available force.
    parameters = tuple(DingPulseWidthParameters(
        fatigue=DingFatigueParameters(a_rest=value, tau1_rest=.06, km_rest=.13,
                                      alpha_a=alpha, alpha_tau1=2e-5,
                                      alpha_km=2e-5, tau_fat=tau),
        tauc=.011, tau2=.001, pd0=.0001, pdt=.0002, pulse_width_max=.0006,
    ) for value, alpha, tau in zip(rest, args["alpha_a"], args["tau_fat"], strict=True))
    intervals = tuple(MomentTrackingInterval(
        duration=1 / phases,
        calcium_amplitudes=tuple(1. + .1 * np.sin(angle + np.arange(4))),
        mechanical_gains=tuple(.9 + .03 * np.cos(angle + np.arange(4))),
        moment_coefficients=tuple(.02 + .005 * np.cos(angle + np.arange(4))),
        target_moments=(0.,) * 4,
    ) for angle in angles)
    terminal = np.column_stack((np.full(4, .14), 20 + np.arange(4), rest * .7,
                                np.full(4, .063), np.full(4, .14)))
    reference_force = np.maximum(.1, .5 * envelope)
    envelope_latencies = []
    for _ in range(repeats):
        start = perf_counter()
        live = build_isokinetic_max_pw_envelope(
            terminal_states=terminal, reference_force=reference_force, intervals=intervals,
            pulse_width_parameters=parameters, angular_velocity_rad_s=-2 * np.pi,
        )
        propose_physio_update(completed_cycles=20, certified=True, incumbent_weights=np.ones(4),
                              config=PhysioUpdateConfig(update_every_cycles=20), **{
                                  key: live[key] for key in (
                                      "current_capacity", "rest_capacity", "alpha_a", "tau_fat",
                                      "reference_force", "available_positive_power", "phase_durations",
                                      "required_active_work",
                                  )
                              })
        envelope_latencies.append(perf_counter() - start)
    return {
        "kind": "synthetic_kernel_audit_only", "physical_rho_runs": 0,
        "force_envelope_validated": False, "endurance_improvement_validated": False,
        "phase_count": phases, "muscle_count": 4, "timing_repeats": repeats,
        "median_update_time_s": float(np.median(latencies)),
        "p95_update_time_s": float(np.percentile(latencies, 95)),
        "max_update_time_s": float(np.max(latencies)),
        "synthetic_isokinetic_full_envelope_median_time_s": float(np.median(envelope_latencies)),
        "synthetic_isokinetic_full_envelope_p95_time_s": float(np.percentile(envelope_latencies, 95)),
        "synthetic_isokinetic_full_envelope_max_time_s": float(np.max(envelope_latencies)),
        "synthetic_isokinetic_full_envelope_uses_ocp": False,
        "campaigns": campaigns,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.output.exists():
        parser.error("--output already exists; choose a new audit file")
    result = audit(args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, default=_json, allow_nan=False, indent=2)
    print(json.dumps({key: result[key] for key in (
        "kind", "physical_rho_runs", "median_update_time_s", "p95_update_time_s")}, indent=2))


if __name__ == "__main__":
    main()
