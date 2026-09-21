#!/usr/bin/env python3
"""Benchmark three PW continuation strategies after one optimized cycle.

The script accepts either a certified RHO NPZ plus its reduced-mechanics
profile, or a deterministic synthetic bilateral-like case.  All three methods
start from the same terminal state of the source cycle:

* repeat the source PW profile;
* recover PW values that preserve each individual muscle moment;
* preserve only the required total crank moment and redistribute it between
  muscles through a bounded phase-local allocation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.adaptive_moment_rollout import (  # noqa: E402
    DingPulseWidthParameters,
    MomentTrackingInterval,
    propagate_ding_pulse_width_interval,
    rollout_fixed_pulse_width_policy,
)
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters  # noqa: E402
from cocofest.optimization.moment_viability_preview import (  # noqa: E402
    CompactMomentViabilityPolicy,
    FiveCycleViabilityConfig,
    MechanicalGainDomainError,
    audit_policy_mechanical_gains,
    run_five_cycle_viability_preview,
)
from cocofest.optimization.rho_adaptive_moment_policy import (  # noqa: E402
    build_rho_adaptive_moment_policy,
)
from cocofest.optimization.rho_rollout_adapter import (  # noqa: E402
    _build_actual_muscle_models,
    _default_model_path,
    select_certified_rho_cycle,
)
from cocofest.dynamics.reduced_cycling import ReducedCyclingDynamics  # noqa: E402
from cocofest.models.reduced_cycling_model import duplicate_bilateral_muscles  # noqa: E402


def _synthetic_case(*, muscle_count: int, interval_count: int):
    """Return a deterministic source cycle and its terminal preview state."""

    if muscle_count < 1 or interval_count < 1:
        raise ValueError("muscle_count and interval_count must be positive.")
    parameters = tuple(
        DingPulseWidthParameters(
            fatigue=DingFatigueParameters(
                a_rest=1200.0 + 30.0 * index,
                tau1_rest=0.060601,
                km_rest=0.137,
                alpha_a=-0.08,
                alpha_tau1=2.1e-6,
                alpha_km=1.9e-6,
                tau_fat=127.0,
            ),
            tauc=0.011,
            tau2=0.001,
            pd0=0.000131405,
            pdt=0.000194138,
            pulse_width_max=0.0006,
        )
        for index in range(muscle_count)
    )
    initial = np.asarray(
        [
            [
                0.1629821583533315,
                18.0 + index,
                0.99 * muscle.fatigue.a_rest,
                1.01 * muscle.fatigue.tau1_rest,
                1.01 * muscle.fatigue.km_rest,
            ]
            for index, muscle in enumerate(parameters)
        ]
    )
    phase = 2.0 * np.pi * np.arange(interval_count) / interval_count
    pulse_widths = np.empty((muscle_count, interval_count))
    for muscle_index in range(muscle_count):
        pulse_widths[muscle_index] = (
            0.00033
            + 0.000025 * np.sin(phase + 0.47 * muscle_index)
            + 0.000004 * (muscle_index / max(1, muscle_count - 1) - 0.5)
        )

    duration = 1.0 / interval_count
    amplitudes = tuple(1.0597355478114694 for _ in parameters)
    # First settle the fast states.  This does not remove slow Ding fatigue.
    current = initial.copy()
    for _ in range(3):
        for interval_index in range(interval_count):
            for muscle_index, muscle in enumerate(parameters):
                current[muscle_index] = propagate_ding_pulse_width_interval(
                    current[muscle_index],
                    pulse_width=pulse_widths[muscle_index, interval_index],
                    duration=duration,
                    calcium_amplitude=amplitudes[muscle_index],
                    mechanical_gain=0.94 + 0.04 * np.cos(phase[interval_index] - 0.2 * muscle_index),
                    parameters=muscle,
                    integration_substeps=8,
                )

    source_initial = current.copy()
    target_moments = np.empty_like(pulse_widths)
    intervals = []
    for interval_index in range(interval_count):
        gains = tuple(
            0.94 + 0.04 * np.cos(phase[interval_index] - 0.2 * muscle_index)
            for muscle_index in range(muscle_count)
        )
        coefficients = tuple(
            (0.035 + 0.008 * np.sin(phase[interval_index] + 0.31 * muscle_index))
            * (1.0 if muscle_index % 4 < 2 else -1.0)
            for muscle_index in range(muscle_count)
        )
        for muscle_index, muscle in enumerate(parameters):
            current[muscle_index] = propagate_ding_pulse_width_interval(
                current[muscle_index],
                pulse_width=pulse_widths[muscle_index, interval_index],
                duration=duration,
                calcium_amplitude=amplitudes[muscle_index],
                mechanical_gain=gains[muscle_index],
                parameters=muscle,
                integration_substeps=8,
            )
            target_moments[muscle_index, interval_index] = (
                coefficients[muscle_index] * current[muscle_index, 1]
            )
        intervals.append(
            MomentTrackingInterval(
                duration=duration,
                calcium_amplitudes=amplitudes,
                mechanical_gains=gains,
                moment_coefficients=coefficients,
                target_moments=tuple(target_moments[:, interval_index]),
            )
        )
    policy = CompactMomentViabilityPolicy(
        intervals=tuple(intervals),
        parameters=parameters,
        source_pulse_widths=pulse_widths,
        muscle_names=tuple(f"muscle_{index}" for index in range(muscle_count)),
    )
    return policy, source_initial, current.copy()


def _source_case(args, *, state_roundoff_tolerance: float):
    muscle_models = None
    reduced = ReducedCyclingDynamics.load(args.reduced_profile)
    reduced_names = tuple(str(name) for name in reduced.muscle_names)
    bilateral = bool(reduced_names) and all(
        name.startswith(("right_", "left_")) for name in reduced_names
    )
    if bilateral:
        selected, _ = select_certified_rho_cycle(
            args.source,
            cycle_index=args.cycle_index,
            cycle_period=args.cycle_period,
        )
        muscle_models = duplicate_bilateral_muscles(
            _build_actual_muscle_models(selected, _default_model_path())
        )
    source = build_rho_adaptive_moment_policy(
        args.source,
        args.reduced_profile,
        cycle_index=args.cycle_index,
        cycle_period=args.cycle_period,
        muscle_models=muscle_models,
        reduced_dynamics=reduced,
    )
    policy = CompactMomentViabilityPolicy.from_rho_policy(source)
    source_initial = np.asarray(source.initial_states, dtype=float).copy()
    negative_cn = source_initial[:, 0] < 0.0
    if np.any(source_initial[:, 0] < -state_roundoff_tolerance):
        raise ValueError(
            "The certified source contains Cn below the declared roundoff tolerance."
        )
    source_projection_count = int(np.count_nonzero(negative_cn))
    source_projection_maximum = (
        float(np.max(np.abs(source_initial[negative_cn, 0])))
        if source_projection_count
        else 0.0
    )
    source_initial[negative_cn, 0] = 0.0
    gain_audit = audit_policy_mechanical_gains(
        policy, integration_substeps=args.integration_substeps
    )
    if not gain_audit.valid:
        raise MechanicalGainDomainError(gain_audit)
    source_cycle = rollout_fixed_pulse_width_policy(
        source_initial,
        intervals=source.intervals,
        parameters=source.parameters,
        pulse_widths=policy.source_pulse_widths,
        horizon_cycles=1,
        integration_substeps=args.integration_substeps,
    )
    return (
        policy,
        source_initial,
        source_cycle.state_history[-1],
        {
            "projection_count": source_projection_count,
            "projection_maximum_absolute": source_projection_maximum,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_argument_group("certified RHO input")
    inputs.add_argument("--source", type=Path, help="Certified compact RHO NPZ.")
    inputs.add_argument("--reduced-profile", type=Path, help="Matching reduced mechanics NPZ.")
    inputs.add_argument("--cycle-index", type=int, default=0)
    inputs.add_argument("--cycle-period", type=float)
    synthetic = parser.add_argument_group("synthetic input")
    synthetic.add_argument("--synthetic-muscles", type=int, default=8)
    synthetic.add_argument("--synthetic-intervals", type=int, default=30)
    parser.add_argument("--horizon-cycles", type=int, default=5)
    parser.add_argument("--integration-substeps", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if (args.source is None) != (args.reduced_profile is None):
        raise SystemExit("--source and --reduced-profile must be provided together.")
    config = FiveCycleViabilityConfig(
        horizon_cycles=args.horizon_cycles,
        integration_substeps=args.integration_substeps,
    )
    if args.source is None:
        policy, source_initial, preview_initial = _synthetic_case(
            muscle_count=args.synthetic_muscles,
            interval_count=args.synthetic_intervals,
        )
        source_roundoff_audit = {
            "projection_count": 0,
            "projection_maximum_absolute": 0.0,
        }
        input_kind = "synthetic"
    else:
        try:
            policy, source_initial, preview_initial, source_roundoff_audit = _source_case(
                args,
                state_roundoff_tolerance=config.state_roundoff_tolerance,
            )
        except MechanicalGainDomainError as error:
            payload = {
                "schema": "cocofest-five-cycle-viability-benchmark-v1",
                "input_kind": "certified_rho",
                "configuration": {
                    key: value
                    for key, value in asdict(config).items()
                    if key != "allocation_options"
                },
                "summary": {
                    "status": "unsupported_mechanical_gain_domain",
                    "feasible": False,
                    "reason": str(error),
                    "mechanical_gain_audit": error.audit.summary(),
                    "clipping_applied": False,
                },
            }
            rendered = json.dumps(payload, indent=2, sort_keys=True)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered + "\n", encoding="utf-8")
            print(rendered)
            return 2
        input_kind = "certified_rho"
    result = run_five_cycle_viability_preview(
        policy,
        initial_states=preview_initial,
        config=config,
    )
    payload = {
        "schema": "cocofest-five-cycle-viability-benchmark-v1",
        "input_kind": input_kind,
        "source_initial_state_shape": list(source_initial.shape),
        "preview_initial_state_shape": list(preview_initial.shape),
        "source_initial_cn_roundoff_audit": source_roundoff_audit,
        "configuration": {
            key: value
            for key, value in asdict(config).items()
            if key != "allocation_options"
        },
        "summary": result.summary(),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result.feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())
