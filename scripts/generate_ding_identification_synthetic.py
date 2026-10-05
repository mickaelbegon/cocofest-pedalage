"""Write eight reproducible single-muscle Ding pedal-sensor trials.

This example uses deliberately simple constant geometry. Replace the
``CyclingTrialSpec`` projection arrays with bioMod-derived curves before a
subject-specific study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cocofest.identification.synthetic_cycling import (
    BILATERAL_MUSCLE_NAMES,
    SensorNoise,
    generate_trial,
    simple_mono_muscle_spec,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--frequency-hz", type=float, default=30.0)
    parser.add_argument("--pulse-width-us", type=float, default=300.0)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--sample-rate-hz", type=float, default=200.0)
    parser.add_argument("--two-axis", action="store_true")
    parser.add_argument("--sensor-noise-n", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must name a new directory, so existing data are never overwritten")
    noise = SensorNoise(
        tangential_std_n=args.sensor_noise_n,
        radial_std_n=args.sensor_noise_n if args.two_axis else 0.0,
    )
    specs = [
        simple_mono_muscle_spec(
            muscle,
            frequency_hz=args.frequency_hz,
            pulse_width_s=args.pulse_width_us * 1e-6,
            duration_s=args.duration_s,
            sample_rate_hz=args.sample_rate_hz,
            radial_gain=0.1 if args.two_axis else None,
        )
        for muscle in BILATERAL_MUSCLE_NAMES
    ]
    trials = [generate_trial(spec, noise=noise, seed=args.seed + index)
              for index, spec in enumerate(specs)]
    args.output.mkdir(parents=True)
    for trial in trials:
        spec = trial.spec
        np.savez_compressed(
            args.output / f"{spec.muscle_name}.npz",
            time_s=spec.time_s,
            angle_rad=spec.angle_rad,
            pulse_times_s=spec.pulse_times_s,
            pulse_width_s=spec.pulse_width_s,
            moment_arm_m=spec.moment_arm_m,
            crank_radius_m=np.asarray(spec.crank_radius_m),
            radial_gain=np.asarray([] if spec.radial_gain is None else spec.radial_gain),
            observed=trial.observed,
            noiseless=trial.noiseless,
            state=trial.state,
            noise_std=trial.noise_std,
        )
    metadata = {
        "schema": "cocofest.ding_identification.synthetic_mono_muscle.v1",
        "muscle_names": list(BILATERAL_MUSCLE_NAMES),
        "channels": ["tangential_force_n"] + (["radial_force_n"] if args.two_axis else []),
        "frequency_hz": args.frequency_hz,
        "pulse_width_us": args.pulse_width_us,
        "duration_s": args.duration_s,
        "sample_rate_hz": args.sample_rate_hz,
        "sensor_noise_n": args.sensor_noise_n,
        "seed": args.seed,
        "geometry": "constant demonstration projection; not subject-specific",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(trials)} trials to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
