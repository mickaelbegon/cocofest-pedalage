#!/usr/bin/env python3
"""Adapt a certified cycling RHO NPZ export to H=5/10/20 Ding rollouts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Headless-safe before importing the package graph (which may import plotting
# dependencies through Bioptim in clinical/CI environments).
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cocofest-matplotlib")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cocofest.optimization.rho_rollout_adapter import (
    DEFAULT_COLLOCATION_FORCE_ODE_TOLERANCE_N_PER_S,
    DEFAULT_HORIZONS,
    DEFAULT_MIDPOINT_INVERSE_COVERAGE_MINIMUM,
    DEFAULT_MIDPOINT_FORCE_ODE_TOLERANCE_N_PER_S,
    DEFAULT_MIDPOINT_PULSE_WIDTH_ERROR_TOLERANCE_S,
    REPORT_SCHEMA,
    build_rho_endurance_rollout_report,
    write_rho_endurance_rollout_report,
)


def _horizons(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("horizons must be comma-separated integers.") from error
    if not values or any(item < 1 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("horizons must contain unique positive integers.")
    return values


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Certified RHO .npz export.")
    parser.add_argument("--reduced-profile", type=Path, required=True, help="Reduced cycling Fourier cache.")
    parser.add_argument("--output", type=Path, required=True, help="Compact JSON output path.")
    parser.add_argument("--model-path", type=Path, help="Source bioMod; defaults to the Wu cycling model.")
    parser.add_argument("--horizons", type=_horizons, default=DEFAULT_HORIZONS)
    parser.add_argument("--cycle-period", type=float)
    parser.add_argument(
        "--force-harmonics",
        type=int,
        default=12,
        help="Harmonics fitted to sqrt(F); the exact squared force series has twice this order.",
    )
    parser.add_argument("--cn-harmonics", type=int)
    parser.add_argument("--kinematic-harmonics", type=int, default=12)
    parser.add_argument("--periodicity-relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--force-seam-absolute-tolerance-n", type=float, default=1e-3)
    parser.add_argument("--cn-seam-absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--fourier-relative-rmse-tolerance", type=float, default=5e-2)
    parser.add_argument("--fourier-relative-maximum-tolerance", type=float, default=0.35)
    parser.add_argument("--kinematic-consistency-tolerance", type=float, default=0.1)
    parser.add_argument("--negative-force-tolerance", type=float, default=1e-10)
    parser.add_argument("--pulse-width-bound-tolerance-s", type=float, default=1e-10)
    parser.add_argument(
        "--collocation-force-ode-tolerance-n-per-s",
        type=float,
        default=DEFAULT_COLLOCATION_FORCE_ODE_TOLERANCE_N_PER_S,
        help=(
            "Exact transcription gate: maximum force-equation defect at any "
            "enforced collocation stage (default: 1e-4 N/s)."
        ),
    )
    parser.add_argument(
        "--midpoint-pulse-width-error-tolerance-s",
        type=float,
        default=DEFAULT_MIDPOINT_PULSE_WIDTH_ERROR_TOLERANCE_S,
        help=(
            "Compact-policy quality gate: maximum inferred-vs-exported pulse-width "
            "error among finite midpoint inversions (default: 10 us)."
        ),
    )
    parser.add_argument(
        "--midpoint-force-ode-tolerance-n-per-s",
        type=float,
        default=DEFAULT_MIDPOINT_FORCE_ODE_TOLERANCE_N_PER_S,
        help=(
            "Compact-policy quality gate: maximum Ding ODE residual at non-enforced "
            "interval midpoints (default: 10 N/s)."
        ),
    )
    parser.add_argument(
        "--midpoint-inverse-coverage-minimum",
        type=float,
        default=DEFAULT_MIDPOINT_INVERSE_COVERAGE_MINIMUM,
        help=(
            "Minimum fraction of finite bounded PW inversions at non-enforced "
            "midpoints (default: 0.90; failures remain fully reported)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_cli().parse_args(argv)
    try:
        report = build_rho_endurance_rollout_report(
            args.source,
            args.reduced_profile,
            horizons=args.horizons,
            cycle_period=args.cycle_period,
            force_harmonics=args.force_harmonics,
            cn_harmonics=args.cn_harmonics,
            kinematic_harmonics=args.kinematic_harmonics,
            periodicity_relative_tolerance=args.periodicity_relative_tolerance,
            force_seam_absolute_tolerance_n=args.force_seam_absolute_tolerance_n,
            cn_seam_absolute_tolerance=args.cn_seam_absolute_tolerance,
            fourier_relative_rmse_tolerance=args.fourier_relative_rmse_tolerance,
            fourier_relative_maximum_tolerance=args.fourier_relative_maximum_tolerance,
            kinematic_consistency_tolerance=args.kinematic_consistency_tolerance,
            negative_force_tolerance=args.negative_force_tolerance,
            pulse_width_bound_tolerance_s=args.pulse_width_bound_tolerance_s,
            collocation_force_ode_tolerance_n_per_s=(
                args.collocation_force_ode_tolerance_n_per_s
            ),
            midpoint_pulse_width_error_tolerance_s=(
                args.midpoint_pulse_width_error_tolerance_s
            ),
            midpoint_force_ode_tolerance_n_per_s=(
                args.midpoint_force_ode_tolerance_n_per_s
            ),
            midpoint_inverse_coverage_minimum=args.midpoint_inverse_coverage_minimum,
            model_path=args.model_path,
        )
    except Exception as error:
        report = {
            "schema": REPORT_SCHEMA,
            "status": "invalid_input",
            "error_type": type(error).__name__,
            "error": str(error),
            "horizons": [],
        }
    write_rho_endurance_rollout_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
