#!/usr/bin/env python3
"""Validate frozen-policy endurance rollouts against later RHO cycles only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
# This audit CLI creates no figures and must remain usable on headless workers.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cocofest-matplotlib")

from cocofest.optimization.rho_retrospective_validation import (
    DEFAULT_ANCHOR_STRIDE,
    DEFAULT_OBSERVED_PULSE_WIDTH_TOLERANCE_S,
    DEFAULT_SCIENTIFIC_CRITERIA,
    DEFAULT_SATURATION_THRESHOLD,
    REPORT_SCHEMA,
    build_rho_retrospective_validation_report,
    write_rho_retrospective_validation,
)
from cocofest.optimization.rho_rollout_adapter import DEFAULT_HORIZONS


def _positive_csv(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item < 1 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be unique positive integers")
    return values


def _anchors(value: str) -> tuple[int, ...] | str:
    if value.strip().lower() == "all":
        return "all"
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("anchors must be 'all' or comma-separated integers") from error
    if not values or any(item < 0 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("anchors must be unique non-negative integers")
    return values


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Concatenated certified RHO NPZ.")
    parser.add_argument("--reduced-profile", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument(
        "--anchors",
        type=_anchors,
        default=None,
        help="Zero-based anchor indices, or 'all'. Default: every --anchor-stride cycles.",
    )
    parser.add_argument("--anchor-stride", type=int, default=DEFAULT_ANCHOR_STRIDE)
    parser.add_argument("--horizons", type=_positive_csv, default=DEFAULT_HORIZONS)
    parser.add_argument("--saturation-threshold", type=float, default=DEFAULT_SATURATION_THRESHOLD)
    parser.add_argument(
        "--observed-pulse-width-tolerance-s",
        type=float,
        default=DEFAULT_OBSERVED_PULSE_WIDTH_TOLERANCE_S,
        help="At most 1e-8 s; accepts only solver-scale bound noise and never clips values.",
    )
    parser.add_argument(
        "--minimum-valid-anchors-per-horizon",
        type=int,
        default=DEFAULT_SCIENTIFIC_CRITERIA["minimum_valid_anchors_per_horizon"],
    )
    parser.add_argument(
        "--maximum-state-rest-normalized-rmse",
        type=float,
        default=0.05,
        help="Common prespecified RMSE limit for A, Tau1 and Km, normalized by rest value.",
    )
    parser.add_argument(
        "--maximum-utilization-rmse",
        type=float,
        default=DEFAULT_SCIENTIFIC_CRITERIA["maximum_utilization_rmse"],
    )
    parser.add_argument(
        "--maximum-policy-drift-normalized-pw-rmse",
        type=float,
        default=DEFAULT_SCIENTIFIC_CRITERIA["maximum_policy_drift_normalized_pw_rmse"],
    )
    parser.add_argument(
        "--maximum-policy-drift-normalized-pw-p95",
        type=float,
        default=DEFAULT_SCIENTIFIC_CRITERIA["maximum_policy_drift_normalized_pw_p95"],
    )
    parser.add_argument(
        "--minimum-critical-muscle-match-fraction",
        type=float,
        default=DEFAULT_SCIENTIFIC_CRITERIA["minimum_critical_muscle_match_fraction"],
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--cycle-period", type=float)
    parser.add_argument("--force-harmonics", type=int, default=12)
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
        default=1e-4,
        help="Maximum force-equation defect at any enforced collocation stage.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_cli().parse_args(argv)
    adapter_options = {
        "cycle_period": args.cycle_period,
        "force_harmonics": args.force_harmonics,
        "cn_harmonics": args.cn_harmonics,
        "kinematic_harmonics": args.kinematic_harmonics,
        "periodicity_relative_tolerance": args.periodicity_relative_tolerance,
        "force_seam_absolute_tolerance_n": args.force_seam_absolute_tolerance_n,
        "cn_seam_absolute_tolerance": args.cn_seam_absolute_tolerance,
        "fourier_relative_rmse_tolerance": args.fourier_relative_rmse_tolerance,
        "fourier_relative_maximum_tolerance": args.fourier_relative_maximum_tolerance,
        "kinematic_consistency_tolerance": args.kinematic_consistency_tolerance,
        "negative_force_tolerance": args.negative_force_tolerance,
        "pulse_width_bound_tolerance_s": args.pulse_width_bound_tolerance_s,
        "collocation_force_ode_tolerance_n_per_s": (
            args.collocation_force_ode_tolerance_n_per_s
        ),
        "model_path": args.model_path,
    }
    adapter_options = {key: value for key, value in adapter_options.items() if value is not None}
    try:
        report = build_rho_retrospective_validation_report(
            args.source,
            args.reduced_profile,
            horizons=args.horizons,
            anchors=args.anchors,
            anchor_stride=args.anchor_stride,
            saturation_threshold=args.saturation_threshold,
            observed_pulse_width_tolerance_s=args.observed_pulse_width_tolerance_s,
            scientific_criteria={
                "minimum_valid_anchors_per_horizon": (
                    args.minimum_valid_anchors_per_horizon
                ),
                "maximum_A_rest_normalized_rmse": (
                    args.maximum_state_rest_normalized_rmse
                ),
                "maximum_Tau1_rest_normalized_rmse": (
                    args.maximum_state_rest_normalized_rmse
                ),
                "maximum_Km_rest_normalized_rmse": (
                    args.maximum_state_rest_normalized_rmse
                ),
                "maximum_utilization_rmse": args.maximum_utilization_rmse,
                "maximum_policy_drift_normalized_pw_rmse": (
                    args.maximum_policy_drift_normalized_pw_rmse
                ),
                "maximum_policy_drift_normalized_pw_p95": (
                    args.maximum_policy_drift_normalized_pw_p95
                ),
                "minimum_critical_muscle_match_fraction": (
                    args.minimum_critical_muscle_match_fraction
                ),
            },
            adapter_options=adapter_options,
        )
    except Exception as error:
        report = {
            "schema": REPORT_SCHEMA,
            "processing_status": "invalid_input",
            "scientific_validation_status": "not_evaluable",
            "data_policy": "rho_only_no_full_horizon_data",
            "error_type": type(error).__name__,
            "error": str(error),
            "records": [],
        }
    write_rho_retrospective_validation(args.json_output, args.csv_output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return (
        0
        if report["processing_status"] in {"complete", "complete_with_censoring"}
        and report["scientific_validation_status"] == "passed"
        else 2
    )


if __name__ == "__main__":
    sys.exit(main())
