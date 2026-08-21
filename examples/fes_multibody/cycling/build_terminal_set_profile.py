#!/usr/bin/env python3
"""Build an auditable mechanical terminal-set profile from certified RHO traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METADATA_KEY = "metadata__json"
STATE_PREFIX = "states__"
FULL_TURN = 2.0 * np.pi


def _load_certified_boundaries(path: Path) -> tuple[dict, list[dict], str, str]:
    with np.load(path, allow_pickle=False) as archive:
        if METADATA_KEY not in archive.files:
            raise ValueError(f"{path} has no trajectory metadata.")
        metadata = json.loads(str(archive[METADATA_KEY].item()))
        if metadata.get("producer_mode") != "receding_horizon_concatenation":
            raise ValueError(f"{path} is not a certified RHO concatenation.")
        stimulations = int(metadata["stimulations_per_cycle"])
        cycles = int(metadata["cycles_per_window"])
        theta = np.asarray(archive[f"{STATE_PREFIX}theta"], dtype=float)
        omega = np.asarray(archive[f"{STATE_PREFIX}omega"], dtype=float)
        capacity_keys = sorted(
            key for key in archive.files if key.startswith(f"{STATE_PREFIX}A_")
        )
        if theta.shape != (1, cycles * stimulations + 1):
            raise ValueError(f"Unexpected theta shape in {path}: {theta.shape}.")
        if omega.shape != theta.shape:
            raise ValueError(f"theta and omega shapes differ in {path}.")
        if not capacity_keys:
            raise ValueError(f"{path} has no Ding A_* capacity states.")
        capacities = {
            key.removeprefix(STATE_PREFIX): np.asarray(archive[key], dtype=float)
            for key in capacity_keys
        }

    boundary_nodes = np.arange(cycles + 1, dtype=int) * stimulations
    theta_boundary = theta[0, boundary_nodes]
    omega_boundary = omega[0, boundary_nodes]
    # The benchmark pedals with omega < 0. A replay checkpoint must retain the
    # original experiment reference and its global cycle index: resetting the
    # target to theta_boundary[0] would hide drift accumulated before replay.
    origin_reference = metadata.get("absolute_wheel_q_origin_reference")
    start_cycle = metadata.get("absolute_wheel_q_start_cycle_index")
    if origin_reference is not None and start_cycle is not None:
        theta_targets = float(origin_reference) - FULL_TURN * (
            int(start_cycle) + np.arange(cycles + 1)
        )
        angle_reference = "global_absolute_cycle"
    else:
        theta_targets = theta_boundary[0] - FULL_TURN * np.arange(cycles + 1)
        angle_reference = "legacy_local_source_diagnostic_only"
    source_initial_capacity = {
        key: float(values.reshape(-1, values.shape[-1])[0, 0])
        for key, values in capacities.items()
    }
    metadata_scales = metadata.get("fatigue_capacity_scales") or {}
    capacity_scales = {
        key: float(metadata_scales[key])
        for key in capacities
        if key in metadata_scales
    }
    if len(capacity_scales) == len(capacities):
        normalization = "rested_ding_a_scale"
    else:
        # Legacy RHO exports did not retain a_scale.  They remain useful for a
        # diagnostic plot, but can never make the profile certification-ready:
        # normalizing each source by its own first boundary would hide fatigue
        # already accumulated during warmup or a replay checkpoint.
        capacity_scales = source_initial_capacity
        normalization = "legacy_source_initial_diagnostic_only"
    if any(
        value <= 0.0 or not np.isfinite(value)
        for value in capacity_scales.values()
    ):
        raise ValueError(f"{path} has invalid initial Ding capacity values.")

    samples = []
    for cycle, node in enumerate(boundary_nodes):
        ratios = []
        ratios_by_state = {}
        for key, values in capacities.items():
            flattened = values.reshape(-1, values.shape[-1])
            state_ratios = flattened[:, node] / capacity_scales[key]
            ratios.extend(state_ratios.tolist())
            ratios_by_state[key] = float(np.min(state_ratios))
        samples.append(
            {
                "cycle": int(cycle),
                "theta_phase_error_rad": float(
                    theta_boundary[cycle] - theta_targets[cycle]
                ),
                "omega_rad_s": float(omega_boundary[cycle]),
                "minimum_capacity_ratio": float(min(ratios)),
                "capacity_ratio_by_state": ratios_by_state,
            }
        )
    return metadata, samples, normalization, angle_reference


def _interval(values: np.ndarray, padding: float) -> dict:
    return {
        "observed_min": float(np.min(values)),
        "observed_max": float(np.max(values)),
        "lower": float(np.min(values) - padding),
        "upper": float(np.max(values) + padding),
        "padding": float(padding),
    }


def build_terminal_set_profile(
    sources: list[Path],
    *,
    capacity_bin_width: float = 0.1,
    theta_padding_rad: float = 0.002,
    omega_padding_rad_s: float = 0.25,
) -> dict:
    """Aggregate boundary envelopes conditioned on load and Ding capacity.

    The result is intentionally not a deployable invariant set. It becomes a
    candidate for an online bound only after at least three distinct loads and
    20 certified boundary samples per populated load/capacity cell.
    """

    if not sources:
        raise ValueError("At least one certified trajectory is required.")
    if not 0.0 < capacity_bin_width <= 1.0:
        raise ValueError("capacity_bin_width must be in (0, 1].")

    rows = []
    source_summaries = []
    for source in sources:
        source = Path(source).resolve()
        metadata, samples, normalization, angle_reference = (
            _load_certified_boundaries(source)
        )
        load_nm = float(metadata["constant_crank_torque"])
        for sample in samples:
            clipped = np.clip(sample["minimum_capacity_ratio"], 0.0, 1.0)
            bin_count = int(np.ceil(1.0 / capacity_bin_width))
            bin_index = min(
                int((1.0 - clipped) / capacity_bin_width), bin_count - 1
            )
            row = dict(sample)
            row.update(
                {
                    "load_nm": load_nm,
                    "capacity_bin_index": bin_index,
                    "source": str(source),
                }
            )
            rows.append(row)
        source_summaries.append(
            {
                "path": str(source),
                "solver": metadata.get("producer_solver"),
                "collocation_degree": metadata.get("producer_collocation_degree"),
                "load_nm": load_nm,
                "certified_cycles": int(metadata["cycles_per_window"]),
                "capacity_normalization": normalization,
                "angle_reference": angle_reference,
            }
        )

    bins = []
    fatigue_state_keys = sorted(
        {key for row in rows for key in row["capacity_ratio_by_state"]}
    )
    grouped_keys = sorted({(row["load_nm"], row["capacity_bin_index"]) for row in rows})
    for load_nm, bin_index in grouped_keys:
        group = [
            row
            for row in rows
            if row["load_nm"] == load_nm and row["capacity_bin_index"] == bin_index
        ]
        capacity_upper = 1.0 - bin_index * capacity_bin_width
        capacity_lower = max(0.0, capacity_upper - capacity_bin_width)
        bins.append(
            {
                "load_nm": load_nm,
                "capacity_ratio_lower": capacity_lower,
                "capacity_ratio_upper": capacity_upper,
                "sample_count": len(group),
                "theta_phase_error_rad": _interval(
                    np.asarray([row["theta_phase_error_rad"] for row in group]),
                    theta_padding_rad,
                ),
                "omega_rad_s": _interval(
                    np.asarray([row["omega_rad_s"] for row in group]),
                    omega_padding_rad_s,
                ),
                "capacity_ratio_by_state": {
                    key: _interval(
                        np.asarray(
                            [row["capacity_ratio_by_state"][key] for row in group]
                        ),
                        0.0,
                    )
                    for key in fatigue_state_keys
                },
            }
        )

    distinct_loads = sorted({row["load_nm"] for row in rows})
    under_sampled = [
        {"load_nm": row["load_nm"], "capacity_ratio_lower": row["capacity_ratio_lower"]}
        for row in bins
        if row["sample_count"] < 20
    ]
    legacy_capacity_sources = [
        row["path"]
        for row in source_summaries
        if row["capacity_normalization"] != "rested_ding_a_scale"
    ]
    legacy_angle_sources = [
        row["path"]
        for row in source_summaries
        if row["angle_reference"] != "global_absolute_cycle"
    ]
    certification_ready = (
        len(distinct_loads) >= 3
        and not under_sampled
        and not legacy_capacity_sources
        and not legacy_angle_sources
    )
    return {
        "schema": "cocofest-mechanical-terminal-set-profile-v1",
        "status": "candidate" if certification_ready else "diagnostic_only",
        "absolute_angle_target": "theta_initial_minus_2pi_times_cycle",
        "conditioning_variables": ["constant_crank_torque", "minimum_A_capacity_ratio"],
        "reported_fatigue_state_ratios": fatigue_state_keys,
        "sources": source_summaries,
        "coverage": {
            "distinct_loads_nm": distinct_loads,
            "boundary_sample_count": len(rows),
            "minimum_required_distinct_loads": 3,
            "minimum_required_samples_per_bin": 20,
            "under_sampled_bins": under_sampled,
            "legacy_capacity_sources": legacy_capacity_sources,
            "legacy_angle_sources": legacy_angle_sources,
            "certification_ready": certification_ready,
        },
        "bins": bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity-bin-width", type=float, default=0.1)
    parser.add_argument("--theta-padding-rad", type=float, default=0.002)
    parser.add_argument("--omega-padding-rad-s", type=float, default=0.25)
    args = parser.parse_args()
    profile = build_terminal_set_profile(
        args.source,
        capacity_bin_width=args.capacity_bin_width,
        theta_padding_rad=args.theta_padding_rad,
        omega_padding_rad_s=args.omega_padding_rad_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    print(json.dumps(profile["coverage"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
