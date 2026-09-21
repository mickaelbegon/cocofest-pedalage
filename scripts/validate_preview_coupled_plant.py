"""Replay compact-preview PW on the coupled reduced plant.

This is the third validation layer for the five-cycle preview policy.  Unlike
the compact-policy benchmark, it does not freeze theta/omega while applying
future PW: both preview and normal RHO commands are propagated through the
same continuous reduced mechanics and five-state Ding plant.  It remains an
open-loop *plant* comparison because the normal-RHO commands come from an
archive; a later controller validation must solve the next OCP from the
preview-rolled state.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.dynamics.reduced_cycling import ReducedCyclingDynamics
from cocofest.optimization.compact_muscle_prediction import CompactMusclePredictor
from cocofest.optimization.rho_adaptive_moment_policy import build_rho_adaptive_moment_policy
from cocofest.optimization.solver_cross_rollout import evaluate_rollout, load_source
from scripts.benchmark_cycle_preview_controller import preview_anchor_indices, sanitize_initial_calcium


DEFAULT_SOURCE = Path(
    "ipopt-linear-solver-150-20260904/resistance-0p10Nm/"
    "ipopt-sx-radau5-ma57-150-max2000-reduced/validated-rho-trajectory.npz"
)
DEFAULT_PROFILE = Path("benchmark-seed/reduced-cycling-fourier12.npz")
DEFAULT_OUTPUT = Path(".cache/preview-coupled-plant-validation")
SLOW_NAMES = ("A", "Tau1", "Km")


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _state_rows(source, muscles: tuple[str, ...], state_names: tuple[str, ...]) -> np.ndarray:
    """Extract named muscle states from a :class:`RolloutSource` boundary."""

    lookup = {name: index for index, name in enumerate(source.state_names)}
    rows = []
    for muscle in muscles:
        rows.append([source.initial_state[lookup[f"{state}_{muscle}"]] for state in state_names])
    return np.asarray(rows, dtype=float)


def flatten_preview_pulse_widths(pulse_widths: np.ndarray) -> np.ndarray:
    """Convert (cycle, muscle, phase) preview PW to plant control ordering."""

    values = np.asarray(pulse_widths, dtype=float)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("preview pulse widths must be a finite (cycle, muscle, phase) array.")
    return values.transpose(1, 0, 2).reshape(values.shape[1], -1)


def _boundary_slow_states(arrays: dict, source, muscles: tuple[str, ...]) -> np.ndarray:
    """Return A/Tau1/Km at initial and every cycle end from a plant rollout."""

    lookup = {name: index for index, name in enumerate(arrays["state_names"].tolist())}
    nodes = np.asarray(arrays["shooting_states"], dtype=float)
    columns = np.arange(source.cycles + 1) * source.intervals_per_cycle
    values = np.empty((source.cycles + 1, len(muscles), len(SLOW_NAMES)))
    for muscle_index, muscle in enumerate(muscles):
        for state_index, state in enumerate(SLOW_NAMES):
            values[:, muscle_index, state_index] = nodes[lookup[f"{state}_{muscle}"], columns]
    return values


def _slow_comparison(preview: np.ndarray, normal: np.ndarray, parameters: dict, muscles: tuple[str, ...]) -> dict:
    rest = np.asarray([[parameters[muscle]["a_scale"], parameters[muscle]["tau1_rest"], parameters[muscle]["km_rest"]]
                       for muscle in muscles], dtype=float)
    delta = preview / rest[None, :, :] - normal / rest[None, :, :]
    terminal = delta[-1]
    return {
        "initial_boundary_maximum_absolute_normalized_difference": float(np.max(np.abs(delta[0]))),
        "maximum_absolute_normalized_difference": float(np.max(np.abs(delta))),
        "terminal_preview_minus_normal_rho": {
            "mean_capacity_ratio": float(np.mean(terminal[:, 0])),
            "mean_tau1_ratio": float(np.mean(terminal[:, 1])),
            "mean_km_ratio": float(np.mean(terminal[:, 2])),
            "minimum_capacity_ratio": float(np.min(terminal[:, 0])),
            "maximum_tau1_ratio": float(np.max(terminal[:, 1])),
            "maximum_km_ratio": float(np.max(terminal[:, 2])),
        },
    }, delta


def _physical_ok(metrics: dict) -> bool:
    physical = metrics.get("physical_metrics", {})
    return all(float(physical.get(field, 0.0) or 0.0) <= 0.0 for field in (
        "physical_pw_bound_violation_s", "sampled_phase_bound_violation_rad",
        "sampled_velocity_bound_violation_rad_s", "sampled_load_bound_violation_nm",
    ))


def _run_anchor(args, profile, anchor: int) -> tuple[dict, dict[str, np.ndarray]]:
    # The preview starts at end(anchor), which is the incoming state of the
    # immediately following archive cycle.  This makes the plant comparison
    # time-aligned even though the preview holds the anchor reference.
    normal = load_source(
        args.source, profile, cycle_start=anchor + 1, cycles=args.preview_cycles,
        cycle_duration=args.cycle_period, formulation_override=args.legacy_formulation,
        model_config=args.model_config,
    )
    policy = build_rho_adaptive_moment_policy(
        args.source, args.reduced_profile, cycle_index=anchor,
        cycle_period=args.cycle_period, model_path=args.model_path,
    )
    initial = sanitize_initial_calcium(_state_rows(normal, policy.muscle_names, ("Cn", "F", "A", "Tau1", "Km")))
    predictor = CompactMusclePredictor(policy.intervals, policy.parameters, substeps=args.substeps)
    compact = predictor.rollout(initial, horizon_cycles=args.preview_cycles, moment_tolerance=args.moment_tolerance)
    record: dict[str, Any] = {
        "anchor_zero_based": anchor,
        "preview_cycles": args.preview_cycles,
        "compact_status": compact.status,
        "completed_intervals": compact.completed_intervals,
        "first_failure": compact.first_failure,
        "scope": "coupled_open_loop_plant_not_reoptimized_closed_loop",
    }
    if compact.status != "complete" or compact.completed_cycles != args.preview_cycles:
        record["fallback_required"] = True
        return record, {}
    preview = replace(normal, controls=flatten_preview_pulse_widths(compact.pulse_widths))
    normal_metrics, normal_arrays = evaluate_rollout(
        normal, profile, evaluator="dop853", samples_per_interval=args.samples_per_interval,
        rtol=args.rtol, atol=args.atol,
    )
    preview_metrics, preview_arrays = evaluate_rollout(
        preview, profile, evaluator="dop853", samples_per_interval=args.samples_per_interval,
        rtol=args.rtol, atol=args.atol,
    )
    normal_slow = _boundary_slow_states(normal_arrays, normal, policy.muscle_names)
    preview_slow = _boundary_slow_states(preview_arrays, preview, policy.muscle_names)
    fatigue, slow_delta = _slow_comparison(preview_slow, normal_slow, normal.parameters, policy.muscle_names)
    record.update(
        fallback_required=False,
        preview_physical_gate_passed=_physical_ok(preview_metrics),
        normal_rho_physical_gate_passed=_physical_ok(normal_metrics),
        preview_physical_metrics=preview_metrics["physical_metrics"],
        normal_rho_physical_metrics=normal_metrics["physical_metrics"],
        fatigue_against_normal_rho=fatigue,
    )
    arrays = {
        f"anchor_{anchor}_preview_shooting_states": preview_arrays["shooting_states"],
        f"anchor_{anchor}_normal_rho_shooting_states": normal_arrays["shooting_states"],
        f"anchor_{anchor}_preview_minus_normal_rho_slow_ratio": slow_delta,
        f"anchor_{anchor}_preview_pulse_widths_s": preview.controls,
    }
    return record, arrays


def _summary(blocks: list[dict]) -> dict:
    feasible = [block for block in blocks if not block.get("fallback_required")]
    gated = [block for block in feasible if block.get("preview_physical_gate_passed")]
    deltas = [block["fatigue_against_normal_rho"]["terminal_preview_minus_normal_rho"] for block in feasible]
    return {
        "anchors": len(blocks),
        "compact_preview_complete": len(feasible),
        "fallback_required": len(blocks) - len(feasible),
        "preview_physical_gate_passed": len(gated),
        "normal_rho_physical_gate_passed": sum(bool(block.get("normal_rho_physical_gate_passed")) for block in feasible),
        "mean_terminal_preview_minus_normal_rho_capacity_ratio": (
            float(np.mean([item["mean_capacity_ratio"] for item in deltas])) if deltas else None
        ),
        "mean_terminal_preview_minus_normal_rho_tau1_ratio": (
            float(np.mean([item["mean_tau1_ratio"] for item in deltas])) if deltas else None
        ),
        "mean_terminal_preview_minus_normal_rho_km_ratio": (
            float(np.mean([item["mean_km_ratio"] for item in deltas])) if deltas else None
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reduced-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--model-config", type=Path, default=None,
        help="Complete declared muscle parameters, required for legacy archives without embedded parameters.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview-cycles", type=int, default=5)
    parser.add_argument("--max-blocks", type=int, default=20)
    parser.add_argument("--anchor-stride", type=int, default=5)
    parser.add_argument("--cycle-period", type=float, default=1.0)
    parser.add_argument(
        "--legacy-formulation", choices=("dynamic", "isokinetic"), default=None,
        help="Explicit provenance declaration only for legacy archives missing formulation.",
    )
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--moment-tolerance", type=float, default=1e-8)
    parser.add_argument("--samples-per-interval", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-11)
    args = parser.parse_args(argv)
    if args.preview_cycles < 1 or args.max_blocks < 1 or args.anchor_stride < 1:
        raise ValueError("preview cycles, max blocks and anchor stride must be positive.")
    profile = ReducedCyclingDynamics.load(args.reduced_profile)
    # A preview beginning at anchor uses normal RHO boundaries anchor+1 through
    # anchor+horizon, so reserve those source cycles.
    first = build_rho_adaptive_moment_policy(args.source, args.reduced_profile, cycle_index=0,
                                              cycle_period=args.cycle_period, model_path=args.model_path)
    anchors = preview_anchor_indices(first.source_cycle_count - args.preview_cycles, args.preview_cycles,
                                     args.max_blocks, anchor_stride=args.anchor_stride)
    args.output.mkdir(parents=True, exist_ok=True)
    records, arrays = [], {}
    for anchor in anchors:
        record, values = _run_anchor(args, profile, anchor)
        records.append(record)
        arrays.update(values)
        print(json.dumps({key: record.get(key) for key in (
            "anchor_zero_based", "compact_status", "fallback_required", "preview_physical_gate_passed",
        )}), flush=True)
    report = {
        "schema": "cocofest-preview-coupled-plant-validation-v1",
        "scope": "coupled_open_loop_plant_not_reoptimized_closed_loop",
        "source": str(args.source.resolve()),
        "reduced_profile": str(args.reduced_profile.resolve()),
        "settings": {"preview_cycles": args.preview_cycles, "anchors": anchors,
                     "anchor_stride": args.anchor_stride, "substeps": args.substeps,
                     "samples_per_interval": args.samples_per_interval, "rtol": args.rtol, "atol": args.atol,
                     "legacy_formulation": args.legacy_formulation,
                     "model_config": None if args.model_config is None else str(args.model_config.resolve())},
        "blocks": records,
        "summary": _summary(records),
    }
    (args.output / "report.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    print(f"Report: {args.output / 'report.json'}", flush=True)
    return report


if __name__ == "__main__":
    main()
