"""Benchmark a five-cycle Ding-preview controller against RHO solved each cycle.

This is deliberately a *prospective open-loop emulation*, not a replacement
for the RHO controller.  For each archived, certified RHO cycle it:

1. charges the measured IPOPT solve time for that cycle (the only OCP solve
   in the proposed five-cycle block);
2. derives phase-wise muscle-moment targets from that cycle;
3. rolls the complete five-state Ding model forward for five cycles while a
   small bounded allocator adapts pulse widths; and
4. independently replays those pulse widths with the full Ding equations.

The reference is the five corresponding archived IPOPT solves.  This makes
the latency accounting explicit, while keeping the preview outside the NLP.
It does *not* claim closed-loop equivalence: the archive is not advanced with
the preview's controls.  A coupled controller experiment must subsequently
establish that property.

Typical unilateral invocation::

    python scripts/benchmark_cycle_preview_controller.py \
      --source ipopt-linear-solver-150-20260904/resistance-0p10Nm/\
ipopt-sx-radau5-ma57-150-max2000-reduced/validated-rho-trajectory.npz \
      --baseline-json ipopt-linear-solver-150-20260904/resistance-0p10Nm/\
ipopt-sx-radau5-ma57-150-max2000-reduced/result.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.compact_muscle_prediction import CompactMusclePredictor
from cocofest.optimization.rho_adaptive_moment_policy import build_rho_adaptive_moment_policy
from cocofest.optimization.rho_rollout_adapter import select_certified_rho_cycle
from scripts.validate_local_endurance_value import _full_ding_replay, _source_terminal_states


DEFAULT_SOURCE = Path(
    "ipopt-linear-solver-150-20260904/resistance-0p10Nm/"
    "ipopt-sx-radau5-ma57-150-max2000-reduced/validated-rho-trajectory.npz"
)
DEFAULT_BASELINE_JSON = Path(
    "ipopt-linear-solver-150-20260904/resistance-0p10Nm/"
    "ipopt-sx-radau5-ma57-150-max2000-reduced/result.json"
)
DEFAULT_PROFILE = Path("benchmark-seed/reduced-cycling-fourier12.npz")
DEFAULT_OUTPUT = Path(".cache/cycle-preview-controller-benchmark")
FATIGUE_STATE_NAMES = ("A", "Tau1", "Km")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
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


def _result_entry(document: dict) -> dict:
    """Return the single solver record emitted by the comparison benchmark."""
    results = document.get("results")
    if isinstance(results, list):
        if len(results) != 1:
            raise ValueError("baseline JSON must contain exactly one solver result.")
        results = results[0]
    if not isinstance(results, dict):
        raise ValueError("baseline JSON has no readable solver result.")
    return results


def baseline_window_solver_times(document: dict) -> np.ndarray:
    """Extract validated one-cycle IPOPT times, preserving archive order."""
    result = _result_entry(document)
    windows = result.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("baseline JSON has no per-window timing records.")
    values = []
    for index, window in enumerate(windows):
        if not isinstance(window, dict) or not bool(window.get("validated")):
            raise ValueError(f"baseline window {index} is not validated.")
        value = window.get("solver_time_s")
        if not isinstance(value, (float, int)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"baseline window {index} has no finite solver_time_s.")
        values.append(float(value))
    return np.asarray(values)


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean_s": None, "median_s": None, "p90_s": None, "max_s": None}
    return {
        "count": int(len(values)), "mean_s": float(np.mean(values)),
        "median_s": float(np.median(values)), "p90_s": float(np.quantile(values, .9)),
        "max_s": float(np.max(values)),
    }


def preview_anchor_indices(
    cycle_count: int,
    horizon_cycles: int,
    max_blocks: int,
    *,
    anchor_stride: int | None = None,
) -> list[int]:
    """Select preview anchors, disjoint by default or sliding for validation."""
    if cycle_count == 1:
        return [0]
    if cycle_count < horizon_cycles:
        raise ValueError("source has fewer cycles than the preview horizon (and is not a one-cycle seed).")
    stride = horizon_cycles if anchor_stride is None else anchor_stride
    if isinstance(stride, bool) or int(stride) != stride or stride < 1:
        raise ValueError("anchor_stride must be a positive integer when provided.")
    # ``cycle_count - horizon + 1`` is a count, hence a valid exclusive stop.
    return list(range(0, min(cycle_count - horizon_cycles + 1, int(stride) * max_blocks), int(stride)))


def _physical_state_audit(history: np.ndarray, parameters) -> dict:
    rest = np.asarray([item.fatigue.rest_state for item in parameters])
    return {
        "minimum_cn": float(np.min(history[:, :, 0])),
        "minimum_force_n": float(np.min(history[:, :, 1])),
        "minimum_capacity_ratio": float(np.min(history[:, :, 2] / rest[:, 0])),
        "minimum_tau1_s": float(np.min(history[:, :, 3])),
        "minimum_km": float(np.min(history[:, :, 4])),
    }


def rho_future_fatigue_boundaries(
    cycle,
    muscle_names: tuple[str, ...],
    *,
    horizon_cycles: int,
) -> np.ndarray:
    """Return archived slow states at the shared RHO boundary and successors.

    The compact preview begins at the *end* of its anchor RHO cycle.  Thus its
    first predicted cycle is compared with the next normal-RHO cycle, not with
    the anchor itself.  Keeping this off-by-one explicit is essential: the
    comparison asks whether holding the anchor's moment reference changes
    fatigue relative to re-solving each of the following cycles.
    """

    if horizon_cycles < 1:
        raise ValueError("horizon_cycles must be positive.")
    stride = cycle.stimulations_per_cycle * cycle.state_columns_per_interval
    columns = cycle.end_column + stride * np.arange(horizon_cycles + 1)
    state_length = next(iter(cycle.states.values())).size
    if columns[-1] >= state_length:
        raise ValueError("RHO archive does not contain all requested future fatigue boundaries.")
    values = np.empty((horizon_cycles + 1, len(muscle_names), len(FATIGUE_STATE_NAMES)))
    for muscle_index, muscle_name in enumerate(muscle_names):
        for state_index, state_name in enumerate(FATIGUE_STATE_NAMES):
            key = f"{state_name}_{muscle_name}"
            if key not in cycle.states:
                raise ValueError(f"RHO archive is missing fatigue state {key!r}.")
            values[:, muscle_index, state_index] = cycle.states[key][columns]
    return values


def compare_future_fatigue(
    full_history: np.ndarray,
    rho_boundaries: np.ndarray,
    parameters,
    *,
    intervals_per_cycle: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Compare preview fatigue with normal RHO at equal future cycle boundaries.

    ``A`` decreases with fatigue whereas ``Tau1`` and ``Km`` increase.  The
    report retains all three normalized physical states rather than inventing
    an unvalidated scalar fatigue score.
    """

    full_history = np.asarray(full_history, dtype=float)
    rho_boundaries = np.asarray(rho_boundaries, dtype=float)
    if intervals_per_cycle < 1 or full_history.ndim != 3 or rho_boundaries.ndim != 3:
        raise ValueError("fatigue histories must be 3-D and intervals_per_cycle positive.")
    preview = full_history[::intervals_per_cycle, :, 2:]
    expected = (rho_boundaries.shape[0], len(parameters), len(FATIGUE_STATE_NAMES))
    if preview.shape != expected or rho_boundaries.shape != expected:
        raise ValueError("Preview and normal-RHO fatigue boundaries have incompatible shapes.")
    rest = np.asarray([item.fatigue.rest_state for item in parameters], dtype=float)
    if rest.shape != (len(parameters), 3) or np.any(rest <= 0):
        raise ValueError("fatigue rest states must be finite and positive.")
    preview_ratio = preview / rest[None, :, :]
    rho_ratio = rho_boundaries / rest[None, :, :]
    delta = preview_ratio - rho_ratio
    # At the first boundary both methods start from exactly the same archived
    # state.  This is a structural check on the declared comparison alignment.
    initial_error = float(np.max(np.abs(delta[0])))
    terminal_delta = delta[-1]
    terminal_capacity_delta = terminal_delta[:, 0]
    terminal_tau1_delta = terminal_delta[:, 1]
    terminal_km_delta = terminal_delta[:, 2]
    # Positive means the preview is more fatigued for every component below.
    worse = ((terminal_capacity_delta < 0.0) | (terminal_tau1_delta > 0.0) | (terminal_km_delta > 0.0))
    report = {
        "scope": "preview holds the anchor moment reference; normal RHO re-solves each following cycle",
        "state_order": list(FATIGUE_STATE_NAMES),
        "normalization": "each state divided by its muscle-specific rest value",
        "initial_boundary_maximum_absolute_normalized_difference": initial_error,
        "maximum_absolute_normalized_slow_state_difference": float(np.max(np.abs(delta))),
        "terminal_preview_minus_normal_rho": {
            "mean_capacity_ratio": float(np.mean(terminal_capacity_delta)),
            "minimum_capacity_ratio": float(np.min(terminal_capacity_delta)),
            "mean_tau1_ratio": float(np.mean(terminal_tau1_delta)),
            "maximum_tau1_ratio": float(np.max(terminal_tau1_delta)),
            "mean_km_ratio": float(np.mean(terminal_km_delta)),
            "maximum_km_ratio": float(np.max(terminal_km_delta)),
            "muscles_with_any_worse_terminal_fatigue_indicator": int(np.count_nonzero(worse)),
        },
    }
    arrays = {
        "preview_fatigue_ratio": preview_ratio,
        "normal_rho_fatigue_ratio": rho_ratio,
        "preview_minus_normal_rho_fatigue_ratio": delta,
    }
    return report, arrays


def sanitize_initial_calcium(states: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    """Remove only numerical Cn underflow at an archived cycle boundary.

    Cn is non-negative analytically.  A materially negative value means the
    source is invalid and must not be silently projected; values in
    ``[-tolerance, 0)`` are serialization/collocation round-off and are set to
    zero before the independent Ding replay.
    """
    values = np.asarray(states, dtype=float).copy()
    if values.ndim != 2 or values.shape[1] != 5 or not np.all(np.isfinite(values)):
        raise ValueError("initial Ding states must be a finite (muscles, 5) array.")
    if np.any(values[:, 0] < -tolerance):
        raise ValueError("source has materially negative Cn; refusing to sanitize it.")
    values[:, 0] = np.maximum(values[:, 0], 0.0)
    return values


def _run_block(args, anchor: int, baseline_times: np.ndarray) -> tuple[dict, dict]:
    started = perf_counter()
    policy = build_rho_adaptive_moment_policy(
        args.source, args.reduced_profile, cycle_index=anchor, cycle_period=args.cycle_period,
        model_path=args.model_path,
    )
    policy_build_s = perf_counter() - started
    cycle, _ = select_certified_rho_cycle(args.source, cycle_index=anchor, cycle_period=args.cycle_period)
    initial_raw = _source_terminal_states(cycle, policy.muscle_names)
    initial = sanitize_initial_calcium(initial_raw, tolerance=args.cn_roundoff_tolerance)

    started = perf_counter()
    try:
        predictor = CompactMusclePredictor(policy.intervals, policy.parameters, substeps=args.substeps)
    except ValueError as error:
        # A negative force-length/velocity gain is not a numerical failure we
        # may hide by clipping: it would alter the Ding force equation.  Keep
        # the rejected block in the report so bilateral profile defects are
        # visible to the controller work rather than producing a traceback.
        predictor_build_s = perf_counter() - started
        return {
            "anchor_zero_based": anchor,
            "preview_cycles": args.preview_cycles,
            "source_cycle_certified": bool(cycle.certified),
            "initial_cn_roundoff_corrections": int(np.count_nonzero(initial_raw[:, 0] < 0)),
            "policy_build_s": policy_build_s,
            "predictor_build_s": predictor_build_s,
            "preview_rollout_s": None,
            "preview_online_overhead_s": policy_build_s + predictor_build_s,
            "compact_status": "invalid_preview_model",
            "first_failure": {"status": "predictor_construction_error", "message": str(error)},
            "baseline_block_solver_time_s": float(np.sum(baseline_times[anchor:anchor + args.preview_cycles])),
            "proposed_first_ocp_solver_time_s": float(baseline_times[anchor]),
            "cycle_period_s": policy.period,
            "preview_certified": False,
        }, {}
    predictor_build_s = perf_counter() - started
    started = perf_counter()
    compact = predictor.rollout(initial, horizon_cycles=args.preview_cycles, moment_tolerance=args.moment_tolerance)
    rollout_s = perf_counter() - started

    record = {
        "anchor_zero_based": anchor,
        "preview_cycles": args.preview_cycles,
        "source_cycle_certified": bool(cycle.certified),
        "initial_cn_roundoff_corrections": int(np.count_nonzero(initial_raw[:, 0] < 0)),
        "policy_build_s": policy_build_s,
        "predictor_build_s": predictor_build_s,
        "preview_rollout_s": rollout_s,
        "preview_online_overhead_s": policy_build_s + predictor_build_s + rollout_s,
        "compact_status": compact.status,
        "completed_cycles": compact.completed_cycles,
        "completed_intervals": compact.completed_intervals,
        "first_failure": compact.first_failure,
        "baseline_block_solver_time_s": float(np.sum(baseline_times[anchor:anchor + args.preview_cycles])),
        "proposed_first_ocp_solver_time_s": float(baseline_times[anchor]),
        "cycle_period_s": policy.period,
    }
    # ``preview_rollout_s`` is the online kernel: in a controller the model,
    # reduced mechanics and allocation workspace stay resident.  The separate
    # end-to-end emulation time includes archive parsing and policy assembly;
    # it is reported for reproducibility but is not a fair real-time charge.
    record["proposed_block_online_kernel_time_s"] = (
        record["proposed_first_ocp_solver_time_s"] + record["preview_rollout_s"]
    )
    record["proposed_block_end_to_end_emulation_time_s"] = (
        record["proposed_first_ocp_solver_time_s"] + record["preview_online_overhead_s"]
    )
    record.update(online_time_saved_s=None, block_speedup=None, first_cycle_deadline_passed=False)
    arrays = {}
    if compact.status != "complete" or compact.completed_cycles != args.preview_cycles:
        record["preview_certified"] = False
        return record, arrays

    started = perf_counter()
    try:
        full_history, full_moments = _full_ding_replay(
            initial, policy, compact.pulse_widths, compact.completed_intervals, args.full_ding_substeps
        )
    except (ValueError, FloatingPointError, OverflowError) as error:
        record.update(full_ding_replay_status="failed", full_ding_replay_message=str(error),
                      full_ding_replay_s=perf_counter() - started, preview_certified=False)
        return record, arrays
    replay_s = perf_counter() - started
    targets = np.asarray([item.target_moments for item in policy.intervals])
    target_moments = np.tile(targets, (args.preview_cycles, 1))
    individual_error = full_moments - target_moments
    total_error = individual_error.sum(axis=1)
    slow_rest = np.asarray([item.fatigue.rest_state for item in policy.parameters])[None, :, :]
    slow_error = ((compact.state_history[: compact.completed_intervals + 1, :, 2:] - full_history[:, :, 2:])
                  / slow_rest)
    max_total_error = float(np.max(np.abs(total_error)))
    max_individual_error = float(np.max(np.abs(individual_error)))
    max_slow_error = float(np.max(np.abs(slow_error)))
    state = _physical_state_audit(full_history, policy.parameters)
    rho_fatigue_boundaries = rho_future_fatigue_boundaries(
        cycle,
        policy.muscle_names,
        horizon_cycles=args.preview_cycles,
    )
    fatigue_comparison, fatigue_arrays = compare_future_fatigue(
        full_history,
        rho_fatigue_boundaries,
        policy.parameters,
        intervals_per_cycle=len(policy.intervals),
    )
    pw_lower = np.asarray([item.pd0 for item in policy.parameters])[None, :, None]
    pw_upper = np.asarray([item.pulse_width_max for item in policy.parameters])[None, :, None]
    pw_violation = float(max(
        np.max(np.maximum(pw_lower - compact.pulse_widths, 0.0)),
        np.max(np.maximum(compact.pulse_widths - pw_upper, 0.0)),
    ))
    record.update(
        full_ding_replay_status="complete", full_ding_replay_s=replay_s,
        maximum_total_crank_moment_error_nm=max_total_error,
        rms_total_crank_moment_error_nm=float(np.sqrt(np.mean(total_error ** 2))),
        maximum_individual_muscle_moment_error_nm=max_individual_error,
        maximum_normalized_compact_slow_state_error=max_slow_error,
        maximum_pw_bound_violation_s=pw_violation,
        full_ding_state_audit=state,
        fatigue_against_normal_rho=fatigue_comparison,
        # Future validation requires a coupled plant rollout; this gate only
        # certifies the numerical Ding/moment preview used by this protocol.
        preview_certified=bool(
            max_total_error <= args.total_moment_tolerance_nm
            and max_slow_error <= args.slow_state_tolerance
            and pw_violation <= 1e-12
            and state["minimum_cn"] >= 0 and state["minimum_force_n"] >= 0
            and state["minimum_capacity_ratio"] > 0 and state["minimum_tau1_s"] > 0 and state["minimum_km"] > 0
        ),
    )
    add_certified_speed_comparison(record)
    arrays = {
        f"anchor_{anchor}_total_moment_error_nm": total_error,
        f"anchor_{anchor}_individual_moment_error_nm": individual_error,
        f"anchor_{anchor}_full_ding_state_history": full_history,
        f"anchor_{anchor}_pulse_widths_s": compact.pulse_widths,
        f"anchor_{anchor}_normal_rho_fatigue_boundaries": rho_fatigue_boundaries,
        **{f"anchor_{anchor}_{name}": values for name, values in fatigue_arrays.items()},
    }
    return record, arrays


def add_certified_speed_comparison(record: dict) -> None:
    """A short failed prefix cannot be compared with an entire baseline block."""
    record.update(online_time_saved_s=None, block_speedup=None, first_cycle_deadline_passed=False)
    if not record.get("preview_certified"):
        return
    elapsed = record["proposed_block_online_kernel_time_s"]
    record["online_time_saved_s"] = record["baseline_block_solver_time_s"] - elapsed
    record["block_speedup"] = record["baseline_block_solver_time_s"] / elapsed if elapsed > 0 else None
    record["first_cycle_deadline_passed"] = bool(elapsed < record["cycle_period_s"])


def summarize_blocks(blocks: list[dict]) -> dict:
    """Summarize speed only over preview blocks that passed the numerical gate."""
    valid = [item for item in blocks if item.get("preview_certified")]
    fatigue = [item["fatigue_against_normal_rho"] for item in valid if "fatigue_against_normal_rho" in item]
    capacity = np.asarray([
        item["terminal_preview_minus_normal_rho"]["mean_capacity_ratio"] for item in fatigue
    ])
    tau1 = np.asarray([
        item["terminal_preview_minus_normal_rho"]["mean_tau1_ratio"] for item in fatigue
    ])
    km = np.asarray([
        item["terminal_preview_minus_normal_rho"]["mean_km_ratio"] for item in fatigue
    ])
    return {
        "blocks_requested": len(blocks),
        "preview_numerically_certified_blocks": len(valid),
        "all_preview_blocks_numerically_certified": len(valid) == len(blocks),
        "baseline_five_cycle_solver_time": _stats(np.asarray([item["baseline_block_solver_time_s"] for item in blocks])),
        "proposed_five_cycle_online_kernel_time": _stats(np.asarray([
            item["proposed_block_online_kernel_time_s"] for item in valid
        ])),
        "proposed_five_cycle_end_to_end_emulation_time": _stats(np.asarray([
            item["proposed_block_end_to_end_emulation_time_s"] for item in valid
        ])),
        "preview_overhead_only": _stats(np.asarray([item["preview_online_overhead_s"] for item in valid])),
        "five_cycle_speedup": _stats(np.asarray([item["block_speedup"] for item in valid])),
        "first_cycle_deadline_passed": sum(bool(item.get("first_cycle_deadline_passed")) for item in valid),
        "minimum_capacity_ratio": (
            min(item["full_ding_state_audit"]["minimum_capacity_ratio"] for item in valid) if valid else None
        ),
        "maximum_total_crank_moment_error_nm": (
            max(item["maximum_total_crank_moment_error_nm"] for item in valid) if valid else None
        ),
        "maximum_individual_muscle_moment_error_nm": (
            max(item["maximum_individual_muscle_moment_error_nm"] for item in valid) if valid else None
        ),
        "fatigue_against_normal_rho": {
            "comparison_blocks": len(fatigue),
            "mean_terminal_preview_minus_normal_rho_capacity_ratio": (
                float(np.mean(capacity)) if len(capacity) else None
            ),
            "mean_terminal_preview_minus_normal_rho_tau1_ratio": (
                float(np.mean(tau1)) if len(tau1) else None
            ),
            "mean_terminal_preview_minus_normal_rho_km_ratio": (
                float(np.mean(km)) if len(km) else None
            ),
            "maximum_initial_boundary_alignment_error": (
                max(item["initial_boundary_maximum_absolute_normalized_difference"] for item in fatigue)
                if fatigue else None
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--reduced-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--model-path", type=Path, default=None,
                        help="bioMod used to reconstruct the matching Ding muscle models.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview-cycles", type=int, default=5)
    parser.add_argument("--max-blocks", type=int, default=20)
    parser.add_argument(
        "--anchor-stride", type=int, default=None,
        help="Cycle spacing between anchors; default is preview-cycles (disjoint blocks). Use 1 for a sliding validation.",
    )
    parser.add_argument("--cycle-period", type=float, default=1.0)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--full-ding-substeps", type=int, default=16)
    parser.add_argument("--moment-tolerance", type=float, default=1e-8)
    parser.add_argument("--total-moment-tolerance-nm", type=float, default=1e-3)
    parser.add_argument("--slow-state-tolerance", type=float, default=1e-3)
    parser.add_argument("--cn-roundoff-tolerance", type=float, default=1e-10)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.preview_cycles < 1 or args.max_blocks < 1:
        raise ValueError("preview-cycles and max-blocks must be positive.")
    baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    baseline_times = baseline_window_solver_times(baseline)
    cycle, _ = select_certified_rho_cycle(args.source, cycle_index=0, cycle_period=args.cycle_period)
    anchors = preview_anchor_indices(
        cycle.cycle_count,
        args.preview_cycles,
        args.max_blocks,
        anchor_stride=args.anchor_stride,
    )
    required_baseline_windows = max(anchor + args.preview_cycles for anchor in anchors)
    if len(baseline_times) < required_baseline_windows:
        raise ValueError("baseline JSON contains fewer validated windows than a requested preview block.")
    args.output.mkdir(parents=True, exist_ok=True)
    arrays, blocks = {}, []
    for anchor in anchors:
        block, block_arrays = _run_block(args, anchor, baseline_times)
        blocks.append(block)
        arrays.update(block_arrays)
        print(json.dumps({key: block.get(key) for key in (
            "anchor_zero_based", "compact_status", "preview_certified", "preview_online_overhead_s",
            "baseline_block_solver_time_s", "proposed_block_online_kernel_time_s", "block_speedup",
        )}), flush=True)
    report = {
        "schema": "cocofest-cycle-preview-controller-benchmark-v1",
        "comparison_scope": "open_loop_preview_emulation_not_closed_loop_controller_validation",
        "source": str(args.source.resolve()),
        "baseline_json": str(args.baseline_json.resolve()),
        "baseline_per_cycle_solver_time": _stats(baseline_times),
        "settings": {
            "preview_cycles": args.preview_cycles, "anchors": anchors,
            "anchor_stride": args.anchor_stride if args.anchor_stride is not None else args.preview_cycles,
            "substeps": args.substeps,
            "full_ding_substeps": args.full_ding_substeps,
            "total_moment_tolerance_nm": args.total_moment_tolerance_nm,
            "slow_state_tolerance": args.slow_state_tolerance,
            "cn_roundoff_tolerance": args.cn_roundoff_tolerance,
        },
        "blocks": blocks,
        "summary": summarize_blocks(blocks),
    }
    (args.output / "report.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    print(f"Report: {args.output / 'report.json'}", flush=True)
    return report


if __name__ == "__main__":
    main()
