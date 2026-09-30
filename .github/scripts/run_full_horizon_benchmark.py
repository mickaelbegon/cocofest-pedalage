#!/usr/bin/env python3
"""Run an RSS-bounded RHO-to-full-horizon continuation with IPOPT/MA57.

Two consecutive reduced RHO cycles first initialize FHO_2.  Every subsequent
problem is built from the last certified full-horizon solution.  By default the
horizon grows by one cycle.  An adaptive step can instead append several
autonomous terminal-state RHO cycles before solving the next FHO; a rejected
multi-cycle jump automatically falls back to the certified one-cycle ladder.
Each local RHO first uses the carrier's extrapolated last cycle. Initial-state
homotopy is a recovery recipe; the long RHO reference is never a prerequisite.

Optionally, a failed one-cycle RHO handoff falls back to a direct FHO terminal
bridge.  This deliberately separates ``RHO cannot produce cycle N+1`` from
``FHO_N+1 is infeasible``: the latter is tested by IPOPT on an MX, uncompiled
FHO transcription seeded from the certified FHO terminal state.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Iterable

import numpy as np

from cocofest.optimization.solution_archive import physical_archive_arrays

GIB = 1024**3
SMALL_RUNNER_RSS_LIMIT_GIB = 12.5
LARGE_RUNNER_RSS_LIMIT_GIB = 97.5
RHO_INITIAL_STATE_HOMOTOPY_MIN_STEP = 1.0 / 256.0


def _refresh_physical_archive(
    payload: dict[str, np.ndarray], metadata: dict,
) -> dict:
    """Regenerate physical replay arrays after changing a trajectory length.

    The FHO/RHO seed constructors frequently reduce a multi-cycle archive to
    one cycle.  Retaining the source ``physical__*`` arrays or its
    ``physical_cycles`` metadata then makes the otherwise valid seed fail
    strict archive validation before the NLP is built.
    """

    for key in tuple(payload):
        if key.startswith("physical__"):
            payload.pop(key)
    states = {
        key.removeprefix("states__"): value
        for key, value in payload.items()
        if key.startswith("states__")
    }
    controls = {
        key.removeprefix("controls__"): value
        for key, value in payload.items()
        if key.startswith("controls__")
    }
    arrays, refreshed = physical_archive_arrays(states, controls, metadata)
    payload.update(arrays)
    return refreshed


class RhoReferenceCycleUnavailable(ValueError):
    """A valid RHO reference ends before the requested continuation cycle."""

    def __init__(self, target_cycle: int, available_cycles: int):
        self.target_cycle = target_cycle
        self.available_cycles = available_cycles
        super().__init__(
            f"RHO reference contains {available_cycles} certified cycles; "
            f"cycle {target_cycle} is unavailable."
        )


def horizon_sweep_targets(max_cycles: int) -> list[int]:
    """Return the strict one-cycle-at-a-time FHO continuation ladder."""

    if max_cycles < 2:
        raise ValueError("max_cycles must be at least two.")
    return list(range(2, max_cycles + 1))


def adaptive_continuation_target(
    current_cycles: int, max_cycles: int, step_cycles: int
) -> int:
    """Return the next bounded continuation target."""

    if current_cycles < 2 or max_cycles < current_cycles:
        raise ValueError("The continuation interval is inconsistent.")
    if step_cycles < 1:
        raise ValueError("step_cycles must be strictly positive.")
    return min(max_cycles, current_cycles + step_cycles)


def objective_gate(
    candidate_objective: float | None,
    additive_seed_objective: float | None,
    relative_tolerance: float,
) -> dict:
    """Check that a jump did not worsen its concatenated feasible seed."""

    if relative_tolerance < 0 or not math.isfinite(relative_tolerance):
        raise ValueError("relative_tolerance must be finite and non-negative.")
    comparable = bool(
        candidate_objective is not None
        and additive_seed_objective is not None
        and math.isfinite(candidate_objective)
        and math.isfinite(additive_seed_objective)
    )
    relative_degradation = None
    passes = True
    if comparable:
        scale = max(abs(additive_seed_objective), 1e-12)
        relative_degradation = (
            candidate_objective - additive_seed_objective
        ) / scale
        passes = relative_degradation <= relative_tolerance
    return {
        "comparable": comparable,
        "candidate": candidate_objective,
        "additive_seed_reference": additive_seed_objective,
        "relative_degradation": relative_degradation,
        "relative_tolerance": relative_tolerance,
        "passes": passes,
    }


def refinement_targets(last_success: int, first_failure: int) -> list[int]:
    """Fill the final coarse interval without retrying either endpoint."""

    if last_success < 0 or first_failure <= last_success:
        raise ValueError("The refinement interval must be ordered.")
    return list(range(last_success + 1, first_failure))


def automatic_rss_limit_gib(total_memory_bytes: int) -> float:
    """Choose a conservative RSS cap for 16 GiB and 128 GiB runners."""

    if total_memory_bytes <= 0:
        raise ValueError("total_memory_bytes must be strictly positive.")
    total_gib = total_memory_bytes / GIB
    if total_gib <= 32.0:
        return min(SMALL_RUNNER_RSS_LIMIT_GIB, 0.80 * total_gib)
    if total_gib >= 96.0:
        return min(LARGE_RUNNER_RSS_LIMIT_GIB, 0.80 * total_gib)
    # Intermediate machines are not benchmark targets, but retaining roughly
    # 22 % headroom is safer than extrapolating the 128 GiB absolute cap.
    return 0.78 * total_gib


def _read_positive_integer(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw.isdigit():
        return None
    value = int(raw)
    return value if value > 0 else None


def available_memory_bytes() -> int:
    """Return the tighter physical/cgroup allocation, with a macOS fallback."""

    physical = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                physical = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass

    cgroup_limits = [
        _read_positive_integer(Path("/sys/fs/cgroup/memory.max")),
        _read_positive_integer(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")),
    ]
    candidates = [value for value in (physical, *cgroup_limits) if value]
    if not candidates:
        try:
            page_count = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if page_count > 0 and page_size > 0:
                candidates.append(page_count * page_size)
        except (OSError, ValueError, TypeError):
            pass
    if not candidates:
        raise RuntimeError("Cannot determine the runner memory allocation.")
    # Some cgroup v1 hosts expose an effectively infinite sentinel.
    finite = [value for value in candidates if value < (1 << 60)]
    return min(finite or candidates)


def _child_pids(pid: int) -> tuple[int, ...]:
    children_path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return tuple(int(value) for value in children_path.read_text().split())
    except (OSError, ValueError):
        pass
    try:
        completed = subprocess.run(
            ["pgrep", "-P", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
        return tuple(int(value) for value in completed.stdout.split())
    except (OSError, ValueError):
        return ()


def process_tree_pids(root_pid: int) -> set[int]:
    pending = [root_pid]
    observed: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in observed:
            continue
        observed.add(pid)
        pending.extend(_child_pids(pid))
    return observed


def _process_rss_bytes(pid: int) -> int:
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        if line.startswith("VmRSS:"):
            try:
                return int(line.split()[1]) * 1024
            except (ValueError, IndexError):
                return 0
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
        rss_kib = int(completed.stdout.strip())
        return rss_kib * 1024 if rss_kib > 0 else 0
    except (OSError, ValueError):
        pass
    return 0


def process_tree_rss_bytes(root_pid: int) -> int:
    return sum(_process_rss_bytes(pid) for pid in process_tree_pids(root_pid))


@dataclass
class MonitoredRun:
    command: list[str]
    return_code: int
    peak_rss_bytes: int
    elapsed_s: float
    memory_limit_exceeded: bool
    timed_out: bool
    log_path: str


def _terminate_process_group(process: subprocess.Popen, grace_s: float = 10.0) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_monitored(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    rss_limit_bytes: int,
    heartbeat_label: str = "solver",
    poll_interval_s: float = 0.5,
    timeout_s: float | None = None,
) -> MonitoredRun:
    """Run one solver process and stop its whole process group at the RSS cap."""

    if rss_limit_bytes <= 0:
        raise ValueError("rss_limit_bytes must be strictly positive.")
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("timeout_s must be strictly positive when provided.")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    peak_rss = 0
    memory_limit_exceeded = False
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(
            f"full-horizon start: stage={heartbeat_label} pid={process.pid} "
            f"timeout={timeout_s if timeout_s is not None else 'none'}s",
            flush=True,
        )
        next_heartbeat = start + 30.0
        while process.poll() is None:
            rss = process_tree_rss_bytes(process.pid)
            peak_rss = max(peak_rss, rss)
            if rss >= rss_limit_bytes:
                memory_limit_exceeded = True
                message = (
                    f"RSS limit reached: {rss / GIB:.3f} GiB >= "
                    f"{rss_limit_bytes / GIB:.3f} GiB"
                )
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
                _terminate_process_group(process)
                break
            now = time.monotonic()
            if timeout_s is not None and now - start >= timeout_s:
                timed_out = True
                message = f"Attempt timeout reached after {now - start:.1f} s"
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
                _terminate_process_group(process)
                break
            if now >= next_heartbeat:
                elapsed_s = now - start
                timeout_remaining_s = (
                    None
                    if timeout_s is None
                    else max(0.0, timeout_s - elapsed_s)
                )
                print(
                    f"full-horizon heartbeat: stage={heartbeat_label} "
                    f"pid={process.pid} elapsed={elapsed_s:.1f}s "
                    f"timeout_remaining="
                    f"{timeout_remaining_s if timeout_remaining_s is not None else 'none'}s "
                    f"rss={rss / GIB:.3f} GiB peak={peak_rss / GIB:.3f} GiB",
                    flush=True,
                )
                next_heartbeat = now + 30.0
            time.sleep(poll_interval_s)
        return_code = process.wait()
        peak_rss = max(peak_rss, process_tree_rss_bytes(process.pid))

    return MonitoredRun(
        command=command,
        return_code=return_code,
        peak_rss_bytes=peak_rss,
        elapsed_s=time.monotonic() - start,
        memory_limit_exceeded=memory_limit_exceeded,
        timed_out=timed_out,
        log_path=str(log_path),
    )


def _load_metadata(data) -> dict:
    if "metadata__json" not in data.files:
        raise ValueError("The RHO seed has no metadata__json entry.")
    return json.loads(str(data["metadata__json"].item()))


def write_rho_seed_prefix(
    source_path: Path, output_path: Path, target_cycles: int
) -> dict:
    """Slice a concatenated RHO seed to an exact multi-cycle prefix."""

    if target_cycles < 1:
        raise ValueError("target_cycles must be strictly positive.")
    with np.load(source_path, allow_pickle=False) as data:
        metadata = _load_metadata(data)
        source_cycles = int(metadata["cycles_per_window"])
        if target_cycles > source_cycles:
            raise ValueError(
                f"Cannot extract {target_cycles} cycles from {source_cycles}."
            )
        payload: dict[str, np.ndarray] = {}
        discarded_failed_window = False
        for key in data.files:
            if key == "metadata__json":
                continue
            values = np.asarray(data[key])
            if key.startswith("states__"):
                intervals, remainder = divmod(values.shape[-1] - 1, source_cycles)
                if remainder:
                    # Compact RHO export retains the last attempted window,
                    # even when that window failed and the metadata correctly
                    # reports only the certified prefix.  A complete trailing
                    # attempt has the same per-cycle layout as the prefix and
                    # must be discarded before an FHO seed is sliced.
                    intervals, trailing_remainder = divmod(
                        values.shape[-1] - 1, source_cycles + 1
                    )
                    if trailing_remainder:
                        raise ValueError(
                            f"State seed '{key}' cannot be divided into "
                            f"{source_cycles} certified cycles (or one trailing failed window)."
                        )
                    discarded_failed_window = True
                payload[key] = values[..., : target_cycles * intervals + 1]
            elif key.startswith("controls__"):
                nodes, remainder = divmod(values.shape[-1], source_cycles)
                if remainder:
                    nodes, trailing_remainder = divmod(
                        values.shape[-1], source_cycles + 1
                    )
                    if trailing_remainder:
                        raise ValueError(
                            f"Control seed '{key}' cannot be divided into "
                            f"{source_cycles} certified cycles (or one trailing failed window)."
                        )
                    discarded_failed_window = True
                payload[key] = values[..., : target_cycles * nodes]
            else:
                payload[key] = values
    metadata.update(
        {
            "cycles_per_window": target_cycles,
            "producer_mode": "receding_horizon_prefix",
            "producer_source_cycles": source_cycles,
            "discarded_trailing_failed_rho_window": discarded_failed_window,
        }
    )
    metadata = _refresh_physical_archive(payload, metadata)
    payload["metadata__json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return metadata


def write_rho_seed_cycle(
    source_path: Path, output_path: Path, cycle_number: int
) -> dict:
    """Extract one one-based cycle from a concatenated reduced RHO trace."""

    with np.load(source_path, allow_pickle=False) as data:
        metadata = _load_metadata(data)
        source_cycles = int(metadata["cycles_per_window"])
        if source_cycles < 1:
            raise ValueError("The RHO reference must contain a positive number of cycles.")
        if cycle_number < 1:
            raise ValueError(
                f"cycle_number must be in [1, {source_cycles}], got {cycle_number}."
            )
        if cycle_number > source_cycles:
            raise RhoReferenceCycleUnavailable(cycle_number, source_cycles)
        cycle_index = cycle_number - 1
        payload: dict[str, np.ndarray] = {}
        discarded_failed_window = False
        for key in data.files:
            if key == "metadata__json":
                continue
            values = np.asarray(data[key])
            if key.startswith("states__"):
                intervals, remainder = divmod(values.shape[-1] - 1, source_cycles)
                if remainder:
                    intervals, trailing_remainder = divmod(
                        values.shape[-1] - 1, source_cycles + 1
                    )
                    if trailing_remainder:
                        raise ValueError(f"State seed '{key}' has an invalid cycle layout.")
                    discarded_failed_window = True
                start = cycle_index * intervals
                payload[key] = values[..., start : start + intervals + 1].copy()
            elif key.startswith("controls__"):
                nodes, remainder = divmod(values.shape[-1], source_cycles)
                if remainder:
                    nodes, trailing_remainder = divmod(values.shape[-1], source_cycles + 1)
                    if trailing_remainder:
                        raise ValueError(
                            f"Control seed '{key}' has an invalid cycle layout."
                        )
                    discarded_failed_window = True
                start = cycle_index * nodes
                payload[key] = values[..., start : start + nodes].copy()
            else:
                payload[key] = values
    metadata.update(
        {
            "cycles_per_window": 1,
            "producer_mode": "receding_horizon_cycle_extraction",
            "producer_source_cycles": source_cycles,
            "discarded_trailing_failed_rho_window": discarded_failed_window,
            "producer_cycle_number": cycle_number,
        }
    )
    metadata = _refresh_physical_archive(payload, metadata)
    payload["metadata__json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return metadata


def write_rho_initial_state_homotopy_seed(
    source_solution_path: Path,
    reference_cycle_path: Path,
    target_fho_path: Path,
    output_path: Path,
    fraction: float,
) -> dict:
    """Move a one-cycle RHO warm start toward the terminal state of FHO_N."""

    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be finite and lie in [0, 1].")
    with np.load(source_solution_path, allow_pickle=False) as source, np.load(
        reference_cycle_path, allow_pickle=False
    ) as reference, np.load(target_fho_path, allow_pickle=False) as target:
        source_metadata = _load_metadata(source)
        reference_metadata = _load_metadata(reference)
        target_metadata = _load_metadata(target)
        if any(
            metadata.get("mechanical_formulation") != "reduced"
            for metadata in (source_metadata, reference_metadata, target_metadata)
        ):
            raise ValueError(
                "The RHO initial-state homotopy requires reduced mechanics."
            )
        if (
            int(source_metadata["cycles_per_window"]) != 1
            or int(reference_metadata["cycles_per_window"]) != 1
        ):
            raise ValueError(
                "The homotopy source and reference must contain one cycle."
            )

        source_keys = set(source.files) - {"metadata__json"}
        reference_keys = set(reference.files) - {"metadata__json"}
        # ``applied_pulse_widths__*`` are RHO export telemetry, not FHO
        # decision variables.  They are retained from the one-cycle RHO seed
        # below but are intentionally absent from a monolithic FHO solution.
        target_required_keys = {
            key for key in source_keys if key.startswith(("states__", "controls__"))
        }
        if source_keys != reference_keys or not target_required_keys <= set(target.files):
            raise ValueError(
                "The homotopy checkpoints do not expose matching variables."
            )

        payload: dict[str, np.ndarray] = {}
        maximum_initial_state_change = 0.0
        theta_winding_shift = 0.0
        for key in sorted(source_keys):
            source_values = np.asarray(source[key])
            reference_values = np.asarray(reference[key])
            if key.startswith("applied_pulse_widths__"):
                # Execution telemetry is neither an FHO decision variable nor
                # part of the homotopy.  It can have a different sampling
                # layout after a multi-cycle prefix and is intentionally omitted.
                continue
            if source_values.shape != reference_values.shape:
                raise ValueError(f"Homotopy variable '{key}' has incompatible shapes.")
            if key.startswith("states__"):
                target_values = np.asarray(target[key])
                source_for_homotopy = source_values
                reference_for_homotopy = reference_values
                if key == "states__theta":
                    target_initial = target_values[..., -1:]
                    winding_turns = np.rint(
                        (target_initial - reference_values[..., :1]) / (2.0 * np.pi)
                    )
                    candidate_shift = winding_turns * (2.0 * np.pi)
                    residual = (
                        target_initial
                        - reference_values[..., :1]
                        - candidate_shift
                    )
                    if float(np.max(np.abs(residual))) <= 0.05:
                        reference_for_homotopy = reference_values + candidate_shift
                        source_shift = np.rint(
                            (
                                reference_for_homotopy[..., :1]
                                - source_values[..., :1]
                            )
                            / (2.0 * np.pi)
                        ) * (2.0 * np.pi)
                        source_for_homotopy = source_values + source_shift
                        theta_winding_shift = float(candidate_shift.reshape(-1)[0])
                desired_initial = reference_for_homotopy[..., :1] + fraction * (
                    target_values[..., -1:] - reference_for_homotopy[..., :1]
                )
                change = desired_initial - source_for_homotopy[..., :1]
                maximum_initial_state_change = max(
                    maximum_initial_state_change,
                    float(np.max(np.abs(change))),
                )
                if key == "states__theta":
                    shifted = source_for_homotopy + change
                else:
                    shifted = source_for_homotopy + change * np.linspace(
                        1.0, 0.0, source_values.shape[-1]
                    )
                shifted[..., 0] = desired_initial[..., 0]
                payload[key] = shifted
            else:
                payload[key] = source_values.copy()

    metadata = dict(source_metadata)
    metadata.update(
        {
            "cycles_per_window": 1,
            "producer_mode": "rho_initial_state_homotopy",
            "homotopy_fraction": float(fraction),
            "homotopy_reference_cycle_number": reference_metadata.get(
                "producer_cycle_number"
            ),
            "homotopy_target_fho_cycles": target_metadata.get("cycles_per_window"),
            "homotopy_maximum_initial_state_change": maximum_initial_state_change,
            "homotopy_theta_winding_shift": theta_winding_shift,
        }
    )
    metadata = _refresh_physical_archive(payload, metadata)
    payload["metadata__json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return metadata


def write_fho_terminal_continuation_seed(source_path: Path, output_path: Path) -> dict:
    """Extrapolate the last FHO cycle into a one-cycle seed starting at its end.

    The state construction matches the established one-cycle tiling policy in
    the cycling example.  In particular, the first node is copied exactly from
    the FHO terminal state; the actual reduced RHO solve restores dynamics and
    physical feasibility before this cycle may seed a longer FHO.
    """

    with np.load(source_path, allow_pickle=False) as data:
        metadata = _load_metadata(data)
        source_cycles = int(metadata["cycles_per_window"])
        if source_cycles < 1:
            raise ValueError("The full-horizon source must contain a cycle.")
        if metadata.get("mechanical_formulation") != "reduced":
            raise ValueError("The continuation source must use reduced mechanics.")

        payload: dict[str, np.ndarray] = {}
        terminal_theta_phase_projection_rad = 0.0
        for key in data.files:
            if key == "metadata__json":
                continue
            values = np.asarray(data[key])
            if key.startswith("states__"):
                state_key = key.split("__", 1)[1]
                intervals, remainder = divmod(values.shape[-1] - 1, source_cycles)
                if remainder or intervals < 1:
                    raise ValueError(
                        f"State solution '{key}' has an invalid cycle layout."
                    )
                last_cycle = np.asarray(values[..., -intervals - 1 :], dtype=float)
                drift = last_cycle[:, -1:] - last_cycle[:, :1]
                continuation = np.empty_like(last_cycle)
                continuation[:, :1] = last_cycle[:, -1:]
                continuation[:, 1:] = last_cycle[:, 1:]
                if state_key == "theta":
                    # The FHO terminal bound permits a small phase slack.  A
                    # terminal RHO, however, starts at that exact terminal
                    # state and must complete one physical winding.  Repeating
                    # the previous FHO increment transfers the slack twice and
                    # leaves the RHO seed on its terminal bound.  Project only
                    # the *seed* phase progressively so x(0) is untouched and
                    # theta(T) = theta(0) +/- 2*pi exactly.
                    continuation[:, 1:] += drift
                    winding = np.where(drift < 0.0, -2.0 * np.pi, 2.0 * np.pi)
                    desired_terminal = continuation[:, :1] + winding
                    phase_correction = desired_terminal - continuation[:, -1:]
                    continuation[:, 1:] += phase_correction * np.linspace(
                        1.0 / intervals, 1.0, intervals
                    )
                    terminal_theta_phase_projection_rad = float(
                        np.max(np.abs(phase_correction))
                    )
                elif state_key == "q":
                    continuation[-1:, 1:] += drift[-1:, :]
                    if continuation.shape[0] > 1:
                        continuation[:-1, 1:] += drift[:-1, :] * np.linspace(
                            1.0, 0.0, intervals
                        )
                elif state_key.startswith("F_"):
                    # Force is a fast, phase-dependent state.  Unlike fatigue
                    # states, translating the whole repeated cycle by its
                    # boundary drift can make the force nonphysical (the
                    # FHO_91 -> RHO_92 handoff, for instance, produced a
                    # negative Delt_post force).  Preserve the exact terminal
                    # state at node zero and fade the template correction over
                    # a short initial segment.  The subsequent RHO projection
                    # and solve are still responsible for restoring dynamics.
                    #
                    # This is deliberately local: it does not alter the
                    # certified FHO trajectory or its terminal slack.
                    transition_nodes = min(12, intervals)
                    correction_weights = np.zeros(intervals, dtype=float)
                    correction_weights[:transition_nodes] = 1.0 - (
                        np.arange(1, transition_nodes + 1, dtype=float)
                        / float(transition_nodes)
                    )
                    continuation[:, 1:] += drift * correction_weights
                    # Muscle force has a physical lower bound at zero.  The
                    # transition is only a warm start, but keeping it inside
                    # this bound avoids presenting IPOPT with an artificial
                    # infeasible force state solely due to seed stitching.
                    continuation[:, 1:] = np.maximum(continuation[:, 1:], 0.0)
                elif state_key.startswith(("A_", "Tau1_", "Km_")):
                    continuation[:, 1:] += drift
                else:
                    continuation[:, 1:] += drift * np.linspace(1.0, 0.0, intervals)
                payload[key] = continuation
            elif key.startswith("controls__"):
                nodes, remainder = divmod(values.shape[-1], source_cycles)
                if remainder or nodes < 1:
                    raise ValueError(
                        f"Control solution '{key}' has an invalid cycle layout."
                    )
                payload[key] = np.asarray(values[..., -nodes:]).copy()
            else:
                payload[key] = values

    metadata.update(
        {
            "cycles_per_window": 1,
            "producer_mode": "full_horizon_terminal_continuation",
            "producer_source_cycles": source_cycles,
            "producer_terminal_theta_phase_projection_rad": (
                terminal_theta_phase_projection_rad
            ),
        }
    )
    metadata = _refresh_physical_archive(payload, metadata)
    payload["metadata__json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return metadata


def append_rho_extension_cycle(
    prefix_path: Path,
    extension_path: Path,
    output_path: Path,
) -> dict:
    """Append one terminal-state RHO directly to the certified FHO_N solution."""

    with np.load(prefix_path, allow_pickle=False) as prefix_data, np.load(
        extension_path, allow_pickle=False
    ) as extension_data:
        prefix_metadata = _load_metadata(prefix_data)
        extension_metadata = _load_metadata(extension_data)
        prefix_cycles = int(prefix_metadata["cycles_per_window"])
        if int(extension_metadata["cycles_per_window"]) != 1:
            raise ValueError("The RHO extension must contain exactly one cycle.")
        if prefix_metadata.get("mechanical_formulation") != "reduced" or (
            extension_metadata.get("mechanical_formulation") != "reduced"
        ):
            raise ValueError("Both concatenated RHO seeds must use reduced mechanics.")

        prefix_keys = {
            key for key in prefix_data.files
            if key != "metadata__json"
            and not key.startswith(("applied_pulse_widths__", "physical__"))
        }
        extension_keys = {
            key for key in extension_data.files
            if key != "metadata__json"
            and not key.startswith(("applied_pulse_widths__", "physical__"))
        }
        if prefix_keys != extension_keys:
            raise ValueError("The RHO prefix and extension variables do not match.")

        payload: dict[str, np.ndarray] = {}
        maximum_boundary_change = 0.0
        for key in sorted(prefix_keys):
            prefix = np.asarray(prefix_data[key])
            extension = np.asarray(extension_data[key])
            if prefix.shape[:-1] != extension.shape[:-1]:
                raise ValueError(f"RHO seed '{key}' has incompatible row dimensions.")
            if key.startswith("states__"):
                intervals, remainder = divmod(prefix.shape[-1] - 1, prefix_cycles)
                if remainder or extension.shape[-1] != intervals + 1:
                    raise ValueError(f"RHO state '{key}' has incompatible cycle nodes.")
                stitched_prefix = prefix.copy()
                maximum_boundary_change = max(
                    maximum_boundary_change,
                    float(np.max(np.abs(stitched_prefix[..., -1] - extension[..., 0]))),
                )
                stitched_prefix[..., -1] = extension[..., 0]
                payload[key] = np.concatenate(
                    (stitched_prefix, extension[..., 1:]), axis=-1
                )
            elif key.startswith("controls__"):
                nodes, remainder = divmod(prefix.shape[-1], prefix_cycles)
                if remainder or extension.shape[-1] != nodes:
                    raise ValueError(
                        f"RHO control '{key}' has incompatible cycle nodes."
                    )
                payload[key] = np.concatenate((prefix, extension), axis=-1)
            else:
                if prefix.shape != extension.shape or not np.array_equal(
                    prefix, extension
                ):
                    raise ValueError(f"RHO auxiliary value '{key}' does not match.")
                payload[key] = prefix

    metadata = dict(prefix_metadata)
    metadata.update(
        {
            "cycles_per_window": prefix_cycles + 1,
            "producer_mode": "full_horizon_plus_terminal_rho",
            "producer_prefix_cycles": prefix_cycles,
            "producer_extension_cycles": 1,
            "replaced_reduced_boundary_maximum_absolute_change": (
                maximum_boundary_change
            ),
        }
    )
    metadata = _refresh_physical_archive(payload, metadata)
    payload["metadata__json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return metadata


def _terminal_handoff_error(source_path: Path, extension_path: Path) -> float:
    """Audit every first-node state, including fatigue and stimulation memory."""
    with np.load(source_path, allow_pickle=False) as source, np.load(
        extension_path, allow_pickle=False
    ) as extension:
        keys = {key for key in source.files if key.startswith("states__")}
        if not keys or keys != {key for key in extension.files if key.startswith("states__")}:
            return math.inf
        maximum = 0.0
        for key in keys:
            terminal = np.asarray(source[key], dtype=float)[..., -1]
            initial = np.asarray(extension[key], dtype=float)[..., 0]
            if terminal.shape != initial.shape or not np.all(np.isfinite(initial)):
                return math.inf
            error = float(np.max(np.abs(initial - terminal)))
            if not math.isfinite(error):
                return math.inf
            maximum = max(maximum, error)
        return maximum


def _linear_solver_for(solver: str) -> str:
    """Return the benchmark's linear-solver contract for an NLP backend."""

    return "ma57" if solver == "ipopt" else "mumps"


def _benchmark_success(
    result_path: Path,
    *,
    expected_mode: str,
    expected_cycles: int,
    expected_solver: str,
    expected_mechanics: str | None = None,
) -> bool:
    """Require a solver and physical certificate for the complete horizon."""

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = payload["results"][0]
        configuration = payload["configurations"][expected_solver]
        if expected_mechanics is None:
            expected_mechanics = "full" if expected_mode == "single_shot" else "reduced"
        expected_use_sx = expected_solver == "ipopt" and expected_mode != "single_shot"
        return bool(
            result["success"]
            and result.get("solver_success") is True
            and result.get("physical_success") is True
            and result.get("solver") == expected_solver
            and result.get("mode") == expected_mode
            and int(result.get("covered_cycles") or 0) == expected_cycles
            and int(result.get("physically_validated_cycles") or 0) == expected_cycles
            and configuration.get("single_shot") is (expected_mode == "single_shot")
            and configuration.get("mechanical_formulation") == expected_mechanics
            and int(configuration.get("cycles_per_window") or 0)
            == (expected_cycles if expected_mode == "single_shot" else 1)
            and int(configuration.get("n_windows") or 0) == expected_cycles
            and configuration.get("use_sx") is expected_use_sx
            and configuration.get(f"{expected_solver}_linear_solver")
            == _linear_solver_for(expected_solver)
        )
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def _benchmark_payload_is_readable(result_path: Path) -> bool:
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = payload["results"][0]
        return isinstance(result, dict) and result.get("error") is None
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def _benchmark_window_objective(result_path: Path | None) -> float | None:
    """Read the additive objective reported by one benchmark result."""

    if result_path is None:
        return None
    try:
        payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
        value = float(payload["results"][0]["window_objective_sum"])
        return value if math.isfinite(value) else None
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def _rho_extension_success(result_path: Path) -> bool:
    """Certify one solved physical cycle independently of endurance semantics.

    A one-cycle extension is a warm-start construction, not an endurance
    campaign.  The global endurance verdict may therefore reject a physically
    validated cycle merely because a Ding capacity decreased.  The extension
    remains usable when IPOPT converged and its sole window is certified.
    """

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = payload["results"][0]
        configuration = payload["configurations"]["ipopt"]
        windows = result.get("windows") or []
        return bool(
            result.get("solver_success") is True
            and result.get("solver") == "ipopt"
            and result.get("mode") == "rho"
            and int(result.get("covered_cycles") or 0) == 1
            and int(result.get("physically_validated_cycles") or 0) == 1
            and len(windows) == 1
            and windows[0].get("validated") is True
            and configuration.get("single_shot") is False
            and configuration.get("mechanical_formulation") == "reduced"
            and int(configuration.get("cycles_per_window") or 0) == 1
            and int(configuration.get("n_windows") or 0) == 1
            and configuration.get("use_sx") is True
            and configuration.get("ipopt_linear_solver") == "ma57"
        )
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def _log_has_unknown_mumps_warning(log_path: str | Path) -> bool:
    log_path = Path(log_path)
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "libMAD WARNING: option linear_solver is of unknown type mumps" in text


def _benchmark_validated_cycles(
    result_path: Path,
    *,
    expected_mode: str,
    expected_solver: str,
    expected_requested_cycles: int,
) -> int:
    """Return the complete solver/physical prefix reported for one backend."""

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = payload["results"][0]
        configuration = payload["configurations"][expected_solver]
        if (
            result.get("solver") != expected_solver
            or result.get("mode") != expected_mode
            or configuration.get("single_shot") is not (expected_mode == "single_shot")
            or configuration.get("mechanical_formulation") != "reduced"
            or int(configuration.get("cycles_per_window") or 0) != 1
            or int(configuration.get("n_windows") or 0) != expected_requested_cycles
            or configuration.get("use_sx") is not True
            or configuration.get(f"{expected_solver}_linear_solver")
            != _linear_solver_for(expected_solver)
        ):
            return 0
        covered = int(result.get("covered_cycles") or 0)
        physical = int(result.get("physically_validated_cycles") or 0)
        return min(covered, physical)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return 0


def _rho_prefix_stop_assessment(result_path: Path) -> dict:
    """Keep a RHO seed ceiling distinct from a physiological verdict."""

    unknown = {
        "classification": "unknown",
        "physiological_limit_certified": False,
        "grid_or_transfer_diagnostic_present": False,
        "solver_or_numerical_failure_present": False,
        "failed_cycle": None,
        "evidence": ["unreadable_rho_result"],
    }
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = payload["results"][0]
        diagnostics = result.get("physical_crank_diagnostics") or {}
        issues = [str(issue) for issue in diagnostics.get("issues") or []]
        grid_issue = any("wheel_cycle_grid" in issue for issue in issues)
        failed_window = next(
            (
                window
                for window in reversed(result.get("windows") or [])
                if window.get("validated") is False
            ),
            None,
        )
        solver_failed = bool(
            failed_window
            and (
                failed_window.get("solver_converged") is False
                or failed_window.get("primal_feasible") is False
            )
        )
        fatigue = result.get("fatigue_endurance_outcome") or {}
        fatigue_accepted = fatigue.get("accepted") is True
        first_failed = result.get("first_failed_rho")
        failed_cycle = (
            int(first_failed)
            if isinstance(first_failed, (int, float))
            else None
        )
        evidence = list(issues)
        if failed_window:
            evidence.extend(
                [
                    "failed_window_solver_converged="
                    f"{failed_window.get('solver_converged')}",
                    "failed_window_primal_feasible="
                    f"{failed_window.get('primal_feasible')}",
                ]
            )
        if solver_failed:
            classification = "numerical_or_optimization_failure"
        elif grid_issue:
            classification = "grid_or_transfer_diagnostic"
        elif fatigue_accepted:
            classification = "physiological_endurance_limit"
        else:
            classification = "inconclusive"
        return {
            "classification": classification,
            "physiological_limit_certified": bool(
                classification == "physiological_endurance_limit"
            ),
            "grid_or_transfer_diagnostic_present": grid_issue,
            "solver_or_numerical_failure_present": solver_failed,
            "failed_cycle": failed_cycle,
            "evidence": evidence,
        }
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return unknown


def _seed_cycle_count(seed_path: Path) -> int:
    try:
        with np.load(seed_path, allow_pickle=False) as data:
            return int(_load_metadata(data)["cycles_per_window"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def _common_solver_options(
    args: argparse.Namespace, *, standard_warmup_seed: bool = True
) -> list[str]:
    torque_option = (
        ["--signed-crank-torque", str(args.signed_crank_torque)]
        if getattr(args, "signed_crank_torque", None) is not None
        else ["--crank-assistance", str(args.crank_assistance)]
    )
    warmup_options = (
        [
            "--standard-warmup-seed",
            str(args.workspace / ".github/benchmark-seeds/legacy-resistive-0p22-warmup.npz"),
            "--legacy-standard-warmup-seed-signed-torque",
            "0.22",
            "--standard-warmup-seed-continuation",
        ]
        if standard_warmup_seed else []
    )
    return [
        "--objective",
        "fatigue",
        "--ipopt-profile",
        "periodic_collocation",
        # Keep the non-strict periodic profile so the monolithic FHO can use
        # MX, but make Radau-5 the shared transcription for the reference RHO
        # and for both IPOPT and MadNLP full-horizon solves.
        "--ipopt-collocation-degree",
        "5",
        "--ipopt-collocation-method",
        "radau",
        "--ipopt-enforce-start-constraints",
        "--stimulations-per-cycle",
        "30",
        "--n-threads",
        str(args.n_threads),
        *torque_option,
        "--nlp-tolerance",
        "1e-8",
        "--primal-feasibility-threshold",
        "1e-5",
        *warmup_options,
        "--warmup-ipopt-linear-solver",
        "ma57",
        "--ipopt-linear-solver",
        "ma57",
        "--madnlp-linear-solver",
        "mumps",
        "--madnlp-max-iter",
        str(args.max_iterations),
        "--ipopt-disable-historical-initial-guess",
        "--reduced-cycling-profile",
        str(args.seed_dir / "reduced-cycling-fourier12.npz"),
        "--state-scaling",
        "full",
        "--first-node-wheel-q-slack",
        "0",
        "--terminal-wheel-q-slack",
        str(args.terminal_wheel_q_slack),
        "--compact-rho-output",
        "--print-traces",
    ]


def _rho_command(
    args: argparse.Namespace,
    result_path: Path,
    seed_path: Path,
    *,
    n_windows: int | None = None,
    common_initial_solution: Path | None = None,
) -> list[str]:
    if n_windows is None:
        n_windows = args.max_cycles
    if n_windows < 1:
        raise ValueError("n_windows must be strictly positive.")
    is_terminal_extension = common_initial_solution is not None
    if common_initial_solution is None:
        common_initial_solution = args.seed_dir / "common-reduced.npz"
    command = [
        args.python,
        str(
            args.workspace
            / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"
        ),
        "--solvers",
        "ipopt",
        *_common_solver_options(args, standard_warmup_seed=not is_terminal_extension),
    ]
    if is_terminal_extension:
        # A local continuation already carries the certified chronology and
        # terminal fatigue; a fresh warmup would change its initial problem.
        command.append("--ipopt-disable-standard-warmup")
    # The RHO transcription stays SX and its numerical initial state is an
    # input (``x0``), not part of the generated NLP.  The reference RHO keeps
    # its in-process compilation, while the one-cycle homotopy extensions use
    # a persistent content-addressed cache: the first child process builds the
    # native evaluator and later children load that same library despite using
    # different warm-start states.  Do not apply this to FHO: its MX graph is
    # intentionally interpreted, without C compilation.
    if n_windows > 1:
        command.append("--ipopt-c-compile")
    else:
        cache_dir = Path(
            getattr(
                args,
                "rho_transition_c_cache_dir",
                Path(args.output_dir) / "rho-transition-ipopt-c-cache",
            )
        )
        command.extend(
            [
                "--ipopt-c-compile",
                "--ipopt-c-cache-dir",
                str(cache_dir),
                "--ipopt-c-cache-name",
                "rho-transition-sx",
            ]
        )
    command.extend(
        [
            "--ipopt-use-sx",
            "--no-optional-nlp-periodic-ipopt-hot-start",
            "--ipopt-enable-periodic-fes-warmup-projection",
            "--periodic-fes-warmup-projection-strategy",
            "rollout",
            "--initial-guess-diagnostics",
            "--ipopt-max-iter",
            str(args.max_iterations),
            "--cycles-per-window",
            "1",
            "--n-windows",
            str(n_windows),
            "--max-consecutive-failing",
            "1",
            # A failed RHO primal is not a valid state transition. Keeping the
            # last certified first node makes the reported RHO ceiling usable
            # as an FHO seed boundary instead of exporting a shifted failed
            # terminal state.
            "--retry-failed-rho-without-advance",
            "--mechanical-formulation",
            "reduced",
            "--common-initial-solution",
            str(common_initial_solution),
            "--common-initial-solution-recenter-first-node-bounds",
            "--adopt-common-initial-solution-warmup-cycles",
            "--receding-horizon-solution-output",
            str(seed_path),
            "--allow-partial-receding-horizon-solution-output",
            "--output-json",
            str(result_path),
        ]
    )
    if getattr(args, "reduced_terminal_half_step_velocity_guard", False):
        # This is intentionally limited to the one-cycle RHO handoffs.  It
        # makes the state inherited by the next RHO feasible for the first
        # frozen half interval, without changing the MX full-horizon NLP.
        command.append("--reduced-terminal-half-step-velocity-guard")
    return command


def _full_horizon_command(
    args: argparse.Namespace,
    cycles: int,
    seed_path: Path,
    result_path: Path,
    solution_path: Path,
    *,
    mechanical_formulation: str = "reduced",
    prefix_solution_path: Path | None = None,
) -> list[str]:
    if mechanical_formulation not in {"full", "reduced"}:
        raise ValueError("mechanical_formulation must be 'full' or 'reduced'.")
    full_horizon_solver = getattr(args, "full_horizon_solver", "ipopt")
    command = [
        args.python,
        str(
            args.workspace
            / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"
        ),
        "--solvers",
        full_horizon_solver,
        *_common_solver_options(args, standard_warmup_seed=False),
        "--ipopt-no-use-sx",
        "--ipopt-max-iter",
        str(args.max_iterations),
    ]
    # Every FHO consumes an explicit certified RHO/FHO seed, including FHO_1.
    # Its warmup chronology belongs to that seed, not to the horizon length.
    # Running a fresh warmup here either changes the initial problem or rejects
    # the seed's warmup_cycles_consumed before the numerical solve even starts.
    command.extend(
        [
            "--ipopt-disable-standard-warmup",
            "--adopt-common-initial-solution-warmup-cycles",
        ]
    )
    command.extend(
        [
            "--optional-nlp-periodic-ipopt-hot-start",
            # Store the block/FES/RK4 seed defects for MadNLP as well as ACADOS.
            # These diagnostics distinguish a continuous RHO/FHO handoff from a
            # dynamically infeasible monolithic transcription.
            "--initial-guess-diagnostics",
            "--periodic-ipopt-refinement-use-sx",
            "--periodic-ipopt-refinement-iterations",
            str(args.max_iterations),
            "--single-shot",
            "--cycles-per-window",
            str(cycles),
            "--n-windows",
            str(cycles),
            "--mechanical-formulation",
            mechanical_formulation,
            "--full-contact-position-tolerance",
            "2e-5",
            "--common-initial-solution",
            str(seed_path),
            "--common-initial-solution-output",
            str(solution_path),
            "--output-json",
            str(result_path),
        ]
    )
    if full_horizon_solver == "madnlp":
        command.append("--exact-initial-nlp-audit")
    if prefix_solution_path is not None:
        command.extend(["--full-horizon-prefix-solution", str(prefix_solution_path)])
    return command


def _seed_handoff_error(result_path: Path, log_path: Path) -> str | None:
    """Identify explicit seed contract exceptions without inferring fatigue."""

    messages: list[str] = []
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        error = payload["results"][0].get("error")
        if error is not None:
            messages.append(str(error))
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        pass
    try:
        # Normal diagnostic lines mention the seed too; only exception lines
        # indicate a failed contract.
        messages.extend(
            line
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.startswith(("ValueError:", "KeyError:", "FileNotFoundError:"))
        )
    except OSError:
        pass
    for message in messages:
        if any(
            marker in message.lower()
            for marker in (
                "common initial solution",
                "full-horizon prefix",
                "warmup_cycles_consumed",
            )
        ):
            return message
    return None


def _attempt_record(
    cycles: int,
    phase: str,
    monitored: MonitoredRun,
    result_path: Path,
    *,
    mechanical_formulation: str = "reduced",
    solution_path: Path | None = None,
    prefix_solution_path: Path | None = None,
    expected_solver: str = "ipopt",
) -> dict:
    unknown_mumps_warning = _log_has_unknown_mumps_warning(Path(monitored.log_path))
    handoff_error = _seed_handoff_error(result_path, Path(monitored.log_path))
    certificate = _benchmark_success(
        result_path,
        expected_mode="single_shot",
        expected_cycles=cycles,
        expected_solver=expected_solver,
        expected_mechanics=mechanical_formulation,
    )
    solution_available = solution_path is None or solution_path.is_file()
    infrastructure_error = bool(
        not monitored.memory_limit_exceeded
        and not monitored.timed_out
        and (
            monitored.return_code != 0
            or not _benchmark_payload_is_readable(result_path)
            or unknown_mumps_warning
            or handoff_error is not None
            or (certificate and not solution_available)
        )
    )
    success = bool(
        certificate
        and solution_available
        and monitored.return_code == 0
        and not monitored.memory_limit_exceeded
        and not monitored.timed_out
        and not infrastructure_error
    )
    failure_kind = (
        None
        if success
        else (
            "memory_limit"
            if monitored.memory_limit_exceeded
            else (
                "timeout"
                if monitored.timed_out
                else (
                    "seed_handoff_error"
                    if handoff_error is not None
                    else (
                        "infrastructure_error" if infrastructure_error else "solver_failure"
                    )
                )
            )
        )
    )
    return {
        "cycles": cycles,
        "mechanical_formulation": mechanical_formulation,
        "phase": phase,
        "success": success,
        "failure_kind": failure_kind,
        "failure_class": (
            None if success else "resource_limit" if monitored.memory_limit_exceeded or monitored.timed_out
            else "handoff_failure" if handoff_error is not None
            else "infrastructure_error" if infrastructure_error else "numerical_unresolved"
        ),
        "certificate_valid": certificate,
        "solution_available": solution_available,
        "infrastructure_error": infrastructure_error,
        "seed_handoff_error": handoff_error,
        "physiological_limit_certified": False,
        "unknown_mumps_warning": unknown_mumps_warning,
        "result_path": str(result_path),
        "solution_path": None if solution_path is None else str(solution_path),
        "seed_origin": (
            "rho_plus_certified_fho_prefix"
            if prefix_solution_path is not None
            else "rho_prefix"
        ),
        "prefix_solution_path": (
            None if prefix_solution_path is None else str(prefix_solution_path)
        ),
        "peak_rss_bytes": monitored.peak_rss_bytes,
        "peak_rss_gib": monitored.peak_rss_bytes / GIB,
        "return_code": monitored.return_code,
        "elapsed_s": monitored.elapsed_s,
        "memory_limit_exceeded": monitored.memory_limit_exceeded,
        "timed_out": monitored.timed_out,
        "log_path": monitored.log_path,
    }


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, report: dict) -> None:
    prefix_assessment = report.get("rho_prefix_stop_assessment") or {}
    prefix_classification = prefix_assessment.get("classification", "unknown")
    lines = [
        "# RHO reduced vs full-horizon size homotopy",
        "",
        f"- Limite RSS : `{report['rss_limit_gib']:.3f} GiB`",
        f"- Plafond demandé : `{report['max_cycles']} cycles`",
        f"- RHO de référence disponibles : `{report['rho_available_cycles']} cycles`",
        (
            "- Pas adaptatif demandé : "
            f"`+{report.get('continuation_step_cycles', 1)} cycles`"
        ),
        (
            "- Tolérance objective des sauts : "
            f"`{100 * report.get('jump_objective_relative_tolerance', 0.0):.3f} %`"
        ),
        f"- Replis +1 : `{len(report.get('adaptive_fallback_events', []))}`",
        f"- Chaîne RHO/FHO construite : `{report.get('homotopy_constructed_cycles', 0)} cycles`",
        f"- Plus grand full horizon validé : `{report['largest_successful_cycles']}`",
        f"- Trous de convergence : `{report.get('solver_gap_cycles', [])}`",
        (
            "- Diagnostic du plafond de graine RHO : "
            f"`{prefix_classification}` "
            "(ce diagnostic n'est pas, à lui seul, une limite physiologique)"
        ),
        (
            "- Initialisation : `FHO_N certifié + un RHO résolu depuis son état "
            "terminal`"
        ),
        (
            "- Amorçage reduced à deux RHO : "
            f"`{'succès' if report['rho']['success'] else 'échec'}`, "
            f"pic RSS `{report['rho']['peak_rss_gib']:.3f} GiB`, "
            f"temps `{report['rho']['elapsed_s']:.1f} s`"
        ),
        f"- Arrêt : `{report['stop_reason']}`",
        "",
        "## Extensions RHO depuis le terminal FHO",
        "",
    ]
    extension_attempts = report.get("extension_rho_attempts", [])
    if extension_attempts:
        lines.extend(
            [
                "| Source | RHO cible | Saut FHO | Succès | Échec | Pic RSS (GiB) | Temps (s) |",
                "|:---|---:|---:|:---:|:---|---:|---:|",
            ]
        )
        for extension in extension_attempts:
            source_label = extension.get("source_label") or (
                f"FHO_{extension['after_full_horizon_cycles']}"
            )
            lines.append(
                f"| {source_label} | "
                f"{extension['target_cycle']} | "
                f"+{extension.get('adaptive_step_cycles', 1)} | "
                f"{'oui' if extension['success'] else 'non'} | "
                f"{extension.get('failure_kind') or '—'} | "
                f"{extension['peak_rss_gib']:.3f} | "
                f"{extension['elapsed_s']:.1f} |"
            )
        lines.append("")
    else:
        lines.extend(["- Aucune extension nécessaire ou atteinte.", ""])
    lines.extend(
        [
            "## Sweep FHO reduced/MX",
            "",
            "| Passage | Phase | Seed | Convergé | Promu | Δ objectif/seed | Échec | Pic RSS (GiB) | Temps (s) |",
            "|:---|:---|:---|:---:|:---:|---:|:---|---:|---:|",
        ]
    )
    for attempt in report["full_horizon_attempts"]:
        objective_degradation = (attempt.get("objective_gate") or {}).get(
            "relative_degradation"
        )
        objective_cell = (
            "—"
            if objective_degradation is None
            else f"{100 * objective_degradation:.3f} %"
        )
        lines.append(
            f"| {attempt.get('adaptive_source_cycles', 0)}→{attempt['cycles']} "
            f"(+{attempt.get('adaptive_step_cycles', attempt['cycles'])}) | "
            f"{attempt['phase']} | "
            f"{attempt.get('seed_origin', 'rho_prefix')} | "
            f"{'oui' if attempt['success'] else 'non'} | "
            f"{'oui' if attempt.get('accepted_for_continuation', attempt['success']) else 'non'} | "
            f"{objective_cell} | "
            f"{attempt.get('failure_kind') or '—'} | "
            f"{attempt['peak_rss_gib']:.3f} | {attempt['elapsed_s']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_memory_limit(raw: str, total_memory: int) -> float:
    if raw.lower() == "auto":
        return automatic_rss_limit_gib(total_memory)
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("--memory-limit-gib must be 'auto' or a positive number.")
    if value * GIB >= total_memory:
        raise ValueError(
            "--memory-limit-gib must leave headroom below the detected allocation."
        )
    return value


def _parse_attempt_timeout(raw: str) -> float | None:
    """Parse a positive timeout, or the explicit unlimited ``none`` value."""

    if raw.strip().lower() == "none":
        return None
    try:
        value = float(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a strictly positive number or 'none'"
        ) from error
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(
            "must be a strictly positive finite number or 'none'"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cycles", type=int, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume the adaptive continuation from the largest certified FHO in "
            "--output-dir, reusing its RHO seed and solution artifacts."
        ),
    )
    parser.add_argument("--memory-limit-gib", default="auto")
    parser.add_argument("--n-threads", type=int, required=True)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument(
        "--bootstrap-rho-prefix-cycles",
        type=int,
        choices=(2, 3),
        default=2,
        help="Build the first FHO directly from 2 or 3 certified RHO reference cycles.",
    )
    parser.add_argument(
        "--continuation-step-cycles",
        type=int,
        default=1,
        help=(
            "Preferred number of RHO cycles appended before the next FHO. "
            "A failed multi-cycle jump is retried with one cycle."
        ),
    )
    parser.add_argument(
        "--jump-objective-relative-tolerance",
        type=float,
        default=0.005,
        help=(
            "Maximum relative objective degradation of a multi-cycle FHO "
            "against its additive FHO+RHO seed before falling back to +1."
        ),
    )
    parser.add_argument(
        "--allow-fho-terminal-bridge",
        action="store_true",
        help=(
            "When a one-cycle RHO extension cannot be certified, try the next "
            "MX FHO directly from the certified FHO terminal state instead of "
            "treating the RHO prefix ceiling as an FHO ceiling."
        ),
    )
    parser.add_argument(
        "--full-horizon-solver",
        choices=("madnlp", "ipopt"),
        default="ipopt",
        help="NLP solver used for the reduced/MX monolithic FHO problems.",
    )
    torque_group = parser.add_mutually_exclusive_group()
    torque_group.add_argument(
        "--crank-assistance",
        type=float,
        default=0.0,
        help="Non-negative assistance magnitude in N.m (the historical default).",
    )
    torque_group.add_argument(
        "--signed-crank-torque",
        type=float,
        default=None,
        help=(
            "Signed crank torque in N.m; a positive value is resistive for "
            "the nominal negative crank velocity."
        ),
    )
    parser.add_argument("--terminal-wheel-q-slack", type=float, default=0.002)
    parser.add_argument(
        "--reduced-terminal-half-step-velocity-guard",
        action="store_true",
        help=(
            "Apply the optional reduced cadence half-step guard to terminal "
            "RHO nodes used to extend a certified FHO."
        ),
    )
    parser.add_argument(
        "--rho-only",
        action="store_true",
        help=(
            "solve and certify only the concatenated RHO reference; skip all "
            "full-horizon attempts"
        ),
    )
    parser.add_argument(
        "--rho-extension-only",
        action="store_true",
        help=(
            "With --resume, extend the last certified FHO using only sequential "
            "one-cycle RHO handoffs.  This is a feasibility probe: it never "
            "constructs or solves a new full-horizon NLP."
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-interval-s", type=float, default=0.5)
    parser.add_argument(
        "--attempt-timeout-s",
        type=_parse_attempt_timeout,
        default=None,
        metavar="SECONDS|none",
        help="Wall-time cap per solver attempt; 'none' disables it (default).",
    )
    return parser


def _resume_run(
    args: argparse.Namespace,
    *,
    report_path: Path,
    markdown_path: Path,
    total_memory: int,
    rss_limit_gib: float,
    rss_limit_bytes: int,
) -> int:
    """Continue an interrupted sweep without rebuilding its certified prefix."""

    if not report_path.is_file():
        raise FileNotFoundError(f"Cannot resume without {report_path}.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    relocated_artifact_paths = rebase_report_artifact_paths(report, args.output_dir)
    if report.get("schema") != "cocofest-full-horizon-sweep-v2":
        raise ValueError("The existing report has an unsupported schema.")
    if report.get("full_horizon_solver") != args.full_horizon_solver:
        raise ValueError(
            "The resumed --full-horizon-solver must match the existing report."
        )
    if int(report.get("bootstrap_rho_prefix_cycles") or 2) != args.bootstrap_rho_prefix_cycles:
        raise ValueError("The resumed bootstrap RHO prefix must match the existing report.")
    if report.get("allow_fho_terminal_bridge", False):
        args.allow_fho_terminal_bridge = True

    rho = report.get("rho") or {}
    rho_seed_path = Path(str(rho.get("seed_path", "")))
    rho_available_cycles = int(report.get("rho_available_cycles") or 0)
    largest_successful_cycles = int(report.get("largest_successful_cycles") or 0)
    if largest_successful_cycles < 2 and (
        not rho_seed_path.is_file() or _seed_cycle_count(rho_seed_path) != rho_available_cycles
    ):
        raise ValueError("The bootstrap concatenated RHO seed is unavailable or inconsistent.")
    successful_attempts = [
        attempt
        for attempt in report.get("full_horizon_attempts", [])
        if attempt.get("success")
        and attempt.get("accepted_for_continuation", True)
        and int(attempt.get("cycles") or 0) == largest_successful_cycles
    ]
    if largest_successful_cycles >= 2 and not successful_attempts:
        raise ValueError("The report does not identify the last certified FHO solution.")
    current_full_solution = (
        Path(successful_attempts[-1]["solution_path"])
        if largest_successful_cycles >= 2
        else None
    )
    if current_full_solution is not None and not current_full_solution.is_file():
        raise FileNotFoundError(f"Missing certified FHO solution: {current_full_solution}")

    # The reference is an optional template after the certified FHO bootstrap.
    # Its numerical stopping point must not cap autonomous post-FHO RHO solves.
    effective_max_cycles = args.max_cycles
    if effective_max_cycles < largest_successful_cycles:
        raise ValueError("--max-cycles cannot be below the certified resume point.")
    previous_stop_reason = report.get("stop_reason")
    report.update(
        {
            "max_cycles": args.max_cycles,
            "effective_max_cycles": effective_max_cycles,
            "total_memory_bytes": total_memory,
            "total_memory_gib": total_memory / GIB,
            "rss_limit_bytes": rss_limit_bytes,
            "rss_limit_gib": rss_limit_gib,
            "continuation_step_cycles": args.continuation_step_cycles,
            "jump_objective_relative_tolerance": (
                args.jump_objective_relative_tolerance
            ),
            "allow_fho_terminal_bridge": args.allow_fho_terminal_bridge,
            "initialization": "autonomous_certified_terminal_rho_chain",
            "stop_class": None,
            "stop_reason": "running",
        }
    )
    report.setdefault("adaptive_fallback_events", [])
    report.setdefault("resume_events", []).append(
        {
            "from_cycles": largest_successful_cycles,
            "previous_stop_reason": previous_stop_reason,
            "output_dir": str(args.output_dir),
            "relocated_artifact_paths": relocated_artifact_paths,
        }
    )
    _write_report(report_path, report)

    if largest_successful_cycles < args.bootstrap_rho_prefix_cycles:
        bootstrap_function = (
            _bootstrap_direct_three_cycle_horizon
            if args.bootstrap_rho_prefix_cycles == 3
            else _bootstrap_full_horizon
        )
        bootstrap = bootstrap_function(
            args,
            report=report,
            report_path=report_path,
            markdown_path=markdown_path,
            rho_seed_path=rho_seed_path,
            rss_limit_bytes=rss_limit_bytes,
        )
        if not bootstrap["success"]:
            return 3 if bootstrap["infrastructure_error"] else 0
        largest_successful_cycles = args.bootstrap_rho_prefix_cycles
        current_full_solution = Path(bootstrap["solution_path"])

    if args.rho_extension_only:
        return _probe_rho_extensions(
            args,
            report=report,
            report_path=report_path,
            markdown_path=markdown_path,
            rho_seed_path=rho_seed_path,
            effective_max_cycles=effective_max_cycles,
            current_cycles=largest_successful_cycles,
            current_solution=current_full_solution,
            rss_limit_bytes=rss_limit_bytes,
        )

    return _continue_adaptively(
        args,
        report=report,
        report_path=report_path,
        markdown_path=markdown_path,
        rho_seed_path=rho_seed_path,
        effective_max_cycles=effective_max_cycles,
        current_cycles=largest_successful_cycles,
        current_full_solution=current_full_solution,
        rss_limit_bytes=rss_limit_bytes,
    )


def _relocate_report_artifact(raw_path: str, output_dir: Path) -> Path | None:
    """Locate a moved campaign artifact while preserving its internal layout."""

    original = Path(raw_path)
    if original.is_file():
        return original
    anchors = (
        "rho-reduced",
        "homotopy-seeds",
        "adaptive-attempts",
        "paired-reduced-controls",
    )
    parts = original.parts
    for index, part in enumerate(parts):
        if part in anchors or part.startswith(
            ("full-horizon-", "rho-extension-after-fho-")
        ):
            candidate = output_dir.joinpath(*parts[index:])
            if candidate.is_file():
                return candidate
    return None


def rebase_report_artifact_paths(report: dict, output_dir: Path) -> int:
    """Rewrite stale absolute artifact paths after a campaign directory move."""

    relocated = 0

    def visit(value) -> None:
        nonlocal relocated
        if isinstance(value, dict):
            for key, child in value.items():
                if key.endswith("_path") and isinstance(child, str) and child:
                    replacement = _relocate_report_artifact(child, output_dir)
                    if replacement is not None and replacement != Path(child):
                        value[key] = str(replacement)
                        relocated += 1
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(report)
    return relocated


def _run_horizon_attempt(
    args: argparse.Namespace,
    *,
    rho_seed: Path,
    cycles: int,
    phase: str,
    chance: int,
    rss_limit_bytes: int,
    mechanical_formulation: str = "reduced",
    prefix_solution_path: Path | None = None,
    heartbeat_seed_label: str | None = None,
    common_initial_solution: Path | None = None,
) -> dict:
    if chance < 1:
        raise ValueError("chance must be strictly positive.")
    case_prefix = "full-horizon"
    case_dir = args.output_dir / f"{case_prefix}-{cycles:04d}" / f"chance-{chance}"
    seed_path = case_dir / (
        "rho-reduced-prefix.npz"
        if common_initial_solution is None else "common-initial-solution.npz"
    )
    result_path = case_dir / "result.json"
    solution_path = case_dir / "full-solution.npz"
    for stale_path in (result_path, solution_path):
        stale_path.unlink(missing_ok=True)
    if common_initial_solution is None:
        write_rho_seed_prefix(rho_seed, seed_path, cycles)
    else:
        if cycles != 1 or prefix_solution_path is not None:
            raise ValueError("The common initial solution bootstrap is only valid for FHO_1.")
        if _seed_cycle_count(common_initial_solution) != 1:
            raise ValueError("The common initial solution must contain exactly one certified cycle.")
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        # Snapshot the original certified primal exactly. RHO_1 re-solves this
        # cycle and exports a different trajectory, which can place FHO_1 in a
        # different basin even though the initial state/chronology agree.
        shutil.copyfile(common_initial_solution, seed_path)
    monitored = run_monitored(
        _full_horizon_command(
            args,
            cycles,
            seed_path,
            result_path,
            solution_path,
            mechanical_formulation=mechanical_formulation,
            prefix_solution_path=prefix_solution_path,
        ),
        cwd=args.workspace,
        log_path=case_dir / "solver.log",
        rss_limit_bytes=rss_limit_bytes,
        heartbeat_label=(
            f"FHO_{cycles} solver={getattr(args, 'full_horizon_solver', 'ipopt')} "
            f"seed={heartbeat_seed_label or f'FHO_{cycles - 1}+RHO_{cycles}'}"
        ),
        poll_interval_s=args.poll_interval_s,
        timeout_s=args.attempt_timeout_s,
    )
    record = _attempt_record(
        cycles,
        phase,
        monitored,
        result_path,
        mechanical_formulation=mechanical_formulation,
        solution_path=solution_path,
        prefix_solution_path=prefix_solution_path,
        expected_solver=getattr(args, "full_horizon_solver", "ipopt"),
    )
    record["seed_source_path"] = str(
        rho_seed if common_initial_solution is None else common_initial_solution
    )
    if common_initial_solution is not None:
        record["seed_origin"] = "common_initial_solution"
    return record


def _run_extension_rho(
    args: argparse.Namespace,
    *,
    source_full_solution: Path,
    reference_rho_seed: Path,
    after_cycles: int,
    rss_limit_bytes: int,
    source_label: str | None = None,
    run_number: int = 1,
) -> dict:
    """Solve a local RHO at the carrier terminal, then recover by homotopy.

    The carrier is either the certified FHO or the preceding certified local
    RHO. Its last cycle is sufficient; a long reference is only a recovery
    template. Intermediate homotopy solutions never extend the FHO prefix.
    """

    if run_number < 1:
        raise ValueError("run_number must be strictly positive.")
    case_dir = args.output_dir / f"rho-extension-after-fho-{after_cycles:04d}"
    if run_number > 1:
        case_dir = case_dir / f"retry-{run_number:02d}"
    local_seed_path = case_dir / "terminal-continuation-seed.npz"
    write_fho_terminal_continuation_seed(source_full_solution, local_seed_path)
    reference_cycle_path = case_dir / "reference-rho-cycle.npz"
    reference_origin = "carrier_last_certified_cycle"
    # Build the autonomous fallback first. A damaged optional reference must
    # not prevent a solve from an intact certified carrier.
    write_rho_seed_cycle(
        source_full_solution, reference_cycle_path, _seed_cycle_count(source_full_solution)
    )
    reference_warning = None
    if reference_rho_seed.is_file():
        try:
            write_rho_seed_cycle(reference_rho_seed, reference_cycle_path, after_cycles + 1)
            reference_origin = "long_rho_reference"
        except (RhoReferenceCycleUnavailable, ValueError, KeyError, OSError) as error:
            reference_warning = str(error)

    accepted_fraction = 0.0
    step = 0.25
    minimum_step = RHO_INITIAL_STATE_HOMOTOPY_MIN_STEP
    source_solution_path = reference_cycle_path
    stages: list[dict] = []
    peak_rss_bytes = 0
    elapsed_s = 0.0
    final_solution_path: Path | None = None
    final_result_path: Path | None = None
    final_seed_path: Path | None = None
    failure_kind = None
    infrastructure_error = False
    unknown_mumps_warning = False
    memory_limit_exceeded = False
    timed_out = False
    return_code = 0
    direct_attempt = True

    while accepted_fraction < 1.0:
        fraction = 1.0 if direct_attempt else min(1.0, accepted_fraction + step)
        attempt_number = len(stages) + 1
        stage_dir = case_dir / f"stage-{attempt_number:02d}-{fraction:.6f}"
        seed_path = stage_dir / "homotopy-seed.npz"
        result_path = stage_dir / "result.json"
        solution_path = stage_dir / "rho-solution.npz"
        for stale_path in (result_path, solution_path):
            stale_path.unlink(missing_ok=True)
        if direct_attempt:
            seed_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_seed_path, seed_path)
            seed_metadata = {"homotopy_maximum_initial_state_change": 0.0}
        else:
            seed_metadata = write_rho_initial_state_homotopy_seed(
                source_solution_path,
                reference_cycle_path,
                source_full_solution,
                seed_path,
                fraction,
            )
        monitored = run_monitored(
            _rho_command(
                args,
                result_path,
                solution_path,
                n_windows=1,
                common_initial_solution=seed_path,
            ),
            cwd=args.workspace,
            log_path=stage_dir / "solver.log",
            rss_limit_bytes=rss_limit_bytes,
            heartbeat_label=(
                f"RHO_{after_cycles + 1} homotopy={fraction:.6f} "
                f"attempt={attempt_number} "
                f"source={source_label or f'FHO_{after_cycles}'}"
            ),
            poll_interval_s=args.poll_interval_s,
            timeout_s=args.attempt_timeout_s,
        )
        certificate = _rho_extension_success(result_path)
        solution_available = bool(
            solution_path.is_file() and _seed_cycle_count(solution_path) == 1
        )
        handoff_error = (
            _terminal_handoff_error(source_full_solution, solution_path)
            if solution_available and fraction == 1.0 else None
        )
        handoff_valid = handoff_error is None or handoff_error <= 1e-5
        stage_unknown_warning = _log_has_unknown_mumps_warning(monitored.log_path)
        stage_infrastructure_error = bool(
            not monitored.memory_limit_exceeded
            and not monitored.timed_out
            and (
                monitored.return_code != 0
                or not _benchmark_payload_is_readable(result_path)
                or stage_unknown_warning
                or (certificate and not solution_available)
            )
        )
        stage_success = bool(
            certificate
            and solution_available
            and handoff_valid
            and not stage_infrastructure_error
            and monitored.return_code == 0
            and not monitored.memory_limit_exceeded
            and not monitored.timed_out
        )
        stage_failure_kind = (
            None
            if stage_success
            else (
                "memory_limit"
                if monitored.memory_limit_exceeded
                else (
                    "timeout"
                    if monitored.timed_out
                    else (
                        "infrastructure_error"
                        if stage_infrastructure_error
                        else "initial_state_mismatch" if not handoff_valid else "solver_failure"
                    )
                )
            )
        )
        stages.append(
            {
                "attempt": attempt_number,
                "recipe": "terminal_projection" if direct_attempt else "initial_state_homotopy",
                "fraction": fraction,
                "step": step,
                "success": stage_success,
                "failure_kind": stage_failure_kind,
                "terminal_handoff_max_abs_error": handoff_error,
                "maximum_initial_state_change": seed_metadata[
                    "homotopy_maximum_initial_state_change"
                ],
                "result_path": str(result_path),
                "solution_path": str(solution_path),
                "seed_path": str(seed_path),
                "log_path": monitored.log_path,
                "peak_rss_gib": monitored.peak_rss_bytes / GIB,
                "elapsed_s": monitored.elapsed_s,
            }
        )
        peak_rss_bytes = max(peak_rss_bytes, monitored.peak_rss_bytes)
        elapsed_s += monitored.elapsed_s
        return_code = monitored.return_code
        unknown_mumps_warning = unknown_mumps_warning or stage_unknown_warning
        infrastructure_error = infrastructure_error or stage_infrastructure_error
        memory_limit_exceeded = memory_limit_exceeded or monitored.memory_limit_exceeded
        timed_out = timed_out or monitored.timed_out
        final_result_path = result_path
        final_seed_path = seed_path

        if stage_success:
            accepted_fraction = fraction
            source_solution_path = solution_path
            final_solution_path = solution_path
            step = min(0.25, step * 2.0)
            continue
        failure_kind = stage_failure_kind
        if (
            stage_infrastructure_error
            or monitored.memory_limit_exceeded
            or monitored.timed_out
            or step / 2.0 < minimum_step
        ):
            break
        if direct_attempt:
            direct_attempt = False
            step = 0.25
            continue
        step /= 2.0

    success = accepted_fraction == 1.0 and final_solution_path is not None
    if success:
        failure_kind = None
    return {
        "after_full_horizon_cycles": after_cycles,
        "target_cycle": after_cycles + 1,
        "run_number": run_number,
        "source_label": source_label or f"FHO_{after_cycles}",
        "success": success,
        "failure_kind": failure_kind,
        "failure_class": (
            None if success else "resource_limit" if memory_limit_exceeded or timed_out
            else "infrastructure_error" if infrastructure_error else "handoff_numerical_unresolved"
        ),
        "physiological_limit_certified": False,
        "certificate_valid": success,
        "solution_available": final_solution_path is not None,
        "infrastructure_error": infrastructure_error,
        "unknown_mumps_warning": unknown_mumps_warning,
        "seed_origin": "certified_terminal_local_rho",
        "provenance": {
            "source_certificate_solution": str(source_full_solution),
            "initial_state_origin": f"{source_label or f'FHO_{after_cycles}'}:terminal",
            "primal_seed_path": str(local_seed_path),
            "dual_policy": "fresh_solver_no_transferred_multipliers",
            "recovery_template_origin": reference_origin,
            "optional_reference_warning": reference_warning,
        },
        "accepted_fraction": accepted_fraction,
        "homotopy_stages": stages,
        "reference_cycle_path": str(reference_cycle_path),
        "continuation_seed_path": (
            None if final_seed_path is None else str(final_seed_path)
        ),
        "continuation_source_cycles": after_cycles,
        "result_path": (None if final_result_path is None else str(final_result_path)),
        "solution_path": (
            None if final_solution_path is None else str(final_solution_path)
        ),
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_gib": peak_rss_bytes / GIB,
        "return_code": return_code,
        "elapsed_s": elapsed_s,
        "memory_limit_exceeded": memory_limit_exceeded,
        "timed_out": timed_out,
        "log_path": None if not stages else stages[-1]["log_path"],
    }


def _certified_fho_objective(report: dict, cycles: int) -> float | None:
    for attempt in reversed(report.get("full_horizon_attempts", [])):
        if (
            attempt.get("success")
            and attempt.get("accepted_for_continuation", True)
            and int(attempt.get("cycles") or 0) == cycles
        ):
            return _benchmark_window_objective(Path(attempt["result_path"]))
    return None


def _next_attempt_number(records: list[dict], cycle_key: str, cycle: int) -> int:
    """Return a non-destructive ordinal for retrying an existing cycle."""

    return 1 + sum(int(record.get(cycle_key) or 0) == cycle for record in records)


def _next_horizon_chance(records: list[dict], cycles: int, output_dir: Path) -> int:
    """Account for both reported and abruptly interrupted FHO attempts."""

    reported = _next_attempt_number(records, "cycles", cycles)
    case_dir = output_dir / f"full-horizon-{cycles:04d}"
    existing = [
        int(path.name.removeprefix("chance-"))
        for path in case_dir.glob("chance-*")
        if path.is_dir() and path.name.removeprefix("chance-").isdigit()
    ]
    return max([reported, *(number + 1 for number in existing)])


def _bootstrap_direct_three_cycle_horizon(
    args: argparse.Namespace,
    *,
    report: dict,
    report_path: Path,
    markdown_path: Path,
    rho_seed_path: Path,
    rss_limit_bytes: int,
) -> dict:
    """Certify FHO_3 from RHO_1..3 before starting +3 continuation.

    This opt-in bootstrap has no FHO_1/2 dependency. It preserves the same
    per-attempt certificate and report contract as the ordinary bootstrap.
    """

    cycles = 3
    if _seed_cycle_count(rho_seed_path) < cycles:
        raise ValueError("Direct FHO_3 bootstrap needs three certified RHO cycles.")
    records = report.setdefault("full_horizon_attempts", [])
    chance = _next_horizon_chance(records, cycles, args.output_dir)
    report["bootstrap_state"] = {
        "cycles": cycles,
        "phase": "bootstrap_direct_rho3",
        "chance": chance,
        "status": "running",
    }
    report["stop_reason"] = "running"
    _write_report(report_path, report)
    attempt = _run_horizon_attempt(
        args,
        rho_seed=rho_seed_path,
        cycles=cycles,
        phase="bootstrap_direct_rho3",
        chance=chance,
        rss_limit_bytes=rss_limit_bytes,
        heartbeat_seed_label="RHO_1..RHO_3",
    )
    attempt.update(
        {
            "chance": chance,
            "adaptive_source_cycles": 0,
            "adaptive_step_cycles": cycles,
            "accepted_for_continuation": bool(attempt["success"]),
        }
    )
    records.append(attempt)
    report["homotopy_constructed_cycles"] = max(
        int(report.get("homotopy_constructed_cycles") or 0), cycles
    )
    report["bootstrap_state"]["status"] = (
        "certified" if attempt["success"] else "failed"
    )
    if attempt["success"]:
        report["largest_successful_cycles"] = cycles
        report["solver_gap_cycles"] = []
        report["stop_reason"] = (
            "requested_ceiling_reached" if args.max_cycles == cycles else "running"
        )
    else:
        report["stop_reason"] = attempt["failure_kind"]
        report["solver_gap_cycles"] = (
            [cycles] if attempt["failure_kind"] == "solver_failure" else []
        )
    _write_report(report_path, report)
    _write_markdown(markdown_path, report)
    return attempt


def _bootstrap_full_horizon(
    args: argparse.Namespace,
    *,
    report: dict,
    report_path: Path,
    markdown_path: Path,
    rho_seed_path: Path,
    rss_limit_bytes: int,
) -> dict:
    """Reach certified FHO_2, retaining each recoverable bootstrap checkpoint.

    A direct two-cycle attempt is tried once. On numerical failure, certify
    FHO_1 from the original common seed and splice it onto the second RHO cycle
    before retrying FHO_2. Resume
    can enter at any of these boundaries without recalculating the reference
    RHO or overwriting earlier solver attempts.
    """

    records = report.setdefault("full_horizon_attempts", [])

    def certified_checkpoint(cycles: int) -> dict | None:
        for record in reversed(records):
            if (
                int(record.get("cycles") or 0) == cycles
                and record.get("success") is True
                and record.get("certificate_valid") is True
                and record.get("accepted_for_continuation", True)
                and record.get("solution_path")
                and record.get("result_path")
                and _seed_cycle_count(Path(record["solution_path"])) == cycles
                and _benchmark_success(
                    Path(record["result_path"]),
                    expected_mode="single_shot",
                    expected_cycles=cycles,
                    expected_solver=args.full_horizon_solver,
                    expected_mechanics="reduced",
                )
            ):
                return record
        return None

    def attempt(cycles: int, phase: str, prefix: dict | None = None) -> dict:
        common_seed = args.seed_dir / "common-reduced.npz" if cycles == 1 else None
        seed = rho_seed_path
        if common_seed is None:
            seed = args.output_dir / "homotopy-seeds" / f"rho-prefix-{cycles:04d}.npz"
            write_rho_seed_prefix(rho_seed_path, seed, cycles)
        chance = _next_horizon_chance(records, cycles, args.output_dir)
        report["stop_reason"] = "running"
        report["bootstrap_state"] = {
            "cycles": cycles, "phase": phase, "chance": chance, "status": "running"
        }
        _write_report(report_path, report)
        result = _run_horizon_attempt(
            args,
            rho_seed=seed,
            cycles=cycles,
            phase=phase,
            chance=chance,
            rss_limit_bytes=rss_limit_bytes,
            prefix_solution_path=(
                None if prefix is None else Path(prefix["solution_path"])
            ),
            heartbeat_seed_label=(
                "common_reduced"
                if common_seed is not None
                else (f"RHO_1..RHO_{cycles}" if prefix is None else "FHO_1+RHO_2")
            ),
            common_initial_solution=common_seed,
        )
        source_cycles = 0 if prefix is None else 1
        result.update(
            {
                "chance": chance,
                "adaptive_source_cycles": source_cycles,
                "adaptive_step_cycles": cycles - source_cycles,
                "accepted_for_continuation": bool(result["success"]),
            }
        )
        records.append(result)
        report["homotopy_constructed_cycles"] = max(
            int(report.get("homotopy_constructed_cycles") or 0), cycles
        )
        if result["success"]:
            report["largest_successful_cycles"] = cycles
        report["bootstrap_state"]["status"] = (
            "certified" if result["success"] else "failed"
        )
        _write_report(report_path, report)
        return result

    def finish(result: dict) -> dict:
        if result["success"] and result["cycles"] == 2:
            report["largest_successful_cycles"] = 2
            report["solver_gap_cycles"] = []
            report["stop_reason"] = (
                "requested_ceiling_reached"
                if args.max_cycles == 2
                else (
                    "rho_prefix_ceiling_reached"
                    if report.get("effective_max_cycles") == 2
                    else "running"
                )
            )
        else:
            # The final bootstrap verdict must refer to the actual failing
            # attempt, including FHO_1 or the retried FHO_2, not to stale FHO_2.
            report["stop_reason"] = result["failure_kind"]
            report["solver_gap_cycles"] = (
                [result["cycles"]]
                if result["failure_kind"] == "solver_failure"
                else []
            )
        _write_report(report_path, report)
        _write_markdown(markdown_path, report)
        return result

    bootstrap_two = certified_checkpoint(2)
    if bootstrap_two is not None:
        return finish(bootstrap_two)

    bootstrap_one = certified_checkpoint(1)
    previous_bootstrap_attempt = any(
        int(record.get("cycles") or 0) in (1, 2) for record in records
    )
    if bootstrap_one is None and not previous_bootstrap_attempt:
        direct_two = attempt(2, "bootstrap")
        if direct_two["success"] or direct_two["failure_kind"] != "solver_failure":
            return finish(direct_two)

    if bootstrap_one is None:
        bootstrap_one = attempt(1, "bootstrap_fho1")
        if not bootstrap_one["success"]:
            return finish(bootstrap_one)
    else:
        report.setdefault("bootstrap_reuse_events", []).append(
            {"cycles": 1, "solution_path": bootstrap_one["solution_path"]}
        )
        report["largest_successful_cycles"] = 1
        _write_report(report_path, report)

    return finish(attempt(2, "bootstrap_from_fho1", prefix=bootstrap_one))


def _next_extension_run_number(
    records: list[dict], target_cycle: int, output_dir: Path
) -> int:
    """Account for both reported and abruptly interrupted RHO extensions."""

    reported = _next_attempt_number(records, "target_cycle", target_cycle)
    case_dir = output_dir / f"rho-extension-after-fho-{target_cycle - 1:04d}"
    existing = []
    if case_dir.is_dir() and any(case_dir.glob("stage-*")):
        existing.append(1)
    existing.extend(
        int(path.name.removeprefix("retry-"))
        for path in case_dir.glob("retry-*")
        if path.is_dir() and path.name.removeprefix("retry-").isdigit()
    )
    return max([reported, *(number + 1 for number in existing)])


def _run_fho_terminal_bridge(
    args: argparse.Namespace,
    *,
    source_full_solution: Path,
    target_cycles: int,
    rss_limit_bytes: int,
    chance: int,
) -> dict:
    """Try FHO_(N+1) without requiring an RHO_(N+1) prefix.

    The bridge is only valid for exactly one appended cycle.  Its seed is the
    certified FHO_N solution plus an extrapolation whose first state node is
    exactly FHO_N's terminal state.  The monolithic FHO solve, rather than an
    RHO transfer, is therefore the sole feasibility authority for N+1.
    """

    source_cycles = _seed_cycle_count(source_full_solution)
    if target_cycles != source_cycles + 1:
        raise ValueError("An FHO terminal bridge must append exactly one cycle.")
    bridge_dir = (
        args.output_dir
        / "fho-terminal-bridges"
        / f"from-{source_cycles:04d}-to-{target_cycles:04d}"
        / f"chance-{chance}"
    )
    extension_seed = bridge_dir / "terminal-cycle-seed.npz"
    concatenated_seed = bridge_dir / "fho-terminal-bridge-seed.npz"
    write_fho_terminal_continuation_seed(source_full_solution, extension_seed)
    bridge_metadata = append_rho_extension_cycle(
        source_full_solution, extension_seed, concatenated_seed
    )
    attempt = _run_horizon_attempt(
        args,
        rho_seed=concatenated_seed,
        cycles=target_cycles,
        phase="fho_terminal_bridge",
        chance=chance,
        rss_limit_bytes=rss_limit_bytes,
        prefix_solution_path=source_full_solution,
        heartbeat_seed_label=f"FHO_{source_cycles}+terminal_bridge",
    )
    attempt.update(
        {
            "bridge_source_cycles": source_cycles,
            "bridge_seed_path": str(concatenated_seed),
            "bridge_terminal_cycle_seed_path": str(extension_seed),
            "bridge_boundary_maximum_absolute_change": bridge_metadata[
                "replaced_reduced_boundary_maximum_absolute_change"
            ],
            "seed_origin": "certified_fho_terminal_direct_bridge",
            "accepted_for_continuation": bool(attempt["success"]),
        }
    )
    return attempt


def _continue_adaptively(
    args: argparse.Namespace,
    *,
    report: dict,
    report_path: Path,
    markdown_path: Path,
    rho_seed_path: Path,
    effective_max_cycles: int,
    current_cycles: int,
    current_full_solution: Path,
    rss_limit_bytes: int,
) -> int:
    """Advance by the preferred jump, with a certified +1 fallback."""

    solver_gap_cycles: list[int] = []
    report.setdefault("adaptive_fallback_events", [])
    while current_cycles < effective_max_cycles:
        preferred_target = adaptive_continuation_target(
            current_cycles, effective_max_cycles, args.continuation_step_cycles
        )
        step_sizes = [preferred_target - current_cycles]
        if step_sizes[0] > 1:
            step_sizes.append(1)

        accepted = False
        for step_cycles in step_sizes:
            target_cycles = current_cycles + step_cycles
            attempt_args = argparse.Namespace(**vars(args))
            if step_cycles > 1:
                attempt_args.output_dir = (
                    args.output_dir
                    / "adaptive-attempts"
                    / f"from-{current_cycles:04d}-to-{target_cycles:04d}"
                )

            carrier_solution = current_full_solution
            concatenated_seed = current_full_solution
            extension_objectives: list[float] = []
            step_failure: str | None = None
            infrastructure_error = False
            for extension_cycle in range(current_cycles + 1, target_cycles + 1):
                extension_run_number = _next_extension_run_number(
                    report["extension_rho_attempts"],
                    extension_cycle,
                    attempt_args.output_dir,
                )
                extension_attempt = _run_extension_rho(
                    attempt_args,
                    source_full_solution=carrier_solution,
                    reference_rho_seed=rho_seed_path,
                    after_cycles=extension_cycle - 1,
                    rss_limit_bytes=rss_limit_bytes,
                    source_label=(
                        f"FHO_{current_cycles}"
                        if extension_cycle == current_cycles + 1
                        else f"RHO_{extension_cycle - 1}"
                    ),
                    run_number=extension_run_number,
                )
                extension_attempt.update(
                    {
                        "source_fho_certificate_path": str(current_full_solution),
                        "adaptive_source_cycles": current_cycles,
                        "adaptive_target_cycles": target_cycles,
                        "adaptive_step_cycles": step_cycles,
                    }
                )
                report["extension_rho_attempts"].append(extension_attempt)
                _write_report(report_path, report)
                if extension_attempt.get("memory_limit_exceeded") or extension_attempt.get("timed_out"):
                    report["stop_reason"] = "rho_extension_" + str(extension_attempt["failure_kind"])
                    report["stop_class"] = "resource_limit"
                    report["physiological_limit_certified"] = False
                    _write_report(report_path, report)
                    _write_markdown(markdown_path, report)
                    return 3
                if extension_attempt["infrastructure_error"]:
                    infrastructure_error = True
                    step_failure = "rho_extension_infrastructure_error"
                    break
                if not extension_attempt["success"]:
                    step_failure = "rho_extension_" + str(
                        extension_attempt["failure_kind"]
                    )
                    break

                extension_objective = _benchmark_window_objective(
                    Path(extension_attempt["result_path"])
                )
                if extension_objective is not None:
                    extension_objectives.append(extension_objective)
                next_seed = (
                    attempt_args.output_dir
                    / "homotopy-seeds"
                    / (
                        f"fho-{current_cycles:04d}-plus-rho-through-"
                        f"{extension_cycle:04d}.npz"
                    )
                )
                append_rho_extension_cycle(
                    concatenated_seed,
                    Path(extension_attempt["solution_path"]),
                    next_seed,
                )
                concatenated_seed = next_seed
                carrier_solution = Path(extension_attempt["solution_path"])

            # A physical/numerical failure of the auxiliary RHO handoff is not
            # evidence that the monolithic FHO has reached its feasibility
            # limit.  For the one-cycle fallback, test that question directly
            # from the terminal state of the last certified FHO.
            if (
                step_failure is not None
                and step_cycles == 1
                and getattr(args, "allow_fho_terminal_bridge", False)
                and not infrastructure_error
            ):
                chance = _next_horizon_chance(
                    report["full_horizon_attempts"], target_cycles, args.output_dir
                )
                bridge_attempt = _run_fho_terminal_bridge(
                    args,
                    source_full_solution=current_full_solution,
                    target_cycles=target_cycles,
                    rss_limit_bytes=rss_limit_bytes,
                    chance=chance,
                )
                bridge_attempt.update(
                    {
                        "adaptive_source_cycles": current_cycles,
                        "adaptive_step_cycles": 1,
                        "rho_extension_failure": step_failure,
                    }
                )
                report["full_horizon_attempts"].append(bridge_attempt)
                report.setdefault("terminal_bridge_attempts", []).append(
                    {
                        "from_cycles": current_cycles,
                        "to_cycles": target_cycles,
                        "success": bridge_attempt["success"],
                        "failure_kind": bridge_attempt["failure_kind"],
                        "rho_extension_failure": step_failure,
                        "result_path": bridge_attempt["result_path"],
                        "solution_path": bridge_attempt["solution_path"],
                    }
                )
                _write_report(report_path, report)
                if bridge_attempt["infrastructure_error"]:
                    report["stop_reason"] = "fho_terminal_bridge_infrastructure_error"
                    _write_report(report_path, report)
                    _write_markdown(markdown_path, report)
                    return 3
                if bridge_attempt["success"]:
                    current_cycles = target_cycles
                    current_full_solution = Path(bridge_attempt["solution_path"])
                    report["homotopy_constructed_cycles"] = current_cycles
                    report["largest_successful_cycles"] = current_cycles
                    report["stop_reason"] = (
                        "requested_ceiling_reached"
                        if current_cycles == args.max_cycles
                        else "running"
                    )
                    _write_report(report_path, report)
                    accepted = True
                    break
                step_failure = "fho_terminal_bridge_" + str(
                    bridge_attempt["failure_kind"]
                )

            if step_failure is None:
                chance = _next_horizon_chance(
                    report["full_horizon_attempts"],
                    target_cycles,
                    attempt_args.output_dir,
                )
                attempt = _run_horizon_attempt(
                    attempt_args,
                    rho_seed=concatenated_seed,
                    cycles=target_cycles,
                    phase="adaptive_continuation",
                    chance=chance,
                    rss_limit_bytes=rss_limit_bytes,
                    prefix_solution_path=current_full_solution,
                    heartbeat_seed_label=(
                        f"FHO_{current_cycles}+RHO_"
                        f"{current_cycles + 1}..RHO_{target_cycles}"
                    ),
                )
                baseline_objective = _certified_fho_objective(report, current_cycles)
                additive_reference = (
                    baseline_objective + sum(extension_objectives)
                    if baseline_objective is not None
                    and len(extension_objectives) == step_cycles
                    else None
                )
                gate = objective_gate(
                    _benchmark_window_objective(Path(attempt["result_path"])),
                    additive_reference,
                    args.jump_objective_relative_tolerance,
                )
                jump_gate_passes = bool(
                    step_cycles == 1 or (gate["comparable"] and gate["passes"])
                )
                attempt.update(
                    {
                        "construction_method": "certified_post_fho_rho_chain",
                        "source_fho_certificate_path": str(current_full_solution),
                        "chance": chance,
                        "adaptive_source_cycles": current_cycles,
                        "adaptive_step_cycles": step_cycles,
                        "objective_gate": gate,
                        "accepted_for_continuation": bool(
                            attempt["success"] and jump_gate_passes
                        ),
                    }
                )
                report["full_horizon_attempts"].append(attempt)
                _write_report(report_path, report)
                infrastructure_error = bool(attempt["infrastructure_error"])
                if infrastructure_error:
                    step_failure = "infrastructure_error"
                elif not attempt["success"]:
                    step_failure = str(attempt["failure_kind"])
                elif not jump_gate_passes:
                    step_failure = (
                        "jump_objective_unavailable"
                        if not gate["comparable"]
                        else "jump_objective_degradation"
                    )
                else:
                    current_cycles = target_cycles
                    current_full_solution = Path(attempt["solution_path"])
                    report["homotopy_constructed_cycles"] = current_cycles
                    report["largest_successful_cycles"] = current_cycles
                    report["stop_reason"] = (
                        "requested_ceiling_reached"
                        if current_cycles == args.max_cycles
                        else (
                            "rho_prefix_ceiling_reached"
                            if current_cycles == effective_max_cycles
                            else "running"
                        )
                    )
                    _write_report(report_path, report)
                    accepted = True
                    break

            if infrastructure_error:
                report["stop_reason"] = step_failure
                _write_report(report_path, report)
                _write_markdown(markdown_path, report)
                return 3
            if step_cycles > 1:
                report["adaptive_fallback_events"].append(
                    {
                        "from_cycles": current_cycles,
                        "rejected_target_cycles": target_cycles,
                        "rejected_step_cycles": step_cycles,
                        "reason": step_failure,
                        "fallback_step_cycles": 1,
                    }
                )
                _write_report(report_path, report)
                continue

            solver_gap_cycles.append(target_cycles)
            report["stop_reason"] = step_failure
            report["stop_class"] = (
                "resource_limit" if step_failure and any(word in step_failure for word in ("memory_limit", "timeout"))
                else "handoff_numerical_unresolved" if step_failure and step_failure.startswith("rho_extension_")
                else "numerical_unresolved"
            )
            report["physiological_limit_certified"] = False
            break

        if not accepted:
            break

    report["solver_gap_cycles"] = solver_gap_cycles
    _write_markdown(markdown_path, report)
    _write_report(report_path, report)
    return 0


def _probe_rho_extensions(
    args: argparse.Namespace,
    *,
    report: dict,
    report_path: Path,
    markdown_path: Path,
    rho_seed_path: Path,
    effective_max_cycles: int,
    current_cycles: int,
    current_solution: Path,
    rss_limit_bytes: int,
) -> int:
    """Certify the RHO-only tail from a fixed FHO terminal state.

    This deliberately does not call ``append_rho_extension_cycle`` nor create
    an FHO candidate.  It answers the narrower question: how many locally
    feasible, audited one-cycle handoffs can follow the last certified FHO?
    """

    report.setdefault("extension_rho_attempts", [])
    report["rho_extension_probe"] = {
        "source_fho_cycles": current_cycles,
        "maximum_probe_cycles": effective_max_cycles,
        "status": "running",
    }
    _write_report(report_path, report)

    carrier_solution = current_solution
    source_fho_solution = current_solution
    while current_cycles < effective_max_cycles:
        target_cycle = current_cycles + 1
        run_number = _next_extension_run_number(
            report["extension_rho_attempts"], target_cycle, args.output_dir
        )
        attempt = _run_extension_rho(
            args,
            source_full_solution=carrier_solution,
            reference_rho_seed=rho_seed_path,
            after_cycles=current_cycles,
            rss_limit_bytes=rss_limit_bytes,
            source_label=(
                f"FHO_{current_cycles}"
                if carrier_solution == source_fho_solution
                else f"RHO_{current_cycles}"
            ),
            run_number=run_number,
        )
        attempt.update(
            {
                "source_fho_certificate_path": str(source_fho_solution),
                "rho_extension_probe": True,
                "probe_source_cycles": current_cycles,
                "probe_target_cycles": target_cycle,
            }
        )
        report["extension_rho_attempts"].append(attempt)
        report["rho_extension_probe"].update(
            {
                "last_attempted_cycle": target_cycle,
                "last_successful_cycle": (
                    target_cycle if attempt.get("success") else current_cycles
                ),
            }
        )
        _write_report(report_path, report)

        if attempt.get("success"):
            carrier_solution = Path(attempt["solution_path"])
            current_cycles = target_cycle
            continue

        failure_kind = str(attempt.get("failure_kind") or "unknown")
        report["rho_extension_probe"].update(
            {"status": "stopped", "failure_kind": failure_kind}
        )
        report["stop_reason"] = "rho_extension_probe_" + failure_kind
        report["stop_class"] = (
            "resource_limit"
            if attempt.get("memory_limit_exceeded") or attempt.get("timed_out")
            else ("infrastructure_error" if attempt.get("infrastructure_error") else "rho_handoff_failure")
        )
        _write_report(report_path, report)
        _write_markdown(markdown_path, report)
        return 3 if attempt.get("infrastructure_error") else 0

    report["rho_extension_probe"].update(
        {"status": "requested_ceiling_reached", "last_successful_cycle": current_cycles}
    )
    report["stop_reason"] = "rho_extension_probe_requested_ceiling_reached"
    report["stop_class"] = None
    _write_report(report_path, report)
    _write_markdown(markdown_path, report)
    return 0


def run(args: argparse.Namespace) -> int:
    args.workspace = args.workspace.resolve()
    args.seed_dir = args.seed_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    # Keep this campaign-level path stable when adaptive +N continuation
    # creates per-attempt output directories.  ``Namespace(**vars(args))``
    # carries it into every one-cycle RHO child process.
    args.rho_transition_c_cache_dir = (
        args.output_dir / "rho-transition-ipopt-c-cache"
    ).resolve()
    if args.max_cycles < 2:
        raise ValueError("--max-cycles must be at least two.")
    if args.max_cycles < args.bootstrap_rho_prefix_cycles:
        raise ValueError("--max-cycles must cover the chosen bootstrap RHO prefix.")
    if args.n_threads < 1:
        raise ValueError("--n-threads must be strictly positive.")
    if args.continuation_step_cycles < 1:
        raise ValueError("--continuation-step-cycles must be strictly positive.")
    if (
        args.jump_objective_relative_tolerance < 0
        or not math.isfinite(args.jump_objective_relative_tolerance)
    ):
        raise ValueError(
            "--jump-objective-relative-tolerance must be finite and non-negative."
        )
    if args.poll_interval_s <= 0:
        raise ValueError("--poll-interval-s must be strictly positive.")
    if args.attempt_timeout_s is not None and args.attempt_timeout_s <= 0:
        raise ValueError("--attempt-timeout-s must be strictly positive.")

    total_memory = available_memory_bytes()
    rss_limit_gib = _parse_memory_limit(args.memory_limit_gib, total_memory)
    rss_limit_bytes = int(rss_limit_gib * GIB)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "full-horizon-report.json"
    markdown_path = args.output_dir / "full-horizon-report.md"
    if args.resume:
        return _resume_run(
            args,
            report_path=report_path,
            markdown_path=markdown_path,
            total_memory=total_memory,
            rss_limit_gib=rss_limit_gib,
            rss_limit_bytes=rss_limit_bytes,
        )
    report = {
        "schema": "cocofest-full-horizon-sweep-v2",
        "max_cycles": args.max_cycles,
        "rho_available_cycles": 0,
        "total_memory_bytes": total_memory,
        "total_memory_gib": total_memory / GIB,
        "rss_limit_bytes": rss_limit_bytes,
        "rss_limit_gib": rss_limit_gib,
        "rho_graph": "SX",
        "full_horizon_graph": "MX",
        "rho_solver": "ipopt",
        "rho_transition_codegen": {
            "graph": "SX",
            "c_compile": True,
            "cache_dir": str(args.rho_transition_c_cache_dir),
            "cache_name": "rho-transition-sx",
        },
        "full_horizon_solver": args.full_horizon_solver,
        "rho_only": args.rho_only,
        "linear_solver": "ma57",
        "initialization": "autonomous_certified_terminal_rho_chain",
        "allow_fho_terminal_bridge": args.allow_fho_terminal_bridge,
        "continuation_step_cycles": args.continuation_step_cycles,
        "bootstrap_rho_prefix_cycles": args.bootstrap_rho_prefix_cycles,
        "jump_objective_relative_tolerance": (
            args.jump_objective_relative_tolerance
        ),
        "adaptive_fallback_events": [],
        "rho": None,
        "paired_reduced_control_attempts": [],
        "extension_rho_attempts": [],
        "terminal_bridge_attempts": [],
        "homotopy_constructed_cycles": 0,
        "full_horizon_attempts": [],
        "largest_successful_cycles": 0,
        "stop_reason": "not_started",
    }
    _write_report(report_path, report)

    rho_dir = args.output_dir / "rho-reduced"
    rho_result_path = rho_dir / "result.json"
    rho_seed_path = rho_dir / "concatenated-solution.npz"
    for stale_path in (rho_result_path, rho_seed_path):
        stale_path.unlink(missing_ok=True)
    rho_monitored = run_monitored(
        _rho_command(args, rho_result_path, rho_seed_path, n_windows=args.max_cycles),
        cwd=args.workspace,
        log_path=rho_dir / "solver.log",
        rss_limit_bytes=rss_limit_bytes,
        heartbeat_label=f"RHO_reference cycles=1..{args.max_cycles}",
        poll_interval_s=args.poll_interval_s,
        timeout_s=args.attempt_timeout_s,
    )
    report["rho"] = {
        **asdict(rho_monitored),
        "peak_rss_gib": rho_monitored.peak_rss_bytes / GIB,
        "result_path": str(rho_result_path),
        "seed_path": str(rho_seed_path),
    }
    rho_result_cycles = _benchmark_validated_cycles(
        rho_result_path,
        expected_mode="rho",
        expected_solver="ipopt",
        expected_requested_cycles=args.max_cycles,
    )
    rho_seed_cycles = _seed_cycle_count(rho_seed_path)
    rho_available_cycles = (
        rho_result_cycles if rho_result_cycles == rho_seed_cycles else 0
    )
    report["rho_available_cycles"] = rho_available_cycles
    # Keep the seed-ceiling reason separate from the endurance verdict.  This
    # remains useful even when RHO supplied a valid prefix: it explains why a
    # later FHO continuation cannot be extended beyond that prefix.
    report["rho_prefix_stop_assessment"] = _rho_prefix_stop_assessment(
        rho_result_path
    )
    report["rho"].update(
        {
            "success": (
                rho_available_cycles >= 2
                and (
                    not args.rho_only
                    or rho_available_cycles == args.max_cycles
                )
                and rho_monitored.return_code == 0
                and not rho_monitored.memory_limit_exceeded
                and not rho_monitored.timed_out
                and not _log_has_unknown_mumps_warning(rho_monitored.log_path)
            ),
            "certificate_valid": rho_result_cycles >= 2,
            "unknown_mumps_warning": _log_has_unknown_mumps_warning(
                rho_monitored.log_path
            ),
            "validated_cycles": rho_result_cycles,
            "seed_cycles": rho_seed_cycles,
            "requested_ceiling_reached": rho_available_cycles == args.max_cycles,
        }
    )
    if not report["rho"]["success"]:
        rho_infrastructure_error = bool(
            not rho_monitored.memory_limit_exceeded
            and not rho_monitored.timed_out
            and (
                rho_monitored.return_code != 0
                or not _benchmark_payload_is_readable(rho_result_path)
            )
        )
        report["stop_reason"] = (
            "rho_memory_limit"
            if rho_monitored.memory_limit_exceeded
            else (
                "rho_timeout"
                if rho_monitored.timed_out
                else (
                    "rho_infrastructure_error"
                    if rho_infrastructure_error
                    else (
                        "rho_incomplete"
                        if args.rho_only and rho_available_cycles >= 2
                        else "rho_solver_failure"
                    )
                )
            )
        )
        _write_report(report_path, report)
        _write_markdown(markdown_path, report)
        return 3 if rho_infrastructure_error else 2

    if args.rho_only:
        report["stop_reason"] = "rho_only_completed"
        _write_report(report_path, report)
        _write_markdown(markdown_path, report)
        print(
            f"RHO-only completed: {rho_available_cycles}/{args.max_cycles} "
            "cycles certified; FHO skipped.",
            flush=True,
        )
        return 0

    effective_max_cycles = args.max_cycles
    report["effective_max_cycles"] = effective_max_cycles
    bootstrap_function = (
        _bootstrap_direct_three_cycle_horizon
        if args.bootstrap_rho_prefix_cycles == 3
        else _bootstrap_full_horizon
    )
    attempt = bootstrap_function(
        args,
        report=report,
        report_path=report_path,
        markdown_path=markdown_path,
        rho_seed_path=rho_seed_path,
        rss_limit_bytes=rss_limit_bytes,
    )
    if not attempt["success"]:
        return 3 if attempt["infrastructure_error"] else 0

    current_full_solution = Path(attempt["solution_path"])
    report["largest_successful_cycles"] = args.bootstrap_rho_prefix_cycles
    report["stop_reason"] = (
        "requested_ceiling_reached"
        if args.max_cycles == args.bootstrap_rho_prefix_cycles
        else (
            "rho_prefix_ceiling_reached"
            if effective_max_cycles == args.bootstrap_rho_prefix_cycles
            else "running"
        )
    )
    # Persist the certified bootstrap before any local RHO extension.
    _write_report(report_path, report)
    return _continue_adaptively(
        args,
        report=report,
        report_path=report_path,
        markdown_path=markdown_path,
        rho_seed_path=rho_seed_path,
        effective_max_cycles=effective_max_cycles,
        current_cycles=args.bootstrap_rho_prefix_cycles,
        current_full_solution=current_full_solution,
        rss_limit_bytes=rss_limit_bytes,
    )


def main(argv: Iterable[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
