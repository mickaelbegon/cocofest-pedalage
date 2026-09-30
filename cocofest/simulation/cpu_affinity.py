"""Small, dependency-free CPU-affinity discovery helpers for the desktop GUI."""
from __future__ import annotations

import os
from pathlib import Path


def parse_cpu_list(value: str) -> frozenset[int]:
    """Parse Linux's ``Cpus_allowed_list`` representation (for example ``0-3,8``)."""
    cpus: set[int] = set()
    for token in value.strip().split(","):
        token = token.strip()
        if not token:
            continue
        try:
            lower, upper = (int(part) for part in token.split("-", 1)) if "-" in token else (int(token), int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid CPU list: {value!r}") from exc
        if lower < 0 or upper < lower:
            raise ValueError(f"Invalid CPU list: {value!r}")
        cpus.update(range(lower, upper + 1))
    if not cpus:
        raise ValueError(f"Invalid CPU list: {value!r}")
    return frozenset(cpus)


def available_cpu_ids() -> tuple[int, ...]:
    """Return CPUs which this GUI process is allowed to assign to child solvers."""
    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(os.sched_getaffinity(0)))
    return tuple(range(os.cpu_count() or 1))


def reserved_cpu_ids(proc_root: Path = Path("/proc"), *, current_pid: int | None = None) -> frozenset[int]:
    """Find dedicated CPUs already reserved by another pinned Linux process.

    Only singleton CPU masks are considered reservations.  Processes with a
    broad scheduling mask are normal system activity, not an exclusive claim
    to every CPU in that mask.  This catches the solver workers spawned by the
    independent-arm coordinator after they pin themselves.
    """
    current_pid = os.getpid() if current_pid is None else current_pid
    reserved: set[int] = set()
    try:
        candidates = tuple(proc_root.iterdir())
    except OSError:
        return frozenset()
    for candidate in candidates:
        if not candidate.name.isdigit() or int(candidate.name) == current_pid:
            continue
        try:
            lines = (candidate / "status").read_text(encoding="utf-8").splitlines()
        except OSError:
            continue  # A process may legitimately exit while the menu opens.
        mask = next((line.partition(":")[2].strip() for line in lines if line.startswith("Cpus_allowed_list:")), None)
        if mask is None:
            continue
        try:
            allowed = parse_cpu_list(mask)
        except ValueError:
            continue
        if len(allowed) == 1:
            reserved.update(allowed)
    return frozenset(reserved)
