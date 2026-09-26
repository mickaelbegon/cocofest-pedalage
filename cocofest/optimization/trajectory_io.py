"""Solver-independent NPZ trajectory I/O and scoped checkpoint publication.

The periodic cycling example retains its historical function names as wrappers.
New consumers can read archives without importing the solver driver or Bioptim.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path

import numpy as np

from .solution_archive import physical_archive_arrays, validate_physical_archive


class WarmupSolutionAdapter:
    """Minimal trajectory view shared by seed transfer and archive consumers."""

    def __init__(self, states, controls, metadata=None):
        self._states = states
        self._controls = controls
        self.metadata = metadata

    def decision_states(self, to_merge=None):
        return self._states

    def decision_controls(self, to_merge=None):
        return self._controls


def save_trajectory(path, states, controls, metadata=None, applied_pulse_widths=None):
    """Write the legacy arrays and, when applicable, the physical v1 contract."""
    payload = {}
    if metadata is not None:
        physical_arrays, metadata = physical_archive_arrays(states, controls, metadata)
        payload.update(physical_arrays)
    for prefix, arrays in (("states", states), ("controls", controls),
                           ("applied_pulse_widths", applied_pulse_widths or {})):
        payload.update({f"{prefix}__{key}": np.asarray(value) for key, value in arrays.items()})
    if metadata is not None:
        payload["metadata__json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )
    np.savez(path, **payload)


def load_trajectory(path):
    """Read old seeds and validate any explicitly versioned physical contract."""
    with np.load(path, allow_pickle=False) as data:
        states = {key.split("__", 1)[1]: np.asarray(data[key])
                  for key in data.files if key.startswith("states__")}
        controls = {key.split("__", 1)[1]: np.asarray(data[key])
                    for key in data.files if key.startswith("controls__")}
        metadata = (json.loads(str(data["metadata__json"].item()))
                    if "metadata__json" in data.files else None)
        validate_physical_archive(data, metadata or {})
    return WarmupSolutionAdapter(states, controls, metadata)


_checkpoint_hook = ContextVar("cocofest_checkpoint_publication_hook", default=None)


@contextmanager
def checkpoint_publication_hook(hook):
    """Decorate checkpoint publication for this context and restore on failure.

    ``hook(write, path, nmpc, args, completed_windows=...)`` receives the raw
    writer explicitly; it can atomically publish a seed plus its receipt.
    """
    token = _checkpoint_hook.set(hook)
    try:
        yield
    finally:
        _checkpoint_hook.reset(token)


def save_rho_replay_checkpoint(path, nmpc, args, *, completed_windows, metadata_factory):
    """Persist the shifted primal using the active publication policy."""
    def write(target, nmpc, args, *, completed_windows):
        nlp = nmpc.nlp[0]
        states = {key: np.asarray(nlp.x_init[key].init, dtype=float).copy()
                  for key in nlp.x_init.keys()}
        controls = {key: np.asarray(nlp.u_init[key].init, dtype=float).copy()
                    for key in nlp.u_init.keys()}
        if not states or not controls:
            raise RuntimeError("The shifted RHO primal has no state or control trace.")
        metadata = dict(metadata_factory(args), producer_mode="rho_replay_checkpoint",
                        producer_completed_windows=int(completed_windows))
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        save_trajectory(target, states, controls, metadata)

    hook = _checkpoint_hook.get()
    if hook is None:
        write(path, nmpc, args, completed_windows=completed_windows)
    else:
        hook(write, path, nmpc, args, completed_windows=completed_windows)
