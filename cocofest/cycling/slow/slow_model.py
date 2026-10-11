"""Observed slow states and conditional Ding propagation for simulation.

An oracle state is an observation of a certified RHO trajectory. A propagated
array is deliberately not a ``SlowState``: predictions must not acquire the
``simulator`` provenance of the observation that initialized them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np

from cocofest.optimization.ding_slow_cycle import DingSlowCycleMap


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class OracleProvenance:
    """Content-addressed observation receipt, not a cryptographic signature.

    Certification remains the existing RHO certification protocol's decision;
    this receipt binds the data and certificate read by the oracle loader.
    """

    archive_path: str
    archive_sha256: str
    certification_path: str
    certification_sha256: str
    model_sha256: str
    cycle: int

    def __post_init__(self):
        for name in ("archive_sha256", "certification_sha256", "model_sha256"):
            _digest(getattr(self, name), name)
        if not self.archive_path or not self.certification_path:
            raise ValueError("archive and certification paths are required")
        if isinstance(self.cycle, bool) or not isinstance(self.cycle, int) or self.cycle < 0:
            raise ValueError("cycle must be a non-negative integer")


@dataclass(frozen=True)
class SlowState:
    """Exact terminal A, Tau1 and Km observations, one entry per named muscle."""

    muscle_names: tuple[str, ...]
    A: tuple[float, ...]
    Tau1: tuple[float, ...]
    Km: tuple[float, ...]
    provenance: OracleProvenance
    source: str = "simulator"

    def __post_init__(self):
        object.__setattr__(self, "muscle_names", tuple(self.muscle_names))
        if not self.muscle_names or len(set(self.muscle_names)) != len(self.muscle_names):
            raise ValueError("muscle_names must be nonempty and unique")
        if any(not isinstance(name, str) or not name for name in self.muscle_names):
            raise ValueError("muscle names must be nonempty strings")
        for name in ("A", "Tau1", "Km"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (len(self.muscle_names),) or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"{name} must contain one finite positive value per muscle")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        if self.source != "simulator" or not isinstance(self.provenance, OracleProvenance):
            raise ValueError("oracle states require source='simulator' and OracleProvenance")

    @property
    def matrix(self) -> np.ndarray:
        """Return a copy of shape (muscles, 3), columns ordered A, Tau1, Km."""
        return np.asarray([self.A, self.Tau1, self.Km], dtype=float).T.copy()

    @property
    def receipt_sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()


def read_oracle_state(
    archive_path: str | Path,
    certification_path: str | Path,
    *,
    muscle_names: Sequence[str],
    model_sha256: str,
    cycle: int,
) -> SlowState:
    """Read the last node of an existing certified witness-cycle NPZ.

    ``certification_path`` is the frozen witness ``summary.json`` with records
    ``{"cycle": k, "certified": true}``. Prepared/shuffled warm-start primals
    are not accepted. The caller binds ``model_sha256`` to the frozen protocol;
    this function does not independently rerun its feasibility certification.
    """
    archive_path, certification_path = Path(archive_path), Path(certification_path)
    if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 0:
        raise ValueError("cycle must be a non-negative integer")
    if archive_path.name != f"witness-cycle-{cycle}.npz":
        raise ValueError("oracle requires the named solved witness-cycle archive")
    if archive_path.resolve().parent != certification_path.resolve().parent:
        raise ValueError("witness archive and certification must share their campaign directory")
    certificate_bytes = certification_path.read_bytes()
    certificate = json.loads(certificate_bytes)
    records = [row for row in certificate["cycles"] if row["cycle"] == cycle]
    if len(records) != 1 or records[0].get("certified") is not True:
        raise ValueError("cycle must have exactly one explicitly certified record")
    archive_bytes = archive_path.read_bytes()
    columns, shape = [], None
    with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as archive:
        for state in ("A", "Tau1", "Km"):
            terminal = []
            for muscle in muscle_names:
                values = np.asarray(archive[f"states__{state}_{muscle}"], dtype=float)
                if values.ndim == 2 and values.shape[0] == 1:
                    values = values[0]
                if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
                    raise ValueError("witness states must be finite single-muscle trajectories")
                if shape is not None and values.shape != shape:
                    raise ValueError("all witness slow states must share the same node grid")
                shape = values.shape
                terminal.append(float(values[-1]))
            columns.append(tuple(terminal))
    provenance = OracleProvenance(str(archive_path.resolve()), hashlib.sha256(archive_bytes).hexdigest(),
        str(certification_path.resolve()), hashlib.sha256(certificate_bytes).hexdigest(), model_sha256, cycle)
    return SlowState(tuple(muscle_names), *columns, provenance=provenance)


def propagate_prescribed_force(
    state: SlowState, maps: Mapping[str, DingSlowCycleMap], repetitions: int = 1,
) -> np.ndarray:
    """Conditional exact slow flow, not a force or feasibility prediction.

    The supplied force polynomials are assumed unchanged at every repetition.
    Full A/Tau1/Km vectors are retained, including off-manifold initial offsets.
    """
    if set(maps) != set(state.muscle_names):
        raise ValueError("cycle maps must match the observed muscle names exactly")
    durations = [maps[name].duration for name in state.muscle_names]
    if not np.allclose(durations, durations[0], rtol=0, atol=1e-12):
        raise ValueError("muscle cycle maps must have a common cycle duration")
    return np.stack([maps[name].propagate(row, repetitions)
                     for name, row in zip(state.muscle_names, state.matrix, strict=True)])
