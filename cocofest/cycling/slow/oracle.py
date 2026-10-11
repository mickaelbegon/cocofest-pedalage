"""Measured simulator slow states with byte-pinned RHO provenance.

Solved trajectories contribute their last node. Prepared restart archives
contribute their frozen initial equality, which is the previous cycle's end;
their shifted warm-start trajectory must never be interpreted as a measurement.
Certification refers to the discrete RHO problem, not continuous ODE accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


STATE_ORDER = ("A", "Tau1", "Km")
AUDIT_SCOPE = "complete_discrete_NLP_g_and_x; no continuous-ODE replay"


@dataclass(frozen=True)
class OracleSlowState:
    """Immutable physical states, muscle-major, with (A, Tau1, Km) columns."""

    side: str
    completed_cycles: int
    muscles: tuple[str, ...]
    values: tuple[tuple[float, float, float], ...]
    source_kind: str
    source_node: str
    artifacts: tuple[tuple[str, str], ...]
    model_sha256: str

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)

    def as_dict(self) -> dict[str, float]:
        return {f"{state}_{muscle}": value for muscle, row in zip(self.muscles, self.values, strict=True)
                for state, value in zip(STATE_ORDER, row, strict=True)}

    def verify_files(self) -> None:
        for path, digest in self.artifacts:
            _checked_path(Path(path), digest)


def _object(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def _path(reference: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Missing provenance path")
    path = Path(value)
    return (path if path.is_absolute() else reference.parent / path).resolve(strict=True)


def _checked_path(path: Path, expected: str) -> tuple[str, str]:
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"Provenance digest mismatch: {path}")
    return str(path), digest


def _arguments(side: str, muscles: Sequence[str]) -> tuple[str, ...]:
    muscles = tuple(muscles)
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    if (not muscles or any(not isinstance(m, str) or not m for m in muscles)
            or len(set(muscles)) != len(muscles)):
        raise ValueError("muscles must be a nonempty sequence of distinct names")
    return muscles


def _read_values(archive, muscles, *, prepared: bool):
    rows, shapes = [], set()
    for muscle in muscles:
        row = []
        for state in STATE_ORDER:
            key = f"{state}_{muscle}"
            try:
                if prepared:
                    lower = np.asarray(archive[f"problem__x_bounds:{key}:min"], float)
                    upper = np.asarray(archive[f"problem__x_bounds:{key}:max"], float)
                    if (lower.ndim != 2 or lower.shape != upper.shape
                            or not np.array_equal(lower[:, :1], upper[:, :1])):
                        raise ValueError(f"Slow state {key} lacks a frozen initial equality")
                    array, column = lower, 0
                else:
                    array, column = np.asarray(archive[f"states__{key}"], float), -1
                if array.ndim != 2 or array.shape[0] != 1 or array.shape[1] < (1 if prepared else 2):
                    raise ValueError(f"Slow state {key} has an invalid node grid")
                if not np.isfinite(array).all():
                    raise ValueError(f"Slow state {key} contains nonfinite values")
                value = float(array[0, column])
                if value <= 0:
                    raise ValueError(f"Slow state {key} must be positive")
                row.append(value)
                shapes.add(array.shape)
            except KeyError as error:
                raise ValueError(f"Missing slow state {key}") from error
        rows.append(tuple(row))
    if len(shapes) != 1:
        raise ValueError("Slow-state node grids differ")
    return tuple(rows)


def load_certified_checkpoint(receipt: Path, *, side: str, muscles: Sequence[str]) -> OracleSlowState:
    """Read a certified bilateral restart without constructing a solver."""
    from cocofest.simulation.rho_restart_checkpoint import verify_prepared_checkpoint

    muscles = _arguments(side, muscles)
    receipt = Path(receipt).resolve(strict=True)
    document = _object(receipt)
    cycle = document.get("completed_cycles")
    if (document.get("schema_version") != 1
            or document.get("checkpoint_kind") != "certified_bilateral_shifted_primal"
            or document.get("exact_bilateral_restart") is not True
            or type(cycle) is not int or cycle < 1):
        raise ValueError("A certified exact bilateral restart receipt is required")
    arm = document.get("arms", {}).get(side, {})
    if (arm.get("certified") is not True or arm.get("replay_roundtrip_exact") is not True
            or arm.get("completed_cycles") != cycle or arm.get("stimulation_history_complete") is not True):
        raise ValueError("Arm lacks certified exact state/history restoration")
    archive = _path(receipt, arm.get("primal_path"))
    model = _path(receipt, arm.get("model_path"))
    configuration = _path(receipt, document.get("configuration_path"))
    artifacts = ((str(receipt), sha256(receipt.read_bytes()).hexdigest()),
                 _checked_path(archive, arm.get("sha256")),
                 _checked_path(model, arm.get("model_sha256")),
                 _checked_path(configuration, document.get("configuration_sha256")))
    payload = _object(configuration)
    declared_model = payload.get(f"{side}_model_config", payload.get("model_config"))
    # The existing independent-arm runner resolves these declarations from its
    # launch directory; saved production configurations use absolute paths.
    if not isinstance(declared_model, str) or Path(declared_model).expanduser().resolve() != model:
        raise ValueError("Receipt model differs from the configured arm model")
    verified = verify_prepared_checkpoint(archive, completed_cycles=cycle)
    if (verified["prepared_problem_sha256"] != arm.get("prepared_problem_sha256")
            or verified["prepared_problem_sha256"] != arm.get("restored_problem_sha256")
            or verified["prepared_primal_signature"] != arm.get("prepared_primal_signature")
            or verified["metadata"].get("prepared_stimulation_history_complete") is not True):
        raise ValueError("Prepared/restored problem or stimulation history differs")
    with np.load(archive, allow_pickle=False) as data:
        values = _read_values(data, muscles, prepared=True)
    result = OracleSlowState(side, cycle, muscles, values, "certified_bilateral_restart",
                             "frozen_initial_equality_of_next_cycle", artifacts, arm["model_sha256"])
    result.verify_files()
    return result


def load_certified_cycle(receipt: Path, *, cycle: int, side: str,
                         muscles: Sequence[str], maximum_tolerance: float = 1e-5) -> OracleSlowState:
    """Read a solved witness from the existing unilateral-continuation receipt.

    The receipt must pin its source checkpoint and witness bytes, carry the
    complete discrete-NLP audit, and identify the requested side. A trajectory
    bearing only a successful solver status is insufficient.
    """
    muscles = _arguments(side, muscles)
    if type(cycle) is not int or cycle < 1:
        raise ValueError("cycle must be a positive integer")
    if isinstance(maximum_tolerance, bool) or not math.isfinite(maximum_tolerance) or maximum_tolerance <= 0:
        raise ValueError("maximum_tolerance must be finite and positive")
    receipt = Path(receipt).resolve(strict=True)
    document = _object(receipt)
    if (document.get("schema_version") != 1 or document.get("kind") != "unilateral_continuation_phase"
            or document.get("return_policy", {}).get("side") != side
            or document.get("muscle_order") != list(muscles)):
        raise ValueError("Receipt kind, side or muscle order differs")
    source_path = _path(receipt, document.get("source_receipt"))
    source_artifact = _checked_path(source_path, document.get("source_receipt_sha256"))
    source = load_certified_checkpoint(source_path, side=side, muscles=muscles)
    if (document.get("anchor_completed_cycles") != source.completed_cycles
            or document.get("source_archive_sha256") != source.artifacts[1][1]
            or cycle <= source.completed_cycles):
        raise ValueError("Witness source checkpoint or cycle differs")
    rows = document.get("cycles", [])
    matches = [row for row in rows if isinstance(row, dict) and row.get("cycle") == cycle]
    if len(matches) != 1:
        raise ValueError("Receipt must contain exactly one requested cycle")
    row = matches[0]
    tolerance = document.get("audit_tolerance")
    violation = row.get("maximum_normalized_violation")
    audit = row.get("independent_audit", {})
    if (row.get("certified") is not True or type(tolerance) not in (int, float)
            or not math.isfinite(tolerance) or not 0 < tolerance <= maximum_tolerance
            or type(violation) not in (int, float) or not math.isfinite(violation)
            or not 0 <= violation <= tolerance or audit.get("scope") != AUDIT_SCOPE):
        raise ValueError("Cycle lacks a passing complete discrete-NLP audit")
    for key in ("constraint_normalized_violation", "decision_normalized_violation", "terminal_work_normalized_violation"):
        value = audit.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= tolerance:
            raise ValueError(f"Cycle audit lacks a passing {key}")
    witness = _path(receipt, row.get("witness"))
    witness_artifact = _checked_path(witness, row.get("witness_sha256"))
    with np.load(witness, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata__json"].item()))
        if metadata.get("producer_mode") == "rho_replay_checkpoint" or metadata.get("independent_audit") != audit:
            raise ValueError("Witness is a prepared guess or its audit differs from the receipt")
        values = _read_values(data, muscles, prepared=False)
    artifacts = ((str(receipt), sha256(receipt.read_bytes()).hexdigest()), witness_artifact,
                 source_artifact, *source.artifacts)
    result = OracleSlowState(side, cycle, muscles, values, "certified_rho_trajectory", "last_node",
                             tuple(dict.fromkeys(artifacts)), source.model_sha256)
    terminal = row.get("terminal_slow_states")
    if not isinstance(terminal, dict) or any(terminal.get(k) != v for k, v in result.as_dict().items()):
        raise ValueError("Receipt terminal slow states differ from the witness last node")
    result.verify_files()
    return result
