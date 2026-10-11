"""Strict archival I/O for local capacity-frontier exchange slices.

The capacity-frontier solves are deliberately offline.  This module converts
their independent-rebuild receipts into the small immutable records consumed
by :mod:`capacity_frontier_interpolation`, without hiding provenance in a
plotting summary.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Mapping

from cocofest.optimization.capacity_frontier_interpolation import TricepsExchangeSlice


def _digest_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _epsilon_muscle(upper: object, order: tuple[str, ...]) -> str | None:
    if not isinstance(upper, list) or len(upper) != len(order):
        return None
    active = [name for name, value in zip(order, upper) if value is not None]
    if len(active) != 1 or active[0] == "Triceps":
        return None
    return active[0]


def _capacity_from_report(value: Mapping) -> tuple[float, ...]:
    capacity = value.get("support_smoke", {}).get("initial_capacity_ratio")
    if not isinstance(capacity, list):
        raise ValueError("Receipt lacks support_smoke.initial_capacity_ratio")
    return tuple(float(item) for item in capacity)


def slice_from_raw_reports(paths: Iterable[Path], *, checkpoint_cycle: int) -> TricepsExchangeSlice:
    """Build one exact-state exchange slice from one support and three ε receipts."""
    loaded = [(Path(path).resolve(strict=True), _load(path)) for path in paths]
    if not loaded:
        raise ValueError("At least four receipts are required")
    first = loaded[0][1]
    order = tuple(first.get("muscle_order", ()))
    domain = tuple(float(value) for value in first.get("capacity_domain", ()))
    condition = first.get("condition_sha256")
    if not order or len(domain) != 2 or not condition:
        raise ValueError("Receipt lacks muscle_order, capacity_domain or condition_sha256")
    minimum, epsilon, upper, provenance = None, {}, {}, {}
    for path, value in loaded:
        if (tuple(value.get("muscle_order", ())) != order
                or tuple(float(item) for item in value.get("capacity_domain", ())) != domain
                or value.get("condition_sha256") != condition):
            raise ValueError("Receipts mix conditioned states, muscle order or capacity domain")
        muscle = _epsilon_muscle(value.get("epsilon_upper"), order)
        digest = _digest_path(path)
        if muscle is None:
            support = tuple(float(item) for item in value.get("support", ()))
            if support != tuple(float(name == "Triceps") for name in order):
                raise ValueError("The non-epsilon receipt must be a triceps support witness")
            if minimum is not None:
                raise ValueError("Exactly one triceps support receipt is required")
            minimum, provenance["minimum_triceps"] = _capacity_from_report(value), digest
        else:
            if muscle in epsilon:
                raise ValueError(f"Duplicate epsilon receipt for {muscle}")
            epsilon[muscle] = _capacity_from_report(value)
            upper[muscle] = float(value["epsilon_upper"][order.index(muscle)])
            provenance[muscle] = digest
    if minimum is None or set(epsilon) != set(order) - {"Triceps"}:
        raise ValueError("Need one triceps support receipt and one epsilon receipt per other muscle")
    result = TricepsExchangeSlice(str(condition), checkpoint_cycle, order, domain, minimum, epsilon, upper, provenance)
    result.validate()
    return result


def slice_from_summary(summary_path: Path, *, checkpoint_cycle: int, capacity_domain: tuple[float, float],
                       epsilon_upper: float) -> TricepsExchangeSlice:
    """Recover a slice from an immutable pilot summary when raw receipts are absent.

    The summary must retain per-witness hashes, numerical-gate decisions, the
    support vector, and the exact ε bound.  It is therefore sufficient for a
    reproducible *archival* record, but raw receipts remain preferred.
    """
    path = Path(summary_path).resolve(strict=True)
    summary = _load(path)
    order = tuple(summary.get("muscle_order", ()))
    condition = summary.get("condition_sha256")
    if not order or not condition:
        raise ValueError("Summary lacks muscle_order or condition_sha256")
    minimum, epsilon, provenance = None, {}, {}
    expected_support = tuple(float(name == "Triceps") for name in order)
    for row in summary.get("reports", []):
        if not row.get("numeric_gate_passed"):
            continue
        support = tuple(float(item) for item in row.get("support", ()))
        capacity = tuple(float(item) for item in row.get("capacity_ratio", ()))
        digest = row.get("sha256")
        if len(capacity) != len(order) or not isinstance(digest, str) or len(digest) != 64:
            continue
        muscle = _epsilon_muscle(row.get("epsilon_upper"), order)
        if muscle is None and support == expected_support:
            if minimum is not None:
                raise ValueError("Summary contains multiple accepted triceps support witnesses")
            minimum, provenance["minimum_triceps"] = capacity, digest
            continue
        if muscle is not None and abs(float(row["epsilon_upper"][order.index(muscle)]) - epsilon_upper) <= 1e-12:
            if muscle in epsilon:
                raise ValueError(f"Summary contains duplicate epsilon witness for {muscle}")
            epsilon[muscle], provenance[muscle] = capacity, digest
    if minimum is None or set(epsilon) != set(order) - {"Triceps"}:
        raise ValueError("Summary does not contain one accepted triceps support and every requested epsilon witness")
    result = TricepsExchangeSlice(str(condition), checkpoint_cycle, order, capacity_domain, minimum, epsilon,
                                  {name: float(epsilon_upper) for name in epsilon}, provenance)
    result.validate()
    return result


def serialize_slice(value: TricepsExchangeSlice) -> dict:
    value.validate()
    body = {
        "schema_version": 1,
        "kind": "exact_conditioned_triceps_exchange_slice",
        "condition_sha256": value.condition_sha256,
        "checkpoint_cycle": value.checkpoint_cycle,
        "muscle_order": list(value.muscle_order),
        "capacity_domain": list(value.capacity_domain),
        "minimum_triceps_capacity": list(value.minimum_triceps_capacity),
        "epsilon_capacity": {name: list(capacity) for name, capacity in value.epsilon_capacity.items()},
        "epsilon_upper": dict(value.epsilon_upper),
        "witness_sha256": dict(value.witness_sha256),
        "exchange_slopes": value.exchange_slopes(),
    }
    return {**body, "slice_sha256": sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def deserialize_slice(path: Path) -> TricepsExchangeSlice:
    value = _load(path)
    if value.get("kind") != "exact_conditioned_triceps_exchange_slice":
        raise ValueError("Wrong exchange-slice kind")
    result = TricepsExchangeSlice(
        str(value["condition_sha256"]), int(value["checkpoint_cycle"]), tuple(value["muscle_order"]),
        tuple(float(item) for item in value["capacity_domain"]),
        tuple(float(item) for item in value["minimum_triceps_capacity"]),
        {str(name): tuple(float(item) for item in capacity) for name, capacity in value["epsilon_capacity"].items()},
        {str(name): float(item) for name, item in value["epsilon_upper"].items()},
        {str(name): str(item) for name, item in value["witness_sha256"].items()},
    )
    result.validate()
    if value.get("slice_sha256") != serialize_slice(result)["slice_sha256"]:
        raise ValueError("Exchange-slice checksum mismatch")
    return result
