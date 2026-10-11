"""Offline, auditable designs for capacity-frontier experiments.

This module deliberately does *not* solve an OCP.  It specifies the finite
family of supporting directions to be solved offline and provides the exact
rules used to retain observations afterwards.  In particular, an IPOPT
failure is never a frontier point and cannot dominate another observation.

This module distinguishes two objects which must not be conflated:

* a local **cycle-feasibility capacity atlas**, whose variables are capacities
  at the *start* of one conditioned cycle;
* a Pareto description of observed RHO continuations, which is useful but is
  not itself an atlas of feasible initial capacities.

Neither object is the result of maximizing fatigue in a single cycle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1


def _names(muscles: Sequence[str]) -> tuple[str, ...]:
    values = tuple(muscles)
    if len(values) < 2 or len(set(values)) != len(values):
        raise ValueError("At least two distinct muscles are required")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("Muscle names must be non-empty strings")
    return values


def _vector(values: Sequence[float], *, name: str, dimension: int | None = None) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not vector.size or (dimension is not None and vector.size != dimension):
        raise ValueError(f"{name} must be a non-empty vector with the expected dimension")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def _unit_simplex(vector: Sequence[float], *, name: str, dimension: int) -> tuple[float, ...]:
    vector = _vector(vector, name=name, dimension=dimension)
    if np.any(vector < 0) or not float(vector.sum()) > 0:
        raise ValueError(f"{name} must be non-negative with a positive sum")
    return tuple((vector / vector.sum()).tolist())


@dataclass(frozen=True)
class CapacityFrontierDirection:
    """A non-negative support direction for a terminal capacity trade-off."""

    direction_id: str
    support: tuple[float, ...]
    family: str

    def __post_init__(self) -> None:
        if not isinstance(self.direction_id, str) or not self.direction_id:
            raise ValueError("direction_id must be non-empty")
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("family must be non-empty")
        _unit_simplex(self.support, name="support", dimension=len(self.support))


@dataclass(frozen=True)
class CertifiedCapacityFrontierPoint:
    """One observed endpoint, comparable only inside a fixed physical context.

    ``certified_cycles`` is the number of completed *physical* cycles.  The
    endpoint capacity comes from that last certified boundary.  A point may
    be admitted only after an independent frozen-state feasibility audit; the
    corresponding receipt is retained rather than replacing it by a boolean
    inferred from solver status.
    """

    point_id: str
    direction_id: str
    certified_cycles: int
    terminal_capacity_ratio: tuple[float, ...]
    context_sha256: str
    continuation_receipt_sha256: str
    frozen_feasibility_receipt_sha256: str

    def validate(self, *, dimension: int, context_sha256: str | None = None) -> None:
        if not isinstance(self.point_id, str) or not self.point_id:
            raise ValueError("point_id must be non-empty")
        if not isinstance(self.direction_id, str) or not self.direction_id:
            raise ValueError("direction_id must be non-empty")
        if type(self.certified_cycles) is not int or self.certified_cycles < 0:
            raise ValueError("certified_cycles must be a non-negative integer")
        capacity = _vector(self.terminal_capacity_ratio, name="terminal_capacity_ratio", dimension=dimension)
        if np.any(capacity < 0) or np.any(capacity > 1 + 1e-9):
            raise ValueError("terminal capacity ratios must be in [0, 1]")
        for name, value in (("context_sha256", self.context_sha256),
                            ("continuation_receipt_sha256", self.continuation_receipt_sha256),
                            ("frozen_feasibility_receipt_sha256", self.frozen_feasibility_receipt_sha256)):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")
        if context_sha256 is not None and self.context_sha256 != context_sha256:
            raise ValueError("Capacity frontier points mix physical contexts")


def support_directions(muscles: Sequence[str], *, random_directions: int = 0,
                       seed: int = 0) -> tuple[CapacityFrontierDirection, ...]:
    """Return a deterministic, symmetric finite simplex design.

    Unit supports identify individual bottlenecks; pair supports test simple
    substitutions; the uniform direction preserves the aggregate baseline.
    Optional Dirichlet supports fill the simplex without treating arbitrary
    negative "weights" as a physiological preference.
    """
    muscles = _names(muscles)
    if type(random_directions) is not int or random_directions < 0:
        raise ValueError("random_directions must be a non-negative integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    dimension = len(muscles)
    rows: list[CapacityFrontierDirection] = []

    def append(direction_id: str, support: Sequence[float], family: str) -> None:
        rows.append(CapacityFrontierDirection(direction_id, _unit_simplex(
            support, name="support", dimension=dimension), family))

    append("uniform", np.ones(dimension), "uniform")
    for index, muscle in enumerate(muscles):
        support = np.zeros(dimension)
        support[index] = 1.
        append(f"single-{muscle}", support, "single")
    for left in range(dimension):
        for right in range(left + 1, dimension):
            support = np.zeros(dimension)
            support[[left, right]] = 1.
            append(f"pair-{muscles[left]}-{muscles[right]}", support, "pair")
    generator = np.random.Generator(np.random.PCG64(seed))
    for index in range(random_directions):
        append(f"simplex-{index + 1:03d}", generator.dirichlet(np.ones(dimension)), "simplex")
    return tuple(rows)


def frontier_support_score(point: CertifiedCapacityFrontierPoint,
                           direction: CapacityFrontierDirection) -> float:
    """Report a secondary terminal-capacity score, never an endurance score."""
    capacity = _vector(point.terminal_capacity_ratio, name="terminal_capacity_ratio")
    support = _vector(direction.support, name="support", dimension=capacity.size)
    return float(support @ capacity)


def pareto_nondominated(points: Sequence[CertifiedCapacityFrontierPoint], *,
                        capacity_tolerance: float = 1e-9) -> tuple[CertifiedCapacityFrontierPoint, ...]:
    """Keep points not dominated in (certified cycles, terminal capacities).

    A point dominates another only if it has at least as many certified cycles
    and no smaller capacity in every muscle, with one strict improvement.  A
    direction is intentionally not part of the comparison: it is provenance,
    whereas the observed continuation is the physical outcome.
    """
    rows = tuple(points)
    if not rows:
        return ()
    tolerance = float(capacity_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("capacity_tolerance must be finite and non-negative")
    dimension = len(rows[0].terminal_capacity_ratio)
    context = rows[0].context_sha256
    ids = set()
    for point in rows:
        point.validate(dimension=dimension, context_sha256=context)
        if point.point_id in ids:
            raise ValueError("Frontier point ids must be unique")
        ids.add(point.point_id)
    kept = []
    for candidate in rows:
        candidate_capacity = np.asarray(candidate.terminal_capacity_ratio)
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            other_capacity = np.asarray(other.terminal_capacity_ratio)
            no_worse = (other.certified_cycles >= candidate.certified_cycles
                        and np.all(other_capacity >= candidate_capacity - tolerance))
            strictly_better = (other.certified_cycles > candidate.certified_cycles
                               or np.any(other_capacity > candidate_capacity + tolerance))
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            kept.append(candidate)
    return tuple(kept)


def build_capacity_frontier_plan(muscles: Sequence[str], *, context_sha256: str,
                                 maximum_cycles: int, random_directions: int = 0,
                                 seed: int = 0, endpoint_tolerance_cycles: int = 0) -> dict:
    """Create an immutable offline plan; execution and certification are separate.

    The plan explicitly rejects the tempting but incorrect primary objective
    ``max sum(c_m * (1-A_m/A_rest))``: it rewards fatigue.  Fatigue is used
    only to *describe* an observed endpoint.  The primary objective is the
    number of independently certified physical cycles.
    """
    muscles = _names(muscles)
    if not isinstance(context_sha256, str) or len(context_sha256) != 64:
        raise ValueError("context_sha256 must be a SHA-256 hex digest")
    if type(maximum_cycles) is not int or maximum_cycles <= 0:
        raise ValueError("maximum_cycles must be a positive integer")
    if type(endpoint_tolerance_cycles) is not int or endpoint_tolerance_cycles < 0:
        raise ValueError("endpoint_tolerance_cycles must be a non-negative integer")
    directions = support_directions(muscles, random_directions=random_directions, seed=seed)
    value = {
        "schema_version": SCHEMA_VERSION,
        "kind": "offline_certified_continuation_policy_pareto_plan",
        "context_sha256": context_sha256,
        "muscle_order": list(muscles),
        "maximum_cycles": maximum_cycles,
        "endpoint_tolerance_cycles": endpoint_tolerance_cycles,
        "directions": [asdict(direction) for direction in directions],
        "primary_value": "maximise_certified_continuation_cycles_at_fixed_task",
        "secondary_value": "within_primary_tolerance_maximise_support_dot_terminal_capacity_ratio",
        "prohibited_primary_value": "maximise_weighted_fatigue",
        "required_certificates": [
            "exact_start_state_and_stimulation_history",
            "full_RHO_continuation_receipt",
            "independent_frozen_state_cycle_feasibility_receipt",
            "DOP853_replay_for_reported_endpoint",
        ],
        "status_rules": {
            "ipopt_failure": "indeterminate_not_a_frontier_limit",
            "maximum_cycles_reached": "right_censored_not_a_failure_limit",
            "frontier_point": "requires_all_required_certificates",
        },
        "online_use": "offline_only_until_holdout_replay_and_RHO_ablation_pass",
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**value, "plan_sha256": sha256(canonical).hexdigest()}


def build_capacity_feasibility_frontier_plan(
        muscles: Sequence[str], *, context_sha256: str, fast_condition_sha256: str,
        random_directions: int = 0, seed: int = 0) -> dict:
    """Plan the proper local atlas: minimal feasible *initial* capacities.

    The executor must parameterize the compatible slow Ding states on their
    one-dimensional-per-muscle manifold.  ``Cn``, ``F`` and the stimulation
    history are not decision variables in this atlas: their exact conditioned
    values are identified by ``fast_condition_sha256``.  This avoids a solver
    obtaining force from an unprepared initial state.
    """
    muscles = _names(muscles)
    for name, digest in (("context_sha256", context_sha256),
                         ("fast_condition_sha256", fast_condition_sha256)):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{name} must be a SHA-256 hex digest")
    directions = support_directions(muscles, random_directions=random_directions, seed=seed)
    value = {
        "schema_version": SCHEMA_VERSION,
        "kind": "offline_local_cycle_capacity_feasibility_frontier_plan",
        "context_sha256": context_sha256,
        "fast_condition_sha256": fast_condition_sha256,
        "muscle_order": list(muscles),
        "directions": [asdict(direction) for direction in directions],
        "capacity_coordinate": "A_initial/A_rest",
        "slow_state_parameterization": (
            "per-muscle exact Ding manifold: A, Tau1 and Km co-vary through shared q"),
        "conditioned_fast_state": "Cn, F and complete stimulation history fixed from exact receipt",
        "cycle_contract": ["geometric closure modulo 2pi", "cadence", "net cycle work",
                           "effective torque/assistance bounds", "PW and slew bounds",
                           "next-half-cycle guard"],
        "support_problem": "minimise_support_dot_initial_capacity_ratio_subject_to_exact_cycle_feasibility",
        "nonconvex_coverage": "add_epsilon_constraint_solves_and_multiple_starts_before_claiming_a_front",
        "required_confirmation": "independent_feasibility_rebuild_and_dense_DOP853_replay_for_each_witness",
        "status_rules": {
            "solver_failure": "indeterminate_not_infeasible",
            "feasible_witness": "requires_all_constraint_audits_and_replay",
            "interpolation": "guidance_only_not_a_feasibility_certificate",
        },
        "online_use": "disabled_until_pilot_holdout_validates_margin_pitches",
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**value, "plan_sha256": sha256(canonical).hexdigest()}
