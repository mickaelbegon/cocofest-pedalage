"""Auxiliary OCP graph for a local Ding capacity-feasibility support solve.

The production RHO graph is never edited.  This class owns copied numeric
bounds and an auxiliary objective, adds only the two equalities per muscle
needed to retain the exact slow Ding manifold, and leaves every original
physical constraint untouched.  It intentionally makes no claim of global
optimality: every returned point still needs an independent rebuild and dense
replay.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .conditional_slow_manifold_adapter import SlowManifoldInitialStateAdapter


class CapacityFrontierSupportProblem:
    """Minimise a positive support of initial capacities under one-cycle feasibility."""

    def __init__(self, *, graph, adapter: SlowManifoldInitialStateAdapter,
                 support: Sequence[float], epsilon_upper: Sequence[float] | None = None,
                 fixed_capacity: Sequence[float] | None = None):
        import casadi as ca

        required_adapter_attributes = ("verify", "muscle_order", "manifolds", "indices", "coefficients",
                                       "center", "limits", "condition_sha256")
        if any(not hasattr(adapter, attribute) for attribute in required_adapter_attributes):
            raise TypeError("adapter must provide the SlowManifoldInitialStateAdapter contract")
        adapter.verify()
        support = np.asarray(support, float)
        if (support.shape != (len(adapter.muscle_order),) or not np.all(np.isfinite(support))
                or np.any(support < 0) or not support.sum() > 0):
            raise ValueError("support must be finite, non-negative and non-zero per muscle")
        support /= support.sum()
        if epsilon_upper is None:
            epsilon_upper = np.full(len(adapter.muscle_order), np.nan)
        epsilon_upper = np.asarray(epsilon_upper, float)
        if (epsilon_upper.shape != (len(adapter.muscle_order),)
                or np.any(np.isinf(epsilon_upper))
                or np.any(np.isfinite(epsilon_upper) & ((epsilon_upper < 0) | (epsilon_upper > 1)))):
            raise ValueError("epsilon_upper must contain one NaN or physical [0,1] upper bound per muscle")
        if fixed_capacity is None:
            fixed_capacity = np.full(len(adapter.muscle_order), np.nan)
        fixed_capacity = np.asarray(fixed_capacity, float)
        if (fixed_capacity.shape != (len(adapter.muscle_order),) or np.any(np.isinf(fixed_capacity))
                or np.any(np.isfinite(fixed_capacity) & ((fixed_capacity < 0) | (fixed_capacity > 1)))):
            raise ValueError("fixed_capacity must contain one NaN or physical [0,1] equality per muscle")
        x, g = graph["x"], graph["g"]
        if int(x.numel()) != adapter.limits["lbx"].size:
            raise ValueError("Adapter and graph decision dimensions differ")
        original_lbg = np.asarray(adapter.limits["lbg"], float).reshape(-1)
        original_ubg = np.asarray(adapter.limits["ubg"], float).reshape(-1)
        if int(g.numel()) != original_lbg.size or original_ubg.shape != original_lbg.shape:
            raise ValueError("Adapter and graph constraint dimensions differ")

        capacities, slow_equalities = [], []
        for manifold, indices, coefficients in zip(adapter.manifolds, adapter.indices, adapter.coefficients):
            a, tau1, km = (x[index] * coefficient for index, coefficient in zip(indices, coefficients))
            ratio = a / manifold.a_rest
            capacities.append(ratio)
            q = (a - manifold.a_rest) / manifold.alpha_a
            tau1_expected = manifold.tau1_rest + manifold.alpha_tau1 * q
            km_expected = manifold.km_rest + manifold.alpha_km * q
            slow_equalities.extend((tau1 - tau1_expected, km - km_expected))
        self.adapter = adapter
        self.support = support
        self.initial_capacity = ca.vertcat(*capacities)
        epsilon_indices = tuple(np.flatnonzero(np.isfinite(epsilon_upper)).tolist())
        epsilon_constraints = [capacities[index] for index in epsilon_indices]
        fixed_indices = tuple(np.flatnonzero(np.isfinite(fixed_capacity)).tolist())
        fixed_constraints = [capacities[index] for index in fixed_indices]
        self.graph = {"x": x, "g": ca.vertcat(g, *slow_equalities, *epsilon_constraints, *fixed_constraints),
                      "f": ca.dot(ca.DM(support), self.initial_capacity)}
        self.original_constraint_count = int(g.numel())
        self._evaluate = ca.Function("capacity_frontier_constraint_audit", [x], [self.graph["g"]])
        self._capacity = ca.Function("capacity_frontier_initial_capacity", [x], [self.initial_capacity])
        self._original_objective_sha256 = adapter.condition_sha256
        self._extra_lbg = np.zeros(len(slow_equalities))
        self._extra_ubg = np.zeros(len(slow_equalities))
        self.epsilon_indices = epsilon_indices
        self.epsilon_upper = epsilon_upper
        self._epsilon_lbg = np.full(len(epsilon_constraints), -np.inf)
        self._epsilon_ubg = epsilon_upper[list(epsilon_indices)]
        self.fixed_indices = fixed_indices
        self.fixed_capacity = fixed_capacity
        self._fixed_lbg = fixed_capacity[list(fixed_indices)]
        self._fixed_ubg = fixed_capacity[list(fixed_indices)]

    def limits_for_domain(self, *, lower: Sequence[float], upper: Sequence[float]) -> dict[str, np.ndarray]:
        """Relax only initial slow-state equalities to an explicit ratio box.

        The two manifold equalities in ``graph`` retain the state coupling.  A
        range rather than a fixed initial capacity is therefore an actual
        support optimization, unlike the older A-only point probes.
        """
        self.adapter.verify()
        lower = np.asarray(lower, float)
        upper = np.asarray(upper, float)
        shape = (len(self.adapter.muscle_order),)
        if (lower.shape != shape or upper.shape != shape or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper)) or np.any(lower < 0) or np.any(upper > 1)
                or np.any(lower > upper)):
            raise ValueError("lower/upper must be ordered physical capacity-ratio vectors in [0,1]")
        limits = {key: value.copy() for key, value in self.adapter.limits.items()}
        for lo, hi, manifold, indices, coefficients, center in zip(
                lower, upper, self.adapter.manifolds, self.adapter.indices,
                self.adapter.coefficients, self.adapter.center):
            low_states, high_states = manifold.states_from_capacity_ratio(float(lo)), manifold.states_from_capacity_ratio(float(hi))
            center_states = manifold.states_from_capacity_ratio(float(np.clip(center, lo, hi)))
            for low_state, high_state, center_state, index, coefficient in zip(
                    low_states, high_states, center_states, indices, coefficients):
                limits["lbx"][index] = min(low_state, high_state) / coefficient
                limits["ubx"][index] = max(low_state, high_state) / coefficient
                limits["x0"][index] = center_state / coefficient
        limits["lbg"] = np.concatenate((limits["lbg"], self._extra_lbg))
        limits["ubg"] = np.concatenate((limits["ubg"], self._extra_ubg))
        limits["lbg"] = np.concatenate((limits["lbg"], self._epsilon_lbg))
        limits["ubg"] = np.concatenate((limits["ubg"], self._epsilon_ubg))
        limits["lbg"] = np.concatenate((limits["lbg"], self._fixed_lbg))
        limits["ubg"] = np.concatenate((limits["ubg"], self._fixed_ubg))
        return limits

    def audit(self, vector, *, tolerance: float = 1e-7) -> dict:
        """Check copied bounds, original constraints and manifold equalities."""
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        vector = np.asarray(vector, float).reshape(-1)
        if vector.size != self.adapter.limits["lbx"].size or not np.all(np.isfinite(vector)):
            raise ValueError("vector has an invalid decision dimension")
        values = np.asarray(self._evaluate(vector), float).reshape(-1)
        source = self.adapter.limits
        original = values[:self.original_constraint_count]
        original_violation = max(np.max(source["lbg"] - original), np.max(original - source["ubg"]), 0.)
        slow_end = self.original_constraint_count + 2 * len(self.adapter.muscle_order)
        manifold_violation = float(np.max(np.abs(values[self.original_constraint_count:slow_end])))
        epsilon_values = values[slow_end:]
        fixed_start = slow_end + len(self.epsilon_indices)
        epsilon_values = values[slow_end:fixed_start]
        epsilon_violation = (float(np.max(np.maximum(epsilon_values - self._epsilon_ubg, 0.)))
                             if epsilon_values.size else 0.)
        fixed_values = values[fixed_start:]
        fixed_violation = (float(np.max(np.abs(fixed_values - self._fixed_lbg))) if fixed_values.size else 0.)
        capacity = np.asarray(self._capacity(vector), float).reshape(-1)
        return {"initial_capacity_ratio": capacity.tolist(),
                "support_value": float(self.support @ capacity),
                "original_constraint_max_violation": float(original_violation),
                "slow_manifold_max_residual": manifold_violation,
                "epsilon_capacity_max_violation": epsilon_violation,
                "epsilon_upper": [None if not np.isfinite(value) else float(value)
                                  for value in self.epsilon_upper],
                "fixed_capacity": [None if not np.isfinite(value) else float(value)
                                   for value in self.fixed_capacity],
                "fixed_capacity_max_violation": fixed_violation,
                "passed_discrete_auxiliary_audit": bool(max(original_violation, manifold_violation,
                                                             epsilon_violation, fixed_violation) <= tolerance),
                "certified_feasibility": False,
                "requires_independent_rebuild_and_dense_replay": True}
