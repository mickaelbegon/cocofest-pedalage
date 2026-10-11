from hashlib import sha256

import numpy as np
import pytest

from cocofest.optimization.capacity_frontier import (
    CapacityFrontierDirection,
    CertifiedCapacityFrontierPoint,
    build_capacity_feasibility_frontier_plan,
    build_capacity_frontier_plan,
    pareto_nondominated,
    support_directions,
)


HASH = sha256(b"context").hexdigest()


def point(identifier, cycles, capacity, direction="uniform"):
    return CertifiedCapacityFrontierPoint(identifier, direction, cycles, tuple(capacity), HASH,
                                         sha256((identifier + "-run").encode()).hexdigest(),
                                         sha256((identifier + "-frozen").encode()).hexdigest())


def test_support_directions_cover_symmetric_simplex_families_reproducibly():
    first = support_directions(("Biceps", "Triceps", "Deltoid"), random_directions=2, seed=7)
    second = support_directions(("Biceps", "Triceps", "Deltoid"), random_directions=2, seed=7)
    assert first == second
    assert [row.family for row in first].count("single") == 3
    assert [row.family for row in first].count("pair") == 3
    assert first[0].direction_id == "uniform"
    assert all(np.isclose(sum(row.support), 1.) and min(row.support) >= 0 for row in first)


def test_nondominated_requires_simultaneous_endurance_and_capacity_improvement():
    weaker = point("weaker", 100, (.6, .6))
    better = point("better", 101, (.6, .7))
    tradeoff = point("tradeoff", 102, (.4, .95))
    assert pareto_nondominated((weaker, better, tradeoff)) == (better, tradeoff)


def test_nondominated_rejects_context_mixing_and_invalid_receipt():
    left, right = point("left", 1, (.8, .8)), point("right", 1, (.8, .8))
    object.__setattr__(right, "context_sha256", "f" * 64)
    with pytest.raises(ValueError, match="contexts"):
        pareto_nondominated((left, right))


def test_plan_makes_endurance_primary_and_fatigue_objective_prohibited():
    plan = build_capacity_frontier_plan(("Biceps", "Triceps"), context_sha256=HASH,
                                        maximum_cycles=300, random_directions=1, seed=3)
    assert plan["primary_value"] == "maximise_certified_continuation_cycles_at_fixed_task"
    assert plan["prohibited_primary_value"] == "maximise_weighted_fatigue"
    assert plan["directions"][0]["direction_id"] == "uniform"
    assert len(plan["plan_sha256"]) == 64


def test_feasibility_atlas_conditions_fast_state_and_uses_initial_capacity():
    plan = build_capacity_feasibility_frontier_plan(
        ("Biceps", "Triceps"), context_sha256=HASH,
        fast_condition_sha256=sha256(b"fast state").hexdigest(), random_directions=1,
    )
    assert plan["kind"] == "offline_local_cycle_capacity_feasibility_frontier_plan"
    assert plan["capacity_coordinate"] == "A_initial/A_rest"
    assert "shared q" in plan["slow_state_parameterization"]
    assert plan["status_rules"]["solver_failure"] == "indeterminate_not_infeasible"


def test_invalid_direction_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        CapacityFrontierDirection("bad", (-1., 2.), "bad")
