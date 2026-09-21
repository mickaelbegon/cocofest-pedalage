import pytest

from cocofest.optimization.independent_arm_isokinetic import (
    IndependentArmOutcome,
    IndependentArmRequest,
    runtime_resistance_contract,
    synthesize_independent_arms,
)


def _outcome(side, resistance, *, wall=2.0, feasible=True):
    return IndependentArmOutcome(
        IndependentArmRequest(side, -2 * 3.141592653589793, 0.0, resistance),
        produced_work_j=10 if side == "right" else 9,
        final_capacity_ratio=.7 if side == "right" else .6,
        maximum_fatigue=.3 if side == "right" else .4,
        feasible=feasible, objective=1.2 if side == "right" else 1.1,
        solver_time_s=wall - .1, wall_time_s=wall,
    )


def test_synthesis_keeps_ocps_independent_and_accounts_for_parallel_time():
    report = synthesize_independent_arms(_outcome("right", .2, wall=2), _outcome("left", .1, wall=3))
    assert report["architecture"] == "two_independent_unilateral_isokinetic_ocps_v1"
    assert report["both_feasible"]
    assert report["total_produced_work_j"] == 19
    assert report["parallel_wall_time_s"] == 3
    assert report["serial_wall_time_s"] == 5
    assert report["worst_final_capacity_ratio"] == .6
    assert report["equivalent_resistance_imbalance_nm"] == pytest.approx(.1)
    assert report["right_imposed_work_j"] == pytest.approx(.2 * 2 * 3.141592653589793)


def test_synthesis_refuses_unsynchronized_phase_or_speed():
    right, left = _outcome("right", .2), _outcome("left", .1)
    bad = IndependentArmOutcome(
        IndependentArmRequest("left", -5.0, 0.0, .1), 9, .6, .4, True, 1.1, 1.9, 2.0)
    with pytest.raises(ValueError, match="angular velocity"):
        synthesize_independent_arms(right, bad)
    wrong_turns = IndependentArmOutcome(
        IndependentArmRequest("left", -2 * 3.141592653589793, 0.0, .1, turns=2),
        9, .6, .4, True, 1.1, 1.9, 2.0)
    with pytest.raises(ValueError, match="number of turns"):
        synthesize_independent_arms(right, wrong_turns)
    assert not synthesize_independent_arms(right, _outcome("left", .1, feasible=False))["both_feasible"]


@pytest.mark.parametrize("backend", ["ipopt", "acados"])
def test_runtime_work_target_requires_mutable_terminal_bounds(backend):
    current = runtime_resistance_contract(backend, terminal_work_bound_updatable=False)
    assert not current["runtime_equivalent_resistance_change_supported"]
    assert current["requires_recompilation"]
    future = runtime_resistance_contract(backend, terminal_work_bound_updatable=True)
    assert future["runtime_equivalent_resistance_change_supported"]
    assert not future["requires_recompilation"]
