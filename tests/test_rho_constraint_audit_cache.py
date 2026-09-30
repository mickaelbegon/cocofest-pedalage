"""Constraint labels may be cached; numerical RHO certification must stay live."""

from types import SimpleNamespace

import numpy as np
import pytest

from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as periodic


def _solution():
    penalty = SimpleNamespace(
        name="terminal_angle", type="CUSTOM", node_idx=[2], rows_by_node={2: 2}
    )
    phase = SimpleNamespace(ns=2, g_internal=[], g=[penalty])
    ocp = SimpleNamespace(g_internal=[], g=[], nlp=[phase])
    interface = SimpleNamespace(
        nlp={"g": object()}, calls=0,
        limits={"lbg": np.zeros(2), "ubg": np.ones(2),
                "lbx": np.zeros(1), "ubx": np.ones(1)},
    )

    def get_all_penalties(owner, penalties, get_bounds=False):
        interface.calls += 1
        return {node: np.zeros((rows, 1)) for node, rows in penalties[0].rows_by_node.items()}, {}

    interface.get_all_penalties = get_all_penalties
    ocp.ocp_solver = interface
    return SimpleNamespace(
        ocp=ocp, constraints=np.array([0.0, 1.000001]), vector=np.zeros(1),
        decision_states=lambda **kwargs: {"theta": np.zeros((1, 3))},
        decision_controls=lambda **kwargs: {"PW": np.zeros((1, 2))},
    )


def test_cached_row_labels_match_fresh_labels_at_every_index():
    solution = _solution()
    interface = solution.ocp.ocp_solver
    expected = [periodic._constraint_vector_block(solution, index) for index in (0, 1, 2)]
    assert interface.calls == 1
    for index, label in enumerate(expected):
        del interface._cocofest_constraint_row_layout
        assert periodic._constraint_vector_block(solution, index) == label
    label = periodic._constraint_vector_block(solution, 0)
    label["declared_nodes"].append(99)
    assert periodic._constraint_vector_block(solution, 0)["declared_nodes"] == [2]


@pytest.mark.parametrize("change", ["graph", "nodes", "penalty", "stage_order", "empty_slot"])
def test_constraint_layout_rebuilds_after_structure_change(change):
    solution = _solution()
    interface = solution.ocp.ocp_solver
    periodic._constraint_vector_block(solution, 0)
    if change == "graph":
        interface.nlp["g"] = object()
        solution.ocp.nlp[0].g[0].rows_by_node = {2: 3}
    elif change == "nodes":
        solution.ocp.nlp[0].g[0].node_idx = [1]
    elif change == "penalty":
        solution.ocp.nlp[0].g[0] = SimpleNamespace(**vars(solution.ocp.nlp[0].g[0]))
    elif change == "stage_order":
        interface.stage_wise_multi_thread_constraints = True
    else:
        solution.ocp.nlp[0].g.insert(0, None)
    result = periodic._constraint_vector_block(solution, 0)
    assert interface.calls == 2
    if change == "graph":
        assert result["global_stop"] == 3
    if change == "nodes":
        assert result["declared_nodes"] == [1]
    if change == "empty_slot":
        assert result["penalty_index"] == 1


def test_numerical_audit_reads_changed_bounds_and_constraint_values_with_cached_labels():
    solution = _solution()
    interface = solution.ocp.ocp_solver
    first = periodic._solution_feasibility_summary(solution, tolerance=1e-6)
    assert first["passes_tolerance"]
    interface.limits["ubg"][1] = 0.5
    second = periodic._solution_feasibility_summary(solution, tolerance=1e-6)
    assert not second["passes_tolerance"]
    assert second["constraint_bound_violation"] == pytest.approx(0.500001)
    solution.constraints[1] = 0.5
    third = periodic._solution_feasibility_summary(solution, tolerance=1e-6)
    assert third["passes_tolerance"]
    assert interface.calls == 1


def test_missing_backend_graph_does_not_cache_layout():
    solution = _solution()
    interface = solution.ocp.ocp_solver
    interface.nlp = {}
    periodic._constraint_vector_block(solution, 0)
    periodic._constraint_vector_block(solution, 1)
    assert interface.calls == 2
