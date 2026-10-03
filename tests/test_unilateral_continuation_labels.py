"""Scientific partition and conservative endpoint-attribution contracts."""
import numpy as np
from scripts.run_unilateral_continuation_labels import action_design, classify_terminal


def test_actions_preserve_scale_and_keep_held_out_directions_disjoint():
    actions = action_design()
    assert len(actions) == 9
    for action in actions:
        values = np.asarray(list(action["weights"].values()))
        assert np.all(values > 0)
        assert np.isclose(values.mean(), 1.)
        assert abs(sum(action["centered_log_direction"])) < 1e-12
    train = [a for a in actions if a["partition"] == "train"]
    holdout = [a for a in actions if a["partition"] == "holdout"]
    assert len(train) == 6 and len(holdout) == 2
    assert all(not np.allclose(t["centered_log_direction"], h["centered_log_direction"])
               for t in train for h in holdout)
    directions = np.asarray([a["centered_log_direction"] for a in train[::2]])
    assert np.allclose(directions @ directions.T, np.eye(3))


def test_local_infeasibility_without_fatigue_counterfactual_is_not_label():
    row = {"zero_objective_feasibility_probe": {"available": True, "constraints_unchanged": True,
        "feasible_witness": False, "solver_stats": {"return_status": "Infeasible_Problem_Detected"}}}
    assert classify_terminal(row, censored=False)[0] == "numerical_stop"
    row["counterfactual_probes"] = {"fatigue_rest": {"feasible_witness": True, "physical_rho_advanced": False}}
    assert classify_terminal(row, censored=False)[0] == "physiological_failure_certified"
    row["zero_objective_feasibility_probe"]["feasible_witness"] = True
    assert classify_terminal(row, censored=False)[0] == "numerical_stop"


def test_budget_hit_is_censored_and_not_exact_endurance():
    assert classify_terminal({}, censored=True)[0] == "right_censored"
    assert classify_terminal({}, censored=False)[0] == "numerical_stop"
