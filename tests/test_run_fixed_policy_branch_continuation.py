from scripts.run_fixed_policy_branch_continuation import classify_terminal


def test_terminal_classification_requires_both_frozen_infeasibility_and_fatigue_rest_witness():
    terminal = {"zero_objective_feasibility_probe": {"available": True, "constraints_unchanged": True,
                "feasible_witness": False, "solver_stats": {"return_status": "Infeasible_Problem_Detected"}},
                "counterfactual_probes": {"fatigue_rest": {"feasible_witness": True, "physical_rho_advanced": False}}}
    assert classify_terminal(terminal, censored=False)[0] == "local_frozen_physiological_failure"
    assert classify_terminal({}, censored=False)[0] == "numerical_or_physical_indeterminate"
    assert classify_terminal({}, censored=True)[0] == "right_censored"
