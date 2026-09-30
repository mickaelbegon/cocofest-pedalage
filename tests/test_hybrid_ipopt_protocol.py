import pytest

from cocofest.optimization.hybrid_ipopt_protocol import (
    HybridStoppingPolicy,
    LbfgsBlockObservation,
    decide_lbfgs_handoff,
    extract_ipopt_block_observation,
)


def _observation(block, iterations, objective, violation=1e-5):
    return LbfgsBlockObservation(block, iterations, objective, violation, f"block-{block}.npz")


def test_switches_only_when_objective_and_independent_audit_pass():
    decision = decide_lbfgs_handoff(
        [_observation(1, 25, 100.0), _observation(2, 50, 100.00005)],
        HybridStoppingPolicy(),
    )
    assert decision["switch_to_exact"] is True
    assert decision["objective_window_iterations"] == 25
    assert decision["reason"] == "objective_stable_and_constraints_reasonable"


def test_refuses_small_objective_change_when_audit_is_too_large():
    decision = decide_lbfgs_handoff(
        [_observation(1, 25, 100.0), _observation(2, 50, 100.00005, 2e-4)],
        HybridStoppingPolicy(),
    )
    assert decision["switch_to_exact"] is False
    assert decision["reason"] == "audited_constraint_violation_too_large"


def test_refuses_non_20_to_30_iteration_window():
    decision = decide_lbfgs_handoff(
        [_observation(1, 10, 100.0), _observation(2, 20, 100.0)],
        HybridStoppingPolicy(),
    )
    assert decision["reason"] == "objective_window_not_20_to_30_iterations"


def test_structured_extraction_refuses_ipopt_inf_pr_as_audit():
    payload = {
        "results": [{
            "nlp_solver_stats": [{
                "return_status": "Maximum_Iterations_Exceeded",
                "iteration_diagnostics": {"obj": {"final": 12.0}, "inf_pr": {"final": 1e-12}},
            }],
        }],
    }
    with pytest.raises(ValueError, match="independent output audit"):
        extract_ipopt_block_observation(payload, checkpoint_path="x.npz", block=1, cumulative_iterations=25)


def test_structured_extraction_reads_exporter_audit():
    payload = {
        "results": [{
            "uncertified_output_audit": {"effective_primal_infeasibility": 3e-5},
            "nlp_solver_stats": [{
                "return_status": "Maximum_Iterations_Exceeded",
                "iteration_diagnostics": {"obj": {"final": 12.0}},
            }],
        }],
    }
    result = extract_ipopt_block_observation(payload, checkpoint_path="x.npz", block=1, cumulative_iterations=25)
    assert result.objective == 12.0
    assert result.audited_constraint_violation == 3e-5


def test_structured_extraction_reads_single_shot_window_feasibility_audit():
    """Limited-memory single shots export their independent audit per window."""

    payload = {
        "results": [{
            "windows": [{"feasibility": {"effective_primal_infeasibility": 2e-3}}],
            "nlp_solver_stats": [{
                "return_status": "Maximum_Iterations_Exceeded",
                "iteration_diagnostics": {"obj": {"final": 12.0}, "inf_pr": {"final": 1e-12}},
            }],
        }],
    }
    result = extract_ipopt_block_observation(payload, checkpoint_path="x.npz", block=1, cumulative_iterations=25)
    assert result.objective == 12.0
    assert result.audited_constraint_violation == 2e-3
