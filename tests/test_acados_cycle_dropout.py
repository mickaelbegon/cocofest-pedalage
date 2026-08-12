import argparse
import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "docs/cycling_solver_benchmark/analyze_acados_cycle_dropout.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_acados_cycle_dropout", SCRIPT)
dropout = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = dropout
SPEC.loader.exec_module(dropout)


def test_parse_cycle_indices_rejects_first_cycle_and_deduplicates():
    assert dropout.parse_cycle_indices("10, 20,10") == (10, 20)
    with pytest.raises(argparse.ArgumentTypeError):
        dropout.parse_cycle_indices("1,2")


def test_dropout_summary_counts_only_jointly_physical_rollouts():
    rows = [
        {
            "baseline": {
                "passes_terminal_angle": True,
                "passes_velocity_bounds": True,
            },
            "hold_previous": {
                "passes_terminal_angle": True,
                "passes_velocity_bounds": False,
            },
            "hold_minus_baseline_terminal_theta_rad": 0.1,
            "hold_minus_baseline_terminal_omega_rad_s": -0.2,
            "hold_minus_baseline_fatigue_auc": 0.01,
            "hold_minus_baseline_fatigue_objective": 2.0,
            "hold_minus_baseline_capacity_ratio": {
                "Biceps": -0.003,
                "Triceps": 0.001,
            },
        },
        {
            "baseline": {
                "passes_terminal_angle": False,
                "passes_velocity_bounds": True,
            },
            "hold_previous": {
                "passes_terminal_angle": True,
                "passes_velocity_bounds": True,
            },
            "hold_minus_baseline_terminal_theta_rad": -0.3,
            "hold_minus_baseline_terminal_omega_rad_s": 0.4,
            "hold_minus_baseline_fatigue_auc": -0.02,
            "hold_minus_baseline_fatigue_objective": -1.0,
            "hold_minus_baseline_capacity_ratio": {
                "Biceps": 0.002,
                "Triceps": -0.004,
            },
        },
    ]

    summary = dropout.summarize_rows(rows)

    assert summary["cycle_count"] == 2
    assert summary["baseline_physical_count"] == 1
    assert summary["hold_previous_physical_count"] == 1
    assert summary["absolute_hold_minus_baseline_terminal_theta_rad"]["maximum"] == pytest.approx(0.3)
    assert summary["hold_minus_baseline_capacity_ratio"]["Biceps"]["minimum"] == pytest.approx(-0.003)


def test_dropout_script_documents_that_failed_nlp_states_are_not_shifted():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "last certified state" in text
    assert "failed NLP state trajectory must never be shifted" in text
    assert "future PW are not reoptimized" in text


def test_attempt_analysis_separates_iteration_cap_and_fallback_transition():
    def attempt(rho, status, advanced, iterations, certifier="target_solver", infeasibility=0.0):
        return {
            "target_rho": rho,
            "status": status,
            "advanced": advanced,
            "iterations": iterations,
            "certifier": certifier,
            "feasibility": {"effective_primal_infeasibility": infeasibility},
        }

    result = {
        "solver_attempt_accounting": {
            "attempts": [
                attempt(1, 0, True, 3),
                attempt(2, 2, False, 30, infeasibility=2e-3),
                attempt(2, 2, True, 30, certifier="ipopt_radau"),
                attempt(3, 2, False, 30, infeasibility=3e-3),
                attempt(3, 0, True, 8),
                attempt(4, 0, True, 4),
            ]
        }
    }

    analysis = dropout.analyze_solver_attempts(result, budgets=(3, 5, 10))

    assert analysis["native_success_count"] == 3
    assert analysis["historical_success_fraction_within_budget"] == {
        "3": pytest.approx(1 / 3),
        "5": pytest.approx(2 / 3),
        "10": pytest.approx(1.0),
    }
    assert analysis["failed_native_primal_infeasibility_distribution"]["strict_1e-5_count"] == 0
    assert analysis["next_rho_after_fallback"]["first_attempt_failure_fraction"] == pytest.approx(1.0)
