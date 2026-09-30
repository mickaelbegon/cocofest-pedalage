"""Unit tests for the non-overclaiming frozen-state viability interpretation."""

from cocofest.optimization.endurance_viability import (
    assess_frozen_state_one_cycle_probe,
    normalized_constraint_deficit,
)
from scripts.probe_rho_endurance_viability import build_probe_benchmark_arguments


def _source(validated=12):
    return {"validated_cycles": validated}


def _failed_probe(residual=2e-6, tolerance=1e-7):
    return {
        "solver": "ipopt", "physical_success": False, "validated_cycles": 0,
        "status": 1,
        "solver_attempt_accounting": {"attempts": [
            {"advanced": False, "feasibility": {
                "effective_primal_infeasibility": residual,
                "feasibility_threshold": tolerance,
            }}
        ]},
    }


def test_certified_probe_establishes_one_next_cycle_but_not_global_endurance():
    report = assess_frozen_state_one_cycle_probe(
        _source(), {"solver": "ipopt", "physical_success": True, "validated_cycles": 1},
        checkpoint_completed_windows=12, launcher_returncode=0,
    )
    assert report["status"] == "feasible_closed_cycle_found"
    assert report["certificate_valid"] is True
    assert report["next_cycle_viable"] is True
    assert report["global_infeasibility_proven"] is False


def test_failed_probe_is_never_labeled_as_a_physiological_limit():
    report = assess_frozen_state_one_cycle_probe(
        _source(), _failed_probe(), checkpoint_completed_windows=12, launcher_returncode=0,
    )
    assert report["status"] == "no_feasible_closed_cycle_found_by_local_probe"
    assert report["next_cycle_viable"] is None
    assert report["normalized_constraint_deficit"]["value"] == 20.0
    assert report["global_infeasibility_proven"] is False


def test_checkpoint_must_follow_exactly_the_last_certified_cycle():
    report = assess_frozen_state_one_cycle_probe(
        _source(), _failed_probe(), checkpoint_completed_windows=11,
    )
    assert report["status"] == "invalid_checkpoint_source_mismatch"
    assert report["certificate_valid"] is False


def test_deficit_requires_a_nonadvanced_attempt_with_a_tolerance():
    assert normalized_constraint_deficit({})["available"] is False
    assert normalized_constraint_deficit(_failed_probe())["kind"] == "normalized_constraint_residual"


def test_probe_arguments_replace_trajectory_and_output_controls():
    arguments = [
        "--solvers", "ipopt", "--n-windows", "1500", "--cycles-per-window", "1",
        "--output-json", "old.json", "--common-initial-solution", "old.npz",
        "--common-initial-solution-recenter-first-node-bounds",
        "--rho-replay-checkpoint-output", "old-checkpoint.npz",
        "--retry-failed-rho-without-advance",
    ]
    result = build_probe_benchmark_arguments(arguments, checkpoint=__import__("pathlib").Path("next.npz"), output_json=__import__("pathlib").Path("probe.json"))
    assert result.count("--n-windows") == 1
    assert result[result.index("--n-windows") + 1] == "1"
    assert "old.json" not in result and "old.npz" not in result
    assert "--common-initial-solution-feasibility-probe" in result
