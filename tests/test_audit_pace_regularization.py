from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from cocofest.optimization.endurance_weight_supervisor import (
    CandidateEvaluation, CandidateRolloutResult, EnduranceWeightSupervisor,
    WeightCandidate, WeightSupervisorConfig,
)
from scripts.audit_pace_regularization import audit_journal, guard_rejections, prepare_pilot


BASE = CandidateRolloutResult(
    50., True, False, -.1, "complete_with_deficit", full_horizon_normalized_deficit=.01,
    terminal_normalized_reserve=.3, full_horizon_mean_squared_fatigue=.1,
    first_block_mean_squared_fatigue=.08, terminal_minimum_capacity=.4)


@pytest.mark.parametrize("changes", [
    {}, {"full_horizon_mean_squared_fatigue": .09},
    {"full_horizon_mean_squared_fatigue": .09, "terminal_minimum_capacity": .399},
    {"full_horizon_normalized_deficit": .001, "first_block_mean_squared_fatigue": .081},
    {"full_horizon_normalized_deficit": .00999999},
    {"full_horizon_normalized_deficit": .009},
])
def test_explanation_matches_production_guard(changes):
    config = WeightSupervisorConfig(selection_mode="guarded_fatigue")
    incumbent = CandidateEvaluation(WeightCandidate(0, "incumbent", None, 0., (1.,) * 4), BASE, 0.)
    result = replace(BASE, **changes)
    candidate = CandidateEvaluation(WeightCandidate(1, "increase", "biceps", .1, (1.,) * 4), result, 0.)
    accepted = EnduranceWeightSupervisor(config)._passes_fatigue_guard(candidate, incumbent)
    assert (not guard_rejections(asdict(result), asdict(BASE), asdict(config))) is accepted


def test_plateau_is_distinguished_from_noninferiority_rejection():
    config = WeightSupervisorConfig(selection_mode="guarded_fatigue")
    evaluations = [asdict(CandidateEvaluation(WeightCandidate(index, kind, None, 0., (1.,) * 4), BASE, 0.))
                   for index, kind in enumerate(("incumbent", "increase", "decrease"))]
    report = audit_journal([{"event": "projection_completed", "source_cycle_index": 300,
                            "candidate_evaluations": evaluations, "selection_config": asdict(config)}])
    assert report["rejection_counts"] == {"no_material_gain": 2}
    assert set(report["projections"][0]["metric_spreads"].values()) == {0.}


def test_prepared_pilot_preserves_seed_and_does_not_overwrite_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {"initial_weight_basis": "unit", "initial_weights": {"biceps": 1.},
              "policy": {"max_cycles": 1500, "projection_fatigue_reference_regularization": 1e-12,
                         "projection_horizon_cycles": 50}}
    weights = source / "weights.json"
    weights.write_text(json.dumps(config))
    (source / "configuration-audit.json").write_text(json.dumps({
        "weights_config_path": str(weights), "model_config_path": "/model.json",
        "arguments": ["--n-windows", "1500", "--output-json", "/original/result.json",
                      "--common-initial-solution", "/original/seed.npz",
                      "--common-initial-solution-recenter-first-node-bounds", "--adopt-common-initial-solution-warmup-cycles",
                      "--rho-prepared-checkpoint-output-template", "/original/cycle-{completed_windows}.npz",
                      "--rho-prepared-checkpoint-windows", "20,40,60"]}))
    output = tmp_path / "pilot"
    path = prepare_pilot(pace_directory=source, output_directory=output, python=Path("/python"), epsilon=1e-5)
    prepared = json.loads(path.read_text())
    assert prepared["launched"] is False
    command = prepared["command"]
    assert command.count("--common-initial-solution") == 1
    assert command[command.index("--common-initial-solution") + 1] == "/original/seed.npz"
    assert command.count("--common-initial-solution-recenter-first-node-bounds") == 1
    assert "/original/result.json" not in command
    assert json.loads(weights.read_text()) == config
    assert json.loads((output / "weights.json").read_text())["policy"]["projection_fatigue_reference_regularization"] == 1e-5
