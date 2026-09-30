import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_pace_decision_fidelity.py"
SPEC = importlib.util.spec_from_file_location("validate_pace_decision_fidelity", MODULE_PATH)
pace_validation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pace_validation)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _source(tmp_path: Path) -> tuple[Path, Path, Path]:
    pace = tmp_path / "pace"
    model = tmp_path / "model.json"
    _write(model, {"schema_version": 1, "muscles": {name: {} for name in pace_validation.MUSCLE_NAMES}})
    _write(pace / "configuration-audit.json", {
        "model_config_path": str(model), "weights_journal_path": str(pace / "weights.jsonl"),
        "arguments": ["--solvers", "ipopt", "--n-windows", "1500", "--output-json", "old.json",
                      "--common-initial-solution", "old.npz", "--common-initial-solution-recenter-first-node-bounds",
                      "--adopt-common-initial-solution-warmup-cycles", "--rho-prepared-checkpoint-output-template", "old-{completed_windows}.npz",
                      "--rho-prepared-checkpoint-windows", "20,40", "--compact-rho-output"],
    })
    journal = [
        {"event": "boundary", "cycle_index": 60, "weights": [1.1, 1.2, 1.3, 1.4]},
        {"event": "projection_completed", "source_cycle_index": 60, "application_cycle_index": 63,
         "selected_candidate": {"weights": [1.2, 1.1, 1.0, .9]}, "selection_basis": "guarded",
         "selected_evidence": {"full_horizon_mean_squared_fatigue": .1, "terminal_minimum_capacity": .8}},
    ]
    (pace / "weights.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (pace / "weights.jsonl").write_text("\n".join(json.dumps(item) for item in journal) + "\n", encoding="utf-8")
    checkpoint = pace / "checkpoints" / "cycle-60.npz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"placeholder")
    _write(checkpoint.with_suffix(".receipt.json"), {"completed_windows": 60})
    bo = tmp_path / "bo.json"
    _write(bo, {"initial_weights": {"Delt_ant": .7, "Delt_post": 1.1, "Biceps": .3, "Triceps": 3.8}})
    return pace, bo, model


def test_prepare_uses_exact_checkpoint_and_static_candidates(tmp_path):
    pace, bo, _ = _source(tmp_path)
    manifest = pace_validation.prepare_validation(
        pace_directory=pace, bo_weights_config=bo, output_directory=tmp_path / "out",
        python=Path(__file__).resolve(), checkpoints=(60,), horizons=(1, 5),
    )
    document = json.loads(manifest.read_text())
    assert len(document["jobs"]) == 8  # current, unit, BO, proposal x horizons 1, 5
    proposal = next(job for job in document["jobs"] if job["candidate_id"] == "pace_proposal" and job["horizon_cycles"] == 5)
    assert proposal["checkpoint_path"].endswith("cycle-60.npz")
    assert proposal["weights"] == [1.2, 1.1, 1.0, .9]
    command = proposal["command"]
    assert command.count("--common-initial-solution") == 1
    assert command[command.index("--common-initial-solution") + 1] == proposal["checkpoint_path"]
    assert command[command.index("--n-windows") + 1] == "5"
    assert "--rho-prepared-checkpoint-output-template" not in command
    weights = json.loads(Path(proposal["weights_config"]).read_text())
    assert weights["policy"]["adaptation_enabled"] is False


def test_prepare_does_not_invent_missing_pace_proposal(tmp_path):
    pace, bo, _ = _source(tmp_path)
    (pace / "weights.jsonl").write_text(json.dumps({"event": "boundary", "cycle_index": 60, "weights": [1, 1, 1, 1]}) + "\n")
    manifest = pace_validation.prepare_validation(
        pace_directory=pace, bo_weights_config=bo, output_directory=tmp_path / "out",
        python=Path(__file__).resolve(), checkpoints=(60,), horizons=(1,),
    )
    assert {job["candidate_id"] for job in json.loads(manifest.read_text())["jobs"]} == {
        "pace_current", "unit", "bo_fixed"
    }


def test_extra_candidate_keeps_own_provenance_and_exact_restart(tmp_path):
    pace, bo, _ = _source(tmp_path)
    extra = tmp_path / "regularized.json"
    _write(extra, {"initial_weights": dict(zip(pace_validation.MUSCLE_NAMES, [1.2, 1., 1., .9]))})
    manifest = pace_validation.prepare_validation(
        pace_directory=pace, bo_weights_config=bo, output_directory=tmp_path / "out",
        python=Path(__file__).resolve(), checkpoints=(60,), horizons=(5,),
        candidate_configs={"epsilon_1em05": extra},
    )
    document = json.loads(manifest.read_text())
    candidate = next(job for job in document["jobs"] if job["candidate_id"] == "epsilon_1em05")
    assert candidate["weights"] == [1.2, 1., 1., .9]
    assert candidate["checkpoint_path"].endswith("cycle-60.npz")
    config = json.loads(Path(candidate["weights_config"]).read_text())
    assert config["policy"]["adaptation_enabled"] is False
    assert "regularized.json" in config["calibration"]["decision_fidelity"]["source"]
    assert document["extra_candidate_configs"] == {"epsilon_1em05": str(extra)}


@pytest.mark.parametrize("candidate_id", ["bo_fixed", "../escape", ".", "a/b", ""])
def test_extra_candidate_rejects_reserved_or_path_ids(tmp_path, candidate_id):
    pace, bo, _ = _source(tmp_path)
    with pytest.raises(ValueError, match="candidate id"):
        pace_validation.prepare_validation(
            pace_directory=pace, bo_weights_config=bo, output_directory=tmp_path / "out",
            python=Path(__file__).resolve(), checkpoints=(60,), horizons=(1,),
            candidate_configs={candidate_id: bo},
        )


def test_summary_ranks_lowest_complete_fatigue_and_reports_pace_regret(tmp_path):
    pace, bo, _ = _source(tmp_path)
    manifest = pace_validation.prepare_validation(
        pace_directory=pace, bo_weights_config=bo, output_directory=tmp_path / "out",
        python=Path(__file__).resolve(), checkpoints=(60,), horizons=(1,),
    )
    jobs = json.loads(manifest.read_text())["jobs"]
    fatigue = {"pace_current": 3.0, "unit": 2.0, "bo_fixed": 1.0, "pace_proposal": 1.5}
    for job in jobs:
        output = Path(job["result_path"])
        _write(output, {"results": [{
            "solver_success": True, "physical_success": True, "validated_cycles": 1,
            "fatigue_auc_cycles": fatigue[job["candidate_id"]], "min_A_capacity_ratio": .8,
            "control_saturation": [{"terminal_upper_fraction": .25}],
            "physical_crank_diagnostics": {"absolute_cycle_tolerance": .01, "absolute_cycle_errors": [.001]},
        }]})
    summary_path = pace_validation.summarize(manifest, plot=False)
    summary = json.loads(summary_path.read_text())
    comparison = summary["comparisons"][0]
    assert comparison["best_candidate_id"] == "bo_fixed"
    assert comparison["pace_selected_actual_rank"] == 4
    assert comparison["pace_selection_regret_fatigue_auc_cycles"] == 2.0
    unit = next(entry for entry in summary["entries"] if entry["candidate_id"] == "unit")
    assert unit["signed_fatigue_auc_gain_vs_unit"] == 0.0
