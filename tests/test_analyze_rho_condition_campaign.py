import json
from pathlib import Path

from scripts.analyze_rho_condition_campaign import analyze, collect


def _result(windows):
    return {"schema_version": 1, "results": [{"solver": "ipopt", "windows": windows}]}


def test_analysis_keeps_partial_and_invalid_arms_visible(tmp_path):
    root = tmp_path / "campaign"
    rho = root / "m0" / "rho" / "result.json"
    pace = root / "m0" / "rho-pace" / "result.json"
    journal = root / "m0" / "rho-pace" / "weights.jsonl"
    rho.parent.mkdir(parents=True)
    pace.parent.mkdir(parents=True)
    rho.write_text(json.dumps(_result([
        {"rho": 1, "solver_time_s": .4, "wall_time_s": .5, "effective_wall_time_s": .6,
         "solver_converged": True, "validated": True, "primal_feasible": True},
        {"rho": 2, "solver_time_s": None, "wall_time_s": 1., "effective_wall_time_s": 1.,
         "solver_converged": False, "validated": False, "primal_feasible": False},
    ])))
    pace.write_text(json.dumps(_result([
        {"rho": 1, "solver_time_s": .7, "wall_time_s": .8, "effective_wall_time_s": .9,
         "solver_converged": True, "validated": True, "primal_feasible": True},
    ])))
    journal.write_text("\n".join(json.dumps(row) for row in [
        {"event": "configuration", "muscle_names": ["Biceps", "Triceps"]},
        {"event": "boundary", "cycle_index": 0, "status": "applied", "weights": [1., 1.]},
        {"event": "boundary", "cycle_index": 2, "status": "applied", "weights": [1.2, .8]},
    ]) + "\n")
    manifest = {"campaign_id": "fixture", "resistance_nm": .22, "cycles": 3000, "arms": [
        {"id": "m0/rho", "model_id": "m0", "condition": "rho", "result_path": str(rho)},
        {"id": "m0/physio", "model_id": "m0", "condition": "rho-physio", "result_path": str(root / "missing.json"), "weights_journal_path": str(root / "missing.jsonl")},
        {"id": "m0/pace", "model_id": "m0", "condition": "rho-pace", "result_path": str(pace), "weights_journal_path": str(journal)},
    ]}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    report = analyze(manifest_path, root / "analysis", rolling_window=3)

    assert report["summary"]["observed_windows"] == 3
    assert report["summary"]["validated_windows"] == 2
    assert report["summary"]["arms_pending_or_unusable"] == 1
    assert report["summary"]["weight_observations"] == 4
    assert (root / "analysis" / "rho-convergence-timing.png").is_file()
    assert (root / "analysis" / "rho-weight-evolution.png").is_file()
    persisted = json.loads((root / "analysis" / "rho-condition-analysis.json").read_text())
    assert any("result pending" in diagnostic for diagnostic in persisted["diagnostics"])


def test_collect_accepts_legacy_baseline_rho_name(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(json.dumps(_result([])))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"arms": [{"id": "m/baseline_rho", "model_id": "m", "arm_id": "baseline_rho", "result_path": str(result)}]}))
    report = collect(manifest)
    assert report["arms"][0]["condition"] == "rho"
    assert report["arms"][0]["state"] == "unusable_result"
