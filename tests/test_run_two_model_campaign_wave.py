import json
from pathlib import Path

import pytest

from scripts import run_two_model_campaign_wave as wave


def _job(identifier, directory):
    return {
        "id": identifier, "argv": ["fake", identifier], "cpu_ids": [0, 1],
        "environment": {"OMP_NUM_THREADS": "1"}, "working_directory": str(directory),
        "output_directory": str(directory / identifier),
    }


def test_dry_run_returns_sealed_commands_without_starting(monkeypatch, tmp_path):
    jobs = [_job("a/baseline_rho", tmp_path), _job("b/baseline_rho", tmp_path)]
    monkeypatch.setattr(wave, "next_jobs", lambda manifest, reviews: jobs)
    assert wave.launch_wave({"manifest_sha256": "sealed"}, {}, dry_run=True) == [
        {"id": job["id"], "argv": job["argv"], "dry_run": True} for job in jobs
    ]
    assert not (tmp_path / "a" / "baseline_rho").exists()


def test_launch_waits_for_one_wave_and_writes_unreviewed_receipts(monkeypatch, tmp_path):
    jobs = [_job("a/baseline_rho", tmp_path), _job("b/baseline_rho", tmp_path)]
    monkeypatch.setattr(wave, "next_jobs", lambda manifest, reviews: jobs)
    calls = []

    class Process:
        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs))
            self.returncode = 0

        def wait(self):
            return self.returncode

    monkeypatch.setattr(wave.subprocess, "Popen", Process)
    output = wave.launch_wave({"manifest_sha256": "sealed"}, {})
    assert [entry["id"] for entry in output] == [job["id"] for job in jobs]
    assert len(calls) == 2
    for job in jobs:
        receipt = Path(job["output_directory"]) / "launcher-receipt.json"
        record = json.loads(receipt.read_text())
        assert record["physical_outcome"] == "unreviewed_process_exit"
        assert record["manifest_sha256"] == "sealed"
        assert Path(record["output_log"]).is_file()


def test_existing_receipt_refuses_relaunch(monkeypatch, tmp_path):
    job = _job("a/baseline_rho", tmp_path)
    receipt = Path(job["output_directory"]) / "launcher-receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}")
    monkeypatch.setattr(wave, "next_jobs", lambda manifest, reviews: [job])
    with pytest.raises(FileExistsError):
        wave.launch_wave({"manifest_sha256": "sealed"}, {})


def _saved_campaign(monkeypatch, tmp_path):
    job = _job("a/baseline_rho", tmp_path)
    directory = Path(job["output_directory"])
    directory.mkdir(parents=True)
    job.update(model_id="a", arm_id="baseline_rho",
               result_path=str(directory / "result.json"),
               configuration_audit_path=str(directory / "audit.json"),
               weights_journal_path=None)
    manifest = {"manifest_sha256": "sealed", "arms": [job]}
    reviews = {"manifest_sha256": "sealed", "models": {}}
    monkeypatch.setattr(wave, "verify_manifest", lambda manifest: None)

    def blocked(manifest, reviews):
        raise ValueError("Existing unreviewed result requires attention: a/baseline_rho")

    monkeypatch.setattr(wave, "next_jobs", blocked)
    return job, manifest, reviews


def test_status_explains_completed_result_without_certifying_or_mutating(monkeypatch, tmp_path):
    job, manifest, reviews = _saved_campaign(monkeypatch, tmp_path)
    result = Path(job["result_path"])
    result.write_text(json.dumps({"results": [{"success": True, "validated_cycles": 100,
                                               "attempted_windows": 100, "error": None}]}))
    wave._receipt_path(job).write_text(json.dumps({"job_id": job["id"], "manifest_sha256": "sealed",
                                                  "returncode": 0}))
    original = result.read_bytes()
    status = wave.campaign_status(manifest, reviews)
    entry = status["jobs"][0]
    assert status["solvers_started"] == 0
    assert "Existing unreviewed" in status["scheduler_error"]
    assert entry["status"] == "existing_artifacts_require_review"
    assert entry["process_returncode"] == 0
    assert entry["reported_solver_results"][0]["validated_cycles"] == 100
    assert entry["review_record_present"] is False
    assert "Preserve these artifacts" in entry["next_step"]
    assert result.read_bytes() == original
    assert reviews["models"] == {}


def test_status_survives_partial_json_and_rejects_foreign_receipt(monkeypatch, tmp_path):
    job, manifest, reviews = _saved_campaign(monkeypatch, tmp_path)
    Path(job["result_path"]).write_text('{"results":')
    wave._receipt_path(job).write_text(json.dumps({"job_id": job["id"], "manifest_sha256": "other",
                                                  "returncode": 0}))
    entry = wave.campaign_status(manifest, reviews)["jobs"][0]
    assert len(entry["diagnostic_errors"]) == 2
    assert "process_returncode" not in entry
    assert "reported_solver_results" not in entry


def test_cli_blocked_campaign_has_actionable_error_without_traceback(monkeypatch, tmp_path, capsys):
    _, manifest, reviews = _saved_campaign(monkeypatch, tmp_path)
    manifest_file = tmp_path / "manifest.json"
    reviews_file = tmp_path / "reviews.json"
    manifest_file.write_text(json.dumps(manifest))
    reviews_file.write_text(json.dumps(reviews))
    with pytest.raises(SystemExit) as error:
        wave.main(["--manifest", str(manifest_file), "--reviews", str(reviews_file)])
    assert error.value.code == 2
    output = capsys.readouterr().err
    assert "--status" in output
    assert "reviewed before" in output
    assert "Traceback" not in output
