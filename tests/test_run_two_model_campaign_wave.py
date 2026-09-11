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
