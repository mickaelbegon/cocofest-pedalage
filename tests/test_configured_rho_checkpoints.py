import json
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.configured_cycling_model import FINGERPRINT_KEY, require_seed_fingerprint
from cocofest.optimization.configured_rho_checkpoints import (
    atomic_json, checkpoint_cli, configured_checkpoint_writer, last_journal_record,
)


CONFIG = {FINGERPRINT_KEY: "model-fixture", "muscles": {"Triceps": {"alpha_a": -.12}},
          "case_id": "fixture"}


def writer(path, nmpc, args, *, completed_windows):
    np.savez(path, state__A_Triceps=np.array([[10., 9.]]),
             metadata__json=np.asarray(json.dumps({"producer_completed_windows": completed_windows})))


def test_publish_complete_fingerprinted_pair_and_restore_hook(tmp_path):
    module = SimpleNamespace(_save_rho_replay_checkpoint=writer)
    directory = tmp_path / "checkpoints"
    journal = tmp_path / "weights.jsonl"
    journal.write_text('{"cycle_index":19,"weights":[1,1,1,1],"event":"held"}\n')
    with configured_checkpoint_writer(directory, CONFIG, condition="rho-pace", arguments=["--n-windows", "2000"],
                                      weights_journal=journal, module=module):
        module._save_rho_replay_checkpoint(directory / "cycle-20.npz", None, None, completed_windows=20)
        receipt = json.loads((directory / "latest.json").read_text())
        assert receipt["completed_windows"] == 20
        assert receipt["target_rho"] == 21
        assert receipt["sha256"] == sha256((directory / "cycle-20.npz").read_bytes()).hexdigest()
        assert receipt["pace_journal_snapshot"]["weights"] == [1, 1, 1, 1]
        assert receipt["complete_campaign_resume_supported"] is False
        require_seed_fingerprint(directory / "cycle-20.npz", CONFIG[FINGERPRINT_KEY])
    assert module._save_rho_replay_checkpoint is writer
    assert json.loads((directory / "manifest.json").read_text())["status"] == "launcher_completed"


def test_failed_write_preserves_previous_checkpoint_and_manifest_evidence(tmp_path):
    calls = 0

    def broken_writer(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("disk fixture failure")
        writer(*args, **kwargs)

    module = SimpleNamespace(_save_rho_replay_checkpoint=broken_writer)
    directory = tmp_path / "checkpoints"
    with pytest.raises(RuntimeError, match="disk fixture"):
        with configured_checkpoint_writer(directory, CONFIG, condition="rho", arguments=[], module=module):
            module._save_rho_replay_checkpoint(directory / "cycle-20.npz", None, None, completed_windows=20)
            module._save_rho_replay_checkpoint(directory / "cycle-40.npz", None, None, completed_windows=40)
    assert json.loads((directory / "latest.json").read_text())["completed_windows"] == 20
    assert not (directory / "cycle-40.npz").exists()
    assert not list(directory.glob(".primal-*.npz"))
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "exception"
    assert manifest["physical_failure_proven"] is False
    assert module._save_rho_replay_checkpoint is broken_writer


def test_existing_directory_and_checkpoint_are_never_overwritten(tmp_path):
    module = SimpleNamespace(_save_rho_replay_checkpoint=writer)
    directory = tmp_path / "checkpoints"
    with configured_checkpoint_writer(directory, CONFIG, condition="rho", arguments=[], module=module):
        path = directory / "cycle-20.npz"
        module._save_rho_replay_checkpoint(path, None, None, completed_windows=20)
        before = path.read_bytes()
        with pytest.raises(FileExistsError):
            module._save_rho_replay_checkpoint(path, None, None, completed_windows=20)
        assert before == path.read_bytes()
    with pytest.raises(FileExistsError):
        with configured_checkpoint_writer(directory, CONFIG, condition="rho", arguments=[], module=module):
            pass


def test_milestones_enable_certified_transfer_and_do_not_request_final_seed(tmp_path):
    from examples.fes_multibody.cycling.cycling_fes_solver_comparison import build_cli
    options = checkpoint_cli(20, tmp_path, 2000)
    args = build_cli().parse_args(options)
    assert args.retry_failed_rho_without_advance is True
    assert args.rho_prepared_checkpoint_windows[0] == 20
    assert args.rho_prepared_checkpoint_windows[-1] == 1980
    assert args.rho_prepared_checkpoint_output_template.endswith("cycle-{completed_windows}.npz")


@pytest.mark.parametrize("every,directory,cycles", [(0, ".", 20), (20, ".", 20), (1, None, 20),
                                                    (None, ".", 20)])
def test_bad_checkpoint_requests_are_refused(every, directory, cycles):
    with pytest.raises(ValueError):
        checkpoint_cli(every, directory, cycles)


def test_journal_snapshot_survives_partial_last_line(tmp_path):
    journal = tmp_path / "journal"
    journal.write_text('{"event":"held","cycle_index":20,"weights":[1,2]}\n{"interrupted":')
    assert last_journal_record(journal)["cycle_index"] == 20


def test_journal_path_is_preserved_when_benchmark_changes_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    journal = tmp_path / "weights.jsonl"
    journal.write_text('{"cycle_index":19,"weights":[1,1,1,1]}\n')
    module = SimpleNamespace(_save_rho_replay_checkpoint=writer)
    directory = tmp_path / "checkpoints"
    elsewhere = tmp_path / "example"
    elsewhere.mkdir()
    with configured_checkpoint_writer(directory, CONFIG, condition="rho-pace", arguments=[],
                                      weights_journal="weights.jsonl", module=module):
        monkeypatch.chdir(elsewhere)
        module._save_rho_replay_checkpoint(directory / "cycle-20.npz", None, None, completed_windows=20)
    assert json.loads((directory / "latest.json").read_text())["pace_journal_snapshot"]["cycle_index"] == 19


def test_invalid_json_does_not_replace_previous_audit(tmp_path):
    path = tmp_path / "audit.json"
    atomic_json(path, {"status": "running"})
    with pytest.raises(ValueError):
        atomic_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"status": "running"}
