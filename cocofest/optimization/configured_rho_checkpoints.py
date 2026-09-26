"""Durable, model-fingerprinted primal checkpoints for configured RHO runs.

These are recovery *inputs*, not a claim of bitwise/full campaign continuation:
the PACE worker, controller clock and optimizer multipliers are not serialized.
"""

from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .configured_cycling_model import annotate_generated_seed, FINGERPRINT_KEY


def atomic_json(path, value):
    """Publish complete JSON, keeping the previous version until replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", dir=path.parent, prefix="." + path.name,
                                suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def last_journal_record(path):
    """Snapshot latest complete PACE line without rereading a growing journal."""
    if path is None or not Path(path).is_file():
        return None
    with Path(path).open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 262144))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        return {key: record[key] for key in
                ("event", "cycle_index", "weights", "ocp_cost_connected") if key in record}
    return None


def checkpoint_cli(every, directory, n_windows):
    if (every is None) != (directory is None):
        raise ValueError("--checkpoint-every and --checkpoint-directory must be supplied together")
    if every is None:
        return []
    if every < 1 or every >= n_windows:
        raise ValueError("checkpoint-every must be positive and smaller than n-windows")
    # At the final boundary there is no next window to prepare. Final result is
    # the completion record; protect the intermediate boundaries here.
    windows = range(every, n_windows, every)
    return ["--retry-failed-rho-without-advance",
            "--rho-prepared-checkpoint-output-template",
            str(Path(directory).expanduser().resolve() / "cycle-{completed_windows}.npz"),
            "--rho-prepared-checkpoint-windows", ",".join(map(str, windows))]


@contextmanager
def configured_checkpoint_writer(directory, config, *, condition, arguments,
                                 weights_config=None, weights_journal=None, module=None):
    """Publish a seed+receipt pair only after the benchmark certifies transfer.

    A fresh directory is mandatory. An interrupted write can leave an orphan
    NPZ, but latest.json only points to a fully committed receipt+NPZ pair.
    """
    if directory is None:
        yield
        return
    from .trajectory_io import checkpoint_publication_hook
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    weights_journal = (None if weights_journal is None
                       else Path(weights_journal).expanduser().resolve())
    weights = None if weights_config is None else json.loads(Path(weights_config).read_text())
    manifest = {"schema_version": 1, "status": "running", "condition": condition,
                "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
                "model": config, "arguments": list(arguments), "weights_config": weights,
                "complete_campaign_resume_supported": False,
                "checkpoint_kind": "certified_shifted_primal_with_audit",
                "physical_failure_proven": False}
    atomic_json(directory / "manifest.json", manifest)
    # ``module`` remains supported for callers injecting the historical hook.
    # Normal execution uses the package policy and never imports examples.
    original = None if module is None else module._save_rho_replay_checkpoint

    def publish(write, path, nmpc, args, *, completed_windows):
        target = Path(path).expanduser().resolve()
        if target.parent != directory:
            # A separate explicit legacy checkpoint keeps its own semantics.
            return write(path, nmpc, args, completed_windows=completed_windows)
        receipt_path = target.with_suffix(".receipt.json")
        if target.exists() or receipt_path.exists():
            raise FileExistsError(f"Refusing to replace existing checkpoint: {target}")
        temporary = None
        try:
            with NamedTemporaryFile(dir=directory, prefix=".primal-", suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
            write(temporary, nmpc, args, completed_windows=completed_windows)
            annotate_generated_seed(temporary, config, condition=condition)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            digest = sha256(temporary.read_bytes()).hexdigest()
            receipt = {"schema_version": 1, "completed_windows": int(completed_windows),
                       "target_rho": int(completed_windows) + 1,
                       "path": str(target), "sha256": digest,
                       FINGERPRINT_KEY: config[FINGERPRINT_KEY], "condition": condition,
                       "saved_at": datetime.now(timezone.utc).isoformat(),
                       "checkpoint_kind": manifest["checkpoint_kind"],
                       "pace_journal_snapshot": last_journal_record(weights_journal),
                       "complete_campaign_resume_supported": False,
                       "unserialized_state": ["PACE worker and pending proposal", "PACE global clock",
                                              "solver multipliers", "full campaign metrics"],
                       "physiological_infeasibility_proven": False}
            os.replace(temporary, target)
            atomic_json(receipt_path, receipt)
            atomic_json(directory / "latest.json", {**receipt, "receipt": str(receipt_path)})
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    if module is not None:
        def save(path, nmpc, args, *, completed_windows):
            return publish(original, path, nmpc, args, completed_windows=completed_windows)
        module._save_rho_replay_checkpoint = save
    try:
        with checkpoint_publication_hook(publish) if module is None else nullcontext():
            yield
    except BaseException as error:
        manifest.update(status="exception", error=f"{type(error).__name__}: {error}")
        raise
    else:
        manifest["status"] = "launcher_completed"
        manifest["physical_outcome"] = "consult_benchmark_result"
    finally:
        if module is not None:
            module._save_rho_replay_checkpoint = original
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(directory / "manifest.json", manifest)
