#!/usr/bin/env python3
"""Run one reviewed-ready wave of at most two sealed campaign jobs.

The command deliberately stops after one wave. A reviewer must add the
physical-certification receipts for completed jobs before a successor can be
started. It is therefore a small process executor, not an automatic claim of
fatigue failure or endurance.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_two_model_resistance_campaign import next_jobs


def _load_json(path: Path, *, label: str):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _receipt_path(job):
    return Path(job["output_directory"]).resolve() / "launcher-receipt.json"


def _log_path(job):
    return Path(job["output_directory"]).resolve() / "launcher-output.log"


def launch_wave(manifest, reviews, *, dry_run=False):
    """Launch exactly the eligible independent jobs and wait for their exits.

    ``next_jobs`` performs all manifest, review, dependency and two-slot
    checks. This function never invokes a successor after an exit: that
    requires an explicit review in a later invocation.
    """
    jobs = next_jobs(manifest, reviews)
    if not jobs:
        return []
    if dry_run:
        return [{"id": job["id"], "argv": job["argv"], "dry_run": True} for job in jobs]
    started = []
    try:
        for job in jobs:
            receipt = _receipt_path(job)
            log = _log_path(job)
            if receipt.exists() or log.exists():
                raise FileExistsError(f"Refusing to overwrite prior launcher artifacts: {receipt.parent}")
            receipt.parent.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update(job["environment"])
            stream = log.open("x", encoding="utf-8")
            process = subprocess.Popen(
                job["argv"], cwd=job["working_directory"], env=environment,
                stdout=stream, stderr=subprocess.STDOUT, text=True,
            )
            started.append((job, process, receipt, log, stream))
    except BaseException:
        for _, process, _, _, stream in started:
            process.terminate()
            process.wait()
            stream.close()
        if "stream" in locals() and not any(stream is item[4] for item in started):
            stream.close()
        raise
    records = []
    for job, process, receipt, log, stream in started:
        process.wait()
        stream.close()
        record = {
            "schema_version": 1,
            "job_id": job["id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "argv": job["argv"], "cpu_ids": job["cpu_ids"],
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": process.returncode,
            "output_log": str(log),
            "physical_outcome": "unreviewed_process_exit",
            "next_step": "review result, configuration audit and journal before a successor",
        }
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        records.append({"id": job["id"], "returncode": process.returncode, "receipt": str(receipt)})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True,
                        help="Review JSON bound to the exact sealed manifest")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the one or two eligible exact argv lists without launching")
    args = parser.parse_args(argv)
    manifest = _load_json(args.manifest.resolve(), label="manifest")
    reviews = _load_json(args.reviews.resolve(), label="reviews")
    records = launch_wave(manifest, reviews, dry_run=args.dry_run)
    print(json.dumps(records, indent=2, allow_nan=False))
    if not args.dry_run and any(item["returncode"] != 0 for item in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
