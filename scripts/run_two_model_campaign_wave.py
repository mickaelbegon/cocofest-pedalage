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

from scripts.build_two_model_resistance_campaign import next_jobs, verify_manifest


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


def campaign_status(manifest, reviews):
    """Describe saved artifacts without treating a solver result as a review.

    A receipt is evidence of a process exit only. The scheduler remains the
    authority on readiness and verifies all scientific-review bindings.
    """
    verify_manifest(manifest)
    if reviews.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Reviews must identify this exact campaign manifest")
    scheduler_error = None
    try:
        ready = {job["id"] for job in next_jobs(manifest, reviews)}
    except (OSError, ValueError) as error:
        ready = set()
        scheduler_error = str(error)
    records = []
    for job in manifest["arms"]:
        paths = {"result": Path(job["result_path"]),
                 "configuration_audit": Path(job["configuration_audit_path"]),
                 "launcher_receipt": _receipt_path(job), "launcher_log": _log_path(job)}
        if job.get("weights_journal_path"):
            paths["weights_journal"] = Path(job["weights_journal_path"])
        record = {"id": job["id"], "status": "waiting_for_predecessor_or_initial_review",
                  "artifacts": {name: {"path": str(path), "exists": path.exists()}
                                for name, path in paths.items()},
                  "review_record_present": job["arm_id"] in reviews.get("models", {}).get(job["model_id"], {}),
                  "diagnostic_errors": []}
        if record["review_record_present"]:
            record["status"] = "review_record_present"
        elif any(path.exists() for path in paths.values()):
            record["status"] = "existing_artifacts_require_review"
            record["next_step"] = (
                "Inspect result, configuration audit, launcher log and any weights journal; "
                "validate the required scientific gates and add a bound review to reviews.json. "
                "Preserve these artifacts; do not rerun this output directory."
            )
        elif job["id"] in ready:
            record["status"] = "ready"
        if paths["launcher_receipt"].is_file():
            try:
                receipt = _load_json(paths["launcher_receipt"], label="launcher receipt")
                if (receipt.get("job_id") != job["id"]
                        or receipt.get("manifest_sha256") != manifest["manifest_sha256"]):
                    raise ValueError("Launcher receipt does not identify this exact job and manifest")
                record["process_returncode"] = receipt.get("returncode")
                record["finished_at_utc"] = receipt.get("finished_at_utc")
            except ValueError as error:
                record["diagnostic_errors"].append(str(error))
        if paths["result"].is_file():
            try:
                result = _load_json(paths["result"], label="result")
                rows = result.get("results")
                if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                    raise ValueError("Result must contain a list of solver-result objects")
                # These are reported values, not independently certified outcomes.
                keys = ("solver", "success", "validated_cycles", "attempted_windows", "error")
                record["reported_solver_results"] = [
                    {key: row[key] for key in keys if key in row} for row in rows]
            except ValueError as error:
                record["diagnostic_errors"].append(str(error))
        records.append(record)
    return {"manifest_sha256": manifest["manifest_sha256"], "scheduler_error": scheduler_error,
            "jobs": records, "solvers_started": 0}


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
    parser.add_argument("--status", action="store_true",
                        help="Inspect saved results, exit receipts and review blockers without launching")
    args = parser.parse_args(argv)
    try:
        manifest = _load_json(args.manifest.resolve(), label="manifest")
        reviews = _load_json(args.reviews.resolve(), label="reviews")
        if args.status:
            print(json.dumps(campaign_status(manifest, reviews), indent=2, allow_nan=False))
            return
        records = launch_wave(manifest, reviews, dry_run=args.dry_run)
    except (OSError, ValueError) as error:
        parser.error(f"{error}. Inspect this campaign with the same command plus --status. "
                     "Existing results must be reviewed before their successor can start.")
    print(json.dumps(records, indent=2, allow_nan=False))
    if not args.dry_run and any(item["returncode"] != 0 for item in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
