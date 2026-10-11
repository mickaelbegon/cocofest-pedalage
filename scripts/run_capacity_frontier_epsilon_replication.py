#!/usr/bin/env python3
"""Execute a sealed local ε-cut replication plan on explicitly idle CPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_plan(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text())
    if value.get("kind") != "capacity_frontier_checkpoint_epsilon_replication_plan":
        raise ValueError("Unsupported epsilon-replication plan")
    jobs = value.get("jobs")
    if not isinstance(jobs, list) or not jobs or len({row.get("id") for row in jobs}) != len(jobs):
        raise ValueError("Plan needs uniquely identified jobs")
    for row in jobs:
        receipt = Path(row["receipt"]).resolve(strict=True)
        if sha256(receipt.read_bytes()).hexdigest() != row.get("receipt_sha256"):
            raise ValueError(f"Receipt changed for {row.get('id')}")
        if (row.get("side") != "left" or row.get("support") != [0., 0., 0., 1.]
                or len(row.get("epsilon_upper", ())) != 4 or len(row.get("capacity_domain", ())) != 2):
            raise ValueError(f"Malformed epsilon job {row.get('id')}")
    return value


def command_for(job: dict, *, python: Path, cpu: int) -> list[str]:
    lower, upper = job["capacity_domain"]
    epsilon = ["nan" if value is None else f"{float(value):.17g}" for value in job["epsilon_upper"]]
    return [str(python), str(ROOT / "scripts/verify_capacity_frontier_adapter.py"),
            "--receipt", str(Path(job["receipt"]).resolve()), "--side", job["side"], "--cpu", str(cpu),
            "--lower-ratio", f"{float(lower):.17g}", "--upper-ratio", f"{float(upper):.17g}",
            "--solve-support", "--validate-independent", "--audit-baseline-dop853",
            "--max-iterations", str(job["max_iterations"]), "--support", *(str(value) for value in job["support"]),
            "--epsilon-upper", *epsilon, "--output", str(Path(job["output"]).resolve())]


def execute(plan: dict, *, python: Path, cpu_ids: tuple[int, ...], output_directory: Path) -> list[dict]:
    if not cpu_ids or len(set(cpu_ids)) != len(cpu_ids) or any(cpu < 0 for cpu in cpu_ids):
        raise ValueError("A nonempty unique CPU list is required")
    python = Path(python).resolve(strict=True)
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    jobs = list(plan["jobs"])
    if any(Path(row["output"]).exists() for row in jobs):
        raise FileExistsError("A planned result already exists; refuse a mixed rerun")

    def one(index_job):
        index, job = index_job
        cpu = cpu_ids[index % len(cpu_ids)]
        log = output_directory / f"{job['id']}.log"
        command = command_for(job, python=python, cpu=cpu)
        with log.open("x", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       text=True, check=False)
        output = Path(job["output"])
        return {"id": job["id"], "cpu": cpu, "returncode": completed.returncode,
                "log": str(log), "output": str(output), "output_exists": output.is_file()}

    records = []
    with ThreadPoolExecutor(max_workers=len(cpu_ids)) as pool:
        futures = [pool.submit(one, item) for item in enumerate(jobs)]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    records.sort(key=lambda row: row["id"])
    receipt = {"schema_version": 1, "kind": "capacity_frontier_epsilon_replication_execution",
               "plan_sha256": sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
               "python": str(python), "cpu_ids": list(cpu_ids), "records": records,
               "success": all(row["returncode"] == 0 and row["output_exists"] for row in records)}
    with (output_directory / "execution.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cpu-ids", required=True, help="Comma-separated dedicated CPU identifiers")
    parser.add_argument("--execution-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    cpus = tuple(int(item) for item in args.cpu_ids.split(",") if item)
    records = execute(load_plan(args.plan), python=args.python, cpu_ids=cpus,
                      output_directory=args.execution_directory)
    if not all(row["returncode"] == 0 and row["output_exists"] for row in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
