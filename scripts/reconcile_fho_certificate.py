#!/usr/bin/env python3
"""Record an already certified terminal-bridge FHO attempt in its report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cycles", type=int, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    report_path = root / "full-horizon-report.json"
    report = json.loads(report_path.read_text())
    attempts = report.setdefault("full_horizon_attempts", [])
    if any(int(item.get("cycles") or 0) == args.cycles and item.get("success") for item in attempts):
        return
    previous = max((item for item in attempts if item.get("success")), key=lambda item: int(item["cycles"]))
    if int(previous["cycles"]) != args.cycles - 1:
        raise ValueError("Certificate is not the immediate continuation of the recorded horizon.")
    case = root / f"full-horizon-{args.cycles:04d}" / "chance-1"
    result_path, solution_path, log_path = case / "result.json", case / "full-solution.npz", case / "solver.log"
    payload = json.loads(result_path.read_text())
    result = payload["results"][0]
    window = result["windows"][0]
    if not (result.get("success") and window.get("status") == 0 and window.get("validated") and solution_path.is_file()):
        raise ValueError("The candidate FHO is not certified.")
    bridge = root / "fho-terminal-bridges" / f"from-{args.cycles - 1:04d}-to-{args.cycles:04d}" / "chance-1"
    attempt = {
        "accepted_for_continuation": True, "adaptive_source_cycles": args.cycles - 1,
        "adaptive_step_cycles": 1, "bridge_boundary_maximum_absolute_change": 0.0,
        "bridge_seed_path": str(bridge / "fho-terminal-bridge-seed.npz"),
        "bridge_source_cycles": args.cycles - 1,
        "bridge_terminal_cycle_seed_path": str(bridge / "terminal-cycle-seed.npz"),
        "certificate_valid": True, "cycles": args.cycles,
        "elapsed_s": float(result.get("wall_time_s", 0.0)), "failure_kind": None,
        "infrastructure_error": False, "log_path": str(log_path), "mechanical_formulation": "reduced",
        "memory_limit_exceeded": False, "peak_rss_bytes": None, "peak_rss_gib": None,
        "phase": "fho_terminal_bridge", "physiological_limit_certified": False,
        "prefix_solution_path": previous["solution_path"], "result_path": str(result_path),
        "return_code": 0, "rho_extension_failure": "rho_extension_reference_cycle_unavailable",
        "seed_handoff_error": None, "seed_origin": "certified_fho_terminal_direct_bridge",
        "seed_source_path": str(bridge / "fho-terminal-bridge-seed.npz"),
        "solution_available": True, "solution_path": str(solution_path), "success": True,
        "timed_out": False, "unknown_mumps_warning": False,
    }
    attempts.append(attempt)
    report["largest_successful_cycles"] = args.cycles
    report["stop_reason"] = "interrupted_after_certified_attempt_reconciled"
    report.setdefault("reconciliation_events", []).append({"cycles": args.cycles, "result_path": str(result_path)})
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)


if __name__ == "__main__":
    main()
