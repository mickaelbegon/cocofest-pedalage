#!/usr/bin/env python3
"""Run a safe limited-memory -> exact IPOPT FHO continuation.

The base command is deliberately supplied as a JSON argv list.  It keeps the
scientific FHO transcription in the normal comparison CLI, while this small
driver owns only the numerical protocol and its provenance.  No active
campaign is discovered, modified, or stopped by this script.

Example (the JSON list contains all fixed FHO arguments, but no mutable IPOPT
or output arguments)::

  scripts/run_hybrid_fho_ipopt.py --base-command-json fho65-command.json \
      --initial-seed rho-prefix.npz --output-dir trial --horizon-cycles 65
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

from cocofest.optimization.hybrid_ipopt_protocol import (
    HybridStoppingPolicy,
    decide_lbfgs_handoff,
    extract_ipopt_block_observation,
)


MUTABLE_OPTIONS_WITH_VALUE = {
    "--ipopt-max-iter",
    "--ipopt-hessian-approximation",
    "--ipopt-limited-memory-max-history",
    "--common-initial-solution",
    "--common-initial-solution-output",
    "--output-json",
}
MUTABLE_FLAGS = {
    "--allow-primal-feasible-common-initial-solution-output",
    "--allow-finite-uncertified-common-initial-solution-output",
}


def _read_argv(path: Path) -> list[str]:
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError("--base-command-json must contain a non-empty JSON array of strings.")
    return list(value)


def _strip_mutable_options(argv: Iterable[str]) -> list[str]:
    """Reject mutable flags in the base command instead of silently shadowing."""

    result: list[str] = []
    items = list(argv)
    index = 0
    while index < len(items):
        item = items[index]
        if item in MUTABLE_OPTIONS_WITH_VALUE:
            raise ValueError(f"{item} belongs to the hybrid driver, not the base command JSON.")
        if item in MUTABLE_FLAGS:
            raise ValueError(f"{item} belongs to the hybrid driver, not the base command JSON.")
        result.append(item)
        index += 1
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    return int(completed.returncode)


def _phase_command(
    base: list[str], *, seed: Path, result: Path, output: Path,
    max_iterations: int, hessian: str, history: int | None,
    allow_uncertified_output: bool,
) -> list[str]:
    command = [
        *base,
        "--common-initial-solution", str(seed),
        "--common-initial-solution-output", str(output),
        "--output-json", str(result),
        "--ipopt-max-iter", str(max_iterations),
        "--ipopt-hessian-approximation", hessian,
    ]
    if history is not None:
        command.extend(["--ipopt-limited-memory-max-history", str(history)])
    if allow_uncertified_output:
        command.append("--allow-finite-uncertified-common-initial-solution-output")
    return command


def run_hybrid(args: argparse.Namespace) -> dict[str, Any]:
    base = _strip_mutable_options(_read_argv(args.base_command_json))
    policy = HybridStoppingPolicy(
        block_iterations=args.block_iterations,
        relative_objective_tolerance=args.relative_objective_tolerance,
        audited_constraint_tolerance=args.audited_constraint_tolerance,
    )
    output_dir = args.output_dir.resolve()
    blocks_dir = output_dir / "limited-memory-blocks"
    report_path = output_dir / "hybrid-report.json"
    report: dict[str, Any] = {
        "schema": "cocofest-fho-hybrid-ipopt-v1",
        "horizon_cycles": args.horizon_cycles,
        "initial_seed": str(args.initial_seed.resolve()),
        "limited_memory": {
            "role": "non_certifying_warm_start_only",
            "max_iterations": args.max_lbfgs_iterations,
            "history": args.limited_memory_history,
            "policy": {
                "block_iterations": policy.block_iterations,
                "relative_objective_tolerance": policy.relative_objective_tolerance,
                "audited_constraint_tolerance": policy.audited_constraint_tolerance,
            },
            "blocks": [],
        },
        "exact": {"role": "only_certifier", "started": False},
        "status": "running",
    }
    _write_json(report_path, report)
    seed = args.initial_seed.resolve()
    history = []
    cumulative = 0
    switch = None

    while cumulative < args.max_lbfgs_iterations:
        budget = min(policy.block_iterations, args.max_lbfgs_iterations - cumulative)
        # A shortened final block is allowed as a budget safeguard, but cannot
        # establish the requested 20--30 iteration objective criterion.
        block_number = len(history) + 1
        block_dir = blocks_dir / f"block-{block_number:03d}"
        checkpoint = block_dir / "warm-start-uncertified.npz"
        result_path = block_dir / "result.json"
        command = _phase_command(
            base, seed=seed, result=result_path, output=checkpoint,
            max_iterations=budget, hessian="limited-memory",
            history=args.limited_memory_history, allow_uncertified_output=True,
        )
        return_code = _run(command, block_dir / "solver.log")
        row: dict[str, Any] = {
            "block": block_number, "requested_iterations": budget,
            "return_code": return_code, "seed": str(seed),
            "result_path": str(result_path), "checkpoint_path": str(checkpoint),
        }
        if not result_path.exists() or not checkpoint.exists():
            row["accepted"] = False
            row["reason"] = "missing_finite_uncertified_checkpoint_or_result"
            report["limited_memory"]["blocks"].append(row)
            report["status"] = "limited_memory_checkpoint_unavailable"
            _write_json(report_path, report)
            return report
        payload = json.loads(result_path.read_text())
        try:
            solver_stats = payload["results"][0]["nlp_solver_stats"][0]
            actual = int(solver_stats["iter_count"])
            cumulative += actual
            observation = extract_ipopt_block_observation(
                payload, checkpoint_path=str(checkpoint), block=block_number,
                cumulative_iterations=cumulative,
            )
        except (KeyError, TypeError, ValueError) as error:
            row["accepted"] = False
            row["reason"] = f"structured_audit_unavailable:{type(error).__name__}:{error}"
            report["limited_memory"]["blocks"].append(row)
            report["status"] = "limited_memory_audit_unavailable"
            _write_json(report_path, report)
            return report
        row.update({"accepted": True, **observation.__dict__})
        history.append(observation)
        switch = decide_lbfgs_handoff(history, policy)
        row["handoff_decision"] = switch
        report["limited_memory"]["blocks"].append(row)
        _write_json(report_path, report)
        seed = checkpoint
        if switch["switch_to_exact"]:
            break
        if actual < budget:
            report["status"] = "limited_memory_terminated_before_budget"
            _write_json(report_path, report)
            return report

    # Reaching the 3*N cap is still a safe warm-start hand-off.  The report
    # distinguishes it from an early criterion-triggered transition.
    handoff_reason = switch["reason"] if switch else "limited_memory_budget_reached"
    exact_dir = output_dir / "exact-restoration"
    exact_result = exact_dir / "result.json"
    exact_solution = exact_dir / "full-solution.npz"
    command = _phase_command(
        base, seed=seed, result=exact_result, output=exact_solution,
        max_iterations=args.exact_max_iterations, hessian="exact", history=None,
        allow_uncertified_output=False,
    )
    report["exact"].update({
        "started": True, "handoff_reason": handoff_reason,
        "seed": str(seed), "result_path": str(exact_result),
        "solution_path": str(exact_solution),
    })
    _write_json(report_path, report)
    report["exact"]["return_code"] = _run(command, exact_dir / "solver.log")
    report["exact"]["solution_exported"] = exact_solution.exists()
    report["status"] = "exact_completed" if exact_result.exists() else "exact_result_unavailable"
    _write_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-command-json", type=Path, required=True)
    parser.add_argument("--initial-seed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon-cycles", type=int, required=True)
    parser.add_argument("--block-iterations", type=int, default=25)
    parser.add_argument("--max-lbfgs-iterations", type=int, default=None)
    parser.add_argument("--limited-memory-history", type=int, default=10)
    parser.add_argument("--exact-max-iterations", type=int, default=4000)
    parser.add_argument("--relative-objective-tolerance", type=float, default=1e-6)
    parser.add_argument("--audited-constraint-tolerance", type=float, default=1e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.horizon_cycles < 1:
        raise ValueError("--horizon-cycles must be positive.")
    if args.max_lbfgs_iterations is None:
        args.max_lbfgs_iterations = 3 * args.horizon_cycles
    if args.max_lbfgs_iterations < args.block_iterations:
        raise ValueError("--max-lbfgs-iterations must cover at least one block.")
    if args.limited_memory_history < 1:
        raise ValueError("--limited-memory-history must be positive.")
    report = run_hybrid(args)
    print(json.dumps({"status": report["status"], "report": str(args.output_dir / "hybrid-report.json")}, indent=2))
    if report["status"] != "exact_completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
