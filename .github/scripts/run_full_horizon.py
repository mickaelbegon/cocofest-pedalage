#!/usr/bin/env python
"""Run the full-horizon continuation from an IDE and print its report.

``run_full_horizon_benchmark.py`` grows a monolithic full-horizon problem (FHO)
from two reduced RHO cycles, re-solving from the last certified solution. It
already writes ``full-horizon-report.md``; this driver exists so the run can be
started from a plain PyCharm run configuration, with the environment handled for
you, and so the report is echoed to the IDE console when the run ends.

The environment policy is shared with ``run_benchmarks.py``. The important
difference is which Conda environment is used: the monolithic FHO problems are
solved by MadNLP unless ``--solver ipopt`` is given, so the default run needs
``cocofest-madnlp32`` and its CasADi ABI 1, exactly as the ``full_horizon`` job
of the workflow does. This driver selects the right environment either way, so
the IDE interpreter does not have to match.

The continuation is expected to stop early: the current limitation is numerical
rather than memory-bound, and a larger machine does not fix it without a better
multi-cycle seed. A run that halts below ``--max-cycles`` is a result, not an
infrastructure failure -- read the report and the per-FHO JSON.

Examples
--------
::

    python .github/scripts/run_full_horizon.py --max-cycles 6
    python .github/scripts/run_full_horizon.py --solver ipopt --max-cycles 4
    python .github/scripts/run_full_horizon.py --resume
    python .github/scripts/run_full_horizon.py --resume-from /path/to/results --max-cycles 100
    python .github/scripts/run_full_horizon.py --report-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmarks import REPO_ROOT, base_environment, conda_env_prefix  # noqa: E402

BENCHMARK = REPO_ROOT / ".github" / "scripts" / "run_full_horizon_benchmark.py"
REPORT_NAME = "full-horizon-report.md"
REPORT_JSON_NAME = "full-horizon-report.json"
SEED_FILES = ("common-reduced.npz", "common-full.npz", "reduced-cycling-fourier12.npz")

# The continuation is long and mostly silent; echo the ladder milestones and any
# failure signature, and keep the rest in the log file.
INTERESTING = re.compile(
    r"FHO|full[- ]horizon|cycles?=|certified|fallback|jump|rss|RSS|peak"
    r"|Traceback|Error|FAILED|Killed|OOM|MemoryError|timeout"
)
NOISE = re.compile(r"CUDA\.jl|CUDA runtime|CUDA_Runtime|cuda\.juliagpu\.org")


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        help=(
            "largest horizon explored; defaults to 6 for a new run and to the "
            "stored target when resuming"
        ),
    )
    parser.add_argument(
        "--solver",
        choices=("madnlp", "ipopt"),
        help="FHO solver; inferred from the stored report when resuming",
    )
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--numeric-threads", type=int, default=1)
    parser.add_argument("--memory-limit-gib", default="auto",
                        help="process-tree peak RSS cap, or 'auto'")
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument(
        "--continuation-step-cycles",
        type=int,
        help="RHO cycles appended before the next FHO (new-run default: 3)",
    )
    parser.add_argument("--jump-objective-relative-tolerance", type=float)
    parser.add_argument("--assistance", default="0.00")
    parser.add_argument("--q-slack", default="0.002")
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--output-dir")
    location.add_argument(
        "--resume-from",
        type=Path,
        metavar="DIR",
        help=(
            "resume from a campaign directory (or one of its FHO checkpoint "
            "subdirectories); implies --resume"
        ),
    )
    parser.add_argument("--attempt-timeout-s", type=float, default=None,
                        help="wall-time cap per solver attempt")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the largest certified FHO in --output-dir")
    parser.add_argument("--report-only", action="store_true",
                        help="print the existing report without solving")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def find_resume_directory(raw_path: Path) -> Path:
    """Resolve a report directory from a campaign, report, or checkpoint path."""

    path = raw_path.expanduser().resolve()
    if path.is_file():
        if path.name != REPORT_JSON_NAME:
            raise ValueError(
                f"--resume-from expects a directory or {REPORT_JSON_NAME}, got {path}."
            )
        return path.parent
    if not path.is_dir():
        raise FileNotFoundError(f"Resume directory does not exist: {path}")

    for candidate in (path, *path.parents):
        if (candidate / REPORT_JSON_NAME).is_file():
            return candidate

    reports = sorted(path.rglob(REPORT_JSON_NAME))
    if len(reports) == 1:
        return reports[0].parent
    if not reports:
        raise FileNotFoundError(
            f"No {REPORT_JSON_NAME} was found in or above {path}."
        )
    raise ValueError(
        f"Several full-horizon reports were found below {path}; pass the exact "
        "campaign directory."
    )


def apply_run_defaults(args: argparse.Namespace, output_dir: Path) -> dict | None:
    """Apply new-run defaults or preserve the scientific settings of a resume."""

    report = None
    if args.resume:
        report_path = output_dir / REPORT_JSON_NAME
        if not report_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {report_path}.")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if args.max_cycles is None:
            args.max_cycles = int(report["max_cycles"])
        if args.solver is None:
            args.solver = str(report["full_horizon_solver"])
        if args.continuation_step_cycles is None:
            args.continuation_step_cycles = int(
                report.get("continuation_step_cycles") or 1
            )
        if args.jump_objective_relative_tolerance is None:
            stored_tolerance = report.get("jump_objective_relative_tolerance")
            args.jump_objective_relative_tolerance = float(
                0.005 if stored_tolerance is None else stored_tolerance
            )
    else:
        args.max_cycles = 6 if args.max_cycles is None else args.max_cycles
        args.solver = "madnlp" if args.solver is None else args.solver
        args.continuation_step_cycles = (
            3
            if args.continuation_step_cycles is None
            else args.continuation_step_cycles
        )
        args.jump_objective_relative_tolerance = (
            0.005
            if args.jump_objective_relative_tolerance is None
            else args.jump_objective_relative_tolerance
        )
    return report


def show_report(output_dir: Path) -> bool:
    report = output_dir / REPORT_NAME
    if not report.exists():
        print(f"\nNo {REPORT_NAME} in {output_dir}.")
        checkpoints = sorted(output_dir.glob("**/result.json")) if output_dir.exists() else []
        if checkpoints:
            print(f"{len(checkpoints)} per-FHO JSON file(s) are present:")
            for path in checkpoints[:10]:
                print(f"  {_display_path(path)}")
        return False

    print(f"\n{'=' * 100}")
    print(f"  FULL-HORIZON REPORT  --  {_display_path(report)}")
    print(f"{'=' * 100}")
    print(report.read_text(encoding="utf-8").rstrip())
    print(f"{'=' * 100}\n")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    if args.resume_from is not None:
        output_dir = find_resume_directory(args.resume_from)
        args.resume = True
    else:
        raw_output_dir = Path(args.output_dir or "full-horizon-results").expanduser()
        output_dir = (
            raw_output_dir.resolve()
            if raw_output_dir.is_absolute()
            else (REPO_ROOT / raw_output_dir).resolve()
        )
    resume_report = apply_run_defaults(args, output_dir)

    if args.report_only:
        return 0 if show_report(output_dir) else 1

    if not BENCHMARK.exists():
        print(f"Missing {BENCHMARK}", file=sys.stderr)
        return 1

    seed_dir = REPO_ROOT / "benchmark-seed"
    missing = [name for name in SEED_FILES if not (seed_dir / name).exists()]
    if missing:
        print(f"Missing seed file(s): {', '.join(missing)}", file=sys.stderr)
        print("Build them first: python .github/scripts/run_benchmarks.py --seed-only",
              file=sys.stderr)
        return 1

    suite = "madnlp32" if args.solver == "madnlp" else "rho32"
    prefix = conda_env_prefix(suite)
    if prefix is None:
        print(f"Environment cocofest-{suite} not found; it is required by "
              f"--solver {args.solver}.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    interpreter = str(prefix / "bin" / "python")
    command = [
        interpreter,
        str(BENCHMARK),
        "--workspace", str(REPO_ROOT),
        "--seed-dir", str(seed_dir),
        "--output-dir", str(output_dir),
        "--max-cycles", str(args.max_cycles),
        "--memory-limit-gib", str(args.memory_limit_gib),
        "--n-threads", str(args.threads),
        "--max-iterations", str(args.max_iterations),
        "--continuation-step-cycles", str(args.continuation_step_cycles),
        "--jump-objective-relative-tolerance", str(args.jump_objective_relative_tolerance),
        "--full-horizon-solver", args.solver,
        "--crank-assistance", args.assistance,
        "--terminal-wheel-q-slack", args.q_slack,
        # Child solver processes must use the same environment as this driver.
        "--python", interpreter,
    ]
    if args.attempt_timeout_s is not None:
        command += ["--attempt-timeout-s", str(args.attempt_timeout_s)]
    if args.resume:
        command.append("--resume")

    environment = base_environment(prefix, suite, args.threads, args.numeric_threads)
    log_path = output_dir / "driver.log"

    print(f"Repository  : {REPO_ROOT}")
    print(f"Environment : {prefix.name}  (solver: {args.solver})")
    if resume_report is not None:
        print(
            "Resume      : "
            f"{output_dir} from FHO_"
            f"{int(resume_report.get('largest_successful_cycles') or 0)}"
        )
    print(f"Horizon     : up to {args.max_cycles} cycles, step "
          f"{args.continuation_step_cycles}, RSS cap {args.memory_limit_gib}")
    print(f"Threads     : {args.threads} (numerical libraries: {args.numeric_threads})")
    print(f"Full log    : {log_path}")
    print(f"{'=' * 78}", flush=True)

    started = time.monotonic()
    log_mode = "a" if args.resume else "w"
    with log_path.open(log_mode, encoding="utf-8") as log_file:
        if args.resume:
            log_file.write(
                f"\n{'=' * 78}\nRESUME to FHO_{args.max_cycles} at "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 78}\n"
            )
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            if args.verbose or (INTERESTING.search(line) and not NOISE.search(line)):
                print(f"  {line.rstrip()}", flush=True)
        returncode = process.wait()

    print(f"\nContinuation finished with exit {returncode} in "
          f"{time.monotonic() - started:.1f} s.", flush=True)

    has_report = show_report(output_dir)
    if returncode != 0:
        print(f"Last lines of {log_path}:")
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]:
            print(f"  | {line}")
        # A continuation that stops early still produced a report; that is a
        # scientific outcome, not a driver failure.
        return 0 if has_report else returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
