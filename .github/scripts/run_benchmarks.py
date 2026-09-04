#!/usr/bin/env python
"""Run the cycling solver benchmark from an IDE and print a report at the end.

This is a thin driver around ``run_cycling_benchmark_case.sh``, which stays the
single source of truth for the protocol of sections 11.1 and 11.2 of
``docs/cycling_solver_benchmark/linux_32core_setup.md``. The driver exists so the
benchmark can be started from a plain PyCharm run configuration, with no shell
setup and no environment variables to fill in by hand.

It takes care of the three things an IDE gets wrong compared to a shell that ran
``benchmark_env.sh``:

* ``CONDA_PREFIX`` is unset when the IDE launches the interpreter directly, and
  the patched ``acados_template`` reads it unguarded (``KeyError``);
* matplotlib defaults to ``TkAgg`` in ``cocofest.result.plot`` and pops GUI
  windows open mid-run, or fails outright without a display;
* the benchmark scripts need ``GITHUB_WORKSPACE``, ``PYTHONPATH`` and the
  single-thread numerical policy of section 9.1.

The two Conda environments hold different CasADi ABIs, so a case is always run
with the interpreter of the environment that owns its solver. Cases belonging to
the other environment are dispatched to it automatically; the driver therefore
behaves the same whichever of the two interpreters the IDE is configured with.

Examples
--------
Run everything the machine can run, then report::

    python .github/scripts/run_benchmarks.py

Run one suite over five RHO::

    python .github/scripts/run_benchmarks.py --suite rho32 --cycles 5

Re-report without re-running any solver::

    python .github/scripts/run_benchmarks.py --report-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Set before anything imports cocofest: cocofest/result/plot.py calls
# matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg")) at import time.
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_RUNNER = REPO_ROOT / ".github" / "scripts" / "run_cycling_benchmark_case.sh"
SEED_BUILDER = REPO_ROOT / ".github" / "scripts" / "build_benchmark_seed_linux.sh"
SUMMARIZER = REPO_ROOT / ".github" / "scripts" / "summarize_cycling_benchmark.py"
SEED_FILES = (
    "common-reduced.npz",
    "common-full.npz",
    "reduced-cycling-fourier12.npz",
)


def finite_float(value: str) -> float:
    """Parse a finite CLI floating-point value."""

    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"must be a number, got {value!r}") from error
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"must be finite, got {value!r}")
    return parsed


def nonnegative_finite_float(value: str) -> float:
    """Parse a finite energy-equivalent torque."""

    parsed = finite_float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to zero")
    return parsed


def negative_finite_float(value: str) -> float:
    """Parse the crank convention's strictly negative isokinetic speed."""

    parsed = finite_float(value)
    if parsed >= 0:
        raise argparse.ArgumentTypeError("must be strictly negative")
    return parsed


def isokinetic_configuration_name(args: argparse.Namespace) -> str:
    """Return a collision-resistant name for one isokinetic configuration."""

    if args.formulation == "dynamic":
        return ""
    torque = format(args.energy_equivalent_torque, ".12g")
    omega = format(args.isokinetic_omega, ".12g")
    torque_min = format(args.load_torque_min, ".12g")
    torque_max = format(args.load_torque_max, ".12g")
    return (
        f"isokinetic-torque-{torque}-omega-{omega}"
        f"-load-{torque_min}-to-{torque_max}"
    )


def isokinetic_directory_suffix(args: argparse.Namespace) -> str:
    """Name an isokinetic campaign without changing dynamic result paths."""

    name = isokinetic_configuration_name(args)
    return f"-{name}" if name else ""


def case_result_dir_name(case: "Case", args: argparse.Namespace) -> str:
    """Return the case directory, isolated for an isokinetic campaign."""

    return f"{case.result_dir_name}{isokinetic_directory_suffix(args)}"


def default_worker_threads() -> int:
    """Return the physical cores available to this process when detectable."""

    logical_count = os.cpu_count() or 1
    try:
        allowed_cpus = os.sched_getaffinity(0)
    except AttributeError:
        allowed_cpus = range(logical_count)
    topology = set()
    for cpu in allowed_cpus:
        topology_root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            topology.add(
                (
                    (topology_root / "physical_package_id").read_text().strip(),
                    (topology_root / "core_id").read_text().strip(),
                )
            )
        except OSError:
            return len(allowed_cpus)
    return len(topology) or len(allowed_cpus)


@dataclass(frozen=True)
class Case:
    """One cell of the benchmark matrix offered by the repository.

    `key` selects the case on the command line and labels it in the report.
    `slug` is the first positional argument of run_cycling_benchmark_case.sh,
    which only uses it to name the result directory ``<slug>-<mechanics>``, so
    two cases may share a slug as long as their mechanics differ.
    """

    key: str
    solver: str
    backend: str
    collocation_degree: int
    compile_nlp: bool
    suite: str
    mechanics: str = "reduced"
    ipopt_profile: str = "periodic_collocation"
    dual_warm_start: str = "auto"
    target_refinement: str = "auto"
    kkt_predictor: bool = False
    kkt_dual_mode: str = "reset"
    slug: str = ""
    runner: str = "script"
    transcription: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.slug:
            object.__setattr__(self, "slug", self.key)
        if not self.transcription:
            object.__setattr__(self, "transcription", f"Radau {self.collocation_degree}")
        evaluators = "compiled" if self.compile_nlp else "interpreted"
        object.__setattr__(
            self, "tags", tuple(dict.fromkeys(self.tags + (evaluators, self.mechanics, self.solver)))
        )

    @property
    def result_dir_name(self) -> str:
        return f"{self.slug}-{self.mechanics}"


def _radau(solver: str, backend: str, suite: str, degree: int, *,
           mechanics: str = "reduced", compiled: bool = False) -> Case:
    """A scientific-radauN diagnostic case, as the workflow spells it."""
    stem = "fatrop-collocation" if solver == "fatrop" else (
        "madnlp-mumps" if solver == "madnlp" else solver)
    slug = f"{stem}-radau{degree}"
    key = slug if mechanics == "reduced" else f"{slug}-full"
    if compiled:
        key = f"{key}-compiled"
    return Case(key, solver, backend, degree, compiled, suite,
                mechanics=mechanics, ipopt_profile=f"scientific-radau{degree}",
                dual_warm_start="off", target_refinement="true",
                slug=slug, tags=("radau-sweep",))


CASES: tuple[Case, ...] = (
    # --- core matrix: periodic_collocation profile, Radau 3, both evaluator modes
    Case("ipopt", "ipopt", "mumps", 3, True, "rho32", tags=("core",)),
    Case("ipopt-interpreted", "ipopt", "mumps", 3, False, "rho32", tags=("core",)),
    Case("fatrop-collocation", "fatrop", "fatrop", 3, True, "rho32", tags=("core",)),
    Case("fatrop-collocation-interpreted", "fatrop", "fatrop", 3, False, "rho32",
         tags=("core",)),
    Case("madnlp-mumps", "madnlp", "mumps", 3, True, "madnlp32", tags=("core",)),
    Case("madnlp-mumps-interpreted", "madnlp", "mumps", 3, False, "madnlp32",
         tags=("core",)),

    # --- full mechanical formulation; the workflow never compiles these
    Case("ipopt-full", "ipopt", "mumps", 3, False, "rho32",
         mechanics="full", slug="ipopt", tags=("core",)),
    Case("fatrop-collocation-full", "fatrop", "fatrop", 3, False, "rho32",
         mechanics="full", slug="fatrop-collocation", tags=("core",)),
    Case("madnlp-mumps-full", "madnlp", "mumps", 3, False, "madnlp32",
         mechanics="full", slug="madnlp-mumps", tags=("core",)),

    # --- Radau order sweep, the transcription-error diagnostic of section 15
    _radau("ipopt", "mumps", "rho32", 4),
    _radau("ipopt", "mumps", "rho32", 5),
    _radau("ipopt", "mumps", "rho32", 6),
    _radau("ipopt", "mumps", "rho32", 5, mechanics="full"),
    _radau("madnlp", "mumps", "madnlp32", 4),
    _radau("madnlp", "mumps", "madnlp32", 5),
    _radau("madnlp", "mumps", "madnlp32", 6),
    _radau("madnlp", "mumps", "madnlp32", 5, mechanics="full"),

    # --- cross-solver comparison at a fixed transcription, compiled evaluators
    Case("ipopt-radau3-comparison", "ipopt", "mumps", 3, True, "rho32",
         ipopt_profile="scientific-radau3", dual_warm_start="off",
         target_refinement="true", tags=("comparison",)),
    Case("fatrop-radau3-comparison", "fatrop", "fatrop", 3, True, "rho32",
         ipopt_profile="scientific-radau3", dual_warm_start="off",
         target_refinement="true", tags=("comparison",)),
    Case("madnlp-radau3-comparison", "madnlp", "mumps", 3, True, "madnlp32",
         ipopt_profile="scientific-radau3", dual_warm_start="off",
         target_refinement="true", tags=("comparison",)),
    Case("ipopt-radau5-comparison", "ipopt", "mumps", 5, True, "rho32",
         ipopt_profile="scientific-radau5", dual_warm_start="off",
         target_refinement="true", tags=("comparison",)),
    Case("fatrop-radau5-comparison", "fatrop", "fatrop", 5, True, "rho32",
         ipopt_profile="scientific-radau5", dual_warm_start="off",
         target_refinement="true", tags=("comparison",)),
    Case("madnlp-radau5-comparison", "madnlp", "mumps", 5, True, "madnlp32",
         ipopt_profile="scientific-radau5", dual_warm_start="off",
         target_refinement="true", tags=("comparison",)),

    # --- IPOPT warm-start and parametric-KKT variants
    Case("ipopt-dual-all", "ipopt", "mumps", 3, True, "rho32",
         dual_warm_start="all", tags=("variants",)),
    Case("ipopt-kkt-reset", "ipopt", "mumps", 3, True, "rho32",
         kkt_predictor=True, kkt_dual_mode="reset", tags=("variants",)),
    Case("ipopt-kkt-preserve", "ipopt", "mumps", 3, True, "rho32",
         kkt_predictor=True, kkt_dual_mode="preserve", dual_warm_start="all",
         tags=("variants",)),
    Case("ipopt-kkt-predict", "ipopt", "mumps", 3, True, "rho32",
         kkt_predictor=True, kkt_dual_mode="predict", tags=("variants",)),

    # --- ACADOS, which is not a run_cycling_benchmark_case.sh solver
    Case("acados-irk", "acados", "hpipm", 0, False, "rho32",
         runner="acados", transcription="SQP/IRK GL 4x5", tags=("acados",)),
)

ALL_TAGS = tuple(dict.fromkeys(tag for case in CASES for tag in case.tags))


# Lines worth echoing to the IDE console. Everything else goes to the per-case
# log file only, because a single case emits thousands of lines.
INTERESTING = re.compile(
    r"solver overview \||window\[\d+\] status=|benchmark heartbeat|benchmark JSON:"
    r"|Traceback|FAILED|Segmentation fault|Killed|MemoryError"
)
# CUDA warnings come from the optional MadNLPGPU extension on a CPU-only host.
NOISE = re.compile(r"CUDA\.jl|CUDA runtime|CUDA_Runtime|cuda\.juliagpu\.org")


def conda_env_prefix(suite: str) -> Path | None:
    """Locate the Conda prefix owning `suite`, starting from the running one."""

    name = f"cocofest-{suite}"
    running = Path(sys.prefix)
    candidates = [
        running if running.name == name else None,
        running.parent / name,
        Path(os.environ.get("MINIFORGE_PREFIX", Path.home() / "miniforge3")) / "envs" / name,
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "bin" / "python").exists():
            return candidate
    return None


def base_environment(prefix: Path, suite: str, threads: int,
                     numeric_threads: int = 1) -> dict[str, str]:
    """Everything an IDE-launched benchmark process needs but does not inherit.

    Shared with run_full_horizon.py so both drivers apply one policy.
    """

    env = os.environ.copy()

    # Putting the owning environment first makes the `python` that the benchmark
    # shell scripts invoke the right one, even when the IDE started us from the
    # other environment.
    env["CONDA_PREFIX"] = str(prefix)
    env["CONDA_DEFAULT_ENV"] = prefix.name
    env["PATH"] = os.pathsep.join([str(prefix / "bin"), env.get("PATH", "")])

    env["GITHUB_WORKSPACE"] = str(REPO_ROOT)
    existing_pythonpath = env.get("PYTHONPATH", "")
    if str(REPO_ROOT) not in existing_pythonpath.split(os.pathsep):
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT)] + ([existing_pythonpath] if existing_pythonpath else [])
        )
    env["MPLBACKEND"] = "Agg"
    # Do not let headless/local runners write caches into $HOME. Apart from
    # making runs self-contained, this also supports service accounts and
    # sandboxed runners whose home directory is deliberately read-only.
    matplotlib_cache = REPO_ROOT / ".cache" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(matplotlib_cache)
    env["CASADI_CXX_ABI"] = "0" if suite == "rho32" else "1"

    # Not strictly required -- libacados and the CasADi MadNLP plugin both carry
    # a usable RUNPATH -- but it keeps an IDE run byte-for-byte comparable with
    # the documented shell protocol.
    if suite == "rho32":
        library_dirs = [str(prefix / "lib")]
    else:
        cache = REPO_ROOT / ".cache" / "madnlp-mumps"
        library_dirs = [str(cache / "lib"), str(cache / "share" / "julia" / "lib")]

        # Julia packages can still be read from the user's normal depot, while
        # logs, scratch metadata and compiled caches land in the repository's
        # ignored writable cache. Preserve an explicit depot path when callers
        # already provide one.
        writable_depot = REPO_ROOT / ".cache" / "julia-benchmark-depot"
        writable_depot.mkdir(parents=True, exist_ok=True)
        fallback_depot = env.get("JULIA_DEPOT_PATH") or str(Path.home() / ".julia")
        depots = fallback_depot.split(os.pathsep)
        if str(writable_depot) not in depots:
            env["JULIA_DEPOT_PATH"] = os.pathsep.join([str(writable_depot), *depots])
    previous = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        library_dirs + ([previous] if previous else [])
    )

    # Section 9.1: stage/map parallelism on the benchmark side, one thread in
    # every numerical library so timings stay comparable.
    env["BENCHMARK_THREADS"] = str(threads)
    for variable in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "JULIA_NUM_THREADS",
    ):
        env[variable] = str(numeric_threads)
    env["OMP_DYNAMIC"] = "FALSE"
    return env


def build_case_environment(case: Case, prefix: Path, args: argparse.Namespace) -> dict[str, str]:
    env = base_environment(prefix, case.suite, args.threads, args.numeric_threads)

    # run_cycling_benchmark_case.sh lets these environment variables win over its
    # positional arguments, so set them from the case rather than inheriting a
    # stale value from the caller's shell.
    # The isokinetic RHO transfer is materially better conditioned in MadNLP
    # when bound multipliers are retained.  Its default 1e-6 termination can
    # otherwise report success while the independently reconstructed primal
    # still violates a bound by O(2e-5).  Keep this backend-specific numerical
    # policy out of the continuous OCP and make the benchmark gate stricter.
    env["DUAL_WARM_START"] = (
        "bounds"
        if args.formulation == "isokinetic" and case.solver == "madnlp"
        else case.dual_warm_start
    )
    if args.formulation == "isokinetic" and case.solver == "madnlp":
        env["BENCHMARK_NLP_TOLERANCE"] = "1e-8"
    env["PARAMETRIC_KKT_PREDICTOR"] = "true" if case.kkt_predictor else "false"
    env["PARAMETRIC_KKT_DUAL_MODE"] = case.kkt_dual_mode

    env["BENCHMARK_CYCLES"] = str(args.cycles)
    env["BENCHMARK_CYCLES_PER_WINDOW"] = str(args.cycles_per_window)
    env["BENCHMARK_ASSISTANCE"] = args.assistance
    env["BENCHMARK_Q_SLACK"] = args.q_slack
    env["BENCHMARK_MAX_ITER"] = str(args.max_iter)
    # The shell runner reads these names too, so direct shell and IDE launches
    # use one unambiguous interface for every NLP backend.
    env["BENCHMARK_FORMULATION"] = args.formulation
    env["BENCHMARK_ENERGY_EQUIVALENT_TORQUE"] = str(args.energy_equivalent_torque)
    env["BENCHMARK_ISOKINETIC_OMEGA"] = str(args.isokinetic_omega)
    env["BENCHMARK_LOAD_TORQUE_MIN"] = str(args.load_torque_min)
    env["BENCHMARK_LOAD_TORQUE_MAX"] = str(args.load_torque_max)
    if args.formulation == "isokinetic":
        # The isokinetic reference is IPOPT/MA57.  A deployment may have MA57
        # linked directly into IPOPT; otherwise IPOPT_HSL_LIBRARY supplies the
        # CoinHSL shared object through the shell runner.
        env["IPOPT_LINEAR_SOLVER"] = "ma57"
        env["WARMUP_IPOPT_LINEAR_SOLVER"] = "ma57"
        if args.ipopt_hsl_library is not None:
            env["IPOPT_HSL_LIBRARY"] = str(args.ipopt_hsl_library)
    return env


def build_command(case: Case, prefix: Path, args: argparse.Namespace) -> tuple[list[str], Path]:
    """Return the command for `case` and the directory it must run from."""

    if case.runner == "script":
        return (
            [
                "bash",
                str(CASE_RUNNER),
                case.slug,
                case.solver,
                case.mechanics,
                case.backend,
                "collocation",
                args.output_root,
                str(args.cycles),
                "true" if case.compile_nlp else "false",
                "sx",
                "none",
                str(case.collocation_degree),
                case.ipopt_profile,
                case.dual_warm_start,
                case.target_refinement,
            ],
            REPO_ROOT,
        )

    # ACADOS. Flags mirror the reduced reference case of
    # run_ryzen5950x_endurance_sweep.sh; only the horizon, the thread count and
    # the crank torque follow this driver's options. ACADOS emits its generated
    # C into the working directory, so each case gets its own codegen folder.
    case_dir = REPO_ROOT / args.output_root / case_result_dir_name(case, args)
    codegen_dir = case_dir / "codegen"
    codegen_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = REPO_ROOT / "benchmark-seed"
    command = [
        "python",
        str(REPO_ROOT / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"),
        "--solvers", "acados",
        "--objective", "fatigue",
        "--ipopt-profile",
        "scientific-radau5" if args.formulation == "isokinetic" else "periodic_collocation",
        "--ipopt-use-sx",
        "--ipopt-enforce-start-constraints",
        "--cycles-per-window", str(args.cycles_per_window),
        "--stimulations-per-cycle", "30",
        "--n-windows", str(args.cycles),
        "--n-threads", str(args.threads),
        "--crank-assistance", args.assistance,
        "--primal-feasibility-threshold", "1e-5",
        "--max-consecutive-failing", "2",
        "--retry-failed-rho-without-advance",
        *(
            [
                "--standard-warmup-seed",
                str(
                    REPO_ROOT
                    / ".github/benchmark-seeds/legacy-resistive-0p22-warmup.npz"
                ),
                "--legacy-standard-warmup-seed-signed-torque",
                "0.22",
                "--standard-warmup-seed-continuation",
                "--common-initial-solution",
                str(seed_dir / f"common-{case.mechanics}.npz"),
            ]
            if args.formulation == "dynamic"
            else []
        ),
        "--warmup-ipopt-linear-solver",
        "ma57" if args.formulation == "isokinetic" else "mumps",
        "--ipopt-linear-solver",
        "ma57" if args.formulation == "isokinetic" else "mumps",
        *(
            ["--ipopt-hsl-library", str(args.ipopt_hsl_library)]
            if args.ipopt_hsl_library is not None
            else []
        ),
        "--ipopt-disable-historical-initial-guess",
        "--reduced-cycling-profile", str(seed_dir / "reduced-cycling-fourier12.npz"),
        "--state-scaling", "full",
        "--first-node-wheel-q-slack", "0",
        "--terminal-wheel-q-slack", args.q_slack,
        "--mechanical-formulation", case.mechanics,
        "--formulation", args.formulation,
        "--energy-equivalent-torque", str(args.energy_equivalent_torque),
        "--isokinetic-omega", str(args.isokinetic_omega),
        "--load-torque-min", str(args.load_torque_min),
        "--load-torque-max", str(args.load_torque_max),
        "--experimental-reduced-acados",
        "--acados-dir", str(prefix),
        "--acados-check-reuse-possible",
        "--acados-max-iter", "100",
        "--acados-nlp-solver-type",
        (
            "SQP_WITH_FEASIBLE_QP"
            if args.formulation == "isokinetic"
            else "SQP"
        ),
        *(
            ["--acados-search-direction-mode", "BYRD_OMOJOKUN"]
            if args.formulation == "isokinetic"
            else []
        ),
        "--acados-integrator-type", "IRK",
        "--acados-collocation-type", "GAUSS_LEGENDRE",
        "--acados-sim-stages", "4",
        "--acados-sim-steps", "5",
        "--acados-newton-iter", "5",
        "--acados-stationarity-tolerance", "5e-3",
        *(
            ["--acados-control-homotopy-release-final-radius"]
            if args.formulation == "dynamic"
            else []
        ),
        *(
            ["--periodic-ipopt-refinement-each-window"]
            if args.formulation == "isokinetic"
            else []
        ),
        *(
            [
                "--periodic-ipopt-refinement-ode-solver", "collocation",
                "--periodic-ipopt-refinement-collocation-degree", "5",
                "--periodic-ipopt-refinement-collocation-method", "radau",
            ]
            if args.formulation == "isokinetic"
            else []
        ),
        "--compact-rho-output",
        "--print-traces",
        "--codegen-tag", f"ide-{case.key}",
        "--output-json", str(case_dir / "result.json"),
    ]
    return command, codegen_dir


def run_case(case: Case, prefix: Path, args: argparse.Namespace, log_dir: Path) -> int:
    command, working_directory = build_command(case, prefix, args)
    log_path = log_dir / f"{case.key}.log"
    started = time.monotonic()

    print(f"\n{'=' * 78}")
    print(f"  {case.key}  ({case.solver}/{case.mechanics}, {case.transcription}, "
          f"{'compiled' if case.compile_nlp else 'interpreted'} evaluators)")
    print(f"  environment : {prefix.name}")
    print(f"  full log    : {log_path}")
    print(f"{'=' * 78}", flush=True)

    environment = build_case_environment(case, prefix, args)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=working_directory,
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
                print(f"    {line.rstrip()}", flush=True)
        returncode = process.wait()

    elapsed = time.monotonic() - started
    print(f"  -> exit {returncode} in {elapsed:.1f} s", flush=True)
    if returncode != 0:
        print(f"  last lines of {log_path}:", flush=True)
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        for line in tail:
            print(f"    | {line}", flush=True)
    return returncode


def read_result(result_path: Path) -> dict | None:
    """Return the first result record, or None when it cannot be read."""

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return None
    return results[0]


def _first_window_field(record: dict, field: str):
    windows = record.get("windows")
    if isinstance(windows, list) and windows and isinstance(windows[0], dict):
        return windows[0].get(field)
    return None


def report(selected: list[Case], args: argparse.Namespace) -> int:
    output_root = REPO_ROOT / args.output_root
    rows = []
    for case in selected:
        record = read_result(output_root / case_result_dir_name(case, args) / "result.json")
        rows.append((case, record))

    print(f"\n{'=' * 100}")
    print(f"  BENCHMARK REPORT  --  {args.cycles} RHO, "
          f"{args.threads} benchmark threads, resistance {args.assistance}")
    print(f"{'=' * 100}")
    header = (f"{'case':30} {'env':10} {'status':9} {'RHO':>7} {'iter':>6} "
              f"{'objective':>15} {'solve s':>9} {'wall s':>8}")
    print(header)
    print("-" * len(header))

    passed = 0
    failures = 0
    for case, record in rows:
        if record is None:
            failures += 1
            print(f"{case.key:30} {case.suite:10} {'NO RESULT':9} "
                  f"{'-':>7} {'-':>6} {'-':>15} {'-':>9} {'-':>8}")
            continue

        ok = bool(record.get("success"))
        passed += 1 if ok else 0
        failures += 0 if ok else 1
        validated = record.get("validated_cycles")
        requested = record.get("requested_cycles")
        objective = record.get("window_objective_sum")
        iterations = _first_window_field(record, "iterations")
        solve_seconds = _first_window_field(record, "solver_time_s")
        wall_seconds = record.get("end_to_end_wall_time_s")

        def number(value, spec: str) -> str:
            return format(value, spec) if isinstance(value, (int, float)) else "-"

        print(f"{case.key:30} {case.suite:10} {'PASS' if ok else 'FAIL':9} "
              f"{f'{validated}/{requested}':>7} {number(iterations, '>6'):>6} "
              f"{number(objective, '>15.9f'):>15} {number(solve_seconds, '>9.3f'):>9} "
              f"{number(wall_seconds, '>8.1f'):>8}")

        if not ok:
            detail = record.get("error") or f"first failed RHO: {record.get('first_failed_rho')}"
            print(f"{'':30} -> {detail}")

    print("-" * len(header))
    print(f"  {passed}/{len(rows)} cases passed"
          + (f", {failures} to investigate" if failures else ""))

    # Objective spread is the cross-solver agreement check: the same transcription
    # solved by different solvers must land on the same optimum.
    by_transcription: dict[str, list[float]] = {}
    for case, record in rows:
        if record is None:
            continue
        value = record.get("window_objective_sum")
        if isinstance(value, (int, float)):
            by_transcription.setdefault(case.transcription, []).append(value)
    for transcription, values in sorted(by_transcription.items()):
        if len(values) > 1:
            spread = (max(values) - min(values)) / abs(min(values))
            print(f"  {transcription}: {len(values)} solvers agree to {spread:.2e} relative")
        else:
            print(f"  {transcription}: single solver, no cross-check")

    summary_dir = REPO_ROOT / args.summary_dir
    if args.formulation == "isokinetic":
        summary_dir /= isokinetic_configuration_name(args)
    result_files = sorted(
        str(output_root / case_result_dir_name(case, args) / "result.json")
        for case, record in rows
        if record is not None
    )
    if result_files and not args.no_summary:
        print(f"\n  writing detailed summary to {summary_dir} ...", flush=True)
        completed = subprocess.run(
            [sys.executable, str(SUMMARIZER), *result_files, "--output-dir", str(summary_dir)],
            cwd=REPO_ROOT,
            env={**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": str(REPO_ROOT)},
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            print(f"  summary tables : {summary_dir}/benchmark-comparison.md")
            print(f"  per-RHO timings: {summary_dir}/rho-timings.csv")
            if completed.stdout.strip():
                for line in completed.stdout.strip().splitlines():
                    print(f"  note: {line}")
        else:
            print(f"  summary failed (exit {completed.returncode}):")
            for line in (completed.stderr or completed.stdout).strip().splitlines()[-8:]:
                print(f"    | {line}")

    print(f"{'=' * 100}\n")
    return failures


def check_seed() -> list[str]:
    seed_dir = REPO_ROOT / "benchmark-seed"
    return [name for name in SEED_FILES if not (seed_dir / name).exists()]


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        choices=("rho32", "madnlp32", "all"),
        default="all",
        help="which environment's cases to run (default: all)",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="CASE",
        help="explicit case keys; overrides --suite and --tags. See --list.",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        metavar="TAG",
        help=f"keep only cases carrying every given tag. Available: {', '.join(ALL_TAGS)}",
    )
    parser.add_argument("--cycles", type=int, default=1, help="RHO count (default: 1)")
    parser.add_argument("--cycles-per-window", type=int, default=1)
    parser.add_argument("--threads", type=int, default=default_worker_threads(),
                        help="BENCHMARK_THREADS (default: available physical cores)")
    parser.add_argument("--numeric-threads", type=int, default=1,
                        help="threads inside BLAS/OpenMP/Julia (default: 1, see section 9.1)")
    parser.add_argument("--assistance", default="0.00",
                        help="crank torque; 'signed:+0.15' is a resistance of 0.15 N.m")
    parser.add_argument("--q-slack", default="0.002")
    parser.add_argument(
        "--ipopt-hsl-library",
        type=Path,
        default=(
            Path(os.environ["IPOPT_HSL_LIBRARY"])
            if os.environ.get("IPOPT_HSL_LIBRARY")
            else None
        ),
        help=(
            "optional CoinHSL shared library passed to IPOPT as hsllib; "
            "needed when MA57 is not linked directly into IPOPT"
        ),
    )
    parser.add_argument(
        "--formulation",
        choices=("dynamic", "isokinetic"),
        default="dynamic",
        help="cycling mechanics formulation (default: dynamic)",
    )
    parser.add_argument(
        "--energy-equivalent-torque",
        type=nonnegative_finite_float,
        default=0.2,
        metavar="N_M",
        help="non-negative torque defining the isokinetic work target (default: 0.2)",
    )
    parser.add_argument(
        "--isokinetic-omega",
        type=negative_finite_float,
        default=-2 * math.pi,
        metavar="RAD_S",
        help="strictly negative target crank velocity (default: -2*pi)",
    )
    parser.add_argument(
        "--load-torque-min",
        type=finite_float,
        default=-3.0,
        metavar="N_M",
        help="finite lower bound on isokinetic load torque (default: -3)",
    )
    parser.add_argument(
        "--load-torque-max",
        type=finite_float,
        default=3.0,
        metavar="N_M",
        help="finite upper bound on isokinetic load torque (default: 3)",
    )
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--output-root", default="local-results")
    parser.add_argument("--summary-dir", default="local-summary")
    parser.add_argument("--verbose", action="store_true",
                        help="echo every solver line instead of the milestones only")
    parser.add_argument("--report-only", action="store_true",
                        help="report on existing results without running any solver")
    parser.add_argument("--no-summary", action="store_true",
                        help="skip summarize_cycling_benchmark.py")
    parser.add_argument("--build-seed", action="store_true",
                        help="build benchmark-seed/ first (needs the rho32 environment)")
    parser.add_argument("--seed-only", action="store_true",
                        help="build benchmark-seed/ and stop, running no solver case")
    parser.add_argument("--list", action="store_true", help="list the cases and exit")
    args = parser.parse_args(argv)
    if not args.load_torque_min < 0.0 < args.load_torque_max:
        parser.error(
            "--load-torque-min/--load-torque-max must allow assistance and "
            "resistance (MIN < 0 < MAX)"
        )
    if args.energy_equivalent_torque > args.load_torque_max:
        parser.error(
            "--energy-equivalent-torque cannot exceed --load-torque-max"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)

    if args.list:
        print(f"{'case':32} {'env':10} {'solver':7} {'mech':8} {'transcription':16} "
              f"{'evaluators':12} profile")
        for case in CASES:
            print(f"{case.key:32} {case.suite:10} {case.solver:7} {case.mechanics:8} "
                  f"{case.transcription:16} "
                  f"{'compiled' if case.compile_nlp else 'interpreted':12} "
                  f"{case.ipopt_profile}")
        print(f"\n{len(CASES)} cases. Tags: {', '.join(ALL_TAGS)}")
        return 0

    if args.cases:
        known = {case.key: case for case in CASES}
        unknown = [name for name in args.cases if name not in known]
        if unknown:
            print(f"Unknown case(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"Available: {', '.join(known)}", file=sys.stderr)
            return 2
        selected = [known[name] for name in args.cases]
    elif args.suite == "all":
        selected = list(CASES)
    else:
        selected = [case for case in CASES if case.suite == args.suite]

    if args.tags and not args.cases:
        unknown = [tag for tag in args.tags if tag not in ALL_TAGS]
        if unknown:
            print(f"Unknown tag(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"Available: {', '.join(ALL_TAGS)}", file=sys.stderr)
            return 2
        selected = [c for c in selected if all(tag in c.tags for tag in args.tags)]
        if not selected:
            print(f"No case carries every tag: {', '.join(args.tags)}", file=sys.stderr)
            return 2

    if args.report_only:
        return 1 if report(selected, args) else 0

    if not CASE_RUNNER.exists():
        print(f"Missing {CASE_RUNNER}", file=sys.stderr)
        return 1

    if args.build_seed or args.seed_only:
        prefix = conda_env_prefix("rho32")
        if prefix is None:
            print("Building the seed needs the cocofest-rho32 environment.", file=sys.stderr)
            return 1
        print("Building benchmark-seed/ (two IPOPT solves) ...", flush=True)
        seed_environment = build_case_environment(CASES[0], prefix, args)
        completed = subprocess.run(
            ["bash", str(SEED_BUILDER)], cwd=REPO_ROOT, env=seed_environment
        )
        if completed.returncode != 0:
            print("Seed build failed.", file=sys.stderr)
            return completed.returncode
        if args.seed_only:
            print("Seed ready in benchmark-seed/; no solver case was run.")
            return 0

    missing_seed = check_seed()
    if missing_seed:
        print(f"Missing seed file(s): {', '.join(missing_seed)}", file=sys.stderr)
        print("Run this script once with --build-seed, or follow section 10 of "
              "docs/cycling_solver_benchmark/linux_32core_setup.md.", file=sys.stderr)
        return 1

    # Resolve every environment up front so a missing one is reported before a
    # long run rather than halfway through it.
    prefixes: dict[str, Path] = {}
    runnable: list[Case] = []
    for case in selected:
        if case.suite not in prefixes:
            found = conda_env_prefix(case.suite)
            if found is None:
                print(f"Environment cocofest-{case.suite} not found; "
                      f"skipping its cases.", file=sys.stderr)
            prefixes[case.suite] = found  # type: ignore[assignment]
        if prefixes[case.suite] is not None:
            runnable.append(case)

    if not runnable:
        print("No case can be run: neither environment was found.", file=sys.stderr)
        return 1

    log_dir = REPO_ROOT / args.output_root / "_logs"
    if args.formulation == "isokinetic":
        log_dir /= isokinetic_configuration_name(args)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository : {REPO_ROOT}")
    print(f"Cases      : {', '.join(case.key for case in runnable)}")
    print(f"RHO        : {args.cycles}   threads: {args.threads} "
          f"(numerical libraries: {args.numeric_threads})")

    started = time.monotonic()
    for case in runnable:
        run_case(case, prefixes[case.suite], args, log_dir)
    print(f"\nAll cases finished in {time.monotonic() - started:.1f} s.")

    return 1 if report(runnable, args) else 0


if __name__ == "__main__":
    raise SystemExit(main())
