"""Launch an isolated exact FHO65 with the corrected Bioptim ThreadMap build.

The launch is deliberately fixed to the audited 0.3 Nm FHO65 seed and FHO62
prefix. It creates a new output directory and never changes campaign data.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python")
WORKTREE = REPO / ".benchmark-deps/bioptim-threadmap-continuity"
EXPECTED_COMMIT = "33424093d6ed4070266f4c2b43f8fd20e9719bf4"
HSL = Path("/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so")
BASE = REPO / "local-results/fho65-native-hessian-audit-20260925/threads-8-ones/command.json"
OUTPUT = REPO / "local-results/fho65-threadmap-exact-12c-20260925"
CPUS = "16-27"
MAX_WALL_SECONDS = 21600


def option(args: list[str], flag: str, value: str) -> None:
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args.extend((flag, value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_metadata(path: Path, *, cycles: int) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata__json"].item()))
    if int(metadata["cycles_per_window"]) != cycles:
        raise ValueError(f"{path} does not contain {cycles} cycles")
    if float(metadata["constant_crank_torque"]) != 0.3:
        raise ValueError(f"{path} has a different signed crank torque")
    if metadata["crank_torque_role"] != "resistive":
        raise ValueError(f"{path} does not encode resistance")
    if metadata["formulation"] != "dynamic" or metadata["mechanical_formulation"] != "reduced":
        raise ValueError(f"{path} has a different mechanical formulation")
    return {key: metadata.get(key) for key in (
        "cycles_per_window", "constant_crank_torque", "crank_torque_role",
        "formulation", "mechanical_formulation")}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not set(range(16, 28)).issubset(os.sched_getaffinity(0)):
        raise RuntimeError("CPU affinity 16-27 is unavailable")
    for path in (PYTHON, HSL, BASE):
        path.resolve(strict=True)
    commit = subprocess.check_output(
        ("git", "-C", str(WORKTREE), "rev-parse", "HEAD"), text=True).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Unexpected corrected Bioptim commit: {commit}")
    dirty = subprocess.check_output(
        ("git", "-C", str(WORKTREE), "status", "--porcelain=v1"), text=True)
    if dirty:
        raise RuntimeError("Corrected Bioptim worktree is dirty")

    base = json.loads(BASE.read_text())
    if Path(base[0]) != PYTHON or "--single-shot" not in base:
        raise RuntimeError("Base command is not the audited FHO65 single-shot")
    cli = base[2:]
    for forbidden in ("--ipopt-c-compile", "--ipopt-function-transform"):
        if forbidden in cli:
            raise RuntimeError(f"Compilation/transformation forbidden: {forbidden}")
    if cli[cli.index("--signed-crank-torque") + 1] != "0.3":
        raise RuntimeError("Base command is not the audited 0.3 Nm resistance")
    if cli[cli.index("--cycles-per-window") + 1] != "65":
        raise RuntimeError("Base command is not FHO65")
    if cli[cli.index("--n-windows") + 1] != "65":
        raise RuntimeError("Base command does not begin at cycle 1")
    seed = Path(cli[cli.index("--common-initial-solution") + 1]).resolve(strict=True)
    prefix = Path(cli[cli.index("--full-horizon-prefix-solution") + 1]).resolve(strict=True)
    seed_meta = seed_metadata(seed, cycles=65)
    prefix_meta = seed_metadata(prefix, cycles=62)
    prefix_result = prefix.with_name("result.json")
    prior = json.loads(prefix_result.read_text())
    if not prior["results"][0]["solver_success"] or not prior["results"][0]["success"]:
        raise RuntimeError("FHO62 prefix is not a successful reference")

    env = os.environ.copy()
    env.update({key: "1" for key in (
        "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env["OMP_DYNAMIC"] = "FALSE"
    env["PYTHONPATH"] = f"{WORKTREE}:{REPO}"
    env["LD_LIBRARY_PATH"] = f"{PYTHON.parent.parent / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"

    preflight = subprocess.run(
        (str(PYTHON), "-c",
         "import bioptim,inspect; from bioptim.limits.penalty_option import PenaltyOption; "
         "assert 'unused_continuity_map' in inspect.getsource(PenaltyOption._set_penalty_function); "
         "print(bioptim.__file__)"),
        env=env, cwd=REPO, text=True, capture_output=True, check=True,
    )
    if Path(preflight.stdout.strip()).resolve() != (WORKTREE / "bioptim/__init__.py").resolve():
        raise RuntimeError(f"Unexpected Bioptim import: {preflight.stdout!r}")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    env["MPLCONFIGDIR"] = str(OUTPUT / "mpl")
    (OUTPUT / "mpl").mkdir()
    option(cli, "--n-threads", "12")
    option(cli, "--ipopt-max-iter", "4000")
    option(cli, "--ipopt-hessian-approximation", "exact")
    option(cli, "--ipopt-linear-solver", "ma57")
    option(cli, "--ipopt-hsl-library", str(HSL))
    option(cli, "--ipopt-print-level", "5")
    option(cli, "--common-initial-solution-output", str(OUTPUT / "full-solution.npz"))
    option(cli, "--output-json", str(OUTPUT / "result.json"))
    command = (
        "taskset", "-c", CPUS, "timeout", "--signal=INT", "--kill-after=60s",
        f"{MAX_WALL_SECONDS}s", str(PYTHON), base[1], *cli,
    )
    manifest = {
        "mode": "fresh_single_shot_fho65_exact",
        "corrected_bioptim": str(WORKTREE), "corrected_bioptim_commit": commit,
        "python": str(PYTHON), "casadi_graph": "MX", "c_compile": False,
        "ipopt_hessian_approximation": "exact", "ipopt_linear_solver": "ma57",
        "ipopt_max_iter": 4000, "max_wall_seconds": MAX_WALL_SECONDS,
        "cpu_affinity": CPUS, "bioptim_threads": 12,
        "numeric_threads": {key: env[key] for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "seed": {"path": str(seed), "sha256": sha256(seed), "metadata": seed_meta},
        "prefix": {"path": str(prefix), "sha256": sha256(prefix), "metadata": prefix_meta},
        "base_command": str(BASE), "command": list(command),
        "result": str(OUTPUT / "result.json"),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (OUTPUT / "runner.log").open("wb") as log:
        child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True)
    (OUTPUT / "runner.pid").write_text(str(child.pid) + "\n")
    print(json.dumps({"pid": child.pid, "output": str(OUTPUT), "commit": commit,
                      "seed_cycles": seed_meta["cycles_per_window"],
                      "prefix_cycles": prefix_meta["cycles_per_window"]}))


if __name__ == "__main__":
    main()
