#!/usr/bin/env python3
"""Rebuild fixed-weight Ding-local RHO-BO trajectories and continue each with FHO.

Each resistance gets an isolated worker/output tree.  RHO runs first with the
best available fixed BO weights and exports its complete certified prefix.
The same prefix is then re-optimised as an FHO before any cycles are added.
Subsequent FHO attempts try +3, +2, then +1, with only accepted FHO archives
promoted.  CasADi graphs are process-local; preparation manifests document
the next horizons but do not claim to cache/reuse a live graph.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PY = Path("/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python")
HSL = Path("/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so")
CYCLING = ROOT / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"
LOCAL_ADAPTER = ROOT / "scripts/benchmark_ding_radau5_local_ab.py"
RESISTANCES = (1.30, 1.20, 1.15, 1.00)
WEIGHTS = {
    1.30: (0.027072029708452478, 0.018486312493212677, 0.1415022510966415, 0.9999686296611382),
    1.20: (0.7759829510975516, 0.10285000528229762, 0.2734333864989573, 0.9776964807436896),
    1.15: (0.1910322016669409, 0.3183903113732652, 0.12432087215104703, 0.8232748612680016),
    1.00: (0.05819864273668182, 0.04710259143276053, 0.17401047897077157, 0.9298237222936032),
}
CPU_SETS = {1.30: "0-9", 1.20: "10-19", 1.15: "20-29", 1.00: "0-9"}
RELATIVE_ENERGY_LIMIT = 0.002
MAX_RHO_CYCLES = 240
FHO_THREADS = 10


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def env_for(out: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLCONFIGDIR": str(out / ".matplotlib"),
    }


def run_logged(command: list[str], log: Path, out: Path, *, allow_failure: bool = False) -> bool:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=ROOT, env=env_for(out), stdout=stream,
                                   stderr=subprocess.STDOUT, check=False)
    if completed.returncode and not allow_failure:
        raise RuntimeError(f"Command exited {completed.returncode}; see {log}")
    return completed.returncode == 0


def objective_args(weights: tuple[float, ...]) -> list[str]:
    return ["--objective", "fatigue", "--objective-shape", "quadratic",
            "--parametric-fatigue-weights", "--fatigue-weight-values",
            *(f"{x:.15g}" for x in weights)]


def common_protocol(resistance: float) -> list[str]:
    return ["--solvers", "ipopt", *objective_args(WEIGHTS[resistance]),
            "--formulation", "isokinetic", "--mechanical-formulation", "reduced",
            "--isokinetic-kinematics", "prescribed", "--isokinetic-omega", "-6.283185307179586",
            "--enforce-isokinetic-work-per-cycle", "--energy-equivalent-torque", f"{resistance:.12g}",
            "--load-torque-min", "-4", "--load-torque-max", "4", "--stimulations-per-cycle", "30",
            "--ipopt-profile", "periodic_collocation", "--ipopt-collocation-method", "radau",
            "--ipopt-collocation-degree", "5", "--ipopt-use-sx", "--ipopt-linear-solver", "ma57",
            "--warmup-ipopt-linear-solver", "ma57", "--ipopt-hsl-library", str(HSL),
            "--nlp-tolerance", "1e-8", "--primal-feasibility-threshold", "1e-5",
            "--ipopt-enforce-start-constraints", "--ipopt-disable-standard-warmup",
            "--adopt-common-initial-solution-warmup-cycles",
            "--common-initial-solution-recenter-first-node-bounds"]


def rho_full_command(resistance: float, out: Path) -> list[str]:
    return ["taskset", "-c", CPU_SETS[resistance], str(PY), str(LOCAL_ADAPTER), "local", "--",
            *common_protocol(resistance), "--ipopt-max-iter", "1000", "--cycles-per-window", "1",
            "--n-windows", str(MAX_RHO_CYCLES), "--n-threads", "1", "--compact-rho-output",
            "--retry-failed-rho-without-advance", "--receding-horizon-solution-output",
            str(out / "rho-bo-trajectory.npz"), "--allow-partial-receding-horizon-solution-output",
            "--rho-replay-checkpoint-output", str(out / "rho-last-certified-cycle.npz"),
            "--output-json", str(out / "rho-bo-result.json")]


def fho_command(resistance: float, out: Path, seed: Path, solution: Path, result: Path,
                cycles: int, *, prefix: Path | None = None, iterations: int = 1200) -> list[str]:
    cmd = ["taskset", "-c", CPU_SETS[resistance], str(PY), str(LOCAL_ADAPTER), "local", "--",
           *common_protocol(resistance), "--ipopt-max-iter", str(iterations), "--optional-nlp-periodic-ipopt-hot-start",
           "--initial-guess-diagnostics", "--allow-primal-feasible-common-initial-solution-output",
           "--single-shot", "--cycles-per-window", str(cycles), "--n-windows", str(cycles),
           "--n-threads", str(FHO_THREADS), "--common-initial-solution", str(seed)]
    if prefix is not None:
        cmd.extend(["--full-horizon-prefix-solution", str(prefix)])
    cmd.extend(["--common-initial-solution-output", str(solution), "--output-json", str(result)])
    return cmd


def read_result(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw["results"][0]


def rho_cycles(result: Path, archive: Path) -> int:
    data = read_result(result)
    count = int(data.get("validated_cycles") or data.get("covered_cycles") or 0)
    if not archive.is_file():
        raise RuntimeError(f"RHO did not export a complete/prefix archive: {archive}")
    with np_load(archive) as loaded:
        meta = json.loads(str(loaded["metadata__json"]))
        archive_cycles = int(meta.get("physical_cycles") or meta.get("producer_requested_cycles") or 0)
    if archive_cycles < 1:
        raise RuntimeError(f"RHO archive has no physical cycle count: {archive}")
    return min(count, archive_cycles)


def np_load(path: Path):
    import numpy as np
    return np.load(path, allow_pickle=False)


def prepare_manifests(out: Path, start: int, resistance: float) -> None:
    folder = out / "prepared"
    folder.mkdir(parents=True, exist_ok=True)
    for delta in (1, 2, 3):
        atomic_json(folder / f"fho-{start + delta:04d}-launch-contract.json", {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "prepared_contract_only",
            "resistance_nm": resistance,
            "horizon_cycles": start + delta,
            "weights_delt_ant_delt_post_biceps_triceps": WEIGHTS[resistance],
            "formulation": "prescribed isokinetic, per-cycle work constraint",
            "ding_local": "exact SX local condensation; reconstruct/audit original variables",
            "solver": "IPOPT/MA57, Radau-5",
            "threads": FHO_THREADS,
            "omp_blas_threads": 1,
            "note": "No CasADi graph is cached across processes; each attempt builds its own graph.",
        })


def run_initial_fho(resistance: float, out: Path, rho_archive: Path, cycles: int, helper: Any) -> tuple[Path, Path]:
    work = out / "initial-fho"
    work.mkdir(parents=True, exist_ok=True)
    seed = work / "rho-bo-rebased-seed.npz"
    result = work / f"fho-{cycles:04d}-result.json"
    solution = work / f"fho-{cycles:04d}-solution.npz"
    run_logged([str(PY), str(ROOT / "scripts/rebase_rho_energy_for_fho.py"), "--input", str(rho_archive),
                "--output", str(seed), "--cycles", str(cycles), "--work-per-cycle-j", f"{resistance * 2 * 3.141592653589793:.15g}"],
               work / "rebase.log", out)
    atomic_json(work / "status.json", {"stage": "initial_fho_running", "cycles": cycles,
                                      "started_at_utc": datetime.now(timezone.utc).isoformat()})
    prepare_manifests(out, cycles, resistance)
    run_logged(fho_command(resistance, out, seed, solution, result, cycles, iterations=4000),
               work / "driver.log", out)
    if not solution.is_file() or not result.is_file():
        raise RuntimeError(f"Initial FHO_{cycles} did not export both solution and result.")
    audit = helper.audit_summary(result, RELATIVE_ENERGY_LIMIT)
    atomic_json(work / "status.json", {"stage": "initial_fho_finished", "cycles": cycles, **audit})
    if not audit["operational_continuation_accepted"]:
        raise RuntimeError(f"Initial FHO_{cycles} failed the operational certificate; see {result}")
    return solution, result


def run_rho_cycle(resistance: float, out: Path, seed: Path, cycle: int, work: Path) -> tuple[Path, Path, bool]:
    work.mkdir(parents=True, exist_ok=True)
    seed_copy = work / "terminal-seed.npz"
    if seed.resolve() != seed_copy.resolve():
        shutil.copy2(seed, seed_copy)
    candidate = work / "solution.npz"
    result = work / "result.json"
    cmd = ["taskset", "-c", CPU_SETS[resistance], str(PY), str(LOCAL_ADAPTER), "local", "--",
           *common_protocol(resistance), "--ipopt-max-iter", "1200", "--single-shot",
           "--cycles-per-window", "1", "--n-windows", "1", "--n-threads", "1",
           "--allow-primal-feasible-common-initial-solution-output", "--common-initial-solution", str(seed_copy),
           "--common-initial-solution-output", str(candidate), "--output-json", str(result)]
    ok = run_logged(cmd, work / "driver.log", out, allow_failure=True)
    return candidate, result, ok


def run_worker(resistance: float, root: Path, helper: Any, runner: Any, *, resume_existing: bool = False) -> dict[str, Any]:
    out = root / f"{resistance:.3f}".replace(".", "p")
    out.mkdir(parents=True, exist_ok=resume_existing)
    report: dict[str, Any] = {"resistance_nm": resistance, "weights": WEIGHTS[resistance],
                              "cpu_set": CPU_SETS[resistance], "stage": "rho_reconstruction"}
    rho_archive = out / "rho-bo-trajectory.npz"
    rho_result = out / "rho-bo-result.json"
    try:
        if not resume_existing:
            atomic_json(out / "campaign-status.json", report)
            run_logged(rho_full_command(resistance, out), out / "rho-bo-driver.log", out, allow_failure=True)
        elif not rho_archive.is_file() or not rho_result.is_file():
            raise FileNotFoundError(f"Cannot resume without RHO trajectory/result in {out}")
        cycles = rho_cycles(rho_result, rho_archive)
        if cycles < 1:
            raise RuntimeError("No certified RHO-BO cycle to seed FHO.")
        report.update({"stage": "rho_reconstructed", "rho_certified_cycles": cycles,
                       "rho_archive": str(rho_archive)})
        atomic_json(out / "campaign-status.json", report)
        if resume_existing:
            options = [(cycles, out / "initial-fho" / f"fho-{cycles:04d}-solution.npz",
                        out / "initial-fho" / f"fho-{cycles:04d}-result.json")]
            for result_path in out.glob("fho-*-result.json"):
                try:
                    n = int(result_path.name.split("-")[1])
                    options.append((n, out / f"fho-{n:04d}-solution.npz", result_path))
                except (ValueError, IndexError):
                    continue
            accepted = []
            for n, solution_path, result_path in options:
                if solution_path.is_file() and result_path.is_file():
                    audit = helper.audit_summary(result_path, RELATIVE_ENERGY_LIMIT)
                    if audit["operational_continuation_accepted"]:
                        accepted.append((n, solution_path, result_path))
            if not accepted:
                raise RuntimeError(f"No accepted existing FHO prefix to resume in {out}")
            current, current_solution, current_result = max(accepted, key=lambda item: item[0])
        else:
            current_solution, current_result = run_initial_fho(resistance, out, rho_archive, cycles, helper)
            current = cycles
        report.update({"stage": "fho_prefix_accepted", "fho_certified_cycles": current,
                       "fho_result": str(current_result), "resumed": resume_existing})
        atomic_json(out / "campaign-status.json", report)
        while current < MAX_RHO_CYCLES:
            advanced = False
            for increment in (3, 2, 1):
                base = out / "attempts" / f"from-{current:04d}-plus{increment}"
                attempt = base
                retry = 0
                while attempt.exists():
                    retry += 1
                    attempt = base.with_name(f"{base.name}-retry{retry:02d}")
                attempt.mkdir(parents=True, exist_ok=False)
                prefix = current_solution
                merged = prefix
                appended = 0
                nonoptimal = False
                for cycle in range(current + 1, current + increment + 1):
                    rho_dir = attempt / f"rho-{cycle:04d}"
                    terminal = rho_dir / "terminal-seed.npz"
                    runner.write_fho_terminal_continuation_seed(merged, terminal)
                    helper.reset_local_cycle_integral(terminal, rho_archive, cycle, runner)
                    candidate, result, process_ok = run_rho_cycle(resistance, out, terminal, cycle, rho_dir)
                    if candidate.exists():
                        helper.prescribed_compatible_reference(candidate, candidate)
                    valid = process_ok and candidate.exists() and helper.rho_window_valid(result)
                    if not valid:
                        if not helper.finite_rho_candidate(candidate, result):
                            if helper.finite_solution_archive(terminal):
                                candidate = terminal
                                nonoptimal = True
                            else:
                                break
                        else:
                            nonoptimal = True
                    if helper.terminal_handoff_error_excluding_cycle_integrals(merged, candidate) > 1e-8:
                        break
                    next_merged = attempt / f"combined-through-{cycle:04d}.npz"
                    helper.append_local_rho_to_cumulative_fho(runner, merged, candidate, next_merged,
                                                               nonoptimal_seed=nonoptimal)
                    merged = next_merged
                    appended += 1
                    if nonoptimal:
                        break
                if appended < 1:
                    continue
                target = current + appended
                seed = attempt / f"fho-seed-{target:04d}.npz"
                shutil.copy2(merged, seed)
                result = attempt / f"fho-{target:04d}-result.json"
                solution = attempt / f"fho-{target:04d}-solution.npz"
                prepare_manifests(out, current, resistance)
                cmd = fho_command(resistance, out, seed, solution, result, target,
                                  prefix=prefix, iterations=1600)
                run_logged(cmd, attempt / "fho-driver.log", out, allow_failure=True)
                if not result.exists() or not solution.exists():
                    continue
                audit = helper.audit_summary(result, RELATIVE_ENERGY_LIMIT)
                atomic_json(attempt / "fho-audit.json", {"attempted_cycles": target, **audit})
                if not audit["operational_continuation_accepted"]:
                    continue
                for src, dst in ((solution, out / f"fho-{target:04d}-solution.npz"),
                                 (result, out / f"fho-{target:04d}-result.json")):
                    tmp = dst.with_name(dst.name + ".tmp")
                    shutil.copy2(src, tmp)
                    tmp.replace(dst)
                current_solution = out / f"fho-{target:04d}-solution.npz"
                current_result = out / f"fho-{target:04d}-result.json"
                current = target
                report.update({"stage": "fho_advanced", "fho_certified_cycles": current,
                               "fho_result": str(current_result), "nonoptimal_rho_seed": nonoptimal})
                atomic_json(out / "campaign-status.json", report)
                advanced = True
                break
            if not advanced:
                report.update({"stage": "stopped_no_accepted_extension", "fho_certified_cycles": current})
                atomic_json(out / "campaign-status.json", report)
                break
        return report
    except Exception as exc:
        report.update({"stage": "error", "error": f"{type(exc).__name__}: {exc}"})
        atomic_json(out / "campaign-status.json", report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resistance", type=float, action="append", choices=RESISTANCES,
                        help="Run only the selected resistance(s); may be repeated.")
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    selected = tuple(args.resistance) if args.resistance else RESISTANCES
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=args.resume_existing)
    helper = load_module("fho_operational_helper", ROOT / "scripts/run_isokinetic_fho_operational_continuation.py")
    runner = helper.load_runner()
    records = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(selected))) as pool:
        futures = {pool.submit(run_worker, r, root, helper, runner,
                               resume_existing=args.resume_existing): r for r in selected}
        for future in as_completed(futures):
            records.append(future.result())
            atomic_json(root / "summary.json", {"updated_at_utc": datetime.now(timezone.utc).isoformat(),
                                                "campaigns": records})
    atomic_json(root / "summary.json", {"updated_at_utc": datetime.now(timezone.utc).isoformat(),
                                        "campaigns": records})


if __name__ == "__main__":
    main()
