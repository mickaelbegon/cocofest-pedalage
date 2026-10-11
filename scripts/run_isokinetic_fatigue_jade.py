#!/usr/bin/env python3
"""Two-generation, parallel JADE search for fatigue-only RHO weights."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

MUSCLES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
INCUMBENT = (0.10358741142405141, 0.1974466952499519, 0.20604694950609628, 0.9340662537753645)


def _score(path: Path) -> tuple[float, int, float | None, dict[str, Any]]:
    if not path.is_file():
        return 0.0, 0, None, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))["results"][0]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return 0.0, 0, None, {}
    cycles = int(payload.get("bo_screen_cycles_unvalidated", payload.get("covered_cycles")) or 0)
    reserve = payload.get("bo_screen_terminal_min_capacity_ratio_unvalidated")
    try:
        reserve = float(reserve)
        if not np.isfinite(reserve):
            reserve = None
    except (ValueError, TypeError):
        reserve = None
    # Same continuous tie-break screen as the recent campaigns. The OCP itself
    # remains fatigue-only; this is not a physiological certification score.
    value = cycles + (0.999 * float(np.clip(reserve, 0.0, 1.0)) if reserve is not None else 0.0)
    return value, cycles, reserve, payload


def _best_completed(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [r for r in rows if r.get("returncode") == 0 and not r.get("timed_out") and r.get("result_exists")]
    return max(eligible, key=lambda r: r.get("score_unvalidated", 0.0), default=None)


def _run_candidate(generation: int, index: int, weights: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    directory = args.output / f"generation_{generation:02d}" / f"candidate_{index:03d}"
    directory.mkdir(parents=True, exist_ok=False)
    result = directory / "result.json"
    cmd = [
        args.python, "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py",
        "--solvers", "ipopt", "--objective", "fatigue", "--objective-shape", "quadratic",
        "--parametric-fatigue-weights", "--fatigue-weight-values", *(f"{v:.12g}" for v in weights),
        "--formulation", "isokinetic", "--mechanical-formulation", "reduced",
        "--isokinetic-kinematics", "prescribed", "--isokinetic-omega", str(-2 * np.pi),
        "--energy-equivalent-torque", str(args.resistance), "--load-torque-min", "-4", "--load-torque-max", "4",
        "--stimulations-per-cycle", "30", "--ipopt-profile", "periodic_collocation",
        "--ipopt-collocation-method", "radau", "--ipopt-collocation-degree", "5", "--ipopt-use-sx",
        "--ipopt-linear-solver", "ma57", "--warmup-ipopt-linear-solver", "ma57",
        "--ipopt-hsl-library", args.hsl_library, "--ipopt-max-iter", "4000",
        "--nlp-tolerance", "1e-8", "--primal-feasibility-threshold", "1e-5",
        "--ipopt-enforce-start-constraints", "--cycles-per-window", "1", "--n-windows", str(args.max_cycles),
        "--n-threads", "1", "--compact-rho-output", "--retry-failed-rho-without-advance",
        "--common-initial-solution", str(args.shared_seed), "--output-json", str(result),
    ]
    cmd = [args.python, "scripts/benchmark_ding_radau5_local_ab.py", "local", "--", *cmd[2:]]
    env = {**os.environ, "PYTHONPATH": str(args.root), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "COCOFEST_DEFERRED_RHO_SUMMARY": "1",
           "DING_ONLINE_JOURNAL": "1", "DING_LOCAL_SAVE_INPUTS": "0"}
    start = time.monotonic()
    with (directory / "driver.log").open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(["taskset", "-c", args.cpu_set, *cmd], cwd=args.root,
                                       stdout=log, stderr=subprocess.STDOUT, env=env,
                                       timeout=args.candidate_timeout_s)
            rc, timed_out = completed.returncode, False
        except subprocess.TimeoutExpired:
            rc, timed_out = 124, True
            log.write("\nJADE_CANDIDATE_TIMEOUT\n")
    elapsed = time.monotonic() - start
    value, cycles, reserve, payload = _score(result)
    row = {"generation": generation, "candidate": index, "weights": dict(zip(MUSCLES, map(float, weights), strict=True)),
           "score_unvalidated": value, "screen_cycles_unvalidated": cycles,
           "terminal_min_capacity_unvalidated": reserve, "returncode": rc, "timed_out": timed_out,
           "wall_time_s": elapsed, "wall_time_per_screen_cycle_s": elapsed / cycles if cycles else None,
           "result_exists": result.is_file(), "result_path": str(result),
           "solver_success": payload.get("solver_success"), "first_failed_rho": payload.get("first_failed_rho")}
    (directory / "observation.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row


def _latin_hypercube(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    x = np.empty((n, d))
    for j in range(d):
        x[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return x


def _jade_trials(pop: np.ndarray, fitness: np.ndarray, archive: list[np.ndarray], rng: np.random.Generator,
                 mu_f: float, mu_cr: float, p: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = pop.shape
    order = np.argsort(fitness)[::-1]
    pbest_count = max(2, int(np.ceil(p * n)))
    trials = np.empty_like(pop)
    used_f, used_cr = np.empty(n), np.empty(n)
    pool = np.vstack((pop, np.asarray(archive))) if archive else pop
    for i in range(n):
        f = -1.0
        while f <= 0:
            f = mu_f + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
        f = min(f, 1.0)
        cr = float(np.clip(rng.normal(mu_cr, 0.1), 0.0, 1.0))
        pbest = pop[int(rng.choice(order[:pbest_count]))]
        r1_choices = [j for j in range(n) if j != i]
        r1 = pop[int(rng.choice(r1_choices))]
        r2_pool = np.vstack((pool, pop[i]))
        r2_choices = np.arange(len(r2_pool))
        r2_choices = r2_choices[~np.all(np.isclose(r2_pool, pop[i]), axis=1)]
        r2 = r2_pool[int(rng.choice(r2_choices))]
        mutant = pop[i] + f * (pbest - pop[i]) + f * (r1 - r2)
        # Reflection keeps mutation at the boundaries without accumulating
        # candidates on 0/1 through clipping.
        mutant = 1.0 - np.abs(np.mod(mutant, 2.0) - 1.0)
        mask = rng.random(d) < cr
        mask[int(rng.integers(d))] = True
        trials[i] = np.where(mask, mutant, pop[i])
        used_f[i], used_cr[i] = f, cr
    return trials, used_f, used_cr


def _prepare_seed(args: argparse.Namespace) -> dict[str, Any]:
    seed_dir = args.output / "shared-rho-seed"
    seed_dir.mkdir()
    seed, result = seed_dir / "one-cycle-unit-fatigue.npz", seed_dir / "result.json"
    command = [args.python, "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py",
        "--solvers", "ipopt", "--objective", "fatigue", "--objective-shape", "quadratic",
        "--parametric-fatigue-weights", "--fatigue-weight-values", "1", "1", "1", "1",
        "--formulation", "isokinetic", "--mechanical-formulation", "reduced",
        "--isokinetic-kinematics", "prescribed", "--isokinetic-omega", str(-2 * np.pi),
        "--energy-equivalent-torque", str(args.resistance), "--load-torque-min", "-4", "--load-torque-max", "4",
        "--stimulations-per-cycle", "30", "--ipopt-profile", "periodic_collocation",
        "--ipopt-collocation-method", "radau", "--ipopt-collocation-degree", "5", "--ipopt-use-sx",
        "--ipopt-linear-solver", "ma57", "--warmup-ipopt-linear-solver", "ma57", "--ipopt-hsl-library", args.hsl_library,
        "--ipopt-max-iter", "4000", "--nlp-tolerance", "1e-8", "--primal-feasibility-threshold", "1e-5",
        "--ipopt-enforce-start-constraints", "--cycles-per-window", "1", "--n-windows", "1", "--n-threads", "1",
        "--compact-rho-output", "--output-json", str(result), "--common-initial-solution-output", str(seed),
        "--allow-primal-feasible-common-initial-solution-output"]
    command = [args.python, "scripts/benchmark_ding_radau5_local_ab.py", "local", "--", *command[2:]]
    env = {**os.environ, "PYTHONPATH": str(args.root), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "DING_LOCAL_SAVE_INPUTS": "0"}
    start = time.monotonic()
    with (seed_dir / "driver.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(["taskset", "-c", args.cpu_set, *command], cwd=args.root,
                                   stdout=log, stderr=subprocess.STDOUT, env=env, timeout=args.seed_timeout_s)
    if completed.returncode or not seed.is_file() or not result.is_file():
        raise RuntimeError(f"Shared one-cycle seed failed; see {seed_dir}")
    payload = json.loads(result.read_text(encoding="utf-8"))["results"][0]
    if int(payload.get("covered_cycles") or 0) < 1:
        raise RuntimeError("Shared unit-fatigue RHO seed did not solve one cycle")
    args.shared_seed = seed
    return {"path": str(seed), "covered_cycles": 1, "wall_time_s": time.monotonic() - start,
            "method": "single one-cycle unit-weight RHO seed shared by all candidates and final validation"}


def _validate_best(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    directory = args.output / "final-validation"
    directory.mkdir()
    weights = tuple(float(row["weights"][name]) for name in MUSCLES)
    result = directory / "result.json"
    command = [args.python, "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py",
        "--solvers", "ipopt", "--objective", "fatigue", "--objective-shape", "quadratic",
        "--parametric-fatigue-weights", "--fatigue-weight-values", *(f"{v:.12g}" for v in weights),
        "--formulation", "isokinetic", "--mechanical-formulation", "reduced",
        "--isokinetic-kinematics", "prescribed", "--isokinetic-omega", str(-2 * np.pi),
        "--energy-equivalent-torque", str(args.resistance), "--load-torque-min", "-4", "--load-torque-max", "4",
        "--stimulations-per-cycle", "30", "--ipopt-profile", "periodic_collocation",
        "--ipopt-collocation-method", "radau", "--ipopt-collocation-degree", "5", "--ipopt-use-sx",
        "--ipopt-linear-solver", "ma57", "--warmup-ipopt-linear-solver", "ma57",
        "--ipopt-hsl-library", args.hsl_library, "--ipopt-max-iter", "4000", "--nlp-tolerance", "1e-8",
        "--primal-feasibility-threshold", "1e-5", "--ipopt-enforce-start-constraints",
        "--cycles-per-window", "1", "--n-windows", str(args.max_cycles), "--n-threads", "1",
        "--compact-rho-output", "--retry-failed-rho-without-advance", "--common-initial-solution",
        str(args.shared_seed), "--output-json", str(result)]
    command = [args.python, "scripts/benchmark_ding_radau5_local_ab.py", "local", "--", *command[2:]]
    env = {**os.environ, "PYTHONPATH": str(args.root), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "DING_LOCAL_SAVE_INPUTS": "0",
           "COCOFEST_COMPACT_FINAL_SUMMARY": "1"}
    start = time.monotonic()
    with (directory / "driver.log").open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(["taskset", "-c", args.cpu_set, *command], cwd=args.root,
                stdout=log, stderr=subprocess.STDOUT, env=env, timeout=args.candidate_timeout_s)
            rc, timed_out = completed.returncode, False
        except subprocess.TimeoutExpired:
            rc, timed_out = 124, True
    value, screen_cycles, reserve, payload = _score(result)
    validation = {"status": "timed_out" if timed_out else ("completed" if rc == 0 and payload else "failed"),
        "returncode": rc, "timed_out": timed_out, "wall_time_s": time.monotonic() - start,
        "weights": row["weights"], "screen_cycles_unvalidated": screen_cycles,
        "validated_cycles": payload.get("validated_cycles"),
        "physiologically_validated_cycles": payload.get("physiologically_validated_cycles"),
        "covered_cycles": payload.get("covered_cycles"), "solver_success": payload.get("solver_success"),
        "physical_success": payload.get("physical_success"), "result_path": str(result)}
    (directory / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", required=True); parser.add_argument("--hsl-library", required=True)
    parser.add_argument("--cpu-set", default="0-29"); parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--resistance", type=float, default=0.73); parser.add_argument("--max-cycles", type=int, default=1500)
    parser.add_argument("--generations", type=int, default=2, help="Initial population plus JADE generations")
    parser.add_argument("--population", type=int, default=30); parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--candidate-timeout-s", type=float, default=2400); parser.add_argument("--seed-timeout-s", type=float, default=180)
    parser.add_argument("--mu-f", type=float, default=0.5); parser.add_argument("--mu-cr", type=float, default=0.5)
    parser.add_argument("--p-best", type=float, default=0.2)
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    summary_path = args.output / "summary.json"
    summary: dict[str, Any] = {"status": "preparing_shared_rho_seed", "completed": 0, "best": None}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    seed_data = _prepare_seed(args)
    campaign = {"algorithm": "JADE adaptive differential evolution", "resistance_nm": args.resistance,
        "objective": "fatigue only; no PW/control, balance or reserve term in the OCP",
        "screen_score": "unvalidated RHO cycles + 0.999*clipped terminal min capacity ratio; tie-break only",
        "weights_order": MUSCLES, "bounds": [0.0, 1.0], "population_size": args.population,
        "generations_including_initial_population": args.generations, "workers": args.workers,
        "cpu_affinity": args.cpu_set, "one_thread_per_candidate": True, "max_cycles_per_candidate": args.max_cycles,
        "candidate_timeout_s": args.candidate_timeout_s, "ding_local_exact_radau5": True,
        "isokinetic_kinematics": "prescribed", "mechanical_formulation": "reduced",
        "hessian": "CasADi SX, IPOPT MA57", "common_seed": seed_data,
        "initial_population": "29-point Latin hypercube plus re-evaluation of historical 0.73 fatigue-only incumbent",
        "historical_incumbent": dict(zip(MUSCLES, INCUMBENT, strict=True)),
        "jaDE_parameters": {"mu_F": args.mu_f, "mu_CR": args.mu_cr, "p_best_fraction": args.p_best,
            "archive": "replaced parents, random truncation to population size"},
        "final_validation": "one compact exact audit of the best completed candidate after search",
        "started_at_utc": datetime.now(timezone.utc).isoformat()}
    (args.output / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n", encoding="utf-8")
    summary.update(status="candidate_search_running", shared_seed=seed_data, generation=0,
                   population_size=args.population, completed=0)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pop = _latin_hypercube(rng, args.population, 4)
    pop[0] = np.asarray(INCUMBENT)
    archive: list[np.ndarray] = []
    mu_f, mu_cr = args.mu_f, args.mu_cr
    all_rows: list[dict[str, Any]] = []
    fitness = np.full(args.population, -np.inf)
    for generation in range(args.generations):
        if generation:
            pop, factors, crossovers = _jade_trials(pop, fitness, archive, rng, mu_f, mu_cr, args.p_best)
        else:
            factors, crossovers = np.zeros(args.population), np.zeros(args.population)
        rows: list[dict[str, Any]] = [None] * args.population  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_candidate, generation, i, pop[i].copy(), args): i for i in range(args.population)}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    rows[i] = future.result()
                except Exception as error:
                    rows[i] = {"generation": generation, "candidate": i, "weights": dict(zip(MUSCLES, pop[i].tolist())),
                                "score_unvalidated": 0.0, "screen_cycles_unvalidated": 0,
                                "returncode": 125, "error": f"{type(error).__name__}: {error}", "timed_out": False}
                valid = [r for r in rows if r is not None]
                best = _best_completed([*all_rows, *valid])
                summary.update(generation=generation, completed=len(valid) + args.population * generation,
                    current_generation_completed=len(valid), best=best,
                    elapsed_s=time.monotonic() - started)
                summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        new_fitness = np.asarray([r.get("score_unvalidated", 0.0)
                                  if r.get("returncode") == 0 and not r.get("timed_out") and r.get("result_exists")
                                  else -np.inf for r in rows])
        if generation:
            successes_f, successes_cr = [], []
            old_pop, old_fitness = parent_pop, parent_fitness
            for i in range(args.population):
                if new_fitness[i] >= old_fitness[i]:
                    archive.append(old_pop[i].copy())
                    pop[i], fitness[i] = pop[i].copy(), new_fitness[i]
                    successes_f.append(float(factors[i])); successes_cr.append(float(crossovers[i]))
                else:
                    pop[i], fitness[i] = old_pop[i], old_fitness[i]
            if len(archive) > args.population:
                del archive[int(rng.integers(len(archive)))]
            if successes_f:
                sf = np.asarray(successes_f)
                mu_f = float(np.sum(sf * sf) / np.sum(sf))
                mu_cr = float(np.mean(successes_cr))
        else:
            fitness = new_fitness
        all_rows.extend(rows)
        (args.output / f"generation_{generation:02d}.json").write_text(json.dumps({
            "generation": generation, "rows": rows, "scores_unvalidated": new_fitness.tolist(),
            "mu_f_after": mu_f, "mu_cr_after": mu_cr}, indent=2) + "\n", encoding="utf-8")
        summary.update(completed=len(all_rows), best=_best_completed(all_rows),
                       mu_f=mu_f, mu_cr=mu_cr, status="candidate_search_running")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        parent_pop, parent_fitness = pop.copy(), fitness.copy()
    (args.output / "all-observations.json").write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")
    summary.update(status="candidate_search_finished", elapsed_s=time.monotonic() - started,
                   completed=len(all_rows), best=_best_completed(all_rows))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    winner = summary["best"]
    if winner and not winner.get("timed_out") and winner.get("result_exists"):
        summary.update(status="final_validation_running", final_validation_trial=winner["candidate"],
                       final_validation_generation=winner["generation"])
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summary["final_validation"] = _validate_best(winner, args)
    summary.update(status="complete", total_elapsed_s=time.monotonic() - started)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
