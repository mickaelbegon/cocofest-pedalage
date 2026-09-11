"""Build, inspect and gate a sequential constant-resistance comparison.

This module never starts a solver. Reviewer-supplied certificates are explicit
inputs; a process exit code is not a physical certificate. Commands are argv
lists, never shell snippets. No arbitrary benchmark overrides are accepted.
"""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ORDER = ("baseline_rho", "rho_pace", "baseline_fho")
INITIAL_GATES = ("seed_physically_certified", "same_initial_state_and_model_parameters",
                 "dynamic_constant_resistance_verified", "validation_thresholds_declared")
RESULT_GATES = ("same_campaign_task_verified", "physical_prefix_replayed",
                "pw_force_and_state_bounds_checked", "solver_status_and_residuals_audited",
                "termination_classification_reviewed")
PACE_GATES = ("pace_objective_update_receipts_verified", "pace_journal_matches_result")


def artifact(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected file: {path}")
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def build_manifest(*, campaign_id, resistance_nm, cycles, seed, reduced_profile,
                   pace_config, output_directory, python=sys.executable,
                   linear_solver="ma57", hsl_library=None, max_iterations=2000, threads=1):
    """Declare three arms without launching them or certifying their inputs."""
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ValueError("campaign_id must be nonempty")
    if isinstance(resistance_nm, bool) or not math.isfinite(resistance_nm) or resistance_nm <= 0:
        raise ValueError("resistance_nm must be explicit, finite and positive")
    for value, label, maximum in ((cycles, "cycles", 100),
                                  (max_iterations, "max_iterations", None),
                                  (threads, "threads", None)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1 or (maximum and value > maximum):
            raise ValueError(f"Invalid {label}; cycles are limited to 1–100")
    if linear_solver not in ("ma57", "mumps"):
        raise ValueError("linear_solver must be ma57 or mumps")
    inputs = {"seed": artifact(seed), "reduced_profile": artifact(reduced_profile),
              "pace_config": artifact(pace_config), "python": artifact(python)}
    if linear_solver == "ma57":
        if hsl_library is None:
            raise ValueError("MA57 requires an explicit existing --hsl-library file")
        inputs["hsl_library"] = artifact(hsl_library)
    elif hsl_library is not None:
        raise ValueError("--hsl-library is only valid for an MA57 campaign")
    config = json.loads(Path(inputs["pace_config"]["path"]).read_text())
    if not isinstance(config, dict) or not config.get("initial_weight_basis"):
        raise ValueError("PACE config needs an explicit initial_weight_basis")
    policy = config.get("policy", {})
    if not isinstance(policy, dict):
        raise ValueError("PACE policy must be an object")
    limit = policy.get("max_cycles", 100)
    if isinstance(limit, bool) or not isinstance(limit, int) or not cycles <= limit <= 100:
        raise ValueError("Campaign cycles exceed declared PACE limit")
    out = Path(output_directory).resolve()
    executable = inputs["python"]["path"]
    common = ["--solvers", "ipopt", "--objective", "fatigue", "--objective-shape", "quadratic",
              "--formulation", "dynamic", "--mechanical-formulation", "reduced",
              "--signed-crank-torque", str(float(resistance_nm)),
              "--common-initial-solution", inputs["seed"]["path"],
              "--reduced-cycling-profile", inputs["reduced_profile"]["path"],
              "--ipopt-disable-standard-warmup", "--adopt-common-initial-solution-warmup-cycles",
              "--ipopt-profile", "periodic_collocation", "--ipopt-collocation-degree", "5",
              "--ipopt-collocation-method", "radau", "--stimulations-per-cycle", "30",
              "--ipopt-use-sx", "--ipopt-linear-solver", linear_solver,
              "--ipopt-max-iter", str(max_iterations), "--n-threads", str(threads),
              "--max-consecutive-failing", "1"]
    environment = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg"}
    if linear_solver == "ma57":
        library = inputs["hsl_library"]["path"]
        common += ["--ipopt-hsl-library", library]
        environment["IPOPT_HSL_LIBRARY"] = library
    benchmark = "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"
    launcher = "scripts/run_rho_pace_benchmark.py"
    arms = []
    for index, name in enumerate(ORDER):
        directory = out / name
        result_path = str(directory / "result.json")
        if Path(result_path).exists() or (directory / "pace.jsonl").exists():
            raise ValueError(f"Refusing an output that already contains results: {directory}")
        single = name == "baseline_fho"
        arguments = common + ["--cycles-per-window", str(cycles if single else 1),
                              "--n-windows", str(cycles), "--output-json", result_path]
        arguments += ["--single-shot"] if single else ["--compact-rho-output"]
        prefix = [executable, str(ROOT / benchmark)]
        journal = None
        if name == "rho_pace":
            journal = str(directory / "pace.jsonl")
            prefix = [executable, str(ROOT / launcher), "--pace-config", inputs["pace_config"]["path"],
                      "--pace-journal", journal, "--"]
        arms.append({"id": name, "after": None if index == 0 else ORDER[index - 1],
                     "argv": prefix + arguments, "result_path": result_path,
                     "pace_journal_path": journal, "output_directory": str(directory),
                     "horizon_cycles": cycles if single else 1,
                     "requested_executed_cycles": cycles,
                     "objective": "adaptive_weighted_relative_capacity_loss_squared" if name == "rho_pace"
                                  else "uniform_relative_capacity_loss_squared",
                     "required_review_gates": list(RESULT_GATES + (PACE_GATES if name == "rho_pace" else ()))})
    code_paths = [benchmark, launcher, "cocofest/optimization/rho_pace.py",
                  "examples/fes_multibody/cycling/cycling_pulse_width_mhe.py",
                  "examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py",
                  "scripts/build_resistance_comparison_campaign.py"]
    manifest = {"schema": "dynamic-resistance-comparison-v1", "campaign_id": campaign_id,
                "working_directory": str(ROOT), "resistance_nm": float(resistance_nm),
                "resistance_policy": "one_constant_positive_signed_torque_for_all_arms",
                "formulation": "dynamic", "mechanical_formulation": "reduced", "cycles": cycles,
                "inputs": inputs, "code": [artifact(ROOT / path) for path in code_paths],
                "initial_weight_basis": config["initial_weight_basis"],
                "initial_required_review_gates": list(INITIAL_GATES), "arms": arms,
                "environment": environment,
                "comparison": {"common_metrics": ["certified_executed_cycles", "tracking_error",
                                "unweighted_capacity_loss", "pw_force_bounds", "solver_latency"],
                               "fho_used_for_pace_calibration": False,
                               "finite_fho_proves_global_endurance_optimum": False,
                               "numerical_failure_is_fatigue": False},
                "execution_started": False}
    manifest["manifest_sha256"] = _digest(manifest)
    return manifest


def verify_manifest(manifest):
    """Reject changed commands, resistance, inputs or implementation files."""
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _digest(body):
        raise ValueError("Manifest changed; declare a new campaign instead")
    for expected in [*manifest["inputs"].values(), *manifest["code"]]:
        if artifact(expected["path"]) != expected:
            raise ValueError(f"Campaign input or code changed: {expected['path']}")


def _review(review, gates, *, arm=None, cycles=None):
    if not isinstance(review, dict):
        raise ValueError("A reviewed certificate is required before proceeding")
    if any(review.get("gates", {}).get(gate) is not True for gate in gates):
        raise ValueError("Required certification gates are not all passed")
    evidence = review.get("evidence", [])
    if not evidence or any(artifact(item["path"]) != item for item in evidence):
        raise ValueError("Review requires unchanged evidence artifacts")
    if arm is not None:
        if review.get("outcome") not in ("completed", "numerical_or_unresolved_stop"):
            raise ValueError("Never infer fatigue from an optimization failure")
        count = review.get("certified_executed_cycles")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= cycles:
            raise ValueError("A nonempty certified physical prefix is required")
        if review["outcome"] == "completed" and count != cycles:
            raise ValueError("Completion requires all requested cycles certified")
        result = artifact(arm["result_path"])
        if result not in evidence:
            raise ValueError("Review must include this arm's benchmark result")
        if arm["pace_journal_path"] and artifact(arm["pace_journal_path"]) not in evidence:
            raise ValueError("PACE review must include its matching journal")


def next_command(manifest, reviews):
    """Return the next argv only after predecessor review, or None when done.

    This verifies artifacts and explicit review receipts, not physical truth.
    A qualified audit must populate the receipts; the builder does not infer
    them from solver status, process exit, or a requested cycle count.
    """
    verify_manifest(manifest)
    if reviews.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Reviews must identify this exact campaign manifest")
    _review(reviews.get("initial"), INITIAL_GATES)
    for arm in manifest["arms"]:
        review = reviews.get(arm["id"])
        if review is None:
            if Path(arm["result_path"]).exists() or (arm["pace_journal_path"] and Path(arm["pace_journal_path"]).exists()):
                raise ValueError("Existing arm artifacts need review; never overwrite/relaunch them")
            return list(arm["argv"])
        _review(review, arm["required_review_gates"], arm=arm, cycles=manifest["cycles"])
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--resistance-nm", required=True, type=float)
    parser.add_argument("--cycles", required=True, type=int)
    parser.add_argument("--seed", required=True, type=Path)
    parser.add_argument("--reduced-profile", required=True, type=Path)
    parser.add_argument("--pace-config", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--linear-solver", choices=("ma57", "mumps"), default="ma57")
    parser.add_argument("--hsl-library", type=Path,
                        help="Required existing HSL library for MA57; path and SHA-256 are sealed")
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=1)
    args = vars(parser.parse_args(argv))
    target = args.pop("manifest")
    manifest = build_manifest(**args)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents destroying a previously declared campaign.
    with target.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Manifest written: {target}; no solver started")


if __name__ == "__main__":
    main()
