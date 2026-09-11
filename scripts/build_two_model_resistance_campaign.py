"""Declare eight configured dynamic solves and a two-worker dependency plan.

No process is launched. Each model follows RHO -> RHO-Physio -> RHO-PACE ->
FHO; the two independent model chains may progress concurrently. The shared
model/weight sources, resistance and CPU assignments are sealed in a manifest.
"""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_resistance_comparison_campaign import (
    INITIAL_GATES, RESULT_GATES, _digest, _review, artifact, build_manifest, verify_manifest,
)

WRAPPER = ROOT / "scripts/run_configured_cycling_benchmark.py"
ARM_CONDITIONS = (("baseline_rho", "rho"), ("rho_physio", "rho-physio"),
                  ("rho_pace", "rho-pace"), ("baseline_fho", "fho"))
MODEL_GATES = (*INITIAL_GATES, "seed_matches_declared_model_variant",
               "weights_match_variant_and_have_no_fho_calibration")
WEIGHT_GATES = ("configured_weights_receipts_verified", "weights_journal_matches_result")
MUSCLE_NAMES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _cpu_allocations(cores_per_job, numeric_threads, cpu_ids):
    cores = _integer(cores_per_job, "cores_per_job")
    numeric = _integer(numeric_threads, "numeric_threads")
    if cores % numeric:
        raise ValueError("cores_per_job must be divisible by numeric_threads")
    available = tuple(sorted(os.sched_getaffinity(0)))
    supplied = available if cpu_ids is None else tuple(cpu_ids)
    if (any(isinstance(v, bool) or not isinstance(v, int) for v in supplied)
            or len(set(supplied)) != len(supplied) or not set(supplied) <= set(available)):
        raise ValueError("cpu_ids must be unique CPU IDs available to this process")
    if len(supplied) < 2 * cores:
        raise ValueError("Two disjoint CPU allocations require at least 2*cores_per_job available IDs")
    return [list(supplied[:cores]), list(supplied[cores:2 * cores])], cores // numeric


def _config_inputs(variant):
    required = {"model_id", "model_config", "seed", "seed_certificate", "reduced_profile", "weights_config"}
    if not isinstance(variant, dict) or set(variant) != required:
        raise ValueError(f"Each variant must declare exactly {sorted(required)}")
    name = variant["model_id"]
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name) is None:
        raise ValueError("model_id must be a safe nonempty identifier")
    model_file = artifact(variant["model_config"])
    weights_file = artifact(variant["weights_config"])
    model = json.loads(Path(model_file["path"]).read_text())
    weights = json.loads(Path(weights_file["path"]).read_text())
    if (not isinstance(model, dict) or model.get("schema_version") != 1
            or not model.get("case_id") or not model.get("provenance")
            or not isinstance(model.get("muscles"), dict)
            or set(model["muscles"]) != set(MUSCLE_NAMES)):
        raise ValueError("model-config needs schema_version=1, case_id, provenance and four named muscles")
    if (not isinstance(weights, dict) or not weights.get("initial_weight_basis")
            or not isinstance(weights.get("initial_weights"), dict)
            or set(weights["initial_weights"]) != set(MUSCLE_NAMES)):
        raise ValueError("weights-config needs explicit positive named weights and provenance")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
           for v in weights["initial_weights"].values()):
        raise ValueError("The configured Physio/PACE adapters require positive finite weights")
    # Resolve the same defaults as the runtime, without building or solving an OCP.
    from cocofest.optimization.configured_cycling_model import resolve_model_config
    return model_file, weights_file, resolve_model_config(model), weights


def _seed_preparation_inputs(variant, model_file, model, resistance_nm):
    from cocofest.optimization.configured_cycling_model import FINGERPRINT_KEY, require_seed_fingerprint

    seed = artifact(variant["seed"])
    profile = artifact(variant["reduced_profile"])
    certificate_file = artifact(variant["seed_certificate"])
    certificate = json.loads(Path(certificate_file["path"]).read_text())
    if (type(certificate.get("schema_version")) is not int or certificate["schema_version"] != 1
            or certificate.get("stage") != "per_model_seed_preparation"
            or certificate.get(FINGERPRINT_KEY) != model[FINGERPRINT_KEY]
            or certificate.get("seed") != seed or certificate.get("model_config") != model_file
            or certificate.get("reduced_profile") != profile
            or certificate.get("resistance_nm") != resistance_nm):
        raise ValueError("Seed preparation certificate must bind this model, seed, profile and resistance")
    require_seed_fingerprint(seed["path"], model[FINGERPRINT_KEY])
    _review(certificate, INITIAL_GATES)
    return {"seed_certificate": certificate_file,
            **{f"seed_preparation_evidence_{index}": item
               for index, item in enumerate(certificate["evidence"])}}, model[FINGERPRINT_KEY]


def build_two_model_manifest(*, campaign_id, resistance_nm, cycles, variants,
                             output_directory, hsl_library, cores_per_job,
                             numeric_threads=1, cpu_ids=None, python=sys.executable,
                             max_iterations=2000, nlp_tolerance=1e-8, taskset=None):
    """Build a sealed two-chain DAG. CPU IDs denote logical OS CPUs.

    Select one CPU ID per physical core explicitly when SMT isolation is
    required. Numeric threads are shared by OMP/OpenBLAS/MKL; benchmark worker
    threads equal cores_per_job/numeric_threads, bounding the declared product.
    Runtime affinity and the environment must both be honored by the executor.
    """
    if not isinstance(variants, (list, tuple)) or len(variants) != 2:
        raise ValueError("Exactly two explicitly configured model variants are required")
    configs = [_config_inputs(variant) for variant in variants]
    if len({v["model_id"] for v in variants}) != 2:
        raise ValueError("The two model IDs must differ")
    if configs[0][2]["muscles"] == configs[1][2]["muscles"]:
        raise ValueError("The two model variants must declare different muscle parameters")
    if len({artifact(variant["reduced_profile"])["sha256"] for variant in variants}) != 1:
        raise ValueError("Ding-variant comparison requires the same reduced mechanical profile")
    allocations, workers = _cpu_allocations(cores_per_job, numeric_threads, cpu_ids)
    taskset_file = artifact(taskset or shutil.which("taskset") or "/usr/bin/taskset")
    wrapper_file = artifact(WRAPPER)
    manifest = None
    all_arms, chains, all_inputs = [], [], {}
    for variant_index, (variant, config) in enumerate(zip(variants, configs, strict=True)):
        model_id = variant["model_id"]
        model_file, weights_file, model, weights = config
        preparation_inputs, fingerprint = _seed_preparation_inputs(
            variant, model_file, model, resistance_nm)
        directory = Path(output_directory).resolve() / model_id
        original = build_manifest(
            campaign_id=campaign_id, resistance_nm=resistance_nm, cycles=cycles,
            seed=variant["seed"], reduced_profile=variant["reduced_profile"],
            pace_config=weights_file["path"], output_directory=directory, python=python,
            linear_solver="ma57", hsl_library=hsl_library,
            max_iterations=max_iterations, nlp_tolerance=nlp_tolerance, threads=workers,
        )
        if manifest is None:
            manifest = deepcopy(original)
        for key, value in original["inputs"].items():
            all_inputs[f"{model_id}:{key}"] = value
        all_inputs[f"{model_id}:model_config"] = model_file
        all_inputs[f"{model_id}:weights_config"] = weights_file
        all_inputs.update({f"{model_id}:{key}": value for key, value in preparation_inputs.items()})
        chain_ids = []
        for arm_index, (arm_id, condition) in enumerate(ARM_CONDITIONS):
            job_id = f"{model_id}/{arm_id}"
            target = directory / arm_id
            result = str(target / "result.json")
            configuration_audit = str(target / "result.configuration.json")
            journal = str(target / "weights.jsonl") if condition in ("rho-physio", "rho-pace") else None
            if Path(result).exists() or Path(configuration_audit).exists() or (journal and Path(journal).exists()):
                raise ValueError(f"Existing results must not be overwritten: {target}")
            template = original["arms"][2 if condition == "fho" else 0]
            benchmark_args = list(template["argv"][2:])
            benchmark_args[benchmark_args.index("--output-json") + 1] = result
            wrapper_args = [original["inputs"]["python"]["path"], wrapper_file["path"],
                            "--model-config", model_file["path"], "--condition", condition,
                            "--configuration-audit", configuration_audit]
            if journal:
                wrapper_args += ["--weights-config", weights_file["path"], "--weights-journal", journal]
            affinity = ",".join(map(str, allocations[variant_index]))
            argv = [taskset_file["path"], "--cpu-list", affinity, *wrapper_args, "--", *benchmark_args]
            gates = [*RESULT_GATES, "configured_model_receipt_verified"]
            if journal:
                gates.extend(WEIGHT_GATES)
                gates.append("static_weights_unchanged" if condition == "rho-physio"
                             else "adaptive_weight_updates_audited")
            all_arms.append({
                "id": job_id, "arm_id": arm_id, "condition": condition, "model_id": model_id,
                "after": chain_ids[-1] if chain_ids else None,
                "argv": argv, "result_path": result, "weights_journal_path": journal,
                "configuration_audit_path": configuration_audit,
                "output_directory": str(target), "cpu_ids": allocations[variant_index],
                "required_review_gates": gates, "horizon_cycles": cycles if condition == "fho" else 1,
                "requested_executed_cycles": cycles,
                "model_config_sha256": model_file["sha256"],
                "muscle_parameter_fingerprint": fingerprint,
                "weights_config_sha256": weights_file["sha256"] if journal else None,
                "weights_applied": bool(journal),
                "weights_policy": "fixed_initial" if condition == "rho-physio" else
                                  "adaptive_same_initial" if condition == "rho-pace" else "uniform",
            })
            chain_ids.append(job_id)
        chains.append({"model_id": model_id, "model_case_id": model["case_id"],
                       "muscle_parameter_fingerprint": fingerprint,
                       "seed_preparation_is_outside_comparison": True,
                       "model_provenance": model["provenance"], "initial_weight_basis": weights["initial_weight_basis"],
                       "jobs": chain_ids, "cpu_ids": allocations[variant_index]})
    manifest.pop("manifest_sha256", None)
    manifest.pop("initial_weight_basis", None)
    manifest.update({
        "schema": "two-model-dynamic-resistance-comparison-v1", "inputs": all_inputs,
        "arms": all_arms, "model_chains": chains, "initial_required_review_gates": list(MODEL_GATES),
        "schedule": {"max_concurrent_jobs": 2, "cores_per_job": cores_per_job,
                     "cpu_id_semantics": "logical_OS_CPU_identifiers; choose topology explicitly",
                     "numeric_threads": numeric_threads, "benchmark_worker_threads": workers,
                     "declared_threads_per_job_bound": workers * numeric_threads,
                     "launch_waves_if_all_predecessors_certified":
                         [[chain["jobs"][index] for chain in chains] for index in range(4)],
                     "wait_for_entire_wave": False,
                     "rule": "one eligible job per model; never advance an unreviewed predecessor"},
    })
    manifest["inputs"]["taskset"] = taskset_file
    manifest["code"] += [wrapper_file, artifact(__file__),
                         *[artifact(ROOT / path) for path in (
                             "cocofest/optimization/configured_cycling_model.py",
                             "cocofest/custom_objectives.py",
                             "cocofest/models/ding2003/ding2003.py",
                             "cocofest/models/ding2007/ding2007.py",
                             "cocofest/models/ding2007/ding2007_with_fatigue.py")]]
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        manifest["environment"][variable] = str(numeric_threads)
    manifest["environment"]["OMP_MAX_ACTIVE_LEVELS"] = "1"
    for arm in manifest["arms"]:
        arm["environment"] = dict(manifest["environment"])
        arm["working_directory"] = manifest["working_directory"]
    manifest["manifest_sha256"] = _digest(manifest)
    return manifest


def next_jobs(manifest, reviews, *, running_job_ids=()):
    """Return at most the currently free CPU slots, without running anything.

    The caller must report all running campaign jobs. A planner cannot enforce
    a global process cap against unreported or manually started processes.
    Missing initial reviews leave only that model blocked; failed supplied
    review gates fail closed. Result files without a review require attention.
    """
    verify_manifest(manifest)
    if reviews.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Reviews must identify this exact campaign manifest")
    jobs = {arm["id"]: arm for arm in manifest["arms"]}
    running = tuple(running_job_ids)
    if len(running) > 2 or len(set(running)) != len(running) or not set(running) <= set(jobs):
        raise ValueError("At most two unique known jobs may run")
    if len({jobs[name]["model_id"] for name in running}) != len(running):
        raise ValueError("Two arms of the same model must never run concurrently")
    candidates = []
    recognized_running = set()
    model_reviews = reviews.get("models", {})
    for chain in manifest["model_chains"]:
        receipts = model_reviews.get(chain["model_id"], {})
        if "initial" not in receipts:
            continue
        _review(receipts["initial"], MODEL_GATES)
        for name in chain["jobs"]:
            arm = jobs[name]
            if name in running:
                if arm["arm_id"] in receipts:
                    raise ValueError("A reviewed completed job cannot still be marked running")
                recognized_running.add(name)
                break
            review = receipts.get(arm["arm_id"])
            if review is None:
                if Path(arm["result_path"]).exists() or Path(arm["configuration_audit_path"]).exists() or (arm["weights_journal_path"]
                                                       and Path(arm["weights_journal_path"]).exists()):
                    raise ValueError(f"Existing unreviewed result requires attention: {name}")
                candidates.append(deepcopy(arm))
                break
            review_arm = {"result_path": arm["result_path"],
                          "pace_journal_path": arm["weights_journal_path"]}
            _review(review, arm["required_review_gates"], arm=review_arm, cycles=manifest["cycles"])
            if artifact(arm["configuration_audit_path"]) not in review["evidence"]:
                raise ValueError("Review must include this arm's configured-model audit")
    if recognized_running != set(running):
        raise ValueError("A running job lacks its certified predecessor or model-initial review")
    return candidates[:2 - len(running)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--resistance-nm", required=True, type=float)
    parser.add_argument("--cycles", required=True, type=int)
    parser.add_argument("--variants", required=True, type=Path,
                        help="JSON list of exactly two explicit model/seed/profile/weight descriptors")
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--hsl-library", required=True, type=Path)
    parser.add_argument("--cores-per-job", required=True, type=int)
    parser.add_argument("--numeric-threads", type=int, default=1)
    parser.add_argument("--cpu-ids", type=lambda text: tuple(int(item) for item in text.split(",")))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--nlp-tolerance", type=float, default=1e-8)
    args = vars(parser.parse_args(argv))
    destination, descriptor = args.pop("manifest"), args.pop("variants").expanduser().resolve()
    if not descriptor.is_file():
        parser.error(
            f"Variants descriptor not found: {descriptor}. Create it first with "
            "scripts/prepare_two_model_variants.py --run-directory RUN "
            "--reduced-profile PROFILE --output VARIANTS_JSON. "
            "Seeds and their physical-review certificates are separate prerequisites."
        )
    try:
        variants = json.loads(descriptor.read_text())
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot read variants descriptor {descriptor}: {exc}")
    if isinstance(variants, list):
        for variant in variants:
            if not isinstance(variant, dict):
                parser.error("Each entry in the variants descriptor must be a JSON object")
            for key in ("model_config", "seed", "seed_certificate", "reduced_profile", "weights_config"):
                if key in variant and not isinstance(variant[key], str):
                    parser.error(f"Variant field {key} must be a path string")
                if key in variant and not Path(variant[key]).is_absolute():
                    variant[key] = str(descriptor.parent / variant[key])
    try:
        manifest = build_two_model_manifest(variants=variants, **args)
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot build campaign: {exc}")
    manifest["inputs"]["variant_descriptor"] = artifact(descriptor)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = _digest(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Eight-job manifest written: {destination}; no solver started")


if __name__ == "__main__":
    main()
