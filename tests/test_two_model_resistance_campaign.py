"""Eight-arm declarations and CPU/dependency gates without building an OCP."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import numpy as np

from scripts import build_two_model_resistance_campaign as campaign
from scripts.build_resistance_comparison_campaign import artifact
from cocofest.optimization.configured_cycling_model import resolve_model_config, FINGERPRINT_KEY


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign.os, "sched_getaffinity", lambda _: set(range(16)))
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text("# Runtime integration is independently tested by its owner.\n")
    monkeypatch.setattr(campaign, "WRAPPER", wrapper)
    hsl = tmp_path / "hsl.so"
    hsl.write_bytes(b"HSL fixture")
    taskset = tmp_path / "taskset"
    taskset.write_bytes(b"taskset fixture: never executed")
    variants = []
    for index in range(2):
        model = tmp_path / f"model{index}.json"
        weights = tmp_path / f"weights{index}.json"
        seed = tmp_path / f"seed{index}.npz"
        profile = tmp_path / f"profile{index}.npz"
        muscles = {name: {"Fmax": 10., "a_scale": 100., "alpha_a": -.1 * (index + 1),
                          "tau_fat": 100.} for name in campaign.MUSCLE_NAMES}
        model.write_text(json.dumps({"schema_version": 1, "case_id": f"model{index}",
                                     "provenance": "synthetic test only", "muscles": muscles}))
        weights.write_text(json.dumps({"initial_weight_basis": "synthetic independent weights",
                                       "initial_weights": dict.fromkeys(campaign.MUSCLE_NAMES, 1.),
                                       "policy": {"max_cycles": 100}}))
        resolved = resolve_model_config(json.loads(model.read_text()))
        np.savez(seed, metadata__json=np.asarray(json.dumps({FINGERPRINT_KEY: resolved[FINGERPRINT_KEY]})))
        profile.write_bytes(b"profile fixture")
        validation = tmp_path / f"seed-validation{index}.json"
        validation.write_text('{"fixture_only":true}')
        certificate = tmp_path / f"seed-certificate{index}.json"
        certificate.write_text(json.dumps({"schema_version": 1, "stage": "per_model_seed_preparation",
                                           FINGERPRINT_KEY: resolved[FINGERPRINT_KEY],
                                           "seed": artifact(seed), "model_config": artifact(model),
                                           "reduced_profile": artifact(profile), "resistance_nm": .1,
                                           "gates": dict.fromkeys(campaign.INITIAL_GATES, True),
                                           "evidence": [artifact(validation)]}))
        variants.append(dict(model_id=f"model{index}", model_config=model, weights_config=weights,
                             seed=seed, seed_certificate=certificate, reduced_profile=profile))
    return dict(campaign_id="eight-test", resistance_nm=.1, cycles=4, variants=variants,
                output_directory=tmp_path / "out", hsl_library=hsl, cores_per_job=4,
                numeric_threads=2, cpu_ids=list(range(8)), python=sys.executable, taskset=taskset)


def _initial_reviews(manifest, tmp_path):
    report = tmp_path / "initial-audit.json"
    report.write_text('{"fixture_only":true}')
    return {"manifest_sha256": manifest["manifest_sha256"], "models": {
        chain["model_id"]: {"initial": {"gates": dict.fromkeys(campaign.MODEL_GATES, True),
                                        "evidence": [artifact(report)]}}
        for chain in manifest["model_chains"]}}


def _certify(manifest, reviews, job_id, cycles=None, outcome="completed"):
    job = next(arm for arm in manifest["arms"] if arm["id"] == job_id)
    result = Path(job["result_path"])
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text('{"fixture_only":true}')
    evidence = [artifact(result)]
    audit = Path(job["configuration_audit_path"])
    audit.write_text('{"fixture_only":true}')
    evidence.append(artifact(audit))
    if job["weights_journal_path"]:
        journal = Path(job["weights_journal_path"])
        journal.write_text('{"receipt_fixture_only":true}\n')
        evidence.append(artifact(journal))
    review = {"gates": dict.fromkeys(job["required_review_gates"], True), "evidence": evidence,
              "outcome": outcome, "certified_executed_cycles": manifest["cycles"] if cycles is None else cycles}
    reviews["models"][job["model_id"]][job["arm_id"]] = review
    return review


def _ids(jobs):
    return [job["id"] for job in jobs]


def _value(argv, key):
    return argv[argv.index(key) + 1]


def test_eight_commands_four_conditions_two_distinct_models_and_one_resistance(inputs):
    manifest = campaign.build_two_model_manifest(**inputs)
    assert len(manifest["arms"]) == 8
    assert manifest["schedule"]["max_concurrent_jobs"] == 2
    assert manifest["schedule"]["launch_waves_if_all_predecessors_certified"] == [
        [f"model0/{arm}", f"model1/{arm}"] for arm, _ in campaign.ARM_CONDITIONS]
    for index, chain in enumerate(manifest["model_chains"]):
        jobs = [job for job in manifest["arms"] if job["model_id"] == chain["model_id"]]
        assert [job["condition"] for job in jobs] == ["rho", "rho-physio", "rho-pace", "fho"]
        assert [job["after"] for job in jobs] == [None] + chain["jobs"][:-1]
        assert len({job["model_config_sha256"] for job in jobs}) == 1
        assert jobs[1]["weights_config_sha256"] == jobs[2]["weights_config_sha256"]
        for job in jobs:
            argv = job["argv"]
            assert _value(argv, "--model-config") == str(inputs["variants"][index]["model_config"])
            assert _value(argv, "--formulation") == "dynamic"
            assert _value(argv, "--signed-crank-torque") == "0.1"
            assert _value(argv, "--ipopt-hsl-library") == str(inputs["hsl_library"])
            assert _value(argv, "--common-initial-solution") == str(inputs["variants"][index]["seed"])
            assert "--weights-config" in argv if job["weights_applied"] else "--weights-config" not in argv
            assert "--single-shot" in argv if job["condition"] == "fho" else "--single-shot" not in argv
    assert len({job["result_path"] for job in manifest["arms"]}) == 8
    assert not inputs["output_directory"].exists()


def test_cpu_affinity_and_numeric_worker_product_are_explicit(inputs):
    manifest = campaign.build_two_model_manifest(**inputs)
    assert manifest["schedule"]["benchmark_worker_threads"] == 2
    assert manifest["schedule"]["declared_threads_per_job_bound"] == 4
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert manifest["environment"][key] == "2"
    assert manifest["environment"]["OMP_MAX_ACTIVE_LEVELS"] == "1"
    assert manifest["model_chains"][0]["cpu_ids"] == [0, 1, 2, 3]
    assert manifest["model_chains"][1]["cpu_ids"] == [4, 5, 6, 7]
    for arm in manifest["arms"]:
        assert _value(arm["argv"], "--n-threads") == "2"
        assert _value(arm["argv"], "--cpu-list") == ",".join(map(str, arm["cpu_ids"]))


@pytest.mark.parametrize("changes", [{"cores_per_job": 5}, {"numeric_threads": 3},
                                      {"cpu_ids": [0, 0, 1, 2, 3, 4, 5, 6]},
                                      {"cpu_ids": list(range(9, 17))}, {"numeric_threads": 0}])
def test_invalid_or_oversubscribed_cpu_plan_is_refused(inputs, changes):
    with pytest.raises(ValueError):
        campaign.build_two_model_manifest(**{**inputs, **changes})


def test_independent_progress_never_exceeds_two_jobs_or_skips_model_predecessor(inputs, tmp_path):
    manifest = campaign.build_two_model_manifest(**inputs)
    reviews = _initial_reviews(manifest, tmp_path)
    assert _ids(campaign.next_jobs(manifest, reviews)) == ["model0/baseline_rho", "model1/baseline_rho"]
    assert _ids(campaign.next_jobs(manifest, reviews, running_job_ids=["model0/baseline_rho"])) == ["model1/baseline_rho"]
    assert campaign.next_jobs(manifest, reviews, running_job_ids=["model0/baseline_rho", "model1/baseline_rho"]) == []
    _certify(manifest, reviews, "model0/baseline_rho")
    assert _ids(campaign.next_jobs(manifest, reviews, running_job_ids=["model1/baseline_rho"])) == ["model0/rho_physio"]
    _certify(manifest, reviews, "model0/rho_physio")
    assert _ids(campaign.next_jobs(manifest, reviews, running_job_ids=["model1/baseline_rho"])) == ["model0/rho_pace"]
    _certify(manifest, reviews, "model0/rho_pace")
    assert _ids(campaign.next_jobs(manifest, reviews, running_job_ids=["model1/baseline_rho"])) == ["model0/baseline_fho"]
    _certify(manifest, reviews, "model0/baseline_fho")
    for arm, _ in campaign.ARM_CONDITIONS:
        _certify(manifest, reviews, f"model1/{arm}")
    assert campaign.next_jobs(manifest, reviews) == []


@pytest.mark.parametrize("running", [["model0/rho_pace"],
                                      ["model0/baseline_rho", "model0/rho_physio"],
                                      ["model0/baseline_rho", "model1/baseline_rho", "model1/rho_pace"],
                                      ["missing"]])
def test_unordered_or_unreported_dependency_cannot_be_running(inputs, tmp_path, running):
    manifest = campaign.build_two_model_manifest(**inputs)
    with pytest.raises(ValueError):
        campaign.next_jobs(manifest, _initial_reviews(manifest, tmp_path), running_job_ids=running)


def test_missing_model_review_blocks_only_its_chain(inputs, tmp_path):
    manifest = campaign.build_two_model_manifest(**inputs)
    reviews = _initial_reviews(manifest, tmp_path)
    reviews["models"].pop("model0")
    assert _ids(campaign.next_jobs(manifest, reviews)) == ["model1/baseline_rho"]


def test_static_journal_and_model_receipt_gates_and_failure_classification(inputs, tmp_path):
    manifest = campaign.build_two_model_manifest(**inputs)
    reviews = _initial_reviews(manifest, tmp_path)
    _certify(manifest, reviews, "model0/baseline_rho", cycles=1, outcome="numerical_or_unresolved_stop")
    static = _certify(manifest, reviews, "model0/rho_physio")
    static["gates"]["static_weights_unchanged"] = False
    with pytest.raises(ValueError, match="gates"):
        campaign.next_jobs(manifest, reviews)
    static["gates"]["static_weights_unchanged"] = True
    static["outcome"] = "fatigue"
    with pytest.raises(ValueError, match="Never infer fatigue"):
        campaign.next_jobs(manifest, reviews)


def test_model_and_weights_changes_invalidate_whole_campaign(inputs, tmp_path):
    manifest = campaign.build_two_model_manifest(**inputs)
    reviews = _initial_reviews(manifest, tmp_path)
    path = inputs["variants"][1]["model_config"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="input or code changed"):
        campaign.next_jobs(manifest, reviews)


def test_variants_must_differ_in_parameters_and_use_positive_named_weights(inputs):
    bad = deepcopy(inputs)
    bad["variants"][1]["model_config"] = bad["variants"][0]["model_config"]
    with pytest.raises(ValueError, match="different muscle parameters"):
        campaign.build_two_model_manifest(**bad)
    weights = inputs["variants"][0]["weights_config"]
    content = json.loads(weights.read_text())
    content["initial_weights"]["Triceps"] = 0.
    weights.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="positive finite weights"):
        campaign.build_two_model_manifest(**inputs)


def test_existing_static_artifacts_cannot_be_relaunched(inputs, tmp_path):
    manifest = campaign.build_two_model_manifest(**inputs)
    reviews = _initial_reviews(manifest, tmp_path)
    _certify(manifest, reviews, "model0/baseline_rho")
    _certify(manifest, reviews, "model0/rho_physio")
    reviews["models"]["model0"].pop("rho_physio")
    with pytest.raises(ValueError, match="unreviewed result"):
        campaign.next_jobs(manifest, reviews)
    with pytest.raises(ValueError, match="already contains results|Existing results"):
        campaign.build_two_model_manifest(**inputs)


@pytest.mark.parametrize("fault", ["missing", "wrong_seed", "unversioned_seed", "failed_gate", "wrong_resistance"])
def test_seed_preparation_must_be_certified_for_each_effective_model(inputs, fault):
    variant = inputs["variants"][1]
    certificate_path = variant["seed_certificate"]
    certificate = json.loads(certificate_path.read_text())
    if fault == "missing":
        variant.pop("seed_certificate")
    elif fault in ("wrong_seed", "unversioned_seed"):
        np.savez(variant["seed"], metadata__json=np.asarray(json.dumps(
            {} if fault == "unversioned_seed" else {FINGERPRINT_KEY: "nominal-model-fingerprint"})))
        certificate["seed"] = artifact(variant["seed"])
    elif fault == "failed_gate":
        certificate["gates"]["seed_physically_certified"] = False
    else:
        certificate["resistance_nm"] = .2
    certificate_path.write_text(json.dumps(certificate))
    with pytest.raises(ValueError):
        campaign.build_two_model_manifest(**inputs)


def test_seed_preparation_evidence_and_hsl_remain_sealed(inputs, tmp_path):
    manifest = campaign.build_two_model_manifest(**inputs)
    reviews = _initial_reviews(manifest, tmp_path)
    evidence = manifest["inputs"]["model1:seed_preparation_evidence_0"]
    Path(evidence["path"]).write_text("changed physical validation")
    with pytest.raises(ValueError, match="input or code changed"):
        campaign.next_jobs(manifest, reviews)


def test_review_needs_bound_configuration_audit_not_just_true_gate(inputs, tmp_path):
    manifest = campaign.build_two_model_manifest(**inputs)
    reviews = _initial_reviews(manifest, tmp_path)
    review = _certify(manifest, reviews, "model0/baseline_rho")
    review["evidence"] = review["evidence"][:1]
    with pytest.raises(ValueError, match="configured-model audit"):
        campaign.next_jobs(manifest, reviews)


def test_model_variation_cannot_silently_change_mechanical_profile(inputs):
    inputs["variants"][1]["reduced_profile"].write_bytes(b"different geometry")
    with pytest.raises(ValueError, match="same reduced mechanical profile"):
        campaign.build_two_model_manifest(**inputs)


def test_generated_benchmark_arguments_parse_without_running_solver(inputs):
    from examples.fes_multibody.cycling.cycling_fes_solver_comparison import build_cli

    manifest = campaign.build_two_model_manifest(**inputs)
    for job in manifest["arms"]:
        parsed = build_cli().parse_args(job["argv"][job["argv"].index("--") + 1:])
        assert parsed.formulation == "dynamic"
        assert parsed.mechanical_formulation == "reduced"
        assert parsed.resistive_torque == .1
        assert parsed.single_shot == (job["condition"] == "fho")
        assert parsed.cycles_per_window == (4 if parsed.single_shot else 1)
        assert parsed.n_windows == 4
        assert job["environment"]["IPOPT_HSL_LIBRARY"] == str(inputs["hsl_library"])


def test_hsl_change_invalidates_eight_arm_campaign(inputs, tmp_path):
    manifest = campaign.build_two_model_manifest(**inputs)
    reviews = _initial_reviews(manifest, tmp_path)
    inputs["hsl_library"].write_bytes(b"different HSL build")
    with pytest.raises(ValueError, match="input or code changed"):
        campaign.next_jobs(manifest, reviews)
