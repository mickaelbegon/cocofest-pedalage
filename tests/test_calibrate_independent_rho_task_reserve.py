"""Audit provenance and holdout leakage without launching expensive RHO solves."""

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.task_reserve import (
    DEFAULT_CONSTRAINT_GROUPS, ProbeEvidence, TaskReserveCheckpoint, evaluate_work_reserve,
)
from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint
from scripts import calibrate_independent_rho_task_reserve as campaign


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def case(tmp_path, monkeypatch):
    model = tmp_path / "model.json"
    write_json(model, {"synthetic": True})
    sources = {}
    manifest = {"schema_version": 1, "side": "right", "work_scales": [1., 1.1, 1.2, 1.3],
                "tolerance": 1e-5,
                "coordinates": [{"state_key": "A_Biceps", "index": 0, "scale": 1., "offset": 0.}],
                "fit": {"center": [.85], "trust_radius": [.1], "maximum_error": .01}, "samples": []}
    for index, (value, maximum, partition) in enumerate(((.8, 1.1, "train"), (.9, 1.3, "train"),
                                                         (.85, 1.2, "holdout"))):
        name = f"c{index}"
        program = SimpleNamespace(nlp=[SimpleNamespace(
            # Deliberately unrelated warm-start: only frozen initial equality is authoritative.
            x_init={"A_Biceps": SimpleNamespace(init=np.array([[42., 41.]]))},
            u_init={"last_pulse_width_Biceps": SimpleNamespace(init=np.array([[.0002]]))},
            x_bounds={"A_Biceps": SimpleNamespace(min=np.array([[value, 0., 0.]]),
                                                   max=np.array([[value, 2., 2.]]))},
            u_bounds={"last_pulse_width_Biceps": SimpleNamespace(min=np.array([[0.]]),
                                                                max=np.array([[.0006]]))},
        )], parameter_init={}, parameter_bounds={}, absolute_wheel_q_cycle_index=index + 1)
        archive, receipt = tmp_path / f"{name}.npz", tmp_path / f"{name}-receipt.json"
        export_prepared_checkpoint(archive, program, completed_cycles=index + 1, model_path=model)
        write_json(receipt, {"synthetic_receipt": index})
        source = TaskReserveCheckpoint.from_archive(archive, completed_cycles=index + 1, model_path=model,
                                                    task_context={"side": "right", "nominal_work_j": 1.})
        sources[receipt] = source

        def evaluator(request):
            valid = request.work_scale <= maximum
            witness = tmp_path / f"{name}-{request.work_scale}.npz"
            if valid:
                metadata = {"source_archive_sha256": source.archive_sha256,
                            "prepared_problem_sha256": source.prepared_problem_sha256,
                            "work_scale": request.work_scale}
                np.savez_compressed(witness, states__A_Biceps=np.array([[value, value]]),
                                    controls__last_pulse_width_Biceps=np.array([[.0002]]),
                                    vector=np.array([value]), metadata__json=np.asarray(json.dumps(metadata)))
            return ProbeEvidence(request.work_scale, source.prepared_problem_sha256, source.task_context_sha256,
                                 source.model_sha256, "0" if valid else "infeasible", 1e-8 if valid else 1.,
                                 1e-5, valid, str(witness) if valid else None, .01,
                                 DEFAULT_CONSTRAINT_GROUPS, True)

        result = tmp_path / f"{name}-probe.json"
        write_json(result, asdict(evaluate_work_reserve(source, manifest["work_scales"], evaluator)))
        manifest["samples"].append({"id": name, "receipt": receipt.name,
                                    "partition": partition, "probe_result": result.name})
    monkeypatch.setattr(campaign, "_load_arm", lambda receipt, side: (sources[Path(receipt)], {}))
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    return SimpleNamespace(path=path, manifest=manifest, sources=sources, root=tmp_path)


def test_affine_fit_uses_frozen_states_and_separate_holdout(case):
    output = case.root / "out"
    audit_path = campaign.run(case.path, output_directory=output)
    audit = json.loads(audit_path.read_text())
    assert audit["accepted"]
    assert audit["fit"]["gradient"] == pytest.approx([2.])
    assert audit["fit"]["rank"] == 2
    assert audit["sample_counts"] == {"train": 2, "holdout": 1, "manifest": 3}
    assert audit["task_context"]["coordinate_layout"] == case.manifest["coordinates"]
    assert audit["physical_task_context_sha256"] != audit["task_context_sha256"]
    assert audit["fit"]["task_context_sha256"] == audit["task_context_sha256"]
    assert audit["samples"][0]["coordinates"][0]["raw_value"] == .8
    assert len(audit["samples"][0]["probe"]["witnesses"]) == 2
    assert not audit["globally_certified"]
    assert not audit["continuous_ode_replay_performed"]
    saved = json.loads((output / "local-reserve-model.json").read_text())
    assert saved["model"]["gradient"] == pytest.approx([2.])
    assert saved["calibration_audit_sha256"]
    with pytest.raises(FileExistsError, match="overwrite"):
        campaign.run(case.path, output_directory=output)


@pytest.mark.parametrize("change,match", [
    (lambda m: m.update(work_scales=[1.1, 1.2]), "nominal"),
    (lambda m: m["fit"].update(minimum_train_samples=1), "dimension"),
    (lambda m: m["fit"].update(minimum_holdout_samples=0), "holdout"),
    (lambda m: m["samples"][0].update(partition="automatic"), "partition|declare"),
    (lambda m: m["samples"][1].update(receipt=m["samples"][0]["receipt"]), "twice"),
    (lambda m: m["samples"][0].update(id="../escape"), "safe"),
])
def test_manifest_fail_closed(case, change, match):
    change(case.manifest)
    write_json(case.path, case.manifest)
    with pytest.raises(ValueError, match=match):
        campaign.load_manifest(case.path)


def test_insufficient_holdout_writes_rejected_audit_only(case):
    case.manifest["samples"] = case.manifest["samples"][:2]
    write_json(case.path, case.manifest)
    output = case.root / "out"
    audit = json.loads(campaign.run(case.path, output_directory=output).read_text())
    assert not audit["accepted"]
    assert "insufficient_independent_holdout_samples" in audit["reasons"]
    assert not (output / "local-reserve-model.json").exists()


def test_out_of_trust_box_writes_rejected_audit_without_model(case, monkeypatch):
    case.manifest["fit"]["trust_radius"] = [.01]
    write_json(case.path, case.manifest)
    monkeypatch.setattr(campaign, "fit_local_reserve",
                        lambda *args, **kwargs: pytest.fail("out-of-domain fit must not run"))
    output = case.root / "out"
    audit_path = campaign.run(case.path, output_directory=output)
    audit = json.loads(audit_path.read_text())
    assert audit["accepted"] is False
    assert audit["reasons"] == ["train_sample_outside_trust_box"]
    assert audit["fit"] is None
    assert audit["model_path"] is None
    assert not (output / "local-reserve-model.json").exists()
    assert {item["sample_id"] for item in audit["trust_box_violations"]} == {"c0", "c1"}
    assert all(item["state_key"] == "A_Biceps" for item in audit["trust_box_violations"])
    assert audit["trust_box_violations"][0]["normalized_value"] == .8
    assert audit["trust_box_violations"][0]["lower_bound"] == pytest.approx(.84)


def test_out_of_trust_box_holdout_also_writes_rejected_audit(case, monkeypatch):
    case.manifest["samples"][0]["partition"] = "holdout"
    case.manifest["samples"][2]["partition"] = "train"
    case.manifest["fit"].update(center=[.875], trust_radius=[.05])
    write_json(case.path, case.manifest)
    monkeypatch.setattr(campaign, "fit_local_reserve",
                        lambda *args, **kwargs: pytest.fail("out-of-domain fit must not run"))
    output = case.root / "out"
    audit = json.loads(campaign.run(case.path, output_directory=output).read_text())
    assert audit["reasons"] == ["holdout_sample_outside_trust_box"]
    assert audit["fit"] is None
    assert audit["trust_box_violations"][0]["sample_id"] == "c0"
    assert audit["trust_box_violations"][0]["partition"] == "holdout"
    assert not (output / "local-reserve-model.json").exists()


@pytest.mark.parametrize("change,match", [
    (lambda v: v["checkpoint"].update(archive_sha256="tampered"), "provenance"),
    (lambda v: v["evidence"][0].update(tolerance=1e-3), "tolerance"),
    (lambda v: v["evidence"][0].update(maximum_normalized_constraint_violation=.1), "summary"),
    (lambda v: v["evidence"][0].update(validated_constraint_groups=["task_work"]), "summary"),
    (lambda v: v["evidence"].pop(), "grid"),
    (lambda v: v.update(physiological_failure_certified=True), "unsupported"),
])
def test_artifact_success_and_provenance_cannot_override_evidence(case, change, match):
    artifact = case.root / case.manifest["samples"][0]["probe_result"]
    value = json.loads(artifact.read_text())
    change(value)
    write_json(artifact, value)
    with pytest.raises(ValueError, match=match):
        campaign.run(case.path, output_directory=case.root / "out")
    assert not (case.root / "out" / "local-reserve-model.json").exists()


def test_checkpoint_alias_cannot_leak_between_partitions(case):
    receipts = list(case.sources)
    case.sources[receipts[-1]] = case.sources[receipts[0]]
    with pytest.raises(ValueError, match="leak"):
        campaign.run(case.path, output_directory=case.root / "out")


def test_missing_artifact_never_launches_unless_explicit(case, monkeypatch):
    sample = case.manifest["samples"][0]
    sample.pop("probe_result")
    write_json(case.path, case.manifest)
    calls = []
    monkeypatch.setattr(campaign, "run_probe", lambda *args, **kwargs: calls.append(True))
    with pytest.raises(FileNotFoundError, match="run-missing"):
        campaign.run(case.path, output_directory=case.root / "out")
    assert calls == []


def test_missing_probes_use_existing_runner_without_overwriting_existing(case, monkeypatch):
    sample = case.manifest["samples"][0]
    existing = case.root / sample.pop("probe_result")
    write_json(case.path, case.manifest)
    calls = []

    def probe(receipt, **kwargs):
        calls.append((receipt, kwargs))
        return existing

    monkeypatch.setattr(campaign, "run_probe", probe)
    campaign.run(case.path, output_directory=case.root / "out", run_missing=True)
    assert len(calls) == 1
    assert calls[0][1]["work_scales"] == [1., 1.1, 1.2, 1.3]
    assert calls[0][1]["side"] == "right"


def test_same_reserve_everywhere_is_not_directional_evidence(case):
    # Retain the nominal and 1.1 witnesses and mark larger factors unvalidated.
    for sample in case.manifest["samples"]:
        path = case.root / sample["probe_result"]
        value = json.loads(path.read_text())
        for report in value["evidence"]:
            if report["work_scale"] > 1.1:
                report.update(independent_validation_passed=False, witness_id=None)
        value.update(feasible_work_scales=[1., 1.1], work_scale_lower_bound=1.1,
                     reserve_lower_bound=1.1 - 1.)
        write_json(path, value)
    audit = json.loads(campaign.run(case.path, output_directory=case.root / "out").read_text())
    assert not audit["accepted"]
    assert "no_observed_reserve_contrast_for_directional_cost" in audit["reasons"]
    assert not (case.root / "out" / "local-reserve-model.json").exists()


def branch_case(case, monkeypatch):
    """Turn the numerical fixture into three mocked, predeclared branch endpoints."""
    plan_path = case.root / "branch-plan.json"
    policies = [{"id": "p0", "partition": "train"}, {"id": "p1", "partition": "train"},
                {"id": "p2", "partition": "holdout"}]
    plan = {"schema_version": 1, "kind": "local_task_reserve_experiment_plan", "side": "right",
            "local_neighborhoods": [{"anchor_id": "anchor", "anchor_receipt": "anchor.json"}],
            "reachable_branch_design": {"branches": policies}}
    write_json(plan_path, plan)
    plan_sha = sha256(plan_path.read_bytes()).hexdigest()
    for index, sample in enumerate(case.manifest["samples"]):
        receipt = case.root / sample["receipt"]
        write_json(receipt, {"side": "right", "plan_path": str(plan_path), "plan_sha256": plan_sha,
                             "anchor_id": "anchor", "policy": policies[index], "branch_cycle": 1,
                             "source_receipt": "anchor.json", "source_receipt_sha256": "anchor-digest"})
    case.manifest["source_kind"] = campaign.BRANCH_SOURCE
    case.manifest["experiment_plan"] = plan_path.name
    write_json(case.path, case.manifest)
    monkeypatch.setattr(campaign, "load_branch_endpoint",
                        lambda receipt: (case.sources[Path(receipt)], {"mock": "payload"}))
    monkeypatch.setattr(campaign, "_load_arm",
                        lambda *args: pytest.fail("bilateral loader must not read branch receipts"))
    return plan_path


def test_branch_endpoint_fit_preserves_policy_provenance_and_physical_context(case, monkeypatch):
    plan_path = branch_case(case, monkeypatch)
    audit = json.loads(campaign.run(case.path, output_directory=case.root / "out").read_text())
    assert audit["accepted"]
    assert audit["source_kind"] == campaign.BRANCH_SOURCE
    assert audit["experiment_plan"] == str(plan_path)
    assert audit["task_context"]["side"] == "right"
    assert "policy_id" not in audit["task_context"]
    assert [row["policy_provenance"]["policy_id"] for row in audit["samples"]] == ["p0", "p1", "p2"]
    assert [row["policy_provenance"]["policy_partition"] for row in audit["samples"]] == [
        "train", "train", "holdout"]


def test_branch_manifest_partition_cannot_override_predeclared_policy(case, monkeypatch):
    branch_case(case, monkeypatch)
    case.manifest["samples"][2]["partition"] = "train"
    write_json(case.path, case.manifest)
    with pytest.raises(ValueError, match="partition"):
        campaign.run(case.path, output_directory=case.root / "out", run_missing=True)
    assert not (case.root / "out").exists()


def test_branch_shared_prefix_endpoints_cannot_count_as_independent(case, monkeypatch):
    branch_case(case, monkeypatch)
    second = case.root / case.manifest["samples"][1]["receipt"]
    receipt = json.loads(second.read_text())
    receipt["policy"]["id"] = "p0"
    receipt["branch_cycle"] = 3
    write_json(second, receipt)
    with pytest.raises(ValueError, match="Shared-prefix"):
        campaign.run(case.path, output_directory=case.root / "out")


def test_branch_plan_mismatch_preflights_all_samples_before_probe(case, monkeypatch):
    branch_case(case, monkeypatch)
    case.manifest["samples"][0].pop("probe_result")
    write_json(case.path, case.manifest)
    last = case.root / case.manifest["samples"][-1]["receipt"]
    receipt = json.loads(last.read_text())
    receipt["plan_sha256"] = "another-plan"
    write_json(last, receipt)
    monkeypatch.setattr(campaign, "run_from_checkpoint",
                        lambda *args, **kwargs: pytest.fail("no probe before complete preflight"))
    with pytest.raises(ValueError, match="plan"):
        campaign.run(case.path, output_directory=case.root / "out", run_missing=True)
    assert not (case.root / "out").exists()


def test_missing_branch_probe_uses_validated_checkpoint(case, monkeypatch):
    branch_case(case, monkeypatch)
    first = case.manifest["samples"][0]
    existing = case.root / first.pop("probe_result")
    write_json(case.path, case.manifest)
    calls = []

    def fake_probe(source, payload, **kwargs):
        calls.append((source, payload, kwargs))
        return existing

    monkeypatch.setattr(campaign, "run_from_checkpoint", fake_probe)
    campaign.run(case.path, output_directory=case.root / "out", run_missing=True)
    assert len(calls) == 1
    assert calls[0][0] is case.sources[case.root / first["receipt"]]
    assert calls[0][1] == {"mock": "payload"}
    assert calls[0][2]["work_scales"] == [1., 1.1, 1.2, 1.3]


def test_heldout_reserve_must_be_predictable(case):
    # A train fit expects .2 at .85, but the independently witnessed holdout is .1.
    sample = case.manifest["samples"][-1]
    path = case.root / sample["probe_result"]
    value = json.loads(path.read_text())
    for report in value["evidence"]:
        if report["work_scale"] > 1.1:
            report.update(independent_validation_passed=False, witness_id=None)
    value.update(feasible_work_scales=[1., 1.1], work_scale_lower_bound=1.1,
                 reserve_lower_bound=1.1 - 1.)
    write_json(path, value)
    audit = json.loads(campaign.run(case.path, output_directory=case.root / "out").read_text())
    assert not audit["accepted"]
    assert "training_error_exceeds_tolerance" in audit["reasons"]


def test_witness_source_bytes_are_verified(case):
    artifact = case.root / case.manifest["samples"][0]["probe_result"]
    value = json.loads(artifact.read_text())
    witness = Path(value["evidence"][0]["witness_id"])
    np.savez_compressed(witness, metadata__json=np.asarray(json.dumps({"work_scale": 1.})), vector=[1.])
    with pytest.raises(ValueError, match="provenance"):
        campaign.run(case.path, output_directory=case.root / "out")


@pytest.mark.parametrize("field", ["task_context_sha256", "model_sha256"])
def test_different_tasks_or_models_cannot_enter_same_fit(case, field):
    receipt = list(case.sources)[1]
    case.sources[receipt] = replace(case.sources[receipt], **{field: "other-source"})
    with pytest.raises(ValueError, match="mixes"):
        campaign.run(case.path, output_directory=case.root / "out")


def test_coordinates_must_be_fixed_initial_equalities(tmp_path):
    archive = tmp_path / "free-state.npz"
    np.savez_compressed(archive, **{"problem__x_bounds:A_Biceps:min": [[.8, 0., 0.]],
                                   "problem__x_bounds:A_Biceps:max": [[1., 2., 2.]]})
    source = SimpleNamespace(archive_path=str(archive), verify_files=lambda: None)
    with pytest.raises(ValueError, match="frozen initial equality"):
        campaign.checkpoint_coordinates(source, [campaign.TaskReserveStateCoordinate("A_Biceps")])


def test_rejected_rank_never_publishes_model(case, monkeypatch):
    real_fit = campaign.fit_local_reserve

    def reject(*args, **kwargs):
        fitted = real_fit(*args, **kwargs)
        return replace(fitted, accepted=False, reasons=("rank_deficient",), rank=1,
                       condition_number=float("inf"))

    monkeypatch.setattr(campaign, "fit_local_reserve", reject)
    output = case.root / "out"
    audit = json.loads(campaign.run(case.path, output_directory=output).read_text())
    assert not audit["accepted"]
    assert audit["fit"]["condition_number"] is None
    assert "rank_deficient" in audit["reasons"]
    assert not (output / "local-reserve-model.json").exists()


def test_partial_probe_directory_is_not_reused(case):
    case.manifest["samples"][0].pop("probe_result")
    write_json(case.path, case.manifest)
    output = case.root / "out"
    partial = output / "probes" / "c0"
    partial.mkdir(parents=True)
    write_json(partial / "leftover.json", {"from_interrupted_probe": True})
    with pytest.raises(FileExistsError, match="Partial probe"):
        campaign.run(case.path, output_directory=output, run_missing=True)
