"""Lifecycle and scientific classification tests without expensive solver runs."""
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import time

import pytest

from cocofest.simulation.async_bayesian_model import (
    AsyncBayesianCampaignConfig, SimResult, classify_result, muscle_weight_coordinate_name,
    muscle_weight_coordinates,
)
from cocofest.simulation.async_bayesian_runner import (
    _administrative_evidence, campaign_lock, evaluate_trial, read_applied_weight_audit, run_campaign,
    write_trial_weights_config,
)
from cocofest.simulation.async_bayesian_study import (
    finish_trial, open_study, recover_interrupted_trials, suggest_parameters, write_json, write_summary,
)
from cocofest.simulation.config import SimulationConfig
from cocofest.simulation.launch import LaunchPlan


def campaign_for(path, **kwargs):
    return AsyncBayesianCampaignConfig(base_config=SimulationConfig().to_dict(), output_root=str(path),
                                       n_trials=3, workers=2,
                                       search_space={"threads": {"type": "categorical", "choices": [1]}}, **kwargs)


def fake_evaluator(data, parameters, number, prefix, coordinator_pid):
    root = Path(data["output_root"]) / "trials" / f"trial_{number:06d}"
    root.mkdir(parents=True)
    time.sleep(.05)
    result = SimResult("observed", score=float(number + 1), validated_cycles=data["max_cycles"])
    entry = {"trial": number, "worker_pid": os.getpid(), "sim_result": result.to_dict()}
    write_json(root / "observation.json", entry)
    return entry


def test_campaign_lock_rejects_concurrent_coordinator(tmp_path):
    with campaign_lock(tmp_path):
        with pytest.raises(RuntimeError, match="coordinateur"):
            with campaign_lock(tmp_path):
                pass
    with campaign_lock(tmp_path):
        pass


def test_forced_stop_requires_explicit_partial_result_evidence():
    assert not _administrative_evidence({"validated_cycles": 3}, 10)
    assert not _administrative_evidence({"validated_cycles": 99, "administrative_censored": True}, 10)
    assert _administrative_evidence({"validated_cycles": 3, "termination_reason": "timeout"}, 10)


def test_capacity_saturation_endpoint_is_labelled_proxy_observation():
    result = {
        "success": False, "validated_cycles": 166,
        "fatigue_endurance_outcome": {
            "accepted": False, "label": "unconfirmed_endurance_stop",
            "evidence": ["ding_force_capacity_materially_decreased", "pulse_width_upper_bound_active"],
        },
        "continuous_endurance": {
            "available": True, "censored": False, "continuous_cycle_equivalent": 166.3,
            "method": "capacity_recruitment_proxy_v1",
        },
    }
    observation = classify_result(result, 2000, "continuous_endurance")
    assert observation.status == "observed"
    assert observation.reason == "capacity_saturation_proxy_endpoint"
    assert observation.score == pytest.approx(166.3)


def test_controller_muscle_weight_contract_generates_unique_effective_weights(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"muscles": {"Delt_ant": {}, "Delt_post": {}, "Biceps": {}, "Triceps": {}}}))
    baseline = tmp_path / "baseline-weights.json"
    baseline.write_text(json.dumps({
        "initial_weight_basis": "unit-test calibration",
        "initial_weights": {"Delt_ant": 2, "Delt_post": .1, "Biceps": 1, "Triceps": .25},
        "policy": {"max_cycles": 20, "min_relative_weight": .25, "max_relative_weight": 4},
    }))
    parameters = {muscle_weight_coordinate_name("Delt_ant"): .4,
                  muscle_weight_coordinate_name("Delt_post"): -.5,
                  muscle_weight_coordinate_name("Triceps"): .75}
    campaign = AsyncBayesianCampaignConfig(
        base_config=SimulationConfig(mode="rho-physio", solver="ipopt", model_config=str(model),
                                       weights_config=str(baseline)).to_dict(),
        output_root=str(tmp_path / "campaign"), study_kind="controller", metric="continuous_endurance",
        search_space={name: {"type": "float", "low": -1., "high": 1.}
                      for name in parameters},
    )
    assert campaign.validate() is campaign
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    path = Path(write_trial_weights_config(SimulationConfig.from_dict(campaign.base_config), parameters,
                                           trial_root, 7))
    generated = json.loads(path.read_text())
    effective = generated["initial_weights"]
    assert generated["policy"] == {"max_cycles": 20, "min_relative_weight": .25, "max_relative_weight": 4}
    assert set(effective) == {"Delt_ant", "Delt_post", "Biceps", "Triceps"}
    assert all(.25 <= value <= 4. for value in effective.values())
    assert pytest.approx(1.) == math.prod(effective.values()) ** .25
    assert generated["calibration"]["bayesian_optimization"]["trial"] == 7
    calibration = generated["calibration"]["bayesian_optimization"]
    assert calibration["schema"] == "centered_log_relative_weights_v2"
    assert calibration["reference_muscle"] == "Biceps"
    assert calibration["coordinates"] == {"Delt_ant": .4, "Delt_post": -.5, "Triceps": .75}
    assert calibration["effective_initial_weights"] == effective
    assert "baseline_sha256=" in generated["initial_weight_basis"]


def test_centered_log_geometry_is_one_to_one_and_needs_no_projection():
    names = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
    first = {muscle_weight_coordinate_name("Delt_ant"): -.1,
             muscle_weight_coordinate_name("Delt_post"): .2,
             muscle_weight_coordinate_name("Triceps"): .6}
    second = {**first, muscle_weight_coordinate_name("Triceps"): .7}
    a = muscle_weight_coordinates(first, names, min_weight=.25, max_weight=4.)
    b = muscle_weight_coordinates(second, names, min_weight=.25, max_weight=4.)
    assert a["reference_muscle"] == "Biceps"
    assert a["effective_initial_weights"] != b["effective_initial_weights"]
    assert pytest.approx(1., rel=1e-12) == a["geometric_mean"]
    assert all(.25 <= value <= 4. for value in a["effective_initial_weights"].values())
    assert sum(a["effective_centered_log_weights"].values()) == pytest.approx(0.)


def test_trial_weight_geometry_rejects_missing_or_invalid_baseline_weights(tmp_path):
    baseline = tmp_path / "baseline-weights.json"
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"muscles": {"Biceps": {}, "Triceps": {}}}))
    base = SimulationConfig(model_config=str(model), weights_config=str(baseline))
    parameters = {muscle_weight_coordinate_name("Triceps"): 0.}
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    baseline.write_text(json.dumps({"initial_weight_basis": "test", "initial_weights": {}}))
    with pytest.raises(ValueError, match="exactement les poids"):
        write_trial_weights_config(base, parameters, trial_root, 1)
    baseline.write_text(json.dumps({"initial_weight_basis": "test", "initial_weights": {"Biceps": 0, "Triceps": 1}}))
    with pytest.raises(ValueError, match="non positif"):
        write_trial_weights_config(base, parameters, trial_root, 1)


def test_muscle_weight_contract_rejects_acados_legacy_and_incomplete_mapping(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"muscles": {"Biceps": {}, "Triceps": {}}}))
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"initial_weight_basis": "test", "initial_weights": {"Biceps": 1}}))
    partial = {muscle_weight_coordinate_name("Biceps"): {"type": "float", "low": -1., "high": 1.}}
    with pytest.raises(ValueError, match="coordonnée"):
        AsyncBayesianCampaignConfig(
            base_config=SimulationConfig(mode="rho-physio", model_config=str(model), weights_config=str(weights)).to_dict(),
            output_root=str(tmp_path / "partial"), study_kind="controller", metric="continuous_endurance",
            search_space=partial,
        ).validate()
    complete = {muscle_weight_coordinate_name("Triceps"): {"type": "float", "low": -1., "high": 1.}}
    with pytest.raises(ValueError, match="pas encore connectés à ACADOS"):
        AsyncBayesianCampaignConfig(
            base_config=SimulationConfig(mode="rho-physio", solver="acados", model_config=str(model),
                                         weights_config=str(weights)).to_dict(),
            output_root=str(tmp_path / "acados"), study_kind="controller", metric="continuous_endurance",
            search_space=complete,
        ).validate()
    with pytest.raises(ValueError, match="v1"):
        AsyncBayesianCampaignConfig(
            base_config=SimulationConfig(mode="rho-physio", model_config=str(model), weights_config=str(weights)).to_dict(),
            output_root=str(tmp_path / "legacy"), study_kind="controller", metric="continuous_endurance",
            search_space={"muscle_weight__Biceps": {"type": "float", "low": .1, "high": 2.}},
        ).validate()


def test_weight_audit_reads_the_controller_receipt(tmp_path):
    result = tmp_path / "result.json"
    journal = tmp_path / "weights.jsonl"
    journal.write_text(json.dumps({
        "event": "configuration", "muscle_names": ["Biceps", "Triceps"],
        "initial_weights": [1., 1.], "initial_weight_normalization": "geometric_mean_one_then_log_box_projection",
        "initial_projection_changed_ratios": False,
    }) + "\n")
    audit = read_applied_weight_audit(LaunchPlan((), tmp_path, {}, "rho32", result),
                                      {"Biceps": 1., "Triceps": 1.})
    assert audit["status"] == "controller_receipt"
    assert audit["matches_requested_relative_weights"] is True


def test_worker_runs_fresh_process_and_certifies_result(tmp_path, monkeypatch):
    import cocofest.simulation.async_bayesian_runner as runner
    from types import SimpleNamespace
    campaign = campaign_for(tmp_path)
    result_path = tmp_path / "fresh.json"
    program = "import json,pathlib;pathlib.Path(%r).write_text(json.dumps(%r))" % (
        str(result_path), {"results": [{"solver": "ipopt", "success": True,
                                       "validated_cycles": 10, "solver_time_per_cycle_s": .25}]})
    plan = LaunchPlan((sys.executable, "-c", program), tmp_path, {}, "rho32", result_path)
    monkeypatch.setattr(runner, "build_launch_plan", lambda *args: plan)
    monkeypatch.setattr(runner, "runtime_helpers", lambda: SimpleNamespace(
        base_environment=lambda *args: dict(os.environ)))
    entry = evaluate_trial(campaign.to_dict(), {"threads": 1}, 0,
                           str(Path(sys.executable).parent.parent), os.getppid())
    assert entry["sim_result"]["score"] == .25
    assert entry["sim_result"]["status"] == "observed"
    assert (tmp_path / "trials/trial_000000/launch.json").exists()


def test_acados_trial_builds_a_fresh_high_budget_ma57_seed(tmp_path, monkeypatch):
    """The target must never reuse the campaign-level ACADOS seed."""
    import cocofest.simulation.async_bayesian_runner as runner
    from types import SimpleNamespace
    base = replace(SimulationConfig(solver="acados", acados_ipopt_cycle1_seed="old.npz"),
                   output_root=str(tmp_path))
    campaign = AsyncBayesianCampaignConfig(
        base_config=base.to_dict(), output_root=str(tmp_path), max_cycles=2,
        search_space={"threads": {"type": "categorical", "choices": [1]}},
    )
    seen = []

    def plan_for(config, *args):
        seen.append(config)
        if config.solver == "ipopt":
            archive = Path(config.extra_arguments[-1])
            result = tmp_path / "seed-result.json"
            program = "import json,pathlib; pathlib.Path(%r).parent.mkdir(parents=True,exist_ok=True); pathlib.Path(%r).write_bytes(b'x'); pathlib.Path(%r).write_text(json.dumps({'results':[{'solver':'ipopt','success':True,'validated_cycles':1}]}))" % (str(archive), str(archive), str(result))
            return LaunchPlan((sys.executable, "-c", program), tmp_path, {}, "rho32", result)
        result = tmp_path / "target-result.json"
        program = "import json,pathlib; pathlib.Path(%r).write_text(json.dumps({'results':[{'solver':'acados','success':True,'validated_cycles':2,'solver_time_per_cycle_s':.2}]}))" % str(result)
        return LaunchPlan((sys.executable, "-c", program), tmp_path, {}, "rho32", result)

    monkeypatch.setattr(runner, "build_launch_plan", plan_for)
    monkeypatch.setattr(runner, "runtime_helpers", lambda: SimpleNamespace(
        base_environment=lambda *args: dict(os.environ)))
    entry = evaluate_trial(campaign.to_dict(), {"threads": 1}, 0,
                           str(Path(sys.executable).parent.parent), os.getppid())
    assert entry["sim_result"]["status"] == "observed"
    seed = seen[0]
    target = seen[1]
    assert seed.solver == "ipopt" and seed.ipopt_linear_solver == "ma57" and seed.cycles == 1
    assert ("--ipopt-max-iter", "10000") == seed.extra_arguments[-4:-2]
    assert target.acados_ipopt_cycle1_seed != "old.npz"
    assert Path(target.acados_ipopt_cycle1_seed).is_file()


def test_censoring_excluded_from_sampler_and_contract_checked(tmp_path):
    optuna = pytest.importorskip("optuna")
    campaign = campaign_for(tmp_path)
    study = open_study(campaign)
    observed = study.ask()
    finish_trial(study, observed, SimResult("observed", score=.4))
    censored = study.ask()
    finish_trial(study, censored, SimResult("horizon_censored", lower_bound=10.0, validated_cycles=10))
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[1].state == optuna.trial.TrialState.FAIL
    assert study.trials[1].user_attrs["scientific_status"] == "horizon_censored"
    assert study.best_value == .4
    assert len(open_study(replace(campaign, n_trials=8)).trials) == 2
    with pytest.raises(ValueError, match="contrat scientifique"):
        open_study(replace(campaign, max_cycles=20))
    with pytest.raises(ValueError, match="contrat scientifique"):
        open_study(replace(campaign, study_name="another-study"))


def test_summary_separates_tpe_best_from_a_longer_censored_prefix(tmp_path):
    """A censored 1200-cycle run must not be hidden behind a 1156 proxy score."""
    pytest.importorskip("optuna")
    study = open_study(campaign_for(tmp_path))
    observed, censored = study.ask(), study.ask()
    finish_trial(study, observed, SimResult("observed", score=1156.1, validated_cycles=1156,
                                             reason="capacity_saturation_proxy_endpoint"))
    finish_trial(study, censored, SimResult("horizon_censored", lower_bound=1200., validated_cycles=1200,
                                             reason="requested_horizon_completed"))

    summary = write_summary(study, tmp_path)

    assert summary["best"]["trial"] == observed.number
    assert summary["best_validated_prefix"]["trial"] == censored.number
    assert summary["best_validated_prefix"]["validated_cycles"] == 1200
    assert summary["best_censored_lower_bound"]["trial"] == censored.number


def test_recovery_uses_durable_observation_only(tmp_path):
    optuna = pytest.importorskip("optuna")
    study = open_study(campaign_for(tmp_path))
    finished, interrupted = study.ask(), study.ask()
    path = tmp_path / "trials" / f"trial_{finished.number:06d}"
    path.mkdir(parents=True)
    write_json(path / "observation.json", {"sim_result": SimResult("observed", score=.8).to_dict()})
    recover_interrupted_trials(study, tmp_path)
    assert study.trials[finished.number].value == .8
    assert study.trials[interrupted.number].state == optuna.trial.TrialState.FAIL
    assert "interrupted" in study.trials[interrupted.number].user_attrs["sim_result"]["reason"]


def test_process_campaign_resume_respects_total_budget(tmp_path):
    pytest.importorskip("optuna")
    campaign = campaign_for(tmp_path)
    assert run_campaign(campaign, Path(sys.executable).parent.parent, evaluator=fake_evaluator) == 0
    initial = json.loads((tmp_path / "summary.json").read_text())
    assert initial["counts"]["observed"] == 3
    assert initial["best"]["score"] == 1.0
    assert run_campaign(replace(campaign, n_trials=5), Path(sys.executable).parent.parent,
                        evaluator=fake_evaluator) == 0
    assert json.loads((tmp_path / "summary.json").read_text())["counts"]["observed"] == 5
    entries = [json.loads(p.read_text()) for p in tmp_path.glob("trials/*/observation.json")]
    assert len(entries) == 5
    assert all(entry["worker_pid"] != os.getpid() for entry in entries)
