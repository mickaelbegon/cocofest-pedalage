"""Optuna persistence adapter, imported without loading scientific packages."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .async_bayesian_model import SimResult


def require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna est requis pour les campagnes asynchrones (pip install optuna).") from exc
    return optuna


def write_json(path, value):
    """Publish a complete snapshot atomically, including across interruptions."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def open_study(campaign):
    """Open local journal storage; callers must hold the campaign lock."""
    optuna = require_optuna()
    root = Path(campaign.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    contract = json.loads(json.dumps(campaign.scientific_contract(), sort_keys=True, allow_nan=False))
    campaign_path = root / "campaign.json"
    if campaign_path.exists():
        from .async_bayesian_model import AsyncBayesianCampaignConfig
        previous = AsyncBayesianCampaignConfig.from_json(campaign_path.read_text(encoding="utf-8"))
        previous_contract = json.loads(json.dumps(previous.scientific_contract(), sort_keys=True))
        if previous_contract != contract:
            raise ValueError("Le contrat scientifique du dossier a changé : choisissez un nouveau dossier.")
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    storage = JournalStorage(JournalFileBackend(str(root / "optuna.journal")))
    sampler = optuna.samplers.TPESampler(seed=campaign.seed, n_startup_trials=campaign.n_startup_trials,
                                         constant_liar=True)
    study = optuna.create_study(storage=storage, study_name=campaign.study_name,
                               direction=campaign.direction, sampler=sampler, load_if_exists=True)
    # Normalize tuples to their persistent JSON representation before comparison.
    saved = study.user_attrs.get("scientific_contract")
    if saved is not None and saved != contract:
        raise ValueError("Le contrat scientifique a changé : utilisez un nouveau dossier/nom d'étude.")
    if saved is None and study.trials:
        raise ValueError("L'étude existante ne possède pas de contrat scientifique vérifiable.")
    if study.direction.name.lower() != campaign.direction:
        raise ValueError("La direction de l'étude existante est incompatible.")
    study.set_user_attr("scientific_contract", contract)
    study.set_user_attr("censoring_policy", "FAIL excludes censored observations; scientific status is in sim_result")
    for parameters in campaign.enqueue_trials:
        study.enqueue_trial(parameters, skip_if_exists=True)
    write_json(root / "campaign.json", campaign.to_dict())
    return study


def suggest_parameters(trial, space):
    parameters = {}
    for name, spec in space.items():
        if spec["type"] == "categorical":
            value = trial.suggest_categorical(name, spec["choices"])
        elif spec["type"] == "int":
            value = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
        else:
            value = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        parameters[name] = value
    return parameters


def finish_trial(study, trial, result: SimResult):
    """Never feed censoring bounds or technical errors as objective values.

    Optuna TPE also learns from PRUNED trials, so FAIL is deliberately an
    internal sampler state here, independent of the scientific result status.
    """
    optuna = require_optuna()
    trial.set_user_attr("sim_result", result.to_dict())
    trial.set_user_attr("scientific_status", result.status)
    if result.status == "observed" and result.score is not None:
        study.tell(trial, result.score)
    else:
        study.tell(trial, state=optuna.trial.TrialState.FAIL)


def recover_interrupted_trials(study, root):
    """Recover durable completed artifacts; interrupted simulations are not resumed."""
    optuna = require_optuna()
    for frozen in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        trial = optuna.trial.Trial(study, frozen._trial_id)
        path = Path(root) / "trials" / f"trial_{trial.number:06d}" / "observation.json"
        try:
            result = SimResult(**json.loads(path.read_text(encoding="utf-8"))["sim_result"])
        except (OSError, ValueError, TypeError, KeyError):
            result = SimResult("technical_failure", reason="coordinator_interrupted_without_completed_artifact")
        finish_trial(study, trial, result)


def write_summary(study, root):
    trials = study.get_trials(deepcopy=False)
    counts = Counter(t.user_attrs.get("scientific_status", t.state.name.lower()) for t in trials)
    completed = [t for t in trials if t.state.name == "COMPLETE"]
    best = None
    if completed:
        trial = study.best_trial
        best = {"trial": trial.number, "score": trial.value, "parameters": trial.params,
                "sim_result": trial.user_attrs.get("sim_result"),
                "trial_directory": trial.user_attrs.get("trial_directory")}
    summary = {"study_name": study.study_name, "direction": study.direction.name.lower(),
               "counts": dict(counts), "trials": len(trials), "best": best,
               "censored_trials": [{"trial": t.number, "parameters": t.params,
                                     "sim_result": t.user_attrs["sim_result"]}
                                    for t in trials if t.user_attrs.get("scientific_status") in
                                    ("horizon_censored", "administrative_censored")]}
    write_json(Path(root) / "summary.json", summary)
    write_json(Path(root) / "best.json", best or {"reason": "no_exact_certified_observation"})
    return summary
