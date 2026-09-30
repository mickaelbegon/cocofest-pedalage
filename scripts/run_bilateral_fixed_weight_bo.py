#!/usr/bin/env python3
"""Run an auditable bilateral fixed-muscle-weight Bayesian optimisation.

Unlike the historical unilateral muscle-weight BO, each trial is a true
two-arm isokinetic RHO experiment.  A candidate normally supplies one shared,
centred log-relative weight vector.  For deliberately asymmetric left/right
muscle models, ``separate_arm_weights`` instead exposes a vector per arm; the
existing capacity-feedback supervisor remains responsible for the bounded work
split.
The objective is the common certified cycle prefix, evaluated progressively
at the declared fidelities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.simulation.async_bayesian_model import (
    muscle_weight_coordinate_name,
    muscle_weight_coordinates,
)
from cocofest.simulation.bilateral_endurance_campaign import (
    BilateralEnduranceCampaignConfig,
    BilateralEnduranceCandidate,
)
from cocofest.simulation.bilateral_endurance_runner import run_candidate


MUSCLE_NAMES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")


def candidate_from_coordinates(coordinates: Mapping[str, float], *, total_torque_nm: float,
                               initial_right_fraction: float, min_weight: float,
                               max_weight: float, separate_arm_weights: bool = False) -> tuple[BilateralEnduranceCandidate, dict[str, Any]]:
    """Map BO coordinates to shared or side-specific fixed bilateral weights."""
    def geometry_for(side: str | None):
        prefix = "" if side is None else f"{side}__"
        payload = {
            muscle_weight_coordinate_name(name): float(coordinates[f"{prefix}{name}"])
            for name in MUSCLE_NAMES if name != "Biceps"
        }
        return muscle_weight_coordinates(payload, MUSCLE_NAMES, min_weight=min_weight, max_weight=max_weight)

    right_geometry = geometry_for("right" if separate_arm_weights else None)
    left_geometry = geometry_for("left") if separate_arm_weights else right_geometry
    right = right_geometry["effective_initial_weights"]
    left = left_geometry["effective_initial_weights"]
    candidate = BilateralEnduranceCandidate(
        total_equivalent_mean_torque_nm=float(total_torque_nm),
        initial_right_fraction=float(initial_right_fraction),
        right_weights=right,
        left_weights=left,
    ).validate()
    geometry = (
        {"mode": "side_specific", "right": right_geometry, "left": left_geometry}
        if separate_arm_weights else {"mode": "shared", **right_geometry}
    )
    return candidate, geometry


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _summary(study, output: Path) -> None:
    records = []
    for trial in study.trials:
        records.append({
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "parameters": trial.params,
            "user_attributes": trial.user_attrs,
        })
    completed = [item for item in records if item["state"] == "COMPLETE" and item["value"] is not None]
    best = max(completed, key=lambda item: item["value"], default=None)
    _write(output / "summary.json", {"objective": "common_certified_cycles", "trials": records, "best": best})


def run(document: Mapping[str, Any]) -> None:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - environment-specific dependency.
        raise RuntimeError("Optuna is required for bilateral fixed-weight BO.") from exc

    campaign_data = document.get("campaign")
    if not isinstance(campaign_data, Mapping):
        raise ValueError("campaign must be an object.")
    campaign = BilateralEnduranceCampaignConfig.from_dict(campaign_data)
    campaign.validate()
    trials = document.get("n_trials", 8)
    startup = document.get("n_startup_trials", 4)
    seed = document.get("seed", 20260926)
    if type(trials) is not int or trials < 2:
        raise ValueError("n_trials must be an integer >= 2.")
    if type(startup) is not int or not 1 <= startup < trials:
        raise ValueError("n_startup_trials must be in 1..n_trials-1.")
    if type(seed) is not int:
        raise ValueError("seed must be an integer.")
    total = float(document.get("total_equivalent_mean_torque_nm", 1.6))
    fraction = float(document.get("initial_right_fraction", .5))
    bounds = document.get("relative_weight_bounds", [.25, 4.0])
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("relative_weight_bounds must be [minimum, maximum].")
    lower, upper = map(float, bounds)
    separate_arm_weights = document.get("separate_arm_weights", False)
    if type(separate_arm_weights) is not bool:
        raise ValueError("separate_arm_weights must be boolean.")
    output = Path(campaign.output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{output / 'study.sqlite3'}"
    study = optuna.create_study(
        study_name=str(document.get("study_name", "bilateral-fixed-weights")), direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup),
        storage=storage, load_if_exists=True,
    )
    if not study.trials:
        sides = ("right", "left") if separate_arm_weights else (None,)
        study.enqueue_trial({
            f"{'' if side is None else side + '__'}{muscle_weight_coordinate_name(name)}": 0.0
            for side in sides for name in MUSCLE_NAMES if name != "Biceps"
        })

    def objective(trial):
        sides = ("right", "left") if separate_arm_weights else (None,)
        coordinates = {
            f"{'' if side is None else side + '__'}{name}": trial.suggest_float(
                f"{'' if side is None else side + '__'}{muscle_weight_coordinate_name(name)}", -1.0, 1.0)
            for side in sides for name in MUSCLE_NAMES if name != "Biceps"
        }
        candidate, geometry = candidate_from_coordinates(
            coordinates, total_torque_nm=total, initial_right_fraction=fraction,
            min_weight=lower, max_weight=upper, separate_arm_weights=separate_arm_weights,
        )
        candidate_root = Path(campaign.output_root).expanduser() / "candidates" / candidate.identifier
        if candidate_root.exists():
            # TPE may resuggest an existing optimum after a resumed study.
            # It has already been evaluated scientifically, so do not turn a
            # duplicate filesystem collision into an artificial score of 0.
            trial.set_user_attr("duplicate_candidate_id", candidate.identifier)
            raise optuna.TrialPruned("candidate already evaluated")
        try:
            observations = run_candidate(campaign, candidate)
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            raise optuna.TrialPruned("candidate execution failed") from exc
        final = observations[-1]
        trial.set_user_attr("candidate_id", candidate.identifier)
        trial.set_user_attr("weight_geometry", geometry)
        trial.set_user_attr("observations", [item.to_dict() for item in observations])
        # A promotion that reaches the requested fidelity is administratively
        # censored there, and remains a valid lower-bound observation.
        return float(final.validated_cycles)

    # An enqueued baseline is not a completed budget item.  More importantly,
    # an Optuna duplicate or a technical launch error must not silently count
    # toward the requested *scientific* candidate budget.
    def valid_evaluations() -> int:
        return sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            and "error" not in trial.user_attrs
            and "duplicate_candidate_id" not in trial.user_attrs
            for trial in study.trials
        )

    attempts = 0
    max_attempts = max(8, 3 * trials)
    while valid_evaluations() < trials:
        if attempts >= max_attempts:
            raise RuntimeError("Too many duplicate or technical BO attempts before reaching the requested valid budget.")
        study.optimize(objective, n_trials=1)
        attempts += 1
        _summary(study, output)
    _summary(study, output)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    args = parser.parse_args(argv)
    data = json.loads(args.campaign.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("The BO configuration must be a JSON object.")
    run(data)


if __name__ == "__main__":
    main()
