"""Run one bilateral endurance candidate through its declared fidelities.

This is intentionally a campaign *executor*, not a Bayesian sampler.  It
gives a GUI or a future sampler a durable, auditable primitive: candidates
are run at 30/120/300/600 (or a declared smoke schedule) and are promoted
only after a certified result at the preceding fidelity.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from cocofest.optimization.independent_arm_rho_pace import IndependentArmResistancePace, IndependentArmRhoPaceConfig
from .bilateral_endurance_campaign import (
    BilateralEnduranceCampaignConfig,
    BilateralEnduranceCandidate,
    BilateralEnduranceObservation,
    BilateralTorqueBracketConfig,
    assess_summary,
)


def _factory(specification: str):
    try:
        module, name = specification.split(":", 1)
        factory = getattr(importlib.import_module(module), name)
    except (ValueError, ImportError, AttributeError) as exc:
        raise ValueError("factory must be an importable 'package.module:function'.") from exc
    if not callable(factory):
        raise ValueError("factory must name a callable.")
    return factory


def run_candidate(campaign: BilateralEnduranceCampaignConfig, candidate: BilateralEnduranceCandidate) -> list[BilateralEnduranceObservation]:
    """Run and persist each promoted fidelity, stopping on any non-feasible result."""
    campaign.validate()
    candidate.validate()
    observations: list[BilateralEnduranceObservation] = []
    for fidelity in campaign.fidelities:
        payload = campaign.payload_for(candidate, fidelity)
        root = campaign.candidate_directory(candidate, fidelity)
        if root.exists():
            raise FileExistsError(f"Candidate artifact already exists: {root}")
        root.mkdir(parents=True)
        (root / "input.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        try:
            coordinator = _factory(payload["factory"])(payload)
            pace = IndependentArmResistancePace(IndependentArmRhoPaceConfig(**payload["resistance_pace"]))
            summary = coordinator.run_with_resistance_pace_to_directory(root, pace, cycles=fidelity)
            observation = assess_summary(summary, fidelity)
        except BaseException as error:
            observation = BilateralEnduranceObservation("technical_failure", fidelity, 0, None,
                                                       f"{type(error).__name__}: {error}")
        (root / "observation.json").write_text(json.dumps(observation.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        observations.append(observation)
        if not observation.promotable:
            break
    root = Path(campaign.output_root).expanduser() / "candidates" / candidate.identifier
    (root / "promotion.json").write_text(json.dumps({"candidate": candidate.canonical_dict(),
        "observations": [item.to_dict() for item in observations]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return observations


def run_torque_bracket(campaign: BilateralEnduranceCampaignConfig, policy: BilateralEnduranceCandidate,
                       bracket: BilateralTorqueBracketConfig, *, evaluate=run_candidate) -> dict[str, Any]:
    """Bracket maximum certified torque, halting rather than guessing on errors.

    ``evaluate`` is injectable so this numerical policy can be tested without
    invoking IPOPT.  A result is feasible only if its final fidelity is
    feasible.  Any technical observation invalidates the bracket instead of
    silently being treated as a fatigue limit.
    """
    campaign.validate()
    policy.validate()
    bracket.validate()
    root = Path(campaign.output_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []

    def candidate_at(torque: float):
        return BilateralEnduranceCandidate(torque, policy.initial_right_fraction,
                                           policy.right_weights, policy.left_weights)

    def observe(torque: float):
        candidate = candidate_at(torque)
        observations = evaluate(campaign, candidate)
        final = observations[-1] if observations else BilateralEnduranceObservation(
            "technical_failure", 0, 0, None, "empty_evaluation")
        entry = {"torque_nm": torque, "candidate_id": candidate.identifier,
                 "observations": [item.to_dict() for item in observations], "final_status": final.status}
        history.append(entry)
        return final

    lower, upper = float(bracket.lower_torque_nm), float(bracket.upper_torque_nm)
    lower_outcome = observe(lower)
    if lower_outcome.status != "feasible":
        result = {"status": "invalid_lower_bound", "history": history}
    else:
        upper_outcome = observe(upper)
        if upper_outcome.status == "feasible":
            result = {"status": "upper_bound_feasible", "history": history,
                      "certified_lower_torque_nm": upper}
        elif upper_outcome.status != "infeasible":
            result = {"status": "upper_bound_not_classified", "history": history}
        else:
            while upper - lower > bracket.tolerance_nm and len(history) < bracket.max_evaluations:
                middle = .5 * (lower + upper)
                outcome = observe(middle)
                if outcome.status == "feasible":
                    lower = middle
                elif outcome.status == "infeasible":
                    upper = middle
                else:
                    result = {"status": "interrupted_by_unclassified_run", "history": history,
                              "certified_lower_torque_nm": lower, "unverified_upper_torque_nm": upper}
                    break
            else:
                result = {"status": "converged" if upper - lower <= bracket.tolerance_nm else "budget_exhausted",
                          "history": history, "certified_lower_torque_nm": lower,
                          "infeasible_upper_torque_nm": upper, "bracket_width_nm": upper - lower}
    (root / "torque-bracket.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path, help="JSON object with campaign, candidate, and optional bracket fields")
    args = parser.parse_args(argv)
    data: dict[str, Any] = json.loads(args.campaign.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) not in ({"campaign", "candidate"}, {"campaign", "candidate", "bracket"}):
        raise ValueError("campaign JSON must contain campaign/candidate and optional bracket.")
    campaign = BilateralEnduranceCampaignConfig.from_dict(data["campaign"])
    candidate = BilateralEnduranceCandidate.from_dict(data["candidate"])
    if "bracket" in data:
        result = run_torque_bracket(campaign, candidate, BilateralTorqueBracketConfig(**data["bracket"]))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "converged" else 2
    observations = run_candidate(campaign, candidate)
    print(json.dumps([item.to_dict() for item in observations], indent=2, sort_keys=True))
    return 0 if observations and observations[-1].promotable else 2


if __name__ == "__main__":
    raise SystemExit(main())
