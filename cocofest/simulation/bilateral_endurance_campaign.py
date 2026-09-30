"""Portable contracts for a certified bilateral 10-minute endurance campaign.

This module deliberately does not select a numerical optimiser.  It turns a
candidate into the small process-runner payload already used for independent
isokinetic arms, and classifies persisted run summaries from certified
terminal counterfactuals rather than from an IPOPT status alone. A scheduler
can therefore use the contract with deterministic bracketing, Optuna, or
another Bayesian surrogate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


DEFAULT_FIDELITIES = (30, 120, 300, 600)
STATUSES = ("feasible", "infeasible", "numerical_failure", "horizon_censored", "technical_failure", "administrative_censored")


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f"{name} must be strictly positive.")
    return value


def _normalised_weights(weights: Mapping[str, Any], side: str) -> dict[str, float]:
    if not isinstance(weights, Mapping) or len(weights) != 4 or any(not isinstance(name, str) or not name for name in weights):
        raise ValueError(f"{side}_weights must name exactly four muscles.")
    values = {str(name): _finite(value, f"{side}_weights[{name!r}]", positive=True)
              for name, value in weights.items()}
    log_mean = sum(math.log(value) for value in values.values()) / len(values)
    return {name: math.exp(math.log(value) - log_mean) for name, value in values.items()}


@dataclass(frozen=True)
class BilateralEnduranceCandidate:
    """One policy point; total work is derived, never independently supplied."""

    total_equivalent_mean_torque_nm: float
    initial_right_fraction: float
    right_weights: dict[str, float]
    left_weights: dict[str, float]

    def validate(self) -> "BilateralEnduranceCandidate":
        _finite(self.total_equivalent_mean_torque_nm, "total_equivalent_mean_torque_nm", positive=True)
        fraction = _finite(self.initial_right_fraction, "initial_right_fraction")
        if not 0 < fraction < 1:
            raise ValueError("initial_right_fraction must lie strictly between zero and one.")
        _normalised_weights(self.right_weights, "right")
        _normalised_weights(self.left_weights, "left")
        return self

    def canonical_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "total_equivalent_mean_torque_nm": float(self.total_equivalent_mean_torque_nm),
            "initial_right_fraction": float(self.initial_right_fraction),
            "right_weights": _normalised_weights(self.right_weights, "right"),
            "left_weights": _normalised_weights(self.left_weights, "left"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BilateralEnduranceCandidate":
        if not isinstance(data, Mapping):
            raise ValueError("candidate must be an object.")
        expected = set(cls.__dataclass_fields__)
        unknown = set(data) - expected
        if unknown:
            raise ValueError(f"Unknown candidate fields: {sorted(unknown)}")
        return cls(**dict(data)).validate()

    @property
    def identifier(self) -> str:
        encoded = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class BilateralEnduranceCampaignConfig:
    """Immutable protocol shared by all promoted candidate runs."""

    base_payload: dict[str, Any]
    output_root: str
    fidelities: tuple[int, ...] = DEFAULT_FIDELITIES
    target_cycles: int = 600
    update_every_cycles: int = 10
    initial_weight_basis: str = "uniform independent-arm baseline; no FHO data"
    adaptation_enabled: bool = False

    def __post_init__(self):
        if isinstance(self.fidelities, list):
            object.__setattr__(self, "fidelities", tuple(self.fidelities))

    def validate(self) -> "BilateralEnduranceCampaignConfig":
        if not isinstance(self.base_payload, dict):
            raise ValueError("base_payload must be an object.")
        required = {"factory", "solver", "formulation", "cycles_per_window", "parallel", "omega_rad_s"}
        missing = sorted(name for name in required if name not in self.base_payload)
        if missing:
            raise ValueError(f"base_payload missing required fields: {missing}")
        expected = {"solver": "ipopt", "formulation": "isokinetic", "cycles_per_window": 1, "parallel": True}
        for name, value in expected.items():
            if self.base_payload.get(name) != value:
                raise ValueError(f"The endurance campaign requires {name}={value!r}.")
        omega = _finite(self.base_payload["omega_rad_s"], "omega_rad_s")
        if omega >= 0:
            raise ValueError("omega_rad_s must be negative.")
        if not isinstance(self.output_root, str) or not self.output_root.strip() or "\0" in self.output_root:
            raise ValueError("output_root is required.")
        if type(self.target_cycles) is not int or self.target_cycles < 1:
            raise ValueError("target_cycles must be a positive integer.")
        if (not self.fidelities or any(type(value) is not int or value < 1 for value in self.fidelities)
                or tuple(sorted(set(self.fidelities))) != self.fidelities or self.fidelities[-1] != self.target_cycles):
            raise ValueError("fidelities must be strictly increasing and end at target_cycles.")
        if type(self.update_every_cycles) is not int or self.update_every_cycles < 1:
            raise ValueError("update_every_cycles must be a positive integer.")
        if not isinstance(self.initial_weight_basis, str) or not self.initial_weight_basis.strip():
            raise ValueError("initial_weight_basis is required for provenance.")
        if type(self.adaptation_enabled) is not bool:
            raise ValueError("adaptation_enabled must be boolean.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BilateralEnduranceCampaignConfig":
        if not isinstance(data, Mapping):
            raise ValueError("campaign must be an object.")
        expected = set(cls.__dataclass_fields__)
        unknown = set(data) - expected
        if unknown:
            raise ValueError(f"Unknown campaign fields: {sorted(unknown)}")
        return cls(**dict(data)).validate()

    def payload_for(self, candidate: BilateralEnduranceCandidate, cycles: int) -> dict[str, Any]:
        """Produce the only runner payload allowed for this candidate/fidelity."""
        self.validate()
        candidate.validate()
        if cycles not in self.fidelities:
            raise ValueError("A candidate may only run at a declared fidelity.")
        item = candidate.canonical_dict()
        total, fraction = item["total_equivalent_mean_torque_nm"], item["initial_right_fraction"]
        right, left = total * fraction, total * (1.0 - fraction)
        payload = dict(self.base_payload)
        resistance_pace = dict(payload.get("resistance_pace", {}))
        muscle_pace = dict(payload.get("muscle_pace", {}))
        resistance_pace.update({
            "total_equivalent_mean_torque_nm": total,
            "initial_right_fraction": fraction,
            "capacity_feedback": True,
            "update_every_cycles": self.update_every_cycles,
        })
        muscle_pace.update({
            "right_initial_weights": item["right_weights"],
            "left_initial_weights": item["left_weights"],
            "initial_weight_basis": self.initial_weight_basis,
            "update_every_cycles": self.update_every_cycles,
            "adaptation_enabled": self.adaptation_enabled,
        })
        payload.update({
            "cycles": cycles,
            "right_equivalent_mean_torque_nm": right,
            "left_equivalent_mean_torque_nm": left,
            "resistance_pace": resistance_pace,
            "muscle_pace": muscle_pace,
            "endurance_candidate": {"identifier": candidate.identifier, **item},
        })
        return payload

    def candidate_directory(self, candidate: BilateralEnduranceCandidate, cycles: int) -> Path:
        self.validate()
        if cycles not in self.fidelities:
            raise ValueError("Unknown fidelity.")
        return Path(self.output_root).expanduser() / "candidates" / candidate.identifier / f"cycles-{cycles:04d}"


@dataclass(frozen=True)
class BilateralEnduranceObservation:
    status: str
    fidelity_cycles: int
    validated_cycles: int
    minimum_capacity_ratio: float | None
    reason: str = ""

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"Unknown observation status: {self.status}")

    @property
    def promotable(self) -> bool:
        return self.status == "feasible"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BilateralTorqueBracketConfig:
    """One-dimensional, monotone search of total torque for a fixed policy.

    The bounded split and relative muscle weights remain unchanged.  This is
    deliberately not a claim about the global optimum over policies; it is a
    numerically economical way to bracket the largest certified load for one
    reproducible policy before spending runs on Bayesian policy variation.
    """

    lower_torque_nm: float
    upper_torque_nm: float
    tolerance_nm: float = .01
    max_evaluations: int = 8

    def validate(self) -> "BilateralTorqueBracketConfig":
        lower = _finite(self.lower_torque_nm, "lower_torque_nm", positive=True)
        upper = _finite(self.upper_torque_nm, "upper_torque_nm", positive=True)
        tolerance = _finite(self.tolerance_nm, "tolerance_nm", positive=True)
        if lower >= upper:
            raise ValueError("lower_torque_nm must be below upper_torque_nm.")
        if tolerance >= upper - lower:
            raise ValueError("tolerance_nm must be smaller than the initial bracket.")
        if type(self.max_evaluations) is not int or self.max_evaluations < 2:
            raise ValueError("max_evaluations must be at least two.")
        return self


def _certified_counterfactual(cycle: Mapping[str, Any], name: str) -> bool:
    """Whether one terminal diagnostic supplied a certified feasible witness."""
    probes = cycle.get("counterfactual_probes")
    if not isinstance(probes, Mapping):
        return False
    candidate = probes.get(name)
    if isinstance(candidate, Mapping):
        return candidate.get("feasible_witness") is True
    if isinstance(candidate, (tuple, list)):
        return any(isinstance(item, Mapping) and item.get("feasible_witness") is True for item in candidate)
    return False


def _counterfactual_endpoint_reason(attempted: list[Mapping[str, Any]]) -> str | None:
    """Classify only experimentally witnessed terminal counterfactuals.

    A rest-fatigue witness is deliberately the strongest attribution: it
    restores exactly A/Tau1/Km while retaining F, Cn, kinematics, work and
    declared PW bounds.  PW relief alone diagnoses recruitment saturation,
    while a lower-work witness merely localizes a load boundary and is not
    sufficient to call the failure physiological.
    """
    fatigue = [item for item in attempted if _certified_counterfactual(item, "fatigue_rest")]
    if fatigue:
        saturation = any(_certified_counterfactual(item, "pw_upper_relief") for item in fatigue)
        return "fatigue_counterfactual_certified_with_pw_saturation" if saturation else "fatigue_counterfactual_certified"
    if any(_certified_counterfactual(item, "pw_upper_relief") for item in attempted):
        return "declared_pulse_width_bound_saturation_counterfactual_certified"
    if any(_certified_counterfactual(item, "work_relief") for item in attempted):
        return "near_load_boundary_work_relief_certified"
    return None


def assess_summary(summary: Mapping[str, Any], fidelity_cycles: int) -> BilateralEnduranceObservation:
    """Classify a process summary using certified endpoint counterfactuals."""
    if type(fidelity_cycles) is not int or fidelity_cycles < 1:
        raise ValueError("fidelity_cycles must be positive.")
    if not isinstance(summary, Mapping):
        return BilateralEnduranceObservation("technical_failure", fidelity_cycles, 0, None, "missing_summary")
    arms = summary.get("arms")
    if not isinstance(arms, Mapping) or any(not isinstance(arms.get(side), Mapping) for side in ("right", "left")):
        return BilateralEnduranceObservation("technical_failure", fidelity_cycles, 0, None, "missing_arm_result")
    validated = []
    reserves = []
    arm_cycles = {}
    for side in ("right", "left"):
        arm = arms[side]
        count = arm.get("validated_cycles", 0)
        validated.append(count if type(count) is int and count >= 0 else 0)
        arm_cycles[side] = tuple(cycle for cycle in arm.get("cycles", ()) if isinstance(cycle, Mapping))
    common = min(validated)
    # A pair is scientifically committed only through its common certified
    # prefix.  Do not let one arm's unpaired attempted next window influence
    # the bilateral fatigue reserve.
    for cycles in arm_cycles.values():
        for cycle in cycles:
            if cycle.get("certified") is True and type(cycle.get("cycle")) is int and cycle["cycle"] <= common:
                value = cycle.get("minimum_capacity_ratio")
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    reserves.append(float(value))
    reserve = min(reserves) if reserves else None
    if summary.get("success") is True and common == fidelity_cycles:
        return BilateralEnduranceObservation("feasible", fidelity_cycles, common, reserve, "both_arms_certified")
    failure = summary.get("failure")
    if summary.get("failure_kind") in {"fatigue_limit", "constraint_violation"} and isinstance(failure, str):
        return BilateralEnduranceObservation("infeasible", fidelity_cycles, common, reserve, failure)
    attempted = [cycle for cycles in arm_cycles.values() for cycle in cycles if cycle.get("certified") is False]
    counterfactual_reason = _counterfactual_endpoint_reason(attempted)
    if counterfactual_reason is not None:
        # A declared PW bound is a physiological stimulation limit of this
        # protocol. A mere lower-work witness, on the other hand, gives a
        # useful margin but is not physiology attribution.
        status = (
            "infeasible"
            if counterfactual_reason != "near_load_boundary_work_relief_certified"
            else "technical_failure"
        )
        return BilateralEnduranceObservation(status, fidelity_cycles, common, reserve,
                                            counterfactual_reason)
    if any((cycle.get("feasibility") or {}).get("failure_reason") == "primal_infeasibility_above_threshold"
           and float(cycle.get("solver_time_s", 0.)) >= 59. for cycle in attempted):
        return BilateralEnduranceObservation("numerical_failure", fidelity_cycles, common, reserve,
                                            "solver_time_limit_with_primal_infeasibility")
    return BilateralEnduranceObservation("technical_failure", fidelity_cycles, common, reserve,
                                        str(failure) if isinstance(failure, str) else "uncertified_summary")
