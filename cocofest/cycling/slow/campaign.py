"""Read-only A0/A1 protocol construction, with no solver or launch path.

The plan pins caller-provided inputs. Reference endurance is a predeclared
comparison target, never a result inferred from a file name or a solver status.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path


@dataclass(frozen=True)
class SlowCampaignInputs:
    model_left: Path
    model_right: Path
    rho_config: Path
    baseline_source: Path
    split_config: Path
    horizon_cycles: int
    update_every_cycles: int = 10


def _reference(path: Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Campaign input must be a file: {resolved}")
    return {"path": str(resolved), "sha256": sha256(resolved.read_bytes()).hexdigest()}


def _config_reference(path: Path) -> dict:
    reference = _reference(path)
    document = json.loads(Path(reference["path"]).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not document:
        raise ValueError("Configuration must be a nonempty JSON object")
    # Reject nonstandard JSON NaN/Infinity before including the exact snapshot.
    json.dumps(document, allow_nan=False)
    return {**reference, "document": document}


def build_slow_campaign_plan(inputs: SlowCampaignInputs) -> dict:
    """Return a deterministic, non-executable plan without writing any files.

    Configuration snapshots describe source material, not a resolved RHO
    configuration. A future runner must resolve case overrides and certify
    every accepted cycle. This builder cannot enable online decisions, even
    when a source configuration happens to contain validation flags.
    """
    for name in ("horizon_cycles", "update_every_cycles"):
        if type(getattr(inputs, name)) is not int or getattr(inputs, name) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    sources = {
        "model_left": _reference(inputs.model_left),
        "model_right": _reference(inputs.model_right),
        "rho_config": _config_reference(inputs.rho_config),
        "baseline_source": _reference(inputs.baseline_source),
        "split_config": _config_reference(inputs.split_config),
    }
    common = {
        "horizon_cycles": inputs.horizon_cycles,
        "state_source": "certified_rho_oracle",
        "muscle_weights": {"policy": "unit", "value_for_every_muscle": 1.0},
        "required_cycle_acceptance": "full_discrete_rho_audit",
        "failure_classification": "frozen_feasibility_test",
        "bare_solver_failure_classification": "numerical_unresolved",
    }
    plan = {
        "schema": "slow-layer-ablation-plan-v1",
        "status": "plan_only",
        "sources": sources,
        "execution": {
            "launch_supported": False,
            "online_activation": False,
            "online_activation_requires": [
                "validated_conditional_nhat_on_instrumented_archives",
                "validated_phi_on_predeclared_holdout",
                "separately_reviewed_runner_integration",
            ],
            "validation_status": {"nhat": "not_assessed", "phi": "not_assessed"},
        },
        "cases": [
            {
                "id": "A0", **common,
                "torque_policy": "fixed_per_arm",
                "torque_nm": {"left": 0.96, "right": 0.96},
                "slow_split_enabled": False,
                "predeclared_criteria": {
                    "purpose": "fixed_load_unit_weight_control_for_weight_effect_isolation",
                    "invariants": ["left_torque_0.96_nm", "right_torque_0.96_nm", "all_weights_unit"],
                    "weight_effect_estimable_in_this_case_alone": False,
                },
            },
            {
                "id": "A1", **common,
                "torque_policy": "slow_equalization_at_fixed_total",
                "initial_torque_nm": {"left": 0.96, "right": 0.96},
                "total_torque_nm": 1.92,
                "slow_split_enabled": True,
                "update_every_cycles": inputs.update_every_cycles,
                "split_config_source": "sources.split_config",
                "predeclared_criteria": {
                    "purpose": "reference_reproduction_with_only_torque_split_varying",
                    "reference_certified_cycles": 211,
                    "reference_source": "sources.baseline_source",
                    "reference_is_observed_by_builder": False,
                    "reference_comparison": "exact_certified_cycle_count_and_matching_frozen_endpoint_class",
                    "horizon_sufficient_to_observe_reference_endpoint": inputs.horizon_cycles > 211,
                    "censoring_policy": "no_endpoint_reproduction_claim_if_horizon_reached",
                    "invariants": ["total_torque_1.92_nm", "all_weights_unit", "same_rho_and_models_as_A0"],
                },
            },
        ],
        "results": None,
    }
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False)
    plan["plan_sha256"] = sha256(payload.encode("utf-8")).hexdigest()
    return plan
