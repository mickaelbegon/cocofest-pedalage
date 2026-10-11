"""Preregistration and descriptive scoring of local fast-state closure probes.

Nothing in this module certifies a reduced Markov state. A finite envelope of
fast coordinates is not evidence that their effects can be omitted from Phi.
The plan deliberately has no solver launch or online activation path.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .oracle import load_certified_checkpoint


@dataclass(frozen=True)
class DecisionObservation:
    """Already-audited branch outcome, not an artifact or certification API.

    A caller must validate branch receipts before using this descriptive
    scorer. Counts are additional accepted cycles after the same anchor;
    equality of the experiment keys must cover slow state, action, model,
    discretization, return policy and stopping rule, excluding fast state.
    """

    experiment_key: str
    fast_state_key: str
    accepted_cycles: int
    endpoint: str


def decision_pair_effect(first: DecisionObservation, second: DecisionObservation,
                         *, resolution_cycles: float) -> dict:
    """Compare identical slow/action problems using an independently set resolution.

    Right-censored differences are differences of lower bounds, not identified
    effects; numerical failures also remain unresolved. A frozen local endpoint
    has protocol-conditional meaning, not a global physiological certificate.
    """
    if (type(resolution_cycles) not in (int, float) or not math.isfinite(resolution_cycles)
            or resolution_cycles < 0):
        raise ValueError("resolution_cycles must be finite and nonnegative")
    if not first.experiment_key or first.experiment_key != second.experiment_key:
        raise ValueError("Pair differs in slow state, action or experimental contract")
    for item in (first, second):
        if (not item.fast_state_key or type(item.accepted_cycles) is not int or item.accepted_cycles < 0
                or item.endpoint not in ("frozen_local_endpoint", "right_censored", "numerical_unresolved")):
            raise ValueError("Invalid branch observation")
    delta = second.accepted_cycles - first.accepted_cycles
    identified = all(item.endpoint == "frozen_local_endpoint" for item in (first, second))
    ratio = None if not identified or resolution_cycles == 0 else abs(delta) / resolution_cycles
    return {
        "pair_kind": "repeat_control" if first.fast_state_key == second.fast_state_key else "fast_probe",
        "identified": identified,
        "cycle_difference": delta if identified else None,
        "observed_lower_bound_difference": delta,
        "resolution_cycles": resolution_cycles,
        "effect_over_resolution": ratio,
        "classification": ("inconclusive" if not identified else
                           "resolved_effect" if abs(delta) > resolution_cycles else "within_observed_resolution"),
        "certified": False,
        "closure_established": False,
    }


def _inventory(archive_path: str) -> dict:
    """Describe frozen boundary/history arrays, never shifted primal states."""
    groups = {name: [] for name in ("Cn", "F", "PW_history", "mechanics", "other_state", "runtime")}
    with np.load(archive_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata__json"].item()))
        groups["runtime"] = [{"key": key, "value": value}
                             for key, value in sorted(metadata.get("runtime_state", {}).items())]
        for key in sorted(data.files):
            if key.startswith("problem__x_bounds:") and key.endswith(":min"):
                state = key.removeprefix("problem__x_bounds:").removesuffix(":min")
                if state.startswith(("A_", "Tau1_", "Km_")):
                    continue
                group = "Cn" if state.startswith("Cn_") else "F" if state.startswith("F_") else (
                    "mechanics" if state in ("q", "qdot", "qddot", "theta", "omega", "E_prod") else "other_state")
                upper_key = key.removesuffix(":min") + ":max"
                lower, upper = np.asarray(data[key], float), np.asarray(data[upper_key], float)
                if (lower.ndim != 2 or lower.shape != upper.shape or lower.shape[1] < 1
                        or not np.isfinite(lower[:, 0]).all() or not np.isfinite(upper[:, 0]).all()):
                    raise ValueError(f"Invalid boundary array: {key}")
                groups[group].append({"key": key, "initial_lower": lower[:, 0].tolist(),
                                      "initial_upper": upper[:, 0].tolist(),
                                      "frozen": bool(np.array_equal(lower[:, 0], upper[:, 0]))})
            elif key.startswith("controls__last_pulse_width_"):
                array = np.asarray(data[key], float)
                if not np.isfinite(array).all():
                    raise ValueError("Nonfinite archived stimulation history")
                groups["PW_history"].append({"key": key, "shape": list(array.shape),
                                             "values": array.tolist()})
    return groups


def build_fast_probe_plan(receipts, *, side: str, muscles, control_repeats: int = 3) -> dict:
    """Pin exact restart sources and declare the experiment before seeing outcomes.

    At fixed frequency and phase, Cn can be fixed by the schedule. An archived
    invariant is recorded as such; it must not be perturbed arbitrarily to
    manufacture a reachable history. Nearby natural pairs with unequal slow
    states are confounded and can only be sensitivity diagnostics.
    """
    if type(control_repeats) is not int or control_repeats < 2:
        raise ValueError("At least two independent repeat controls are required")
    receipts = tuple(Path(path).resolve(strict=True) for path in receipts)
    if not receipts or len(set(receipts)) != len(receipts):
        raise ValueError("Require distinct checkpoint receipts")
    sources = []
    for receipt in receipts:
        state = load_certified_checkpoint(receipt, side=side, muscles=muscles)
        sources.append({"receipt": str(receipt), "completed_cycles": state.completed_cycles,
                        "model_sha256": state.model_sha256, "slow_state": state.as_dict(),
                        "artifacts": [{"path": path, "sha256": digest} for path, digest in state.artifacts],
                        "fast_inventory": _inventory(state.artifacts[1][0])})
    plan = {
        "schema": "slow-phi-fast-closure-preregistration-v1", "status": "preregistered_not_executed",
        "side": side, "muscles": list(muscles), "sources": sources,
        "execution": {"launch_supported": False, "online_activation": False},
        "locality": "separate experiment per model hash, checkpoint, physical task and return policy",
        "actions": {"left_transfer_nm": [-0.05, 0.0, 0.05], "total_torque_nm": 1.92,
                    "hold_cycles": 10, "muscle_weights": "unit", "return_policy": "common_frozen_policy"},
        "controls": {"independent_repeats_per_action": control_repeats,
                     "resolution_cycles": "maximum paired absolute cycle difference among uncensored independent controls of the same checkpoint/action; max across pilot strata",
                     "resolution_status": "unmeasured",
                     "resolution_freeze": "before fast-probe outcomes are inspected",
                     "zero_spread": "zero observed resolution only; not a confidence bound"},
        "probe_protocol": [
            "Restore each full certified checkpoint, including all states, stimulation history and runtime phase.",
            "First search natural archived pairs with exactly equal slow states and physical task/action, but distinct fast state/history.",
            "If absent, generate admissible histories through audited preparatory RHO branches; require a constrained matching solve to return to the same slow boundary before comparing fast states.",
            "Audit each matching solve and every applied cycle. A feasible matching solve establishes local discrete admissibility, not physiological reachability outside its archived prefix.",
            "Cover Cn, F, PW history and mechanics jointly; record invariant groups from actual schedule/constraints rather than fabricating variations.",
            "If exact slow matching or coverage cannot be achieved, record closure as inconclusive and retain fast coordinates in Phi.",
            "For each matched fast history apply all three actions for ten cycles, then the same continuation to the frozen local endpoint or censoring limit.",
        ],
        "measurements": ["audited_cycle_feasibility", "additional_certified_cycles_to_frozen_local_endpoint",
                         "per_muscle_exponentially_weighted_force_integral_with_tau_fat_and_grid",
                         "three_action_ranking_with_ties_at_control_resolution"],
        "comparison": {"exact_match_fields": ["model_hash", "task_hash", "slow_boundary", "action",
                                               "return_policy", "stopping_rule"],
                       "primary_effect": "absolute paired additional-cycle difference divided by calibrated control resolution in cycles",
                       "zero_resolution": "report absolute difference; any nonzero difference is resolved, ratio is undefined",
                       "decision_failure": "action ranking reversal beyond calibrated resolution or audited feasibility change",
                       "force_output": "diagnostic in N s, never divided by cycle resolution",
                       "unresolved_policy": "numerical failure or right censoring cannot establish equivalence",
                       "scope": "local evidence only; no min/max fast envelope or finite probe set establishes Markov closure"},
        "gate": {"status": "not_assessed", "closure_established": False,
                 "requires": ["calibrated_repeated_controls", "audited_matched_fast_pairs",
                              "declared_fast_group_coverage_or_verified_invariance", "action_family_holdout",
                              "decision_ranking_validation", "weighted_force_validation"]},
        "results": None,
    }
    plan["plan_sha256"] = sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return plan
