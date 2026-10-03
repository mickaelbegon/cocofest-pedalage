#!/usr/bin/env python3
"""Audit witnessed one-cycle load margins on time and reachable policy holdouts.

This is an offline diagnostic. A feasible nonlinear OCP witness is a lower
bound on achievable load, not a globally optimized capacity or an endurance
value. The script never enables a terminal costate.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


TIME_TRAIN = (80, 100, 120, 140)
TIME_HOLDOUT = 160
BRANCH_TRAIN = (
    "branch-unit-c141", "branch-unit-c143",
    "branch-c1plus-c143", "branch-c1minus-c143",
    "branch-c2plus-c143", "branch-c2minus-c143",
    "branch-c3plus-c143", "branch-c3minus-c143",
)
BRANCH_HOLDOUT = (
    "branch-mixedplus-c142", "branch-mixedminus-c142",
    "branch-mixedplus-c144", "branch-mixedminus-c144",
)
MUSCLES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
STATE_SCALES = {"Cn": 1., "Tau1": .05, "Km": .1}


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _record(directory: Path, *, label: str, cycle: int) -> dict:
    request_path, result_path = directory / "request.json", directory / "result.json"
    request, result = json.loads(request_path.read_text()), json.loads(result_path.read_text())
    checkpoint = request["checkpoint"]
    archive, model_path = Path(checkpoint["archive_path"]), Path(checkpoint["model_path"])
    if (_digest(archive) != checkpoint["archive_sha256"] or
            _digest(model_path) != checkpoint["model_sha256"] or
            checkpoint["completed_cycles"] != cycle or result["failure_reason"] is not None):
        raise ValueError(f"Checkpoint/result provenance failed: {label}")
    for kind in ("nominal_witness", "load_witness"):
        witness = result[kind]
        if (not witness["independent_validation_passed"] or
                not witness["source_state_and_history_restored"] or
                witness["prepared_problem_sha256"] != checkpoint["prepared_problem_sha256"] or
                witness["model_sha256"] != checkpoint["model_sha256"] or
                witness["maximum_normalized_constraint_violation"] > 1e-6 or
                not Path(witness["witness_id"]).is_file()):
            raise ValueError(f"Uncertified {kind}: {label}")
    load = float(result["load_witness"]["work_scale"])
    if not 1. <= load < request["load_factor_upper_bound"] - 1e-6:
        raise ValueError(f"Invalid/censored load witness: {label}")
    model = json.loads(model_path.read_text())
    values = []
    with np.load(archive, allow_pickle=False) as prepared:
        for muscle in MUSCLES:
            parameters = model["muscles"][muscle]
            for state in ("Cn", "F", "A", "Tau1", "Km"):
                key = f"{state}_{muscle}"
                scale = (float(parameters["Fmax"]) if state == "F" else
                         float(parameters["a_scale"]) if state == "A" else STATE_SCALES[state])
                values.append(float(prepared[f"problem__x_bounds:{key}:min"][0, 0]) / scale)
    return {"label": label, "cycle": cycle, "reserve_witness": load - 1.,
            "load_factor_witness": load, "normalized_ding_state": values,
            "archive_sha256": checkpoint["archive_sha256"],
            "request_sha256": _digest(request_path), "result_sha256": _digest(result_path),
            "maximum_normalized_constraint_violation": max(
                result[k]["maximum_normalized_constraint_violation"]
                for k in ("nominal_witness", "load_witness"))}


def _poly_predict(cycles, reserves, target, degree: int) -> float:
    x = (np.asarray(cycles, float) - target) / 20.
    matrix = np.vander(x, N=degree + 1, increasing=True)
    coef = np.linalg.lstsq(matrix, np.asarray(reserves, float), rcond=None)[0]
    return float(coef[0])


def _local_design(records: list[dict], anchor: dict, *, full: bool) -> np.ndarray:
    values = np.asarray([row["normalized_ding_state"] for row in records])
    center = np.asarray(anchor["normalized_ding_state"])
    # A-only is the proposed four-muscle capacity fit. The full-state design
    # retains Cn/F/Tau1/Km to expose omitted fast-state variation and rank.
    selected = np.arange(20) if full else np.asarray([2, 7, 12, 17])
    states = (values[:, selected] - center[selected]) / .01
    cycles = (np.asarray([row["cycle"] for row in records]) - anchor["cycle"])[:, None]
    return np.column_stack([np.ones(len(records)), cycles, states])


def _ridge_predict(train: list[dict], target: list[dict], anchor: dict, *, full: bool,
                   alpha: float = 1e-3) -> tuple[np.ndarray, dict]:
    design = _local_design(train, anchor, full=full)
    values = np.asarray([row["reserve_witness"] for row in train])
    # Preserve the intercept while regularizing the time/state gradients.
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ values)
    singular = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design, tol=1e-8))
    condition = float(singular[0] / singular[-1]) if rank == design.shape[1] else None
    prediction = _local_design(target, anchor, full=full) @ coef
    return prediction, {"training_rank": rank, "required_rank": design.shape[1],
                        "condition_number": condition, "ridge_alpha": alpha,
                        "coefficients": coef.tolist()}


def assess(time: list[dict], branches: list[dict]) -> dict:
    time.sort(key=lambda row: row["cycle"])
    by_cycle = {row["cycle"]: row for row in time}
    if set(by_cycle) != set(TIME_TRAIN) | {TIME_HOLDOUT}:
        raise ValueError("Temporal train/holdout partitions changed")
    prior = [by_cycle[cycle] for cycle in TIME_TRAIN]
    measured = by_cycle[TIME_HOLDOUT]["reserve_witness"]
    temporal = {}
    for name, window, degree in (("last_two_linear", 2, 1), ("all_four_linear", 4, 1),
                                 ("last_three_quadratic", 3, 2), ("all_four_quadratic", 4, 2)):
        fit = prior[-window:]
        predicted = _poly_predict([row["cycle"] for row in fit],
                                  [row["reserve_witness"] for row in fit], TIME_HOLDOUT, degree)
        temporal[name] = {"prediction": predicted, "measured": measured,
                          "absolute_error": abs(predicted - measured),
                          "passes_0p01_abs_error_gate": abs(predicted - measured) <= .01}
    by_label = {row["label"]: row for row in branches}
    if set(by_label) != set(BRANCH_TRAIN) | set(BRANCH_HOLDOUT):
        raise ValueError("Reachable branch train/holdout partitions changed")
    anchor = by_cycle[140]
    training = [anchor] + [by_label[label] for label in BRANCH_TRAIN]
    heldout = [by_label[label] for label in BRANCH_HOLDOUT]
    local = {}
    for mode, full in (("A_only", False), ("full_Ding_state", True)):
        predicted, quality = _ridge_predict(training, heldout, anchor, full=full)
        holdout_rows = [{"label": row["label"], "cycle": row["cycle"],
                         "measured": row["reserve_witness"], "predicted": float(value),
                         "absolute_error": abs(float(value) - row["reserve_witness"])}
                        for row, value in zip(heldout, predicted)]
        # Hold one entire contrast policy out together, never one sibling
        # endpoint that shares the same stimulation prefix.
        groups = (("branch-c1plus-c143", "branch-c1minus-c143"),
                  ("branch-c2plus-c143", "branch-c2minus-c143"),
                  ("branch-c3plus-c143", "branch-c3minus-c143"))
        policy_errors = []
        for group in groups:
            reduced = [row for row in training if row["label"] not in group]
            target = [by_label[label] for label in group]
            candidate, _ = _ridge_predict(reduced, target, anchor, full=full)
            policy_errors.extend(abs(float(value) - row["reserve_witness"])
                                 for row, value in zip(target, candidate))
        local[mode] = {**quality, "heldout": holdout_rows,
                       "leave_contrast_pair_out_max_abs_error": max(policy_errors),
                       "heldout_max_abs_error": max(row["absolute_error"] for row in holdout_rows),
                       "heldout_reserve_contrast_c142": abs(heldout[0]["reserve_witness"] -
                                                           heldout[1]["reserve_witness"]),
                       "heldout_within_source_A_trust": all(
                           max(abs(np.asarray(row["normalized_ding_state"])[[2, 7, 12, 17]] -
                                   np.asarray(anchor["normalized_ding_state"])[[2, 7, 12, 17]])) <= .01
                           for row in heldout),
                       "passes_numeric_gate": quality["training_rank"] == quality["required_rank"]
                           and quality["condition_number"] is not None
                           and quality["condition_number"] <= 100
                           and max(policy_errors) <= .01
                           and max(row["absolute_error"] for row in holdout_rows) <= .01
                           and abs(heldout[0]["reserve_witness"] - heldout[1]["reserve_witness"]) >= .01
                           and all(max(abs(np.asarray(row["normalized_ding_state"])[[2, 7, 12, 17]] -
                                       np.asarray(anchor["normalized_ding_state"])[[2, 7, 12, 17]])) <= .01
                                   for row in heldout)}
    return {"schema_version": 1, "kind": "offline_task_load_margin_value_holdout_assessment",
            "temporal": temporal, "reachable_branch": local,
            "time_records": time, "branch_records": branches,
            "costate_activation_authorized": False,
            "activation_blockers": ["load factors are local feasible lower-bound witnesses, not global maxima",
                                    "only one anchor, one side and short continuations",
                                    "no validated remaining-cycle target or independent endurance ranking"],
            "gates": {"absolute_reserve_error": .01, "maximum_design_condition_number": 100.,
                      "minimum_reproducible_policy_reserve_contrast": .01}}


def envelope_fast_state_diagnostic(report: dict, envelope_path: Path) -> dict:
    """Decompose archived KKT prediction; never treat it as a validated fit."""
    envelope = json.loads(envelope_path.read_text())
    center = np.asarray(envelope["normalized_coordinates"], float)
    gradient = np.asarray(envelope["gradient"], float)
    anchor = next(row for row in report["time_records"] if row["cycle"] == 140)
    if center.shape != (20,) or gradient.shape != (20,) or not np.allclose(
            center, anchor["normalized_ding_state"], atol=1e-9, rtol=0):
        raise ValueError("Full-state envelope does not belong to c140")
    slow = np.asarray([2, 7, 12, 17])
    fast = np.asarray([index for index in range(20) if index not in slow])
    rows = []
    for row in report["branch_records"]:
        difference = np.asarray(row["normalized_ding_state"]) - center
        contribution = gradient * difference
        rows.append({"label": row["label"], "cycle": row["cycle"],
                     "measured": row["reserve_witness"],
                     "A_only_predicted": anchor["reserve_witness"] + float(sum(contribution[slow])),
                     "full_state_predicted": anchor["reserve_witness"] + float(sum(contribution)),
                     "A_contribution": float(sum(contribution[slow])),
                     "omitted_fast_state_contribution": float(sum(contribution[fast])),
                     "max_coordinate_shift": float(max(abs(difference))),
                     "max_A_coordinate_shift": float(max(abs(difference[slow]))),
                     "inside_source_A_trust_radius": bool(max(abs(difference[slow])) <= .01),
                     "inside_0p01_full_state_diagnostic_box": bool(max(abs(difference)) <= .01)})
    return {"source": str(envelope_path), "independently_validated": False,
            "method": "archived_single_point_KKT_envelope_linearization",
            "rows": rows,
            "interpretation": "Fast-state terms diagnose omitted-variable sensitivity only; "
                              "the 20-state gradient has no independent reoptimization certificate here."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-directory", type=Path, required=True)
    parser.add_argument("--historical-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envelope", type=Path, required=True,
                        help="Archived c140 full-state KKT envelope for omitted fast-state audit")
    args = parser.parse_args()
    time = [_record((args.new_directory if cycle in (80, 100) else args.historical_directory)
                    / f"c{cycle}", label=f"c{cycle}", cycle=cycle)
            for cycle in (*TIME_TRAIN, TIME_HOLDOUT)]
    branch_cycles = {label: int(label[-3:]) for label in (*BRANCH_TRAIN, *BRANCH_HOLDOUT)}
    branches = [_record(args.new_directory / label, label=label, cycle=cycle)
                for label, cycle in branch_cycles.items()]
    report = assess(time, branches)
    report["fast_state_diagnostic"] = envelope_fast_state_diagnostic(report, args.envelope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"temporal": report["temporal"],
                      "reachable_branch": report["reachable_branch"],
                      "costate_activation_authorized": False}, indent=2))


if __name__ == "__main__":
    main()
