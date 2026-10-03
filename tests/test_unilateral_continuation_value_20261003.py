from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_unilateral_continuation_value_20261003 import (
    _exact_labels, _predeclared_rows, endpoint_coordinates, fit_reachable_direction,
)


NAMES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
POLICY = "a" * 64


def _rows(anchor="c120"):
    base = {"id": "unit", "anchor_id": anchor, "family": "unit", "partition": "train",
            "intervention_cycles": 1, "return_policy_sha256": POLICY,
            "muscle_order": list(NAMES), "capacity_ratios": dict.fromkeys(NAMES, .7),
            "last_certified_cycle": 169, "label_kind": "physiological_failure_certified"}
    rows = [base]
    cases = [("a+", "train", (.01, 0, 0, 0), 171),
             ("a-", "train", (-.01, 0, 0, 0), 167),
             ("b+", "train", (0, .01, 0, 0), 170),
             ("b-", "train", (0, -.01, 0, 0), 168),
             ("h1", "holdout", (.005, .005, 0, 0), 171),
             ("h2", "holdout", (-.005, -.005, 0, 0), 167),
             ("h3", "holdout", (.005, -.005, 0, 0), 170)]
    for family, partition, delta, end in cases:
        rows.append({**base, "id": family, "family": family, "partition": partition,
                     "capacity_ratios": {name: .7 + value for name, value in zip(NAMES, delta)},
                     "last_certified_cycle": end})
    return rows


def test_reachable_fit_uses_only_train_span_and_independent_families():
    report = fit_reachable_direction(_rows())
    assert report["accepted"]
    assert report["reachable_rank"] == 2
    assert report["exact_train_families"] == 4
    assert report["exact_holdout_families"] == 3
    assert report["heldout_sign_correct"] == 3
    assert all(item["trusted"] for item in report["holdout"])
    assert report["activation_allowed"] is False


def test_off_span_holdout_rejected_even_when_label_looks_favorable():
    rows = _rows()
    rows[-1]["capacity_ratios"]["Biceps"] += .01
    report = fit_reachable_direction(rows)
    assert not report["accepted"]
    assert "holdout_outside_reachable_training_span" in report["reasons"]


def test_two_predeclared_holdout_families_can_pass_direction_gate():
    rows = [row for row in _rows() if row["family"] != "h3"]
    report = fit_reachable_direction(rows)
    assert report["accepted"]
    assert report["heldout_pair_total"] == 1


def test_uncertified_stop_is_not_a_value_label():
    rows = _rows()
    rows[-1]["label_kind"] = "numerical_stop"
    labelled, excluded = _exact_labels(rows)
    assert all(item["id"] != "h3" for item in labelled)
    assert excluded == [{"id": "h3", "reason": "no_exact_label_or_matched_unit_baseline"}]


def test_shared_prefix_family_cannot_leak_between_partitions():
    policy = {"equivalent_mean_torque_nm": .96, "maximum_absolute_cycles": 240}
    manifest = {"schema_version": 1, "kind": "unilateral_continuation_label_protocol",
                "side": "left", "return_policy": policy,
                "return_policy_sha256": sha256(json.dumps(policy, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest(),
                "jobs": [{"id": "a1", "anchor": {"cycle": 120},
                          "action": {"action_group": "a", "partition": "train"},
                          "intervention_cycles": 1},
                         {"id": "a3", "anchor": {"cycle": 120},
                          "action": {"action_group": "a", "partition": "holdout"},
                          "intervention_cycles": 3}]}
    with pytest.raises(ValueError, match="leaks"):
        _predeclared_rows(manifest)


def test_ding_offset_is_audited_and_not_silently_projected(tmp_path: Path):
    source = json.loads((Path(__file__).parents[1] / "asymmetric-sides-r192-rho-bo-20260928"
                         / "model-left.json").read_text())
    from cocofest.optimization.configured_cycling_model import resolve_model_config

    resolved = resolve_model_config(source)
    data = {}
    for name in NAMES:
        muscle = resolved["muscles"][name]
        data[f"states__A_{name}"] = np.array([[.7 * muscle["a_scale"]]])
        data[f"states__Tau1_{name}"] = np.array([[muscle["tau1_rest"] +
            muscle["alpha_tau1"] / muscle["alpha_a"] * (-.3 * muscle["a_scale"])]])
        data[f"states__Km_{name}"] = np.array([[muscle["km_rest"] +
            muscle["alpha_km"] / muscle["alpha_a"] * (-.3 * muscle["a_scale"])]])
        data[f"states__Cn_{name}"] = np.array([[0.]])
        data[f"states__F_{name}"] = np.array([[1.]])
        data[f"controls__last_pulse_width_{name}"] = np.full((1, 30), .0002)
    for name in ("theta", "omega", "E_prod"):
        data[f"states__{name}"] = np.array([[0.]])
    archive = tmp_path / "endpoint.npz"
    np.savez_compressed(archive, **data)
    assert endpoint_coordinates(archive, source)["slow_manifold_passed"]
    data["states__Tau1_Delt_ant"] += .01
    np.savez_compressed(archive, **data)
    assert not endpoint_coordinates(archive, source)["slow_manifold_passed"]
