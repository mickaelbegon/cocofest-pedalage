from types import SimpleNamespace

import numpy as np
import pytest

from scripts import validate_local_endurance_value as validation


def test_case_parser_is_explicit_about_zero_based_anchor_and_positive_horizon():
    assert validation._case("112:30") == (112, 30)
    for invalid in ("112", "-1:10", "0:0", "zero:10"):
        with pytest.raises(Exception):
            validation._case(invalid)


def test_replay_failure_is_reported_instead_of_aborting_campaign(monkeypatch):
    coordinates = SimpleNamespace(decode=lambda point: np.ones((1, 5)))
    compact = SimpleNamespace(
        status="complete",
        completed_intervals=1,
        pulse_widths=np.ones((1, 1, 1)),
    )
    predictor = SimpleNamespace(rollout=lambda initial, horizon_cycles: compact)
    policy = SimpleNamespace(intervals=(object(),), parameters=(object(),))

    def fail(*args, **kwargs):
        raise ValueError("deliberate Ding-domain failure")

    monkeypatch.setattr(validation, "_full_ding_replay", fail)
    report, arrays = validation._replay_point(
        "probe",
        np.ones(2),
        coordinates=coordinates,
        predictor=predictor,
        policy=policy,
        horizon=1,
        substeps=1,
    )
    assert report["full_ding_replay_status"] == "failed"
    assert "deliberate Ding-domain failure" in report["full_ding_replay_error"]
    assert arrays == {}


def test_missing_training_records_has_no_fictitious_center_status():
    fit = SimpleNamespace(metadata={"training_records": []})
    assert validation._observed_center_status(fit) is None


def test_initial_trust_box_does_not_cross_zero_damage_or_force():
    center = np.array([0.001, 0.2, 0.004, 1.0])
    radius = validation._initial_trust_radius(
        center,
        2,
        damage_radius=0.1,
        force_radius=0.2,
        maximum_damage_fraction=0.25,
        maximum_force_fraction=0.25,
    )
    np.testing.assert_allclose(radius, [0.00025, 0.05, 0.001, 0.2])
    assert np.all(center - radius >= 0)
    with pytest.raises(ValueError, match="strictly positive"):
        validation._initial_trust_radius(
            [0.0, 0.2, 0.004, 1.0],
            2,
            damage_radius=0.1,
            force_radius=0.2,
            maximum_damage_fraction=0.25,
            maximum_force_fraction=0.25,
        )
