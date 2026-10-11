import pytest

from scripts.slow_layer.run_causal_torque_pilot import job_action_schedule
from scripts.slow_layer.run_torque_transfer_dose_duration import declared_design, make_jobs


def test_full_design_has_matched_controls_and_all_declared_doses():
    design = declared_design()
    anchors = [{"cycle": 120}, {"cycle": 140}]
    jobs = make_jobs(anchors, design, control_repeats=5)
    # Per anchor: 2 durations * (5 controls + 3 negative + 1 positive) = 18.
    assert len(jobs) == 36
    assert {job["intervention_cycles"] for job in jobs} == {5, 10}
    assert {job["delta_left_nm"] for job in jobs if not job["is_control"]} == {-.025, -.05, -.1, .05}
    for cycle in (120, 140):
        for duration in (5, 10):
            controls = [job for job in jobs if job["anchor"]["cycle"] == cycle
                        and job["intervention_cycles"] == duration and job["is_control"]]
            assert len(controls) == 5


def test_fixed_design_refuses_fewer_than_the_preregistered_matched_controls():
    design = declared_design()
    with pytest.raises(ValueError, match="five controls"):
        make_jobs([{"cycle": 120}], design, control_repeats=1)


def test_sealed_job_schedule_preserves_total_and_explicitly_returns_equal_split():
    job = {"delta_left_nm": .1, "intervention_cycles": 5}
    assert job_action_schedule(job, "intervention") == pytest.approx({"left": 1.06, "right": .86})
    assert sum(job_action_schedule(job, "intervention").values()) == pytest.approx(1.92)
    assert job_action_schedule(job, "continuation") == {"left": .96, "right": .96}
    with pytest.raises(ValueError, match="Malformed"):
        job_action_schedule({"delta_left_nm": .05, "intervention_cycles": 0}, "intervention")
