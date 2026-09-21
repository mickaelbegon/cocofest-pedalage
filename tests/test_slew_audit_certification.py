"""Failed or uncertified RHO attempts must never become executed seams."""
from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.pulse_width_slew import attach_slew_audit


def _audit(certification):
    # The first seam is 20 us; the next, involving cycle 3, is 200 us.
    summary = {
        "control_traces": {
            "last_pulse_width_m": np.array([200, 210, 220, 240, 250, 260, 460, 470, 480]) * 1e-6
        },
        **certification,
    }
    attach_slew_audit(summary, SimpleNamespace(stimulations_per_cycle=3, pulse_width_max_step_us=100))
    return summary["pulse_width_slew_audit"]


@pytest.mark.parametrize("certification", [
    {"validated_cycles": 2},
    {"physically_validated_cycles": 2, "validated_cycles": 3},
    {"window_statuses": [0, 0, 2], "window_feasibility": [{"passes_tolerance": True}] * 3},
    # A successful solver return does not certify an infeasible trajectory.
    {"window_statuses": [0, 0, 0], "window_feasibility": [
        {"passes_tolerance": True}, {"passes_tolerance": True}, {"passes_tolerance": False}]},
])
def test_failed_tail_seam_is_reported_only_as_attempted(certification):
    audit = _audit(certification)
    muscle = audit["muscles"]["m"]
    assert audit["certified_cycle_count"] == 2
    assert audit["trace_scope"] == "all_exported_attempts"
    assert muscle["maximum_executed_cycle_seam_change_us"] == pytest.approx(20)
    assert muscle["maximum_attempted_cycle_seam_change_us"] == pytest.approx(200)
    assert muscle["maximum_bound_violation_us"] == pytest.approx(100)


@pytest.mark.parametrize("certification", [
    {"validated_cycles": 0}, {"validated_cycles": 1}, {},
    {"window_statuses": [0, 2, 0], "window_feasibility": [{"passes_tolerance": True}] * 3},
])
def test_no_executed_seam_without_two_consecutive_certified_cycles(certification):
    muscle = _audit(certification)["muscles"]["m"]
    assert muscle["maximum_executed_cycle_seam_change_us"] is None
    assert muscle["maximum_attempted_cycle_seam_change_us"] == pytest.approx(200)


def test_all_successful_cycles_keep_the_previous_executed_seam_value():
    audit = _audit({"window_statuses": [0, 0, 0], "window_feasibility": [
        {"passes_tolerance": True}] * 3, "solver_success": True, "covered_cycles": 3})
    assert audit["muscles"]["m"]["maximum_executed_cycle_seam_change_us"] == pytest.approx(200)


def test_a_fully_certified_multicycle_horizon_includes_its_exported_tail():
    audit = _audit({"window_statuses": [0], "window_feasibility": [
        {"passes_tolerance": True}], "solver_success": True, "covered_cycles": 3})
    assert audit["certified_cycle_count"] == 3
    assert audit["muscles"]["m"]["maximum_executed_cycle_seam_change_us"] == pytest.approx(200)
