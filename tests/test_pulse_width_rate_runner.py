from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from examples.fes_multibody.cycling import (
    cycling_pulse_width_mhe_acados_periodic as runner,
)
from scripts import benchmark_pulse_width_rate_state as benchmark


def test_rate_state_cli_and_signature_are_explicit():
    parser = runner.build_argument_parser()
    direct = parser.parse_args([])
    rate = parser.parse_args(
        [
            "--pulse-width-control-mode",
            "rate_state",
            "--pulse-width-max-rate-us-per-s",
            "3000",
        ]
    )

    assert runner.pulse_width_rate_signature(direct) == {
        "pulse_width_control_mode": "direct",
        "pulse_width_control_representation": "direct_zoh_v1",
        "pulse_width_max_rate_us_per_s": None,
    }
    assert runner.pulse_width_rate_signature(rate) == {
        "pulse_width_control_mode": "rate_state",
        "pulse_width_control_representation": "pw_state_rate_control_zoh_v1",
        "pulse_width_max_rate_us_per_s": 3000.0,
    }


def test_rate_state_audit_separates_applied_pw_from_rate_controls():
    args = argparse.Namespace(
        pulse_width_control_mode="rate_state",
        pulse_width_max_rate_us_per_s=3000.0,
        pulse_width_max_rate_s_per_s=0.003,
        ode_solver="collocation",
        collocation_degree=1,
        stimulations_per_cycle=2,
        formulation="dynamic",
    )
    pw = np.array([[100e-6, 125e-6, 150e-6, 175e-6, 200e-6]])
    rate = np.array([[100e-6, 100e-6]])
    summary = {
        "state_traces": {"last_pulse_width_Biceps": pw},
        "control_traces": {"pulse_width_rate_Biceps": rate},
        "initial_guess_state_traces": {"last_pulse_width_Biceps": pw},
        "initial_guess_control_traces": {"pulse_width_rate_Biceps": rate},
    }

    runner.attach_pulse_width_command_audit(summary, args)

    np.testing.assert_allclose(
        summary["applied_pulse_width_traces"]["last_pulse_width_Biceps"],
        [[100e-6, 150e-6]],
    )
    assert "last_pulse_width_Biceps" not in summary["control_traces"]
    audit = summary["pulse_width_command"]["rate_bound_audit"]
    assert audit["passes_bound"] is True
    assert audit["maximum_observed_applied_increase_us"] == pytest.approx(50.0)
    assert summary["pulse_width_command"]["applied_trace_source"] == (
        "state_at_shooting_node"
    )


class _Solution:
    def decision_states(self, **_):
        return {"last_pulse_width_Biceps": np.array([[100e-6, 150e-6]])}

    def decision_controls(self, **_):
        return {"pulse_width_rate_Biceps": np.array([[50e-6]])}


def test_npz_export_uses_a_distinct_applied_pulse_width_namespace(tmp_path):
    path = tmp_path / "rate-state.npz"
    runner._save_warmup_cache(
        path,
        _Solution(),
        metadata={"pulse_width_control_mode": "rate_state"},
        applied_pulse_widths={
            "last_pulse_width_Biceps": np.array([[100e-6]])
        },
    )

    with np.load(path, allow_pickle=False) as data:
        assert "states__last_pulse_width_Biceps" in data.files
        assert "controls__pulse_width_rate_Biceps" in data.files
        assert "controls__last_pulse_width_Biceps" not in data.files
        assert "applied_pulse_widths__last_pulse_width_Biceps" in data.files
        metadata = json.loads(str(data["metadata__json"].item()))
    assert metadata["pulse_width_control_mode"] == "rate_state"


def test_benchmark_arms_differ_only_by_rate_representation(tmp_path):
    common = dict(
        reduced_profile=tmp_path / "profile.npz",
        n_windows=2,
        stimulations_per_cycle=30,
        maximum_rate_us_per_s=3000.0,
        maximum_iterations=100,
        n_threads=1,
        hsl_library=None,
    )
    direct = benchmark.build_arm_cli(
        mode="direct", solution_output=tmp_path / "direct.npz", **common
    )
    rate = benchmark.build_arm_cli(
        mode="rate_state", solution_output=tmp_path / "rate.npz", **common
    )

    assert "--ipopt-linear-solver" in direct
    assert direct[direct.index("--ipopt-linear-solver") + 1] == "ma57"
    assert "--objective" in direct
    assert direct[direct.index("--objective") + 1] == "fatigue"
    assert "--pulse-width-max-rate-us-per-s" not in direct
    assert direct[direct.index("--pulse-width-max-step-us") + 1] == "100.0"
    assert rate[rate.index("--pulse-width-max-rate-us-per-s") + 1] == "3000.0"
