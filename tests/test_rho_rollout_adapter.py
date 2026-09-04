import json
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from cocofest.optimization.rho_rollout_adapter import (
    adapter_completion_status,
    build_rho_endurance_rollout_report,
    fit_nonnegative_periodic_fourier_force_profile,
    select_last_certified_rho_cycle,
    square_periodic_fourier_profile,
    write_rho_endurance_rollout_report,
)


MUSCLES = ("m1", "m2")


class _ReducedDynamics:
    muscle_names = MUSCLES
    muscle_geometry = object()
    kinematics = SimpleNamespace(direction=-1)

    @staticmethod
    def muscle_relationships(theta, omega):
        assert np.isfinite(theta)
        assert np.isfinite(omega)
        return np.array([0.90, 0.95]), np.array([1.0, 0.98]), np.array([0.02, 0.01])


def _models():
    return [
        SimpleNamespace(
            muscle_name=name,
            a_scale=4920.0 + 100.0 * index,
            tau1_rest=0.060601,
            km_rest=0.137,
            alpha_a=-0.04,
            alpha_tau1=2.1e-6,
            alpha_km=1.9e-6,
            tau_fat=127.0,
            tau2=0.001,
            pd0=0.000131405,
            pdt=0.000194138,
        )
        for index, name in enumerate(MUSCLES)
    ]


def _write_archive(path, *, certified=True, break_periodicity=False, force_scale=1.0):
    cycles = 2
    stimulations = 6
    degree = 3
    points = np.asarray(ca.collocation_points(degree, "radau"))
    times = [0.0]
    for interval in range(cycles * stimulations):
        times.extend((interval + points) / stimulations)
        times.append((interval + 1.0) / stimulations)
    times = np.asarray(times)
    local_time = np.mod(times, 1.0)
    payload = {
        "states__theta": (-2.0 * np.pi * times)[None, :],
        "states__omega": np.full((1, times.size), -2.0 * np.pi),
    }
    for index, name in enumerate(MUSCLES):
        payload[f"states__F_{name}"] = (
            force_scale
            * (8.0 + index + (1.0 + 0.2 * index) * np.cos(2.0 * np.pi * local_time))
        )[None, :]
        payload[f"states__Cn_{name}"] = (
            0.40 + 0.02 * np.sin(2.0 * np.pi * local_time)
        )[None, :]
        payload[f"states__A_{name}"] = np.full((1, times.size), 4800.0 + 100.0 * index)
        payload[f"states__Tau1_{name}"] = np.full((1, times.size), 0.065)
        payload[f"states__Km_{name}"] = np.full((1, times.size), 0.145)
        payload[f"controls__last_pulse_width_{name}"] = np.full(
            (1, cycles * stimulations), 0.0003
        )
    if break_periodicity:
        payload["states__F_m2"][0, -1] += 4.0
        payload["states__Cn_m1"][0, -1] += 0.1
    metadata = {
        "schema": "test-rho",
        "producer_mode": "receding_horizon_concatenation" if certified else "common_seed",
        "cycles_per_window": cycles,
        "stimulations_per_cycle": stimulations,
        "producer_collocation_degree": degree,
        "producer_collocation_method": "radau",
        "cycle_duration_s": 1.0,
        "model_formulation": "periodic_node",
        "pulse_width_maximum_s": 0.0006,
        "activate_force_length_relationship": True,
        "activate_force_velocity_relationship": True,
        "activate_passive_force_relationship": True,
        "terminal_wheel_q_slack": 1e-6,
    }
    payload["metadata__json"] = np.asarray(json.dumps(metadata))
    np.savez(path, **payload)


def _build(source, reduced_profile, **kwargs):
    return build_rho_endurance_rollout_report(
        source,
        reduced_profile,
        horizons=(1, 2),
        force_harmonics=2,
        cn_harmonics=2,
        kinematic_harmonics=2,
        muscle_models=_models(),
        reduced_dynamics=_ReducedDynamics(),
        **kwargs,
    )


def test_last_cycle_selection_uses_declared_collocation_layout(tmp_path):
    source = tmp_path / "rho.npz"
    _write_archive(source)

    selected, period_basis = select_last_certified_rho_cycle(source)

    assert selected.certified
    assert selected.cycle_index == 1
    assert selected.collocation_degree == 3
    assert selected.stimulations_per_cycle == 6
    assert selected.state_columns_per_interval == 4
    assert selected.sample_times.size == 18
    assert selected.start_column == 24
    assert selected.end_column == 48
    assert period_basis == "metadata.cycle_duration_s"


def test_squared_root_fourier_conversion_is_exact_and_nonnegative():
    time = np.linspace(0.0, 1.0, 41, endpoint=False)
    force = np.vstack(
        (
            (2.0 + 0.4 * np.cos(2.0 * np.pi * time) - 0.2 * np.sin(4.0 * np.pi * time)) ** 2,
            (1.0 + 0.1 * np.sin(2.0 * np.pi * time)) ** 2,
        )
    )
    fitted = fit_nonnegative_periodic_fourier_force_profile(
        force,
        time,
        period=1.0,
        root_harmonic_count=2,
    )
    query = np.linspace(-0.5, 1.5, 501)

    expected = np.vstack(
        (
            (2.0 + 0.4 * np.cos(2.0 * np.pi * query) - 0.2 * np.sin(4.0 * np.pi * query)) ** 2,
            (1.0 + 0.1 * np.sin(2.0 * np.pi * query)) ** 2,
        )
    )
    np.testing.assert_allclose(fitted.evaluate(query), expected, rtol=0.0, atol=2e-14)
    assert fitted.harmonic_count == 4
    assert np.min(fitted.evaluate(query)) >= 0.0

    # The public algebraic converter is the exact square, not a refit.
    root = fit_nonnegative_periodic_fourier_force_profile(
        np.sqrt(force), time, period=1.0, root_harmonic_count=2
    )
    squared_again = square_periodic_fourier_profile(root)
    np.testing.assert_allclose(squared_again.evaluate(query), root.evaluate(query) ** 2, atol=2e-14)


def test_adapter_builds_compact_horizons_from_a_valid_periodic_cycle(tmp_path):
    source = tmp_path / "rho.npz"
    reduced_profile = tmp_path / "reduced.npz"
    output = tmp_path / "rollout.json"
    _write_archive(source)
    reduced_profile.write_bytes(b"test-profile")

    report = _build(source, reduced_profile)
    write_rho_endurance_rollout_report(output, report)
    loaded = json.loads(output.read_text())

    assert report["status"] == "complete"
    assert report["rejection_reasons"] == []
    assert [row["horizon_cycles"] for row in report["horizons"]] == [1, 2]
    assert report["selection"]["selected_cycle_index"] == 1
    assert report["selection"]["fourier_sample_count"] == 18
    assert report["configuration"]["force_sqrt_harmonics"] == 2
    assert report["configuration"]["force_resulting_harmonics"] == 4
    assert report["audits"]["force_fourier"]["fit_space"] == "sqrt_force"
    assert report["audits"]["force_fourier"]["clipping_applied"] is False
    assert report["reference_policy"]["evaluated"] is True
    assert report["reference_policy"]["reproducible"] is True
    assert report["reference_policy"]["status_counts"] == {"ok": 12}
    assert len(report["model_parameters"]) == 2
    alignment = report["audits"]["midpoint_alignment"]
    assert alignment["policy"] == "shared_equal_stimulation_interval_midpoints"
    assert alignment["times_s"] == [pytest.approx((index + 0.5) / 6.0) for index in range(6)]
    assert alignment["all_array_shapes"] == [2, 6]
    assert loaded == report
    assert "NaN" not in output.read_text()


def test_adapter_reports_uncertified_nonperiodic_source_without_forcing_rollout(tmp_path):
    source = tmp_path / "bad-rho.npz"
    reduced_profile = tmp_path / "reduced.npz"
    _write_archive(source, certified=False, break_periodicity=True)
    reduced_profile.write_bytes(b"test-profile")

    report = _build(source, reduced_profile)
    codes = {reason["code"] for reason in report["rejection_reasons"]}

    assert report["status"] == "rejected"
    assert report["horizons"] == []
    assert report["reference_policy"] == {
        "evaluated": False,
        "reproducible": None,
        "reason": "input_adaptation_rejected",
    }
    assert "source_not_certified" in codes
    assert "non_periodic_force" in codes
    assert "non_periodic_cn" in codes
    assert report["audits"]["force_endpoint_periodicity"]["difference"][1] == 4.0
    assert report["audits"]["cn_endpoint_periodicity"]["difference"][0] == pytest.approx(0.1)


def test_small_absolute_force_seam_and_solver_scale_pw_residual_are_audited_not_clipped(tmp_path):
    source = tmp_path / "rho.npz"
    reduced_profile = tmp_path / "reduced.npz"
    _write_archive(source)
    reduced_profile.write_bytes(b"test-profile")
    with np.load(source, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    payload["states__F_m2"][0, -1] += 5e-4
    payload["controls__last_pulse_width_m1"][0, -1] = 0.000131405 - 5e-11
    np.savez(source, **payload)

    report = _build(source, reduced_profile)

    assert report["status"] == "complete"
    assert report["audits"]["force_endpoint_periodicity"]["difference"][1] == pytest.approx(5e-4)
    assert report["audits"]["pulse_width_controls"]["m1"]["minimum_s"] == pytest.approx(
        0.000131405 - 5e-11
    )
    assert report["configuration"]["thresholds"]["pulse_width_bound_tolerance_s"] == 1e-10


def test_reference_gate_uses_source_cycle_collocation_states_and_keeps_horizons(tmp_path):
    source = tmp_path / "rho.npz"
    reduced_profile = tmp_path / "reduced.npz"
    _write_archive(source, force_scale=30.0)
    reduced_profile.write_bytes(b"test-profile")
    with np.load(source, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    global_time = -payload["states__theta"][0] / (2.0 * np.pi)
    payload["states__A_m1"][0] = 4700.0 + 7.0 * global_time + 2.0 * global_time**2
    np.savez(source, **payload)

    report = _build(source, reduced_profile)

    reference = report["reference_policy"]
    assert report["status"] == "reference_policy_not_reproducible"
    assert report["rejection_reasons"] == []
    assert reference["evaluated"] is True
    assert reference["reproducible"] is False
    assert reference["failure_count"] > 0
    assert reference["first_failure"]["source_cycle_index"] == 1
    assert reference["first_failure"]["status"] in {
        "recruitment_exceeds_capacity",
        "pulse_width_limit_exceeded",
    }
    midpoint_global_time = 1.0 + (np.arange(6) + 0.5) / 6.0
    expected_a = 4700.0 + 7.0 * midpoint_global_time + 2.0 * midpoint_global_time**2
    assert reference["source_slow_state_ranges"]["A"]["minimum"] == pytest.approx(
        expected_a.min()
    )
    assert reference["source_slow_state_ranges"]["A"]["maximum"] == pytest.approx(
        4900.0
    )
    assert len(report["horizons"]) == 2


def test_completion_status_does_not_depend_on_future_rollout_feasibility():
    assert adapter_completion_status([], reference_policy_reproducible=True) == "complete"
    assert (
        adapter_completion_status([], reference_policy_reproducible=False)
        == "reference_policy_not_reproducible"
    )
    assert (
        adapter_completion_status(
            [{"code": "input"}],
            reference_policy_reproducible=False,
        )
        == "rejected"
    )


def test_cli_returns_two_for_nonreproducible_reference_policy(tmp_path, monkeypatch):
    from scripts import analyze_rho_endurance_rollout as cli

    monkeypatch.setattr(
        cli,
        "build_rho_endurance_rollout_report",
        lambda *args, **kwargs: {
            "schema": "test",
            "status": "reference_policy_not_reproducible",
            "reference_policy": {"evaluated": True, "reproducible": False},
            "horizons": [{"horizon_cycles": 5, "feasible": False}],
        },
    )
    exit_code = cli.main(
        [
            "--source",
            str(tmp_path / "source.npz"),
            "--reduced-profile",
            str(tmp_path / "profile.npz"),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert exit_code == 2
    assert json.loads((tmp_path / "report.json").read_text())["status"] == (
        "reference_policy_not_reproducible"
    )
