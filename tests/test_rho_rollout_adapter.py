import json
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

import cocofest.optimization.rho_rollout_adapter as rollout_adapter_module
from cocofest.optimization.rho_rollout_adapter import (
    _lagrange_derivative_weights,
    _state_midpoints_and_derivatives_from_collocation,
    adapter_completion_status,
    build_rho_endurance_rollout_report,
    fit_nonnegative_periodic_fourier_force_profile,
    select_last_certified_rho_cycle,
    square_periodic_fourier_profile,
    write_rho_endurance_rollout_report,
)
from cocofest.optimization.periodic_force_profile import PeriodicFourierForceProfile


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
    kwargs.setdefault("policy_representation", "fourier")
    kwargs.setdefault("collocation_force_ode_tolerance_n_per_s", 1e6)
    # The synthetic fixture is not a discretized Ding trajectory.  Tests that
    # exercise a strict midpoint-quality rejection override these explicitly.
    kwargs.setdefault("midpoint_force_ode_tolerance_n_per_s", 1e6)
    kwargs.setdefault("midpoint_pulse_width_error_tolerance_s", 1.0)
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


def test_lagrange_derivative_weights_are_exact_for_polynomial_basis():
    nodes = np.array([0.0, 0.1, 0.4, 0.75, 1.0])
    query = 0.37
    weights = _lagrange_derivative_weights(nodes, query)

    for power in range(nodes.size):
        expected = 0.0 if power == 0 else power * query ** (power - 1)
        assert nodes**power @ weights == pytest.approx(expected, abs=2e-13)


def test_radau_midpoint_value_and_time_derivative_use_same_local_polynomial(tmp_path):
    source = tmp_path / "rho.npz"
    _write_archive(source)
    with np.load(source, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files}

    # The synthetic archive spans two seconds/cycles.  A quartic is exactly
    # represented by every degree-3 Radau interval only up to cubic locally,
    # so use a cubic to exercise all basis terms without interpolation error.
    global_time = -payload["states__theta"][0] / (2.0 * np.pi)
    payload["states__F_m1"][0] = 3.0 + 2.0 * global_time - global_time**2 + 0.25 * global_time**3
    np.savez(source, **payload)

    cycle, _ = select_last_certified_rho_cycle(source)
    values, derivatives = _state_midpoints_and_derivatives_from_collocation(
        cycle, "F", ("m1",)
    )
    midpoint_global_time = 1.0 + (np.arange(6) + 0.5) / 6.0
    expected_values = (
        3.0
        + 2.0 * midpoint_global_time
        - midpoint_global_time**2
        + 0.25 * midpoint_global_time**3
    )
    expected_derivatives = 2.0 - 2.0 * midpoint_global_time + 0.75 * midpoint_global_time**2

    np.testing.assert_allclose(values[0], expected_values, rtol=0.0, atol=2e-13)
    np.testing.assert_allclose(derivatives[0], expected_derivatives, rtol=0.0, atol=2e-12)


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
    fidelity = report["adapted_policy_fidelity"]
    assert fidelity["evaluated"] is True
    assert fidelity["passed"] is True
    assert fidelity["status_counts"] == {"ok": 12}
    assert fidelity["approximation_quality"]["passed"] is True
    transcription = report["audits"]["source_transcription"]["force_ode"]
    assert transcription["sample_count"] == 36
    assert transcription["passed"] is True
    assert transcription["role"] == (
        "exact_source_NLP_transcription_gate"
    )
    assert len(report["model_parameters"]) == 2
    alignment = report["audits"]["midpoint_alignment"]
    assert alignment["policy"] == "shared_equal_stimulation_interval_midpoints"
    assert alignment["times_s"] == [pytest.approx((index + 0.5) / 6.0) for index in range(6)]
    assert alignment["all_array_shapes"] == [2, 6]
    assert loaded == report
    assert "NaN" not in output.read_text()


def test_adapter_rejects_force_polynomial_that_violates_ode_at_radau_stages(tmp_path):
    source = tmp_path / "rho.npz"
    reduced_profile = tmp_path / "reduced.npz"
    _write_archive(source)
    reduced_profile.write_bytes(b"test-profile")

    report = _build(
        source,
        reduced_profile,
        collocation_force_ode_tolerance_n_per_s=1e-8,
    )

    assert report["status"] == "rejected"
    assert report["audits"]["source_transcription"]["force_ode"]["passed"] is False
    assert "source_collocation_force_ode_residual_too_large" in {
        reason["code"] for reason in report["rejection_reasons"]
    }


def test_midpoint_approximation_has_distinct_configurable_quality_gate(tmp_path):
    source = tmp_path / "rho.npz"
    reduced_profile = tmp_path / "reduced.npz"
    _write_archive(source)
    reduced_profile.write_bytes(b"test-profile")

    report = _build(
        source,
        reduced_profile,
        collocation_force_ode_tolerance_n_per_s=1e6,
        midpoint_force_ode_tolerance_n_per_s=1e-8,
        midpoint_pulse_width_error_tolerance_s=1.0,
    )

    assert report["status"] == "midpoint_approximation_out_of_tolerance"
    assert report["rejection_reasons"] == []
    assert report["audits"]["source_transcription"]["force_ode"]["passed"] is True
    quality = report["adapted_policy_fidelity"]["approximation_quality"]
    assert quality["passed"] is False
    assert quality["midpoint_force_ode"]["passed"] is False
    assert quality["inferred_vs_exported_pulse_width"]["passed"] is True

    pulse_width_report = _build(
        source,
        reduced_profile,
        collocation_force_ode_tolerance_n_per_s=1e6,
        midpoint_force_ode_tolerance_n_per_s=1e6,
        midpoint_pulse_width_error_tolerance_s=1e-12,
    )
    pulse_width_quality = pulse_width_report["adapted_policy_fidelity"][
        "approximation_quality"
    ]
    assert pulse_width_report["status"] == "midpoint_approximation_out_of_tolerance"
    assert pulse_width_quality["midpoint_force_ode"]["passed"] is True
    assert pulse_width_quality["inferred_vs_exported_pulse_width"]["passed"] is False


def test_adapted_policy_fidelity_detects_divergence_from_local_collocation(
    tmp_path, monkeypatch
):
    source = tmp_path / "rho.npz"
    reduced_profile = tmp_path / "reduced.npz"
    _write_archive(source)
    reduced_profile.write_bytes(b"test-profile")
    original_fit = rollout_adapter_module.fit_nonnegative_periodic_fourier_force_profile

    def distorted_fit(*args, **kwargs):
        profile = original_fit(*args, **kwargs)
        return PeriodicFourierForceProfile(
            period=profile.period,
            mean=profile.mean + 1e6,
            cosine=profile.cosine,
            sine=profile.sine,
        )

    monkeypatch.setattr(
        rollout_adapter_module,
        "fit_nonnegative_periodic_fourier_force_profile",
        distorted_fit,
    )
    report = _build(
        source,
        reduced_profile,
        fourier_relative_rmse_tolerance=1e9,
        fourier_relative_maximum_tolerance=1e9,
        midpoint_force_ode_tolerance_n_per_s=1e6,
        midpoint_pulse_width_error_tolerance_s=1.0,
        midpoint_inverse_coverage_minimum=0.01,
    )

    local = report["audits"]["source_midpoint_interpolation"]
    adapted = report["adapted_policy_fidelity"]
    assert local["approximation_quality"]["midpoint_force_ode"]["passed"] is True
    assert adapted["approximation_quality"]["midpoint_force_ode"]["passed"] is False
    assert local["policy_source"] != adapted["policy_source"]
    assert report["status"] == "midpoint_approximation_out_of_tolerance"


def test_adapter_reports_uncertified_nonperiodic_source_without_forcing_rollout(tmp_path):
    source = tmp_path / "bad-rho.npz"
    reduced_profile = tmp_path / "reduced.npz"
    _write_archive(source, certified=False, break_periodicity=True)
    reduced_profile.write_bytes(b"test-profile")

    report = _build(source, reduced_profile)
    codes = {reason["code"] for reason in report["rejection_reasons"]}

    assert report["status"] == "rejected"
    assert report["horizons"] == []
    assert report["adapted_policy_fidelity"] == {
        "evaluated": False,
        "passed": None,
        "reason": "input_adaptation_rejected",
    }
    assert report["reference_policy"]["deprecated_alias_of"] == "adapted_policy_fidelity"
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

    report = _build(
        source,
        reduced_profile,
        midpoint_inverse_coverage_minimum=0.01,
    )

    reference = report["audits"]["source_midpoint_interpolation"]
    assert report["status"] == "complete"
    assert report["rejection_reasons"] == []
    assert reference["evaluated"] is True
    assert reference["passed"] is True
    assert reference["approximation_quality"]["passed"] is True
    assert reference["failure_count"] > 0
    assert reference["all_midpoint_inversions_feasible"] is False
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
    assert adapter_completion_status([], midpoint_approximation_acceptable=True) == "complete"
    assert (
        adapter_completion_status([], midpoint_approximation_acceptable=False)
        == "midpoint_approximation_out_of_tolerance"
    )
    assert (
        adapter_completion_status(
            [{"code": "input"}],
            midpoint_approximation_acceptable=False,
        )
        == "rejected"
    )


def test_cli_returns_two_for_midpoint_approximation_out_of_tolerance(tmp_path, monkeypatch):
    from scripts import analyze_rho_endurance_rollout as cli

    monkeypatch.setattr(
        cli,
        "build_rho_endurance_rollout_report",
        lambda *args, **kwargs: {
            "schema": "test",
            "status": "midpoint_approximation_out_of_tolerance",
            "adapted_policy_fidelity": {"evaluated": True, "passed": False},
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
        "midpoint_approximation_out_of_tolerance"
    )


def test_collocation_policy_retains_source_phases_and_exports_fixed_size_parameters(tmp_path):
    source = tmp_path / "rho.npz"
    reduced_profile = tmp_path / "reduced.npz"
    _write_archive(source)
    reduced_profile.write_bytes(b"test-profile")
    report = _build(source, reduced_profile, policy_representation="collocation")
    assert report["status"] == "complete"
    assert report["configuration"]["force_polynomial_coefficient_shape"] == [2, 6, 4]
    assert report["configuration"]["force_resulting_harmonics"] is None
    assert "force_fourier" not in report["audits"]
    for error in report["audits"]["policy_vs_source_collocation_midpoints"]["errors"].values():
        assert error["maximum_absolute_error"] < 1e-11
    assert report["adapted_policy_fidelity"]["policy_source"] == "periodic_collocation_policy_used_by_endurance_rollout"
    packed = report["rollout_objective_profile"]
    assert packed["parameter_size"] == len(packed["parameters"]) == 9 * 2 * 6
    assert packed["source_policy_gate_passed"] is True
    assert "future_feasibility_not_certified" in packed["certification_scope"]
    assert report["rollout_outcome"]["all_requested_horizons_feasible"] == all(row["feasible"] for row in report["horizons"])


def test_collocation_policy_rejects_hidden_negative_overshoot(tmp_path):
    source = tmp_path / "rho.npz"
    reduced_profile = tmp_path / "reduced.npz"
    _write_archive(source)
    reduced_profile.write_bytes(b"test-profile")
    with np.load(source, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    # Positive shooting/stage values, negative between the first two stages.
    nodes = np.r_[0., ca.collocation_points(3, "radau")]
    payload["states__F_m1"][0, 24:28] = 100. * (nodes - .37)**2 - .1
    np.savez(source, **payload)
    report = _build(source, reduced_profile, policy_representation="collocation")
    assert report["status"] == "rejected"
    assert "negative_continuous_force" in {reason["code"] for reason in report["rejection_reasons"]}
    assert report["audits"]["force_policy"]["clipping_applied"] is False
    assert "rollout_objective_profile" not in report
