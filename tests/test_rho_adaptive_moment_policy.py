import json
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from cocofest.optimization.rho_adaptive_moment_policy import (
    build_rho_adaptive_moment_policy,
)


MUSCLES = ("m1", "m2")


class _Model(SimpleNamespace):
    def post_stimulation_amplitude(self):
        return 1.05


class _Reduced:
    muscle_names = MUSCLES
    muscle_geometry = object()

    @staticmethod
    def coefficient_values(theta):
        return {"muscle_effectiveness": np.array([0.05, -0.04])}

    @staticmethod
    def muscle_relationships(theta, omega):
        return np.array([0.9, 0.95]), np.array([1.0, 0.98]), np.array([0.02, 0.01])


def _models():
    return tuple(
        _Model(
            muscle_name=name,
            a_scale=4900.0 + 100.0 * index,
            tau1_rest=0.060601,
            km_rest=0.137,
            alpha_a=-0.04,
            alpha_tau1=2.1e-6,
            alpha_km=1.9e-6,
            tau_fat=127.0,
            tauc=0.011,
            tau2=0.001,
            pd0=0.000131405,
            pdt=0.000194138,
        )
        for index, name in enumerate(MUSCLES)
    )


def _write_source(path, *, calcium_formulation="exact_exponential_periodic_node"):
    stimulations = 3
    degree = 3
    nodes = np.asarray(ca.collocation_points(degree, "radau"))
    time = [0.0]
    for interval in range(stimulations):
        time.extend((interval + nodes) / stimulations)
        time.append((interval + 1.0) / stimulations)
    time = np.asarray(time)
    payload = {
        "states__theta": (-2.0 * np.pi * time)[None, :],
        "states__omega": np.full((1, time.size), -2.0 * np.pi),
    }
    for muscle_index, name in enumerate(MUSCLES):
        payload[f"states__Cn_{name}"] = np.full((1, time.size), 0.2)
        payload[f"states__F_{name}"] = (
            20.0 + muscle_index + 2.0 * np.sin(2.0 * np.pi * time)
        )[None, :]
        payload[f"states__A_{name}"] = np.full((1, time.size), 4800.0)
        payload[f"states__Tau1_{name}"] = np.full((1, time.size), 0.065)
        payload[f"states__Km_{name}"] = np.full((1, time.size), 0.145)
        payload[f"controls__last_pulse_width_{name}"] = np.full(
            (1, stimulations), 0.0003 + muscle_index * 1e-5
        )
    payload["metadata__json"] = np.asarray(
        json.dumps(
            {
                "producer_mode": "receding_horizon_concatenation",
                "cycles_per_window": 1,
                "stimulations_per_cycle": stimulations,
                "producer_collocation_degree": degree,
                "producer_collocation_method": "radau",
                "cycle_duration_s": 1.0,
                "model_formulation": "periodic_node",
                "calcium_forcing_formulation": calcium_formulation,
                "mechanical_formulation": "reduced",
                "pulse_width_maximum_s": 0.0006,
                "activate_force_length_relationship": True,
                "activate_force_velocity_relationship": True,
                "activate_passive_force_relationship": True,
            }
        )
    )
    np.savez(path, **payload)


def test_builder_retains_individual_moment_targets_and_source_pw(tmp_path):
    source = tmp_path / "rho.npz"
    _write_source(source)

    policy = build_rho_adaptive_moment_policy(
        source,
        tmp_path / "unused-reduced.npz",
        muscle_models=_models(),
        reduced_dynamics=_Reduced(),
    )

    assert policy.muscle_names == MUSCLES
    assert len(policy.intervals) == 3
    np.testing.assert_allclose(policy.source_pulse_widths[0], 0.0003)
    np.testing.assert_allclose(policy.source_pulse_widths[1], 0.00031)
    for interval_index, interval in enumerate(policy.intervals):
        np.testing.assert_allclose(
            interval.target_moments,
            policy.target_moments[:, interval_index],
        )
        assert interval.mechanical_gains[0](interval.duration / 2.0) == pytest.approx(0.92)
        assert interval.mechanical_gains[1](interval.duration / 2.0) == pytest.approx(0.941)


def test_builder_rejects_a_calcium_law_that_differs_from_the_rollout(tmp_path):
    source = tmp_path / "rho.npz"
    _write_source(source, calcium_formulation="truncated_history")

    with pytest.raises(ValueError, match="calcium_forcing_formulation"):
        build_rho_adaptive_moment_policy(
            source,
            tmp_path / "unused-reduced.npz",
            muscle_models=_models(),
            reduced_dynamics=_Reduced(),
        )
