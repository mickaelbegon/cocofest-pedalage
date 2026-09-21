"""Checks for the coupled RHS and archive/transcription diagnostic."""

import numpy as np

from cocofest.models.ding2007.ding2007_with_fatigue_periodic_node import (
    DingModelPulseWidthFrequencyWithFatiguePeriodicNode,
)
from cocofest.optimization.solver_cross_rollout import RolloutSource, _physical_rhs, evaluate_rollout
from scripts.diagnose_archived_replay_fidelity import diagnose, infer_archive_alpha_a


class SimpleMechanics:
    muscle_names = ("test",)

    def acceleration(self, theta, omega, forces, external_crank_torque=0.):
        return external_crank_torque

    def coefficient_values(self, theta):
        return {"external_torque_effectiveness": 1.}


def source_and_model():
    model = DingModelPulseWidthFrequencyWithFatiguePeriodicNode(
        muscle_name="test", stim_interval=.01, sum_stim_truncation=6
    )
    names = ("tauc", "a_scale", "alpha_a", "tau_fat", "alpha_tau1", "alpha_km",
             "tau1_rest", "km_rest", "tau2", "pd0", "pdt")
    pars = {key: float(getattr(model, key)) for key in names}
    pars["Fmax"] = 248.
    metadata = dict(formulation="dynamic", ding_sum_stim_truncation=6,
                    signed_crank_torque_nm=.1, pulse_width_maximum_s=.0006,
                    activate_force_length_relationship=False,
                    activate_force_velocity_relationship=False,
                    activate_passive_force_relationship=False)
    initial = np.array([.3, 10., .95*model.a_scale, model.tau1_rest, model.km_rest, 0., -6.])
    source = RolloutSource(metadata, {}, ("test",), {"test": pars},
                           np.full((1, 2), .0003), np.tile(initial[:, None], (1, 3)),
                           ("Cn_test", "F_test", "A_test", "Tau1_test", "Km_test", "theta", "omega"),
                           .02, 2, 0, 1)
    return source, model


def test_coupled_ding_rhs_matches_ocp_model_with_the_same_forcing():
    from casadi import DM

    source, model = source_and_model()
    rhs = _physical_rhs(source, SimpleMechanics(), source.controls[:, 0])
    for local_time in (0., .003, .01):
        expected = np.asarray(model.system_dynamics(
            states=DM(source.initial_state[:5]), controls=DM(source.controls[:, 0]),
            time=DM([local_time]),
            numerical_timeseries=DM([model.post_stimulation_amplitude(), 0.]),
        )).ravel()
        np.testing.assert_allclose(rhs(local_time, source.initial_state)[:5], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rhs(.003, source.initial_state)[-2:], [-6., .1])


def test_diagnostic_exposes_archive_node_corruption_without_changing_common_rhs():
    source, _ = source_and_model()
    _, reference = evaluate_rollout(source, SimpleMechanics())
    source.shooting_states = reference["shooting_states"].copy()
    clean = diagnose(source, SimpleMechanics(), continuous=False)
    assert clean["local_one_interval_defects"]["dop853"]["maximum_scaled"] < 1e-10
    source.shooting_states[-2, 1] += .2
    corrupt = diagnose(source, SimpleMechanics(), continuous=False)
    errors = corrupt["local_one_interval_defects"]["dop853"]["maximum_absolute_by_state"]
    assert abs(errors["theta"]-.2) < 1e-10
    assert errors["omega"] < 1e-10


def test_saved_collocation_fatigue_equation_identifies_factor_ten_parameter_mismatch(tmp_path):
    from cocofest.optimization.solver_cross_rollout import collocation_tableau

    source, _ = source_and_model()
    nodes, _, _, _ = collocation_tableau("radau5")
    duration = .01
    times = np.r_[0., nodes, 1.]*duration
    p = source.parameters["test"]
    true_alpha, force = -.8, 20.
    equilibrium = p["a_scale"]+p["tau_fat"]*true_alpha*force
    capacity = equilibrium+(p["a_scale"]-equilibrium)*np.exp(-times/p["tau_fat"])
    archive = tmp_path/"known-fatigue.npz"
    np.savez(archive, states__A_test=capacity, states__F_test=np.full(7, force))
    source.provenance = {"state_stride": 6, "source": {"path": str(archive)}}
    source.duration, source.intervals_per_cycle = duration, 1
    source.controls = source.controls[:, :1]
    p["alpha_a"] = true_alpha/10
    result = infer_archive_alpha_a(source)["muscles"]["test"]
    assert abs(result["inferred_alpha_a"]-true_alpha) < 1e-9
    assert result["maximum_capacity_equation_defect_inferred"] < 1e-11
    assert result["maximum_capacity_equation_defect_declared"] > .14
