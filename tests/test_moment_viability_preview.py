import json
from dataclasses import replace

import numpy as np
import pytest

from cocofest.optimization.adaptive_moment_rollout import (
    DingPulseWidthParameters,
    MomentTrackingInterval,
    propagate_ding_pulse_width_interval,
)
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.moment_viability_preview import (
    CompactMomentViabilityPolicy,
    FiveCycleViabilityConfig,
    MechanicalGainDomainError,
    audit_policy_mechanical_gains,
    run_five_cycle_viability_preview,
)


def _muscle(capacity=1200.0):
    return DingPulseWidthParameters(
        fatigue=DingFatigueParameters(
            a_rest=capacity,
            tau1_rest=0.060601,
            km_rest=0.137,
            alpha_a=-0.4,
            alpha_tau1=2.1e-5,
            alpha_km=1.9e-5,
            tau_fat=127.0,
        ),
        tauc=0.011,
        tau2=0.001,
        pd0=0.000131405,
        pdt=0.000194138,
        pulse_width_max=0.0006,
    )


def _initial(muscle, force):
    return np.asarray(
        [
            0.1629821583533315,
            force,
            0.98 * muscle.fatigue.a_rest,
            1.02 * muscle.fatigue.tau1_rest,
            1.02 * muscle.fatigue.km_rest,
        ]
    )


def _reproducible_policy():
    parameters = (_muscle(), _muscle(1100.0))
    initial = np.vstack((_initial(parameters[0], 25.0), _initial(parameters[1], 22.0)))
    pulse_widths = np.asarray([[0.00034], [0.00036]])
    duration = 1.0 / 30.0
    amplitude = (1.0597355478114694, 1.0597355478114694)
    gains = (0.96, 0.90)
    coefficients = (0.05, -0.04)
    targets = []
    for index, parameter in enumerate(parameters):
        endpoint = propagate_ding_pulse_width_interval(
            initial[index],
            pulse_width=pulse_widths[index, 0],
            duration=duration,
            calcium_amplitude=amplitude[index],
            mechanical_gain=gains[index],
            parameters=parameter,
            integration_substeps=8,
        )
        targets.append(coefficients[index] * endpoint[1])
    interval = MomentTrackingInterval(
        duration=duration,
        calcium_amplitudes=amplitude,
        mechanical_gains=gains,
        moment_coefficients=coefficients,
        target_moments=tuple(targets),
    )
    return (
        CompactMomentViabilityPolicy(
            intervals=(interval,),
            parameters=parameters,
            source_pulse_widths=pulse_widths,
            muscle_names=("agonist", "antagonist"),
        ),
        initial,
    )


def test_five_cycle_preview_compares_three_policies_with_full_ding_states():
    policy, initial = _reproducible_policy()
    result = run_five_cycle_viability_preview(
        policy,
        initial_states=initial,
        config=FiveCycleViabilityConfig(horizon_cycles=5, integration_substeps=8),
    )

    assert result.feasible
    assert result.individual_moment.status == "complete"
    assert result.total_moment.status == "complete"
    assert result.fixed is not None
    assert result.total_moment.state_history.shape == (6, 2, 5)
    assert result.individual_moment.state_history.shape == (6, 2, 5)
    assert result.future_pulse_width_decision_count == 0
    assert result.explicit_horizon_pulse_width_count == 10
    assert policy.reference_parameter_count == 2
    assert result.maximum_absolute_individual_moment_error < 1e-8
    assert result.maximum_absolute_total_policy_moment_error < 1e-8
    assert result.maximum_absolute_fixed_moment_error > 1e-6
    assert 0.0 <= result.maximum_pulse_width_utilization <= 1.0
    assert result.minimum_capacity_ratio > 0.0
    assert result.fixed_elapsed_time_s >= 0.0
    assert result.individual_elapsed_time_s >= 0.0
    assert result.total_elapsed_time_s >= 0.0
    json.dumps(result.summary())


def test_total_moment_policy_can_redistribute_when_individual_tracking_is_infeasible():
    parameters = (_muscle(), _muscle(1100.0))
    initial = np.vstack((_initial(parameters[0], 35.0), _initial(parameters[1], 35.0)))
    duration = 1.0 / 30.0
    gains = (0.96, 0.90)
    coefficients = (0.05, 0.05)
    bounds = np.empty((2, 2))
    for muscle_index, parameter in enumerate(parameters):
        endpoint_moments = []
        for pulse_width in (parameter.pd0, parameter.pulse_width_max):
            endpoint = propagate_ding_pulse_width_interval(
                initial[muscle_index],
                pulse_width=pulse_width,
                duration=duration,
                calcium_amplitude=1.0597355478114694,
                mechanical_gain=gains[muscle_index],
                parameters=parameter,
                integration_substeps=8,
            )
            endpoint_moments.append(coefficients[muscle_index] * endpoint[1])
        bounds[muscle_index] = sorted(endpoint_moments)
    required_total = float(np.sum(np.mean(bounds, axis=1)))
    targets = (bounds[0, 1] + 0.01, required_total - bounds[0, 1] - 0.01)
    interval = MomentTrackingInterval(
        duration=duration,
        calcium_amplitudes=(1.0597355478114694, 1.0597355478114694),
        mechanical_gains=gains,
        moment_coefficients=coefficients,
        target_moments=targets,
    )
    policy = CompactMomentViabilityPolicy(
        intervals=(interval,),
        parameters=parameters,
        source_pulse_widths=np.full((2, 1), 0.00035),
    )

    result = run_five_cycle_viability_preview(
        policy,
        initial_states=initial,
        config=FiveCycleViabilityConfig(horizon_cycles=1, integration_substeps=8),
    )

    assert result.individual_moment.status == "infeasible"
    assert result.individual_moment.completed_cycles == 0
    assert result.total_moment.status == "complete"
    assert result.total_policy_total_moments[0, 0] == pytest.approx(required_total, abs=1e-8)
    assert not np.allclose(result.total_moment.allocated_moments[0, :, 0], targets)


def test_policy_and_config_reject_ambiguous_or_out_of_bound_inputs():
    policy, initial = _reproducible_policy()
    assert not policy.source_pulse_widths.flags.writeable
    with pytest.raises(ValueError, match="PW bound"):
        CompactMomentViabilityPolicy(
            intervals=policy.intervals,
            parameters=policy.parameters,
            source_pulse_widths=np.full_like(policy.source_pulse_widths, 1.0),
        )
    with pytest.raises(ValueError, match="horizon_cycles"):
        FiveCycleViabilityConfig(horizon_cycles=0)
    with pytest.raises(ValueError, match="shape"):
        run_five_cycle_viability_preview(policy, initial_states=initial[:, :4])


def test_only_machine_roundoff_in_initial_cn_is_projected_and_audited():
    policy, initial = _reproducible_policy()
    roundoff = initial.copy()
    roundoff[0, 0] = -1.6e-16
    result = run_five_cycle_viability_preview(
        policy,
        initial_states=roundoff,
        config=FiveCycleViabilityConfig(horizon_cycles=1, integration_substeps=4),
    )
    assert result.initial_cn_roundoff_projection_count == 1
    assert result.initial_cn_roundoff_projection_maximum_absolute == pytest.approx(1.6e-16)

    invalid = initial.copy()
    invalid[0, 0] = -1e-6
    with pytest.raises(ValueError, match="physiologically negative"):
        run_five_cycle_viability_preview(
            policy,
            initial_states=invalid,
            config=FiveCycleViabilityConfig(horizon_cycles=1),
        )


def test_negative_mechanical_gain_is_audited_and_never_clipped():
    policy, initial = _reproducible_policy()
    invalid_interval = replace(policy.intervals[0], mechanical_gains=(-0.01, 0.9))
    invalid_policy = CompactMomentViabilityPolicy(
        intervals=(invalid_interval,),
        parameters=policy.parameters,
        source_pulse_widths=policy.source_pulse_widths,
    )
    audit = audit_policy_mechanical_gains(invalid_policy, integration_substeps=4)
    assert not audit.valid
    assert audit.minimum_gain == pytest.approx(-0.01)
    assert audit.nonpositive_count == 9
    with pytest.raises(MechanicalGainDomainError, match="No clipping"):
        run_five_cycle_viability_preview(
            invalid_policy,
            initial_states=initial,
            config=FiveCycleViabilityConfig(horizon_cycles=1, integration_substeps=4),
        )


def test_callable_mechanical_gain_is_cached_without_resampling():
    policy, initial = _reproducible_policy()
    calls = []

    def gain(local_time):
        calls.append(local_time)
        return 0.96 + 0.001 * local_time

    interval = replace(policy.intervals[0], mechanical_gains=(gain, 0.90))
    cached_policy = CompactMomentViabilityPolicy(
        intervals=(interval,),
        parameters=policy.parameters,
        source_pulse_widths=policy.source_pulse_widths,
    )
    run_five_cycle_viability_preview(
        cached_policy,
        initial_states=initial,
        config=FiveCycleViabilityConfig(horizon_cycles=1, integration_substeps=4),
    )

    # Nine audit stages plus a small set of exact RK4 float keys shared by all
    # inversion candidates and all three policies. Without caching, this is
    # several hundred reduced-mechanics calls even for this one interval.
    assert len(calls) <= 20
