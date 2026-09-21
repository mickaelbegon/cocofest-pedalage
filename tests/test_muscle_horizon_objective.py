import numpy as np
import pytest

from cocofest.optimization.adaptive_moment_rollout import (
    DingPulseWidthParameters,
    MomentTrackingInterval,
    propagate_ding_pulse_width_interval,
)
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.muscle_horizon_objective import (
    MUSCLE_HORIZON_DOMAIN_MARGIN_NAMES,
    MuscleHorizonLayout,
    build_muscle_horizon_function,
    pack_muscle_horizon_profile,
)


def _muscle(capacity=4920.0):
    return DingPulseWidthParameters(
        fatigue=DingFatigueParameters(
            a_rest=capacity,
            tau1_rest=0.060601,
            km_rest=0.137,
            alpha_a=-0.04,
            alpha_tau1=2.1e-6,
            alpha_km=1.9e-6,
            tau_fat=127.0,
        ),
        tauc=0.011,
        tau2=0.001,
        pd0=0.000131405,
        pdt=0.000194138,
        pulse_width_max=0.0006,
    )


def _initial(muscle, force=20.0):
    return np.array(
        [
            0.1629821583533315,
            force,
            0.98 * muscle.fatigue.a_rest,
            1.02 * muscle.fatigue.tau1_rest,
            1.02 * muscle.fatigue.km_rest,
        ]
    )


def _single_muscle_policy(substeps=2):
    muscle = _muscle()
    initial = _initial(muscle)
    pulse_widths = np.array([[0.00031, 0.00036]])
    gains = (0.92, 0.98)
    current = initial.copy()
    intervals = []
    for interval_index in range(2):
        current = propagate_ding_pulse_width_interval(
            current,
            pulse_width=pulse_widths[0, interval_index],
            duration=1.0 / 30.0,
            calcium_amplitude=1.0597355478114694,
            mechanical_gain=gains[interval_index],
            parameters=muscle,
            integration_substeps=substeps,
        )
        intervals.append(
            MomentTrackingInterval(
                duration=1.0 / 30.0,
                calcium_amplitudes=(1.0597355478114694,),
                mechanical_gains=(gains[interval_index],),
                moment_coefficients=(-0.045,),
                target_moments=(-0.045 * current[1],),
            )
        )
    return muscle, initial, tuple(intervals), pulse_widths


@pytest.mark.parametrize("symbolic_type", ["SX", "MX"])
def test_symbolic_horizon_matches_independent_numeric_full_ding_rollout(symbolic_type):
    muscle, initial, intervals, source_pulse_widths = _single_muscle_policy(substeps=2)
    layout = MuscleHorizonLayout(
        muscle_count=1,
        interval_count=2,
        horizon_cycles=2,
        integration_substeps=2,
    )
    packed = pack_muscle_horizon_profile(intervals, source_pulse_widths, layout)
    function = build_muscle_horizon_function(
        muscles=(muscle,), layout=layout, symbolic_type=symbolic_type
    )
    outputs = function(initial, packed.parameters, packed.future_pulse_width_seed)

    current = initial.copy()
    expected_moments = []
    expected_residuals = []
    for _ in range(layout.horizon_cycles):
        for interval_index, interval in enumerate(intervals):
            current = propagate_ding_pulse_width_interval(
                current,
                pulse_width=source_pulse_widths[0, interval_index],
                duration=interval.duration,
                calcium_amplitude=interval.calcium_amplitudes[0],
                mechanical_gain=interval.mechanical_gains[0],
                parameters=muscle,
                integration_substeps=layout.integration_substeps,
            )
            moment = interval.moment_coefficients[0] * current[1]
            expected_moments.append(moment)
            expected_residuals.append(moment - interval.target_moments[0])
    np.testing.assert_allclose(np.asarray(outputs[3]).reshape(-1), expected_moments, atol=2e-12)
    np.testing.assert_allclose(np.asarray(outputs[4]).reshape(-1), expected_residuals, atol=2e-12)
    np.testing.assert_allclose(np.asarray(outputs[5]).reshape(-1), current, atol=2e-12)
    assert outputs[6].numel() == layout.domain_margin_size
    assert layout.domain_margin_size == (
        len(MUSCLE_HORIZON_DOMAIN_MARGIN_NAMES) * layout.future_pulse_width_size
    )


def test_profile_pack_samples_time_varying_gain_at_each_rk_stage():
    muscle, _, intervals, source_pulse_widths = _single_muscle_policy(substeps=1)
    interval = MomentTrackingInterval(
        duration=0.2,
        calcium_amplitudes=intervals[0].calcium_amplitudes,
        mechanical_gains=(lambda time: 1.0 + 2.0 * time,),
        moment_coefficients=intervals[0].moment_coefficients,
        target_moments=intervals[0].target_moments,
    )
    layout = MuscleHorizonLayout(1, 1, integration_substeps=2)
    packed = pack_muscle_horizon_profile((interval,), source_pulse_widths[:, :1], layout)
    slices = layout.parameter_slices()

    np.testing.assert_allclose(packed.parameters[slices["gain_start"]], [1.0, 1.2])
    np.testing.assert_allclose(packed.parameters[slices["gain_midpoint"]], [1.1, 1.3])
    np.testing.assert_allclose(packed.parameters[slices["gain_endpoint"]], [1.2, 1.4])


def test_future_pw_jacobian_matches_central_differences():
    import casadi as ca

    muscle, initial, intervals, source_pulse_widths = _single_muscle_policy(substeps=2)
    layout = MuscleHorizonLayout(1, 2, horizon_cycles=2, integration_substeps=2)
    packed = pack_muscle_horizon_profile(intervals, source_pulse_widths, layout)
    function = build_muscle_horizon_function(muscles=(muscle,), layout=layout)
    pulse_widths = ca.SX.sym("future_pw", layout.future_pulse_width_size)
    outputs = function(initial, packed.parameters, pulse_widths)
    observed_vector = ca.vertcat(outputs[0], outputs[4], outputs[5])
    jacobian = ca.Function("muscle_horizon_jacobian", [pulse_widths], [ca.jacobian(observed_vector, pulse_widths)])
    analytic = np.asarray(jacobian(packed.future_pulse_width_seed))

    step = 1e-9
    central = np.empty_like(analytic)
    for column in range(layout.future_pulse_width_size):
        plus = packed.future_pulse_width_seed.copy()
        minus = packed.future_pulse_width_seed.copy()
        plus[column] += step
        minus[column] -= step
        plus_outputs = function(initial, packed.parameters, plus)
        minus_outputs = function(initial, packed.parameters, minus)
        plus_vector = np.concatenate(
            ([float(plus_outputs[0])], np.asarray(plus_outputs[4]).reshape(-1), np.asarray(plus_outputs[5]).reshape(-1))
        )
        minus_vector = np.concatenate(
            ([float(minus_outputs[0])], np.asarray(minus_outputs[4]).reshape(-1), np.asarray(minus_outputs[5]).reshape(-1))
        )
        central[:, column] = (plus_vector - minus_vector) / (2.0 * step)
    np.testing.assert_allclose(analytic, central, rtol=3e-6, atol=2e-5)


def test_layout_and_packer_reject_dimension_changes():
    muscle, _, intervals, source_pulse_widths = _single_muscle_policy()
    layout = MuscleHorizonLayout(1, 2)
    with pytest.raises(ValueError, match="source_pulse_widths"):
        pack_muscle_horizon_profile(intervals, np.vstack((source_pulse_widths, source_pulse_widths)), layout)
    function = build_muscle_horizon_function(muscles=(muscle,), layout=layout)
    packed = pack_muscle_horizon_profile(intervals, source_pulse_widths, layout)
    with pytest.raises(ValueError, match="future_pulse_widths"):
        # The check occurs while constructing expressions, before a CasADi call.
        from cocofest.optimization.muscle_horizon_objective import build_muscle_horizon_expressions

        build_muscle_horizon_expressions(
            np.zeros(layout.initial_state_size),
            packed.parameters,
            np.zeros(layout.future_pulse_width_size - 1),
            muscles=(muscle,),
            layout=layout,
        )
    assert function.size1_in("future_pulse_widths") == layout.future_pulse_width_size
