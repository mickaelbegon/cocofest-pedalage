import numpy as np
import pytest

from cocofest.optimization.periodic_force_profile import (
    fit_periodic_fourier_force_profile,
    periodic_force_profile_audit,
)


def _signals(time, period):
    omega = 2.0 * np.pi / period
    return np.vstack(
        (
            50.0 + 10.0 * np.cos(omega * time) + 4.0 * np.sin(2.0 * omega * time),
            20.0 + 3.0 * np.sin(omega * time) - 2.0 * np.cos(3.0 * omega * time),
        )
    )


def _derivatives(time, period):
    omega = 2.0 * np.pi / period
    return np.vstack(
        (
            -10.0 * omega * np.sin(omega * time) + 8.0 * omega * np.cos(2.0 * omega * time),
            3.0 * omega * np.cos(omega * time) + 6.0 * omega * np.sin(3.0 * omega * time),
        )
    )


def test_fourier_fit_recovers_periodic_force_and_analytic_derivative():
    period = 0.8
    nodes = np.arange(32) * period / 32
    profile = fit_periodic_fourier_force_profile(
        _signals(nodes, period), period=period, harmonic_count=3
    )
    query = np.linspace(-0.2, 1.7, 71)

    np.testing.assert_allclose(profile.evaluate(query), _signals(query, period), atol=2e-13)
    np.testing.assert_allclose(profile.derivative(query), _derivatives(query, period), atol=2e-12)


def test_endpoint_is_checked_and_removed():
    period = 1.0
    nodes = np.linspace(0.0, period, 33)
    samples = _signals(nodes, period)
    profile = fit_periodic_fourier_force_profile(
        samples,
        period=period,
        harmonic_count=3,
        endpoint_included=True,
    )
    audit = periodic_force_profile_audit(profile, samples, endpoint_included=True)

    assert audit["sample_count"] == 32
    assert audit["maximum_absolute_error"] < 1e-12
    assert not audit["negative_reconstruction"]

    samples[:, -1] += 1e-3
    with pytest.raises(ValueError, match="endpoint"):
        fit_periodic_fourier_force_profile(
            samples,
            period=period,
            harmonic_count=3,
            endpoint_included=True,
        )


def test_midpoint_samples_define_visible_piecewise_rollout_assumption():
    period = 0.6
    nodes = np.arange(24) * period / 24
    profile = fit_periodic_fourier_force_profile(
        _signals(nodes, period), period=period, harmonic_count=3
    )

    force, durations = profile.piecewise_midpoints(12)

    np.testing.assert_allclose(np.sum(durations), period)
    np.testing.assert_allclose(
        force,
        _signals((np.arange(12) + 0.5) * period / 12, period),
        atol=2e-13,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"period": 0.0, "harmonic_count": 1},
        {"period": 1.0, "harmonic_count": 0},
        {"period": 1.0, "harmonic_count": 5},
    ],
)
def test_invalid_profile_specification_is_rejected(kwargs):
    with pytest.raises(ValueError):
        fit_periodic_fourier_force_profile(np.ones((2, 5)), **kwargs)


def test_audit_never_hides_negative_fourier_overshoot():
    samples = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    profile = fit_periodic_fourier_force_profile(
        samples, period=1.0, harmonic_count=2
    )

    audit = periodic_force_profile_audit(profile, samples)

    assert "minimum_reconstructed_force" in audit
    assert isinstance(audit["negative_reconstruction"], bool)


def test_continuous_positivity_certificate_finds_an_extremum_missed_by_midpoints():
    # At the two midpoint samples this signal equals +0.1, yet it is -0.9 at
    # half a period. A midpoint-only sign check would therefore be unsound.
    profile = fit_periodic_fourier_force_profile(
        np.array([[1.1, 0.1, -0.9, 0.1]]),
        period=1.0,
        harmonic_count=1,
    )

    midpoint_force, _ = profile.piecewise_midpoints(2)
    certificate = profile.force_positivity_certificate()

    assert np.all(midpoint_force > 0.0)
    assert certificate.minimum_force[0] == pytest.approx(-0.9)
    assert certificate.minimum_time[0] == pytest.approx(0.5)
    assert not certificate.certified_nonnegative[0]


def test_continuous_positivity_certificate_accepts_positive_fourier_signal():
    period = 0.8
    time = np.arange(32) * period / 32
    profile = fit_periodic_fourier_force_profile(
        _signals(time, period), period=period, harmonic_count=3
    )

    certificate = profile.force_positivity_certificate()

    assert np.all(certificate.certified_nonnegative)
    assert np.all(certificate.minimum_force > 0.0)
