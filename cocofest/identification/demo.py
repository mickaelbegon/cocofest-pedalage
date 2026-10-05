"""Small, reproducible end-to-end demonstration of staged Ding identification.

The default geometry and noise are synthetic. A saved reduced bioMod profile
can replace the tangential projection and Ding mechanical gain. Supplying the
source bioMod additionally derives the two-axis endpoint-force map from the
hand Jacobian and inverse dynamics.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .cycling_design import design_trial_catalogue
from .cycling_geometry import PlanarEndpointForceAdapter, ReducedGeometryAdapter
from .sequential import ParameterSpec, TrialObservation, fit_parameters
from .synthetic_cycling import (
    CyclingTrialSpec, SensorNoise, default_ding_parameters, generate_trial,
    predict_trial, simple_mono_muscle_spec,
)


def _fatigue_spec(muscle: str, *, drive_s: float, recovery_s: float,
                  radial_gain: float | None) -> CyclingTrialSpec:
    """A complete fresh-start drive, recovery, and identical probe block."""
    duration = drive_s + recovery_s + 0.3
    spec = simple_mono_muscle_spec(
        muscle, frequency_hz=30., pulse_width_s=0.00045,
        duration_s=duration, sample_rate_hz=25., radial_gain=radial_gain,
    )
    drive = np.arange(0., drive_s, 1 / 30.)
    probe = np.arange(drive_s + recovery_s, duration - 0.01, 1 / 30.)
    return replace(
        spec, name=f"fatigue_drive{drive_s:g}_recovery{recovery_s:g}",
        pulse_times_s=np.r_[drive, probe],
        pulse_width_s=np.r_[np.full(drive.size, 0.00045), np.full(probe.size, 0.00035)],
    )


def run_demo(*, muscle: str = "right_Biceps", two_axis: bool = False,
             seed: int = 7, profile_path: str | Path | None = None,
             profile_muscle_name: str | None = None,
             model_path: str | Path | None = None) -> dict:
    """Design, synthesize, estimate and validate one muscle's force and fatigue.

    The local D-optimal design uses an approximate parameter ensemble, never
    the hidden synthetic truth. Each fatigue candidate starts fully recovered;
    selected candidates therefore represent independent blocks. The reported
    holdout errors are computed after fitting and do not influence selection.
    """
    base = default_ding_parameters(muscle)
    radial_gain = 0.12 if two_axis else None
    adapter = None
    if profile_path is not None:
        mapping = {muscle: profile_muscle_name} if profile_muscle_name else None
        adapter = (
            PlanarEndpointForceAdapter.from_paths(
                profile_path, model_path, muscle_map=mapping,
            )
            if model_path is not None
            else ReducedGeometryAdapter.load(profile_path, muscle_map=mapping)
        )
    elif model_path is not None:
        raise ValueError("model_path requires profile_path so source-model provenance can be checked.")

    def geometry(spec: CyclingTrialSpec) -> CyclingTrialSpec:
        if adapter is None:
            return spec
        if isinstance(adapter, PlanarEndpointForceAdapter):
            return adapter.apply(
                spec, omega_rad_s=2 * np.pi / 4, include_radial=two_axis,
            )
        return adapter.apply(spec, omega_rad_s=2 * np.pi / 4,
                             radial_gain=spec.radial_gain)

    force_candidates = [geometry(simple_mono_muscle_spec(
        muscle, frequency_hz=frequency, pulse_width_s=width,
        duration_s=0.28, sample_rate_hz=25., radial_gain=radial_gain,
    )) for frequency, width in ((10., 0.00022), (30., 0.0004), (50., 0.00055))]
    force_holdout_spec = geometry(simple_mono_muscle_spec(
        muscle, frequency_hz=20., pulse_width_s=0.00032,
        duration_s=0.36, sample_rate_hz=25., radial_gain=radial_gain,
    ))
    force_ensemble = [
        {"a_rest": base.fatigue.a_rest * gain, "pdt": base.pdt * recruitment}
        for gain, recruitment in ((0.9, 0.9), (1.1, 1.1))
    ]
    force_design = design_trial_catalogue(
        force_candidates, ("a_rest", "pdt"), force_ensemble,
        (base.fatigue.a_rest, base.pdt), count=2,
        noise_std_n=[0.03, 0.03] if two_axis else 0.03,
        duration_budget_s=0.8,
    )
    selected_force = {entry["name"] for entry in force_design["selected"]}
    if len(selected_force) != 2:
        raise RuntimeError("Force design did not select two admissible trials")

    truth = {
        "a_rest": 1.08 * base.fatigue.a_rest,
        "pdt": 1.12 * base.pdt,
        "alpha_a": 1.4 * base.fatigue.alpha_a,
    }
    noise = SensorNoise(tangential_std_n=0.03,
                        radial_std_n=0.03 if two_axis else 0.)

    def observation(spec: CyclingTrialSpec, trial_seed: int) -> TrialObservation:
        trial = generate_trial(spec, parameters=truth, noise=noise, seed=trial_seed)
        return TrialObservation(spec.name, trial.observed, trial.noise_std, spec)

    force_trials = [observation(spec, seed + index)
                    for index, spec in enumerate(force_candidates) if spec.name in selected_force]
    force_holdout = observation(force_holdout_spec, seed + 10)
    predictor = lambda parameters, context: predict_trial(parameters, context)[0]
    force_specs = (
        ParameterSpec("a_rest", base.fatigue.a_rest, 0.7 * base.fatigue.a_rest,
                      1.4 * base.fatigue.a_rest, base.fatigue.a_rest),
        ParameterSpec("pdt", base.pdt, 0.0001, 0.0004, 0.0002),
    )
    # Force is fitted before fatigue design, so its frozen estimate forms the
    # nominal mechanical/force component of the fatigue candidate ensemble.
    force_fit = fit_parameters(
        predictor, force_trials, force_specs,
        fixed_parameters={"alpha_a": base.fatigue.alpha_a},
        holdout=[force_holdout], max_nfev=80,
    )
    if not force_fit.converged:
        raise RuntimeError(f"Force fit failed: {force_fit.message}")
    fatigue_candidates = [geometry(_fatigue_spec(muscle, drive_s=drive, recovery_s=recovery,
                                                  radial_gain=radial_gain))
                          for drive, recovery in ((4., 1.), (8., 2.))]
    fatigue_holdout_spec = geometry(_fatigue_spec(muscle, drive_s=6., recovery_s=1.5,
                                                  radial_gain=radial_gain))
    fatigue_ensemble = [
        {"a_rest": force_fit.parameters["a_rest"], "pdt": force_fit.parameters["pdt"],
         "alpha_a": base.fatigue.alpha_a * factor}
        for factor in (0.8, 1.2)
    ]
    fatigue_design = design_trial_catalogue(
        fatigue_candidates, ("alpha_a",), fatigue_ensemble,
        (abs(base.fatigue.alpha_a),), count=1,
        noise_std_n=[0.03, 0.03] if two_axis else 0.03,
        duration_budget_s=11., time_normalized=False,
    )
    selected_fatigue = {entry["name"] for entry in fatigue_design["selected"]}
    if len(selected_fatigue) != 1:
        raise RuntimeError("Fatigue design did not select an admissible trial")
    fatigue_trials = [observation(spec, seed + 20 + index)
                      for index, spec in enumerate(fatigue_candidates) if spec.name in selected_fatigue]
    fatigue_holdout = observation(fatigue_holdout_spec, seed + 30)
    fatigue_specs = (
        ParameterSpec("alpha_a", base.fatigue.alpha_a, 2.5 * base.fatigue.alpha_a,
                      0.1 * base.fatigue.alpha_a, abs(base.fatigue.alpha_a)),
    )
    fatigue_fit = fit_parameters(
        predictor, fatigue_trials, fatigue_specs,
        fixed_parameters={name: value for name, value in force_fit.parameters.items()
                          if name != "alpha_a"},
        holdout=[fatigue_holdout], max_nfev=80,
    )
    if not fatigue_fit.converged:
        raise RuntimeError(f"Fatigue fit failed: {fatigue_fit.message}")
    return {
        "schema": "cocofest.ding_identification.end_to_end_demo.v1",
        "muscle": muscle, "channels": ["tangential_n"] + (["radial_n"] if two_axis else []),
        "seed": seed,
        "geometry": adapter.provenance if adapter is not None else {"kind": "constant_demonstration"},
        "force_design": force_design, "fatigue_design": fatigue_design,
        "force_fit": force_fit.to_dict(), "fatigue_fit": fatigue_fit.to_dict(),
        "synthetic_truth": truth,
        "holdout_names": [force_holdout.name, fatigue_holdout.name],
        "assumptions": [
            "local Fisher information; independent Gaussian sensor samples",
            "each selected fatigue block starts fully recovered",
            "fatigue covariance is conditional on the frozen force estimate",
            (
                "radial gain comes from the bioMod endpoint inverse dynamics"
                if isinstance(adapter, PlanarEndpointForceAdapter)
                else "radial gain requires separate sensor or reaction-force calibration"
            ),
        ],
    }
