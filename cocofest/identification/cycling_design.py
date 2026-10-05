"""Bridge between explicit Ding cycling trials and robust information design."""

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np

from .optimal_design import TrialInformation, fisher_information, scaled_sensitivities, select_trials
from .synthetic_cycling import CyclingTrialSpec, default_ding_parameters, predict_trial, simple_mono_muscle_spec


def mono_muscle_catalogue(muscle_name: str, stage: str = "force", *, radial_gain: float | None = None) -> list[CyclingTrialSpec]:
    """Illustrative simulation catalogue, requiring subject-specific limits.

    Fresh-force: frequency x PW train followed by force relaxation. Fatigue:
    repeated drive, recovery and identical probe. Complete fatigue blocks begin
    fresh and are alternatives; concatenating selected blocks requires an
    explicit inter-block recovery/initial-state protocol. Geometry is a constant
    smoke-test projection, to be replaced by measured/modelled cycling geometry.
    """
    if stage not in ("force", "fatigue"):
        raise ValueError("stage must be force or fatigue")
    catalogue = []
    if stage == "force":
        cases = [(frequency, width, 0.5, 0.3) for frequency in (10., 30., 50.) for width in (0.0002, 0.00035, 0.00055)]
    else:
        cases = [(30., width, drive, recovery) for width in (0.0003, 0.0005) for drive, recovery in ((10., 10.), (20., 40.))]
    for frequency, width, drive, recovery in cases:
        probe_duration = 0.0 if stage == "force" else 0.5
        spec = simple_mono_muscle_spec(
            muscle_name, frequency_hz=frequency, pulse_width_s=width,
            duration_s=drive + recovery + probe_duration, sample_rate_hz=100., radial_gain=radial_gain,
        )
        train = np.arange(0., drive, 1 / frequency)
        probe = np.arange(drive + recovery, drive + recovery + probe_duration, 1 / 30.)
        catalogue.append(replace(
            spec, name=f"{spec.name}_{stage}_{drive:g}s_recovery{recovery:g}s",
            pulse_times_s=np.concatenate((train, probe)),
            pulse_width_s=np.concatenate((np.full(train.size, width), np.full(probe.size, 0.00035))),
        ))
    return catalogue


def design_trial_catalogue(
    catalogue: Sequence[CyclingTrialSpec],
    parameter_names: Sequence[str],
    parameter_ensemble: Sequence[Mapping[str, float]],
    scales: Sequence[float],
    *,
    count: int = 3,
    noise_std_n: float | Sequence[float] = 1.0,
    recovery_overhead_s: float = 0.,
    maximum_capacity_loss: float = 1.0,
    difference_step: float = 1e-4,
    **selection_options,
) -> dict:
    """Compute candidates at an ensemble and select complementary trials.

    Each ensemble mapping supplies varied targets AND any uncertain fixed
    parameters. During fatigue design, these fixed entries can include fresh
    force posterior draws. ``maximum_capacity_loss`` is only a simulation
    admissibility criterion, not a medically validated stopping threshold.
    Independent Gaussian observations are assumed; downsample/prewhiten actual
    recordings before using this wrapper. Use low-level design for full noise
    covariance and nuisance profiling.
    """
    if not parameter_names or len(set(parameter_names)) != len(parameter_names) or not parameter_ensemble:
        raise ValueError("Unique parameter names and a nonempty ensemble are required")
    if not np.isfinite(recovery_overhead_s) or recovery_overhead_s < 0:
        raise ValueError("recovery overhead must be finite and nonnegative")
    if not np.isfinite(maximum_capacity_loss) or not 0 <= maximum_capacity_loss <= 1:
        raise ValueError("maximum_capacity_loss must lie in [0, 1]")
    candidates = []
    for spec in catalogue:
        information, losses = [], []
        for member in parameter_ensemble:
            values = np.array([member[name] for name in parameter_names], float)

            def predict(vector):
                params = dict(member)
                params.update(zip(parameter_names, vector))
                return predict_trial(params, spec)[0].ravel()

            observations, states = predict_trial(member, spec)
            sigma = np.broadcast_to(noise_std_n, observations.shape).ravel()
            jac = scaled_sensitivities(predict, values, np.asarray(scales), sigma, step=difference_step)
            information.append(fisher_information(jac))
            a_rest = member.get("a_rest", default_ding_parameters(spec.muscle_name).fatigue.a_rest)
            losses.append(max(0., float(1 - np.min(states[:, 2]) / a_rest)))
        cost = max(losses)
        candidates.append(TrialInformation(
            spec.name, np.stack(information), float(spec.time_s[-1]) + recovery_overhead_s,
            fatigue_cost=cost, admissible=cost <= maximum_capacity_loss,
        ))
    report = select_trials(candidates, count, **selection_options)
    report.update({
        "parameter_names": list(parameter_names),
        "parameter_scales": list(scales),
        "ensemble_size": len(parameter_ensemble),
        "candidate_names": [trial.name for trial in catalogue],
        "candidate_capacity_loss": [candidate.fatigue_cost for candidate in candidates],
        "geometry_status": "explicit_input_requires_task_calibration",
        "noise_assumption": "independent_Gaussian_samples_and_channels",
        "initial_state_assumption": "fresh_at_start_of_each_complete_candidate_block",
    })
    return report
