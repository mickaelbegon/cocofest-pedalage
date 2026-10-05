"""Whole-trial uncertainty propagation for sequential force/fatigue fitting.

Resampling units must be independent complete acquisitions. Never split a
continuous drive/recovery/probe sequence: its context must propagate the full
state history. Within-condition strata preserve a designed experiment's mix.
"""

from collections import defaultdict
from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np

from .sequential import ParameterSpec, Predictor, TrialObservation, fit_force_then_fatigue


def _groups(trials, strata):
    names = [trial.name for trial in trials]
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Each stage needs at least two uniquely named independent trials")
    if strata is not None and set(strata) != set(names):
        raise ValueError("Strata must label every training trial exactly once")
    groups = defaultdict(list)
    for trial in trials:
        groups["all" if strata is None else strata[trial.name]].append(trial)
    if all(len(group) == 1 for group in groups.values()):
        raise ValueError("At least one stratum per stage needs replicated independent trials")
    return list(groups.values())


def _resample(groups, rng, prefix):
    sampled, sources = [], []
    for group in groups:
        for index in rng.integers(0, len(group), size=len(group)):
            trial = group[index]
            sources.append(trial.name)
            # Duplicate draws need distinct estimator IDs; all samples, channels,
            # noise weights and the complete physical context remain together.
            sampled.append(replace(trial, name=f"{prefix}_{len(sampled)}"))
    return sampled, sources


def bootstrap_force_then_fatigue(
    predictor: Predictor,
    force_trials: Sequence[TrialObservation],
    fatigue_trials: Sequence[TrialObservation],
    force_specs: Sequence[ParameterSpec],
    fatigue_specs: Sequence[ParameterSpec],
    *,
    fixed_parameters: Mapping[str, float] | None = None,
    force_strata: Mapping[str, str] | None = None,
    fatigue_strata: Mapping[str, str] | None = None,
    n_resamples: int = 200,
    seed: int = 0,
    confidence_level: float = 0.95,
    minimum_success_fraction: float = 0.8,
    max_nfev: int = 300,
) -> dict:
    """Refit both stages for each stratified cluster-bootstrap replicate.

Intervals are pointwise percentile intervals for the bounded MAP estimator,
not Bayesian credible intervals or guaranteed coverage. Priors are held fixed.
Force/fatigue acquisitions are sampled independently: paired sessions or shared
subject drift require a session-level joint bootstrap, not this routine.
Fewer than 200 draws is supported for tests but explicitly flagged.
"""
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 2:
        raise ValueError("n_resamples must be an integer of at least two")
    if not 0 < confidence_level < 1 or not 0 < minimum_success_fraction <= 1:
        raise ValueError("Invalid confidence level or minimum success fraction")
    force_trials, fatigue_trials = tuple(force_trials), tuple(fatigue_trials)
    force_specs, fatigue_specs = tuple(force_specs), tuple(fatigue_specs)
    if {trial.name for trial in force_trials} & {trial.name for trial in fatigue_trials}:
        raise ValueError("Force and fatigue trial names must be disjoint")
    force_groups, fatigue_groups = _groups(force_trials, force_strata), _groups(fatigue_trials, fatigue_strata)
    # Establish a valid baseline before processing stochastic replicates.
    baseline = fit_force_then_fatigue(
        predictor, force_trials, fatigue_trials, force_specs, fatigue_specs,
        fixed_parameters=fixed_parameters, max_nfev=max_nfev,
    )
    if not baseline["fatigue"].converged:
        raise RuntimeError("Baseline fatigue fit did not converge")
    rng = np.random.default_rng(seed)
    successful, failed = [], []
    for replicate in range(n_resamples):
        force, force_sources = _resample(force_groups, rng, "force")
        fatigue, fatigue_sources = _resample(fatigue_groups, rng, "fatigue")
        provenance = {"replicate": replicate, "force_sources": force_sources, "fatigue_sources": fatigue_sources}
        try:
            result = fit_force_then_fatigue(
                predictor, force, fatigue, force_specs, fatigue_specs,
                fixed_parameters=fixed_parameters, max_nfev=max_nfev,
            )
        except RuntimeError as error:
            failed.append({**provenance, "reason": str(error)})
            continue
        if not result["fatigue"].converged:
            failed.append({**provenance, "reason": result["fatigue"].message})
            continue
        successful.append({
            **provenance,
            "parameters": result["fatigue"].parameters,
            "force_parameters": {spec.name: result["force"].parameters[spec.name] for spec in force_specs},
            "full_rank": {stage: fit.fisher["full_rank"] for stage, fit in result.items()},
            "active_bounds": {stage: fit.active_bounds for stage, fit in result.items()},
        })
    adequate = len(successful) >= max(2, int(np.ceil(n_resamples * minimum_success_fraction)))
    intervals = None
    if adequate:
        tail = (1 - confidence_level) / 2
        intervals = {
            spec.name: dict(zip(("lower", "median", "upper"), np.quantile(
                [draw["parameters"][spec.name] for draw in successful], [tail, .5, 1 - tail]
            ).tolist())) for spec in (*force_specs, *fatigue_specs)
        }
    warnings = []
    if n_resamples < 200:
        warnings.append("Few bootstrap replicates: tail quantiles are unstable")
    if not adequate:
        warnings.append("Insufficient converged replicates: intervals withheld")
    if failed:
        warnings.append("Intervals condition on converged replicates; failures may bias uncertainty")
    if any(not all(draw["full_rank"].values()) for draw in successful):
        warnings.append("Some data Jacobians are rank deficient; bootstrap does not establish identifiability")
    if any(any(value for bounds in draw["active_bounds"].values() for value in bounds.values()) for draw in successful):
        warnings.append("Bound-active estimates make percentile coverage nonregular")
    if any(len(group) == 1 for group in (*force_groups, *fatigue_groups)):
        warnings.append("Singleton strata contribute no between-acquisition uncertainty")
    return {
        "method": "stratified_whole_trial_percentile_bootstrap",
        "seed": seed, "requested_resamples": n_resamples, "successful_resamples": len(successful),
        "confidence_level": confidence_level, "minimum_success_fraction": minimum_success_fraction,
        "adequate_success_fraction": adequate, "intervals": intervals,
        "baseline": {stage: fit.to_dict() for stage, fit in baseline.items()},
        "replicates": successful, "failed_replicates": failed, "warnings": warnings,
        "force_parameter_names": [spec.name for spec in force_specs],
        "fatigue_parameter_names": [spec.name for spec in fatigue_specs],
        "fixed_parameters": dict(fixed_parameters or {}),
        "resampling_assumption": "independent_acquisitions_within_each_stage_and_stratum",
        "strata": {"force": [[t.name for t in group] for group in force_groups],
                   "fatigue": [[t.name for t in group] for group in fatigue_groups]},
    }


def fatigue_design_ensemble(bootstrap_report: Mapping, fatigue_members: Sequence[Mapping[str, float]]) -> list[dict]:
    """Cross force-bootstrap draws with declared fatigue parameter scenarios.

The ensemble can be passed directly to ``design_trial_catalogue``. Correlations
between force parameters are retained within each draw. Fatigue scenarios are
assumed independent of force uncertainty; for a post-fatigue design use each
complete ``replicates[i]['parameters']`` instead to preserve joint correlations.
This spreads local Fisher calculations over uncertain force gains; it does not
marginalize nuisance parameters from Fisher information or supply a posterior.
"""
    if not bootstrap_report.get("adequate_success_fraction") or not bootstrap_report.get("replicates"):
        raise ValueError("A bootstrap with sufficient successful replicates is required")
    if not fatigue_members:
        raise ValueError("At least one fatigue parameter scenario is required")
    fatigue_names = set(bootstrap_report["fatigue_parameter_names"])
    ensemble = []
    for member in fatigue_members:
        if set(member) != fatigue_names or not all(np.isfinite(v) for v in member.values()):
            raise ValueError("Fatigue scenarios must supply exactly the fitted fatigue parameters with finite values")
        for draw in bootstrap_report["replicates"]:
            ensemble.append({**bootstrap_report["fixed_parameters"], **draw["force_parameters"], **member})
    return ensemble
