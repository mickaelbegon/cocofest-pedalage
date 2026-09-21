# PACE: guarded fatigue selection

The 0.15 Nm, 500-cycle experiment showed that the previous torque-deficit
ranking could worsen observed fatigue even though its compact rollout score
improved. The predictor allocates muscle moments under prescribed mechanics;
the fast RHO optimizes a fatigue cost with mechanical dynamics. They are
different policies. Better surrogate scores do not prove better RHO endurance.

The predictive controller now enables `projection_fatigue_guard` by default.
The worker still runs asynchronously, uses the same candidate batch and the
same 100-cycle horizon when configured, and cannot delay the fast solver.

## What changes

1. Candidate weights move at most `max_log_step` from the incumbent (default
   `log(1.1)`). This applies before rollout to every candidate, including a
   return toward uniform weights. The existing global weight box and geometric
   mean normalization are preserved. The exact evaluated weights are handed to
   the OCP: there is no unevaluated smoothing afterward.
2. For every completed rollout, we measure unweighted squared fatigue from
   `1 - A/A_rest`, integrated with duration-weighted phase trapezoids. We keep
   both the full-horizon mean and the first 20-cycle block mean. We also record
   the minimum terminal capacity across muscles. Candidate weights do not
   enter these metrics, so a candidate cannot improve its own audit merely by
   reducing a muscle's objective coefficient.
3. A candidate must not increase full-horizon or first-block squared fatigue
   (absolute numerical tolerance `1e-10`), nor reduce the minimum terminal
   capacity by more than `1e-5`. It must also preserve the full-horizon moment
   deficit to `1e-12`.
4. Passing these tests is necessary but insufficient. The candidate must reduce
   either moment deficit or full-horizon squared fatigue by more than both
   1% and `1e-6` absolute. These are explicit provisional deadbands, not
   statistically calibrated uncertainty bounds. Reserve-only changes and
   tiny score fluctuations keep incumbent weights.
5. Missing, failed or truncated fatigue evidence fails closed. A ready
   proposal is still subject to the asynchronous freshness checks and can be
   applied only at a certified RHO boundary. The controller additionally
   refuses an oversized step or an audit from the unguarded selector.

The incumbent is always an evaluated candidate. When no candidate passes, the
journal records `guarded_hold_incumbent`. Successful changes record
`guarded_fatigue_improvement` and all evidence remains in candidate evaluations.
This reduces large weight drift; it does not mathematically prohibit many
successive justified steps from reaching the configured global bounds.

## Numerical checkpoint validation

`scripts/validate_guarded_pace_policy.py` compares both selectors on the same
certified checkpoint and explicitly initializes incumbent weights to uniform.
This is a local counterfactual at the supplied state, not replay of the
checkpoint's historical policy.

At the real cycle-340 checkpoint, 0.15 Nm, 30 Hz, with 100 projected cycles and
16 substeps, the guarded nine-candidate batch took **2.99 s** (legacy 3.09 s).
It selected `[1.03228, 1.03228, 0.90909, 1.03228]`. Compared with its uniform
incumbent at that same state:

| Projected quantity | Incumbent | Guarded selection |
| --- | ---: | ---: |
| Mean squared torque deficit | 0.00006659 | 0.00005723 |
| Full-horizon mean squared fatigue | 0.086514 | 0.085228 |
| First-20-cycle mean squared fatigue | 0.065761 | 0.065495 |
| Terminal minimum A/A_rest | 0.489675 | 0.501861 |

Artifact: `pace-guard-validation-20260912/checkpoint340.json`.
The legacy selector's larger step also improved these local metrics, but is
outside the new trust region. This checkpoint therefore demonstrates an
actionable, inexpensive guarded update; it does not establish superiority
over legacy PACE in closed loop. A fresh matched 500-cycle campaign is needed.

No FHO data, extra OCP, recovery solve, or additional rollout is used by the
guard. The additional cost consists only of reading already propagated states.
The next scientific limitation remains identification of the difference
between the compact allocator and the RHO response; these safeguards bound
surrogate risk but cannot guarantee real closed-loop non-degradation.
