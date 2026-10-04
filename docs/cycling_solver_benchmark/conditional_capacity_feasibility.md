# Conditional one-cycle capacity feasibility: offline prototype

The experimental question is: **at this complete prepared state and history,
which small initial changes of muscle capacity still admit the nominal cycle?**
It is narrower than finding a universal maximum fatigue, and does not claim to
predict remaining endurance. The production RHO retains its unit fatigue cost.

## Physical meaning of the slice

The varied coordinates are the initial ratios `A_m/a_scale_m`. All initial
`Cn`, `F`, `Tau1`, `Km`, mechanical states, PW history and other numerical data
retain the exact checkpoint condition. The artifact explicitly lists every
conditioned state's initial lower/upper bounds; if the source contains an
interval, the condition remains that interval, not a claimed exact scalar.
All path and terminal constraints, including the next-half-cycle guard, remain
in the same discrete dynamics graph.

A normalized capacity perturbation at fixed `Tau1` and `Km` is a **synthetic
conditional experiment**, not a reachable state of the original fatigue
trajectory. A witness answers the one-cycle existence question for that state;
it cannot establish that the patient can reach the state or sustain it. The
condition digest also pins the source's numerical seed and stimulation history,
so the current contract is deliberately stricter than a purely physical state
equivalence relation.

The initial pilot defaults to a local box of 0.05 in normalized capacity and
explicit positive/negative perturbations. Every predeclared sample is evaluated.
No binary search, downward-closed feasible set, or monotone boundary is assumed.
Changes of any conditioned state invalidate pooling or comparing the same
capacity coordinates as though they defined one common surface.

## Frozen feasibility test

The source is rebuilt and restored from a certified exact bilateral receipt.
A nominal unit-fatigue solve must first produce an independent full-NLP witness.
The auxiliary oracle then owns a separate CasADi problem using the **same**
`x` and `g` graph and copied bounds. It never calls `update_objectives`, never
changes production fatigue parameters, and never advances the physical RHO.

For a requested capacity vector, only its initial A equalities are replaced.
The auxiliary problem allows terminal produced work in `[0, W_required]` and
minimizes the dimensionless missing work:

`deficit = (W_required - E_prod(T)) / W_required`.

This linear deficit has a nonzero gradient near zero, avoiding the flatness of
a squared deficit. Dynamics, closure, device limits, source state/history and
all other constraints remain hard. If even the relaxed problem fails, the
answer remains undetermined.

The solved vector is independently evaluated against the complete graph and
decision bounds, both with the relaxed work bound and with the original nominal
work equality. Work is additionally normalized by `W_required`, so a small
absolute residual at a small work target cannot produce a false witness.

The evidence distinguishes:

- **Nominal feasible witness:** every original constraint/bound and the
  normalized work equality passes. This conclusion does not depend on the
  solver's success string.
- **Validated relaxed trajectory with positive deficit:** the observed deficit
  is an upper bound on the minimum deficit. A better solution might exist;
  this is not a lower bound or an infeasibility proof.
- **Failed or unvalidated candidate:** undetermined. It does not define the
  fatigue frontier or a negative training label.

The current result therefore contains witnessed samples and undetermined
samples, `boundary_estimate: null`, `activation_allowed: false` and
`physiological_failure_certified: false`. No interpolated separating plane,
terminal penalty, or RHO weight update is enabled.

## API and provenance

`cocofest.optimization.conditional_capacity_feasibility` provides:

- `ConditionalCapacitySlice.from_checkpoint`: normalized coordinate layout,
  immutable source provenance and complete conditional-state signature.
- `directional_points`: explicit centered directional design with domain and
  trust-box checks; no silent clipping.
- `FrozenTaskDeficit`: auxiliary graph, copied numerical bounds, normalized
  work deficit and independent nominal/relaxed audits.
- `mapping_from_program`: symbolic mapping of normalized states to scaled NLP
  variable positions, avoiding guessed offsets.
- `summarize_slice`: complete predeclared sample accounting and conditional
  witness summaries, without a fitted frontier.

`scripts/probe_conditional_capacity_feasibility.py` writes a predeclared plan,
per-point candidate NPZ files, progress and a final
`conditional-capacity-feasibility-v1` report. It retains the source receipt,
archive/model/task hashes, condition digest, original objective and constraint
graph fingerprints, observed deficits, residuals, solver statuses and timing.
Each point verifies that the source prepared state and objective are unchanged.
Failures are saved as undetermined; candidate files are not automatically
labelled witnesses.

```bash
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
"$PY" scripts/probe_conditional_capacity_feasibility.py \
  --receipt /absolute/path/cycle-140/receipt.json \
  --side left --muscles Biceps --offsets -0.005 0.005 \
  --cpu 12 --max-iterations 200 \
  --output-directory /absolute/path/new-capacity-smoke
```

## Numerical validation and smoke

Tests use an exact force-filter model with integrated work and an independently
solved linear program to demonstrate two counterexamples:

1. The same A with different initial force gives different work feasibility.
   The condition signature also rejects changed `Tau1` or PW history.
2. With a positive minimum stimulation and an exact required work, low capacity
   can be insufficient while excessive capacity cannot be throttled down enough.
   Feasibility in A need not be monotone.

Another test deliberately supplies a suboptimal, positive-deficit trajectory
when a zero-deficit trajectory exists. The classifier must retain
`indeterminate`, demonstrating why a local residual alone is insufficient.

The 2026-10-04 real smoke used the left c140 checkpoint at 0.96 Nm per arm,
30 stimulations/cycle, reduced mechanics, Radau-5 and IPOPT/MA57. The center and
Biceps capacity changes of ±0.005 all produced nominal witnesses:

| Sample | Normalized work deficit | Constraint residual | Solve plus candidate audit/export |
| --- | ---: | ---: | ---: |
| Center | 1.679e-7 | 2.161e-10 | 0.365 s |
| Biceps −0.005 | 1.679e-7 | 2.160e-10 | 0.358 s |
| Biceps +0.005 | 1.679e-7 | 2.165e-10 | 0.359 s |

These timings exclude source/auxiliary construction; they are not an online
budget claim. Each auxiliary solve used 21 iterations. The source prepared
digest was unchanged and zero physical cycles were advanced. This small region
was feasible throughout; it supplied no measured frontier or useful deficit
gradient. The runtime artifact is
`asymmetric-sides-r192-rho-bo-20260928/conditional-capacity-smoke-c140-left-20261004/conditional-capacity-report.json`.

## Next validation gates

Choose late certified checkpoints and predeclare wider but local capacity
directions. Repeat uncertain points with alternative starts and a stricter
audit, retain all failures as undetermined, and compare directions conditional
on the complete state. Validate representative nominal witnesses with an
independent continuous DOP853 replay before interpreting them physiologically.

Only then consider an empirically supported local feasibility surrogate, using
held-out complete-state conditions and explicit abstention outside support.
Any distance to sampled feasible capacities must be named as such, not distance
to the infeasible frontier. A later endurance objective would additionally need
matched real-RHO continuation evidence; this prototype implements no such cost.
