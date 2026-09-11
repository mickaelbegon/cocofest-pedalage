# RHO-PACE: Physiological Adaptive Cost for Endurance

This experimental comparison arm changes the actual quadratic RHO fatigue
cost to `10000 * sum_m(weight_m * (1-A_m/A_scale_m)^2)`, with the same original
time integration. It uses dynamic reduced mechanics and a constant positive
resistive crank torque for negative rotation. It is limited to 100 RHO cycles.
It does not implement an isokinetic load or a learned FHO value function.

The isolated launcher wraps `FesNmpcMsk.solve_fes_nmpc` during its own process
and replaces the fatigue objective through Bioptim's public
`update_objectives` API. It preserves the original benchmark callback and
its stopping/physical validation logic. It supports IPOPT only, with C
compilation disabled and compact output enabled. No pre-existing source
file is modified by this integration.

The same adapter supports **RHO-Physio**, the static initial-physiological
condition, with `policy.adaptation_enabled=false`. Explicit named initial
weights and their provenance are mandatory; the cost is applied once and
then held. The configured four-condition runner, schemas and model/seed
compatibility rules are described in `configured_cycling_conditions.md`.

## Declared physiological policy

Every five certified cycles by default, target log weights are
`log(initial_weight) - gain * log(A/A_scale)`. The target is projected into
the declared relative-weight box with zero mean log weight. The current
log weights move toward it with smoothing 0.2 and maximum per-update change
`log(1.1)`. A more depleted muscle is penalized more. This causal heuristic
uses only the completed RHO state and declared muscle parameters. Its
endurance benefit has not been established; it is not the separate
candidate-rollout supervisor or a physiological endurance certificate.

For physiological initialization, use the strictly positive raw weights or
the admissible **max-normalized** weights returned by
`calculate_physiological_muscle_weights`, after checking
`usable_for_controller=True`. Never use legacy min-max display values,
which can assign a zero cost to one muscle. Record the exact calibration
case, parameters, geometry source and normalization in `initial_weight_basis`.
The launcher requires an exact mapping from actual model muscle names to
these weights. It has no FHO input and does not silently infer a calibration.

The controller geometrically normalizes supplied positive weights and then
projects them into the declared box (default 0.25 to 4). Small positive
physiological weights remain admissible; supplied, normalized and projected
weights are all journaled. Projection can alter their relative ratios.
Adjust the bounds explicitly if preserving the calibration's wider range
is important. The common geometric scale stays one; the global fatigue
objective multiplier is retained.

In particular, article `raw/max` inputs are preserved verbatim as
`supplied_initial_weights`. PACE's applied weights are a **separate relative
cost normalization**, explicitly marked `applied_weights_are_article_raw_max=false`.
Geometric normalization preserves their ratios before box projection;
`initial_projection_changed_ratios` records whether the projection changed
them. Applied weights must not be labeled article `raw/max` weights.

## Launch

The checked-in `rho_pace_uniform_start.json` is an executable **uniform-start
control**, not a physiological calibration. Use it to validate the adapter
and replace the weights/basis with an audited physiological calculation for
the physiological arm. Pass the same seed, dynamics, resistive torque,
transcription, solver tolerances and termination options as baseline RHO.

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cocofest-mpl \
  /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/run_rho_pace_benchmark.py \
  --pace-config docs/cycling_solver_benchmark/rho_pace_uniform_start.json \
  --pace-journal /tmp/rho-pace-smoke/pace.jsonl -- \
  --solvers ipopt --objective fatigue --objective-shape quadratic \
  --formulation dynamic --mechanical-formulation reduced \
  --signed-crank-torque 0.22 --cycles-per-window 1 --n-windows 2 \
  --ipopt-profile periodic_collocation --ipopt-collocation-degree 5 \
  --ipopt-collocation-method radau --stimulations-per-cycle 30 \
  --compact-rho-output --ipopt-use-sx --ipopt-linear-solver mumps \
  --ipopt-max-iter 100 --n-threads 1 \
  --output-json /tmp/rho-pace-smoke/result.json
```

The example runs a fresh initial solve using benchmark defaults; for the
scientific comparison also pass the campaign's explicit common certified
seed and reduced profile. Extend to `--n-windows 100` only after its smoke
run is physically validated. The PACE journal must be a new path; an existing
file is refused to prevent mixing runs.

The standard benchmark result and the PACE JSONL sidecar must be retained
together. Original unweighted fatigue metrics remain suitable common
comparison metrics; they are not the newly weighted optimization objective.
The sidecar records parameters, signed resistance, initial and adapted
weights, update receipts, every completed solver status, certification,
holds/refusals and final launcher status. A missing/unrecognized objective,
failed update receipt or unsupported mode aborts the arm. It must be marked
unavailable rather than reported as successful adaptive RHO. Completion of
the launcher itself does not certify physical completion of the campaign.

## Verification and limits

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cocofest-mpl OPENBLAS_NUM_THREADS=1 \
  /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  -m pytest -q tests/test_rho_pace.py
```

Tests include two actual Bioptim/IPOPT/MUMPS solves with a changed cost,
symbolic value and gradient, slow bounded updates, initial projection,
strict refusal behavior and launcher mode validation. The tiny integration
test proves objective injection; it does not validate a cycling endurance
trajectory. A full dynamic PACE cycling comparison remains to be run.
