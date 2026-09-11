# Four configured cycling conditions

The isolated `scripts/run_configured_cycling_benchmark.py` runner exposes:

```text
--model-config PATH
--condition {rho,rho-physio,rho-pace,fho}
[--weights-config PATH --weights-journal PATH]
[--configuration-audit PATH]
-- <common cycling_fes_solver_comparison arguments>
```

All four conditions apply the same explicitly configured Ding model before
constructing OCP dynamics. `rho` retains the original unweighted objective.
`rho-physio` applies the initial physiological cost weights once, then keeps
them fixed. `rho-pace` starts with those same weights and adapts them on the
declared slow schedule. `fho` calls the canonical benchmark in `--single-shot`
mode with the same model adapter; its `--cycles-per-window` must equal
`--n-windows`. FHO does not supply data to either physiological controller.

Dynamic reduced mechanics, positive explicit signed resistive torque, IPOPT,
an explicit `--output-json`, and at most 100 cycles are required. Weighted
conditions additionally require explicit named weights and a journal, as
well as the existing PACE restrictions (quadratic fatigue cost, no IPOPT C
compilation, compact output). Output paths must be fresh and distinct. The
runner restores its model factory aliases and working directory on return.

## Model configuration

The schema is `configured_cycling_model.schema.json`. A version-1 config has
exactly `schema_version`, `case_id`, `provenance`, and `muscles`. The latter
maps every actual muscle name to an explicit parameter mapping. For the
current cycling model the names are Delt_ant, Delt_post, Biceps and Triceps.
Each muscle requires `Fmax`, `a_scale`, `alpha_a`, and `tau_fat`.

Optional supported Ding fields are `alpha_tau1`, `alpha_km`, `tau1_rest`,
`km_rest`, `tauc`, `tau2`, `pd0`, and `pdt`. Omitted fields are resolved from
the installed Ding pulse-width-with-fatigue class, and all effective values
are included in the audit and fingerprint. Unknown fields, invalid signs,
non-finite values and mismatched muscle names are refused. `Fmax` maps to
the model's `fmax`; `a_rest` is synchronized with `a_scale` and cannot be
independently overridden. No arbitrary attribute or executable-code mapping
is accepted.

For the proposed Triceps-only variants, keep every other value fixed and
set Triceps `alpha_a` to -0.12 or -0.48, against nominal -0.24. This is a
parameter sensitivity comparison, not a clinically calibrated subject model.
The wrapper patches both the regular factory and the alias used by standard
warmup/refinement, before construction of the symbolic OCP.

## Static and adaptive weights

The schema is `rho_physiological_weights.schema.json`:

```text
initial_weights: {actual_muscle_name: strictly_positive_number, ...}
initial_weight_basis: nonempty provenance of calibration/geometry/raw-max calculation
policy: {adaptation_enabled: boolean, ...existing PACE policy settings...}
```

The configured runner derives `adaptation_enabled` from its condition:
false for `rho-physio`, true for `rho-pace`. Thus the same weights file can
serve both arms, without making configuration copies that drift apart.
The lower-level PACE launcher also supports a static config by setting
`policy.adaptation_enabled=false`; that mode requires explicit initial
weights and provenance and emits arm `RHO-Physio`.

RHO-Physio journals exactly one initial `applied` event followed by `held`
events with reason `static_physiological_weights`, including boundaries at
which PACE would adapt. It never calls the cost writer again. Both conditions
apply the existing initial geometric normalization and declared box
projection identically, preserving the original supplied article `raw/max`
values separately. Applied relative weights must not be mislabeled article
`raw/max` values. Use identical box bounds to isolate adaptation in the
static-versus-adaptive comparison.

## Seed compatibility and output provenance

The model fingerprint is SHA-256 over canonical complete effective muscle
parameters. The case name and descriptive provenance do not change it.
All explicit common-initial, standard-warmup and full-horizon-prefix NPZ
inputs must contain the matching `metadata__json.muscle_parameter_fingerprint`.
Unversioned seeds and seeds from another variant are refused **before the
solver is called**. Historical pickled initial guesses are disabled because
they do not have this compatibility metadata.

Consequently, an existing nominal seed cannot be reused for either
Triceps-alpha variant. Generate initial trajectories through this runner
without external seed arguments first, so its warmup uses the configured
model. The wrapper adds the fingerprint, complete parameter mapping, case
and condition to newly generated explicit NPZ outputs after the benchmark
returns. That metadata establishes the model identity; it does not establish
physical certification. Continue to require the benchmark's solver and
physical certificates before using a generated trajectory as a seed.

Templated checkpoint outputs are currently refused because this wrapper
cannot reliably identify every generated path to annotate it. Use explicit
`--common-initial-solution-output`, `--receding-horizon-solution-output` or
`--rho-replay-checkpoint-output` paths.

The configuration audit defaults to the result JSON's stem plus
`.configuration.json`. It includes the condition, resolved parameters,
fingerprint, config hashes, seed checks and every model factory invocation.
For weighted arms retain that audit, the benchmark result and weight journal
together. A completed launcher is not a physical success certificate; the
benchmark result remains authoritative for the validated prefix.
