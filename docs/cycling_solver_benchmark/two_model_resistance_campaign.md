# Two Ding variants, four ordered conditions

`scripts/build_two_model_resistance_campaign.py` declares eight comparison
jobs without launching a process or constructing an OCP. It resolves model
defaults using the runtime adapter and reads NPZ metadata. Use the scientific
Python environment with `MPLBACKEND=Agg`.

For each of exactly two distinct effective muscle-parameter configurations:

1. Baseline RHO: uniform fatigue objective, one-cycle horizon.
2. RHO-Physio: fixed declared initial muscle weights, one-cycle horizon.
3. RHO-PACE: the same initial weights and objective family, adaptive updates.
4. Baseline FHO: uniform fatigue objective, one finite horizon covering the
   requested number of cycles.

All jobs use dynamic reduced mechanics, the same reduced-profile contents,
one explicit positive constant resistance, IPOPT/MA57 and the same declared
discretization. Each model's four jobs consume its own identical seed. The
wrapper applies the same resolved model configuration to all four conditions,
including warmup factories. Baseline objectives do not consume physiological
weights. FHO is neither a weight-calibration input nor proof of globally optimal
endurance. This protocol tests weighted relative-capacity-loss-squared costs;
it does not reproduce the article's integral of weighted RMS absolute loss.

## First: prepare seeds for both variants

Preparation and certification are mandatory **before sealing the campaign**,
and are outside the eight comparison OCPs. For each variant, run the configured
wrapper with `--condition rho --model-config MODEL.json`, explicit fixed
positive `--signed-crank-torque`, dynamic/reduced mechanics, its reduced profile,
and `--common-initial-solution-output SEED.npz`. Use fresh output paths and no
external seed for the initial preparation: the wrapper disables historical
unversioned seeds. The preparation command may enable standard warmup. Its
result, exported first converged target-window seed and configuration audit
must then be checked physically; successful launcher completion alone is not
certification. A seed from another model, including a nominal seed reused for
a changed Ding model, is refused even if its trajectory looks plausible.

After actual review, create a certificate JSON with this schema. File records
are absolute `{ "path": "...", "sha256": "..." }` records returned by
`scripts.build_resistance_comparison_campaign.artifact`:

```json
{
  "schema_version": 1,
  "stage": "per_model_seed_preparation",
  "muscle_parameter_fingerprint": "full resolved model fingerprint",
  "seed": {"path": "/absolute/SEED.npz", "sha256": "..."},
  "model_config": {"path": "/absolute/MODEL.json", "sha256": "..."},
  "reduced_profile": {"path": "/absolute/PROFILE.npz", "sha256": "..."},
  "resistance_nm": 0.1,
  "gates": {
    "seed_physically_certified": true,
    "same_initial_state_and_model_parameters": true,
    "dynamic_constant_resistance_verified": true,
    "validation_thresholds_declared": true
  },
  "evidence": [{"path": "/absolute/seed-validation.json", "sha256": "..."}]
}
```

The review must cover solver residuals, state/PW/force bounds, mechanical
replay, seed chronology and configured-model build receipts. Include the
preparation result, configuration audit and physical validation report among
the evidence records. The builder verifies the certificate's bindings,
reviewer-declared gates, evidence hashes and actual seed NPZ fingerprint; it
does not independently establish physical truth. Never set the gates from
process exit status alone. The fingerprint is resolved by
`resolve_model_config` and written into seed metadata by the configured wrapper.

## Then: launch two four-arm DAGs

Declare the variants in a JSON list; paths may be relative to this descriptor:

```json
[
  {"model_id": "ding_a", "model_config": "ding_a.json",
   "seed": "ding_a/seed.npz", "seed_certificate": "ding_a/certificate.json",
   "reduced_profile": "profile.npz", "weights_config": "weights_a.json"},
  {"model_id": "ding_b", "model_config": "ding_b.json",
   "seed": "ding_b/seed.npz", "seed_certificate": "ding_b/certificate.json",
   "reduced_profile": "profile.npz", "weights_config": "weights_b.json"}
]
```

Model configs follow `run_configured_cycling_benchmark.py` schema version 1:
`case_id`, `provenance`, and four named muscles (`Delt_ant`, `Delt_post`,
`Biceps`, `Triceps`) each declaring `Fmax`, `a_scale`, `alpha_a`, `tau_fat`.
Supported optional Ding fields are resolved and included in the fingerprint.
Each weights config requires provenance `initial_weight_basis`, four positive
named `initial_weights`, and its PACE `policy`. Physio and PACE share that file;
the condition selects disabled/enabled adaptation. Zero published min–max
weights are not silently floored: choose and document an admissible positive
weight family before this protocol. The seed certificate is not a certification
of the scientific validity of that choice.

Example declaration (replace paths and CPU IDs with verified local values):

```bash
MPLBACKEND=Agg python scripts/build_two_model_resistance_campaign.py \
  --campaign-id ding-comparison --resistance-nm 0.10 --cycles 100 \
  --variants /absolute/variants.json --output-directory /absolute/results \
  --manifest /absolute/campaign.json --hsl-library /absolute/libcoinhsl.so \
  --cores-per-job 4 --numeric-threads 1 --cpu-ids 0,1,2,3,4,5,6,7
```

This writes only the sealed manifest. It contains complete argv, working
directory and environment for each job, distinct result/configuration-audit
paths, weighted-condition journal paths, and disjoint `taskset` CPU masks.
MA57's explicit library path and SHA-256, Python, taskset, inputs, seed evidence
and relevant implementation files are sealed. Changes require a new manifest.

`next_jobs(manifest, reviews, running_job_ids=...)` returns at most two eligible
jobs, minus the slots occupied by reported running jobs. Its review mapping is
`{manifest_sha256, models: {model_id: {initial: REVIEW, baseline_rho: REVIEW,
rho_physio: REVIEW, rho_pace: REVIEW, baseline_fho: REVIEW}}}`. Every review
declares the listed gates and unchanged evidence; result reviews additionally
need `outcome` and `certified_executed_cycles`. Require the exact result,
configured-model audit, and weighted journal where applicable in the evidence.
Initial review also checks variant weights were not calibrated from FHO.

A caller launches returned argv with its supplied environment and working
directory, reports **all** running jobs, then physically reviews each finished
job before requesting successors. This is a planner, not an OS scheduler:
unreported/manual processes cannot be constrained. At most one job per model
is eligible, in the order above. The two independent chains need not wait at
a global stage barrier. A zero-cycle or unreviewed failure blocks its chain;
a reviewed nonempty physical prefix may advance only with the explicit
`numerical_or_unresolved_stop` classification. Numerical failure is never
called fatigue. Completing a finite cycle budget is not an endurance limit.

`cores-per-job` counts logical OS CPU IDs. Select one ID per physical core if
SMT isolation is required; no physical-topology inference is performed.
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` equal
`numeric-threads`; benchmark worker threads equal their integer quotient
`cores-per-job / numeric-threads`. The product bounds declared parallelism;
the affinity mask bounds where actual runtime threads execute. Numeric thread
count 1 is the simple default. The caller should reserve these CPUs and account
for other running applications before launching anything.

## Fast validation only

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q tests/test_two_model_resistance_campaign.py \
  tests/test_resistance_comparison_campaign.py
```

These tests create synthetic artifacts and test declarations, provenance,
seed/model refusal, output protection, review gates and scheduling. They run
no optimization and provide no new closed-loop or clinical evidence.
