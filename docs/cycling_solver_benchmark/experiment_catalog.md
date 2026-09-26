# Experiment artifact catalog (issue #12, first tranche)

The catalog indexes existing `result.json` artifacts without moving, deleting, or
rewriting them. It has no solver dependency and does not change simulation
execution. New campaign directories can follow the existing convention
`local-results/<campaign>/<stage>/<solver>/result.json`; historical layouts remain
readable. Artifact publication and campaign launch are outside this first tranche.

## Commands

Run from the repository root or with the package installed:

```bash
python -m cocofest.evaluation.experiment_catalog index local-results/frequency-30hz-100cycles-20260922 --dry-run
python -m cocofest.evaluation.experiment_catalog index local-results/frequency-30hz-100cycles-20260922 --output /tmp/cocofest-catalog.json
python -m cocofest.evaluation.experiment_catalog query /tmp/cocofest-catalog.json --solver ipopt --stage K7 --frequency-hz 30
```

`index` accepts several campaign roots or individual result files. Overlapping
roots are deduplicated. Without `--output`, it prints the proposed catalog;
`--dry-run` also suppresses writes when `--output` is provided. An existing catalog
is replaced atomically. A result artifact cannot be used as the output path.
Exit code 1 means some roots/files/entries could not be indexed; valid records
and a structured error list are still emitted. An active producer's incomplete
JSON is reported as an error; rerun indexing once it finishes writing.

## Schema v1

The root has `schema_version`, `roots`, `scanned_file_count`, `records`, and
`errors`. Each result entry becomes one record:

| Field | Meaning |
| --- | --- |
| `id` | Absolute file URI and result entry index/key |
| `artifact` | URI, SHA256 of the original complete bytes, byte count, source schema version, entry selector |
| `dimensions` | Reported solver, linear solver, formulation, mechanical/Ding formulation, frequency, stage |
| `dimension_sources` | How frequency and stage were obtained |
| `status` | Observed success/error fields and conservative catalog classification |
| `metrics` | Existing scalar timing/cycle metrics and timing-population metadata |
| `audits` | Existing mechanical/DOP853 evidence, with no new certification |

The index handles the standard comparison document (`configurations` keyed by
solver plus a `results` list), keyed results maps, and standalone result objects.
Standalone configuration can be provided in `configuration`. Unsupported or
malformed result entries are reported explicitly. The SHA256 covers fields that
the lightweight index omits as well as indexed fields; it detects later source
changes. This is a snapshot hash, not an immutable archive or a physical problem
identity hash.

Frequency comes from explicit frequency fields, from the recorded calcium
stimulation interval, or from stimulation count divided by an explicit cycle
duration. A count alone is insufficient. Stage comes from explicit configuration
or a conventional `K7`, `K2_corrected`, etc. path component. Missing dimensions
remain null. Queries match solver/formulation/stage exactly and frequency with a
small numerical tolerance. `formulation` is the physical mode (`dynamic`,
`isokinetic`, etc.), while `mechanical_formulation` retains the reduction choice.

`reported_success` only means the source's `success` field was true. A record can
still contain `physical_success=false` or failing audits. Missing audits never
imply passing, and a solver-only success never becomes scientific success.
Native/hybrid timing fields retain their original names and units; this index
does not average populations or compare internal objectives across solvers.
Legacy nonfinite values become null so the catalog is valid JSON.

## Library use and remaining work

`index_campaigns(roots)` returns the catalog dictionary;
`query_catalog(catalog, solver=..., formulation=..., frequency_hz=..., stage=...)`
returns matching records. Both are solver-independent. Source artifacts remain
the authority for detailed trajectories, full configurations, and audits.

Next tranches can add a declarative campaign identity linked to resolved
configuration hashes, seed/model/environment provenance, and a versioned compact
catalog used for annex tables. This implementation intentionally keeps automatic
artifact migration and scientific revalidation outside indexing.
