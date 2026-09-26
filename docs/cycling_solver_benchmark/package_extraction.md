# Incremental cycling engine extraction (#8)

The first extraction boundary is trajectory persistence and result interpretation.
It builds on the contracts introduced for issues #6 and #7:

| Responsibility | Package module |
| --- | --- |
| Physical solution metadata and validation | `cocofest.optimization.solution_archive` |
| Legacy NPZ arrays, trajectory adapter, checkpoint dispatch | `cocofest.optimization.trajectory_io` |
| Durable checkpoint receipts and model fingerprint | `cocofest.optimization.configured_rho_checkpoints` |
| Result timing populations and audit availability | `cocofest.evaluation.result_schema` |

These modules do not import the cycling examples or instantiate a solver.
The existing periodic driver's `_WarmupSolutionAdapter`, `_save_warmup_cache`,
`_load_warmup_cache`, and `_save_rho_replay_checkpoint` names remain compatible.
They delegate persistence to the package. Loading an explicitly versioned
physical archive validates its declared contract; unversioned NPZ seeds retain
the legacy interpretation.

`configured_checkpoint_writer` uses a scoped package publication hook by default.
This removes its dependency on importing and monkey-patching the full example
driver. The hook receives the raw writer and can publish an atomic seed/receipt
pair. Context-local restoration handles nested scopes and exceptions. The
optional `module=` injection remains supported for existing consumers and tests.
The configured-model factory context still patches solver construction aliases;
its removal belongs to a later solver-construction boundary.

This step does not migrate `solve_case`, change solver options, or change the
mathematical problem. The example still owns the mapping from solver arguments
to physical metadata.

The second extraction boundary introduces
`cocofest.optimization.cycling_problem.CyclingProblemFactory` and
`CyclingRunContext`. The periodic driver now passes its already validated model,
MHE data, cycle data and simulation conditions through this package boundary
before it configures the resulting live NMPC. The builder is injected: this
retains the historical Bioptim construction while permitting package-only tests
and future model/solver factories without importing `examples`.

The next step is moving the remaining solver orchestration into explicit
backends. Issue #8 remains a parent tracking that progressive migration.

Validation covers package-only checkpoint publication, old/new archive reads,
legacy driver imports, scoped hook restoration, fingerprint receipts and replay.
The focused suite is `tests/test_trajectory_io.py`,
`tests/test_configured_rho_checkpoints.py`, `tests/test_solution_archive.py`, and
`tests/test_result_schema.py`.
