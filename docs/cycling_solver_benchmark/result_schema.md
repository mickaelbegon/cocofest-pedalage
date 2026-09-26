# Benchmark result contract (version 4)

`cycling_fes_solver_comparison.write_benchmark_summary` writes strict JSON with
`schema_version: 4`. Existing version-3 scalar metrics and detailed audit fields
remain present. Readers accepting version 3 can continue reading these fields;
readers validating the version number must accept 4 explicitly.

## Windows and physical cycles

A window is one optimization problem. A monolithic FHO of 200 cycles requests
**one window and 200 cycles**. A RHO advancing one cycle 200 times requests
200 windows and 200 physical cycles, even when each look-ahead window contains
several future cycles. `attempted_windows` records actual solves. Recovered
solutions and failed attempts retain their existing detailed fields.

Each serialized solver row now includes `requested_windows` separately from
`requested_cycles`, `validated_windows`, `nlp_validated_cycles`, and
`physically_validated_cycles`. The last two are different certificates: a
solver-valid trajectory need not pass the independent mechanical checks.
Failed FHO construction retains its requested horizon and records zero solves.

## Timing populations and units

`timing_populations` describes the exact zero-based window indices behind the
warm metrics. Its `hot` population contains the validated prefix after window
0, including any hybrid certificate. `target_solver_only_hot` excludes hybrid
certifiers. Every population records separate finite sample counts for solver
time, wall time, and iterations. Missing or non-finite values do not enter the
corresponding percentile.

The median and P90 of hot solve times have units **s/window**. They are not
automatically s/cycle when a window covers more than one cycle. Per-cycle
times are the sum of validated-prefix solve times (including the first solve)
divided by the number of physically validated cycles. A monolithic FHO has no
hot population, so its hot quantiles are `null`, regardless of horizon length.
The legacy total runtime, setup, compilation, and orchestration fields retain
their meanings. Diagnostic timings belong outside the solve timings.

## Audit evidence and availability

`audit_registry` indexes each detailed audit field with a role, scope,
availability status, number of records, and `passes_tolerance` when that
predicate actually exists. Status is `not_recorded`, `unavailable`, `partial`,
or `recorded`. A recorded DOP853 diagnostic is evidence of a calculation, not
an automatic scientific pass. Missing audits are not passing audits.

For FHO single-shot runs, requested diagnostics now run independently of
`echo`. The JSON retains `integrator_map_initial_guess`,
`integrator_map_final_solution`, and
`integrator_map_final_solution_wall_time_s`. The final solution is inspected
at the first interval, the end of the first cycle, the middle of the horizon,
and the final interval (duplicate indices collapse). Each map resets DOP853
at the optimized state of that interval and compares its endpoint. These maps
measure **local consistency**, not continuous open-loop drift.

The existing RHO continuous-prefix evaluator resets the model clock at each
RHO cycle. It is not a continuous-clock monolithic FHO replay. A requested FHO
full replay therefore explicitly records
`available: false, reason: single_shot_requires_continuous_clock_replay`,
with a pointer to the available local maps. A compatible external continuous
replay may still evaluate the exported FHO archive. This limitation is visible
in JSON rather than silently omitting the requested audit.

`--parametric-kkt-audit` now runs for certified single-shot NLP solutions as
well as RHO. It records window 0 and the covered cycle count. Failed or
unavailable derivative/KKT evaluations preserve their reason and do not alter
the primal or mechanical certificate.

## Verification

`tests/test_result_schema.py` covers window/cycle units, hybrid timing
populations, missing versus failed audits, verbosity-independent FHO evidence,
diagnostic failures, and failure during FHO setup. Existing periodic benchmark
regressions verify serialization and the two-cycle single-shot certificate.
