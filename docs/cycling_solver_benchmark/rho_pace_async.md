# Asynchronous predictive PACE

`scripts/run_rho_pace_benchmark.py` runs `predictive_moment` proposals in a
separate spawned process by default (`projection_async=true`). The main RHO
callback copies the certified solution's numerical arrays and muscle
parameters. Neither the live OCP nor its symbolic model enters the worker.
The worker has no objective writer and has a lower OS scheduling priority
on systems supporting `nice`.

An optional `projection_worker_cpu_ids` field pins the child to dedicated CPU
cores.  This matters when the RHO launcher itself uses `taskset`: otherwise
the spawned child inherits the RHO core and a long projection steals time from
the fast solve.  An unsupported or denied requested affinity is a recorded
worker failure; it is never silently ignored.

At every certified RHO boundary, the launcher checks for a completed
proposal without waiting. A valid result is applied through the original
transactional OCP writer at that boundary, including between the scheduled
slow calls. A missing result holds the current weights. At most one job is
running; slow calls cannot create a backlog.

The acceptance budget is
`min(projection_budget_seconds, update_every_cycles * target_cycle_seconds * projection_budget_fraction)`.
Defaults `K=20`, `target_cycle_seconds=1`, and fraction `0.8` give 16 seconds,
even if an old configuration declares a 600-second projection budget.
This includes spawn/import/transport time. A result is rejected if its
completion time exceeds that budget, its source is older than K cycles, or
its incumbent weights no longer match the current OCP weights. Expired work
is terminated at the next RHO boundary. Process shutdown never waits for
the outstanding rollout.

The initial horizon is the configured `projection_horizon_cycles`; it is
also its hard upper bound. A timeout halves the next horizon. Completed
evaluations adapt it towards 70% of the deadline, with increases limited to
25% per call. Thus a requested cap of 100 remains a cap, not a promise that
every hardware configuration can compute 100 cycles for every candidate
within one call period. Horizon reduction is recorded in the journal.

For an actually evaluated 300-cycle rollout at this workload, use a slower
supervisory cadence, e.g. `K=60`, `projection_budget_fraction=0.9` (54 s),
and a distinct worker core.  The RHO OCP still solves once per cycle and is
not blocked; the cost is updated only when the independent proposal finishes
while its 60-cycle source-age gate remains valid.  This is a two-speed
configuration, not a claim that a 20-cycle/16-s configuration evaluates H=300.

`projection_submitted` and `projection_completed` events record source and
application cycles, incumbent weights, actual horizon, effective budget,
worker wall time and rejection reason. The final boundary event separately
records whether the OCP cost was actually updated. Numerical rollouts remain
heuristic projections, not certified future RHO trajectories.

This architecture removes synchronous rollout latency from the RHO loop.
It does not certify hard real-time OS scheduling or a total RHO mean of one
second: that requires timing the solver, certification, callback, transport
and any CPU contention together on the actual campaign. Setting
`projection_async=false` retains synchronous operation for diagnostics,
with the same effective supervisor budget (checked between candidates).

Focused verification:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python -m pytest \
  tests/test_rho_pace_async.py tests/test_rho_pace.py -q
```

Tests include real spawned-process results, a deliberately slow child that
does not block 19 simulated RHO boundaries, deadline cancellation, stale and
wrong-incumbent rejection, and transactional application at the next
certified boundary.
