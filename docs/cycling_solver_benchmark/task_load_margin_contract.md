# Experimental task-load margin: supervisor contract

`cocofest.optimization.task_load_margin` implements the pure acceptance and
selection layer `task_load_margin_v1`. It does **not** implement an OCP solver,
worker process, or production RHO binding. Existing RHO behaviour is unchanged.

## Physical problem to be implemented by the oracle

At a complete, certified checkpoint, restore all mechanical and Ding states,
the stimulation history, and the prescribed task. Solve one complete cycle,
minimizing `-lambda`, where lambda multiplies the prescribed nominal work (or
resistance in the explicitly identified task). Retain every original constraint,
including cycle closure, state/PW bounds, physiological bounds, and the next-cycle
half-step constraint where present. Scale every associated task-work constraint
consistently. Do not silently substitute positive muscle work for net task work.

The checkpoint and context hashes identify arm, parameters, cadence, task,
constraint definitions, state normalization and solver tolerances. The oracle
must verify archive/model bytes before and after solving using
`TaskReserveCheckpoint.verify_files()`. The application supplies every active
constraint group when creating the checkpoint; the default groups alone do not
establish that a task-specific constraint has been checked.

Solve nominal lambda=1 explicitly as well: a witness at higher work need not
prove nominal feasibility in a nonconvex equality-constrained cycling problem.
A feasible optimum is an achievable-load lower bound, not a certified global
maximum. Solver failures leave the margin unknown, never zero or physiological
failure. A load cap is an artificial numerical bound; reaching it prevents using
that solution's sensitivity and calls for a larger cap and a fresh solve.

## API and data flow

1. Construct a `TaskLoadMarginRequest` from the checkpoint, normalized coordinate
   center, coordinate names, physical domain, trust box, request ID and host
   monotonic timestamp. Normalization definitions belong in the hashed task
   context; gradients and trust radii use these same normalized coordinates.
2. An independent worker returns `TaskLoadMarginResult`, with two full-constraint
   `ProbeEvidence` witnesses (nominal and optimized load), solution artifact,
   gradient, KKT diagnostics and independent local validation. A failed worker
   returns `failure_reason` and leaves missing numerical fields as `None`.
3. `select_task_load_margin` receives only already completed results. It checks
   request identity, provenance, residuals, age, domain and trust, then chooses
   the newest usable source checkpoint, not the largest margin or most recently
   finished stale job. A failed new job may preserve a previous still-valid
   direction. If no valid direction remains, the action is `deactivate`.
4. The fast RHO receives numeric `[weight, center, gradient]` and minimizes the
   additional affine term `-weight * gradient · (terminal_state - center)`.
   The constant margin value is omitted because it does not affect the minimizer.
   There is no implicit factor of 10000. Scale selection still requires ablation
   against the existing fatigue objective and recorded objective contributions.
5. Revalidate age, physical domain and trust region at the **solved terminal
   point**, before accepting a cycle. A violation calls for disabling the extra
   objective and resolving from the same certified boundary; it does not justify
   clipping the point or declaring fatigue failure. No extra physical constraint
   should be inferred from an empirical trust box.

Polling, CPU allocation, process isolation, immutable atomic result publication,
and fallback re-solves are integration duties. The fast RHO must not wait for a
worker. Default acceptance ages are 20 cycles and 20 seconds from **request
creation**, not worker completion. Both thresholds are policy parameters; a
worker that takes too long cannot make an old checkpoint fresh by finishing now.

## Sensitivity convention and scientific gates

`load_gradient_from_kkt` assumes the solver problem minimizes `f=-lambda` and
uses `L=f+mu.T*c`. With physical checkpoint coordinates `p` and normalized
`x=(p-offset)/scale`, it returns:

`d lambda / d x = -scale * (f_p + c_p.T * mu + bound_parameter_contribution)`.

Bound terms are required explicitly because fixing initial states can introduce
parameter-dependent bounds. Passing zero is correct only for parameter-independent
bounds. Solver multiplier signs and objective/constraint scaling must first be
converted to this convention. `multipliers_artifact` can retain the raw solver
evidence; `gradient` can also come from centered finite differences of fully
reoptimized OCPs, labelled `finite_difference_reoptimized`.

Primal feasibility alone does not validate a gradient. Acceptance requires
stationarity, complementarity and dual feasibility residuals below policy limits,
regular local sensitivity, independent gradient checks, and held-out reoptimized
OCP checks of the affine load approximation within the trust region. These
checks remain necessary for finite-difference gradients because inaccurate local
optima also corrupt differences. Residual normalization and any branch/active-set
changes must be recorded by the oracle. One holdout is the API minimum, not a
scientific recommendation for a high-dimensional model; coverage should span
the directions actually explored by the RHO.

Do not freeze fast calcium/force states silently when using only slow fatigue
coordinates: the local envelope is conditional on all other checkpoint states.
Their variation across subsequent cycles must be covered by application-level
domain validation or treated as context invalidation. A slow-only direction can
be tested experimentally, but it is not automatically a value function over
arbitrary fast states/history.

## Validation status

Unit tests use synthetic complete-oracle evidence and an analytic KKT example to
test signs, normalization, provenance, missing evidence, nonfinite diagnostics,
load caps, independent nominal witnesses, stale/out-of-order worker results,
domain/trust rejection and fallback. They do not validate a real cycling load
OCP or an endurance improvement. Those are subsequent integration benchmarks.
