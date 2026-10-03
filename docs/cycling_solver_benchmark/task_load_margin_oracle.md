# One-cycle attainable-load oracle

The experimental supervisor is built by `scripts/probe_independent_rho_task_load_margin.py`
from a certified bilateral checkpoint receipt. It creates independent unilateral
programs, verifies source model/configuration/archive digests, restores the full
prepared muscle/mechanical state and stimulation history, and audits the nominal
work solve explicitly. It then builds a second program to maximize attainable
terminal work. The load factor is analytically eliminated as
`lambda = E_prod(T) / nominal_work`: minimizing `-lambda` needs no new state.

All pre-existing objectives are assigned zero weight. Only the terminal E_prod
equality is replaced by `[0, nominal_work * cap]`. All remaining dynamics,
PW/path bounds, closure and next-half-cycle constraints remain unchanged.
Consequently this is a terminal-work margin for the current task constraints,
not a uniformly rescaled mechanical-load problem. In particular, the next
half-cycle guard still enforces its original prescribed demand. A solution on
the artificial cap is rejected for direction activation by the contract.

The independent audit recomputes every discrete NLP constraint and checks all
decision bounds, plus explicit terminal work and complete PW trajectories.
It also evaluates the actual stationarity, complementarity and signed bound
multiplier residuals from CasADi's full scaled graph and Bioptim multipliers.
Missing dual information remains unavailable; no diagnostic is manufactured
from a successful solver status. This audit is discrete feasibility and does
not replace an independent continuous-ODE replay.

`--finite-difference-step 0.001` additionally rebuilds and reoptimizes two
problems per muscle, moving its initial normalized `A/a_scale` coordinate and
holding all unselected initial states and history fixed. Every perturbed result
has its own full feasibility audit and saved primal/dual vectors. A failed or
capped perturbation rejects the derivative. This optional direction remains
disabled until smaller-step gradient validation and independent held-out
complete OCPs establish local fidelity and regularity. These selected A-only
coordinates do not describe all Ding fatigue states.

The CLI pins its process to a requested CPU. Launch it as a separate worker;
RHO does not synchronously wait for it. JSON and primal artifacts are published
with atomic replacement, and an existing run directory is refused. There is
no production RHO reader yet; the acceptance contract must gate any future
reader by context hashes, state trust region, elapsed time and cycle age.

Example (from repository root; choose a free CPU):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/probe_independent_rho_task_load_margin.py \
  --receipt asymmetric-sides-r192-rho-bo-20260928/task-reserve-rho-instrumented-20260930/results/checkpoints/cycle-140/receipt.json \
  --side left --cpu 5 --upper-bound 3 --tolerance 1e-6 \
  --output-directory asymmetric-sides-r192-rho-bo-20260928/task-load-margin-new-run
```

## First real test, 2026-10-01

At the left-arm cycle-140 checkpoint, the nominal work is feasible and the
optimized load witness is **1.1802101892 times nominal**. The full 4171-row,
4288-variable audit gives normalized constraint violation `7.13e-7`,
stationarity `4.44e-8`, complementarity `9.43e-8`, and dual feasibility zero.
The optimizer itself takes 0.572 s; rebuilding plus solving and auditing the
load problem takes 7.43 s (nominal construction/solve/audit adds 5.05 s).
This supports scheduling on a separate CPU, with reusable graph construction
needed for frequent updates. It does not establish a global load maximum,
gradient fidelity, endurance gain, or physiological failure threshold.

Evidence is in
`asymmetric-sides-r192-rho-bo-20260928/task-load-margin-c140-left-20261001/result.json`.
The direction is intentionally reported as disabled.

The optional central-difference experiment at the same checkpoint, normalized
step 0.001, completed all eight perturbed OCPs with maximum audit violation
`7.14e-7`. The derivative in order `(A_Delt_ant, A_Delt_post, A_Biceps,
A_Triceps)/a_scale` is `(0.117650, 0.111617, 0.443394, 1.153530)`.
Total wall time including nominal, center and eight rebuilt probes is 72.43 s.
Artifacts are in `task-load-margin-c140-left-fd-20261001` under the same campaign.
This establishes a working reoptimized derivative path, not yet its fidelity
under changes of finite-difference step or independent held-out perturbations.
# Local sensitivity and guarded RHO replay (2026-10-01)

The new `initial_bound_envelope_gradient` maps normalized initial states to
the actual scaled decision vector symbolically, then applies the bound-dual
envelope for min(-lambda). Fixed variable offsets are not assumed. It requires
finite complete multipliers and matching fixed physical bounds. On the left
arm source at cycle140, its A-only gradient differs from the independently
reoptimized derivative by at most 1.895e-4. This avoids eight extra OCP solves
merely to *estimate* the four A sensitivities; independent validation remains
necessary.

`scripts/validate_task_load_margin_direction.py` restores every source state,
bound and stimulation history before each independent perturbation. Reusing
the same objective graph does not reuse a previous point's optimum or boundary.
Every observation stores primal and complete scaled-NLP KKT diagnostics plus
trajectory/multipliers. Reuse of identical derivative evidence at a fixed source
is explicit and preserves its original artifact paths.

Measured local validations, all at the same c140 left source:

| Coordinates/domain | Independent evidence | Maximum derivative error | Maximum held-out lambda error | Result |
|---|---|---:|---:|---|
| A/a_scale, radius .01 | 8 smaller-step OCPs + 8 mixed holdouts | 6.71e-10 (FD vs FD) | 6.50e-8 | Pass, conditional on other states |
| All 20 Ding states, small box | 40 FD OCPs + 8 mixed holdouts | .001091 (KKT vs FD) | 1.635e-5 | Pass locally |
| All states, isotropic .05 except positivity | First wider holdout | Unknown | Unknown | Refused: complete NLP violation .04934 |
| Periodic box: Cn .001, A/Km .01, F/Tau1 .03 | Same 40 FD + 8 new holdouts | .001091 | .0001140 | Pass locally |

F_Delt_ant's symmetric radius is additionally limited to .9 times its positive
normalized center. Eight holdouts in 20 dimensions are empirical checks, not
a guarantee on every point of the box. The result is not a globally optimal
load bound, endurance model, or continuous-time feasibility certificate.

`TaskLoadMarginObjectiveBinding` adds a nonquadratic affine terminal objective
through the existing fixed reserve parameter channel, coexisting with fatigue
parameters. It validates source/model/task hashes, artifact integrity, age,
coordinate domain, trust and conditioned context. A failed terminal gate is
handled by disabling the extra objective and re-solving from the same physical
boundary before any state transfer.

The current replay adapter only accepts complete Ding coordinates and the
fixed-history periodic-node model. In this model, PW changes force recruitment,
not the periodic calcium impulse amplitude. Its conditioned context checks
cadence, stimulation regime, and crank phase modulo 2*pi; the per-cycle work
accumulator resets to zero. Additional unmodeled states/model classes are
refused.

`scripts/run_task_load_margin_replay.py` is explicitly **offline**, preserving
source timestamps and allowing wall age 1e9 s while retaining the 20-cycle age
limit. It uses a single validated direction; it is not an online refreshing
supervisor. Augmented-coordinate metadata are reannotated with a new context
hash while referencing the untouched original source and trajectory evidence.
Physical source arrays and original fixed parameters are restored with the
strict archive adapter; the new objective-only parameter starts identically
zero and its addition is explicitly reported.

Three-cycle replay evidence:

- Small box, weight10000: three certified cycles, zero committed active-cost
  cycles; the terminal trust gate triggers a successful ordinary-RHO fallback.
- Periodic box .03, weights1000 and10000: three certified cycles, zero active
  commits. At weight1000, F_Delt_post changes by .03556 at the first proposal,
  exceeding the validated .03 radius.
- Periodic box .03, weight100: three certified cycles, one active commit
  (cycle141, IPOPT solve .456 s), then ordinary-RHO fallback as the fixed source
  becomes too distant. This verifies integration, not improved endurance.

Artifact root: `asymmetric-sides-r192-rho-bo-20260928/`, under
`task-load-margin-c140-left-full-validated-20261001`,
`task-load-margin-c140-left-full-tube03-validated-20261001`, and
`task-load-margin-c140-left-tube03-replay3-w100-20261001`.

An endurance run of this replay would measure a *one-source local intervention*
followed by the baseline. Report its active-commit count alongside its cycle
count; do not describe it as a fully refreshed load-margin supervisor.
