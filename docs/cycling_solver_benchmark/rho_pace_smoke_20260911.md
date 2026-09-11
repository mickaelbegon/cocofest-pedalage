# RHO-PACE short dynamic smoke: resolved 2026-09-11

The PACE adapter validates **2/2 dynamic RHO cycles** at +0.10 Nm constant
resistance when launched with the same effective IPOPT options as the
certified baseline. The uniform-start PACE configuration is a control for
objective integration; its default five-cycle update interval is not reached
in this two-cycle test. This does not demonstrate an endurance advantage.

## What caused the previous failure

The unsuccessful `/tmp/pace-live-retest-20260910/result.json` did not use the
baseline options. In particular its `ipopt_hsl_library` was null, while the
baseline explicitly selected the working LP64-compatible HSL library:

`/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so`

The failed optimization took only 0.45 ms. A separate minimal CasADi IPOPT
problem with `linear_solver=ma57` and no `hsllib` reproduced
`return_status=Invalid_Option`, `success=False`, and zero objective or
constraint evaluations. Its `iter_count` was an uninitialized-looking large
integer; that value is not meaningful optimization work. The prior smoke
also had different iteration limits, tolerances and thread count.

No PACE mathematical or Bioptim objective-injection defect was found in this
comparison. The library loading failure occurred before optimization.

A separate launcher defect explains the earlier apparently missing journals:
the benchmark changes its working directory to the cycling example directory.
Relative PACE journal paths were resolved only afterwards, creating the files
under `examples/fes_multibody/cycling/resistance-fho-pilots-20260910/...`.
Those original journals were found there. The launcher now resolves its
configuration and journal paths before calling the benchmark, and restores
the caller's working directory on return. The controller also stores an
absolute journal path. A regression test reproduces the directory change and
verifies that the journal appears at the caller's requested location.

## Comparable result

Reference baseline:
`resistance-fho-pilots-20260910/sequential-examples-0p10/rho-baseline-2/`

PACE outputs:
`/tmp/pace-exact-baseline-20260911/{result.json,pace.jsonl,concatenated-solution.npz}`

PACE console log: `/tmp/pace-exact-baseline-20260911.log`.

The two `configurations.ipopt` JSON objects have **no differences**. Both
results report `success=true`, `covered_cycles=2` and
`physically_validated_cycles=2`. Both objective sums are
`7.688086173805749`, and their common unweighted executed fatigue objective
is `7.446998936186781`. Every exported state and control array is bitwise
identical (maximum absolute difference 0).

The PACE journal independently records the initial `applied` event with
`ocp_cost_connected=true`, receipt `bioptim.update_objectives`, both completed
windows certified with solver status 0, and the normal two-cycle stop. Thus
this result includes objective attachment evidence, not only a benchmark
result that could have come from an unmodified baseline launcher.

## Actual adaptation in a second bounded smoke

A second two-cycle run used the same solver options with the explicitly
test-only policy override `update_every_cycles=1`, `max_cycles=2`. Its config
is `/tmp/pace-update-every-cycle-smoke-20260911.json`; results and journal are
under `/tmp/pace-adaptation-smoke-20260911/`. It also validates **2/2 cycles**,
with both solver statuses 0. Before the second solve the real applied
weights change from `[1,1,1,1]` to, in model order Delt_ant, Delt_post, Biceps,
Triceps:

`[1.0015407640941687, 1.000580986376519, 0.9993448919402975, 0.9985359994283924]`

The resulting objective sum is `7.692303485194472`. This proves a causal
weight update reaches the real cycling OCP and permits a subsequent
certified solve. It does not validate a different production cadence or
establish any endurance gain; the default slow cadence remains five cycles.

## Reproduce

Run from the repository root. Use a new journal/output directory for each
run; existing journals are intentionally refused.

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cocofest-mpl OPENBLAS_NUM_THREADS=1 \
  /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/run_rho_pace_benchmark.py \
  --pace-config docs/cycling_solver_benchmark/rho_pace_uniform_start.json \
  --pace-journal /tmp/pace-exact-baseline-20260911/pace.jsonl -- \
  --solvers ipopt --objective fatigue --objective-shape quadratic \
  --formulation dynamic --mechanical-formulation reduced \
  --signed-crank-torque 0.1 --cycles-per-window 1 --n-windows 2 \
  --ipopt-profile scientific_radau5 --compact-rho-output --ipopt-use-sx \
  --ipopt-linear-solver ma57 \
  --ipopt-hsl-library /home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so \
  --warmup-ipopt-linear-solver ma57 --ipopt-max-iter 2000 --n-threads 4 \
  --nlp-tolerance 1e-8 --primal-feasibility-threshold 1e-5 \
  --ipopt-disable-historical-initial-guess \
  --common-initial-solution resistance-fho-pilots-20260910/current-ma57-seed-diagnostic/common-reduced-0p10.npz \
  --common-initial-solution-recenter-first-node-bounds \
  --adopt-common-initial-solution-warmup-cycles \
  --reduced-cycling-profile resistance-fho-pilots-20260910/seed-0p10/reduced-cycling-fourier12.npz \
  --standard-warmup-seed .github/benchmark-seeds/legacy-resistive-0p22-warmup.npz \
  --legacy-standard-warmup-seed-signed-torque 0.22 \
  --standard-warmup-seed-continuation \
  --allow-partial-receding-horizon-solution-output \
  --receding-horizon-solution-output /tmp/pace-exact-baseline-20260911/concatenated-solution.npz \
  --output-json /tmp/pace-exact-baseline-20260911/result.json
```

The unit integration test in `tests/test_rho_pace.py` now solves the original
baseline objective before attachment, verifies its equality with unit PACE
weights after attachment, and then verifies that doubling the weights
doubles the real solved cost. This covers global weighting and Bioptim time
integration in addition to the symbolic residual formula.
