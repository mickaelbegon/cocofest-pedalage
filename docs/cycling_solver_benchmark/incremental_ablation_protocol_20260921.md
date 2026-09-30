# Incremental solver ablation protocol

This is an executable-command proposal, not a completed experiment. The companion
`scripts/plan_incremental_solver_ablation.py` only prints a manifest; it does not
launch solvers or reserve CPU cores. The campaign coordinator must supply a verified
CPU allocation before executing any command.

## Causal ordering

Use 50 stimulations per revolution throughout the initial dynamic/isoresistive
ladder. Current production code explicitly rejects Radau degrees 3 and 4 at 30 Hz
because of earlier DOP853 discrepancies. Changing frequency between rows would
confound the Radau comparison. A later 30 Hz replication starts at degree 5.

| Step | Change relative to its parent | Solver |
|---|---|---|
| K0 | Article-based reproduction: historical Ding, full mechanics, MX, Radau degree 3 | IPOPT/MA57 |
| K1 | Radau degree 5 | IPOPT/MA57 |
| K2 | Constant crank torque representation replacing external force input | IPOPT/MA57 |
| K3 | Reduced mechanics, same Ding and integration | IPOPT/MA57 |
| K4 | Periodic-node Ding runtime representation | IPOPT/MA57 |
| K5 | SX replacing MX | IPOPT/MA57 |
| K6 | Compile only `nlp_hess_l`, exact Hessian unchanged | IPOPT/MA57 |
| K7 | Compact RHO output | IPOPT/MA57 |
| K8 | Hard adjacent PW bound, common lifting formulation | IPOPT/MA57 |
| K9 | Add normalized quadratic PW-increment cost | IPOPT/MA57 |
| A0 | Full mechanics, periodic Ding/SX, native IRK | ACADOS |
| A1 | Reduced mechanics, same native IRK | ACADOS |
| A2 | Local Ding reduction, same native IRK | ACADOS |
| A3 | Compact RHO output | ACADOS |
| A4 | Hard adjacent PW bound, common lifting formulation | ACADOS |
| A5 | Same normalized PW-increment cost | ACADOS |

K0 reproduces the article's numerical approach in current code and is not claimed
to reproduce its published timing or every original parameter. Explicit options
are essential: the current `historical` CLI profile itself defaults to SX, whereas
the requested MX-to-SX ablation requires explicit MX in K0--K4.

Reduced mechanics requires `torque_application=constant`, hence the extra K2 bridge.
The K2 check should compare generalized crank force and mechanical RHS, not simply
assume equivalent signs. Periodic-node Ding is a change of calcium representation;
it is not a periodic terminal constraint on fatigue.

The local IPOPT Ding adapter is a separate branch from K5, run through
`scripts/benchmark_ding_radau5_local_ab.py baseline|local -- ...`. It reconstructs
states before returning solutions. The distinct symbolic-stage adapter rejects
C compilation/function transformation, so local reduction followed by Hessian
compilation cannot be presented as a supported cumulative production path without
an additional integration check. Its frozen-input replay is useful for isolating
solver cost, but must be identified separately from a genuine free-running RHO.

The main script calls degree 5 “Radau-5”: five Radau IIA stages, formal order nine
for a smooth ODE. This is not the conventional three-stage, fifth-order method
often called RADAU5. ACADOS stage support must be checked against the installed
native build. Start with its demonstrated IRK settings and explicitly record
family, number of stages, substeps, Newton iterations and tolerance. A cross-solver
comparison using different maps is a matched-accuracy comparison, not an identical
discretization experiment. Add an IPOPT IRK bridge with the same valid tableau to
isolate transcription differences.

## Runtime protocol

1. Record Git/worktree provenance, Python/Bioptim/CasADi/ACADOS/HSL versions, machine,
   process affinity and the background campaign affinities. Allocate physical cores
   avoiding SMT siblings of occupied cores. CPU pinning alone is not an exclusive
   reservation if another process is allowed on those cores.
2. Every variant uses its own fresh working directory, generated solver directory,
   compilation-cache namespace and output names. Do not delete existing caches.
   Force BLAS/OpenMP threads to one and also set `CMAKE_BUILD_PARALLEL_LEVEL=1`;
   the environment helper otherwise defaults the latter to all available CPUs.
3. Use a 2-window feasibility smoke before a 100-window run. Repeat timing cases
   at least three times with reversed/interleaved ordering. Give failed cases the
   same iteration/timeout budgets, retaining attempted and certified windows.
4. For each pair, freeze physical parameters, stimulation grid, objective, scaling,
   state/control bounds, initial state, horizon, torque sign and RHO transfer policy.
   Match initial physical guesses through an explicit conversion when dimensions
   differ; do not force an incompatible seed past its scientific fingerprint.
5. Report setup from an empty code cache and then a separate cache-hit startup.
   An empty application cache does not mean the OS filesystem cache is cold.
6. Reintegrate recorded controls with DOP853 after timing is finalized. Preserve
   ZOH discontinuity boundaries and the same initial full physical state. Use
   one-cycle reset and continuous multi-cycle replay as distinct metrics.

## Required timing outputs

Report raw per-window samples plus first solve, median, P90, P95, maximum and
certified count. Define hot samples before running, e.g. windows 3--100; retain all
samples too. Report solver native time, Python solve-call wall time and full-cycle
wall latency separately. Count retries and compilation during a cycle in latency.

Cold components: imports, mechanical profile build/load, OCP symbolic build,
Hessian construction/transformation, C export, native compilation/link, capsule
creation/load, warmup solve/projection and first target solve. Existing JSON
`execution_timing.pre_solve_setup_wall_time_s` is aggregate setup;
`reduced_profile_build_time_s`, orchestration profile and compiled-reuse audit add
detail. Do not relabel aggregate setup as compilation: granular native build hooks
are needed where those durations are absent. Report unavailable components as null.

Compute compile amortization only for positive hot savings:
`ceil(extra_cold_seconds / hot_seconds_saved_per_window)`. Include setup overhead
of both variants and report the horizon beyond which compilation repays its cost.

## Accuracy and objective outputs

For every certified window: terminal and maximal phase/velocity discrepancies,
force/capacity errors by muscle, full holonomic position/velocity defects after
reconstruction, energy/work discrepancy, original physical bound violation,
DOP853 evaluation count and tolerance. Isokinetic phase/velocity are prescribed;
their zero error is not an integrator-quality score. Compare force, fatigue,
inferred load and work there.

For PW variants, separately report canonical fatigue objective, normalized slew
penalty and total objective. Recompute the canonical cost from DOP853 states with
one common quadrature: native IPOPT/ACADOS least-squares conventions differ by a
one-half factor. Include RMS/max PW increments, including the executed RHO seam,
and state that the current soft regularizer excludes the seam.

ACADOS supports the stage-local PW lifting. `direct_constraints` is restricted to
IPOPT/MadNLP and should be an IPOPT-only efficiency side branch. Choose weights
0, 0.001, 0.01, 0.1, 1 with a fixed 100 us normalization; each weight is compared
to the hard-bound-only parent rather than pretending that a changed weight is an
additional feature. Any regularization changes the optimization problem.

The isokinetic series is a separate reduced-mechanics replication of K4 onward and
A1 onward. It prescribes work per turn via equivalent mean torque. It does not
replace the constant external torque of the dynamic/isoresistive ladder.
