# Unilateral prescribed isokinetic Ding reduction, frozen 100-window A/B

The benchmark combines the K5-I1 prescribed-kinematics RHO with exact local
Radau-5 reconstruction of `Cn`, `Tau1`, and `Km`. It uses one arm, four Ding
muscles, the unit fatigue objective, 30 stimulations per cycle, reduced
mechanics, SX IPOPT/MA57, and one numeric thread. The equivalent energy target
is 0.2 Nm at −2π rad/s with inferred load bounded to [−3, 3] Nm.

Run from the repository root:

```bash
bash scripts/run_prescribed_ding_local_ab.sh \
  local-results/prescribed-ding-local-ab-20261004/paired100 100
```

The baseline RHO records the original-space IPOPT initial vector, bounds, and
dual warm starts for each window. The local solver receives those exact same
inputs, then returns the recorded baseline solution to the outer RHO loop so
the next window has exactly the same state and warm start. The local candidate
solution is still independently solved and audited against the full NLP. Thus
this measures a matched solver call, **not** a freely propagated local RHO.

| Quantity | Full Ding | Local Ding |
|---|---:|---:|
| Successful NLP calls | 100/100 | 100/100 |
| NLP variables / constraints | 3,922 / 3,811 | 1,750 / 1,651 |
| Hot IPOPT median (cycle 1 excluded) | 0.3797 s | 0.07850 s |
| Hot IPOPT P90 | 0.4371 s | 0.08921 s |
| Hot IPOPT iterations, median | 43 | 36 |
| Hot complete RHO iteration, median | 0.4662 s | 0.1746 s |
| Maximum violation of full NLP | 9.09 × 10⁻⁷ | 7.39 × 10⁻⁷ |
| Maximum discrete equivalence certificate | — | 8.00 × 10⁻¹⁵ |

The hot IPOPT median improves by **4.84×**. All archived frozen solver inputs
match exactly. The complete-iteration number contains this experiment's archive
I/O and candidate reconstruction, so it is not a production-control latency.

The candidate solutions are feasible but not identical. Their maximum PW
difference is **13.45 µs** across all 100 windows and four muscles. Their
maximum absolute objective difference is **6.80 × 10⁻⁵**; the average signed
relative objective difference is −0.139%. The first window is a particularly
clear example of different local minimizers: full 0.0004964 versus local
0.0004284. This is compatible with exact discrete elimination in a nonconvex
problem; it does not establish that propagated fatigue or endurance is equal.

The next scientific gate is a freely propagated local RHO over at least 100
cycles, followed by dense physical replay and an A/B of two unilateral arms
running concurrently on separate cores. The compiled local evaluator is a
separate experimental option and needs its own matched timing run.

Machine-readable evidence: `local-results/prescribed-ding-local-ab-20261004/paired100/summary.json`,
with individual full-NLP audits in the sibling `baseline/symbolic-audit` and
`local/symbolic-audit` directories.
