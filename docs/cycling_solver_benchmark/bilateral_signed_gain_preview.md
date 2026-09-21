# Bilateral preview: signed gain and invalid fibre geometry

The compact predictor previously required `g = fl * fv + fp > 0`. That is a
stronger assumption than the actual OCP. Both `Ding2003.f_dot_fun` and the
reduced OCP use `dF/dt = g * (recruitment * activation - F / relaxation)`.
The signed crank moment is formed separately as `moment_coefficient * F`.
Changing the sign of a moment arm therefore cannot fix this gain issue.

The predictor and the independent full Ding interval replay now preserve
finite signed gains. The scalar and batched compact maps intersect the PW
recruitment bound with `F >= 0` at each exponential integration substep.
No gain or force is projected. A negative gain can give a negative recruitment
slope; the total-moment allocator uses sorted endpoint bounds and retains that
slope when inverting recruitment. Independent full Ding replay remains needed:
slow-state freezing is an approximation, and these bounds are not a continuous
full-model certificate.

This makes the preview faithful to the existing OCP algebra; it does **not**
make a negative muscle-force gain physiologically valid. With negative gain,
zero recruitment amplifies existing force and recruitment can reduce it.

## Evidence on the existing bilateral profile

Profile `/tmp/cocofest-bilateral-smoke-profile.npz`, 510 evenly spaced crank
angles, omega = -2 pi rad/s:

- active force-length minimum: 1.286e-5;
- force-velocity minimum: 0.9191;
- passive gain negative in all 4,080 muscle-angle samples, minimum -0.0186565;
- total gain negative in 1,910 samples; posterior deltoid gain is negative
  throughout the cycle (approximately -0.01864 to -0.01850).

The deeper problem is fibre geometry. Normalized fibre length ranges in the
profile are approximately anterior deltoid [0.417, 0.702], posterior deltoid
[-0.498, -0.275], biceps [-0.053, 0.580], triceps [0.297, 0.483], with left/right
symmetry. These are not Fourier overshoots: direct biorbd evaluation of
`Modified_Wu_Shoulder_Model_Cycling_Bilateral.bioMod` at the profile's theta=0
configuration returns right posterior-deltoid fibre length -0.03954 m and right
biceps -0.00862 m (left posterior deltoid -0.06081 m). The reduced profile samples
`muscle.length(...) / optimalLength()` directly.

Muscle path geometry and tendon slack lengths must therefore be reconciled
before a new bilateral physiological benchmark. Clipping the passive term,
fibre length, or gain would change the current model without correcting its
geometry. Existing bilateral solver timings should be identified as timings of
the current mathematical model, not evidence of physiological validity.

## Repeat of the five-cycle preview

Command (Python from the `cocofest-rho32` environment):

```bash
python scripts/benchmark_cycle_preview_controller.py \
  --source two-timescale-bilateral-preview-20260919/source-cycle.npz \
  --baseline-json bilateral-ding-local-benchmark-100-20260919/ipopt-ding-local.json \
  --reduced-profile /tmp/cocofest-bilateral-smoke-profile.npz \
  --model-path examples/msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling_Bilateral.bioMod \
  --max-blocks 1 --output /tmp/cocofest-signed-bilateral-preview
```

The predictor now constructs and completes one interval. At zero-based phase 1,
the requested total moment is -0.0313914 Nm, while the compact reachable interval
is [-0.337482, -0.0672249] Nm: a 0.0358335 Nm upper-bound deficit. This is failure
of this allocation policy, not proof of global infeasibility. No certified
five-cycle result or speedup is reported. The report now explicitly leaves
speedup and time-saved fields null for every uncertified/incomplete block.

Tests compare signed/zero-gain compact propagation with full Ding integration,
check the state-dependent force bound, scalar/batch parity, and suppress speed
claims for failed prefixes.
