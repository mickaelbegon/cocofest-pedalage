# Terminal positive-work capacity under maximum pulse widths

The experimental objective maximizes the sum of positive muscle work that a
prescribed one-cycle stimulation policy produces from the candidate terminal
RHO state. It uses the complete five Ding states `(Cn, F, A, Tau1, Km)` of
each muscle, including residual calcium and force. It is intended for a fixed
isokinetic trajectory; geometry and velocity are frozen numerical data.

For prescribed velocity `omega`, moment coefficient `b_i(t)`, and predicted
force `F_i(t)`, the capacity proxy is

`W_plus(x_end) = sum_i integral max(omega*b_i(t), 0) * F_i(t) dt`.

The reduced-model sign is `E_prod_dot = omega * b_i * F_i`. At negative
angular velocity, a negative moment is propulsive. Positive *signed torque*
alone would therefore select the wrong phases. The positive-part operation is
performed on numerical geometry when packing the profile. The objective is
differentiable in every terminal state on the valid Ding domain. A suitable
Mayer contribution for minimization is `-lambda * W_plus / W_reference`, with
a fixed positive work scale. This encourages greater remaining work capacity;
it does not assume a muscle's force scale alone represents its fatigue.

Two explicit future policies are available:

- `all_intervals_pw_max` (default): PW is maximal on all stimulation intervals.
- `selected_propulsive_intervals`: PW is maximal only where midpoint
  `omega*b_i > 0`; PW equals the muscle's `pd0` elsewhere, giving zero
  recruitment in the existing Ding PW law. A custom fixed binary mask is
  also supported. This is an excitation mask, separate from the quadrature
  coefficients that discard braking work at each integration stage.

These policies can yield different work because force persists across phase
boundaries. Restricting excitation to currently propulsive phases is not a
proof of maximal possible work: stimulation before such a phase can create
useful force later. A mask is held fixed as terminal states vary. Neither
policy is optimized internally, and neither provides a global upper bound on
work or a certificate of endurance. Muscle independence follows the existing
prescribed-kinematics model; external load sharing and simultaneous mechanical
feasibility remain the actual RHO's responsibility. The summed positive work
does not penalize braking contributions or guarantee enough work at every
crank angle.

## Numerical implementation

`max_pw_work_capacity.py` reuses the muscle-horizon Ding equations. Calcium is
propagated analytically within each stimulation interval. Force and all three
fatigue states are propagated with RK4, together with work using matching RK
stages. The default is 16 substeps per stimulation interval. This is separate
from the enclosing RHO's scientific Radau-5 transcription. It is not Radau-3.
Mechanical gain and moment coefficients can be sampled from interval-local
callables. When only interval endpoint coefficients are supplied, geometry is
held constant within each interval and that approximation must be acknowledged.

The function is built once for fixed muscles, interval count, and substep
count. Terminal states and packed profile are its two numerical inputs. No
future decision variables, FHO, QP, or nested optimization are introduced.
Parameters are packed in C order; `layout.slices()` exposes every block,
including the exact binary `stimulation_mask` for provenance/audit. The
existing stimulation-history/calcium amplitude contract must be preserved
across the terminal boundary; this code does not infer missing history.

Outputs are total work in joules, per-muscle work, propagated five-state
endpoints, and detailed Ding domain margins at all RK stages and endpoints.
The margins must be checked after a trial; negative force or depleted capacity
is never clipped to make a trial look valid. These detailed audit outputs are
not intended to become tens of thousands of extra NLP constraints. A future
RHO integration must reject proxy results outside the valid domain and report
that rejection rather than claiming physiological failure.

## RHO coupling without embedding the rollout

Embedding the full 30-phase, RK4×16 capacity rollout directly in IPOPT made a
short two-arm RHO smoke test roughly 2.2 times slower. The active RHO binding
therefore evaluates the exact proxy only after a certified cycle, at the
accepted state `x_ref`. It stores the numerical gradient `g = dW_plus/dx` and
uses the following fixed-size terminal term in the next RHO:

`-lambda / W_reference * g^T (x_end - x_ref)`.

The omitted constant `W_plus(x_ref)` cannot change the minimizer. The
objective has only two extra 5-state vectors per muscle (reference and
gradient), instead of the rollout graph. The exact rollout is re-evaluated
after every certified cycle to refresh that local model and audit its Ding
domain. Thus the term is a local opportunity direction, not a global
prediction of future endurance; large departures from `x_ref` need a
trust-region safeguard or a fresh boundary update before making endurance
claims.

For an ablation, `gradient_filter="slow_fatigue_states"` retains only
sensitivities to Ding `A`, `Tau1`, and `Km` for each muscle. It sets the `Cn`
and `F` components of the gradient passed to the RHO to zero. The default
`gradient_filter="full"` preserves the complete derivative and prior
behavior. Both modes evaluate the same exact `W_plus` at each certified
boundary. The filter does not alter dynamics or make `W_plus` a feasibility
certificate.

The boundary audit reports exact total and per-muscle work, raw gradient
norms, and each muscle's five derivatives grouped into fast (`Cn`, `F`) and
slow (`A`, `Tau1`, `Km`) states. Raw norms mix the states' physical units;
they are diagnostics, not dimensionless muscle rankings. From the second
accepted checkpoint, the audit also reports the previous affine model's
predicted work, its error against the new exact rollout, the state step, and
the fast/slow directional contribution for each muscle. This measures the
local model's fidelity along the realized RHO trajectory. Invalid Ding
domains do not yield a gradient or Taylor diagnostic and continue to trigger
the pre-transfer hold. The final run summary includes the selected gradient
filter and last audit; the fixed profile SHA-256, weight, work reference, and
muscle ordering remain available for reproduction.

For the independent-arm driver, set
`experimental_max_pw_work_gradient_filter` to `full` or
`slow_fatigue_states` in the JSON payload, alongside the positive
`experimental_max_pw_work_weight`. The equivalent command-line switch is
`--experimental-max-pw-work-gradient-filter`. A paired test should use the
same load, muscle parameters, solver, and initial state in both modes.

## Usage

```python
layout = MaxPwWorkCapacityLayout(len(muscles), len(intervals))
profile = pack_max_pw_work_profile(
    intervals, layout, angular_velocity_rad_s=-2*np.pi,
    moment_coefficient_functions=moment_coefficients,
    stimulation_policy="all_intervals_pw_max",
)
function = build_max_pw_work_capacity_function(muscles=muscles, layout=layout)
work, work_by_muscle, next_state, margins = function(terminal_state, profile)
valid_domain = np.all(np.asarray(margins).ravel()
                      >= max_pw_work_domain_lower_bounds(layout))
```

## Validation scope

Tests independently integrate all five Ding equations and work with SciPy
DOP853 at 30 and 50 Hz for both policies. They require relative error below
0.003% for terminal states and integrated work on this representative fixture.
They also test two-muscle SX/MX gradients against finite differences, work
sign, zero braking credit, mask behavior, numerical profile reuse, depletion,
and a four-muscle function/gradient microbenchmark. The domain gate and this
fixture are numerical validation, not population-level scientific validation.

At 30 Hz, RK4 with only four substeps had force error about 0.13%; sixteen
substeps reduced it to below 0.002% in the fixture. The test prints actual
errors and timing. End-to-end IPOPT solve overhead, real-checkpoint robustness,
and endurance benefit still require paired RHO benchmarks.
