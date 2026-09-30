# Reduced direct versus inverse dynamics: code audit and first experiment

Date: 2026-09-25. This report concerns the current working tree, including existing uncommitted developments. No production dynamics were changed and no full RHO speedup is claimed.

## Finding

**The reduced isokinetic implementation already uses inverse dynamics.** It prescribes crank speed, eliminates the required load torque analytically and integrates produced work. The reduced isoresistance/dynamic implementation uses forward dynamics, but its online mechanical solve is already a scalar division by effective inertia. It does not solve a full constrained multibody linear system online.

For dynamic cycling, an equivalent implicit inverse-balance residual is feasible and inexpensive to prototype. A compiled mechanical-kernel experiment shows a 13.8% reduction in residual/Jacobian/Hessian evaluation time. This is a local kernel gain; its effect on a complete RHO can be much smaller or negative because conditioning, sparsity and solver iterations also change. For isokinetic cycling, the more relevant new experiment would eliminate the remaining prescribed angle/speed variables and precompute stage-dependent mechanical coefficients.

## What the code does today

| Component | Implementation and role |
|---|---|
| `cocofest/dynamics/reduced_cycling.py`, `ReducedCyclingDynamics.build` | Offline projection of mass, gravity, quadratic-velocity and muscle-effectiveness terms onto the contact tangent; Fourier fitting. |
| Same file, `casadi_acceleration` | Dynamic forward model: one scalar numerator divided by effective inertia. |
| `cocofest/models/reduced_cycling_model.py`, `required_load_torque` | Isokinetic inverse model: analytically solves the scalar balance for load. |
| Same file, `dynamics` | Dynamic branch evolves angle and speed; isokinetic branch fixes speed and evolves work. The work expression already cancels division/remultiplication by external effectiveness. |
| `cocofest/optimization/isokinetic_cycling.py` | Independent inverse-load reconstruction, sign conventions, work targets and certification. |
| `examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py`, `patched_export_model` | ACADOS currently receives an implicit expression constructed as `xdot - f_expl`; it is an explicit ODE embedded in IRK, not the inverse balance below. |
| `cocofest/optimization/acados_ding_local_reduction.py`, `transform` | Rebuilds `f_impl_expr = xdot - f_expl` after Ding reduction. Any future mechanical implicit residual must survive this transformation. |

The earlier reduced-only implementation (`1d4f50ac`, examined for provenance) contained only the direct branch. The isokinetic feature (`3932bc0f`) already introduced the current analytical inverse-load elimination. Thus another switch called “inverse dynamics” would add nothing to that isokinetic branch. These commit identifiers document this code audit; they are not scientific citations or replacements for the article describing the original model.

## Equations and equivalence

Let the holonomic contact manifold be parameterized by crank angle:

\[
q=Q(\theta),\quad T=Q_{\theta},\quad K=Q_{\theta\theta},\quad
\dot q=T\omega,\quad \ddot q=T\alpha+K\omega^2.
\]

The full mechanical balance is

\[
M(q)\ddot q+h(q,\dot q)=\tau_m(q,F)+e(q)\tau_L+J_c(q)^T\lambda.
\]

Since admissible motion satisfies \(J_cT=0\), tangent projection eliminates contact reactions. Define

\[
I=T^TMT>0,\quad g=T^Th(q,0),\quad
c=T^T\{h(q,T)-h(q,0)+MK\},
\]
\[
b_m=-\frac{\partial\ell_m}{\partial q}T,\quad b_e=T^Te,
\quad N(\theta,\omega,F,\tau_L)=\sum_m b_mF_m+b_e\tau_L-g-c\omega^2.
\]

The implementation samples these functions, then evaluates their Fourier approximations. All equivalence statements here are between formulations of the **same fitted reduced profile**; this experiment does not revalidate Fourier approximation against the full biorbd mechanics.

For dynamic/isoresistance cycling, the present model is

\[
\dot\theta=\omega,\qquad \dot\omega=N/I,\qquad \dot y=f_{Ding}(t,y,PW;\theta,\omega).
\]

The external load \(\tau_L\) is prescribed. Phase advance, initial state, terminal conditions and speed inequalities remain part of the OCP. An equivalent stage residual is

\[
r_{direct}=\alpha-N/I=0,
\qquad
r_{inverse}=\frac{I\alpha-N}{I_{ref}}=0,
\quad I_{ref}>0\text{ fixed}.
\]

Here \(\alpha\) is the derivative of the collocation polynomial, or the existing IRK stage slope. It need not become a new control, state or algebraic variable. At each stage

\[
r_{inverse}=\frac{I(\theta)}{I_{ref}}r_{direct}.
\]

Consequently the collocation feasible set is unchanged when inertia is positive. Away from feasibility, gradients and Hessians change because this is **state-dependent residual scaling**. Equal absolute tolerances on these two residuals do not imply equal physical acceleration error. Feasibility comparisons must reconstruct \(\alpha-N/I\), or the same state-scaled collocation defect, for both variants. At a feasible point the constraint Jacobians differ by row scaling; corresponding optimal multipliers also rescale. Existing warm-start duals cannot be reused unchanged after switching formulations.

A literal acceleration-control inverse transcription would instead add \(\alpha\) as a decision variable and enforce the balance separately. It offers no automatic state reduction here, may enlarge the KKT system, and a single held acceleration per shooting interval is not equivalent to stage-wise Radau acceleration. Eliminating one muscle force from the balance is likewise not a free optimization: its Ding force differential equation must still be enforced, potentially producing a more difficult DAE.

For isokinetic cycling, the existing implementation imposes

\[
\omega=\omega_0,\quad \alpha=0,\quad
\theta(t)=\theta_0+\omega_0t,\quad
\tau_L=\frac{g+c\omega_0^2-\sum_m b_mF_m}{b_e}.
\]

Zero crank acceleration does **not** mean zero generalized acceleration: \(\ddot q=K\omega_0^2\). Thus the inertial/velocity contribution in \(c\omega_0^2\) must remain. The model enforces bounded instantaneous reconstructed load and a prescribed net work:

\[
\dot E=-\tau_Lb_e\omega_0
=\left(\sum_m b_mF_m-g-c\omega_0^2\right)\omega_0,
\qquad E(T)-E(0)=2\pi n\bar\tau.
\]

The code uses negative speed; positive load is resistive when external effectiveness is positive. A global moment must use its angular-velocity Jacobian projection \(e\), as already corrected in schema 3. It must not be reinterpreted as a force or as a moment on only one relative joint. Isokinetic work control and prescribed constant isoresistance are different problems: replacing one by the other is not an equivalent acceleration experiment.

## Isolated measured comparison

Executable: `scripts/benchmark_reduced_inverse_kernel.py`. Output: `local-results/reduced-inverse-kernel-20260925/alternating/result.json`.

The experiment uses the existing corrected single-arm Fourier-12 profile at `local-results/rho-dynamics-global-moment-0p2/seed-0p2/reduced-cycling-fourier12.npz`. CasADi 3.7.2 generates SX residual, Jacobian and Hessian expressions, compiled with GCC `-O3`. Evaluation is serial on CPU affinity 12–15, numeric libraries mono-thread, 4096 identical seeded random inputs, 21 batches alternating evaluation order. Input ranges are angle one full turn, speed −9 to −3 rad/s, acceleration −30 to 30 rad/s², load −3 to 3 Nm, and each muscle force 0–400 N. These deliberately varied states are algebraic test states, not certified feasible physiological trajectories.

| Metric | Present direct residual | Implicit inverse-balance residual |
|---|---:|---:|
| Median evaluation, residual + Jacobian + Hessian | 1.347 µs | 1.161 µs |
| P90 batch-average evaluation | 1.375 µs | 1.184 µs |
| CasADi expression instructions | 4543 | 3576 |
| Jacobian structural nonzeros | 8 | 8 |
| Hessian structural nonzeros | 14 | 16 |
| C generation + compilation (single observation) | 0.449 s | 0.299 s |

The extra Hessian entries arise from the angle–acceleration product \(I(\theta)\alpha\). All eight kernel inputs are differentiated, including the load; a full isoresistance NLP holds that load fixed, so these counts are **not** full-NLP nonzero counts. The microbenchmark also includes a CasADi mapped-call wrapper; its microseconds should not be equated with solver-internal timings.

Effective inertia ranges from 0.004415 to 0.020055 kg·m² (median reference 0.010413). The residual identity error is at most \(2.27\times10^{-13}\), and the implicit residual at direct-model roots is at most \(1.71\times10^{-13}\).

This establishes algebraic correctness and a local evaluation advantage, not reduced iteration count. Even if this kernel occupied 20% of total RHO time, a 13.8% reduction confined to it would save only about 2.8% overall at unchanged iterations and linear-algebra cost. The actual fraction has not been measured. The result is more plausible as a conditioning experiment than as a large arithmetic acceleration.

Reproduction:

```bash
taskset -c 12-15 env \
  PYTHONPATH=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/benchmark_reduced_inverse_kernel.py \
  --profile local-results/rho-dynamics-global-moment-0p2/seed-0p2/reduced-cycling-fourier12.npz \
  --output local-results/reduced-inverse-kernel-20260925/reproduction
```

## Integration consequences and next experiment

For IPOPT/MA57, change only the angular-acceleration collocation defect after proper conversion between physical and scaled variables. Retain the explicit right-hand side for replay and initialization. Do not alter PW bounds, fatigue weights, Radau tableau, frequency, speed inequalities or seeds. No extra state is needed. The code currently constructs `DynamicsEvaluation.defects`; whether the chosen Bioptim collocation path consumes custom defects must be verified explicitly before interpreting any A/B result.

For ACADOS IRK, replace only the mechanical row of `f_impl_expr` by the scaled inverse balance, preserve other rows and the same explicit RHS. Apply the replacement after Ding-local reduction, or teach that transformation to preserve it; otherwise `transform` silently overwrites it. ERK uses the explicit RHS and cannot test this benefit. IRK must still solve an implicit stage system; the change does not eliminate that solve. Ding periodic/SX and local elimination of Ding states are mathematically compatible with either mechanical residual.

The smallest informative full test is an IPOPT/MA57 dynamic RHO A/B over 10 matched windows at 30 Hz, current K7 settings, direct versus implicit residual, with the same initial primal and cold-start dual policy. Record first-window cold time, hot median/P90, iterations, evaluator and KKT timing, physical defect, objective and DOP853 milestone replay. Promote only after feasibility and physical replay match and net wall time improves. A third fixed-scaled direct residual can distinguish simple magnitude effects from state-dependent inverse scaling if the initial pair differs materially.

For the **isokinetic** branch, prioritize an independent experiment replacing the two mechanically prescribed states by \(\theta_0+\omega_0t\) and \(\omega_0\), retaining the work state and the same inverse-load constraints. Fourier coefficients and geometry can then be supplied at known Radau stages. This can remove mechanical decision variables and angle derivatives through muscle geometry. It requires stage-aware time/phase handling for Radau, pulse timing, RHO phase advance and each independent arm; interpolation must preserve the intended accuracy. Such a change has not been implemented or timed in this audit.
