# Normalized pulse-width increment regularization

The optional `--pulse-width-slew-weight` (default `0`) adds a quadratic cost
to the existing `delta_pw_v1` lifting. It requires an explicit positive
`--pulse-width-max-step-us` hard bound, reduced mechanics, and one-cycle
windows. `--pulse-width-slew-reference-us` defaults to `100` and must be
positive. This reference is independent of both the hard bound and the
numerical variable scaling.

For M muscles, N control intervals, and a fixed interval duration dt, the
Bioptim Lagrange residual is the vector `delta_pw / reference_s`. Its weight
is `lambda / (M * (N - 1) * dt)`, so integration gives

    J_delta = lambda / (M * (N - 1)) * sum_m sum_k (delta_pw[m,k] / reference_s)^2.

The sum includes N auxiliary increments. The first N-1 equal the differences
between physical controls inside the window. The last reaches a free
terminal carrier and has no physical successor: a positive weight drives
it to zero. At weight zero its existing, uniformly bounded freedom remains.
No periodic closure is imposed. The executed preceding-cycle-to-first-control
seam retains its hard bound and is not penalized by this cost.

The custom residual is stage-local, uses physical unscaled controls, and
exports to ACADOS `NONLINEAR_LS` with `multi_thread=False`. Bioptim 3.5's
`MINIMIZE_CONTROL, derivative=True` is unsuitable here: for constant controls
it evaluates the same control at both ends of an interval and returns zero.

The configuration fields are `pulse_width_slew_weight` and
`pulse_width_slew_reference_us`. They propagate through the GUI and launchers
and are included in cache/seed signatures. A certified seed with a different
regularization must be regenerated. The generic assisted warmup, whose lift
is disabled, also disables this cost; the target solve uses the requested
weight.

`pulse_width_slew_regularization_audit` reports the average normalized
intra-window squared physical differences and their weighted cost. It
explicitly excludes the final auxiliary increment and the executed seam;
therefore it is not the entire solver cost away from a converged optimum.
The original hard-bound audit still includes executed seams.

An initial sweep can use weights `0, .001, .01, .1, 1, 10, 100`, with the
reference fixed at 100 microseconds. Compare physical fatigue/reserves and
the original objective separately from the regularization, plus RMS/max PW
changes, constraint violations, KKT residuals, success rate, iteration count,
and hot-window median/p95 solve time. Report compilation and first-solve
time separately. Reconstruct the same canonical cost when comparing
IPOPT and ACADOS; ACADOS uses a conventional one-half factor in least-squares
costs.

Targeted checks:

    MPLCONFIGDIR=/tmp/mpl-slew-reg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python -m pytest -q \
      tests/test_slew_regularization.py tests/test_pulse_width_slew.py \
      tests/test_slew_seed_transfer.py tests/test_simulation_core.py tests/test_simulation_gui.py
