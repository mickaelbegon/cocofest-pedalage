# Experimental odd-control PW interpolation

Enable `--pulse-width-odd-interpolation` in the periodic cycling runner with
IPOPT or MadNLP, reduced mechanics, one-cycle windows, 50 stimulations and a
positive `--pulse-width-max-step-us` bound. It is disabled by default.

For each muscle, indices are zero-based: `PW[2j+1] = (PW[2j]+PW[2j+2])/2`
for `j=0..23`. The final physical control `PW[49]` stays free; no cycle-wrap
equality is added. The physical pulse train still has 50 zero-order-held
stimulations. With four muscles, 96 scalar equalities are added and no
decision variable is eliminated, so a runtime gain must be measured.

The exact `delta_pw_v1` lift already guarantees `delta[k]=PW[k+1]-PW[k]`
for `k=0..48`. Equating `delta[2j]` and `delta[2j+1]` therefore imposes the
requested means. The regular NLP constraint uses the current and next
control symbols. ACADOS is rejected because its exporter does not support
this cross-stage constraint. No `derivative=True` option is used.

The standard IPOPT warmup remains a target-independent physical guess.
The accepted common cycle-1 seed must be regenerated and certified with the
matching interpolation flag: common seeds for the other constraint set are
rejected. Model signatures and source stamps include the new option/code.
RHO transfer is not projected onto the equality; each target solve enforces
it. Exported summaries include an audit of physical interpolation residuals
in microseconds, covering all exported attempts (not a success certificate).

Validation includes a solved 50-interval IPOPT Radau-5 toy problem that
simultaneously checks the interpolation equality, the exact increment lift,
the 100 µs hard bound and the absence of periodic closure. This checks the
constraint implementation, not performance or physiological optimality.
