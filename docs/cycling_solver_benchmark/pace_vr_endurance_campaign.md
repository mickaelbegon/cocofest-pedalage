# Bilateral PACE-VR endurance campaign

`scripts/run_pace_vr_endurance_campaign.py` prepares a matched comparison of
unit-weight RHO, the existing capacity-feedback RHO-PACE, and an asynchronous
PACE-VR candidate. It uses the existing asymmetric Ding models at 1.92 Nm total
(0.96 Nm per arm), 30 Hz, one-cycle RHO, Radau degree 5, and IPOPT/MA57. The
input PACE-VR JSON must explicitly enable the experimental asynchronous
supervisor; the harness never silently substitutes a baseline condition.

Example (choose eight actually available, distinct CPU IDs):

```bash
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
RUN=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage/pace-vr-r192-campaign
$PY scripts/run_pace_vr_endurance_campaign.py prepare \
  --run-directory "$RUN" \
  --pace-vr-config asymmetric-sides-r192-rho-bo-20260928/pace-vr-async-h100-k20-template.json \
  --unit-cpus 12,13 --pace-cpus 14,15 --pace-vr-cpus 16,17 \
  --pace-vr-supervisor-cpus 18,19 --max-cycles 600 --python "$PY"
$PY scripts/run_pace_vr_endurance_campaign.py run --run-directory "$RUN" --max-parallel 3
$PY scripts/run_pace_vr_endurance_campaign.py analyze --run-directory "$RUN"
```

The three RHO pairs run on six dedicated CPUs; PACE-VR's bilateral slow supervisor runs
on two further CPUs. Every run has a distinct result directory. The launcher sets
common BLAS/OpenMP thread limits to one and writes separate logs. `analyze`
reports certified *pair* cycles, the mean and p95 solver time for each arm,
pair wall timing, last certified capacity ratios, weight changes, and available
PACE-VR value receipts. The report is written to `comparison.json` in the
campaign directory. Supervisor runtime, deadline misses, stale proposals,
source-to-application lag, and fast RHO preparation time at 20-cycle
boundaries are reported separately. The recorded CPU affinity and these
receipts should be checked before calling the timing comparison valid.

For runs using candidate screening, each arm also has a
`pace_vr_candidate_screen` audit in `comparison.json`: the number of screened
horizon attempts, proposals versus incumbent retentions, the candidate and
horizon selected for application, predicted surrogate improvement, and rollout
runtime. `audit_available: false` means the run predates candidate screening;
it is not evidence that the incumbent won. The improvement is a score on the
reduced rollout, not a measured gain in endurance. To inspect an existing
campaign without modifying its output directory, pass a separate report path:

```bash
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
RUN=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage/asymmetric-sides-r192-rho-bo-20260928/pace-vr-ladder-h100-50-25-20-5-20260930
$PY scripts/run_pace_vr_endurance_campaign.py analyze \
  --run-directory "$RUN" --report /tmp/pace-vr-existing-comparison.json
```

`run` may be invoked again to fill still-pending conditions. It skips complete
cap-limited results. A stopped result is skipped only after a separate
`physiological_review.json` containing the matching `completed_rho_cycles`, a
`physiological_infeasibility` verdict, and an existing `evidence_path` has been
placed in that condition's result directory. An interrupted directory or an
unreviewed stop blocks the next launch. This is campaign-level resume; the
current process RHO cannot reconstruct its complete dynamic state from an
interrupted output directory and continue from that physical cycle.

`summary.json` counts the failed attempted boundary as `completed_rho_cycles`.
For example, if the left arm fails its 171st attempt, the paired certified
prefix contains 170 cycles. Neither a failed IPOPT solve nor a rejected
surrogate rollout alone establishes a physiological endpoint. A separate
frozen-state feasibility review is needed before marking one.
