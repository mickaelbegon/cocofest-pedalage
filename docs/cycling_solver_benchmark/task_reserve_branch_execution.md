# Short reachable-state branch executor

`scripts/run_local_task_reserve_branch.py` executes one policy from a
`local_task_reserve_experiment_plan`. It starts from the plan's certified exact
bilateral anchor, but advances only the requested arm. A branch is not a
bilateral continuation or an endurance benchmark.

```bash
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/run_local_task_reserve_branch.py \
  --plan /absolute/path/local-plan.json \
  --anchor c140 --branch contrast-1-plus \
  --output-directory /absolute/path/new-branch-directory
```

Run the `unit` branch first to check the restored nominal continuation. Training
policies export at 1 and 3 cycles; predeclared holdout policies export at 2 and
4 cycles. Endpoints from one branch retain its same partition and shared
trajectory provenance.

The worker builds the original configured RHO once, restores its prepared
states, bounds, parameters and stimulation history, then updates only the
existing fixed fatigue-weight parameter. It checks that the primal, physical
bounds and other parameters remain identical during the weight update. Each
cycle is solved at nominal work and independently audited against every NLP
constraint and decision bound before the native RHO advance. The compiled
solver object must remain identical across successive solves. No symbolic
objective replacement or synthetic fatigue-state perturbation is used.

Each reached endpoint has a prepared NPZ and full-cycle witness artifacts. A
second spawned process rebuilds and restores that endpoint; only matching
digests produce `cycle-N-receipt.json`. Its kind is
`certified_unilateral_task_reserve_branch_endpoint`; `exact_bilateral_restart`
is always false. Existing bilateral receipt loaders remain strict and unchanged.
The separate `load_branch_endpoint` loader checks source/plan digests, policy,
work, complete certified prefix, witness hashes, archive bytes and fresh
restoration. It returns a `TaskReserveCheckpoint` for subsequent load probes.

To calibrate from these endpoints, use the existing calibration script with a
separate branch manifest. Set `source_kind` to
`certified_unilateral_branch_endpoint` and `experiment_plan` to the exact plan
used by every receipt. Each sample's `receipt` is a `cycle-N-receipt.json` and
its `partition` must match that policy's partition in the plan. The remaining
`work_scales`, `coordinates`, `fit`, `tolerance`, and `samples` fields have the
same meaning as in the bilateral calibration manifest. For example:

```json
{
  "schema_version": 1,
  "side": "right",
  "source_kind": "certified_unilateral_branch_endpoint",
  "experiment_plan": "/absolute/path/local-plan.json",
  "work_scales": [1.0, 1.05, 1.1],
  "tolerance": 1e-5,
  "coordinates": [{"state_key": "A_Biceps", "index": 0, "scale": 1.0, "offset": 0.0}],
  "fit": {"center": [0.8], "trust_radius": [0.1]},
  "samples": [{"id": "anchor140-contrast1-cycle3", "partition": "train",
               "receipt": "/absolute/path/cycle-3-receipt.json"}]
}
```

All endpoint receipts and coordinates are checked before any missing probes
run. Endpoints sharing the same anchor and branch policy also share a trajectory
prefix and cannot be counted as independent samples in one fit. The audit keeps
the physical context common while recording anchor, policy, cycle and source
receipt separately for each observation. The usual minimum number of train and
holdout samples still applies; the one-sample example above only shows format.

Physical-task context keeps all source configuration fields except the two
initial policy-weight vectors, which are recorded separately. Model, load,
cadence, constraints, integration, solver and tolerances are retained. Source
files are never modified. The initial pilot requires a unit-weight anchor,
parametric weights already present, and the repeat PW transfer: non-repeat
predictors need historical memory not represented by the current archive.

Failed nominal cycles stop the branch before its physical state is advanced.
Previously certified endpoints remain useful; a failed solve is indeterminate,
not proof of physiological failure. Certification currently covers the
discrete transcription and exact prepared-history restoration, not independent
continuous DOP853 replay. Trust-box, reachable-design rank, held-out reserve
prediction and matched endurance continuation remain later validation stages.
