# Declarative campaign planning (issue #9, first tranche)

`python -m cocofest.simulation.campaign_manifest campaign.json` emits a JSON
dry-run plan. It reads existing artifacts but creates no directories, changes
no results and starts no processes. The launcher uses the same resolved
configuration and configuration hash as the GUI and simulation CLI.

```json
{
  "schema_version": 1,
  "campaign_id": "30hz-pilot",
  "output_root": "local-results",
  "reserved_cpus": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
  "lanes": {
    "ma57": {"cpus": [12, 13, 14, 15], "runtime_prefix": "/path/to/runtime"},
    "mumps": {"cpus": [16, 17, 18, 19], "runtime_prefix": "/path/to/runtime"}
  },
  "cases": [
    {"id": "K5-ma57", "lane": "ma57", "profile": "K5", "config": {"cycles": 100, "stimulations_per_cycle": 30}},
    {"id": "K7-ma57", "lane": "ma57", "profile": "K7", "config": {"cycles": 100, "stimulations_per_cycle": 30}},
    {"id": "K5-mumps", "lane": "mumps", "profile": "K5", "config": {"cycles": 100, "stimulations_per_cycle": 30, "ipopt_linear_solver": "mumps"}}
  ]
}
```

CPU lanes must be disjoint and cannot use a reserved CPU. All numerical
libraries receive a one-thread environment. Bioptim worker threads may be
specified in `config.threads`, up to the number of CPUs owned by the lane.
The planned command prefixes the existing launch argv with `taskset` and an
explicit CPU list. Availability of CPUs, taskset, runtime ABI, HSL licenses
and other external resources is not established by this planning step.

Each case owns `output_root/campaign_id/case_id`; native caches consequently
have distinct roots. Paths are resolved relative to `--root` (the repository
by default). `output_root`, `numeric_threads` and `dry_run` are campaign-managed
and cannot appear inside a case configuration. Unknown fields, unsafe case
identifiers, profile conflicts and unsupported simulation configurations fail
before producing a plan.

Cases in a lane implicitly depend on the previous case in that lane. Optional
`depends_on` lists express cross-lane prerequisites; each dependency must name
an earlier case, preventing dependency cycles. Independent ready lanes may be
scheduled in parallel by a future executor. A prerequisite is satisfied when
its result is *recorded*, including a documented native failure. This reflects
the campaign protocol allowing progress after reported failures; it is not a
scientific validation gate.

| State | Interpretation |
| --- | --- |
| `ready` | Empty/absent output directory and prerequisites recorded |
| `waiting` | Output is empty/absent but prerequisites have no recorded result |
| `recorded` | Non-empty result JSON plus matching effective configuration hash |
| `incomplete` | Existing artifacts need review (missing/invalid JSON or metadata) |
| `conflict` | Existing effective hash differs or output/result has incompatible structure |

`recorded` certifies neither convergence, feasibility nor dynamic consistency.
Scientific audit results remain in the result document and experiment catalog.
The dry run never replaces an incomplete or conflicting output. It does not
infer that a running process is safe to resume. Result hashes, semantic audits,
hybrid recoveries, runtime preflight enforcement, process supervision and
declarative K0–K10/sweep manifests remain follow-up work for issue #9.

At present, cumulative named profiles are the existing IPOPT `K5`, `K7`, `K9`.
Other solver configurations use explicit `config` fields and the capability
registry; selecting an IPOPT profile while overriding its solver is rejected.
`stimulations_per_cycle` denotes a count: it equals Hz only when cycle duration
is one second. The manifest does not turn arbitrary counts into Hz.
