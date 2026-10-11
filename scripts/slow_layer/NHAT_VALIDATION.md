# Conditional Nhat diagnostic input contract

Run `python scripts/slow_layer/validate_nhat.py manifest.json`. The command
prints JSON and does not launch a solver, alter source archives, or activate a
controller. `nhat_manifest.template.json` is deliberately incomplete: nulls
are required study choices, not defaults. T1/T3 identify the prescribed study
datasets; the validator does not infer a dataset from a directory name.

All case artifact references contain `path` and lowercase `sha256`. Paths are
relative to the manifest, except a label's `audit_artifact`, which is relative
to that label. Oracle receipts use the existing byte-pinned complete-NLP
certification loader. The certified cycle uses the solved final node; a
checkpoint uses the next cycle's frozen initial equality.

The bounds JSON must contain `muscle_order`, `lower`, `upper`, and a nonempty
`declaration_reference` naming the externally prescribed study rule. Bounds
have shape `(muscles, 3)` in `A, Tau1, Km` order, use finite numbers, and are
inclusive. No default physiological threshold is supplied.

The force-profile JSON must contain `muscle_order`, `model_sha256`,
`assumption: "repeated_prescribed_force"`, `source_reference`, and `muscles`.
Each named muscle contains `parameters`, `force_samples`,
`interval_durations`, and `normalized_nodes`. Parameters use
`DingFatigueParameters` fields (`a_rest`, `tau1_rest`, `km_rest`, `alpha_a`,
`alpha_tau1`, `alpha_km`, `tau_fat`). Samples have shape `(intervals, nodes)`;
nodes increase from 0 to 1. These describe the supplied collocation force
polynomials, not a prediction of future RHO forces.

A continuation-label JSON contains:

- `schema_version: 1`, `certified: true`, `anchor_cycle`, `observed_cycles`;
- `initial_state_receipt_sha256`, the converted `SlowState.receipt_sha256`
  returned by `_oracle`, binding the exact initial observation and provenance;
- `protocol_id`, `audit_reference`, and a pinned `audit_artifact`;
- `censoring: "none"` or `"right"`;
- `terminal_classification`, and for an exact endpoint, `terminal_certified: true`.

`observed_cycles` counts additional certified completed cycles after the
anchor, not absolute cycle numbers. Exact endpoints require the external
protocol's `frozen_feasibility_exhaustion` attestation. A censored endpoint
must give `administrative_horizon`, `interrupted`, or
`unresolved_solver_failure`. A solver failure alone cannot supply an exact
endpoint. The diagnostic verifies artifact bytes and explicit attestations;
it does not rerun or independently prove a terminal infeasibility result.

The output gives per-case signed/absolute cycle errors only when both
endpoints are uncensored, interval errors otherwise (`null` means an
unbounded endpoint), exact-pair coverage, censoring counts, and input
provenance. Aggregate errors describe only the uncensored subset; they are
not population-wide unbiased accuracy estimates. A threshold violation at
cycle n means n-1 completed in-bound cycles. An initial violation means zero.

No acceptance criterion is prescribed in this format. Even a zero error
produces `validation_conclusion: diagnostics_only_no_prescribed_acceptance_criterion`
and `controller_enabled: false`. Synthetic tests establish implementation
correctness and cannot establish endurance validity.
