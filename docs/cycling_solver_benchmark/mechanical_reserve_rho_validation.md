# Validate the first mechanical-reserve RHO update

After a bilateral run of at least two cycles, audit its saved execution record:

```bash
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
"$PY" scripts/validate_mechanical_reserve_rho_run.py \
  asymmetric-sides-r192-rho-bo-20260928/mechanical-reserve-projection-smoke-3-results/summary.json
```

Exit code 0 means both arms have certified consecutive cycles; cycle 1 has zero reserve activation; the certified profile from cycle 1 is applied at cycle 2; the objective graph was built once; and the same compiled solver was observed across cycles. Exit code 1 prints explicit failed checks. The audit also requires recorded metadata to mark the mechanical margin as an **uncertified proxy** and not an endurance prediction.

This validates runtime wiring and the solved RHO cycles. It does not establish future feasibility or an endurance benefit. Those require comparison runs and physiological failure analysis.
