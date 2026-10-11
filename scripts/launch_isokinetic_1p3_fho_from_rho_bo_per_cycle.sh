#!/usr/bin/env bash
# Global FHO refinement of the full RHO-BO fatigue prefix, then +1/+2/+3 RHO
# extensions.  Every FHO boundary carries one prescribed cycle of work.
set -euo pipefail

ROOT=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
HSL=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so
RHO_BO="$ROOT/local-results/isokinetic-prescribed-unit-1p3-pw2-20261001/analysis/replay-rho-bo-min-fatigue/trajectory.npz"
OUT="$ROOT/local-results/isokinetic-prescribed-unit-1p3-fho-from-rho-bo-per-cycle-radau5-20261002/fho-continuation"
WEIGHTS=(0.163448160932 0.772705726593 0.211062370012 0.942782170747)
START=82
FHO_ITER_STANDARD=800
FHO_ITER_RETRY=1500

export PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
mkdir -p "$OUT" "$OUT/attempts" "$OUT/attempt-logs"

"$PY" "$ROOT/scripts/rebase_rho_energy_for_fho.py" \
  --input "$RHO_BO" --output "$OUT/rho-bo-cumulative-seed.npz" \
  --cycles "$START" --work-per-cycle-j 8.168140899333462

# The BO weights remain fixed.  This is a single global FHO optimisation of
# the RHO-BO trajectory, not a relabelling of the RHO solution.
if [ ! -f "$OUT/fho-0082-result.json" ]; then
taskset -c 0-11 "$PY" "$ROOT/examples/fes_multibody/cycling/cycling_fes_solver_comparison.py" \
  --solvers ipopt --objective fatigue --objective-shape quadratic \
  --parametric-fatigue-weights --fatigue-weight-values "${WEIGHTS[@]}" \
  --formulation isokinetic --mechanical-formulation reduced \
  --isokinetic-kinematics prescribed --enforce-isokinetic-work-per-cycle \
  --isokinetic-omega -6.283185307179586 --energy-equivalent-torque 1.3 \
  --load-torque-min -4 --load-torque-max 4 --stimulations-per-cycle 30 \
  --ipopt-profile periodic_collocation --ipopt-collocation-method radau \
  --ipopt-collocation-degree 5 --ipopt-no-use-sx --ipopt-linear-solver ma57 \
  --warmup-ipopt-linear-solver ma57 --ipopt-hsl-library "$HSL" \
  --ipopt-max-iter "$FHO_ITER_STANDARD" --nlp-tolerance 1e-8 --primal-feasibility-threshold 1e-5 \
  --ipopt-enforce-start-constraints --ipopt-disable-standard-warmup \
  --adopt-common-initial-solution-warmup-cycles --optional-nlp-periodic-ipopt-hot-start \
  --initial-guess-diagnostics --single-shot --cycles-per-window "$START" --n-windows "$START" \
  --n-threads 12 --common-initial-solution "$OUT/rho-bo-cumulative-seed.npz" \
  --common-initial-solution-output "$OUT/fho-0082-solution.npz" \
  --output-json "$OUT/fho-0082-result.json" >"$OUT/fho-0082-driver.log" 2>&1
fi

current=$(find "$OUT" -maxdepth 1 -name 'fho-????-continuation-provenance.json' -printf '%f\n' \
  | sed -E 's/fho-0*([0-9]+)-continuation-provenance\.json/\1/' | sort -n | tail -1)
current=${current:-$START}
latest_promoted_cycle() {
  find "$OUT" -maxdepth 1 -name 'fho-????-continuation-provenance.json' -printf '%f\n' \
    | sed -E 's/fho-0*([0-9]+)-continuation-provenance\.json/\1/' | sort -n | tail -1
}
while [ "$current" -lt 180 ]; do
  advanced=false
  for increment in 3 2 1; do
    attempt="$OUT/attempts/from-${current}-plus${increment}-$(date -u +%Y%m%dT%H%M%SZ)"
    if taskset -c 0-11 "$PY" "$ROOT/scripts/run_isokinetic_fho_operational_continuation.py" \
      --output-dir "$OUT" --reference-rho "$RHO_BO" --start-cycles "$current" \
      --extension-cycles "$increment" --attempt-label "plus${increment}" \
      --attempt-dir "$attempt" --relative-energy-limit 0.002 --collocation-degree 5 \
      --fho-ipopt-max-iter "$FHO_ITER_STANDARD" \
      --objective fatigue --objective-weight-values "${WEIGHTS[@]}" \
      --allow-rho-failure-fho-bridge \
      >"$OUT/attempt-logs/$(basename "$attempt").log" 2>&1; then
      next=$(latest_promoted_cycle)
      if [ -z "$next" ] || [ "$next" -le "$current" ]; then
        echo "ERROR: successful continuation did not promote a later FHO." >&2
        exit 1
      fi
      current=$next
      advanced=true
      break
    else
      # A failed RHO shortens a requested +3/+2 extension to the first
      # non-converged cycle, so discover the FHO actually attempted.
      result=$(find "$attempt" -maxdepth 1 -name 'fho-????-result.json' -print -quit)
      if [ -n "$result" ] && "$PY" "$ROOT/scripts/fho_iteration_retry_eligible.py" "$result" \
        --iteration-budget "$FHO_ITER_STANDARD"; then
        retry="${attempt}-retry1500"
        if taskset -c 0-11 "$PY" "$ROOT/scripts/run_isokinetic_fho_operational_continuation.py" \
          --output-dir "$OUT" --reference-rho "$RHO_BO" --start-cycles "$current" \
          --extension-cycles "$increment" --attempt-label "plus${increment}-retry1500" \
          --attempt-dir "$retry" --relative-energy-limit 0.002 --collocation-degree 5 \
          --fho-ipopt-max-iter "$FHO_ITER_RETRY" \
          --objective fatigue --objective-weight-values "${WEIGHTS[@]}" \
          --allow-rho-failure-fho-bridge \
          >"$OUT/attempt-logs/$(basename "$retry").log" 2>&1; then
          next=$(latest_promoted_cycle)
          if [ -z "$next" ] || [ "$next" -le "$current" ]; then
            echo "ERROR: successful retry did not promote a later FHO." >&2
            exit 1
          fi
          current=$next
          advanced=true
          break
        fi
      fi
      # If the runner already shortened this request at the first failed RHO,
      # +2 and +1 would reconstruct exactly the same FHO.  Stop instead of
      # wasting two duplicate FHO solves.
      if "$PY" -c '
import json, sys
from pathlib import Path
attempt = Path(sys.argv[1])
for audit_path in attempt.glob("*audit.json"):
    try:
        if json.loads(audit_path.read_text()).get("extension_truncated_at_first_nonconverged_rho"):
            raise SystemExit(0)
    except (OSError, ValueError):
        pass
raise SystemExit(1)
' "$attempt"; then
        echo "STOP: FHO bridge from first non-converged RHO failed; no duplicate +2/+1 retry." >&2
        break
      fi
    fi
  done
  if [ "$advanced" != true ]; then
    printf 'STOP: no accepted +3, +2, or +1 extension from FHO_%04d.\n' "$current" \
      >"$OUT/STOP-after-fho-${current}.txt"
    exit 0
  fi
done
