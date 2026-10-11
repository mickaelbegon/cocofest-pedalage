#!/usr/bin/env bash
# Extend the prescribed-isokinetic fatigue FHO beyond 78 cycles in 3-cycle
# increments. RHO attempts improve the warm start but never veto an FHO
# restoration: only the FHO operational audit (IPOPT + mechanics + <=0.1 %
# high-accuracy work error) decides whether the chain advances.
set -euo pipefail

ROOT=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
OUT="$ROOT/local-results/isokinetic-prescribed-unit-1p3-fho-bo64-20260930/fho-continuation"
# The PW² one-cycle continuation is the current owner of CPU 0--11.
WAIT_UNIT=fho-pw-bo-from73-20261001.service
# FHO85 is the last accepted operational prefix.  Resume exactly there rather
# than replaying the already audited 79--85 bridge attempts.
START=85
STEP=3
MAX=180

export PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
mkdir -p "$OUT/attempt-logs" "$OUT/attempts"

# This is the first queue consumer.  PW² waits for this service below; do not
# wait for the PW² *queue* service here, otherwise the two wrappers deadlock.

current=$START
while [ "$current" -lt "$MAX" ]; do
    advanced=false
    for increment in 3 2 1; do
        [ $((current + increment)) -le "$MAX" ] || continue
        # Each fallback is immutable.  In particular, a rejected +3 archive
        # can never be mistaken for the +2/+1 warm start on a later retry.
        attempt_tag="from-${current}-plus${increment}-$(date -u +%Y%m%dT%H%M%SZ)"
        attempt_dir="$OUT/attempts/$attempt_tag"
        log="$OUT/attempt-logs/${attempt_tag}.log"
        options=(
            --output-dir "$OUT" --start-cycles "$current" --extension-cycles "$increment"
            --attempt-label "plus${increment}" --attempt-dir "$attempt_dir" --relative-energy-limit 0.001
            --allow-rho-failure-fho-bridge
        )
        # Keep the legacy one-cycle finite-RHO option as an extra seed source.
        # The bridge above also covers +3/+2 and a RHO that exports no result.
        if [ "$increment" -eq 1 ]; then
            options+=(--allow-nonoptimal-single-rho-seed)
        fi
        if taskset -c 0-11 "$PY" "$ROOT/scripts/run_isokinetic_fho_operational_continuation.py" \
            "${options[@]}" >"$log" 2>&1; then
            current=$((current + increment))
            advanced=true
            break
        fi
    done
    if [ "$advanced" != true ]; then
        printf 'STOP: +3, +2, and +1 extensions from FHO_%04d were rejected; see attempt-logs/ and attempts/.\n' "$current" \
          >"$OUT/STOP-after-fho-${current}.txt"
        exit 0
    fi
done

printf 'STOP: reached configured safety cap FHO_%04d without a rejected extension.\n' "$current" \
  >"$OUT/STOP-after-fho-${current}.txt"
