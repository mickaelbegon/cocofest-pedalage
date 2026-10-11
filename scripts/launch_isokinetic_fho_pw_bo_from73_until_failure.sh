#!/usr/bin/env bash
# Continue the fixed-weight PW² FHO one RHO cycle at a time from its last
# accepted operational prefix, FHO78.
set -euo pipefail

ROOT=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
BASE="$ROOT/local-results/isokinetic-prescribed-unit-1p3-pw2-20261001/fho-pw-bo-step1"
OUT="$BASE/continuation"
SOURCE="$BASE/fho-0073"
REFERENCE="$ROOT/local-results/isokinetic-prescribed-unit-1p3-pw2-20261001/rho-unit-pw/validated-rho-trajectory.npz"
WAIT_UNIT=fho-fatigue-beyond78-20261001-rerun.service
MAX=150
WEIGHTS=(1.6197008272329414 0.6423432578716108 0.8614265710555464 1.1157830567058247)

export PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT/attempts" "$OUT/attempt-logs"

# Promote the completed 73-cycle candidate into this continuation namespace
# once.  Its strict result stays untouched; the generic runner applies the
# explicit 0.1 % operational policy before it can seed RHO74.
if [ ! -f "$OUT/fho-0073-solution.npz" ]; then
    cp "$SOURCE/solution.npz" "$OUT/fho-0073-solution.npz"
    cp "$SOURCE/result.json" "$OUT/fho-0073-result.json"
fi

# The fatigue continuation owns CPU 0--11 until it stops.
while systemctl --user is-active --quiet "$WAIT_UNIT"; do
    sleep 30
done

current=78
while [ "$current" -lt "$MAX" ]; do
    next=$((current + 1))
    tag="from-${current}-plus1-$(date -u +%Y%m%dT%H%M%SZ)"
    attempt="$OUT/attempts/$tag"
    log="$OUT/attempt-logs/${tag}.log"
    if ! taskset -c 0-11 "$PY" "$ROOT/scripts/run_isokinetic_fho_operational_continuation.py" \
        --output-dir "$OUT" --attempt-dir "$attempt" --attempt-label plus1 \
        --start-cycles "$current" --extension-cycles 1 --relative-energy-limit 0.001 \
        --objective control --objective-weight-values "${WEIGHTS[@]}" \
        --reference-rho "$REFERENCE" --allow-nonoptimal-single-rho-seed \
        --allow-rho-failure-fho-bridge \
        >"$log" 2>&1; then
        printf 'STOP: FHO-PW² extension from %04d rejected; see %s\n' "$current" "$log" \
          >"$OUT/STOP-after-fho-${current}.txt"
        exit 0
    fi
    current=$next
done

printf 'STOP: reached safety cap FHO-PW²_%04d without rejected extension.\n' "$current" \
  >"$OUT/STOP-after-fho-${current}.txt"
