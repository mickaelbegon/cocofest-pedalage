#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
HSL=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so
OUT="$ROOT/local-results/isokinetic-prescribed-unit-1p3-fho-bo64-20260930/rho-bo-round2"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
taskset -c 12-19 "$PY" "$ROOT/scripts/run_isokinetic_unit_fixed_weight_bo.py" \
  --output "$OUT" --python "$PY" --root "$ROOT" --hsl-library "$HSL" \
  --resistance 1.3 --max-cycles 160 --trials 64 --workers 8 --cpu-set 12-19 --seed 20261001 \
  --initial-weights 0.16344816093170542 0.7727057265931767 0.21106237001190037 0.9427821707470022 \
  >"$ROOT/local-results/isokinetic-prescribed-unit-1p3-fho-bo64-20260930/rho-bo-round2-driver.log" 2>&1
