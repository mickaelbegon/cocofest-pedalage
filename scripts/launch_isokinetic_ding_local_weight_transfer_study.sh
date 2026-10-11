#!/usr/bin/env bash
set -euo pipefail

root=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
python=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
hsl=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so
output="$root/local-results/isokinetic-ding-local-weight-transfer-5r-48x8-20261004"

export PYTHONPATH="$root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec taskset -c 0-11 "$python" "$root/scripts/run_isokinetic_ding_local_weight_transfer_study.py" \
  --output "$output" --python "$python" --root "$root" --hsl-library "$hsl" \
  --resistances 1.15 1.20 1.25 1.30 1.35 --max-cycles 180 --trials 48 --workers 8 \
  --cpu-set 0-11 --ipopt-max-iter 1000 --startup-trials 16
