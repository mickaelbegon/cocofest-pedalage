#!/usr/bin/env bash
set -euo pipefail

root=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
python=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
hsl=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so
previous="$root/local-results/isokinetic-ding-local-bo-0p75-20261005/bo/0p75/summary.json"
output="$root/local-results/isokinetic-ding-local-bo-0p73-20261005"

export PYTHONPATH="$root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

while true; do
  if test -f "$previous" && "$python" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["completed"] == 48 else 1)' "$previous"; then
    break
  fi
  sleep 30
done

# Let the completed 0.75 worker pool exit before taking the same CPU set.
sleep 20
exec taskset -c 0-11 "$python" "$root/scripts/run_isokinetic_ding_local_endurance_sweep_bo.py" \
  --output "$output" --python "$python" --root "$root" --hsl-library "$hsl" \
  --start-resistance 0.73 --step 0.10 --minimum-resistance 0.73 \
  --target-cycles 1100 --max-cycles 1500 --trials 48 --workers 8 --cpu-set 0-11 \
  --ipopt-max-iter 1000 --startup-trials 16
