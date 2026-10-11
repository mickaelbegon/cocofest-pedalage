#!/usr/bin/env bash
set -uo pipefail

root=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
python=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
hsl=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so
output_root=${1:?Pass an unused campaign output directory}

if [[ -e "$output_root" ]]; then
    echo "Output directory already exists: $output_root" >&2
    exit 2
fi
mkdir -p "$output_root"
cd "$root" || exit 2

for resistance in 0.75 0.80 0.90 1.00 1.15 1.20 1.25 1.30; do
    tag=${resistance/./p}
    output="$output_root/resistance-${tag}nm"
    echo "$(date -u +%FT%TZ) START ${resistance} Nm: $output"
    if "$python" scripts/run_isokinetic_fatigue_objective_bo.py \
        --variant worst_reserve \
        --root "$root" \
        --python "$python" \
        --hsl-library "$hsl" \
        --output "$output" \
        --cpu-set 0-29 \
        --resistance "$resistance" \
        --max-cycles 1500 \
        --workers 30 \
        --time-limit-s 14400 \
        --candidate-timeout-s 2400 \
        --trials 100 \
        --startup-trials 30 \
        --seed 20261011 \
        --ding-local \
        --final-validate \
        --validation-timeout-s 3600; then
        echo "$(date -u +%FT%TZ) END ${resistance} Nm"
    else
        status=$?
        echo "$(date -u +%FT%TZ) ERROR ${resistance} Nm: exit $status" >&2
    fi
done
