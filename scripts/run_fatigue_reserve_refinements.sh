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

for spec in '0.73 144 20261012' '1.30 300 20261013'; do
    read -r resistance total_trials random_seed <<< "$spec"
    if [[ "$resistance" == 0.73 ]]; then
        tag=0p73-plus44
        source_study="$root/local-results/fatigue-terminal-reserve-bo-0p73-100-20261009T1252Z/study.sqlite3"
    else
        tag=1p30-plus200
        source_study="$root/local-results/fatigue-terminal-reserve-resistance-sweep-20261009T1435Z/resistance-1p30nm/study.sqlite3"
    fi
    output="$output_root/$tag"
    echo "$(date -u +%FT%TZ) START ${resistance} Nm: $output"
    if "$python" scripts/run_isokinetic_fatigue_objective_bo.py \
        --variant worst_reserve \
        --root "$root" \
        --python "$python" \
        --hsl-library "$hsl" \
        --output "$output" \
        --warm-start-study "$source_study" \
        --cpu-set 0-29 \
        --resistance "$resistance" \
        --max-cycles 1500 \
        --workers 30 \
        --time-limit-s 14400 \
        --candidate-timeout-s 2400 \
        --trials "$total_trials" \
        --startup-trials 30 \
        --seed "$random_seed" \
        --ding-local \
        --final-validate \
        --validation-timeout-s 3600; then
        echo "$(date -u +%FT%TZ) END ${resistance} Nm"
    else
        status=$?
        echo "$(date -u +%FT%TZ) ERROR ${resistance} Nm: exit $status" >&2
    fi
done
