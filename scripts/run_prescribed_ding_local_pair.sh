#!/usr/bin/env bash
# Computational two-arm latency benchmark with one persistent process per arm.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?expected output directory}"
WINDOWS="${2:-100}"
LEFT_CPU="${3:-12}"
RIGHT_CPU="${4:-13}"
CACHE="${5:-$ROOT/local-results/prescribed-ding-local-autonomous-20261004/compiled100/cache}"
[[ "$LEFT_CPU" != "$RIGHT_CPU" ]] || { echo "CPUs must differ" >&2; exit 2; }
mkdir -p "$OUT/left" "$OUT/right" "$OUT/barrier"
OUT="$(realpath "$OUT")"
CACHE="$(realpath "$CACHE")"
cd "$ROOT"
export PYTHONPATH="$ROOT"
start_ns="$(date +%s%N)"
DING_PAIR_SIDE=left DING_PAIR_BARRIER_DIR="$OUT/barrier" \
  taskset -c "$LEFT_CPU" bash scripts/run_prescribed_ding_local_propagated.sh \
  compiled "$OUT/left" "$WINDOWS" "$CACHE" >"$OUT/left/solver.log" 2>&1 &
left_pid=$!
DING_PAIR_SIDE=right DING_PAIR_BARRIER_DIR="$OUT/barrier" \
  taskset -c "$RIGHT_CPU" bash scripts/run_prescribed_ding_local_propagated.sh \
  compiled "$OUT/right" "$WINDOWS" "$CACHE" >"$OUT/right/solver.log" 2>&1 &
right_pid=$!
left_status=0
right_status=0
wait "$left_pid" || left_status=$?
wait "$right_pid" || right_status=$?
end_ns="$(date +%s%N)"
printf '%s\n' "$start_ns $end_ns $left_status $right_status $LEFT_CPU $RIGHT_CPU" >"$OUT/process-timing.txt"
if [[ "$left_status" -ne 0 || "$right_status" -ne 0 ]]; then
  echo "paired workers failed: left=$left_status right=$right_status" >&2
  exit 1
fi
"${PYTHON:-/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python}" \
  scripts/summarize_prescribed_ding_local_pair.py "$OUT"
