#!/usr/bin/env bash
# Replays recorded original-space IPOPT calls for a fair local Radau-5 A/B.
set -euo pipefail
source .github/scripts/benchmark_env.sh rho32
mode="${1:-local}"
inputs="${2:?Pass the baseline symbolic-audit directory containing nlp*_call*.npz}"
destination="${3:-ding-local-radau5-frozen-$(date +%Y%m%dT%H%M%S)/$mode}"
windows="${4:-9}"
mkdir -p "$destination"
# The benchmark example changes its working directory while preparing the OCP.
# Freeze these paths before starting it, otherwise relative archive paths break.
inputs="$(realpath "$inputs")"
DING_FROZEN_INPUT_DIR="$inputs" DING_FROZEN_CALL_OFFSET="${DING_FROZEN_CALL_OFFSET:-0}" \
  DING_FROZEN_RETURN_RECORDED=1 \
  bash scripts/run_ding_radau5_local_ab.sh "$windows" "$mode" "$destination"
