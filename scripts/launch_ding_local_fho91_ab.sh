#!/usr/bin/env bash
# 10/30/60/91-cycle Ding-local FHO ladder. Each stage must complete before
# the next one is attempted, so the 91-cycle test is reached only if SX scales.
# The production MX FHO is not rerun: its archived timing is the reference.
set -euo pipefail

ROOT=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
HSL=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so
REFERENCE="$ROOT/local-results/isokinetic-prescribed-unit-1p3-fho-from-rho-bo-per-cycle-radau5-20261002/fho-continuation"
SOURCE="$REFERENCE/attempts/from-88-plus3-20261002T232423Z"
OUT="$ROOT/local-results/ding-local-fho91-20261004"
WEIGHTS=(0.163448160932 0.772705726593 0.211062370012 0.942782170747)

export PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$ROOT"
if [ -e "$OUT" ]; then
  echo "Refusing to overwrite $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"

COMMON=(
  --solvers ipopt --objective fatigue --objective-shape quadratic
  --parametric-fatigue-weights --fatigue-weight-values "${WEIGHTS[@]}"
  --formulation isokinetic --mechanical-formulation reduced
  --isokinetic-kinematics prescribed --enforce-isokinetic-work-per-cycle
  --isokinetic-omega -6.283185307179586 --energy-equivalent-torque 1.3
  --load-torque-min -4 --load-torque-max 4 --stimulations-per-cycle 30
  --ipopt-profile periodic_collocation --ipopt-collocation-method radau
  --ipopt-collocation-degree 5 --ipopt-use-sx --ipopt-linear-solver ma57
  --warmup-ipopt-linear-solver ma57 --ipopt-hsl-library "$HSL"
  --ipopt-max-iter 800 --nlp-tolerance 1e-8 --primal-feasibility-threshold 1e-5
  --ipopt-enforce-start-constraints --ipopt-disable-standard-warmup
  --adopt-common-initial-solution-warmup-cycles --optional-nlp-periodic-ipopt-hot-start
  --initial-guess-diagnostics --single-shot --n-threads 12
)

run_local_case() {
  local cycles=$1
  local seed=$2
  local destination="$OUT/local-$(printf '%04d' "$cycles")"
  mkdir -p "$destination"
  taskset -c 0-11 "$PY" "$ROOT/scripts/benchmark_ding_radau5_local_ab.py" local -- "${COMMON[@]}" \
    --cycles-per-window "$cycles" --n-windows "$cycles" \
    --common-initial-solution "$seed" \
    --common-initial-solution-output "$destination/solution.npz" \
    --output-json "$destination/result.json" \
    >"$destination/driver.log" 2>&1
}

# The local adapter requires SX.  It is compared with the already recorded
# production FHO91 MX timing; no new full/SX FHO is launched.
for cycles in 10 30 60; do
  seed="$OUT/seeds/rho-bo-${cycles}.npz"
  "$PY" "$ROOT/scripts/truncate_fho_seed_cycles.py" \
    --input "$REFERENCE/rho-bo-cumulative-seed.npz" --output "$seed" --cycles "$cycles"
  run_local_case "$cycles" "$seed"
done
run_local_case 91 "$SOURCE/fho-0088-plus-rho-through-0091.npz"

"$PY" - "$OUT" "$REFERENCE/fho-0091-result.json" <<'PY'
import json
import sys
from pathlib import Path

out, production = map(Path, sys.argv[1:])
summary = {"production_mx_reference": str(production), "cases": {}}
for cycles in (10, 30, 60, 91):
    mode = f"local-{cycles:04d}"
    result = json.loads((out / mode / "result.json").read_text())["results"][0]
    metric = json.loads((out / mode / "symbolic-audit" / "nlp0_metrics.json").read_text())[-1]
    stats = (result.get("nlp_solver_stats") or [{}])[0]
    summary["cases"][mode] = {
        "success": bool(result.get("success")),
        "iterations": stats.get("iter_count"),
        "solver_wall_s": result.get("solver_time_s"),
        "wall_s": result.get("wall_time_s"),
        "original_nx": metric.get("original_nx"),
        "solver_nx": metric.get("solver_nx"),
        "original_ng": metric.get("original_ng"),
        "solver_ng": metric.get("solver_ng"),
        "solver_build_s": metric.get("solver_build_s"),
        "local_map_s": metric.get("mapping_s"),
        "solve_call_s": metric.get("call_wall_s"),
        "reconstruction_audit_s": metric.get("reconstruction_s"),
        "original_max_violation": metric.get("original_max_bound_constraint_violation"),
        "local_certificate": metric.get("discrete_equivalence_certificate_max"),
    }
production_result = json.loads(production.read_text())["results"][0]
production_stats = (production_result.get("nlp_solver_stats") or [{}])[0]
summary["production_mx"] = {
    "success": bool(production_result.get("success")),
    "iterations": production_stats.get("iter_count"),
    "solver_wall_s": production_result.get("solver_time_s"),
    "wall_s": production_result.get("wall_time_s"),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
