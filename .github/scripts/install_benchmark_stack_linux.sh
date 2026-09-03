#!/usr/bin/env bash
set -euo pipefail

# One-shot, resumable installer for a cycling-benchmark environment.
#
#   bash .github/scripts/install_benchmark_stack_linux.sh rho32      # IPOPT, FATROP, ACADOS
#   bash .github/scripts/install_benchmark_stack_linux.sh madnlp32   # MadNLP/MUMPS + IPOPT
#
# It performs sections 5 to 8 of docs/cycling_solver_benchmark/linux_32core_setup.md.
# Every step is skipped when its artefact already exists, so the script can be
# re-run after a failure without redoing the long compilations. Use --force-step
# to redo one of them explicitly.
#
# Options:
#   --force-step STEP   redo one of: deps, env, casadi, biorbd, bioptim, acados, libmad
#   --jobs N            build parallelism (default: nproc)
#   --list-steps        print the step order for this flavour and exit

readonly BIOPTIM_REPOSITORY="https://github.com/mickaelbegon/BiorbdOptim.git"
readonly BIOPTIM_COMMIT="f7a0d722526967d9a81a8ad596ddb911d32a0bfe"
readonly ACADOS_COMMIT="59d93e17d2985fdd73fc58b8a83ed8f83a024171"
readonly LIBMAD_REPOSITORY="https://github.com/mickaelbegon/libMad.git"
readonly LIBMAD_COMMIT="5529f23a6bff33c566ad954da38d352f1f172356"
readonly JULIAC_COMMIT="73be8587a80bbb65dab7acd71d406f72867a3571"
readonly CASADI_WHEEL_VERSION="3.7.2"
readonly CASADI_MADNLP_COMMIT="973b086f4dcda9f49cd9c1948432ae4b7ee54886"

flavour=""
force_step=""
jobs="$(nproc)"
list_steps=false

while (( $# > 0 )); do
  case "$1" in
    rho32|madnlp32) flavour="$1"; shift ;;
    --force-step) force_step="${2:?--force-step needs a value}"; shift 2 ;;
    --jobs) jobs="${2:?--jobs needs a value}"; shift 2 ;;
    --list-steps) list_steps=true; shift ;;
    -h|--help) sed -n '3,20p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$flavour" != rho32 && "$flavour" != madnlp32 ]]; then
  echo "usage: $0 {rho32|madnlp32} [--force-step STEP] [--jobs N]" >&2
  exit 2
fi
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "--jobs must be a strictly positive integer, got '$jobs'." >&2
  exit 2
fi

if [[ "$flavour" == rho32 ]]; then
  steps=(deps env casadi biorbd bioptim acados)
else
  steps=(deps env libmad casadi biorbd bioptim)
fi
if [[ "$list_steps" == true ]]; then
  printf '%s\n' "${steps[@]}"
  exit 0
fi
if [[ -n "$force_step" ]]; then
  found=false
  for s in "${steps[@]}"; do [[ "$s" == "$force_step" ]] && found=true; done
  if [[ "$found" != true ]]; then
    echo "--force-step '$force_step' is not one of: ${steps[*]}" >&2
    exit 2
  fi
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${COCOFEST_ROOT:-$(cd "$script_dir/../.." && pwd)}"
cd "$root"
env_name="cocofest-$flavour"
miniforge="${MINIFORGE_PREFIX:-$HOME/miniforge3}"
deps_dir="$root/.benchmark-deps"

log() { printf '\n=== %s ===\n' "$*"; }
skip() { printf '    (already done: %s)\n' "$*"; }
should_run() { [[ "$force_step" == "$1" ]]; }

# --------------------------------------------------------------- preflight ---
# Only tools the host must provide. cmake and ninja deliberately are not in
# this list: they ship with the Conda environment created below, so probing for
# them here would reject a perfectly good host.
missing_tools=()
for tool in git curl tar jq cc; do
  command -v "$tool" >/dev/null 2>&1 || missing_tools+=("$tool")
done
if (( ${#missing_tools[@]} > 0 )); then
  echo "Missing host tools: ${missing_tools[*]}" >&2
  echo "Install them first (section 4 of linux_32core_setup.md)." >&2
  exit 1
fi

if [[ ! -r "$miniforge/etc/profile.d/conda.sh" ]]; then
  echo "Conda not found at $miniforge. Install Miniforge (section 5) or set MINIFORGE_PREFIX." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$miniforge/etc/profile.d/conda.sh"

# -------------------------------------------------------------------- deps ---
log "step deps: pinned sources"
mkdir -p "$deps_dir"
clone_pinned() {
  local repo="$1" dest="$2" commit="$3" recurse="$4"
  if [[ ! -d "$dest/.git" ]]; then
    if [[ "$recurse" == recurse ]]; then
      git clone --recurse-submodules "$repo" "$dest"
    else
      git clone "$repo" "$dest"
    fi
  fi
  if [[ "$(git -C "$dest" rev-parse HEAD)" != "$commit" ]] || should_run deps; then
    git -C "$dest" fetch --all --tags --quiet || true
    git -C "$dest" checkout --quiet "$commit"
    if [[ "$recurse" == recurse ]]; then
      git -C "$dest" submodule update --init --recursive
    fi
  else
    skip "$(basename "$dest") at $commit"
  fi
}
clone_pinned "$BIOPTIM_REPOSITORY" "$deps_dir/bioptim" "$BIOPTIM_COMMIT" recurse
clone_pinned "$LIBMAD_REPOSITORY" "$deps_dir/libMad" "$LIBMAD_COMMIT" plain

for check in "$deps_dir/bioptim:$BIOPTIM_COMMIT" \
             "$deps_dir/bioptim/external/acados:$ACADOS_COMMIT" \
             "$deps_dir/libMad:$LIBMAD_COMMIT"; do
  path="${check%:*}"; want="${check##*:}"
  got="$(git -C "$path" rev-parse HEAD)"
  if [[ "$got" != "$want" ]]; then
    echo "Pin mismatch in $path: expected $want, found $got" >&2
    exit 1
  fi
done
test -f "$deps_dir/bioptim/pyproject.toml"
echo "    pins verified"

# --------------------------------------------------------------------- env ---
log "step env: Conda environment $env_name"
conda_envs="$(conda env list)"
if grep -qE "^${env_name}[[:space:]]" <<<"$conda_envs" && ! should_run env; then
  skip "$env_name exists"
  conda env update --name "$env_name" \
    --file .github/cycling-benchmark-linux-environment.yml --prune >/dev/null
else
  conda env create --name "$env_name" \
    --file .github/cycling-benchmark-linux-environment.yml 2>/dev/null ||
    conda env update --name "$env_name" \
      --file .github/cycling-benchmark-linux-environment.yml --prune
fi
conda activate "$env_name"

missing_env_tools=()
for tool in cmake ninja python; do
  command -v "$tool" >/dev/null 2>&1 || missing_env_tools+=("$tool")
done
if (( ${#missing_env_tools[@]} > 0 )); then
  echo "$env_name is missing: ${missing_env_tools[*]}" >&2
  echo "Rebuild it with --force-step env." >&2
  exit 1
fi
echo "    cmake $(cmake --version | head -1 | awk '{print $3}'), python $(python -c 'import platform;print(platform.python_version())')"

export CMAKE_BUILD_PARALLEL_LEVEL="$jobs"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"

# ------------------------------------------------------------------ libmad ---
if [[ "$flavour" == madnlp32 ]]; then
  log "step libmad: Julia runtime and libMad/MUMPS"
  export PATH="$HOME/.juliaup/bin:$HOME/.julia/bin:$PATH"
  if ! command -v julia >/dev/null 2>&1; then
    echo "Julia not found. Install it first (section 8.1):" >&2
    echo "  curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.12.6" >&2
    exit 1
  fi
  bash "$script_dir/check_libmad_host_linux.sh"
  if [[ -f "$root/.cache/madnlp-mumps/lib/libMad.so" ]] && ! should_run libmad; then
    skip "libMad.so present"
  else
    mkdir -p "$root/.cache/madnlp-mumps"
    bash "$script_dir/install_libmad_mumps_linux.sh" \
      "$deps_dir/libMad" "$root/.cache/madnlp-mumps" "$JULIAC_COMMIT"
  fi
  export LD_LIBRARY_PATH="$root/.cache/madnlp-mumps/lib:$root/.cache/madnlp-mumps/share/julia/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ------------------------------------------------------------------ casadi ---
log "step casadi"
casadi_ok() {
  python - "$1" <<'PY' 2>/dev/null
import sys
import casadi as cas
want_madnlp = sys.argv[1] == "madnlp32"
assert cas.__version__.startswith("3.7.2")
assert cas.has_nlpsol("ipopt")
if want_madnlp:
    assert cas.has_nlpsol("madnlp")
    assert "-DCASADI_WITH_THREAD" in cas.CasadiMeta.compiler_flags()
else:
    assert cas.has_nlpsol("fatrop")
PY
}
if casadi_ok "$flavour" && ! should_run casadi; then
  skip "CasADi $(python -c 'import casadi;print(casadi.__version__)') already satisfies $flavour"
elif [[ "$flavour" == rho32 ]]; then
  export CASADI_CXX_ABI=0
  python -m pip install --no-deps --force-reinstall "casadi==$CASADI_WHEEL_VERSION"
else
  export CASADI_CXX_ABI=1
  export CASADI_VERSION="$CASADI_WHEEL_VERSION"
  export CASADI_MADNLP_COMMIT
  bash "$script_dir/install_casadi_madnlp_linux.sh" "$root/.cache/madnlp-mumps"
fi

# ------------------------------------------------------------------ biorbd ---
log "step biorbd (CasADi ABI ${CASADI_CXX_ABI:-0})"
if [[ "$flavour" == rho32 ]]; then export CASADI_CXX_ABI=0; else export CASADI_CXX_ABI=1; fi
if python -c 'import biorbd_casadi' >/dev/null 2>&1 && ! should_run biorbd; then
  skip "biorbd_casadi $(python -c 'import biorbd_casadi as b;print(b.__version__)')"
else
  bash "$script_dir/install_biorbd_casadi_linux.sh"
fi

# ----------------------------------------------------------------- bioptim ---
log "step bioptim"
bioptim_path="$(python -c 'import bioptim,pathlib;print(pathlib.Path(bioptim.__file__).resolve())' 2>/dev/null || true)"
if [[ "$bioptim_path" == "$deps_dir/bioptim/bioptim/__init__.py" ]] && ! should_run bioptim; then
  skip "editable install points at $deps_dir/bioptim"
else
  # A stale editable install survives the deletion of its source tree, so always
  # uninstall before reinstalling rather than trusting the import to fail.
  if [[ -n "$bioptim_path" ]]; then
    echo "    replacing bioptim loaded from $bioptim_path"
  fi
  python -m pip uninstall -y bioptim >/dev/null 2>&1 || true
  python -m pip install --no-deps -e "$deps_dir/bioptim"
fi

# ------------------------------------------------------------------ acados ---
if [[ "$flavour" == rho32 ]]; then
  log "step acados"
  # t_renderer is part of a complete ACADOS install: without it code generation
  # fails at the first solve, so a prefix missing it is not "already done".
  if [[ -f "$CONDA_PREFIX/lib/libacados.so" && -f "$CONDA_PREFIX/lib/libhpipm.so" &&
        -x "$CONDA_PREFIX/bin/t_renderer" ]] &&
     python -c 'import acados_template' >/dev/null 2>&1 && ! should_run acados; then
    skip "libacados.so, t_renderer and acados_template present"
  elif [[ -f "$CONDA_PREFIX/lib/libacados.so" && ! -x "$CONDA_PREFIX/bin/t_renderer" ]] &&
       ! should_run acados; then
    echo "    ACADOS libraries present, installing the missing t_renderer only"
    bash "$script_dir/install_acados_tera_renderer.sh" "$CONDA_PREFIX"
  else
    bash "$script_dir/install_acados_linux.sh" "$deps_dir/bioptim" "$jobs" "$CONDA_PREFIX"
  fi
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ---------------------------------------------------------------- validate ---
log "validation"
python - "$flavour" <<'PY'
import sys

import bioptim
import biorbd_casadi
import casadi as cas

flavour = sys.argv[1]
print(f"CasADi           {cas.__version__}")
print(f"biorbd           {biorbd_casadi.__version__}")
print(f"Bioptim          {bioptim.__version__}")
print(f"Bioptim path     {bioptim.__file__}")
assert cas.has_nlpsol("ipopt"), "ipopt missing"
if flavour == "rho32":
    import acados_template
    print(f"acados_template  {acados_template.__file__}")
    assert cas.has_nlpsol("fatrop"), "fatrop missing"
else:
    assert cas.has_nlpsol("madnlp"), "madnlp missing"
    assert "-DCASADI_WITH_THREAD" in cas.CasadiMeta.compiler_flags()
    assert hasattr(bioptim.Solver, "MADNLP"), "bioptim has no MADNLP solver"
print("solver backends OK")
PY

python -m pytest -q \
  tests/shard1/test_solver_backends.py \
  tests/shard1/test_reduced_cycling.py \
  tests/test_benchmark_readme.py

log "$env_name installed"
echo "Next: source .github/scripts/benchmark_env.sh $flavour"
