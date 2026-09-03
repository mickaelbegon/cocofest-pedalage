# shellcheck shell=bash
#
# Activate one of the two cycling-benchmark environments with every variable the
# benchmark scripts expect. Source it, do not execute it:
#
#   source .github/scripts/benchmark_env.sh rho32      # IPOPT, FATROP, ACADOS
#   source .github/scripts/benchmark_env.sh madnlp32   # MadNLP/MUMPS + IPOPT
#
# Optional overrides, exported before sourcing:
#   COCOFEST_ROOT       repository checkout (default: this script's repository)
#   MINIFORGE_PREFIX    Conda installation (default: $HOME/miniforge3)
#   BENCHMARK_THREADS   --n-threads used by the benchmark (default: nproc)
#   NUMERIC_THREADS     BLAS/OpenMP/Julia threads (default: 1)
#
# Replaces the repeated export blocks of sections 7, 8, 10 and 11 of
# docs/cycling_solver_benchmark/linux_32core_setup.md.

_cocofest_env_setup() {
  local flavour="${1:-}"

  case "$flavour" in
    rho32|madnlp32) ;;
    *)
      echo "usage: source benchmark_env.sh {rho32|madnlp32}" >&2
      return 2
      ;;
  esac

  # ${BASH_SOURCE[0]} still resolves correctly when the file is sourced.
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return 1
  local root="${COCOFEST_ROOT:-$(cd "$script_dir/../.." && pwd)}"
  if [[ ! -f "$root/.github/cycling-benchmark-linux-environment.yml" ]]; then
    echo "COCOFEST_ROOT=$root does not look like a Cocofest checkout." >&2
    return 1
  fi

  local miniforge="${MINIFORGE_PREFIX:-$HOME/miniforge3}"
  if [[ ! -r "$miniforge/etc/profile.d/conda.sh" ]]; then
    echo "Conda not found at $miniforge (see linux_32core_setup.md section 5)." >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "$miniforge/etc/profile.d/conda.sh" || return 1

  local env_name="cocofest-$flavour"
  conda activate "$env_name" || {
    echo "Missing Conda environment '$env_name' (see linux_32core_setup.md sections 7 and 8)." >&2
    return 1
  }

  export COCOFEST_ROOT="$root"
  export GITHUB_WORKSPACE="$root"
  case ":${PYTHONPATH:-}:" in
    *":$root:"*) ;;
    *) export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" ;;
  esac

  # Keep runtime caches with the benchmark checkout. This works for headless
  # service accounts and runners whose home directory is read-only.
  export MPLBACKEND="${MPLBACKEND:-Agg}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-$root/.cache/matplotlib}"
  mkdir -p "$MPLCONFIGDIR" || return 1

  # Reproducible thread policy: stage/map parallelism on the benchmark side,
  # one thread inside every numerical library (section 9.1).
  export BENCHMARK_THREADS="${BENCHMARK_THREADS:-$(nproc)}"
  export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
  local numeric="${NUMERIC_THREADS:-1}"
  export OMP_NUM_THREADS="$numeric"
  export OMP_THREAD_LIMIT="$numeric"
  export OPENBLAS_NUM_THREADS="$numeric"
  export MKL_NUM_THREADS="$numeric"
  export NUMEXPR_NUM_THREADS="$numeric"
  export JULIA_NUM_THREADS="$numeric"

  local libdirs
  if [[ "$flavour" == rho32 ]]; then
    export CASADI_CXX_ABI=0
    libdirs="$CONDA_PREFIX/lib"
  else
    export CASADI_CXX_ABI=1
    export CASADI_VERSION=3.7.2
    export CASADI_MADNLP_COMMIT=973b086f4dcda9f49cd9c1948432ae4b7ee54886
    export JULIAC_COMMIT=73be8587a80bbb65dab7acd71d406f72867a3571
    export PATH="$HOME/.juliaup/bin:$HOME/.julia/bin:$PATH"
    local writable_julia_depot="$root/.cache/julia-benchmark-depot"
    mkdir -p "$writable_julia_depot" || return 1
    case ":${JULIA_DEPOT_PATH:-}:" in
      *":$writable_julia_depot:"*) ;;
      *) export JULIA_DEPOT_PATH="$writable_julia_depot:${JULIA_DEPOT_PATH:-$HOME/.julia}" ;;
    esac
    libdirs="$root/.cache/madnlp-mumps/lib:$root/.cache/madnlp-mumps/share/julia/lib"
  fi
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${libdirs%%:*}:"*) ;;
    *) export LD_LIBRARY_PATH="$libdirs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac

  echo "$env_name ready"
  echo "  COCOFEST_ROOT     $COCOFEST_ROOT"
  echo "  CONDA_PREFIX      $CONDA_PREFIX"
  echo "  BENCHMARK_THREADS $BENCHMARK_THREADS (numeric libraries: $numeric)"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "benchmark_env.sh must be sourced, not executed:" >&2
  echo "  source ${BASH_SOURCE[0]} {rho32|madnlp32}" >&2
  exit 2
fi

_cocofest_env_setup "$@"
