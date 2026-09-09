#!/usr/bin/env bash
# Validate the inputs for a licensed CoinHSL rebuild in the active Conda env.
#
# This script intentionally never runs configure, make, install, conda, or a
# download.  It is a fail-fast pre-mutation gate: the license holder chooses
# the vendor-specific CoinHSL build command after this check succeeds.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/licensed/CoinHSL-source" >&2
  exit 2
fi

source_dir="$1"
if [[ ! -d "$source_dir" ]]; then
  echo "CoinHSL source directory is unavailable: $source_dir" >&2
  echo "Obtain the licensed CoinHSL source from its rights holder; this repository does not ship it." >&2
  exit 1
fi
if [[ -z "${CONDA_PREFIX:-}" || ! -d "$CONDA_PREFIX/lib" ]]; then
  echo "Activate the target Conda environment first (for example: source .github/scripts/benchmark_env.sh rho32)." >&2
  exit 1
fi

fortran_compiler="${FC:-}"
if [[ -z "$fortran_compiler" ]]; then
  for candidate in \
    "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gfortran" \
    "$CONDA_PREFIX/bin/gfortran"; do
    if [[ -x "$candidate" ]]; then
      fortran_compiler="$candidate"
      break
    fi
  done
fi
if [[ -z "$fortran_compiler" || ! -x "$fortran_compiler" ]]; then
  echo "No executable Conda Fortran compiler was found in $CONDA_PREFIX/bin." >&2
  echo "Install a compiler toolchain into this target environment before building CoinHSL; do not use the legacy libgfortran.so.4 workaround." >&2
  exit 1
fi

c_compiler="${CC:-}"
if [[ -z "$c_compiler" ]]; then
  for candidate in \
    "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
    "$CONDA_PREFIX/bin/gcc"; do
    if [[ -x "$candidate" ]]; then
      c_compiler="$candidate"
      break
    fi
  done
fi
if [[ -z "$c_compiler" || ! -x "$c_compiler" ]]; then
  echo "No executable Conda C compiler was found in $CONDA_PREFIX/bin." >&2
  echo "Install the matching C/Fortran toolchain into this target environment before building CoinHSL." >&2
  exit 1
fi

fortran_runtime="$CONDA_PREFIX/lib/libgfortran.so"
if [[ ! -e "$fortran_runtime" ]]; then
  echo "Missing active Conda Fortran runtime: $fortran_runtime" >&2
  exit 1
fi
fortran_runtime_name="$(basename "$(readlink -f "$fortran_runtime")")"
if [[ ! "$fortran_runtime_name" =~ ^libgfortran\.so\.([0-9]+) ]]; then
  echo "Could not determine the ABI major of $fortran_runtime." >&2
  exit 1
fi
fortran_abi_major="${BASH_REMATCH[1]}"

echo "CoinHSL rebuild prerequisites are ready (no files were modified)."
echo "  source:   $(readlink -f "$source_dir")"
echo "  CC:       $c_compiler"
echo "  FC:       $fortran_compiler"
echo "  runtime:  $fortran_runtime_name"
echo
echo "Build with the licensed vendor procedure while keeping these settings:"
echo "  export CC=$c_compiler"
echo "  export FC=$fortran_compiler"
echo "  export F77=$fortran_compiler"
echo "  export LDFLAGS='-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib'"
echo "  export LIBS='-llapack -lblas'"
echo
echo "Before any install, require readelf -d on the candidate library to list only"
echo "libgfortran.so.${fortran_abi_major} for libgfortran, then run the isolated"
echo "MA57 probe documented in linux_32core_setup.md. Do not install a candidate"
echo "whose probe reports a METIS runtime error or production_ready=false."
