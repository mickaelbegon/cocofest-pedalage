#!/usr/bin/env bash
set -euo pipefail

# Wrapper around bioptim's external/acados_install_linux.sh.
#
# It exists for two reasons that the upstream script cannot address on its own:
#
#   1. Toolchain compatibility. ACADOS 0.5.5 vendors qpOASES_e, whose
#      QProblemCPY() passes `Constraints**` where `Constraints*` is expected.
#      GCC 13 and earlier reported that as a warning, so the code built on the
#      ubuntu-24.04 CI image. GCC 14 promoted -Wincompatible-pointer-types (and
#      three sibling diagnostics) to errors by default, which breaks the build
#      on Ubuntu 25.10/26.04 hosts. Restoring them to warnings reproduces the
#      exact translation the CI has always shipped.
#
#   2. Failure detection. The upstream script runs without `set -e`, so a failed
#      `make install` is followed by a successful `pip install .` and the script
#      exits 0 while $CONDA_PREFIX/lib holds no libacados.so. This wrapper
#      asserts the artefacts really exist.
#
# Usage: install_acados_linux.sh [BIOPTIM_DIR] [NB_CPU] [INSTALL_PREFIX]

bioptim_dir="${1:-${COCOFEST_ROOT:-$(git rev-parse --show-toplevel)}/.benchmark-deps/bioptim}"
build_jobs="${2:-${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}}"
install_prefix="${3:-${CONDA_PREFIX:-}}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer is intended for Linux." >&2
  exit 1
fi
if [[ -z "$install_prefix" ]]; then
  echo "CONDA_PREFIX must point to the active benchmark environment." >&2
  exit 1
fi

acados_installer="$bioptim_dir/external/acados_install_linux.sh"
if [[ ! -f "$acados_installer" ]]; then
  echo "Missing $acados_installer; clone the pinned bioptim first (see linux_32core_setup.md section 6)." >&2
  exit 1
fi

# Keep only the -Wno-* switches this compiler actually understands, so the same
# script stays valid on GCC 13, GCC 15 and Clang.
compat_flags=()
probe_source="$(mktemp --suffix=.c)"
trap 'rm -f "$probe_source"' EXIT
echo 'int main(void) { return 0; }' > "$probe_source"
for flag in -Wno-incompatible-pointer-types \
            -Wno-implicit-function-declaration \
            -Wno-int-conversion \
            -Wno-return-mismatch; do
  if "${CC:-cc}" "$flag" -Werror -c "$probe_source" -o /dev/null >/dev/null 2>&1; then
    compat_flags+=("$flag")
  fi
done

if (( ${#compat_flags[@]} > 0 )); then
  echo "ACADOS toolchain compatibility flags: ${compat_flags[*]}"
  export CFLAGS="${CFLAGS:+$CFLAGS }${compat_flags[*]}"
fi

# The upstream script wipes acados/build, so CMake performs a fresh configure
# and seeds CMAKE_C_FLAGS from $CFLAGS.
export CMAKE_BUILD_PARALLEL_LEVEL="$build_jobs"
bash "$acados_installer" "$build_jobs" "$install_prefix"

# ACADOS renders its generated C through tera. The upstream installer does not
# ship it, and acados_template downloads an unpinned build on first use when it
# is missing, so install the pinned release here instead.
if [[ -x "$install_prefix/bin/t_renderer" ]]; then
  echo "t_renderer already present at $install_prefix/bin/t_renderer"
else
  bash "$(dirname "${BASH_SOURCE[0]}")/install_acados_tera_renderer.sh" "$install_prefix"
fi

missing=()
for artefact in lib/libacados.so lib/libhpipm.so lib/libblasfeo.so \
                lib/link_libs.json lib/git_commit_hash bin/t_renderer; do
  [[ -e "$install_prefix/$artefact" ]] || missing+=("$artefact")
done
if (( ${#missing[@]} > 0 )); then
  echo "ACADOS install incomplete under $install_prefix:" >&2
  printf '  missing %s\n' "${missing[@]}" >&2
  echo "The upstream installer does not use 'set -e'; re-read its output for the first compiler error." >&2
  exit 1
fi

python - <<'PY_CHECK'
import acados_template
from acados_template import get_tera

print("acados_template", acados_template.__file__)
print("t_renderer", get_tera())
PY_CHECK
echo "ACADOS installed under $install_prefix"
