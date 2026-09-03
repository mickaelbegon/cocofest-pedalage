#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "The compiled libMad benchmark runtime requires Linux." >&2
  exit 1
fi

# Julia 1.12's generated sysimage references a symbol version introduced in
# GCC 13. Check the library the linker will actually resolve, rather than the
# compiler's marketing version (which is unreliable when cc is Clang).
libgcc_path="$(cc -print-file-name=libgcc_s.so.1 2>/dev/null || true)"
if [[ -z "$libgcc_path" || "$libgcc_path" == "libgcc_s.so.1" || ! -f "$libgcc_path" ]]; then
  echo "Unable to resolve libgcc_s.so.1 through cc." >&2
  exit 1
fi

# Read the ELF version-definition table when binutils exposes it, and fall back
# to the raw string scan otherwise. The result is materialised in a variable
# before being matched: piping straight into `grep -q` makes grep exit on the
# first hit, which kills the producer with SIGPIPE and -- under `pipefail` --
# turns a successful check into a spurious failure.
libgcc_symbol_versions="$(
  {
    readelf --version-info "$libgcc_path" 2>/dev/null ||
      strings -a "$libgcc_path" 2>/dev/null ||
      true
  } | grep -oE 'GCC_[0-9]+(\.[0-9]+)*' | sort -u
)"

if ! grep -qxF 'GCC_13.0.0' <<<"$libgcc_symbol_versions"; then
  echo "$libgcc_path does not export GCC_13.0.0, required by the Julia 1.12 runtime." >&2
  echo "Versions found: $(tr '\n' ' ' <<<"$libgcc_symbol_versions")" >&2
  echo "Use ubuntu-24.04 (the workflow default) or a newer x86-64 host." >&2
  exit 1
fi

echo "Compatible MadNLP/MUMPS host runtime: $libgcc_path"
