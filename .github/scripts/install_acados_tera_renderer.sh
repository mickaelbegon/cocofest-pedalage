#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <acados-install-prefix>" >&2
  exit 2
fi

install_prefix="$1"
version="${TERA_RENDERER_VERSION:-0.2.0}"
architecture="$(uname -m)"

case "$architecture" in
  x86_64|amd64)
    release_architecture="amd64"
    expected_sha256="1973ddd1b536dcd4a059a22f7259c3b0e6b6c878cbfbe7e9b962eba195f361d9"
    ;;
  aarch64|arm64)
    release_architecture="arm64"
    expected_sha256="a29f563c7f49368b1802864cdf2bad9d50fe0fb7180c9d210c188a30bb23f167"
    ;;
  *)
    echo "Unsupported t_renderer architecture: $architecture" >&2
    exit 1
    ;;
esac

if [[ "$version" != "0.2.0" ]]; then
  echo "No pinned checksum is registered for t_renderer $version." >&2
  exit 1
fi

target="$install_prefix/bin/t_renderer"
temporary_directory="$(mktemp -d)"
trap 'rm -rf "$temporary_directory"' EXIT
download="$temporary_directory/t_renderer"
url="https://github.com/acados/tera_renderer/releases/download/v${version}/t_renderer-v${version}-linux-${release_architecture}"

mkdir -p "$(dirname "$target")"
curl \
  --fail \
  --location \
  --retry 5 \
  --retry-all-errors \
  --retry-delay 2 \
  --output "$download" \
  "$url"
echo "$expected_sha256  $download" | sha256sum -c -
install -m 0755 "$download" "$target"

echo "Installed pinned t_renderer $version ($release_architecture) at $target"
