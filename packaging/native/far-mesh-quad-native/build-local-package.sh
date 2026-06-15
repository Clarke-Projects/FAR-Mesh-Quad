#!/usr/bin/env bash
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${PKG_ROOT}/../../.." && pwd)"
VERSION="0.1.0"
STAGING_ROOT="${PKG_ROOT}/_staging"
STAGING="${STAGING_ROOT}/far-mesh-quad-native-${VERSION}"

rm -rf "${STAGING_ROOT}"
mkdir -p "${STAGING}"

copy_dir() {
  local name="$1"
  if [[ -d "${PROJECT_ROOT}/${name}" ]]; then
    echo "Copying ${name}"
    cp -a "${PROJECT_ROOT}/${name}" "${STAGING}/"
  fi
}

copy_file() {
  local name="$1"
  if [[ -f "${PROJECT_ROOT}/${name}" ]]; then
    echo "Copying ${name}"
    cp -a "${PROJECT_ROOT}/${name}" "${STAGING}/"
  fi
}

copy_dir far_mesh
copy_dir bin
copy_dir quadwild-bimdf
copy_dir scripts
copy_dir requirements

copy_file README.md
copy_file LICENSE
copy_file pyproject.toml

find "${STAGING}" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "${STAGING}" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
find "${STAGING}" -type d \( -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} +

echo
echo "== Native staging size =="
du -sh "${STAGING}"

cp "${PKG_ROOT}/far-mesh-quad-native" "${PKG_ROOT}/far-mesh-quad-native.desktop" "${PKG_ROOT}/PKGBUILD" "${PKG_ROOT}/" >/dev/null 2>&1 || true

rm -f "${PKG_ROOT}/far-mesh-quad-native-${VERSION}.tar.zst"
tar --zstd -cf "${PKG_ROOT}/far-mesh-quad-native-${VERSION}.tar.zst" -C "${STAGING_ROOT}" "far-mesh-quad-native-${VERSION}"

cd "${PKG_ROOT}"
makepkg -Cf

echo
echo "Package build complete:"
ls -lh "${PKG_ROOT}"/far-mesh-quad-native-*.pkg.tar.zst
