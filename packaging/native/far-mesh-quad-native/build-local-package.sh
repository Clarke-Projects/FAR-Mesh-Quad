#!/usr/bin/env bash
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${PKG_ROOT}/../../.." && pwd)"
VERSION="0.1.2"
STAGING_ROOT="${PKG_ROOT}/_staging"
STAGING="${STAGING_ROOT}/far-mesh-quad-native-${VERSION}"
MODE="makepkg"
CLEAN=1

usage() {
  cat <<USAGE
Usage: $0 [--stage-only|--makepkg] [--no-clean]

Creates the curated Option B native runtime staging tree.

Modes:
  --stage-only   Create _staging and the source tar.zst archive, then stop.
  --makepkg      Create staging/archive and run makepkg -Cf. Default.

Options:
  --no-clean     Reuse existing _staging instead of deleting it first.
  -h, --help     Show this help.

This script intentionally copies only the runtime payload manifest. It must not
copy development virtual environments, pkg/src build leftovers, __pycache__, or
local experiment folders into the native package payload.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage-only|--skip-makepkg) MODE="stage-only"; shift ;;
    --makepkg) MODE="makepkg"; shift ;;
    --no-clean) CLEAN=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

copy_dir() {
  local name="$1"
  if [[ -d "${PROJECT_ROOT}/${name}" ]]; then
    echo "Copying ${name}"
    cp -a "${PROJECT_ROOT}/${name}" "${STAGING}/"
  else
    echo "Missing optional directory: ${name}" >&2
  fi
}

copy_file() {
  local name="$1"
  if [[ -f "${PROJECT_ROOT}/${name}" ]]; then
    echo "Copying ${name}"
    cp -a "${PROJECT_ROOT}/${name}" "${STAGING}/"
  else
    echo "Missing optional file: ${name}" >&2
  fi
}

require_path() {
  local rel="$1"
  if [[ ! -e "${PROJECT_ROOT}/${rel}" ]]; then
    echo "Required payload path missing: ${rel}" >&2
    exit 1
  fi
}

require_path far_mesh
require_path bin
require_path "bin/Instant Meshes"
require_path bin/quadwild
require_path bin/config
require_path quadwild-bimdf
require_path scripts
require_path LICENSE

if [[ "$CLEAN" -eq 1 ]]; then
  rm -rf "${STAGING_ROOT}"
fi
mkdir -p "${STAGING}"

# Curated native runtime payload. Do not replace this with a project-root copy.
copy_dir far_mesh
copy_dir bin
copy_dir quadwild-bimdf
copy_dir scripts
copy_dir requirements

copy_file README.md
copy_file LICENSE
copy_file pyproject.toml

# Runtime and package-provenance license bundle. In the source tree this is
# normally under bin/License, which is already copied with bin/. Keep this
# explicit check because licenses are part of the Option B package contract.
if [[ ! -d "${STAGING}/bin/License" ]]; then
  echo "Warning: ${STAGING}/bin/License is missing; package license payload may be incomplete." >&2
fi

# Remove Python/build cache material from the app payload.
find "${STAGING}" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "${STAGING}" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
find "${STAGING}" -type d \( -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} +
find "${STAGING}" -type d \( -name ".git" -o -name ".hg" -o -name ".svn" \) -prune -exec rm -rf {} +

# Known development leftovers that are not required at runtime. These may not
# exist in a clean public clone, but removing them here keeps dev packaging safe.
rm -rf "${STAGING}/scripts/test_cleanup_archive" 2>/dev/null || true
find "${STAGING}/far_mesh" -maxdepth 1 -type f \( -name "*.before_*" -o -name "*.bak" -o -name "*.orig" \) -delete 2>/dev/null || true

echo
echo "== Native staging size =="
du -sh "${STAGING}"

# Synchronize package-side icon and license sources for makepkg. Stage-only users
# can ignore these copies, but keeping them current makes the package target match
# the runtime staging payload.
if [[ -f "${PROJECT_ROOT}/far_mesh/gui/assets/icons/Icon_FAR_Mesh_Quad.png" ]]; then
  cp -a "${PROJECT_ROOT}/far_mesh/gui/assets/icons/Icon_FAR_Mesh_Quad.png" "${PKG_ROOT}/far-mesh-quad-native.png"
else
  echo "Warning: FAR MESH icon source missing; desktop package icon may be incomplete." >&2
fi

if [[ -d "${PROJECT_ROOT}/bin/License" ]]; then
  cp -a "${PROJECT_ROOT}/bin/License"/LICENSE_*.txt "${PKG_ROOT}/" 2>/dev/null || true
fi

rm -f "${PKG_ROOT}/far-mesh-quad-native-${VERSION}.tar.zst"
tar --zstd -cf "${PKG_ROOT}/far-mesh-quad-native-${VERSION}.tar.zst" -C "${STAGING_ROOT}" "far-mesh-quad-native-${VERSION}"

echo
echo "Staging archive ready: ${PKG_ROOT}/far-mesh-quad-native-${VERSION}.tar.zst"

if [[ "$MODE" == "stage-only" ]]; then
  echo "Stage-only mode complete; makepkg was not run."
  exit 0
fi

cd "${PKG_ROOT}"
makepkg -Cf

echo
echo "Package build complete:"
ls -lh "${PKG_ROOT}"/far-mesh-quad-native-*.pkg.tar.zst
