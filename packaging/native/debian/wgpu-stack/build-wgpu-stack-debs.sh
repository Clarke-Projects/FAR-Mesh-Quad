#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/packaging/native/wgpu-stack/packages"
OUT_DIR="${PROJECT_ROOT}/packaging/native/debian/wgpu-stack/_deb"
WORK_DIR="${PROJECT_ROOT}/packaging/native/debian/wgpu-stack/_build"

log() { printf '\033[1;34m[wgpu-deb]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[wgpu-deb:error]\033[0m %s\n' "$*" >&2; exit 1; }

require_file() {
    [[ -f "$1" ]] || die "Missing required file: $1"
}

relocate_arch_site_to_debian_dist() {
    local pkg_root="$1"
    local arch_site="${pkg_root}/usr/lib/python3.14/site-packages"
    local deb_site="${pkg_root}/usr/lib/python3/dist-packages"

    if [[ -d "$arch_site" ]]; then
        log "Relocating Python payload to Debian dist-packages"
        mkdir -p "$deb_site"

        shopt -s dotglob nullglob
        mv "$arch_site"/* "$deb_site"/
        shopt -u dotglob nullglob

        rm -rf "${pkg_root}/usr/lib/python3.14"
    fi
}

build_pkg() {
    local arch_pkg="$1"
    local deb_name="$2"
    local version="$3"
    local arch="$4"
    local depends="$5"
    local description="$6"

    local pkg_root="${WORK_DIR}/${deb_name}"
    local debian_dir="${pkg_root}/DEBIAN"

    rm -rf "$pkg_root"
    mkdir -p "$pkg_root" "$debian_dir"

    log "Extracting ${arch_pkg}"
    tar --zstd -xf "$arch_pkg" -C "$pkg_root"

    rm -f \
        "${pkg_root}/.BUILDINFO" \
        "${pkg_root}/.MTREE" \
        "${pkg_root}/.PKGINFO"

    find "$pkg_root" -type d -name '__pycache__' -prune -exec rm -rf {} +

    relocate_arch_site_to_debian_dist "$pkg_root"

    cat > "${debian_dir}/control" <<EOF
Package: ${deb_name}
Version: ${version}
Section: python
Priority: optional
Architecture: ${arch}
Maintainer: Mathias Clarke <noreply@example.invalid>
Depends: ${depends}
Description: ${description}
 FAR MESH Quad native Ubuntu Option B dependency package.
 This package installs the WGPU/pygfx runtime component into
 /usr/lib/python3/dist-packages for system-Python operation.
EOF

    dpkg-deb --build --root-owner-group "$pkg_root" "${OUT_DIR}/${deb_name}_${version}_${arch}.deb" >/dev/null
    log "Built ${OUT_DIR}/${deb_name}_${version}_${arch}.deb"
}

main() {
    command -v dpkg-deb >/dev/null 2>&1 || die "dpkg-deb not found. Install dpkg-dev."

    mkdir -p "$OUT_DIR" "$WORK_DIR"
    rm -f "${OUT_DIR}"/*.deb 2>/dev/null || true

    require_file "${SRC_DIR}/python-rendercanvas-2.6.3-1-any.pkg.tar.zst"
    require_file "${SRC_DIR}/python-wgpu-0.31.0-1-any.pkg.tar.zst"
    require_file "${SRC_DIR}/python-pylinalg-0.6.8-1-any.pkg.tar.zst"
    require_file "${SRC_DIR}/python-pygfx-0.16.0-1-any.pkg.tar.zst"

    build_pkg \
        "${SRC_DIR}/python-rendercanvas-2.6.3-1-any.pkg.tar.zst" \
        "python3-rendercanvas-farmesh" \
        "2.6.3-1" \
        "all" \
        "python3 (>= 3.14), python3-pyside6.qtcore, python3-pyside6.qtgui, python3-pyside6.qtwidgets" \
        "rendercanvas runtime for FAR MESH Quad"

    build_pkg \
        "${SRC_DIR}/python-wgpu-0.31.0-1-any.pkg.tar.zst" \
        "python3-wgpu-farmesh" \
        "0.31.0-1" \
        "amd64" \
        "python3 (>= 3.14), python3-cffi, libvulkan1" \
        "wgpu runtime for FAR MESH Quad including libwgpu native"

    build_pkg \
        "${SRC_DIR}/python-pylinalg-0.6.8-1-any.pkg.tar.zst" \
        "python3-pylinalg-farmesh" \
        "0.6.8-1" \
        "all" \
        "python3 (>= 3.14), python3-numpy" \
        "pylinalg runtime for FAR MESH Quad"

    build_pkg \
        "${SRC_DIR}/python-pygfx-0.16.0-1-any.pkg.tar.zst" \
        "python3-pygfx-farmesh" \
        "0.16.0-1" \
        "all" \
        "python3 (>= 3.14), python3-numpy, python3-hsluv, python3-freetype, python3-uharfbuzz, python3-rendercanvas-farmesh, python3-wgpu-farmesh, python3-pylinalg-farmesh" \
        "pygfx runtime for FAR MESH Quad"

    log "Done. Packages:"
    ls -lh "${OUT_DIR}"/*.deb
}

main "$@"
