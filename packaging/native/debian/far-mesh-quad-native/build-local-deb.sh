#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

DEB_ROOT="${PROJECT_ROOT}/packaging/native/debian/far-mesh-quad-native"
OUT_DIR="${DEB_ROOT}/_deb"
WORK_DIR="${DEB_ROOT}/_build"

PKG_NAME="far-mesh-quad-native"
PKG_VERSION="0.1.4.2-1"
APP_ROOT="/opt/far-mesh-quad-native"
BIN_PATH="/usr/bin/far-mesh-quad-native"

log() { printf '\033[1;34m[far-mesh-deb]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[far-mesh-deb:warning]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[far-mesh-deb:error]\033[0m %s\n' "$*" >&2; exit 1; }

require_file() {
    [[ -f "$1" ]] || die "Missing required file: $1"
}

require_dir() {
    [[ -d "$1" ]] || die "Missing required directory: $1"
}

copy_dir_if_present() {
    local src="$1"
    local dst="$2"

    if [[ -d "$src" ]]; then
        log "Copying $(basename "$src")"
        mkdir -p "$(dirname "$dst")"
        cp -a "$src" "$dst"
    else
        warn "Missing optional directory: $src"
    fi
}

copy_file_if_present() {
    local src="$1"
    local dst="$2"

    if [[ -f "$src" ]]; then
        log "Copying $(basename "$src")"
        mkdir -p "$(dirname "$dst")"
        cp -a "$src" "$dst"
    else
        warn "Missing optional file: $src"
    fi
}

validate_source_tree() {
    log "Validating source tree with Ubuntu /usr/bin/python3"

    cd "$PROJECT_ROOT"

    /usr/bin/python3 -m compileall far_mesh >/dev/null

    /usr/bin/python3 - <<'PY'
from far_mesh.core.bore import RebuildResult, RebuildTargetPatch
from far_mesh.core.bore.selection.region_select import select_region_data
from far_mesh.core.bore.selection.mesh_realization import build_opening_evidence_ledger_from_arrays
from far_mesh.core.bore.recognition.recognition import recognize_bore_region_selection
from far_mesh.core.bore.recognition.recognition_component_engine import component_engine_feature_candidates
from far_mesh.core.bore.rebuild.rebuild import delete_and_rebuild_candidate_region
from far_mesh.core.bore.rebuild.rebuild_inventory import rebuild_refactor_inventory_v177k

print("OK source Bore public API", RebuildResult.__name__, RebuildTargetPatch.__name__)
print("OK source Bore selection", select_region_data.__name__)
print("OK source Bore mesh realization", build_opening_evidence_ledger_from_arrays.__name__)
print("OK source Bore recognition", recognize_bore_region_selection.__name__, component_engine_feature_candidates.__name__)
print("OK source Bore rebuild", delete_and_rebuild_candidate_region.__name__)
print("OK source Bore inventory", rebuild_refactor_inventory_v177k()["checkpoint"])
PY
}

install_runtime_payload() {
    local pkg_root="$1"
    local app_dst="${pkg_root}${APP_ROOT}"

    log "Installing curated runtime payload into ${APP_ROOT}"

    require_dir "${PROJECT_ROOT}/far_mesh"

    rm -rf "$app_dst"
    mkdir -p "$app_dst"

    copy_dir_if_present "${PROJECT_ROOT}/far_mesh" "${app_dst}/far_mesh"
    copy_dir_if_present "${PROJECT_ROOT}/bin" "${app_dst}/bin"
    copy_dir_if_present "${PROJECT_ROOT}/quadwild-bimdf" "${app_dst}/quadwild-bimdf"
    copy_dir_if_present "${PROJECT_ROOT}/scripts" "${app_dst}/scripts"
    copy_dir_if_present "${PROJECT_ROOT}/requirements" "${app_dst}/requirements"

    copy_file_if_present "${PROJECT_ROOT}/README.md" "${app_dst}/README.md"
    copy_file_if_present "${PROJECT_ROOT}/LICENSE" "${app_dst}/LICENSE"
    copy_file_if_present "${PROJECT_ROOT}/pyproject.toml" "${app_dst}/pyproject.toml"

    find "$app_dst" -type d -name '__pycache__' -prune -exec rm -rf {} +
    find "$app_dst" -type f -name '*.pyc' -delete
}

install_exec_wrapper() {
    local pkg_root="$1"
    local wrapper_tmp="${WORK_DIR}/_wrapper"
    local tmp_c="${wrapper_tmp}/far-mesh-quad-native-wrapper.c"
    local tmp_bin="${wrapper_tmp}/far-mesh-quad-native"

    log "Compiling Ubuntu exec wrapper"
    rm -rf "$wrapper_tmp"
    mkdir -p "$wrapper_tmp"

    cat > "$tmp_c" <<'EOF'
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    const char *app_root = "/opt/far-mesh-quad-native";
    const char *python = "/usr/bin/python3";

    if (chdir(app_root) != 0) {
        fprintf(stderr, "far-mesh-quad-native: failed to chdir to %s: %s\n", app_root, strerror(errno));
        return 1;
    }

    char **newargv = calloc((size_t)argc + 3, sizeof(char *));
    if (!newargv) {
        fprintf(stderr, "far-mesh-quad-native: calloc failed\n");
        return 1;
    }

    /*
     * Keep argv[0] as the real Python executable.
     *
     * If argv[0] is "far-mesh-quad-native", Python multiprocessing and
     * FAR MESH background task routing can treat the wrapper as the Python
     * executable and relaunch the full GUI when starting worker tasks.
     */
    newargv[0] = "/usr/bin/python3";
    newargv[1] = "-m";
    newargv[2] = "far_mesh.main";

    for (int i = 1; i < argc; ++i) {
        newargv[i + 2] = argv[i];
    }

    newargv[argc + 2] = NULL;

    execv(python, newargv);

    fprintf(stderr, "far-mesh-quad-native: failed to exec %s: %s\n", python, strerror(errno));
    free(newargv);
    return 1;
}
EOF

    gcc -O2 -Wall -Wextra -o "$tmp_bin" "$tmp_c"

    install -Dm755 "$tmp_bin" "${pkg_root}${BIN_PATH}"
}

install_desktop_assets() {
    local pkg_root="$1"

    log "Installing desktop entry"

    mkdir -p "${pkg_root}/usr/share/applications"

    cat > "${pkg_root}/usr/share/applications/far-mesh-quad-native.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=FAR MESH Quad
Comment=Topology-aware mesh processing workstation
Exec=/usr/bin/far-mesh-quad-native
Icon=far-mesh-quad-native
Terminal=false
Categories=Graphics;Engineering;Science;
StartupWMClass=far-mesh-quad-native
EOF

    local icon_src=""
    for candidate in \
        "${PROJECT_ROOT}/packaging/native/far-mesh-quad-native/far-mesh-quad-native.png" \
        "${PROJECT_ROOT}/far_mesh/gui/assets/logos/far-mesh-quad-native.png" \
        "${PROJECT_ROOT}/far_mesh/gui/assets/logos/far_mesh_quad.png" \
        "${PROJECT_ROOT}/docs/images/Pockets.png"
    do
        if [[ -f "$candidate" ]]; then
            icon_src="$candidate"
            break
        fi
    done

    if [[ -n "$icon_src" ]]; then
        log "Installing icon from ${icon_src}"
        install -Dm644 \
            "$icon_src" \
            "${pkg_root}/usr/share/icons/hicolor/256x256/apps/far-mesh-quad-native.png"
    else
        warn "No icon asset found; desktop entry will still be installed."
    fi
}

install_license_bundle() {
    local pkg_root="$1"

    if [[ -d "${pkg_root}${APP_ROOT}/bin/License" ]]; then
        log "Installing license bundle"
        mkdir -p "${pkg_root}/usr/share/licenses/far-mesh-quad-native"
        cp -a "${pkg_root}${APP_ROOT}/bin/License/." \
            "${pkg_root}/usr/share/licenses/far-mesh-quad-native/"
    else
        warn "No license bundle found at staged bin/License"
    fi
}

write_maintainer_scripts() {
    local pkg_root="$1"
    local debian_dir="${pkg_root}/DEBIAN"

    cat > "${debian_dir}/postinst" <<'EOF'
#!/usr/bin/env bash
set -e

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database /usr/share/applications || true
fi

if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor || true
fi

exit 0
EOF

    cat > "${debian_dir}/postrm" <<'EOF'
#!/usr/bin/env bash
set -e

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database /usr/share/applications || true
fi

if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor || true
fi

exit 0
EOF

    chmod 755 "${debian_dir}/postinst" "${debian_dir}/postrm"
}

write_control() {
    local pkg_root="$1"
    local debian_dir="${pkg_root}/DEBIAN"

    mkdir -p "$debian_dir"

    cat > "${debian_dir}/control" <<EOF
Package: ${PKG_NAME}
Version: ${PKG_VERSION}
Section: graphics
Priority: optional
Architecture: amd64
Maintainer: Mathias Clarke <noreply@example.invalid>
Depends: python3 (>= 3.14), python3-numpy, python3-scipy, python3-sklearn, python3-pandas, python3-matplotlib, python3-yaml, python3-psutil, python3-requests, python3-pil, python3-trimesh, python3-open3d, python3-pyvista, python3-vtk9, python3-pyside6.qtcore, python3-pyside6.qtgui, python3-pyside6.qtwidgets, python3-cffi, python3-hsluv, python3-freetype, python3-uharfbuzz, python3-rendercanvas-farmesh, python3-wgpu-farmesh, python3-pylinalg-farmesh, python3-pygfx-farmesh, libgl1, libegl1, libvulkan1, mesa-vulkan-drivers, libxkbcommon-x11-0, libxcb-cursor0, libxcb-xinerama0, libxcb-randr0, libxcb-keysyms1, libxcb-icccm4, libxcb-image0, libxcb-render-util0, libxcb-shape0, desktop-file-utils, shared-mime-info, hicolor-icon-theme
Description: FAR MESH Quad native mesh processing workstation
 FAR MESH Quad is a topology-aware mesh processing workstation.
 This package installs the curated native Option B runtime payload under
 /opt/far-mesh-quad-native and the launcher at /usr/bin/far-mesh-quad-native.
EOF
}

validate_package_root() {
    local pkg_root="$1"

    log "Validating package root payload"

    cd "${pkg_root}${APP_ROOT}"

    /usr/bin/python3 - <<'PY'
import far_mesh
print("OK package-root far_mesh import", getattr(far_mesh, "__file__", ""))

from far_mesh.core.bore import RebuildResult, RebuildTargetPatch
from far_mesh.core.bore.selection.region_select import select_region_data
from far_mesh.core.bore.selection.mesh_realization import build_opening_evidence_ledger_from_arrays
from far_mesh.core.bore.recognition.recognition import recognize_bore_region_selection
from far_mesh.core.bore.recognition.recognition_component_engine import component_engine_feature_candidates
from far_mesh.core.bore.rebuild.rebuild import delete_and_rebuild_candidate_region
from far_mesh.core.bore.rebuild.rebuild_inventory import rebuild_refactor_inventory_v177k

print("OK package-root Bore public API", RebuildResult.__name__, RebuildTargetPatch.__name__)
print("OK package-root Bore selection", select_region_data.__name__)
print("OK package-root Bore mesh realization", build_opening_evidence_ledger_from_arrays.__name__)
print("OK package-root Bore recognition", recognize_bore_region_selection.__name__, component_engine_feature_candidates.__name__)
print("OK package-root Bore rebuild", delete_and_rebuild_candidate_region.__name__)
print("OK package-root Bore inventory", rebuild_refactor_inventory_v177k()["checkpoint"])
PY
}

build_deb() {
    local pkg_root="${WORK_DIR}/${PKG_NAME}"

    rm -rf "$pkg_root"
    mkdir -p "$pkg_root" "$OUT_DIR" "$WORK_DIR"

    write_control "$pkg_root"
    write_maintainer_scripts "$pkg_root"
    install_runtime_payload "$pkg_root"
    install_exec_wrapper "$pkg_root"
    install_desktop_assets "$pkg_root"
    install_license_bundle "$pkg_root"
    validate_package_root "$pkg_root"

    log "Building ${PKG_NAME}_${PKG_VERSION}_amd64.deb"
    dpkg-deb --build --root-owner-group \
        "$pkg_root" \
        "${OUT_DIR}/${PKG_NAME}_${PKG_VERSION}_amd64.deb" >/dev/null

    log "Built:"
    ls -lh "${OUT_DIR}/${PKG_NAME}_${PKG_VERSION}_amd64.deb"
}

main() {
    command -v dpkg-deb >/dev/null 2>&1 || die "dpkg-deb not found. Install dpkg-dev."
    command -v gcc >/dev/null 2>&1 || die "gcc not found. Install build-essential."

    validate_source_tree
    build_deb
}

main "$@"
