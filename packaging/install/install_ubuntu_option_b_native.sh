#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WGPU_BUILDER="${PROJECT_ROOT}/packaging/native/debian/wgpu-stack/build-wgpu-stack-debs.sh"
WGPU_DEB_DIR="${PROJECT_ROOT}/packaging/native/debian/wgpu-stack/_deb"

APP_BUILDER="${PROJECT_ROOT}/packaging/native/debian/far-mesh-quad-native/build-local-deb.sh"
APP_DEB_DIR="${PROJECT_ROOT}/packaging/native/debian/far-mesh-quad-native/_deb"
APP_DEB="${APP_DEB_DIR}/far-mesh-quad-native_0.1.4-1_amd64.deb"

YES=0
SKIP_APT_UPDATE=0
SKIP_BUILD=0
SKIP_INSTALL=0
SKIP_VALIDATION=0
LAUNCH_AFTER=0

log() { printf '\033[1;34m[far-mesh-ubuntu-native]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[far-mesh-ubuntu-native:warning]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[far-mesh-ubuntu-native:error]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<USAGE
Usage: $0 [options]

Builds and installs FAR MESH Quad using the Ubuntu native Option B path:
  - no venv
  - no pip
  - system /usr/bin/python3
  - apt dependencies
  - FAR MESH-owned .deb packages for WGPU/pygfx stack
  - /opt/far-mesh-quad-native runtime
  - /usr/bin/far-mesh-quad-native launcher

Options:
  -y, --yes              Non-interactive apt operations.
  --skip-apt-update      Do not run apt-get update.
  --skip-build           Do not rebuild .deb packages.
  --skip-install         Build only, do not install.
  --skip-validation      Do not run installed-root validation.
  --launch               Launch FAR MESH Quad after install.
  -h, --help             Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes) YES=1; shift ;;
        --skip-apt-update) SKIP_APT_UPDATE=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        --skip-validation) SKIP_VALIDATION=1; shift ;;
        --launch) LAUNCH_AFTER=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

apt_yes_args=()
if [[ "$YES" -eq 1 ]]; then
    apt_yes_args=(-y)
fi

require_ubuntu() {
    [[ -r /etc/os-release ]] || die "/etc/os-release not found"

    # shellcheck disable=SC1091
    source /etc/os-release

    [[ "${ID:-}" == "ubuntu" ]] || die "This installer currently targets Ubuntu only. Found ID='${ID:-unknown}'."

    log "Ubuntu detected: ${PRETTY_NAME:-unknown}"
    log "Python: $(python3 --version 2>&1)"
}

install_apt_dependencies() {
    log "Installing Ubuntu native dependencies"

    if [[ "$SKIP_APT_UPDATE" -eq 0 ]]; then
        sudo apt-get update
    fi

    sudo apt-get install "${apt_yes_args[@]}" \
      git \
      ca-certificates \
      curl \
      build-essential \
      cmake \
      ninja-build \
      pkg-config \
      tar \
      zstd \
      dpkg-dev \
      python3 \
      python3-dev \
      python3-numpy \
      python3-scipy \
      python3-sklearn \
      python3-pandas \
      python3-matplotlib \
      python3-yaml \
      python3-psutil \
      python3-requests \
      python3-pil \
      python3-trimesh \
      python3-open3d \
      python3-pyvista \
      python3-vtk9 \
      python3-pyside6.qtcore \
      python3-pyside6.qtgui \
      python3-pyside6.qtwidgets \
      python3-cffi \
      python3-hsluv \
      python3-freetype \
      python3-uharfbuzz \
      libgl1 \
      libegl1 \
      libvulkan1 \
      mesa-vulkan-drivers \
      vulkan-tools \
      libxkbcommon-x11-0 \
      libxcb-cursor0 \
      libxcb-xinerama0 \
      libxcb-randr0 \
      libxcb-keysyms1 \
      libxcb-icccm4 \
      libxcb-image0 \
      libxcb-render-util0 \
      libxcb-shape0 \
      desktop-file-utils \
      shared-mime-info \
      hicolor-icon-theme
}

build_packages() {
    [[ -x "$WGPU_BUILDER" ]] || die "Missing WGPU builder: $WGPU_BUILDER"
    [[ -x "$APP_BUILDER" ]] || die "Missing app builder: $APP_BUILDER"

    log "Building WGPU dependency .deb packages"
    bash "$WGPU_BUILDER"

    log "Building FAR MESH Quad app .deb package"
    bash "$APP_BUILDER"
}

install_packages() {
    [[ -d "$WGPU_DEB_DIR" ]] || die "Missing WGPU .deb output directory: $WGPU_DEB_DIR"
    [[ -f "$APP_DEB" ]] || die "Missing FAR MESH app .deb: $APP_DEB"

    log "Installing FAR MESH-owned WGPU dependency .deb packages"
    sudo apt-get install "${apt_yes_args[@]}" "${WGPU_DEB_DIR}"/*.deb

    log "Installing FAR MESH Quad native app .deb"
    sudo apt-get install "${apt_yes_args[@]}" "$APP_DEB"
}

validate_installed_runtime() {
    log "Validating installed /opt runtime"

    [[ -x /usr/bin/far-mesh-quad-native ]] || die "Missing launcher: /usr/bin/far-mesh-quad-native"
    [[ -d /opt/far-mesh-quad-native ]] || die "Missing app root: /opt/far-mesh-quad-native"
    [[ -f /usr/share/applications/far-mesh-quad-native.desktop ]] || die "Missing desktop entry"

    unset PYTHONPATH || true

    cd /opt/far-mesh-quad-native

    /usr/bin/python3 - <<'PY'
import importlib
import far_mesh

print("OK far_mesh import from installed app root", getattr(far_mesh, "__file__", ""))

mods = [
    "PySide6", "numpy", "scipy", "sklearn", "pandas", "matplotlib",
    "yaml", "psutil", "requests", "PIL", "trimesh", "open3d",
    "pyvista", "vtk", "rendercanvas", "wgpu", "pylinalg", "pygfx",
]

for mod in mods:
    m = importlib.import_module(mod)
    print(f"{mod:14} OK  {getattr(m, '__file__', '')}")

from far_mesh.core.bore import RebuildResult, RebuildTargetPatch
from far_mesh.core.bore.selection.region_select import select_region_data
from far_mesh.core.bore.selection.mesh_realization import build_opening_evidence_ledger_from_arrays
from far_mesh.core.bore.recognition.recognition import recognize_bore_region_selection
from far_mesh.core.bore.recognition.recognition_component_engine import component_engine_feature_candidates
from far_mesh.core.bore.rebuild.rebuild import delete_and_rebuild_candidate_region
from far_mesh.core.bore.rebuild.rebuild_inventory import rebuild_refactor_inventory_v177k

print("OK folderized Bore public API", RebuildResult.__name__, RebuildTargetPatch.__name__)
print("OK folderized Bore selection", select_region_data.__name__)
print("OK folderized Bore mesh realization", build_opening_evidence_ledger_from_arrays.__name__)
print("OK folderized Bore recognition", recognize_bore_region_selection.__name__, component_engine_feature_candidates.__name__)
print("OK folderized Bore rebuild", delete_and_rebuild_candidate_region.__name__)
print("OK folderized Bore inventory", rebuild_refactor_inventory_v177k()["checkpoint"])
PY

    command -v far-mesh-quad-native
    grep -E 'Name=|Exec=|Icon=|StartupWMClass=' /usr/share/applications/far-mesh-quad-native.desktop

    log "Validation complete. Launch with: /usr/bin/far-mesh-quad-native"
}

main() {
    require_ubuntu
    install_apt_dependencies

    if [[ "$SKIP_BUILD" -eq 0 ]]; then
        build_packages
    fi

    if [[ "$SKIP_INSTALL" -eq 0 ]]; then
        install_packages
    fi

    if [[ "$SKIP_VALIDATION" -eq 0 ]]; then
        validate_installed_runtime
    fi

    if [[ "$LAUNCH_AFTER" -eq 1 ]]; then
        log "Launching FAR MESH Quad"
        far-mesh-quad-native
    fi
}

main "$@"
