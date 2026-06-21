#!/usr/bin/env bash
set -Eeuo pipefail

# FAR MESH Quad source-build installer
# First supported target: Arch Linux / CachyOS, Option B native no-venv install.
# This script intentionally installs a curated runtime payload only. It never
# copies the whole development tree into /opt.

APP_ROOT_DEFAULT="/opt/far-mesh-quad-native"
BIN_PATH_DEFAULT="/usr/bin/far-mesh-quad-native"
DESKTOP_PATH_DEFAULT="/usr/share/applications/far-mesh-quad-native.desktop"
ICON_PATH_DEFAULT="/usr/share/icons/hicolor/256x256/apps/far-mesh-quad-native.png"
LICENSE_PATH_DEFAULT="/usr/share/licenses/far-mesh-quad-native"

TARGET="auto"
PROJECT_ROOT=""
APP_ROOT="$APP_ROOT_DEFAULT"
BIN_PATH="$BIN_PATH_DEFAULT"
YES=0
DRY_RUN=0
SKIP_SYSTEM_UPDATE=1
SKIP_AUR=0
SKIP_TRIMESH_PIP=0
SKIP_VENDOR_PACKAGES=0
SKIP_DESKTOP=0
SKIP_VALIDATION=0
LAUNCH_AFTER_INSTALL=0
ALLOW_SYSTEM_PIP=1
ALLOW_AUR_NODEPS=1
FORCE_GCC14_FILE_OVERRIDE=0

log() { printf '\033[1;34m[far-mesh-install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[far-mesh-install:warning]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[far-mesh-install:error]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<USAGE
Usage: $0 [options]

Installs FAR MESH Quad from a cloned source repository using the Option B native
no-venv layout. First target: Arch Linux / CachyOS.

Options:
  --target auto|arch|cachyos   Target distro family. Default: auto.
  --repo PATH                  Source repository root. Default: auto-detect.
  --app-root PATH              Install root. Default: $APP_ROOT_DEFAULT
  --bin-path PATH              Launcher path. Default: $BIN_PATH_DEFAULT
  -y, --yes                    Non-interactive pacman/yay/pip operations.
  --dry-run                    Print actions without executing them.
  --system-update              Run full pacman -Syyuu before dependency install. Default: off.
  --skip-system-update         Compatibility no-op; full system update is already off by default.
  --skip-aur                   Do not install AUR dependencies.
  --skip-trimesh-pip           Do not use sudo pip fallback for trimesh.
  --skip-vendor-packages       Do not install vendored Open3D/WGPU packages.
  --skip-desktop               Do not install .desktop file or icon.
  --skip-validation            Do not run validation probes.
  --launch                     Launch FAR MESH Quad after install.
  --no-system-pip              Forbid sudo pip --break-system-packages.
  --strict-aur-deps            Do not pass --nodeps to yay AUR installs.
  --force-gcc14-file-override  Deprecated; kept for compatibility and ignored unless a bad package payload owns gcc-14.
  -h, --help                   Show this help.

Notes:
  - Run this from the cloned FAR-Mesh-Quad repository, or pass --repo.
  - The script uses sudo for system installation steps.
  - It installs only the curated runtime staging payload, not the full dev tree.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --repo) PROJECT_ROOT="${2:-}"; shift 2 ;;
    --app-root) APP_ROOT="${2:-}"; shift 2 ;;
    --bin-path) BIN_PATH="${2:-}"; shift 2 ;;
    -y|--yes) YES=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --system-update) SKIP_SYSTEM_UPDATE=0; shift ;;
    --skip-system-update) SKIP_SYSTEM_UPDATE=1; shift ;;
    --skip-aur) SKIP_AUR=1; shift ;;
    --skip-trimesh-pip) SKIP_TRIMESH_PIP=1; shift ;;
    --skip-vendor-packages) SKIP_VENDOR_PACKAGES=1; shift ;;
    --skip-desktop) SKIP_DESKTOP=1; shift ;;
    --skip-validation) SKIP_VALIDATION=1; shift ;;
    --launch) LAUNCH_AFTER_INSTALL=1; shift ;;
    --no-system-pip) ALLOW_SYSTEM_PIP=0; shift ;;
    --strict-aur-deps) ALLOW_AUR_NODEPS=0; shift ;;
    --force-gcc14-file-override) FORCE_GCC14_FILE_OVERRIDE=1; warn "--force-gcc14-file-override is deprecated; v4 no longer touches /usr/bin/gcc-14 unless a package payload incorrectly owns it."; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

run_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

have() { command -v "$1" >/dev/null 2>&1; }

sudo_cmd() {
  if [[ "$EUID" -eq 0 ]]; then
    run_cmd "$@"
  else
    run_cmd sudo "$@"
  fi
}

pacman_args=()
yay_args=()
if [[ "$YES" -eq 1 ]]; then
  pacman_args+=(--noconfirm)
  yay_args+=(--noconfirm)
fi

if [[ -z "$PROJECT_ROOT" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
fi

PKG_ROOT="${PROJECT_ROOT}/packaging/native/far-mesh-quad-native"
WGPU_ROOT="${PROJECT_ROOT}/packaging/native/wgpu-stack"
WGPU_PACKAGES_DIR="${WGPU_ROOT}/packages"
OPEN3D_PKG="${PROJECT_ROOT}/packaging/native/prebuilt/python-open3d-1:0.19.0-13-x86_64.pkg.tar.zst"
STAGING="${PKG_ROOT}/_staging/far-mesh-quad-native-0.1.2"
DESKTOP_SRC="${PKG_ROOT}/far-mesh-quad-native.desktop"
WRAPPER_SRC="${PKG_ROOT}/far-mesh-quad-native-wrapper.c"
WRAPPER_BUILD="${PKG_ROOT}/far-mesh-quad-native-wrapper"
ICON_SRC="${PROJECT_ROOT}/far_mesh/gui/assets/icons/Icon_FAR_Mesh_Quad.png"
DESKTOP_PATH="$DESKTOP_PATH_DEFAULT"
ICON_PATH="$ICON_PATH_DEFAULT"
LICENSE_PATH="$LICENSE_PATH_DEFAULT"

confirm_or_die() {
  local message="$1"
  if [[ "$YES" -eq 1 || "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  printf '%s [y/N] ' "$message"
  read -r answer
  case "$answer" in
    y|Y|yes|YES) return 0 ;;
    *) die "Cancelled by user." ;;
  esac
}

read_os_release_value() {
  local key="$1"
  if [[ -r /etc/os-release ]]; then
    awk -F= -v k="$key" '$1 == k { gsub(/^"|"$/, "", $2); print $2 }' /etc/os-release
  fi
}

validate_target() {
  local id id_like
  id="$(read_os_release_value ID || true)"
  id_like="$(read_os_release_value ID_LIKE || true)"
  if [[ "$TARGET" == "auto" ]]; then
    case "${id} ${id_like}" in
      *cachyos*|*arch*) TARGET="arch" ;;
      *) die "Auto-detect found ID='${id}' ID_LIKE='${id_like}'. This installer patch currently supports Arch/CachyOS only." ;;
    esac
  fi
  case "$TARGET" in
    arch|cachyos) : ;;
    *) die "Unsupported target '$TARGET'. First patch supports arch/cachyos only." ;;
  esac
  have pacman || die "pacman not found. This target requires Arch/CachyOS."
}

require_source_tree() {
  local missing=0
  local paths=(
    "far_mesh"
    "bin"
    "bin/Instant Meshes"
    "bin/quadwild"
    "bin/config"
    "quadwild-bimdf"
    "scripts"
    "packaging/native/far-mesh-quad-native/PKGBUILD"
    "packaging/native/far-mesh-quad-native/build-local-package.sh"
    "packaging/native/far-mesh-quad-native/far-mesh-quad-native-wrapper.c"
    "packaging/native/far-mesh-quad-native/far-mesh-quad-native.desktop"
    "packaging/native/prebuilt/python-open3d-1:0.19.0-13-x86_64.pkg.tar.zst"
    "packaging/native/wgpu-stack/packages"
  )
  for rel in "${paths[@]}"; do
    if [[ ! -e "${PROJECT_ROOT}/${rel}" ]]; then
      warn "Missing required source path: ${rel}"
      missing=1
    fi
  done
  [[ "$missing" -eq 0 ]] || die "Source tree is incomplete for Option B install."
}

install_arch_dependencies() {
  log "Installing Arch/CachyOS official dependencies"
  local deps=(
    git base-devel tar zstd gcc python python-pip cmake ninja
    assimp eigen glfw pybind11 libc++ zeromq cppzmq
    pyside6 shiboken6 vtk python-matplotlib python-yaml python-psutil
    python-requests python-pillow blas lapack suitesparse mesa
    vulkan-icd-loader qt6-base qt6-wayland xcb-util-cursor libxkbcommon
    cli11 openvr anari-sdk adios2 libharu gl2ps netcdf paraview-catalyst
    cgns boost proj utf8cpp viskores fast_float embree ospray pdal liblas
    python-scikit-learn python-pandas python-setuptools-scm python-build
    python-wheel python-installer onnxruntime-cpu python-cffi python-pycparser
    python-jinja python-markupsafe python-freetype-py python-hsluv python-uharfbuzz
    desktop-file-utils gtk-update-icon-cache
  )
  if [[ "$SKIP_SYSTEM_UPDATE" -eq 0 ]]; then
    confirm_or_die "Run full system update with pacman -Syyuu before installing dependencies?"
    sudo_cmd pacman -Syyuu "${pacman_args[@]}"
  else
    warn "Skipping full system update by default. Use --system-update on a fresh system when you want pacman -Syyuu."
  fi
  sudo_cmd pacman -S --needed "${pacman_args[@]}" "${deps[@]}"
}

ensure_yay() {
  if have yay; then
    return 0
  fi
  [[ "$SKIP_AUR" -eq 0 ]] || die "yay is not installed and --skip-aur was supplied."
  confirm_or_die "Install yay AUR helper from AUR?"
  local tmp
  tmp="$(mktemp -d)"

  # In dry-run mode, git clone is intentionally not executed, so the target
  # directory will not exist. Print the planned clone/build commands and return
  # without trying to cd into a directory that dry-run did not create.
  if [[ "$DRY_RUN" -eq 1 ]]; then
    run_cmd git clone https://aur.archlinux.org/yay.git "$tmp/yay"
    run_cmd bash -lc "cd '$tmp/yay' && makepkg -si ${pacman_args[*]}"
    return 0
  fi

  run_cmd git clone https://aur.archlinux.org/yay.git "$tmp/yay"
  (cd "$tmp/yay" && run_cmd makepkg -si "${pacman_args[@]}")
  rm -rf "$tmp"
}

install_aur_dependencies() {
  if [[ "$SKIP_AUR" -eq 1 ]]; then
    warn "Skipping AUR dependencies. PyVista/PyVistaQt/Scooby/Nanoflann may be missing."
    return 0
  fi
  ensure_yay
  log "Installing AUR dependencies"
  local aur_deps=(python-scooby python-pyvista python-pyvistaqt python-dash nanoflann)
  local args=(-S --needed)
  if [[ "$ALLOW_AUR_NODEPS" -eq 1 ]]; then
    warn "Using yay --nodeps for known fresh-system dependency conflicts."
    args+=(--nodeps)
  fi
  args+=("${yay_args[@]}")
  run_cmd yay "${args[@]}" "${aur_deps[@]}"
}

package_owns_compiler_aliases() {
  local pkg
  for pkg in "$@"; do
    if pacman -Qlp "$pkg" 2>/dev/null | grep -Eq '/usr/bin/(gcc|g\+\+)-14$'; then
      return 0
    fi
  done
  return 1
}

install_vendored_packages() {
  if [[ "$SKIP_VENDOR_PACKAGES" -eq 1 ]]; then
    warn "Skipping vendored Open3D/WGPU package installation."
    return 0
  fi
  log "Installing vendored Open3D and WGPU package stack"
  [[ -f "$OPEN3D_PKG" ]] || die "Open3D package not found: $OPEN3D_PKG"
  mapfile -t wgpu_pkgs < <(find "$WGPU_PACKAGES_DIR" -maxdepth 1 -type f -name '*.pkg.tar.zst' | sort)
  [[ "${#wgpu_pkgs[@]}" -gt 0 ]] || die "No WGPU package artifacts found in $WGPU_PACKAGES_DIR"

  local all_vendor_pkgs=("$OPEN3D_PKG" "${wgpu_pkgs[@]}")

  # v4 rule: do not inspect, remove, move, or recreate /usr/bin/gcc-14 or
  # /usr/bin/g++-14. The current preserved Open3D package only owns
  # /usr/bin/open3d, not compiler aliases. If a future/bad package payload does
  # own compiler aliases, stop and ask for the package to be repacked instead of
  # mutating system compiler files.
  if [[ "$DRY_RUN" -eq 1 ]]; then
    sudo_cmd pacman -U --needed "${pacman_args[@]}" "${all_vendor_pkgs[@]}"
    warn "Dry-run: vendored package install would run without gcc-14/g++-14 mutation."
    return 0
  fi

  if package_owns_compiler_aliases "${all_vendor_pkgs[@]}"; then
    die "A vendored package payload owns /usr/bin/gcc-14 or /usr/bin/g++-14. Refusing to mutate system compiler files; rebuild/repack that package instead."
  fi

  sudo_cmd pacman -U --needed "${pacman_args[@]}" "${all_vendor_pkgs[@]}"
}

ensure_trimesh() {
  if /usr/bin/python - <<'PY' >/dev/null 2>&1
import trimesh
PY
  then
    log "trimesh import OK under /usr/bin/python"
    return 0
  fi
  if [[ "$SKIP_TRIMESH_PIP" -eq 1 || "$ALLOW_SYSTEM_PIP" -eq 0 ]]; then
    die "trimesh is missing and system-pip fallback was disabled."
  fi
  warn "trimesh is missing from system Python. Installing with sudo pip --break-system-packages, matching the tested CachyOS procedure."
  confirm_or_die "Install trimesh into system Python with sudo pip --break-system-packages?"
  sudo_cmd /usr/bin/python -m pip install trimesh --break-system-packages
}

run_wgpu_probe() {
  local probe="${WGPU_ROOT}/test-system-wgpu.sh"
  if [[ -f "$probe" ]]; then
    log "Running WGPU system-Python probe"
    run_cmd bash "$probe"
  else
    warn "WGPU probe script not found; running inline import probe."
    run_cmd /usr/bin/python - <<'PY'
mods = ["PySide6", "rendercanvas", "wgpu", "pylinalg", "pygfx"]
for m in mods:
    __import__(m)
    print("OK", m)
from rendercanvas.qt import QRenderWidget
print("rendercanvas Qt + WGPU stack OK under system Python")
PY
  fi
}

stage_payload() {
  log "Creating clean runtime staging payload"
  run_cmd bash "${PKG_ROOT}/build-local-package.sh" --stage-only
  if [[ "$DRY_RUN" -eq 1 ]]; then
    if [[ -d "$STAGING" ]]; then
      log "Dry-run: existing staging directory found: $STAGING"
    else
      warn "Dry-run: staging command was not executed, so $STAGING was not created."
    fi
    return 0
  fi
  [[ -d "$STAGING" ]] || die "Staging directory was not created: $STAGING"
}

compile_wrapper() {
  log "Compiling native C wrapper"
  if have /usr/bin/python-config; then
    run_cmd gcc -O2 -Wall -Wextra $(/usr/bin/python-config --embed --cflags 2>/dev/null || /usr/bin/python-config --cflags) \
      -o "$WRAPPER_BUILD" "$WRAPPER_SRC" \
      $(/usr/bin/python-config --embed --ldflags 2>/dev/null || /usr/bin/python-config --ldflags) -ldl -lm
  elif have python-config; then
    run_cmd gcc -O2 -Wall -Wextra $(python-config --embed --cflags 2>/dev/null || python-config --cflags) \
      -o "$WRAPPER_BUILD" "$WRAPPER_SRC" \
      $(python-config --embed --ldflags 2>/dev/null || python-config --ldflags) -ldl -lm
  else
    local pyver
    pyver="$(/usr/bin/python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
    run_cmd gcc -O2 -Wall -Wextra -I"/usr/include/python${pyver}" \
      -o "$WRAPPER_BUILD" "$WRAPPER_SRC" \
      -L/usr/lib -l"python${pyver}" -ldl -lm
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    warn "Dry-run: wrapper compile was not executed; not checking $WRAPPER_BUILD."
    return 0
  fi
  [[ -x "$WRAPPER_BUILD" ]] || die "Wrapper compile did not produce executable: $WRAPPER_BUILD"
}

install_payload() {
  log "Installing curated payload to ${APP_ROOT}"
  sudo_cmd rm -rf "$APP_ROOT"
  sudo_cmd install -dm755 "$APP_ROOT"
  sudo_cmd cp -a "${STAGING}/." "$APP_ROOT/"
  sudo_cmd install -Dm755 "$WRAPPER_BUILD" "$BIN_PATH"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    warn "Dry-run: payload install was not executed; skipping installed license-directory checks."
    return 0
  fi

  if [[ -d "${APP_ROOT}/bin/License" ]]; then
    log "Installing license bundle to ${LICENSE_PATH}"
    sudo_cmd install -dm755 "$LICENSE_PATH"
    sudo_cmd cp -a "${APP_ROOT}/bin/License/." "$LICENSE_PATH/"
    sudo_cmd find "$LICENSE_PATH" -type d -exec chmod 755 '{}' '+'
    sudo_cmd find "$LICENSE_PATH" -type f -exec chmod 644 '{}' '+'
  else
    warn "No license bundle found at ${APP_ROOT}/bin/License"
  fi
}

install_desktop_integration() {
  if [[ "$SKIP_DESKTOP" -eq 1 ]]; then
    warn "Skipping desktop integration."
    return 0
  fi
  log "Installing desktop entry and icon"
  [[ -f "$DESKTOP_SRC" ]] || die "Desktop file missing: $DESKTOP_SRC"
  [[ -f "$ICON_SRC" ]] || die "Icon file missing: $ICON_SRC"
  sudo_cmd install -Dm644 "$DESKTOP_SRC" "$DESKTOP_PATH"
  sudo_cmd install -Dm644 "$ICON_SRC" "$ICON_PATH"

  if have update-desktop-database; then
    sudo_cmd update-desktop-database /usr/share/applications || true
  fi
  if have gtk-update-icon-cache; then
    sudo_cmd gtk-update-icon-cache -f -t /usr/share/icons/hicolor || true
  fi
  if have kbuildsycoca6; then
    run_cmd kbuildsycoca6 || true
  elif have kbuildsycoca5; then
    run_cmd kbuildsycoca5 || true
  fi
}

validate_install() {
  if [[ "$SKIP_VALIDATION" -eq 1 ]]; then
    warn "Skipping validation."
    return 0
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    warn "Dry-run: skipping installed runtime validation because no files were installed."
    return 0
  fi
  log "Validating installed runtime"
  [[ -x "$BIN_PATH" ]] || die "Launcher missing or not executable: $BIN_PATH"
  [[ -d "$APP_ROOT/far_mesh" ]] || die "Installed far_mesh missing: $APP_ROOT/far_mesh"
  [[ -d "$APP_ROOT/bin" ]] || die "Installed bin assets missing: $APP_ROOT/bin"
  [[ -f "$APP_ROOT/bin/Instant Meshes" ]] || die "Instant Meshes missing from installed bin assets"
  [[ -f "$APP_ROOT/bin/quadwild" ]] || die "quadwild missing from installed bin assets"
  [[ -d "$APP_ROOT/quadwild-bimdf" ]] || die "quadwild-bimdf missing from installed payload"

  run_cmd /usr/bin/python - <<'PY'
mods = [
    "PySide6", "numpy", "scipy", "sklearn", "pandas", "matplotlib",
    "yaml", "psutil", "requests", "PIL", "trimesh", "open3d",
    "pyvista", "vtk", "rendercanvas", "wgpu", "pylinalg", "pygfx",
]
for mod in mods:
    __import__(mod)
    print("OK", mod)
print("FAR MESH native dependency import probe OK")
PY

  (cd "$APP_ROOT" && run_cmd /usr/bin/python - <<'PY'
import far_mesh
print("OK far_mesh import from installed app root", getattr(far_mesh, "__file__", ""))
PY
  )

  if [[ "$SKIP_DESKTOP" -eq 0 ]]; then
    [[ -f "$DESKTOP_PATH" ]] || die "Desktop file missing after install: $DESKTOP_PATH"
    [[ -f "$ICON_PATH" ]] || die "Icon missing after install: $ICON_PATH"
  fi
  log "Validation complete. Launch with: ${BIN_PATH}"
}

main() {
  log "FAR MESH Quad Option B source installer"
  log "Project root: ${PROJECT_ROOT}"
  validate_target
  require_source_tree
  install_arch_dependencies
  install_aur_dependencies
  install_vendored_packages
  ensure_trimesh
  run_wgpu_probe
  stage_payload
  compile_wrapper
  install_payload
  install_desktop_integration
  validate_install
  if [[ "$LAUNCH_AFTER_INSTALL" -eq 1 ]]; then
    log "Launching FAR MESH Quad"
    run_cmd "$BIN_PATH"
  fi
}

main "$@"
