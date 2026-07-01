# FAR MESH Quad — Native Option B source install on Linux

This guide covers the tested **Option B native install** paths for FAR MESH Quad.

Validated targets:

* Arch / CachyOS
* Ubuntu 26.04 LTS

Option B installs FAR MESH Quad as a native system application without an application virtual environment.

## What this installs

The native installers create this runtime layout:

```text
/opt/far-mesh-quad-native/          curated FAR MESH Quad runtime payload
/usr/bin/far-mesh-quad-native       native launcher
/usr/share/applications/far-mesh-quad-native.desktop
/usr/share/icons/hicolor/256x256/apps/far-mesh-quad-native.png
/usr/share/licenses/far-mesh-quad-native/
```

Both validated paths use system Python and native system packages. They do not create a private application venv and they do not perform runtime pip installation.

## Validated targets

### Arch / CachyOS

```text
Distribution: CachyOS / Arch-family Linux
Python:       system /usr/bin/python, currently Python 3.14 on the validated target
Viewport:     WGPU primary path, PyVista fallback available
Install mode: native system install, no application venv
Launcher:     native C wrapper using system CPython
```

### Ubuntu

```text
Distribution: Ubuntu 26.04 LTS
Python:       system /usr/bin/python3, Python 3.14 on the validated target
Viewport:     WGPU primary path, PyVista fallback available
Install mode: native system install, no application venv
Launcher:     native C exec wrapper launching /usr/bin/python3 -m far_mesh.main
```

## Repository payload required

A clone intended for native source install must preserve these source-of-truth payloads:

```text
bin/Instant Meshes
bin/quadwild
bin/config
bin/License
quadwild-bimdf/
packaging/native/wgpu-stack/packages/python-rendercanvas-2.6.3-1-any.pkg.tar.zst
packaging/native/wgpu-stack/packages/python-wgpu-0.31.0-1-any.pkg.tar.zst
packaging/native/wgpu-stack/packages/python-pylinalg-0.6.8-1-any.pkg.tar.zst
packaging/native/wgpu-stack/packages/python-pygfx-0.16.0-1-any.pkg.tar.zst
```

Arch / CachyOS additionally uses:

```text
packaging/native/prebuilt/python-open3d-1:0.19.0-13-x86_64.pkg.tar.zst
packaging/native/far-mesh-quad-native/build-local-package.sh
packaging/native/far-mesh-quad-native/far-mesh-quad-native-wrapper.c
packaging/native/far-mesh-quad-native/far-mesh-quad-native.desktop
packaging/install/install_far_mesh_quad.sh
```

Ubuntu additionally uses:

```text
packaging/install/install_ubuntu_option_b_native.sh
packaging/install/README_SOURCE_INSTALL_UBUNTU_OPTION_B.md
packaging/native/debian/wgpu-stack/build-wgpu-stack-debs.sh
packaging/native/debian/far-mesh-quad-native/build-local-deb.sh
```

Do not replace these payloads with generated `pkg/`, `src/`, `_staging`, `_build`, `_deb`, cache, or virtual-environment directories.

## Fresh clone

```bash
git clone https://github.com/Clarke-Projects/FAR-Mesh-Quad.git
cd FAR-Mesh-Quad
```

## Arch / CachyOS install

Preview the install first:

```bash
bash packaging/install/install_far_mesh_quad.sh --target cachyos --dry-run
```

On a fresh machine, explicitly allow the system update:

```bash
bash packaging/install/install_far_mesh_quad.sh --target cachyos --yes --system-update
```

On a machine that was already updated, omit the full system update:

```bash
bash packaging/install/install_far_mesh_quad.sh --target cachyos --yes
```

## Ubuntu 26.04 install

Run the Ubuntu native Option B installer:

```bash
bash packaging/install/install_ubuntu_option_b_native.sh -y
```

The installer performs the complete native path:

1. Installs Ubuntu apt dependencies.
2. Builds FAR MESH-owned `.deb` packages for:
   * `python3-rendercanvas-farmesh`
   * `python3-wgpu-farmesh`
   * `python3-pylinalg-farmesh`
   * `python3-pygfx-farmesh`
3. Builds the FAR MESH app package:
   * `far-mesh-quad-native`
4. Installs the WGPU stack packages.
5. Installs the FAR MESH app package.
6. Validates the installed `/opt` runtime.

The Ubuntu WGPU packages are produced from the vendored WGPU stack payloads and install their Python modules into:

```text
/usr/lib/python3/dist-packages
```

The FAR MESH app package installs:

```text
/opt/far-mesh-quad-native
/usr/bin/far-mesh-quad-native
/usr/share/applications/far-mesh-quad-native.desktop
/usr/share/icons/hicolor/256x256/apps/far-mesh-quad-native.png
/usr/share/licenses/far-mesh-quad-native/
```

## Launch

```bash
far-mesh-quad-native
```

Desktop-launch test:

```bash
gtk-launch far-mesh-quad-native
```

Expected process identity:

```bash
ps aux | grep far-mesh | grep -v grep
```

The main process should appear as:

```text
far-mesh-quad-native
```

Helper/resource-tracker/worker processes may still appear as Python child processes. That is expected.

## Verify installed license bundle

```bash
ls -1 /opt/far-mesh-quad-native/bin/License
ls -1 /usr/share/licenses/far-mesh-quad-native
```

Both locations should contain the same license bundle.

## Reinstall from an updated source tree

### Arch / CachyOS

After pulling new source changes:

```bash
git pull
bash packaging/install/install_far_mesh_quad.sh --target cachyos --yes
```

This restages the curated runtime payload, recompiles the native wrapper, reinstalls `/opt/far-mesh-quad-native`, refreshes desktop integration, and reruns validation.

### Ubuntu

After pulling new source changes:

```bash
git pull
bash packaging/install/install_ubuntu_option_b_native.sh -y --skip-apt-update
```

This rebuilds the local WGPU `.deb` packages, rebuilds the FAR MESH app `.deb`, reinstalls the packages, refreshes desktop integration, and reruns installed-root validation.

## What the installers validate

The native installers validate:

```text
system Python imports for PySide6, NumPy/SciPy/sklearn/pandas/matplotlib
trimesh
Open3D
PyVista / VTK
rendercanvas / wgpu / pylinalg / pygfx
far_mesh import from /opt/far-mesh-quad-native
license bundle installation
```

Ubuntu validation additionally prints the installed-root BoreTool checkpoint:

```text
OK folderized Bore inventory v177k_rebuild_zero_safety_audit_validation_counts_no_behavior_change_for_valid_diagnostics
```

## Important safety behavior

### Arch / CachyOS

The installer does not run a full system update by default.
Use `--system-update` only when you intentionally want:

```bash
sudo pacman -Syyuu
```

The installer does not remove, move, or recreate `/usr/bin/gcc-14` or `/usr/bin/g++-14`.
Vendored package payloads are checked instead. If a future package tries to own compiler binaries, the installer should stop and the package should be repacked rather than mutating system compiler files.

### Ubuntu

The Ubuntu installer uses `/usr/bin/python3`.

Do not fix Ubuntu by creating a fake `/usr/bin/python` symlink. The Ubuntu Option B path is intentionally built around system `/usr/bin/python3`.

The installer uses `apt-get` for system dependencies and local `.deb` packages for the FAR MESH-owned WGPU stack and app package.

## Known expected warnings

During staging, these may appear and are not fatal:

```text
Missing optional directory: requirements
Missing optional file: pyproject.toml
```

They mean those optional payload items were not present in this repository state. The staged runtime payload is still valid when validation passes.

On some Wayland / Mesa / hybrid-GPU systems, WGPU or Qt may print diagnostic messages such as:

```text
Unable to find extension: VK_EXT_physical_device_drm
No config found!
EGL says it can present to the window but not natively
Max vertex attribute stride unknown. Assuming it is 2048
qt.qpa.services: Failed to register with host portal ...
```

These are not install failures when the application opens and the WGPU viewport backend starts.

## Generated package outputs

Generated build directories and package outputs should not be committed:

```text
packaging/**/pkg/
packaging/**/src/
packaging/**/_staging/
packaging/native/debian/wgpu-stack/_build/
packaging/native/debian/wgpu-stack/_deb/
packaging/native/debian/far-mesh-quad-native/_build/
packaging/native/debian/far-mesh-quad-native/_deb/
```

Final built application packages should normally be published as GitHub Release assets rather than committed as ordinary source files.
