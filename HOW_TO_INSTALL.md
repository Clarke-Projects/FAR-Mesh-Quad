# FAR MESH Quad — Install from Source on Arch / CachyOS

This guide covers the tested **Option B native install** path for Arch-family systems, especially CachyOS.
It installs FAR MESH Quad as a native system application without an application virtual environment.

## What this installs

The installer creates this runtime layout:

```text
/opt/far-mesh-quad-native/          curated FAR MESH Quad runtime payload
/usr/bin/far-mesh-quad-native       native C wrapper launcher
/usr/share/applications/far-mesh-quad-native.desktop
/usr/share/icons/hicolor/256x256/apps/far-mesh-quad-native.png
/usr/share/licenses/far-mesh-quad-native/
```

The installed launcher starts the application through the native wrapper, which embeds the system CPython and runs `far_mesh.main` from `/opt/far-mesh-quad-native`.

## Tested target

```text
Distribution: CachyOS / Arch-family Linux
Python:       system /usr/bin/python, currently Python 3.14 on the validated target
Viewport:     WGPU primary path, PyVista fallback available
Install mode: native system install, no application venv
```

## Repository payload required

A clone intended for native source install must contain these source-of-truth payloads:

```text
bin/Instant Meshes
bin/quadwild
bin/config
quadwild-bimdf/
packaging/native/prebuilt/python-open3d-1:0.19.0-13-x86_64.pkg.tar.zst
packaging/native/wgpu-stack/packages/python-rendercanvas-2.6.3-1-any.pkg.tar.zst
packaging/native/wgpu-stack/packages/python-wgpu-0.31.0-1-any.pkg.tar.zst
packaging/native/wgpu-stack/packages/python-pylinalg-0.6.8-1-any.pkg.tar.zst
packaging/native/wgpu-stack/packages/python-pygfx-0.16.0-1-any.pkg.tar.zst
packaging/native/far-mesh-quad-native/build-local-package.sh
packaging/native/far-mesh-quad-native/far-mesh-quad-native-wrapper.c
packaging/native/far-mesh-quad-native/far-mesh-quad-native.desktop
packaging/install/install_far_mesh_quad.sh
```

Do not replace these prebuilt package artifacts with generated `pkg/`, `src/`, `_staging`, cache, or virtual-environment directories.

## Fresh clone install

```bash
git clone https://github.com/Clarke-Projects/FAR-Mesh-Quad.git
cd FAR-Mesh-Quad
```

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

## Launch

```bash
far-mesh-quad-native
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

After pulling new source changes:

```bash
git pull
bash packaging/install/install_far_mesh_quad.sh --target cachyos --yes
```

This restages the curated runtime payload, recompiles the native wrapper, reinstalls `/opt/far-mesh-quad-native`, refreshes desktop integration, and reruns validation.

## What the installer validates

The installer checks:

```text
system Python imports for PySide6, NumPy/SciPy/sklearn/pandas/matplotlib
trimesh
Open3D
PyVista / VTK
rendercanvas / wgpu / pylinalg / pygfx
far_mesh import from /opt/far-mesh-quad-native
WGPU Qt binding under system Python
license bundle installation
```

## Important safety behavior

The installer does not run a full system update by default.
Use `--system-update` only when you intentionally want:

```bash
sudo pacman -Syyuu
```

The installer does not remove, move, or recreate `/usr/bin/gcc-14` or `/usr/bin/g++-14`.
Vendored package payloads are checked instead. If a future package tries to own compiler binaries, the installer should stop and the package should be repacked rather than mutating system compiler files.

## Known expected warnings

During staging, these may appear and are not fatal:

```text
Missing optional directory: requirements
Missing optional file: pyproject.toml
```

They mean those optional payload items were not present in this repository state. The staged runtime payload is still valid when validation passes.
