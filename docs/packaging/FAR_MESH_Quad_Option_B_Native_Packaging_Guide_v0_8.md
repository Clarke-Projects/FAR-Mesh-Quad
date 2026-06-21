# FAR MESH Quad Option B Native Packaging Guide v0.8

## Purpose

This guide records the validated native packaging/install contract for FAR MESH Quad on Arch/CachyOS.

Option B means:

```text
native system install
no application virtual environment
system /usr/bin/python
native C wrapper launcher
curated runtime payload under /opt/far-mesh-quad-native
```

## Validated runtime layout

```text
/opt/far-mesh-quad-native/
/usr/bin/far-mesh-quad-native
/usr/share/applications/far-mesh-quad-native.desktop
/usr/share/icons/hicolor/256x256/apps/far-mesh-quad-native.png
/usr/share/licenses/far-mesh-quad-native/
```

The launcher process should appear as `far-mesh-quad-native`.

## Native wrapper contract

The C wrapper must:

```text
resolve FAR_MESH_NATIVE_APP_ROOT or default to /opt/far-mesh-quad-native
set FAR_MESH_APP_ROOT
set FAR_MESH_QUADWILD_ROOT
prepend quadwild-bimdf/build/Build/lib to LD_LIBRARY_PATH
clear VTK render-window override environment variables
default QT_QPA_PLATFORM=xcb
chdir to the app root
set multiprocessing executable to /usr/bin/python
run far_mesh.main through runpy.run_module(..., run_name="__main__", alter_sys=True)
```

Production process-name hacks such as `exec -a` or `prctl` are not part of this contract.

## Dependency binding model

The source installer binds the runtime through a mix of system packages, AUR packages, and preserved vendored binary packages.

Preserved vendored package artifacts are source-of-truth packaging inputs, not cleanup trash:

```text
packaging/native/prebuilt/python-open3d-1:0.19.0-13-x86_64.pkg.tar.zst
packaging/native/wgpu-stack/packages/*.pkg.tar.zst
```

Generated build trees such as `pkg/`, `src/`, `_staging`, Python `__pycache__`, wheels, local virtual environments, and development scratch folders are not native runtime payload.

## System update policy

The installer must not run a full system update by default.

Default behavior:

```bash
bash packaging/install/install_far_mesh_quad.sh --target cachyos --yes
```

Fresh-system behavior when explicitly requested:

```bash
bash packaging/install/install_far_mesh_quad.sh --target cachyos --yes --system-update
```

This runs:

```bash
sudo pacman -Syyuu
```

The old `--skip-system-update` flag is kept only as a compatibility/no-op flag.

## Staging contract

The native staging command is:

```bash
bash packaging/native/far-mesh-quad-native/build-local-package.sh --stage-only
```

`--stage-only` creates:

```text
packaging/native/far-mesh-quad-native/_staging/far-mesh-quad-native-0.1.2/
packaging/native/far-mesh-quad-native/far-mesh-quad-native-0.1.2.tar.zst
```

It does not run `makepkg`.

`--makepkg` remains available for package-building workflows.

## License bundle contract

A successful install must preserve the full license bundle in both locations:

```text
/opt/far-mesh-quad-native/bin/License/
/usr/share/licenses/far-mesh-quad-native/
```

The validated bundle contains:

```text
LICENSE_cgg-bern_quadwild-bimdf.txt
LICENSE_FAR_Mesh_Quad.txt
LICENSE_instant-meshes.txt
LICENSE_NumPy.txt
LICENSE_Open3D.txt
LICENSE_pygfx.txt
LICENSE_pygfx_wgpu-py.txt
LICENSE_pyside.txt
LICENSE_pyvista.txt
LICENSE_scikit-learn.txt
LICENSE_SciPy.txt
LICENSE_trimesh.txt
LICENSE_VTK.txt
```

## Safety invariant: no compiler mutation

The installer must not remove, move, replace, or recreate these system files:

```text
/usr/bin/gcc-14
/usr/bin/g++-14
```

If a future vendored package tries to install compiler binaries under `/usr/bin`, the package is wrong and must be repacked.

## Validated success criteria

A successful source install must pass:

```text
WGPU system-Python probe
Open3D import under /usr/bin/python
PyVista/VTK import under /usr/bin/python
trimesh import under /usr/bin/python
far_mesh import from /opt/far-mesh-quad-native
license bundle copied to both required locations
desktop entry and icon installed
native wrapper installed as /usr/bin/far-mesh-quad-native
```

Representative successful validation tail:

```text
FAR MESH native dependency import probe OK
OK far_mesh import from installed app root /opt/far-mesh-quad-native/far_mesh/__init__.py
[far-mesh-install] Validation complete. Launch with: /usr/bin/far-mesh-quad-native
```

## Public clone test plan

After pushing the installer and documentation:

```bash
cd /tmp
git clone https://github.com/Clarke-Projects/FAR-Mesh-Quad.git FAR-Mesh-Quad-clone-test
cd FAR-Mesh-Quad-clone-test
bash packaging/install/install_far_mesh_quad.sh --target cachyos --dry-run
bash packaging/install/install_far_mesh_quad.sh --target cachyos --yes
far-mesh-quad-native
```

For a fresh VM or clean machine, add `--system-update` to the real install command.
