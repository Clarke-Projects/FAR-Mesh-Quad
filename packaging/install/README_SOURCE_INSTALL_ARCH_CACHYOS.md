# FAR MESH Quad Native Source Installer — Arch / CachyOS

This directory contains the source-install entry point for the tested **Option B native** FAR MESH Quad package path.

## Main command

From the repository root:

```bash
bash packaging/install/install_far_mesh_quad.sh --target cachyos --dry-run
bash packaging/install/install_far_mesh_quad.sh --target cachyos --yes
```

For a truly fresh system where a full package database refresh and system upgrade is desired:

```bash
bash packaging/install/install_far_mesh_quad.sh --target cachyos --yes --system-update
```

`--system-update` is explicit by design. The installer does **not** run `pacman -Syyuu` by default.

## What the installer does

1. Installs required Arch/CachyOS packages through `pacman`.
2. Ensures `yay` exists when AUR packages are needed.
3. Installs known AUR dependencies.
4. Installs preserved vendored packages:
   - `python-open3d-1:0.19.0-13-x86_64.pkg.tar.zst`
   - WGPU stack packages from `packaging/native/wgpu-stack/packages/`
5. Checks/imports `trimesh` under `/usr/bin/python`.
6. Runs the WGPU system-Python probe.
7. Calls the native packaging script in `--stage-only` mode.
8. Compiles `far-mesh-quad-native-wrapper.c`.
9. Installs the curated runtime payload to `/opt/far-mesh-quad-native`.
10. Installs `/usr/bin/far-mesh-quad-native`.
11. Installs the desktop file and icon.
12. Installs the full license bundle to `/usr/share/licenses/far-mesh-quad-native`.
13. Runs runtime validation, including the v0.1.3 folderized BoreTool import probe.

## Modes

```text
--dry-run            Print planned commands without system mutation.
--yes                Non-interactive mode; passes --noconfirm where appropriate.
--system-update      Explicitly run sudo pacman -Syyuu before dependency install.
--skip-system-update Compatibility/no-op flag; full update is already skipped by default.
--target cachyos     Current supported target profile.
```

## Native staging contract

The installer uses:

```bash
bash packaging/native/far-mesh-quad-native/build-local-package.sh --stage-only
```

The staging script must copy only the curated runtime payload:

```text
far_mesh/
bin/
quadwild-bimdf/
scripts/
README.md
LICENSE
```

Optional inputs such as `requirements/` and `pyproject.toml` are copied when present.

Do not stage virtual environments, `_staging` outputs, package `pkg/`/`src/` build directories, Python caches, or local development scratch data.

For v0.1.3 the staged payload is also validated for the folderized BoreTool layout:

```text
far_mesh/core/bore/selection/
far_mesh/core/bore/recognition/
far_mesh/core/bore/rebuild/
```

The validation imports the public `far_mesh.core.bore` API plus direct imports from `selection.region_select`, `selection.mesh_realization`, `recognition.recognition`, `recognition.recognition_component_engine`, `rebuild.rebuild`, and `rebuild.rebuild_inventory`.

## Compiler-file safety

The installer does not mutate `/usr/bin/gcc-14` or `/usr/bin/g++-14`.

Vendored package payloads are checked before installation. If a package ever contains compiler binaries under `/usr/bin`, the correct fix is to repack that package, not to remove or replace system compiler files.

## Expected successful validation tail

A successful install should end with output similar to:

```text
OK rendercanvas
OK wgpu
OK pylinalg
OK pygfx
FAR MESH native dependency import probe OK
OK far_mesh import from installed app root /opt/far-mesh-quad-native/far_mesh/__init__.py
OK folderized Bore public API RebuildResult RebuildTargetPatch
OK folderized Bore selection select_region_data
OK folderized Bore recognition recognize_bore_region_selection component_engine_feature_candidates
OK folderized Bore rebuild delete_and_rebuild_candidate_region
[far-mesh-install] Validation complete. Launch with: /usr/bin/far-mesh-quad-native
```

Then launch:

```bash
far-mesh-quad-native
```


## v0.1.3 BoreTool folderization validation

Version 0.1.3 prepares the native installer for the folderized BoreTool package layout. The installer and staging helper now fail early if stale imports such as `far_mesh.core.bore.recognition_component_engine` remain after the move to `far_mesh.core.bore.recognition.recognition_component_engine`.

This is a packaging validation change only. It does not change BoreTool geometry, recognition, rebuild behavior, or the native wrapper model.
