# FAR MESH Quad — Advanced Manual Native Package Build

This is the advanced/manual package-build procedure for maintainers. For normal public installation from a clean clone, use `HOW_TO_INSTALL.md` and the source installer script.

This manual procedure builds the native no-venv FAR MESH Quad package directly from a clean GitHub clone and then installs the produced package with `pacman -U`.

Validated target:

```text
Arch / CachyOS Linux
Native package: far-mesh-quad-native
Install root: /opt/far-mesh-quad-native
Launcher: /usr/bin/far-mesh-quad-native
Viewport backend: WGPU
```

### 1. Clone the repository

```bash
cd ~/Schreibtisch

git clone https://github.com/Clarke-Projects/FAR-Mesh-Quad.git FAR_Mesh_Quad_clone_test

cd FAR_Mesh_Quad_clone_test
```

### 2. Verify required payload is present

```bash
test -f 'bin/Instant Meshes' && echo "OK Instant Meshes"
test -f bin/quadwild && echo "OK quadwild"
test -d bin/config && echo "OK bin/config"
test -d quadwild-bimdf && echo "OK quadwild-bimdf"

test -f 'packaging/native/prebuilt/python-open3d-1:0.19.0-13-x86_64.pkg.tar.zst' && echo "OK Open3D prebuilt"
test -d packaging/native/wgpu-stack/packages && echo "OK WGPU package folder"

test -f packaging/native/far-mesh-quad-native/PKGBUILD && echo "OK PKGBUILD"
test -f packaging/native/far-mesh-quad-native/build-local-package.sh && echo "OK build script"
```

Expected result:

```text
OK Instant Meshes
OK quadwild
OK bin/config
OK quadwild-bimdf
OK Open3D prebuilt
OK WGPU package folder
OK PKGBUILD
OK build script
```

### 3. Install vendored native dependency packages

FAR MESH Quad vendors the Open3D package and the WGPU stack packages used by the native build.

```bash
sudo pacman -U --needed \
  'packaging/native/prebuilt/python-open3d-1:0.19.0-13-x86_64.pkg.tar.zst' \
  packaging/native/wgpu-stack/packages/*.pkg.tar.zst
```

If the packages are already installed, pacman may print warnings like:

```text
is up to date -- skipping
there is nothing to do
```

That is okay.

### 4. Build the native package

```bash
cd packaging/native/far-mesh-quad-native

bash build-local-package.sh
```

Expected package output:

```text
far-mesh-quad-native-0.1.2-1-x86_64.pkg.tar.zst
```

A successful build ends with output similar to:

```text
Package build complete:
far-mesh-quad-native-0.1.2-1-x86_64.pkg.tar.zst
```

### 5. Install the package

```bash
sudo pacman -U far-mesh-quad-native-0.1.2-1-x86_64.pkg.tar.zst
```

The package installs to:

```text
/opt/far-mesh-quad-native/
```

The launcher is installed as:

```text
/usr/bin/far-mesh-quad-native
```

### 6. Launch FAR MESH Quad

```bash
far-mesh-quad-native
```

Expected startup output:

```text
[viewport_factory] Using WGPU viewport backend.
preconfigure_default_device (pygfx): required_features set to {'!float32-filterable'} removes earlier set {'float32-filterable'} from the set.
Unable to find extension: VK_EXT_physical_device_drm
```

The `VK_EXT_physical_device_drm` message is a non-blocking Vulkan/WGPU warning. It does not indicate a failed launch.

### 7. Optional dependency smoke test

After installation, this command verifies the native package can import the important runtime modules:

```bash
cd /opt/far-mesh-quad-native

/usr/bin/python -c '
import far_mesh
import trimesh, open3d, pyvista, pyvistaqt, vtk
import rendercanvas, wgpu, pylinalg, pygfx
print("FAR MESH native package dependency smoke test OK")
'
```

Expected output:

```text
FAR MESH native package dependency smoke test OK
```

### 8. Validation checkpoint

A clean clone is considered packaging-valid when all of the following pass:

```text
Repository clone: PASS
Required runtime payload present: PASS
Vendored dependency packages present: PASS
Native package build: PASS
Package install: PASS
Application launch: PASS
WGPU startup: PASS
```

Validated package result:

```text
far-mesh-quad-native-0.1.2-1-x86_64.pkg.tar.zst
```

Validated installed package size:

```text
approximately 34 MiB
```

### Relationship to HOW_TO_INSTALL.md

`HOW_TO_INSTALL.md` is the primary user-facing install guide. It runs the higher-level source installer:

```bash
bash packaging/install/install_far_mesh_quad.sh --target cachyos --yes
```

This document remains useful when you specifically want to test the lower-level package build path.

### Notes

Do not build from inside the original development tree when testing repository completeness. Use a clean clone.

Do not commit generated build folders such as:

```text
packaging/**/pkg/
packaging/**/src/
packaging/**/_staging/
```

Do not commit final built application packages unless intentionally publishing them. Final release packages are better uploaded as GitHub Release assets.

The repository intentionally includes the small runtime tool payloads required by FAR MESH Quad:

```text
bin/Instant Meshes
bin/quadwild
bin/config/
quadwild-bimdf/
```

The repository also intentionally includes vendored native dependency packages needed for reproducible Option B packaging:

```text
packaging/native/prebuilt/python-open3d-1:0.19.0-13-x86_64.pkg.tar.zst
packaging/native/wgpu-stack/packages/*.pkg.tar.zst
```
