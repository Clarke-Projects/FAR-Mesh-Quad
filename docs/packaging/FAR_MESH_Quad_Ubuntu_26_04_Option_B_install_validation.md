# FAR MESH Quad Ubuntu 26.04 native Option B install validation

Validation target:

- Distribution: Ubuntu 26.04 LTS
- Python: system /usr/bin/python3, Python 3.14.x
- Install model: native Option B
- Runtime root: /opt/far-mesh-quad-native
- Launcher: /usr/bin/far-mesh-quad-native
- Desktop entry: /usr/share/applications/far-mesh-quad-native.desktop
- WGPU stack location: /usr/lib/python3/dist-packages
- Application package: far-mesh-quad-native
- No application venv
- No runtime pip install

Validated package path:

- Built FAR MESH-owned WGPU dependency .deb packages:
  - python3-rendercanvas-farmesh
  - python3-wgpu-farmesh
  - python3-pylinalg-farmesh
  - python3-pygfx-farmesh
- Built FAR MESH application .deb package:
  - far-mesh-quad-native
- Installed the packages through apt.
- Verified /usr/bin/far-mesh-quad-native exists.
- Verified /opt/far-mesh-quad-native exists.
- Verified /usr/share/applications/far-mesh-quad-native.desktop exists.
- Verified gtk-launch far-mesh-quad-native starts the application.
- Verified process identity:
  - far-mesh-quad-native -m far_mesh.main

Installed-root validation passed:

- far_mesh imported from:
  - /opt/far-mesh-quad-native/far_mesh/__init__.py
- Native dependency imports passed:
  - PySide6
  - numpy
  - scipy
  - sklearn
  - pandas
  - matplotlib
  - yaml
  - psutil
  - requests
  - PIL
  - trimesh
  - open3d
  - pyvista
  - vtk
  - rendercanvas
  - wgpu
  - pylinalg
  - pygfx

Folderized BoreTool validation passed:

- RebuildResult
- RebuildTargetPatch
- select_region_data
- build_opening_evidence_ledger_from_arrays
- recognize_bore_region_selection
- component_engine_feature_candidates
- delete_and_rebuild_candidate_region

Expected BoreTool checkpoint:

    v177k_rebuild_zero_safety_audit_validation_counts_no_behavior_change_for_valid_diagnostics

Clean reinstall validation:

- Removed far-mesh-quad-native and the four FAR MESH WGPU stack packages.
- Confirmed launcher, /opt runtime, and desktop entry were removed.
- Reinstalled using:
    bash packaging/install/install_ubuntu_option_b_native.sh -y --skip-apt-update
- Rebuilt WGPU .deb packages.
- Rebuilt FAR MESH application .deb package.
- Reinstalled all packages.
- Revalidated installed /opt runtime.
- Revalidated dependency imports.
- Revalidated BoreTool checkpoint.
- Revalidated gtk-launch startup.

Observed non-fatal runtime diagnostics:

- Unable to find extension: VK_EXT_physical_device_drm
- No config found!
- EGL says it can present to the window but not natively
- Max vertex attribute stride unknown. Assuming it is 2048
- qt.qpa.services: Failed to register with host portal

These diagnostics did not block startup. The WGPU viewport backend started successfully.
