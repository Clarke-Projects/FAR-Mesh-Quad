# FAR MESH Quad Ubuntu Native Option B Install

This installs FAR MESH Quad on Ubuntu using the native Option B model:

- no venv
- no pip runtime install
- system /usr/bin/python3
- apt dependencies
- FAR MESH-owned .deb packages for rendercanvas / wgpu / pylinalg / pygfx
- runtime in /opt/far-mesh-quad-native
- launcher at /usr/bin/far-mesh-quad-native
- desktop entry at /usr/share/applications/far-mesh-quad-native.desktop

## Install

From a fresh clone:

    cd FAR-Mesh-Quad
    bash packaging/install/install_ubuntu_option_b_native.sh -y

## Launch

Terminal:

    far-mesh-quad-native

Desktop:

    FAR MESH Quad

Direct desktop-launch test:

    gtk-launch far-mesh-quad-native

## Validation

The installer validates:

- installed /opt runtime import
- native dependency imports
- rendercanvas / wgpu / pylinalg / pygfx imports
- folderized BoreTool imports
- rebuild inventory checkpoint

Expected BoreTool checkpoint:

    v177k_rebuild_zero_safety_audit_validation_counts_no_behavior_change_for_valid_diagnostics

Expected installed paths:

    /opt/far-mesh-quad-native
    /usr/bin/far-mesh-quad-native
    /usr/share/applications/far-mesh-quad-native.desktop

## Generated files

Do not commit generated package output directories:

    packaging/native/debian/wgpu-stack/_build/
    packaging/native/debian/wgpu-stack/_deb/
    packaging/native/debian/far-mesh-quad-native/_build/
    packaging/native/debian/far-mesh-quad-native/_deb/

These are local build artifacts only.
