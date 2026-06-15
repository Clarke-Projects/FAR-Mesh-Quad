from __future__ import annotations

import os
from typing import Any

_VALID_VIEWPORT_BACKENDS = {"pyvista", "wgpu"}
_DEFAULT_VIEWPORT_BACKEND = "wgpu"


def get_requested_viewport_backend(explicit_backend: str | None = None) -> str:
    """
    Resolve the requested viewport backend.

    Selection order:
    1. explicit function argument
    2. FAR_MESH_VIEWPORT_BACKEND environment variable
    3. default: WGPU

    Design intent:
    - WGPU is the primary Farmesh 3 viewport
    - PyVista remains the compatibility / fallback backend
    """
    raw_value = explicit_backend
    if raw_value is None:
        raw_value = os.environ.get("FAR_MESH_VIEWPORT_BACKEND", _DEFAULT_VIEWPORT_BACKEND)

    selected = str(raw_value).strip().lower()

    if selected not in _VALID_VIEWPORT_BACKENDS:
        print(
            f"[viewport_factory] Unknown viewport backend {selected!r}; "
            f"falling back to {_DEFAULT_VIEWPORT_BACKEND!r}."
        )
        return _DEFAULT_VIEWPORT_BACKEND

    return selected


def _create_wgpu_viewport(parent: Any = None, *, config: Any = None):
    from .wgpu_viewport import WgpuViewport

    return WgpuViewport(parent=parent, config=config)


def _create_pyvista_viewport(parent: Any = None, *, config: Any = None):
    from .pyvista_viewport import PyVistaViewport

    return PyVistaViewport(parent=parent, config=config)


def create_viewport(
    parent: Any = None,
    *,
    config: Any = None,
    backend: str | None = None,
):
    """
    Create the requested embedded viewport backend.

    Important design rule:
    - backend modules are imported lazily here
    - WGPU is attempted first by default
    - PyVista is the compatibility fallback if WGPU fails
    """
    selected = get_requested_viewport_backend(backend)

    if selected == "pyvista":
        print("[viewport_factory] Using PyVista viewport backend (explicit request).")
        return _create_pyvista_viewport(parent=parent, config=config)

    try:
        print("[viewport_factory] Using WGPU viewport backend.")
        return _create_wgpu_viewport(parent=parent, config=config)
    except Exception as exc:
        print(
            "[viewport_factory] WGPU backend failed, "
            f"falling back to PyVista: {exc!r}"
        )

    print("[viewport_factory] Using PyVista viewport backend (fallback).")
    return _create_pyvista_viewport(parent=parent, config=config)


__all__ = [
    "create_viewport",
    "get_requested_viewport_backend",
]
