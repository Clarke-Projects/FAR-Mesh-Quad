# far_mesh/viewer/__init__.py
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .viewport_config import ViewportConfig
from .viewport_factory import create_viewport, get_requested_viewport_backend

if TYPE_CHECKING:
    from .pyvista_viewport import PyVistaViewport, PyVistaViewportConfig, ViewportError
    from .wgpu_viewport import WgpuViewport, WgpuViewportConfig, WgpuViewportError
    from .viewport_protocol import ViewportProtocol


# Shared config type, with compatibility aliases for older imports.
PyVistaViewportConfig = ViewportConfig
WgpuViewportConfig = ViewportConfig


__all__ = [
    "ViewportConfig",
    "ViewportProtocol",
    "PyVistaViewportConfig",
    "WgpuViewportConfig",
    "create_viewport",
    "get_requested_viewport_backend",
    "PyVistaViewport",
    "WgpuViewport",
    "ViewportError",
    "WgpuViewportError",
]


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "PyVistaViewport": (".pyvista_viewport", "PyVistaViewport"),
    "WgpuViewport": (".wgpu_viewport", "WgpuViewport"),
    "ViewportError": (".pyvista_viewport", "ViewportError"),
    "WgpuViewportError": (".wgpu_viewport", "WgpuViewportError"),
    "ViewportProtocol": (".viewport_protocol", "ViewportProtocol"),
}


def __getattr__(name: str) -> Any:
    """
    Lazy export layer for the viewer package.

    Goals:
    - avoid importing heavy backend modules on package import
    - keep backend modules optional until requested
    - expose a stable backend-neutral API from far_mesh.viewer
    """
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name, package=__name__)
        return getattr(module, attr_name)

    if name in {"ViewportConfig", "PyVistaViewportConfig", "WgpuViewportConfig"}:
        return ViewportConfig

    if name == "create_viewport":
        return create_viewport

    if name == "get_requested_viewport_backend":
        return get_requested_viewport_backend

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy exports to dir() and IDE/introspection helpers."""
    return sorted(set(globals().keys()) | set(__all__))
