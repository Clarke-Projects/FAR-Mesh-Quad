from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from .viewport_config import ViewportConfig


@runtime_checkable
class ViewportProtocol(Protocol):
    """
    Backend-neutral interface for embedded FAR Mesh viewport implementations.

    Purpose:
    - define the API expected by MainWindow and related UI/controller code
    - keep backend modules interchangeable
    - support gradual migration from PyVista to WGPU-first rendering
    - expose the selection/tool hooks needed by the future left tool panel

    Notes:
    - Qt signals are typed as Any because concrete backend widgets expose
      PySide6 Signal objects, not plain Python callables.
    - This protocol is intentionally practical rather than minimal.
    """

    BACKEND_NAME: str

    DISPLAY_PRESETS: dict[str, dict[str, Any]]
    CAMERA_PRESETS: set[str]
    COMPARE_MODES: set[str]
    SELECTION_MODES: set[str]
    DIAGNOSTIC_MODES: set[str]

    status_changed: Any
    mesh_loaded: Any
    mesh_failed: Any
    point_picked: Any
    selection_changed: Any
    compare_mode_changed: Any

    def __init__(
        self,
        parent: Any = None,
        *,
        config: ViewportConfig | None = None,
    ) -> None:
        ...

    # ------------------------------------------------------------------
    # state / capability queries
    # ------------------------------------------------------------------
    @property
    def current_path(self) -> str | None:
        ...

    def has_mesh(self) -> bool:
        ...

    def get_current_mesh_data(self) -> Any | None:
        ...

    def get_original_mesh_data(self) -> Any | None:
        ...

    def get_selection_mode(self) -> str:
        ...

    def get_display_preset(self) -> str:
        ...

    def get_compare_mode(self) -> str:
        ...

    def get_selected_cell_ids(self) -> list[int]:
        ...

    def get_selected_point_ids(self) -> list[int]:
        ...

    def get_selected_edge_ids(self) -> list[int]:
        ...

    def get_last_picked_world_pos(self) -> tuple[float, float, float] | None:
        ...

    def get_selection_state(self) -> dict[str, Any]:
        ...

    def get_capabilities(self) -> dict[str, bool]:
        ...

    def get_diagnostic_mode(self) -> str:
        ...

    # ------------------------------------------------------------------
    # mesh load / update API
    # ------------------------------------------------------------------
    def clear_scene(self) -> None:
        ...

    def load_file(self, path: str | Path) -> None:
        ...

    def load_dataset(self, dataset: Any, *, source_name: str = "dataset") -> None:
        ...

    def load_trimesh(self, mesh: Any, *, source_name: str = "trimesh") -> None:
        ...

    def set_mesh_data(
        self,
        dataset: Any,
        *,
        source_name: str = "dataset",
        keep_camera: bool = True,
        set_as_original: bool = False,
    ) -> None:
        ...

    def replace_mesh(
        self,
        dataset: Any,
        *,
        source_name: str = "dataset",
        keep_camera: bool = True,
    ) -> None:
        ...

    def update_mesh_geometry(self, dataset: Any, *, keep_camera: bool = True) -> None:
        ...

    def update_display_only(self) -> None:
        ...

    def reload_current_file(self) -> None:
        ...

    def set_original_mesh_data(self, dataset: Any | None) -> None:
        ...

    # ------------------------------------------------------------------
    # screenshots / camera
    # ------------------------------------------------------------------
    def capture_image(self, output_path: str | Path) -> str:
        ...

    def reset_camera(self) -> None:
        ...

    def view_isometric(self) -> None:
        ...

    def apply_camera_preset(self, preset: str) -> None:
        ...

    def focus_on_bounds(
        self,
        bounds: tuple[float, float, float, float, float, float],
    ) -> None:
        ...

    def focus_on_selection(self) -> None:
        ...

    # ------------------------------------------------------------------
    # display presets / compare modes / diagnostics
    # ------------------------------------------------------------------
    def apply_display_preset(self, preset: str) -> None:
        ...

    def set_compare_mode(self, mode: str) -> None:
        ...

    def set_diagnostic_mode(self, mode: str) -> None:
        ...

    # ------------------------------------------------------------------
    # edge / grid / axes / boundary / clip / host info controls
    # ------------------------------------------------------------------
    def set_edges_visible(self, enabled: bool) -> None:
        ...

    def set_edge_width(self, width: float) -> None:
        ...

    def set_grid_visible(self, enabled: bool) -> None:
        ...

    def set_axes_visible(self, enabled: bool) -> None:
        ...

    def set_boundary_highlight_visible(self, enabled: bool) -> None:
        ...

    def set_clip_plane(
        self,
        axis: str,
        fraction: float = 0.5,
        *,
        invert: bool = False,
    ) -> None:
        ...

    def clear_clip(self) -> None:
        ...

    def set_host_info_visible(self, visible: bool) -> None:
        ...

    def toggle_host_info_visible(self) -> None:
        ...

    def is_host_info_visible(self) -> bool:
        ...

    # ------------------------------------------------------------------
    # selection subsystem
    # ------------------------------------------------------------------
    def set_selection_mode(self, mode: str) -> None:
        ...

    def clear_selection(self) -> None:
        ...

    def highlight_cells(self, cell_ids: list[int]) -> None:
        ...

    def highlight_points(self, point_ids: list[int]) -> None:
        ...

    def grow_selection(self) -> None:
        ...

    def shrink_selection(self) -> None:
        ...

    def set_brush_selection_enabled(self, enabled: bool) -> None:
        ...

    def is_brush_selection_enabled(self) -> bool:
        ...

    def enable_surface_picking(
        self,
        callback: Callable[[tuple[float, float, float]], None] | None = None,
    ) -> None:
        ...

    def enable_mesh_picking(
        self,
        callback: Callable[[Any], None] | None = None,
    ) -> None:
        ...

    def disable_picking(self) -> None:
        ...

    # ------------------------------------------------------------------
    # overlay primitives
    # ------------------------------------------------------------------
    def clear_overlays(self) -> None:
        ...

    def show_marker(
        self,
        position: tuple[float, float, float],
        *,
        name: str = "marker",
        radius: float | None = None,
        color: str | None = None,
    ) -> None:
        ...

    def show_polyline(
        self,
        points: list[tuple[float, float, float]],
        *,
        name: str = "polyline",
        color: str | None = None,
        width: float = 3.0,
        closed: bool = False,
    ) -> None:
        ...

    def show_preview_mesh(
        self,
        dataset: Any,
        *,
        color: str = "#7ee787",
        opacity: float = 0.35,
        show_edges: bool = True,
        line_width: float = 1.5,
    ) -> None:
        ...

    def clear_preview_mesh(self) -> None:
        ...

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        ...


__all__ = [
    "ViewportProtocol",
]
