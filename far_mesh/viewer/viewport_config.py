from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ViewportConfig:
    """
    Backend-neutral viewport configuration shared by all embedded viewport
    implementations.

    Design rules:
    - keep this file free of PyVista, VTK, WGPU, or Qt rendering logic
    - only store visual defaults and common frontend-facing behavior defaults
    - allow MainWindow and viewer factory code to configure a viewport without
      importing a specific backend module
    """

    # ------------------------------------------------------------------
    # default visible states
    # ------------------------------------------------------------------
    show_edges_default: bool = True
    edge_width_default: float = 1.5
    show_grid_default: bool = True
    show_axes_default: bool = True
    show_boundary_default: bool = False
    show_host_info_default: bool = False

    # ------------------------------------------------------------------
    # default interaction / mode states
    # ------------------------------------------------------------------
    display_preset_default: str = "inspection_edges"
    compare_mode_default: str = "current_only"
    selection_mode_default: str = "none"
    diagnostic_mode_default: str = "none"
    brush_selection_default: bool = True

    # ------------------------------------------------------------------
    # colors
    # ------------------------------------------------------------------
    background_color: str = "#20242b"
    mesh_color: str = "#d7dde6"
    wire_color: str = "#101317"
    selection_color: str = "#ff4fa3"
    point_selection_color: str = "#ff4fa3"
    hover_color: str = "#ffff00"
    boundary_color: str = "#ffb347"
    compare_color: str = "#66d9ef"
    overlay_color: str = "#8be9fd"
    preview_color: str = "#7ee787"

    # ------------------------------------------------------------------
    # overlay / marker scaling
    # ------------------------------------------------------------------
    point_marker_scale: float = 0.005
    point_marker_min_radius: float = 0.01
    preview_opacity_default: float = 0.35

    def validate(self) -> None:
        """
        Validate simple shared config invariants.
        """
        if self.edge_width_default <= 0.0:
            raise ValueError("edge_width_default must be > 0")
        if self.point_marker_scale <= 0.0:
            raise ValueError("point_marker_scale must be > 0")
        if self.point_marker_min_radius <= 0.0:
            raise ValueError("point_marker_min_radius must be > 0")
        if not (0.0 < self.preview_opacity_default <= 1.0):
            raise ValueError("preview_opacity_default must be in (0, 1]")


__all__ = [
    "ViewportConfig",
]
