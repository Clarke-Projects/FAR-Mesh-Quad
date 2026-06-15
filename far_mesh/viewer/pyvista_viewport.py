# far_mesh/viewer/pyvista_viewport.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from far_mesh.core.selection_edges import resolve_edge_index_from_pick_info, select_edge_region

from .viewport_config import ViewportConfig


class ViewportError(RuntimeError):
    """Base error for PyVista viewport failures."""


# Backward-compatibility alias.
PyVistaViewportConfig = ViewportConfig


class PyVistaViewport(QWidget):
    BACKEND_NAME = "pyvista"

    status_changed = Signal(str)
    mesh_loaded = Signal(str)
    mesh_failed = Signal(str)
    point_picked = Signal(tuple)
    selection_changed = Signal(object)
    compare_mode_changed = Signal(str)

    DISPLAY_PRESETS: dict[str, dict[str, Any]] = {
        "viewer_clean": {
            "show_grid": True,
            "show_axes": True,
            "show_edges": False,
            "main_opacity": 1.0,
            "smooth_shading": True,
            "ambient": 0.18,
            "diffuse": 0.88,
            "specular": 0.08,
        },
        "inspection_edges": {
            "show_grid": True,
            "show_axes": True,
            "show_edges": True,
            "main_opacity": 1.0,
            "smooth_shading": True,
            "ambient": 0.20,
            "diffuse": 0.84,
            "specular": 0.06,
        },
        "repair_selection": {
            "show_grid": True,
            "show_axes": True,
            "show_edges": True,
            "main_opacity": 0.95,
            "smooth_shading": True,
            "ambient": 0.22,
            "diffuse": 0.82,
            "specular": 0.06,
        },
        "shaded_only": {
            "show_grid": False,
            "show_axes": True,
            "show_edges": False,
            "main_opacity": 1.0,
            "smooth_shading": True,
            "ambient": 0.18,
            "diffuse": 0.90,
            "specular": 0.08,
        },
        "shaded + wireframe": {
            "show_grid": True,
            "show_axes": True,
            "show_edges": True,
            "main_opacity": 1.0,
            "smooth_shading": True,
            "ambient": 0.20,
            "diffuse": 0.84,
            "specular": 0.06,
        },
        "wireframe": {
            "show_grid": True,
            "show_axes": True,
            "show_edges": True,
            "main_opacity": 0.18,
            "smooth_shading": False,
            "ambient": 0.25,
            "diffuse": 0.75,
            "specular": 0.0,
        },
    }

    CAMERA_PRESETS = {
        "isometric",
        "front",
        "back",
        "left",
        "right",
        "top",
        "bottom",
    }

    COMPARE_MODES = {
        "current_only",
        "original_only",
        "overlay_ghost",
    }

    SELECTION_MODES = {
        "none",
        "point",
        "face",
        "edge",
        "mesh",
    }

    DIAGNOSTIC_MODES = {
        "none",
        "normals",
        "face_orientation",
        "uv_checker",
        "vertex_colors",
        "boundaries_only",
        "non_manifold",
        "heatmap_distance",
    }

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        config: ViewportConfig | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config or ViewportConfig()
        validate = getattr(self.config, "validate", None)
        if callable(validate):
            try:
                validate()
            except Exception:
                pass

        self._current_path: str | None = None
        self._current_mesh_data: pv.DataSet | None = None
        self._original_mesh_data: pv.DataSet | None = None
        self._display_mesh_data: pv.DataSet | None = None
        self._current_render_kwargs: dict[str, Any] = {}

        self._mesh_actor: Any | None = None
        self._wire_actor: Any | None = None
        self._compare_actor: Any | None = None
        self._floor_grid_actor: Any | None = None
        self._selection_actor: Any | None = None
        self._boundary_actor: Any | None = None
        self._tool_preview_actor: Any | None = None
        self._edge_selection_actor: Any | None = None
        self._overlay_actors: dict[str, Any] = {}

        self._selected_actor: Any | None = None
        self._overlay_specs: dict[str, dict[str, Any]] = {}
        self._tool_preview_data: pv.DataSet | None = None
        self._tool_preview_kwargs: dict[str, Any] = {}

        self._selection_mode: str = getattr(self.config, "selection_mode_default", "none")
        if self._selection_mode not in self.SELECTION_MODES:
            self._selection_mode = "none"

        self._selected_cell_ids: list[int] = []
        self._selected_point_ids: list[int] = []
        self._selected_edge_ids: list[int] = []
        self._edge_region_strategy: str = "safe"
        self._last_picked_world_pos: tuple[float, float, float] | None = None
        self._surface_pick_callback: Callable[[tuple[float, float, float]], None] | None = None
        self._picking_enabled = False

        self._compare_mode: str = getattr(self.config, "compare_mode_default", "current_only")
        if self._compare_mode not in self.COMPARE_MODES:
            self._compare_mode = "current_only"

        self._diagnostic_mode: str = getattr(self.config, "diagnostic_mode_default", "none")
        if self._diagnostic_mode not in self.DIAGNOSTIC_MODES:
            self._diagnostic_mode = "none"

        self._show_boundary_edges: bool = bool(getattr(self.config, "show_boundary_default", False))
        self._show_host_info: bool = bool(getattr(self.config, "show_host_info_default", False))
        self._brush_selection_enabled: bool = bool(getattr(self.config, "brush_selection_default", False))

        self._clip_axis: str | None = None
        self._clip_value: float | None = None
        self._clip_invert: bool = False

        self._show_edges = self.config.show_edges_default
        self._edge_width = self.config.edge_width_default
        self._show_grid = self.config.show_grid_default
        self._show_axes = self.config.show_axes_default

        self._display_preset = getattr(self.config, "display_preset_default", "inspection_edges")
        if self._display_preset not in self.DISPLAY_PRESETS:
            self._display_preset = "inspection_edges"

        self._camera_preset = "isometric"
        self._camera_state: dict[str, Any] | None = None

        self._current_vertices: np.ndarray | None = None
        self._current_faces: np.ndarray | None = None
        self._open_edges: np.ndarray | None = None
        self._edge_index_to_vertices: np.ndarray | None = None
        self._edge_key_to_index: Dict[Tuple[int, int], int] = {}
        self._edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
        self._vertex_adjacency: Dict[int, Set[int]] = {}

        self.plotter = QtInteractor(self)
        self.plotter.set_background(self.config.background_color)
        self._configure_render_quality()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.plotter.interactor)

        self._apply_display_preset_to_flags(self._display_preset)
        self._rebuild_scene(reset_camera=False)
        self.status_changed.emit("PyVista viewport ready.")

    # ------------------------------------------------------------------
    # state / capability queries
    # ------------------------------------------------------------------
    @property
    def current_path(self) -> str | None:
        return self._current_path

    def has_mesh(self) -> bool:
        return self._current_mesh_data is not None

    def get_current_mesh_data(self) -> pv.DataSet | None:
        return self._current_mesh_data

    def get_original_mesh_data(self) -> pv.DataSet | None:
        return self._original_mesh_data

    def get_selection_mode(self) -> str:
        return self._selection_mode

    def get_display_preset(self) -> str:
        return self._display_preset

    def get_compare_mode(self) -> str:
        return self._compare_mode

    def get_diagnostic_mode(self) -> str:
        return self._diagnostic_mode

    def get_selected_cell_ids(self) -> list[int]:
        return list(self._selected_cell_ids)

    def get_selected_point_ids(self) -> list[int]:
        return list(self._selected_point_ids)

    def get_selected_edge_ids(self) -> list[int]:
        return list(self._selected_edge_ids)

    def get_last_picked_world_pos(self) -> tuple[float, float, float] | None:
        return self._last_picked_world_pos

    def get_selection_state(self) -> dict[str, Any]:
        return {
            "mode": self._selection_mode,
            "selected_cell_ids": list(self._selected_cell_ids),
            "selected_point_ids": list(self._selected_point_ids),
            "selected_edge_ids": list(self._selected_edge_ids),
            "edge_region_strategy": self._edge_region_strategy,
            "last_picked_world_pos": self._last_picked_world_pos,
            "brush_select_enabled": self._brush_selection_enabled,
            "diagnostic_mode": self._diagnostic_mode,
            "host_info_visible": self._show_host_info,
        }

    def get_capabilities(self) -> dict[str, bool]:
        edge_ready = self._edge_index_to_vertices is not None and len(self._edge_index_to_vertices) > 0
        return {
            "embedded": True,
            "compare_mode": True,
            "clip_plane": True,
            "point_picking": True,
            "face_picking": True,
            "mesh_picking": True,
            "edge_picking": bool(edge_ready),
            "edge_selection": bool(edge_ready),
            "feature_edge_region_select": bool(edge_ready),
            "boundary_edges": True,
            "preview_mesh": True,
            "overlays": True,
            "screenshots": True,
            "face_rgba": True,
            "texture_materials": True,
            "brush_face_selection": False,
            "brush_edge_selection": False,
            "diagnostic_modes": False,
            "host_info_panel": False,
        }

    # ------------------------------------------------------------------
    # mesh load / update API
    # ------------------------------------------------------------------
    def clear_scene(self) -> None:
        self.disable_picking()

        self._current_path = None
        self._current_mesh_data = None
        self._display_mesh_data = None
        self._current_render_kwargs = {}

        self._mesh_actor = None
        self._wire_actor = None
        self._compare_actor = None
        self._floor_grid_actor = None
        self._selection_actor = None
        self._boundary_actor = None
        self._tool_preview_actor = None
        self._edge_selection_actor = None
        self._selected_actor = None

        self._selected_cell_ids = []
        self._selected_point_ids = []
        self._selected_edge_ids = []
        self._last_picked_world_pos = None
        self._current_vertices = None
        self._current_faces = None
        self._open_edges = None
        self._edge_index_to_vertices = None
        self._edge_key_to_index = {}
        self._edge_to_faces = {}
        self._vertex_adjacency = {}

        self._overlay_actors.clear()
        self._overlay_specs.clear()
        self._tool_preview_data = None
        self._tool_preview_kwargs = {}

        self._clip_axis = None
        self._clip_value = None
        self._clip_invert = False

        self.plotter.clear()
        self._rebuild_scene(reset_camera=False)
        self.selection_changed.emit(self.get_selection_state())
        self.status_changed.emit("Viewport cleared.")

    def load_file(self, path: str | Path) -> None:
        mesh_path = Path(path).expanduser().resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Viewport input file does not exist: {mesh_path}")

        try:
            dataset, render_kwargs = self._load_dataset_with_visuals(mesh_path)
        except Exception as exc:
            message = f"PyVista failed to read file: {mesh_path}\nUnderlying error: {exc!r}"
            self.mesh_failed.emit(message)
            raise ViewportError(message) from exc

        self._current_render_kwargs = render_kwargs
        self.set_mesh_data(dataset, source_name=str(mesh_path))
        self._current_path = str(mesh_path)
        self.mesh_loaded.emit(str(mesh_path))
        self.status_changed.emit(f"Loaded mesh: {mesh_path.name}")

    def load_dataset(self, dataset: pv.DataSet, *, source_name: str = "dataset") -> None:
        self._current_render_kwargs = {}
        self.set_mesh_data(dataset, source_name=source_name)

    def load_trimesh(self, mesh: Any, *, source_name: str = "trimesh") -> None:
        dataset = self._dataset_from_trimesh_like(mesh)
        self._current_render_kwargs = {}
        self.set_mesh_data(dataset, source_name=source_name)

    def set_mesh_data(
        self,
        dataset: pv.DataSet,
        *,
        source_name: str = "dataset",
        keep_camera: bool = True,
        set_as_original: bool = False,
    ) -> None:
        if keep_camera:
            self._save_camera_state()

        self.disable_picking()
        self.clear_selection()

        self._current_mesh_data = self._coerce_dataset(dataset)
        self._sync_topology_from_current_mesh()
        if set_as_original or self._original_mesh_data is None:
            self._original_mesh_data = self._current_mesh_data.copy(deep=True)

        self._display_mesh_data = self._apply_current_clip(self._current_mesh_data)
        self._rebuild_scene(reset_camera=not keep_camera)
        self._restore_or_apply_default_camera()
        self.plotter.render()

        self.status_changed.emit(f"Viewport updated from {source_name}.")

    def replace_mesh(
        self,
        dataset: pv.DataSet,
        *,
        source_name: str = "dataset",
        keep_camera: bool = True,
    ) -> None:
        self.set_mesh_data(dataset, source_name=source_name, keep_camera=keep_camera)

    def update_mesh_geometry(self, dataset: pv.DataSet, *, keep_camera: bool = True) -> None:
        self.set_mesh_data(dataset, source_name="updated geometry", keep_camera=keep_camera)

    def update_display_only(self) -> None:
        if self._current_mesh_data is None:
            self._rebuild_scene(reset_camera=False)
            return

        self._save_camera_state()
        self._display_mesh_data = self._apply_current_clip(self._current_mesh_data)
        self._rebuild_scene(reset_camera=False)
        self._restore_or_apply_default_camera()
        self.plotter.render()

    def reload_current_file(self) -> None:
        if self._current_path is None:
            self.status_changed.emit("No mesh file to reload.")
            return
        self.load_file(self._current_path)

    def set_original_mesh_data(self, dataset: pv.DataSet | None) -> None:
        self._original_mesh_data = None if dataset is None else self._coerce_dataset(dataset)
        self.update_display_only()

    # ------------------------------------------------------------------
    # screenshots / camera
    # ------------------------------------------------------------------
    def capture_image(self, output_path: str | Path) -> str:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        self.plotter.screenshot(str(out))
        self.status_changed.emit(f"Screenshot saved: {out.name}")
        return str(out)

    def reset_camera(self) -> None:
        if self._display_mesh_data is not None:
            self.apply_camera_preset(self._camera_preset)
        else:
            self.plotter.reset_camera()
            self.plotter.render()
            self.status_changed.emit("Camera reset.")

    def view_isometric(self) -> None:
        self.apply_camera_preset("isometric")

    def apply_camera_preset(self, preset: str) -> None:
        preset = preset.lower().strip()
        if preset not in self.CAMERA_PRESETS:
            raise ValueError(f"Unknown camera preset: {preset}")

        self._camera_preset = preset
        target = self._selection_bounds_or_mesh_bounds()
        if target is None:
            self.plotter.render()
            return

        self._apply_camera_preset_for_bounds(target, preset)
        self.plotter.render()
        self.status_changed.emit(f"Camera preset: {preset}")

    def focus_on_bounds(self, bounds: tuple[float, float, float, float, float, float]) -> None:
        self._apply_camera_preset_for_bounds(bounds, self._camera_preset)
        self.plotter.render()
        self.status_changed.emit("Focused bounds.")

    def focus_on_selection(self) -> None:
        bounds = self._selection_bounds()
        if bounds is None:
            self.status_changed.emit("No selection to focus.")
            return
        self.focus_on_bounds(bounds)

    # ------------------------------------------------------------------
    # display / compare / diagnostic
    # ------------------------------------------------------------------
    def apply_display_preset(self, preset: str) -> None:
        preset = preset.strip()
        if preset not in self.DISPLAY_PRESETS:
            raise ValueError(f"Unknown display preset: {preset}")

        self._display_preset = preset
        self._apply_display_preset_to_flags(preset)
        self.update_display_only()
        self.status_changed.emit(f"Display preset: {preset}")

    def set_compare_mode(self, mode: str) -> None:
        mode = mode.strip()
        if mode not in self.COMPARE_MODES:
            raise ValueError(f"Unknown compare mode: {mode}")

        self._compare_mode = mode
        self.update_display_only()
        self.compare_mode_changed.emit(mode)
        self.status_changed.emit(f"Compare mode: {mode}")

    def set_diagnostic_mode(self, mode: str) -> None:
        mode = mode.strip()
        if mode not in self.DIAGNOSTIC_MODES:
            raise ValueError(f"Unknown diagnostic mode: {mode}")

        self._diagnostic_mode = mode
        # Harmless stub for protocol parity.
        # PyVista fallback keeps current rendering unchanged.
        self.selection_changed.emit(self.get_selection_state())
        if mode == "none":
            self.status_changed.emit("PyVista diagnostic mode cleared.")
        else:
            self.status_changed.emit(
                f"PyVista diagnostic mode set to '{mode}' (placeholder/no-op fallback)."
            )

    # ------------------------------------------------------------------
    # edge / grid / axes / boundary / clip / host info
    # ------------------------------------------------------------------
    def set_edges_visible(self, enabled: bool) -> None:
        self._show_edges = bool(enabled)
        self.update_display_only()

    def set_edge_width(self, width: float) -> None:
        width = float(width)
        if width <= 0:
            raise ValueError("edge width must be > 0")
        self._edge_width = width
        self.update_display_only()

    def set_grid_visible(self, enabled: bool) -> None:
        self._show_grid = bool(enabled)
        self.update_display_only()

    def set_axes_visible(self, enabled: bool) -> None:
        self._show_axes = bool(enabled)
        self.update_display_only()

    def set_boundary_highlight_visible(self, enabled: bool) -> None:
        self._show_boundary_edges = bool(enabled)
        self.update_display_only()

    def set_clip_plane(
        self,
        axis: str,
        fraction: float = 0.5,
        *,
        invert: bool = False,
    ) -> None:
        if self._current_mesh_data is None:
            return

        axis = axis.lower().strip()
        if axis not in {"x", "y", "z"}:
            raise ValueError("axis must be one of: x, y, z")

        fraction = float(np.clip(fraction, 0.0, 1.0))
        xmin, xmax, ymin, ymax, zmin, zmax = self._current_mesh_data.bounds

        if axis == "x":
            value = xmin + (xmax - xmin) * fraction
        elif axis == "y":
            value = ymin + (ymax - ymin) * fraction
        else:
            value = zmin + (zmax - zmin) * fraction

        self._clip_axis = axis
        self._clip_value = float(value)
        self._clip_invert = bool(invert)

        self.update_display_only()
        self.status_changed.emit(f"Clip plane set: {axis}={value:.3f}")

    def clear_clip(self) -> None:
        self._clip_axis = None
        self._clip_value = None
        self._clip_invert = False
        self.update_display_only()
        self.status_changed.emit("Clip cleared.")

    def set_host_info_visible(self, visible: bool) -> None:
        # Harmless protocol-parity stub.
        self._show_host_info = bool(visible)
        self.selection_changed.emit(self.get_selection_state())
        self.status_changed.emit(
            f"PyVista host info visibility set to {self._show_host_info} (no visible panel in fallback backend)."
        )

    def toggle_host_info_visible(self) -> None:
        self.set_host_info_visible(not self._show_host_info)

    def is_host_info_visible(self) -> bool:
        return bool(self._show_host_info)

    # ------------------------------------------------------------------
    # selection subsystem
    # ------------------------------------------------------------------
    def set_selection_mode(self, mode: str) -> None:
        mode = mode.strip().lower()
        if mode not in self.SELECTION_MODES:
            raise ValueError(f"Unknown selection mode: {mode}")

        self.disable_picking()
        self._selection_mode = mode

        if mode == "none":
            self.status_changed.emit("Selection disabled.")
            self.selection_changed.emit(self.get_selection_state())
            return
        if mode == "point":
            self._enable_point_selection_internal()
            self.status_changed.emit("Point selection enabled.")
            self.selection_changed.emit(self.get_selection_state())
            return
        if mode == "face":
            self._enable_face_selection_internal()
            self.status_changed.emit("Face selection enabled.")
            self.selection_changed.emit(self.get_selection_state())
            return
        if mode == "mesh":
            self._enable_mesh_selection_internal()
            self.status_changed.emit("Mesh selection enabled.")
            self.selection_changed.emit(self.get_selection_state())
            return
        if mode == "edge":
            if self._edge_index_to_vertices is None or len(self._edge_index_to_vertices) == 0:
                self.status_changed.emit("Edge selection unavailable: current PyVista mesh has no edge topology.")
                self.selection_changed.emit(self.get_selection_state())
                return
            self._enable_edge_selection_internal()
            self.status_changed.emit("PyVista edge selection enabled.")
            self.selection_changed.emit(self.get_selection_state())

    def clear_selection(self) -> None:
        self._selected_cell_ids.clear()
        self._selected_point_ids.clear()
        self._selected_edge_ids.clear()
        self._selected_actor = None
        self._last_picked_world_pos = None

        for attr_name in ("_selection_actor", "_edge_selection_actor"):
            actor = getattr(self, attr_name, None)
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
                setattr(self, attr_name, None)

        # keep overlay markers separate; do not clear all overlays here
        self.plotter.render()
        self.selection_changed.emit(self.get_selection_state())

    def highlight_cells(self, cell_ids: list[int]) -> None:
        if self._display_mesh_data is None:
            return

        self._selected_cell_ids = sorted({int(v) for v in cell_ids if int(v) >= 0})
        self._selected_point_ids = []
        self._selected_edge_ids = []

        if not self._selected_cell_ids:
            self.clear_selection()
            return

        if self._selection_actor is not None:
            try:
                self.plotter.remove_actor(self._selection_actor)
            except Exception:
                pass
            self._selection_actor = None

        try:
            selected = self._display_mesh_data.extract_cells(self._selected_cell_ids)
            self._selection_actor = self.plotter.add_mesh(
                selected,
                color=self.config.selection_color,
                opacity=0.95,
                show_edges=True,
                line_width=max(2.0, self._edge_width + 1.5),
                lighting=False,
                pickable=False,
            )
        except Exception as exc:
            raise ViewportError(f"Could not highlight selected cells: {exc!r}") from exc

        self.plotter.render()
        self.selection_changed.emit(self.get_selection_state())

    def highlight_points(self, point_ids: list[int]) -> None:
        if self._display_mesh_data is None:
            return

        self._selected_point_ids = sorted({int(v) for v in point_ids if int(v) >= 0})
        self._selected_cell_ids = []

        if not self._selected_point_ids:
            self.clear_selection()
            return

        if self._selection_actor is not None:
            try:
                self.plotter.remove_actor(self._selection_actor)
            except Exception:
                pass
            self._selection_actor = None

        try:
            selected = self._display_mesh_data.extract_points(self._selected_point_ids)
            self._selection_actor = self.plotter.add_mesh(
                selected,
                color=self.config.selection_color,
                point_size=15,
                render_points_as_spheres=True,
                lighting=False,
                pickable=False,
            )
        except Exception as exc:
            raise ViewportError(f"Could not highlight selected points: {exc!r}") from exc

        self.plotter.render()
        self.selection_changed.emit(self.get_selection_state())

    def grow_selection(self) -> None:
        """
        Conservative, backend-neutral growth helper for selected faces.

        For PyVista fallback this grows by shared-point neighborhood.
        For point selection it currently does nothing beyond preserving state.
        """
        if self._display_mesh_data is None:
            return

        if self._selected_cell_ids:
            expanded = set(self._selected_cell_ids)
            point_ids: set[int] = set()

            for cid in self._selected_cell_ids:
                try:
                    cell = self._display_mesh_data.get_cell(cid)
                    point_ids.update(int(pid) for pid in cell.point_ids)
                except Exception:
                    continue

            for cid in range(int(self._display_mesh_data.n_cells)):
                if cid in expanded:
                    continue
                try:
                    cell = self._display_mesh_data.get_cell(cid)
                    if any(int(pid) in point_ids for pid in cell.point_ids):
                        expanded.add(cid)
                except Exception:
                    continue

            self.highlight_cells(sorted(expanded))
            self.status_changed.emit("PyVista selection grown.")
            return

        if self._selected_point_ids:
            self.status_changed.emit("PyVista point-selection grow is not implemented in fallback backend.")
            return

        self.status_changed.emit("No selection to grow.")

    def shrink_selection(self) -> None:
        """
        Conservative fallback shrink helper for selected faces.

        Removes selected cells that touch any point shared with a non-selected cell.
        """
        if self._display_mesh_data is None:
            return

        if self._selected_cell_ids:
            selected = set(self._selected_cell_ids)
            point_to_cells: dict[int, set[int]] = {}

            for cid in range(int(self._display_mesh_data.n_cells)):
                try:
                    cell = self._display_mesh_data.get_cell(cid)
                    for pid in cell.point_ids:
                        point_to_cells.setdefault(int(pid), set()).add(cid)
                except Exception:
                    continue

            kept: list[int] = []
            for cid in sorted(selected):
                try:
                    cell = self._display_mesh_data.get_cell(cid)
                    is_boundary = False
                    for pid in cell.point_ids:
                        neighbors = point_to_cells.get(int(pid), set())
                        if any(nid not in selected for nid in neighbors):
                            is_boundary = True
                            break
                    if not is_boundary:
                        kept.append(cid)
                except Exception:
                    continue

            self.highlight_cells(kept)
            self.status_changed.emit("PyVista selection shrunk.")
            return

        if self._selected_point_ids:
            self.status_changed.emit("PyVista point-selection shrink is not implemented in fallback backend.")
            return

        self.status_changed.emit("No selection to shrink.")

    def set_brush_selection_enabled(self, enabled: bool) -> None:
        # Harmless protocol-parity stub: PyVista fallback does not implement WGPU-style drag brush selection.
        self._brush_selection_enabled = bool(enabled)
        self.selection_changed.emit(self.get_selection_state())
        self.status_changed.emit(
            f"PyVista brush selection flag set to {self._brush_selection_enabled} (no drag-brush implementation in fallback backend)."
        )

    def is_brush_selection_enabled(self) -> bool:
        return bool(self._brush_selection_enabled)

    def enable_surface_picking(
        self,
        callback: Callable[[tuple[float, float, float]], None] | None = None,
    ) -> None:
        self._surface_pick_callback = callback
        self.set_selection_mode("face")

    def enable_mesh_picking(
        self,
        callback: Callable[[Any], None] | None = None,
    ) -> None:
        self.disable_picking()
        self._selection_mode = "mesh"
        self._selected_actor = None

        def _on_pick(picked: Any) -> None:
            self._selected_actor = picked
            if callback is not None:
                callback(picked)
            self.selection_changed.emit(self.get_selection_state())

        self.plotter.enable_mesh_picking(
            callback=_on_pick,
            show=False,
            left_clicking=True,
            use_actor=False,
        )
        self._picking_enabled = True
        self.status_changed.emit("Mesh picking enabled.")
        self.selection_changed.emit(self.get_selection_state())

    def disable_picking(self) -> None:
        disable = getattr(self.plotter, "disable_picking", None)
        if callable(disable):
            try:
                disable()
            except Exception:
                pass
        self._picking_enabled = False
        self._selection_mode = "none"

    # ------------------------------------------------------------------
    # overlay primitives
    # ------------------------------------------------------------------
    def clear_overlays(self) -> None:
        for actor in self._overlay_actors.values():
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self._overlay_actors.clear()
        self._overlay_specs.clear()

        if self._tool_preview_actor is not None:
            try:
                self.plotter.remove_actor(self._tool_preview_actor)
            except Exception:
                pass
            self._tool_preview_actor = None

        self._tool_preview_data = None
        self._tool_preview_kwargs = {}
        self.plotter.render()

    def show_marker(
        self,
        position: tuple[float, float, float],
        *,
        name: str = "marker",
        radius: float | None = None,
        color: str | None = None,
    ) -> None:
        pos = tuple(float(v) for v in position)
        color = color or self.config.overlay_color
        radius = radius or self._default_marker_radius()

        self._overlay_specs[name] = {
            "type": "marker",
            "position": pos,
            "radius": float(radius),
            "color": color,
        }
        self._rebuild_overlay_actor(name)
        self.plotter.render()

    def show_polyline(
        self,
        points: list[tuple[float, float, float]],
        *,
        name: str = "polyline",
        color: str | None = None,
        width: float = 3.0,
        closed: bool = False,
    ) -> None:
        if len(points) < 2:
            return

        pts = np.asarray(points, dtype=float)
        color = color or self.config.overlay_color

        self._overlay_specs[name] = {
            "type": "polyline",
            "points": pts,
            "color": color,
            "width": float(width),
            "closed": bool(closed),
        }
        self._rebuild_overlay_actor(name)
        self.plotter.render()

    def show_preview_mesh(
        self,
        dataset: pv.DataSet | Any,
        *,
        color: str = "#7ee787",
        opacity: float = 0.35,
        show_edges: bool = True,
        line_width: float = 1.5,
    ) -> None:
        self._tool_preview_data = self._coerce_dataset(dataset)
        self._tool_preview_kwargs = {
            "color": color,
            "opacity": float(opacity),
            "show_edges": bool(show_edges),
            "line_width": float(line_width),
            "pickable": False,
            "lighting": False,
        }
        self.update_display_only()

    def clear_preview_mesh(self) -> None:
        self._tool_preview_data = None
        self._tool_preview_kwargs = {}
        self.update_display_only()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        self.disable_picking()
        try:
            self.plotter.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # data conversion / load helpers
    # ------------------------------------------------------------------
    def _coerce_dataset(self, dataset: pv.DataSet | Any) -> pv.DataSet:
        if isinstance(dataset, pv.DataSet):
            return dataset.copy(deep=True)

        if hasattr(dataset, "vertices") and hasattr(dataset, "faces"):
            return self._dataset_from_trimesh_like(dataset)

        raise ViewportError(f"Unsupported mesh input type: {type(dataset)!r}")

    def _dataset_from_trimesh_like(self, mesh: Any) -> pv.DataSet:
        try:
            vertices = np.asarray(mesh.vertices, dtype=float)
            faces = np.asarray(mesh.faces, dtype=np.int64)
        except Exception as exc:
            raise ViewportError(
                f"Could not extract vertices/faces from trimesh-like object: {exc!r}"
            ) from exc

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ViewportError("Expected vertices shaped (N, 3).")
        if faces.ndim != 2 or faces.shape[1] < 3:
            raise ViewportError("Expected faces shaped (M, 3+).")

        face_sizes = np.full((faces.shape[0], 1), faces.shape[1], dtype=np.int64)
        face_array = np.hstack((face_sizes, faces)).ravel()

        return pv.PolyData(vertices, face_array)

    def _load_dataset_with_visuals(self, path: Path) -> tuple[pv.DataSet, dict[str, Any]]:
        suffix = path.suffix.lower()
        render_kwargs: dict[str, Any] = {}

        if suffix in {".obj", ".ply", ".stl", ".off"}:
            try:
                import trimesh

                loaded = trimesh.load(path, force="mesh", process=False)
                if isinstance(loaded, trimesh.Scene):
                    geometries = [g for g in loaded.geometry.values() if hasattr(g, "faces")]
                    if not geometries:
                        raise ViewportError(f"No mesh geometry found in scene: {path}")
                    loaded = trimesh.util.concatenate(geometries)
                if hasattr(loaded, "faces") and hasattr(loaded, "vertices"):
                    poly, render_kwargs = self._dataset_from_trimesh_with_visuals(loaded, path)
                    return poly, render_kwargs
            except Exception:
                pass

        try:
            dataset = pv.read(path)
            if dataset.active_scalars_name is not None:
                render_kwargs = {"scalars": dataset.active_scalars_name}
                arr = dataset.active_scalars
                if arr is not None and arr.ndim == 2 and arr.shape[1] in (3, 4):
                    render_kwargs["rgb"] = True
                    if arr.shape[1] == 4:
                        render_kwargs["rgba"] = True
            return dataset, render_kwargs
        except Exception as exc:
            raise ViewportError(f"Could not load mesh: {path}") from exc

    def _dataset_from_trimesh_with_visuals(
        self, mesh: Any, source_path: Path
    ) -> tuple[pv.DataSet, dict[str, Any]]:
        poly = self._dataset_from_trimesh_like(mesh)
        render_kwargs: dict[str, Any] = {}

        visual = getattr(mesh, "visual", None)
        if visual is None:
            return poly, render_kwargs

        if getattr(visual, "kind", None) == "texture":
            uv = getattr(visual, "uv", None)
            material = getattr(visual, "material", None)

            texture = self._texture_from_trimesh_material(material)

            if texture is None and material is not None:
                texture = self._load_texture_from_mtl(source_path)

            if uv is not None:
                uv = np.asarray(uv, dtype=float)
                if uv.ndim == 2 and uv.shape[0] == poly.n_points and uv.shape[1] >= 2:
                    uv2 = uv[:, :2].copy()
                    uv2[:, 1] = 1.0 - uv2[:, 1]
                    poly.active_texture_coordinates = uv2

                    if texture is not None:
                        render_kwargs = {"texture": texture, "color": "white"}
                        return poly, render_kwargs

        face_colors = getattr(visual, "face_colors", None)
        if face_colors is not None:
            face_colors = np.asarray(face_colors)
            if len(face_colors) == poly.n_cells:
                if face_colors.ndim == 2 and face_colors.shape[1] >= 3:
                    if face_colors.shape[1] == 3:
                        alpha = np.full((face_colors.shape[0], 1), 255, dtype=np.uint8)
                        face_colors = np.hstack((face_colors.astype(np.uint8), alpha))
                    else:
                        face_colors = face_colors[:, :4].astype(np.uint8)
                    poly.cell_data["face_rgba"] = face_colors
                    render_kwargs = {
                        "scalars": "face_rgba",
                        "rgb": True,
                        "rgba": True,
                        "preference": "cell",
                        "interpolate_before_map": False,
                    }
                    return poly, render_kwargs

        vertex_colors = getattr(visual, "vertex_colors", None)
        if vertex_colors is not None:
            vertex_colors = np.asarray(vertex_colors)
            if len(vertex_colors) == poly.n_points:
                if vertex_colors.ndim == 2 and vertex_colors.shape[1] >= 3:
                    if vertex_colors.shape[1] == 3:
                        alpha = np.full((vertex_colors.shape[0], 1), 255, dtype=np.uint8)
                        vertex_colors = np.hstack((vertex_colors.astype(np.uint8), alpha))
                    else:
                        vertex_colors = vertex_colors[:, :4].astype(np.uint8)
                    poly.point_data["vertex_rgba"] = vertex_colors
                    render_kwargs = {
                        "scalars": "vertex_rgba",
                        "rgb": True,
                        "rgba": True,
                        "preference": "point",
                        "interpolate_before_map": False,
                    }
                    return poly, render_kwargs

        material = getattr(visual, "material", None)
        if material is not None:
            main_color = getattr(material, "main_color", None)
            if main_color is not None:
                color = np.asarray(main_color, dtype=float)
                if len(color) >= 3:
                    render_kwargs = {"color": color[:3] / 255.0}

        return poly, render_kwargs

    def _texture_from_trimesh_material(self, material: Any) -> pv.Texture | None:
        if material is None:
            return None

        image = getattr(material, "image", None)
        if image is None:
            return None

        try:
            if hasattr(image, "convert"):
                img = image.convert("RGBA")
                arr = np.asarray(img)
            else:
                arr = np.asarray(image)

            if arr.ndim == 2:
                alpha = np.full_like(arr, 255)
                arr = np.stack([arr, arr, arr, alpha], axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 3:
                alpha = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=arr.dtype)
                arr = np.concatenate([arr, alpha], axis=2)
            elif arr.ndim != 3 or arr.shape[2] not in {3, 4}:
                return None

            arr = np.flipud(arr)
            return pv.Texture(arr)
        except Exception:
            return None

    def _load_texture_from_mtl(self, obj_path: Path) -> pv.Texture | None:
        try:
            obj_content = obj_path.read_text(encoding="utf-8", errors="ignore")
            mtl_line = None
            for line in obj_content.splitlines():
                if line.startswith("mtllib "):
                    mtl_line = line.strip()
                    break
            if mtl_line is None:
                return None

            mtl_filename = mtl_line.split(" ", 1)[1].strip()
            mtl_path = obj_path.parent / mtl_filename
            if not mtl_path.exists():
                return None

            mtl_content = mtl_path.read_text(encoding="utf-8", errors="ignore")
            map_kd = None
            for line in mtl_content.splitlines():
                if line.startswith("map_Kd "):
                    map_kd = line.split(" ", 1)[1].strip()
                    break
            if map_kd is None:
                return None

            tex_path = mtl_path.parent / map_kd
            if not tex_path.exists():
                for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tga"]:
                    alt = tex_path.with_suffix(ext)
                    if alt.exists():
                        tex_path = alt
                        break

            if tex_path.exists():
                try:
                    img = pv.read_texture(tex_path)
                    return img
                except Exception:
                    pass

                try:
                    from PIL import Image

                    img = Image.open(tex_path).convert("RGBA")
                    arr = np.asarray(img)
                    arr = np.flipud(arr)
                    return pv.Texture(arr)
                except Exception:
                    pass
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # scene rebuild helpers
    # ------------------------------------------------------------------
    def _rebuild_scene(self, *, reset_camera: bool) -> None:
        self.plotter.clear()

        self._mesh_actor = None
        self._wire_actor = None
        self._compare_actor = None
        self._floor_grid_actor = None
        self._selection_actor = None
        self._boundary_actor = None
        self._tool_preview_actor = None
        self._overlay_actors.clear()

        self._apply_scene_decorations()

        if self._display_mesh_data is not None:
            self._build_main_mesh_actor()
            self._build_wire_overlay_actor()
            self._build_compare_actor()
            self._build_boundary_actor()
            self._build_selection_actor()
            self._build_tool_preview_actor()
            self._rebuild_all_overlay_actors()

        if reset_camera and self._display_mesh_data is not None:
            self._apply_camera_preset_for_bounds(self._display_mesh_data.bounds, self._camera_preset)

    def _build_main_mesh_actor(self) -> None:
        if self._display_mesh_data is None:
            return

        data = self._mesh_data_for_compare_mode()
        if data is None:
            return

        preset = self.DISPLAY_PRESETS[self._display_preset]
        mesh_kwargs: dict[str, Any] = {
            "opacity": float(preset["main_opacity"]),
            "smooth_shading": bool(preset["smooth_shading"]),
            "ambient": float(preset["ambient"]),
            "diffuse": float(preset["diffuse"]),
            "specular": float(preset["specular"]),
            "pickable": True,
            "show_edges": False,
        }

        if "texture" in self._current_render_kwargs:
            mesh_kwargs.update(self._current_render_kwargs)
            mesh_kwargs.pop("scalars", None)
            mesh_kwargs.pop("rgb", None)
            mesh_kwargs.pop("rgba", None)
        elif "scalars" in self._current_render_kwargs:
            mesh_kwargs.update(self._current_render_kwargs)
            mesh_kwargs.pop("color", None)
        else:
            mesh_kwargs["color"] = self.config.mesh_color

        self._mesh_actor = self.plotter.add_mesh(data, **mesh_kwargs)

    def _build_wire_overlay_actor(self) -> None:
        if self._display_mesh_data is None:
            return
        if not self._show_edges:
            return

        data = self._mesh_data_for_compare_mode()
        if data is None:
            return

        self._wire_actor = self.plotter.add_mesh(
            data,
            style="wireframe",
            color=self.config.wire_color,
            opacity=0.95,
            line_width=float(self._edge_width),
            lighting=False,
            pickable=False,
        )

    def _build_compare_actor(self) -> None:
        if self._compare_mode != "overlay_ghost":
            return
        if self._original_mesh_data is None or self._display_mesh_data is None:
            return

        self._compare_actor = self.plotter.add_mesh(
            self._original_mesh_data,
            color=self.config.compare_color,
            opacity=0.22,
            smooth_shading=True,
            lighting=False,
            pickable=False,
            show_edges=False,
        )

    def _build_boundary_actor(self) -> None:
        if not self._show_boundary_edges:
            return
        if self._display_mesh_data is None:
            return

        try:
            edges = self._display_mesh_data.extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
                manifold_edges=False,
                non_manifold_edges=True,
            )
        except Exception:
            return

        if edges.n_cells <= 0:
            return

        self._boundary_actor = self.plotter.add_mesh(
            edges,
            color=self.config.boundary_color,
            line_width=max(2.0, self._edge_width + 1.0),
            lighting=False,
            pickable=False,
        )

    def _build_selection_actor(self) -> None:
        if self._display_mesh_data is None:
            return

        if self._selected_cell_ids:
            try:
                selected = self._display_mesh_data.extract_cells(self._selected_cell_ids)
                self._selection_actor = self.plotter.add_mesh(
                    selected,
                    color=self.config.selection_color,
                    opacity=0.95,
                    show_edges=True,
                    line_width=max(2.0, self._edge_width + 1.5),
                    lighting=False,
                    pickable=False,
                )
            except Exception:
                self._selection_actor = None
            return

        if self._selected_point_ids:
            try:
                selected = self._display_mesh_data.extract_points(self._selected_point_ids)
                self._selection_actor = self.plotter.add_mesh(
                    selected,
                    color=self.config.selection_color,
                    point_size=15,
                    render_points_as_spheres=True,
                    lighting=False,
                    pickable=False,
                )
            except Exception:
                self._selection_actor = None

    def _build_tool_preview_actor(self) -> None:
        if self._tool_preview_data is None:
            return

        kwargs = {
            "color": getattr(self.config, "preview_color", "#7ee787"),
            "opacity": getattr(self.config, "preview_opacity_default", 0.35),
            "show_edges": True,
            "line_width": 1.5,
            "pickable": False,
            "lighting": False,
        }
        kwargs.update(self._tool_preview_kwargs)
        self._tool_preview_actor = self.plotter.add_mesh(self._tool_preview_data, **kwargs)

    def _rebuild_all_overlay_actors(self) -> None:
        for name in list(self._overlay_specs.keys()):
            self._rebuild_overlay_actor(name)

    def _rebuild_overlay_actor(self, name: str) -> None:
        spec = self._overlay_specs.get(name)
        if spec is None:
            return

        existing = self._overlay_actors.get(name)
        if existing is not None:
            try:
                self.plotter.remove_actor(existing)
            except Exception:
                pass
            self._overlay_actors.pop(name, None)

        actor = None
        kind = spec["type"]

        if kind == "marker":
            sphere = pv.Sphere(
                radius=float(spec["radius"]),
                center=tuple(spec["position"]),
                theta_resolution=20,
                phi_resolution=20,
            )
            actor = self.plotter.add_mesh(
                sphere,
                color=spec["color"],
                opacity=1.0,
                lighting=False,
                pickable=False,
            )
        elif kind == "polyline":
            polyline = pv.lines_from_points(
                np.asarray(spec["points"], dtype=float),
                close=bool(spec["closed"]),
            )
            actor = self.plotter.add_mesh(
                polyline,
                color=spec["color"],
                line_width=float(spec["width"]),
                lighting=False,
                pickable=False,
            )

        if actor is not None:
            self._overlay_actors[name] = actor

    def _mesh_data_for_compare_mode(self) -> pv.DataSet | None:
        if self._compare_mode == "current_only":
            return self._display_mesh_data
        if self._compare_mode == "original_only":
            return self._original_mesh_data
        if self._compare_mode == "overlay_ghost":
            return self._display_mesh_data
        return self._display_mesh_data

    def _apply_scene_decorations(self) -> None:
        remove_bounds_axes = getattr(self.plotter, "remove_bounds_axes", None)
        if callable(remove_bounds_axes):
            try:
                remove_bounds_axes()
            except Exception:
                pass

        if self._show_grid:
            grid = self._build_floor_grid_polydata()
            self._floor_grid_actor = self.plotter.add_mesh(
                grid,
                color="#2c313a",
                line_width=1.0,
                opacity=0.85,
                pickable=False,
                lighting=False,
            )

        if self._show_axes:
            try:
                self.plotter.show_axes()
            except Exception:
                pass
        else:
            hide_axes = getattr(self.plotter, "hide_axes", None)
            if callable(hide_axes):
                try:
                    hide_axes()
                except Exception:
                    pass

    def _build_floor_grid_polydata(self) -> pv.PolyData:
        if self._display_mesh_data is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = self._display_mesh_data.bounds
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            sx = max(xmax - xmin, 1e-6)
            sy = max(ymax - ymin, 1e-6)
            sz = max(zmax - zmin, 1e-6)
            span = max(sx, sy, 1.0)
            pad = span * 0.45

            gx0 = cx - (span * 0.5 + pad)
            gx1 = cx + (span * 0.5 + pad)
            gy0 = cy - (span * 0.5 + pad)
            gy1 = cy + (span * 0.5 + pad)
            gz = zmin - sz * 0.04
            step = self._nice_grid_step(span / 10.0)
        else:
            gx0, gx1 = -1.0, 1.0
            gy0, gy1 = -1.0, 1.0
            gz = -0.1
            step = 0.2

        xs = np.arange(gx0, gx1 + step * 0.5, step, dtype=float)
        ys = np.arange(gy0, gy1 + step * 0.5, step, dtype=float)

        points: list[list[float]] = []
        lines: list[int] = []

        for x in xs:
            i0 = len(points)
            points.append([x, gy0, gz])
            points.append([x, gy1, gz])
            lines.extend([2, i0, i0 + 1])

        for y in ys:
            i0 = len(points)
            points.append([gx0, y, gz])
            points.append([gx1, y, gz])
            lines.extend([2, i0, i0 + 1])

        return pv.PolyData(np.array(points, dtype=float), lines=np.array(lines, dtype=int))

    @staticmethod
    def _nice_grid_step(approx_step: float) -> float:
        exp = np.floor(np.log10(approx_step))
        base = approx_step / (10 ** exp)
        if base < 1.5:
            nice = 1.0
        elif base < 3.5:
            nice = 2.0
        elif base < 7.5:
            nice = 5.0
        else:
            nice = 10.0
        return nice * (10 ** exp)

    def _save_camera_state(self) -> None:
        try:
            camera = self.plotter.camera
            self._camera_state = {
                "position": camera.GetPosition(),
                "focal_point": camera.GetFocalPoint(),
                "view_up": camera.GetViewUp(),
                "view_angle": camera.GetViewAngle(),
            }
        except Exception:
            self._camera_state = None

    def _restore_camera_state(self) -> bool:
        if self._camera_state is None:
            return False

        try:
            camera = self.plotter.camera
            camera.SetPosition(*self._camera_state["position"])
            camera.SetFocalPoint(*self._camera_state["focal_point"])
            camera.SetViewUp(*self._camera_state["view_up"])
            camera.SetViewAngle(self._camera_state["view_angle"])

            renderer = getattr(self.plotter, "renderer", None)
            if renderer is not None:
                renderer.ResetCameraClippingRange()
            return True
        except Exception:
            return False

    def _restore_or_apply_default_camera(self) -> None:
        restored = self._restore_camera_state()
        if not restored and self._display_mesh_data is not None:
            self._apply_camera_preset_for_bounds(self._display_mesh_data.bounds, self._camera_preset)

    def _selection_bounds_or_mesh_bounds(self) -> tuple[float, float, float, float, float, float] | None:
        bounds = self._selection_bounds()
        if bounds is not None:
            return bounds
        if self._display_mesh_data is not None:
            return self._display_mesh_data.bounds
        return None

    def _selection_bounds(self) -> tuple[float, float, float, float, float, float] | None:
        if self._display_mesh_data is None:
            return None

        if self._selected_cell_ids:
            try:
                selected = self._display_mesh_data.extract_cells(self._selected_cell_ids)
                return selected.bounds
            except Exception:
                return None

        if self._selected_point_ids:
            try:
                selected = self._display_mesh_data.extract_points(self._selected_point_ids)
                return selected.bounds
            except Exception:
                return None

        return None

    def _apply_camera_preset_for_bounds(
        self,
        bounds: tuple[float, float, float, float, float, float],
        preset: str,
    ) -> None:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        center = np.array(
            [
                0.5 * (xmin + xmax),
                0.5 * (ymin + ymax),
                0.5 * (zmin + zmax),
            ],
            dtype=float,
        )

        extents = np.array(
            [
                max(xmax - xmin, 1e-6),
                max(ymax - ymin, 1e-6),
                max(zmax - zmin, 1e-6),
            ],
            dtype=float,
        )
        diagonal = float(np.linalg.norm(extents))
        distance = max(diagonal * 1.35, 1.25)

        direction_map = {
            "isometric": np.array([1.55, -1.35, 0.95], dtype=float),
            "front": np.array([0.0, -1.0, 0.0], dtype=float),
            "back": np.array([0.0, 1.0, 0.0], dtype=float),
            "left": np.array([-1.0, 0.0, 0.0], dtype=float),
            "right": np.array([1.0, 0.0, 0.0], dtype=float),
            "top": np.array([0.0, 0.0, 1.0], dtype=float),
            "bottom": np.array([0.0, 0.0, -1.0], dtype=float),
        }
        direction = direction_map[preset]
        direction /= np.linalg.norm(direction)

        position = center + direction * distance

        camera = self.plotter.camera
        camera.SetFocalPoint(*center)
        camera.SetPosition(*position)

        if preset in {"top", "bottom"}:
            camera.SetViewUp(0.0, 1.0, 0.0)
        else:
            camera.SetViewUp(0.0, 0.0, 1.0)

        renderer = getattr(self.plotter, "renderer", None)
        if renderer is not None:
            try:
                renderer.ResetCameraClippingRange()
            except Exception:
                pass

        try:
            camera.Zoom(1.12)
        except Exception:
            pass

    def _apply_display_preset_to_flags(self, preset: str) -> None:
        data = self.DISPLAY_PRESETS[preset]
        self._show_grid = bool(data["show_grid"])
        self._show_axes = bool(data["show_axes"])
        self._show_edges = bool(data["show_edges"])

    def _apply_current_clip(self, dataset: pv.DataSet) -> pv.DataSet:
        if self._clip_axis is None or self._clip_value is None:
            return dataset.copy(deep=True)

        normal_map = {
            "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
        }
        origin_map = {
            "x": (self._clip_value, 0.0, 0.0),
            "y": (0.0, self._clip_value, 0.0),
            "z": (0.0, 0.0, self._clip_value),
        }

        try:
            clipped = dataset.clip(
                normal=normal_map[self._clip_axis],
                origin=origin_map[self._clip_axis],
                invert=self._clip_invert,
            )
            return clipped
        except Exception:
            return dataset.copy(deep=True)

    # ------------------------------------------------------------------
    # PyVista/WGPU parity helpers
    # ------------------------------------------------------------------
    def _configure_render_quality(self) -> None:
        """Best-effort visual quality setup for the compatibility backend."""
        for call in (
            lambda: self.plotter.enable_anti_aliasing("ssaa"),
            lambda: self.plotter.enable_depth_peeling(),
        ):
            try:
                call()
            except Exception:
                pass

        try:
            self.plotter.renderer.SetUseFXAA(True)
        except Exception:
            pass

    def _sync_topology_from_current_mesh(self) -> None:
        self._current_vertices = None
        self._current_faces = None
        self._open_edges = None
        self._edge_index_to_vertices = None
        self._edge_key_to_index = {}
        self._edge_to_faces = {}
        self._vertex_adjacency = {}

        if self._current_mesh_data is None:
            return
        try:
            vertices, faces = self._extract_vertices_faces(self._current_mesh_data)
        except Exception:
            return

        self._current_vertices = vertices
        self._current_faces = faces
        self._edge_index_to_vertices = self._build_unique_edges(faces)
        self._edge_key_to_index = {
            (int(a), int(b)) if int(a) <= int(b) else (int(b), int(a)): int(i)
            for i, (a, b) in enumerate(self._edge_index_to_vertices)
        }
        self._open_edges = self._find_open_edges(faces)
        self._edge_to_faces = self._build_edge_adjacency(faces)
        self._vertex_adjacency = self._build_vertex_adjacency(faces)

    def _extract_vertices_faces(self, dataset: pv.DataSet) -> tuple[np.ndarray, np.ndarray]:
        data = self._coerce_dataset(dataset)
        try:
            tri = data.triangulate()
        except Exception:
            tri = data
        vertices = np.asarray(tri.points, dtype=float)
        raw_faces = getattr(tri, "faces", None)
        faces = self._faces_from_pyvista_faces(raw_faces)
        return vertices, faces

    @staticmethod
    def _faces_from_pyvista_faces(raw_faces: Any) -> np.ndarray:
        arr = np.asarray(raw_faces, dtype=np.int64).ravel()
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.int32)
        out: list[tuple[int, int, int]] = []
        i = 0
        n = int(arr.size)
        while i < n:
            count = int(arr[i])
            i += 1
            if count <= 0 or i + count > n:
                break
            verts = [int(v) for v in arr[i : i + count]]
            i += count
            if len(verts) < 3:
                continue
            root = verts[0]
            for j in range(1, len(verts) - 1):
                out.append((root, verts[j], verts[j + 1]))
        if not out:
            return np.empty((0, 3), dtype=np.int32)
        return np.asarray(out, dtype=np.int32)

    @staticmethod
    def _build_unique_edges(faces: np.ndarray) -> np.ndarray:
        if faces.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).astype(np.int32, copy=False)
        edges = np.sort(edges, axis=1)
        return np.unique(edges, axis=0).astype(np.int32, copy=False)

    @staticmethod
    def _find_open_edges(faces: np.ndarray) -> np.ndarray:
        if faces.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).astype(np.int32, copy=False)
        edges_sorted = np.sort(edges, axis=1)
        _unique, inverse, counts = np.unique(edges_sorted, axis=0, return_inverse=True, return_counts=True)
        open_edge_indices = [i for i in range(len(edges_sorted)) if counts[inverse[i]] == 1]
        if not open_edge_indices:
            return np.empty((0, 2), dtype=np.int32)
        return edges[np.asarray(open_edge_indices, dtype=np.int32)]

    @staticmethod
    def _build_edge_adjacency(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
        edge_map: Dict[Tuple[int, int], List[int]] = {}
        for fi, tri in enumerate(faces):
            for e in ((int(tri[0]), int(tri[1])), (int(tri[1]), int(tri[2])), (int(tri[2]), int(tri[0]))):
                key = (e[0], e[1]) if e[0] <= e[1] else (e[1], e[0])
                edge_map.setdefault(key, []).append(int(fi))
        return edge_map

    @staticmethod
    def _build_vertex_adjacency(faces: np.ndarray) -> Dict[int, Set[int]]:
        adjacency: Dict[int, Set[int]] = {}
        for tri in faces:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            adjacency.setdefault(a, set()).update((b, c))
            adjacency.setdefault(b, set()).update((a, c))
            adjacency.setdefault(c, set()).update((a, b))
        return adjacency

    def set_edge_region_strategy(self, strategy: str | None) -> None:
        value = str(strategy or "safe").strip().lower()
        if value not in {"safe", "open_component", "feature", "ring", "aggressive", "single", "bore_rim"}:
            value = "safe"
        self._edge_region_strategy = value
        self.selection_changed.emit(self.get_selection_state())

    def get_edge_region_strategy(self) -> str:
        return str(getattr(self, "_edge_region_strategy", "safe") or "safe")

    def highlight_edges(self, edge_ids: list[int]) -> None:
        edge_count = len(self._edge_index_to_vertices) if self._edge_index_to_vertices is not None else 0
        selected = sorted({int(v) for v in edge_ids if 0 <= int(v) < edge_count})
        self._selected_edge_ids = selected
        self._selected_cell_ids = []
        self._selected_point_ids = []
        if selected:
            self._selection_mode = "edge"
        self._rebuild_edge_selection_actor()
        self.plotter.render()
        self.selection_changed.emit(self.get_selection_state())

    def _build_edge_selection_actor(self) -> None:
        if self._edge_selection_actor is not None:
            try:
                self.plotter.remove_actor(self._edge_selection_actor)
            except Exception:
                pass
            self._edge_selection_actor = None
        if not self._selected_edge_ids:
            return
        poly = self._polydata_from_edge_ids(self._selected_edge_ids)
        if poly is None or getattr(poly, "n_cells", 0) <= 0:
            return
        self._edge_selection_actor = self.plotter.add_mesh(
            poly,
            color=self.config.selection_color,
            line_width=max(3.0, float(self._edge_width) + 2.0),
            lighting=False,
            pickable=False,
            render_lines_as_tubes=True,
        )

    def _rebuild_edge_selection_actor(self) -> None:
        self._build_edge_selection_actor()

    def _polydata_from_edge_ids(self, edge_ids: list[int] | tuple[int, ...]) -> pv.PolyData | None:
        if self._current_vertices is None or self._edge_index_to_vertices is None:
            return None
        valid_edges: list[tuple[int, int]] = []
        for raw in edge_ids:
            idx = int(raw)
            if idx < 0 or idx >= len(self._edge_index_to_vertices):
                continue
            a, b = self._edge_index_to_vertices[idx]
            valid_edges.append((int(a), int(b)))
        if not valid_edges:
            return None
        poly = pv.PolyData(self._current_vertices)
        lines = np.empty((len(valid_edges), 3), dtype=np.int64)
        for i, (a, b) in enumerate(valid_edges):
            lines[i] = (2, a, b)
        poly.lines = lines.ravel()
        return poly

    def _resolve_edge_index_from_world_pos(self, pos: tuple[float, float, float]) -> int | None:
        if self._current_mesh_data is None or self._current_faces is None:
            return None
        try:
            face_index = int(self._current_mesh_data.find_closest_cell(pos))
        except Exception:
            face_index = -1
        info = {
            "face_index": face_index,
            "world_pos": tuple(float(v) for v in pos),
            "position": tuple(float(v) for v in pos),
            "point": tuple(float(v) for v in pos),
        }
        return resolve_edge_index_from_pick_info(
            info,
            faces=self._current_faces,
            edge_key_to_index=self._edge_key_to_index,
            vertices=self._current_vertices,
            fallback_face_index=face_index,
        )

    def _get_connected_edge_region(self, edge_index: int) -> set[int]:
        if self._current_vertices is None or self._edge_index_to_vertices is None:
            return {int(edge_index)}
        try:
            region = select_edge_region(
                vertices=self._current_vertices,
                faces=self._current_faces,
                edge_index_to_vertices=self._edge_index_to_vertices,
                edge_to_faces=self._edge_to_faces,
                open_edges=self._open_edges,
                start_edge_index=int(edge_index),
                strategy=self._edge_region_strategy,
            )
            edge_ids = {int(v) for v in getattr(region, "edge_ids", ())}
            return edge_ids or {int(edge_index)}
        except Exception:
            return {int(edge_index)}

    def _apply_edge_pick(self, edge_index: int) -> None:
        if self._edge_index_to_vertices is None:
            return
        if edge_index < 0 or edge_index >= len(self._edge_index_to_vertices):
            return
        strategy = self.get_edge_region_strategy()
        if strategy in {"bore_rim", "ring", "aggressive", "open_component", "feature"}:
            selected = self._get_connected_edge_region(int(edge_index))
        else:
            selected = {int(edge_index)}
        self.highlight_edges(sorted(selected))
        self.status_changed.emit(f"Selected {len(self._selected_edge_ids)} edge(s).")

    def _grow_edge_selection(self) -> None:
        if not self._selected_edge_ids or self._edge_index_to_vertices is None:
            self.status_changed.emit("No edge selection to grow.")
            return
        grown = set(int(v) for v in self._selected_edge_ids)
        for edge_index in list(grown):
            grown.update(self._get_connected_edge_region(edge_index))
            try:
                a, b = self._edge_index_to_vertices[int(edge_index)]
                for i, candidate in enumerate(self._edge_index_to_vertices):
                    ca, cb = int(candidate[0]), int(candidate[1])
                    if ca in {int(a), int(b)} or cb in {int(a), int(b)}:
                        grown.add(int(i))
            except Exception:
                pass
        self.highlight_edges(sorted(grown))
        self.status_changed.emit(f"Edge selection grown to {len(grown)} edge(s).")

    def _shrink_edge_selection(self) -> None:
        if not self._selected_edge_ids:
            self.status_changed.emit("No edge selection to shrink.")
            return
        if len(self._selected_edge_ids) <= 1:
            self.highlight_edges([])
            self.status_changed.emit("Edge selection cleared.")
            return
        self.highlight_edges(self._selected_edge_ids[:-1])
        self.status_changed.emit(f"Edge selection shrunk to {len(self._selected_edge_ids)} edge(s).")

    # ------------------------------------------------------------------
    # picking helpers
    # ------------------------------------------------------------------
    def _enable_point_selection_internal(self) -> None:
        self.disable_picking()

        def _on_pick(picked: Any) -> None:
            if picked is None:
                return
            try:
                pos = tuple(float(v) for v in picked)
                self._last_picked_world_pos = pos
                self.point_picked.emit(pos)
                self.show_marker(pos, name="picked_point")
                if self._surface_pick_callback is not None:
                    self._surface_pick_callback(pos)
            except Exception:
                pass

        self.plotter.enable_surface_point_picking(
            callback=_on_pick,
            show_point=False,
            left_clicking=True,
            pickable_window=False,
        )
        self._picking_enabled = True

    def _enable_face_selection_internal(self) -> None:
        self.disable_picking()

        def _on_pick(picked: Any) -> None:
            if picked is None:
                return
            try:
                pos = tuple(float(v) for v in picked)
                self._last_picked_world_pos = pos
                self.point_picked.emit(pos)
                if self._display_mesh_data is not None:
                    closest_cell = self._display_mesh_data.find_closest_cell(pos)
                    if closest_cell >= 0:
                        self.highlight_cells([closest_cell])
                if self._surface_pick_callback is not None:
                    self._surface_pick_callback(pos)
            except Exception:
                pass

        self.plotter.enable_surface_point_picking(
            callback=_on_pick,
            show_point=False,
            left_clicking=True,
            pickable_window=False,
        )
        self._picking_enabled = True

    def _enable_edge_selection_internal(self) -> None:
        self.disable_picking()

        def _on_pick(picked: Any) -> None:
            if picked is None:
                return
            try:
                pos = tuple(float(v) for v in picked)
                self._last_picked_world_pos = pos
                self.point_picked.emit(pos)
                edge_index = self._resolve_edge_index_from_world_pos(pos)
                if edge_index is not None:
                    self._apply_edge_pick(edge_index)
                elif self._surface_pick_callback is not None:
                    self._surface_pick_callback(pos)
            except Exception:
                pass

        self.plotter.enable_surface_point_picking(
            callback=_on_pick,
            show_point=False,
            left_clicking=True,
            pickable_window=False,
        )
        self._picking_enabled = True

    def _enable_mesh_selection_internal(self) -> None:
        self.disable_picking()

        def _on_pick(picked: Any) -> None:
            self._selected_actor = picked
            self.selection_changed.emit(self.get_selection_state())

        self.plotter.enable_mesh_picking(
            callback=_on_pick,
            show=False,
            left_clicking=True,
            use_actor=False,
        )
        self._picking_enabled = True

    def _default_marker_radius(self) -> float:
        data = self._display_mesh_data or self._current_mesh_data
        if data is None:
            return getattr(self.config, "point_marker_min_radius", 0.03)
        xmin, xmax, ymin, ymax, zmin, zmax = data.bounds
        diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
        scale = getattr(self.config, "point_marker_scale", 0.01)
        min_radius = getattr(self.config, "point_marker_min_radius", 0.01)
        return max(diag * scale, min_radius)


__all__ = [
    "ViewportError",
    "PyVistaViewportConfig",
    "PyVistaViewport",
]
