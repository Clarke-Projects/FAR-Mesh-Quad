"""WGPU viewport implementation.

The viewport is a picking and display backend.  It may resolve the primitive
under the pointer and forward a generic edge-region strategy to the core
selection layer, but it does not own feature semantics.  BoreTool-specific
meaning starts after selection evidence is handed to the controller/tool stack.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from far_mesh.core.selection_edges import resolve_edge_index_from_pick_info, select_edge_region
from .viewport_config import ViewportConfig


class WgpuViewportError(RuntimeError):
    """Base error for WGPU viewport failures."""


WgpuViewportConfig = ViewportConfig


class _BaseRenderHost(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = "WGPU viewport host"
        self._info_lines: list[str] = []

    @property
    def backend_summary(self) -> str:
        return "unknown"

    @property
    def is_real_canvas(self) -> bool:
        return False

    def set_info(self, title: str, lines: list[str]) -> None:
        self._title = title
        self._info_lines = list(lines)
        self.update()

    def request_draw(self) -> None:
        self.update()

    def capture_image(self, output_path: str | Path) -> str:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        pixmap: QPixmap = self.grab()
        if pixmap.isNull():
            raise WgpuViewportError("Qt grab() returned an empty pixmap for WGPU viewport capture.")
        if not pixmap.save(str(out)):
            raise WgpuViewportError(f"Could not save WGPU viewport capture to {out}")
        return str(out)


class _FallbackRenderHost(_BaseRenderHost):
    def __init__(
        self,
        *,
        background_color: str,
        foreground_color: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._background_color = QColor(background_color)
        self._foreground_color = QColor(foreground_color)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @property
    def backend_summary(self) -> str:
        return "fallback_qt_painter_host"

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect()
        painter.fillRect(rect, self._background_color)

        grid_pen = QPen(QColor("#2c313a"))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)

        step = 28
        for x in range(0, rect.width(), step):
            painter.drawLine(x, 0, x, rect.height())
        for y in range(0, rect.height(), step):
            painter.drawLine(0, y, rect.width(), y)

        border_pen = QPen(QColor("#313843"))
        border_pen.setWidth(1)
        painter.setPen(border_pen)
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        body_font = QFont()
        body_font.setPointSize(10)

        painter.setPen(QPen(self._foreground_color))
        painter.setFont(title_font)

        title_y = max(42, rect.height() // 2 - 80)
        painter.drawText(
            rect.adjusted(24, title_y - 16, -24, 0),
            Qt.AlignmentFlag.AlignHCenter,
            self._title,
        )

        painter.setFont(body_font)
        info_text = "\n".join(self._info_lines) if self._info_lines else "No host information available."
        info_rect = rect.adjusted(48, title_y + 8, -48, -48)
        painter.drawText(
            info_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            info_text,
        )
        painter.end()


class _CanvasRenderHost(_BaseRenderHost):
    def __init__(
        self,
        *,
        canvas_widget: QWidget,
        backend_summary: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._canvas_widget = canvas_widget
        self._backend_summary = backend_summary

        self._footer = QLabel(self)
        self._footer.setWordWrap(True)
        self._footer.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._footer.setStyleSheet(
            """
            QLabel {
                color: #c6cfdb;
                background-color: #20242b;
                border-top: 1px solid #313843;
                padding: 8px;
                font-size: 11px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._canvas_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas_widget.setMouseTracking(True)
        layout.addWidget(self._canvas_widget, 1)
        layout.addWidget(self._footer, 0)

        self._info_panel_visible = True

    @property
    def backend_summary(self) -> str:
        return self._backend_summary

    @property
    def is_real_canvas(self) -> bool:
        return True

    def set_info(self, title: str, lines: list[str]) -> None:
        super().set_info(title, lines)
        self._footer.setText(f"{title}\n" + "\n".join(lines))

    def set_info_panel_visible(self, visible: bool) -> None:
        self._info_panel_visible = bool(visible)
        self._footer.setVisible(self._info_panel_visible)

    def is_info_panel_visible(self) -> bool:
        return bool(self._info_panel_visible)

    def request_draw(self) -> None:
        update = getattr(self._canvas_widget, "update", None)
        if callable(update):
            try:
                update()
            except Exception:
                pass
        super().request_draw()

    def capture_image(self, output_path: str | Path) -> str:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        pixmap = self._canvas_widget.grab()
        if pixmap.isNull():
            pixmap = self.grab()
        if pixmap.isNull():
            raise WgpuViewportError("Qt grab() returned an empty pixmap for WGPU canvas capture.")
        if not pixmap.save(str(out)):
            raise WgpuViewportError(f"Could not save WGPU canvas capture to {out}")
        return str(out)


class _OrientationOverlay(QWidget):
    """Bottom-left orientation marker that follows the active camera orientation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(96, 96)
        self._camera_rotation_inverse = np.eye(3, dtype=float)

    def set_camera_rotation_inverse(self, rotation_inverse: np.ndarray | None) -> None:
        if rotation_inverse is None:
            self._camera_rotation_inverse = np.eye(3, dtype=float)
            self.update()
            return
        arr = np.asarray(rotation_inverse, dtype=float)
        if arr.shape != (3, 3):
            return
        self._camera_rotation_inverse = arr.copy()
        self.update()

    def _axis_camera_vectors(self) -> list[tuple[np.ndarray, float, QColor, str]]:
        axis_specs = [
            (np.array([1.0, 0.0, 0.0], dtype=float), QColor("#ff9a3c"), "x"),
            (np.array([0.0, 1.0, 0.0], dtype=float), QColor("#7ad97a"), "y"),
            (np.array([0.0, 0.0, 1.0], dtype=float), QColor("#65a9ff"), "z"),
        ]
        items: list[tuple[np.ndarray, float, QColor, str]] = []
        for axis_world, color, label in axis_specs:
            axis_cam = self._camera_rotation_inverse @ axis_world
            items.append((axis_cam, float(axis_cam[2]), color, label))
        items.sort(key=lambda item: item[1])
        return items

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(20, 24, 30, 150))

        # Keep the visual background circle and the axis pivot locked to the
        # same center point.  The old overlay used a hard-coded lower-left
        # origin while the circle was centered elsewhere, which made the
        # dynamic axis marker look off-center inside the circle.
        circle_rect = QRectF(8, 8, 78, 78)
        painter.drawEllipse(circle_rect)

        circle_center = circle_rect.center()
        origin = np.array([float(circle_center.x()), float(circle_center.y())], dtype=float)
        scale = 24.0

        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)

        for axis_cam, _depth, color, label in self._axis_camera_vectors():
            end = origin + np.array([axis_cam[0], -axis_cam[1]], dtype=float) * scale
            vx, vy = end[0] - origin[0], end[1] - origin[1]
            length = max(float((vx * vx + vy * vy) ** 0.5), 1.0)
            ux, uy = vx / length, vy / length
            px, py = -uy, ux

            pen = QPen(color)
            pen.setWidth(3)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawLine(
                QPointF(float(origin[0]), float(origin[1])),
                QPointF(float(end[0]), float(end[1])),
            )

            arrow_len = 7.0
            arrow_w = 4.0
            tip = end
            left = end - np.array([ux, uy]) * arrow_len + np.array([px, py]) * arrow_w
            right = end - np.array([ux, uy]) * arrow_len - np.array([px, py]) * arrow_w
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(float(tip[0]), float(tip[1])),
                        QPointF(float(left[0]), float(left[1])),
                        QPointF(float(right[0]), float(right[1])),
                    ]
                )
            )

            painter.setPen(color)
            text_dx = 5 if vx >= 0 else -11
            text_dy = -4 if vy <= 0 else 12
            painter.drawText(int(end[0] + text_dx), int(end[1] + text_dy), label)

        painter.setPen(QPen(QColor("#d7dde6")))
        painter.setBrush(QColor("#d7dde6"))
        painter.drawEllipse(QRectF(float(origin[0] - 3), float(origin[1] - 3), 6, 6))
        painter.end()

    def reposition(self, host_rect) -> None:
        margin = 12
        self.move(margin, max(margin, host_rect.height() - self.height() - 28))


def _discover_qt_wgpu_canvas() -> tuple[Any | None, QWidget | None, str, list[str]]:
    notes: list[str] = []

    if os.environ.get("FAR_MESH_WGPU_DISABLE_CANVAS", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None, None, "disabled_by_env", ["Qt WGPU canvas discovery disabled by FAR_MESH_WGPU_DISABLE_CANVAS."]

    try:
        import PySide6  # noqa: F401
    except Exception as exc:
        return None, None, "pyside6_unavailable", [f"PySide6 import failed: {exc!r}"]

    try:
        rcqt = importlib.import_module("rendercanvas.qt")
    except Exception as exc:
        return None, None, "rendercanvas_qt_unavailable", [f"rendercanvas.qt import failed: {exc!r}"]

    for class_name in ("QRenderWidget", "QRenderCanvas", "RenderCanvas"):
        cls = getattr(rcqt, class_name, None)
        if cls is None:
            notes.append(f"{class_name}: not present")
            continue
        try:
            obj = cls()
        except Exception as exc:
            notes.append(f"{class_name}: instantiation failed: {exc!r}")
            continue

        if isinstance(obj, QWidget):
            return obj, obj, f"rendercanvas.qt.{class_name}", notes

        for attr_name in ("widget", "_widget", "native", "qwidget"):
            widget = getattr(obj, attr_name, None)
            if isinstance(widget, QWidget):
                return obj, widget, f"rendercanvas.qt.{class_name}.{attr_name}", notes

        notes.append(f"{class_name}: instantiated object is not QWidget-embeddable")

    return None, None, "no_rendercanvas_qt_widget_found", notes


class WgpuViewport(QWidget):
    BACKEND_NAME = "wgpu"

    status_changed = Signal(str)
    mesh_loaded = Signal(str)
    mesh_failed = Signal(str)
    point_picked = Signal(tuple)
    selection_changed = Signal(object)
    compare_mode_changed = Signal(str)

    DISPLAY_PRESETS: dict[str, dict[str, Any]] = {
        "viewer_clean": {"solid": True, "wire": False},
        "inspection_edges": {"solid": True, "wire": True},
        "repair_selection": {"solid": True, "wire": True},
        "shaded_only": {"solid": True, "wire": False},
        "shaded + wireframe": {"solid": True, "wire": True},
        "wireframe": {"solid": False, "wire": True},
    }

    CAMERA_PRESETS = {"isometric", "front", "back", "left", "right", "top", "bottom"}
    COMPARE_MODES = {"current_only", "original_only", "overlay_ghost"}
    SELECTION_MODES = {"none", "point", "face", "edge", "mesh"}
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

    def __init__(self, parent: QWidget | None = None, *, config: ViewportConfig | None = None) -> None:
        super().__init__(parent)
        self.config = config or ViewportConfig()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._current_path: str | None = None
        self._current_mesh_source_name: str | None = None
        self._current_mesh_object: Any | None = None
        self._original_mesh_object: Any | None = None

        self._selection_mode: str = "none"
        self._compare_mode: str = "current_only"
        self._display_preset: str = "inspection_edges"
        self._camera_preset: str = "isometric"
        self._diagnostic_mode: str = "none"

        self._selected_cell_ids: Set[int] = set()
        self._selected_point_ids: Set[int] = set()
        self._selected_edge_ids: Set[int] = set()
        self._edge_region_strategy: str = "safe"
        self._hover_face: int | None = None
        self._last_picked_world_pos: tuple[float, float, float] | None = None
        self._last_hover_pick_info: dict[str, Any] | None = None
        self._last_edge_pick_seed: dict[str, Any] = {}

        self._show_edges = self.config.show_edges_default
        self._edge_width = self.config.edge_width_default
        self._show_grid = self.config.show_grid_default
        self._show_axes = self.config.show_axes_default
        self._show_boundary_edges = False
        self._show_host_info = False
        self._solid_visible = True
        self._wire_visible = bool(self._show_edges)

        self._clip_axis: str | None = None
        self._clip_value: float | None = None
        self._clip_invert: bool = False

        self._surface_pick_callback: Callable[[tuple[float, float, float]], None] | None = None

        self._brush_select_enabled = True

        self._brush_drag_active = False
        self._brush_drag_target: str | None = None
        self._brush_drag_toggle = False
        self._brush_drag_last_face: int | None = None
        self._brush_drag_last_point: int | None = None
        self._brush_drag_seen_faces: Set[int] = set()
        self._brush_drag_seen_points: Set[int] = set()

        self._host_notes: list[str] = []
        self._render_host: _BaseRenderHost | None = None
        self._orientation_overlay: _OrientationOverlay | None = None
        self._canvas_obj: Any | None = None
        self._canvas_widget: QWidget | None = None

        self._gfx: Any | None = None
        self._renderer: Any | None = None
        self._scene: Any | None = None
        self._camera: Any | None = None
        self._controller: Any | None = None
        self._mesh_node: Any | None = None
        self._wire_node: Any | None = None
        self._selection_node: Any | None = None
        self._selection_wire_node: Any | None = None
        self._point_selection_node: Any | None = None
        self._edge_selection_node: Any | None = None
        self._boundary_node: Any | None = None
        self._hover_node: Any | None = None
        self._original_mesh_node: Any | None = None
        self._grid_helper: Any | None = None
        self._overlay_objects: dict[str, Any] = {}
        self._preview_mesh_node: Any | None = None
        self._preview_mesh_wire_node: Any | None = None

        self._current_vertices: np.ndarray | None = None
        self._current_faces: np.ndarray | None = None
        self._open_edges: np.ndarray | None = None
        self._edge_index_to_vertices: np.ndarray | None = None
        self._edge_key_to_index: Dict[Tuple[int, int], int] = {}
        self._edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
        self._vertex_adjacency: Dict[int, Set[int]] = {}

        self._gfx_ready = False

        self._build_ui()
        self._initialize_gfx_runtime()
        self._refresh_host_state()
        state = self._render_host.backend_summary if self._render_host else "no-host"
        self.status_changed.emit(f"WGPU viewport initialized ({state}).")

    # ------------------------------------------------------------------
    # UI / host
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        canvas_obj, canvas_widget, backend_summary, notes = _discover_qt_wgpu_canvas()
        self._canvas_obj = canvas_obj
        self._canvas_widget = canvas_widget
        self._host_notes = list(notes)

        if canvas_widget is not None:
            self._render_host = _CanvasRenderHost(canvas_widget=canvas_widget, backend_summary=backend_summary, parent=self)
        else:
            self._render_host = _FallbackRenderHost(
                background_color=self.config.background_color,
                foreground_color=self.config.mesh_color,
                parent=self,
            )

        self._render_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._render_host, 1)

        self._orientation_overlay = _OrientationOverlay(self._render_host)
        self._orientation_overlay.reposition(self._render_host.rect())
        self._orientation_overlay.setVisible(bool(self._show_axes))
        self._orientation_overlay.raise_()

        if isinstance(self._render_host, _CanvasRenderHost):
            self._render_host.set_info_panel_visible(self._show_host_info)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._orientation_overlay is not None and self._render_host is not None:
            self._orientation_overlay.reposition(self._render_host.rect())
        self._request_draw()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if self._orientation_overlay is not None and self._render_host is not None:
            self._orientation_overlay.reposition(self._render_host.rect())
            self._orientation_overlay.raise_()
        self._request_draw()

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._reset_brush_drag()
        self._hover_face = None
        self._rebuild_hover_overlay()
        super().leaveEvent(event)

    def _append_host_note(self, note: str) -> None:
        if note not in self._host_notes:
            self._host_notes.append(note)

    def _host_info_lines(self) -> list[str]:
        mesh_state = "loaded" if self._current_mesh_object is not None else "empty"
        source = self._current_mesh_source_name or "none"
        lines = [
            f"Backend host: {self._render_host.backend_summary if self._render_host else 'unknown'}",
            f"Mesh state: {mesh_state}",
            f"Source: {source}",
            f"Compare mode: {self._compare_mode}",
            f"Selection mode: {self._selection_mode}",
            f"Display preset: {self._display_preset}",
            f"Camera preset: {self._camera_preset}",
            f"Brush selection: {'on' if self._brush_select_enabled else 'off'}",
            f"Diagnostic mode: {self._diagnostic_mode}",
        ]
        lines.append("Rendering pipeline: pygfx + wgpu active" if self._gfx_ready else "Rendering pipeline: bootstrap host only")
        if self._host_notes:
            lines.extend(["", "Discovery notes:"])
            for note in self._host_notes[:8]:
                lines.append(f"- {note}")
        return lines

    def _refresh_host_state(self) -> None:
        if self._render_host is None:
            return

        title = "WGPU viewport host (pygfx active)" if self._gfx_ready else (
            "WGPU viewport host (rendercanvas attached)" if self._render_host.is_real_canvas else "WGPU viewport host (diagnostic fallback)"
        )
        self._render_host.set_info(title, self._host_info_lines())

        if isinstance(self._render_host, _CanvasRenderHost):
            self._render_host.set_info_panel_visible(self._show_host_info)

        if self._orientation_overlay is not None:
            self._orientation_overlay.reposition(self._render_host.rect())
            self._orientation_overlay.raise_()
            self._orientation_overlay.setVisible(bool(self._show_axes))
            self._update_orientation_overlay()

        self._render_host.request_draw()

    def _update_orientation_overlay(self) -> None:
        if self._orientation_overlay is None:
            return
        if self._camera is None:
            self._orientation_overlay.set_camera_rotation_inverse(None)
            return

        world = getattr(self._camera, "world", None)
        if world is None:
            self._orientation_overlay.set_camera_rotation_inverse(None)
            return

        rotation_inverse: np.ndarray | None = None
        try:
            inv = np.asarray(world.inverse_matrix, dtype=float)
            if inv.shape[0] >= 3 and inv.shape[1] >= 3:
                rotation_inverse = inv[:3, :3]
        except Exception:
            rotation_inverse = None

        if rotation_inverse is None:
            try:
                rot = np.asarray(world.rotation_matrix, dtype=float)
                if rot.shape == (3, 3):
                    rotation_inverse = rot.T
            except Exception:
                rotation_inverse = None

        self._orientation_overlay.set_camera_rotation_inverse(rotation_inverse)

    def set_host_info_visible(self, visible: bool) -> None:
        self._show_host_info = bool(visible)
        self._refresh_host_state()

    def toggle_host_info_visible(self) -> None:
        self.set_host_info_visible(not self._show_host_info)

    def is_host_info_visible(self) -> bool:
        return bool(self._show_host_info)

    # ------------------------------------------------------------------
    # Renderer bootstrap
    # ------------------------------------------------------------------
    def _initialize_gfx_runtime(self) -> None:
        if self._canvas_obj is None:
            self._append_host_note("No Qt WGPU canvas available; using diagnostic fallback.")
            return

        try:
            gfx = importlib.import_module("pygfx")
        except Exception as exc:
            self._append_host_note(f"pygfx import failed: {exc!r}")
            return

        try:
            gfx.renderers.wgpu.enable_wgpu_features("!float32-filterable")
        except Exception as exc:
            self._append_host_note(f"enable_wgpu_features failed: {exc!r}")

        try:
            renderer = gfx.WgpuRenderer(self._canvas_obj)
            scene = gfx.Scene()
            camera = gfx.PerspectiveCamera(50, 1.0)
            controller = gfx.OrbitController(camera, register_events=renderer)
            self._controller = controller

            try:
                renderer.add_event_handler(self._on_renderer_click, "click")
            except Exception:
                pass
            try:
                renderer.add_event_handler(self._on_renderer_pointer_up, "pointer_up")
            except Exception:
                pass

            try:
                grid = gfx.GridHelper(size=10.0, divisions=20, color1="#2c313a", color2="#3a4150")
                grid.local.euler_x = float(np.pi / 2.0)
                grid.local.position = (0, 0, -0.01)
                grid.visible = self._show_grid
                scene.add(grid)
                self._grid_helper = grid
            except Exception:
                self._grid_helper = None

            try:
                scene.add(gfx.AmbientLight())
            except Exception:
                pass

            try:
                light = gfx.DirectionalLight()
                light.local.position = (3.0, -5.0, 6.0)
                scene.add(light)
            except Exception:
                pass

            self._gfx = gfx
            self._renderer = renderer
            self._scene = scene
            self._camera = camera
            self._gfx_ready = True
            self._request_draw()
        except Exception as exc:
            self._append_host_note(f"pygfx renderer initialization failed: {exc!r}")
            self._gfx_ready = False

    def _request_draw(self) -> None:
        if not self._gfx_ready or self._canvas_obj is None:
            if self._render_host is not None:
                self._render_host.request_draw()
            return

        request_draw = getattr(self._canvas_obj, "request_draw", None)
        if callable(request_draw):
            try:
                request_draw(self._draw_frame)
                return
            except TypeError:
                try:
                    request_draw()
                except Exception:
                    pass
            except Exception:
                pass

        self._draw_frame()

    def _draw_frame(self) -> None:
        if not self._gfx_ready or self._renderer is None or self._scene is None or self._camera is None:
            return
        try:
            if self._canvas_widget is not None:
                width = max(1, self._canvas_widget.width())
                height = max(1, self._canvas_widget.height())
                if hasattr(self._camera, "aspect"):
                    self._camera.aspect = float(width) / float(height)

            self._update_orientation_overlay()
            self._renderer.render(self._scene, self._camera)
        except Exception as exc:
            self._append_host_note(f"draw failed: {exc!r}")
            self._gfx_ready = False
            self._refresh_host_state()

    def _set_controller_enabled(self, enabled: bool) -> None:
        if self._controller is None:
            return
        for attr_name in ("enabled", "is_enabled"):
            if hasattr(self._controller, attr_name):
                try:
                    setattr(self._controller, attr_name, bool(enabled))
                    return
                except Exception:
                    pass

    def _reset_brush_drag(self) -> None:
        self._brush_drag_active = False
        self._brush_drag_target = None
        self._brush_drag_toggle = False
        self._brush_drag_last_face = None
        self._brush_drag_last_point = None
        self._brush_drag_seen_faces.clear()
        self._brush_drag_seen_points.clear()
        self._set_controller_enabled(True)

    # ------------------------------------------------------------------
    # Mesh data helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _triangulate_faces_array(faces_raw: np.ndarray) -> np.ndarray:
        if faces_raw.ndim != 1:
            raise WgpuViewportError("Expected a 1D faces array for VTK/PyVista-style mesh input.")

        triangles: list[list[int]] = []
        i = 0
        total = int(faces_raw.size)
        while i < total:
            n = int(faces_raw[i])
            if n < 3:
                i += 1 + n
                continue
            verts = faces_raw[i + 1: i + 1 + n].astype(np.int32, copy=False)
            if verts.size != n:
                raise WgpuViewportError("Malformed VTK/PyVista faces array while triangulating.")
            for j in range(1, n - 1):
                triangles.append([int(verts[0]), int(verts[j]), int(verts[j + 1])])
            i += 1 + n

        if not triangles:
            raise WgpuViewportError("No triangulatable faces found in mesh input.")
        return np.asarray(triangles, dtype=np.int32)

    def _extract_vertices_faces(self, mesh: Any) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces)

            if faces.ndim == 1:
                faces = self._triangulate_faces_array(faces.astype(np.int64, copy=False))
            elif faces.ndim == 2 and faces.shape[1] > 3:
                tri_list = []
                for row in faces:
                    row = np.asarray(row, dtype=np.int32)
                    for j in range(1, row.shape[0] - 1):
                        tri_list.append([int(row[0]), int(row[j]), int(row[j + 1])])
                faces = np.asarray(tri_list, dtype=np.int32)
            else:
                faces = faces.astype(np.int32, copy=False)

            if vertices.ndim != 2 or vertices.shape[1] != 3:
                raise WgpuViewportError("Expected vertices shaped (N, 3).")
            if faces.ndim != 2 or faces.shape[1] != 3:
                raise WgpuViewportError("Expected triangle faces shaped (M, 3).")
            return vertices, faces

        if hasattr(mesh, "points") and hasattr(mesh, "faces"):
            vertices = np.asarray(mesh.points, dtype=np.float32)
            faces_raw = np.asarray(mesh.faces)
            faces = self._triangulate_faces_array(faces_raw.astype(np.int64, copy=False))
            if vertices.ndim != 2 or vertices.shape[1] != 3:
                raise WgpuViewportError("Expected points shaped (N, 3).")
            return vertices, faces

        raise WgpuViewportError(f"Unsupported mesh input type for WGPU conversion: {type(mesh)!r}")

    @staticmethod
    def _normalize_color_rows(arr: Any) -> Optional[np.ndarray]:
        if arr is None:
            return None
        try:
            arr = np.asarray(arr)
        except Exception:
            return None
        if arr.ndim != 2 or arr.shape[1] < 3:
            return None
        arr = arr[:, :4]
        if arr.dtype.kind == "f":
            if float(np.nanmax(arr)) <= 1.0 + 1e-6:
                arr = np.clip(arr, 0.0, 1.0)
            else:
                arr = np.clip(arr, 0.0, 255.0) / 255.0
        else:
            arr = np.clip(arr, 0, 255).astype(np.float32) / 255.0
        if arr.shape[1] == 3:
            alpha = np.ones((arr.shape[0], 1), dtype=np.float32)
            arr = np.hstack([arr.astype(np.float32), alpha])
        return arr.astype(np.float32, copy=False)

    def _remove_node(self, node_attr: str) -> None:
        node = getattr(self, node_attr, None)
        if self._scene is None or node is None:
            setattr(self, node_attr, None)
            return
        try:
            self._scene.remove(node)
        except Exception:
            pass
        setattr(self, node_attr, None)

    def _fit_camera_to_bounds(self, vertices: np.ndarray) -> None:
        if self._camera is None or vertices.size == 0:
            return

        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        center = 0.5 * (mins + maxs)
        extents = np.maximum(maxs - mins, 1e-6)
        diagonal = float(np.linalg.norm(extents))
        radius = max(diagonal * 0.5, 1e-3)

        direction_map = {
            "isometric": np.array([1.55, -1.35, 0.95], dtype=np.float32),
            "front": np.array([0.0, -1.0, 0.0], dtype=np.float32),
            "back": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "left": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            "right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "top": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "bottom": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        }

        direction = direction_map.get(self._camera_preset, direction_map["isometric"]).astype(np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            direction = direction_map["isometric"]
            norm = float(np.linalg.norm(direction))
        direction = direction / norm

        try:
            show_object = getattr(self._camera, "show_object", None)
            if callable(show_object):
                show_object(
                    (float(center[0]), float(center[1]), float(center[2]), float(radius)),
                    view_dir=(float(direction[0]), float(direction[1]), float(direction[2])),
                    up=(0.0, 0.0, 1.0),
                    scale=1.15,
                )
                return
        except Exception as exc:
            self._append_host_note(f"camera show_object failed: {exc!r}")

        distance = max(diagonal * 1.8, 1.25)
        position = center + direction * distance
        try:
            self._camera.local.position = tuple(float(v) for v in position)
            self._camera.look_at(tuple(float(v) for v in center))
        except Exception as exc:
            self._append_host_note(f"camera fit fallback failed: {exc!r}")

    def _build_mesh_geometry_and_material(self, mesh: Any, vertices: np.ndarray, faces: np.ndarray) -> tuple[Any, Any]:
        gfx = self._gfx
        if gfx is None:
            raise WgpuViewportError("pygfx is not initialized.")

        geometry_kwargs: dict[str, Any] = {
            "positions": vertices.astype(np.float32, copy=False),
            "indices": faces.astype(np.int32, copy=False),
        }

        try:
            if hasattr(mesh, "vertex_normals"):
                vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
                if vertex_normals.shape == vertices.shape:
                    geometry_kwargs["normals"] = vertex_normals
        except Exception:
            pass

        visual = getattr(mesh, "visual", None)

        if visual is not None and getattr(visual, "kind", None) == "texture":
            uv = getattr(visual, "uv", None)
            mat = getattr(visual, "material", None)
            image = getattr(mat, "image", None) if mat is not None else None
            if uv is not None and image is not None:
                uv = np.asarray(uv, dtype=np.float32)
                if uv.ndim == 2 and uv.shape[0] == len(vertices) and uv.shape[1] >= 2:
                    uv2 = uv[:, :2].copy()
                    uv2[:, 1] = 1.0 - uv2[:, 1]
                    geometry_kwargs["texcoords"] = uv2.astype(np.float32, copy=False)
                    tex_arr = np.asarray(image)
                    texture = gfx.Texture(tex_arr, dim=2)
                    geometry = gfx.Geometry(**geometry_kwargs)
                    material = gfx.MeshPhongMaterial(map=texture, color="#ffffff", pick_write=True)
                    return geometry, material

        face_colors = self._normalize_color_rows(getattr(visual, "face_colors", None) if visual is not None else None)
        if face_colors is not None and len(face_colors) == len(faces):
            geometry_kwargs["colors"] = face_colors
            geometry = gfx.Geometry(**geometry_kwargs)
            material = gfx.MeshPhongMaterial(color_mode="face", pick_write=True)
            return geometry, material

        vertex_colors = self._normalize_color_rows(getattr(visual, "vertex_colors", None) if visual is not None else None)
        if vertex_colors is not None and len(vertex_colors) == len(vertices):
            geometry_kwargs["colors"] = vertex_colors
            geometry = gfx.Geometry(**geometry_kwargs)
            material = gfx.MeshPhongMaterial(color_mode="vertex", pick_write=True)
            return geometry, material

        geometry = gfx.Geometry(**geometry_kwargs)
        material = gfx.MeshPhongMaterial(color=self.config.mesh_color, pick_write=True)
        return geometry, material

    def _apply_clip_plane_to_material(self, material: Any) -> None:
        try:
            if self._clip_axis is None or self._clip_value is None:
                material.clip_planes = []
                return

            normal = [0.0, 0.0, 0.0]
            normal[{"x": 0, "y": 1, "z": 2}[self._clip_axis]] = 1.0
            distance = -float(self._clip_value)

            if self._clip_invert:
                normal = [-float(v) for v in normal]
                distance = -distance

            material.clip_planes = [(*normal, distance)]
        except Exception:
            pass

    def _update_floor_grid(self, vertices: np.ndarray) -> None:
        if self._grid_helper is None:
            return

        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        center = 0.5 * (mins + maxs)
        extents = np.maximum(maxs - mins, 1e-6)
        span = float(max(extents[0], extents[2], 1.0))
        size = max(span * 2.2, 2.0)
        divisions = int(np.clip(np.ceil(span * 8), 12, 48))
        if divisions % 2:
            divisions += 1

        try:
            gfx = self._gfx
            if gfx is None or self._scene is None:
                return

            old_visible = bool(getattr(self._grid_helper, "visible", True))
            try:
                self._scene.remove(self._grid_helper)
            except Exception:
                pass

            grid = gfx.GridHelper(
                size=size,
                divisions=divisions,
                color1="#3c4658",
                color2="#293243",
                thickness=1,
            )
            try:
                grid.local.euler_x = float(np.pi / 2.0)
                grid.local.position = (
                    float(center[0]),
                    float(center[1]),
                    float(mins[2] - 0.04 * max(extents[2], 1.0)),
                )
            except Exception:
                pass

            grid.visible = self._show_grid and old_visible
            self._scene.add(grid)
            self._grid_helper = grid
        except Exception as exc:
            self._append_host_note(f"grid update failed: {exc!r}")

    @staticmethod
    def _build_unique_edges(faces: np.ndarray) -> np.ndarray:
        if faces.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        e01 = faces[:, [0, 1]]
        e12 = faces[:, [1, 2]]
        e20 = faces[:, [2, 0]]
        edges = np.vstack([e01, e12, e20]).astype(np.int32, copy=False)
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        return edges

    @staticmethod
    def _find_open_edges(faces: np.ndarray) -> np.ndarray:
        if faces.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        e01 = faces[:, [0, 1]]
        e12 = faces[:, [1, 2]]
        e20 = faces[:, [2, 0]]
        edges = np.vstack([e01, e12, e20]).astype(np.int32, copy=False)
        edges_sorted = np.sort(edges, axis=1)
        unique, inverse, counts = np.unique(edges_sorted, axis=0, return_inverse=True, return_counts=True)

        open_edge_indices: list[int] = []
        for i in range(len(edges_sorted)):
            if counts[inverse[i]] == 1:
                open_edge_indices.append(i)

        if not open_edge_indices:
            return np.empty((0, 2), dtype=np.int32)

        return edges[np.asarray(open_edge_indices, dtype=np.int32)]

    def _build_edge_adjacency(self, faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
        edge_map: Dict[Tuple[int, int], List[int]] = {}
        for fi, tri in enumerate(faces):
            for e in ((int(tri[0]), int(tri[1])), (int(tri[1]), int(tri[2])), (int(tri[2]), int(tri[0]))):
                key = tuple(sorted(e))
                edge_map.setdefault(key, []).append(fi)
        return edge_map

    def _build_vertex_adjacency(self, faces: np.ndarray) -> Dict[int, Set[int]]:
        adjacency: Dict[int, Set[int]] = {}
        for tri in faces:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            adjacency.setdefault(a, set()).update((b, c))
            adjacency.setdefault(b, set()).update((a, c))
            adjacency.setdefault(c, set()).update((a, b))
        return adjacency

    # ------------------------------------------------------------------
    # Overlay rebuilds
    # ------------------------------------------------------------------
    def _rebuild_wire_node(self) -> None:
        self._remove_node("_wire_node")
        if not self._gfx_ready or self._scene is None or self._gfx is None:
            return
        if self._current_vertices is None or self._current_faces is None:
            return

        edges = self._build_unique_edges(self._current_faces)
        if len(edges) == 0:
            return

        positions = self._current_vertices[edges.reshape(-1)].astype(np.float32, copy=False)
        geometry = self._gfx.Geometry(positions=positions)
        material = self._gfx.LineSegmentMaterial(
            color=self.config.wire_color,
            thickness=max(1.0, float(self._edge_width)),
        )
        node = self._gfx.Line(geometry, material)
        try:
            node.render_order = 2
        except Exception:
            pass
        self._scene.add(node)
        self._wire_node = node

    def _rebuild_boundary_node(self) -> None:
        self._remove_node("_boundary_node")
        if not self._gfx_ready or self._scene is None or self._gfx is None:
            return
        if self._current_vertices is None or self._current_faces is None:
            return
        if not self._show_boundary_edges:
            return

        if self._open_edges is None or len(self._open_edges) == 0:
            return

        positions = self._current_vertices[self._open_edges.reshape(-1)].astype(np.float32, copy=False)
        geometry = self._gfx.Geometry(positions=positions)
        material = self._gfx.LineMaterial(
            color=self.config.boundary_color,
            thickness=max(2.5, float(self._edge_width) + 2.0),
        )
        node = self._gfx.Line(geometry, material)
        try:
            node.render_order = 10
        except Exception:
            pass
        self._scene.add(node)
        self._boundary_node = node

    def _rebuild_selection_overlay(self) -> None:
        self._remove_node("_selection_node")
        self._remove_node("_selection_wire_node")
        if not self._gfx_ready or self._scene is None or self._gfx is None:
            return
        if self._current_vertices is None or self._current_faces is None:
            return
        if not self._selected_cell_ids:
            return

        valid = [i for i in self._selected_cell_ids if 0 <= i < len(self._current_faces)]
        if not valid:
            return

        faces = self._current_faces[np.asarray(valid, dtype=np.int32)]
        geometry = self._gfx.Geometry(
            positions=self._current_vertices.astype(np.float32, copy=False),
            indices=faces.astype(np.int32, copy=False),
        )

        material = self._gfx.MeshBasicMaterial(
            color=self.config.selection_color,
            opacity=0.78,
            pick_write=False,
        )
        try:
            material.side = "both"
        except Exception:
            pass

        node = self._gfx.Mesh(geometry, material)
        try:
            node.render_order = 20
        except Exception:
            pass
        self._scene.add(node)
        self._selection_node = node

        edge_positions = []
        verts = self._current_vertices
        for tri in faces:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            va = verts[a]
            vb = verts[b]
            vc = verts[c]
            edge_positions.extend([va, vb, vb, vc, vc, va])

        if edge_positions:
            edge_positions_arr = np.asarray(edge_positions, dtype=np.float32)
            line_geometry = self._gfx.Geometry(positions=edge_positions_arr)
            line_material = self._gfx.LineMaterial(
                color=self.config.selection_color,
                thickness=max(2.5, float(self._edge_width) + 1.5),
            )
            line_node = self._gfx.Line(line_geometry, line_material)
            try:
                line_node.render_order = 21
            except Exception:
                pass
            self._scene.add(line_node)
            self._selection_wire_node = line_node

    def _rebuild_point_selection_overlay(self) -> None:
        self._remove_node("_point_selection_node")
        if not self._gfx_ready or self._scene is None or self._gfx is None:
            return
        if self._current_vertices is None:
            return
        if not self._selected_point_ids:
            return

        valid = [i for i in self._selected_point_ids if 0 <= i < len(self._current_vertices)]
        if not valid:
            return

        positions = self._current_vertices[np.asarray(valid, dtype=np.int32)].astype(np.float32, copy=False)
        geometry = self._gfx.Geometry(positions=positions)
        material = self._gfx.PointsMaterial(
            color=self.config.selection_color,
            size=10.0,
            pick_write=False,
        )
        node = self._gfx.Points(geometry, material)
        try:
            node.render_order = 22
        except Exception:
            pass
        self._scene.add(node)
        self._point_selection_node = node

    def _rebuild_edge_selection_overlay(self) -> None:
        self._remove_node("_edge_selection_node")
        if not self._gfx_ready or self._scene is None or self._gfx is None:
            return
        if self._current_vertices is None or self._edge_index_to_vertices is None:
            return
        if not self._selected_edge_ids:
            return

        edge_count = len(self._edge_index_to_vertices)
        valid = [int(i) for i in self._selected_edge_ids if 0 <= int(i) < edge_count]
        if not valid:
            return

        edges = self._edge_index_to_vertices[np.asarray(valid, dtype=np.int32)]
        positions = self._current_vertices[edges.reshape(-1)].astype(np.float32, copy=False)
        geometry = self._gfx.Geometry(positions=positions)
        try:
            material = self._gfx.LineSegmentMaterial(
                color=self.config.selection_color,
                thickness=max(3.0, float(self._edge_width) + 3.0),
                pick_write=False,
            )
        except Exception:
            material = self._gfx.LineMaterial(
                color=self.config.selection_color,
                thickness=max(3.0, float(self._edge_width) + 3.0),
            )
        node = self._gfx.Line(geometry, material)
        try:
            node.render_order = 24
        except Exception:
            pass
        self._scene.add(node)
        self._edge_selection_node = node

    def _rebuild_hover_overlay(self) -> None:
        self._remove_node("_hover_node")
        if not self._gfx_ready or self._scene is None or self._gfx is None:
            return
        if self._current_vertices is None or self._current_faces is None:
            return
        if self._hover_face is None or self._hover_face < 0 or self._hover_face >= len(self._current_faces):
            return

        face = self._current_faces[self._hover_face]
        verts = self._current_vertices
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        edge_positions = np.array(
            [
                verts[a], verts[b],
                verts[b], verts[c],
                verts[c], verts[a],
            ],
            dtype=np.float32,
        )

        geometry = self._gfx.Geometry(positions=edge_positions)
        material = self._gfx.LineMaterial(
            color="#ffff00",
            thickness=max(3.0, float(self._edge_width) + 1.0),
        )
        node = self._gfx.Line(geometry, material)
        try:
            node.render_order = 15
        except Exception:
            pass
        self._scene.add(node)
        self._hover_node = node

    def _update_display_visibility(self) -> None:
        show_current = self._compare_mode != "original_only"

        if self._mesh_node is not None:
            self._mesh_node.visible = bool(show_current and self._solid_visible)
        if self._wire_node is not None:
            self._wire_node.visible = bool(show_current and self._wire_visible)
        if self._selection_node is not None:
            self._selection_node.visible = bool(show_current and bool(self._selected_cell_ids))
        if self._selection_wire_node is not None:
            self._selection_wire_node.visible = bool(show_current and bool(self._selected_cell_ids))
        if self._point_selection_node is not None:
            self._point_selection_node.visible = bool(show_current and bool(self._selected_point_ids))
        if self._edge_selection_node is not None:
            self._edge_selection_node.visible = bool(show_current and bool(self._selected_edge_ids))
        if self._boundary_node is not None:
            self._boundary_node.visible = bool(show_current and self._show_boundary_edges)
        if self._hover_node is not None:
            self._hover_node.visible = bool(show_current and self._hover_face is not None)
        if self._original_mesh_node is not None:
            self._original_mesh_node.visible = self._compare_mode in {"original_only", "overlay_ghost"}

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------
    def _resolve_vertex_index_from_pick_info(self, info: dict[str, Any]) -> int | None:
        vertex_index = info.get("vertex_index")
        if vertex_index is not None:
            try:
                return int(vertex_index)
            except Exception:
                pass

        if self._current_faces is None:
            return None

        face_index = info.get("face_index")
        if face_index is None:
            return None

        try:
            face_index = int(face_index)
        except Exception:
            return None

        if face_index < 0 or face_index >= len(self._current_faces):
            return None

        tri = self._current_faces[face_index]
        bary_raw = info.get("face_coord")
        if bary_raw is not None:
            try:
                bary = np.asarray(bary_raw, dtype=float).reshape(-1)
                if bary.size >= 3:
                    return int(tri[int(np.argmax(bary[:3]))])
            except Exception:
                pass

        return int(tri[0])

    def _resolve_edge_index_from_pick_info(self, info: dict[str, Any]) -> int | None:
        payload = info if isinstance(info, dict) else {}

        edge_index = resolve_edge_index_from_pick_info(
            payload,
            faces=self._current_faces,
            vertices=self._current_vertices,
            edge_key_to_index=self._edge_key_to_index,
            fallback_face_index=self._hover_face,
        )
        if edge_index is not None:
            return int(edge_index)

        hover_payload = getattr(self, "_last_hover_pick_info", None)
        if isinstance(hover_payload, dict) and hover_payload:
            edge_index = resolve_edge_index_from_pick_info(
                hover_payload,
                faces=self._current_faces,
                vertices=self._current_vertices,
                edge_key_to_index=self._edge_key_to_index,
                fallback_face_index=self._hover_face,
            )
            if edge_index is not None:
                return int(edge_index)

        return None

    def _update_pick_world_pos_from_face(self, face_index: int, info: dict[str, Any]) -> None:
        if self._current_faces is None or self._current_vertices is None:
            return
        if face_index < 0 or face_index >= len(self._current_faces):
            return
        try:
            tri = self._current_faces[face_index]
            tri_pos = self._current_vertices[tri[:3]].astype(float)

            bary_raw = info.get("face_coord")
            if bary_raw is not None:
                bary = np.asarray(bary_raw, dtype=float).reshape(-1)
                if bary.size >= 3:
                    pos = bary[:3] @ tri_pos
                else:
                    pos = tri_pos.mean(axis=0)
            else:
                pos = tri_pos.mean(axis=0)

            self._last_picked_world_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
            self.point_picked.emit(self._last_picked_world_pos)
            if self._surface_pick_callback is not None:
                self._surface_pick_callback(self._last_picked_world_pos)
        except Exception:
            pass

    def _update_pick_world_pos_from_vertex(self, vertex_index: int) -> None:
        if self._current_vertices is None:
            return
        if vertex_index < 0 or vertex_index >= len(self._current_vertices):
            return
        try:
            pos = self._current_vertices[vertex_index]
            self._last_picked_world_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
            self.point_picked.emit(self._last_picked_world_pos)
            if self._surface_pick_callback is not None:
                self._surface_pick_callback(self._last_picked_world_pos)
        except Exception:
            pass

    def _get_boundary_loop_faces(self, start_face: int) -> Set[int]:
        """Return all faces connected to start_face along the same open boundary loop."""
        if self._current_faces is None or self._open_edges is None or len(self._open_edges) == 0:
            return set()
        if start_face < 0 or start_face >= len(self._current_faces):
            return set()

        open_edge_keys = {tuple(sorted((int(e[0]), int(e[1])))) for e in self._open_edges}

        tri = self._current_faces[start_face]
        start_edge = None
        for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = tuple(sorted((int(e[0]), int(e[1]))))
            if key in open_edge_keys:
                start_edge = key
                break
        if start_edge is None:
            return set()

        open_edge_face_map: Dict[Tuple[int, int], int] = {}
        for edge_key in open_edge_keys:
            faces = self._edge_to_faces.get(edge_key, [])
            if len(faces) == 1:
                open_edge_face_map[edge_key] = faces[0]

        vertex_to_open_edges: Dict[int, List[Tuple[int, int]]] = {}
        for edge_key in open_edge_keys:
            a, b = edge_key
            vertex_to_open_edges.setdefault(a, []).append(edge_key)
            vertex_to_open_edges.setdefault(b, []).append(edge_key)

        visited_edges: Set[Tuple[int, int]] = set()
        stack = [start_edge]
        while stack:
            edge = stack.pop()
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            a, b = edge
            for v in (a, b):
                for neighbor_edge in vertex_to_open_edges.get(v, []):
                    if neighbor_edge not in visited_edges:
                        stack.append(neighbor_edge)

        selected_faces: Set[int] = set()
        for edge_key in visited_edges:
            face = open_edge_face_map.get(edge_key)
            if face is not None:
                selected_faces.add(face)
        return selected_faces

    def _get_connected_point_region(self, start_vertex: int) -> Set[int]:
        if self._current_vertices is None:
            return set()
        if start_vertex < 0 or start_vertex >= len(self._current_vertices):
            return set()

        visited: Set[int] = set()
        stack = [int(start_vertex)]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            for nv in self._vertex_adjacency.get(v, set()):
                if nv not in visited:
                    stack.append(nv)
        return visited

    def _apply_face_pick(self, face_index: int, modifiers: set[str], *, brush_drag: bool = False) -> None:
        if self._current_faces is None:
            return
        if face_index < 0 or face_index >= len(self._current_faces):
            return

        selected = set(self._selected_cell_ids)
        ctrl_pressed = bool({"Ctrl", "Control", "Meta"} & modifiers)

        if ctrl_pressed and not brush_drag and self._selection_mode == "face":
            loop_faces = self._get_boundary_loop_faces(face_index)
            if loop_faces:
                if loop_faces.issubset(selected):
                    selected.difference_update(loop_faces)
                else:
                    selected.update(loop_faces)
                self.highlight_cells(sorted(selected))
                return

        if self._brush_select_enabled:
            toggle_mode = self._brush_drag_toggle or ctrl_pressed
            if toggle_mode:
                if brush_drag:
                    if face_index in self._brush_drag_seen_faces:
                        return
                    self._brush_drag_seen_faces.add(face_index)
                if face_index in selected:
                    selected.remove(face_index)
                else:
                    selected.add(face_index)
            else:
                selected.add(face_index)
        else:
            shift = "Shift" in modifiers
            if ctrl_pressed:
                if face_index in selected:
                    selected.remove(face_index)
                else:
                    selected.add(face_index)
            elif shift:
                selected.add(face_index)
            else:
                selected = {face_index}

        self.highlight_cells(sorted(selected))

    def _apply_point_pick(self, vertex_index: int, modifiers: set[str], *, brush_drag: bool = False) -> None:
        if self._current_vertices is None:
            return
        if vertex_index < 0 or vertex_index >= len(self._current_vertices):
            return

        selected = set(self._selected_point_ids)
        ctrl_pressed = bool({"Ctrl", "Control", "Meta"} & modifiers)
        shift_pressed = "Shift" in modifiers
        alt_pressed = bool({"Alt", "Option"} & modifiers)

        if alt_pressed and not brush_drag:
            region = self._get_connected_point_region(vertex_index)
            if not region:
                return
            if ctrl_pressed:
                if region.issubset(selected):
                    selected.difference_update(region)
                else:
                    selected.update(region)
            elif shift_pressed:
                selected.update(region)
            else:
                selected = set(region)
            self.highlight_points(sorted(selected))
            return

        if self._brush_select_enabled:
            toggle_mode = self._brush_drag_toggle or ctrl_pressed
            if toggle_mode:
                if brush_drag:
                    if vertex_index in self._brush_drag_seen_points:
                        return
                    self._brush_drag_seen_points.add(vertex_index)
                if vertex_index in selected:
                    selected.remove(vertex_index)
                else:
                    selected.add(vertex_index)
            else:
                selected.add(vertex_index)
        else:
            if ctrl_pressed:
                if vertex_index in selected:
                    selected.remove(vertex_index)
                else:
                    selected.add(vertex_index)
            elif shift_pressed:
                selected.add(vertex_index)
            else:
                selected = {vertex_index}

        self.highlight_points(sorted(selected))

    def _edge_pick_seed_payload(
        self,
        edge_index: int,
        *,
        modifiers: set[str] | None = None,
        brush_drag: bool = False,
    ) -> dict[str, Any]:
        """Preserve the primitive the operator actually clicked.

        BoreTool/Region Select need this seed before ``bore_rim`` expansion turns
        a single click into a connected edge cloud.  This payload is not feature
        identity; it is raw pick evidence for seed-local rim normalization.
        """

        payload: dict[str, Any] = {
            "metadata_contract": "viewport_click_seed_primitive_v143",
            "backend": self.BACKEND_NAME,
            "selection_origin": "edge_pick_before_region_expansion",
            "seed_edge_id": int(edge_index),
            "clicked_edge_id": int(edge_index),
            "edge_region_strategy": str(self._edge_region_strategy),
            "brush_drag": bool(brush_drag),
            "modifiers": tuple(sorted(str(v) for v in (modifiers or set()))),
        }
        try:
            if self._edge_index_to_vertices is not None and 0 <= int(edge_index) < len(self._edge_index_to_vertices):
                a, b = self._edge_index_to_vertices[int(edge_index)]
                edge_vertices = (int(a), int(b))
                payload["seed_edge_vertex_ids"] = edge_vertices
                key = tuple(sorted(edge_vertices))
                faces = ()
                try:
                    if isinstance(self._edge_to_faces, dict):
                        faces = tuple(int(v) for v in self._edge_to_faces.get(key, ()))
                except Exception:
                    faces = ()
                payload["seed_adjacent_face_ids"] = faces
        except Exception:
            pass
        if self._last_picked_world_pos is not None:
            payload["seed_pick_point"] = tuple(float(v) for v in self._last_picked_world_pos)
        return payload

    def _get_connected_edge_region(self, start_edge_index: int) -> Set[int]:
        if self._current_vertices is None or self._edge_index_to_vertices is None:
            return set()
        try:
            strategy = str(getattr(self, "_edge_region_strategy", "safe") or "safe")
            # v173l: Bore opening Ctrl/Cmd-click must never expose the broad raw
            # connected edge cloud to the viewport, but the cap must not break
            # fine/high-resolution openings with many small segments.  The
            # measured-annular-rail guard in selection_edges validates circularity
            # and loop-like topology; this value is only a live traversal safety cap.
            region = select_edge_region(
                vertices=self._current_vertices,
                faces=self._current_faces,
                edge_index_to_vertices=self._edge_index_to_vertices,
                edge_to_faces=self._edge_to_faces,
                open_edges=self._open_edges,
                start_edge_index=int(start_edge_index),
                strategy=strategy,
                max_selected_edges=(768 if strategy == "bore_rim" else None),
            )
            return {int(v) for v in region.edge_ids}
        except Exception as exc:
            self._append_host_note(f"edge region selection failed: {exc!r}")
            if 0 <= int(start_edge_index) < len(self._edge_index_to_vertices):
                return {int(start_edge_index)}
            return set()

    def _apply_edge_pick(self, edge_index: int, modifiers: set[str], *, brush_drag: bool = False) -> None:
        if self._edge_index_to_vertices is None:
            return
        if edge_index < 0 or edge_index >= len(self._edge_index_to_vertices):
            return

        self._last_edge_pick_seed = self._edge_pick_seed_payload(
            int(edge_index),
            modifiers=modifiers,
            brush_drag=brush_drag,
        )

        selected = set(self._selected_edge_ids)
        ctrl_pressed = bool({"Ctrl", "Control", "Meta"} & modifiers)
        shift_pressed = "Shift" in modifiers

        if ctrl_pressed and not brush_drag:
            region = self._get_connected_edge_region(edge_index)
            if region:
                if region.issubset(selected):
                    selected.difference_update(region)
                else:
                    selected.update(region)
                self._last_edge_pick_seed["expanded_edge_count"] = int(len(selected))
                self.highlight_edges(sorted(selected))
                return

        if shift_pressed:
            selected.add(int(edge_index))
        elif ctrl_pressed:
            if int(edge_index) in selected:
                selected.remove(int(edge_index))
            else:
                selected.add(int(edge_index))
        else:
            selected = {int(edge_index)}

        self._last_edge_pick_seed["expanded_edge_count"] = int(len(selected))
        self.highlight_edges(sorted(selected))

    @staticmethod
    def _event_is_primary_pointer_button(event: object) -> bool:
        """Return False for secondary/right-click events so camera orbit remains untouched."""
        button = getattr(event, "button", None)
        if button is None:
            button = getattr(event, "buttons", None)
        if button is None:
            return True

        if isinstance(button, str):
            value = button.strip().lower()
            return value in {
                "",
                "0",
                "1",
                "left",
                "primary",
                "left_button",
                "leftbutton",
                "mouse_left",
                "mousebutton.leftbutton",
            }

        try:
            return int(button) in {0, 1}
        except Exception:
            return True


    # ------------------------------------------------------------------
    # Pointer events
    # ------------------------------------------------------------------
    def _on_mesh_pointer_down(self, event) -> None:
        if not self._event_is_primary_pointer_button(event):
            return

        info = getattr(event, "pick_info", None) or {}
        modifiers = set(getattr(event, "modifiers", ()) or ())

        if isinstance(info, dict) and info:
            self._last_hover_pick_info = dict(info)

        if self._selection_mode == "edge":
            edge_index = self._resolve_edge_index_from_pick_info(info)
            if edge_index is None:
                return

            face_index = info.get("face_index") if isinstance(info, dict) else None
            if face_index is None and isinstance(getattr(self, "_last_hover_pick_info", None), dict):
                face_index = self._last_hover_pick_info.get("face_index")
            if face_index is not None:
                try:
                    self._update_pick_world_pos_from_face(int(face_index), info if isinstance(info, dict) else {})
                except Exception:
                    pass

            self._apply_edge_pick(int(edge_index), modifiers)
            face_index = info.get("face_index") if isinstance(info, dict) else None
            if face_index is None and isinstance(getattr(self, "_last_hover_pick_info", None), dict):
                face_index = self._last_hover_pick_info.get("face_index")
            if face_index is not None:
                try:
                    self._update_pick_world_pos_from_face(int(face_index), info if isinstance(info, dict) else {})
                except Exception:
                    pass

            try:
                event.stop_propagation()
            except Exception:
                pass
            return

        if self._selection_mode == "point":
            vertex_index = self._resolve_vertex_index_from_pick_info(info)
            if vertex_index is None:
                return

            if self._brush_select_enabled:
                self._brush_drag_active = True
                self._brush_drag_target = "point"
                self._brush_drag_toggle = bool({"Ctrl", "Control", "Meta"} & modifiers)
                self._brush_drag_last_point = None
                self._brush_drag_seen_points.clear()
                self._set_controller_enabled(False)

            self._apply_point_pick(int(vertex_index), modifiers)
            self._update_pick_world_pos_from_vertex(int(vertex_index))

            if self._brush_select_enabled:
                self._brush_drag_last_point = int(vertex_index)
                if self._brush_drag_toggle:
                    self._brush_drag_seen_points.add(int(vertex_index))
                try:
                    event.stop_propagation()
                except Exception:
                    pass
            return

        if self._selection_mode != "face":
            return

        face_index = info.get("face_index")
        if face_index is None:
            return

        face_index = int(face_index)

        if self._brush_select_enabled:
            self._brush_drag_active = True
            self._brush_drag_target = "face"
            self._brush_drag_toggle = bool({"Ctrl", "Control", "Meta"} & modifiers)
            self._brush_drag_last_face = None
            self._brush_drag_seen_faces.clear()
            self._set_controller_enabled(False)

        self._apply_face_pick(face_index, modifiers)
        self._update_pick_world_pos_from_face(face_index, info)

        if self._brush_select_enabled:
            self._brush_drag_last_face = face_index
            if self._brush_drag_toggle:
                self._brush_drag_seen_faces.add(face_index)
            try:
                event.stop_propagation()
            except Exception:
                pass


    def _on_mesh_pointer_move(self, event) -> None:
        info = getattr(event, "pick_info", None) or {}
        if isinstance(info, dict) and info:
            self._last_hover_pick_info = dict(info)

        face_index = info.get("face_index")
        new_hover = int(face_index) if face_index is not None else None
        if new_hover != self._hover_face:
            self._hover_face = new_hover
            self._rebuild_hover_overlay()
            self._update_display_visibility()
            self._request_draw()

        if not self._brush_select_enabled or not self._brush_drag_active:
            return

        if self._brush_drag_target == "face":
            if self._selection_mode != "face":
                return
            if face_index is None:
                try:
                    event.stop_propagation()
                except Exception:
                    pass
                return
            face_index = int(face_index)
            if self._brush_drag_last_face == face_index:
                try:
                    event.stop_propagation()
                except Exception:
                    pass
                return
            self._apply_face_pick(face_index, set(), brush_drag=True)
            self._update_pick_world_pos_from_face(face_index, info)
            self._brush_drag_last_face = face_index
            try:
                event.stop_propagation()
            except Exception:
                pass
            return

        if self._brush_drag_target == "point":
            if self._selection_mode != "point":
                return
            vertex_index = self._resolve_vertex_index_from_pick_info(info)
            if vertex_index is None:
                try:
                    event.stop_propagation()
                except Exception:
                    pass
                return
            vertex_index = int(vertex_index)
            if self._brush_drag_last_point == vertex_index:
                try:
                    event.stop_propagation()
                except Exception:
                    pass
                return
            self._apply_point_pick(vertex_index, set(), brush_drag=True)
            self._update_pick_world_pos_from_vertex(vertex_index)
            self._brush_drag_last_point = vertex_index
            try:
                event.stop_propagation()
            except Exception:
                pass

    def _on_mesh_pointer_up(self, event) -> None:
        del event
        self._reset_brush_drag()

    def _on_renderer_pointer_up(self, event) -> None:
        del event
        if not self._brush_drag_active:
            return
        self._reset_brush_drag()

    def _on_renderer_click(self, event) -> None:
        if self._selection_mode not in {"face", "point", "edge"}:
            return
        if not self._event_is_primary_pointer_button(event):
            return
        modifiers = set(getattr(event, "modifiers", ()) or ())
        if {"Shift", "Ctrl", "Control", "Meta", "Alt", "Option"} & modifiers:
            return
        target = getattr(event, "target", None)
        if target is self._renderer and not self._brush_select_enabled:
            self.clear_selection()


    # ------------------------------------------------------------------
    # Mesh upload
    # ------------------------------------------------------------------
    def _upload_mesh(self, mesh: Any, *, source_name: str) -> None:
        self._current_mesh_source_name = source_name
        self._current_mesh_object = mesh

        self._selected_cell_ids.clear()
        self._selected_point_ids.clear()
        self._selected_edge_ids.clear()
        self._hover_face = None
        self._last_picked_world_pos = None
        self._last_hover_pick_info = None
        self._last_edge_pick_seed = {}
        self._reset_brush_drag()

        # Keep CPU-side topology available even when the real WGPU scene is not
        # active yet or the fallback Qt host is used.  Selection helpers such as
        # edge click resolution, grow_selection(), and shrink_selection() depend
        # on these arrays being populated before any rendering-only early return.
        vertices, faces = self._extract_vertices_faces(mesh)
        self._current_vertices = vertices.astype(np.float32, copy=True)
        self._current_faces = faces.astype(np.int32, copy=True)
        self._edge_index_to_vertices = self._build_unique_edges(self._current_faces)
        self._edge_key_to_index = {
            tuple(sorted((int(edge[0]), int(edge[1])))): int(i)
            for i, edge in enumerate(self._edge_index_to_vertices)
        }
        self._edge_to_faces = self._build_edge_adjacency(self._current_faces)
        self._vertex_adjacency = self._build_vertex_adjacency(self._current_faces)
        self._open_edges = self._find_open_edges(self._current_faces)

        if not self._gfx_ready or self._scene is None or self._gfx is None:
            self._refresh_host_state()
            self.selection_changed.emit(self.get_selection_state())
            self._request_draw()
            return

        geometry, material = self._build_mesh_geometry_and_material(mesh, vertices, faces)
        self._apply_clip_plane_to_material(material)

        node = self._gfx.Mesh(geometry, material)
        try:
            node.add_event_handler(self._on_mesh_pointer_down, "pointer_down")
            node.add_event_handler(self._on_mesh_pointer_move, "pointer_move")
            node.add_event_handler(self._on_mesh_pointer_up, "pointer_up")
        except Exception:
            pass

        self._remove_node("_mesh_node")
        self._scene.add(node)
        self._mesh_node = node

        self._rebuild_wire_node()
        self._rebuild_boundary_node()
        self._rebuild_selection_overlay()
        self._rebuild_point_selection_overlay()
        self._rebuild_edge_selection_overlay()
        self._rebuild_hover_overlay()

        if self._original_mesh_object is None:
            self._original_mesh_object = mesh

        if self._original_mesh_object is mesh or self._original_mesh_node is None:
            self._remove_node("_original_mesh_node")
            ghost_material = self._gfx.MeshPhongMaterial(
                color=self.config.compare_color,
                opacity=0.22,
                pick_write=False,
            )
            self._apply_clip_plane_to_material(ghost_material)
            ghost = self._gfx.Mesh(geometry, ghost_material)
            try:
                ghost.render_order = 1
            except Exception:
                pass
            ghost.visible = False
            self._scene.add(ghost)
            self._original_mesh_node = ghost

        self._update_floor_grid(vertices)
        self._fit_camera_to_bounds(vertices)
        self._update_display_visibility()
        self._refresh_host_state()
        self.selection_changed.emit(self.get_selection_state())
        self._request_draw()


    # ------------------------------------------------------------------
    # Public state access
    # ------------------------------------------------------------------
    @property
    def current_path(self) -> str | None:
        return self._current_path

    def has_mesh(self) -> bool:
        return self._current_mesh_object is not None

    def get_current_mesh_data(self) -> Any | None:
        return self._current_mesh_object

    def get_original_mesh_data(self) -> Any | None:
        return self._original_mesh_object

    def get_selection_mode(self) -> str:
        return self._selection_mode

    def get_display_preset(self) -> str:
        return self._display_preset

    def get_compare_mode(self) -> str:
        return self._compare_mode

    def get_selected_cell_ids(self) -> list[int]:
        return list(self._selected_cell_ids)

    def get_selected_point_ids(self) -> list[int]:
        return list(self._selected_point_ids)

    def get_selected_edge_ids(self) -> list[int]:
        return sorted(int(v) for v in self._selected_edge_ids)

    def get_last_picked_world_pos(self) -> tuple[float, float, float] | None:
        return self._last_picked_world_pos

    def get_selection_state(self) -> dict[str, Any]:
        return {
            "mode": self._selection_mode,
            "selected_cell_ids": sorted(int(v) for v in self._selected_cell_ids),
            "selected_point_ids": sorted(int(v) for v in self._selected_point_ids),
            "selected_edge_ids": sorted(int(v) for v in self._selected_edge_ids),
            "edge_region_strategy": self._edge_region_strategy,
            "last_picked_world_pos": self._last_picked_world_pos,
            "edge_pick_seed": dict(self._last_edge_pick_seed or {}),
            "brush_select_enabled": self._brush_select_enabled,
        }

    def get_capabilities(self) -> dict[str, bool]:
        return {
            "embedded": True,
            "compare_mode": True,
            "clip_plane": True,
            "point_picking": True,
            "face_picking": True,
            "edge_picking": True,
            "mesh_picking": False,
            "boundary_edges": True,
            "boundary_loop_face_select": True,
            "connected_point_region_select": True,
            "edge_selection": True,
            "feature_edge_region_select": True,
            "preview_mesh": True,
            "overlays": True,
            "screenshots": True,
            "face_rgba": True,
            "texture_materials": True,
            "brush_face_selection": True,
            "brush_point_selection": True,
            "brush_edge_selection": False,
            "diagnostic_modes": True,
            "real_canvas_host": bool(self._render_host and self._render_host.is_real_canvas),
            "gpu_scene_active": bool(self._gfx_ready),
        }

    # ------------------------------------------------------------------
    # Scene / loading
    # ------------------------------------------------------------------
    def clear_scene(self) -> None:
        self._current_path = None
        self._current_mesh_source_name = None
        self._current_mesh_object = None
        self._selected_cell_ids.clear()
        self._selected_point_ids.clear()
        self._selected_edge_ids.clear()
        self._hover_face = None
        self._last_picked_world_pos = None
        self._reset_brush_drag()

        self._remove_node("_mesh_node")
        self._remove_node("_wire_node")
        self._remove_node("_selection_node")
        self._remove_node("_selection_wire_node")
        self._remove_node("_point_selection_node")
        self._remove_node("_edge_selection_node")
        self._remove_node("_boundary_node")
        self._remove_node("_hover_node")
        self._remove_node("_original_mesh_node")

        self._current_vertices = None
        self._current_faces = None
        self._open_edges = None
        self._edge_index_to_vertices = None
        self._edge_key_to_index.clear()
        self._edge_to_faces.clear()
        self._vertex_adjacency.clear()

        self.clear_overlays()
        self._refresh_host_state()
        self._request_draw()
        self.selection_changed.emit(self.get_selection_state())
        self.status_changed.emit("WGPU viewport scene cleared.")

    def load_file(self, path: str | Path) -> None:
        mesh_path = Path(path).expanduser().resolve()
        if not mesh_path.is_file():
            message = f"WGPU viewport input file does not exist: {mesh_path}"
            self.mesh_failed.emit(message)
            raise FileNotFoundError(message)

        self._current_path = str(mesh_path)
        try:
            import trimesh

            loaded = trimesh.load(mesh_path, force="mesh", process=False)
            if isinstance(loaded, trimesh.Scene):
                geometries = [g for g in loaded.geometry.values() if hasattr(g, "faces") and hasattr(g, "vertices")]
                if not geometries:
                    raise WgpuViewportError(f"No mesh geometry found in scene: {mesh_path}")
                loaded = trimesh.util.concatenate(geometries)

            self._upload_mesh(loaded, source_name=str(mesh_path))
            if self._original_mesh_object is None:
                self._original_mesh_object = loaded
        except Exception as exc:
            message = f"WGPU viewport could not load mesh file {mesh_path}: {exc!r}"
            self.mesh_failed.emit(message)
            raise WgpuViewportError(message) from exc

        self.mesh_loaded.emit(str(mesh_path))
        self.status_changed.emit(f"Viewport loaded mesh: {mesh_path}")

    def load_dataset(self, dataset: Any, *, source_name: str = "dataset") -> None:
        self._current_path = None
        self._upload_mesh(dataset, source_name=source_name)
        if self._original_mesh_object is None:
            self._original_mesh_object = dataset
        self.status_changed.emit(f"WGPU viewport accepted dataset: {source_name}")

    def load_trimesh(self, mesh: Any, *, source_name: str = "trimesh") -> None:
        self._current_path = None
        self._upload_mesh(mesh, source_name=source_name)
        if self._original_mesh_object is None:
            self._original_mesh_object = mesh
        self.status_changed.emit(f"WGPU viewport accepted trimesh source: {source_name}")

    def set_mesh_data(
        self,
        dataset: Any,
        *,
        source_name: str = "dataset",
        keep_camera: bool = True,
        set_as_original: bool = False,
    ) -> None:
        del keep_camera
        self._upload_mesh(dataset, source_name=source_name)
        if set_as_original or self._original_mesh_object is None:
            self._original_mesh_object = dataset
        self.status_changed.emit(f"WGPU viewport mesh data set from {source_name}")

    def replace_mesh(self, dataset: Any, *, source_name: str = "dataset", keep_camera: bool = True) -> None:
        self.set_mesh_data(dataset, source_name=source_name, keep_camera=keep_camera)

    def update_mesh_geometry(self, dataset: Any, *, keep_camera: bool = True) -> None:
        self.set_mesh_data(dataset, source_name="updated geometry", keep_camera=keep_camera)

    def update_display_only(self) -> None:
        self._refresh_host_state()
        self._request_draw()

    def reload_current_file(self) -> None:
        if self._current_path is None:
            self.status_changed.emit("No mesh file to reload.")
            return
        self.load_file(self._current_path)

    def set_original_mesh_data(self, dataset: Any | None) -> None:
        self._original_mesh_object = dataset
        self._refresh_host_state()

    def capture_image(self, output_path: str | Path) -> str:
        if self._render_host is None:
            raise WgpuViewportError("No WGPU render host is available for capture.")
        out = self._render_host.capture_image(output_path)
        self.status_changed.emit(f"WGPU viewport capture saved: {Path(out).name}")
        return out

    # ------------------------------------------------------------------
    # Camera / display
    # ------------------------------------------------------------------
    def reset_camera(self) -> None:
        if self._current_mesh_object is not None:
            try:
                vertices, _faces = self._extract_vertices_faces(self._current_mesh_object)
                self._update_floor_grid(vertices)
                self._fit_camera_to_bounds(vertices)
                self._request_draw()
            except Exception:
                pass
        self.status_changed.emit("WGPU viewport camera reset.")

    def view_isometric(self) -> None:
        self.apply_camera_preset("isometric")

    def apply_camera_preset(self, preset: str) -> None:
        preset = preset.lower().strip()
        if preset not in self.CAMERA_PRESETS:
            raise ValueError(f"Unknown camera preset: {preset}")

        self._camera_preset = preset
        if self._current_mesh_object is not None:
            try:
                vertices, _faces = self._extract_vertices_faces(self._current_mesh_object)
                self._update_floor_grid(vertices)
                self._fit_camera_to_bounds(vertices)
            except Exception:
                pass

        self._refresh_host_state()
        self._request_draw()
        self.status_changed.emit(f"Camera preset: {preset}")

    def focus_on_bounds(self, bounds: tuple[float, float, float, float, float, float]) -> None:
        if self._camera is None:
            self.status_changed.emit("WGPU viewport focus_on_bounds requested.")
            return

        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        vertices = np.asarray([[xmin, ymin, zmin], [xmax, ymax, zmax]], dtype=np.float32)
        self._update_floor_grid(vertices)
        self._fit_camera_to_bounds(vertices)
        self._request_draw()
        self.status_changed.emit("WGPU viewport focused bounds.")

    def focus_on_selection(self) -> None:
        if self._current_vertices is None or self._current_faces is None:
            self.status_changed.emit("WGPU viewport focus_on_selection skipped: no mesh.")
            return

        if self._selected_cell_ids:
            valid = [i for i in self._selected_cell_ids if 0 <= i < len(self._current_faces)]
            if not valid:
                self.status_changed.emit("WGPU viewport focus_on_selection skipped: invalid face selection.")
                return
            face_ids = np.asarray(valid, dtype=np.int32)
            vertex_ids = np.unique(self._current_faces[face_ids].reshape(-1).astype(np.int32, copy=False))
            if vertex_ids.size == 0:
                self.status_changed.emit("WGPU viewport focus_on_selection skipped: empty face selection.")
                return
            vertices = self._current_vertices[vertex_ids]
            self._fit_camera_to_bounds(vertices)
            self._request_draw()
            self.status_changed.emit("WGPU viewport focused selection.")
            return

        if self._selected_point_ids:
            valid_points = [i for i in self._selected_point_ids if 0 <= i < len(self._current_vertices)]
            if not valid_points:
                self.status_changed.emit("WGPU viewport focus_on_selection skipped: invalid point selection.")
                return
            vertices = self._current_vertices[np.asarray(valid_points, dtype=np.int32)]
            self._fit_camera_to_bounds(vertices)
            self._request_draw()
            self.status_changed.emit("WGPU viewport focused selection.")
            return

        self.status_changed.emit("WGPU viewport focus_on_selection skipped: no selection.")

    def apply_display_preset(self, preset: str) -> None:
        preset = preset.strip()
        if preset not in self.DISPLAY_PRESETS:
            raise ValueError(f"Unknown display preset: {preset}")

        self._display_preset = preset
        opts = self.DISPLAY_PRESETS[preset]
        self._solid_visible = bool(opts.get("solid", True))
        self._wire_visible = bool(opts.get("wire", False))
        self._show_edges = self._wire_visible

        self._update_display_visibility()
        self._refresh_host_state()
        self._request_draw()
        self.status_changed.emit(f"Display preset: {preset}")

    def set_compare_mode(self, mode: str) -> None:
        mode = mode.strip()
        if mode not in self.COMPARE_MODES:
            raise ValueError(f"Unknown compare mode: {mode}")

        self._compare_mode = mode
        self._update_display_visibility()
        self._refresh_host_state()
        self.compare_mode_changed.emit(mode)
        self._request_draw()
        self.status_changed.emit(f"Compare mode: {mode}")

    def set_edges_visible(self, enabled: bool) -> None:
        self._show_edges = bool(enabled)
        self._wire_visible = bool(enabled)
        if not enabled and self._display_preset == "wireframe":
            self._solid_visible = True

        self._update_display_visibility()

        if self._preview_mesh_wire_node is not None:
            try:
                self._preview_mesh_wire_node.visible = bool(enabled)
            except Exception:
                pass

        self._refresh_host_state()
        self._request_draw()

    def set_edge_width(self, width: float) -> None:
        width = float(width)
        if width <= 0:
            raise ValueError("edge width must be > 0")

        self._edge_width = width
        self._rebuild_wire_node()
        self._rebuild_boundary_node()
        self._rebuild_selection_overlay()
        self._rebuild_edge_selection_overlay()
        self._rebuild_hover_overlay()
        self._update_display_visibility()
        self._refresh_host_state()
        self._request_draw()

    def set_grid_visible(self, enabled: bool) -> None:
        self._show_grid = bool(enabled)
        if self._grid_helper is not None:
            try:
                self._grid_helper.visible = self._show_grid
            except Exception:
                pass
        self._refresh_host_state()
        self._request_draw()

    def set_axes_visible(self, enabled: bool) -> None:
        self._show_axes = bool(enabled)
        if self._orientation_overlay is not None:
            self._orientation_overlay.setVisible(self._show_axes)
            self._orientation_overlay.reposition(self._render_host.rect() if self._render_host is not None else self.rect())
            self._orientation_overlay.raise_()
            self._orientation_overlay.update()
        self._refresh_host_state()
        self._request_draw()

    def set_boundary_highlight_visible(self, enabled: bool) -> None:
        self._show_boundary_edges = bool(enabled)
        self._rebuild_boundary_node()
        self._update_display_visibility()
        self._refresh_host_state()
        self._request_draw()

    def set_clip_plane(self, axis: str, fraction: float = 0.5, *, invert: bool = False) -> None:
        if self._current_vertices is None:
            return

        axis = axis.lower().strip()
        if axis not in {"x", "y", "z"}:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

        bounds_min = self._current_vertices.min(axis=0)
        bounds_max = self._current_vertices.max(axis=0)
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        value = float(bounds_min[idx] + (bounds_max[idx] - bounds_min[idx]) * float(fraction))

        self._clip_axis = axis
        self._clip_value = value
        self._clip_invert = bool(invert)

        for node in (self._mesh_node, self._original_mesh_node, self._preview_mesh_node):
            if node is not None and hasattr(node, "material"):
                self._apply_clip_plane_to_material(node.material)

        self._request_draw()
        self.status_changed.emit(f"Clip plane set: {axis}={value:.3f}")

    def clear_clip(self) -> None:
        self._clip_axis = None
        self._clip_value = None
        self._clip_invert = False

        for node in (self._mesh_node, self._original_mesh_node, self._preview_mesh_node):
            if node is not None and hasattr(node, "material"):
                try:
                    node.material.clip_planes = []
                except Exception:
                    pass

        self._request_draw()
        self.status_changed.emit("WGPU viewport clip cleared.")

    def set_diagnostic_mode(self, mode: str) -> None:
        mode = mode.strip()
        if mode not in self.DIAGNOSTIC_MODES:
            raise ValueError(f"Unknown diagnostic mode: {mode}")
        self._diagnostic_mode = mode
        self._refresh_host_state()
        self._request_draw()
        self.status_changed.emit(f"Diagnostic mode: {mode}")

    # ------------------------------------------------------------------
    # Selection API
    # ------------------------------------------------------------------
    def set_selection_mode(self, mode: str) -> None:
        mode = mode.strip().lower()
        if mode not in self.SELECTION_MODES:
            raise ValueError(f"Unknown selection mode: {mode}")

        self._selection_mode = mode
        if mode not in {"face", "point", "edge"}:
            self._reset_brush_drag()

        self._refresh_host_state()
        self.selection_changed.emit(self.get_selection_state())
        self.status_changed.emit(f"WGPU viewport selection mode: {mode}")

    def set_edge_region_strategy(self, strategy: str | None) -> None:
        """Set the generic core edge-region strategy used for Ctrl/Cmd edge expansion.

        The viewport does not implement Recognition, ownership, or rebuild
        logic. It stores this string and passes it through to
        ``core.selection_edges.select_edge_region(...)`` when Ctrl/Cmd-clicking
        an edge. Normal tools use ``safe``; Bore tools may request ``bore_rim``
        for local rim evidence.
        """

        value = str(strategy or "safe").strip().lower()
        if value not in {"safe", "open_component", "feature", "ring", "aggressive", "single", "bore_rim"}:
            value = "safe"
        self._edge_region_strategy = value
        self._refresh_host_state()

    def get_edge_region_strategy(self) -> str:
        """Return the current generic core edge-region strategy."""

        return str(getattr(self, "_edge_region_strategy", "safe") or "safe")

    def clear_selection(self) -> None:
        self._selected_cell_ids.clear()
        self._selected_point_ids.clear()
        self._selected_edge_ids.clear()
        self._last_picked_world_pos = None
        self._last_hover_pick_info = None
        self._last_edge_pick_seed = {}
        self._reset_brush_drag()
        self._rebuild_selection_overlay()
        self._rebuild_point_selection_overlay()
        self._rebuild_edge_selection_overlay()
        self._update_display_visibility()
        self._refresh_host_state()
        self.selection_changed.emit(self.get_selection_state())
        self._request_draw()

    def highlight_cells(self, cell_ids: list[int]) -> None:
        face_count = len(self._current_faces) if self._current_faces is not None else None
        selected: set[int] = set()
        for raw in cell_ids:
            value = int(raw)
            if value < 0:
                continue
            if face_count is not None and value >= face_count:
                continue
            selected.add(value)

        self._selected_cell_ids = selected
        self._selected_point_ids.clear()
        self._selected_edge_ids.clear()
        if selected:
            self._selection_mode = "face"
        self._rebuild_selection_overlay()
        self._rebuild_point_selection_overlay()
        self._rebuild_edge_selection_overlay()
        self._update_display_visibility()
        self._refresh_host_state()
        self.selection_changed.emit(self.get_selection_state())
        self._request_draw()

    def highlight_points(self, point_ids: list[int]) -> None:
        vertex_count = len(self._current_vertices) if self._current_vertices is not None else None
        selected: set[int] = set()
        for raw in point_ids:
            value = int(raw)
            if value < 0:
                continue
            if vertex_count is not None and value >= vertex_count:
                continue
            selected.add(value)

        self._selected_point_ids = selected
        self._selected_cell_ids.clear()
        self._selected_edge_ids.clear()
        if selected:
            self._selection_mode = "point"
        self._rebuild_selection_overlay()
        self._rebuild_point_selection_overlay()
        self._rebuild_edge_selection_overlay()
        self._update_display_visibility()
        self._refresh_host_state()
        self.selection_changed.emit(self.get_selection_state())
        self._request_draw()

    def highlight_edges(self, edge_ids: list[int]) -> None:
        edge_count = len(self._edge_index_to_vertices) if self._edge_index_to_vertices is not None else None
        selected: set[int] = set()
        for raw in edge_ids:
            value = int(raw)
            if value < 0:
                continue
            if edge_count is not None and value >= edge_count:
                continue
            selected.add(value)

        self._selected_edge_ids = selected
        self._selected_cell_ids.clear()
        self._selected_point_ids.clear()
        if selected:
            self._selection_mode = "edge"
        self._rebuild_selection_overlay()
        self._rebuild_point_selection_overlay()
        self._rebuild_edge_selection_overlay()
        self._update_display_visibility()
        self._refresh_host_state()
        self.selection_changed.emit(self.get_selection_state())
        self._request_draw()

    def grow_selection(self) -> None:
        if self._selection_mode == "edge" or self._selected_edge_ids:
            self._grow_edge_selection()
            return
        if self._selection_mode == "point" or self._selected_point_ids:
            self._grow_point_selection()
            return
        self._grow_face_selection()

    def shrink_selection(self) -> None:
        if self._selection_mode == "edge" or self._selected_edge_ids:
            self._shrink_edge_selection()
            return
        if self._selection_mode == "point" or self._selected_point_ids:
            self._shrink_point_selection()
            return
        self._shrink_face_selection()

    def _grow_edge_selection(self) -> None:
        if not self._selected_edge_ids or self._edge_index_to_vertices is None:
            self.status_changed.emit("No edge selection to grow.")
            return

        edge_count = len(self._edge_index_to_vertices)
        selected = {int(v) for v in self._selected_edge_ids if 0 <= int(v) < edge_count}
        if not selected:
            self.highlight_edges([])
            self.status_changed.emit("No valid edge selection to grow.")
            return

        grown = set(selected)
        for edge_index in sorted(selected):
            try:
                region = self._get_connected_edge_region(int(edge_index))
                if region:
                    grown.update(region)
                    continue
            except Exception:
                pass
            edge = self._edge_index_to_vertices[int(edge_index)]
            a, b = int(edge[0]), int(edge[1])
            for i, candidate in enumerate(self._edge_index_to_vertices):
                ca, cb = int(candidate[0]), int(candidate[1])
                if ca in {a, b} or cb in {a, b}:
                    grown.add(int(i))

        self.highlight_edges(sorted(grown))
        self.status_changed.emit(
            f"Edge selection grown: {len(selected)} -> {len(grown)} edge(s)."
        )

    def _shrink_edge_selection(self) -> None:
        if not self._selected_edge_ids or self._edge_index_to_vertices is None:
            self.status_changed.emit("No edge selection to shrink.")
            return

        edge_count = len(self._edge_index_to_vertices)
        selected = {int(v) for v in self._selected_edge_ids if 0 <= int(v) < edge_count}
        if not selected:
            self.highlight_edges([])
            self.status_changed.emit("No valid edge selection to shrink.")
            return

        vertex_to_edges: dict[int, set[int]] = {}
        for i, edge in enumerate(self._edge_index_to_vertices):
            a, b = int(edge[0]), int(edge[1])
            vertex_to_edges.setdefault(a, set()).add(int(i))
            vertex_to_edges.setdefault(b, set()).add(int(i))

        boundary: set[int] = set()
        for edge_index in sorted(selected):
            a, b = (int(v) for v in self._edge_index_to_vertices[int(edge_index)])
            neighbors = (vertex_to_edges.get(a, set()) | vertex_to_edges.get(b, set())) - {int(edge_index)}
            if any(neighbor not in selected for neighbor in neighbors):
                boundary.add(int(edge_index))

        shrunk = selected - boundary
        self.highlight_edges(sorted(shrunk))
        self.status_changed.emit(
            f"Edge selection shrunk: {len(selected)} -> {len(shrunk)} edge(s)."
        )

    def _grow_face_selection(self) -> None:
        if not self._selected_cell_ids or self._current_faces is None:
            self.status_changed.emit("No face selection to grow.")
            return

        face_count = len(self._current_faces)
        selected = {int(v) for v in self._selected_cell_ids if 0 <= int(v) < face_count}
        if not selected:
            self.highlight_cells([])
            self.status_changed.emit("No valid face selection to grow.")
            return

        new_selected = set(selected)
        for f in selected:
            tri = self._current_faces[f]
            for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = tuple(sorted((int(e[0]), int(e[1]))))
                for nf in self._edge_to_faces.get(key, []):
                    nf_int = int(nf)
                    if 0 <= nf_int < face_count:
                        new_selected.add(nf_int)

        self.highlight_cells(sorted(new_selected))
        self.status_changed.emit(
            f"Face selection grown: {len(selected)} -> {len(new_selected)} face(s)."
        )

    def _shrink_face_selection(self) -> None:
        if not self._selected_cell_ids or self._current_faces is None:
            self.status_changed.emit("No face selection to shrink.")
            return

        face_count = len(self._current_faces)
        selected = {int(v) for v in self._selected_cell_ids if 0 <= int(v) < face_count}
        if not selected:
            self.highlight_cells([])
            self.status_changed.emit("No valid face selection to shrink.")
            return

        boundary = set()
        for f in selected:
            tri = self._current_faces[f]
            for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = tuple(sorted((int(e[0]), int(e[1]))))
                faces = self._edge_to_faces.get(key, [])
                if len(faces) == 1 or any(nf not in selected for nf in faces):
                    boundary.add(int(f))
                    break

        new_selected = selected - boundary
        self.highlight_cells(sorted(new_selected))
        self.status_changed.emit(
            f"Face selection shrunk: {len(selected)} -> {len(new_selected)} face(s)."
        )

    def _grow_point_selection(self) -> None:
        if not self._selected_point_ids or self._current_vertices is None:
            self.status_changed.emit("No point selection to grow.")
            return

        vertex_count = len(self._current_vertices)
        selected = {int(v) for v in self._selected_point_ids if 0 <= int(v) < vertex_count}
        if not selected:
            self.highlight_points([])
            self.status_changed.emit("No valid point selection to grow.")
            return

        new_selected = set(selected)
        for vertex_id in selected:
            for neighbor in self._vertex_adjacency.get(int(vertex_id), set()):
                neighbor_int = int(neighbor)
                if 0 <= neighbor_int < vertex_count:
                    new_selected.add(neighbor_int)

        self.highlight_points(sorted(new_selected))
        self.status_changed.emit(
            f"Point selection grown: {len(selected)} -> {len(new_selected)} point(s)."
        )

    def _shrink_point_selection(self) -> None:
        if not self._selected_point_ids or self._current_vertices is None:
            self.status_changed.emit("No point selection to shrink.")
            return

        vertex_count = len(self._current_vertices)
        selected = {int(v) for v in self._selected_point_ids if 0 <= int(v) < vertex_count}
        if not selected:
            self.highlight_points([])
            self.status_changed.emit("No valid point selection to shrink.")
            return

        boundary: set[int] = set()
        for vertex_id in selected:
            neighbors = {
                int(neighbor)
                for neighbor in self._vertex_adjacency.get(int(vertex_id), set())
                if 0 <= int(neighbor) < vertex_count
            }
            if not neighbors or any(neighbor not in selected for neighbor in neighbors):
                boundary.add(int(vertex_id))

        new_selected = selected - boundary
        self.highlight_points(sorted(new_selected))
        self.status_changed.emit(
            f"Point selection shrunk: {len(selected)} -> {len(new_selected)} point(s)."
        )


    def set_brush_selection_enabled(self, enabled: bool) -> None:
        self._brush_select_enabled = bool(enabled)
        if not self._brush_select_enabled:
            self._reset_brush_drag()
        self._refresh_host_state()
        self.selection_changed.emit(self.get_selection_state())
        self.status_changed.emit(f"WGPU brush selection {'enabled' if self._brush_select_enabled else 'disabled'}.")

    def is_brush_selection_enabled(self) -> bool:
        return bool(self._brush_select_enabled)

    def enable_surface_picking(self, callback: Callable[[tuple[float, float, float]], None] | None = None) -> None:
        self._surface_pick_callback = callback
        self.set_selection_mode("face")
        self.status_changed.emit("WGPU surface picking enabled.")

    def enable_point_picking(self, callback: Callable[[tuple[float, float, float]], None] | None = None) -> None:
        self._surface_pick_callback = callback
        self.set_selection_mode("point")
        self.status_changed.emit("WGPU point picking enabled.")

    def enable_mesh_picking(self, callback: Callable[[Any], None] | None = None) -> None:
        del callback
        self.status_changed.emit("WGPU mesh picking is not implemented yet.")

    def disable_picking(self) -> None:
        self._surface_pick_callback = None
        self._selection_mode = "none"
        self._reset_brush_drag()
        self._refresh_host_state()
        self.selection_changed.emit(self.get_selection_state())
        self.status_changed.emit("WGPU picking disabled.")

    # ------------------------------------------------------------------
    # Generic overlays / previews
    # ------------------------------------------------------------------
    def clear_overlays(self) -> None:
        if self._scene is not None:
            for actor in self._overlay_objects.values():
                try:
                    self._scene.remove(actor)
                except Exception:
                    pass
        self._overlay_objects.clear()

        for node_attr in ("_preview_mesh_node", "_preview_mesh_wire_node"):
            node = getattr(self, node_attr, None)
            if node is not None and self._scene is not None:
                try:
                    self._scene.remove(node)
                except Exception:
                    pass
                setattr(self, node_attr, None)

        self._request_draw()
        self.status_changed.emit("WGPU overlays cleared.")

    def show_marker(
        self,
        position: tuple[float, float, float],
        *,
        name: str = "marker",
        radius: float | None = None,
        color: str | None = None,
    ) -> None:
        if not self._gfx_ready or self._scene is None or self._gfx is None:
            return

        r = float(radius or 0.03)
        sphere = self._gfx.Mesh(
            self._gfx.sphere_geometry(radius=r),
            self._gfx.MeshPhongMaterial(color=color or self.config.overlay_color),
        )
        sphere.local.position = tuple(float(v) for v in position)

        old = self._overlay_objects.get(name)
        if old is not None:
            try:
                self._scene.remove(old)
            except Exception:
                pass

        self._scene.add(sphere)
        self._overlay_objects[name] = sphere
        self._request_draw()

    def show_polyline(
        self,
        points: list[tuple[float, float, float]],
        *,
        name: str = "polyline",
        color: str | None = None,
        width: float = 3.0,
        closed: bool = False,
    ) -> None:
        if not self._gfx_ready or self._scene is None or self._gfx is None or len(points) < 2:
            return

        pts = np.asarray(points, dtype=np.float32)
        if closed:
            pts = np.vstack([pts, pts[0]])

        line = self._gfx.Line(
            self._gfx.Geometry(positions=pts),
            self._gfx.LineMaterial(color=color or self.config.overlay_color, thickness=float(width)),
        )

        old = self._overlay_objects.get(name)
        if old is not None:
            try:
                self._scene.remove(old)
            except Exception:
                pass

        self._scene.add(line)
        self._overlay_objects[name] = line
        self._request_draw()

    def show_preview_mesh(
        self,
        dataset: Any,
        *,
        color: str = "#7ee787",
        opacity: float = 0.35,
        show_edges: bool = True,
        line_width: float = 1.5,
    ) -> None:
        """
        Show a non-pickable preview mesh as filled translucent surface geometry.

        Phase 2E preview behavior:
        - the preview surface is always rendered as filled triangles
        - the optional edge outline is drawn as a separate line overlay
        - the surface is double-sided when the material supports it

        The old implementation used material.wireframe=True when show_edges=True.
        In pygfx that turns the mesh itself into a wireframe, so small hole-fill
        patches appeared as an outline only. The preview now follows the same
        visual model as the selection overlay: filled mesh plus optional outline.
        """
        if not self._gfx_ready or self._scene is None or self._gfx is None:
            return

        vertices, faces = self._extract_vertices_faces(dataset)
        if vertices.size == 0 or faces.size == 0:
            return

        self.clear_preview_mesh()

        geometry = self._gfx.Geometry(
            positions=vertices.astype(np.float32, copy=False),
            indices=faces.astype(np.int32, copy=False),
        )

        try:
            material = self._gfx.MeshBasicMaterial(
                color=color,
                opacity=float(opacity),
                pick_write=False,
            )
        except TypeError:
            material = self._gfx.MeshBasicMaterial(
                color=color,
                opacity=float(opacity),
            )

        # Make small one-sided preview patches visible from both sides when the
        # installed pygfx material exposes a side/culling option.
        for attr_name, value in (
            ("side", "both"),
            ("cull_mode", "none"),
        ):
            try:
                setattr(material, attr_name, value)
            except Exception:
                pass

        # Keep the preview as a filled surface. Edges are handled by a separate
        # line node below.
        try:
            material.wireframe = False
        except Exception:
            pass

        self._apply_clip_plane_to_material(material)

        surface_node = self._gfx.Mesh(geometry, material)
        try:
            surface_node.render_order = 30
        except Exception:
            pass

        self._scene.add(surface_node)
        self._preview_mesh_node = surface_node

        if show_edges:
            try:
                edges = self._build_unique_edges(faces.astype(np.int32, copy=False))
                if len(edges) > 0:
                    edge_positions = vertices[edges.reshape(-1)].astype(np.float32, copy=False)
                    edge_geometry = self._gfx.Geometry(positions=edge_positions)
                    try:
                        edge_material = self._gfx.LineSegmentMaterial(
                            color=color,
                            thickness=max(1.0, float(line_width)),
                        )
                    except Exception:
                        edge_material = self._gfx.LineMaterial(
                            color=color,
                            thickness=max(1.0, float(line_width)),
                        )
                    edge_node = self._gfx.Line(edge_geometry, edge_material)
                    try:
                        edge_node.render_order = 31
                    except Exception:
                        pass
                    edge_node.visible = bool(show_edges)
                    self._scene.add(edge_node)
                    self._preview_mesh_wire_node = edge_node
            except Exception as exc:
                self._append_host_note(f"preview edge overlay failed: {exc!r}")

        self._request_draw()
        self.status_changed.emit("WGPU preview mesh updated.")

    def clear_preview_mesh(self) -> None:
        for node_attr in ("_preview_mesh_node", "_preview_mesh_wire_node"):
            node = getattr(self, node_attr, None)
            if node is not None and self._scene is not None:
                try:
                    self._scene.remove(node)
                except Exception:
                    pass
                setattr(self, node_attr, None)
        self._request_draw()
        self.status_changed.emit("WGPU preview mesh cleared.")

    def shutdown(self) -> None:
        self._reset_brush_drag()
        self.status_changed.emit("WGPU viewport shutdown.")
