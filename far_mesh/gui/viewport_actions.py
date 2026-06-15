from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtWidgets import QFileDialog, QMessageBox


class ViewportActionsMixin:
    """MainWindow mixin for viewport signal wiring, state sync, and viewer actions."""

    def _connect_viewport_signals(self) -> None:
            self.viewport.status_changed.connect(self._on_viewport_status)
            self.viewport.mesh_loaded.connect(self._on_viewport_mesh_loaded)
            self.viewport.mesh_failed.connect(self._on_viewport_mesh_failed)
            self.viewport.point_picked.connect(self._on_viewport_point_picked)
            self.viewport.selection_changed.connect(self._on_viewport_selection_changed)

    def _safe_viewport_capabilities(self) -> dict[str, bool]:
            try:
                caps = self.viewport.get_capabilities()
                if isinstance(caps, dict):
                    return {str(k): bool(v) for k, v in caps.items()}
            except Exception:
                pass
            return {}

    def _connected_point_capability_enabled(self, caps: dict[str, bool]) -> bool:
            return bool(
                caps.get("connected_point_selection", False)
                or caps.get("connected_point_region_select", False)
            )

    def _viewport_mode_available(self, mode: str) -> bool:
            """Return whether the active viewport exposes a raw selection mode.

            Capability dictionaries are advisory and older app states/tests may
            miss newer edge keys even when the viewport actually supports edge
            selection.  The GUI should not gray out a mode that is present on
            the viewport protocol surface.
            """
            mode_key = str(mode or "").strip().lower()
            try:
                raw_modes = getattr(self.viewport, "SELECTION_MODES", ())
                if mode_key in {str(v).strip().lower() for v in raw_modes}:
                    return True
            except Exception:
                pass

            if mode_key == "point":
                return bool(
                    hasattr(self.viewport, "highlight_points")
                    or hasattr(self.viewport, "get_selected_point_ids")
                )
            if mode_key == "face":
                return bool(
                    hasattr(self.viewport, "highlight_cells")
                    or hasattr(self.viewport, "get_selected_cell_ids")
                )
            if mode_key == "edge":
                return bool(
                    hasattr(self.viewport, "highlight_edges")
                    or hasattr(self.viewport, "get_selected_edge_ids")
                    or hasattr(self.viewport, "_get_connected_edge_region")
                )
            if mode_key == "mesh":
                return bool(hasattr(self.viewport, "enable_mesh_picking"))
            return False

    def _selection_capability_enabled(
        self,
        caps: dict[str, bool],
        *capability_names: str,
        mode: str,
    ) -> bool:
            """Combine explicit backend caps with protocol-surface fallback.

            This keeps new modes selectable while capability propagation catches
            up across MainWindow/viewport reload paths.
            """
            for name in capability_names:
                if caps.get(name, False):
                    return True
            return self._viewport_mode_available(mode)

    def _sync_viewport_ui_from_backend(self) -> None:
            self.viewport_title_label.setText(f"Viewport ({getattr(self.viewport, 'BACKEND_NAME', 'unknown')})")
            caps = self._safe_viewport_capabilities()

            self.selection_controller.sync_from_viewport(reason="sync_viewport_ui_from_backend")

            self._suppress_viewer_sync = True
            try:
                self._sync_viewer_controls_from_viewport_state()
                self._sync_brush_controls_from_controller_state()
                self._sync_controls_from_capabilities(caps)
                self._apply_backend_specific_explanations(caps)
                self._update_brush_action_state(caps)
            finally:
                self._suppress_viewer_sync = False

    def _sync_viewer_controls_from_viewport_state(self) -> None:
            preset = self._safe_call(lambda: self.viewport.get_display_preset(), "inspection_edges")
            compare = self._safe_call(lambda: self.viewport.get_compare_mode(), "current_only")
            selection = self.selection_controller.viewport_mode_from_semantic(
                getattr(self.selection_controller.session, "mode", None)
            )
            backend_name = getattr(self.viewport, "BACKEND_NAME", "unknown")

            self._set_combo_current_data(self.viewer_preset_combo, preset)
            self._set_combo_current_data(self.viewport_quick_preset_combo, preset)
            self._set_combo_current_data(self.viewer_compare_combo, compare)
            self._set_combo_current_data(self.viewport_quick_compare_combo, compare)
            self._set_combo_current_data(self.viewer_selection_combo, selection)

            self.viewer_grid_check.setChecked(bool(self._safe_attr("_show_grid", True)))
            self.viewer_axes_check.setChecked(bool(self._safe_attr("_show_axes", True)))
            self.viewer_edges_check.setChecked(bool(self._safe_attr("_show_edges", True)))
            self.viewer_edge_width_spin.setValue(float(self._safe_attr("_edge_width", 1.5)))
            self.viewer_boundary_check.setChecked(bool(self._safe_attr("_show_boundary_edges", False)))

            self.viewport_quick_grid_check.setChecked(self.viewer_grid_check.isChecked())
            self.viewport_quick_axes_check.setChecked(self.viewer_axes_check.isChecked())

            diagnostics_supported = (
                hasattr(self.viewport, "set_host_info_visible")
                and hasattr(self.viewport, "is_host_info_visible")
            )
            diagnostics_visible = (
                bool(self._safe_call(lambda: self.viewport.is_host_info_visible(), False))
                if diagnostics_supported else False
            )

            self.viewer_diagnostics_check.setChecked(diagnostics_visible)
            self.viewport_toggle_info_btn.setChecked(diagnostics_visible)
            self.action_toggle_viewport_diagnostics.setChecked(diagnostics_visible)
            self.action_toggle_viewport_diagnostics.setEnabled(diagnostics_supported)
            self.viewport_toggle_info_btn.setEnabled(diagnostics_supported)
            self.viewport_toggle_info_btn.setVisible(diagnostics_supported)

            self.viewport_status_label.setText(f"{backend_name} viewport ready")

    def _sync_brush_controls_from_controller_state(self) -> None:
            if not hasattr(self, "brush_selection_mode_combo"):
                return

            mode_raw = self.selection_controller.viewport_mode_from_semantic(
                getattr(self.selection_controller.session, "mode", None)
            )
            brush_enabled = bool(getattr(self.selection_controller.session, "brush_enabled", False))
            boundary = bool(getattr(self.selection_controller.session, "boundary_highlight", False))

            brush_combo_mode = mode_raw if mode_raw in {"face", "point", "edge"} else "face"
            self._set_combo_current_data(self.brush_selection_mode_combo, brush_combo_mode)

            if hasattr(self, "brush_enable_check"):
                self.brush_enable_check.setChecked(brush_enabled)
            if hasattr(self, "brush_boundary_check"):
                self.brush_boundary_check.setChecked(boundary)

    def _sync_controls_from_capabilities(self, caps: dict[str, bool]) -> None:
            compare_enabled = caps.get("compare_mode", True)
            self.viewer_compare_combo.setEnabled(compare_enabled)
            self.viewport_quick_compare_combo.setEnabled(compare_enabled)

            clip_enabled = caps.get("clip_plane", False)
            self.viewer_clip_axis_combo.setEnabled(clip_enabled)
            self.viewer_clip_fraction_spin.setEnabled(clip_enabled)
            self.viewer_clip_invert_check.setEnabled(clip_enabled)
            self.viewer_apply_clip_btn.setEnabled(clip_enabled)
            self.viewer_clear_clip_btn.setEnabled(clip_enabled)

            boundary_enabled = caps.get("boundary_edges", False)
            self.viewer_boundary_check.setEnabled(boundary_enabled)
            if hasattr(self, "brush_boundary_check"):
                self.brush_boundary_check.setEnabled(boundary_enabled)

            overlays_enabled = caps.get("overlays", False)
            self.viewer_drop_marker_btn.setEnabled(overlays_enabled)

            screenshots_enabled = caps.get("screenshots", True)
            self.viewer_screenshot_btn.setEnabled(screenshots_enabled and self.viewport.has_mesh())

            point_enabled = self._selection_capability_enabled(caps, "point_picking", mode="point")
            face_enabled = self._selection_capability_enabled(caps, "face_picking", mode="face")
            edge_enabled = self._selection_capability_enabled(
                caps,
                "edge_picking",
                "edge_selection",
                mode="edge",
            )
            mesh_enabled = bool(caps.get("mesh_picking", False))
            connected_points_enabled = self._connected_point_capability_enabled(caps)

            self._set_combo_item_enabled(self.viewer_selection_combo, "point", point_enabled)
            self._set_combo_item_enabled(self.viewer_selection_combo, "face", face_enabled)
            self._set_combo_item_enabled(self.viewer_selection_combo, "edge", edge_enabled)
            self._set_combo_item_enabled(self.viewer_selection_combo, "mesh", mesh_enabled)

            session_raw = self.selection_controller.viewport_mode_from_semantic(
                getattr(self.selection_controller.session, "mode", None)
            )
            if session_raw == "point" and not point_enabled:
                self._set_combo_current_data(self.viewer_selection_combo, "none")
            if session_raw == "face" and not face_enabled:
                self._set_combo_current_data(self.viewer_selection_combo, "none")
            if session_raw == "edge" and not edge_enabled:
                self._set_combo_current_data(self.viewer_selection_combo, "none")
            if session_raw == "mesh" and not mesh_enabled:
                self._set_combo_current_data(self.viewer_selection_combo, "none")

            focus_selection_enabled = point_enabled or face_enabled or edge_enabled or mesh_enabled
            self.viewer_focus_selection_btn.setEnabled(focus_selection_enabled)

            if hasattr(self, "brush_selection_mode_combo"):
                self._set_combo_item_enabled(self.brush_selection_mode_combo, "face", face_enabled)
                self._set_combo_item_enabled(self.brush_selection_mode_combo, "point", point_enabled)
                self._set_combo_item_enabled(self.brush_selection_mode_combo, "edge", edge_enabled)

                brush_mode = self.selection_controller.viewport_mode_from_semantic(
                    getattr(self.selection_controller.session, "mode", None)
                )
                brush_capable = (
                    (brush_mode == "face" and caps.get("brush_face_selection", False))
                    or (brush_mode == "point" and caps.get("brush_point_selection", False))
                    or (brush_mode == "edge" and caps.get("brush_edge_selection", False))
                )

                self.brush_enable_check.setEnabled(brush_capable)
                self.brush_clear_btn.setEnabled(point_enabled or face_enabled or edge_enabled or mesh_enabled)
                self.brush_focus_btn.setEnabled(focus_selection_enabled)
                self.brush_connected_points_btn.setEnabled(
                    brush_mode == "point"
                    and point_enabled
                    and connected_points_enabled
                    and hasattr(self.viewport, "select_connected_points_from_vertex")
                )

    def _apply_backend_specific_explanations(self, caps: dict[str, bool]) -> None:
            backend_name = str(getattr(self.viewport, "BACKEND_NAME", "unknown"))
            is_wgpu = backend_name == "wgpu"
            point_enabled = self._selection_capability_enabled(caps, "point_picking", mode="point")
            face_enabled = self._selection_capability_enabled(caps, "face_picking", mode="face")
            edge_enabled = self._selection_capability_enabled(
                caps,
                "edge_picking",
                "edge_selection",
                mode="edge",
            )
            mesh_enabled = bool(caps.get("mesh_picking", False))
            brush_points = bool(caps.get("brush_point_selection", False))
            brush_faces = bool(caps.get("brush_face_selection", False))
            brush_edges = bool(caps.get("brush_edge_selection", False))
            connected_points = self._connected_point_capability_enabled(caps)

            self.viewer_boundary_check.setToolTip("Highlight open mesh boundaries in the current viewport.")
            self.viewer_focus_selection_btn.setToolTip("Focus the camera on the current face, point, or edge selection.")
            self.viewer_clear_selection_btn.setToolTip("Clear the current viewport selection.")
            self.viewport_open_viewer_btn.setToolTip("Open the full viewer control page.")
            self.viewport_open_brush_btn.setToolTip("Open the brush / tools page for face, point, and edge selection workflows.")

            selection_tip = [
                "Selection tools available in this backend:",
                f"• Point picking: {'yes' if point_enabled else 'no'}",
                f"• Face picking: {'yes' if face_enabled else 'no'}",
                f"• Edge picking: {'yes' if edge_enabled else 'no'}",
                f"• Mesh picking: {'yes' if mesh_enabled else 'no'}",
            ]
            if is_wgpu and not (point_enabled and face_enabled and edge_enabled and mesh_enabled):
                selection_tip.append("WGPU still exposes part of the selection API as backend-specific placeholders.")
            self.viewer_selection_combo.setToolTip("\n".join(selection_tip))

            point_tip = (
                "Point picking on mesh vertices is available."
                if point_enabled else
                "Point picking is unavailable in the active viewport backend."
            )
            face_tip = (
                "Face picking is available."
                if face_enabled else
                "Face picking is unavailable in the active viewport backend."
            )
            mesh_tip = (
                "Mesh picking is available."
                if mesh_enabled else
                "Mesh picking is unavailable in the active viewport backend."
            )
            edge_tip = (
                "Edge selection is available. Click selects one edge; Ctrl-click selects a connected edge chain."
                if edge_enabled else
                "Edge selection is unavailable in the active viewport backend."
            )

            self._set_combo_item_tooltip(self.viewer_selection_combo, "point", point_tip)
            self._set_combo_item_tooltip(self.viewer_selection_combo, "face", face_tip)
            self._set_combo_item_tooltip(self.viewer_selection_combo, "mesh", mesh_tip)
            self._set_combo_item_tooltip(self.viewer_selection_combo, "edge", edge_tip)

            self.viewer_preset_combo.setToolTip("Choose a display preset tuned for inspection, repair, or lightweight viewing.")
            self.viewport_quick_preset_combo.setToolTip(self.viewer_preset_combo.toolTip())
            self.viewer_edges_check.setToolTip("Show the wire overlay on top of shaded mesh rendering.")
            self.viewer_edge_width_spin.setToolTip("Adjust the displayed wire overlay thickness.")
            self.viewer_grid_check.setToolTip("Show the floor grid plane aligned with the active viewport scene.")
            self.viewer_axes_check.setToolTip("Show the viewport orientation marker / axes display.")
            self.viewport_quick_grid_check.setToolTip(self.viewer_grid_check.toolTip())
            self.viewport_quick_axes_check.setToolTip(self.viewer_axes_check.toolTip())
            self.viewer_camera_combo.setToolTip("Choose a standard camera orientation.")
            self.viewer_apply_camera_btn.setToolTip("Apply the selected camera preset.")
            self.viewer_reset_btn.setToolTip("Reset the camera to frame the current mesh.")
            self.viewport_reset_btn.setToolTip(self.viewer_reset_btn.toolTip())

            if hasattr(self, "brush_backend_note_label"):
                if is_wgpu:
                    self.brush_backend_note_label.setText(
                        "WGPU backend active. Face, point, and edge selection share this left-side tools page. "
                        "Enable brush only when you want drag-paint selection; leave it off for normal orbit + click picking. "
                        "Edge selection is click/Ctrl-click based."
                    )
                else:
                    self.brush_backend_note_label.setText(
                        "Fallback backend active. The brush page still uses the shared viewport protocol, "
                        "but the richest point/face/edge workflows are expected on WGPU first."
                    )

                brush_tip = (
                    "Drag brush selection is available for the current mode."
                    if (brush_points or brush_faces or brush_edges) else
                    "Drag brush selection is unavailable in the active viewport backend."
                )
                connected_tip = (
                    "Connected point-region selection is available."
                    if connected_points else
                    "Connected point-region selection is unavailable in the active viewport backend."
                )

                self.brush_selection_mode_combo.setToolTip(point_tip + "\n" + face_tip + "\n" + edge_tip)
                self.brush_enable_check.setToolTip(brush_tip)
                self.brush_connected_points_btn.setToolTip(connected_tip)
                self.brush_grow_btn.setToolTip("Grow the current face, point, or edge selection.")
                self.brush_shrink_btn.setToolTip("Shrink the current face, point, or edge selection.")
                self.brush_clear_btn.setToolTip("Clear the current viewport selection.")
                self.brush_focus_btn.setToolTip("Focus the camera on the current selection.")
                self.brush_boundary_check.setToolTip("Highlight open mesh boundaries in the current viewport.")

    def _on_viewport_status(self, message: str) -> None:
            self.viewport_status_label.setText(message)
            self.statusBar().showMessage(message, 4000)

    def _on_viewport_mesh_loaded(self, path: str) -> None:
            self.log(f"Viewport loaded mesh: {path}")
            self.selection_controller.reapply_after_refresh(reason="viewport_mesh_loaded")
            self._sync_viewport_ui_from_backend()

    def _on_viewport_mesh_failed(self, message: str) -> None:
            self.log(message)

    def _on_viewport_point_picked(self, point: tuple) -> None:
            self.statusBar().showMessage(f"Picked point: {point}", 3000)
            self._update_brush_action_state()

    def _on_viewport_selection_changed(self, state: object) -> None:
            payload = state if isinstance(state, dict) else None
            self.selection_controller.sync_from_viewport(payload, reason="main_window_selection_changed")

            self._suppress_viewer_sync = True
            try:
                self._sync_viewer_controls_from_viewport_state()
                self._sync_brush_controls_from_controller_state()
                self._update_brush_action_state()
            finally:
                self._suppress_viewer_sync = False

    def _on_display_preset_changed(self) -> None:
            if self._suppress_viewer_sync:
                return
            self._apply_display_preset_value(self.viewer_preset_combo.currentData())

    def _on_quick_display_preset_changed(self) -> None:
            if self._suppress_viewer_sync:
                return
            self._apply_display_preset_value(self.viewport_quick_preset_combo.currentData())

    def _apply_display_preset_value(self, preset: str | None) -> None:
            if not preset:
                return
            try:
                self.viewport.apply_display_preset(preset)
                self.log(f"Display preset: {preset}")
                self._sync_viewer_controls_from_viewport_state()
            except Exception as exc:
                self.log(f"Display preset failed: {exc}")

    def _on_compare_mode_changed(self) -> None:
            if self._suppress_viewer_sync:
                return
            self._apply_compare_mode_value(self.viewer_compare_combo.currentData())

    def _on_quick_compare_mode_changed(self) -> None:
            if self._suppress_viewer_sync:
                return
            self._apply_compare_mode_value(self.viewport_quick_compare_combo.currentData())

    def _apply_compare_mode_value(self, mode: str | None) -> None:
            if not mode:
                return
            try:
                self.viewport.set_compare_mode(mode)
                self.log(f"Compare mode: {mode}")
                self._sync_viewer_controls_from_viewport_state()
            except Exception as exc:
                self.log(f"Compare mode failed: {exc}")

    def _apply_camera_preset(self) -> None:
            try:
                self.viewport.apply_camera_preset(self.viewer_camera_combo.currentData())
                self.log(f"Camera preset: {self.viewer_camera_combo.currentData()}")
            except Exception as exc:
                self.log(f"Camera preset failed: {exc}")

    def _reset_camera_from_quickbar(self) -> None:
            try:
                self.viewport.reset_camera()
            except Exception as exc:
                self.log(f"Reset camera failed: {exc}")

    def _apply_clip(self) -> None:
            try:
                self.viewport.set_clip_plane(
                    self.viewer_clip_axis_combo.currentData(),
                    self.viewer_clip_fraction_spin.value(),
                    invert=self.viewer_clip_invert_check.isChecked(),
                )
                self.log(
                    f"Clip set: axis={self.viewer_clip_axis_combo.currentData()} "
                    f"fraction={self.viewer_clip_fraction_spin.value():.2f} "
                    f"invert={self.viewer_clip_invert_check.isChecked()}"
                )
            except Exception as exc:
                self.log(f"Clip failed: {exc}")
                QMessageBox.warning(self, "Clip failed", str(exc))

    def _drop_marker_at_last_pick(self) -> None:
            point = self.viewport.get_last_picked_world_pos()
            if point is None:
                QMessageBox.information(self, "No picked point", "Pick a point or face first.")
                return
            try:
                self.viewport.show_marker(point, name="last_pick_marker")
                self.log(f"Marker placed at: {point}")
            except Exception as exc:
                self.log(f"Marker placement failed: {exc}")
                QMessageBox.warning(self, "Marker failed", str(exc))

    def _on_viewer_edges_toggled(self, enabled: bool) -> None:
            if self._suppress_viewer_sync:
                return
            self.viewport.set_edges_visible(enabled)
            self.viewport_quick_preset_combo.clearFocus()

    def _on_viewer_grid_toggled(self, enabled: bool) -> None:
            if self._suppress_viewer_sync:
                return
            self.viewport.set_grid_visible(enabled)
            self._suppress_viewer_sync = True
            try:
                self.viewport_quick_grid_check.setChecked(enabled)
            finally:
                self._suppress_viewer_sync = False

    def _on_viewer_axes_toggled(self, enabled: bool) -> None:
            if self._suppress_viewer_sync:
                return
            self.viewport.set_axes_visible(enabled)
            self._suppress_viewer_sync = True
            try:
                self.viewport_quick_axes_check.setChecked(enabled)
            finally:
                self._suppress_viewer_sync = False

    def _on_quick_grid_toggled(self, enabled: bool) -> None:
            if self._suppress_viewer_sync:
                return
            self.viewport.set_grid_visible(enabled)
            self._suppress_viewer_sync = True
            try:
                self.viewer_grid_check.setChecked(enabled)
            finally:
                self._suppress_viewer_sync = False

    def _on_quick_axes_toggled(self, enabled: bool) -> None:
            if self._suppress_viewer_sync:
                return
            self.viewport.set_axes_visible(enabled)
            self._suppress_viewer_sync = True
            try:
                self.viewer_axes_check.setChecked(enabled)
            finally:
                self._suppress_viewer_sync = False

    def _set_viewport_info_visible(self, visible: bool) -> None:
            if self._suppress_viewer_sync:
                return
            if not hasattr(self.viewport, "set_host_info_visible"):
                return
            try:
                self.viewport.set_host_info_visible(bool(visible))
            except Exception as exc:
                self.log(f"Viewport diagnostics toggle failed: {exc}")
                return
            self._suppress_viewer_sync = True
            try:
                state = bool(self._safe_call(lambda: self.viewport.is_host_info_visible(), visible))
                self.viewer_diagnostics_check.setChecked(state)
                self.viewport_toggle_info_btn.setChecked(state)
                self.action_toggle_viewport_diagnostics.setChecked(state)
            finally:
                self._suppress_viewer_sync = False

    def _toggle_viewport_info_panel(self) -> None:
            if not hasattr(self.viewport, "toggle_host_info_visible"):
                return
            try:
                self.viewport.toggle_host_info_visible()
            except Exception as exc:
                self.log(f"Viewport diagnostics toggle failed: {exc}")
                return
            self._sync_viewer_controls_from_viewport_state()

    def capture_viewport_image(self) -> None:
            if not self.viewport.has_mesh():
                QMessageBox.information(self, "No mesh loaded", "Load a mesh before capturing a screenshot.")
                return
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save viewport screenshot",
                str(Path.cwd() / "viewport.png"),
                "PNG Image (*.png)",
            )
            if not path:
                return
            try:
                out = self.viewport.capture_image(path)
            except Exception as exc:
                QMessageBox.critical(self, "Screenshot failed", str(exc))
                self.log(f"Screenshot failed: {exc}")
                return
            self.log(f"Viewport screenshot saved: {out}")

    def _reset_viewport_original_from_current(self) -> None:
            current = self.viewport.get_current_mesh_data()
            if current is None:
                return
            self.viewport.set_original_mesh_data(current)
            self.viewport.set_compare_mode("current_only")
            self._set_combo_current_data(self.viewer_compare_combo, "current_only")
            self._set_combo_current_data(self.viewport_quick_compare_combo, "current_only")

    def _load_new_source_into_viewport(self, source_path: str) -> None:
            resolved = str(Path(source_path).expanduser().resolve())
            self.viewport.load_file(resolved)
            self._reset_viewport_original_from_current()
            self.selection_controller.reapply_after_refresh(reason="load_new_source")
            self._sync_viewport_ui_from_backend()

    def _capture_viewer_ui_state(self) -> dict[str, Any]:
            return {
                "preset": self.viewer_preset_combo.currentData(),
                "compare": self.viewer_compare_combo.currentData(),
                "show_edges": self.viewer_edges_check.isChecked(),
                "edge_width": self.viewer_edge_width_spin.value(),
                "show_grid": self.viewer_grid_check.isChecked(),
                "show_axes": self.viewer_axes_check.isChecked(),
                "show_boundary": self.viewer_boundary_check.isChecked(),
                "diagnostics": self.viewer_diagnostics_check.isChecked(),
            }

    def _restore_viewer_ui_state(self, state: dict[str, Any]) -> None:
            self._suppress_viewer_sync = True
            try:
                self._set_combo_current_data(self.viewer_preset_combo, state.get("preset", "inspection_edges"))
                self._set_combo_current_data(self.viewport_quick_preset_combo, state.get("preset", "inspection_edges"))
                self._set_combo_current_data(self.viewer_compare_combo, state.get("compare", "current_only"))
                self._set_combo_current_data(self.viewport_quick_compare_combo, state.get("compare", "current_only"))
                self.viewer_edges_check.setChecked(bool(state.get("show_edges", True)))
                self.viewer_edge_width_spin.setValue(float(state.get("edge_width", 1.5)))
                self.viewer_grid_check.setChecked(bool(state.get("show_grid", True)))
                self.viewer_axes_check.setChecked(bool(state.get("show_axes", True)))
                self.viewer_boundary_check.setChecked(bool(state.get("show_boundary", False)))
                self.viewer_diagnostics_check.setChecked(bool(state.get("diagnostics", False)))
                self.viewport_quick_grid_check.setChecked(self.viewer_grid_check.isChecked())
                self.viewport_quick_axes_check.setChecked(self.viewer_axes_check.isChecked())
                self.viewport_toggle_info_btn.setChecked(self.viewer_diagnostics_check.isChecked())
                self.action_toggle_viewport_diagnostics.setChecked(self.viewer_diagnostics_check.isChecked())
            finally:
                self._suppress_viewer_sync = False

    def _refresh_viewport_from_processor(self, preferred_source_path: str | None = None) -> None:
            viewer_state = self._capture_viewer_ui_state()
            self._clear_manual_edit_preview(silent=True)

            if preferred_source_path:
                resolved = str(Path(preferred_source_path).expanduser().resolve())
                if Path(resolved).is_file():
                    self.viewport.load_file(resolved)
                    self.current_output_path = resolved
                    self._set_mesh_info_path_only(resolved)
                    self._reset_viewport_original_from_current()
                    self._restore_viewer_ui_state(viewer_state)
                    self._reapply_viewer_state_to_viewport(viewer_state)
                    self.selection_controller.reapply_after_refresh(reason="refresh_from_output_file")
                    self.log(f"Viewport refreshed from output file: {resolved}")
                    self._sync_viewport_ui_from_backend()
                    return

            mesh = getattr(self.processor, "mesh", None)
            if mesh is not None:
                try:
                    self.viewport.load_trimesh(mesh, source_name="processor mesh")
                    self._set_mesh_info_from_trimesh(mesh)
                    self._restore_viewer_ui_state(viewer_state)
                    self._reapply_viewer_state_to_viewport(viewer_state)
                    self.selection_controller.reapply_after_refresh(reason="refresh_from_processor_mesh")
                    self.log("Viewport refreshed from processor mesh.")
                    self._sync_viewport_ui_from_backend()
                    return
                except Exception as exc:
                    self.log(f"Viewport mesh sync fallback: {exc}")

            candidate_path = self.current_output_path or self.current_mesh_path
            if candidate_path:
                resolved = str(Path(candidate_path).expanduser().resolve())
                if Path(resolved).is_file():
                    self.viewport.load_file(resolved)
                    self._set_mesh_info_path_only(resolved)
                    self._reset_viewport_original_from_current()
                    self._restore_viewer_ui_state(viewer_state)
                    self._reapply_viewer_state_to_viewport(viewer_state)
                    self.selection_controller.reapply_after_refresh(reason="refresh_from_fallback_path")
                    self.log(f"Viewport refreshed from fallback path: {resolved}")
                    self._sync_viewport_ui_from_backend()
                    return

            self.viewport.clear_scene()
            self._set_mesh_info_empty()
            self._reset_hole_fill_ui(status="No mesh loaded.")
            self.selection_controller.clear_selection(keep_mode=False, push=False, reason="refresh_clear_scene")
            self._sync_viewport_ui_from_backend()

    def _reapply_viewer_state_to_viewport(self, state: dict[str, Any]) -> None:
            try:
                self.viewport.set_edges_visible(bool(state.get("show_edges", True)))
            except Exception:
                pass
            try:
                self.viewport.set_edge_width(float(state.get("edge_width", 1.5)))
            except Exception:
                pass
            try:
                self.viewport.set_grid_visible(bool(state.get("show_grid", True)))
            except Exception:
                pass
            try:
                self.viewport.set_axes_visible(bool(state.get("show_axes", True)))
            except Exception:
                pass
            try:
                self.viewport.set_boundary_highlight_visible(bool(state.get("show_boundary", False)))
            except Exception:
                pass
            compare = state.get("compare")
            if compare:
                try:
                    self.viewport.set_compare_mode(compare)
                except Exception:
                    pass
            preset = state.get("preset")
            if preset:
                try:
                    self.viewport.apply_display_preset(preset)
                except Exception:
                    pass
            if hasattr(self.viewport, "set_host_info_visible"):
                try:
                    self.viewport.set_host_info_visible(bool(state.get("diagnostics", False)))
                except Exception:
                    pass

    def _safe_call(self, func: Callable[[], Any], default: Any) -> Any:
            try:
                return func()
            except Exception:
                return default

    def _safe_attr(self, name: str, default: Any) -> Any:
            return getattr(self.viewport, name, default)
