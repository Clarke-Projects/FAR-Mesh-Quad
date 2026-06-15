from __future__ import annotations

from typing import Any


class SelectionActionsMixin:
    """MainWindow-side selection and brush UI glue.

    SelectionController remains the semantic selection authority. This mixin
    only coordinates MainWindow widgets, viewport sync, and controller calls.
    """

    def _seed_selection_controller_from_ui_defaults(self) -> None:
        raw_mode = (
            self.brush_selection_mode_combo.currentData()
            if hasattr(self, "brush_selection_mode_combo")
            else "face"
        )
        semantic_mode = self.selection_controller.semantic_mode_from_viewport(raw_mode)
        brush_enabled = bool(self.brush_enable_check.isChecked()) if hasattr(self, "brush_enable_check") else False
        boundary = bool(self.brush_boundary_check.isChecked()) if hasattr(self, "brush_boundary_check") else False

        self.selection_controller.set_mode(semantic_mode, push=False, reason="seed_from_ui")
        self.selection_controller.set_brush_enabled(brush_enabled, push=False, reason="seed_from_ui")
        self.selection_controller.set_boundary_highlight(boundary, push=False, reason="seed_from_ui")
        self.selection_controller.push_session_to_viewport(reason="seed_from_ui")

    def _selection_mode_from_ui_raw(self, value: str | None) -> str:
        return self.selection_controller.semantic_mode_from_viewport(value).value

    def _selection_mode_to_viewport_raw(self, value: object) -> str:
        raw = getattr(value, "value", value)
        return self.selection_controller.viewport_mode_from_semantic(raw)

    def _selected_edge_ids_for_gui(self) -> tuple[int, ...]:
        """Return selected edge ids when the controller/backend supports them.

        Older tests and fallback controller doubles may not expose
        selected_edge_ids() yet.  Edge selection is optional for those paths,
        so GUI state should degrade to an empty edge selection instead of
        failing.
        """
        getter = getattr(self.selection_controller, "selected_edge_ids", None)
        if callable(getter):
            try:
                return tuple(int(v) for v in (getter() or ()))
            except Exception:
                return ()

        state = getattr(self.selection_controller, "state", None)
        raw = getattr(state, "selected_edge_ids", ()) if state is not None else ()
        try:
            return tuple(int(v) for v in (raw or ()))
        except Exception:
            return ()

    def _current_semantic_mode_value(self) -> str:
        mode = getattr(getattr(self.selection_controller, "state", None), "mode", None)
        return str(getattr(mode, "value", "none") or "none").strip().lower()

    def _preferred_manual_selection_kind(self) -> str:
        try:
            mode = self.selection_controller.preferred_manual_mode().value
            if mode == "face":
                return "faces"
            if mode == "vertex":
                return "vertices"
        except Exception:
            pass

        face_ids = self.selection_controller.selected_face_ids()
        vertex_ids = self.selection_controller.selected_vertex_ids()
        edge_ids = self._selected_edge_ids_for_gui()
        if len(face_ids) > 0:
            return "faces"
        if len(vertex_ids) > 0:
            return "vertices"
        if len(edge_ids) > 0:
            return "edges"
        return "none"

    def _selection_summary(self) -> dict[str, Any]:
        face_ids = self.selection_controller.selected_face_ids()
        vertex_ids = self.selection_controller.selected_vertex_ids()
        edge_ids = self._selected_edge_ids_for_gui()

        actual_mode = getattr(getattr(self.selection_controller, "state", None), "mode", None)
        actual_mode_value = str(getattr(actual_mode, "value", "none") or "none").strip().lower()

        session_mode = getattr(getattr(self.selection_controller, "session", None), "mode", None)
        session_mode_value = str(getattr(session_mode, "value", "none") or "none").strip().lower()

        preferred_mode = self._preferred_manual_selection_kind()
        viewport_mode = self.selection_controller.viewport_mode_from_semantic(session_mode)
        brush_enabled = bool(getattr(getattr(self.selection_controller, "session", None), "brush_enabled", False))

        if preferred_mode == "faces":
            normalized_mode = "face"
        elif preferred_mode == "vertices":
            normalized_mode = "vertex"
        elif preferred_mode == "edges":
            normalized_mode = "edge"
        else:
            normalized_mode = preferred_mode

        return {
            "mode": normalized_mode,
            "actual_mode": actual_mode_value,
            "session_mode": session_mode_value,
            "viewport_mode": viewport_mode,
            "brush_enabled": brush_enabled,
            "selected_faces": len(face_ids),
            "selected_vertices": len(vertex_ids),
            "selected_edges": len(edge_ids),
        }

    def _update_brush_action_state(self, caps: dict[str, bool] | None = None) -> None:
        if not hasattr(self, "brush_selection_info_label"):
            return

        if caps is None:
            caps = self._safe_viewport_capabilities()

        summary = self._selection_summary()
        face_ids = self.selection_controller.selected_face_ids()
        vertex_ids = self.selection_controller.selected_vertex_ids()
        edge_ids = self._selected_edge_ids_for_gui()
        mode = str(summary.get("mode", "none"))
        viewport_mode = str(summary.get("viewport_mode", "none"))
        brush_enabled = bool(summary.get("brush_enabled", False))

        self.brush_selection_info_label.setText(
            f"Mode: {viewport_mode} | Faces: {len(face_ids)} | Points: {len(vertex_ids)} | Edges: {len(edge_ids)} | Brush: {'on' if brush_enabled else 'off'}"
        )

        face_active = mode == "face"
        vertex_active = mode == "vertex"
        edge_active = mode == "edge"
        connected_points_enabled = self._connected_point_capability_enabled(caps)

        can_grow_or_shrink = (
            (face_active and len(face_ids) > 0)
            or (vertex_active and len(vertex_ids) > 0)
            or (edge_active and len(edge_ids) > 0)
        )

        self.brush_grow_btn.setEnabled(
            can_grow_or_shrink and hasattr(self.viewport, "grow_selection")
        )
        self.brush_shrink_btn.setEnabled(
            can_grow_or_shrink and hasattr(self.viewport, "shrink_selection")
        )
        self.brush_connected_points_btn.setEnabled(
            vertex_active
            and len(vertex_ids) > 0
            and connected_points_enabled
            and hasattr(self.viewport, "select_connected_points_from_vertex")
        )

        has_mesh = getattr(self.processor, "mesh", None) is not None or self.current_mesh_path is not None
        busy = self._worker is not None

        if hasattr(self, "topology_analyze_btn"):
            self.topology_analyze_btn.setEnabled(has_mesh and not busy)
        if hasattr(self, "topology_find_holes_btn"):
            self.topology_find_holes_btn.setEnabled(has_mesh and not busy)
        has_hole_candidates = bool(getattr(self, "_last_hole_candidates", []))
        has_selected_edge_boundary = bool(has_mesh and edge_active and len(edge_ids) > 0)
        has_hole_fill_boundary_source = bool(has_hole_candidates or has_selected_edge_boundary)

        if hasattr(self, "hole_fill_candidate_combo"):
            self.hole_fill_candidate_combo.setEnabled(has_hole_candidates and not busy)
        if hasattr(self, "hole_fill_method_combo"):
            self.hole_fill_method_combo.setEnabled(has_hole_fill_boundary_source and not busy)
        if hasattr(self, "hole_fill_max_area_spin"):
            self.hole_fill_max_area_spin.setEnabled(has_mesh and not busy)
        if hasattr(self, "hole_fill_max_perimeter_spin"):
            self.hole_fill_max_perimeter_spin.setEnabled(has_mesh and not busy)
        if hasattr(self, "hole_fill_preview_btn"):
            self.hole_fill_preview_btn.setEnabled(has_hole_fill_boundary_source and not busy)
        is_batch_preview = False
        checker = getattr(self, "_hole_fill_preview_is_batch", None)
        if callable(checker):
            try:
                is_batch_preview = bool(checker(self._hole_fill_preview))
            except Exception:
                is_batch_preview = False
        if hasattr(self, "hole_fill_commit_btn"):
            self.hole_fill_commit_btn.setEnabled(
                self._hole_fill_preview is not None and not is_batch_preview and not busy
            )
        if hasattr(self, "hole_fill_cancel_btn"):
            self.hole_fill_cancel_btn.setEnabled(self._hole_fill_preview is not None and not busy)
        if hasattr(self, "hole_fill_batch_preview_btn"):
            self.hole_fill_batch_preview_btn.setEnabled(has_hole_candidates and not busy)
        if hasattr(self, "hole_fill_batch_commit_btn"):
            is_batch_preview = False
            checker = getattr(self, "_hole_fill_preview_is_batch", None)
            if callable(checker):
                try:
                    is_batch_preview = bool(checker(self._hole_fill_preview))
                except Exception:
                    is_batch_preview = False
            self.hole_fill_batch_commit_btn.setEnabled(
                self._hole_fill_preview is not None and is_batch_preview and not busy
            )

        if hasattr(self, "manual_edit_preview_btn"):
            op = self.manual_edit_operation_combo.currentData() if hasattr(self, "manual_edit_operation_combo") else None

            valid_for_op = False
            if op in {"delete_faces", "group_cleanup", "group_reduce"}:
                valid_for_op = len(face_ids) > 0
            elif op == "delete_vertices":
                valid_for_op = len(vertex_ids) > 0
            elif op in {"cleanup", "reduce", "smooth_laplacian", "clip_plane"}:
                valid_for_op = len(face_ids) > 0 or len(vertex_ids) > 0

            self.manual_edit_preview_btn.setEnabled(has_mesh and valid_for_op and not busy)
            self.manual_edit_commit_btn.setEnabled(self._manual_edit_preview is not None and not busy)
            self.manual_edit_cancel_btn.setEnabled(self._manual_edit_preview is not None and not busy)

    def _on_page_shown(self, key: str) -> None:
        if key == self.PAGE_BRUSH:
            self.selection_controller.push_session_to_viewport(reason="page_shown_brush")
            self._sync_viewport_ui_from_backend()

    def _on_brush_page_requested(self) -> None:
        self.selection_controller.push_session_to_viewport(reason="brush_page_requested")
        self._sync_viewport_ui_from_backend()

    def _apply_brush_mode_from_ui(self) -> None:
        if not hasattr(self, "brush_selection_mode_combo"):
            return

        raw_mode = str(self.brush_selection_mode_combo.currentData() or "face")
        semantic_mode = self.selection_controller.semantic_mode_from_viewport(raw_mode)
        enabled = bool(self.brush_enable_check.isChecked()) if hasattr(self, "brush_enable_check") else False

        self.selection_controller.apply_brush_mode(
            semantic_mode,
            enabled=enabled,
            reason="apply_brush_mode_from_ui",
        )
        self._sync_viewport_ui_from_backend()
        self.statusBar().showMessage(
            f"Selection mode active: {self.selection_controller.viewport_mode_from_semantic(semantic_mode)} | drag {'on' if enabled else 'off'}",
            2000,
        )

    def _on_brush_selection_mode_changed(self) -> None:
        if self._suppress_viewer_sync:
            return
        self._apply_brush_mode_from_ui()
        self._on_manual_edit_operation_changed()

    def _on_brush_enabled_toggled(self, enabled: bool) -> None:
        if self._suppress_viewer_sync:
            return

        raw_mode = (
            self.brush_selection_mode_combo.currentData()
            if hasattr(self, "brush_selection_mode_combo")
            else "face"
        )
        semantic_mode = self.selection_controller.semantic_mode_from_viewport(raw_mode)

        self.selection_controller.apply_brush_mode(
            semantic_mode,
            enabled=bool(enabled),
            reason="brush_enabled_toggled",
        )
        self._sync_viewport_ui_from_backend()

    def _on_brush_boundary_toggled(self, enabled: bool) -> None:
        if self._suppress_viewer_sync:
            return
        self.selection_controller.set_boundary_highlight(bool(enabled), reason="brush_boundary_toggled")
        self._sync_viewport_ui_from_backend()

    def _grow_current_selection(self) -> None:
        changed = bool(self.selection_controller.grow_face_selection())
        if not changed:
            self.selection_controller.sync_from_viewport(reason="grow_face_selection_noop")
        self._update_brush_action_state()

    def _shrink_current_selection(self) -> None:
        changed = bool(self.selection_controller.shrink_face_selection())
        if not changed:
            self.selection_controller.sync_from_viewport(reason="shrink_face_selection_noop")
        self._update_brush_action_state()

    def _select_connected_points_from_current(self) -> None:
        changed = bool(self.selection_controller.select_connected_points_from_current())
        if not changed:
            self.selection_controller.sync_from_viewport(reason="select_connected_points_noop")
        self._update_brush_action_state()

    def _on_selection_mode_changed(self) -> None:
        if self._suppress_viewer_sync:
            return

        raw_mode = str(self.viewer_selection_combo.currentData() or "none")
        semantic_mode = self.selection_controller.semantic_mode_from_viewport(raw_mode)

        self.selection_controller.apply_viewer_mode(
            semantic_mode,
            reason="viewer_selection_mode_changed",
        )

        self._suppress_viewer_sync = True
        try:
            brush_raw = self.selection_controller.viewport_mode_from_semantic(semantic_mode)
            if hasattr(self, "brush_selection_mode_combo") and brush_raw in {"face", "point", "edge"}:
                self._set_combo_current_data(self.brush_selection_mode_combo, brush_raw)
            if hasattr(self, "brush_enable_check"):
                self.brush_enable_check.setChecked(False)
        finally:
            self._suppress_viewer_sync = False

        self._sync_viewport_ui_from_backend()
        self._on_manual_edit_operation_changed()
        self._update_brush_action_state()
        self.statusBar().showMessage(
            f"Selection mode: {self.selection_controller.viewport_mode_from_semantic(semantic_mode)} (viewer interaction)",
            2000,
        )

    def _clear_viewport_selection(self) -> None:
        self.selection_controller.clear_selection(
            keep_mode=True,
            push=True,
            reason="clear_viewport_selection",
        )
        self._sync_viewport_ui_from_backend()
        self.statusBar().showMessage("Selection cleared.", 2000)
