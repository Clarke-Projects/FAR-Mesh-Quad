from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

from far_mesh.core.manual_edit_pipeline import ManualEditResult

from .project_actions import _update_project_status_ui_if_available


class HistoryActionsMixin:
    """Undo/redo GUI-controller behavior for MainWindow."""

    def _update_undo_redo_action_state(self) -> None:
        busy = self._worker is not None
        can_undo = bool(self._safe_call(lambda: self.processor.can_undo(), False))
        can_redo = bool(self._safe_call(lambda: self.processor.can_redo(), False))

        if hasattr(self, "action_undo"):
            self.action_undo.setEnabled(can_undo and not busy)
        if hasattr(self, "action_redo"):
            self.action_redo.setEnabled(can_redo and not busy)

    def _apply_mesh_history_navigation_result(
        self,
        result: ManualEditResult,
        *,
        label: str,
    ) -> None:
        self.current_output_path = None

        self._clear_manual_edit_preview(silent=True)
        self._clear_hole_fill_preview(silent=True)

        self._refresh_viewport_from_processor()
        self._set_mesh_info_from_trimesh(self.processor.mesh)

        try:
            self.selection_controller.clear_selection(
                keep_mode=True,
                push=True,
                reason=f"{label.lower()}_mesh_operation",
            )
        except Exception:
            pass

        if hasattr(self, "manual_edit_status_label"):
            self.manual_edit_status_label.setText(
                f"{label}: {result.operation} | "
                f"faces {result.before_faces} -> {result.after_faces} | "
                f"vertices {result.before_vertices} -> {result.after_vertices}"
            )

        self._reset_hole_fill_ui(
            status=(
                f"{label} complete: faces {result.before_faces} -> {result.after_faces} | "
                f"vertices {result.before_vertices} -> {result.after_vertices}. "
                "Run Find Hole Candidates again if needed."
            )
        )

        self._set_topology_result_text(
            "\n".join(
                [
                    f"{label} complete",
                    f"Operation: {result.operation}",
                    f"Faces: {result.before_faces} -> {result.after_faces}",
                    f"Vertices: {result.before_vertices} -> {result.after_vertices}",
                    "",
                    "Viewport refreshed from MeshProcessor.",
                    "Selection, stale previews, and hole candidates were cleared.",
                ]
            )
        )

        for note in result.notes:
            self.log(f"{label} note: {note}")

        self.log(
            f"{label} complete: {result.operation} | "
            f"faces {result.before_faces} -> {result.after_faces} | "
            f"vertices {result.before_vertices} -> {result.after_vertices}"
        )
        self.statusBar().showMessage(f"{label} complete", 3000)
        self._show_page(self.PAGE_VIEWER)
        self._sync_viewport_ui_from_backend()
        self._update_undo_redo_action_state()
        _update_project_status_ui_if_available(self)

    def undo_mesh_operation(self) -> None:
        if not self.processor.can_undo():
            QMessageBox.information(self, "Nothing to undo", "No mesh operation is available to undo.")
            self._update_undo_redo_action_state()
            _update_project_status_ui_if_available(self)
            return

        def task() -> object:
            return self.processor.undo_last_mesh_operation()

        def on_success(result: object) -> None:
            assert isinstance(result, ManualEditResult)
            self._apply_mesh_history_navigation_result(result, label="Undo")

        self._run_task("Undoing last mesh operation...", task, on_success)

    def redo_mesh_operation(self) -> None:
        if not self.processor.can_redo():
            QMessageBox.information(self, "Nothing to redo", "No mesh operation is available to redo.")
            self._update_undo_redo_action_state()
            _update_project_status_ui_if_available(self)
            return

        def task() -> object:
            return self.processor.redo_last_mesh_operation()

        def on_success(result: object) -> None:
            assert isinstance(result, ManualEditResult)
            self._apply_mesh_history_navigation_result(result, label="Redo")

        self._run_task("Redoing mesh operation...", task, on_success)
