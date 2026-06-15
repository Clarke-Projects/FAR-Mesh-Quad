from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox

from .worker import WorkerThread


def _update_project_status_ui_if_available(window: object) -> None:
    """Refresh project status widgets when the current object provides them.

    Some focused GUI-adapter tests call MainWindow methods on light fake
    window objects. Those fakes intentionally do not build the full UI, so
    the I3 project-status refresh hook must be optional in adapter paths.
    """
    updater = getattr(window, "_update_project_status_ui", None)
    if callable(updater):
        updater()


class TaskRunnerMixin:
    """GUI task/worker orchestration for MainWindow.

    This mixin intentionally remains in far_mesh.gui: it coordinates Qt widgets,
    QMessageBox display, status bar updates, the GUI WorkerThread, and the
    two-phase close handoff after background work finishes. It is not part of
    the system execution layer.
    """

    def _set_busy(self, busy: bool) -> None:
        has_mesh = getattr(self.processor, "mesh", None) is not None or self.current_mesh_path is not None
        self.load_btn.setEnabled(not busy)
        self.load_file_btn.setEnabled(not busy)
        self.repair_btn.setEnabled(not busy)
        self.repair_btn_run.setEnabled(has_mesh and not busy)
        self.remesh_btn.setEnabled(not busy)
        self.remesh_btn_run.setEnabled(has_mesh and not busy)
        reduce_backend = self.reduce_backend_combo.currentData() if self.reduce_backend_combo.count() else None
        reduce_enabled = has_mesh and not busy and reduce_backend not in (None, "unavailable")
        self.reduce_btn.setEnabled(not busy)
        self.reduce_btn_run.setEnabled(reduce_enabled)
        self.save_btn.setEnabled(has_mesh and not busy)
        if hasattr(self, "action_open_project"):
            self.action_open_project.setEnabled(not busy)
        if hasattr(self, "action_save_project"):
            self.action_save_project.setEnabled(not busy)
        if hasattr(self, "action_save_project_as"):
            self.action_save_project_as.setEnabled(not busy)
        if hasattr(self, "load_open_project_btn"):
            self.load_open_project_btn.setEnabled(not busy)
        if hasattr(self, "load_save_project_btn"):
            self.load_save_project_btn.setEnabled(not busy)
        if hasattr(self, "load_save_project_as_btn"):
            self.load_save_project_as_btn.setEnabled(not busy)
        self.action_save.setEnabled(has_mesh and not busy)
        self.viewer_btn.setEnabled(not busy)
        self.brush_btn.setEnabled(not busy)
        self.viewer_screenshot_btn.setEnabled(has_mesh and not busy)
        if hasattr(self, "topology_analyze_btn"):
            self.topology_analyze_btn.setEnabled(has_mesh and not busy)
        if hasattr(self, "topology_find_holes_btn"):
            self.topology_find_holes_btn.setEnabled(has_mesh and not busy)
        has_hole_candidates = bool(getattr(self, "_last_hole_candidates", []))
        if hasattr(self, "hole_fill_candidate_combo"):
            self.hole_fill_candidate_combo.setEnabled(has_hole_candidates and not busy)
        if hasattr(self, "hole_fill_method_combo"):
            self.hole_fill_method_combo.setEnabled(has_hole_candidates and not busy)
        if hasattr(self, "hole_fill_max_area_spin"):
            self.hole_fill_max_area_spin.setEnabled(has_mesh and not busy)
        if hasattr(self, "hole_fill_max_perimeter_spin"):
            self.hole_fill_max_perimeter_spin.setEnabled(has_mesh and not busy)
        if hasattr(self, "hole_fill_preview_btn"):
            self.hole_fill_preview_btn.setEnabled(has_hole_candidates and not busy)
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
            if busy:
                self.manual_edit_preview_btn.setEnabled(False)
                self.manual_edit_commit_btn.setEnabled(False)
                self.manual_edit_cancel_btn.setEnabled(False)
            else:
                self._update_brush_action_state()
        self.progress_bar.setVisible(busy)
        self._update_undo_redo_action_state()

    def _run_task(
        self,
        description: str,
        task: Callable[[], object],
        success_handler: Callable[[object], None],
        failure_handler: Callable[[object], object | None] | None = None,
    ) -> None:
        if self._worker is not None:
            QMessageBox.information(self, "Busy", "Please wait for the current operation to finish.")
            return

        self.log(description)
        self.statusBar().showMessage(description)
        self._set_busy(True)

        worker = WorkerThread(task, self)
        self._worker = worker

        def on_success(payload: object) -> None:
            if getattr(self, "_closing", False):
                self.log("Task finished during application shutdown; result ignored.")
                return

            try:
                success_handler(payload)
            except Exception as exc:
                self.log(f"Post-processing failed: {exc}")
                QMessageBox.critical(self, "Error", str(exc))

        def on_failure(tb: str) -> None:
            if getattr(self, "_closing", False):
                self.log("Task cancelled during application shutdown.")
                return

            self.log(tb)
            display_message = str(tb)
            if failure_handler is not None:
                try:
                    handled_message = failure_handler(tb)
                except Exception as exc:
                    self.log(f"Failure post-processing failed: {exc}")
                else:
                    if isinstance(handled_message, str) and handled_message.strip():
                        display_message = handled_message

            _update_project_status_ui_if_available(self)
            QMessageBox.critical(self, "Operation failed", display_message)

        def on_finished() -> None:
            self._worker = None

            if getattr(self, "_closing", False):
                try:
                    self._set_busy(False)
                except Exception:
                    pass

                if getattr(self, "_close_after_worker", False):
                    self._close_after_worker = False
                    QTimer.singleShot(0, self.close)

                return

            self._set_busy(False)
            _update_project_status_ui_if_available(self)
            self.statusBar().showMessage("Ready", 3000)

        worker.succeeded.connect(on_success)
        worker.failed.connect(on_failure)
        worker.finished.connect(on_finished)
        worker.start()
