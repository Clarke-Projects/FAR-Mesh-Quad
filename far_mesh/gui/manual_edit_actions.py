from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtWidgets import QFileDialog, QMessageBox

from far_mesh.core.manual_edit_pipeline import (
    ManualEditPreview,
    ManualEditRequest,
    ManualEditResult,
    ManualSelection,
)
from far_mesh.core.quad_group_adapter import QuadGroupProcessOptions

from .project_actions import _update_project_status_ui_if_available


class ManualEditActionsMixin:
    """GUI-controller behavior for preview-first manual edit workflows."""

    def _effective_manual_edit_operation(
        self,
        operation: str | None,
        selection: ManualSelection,
    ) -> str | None:
        if operation in {"cleanup", "reduce"} and selection.mode == "faces":
            return "group_cleanup" if operation == "cleanup" else "group_reduce"
        return operation

    def _on_manual_edit_operation_changed(self) -> None:
        if not hasattr(self, "manual_edit_operation_combo"):
            return

        op = self.manual_edit_operation_combo.currentData()
        preferred_kind = self._preferred_manual_selection_kind()
        grouped = op in {"group_cleanup", "group_reduce"} or (op in {"cleanup", "reduce"} and preferred_kind == "faces")

        roi_reduce = op == "reduce" and not grouped
        roi_cleanup = op == "cleanup" and not grouped
        roi_smooth = op == "smooth_laplacian"
        roi_clip = op == "clip_plane"
        delete_faces = op == "delete_faces"
        delete_vertices = op == "delete_vertices"

        self.manual_edit_target_faces_spin.setEnabled(roi_reduce)
        self.manual_edit_boundary_weight_spin.setEnabled(roi_reduce or grouped)
        self.manual_edit_allow_non_manifold_check.setEnabled(roi_cleanup or grouped)
        self.manual_edit_smooth_iters_spin.setEnabled(roi_smooth)
        self.manual_edit_smooth_lambda_spin.setEnabled(roi_smooth)

        for spin in (
            getattr(self, "manual_edit_clip_point_x_spin", None),
            getattr(self, "manual_edit_clip_point_y_spin", None),
            getattr(self, "manual_edit_clip_point_z_spin", None),
            getattr(self, "manual_edit_clip_normal_x_spin", None),
            getattr(self, "manual_edit_clip_normal_y_spin", None),
            getattr(self, "manual_edit_clip_normal_z_spin", None),
        ):
            if spin is not None:
                spin.setEnabled(roi_clip)

        self.manual_edit_group_decode_combo.setEnabled(grouped)
        self.manual_edit_texture_path_edit.setEnabled(grouped)
        self.manual_edit_texture_browse_btn.setEnabled(grouped)
        self.manual_edit_group_target_ratio_spin.setEnabled(op == "group_reduce")

        if hasattr(self, "manual_edit_info_label"):
            if grouped and op in {"cleanup", "reduce"} and preferred_kind == "faces":
                self.manual_edit_info_label.setText(
                    "Face cleanup / reduce are automatically routed through patch-aware group processing and return a full merged preview."
                )
            elif grouped:
                self.manual_edit_info_label.setText(
                    "Patch-aware group processing routes through the QuadWild group adapter and returns a full merged preview."
                )
            elif delete_faces or delete_vertices:
                self.manual_edit_info_label.setText(
                    "Delete operations preview the full resulting mesh and can be committed directly."
                )
            elif roi_clip:
                self.manual_edit_info_label.setText(
                    "Clip preview shows only the clipped ROI. Commit stays disabled until a whole-mesh merge strategy is added."
                )
            else:
                self.manual_edit_info_label.setText(
                    "Manual edit preview uses the current viewport selection as its input. "
                    "Point-based cleanup / reduce / smooth previews are intentionally non-committable in the initial patch."
                )

        self._update_brush_action_state()

    def _on_manual_edit_browse_texture_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose grouping texture",
            str(Path.cwd()),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if not path:
            return
        self.manual_edit_texture_path_edit.setText(str(Path(path).expanduser().resolve()))

    def _get_current_selection_payload(self) -> tuple[str, list[int], list[int]]:
        mode = self.selection_controller.preferred_manual_mode().value
        selected_faces = [int(v) for v in self.selection_controller.selected_face_ids()]
        selected_points = [int(v) for v in self.selection_controller.selected_vertex_ids()]
        return mode, selected_faces, selected_points

    def _build_manual_edit_request(self) -> tuple[ManualEditRequest, QuadGroupProcessOptions | None]:
        operation = self.manual_edit_operation_combo.currentData()
        mode, selected_faces, selected_points = self._get_current_selection_payload()

        if operation in {"group_cleanup", "group_reduce", "delete_faces"}:
            if not selected_faces:
                raise ValueError("This operation requires a face selection.")
            selection = ManualSelection(
                mode="faces",
                face_ids=np.asarray(selected_faces, dtype=np.int64),
                source_path=self.current_output_path or self.current_mesh_path,
            )
        elif operation == "delete_vertices":
            if not selected_points:
                raise ValueError("This operation requires a vertex selection.")
            selection = ManualSelection(
                mode="vertices",
                vertex_ids=np.asarray(selected_points, dtype=np.int64),
                source_path=self.current_output_path or self.current_mesh_path,
            )
        else:
            preferred_kind = self._preferred_manual_selection_kind()
            if mode == "vertex" and selected_points:
                selection = ManualSelection(
                    mode="vertices",
                    vertex_ids=np.asarray(selected_points, dtype=np.int64),
                    source_path=self.current_output_path or self.current_mesh_path,
                )
            elif preferred_kind == "faces" and selected_faces:
                selection = ManualSelection(
                    mode="faces",
                    face_ids=np.asarray(selected_faces, dtype=np.int64),
                    source_path=self.current_output_path or self.current_mesh_path,
                )
            elif selected_points:
                selection = ManualSelection(
                    mode="vertices",
                    vertex_ids=np.asarray(selected_points, dtype=np.int64),
                    source_path=self.current_output_path or self.current_mesh_path,
                )
            else:
                raise ValueError("Select faces or points first.")

        params: dict[str, Any] = {
            "target_triangles": int(self.manual_edit_target_faces_spin.value()),
            "boundary_weight": float(self.manual_edit_boundary_weight_spin.value()),
            "allow_non_manifold_edge_removal": bool(self.manual_edit_allow_non_manifold_check.isChecked()),
            "number_of_iterations": int(self.manual_edit_smooth_iters_spin.value()),
            "lambda_filter": float(self.manual_edit_smooth_lambda_spin.value()),
            "point": [
                float(self.manual_edit_clip_point_x_spin.value()),
                float(self.manual_edit_clip_point_y_spin.value()),
                float(self.manual_edit_clip_point_z_spin.value()),
            ],
            "normal": [
                float(self.manual_edit_clip_normal_x_spin.value()),
                float(self.manual_edit_clip_normal_y_spin.value()),
                float(self.manual_edit_clip_normal_z_spin.value()),
            ],
        }

        effective_operation = self._effective_manual_edit_operation(operation, selection)

        req = ManualEditRequest(
            operation=effective_operation,
            selection=selection,
            preview_only=True,
            parameters=params,
        )

        group_opts: QuadGroupProcessOptions | None = None
        if effective_operation in {"group_cleanup", "group_reduce"}:
            texture_path = self.manual_edit_texture_path_edit.text().strip() or None
            group_opts = QuadGroupProcessOptions(
                decode_mode=str(self.manual_edit_group_decode_combo.currentData() or "auto"),
                texture_path=texture_path,
                cleanup=True,
                reduce=(effective_operation == "group_reduce"),
                target_ratio=float(self.manual_edit_group_target_ratio_spin.value()),
                boundary_weight=float(self.manual_edit_boundary_weight_spin.value()),
                allow_non_manifold_edge_removal=bool(
                    self.manual_edit_allow_non_manifold_check.isChecked()
                ),
            )

        return req, group_opts

    def _show_manual_preview_mesh(self, preview: ManualEditPreview) -> None:
        if not hasattr(self.viewport, "show_preview_mesh"):
            raise RuntimeError("The active viewport backend does not support preview meshes.")
        self.viewport.show_preview_mesh(
            preview.preview_mesh,
            color="#7ee787",
            opacity=0.35,
            show_edges=True,
        )

    def _clear_manual_edit_preview(self, *, silent: bool = False) -> None:
        self._manual_edit_preview = None
        if hasattr(self.viewport, "clear_preview_mesh"):
            try:
                self.viewport.clear_preview_mesh()
            except Exception:
                if not silent:
                    raise
        if hasattr(self, "manual_edit_status_label"):
            self.manual_edit_status_label.setText("No manual edit preview active.")
        if hasattr(self, "manual_edit_commit_btn"):
            self.manual_edit_commit_btn.setEnabled(False)
        if hasattr(self, "manual_edit_cancel_btn"):
            self.manual_edit_cancel_btn.setEnabled(False)
        self._update_brush_action_state()

    def _on_manual_edit_preview_clicked(self) -> None:
        if getattr(self.processor, "mesh", None) is None and self.current_mesh_path is None:
            QMessageBox.information(self, "No mesh loaded", "Load a mesh first.")
            return
        try:
            req, group_opts = self._build_manual_edit_request()
        except Exception as exc:
            QMessageBox.warning(self, "Manual edit preview", str(exc))
            self.log(f"Manual edit request failed: {exc}")
            return

        def task() -> object:
            return self.processor.build_manual_edit_preview_routed(
                req,
                group_opts=group_opts,
            )

        def on_success(result: object) -> None:
            assert isinstance(result, ManualEditPreview)
            self._manual_edit_preview = result
            self._show_manual_preview_mesh(result)
            summary = result.selection_summary or {}
            self.manual_edit_status_label.setText(
                f"Preview ready: {result.operation} | "
                f"mode={summary.get('mode')} | "
                f"faces={summary.get('selected_faces')} | "
                f"vertices={summary.get('selected_vertices')}"
            )
            committable = not any(note == "__ROI_ONLY_PREVIEW__" for note in result.notes)
            self.manual_edit_commit_btn.setEnabled(committable)
            self.manual_edit_cancel_btn.setEnabled(True)
            self._show_page(self.PAGE_BRUSH)
            self.selection_controller.push_session_to_viewport(reason="manual_preview_ready")
            self._update_brush_action_state()
            for note in result.notes:
                self.log(f"Manual edit preview note: {note}")
            if not committable:
                self.log("Manual edit preview is ROI-only and cannot be committed yet.")
            self.log(f"Manual edit preview ready: {result.operation}")

        self._run_task("Building manual edit preview...", task, on_success)

    def _on_manual_edit_commit_clicked(self) -> None:
        if self._manual_edit_preview is None:
            QMessageBox.information(self, "No preview", "Build a manual edit preview first.")
            return

        preview = self._manual_edit_preview
        if any(note == "__ROI_ONLY_PREVIEW__" for note in preview.notes):
            QMessageBox.information(
                self,
                "Preview only",
                "This preview changes only a local ROI and cannot be committed as a whole-mesh edit yet. "
                "For face cleanup or reduce, use the face brush workflow which routes through grouped processing.",
            )
            return

        def task() -> object:
            return self.processor.commit_manual_edit_preview(preview)

        def on_success(result: object) -> None:
            assert isinstance(result, ManualEditResult)
            self._clear_manual_edit_preview(silent=True)
            self.current_output_path = None
            self._refresh_viewport_from_processor()
            try:
                self.selection_controller.clear_selection(
                    keep_mode=True,
                    push=True,
                    reason="manual_edit_committed",
                )
            except Exception:
                pass
            self._reset_hole_fill_ui(status="Mesh changed. Run Find Hole Candidates again.")
            self._set_mesh_info_from_trimesh(self.processor.mesh)
            self.manual_edit_status_label.setText(
                f"Committed: {result.operation} | faces {result.before_faces} -> {result.after_faces} | "
                f"vertices {result.before_vertices} -> {result.after_vertices}"
            )
            for note in result.notes:
                self.log(f"Manual edit note: {note}")
            self.log(
                f"Manual edit committed: {result.operation} | "
                f"faces {result.before_faces} -> {result.after_faces} | "
                f"vertices {result.before_vertices} -> {result.after_vertices}"
            )
            self._show_page(self.PAGE_VIEWER)
            self._update_undo_redo_action_state()
            _update_project_status_ui_if_available(self)
            self._sync_viewport_ui_from_backend()

        self._run_task("Committing manual edit preview...", task, on_success)

    def _on_manual_edit_cancel_clicked(self) -> None:
        self._clear_manual_edit_preview(silent=True)
        self.statusBar().showMessage("Manual edit preview cleared.", 2000)
        self.log("Manual edit preview cleared.")
