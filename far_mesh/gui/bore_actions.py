"""Bore display mixin.

This is the only Bore-specific file in ``far_mesh/gui``. It is intentionally
display-only.

Clean architecture
------------------
    Edge selection -> core BoreTool -> BoreActions display

``bore_actions.py`` does not call region selection, recognition, component
classification, rebuild target construction, or rebuild. It renders
already-prepared DTOs returned by ``far_mesh.core.bore.tool`` and exposes legacy
Qt slot names as pass-through hooks to the core-owned ``BoreToolRuntime``.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import trimesh
from PySide6.QtWidgets import QMessageBox, QTreeWidgetItem

try:  # PySide enum spelling differs across minor versions/test stubs.
    from PySide6.QtCore import Qt
except Exception:  # pragma: no cover
    Qt = None  # type: ignore[assignment]

from .ui_helpers import FAR_SIGNAL_ORANGE
from far_mesh.core.bore.tool import BoreCandidateView, BoreInsideBoundaryPreview, BoreToolDisplayResult, BoreToolRuntime

RGBA = tuple[int, int, int, int]
DEFAULT_REBUILT_FACE_COLOR: RGBA = (0, 213, 255, 255)


class BoreActionsMixin:
    """Display-only Bore UI mixin.

    All Bore logic and workflow live outside this file. Methods starting with
    ``_on_bore_*`` remain only as compatibility signal endpoints and immediately
    delegate to the core-owned ``BoreToolRuntime``.
    """

    # ------------------------------------------------------------------
    # Core BoreTool access: compatibility hook, no Bore logic here
    # ------------------------------------------------------------------

    def _bore_tool(self) -> BoreToolRuntime:
        tool = getattr(self, "_bore_tool_runtime_instance", None)
        if tool is None:
            tool = BoreToolRuntime(self)
            self._bore_tool_runtime_instance = tool
        return tool

    # ------------------------------------------------------------------
    # Generic display helpers
    # ------------------------------------------------------------------

    def _bore_display_log(self, message: str) -> None:
        if hasattr(self, "log"):
            self.log(message)
        else:
            print(message)

    # Compatibility name used by older callers/tests.
    def _bore_log(self, message: str) -> None:
        self._bore_display_log(message)

    def _bore_display_status(self, message: str, timeout: int = 2000) -> None:
        if hasattr(self, "statusBar"):
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(message, timeout)

    def _bore_status(self, message: str, timeout: int = 2000) -> None:
        self._bore_display_status(message, timeout)

    def _bore_display_info(self, title: str, message: str) -> None:
        try:
            QMessageBox.information(self, title, message)
        except Exception:
            self._bore_display_log(f"{title}: {message}")
            self._bore_display_status(message, 3000)

    def _bore_display_critical(self, title: str, message: str) -> None:
        try:
            QMessageBox.critical(self, title, message)
        except Exception:
            self._bore_display_log(f"{title}: {message}")
            self._bore_display_status(message, 5000)

    def _bore_display_confirm(self, title: str, message: str) -> bool:
        try:
            answer = QMessageBox.question(
                self,
                title,
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            return bool(answer == QMessageBox.StandardButton.Yes)
        except Exception:
            self._bore_display_log(f"{title}: {message}")
            return True

    @staticmethod
    def _bore_display_int_tuple(values: object) -> tuple[int, ...]:
        try:
            return tuple(sorted({int(v) for v in tuple(values or ()) if int(v) >= 0}))
        except Exception:
            return ()

    # ------------------------------------------------------------------
    # Face-preview overlay.  Display only.
    # ------------------------------------------------------------------

    def _bore_clear_face_preview(self) -> None:
        viewport = getattr(self, "viewport", None)
        if viewport is None:
            return
        if hasattr(viewport, "clear_preview_mesh"):
            try:
                viewport.clear_preview_mesh()
            except Exception:
                pass
        elif hasattr(viewport, "show_preview_mesh"):
            try:
                viewport.show_preview_mesh(None)
            except Exception:
                pass
        for name in ("highlight_cells", "set_face_selection"):
            fn = getattr(viewport, name, None)
            if callable(fn):
                try:
                    fn(())
                except Exception:
                    pass

    @staticmethod
    def _bore_overlay_offset(mesh: trimesh.Trimesh) -> float:
        try:
            bounds = np.asarray(mesh.bounds, dtype=float)
            diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
        except Exception:
            diagonal = 1.0
        if not np.isfinite(diagonal) or diagonal <= 0.0:
            diagonal = 1.0
        return max(diagonal * 1.0e-5, 1.0e-6)

    def _bore_build_selected_face_overlay_mesh(self, face_ids: Iterable[int]) -> trimesh.Trimesh:
        mesh = getattr(getattr(self, "processor", None), "mesh", None)
        if mesh is None:
            raise ValueError("No mesh available for Bore face preview.")
        vertices = np.asarray(getattr(mesh, "vertices", ()), dtype=float)
        faces = np.asarray(getattr(mesh, "faces", ()), dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] < 3 or faces.ndim != 2 or faces.shape[1] < 3:
            raise ValueError("Mesh does not contain valid vertices/faces for Bore face preview.")
        ids = np.asarray(tuple(int(fid) for fid in face_ids), dtype=np.int64).reshape(-1)
        ids = ids[(ids >= 0) & (ids < len(faces))]
        if ids.size == 0:
            raise ValueError("No valid face IDs for Bore face preview.")

        offset = self._bore_overlay_offset(mesh)
        out_vertices: list[np.ndarray] = []
        out_faces: list[list[int]] = []
        for fid in ids.tolist():
            source_face = faces[int(fid), :3]
            face_vertices = vertices[source_face, :3]
            normal = np.cross(face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0])
            length = float(np.linalg.norm(normal))
            if np.isfinite(length) and length > 1.0e-12:
                normal = normal / length
            else:
                normal = np.zeros(3, dtype=float)
            start = len(out_vertices)
            lifted = face_vertices + normal.reshape(1, 3) * offset
            out_vertices.extend(lifted.copy())
            tri = [start, start + 1, start + 2]
            out_faces.append(tri)
            out_faces.append(list(reversed(tri)))
        return trimesh.Trimesh(vertices=np.asarray(out_vertices, dtype=float), faces=np.asarray(out_faces, dtype=np.int64), process=False)

    def _bore_highlight_faces(
        self,
        face_ids: Iterable[int],
        *,
        color: str = FAR_SIGNAL_ORANGE,
        semantic_selection: bool = False,
    ) -> None:
        ids = self._bore_display_int_tuple(face_ids)
        if not ids:
            self._bore_clear_face_preview()
            return
        viewport = getattr(self, "viewport", None)
        self._bore_clear_face_preview()
        if viewport is not None:
            solid_display_used = False

            # The WGPU viewport's filled-face display path is the old Bore region
            # visualization behavior.  Use it as a viewport-local display layer,
            # not as SelectionController semantic state.  This restores the
            # "selected edges translate to visible faces" behavior without
            # putting Bore decisions back into bore_actions.py.
            if hasattr(viewport, "set_face_selection"):
                try:
                    viewport.set_face_selection(ids, color=color)
                    solid_display_used = True
                except TypeError:
                    try:
                        viewport.set_face_selection(ids)
                        solid_display_used = True
                    except Exception:
                        solid_display_used = False
                except Exception:
                    solid_display_used = False

            overlay_used = False
            try:
                overlay = self._bore_build_selected_face_overlay_mesh(ids)
                if hasattr(viewport, "show_preview_mesh"):
                    try:
                        viewport.show_preview_mesh(overlay, color=color)
                    except TypeError:
                        viewport.show_preview_mesh(overlay)
                    overlay_used = True
                elif hasattr(viewport, "set_preview_mesh"):
                    viewport.set_preview_mesh(overlay)
                    overlay_used = True
            except Exception as exc:
                self._bore_display_log(f"Bore solid overlay mesh failed; trying viewport face/cell fallback: {exc}")

            # Edge/cell highlight is now only a last-resort fallback.  Using it
            # unconditionally made the Bore preview look like selected edges
            # instead of filled wall faces.
            if not solid_display_used and not overlay_used and hasattr(viewport, "highlight_cells"):
                try:
                    viewport.highlight_cells(ids, color=color)
                except TypeError:
                    try:
                        viewport.highlight_cells(ids)
                    except Exception:
                        pass
                except Exception:
                    pass

        # Display previews must not write into SelectionController.  This local
        # cache is only used by the Bore display/rebuild UI state.
        self._bore_selected_face_ids = ids

    # ------------------------------------------------------------------
    # Display DTO rendering
    # ------------------------------------------------------------------

    def _bore_display_clear_all(self, *, clear_semantic_selection: bool = False) -> None:
        self._bore_clear_face_preview()
        self._bore_display_clear_candidates()
        self._bore_selected_face_ids = ()
        if clear_semantic_selection:
            viewport = getattr(self, "viewport", None)
            if viewport is not None and hasattr(viewport, "set_face_selection"):
                try:
                    viewport.set_face_selection(())
                except Exception:
                    pass
        for name, text in (
            ("bore_analysis_text", ""),
            ("bore_preview_text", ""),
            ("bore_selected_faces_label", "Selected faces: 0"),
            ("bore_opposite_rim_label", "Selection boundary loops: -"),
            ("bore_preview_status_label", "No preview yet."),
            ("bore_boundary_status_label", ""),
        ):
            widget = getattr(self, name, None)
            if widget is None:
                continue
            try:
                if hasattr(widget, "setPlainText"):
                    widget.setPlainText(text)
                else:
                    widget.setText(text)
            except Exception:
                pass
        self._bore_display_set_action_state(
            has_mesh=getattr(getattr(self, "processor", None), "mesh", None) is not None,
            selected_edge_count=0,
            has_candidates=False,
            has_selected_candidate=False,
            has_preview=False,
            can_rebuild=False,
        )

    def _bore_display_clear_candidates(self) -> None:
        self._bore_display_result = None
        self._bore_display_selected_candidate_id = ""
        self._bore_display_previewed_candidate_id = ""
        combo = getattr(self, "bore_feature_candidate_combo", None)
        if combo is not None:
            try:
                combo.blockSignals(True)
                combo.clear()
                combo.setEnabled(False)
                combo.blockSignals(False)
            except Exception:
                pass
        tree = getattr(self, "bore_feature_candidate_tree", None)
        if tree is not None:
            try:
                tree.blockSignals(True)
                tree.clear()
                tree.setEnabled(False)
                tree.blockSignals(False)
            except Exception:
                pass
        label = getattr(self, "bore_feature_candidate_status_label", None)
        if label is not None:
            try:
                label.setText("No Bore candidates listed.")
            except Exception:
                pass
        self._bore_display_set_action_state(
            has_mesh=getattr(getattr(self, "processor", None), "mesh", None) is not None,
            selected_edge_count=0,
            has_candidates=False,
            has_selected_candidate=False,
            has_preview=False,
            can_rebuild=False,
        )

    def _bore_display_analysis_result(self, result: BoreToolDisplayResult) -> None:
        self._bore_display_result = result
        self._bore_display_selected_candidate_id = result.selected_candidate_id
        self._bore_display_previewed_candidate_id = ""

        if hasattr(self, "bore_analysis_text"):
            try:
                self.bore_analysis_text.setPlainText(result.analysis_text)
            except Exception:
                pass
        if hasattr(self, "bore_preview_text"):
            try:
                self.bore_preview_text.setPlainText(result.preview_text)
            except Exception:
                pass
        if hasattr(self, "bore_selected_faces_label"):
            try:
                self.bore_selected_faces_label.setText(f"Neutral volume cutout faces: {len(result.region_face_ids)}")
            except Exception:
                pass
        if hasattr(self, "bore_opposite_rim_label"):
            try:
                boundary_count = result.diagnostics.get("boundary_loop_count", result.diagnostics.get("wall_boundary_loop_count", "-")) if isinstance(result.diagnostics, Mapping) else "-"
                self.bore_opposite_rim_label.setText(f"Selection boundary loops: {boundary_count}")
            except Exception:
                pass
        if hasattr(self, "bore_boundary_status_label"):
            try:
                self.bore_boundary_status_label.setText(result.boundary_status_text)
            except Exception:
                pass
        if hasattr(self, "bore_preview_status_label"):
            try:
                self.bore_preview_status_label.setText(result.status_text)
            except Exception:
                pass

        self._bore_display_populate_candidate_widgets(result.candidates, selected_candidate_id=result.selected_candidate_id)
        has_region_preview = bool(getattr(result, "region_preview_face_ids", ()))
        if has_region_preview:
            self._bore_display_region_preview(result)
        self._bore_display_set_action_state(
            has_mesh=getattr(getattr(self, "processor", None), "mesh", None) is not None,
            selected_edge_count=len(result.normalized_edge_ids or result.selected_edge_ids),
            has_candidates=bool(result.candidates),
            has_selected_candidate=bool(result.selected_candidate_id),
            has_preview=has_region_preview,
            can_rebuild=False,
        )

    def _bore_display_region_preview(self, result: BoreToolDisplayResult) -> None:
        """Display the routed Region Select neutral volume cutout preview.

        This is not a recognized candidate preview and does not enable rebuild.
        It answers the first visual contract in the BoreTool diagram: selected
        edge/rim evidence must become a visible neutral mesh volume cutout before
        recognition candidate preview is inspected.
        """

        ids = self._bore_display_int_tuple(getattr(result, "region_preview_face_ids", ()))
        if not ids:
            return
        self._bore_display_previewed_candidate_id = ""
        self._bore_highlight_faces(ids, semantic_selection=False)
        if hasattr(self, "bore_selected_faces_label"):
            try:
                self.bore_selected_faces_label.setText(f"Neutral volume cutout faces: {len(ids)}")
            except Exception:
                pass
        if hasattr(self, "bore_preview_status_label"):
            try:
                self.bore_preview_status_label.setText(
                    f"Neutral volume cutout preview: {len(ids)} faces. Candidate preview is separate."
                )
            except Exception:
                pass

    def _bore_display_populate_candidate_widgets(self, candidates: tuple[BoreCandidateView, ...], *, selected_candidate_id: str = "") -> None:
        combo = getattr(self, "bore_feature_candidate_combo", None)
        if combo is not None:
            try:
                combo.blockSignals(True)
                combo.clear()
                for candidate in candidates:
                    combo.addItem(candidate.label, candidate.candidate_id)
                combo.setEnabled(bool(candidates))
                if selected_candidate_id:
                    for idx, candidate in enumerate(candidates):
                        if candidate.candidate_id == selected_candidate_id:
                            combo.setCurrentIndex(idx)
                            break
                combo.blockSignals(False)
            except Exception:
                pass

        tree = getattr(self, "bore_feature_candidate_tree", None)
        if tree is not None:
            try:
                tree.blockSignals(True)
                tree.clear()
                for idx, candidate in enumerate(candidates):
                    item = QTreeWidgetItem([candidate.table_object, candidate.table_faces, candidate.table_geometry, candidate.table_role])
                    if Qt is not None:
                        item.setData(0, Qt.ItemDataRole.UserRole, candidate.candidate_id)
                    try:
                        item.setToolTip(0, candidate.description)
                        item.setToolTip(2, candidate.description)
                    except Exception:
                        pass
                    tree.addTopLevelItem(item)
                    if candidate.candidate_id == selected_candidate_id:
                        tree.setCurrentItem(item)
                        try:
                            item.setSelected(True)
                        except Exception:
                            pass
                tree.setEnabled(bool(candidates))
                try:
                    tree.resizeColumnToContents(0)
                    tree.resizeColumnToContents(1)
                    tree.resizeColumnToContents(2)
                except Exception:
                    pass
                tree.blockSignals(False)
            except Exception:
                pass

        status = getattr(self, "bore_feature_candidate_status_label", None)
        if status is not None:
            try:
                if candidates:
                    rebuildable = sum(1 for item in candidates if item.can_rebuild)
                    status.setText(f"{len(candidates)} candidate(s) listed; {rebuildable} rebuild-authorized by BoreTool.")
                else:
                    status.setText("No candidates returned yet. Neutral volume preview remains active.")
            except Exception:
                pass

    def _bore_display_select_candidate(self, candidate_id: str) -> None:
        result = getattr(self, "_bore_display_result", None)
        if not isinstance(result, BoreToolDisplayResult):
            return
        candidate = result.candidate_by_id(candidate_id)
        if candidate is None:
            return
        self._bore_display_selected_candidate_id = candidate_id
        combo = getattr(self, "bore_feature_candidate_combo", None)
        if combo is not None:
            try:
                for idx, item in enumerate(result.candidates):
                    if item.candidate_id == candidate_id:
                        combo.setCurrentIndex(idx)
                        break
            except Exception:
                pass
        tree = getattr(self, "bore_feature_candidate_tree", None)
        if tree is not None:
            try:
                for idx, item in enumerate(result.candidates):
                    if item.candidate_id == candidate_id:
                        tree_item = tree.topLevelItem(idx)
                        if tree_item is not None:
                            tree.setCurrentItem(tree_item)
                        break
            except Exception:
                pass
        status = getattr(self, "bore_feature_candidate_status_label", None)
        if status is not None:
            try:
                status.setText(candidate.label)
            except Exception:
                pass
        self._bore_display_set_action_state(
            has_mesh=getattr(getattr(self, "processor", None), "mesh", None) is not None,
            selected_edge_count=0,
            has_candidates=True,
            has_selected_candidate=True,
            has_preview=bool(getattr(self, "_bore_display_previewed_candidate_id", "")),
            can_rebuild=False,
        )

    def _bore_display_preview_candidate(self, candidate: BoreCandidateView, *, auto: bool = False) -> None:
        self._bore_display_selected_candidate_id = candidate.candidate_id
        self._bore_display_previewed_candidate_id = candidate.candidate_id
        self._bore_highlight_faces(candidate.display_face_ids, semantic_selection=False)
        if hasattr(self, "bore_feature_candidate_status_label"):
            try:
                self.bore_feature_candidate_status_label.setText(candidate.label)
            except Exception:
                pass
        if hasattr(self, "bore_preview_status_label"):
            try:
                prefix = "Auto-previewing" if auto else "Previewing"
                self.bore_preview_status_label.setText(f"{prefix} {candidate.table_object} | {candidate.face_count} faces | {candidate.table_role}")
            except Exception:
                pass
        if hasattr(self, "bore_preview_text"):
            try:
                self.bore_preview_text.setPlainText(candidate.description)
            except Exception:
                pass
        self._bore_display_set_action_state(
            has_mesh=getattr(getattr(self, "processor", None), "mesh", None) is not None,
            selected_edge_count=0,
            has_candidates=True,
            has_selected_candidate=True,
            has_preview=True,
            can_rebuild=bool(candidate.can_rebuild),
        )

    def _bore_display_reset_candidate_preview(self) -> None:
        self._bore_display_previewed_candidate_id = ""
        self._bore_clear_face_preview()
        result = getattr(self, "_bore_display_result", None)
        if hasattr(self, "bore_selected_faces_label"):
            try:
                if isinstance(result, BoreToolDisplayResult):
                    self.bore_selected_faces_label.setText(f"Neutral volume cutout faces: {len(result.region_face_ids)}")
                else:
                    self.bore_selected_faces_label.setText("Neutral volume cutout faces: -")
            except Exception:
                pass
        if hasattr(self, "bore_preview_status_label"):
            try:
                self.bore_preview_status_label.setText("Candidate preview cleared.")
            except Exception:
                pass
        if hasattr(self, "bore_preview_text"):
            try:
                self.bore_preview_text.setPlainText("")
            except Exception:
                pass
        self._bore_display_set_action_state(
            has_mesh=getattr(getattr(self, "processor", None), "mesh", None) is not None,
            selected_edge_count=0,
            has_candidates=isinstance(result, BoreToolDisplayResult) and bool(result.candidates),
            has_selected_candidate=bool(getattr(self, "_bore_display_selected_candidate_id", "")),
            has_preview=False,
            can_rebuild=False,
        )

    def _bore_display_inside_boundary_preview(self, result: BoreInsideBoundaryPreview) -> None:
        self._bore_highlight_faces(result.face_ids, semantic_selection=False)
        if hasattr(self, "bore_selected_faces_label"):
            try:
                self.bore_selected_faces_label.setText(f"Selected interior faces: {len(result.face_ids)}")
            except Exception:
                pass
        if hasattr(self, "bore_analysis_text"):
            try:
                self.bore_analysis_text.setPlainText(result.analysis_text)
            except Exception:
                pass
        if hasattr(self, "bore_preview_status_label"):
            try:
                self.bore_preview_status_label.setText(result.status_text)
            except Exception:
                pass

    def _bore_display_edge_pick_active(self) -> None:
        if hasattr(self, "bore_boundary_status_label"):
            try:
                self.bore_boundary_status_label.setText("Edge-pick mode active. BoreTool will receive selected edge IDs as raw evidence.")
            except Exception:
                pass
        if hasattr(self, "bore_preview_status_label"):
            try:
                self.bore_preview_status_label.setText("Bore edge-pick mode active.")
            except Exception:
                pass
        if hasattr(self, "bore_preview_text"):
            try:
                self.bore_preview_text.setPlainText("Select Bore opening/rim edges. Display layer does not interpret the selection.")
            except Exception:
                pass

    def _bore_display_rebuild_success(
        self,
        *,
        result: object,
        candidate: BoreCandidateView,
        added_face_ids: tuple[int, ...],
        removed_count: int,
        added_count: int,
        before_faces: int,
        after_faces: int,
        quad_density_label: str,
    ) -> None:
        self._bore_clear_face_preview()
        if added_face_ids:
            self._bore_highlight_faces(added_face_ids, color="#00d5ff", semantic_selection=False)
        if hasattr(self, "bore_selected_faces_label"):
            try:
                self.bore_selected_faces_label.setText(f"Rebuilt faces: {len(added_face_ids)}")
            except Exception:
                pass
        diagnostics = getattr(result, "diagnostics", {}) or {}
        if hasattr(self, "bore_analysis_text"):
            lines = [
                "BoreTool rebuild committed",
                f"Candidate: {candidate.label}",
                f"Rebuild input faces: {len(candidate.rebuild_face_ids)}",
                f"Removed faces: {removed_count}",
                f"Added faces/quads: {added_count}",
                f"Faces: {before_faces} -> {after_faces}",
                "Status: Success",
                "",
            ]
            if isinstance(diagnostics, Mapping):
                for key in (
                    "mode",
                    "topology_policy",
                    "quad_density_mode",
                    "measured_patch_attempt_count",
                    "measured_patch_selected_attempt_index",
                    "measured_patch_boundary_loop_count",
                    "boundary_edge_count_after",
                    "watertight_after",
                ):
                    if key in diagnostics:
                        lines.append(f"{key}: {diagnostics[key]}")
            try:
                self.bore_analysis_text.setPlainText("\n".join(lines))
            except Exception:
                pass
        if hasattr(self, "bore_preview_status_label"):
            try:
                self.bore_preview_status_label.setText(f"Bore rebuild committed: removed {removed_count}, added {added_count}.")
            except Exception:
                pass
        if hasattr(self, "bore_preview_text"):
            try:
                self.bore_preview_text.setPlainText(
                    "Bore rebuild committed to the active mesh.\n"
                    f"Removed faces: {removed_count}\n"
                    f"Added faces/quads: {added_count}\n"
                    f"Quad density: {quad_density_label}\n"
                    f"Faces: {before_faces} -> {after_faces}"
                )
            except Exception:
                pass
        self._bore_display_clear_candidates()

    def _bore_display_error(self, message: str) -> None:
        if hasattr(self, "bore_analysis_text"):
            try:
                self.bore_analysis_text.setPlainText(f"Error: {message}")
            except Exception:
                pass
        if hasattr(self, "bore_preview_status_label"):
            try:
                self.bore_preview_status_label.setText("BoreTool error. See diagnostics.")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Button/action state display
    # ------------------------------------------------------------------

    def _bore_display_set_button_enabled(self, names: tuple[str, ...], enabled: bool) -> None:
        for name in names:
            widget = getattr(self, name, None)
            if widget is None:
                continue
            try:
                widget.setEnabled(bool(enabled))
            except Exception:
                pass

    def _bore_display_set_action_state(
        self,
        *,
        has_mesh: bool,
        selected_edge_count: int,
        has_candidates: bool,
        has_selected_candidate: bool,
        has_preview: bool,
        can_rebuild: bool,
    ) -> None:
        self._bore_display_set_button_enabled(("bore_select_opening_btn",), has_mesh)
        self._bore_display_set_button_enabled(("bore_select_inside_boundary_btn",), has_mesh and selected_edge_count > 0)
        self._bore_display_set_button_enabled(("bore_select_wall_faces_btn", "bore_list_candidates_btn"), has_mesh and selected_edge_count > 0)
        self._bore_display_set_button_enabled(
            (
                "bore_preview_feature_candidate_btn",
                "bore_preview_candidate_btn",
                "bore_preview_selected_object_btn",
                "bore_preview_selected_obj_btn",
                "bore_view_feature_candidate_btn",
                "bore_view_candidate_btn",
                "bore_view_selected_object_btn",
                "bore_view_selected_obj_btn",
            ),
            has_mesh and has_candidates and has_selected_candidate,
        )
        self._bore_display_set_button_enabled(
            ("bore_reset_candidate_preview_btn", "bore_reset_feature_candidate_preview_btn"),
            bool(has_preview),
        )
        self._bore_display_set_button_enabled(
            ("bore_rebuild_wall_faces_btn", "bore_rebuild_preview_btn"),
            has_mesh and bool(can_rebuild),
        )
        if hasattr(self, "bore_selected_edges_label"):
            try:
                self.bore_selected_edges_label.setText(f"Selected edges: {int(selected_edge_count)}")
            except Exception:
                pass

    # Compatibility name for existing update calls.
    def _update_bore_action_state(self) -> None:
        self._bore_tool()._push_action_state()

    def _bore_display_current_quad_density_mode(self) -> str:
        combo = getattr(self, "bore_quad_density_combo", None)
        if combo is not None:
            try:
                data = combo.currentData()
                if data:
                    return str(data)
            except Exception:
                pass
            try:
                text = str(combo.currentText()).lower()
                if "lean" in text or "low" in text:
                    return "lean_pi_opening"
                if "pi" in text or "balanced" in text:
                    return "pi_opening"
            except Exception:
                pass
        return "lean_pi_opening"

    # ------------------------------------------------------------------
    # Legacy Qt slot names.  Pass-through only.
    # ------------------------------------------------------------------

    def _on_bore_page_requested(self) -> None:
        self._bore_tool().on_page_requested()

    def _on_page_shown(self, key: str) -> None:
        self._bore_tool().on_page_shown(key)

    def _on_bore_select_opening_clicked(self) -> None:
        self._bore_tool().on_select_opening_clicked()

    def _on_bore_enable_edge_selection_clicked(self) -> None:
        self._bore_tool().on_select_opening_clicked()

    def _on_bore_enable_edge_brush_clicked(self) -> None:
        self._bore_tool().on_select_opening_clicked()

    def _on_bore_boundary_highlight_toggled(self, enabled: bool) -> None:
        self._bore_tool().on_boundary_highlight_toggled(enabled)

    def _on_bore_focus_selection_clicked(self) -> None:
        self._bore_tool().on_focus_selection_clicked()

    def _on_bore_clear_selection_clicked(self) -> None:
        self._bore_tool().on_clear_selection_clicked()

    def _bore_clear_selection_after_rebuild(self) -> None:
        self._bore_tool().clear_after_rebuild()

    def _on_bore_select_inside_boundary_clicked(self) -> None:
        self._bore_tool().on_preview_inside_boundary_clicked()

    def _on_bore_select_wall_faces_clicked(self) -> None:
        self._bore_tool().on_list_candidates_clicked()

    def _on_bore_list_candidates_clicked(self) -> None:
        self._bore_tool().on_list_candidates_clicked()

    def _on_bore_feature_candidate_changed(self, index: int) -> None:
        self._bore_tool().on_candidate_changed(index)

    def _on_bore_feature_candidate_tree_changed(self, current: object, _previous: object | None = None) -> None:
        candidate_id = ""
        if current is not None and Qt is not None:
            try:
                candidate_id = str(current.data(0, Qt.ItemDataRole.UserRole) or "")
            except Exception:
                candidate_id = ""
        if not candidate_id:
            tree = getattr(self, "bore_feature_candidate_tree", None)
            try:
                index = int(tree.indexOfTopLevelItem(current))
            except Exception:
                index = -1
            self._bore_tool().on_candidate_changed(index)
            return
        self._bore_tool().on_candidate_changed(candidate_id)

    def _on_bore_preview_feature_candidate_clicked(self) -> None:
        self._bore_tool().preview_current_candidate()

    def _on_bore_view_feature_candidate_clicked(self) -> None:
        self._bore_tool().preview_current_candidate()

    def _on_bore_reset_feature_candidate_preview_clicked(self) -> None:
        self._bore_tool().reset_preview()

    def _on_bore_rebuild_wall_faces_clicked(self) -> None:
        self._bore_tool().rebuild_previewed_candidate()


__all__ = ["BoreActionsMixin"]
