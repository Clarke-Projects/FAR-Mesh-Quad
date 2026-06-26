from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QFileDialog, QMessageBox

from .constants import CAD_PRESETS
from .project_actions import _update_project_status_ui_if_available
from .payload_formatters import (
    format_reduce_payload_lines,
    format_remesh_payload_lines,
    format_repair_inspection_lines,
    format_repair_payload_lines,
)


class MeshActionsMixin:
    """Classic mesh workstation actions for MainWindow.

    This mixin owns GUI-side load/export/repair/remesh/reduce orchestration and
    mesh information/log formatting helpers. Mesh truth remains owned by
    MeshProcessor; this layer only builds GUI requests, delegates work, and
    refreshes UI state.  After mesh replacement it also clears stale selection
    evidence so old edge/face IDs cannot be reused by feature tools.
    """

    def _populate_runtime_options(self) -> None:
            self._populate_repair_methods()
            self._populate_reduce_backends()
            self._populate_stage1_presets()
            self._populate_quadwild_cleanup_methods()
            self._on_remesh_backend_changed()
            if hasattr(self, "manual_edit_operation_combo"):
                self._on_manual_edit_operation_changed()

    def _populate_repair_methods(self) -> None:
            self.repair_method_combo.clear()
            if hasattr(self.processor.repairer, "available_options"):
                methods = list(self.processor.repairer.available_options())
            else:
                methods = list(self.processor.repairer.available_methods())
            if "cad_workflow" not in methods:
                methods.append("cad_workflow")
            preferred = [
                "cad_safe",
                "cad_preserve_features",
                "light_normalize",
                "topology_cleanup",
                "scan_closing",
                "hybrid",
                "cad_safe_pymeshlab",
                "pymeshlab",
                "pymeshfix",
                "open3d",
                "trimesh",
                "cad_workflow",
            ]
            ordered = [m for m in preferred if m in methods] + [m for m in methods if m not in preferred]
            for method in ordered:
                self.repair_method_combo.addItem(method, method)
            idx = self.repair_method_combo.findData("cad_safe")
            if idx < 0:
                idx = self.repair_method_combo.findData("trimesh")
            if idx >= 0:
                self.repair_method_combo.setCurrentIndex(idx)

    def _populate_reduce_backends(self) -> None:
            self.reduce_backend_combo.clear()
            available = self.processor.reducer.available_backends()
            for key, label in available.items():
                self.reduce_backend_combo.addItem(label, key)
            if self.reduce_backend_combo.count() == 0:
                self.reduce_backend_combo.addItem("Open3D unavailable", "unavailable")
                self.reduce_btn_run.setEnabled(False)

    def _populate_stage1_presets(self) -> None:
            self.quadwild_stage1_combo.clear()
            presets: dict[str, str] = {}
            runner = getattr(self.processor, "quadwild_bimdf_runner", None)
            if runner is not None:
                try:
                    presets = runner.get_available_stage1_presets()
                except Exception:
                    presets = {}
            for label, rel in presets.items():
                self.quadwild_stage1_combo.addItem(label, rel)
            if self.quadwild_stage1_combo.count() == 0:
                self.quadwild_stage1_combo.addItem("basic_setup.txt", "config/prep_config/basic_setup.txt")

    def _populate_quadwild_cleanup_methods(self) -> None:
            self.quadwild_cleanup_combo.clear()
            if hasattr(self.processor.repairer, "available_options"):
                methods = list(self.processor.repairer.available_options())
            else:
                methods = list(self.processor.repairer.available_methods())
            preferred = [
                "cad_safe",
                "cad_preserve_features",
                "cad_safe_pymeshlab",
                "pymeshfix",
                "open3d",
                "trimesh",
                "hybrid",
                "pymeshlab",
            ]
            ordered = [m for m in preferred if m in methods] + [m for m in methods if m not in preferred]
            for method in ordered:
                self.quadwild_cleanup_combo.addItem(method, method)
            idx = self.quadwild_cleanup_combo.findData("cad_safe")
            if idx < 0:
                idx = self.quadwild_cleanup_combo.findData("cad_safe_pymeshlab")
            if idx >= 0:
                self.quadwild_cleanup_combo.setCurrentIndex(idx)

    def _on_remesh_backend_changed(self) -> None:
            backend = self.remesh_backend_combo.currentData()
            self.instant_group.setVisible(backend == "instant_meshes")
            self.quad_group.setVisible(backend == "quadwild_bimdf")

    def _apply_cad_preset(self) -> None:
            preset_name = self.cad_preset_combo.currentData()
            preset = CAD_PRESETS.get(preset_name)
            if not preset:
                return
            self.quadwild_sharp_spin.setValue(float(preset["quadwild_sharp"]))
            self.quadwild_alpha_spin.setValue(float(preset["quadwild_alpha"]))
            self.quadwild_scale_spin.setValue(float(preset["quadwild_scale"]))
            self.quadwild_auto_reduce_check.setChecked(bool(preset["auto_reduce"]))
            self.quadwild_auto_reduce_target_spin.setValue(int(preset["auto_reduce_target"]))
            self.quadwild_auto_reduce_boundary_weight_spin.setValue(
                float(preset.get("quadwild_auto_reduce_boundary_weight", preset.get("auto_reduce_boundary_weight", 5.0)))
            )

    def _set_mesh_info_empty(self) -> None:
            self.vertices_label.setText("Vertices: -")
            self.faces_label.setText("Faces: -")
            self.bounds_label.setText("Bounds: -")
            self.watertight_label.setText("Watertight: -")

    def _set_mesh_info_path_only(self, path: str) -> None:
            self.vertices_label.setText("Vertices: -")
            self.faces_label.setText("Faces: -")
            self.bounds_label.setText(f"Bounds: from {Path(path).name}")
            self.watertight_label.setText("Watertight: -")

    def _set_mesh_info_from_trimesh(self, mesh: Any) -> None:
            vertices = getattr(mesh, "vertices", None)
            faces = getattr(mesh, "faces", None)
            bounds = getattr(mesh, "bounds", None)
            watertight = getattr(mesh, "is_watertight", None)

            self.vertices_label.setText(f"Vertices: {len(vertices):,}" if vertices is not None else "Vertices: -")
            self.faces_label.setText(f"Faces: {len(faces):,}" if faces is not None else "Faces: -")

            if bounds is not None:
                try:
                    b = [float(v) for row in bounds for v in row]
                    bounds_text = (
                        f"x[{b[0]:.3f}, {b[3]:.3f}] "
                        f"y[{b[1]:.3f}, {b[4]:.3f}] "
                        f"z[{b[2]:.3f}, {b[5]:.3f}]"
                    )
                except Exception:
                    bounds_text = str(bounds)
                self.bounds_label.setText(f"Bounds: {bounds_text}")
            else:
                self.bounds_label.setText("Bounds: -")

            if watertight is None:
                self.watertight_label.setText("Watertight: -")
            else:
                self.watertight_label.setText(f"Watertight: {'Yes' if bool(watertight) else 'No'}")

    def _log_repair_inspection(self, label: str, inspection: dict[str, Any] | None) -> None:
            for line in format_repair_inspection_lines(label, inspection):
                self.log(line)

    def _log_repair_payload(self, payload: dict[str, Any]) -> None:
            for line in format_repair_payload_lines(payload):
                self.log(line)

    def _log_reduce_payload(self, payload: dict[str, Any]) -> None:
            for line in format_reduce_payload_lines(payload):
                self.log(line)

    def _log_remesh_payload(self, payload: dict[str, Any]) -> None:
            for line in format_remesh_payload_lines(payload):
                self.log(line)

    def _refresh_mesh_navigation_state_after_mesh_change(self, *, reason: str) -> None:
            """Clear stale selection/navigation caches after mesh source changes.

            A no-op Reduce was useful during coarse Bore tests because it forced
            the processor/viewport path to refresh even when face count did not
            change.  Make that cache cleanup explicit after load/replacement so
            edge IDs, face adjacency, clicked-seed metadata, and viewport
            selection overlays do not survive across mesh states.
            """

            controller = getattr(self, "selection_controller", None)
            if controller is None:
                controller = getattr(self, "selection", None)
            cleared_controller = False
            if controller is not None and hasattr(controller, "clear_selection"):
                try:
                    controller.clear_selection(keep_mode=True, push=True, reason=reason)
                    cleared_controller = True
                except Exception:
                    cleared_controller = False

            viewport = getattr(self, "viewport", None)
            if viewport is None:
                viewport = getattr(self, "viewer", None)
            viewport_clears = 0
            if viewport is not None:
                for name, args in (
                    ("clear_selection", ()),
                    ("set_edge_selection", ((),)),
                    ("set_selected_edge_ids", ((),)),
                    ("clear_edge_selection", ()),
                    ("set_face_selection", ((),)),
                    ("highlight_cells", ((),)),
                    ("clear_face_selection", ()),
                    ("clear_polyline", ("bore_resolved_opening",)),
                ):
                    fn = getattr(viewport, name, None)
                    if callable(fn):
                        try:
                            fn(*args)
                            viewport_clears += 1
                        except Exception:
                            pass
                for name in ("rebuild_topology_cache", "invalidate_pick_cache", "refresh_edge_index_cache"):
                    fn = getattr(viewport, name, None)
                    if callable(fn):
                        try:
                            fn()
                            viewport_clears += 1
                        except Exception:
                            pass

            if hasattr(self, "log"):
                self.log(
                    "Mesh navigation state refreshed after mesh change: "
                    f"reason={reason}; controller={'yes' if cleared_controller else 'no'}; "
                    f"viewport_ops={viewport_clears}."
                )

    def load_mesh(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open mesh",
                str(Path.cwd()),
                "Mesh Files (*.obj *.ply *.stl *.off *.glb *.gltf *.vtp *.vtk *.vtu *.3mf *.fbx *.dae);;All Files (*)",
            )
            if not path:
                return

            def task() -> object:
                mesh = self.processor.load_mesh(path)
                return {"mesh": mesh, "source_path": path}

            def on_success(payload: object) -> None:
                assert isinstance(payload, dict)
                self.current_mesh_path = str(Path(payload["source_path"]).expanduser().resolve())
                self.current_output_path = None
                self._load_new_source_into_viewport(self.current_mesh_path)
                self._refresh_mesh_navigation_state_after_mesh_change(reason="mesh_load_source_replacement_v142")
                self._clear_manual_edit_preview(silent=True)
                self._reset_hole_fill_ui(status="No hole candidates yet. Run Find Hole Candidates.")
                mesh = getattr(self.processor, "mesh", None)
                if mesh is not None:
                    self._set_mesh_info_from_trimesh(mesh)
                else:
                    self._set_mesh_info_path_only(self.current_mesh_path)
                self.current_file_label.setText(Path(self.current_mesh_path).name)
                self._show_page(self.PAGE_VIEWER)
                self._sync_viewport_ui_from_backend()
                self._set_topology_result_text(
                    "Mesh loaded. Use Analyze Topology or Find Hole Candidates from Brush / Tools.\n"
                    "Face selections are used automatically when present."
                )
                _update_project_status_ui_if_available(self)
                self.log("Mesh loaded successfully.")

            self._run_task("Loading mesh...", task, on_success)

    def save_mesh(self) -> None:
            mesh = getattr(self.processor, "mesh", None)
            if mesh is None:
                QMessageBox.information(self, "No mesh loaded", "Load or generate a mesh before saving.")
                return
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save mesh as",
                str(Path.cwd() / "mesh_output.obj"),
                "OBJ (*.obj);;PLY (*.ply);;STL (*.stl);;All Files (*)",
            )
            if not path:
                return
            try:
                self.processor.save_mesh(path)
            except Exception as exc:
                QMessageBox.critical(self, "Save failed", str(exc))
                self.log(f"Save failed: {exc}")
                return
            self.current_output_path = str(Path(path).expanduser().resolve())
            self.log(f"Mesh saved: {self.current_output_path}")

    def _open3d_tensor_fill_holes_policy_limits_from_gui(self) -> dict[str, int]:
        def _spin_value(name: str, default: int) -> int:
            widget = getattr(self, name, None)
            if widget is None:
                return int(default)
            try:
                return int(widget.value())
            except Exception:
                return int(default)

        return {
            "max_faces_added": _spin_value("repair_o3d_max_faces_spin", 100),
            "max_vertices_added": _spin_value("repair_o3d_max_vertices_spin", 0),
            "max_candidate_delta": _spin_value("repair_o3d_max_candidate_delta_spin", 1),
        }

    def _log_open3d_tensor_fill_holes_policy_evaluation(self, evaluation: dict[str, Any]) -> None:
        allowed = bool(evaluation.get("allowed"))
        self.log(f"  policy: {'allowed' if allowed else 'blocked'}")

        limits = evaluation.get("limits") or {}
        if isinstance(limits, dict):
            self.log(
                "  policy limits: "
                f"max_faces_added={limits.get('max_faces_added')}, "
                f"max_vertices_added={limits.get('max_vertices_added')}, "
                f"max_candidate_delta={limits.get('max_candidate_delta')}"
            )

        reasons = evaluation.get("reasons") or []
        for reason in reasons:
            self.log(f"  policy reason: {reason}")

        warnings = evaluation.get("warnings") or []
        for warning in warnings:
            self.log(f"  policy warning: {warning}")

    @staticmethod
    def _open3d_dry_run_candidate_kind_counts(payload: dict[str, Any], key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        candidates = payload.get(key) or []
        if not isinstance(candidates, list):
            return counts

        for item in candidates:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "unknown")
            counts[kind] = counts.get(kind, 0) + 1

        return counts

    def _log_open3d_tensor_fill_holes_dry_run_payload(self, payload: dict[str, Any]) -> None:
        self.log("Open3D tensor fill_holes dry run:")
        self.log(f"  dry_run: {payload.get('dry_run')}")
        self.log(f"  hole_size: {payload.get('hole_size')}")
        self.log(f"  elapsed: {float(payload.get('elapsed_seconds', 0.0) or 0.0):.2f}s")
        self.log(
            "  candidates: "
            f"{payload.get('candidate_count_before')} -> {payload.get('candidate_count_after')} "
            f"(delta {payload.get('filled_candidate_delta')})"
        )
        self.log(
            "  geometry delta: "
            f"faces +{payload.get('added_faces')}, "
            f"vertices +{payload.get('added_vertices')}"
        )

        before_stats = payload.get("stats_before") or {}
        after_stats = payload.get("stats_after") or {}
        if before_stats or after_stats:
            self.log(
                "  stats: "
                f"vertices {before_stats.get('vertices')} -> {after_stats.get('vertices')}, "
                f"faces {before_stats.get('faces')} -> {after_stats.get('faces')}, "
                f"watertight {before_stats.get('watertight')} -> {after_stats.get('watertight')}"
            )

        self._log_repair_inspection(
            "Open3D dry-run inspection before",
            payload.get("inspection_before"),
        )
        self._log_repair_inspection(
            "Open3D dry-run inspection after",
            payload.get("inspection_after"),
        )

        before_counts = self._open3d_dry_run_candidate_kind_counts(
            payload,
            "candidate_diagnostics_before",
        )
        after_counts = self._open3d_dry_run_candidate_kind_counts(
            payload,
            "candidate_diagnostics_after",
        )

        if before_counts:
            parts = [f"{kind}={count}" for kind, count in sorted(before_counts.items())]
            self.log(f"  candidate kinds before: {', '.join(parts)}")
        if after_counts:
            parts = [f"{kind}={count}" for kind, count in sorted(after_counts.items())]
            self.log(f"  candidate kinds after: {', '.join(parts)}")

        notes = payload.get("notes") or []
        for note in notes:
            self.log(f"  note: {note}")

    def _log_open3d_tensor_fill_holes_guarded_repair_blocked(self, payload: dict[str, Any]) -> None:
        self.log("Open3D tensor fill_holes repair blocked:")
        message = str(payload.get("error") or "policy blocked repair")
        prefix = "Open3D tensor fill_holes repair blocked:"
        reason = message[len(prefix):].strip() if message.startswith(prefix) else message
        if reason:
            self.log(f"  reason: {reason}")

        limits = payload.get("limits") or {}
        if isinstance(limits, dict) and limits:
            self.log(
                "  policy limits: "
                f"max_faces_added={limits.get('max_faces_added')}, "
                f"max_vertices_added={limits.get('max_vertices_added')}, "
                f"max_candidate_delta={limits.get('max_candidate_delta')}"
            )

        self.log("  no mesh changes were committed.")

    def _log_open3d_tensor_fill_holes_guarded_repair_payload(self, payload: dict[str, Any]) -> None:
        self.log("Open3D tensor fill_holes repair committed:")
        self.log(f"  operation: {payload.get('operation')}")
        self.log(f"  method: {payload.get('method')}")
        self.log(f"  hole_size: {payload.get('hole_size')}")

        before_stats = payload.get("stats_before") or {}
        after_stats = payload.get("stats_after") or {}
        if before_stats or after_stats:
            self.log(
                "  stats: "
                f"vertices {before_stats.get('vertices')} -> {after_stats.get('vertices')}, "
                f"faces {before_stats.get('faces')} -> {after_stats.get('faces')}, "
                f"watertight {before_stats.get('watertight')} -> {after_stats.get('watertight')}"
            )

        dry_run_report = payload.get("dry_run_report") or {}
        if isinstance(dry_run_report, dict):
            self.log(
                "  dry-run candidates: "
                f"{dry_run_report.get('candidate_count_before')} -> "
                f"{dry_run_report.get('candidate_count_after')} "
                f"(delta {dry_run_report.get('filled_candidate_delta')})"
            )
            self.log(
                "  geometry delta: "
                f"faces +{dry_run_report.get('added_faces')}, "
                f"vertices +{dry_run_report.get('added_vertices')}"
            )

        policy = payload.get("policy_evaluation") or {}
        if isinstance(policy, dict):
            self._log_open3d_tensor_fill_holes_policy_evaluation(policy)

        history_dir = payload.get("history_dir")
        if history_dir:
            self.log(f"  history: {history_dir}")

        notes = payload.get("notes") or []
        for note in notes:
            self.log(f"  note: {note}")

    def run_open3d_fill_holes_guarded_repair(self) -> None:
        mesh = getattr(self.processor, "mesh", None)
        if mesh is None:
            QMessageBox.information(self, "No mesh loaded", "Load a mesh first.")
            return

        hole_size = 1_000_000.0
        if hasattr(self, "repair_o3d_hole_size_spin"):
            try:
                hole_size = float(self.repair_o3d_hole_size_spin.value())
            except Exception:
                hole_size = 1_000_000.0

        limits = self._open3d_tensor_fill_holes_policy_limits_from_gui()

        def task() -> object:
            try:
                return self.processor.repair_open3d_tensor_fill_holes_guarded(
                    hole_size=hole_size,
                    max_faces_added=limits["max_faces_added"],
                    max_vertices_added=limits["max_vertices_added"],
                    max_candidate_delta=limits["max_candidate_delta"],
                )
            except Exception as exc:
                # Policy-blocked Open3D repairs may be raised by the core as
                # ValueError or as a MeshProcessor-level exception. Keep the
                # clean GUI blocked-repair path for that known condition, but
                # re-raise every unrelated exception so real failures still go
                # through the normal task failure / traceback path.
                message = str(exc)
                if "Open3D tensor fill_holes repair blocked:" not in message:
                    raise
                return {
                    "operation": "open3d_tensor_fill_holes_repair",
                    "blocked": True,
                    "error": message,
                    "hole_size": hole_size,
                    "limits": dict(limits),
                }

        def on_success(result: object) -> None:
            assert isinstance(result, dict)

            if result.get("blocked"):
                self._log_open3d_tensor_fill_holes_guarded_repair_blocked(result)
                if hasattr(self, "statusBar"):
                    self.statusBar().showMessage("Open3D fill_holes repair blocked", 4000)
                return

            self.current_output_path = None
            self._log_open3d_tensor_fill_holes_guarded_repair_payload(result)

            refresh = getattr(self, "_refresh_viewport_from_processor", None)
            if callable(refresh):
                refresh()
            else:
                mesh_after = getattr(self.processor, "mesh", None)
                if mesh_after is not None and hasattr(self, "_set_mesh_info_from_trimesh"):
                    self._set_mesh_info_from_trimesh(mesh_after)

            reset_holes = getattr(self, "_reset_hole_fill_ui", None)
            if callable(reset_holes):
                reset_holes(
                    status=(
                        "Mesh repaired with Open3D tensor fill_holes. "
                        "Run Analyze Topology / Find Hole Candidates again."
                    )
                )

            try:
                _update_project_status_ui_if_available(self)
            except Exception as exc:
                if hasattr(self, "log"):
                    self.log(f"Project status update skipped after Open3D repair: {exc}")

            if hasattr(self, "statusBar"):
                self.statusBar().showMessage("Open3D fill_holes repair committed", 3000)

        self._run_task("Applying Open3D fill_holes repair...", task, on_success)

    def run_open3d_fill_holes_dry_run(self) -> None:
        mesh = getattr(self.processor, "mesh", None)
        if mesh is None:
            QMessageBox.information(self, "No mesh loaded", "Load a mesh first.")
            return

        hole_size = 1_000_000.0
        if hasattr(self, "repair_o3d_hole_size_spin"):
            try:
                hole_size = float(self.repair_o3d_hole_size_spin.value())
            except Exception:
                hole_size = 1_000_000.0

        def task() -> object:
            return self.processor.repairer.inspect_open3d_tensor_fill_holes(
                mesh,
                hole_size=hole_size,
            )

        def on_success(result: object) -> None:
            assert isinstance(result, dict)
            self._log_open3d_tensor_fill_holes_dry_run_payload(result)

            evaluator = getattr(
                self.processor.repairer,
                "evaluate_open3d_tensor_fill_holes_policy",
                None,
            )
            limit_getter = getattr(
                self,
                "_open3d_tensor_fill_holes_policy_limits_from_gui",
                None,
            )
            policy_logger = getattr(
                self,
                "_log_open3d_tensor_fill_holes_policy_evaluation",
                None,
            )

            if callable(evaluator) and callable(limit_getter) and callable(policy_logger):
                evaluation = evaluator(result, **limit_getter())
                policy_logger(evaluation)

            if hasattr(self, "statusBar"):
                self.statusBar().showMessage("Open3D fill_holes dry run complete", 3000)

        self._run_task("Running Open3D fill_holes dry run...", task, on_success)

    def run_repair(self) -> None:
            if getattr(self.processor, "mesh", None) is None and self.current_mesh_path is None:
                QMessageBox.information(self, "No mesh loaded", "Load a mesh first.")
                return
            method = self.repair_method_combo.currentData()
            join_comp = self.repair_join_comp_check.isChecked()
            fill_holes = self.repair_fill_holes_check.isChecked()
            advanced_options: dict[str, Any] = {}
            if hasattr(self, "repair_advanced_group") and self.repair_advanced_group.isEnabled() and self.repair_advanced_group.isChecked():
                if self.repair_preserve_features_check.isChecked() and method == "cad_safe":
                    method = "cad_preserve_features"
                advanced_options["edge_method"] = (
                    "Remove Faces" if self.repair_edge_method_combo.currentData() == "remove_faces"
                    else "Split Vertices"
                )
                advanced_options["vertex_drift"] = self.repair_vertex_drift_spin.value()
                advanced_options["t_vertex_enabled"] = self.repair_tvertex_enable_check.isChecked()
                if advanced_options["t_vertex_enabled"]:
                    tvm = self.repair_tvertex_method_combo.currentData()
                    advanced_options["t_vertex_method"] = "Edge Flip" if tvm == "edge_flip" else "Edge Collapse"
                    advanced_options["t_vertex_threshold"] = self.repair_tvertex_threshold_spin.value()
                    advanced_options["t_vertex_repeat"] = self.repair_tvertex_repeat_check.isChecked()

            def task() -> object:
                return self.processor.repair(
                    method=method,
                    join_comp=join_comp,
                    fill_holes=fill_holes,
                    repair_options=advanced_options,
                )

            def on_success(result: object) -> None:
                assert isinstance(result, dict)
                self._refresh_viewport_from_processor()
                self._set_mesh_info_from_trimesh(self.processor.mesh)
                self._log_repair_payload(result)
                self._show_page(self.PAGE_VIEWER)
                _update_project_status_ui_if_available(self)
                self._sync_viewport_ui_from_backend()
                self.log("Mesh repair completed.")

            self._run_task("Running repair...", task, on_success)

    def run_remesh(self) -> None:
            if getattr(self.processor, "mesh", None) is None and self.current_mesh_path is None:
                QMessageBox.information(self, "No mesh loaded", "Load a mesh first.")
                return
            backend = self.remesh_backend_combo.currentData()
            use_source_mode = self.remesh_source_combo.currentData() == "source"

            def task() -> object:
                if use_source_mode and self.current_mesh_path:
                    self.processor.load_mesh(self.current_mesh_path)
                if backend == "instant_meshes":
                    return self.processor.remesh(
                        backend="instant_meshes",
                        target_faces=self.remesh_target_faces_spin.value(),
                        crease_angle=self.remesh_crease_angle_spin.value(),
                        smooth_iterations=self.remesh_smooth_iters_spin.value(),
                        deterministic=self.remesh_deterministic_check.isChecked(),
                    )
                stage2_rel = self.quadwild_stage2_combo.currentData()
                if not stage2_rel:
                    stage2_rel = self.quadwild_stage2_combo.currentText().strip()
                return self.processor.remesh(
                    backend="quadwild_bimdf",
                    quadwild_stage1_config_rel=self.quadwild_stage1_combo.currentData(),
                    quadwild_stage2_config_rel=stage2_rel,
                    quadwild_do_remesh=self.quadwild_do_remesh_check.isChecked(),
                    quadwild_sharp_feature_threshold=self.quadwild_sharp_spin.value(),
                    quadwild_alpha=self.quadwild_alpha_spin.value(),
                    quadwild_scale_factor=self.quadwild_scale_spin.value(),
                    quadwild_use_original_input_file=self.quadwild_use_original_check.isChecked(),
                    quadwild_pre_repair_workflow=self.quadwild_workflow_check.isChecked(),
                    quadwild_cleanup_method=self.quadwild_cleanup_combo.currentData(),
                    quadwild_fill_holes=self.quadwild_fill_holes_check.isChecked(),
                    auto_reduce_after_quadwild=self.quadwild_auto_reduce_check.isChecked(),
                    auto_reduce_backend="open3d",
                    auto_reduce_target_faces=self.quadwild_auto_reduce_target_spin.value(),
                    auto_reduce_boundary_weight=self.quadwild_auto_reduce_boundary_weight_spin.value(),
                    auto_reduce_cleanup=self.quadwild_auto_reduce_cleanup_check.isChecked(),
                    post_decimate=self.quadwild_post_decimate_check.isChecked(),
                    decimate_target_faces=self.quadwild_decimate_target_spin.value(),
                )

            def on_success(result: object) -> None:
                assert isinstance(result, dict)
                preferred = self._extract_path_from_result(result)
                self._refresh_viewport_from_processor(preferred_source_path=preferred)
                self._set_mesh_info_from_trimesh(self.processor.mesh)
                self._log_remesh_payload(result)
                self._show_page(self.PAGE_VIEWER)
                _update_project_status_ui_if_available(self)
                self._sync_viewport_ui_from_backend()
                self.log("Remeshing completed.")

            self._run_task("Running remesh...", task, on_success)

    def run_reduce(self) -> None:
            if getattr(self.processor, "mesh", None) is None and self.current_mesh_path is None:
                QMessageBox.information(self, "No mesh loaded", "Load a mesh first.")
                return
            backend = self.reduce_backend_combo.currentData()
            if backend in (None, "unavailable"):
                QMessageBox.warning(self, "Reducer unavailable", "No active reduction backend is available.")
                return

            def task() -> object:
                return self.processor.reduce(
                    backend=backend,
                    target_faces=self.reduce_target_faces_spin.value(),
                    boundary_weight=self.reduce_boundary_weight_spin.value(),
                    cleanup=self.reduce_cleanup_check.isChecked(),
                )

            def on_success(result: object) -> None:
                assert isinstance(result, dict)
                self._refresh_viewport_from_processor()
                self._refresh_mesh_navigation_state_after_mesh_change(reason="mesh_reduce_refresh_v142")
                self._set_mesh_info_from_trimesh(self.processor.mesh)
                self._log_reduce_payload(result)
                self._show_page(self.PAGE_VIEWER)
                _update_project_status_ui_if_available(self)
                self._sync_viewport_ui_from_backend()
                self.log("Reduction completed.")

            self._run_task("Running reduction...", task, on_success)

    def _extract_path_from_result(self, result: object) -> str | None:
            if result is None:
                return None
            if isinstance(result, (str, Path)):
                return str(Path(result).expanduser().resolve())
            if isinstance(result, dict):
                for key in ("output_path", "file_path", "path", "source_path", "mesh_path"):
                    value = result.get(key)
                    if isinstance(value, (str, Path)):
                        return str(Path(value).expanduser().resolve())
            for key in ("output_path", "file_path", "path", "source_path", "mesh_path"):
                value = getattr(result, key, None)
                if isinstance(value, (str, Path)):
                    return str(Path(value).expanduser().resolve())
            return None
