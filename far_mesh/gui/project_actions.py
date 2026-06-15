from __future__ import annotations

import json
import shutil
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox

from far_mesh.core.project_storage import PROJECT_FILE_NAME, SESSION_STATE_FILE_NAME, ProjectStorage


def _update_project_status_ui_if_available(window: object) -> None:
    """Refresh project status widgets when the current object provides them.

    Some focused GUI-adapter tests call MainWindow methods on light fake
    window objects. Those fakes intentionally do not build the full UI, so
    the I3 project-status refresh hook must be optional in adapter paths.
    """
    updater = getattr(window, "_update_project_status_ui", None)
    if callable(updater):
        updater()


class ProjectActionsMixin:
    """Project open/save/status GUI-controller helpers.

    This mixin keeps project persistence UI behavior out of the main
    application shell while preserving MainWindow as the GUI orchestrator.
    MeshProcessor and ProjectStorage remain the authorities for mesh/project
    state; this layer only coordinates dialogs, status labels, and logs.
    """

    @staticmethod
    def _normalize_project_root_path(path: str | Path) -> Path:
        root = Path(path).expanduser()
        if root.name.endswith(".farmesh3"):
            return root
        return Path(str(root) + ".farmesh3")

    def _current_project_storage_root(self) -> Path | None:
        root = self._safe_call(lambda: self.processor.project_storage_root(), None)
        if root is not None:
            return Path(root).expanduser().resolve()

        storage = self._safe_call(lambda: self.processor.current_project_storage(), None)
        storage_root = getattr(storage, "root", None)
        if storage_root is not None:
            return Path(storage_root).expanduser().resolve()
        return None

    def _sync_project_state_metadata_for_gui(self, reason: str = "gui_save_project") -> object:
        """Sync project/session metadata through MeshProcessor.

        Different development checkpoints used either ``sync_reason`` or
        ``reason`` as the keyword name, so keep the adapter tolerant.
        """
        sync = getattr(self.processor, "sync_project_state_metadata")
        reason_text = str(reason or "gui_save_project")
        for kwargs in (
            {"sync_reason": reason_text},
            {"reason": reason_text},
            {},
        ):
            try:
                return sync(**kwargs)
            except TypeError:
                continue
        return sync()

    @staticmethod
    def _read_storage_metadata_for_save_as(storage: object) -> dict[str, object]:
        """Read metadata from the current storage without failing Save As.

        Save Project As copies the active storage folder.  For unsaved sessions
        the authoritative state is ``project_state.json``; for saved projects it
        is ``project.json``.  Reading it before the copy gives us the correct
        undo/redo stack to migrate into the saved target project.
        """
        reader = getattr(storage, "read_metadata", None)
        if callable(reader):
            for kwargs in ({"required": False}, {}):
                try:
                    data = reader(**kwargs)
                except TypeError:
                    continue
                except Exception:
                    return {}
                return data if isinstance(data, dict) else {}
        return {}

    @staticmethod
    def _read_json_metadata_file_for_save_as(path: Path) -> dict[str, object]:
        try:
            if not path.exists() or not path.is_file():
                return {}
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _metadata_restore_score_for_save_as(metadata: object, *, prefer_project_json: bool = False) -> tuple[int, int, int, int]:
        """Return a score for choosing restore metadata during Save As.

        The important signal is whether the metadata indexes undo/redo history.
        A copied unsaved session may have a valid ``project_state.json`` while
        the saved-project ``project.json`` is missing or empty.
        """
        if not isinstance(metadata, dict):
            return (0, 0, 0, 1 if prefer_project_json else 0)
        extra = metadata.get("extra")
        if not isinstance(extra, dict):
            extra = {}

        undo_stack = extra.get("undo_stack")
        redo_stack = extra.get("redo_stack")
        undo_count = len(undo_stack) if isinstance(undo_stack, list) else 0
        redo_count = len(redo_stack) if isinstance(redo_stack, list) else 0
        latest = 1 if extra.get("latest_history_entry") else 0
        current = 1 if extra.get("current_mesh_snapshot") else 0
        availability = int(bool(extra.get("can_undo"))) + int(bool(extra.get("can_redo")))

        # History is weighted above current mesh because losing undo/redo while
        # retaining the current snapshot was the observed Save As bug.
        history_score = (undo_count + redo_count) * 10 + latest
        return (history_score, current, availability, 1 if prefer_project_json else 0)

    @classmethod
    def _select_save_as_restore_metadata(
        cls,
        *,
        target_root: Path,
        source_metadata: dict[str, object] | None,
    ) -> dict[str, object] | None:
        """Choose the metadata used immediately after Save Project As copy.

        Prefer the live source metadata captured just before copying.  Fall back
        to the copied target files and choose whichever contains the richest
        current/history state.  This migrates unsaved-session ``project_state``
        into saved-project ``project.json`` without weakening ProjectStorage's
        filename contract.
        """
        candidates: list[tuple[tuple[int, int, int, int], dict[str, object]]] = []

        if isinstance(source_metadata, dict) and source_metadata:
            candidates.append((cls._metadata_restore_score_for_save_as(source_metadata), source_metadata))

        project_metadata = cls._read_json_metadata_file_for_save_as(target_root / PROJECT_FILE_NAME)
        if project_metadata:
            candidates.append((cls._metadata_restore_score_for_save_as(project_metadata, prefer_project_json=True), project_metadata))

        session_metadata = cls._read_json_metadata_file_for_save_as(target_root / SESSION_STATE_FILE_NAME)
        if session_metadata:
            candidates.append((cls._metadata_restore_score_for_save_as(session_metadata), session_metadata))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    @staticmethod
    def _restore_result_lines(result: object) -> list[str]:
        if result is None:
            return ["Restore completed."]

        if isinstance(result, dict):
            lines: list[str] = []
            for key in (
                "current_mesh_restored",
                "restored_current_mesh",
                "restored_undo_entries",
                "restored_redo_entries",
                "skipped_current_mesh",
                "skipped_current_mesh_snapshot",
                "current_mesh_restore_error",
                "skipped_history_entries",
                "skipped_references",
                "metadata_schema_supported",
                "metadata_schema_error",
                "project_open_failed",
                "error",
            ):
                if key in result:
                    lines.append(f"{key}: {result.get(key)}")
            return lines or [str(result)]

        summary = getattr(result, "summary", None)
        if callable(summary):
            try:
                return [str(summary())]
            except Exception:
                pass

        parts: list[str] = []
        for key in (
            "current_mesh_restored",
            "restored_current_mesh",
            "restored_undo_entries",
            "restored_redo_entries",
            "skipped_references",
            "skipped_current_mesh_snapshot",
            "current_mesh_restore_error",
            "skipped_history_entries",
            "metadata_schema_supported",
            "metadata_schema_error",
            "project_open_failed",
            "error",
        ):
            if hasattr(result, key):
                parts.append(f"{key}: {getattr(result, key)}")
        return parts or [str(result)]

    @staticmethod
    def _format_bytes_for_gui(value: object) -> str:
        try:
            size = int(value)
        except Exception:
            return "-"

        if size < 0:
            return "-"

        units = ("B", "KB", "MB", "GB", "TB")
        amount = float(size)
        unit = units[0]
        for unit in units:
            if amount < 1024.0 or unit == units[-1]:
                break
            amount /= 1024.0

        if unit == "B":
            return f"{int(amount)} B"
        return f"{amount:.1f} {unit}"

    @staticmethod
    def _safe_dict_get(mapping: object, key: str, default: object = None) -> object:
        if isinstance(mapping, dict):
            return mapping.get(key, default)
        return getattr(mapping, key, default)

    def _project_metadata_extra_for_gui(self) -> dict[str, object]:
        storage = self._safe_call(lambda: self.processor.current_project_storage(), None)
        if storage is None:
            return {}

        metadata = self._safe_call(lambda: storage.read_metadata(), {})
        if not isinstance(metadata, dict):
            return {}

        extra = metadata.get("extra")
        return extra if isinstance(extra, dict) else {}

    def _project_disk_usage_text_for_gui(self) -> str:
        usage = self._safe_call(lambda: self.processor.project_disk_usage(), None)
        if usage is None:
            return "-"

        root_bytes = self._safe_dict_get(usage, "project_root_bytes")
        if root_bytes is None:
            root_bytes = self._safe_dict_get(usage, "total_bytes")

        parts: list[str] = []
        if root_bytes is not None:
            parts.append(f"Total {self._format_bytes_for_gui(root_bytes)}")

        bucket_labels = (
            ("snapshots_bytes", "snapshots"),
            ("previews_bytes", "previews"),
            ("history_bytes", "history"),
            ("temp_bytes", "temp"),
        )
        bucket_parts: list[str] = []
        for attr, label in bucket_labels:
            value = self._safe_dict_get(usage, attr)
            if value in (None, ""):
                continue
            bucket_parts.append(f"{label} {self._format_bytes_for_gui(value)}")

        if bucket_parts:
            parts.append("; ".join(bucket_parts))

        file_count = self._safe_dict_get(usage, "file_count")
        dir_count = self._safe_dict_get(usage, "dir_count")
        count_parts: list[str] = []
        if file_count is not None:
            count_parts.append(f"{file_count} files")
        if dir_count is not None:
            count_parts.append(f"{dir_count} dirs")
        if count_parts:
            parts.append(", ".join(count_parts))

        return " | ".join(parts) if parts else "-"

    @staticmethod
    def _restore_warning_severity_for_gui(text: object) -> str:
        body = str(text or "").strip()
        if not body:
            return "ok"

        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if not lines:
            return "ok"

        lowered_body = body.lower()
        if lowered_body == "no restore warnings.":
            return "ok"

        def _split_line(line: str) -> tuple[str, str]:
            if ":" not in line:
                return "", line.strip()
            key, value = line.split(":", 1)
            return key.strip().lower(), value.strip()

        def _is_empty_value(value: str) -> bool:
            return value.strip().lower() in {
                "",
                "none",
                "false",
                "0",
                "[]",
                "{}",
                "()",
            }

        severity = "ok"

        for line in lines:
            lowered = line.lower()
            key, value = _split_line(line)
            lowered_value = value.lower()
            empty_value = _is_empty_value(value)

            # Positive schema support is informational.
            if key in {"metadata_schema_supported", "schema_supported"}:
                if lowered_value == "false":
                    return "error"
                continue

            # Error fields only become errors when their value is meaningful.
            if key in {
                "metadata_schema_error",
                "project_open_failed",
                "error",
            }:
                if not empty_value:
                    return "error"
                continue

            # Skipped fields are warnings only when something was actually skipped.
            if "skipped" in key or "skipped" in lowered:
                if not empty_value:
                    severity = "warning"
                continue

            # Free-text hard errors.
            if any(
                token in lowered
                for token in (
                    "corrupt",
                    "invalid",
                    "failed",
                    "unsupported",
                    "newer than supported",
                )
            ):
                if ":" not in line or not empty_value:
                    return "error"

        return severity

    def _apply_project_restore_warning_severity_for_gui(self, severity: str) -> None:
        widget = getattr(self, "project_status_restore_warnings_text", None)
        if widget is None:
            return

        normalized = str(severity or "ok")
        if normalized not in {"ok", "warning", "error"}:
            normalized = "ok"

        set_property = getattr(widget, "setProperty", None)
        if callable(set_property):
            set_property("restoreSeverity", normalized)

        style_getter = getattr(widget, "style", None)
        style = style_getter() if callable(style_getter) else None
        if style is not None:
            unpolish = getattr(style, "unpolish", None)
            polish = getattr(style, "polish", None)
            if callable(unpolish):
                unpolish(widget)
            if callable(polish):
                polish(widget)

        updater = getattr(widget, "update", None)
        if callable(updater):
            updater()


    def _project_history_stack_text_for_gui(self) -> str:
        extra = self._project_metadata_extra_for_gui()

        def _stack_lines(label: str, value: object) -> list[str]:
            if not isinstance(value, list) or not value:
                return [f"{label}: empty"]

            count = len(value)
            unit = "entry" if count == 1 else "entries"
            lines = [f"{label}: {count} {unit}"]

            for index, item in enumerate(value[:5], start=1):
                lines.append(f"  {index}. {item}")

            if count > 5:
                lines.append(f"  ... {count - 5} more")

            return lines

        lines: list[str] = []
        lines.extend(_stack_lines("Undo stack", extra.get("undo_stack")))
        lines.extend(_stack_lines("Redo stack", extra.get("redo_stack")))
        return "\n".join(lines)

    @staticmethod
    def _restore_warning_line_has_value_for_gui(line: object) -> bool:
        text = str(line).strip()
        if not text:
            return False

        lowered = text.lower()
        if lowered in {
            "no restore warnings.",
            "restore completed.",
        }:
            return False

        if ":" not in text:
            return True

        _key, value = text.split(":", 1)
        value = value.strip()
        lowered_value = value.lower()

        return lowered_value not in {
            "",
            "none",
            "false",
            "0",
            "[]",
            "{}",
            "()",
        }

    def _project_restore_warnings_text_for_gui(self) -> str:
        result = getattr(self, "_last_project_restore_result", None)
        if result is None:
            return "No restore warnings."

        lines = self._restore_result_lines(result)
        diagnostic_lines: list[str] = []

        for line in lines:
            text = str(line)
            lowered = text.lower()
            if (
                "skipped" in lowered
                or "corrupt" in lowered
                or "invalid" in lowered
                or "failed" in lowered
                or "error" in lowered
                or "unsupported" in lowered
                or "schema" in lowered
            ):
                diagnostic_lines.append(text)

        return "\n".join(diagnostic_lines) if diagnostic_lines else "No restore warnings."

    def _project_status_summary_for_gui(self) -> dict[str, str]:
        storage = self._safe_call(lambda: self.processor.current_project_storage(), None)
        storage_root = getattr(storage, "root", None)
        is_saved_project = bool(getattr(storage, "is_saved_project", False))

        root_text = "-"
        if self.current_project_path:
            root_text = str(Path(self.current_project_path).expanduser())
        elif storage_root is not None:
            root_text = str(Path(storage_root).expanduser())

        mode_text = "Saved project" if is_saved_project else "Unsaved session"

        mesh = getattr(self.processor, "mesh", None)
        if mesh is None:
            mesh_text = "No mesh loaded"
        else:
            vertices = getattr(mesh, "vertices", None)
            faces = getattr(mesh, "faces", None)
            vertex_count = len(vertices) if vertices is not None else 0
            face_count = len(faces) if faces is not None else 0
            mesh_text = f"{vertex_count:,} vertices / {face_count:,} faces"

        extra = self._project_metadata_extra_for_gui()

        can_undo = bool(self._safe_call(lambda: self.processor.can_undo(), False))
        can_redo = bool(self._safe_call(lambda: self.processor.can_redo(), False))

        undo_stack = extra.get("undo_stack")
        redo_stack = extra.get("redo_stack")
        undo_count = len(undo_stack) if isinstance(undo_stack, list) else None
        redo_count = len(redo_stack) if isinstance(redo_stack, list) else None

        def _history_state_text(available: bool, count: int | None) -> str:
            state = "available" if available else "none"
            if count is None:
                return state
            unit = "entry" if count == 1 else "entries"
            return f"{state} ({count} {unit})"

        history_text = (
            f"Undo: {_history_state_text(can_undo, undo_count)} | "
            f"Redo: {_history_state_text(can_redo, redo_count)}"
        )

        current_snapshot = extra.get("current_mesh_snapshot")
        current_snapshot_text = str(current_snapshot) if current_snapshot else "-"

        latest_history_entry = extra.get("latest_history_entry")
        latest_history_entry_text = str(latest_history_entry) if latest_history_entry else "-"

        sync_reason = extra.get("sync_reason")
        sync_reason_text = str(sync_reason) if sync_reason else "-"

        latest_operation = (
            extra.get("latest_history_operation")
            or extra.get("last_operation")
            or extra.get("sync_reason")
            or None
        )
        if not latest_operation:
            latest_entry = self._safe_call(lambda: self.processor.last_mesh_history_entry(), None)
            latest_operation = getattr(latest_entry, "operation", None)
        latest_operation_text = str(latest_operation) if latest_operation else "-"

        restore_warnings_text = self._project_restore_warnings_text_for_gui()

        return {
            "mode": mode_text,
            "root": root_text,
            "mesh": mesh_text,
            "current_snapshot": current_snapshot_text,
            "history": history_text,
            "latest_operation": latest_operation_text,
            "latest_history_entry": latest_history_entry_text,
            "sync_reason": sync_reason_text,
            "history_stack": self._project_history_stack_text_for_gui(),
            "disk_usage": self._project_disk_usage_text_for_gui(),
            "restore_warnings": restore_warnings_text,
            "restore_warning_severity": self._restore_warning_severity_for_gui(restore_warnings_text),
        }

    def _update_project_status_ui(self) -> None:
        """
        Refresh the I3 project-status widgets if the current UI provides them.

        This is a read-only GUI projection. ProjectStorage/MeshProcessor remain
        the project and mesh-state authorities; MainWindow only formats their
        current state for display.
        """
        if not hasattr(self, "project_status_mode_label"):
            return

        summary = self._project_status_summary_for_gui()

        self.project_status_mode_label.setText(summary["mode"])
        self.project_status_root_label.setText(summary["root"])
        self.project_status_mesh_label.setText(summary["mesh"])
        if hasattr(self, "project_status_snapshot_label"):
            self.project_status_snapshot_label.setText(summary["current_snapshot"])
        self.project_status_history_label.setText(summary["history"])
        self.project_status_latest_operation_label.setText(summary["latest_operation"])
        if hasattr(self, "project_status_history_entry_label"):
            self.project_status_history_entry_label.setText(summary["latest_history_entry"])
        if hasattr(self, "project_status_sync_reason_label"):
            self.project_status_sync_reason_label.setText(summary["sync_reason"])
        if hasattr(self, "project_status_history_stack_text"):
            self.project_status_history_stack_text.setPlainText(summary["history_stack"])
        self.project_status_disk_usage_label.setText(summary["disk_usage"])

        if hasattr(self, "project_status_restore_warnings_text"):
            self.project_status_restore_warnings_text.setPlainText(summary["restore_warnings"])
            self._apply_project_restore_warning_severity_for_gui(
                summary.get("restore_warning_severity", "ok")
            )

    @staticmethod
    def _project_open_failure_message(error: object) -> str:
        """Return a compact user-facing message for failed project opens."""
        text = str(error or "").strip()
        if not text:
            text = repr(error)

        lowered = text.lower()
        if (
            "schema" in lowered
            or "newer than supported" in lowered
            or "unsupported" in lowered
            or "projectstorageerror" in lowered
        ):
            return (
                "Project open failed: unsupported project version or invalid project metadata schema.\n"
                f"{text}"
            )

        return f"Project open failed.\n{text}"

    def _on_project_open_failed(self, error: object) -> str:
        """Record and surface project-open failures without applying partial GUI state."""
        message = self._project_open_failure_message(error)
        schema_related = any(
            token in message.lower()
            for token in ("schema", "unsupported", "newer than supported")
        )
        self._last_project_restore_result = {
            "project_open_failed": True,
            "metadata_schema_supported": not schema_related,
            "error": message,
        }
        self.log(message)
        self.statusBar().showMessage("Project open failed", 5000)
        _update_project_status_ui_if_available(self)
        return message

    def _apply_project_open_result(self, payload: dict[str, object]) -> None:
        storage = payload["storage"]
        restore_result = payload.get("restore_result")
        self._last_project_restore_result = restore_result
        root = Path(getattr(storage, "root", payload.get("project_root", ""))).expanduser().resolve()

        # Defensive project-open fallback: if the background restore reported that
        # the current mesh was not restored, retry the current-mesh snapshot once
        # on the GUI handoff path.  This protects project open from a transient
        # loader/thread failure while keeping MeshProcessor as the only owner of
        # mesh restoration.  Normal successful opens do not enter this path.
        if isinstance(restore_result, dict) and not bool(restore_result.get("restored_current_mesh")):
            try:
                retry = self.processor.restore_project_state_from_metadata(
                    restore_current_mesh=True,
                    restore_history=False,
                    strict=False,
                )
            except Exception as exc:
                merged = dict(restore_result)
                merged["current_mesh_retry_error"] = str(exc)
                restore_result = merged
                self._last_project_restore_result = restore_result
            else:
                if isinstance(retry, dict) and bool(retry.get("restored_current_mesh")):
                    merged = dict(restore_result)
                    merged["restored_current_mesh"] = True
                    merged["current_mesh_retry_restored"] = True
                    merged["current_mesh_retry_snapshot"] = retry.get("current_mesh_snapshot")
                    restore_result = merged
                    self._last_project_restore_result = restore_result

        self.current_project_path = str(root)
        self.current_mesh_path = None
        self.current_output_path = None

        self._clear_manual_edit_preview(silent=True)
        self._clear_hole_fill_preview(silent=True)
        self._reset_hole_fill_ui(status="Project opened. Run Find Hole Candidates if needed.")

        self._refresh_viewport_from_processor()
        mesh = getattr(self.processor, "mesh", None)
        if mesh is not None:
            self._set_mesh_info_from_trimesh(mesh)
        else:
            self._set_mesh_info_empty()

        try:
            self.selection_controller.clear_selection(
                keep_mode=True,
                push=True,
                reason="project_opened",
            )
        except Exception:
            pass

        self.current_file_label.setText(f"Project: {root.name}")
        self._set_topology_result_text(
            "Project opened from disk.\n"
            "Current mesh and undo/redo stacks were restored from project metadata when available."
        )
        self.log(f"Project opened: {root}")
        for line in self._restore_result_lines(restore_result):
            self.log(f"Project restore: {line}")
        self.statusBar().showMessage("Project opened", 3000)
        self._show_page(self.PAGE_VIEWER)
        self._sync_viewport_ui_from_backend()
        self._update_undo_redo_action_state()
        _update_project_status_ui_if_available(self)

    def open_project(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Open FAR MESH project",
            str(Path.cwd()),
        )
        if not path:
            return
        self.open_project_from_path(path)

    def open_project_from_path(self, project_root: str | Path, *, strict: bool = False) -> None:
        root = Path(project_root).expanduser()

        def task() -> object:
            previous_storage = self.processor.current_project_storage()

            try:
                storage = ProjectStorage.open_existing(root)
                self.processor.set_project_storage(storage)
                restore_result = self.processor.restore_project_state_from_metadata(strict=strict)
            except Exception:
                self.processor.set_project_storage(previous_storage)
                raise

            return {
                "storage": storage,
                "project_root": str(Path(storage.root).expanduser().resolve()),
                "restore_result": restore_result,
            }

        def on_success(result: object) -> None:
            assert isinstance(result, dict)
            self._apply_project_open_result(result)

        failure_handler = getattr(self, "_on_project_open_failed", None)
        self._run_task(
            f"Opening project: {root}",
            task,
            on_success,
            failure_handler if callable(failure_handler) else None,
        )

    def save_project(self) -> None:
        def task() -> object:
            metadata_result = self._sync_project_state_metadata_for_gui()
            storage = self.processor.current_project_storage()
            return {
                "storage": storage,
                "project_root": str(Path(storage.root).expanduser().resolve()),
                "metadata_result": metadata_result,
            }

        def on_success(result: object) -> None:
            assert isinstance(result, dict)
            root = Path(str(result["project_root"])).expanduser().resolve()
            self.current_project_path = str(root)
            self.current_file_label.setText(f"Project: {root.name}")
            self.log(f"Project saved: {root}")
            self.statusBar().showMessage("Project saved", 3000)
            self._update_undo_redo_action_state()
            _update_project_status_ui_if_available(self)

        self._run_task("Saving project metadata...", task, on_success)

    def save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save FAR MESH project as",
            str(Path.cwd() / "Untitled.farmesh3"),
            "FAR MESH Project (*.farmesh3);;All Files (*)",
        )
        if not path:
            return

        target_root = self._normalize_project_root_path(path).expanduser()
        if target_root.exists() and (not target_root.is_dir() or any(target_root.iterdir())):
            QMessageBox.warning(
                self,
                "Project folder not empty",
                "Choose a new or empty .farmesh3 folder for Save Project As.",
            )
            return

        def task() -> object:
            source_root = self._current_project_storage_root()
            if source_root is None:
                raise RuntimeError("No active ProjectStorage root is available.")

            # Make sure the source storage metadata is current before copying.
            # This is critical when converting an unsaved session
            # (project_state.json) into a saved .farmesh3 project (project.json).
            source_sync_result = self._sync_project_state_metadata_for_gui(
                reason="gui_save_project_as_before_copy"
            )
            source_storage = self.processor.current_project_storage()
            source_metadata = self._read_storage_metadata_for_save_as(source_storage)

            target = target_root.expanduser().resolve(strict=False)
            source = source_root.expanduser().resolve(strict=True)
            try:
                target.relative_to(source)
            except ValueError:
                pass
            else:
                raise RuntimeError("Cannot save a project inside its own active storage folder.")

            temp_target = target.with_name(f".{target.name}.tmp_copy")
            if temp_target.exists():
                shutil.rmtree(temp_target)

            shutil.copytree(source, temp_target)
            if target.exists():
                # Only empty directories are accepted above. Remove the empty
                # shell directory so the verified copy can be moved into place.
                target.rmdir()
            temp_target.rename(target)

            storage = ProjectStorage.open_existing(target, is_saved_project=True)
            self.processor.set_project_storage(storage)

            restore_metadata = self._select_save_as_restore_metadata(
                target_root=target,
                source_metadata=source_metadata,
            )
            if restore_metadata is not None:
                restore_result = self.processor.restore_project_state_from_metadata(
                    metadata=restore_metadata,
                    strict=False,
                )
            else:
                restore_result = self.processor.restore_project_state_from_metadata(strict=False)

            # After restore, write the migrated state to the saved-project
            # authority file: project.json.  This preserves undo/redo stacks
            # across closing and reopening the .farmesh3 project.
            metadata_result = self._sync_project_state_metadata_for_gui(
                reason="gui_save_project_as"
            )
            return {
                "storage": storage,
                "project_root": str(Path(storage.root).expanduser().resolve()),
                "restore_result": restore_result,
                "metadata_result": metadata_result,
                "source_metadata_result": source_sync_result,
            }

        def on_success(result: object) -> None:
            assert isinstance(result, dict)
            self._last_project_restore_result = result.get("restore_result")
            root = Path(str(result["project_root"])).expanduser().resolve()
            self.current_project_path = str(root)
            self.current_file_label.setText(f"Project: {root.name}")
            self.log(f"Project saved as: {root}")
            self.statusBar().showMessage("Project saved as", 3000)
            self._update_undo_redo_action_state()
            _update_project_status_ui_if_available(self)

        self._run_task(f"Saving project as: {target_root}", task, on_success)

