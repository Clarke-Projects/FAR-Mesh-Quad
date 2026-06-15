from __future__ import annotations

from datetime import datetime
from typing import Any

from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QMainWindow, QMessageBox

from far_mesh.core.manual_edit_pipeline import ManualEditPreview
from far_mesh.core.mesh_processor import MeshProcessor
from far_mesh.system.execution_manager import (
    get_lifecycle_manager,
    shutdown_execution_lifecycle,
)

from .constants import FAR_MESH_APP_ICON
from .main_window_ui import MainWindowUI
from .history_actions import HistoryActionsMixin
from .manual_edit_actions import ManualEditActionsMixin
from .bore_actions import BoreActionsMixin
from .mesh_actions import MeshActionsMixin
from .project_actions import ProjectActionsMixin, _update_project_status_ui_if_available
from .selection_actions import SelectionActionsMixin
from .task_runner import TaskRunnerMixin
from .topology_tools import TopologyToolsMixin
from .viewport_actions import ViewportActionsMixin
from .selection_controller import SelectionController
from .worker import WorkerThread


class MainWindow(
    QMainWindow,
    MainWindowUI,
    TaskRunnerMixin,
    ProjectActionsMixin,
    HistoryActionsMixin,
    TopologyToolsMixin,
    ManualEditActionsMixin,
    ViewportActionsMixin,
    MeshActionsMixin,
    BoreActionsMixin,
    SelectionActionsMixin,
):
    PAGE_LOAD = "load"
    PAGE_REPAIR = "repair"
    PAGE_REMESH = "remesh"
    PAGE_REDUCE = "reduce"
    PAGE_VIEWER = "viewer"
    PAGE_BRUSH = "brush"
    PAGE_BORE = "bore"

    def __init__(self) -> None:
        super().__init__()
        self.processor = MeshProcessor()
        self.current_mesh_path: str | None = None
        self.current_output_path: str | None = None
        self.current_project_path: str | None = None
        self._worker: WorkerThread | None = None
        self._pages: dict[str, Any] = {}
        self._suppress_viewer_sync = False
        self._manual_edit_preview: ManualEditPreview | None = None
        self._last_hole_candidates: list[object] = []
        self._last_hole_candidate_scope: tuple[int, ...] | None = None
        self._hole_fill_preview: object | None = None
        self._bore_boundary_resource: object | None = None
        self._bore_axis_estimate: object | None = None
        self._bore_wall_region: object | None = None
        self._bore_cleanup_preview: ManualEditPreview | None = None
        self._last_project_restore_result: object | None = None
        self._closing = False
        self._close_after_worker = False
        self._shutdown_started = False

        self.setWindowTitle("FAR MESH Quad")
        self.resize(1660, 1000)

        self._build_ui()

        # Branding integration kept in MainWindow as a safe GUI-shell concern.
        # main_window_ui.py owns layout/styling; this keeps the top-level window
        # title/icon stable even if the UI mixin is reused or refactored later.
        self.setWindowTitle("FAR MESH Quad")
        if FAR_MESH_APP_ICON.exists():
            self.setWindowIcon(QIcon(str(FAR_MESH_APP_ICON)))

        # Shared controller-owned selection / brush model.
        self.selection_controller = SelectionController(self.viewport, parent=self)

        if hasattr(self, "brush_btn"):
            self.brush_btn.clicked.connect(self._on_brush_page_requested)
        if hasattr(self, "viewport_open_brush_btn"):
            self.viewport_open_brush_btn.clicked.connect(self._on_brush_page_requested)
        if hasattr(self, "bore_btn"):
            self.bore_btn.clicked.connect(self._on_bore_page_requested)
        if hasattr(self, "viewport_open_bore_btn"):
            self.viewport_open_bore_btn.clicked.connect(self._on_bore_page_requested)

        self._seed_selection_controller_from_ui_defaults()
        self._connect_viewport_signals()
        self._apply_styles()
        self._populate_runtime_options()
        self._on_manual_edit_operation_changed()
        self._sync_viewport_ui_from_backend()
        self._set_busy(False)
        self._set_mesh_info_empty()
        self._update_undo_redo_action_state()
        _update_project_status_ui_if_available(self)
        self._show_page(self.PAGE_LOAD)

    # ------------------------------------------------------------------
    # menu / actions
    # ------------------------------------------------------------------
    def _create_actions(self) -> None:
        self.action_open_project = QAction("Open Project...", self)
        self.action_open_project.triggered.connect(self.open_project)

        self.action_save_project = QAction("Save Project", self)
        self.action_save_project.setShortcut("Ctrl+S")
        self.action_save_project.triggered.connect(self.save_project)

        self.action_save_project_as = QAction("Save Project As...", self)
        self.action_save_project_as.setShortcut("Ctrl+Shift+S")
        self.action_save_project_as.triggered.connect(self.save_project_as)

        self.action_load = QAction("Load Mesh...", self)
        self.action_load.triggered.connect(self.load_mesh)

        # User-facing mesh export remains separate from editable .farmesh3 project save.
        self.action_save = QAction("Export Mesh As...", self)
        self.action_save.triggered.connect(self.save_mesh)

        self.action_undo = QAction("Undo", self)
        self.action_undo.setShortcut("Ctrl+Z")
        self.action_undo.setEnabled(False)
        self.action_undo.triggered.connect(self.undo_mesh_operation)

        self.action_redo = QAction("Redo", self)
        self.action_redo.setShortcut("Ctrl+Y")
        self.action_redo.setEnabled(False)
        self.action_redo.triggered.connect(self.redo_mesh_operation)

        self.action_quit = QAction("Quit", self)
        self.action_quit.triggered.connect(self.close)

        self.action_toggle_viewport_diagnostics = QAction("Show Viewport Diagnostics", self)
        self.action_toggle_viewport_diagnostics.setCheckable(True)
        self.action_toggle_viewport_diagnostics.toggled.connect(self._set_viewport_info_visible)

    def _create_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        file_menu.addAction(self.action_open_project)
        file_menu.addAction(self.action_save_project)
        file_menu.addAction(self.action_save_project_as)
        file_menu.addSeparator()
        file_menu.addAction(self.action_load)
        file_menu.addAction(self.action_save)
        file_menu.addSeparator()
        file_menu.addAction(self.action_quit)

        edit_menu = menu.addMenu("Edit")
        edit_menu.addAction(self.action_undo)
        edit_menu.addAction(self.action_redo)

        view_menu = menu.addMenu("View")
        view_menu.addAction(self.action_toggle_viewport_diagnostics)

    # ------------------------------------------------------------------
    # runtime options
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # selection controller integration
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # viewport integration
    # ------------------------------------------------------------------
    
    
    
    
    
    
    
    

    
    
    
    
    


    # ------------------------------------------------------------------
    # viewer actions
    # ------------------------------------------------------------------
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # ------------------------------------------------------------------
    # remesh UI helpers
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # mesh info
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # logging / busy
    # ------------------------------------------------------------------
    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"[{timestamp}] {message}")

    # ------------------------------------------------------------------
    # payload logging
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # actions (load, save, repair, remesh, reduce)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    
    
    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def _shutdown_viewport_for_close(self) -> None:
        try:
            if hasattr(self.viewport, "shutdown"):
                self.viewport.shutdown()
        except Exception as exc:
            try:
                self.log(f"Viewport shutdown warning: {exc}")
            except Exception:
                pass

    def closeEvent(self, event) -> None:
        """Coordinate application shutdown with lifecycle-owned external work.

        MainWindow only coordinates the close decision. The system lifecycle
        layer remains responsible for cancelling, terminating, killing, waiting,
        and cleaning up PROCESS / SUBPROCESS work.

        Closing while a WorkerThread is still unwinding is handled as a two-phase
        close: first cancel lifecycle-owned work and ignore the event, then close
        again automatically when the worker emits finished.
        """

        lifecycle = get_lifecycle_manager()

        try:
            active_count = int(lifecycle.active_count())
        except Exception as exc:
            active_count = 0
            try:
                self.log(f"Lifecycle active task check failed: {exc}")
            except Exception:
                pass

        worker_running = self._worker is not None and self._worker.isRunning()

        # A shutdown was already requested and the GUI worker is still unwinding
        # from the cancelled operation. Do not prompt again and do not accept yet.
        if getattr(self, "_closing", False) and worker_running:
            event.ignore()
            return

        # Second close after WorkerThread.finished: no worker is running anymore,
        # so it is now safe to accept the close event.
        if getattr(self, "_closing", False) and not worker_running:
            self._shutdown_viewport_for_close()
            event.accept()
            return

        if active_count > 0 or worker_running:
            reply = QMessageBox.question(
                self,
                "FAR MESH is still working",
                (
                    f"{active_count} isolated task(s) are still running.\n\n"
                    "Cancel running task(s) and close FAR MESH?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply != QMessageBox.Yes:
                event.ignore()
                return

            self._closing = True
            self._close_after_worker = bool(worker_running)

            try:
                self.log(
                    f"Shutdown requested: active lifecycle tasks={active_count}, "
                    f"worker_running={worker_running}."
                )
            except Exception:
                pass

            if not getattr(self, "_shutdown_started", False):
                self._shutdown_started = True
                try:
                    shutdown_execution_lifecycle()
                except Exception as exc:
                    try:
                        self.log(f"Lifecycle shutdown warning: {exc}")
                    except Exception:
                        pass

            if worker_running:
                try:
                    self._set_busy(True)
                    self.statusBar().showMessage(
                        "Cancelling external task before closing...",
                        5000,
                    )
                except Exception:
                    pass
                event.ignore()
                return

        self._closing = True

        if not getattr(self, "_shutdown_started", False):
            self._shutdown_started = True
            try:
                shutdown_execution_lifecycle()
            except Exception as exc:
                try:
                    self.log(f"Lifecycle shutdown warning: {exc}")
                except Exception:
                    pass

        self._shutdown_viewport_for_close()
        event.accept()
