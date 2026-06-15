from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .constants import (
    CAD_PRESETS,
    CAMERA_PRESET_LABELS,
    COMPARE_MODE_LABELS,
    DISPLAY_PRESET_LABELS,
    FAR_MESH_APP_ICON,
    FAR_MESH_DESIGN_REFERENCE,
    FAR_MESH_LOGO,
    GUI_ASSETS_DIR,
    REPAIR_ADVANCED_METHODS,
    REPAIR_STRICT_PRESERVE_METHODS,
    SELECTION_MODE_LABELS,
)
from .ui_helpers import (
    FAR_BORDER,
    FAR_CYAN_GLOW,
    FAR_DEEP_BLACK,
    FAR_ELECTRIC_BLUE,
    FAR_PANEL_BLACK,
    FAR_PANEL_BLUE_BLACK,
    FAR_PANEL_SOFT,
    FAR_SIGNAL_ORANGE,
    FAR_TEXT,
    FAR_TEXT_MUTED,
    FAR_WARM_AMBER,
    ToolPage,
)

from far_mesh.viewer.viewport_config import ViewportConfig
from far_mesh.viewer.viewport_factory import create_viewport


class MainWindowUI:
    """
    UI construction mixin for MainWindow.

    Contains:
    - constants / label maps
    - page construction
    - styles
    - generic combo helpers

    Does not contain:
    - processor orchestration
    - mesh operation logic
    - thread/task execution
    - viewport capability sync logic
    """

    # ------------------------------------------------------------------
    # UI shell
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self._create_actions()
        self._create_menu()

        self.setWindowTitle("FAR MESH Quad")
        if FAR_MESH_APP_ICON.exists():
            self.setWindowIcon(QIcon(str(FAR_MESH_APP_ICON)))

        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)
        root_layout.addWidget(self._build_header())

        vertical_split = QSplitter(Qt.Orientation.Vertical)
        root_layout.addWidget(vertical_split, 1)

        top_split = QSplitter(Qt.Orientation.Horizontal)
        vertical_split.addWidget(top_split)

        self.left_rail = self._build_left_rail()
        self.viewport_panel = self._build_viewport_panel()
        self.tool_panel = self._build_tool_panel()
        self.log_panel = self._build_log_panel()

        top_split.addWidget(self.left_rail)
        top_split.addWidget(self.viewport_panel)
        top_split.addWidget(self.tool_panel)
        top_split.setChildrenCollapsible(False)
        top_split.setStretchFactor(0, 0)
        top_split.setStretchFactor(1, 1)
        top_split.setStretchFactor(2, 0)

        vertical_split.addWidget(self.log_panel)
        vertical_split.setChildrenCollapsible(False)
        vertical_split.setStretchFactor(0, 1)
        vertical_split.setStretchFactor(1, 0)

        self.left_rail.setMinimumWidth(120)
        self.viewport_panel.setMinimumWidth(620)
        self.tool_panel.setMinimumWidth(380)
        self.log_panel.setMinimumHeight(150)

        top_split.setSizes([130, 980, 460])
        vertical_split.setSizes([780, 190])

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar, 1)
        self.statusBar().showMessage("Ready")

    def _build_header(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("HeaderFrame")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        self.header_icon_label = QLabel()
        self.header_icon_label.setObjectName("HeaderIconLabel")
        self.header_icon_label.setFixedSize(44, 44)
        self.header_icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if FAR_MESH_APP_ICON.exists():
            icon_pixmap = QPixmap(str(FAR_MESH_APP_ICON))
            if not icon_pixmap.isNull():
                self.header_icon_label.setPixmap(
                    icon_pixmap.scaled(
                        40,
                        40,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )

        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(1)

        title = QLabel("FAR MESH Quad")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Topology-aware mesh processing workstation")
        subtitle.setObjectName("AppSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)

        self.current_file_label = QLabel("No mesh loaded")
        self.current_file_label.setObjectName("CurrentFileLabel")
        self.current_file_label.setWordWrap(True)
        self.current_file_label.setMinimumWidth(260)
        self.current_file_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        # Visible history navigation buttons.
        # These reuse the existing QAction objects from main_window.py, so:
        # - Ctrl+Z / Ctrl+Y still work
        # - enabled/disabled state stays shared
        # - no duplicate undo/redo logic is introduced
        self.header_undo_btn = QToolButton()
        self.header_undo_btn.setDefaultAction(self.action_undo)
        self.header_undo_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.header_undo_btn.setMinimumHeight(30)
        self.header_undo_btn.setObjectName("HeaderActionButton")

        self.header_redo_btn = QToolButton()
        self.header_redo_btn.setDefaultAction(self.action_redo)
        self.header_redo_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.header_redo_btn.setMinimumHeight(30)
        self.header_redo_btn.setObjectName("HeaderActionButton")

        layout.addWidget(self.header_icon_label)
        layout.addLayout(title_col)
        layout.addStretch(1)
        layout.addWidget(self.header_undo_btn)
        layout.addWidget(self.header_redo_btn)
        layout.addSpacing(12)
        layout.addWidget(self.current_file_label)
        return frame

    def _build_left_rail(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("RailFrame")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        def make_button(
            text: str,
            page_key: str | None = None,
            callback: Any | None = None,
        ) -> QToolButton:
            btn = QToolButton()
            btn.setText(text)
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            btn.setCheckable(page_key is not None)
            btn.setMinimumHeight(42)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            if callback is not None:
                btn.clicked.connect(callback)
            elif page_key is not None:
                btn.clicked.connect(lambda checked=False, key=page_key: self._show_page(key))
            layout.addWidget(btn)
            return btn

        self.load_btn = make_button("Load", self.PAGE_LOAD)
        self.repair_btn = make_button("Repair", self.PAGE_REPAIR)
        self.remesh_btn = make_button("Remesh", self.PAGE_REMESH)
        self.reduce_btn = make_button("Reduce", self.PAGE_REDUCE)
        self.save_btn = make_button("Save", callback=self.save_mesh)
        self.viewer_btn = make_button("Viewer", self.PAGE_VIEWER)
        self.brush_btn = make_button("Brush", self.PAGE_BRUSH)
        self.bore_btn = make_button("Bore", self.PAGE_BORE)

        layout.addStretch(1)
        return frame

    def _build_viewport_panel(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("ViewportFrame")
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QHBoxLayout()
        self.viewport_title_label = QLabel("Viewport")
        self.viewport_title_label.setObjectName("SectionTitle")
        self.viewport_status_label = QLabel("Viewport idle")
        self.viewport_status_label.setObjectName("SubtleLabel")
        header.addWidget(self.viewport_title_label)
        header.addStretch(1)
        header.addWidget(self.viewport_status_label)
        layout.addLayout(header)

        quick = QHBoxLayout()
        quick.setSpacing(8)

        self.viewport_quick_preset_combo = QComboBox()
        self._populate_choice_combo(
            self.viewport_quick_preset_combo,
            DISPLAY_PRESET_LABELS.keys(),
            DISPLAY_PRESET_LABELS,
            ["inspection_edges", "viewer_clean", "repair_selection", "shaded_only", "shaded + wireframe", "wireframe"],
        )
        self.viewport_quick_preset_combo.currentIndexChanged.connect(self._on_quick_display_preset_changed)

        self.viewport_quick_compare_combo = QComboBox()
        self._populate_choice_combo(
            self.viewport_quick_compare_combo,
            COMPARE_MODE_LABELS.keys(),
            COMPARE_MODE_LABELS,
            ["current_only", "original_only", "overlay_ghost"],
        )
        self.viewport_quick_compare_combo.currentIndexChanged.connect(self._on_quick_compare_mode_changed)

        self.viewport_quick_grid_check = QCheckBox("Grid")
        self.viewport_quick_grid_check.toggled.connect(self._on_quick_grid_toggled)

        self.viewport_quick_axes_check = QCheckBox("Axes")
        self.viewport_quick_axes_check.toggled.connect(self._on_quick_axes_toggled)

        self.viewport_toggle_info_btn = QToolButton()
        self.viewport_toggle_info_btn.setText("Diagnostics")
        self.viewport_toggle_info_btn.setCheckable(True)
        self.viewport_toggle_info_btn.clicked.connect(self._toggle_viewport_info_panel)

        self.viewport_reset_btn = QPushButton("Reset Camera")
        self.viewport_reset_btn.clicked.connect(self._reset_camera_from_quickbar)

        self.viewport_open_viewer_btn = QPushButton("Viewer Tools")
        self.viewport_open_viewer_btn.clicked.connect(lambda: self._show_page(self.PAGE_VIEWER))

        self.viewport_open_brush_btn = QPushButton("Brush Tools")
        self.viewport_open_brush_btn.clicked.connect(lambda: self._show_page(self.PAGE_BRUSH))

        self.viewport_open_bore_btn = QPushButton("Bore Tools")
        self.viewport_open_bore_btn.clicked.connect(lambda: self._show_page(self.PAGE_BORE))

        quick.addWidget(QLabel("Preset:"))
        quick.addWidget(self.viewport_quick_preset_combo)
        quick.addWidget(QLabel("Compare:"))
        quick.addWidget(self.viewport_quick_compare_combo)
        quick.addSpacing(8)
        quick.addWidget(self.viewport_quick_grid_check)
        quick.addWidget(self.viewport_quick_axes_check)
        quick.addStretch(1)
        quick.addWidget(self.viewport_toggle_info_btn)
        quick.addWidget(self.viewport_reset_btn)
        quick.addWidget(self.viewport_open_viewer_btn)
        quick.addWidget(self.viewport_open_brush_btn)
        quick.addWidget(self.viewport_open_bore_btn)
        layout.addLayout(quick)

        self.viewport = create_viewport(
            self,
            config=ViewportConfig(
                show_edges_default=True,
                edge_width_default=1.5,
                show_grid_default=True,
                show_axes_default=True,
                background_color="#20242b",
            ),
        )
        self.viewport.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.viewport_title_label.setText(f"Viewport ({getattr(self.viewport, 'BACKEND_NAME', 'unknown')})")
        layout.addWidget(self.viewport, 1)

        footer = QHBoxLayout()
        self.vertices_label = QLabel("Vertices: -")
        self.faces_label = QLabel("Faces: -")
        self.bounds_label = QLabel("Bounds: -")
        self.watertight_label = QLabel("Watertight: -")
        footer.addWidget(self.vertices_label)
        footer.addSpacing(12)
        footer.addWidget(self.faces_label)
        footer.addSpacing(12)
        footer.addWidget(self.bounds_label, 1)
        footer.addSpacing(12)
        footer.addWidget(self.watertight_label)
        layout.addLayout(footer)
        return frame

    def _build_tool_panel(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("ToolFrame")
        frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.tool_title_label = QLabel("Tools")
        self.tool_title_label.setObjectName("SectionTitle")
        layout.addWidget(self.tool_title_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.tool_scroll_area = scroll

        scroll_host = QWidget()
        scroll_host.setMinimumWidth(0)
        scroll_host.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        scroll_layout = QVBoxLayout(scroll_host)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)

        self.tool_stack = QStackedWidget()
        self.tool_stack.setMinimumWidth(0)
        self.tool_stack.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        scroll_layout.addWidget(self.tool_stack)

        scroll.setWidget(scroll_host)
        layout.addWidget(scroll, 1)

        self._add_page(self.PAGE_LOAD, "Load", self._build_load_page())
        self._add_page(self.PAGE_REPAIR, "Repair", self._build_repair_page())
        self._add_page(self.PAGE_REMESH, "Remesh", self._build_remesh_page())
        self._add_page(self.PAGE_REDUCE, "Reduce", self._build_reduce_page())
        self._add_page(self.PAGE_VIEWER, "Viewer", self._build_viewer_page())
        self._add_page(self.PAGE_BRUSH, "Brush / Tools", self._build_brush_page())
        self._add_page(self.PAGE_BORE, "Bore Cleanup", self._build_bore_page())

        self._normalize_tool_panel_responsiveness(self.tool_stack)
        return frame


    def _configure_responsive_form_layout(self, form: QFormLayout) -> None:
        """Make right-panel forms resize inside the tool scroll area."""

        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)

    def _normalize_tool_panel_responsiveness(self, root: QWidget) -> None:
        """Prevent right-panel content from forcing horizontal scrolling."""

        widgets = [root, *root.findChildren(QWidget)]

        for widget in widgets:
            try:
                widget.setMinimumWidth(0)
            except Exception:
                pass

            layout = widget.layout()
            if isinstance(layout, QFormLayout):
                self._configure_responsive_form_layout(layout)

            if isinstance(widget, QGroupBox):
                widget.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Preferred,
                )

            if isinstance(widget, QLabel):
                widget.setWordWrap(True)
                widget.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Preferred,
                )

            if isinstance(widget, QComboBox):
                widget.setMinimumWidth(0)
                widget.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Fixed,
                )
                widget.setSizeAdjustPolicy(
                    QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
                )
                widget.setMinimumContentsLength(0)

            if isinstance(widget, (QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton)):
                widget.setMinimumWidth(0)
                widget.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Fixed,
                )

            if isinstance(widget, QPlainTextEdit):
                widget.setMinimumWidth(0)
                widget.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
                widget.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Preferred,
                )


    def _build_load_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)

        brand_frame = QFrame()
        brand_frame.setObjectName("LoadBrandFrame")
        brand_layout = QVBoxLayout(brand_frame)
        brand_layout.setContentsMargins(12, 12, 12, 12)
        brand_layout.setSpacing(8)

        self.load_logo_label = QLabel()
        self.load_logo_label.setObjectName("LoadLogoLabel")
        self.load_logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.load_logo_label.setMinimumHeight(92)
        if FAR_MESH_LOGO.exists():
            logo_pixmap = QPixmap(str(FAR_MESH_LOGO))
            if not logo_pixmap.isNull():
                self.load_logo_label.setPixmap(
                    logo_pixmap.scaled(
                        380,
                        380,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        else:
            self.load_logo_label.setText("FAR MESH Quad")

        self.load_brand_subtitle_label = QLabel(
            "Project-aware mesh repair, remesh, reduce, selection, and topology tools."
        )
        self.load_brand_subtitle_label.setObjectName("SubtleLabel")
        self.load_brand_subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.load_brand_subtitle_label.setWordWrap(True)

        brand_layout.addWidget(self.load_logo_label)
        brand_layout.addWidget(self.load_brand_subtitle_label)

        mesh_group = QGroupBox("Mesh")
        mesh_form = QFormLayout(mesh_group)

        self.load_notes_label = QLabel(
            "Open a mesh file, load it into the processor, and set it as the viewport compare baseline."
        )
        self.load_notes_label.setWordWrap(True)

        self.load_file_btn = QPushButton("Open Mesh...")
        self.load_file_btn.clicked.connect(self.load_mesh)

        mesh_form.addRow("Notes:", self.load_notes_label)
        mesh_form.addRow("", self.load_file_btn)

        project_group = QGroupBox("Project")
        project_layout = QVBoxLayout(project_group)

        self.project_notes_label = QLabel(
            "Open or save editable .farmesh3 project state. "
            "Project save keeps metadata, snapshots, previews, history, and undo/redo references; "
            "mesh export remains a separate action."
        )
        self.project_notes_label.setWordWrap(True)
        self.project_notes_label.setObjectName("SubtleLabel")
        project_layout.addWidget(self.project_notes_label)

        project_button_row = QHBoxLayout()
        project_button_row.setSpacing(8)

        self.load_open_project_btn = QPushButton("Open Project...")
        self.load_open_project_btn.clicked.connect(self.open_project)

        self.load_save_project_btn = QPushButton("Save Project")
        self.load_save_project_btn.clicked.connect(self.save_project)

        self.load_save_project_as_btn = QPushButton("Save Project As...")
        self.load_save_project_as_btn.clicked.connect(self.save_project_as)

        project_button_row.addWidget(self.load_open_project_btn)
        project_button_row.addWidget(self.load_save_project_btn)
        project_button_row.addWidget(self.load_save_project_as_btn)
        project_layout.addLayout(project_button_row)

        project_status_group = QGroupBox("Project Status")
        status_form = QFormLayout(project_status_group)

        self.project_status_mode_label = QLabel("Unsaved session")
        self.project_status_mode_label.setObjectName("SubtleLabel")

        self.project_status_root_label = QLabel("-")
        self.project_status_root_label.setObjectName("SubtleLabel")
        self.project_status_root_label.setWordWrap(True)
        self.project_status_root_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.project_status_mesh_label = QLabel("No mesh loaded")
        self.project_status_mesh_label.setObjectName("SubtleLabel")
        self.project_status_mesh_label.setWordWrap(True)

        self.project_status_snapshot_label = QLabel("-")
        self.project_status_snapshot_label.setObjectName("SubtleLabel")
        self.project_status_snapshot_label.setWordWrap(True)
        self.project_status_snapshot_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.project_status_history_label = QLabel("Undo: - | Redo: -")
        self.project_status_history_label.setObjectName("SubtleLabel")

        self.project_status_latest_operation_label = QLabel("-")
        self.project_status_latest_operation_label.setObjectName("SubtleLabel")
        self.project_status_latest_operation_label.setWordWrap(True)

        self.project_status_history_entry_label = QLabel("-")
        self.project_status_history_entry_label.setObjectName("SubtleLabel")
        self.project_status_history_entry_label.setWordWrap(True)
        self.project_status_history_entry_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.project_status_sync_reason_label = QLabel("-")
        self.project_status_sync_reason_label.setObjectName("SubtleLabel")
        self.project_status_sync_reason_label.setWordWrap(True)
        self.project_status_sync_reason_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.project_status_history_stack_text = QPlainTextEdit()
        self.project_status_history_stack_text.setReadOnly(True)
        self.project_status_history_stack_text.setMaximumHeight(110)
        self.project_status_history_stack_text.setPlainText("Undo stack: empty\nRedo stack: empty")
        self.project_status_history_stack_text.setObjectName("ProjectStatusHistoryStack")

        self.project_status_disk_usage_label = QLabel("-")
        self.project_status_disk_usage_label.setObjectName("SubtleLabel")
        self.project_status_disk_usage_label.setWordWrap(True)

        self.project_status_restore_warnings_text = QPlainTextEdit()
        self.project_status_restore_warnings_text.setReadOnly(True)
        self.project_status_restore_warnings_text.setMaximumHeight(96)
        self.project_status_restore_warnings_text.setPlainText("No restore warnings.")
        self.project_status_restore_warnings_text.setObjectName("ProjectStatusWarnings")

        status_form.addRow("Mode:", self.project_status_mode_label)
        status_form.addRow("Root:", self.project_status_root_label)
        status_form.addRow("Mesh:", self.project_status_mesh_label)
        status_form.addRow("Current snapshot:", self.project_status_snapshot_label)
        status_form.addRow("History:", self.project_status_history_label)
        status_form.addRow("Latest operation:", self.project_status_latest_operation_label)
        status_form.addRow("Latest history entry:", self.project_status_history_entry_label)
        status_form.addRow("Last sync:", self.project_status_sync_reason_label)
        status_form.addRow("History stack:", self.project_status_history_stack_text)
        status_form.addRow("Disk usage:", self.project_status_disk_usage_label)
        status_form.addRow("Restore warnings:", self.project_status_restore_warnings_text)

        layout.addWidget(brand_frame)
        layout.addWidget(mesh_group)
        layout.addWidget(project_group)
        layout.addWidget(project_status_group)
        layout.addStretch(1)
        return page

    def _build_repair_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        group = QGroupBox("Repair")
        form = QFormLayout(group)

        self.repair_method_combo = QComboBox()
        self.repair_method_combo.currentIndexChanged.connect(self._on_repair_method_ui_changed)

        self.repair_join_comp_check = QCheckBox("Join components")
        self.repair_join_comp_check.setChecked(True)
        self.repair_join_comp_check.setToolTip(
            "Try to merge disconnected components where the workflow supports it."
        )

        self.repair_fill_holes_check = QCheckBox("Fill holes")
        self.repair_fill_holes_check.setChecked(True)
        self.repair_fill_holes_check.setToolTip(
            "Allow the repair workflow to close simple holes when supported."
        )

        self.repair_advanced_group = QGroupBox("Advanced Repair Tuning")
        self.repair_advanced_group.setCheckable(True)
        self.repair_advanced_group.setChecked(False)
        self.repair_advanced_group.setToolTip(
            "Fine-tune PyMeshLab-style cleanup behavior. "
            "Leave this collapsed for workflow defaults."
        )

        repair_adv_layout = QVBoxLayout(self.repair_advanced_group)
        repair_adv_layout.setContentsMargins(8, 8, 8, 8)

        self.repair_advanced_note_label = QLabel(
            "Use these controls only when you want to tune feature preservation or topology cleanup manually. "
            "They are most relevant for PyMeshLab-based workflows such as CAD-safe cleanup."
        )
        self.repair_advanced_note_label.setWordWrap(True)
        self.repair_advanced_note_label.setObjectName("SubtleLabel")
        repair_adv_layout.addWidget(self.repair_advanced_note_label)

        self.repair_advanced_content = QWidget()
        adv_form = QFormLayout(self.repair_advanced_content)

        self.repair_preserve_features_check = QCheckBox("Preserve features (disable T-vertex cleanup)")
        self.repair_preserve_features_check.setToolTip(
            "Prioritize keeping fine engineered features such as bores, rims, and thin local details. "
            "When enabled, T-vertex cleanup is disabled."
        )
        self.repair_preserve_features_check.toggled.connect(self._sync_repair_tvertex_ui)

        self.repair_edge_method_combo = QComboBox()
        self.repair_edge_method_combo.addItem("Split Vertices (preserve shape)", "split_vertices")
        self.repair_edge_method_combo.addItem("Remove Faces (more destructive)", "remove_faces")
        self.repair_edge_method_combo.setToolTip(
            "How to resolve non-manifold edges.\n"
            "Split Vertices is safer for CAD-like meshes because it avoids deleting local surface patches.\n"
            "Remove Faces is more aggressive and may remove problematic regions entirely."
        )

        self.repair_vertex_drift_spin = QDoubleSpinBox()
        self.repair_vertex_drift_spin.setRange(0.0, 0.25)
        self.repair_vertex_drift_spin.setDecimals(4)
        self.repair_vertex_drift_spin.setSingleStep(0.01)
        self.repair_vertex_drift_spin.setValue(0.0)
        self.repair_vertex_drift_spin.setToolTip(
            "Controls how far newly split non-manifold vertices are moved.\n"
            "0.0 keeps them in place and best preserves CAD features.\n"
            "Higher values may improve cleanup but can shift geometry."
        )

        self.repair_tvertex_enable_check = QCheckBox("Enable T-vertex cleanup")
        self.repair_tvertex_enable_check.setChecked(True)
        self.repair_tvertex_enable_check.setToolTip(
            "Fix T-vertices after other cleanup stages.\n"
            "This can help topology, but may simplify or damage delicate features."
        )
        self.repair_tvertex_enable_check.toggled.connect(self._sync_repair_tvertex_ui)

        self.repair_tvertex_method_combo = QComboBox()
        self.repair_tvertex_method_combo.addItem("Edge Flip (safer)", "edge_flip")
        self.repair_tvertex_method_combo.addItem("Edge Collapse (more aggressive)", "edge_collapse")
        self.repair_tvertex_method_combo.setToolTip(
            "Method used for T-vertex cleanup.\n"
            "Edge Flip changes triangulation more gently.\n"
            "Edge Collapse is stronger and more likely to eat into small CAD details."
        )

        self.repair_tvertex_threshold_spin = QDoubleSpinBox()
        self.repair_tvertex_threshold_spin.setRange(1.0, 90.0)
        self.repair_tvertex_threshold_spin.setDecimals(2)
        self.repair_tvertex_threshold_spin.setSingleStep(1.0)
        self.repair_tvertex_threshold_spin.setValue(5.0)
        self.repair_tvertex_threshold_spin.setToolTip(
            "Detection threshold for T-vertex cleanup.\n"
            "Lower values are more conservative.\n"
            "A value around 5.0 is a good starting point for feature preservation."
        )

        self.repair_tvertex_repeat_check = QCheckBox("Repeat until convergence")
        self.repair_tvertex_repeat_check.setToolTip(
            "Run T-vertex cleanup repeatedly until no more candidates are found.\n"
            "This can improve cleanup but is more destructive."
        )

        adv_form.addRow("", self.repair_preserve_features_check)
        adv_form.addRow("Non-manifold edge method:", self.repair_edge_method_combo)
        adv_form.addRow("Vertex drift:", self.repair_vertex_drift_spin)
        adv_form.addRow("", self.repair_tvertex_enable_check)
        adv_form.addRow("T-vertex method:", self.repair_tvertex_method_combo)
        adv_form.addRow("T-vertex threshold:", self.repair_tvertex_threshold_spin)
        adv_form.addRow("", self.repair_tvertex_repeat_check)

        repair_adv_layout.addWidget(self.repair_advanced_content)
        self.repair_advanced_group.toggled.connect(self._on_repair_advanced_group_toggled)

        self.repair_btn_run = QPushButton("Run Repair")
        self.repair_btn_run.clicked.connect(self.run_repair)

        self.repair_o3d_hole_size_spin = QDoubleSpinBox()
        self.repair_o3d_hole_size_spin.setRange(0.001, 1_000_000.0)
        self.repair_o3d_hole_size_spin.setDecimals(3)
        self.repair_o3d_hole_size_spin.setSingleStep(0.1)
        self.repair_o3d_hole_size_spin.setValue(1_000_000.0)
        self.repair_o3d_hole_size_spin.setToolTip(
            "Open3D tensor fill_holes dry-run threshold. "
            "This only reports what would be filled; it does not modify the mesh."
        )

        self.repair_o3d_max_faces_spin = QSpinBox()
        self.repair_o3d_max_faces_spin.setRange(0, 10_000_000)
        self.repair_o3d_max_faces_spin.setValue(100)
        self.repair_o3d_max_faces_spin.setToolTip(
            "Policy guard for Open3D tensor fill_holes dry-run: maximum allowed added faces."
        )

        self.repair_o3d_max_vertices_spin = QSpinBox()
        self.repair_o3d_max_vertices_spin.setRange(0, 10_000_000)
        self.repair_o3d_max_vertices_spin.setValue(0)
        self.repair_o3d_max_vertices_spin.setToolTip(
            "Policy guard for Open3D tensor fill_holes dry-run: maximum allowed added vertices."
        )

        self.repair_o3d_max_candidate_delta_spin = QSpinBox()
        self.repair_o3d_max_candidate_delta_spin.setRange(0, 10_000_000)
        self.repair_o3d_max_candidate_delta_spin.setValue(1)
        self.repair_o3d_max_candidate_delta_spin.setToolTip(
            "Policy guard for Open3D tensor fill_holes dry-run: maximum allowed filled candidates."
        )

        self.repair_o3d_dry_run_btn = QPushButton("Open3D Fill Holes Dry Run")
        self.repair_o3d_dry_run_btn.setToolTip(
            "Inspect Open3D tensor fill_holes on a copy. "
            "Logs candidate counts, faces added, diagnostics, and policy result without changing the mesh."
        )
        self.repair_o3d_dry_run_btn.clicked.connect(self.run_open3d_fill_holes_dry_run)

        self.repair_o3d_apply_btn = QPushButton("Apply Open3D Fill Holes Repair")
        self.repair_o3d_apply_btn.setToolTip(
            "Apply Open3D tensor fill_holes only if the same dry-run policy allows it. "
            "This writes normal mesh history and supports undo/redo."
        )
        self.repair_o3d_apply_btn.clicked.connect(self.run_open3d_fill_holes_guarded_repair)

        form.addRow("Method:", self.repair_method_combo)
        form.addRow("", self.repair_join_comp_check)
        form.addRow("", self.repair_fill_holes_check)
        form.addRow("", self.repair_advanced_group)
        form.addRow("", self.repair_btn_run)
        form.addRow("Open3D hole size:", self.repair_o3d_hole_size_spin)
        form.addRow("Open3D max faces added:", self.repair_o3d_max_faces_spin)
        form.addRow("Open3D max vertices added:", self.repair_o3d_max_vertices_spin)
        form.addRow("Open3D max candidates filled:", self.repair_o3d_max_candidate_delta_spin)
        form.addRow("", self.repair_o3d_dry_run_btn)
        form.addRow("", self.repair_o3d_apply_btn)

        layout.addWidget(group)
        layout.addStretch(1)

        self._update_repair_advanced_ui_for_method(None)
        return page

    def _build_remesh_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        source_group = QGroupBox("Source & Backend")
        source_form = QFormLayout(source_group)

        self.remesh_source_combo = QComboBox()
        self.remesh_source_combo.addItem("Original loaded source", "source")
        self.remesh_source_combo.addItem("Current working mesh", "current")

        self.remesh_backend_combo = QComboBox()
        self.remesh_backend_combo.addItem("Instant Meshes", "instant_meshes")
        self.remesh_backend_combo.addItem("QuadWild-BiMDF", "quadwild_bimdf")
        self.remesh_backend_combo.currentIndexChanged.connect(self._on_remesh_backend_changed)

        source_form.addRow("Input mode:", self.remesh_source_combo)
        source_form.addRow("Backend:", self.remesh_backend_combo)

        cad_group = QGroupBox("CAD Preset")
        cad_form = QFormLayout(cad_group)

        self.cad_preset_combo = QComboBox()
        for label in CAD_PRESETS.keys():
            self.cad_preset_combo.addItem(label, label)
        self.cad_preset_combo.currentIndexChanged.connect(self._apply_cad_preset)
        cad_form.addRow("Preset:", self.cad_preset_combo)

        self.instant_group = QGroupBox("Instant Meshes Parameters")
        instant_form = QFormLayout(self.instant_group)

        self.remesh_target_faces_spin = QSpinBox()
        self.remesh_target_faces_spin.setRange(10, 10_000_000)
        self.remesh_target_faces_spin.setValue(5000)

        self.remesh_crease_angle_spin = QDoubleSpinBox()
        self.remesh_crease_angle_spin.setRange(0.0, 180.0)
        self.remesh_crease_angle_spin.setSingleStep(1.0)
        self.remesh_crease_angle_spin.setValue(30.0)

        self.remesh_smooth_iters_spin = QSpinBox()
        self.remesh_smooth_iters_spin.setRange(0, 200)
        self.remesh_smooth_iters_spin.setValue(2)

        self.remesh_deterministic_check = QCheckBox("Deterministic")
        self.remesh_deterministic_check.setChecked(False)

        instant_form.addRow("Target faces:", self.remesh_target_faces_spin)
        instant_form.addRow("Crease angle:", self.remesh_crease_angle_spin)
        instant_form.addRow("Smooth iters:", self.remesh_smooth_iters_spin)
        instant_form.addRow("", self.remesh_deterministic_check)

        self.quad_group = QGroupBox("QuadWild-BiMDF Parameters")
        quad_form = QFormLayout(self.quad_group)

        self.quadwild_stage1_combo = QComboBox()
        self.quadwild_stage2_combo = QComboBox()
        self.quadwild_stage2_combo.setEditable(True)
        self.quadwild_stage2_combo.addItem(
            "flow_noalign_lemon",
            "config/main_config/flow_noalign_lemon.txt",
        )

        self.quadwild_do_remesh_check = QCheckBox("Stage 1 do_remesh")
        self.quadwild_do_remesh_check.setChecked(True)

        self.quadwild_sharp_spin = QDoubleSpinBox()
        self.quadwild_sharp_spin.setRange(0.0, 180.0)
        self.quadwild_sharp_spin.setSingleStep(1.0)
        self.quadwild_sharp_spin.setValue(35.0)

        self.quadwild_alpha_spin = QDoubleSpinBox()
        self.quadwild_alpha_spin.setDecimals(4)
        self.quadwild_alpha_spin.setRange(0.0001, 10.0)
        self.quadwild_alpha_spin.setSingleStep(0.01)
        self.quadwild_alpha_spin.setValue(0.02)

        self.quadwild_scale_spin = QDoubleSpinBox()
        self.quadwild_scale_spin.setDecimals(3)
        self.quadwild_scale_spin.setRange(0.01, 100.0)
        self.quadwild_scale_spin.setSingleStep(0.1)
        self.quadwild_scale_spin.setValue(1.0)

        self.quadwild_use_original_check = QCheckBox("Prefer original OBJ/PLY for QuadWild")
        self.quadwild_use_original_check.setChecked(True)

        self.quadwild_workflow_check = QCheckBox("Pre-repair workflow")
        self.quadwild_workflow_check.setChecked(False)

        self.quadwild_cleanup_combo = QComboBox()

        self.quadwild_fill_holes_check = QCheckBox("Fill holes in pre-repair workflow")
        self.quadwild_fill_holes_check.setChecked(True)

        self.quadwild_auto_reduce_check = QCheckBox("Auto reduce after QuadWild")
        self.quadwild_auto_reduce_check.setChecked(False)

        self.quadwild_auto_reduce_target_spin = QSpinBox()
        self.quadwild_auto_reduce_target_spin.setRange(10, 10_000_000)
        self.quadwild_auto_reduce_target_spin.setValue(50000)

        self.quadwild_auto_reduce_boundary_weight_spin = QDoubleSpinBox()
        self.quadwild_auto_reduce_boundary_weight_spin.setRange(0.0, 100.0)
        self.quadwild_auto_reduce_boundary_weight_spin.setSingleStep(0.5)
        self.quadwild_auto_reduce_boundary_weight_spin.setValue(5.0)

        self.quadwild_auto_reduce_cleanup_check = QCheckBox("Cleanup after auto reduction")
        self.quadwild_auto_reduce_cleanup_check.setChecked(True)

        self.quadwild_post_decimate_check = QCheckBox("Post decimate shortcut")
        self.quadwild_post_decimate_check.setChecked(False)

        self.quadwild_decimate_target_spin = QSpinBox()
        self.quadwild_decimate_target_spin.setRange(10, 10_000_000)
        self.quadwild_decimate_target_spin.setValue(5000)

        quad_form.addRow("Stage 1 preset:", self.quadwild_stage1_combo)
        quad_form.addRow("Stage 2 config:", self.quadwild_stage2_combo)
        quad_form.addRow("", self.quadwild_do_remesh_check)
        quad_form.addRow("Sharp threshold:", self.quadwild_sharp_spin)
        quad_form.addRow("Alpha:", self.quadwild_alpha_spin)
        quad_form.addRow("Scale factor:", self.quadwild_scale_spin)
        quad_form.addRow("", self.quadwild_use_original_check)
        quad_form.addRow("", self.quadwild_workflow_check)
        quad_form.addRow("Workflow cleanup:", self.quadwild_cleanup_combo)
        quad_form.addRow("", self.quadwild_fill_holes_check)
        quad_form.addRow("", self.quadwild_auto_reduce_check)
        quad_form.addRow("Auto reduce target:", self.quadwild_auto_reduce_target_spin)
        quad_form.addRow("Boundary weight:", self.quadwild_auto_reduce_boundary_weight_spin)
        quad_form.addRow("", self.quadwild_auto_reduce_cleanup_check)
        quad_form.addRow("", self.quadwild_post_decimate_check)
        quad_form.addRow("Post decimate target:", self.quadwild_decimate_target_spin)

        self.remesh_btn_run = QPushButton("Run Remesh")
        self.remesh_btn_run.clicked.connect(self.run_remesh)

        layout.addWidget(source_group)
        layout.addWidget(cad_group)
        layout.addWidget(self.instant_group)
        layout.addWidget(self.quad_group)
        layout.addWidget(self.remesh_btn_run)
        layout.addStretch(1)
        return page

    def _build_reduce_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        group = QGroupBox("Reduce")
        form = QFormLayout(group)

        self.reduce_backend_combo = QComboBox()

        self.reduce_target_faces_spin = QSpinBox()
        self.reduce_target_faces_spin.setRange(10, 10_000_000)
        self.reduce_target_faces_spin.setValue(50000)

        self.reduce_boundary_weight_spin = QDoubleSpinBox()
        self.reduce_boundary_weight_spin.setRange(0.0, 100.0)
        self.reduce_boundary_weight_spin.setSingleStep(0.5)
        self.reduce_boundary_weight_spin.setValue(5.0)

        self.reduce_cleanup_check = QCheckBox("Cleanup after reduction")
        self.reduce_cleanup_check.setChecked(True)

        self.reduce_btn_run = QPushButton("Run Reduction")
        self.reduce_btn_run.clicked.connect(self.run_reduce)

        form.addRow("Backend:", self.reduce_backend_combo)
        form.addRow("Target faces:", self.reduce_target_faces_spin)
        form.addRow("Boundary weight:", self.reduce_boundary_weight_spin)
        form.addRow("", self.reduce_cleanup_check)
        form.addRow("", self.reduce_btn_run)

        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _build_viewer_page(self) -> QWidget:
        """
        Viewer page after selection-pipeline migration phase 1:
        - display / compare / overlays
        - camera tools
        - diagnostics / screenshots / markers

        Active selection and clipping controls are intentionally moved to Brush / Tools.
        """
        page = QWidget()
        layout = QVBoxLayout(page)

        info_group = QGroupBox("Viewer Role")
        info_layout = QVBoxLayout(info_group)
        self.viewer_page_note_label = QLabel(
            "This page is now reserved for display, camera, diagnostics, and capture tools. "
            "Active selection mode, brush interaction, and clip / section controls live under Brush / Tools."
        )
        self.viewer_page_note_label.setWordWrap(True)
        self.viewer_page_note_label.setObjectName("SubtleLabel")
        info_layout.addWidget(self.viewer_page_note_label)

        display_group = QGroupBox("Display")
        display_form = QFormLayout(display_group)

        self.viewer_preset_combo = QComboBox()
        self._populate_choice_combo(
            self.viewer_preset_combo,
            getattr(self.viewport, "DISPLAY_PRESETS", DISPLAY_PRESET_LABELS.keys()),
            DISPLAY_PRESET_LABELS,
            ["inspection_edges", "viewer_clean", "repair_selection", "shaded_only", "shaded + wireframe", "wireframe"],
        )
        self.viewer_preset_combo.currentIndexChanged.connect(self._on_display_preset_changed)

        self.viewer_compare_combo = QComboBox()
        self._populate_choice_combo(
            self.viewer_compare_combo,
            getattr(self.viewport, "COMPARE_MODES", COMPARE_MODE_LABELS.keys()),
            COMPARE_MODE_LABELS,
            ["current_only", "original_only", "overlay_ghost"],
        )
        self.viewer_compare_combo.currentIndexChanged.connect(self._on_compare_mode_changed)

        self.viewer_edges_check = QCheckBox("Show edges / wire overlay")
        self.viewer_edges_check.setChecked(True)
        self.viewer_edges_check.toggled.connect(self._on_viewer_edges_toggled)

        self.viewer_edge_width_spin = QDoubleSpinBox()
        self.viewer_edge_width_spin.setRange(0.1, 10.0)
        self.viewer_edge_width_spin.setSingleStep(0.1)
        self.viewer_edge_width_spin.setValue(1.5)
        self.viewer_edge_width_spin.valueChanged.connect(self.viewport.set_edge_width)

        self.viewer_grid_check = QCheckBox("Show floor grid")
        self.viewer_grid_check.setChecked(True)
        self.viewer_grid_check.toggled.connect(self._on_viewer_grid_toggled)

        self.viewer_axes_check = QCheckBox("Show axes")
        self.viewer_axes_check.setChecked(True)
        self.viewer_axes_check.toggled.connect(self._on_viewer_axes_toggled)

        self.viewer_boundary_check = QCheckBox("Highlight open boundaries")
        self.viewer_boundary_check.setChecked(False)
        self.viewer_boundary_check.toggled.connect(self.viewport.set_boundary_highlight_visible)

        display_form.addRow("Preset:", self.viewer_preset_combo)
        display_form.addRow("Compare:", self.viewer_compare_combo)
        display_form.addRow("", self.viewer_edges_check)
        display_form.addRow("Edge width:", self.viewer_edge_width_spin)
        display_form.addRow("", self.viewer_grid_check)
        display_form.addRow("", self.viewer_axes_check)
        display_form.addRow("", self.viewer_boundary_check)

        camera_group = QGroupBox("Camera")
        camera_form = QFormLayout(camera_group)

        self.viewer_camera_combo = QComboBox()
        self._populate_choice_combo(
            self.viewer_camera_combo,
            getattr(self.viewport, "CAMERA_PRESETS", CAMERA_PRESET_LABELS.keys()),
            CAMERA_PRESET_LABELS,
            ["isometric", "front", "back", "left", "right", "top", "bottom"],
        )

        self.viewer_apply_camera_btn = QPushButton("Apply Camera Preset")
        self.viewer_apply_camera_btn.clicked.connect(self._apply_camera_preset)

        self.viewer_focus_selection_btn = QPushButton("Focus Selection")
        self.viewer_focus_selection_btn.clicked.connect(self.viewport.focus_on_selection)

        self.viewer_reset_btn = QPushButton("Reset Camera")
        self.viewer_reset_btn.clicked.connect(self.viewport.reset_camera)

        camera_form.addRow("Preset:", self.viewer_camera_combo)
        camera_form.addRow("", self.viewer_apply_camera_btn)
        camera_form.addRow("", self.viewer_focus_selection_btn)
        camera_form.addRow("", self.viewer_reset_btn)

        tools_group = QGroupBox("Tools")
        tools_form = QFormLayout(tools_group)

        self.viewer_screenshot_btn = QPushButton("Capture Screenshot")
        self.viewer_screenshot_btn.clicked.connect(self.capture_viewport_image)

        self.viewer_drop_marker_btn = QPushButton("Drop Marker at Last Pick")
        self.viewer_drop_marker_btn.clicked.connect(self._drop_marker_at_last_pick)

        self.viewer_diagnostics_check = QCheckBox("Show viewport diagnostics")
        self.viewer_diagnostics_check.toggled.connect(self._set_viewport_info_visible)

        tools_form.addRow("", self.viewer_screenshot_btn)
        tools_form.addRow("", self.viewer_drop_marker_btn)
        tools_form.addRow("", self.viewer_diagnostics_check)

        layout.addWidget(info_group)
        layout.addWidget(display_group)
        layout.addWidget(camera_group)
        layout.addWidget(tools_group)
        layout.addStretch(1)
        return page

    def _build_brush_page(self) -> QWidget:
        """
        Brush / Tools page after selection-pipeline migration phase 1:
        - active selection mode lives here
        - active clip / section tools live here
        - manual edit preview/commit lives here

        Compatibility note:
        Some moved widgets intentionally keep their historical attribute names
        (viewer_selection_combo, viewer_clip_axis_combo, etc.) so the current
        controller can still bind to them until main_window.py is fully migrated
        to the new SelectionController-owned API.
        """
        page = QWidget()
        layout = QVBoxLayout(page)

        def _connect_if_callable(signal: Any, method_name: str) -> None:
            callback = getattr(self, method_name, None)
            if callable(callback):
                signal.connect(callback)

        def _make_vector3_row(
            prefix: str,
            defaults: tuple[float, float, float],
        ) -> tuple[QWidget, QDoubleSpinBox, QDoubleSpinBox, QDoubleSpinBox]:
            host = QWidget()
            row = QHBoxLayout(host)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)

            spins: list[QDoubleSpinBox] = []
            for axis, value in zip(("X", "Y", "Z"), defaults):
                label = QLabel(axis)
                spin = QDoubleSpinBox()
                spin.setDecimals(4)
                spin.setRange(-1_000_000.0, 1_000_000.0)
                spin.setSingleStep(0.1)
                spin.setValue(float(value))
                row.addWidget(label)
                row.addWidget(spin, 1)
                spins.append(spin)

            setattr(self, f"{prefix}_x_spin", spins[0])
            setattr(self, f"{prefix}_y_spin", spins[1])
            setattr(self, f"{prefix}_z_spin", spins[2])
            return host, spins[0], spins[1], spins[2]

        self.brush_backend_note_label = QLabel(
            "Brush / Tools is now the active interaction page for mesh selection, brush drag, section clipping, "
            "and manual edit previews. Viewer is display-oriented; this page owns selection workflow."
        )
        self.brush_backend_note_label.setObjectName("SubtleLabel")
        self.brush_backend_note_label.setWordWrap(True)

        interaction_group = QGroupBox("Selection Session")
        interaction_form = QFormLayout(interaction_group)

        # Historical compatibility attribute kept for current controller.
        self.viewer_selection_combo = QComboBox()
        self._populate_choice_combo(
            self.viewer_selection_combo,
            getattr(self.viewport, "SELECTION_MODES", SELECTION_MODE_LABELS.keys()),
            SELECTION_MODE_LABELS,
            ["none", "point", "face", "edge", "mesh"],
        )
        self.viewer_selection_combo.currentIndexChanged.connect(self._on_selection_mode_changed)

        self.brush_selection_mode_combo = QComboBox()
        self.brush_selection_mode_combo.addItem("Face Brush", "face")
        self.brush_selection_mode_combo.addItem("Point Brush", "point")
        self.brush_selection_mode_combo.addItem("Edge Brush", "edge")
        self.brush_selection_mode_combo.currentIndexChanged.connect(self._on_brush_selection_mode_changed)

        self.brush_enable_check = QCheckBox("Enable drag brush selection")
        self.brush_enable_check.toggled.connect(self._on_brush_enabled_toggled)

        self.brush_boundary_check = QCheckBox("Highlight open boundaries")
        self.brush_boundary_check.toggled.connect(self._on_brush_boundary_toggled)

        self.brush_selection_info_label = QLabel("Selection: none")
        self.brush_selection_info_label.setWordWrap(True)
        self.brush_selection_info_label.setObjectName("SubtleLabel")

        self.viewer_clear_selection_btn = QPushButton("Clear Selection")
        self.viewer_clear_selection_btn.clicked.connect(self._clear_viewport_selection)

        interaction_form.addRow("Selection via mouse:", self.viewer_selection_combo)
        interaction_form.addRow("Brush mode:", self.brush_selection_mode_combo)
        interaction_form.addRow("Brush:", self.brush_enable_check)
        interaction_form.addRow("Boundaries:", self.brush_boundary_check)
        interaction_form.addRow("State:", self.brush_selection_info_label)
        interaction_form.addRow("", self.viewer_clear_selection_btn)

        actions_group = QGroupBox("Selection Actions")
        actions_form = QFormLayout(actions_group)

        self.brush_focus_btn = QPushButton("Focus Selection")
        self.brush_focus_btn.clicked.connect(self.viewport.focus_on_selection)

        self.brush_clear_btn = QPushButton("Clear Selection")
        self.brush_clear_btn.clicked.connect(self._clear_viewport_selection)

        self.brush_grow_btn = QPushButton("Grow Face Selection")
        self.brush_grow_btn.clicked.connect(self._grow_current_selection)

        self.brush_shrink_btn = QPushButton("Shrink Face Selection")
        self.brush_shrink_btn.clicked.connect(self._shrink_current_selection)

        self.brush_connected_points_btn = QPushButton("Select Connected Points")
        self.brush_connected_points_btn.clicked.connect(self._select_connected_points_from_current)

        actions_form.addRow("Camera:", self.brush_focus_btn)
        actions_form.addRow("Clear:", self.brush_clear_btn)
        actions_form.addRow("Faces:", self.brush_grow_btn)
        actions_form.addRow("Faces:", self.brush_shrink_btn)
        actions_form.addRow("Points:", self.brush_connected_points_btn)

        topology_group = QGroupBox("Topology / Holes")
        topology_layout = QVBoxLayout(topology_group)
        topology_layout.setContentsMargins(8, 8, 8, 8)
        topology_layout.setSpacing(8)

        self.topology_info_label = QLabel(
            "Read-only Phase 2 topology tools. These use the current face selection when faces are selected; "
            "otherwise they analyze the full loaded mesh. No mesh data is modified."
        )
        self.topology_info_label.setWordWrap(True)
        self.topology_info_label.setObjectName("SubtleLabel")

        topology_buttons_row = QWidget()
        topology_buttons_layout = QHBoxLayout(topology_buttons_row)
        topology_buttons_layout.setContentsMargins(0, 0, 0, 0)
        topology_buttons_layout.setSpacing(8)

        self.topology_analyze_btn = QPushButton("Analyze Topology")
        self.topology_find_holes_btn = QPushButton("Find Hole Candidates")

        _connect_if_callable(self.topology_analyze_btn.clicked, "_on_analyze_topology_clicked")
        _connect_if_callable(self.topology_find_holes_btn.clicked, "_on_find_hole_candidates_clicked")

        topology_buttons_layout.addWidget(self.topology_analyze_btn)
        topology_buttons_layout.addWidget(self.topology_find_holes_btn)

        self.topology_result_text = QPlainTextEdit()
        self.topology_result_text.setReadOnly(True)
        self.topology_result_text.setMinimumHeight(130)
        self.topology_result_text.setPlaceholderText(
            "Topology and hole-candidate results will appear here."
        )

        topology_layout.addWidget(self.topology_info_label)
        topology_layout.addWidget(topology_buttons_row)
        topology_layout.addWidget(self.topology_result_text)

        hole_fill_group = QGroupBox("Hole Fill Preview")
        hole_fill_form = QFormLayout(hole_fill_group)

        self.hole_fill_info_label = QLabel(
            "Phase 2 hole filling will be preview-first. Find hole candidates first, "
            "then build a non-destructive fill preview before any commit is allowed."
        )
        self.hole_fill_info_label.setWordWrap(True)
        self.hole_fill_info_label.setObjectName("SubtleLabel")

        self.hole_fill_candidate_combo = QComboBox()
        self.hole_fill_candidate_combo.addItem("No hole candidates detected", None)
        self.hole_fill_candidate_combo.setEnabled(False)
        self.hole_fill_candidate_combo.setToolTip(
            "Detected hole candidates will appear here after running Find Hole Candidates."
        )

        self.hole_fill_method_combo = QComboBox()
        self.hole_fill_method_combo.addItem("Triangulate boundary fan (preview)", "fan")
        self.hole_fill_method_combo.addItem("Fan triangulate alias", "fan_triangulate")
        self.hole_fill_method_combo.addItem("Center fan", "center_fan")
        self.hole_fill_method_combo.setToolTip(
            "Hole-fill methods are capability-gated. "
            "Open3D tensor fill_holes appears when available and currently requires a single hole candidate."
        )

        self.hole_fill_max_area_spin = QDoubleSpinBox()
        self.hole_fill_max_area_spin.setRange(0.0, 1_000_000_000.0)
        self.hole_fill_max_area_spin.setDecimals(4)
        self.hole_fill_max_area_spin.setSingleStep(0.1)
        self.hole_fill_max_area_spin.setValue(0.0)
        self.hole_fill_max_area_spin.setToolTip(
            "Optional candidate filter. 0 means no maximum projected area filter."
        )

        self.hole_fill_max_perimeter_spin = QDoubleSpinBox()
        self.hole_fill_max_perimeter_spin.setRange(0.0, 1_000_000_000.0)
        self.hole_fill_max_perimeter_spin.setDecimals(4)
        self.hole_fill_max_perimeter_spin.setSingleStep(0.1)
        self.hole_fill_max_perimeter_spin.setValue(0.0)
        self.hole_fill_max_perimeter_spin.setToolTip(
            "Optional candidate filter. 0 means no maximum perimeter filter."
        )

        self.hole_fill_preview_btn = QPushButton("Build Fill Preview")
        self.hole_fill_commit_btn = QPushButton("Commit Fill Preview")
        self.hole_fill_cancel_btn = QPushButton("Cancel Fill Preview")

        # Batch hole-fill controls are separate from single-candidate preview.
        # The GUI mixins already contain batch preview/commit handlers and
        # state helpers; these buttons make that existing flow reachable again
        # without changing viewport/selection behavior.
        self.hole_fill_batch_preview_btn = QPushButton("Batch Preview All")
        self.hole_fill_batch_commit_btn = QPushButton("Commit Batch Fill")

        self.hole_fill_preview_btn.setEnabled(False)
        self.hole_fill_commit_btn.setEnabled(False)
        self.hole_fill_cancel_btn.setEnabled(False)
        self.hole_fill_batch_preview_btn.setEnabled(False)
        self.hole_fill_batch_commit_btn.setEnabled(False)

        _connect_if_callable(self.hole_fill_method_combo.currentIndexChanged, "_on_hole_fill_method_changed")
        _connect_if_callable(self.hole_fill_preview_btn.clicked, "_on_hole_fill_preview_clicked")
        _connect_if_callable(self.hole_fill_commit_btn.clicked, "_on_hole_fill_commit_clicked")
        _connect_if_callable(self.hole_fill_cancel_btn.clicked, "_on_hole_fill_cancel_clicked")
        _connect_if_callable(self.hole_fill_batch_preview_btn.clicked, "_on_hole_fill_batch_preview_clicked")
        _connect_if_callable(self.hole_fill_batch_commit_btn.clicked, "_on_hole_fill_batch_commit_clicked")

        hole_fill_buttons_row = QWidget()
        hole_fill_buttons_layout = QHBoxLayout(hole_fill_buttons_row)
        hole_fill_buttons_layout.setContentsMargins(0, 0, 0, 0)
        hole_fill_buttons_layout.setSpacing(8)
        hole_fill_buttons_layout.addWidget(self.hole_fill_preview_btn)
        hole_fill_buttons_layout.addWidget(self.hole_fill_commit_btn)
        hole_fill_buttons_layout.addWidget(self.hole_fill_cancel_btn)

        hole_fill_batch_buttons_row = QWidget()
        hole_fill_batch_buttons_layout = QHBoxLayout(hole_fill_batch_buttons_row)
        hole_fill_batch_buttons_layout.setContentsMargins(0, 0, 0, 0)
        hole_fill_batch_buttons_layout.setSpacing(8)
        hole_fill_batch_buttons_layout.addWidget(self.hole_fill_batch_preview_btn)
        hole_fill_batch_buttons_layout.addWidget(self.hole_fill_batch_commit_btn)

        self.hole_fill_status_label = QLabel("No hole-fill preview active.")
        self.hole_fill_status_label.setWordWrap(True)
        self.hole_fill_status_label.setObjectName("SubtleLabel")

        hole_fill_form.addRow("Notes:", self.hole_fill_info_label)
        hole_fill_form.addRow("Candidate:", self.hole_fill_candidate_combo)
        hole_fill_form.addRow("Method:", self.hole_fill_method_combo)
        hole_fill_form.addRow("Max area:", self.hole_fill_max_area_spin)
        hole_fill_form.addRow("Max perimeter:", self.hole_fill_max_perimeter_spin)
        hole_fill_form.addRow("Preview:", hole_fill_buttons_row)
        hole_fill_form.addRow("Batch:", hole_fill_batch_buttons_row)
        hole_fill_form.addRow("Status:", self.hole_fill_status_label)

        topology_layout.addWidget(hole_fill_group)

        clip_group = QGroupBox("Clip / Section")
        clip_form = QFormLayout(clip_group)

        # Historical compatibility attribute names kept for current controller.
        self.viewer_clip_axis_combo = QComboBox()
        self.viewer_clip_axis_combo.addItem("X", "x")
        self.viewer_clip_axis_combo.addItem("Y", "y")
        self.viewer_clip_axis_combo.addItem("Z", "z")

        self.viewer_clip_fraction_spin = QDoubleSpinBox()
        self.viewer_clip_fraction_spin.setRange(0.0, 1.0)
        self.viewer_clip_fraction_spin.setSingleStep(0.05)
        self.viewer_clip_fraction_spin.setValue(0.50)

        self.viewer_clip_invert_check = QCheckBox("Invert clip side")

        self.viewer_apply_clip_btn = QPushButton("Apply Clip")
        self.viewer_apply_clip_btn.clicked.connect(self._apply_clip)

        self.viewer_clear_clip_btn = QPushButton("Clear Clip")
        self.viewer_clear_clip_btn.clicked.connect(self.viewport.clear_clip)

        clip_form.addRow("Axis:", self.viewer_clip_axis_combo)
        clip_form.addRow("Fraction:", self.viewer_clip_fraction_spin)
        clip_form.addRow("", self.viewer_clip_invert_check)
        clip_form.addRow("", self.viewer_apply_clip_btn)
        clip_form.addRow("", self.viewer_clear_clip_btn)

        manual_group = QGroupBox("Manual Edit Preview")
        manual_form = QFormLayout(manual_group)

        self.manual_edit_info_label = QLabel(
            "Build a preview from the current face or vertex selection. "
            "Face Cleanup and Face Reduce are automatically routed through the grouped patch-aware path in the controller. "
            "Delete Faces / Delete Vertices are direct whole-mesh commits. "
            "Clip and smoothing remain preview-first operations."
        )
        self.manual_edit_info_label.setWordWrap(True)
        self.manual_edit_info_label.setObjectName("SubtleLabel")

        self.manual_edit_operation_combo = QComboBox()
        self.manual_edit_operation_combo.addItem("Cleanup", "cleanup")
        self.manual_edit_operation_combo.addItem("Reduce", "reduce")
        self.manual_edit_operation_combo.addItem("Smooth Laplacian (ROI preview)", "smooth_laplacian")
        self.manual_edit_operation_combo.addItem("Delete Faces", "delete_faces")
        self.manual_edit_operation_combo.addItem("Delete Vertices", "delete_vertices")
        self.manual_edit_operation_combo.addItem("Clip ROI (preview)", "clip_plane")
        self.manual_edit_operation_combo.addItem("Group Cleanup", "group_cleanup")
        self.manual_edit_operation_combo.addItem("Group Reduce", "group_reduce")
        _connect_if_callable(self.manual_edit_operation_combo.currentIndexChanged, "_on_manual_edit_operation_changed")

        self.manual_edit_target_faces_spin = QSpinBox()
        self.manual_edit_target_faces_spin.setRange(1, 10_000_000)
        self.manual_edit_target_faces_spin.setValue(1000)

        self.manual_edit_boundary_weight_spin = QDoubleSpinBox()
        self.manual_edit_boundary_weight_spin.setRange(0.0, 100.0)
        self.manual_edit_boundary_weight_spin.setSingleStep(0.5)
        self.manual_edit_boundary_weight_spin.setValue(5.0)

        self.manual_edit_allow_non_manifold_check = QCheckBox("Allow destructive non-manifold edge cleanup")
        self.manual_edit_allow_non_manifold_check.setChecked(False)

        self.manual_edit_smooth_iters_spin = QSpinBox()
        self.manual_edit_smooth_iters_spin.setRange(1, 500)
        self.manual_edit_smooth_iters_spin.setValue(1)

        self.manual_edit_smooth_lambda_spin = QDoubleSpinBox()
        self.manual_edit_smooth_lambda_spin.setRange(0.0, 10.0)
        self.manual_edit_smooth_lambda_spin.setDecimals(3)
        self.manual_edit_smooth_lambda_spin.setSingleStep(0.1)
        self.manual_edit_smooth_lambda_spin.setValue(0.5)

        clip_point_row, _, _, _ = _make_vector3_row("manual_edit_clip_point", (0.0, 0.0, 0.0))
        clip_normal_row, _, _, _ = _make_vector3_row("manual_edit_clip_normal", (0.0, 0.0, 1.0))

        self.manual_edit_group_decode_combo = QComboBox()
        self.manual_edit_group_decode_combo.addItem("Auto", "auto")
        self.manual_edit_group_decode_combo.addItem("Face Colors", "face_colors")
        self.manual_edit_group_decode_combo.addItem("Material IDs", "material_ids")
        self.manual_edit_group_decode_combo.addItem("Texture Lookup", "texture_lookup")
        self.manual_edit_group_decode_combo.addItem("Synthetic Per Face", "synthetic_faces")

        texture_row_host = QWidget()
        texture_row = QHBoxLayout(texture_row_host)
        texture_row.setContentsMargins(0, 0, 0, 0)
        texture_row.setSpacing(6)

        self.manual_edit_texture_path_edit = QLineEdit()
        self.manual_edit_texture_path_edit.setPlaceholderText(
            "Optional texture path for texture_lookup decode mode"
        )

        self.manual_edit_texture_browse_btn = QPushButton("Browse…")
        self.manual_edit_texture_browse_btn.setMinimumWidth(92)
        _connect_if_callable(
            self.manual_edit_texture_browse_btn.clicked,
            "_on_manual_edit_browse_texture_clicked",
        )

        texture_row.addWidget(self.manual_edit_texture_path_edit, 1)
        texture_row.addWidget(self.manual_edit_texture_browse_btn)

        self.manual_edit_group_target_ratio_spin = QDoubleSpinBox()
        self.manual_edit_group_target_ratio_spin.setRange(0.01, 1.0)
        self.manual_edit_group_target_ratio_spin.setDecimals(3)
        self.manual_edit_group_target_ratio_spin.setSingleStep(0.05)
        self.manual_edit_group_target_ratio_spin.setValue(0.5)

        self.manual_edit_status_label = QLabel("No manual edit preview active.")
        self.manual_edit_status_label.setWordWrap(True)
        self.manual_edit_status_label.setObjectName("SubtleLabel")

        self.manual_edit_preview_btn = QPushButton("Build Preview")
        self.manual_edit_commit_btn = QPushButton("Commit Preview")
        self.manual_edit_cancel_btn = QPushButton("Cancel Preview")
        self.manual_edit_commit_btn.setEnabled(False)
        self.manual_edit_cancel_btn.setEnabled(False)

        _connect_if_callable(self.manual_edit_preview_btn.clicked, "_on_manual_edit_preview_clicked")
        _connect_if_callable(self.manual_edit_commit_btn.clicked, "_on_manual_edit_commit_clicked")
        _connect_if_callable(self.manual_edit_cancel_btn.clicked, "_on_manual_edit_cancel_clicked")

        manual_buttons_row = QWidget()
        manual_buttons_layout = QHBoxLayout(manual_buttons_row)
        manual_buttons_layout.setContentsMargins(0, 0, 0, 0)
        manual_buttons_layout.setSpacing(8)
        manual_buttons_layout.addWidget(self.manual_edit_preview_btn)
        manual_buttons_layout.addWidget(self.manual_edit_commit_btn)
        manual_buttons_layout.addWidget(self.manual_edit_cancel_btn)

        manual_form.addRow("Notes:", self.manual_edit_info_label)
        manual_form.addRow("Operation:", self.manual_edit_operation_combo)
        manual_form.addRow("Target faces:", self.manual_edit_target_faces_spin)
        manual_form.addRow("Boundary weight:", self.manual_edit_boundary_weight_spin)
        manual_form.addRow("Cleanup:", self.manual_edit_allow_non_manifold_check)
        manual_form.addRow("Smooth iterations:", self.manual_edit_smooth_iters_spin)
        manual_form.addRow("Smooth lambda:", self.manual_edit_smooth_lambda_spin)
        manual_form.addRow("Clip point:", clip_point_row)
        manual_form.addRow("Clip normal:", clip_normal_row)
        manual_form.addRow("Group decode:", self.manual_edit_group_decode_combo)
        manual_form.addRow("Texture path:", texture_row_host)
        manual_form.addRow("Group ratio:", self.manual_edit_group_target_ratio_spin)
        manual_form.addRow("Preview:", manual_buttons_row)
        manual_form.addRow("Status:", self.manual_edit_status_label)

        help_group = QGroupBox("How to Use")
        help_layout = QVBoxLayout(help_group)
        help_label = QLabel(
            "Brush / Tools is the active selection surface.\n\n"
            "Selection via mouse sets the viewport picking mode. "
            "Brush mode is a shortcut for face, point, and edge selection sessions. "
            "Only face and point modes support drag-paint selection; edge mode is click/Ctrl-click based for now.\n\n"
            "Face mode: click = select, Shift = add, Ctrl = toggle, Ctrl+click on an open boundary face = boundary region, "
            "drag = paint when brush is enabled.\n\n"
            "Point mode: click = select point, Shift = add, Ctrl = toggle, Alt+click = connected point island, "
            "drag = paint points when brush is enabled.\n\n"
            "Edge mode: click = select one edge, Shift = add one edge, Ctrl/Cmd+click = feature-aware continuous edge chain/loop. "
            "Right-click remains reserved for viewport navigation, not deselection.\n\n"
            "Cleanup and Reduce from face selections are intended to route through grouped patch-aware processing. "
            "If no QuadWild grouping metadata is present, the adapter can fall back to auto or synthetic grouping."
        )
        help_label.setWordWrap(True)
        help_layout.addWidget(help_label)

        layout.addWidget(self.brush_backend_note_label)
        layout.addWidget(interaction_group)
        layout.addWidget(actions_group)
        layout.addWidget(topology_group)
        layout.addWidget(clip_group)
        layout.addWidget(manual_group)
        layout.addWidget(help_group)
        layout.addStretch(1)
        return page


    def _build_bore_page(self) -> QWidget:
        """Bore page: select opening, list candidates, preview one, then rebuild."""

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)

        def _connect_if_callable(signal: Any, method_name: str) -> None:
            callback = getattr(self, method_name, None)
            if callable(callback):
                signal.connect(callback)

        intro_group = QGroupBox("Bore Opening Workflow")
        intro_layout = QVBoxLayout(intro_group)
        intro_layout.setContentsMargins(8, 8, 8, 8)

        self.bore_intro_label = QLabel(
            "Select one bore opening by Ctrl+clicking an edge on the bore rim. "
            "Then list measured Bore candidates, preview the intended object, "
            "and rebuild only the previewed candidate after the guarded commit gate passes."
        )
        self.bore_intro_label.setWordWrap(True)
        self.bore_intro_label.setObjectName("SubtleLabel")
        intro_layout.addWidget(self.bore_intro_label)

        mark_group = QGroupBox("1. Bore Opening Selection")
        mark_form = QFormLayout(mark_group)

        self.bore_marking_note_label = QLabel(
            "Use this single Bore selection mode, then Ctrl+click one edge on the "
            "bore opening. Manual brush selection and separate rim-completion UI "
            "are intentionally not part of this workflow."
        )
        self.bore_marking_note_label.setWordWrap(True)
        self.bore_marking_note_label.setObjectName("SubtleLabel")

        self.bore_select_opening_btn = QPushButton("Select Bore Opening")
        self.bore_select_opening_btn.setToolTip(
            "Activate Bore opening edge-pick mode. Ctrl+click one edge on the bore rim."
        )
        # Compatibility alias for older code paths that still look for the old name.
        self.bore_enable_edge_selection_btn = self.bore_select_opening_btn

        self.bore_focus_selection_btn = QPushButton("Focus Selection")
        self.bore_clear_selection_btn = QPushButton("Clear Selection")
        self.bore_boundary_highlight_check = QCheckBox("Highlight open boundaries while marking")
        self.bore_boundary_highlight_check.setChecked(True)

        _connect_if_callable(
            self.bore_select_opening_btn.clicked,
            "_on_bore_select_opening_clicked",
        )
        _connect_if_callable(
            self.bore_focus_selection_btn.clicked,
            "_on_bore_focus_selection_clicked",
        )
        _connect_if_callable(
            self.bore_clear_selection_btn.clicked,
            "_on_bore_clear_selection_clicked",
        )
        _connect_if_callable(
            self.bore_boundary_highlight_check.toggled,
            "_on_bore_boundary_highlight_toggled",
        )

        selection_buttons = QWidget()
        selection_buttons_layout = QHBoxLayout(selection_buttons)
        selection_buttons_layout.setContentsMargins(0, 0, 0, 0)
        selection_buttons_layout.setSpacing(8)
        selection_buttons_layout.addWidget(self.bore_focus_selection_btn)
        selection_buttons_layout.addWidget(self.bore_clear_selection_btn)

        self.bore_selected_edges_label = QLabel("Selected bore boundary edges: 0")
        self.bore_selected_edges_label.setObjectName("SubtleLabel")
        self.bore_selected_edges_label.setWordWrap(True)

        self.bore_opposite_rim_label = QLabel("Recognition boundary loops: -")
        self.bore_opposite_rim_label.setObjectName("SubtleLabel")
        self.bore_opposite_rim_label.setWordWrap(True)

        self.bore_selected_faces_label = QLabel("Selected faces: 0")
        self.bore_selected_faces_label.setObjectName("SubtleLabel")
        self.bore_selected_faces_label.setWordWrap(True)

        self.bore_boundary_status_label = QLabel(
            "Ctrl+click one edge on the bore opening."
        )
        self.bore_boundary_status_label.setObjectName("SubtleLabel")
        self.bore_boundary_status_label.setWordWrap(True)

        mark_form.addRow("Notes:", self.bore_marking_note_label)
        mark_form.addRow("Mode:", self.bore_select_opening_btn)
        mark_form.addRow("Open edges:", self.bore_boundary_highlight_check)
        mark_form.addRow("Selection:", selection_buttons)
        mark_form.addRow("Edges:", self.bore_selected_edges_label)
        mark_form.addRow("Recognition:", self.bore_opposite_rim_label)
        mark_form.addRow("Faces:", self.bore_selected_faces_label)
        mark_form.addRow("Status:", self.bore_boundary_status_label)

        action_group = QGroupBox("2. Candidate Discovery")
        action_form = QFormLayout(action_group)

        self.bore_select_wall_faces_btn = QPushButton("List Bore Candidates")
        self.bore_select_wall_faces_btn.setEnabled(False)
        self.bore_select_wall_faces_btn.setToolTip(
            "Collect a bounded marked-area AOI and send it to recognition to list measured feature candidates."
        )
        _connect_if_callable(
            self.bore_select_wall_faces_btn.clicked,
            "_on_bore_select_wall_faces_clicked",
        )

        self.bore_rebuild_wall_faces_btn = QPushButton("Delete + Rebuild Previewed Object")
        self.bore_rebuild_wall_faces_btn.setEnabled(False)
        self.bore_rebuild_wall_faces_btn.setToolTip(
            "Rebuild only the currently previewed Bore candidate after the core rebuild gates pass."
        )
        _connect_if_callable(
            self.bore_rebuild_wall_faces_btn.clicked,
            "_on_bore_rebuild_wall_faces_clicked",
        )

        action_buttons_row = QWidget()
        action_buttons_layout = QHBoxLayout(action_buttons_row)
        action_buttons_layout.setContentsMargins(0, 0, 0, 0)
        action_buttons_layout.setSpacing(8)
        action_buttons_layout.addWidget(self.bore_select_wall_faces_btn)

        action_form.addRow("Action:", action_buttons_row)

        candidate_group = QGroupBox("3. Preview / Rebuild Candidate")
        candidate_layout = QVBoxLayout(candidate_group)
        candidate_layout.setContentsMargins(8, 8, 8, 8)
        candidate_layout.setSpacing(8)

        self.bore_candidate_note_label = QLabel(
            "After List Bore Candidates, measured Borehole-family objects appear here. "
            "Select one candidate, preview it, then rebuild the previewed object only. "
            "Mouth is shown as seam/transition evidence unless it owns separate measured faces."
        )
        self.bore_candidate_note_label.setWordWrap(True)
        self.bore_candidate_note_label.setObjectName("SubtleLabel")

        self.bore_feature_candidate_tree = QTreeWidget()
        self.bore_feature_candidate_tree.setColumnCount(4)
        self.bore_feature_candidate_tree.setHeaderLabels(["Object", "Faces", "Geometry", "Role"] )
        self.bore_feature_candidate_tree.setRootIsDecorated(False)
        self.bore_feature_candidate_tree.setAlternatingRowColors(True)
        self.bore_feature_candidate_tree.setMinimumHeight(118)
        self.bore_feature_candidate_tree.setMaximumHeight(170)
        self.bore_feature_candidate_tree.setEnabled(False)
        self.bore_feature_candidate_tree.setToolTip(
            "Measured Bore feature objects. Select one object, then Preview Candidate."
        )

        # Compact combo kept as a compatibility alias for older handlers and narrow panels.
        self.bore_feature_candidate_combo = QComboBox()
        self.bore_feature_candidate_combo.addItem("No Bore feature candidates detected", None)
        self.bore_feature_candidate_combo.setEnabled(False)
        self.bore_feature_candidate_combo.setToolTip(
            "Measured Bore feature candidates. The tree above is the primary candidate list."
        )

        self.bore_quad_density_combo = QComboBox()
        self.bore_quad_density_combo.addItem("Lean / Low — 882 quads test style", "lean")
        self.bore_quad_density_combo.addItem("Smooth / Balanced — 3402 quads test style", "pi")
        self.bore_quad_density_combo.addItem("Full / Original — pre-density-change", "full")
        self.bore_quad_density_combo.setCurrentIndex(0)
        self.bore_quad_density_combo.setToolTip(
            "Bore rebuild quad density. "
            "Lean uses the latest low-density pi/opening rule. "
            "Smooth/Balanced uses the first pi/opening rule. "
            "Full/Original uses the initial equal-edge dense rebuild."
        )

        self.bore_preview_feature_candidate_btn = QPushButton("Preview Selected Object")
        self.bore_preview_feature_candidate_btn.setEnabled(False)
        self.bore_preview_feature_candidate_btn.setToolTip(
            "Highlight the selected measured feature object without committing a mesh change."
        )

        self.bore_reset_candidate_preview_btn = QPushButton("Preview Full Bore Selection")
        self.bore_reset_candidate_preview_btn.setEnabled(False)
        self.bore_reset_candidate_preview_btn.setToolTip(
            "Restore the full wall/feature preview returned by Select Bore Wall Faces."
        )

        _connect_if_callable(
            self.bore_feature_candidate_tree.currentItemChanged,
            "_on_bore_feature_candidate_tree_changed",
        )
        _connect_if_callable(
            self.bore_feature_candidate_combo.currentIndexChanged,
            "_on_bore_feature_candidate_changed",
        )
        _connect_if_callable(
            self.bore_preview_feature_candidate_btn.clicked,
            "_on_bore_preview_feature_candidate_clicked",
        )
        _connect_if_callable(
            self.bore_reset_candidate_preview_btn.clicked,
            "_on_bore_reset_feature_candidate_preview_clicked",
        )

        candidate_buttons_row = QWidget()
        candidate_buttons_layout = QHBoxLayout(candidate_buttons_row)
        candidate_buttons_layout.setContentsMargins(0, 0, 0, 0)
        candidate_buttons_layout.setSpacing(8)
        candidate_buttons_layout.addWidget(self.bore_preview_feature_candidate_btn)
        candidate_buttons_layout.addWidget(self.bore_reset_candidate_preview_btn)
        candidate_buttons_layout.addWidget(self.bore_rebuild_wall_faces_btn)

        self.bore_feature_candidate_status_label = QLabel("No Bore feature candidates yet.")
        self.bore_feature_candidate_status_label.setObjectName("SubtleLabel")
        self.bore_feature_candidate_status_label.setWordWrap(True)

        candidate_layout.addWidget(self.bore_candidate_note_label)
        candidate_layout.addWidget(self.bore_feature_candidate_tree)
        candidate_layout.addWidget(self.bore_feature_candidate_combo)

        density_row = QWidget()
        density_layout = QHBoxLayout(density_row)
        density_layout.setContentsMargins(0, 0, 0, 0)
        density_layout.setSpacing(8)
        density_label = QLabel("Rebuild density:")
        density_label.setObjectName("SubtleLabel")
        density_layout.addWidget(density_label)
        density_layout.addWidget(self.bore_quad_density_combo, 1)
        candidate_layout.addWidget(density_row)

        candidate_layout.addWidget(candidate_buttons_row)
        candidate_layout.addWidget(self.bore_feature_candidate_status_label)

        analysis_group = QGroupBox("4. Result / Diagnostics")
        analysis_layout = QVBoxLayout(analysis_group)

        self.bore_analysis_text = QPlainTextEdit()
        self.bore_analysis_text.setReadOnly(True)
        self.bore_analysis_text.setMinimumHeight(150)
        self.bore_analysis_text.setPlaceholderText("Bore diagnostics will appear here.")
        analysis_layout.addWidget(self.bore_analysis_text)

        preview_group = QGroupBox("5. Selection Summary")
        preview_layout = QVBoxLayout(preview_group)

        self.bore_preview_status_label = QLabel("No Bore opening selected.")
        self.bore_preview_status_label.setObjectName("SubtleLabel")
        self.bore_preview_status_label.setWordWrap(True)

        self.bore_preview_text = QPlainTextEdit()
        self.bore_preview_text.setReadOnly(True)
        self.bore_preview_text.setMinimumHeight(120)
        self.bore_preview_text.setPlaceholderText("Selection count and diagnostics will appear here.")

        preview_layout.addWidget(self.bore_preview_status_label)
        preview_layout.addWidget(self.bore_preview_text)

        layout.addWidget(intro_group)
        layout.addWidget(mark_group)
        layout.addWidget(action_group)
        layout.addWidget(candidate_group)
        layout.addWidget(analysis_group)
        layout.addWidget(preview_group)
        layout.addStretch(1)
        return page

    def _build_log_panel(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("LogFrame")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Log")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(3000)
        layout.addWidget(self.log_output, 1)

        return frame

    def _add_page(self, key: str, title: str, widget: QWidget) -> None:
        button_map = {
            self.PAGE_LOAD: self.load_btn,
            self.PAGE_REPAIR: self.repair_btn,
            self.PAGE_REMESH: self.remesh_btn,
            self.PAGE_REDUCE: self.reduce_btn,
            self.PAGE_VIEWER: self.viewer_btn,
            self.PAGE_BRUSH: self.brush_btn,
            self.PAGE_BORE: self.bore_btn,
        }
        btn = button_map[key]
        self.tool_stack.addWidget(widget)
        self._pages[key] = ToolPage(key=key, title=title, button=btn, widget=widget)

    def _show_page(self, key: str) -> None:
        page = self._pages[key]
        self.tool_stack.setCurrentWidget(page.widget)
        self.tool_title_label.setText(page.title)
        for item in self._pages.values():
            item.button.setChecked(item.key == key)

        callback = getattr(self, "_on_page_shown", None)
        if callable(callback):
            callback(key)

    # ------------------------------------------------------------------
    # Repair UI helpers
    # ------------------------------------------------------------------
    def _on_repair_method_ui_changed(self) -> None:
        method = self.repair_method_combo.currentData() if hasattr(self, "repair_method_combo") else None
        self._update_repair_advanced_ui_for_method(method)

        callback = getattr(self, "_on_repair_method_changed", None)
        if callable(callback):
            callback()

    def _on_repair_advanced_group_toggled(self, checked: bool) -> None:
        if not hasattr(self, "repair_advanced_content"):
            return
        method = self.repair_method_combo.currentData() if hasattr(self, "repair_method_combo") else None
        supports_advanced = bool(method in REPAIR_ADVANCED_METHODS)
        self.repair_advanced_content.setEnabled(bool(checked and supports_advanced))
        self._sync_repair_tvertex_ui()

    def _update_repair_advanced_ui_for_method(self, method: str | None) -> None:
        if not hasattr(self, "repair_advanced_group"):
            return

        supports_advanced = bool(method in REPAIR_ADVANCED_METHODS)
        strict_preserve = bool(method in REPAIR_STRICT_PRESERVE_METHODS)

        self.repair_advanced_group.setEnabled(supports_advanced)

        if supports_advanced:
            if not self.repair_advanced_group.isChecked():
                self.repair_advanced_group.setChecked(False)
        else:
            self.repair_advanced_group.setChecked(False)

        self.repair_preserve_features_check.blockSignals(True)
        self.repair_preserve_features_check.setChecked(strict_preserve)
        self.repair_preserve_features_check.setEnabled(supports_advanced and not strict_preserve)
        self.repair_preserve_features_check.blockSignals(False)

        self.repair_edge_method_combo.setEnabled(supports_advanced)
        self.repair_vertex_drift_spin.setEnabled(supports_advanced)

        self._on_repair_advanced_group_toggled(self.repair_advanced_group.isChecked())

    def _sync_repair_tvertex_ui(self) -> None:
        if not hasattr(self, "repair_tvertex_enable_check"):
            return

        method = self.repair_method_combo.currentData() if hasattr(self, "repair_method_combo") else None
        supports_advanced = bool(method in REPAIR_ADVANCED_METHODS)
        strict_preserve = bool(method in REPAIR_STRICT_PRESERVE_METHODS)

        group_active = bool(
            supports_advanced
            and hasattr(self, "repair_advanced_group")
            and self.repair_advanced_group.isChecked()
        )

        preserve_requested = bool(self.repair_preserve_features_check.isChecked())
        preserve_mode = strict_preserve or preserve_requested

        self.repair_tvertex_enable_check.setEnabled(group_active and not preserve_mode)

        tvertex_active = group_active and not preserve_mode and self.repair_tvertex_enable_check.isChecked()
        self.repair_tvertex_method_combo.setEnabled(tvertex_active)
        self.repair_tvertex_threshold_spin.setEnabled(tvertex_active)
        self.repair_tvertex_repeat_check.setEnabled(tvertex_active)

    # ------------------------------------------------------------------
    # Styles / generic helpers
    # ------------------------------------------------------------------
    def _apply_styles(self) -> None:
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: {FAR_DEEP_BLACK};
                color: {FAR_TEXT};
            }}
            QMenuBar, QMenu, QStatusBar {{
                background: {FAR_PANEL_BLACK};
                color: {FAR_TEXT};
                border: none;
            }}
            QMenuBar::item:selected, QMenu::item:selected {{
                background: {FAR_ELECTRIC_BLUE};
                color: #FFFFFF;
            }}
            QStatusBar {{
                border-top: 1px solid {FAR_BORDER};
            }}
            QFrame#HeaderFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {FAR_PANEL_BLUE_BLACK},
                    stop:0.55 {FAR_PANEL_BLACK},
                    stop:1 #10151D);
                border: 1px solid {FAR_BORDER};
                border-radius: 12px;
            }}
            QFrame#RailFrame, QFrame#ToolFrame, QFrame#ViewportFrame, QFrame#LogFrame {{
                background: {FAR_PANEL_BLUE_BLACK};
                border: 1px solid {FAR_BORDER};
                border-radius: 12px;
            }}
            QFrame#LoadBrandFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0D1724,
                    stop:0.50 #111B28,
                    stop:1 #151015);
                border: 1px solid {FAR_ELECTRIC_BLUE};
                border-radius: 14px;
            }}
            QLabel#HeaderIconLabel {{
                background: rgba(0, 230, 255, 0.06);
                border: 1px solid rgba(0, 230, 255, 0.35);
                border-radius: 10px;
            }}
            QLabel#LoadLogoLabel {{
                color: {FAR_CYAN_GLOW};
                font-size: 24px;
                font-weight: 800;
                letter-spacing: 1px;
            }}
            QLabel#AppTitle {{
                color: {FAR_TEXT};
                font-size: 23px;
                font-weight: 800;
                letter-spacing: 1px;
            }}
            QLabel#AppSubtitle {{
                color: {FAR_CYAN_GLOW};
                font-size: 12px;
                font-weight: 600;
            }}
            QLabel#SectionTitle {{
                color: {FAR_TEXT};
                font-size: 16px;
                font-weight: 700;
            }}
            QLabel#SubtleLabel, QLabel#CurrentFileLabel {{
                color: {FAR_TEXT_MUTED};
            }}
            QLabel#CurrentFileLabel {{
                padding: 6px 8px;
                background: rgba(0, 136, 255, 0.08);
                border: 1px solid rgba(0, 136, 255, 0.25);
                border-radius: 8px;
            }}
            QToolButton {{
                background: #17202D;
                color: {FAR_TEXT};
                border: 1px solid {FAR_BORDER};
                border-radius: 9px;
                padding: 8px 10px;
            }}
            QToolButton:hover {{
                background: #1E2C3D;
                border-color: {FAR_CYAN_GLOW};
            }}
            QToolButton:checked {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {FAR_ELECTRIC_BLUE},
                    stop:1 #005FB3);
                border-color: {FAR_CYAN_GLOW};
                color: #FFFFFF;
            }}
            QToolButton#HeaderActionButton {{
                padding: 6px 12px;
                background: rgba(0, 136, 255, 0.12);
                border-color: rgba(0, 230, 255, 0.35);
            }}
            QPushButton {{
                background: #1D2938;
                color: {FAR_TEXT};
                border: 1px solid #35506C;
                border-radius: 9px;
                padding: 8px 10px;
            }}
            QPushButton:hover {{
                background: #24364A;
                border-color: {FAR_CYAN_GLOW};
            }}
            QPushButton:pressed {{
                background: #112339;
                border-color: {FAR_ELECTRIC_BLUE};
            }}
            QPushButton:disabled, QToolButton:disabled {{
                color: #627082;
                background: #121922;
                border-color: #26313D;
            }}
            QGroupBox {{
                color: {FAR_TEXT};
                background: rgba(18, 23, 32, 0.72);
                border: 1px solid {FAR_BORDER};
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 700;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 7px;
                color: {FAR_CYAN_GLOW};
                background: {FAR_PANEL_BLUE_BLACK};
            }}
            QPlainTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QScrollArea {{
                background: #0F141B;
                color: {FAR_TEXT};
                border: 1px solid #2D4258;
                border-radius: 7px;
                padding: 6px;
                selection-background-color: {FAR_ELECTRIC_BLUE};
            }}
            QPlainTextEdit#ProjectStatusWarnings {{
                background: rgba(0, 136, 255, 0.04);
                border-color: rgba(0, 230, 255, 0.20);
            }}
            QPlainTextEdit#ProjectStatusWarnings[restoreSeverity="ok"] {{
                background: rgba(0, 136, 255, 0.04);
                border-color: rgba(0, 230, 255, 0.20);
            }}
            QPlainTextEdit#ProjectStatusWarnings[restoreSeverity="warning"] {{
                background: rgba(255, 176, 0, 0.06);
                border-color: rgba(255, 176, 0, 0.42);
            }}
            QPlainTextEdit#ProjectStatusWarnings[restoreSeverity="error"] {{
                background: rgba(255, 64, 64, 0.08);
                border-color: rgba(255, 64, 64, 0.58);
            }}
            QComboBox::drop-down {{
                border: none;
                width: 22px;
            }}
            QCheckBox {{
                color: {FAR_TEXT};
                spacing: 8px;
            }}
            QProgressBar {{
                background: #0F141B;
                border: 1px solid {FAR_BORDER};
                border-radius: 6px;
                min-height: 10px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {FAR_ELECTRIC_BLUE},
                    stop:1 {FAR_CYAN_GLOW});
                border-radius: 5px;
            }}
            """
        )

    def _populate_choice_combo(
        self,
        combo: QComboBox,
        available_values: Any,
        labels: dict[str, str],
        preferred_order: list[str],
    ) -> None:
        combo.clear()
        available = set(str(v) for v in available_values)
        ordered = [v for v in preferred_order if v in available]
        ordered.extend(v for v in sorted(available) if v not in ordered)
        for value in ordered:
            combo.addItem(labels.get(value, value.replace("_", " ").title()), value)

    def _set_combo_item_enabled(self, combo: QComboBox, data_value: str, enabled: bool) -> None:
        model = combo.model()
        if model is None:
            return
        for row in range(combo.count()):
            if combo.itemData(row) == data_value:
                item = model.item(row) if hasattr(model, "item") else None
                if item is not None:
                    item.setEnabled(enabled)
                return

    def _set_combo_item_tooltip(self, combo: QComboBox, data_value: str, tooltip: str) -> None:
        model = combo.model()
        if model is None:
            return
        for row in range(combo.count()):
            if combo.itemData(row) == data_value:
                item = model.item(row) if hasattr(model, "item") else None
                if item is not None:
                    item.setData(tooltip, Qt.ItemDataRole.ToolTipRole)
                return

    def _set_combo_current_data(self, combo: QComboBox, value: str | None) -> None:
        if value is None:
            return
        idx = combo.findData(value)
        if idx < 0:
            return
        blocker = QSignalBlocker(combo)
        combo.setCurrentIndex(idx)
        del blocker


__all__ = [
    "MainWindowUI",
    "ToolPage",
    "CAD_PRESETS",
    "DISPLAY_PRESET_LABELS",
    "COMPARE_MODE_LABELS",
    "SELECTION_MODE_LABELS",
    "CAMERA_PRESET_LABELS",
    "REPAIR_ADVANCED_METHODS",
    "REPAIR_STRICT_PRESERVE_METHODS",
    "GUI_ASSETS_DIR",
    "FAR_MESH_APP_ICON",
    "FAR_MESH_LOGO",
    "FAR_MESH_DESIGN_REFERENCE",
    "FAR_DEEP_BLACK",
    "FAR_PANEL_BLACK",
    "FAR_PANEL_BLUE_BLACK",
    "FAR_PANEL_SOFT",
    "FAR_BORDER",
    "FAR_ELECTRIC_BLUE",
    "FAR_CYAN_GLOW",
    "FAR_SIGNAL_ORANGE",
    "FAR_WARM_AMBER",
    "FAR_TEXT",
    "FAR_TEXT_MUTED",
]
