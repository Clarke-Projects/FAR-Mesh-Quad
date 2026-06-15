from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from PySide6.QtCore import QObject, Signal

from .selection_state import (
    InteractionStyle,
    SelectionFilters,
    SelectionMode,
    SelectionOp,
    SelectionSession,
    SelectionSnapshot,
    SelectionState,
    apply_selection_op,
    sanitize_ids,
)


class SelectionController(QObject):
    """
    Controller-owned selection and brush-state bridge.

    Responsibilities
    ----------------
    - own intended interaction/session state
    - track actual selection state coming back from the viewport
    - push selection mode / brush mode into the viewport in a safe order
    - provide one shared place for MainWindow and future tools to query selection

    Architectural rule
    ------------------
    - session = intended state owned by the controller
    - state   = actual current selection snapshot, usually synced from viewport

    Important migration rule
    ------------------------
    This controller is the only place that should translate between:
        viewport raw modes: "none", "point", "face", "edge", "mesh"
        semantic modes:     SelectionMode.NONE / VERTEX / FACE / EDGE / OBJECT
    """

    state_changed = Signal(object)            # SelectionState
    session_changed = Signal(object)          # SelectionSession
    snapshot_emitted = Signal(object)         # SelectionSnapshot
    mode_changed = Signal(str)                # SelectionMode.value
    brush_enabled_changed = Signal(bool)
    interaction_style_changed = Signal(str)   # InteractionStyle.value

    _VIEWPORT_TO_MODE: dict[str, SelectionMode] = {
        "none": SelectionMode.NONE,
        "point": SelectionMode.VERTEX,
        "vertex": SelectionMode.VERTEX,
        "vertices": SelectionMode.VERTEX,
        "face": SelectionMode.FACE,
        "faces": SelectionMode.FACE,
        "edge": SelectionMode.EDGE,
        "edges": SelectionMode.EDGE,
        "mesh": SelectionMode.OBJECT,
        "object": SelectionMode.OBJECT,
        "objects": SelectionMode.OBJECT,
    }

    _MODE_TO_VIEWPORT: dict[SelectionMode, str] = {
        SelectionMode.NONE: "none",
        SelectionMode.VERTEX: "point",
        SelectionMode.FACE: "face",
        SelectionMode.EDGE: "edge",
        SelectionMode.OBJECT: "mesh",
    }

    def __init__(
        self,
        viewport: Any | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._viewport: Any | None = None
        self._state = SelectionState()
        self._session = SelectionSession()
        self._edge_region_strategy = "safe"
        self._signal_connections_ready = False

        if viewport is not None:
            self.bind_viewport(viewport)

    # ------------------------------------------------------------------
    # basic properties
    # ------------------------------------------------------------------
    @property
    def viewport(self) -> Any | None:
        return self._viewport

    @property
    def state(self) -> SelectionState:
        return self._state

    @property
    def session(self) -> SelectionSession:
        return self._session

    def snapshot(self, reason: str = "") -> SelectionSnapshot:
        return SelectionSnapshot(
            state=self._state,
            session=self._session,
            reason=str(reason or ""),
        )

    # ------------------------------------------------------------------
    # raw <-> semantic mode translation
    # ------------------------------------------------------------------
    @classmethod
    def semantic_mode_from_viewport(cls, raw_mode: SelectionMode | str | None) -> SelectionMode:
        if isinstance(raw_mode, SelectionMode):
            return raw_mode
        if raw_mode is None:
            return SelectionMode.NONE
        normalized = str(raw_mode).strip().lower()
        return cls._VIEWPORT_TO_MODE.get(normalized, SelectionMode.NONE)

    @classmethod
    def viewport_mode_from_semantic(cls, mode: SelectionMode | str | None) -> str:
        if isinstance(mode, SelectionMode):
            return cls._MODE_TO_VIEWPORT.get(mode, "none")
        if mode is None:
            return "none"

        normalized = str(mode).strip().lower()

        # Accept both semantic enum values and raw viewport strings.
        try:
            enum_mode = SelectionMode(normalized)
            return cls._MODE_TO_VIEWPORT.get(enum_mode, "none")
        except Exception:
            pass

        coerced = cls._VIEWPORT_TO_MODE.get(normalized, SelectionMode.NONE)
        return cls._MODE_TO_VIEWPORT.get(coerced, "none")

    def current_viewport_mode(self) -> str:
        return self.viewport_mode_from_semantic(self._session.mode)

    def current_edge_region_strategy(self) -> str:
        """Return the generic core edge-region strategy for Ctrl/Cmd edge expansion."""

        return str(getattr(self, "_edge_region_strategy", "safe") or "safe")

    def set_edge_region_strategy(
        self,
        strategy: str | None,
        *,
        push: bool = True,
        reason: str = "set_edge_region_strategy",
    ) -> None:
        """Store and push the generic core edge-region strategy.

        This is deliberately not Bore logic.  It is only selection intent:
        normal edge tools use ``safe``; Bore tools can request ``bore_rim``.
        The viewport only forwards this string to core.selection_edges.
        """

        value = str(strategy or "safe").strip().lower()
        if value not in {"safe", "open_component", "feature", "ring", "aggressive", "single", "bore_rim"}:
            value = "safe"
        self._edge_region_strategy = value
        if push:
            self._push_edge_region_strategy_to_viewport()
        self.snapshot_emitted.emit(self.snapshot(reason))

    # ------------------------------------------------------------------
    # tool-session helpers
    # ------------------------------------------------------------------
    def prepare_tool_edge_selection(
        self,
        *,
        tool: str,
        region_strategy: str = "safe",
        preserve_existing: bool = True,
        reason: str = "prepare_tool_edge_selection",
    ) -> tuple[int, ...]:
        """Prepare a host-owned edge-pick session for a tool.

        This method is intentionally generic host selection lifecycle, not Bore
        recognition logic.  It syncs current viewport state, stores the requested
        edge-region strategy, switches semantic selection to EDGE / viewer mode,
        disables brush painting, and returns any existing edge IDs when the caller
        asked to preserve them.

        Tools such as BoreTool should call this instead of reaching directly into
        raw viewport selection mode and edge-region strategy methods.
        """

        self.sync_from_viewport(reason=reason)
        existing_edges = tuple(int(v) for v in self._state.selected_edge_ids)

        self.set_edge_region_strategy(
            region_strategy,
            push=False,
            reason=f"{reason}:{tool}:edge_region_strategy",
        )

        self._session = (
            self._session
            .with_mode(SelectionMode.EDGE)
            .with_style(InteractionStyle.VIEWER)
            .with_brush_enabled(False)
        )

        self._state = replace(
            self._state,
            mode=SelectionMode.EDGE,
            selected_vertex_ids=(),
            selected_face_ids=(),
            selected_edge_ids=existing_edges if preserve_existing else (),
            selected_mesh_ids=(),
            interaction_style=InteractionStyle.VIEWER,
            brush_enabled=False,
            revision=self._state.revision + 1,
        )

        self._emit_session_changed(reason)
        self._emit_state_changed(reason)
        self.mode_changed.emit(SelectionMode.EDGE.value)
        self.brush_enabled_changed.emit(False)
        self.interaction_style_changed.emit(InteractionStyle.VIEWER.value)

        self.push_session_to_viewport(reason=reason)
        return self._state.selected_edge_ids

    def selected_edge_ids_snapshot(
        self,
        *,
        reason: str = "selected_edge_ids_snapshot",
    ) -> tuple[int, ...]:
        """Return a stable normalized edge-ID snapshot for tool input."""

        self.sync_from_viewport(reason=reason)
        self.snapshot_emitted.emit(self.snapshot(reason))
        return tuple(int(v) for v in self._state.selected_edge_ids)

    def clear_after_mesh_replacement(
        self,
        *,
        reason: str = "clear_after_mesh_replacement",
        edge_region_strategy: str = "safe",
        reapply_session: bool = True,
    ) -> None:
        """Clear stale selection IDs after the active mesh was replaced.

        Mesh replacement invalidates raw face/edge IDs and viewport pick caches.
        This host-owned method centralizes that cleanup so tools do not need to
        call backend-specific viewport cleanup functions directly.
        """

        self.set_edge_region_strategy(
            edge_region_strategy,
            push=False,
            reason=f"{reason}:edge_region_strategy",
        )

        self._state = self._state.cleared(keep_mode=True, revision_delta=1)
        self._emit_state_changed(reason)

        self._clear_selection_in_viewport()
        if reapply_session:
            self.push_session_to_viewport(reason=reason)

    def rearm_tool_edge_selection(
        self,
        *,
        tool: str,
        region_strategy: str = "safe",
        reason: str = "rearm_tool_edge_selection",
    ) -> None:
        """Re-enter semantic EDGE selection after a tool-owned mesh action."""

        self.set_edge_region_strategy(
            region_strategy,
            push=False,
            reason=f"{reason}:{tool}:edge_region_strategy",
        )
        self.set_mode(
            SelectionMode.EDGE,
            clear_other_domains=True,
            push=True,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # viewport binding
    # ------------------------------------------------------------------
    def bind_viewport(self, viewport: Any) -> None:
        """
        Bind a viewport implementing the current FAR MESH viewport protocol surface.

        Expected optional signals:
        - selection_changed
        - point_picked
        - mesh_loaded
        """
        self._viewport = viewport
        self._signal_connections_ready = False
        self._connect_viewport_signals()
        self.sync_from_viewport(reason="bind_viewport")
        self.push_session_to_viewport(reason="bind_viewport")

    def _connect_viewport_signals(self) -> None:
        if self._viewport is None or self._signal_connections_ready:
            return

        self._try_connect_signal(self._viewport, "selection_changed", self._on_viewport_selection_changed)
        self._try_connect_signal(self._viewport, "point_picked", self._on_viewport_point_picked)
        self._try_connect_signal(self._viewport, "mesh_loaded", self._on_viewport_mesh_loaded)

        self._signal_connections_ready = True

    @staticmethod
    def _try_connect_signal(obj: Any, signal_name: str, callback: Callable[..., None]) -> None:
        try:
            signal = getattr(obj, signal_name)
        except Exception:
            return
        try:
            signal.connect(callback)
        except Exception:
            return

    # ------------------------------------------------------------------
    # session mutation
    # ------------------------------------------------------------------
    def set_filters(self, filters: SelectionFilters) -> None:
        self._session = replace(self._session, filters=filters)
        self._emit_session_changed("set_filters")

    def set_mode(
        self,
        mode: SelectionMode | str | None,
        *,
        clear_other_domains: bool = True,
        push: bool = True,
        reason: str = "set_mode",
    ) -> None:
        mode_enum = self._coerce_mode(mode)

        if not self._session.filters.allows_mode(mode_enum):
            return

        self._session = self._session.with_mode(mode_enum)
        self._state = self._state.with_mode(
            mode_enum,
            clear_other_domains=clear_other_domains,
            revision_delta=1,
        )

        self._emit_session_changed(reason)
        self._emit_state_changed(reason)
        self.mode_changed.emit(mode_enum.value)

        if push:
            self.push_session_to_viewport(reason=reason)

    def set_interaction_style(
        self,
        style: InteractionStyle | str | None,
        *,
        push: bool = True,
        reason: str = "set_interaction_style",
    ) -> None:
        style_enum = self._coerce_style(style)
        self._session = self._session.with_style(style_enum)
        self._state = replace(
            self._state,
            interaction_style=style_enum,
            revision=self._state.revision + 1,
        )

        self._emit_session_changed(reason)
        self._emit_state_changed(reason)
        self.interaction_style_changed.emit(style_enum.value)

        if push:
            self.push_session_to_viewport(reason=reason)

    def set_brush_enabled(
        self,
        enabled: bool,
        *,
        push: bool = True,
        reason: str = "set_brush_enabled",
    ) -> None:
        enabled_bool = bool(enabled)

        # Brush interaction only makes sense for face / vertex selection.
        if enabled_bool and self._session.mode not in {SelectionMode.FACE, SelectionMode.VERTEX}:
            self._session = self._session.with_mode(SelectionMode.FACE)
            self._state = self._state.with_mode(
                SelectionMode.FACE,
                clear_other_domains=True,
                revision_delta=1,
            )

        style = InteractionStyle.BRUSH if enabled_bool else InteractionStyle.VIEWER
        self._session = self._session.with_brush_enabled(enabled_bool).with_style(style)
        self._state = replace(
            self._state,
            brush_enabled=enabled_bool,
            interaction_style=style,
            revision=self._state.revision + 1,
        )

        self._emit_session_changed(reason)
        self._emit_state_changed(reason)
        self.brush_enabled_changed.emit(enabled_bool)
        self.interaction_style_changed.emit(style.value)

        if push:
            self.push_session_to_viewport(reason=reason)

    def set_boundary_highlight(
        self,
        enabled: bool,
        *,
        push: bool = True,
        reason: str = "set_boundary_highlight",
    ) -> None:
        enabled_bool = bool(enabled)
        self._session = self._session.with_boundary_highlight(enabled_bool)
        self._state = replace(
            self._state,
            boundary_highlight=enabled_bool,
            revision=self._state.revision + 1,
        )

        self._emit_session_changed(reason)
        self._emit_state_changed(reason)

        if push:
            self._push_boundary_highlight_to_viewport()
            self.snapshot_emitted.emit(self.snapshot(reason))

    def apply_viewer_mode(
        self,
        mode: SelectionMode | str | None,
        *,
        reason: str = "apply_viewer_mode",
    ) -> None:
        """
        Explicit non-brush selection mode from the Viewer / Selection-via-mouse surface.
        """
        mode_enum = self._coerce_mode(mode)

        self._session = (
            self._session
            .with_mode(mode_enum)
            .with_style(InteractionStyle.VIEWER)
            .with_brush_enabled(False)
        )

        self._state = self._state.with_mode(
            mode_enum,
            clear_other_domains=False,
            revision_delta=1,
        )
        self._state = replace(
            self._state,
            interaction_style=InteractionStyle.VIEWER,
            brush_enabled=False,
            revision=self._state.revision + 1,
        )

        self._emit_session_changed(reason)
        self._emit_state_changed(reason)
        self.mode_changed.emit(mode_enum.value)
        self.brush_enabled_changed.emit(False)
        self.interaction_style_changed.emit(InteractionStyle.VIEWER.value)

        self.push_session_to_viewport(reason=reason)

    def apply_brush_mode(
        self,
        mode: SelectionMode | str | None,
        *,
        enabled: bool = True,
        reason: str = "apply_brush_mode",
    ) -> None:
        """
        Brush-page entry point.
        """
        mode_enum = self._coerce_mode(mode)
        if mode_enum not in {SelectionMode.FACE, SelectionMode.VERTEX, SelectionMode.EDGE}:
            mode_enum = SelectionMode.FACE

        self._session = (
            self._session
            .with_mode(mode_enum)
            .with_style(InteractionStyle.BRUSH)
            .with_brush_enabled(bool(enabled))
        )

        self._state = self._state.with_mode(
            mode_enum,
            clear_other_domains=False,
            revision_delta=1,
        )
        self._state = replace(
            self._state,
            interaction_style=InteractionStyle.BRUSH,
            brush_enabled=bool(enabled),
            revision=self._state.revision + 1,
        )

        self._emit_session_changed(reason)
        self._emit_state_changed(reason)
        self.mode_changed.emit(mode_enum.value)
        self.brush_enabled_changed.emit(bool(enabled))
        self.interaction_style_changed.emit(InteractionStyle.BRUSH.value)

        self.push_session_to_viewport(reason=reason)

    # ------------------------------------------------------------------
    # selection mutation helpers
    # ------------------------------------------------------------------
    def clear_selection(
        self,
        *,
        keep_mode: bool = True,
        push: bool = True,
        reason: str = "clear_selection",
    ) -> None:
        self._state = self._state.cleared(
            keep_mode=keep_mode,
            revision_delta=1,
        )

        if not keep_mode:
            self._session = self._session.with_mode(SelectionMode.NONE)

        self._emit_state_changed(reason)
        if not keep_mode:
            self._emit_session_changed(reason)
            self.mode_changed.emit(self._session.mode.value)

        if push:
            self._clear_selection_in_viewport()
            # Reassert current interaction intent after clearing.
            self.push_session_to_viewport(reason=reason)

    def replace_face_selection(
        self,
        face_ids: list[int] | tuple[int, ...],
        *,
        push: bool = False,
        reason: str = "replace_face_selection",
    ) -> None:
        self.apply_face_selection(face_ids, op=SelectionOp.REPLACE, push=push, reason=reason)

    def apply_face_selection(
        self,
        face_ids: list[int] | tuple[int, ...],
        *,
        op: SelectionOp | str = SelectionOp.REPLACE,
        push: bool = False,
        reason: str = "apply_face_selection",
    ) -> None:
        if not self._session.filters.allow_faces:
            return

        self._state = self._state.with_faces(
            face_ids,
            op=op,
            keep_mode=True,
            revision_delta=1,
        )
        self._session = self._session.with_mode(SelectionMode.FACE)

        self._emit_session_changed(reason)
        self._emit_state_changed(reason)
        self.mode_changed.emit(SelectionMode.FACE.value)

        if push:
            self.push_session_to_viewport(reason=reason)

    def replace_vertex_selection(
        self,
        vertex_ids: list[int] | tuple[int, ...],
        *,
        push: bool = False,
        reason: str = "replace_vertex_selection",
    ) -> None:
        self.apply_vertex_selection(vertex_ids, op=SelectionOp.REPLACE, push=push, reason=reason)

    def apply_vertex_selection(
        self,
        vertex_ids: list[int] | tuple[int, ...],
        *,
        op: SelectionOp | str = SelectionOp.REPLACE,
        push: bool = False,
        reason: str = "apply_vertex_selection",
    ) -> None:
        if not self._session.filters.allow_vertices:
            return

        self._state = self._state.with_vertices(
            vertex_ids,
            op=op,
            keep_mode=True,
            revision_delta=1,
        )
        self._session = self._session.with_mode(SelectionMode.VERTEX)

        self._emit_session_changed(reason)
        self._emit_state_changed(reason)
        self.mode_changed.emit(SelectionMode.VERTEX.value)

        if push:
            self.push_session_to_viewport(reason=reason)

    def replace_object_selection(
        self,
        object_ids: list[int] | tuple[int, ...],
        *,
        push: bool = False,
        reason: str = "replace_object_selection",
    ) -> None:
        self.apply_object_selection(object_ids, op=SelectionOp.REPLACE, push=push, reason=reason)

    def apply_object_selection(
        self,
        object_ids: list[int] | tuple[int, ...],
        *,
        op: SelectionOp | str = SelectionOp.REPLACE,
        push: bool = False,
        reason: str = "apply_object_selection",
    ) -> None:
        if not self._session.filters.allow_objects:
            return

        sanitized = tuple(int(v) for v in sanitize_ids(object_ids))
        current = self._state.selected_mesh_ids
        updated = apply_selection_op(current, sanitized, op)

        self._state = replace(
            self._state,
            mode=SelectionMode.OBJECT,
            selected_mesh_ids=updated,
            revision=self._state.revision + 1,
        )
        self._session = self._session.with_mode(SelectionMode.OBJECT)

        self._emit_session_changed(reason)
        self._emit_state_changed(reason)
        self.mode_changed.emit(SelectionMode.OBJECT.value)

        if push:
            self.push_session_to_viewport(reason=reason)

    # ------------------------------------------------------------------
    # viewport sync
    # ------------------------------------------------------------------
    def sync_from_viewport(
        self,
        viewport_state: dict[str, Any] | None = None,
        *,
        reason: str = "sync_from_viewport",
    ) -> SelectionState:
        state_dict = viewport_state if viewport_state is not None else self._read_viewport_selection_state()

        raw_mode = state_dict.get("mode")
        if raw_mode is None and self._viewport is not None and hasattr(self._viewport, "get_selection_mode"):
            raw_mode = self._safe_call(lambda: self._viewport.get_selection_mode(), "none")

        actual_mode = self.semantic_mode_from_viewport(raw_mode)

        selected_faces = tuple(int(v) for v in sanitize_ids(state_dict.get("selected_cell_ids")))
        selected_vertices = tuple(int(v) for v in sanitize_ids(state_dict.get("selected_point_ids")))
        selected_edges = tuple(int(v) for v in sanitize_ids(state_dict.get("selected_edge_ids")))
        selected_mesh = tuple(
            int(v)
            for v in sanitize_ids(
                state_dict.get("selected_mesh_ids", state_dict.get("selected_object_ids"))
            )
        )

        brush_enabled = bool(
            state_dict.get(
                "brush_select_enabled",
                self._safe_call(lambda: self._viewport.is_brush_selection_enabled(), False)
                if self._viewport is not None and hasattr(self._viewport, "is_brush_selection_enabled")
                else False,
            )
        )

        # If the backend reports NONE but still has actual ids, derive a more useful actual mode.
        if actual_mode is SelectionMode.NONE:
            if selected_faces:
                actual_mode = SelectionMode.FACE
            elif selected_vertices:
                actual_mode = SelectionMode.VERTEX
            elif selected_edges:
                actual_mode = SelectionMode.EDGE
            elif selected_mesh:
                actual_mode = SelectionMode.OBJECT

        self._state = replace(
            self._state,
            mode=actual_mode,
            selected_face_ids=selected_faces,
            selected_vertex_ids=selected_vertices,
            selected_edge_ids=selected_edges,
            selected_mesh_ids=selected_mesh,
            brush_enabled=brush_enabled,
            revision=self._state.revision + 1,
        )

        self._emit_state_changed(reason)
        return self._state

    def reapply_after_refresh(self, *, reason: str = "reapply_after_refresh") -> None:
        """
        Called after mesh reload / viewport refresh.
        Selection ids are cleared, but controller-owned session intent is preserved.
        """
        self._state = self._state.cleared(keep_mode=False, revision_delta=1)
        self._state = replace(
            self._state,
            mode=self._session.mode,
            interaction_style=self._session.interaction_style,
            brush_enabled=self._session.brush_enabled,
            boundary_highlight=self._session.boundary_highlight,
            revision=self._state.revision + 1,
        )

        self._emit_state_changed(reason)
        self.push_session_to_viewport(reason=reason)

    def push_session_to_viewport(self, *, reason: str = "push_session_to_viewport") -> None:
        """
        Push controller-owned interaction intent into the viewport.

        Required order:
        1. disable_picking()
        2. set_selection_mode(raw_viewport_mode)
        3. set_brush_selection_enabled(...)
        """
        if self._viewport is None:
            return

        raw_mode = self.viewport_mode_from_semantic(self._session.mode)

        try:
            if hasattr(self._viewport, "disable_picking"):
                self._viewport.disable_picking()
        except Exception:
            pass

        try:
            if hasattr(self._viewport, "set_selection_mode"):
                self._viewport.set_selection_mode(raw_mode)
        except Exception:
            pass

        try:
            if hasattr(self._viewport, "set_brush_selection_enabled"):
                self._viewport.set_brush_selection_enabled(bool(self._session.brush_enabled))
        except Exception:
            pass

        self._push_edge_region_strategy_to_viewport()
        self._push_boundary_highlight_to_viewport()
        self.snapshot_emitted.emit(self.snapshot(reason))

    def _push_edge_region_strategy_to_viewport(self) -> None:
        if self._viewport is None:
            return
        try:
            if hasattr(self._viewport, "set_edge_region_strategy"):
                self._viewport.set_edge_region_strategy(self.current_edge_region_strategy())
        except Exception:
            pass

    def _push_boundary_highlight_to_viewport(self) -> None:
        if self._viewport is None:
            return
        try:
            if hasattr(self._viewport, "set_boundary_highlight_visible"):
                self._viewport.set_boundary_highlight_visible(bool(self._session.boundary_highlight))
        except Exception:
            pass

    def _clear_selection_in_viewport(self) -> None:
        if self._viewport is None:
            return

        # Clear raw viewport selection/pick caches across WGPU, PyVista, and
        # compatibility test doubles.  This is generic selection/display-cache
        # cleanup, not tool-specific geometry logic.
        for name, args in (
            ("clear_selection", ()),
            ("set_edge_selection", ((),)),
            ("set_selected_edge_ids", ((),)),
            ("clear_edge_selection", ()),
            ("set_face_selection", ((),)),
            ("highlight_cells", ((),)),
            ("clear_face_selection", ()),
        ):
            fn = getattr(self._viewport, name, None)
            if callable(fn):
                try:
                    fn(*args)
                except Exception:
                    pass

        try:
            if hasattr(self._viewport, "disable_picking"):
                self._viewport.disable_picking()
        except Exception:
            pass

        try:
            if hasattr(self._viewport, "set_brush_selection_enabled"):
                self._viewport.set_brush_selection_enabled(False)
        except Exception:
            pass

    def _read_viewport_selection_state(self) -> dict[str, Any]:
        if self._viewport is None:
            return {}
        try:
            payload = self._viewport.get_selection_state()
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return {}

    # ------------------------------------------------------------------
    # viewport signal handlers
    # ------------------------------------------------------------------
    def _on_viewport_selection_changed(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else None
        self.sync_from_viewport(data, reason="viewport_selection_changed")

    def _on_viewport_point_picked(self, point: object) -> None:
        try:
            seq = list(point)  # type: ignore[arg-type]
        except Exception:
            seq = None
        self._state = self._state.with_pick_point(seq, revision_delta=1)
        self._emit_state_changed("viewport_point_picked")

    def _on_viewport_mesh_loaded(self, _path: object) -> None:
        self.reapply_after_refresh(reason="viewport_mesh_loaded")

    # ------------------------------------------------------------------
    # query helpers
    # ------------------------------------------------------------------
    def has_any_selection(self) -> bool:
        return self._state.has_any_selection

    def has_face_selection(self) -> bool:
        return bool(self._state.selected_face_ids)

    def has_vertex_selection(self) -> bool:
        return bool(self._state.selected_vertex_ids)

    def has_edge_selection(self) -> bool:
        return bool(self._state.selected_edge_ids)

    def has_object_selection(self) -> bool:
        return bool(self._state.selected_mesh_ids)

    def selected_face_ids(self) -> tuple[int, ...]:
        return self._state.selected_face_ids

    def selected_vertex_ids(self) -> tuple[int, ...]:
        return self._state.selected_vertex_ids

    def selected_edge_ids(self) -> tuple[int, ...]:
        return self._state.selected_edge_ids

    def selected_mesh_ids(self) -> tuple[int, ...]:
        return self._state.selected_mesh_ids

    def preferred_manual_mode(self) -> SelectionMode:
        """
        Return the semantic mode that should drive manual-edit request building.

        Rule order:
        1. actual mode if it has matching ids
        2. face selection if present
        3. vertex selection if present
        4. object selection if present
        5. NONE
        """
        state = self._state

        if state.mode is SelectionMode.FACE and state.selected_face_ids:
            return SelectionMode.FACE
        if state.mode is SelectionMode.VERTEX and state.selected_vertex_ids:
            return SelectionMode.VERTEX
        if state.mode is SelectionMode.EDGE and state.selected_edge_ids:
            return SelectionMode.EDGE
        if state.mode is SelectionMode.OBJECT and state.selected_mesh_ids:
            return SelectionMode.OBJECT
        if state.selected_face_ids:
            return SelectionMode.FACE
        if state.selected_vertex_ids:
            return SelectionMode.VERTEX
        if state.selected_edge_ids:
            return SelectionMode.EDGE
        if state.selected_mesh_ids:
            return SelectionMode.OBJECT
        return SelectionMode.NONE

    def selection_payload(self) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
        """
        Compatibility payload for MainWindow manual-edit request building.
        """
        return (
            self.preferred_manual_mode().value,
            self._state.selected_face_ids,
            self._state.selected_vertex_ids,
        )

    def selection_summary(self) -> dict[str, Any]:
        return {
            "mode": self._state.mode.value,
            "session_mode": self._session.mode.value,
            "viewport_mode": self.current_viewport_mode(),
            "interaction_style": self._state.interaction_style.value,
            "brush_enabled": self._state.brush_enabled,
            "boundary_highlight": self._state.boundary_highlight,
            "selected_faces": len(self._state.selected_face_ids),
            "selected_vertices": len(self._state.selected_vertex_ids),
            "selected_edges": len(self._state.selected_edge_ids),
            "selected_mesh_items": len(self._state.selected_mesh_ids),
            "revision": self._state.revision,
        }

    def build_selection_summary(self) -> dict[str, int | str | bool]:
        return {
            "mode": self.preferred_manual_mode().value,
            "selected_faces": len(self._state.selected_face_ids),
            "selected_vertices": len(self._state.selected_vertex_ids),
            "selected_edges": len(self._state.selected_edge_ids),
            "selected_objects": len(self._state.selected_mesh_ids),
            "brush_enabled": bool(self._state.brush_enabled),
        }

    def normalized_manual_snapshot(self) -> SelectionSnapshot:
        """
        Return a snapshot whose state.mode matches the best currently usable semantic selection.
        """
        normalized_state = replace(self._state, mode=self.preferred_manual_mode())
        return SelectionSnapshot(
            state=normalized_state,
            session=self._session,
            reason="normalized_manual_snapshot",
        )

    def debug_state(self) -> dict[str, Any]:
        return {
            "semantic_mode": self._state.mode.value,
            "session_mode": self._session.mode.value,
            "viewport_mode": self.current_viewport_mode(),
            "face_ids": list(self._state.selected_face_ids),
            "vertex_ids": list(self._state.selected_vertex_ids),
            "edge_ids": list(self._state.selected_edge_ids),
            "mesh_ids": list(self._state.selected_mesh_ids),
            "brush_enabled": bool(self._state.brush_enabled),
            "interaction_style": self._state.interaction_style.value,
            "revision": int(self._state.revision),
        }

    # ------------------------------------------------------------------
    # viewport action wrappers
    # ------------------------------------------------------------------
    def grow_face_selection(self) -> bool:
        """Ask the active viewport to grow the current face selection.

        Returns True when the viewport action was invoked.  The controller then
        immediately syncs from the viewport so callers and tests can observe the
        updated face IDs without relying on Qt signal timing.  Exceptions are
        reported through the viewport status signal when available instead of
        being swallowed silently.
        """
        return self._invoke_viewport_selection_action(
            "grow_selection",
            reason="grow_face_selection",
        )

    def shrink_face_selection(self) -> bool:
        """Ask the active viewport to shrink the current face selection."""
        return self._invoke_viewport_selection_action(
            "shrink_selection",
            reason="shrink_face_selection",
        )

    def select_connected_points_from_current(self, *, append: bool = False) -> bool:
        if not self._state.selected_vertex_ids:
            return False
        return self._invoke_viewport_selection_action(
            "select_connected_points_from_vertex",
            int(self._state.selected_vertex_ids[0]),
            append=bool(append),
            reason="select_connected_points_from_current",
        )

    def _invoke_viewport_selection_action(
        self,
        action_name: str,
        *args: Any,
        reason: str,
        **kwargs: Any,
    ) -> bool:
        if self._viewport is None:
            return False

        action = getattr(self._viewport, action_name, None)
        if not callable(action):
            return False

        try:
            action(*args, **kwargs)
        except Exception as exc:
            self._report_viewport_action_error(action_name, exc)
            return False

        self.sync_from_viewport(reason=reason)
        return True

    def _report_viewport_action_error(self, action_name: str, exc: Exception) -> None:
        signal = getattr(self._viewport, "status_changed", None) if self._viewport is not None else None
        message = f"Viewport selection action failed: {action_name}: {exc!r}"
        try:
            if signal is not None and hasattr(signal, "emit"):
                signal.emit(message)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _emit_state_changed(self, reason: str) -> None:
        self.state_changed.emit(self._state)
        self.snapshot_emitted.emit(self.snapshot(reason))

    def _emit_session_changed(self, reason: str) -> None:
        self.session_changed.emit(self._session)
        self.snapshot_emitted.emit(self.snapshot(reason))

    @staticmethod
    def _safe_call(func: Callable[[], Any], default: Any) -> Any:
        try:
            return func()
        except Exception:
            return default

    @classmethod
    def _coerce_mode(cls, value: SelectionMode | str | None) -> SelectionMode:
        if isinstance(value, SelectionMode):
            return value

        raw = str(value or "").strip().lower()
        try:
            return SelectionMode(raw)
        except Exception:
            return cls.semantic_mode_from_viewport(raw)

    @staticmethod
    def _coerce_style(value: InteractionStyle | str | None) -> InteractionStyle:
        if isinstance(value, InteractionStyle):
            return value

        raw = str(value or "").strip().lower()
        for item in InteractionStyle:
            if item.value == raw:
                return item
        return InteractionStyle.VIEWER


__all__ = ["SelectionController"]
