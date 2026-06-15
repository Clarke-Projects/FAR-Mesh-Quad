from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable

import numpy as np


class SelectionMode(str, Enum):
    NONE = "none"
    VERTEX = "vertex"
    FACE = "face"
    EDGE = "edge"
    OBJECT = "object"


class InteractionStyle(str, Enum):
    VIEWER = "viewer"
    BRUSH = "brush"


class SelectionOp(str, Enum):
    REPLACE = "replace"
    ADD = "add"
    REMOVE = "remove"
    TOGGLE = "toggle"
    CLEAR = "clear"


def _coerce_selection_mode(value: SelectionMode | str | None) -> SelectionMode:
    if isinstance(value, SelectionMode):
        return value

    raw = str(value or "").strip().lower()
    for item in SelectionMode:
        if item.value == raw:
            return item
    return SelectionMode.NONE


def _coerce_selection_op(value: SelectionOp | str | None) -> SelectionOp:
    if isinstance(value, SelectionOp):
        return value

    raw = str(value or "").strip().lower()
    for item in SelectionOp:
        if item.value == raw:
            return item
    return SelectionOp.REPLACE


def sanitize_ids(values: Any) -> tuple[int, ...]:
    """
    Normalize arbitrary selection id input into a stable tuple[int, ...].

    Rules:
    - None -> ()
    - cast to int64 when possible
    - flatten
    - drop negative ids
    - unique + sorted
    """
    if values is None:
        return ()

    try:
        arr = np.asarray(values, dtype=np.int64).reshape(-1)
    except Exception:
        return ()

    if arr.size == 0:
        return ()

    arr = arr[arr >= 0]
    if arr.size == 0:
        return ()

    arr = np.unique(arr)
    return tuple(int(v) for v in arr.tolist())


def apply_selection_op(
    current: Iterable[int] | tuple[int, ...],
    incoming: Iterable[int] | tuple[int, ...],
    op: SelectionOp | str = SelectionOp.REPLACE,
) -> tuple[int, ...]:
    """
    Apply a selection mutation operation and return the new normalized tuple.
    """
    op_enum = _coerce_selection_op(op)
    current_ids = set(sanitize_ids(current))
    incoming_ids = set(sanitize_ids(incoming))

    if op_enum is SelectionOp.CLEAR:
        return ()

    if op_enum is SelectionOp.REPLACE:
        return tuple(sorted(incoming_ids))

    if op_enum is SelectionOp.ADD:
        return tuple(sorted(current_ids | incoming_ids))

    if op_enum is SelectionOp.REMOVE:
        return tuple(sorted(current_ids - incoming_ids))

    if op_enum is SelectionOp.TOGGLE:
        return tuple(sorted(current_ids ^ incoming_ids))

    return tuple(sorted(incoming_ids))


@dataclass(slots=True, frozen=True)
class SelectionFilters:
    """
    Controller-owned selection-domain permissions.

    These do not read the viewport directly. They describe which domains the
    controller/session intends to allow.
    """
    allow_vertices: bool = True
    allow_faces: bool = True
    allow_edges: bool = True
    allow_meshes: bool = True

    def allows_mode(self, mode: SelectionMode | str | None) -> bool:
        mode_enum = _coerce_selection_mode(mode)

        if mode_enum is SelectionMode.NONE:
            return True
        if mode_enum is SelectionMode.VERTEX:
            return self.allow_vertices
        if mode_enum is SelectionMode.FACE:
            return self.allow_faces
        if mode_enum is SelectionMode.EDGE:
            return self.allow_edges
        if mode_enum is SelectionMode.OBJECT:
            return self.allow_meshes
        return False


@dataclass(slots=True, frozen=True)
class SelectionSession:
    """
    Controller-owned intended interaction/session state.

    This is not the same thing as actual current selected ids coming back from
    the viewport. It stores what the controller wants the viewport to be doing.
    """
    mode: SelectionMode = SelectionMode.NONE
    interaction_style: InteractionStyle = InteractionStyle.VIEWER
    brush_enabled: bool = False
    boundary_highlight: bool = False
    filters: SelectionFilters = field(default_factory=SelectionFilters)

    def with_mode(self, mode: SelectionMode | str | None) -> SelectionSession:
        return replace(self, mode=_coerce_selection_mode(mode))

    def with_style(self, style: InteractionStyle | str | None) -> SelectionSession:
        if isinstance(style, InteractionStyle):
            style_enum = style
        else:
            raw = str(style or "").strip().lower()
            style_enum = InteractionStyle.BRUSH if raw == InteractionStyle.BRUSH.value else InteractionStyle.VIEWER
        return replace(self, interaction_style=style_enum)

    def with_brush_enabled(self, enabled: bool) -> SelectionSession:
        return replace(self, brush_enabled=bool(enabled))

    def with_boundary_highlight(self, enabled: bool) -> SelectionSession:
        return replace(self, boundary_highlight=bool(enabled))


@dataclass(slots=True, frozen=True)
class SelectionState:
    """
    Actual current selection state.

    This is the controller's normalized mirror of the viewport selection state.
    """
    mode: SelectionMode = SelectionMode.NONE
    selected_vertex_ids: tuple[int, ...] = ()
    selected_face_ids: tuple[int, ...] = ()
    selected_edge_ids: tuple[int, ...] = ()
    selected_mesh_ids: tuple[int, ...] = ()
    brush_enabled: bool = False
    interaction_style: InteractionStyle = InteractionStyle.VIEWER
    boundary_highlight: bool = False
    last_pick_point: tuple[float, float, float] | None = None
    revision: int = 0

    @property
    def has_any_selection(self) -> bool:
        return bool(
            self.selected_vertex_ids
            or self.selected_face_ids
            or self.selected_edge_ids
            or self.selected_mesh_ids
        )

    @property
    def selected_object_ids(self) -> tuple[int, ...]:
        return self.selected_mesh_ids

    def with_mode(
        self,
        mode: SelectionMode | str | None,
        *,
        clear_other_domains: bool = True,
        revision_delta: int = 1,
    ) -> SelectionState:
        mode_enum = _coerce_selection_mode(mode)

        next_vertices = self.selected_vertex_ids
        next_faces = self.selected_face_ids
        next_edges = self.selected_edge_ids
        next_mesh = self.selected_mesh_ids

        if clear_other_domains:
            if mode_enum is SelectionMode.VERTEX:
                next_faces = ()
                next_edges = ()
                next_mesh = ()
            elif mode_enum is SelectionMode.FACE:
                next_vertices = ()
                next_edges = ()
                next_mesh = ()
            elif mode_enum is SelectionMode.EDGE:
                next_vertices = ()
                next_faces = ()
                next_mesh = ()
            elif mode_enum is SelectionMode.OBJECT:
                next_vertices = ()
                next_faces = ()
                next_edges = ()
            else:
                next_vertices = ()
                next_faces = ()
                next_edges = ()
                next_mesh = ()

        return replace(
            self,
            mode=mode_enum,
            selected_vertex_ids=next_vertices,
            selected_face_ids=next_faces,
            selected_edge_ids=next_edges,
            selected_mesh_ids=next_mesh,
            revision=self.revision + int(revision_delta),
        )

    def with_vertices(
        self,
        vertex_ids: Iterable[int] | tuple[int, ...],
        *,
        op: SelectionOp | str = SelectionOp.REPLACE,
        keep_mode: bool = True,
        revision_delta: int = 1,
    ) -> SelectionState:
        next_ids = apply_selection_op(self.selected_vertex_ids, vertex_ids, op)

        return replace(
            self,
            mode=SelectionMode.VERTEX if keep_mode else self.mode,
            selected_vertex_ids=next_ids,
            selected_face_ids=() if keep_mode else self.selected_face_ids,
            selected_edge_ids=() if keep_mode else self.selected_edge_ids,
            selected_mesh_ids=() if keep_mode else self.selected_mesh_ids,
            revision=self.revision + int(revision_delta),
        )

    def with_faces(
        self,
        face_ids: Iterable[int] | tuple[int, ...],
        *,
        op: SelectionOp | str = SelectionOp.REPLACE,
        keep_mode: bool = True,
        revision_delta: int = 1,
    ) -> SelectionState:
        next_ids = apply_selection_op(self.selected_face_ids, face_ids, op)

        return replace(
            self,
            mode=SelectionMode.FACE if keep_mode else self.mode,
            selected_face_ids=next_ids,
            selected_vertex_ids=() if keep_mode else self.selected_vertex_ids,
            selected_edge_ids=() if keep_mode else self.selected_edge_ids,
            selected_mesh_ids=() if keep_mode else self.selected_mesh_ids,
            revision=self.revision + int(revision_delta),
        )

    def with_edges(
        self,
        edge_ids: Iterable[int] | tuple[int, ...],
        *,
        op: SelectionOp | str = SelectionOp.REPLACE,
        keep_mode: bool = True,
        revision_delta: int = 1,
    ) -> SelectionState:
        next_ids = apply_selection_op(self.selected_edge_ids, edge_ids, op)

        return replace(
            self,
            mode=SelectionMode.EDGE if keep_mode else self.mode,
            selected_edge_ids=next_ids,
            selected_vertex_ids=() if keep_mode else self.selected_vertex_ids,
            selected_face_ids=() if keep_mode else self.selected_face_ids,
            selected_mesh_ids=() if keep_mode else self.selected_mesh_ids,
            revision=self.revision + int(revision_delta),
        )

    def with_mesh_items(
        self,
        mesh_ids: Iterable[int] | tuple[int, ...],
        *,
        op: SelectionOp | str = SelectionOp.REPLACE,
        keep_mode: bool = True,
        revision_delta: int = 1,
    ) -> SelectionState:
        next_ids = apply_selection_op(self.selected_mesh_ids, mesh_ids, op)

        return replace(
            self,
            mode=SelectionMode.OBJECT if keep_mode else self.mode,
            selected_mesh_ids=next_ids,
            selected_vertex_ids=() if keep_mode else self.selected_vertex_ids,
            selected_face_ids=() if keep_mode else self.selected_face_ids,
            selected_edge_ids=() if keep_mode else self.selected_edge_ids,
            revision=self.revision + int(revision_delta),
        )

    def with_pick_point(
        self,
        point: Iterable[float] | None,
        *,
        revision_delta: int = 1,
    ) -> SelectionState:
        normalized: tuple[float, float, float] | None = None
        if point is not None:
            try:
                seq = list(point)
                if len(seq) >= 3:
                    normalized = (float(seq[0]), float(seq[1]), float(seq[2]))
            except Exception:
                normalized = None

        return replace(
            self,
            last_pick_point=normalized,
            revision=self.revision + int(revision_delta),
        )

    def cleared(
        self,
        *,
        keep_mode: bool = False,
        revision_delta: int = 1,
    ) -> SelectionState:
        return replace(
            self,
            mode=self.mode if keep_mode else SelectionMode.NONE,
            selected_vertex_ids=(),
            selected_face_ids=(),
            selected_edge_ids=(),
            selected_mesh_ids=(),
            last_pick_point=None,
            revision=self.revision + int(revision_delta),
        )


@dataclass(slots=True, frozen=True)
class SelectionSnapshot:
    """
    Combined snapshot of actual state + controller session intent.

    Convenience properties are intentionally provided so older call sites can
    still read snapshot.mode / snapshot.face_ids / snapshot.vertex_ids while
    the new model remains state/session based.
    """
    state: SelectionState = field(default_factory=SelectionState)
    session: SelectionSession = field(default_factory=SelectionSession)
    reason: str = ""

    @property
    def mode(self) -> SelectionMode:
        return self.state.mode

    @property
    def face_ids(self) -> np.ndarray:
        return np.asarray(self.state.selected_face_ids, dtype=np.int64)

    @property
    def vertex_ids(self) -> np.ndarray:
        return np.asarray(self.state.selected_vertex_ids, dtype=np.int64)

    @property
    def edge_ids(self) -> np.ndarray:
        return np.asarray(self.state.selected_edge_ids, dtype=np.int64)

    @property
    def object_ids(self) -> np.ndarray:
        return np.asarray(self.state.selected_mesh_ids, dtype=np.int64)

    @property
    def brush_enabled(self) -> bool:
        return bool(self.state.brush_enabled)

    @property
    def interaction_style(self) -> InteractionStyle:
        return self.state.interaction_style

    @property
    def boundary_highlight(self) -> bool:
        return bool(self.state.boundary_highlight)

    @property
    def revision(self) -> int:
        return int(self.state.revision)

    @property
    def has_any_selection(self) -> bool:
        return self.state.has_any_selection


def make_face_selection(
    face_ids: Iterable[int] | tuple[int, ...],
    *,
    brush_enabled: bool = False,
    interaction_style: InteractionStyle = InteractionStyle.VIEWER,
) -> SelectionState:
    return SelectionState(
        mode=SelectionMode.FACE,
        selected_face_ids=sanitize_ids(face_ids),
        brush_enabled=bool(brush_enabled),
        interaction_style=interaction_style,
    )


def make_vertex_selection(
    vertex_ids: Iterable[int] | tuple[int, ...],
    *,
    brush_enabled: bool = False,
    interaction_style: InteractionStyle = InteractionStyle.VIEWER,
) -> SelectionState:
    return SelectionState(
        mode=SelectionMode.VERTEX,
        selected_vertex_ids=sanitize_ids(vertex_ids),
        brush_enabled=bool(brush_enabled),
        interaction_style=interaction_style,
    )


__all__ = [
    "InteractionStyle",
    "SelectionFilters",
    "SelectionMode",
    "SelectionOp",
    "SelectionSession",
    "SelectionSnapshot",
    "SelectionState",
    "apply_selection_op",
    "make_face_selection",
    "make_vertex_selection",
    "sanitize_ids",
]
