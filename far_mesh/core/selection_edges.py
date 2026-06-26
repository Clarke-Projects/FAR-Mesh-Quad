"""Selection-domain edge expansion and live rim-evidence helpers.

This module belongs to the host selection layer, not to Bore Recognition.
It converts viewport picks and controller edge-selection intent into compact,
existing mesh edge IDs.  In Bore ``bore_rim`` mode it may resolve a local
clicked rim/opening evidence set, but it must not classify BORE/POCKET/CHAMFER,
assign surface ownership, build CandidateData, authorize rebuild, or mutate
mesh topology.

Semantic boundary:
    viewport click / conservative edge cloud
    -> selection-domain edge evidence
    -> optional compact local rim evidence for BoreTool

The returned IDs are visual/tool-input evidence only.  Region Select,
Recognition, RebuildTarget, and Rebuild own their later semantic stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

EdgeKey = tuple[int, int]


@dataclass(frozen=True)
class EdgeSelectionRegion:
    edge_ids: tuple[int, ...]
    confidence: float
    mode: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BoreRimSelection:
    """Selection-layer result for a compact user-indicated rim.

    This is not feature identity and not rebuild authority.  It is the
    selection stack's best local edge evidence for the rim/opening the user
    indicated with the active click.
    """
    rim_edge_ids: tuple[int, ...]
    opposite_edge_ids: tuple[int, ...] = ()
    confidence: float = 0.0
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    radius: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

# ----------------------------------------------------------------------
# Public bore‑aware rim selector
# ----------------------------------------------------------------------

def select_bore_rim_loop(
    *,
    vertices: np.ndarray,
    faces: np.ndarray | None,
    edge_index_to_vertices: np.ndarray,
    edge_to_faces: Mapping[EdgeKey, list[int] | tuple[int, ...]] | None,
    open_edges: np.ndarray | None,
    start_edge_index: int,
    max_gap_edge_lengths: float = 3.25,
    max_radius_ratio: float = 1.15,
    min_loop_edges: int = 12,
) -> BoreRimSelection:
    """Return compact rim/opening edge evidence for a BoreTool session.

    This helper is still a selection operation.  It asks the ``bore_rim``
    strategy for a local, user-click-rooted rim evidence set and then measures
    that set for UI/tool handoff.  It does not decide feature identity, own
    wall faces, or authorize rebuild.
    """
    # First get the main rim using ring strategy
    result = select_edge_region(
        vertices=vertices,
        faces=faces,
        edge_index_to_vertices=edge_index_to_vertices,
        edge_to_faces=edge_to_faces,
        open_edges=open_edges,
        start_edge_index=start_edge_index,
        max_gap_edge_lengths=max_gap_edge_lengths,
        strategy="bore_rim",       # dedicated one-loop Bore opening extraction
        allow_same_plane_siblings=False,
        allow_gap_bridge=False,
    )

    if len(result.edge_ids) < int(min_loop_edges):
        return BoreRimSelection(
            rim_edge_ids=(),
            confidence=0.0,
            diagnostics={
                "error": f"Recovered rim too small: {len(result.edge_ids)} edges, need {min_loop_edges}",
                **result.diagnostics,
            },
        )

    # Build edge key to index mapping
    edge_arr = _as_edges(edge_index_to_vertices)
    key_to_index = {_edge_key(edge): i for i, edge in enumerate(edge_arr)}
    rim_keys = {_edge_key(edge_arr[eid]) for eid in result.edge_ids if 0 <= eid < len(edge_arr)}
    if not rim_keys:
        return BoreRimSelection(rim_edge_ids=(), confidence=0.0, diagnostics={"error": "No valid rim keys"})

    # Estimate loop geometry
    rim_vertices = _collect_vertices_from_edges(vertices, rim_keys)
    if len(rim_vertices) < 4:
        return BoreRimSelection(rim_edge_ids=result.edge_ids, confidence=0.5, diagnostics={"warning": "Too few vertices for geometry"})

    center, axis, radius = _fit_loop_geometry(rim_vertices)
    diagnostics = {
        "rim_edge_count": len(result.edge_ids),
        "center": tuple(float(v) for v in center),
        "axis": tuple(float(v) for v in axis),
        "radius": float(radius),
        "selection_diagnostics": result.diagnostics,
    }

    # Try to detect opposite rim
    opposite_keys = _find_opposite_rim(
        vertices=vertices,
        faces=faces,
        edge_index_to_vertices=edge_arr,
        edge_to_faces=edge_to_faces,
        open_edges=open_edges,
        rim_keys=rim_keys,
        center=center,
        axis=axis,
        radius=radius,
        max_radius_ratio=float(max_radius_ratio),
        min_loop_edges=int(min_loop_edges),
    )
    opposite_edge_ids = tuple(key_to_index[key] for key in opposite_keys if key in key_to_index)

    confidence = _estimate_rim_confidence(len(result.edge_ids), len(opposite_edge_ids), radius)
    return BoreRimSelection(
        rim_edge_ids=result.edge_ids,
        opposite_edge_ids=opposite_edge_ids,
        confidence=confidence,
        center=tuple(float(v) for v in center),
        axis=tuple(float(v) for v in axis),
        radius=float(radius),
        diagnostics=diagnostics,
    )


# ----------------------------------------------------------------------
# Original edge region selector (preserved)
# ----------------------------------------------------------------------

def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _pick_point_from_info(info: Mapping[str, Any]) -> np.ndarray | None:
    for key in (
        "world_pos",
        "world_position",
        "position",
        "point",
        "pos",
        "coord",
        "coords",
    ):
        raw = info.get(key)
        if raw is None:
            continue
        try:
            arr = np.asarray(raw, dtype=float).reshape(-1)
        except Exception:
            continue
        if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
            return arr[:3].astype(float, copy=True)
    return None


def _closest_edge_on_triangle_by_point(
    *,
    tri: np.ndarray,
    vertices: np.ndarray,
    point: np.ndarray,
) -> EdgeKey | None:
    try:
        ids = [int(tri[0]), int(tri[1]), int(tri[2])]
        pts = vertices[np.asarray(ids, dtype=np.int32)].astype(float)
    except Exception:
        return None

    best_key: EdgeKey | None = None
    best_dist = float("inf")
    for a, b, p0, p1 in (
        (ids[0], ids[1], pts[0], pts[1]),
        (ids[1], ids[2], pts[1], pts[2]),
        (ids[2], ids[0], pts[2], pts[0]),
    ):
        segment = p1 - p0
        denom = float(np.dot(segment, segment))
        if denom <= 1e-18:
            continue
        t = float(np.clip(np.dot(point - p0, segment) / denom, 0.0, 1.0))
        closest = p0 + t * segment
        dist = float(np.linalg.norm(point - closest))
        if dist < best_dist:
            best_dist = dist
            best_key = (int(a), int(b)) if int(a) <= int(b) else (int(b), int(a))
    return best_key


def _closest_edge_on_triangle_by_barycentric(
    *,
    tri: np.ndarray,
    bary_raw: Any,
) -> EdgeKey | None:
    try:
        bary = np.asarray(bary_raw, dtype=float).reshape(-1)
    except Exception:
        return None
    if bary.size < 3 or not np.all(np.isfinite(bary[:3])):
        return None
    try:
        ids = [int(tri[0]), int(tri[1]), int(tri[2])]
    except Exception:
        return None
    opposite = int(np.argmin(bary[:3]))
    if opposite == 0:
        a, b = ids[1], ids[2]
    elif opposite == 1:
        a, b = ids[0], ids[2]
    else:
        a, b = ids[0], ids[1]
    return (int(a), int(b)) if int(a) <= int(b) else (int(b), int(a))


def resolve_edge_index_from_pick_info(
    info: Mapping[str, Any] | None,
    *,
    faces: np.ndarray | None,
    edge_key_to_index: Mapping[EdgeKey, int],
    vertices: np.ndarray | None = None,
    fallback_face_index: int | None = None,
) -> int | None:
    payload = info or {}
    for direct_key in ("edge_index", "edge_id", "selected_edge_id"):
        direct = _coerce_int(payload.get(direct_key))
        if direct is not None and direct >= 0:
            return int(direct)
    face_index = _coerce_int(payload.get("face_index"))
    if face_index is None:
        face_index = _coerce_int(fallback_face_index)
    if faces is None or face_index is None:
        return None
    if face_index < 0 or face_index >= len(faces):
        return None
    try:
        tri = faces[int(face_index)]
    except Exception:
        return None
    if vertices is not None:
        pick_point = _pick_point_from_info(payload)
        if pick_point is not None:
            key = _closest_edge_on_triangle_by_point(tri=tri, vertices=vertices, point=pick_point)
            if key is not None and key in edge_key_to_index:
                return int(edge_key_to_index[key])
    key = _closest_edge_on_triangle_by_barycentric(tri=tri, bary_raw=payload.get("face_coord"))
    if key is not None and key in edge_key_to_index:
        return int(edge_key_to_index[key])
    try:
        a, b = int(tri[0]), int(tri[1])
        fallback_key = (a, b) if a <= b else (b, a)
    except Exception:
        return None
    value = edge_key_to_index.get(fallback_key)
    return int(value) if value is not None else None


@dataclass(frozen=True)
class _RingFrame:
    center: np.ndarray
    normal: np.ndarray
    radius: float
    median_edge_length: float
    sample_count: int


def _normalize_edge_selection_strategy(strategy: str | None) -> str:
    raw = str(strategy or "safe").strip().lower()
    aliases = {
        "default": "safe",
        "conservative": "safe",
        "chain": "safe",
        "continuous": "safe",
        "loop": "safe",
        "bore": "bore_rim",
        "bore_rim": "bore_rim",
        "bore_opening": "bore_rim",
        "opening": "bore_rim",
        "rim": "bore_rim",
        "component": "open_component",
        "open": "open_component",
        "open_boundary": "open_component",
        "feature": "feature",
        "feature_chain": "feature",
        "feature_ring": "ring",
        "bridge": "ring",
        "old": "aggressive",
        "legacy": "aggressive",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"safe", "bore_rim", "open_component", "feature", "ring", "aggressive", "single"}:
        return "safe"
    return normalized


def select_edge_region(
    *,
    vertices: np.ndarray,
    faces: np.ndarray | None,
    edge_index_to_vertices: np.ndarray,
    edge_to_faces: Mapping[EdgeKey, list[int] | tuple[int, ...]] | None,
    open_edges: np.ndarray | None,
    start_edge_index: int,
    max_gap_edge_lengths: float = 3.25,
    strategy: str = "safe",
    allow_same_plane_siblings: bool | None = None,
    allow_gap_bridge: bool | None = None,
    max_selected_edges: int | None = None,
) -> EdgeSelectionRegion:
    """Feature-aware Ctrl/Cmd edge-region selector.

    Strategy:
    - safe:      conservative, no sibling chains, no gap bridge (default)
    - bore_rim:  Bore opening mode; returns one validated local rim loop when possible
    - ring:      aggressive, collects same-plane open chains and bridges gaps
    - aggressive: same as ring
    - open_component: selects full connected open-boundary component
    - feature:   walks continuous feature chain (no bridges)
    - single:    returns only the clicked edge
    """
    strategy_key = _normalize_edge_selection_strategy(strategy)
    if strategy_key == "bore_rim":
        return _select_bore_opening_edge_region(
            vertices=vertices,
            faces=faces,
            edge_index_to_vertices=edge_index_to_vertices,
            edge_to_faces=edge_to_faces,
            open_edges=open_edges,
            start_edge_index=start_edge_index,
            max_gap_edge_lengths=max_gap_edge_lengths,
            max_selected_edges=max_selected_edges,
        )

    if allow_same_plane_siblings is None:
        allow_same_plane_siblings = strategy_key in {"ring", "aggressive"}
    if allow_gap_bridge is None:
        allow_gap_bridge = strategy_key in {"ring", "aggressive"}

    vertices_arr = _as_vertices(vertices)
    edge_arr = _as_edges(edge_index_to_vertices)
    if edge_arr.size == 0:
        return EdgeSelectionRegion((), 0.0, "empty", {"reason": "no_edges"})
    if start_edge_index < 0 or start_edge_index >= len(edge_arr):
        return EdgeSelectionRegion((), 0.0, "invalid_start_edge", {"start_edge_index": int(start_edge_index)})

    faces_arr = _as_faces_or_none(faces)
    edge_faces = _normalize_edge_faces(edge_to_faces)
    open_keys = _open_edge_keys(open_edges)
    key_to_index = _edge_key_to_index(edge_arr)
    vertex_to_edges = _vertex_to_edges(edge_arr)

    start_key = _edge_key(edge_arr[int(start_edge_index)])
    start_strength = _edge_feature_strength(vertices_arr, faces_arr, edge_faces, start_key)
    start_length = _edge_length(vertices_arr, start_key)

    connected_open = _connected_open_edge_component(start_key, open_keys, vertex_to_edges)

    if strategy_key == "single":
        return EdgeSelectionRegion(
            edge_ids=(int(start_edge_index),),
            confidence=0.5,
            mode="single_edge",
            diagnostics={"start_edge_index": int(start_edge_index), "strategy": strategy_key},
        )

    if start_key in open_keys:
        start_path = _extract_linear_open_path(start_key, open_keys, vertex_to_edges)
        if strategy_key == "open_component":
            seed_keys = set(connected_open) if connected_open else set(start_path)
        elif allow_same_plane_siblings:
            all_open_paths = _extract_all_open_chains(open_keys, vertex_to_edges)
            seed_keys = _select_same_plane_open_paths(
                start_key=start_key,
                start_path=start_path,
                all_paths=all_open_paths,
                vertices=vertices_arr,
                open_keys=open_keys,
                vertex_to_edges=vertex_to_edges,
                faces=faces_arr,
                edge_faces=edge_faces,
                edge_arr=edge_arr,
            )
        else:
            seed_keys = set(start_path)
    else:
        seed_keys = {start_key}

    frame = None
    if strategy_key != "open_component":
        frame = _fit_local_ring_frame(
            vertices_arr,
            edge_arr,
            seed_keys=seed_keys,
            open_keys=open_keys,
            edge_faces=edge_faces,
            faces=faces_arr,
            start_key=start_key,
        )

    visited = set(seed_keys)

    # Walk from path endpoints if available
    walk_starts = _seed_path_walk_starts(seed_keys, start_key)
    if not walk_starts:
        a, b = start_key
        walk_starts = ((int(a), int(b), start_key), (int(b), int(a), start_key))

    for previous_vertex, current_vertex, current_key_for_walk in walk_starts:
        _walk_feature_ring_direction(
            vertices=vertices_arr,
            faces=faces_arr,
            edge_faces=edge_faces,
            edge_arr=edge_arr,
            open_keys=open_keys,
            vertex_to_edges=vertex_to_edges,
            frame=frame,
            visited=visited,
            start_key=start_key,
            current_key=current_key_for_walk,
            previous_vertex=int(previous_vertex),
            current_vertex=int(current_vertex),
            start_strength=start_strength,
            start_length=start_length,
            max_gap_edge_lengths=max_gap_edge_lengths,
            allow_gap_bridge=allow_gap_bridge,
            max_selected_edges=max_selected_edges,
        )
        if max_selected_edges is not None and int(max_selected_edges) > 0 and len(visited) >= int(max_selected_edges):
            break

    if max_selected_edges is not None and int(max_selected_edges) > 0 and len(visited) > int(max_selected_edges):
        visited = set(sorted(visited)[: int(max_selected_edges)])

    edge_ids = tuple(sorted(int(key_to_index[key]) for key in visited if key in key_to_index))
    mode = _classify_mode(start_key, open_keys, len(connected_open), len(edge_ids), frame)
    confidence = _estimate_confidence(mode, len(edge_ids), len(connected_open), frame, start_strength)

    return EdgeSelectionRegion(
        edge_ids=edge_ids,
        confidence=confidence,
        mode=mode,
        diagnostics={
            "strategy": strategy_key,
            "allow_same_plane_siblings": allow_same_plane_siblings,
            "allow_gap_bridge": allow_gap_bridge,
            "selected_edge_count": len(edge_ids),
            "mode": mode,
            "confidence": confidence,
        },
    )


# ----------------------------------------------------------------------
# Helper functions (unchanged from original)
# ----------------------------------------------------------------------

def _as_vertices(vertices: np.ndarray) -> np.ndarray:
    arr = np.asarray(vertices, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.empty((0, 3), dtype=float)
    return arr[:, :3].astype(float, copy=False)


def _as_edges(edge_index_to_vertices: np.ndarray) -> np.ndarray:
    arr = np.asarray(edge_index_to_vertices, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.empty((0, 2), dtype=np.int64)
    return arr[:, :2].astype(np.int64, copy=False)


def _as_faces_or_none(faces: np.ndarray | None) -> np.ndarray | None:
    if faces is None:
        return None
    arr = np.asarray(faces, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    return arr[:, :3].astype(np.int64, copy=False)


def _edge_key(edge: np.ndarray | tuple[int, int] | list[int]) -> EdgeKey:
    a = int(edge[0])
    b = int(edge[1])
    return (a, b) if a <= b else (b, a)


def _edge_key_to_index(edge_arr: np.ndarray) -> dict[EdgeKey, int]:
    return {_edge_key(edge): int(i) for i, edge in enumerate(edge_arr)}


def _normalize_edge_faces(edge_to_faces: Mapping[EdgeKey, list[int] | tuple[int, ...]] | None) -> dict[EdgeKey, list[int]]:
    if not edge_to_faces:
        return {}
    result: dict[EdgeKey, list[int]] = {}
    for key, value in edge_to_faces.items():
        try:
            result[_edge_key(key)] = [int(v) for v in value]
        except Exception:
            continue
    return result


def _open_edge_keys(open_edges: np.ndarray | None) -> set[EdgeKey]:
    if open_edges is None:
        return set()
    arr = np.asarray(open_edges, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return set()
    return {_edge_key(edge) for edge in arr[:, :2]}


def _vertex_to_edges(edge_arr: np.ndarray) -> dict[int, list[EdgeKey]]:
    result: dict[int, list[EdgeKey]] = {}
    for edge in edge_arr:
        key = _edge_key(edge)
        a, b = key
        result.setdefault(a, []).append(key)
        result.setdefault(b, []).append(key)
    return result


def _connected_open_edge_component(start_key: EdgeKey, open_keys: set[EdgeKey], vertex_to_edges: dict[int, list[EdgeKey]]) -> set[EdgeKey]:
    if start_key not in open_keys:
        return set()
    visited: set[EdgeKey] = set()
    stack = [start_key]
    while stack:
        key = stack.pop()
        if key in visited or key not in open_keys:
            continue
        visited.add(key)
        a, b = key
        for vertex in (a, b):
            for neighbor in vertex_to_edges.get(vertex, []):
                if neighbor in open_keys and neighbor not in visited:
                    stack.append(neighbor)
    return visited


def _extract_linear_open_path(start_key: EdgeKey, open_keys: set[EdgeKey], vertex_to_edges: dict[int, list[EdgeKey]]) -> set[EdgeKey]:
    start_key = _edge_key(start_key)
    if start_key not in open_keys:
        return {start_key}

    def open_incident(vertex: int) -> list[EdgeKey]:
        return [e for e in vertex_to_edges.get(vertex, []) if _edge_key(e) in open_keys]

    def other_vertex(edge: EdgeKey, vertex: int) -> int | None:
        a, b = edge
        if vertex == a:
            return b
        if vertex == b:
            return a
        return None

    visited: set[EdgeKey] = {start_key}
    a, b = start_key

    for prev, cur in ((a, b), (b, a)):
        previous_edge = start_key
        current_vertex = cur
        while True:
            incident = open_incident(current_vertex)
            if len(incident) != 2:
                break
            outgoing = [e for e in incident if e != previous_edge]
            if len(outgoing) != 1:
                break
            next_edge = outgoing[0]
            if next_edge in visited:
                break
            nxt_vertex = other_vertex(next_edge, current_vertex)
            if nxt_vertex is None or nxt_vertex == prev:
                break
            visited.add(next_edge)
            prev, current_vertex, previous_edge = current_vertex, nxt_vertex, next_edge
    return visited


def _extract_all_open_chains(open_keys: set[EdgeKey], vertex_to_edges: dict[int, list[EdgeKey]]) -> list[set[EdgeKey]]:
    remaining = set(open_keys)
    chains = []
    def open_incident(v):
        return [e for e in vertex_to_edges.get(v, []) if _edge_key(e) in open_keys]

    while remaining:
        start = next(iter(remaining))
        chain = {start}
        remaining.remove(start)
        a, b = start
        for prev, cur in ((a, b), (b, a)):
            current_vertex = cur
            previous_edge = start
            while True:
                incident = open_incident(current_vertex)
                if len(incident) != 2:
                    break
                outgoing = [e for e in incident if e != previous_edge]
                if len(outgoing) != 1:
                    break
                next_edge = outgoing[0]
                if next_edge not in remaining:
                    break
                chain.add(next_edge)
                remaining.discard(next_edge)
                v1, v2 = next_edge
                next_vertex = v2 if v1 == current_vertex else v1
                previous_edge, current_vertex = next_edge, next_vertex
        chains.append(chain)
    return chains


def _face_normal(vertices: np.ndarray, faces: np.ndarray | None, face_index: int) -> np.ndarray | None:
    if faces is None or face_index < 0 or face_index >= len(faces):
        return None
    tri = faces[face_index]
    pts = vertices[np.asarray(tri[:3], dtype=np.int64)]
    normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    length = np.linalg.norm(normal)
    if length <= 1e-12:
        return None
    return normal / length


def _edge_feature_strength(vertices: np.ndarray, faces: np.ndarray | None, edge_faces: dict[EdgeKey, list[int]], edge_key: EdgeKey) -> float:
    fids = edge_faces.get(_edge_key(edge_key), [])
    if len(fids) <= 1:
        return np.pi
    if len(fids) != 2:
        return np.pi
    n0 = _face_normal(vertices, faces, fids[0])
    n1 = _face_normal(vertices, faces, fids[1])
    if n0 is None or n1 is None:
        return 0.0
    dot = float(np.clip(abs(np.dot(n0, n1)), 0.0, 1.0))
    return float(np.arccos(dot))


def _edge_length(vertices: np.ndarray, edge_key: EdgeKey) -> float:
    a, b = edge_key
    if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
        return 0.0
    return float(np.linalg.norm(vertices[a] - vertices[b]))


def _edge_midpoint(vertices: np.ndarray, edge_key: EdgeKey) -> np.ndarray:
    a, b = edge_key
    return 0.5 * (vertices[a] + vertices[b])


def _fit_local_ring_frame(vertices, edge_arr, *, seed_keys, open_keys, edge_faces, faces, start_key):
    if len(vertices) == 0 or len(edge_arr) == 0:
        return None
    start_mid = _edge_midpoint(vertices, start_key)
    lengths = np.asarray([_edge_length(vertices, _edge_key(edge)) for edge in edge_arr], dtype=float)
    positive = lengths[lengths > 1e-12]
    median_len = float(np.median(positive)) if positive.size else 1.0
    neighborhood_radius = max(4.0 * median_len, 1e-6)
    feature_threshold = np.deg2rad(14.0)

    candidate_keys = set(seed_keys)
    for edge in edge_arr:
        key = _edge_key(edge)
        if np.linalg.norm(_edge_midpoint(vertices, key) - start_mid) > neighborhood_radius:
            continue
        if key in open_keys or _edge_feature_strength(vertices, faces, edge_faces, key) >= feature_threshold:
            candidate_keys.add(key)

    points = []
    for key in candidate_keys:
        a, b = key
        if 0 <= a < len(vertices) and 0 <= b < len(vertices):
            points.append(vertices[a])
            points.append(vertices[b])
    if len(points) < 4:
        return None

    pts = np.asarray(points, dtype=float)
    center = pts.mean(axis=0)
    try:
        centered = pts - center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = np.asarray(vh[-1], dtype=float)
        normal_len = float(np.linalg.norm(normal))
        if normal_len <= 1e-12:
            return None
        normal = normal / normal_len
    except Exception:
        return None

    radial = pts - center
    plane_distance = np.abs(radial @ normal)
    radial_in_plane = radial - np.outer(radial @ normal, normal)
    radii = np.linalg.norm(radial_in_plane, axis=1)
    valid_radii = radii[radii > 1e-12]
    if valid_radii.size < 3:
        return None
    radius = float(np.median(valid_radii))
    if not np.isfinite(radius) or radius <= 1e-12:
        return None
    if float(np.median(plane_distance) / max(radius, 1e-12)) > 0.45:
        return None
    return _RingFrame(center=center, normal=normal, radius=radius, median_edge_length=median_len, sample_count=len(points))


def _ring_scores(vertices, edge_key, frame):
    if frame is None:
        return 0.0, 0.0
    mid = _edge_midpoint(vertices, edge_key)
    radial = mid - frame.center
    plane_error = abs(np.dot(radial, frame.normal))
    radial_in_plane = radial - np.dot(radial, frame.normal) * frame.normal
    r = np.linalg.norm(radial_in_plane)
    plane_score = 1.0 - min(plane_error / max(frame.median_edge_length * 2.5, 1e-12), 1.0)
    radius_score = 1.0 - min(abs(r - frame.radius) / max(frame.radius, frame.median_edge_length, 1e-12), 1.0)
    return float(plane_score), float(radius_score)


def _select_same_plane_open_paths(*, start_key, start_path, all_paths, vertices, open_keys, vertex_to_edges, faces, edge_faces, edge_arr):
    del vertex_to_edges
    frame = _fit_local_ring_frame(vertices, edge_arr,
                                  seed_keys=start_path,
                                  open_keys=open_keys,
                                  edge_faces=edge_faces,
                                  faces=faces,
                                  start_key=start_key)
    if frame is None:
        return set(start_path)
    selected = set(start_path)
    for path in all_paths:
        if path == start_path:
            continue
        plane_scores, radius_scores = [], []
        for ek in path:
            ps, rs = _ring_scores(vertices, ek, frame)
            plane_scores.append(ps)
            radius_scores.append(rs)
        if not plane_scores:
            continue
        avg_plane = sum(plane_scores) / len(plane_scores)
        avg_radius = sum(radius_scores) / len(radius_scores)
        if avg_plane > 0.75 and avg_radius > 0.75:
            selected.update(path)
    return selected


def _edge_direction_from_vertex(vertices, edge_key, current_vertex):
    a, b = edge_key
    if current_vertex == a:
        other = b
    elif current_vertex == b:
        other = a
    else:
        return None, None
    if other < 0 or other >= len(vertices) or current_vertex < 0 or current_vertex >= len(vertices):
        return None, None
    vec = vertices[other] - vertices[current_vertex]
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return None, None
    return vec / norm, int(other)


def _ring_tangent_score(vertices, frame, current_vertex, incoming, outgoing):
    if frame is None:
        return 0.0
    if current_vertex < 0 or current_vertex >= len(vertices):
        return 0.0
    radial = vertices[current_vertex] - frame.center
    axial = np.dot(radial, frame.normal)
    radial = radial - axial * frame.normal
    radial_len = np.linalg.norm(radial)
    if radial_len <= 1e-12:
        return 0.0
    tangent = np.cross(frame.normal, radial / radial_len)
    tangent_len = np.linalg.norm(tangent)
    if tangent_len <= 1e-12:
        return 0.0
    tangent = tangent / tangent_len
    if np.dot(tangent, incoming) < 0.0:
        tangent = -tangent
    return float(np.dot(outgoing, tangent))


def _choose_connected_continuation(vertices, faces, edge_faces, open_keys, vertex_to_edges, frame,
                                   previous_vertex, current_vertex, current_key, visited,
                                   start_strength, start_length):
    incoming = vertices[current_vertex] - vertices[previous_vertex]
    norm = np.linalg.norm(incoming)
    if norm <= 1e-12:
        return None
    incoming = incoming / norm
    feature_threshold = np.deg2rad(14.0)
    weak_feature_threshold = np.deg2rad(6.0)

    best = None
    best_score = -1e9
    ambiguous = False
    for candidate in vertex_to_edges.get(current_vertex, []):
        candidate = _edge_key(candidate)
        if candidate == current_key or candidate in visited:
            continue
        direction, other = _edge_direction_from_vertex(vertices, candidate, current_vertex)
        if direction is None or other is None or other == previous_vertex:
            continue
        direction_score = np.dot(incoming, direction)
        tangent_score = _ring_tangent_score(vertices, frame, current_vertex, incoming, direction)
        strength = _edge_feature_strength(vertices, faces, edge_faces, candidate)
        candidate_is_open = candidate in open_keys
        if start_strength >= feature_threshold and strength < weak_feature_threshold and not candidate_is_open:
            continue
        length = _edge_length(vertices, candidate)
        length_score = min(length, start_length) / max(length, start_length) if length > 1e-12 and start_length > 1e-12 else 0.0
        feature_score = min(strength, start_strength) / max(strength, start_strength, 1e-12) if start_strength >= feature_threshold else 0.0
        plane_score, radius_score = _ring_scores(vertices, candidate, frame)
        if frame is not None:
            if candidate_is_open:
                if plane_score < 0.12 or radius_score < 0.18:
                    continue
            else:
                if plane_score < 0.32 or radius_score < 0.38:
                    continue
        if direction_score < -0.35 and not candidate_is_open:
            continue
        if frame is not None and tangent_score < -0.20:
            continue
        score = (2.35 * direction_score + 2.15 * tangent_score + 1.15 * feature_score +
                 0.75 * plane_score + 0.95 * radius_score + 0.35 * length_score +
                 (0.45 if candidate_is_open else 0.0))
        if score > best_score + 1e-6:
            best_score = score
            best = (candidate, other)
            ambiguous = False
        elif abs(score - best_score) <= 1e-6:
            ambiguous = True
    min_score = 1.65 if frame is not None else (0.55 if start_strength >= feature_threshold else 0.95)
    if ambiguous or best is None or best_score < min_score:
        return None
    return best


def _choose_gap_bridge(vertices, faces, edge_faces, edge_arr, open_keys, frame,
                       previous_vertex, current_vertex, visited, start_strength, start_length, max_gap_edge_lengths):
    if frame is None:
        return None
    incoming = vertices[current_vertex] - vertices[previous_vertex]
    norm = np.linalg.norm(incoming)
    if norm <= 1e-12:
        return None
    incoming = incoming / norm
    local_length = max(frame.median_edge_length, start_length, 1e-12)
    max_gap = max_gap_edge_lengths * local_length
    feature_threshold = np.deg2rad(14.0)
    weak_feature_threshold = np.deg2rad(6.0)
    current_pos = vertices[current_vertex]

    best = None
    best_score = -1e9
    ambiguous = False
    for edge in edge_arr:
        key = _edge_key(edge)
        if key in visited:
            continue
        a, b = key
        if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
            continue
        dist_a = np.linalg.norm(vertices[a] - current_pos)
        dist_b = np.linalg.norm(vertices[b] - current_pos)
        if dist_a <= dist_b:
            landing, other, gap_dist = a, b, dist_a
        else:
            landing, other, gap_dist = b, a, dist_b
        if gap_dist <= 1e-12 or gap_dist > max_gap:
            continue
        outgoing = vertices[other] - vertices[landing]
        out_norm = np.linalg.norm(outgoing)
        if out_norm <= 1e-12:
            continue
        direction_score = np.dot(incoming, outgoing / out_norm)
        if direction_score < -0.15:
            continue
        strength = _edge_feature_strength(vertices, faces, edge_faces, key)
        candidate_is_open = key in open_keys
        if start_strength >= feature_threshold and strength < weak_feature_threshold and not candidate_is_open:
            continue
        plane_score, radius_score = _ring_scores(vertices, key, frame)
        if plane_score < 0.35 or radius_score < 0.35:
            continue
        gap_score = 1.0 - min(gap_dist / max_gap, 1.0)
        length = _edge_length(vertices, key)
        length_score = min(length, start_length) / max(length, start_length) if length > 1e-12 and start_length > 1e-12 else 0.0
        feature_score = min(strength, max(start_strength, feature_threshold)) / max(max(start_strength, feature_threshold), 1e-12)
        score = (2.1 * direction_score + 1.25 * plane_score + 1.35 * radius_score + 1.0 * gap_score +
                 0.45 * feature_score + 0.25 * length_score + (0.55 if candidate_is_open else 0.0))
        if score > best_score + 1e-6:
            best_score = score
            best = (key, landing, other)
            ambiguous = False
        elif abs(score - best_score) <= 1e-6:
            ambiguous = True
    if ambiguous or best is None or best_score < 3.15:
        return None
    return best


def _walk_feature_ring_direction(vertices, faces, edge_faces, edge_arr, open_keys, vertex_to_edges,
                                 frame, visited, start_key, current_key, previous_vertex, current_vertex,
                                 start_strength, start_length, max_gap_edge_lengths,
                                 allow_gap_bridge, max_selected_edges):
    current_key = _edge_key(current_key or start_key)
    start_prev_vertex = previous_vertex
    max_steps = max(1, len(edge_arr))
    bridge_count = 0
    max_bridges = max(1, min(8, len(edge_arr) // 8 + 1))
    for _ in range(max_steps):
        next_connected = _choose_connected_continuation(
            vertices, faces, edge_faces, open_keys, vertex_to_edges, frame,
            previous_vertex, current_vertex, current_key, visited,
            start_strength, start_length)
        if next_connected is not None:
            next_key, next_vertex = next_connected
            visited.add(next_key)
            if max_selected_edges is not None and len(visited) >= max_selected_edges:
                return
            if next_vertex == start_prev_vertex:
                return
            previous_vertex, current_vertex, current_key = current_vertex, next_vertex, next_key
            continue
        if not allow_gap_bridge:
            return
        if bridge_count >= max_bridges:
            return
        bridge = _choose_gap_bridge(
            vertices, faces, edge_faces, edge_arr, open_keys, frame,
            previous_vertex, current_vertex, visited,
            start_strength, start_length, max_gap_edge_lengths)
        if bridge is None:
            return
        bridge_count += 1
        next_key, landing_vertex, next_vertex = bridge
        visited.add(next_key)
        if max_selected_edges is not None and len(visited) >= max_selected_edges:
            return
        if next_vertex == start_prev_vertex:
            return
        previous_vertex, current_vertex, current_key = landing_vertex, next_vertex, next_key


def _seed_path_walk_starts(seed_keys, start_key):
    normalized = {_edge_key(e) for e in seed_keys}
    if not normalized:
        return ()
    adj = {}
    for e in normalized:
        a, b = e
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    # closed loop: no walking needed
    if adj and all(len(n) == 2 for n in adj.values()):
        return ()
    starts = []
    endpoints = [v for v, n in adj.items() if len(n) == 1]
    start_vertices = set(start_key)
    endpoints.sort(key=lambda v: (0 if v in start_vertices else 1, v))
    for ep in endpoints[:4]:
        neighbors = tuple(sorted(adj.get(ep, ())))
        if len(neighbors) != 1:
            continue
        prev = neighbors[0]
        cur_key = _edge_key((prev, ep))
        starts.append((prev, ep, cur_key))
    return tuple(starts)


def _classify_mode(start_key, open_keys, connected_open_count, selected_count, frame):
    if selected_count <= 1:
        return "single_edge"
    if start_key in open_keys and selected_count > connected_open_count:
        return "interrupted_open_boundary_ring"
    if start_key in open_keys:
        return "connected_open_boundary_component"
    if frame is not None:
        return "feature_ring_continuation"
    return "feature_chain_continuation"


def _estimate_confidence(mode, selected_count, connected_open_count, frame, start_strength):
    if selected_count <= 0:
        return 0.0
    if mode == "single_edge":
        return 0.45
    if mode == "connected_open_boundary_component":
        return 0.98
    if mode == "interrupted_open_boundary_ring":
        base = 0.82
        if connected_open_count >= 3:
            base += 0.08
        if frame is not None and frame.sample_count >= 8:
            base += 0.05
        return float(min(base, 0.96))
    if mode == "feature_ring_continuation":
        base = 0.68
        if frame is not None and frame.sample_count >= 8:
            base += 0.12
        if start_strength >= np.deg2rad(14.0):
            base += 0.08
        return float(min(base, 0.88))
    return 0.58




def _rim_resolver_bore_rim_from_safe_selection(
    *,
    vertices: np.ndarray,
    faces: np.ndarray | None,
    edge_arr: np.ndarray,
    edge_faces: Mapping[EdgeKey, list[int] | tuple[int, ...]] | None,
    selected_edge_ids: tuple[int, ...],
    start_edge_index: int,
    min_loop_edges: int,
    max_output_edges: int,
):
    """Run the active clicked-edge rim resolver.

    This helper deliberately lives in ``selection_edges.py`` because it is live
    selection plumbing, not Recognition.  The resolver itself remains in
    ``core.bore.rim_resolver`` and returns compact existing mesh edge IDs only.
    Its output is local rim evidence rooted at the clicked primitive.
    """

    if faces is None:
        return None
    try:
        from .bore.rim_resolver import resolve_bore_rim_edges_from_click_arrays
    except Exception:
        return None
    try:
        return resolve_bore_rim_edges_from_click_arrays(
            vertices=vertices,
            faces=faces,
            edge_index_to_vertices=edge_arr,
            edge_to_faces=edge_faces,
            selected_edge_ids=tuple(int(v) for v in tuple(selected_edge_ids or ())),
            start_edge_index=int(start_edge_index),
            min_loop_edges=int(min_loop_edges),
            max_output_edges=int(max_output_edges),
        )
    except Exception:
        return None


def _region_select_authoritative_bore_rim_from_edge_ids(
    *,
    vertices: np.ndarray,
    faces: np.ndarray | None,
    edge_arr: np.ndarray,
    edge_faces: Mapping[EdgeKey, list[int] | tuple[int, ...]] | None,
    selected_edge_ids: tuple[int, ...],
    min_loop_edges: int,
    require_reduction: bool = False,
) -> tuple[tuple[int, ...], dict[str, object]] | None:
    """Ask Region Select to normalize visual rim evidence when explicitly needed.

    This compatibility bridge is evidence-only.  It may ask Region Select for
    neutral opening/rim edge IDs so the live selection can display a complete
    rim, but the return value is still only selection-domain edge evidence.

    The primary ``bore_rim`` path must stay rooted at the clicked-edge resolver;
    broad visual clouds must not become feature or rebuild authority.
    """

    if faces is None:
        return None
    raw_ids = tuple(sorted({int(v) for v in tuple(selected_edge_ids or ()) if 0 <= int(v) < len(edge_arr)}))
    if not raw_ids:
        return None
    try:
        from .bore.region_select import normalize_opening_rim_edge_ids_from_arrays
    except Exception:
        return None
    try:
        rim_ids, diagnostics = normalize_opening_rim_edge_ids_from_arrays(
            vertices=vertices,
            faces=faces,
            edge_index_to_vertices=edge_arr,
            edge_to_faces=edge_faces,
            selected_edge_ids=raw_ids,
            min_loop_edges=int(min_loop_edges),
        )
    except Exception:
        return None
    rim_ids = tuple(sorted({int(v) for v in tuple(rim_ids or ()) if 0 <= int(v) < len(edge_arr)}))
    if len(rim_ids) < int(min_loop_edges):
        return None
    if require_reduction and len(raw_ids) >= int(min_loop_edges) and len(rim_ids) >= len(raw_ids):
        return None
    return rim_ids, {
        **dict(diagnostics or {}),
        "v119_live_visual_rim_normalization_used": True,
        "v119_visual_raw_edge_count": int(len(raw_ids)),
        "v119_visual_normalized_edge_count": int(len(rim_ids)),
        "v119_visual_selection_contract": "visual_selection_uses_region_select_neutral_rim_authority_not_broad_raw_cloud",
    }


def _region_select_authoritative_bore_rim(
    *,
    vertices: np.ndarray,
    faces: np.ndarray | None,
    edge_arr: np.ndarray,
    edge_faces: Mapping[EdgeKey, list[int] | tuple[int, ...]] | None,
    start_edge_index: int,
    min_loop_edges: int,
) -> tuple[tuple[int, ...], dict[str, object]] | None:
    """Delegate rim completion to Region Select as an evidence-only fallback.

    This is deliberately a small bridge.  The geometric meaning transform lives
    in ``far_mesh.core.bore.selection.region_select``.  ``selection_edges.py`` only uses
    the returned edge IDs for live selection display/tool input; it does not
    inherit Region Select authority and it does not classify the feature.
    """

    if start_edge_index < 0 or start_edge_index >= len(edge_arr):
        return None
    return _region_select_authoritative_bore_rim_from_edge_ids(
        vertices=vertices,
        faces=faces,
        edge_arr=edge_arr,
        edge_faces=edge_faces,
        selected_edge_ids=(int(start_edge_index),),
        min_loop_edges=int(min_loop_edges),
        require_reduction=False,
    )


def _orthonormal_basis_for_annular_rail_v173k(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = _unit_np(axis, np.array([0.0, 0.0, 1.0], dtype=float))
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(axis, ref))) > 0.90:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = _unit_np(np.cross(axis, ref), np.array([0.0, 1.0, 0.0], dtype=float))
    v = _unit_np(np.cross(axis, u), np.array([1.0, 0.0, 0.0], dtype=float))
    return u, v


def _measure_annular_rail_candidate_v173k(
    *,
    vertices: np.ndarray,
    edge_arr: np.ndarray,
    edge_ids: tuple[int, ...],
    start_edge_index: int,
    min_loop_edges: int,
    max_output_edges: int,
    source: str,
) -> tuple[bool, dict[str, object]]:
    """Validate a proposed live selection as neutral measured annular rail evidence.

    The rim resolver may propose a possible rim/rail from topology, but topology
    alone must not become live BoreTool input.  This fast measurement stage fits
    a plane and circle to the proposed edge vertices and rejects broad raw edge
    clouds before the viewport displays them or BoreTool receives them.  It does
    not classify BORE/CHAMFER/POCKET and does not authorize rebuild.
    """

    ids = tuple(sorted({int(v) for v in tuple(edge_ids or ()) if 0 <= int(v) < len(edge_arr)}))
    # v173l: max_output_edges is a live-display safety hint, not a semantic
    # limit on a real high-resolution rail.  Fine meshes may have hundreds of
    # small segments around one near-perfect circular opening; those must be
    # measured and accepted instead of forcing manual per-segment clicks.  Keep
    # an absolute sanity cap only to avoid validating pathological whole-mesh
    # selections.
    max_edges = max(int(max_output_edges), int(min_loop_edges), 1)
    absolute_measurement_cap_v173l = max(int(max_edges), 4096)
    diag: dict[str, object] = {
        "annular_rail_measurement_v173k": True,
        "annular_rail_measurement_source_v173k": str(source),
        "annular_rail_candidate_edge_count_v173k": int(len(ids)),
        "annular_rail_min_loop_edges_v173k": int(min_loop_edges),
        "annular_rail_max_output_edges_v173k": int(max_edges),
        "annular_rail_semantic_role_v173k": "measured_selected_annular_rail_evidence_not_feature_identity",
        "high_resolution_annular_rail_restore_v173l": True,
        "annular_rail_max_output_edges_is_display_hint_v173l": True,
        "annular_rail_absolute_measurement_cap_v173l": int(absolute_measurement_cap_v173l),
    }
    if len(ids) < int(min_loop_edges):
        diag.update({
            "annular_rail_measurement_valid_v173k": False,
            "annular_rail_reject_reason_v173k": "too_few_edges_for_annular_rail_measurement",
        })
        return False, diag
    if len(ids) > absolute_measurement_cap_v173l:
        diag.update({
            "annular_rail_measurement_valid_v173k": False,
            "annular_rail_reject_reason_v173k": "edge_count_exceeds_absolute_annular_rail_measurement_cap_v173l",
        })
        return False, diag

    keys = tuple(_edge_key(edge_arr[int(eid)]) for eid in ids)
    vertex_ids = sorted({int(v) for key in keys for v in key if 0 <= int(v) < len(vertices)})
    if len(vertex_ids) < max(4, int(min_loop_edges) // 2):
        diag.update({
            "annular_rail_measurement_valid_v173k": False,
            "annular_rail_reject_reason_v173k": "too_few_vertices_for_plane_circle_fit",
            "annular_rail_vertex_count_v173k": int(len(vertex_ids)),
        })
        return False, diag

    pts = np.asarray(vertices, dtype=float)[np.asarray(vertex_ids, dtype=np.int64), :3]
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < max(4, int(min_loop_edges) // 2):
        diag.update({
            "annular_rail_measurement_valid_v173k": False,
            "annular_rail_reject_reason_v173k": "non_finite_vertices_for_plane_circle_fit",
        })
        return False, diag

    center0 = pts.mean(axis=0)
    centered = pts - center0.reshape(1, 3)
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        axis = _unit_np(np.asarray(vh[-1], dtype=float), np.array([0.0, 0.0, 1.0], dtype=float))
    except Exception as exc:
        diag.update({
            "annular_rail_measurement_valid_v173k": False,
            "annular_rail_reject_reason_v173k": "plane_fit_failed",
            "annular_rail_error_v173k": str(exc),
        })
        return False, diag

    u_axis, v_axis = _orthonormal_basis_for_annular_rail_v173k(axis)
    xy = np.column_stack((centered @ u_axis, centered @ v_axis))
    try:
        mat = np.column_stack((xy[:, 0], xy[:, 1], np.ones(len(xy))))
        rhs = -(xy[:, 0] * xy[:, 0] + xy[:, 1] * xy[:, 1])
        sol, *_ = np.linalg.lstsq(mat, rhs, rcond=None)
        cx = -0.5 * float(sol[0])
        cy = -0.5 * float(sol[1])
        radius_sq = max(0.0, cx * cx + cy * cy - float(sol[2]))
        radius = float(np.sqrt(radius_sq))
        center = center0 + cx * u_axis + cy * v_axis
    except Exception as exc:
        diag.update({
            "annular_rail_measurement_valid_v173k": False,
            "annular_rail_reject_reason_v173k": "circle_fit_failed",
            "annular_rail_error_v173k": str(exc),
        })
        return False, diag
    if not np.isfinite(radius) or radius <= 1.0e-12:
        diag.update({
            "annular_rail_measurement_valid_v173k": False,
            "annular_rail_reject_reason_v173k": "invalid_circle_radius",
            "annular_rail_radius_v173k": float(radius) if np.isfinite(radius) else 0.0,
        })
        return False, diag

    rel = pts - center.reshape(1, 3)
    axial = rel @ axis.reshape(3)
    radial_vec = rel - np.outer(axial, axis)
    radii = np.linalg.norm(radial_vec, axis=1)
    radii = radii[np.isfinite(radii)]
    if radii.size == 0:
        diag.update({
            "annular_rail_measurement_valid_v173k": False,
            "annular_rail_reject_reason_v173k": "no_valid_radial_samples",
        })
        return False, diag

    edge_lengths = [float(_edge_length(vertices, key)) for key in keys]
    edge_lengths = [v for v in edge_lengths if np.isfinite(v) and v > 1.0e-12]
    median_edge_length = float(np.median(np.asarray(edge_lengths, dtype=float))) if edge_lengths else 1.0

    plane_rms = float(np.sqrt(np.mean(axial * axial))) if len(axial) else 999999.0
    radius_rms = float(np.sqrt(np.mean((radii - radius) ** 2))) if len(radii) else 999999.0
    plane_rel_rms = float(plane_rms / max(float(radius), 1.0e-9))
    radius_rel_rms = float(radius_rms / max(float(radius), 1.0e-9))
    radius_min = float(np.percentile(radii, 5.0 if len(radii) >= 8 else 0.0))
    radius_max = float(np.percentile(radii, 95.0 if len(radii) >= 8 else 100.0))
    radial_spread = float(max(0.0, radius_max - radius_min))
    radial_spread_ratio = float(radial_spread / max(float(radius), 1.0e-9))

    mid_angles: list[float] = []
    for key in keys:
        mid = _edge_midpoint(vertices, key)
        mrel = mid - center
        ma = float(np.dot(mrel, axis))
        mr = mrel - ma * axis
        mx = float(np.dot(mr, u_axis))
        my = float(np.dot(mr, v_axis))
        md = float(np.hypot(mx, my))
        if np.isfinite(md) and md > 1.0e-12:
            mid_angles.append(float(np.arctan2(my, mx)))
    if mid_angles:
        angles = np.sort(np.asarray(mid_angles, dtype=float))
        gaps = np.diff(np.concatenate([angles, angles[:1] + 2.0 * np.pi]))
        max_gap = float(np.max(gaps)) if len(gaps) else 2.0 * np.pi
        angular_coverage = float(max(0.0, min(1.0, 1.0 - max_gap / (2.0 * np.pi))))
        max_gap_degrees = float(np.degrees(max_gap))
    else:
        angular_coverage = 0.0
        max_gap_degrees = 360.0

    degree: dict[int, int] = {}
    for a, b in keys:
        degree[int(a)] = degree.get(int(a), 0) + 1
        degree[int(b)] = degree.get(int(b), 0) + 1
    vertex_count = int(len(degree))
    degree_two_count = int(sum(1 for value in degree.values() if int(value) == 2))
    branch_vertex_count = int(sum(1 for value in degree.values() if int(value) > 2))
    degree_two_ratio = float(degree_two_count / max(vertex_count, 1))
    closed_loop_quality = float(min(1.0, degree_two_ratio + 0.25 * angular_coverage))
    clicked_in_candidate = bool(int(start_edge_index) in set(ids))
    circularity = float(max(0.0, 1.0 - min(radius_rel_rms * 4.0 + plane_rel_rms * 2.0 + max(0.0, radial_spread_ratio - 0.10) * 1.25, 1.0)))

    accepted = bool(
        clicked_in_candidate
        and len(ids) >= int(min_loop_edges)
        and radius > max(1.15 * median_edge_length, 1.0e-9)
        and plane_rel_rms <= 0.16
        and (radius_rel_rms <= 0.13 or radial_spread_ratio <= 0.26)
        and angular_coverage >= 0.33
        and branch_vertex_count <= max(2, int(0.10 * vertex_count))
        and degree_two_ratio >= 0.42
        and circularity >= 0.40
    )
    reasons: list[str] = []
    if not clicked_in_candidate:
        reasons.append("candidate_does_not_contain_clicked_edge")
    if radius <= max(1.15 * median_edge_length, 1.0e-9):
        reasons.append("radius_too_small_for_edge_scale")
    if plane_rel_rms > 0.16:
        reasons.append("plane_fit_residual_too_high")
    if radius_rel_rms > 0.13 and radial_spread_ratio > 0.26:
        reasons.append("circle_fit_radius_scatter_too_high")
    if angular_coverage < 0.33:
        reasons.append("angular_coverage_too_low")
    if branch_vertex_count > max(2, int(0.10 * vertex_count)):
        reasons.append("branching_topology_cloud_not_single_rail")
    if degree_two_ratio < 0.42:
        reasons.append("edge_graph_not_loop_like")
    if circularity < 0.40:
        reasons.append("circle_confidence_too_low")

    diag.update({
        "annular_rail_measurement_valid_v173k": bool(accepted),
        "annular_rail_reject_reasons_v173k": tuple(reasons),
        "annular_rail_reject_reason_v173k": reasons[0] if reasons else "",
        "annular_rail_vertex_count_v173k": int(vertex_count),
        "annular_rail_degree_two_ratio_v173k": float(degree_two_ratio),
        "annular_rail_branch_vertex_count_v173k": int(branch_vertex_count),
        "annular_rail_closed_loop_quality_v173k": float(closed_loop_quality),
        "annular_rail_radius_v173k": float(radius),
        "annular_rail_diameter_v173k": float(2.0 * radius),
        "annular_rail_radius_min_v173k": float(radius_min),
        "annular_rail_radius_max_v173k": float(radius_max),
        "annular_rail_radial_spread_v173k": float(radial_spread),
        "annular_rail_radial_spread_ratio_v173k": float(radial_spread_ratio),
        "annular_rail_plane_fit_rms_v173k": float(plane_rms),
        "annular_rail_plane_fit_rel_rms_v173k": float(plane_rel_rms),
        "annular_rail_circle_fit_rms_v173k": float(radius_rms),
        "annular_rail_circle_fit_rel_rms_v173k": float(radius_rel_rms),
        "annular_rail_angular_coverage_v173k": float(angular_coverage),
        "annular_rail_max_gap_degrees_v173k": float(max_gap_degrees),
        "annular_rail_circularity_v173k": float(circularity),
        "annular_rail_median_edge_length_v173k": float(median_edge_length),
        "annular_rail_center_v173k": tuple(float(v) for v in center),
        "annular_rail_axis_v173k": tuple(float(v) for v in axis),
        "annular_rail_clicked_edge_in_candidate_v173k": bool(clicked_in_candidate),
    })
    return bool(accepted), diag


# ----------------------------------------------------------------------
# Dedicated Bore opening selection helpers
# ----------------------------------------------------------------------

def _select_bore_opening_edge_region(
    *,
    vertices: np.ndarray,
    faces: np.ndarray | None,
    edge_index_to_vertices: np.ndarray,
    edge_to_faces: Mapping[EdgeKey, list[int] | tuple[int, ...]] | None,
    open_edges: np.ndarray | None,
    start_edge_index: int,
    max_gap_edge_lengths: float = 3.25,
    max_selected_edges: int | None = None,
) -> EdgeSelectionRegion:
    """Return one local opening/rim evidence set from a clicked edge.

    This mode deliberately does *not* enable the old broad same-plane sibling
    sweep.  It first gets conservative local selection evidence, keeps the
    clicked edge as the navigation root, tries the compact rim resolver, and
    accepts only a local loop or bounded local handoff.  If no validated loop is
    found it falls back to the conservative result so the UI still shows what
    was clicked.
    """

    vertices_arr = _as_vertices(vertices)
    edge_arr = _as_edges(edge_index_to_vertices)
    if edge_arr.size == 0:
        return EdgeSelectionRegion((), 0.0, "empty", {"reason": "no_edges", "strategy": "bore_rim"})
    if start_edge_index < 0 or start_edge_index >= len(edge_arr):
        return EdgeSelectionRegion((), 0.0, "invalid_start_edge", {"start_edge_index": int(start_edge_index), "strategy": "bore_rim"})

    faces_arr = _as_faces_or_none(faces)
    edge_faces = _normalize_edge_faces(edge_to_faces)
    open_keys = _open_edge_keys(open_edges)
    key_to_index = _edge_key_to_index(edge_arr)
    vertex_to_edges = _vertex_to_edges(edge_arr)
    start_key = _edge_key(edge_arr[int(start_edge_index)])

    # Do NOT ask Region Select/global rim normalization before the clicked-edge
    # resolver.  Region Select operates on neutral evidence after selection;
    # using it here would let an already-contaminated coarse edge cloud become
    # practical selection authority.  Live Bore selection must first root
    # navigation at the exact clicked edge.

    # Conservative first pass: gives us the clicked local feature/open path.
    safe = select_edge_region(
        vertices=vertices_arr,
        faces=faces_arr,
        edge_index_to_vertices=edge_arr,
        edge_to_faces=edge_faces,
        open_edges=np.asarray(tuple(open_keys), dtype=np.int64) if open_keys else None,
        start_edge_index=int(start_edge_index),
        max_gap_edge_lengths=max_gap_edge_lengths,
        strategy="safe",
        allow_same_plane_siblings=False,
        allow_gap_bridge=False,
        max_selected_edges=max_selected_edges,
    )
    safe_keys = {_edge_key(edge_arr[int(eid)]) for eid in safe.edge_ids if 0 <= int(eid) < len(edge_arr)}
    if not safe_keys:
        safe_keys = {start_key}

    # v173j live-selection invariant: the conservative pass is evidence for the
    # clicked-edge resolver, never final Bore Opening display authority when it
    # has expanded into a broad cloud.  This prevents the old pink-cloud
    # regression from returning through safe/no-frame/no-cycle fallbacks.
    live_rim_cap = (
        int(max_selected_edges)
        if max_selected_edges is not None and int(max_selected_edges) > 0
        else 96
    )
    safe_is_broad_cloud = bool(len(safe.edge_ids) > live_rim_cap)

    def _validated_annular_rail(edge_ids: tuple[int, ...], source: str, min_edges: int = 8) -> tuple[bool, dict[str, object]]:
        return _measure_annular_rail_candidate_v173k(
            vertices=vertices_arr,
            edge_arr=edge_arr,
            edge_ids=tuple(int(v) for v in tuple(edge_ids or ())),
            start_edge_index=int(start_edge_index),
            min_loop_edges=int(min_edges),
            max_output_edges=int(live_rim_cap),
            source=str(source),
        )

    def _compact_bore_rim_fallback(reason: str, extra: dict[str, object] | None = None) -> EdgeSelectionRegion:
        """Return a compact clicked-edge-rooted fallback, never the raw cloud."""
        # First try Region Select's neutral rim normalizer from the clicked edge
        # only.  Do not feed it the broad safe cloud.  This preserves the clicked
        # primitive as selection authority while still allowing a complete visual
        # rim when the local resolver could not form one.
        rs = _region_select_authoritative_bore_rim(
            vertices=vertices_arr,
            faces=faces_arr,
            edge_arr=edge_arr,
            edge_faces=edge_faces,
            start_edge_index=int(start_edge_index),
            min_loop_edges=8,
        )
        if rs is not None:
            rim_ids, rs_diag = rs
            rim_ids = tuple(sorted({int(v) for v in rim_ids if 0 <= int(v) < len(edge_arr)}))
            if 8 <= len(rim_ids) <= live_rim_cap:
                measured_ok, measure_diag = _validated_annular_rail(rim_ids, "v173j_region_select_clicked_edge_rim_fallback", min_edges=8)
                if measured_ok:
                    return EdgeSelectionRegion(
                        edge_ids=rim_ids,
                        confidence=0.62,
                        mode="bore_opening_clicked_edge_region_select_measured_annular_rail_fallback",
                        diagnostics={
                            **dict(rs_diag or {}),
                            **dict(measure_diag),
                            **dict(extra or {}),
                            "strategy": "bore_rim",
                            "bore_opening_source": "v173k_clicked_edge_region_select_measured_annular_rail_fallback",
                            "clicked_edge_authority": True,
                            "region_select_authority": "clicked_edge_only_neutral_rim_fallback_after_annular_measurement",
                            "raw_safe_cloud_suppressed_v173j": bool(safe_is_broad_cloud),
                            "safe_selected_edge_count": int(len(safe.edge_ids)),
                            "selected_edge_count": int(len(rim_ids)),
                            "fallback_reason": str(reason),
                            "pink_cloud_regression_guard_v173j": True,
                            "annular_rail_measurement_gate_v173k": True,
                        },
                    )
        return EdgeSelectionRegion(
            edge_ids=(int(start_edge_index),),
            confidence=0.35,
            mode="bore_opening_clicked_edge_only_unresolved",
            diagnostics={
                **dict(extra or {}),
                "strategy": "bore_rim",
                "bore_opening_source": "v173j_clicked_edge_only_no_broad_cloud_fallback",
                "clicked_edge_authority": True,
                "region_select_authority": False,
                "raw_safe_cloud_suppressed_v173j": bool(safe_is_broad_cloud),
                "safe_selected_edge_count": int(len(safe.edge_ids)),
                "selected_edge_count": 1,
                "fallback_reason": str(reason),
                "validated_loop_found": False,
                "pink_cloud_regression_guard_v173j": True,
            },
        )

    # The active auto-rim selection authority is the clicked-edge resolver.  It
    # sees the conservative cloud as evidence, but start_edge_index remains the
    # navigation root: clicked edge -> local rim evidence, not broad cloud ->
    # best global circle.
    resolver_result = _rim_resolver_bore_rim_from_safe_selection(
        vertices=vertices_arr,
        faces=faces_arr,
        edge_arr=edge_arr,
        edge_faces=edge_faces,
        selected_edge_ids=tuple(int(v) for v in safe.edge_ids),
        start_edge_index=int(start_edge_index),
        min_loop_edges=8,
        max_output_edges=(
            max(int(max_selected_edges), 768)
            if max_selected_edges is not None and int(max_selected_edges) > 0
            else 768
        ),
    )
    resolver_reject_diag: dict[str, object] = {}
    if resolver_result is not None and len(resolver_result.edge_ids) >= 8:
        resolver_edge_ids = tuple(sorted(int(v) for v in resolver_result.edge_ids))
        measured_ok, measure_diag = _validated_annular_rail(resolver_edge_ids, "clicked_edge_rim_resolver_output", min_edges=8)
        if measured_ok:
            return EdgeSelectionRegion(
                edge_ids=resolver_edge_ids,
                confidence=float(min(0.96, max(float(resolver_result.confidence), 0.70))),
                mode="bore_opening_clicked_edge_measured_annular_rail",
                diagnostics={
                    **dict(resolver_result.diagnostics or {}),
                    **dict(measure_diag),
                    "strategy": "bore_rim",
                    "bore_opening_source": "v173k_clicked_edge_rim_resolver_measured_annular_rail",
                    "region_select_authority": False,
                    "clicked_edge_authority": True,
                    "start_edge_index": int(start_edge_index),
                    "safe_selected_edge_count": int(len(safe.edge_ids)),
                    "selected_edge_count": int(len(resolver_edge_ids)),
                    "validated_loop_found": True,
                    "annular_rail_measurement_gate_v173k": True,
                },
            )
        resolver_reject_diag = {
            "rim_resolver_output_rejected_by_annular_measurement_v173k": True,
            **dict(measure_diag),
        }

    # If the conservative pass already produced one real loop, keep it.
    safe_cycles = _bore_degree_two_cycles(safe_keys, min_loop_edges=12)
    if safe_cycles:
        best = _choose_best_bore_cycle(
            cycles=safe_cycles,
            vertices=vertices_arr,
            frame=None,
            start_key=start_key,
            key_to_index=key_to_index,
        )
        if best:
            cycle_keys, score = best
            edge_ids = tuple(sorted(int(key_to_index[k]) for k in cycle_keys if k in key_to_index))
            measured_ok, measure_diag = _validated_annular_rail(edge_ids, "safe_degree_two_loop", min_edges=8)
            if measured_ok:
                return EdgeSelectionRegion(
                    edge_ids=edge_ids,
                    confidence=float(min(0.96, 0.72 + 0.02 * min(len(edge_ids), 10))),
                    mode="bore_opening_measured_annular_rail_loop",
                    diagnostics={
                        **dict(measure_diag),
                        "strategy": "bore_rim",
                        "bore_opening_source": "v173k_safe_degree_two_measured_annular_rail",
                        "selected_edge_count": len(edge_ids),
                        "candidate_loop_count": len(safe_cycles),
                        "score": float(score),
                        "safe_selected_edge_count": len(safe.edge_ids),
                        "annular_rail_measurement_gate_v173k": True,
                    },
                )

    frame = _fit_bore_opening_frame(
        vertices=vertices_arr,
        edge_arr=edge_arr,
        seed_keys=safe_keys,
        open_keys=open_keys,
        edge_faces=edge_faces,
        faces=faces_arr,
        start_key=start_key,
    )

    if frame is None or not np.isfinite(frame.radius) or frame.radius <= 1.0e-12:
        return _compact_bore_rim_fallback(
            "safe_fallback_no_frame",
            {
                **safe.diagnostics,
                "validated_loop_found": False,
                "safe_selected_edge_count": len(safe.edge_ids),
            },
        )

    candidate_keys = _bore_opening_candidate_edges(
        vertices=vertices_arr,
        faces=faces_arr,
        edge_arr=edge_arr,
        edge_faces=edge_faces,
        open_keys=open_keys,
        frame=frame,
        start_key=start_key,
        seed_keys=safe_keys,
    )

    high_resolution_candidate_cap_v173l = (
        max(int(max_selected_edges), 768)
        if max_selected_edges is not None and int(max_selected_edges) > 0
        else 768
    )
    if len(candidate_keys) > int(high_resolution_candidate_cap_v173l):
        # Candidate graph is too large for live selection.  This is a topology-cloud
        # safety cap, not the old 96-edge semantic cap.  High-resolution real
        # rings up to the cap are still measured as annular rail evidence.
        return _compact_bore_rim_fallback(
            "safe_fallback_candidate_cap",
            {
                **safe.diagnostics,
                "validated_loop_found": False,
                "candidate_edge_count": len(candidate_keys),
                "max_selected_edges": int(max_selected_edges) if max_selected_edges is not None else None,
                "high_resolution_candidate_cap_v173l": int(high_resolution_candidate_cap_v173l),
                "high_resolution_annular_rail_restore_v173l": True,
            },
        )

    # First try bounded, cheap cycle extraction only.  Phase 6 deliberately
    # removes the expensive anchor path search from edge selection.  The Bore
    # folder now owns damaged-rim measurement, so selection only needs to hand
    # over a local rim-evidence edge set without stalling the UI.
    cycles = _bore_degree_two_cycles(candidate_keys, min_loop_edges=12)
    cycle_source = "strict_degree_two"

    if not cycles:
        cycles = _bore_ring_pruned_cycles(
            candidate_keys,
            vertices=vertices_arr,
            frame=frame,
            start_key=start_key,
            min_loop_edges=12,
        )
        cycle_source = "ring_pruned_degree_two"

    best = _choose_best_bore_cycle(
        cycles=cycles,
        vertices=vertices_arr,
        frame=frame,
        start_key=start_key,
        key_to_index=key_to_index,
    )

    if best:
        cycle_keys, score = best
        edge_ids = tuple(sorted(int(key_to_index[k]) for k in cycle_keys if k in key_to_index))
        measured_ok, measure_diag = _validated_annular_rail(edge_ids, "local_frame_cycle", min_edges=8)
        if measured_ok:
            return EdgeSelectionRegion(
                edge_ids=edge_ids,
                confidence=float(min(0.94, max(0.68, score))),
                mode="bore_opening_measured_annular_rail_loop",
                diagnostics={
                    **dict(measure_diag),
                    "strategy": "bore_rim",
                    "bore_opening_source": "v173k_local_frame_measured_annular_rail_cycle",
                    "bore_opening_cycle_source": cycle_source,
                    "expensive_anchor_cycle_search_enabled": False,
                    "validated_loop_found": True,
                    "selected_edge_count": len(edge_ids),
                    "candidate_edge_count": len(candidate_keys),
                    "candidate_loop_count": len(cycles),
                    "safe_selected_edge_count": len(safe.edge_ids),
                    "ring_frame_available": True,
                    "frame_radius": float(frame.radius),
                    "frame_sample_count": int(frame.sample_count),
                    "score": float(score),
                    "annular_rail_measurement_gate_v173k": True,
                },
            )

    # Damaged/messy meshes may not contain one topological cycle, but they can
    # still contain enough local rim evidence for core.bore.measure to fit the
    # opening.  Return a bounded candidate set instead of searching indefinitely
    # for a perfect cycle.
    handoff_keys = _bore_ranked_candidate_handoff(
        candidate_keys,
        vertices=vertices_arr,
        frame=frame,
        start_key=start_key,
        min_edges=12,
        max_edges=(max(int(max_selected_edges), 768) if max_selected_edges is not None and int(max_selected_edges) > 0 else 768),
    )
    if len(handoff_keys) >= 12:
        edge_ids = tuple(sorted(int(key_to_index[k]) for k in handoff_keys if k in key_to_index))
        measured_ok, measure_diag = _validated_annular_rail(edge_ids, "bounded_local_candidate_handoff", min_edges=8)
        if measured_ok:
            # Do not pass this handoff cloud back through Region Select for visual
            # normalization.  If the clicked-edge resolver did not accept it, global
            # normalization is exactly the path that can jump to unrelated coarse
            # mesh features.
            return EdgeSelectionRegion(
                edge_ids=edge_ids,
                confidence=0.56,
                mode="bore_opening_measured_annular_rail_handoff",
                diagnostics={
                    **dict(measure_diag),
                    "strategy": "bore_rim",
                    "bore_opening_source": "v173k_bounded_local_measured_annular_rail_handoff",
                    "validated_loop_found": True,
                    "measurement_handoff": True,
                    "expensive_anchor_cycle_search_enabled": False,
                    "selected_edge_count": len(edge_ids),
                    "candidate_edge_count": len(candidate_keys),
                    "candidate_loop_count": len(cycles),
                    "safe_selected_edge_count": len(safe.edge_ids),
                    "ring_frame_available": True,
                    "frame_radius": float(frame.radius),
                    "frame_sample_count": int(frame.sample_count),
                    "annular_rail_measurement_gate_v173k": True,
                },
            )

    return _compact_bore_rim_fallback(
        "safe_fallback_no_valid_cycle",
        {
            **safe.diagnostics,
            **resolver_reject_diag,
            "validated_loop_found": False,
            "measurement_handoff": False,
            "expensive_anchor_cycle_search_enabled": False,
            "candidate_edge_count": len(candidate_keys),
            "candidate_loop_count": len(cycles),
            "safe_selected_edge_count": len(safe.edge_ids),
            "ring_frame_available": True,
            "frame_radius": float(frame.radius),
            "annular_rail_measurement_gate_v173k": True,
        },
    )


def _fit_bore_opening_frame(
    *,
    vertices: np.ndarray,
    edge_arr: np.ndarray,
    seed_keys: set[EdgeKey],
    open_keys: set[EdgeKey],
    edge_faces: dict[EdgeKey, list[int]],
    faces: np.ndarray | None,
    start_key: EdgeKey,
) -> _RingFrame | None:
    """Fit a local circular frame from the conservative clicked-edge path."""

    seed_points: list[np.ndarray] = []
    for key in seed_keys:
        a, b = key
        if 0 <= a < len(vertices) and 0 <= b < len(vertices):
            seed_points.append(vertices[a])
            seed_points.append(vertices[b])

    if len(seed_points) < 6:
        return _fit_local_ring_frame(
            vertices,
            edge_arr,
            seed_keys=seed_keys,
            open_keys=open_keys,
            edge_faces=edge_faces,
            faces=faces,
            start_key=start_key,
        )

    pts = np.asarray(seed_points, dtype=float)
    center0 = pts.mean(axis=0)
    centered = pts - center0
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        normal = np.asarray(vh[-1], dtype=float)
        normal = _unit_np(normal, np.array([0.0, 0.0, 1.0], dtype=float))
    except Exception:
        return _fit_local_ring_frame(
            vertices,
            edge_arr,
            seed_keys=seed_keys,
            open_keys=open_keys,
            edge_faces=edge_faces,
            faces=faces,
            start_key=start_key,
        )

    # Build an orthonormal basis in the fitted plane.
    reference = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    u = _unit_np(np.cross(normal, reference), np.array([0.0, 1.0, 0.0], dtype=float))
    v = _unit_np(np.cross(normal, u), np.array([1.0, 0.0, 0.0], dtype=float))

    xy = np.column_stack(((pts - center0) @ u, (pts - center0) @ v))
    # Least-squares circle: x^2 + y^2 + A*x + B*y + C = 0.
    try:
        mat = np.column_stack((xy[:, 0], xy[:, 1], np.ones(len(xy))))
        rhs = -(xy[:, 0] * xy[:, 0] + xy[:, 1] * xy[:, 1])
        sol, *_ = np.linalg.lstsq(mat, rhs, rcond=None)
        cx = -0.5 * float(sol[0])
        cy = -0.5 * float(sol[1])
        radius_sq = max(0.0, cx * cx + cy * cy - float(sol[2]))
        radius = float(np.sqrt(radius_sq))
        center = center0 + cx * u + cy * v
    except Exception:
        center = center0
        radial = centered - np.outer(centered @ normal, normal)
        radii = np.linalg.norm(radial, axis=1)
        positive = radii[radii > 1.0e-12]
        radius = float(np.median(positive)) if len(positive) else 0.0

    lengths = np.asarray([_edge_length(vertices, _edge_key(edge)) for edge in edge_arr], dtype=float)
    positive_lengths = lengths[lengths > 1.0e-12]
    median_len = float(np.median(positive_lengths)) if positive_lengths.size else 1.0
    if not np.isfinite(radius) or radius <= median_len * 1.5:
        # Circle fit can collapse on very short arcs; fall back to the older local frame.
        old = _fit_local_ring_frame(
            vertices,
            edge_arr,
            seed_keys=seed_keys,
            open_keys=open_keys,
            edge_faces=edge_faces,
            faces=faces,
            start_key=start_key,
        )
        return old

    return _RingFrame(center=center.astype(float), normal=normal.astype(float), radius=float(radius), median_edge_length=median_len, sample_count=int(len(pts)))


def _bore_opening_candidate_edges(
    *,
    vertices: np.ndarray,
    faces: np.ndarray | None,
    edge_arr: np.ndarray,
    edge_faces: dict[EdgeKey, list[int]],
    open_keys: set[EdgeKey],
    frame: _RingFrame,
    start_key: EdgeKey,
    seed_keys: set[EdgeKey],
) -> set[EdgeKey]:
    feature_threshold = float(np.deg2rad(6.0))
    max_plane_error = max(frame.median_edge_length * 2.75, frame.radius * 0.16)
    max_radius_error = max(frame.median_edge_length * 3.25, frame.radius * 0.28)
    max_center_dist = frame.radius * 1.55 + frame.median_edge_length * 4.0
    max_edge_length = max(frame.median_edge_length * 8.0, frame.radius * 0.45)

    candidate: set[EdgeKey] = set()
    for edge in edge_arr:
        key = _edge_key(edge)
        a, b = key
        if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
            continue
        mid = _edge_midpoint(vertices, key)
        radial = mid - frame.center
        plane_error = abs(float(np.dot(radial, frame.normal)))
        radial_in_plane = radial - np.dot(radial, frame.normal) * frame.normal
        radial_len = float(np.linalg.norm(radial_in_plane))
        radius_error = abs(radial_len - frame.radius)
        center_dist = float(np.linalg.norm(radial_in_plane))
        if plane_error > max_plane_error or radius_error > max_radius_error or center_dist > max_center_dist:
            continue
        length = _edge_length(vertices, key)
        if length <= 1.0e-12 or length > max_edge_length:
            continue
        strength = _edge_feature_strength(vertices, faces, edge_faces, key)
        is_boundary_or_feature = key in open_keys or strength >= feature_threshold or key in seed_keys
        if not is_boundary_or_feature:
            continue
        tangent = _edge_tangent_alignment(vertices, key, frame)
        if key not in seed_keys and tangent < 0.38:
            continue
        candidate.add(key)

    # Keep the conservative clicked path even if some individual edge scores are weak.
    candidate.update(seed_keys)
    candidate.add(start_key)
    return candidate


def _edge_tangent_alignment(vertices: np.ndarray, edge_key: EdgeKey, frame: _RingFrame | None) -> float:
    if frame is None:
        return 0.0
    a, b = edge_key
    if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
        return 0.0
    edge_vec = vertices[b] - vertices[a]
    edge_len = float(np.linalg.norm(edge_vec))
    if edge_len <= 1.0e-12:
        return 0.0
    edge_dir = edge_vec / edge_len
    mid = _edge_midpoint(vertices, edge_key)
    radial = mid - frame.center
    radial = radial - np.dot(radial, frame.normal) * frame.normal
    radial_len = float(np.linalg.norm(radial))
    if radial_len <= 1.0e-12:
        return 0.0
    tangent = np.cross(frame.normal, radial / radial_len)
    tangent_len = float(np.linalg.norm(tangent))
    if tangent_len <= 1.0e-12:
        return 0.0
    tangent = tangent / tangent_len
    return float(abs(np.dot(edge_dir, tangent)))




def _bore_ring_pruned_cycles(
    edges: set[EdgeKey],
    *,
    vertices: np.ndarray,
    frame: _RingFrame | None,
    start_key: EdgeKey,
    min_loop_edges: int,
) -> list[set[EdgeKey]]:
    """Extract rim-like cycles from a branchy local candidate graph.

    Around a messy bore opening the local graph may contain a valid rim plus
    cap fan/spoke branches.  A strict degree-2 test rejects the whole component.
    This helper keeps only the two strongest ring-like incident edges per vertex
    and then runs the same degree-2 extraction on the reduced graph.
    """

    normalized = {_edge_key(edge) for edge in edges}
    if len(normalized) < int(min_loop_edges):
        return []

    incident: dict[int, list[tuple[float, EdgeKey]]] = {}
    for key in normalized:
        score = _bore_edge_rim_score(vertices, key, frame, start_key)
        a, b = key
        incident.setdefault(int(a), []).append((score, key))
        incident.setdefault(int(b), []).append((score, key))

    kept: set[EdgeKey] = set()
    start_vertices = set(start_key)
    for vertex, scored in incident.items():
        # At the clicked vertices keep the clicked edge, then the best partner.
        scored_sorted = sorted(
            scored,
            key=lambda item: (item[1] == start_key, item[0]),
            reverse=True,
        )
        limit = 2
        for _score, key in scored_sorted[:limit]:
            kept.add(key)

    # A single pass of top-2 pruning can still leave a short side cycle.  Run a
    # second pass after tail pruning so branch vertices are forced toward degree 2.
    kept = _prune_edge_tails(kept)
    if len(kept) < int(min_loop_edges):
        return []

    incident2: dict[int, list[tuple[float, EdgeKey]]] = {}
    for key in kept:
        score = _bore_edge_rim_score(vertices, key, frame, start_key)
        a, b = key
        incident2.setdefault(int(a), []).append((score, key))
        incident2.setdefault(int(b), []).append((score, key))

    kept2: set[EdgeKey] = set()
    for _vertex, scored in incident2.items():
        for _score, key in sorted(scored, key=lambda item: item[0], reverse=True)[:2]:
            kept2.add(key)

    return _bore_degree_two_cycles(kept2, min_loop_edges=int(min_loop_edges))



def _bore_ranked_candidate_handoff(
    edges: set[EdgeKey],
    *,
    vertices: np.ndarray,
    frame: _RingFrame | None,
    start_key: EdgeKey,
    min_edges: int,
    max_edges: int,
) -> set[EdgeKey]:
    """Return a bounded local rim-evidence set for Bore measurement.

    This is intentionally not a cycle finder.  It ranks already-local candidate
    edges by rim-likeness and proximity to the clicked edge, keeps a compact
    radial band around the strongest evidence, and hands that evidence to the
    Bore measurement layer.  This keeps edge selection fast and moves damaged
    topology interpretation into ``far_mesh.core.bore``.
    """

    normalized = {_edge_key(edge) for edge in edges}
    if len(normalized) < int(min_edges):
        return set(normalized)

    max_edges = max(int(min_edges), int(max_edges))
    start_mid = _edge_midpoint(vertices, start_key)

    scored: list[tuple[float, float, EdgeKey]] = []
    radial_values: list[float] = []
    radial_by_key: dict[EdgeKey, float] = {}
    for key in normalized:
        mid = _edge_midpoint(vertices, key)
        proximity = 1.0 / (1.0 + float(np.linalg.norm(mid - start_mid)) / max((frame.radius if frame is not None else 1.0), 1.0e-9))
        rim_score = _bore_edge_rim_score(vertices, key, frame, start_key)
        radial_error = 0.0
        radial_len = 0.0
        if frame is not None:
            rel = mid - frame.center
            rel_plane = rel - np.dot(rel, frame.normal) * frame.normal
            radial_len = float(np.linalg.norm(rel_plane))
            radial_error = abs(radial_len - float(frame.radius)) / max(float(frame.radius), 1.0e-9)
            radial_by_key[key] = radial_len
            radial_values.append(radial_len)
        score = 0.78 * rim_score + 0.22 * proximity - 0.20 * min(radial_error, 1.0)
        scored.append((float(score), float(proximity), key))

    if frame is not None and len(radial_values) >= int(min_edges):
        # Keep the densest radial band.  This removes spokes/chords without
        # graph walking.  Bin width is intentionally broad enough for faceted
        # messy imports, but narrow enough to avoid object silhouettes.
        vals = np.asarray(radial_values, dtype=float)
        bin_width = max(float(frame.median_edge_length) * 2.5, float(frame.radius) * 0.045, 1.0e-9)
        lo = float(np.min(vals))
        bins: dict[int, int] = {}
        for value in vals:
            idx = int(np.floor((float(value) - lo) / bin_width))
            bins[idx] = bins.get(idx, 0) + 1
        best_bin = max(bins.items(), key=lambda item: item[1])[0]
        band_keys = {
            key
            for key, value in radial_by_key.items()
            if abs(int(np.floor((float(value) - lo) / bin_width)) - best_bin) <= 1
        }
        if len(band_keys) >= int(min_edges):
            normalized = band_keys
            scored = [item for item in scored if item[2] in normalized]

    scored.sort(key=lambda item: (item[2] == start_key, item[0], item[1]), reverse=True)
    kept = {key for _score, _proximity, key in scored[:max_edges]}
    kept.add(start_key)
    if len(kept) < int(min_edges):
        kept = set(normalized)
    return kept


def _bore_anchor_cycle_candidates(
    edges: set[EdgeKey],
    *,
    vertices: np.ndarray,
    frame: _RingFrame | None,
    start_key: EdgeKey,
    min_loop_edges: int,
    max_anchor_edges: int = 48,
    max_path_edges: int = 320,
) -> list[set[EdgeKey]]:
    """Find simple cycles in a branchy graph by closing paths around anchor edges.

    For an anchor edge ``(a, b)`` this removes that edge and searches for a
    rim-like path from ``a`` back to ``b``.  The anchor plus the path is a simple
    cycle even if the original graph has branches.  The search is local and
    score-biased, so it does not resurrect the old global same-plane sweep.
    """

    normalized = {_edge_key(edge) for edge in edges}
    if len(normalized) < int(min_loop_edges):
        return []

    start_mid = _edge_midpoint(vertices, start_key)
    anchors = sorted(
        normalized,
        key=lambda key: (
            key != start_key,
            float(np.linalg.norm(_edge_midpoint(vertices, key) - start_mid)),
            -_bore_edge_rim_score(vertices, key, frame, start_key),
        ),
    )[: max(1, int(max_anchor_edges))]

    adj: dict[int, list[EdgeKey]] = {}
    for key in normalized:
        a, b = key
        adj.setdefault(int(a), []).append(key)
        adj.setdefault(int(b), []).append(key)

    cycles: list[set[EdgeKey]] = []
    seen: set[frozenset[EdgeKey]] = set()
    for anchor in anchors:
        a, b = anchor
        path = _bore_best_path_between_vertices(
            adj=adj,
            vertices=vertices,
            frame=frame,
            start_key=start_key,
            source=int(a),
            target=int(b),
            forbidden_edge=anchor,
            min_edges=max(1, int(min_loop_edges) - 1),
            max_edges=int(max_path_edges),
        )
        if not path:
            continue
        cycle = {anchor, *path}
        if len(cycle) < int(min_loop_edges):
            continue
        degrees = _edge_degrees(cycle)
        if not degrees or not all(int(v) == 2 for v in degrees.values()):
            continue
        frozen = frozenset(cycle)
        if frozen in seen:
            continue
        seen.add(frozen)
        cycles.append(set(cycle))
        if len(cycles) >= 12:
            break
    return cycles


def _bore_best_path_between_vertices(
    *,
    adj: dict[int, list[EdgeKey]],
    vertices: np.ndarray,
    frame: _RingFrame | None,
    start_key: EdgeKey,
    source: int,
    target: int,
    forbidden_edge: EdgeKey,
    min_edges: int,
    max_edges: int,
) -> set[EdgeKey] | None:
    """Priority DFS for a rim-like simple path between two vertices."""

    import heapq

    def other_vertex(key: EdgeKey, vertex: int) -> int | None:
        a, b = key
        if int(vertex) == int(a):
            return int(b)
        if int(vertex) == int(b):
            return int(a)
        return None

    # Max-heap through negative priority.  Higher accumulated rim score first.
    heap: list[tuple[float, int, int, tuple[EdgeKey, ...], frozenset[int]]] = []
    heapq.heappush(heap, (0.0, 0, int(source), tuple(), frozenset({int(source)})))
    expansions = 0
    max_expansions = 20000

    while heap and expansions < max_expansions:
        neg_score, length, vertex, path, visited_vertices = heapq.heappop(heap)
        expansions += 1
        if vertex == int(target) and length >= int(min_edges):
            return set(path)
        if length >= int(max_edges):
            continue

        candidates = []
        for edge in adj.get(int(vertex), []):
            key = _edge_key(edge)
            if key == forbidden_edge or key in path:
                continue
            nxt = other_vertex(key, int(vertex))
            if nxt is None:
                continue
            if nxt in visited_vertices and nxt != int(target):
                continue
            score = _bore_edge_rim_score(vertices, key, frame, start_key)
            # Reject very non-rim edges in this fallback.  The candidate graph
            # has already been built from local frame scores, so this threshold
            # mainly removes cap spokes.
            if score < 0.32 and key != start_key:
                continue
            candidates.append((score, key, int(nxt)))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for score, key, nxt in candidates[:5]:
            # Small length penalty prevents arbitrary huge object-outline loops
            # from winning while still allowing real 100+ edge bore rims.
            priority = neg_score - float(score) + 0.0025 * (length + 1)
            heapq.heappush(
                heap,
                (
                    priority,
                    length + 1,
                    int(nxt),
                    tuple((*path, key)),
                    frozenset((*visited_vertices, int(nxt))),
                ),
            )
    return None


def _bore_edge_rim_score(
    vertices: np.ndarray,
    edge_key: EdgeKey,
    frame: _RingFrame | None,
    start_key: EdgeKey,
) -> float:
    """Score how likely an edge is to be part of the clicked bore rim."""

    tangent = _edge_tangent_alignment(vertices, edge_key, frame) if frame is not None else 0.5
    plane_score, radius_score = _ring_scores(vertices, edge_key, frame) if frame is not None else (0.55, 0.55)
    start_mid = _edge_midpoint(vertices, start_key)
    mid = _edge_midpoint(vertices, edge_key)
    local_scale = 1.0
    if frame is not None:
        local_scale = max(float(frame.radius), float(frame.median_edge_length) * 6.0, 1.0e-12)
    proximity = 1.0 / (1.0 + float(np.linalg.norm(mid - start_mid)) / local_scale)
    length = _edge_length(vertices, edge_key)
    length_score = 1.0
    if frame is not None and length > 1.0e-12:
        # Penalize very long diagonals/chords; do not over-penalize small real
        # tessellation edges.
        length_score = 1.0 - min(max(length - frame.median_edge_length * 3.5, 0.0) / max(frame.radius, 1.0e-12), 1.0)
    return float(
        0.34 * tangent
        + 0.24 * plane_score
        + 0.26 * radius_score
        + 0.10 * proximity
        + 0.06 * length_score
    )


def _bore_degree_two_cycles(edges: set[EdgeKey], *, min_loop_edges: int) -> list[set[EdgeKey]]:
    normalized = {_edge_key(edge) for edge in edges}
    if len(normalized) < int(min_loop_edges):
        return []
    pruned = _prune_edge_tails(normalized)
    cycles: list[set[EdgeKey]] = []
    for comp in _connected_edge_components(pruned):
        if len(comp) < int(min_loop_edges):
            continue
        degrees = _edge_degrees(comp)
        if degrees and all(int(deg) == 2 for deg in degrees.values()):
            cycles.append(set(comp))
    return cycles


def _prune_edge_tails(edges: set[EdgeKey]) -> set[EdgeKey]:
    remaining = {_edge_key(edge) for edge in edges}
    changed = True
    while changed:
        changed = False
        degrees = _edge_degrees(remaining)
        leaves = {int(v) for v, degree in degrees.items() if int(degree) <= 1}
        if not leaves:
            break
        drop = {edge for edge in remaining if edge[0] in leaves or edge[1] in leaves}
        if drop:
            remaining.difference_update(drop)
            changed = True
    return remaining


def _edge_degrees(edges: set[EdgeKey]) -> dict[int, int]:
    degrees: dict[int, int] = {}
    for a, b in edges:
        degrees[int(a)] = degrees.get(int(a), 0) + 1
        degrees[int(b)] = degrees.get(int(b), 0) + 1
    return degrees


def _choose_best_bore_cycle(
    *,
    cycles: list[set[EdgeKey]],
    vertices: np.ndarray,
    frame: _RingFrame | None,
    start_key: EdgeKey,
    key_to_index: dict[EdgeKey, int],
) -> tuple[set[EdgeKey], float] | None:
    if not cycles:
        return None
    start_mid = _edge_midpoint(vertices, start_key)
    best: tuple[set[EdgeKey], float] | None = None
    best_score = -1.0e9
    for cycle in cycles:
        if not cycle:
            continue
        contains_start = start_key in cycle
        min_dist = min(float(np.linalg.norm(_edge_midpoint(vertices, edge) - start_mid)) for edge in cycle)
        proximity = 1.0 / (1.0 + min_dist)
        tangent_scores = [_edge_tangent_alignment(vertices, edge, frame) for edge in cycle] if frame is not None else [0.65]
        avg_tangent = float(np.mean(tangent_scores)) if tangent_scores else 0.0
        if frame is not None and avg_tangent < 0.45:
            continue
        plane_scores: list[float] = []
        radius_scores: list[float] = []
        if frame is not None:
            for edge in cycle:
                ps, rs = _ring_scores(vertices, edge, frame)
                plane_scores.append(ps)
                radius_scores.append(rs)
        avg_plane = float(np.mean(plane_scores)) if plane_scores else 0.75
        avg_radius = float(np.mean(radius_scores)) if radius_scores else 0.75
        if frame is not None and (avg_plane < 0.42 or avg_radius < 0.45):
            continue
        # Very small cycles are already rejected.  Do not reward huge loops too much;
        # the clicked-local proximity and ring scores matter more than size.
        size_score = min(float(len(cycle)) / 64.0, 1.0)
        score = (
            (0.45 if contains_start else 0.0)
            + 0.25 * proximity
            + 0.55 * avg_tangent
            + 0.35 * avg_plane
            + 0.35 * avg_radius
            + 0.15 * size_score
        )
        if score > best_score:
            best_score = score
            best = (set(cycle), float(score))
    return best


def _unit_np(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if np.isfinite(norm) and norm > 1.0e-12:
        return arr / norm
    fb = np.asarray(fallback, dtype=float).reshape(3)
    fb_norm = float(np.linalg.norm(fb))
    if np.isfinite(fb_norm) and fb_norm > 1.0e-12:
        return fb / fb_norm
    return np.array([0.0, 0.0, 1.0], dtype=float)

# ----------------------------------------------------------------------
# Opposite rim detection helpers
# ----------------------------------------------------------------------

def _collect_vertices_from_edges(vertices: np.ndarray, edge_keys: set[EdgeKey]) -> np.ndarray:
    ids = set()
    for a, b in edge_keys:
        ids.add(a)
        ids.add(b)
    ids = [i for i in ids if 0 <= i < len(vertices)]
    return vertices[np.asarray(ids, dtype=np.int64)]


def _fit_loop_geometry(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if len(vertices) < 3:
        return np.zeros(3), np.array([0,0,1]), 0.0
    center = vertices.mean(axis=0)
    centered = vertices - center
    cov = centered.T @ centered / len(vertices)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmin(eigvals)]
    norm = np.linalg.norm(axis)
    if norm > 0:
        axis = axis / norm
    else:
        axis = np.array([0,0,1])
    # radius: median distance from center in plane perpendicular to axis
    radial = centered - np.outer(centered @ axis, axis)
    radii = np.linalg.norm(radial, axis=1)
    radius = float(np.median(radii)) if len(radii) else 0.0
    return center, axis, radius


def _find_opposite_rim(
    vertices: np.ndarray,
    faces: np.ndarray | None,
    edge_index_to_vertices: np.ndarray,
    edge_to_faces: Mapping[EdgeKey, list[int] | tuple[int, ...]] | None,
    open_edges: np.ndarray | None,
    rim_keys: set[EdgeKey],
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    max_radius_ratio: float,
    min_loop_edges: int,
) -> set[EdgeKey]:
    """Search for opposite rim loop along the axis direction."""
    # Estimate axial position of rim
    axial_vals = [float(np.dot(vertices[a], axis)) for a, _ in rim_keys] + [float(np.dot(vertices[b], axis)) for _, b in rim_keys]
    if not axial_vals:
        return set()
    rim_axial = np.median(axial_vals)

    # Look for another closed loop with similar radius and center, far along axis
    edge_arr = edge_index_to_vertices
    key_to_index = {_edge_key(edge): i for i, edge in enumerate(edge_arr)}
    # We'll use select_edge_region on candidate edges near the opposite side
    # First get all edges that are near the opposite axial band
    axial_band = max(radius * 1.5, 1.0)
    opposite_axial_candidates = set()
    for i, (a, b) in enumerate(edge_arr):
        za = np.dot(vertices[a], axis)
        zb = np.dot(vertices[b], axis)
        if abs(za - rim_axial) > axial_band * 2 and abs(zb - rim_axial) > axial_band * 2:
            # Possibly opposite side
            opposite_axial_candidates.add(i)
    if not opposite_axial_candidates:
        return set()

    # Try to find a closed loop among those edges by brute force component extraction
    # Build edge set for candidate edges
    candidate_keys = {_edge_key(edge_arr[i]) for i in opposite_axial_candidates}
    # Remove any that are too far in radius or center
    filtered = set()
    for key in candidate_keys:
        a, b = key
        p0, p1 = vertices[a], vertices[b]
        mid = (p0 + p1) * 0.5
        radial = mid - center - np.dot(mid - center, axis) * axis
        r = np.linalg.norm(radial)
        if abs(r - radius) <= radius * max_radius_ratio:
            filtered.add(key)

    # Now find connected components and pick the largest closed loop
    components = _connected_edge_components(filtered)
    best_loop = set()
    for comp in components:
        if len(comp) < min_loop_edges:
            continue
        # Check if comp forms a closed loop (all degree 2)
        adj = {}
        for a,b in comp:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        if all(len(adj[v]) == 2 for v in adj):
            if len(comp) > len(best_loop):
                best_loop = comp
    return best_loop


def _connected_edge_components(edges: set[EdgeKey]) -> list[set[EdgeKey]]:
    if not edges:
        return []
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    visited = set()
    components = []
    for start in adj:
        if start in visited:
            continue
        stack = [start]
        comp = set()
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            for nbr in adj.get(v, ()):
                comp.add((min(v,nbr), max(v,nbr)))
                if nbr not in visited:
                    stack.append(nbr)
        components.append(comp)
    return components


# ----------------------------------------------------------------------
# Helper for opposite rim geometry
# ----------------------------------------------------------------------

def _estimate_rim_confidence(rim_count: int, opposite_count: int, radius: float) -> float:
    base = 0.5
    if rim_count >= 16:
        base += 0.2
    elif rim_count >= 8:
        base += 0.1
    if opposite_count >= 8:
        base += 0.2
    if radius > 0:
        base += min(0.1, 0.01 * radius)
    return min(0.98, base)


__all__ = [
    "EdgeSelectionRegion",
    "BoreRimSelection",
    "select_edge_region",
    "select_bore_rim_loop",
    "resolve_edge_index_from_pick_info",
]
