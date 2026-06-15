"""Shared Bore mesh-topology helpers.

This module is the first extraction point for duplicated helpers that currently
exist in region_select.py, geometry.py, measure.py and rebuild.py.  It performs
only deterministic topology normalization/querying.  It does not classify
features and does not mutate mesh data.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable, Mapping, Sequence

import numpy as np

from .types import EdgeKey


def normalize_edge(edge: object) -> EdgeKey:
    """Return a sorted 2-vertex edge key."""

    try:
        a, b = tuple(edge)[:2]  # type: ignore[arg-type]
        ia = int(a)
        ib = int(b)
    except Exception as exc:
        raise ValueError(f"Invalid edge: {edge!r}") from exc
    if ia == ib:
        raise ValueError(f"Degenerate edge: {edge!r}")
    return (ia, ib) if ia < ib else (ib, ia)


def face_edges(face: Sequence[int]) -> tuple[EdgeKey, EdgeKey, EdgeKey]:
    """Return the three normalized edges of a triangular runtime face."""

    if len(face) < 3:
        raise ValueError(f"Face needs at least three vertices: {face!r}")
    a, b, c = int(face[0]), int(face[1]), int(face[2])
    return (normalize_edge((a, b)), normalize_edge((b, c)), normalize_edge((c, a)))


def build_edge_to_faces(faces: np.ndarray) -> dict[EdgeKey, tuple[int, ...]]:
    """Build edge -> adjacent face IDs for runtime triangle faces."""

    arr = np.asarray(faces, dtype=np.int64)
    table: dict[EdgeKey, list[int]] = defaultdict(list)
    for fid, face in enumerate(arr):
        for edge in face_edges(face):
            table[edge].append(int(fid))
    return {edge: tuple(ids) for edge, ids in table.items()}


def boundary_edges_for_face_patch(faces: np.ndarray, face_ids: Iterable[int]) -> tuple[EdgeKey, ...]:
    """Return patch boundary edges: edges used by exactly one selected face."""

    arr = np.asarray(faces, dtype=np.int64)
    selected = {int(fid) for fid in face_ids if 0 <= int(fid) < len(arr)}
    counts: dict[EdgeKey, int] = defaultdict(int)
    for fid in selected:
        for edge in face_edges(arr[int(fid)]):
            counts[edge] += 1
    return tuple(sorted(edge for edge, count in counts.items() if count == 1))


def face_adjacency_for_patch(faces: np.ndarray, face_ids: Iterable[int]) -> dict[int, tuple[int, ...]]:
    """Return selected-face adjacency via shared internal edges."""

    arr = np.asarray(faces, dtype=np.int64)
    selected = {int(fid) for fid in face_ids if 0 <= int(fid) < len(arr)}
    edge_to_selected_faces: dict[EdgeKey, list[int]] = defaultdict(list)
    for fid in selected:
        for edge in face_edges(arr[int(fid)]):
            edge_to_selected_faces[edge].append(int(fid))
    adjacency: dict[int, set[int]] = {fid: set() for fid in selected}
    for adjacent in edge_to_selected_faces.values():
        if len(adjacent) < 2:
            continue
        for fid in adjacent:
            adjacency[fid].update(other for other in adjacent if other != fid)
    return {fid: tuple(sorted(values)) for fid, values in adjacency.items()}


def connected_face_components(faces: np.ndarray, face_ids: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    """Return connected components of a face patch, largest first."""

    selected = {int(fid) for fid in face_ids if int(fid) >= 0}
    if not selected:
        return ()
    adjacency = face_adjacency_for_patch(faces, selected)
    remaining = set(selected)
    components: list[tuple[int, ...]] = []
    while remaining:
        start = next(iter(remaining))
        queue: deque[int] = deque([start])
        remaining.remove(start)
        component: list[int] = []
        while queue:
            fid = queue.popleft()
            component.append(fid)
            for other in adjacency.get(fid, ()):  # already selected/valid
                if other in remaining:
                    remaining.remove(other)
                    queue.append(other)
        components.append(tuple(sorted(component)))
    components.sort(key=lambda item: (-len(item), item[:1]))
    return tuple(components)


def edge_loop_components(boundary_edges: Iterable[EdgeKey]) -> tuple[tuple[EdgeKey, ...], ...]:
    """Group boundary edges into vertex-connected components."""

    edges = tuple(normalize_edge(edge) for edge in boundary_edges)
    if not edges:
        return ()
    vertex_to_edges: dict[int, set[EdgeKey]] = defaultdict(set)
    for edge in edges:
        a, b = edge
        vertex_to_edges[a].add(edge)
        vertex_to_edges[b].add(edge)
    remaining = set(edges)
    components: list[tuple[EdgeKey, ...]] = []
    while remaining:
        start = next(iter(remaining))
        queue: deque[EdgeKey] = deque([start])
        remaining.remove(start)
        component: list[EdgeKey] = []
        while queue:
            edge = queue.popleft()
            component.append(edge)
            for vertex in edge:
                for other in tuple(vertex_to_edges.get(vertex, ())):
                    if other in remaining:
                        remaining.remove(other)
                        queue.append(other)
        components.append(tuple(sorted(component)))
    components.sort(key=lambda item: (-len(item), item[:1]))
    return tuple(components)


def connected_edge_components(edges: Iterable[EdgeKey]) -> tuple[tuple[EdgeKey, ...], ...]:
    """Group arbitrary edges into vertex-connected components.

    This is the edge-cloud equivalent of ``connected_face_components``.  It is
    used by measurement code for selected rim fragments; it does not classify
    features or mutate mesh data.
    """

    return edge_loop_components(tuple(normalize_edge(edge) for edge in edges))


def edge_graph_stats(edges: Iterable[EdgeKey], *, vertices: np.ndarray) -> dict[str, object]:
    """Return neutral graph statistics for an edge cloud.

    The result is measurement evidence only: component count, closure status,
    endpoint gap, branch count and median edge length.
    """

    normalized = tuple(sorted({normalize_edge(edge) for edge in edges}))
    components = connected_edge_components(normalized)
    degree: dict[int, int] = defaultdict(int)
    lengths: list[float] = []
    verts = np.asarray(vertices, dtype=float)
    for a, b in normalized:
        degree[int(a)] += 1
        degree[int(b)] += 1
        if 0 <= int(a) < len(verts) and 0 <= int(b) < len(verts):
            length = float(np.linalg.norm(verts[int(a), :3] - verts[int(b), :3]))
            if np.isfinite(length) and length > 0.0:
                lengths.append(length)
    endpoints = tuple(sorted(int(v) for v, d in degree.items() if int(d) == 1))
    branch_count = sum(1 for d in degree.values() if int(d) > 2)
    closed = bool(normalized) and len(components) == 1 and all(int(d) == 2 for d in degree.values())
    endpoint_gap = 0.0
    if len(endpoints) == 2:
        a, b = endpoints
        if 0 <= int(a) < len(verts) and 0 <= int(b) < len(verts):
            endpoint_gap = float(np.linalg.norm(verts[int(a), :3] - verts[int(b), :3]))
    return {
        "component_count": int(len(components)),
        "edge_count": int(len(normalized)),
        "vertex_count": int(len(degree)),
        "closed": bool(closed),
        "open_endpoint_count": int(len(endpoints)),
        "endpoint_vertices": endpoints,
        "endpoint_gap": float(endpoint_gap),
        "branch_vertex_count": int(branch_count),
        "degree2_vertex_count": int(sum(1 for d in degree.values() if int(d) == 2)),
        "median_edge_length": float(np.median(lengths)) if lengths else 0.0,
        "component_edge_counts": tuple(int(len(component)) for component in components),
    }


def patch_boundary_loop_components(faces: np.ndarray, face_ids: Iterable[int]) -> tuple[tuple[EdgeKey, ...], ...]:
    """Return boundary-edge components for a selected face patch."""

    return edge_loop_components(boundary_edges_for_face_patch(faces, face_ids))


def summarize_patch_topology(faces: np.ndarray, face_ids: Iterable[int]) -> dict[str, object]:
    """Small serializable topology summary used by diagnostics and ledgers."""

    ids = tuple(sorted({int(fid) for fid in face_ids if int(fid) >= 0}))
    components = connected_face_components(faces, ids)
    boundary_loops = patch_boundary_loop_components(faces, ids)
    return {
        "face_count": int(len(ids)),
        "component_count": int(len(components)),
        "component_face_counts": tuple(int(len(item)) for item in components),
        "boundary_loop_count": int(len(boundary_loops)),
        "boundary_loop_edge_counts": tuple(int(len(item)) for item in boundary_loops),
    }
