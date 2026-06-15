# far_mesh/core/selection_topology.py
"""
Phase 2A / 2B / 2C topology helpers for FAR MESH Quad 3.

This module is intentionally GUI-free and viewport-free.

It computes topology facts from trimesh meshes:
- face adjacency
- connected face components
- boundary edges
- boundary loops
- simple face-region growth
- compact topology reports
- boundary-loop classification hints
- hole candidate detection

Do not import PySide6, MainWindow, viewport classes, or SelectionController here.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import trimesh


Edge = tuple[int, int]


class BoundaryLoopKind(Enum):
    """Semantic classification hint for a boundary loop."""

    OUTER_BOUNDARY = "outer_boundary"
    HOLE_BOUNDARY = "hole_boundary"
    OPEN_CHAIN = "open_chain"
    SELECTED_REGION_BOUNDARY = "selected_region_boundary"
    UNKNOWN = "unknown"


class HoleCandidateKind(Enum):
    """Read-only quality/intent hint for a hole candidate.

    These values do not authorize repair or filling. They are diagnostics for
    UI filtering, future bore-aware tools, and repair dry-run reporting.
    """

    FILLABLE_HOLE = "fillable_hole"
    TINY_CRACK = "tiny_crack"
    BORE_LIKE = "bore_like"
    SELECTED_REGION_BOUNDARY = "selected_region_boundary"
    UNKNOWN = "unknown"


@dataclass
class FaceAdjacency:
    """Face-neighbour graph derived from shared mesh edges."""

    face_count: int
    neighbors: tuple[tuple[int, ...], ...]
    edge_to_faces: Mapping[Edge, tuple[int, ...]]

    def neighbors_of(self, face_id: int) -> tuple[int, ...]:
        if face_id < 0 or face_id >= self.face_count:
            return ()
        return self.neighbors[face_id]


@dataclass
class BoundaryLoop:
    """One extracted boundary chain or closed boundary loop."""

    vertices: tuple[int, ...]
    edges: tuple[Edge, ...]
    closed: bool

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


@dataclass
class BoundaryLoopMeasurement:
    """Measured geometric hints for one boundary loop."""

    vertex_count: int
    edge_count: int
    closed: bool
    perimeter: float
    area_hint: float | None
    centroid: np.ndarray | None


@dataclass
class ClassifiedBoundaryLoop:
    """Boundary loop plus semantic/geometric classification hints."""

    loop: BoundaryLoop
    kind: BoundaryLoopKind
    measurement: BoundaryLoopMeasurement

    @property
    def area_hint(self) -> float | None:
        return self.measurement.area_hint

    @property
    def perimeter(self) -> float:
        return self.measurement.perimeter

    @property
    def centroid(self) -> np.ndarray | None:
        return self.measurement.centroid


@dataclass
class HoleCandidate:
    """
    Candidate hole boundary detected from classified boundary loops.

    This is a read-only topology description. It does not fill, modify, or
    commit mesh data.
    """

    loop: BoundaryLoop
    classified_loop: ClassifiedBoundaryLoop
    boundary_vertices: tuple[int, ...]
    boundary_edges: tuple[Edge, ...]
    perimeter: float
    area_hint: float | None
    centroid: np.ndarray | None
    fill_priority: float


@dataclass
class HoleCandidateDiagnostics:
    """Read-only diagnostic hints for one hole candidate."""

    kind: HoleCandidateKind
    confidence: float
    notes: tuple[str, ...]
    circularity: float | None = None
    perimeter_to_scale: float | None = None
    area_to_scale: float | None = None


@dataclass
class TopologyReport:
    """Compact summary of mesh or selected-region topology."""

    face_count: int
    selected_face_count: int
    component_count: int
    component_sizes: tuple[int, ...]
    boundary_edge_count: int
    boundary_loop_count: int
    closed_boundary_loop_count: int
    open_boundary_chain_count: int


def _validate_mesh_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    if mesh is None:
        raise ValueError("mesh must not be None")

    faces = np.asarray(getattr(mesh, "faces", None), dtype=np.int64)

    if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] < 3:
        raise ValueError("mesh must contain at least one polygonal face")

    return faces


def _normalize_edge(a: int, b: int) -> Edge:
    a = int(a)
    b = int(b)
    return (a, b) if a <= b else (b, a)


def _face_edges(face: Sequence[int]) -> tuple[Edge, ...]:
    verts = [int(v) for v in face]
    edges: list[Edge] = []

    for i, a in enumerate(verts):
        b = verts[(i + 1) % len(verts)]
        if a != b:
            edges.append(_normalize_edge(a, b))

    return tuple(edges)


def _normalize_face_ids(
    face_ids: Iterable[int] | None,
    face_count: int,
) -> np.ndarray:
    if face_ids is None:
        return np.arange(face_count, dtype=np.int64)

    arr = np.asarray(list(face_ids), dtype=np.int64).reshape(-1)

    if arr.size == 0:
        return np.empty(0, dtype=np.int64)

    valid = arr[(arr >= 0) & (arr < face_count)]
    return np.unique(valid).astype(np.int64)


def _unique_loop_vertices(loop: BoundaryLoop) -> tuple[int, ...]:
    """Return loop vertices without a duplicated closing vertex."""

    vertices = tuple(int(v) for v in loop.vertices)

    if len(vertices) >= 2 and vertices[0] == vertices[-1]:
        return vertices[:-1]

    return vertices


def _safe_vertex_positions(
    mesh: trimesh.Trimesh,
    vertex_ids: Sequence[int],
) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)

    if not vertex_ids:
        return np.empty((0, 3), dtype=float)

    ids = np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
    valid = ids[(ids >= 0) & (ids < len(vertices))]

    if valid.size == 0:
        return np.empty((0, 3), dtype=float)

    return vertices[valid]


def _choose_projection_drop_axis(points: np.ndarray) -> int:
    """
    Choose a projection axis for 2D polygon hints.

    This is a heuristic for Phase 2B/2C. It drops the axis with the smallest
    spatial extent, which works well for planar inspection meshes and many
    selected boundary loops.
    """

    if points.size == 0 or points.ndim != 2 or points.shape[1] != 3:
        return 2

    extents = np.ptp(points, axis=0)

    if not np.all(np.isfinite(extents)):
        return 2

    return int(np.argmin(extents))


def _project_points_2d(points: np.ndarray, drop_axis: int) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=float)

    keep_axes = [axis for axis in range(3) if axis != int(drop_axis)]
    return points[:, keep_axes].astype(float, copy=False)


def _signed_polygon_area_2d(points_2d: np.ndarray) -> float:
    if points_2d.ndim != 2 or points_2d.shape[0] < 3 or points_2d.shape[1] != 2:
        return 0.0

    x = points_2d[:, 0]
    y = points_2d[:, 1]

    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _point_in_polygon_2d(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Ray-casting point-in-polygon test.

    Boundary cases are treated as inside enough for classification hints.
    """

    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        return False

    x = float(point[0])
    y = float(point[1])
    inside = False

    j = polygon.shape[0] - 1

    for i in range(polygon.shape[0]):
        xi = float(polygon[i, 0])
        yi = float(polygon[i, 1])
        xj = float(polygon[j, 0])
        yj = float(polygon[j, 1])

        # Boundary check.
        dx = xj - xi
        dy = yj - yi
        cross = (x - xi) * dy - (y - yi) * dx

        if abs(cross) < 1e-12:
            min_x = min(xi, xj) - 1e-12
            max_x = max(xi, xj) + 1e-12
            min_y = min(yi, yj) - 1e-12
            max_y = max(yi, yj) + 1e-12
            if min_x <= x <= max_x and min_y <= y <= max_y:
                return True

        intersects = (yi > y) != (yj > y)
        if intersects:
            denom = yj - yi
            if abs(denom) < 1e-15:
                j = i
                continue
            x_intersect = (xj - xi) * (y - yi) / denom + xi
            if x <= x_intersect:
                inside = not inside

        j = i

    return inside


def _mesh_looks_spatial_3d(mesh: trimesh.Trimesh) -> bool:
    """
    Heuristic: distinguish flat sheet-like meshes from spatial/solid-like meshes.

    A flat single square plane has one near-zero bounding-box axis and should keep
    its single loop as OUTER_BOUNDARY.

    A cube/mechanical part with one deleted face has non-trivial extent in all
    three axes, so a single closed boundary loop is likely an actual hole/opening.
    """

    try:
        bounds = np.asarray(mesh.bounds, dtype=float)
    except Exception:
        return False

    if bounds.shape != (2, 3):
        return False

    extents = np.abs(bounds[1] - bounds[0])
    if not np.all(np.isfinite(extents)):
        return False

    scale = float(np.max(extents)) if extents.size else 0.0
    if scale <= 1e-12:
        return False

    eps = max(scale * 1e-8, 1e-12)
    return int(np.count_nonzero(extents > eps)) >= 3


def _hole_fill_priority(
    *,
    perimeter: float,
    area_hint: float | None,
    edge_count: int,
) -> float:
    """
    Compute a deterministic fill priority.

    Higher values mean "simpler / smaller / safer candidate first".
    This is only a scheduling hint for future tools.
    """

    safe_perimeter = max(float(perimeter), 0.0)
    safe_edge_count = max(int(edge_count), 1)

    perimeter_score = 1.0 / (1.0 + safe_perimeter)
    edge_score = 1.0 / (1.0 + float(safe_edge_count))

    if area_hint is None:
        area_score = 0.0
    else:
        area_score = 1.0 / (1.0 + max(float(area_hint), 0.0))

    return float(area_score + perimeter_score + edge_score)


def _mesh_scale(mesh: trimesh.Trimesh) -> float:
    try:
        bounds = np.asarray(mesh.bounds, dtype=float)
    except Exception:
        return 0.0

    if bounds.shape != (2, 3):
        return 0.0

    extents = np.abs(bounds[1] - bounds[0])
    if not np.all(np.isfinite(extents)):
        return 0.0

    return float(np.max(extents)) if extents.size else 0.0


def _candidate_circularity(
    *,
    area_hint: float | None,
    perimeter: float,
) -> float | None:
    if area_hint is None:
        return None

    safe_area = float(area_hint)
    safe_perimeter = float(perimeter)

    if safe_area <= 0.0 or safe_perimeter <= 1e-12:
        return None

    value = float((4.0 * np.pi * safe_area) / (safe_perimeter * safe_perimeter))
    if not np.isfinite(value):
        return None

    # Numerical/projection artifacts can push this slightly above 1.
    return max(0.0, min(1.0, value))


def diagnose_hole_candidate(
    mesh: trimesh.Trimesh,
    candidate: HoleCandidate,
) -> HoleCandidateDiagnostics:
    """Classify one hole candidate with read-only diagnostic hints.

    This does not change which candidates are returned by find_hole_candidates().
    It only gives downstream UI/repair code safer labels for prioritisation and
    future bore-aware workflows.
    """

    _validate_mesh_faces(mesh)

    if not isinstance(candidate, HoleCandidate):
        raise TypeError("candidate must be a HoleCandidate")

    loop = candidate.loop
    classified_kind = candidate.classified_loop.kind
    edge_count = int(loop.edge_count)
    vertex_count = int(len(candidate.boundary_vertices))
    perimeter = float(candidate.perimeter)
    area_hint = candidate.area_hint

    scale = _mesh_scale(mesh)
    perimeter_to_scale = None if scale <= 1e-12 else perimeter / scale
    area_to_scale = None
    if area_hint is not None and scale > 1e-12:
        area_to_scale = float(area_hint) / (scale * scale)

    circularity = _candidate_circularity(
        area_hint=area_hint,
        perimeter=perimeter,
    )

    notes: list[str] = []

    if classified_kind == BoundaryLoopKind.SELECTED_REGION_BOUNDARY:
        notes.append("candidate comes from selected-region boundary context")
        return HoleCandidateDiagnostics(
            kind=HoleCandidateKind.SELECTED_REGION_BOUNDARY,
            confidence=0.9,
            notes=tuple(notes),
            circularity=circularity,
            perimeter_to_scale=perimeter_to_scale,
            area_to_scale=area_to_scale,
        )

    if not loop.closed:
        notes.append("candidate loop is not closed")
        return HoleCandidateDiagnostics(
            kind=HoleCandidateKind.UNKNOWN,
            confidence=0.2,
            notes=tuple(notes),
            circularity=circularity,
            perimeter_to_scale=perimeter_to_scale,
            area_to_scale=area_to_scale,
        )

    if edge_count <= 3 or vertex_count <= 3:
        notes.append("tiny triangular boundary; likely crack/artifact or deleted triangle")
        return HoleCandidateDiagnostics(
            kind=HoleCandidateKind.TINY_CRACK,
            confidence=0.85,
            notes=tuple(notes),
            circularity=circularity,
            perimeter_to_scale=perimeter_to_scale,
            area_to_scale=area_to_scale,
        )

    if area_to_scale is not None and perimeter_to_scale is not None:
        if area_to_scale <= 1e-6 and perimeter_to_scale <= 0.05:
            notes.append("very small boundary relative to mesh scale")
            return HoleCandidateDiagnostics(
                kind=HoleCandidateKind.TINY_CRACK,
                confidence=0.75,
                notes=tuple(notes),
                circularity=circularity,
                perimeter_to_scale=perimeter_to_scale,
                area_to_scale=area_to_scale,
            )

    if (
        classified_kind == BoundaryLoopKind.HOLE_BOUNDARY
        and edge_count >= 8
        and circularity is not None
        and circularity >= 0.20
    ):
        notes.append("round-ish multi-edge boundary; possible bore/cap opening")
        notes.append("candidate should remain preview-first and Open3D whole-mesh filling must stay guarded")
        return HoleCandidateDiagnostics(
            kind=HoleCandidateKind.BORE_LIKE,
            confidence=min(0.95, 0.55 + 0.4 * circularity),
            notes=tuple(notes),
            circularity=circularity,
            perimeter_to_scale=perimeter_to_scale,
            area_to_scale=area_to_scale,
        )

    if classified_kind == BoundaryLoopKind.HOLE_BOUNDARY:
        notes.append("closed hole boundary candidate")
        return HoleCandidateDiagnostics(
            kind=HoleCandidateKind.FILLABLE_HOLE,
            confidence=0.7,
            notes=tuple(notes),
            circularity=circularity,
            perimeter_to_scale=perimeter_to_scale,
            area_to_scale=area_to_scale,
        )

    notes.append(f"candidate classified loop kind is {classified_kind.value}")
    return HoleCandidateDiagnostics(
        kind=HoleCandidateKind.UNKNOWN,
        confidence=0.3,
        notes=tuple(notes),
        circularity=circularity,
        perimeter_to_scale=perimeter_to_scale,
        area_to_scale=area_to_scale,
    )


def diagnose_hole_candidates(
    mesh: trimesh.Trimesh,
    candidates: Iterable[HoleCandidate] | None = None,
    *,
    face_ids: Iterable[int] | None = None,
    max_area_hint: float | None = None,
    max_perimeter: float | None = None,
) -> tuple[HoleCandidateDiagnostics, ...]:
    """Return diagnostics for hole candidates without mutating mesh data."""

    _validate_mesh_faces(mesh)

    if candidates is None:
        candidates = find_hole_candidates(
            mesh,
            face_ids=face_ids,
            max_area_hint=max_area_hint,
            max_perimeter=max_perimeter,
        )

    return tuple(diagnose_hole_candidate(mesh, candidate) for candidate in candidates)


def build_face_adjacency(mesh: trimesh.Trimesh) -> FaceAdjacency:
    """
    Build a face adjacency graph.

    Two faces are neighbours when they share at least one undirected edge.
    Non-manifold shared edges connect all faces that share that edge.
    """

    faces = _validate_mesh_faces(mesh)
    face_count = int(faces.shape[0])

    edge_to_face_list: dict[Edge, list[int]] = defaultdict(list)

    for face_id, face in enumerate(faces):
        for edge in _face_edges(face):
            edge_to_face_list[edge].append(int(face_id))

    neighbor_sets: list[set[int]] = [set() for _ in range(face_count)]

    for shared_faces in edge_to_face_list.values():
        if len(shared_faces) < 2:
            continue

        for i, face_a in enumerate(shared_faces):
            for face_b in shared_faces[i + 1 :]:
                if face_a != face_b:
                    neighbor_sets[face_a].add(face_b)
                    neighbor_sets[face_b].add(face_a)

    neighbors = tuple(tuple(sorted(items)) for items in neighbor_sets)
    edge_to_faces = {
        edge: tuple(sorted(face_list))
        for edge, face_list in edge_to_face_list.items()
    }

    return FaceAdjacency(
        face_count=face_count,
        neighbors=neighbors,
        edge_to_faces=edge_to_faces,
    )


def extract_connected_components(
    mesh: trimesh.Trimesh,
    face_ids: Iterable[int] | None = None,
    adjacency: FaceAdjacency | None = None,
) -> list[np.ndarray]:
    """
    Extract connected face components.

    If face_ids is provided, components are computed only within that face subset.
    Invalid face IDs are ignored.
    """

    faces = _validate_mesh_faces(mesh)
    face_count = int(faces.shape[0])
    adjacency = adjacency or build_face_adjacency(mesh)

    selected = set(_normalize_face_ids(face_ids, face_count).tolist())
    components: list[np.ndarray] = []

    while selected:
        start = selected.pop()
        queue: deque[int] = deque([start])
        component = [start]

        while queue:
            current = queue.popleft()

            for neighbor in adjacency.neighbors_of(current):
                if neighbor in selected:
                    selected.remove(neighbor)
                    component.append(neighbor)
                    queue.append(neighbor)

        components.append(np.asarray(sorted(component), dtype=np.int64))

    components.sort(key=lambda arr: int(arr[0]) if arr.size else -1)
    return components


def find_boundary_edges(
    mesh: trimesh.Trimesh,
    face_ids: Iterable[int] | None = None,
) -> np.ndarray:
    """
    Find boundary edges for the whole mesh or for a selected face subset.

    For the whole mesh:
        returns open mesh boundary edges.

    For a face subset:
        returns the outer boundary of that selected region, including edges
        between selected and unselected faces.
    """

    faces = _validate_mesh_faces(mesh)
    face_count = int(faces.shape[0])
    selected_faces = _normalize_face_ids(face_ids, face_count)

    edge_counts: dict[Edge, int] = defaultdict(int)

    for face_id in selected_faces:
        for edge in _face_edges(faces[int(face_id)]):
            edge_counts[edge] += 1

    boundary_edges = sorted(edge for edge, count in edge_counts.items() if count == 1)

    if not boundary_edges:
        return np.empty((0, 2), dtype=np.int64)

    return np.asarray(boundary_edges, dtype=np.int64)


def extract_boundary_loops(
    mesh: trimesh.Trimesh,
    boundary_edges: np.ndarray | Iterable[Sequence[int]],
) -> list[BoundaryLoop]:
    """
    Convert boundary edges into closed loops or open chains.

    The mesh argument is accepted for API consistency and future validation,
    but the loop extraction currently only needs the boundary edge list.
    """

    _validate_mesh_faces(mesh)

    raw_edges = np.asarray(list(boundary_edges), dtype=np.int64)

    if raw_edges.size == 0:
        return []

    raw_edges = raw_edges.reshape((-1, 2))

    edges: set[Edge] = {
        _normalize_edge(int(a), int(b))
        for a, b in raw_edges
        if int(a) != int(b)
    }

    vertex_neighbors: dict[int, set[int]] = defaultdict(set)

    for a, b in edges:
        vertex_neighbors[a].add(b)
        vertex_neighbors[b].add(a)

    unused = set(edges)
    loops: list[BoundaryLoop] = []

    def choose_start_edge() -> Edge:
        # Prefer open-chain endpoints first, otherwise choose deterministic edge.
        endpoint_edges = [
            edge
            for edge in unused
            if len(vertex_neighbors[edge[0]]) == 1
            or len(vertex_neighbors[edge[1]]) == 1
        ]

        if endpoint_edges:
            return sorted(endpoint_edges)[0]

        return sorted(unused)[0]

    while unused:
        start_a, start_b = choose_start_edge()

        # For open chains, start from the endpoint when possible.
        if len(vertex_neighbors[start_b]) == 1 and len(vertex_neighbors[start_a]) != 1:
            start_a, start_b = start_b, start_a

        current_edge = _normalize_edge(start_a, start_b)
        unused.remove(current_edge)

        vertices = [start_a, start_b]
        loop_edges = [current_edge]

        previous = start_a
        current = start_b
        closed = False

        while True:
            candidates = [
                n
                for n in vertex_neighbors[current]
                if _normalize_edge(current, n) in unused and n != previous
            ]

            if not candidates:
                candidates = [
                    n
                    for n in vertex_neighbors[current]
                    if _normalize_edge(current, n) in unused
                ]

            if not candidates:
                break

            next_vertex = min(candidates)
            next_edge = _normalize_edge(current, next_vertex)

            unused.remove(next_edge)
            loop_edges.append(next_edge)
            vertices.append(next_vertex)

            previous, current = current, next_vertex

            if current == vertices[0]:
                closed = True
                break

        loops.append(
            BoundaryLoop(
                vertices=tuple(int(v) for v in vertices),
                edges=tuple(loop_edges),
                closed=closed,
            )
        )

    loops.sort(key=lambda loop: (-loop.closed, loop.vertices[0] if loop.vertices else -1))
    return loops


def measure_boundary_loop(
    mesh: trimesh.Trimesh,
    loop: BoundaryLoop,
) -> BoundaryLoopMeasurement:
    """
    Measure one boundary loop.

    The area value is a 2D projection hint, not a guaranteed exact 3D surface
    area. It is sufficient for Phase 2B/2C loop classification and later
    UI/tool decisions, but future bore tools may add stronger geometric fitting.
    """

    _validate_mesh_faces(mesh)

    vertices = np.asarray(mesh.vertices, dtype=float)
    perimeter = 0.0

    for a, b in loop.edges:
        if 0 <= a < len(vertices) and 0 <= b < len(vertices):
            perimeter += float(np.linalg.norm(vertices[int(a)] - vertices[int(b)]))

    unique_vertex_ids = _unique_loop_vertices(loop)
    points = _safe_vertex_positions(mesh, unique_vertex_ids)

    centroid: np.ndarray | None
    if points.size == 0:
        centroid = None
    else:
        centroid = np.mean(points, axis=0)

    area_hint: float | None = None

    if loop.closed and points.shape[0] >= 3:
        drop_axis = _choose_projection_drop_axis(points)
        projected = _project_points_2d(points, drop_axis)
        area_hint = abs(_signed_polygon_area_2d(projected))

    return BoundaryLoopMeasurement(
        vertex_count=loop.vertex_count,
        edge_count=loop.edge_count,
        closed=loop.closed,
        perimeter=perimeter,
        area_hint=area_hint,
        centroid=centroid,
    )


def classify_boundary_loops(
    mesh: trimesh.Trimesh,
    loops: Iterable[BoundaryLoop],
    face_ids: Iterable[int] | None = None,
) -> list[ClassifiedBoundaryLoop]:
    """
    Classify boundary loops into coarse semantic categories.

    Classification rules for Phase 2B/2C:
    - open loops are OPEN_CHAIN
    - closed loops from an explicit selected face subset are SELECTED_REGION_BOUNDARY
    - closed loops on spatial 3D meshes are treated as HOLE_BOUNDARY because
      full-mesh open boundaries on solid-like parts represent openings/holes,
      including the multi-hole case where 2D containment is not meaningful
    - flat/sheet-like full-mesh closed loops are classified as OUTER_BOUNDARY or
      HOLE_BOUNDARY using projected polygon containment when possible
    - unclear cases become UNKNOWN

    These are topology/geometry hints, not destructive repair decisions.
    """

    _validate_mesh_faces(mesh)

    loop_list = list(loops)
    if not loop_list:
        return []

    selected_context = face_ids is not None
    measurements = [measure_boundary_loop(mesh, loop) for loop in loop_list]

    if selected_context:
        classified: list[ClassifiedBoundaryLoop] = []

        for loop, measurement in zip(loop_list, measurements):
            kind = (
                BoundaryLoopKind.SELECTED_REGION_BOUNDARY
                if loop.closed
                else BoundaryLoopKind.OPEN_CHAIN
            )
            classified.append(
                ClassifiedBoundaryLoop(
                    loop=loop,
                    kind=kind,
                    measurement=measurement,
                )
            )

        return classified

    all_loop_points: list[np.ndarray] = []
    for loop in loop_list:
        unique_vertex_ids = _unique_loop_vertices(loop)
        all_loop_points.append(_safe_vertex_positions(mesh, unique_vertex_ids))

    non_empty_points = [points for points in all_loop_points if points.size > 0]
    if non_empty_points:
        combined_points = np.vstack(non_empty_points)
        global_drop_axis = _choose_projection_drop_axis(combined_points)
    else:
        global_drop_axis = 2

    polygons: list[np.ndarray | None] = []
    areas: list[float | None] = []
    centroids_2d: list[np.ndarray | None] = []

    for loop, points in zip(loop_list, all_loop_points):
        if not loop.closed or points.shape[0] < 3:
            polygons.append(None)
            areas.append(None)
            centroids_2d.append(None)
            continue

        polygon = _project_points_2d(points, global_drop_axis)
        signed_area = _signed_polygon_area_2d(polygon)
        area = abs(signed_area)

        polygons.append(polygon)
        areas.append(area)

        if points.size == 0:
            centroids_2d.append(None)
        else:
            centroid_3d = np.mean(points, axis=0)
            centroid_2d = _project_points_2d(centroid_3d.reshape(1, 3), global_drop_axis)[0]
            centroids_2d.append(centroid_2d)

    closed_loop_indices = [
        i
        for i, loop in enumerate(loop_list)
        if loop.closed
    ]

    # For spatial/solid-like meshes, a full-mesh closed boundary loop is an
    # opening in the part, not a planar outer border.  The previous logic only
    # handled the single-loop case and then fell back to 2D containment for
    # multiple loops, which misclassified multiple sphere/cap openings as
    # OUTER_BOUNDARY and produced zero hole candidates.
    spatial_3d_closed_boundaries_are_holes = _mesh_looks_spatial_3d(mesh)

    classified = []

    for i, loop in enumerate(loop_list):
        measurement = measurements[i]

        if not loop.closed:
            kind = BoundaryLoopKind.OPEN_CHAIN
        elif spatial_3d_closed_boundaries_are_holes:
            kind = BoundaryLoopKind.HOLE_BOUNDARY
        elif areas[i] is None or areas[i] <= 1e-12 or polygons[i] is None:
            kind = BoundaryLoopKind.UNKNOWN
        else:
            containment_depth = 0
            centroid = centroids_2d[i]
            current_area = float(areas[i])

            if centroid is None:
                kind = BoundaryLoopKind.UNKNOWN
            else:
                for j, polygon in enumerate(polygons):
                    if i == j or polygon is None:
                        continue

                    other_area = areas[j]
                    if other_area is None:
                        continue

                    # Only larger polygons can contain this loop in the
                    # outer/hole hierarchy. This avoids an outer loop being
                    # misclassified because its centroid lies inside a smaller
                    # inner hole polygon.
                    if float(other_area) <= current_area + 1e-12:
                        continue

                    if _point_in_polygon_2d(centroid, polygon):
                        containment_depth += 1

                if containment_depth % 2 == 0:
                    kind = BoundaryLoopKind.OUTER_BOUNDARY
                else:
                    kind = BoundaryLoopKind.HOLE_BOUNDARY

        classified.append(
            ClassifiedBoundaryLoop(
                loop=loop,
                kind=kind,
                measurement=measurement,
            )
        )

    classified.sort(
        key=lambda item: (
            item.kind.value,
            -(item.area_hint or 0.0),
            item.loop.vertices[0] if item.loop.vertices else -1,
        )
    )

    return classified


def find_hole_candidates(
    mesh: trimesh.Trimesh,
    face_ids: Iterable[int] | None = None,
    max_area_hint: float | None = None,
    max_perimeter: float | None = None,
) -> list[HoleCandidate]:
    """
    Find likely hole candidates from boundary-loop classification.

    For a full mesh:
        only HOLE_BOUNDARY loops are returned.

    For a selected face subset:
        closed SELECTED_REGION_BOUNDARY loops are returned as candidates,
        because a selected region may intentionally represent the user-chosen
        repair area.

    Filters:
        max_area_hint:
            If provided, candidates with projected area_hint greater than this
            value are skipped. Candidates without an area_hint are skipped when
            this filter is active.

        max_perimeter:
            If provided, candidates with perimeter greater than this value are
            skipped.

    This function is read-only. It never modifies mesh data.
    """

    _validate_mesh_faces(mesh)

    boundary_edges = find_boundary_edges(mesh, face_ids=face_ids)
    loops = extract_boundary_loops(mesh, boundary_edges)
    classified_loops = classify_boundary_loops(mesh, loops, face_ids=face_ids)

    selected_context = face_ids is not None
    candidates: list[HoleCandidate] = []

    for classified in classified_loops:
        loop = classified.loop

        if not loop.closed:
            continue

        if selected_context:
            if classified.kind != BoundaryLoopKind.SELECTED_REGION_BOUNDARY:
                continue
        elif classified.kind != BoundaryLoopKind.HOLE_BOUNDARY:
            continue

        perimeter = float(classified.perimeter)
        area_hint = classified.area_hint

        if max_perimeter is not None and perimeter > float(max_perimeter):
            continue

        if max_area_hint is not None:
            if area_hint is None:
                continue
            if float(area_hint) > float(max_area_hint):
                continue

        boundary_vertices = _unique_loop_vertices(loop)
        boundary_edges_tuple = tuple(loop.edges)

        fill_priority = _hole_fill_priority(
            perimeter=perimeter,
            area_hint=area_hint,
            edge_count=loop.edge_count,
        )

        candidates.append(
            HoleCandidate(
                loop=loop,
                classified_loop=classified,
                boundary_vertices=boundary_vertices,
                boundary_edges=boundary_edges_tuple,
                perimeter=perimeter,
                area_hint=area_hint,
                centroid=classified.centroid,
                fill_priority=fill_priority,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate.fill_priority,
            candidate.area_hint if candidate.area_hint is not None else float("inf"),
            candidate.perimeter,
            candidate.boundary_vertices[0] if candidate.boundary_vertices else -1,
        )
    )

    return candidates


def region_grow_faces(
    mesh: trimesh.Trimesh,
    seed_faces: Iterable[int],
    constraints: Mapping[str, Any] | None = None,
    adjacency: FaceAdjacency | None = None,
) -> np.ndarray:
    """
    Basic face-region growth from seed faces.

    Supported constraints:
        allowed_faces: iterable[int] | None
        blocked_faces: iterable[int] | None
        max_faces: int | None

    This is intentionally simple for Phase 2A. Bore-specific constraints should
    be added later, after boundary loops and components are stable.
    """

    faces = _validate_mesh_faces(mesh)
    face_count = int(faces.shape[0])
    adjacency = adjacency or build_face_adjacency(mesh)
    constraints = constraints or {}

    seeds = _normalize_face_ids(seed_faces, face_count)

    if seeds.size == 0:
        return np.empty(0, dtype=np.int64)

    allowed_raw = constraints.get("allowed_faces")
    blocked_raw = constraints.get("blocked_faces")
    max_faces = constraints.get("max_faces")

    if allowed_raw is None:
        allowed = set(range(face_count))
    else:
        allowed = set(_normalize_face_ids(allowed_raw, face_count).tolist())

    blocked = (
        set(_normalize_face_ids(blocked_raw, face_count).tolist())
        if blocked_raw is not None
        else set()
    )

    max_count = int(max_faces) if max_faces is not None else None
    if max_count is not None and max_count <= 0:
        return np.empty(0, dtype=np.int64)

    visited: set[int] = set()
    queue: deque[int] = deque()

    for seed in seeds.tolist():
        seed = int(seed)
        if seed in allowed and seed not in blocked:
            visited.add(seed)
            queue.append(seed)

    while queue:
        current = queue.popleft()

        if max_count is not None and len(visited) >= max_count:
            break

        for neighbor in adjacency.neighbors_of(current):
            if neighbor in visited or neighbor not in allowed or neighbor in blocked:
                continue

            visited.add(neighbor)
            queue.append(neighbor)

            if max_count is not None and len(visited) >= max_count:
                break

    return np.asarray(sorted(visited), dtype=np.int64)


def analyze_selection_topology(
    mesh: trimesh.Trimesh,
    face_ids: Iterable[int] | None = None,
) -> TopologyReport:
    """
    Build a compact topology report for the full mesh or selected face subset.

    This is useful for future GUI/tool layers because they can ask one pure-core
    function for component, boundary-edge, and boundary-loop facts without
    duplicating topology logic.
    """

    faces = _validate_mesh_faces(mesh)
    face_count = int(faces.shape[0])
    selected_faces = _normalize_face_ids(face_ids, face_count)

    adjacency = build_face_adjacency(mesh)

    components = extract_connected_components(
        mesh,
        face_ids=selected_faces,
        adjacency=adjacency,
    )

    boundary_edges = find_boundary_edges(
        mesh,
        face_ids=selected_faces,
    )

    boundary_loops = extract_boundary_loops(
        mesh,
        boundary_edges,
    )

    closed_count = sum(1 for loop in boundary_loops if loop.closed)
    open_count = len(boundary_loops) - closed_count

    return TopologyReport(
        face_count=face_count,
        selected_face_count=int(selected_faces.size),
        component_count=len(components),
        component_sizes=tuple(int(len(component)) for component in components),
        boundary_edge_count=int(len(boundary_edges)),
        boundary_loop_count=len(boundary_loops),
        closed_boundary_loop_count=closed_count,
        open_boundary_chain_count=open_count,
    )
