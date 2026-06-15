from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import trimesh


@dataclass(frozen=True)
class PatchQualityReport:
    """Numerical quality summary for a generated hole patch."""

    patch_face_count: int
    patch_vertex_count: int
    boundary_vertex_count: int
    movable_vertex_count: int

    patch_edge_count: int
    patch_edge_length_min: float
    patch_edge_length_median: float
    patch_edge_length_max: float

    context_edge_length_median: float | None
    patch_to_context_median_edge_ratio: float | None

    min_triangle_angle_degrees: float
    median_triangle_angle_degrees: float
    max_triangle_aspect_ratio: float
    median_triangle_aspect_ratio: float
    degenerate_face_count: int

    valence_min: int
    valence_median: float
    valence_max: int

    boundary_normal_mean_deviation_degrees: float | None
    boundary_normal_max_deviation_degrees: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_index_tuple(values: Sequence[int] | np.ndarray | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def _unique_edges_from_faces(faces: np.ndarray) -> tuple[tuple[int, int], ...]:
    edges: set[tuple[int, int]] = set()
    for face in np.asarray(faces, dtype=np.int64):
        if len(face) < 3:
            continue
        tri = [int(face[0]), int(face[1]), int(face[2])]
        for index, a in enumerate(tri):
            b = tri[(index + 1) % 3]
            if a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            edges.add(edge)
    return tuple(sorted(edges))


def _triangle_area2(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.shape != (3, 3) or not np.isfinite(pts).all():
        return 0.0
    return float(np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0])))


def _triangle_edge_lengths(points: np.ndarray) -> tuple[float, float, float]:
    pts = np.asarray(points, dtype=float)
    return (
        float(np.linalg.norm(pts[1] - pts[0])),
        float(np.linalg.norm(pts[2] - pts[1])),
        float(np.linalg.norm(pts[0] - pts[2])),
    )


def _triangle_angles_degrees(points: np.ndarray) -> tuple[float, float, float]:
    lengths = _triangle_edge_lengths(points)
    a, b, c = lengths

    if min(lengths) <= 1.0e-12:
        return (0.0, 0.0, 0.0)

    def angle(opposite: float, side_a: float, side_b: float) -> float:
        denom = max(2.0 * side_a * side_b, 1.0e-300)
        value = (side_a * side_a + side_b * side_b - opposite * opposite) / denom
        return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))

    return (
        angle(a, b, c),
        angle(b, c, a),
        angle(c, a, b),
    )


def _triangle_aspect_ratio(points: np.ndarray) -> float:
    lengths = _triangle_edge_lengths(points)
    shortest = min(lengths)
    longest = max(lengths)
    area2 = _triangle_area2(points)

    if shortest <= 1.0e-12 or area2 <= 1.0e-12:
        return float("inf")

    return float(longest / shortest)


def mesh_edge_length_median(
    mesh: trimesh.Trimesh,
    *,
    face_ids: Sequence[int] | None = None,
) -> float:
    """Return median unique edge length for the whole mesh or a face subset."""

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces_all = np.asarray(mesh.faces, dtype=np.int64)

    if face_ids is None:
        faces = faces_all
    else:
        ids = _as_index_tuple(face_ids)
        if not ids:
            raise ValueError("face_ids must not be empty")
        faces = faces_all[list(ids)]

    edges = _unique_edges_from_faces(faces)
    lengths: list[float] = []

    for a, b in edges:
        if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
            continue
        length = float(np.linalg.norm(vertices[b] - vertices[a]))
        if np.isfinite(length) and length > 1.0e-12:
            lengths.append(length)

    if not lengths:
        raise ValueError("could not compute mesh edge-length median")

    return float(np.median(np.asarray(lengths, dtype=float)))


def _face_normal(vertices: np.ndarray, face: np.ndarray) -> np.ndarray:
    pts = vertices[np.asarray(face[:3], dtype=np.int64)]
    normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12 or not np.isfinite(norm):
        return np.zeros(3, dtype=float)
    return normal / norm


def _normalize_or_none(value: np.ndarray) -> np.ndarray | None:
    vec = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-12 or not np.isfinite(norm):
        return None
    return vec / norm


def compute_boundary_target_normals(
    mesh: trimesh.Trimesh,
    boundary_vertex_ids: Sequence[int],
    *,
    excluded_face_ids: Sequence[int] = (),
) -> dict[int, np.ndarray]:
    """Compute target normals from surrounding faces.

    The excluded faces are usually newly generated patch faces. For the original
    source mesh before preview, excluded_face_ids can be empty.
    """

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    excluded = set(_as_index_tuple(excluded_face_ids))
    boundary_ids = set(_as_index_tuple(boundary_vertex_ids))

    accum: dict[int, np.ndarray] = {
        int(vertex_id): np.zeros(3, dtype=float)
        for vertex_id in boundary_ids
    }

    for face_index, face in enumerate(faces):
        if int(face_index) in excluded:
            continue

        touched = [int(v) for v in face[:3] if int(v) in boundary_ids]
        if not touched:
            continue

        normal = _face_normal(vertices, face)
        if float(np.linalg.norm(normal)) <= 1.0e-12:
            continue

        area2 = _triangle_area2(vertices[np.asarray(face[:3], dtype=np.int64)])
        weight = max(float(area2), 1.0e-12)

        for vertex_id in touched:
            accum[int(vertex_id)] += normal * weight

    out: dict[int, np.ndarray] = {}
    missing: list[int] = []

    for vertex_id, normal_sum in accum.items():
        normal = _normalize_or_none(normal_sum)
        if normal is None:
            missing.append(int(vertex_id))
        else:
            out[int(vertex_id)] = normal

    if missing:
        raise ValueError(
            "could not compute boundary target normals for vertex ids: "
            + ", ".join(str(v) for v in sorted(missing))
        )

    return out


def _vertex_normals_from_face_subset(
    mesh: trimesh.Trimesh,
    *,
    face_ids: Sequence[int],
    vertex_ids: Sequence[int],
) -> dict[int, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    wanted = set(_as_index_tuple(vertex_ids))
    accum: dict[int, np.ndarray] = {
        int(vertex_id): np.zeros(3, dtype=float)
        for vertex_id in wanted
    }

    for face_index in _as_index_tuple(face_ids):
        face = faces[int(face_index)]
        touched = [int(v) for v in face[:3] if int(v) in wanted]
        if not touched:
            continue

        normal = _face_normal(vertices, face)
        if float(np.linalg.norm(normal)) <= 1.0e-12:
            continue

        area2 = _triangle_area2(vertices[np.asarray(face[:3], dtype=np.int64)])
        weight = max(float(area2), 1.0e-12)

        for vertex_id in touched:
            accum[int(vertex_id)] += normal * weight

    out: dict[int, np.ndarray] = {}
    for vertex_id, normal_sum in accum.items():
        normal = _normalize_or_none(normal_sum)
        if normal is not None:
            out[int(vertex_id)] = normal

    return out


def _normal_deviation_degrees(a: np.ndarray, b: np.ndarray) -> float:
    na = _normalize_or_none(a)
    nb = _normalize_or_none(b)
    if na is None or nb is None:
        return float("nan")
    dot = float(np.clip(np.dot(na, nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def analyze_patch_quality(
    mesh: trimesh.Trimesh,
    *,
    patch_face_ids: Sequence[int],
    boundary_vertex_ids: Sequence[int] = (),
    movable_vertex_ids: Sequence[int] = (),
    context_edge_length_median: float | None = None,
    target_boundary_normals: Mapping[int, np.ndarray] | None = None,
) -> PatchQualityReport:
    """Analyze a patch embedded in a preview mesh.

    patch_face_ids should refer to faces inside mesh.faces. boundary_vertex_ids
    are fixed seam vertices. movable_vertex_ids are usually the new patch
    vertices.
    """

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces_all = np.asarray(mesh.faces, dtype=np.int64)
    patch_ids = _as_index_tuple(patch_face_ids)

    if not patch_ids:
        raise ValueError("patch_face_ids must not be empty")

    patch_faces = faces_all[list(patch_ids)]
    patch_vertex_ids = tuple(sorted({int(v) for face in patch_faces for v in face[:3]}))
    boundary_ids = _as_index_tuple(boundary_vertex_ids)
    movable_ids = _as_index_tuple(movable_vertex_ids)

    patch_edges = _unique_edges_from_faces(patch_faces)
    edge_lengths: list[float] = []
    valence: dict[int, int] = {int(v): 0 for v in patch_vertex_ids}

    for a, b in patch_edges:
        if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
            continue
        length = float(np.linalg.norm(vertices[b] - vertices[a]))
        if np.isfinite(length) and length > 1.0e-12:
            edge_lengths.append(length)
        valence[int(a)] = valence.get(int(a), 0) + 1
        valence[int(b)] = valence.get(int(b), 0) + 1

    if edge_lengths:
        edge_array = np.asarray(edge_lengths, dtype=float)
        edge_min = float(np.min(edge_array))
        edge_median = float(np.median(edge_array))
        edge_max = float(np.max(edge_array))
    else:
        edge_min = 0.0
        edge_median = 0.0
        edge_max = 0.0

    all_angles: list[float] = []
    aspect_ratios: list[float] = []
    degenerate_count = 0

    for face in patch_faces:
        pts = vertices[np.asarray(face[:3], dtype=np.int64)]
        area2 = _triangle_area2(pts)
        if area2 <= 1.0e-12:
            degenerate_count += 1

        all_angles.extend(_triangle_angles_degrees(pts))
        aspect_ratios.append(_triangle_aspect_ratio(pts))

    angle_array = np.asarray(all_angles or [0.0], dtype=float)
    aspect_array = np.asarray(aspect_ratios or [float("inf")], dtype=float)
    finite_aspects = aspect_array[np.isfinite(aspect_array)]

    if len(finite_aspects) == 0:
        median_aspect = float("inf")
    else:
        median_aspect = float(np.median(finite_aspects))

    context_ratio: float | None
    if context_edge_length_median is not None and float(context_edge_length_median) > 1.0e-12:
        context_ratio = float(edge_median / float(context_edge_length_median))
    else:
        context_ratio = None

    valence_values = np.asarray(list(valence.values()) or [0], dtype=float)

    normal_mean: float | None = None
    normal_max: float | None = None

    if target_boundary_normals is not None and boundary_ids:
        patch_normals = _vertex_normals_from_face_subset(
            mesh,
            face_ids=patch_ids,
            vertex_ids=boundary_ids,
        )
        deviations: list[float] = []
        for vertex_id in boundary_ids:
            if int(vertex_id) not in patch_normals:
                continue
            if int(vertex_id) not in target_boundary_normals:
                continue
            deviation = _normal_deviation_degrees(
                patch_normals[int(vertex_id)],
                np.asarray(target_boundary_normals[int(vertex_id)], dtype=float),
            )
            if np.isfinite(deviation):
                deviations.append(float(deviation))

        if deviations:
            deviation_array = np.asarray(deviations, dtype=float)
            normal_mean = float(np.mean(deviation_array))
            normal_max = float(np.max(deviation_array))

    return PatchQualityReport(
        patch_face_count=int(len(patch_faces)),
        patch_vertex_count=int(len(patch_vertex_ids)),
        boundary_vertex_count=int(len(boundary_ids)),
        movable_vertex_count=int(len(movable_ids)),
        patch_edge_count=int(len(patch_edges)),
        patch_edge_length_min=edge_min,
        patch_edge_length_median=edge_median,
        patch_edge_length_max=edge_max,
        context_edge_length_median=(
            None if context_edge_length_median is None else float(context_edge_length_median)
        ),
        patch_to_context_median_edge_ratio=context_ratio,
        min_triangle_angle_degrees=float(np.min(angle_array)),
        median_triangle_angle_degrees=float(np.median(angle_array)),
        max_triangle_aspect_ratio=float(np.max(aspect_array)),
        median_triangle_aspect_ratio=median_aspect,
        degenerate_face_count=int(degenerate_count),
        valence_min=int(np.min(valence_values)),
        valence_median=float(np.median(valence_values)),
        valence_max=int(np.max(valence_values)),
        boundary_normal_mean_deviation_degrees=normal_mean,
        boundary_normal_max_deviation_degrees=normal_max,
    )
