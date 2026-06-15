from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import trimesh


@dataclass(frozen=True)
class PatchTopologyReport:
    patch_face_count: int
    patch_vertex_count: int
    patch_component_count: int
    patch_boundary_edge_count: int
    expected_seam_edge_count: int
    seam_edge_count: int
    missing_seam_edge_count: int
    extra_open_boundary_edge_count: int
    nonmanifold_patch_edge_count: int
    seam_coverage_ratio: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NormalCompatibleSupportContext:
    boundary_vertex_ids: tuple[int, ...]
    seed_face_ids: tuple[int, ...]
    support_face_ids: tuple[int, ...]
    support_vertex_ids: tuple[int, ...]
    rejected_face_ids: tuple[int, ...]
    mean_normal: tuple[float, float, float]
    normal_spread_degrees: float
    contamination_score: float
    smooth_dot_threshold: float
    max_rings: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LocalMLSSurfaceProjector:
    centroid: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    normal_w: np.ndarray
    sample_uv: np.ndarray
    sample_w: np.ndarray
    nearest_count: int = 16
    ridge: float = 1.0e-10

    def project(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float).reshape(3)
        local = point - self.centroid
        uv = np.asarray(
            [
                float(np.dot(local, self.tangent_u)),
                float(np.dot(local, self.tangent_v)),
            ],
            dtype=float,
        )

        if len(self.sample_uv) == 0:
            return point.copy()

        delta = self.sample_uv - uv.reshape(1, 2)
        dist2 = np.einsum("ij,ij->i", delta, delta)
        order = np.argsort(dist2)
        count = min(int(self.nearest_count), len(order))
        chosen = order[:count]

        duv = delta[chosen]
        values = self.sample_w[chosen]
        weights = 1.0 / np.maximum(dist2[chosen], 1.0e-12)

        if count >= 6:
            # Local quadratic height field:
            # w = a + b*du + c*dv + d*du^2 + e*du*dv + f*dv^2
            design = np.column_stack(
                [
                    np.ones(count, dtype=float),
                    duv[:, 0],
                    duv[:, 1],
                    duv[:, 0] * duv[:, 0],
                    duv[:, 0] * duv[:, 1],
                    duv[:, 1] * duv[:, 1],
                ]
            )
        elif count >= 3:
            # Fallback local plane.
            design = np.column_stack(
                [
                    np.ones(count, dtype=float),
                    duv[:, 0],
                    duv[:, 1],
                ]
            )
        else:
            # Last fallback: weighted local average height.
            height = float(np.average(values, weights=weights))
            return self.centroid + uv[0] * self.tangent_u + uv[1] * self.tangent_v + height * self.normal_w

        sqrt_w = np.sqrt(weights).reshape(-1, 1)
        weighted_design = design * sqrt_w
        weighted_values = values * sqrt_w.reshape(-1)

        try:
            coeffs, *_ = np.linalg.lstsq(
                weighted_design.T @ weighted_design + self.ridge * np.eye(design.shape[1]),
                weighted_design.T @ weighted_values,
                rcond=None,
            )
            height = float(coeffs[0])
        except Exception:
            height = float(np.average(values, weights=weights))

        return self.centroid + uv[0] * self.tangent_u + uv[1] * self.tangent_v + height * self.normal_w

    def to_dict(self) -> dict[str, object]:
        return {
            "centroid": tuple(float(v) for v in self.centroid),
            "tangent_u": tuple(float(v) for v in self.tangent_u),
            "tangent_v": tuple(float(v) for v in self.tangent_v),
            "normal_w": tuple(float(v) for v in self.normal_w),
            "sample_count": int(len(self.sample_w)),
            "nearest_count": int(self.nearest_count),
            "ridge": float(self.ridge),
        }


def _as_index_tuple(values: Sequence[int] | np.ndarray | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def _loop_edges(boundary_vertex_ids: Sequence[int]) -> set[tuple[int, int]]:
    ids = _as_index_tuple(boundary_vertex_ids)
    edges: set[tuple[int, int]] = set()
    if len(ids) < 2:
        return edges
    for index, a in enumerate(ids):
        b = ids[(index + 1) % len(ids)]
        if a == b:
            continue
        edges.add((a, b) if a < b else (b, a))
    return edges


def _edge_counts_from_faces(faces: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for face in np.asarray(faces, dtype=np.int64):
        tri = [int(face[0]), int(face[1]), int(face[2])]
        for index, a in enumerate(tri):
            b = tri[(index + 1) % 3]
            if a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            counts[edge] = counts.get(edge, 0) + 1
    return counts


def _triangle_area2(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.shape != (3, 3) or not np.isfinite(pts).all():
        return 0.0
    return float(np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0])))


def _face_normal(vertices: np.ndarray, face: np.ndarray) -> np.ndarray:
    pts = vertices[np.asarray(face[:3], dtype=np.int64)]
    normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12 or not np.isfinite(norm):
        return np.zeros(3, dtype=float)
    return normal / norm


def _normalize(value: np.ndarray) -> np.ndarray | None:
    vec = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-12 or not np.isfinite(norm):
        return None
    return vec / norm


def _orthonormal_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = _normalize(normal)
    if w is None:
        w = np.asarray([0.0, 0.0, 1.0], dtype=float)

    reference = np.asarray([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(reference, w))) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=float)

    u = np.cross(reference, w)
    u = _normalize(u)
    if u is None:
        u = np.asarray([1.0, 0.0, 0.0], dtype=float)
    v = np.cross(w, u)
    v = _normalize(v)
    if v is None:
        v = np.asarray([0.0, 1.0, 0.0], dtype=float)
    return u, v, w


def _build_face_adjacency(faces: np.ndarray) -> dict[int, set[int]]:
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(np.asarray(faces, dtype=np.int64)):
        tri = [int(face[0]), int(face[1]), int(face[2])]
        for index, a in enumerate(tri):
            b = tri[(index + 1) % 3]
            edge = (a, b) if a < b else (b, a)
            edge_to_faces.setdefault(edge, []).append(int(face_id))

    adjacency: dict[int, set[int]] = {int(i): set() for i in range(len(faces))}
    for attached in edge_to_faces.values():
        if len(attached) < 2:
            continue
        for a in attached:
            for b in attached:
                if a != b:
                    adjacency[int(a)].add(int(b))
    return adjacency


def _build_vertex_to_faces(faces: np.ndarray) -> dict[int, set[int]]:
    mapping: dict[int, set[int]] = {}
    for face_id, face in enumerate(np.asarray(faces, dtype=np.int64)):
        for vertex_id in face[:3]:
            mapping.setdefault(int(vertex_id), set()).add(int(face_id))
    return mapping


def validate_patch_topology(
    mesh: trimesh.Trimesh,
    *,
    patch_face_ids: Sequence[int],
    boundary_vertex_ids: Sequence[int],
) -> PatchTopologyReport:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces_all = np.asarray(mesh.faces, dtype=np.int64)

    patch_ids = _as_index_tuple(patch_face_ids)
    if not patch_ids:
        raise ValueError("patch_face_ids must not be empty")

    patch_faces = faces_all[list(patch_ids)]
    patch_vertices = tuple(sorted({int(v) for face in patch_faces for v in face[:3]}))

    edge_counts = _edge_counts_from_faces(patch_faces)
    patch_boundary_edges = {edge for edge, count in edge_counts.items() if count == 1}
    nonmanifold_edges = {edge for edge, count in edge_counts.items() if count > 2}
    expected_seam_edges = _loop_edges(boundary_vertex_ids)

    seam_edges = patch_boundary_edges & expected_seam_edges
    missing_seam_edges = expected_seam_edges - patch_boundary_edges
    extra_open_edges = patch_boundary_edges - expected_seam_edges

    # Components by shared patch edges.
    edge_to_local_faces: dict[tuple[int, int], list[int]] = {}
    for local_face_id, face in enumerate(patch_faces):
        tri = [int(face[0]), int(face[1]), int(face[2])]
        for index, a in enumerate(tri):
            b = tri[(index + 1) % 3]
            edge = (a, b) if a < b else (b, a)
            edge_to_local_faces.setdefault(edge, []).append(int(local_face_id))

    local_adjacency: dict[int, set[int]] = {i: set() for i in range(len(patch_faces))}
    for local_faces in edge_to_local_faces.values():
        if len(local_faces) < 2:
            continue
        for a in local_faces:
            for b in local_faces:
                if a != b:
                    local_adjacency[a].add(b)

    seen: set[int] = set()
    component_count = 0
    for start in range(len(patch_faces)):
        if start in seen:
            continue
        component_count += 1
        stack = [start]
        seen.add(start)
        while stack:
            current = stack.pop()
            for nxt in local_adjacency[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)

    seam_ratio = (
        float(len(seam_edges) / len(expected_seam_edges))
        if expected_seam_edges
        else 0.0
    )

    return PatchTopologyReport(
        patch_face_count=int(len(patch_faces)),
        patch_vertex_count=int(len(patch_vertices)),
        patch_component_count=int(component_count),
        patch_boundary_edge_count=int(len(patch_boundary_edges)),
        expected_seam_edge_count=int(len(expected_seam_edges)),
        seam_edge_count=int(len(seam_edges)),
        missing_seam_edge_count=int(len(missing_seam_edges)),
        extra_open_boundary_edge_count=int(len(extra_open_edges)),
        nonmanifold_patch_edge_count=int(len(nonmanifold_edges)),
        seam_coverage_ratio=seam_ratio,
    )


def collect_normal_compatible_support_context(
    mesh: trimesh.Trimesh,
    boundary_vertex_ids: Sequence[int],
    *,
    max_rings: int = 2,
    smooth_dot_threshold: float = 0.50,
) -> NormalCompatibleSupportContext:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    if int(max_rings) < 0:
        raise ValueError("max_rings must be >= 0")
    if not (-1.0 <= float(smooth_dot_threshold) <= 1.0):
        raise ValueError("smooth_dot_threshold must be in [-1, 1]")

    boundary_ids = _as_index_tuple(boundary_vertex_ids)
    if not boundary_ids:
        raise ValueError("boundary_vertex_ids must not be empty")

    vertex_to_faces = _build_vertex_to_faces(faces)
    seed_faces = sorted(
        {
            int(face_id)
            for vertex_id in boundary_ids
            for face_id in vertex_to_faces.get(int(vertex_id), set())
        }
    )

    if not seed_faces:
        raise ValueError("could not collect any support seed faces")

    face_normals = np.asarray([_face_normal(vertices, face) for face in faces], dtype=float)
    adjacency = _build_face_adjacency(faces)

    support: set[int] = set(int(face_id) for face_id in seed_faces)
    rejected: set[int] = set()
    frontier: list[tuple[int, int]] = [(int(face_id), 0) for face_id in seed_faces]

    while frontier:
        face_id, depth = frontier.pop(0)
        if depth >= int(max_rings):
            continue

        current_normal = face_normals[int(face_id)]
        for neighbour in sorted(adjacency.get(int(face_id), set())):
            if neighbour in support:
                continue

            neighbour_normal = face_normals[int(neighbour)]
            dot = float(np.dot(current_normal, neighbour_normal))
            if dot >= float(smooth_dot_threshold):
                support.add(int(neighbour))
                frontier.append((int(neighbour), depth + 1))
            else:
                rejected.add(int(neighbour))

    support_face_ids = tuple(sorted(support))
    rejected_face_ids = tuple(sorted(rejected - support))
    support_vertex_ids = tuple(
        sorted({int(v) for face_id in support_face_ids for v in faces[int(face_id), :3]})
    )

    weighted_normal = np.zeros(3, dtype=float)
    for face_id in support_face_ids:
        face = faces[int(face_id)]
        area2 = _triangle_area2(vertices[np.asarray(face[:3], dtype=np.int64)])
        weighted_normal += face_normals[int(face_id)] * max(area2, 1.0e-12)

    mean_normal = _normalize(weighted_normal)
    if mean_normal is None:
        mean_normal = np.asarray([0.0, 0.0, 1.0], dtype=float)

    angles: list[float] = []
    for face_id in support_face_ids:
        n = face_normals[int(face_id)]
        if float(np.linalg.norm(n)) <= 1.0e-12:
            continue
        dot = float(np.clip(np.dot(mean_normal, n), -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(dot))))

    normal_spread = float(max(angles)) if angles else 0.0
    contamination = (
        float(len(rejected_face_ids) / max(len(rejected_face_ids) + len(support_face_ids), 1))
    )

    return NormalCompatibleSupportContext(
        boundary_vertex_ids=boundary_ids,
        seed_face_ids=tuple(int(v) for v in seed_faces),
        support_face_ids=support_face_ids,
        support_vertex_ids=support_vertex_ids,
        rejected_face_ids=rejected_face_ids,
        mean_normal=tuple(float(v) for v in mean_normal),
        normal_spread_degrees=normal_spread,
        contamination_score=contamination,
        smooth_dot_threshold=float(smooth_dot_threshold),
        max_rings=int(max_rings),
    )


def build_mls_surface_projector(
    mesh: trimesh.Trimesh,
    *,
    support_face_ids: Sequence[int],
    nearest_count: int = 16,
    ridge: float = 1.0e-10,
) -> LocalMLSSurfaceProjector:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_ids = _as_index_tuple(support_face_ids)

    if not face_ids:
        raise ValueError("support_face_ids must not be empty")

    support_vertex_ids = tuple(
        sorted({int(v) for face_id in face_ids for v in faces[int(face_id), :3]})
    )
    if len(support_vertex_ids) < 3:
        raise ValueError("MLS projector requires at least three support vertices")

    samples = vertices[list(support_vertex_ids)]

    weighted_centroid = np.zeros(3, dtype=float)
    total_weight = 0.0
    weighted_normal = np.zeros(3, dtype=float)

    for face_id in face_ids:
        face = faces[int(face_id)]
        pts = vertices[np.asarray(face[:3], dtype=np.int64)]
        area2 = max(_triangle_area2(pts), 1.0e-12)
        weighted_centroid += np.mean(pts, axis=0) * area2
        weighted_normal += _face_normal(vertices, face) * area2
        total_weight += area2

    if total_weight <= 1.0e-12:
        centroid = np.mean(samples, axis=0)
    else:
        centroid = weighted_centroid / total_weight

    normal = _normalize(weighted_normal)
    if normal is None:
        centered = samples - np.mean(samples, axis=0)
        try:
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
            normal = _normalize(vh[-1])
        except Exception:
            normal = None
    if normal is None:
        normal = np.asarray([0.0, 0.0, 1.0], dtype=float)

    tangent_u, tangent_v, normal_w = _orthonormal_basis(normal)

    local = samples - centroid.reshape(1, 3)
    sample_uv = np.column_stack(
        [
            local @ tangent_u,
            local @ tangent_v,
        ]
    )
    sample_w = local @ normal_w

    return LocalMLSSurfaceProjector(
        centroid=np.asarray(centroid, dtype=float),
        tangent_u=np.asarray(tangent_u, dtype=float),
        tangent_v=np.asarray(tangent_v, dtype=float),
        normal_w=np.asarray(normal_w, dtype=float),
        sample_uv=np.asarray(sample_uv, dtype=float),
        sample_w=np.asarray(sample_w, dtype=float),
        nearest_count=int(nearest_count),
        ridge=float(ridge),
    )
