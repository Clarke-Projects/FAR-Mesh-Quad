from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import trimesh

from far_mesh.core.hole_patch_quality import PatchQualityReport, analyze_patch_quality


@dataclass(frozen=True)
class PatchRelaxationResult:
    """Result payload for non-destructive patch relaxation."""

    preview_mesh: trimesh.Trimesh
    iterations: int
    accepted_updates: int
    rejected_updates: int
    moved_vertex_ids: tuple[int, ...]
    fixed_vertex_ids: tuple[int, ...]
    patch_face_ids: tuple[int, ...]
    quality_before: PatchQualityReport
    quality_after: PatchQualityReport
    max_displacement: float
    mean_displacement: float
    notes: tuple[str, ...]

    def to_metadata(self) -> dict[str, object]:
        return {
            "iterations": int(self.iterations),
            "accepted_updates": int(self.accepted_updates),
            "rejected_updates": int(self.rejected_updates),
            "moved_vertex_ids": tuple(int(v) for v in self.moved_vertex_ids),
            "fixed_vertex_ids": tuple(int(v) for v in self.fixed_vertex_ids),
            "patch_face_ids": tuple(int(v) for v in self.patch_face_ids),
            "quality_before": self.quality_before.to_dict(),
            "quality_after": self.quality_after.to_dict(),
            "max_displacement": float(self.max_displacement),
            "mean_displacement": float(self.mean_displacement),
            "notes": tuple(str(note) for note in self.notes),
        }


def _as_index_tuple(values: Sequence[int] | np.ndarray | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def _unique_edges_from_faces(faces: np.ndarray) -> tuple[tuple[int, int], ...]:
    edges: set[tuple[int, int]] = set()
    for face in np.asarray(faces, dtype=np.int64):
        tri = [int(face[0]), int(face[1]), int(face[2])]
        for index, a in enumerate(tri):
            b = tri[(index + 1) % 3]
            if a == b:
                continue
            edges.add((a, b) if a < b else (b, a))
    return tuple(sorted(edges))


def _build_patch_adjacency(
    faces: np.ndarray,
    patch_face_ids: tuple[int, ...],
) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}

    for face_id in patch_face_ids:
        face = np.asarray(faces[int(face_id), :3], dtype=np.int64)
        for index, a in enumerate(face):
            b = int(face[(index + 1) % 3])
            c = int(face[(index + 2) % 3])
            adjacency.setdefault(int(a), set()).update({b, c})

    return adjacency


def _build_vertex_to_patch_faces(
    faces: np.ndarray,
    patch_face_ids: tuple[int, ...],
) -> dict[int, tuple[int, ...]]:
    mapping: dict[int, list[int]] = {}

    for face_id in patch_face_ids:
        face = np.asarray(faces[int(face_id), :3], dtype=np.int64)
        for vertex_id in face:
            mapping.setdefault(int(vertex_id), []).append(int(face_id))

    return {
        int(vertex_id): tuple(int(face_id) for face_id in face_ids)
        for vertex_id, face_ids in mapping.items()
    }


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


def _triangle_aspect_ratio(points: np.ndarray) -> float:
    lengths = _triangle_edge_lengths(points)
    shortest = min(lengths)
    longest = max(lengths)
    area2 = _triangle_area2(points)

    if shortest <= 1.0e-12 or area2 <= 1.0e-12:
        return float("inf")

    return float(longest / shortest)


def _candidate_update_is_safe(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    affected_face_ids: tuple[int, ...],
    max_edge_length: float,
    max_triangle_aspect_ratio: float,
) -> bool:
    for face_id in affected_face_ids:
        face = np.asarray(faces[int(face_id), :3], dtype=np.int64)
        pts = vertices[face]

        if _triangle_area2(pts) <= 1.0e-12:
            return False

        lengths = _triangle_edge_lengths(pts)
        if max(lengths) > max_edge_length:
            return False

        if _triangle_aspect_ratio(pts) > max_triangle_aspect_ratio:
            return False

    return True


def _max_patch_edge_length_from_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    patch_face_ids: tuple[int, ...],
) -> float:
    patch_faces = faces[list(patch_face_ids)]
    edges = _unique_edges_from_faces(patch_faces)
    max_length = 0.0

    for a, b in edges:
        length = float(np.linalg.norm(vertices[int(b)] - vertices[int(a)]))
        if np.isfinite(length):
            max_length = max(max_length, length)

    return float(max_length)


def relax_patch_vertices(
    mesh: trimesh.Trimesh,
    *,
    patch_face_ids: Sequence[int],
    movable_vertex_ids: Sequence[int],
    fixed_vertex_ids: Sequence[int] = (),
    iterations: int = 8,
    relaxation_strength: float = 0.35,
    surface_projector: Callable[[int, np.ndarray, np.ndarray], np.ndarray] | None = None,
    surface_weight: float = 0.0,
    max_edge_length_factor: float = 3.0,
    max_triangle_aspect_ratio: float = 50.0,
    context_edge_length_median: float | None = None,
    target_boundary_normals: Mapping[int, np.ndarray] | None = None,
) -> PatchRelaxationResult:
    """Relax movable patch vertices without mutating the source mesh.

    This is the first isolated dynamic-surface kernel. It intentionally keeps
    the update rule simple and deterministic:

    - boundary/fixed vertices never move
    - only movable vertices are updated
    - each movable vertex is pulled toward the average of its patch neighbours
    - a candidate move is rejected if it degenerates affected faces, creates an
      excessive edge length, or violates an aspect-ratio guard

    Later checkpoints can add stronger surface/normal guidance and a true
    biharmonic/G1 solve on top of this safety skeleton.
    """

    if int(iterations) < 0:
        raise ValueError("iterations must be >= 0")
    if not (0.0 <= float(relaxation_strength) <= 1.0):
        raise ValueError("relaxation_strength must be in [0, 1]")
    if not (0.0 <= float(surface_weight) <= 1.0):
        raise ValueError("surface_weight must be in [0, 1]")
    if float(relaxation_strength) + float(surface_weight) > 1.0 + 1.0e-12:
        raise ValueError("relaxation_strength + surface_weight must be <= 1")
    if float(surface_weight) > 0.0 and surface_projector is None:
        raise ValueError("surface_projector is required when surface_weight > 0")
    if float(max_edge_length_factor) <= 0.0:
        raise ValueError("max_edge_length_factor must be > 0")
    if float(max_triangle_aspect_ratio) <= 1.0:
        raise ValueError("max_triangle_aspect_ratio must be > 1")

    patch_ids = _as_index_tuple(patch_face_ids)
    movable_ids = _as_index_tuple(movable_vertex_ids)
    fixed_ids = _as_index_tuple(fixed_vertex_ids)

    if not patch_ids:
        raise ValueError("patch_face_ids must not be empty")

    movable_set = set(movable_ids)
    fixed_set = set(fixed_ids)
    overlap = sorted(movable_set & fixed_set)
    if overlap:
        raise ValueError(
            "movable and fixed vertex sets overlap: "
            + ", ".join(str(v) for v in overlap)
        )

    source_vertices_before = np.asarray(mesh.vertices).copy()
    source_faces_before = np.asarray(mesh.faces).copy()

    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()

    if max(patch_ids) >= len(faces) or min(patch_ids) < 0:
        raise ValueError("patch_face_ids contain an out-of-range face id")
    if movable_ids and (max(movable_ids) >= len(vertices) or min(movable_ids) < 0):
        raise ValueError("movable_vertex_ids contain an out-of-range vertex id")
    if fixed_ids and (max(fixed_ids) >= len(vertices) or min(fixed_ids) < 0):
        raise ValueError("fixed_vertex_ids contain an out-of-range vertex id")

    quality_before = analyze_patch_quality(
        mesh,
        patch_face_ids=patch_ids,
        boundary_vertex_ids=fixed_ids,
        movable_vertex_ids=movable_ids,
        context_edge_length_median=context_edge_length_median,
        target_boundary_normals=target_boundary_normals,
    )

    if not movable_ids:
        preview = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        if not np.allclose(np.asarray(mesh.vertices), source_vertices_before):
            raise RuntimeError("relax_patch_vertices unexpectedly mutated source vertices")
        if not np.array_equal(np.asarray(mesh.faces), source_faces_before):
            raise RuntimeError("relax_patch_vertices unexpectedly mutated source faces")

        quality_after = analyze_patch_quality(
            preview,
            patch_face_ids=patch_ids,
            boundary_vertex_ids=fixed_ids,
            movable_vertex_ids=movable_ids,
            context_edge_length_median=context_edge_length_median,
            target_boundary_normals=target_boundary_normals,
        )

        return PatchRelaxationResult(
            preview_mesh=preview,
            iterations=int(iterations),
            accepted_updates=0,
            rejected_updates=0,
            moved_vertex_ids=(),
            fixed_vertex_ids=fixed_ids,
            patch_face_ids=patch_ids,
            quality_before=quality_before,
            quality_after=quality_after,
            max_displacement=0.0,
            mean_displacement=0.0,
            notes=(
                "Patch relaxation is preview-only and non-destructive.",
                "No movable vertices were provided; boundary-only patch relaxation was skipped.",
                "Boundary/fixed vertices were held fixed.",
                "No candidate updates were attempted.",
            ),
        )

    adjacency = _build_patch_adjacency(faces, patch_ids)
    vertex_to_patch_faces = _build_vertex_to_patch_faces(faces, patch_ids)

    base_edge_length = (
        float(context_edge_length_median)
        if context_edge_length_median is not None and float(context_edge_length_median) > 1.0e-12
        else float(max(quality_before.patch_edge_length_median, 1.0e-12))
    )
    max_edge_length = float(base_edge_length * max_edge_length_factor)

    # Never make the guard stricter than the existing patch; this lets the
    # kernel improve poor patches instead of rejecting every update immediately.
    existing_max_patch_edge = _max_patch_edge_length_from_vertices(vertices, faces, patch_ids)
    max_edge_length = max(max_edge_length, existing_max_patch_edge * 1.05)

    accepted = 0
    rejected = 0

    for _iteration in range(int(iterations)):
        for vertex_id in movable_ids:
            neighbours = sorted(int(v) for v in adjacency.get(int(vertex_id), set()))
            if not neighbours:
                rejected += 1
                continue

            current = vertices[int(vertex_id)].copy()
            laplacian_target = np.mean(vertices[neighbours], axis=0)

            if surface_projector is not None and float(surface_weight) > 0.0:
                surface_target = np.asarray(
                    surface_projector(int(vertex_id), current.copy(), laplacian_target.copy()),
                    dtype=float,
                ).reshape(3)
                if not np.isfinite(surface_target).all():
                    rejected += 1
                    continue

                keep_weight = 1.0 - float(relaxation_strength) - float(surface_weight)
                proposal = (
                    keep_weight * current
                    + float(relaxation_strength) * laplacian_target
                    + float(surface_weight) * surface_target
                )
            else:
                proposal = (
                    (1.0 - float(relaxation_strength)) * current
                    + float(relaxation_strength) * laplacian_target
                )

            if not np.isfinite(proposal).all():
                rejected += 1
                continue

            affected = vertex_to_patch_faces.get(int(vertex_id), ())
            if not affected:
                rejected += 1
                continue

            previous = vertices[int(vertex_id)].copy()
            vertices[int(vertex_id)] = proposal

            if _candidate_update_is_safe(
                vertices,
                faces,
                affected_face_ids=affected,
                max_edge_length=max_edge_length,
                max_triangle_aspect_ratio=float(max_triangle_aspect_ratio),
            ):
                accepted += 1
            else:
                vertices[int(vertex_id)] = previous
                rejected += 1

    preview = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if not np.allclose(np.asarray(mesh.vertices), source_vertices_before):
        raise RuntimeError("relax_patch_vertices unexpectedly mutated source vertices")
    if not np.array_equal(np.asarray(mesh.faces), source_faces_before):
        raise RuntimeError("relax_patch_vertices unexpectedly mutated source faces")

    quality_after = analyze_patch_quality(
        preview,
        patch_face_ids=patch_ids,
        boundary_vertex_ids=fixed_ids,
        movable_vertex_ids=movable_ids,
        context_edge_length_median=context_edge_length_median,
        target_boundary_normals=target_boundary_normals,
    )

    displacements = np.linalg.norm(vertices[list(movable_ids)] - source_vertices_before[list(movable_ids)], axis=1)

    return PatchRelaxationResult(
        preview_mesh=preview,
        iterations=int(iterations),
        accepted_updates=int(accepted),
        rejected_updates=int(rejected),
        moved_vertex_ids=movable_ids,
        fixed_vertex_ids=fixed_ids,
        patch_face_ids=patch_ids,
        quality_before=quality_before,
        quality_after=quality_after,
        max_displacement=float(np.max(displacements)) if len(displacements) else 0.0,
        mean_displacement=float(np.mean(displacements)) if len(displacements) else 0.0,
        notes=(
            "Patch relaxation is preview-only and non-destructive.",
            "Boundary/fixed vertices were held fixed.",
            "Only movable patch vertices were updated.",
            "Optional surface guidance blended movable vertices toward a projected local surface target.",
            "Candidate moves were rejected when they violated degeneracy, edge-length, or aspect-ratio guards.",
        ),
    )
