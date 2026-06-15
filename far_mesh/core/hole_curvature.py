# H-CORE-V2 LEGACY BOUNDARY
# This file is now treated as Adaptive Surface Fill v1 / experimental legacy geometry.
# Do not add new Adaptive Surface Fill v2 seed/target/controller logic here.
# New v2 work belongs in far_mesh/core/hole_fill/.
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from far_mesh.core.hole_context import (
    build_local_hole_context_mesh,
    candidate_boundary_vertex_ids,
)


@dataclass(frozen=True)
class SphereFit:
    center: tuple[float, float, float]
    radius: float
    rms_error: float
    max_abs_error: float
    point_count: int
    rank: int


def fit_sphere_to_points(points: np.ndarray) -> SphereFit:
    pts = np.asarray(points, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {pts.shape!r}.")
    if len(pts) < 4:
        raise ValueError("Sphere fitting requires at least four points.")
    if not np.isfinite(pts).all():
        raise ValueError("Sphere fitting requires finite points.")

    a = np.column_stack(
        [
            2.0 * pts[:, 0],
            2.0 * pts[:, 1],
            2.0 * pts[:, 2],
            np.ones(len(pts), dtype=float),
        ]
    )
    b = np.sum(pts * pts, axis=1)

    solution, _residuals, rank, _singular_values = np.linalg.lstsq(a, b, rcond=None)

    if int(rank) < 4:
        raise ValueError(
            "Sphere fitting is under-constrained; points are likely planar or degenerate."
        )

    center = np.asarray(solution[:3], dtype=float)
    d = float(solution[3])
    radius_sq = float(np.dot(center, center) + d)

    if radius_sq <= 0.0 or not np.isfinite(radius_sq):
        raise ValueError("Sphere fitting produced an invalid radius.")

    radius = float(np.sqrt(radius_sq))
    distances = np.linalg.norm(pts - center, axis=1)
    errors = distances - radius

    return SphereFit(
        center=(float(center[0]), float(center[1]), float(center[2])),
        radius=radius,
        rms_error=float(np.sqrt(np.mean(errors * errors))),
        max_abs_error=float(np.max(np.abs(errors))),
        point_count=int(len(pts)),
        rank=int(rank),
    )


def project_points_to_sphere(points: np.ndarray, fit: SphereFit) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    center = np.asarray(fit.center, dtype=float)
    directions = pts - center
    norms = np.linalg.norm(directions, axis=1)

    if np.any(norms <= 1.0e-12):
        raise ValueError("Cannot project a point at the fitted sphere center.")

    return center + directions / norms[:, None] * float(fit.radius)


def fit_sphere_for_hole_context(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int = 1,
) -> dict[str, Any]:
    context = build_local_hole_context_mesh(mesh, candidate, rings=rings)
    points = np.asarray(mesh.vertices, dtype=float)[list(context.source_vertex_ids)]
    fit = fit_sphere_to_points(points)

    return {
        "operation": "fit_sphere_for_hole_context",
        "rings": int(rings),
        "point_count": fit.point_count,
        "source_vertex_ids": context.source_vertex_ids,
        "target_boundary_source_vertex_ids": context.target_boundary_source_vertex_ids,
        "fit": fit,
        "notes": (
            "Diagnostic sphere fit only; source mesh was not modified.",
            "This fit can be used by a future curvature-aware preview backend.",
        ),
    }


def build_curvature_sphere_center_fan_preview_mesh(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int = 1,
) -> dict[str, Any]:
    source_vertices_before = np.asarray(mesh.vertices).copy()
    source_faces_before = np.asarray(mesh.faces).copy()

    boundary_ids = candidate_boundary_vertex_ids(candidate)
    if len(boundary_ids) < 3:
        raise ValueError("Curvature sphere fan requires at least three boundary vertices.")

    fit_report = fit_sphere_for_hole_context(mesh, candidate, rings=rings)
    fit = fit_report["fit"]

    vertices = np.array(mesh.vertices, dtype=float, copy=True)
    faces = np.array(mesh.faces, dtype=np.int64, copy=True)

    boundary_points = vertices[list(boundary_ids)]
    boundary_centroid = np.mean(boundary_points, axis=0, keepdims=True)
    projected_center = project_points_to_sphere(boundary_centroid, fit)[0]

    center_vertex_id = int(len(vertices))
    out_vertices = np.vstack([vertices, projected_center])

    fan_faces = []
    for index, vertex_id in enumerate(boundary_ids):
        next_vertex_id = boundary_ids[(index + 1) % len(boundary_ids)]
        fan_faces.append([int(vertex_id), int(next_vertex_id), center_vertex_id])

    out_faces = np.vstack([faces, np.asarray(fan_faces, dtype=np.int64)])
    preview = trimesh.Trimesh(vertices=out_vertices, faces=out_faces, process=False)

    if not np.allclose(np.asarray(mesh.vertices), source_vertices_before):
        raise RuntimeError("curvature sphere preview unexpectedly mutated source vertices")
    if not np.array_equal(np.asarray(mesh.faces), source_faces_before):
        raise RuntimeError("curvature sphere preview unexpectedly mutated source faces")

    return {
        "operation": "curvature_sphere_center_fan_preview",
        "preview_mesh": preview,
        "rings": int(rings),
        "fit": fit,
        "new_vertex_id": center_vertex_id,
        "new_face_ids": tuple(range(len(faces), len(out_faces))),
        "boundary_vertex_ids": boundary_ids,
        "projected_center": (
            float(projected_center[0]),
            float(projected_center[1]),
            float(projected_center[2]),
        ),
        "notes": (
            "Preview-only curvature sphere center-fan patch.",
            "Boundary vertices are fixed.",
            "One new center vertex was projected onto the fitted local sphere.",
            "No source mesh mutation or project/history write occurred.",
        ),
    }




def _triangle_quality_penalty(points: np.ndarray) -> float:
    """Return a small-is-good triangle quality penalty.

    The value is intentionally simple and deterministic:
    - infinite for degenerate triangles
    - lower for triangles with less extreme edge ratios
    """

    pts = np.asarray(points, dtype=float)
    if pts.shape != (3, 3) or not np.isfinite(pts).all():
        return float("inf")

    a, b, c = pts
    ab = float(np.linalg.norm(b - a))
    bc = float(np.linalg.norm(c - b))
    ca = float(np.linalg.norm(a - c))

    shortest = min(ab, bc, ca)
    longest = max(ab, bc, ca)
    if shortest <= 1.0e-12:
        return float("inf")

    area2 = float(np.linalg.norm(np.cross(b - a, c - a)))
    if area2 <= 1.0e-12:
        return float("inf")

    # Edge ratio catches skinny triangles. A light area term helps avoid
    # nearly-flat degenerate cases when the edge ratios are similar.
    return (longest / shortest) + (1.0 / area2) * 1.0e-6


def _triangle_pair_quality_penalty(
    vertices: np.ndarray,
    face_a: tuple[int, int, int],
    face_b: tuple[int, int, int],
) -> float:
    """Return a combined quality penalty for two triangles."""

    verts = np.asarray(vertices, dtype=float)
    return max(
        _triangle_quality_penalty(verts[list(face_a)]),
        _triangle_quality_penalty(verts[list(face_b)]),
    ) + 0.25 * (
        _triangle_quality_penalty(verts[list(face_a)])
        + _triangle_quality_penalty(verts[list(face_b)])
    )


def _best_refined_patch_quad_split(
    vertices: np.ndarray,
    *,
    a: int,
    b: int,
    c: int,
    d: int,
) -> tuple[list[int], list[int]]:
    """Choose the better diagonal split for one refined patch strip cell.

    The cell is:

        a ---- b      previous / outer ring
        |      |
        c ---- d      current / inner ring

    Candidate splits:
    - diagonal a-d: [a, b, d] + [a, d, c]
    - diagonal b-c: [a, b, c] + [b, d, c]

    This removes the fixed diagonal bias that produced visible strip/ripple
    artifacts in the first refined curvature patch prototype.
    """

    split_ad = ([int(a), int(b), int(d)], [int(a), int(d), int(c)])
    split_bc = ([int(a), int(b), int(c)], [int(b), int(d), int(c)])

    score_ad = _triangle_pair_quality_penalty(
        vertices,
        tuple(split_ad[0]),  # type: ignore[arg-type]
        tuple(split_ad[1]),  # type: ignore[arg-type]
    )
    score_bc = _triangle_pair_quality_penalty(
        vertices,
        tuple(split_bc[0]),  # type: ignore[arg-type]
        tuple(split_bc[1]),  # type: ignore[arg-type]
    )

    if score_bc < score_ad:
        return split_bc
    return split_ad



def _context_median_edge_length(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int,
) -> float:
    """Return median unique edge length from the local N-ring hole context."""

    context = build_local_hole_context_mesh(mesh, candidate, rings=rings)
    vertices = np.asarray(context.mesh.vertices, dtype=float)
    faces = np.asarray(context.mesh.faces, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("local context mesh has invalid vertices")
    if faces.ndim != 2 or faces.shape[1] < 3:
        raise ValueError("local context mesh has invalid faces")

    edges: set[tuple[int, int]] = set()
    for face in faces:
        tri = [int(v) for v in face[:3]]
        for index, a in enumerate(tri):
            b = tri[(index + 1) % len(tri)]
            if a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            edges.add(edge)

    lengths: list[float] = []
    for a, b in sorted(edges):
        if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
            continue
        length = float(np.linalg.norm(vertices[b] - vertices[a]))
        if np.isfinite(length) and length > 1.0e-12:
            lengths.append(length)

    if not lengths:
        # Fallback to boundary median if context edge extraction failed.
        boundary_ids = candidate_boundary_vertex_ids(candidate)
        source_vertices = np.asarray(mesh.vertices, dtype=float)
        for index, a in enumerate(boundary_ids):
            b = boundary_ids[(index + 1) % len(boundary_ids)]
            length = float(np.linalg.norm(source_vertices[int(b)] - source_vertices[int(a)]))
            if np.isfinite(length) and length > 1.0e-12:
                lengths.append(length)

    if not lengths:
        raise ValueError("could not estimate local mesh edge length for automatic refinement")

    return float(np.median(np.asarray(lengths, dtype=float)))


def estimate_curvature_sphere_refined_interior_rings(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    fit: SphereFit,
    projected_center: np.ndarray,
    rings: int = 2,
    min_interior_rings: int | None = None,
    max_interior_rings: int = 4,
) -> dict[str, Any]:
    """Estimate refined patch ring count from spherical angular span and mesh density.

    Shape-independent rule:
    - measure directions from fitted sphere center, not world axes
    - compute angle from each boundary point direction to projected center direction
    - convert angular span to arc length with s = r * theta
    - choose enough radial steps so each step is close to local context edge length
    """

    if min_interior_rings is None:
        min_interior_rings = 1

    if min_interior_rings < 0:
        raise ValueError("min_interior_rings must be >= 0")
    if max_interior_rings < min_interior_rings:
        raise ValueError("max_interior_rings must be >= min_interior_rings")

    boundary_ids = candidate_boundary_vertex_ids(candidate)
    if len(boundary_ids) < 3:
        raise ValueError("automatic refined ring estimate requires at least three boundary vertices")

    vertices = np.asarray(mesh.vertices, dtype=float)
    boundary_points = vertices[list(boundary_ids)]

    center = np.asarray(fit.center, dtype=float).reshape(3)
    radius = float(fit.radius)
    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError("automatic refined ring estimate requires a valid fitted radius")

    center_direction = np.asarray(projected_center, dtype=float).reshape(3) - center
    center_norm = float(np.linalg.norm(center_direction))
    if center_norm <= 1.0e-12:
        raise ValueError("projected center is too close to fitted sphere center")
    center_direction /= center_norm

    boundary_dirs = boundary_points - center.reshape(1, 3)
    boundary_norms = np.linalg.norm(boundary_dirs, axis=1)
    if np.any(boundary_norms <= 1.0e-12):
        raise ValueError("boundary contains point too close to fitted sphere center")
    boundary_dirs = boundary_dirs / boundary_norms[:, None]

    dots = np.clip(boundary_dirs @ center_direction, -1.0, 1.0)
    angles = np.arccos(dots)

    # Use a high percentile rather than max so one noisy boundary vertex does
    # not over-subdivide the whole patch. This is the radial low-to-high span.
    angular_span = float(np.percentile(angles, 75.0))
    angular_span = max(0.0, angular_span)

    arc_length = float(radius * angular_span)
    target_edge_length = _context_median_edge_length(mesh, candidate, rings=rings)

    # Number of radial segments from boundary to center. Interior rings are
    # segments - 1 because the center is the final vertex/ring.
    radial_segments = int(np.ceil(arc_length / max(target_edge_length, 1.0e-12)))
    interior_rings = radial_segments - 1
    interior_rings = max(int(min_interior_rings), min(int(max_interior_rings), int(interior_rings)))

    return {
        "operation": "estimate_curvature_sphere_refined_interior_rings",
        "rings": int(rings),
        "interior_rings": int(interior_rings),
        "radial_segments": int(interior_rings + 1),
        "angular_span_radians": float(angular_span),
        "angular_span_degrees": float(np.degrees(angular_span)),
        "arc_length": float(arc_length),
        "target_edge_length": float(target_edge_length),
        "radius": float(radius),
        "min_interior_rings": int(min_interior_rings),
        "max_interior_rings": int(max_interior_rings),
        "boundary_vertex_count": int(len(boundary_ids)),
    }

def build_curvature_sphere_refined_preview_mesh(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int = 2,
    interior_rings: int = 1,
) -> dict[str, Any]:
    """Build a preview-only refined spherical cap patch.

    Compared with build_curvature_sphere_center_fan_preview_mesh(), this adds
    one or more projected interior rings before the final projected center fan.
    Boundary vertices are fixed. Generated vertices are projected to the fitted
    local sphere, reducing radial one-center fan artifacts while staying
    dependency-light and deterministic.
    """

    source_vertices_before = np.asarray(mesh.vertices).copy()
    source_faces_before = np.asarray(mesh.faces).copy()

    boundary_ids = candidate_boundary_vertex_ids(candidate)
    if len(boundary_ids) < 3:
        raise ValueError("Curvature sphere refined patch requires at least three boundary vertices.")
    if interior_rings is not None and int(interior_rings) < 1:
        raise ValueError("interior_rings must be >= 1")

    fit_report = fit_sphere_for_hole_context(mesh, candidate, rings=rings)
    fit = fit_report["fit"]

    vertices = np.array(mesh.vertices, dtype=float, copy=True)
    faces = np.array(mesh.faces, dtype=np.int64, copy=True)

    boundary_points = vertices[list(boundary_ids)]
    boundary_centroid = np.mean(boundary_points, axis=0, keepdims=True)
    projected_center = project_points_to_sphere(boundary_centroid, fit)[0]

    auto_ring_report: dict[str, Any] | None = None
    if interior_rings is None:
        auto_ring_report = estimate_curvature_sphere_refined_interior_rings(
            mesh,
            candidate,
            fit=fit,
            projected_center=projected_center,
            rings=rings,
            min_interior_rings=1,
            max_interior_rings=4,
        )
        interior_rings = int(auto_ring_report["interior_rings"])
    else:
        interior_rings = int(interior_rings)

    generated_vertex_blocks: list[np.ndarray] = []
    generated_ring_ids: list[tuple[int, ...]] = []
    next_vertex_id = int(len(vertices))

    for ring_index in range(1, int(interior_rings) + 1):
        t = float(ring_index) / float(int(interior_rings) + 1)

        # Linear interpolation gives stable parameter placement; projection
        # moves those samples onto the fitted sphere around fit.center.
        raw_ring = (1.0 - t) * boundary_points + t * projected_center.reshape(1, 3)
        projected_ring = project_points_to_sphere(raw_ring, fit)

        ring_ids = tuple(range(next_vertex_id, next_vertex_id + len(projected_ring)))
        next_vertex_id += len(projected_ring)

        generated_vertex_blocks.append(projected_ring)
        generated_ring_ids.append(ring_ids)

    center_vertex_id = next_vertex_id
    generated_vertex_blocks.append(projected_center.reshape(1, 3))

    out_vertices = np.vstack([vertices, *generated_vertex_blocks])

    patch_faces: list[list[int]] = []
    previous_ring = tuple(int(v) for v in boundary_ids)

    for current_ring in generated_ring_ids:
        for index, a in enumerate(previous_ring):
            b = previous_ring[(index + 1) % len(previous_ring)]
            c = current_ring[index]
            d = current_ring[(index + 1) % len(current_ring)]

            face_a, face_b = _best_refined_patch_quad_split(
                out_vertices,
                a=int(a),
                b=int(b),
                c=int(c),
                d=int(d),
            )
            patch_faces.append(face_a)
            patch_faces.append(face_b)

        previous_ring = current_ring

    for index, vertex_id in enumerate(previous_ring):
        next_vertex_id_on_ring = previous_ring[(index + 1) % len(previous_ring)]
        patch_faces.append([int(vertex_id), int(next_vertex_id_on_ring), int(center_vertex_id)])

    if not patch_faces:
        raise ValueError("Curvature sphere refined patch could not build patch faces.")

    seam_constraint_report = build_patch_seam_constraint_report(
        vertices=out_vertices,
        patch_faces=patch_faces,
        boundary_vertex_ids=boundary_ids,
    )

    patch_faces_array = np.asarray(patch_faces, dtype=np.int64)
    out_faces = np.vstack([faces, patch_faces_array])
    preview = trimesh.Trimesh(vertices=out_vertices, faces=out_faces, process=False)

    if not np.allclose(np.asarray(mesh.vertices), source_vertices_before):
        raise RuntimeError("curvature sphere refined preview unexpectedly mutated source vertices")
    if not np.array_equal(np.asarray(mesh.faces), source_faces_before):
        raise RuntimeError("curvature sphere refined preview unexpectedly mutated source faces")

    new_vertex_ids = tuple(range(len(vertices), len(out_vertices)))
    new_face_ids = tuple(range(len(faces), len(out_faces)))

    return {
        "operation": "curvature_sphere_refined_preview",
        "preview_mesh": preview,
        "rings": int(rings),
        "interior_rings": int(interior_rings),
        "auto_interior_rings": auto_ring_report is not None,
        "auto_ring_report": auto_ring_report,
        "fit": fit,
        "new_vertex_ids": new_vertex_ids,
        "new_face_ids": new_face_ids,
        "ring_vertex_ids": tuple(tuple(int(v) for v in ring) for ring in generated_ring_ids),
        "center_vertex_id": int(center_vertex_id),
        "boundary_vertex_ids": boundary_ids,
        "projected_center": (
            float(projected_center[0]),
            float(projected_center[1]),
            float(projected_center[2]),
        ),
        "notes": (
            "Preview-only refined curvature sphere patch.",
            "Boundary vertices are fixed.",
            "Generated interior ring vertices were projected onto the fitted local sphere.",
            "Generated center vertex was projected onto the fitted local sphere.",
            "No source mesh mutation or project/history write occurred.",
        ),
    }


def build_curvature_sphere_grid8_preview_mesh(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int = 2,
    interior_rings: int | None = None,
) -> dict[str, Any]:
    """Build a preview-only spherical cap with radial + directional support.

    This is the next step after curvature_sphere_refined.

    Instead of only creating one generated point per boundary vertex per ring,
    this method creates an alternating generated ring:

        boundary-vertex support point
        boundary-edge midpoint support point
        boundary-vertex support point
        boundary-edge midpoint support point
        ...

    The first strip uses the original boundary edge exactly once, so the patch
    still closes against the existing mesh boundary without needing to split
    surrounding faces. Generated vertices are projected onto the fitted local
    sphere. Source mesh vertices/faces are not mutated.
    """

    source_vertices_before = np.asarray(mesh.vertices).copy()
    source_faces_before = np.asarray(mesh.faces).copy()

    boundary_ids = candidate_boundary_vertex_ids(candidate)
    if len(boundary_ids) < 3:
        raise ValueError("curvature_sphere_grid8 requires at least three boundary vertices")

    fit_report = fit_sphere_for_hole_context(mesh, candidate, rings=rings)
    fit = fit_report["fit"]

    vertices = np.array(mesh.vertices, dtype=float, copy=True)
    faces = np.array(mesh.faces, dtype=np.int64, copy=True)

    boundary_points = vertices[list(boundary_ids)]
    boundary_centroid = np.mean(boundary_points, axis=0, keepdims=True)
    projected_center = project_points_to_sphere(boundary_centroid, fit)[0]

    auto_ring_report: dict[str, Any] | None = None
    if interior_rings is None:
        auto_ring_report = estimate_curvature_sphere_refined_interior_rings(
            mesh,
            candidate,
            fit=fit,
            projected_center=projected_center,
            rings=rings,
            min_interior_rings=1,
            max_interior_rings=4,
        )
        interior_rings = int(auto_ring_report["interior_rings"])
    else:
        interior_rings = int(interior_rings)

    if interior_rings < 1:
        raise ValueError("interior_rings must be >= 1")

    generated_vertex_blocks: list[np.ndarray] = []
    generated_ring_ids: list[tuple[int, ...]] = []
    next_vertex_id = int(len(vertices))
    boundary_count = int(len(boundary_ids))

    for ring_index in range(1, int(interior_rings) + 1):
        t = float(ring_index) / float(int(interior_rings) + 1)
        ring_points: list[np.ndarray] = []

        for index in range(boundary_count):
            a_point = boundary_points[index]
            b_point = boundary_points[(index + 1) % boundary_count]
            edge_midpoint = 0.5 * (a_point + b_point)

            raw_vertex_point = (1.0 - t) * a_point + t * projected_center
            raw_edge_point = (1.0 - t) * edge_midpoint + t * projected_center

            projected_pair = project_points_to_sphere(
                np.vstack(
                    [
                        raw_vertex_point.reshape(1, 3),
                        raw_edge_point.reshape(1, 3),
                    ]
                ),
                fit,
            )
            ring_points.append(projected_pair[0])
            ring_points.append(projected_pair[1])

        projected_ring = np.asarray(ring_points, dtype=float)
        ring_ids = tuple(range(next_vertex_id, next_vertex_id + len(projected_ring)))
        next_vertex_id += len(projected_ring)

        generated_vertex_blocks.append(projected_ring)
        generated_ring_ids.append(ring_ids)

    center_vertex_id = next_vertex_id
    generated_vertex_blocks.append(projected_center.reshape(1, 3))

    out_vertices = np.vstack([vertices, *generated_vertex_blocks])

    patch_faces: list[list[int]] = []
    first_ring = generated_ring_ids[0]

    # Boundary-to-first-ring strip:
    # For each boundary edge a-b:
    #   c = vertex support for a
    #   d = edge midpoint support for a-b
    #   e = vertex support for b
    #
    # Face [a,b,d] uses the original boundary edge exactly once.
    for index, a in enumerate(boundary_ids):
        b = int(boundary_ids[(index + 1) % boundary_count])
        c = int(first_ring[2 * index])
        d = int(first_ring[2 * index + 1])
        e = int(first_ring[(2 * ((index + 1) % boundary_count))])

        patch_faces.append([int(a), int(b), int(d)])
        patch_faces.append([int(a), int(d), int(c)])
        patch_faces.append([int(b), int(e), int(d)])

    previous_ring = first_ring

    # Ring-to-ring strips, if auto subdivision created more than one ring.
    for current_ring in generated_ring_ids[1:]:
        ring_count = len(previous_ring)
        if len(current_ring) != ring_count:
            raise ValueError("grid8 generated rings have mismatched sizes")

        for index, a in enumerate(previous_ring):
            b = previous_ring[(index + 1) % ring_count]
            c = current_ring[index]
            d = current_ring[(index + 1) % ring_count]

            face_a, face_b = _best_refined_patch_quad_split(
                out_vertices,
                a=int(a),
                b=int(b),
                c=int(c),
                d=int(d),
            )
            patch_faces.append(face_a)
            patch_faces.append(face_b)

        previous_ring = current_ring

    # Final projected center fan.
    ring_count = len(previous_ring)
    for index, vertex_id in enumerate(previous_ring):
        next_vertex_id_on_ring = previous_ring[(index + 1) % ring_count]
        patch_faces.append(
            [int(vertex_id), int(next_vertex_id_on_ring), int(center_vertex_id)]
        )

    if not patch_faces:
        raise ValueError("curvature_sphere_grid8 could not build patch faces")

    patch_faces_array = np.asarray(patch_faces, dtype=np.int64)
    out_faces = np.vstack([faces, patch_faces_array])
    preview = trimesh.Trimesh(vertices=out_vertices, faces=out_faces, process=False)

    if not np.allclose(np.asarray(mesh.vertices), source_vertices_before):
        raise RuntimeError("curvature_sphere_grid8 unexpectedly mutated source vertices")
    if not np.array_equal(np.asarray(mesh.faces), source_faces_before):
        raise RuntimeError("curvature_sphere_grid8 unexpectedly mutated source faces")

    new_vertex_ids = tuple(range(len(vertices), len(out_vertices)))
    new_face_ids = tuple(range(len(faces), len(out_faces)))

    return {
        "operation": "curvature_sphere_grid8_preview",
        "preview_mesh": preview,
        "rings": int(rings),
        "interior_rings": int(interior_rings),
        "auto_interior_rings": auto_ring_report is not None,
        "auto_ring_report": auto_ring_report,
        "fit": fit,
        "new_vertex_ids": new_vertex_ids,
        "new_face_ids": new_face_ids,
        "ring_vertex_ids": tuple(tuple(int(v) for v in ring) for ring in generated_ring_ids),
        "center_vertex_id": int(center_vertex_id),
        "boundary_vertex_ids": boundary_ids,
        "directional_support": "boundary_vertices_and_edge_midpoints",
        "projected_center": (
            float(projected_center[0]),
            float(projected_center[1]),
            float(projected_center[2]),
        ),
        "notes": (
            "Preview-only curvature sphere grid8 patch.",
            "Boundary vertices are fixed.",
            "First strip uses each original boundary edge exactly once.",
            "Generated vertex and edge-midpoint support rings were projected onto the fitted local sphere.",
            "Generated center vertex was projected onto the fitted local sphere.",
            "No source mesh mutation or project/history write occurred.",
        ),
    }


def estimate_curvature_sphere_uvgrid_edge_steps(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int = 2,
    target_edge_length: float | None = None,
    min_edge_steps: int = 2,
    max_edge_steps: int = 6,
) -> dict[str, Any]:
    """Estimate boundary/circumferential subdivision for uvgrid patches.

    This is the circumferential companion to the automatic radial ring estimate.
    It quantizes each boundary edge by local mesh spacing, so the patch gets
    support both from boundary vertex directions and from boundary-edge samples.
    """

    if min_edge_steps < 1:
        raise ValueError("min_edge_steps must be >= 1")
    if max_edge_steps < min_edge_steps:
        raise ValueError("max_edge_steps must be >= min_edge_steps")

    boundary_ids = candidate_boundary_vertex_ids(candidate)
    if len(boundary_ids) < 3:
        raise ValueError("uvgrid edge-step estimate requires at least three boundary vertices")

    vertices = np.asarray(mesh.vertices, dtype=float)
    if target_edge_length is None:
        target_edge_length = _context_median_edge_length(mesh, candidate, rings=rings)

    target = float(target_edge_length)
    if target <= 1.0e-12 or not np.isfinite(target):
        raise ValueError("target_edge_length must be positive and finite")

    edge_lengths: list[float] = []
    edge_steps: list[int] = []

    for index, a in enumerate(boundary_ids):
        b = boundary_ids[(index + 1) % len(boundary_ids)]
        length = float(np.linalg.norm(vertices[int(b)] - vertices[int(a)]))
        if not np.isfinite(length) or length <= 1.0e-12:
            steps = int(min_edge_steps)
        else:
            steps = int(np.ceil(length / target))
            steps = max(int(min_edge_steps), min(int(max_edge_steps), steps))

        edge_lengths.append(length)
        edge_steps.append(steps)

    return {
        "operation": "estimate_curvature_sphere_uvgrid_edge_steps",
        "rings": int(rings),
        "target_edge_length": float(target),
        "min_edge_steps": int(min_edge_steps),
        "max_edge_steps": int(max_edge_steps),
        "edge_lengths": tuple(float(v) for v in edge_lengths),
        "edge_steps": tuple(int(v) for v in edge_steps),
        "boundary_vertex_count": int(len(boundary_ids)),
        "resampled_boundary_count": int(sum(edge_steps)),
    }


def build_curvature_sphere_uvgrid_preview_mesh(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int = 2,
    interior_rings: int | None = None,
    target_edge_length: float | None = None,
) -> dict[str, Any]:
    """Build a preview-only local UV-style spherical patch.

    This method approximates a local parametric surface without true NURBS:
    - radial ring count is estimated from fitted-sphere angular span
    - boundary/circumferential samples are estimated from local mesh spacing
    - generated samples are sorted by boundary-loop order
    - all generated vertices are projected back onto the fitted sphere

    Source mesh vertices/faces are not mutated.
    """

    source_vertices_before = np.asarray(mesh.vertices).copy()
    source_faces_before = np.asarray(mesh.faces).copy()

    boundary_ids = candidate_boundary_vertex_ids(candidate)
    if len(boundary_ids) < 3:
        raise ValueError("curvature_sphere_uvgrid requires at least three boundary vertices")

    fit_report = fit_sphere_for_hole_context(mesh, candidate, rings=rings)
    fit = fit_report["fit"]

    vertices = np.array(mesh.vertices, dtype=float, copy=True)
    faces = np.array(mesh.faces, dtype=np.int64, copy=True)

    boundary_points = vertices[list(boundary_ids)]
    boundary_centroid = np.mean(boundary_points, axis=0, keepdims=True)
    projected_center = project_points_to_sphere(boundary_centroid, fit)[0]

    auto_ring_report: dict[str, Any] | None = None
    if interior_rings is None:
        auto_ring_report = estimate_curvature_sphere_refined_interior_rings(
            mesh,
            candidate,
            fit=fit,
            projected_center=projected_center,
            rings=rings,
            min_interior_rings=1,
            max_interior_rings=4,
        )
        interior_rings = int(auto_ring_report["interior_rings"])
    else:
        interior_rings = int(interior_rings)

    if interior_rings < 1:
        raise ValueError("interior_rings must be >= 1")

    edge_step_report = estimate_curvature_sphere_uvgrid_edge_steps(
        mesh,
        candidate,
        rings=rings,
        target_edge_length=target_edge_length,
        min_edge_steps=2,
        max_edge_steps=6,
    )
    edge_steps = tuple(int(v) for v in edge_step_report["edge_steps"])
    sample_count = int(edge_step_report["resampled_boundary_count"])

    edge_start_indices: list[int] = []
    cursor = 0
    for steps in edge_steps:
        edge_start_indices.append(cursor)
        cursor += int(steps)

    if cursor != sample_count:
        raise RuntimeError("uvgrid boundary sample indexing mismatch")

    generated_vertex_blocks: list[np.ndarray] = []
    generated_ring_ids: list[tuple[int, ...]] = []
    next_vertex_id = int(len(vertices))
    boundary_count = int(len(boundary_ids))

    for ring_index in range(1, int(interior_rings) + 1):
        t = float(ring_index) / float(int(interior_rings) + 1)
        raw_ring_points: list[np.ndarray] = []

        for edge_index, steps in enumerate(edge_steps):
            a_point = boundary_points[edge_index]
            b_point = boundary_points[(edge_index + 1) % boundary_count]

            for step_index in range(int(steps)):
                alpha = float(step_index) / float(steps)
                boundary_sample = (1.0 - alpha) * a_point + alpha * b_point
                raw = (1.0 - t) * boundary_sample + t * projected_center
                raw_ring_points.append(np.asarray(raw, dtype=float).reshape(3))

        projected_ring = project_points_to_sphere(
            np.asarray(raw_ring_points, dtype=float),
            fit,
        )

        ring_ids = tuple(range(next_vertex_id, next_vertex_id + len(projected_ring)))
        next_vertex_id += len(projected_ring)

        generated_vertex_blocks.append(projected_ring)
        generated_ring_ids.append(ring_ids)

    center_vertex_id = next_vertex_id
    generated_vertex_blocks.append(projected_center.reshape(1, 3))

    out_vertices = np.vstack([vertices, *generated_vertex_blocks])

    patch_faces: list[list[int]] = []
    first_ring = generated_ring_ids[0]

    # Boundary-to-first-ring strip. Each original boundary edge appears once,
    # then its inside is fanned to the quantized samples for that edge.
    for edge_index, a in enumerate(boundary_ids):
        b = int(boundary_ids[(edge_index + 1) % boundary_count])
        start = int(edge_start_indices[edge_index])
        next_start = int(edge_start_indices[(edge_index + 1) % boundary_count])
        steps = int(edge_steps[edge_index])

        sample_ids = [
            int(first_ring[(start + offset) % sample_count])
            for offset in range(steps)
        ]
        sample_ids.append(int(first_ring[next_start]))

        patch_faces.append([int(a), int(b), int(sample_ids[-1])])

        for local_index in range(steps - 1, -1, -1):
            patch_faces.append(
                [
                    int(a),
                    int(sample_ids[local_index + 1]),
                    int(sample_ids[local_index]),
                ]
            )

    previous_ring = first_ring

    for current_ring in generated_ring_ids[1:]:
        if len(current_ring) != len(previous_ring):
            raise ValueError("uvgrid generated rings have mismatched sample counts")

        for index, a in enumerate(previous_ring):
            b = previous_ring[(index + 1) % sample_count]
            c = current_ring[index]
            d = current_ring[(index + 1) % sample_count]

            face_a, face_b = _best_refined_patch_quad_split(
                out_vertices,
                a=int(a),
                b=int(b),
                c=int(c),
                d=int(d),
            )
            patch_faces.append(face_a)
            patch_faces.append(face_b)

        previous_ring = current_ring

    for index, vertex_id in enumerate(previous_ring):
        next_vertex_id_on_ring = previous_ring[(index + 1) % sample_count]
        patch_faces.append(
            [int(vertex_id), int(next_vertex_id_on_ring), int(center_vertex_id)]
        )

    if not patch_faces:
        raise ValueError("curvature_sphere_uvgrid could not build patch faces")

    patch_faces_array = np.asarray(patch_faces, dtype=np.int64)
    out_faces = np.vstack([faces, patch_faces_array])
    preview = trimesh.Trimesh(vertices=out_vertices, faces=out_faces, process=False)

    if not np.allclose(np.asarray(mesh.vertices), source_vertices_before):
        raise RuntimeError("curvature_sphere_uvgrid unexpectedly mutated source vertices")
    if not np.array_equal(np.asarray(mesh.faces), source_faces_before):
        raise RuntimeError("curvature_sphere_uvgrid unexpectedly mutated source faces")

    new_vertex_ids = tuple(range(len(vertices), len(out_vertices)))
    new_face_ids = tuple(range(len(faces), len(out_faces)))

    return {
        "operation": "curvature_sphere_uvgrid_preview",
        "preview_mesh": preview,
        "rings": int(rings),
        "interior_rings": int(interior_rings),
        "auto_interior_rings": auto_ring_report is not None,
        "auto_ring_report": auto_ring_report,
        "edge_step_report": edge_step_report,
        "fit": fit,
        "new_vertex_ids": new_vertex_ids,
        "new_face_ids": new_face_ids,
        "ring_vertex_ids": tuple(tuple(int(v) for v in ring) for ring in generated_ring_ids),
        "center_vertex_id": int(center_vertex_id),
        "boundary_vertex_ids": boundary_ids,
        "resampled_boundary_count": int(sample_count),
        "projected_center": (
            float(projected_center[0]),
            float(projected_center[1]),
            float(projected_center[2]),
        ),
        "notes": (
            "Preview-only curvature sphere uvgrid patch.",
            "Boundary vertices are fixed.",
            "Boundary loop was quantized by local mesh spacing.",
            "Generated rings and center were projected onto the fitted local sphere.",
            "No source mesh mutation or project/history write occurred.",
        ),
    }


def _normalize_vector_or_raise(value: np.ndarray, *, label: str) -> np.ndarray:
    vec = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-12 or not np.isfinite(norm):
        raise ValueError(f"{label} is degenerate")
    return vec / norm


def _sphere_tangent_frame_at_point(
    *,
    sphere_center: np.ndarray,
    point_on_sphere: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local tangent U/V frame and normal W at a point on the sphere."""

    center = np.asarray(sphere_center, dtype=float).reshape(3)
    point = np.asarray(point_on_sphere, dtype=float).reshape(3)

    w = _normalize_vector_or_raise(point - center, label="sphere tangent normal")

    # Pick the world axis least aligned with W for numerical stability.
    axes = (
        np.asarray([1.0, 0.0, 0.0], dtype=float),
        np.asarray([0.0, 1.0, 0.0], dtype=float),
        np.asarray([0.0, 0.0, 1.0], dtype=float),
    )
    ref = min(axes, key=lambda axis: abs(float(np.dot(axis, w))))

    u = _normalize_vector_or_raise(np.cross(ref, w), label="sphere tangent U")
    v = _normalize_vector_or_raise(np.cross(w, u), label="sphere tangent V")
    return u, v, w


def _points_to_tangent_uv(
    points: np.ndarray,
    *,
    origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    origin_arr = np.asarray(origin, dtype=float).reshape(1, 3)
    rel = pts - origin_arr
    return np.column_stack([rel @ u_axis, rel @ v_axis])


def _point_on_segment_2d(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    eps: float = 1.0e-10,
) -> bool:
    p = np.asarray(point, dtype=float).reshape(2)
    va = np.asarray(a, dtype=float).reshape(2)
    vb = np.asarray(b, dtype=float).reshape(2)

    ab = vb - va
    ap = p - va
    cross = float(ab[0] * ap[1] - ab[1] * ap[0])
    if abs(cross) > eps:
        return False

    dot = float(np.dot(ap, ab))
    if dot < -eps:
        return False

    ab_len_sq = float(np.dot(ab, ab))
    if dot - ab_len_sq > eps:
        return False

    return True


def _point_in_polygon_2d(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Return True when point is inside or on boundary of a simple 2D polygon."""

    p = np.asarray(point, dtype=float).reshape(2)
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError("polygon must have shape (N, 2) with N >= 3")

    inside = False
    n = int(len(poly))

    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]

        if _point_on_segment_2d(p, a, b):
            return True

        yi = float(a[1])
        yj = float(b[1])
        xi = float(a[0])
        xj = float(b[0])
        py = float(p[1])
        px = float(p[0])

        intersects = (yi > py) != (yj > py)
        if intersects:
            x_cross = (xj - xi) * (py - yi) / ((yj - yi) + 1.0e-300) + xi
            if px < x_cross:
                inside = not inside

    return inside


def _polygon_signed_area_2d(polygon: np.ndarray) -> float:
    poly = np.asarray(polygon, dtype=float)
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _dedupe_uv_points(points: np.ndarray, *, decimals: int = 10) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts.reshape(0, 2)

    seen: set[tuple[float, float]] = set()
    kept: list[np.ndarray] = []
    for point in pts:
        key = (round(float(point[0]), decimals), round(float(point[1]), decimals))
        if key in seen:
            continue
        seen.add(key)
        kept.append(np.asarray(point, dtype=float).reshape(2))
    return np.asarray(kept, dtype=float)




def _canonical_edge_key(a: int, b: int) -> tuple[int, int]:
    ia = int(a)
    ib = int(b)
    return (ia, ib) if ia <= ib else (ib, ia)


def _edge_length_from_vertices(vertices: np.ndarray, edge: tuple[int, int]) -> float:
    a, b = edge
    pts = np.asarray(vertices, dtype=float)
    return float(np.linalg.norm(pts[int(b)] - pts[int(a)]))


def _triangle_min_angle_degrees(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.shape != (3, 3) or not np.isfinite(pts).all():
        return 0.0

    a, b, c = pts
    lengths = np.asarray(
        [
            np.linalg.norm(b - c),
            np.linalg.norm(c - a),
            np.linalg.norm(a - b),
        ],
        dtype=float,
    )
    if np.any(lengths <= 1.0e-12):
        return 0.0

    angles: list[float] = []
    for i in range(3):
        x = lengths[(i + 1) % 3]
        y = lengths[(i + 2) % 3]
        z = lengths[i]
        denom = max(2.0 * x * y, 1.0e-12)
        cos_value = np.clip((x * x + y * y - z * z) / denom, -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cos_value))))

    return float(min(angles))




def _orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    pa = np.asarray(a, dtype=float).reshape(2)
    pb = np.asarray(b, dtype=float).reshape(2)
    pc = np.asarray(c, dtype=float).reshape(2)
    return float((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]))


def _triangle_area_abs2d(points: np.ndarray, tri: tuple[int, int, int]) -> float:
    pts = np.asarray(points, dtype=float)
    a, b, c = tri
    return abs(_orient2d(pts[int(a)], pts[int(b)], pts[int(c)]))


def _oriented_triangle_2d(points: np.ndarray, tri: tuple[int, int, int]) -> tuple[int, int, int]:
    pts = np.asarray(points, dtype=float)
    a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
    if _orient2d(pts[a], pts[b], pts[c]) < 0.0:
        return (a, c, b)
    return (a, b, c)


def _strict_segment_intersection_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    *,
    eps: float = 1.0e-10,
) -> bool:
    """Return True when segments AB and CD cross at an interior point."""

    pa = np.asarray(a, dtype=float).reshape(2)
    pb = np.asarray(b, dtype=float).reshape(2)
    pc = np.asarray(c, dtype=float).reshape(2)
    pd = np.asarray(d, dtype=float).reshape(2)

    o1 = _orient2d(pa, pb, pc)
    o2 = _orient2d(pa, pb, pd)
    o3 = _orient2d(pc, pd, pa)
    o4 = _orient2d(pc, pd, pb)

    # Strict crossing only. Shared endpoints / collinear touches are not the
    # flip target and should not destabilize boundary loops.
    return bool(
        ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps))
        and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps))
    )


def _local_triangle_edges(tri: tuple[int, int, int]) -> tuple[tuple[int, int], ...]:
    a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
    return (
        _canonical_edge_key(a, b),
        _canonical_edge_key(b, c),
        _canonical_edge_key(c, a),
    )


def _local_edge_to_triangles(
    triangles: list[tuple[int, int, int]],
) -> dict[tuple[int, int], list[int]]:
    edge_to_triangles: dict[tuple[int, int], list[int]] = {}
    for tri_index, tri in enumerate(triangles):
        for edge in _local_triangle_edges(tri):
            edge_to_triangles.setdefault(edge, []).append(int(tri_index))
    return edge_to_triangles


def _local_triangles_have_edge(
    triangles: list[tuple[int, int, int]],
    edge: tuple[int, int],
) -> bool:
    key = _canonical_edge_key(edge[0], edge[1])
    return any(key in _local_triangle_edges(tri) for tri in triangles)


def _triangle_centroid_inside_polygon_2d(
    points: np.ndarray,
    tri: tuple[int, int, int],
    polygon: np.ndarray,
) -> bool:
    pts = np.asarray(points, dtype=float)
    centroid = np.mean(pts[list(tri)], axis=0)
    return bool(_point_in_polygon_2d(centroid, polygon))


def _recover_constraint_edges_by_flips_2d(
    points: np.ndarray,
    triangles: list[tuple[int, int, int]],
    constraint_edges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    polygon: np.ndarray,
    max_iterations_per_edge: int = 512,
) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
    """Recover constrained UV edges by deterministic local edge flips.

    This is a lightweight internal CDT-style prototype:
    - it does not add vertices
    - it only flips existing interior diagonals
    - it targets missing boundary constraint edges before 3D patch faces are built

    If a constraint cannot be recovered safely, the original triangulation state
    is kept for that edge and the report marks it as failed.
    """

    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)

    current: list[tuple[int, int, int]] = [
        _oriented_triangle_2d(pts, tuple(int(v) for v in tri))
        for tri in triangles
        if len(set(int(v) for v in tri)) == 3
        and _triangle_area_abs2d(pts, tuple(int(v) for v in tri)) > 1.0e-12
    ]

    attempted_edges: list[tuple[int, int]] = []
    recovered_edges: list[tuple[int, int]] = []
    already_present_edges: list[tuple[int, int]] = []
    failed_edges: list[tuple[int, int]] = []
    flip_count_by_edge: dict[tuple[int, int], int] = {}

    for raw_edge in tuple(constraint_edges):
        target = (int(raw_edge[0]), int(raw_edge[1]))
        target_key = _canonical_edge_key(target[0], target[1])
        attempted_edges.append(target)

        if _local_triangles_have_edge(current, target_key):
            already_present_edges.append(target)
            flip_count_by_edge[target] = 0
            continue

        flips_for_edge = 0
        recovered = False

        for _iteration in range(int(max_iterations_per_edge)):
            if _local_triangles_have_edge(current, target_key):
                recovered = True
                break

            edge_to_triangles = _local_edge_to_triangles(current)

            crossing_edge: tuple[int, int] | None = None
            crossing_triangles: list[int] | None = None

            a, b = target
            for edge, tri_indices in edge_to_triangles.items():
                if len(tri_indices) != 2:
                    continue

                u, v = edge
                if len({int(a), int(b), int(u), int(v)}) < 4:
                    continue

                if _strict_segment_intersection_2d(
                    pts[int(a)],
                    pts[int(b)],
                    pts[int(u)],
                    pts[int(v)],
                ):
                    crossing_edge = (int(u), int(v))
                    crossing_triangles = [int(tri_indices[0]), int(tri_indices[1])]
                    break

            if crossing_edge is None or crossing_triangles is None:
                break

            u, v = crossing_edge
            t0_index, t1_index = crossing_triangles
            t0 = current[t0_index]
            t1 = current[t1_index]

            opp0 = [x for x in t0 if int(x) not in {int(u), int(v)}]
            opp1 = [x for x in t1 if int(x) not in {int(u), int(v)}]
            if len(opp0) != 1 or len(opp1) != 1:
                break

            c = int(opp0[0])
            d = int(opp1[0])
            if c == d:
                break

            candidate_a = _oriented_triangle_2d(pts, (c, d, u))
            candidate_b = _oriented_triangle_2d(pts, (d, c, v))

            if len(set(candidate_a)) != 3 or len(set(candidate_b)) != 3:
                break
            if _triangle_area_abs2d(pts, candidate_a) <= 1.0e-12:
                break
            if _triangle_area_abs2d(pts, candidate_b) <= 1.0e-12:
                break
            if not _triangle_centroid_inside_polygon_2d(pts, candidate_a, poly):
                break
            if not _triangle_centroid_inside_polygon_2d(pts, candidate_b, poly):
                break

            # Avoid replacing with duplicate triangles.
            replacement_indices = {int(t0_index), int(t1_index)}
            candidate_keys = {
                tuple(sorted(int(vv) for vv in candidate_a)),
                tuple(sorted(int(vv) for vv in candidate_b)),
            }
            duplicate = False
            for tri_index, tri in enumerate(current):
                if tri_index in replacement_indices:
                    continue
                if tuple(sorted(int(vv) for vv in tri)) in candidate_keys:
                    duplicate = True
                    break
            if duplicate:
                break

            first, second = sorted([int(t0_index), int(t1_index)])
            next_triangles = list(current)
            next_triangles[first] = candidate_a
            next_triangles[second] = candidate_b
            current = next_triangles
            flips_for_edge += 1

        if _local_triangles_have_edge(current, target_key):
            recovered_edges.append(target)
            recovered = True

        if not recovered:
            failed_edges.append(target)

        flip_count_by_edge[target] = int(flips_for_edge)

    return current, {
        "operation": "recover_constraint_edges_by_flips_2d",
        "attempted": bool(attempted_edges),
        "attempted_edge_count": int(len(attempted_edges)),
        "already_present_edge_count": int(len(already_present_edges)),
        "recovered_edge_count": int(len(recovered_edges)),
        "failed_edge_count": int(len(failed_edges)),
        "attempted_edges": tuple(tuple(int(v) for v in edge) for edge in attempted_edges),
        "already_present_edges": tuple(tuple(int(v) for v in edge) for edge in already_present_edges),
        "recovered_edges": tuple(tuple(int(v) for v in edge) for edge in recovered_edges),
        "failed_edges": tuple(tuple(int(v) for v in edge) for edge in failed_edges),
        "flip_count_by_edge": {
            f"{int(edge[0])}:{int(edge[1])}": int(count)
            for edge, count in flip_count_by_edge.items()
        },
        "notes": (
            "Internal lightweight constrained-edge recovery prototype.",
            "Only existing UV triangulation edges were flipped; no vertices were added.",
            "Failed edges should later fall back to local quad-collar recovery.",
        ),
    }


def build_patch_seam_constraint_report(
    *,
    vertices: np.ndarray,
    patch_faces: list[list[int]] | np.ndarray,
    boundary_vertex_ids: tuple[int, ...] | list[int],
) -> dict[str, Any]:
    """Report per-boundary-edge seam coverage for a generated patch.

    This is the diagnostic foundation for constrained UV Delaunay:
    every original boundary edge should be consumed by exactly one patch face.
    """

    boundary_ids = tuple(int(v) for v in boundary_vertex_ids)
    faces = np.asarray(patch_faces, dtype=np.int64).reshape(-1, 3)
    verts = np.asarray(vertices, dtype=float)

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        tri = [int(v) for v in face[:3]]
        for i, a in enumerate(tri):
            b = tri[(i + 1) % 3]
            edge_to_faces.setdefault(_canonical_edge_key(a, b), []).append(int(face_index))

    reports: list[dict[str, Any]] = []
    missing_edges: list[tuple[int, int]] = []
    overused_edges: list[tuple[int, int]] = []
    weak_edges: list[tuple[int, int]] = []

    for index, a in enumerate(boundary_ids):
        b = boundary_ids[(index + 1) % len(boundary_ids)]
        key = _canonical_edge_key(a, b)
        face_ids = tuple(edge_to_faces.get(key, ()))
        covered = len(face_ids) == 1

        local_min_angle = 0.0
        local_quality_penalty = float("inf")
        if face_ids:
            local_points = verts[faces[int(face_ids[0])]]
            local_min_angle = _triangle_min_angle_degrees(local_points)
            local_quality_penalty = float(_triangle_quality_penalty(local_points))

        if len(face_ids) == 0:
            missing_edges.append((int(a), int(b)))
        elif len(face_ids) > 1:
            overused_edges.append((int(a), int(b)))

        # Covered but slivery seam triangles are not missing, but they are weak
        # constraints and future edge recovery/refinement should be allowed.
        is_weak = bool(covered and local_min_angle < 8.0)
        if is_weak:
            weak_edges.append((int(a), int(b)))

        reports.append(
            {
                "edge": (int(a), int(b)),
                "covered": bool(covered),
                "patch_face_count": int(len(face_ids)),
                "patch_face_indices": tuple(int(v) for v in face_ids),
                "local_min_angle_degrees": float(local_min_angle),
                "local_quality_penalty": float(local_quality_penalty)
                if np.isfinite(local_quality_penalty)
                else float("inf"),
                "needs_recovery": bool((not covered) or is_weak),
            }
        )

    covered_count = sum(1 for item in reports if bool(item["covered"]))

    return {
        "expected_seam_edge_count": int(len(boundary_ids)),
        "covered_seam_edge_count": int(covered_count),
        "missing_seam_edge_count": int(len(missing_edges)),
        "overused_seam_edge_count": int(len(overused_edges)),
        "weak_seam_edge_count": int(len(weak_edges)),
        "seam_coverage_ratio": float(covered_count / max(1, len(boundary_ids))),
        "missing_edges": tuple(tuple(int(v) for v in edge) for edge in missing_edges),
        "overused_edges": tuple(tuple(int(v) for v in edge) for edge in overused_edges),
        "weak_edges": tuple(tuple(int(v) for v in edge) for edge in weak_edges),
        "edge_reports": tuple(reports),
    }




def seam_constraint_report_requires_recovery(report: dict[str, Any] | None) -> bool:
    """Return True when a seam report says constrained edge recovery is needed."""

    if not isinstance(report, dict):
        return False

    if int(report.get("missing_seam_edge_count", 0) or 0) > 0:
        return True
    if int(report.get("overused_seam_edge_count", 0) or 0) > 0:
        return True
    if int(report.get("weak_seam_edge_count", 0) or 0) > 0:
        return True

    for edge_report in tuple(report.get("edge_reports", ()) or ()):
        if isinstance(edge_report, dict) and bool(edge_report.get("needs_recovery", False)):
            return True

    return False


def _seam_report_edge_tuple(value: object) -> tuple[int, int] | None:
    try:
        raw = tuple(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if len(raw) != 2:
        return None
    return (int(raw[0]), int(raw[1]))


def build_seam_recovery_decision(report: dict[str, Any] | None) -> dict[str, Any]:
    """Build a JSON-safe local seam recovery decision from a seam report.

    This is a decision layer only. It does not modify geometry.
    Future constrained-edge recovery will use the reported contiguous runs.
    """

    if not isinstance(report, dict):
        return {
            "recovery_required": False,
            "strategy": "none",
            "problem_edge_count": 0,
            "problem_edges": (),
            "problem_edge_runs": (),
            "missing_seam_edge_count": 0,
            "overused_seam_edge_count": 0,
            "weak_seam_edge_count": 0,
            "notes": ("No seam report was available; no recovery decision was made.",),
        }

    problem_keys: set[tuple[int, int]] = set()
    problem_edges_ordered: list[tuple[int, int]] = []

    def _mark(edge_value: object) -> None:
        edge = _seam_report_edge_tuple(edge_value)
        if edge is None:
            return
        key = _canonical_edge_key(edge[0], edge[1])
        if key not in problem_keys:
            problem_keys.add(key)
            problem_edges_ordered.append(edge)

    for edge in tuple(report.get("missing_edges", ()) or ()):
        _mark(edge)
    for edge in tuple(report.get("overused_edges", ()) or ()):
        _mark(edge)
    for edge in tuple(report.get("weak_edges", ()) or ()):
        _mark(edge)

    edge_reports = tuple(report.get("edge_reports", ()) or ())
    ordered_boundary_edges: list[tuple[int, int]] = []
    for edge_report in edge_reports:
        if not isinstance(edge_report, dict):
            continue
        edge = _seam_report_edge_tuple(edge_report.get("edge"))
        if edge is None:
            continue
        ordered_boundary_edges.append(edge)
        if bool(edge_report.get("needs_recovery", False)):
            _mark(edge)

    problem_edge_set = {_canonical_edge_key(a, b) for a, b in problem_edges_ordered}

    runs: list[list[tuple[int, int]]] = []
    current_run: list[tuple[int, int]] = []
    for edge in ordered_boundary_edges:
        if _canonical_edge_key(edge[0], edge[1]) in problem_edge_set:
            current_run.append(edge)
        else:
            if current_run:
                runs.append(current_run)
                current_run = []
    if current_run:
        runs.append(current_run)

    # Boundary loops wrap, so merge first/last runs when both touch the loop ends.
    if len(runs) > 1 and ordered_boundary_edges:
        first_edge = ordered_boundary_edges[0]
        last_edge = ordered_boundary_edges[-1]
        if (
            _canonical_edge_key(first_edge[0], first_edge[1]) in problem_edge_set
            and _canonical_edge_key(last_edge[0], last_edge[1]) in problem_edge_set
        ):
            merged = runs[-1] + runs[0]
            runs = [merged] + runs[1:-1]

    # If edge_reports were unavailable, fall back to singleton runs.
    if not runs and problem_edges_ordered:
        runs = [[edge] for edge in problem_edges_ordered]

    recovery_required = bool(problem_edges_ordered)

    return {
        "recovery_required": bool(recovery_required),
        "strategy": "local_seam_edge_recovery" if recovery_required else "none",
        "problem_edge_count": int(len(problem_edges_ordered)),
        "problem_edges": tuple(tuple(int(v) for v in edge) for edge in problem_edges_ordered),
        "problem_edge_runs": tuple(
            tuple(tuple(int(v) for v in edge) for edge in run)
            for run in runs
        ),
        "missing_seam_edge_count": int(report.get("missing_seam_edge_count", 0) or 0),
        "overused_seam_edge_count": int(report.get("overused_seam_edge_count", 0) or 0),
        "weak_seam_edge_count": int(report.get("weak_seam_edge_count", 0) or 0),
        "seam_coverage_ratio": float(report.get("seam_coverage_ratio", 0.0) or 0.0),
        "notes": (
            "Decision layer only; no patch geometry was modified.",
            "Future recovery should repair only the reported local seam edge runs.",
        ),
    }




def _finite_float_or_none(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def build_surface_context_safety_decision(
    *,
    support_context: dict[str, Any] | None,
    topology_after: dict[str, Any] | None,
    relaxation: dict[str, Any] | None,
    normal_spread_limit_degrees: float = 90.0,
    max_patch_face_to_boundary_ratio: float = 12.0,
    max_patch_vertex_to_boundary_ratio: float = 8.0,
    max_movement_to_context_edge_ratio: float = 3.0,
    min_triangle_angle_floor_degrees: float = 1.0,
) -> dict[str, Any]:
    """Classify whether smooth MLS surface relaxation is safe.

    This is a decision layer only. It does not modify geometry.

    It catches the cube-face failure mode:
    - seam is valid
    - support normals are highly incompatible
    - patch density explodes
    - relaxation movement/quality regresses
    """

    support = dict(support_context or {})
    topo = dict(topology_after or {})
    relax = dict(relaxation or {})
    quality_before = dict(relax.get("quality_before") or {})
    quality_after = dict(relax.get("quality_after") or {})

    reasons: list[str] = []

    normal_spread = _finite_float_or_none(support.get("normal_spread_degrees"))
    high_normal_spread = bool(
        normal_spread is not None
        and normal_spread > float(normal_spread_limit_degrees)
    )
    if high_normal_spread:
        reasons.append("high_normal_spread")

    boundary_count_value = (
        quality_after.get("boundary_vertex_count")
        or topo.get("expected_seam_edge_count")
        or topo.get("seam_edge_count")
        or 0
    )
    boundary_count = max(1, int(boundary_count_value or 0))

    patch_face_count = int(
        quality_after.get("patch_face_count")
        or topo.get("patch_face_count")
        or 0
    )
    patch_vertex_count = int(
        quality_after.get("patch_vertex_count")
        or topo.get("patch_vertex_count")
        or 0
    )

    face_to_boundary_ratio = float(patch_face_count / boundary_count)
    vertex_to_boundary_ratio = float(patch_vertex_count / boundary_count)

    density_explosion = False
    if face_to_boundary_ratio > float(max_patch_face_to_boundary_ratio):
        density_explosion = True
        reasons.append("patch_face_density_explosion")
    if vertex_to_boundary_ratio > float(max_patch_vertex_to_boundary_ratio):
        density_explosion = True
        reasons.append("patch_vertex_density_explosion")

    max_displacement = _finite_float_or_none(relax.get("max_displacement"))
    mean_displacement = _finite_float_or_none(relax.get("mean_displacement"))
    context_edge = (
        _finite_float_or_none(quality_after.get("context_edge_length_median"))
        or _finite_float_or_none(quality_before.get("context_edge_length_median"))
    )

    movement_explosion = False
    movement_to_context_edge_ratio = None
    if (
        max_displacement is not None
        and context_edge is not None
        and context_edge > 1.0e-12
    ):
        movement_to_context_edge_ratio = float(max_displacement / context_edge)
        if movement_to_context_edge_ratio > float(max_movement_to_context_edge_ratio):
            movement_explosion = True
            reasons.append("excessive_relaxation_movement")

    min_angle_before = _finite_float_or_none(
        quality_before.get("min_triangle_angle_degrees")
    )
    min_angle_after = _finite_float_or_none(
        quality_after.get("min_triangle_angle_degrees")
    )
    median_aspect_before = _finite_float_or_none(
        quality_before.get("median_triangle_aspect_ratio")
    )
    median_aspect_after = _finite_float_or_none(
        quality_after.get("median_triangle_aspect_ratio")
    )

    quality_regression = False
    if min_angle_after is not None and min_angle_after < float(min_triangle_angle_floor_degrees):
        quality_regression = True
        reasons.append("min_angle_below_floor")
    elif (
        min_angle_before is not None
        and min_angle_after is not None
        and min_angle_after < 0.5 * min_angle_before
    ):
        quality_regression = True
        reasons.append("min_angle_regression")

    if (
        median_aspect_before is not None
        and median_aspect_after is not None
        and median_aspect_after > 2.0 * max(median_aspect_before, 1.0e-12)
    ):
        quality_regression = True
        reasons.append("median_aspect_regression")

    boundary_normal_before = _finite_float_or_none(
        quality_before.get("boundary_normal_mean_deviation_degrees")
    )
    boundary_normal_after = _finite_float_or_none(
        quality_after.get("boundary_normal_mean_deviation_degrees")
    )

    boundary_normal_regression = False
    if (
        boundary_normal_before is not None
        and boundary_normal_after is not None
        and boundary_normal_after > boundary_normal_before + 10.0
    ):
        boundary_normal_regression = True
        reasons.append("boundary_normal_regression")

    smooth_surface_context_valid = not high_normal_spread
    density_sane = not density_explosion
    relaxation_sane = not (movement_explosion or quality_regression or boundary_normal_regression)

    # Density alone is not a hard rejection. Smooth curved holes can legitimately
    # produce many patch faces, especially in synthetic/high-subdivision tests.
    # Reject only when density combines with invalid surface context, excessive
    # movement, or quality collapse. This keeps the cube-face failure blocked
    # without rejecting otherwise stable sphere patches.
    hard_reject = bool(
        movement_explosion
        or quality_regression
        or (high_normal_spread and density_explosion)
        or (high_normal_spread and boundary_normal_regression)
    )

    if hard_reject:
        recommended_action = "reject_relaxed_preview"
    elif not smooth_surface_context_valid:
        recommended_action = "disable_mls_surface_guidance"
    elif density_explosion:
        recommended_action = "allow_smooth_surface_relaxation_with_density_warning"
    else:
        recommended_action = "allow_smooth_surface_relaxation"

    return {
        "smooth_surface_context_valid": bool(smooth_surface_context_valid),
        "density_sane": bool(density_sane),
        "relaxation_sane": bool(relaxation_sane),
        "recommended_action": recommended_action,
        "reasons": tuple(reasons),
        "support_normal_spread_degrees": normal_spread,
        "normal_spread_limit_degrees": float(normal_spread_limit_degrees),
        "boundary_vertex_count": int(boundary_count),
        "patch_face_count": int(patch_face_count),
        "patch_vertex_count": int(patch_vertex_count),
        "patch_face_to_boundary_ratio": float(face_to_boundary_ratio),
        "patch_vertex_to_boundary_ratio": float(vertex_to_boundary_ratio),
        "max_displacement": max_displacement,
        "mean_displacement": mean_displacement,
        "context_edge_length_median": context_edge,
        "movement_to_context_edge_ratio": movement_to_context_edge_ratio,
        "min_triangle_angle_before_degrees": min_angle_before,
        "min_triangle_angle_after_degrees": min_angle_after,
        "median_aspect_before": median_aspect_before,
        "median_aspect_after": median_aspect_after,
        "boundary_normal_mean_before_degrees": boundary_normal_before,
        "boundary_normal_mean_after_degrees": boundary_normal_after,
        "notes": (
            "Decision layer only; no patch geometry was modified.",
            "High normal spread means one smooth MLS surface should not be trusted.",
            "Density or relaxation explosions should be rejected before commit/integration.",
        ),
    }




def enforce_surface_context_safety_decision(
    decision: dict[str, Any] | None,
    *,
    method: str = "surface_uvdelaunay_relaxed",
) -> None:
    """Raise when the surface-context safety gate rejects a preview.

    This is the first active safety gate:
    - it does not pick another tool
    - it does not modify geometry
    - it prevents known-bad relaxed previews from being returned/committed
    """

    if not isinstance(decision, dict):
        return

    action = str(decision.get("recommended_action") or "")
    if action != "reject_relaxed_preview":
        return

    reasons = tuple(str(v) for v in decision.get("reasons", ()) or ())
    reason_text = ", ".join(reasons) if reasons else "unknown safety reason"

    raise ValueError(
        f"Unsafe {method} preview rejected by surface-context safety gate: "
        f"{reason_text}. "
        "The patch topology may be seam-valid, but the surface/density/relaxation "
        "diagnostics indicate the relaxed preview is not safe to return."
    )



def build_local_density_budget_decision(
    *,
    support_context: dict[str, Any] | None,
    topology_after: dict[str, Any] | None,
    relaxation: dict[str, Any] | None = None,
    feature_normal_spread_degrees: float = 90.0,
    tiny_boundary_vertex_count: int = 8,
    smooth_max_face_to_boundary_ratio: float = 16.0,
    smooth_max_vertex_to_boundary_ratio: float = 10.0,
    feature_max_face_to_boundary_ratio: float = 4.0,
    feature_max_vertex_to_boundary_ratio: float = 3.0,
) -> dict[str, Any]:
    """Return a local density-budget decision for a generated patch.

    This is a decision layer only. It does not modify geometry.

    Purpose:
    - prevent tiny/simple holes from becoming huge dense patches
    - detect feature-like contexts where density must stay minimal
    - prepare for 2J-E6-I4 local density refinement
    """

    support = dict(support_context or {})
    topo = dict(topology_after or {})
    relax = dict(relaxation or {})
    quality_after = dict(relax.get("quality_after") or {})

    reasons: list[str] = []

    normal_spread = _finite_float_or_none(support.get("normal_spread_degrees"))
    feature_like_context = bool(
        normal_spread is not None
        and normal_spread > float(feature_normal_spread_degrees)
    )
    if feature_like_context:
        reasons.append("feature_like_normal_spread")

    boundary_count_value = (
        quality_after.get("boundary_vertex_count")
        or topo.get("expected_seam_edge_count")
        or topo.get("seam_edge_count")
        or 0
    )
    boundary_vertex_count = max(1, int(boundary_count_value or 0))

    patch_face_count = int(
        quality_after.get("patch_face_count")
        or topo.get("patch_face_count")
        or 0
    )
    patch_vertex_count = int(
        quality_after.get("patch_vertex_count")
        or topo.get("patch_vertex_count")
        or 0
    )

    face_to_boundary_ratio = float(patch_face_count / boundary_vertex_count)
    vertex_to_boundary_ratio = float(patch_vertex_count / boundary_vertex_count)

    tiny_boundary = bool(boundary_vertex_count <= int(tiny_boundary_vertex_count))
    if tiny_boundary:
        reasons.append("tiny_boundary_loop")

    max_face_ratio = (
        float(feature_max_face_to_boundary_ratio)
        if feature_like_context or tiny_boundary
        else float(smooth_max_face_to_boundary_ratio)
    )
    max_vertex_ratio = (
        float(feature_max_vertex_to_boundary_ratio)
        if feature_like_context or tiny_boundary
        else float(smooth_max_vertex_to_boundary_ratio)
    )

    face_budget_exceeded = bool(face_to_boundary_ratio > max_face_ratio)
    vertex_budget_exceeded = bool(vertex_to_boundary_ratio > max_vertex_ratio)

    if face_budget_exceeded:
        reasons.append("face_density_budget_exceeded")
    if vertex_budget_exceeded:
        reasons.append("vertex_density_budget_exceeded")

    density_budget_exceeded = bool(face_budget_exceeded or vertex_budget_exceeded)

    if tiny_boundary and feature_like_context:
        recommended_density_mode = "minimal_topology"
    elif density_budget_exceeded:
        recommended_density_mode = "capped_local_refinement"
    else:
        recommended_density_mode = "allow_current_density"

    return {
        "recommended_density_mode": recommended_density_mode,
        "density_budget_exceeded": bool(density_budget_exceeded),
        "feature_like_context": bool(feature_like_context),
        "tiny_boundary_loop": bool(tiny_boundary),
        "boundary_vertex_count": int(boundary_vertex_count),
        "patch_face_count": int(patch_face_count),
        "patch_vertex_count": int(patch_vertex_count),
        "patch_face_to_boundary_ratio": float(face_to_boundary_ratio),
        "patch_vertex_to_boundary_ratio": float(vertex_to_boundary_ratio),
        "max_face_to_boundary_ratio": float(max_face_ratio),
        "max_vertex_to_boundary_ratio": float(max_vertex_ratio),
        "support_normal_spread_degrees": normal_spread,
        "feature_normal_spread_degrees": float(feature_normal_spread_degrees),
        "reasons": tuple(reasons),
        "notes": (
            "Decision layer only; no patch geometry was modified.",
            "Future local density refinement should use this budget before adding interior samples.",
            "Tiny feature-like holes should stay minimal instead of using dense smooth-surface sampling.",
        ),
    }




def build_local_density_refinement_budget(
    density_decision: dict[str, Any] | None,
    *,
    min_face_budget: int = 4,
    min_vertex_budget: int = 4,
) -> dict[str, Any]:
    """Convert a density decision into concrete local refinement limits.

    This is still a planning/budget layer only. It does not modify geometry.

    Future I4-B seed generation should consult this before adding interior
    UV samples, dense rings, or extra refinement points.
    """

    decision = dict(density_decision or {})

    mode = str(decision.get("recommended_density_mode") or "allow_current_density")
    boundary_count = max(1, int(decision.get("boundary_vertex_count") or 0))

    # These are the decision-layer ratios selected by build_local_density_budget_decision().
    max_face_ratio = float(decision.get("max_face_to_boundary_ratio") or 16.0)
    max_vertex_ratio = float(decision.get("max_vertex_to_boundary_ratio") or 10.0)

    max_patch_faces = max(int(min_face_budget), int(np.ceil(boundary_count * max_face_ratio)))
    max_patch_vertices = max(int(min_vertex_budget), int(np.ceil(boundary_count * max_vertex_ratio)))

    # For generated vertices, subtract the fixed boundary loop. This is what
    # future seed builders should use to cap interior/collar/dense samples.
    max_generated_vertices = max(0, int(max_patch_vertices - boundary_count))

    if mode == "minimal_topology":
        # Tiny feature-like holes should stay close to direct boundary-only
        # repair scale. Do not add interior UV samples here; this avoids
        # sphere-projected seed vertices bowing out of planar/feature holes.
        refinement_enabled = False
        target_interior_sample_factor = 0.0
        max_generated_vertices = 0
        max_patch_faces = min(max_patch_faces, max(int(min_face_budget), boundary_count * 2))
        max_patch_vertices = min(max_patch_vertices, boundary_count)
    elif mode == "capped_local_refinement":
        refinement_enabled = True
        target_interior_sample_factor = 0.5
    else:
        refinement_enabled = True
        target_interior_sample_factor = 1.0

    return {
        "recommended_density_mode": mode,
        "refinement_enabled": bool(refinement_enabled),
        "boundary_vertex_count": int(boundary_count),
        "max_patch_faces": int(max_patch_faces),
        "max_patch_vertices": int(max_patch_vertices),
        "max_generated_vertices": int(max_generated_vertices),
        "target_interior_sample_factor": float(target_interior_sample_factor),
        "source_density_budget_exceeded": bool(decision.get("density_budget_exceeded", False)),
        "feature_like_context": bool(decision.get("feature_like_context", False)),
        "tiny_boundary_loop": bool(decision.get("tiny_boundary_loop", False)),
        "notes": (
            "Budget layer only; no geometry was modified.",
            "Future I4-B local refinement should cap generated samples using this budget.",
            "Minimal topology mode limits density for tiny feature-like holes.",
        ),
    }




def build_surface_relaxation_confidence_profile(
    *,
    density_decision: dict[str, Any] | None,
    density_refinement_budget: dict[str, Any] | None,
    requested_iterations: int,
    requested_relaxation_strength: float,
    requested_surface_weight: float,
) -> dict[str, Any]:
    """Return effective relaxation controls from local confidence.

    This is 2J-E6-I5-A: feature-like / tiny minimal-topology regions should
    not trust one smooth MLS surface, and should not run full membrane
    relaxation. Smooth contexts keep the requested settings.
    """

    decision = dict(density_decision or {})
    budget = dict(density_refinement_budget or {})

    mode = str(
        budget.get("recommended_density_mode")
        or decision.get("recommended_density_mode")
        or "allow_current_density"
    )
    feature_like = bool(
        budget.get("feature_like_context", False)
        or decision.get("feature_like_context", False)
    )
    tiny = bool(
        budget.get("tiny_boundary_loop", False)
        or decision.get("tiny_boundary_loop", False)
    )

    effective_iterations = int(requested_iterations)
    effective_strength = float(requested_relaxation_strength)
    effective_surface_weight = float(requested_surface_weight)
    surface_guidance_enabled = True
    relaxation_enabled = effective_iterations > 0 and effective_strength > 0.0

    reasons: list[str] = []

    if feature_like:
        reasons.append("feature_like_context")
    if tiny:
        reasons.append("tiny_boundary_loop")
    if mode == "minimal_topology":
        reasons.append("minimal_topology_density_mode")

    if mode == "minimal_topology" and feature_like:
        # In high-normal-spread tiny feature contexts, a single smooth MLS
        # target is not trustworthy. Keep the sparse seed stable for now.
        surface_guidance_enabled = False
        relaxation_enabled = False
        effective_iterations = 0
        effective_strength = 0.0
        effective_surface_weight = 0.0
        recommended_relaxation_mode = "preserve_minimal_seed"
    elif feature_like:
        # Larger feature-like regions may still allow gentle fairing later,
        # but should not use smooth surface pull yet.
        surface_guidance_enabled = False
        effective_surface_weight = 0.0
        recommended_relaxation_mode = "fair_without_surface_pull"
    else:
        recommended_relaxation_mode = "smooth_surface_guided_relaxation"

    return {
        "recommended_relaxation_mode": recommended_relaxation_mode,
        "surface_guidance_enabled": bool(surface_guidance_enabled),
        "relaxation_enabled": bool(relaxation_enabled),
        "requested_iterations": int(requested_iterations),
        "effective_iterations": int(effective_iterations),
        "requested_relaxation_strength": float(requested_relaxation_strength),
        "effective_relaxation_strength": float(effective_strength),
        "requested_surface_weight": float(requested_surface_weight),
        "effective_surface_weight": float(effective_surface_weight),
        "recommended_density_mode": mode,
        "feature_like_context": bool(feature_like),
        "tiny_boundary_loop": bool(tiny),
        "reasons": tuple(reasons),
        "notes": (
            "Confidence profile for relaxation; geometry behavior is adjusted before relaxation.",
            "Feature-like minimal topology disables smooth MLS pull and full relaxation.",
            "Smooth contexts keep the requested surface-guided relaxation settings.",
        ),
    }


def _generate_uvdelaunay_interior_points(
    boundary_uv: np.ndarray,
    *,
    spacing: float,
    max_points: int = 500,
) -> np.ndarray:
    """Generate interior UV points on a staggered grid clipped to boundary polygon."""

    poly = np.asarray(boundary_uv, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError("boundary_uv must have shape (N, 2) with N >= 3")

    step = float(spacing)
    if step <= 1.0e-12 or not np.isfinite(step):
        raise ValueError("uvdelaunay spacing must be positive and finite")

    min_xy = np.min(poly, axis=0)
    max_xy = np.max(poly, axis=0)

    # Light inset so generated points do not duplicate boundary vertices.
    margin = step * 0.45

    xs = np.arange(float(min_xy[0]) + margin, float(max_xy[0]) - margin + step * 0.25, step)
    ys = np.arange(float(min_xy[1]) + margin, float(max_xy[1]) - margin + step * 0.25, step)

    points: list[np.ndarray] = []
    for row, y in enumerate(ys):
        x_offset = 0.5 * step if row % 2 else 0.0
        for x in xs + x_offset:
            point = np.asarray([float(x), float(y)], dtype=float)
            if _point_in_polygon_2d(point, poly):
                points.append(point)

    # Always try to include the tangent origin, which corresponds to projected cap center.
    center = np.asarray([0.0, 0.0], dtype=float)
    if _point_in_polygon_2d(center, poly):
        points.append(center)

    pts = _dedupe_uv_points(np.asarray(points, dtype=float).reshape(-1, 2))
    if len(pts) > max_points:
        # Deterministic thinning for very large holes.
        indices = np.linspace(0, len(pts) - 1, num=max_points, dtype=np.int64)
        pts = pts[indices]

    return pts




def _plane_tangent_frame_for_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a stable local tangent frame for planar/degenerate sphere-fit fallback."""

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 3:
        raise ValueError("plane tangent frame requires at least three points")

    centered = pts - np.mean(pts, axis=0, keepdims=True)

    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("plane tangent frame SVD failed") from exc

    if vh.shape[0] < 2:
        raise ValueError("plane tangent frame is under-constrained")

    u_axis = np.asarray(vh[0], dtype=float)
    u_norm = float(np.linalg.norm(u_axis))
    if u_norm <= 1.0e-12:
        raise ValueError("plane tangent frame has degenerate primary axis")
    u_axis = u_axis / u_norm

    if vh.shape[0] >= 3:
        normal = np.asarray(vh[-1], dtype=float)
    else:
        normal = np.cross(u_axis, np.asarray(vh[1], dtype=float))

    n_norm = float(np.linalg.norm(normal))
    if n_norm <= 1.0e-12:
        # Deterministic fallback normal not parallel to u_axis.
        probe = np.asarray([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(probe, u_axis))) > 0.95:
            probe = np.asarray([0.0, 1.0, 0.0], dtype=float)
        normal = np.cross(u_axis, probe)
        n_norm = float(np.linalg.norm(normal))

    if n_norm <= 1.0e-12:
        raise ValueError("plane tangent frame has degenerate normal")

    w_axis = normal / n_norm
    v_axis = np.cross(w_axis, u_axis)
    v_norm = float(np.linalg.norm(v_axis))
    if v_norm <= 1.0e-12:
        raise ValueError("plane tangent frame has degenerate secondary axis")
    v_axis = v_axis / v_norm

    return u_axis, v_axis, w_axis


def build_curvature_sphere_uvdelaunay_preview_mesh(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int = 2,
    target_edge_length: float | None = None,
    max_interior_points: int | None = None,
) -> dict[str, Any]:
    """Build a preview-only curvature-aware UV Delaunay patch.

    This method avoids the center-fan and ring-strip singularities used by the
    earlier curvature sphere prototypes:
    - boundary vertices are fixed and reused
    - generated interior vertices are placed in local tangent UV coordinates
    - 2D Delaunay triangulates the local patch
    - generated vertices are projected back onto the fitted sphere

    Source mesh vertices/faces are not mutated.
    """

    try:
        from scipy.spatial import Delaunay  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("curvature_sphere_uvdelaunay requires scipy.spatial.Delaunay") from exc

    source_vertices_before = np.asarray(mesh.vertices).copy()
    source_faces_before = np.asarray(mesh.faces).copy()

    boundary_ids = candidate_boundary_vertex_ids(candidate)
    if len(boundary_ids) < 3:
        raise ValueError("curvature_sphere_uvdelaunay requires at least three boundary vertices")

    vertices = np.array(mesh.vertices, dtype=float, copy=True)
    faces = np.array(mesh.faces, dtype=np.int64, copy=True)

    boundary_points = vertices[list(boundary_ids)]
    boundary_centroid = np.mean(boundary_points, axis=0)

    fit_report: dict[str, Any] | None = None
    fit = None
    seed_surface_kind = "sphere"
    seed_surface_error: str | None = None

    try:
        fit_report = fit_sphere_for_hole_context(mesh, candidate, rings=rings)
        fit = fit_report["fit"]
        projected_center = project_points_to_sphere(
            boundary_centroid.reshape(1, 3),
            fit,
        )[0]

        sphere_center = np.asarray(fit.center, dtype=float)
        u_axis, v_axis, w_axis = _sphere_tangent_frame_at_point(
            sphere_center=sphere_center,
            point_on_sphere=projected_center,
        )
    except ValueError as exc:
        # Planar/degenerate sphere context is not a hard failure for UV
        # Delaunay. It means the correct local seed surface is a plane.
        seed_surface_kind = "plane_fallback"
        seed_surface_error = str(exc)
        projected_center = np.asarray(boundary_centroid, dtype=float).reshape(3)
        u_axis, v_axis, w_axis = _plane_tangent_frame_for_points(boundary_points)

    boundary_uv = _points_to_tangent_uv(
        boundary_points,
        origin=projected_center,
        u_axis=u_axis,
        v_axis=v_axis,
    )

    if abs(_polygon_signed_area_2d(boundary_uv)) <= 1.0e-12:
        raise ValueError("curvature_sphere_uvdelaunay boundary projection is degenerate")

    if target_edge_length is None:
        target_edge_length = _context_median_edge_length(mesh, candidate, rings=rings)

    # Use a denser spacing than the surrounding edge length. The first
    # uvdelaunay prototype used 0.85 and produced too few support vertices for
    # QuadWild/Instant remesh. Keep this deterministic but allow a few retries
    # if the clipped polygon still produces too few interior samples.
    min_interior_points = max(6, int(np.ceil(len(boundary_ids) * 0.5)))

    if max_interior_points is None:
        uvdelaunay_max_points = 800
    else:
        uvdelaunay_max_points = int(max_interior_points)
        if uvdelaunay_max_points < 0:
            raise ValueError("max_interior_points must be >= 0 when provided")

    min_interior_points = min(int(min_interior_points), int(uvdelaunay_max_points))
    spacing = float(target_edge_length) * 0.55

    interior_uv = np.empty((0, 2), dtype=float)
    for attempt in range(6):
        attempt_spacing = spacing * (0.75 ** attempt)
        interior_uv = _generate_uvdelaunay_interior_points(
            boundary_uv,
            spacing=attempt_spacing,
            max_points=uvdelaunay_max_points,
        )
        if len(interior_uv) >= min_interior_points:
            spacing = float(attempt_spacing)
            break

    if len(interior_uv) == 0 and int(uvdelaunay_max_points) > 0:
        raise ValueError("curvature_sphere_uvdelaunay generated no interior points")

    if len(interior_uv) == 0:
        all_uv = np.asarray(boundary_uv, dtype=float)
    else:
        all_uv = np.vstack([boundary_uv, interior_uv])
    triangulation = Delaunay(all_uv)

    boundary_count = int(len(boundary_ids))
    generated_vertex_ids = tuple(
        range(len(vertices), len(vertices) + len(interior_uv))
    )

    if len(interior_uv) == 0:
        projected_interior_points = np.empty((0, 3), dtype=float)
    else:
        interior_tangent_points = (
            projected_center.reshape(1, 3)
            + interior_uv[:, 0:1] * u_axis.reshape(1, 3)
            + interior_uv[:, 1:2] * v_axis.reshape(1, 3)
        )
        if seed_surface_kind == "sphere" and fit is not None:
            projected_interior_points = project_points_to_sphere(interior_tangent_points, fit)
        else:
            projected_interior_points = np.asarray(interior_tangent_points, dtype=float)

    out_vertices = np.vstack([vertices, projected_interior_points])

    raw_uv_triangles: list[tuple[int, int, int]] = []
    for simplex in np.asarray(triangulation.simplices, dtype=np.int64):
        local_tri = tuple(int(v) for v in simplex[:3])
        if len(set(local_tri)) != 3:
            continue

        uv_tri = all_uv[list(local_tri)]
        centroid = np.mean(uv_tri, axis=0)
        if not _point_in_polygon_2d(centroid, boundary_uv):
            continue
        if _triangle_area_abs2d(all_uv, local_tri) <= 1.0e-12:
            continue

        raw_uv_triangles.append(_oriented_triangle_2d(all_uv, local_tri))

    boundary_constraint_edges = tuple(
        (int(index), int((index + 1) % boundary_count))
        for index in range(boundary_count)
    )

    recovered_uv_triangles, constrained_edge_recovery_report = _recover_constraint_edges_by_flips_2d(
        all_uv,
        raw_uv_triangles,
        boundary_constraint_edges,
        polygon=boundary_uv,
    )

    patch_faces: list[list[int]] = []
    for local_tri in recovered_uv_triangles:
        mapped: list[int] = []
        for local_id in local_tri:
            if int(local_id) < boundary_count:
                mapped.append(int(boundary_ids[int(local_id)]))
            else:
                mapped.append(int(generated_vertex_ids[int(local_id) - boundary_count]))

        if len(set(mapped)) != 3:
            continue

        pts = out_vertices[mapped]
        area2 = float(np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0])))
        if area2 <= 1.0e-12:
            continue

        patch_faces.append(mapped)

    if not patch_faces:
        raise ValueError("curvature_sphere_uvdelaunay could not build patch faces")

    patch_faces_array = np.asarray(patch_faces, dtype=np.int64)
    out_faces = np.vstack([faces, patch_faces_array])
    preview = trimesh.Trimesh(vertices=out_vertices, faces=out_faces, process=False)

    if not np.allclose(np.asarray(mesh.vertices), source_vertices_before):
        raise RuntimeError("curvature_sphere_uvdelaunay unexpectedly mutated source vertices")
    if not np.array_equal(np.asarray(mesh.faces), source_faces_before):
        raise RuntimeError("curvature_sphere_uvdelaunay unexpectedly mutated source faces")

    new_vertex_ids = tuple(range(len(vertices), len(out_vertices)))
    new_face_ids = tuple(range(len(faces), len(out_faces)))

    seam_constraint_report = build_patch_seam_constraint_report(
        vertices=out_vertices,
        patch_faces=patch_faces,
        boundary_vertex_ids=boundary_ids,
    )
    seam_recovery_decision = build_seam_recovery_decision(seam_constraint_report)


    return {
        "operation": "curvature_sphere_uvdelaunay_preview",
        "preview_mesh": preview,
        "rings": int(rings),
        "fit": fit,
        "fit_report": fit_report,
        "seed_surface_kind": seed_surface_kind,
        "seed_surface_error": seed_surface_error,
        "target_edge_length": float(target_edge_length),
        "spacing": float(spacing),
        "min_interior_points": int(min_interior_points),
        "max_interior_points": int(uvdelaunay_max_points),
        "boundary_vertex_ids": boundary_ids,
        "boundary_uv": tuple((float(x), float(y)) for x, y in boundary_uv),
        "interior_uv_count": int(len(interior_uv)),
        "delaunay_triangle_count": int(len(patch_faces)),
        "constrained_edge_recovery_report": constrained_edge_recovery_report,
        "seam_constraint_report": seam_constraint_report,
        "seam_recovery_decision": seam_recovery_decision,
        "new_vertex_ids": new_vertex_ids,
        "new_face_ids": new_face_ids,
        "center_vertex_id": None,
        "projected_center": (
            float(projected_center[0]),
            float(projected_center[1]),
            float(projected_center[2]),
        ),
        "frame": {
            "u": (float(u_axis[0]), float(u_axis[1]), float(u_axis[2])),
            "v": (float(v_axis[0]), float(v_axis[1]), float(v_axis[2])),
            "w": (float(w_axis[0]), float(w_axis[1]), float(w_axis[2])),
        },
        "notes": (
            "Preview-only curvature sphere uvdelaunay patch.",
            "Boundary vertices are fixed.",
            "Interior points were generated in local tangent UV coordinates.",
            "2D Delaunay triangulation was clipped by the boundary polygon.",
            "Generated interior vertices were projected onto the selected local seed surface.",
            "No source mesh mutation or project/history write occurred.",
        ),
    }


def build_curvature_sphere_uvdelaunay_relaxed_preview_mesh(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int = 2,
    relaxation_iterations: int = 8,
    relaxation_strength: float = 0.20,
    surface_weight: float = 0.15,
) -> dict[str, Any]:
    """Build a uvdelaunay seed patch and relax only generated vertices."""

    from far_mesh.core.hole_patch_quality import (
        analyze_patch_quality,
        compute_boundary_target_normals,
        mesh_edge_length_median,
    )
    from far_mesh.core.hole_patch_relaxation import relax_patch_vertices
    from far_mesh.core.hole_surface_context import validate_patch_topology

    source_vertices_before = np.asarray(mesh.vertices).copy()
    source_faces_before = np.asarray(mesh.faces).copy()

    seed = build_curvature_sphere_uvdelaunay_preview_mesh(
        mesh,
        candidate,
        rings=rings,
    )
    seed_preview = seed["preview_mesh"]
    if not isinstance(seed_preview, trimesh.Trimesh):
        raise ValueError("uvdelaunay seed did not return a trimesh.Trimesh")

    boundary_ids = tuple(int(v) for v in seed["boundary_vertex_ids"])
    new_vertex_ids = tuple(int(v) for v in seed["new_vertex_ids"])
    new_face_ids = tuple(int(v) for v in seed["new_face_ids"])

    topology_before = validate_patch_topology(
        seed_preview,
        patch_face_ids=new_face_ids,
        boundary_vertex_ids=boundary_ids,
    )

    context_edge_median = mesh_edge_length_median(mesh)
    target_normals = compute_boundary_target_normals(mesh, boundary_ids)

    def _sphere_surface_projector(
        vertex_id: int,
        current: np.ndarray,
        laplacian_target: np.ndarray,
    ) -> np.ndarray:
        del vertex_id
        fit = seed.get("fit")
        if fit is None:
            return np.asarray(laplacian_target, dtype=float)
        try:
            return project_points_to_sphere(
                np.asarray([laplacian_target], dtype=float),
                fit,
            )[0]
        except Exception:
            return project_points_to_sphere(
                np.asarray([current], dtype=float),
                fit,
            )[0]

    relaxation = relax_patch_vertices(
        seed_preview,
        patch_face_ids=new_face_ids,
        fixed_vertex_ids=boundary_ids,
        movable_vertex_ids=new_vertex_ids,
        iterations=int(relaxation_iterations),
        relaxation_strength=float(relaxation_strength),
        surface_projector=_sphere_surface_projector,
        surface_weight=float(surface_weight),
        context_edge_length_median=context_edge_median,
        target_boundary_normals=target_normals,
    )

    preview = relaxation.preview_mesh

    topology_after = validate_patch_topology(
        preview,
        patch_face_ids=new_face_ids,
        boundary_vertex_ids=boundary_ids,
    )

    if not np.allclose(np.asarray(mesh.vertices), source_vertices_before):
        raise RuntimeError("curvature_sphere_uvdelaunay_relaxed unexpectedly mutated source vertices")
    if not np.array_equal(np.asarray(mesh.faces), source_faces_before):
        raise RuntimeError("curvature_sphere_uvdelaunay_relaxed unexpectedly mutated source faces")

    relaxed_points = np.asarray(preview.vertices, dtype=float)[list(new_vertex_ids)]
    seed_points = np.asarray(seed_preview.vertices, dtype=float)[list(new_vertex_ids)]
    displacement = np.linalg.norm(relaxed_points - seed_points, axis=1)

    return {
        "operation": "curvature_sphere_uvdelaunay_relaxed_preview",
        "preview_mesh": preview,
        "rings": int(rings),
        "seed_backend": "curvature_sphere_uvdelaunay",
        "backend": "curvature_sphere_uvdelaunay_relaxed",
        "surface_guidance": "soft_sphere_projection",
        "boundary_vertex_ids": boundary_ids,
        "new_vertex_ids": new_vertex_ids,
        "new_face_ids": new_face_ids,
        "fit": seed.get("fit"),
        "fit_report": seed.get("fit_report"),
        "seed_surface_kind": seed.get("seed_surface_kind"),
        "seed_surface_error": seed.get("seed_surface_error"),
        "interior_uv_count": int(seed.get("interior_uv_count") or 0),
        "delaunay_triangle_count": int(seed.get("delaunay_triangle_count") or 0),
        "topology_before": topology_before.to_dict(),
        "topology_after": topology_after.to_dict(),
        "seam_constraint_report": seed.get("seam_constraint_report"),
        "seam_recovery_decision": seed.get("seam_recovery_decision"),
        "relaxation": relaxation.to_metadata(),
        "relaxation_iterations": int(relaxation_iterations),
        "relaxation_strength": float(relaxation_strength),
        "relaxation_surface_weight": float(surface_weight),
        "relaxed_max_displacement": float(np.max(displacement)) if len(displacement) else 0.0,
        "relaxed_mean_displacement": float(np.mean(displacement)) if len(displacement) else 0.0,
        "notes": (
            "Preview-only uvdelaunay relaxed patch.",
            "Boundary vertices were held fixed.",
            "Only generated patch vertices were relaxed.",
            "Relaxation used quality-gated candidate updates.",
            "No source mesh mutation or project/history write occurred.",
        ),
    }




def build_boundary_loop_diagnostic(boundary_vertex_ids: Any) -> dict[str, Any]:
    """Return diagnostics for simple/non-simple hole boundary loops.

    A valid collar/constrained fill seed expects a single simple loop:
    - at least three unique vertices
    - no non-closing duplicate vertex ids

    Non-consecutive duplicate ids mean the candidate loop is self-touching,
    branched, or represents multiple connected openings. Do not silently
    deduplicate, because that changes the actual boundary topology.
    """

    raw = tuple(int(v) for v in (boundary_vertex_ids or ()))

    normalized = list(raw)
    removed_closing_duplicate = False
    if len(normalized) >= 2 and normalized[0] == normalized[-1]:
        normalized.pop()
        removed_closing_duplicate = True

    positions: dict[int, list[int]] = {}
    for index, vertex_id in enumerate(normalized):
        positions.setdefault(int(vertex_id), []).append(int(index))

    duplicate_positions = {
        int(vertex_id): tuple(int(i) for i in indices)
        for vertex_id, indices in positions.items()
        if len(indices) > 1
    }

    duplicate_vertex_ids = tuple(sorted(int(v) for v in duplicate_positions))

    repeated_edges: list[tuple[int, int]] = []
    edge_positions: dict[tuple[int, int], list[int]] = {}
    count = len(normalized)
    if count >= 2:
        for index, a in enumerate(normalized):
            b = normalized[(index + 1) % count]
            if int(a) == int(b):
                continue
            edge = (int(a), int(b)) if int(a) <= int(b) else (int(b), int(a))
            edge_positions.setdefault(edge, []).append(int(index))

    for edge, indices in edge_positions.items():
        if len(indices) > 1:
            repeated_edges.append(edge)

    simple = (
        len(normalized) >= 3
        and len(duplicate_vertex_ids) == 0
        and len(repeated_edges) == 0
    )

    return {
        "raw_vertex_count": int(len(raw)),
        "boundary_vertex_count": int(len(normalized)),
        "unique_vertex_count": int(len(set(normalized))),
        "removed_closing_duplicate": bool(removed_closing_duplicate),
        "duplicate_vertex_ids": duplicate_vertex_ids,
        "duplicate_positions": duplicate_positions,
        "repeated_edges": tuple(repeated_edges),
        "is_simple_loop": bool(simple),
        "notes": (
            "Boundary loop diagnostic only; no geometry was modified.",
            "Non-closing duplicate vertex ids indicate a non-simple/self-touching loop.",
            "Do not silently deduplicate such loops because it changes topology.",
            "Sealed collar topology requires a simple loop with unique vertex ids.",
        ),
    }


def require_simple_boundary_loop_for_collar(boundary_vertex_ids: Any) -> dict[str, Any]:
    diagnostic = build_boundary_loop_diagnostic(boundary_vertex_ids)
    if not bool(diagnostic.get("is_simple_loop", False)):
        duplicate_ids = diagnostic.get("duplicate_vertex_ids", ())
        repeated_edges = diagnostic.get("repeated_edges", ())
        raise ValueError(
            "Sealed collar requires a simple boundary loop with unique vertex ids. "
            f"duplicate_vertex_ids={duplicate_ids}, repeated_edges={repeated_edges}. "
            "This candidate is likely self-touching, branched, or bore-like; "
            "use the adaptive diagnostic path or split/select the intended boundary loop."
        )
    return diagnostic


def build_surface_uvdelaunay_relaxed_preview_mesh(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int = 2,
    relaxation_iterations: int = 8,
    relaxation_strength: float = 0.20,
    surface_weight: float = 0.20,
    support_rings: int = 2,
    smooth_dot_threshold: float = 0.50,
    biharmonic_max_movement_ratio: float = 1.0,
) -> dict[str, Any]:
    """Build a local-surface UV Delaunay patch and relax it with adaptive guidance.

    Pipeline:
    - preseed support/density diagnosis
    - optional density cap for minimal topology
    - UV Delaunay seed
    - confidence-controlled MLS/Laplacian relaxation
    - gated biharmonic post-fairing when movable vertices exist

    MLS guidance and biharmonic eligibility are intentionally separate:
    a high-normal-spread feature context can disable MLS while still allowing
    biharmonic fairing when the patch has generated/movable vertices.
    """

    from far_mesh.core.hole_patch_quality import (
        analyze_patch_quality,
        compute_boundary_target_normals,
        mesh_edge_length_median,
    )
    from far_mesh.core.hole_patch_relaxation import relax_patch_vertices
    from far_mesh.core.hole_surface_context import (
        build_mls_surface_projector,
        collect_normal_compatible_support_context,
        validate_patch_topology,
    )

    source_vertices_before = np.asarray(mesh.vertices).copy()
    source_faces_before = np.asarray(mesh.faces).copy()

    preseed_boundary_ids = tuple(int(v) for v in candidate_boundary_vertex_ids(candidate))
    boundary_loop_diagnostic = build_boundary_loop_diagnostic(preseed_boundary_ids)

    preseed_support_context = collect_normal_compatible_support_context(
        mesh,
        preseed_boundary_ids,
        max_rings=int(support_rings),
        smooth_dot_threshold=float(smooth_dot_threshold),
    )

    preseed_density_budget_decision = build_local_density_budget_decision(
        support_context=preseed_support_context.to_dict(),
        topology_after={
            "expected_seam_edge_count": int(len(preseed_boundary_ids)),
            "seam_edge_count": int(len(preseed_boundary_ids)),
            "patch_face_count": 0,
            "patch_vertex_count": int(len(preseed_boundary_ids)),
        },
        relaxation={
            "quality_after": {
                "boundary_vertex_count": int(len(preseed_boundary_ids)),
                "patch_face_count": 0,
                "patch_vertex_count": int(len(preseed_boundary_ids)),
            },
        },
    )
    preseed_density_refinement_budget = build_local_density_refinement_budget(
        preseed_density_budget_decision,
    )

    preseed_relaxation_confidence_profile = build_surface_relaxation_confidence_profile(
        density_decision=preseed_density_budget_decision,
        density_refinement_budget=preseed_density_refinement_budget,
        requested_iterations=int(relaxation_iterations),
        requested_relaxation_strength=float(relaxation_strength),
        requested_surface_weight=float(surface_weight),
    )
    effective_relaxation_iterations = int(
        preseed_relaxation_confidence_profile["effective_iterations"]
    )
    effective_relaxation_strength = float(
        preseed_relaxation_confidence_profile["effective_relaxation_strength"]
    )
    effective_surface_weight = float(
        preseed_relaxation_confidence_profile["effective_surface_weight"]
    )

    seed_max_interior_points: int | None = None
    preseed_density_mode = str(
        preseed_density_refinement_budget.get("recommended_density_mode")
        or "allow_current_density"
    )
    preseed_feature_like = bool(
        preseed_density_refinement_budget.get("feature_like_context", False)
    )

    if preseed_density_mode in {"minimal_topology", "capped_local_refinement"} or preseed_feature_like:
        seed_max_interior_points = max(
            0,
            int(preseed_density_refinement_budget.get("max_generated_vertices") or 0),
        )

    seed = build_curvature_sphere_uvdelaunay_preview_mesh(
        mesh,
        candidate,
        rings=rings,
        max_interior_points=seed_max_interior_points,
    )
    seed_preview = seed["preview_mesh"]
    if not isinstance(seed_preview, trimesh.Trimesh):
        raise ValueError("uvdelaunay seed did not return a trimesh.Trimesh")

    boundary_ids = tuple(int(v) for v in seed["boundary_vertex_ids"])
    new_vertex_ids = tuple(int(v) for v in seed["new_vertex_ids"])
    new_face_ids = tuple(int(v) for v in seed["new_face_ids"])

    topology_before = validate_patch_topology(
        seed_preview,
        patch_face_ids=new_face_ids,
        boundary_vertex_ids=boundary_ids,
    )

    if tuple(int(v) for v in boundary_ids) == tuple(int(v) for v in preseed_boundary_ids):
        support_context = preseed_support_context
    else:
        support_context = collect_normal_compatible_support_context(
            mesh,
            boundary_ids,
            max_rings=int(support_rings),
            smooth_dot_threshold=float(smooth_dot_threshold),
        )

    projector = build_mls_surface_projector(
        mesh,
        support_face_ids=support_context.support_face_ids,
        nearest_count=16,
    )

    context_edge_median = mesh_edge_length_median(
        mesh,
        face_ids=support_context.support_face_ids,
    )
    target_normals = compute_boundary_target_normals(
        mesh,
        boundary_ids,
    )

    def _safe_mls_surface_projector(
        vertex_id: int,
        current: np.ndarray,
        laplacian_target: np.ndarray,
    ) -> np.ndarray:
        del vertex_id
        if not bool(preseed_relaxation_confidence_profile.get("surface_guidance_enabled", True)):
            return np.asarray(current, dtype=float)
        try:
            return projector.project(np.asarray(laplacian_target, dtype=float))
        except Exception:
            return np.asarray(current, dtype=float)

    relaxation = relax_patch_vertices(
        seed_preview,
        patch_face_ids=new_face_ids,
        fixed_vertex_ids=boundary_ids,
        movable_vertex_ids=new_vertex_ids,
        iterations=int(effective_relaxation_iterations),
        relaxation_strength=float(effective_relaxation_strength),
        surface_projector=_safe_mls_surface_projector,
        surface_weight=float(effective_surface_weight),
        context_edge_length_median=context_edge_median,
        target_boundary_normals=target_normals,
    )

    preview = relaxation.preview_mesh
    relaxation_metadata = relaxation.to_metadata()

    biharmonic_fairing_metadata: dict[str, Any] | None = None
    biharmonic_fairing_decision: dict[str, Any]

    # Biharmonic eligibility is independent of MLS eligibility.
    # It only requires an actual unknown/movable vertex set.
    if len(new_vertex_ids) > 0:
        from far_mesh.core.hole_biharmonic import fair_patch_biharmonic

        biharmonic_fairing_decision = {
            "attempted": True,
            "applied": False,
            "mode": "post_relaxation_biharmonic_probe",
            "eligible": True,
            "reasons": (),
            "notes": (
                "Biharmonic fairing was attempted because the patch has movable vertices.",
                "MLS may be disabled independently when surface context is unreliable.",
                "The fairing result is accepted only if quality and movement gates pass.",
            ),
        }

        try:
            biharmonic_movement_limit = float(biharmonic_max_movement_ratio)
        except Exception:
            biharmonic_movement_limit = 1.0
        if biharmonic_movement_limit <= 0.0 or not np.isfinite(biharmonic_movement_limit):
            biharmonic_movement_limit = 1.0

        biharmonic_fairing_decision["movement_limit_to_context_edge_ratio"] = float(
            biharmonic_movement_limit
        )

        try:
            fairing = fair_patch_biharmonic(
                preview,
                patch_face_ids=new_face_ids,
                fixed_vertex_ids=boundary_ids,
                movable_vertex_ids=new_vertex_ids,
                laplacian="uniform",
            )

            fair_quality = analyze_patch_quality(
                fairing.preview_mesh,
                patch_face_ids=new_face_ids,
                boundary_vertex_ids=boundary_ids,
                movable_vertex_ids=new_vertex_ids,
                context_edge_length_median=context_edge_median,
                target_boundary_normals=target_normals,
            )
            fair_quality_metadata = fair_quality.to_dict()
            relaxed_quality_metadata = dict(relaxation_metadata.get("quality_after") or {})

            relaxed_min_angle = float(
                relaxed_quality_metadata.get("min_triangle_angle_degrees") or 0.0
            )
            fair_min_angle = float(
                fair_quality_metadata.get("min_triangle_angle_degrees") or 0.0
            )

            relaxed_median_aspect = float(
                relaxed_quality_metadata.get("median_triangle_aspect_ratio") or 0.0
            )
            fair_median_aspect = float(
                fair_quality_metadata.get("median_triangle_aspect_ratio") or 0.0
            )

            fair_degenerate = int(fair_quality_metadata.get("degenerate_face_count") or 0)
            fair_move_ratio = (
                float(fairing.max_displacement) / float(context_edge_median)
                if float(context_edge_median) > 1.0e-12
                else 0.0
            )

            reject_reasons: list[str] = []
            low_displacement_preview: trimesh.Trimesh | None = None
            low_displacement_quality_metadata: dict[str, Any] | None = None
            low_displacement_reject_reasons: list[str] = []
            low_displacement_scale: float | None = None
            low_displacement_max: float | None = None
            low_displacement_mean: float | None = None
            low_displacement_ratio: float | None = None
            if fair_degenerate > 0:
                reject_reasons.append("degenerate_faces_after_biharmonic")
            if relaxed_min_angle > 0.0 and fair_min_angle < relaxed_min_angle * 0.75:
                reject_reasons.append("min_angle_regression_after_biharmonic")
            if (
                relaxed_median_aspect > 0.0
                and fair_median_aspect > max(3.0, relaxed_median_aspect * 1.35)
            ):
                reject_reasons.append("median_aspect_regression_after_biharmonic")
            if fair_move_ratio > biharmonic_movement_limit:
                reject_reasons.append("excessive_biharmonic_movement")

                # H-ADAPT-5C5:
                # Use the full biharmonic result as a direction, but scale the
                # movement down to the adaptive movement limit. This gives the
                # curvature fairing a strict low-displacement chance without
                # moving seam/boundary vertices.
                try:
                    if (
                        fair_move_ratio > 1.0e-12
                        and biharmonic_movement_limit > 0.0
                        and len(new_vertex_ids) > 0
                    ):
                        low_displacement_scale = min(
                            1.0,
                            float(biharmonic_movement_limit) / float(fair_move_ratio),
                        )

                        base_vertices = np.asarray(preview.vertices, dtype=float)
                        fair_vertices = np.asarray(fairing.preview_mesh.vertices, dtype=float)

                        limited_vertices = np.array(base_vertices, dtype=float, copy=True)
                        movable = np.asarray(tuple(int(v) for v in new_vertex_ids), dtype=np.int64)
                        limited_vertices[movable] = (
                            base_vertices[movable]
                            + (fair_vertices[movable] - base_vertices[movable])
                            * float(low_displacement_scale)
                        )

                        low_displacement_preview = trimesh.Trimesh(
                            vertices=limited_vertices,
                            faces=np.asarray(preview.faces, dtype=np.int64),
                            process=False,
                        )

                        low_quality = analyze_patch_quality(
                            low_displacement_preview,
                            patch_face_ids=new_face_ids,
                            boundary_vertex_ids=boundary_ids,
                            movable_vertex_ids=new_vertex_ids,
                            context_edge_length_median=context_edge_median,
                            target_boundary_normals=target_normals,
                        )
                        low_displacement_quality_metadata = low_quality.to_dict()

                        low_points = np.asarray(
                            low_displacement_preview.vertices,
                            dtype=float,
                        )[list(new_vertex_ids)]
                        base_points = np.asarray(preview.vertices, dtype=float)[list(new_vertex_ids)]
                        low_move = np.linalg.norm(low_points - base_points, axis=1)
                        if len(low_move):
                            low_displacement_max = float(np.max(low_move))
                            low_displacement_mean = float(np.mean(low_move))
                        else:
                            low_displacement_max = 0.0
                            low_displacement_mean = 0.0

                        low_displacement_ratio = (
                            float(low_displacement_max) / float(context_edge_median)
                            if float(context_edge_median) > 1.0e-12
                            else 0.0
                        )

                        low_degenerate = int(
                            low_displacement_quality_metadata.get("degenerate_face_count") or 0
                        )
                        low_min_angle = float(
                            low_displacement_quality_metadata.get("min_triangle_angle_degrees")
                            or 0.0
                        )
                        low_median_aspect = float(
                            low_displacement_quality_metadata.get("median_triangle_aspect_ratio")
                            or 0.0
                        )

                        if low_degenerate > 0:
                            low_displacement_reject_reasons.append(
                                "degenerate_faces_after_low_displacement_biharmonic"
                            )
                        if relaxed_min_angle > 0.0 and low_min_angle < relaxed_min_angle * 0.75:
                            low_displacement_reject_reasons.append(
                                "min_angle_regression_after_low_displacement_biharmonic"
                            )
                        if (
                            relaxed_median_aspect > 0.0
                            and low_median_aspect > max(3.0, relaxed_median_aspect * 1.35)
                        ):
                            low_displacement_reject_reasons.append(
                                "median_aspect_regression_after_low_displacement_biharmonic"
                            )
                        if low_displacement_ratio > biharmonic_movement_limit * 1.000001:
                            low_displacement_reject_reasons.append(
                                "excessive_low_displacement_biharmonic_movement"
                            )
                except Exception as low_exc:
                    low_displacement_preview = None
                    low_displacement_quality_metadata = None
                    low_displacement_reject_reasons.append(
                        f"low_displacement_biharmonic_error:{low_exc}"
                    )

            biharmonic_fairing_metadata = fairing.to_metadata()
            biharmonic_fairing_metadata["quality_after"] = fair_quality_metadata
            biharmonic_fairing_metadata["movement_to_context_edge_ratio"] = float(fair_move_ratio)
            biharmonic_fairing_metadata["movement_limit_to_context_edge_ratio"] = float(
                biharmonic_movement_limit
            )

            if reject_reasons:
                if (
                    "excessive_biharmonic_movement" in reject_reasons
                    and low_displacement_preview is not None
                    and low_displacement_quality_metadata is not None
                    and not low_displacement_reject_reasons
                ):
                    full_fairing_metadata = dict(biharmonic_fairing_metadata)
                    preview = low_displacement_preview
                    relaxation_metadata["quality_after"] = low_displacement_quality_metadata
                    relaxation_metadata["biharmonic_fairing_applied"] = True
                    relaxation_metadata["biharmonic_fairing_limited"] = True

                    biharmonic_fairing_metadata = {
                        **full_fairing_metadata,
                        "quality_after": low_displacement_quality_metadata,
                        "max_displacement": float(low_displacement_max or 0.0),
                        "mean_displacement": float(low_displacement_mean or 0.0),
                        "movement_to_context_edge_ratio": float(low_displacement_ratio or 0.0),
                        "movement_limit_to_context_edge_ratio": float(
                            biharmonic_movement_limit
                        ),
                        "low_displacement_scale": float(low_displacement_scale or 0.0),
                        "low_displacement_limited": True,
                        "full_biharmonic_max_displacement": float(fairing.max_displacement),
                        "full_biharmonic_mean_displacement": float(fairing.mean_displacement),
                        "full_biharmonic_movement_to_context_edge_ratio": float(
                            fair_move_ratio
                        ),
                    }
                    biharmonic_fairing_decision = {
                        **biharmonic_fairing_decision,
                        "applied": True,
                        "reasons": ("low_displacement_biharmonic_fairing_applied",),
                        "notes": (
                            "Full biharmonic fairing exceeded the adaptive movement limit.",
                            "A low-displacement scaled fairing candidate was accepted instead.",
                            "Boundary vertices remained fixed; only generated patch vertices were blended.",
                        ),
                        "low_displacement_scale": float(low_displacement_scale or 0.0),
                        "full_reject_reasons": tuple(reject_reasons),
                        "low_displacement_reject_reasons": (),
                    }
                else:
                    combined_reasons = list(reject_reasons)
                    combined_reasons.extend(
                        f"low_displacement:{reason}"
                        for reason in low_displacement_reject_reasons
                    )
                    biharmonic_fairing_decision = {
                        **biharmonic_fairing_decision,
                        "applied": False,
                        "reasons": tuple(combined_reasons),
                        "notes": (
                            "Biharmonic fairing was computed but not applied.",
                            "The pre-fairing preview remains active because the quality or movement gate failed.",
                        ),
                        "low_displacement_scale": (
                            None
                            if low_displacement_scale is None
                            else float(low_displacement_scale)
                        ),
                        "low_displacement_reject_reasons": tuple(
                            low_displacement_reject_reasons
                        ),
                    }
            else:
                preview = fairing.preview_mesh
                relaxation_metadata["quality_after"] = fair_quality_metadata
                relaxation_metadata["biharmonic_fairing_applied"] = True
                biharmonic_fairing_decision = {
                    **biharmonic_fairing_decision,
                    "applied": True,
                    "reasons": (),
                    "notes": (
                        "Biharmonic fairing was applied after the current relaxation stage.",
                        "Quality and movement gates accepted the fairing result.",
                    ),
                }

        except Exception as exc:
            biharmonic_fairing_decision = {
                **biharmonic_fairing_decision,
                "applied": False,
                "reasons": ("biharmonic_fairing_error",),
                "error": str(exc),
                "notes": (
                    "Biharmonic fairing failed and was skipped.",
                    "The pre-fairing preview remains active.",
                ),
            }
    else:
        biharmonic_fairing_decision = {
            "attempted": False,
            "applied": False,
            "eligible": False,
            "mode": "not_applicable_boundary_only_patch",
            "reasons": ("no_movable_vertices",),
            "notes": (
                "Biharmonic fairing was not attempted because there are no movable vertices.",
                "This is not a planar/feature exclusion; it is a mathematical no-op case.",
                "Planar or feature-like patches with movable vertices remain eligible.",
            ),
        }

    topology_after = validate_patch_topology(
        preview,
        patch_face_ids=new_face_ids,
        boundary_vertex_ids=boundary_ids,
    )

    if not np.allclose(np.asarray(mesh.vertices), source_vertices_before):
        raise RuntimeError("surface_uvdelaunay_relaxed unexpectedly mutated source vertices")
    if not np.array_equal(np.asarray(mesh.faces), source_faces_before):
        raise RuntimeError("surface_uvdelaunay_relaxed unexpectedly mutated source faces")

    relaxed_points = np.asarray(preview.vertices, dtype=float)[list(new_vertex_ids)]
    seed_points = np.asarray(seed_preview.vertices, dtype=float)[list(new_vertex_ids)]
    displacement = np.linalg.norm(relaxed_points - seed_points, axis=1)

    support_context_metadata = support_context.to_dict()
    topology_after_metadata = topology_after.to_dict()

    surface_context_decision = build_surface_context_safety_decision(
        support_context=support_context_metadata,
        topology_after=topology_after_metadata,
        relaxation=relaxation_metadata,
    )
    local_density_budget_decision = build_local_density_budget_decision(
        support_context=support_context_metadata,
        topology_after=topology_after_metadata,
        relaxation=relaxation_metadata,
    )
    local_density_refinement_budget = build_local_density_refinement_budget(
        local_density_budget_decision,
    )

    return {
        "operation": "surface_uvdelaunay_relaxed_preview",
        "preview_mesh": preview,
        "rings": int(rings),
        "seed_backend": "curvature_sphere_uvdelaunay",
        "backend": "surface_uvdelaunay_relaxed",
        "surface_guidance": "normal_compatible_mls_quadratic",
        "sphere_fit_diagnostic": seed.get("fit"),
        "boundary_vertex_ids": boundary_ids,
        "new_vertex_ids": new_vertex_ids,
        "new_face_ids": new_face_ids,
        "center_vertex_id": None,
        "interior_uv_count": int(seed["interior_uv_count"]),
        "delaunay_triangle_count": int(seed["delaunay_triangle_count"]),
        "seam_constraint_report": seed.get("seam_constraint_report"),
        "seam_recovery_decision": seed.get("seam_recovery_decision"),
        "boundary_loop_diagnostic": boundary_loop_diagnostic,
        "support_context": support_context_metadata,
        "mls_projector": projector.to_dict(),
        "topology_before": topology_before.to_dict(),
        "topology_after": topology_after_metadata,
        "relaxation": relaxation_metadata,
        "biharmonic_fairing": biharmonic_fairing_metadata,
        "biharmonic_fairing_decision": biharmonic_fairing_decision,
        "surface_context_decision": surface_context_decision,
        "local_density_budget_decision": local_density_budget_decision,
        "local_density_refinement_budget": local_density_refinement_budget,
        "preseed_local_density_budget_decision": preseed_density_budget_decision,
        "preseed_local_density_refinement_budget": preseed_density_refinement_budget,
        "seed_max_interior_points": None if seed_max_interior_points is None else int(seed_max_interior_points),
        "preseed_relaxation_confidence_profile": preseed_relaxation_confidence_profile,
        "requested_relaxation_iterations": int(relaxation_iterations),
        "requested_relaxation_strength": float(relaxation_strength),
        "requested_relaxation_surface_weight": float(surface_weight),
        "relaxation_iterations": int(effective_relaxation_iterations),
        "relaxation_strength": float(effective_relaxation_strength),
        "relaxation_surface_weight": float(effective_surface_weight),
        "support_rings": int(support_rings),
        "smooth_dot_threshold": float(smooth_dot_threshold),
        "relaxed_max_displacement": float(np.max(displacement)) if len(displacement) else 0.0,
        "relaxed_mean_displacement": float(np.mean(displacement)) if len(displacement) else 0.0,
        "notes": (
            "Preview-only local surface UV Delaunay relaxed patch.",
            "Seed topology came from curvature_sphere_uvdelaunay.",
            "Preseed density/context decisions may cap interior sampling.",
            "MLS guidance and biharmonic fairing eligibility are evaluated separately.",
            "Biharmonic fairing is eligible whenever generated/movable vertices exist.",
            "Boundary-only patches have no movable vertices, so biharmonic is a no-op.",
            "Boundary vertices were held fixed.",
            "No source mesh mutation or project/history write occurred.",
        ),
    }



def build_surface_uvdelaunay_sealed_seed_preview_mesh(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int = 2,
    collar_fraction: float = 0.30,
) -> dict[str, Any]:
    """Build a seam-sealed patch seed using an inward quad collar.

    This seed intentionally prioritizes topological correctness:
    - every original boundary edge is consumed exactly once
    - radial collar edges are shared by adjacent collar quads
    - the inner ring is closed with a safe center fan
    - later relaxation can improve geometry without breaking the seam

    The existing uvdelaunay builder is used only to recover the ordered boundary
    loop and diagnostic seed metadata. The generated sealed topology is new.
    """

    from far_mesh.core.hole_sealed_collar import build_sealed_quad_collar_topology
    from far_mesh.core.hole_surface_context import validate_patch_topology

    if not (0.02 <= float(collar_fraction) <= 0.80):
        raise ValueError("collar_fraction must be in [0.02, 0.80]")

    source_vertices_before = np.asarray(mesh.vertices).copy()
    source_faces_before = np.asarray(mesh.faces).copy()

    # Use the existing seed only for boundary-loop order and diagnostics.
    reference = build_curvature_sphere_uvdelaunay_preview_mesh(
        mesh,
        candidate,
        rings=rings,
    )

    boundary_ids = tuple(int(v) for v in reference["boundary_vertex_ids"])
    require_simple_boundary_loop_for_collar(boundary_ids)
    if len(boundary_ids) < 3:
        raise ValueError("sealed patch requires at least three boundary vertices")

    base_vertices = np.asarray(mesh.vertices, dtype=float)
    base_faces = np.asarray(mesh.faces, dtype=np.int64)
    boundary_positions = base_vertices[list(boundary_ids)]

    boundary_centroid = np.mean(boundary_positions, axis=0)
    center_target = np.asarray(reference.get("projected_center", boundary_centroid), dtype=float).reshape(3)
    if not np.isfinite(center_target).all():
        center_target = boundary_centroid

    inner_positions: list[np.ndarray] = []
    for boundary_position in boundary_positions:
        direction = center_target - boundary_position
        if float(np.linalg.norm(direction)) <= 1.0e-12:
            direction = boundary_centroid - boundary_position
        inner_positions.append(
            boundary_position + float(collar_fraction) * direction
        )

    inner_array = np.asarray(inner_positions, dtype=float)
    center_position = np.mean(inner_array, axis=0)

    first_new_vertex_id = int(len(base_vertices))
    inner_vertex_ids = tuple(
        range(first_new_vertex_id, first_new_vertex_id + len(boundary_ids))
    )
    center_vertex_id = first_new_vertex_id + len(boundary_ids)

    topology = build_sealed_quad_collar_topology(
        boundary_ids,
        inner_vertex_ids,
    )

    inner_fan_faces: list[tuple[int, int, int]] = []
    for index, inner_vertex_id in enumerate(inner_vertex_ids):
        next_inner_vertex_id = inner_vertex_ids[(index + 1) % len(inner_vertex_ids)]
        inner_fan_faces.append(
            (int(inner_vertex_id), int(next_inner_vertex_id), int(center_vertex_id))
        )

    new_faces = tuple(topology.collar_faces) + tuple(inner_fan_faces)
    new_face_ids = tuple(range(int(len(base_faces)), int(len(base_faces)) + len(new_faces)))
    new_vertex_ids = tuple(inner_vertex_ids) + (int(center_vertex_id),)

    preview_vertices = np.vstack(
        [
            base_vertices,
            inner_array,
            np.asarray([center_position], dtype=float),
        ]
    )
    preview_faces = np.vstack(
        [
            base_faces,
            np.asarray(new_faces, dtype=np.int64),
        ]
    )

    preview = trimesh.Trimesh(
        vertices=preview_vertices,
        faces=preview_faces,
        process=False,
    )

    topology_report = validate_patch_topology(
        preview,
        patch_face_ids=new_face_ids,
        boundary_vertex_ids=boundary_ids,
    )

    if not np.allclose(np.asarray(mesh.vertices), source_vertices_before):
        raise RuntimeError("sealed seed unexpectedly mutated source vertices")
    if not np.array_equal(np.asarray(mesh.faces), source_faces_before):
        raise RuntimeError("sealed seed unexpectedly mutated source faces")

    return {
        "operation": "surface_uvdelaunay_sealed_seed_preview",
        "preview_mesh": preview,
        "rings": int(rings),
        "backend": "surface_uvdelaunay_sealed_seed",
        "seed_reference_backend": "curvature_sphere_uvdelaunay",
        "collar_mode": "quad_strip_inner_fan",
        "collar_fraction": float(collar_fraction),
        "boundary_vertex_ids": boundary_ids,
        "inner_vertex_ids": tuple(int(v) for v in inner_vertex_ids),
        "center_vertex_id": int(center_vertex_id),
        "new_vertex_ids": new_vertex_ids,
        "new_face_ids": new_face_ids,
        "collar_face_ids": tuple(new_face_ids[: len(topology.collar_faces)]),
        "inner_fill_face_ids": tuple(new_face_ids[len(topology.collar_faces):]),
        "topology": topology_report.to_dict(),
        "reference": {
            "backend": reference.get("backend"),
            "interior_uv_count": int(reference.get("interior_uv_count", 0)),
            "delaunay_triangle_count": int(reference.get("delaunay_triangle_count", 0)),
            "spacing": float(reference.get("spacing", 0.0)),
            "target_edge_length": float(reference.get("target_edge_length", 0.0)),
            "fit": reference.get("fit"),
        },
        "notes": (
            "Preview-only seam-sealed seed.",
            "Every original boundary edge is consumed by the forced quad collar.",
            "The remaining inner ring is closed by a deterministic center fan.",
            "Geometry is intentionally simple; downstream MLS relaxation improves placement.",
            "No source mesh mutation or project/history write occurred.",
        ),
    }


def build_surface_uvdelaunay_sealed_relaxed_preview_mesh(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int = 2,
    collar_fraction: float = 0.30,
    relaxation_iterations: int = 8,
    relaxation_strength: float = 0.20,
    surface_weight: float = 0.20,
    support_rings: int = 2,
    smooth_dot_threshold: float = 0.50,
) -> dict[str, Any]:
    """Build a seam-sealed local-surface patch and relax it with MLS guidance."""

    from far_mesh.core.hole_patch_quality import (
        compute_boundary_target_normals,
        mesh_edge_length_median,
    )
    from far_mesh.core.hole_patch_relaxation import relax_patch_vertices
    from far_mesh.core.hole_surface_context import (
        build_mls_surface_projector,
        collect_normal_compatible_support_context,
        validate_patch_topology,
    )

    source_vertices_before = np.asarray(mesh.vertices).copy()
    source_faces_before = np.asarray(mesh.faces).copy()

    seed = build_surface_uvdelaunay_sealed_seed_preview_mesh(
        mesh,
        candidate,
        rings=rings,
        collar_fraction=collar_fraction,
    )

    seed_preview = seed["preview_mesh"]
    if not isinstance(seed_preview, trimesh.Trimesh):
        raise ValueError("sealed seed did not return a trimesh.Trimesh")

    boundary_ids = tuple(int(v) for v in seed["boundary_vertex_ids"])
    new_vertex_ids = tuple(int(v) for v in seed["new_vertex_ids"])
    new_face_ids = tuple(int(v) for v in seed["new_face_ids"])

    topology_before = validate_patch_topology(
        seed_preview,
        patch_face_ids=new_face_ids,
        boundary_vertex_ids=boundary_ids,
    )

    support_context = collect_normal_compatible_support_context(
        mesh,
        boundary_ids,
        max_rings=int(support_rings),
        smooth_dot_threshold=float(smooth_dot_threshold),
    )
    projector = build_mls_surface_projector(
        mesh,
        support_face_ids=support_context.support_face_ids,
        nearest_count=16,
    )

    context_edge_median = mesh_edge_length_median(
        mesh,
        face_ids=support_context.support_face_ids,
    )
    target_normals = compute_boundary_target_normals(
        mesh,
        boundary_ids,
    )

    def _safe_mls_surface_projector(
        vertex_id: int,
        current: np.ndarray,
        laplacian_target: np.ndarray,
    ) -> np.ndarray:
        del vertex_id
        try:
            return projector.project(np.asarray(laplacian_target, dtype=float))
        except Exception:
            return np.asarray(current, dtype=float)

    relaxation = relax_patch_vertices(
        seed_preview,
        patch_face_ids=new_face_ids,
        fixed_vertex_ids=boundary_ids,
        movable_vertex_ids=new_vertex_ids,
        iterations=int(relaxation_iterations),
        relaxation_strength=float(relaxation_strength),
        surface_projector=_safe_mls_surface_projector,
        surface_weight=float(surface_weight),
        context_edge_length_median=context_edge_median,
        target_boundary_normals=target_normals,
    )

    preview = relaxation.preview_mesh

    topology_after = validate_patch_topology(
        preview,
        patch_face_ids=new_face_ids,
        boundary_vertex_ids=boundary_ids,
    )

    if not np.allclose(np.asarray(mesh.vertices), source_vertices_before):
        raise RuntimeError("surface_uvdelaunay_sealed_relaxed unexpectedly mutated source vertices")
    if not np.array_equal(np.asarray(mesh.faces), source_faces_before):
        raise RuntimeError("surface_uvdelaunay_sealed_relaxed unexpectedly mutated source faces")

    relaxed_points = np.asarray(preview.vertices, dtype=float)[list(new_vertex_ids)]
    seed_points = np.asarray(seed_preview.vertices, dtype=float)[list(new_vertex_ids)]
    displacement = np.linalg.norm(relaxed_points - seed_points, axis=1)

    return {
        "operation": "surface_uvdelaunay_sealed_relaxed_preview",
        "preview_mesh": preview,
        "rings": int(rings),
        "seed_backend": "surface_uvdelaunay_sealed_seed",
        "backend": "surface_uvdelaunay_sealed_relaxed",
        "surface_guidance": "normal_compatible_mls_quadratic",
        "collar_mode": "quad_strip_inner_fan",
        "collar_fraction": float(collar_fraction),
        "boundary_vertex_ids": boundary_ids,
        "inner_vertex_ids": tuple(int(v) for v in seed["inner_vertex_ids"]),
        "center_vertex_id": int(seed["center_vertex_id"]),
        "new_vertex_ids": new_vertex_ids,
        "new_face_ids": new_face_ids,
        "collar_face_ids": tuple(int(v) for v in seed["collar_face_ids"]),
        "inner_fill_face_ids": tuple(int(v) for v in seed["inner_fill_face_ids"]),
        "support_context": support_context.to_dict(),
        "mls_projector": projector.to_dict(),
        "topology_before": topology_before.to_dict(),
        "topology_after": topology_after.to_dict(),
        "relaxation": relaxation.to_metadata(),
        "relaxation_iterations": int(relaxation_iterations),
        "relaxation_strength": float(relaxation_strength),
        "relaxation_surface_weight": float(surface_weight),
        "support_rings": int(support_rings),
        "smooth_dot_threshold": float(smooth_dot_threshold),
        "relaxed_max_displacement": float(np.max(displacement)) if len(displacement) else 0.0,
        "relaxed_mean_displacement": float(np.mean(displacement)) if len(displacement) else 0.0,
        "notes": (
            "Preview-only seam-sealed local surface patch.",
            "Boundary seam is sealed by a forced quad collar before relaxation.",
            "Surface target came from normal-compatible support faces and MLS/quadratic projection.",
            "Boundary vertices were held fixed.",
            "Only generated patch vertices were relaxed.",
            "Relaxation used quality-gated candidate updates.",
            "No source mesh mutation or project/history write occurred.",
        ),
    }


def build_surface_uvdelaunay_sealed_dense_seed_preview_mesh(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int = 2,
    collar_fraction: float = 0.15,
    dense_inner_rings: int = 3,
) -> dict[str, Any]:
    """Build a seam-sealed dense patch seed using a quad collar + inner rings.

    This improves over the first sealed seed:
    - keeps every original boundary edge consumed by a quad collar
    - replaces the large center fan with multiple inner quad/ring strips
    - leaves only a small final fan near the center
    - gives MLS relaxation more degrees of freedom
    """

    from far_mesh.core.hole_sealed_collar import build_sealed_quad_collar_topology
    from far_mesh.core.hole_surface_context import validate_patch_topology

    dense_inner_rings = int(dense_inner_rings)
    if dense_inner_rings < 2:
        raise ValueError("dense_inner_rings must be at least 2")
    if not (0.02 <= float(collar_fraction) <= 0.60):
        raise ValueError("collar_fraction must be in [0.02, 0.60]")

    source_vertices_before = np.asarray(mesh.vertices).copy()
    source_faces_before = np.asarray(mesh.faces).copy()

    reference = build_curvature_sphere_uvdelaunay_preview_mesh(
        mesh,
        candidate,
        rings=rings,
    )

    boundary_ids = tuple(int(v) for v in reference["boundary_vertex_ids"])
    require_simple_boundary_loop_for_collar(boundary_ids)
    if len(boundary_ids) < 3:
        raise ValueError("sealed dense patch requires at least three boundary vertices")

    base_vertices = np.asarray(mesh.vertices, dtype=float)
    base_faces = np.asarray(mesh.faces, dtype=np.int64)
    boundary_positions = base_vertices[list(boundary_ids)]

    boundary_centroid = np.mean(boundary_positions, axis=0)
    center_target = np.asarray(reference.get("projected_center", boundary_centroid), dtype=float).reshape(3)
    if not np.isfinite(center_target).all():
        center_target = boundary_centroid

    # Multiple rings reduce the visible center-pole / indentation artifact.
    # First ring is the seam collar; last ring is still before the center.
    fractions = np.linspace(
        float(collar_fraction),
        0.82,
        dense_inner_rings,
        dtype=float,
    )

    ring_positions: list[np.ndarray] = []
    generated_positions: list[np.ndarray] = []
    for fraction in fractions:
        ring = []
        for boundary_position in boundary_positions:
            position = boundary_position + fraction * (center_target - boundary_position)
            ring.append(position)
            generated_positions.append(position)
        ring_positions.append(np.asarray(ring, dtype=float))

    center_position = np.mean(ring_positions[-1], axis=0)

    first_new_vertex_id = int(len(base_vertices))
    ring_vertex_ids: list[tuple[int, ...]] = []
    cursor = first_new_vertex_id
    for _ring_index in range(dense_inner_rings):
        ids = tuple(range(cursor, cursor + len(boundary_ids)))
        ring_vertex_ids.append(tuple(int(v) for v in ids))
        cursor += len(boundary_ids)

    center_vertex_id = int(cursor)

    topology = build_sealed_quad_collar_topology(
        boundary_ids,
        ring_vertex_ids[0],
    )

    new_faces: list[tuple[int, int, int]] = list(topology.collar_faces)

    # Fill between generated rings with quad strips.
    for ring_index in range(dense_inner_rings - 1):
        outer = ring_vertex_ids[ring_index]
        inner = ring_vertex_ids[ring_index + 1]
        for index, outer_vertex_id in enumerate(outer):
            outer_next = outer[(index + 1) % len(outer)]
            inner_vertex_id = inner[index]
            inner_next = inner[(index + 1) % len(inner)]
            new_faces.append((int(outer_vertex_id), int(outer_next), int(inner_next)))
            new_faces.append((int(outer_vertex_id), int(inner_next), int(inner_vertex_id)))

    # Small final fan only at the innermost ring.
    last_ring = ring_vertex_ids[-1]
    for index, inner_vertex_id in enumerate(last_ring):
        next_inner_vertex_id = last_ring[(index + 1) % len(last_ring)]
        new_faces.append((int(inner_vertex_id), int(next_inner_vertex_id), int(center_vertex_id)))

    generated_vertices = np.vstack(
        [
            np.asarray(generated_positions, dtype=float),
            np.asarray([center_position], dtype=float),
        ]
    )

    new_face_ids = tuple(range(int(len(base_faces)), int(len(base_faces)) + len(new_faces)))
    new_vertex_ids = tuple(range(first_new_vertex_id, center_vertex_id + 1))

    preview_vertices = np.vstack([base_vertices, generated_vertices])
    preview_faces = np.vstack([base_faces, np.asarray(new_faces, dtype=np.int64)])

    preview = trimesh.Trimesh(
        vertices=preview_vertices,
        faces=preview_faces,
        process=False,
    )

    topology_report = validate_patch_topology(
        preview,
        patch_face_ids=new_face_ids,
        boundary_vertex_ids=boundary_ids,
    )

    if not np.allclose(np.asarray(mesh.vertices), source_vertices_before):
        raise RuntimeError("sealed dense seed unexpectedly mutated source vertices")
    if not np.array_equal(np.asarray(mesh.faces), source_faces_before):
        raise RuntimeError("sealed dense seed unexpectedly mutated source faces")

    collar_face_count = len(topology.collar_faces)

    return {
        "operation": "surface_uvdelaunay_sealed_dense_seed_preview",
        "preview_mesh": preview,
        "rings": int(rings),
        "backend": "surface_uvdelaunay_sealed_dense_seed",
        "seed_reference_backend": "curvature_sphere_uvdelaunay",
        "collar_mode": "quad_strip_multiring_inner_fill",
        "collar_fraction": float(collar_fraction),
        "dense_inner_rings": int(dense_inner_rings),
        "boundary_vertex_ids": boundary_ids,
        "ring_vertex_ids": tuple(tuple(int(v) for v in ids) for ids in ring_vertex_ids),
        "inner_vertex_ids": tuple(int(v) for v in ring_vertex_ids[0]),
        "center_vertex_id": int(center_vertex_id),
        "new_vertex_ids": tuple(int(v) for v in new_vertex_ids),
        "new_face_ids": tuple(int(v) for v in new_face_ids),
        "collar_face_ids": tuple(int(v) for v in new_face_ids[:collar_face_count]),
        "inner_fill_face_ids": tuple(int(v) for v in new_face_ids[collar_face_count:]),
        "topology": topology_report.to_dict(),
        "reference": {
            "backend": reference.get("backend"),
            "interior_uv_count": int(reference.get("interior_uv_count", 0)),
            "delaunay_triangle_count": int(reference.get("delaunay_triangle_count", 0)),
            "spacing": float(reference.get("spacing", 0.0)),
            "target_edge_length": float(reference.get("target_edge_length", 0.0)),
            "fit": reference.get("fit"),
        },
        "notes": (
            "Preview-only seam-sealed dense seed.",
            "Every original boundary edge is consumed by the forced quad collar.",
            "The interior is filled by multiple ring strips before the final center fan.",
            "This reduces the indentation/pole artifact of the first sealed seed.",
            "No source mesh mutation or project/history write occurred.",
        ),
    }


def build_surface_uvdelaunay_sealed_dense_relaxed_preview_mesh(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int = 2,
    collar_fraction: float = 0.15,
    dense_inner_rings: int = 3,
    relaxation_iterations: int = 10,
    relaxation_strength: float = 0.18,
    surface_weight: float = 0.28,
    support_rings: int = 2,
    smooth_dot_threshold: float = 0.50,
) -> dict[str, Any]:
    """Build a dense seam-sealed local-surface patch and relax it with MLS guidance."""

    from far_mesh.core.hole_patch_quality import (
        compute_boundary_target_normals,
        mesh_edge_length_median,
    )
    from far_mesh.core.hole_patch_relaxation import relax_patch_vertices
    from far_mesh.core.hole_surface_context import (
        build_mls_surface_projector,
        collect_normal_compatible_support_context,
        validate_patch_topology,
    )

    source_vertices_before = np.asarray(mesh.vertices).copy()
    source_faces_before = np.asarray(mesh.faces).copy()

    seed = build_surface_uvdelaunay_sealed_dense_seed_preview_mesh(
        mesh,
        candidate,
        rings=rings,
        collar_fraction=collar_fraction,
        dense_inner_rings=dense_inner_rings,
    )

    seed_preview = seed["preview_mesh"]
    if not isinstance(seed_preview, trimesh.Trimesh):
        raise ValueError("sealed dense seed did not return a trimesh.Trimesh")

    boundary_ids = tuple(int(v) for v in seed["boundary_vertex_ids"])
    new_vertex_ids = tuple(int(v) for v in seed["new_vertex_ids"])
    new_face_ids = tuple(int(v) for v in seed["new_face_ids"])

    topology_before = validate_patch_topology(
        seed_preview,
        patch_face_ids=new_face_ids,
        boundary_vertex_ids=boundary_ids,
    )

    support_context = collect_normal_compatible_support_context(
        mesh,
        boundary_ids,
        max_rings=int(support_rings),
        smooth_dot_threshold=float(smooth_dot_threshold),
    )
    projector = build_mls_surface_projector(
        mesh,
        support_face_ids=support_context.support_face_ids,
        nearest_count=16,
    )

    context_edge_median = mesh_edge_length_median(
        mesh,
        face_ids=support_context.support_face_ids,
    )
    target_normals = compute_boundary_target_normals(mesh, boundary_ids)

    def _safe_mls_surface_projector(
        vertex_id: int,
        current: np.ndarray,
        laplacian_target: np.ndarray,
    ) -> np.ndarray:
        del vertex_id
        try:
            return projector.project(np.asarray(laplacian_target, dtype=float))
        except Exception:
            return np.asarray(current, dtype=float)

    relaxation = relax_patch_vertices(
        seed_preview,
        patch_face_ids=new_face_ids,
        fixed_vertex_ids=boundary_ids,
        movable_vertex_ids=new_vertex_ids,
        iterations=int(relaxation_iterations),
        relaxation_strength=float(relaxation_strength),
        surface_projector=_safe_mls_surface_projector,
        surface_weight=float(surface_weight),
        context_edge_length_median=context_edge_median,
        target_boundary_normals=target_normals,
    )

    preview = relaxation.preview_mesh

    topology_after = validate_patch_topology(
        preview,
        patch_face_ids=new_face_ids,
        boundary_vertex_ids=boundary_ids,
    )

    if not np.allclose(np.asarray(mesh.vertices), source_vertices_before):
        raise RuntimeError("surface_uvdelaunay_sealed_dense_relaxed unexpectedly mutated source vertices")
    if not np.array_equal(np.asarray(mesh.faces), source_faces_before):
        raise RuntimeError("surface_uvdelaunay_sealed_dense_relaxed unexpectedly mutated source faces")

    relaxed_points = np.asarray(preview.vertices, dtype=float)[list(new_vertex_ids)]
    seed_points = np.asarray(seed_preview.vertices, dtype=float)[list(new_vertex_ids)]
    displacement = np.linalg.norm(relaxed_points - seed_points, axis=1)

    relaxation_metadata = relaxation.to_metadata()
    support_context_metadata = support_context.to_dict()

    collar_vertex_ids = tuple(int(v) for v in seed.get("inner_vertex_ids", ()))
    center_vertex_id = int(seed["center_vertex_id"])
    collar_id_set = set(collar_vertex_ids)
    interior_vertex_ids = tuple(
        int(v)
        for v in new_vertex_ids
        if int(v) not in collar_id_set and int(v) != center_vertex_id
    )

    preview_vertices = np.asarray(preview.vertices, dtype=float)
    seed_vertices = np.asarray(seed_preview.vertices, dtype=float)

    def _movement_stats(vertex_ids: tuple[int, ...]) -> dict[str, float | int]:
        if not vertex_ids:
            return {
                "count": 0,
                "mean": 0.0,
                "max": 0.0,
            }

        deltas = np.linalg.norm(
            preview_vertices[list(vertex_ids)] - seed_vertices[list(vertex_ids)],
            axis=1,
        )
        return {
            "count": int(len(vertex_ids)),
            "mean": float(np.mean(deltas)),
            "max": float(np.max(deltas)),
        }

    surface_projection_deltas = []
    for vertex_id in new_vertex_ids:
        seed_point = seed_vertices[int(vertex_id)]
        try:
            projected = projector.project(seed_point)
            surface_projection_deltas.append(
                float(np.linalg.norm(np.asarray(projected, dtype=float) - seed_point))
            )
        except Exception:
            continue

    support_face_count = int(
        support_context_metadata.get(
            "support_face_count",
            len(getattr(support_context, "support_face_ids", ())),
        )
    )
    support_vertex_count = int(
        support_context_metadata.get(
            "support_vertex_count",
            len(getattr(support_context, "support_vertex_ids", ())),
        )
    )
    support_normal_spread = float(
        support_context_metadata.get(
            "normal_spread_degrees",
            support_context_metadata.get("support_normal_spread_degrees", 0.0),
        )
    )
    support_contamination_score = float(
        support_context_metadata.get(
            "contamination_score",
            support_context_metadata.get("support_contamination_score", 0.0),
        )
    )

    dynamic_diagnostics = {
        "support_face_count": support_face_count,
        "support_vertex_count": support_vertex_count,
        "support_normal_spread_degrees": support_normal_spread,
        "support_contamination_score": support_contamination_score,
        "new_vertex_count": int(len(new_vertex_ids)),
        "collar_movement": _movement_stats(collar_vertex_ids),
        "interior_movement": _movement_stats(interior_vertex_ids),
        "center_movement": _movement_stats((center_vertex_id,)),
        "all_generated_movement": _movement_stats(new_vertex_ids),
        "surface_projection_displacement_mean": (
            float(np.mean(surface_projection_deltas)) if surface_projection_deltas else 0.0
        ),
        "surface_projection_displacement_max": (
            float(np.max(surface_projection_deltas)) if surface_projection_deltas else 0.0
        ),
        "surface_projection_sample_count": int(len(surface_projection_deltas)),
        "relaxation_accepted_updates": int(relaxation_metadata.get("accepted_updates", 0)),
        "relaxation_rejected_updates": int(relaxation_metadata.get("rejected_updates", 0)),
    }

    return {
        "operation": "surface_uvdelaunay_sealed_dense_relaxed_preview",
        "preview_mesh": preview,
        "rings": int(rings),
        "seed_backend": "surface_uvdelaunay_sealed_dense_seed",
        "backend": "surface_uvdelaunay_sealed_dense_relaxed",
        "surface_guidance": "normal_compatible_mls_quadratic",
        "collar_mode": "quad_strip_multiring_inner_fill",
        "collar_fraction": float(collar_fraction),
        "dense_inner_rings": int(dense_inner_rings),
        "boundary_vertex_ids": boundary_ids,
        "ring_vertex_ids": seed["ring_vertex_ids"],
        "inner_vertex_ids": tuple(int(v) for v in seed["inner_vertex_ids"]),
        "center_vertex_id": int(seed["center_vertex_id"]),
        "new_vertex_ids": new_vertex_ids,
        "new_face_ids": new_face_ids,
        "collar_face_ids": tuple(int(v) for v in seed["collar_face_ids"]),
        "inner_fill_face_ids": tuple(int(v) for v in seed["inner_fill_face_ids"]),
        "support_context": support_context.to_dict(),
        "mls_projector": projector.to_dict(),
        "topology_before": topology_before.to_dict(),
        "topology_after": topology_after.to_dict(),
        "relaxation": relaxation_metadata,
        "dynamic_diagnostics": dynamic_diagnostics,
        "relaxation_iterations": int(relaxation_iterations),
        "relaxation_strength": float(relaxation_strength),
        "relaxation_surface_weight": float(surface_weight),
        "support_rings": int(support_rings),
        "smooth_dot_threshold": float(smooth_dot_threshold),
        "relaxed_max_displacement": float(np.max(displacement)) if len(displacement) else 0.0,
        "relaxed_mean_displacement": float(np.mean(displacement)) if len(displacement) else 0.0,
        "notes": (
            "Preview-only dense seam-sealed local surface patch.",
            "Boundary seam is sealed by a forced quad collar before relaxation.",
            "Interior uses multiple dense rings instead of a large center fan.",
            "Surface target came from normal-compatible support faces and MLS/quadratic projection.",
            "Boundary vertices were held fixed.",
            "Only generated patch vertices were relaxed.",
            "No source mesh mutation or project/history write occurred.",
        ),
    }

