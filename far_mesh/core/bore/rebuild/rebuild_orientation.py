"""Triangle orientation helpers for BoreTool rebuild.

This module is intentionally non-mutating.  It may compare generated triangle
winding against deleted source patch normals and return an updated in-memory
plan object, but it must not authorize deletion, apply geometry to a mesh,
validate final topology, mutate project state, or emit RebuildResult.
"""

from __future__ import annotations

import numpy as np

from .rebuild_geometry import unit_normal, unit_vector

REBUILD_ORIENTATION_HELPER_EXTRACTION_CHECKPOINT_V176R = (
    "v176r_rebuild_triangle_orientation_helpers_extracted_no_behavior_change"
)

REBUILD_ORIENTATION_HELPER_NON_MUTATION_CONTRACT_V176R = (
    "orientation_helpers_may_flip_generated_triangle_winding_in_plan_data_only_rebuild_py_remains_mutation_authority"
)

def source_patch_face_references_v176r(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    refs: list[tuple[np.ndarray, np.ndarray]] = []
    for fid in tuple(face_ids or ()):
        if int(fid) < 0 or int(fid) >= len(source_faces):
            continue
        tri = np.asarray(source_faces[int(fid), :3], dtype=np.int64)
        if any(int(v) < 0 or int(v) >= len(vertices) for v in tri):
            continue
        pts = vertices[tri, :3]
        normal = unit_normal(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        if normal is None:
            continue
        refs.append((np.mean(pts, axis=0), normal))
    return tuple(refs)



def orient_triangles_to_source_role_normal_v176r(
    *,
    vertices: np.ndarray,
    generated_vertices: np.ndarray,
    triangles: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    fallback_normal: np.ndarray,
) -> np.ndarray:
    """Orient a generated role surface by the averaged source role normal."""

    refs = source_patch_face_references_v176r(vertices=vertices, source_faces=source_faces, face_ids=face_ids)
    normals = [np.asarray(item[1], dtype=float).reshape(3) for item in refs]
    role_normal = unit_normal(np.sum(np.asarray(normals, dtype=float).reshape((-1, 3)), axis=0)) if normals else None
    if role_normal is None:
        role_normal = unit_vector(fallback_normal)

    output_vertices = np.asarray(vertices, dtype=float).copy()
    gen = np.asarray(generated_vertices, dtype=float).reshape((-1, 3))
    if gen.size:
        output_vertices = np.vstack([output_vertices, gen])
    tris = np.asarray(triangles, dtype=np.int64).reshape((-1, 3)).copy()
    for tri_index, tri in enumerate(tris):
        if any(int(v) < 0 or int(v) >= len(output_vertices) for v in tri):
            continue
        pts = output_vertices[np.asarray(tri, dtype=np.int64), :3]
        normal = unit_normal(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        if normal is not None and float(np.dot(normal, role_normal)) < 0.0:
            tris[tri_index] = np.asarray((int(tri[0]), int(tri[2]), int(tri[1])), dtype=np.int64)
    return tris



def orient_pocket_wall_triangles_by_radial_role_v176r(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    plan: object,
) -> object:
    """Orient pocket wall triangles by the wall role's radial normal sign.

    Nearest-face orientation can preserve source-mesh normal noise.  For the
    circular-pocket side wall, the stronger role rule is radial: generated wall
    normals should have the same inward/outward sign relative to the pocket axis
    as the owned source wall faces.
    """

    refs = source_patch_face_references_v176r(vertices=vertices, source_faces=source_faces, face_ids=face_ids)
    axis_vec = unit_vector(plan.axis)
    base = 0.5 * (np.asarray(plan.center0, dtype=float).reshape(3) + np.asarray(plan.center1, dtype=float).reshape(3))

    def radial_unit(point: np.ndarray) -> np.ndarray | None:
        rel = np.asarray(point, dtype=float).reshape(3) - base
        axial = float(np.dot(rel, axis_vec))
        radial = rel - axial * axis_vec
        length = float(np.linalg.norm(radial))
        if not np.isfinite(length) or length <= 1.0e-12:
            return None
        return radial / length

    signs: list[float] = []
    for centroid, normal in refs:
        r = radial_unit(np.asarray(centroid, dtype=float).reshape(3))
        if r is None:
            continue
        dot = float(np.dot(np.asarray(normal, dtype=float).reshape(3), r))
        if np.isfinite(dot) and abs(dot) > 1.0e-6:
            signs.append(dot)
    if not signs:
        return plan
    role_sign = 1.0 if float(np.median(np.asarray(signs, dtype=float))) >= 0.0 else -1.0

    output_vertices = np.asarray(vertices, dtype=float).copy()
    gen = np.asarray(plan.generated_vertices, dtype=float).reshape((-1, 3))
    if gen.size:
        output_vertices = np.vstack([output_vertices, gen])
    tris = np.asarray(plan.triangles, dtype=np.int64).reshape((-1, 3)).copy()
    flips = 0
    checked = 0
    for tri_index, tri in enumerate(tris):
        if any(int(v) < 0 or int(v) >= len(output_vertices) for v in tri):
            continue
        pts = output_vertices[np.asarray(tri, dtype=np.int64), :3]
        normal = unit_normal(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        r = radial_unit(np.mean(pts, axis=0))
        if normal is None or r is None:
            continue
        checked += 1
        if float(np.dot(normal, r)) * role_sign < 0.0:
            tris[tri_index] = np.asarray((int(tri[0]), int(tri[2]), int(tri[1])), dtype=np.int64)
            flips += 1
    diag = dict(plan.diagnostics)
    diag.update({
        "pocket_wall_normal_orientation_policy_v100": "radial_role_sign_from_owned_source_wall_faces",
        "pocket_wall_radial_normal_role_sign": float(role_sign),
        "pocket_wall_radial_orientation_checked_triangle_count": int(checked),
        "pocket_wall_radial_orientation_flip_count": int(flips),
    })
    return type(plan)(
        generated_vertices=plan.generated_vertices,
        triangles=tris,
        logical_quads=plan.logical_quads,
        loop0=plan.loop0,
        loop1=plan.loop1,
        center0=plan.center0,
        center1=plan.center1,
        axis=plan.axis,
        diagnostics=diag,
    )



def orient_plan_triangles_to_source_patch_v176r(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    plan: object,
) -> object:
    """Orient generated triangles to match the normals of the removed patch.

    Watertight topology does not guarantee useful winding.  The viewport and
    downstream processors still need the replacement triangles to face the same
    way as the deleted surface.  This helper flips generated triangle winding
    whenever the triangle normal disagrees with the nearest original patch face
    normal.
    """

    references = source_patch_face_references_v176r(vertices=vertices, source_faces=source_faces, face_ids=face_ids)
    if not references:
        diag = dict(plan.diagnostics)
        diag.update({
            "normal_orientation_policy": "source_patch_reference_unavailable",
            "normal_flip_count": 0,
            "normal_alignment_median": 0.0,
            "normal_alignment_min": 0.0,
        })
        return type(plan)(
            generated_vertices=plan.generated_vertices,
            triangles=plan.triangles,
            logical_quads=plan.logical_quads,
            loop0=plan.loop0,
            loop1=plan.loop1,
            center0=plan.center0,
            center1=plan.center1,
            axis=plan.axis,
            diagnostics=diag,
        )

    ref_centroids = np.asarray([item[0] for item in references], dtype=float).reshape((-1, 3))
    ref_normals = np.asarray([item[1] for item in references], dtype=float).reshape((-1, 3))

    generated_vertices = np.asarray(plan.generated_vertices, dtype=float).reshape((-1, 3))
    output_vertices = np.asarray(vertices, dtype=float).copy()
    if generated_vertices.size:
        output_vertices = np.vstack([output_vertices, generated_vertices])

    triangles = np.asarray(plan.triangles, dtype=np.int64).reshape((-1, 3)).copy()
    flips = 0
    alignments: list[float] = []
    for tri_index, tri in enumerate(triangles):
        if any(int(v) < 0 or int(v) >= len(output_vertices) for v in tri):
            continue
        pts = output_vertices[np.asarray(tri, dtype=np.int64), :3]
        normal = unit_normal(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        if normal is None:
            continue
        centroid = np.mean(pts, axis=0)
        deltas = ref_centroids - centroid.reshape(1, 3)
        nearest = int(np.argmin(np.sum(deltas * deltas, axis=1)))
        dot = float(np.dot(normal, ref_normals[nearest]))
        if dot < 0.0:
            triangles[tri_index] = np.asarray((int(tri[0]), int(tri[2]), int(tri[1])), dtype=np.int64)
            dot = -dot
            flips += 1
        alignments.append(float(dot))

    diag = dict(plan.diagnostics)
    diag.update({
        "normal_orientation_policy": "match_nearest_deleted_patch_face_normal",
        "normal_flip_count": int(flips),
        "normal_alignment_count": int(len(alignments)),
        "normal_alignment_median": float(np.median(alignments)) if alignments else 0.0,
        "normal_alignment_min": float(min(alignments)) if alignments else 0.0,
    })
    return type(plan)(
        generated_vertices=plan.generated_vertices,
        triangles=triangles,
        logical_quads=plan.logical_quads,
        loop0=plan.loop0,
        loop1=plan.loop1,
        center0=plan.center0,
        center1=plan.center1,
        axis=plan.axis,
        diagnostics=diag,
    )

