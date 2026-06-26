"""BORE-specific frame and protected-loop helpers for BoreTool rebuild.

This module is intentionally non-mutating.  It may read accepted CandidateData
frame metadata and locate physical rim/protected loops for BORE rebuild
planning, but it must not authorize deletion, apply generated geometry to a
mesh, validate final topology, mutate project state, or emit RebuildResult.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import math

import numpy as np

from .rebuild_geometry import loop_center, orthonormal_basis, safe_float, unit_vector
from .rebuild_loops import order_closed_edge_loop_vertices_v176g as order_closed_edge_loop_vertices
from ..topology import build_edge_to_faces, edge_loop_components
from ..types import tuple_ints

EdgeKey = tuple[int, int]

REBUILD_BORE_FRAME_HELPER_EXTRACTION_CHECKPOINT_V176X = (
    "v176x_rebuild_bore_frame_and_protected_loop_helpers_extracted_no_behavior_change"
)

REBUILD_BORE_FRAME_HELPER_NON_MUTATION_CONTRACT_V176X = (
    "bore_frame_helpers_may_resolve_candidate_frame_loops_only_no_delete_authorization_no_trial_no_mutation"
)

REBUILD_BORE_SHAPE_AUTHORITY_EXTRACTION_CHECKPOINT_V176Y = (
    "v176y_rebuild_bore_shape_authority_helpers_extracted_no_behavior_change"
)

REBUILD_BORE_SHAPE_AUTHORITY_NON_MUTATION_CONTRACT_V176Y = (
    "bore_shape_authority_helpers_may_fit_owned_wall_shape_constraints_only_no_delete_authorization_no_trial_no_mutation"
)


def edge_key_v176x(value: object) -> EdgeKey:
    try:
        a, b = value  # type: ignore[misc]
        ia = int(a)
        ib = int(b)
    except Exception:
        return (0, 0)
    return (ia, ib) if ia <= ib else (ib, ia)


def two_opening_frame_from_candidate_metadata_v176x(candidate_metadata: Mapping[str, object]) -> dict[str, object] | None:
    """Extract the accepted two-opening BoreFrame carried by CandidateData.

    Recognition owns the measured frame.  Rebuild may use it only as bounded
    target evidence after CandidateData is already accepted/rebuildable.
    """

    meta = dict(candidate_metadata or {})
    diag = meta.get("diagnostics")
    if not isinstance(diag, Mapping):
        diag = {}
    # v1.6.6: prefer the operational owned-bore frame produced after
    # BoreWallOwnership.  The original measured two-opening frame may have used
    # an internal damaged ring as provisional opposite evidence.
    frame = diag.get("owned_bore_frame") if isinstance(diag, Mapping) else None
    if not isinstance(frame, Mapping):
        frame = meta.get("owned_bore_frame")
    if not isinstance(frame, Mapping):
        frame = diag.get("two_opening_bore_frame") if isinstance(diag, Mapping) else None
    if not isinstance(frame, Mapping):
        frame = meta.get("two_opening_bore_frame")
    if not isinstance(frame, Mapping):
        return None
    try:
        opening_center = np.asarray(frame.get("opening_center", ()), dtype=float).reshape(3)
        opposite_center = np.asarray(frame.get("opposite_center", ()), dtype=float).reshape(3)
    except Exception:
        return None
    delta = opposite_center - opening_center
    depth = float(np.linalg.norm(delta))
    axis = unit_vector(delta if np.isfinite(depth) and depth > 1.0e-9 else frame.get("axis", meta.get("primitive_axis", (0.0, 0.0, 1.0))))
    explicit_depth = safe_float(frame.get("depth", meta.get("owned_bore_frame_depth", meta.get("depth", 0.0))), 0.0)
    if explicit_depth > max(depth, 0.0) + max(1.0e-6, 0.005 * max(depth, 1.0)):
        depth = float(explicit_depth)
        opposite_center = opening_center + axis.reshape(3) * float(depth)
    if not np.isfinite(depth) or depth <= 1.0e-9:
        depth = float(explicit_depth)
    radius = safe_float(frame.get("radius", meta.get("radius", meta.get("primitive_radius", 0.0))), 0.0)
    if radius <= 1.0e-9 or depth <= 1.0e-9:
        return None
    return {
        "opening_center": opening_center,
        "opposite_center": opposite_center,
        "axis": axis,
        "radius": float(radius),
        "depth": float(depth),
        "frame_source": str(frame.get("status", meta.get("bore_frame_depth_source", "candidate_owned_bore_frame"))),
    }

def protected_loop_pair_from_candidate_frame_v176x(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    frame: Mapping[str, object],
    preferred_pool: Iterable[int],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Find both physical rim loops from the accepted candidate BoreFrame."""

    loop_a = edge_loop_near_frame_opening_v176x(
        vertices=vertices,
        source_faces=source_faces,
        center=np.asarray(frame.get("opening_center"), dtype=float).reshape(3),
        axis=unit_vector(frame.get("axis", (0.0, 0.0, 1.0))),
        radius=safe_float(frame.get("radius", 0.0), 0.0),
        preferred_pool=preferred_pool,
    )
    loop_b = edge_loop_near_frame_opening_v176x(
        vertices=vertices,
        source_faces=source_faces,
        center=np.asarray(frame.get("opposite_center"), dtype=float).reshape(3),
        axis=unit_vector(frame.get("axis", (0.0, 0.0, 1.0))),
        radius=safe_float(frame.get("radius", 0.0), 0.0),
        preferred_pool=preferred_pool,
    )
    if loop_a is None or loop_b is None:
        return None
    if set(loop_a) == set(loop_b):
        return None
    return (loop_a, loop_b)

def edge_loop_near_frame_opening_v176x(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    preferred_pool: Iterable[int],
) -> tuple[int, ...] | None:
    """Measure a rim loop near one BoreFrame opening from mesh edges."""

    radius = float(radius)
    if radius <= 1.0e-9:
        return None
    faces_arr = np.asarray(source_faces, dtype=np.int64)[:, :3]
    edge_to_faces = build_edge_to_faces(faces_arr)
    pool = {int(fid) for fid in tuple(preferred_pool or ()) if 0 <= int(fid) < len(faces_arr)}
    axis_vec = unit_vector(axis)
    edge_lengths: list[float] = []
    for edge in edge_to_faces.keys():
        a, b = int(edge[0]), int(edge[1])
        if 0 <= a < len(vertices) and 0 <= b < len(vertices):
            length = float(np.linalg.norm(vertices[a, :3] - vertices[b, :3]))
            if np.isfinite(length) and length > 1.0e-12:
                edge_lengths.append(length)
    edge_scale = float(np.median(edge_lengths)) if edge_lengths else 1.0
    plane_tol = max(0.085 * radius, 4.5 * edge_scale, 0.30)
    radial_tol = max(0.105 * radius, 4.5 * edge_scale, 0.35)

    selected_edges: set[EdgeKey] = set()
    for edge, adjacent in edge_to_faces.items():
        if pool and not any(int(fid) in pool for fid in tuple(adjacent or ())):
            continue
        a, b = int(edge[0]), int(edge[1])
        if not (0 <= a < len(vertices) and 0 <= b < len(vertices)):
            continue
        pa = vertices[a, :3]
        pb = vertices[b, :3]
        mid = 0.5 * (pa + pb)
        rel = mid - center.reshape(3)
        axial = float(np.dot(rel, axis_vec))
        radial_vec = rel - axial * axis_vec
        radial = float(np.linalg.norm(radial_vec))
        if not np.isfinite(axial) or not np.isfinite(radial):
            continue
        if abs(axial) > plane_tol or abs(radial - radius) > radial_tol:
            continue
        tangent = pb - pa
        tlen = float(np.linalg.norm(tangent))
        if tlen <= 1.0e-12:
            continue
        tangent = tangent / tlen
        radial_unit = radial_vec / max(radial, 1.0e-12)
        radial_tangent = abs(float(np.dot(tangent, radial_unit))) if radial > 1.0e-12 else 0.0
        axial_tangent = abs(float(np.dot(tangent, axis_vec)))
        if radial_tangent > 0.92 and axial_tangent < 0.35:
            continue
        selected_edges.add(edge_key_v176x(edge))

    if len(selected_edges) < 6:
        return None
    best_loop: tuple[int, ...] | None = None
    best_score = -1.0
    for comp in edge_loop_components(selected_edges):
        ordered = order_closed_edge_loop_vertices({edge_key_v176x(edge) for edge in tuple(comp or ())})
        if not bool(ordered.get("closed", False)):
            continue
        verts = tuple(int(v) for v in tuple(ordered.get("vertices", ()) or ()))
        if len(verts) < 6:
            continue
        pts = vertices[np.asarray(verts, dtype=np.int64), :3]
        rel = pts - center.reshape(1, 3)
        axial_values = rel @ axis_vec.reshape(3)
        radial_vecs = rel - axial_values.reshape(-1, 1) * axis_vec.reshape(1, 3)
        radii = np.linalg.norm(radial_vecs, axis=1)
        radial_mad = float(np.median(np.abs(radii - radius))) if radii.size else 999999.0
        plane_mad = float(np.median(np.abs(axial_values))) if axial_values.size else 999999.0
        score = float(len(verts)) - 4.0 * radial_mad - 2.0 * plane_mad
        if score > best_score:
            best_score = score
            best_loop = verts
    return best_loop

def full_depth_bore_wall_target_faces_from_candidate_frame_v176x(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    frame: Mapping[str, object],
    face_pool: Iterable[int],
    base_face_ids: Iterable[int],
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Build a bounded full-depth wall delete target from CandidateData frame.

    This is used only for accepted damaged BoreWallOwnership.  It does not let
    RegionData become ownership; the candidate's two-opening frame and wall role
    define the target, while RegionData merely bounds the search pool.
    """

    faces_arr = np.asarray(source_faces, dtype=np.int64)[:, :3]
    pool = {int(fid) for fid in tuple(face_pool or ()) if 0 <= int(fid) < len(faces_arr)}
    base = {int(fid) for fid in tuple(base_face_ids or ()) if 0 <= int(fid) < len(faces_arr)}
    if not pool:
        pool = set(range(len(faces_arr)))
    opening_center = np.asarray(frame.get("opening_center"), dtype=float).reshape(3)
    opposite_center = np.asarray(frame.get("opposite_center"), dtype=float).reshape(3)
    axis_vec = unit_vector(frame.get("axis", opposite_center - opening_center), fallback=opposite_center - opening_center)
    radius = safe_float(frame.get("radius", 0.0), 0.0)
    depth = float(np.linalg.norm(opposite_center - opening_center))
    if radius <= 1.0e-9 or depth <= 1.0e-9:
        return tuple(sorted(base)), {"used": False, "reason": "invalid_candidate_frame"}

    tri = vertices[faces_arr[:, :3], :3]
    centroids = tri.mean(axis=1)
    raw_normals = np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :])
    nlen = np.linalg.norm(raw_normals, axis=1)
    normals = np.zeros_like(raw_normals)
    okn = np.isfinite(nlen) & (nlen > 1.0e-12)
    normals[okn] = raw_normals[okn] / nlen[okn].reshape(-1, 1)

    rel_cent = centroids - opening_center.reshape(1, 3)
    axial_cent = rel_cent @ axis_vec.reshape(3)
    radial_vec_cent = rel_cent - axial_cent.reshape(-1, 1) * axis_vec.reshape(1, 3)
    radial_cent = np.linalg.norm(radial_vec_cent, axis=1)
    radial_dir = np.zeros_like(radial_vec_cent)
    okr = radial_cent > 1.0e-12
    radial_dir[okr] = radial_vec_cent[okr] / radial_cent[okr].reshape(-1, 1)
    normal_axis_abs = np.abs(normals @ axis_vec.reshape(3))
    radial_normal_alignment = np.abs(np.sum(normals * radial_dir, axis=1))

    tri_rel = tri - opening_center.reshape(1, 1, 3)
    tri_ax = tri_rel @ axis_vec.reshape(3)
    face_ax_min = np.nanmin(tri_ax, axis=1)
    face_ax_max = np.nanmax(tri_ax, axis=1)

    edge_lengths: list[float] = []
    for fid in list(base)[:3000]:
        pts = tri[int(fid)]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            length = float(np.linalg.norm(pts[a] - pts[b]))
            if np.isfinite(length) and length > 1.0e-12:
                edge_lengths.append(length)
    edge_scale = float(np.median(edge_lengths)) if edge_lengths else 1.0
    axial_tol = max(0.045 * depth, 0.10 * radius, 4.0 * edge_scale, 0.35)
    radial_tol = max(0.18 * radius, 6.0 * edge_scale, 0.55)

    target: set[int] = set(base)
    for fid in pool:
        if fid < 0 or fid >= len(faces_arr):
            continue
        if not (np.isfinite(face_ax_min[fid]) and np.isfinite(face_ax_max[fid]) and np.isfinite(radial_cent[fid])):
            continue
        between = bool(face_ax_max[fid] >= -axial_tol and face_ax_min[fid] <= depth + axial_tol)
        near_radius = bool(abs(float(radial_cent[fid]) - radius) <= radial_tol)
        sidewall_normal = bool(float(normal_axis_abs[fid]) <= 0.75 and float(radial_normal_alignment[fid]) >= 0.20)
        if between and near_radius and sidewall_normal:
            target.add(int(fid))

    return tuple(sorted(target)), {
        "used": True,
        "source": "candidate_two_opening_frame_full_depth_wall_target",
        "base_face_count": int(len(base)),
        "pool_face_count": int(len(pool)),
        "target_face_count": int(len(target)),
        "added_face_count": int(max(0, len(target) - len(base))),
        "radius": float(radius),
        "depth": float(depth),
        "frame_source": str(frame.get("frame_source", frame.get("status", "candidate_owned_bore_frame"))),
        "radial_tolerance": float(radial_tol),
        "axial_tolerance": float(axial_tol),
    }

def protected_loop_pair_from_selection_v176x(
    *,
    vertices: np.ndarray,
    region_data: object,
    axis: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    loops: list[tuple[int, ...]] = []
    primary = tuple(int(v) for v in tuple(getattr(region_data, "loop_vertices", ()) or ()))
    if len(primary) >= 3:
        loops.append(primary)

    opposite_edges = {edge_key_v176x(edge) for edge in tuple(getattr(region_data, "derived_opposite_rim_edge_ids", ()) or ())}
    if opposite_edges:
        ordered = order_closed_edge_loop_vertices(opposite_edges)
        if bool(ordered.get("closed", False)):
            opp = tuple(int(v) for v in tuple(ordered.get("vertices", ()) or ()))
            if len(opp) >= 3:
                loops.append(opp)

    for edge_loop in tuple(getattr(region_data, "derived_boundary_loops", ()) or ()):
        edges = {edge_key_v176x(edge) for edge in tuple(edge_loop or ())}
        ordered = order_closed_edge_loop_vertices(edges)
        if bool(ordered.get("closed", False)):
            loop = tuple(int(v) for v in tuple(ordered.get("vertices", ()) or ()))
            if len(loop) >= 3 and all(set(loop) != set(existing) for existing in loops):
                loops.append(loop)

    if len(loops) < 2:
        return None

    axis = unit_vector(axis)
    best_pair: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    best_sep = -1.0
    for i, loop_a in enumerate(loops):
        ca = loop_center(vertices, loop_a)
        ta = float(np.dot(ca, axis))
        for loop_b in loops[i + 1:]:
            cb = loop_center(vertices, loop_b)
            tb = float(np.dot(cb, axis))
            sep = abs(tb - ta)
            if sep > best_sep:
                best_sep = sep
                best_pair = (loop_a, loop_b)
    return best_pair

# -----------------------------------------------------------------------------
# BORE owned-wall shape authority helpers (v176y)
# -----------------------------------------------------------------------------

def owned_bore_wall_cylinder_shape_authority_v173x_v176y(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray | None,
    face_ids: Iterable[int] | None,
    axis: np.ndarray,
    center0: np.ndarray,
    center1: np.ndarray,
) -> dict[str, object]:
    """Fit rebuild shape authority from owned BORE_WALL face evidence.

    This is rebuild-side constraint plumbing, not feature recognition.  Recognition
    already emitted CandidateData and the delete patch.  v173x prevents rebuild
    from shaping a BORE from provisional rim/opposite/terminal descriptors when
    the owned wall faces themselves provide the physical cylindrical wall model.

    The fit uses the owned wall face centroids and face normals under the measured
    bore axis.  Boundary loops remain locked seam constraints; they are not used
    as cylinder-shape authority except as a fallback when the owned-wall fit is
    not reliable.
    """

    ids = tuple_ints(face_ids or ())
    if source_faces is None or not ids:
        return {
            "valid": False,
            "reason": "missing_source_faces_or_owned_face_ids",
            "used": False,
        }
    try:
        faces_arr = np.asarray(source_faces, dtype=np.int64)[:, :3]
        verts = np.asarray(vertices, dtype=float)[:, :3]
    except Exception:
        return {"valid": False, "reason": "invalid_source_arrays", "used": False}
    if faces_arr.size == 0 or verts.size == 0:
        return {"valid": False, "reason": "empty_source_arrays", "used": False}

    axis_vec = unit_vector(axis, fallback=np.asarray(center1, dtype=float).reshape(3) - np.asarray(center0, dtype=float).reshape(3))
    c0 = np.asarray(center0, dtype=float).reshape(3)
    c1 = np.asarray(center1, dtype=float).reshape(3)
    origin = 0.5 * (c0 + c1)
    basis_u, basis_v = orthonormal_basis(axis_vec)

    rows: list[list[float]] = []
    rhs: list[float] = []
    samples: list[tuple[float, float, float, float, float, int]] = []
    considered = 0
    for fid in ids:
        if int(fid) < 0 or int(fid) >= len(faces_arr):
            continue
        tri_ids = faces_arr[int(fid), :3]
        if np.any(tri_ids < 0) or np.any(tri_ids >= len(verts)):
            continue
        pts = verts[tri_ids, :3]
        centroid = np.mean(pts, axis=0)
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            continue
        normal = normal / norm
        n_proj = normal - float(np.dot(normal, axis_vec)) * axis_vec
        n_norm = float(np.linalg.norm(n_proj))
        if not np.isfinite(n_norm) or n_norm <= 1.0e-12:
            continue
        n_proj = n_proj / n_norm

        # Orient the radial normal consistently with the current boundary-based
        # centerline guess.  The sign of mesh triangle normals is not semantic;
        # the wall role is radial distance from the bore axis.
        axial_guess = float(np.dot(centroid - c0, axis_vec))
        center_guess = c0 + axial_guess * axis_vec
        radial_guess = centroid - center_guess
        radial_guess = radial_guess - float(np.dot(radial_guess, axis_vec)) * axis_vec
        if float(np.dot(n_proj, radial_guess)) < 0.0:
            n_proj = -n_proj

        n2 = np.array([float(np.dot(n_proj, basis_u)), float(np.dot(n_proj, basis_v))], dtype=float)
        n2_norm = float(np.linalg.norm(n2))
        if not np.isfinite(n2_norm) or n2_norm <= 1.0e-12:
            continue
        n2 = n2 / n2_norm
        p_rel = centroid - origin
        p2 = np.array([float(np.dot(p_rel, basis_u)), float(np.dot(p_rel, basis_v))], dtype=float)
        rows.append([-float(n2[0]), -float(n2[1]), 1.0])
        rhs.append(float(np.dot(n2, p2)))
        considered += 1

    if len(rows) < 12:
        return {
            "valid": False,
            "used": False,
            "reason": "too_few_wall_normal_samples",
            "owned_face_count": int(len(ids)),
            "sample_count": int(len(rows)),
        }

    A = np.asarray(rows, dtype=float)
    b = np.asarray(rhs, dtype=float)
    try:
        sol, residuals, rank, _s = np.linalg.lstsq(A, b, rcond=None)
    except Exception as exc:
        return {
            "valid": False,
            "used": False,
            "reason": "least_squares_failed",
            "error": str(exc),
            "owned_face_count": int(len(ids)),
            "sample_count": int(len(rows)),
        }
    if int(rank) < 3:
        return {
            "valid": False,
            "used": False,
            "reason": "rank_deficient_wall_cylinder_fit",
            "rank": int(rank),
            "owned_face_count": int(len(ids)),
            "sample_count": int(len(rows)),
        }

    center_uv = np.asarray(sol[:2], dtype=float).reshape(2)
    radius = float(sol[2])
    if not np.isfinite(radius) or radius <= 1.0e-9:
        return {
            "valid": False,
            "used": False,
            "reason": "invalid_fitted_wall_radius",
            "fitted_radius": float(radius) if np.isfinite(radius) else 0.0,
            "owned_face_count": int(len(ids)),
            "sample_count": int(len(rows)),
        }

    center_axis_point = origin + center_uv[0] * basis_u + center_uv[1] * basis_v
    # Recompute robust radial errors from centroids to the fitted centerline.
    radial_errors: list[float] = []
    radial_values: list[float] = []
    normal_axis_values: list[float] = []
    radial_align_values: list[float] = []
    axial_values: list[float] = []
    for fid in ids:
        if int(fid) < 0 or int(fid) >= len(faces_arr):
            continue
        tri_ids = faces_arr[int(fid), :3]
        if np.any(tri_ids < 0) or np.any(tri_ids >= len(verts)):
            continue
        pts = verts[tri_ids, :3]
        centroid = np.mean(pts, axis=0)
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            continue
        normal = normal / norm
        rel = centroid - center_axis_point
        axial_value = float(np.dot(rel, axis_vec))
        radial_vec = rel - axial_value * axis_vec
        radial_value = float(np.linalg.norm(radial_vec))
        if not np.isfinite(radial_value) or radial_value <= 1.0e-12:
            continue
        radial_unit = radial_vec / radial_value
        n_proj = normal - float(np.dot(normal, axis_vec)) * axis_vec
        n_norm = float(np.linalg.norm(n_proj))
        if np.isfinite(n_norm) and n_norm > 1.0e-12:
            radial_align_values.append(abs(float(np.dot(n_proj / n_norm, radial_unit))))
        normal_axis_values.append(abs(float(np.dot(normal, axis_vec))))
        radial_values.append(radial_value)
        radial_errors.append(abs(radial_value - radius))
        axial_values.append(axial_value)

    if not radial_values:
        return {
            "valid": False,
            "used": False,
            "reason": "no_radial_values_after_fit",
            "owned_face_count": int(len(ids)),
            "sample_count": int(len(rows)),
        }
    radial_arr = np.asarray(radial_values, dtype=float)
    error_arr = np.asarray(radial_errors, dtype=float)
    normal_axis_arr = np.asarray(normal_axis_values, dtype=float) if normal_axis_values else np.asarray([1.0], dtype=float)
    radial_align_arr = np.asarray(radial_align_values, dtype=float) if radial_align_values else np.asarray([0.0], dtype=float)
    axial_arr = np.asarray(axial_values, dtype=float) if axial_values else np.asarray([0.0], dtype=float)
    radius_median = float(np.median(radial_arr))
    radial_mad = float(np.median(np.abs(radial_arr - radius_median)))
    error_median = float(np.median(error_arr))
    error_p90 = float(np.percentile(error_arr, 90)) if error_arr.size else error_median
    normal_axis_median = float(np.median(normal_axis_arr))
    radial_align_median = float(np.median(radial_align_arr))
    rel_error = float(error_median / max(abs(radius), 1.0e-9))

    valid = bool(
        radius > 1.0e-9
        and len(radial_values) >= max(12, min(64, int(0.02 * len(ids))))
        and rel_error <= 0.08
        and normal_axis_median <= 0.35
        and radial_align_median >= 0.55
    )
    reason = "owned_bore_wall_cylinder_fit_valid" if valid else "owned_bore_wall_cylinder_fit_quality_rejected"
    axial0 = float(np.dot(c0 - center_axis_point, axis_vec))
    axial1 = float(np.dot(c1 - center_axis_point, axis_vec))
    shape_center0 = center_axis_point + axial0 * axis_vec
    shape_center1 = center_axis_point + axial1 * axis_vec
    return {
        "valid": bool(valid),
        "used": bool(valid),
        "reason": reason,
        "owned_face_count": int(len(ids)),
        "sample_count": int(len(radial_values)),
        "fitted_radius": float(radius),
        "median_radius_from_owned_faces": float(radius_median),
        "radial_mad": float(radial_mad),
        "radial_error_median": float(error_median),
        "radial_error_p90": float(error_p90),
        "radial_error_rel_median": float(rel_error),
        "normal_axis_abs_median": float(normal_axis_median),
        "radial_normal_alignment_median": float(radial_align_median),
        "axial_span": float(float(np.max(axial_arr)) - float(np.min(axial_arr))) if axial_arr.size else 0.0,
        "centerline_point": tuple(float(v) for v in center_axis_point.reshape(3)),
        "shape_center0": np.asarray(shape_center0, dtype=float).reshape(3),
        "shape_center1": np.asarray(shape_center1, dtype=float).reshape(3),
        "axis": np.asarray(axis_vec, dtype=float).reshape(3),
        "basis_u": np.asarray(basis_u, dtype=float).reshape(3),
        "basis_v": np.asarray(basis_v, dtype=float).reshape(3),
        "semantic_rule": "owned_bore_wall_faces_define_rebuild_shape_authority_not_rim_endpoint_or_terminal_context",
    }

def owned_bore_wall_cylinder_shape_authority_v173z_v176y(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray | None,
    face_ids: Iterable[int] | None,
    fallback_axis: np.ndarray,
    center0: np.ndarray,
    center1: np.ndarray,
) -> dict[str, object]:
    """Fit a complete BORE cylinder frame from owned wall faces.

    v173x still solved the wall cylinder under the incoming candidate/boundary
    axis.  That kept old RegionData/opposite-loop axis evidence in the rebuild
    shape frame and could make two axes fight: the radius/center came from the
    owned wall, but the ring placement still inherited the old axis/basis.

    v173z makes the owned BORE_WALL patch the shape authority as one atomic
    frame.  It derives:
        owned wall axis  -> PCA of owned wall centroids
        owned wall center -> circle fit in the plane normal to that axis
        owned wall radius -> same circle fit
        owned basis       -> same fitted axis and center

    This is still rebuild-side constraint plumbing.  It does not classify a
    BORE, does not subtract CHAMFER/POCKET faces, and does not change the delete
    patch.  It only decides how to place generated vertices after Recognition
    has already emitted accepted BORE CandidateData.
    """

    ids = tuple_ints(face_ids or ())
    if source_faces is None or not ids:
        return {"valid": False, "used": False, "reason": "missing_source_faces_or_owned_face_ids"}
    try:
        faces_arr = np.asarray(source_faces, dtype=np.int64)[:, :3]
        verts = np.asarray(vertices, dtype=float)[:, :3]
    except Exception:
        return {"valid": False, "used": False, "reason": "invalid_source_arrays"}
    centroids: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for fid in ids:
        fid_i = int(fid)
        if fid_i < 0 or fid_i >= len(faces_arr):
            continue
        tri_ids = faces_arr[fid_i, :3]
        if np.any(tri_ids < 0) or np.any(tri_ids >= len(verts)):
            continue
        pts = verts[tri_ids, :3]
        centroid = np.mean(pts, axis=0)
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            continue
        centroids.append(np.asarray(centroid, dtype=float).reshape(3))
        normals.append(np.asarray(normal / norm, dtype=float).reshape(3))
    if len(centroids) < 24:
        return {
            "valid": False,
            "used": False,
            "reason": "too_few_owned_wall_centroids_for_axis_fit",
            "owned_face_count": int(len(ids)),
            "sample_count": int(len(centroids)),
        }

    pts_arr = np.asarray(centroids, dtype=float).reshape((-1, 3))
    normals_arr = np.asarray(normals, dtype=float).reshape((-1, 3))
    mean = np.mean(pts_arr, axis=0)
    demeaned = pts_arr - mean.reshape(1, 3)
    try:
        cov = (demeaned.T @ demeaned) / max(float(len(pts_arr) - 1), 1.0)
        eigvals, eigvecs = np.linalg.eigh(cov)
    except Exception as exc:
        return {"valid": False, "used": False, "reason": "owned_wall_axis_pca_failed", "error": str(exc)}
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    axis_vec = unit_vector(eigvecs[:, 0], fallback=fallback_axis)
    fallback_vec = unit_vector(fallback_axis, fallback=np.asarray(center1, dtype=float).reshape(3) - np.asarray(center0, dtype=float).reshape(3))
    if float(np.dot(axis_vec, fallback_vec)) < 0.0:
        axis_vec = -axis_vec
    basis_u, basis_v = orthonormal_basis(axis_vec)

    # Fit a circle to the owned wall centroids after removing the fitted axial
    # coordinate.  This gives the centerline point in the normal plane and the
    # wall radius in one coordinate system.
    rel = pts_arr - mean.reshape(1, 3)
    x = rel @ basis_u
    y = rel @ basis_v
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, _resid, rank, _sing = np.linalg.lstsq(A, b, rcond=None)
    except Exception as exc:
        return {"valid": False, "used": False, "reason": "owned_wall_circle_fit_failed", "error": str(exc)}
    if int(rank) < 3:
        return {"valid": False, "used": False, "reason": "owned_wall_circle_fit_rank_deficient", "rank": int(rank)}
    cx, cy, c = float(sol[0]), float(sol[1]), float(sol[2])
    radius_sq = float(c + cx * cx + cy * cy)
    if not np.isfinite(radius_sq) or radius_sq <= 1.0e-12:
        return {"valid": False, "used": False, "reason": "owned_wall_circle_fit_invalid_radius"}
    radius = float(math.sqrt(radius_sq))
    centerline_point = mean + cx * basis_u + cy * basis_v

    axial_values = (pts_arr - centerline_point.reshape(1, 3)) @ axis_vec
    radial_vecs = pts_arr - centerline_point.reshape(1, 3) - axial_values.reshape(-1, 1) * axis_vec.reshape(1, 3)
    radial_values = np.linalg.norm(radial_vecs, axis=1)
    radial_errors = np.abs(radial_values - radius)
    radial_median = float(np.median(radial_values))
    radial_mad = float(np.median(np.abs(radial_values - radial_median)))
    radial_error_median = float(np.median(radial_errors))
    radial_error_p90 = float(np.percentile(radial_errors, 90)) if len(radial_errors) else radial_error_median
    rel_error = float(radial_error_median / max(abs(radius), 1.0e-9))

    radial_align_values: list[float] = []
    normal_axis_values: list[float] = []
    for normal, rv, rv_len in zip(normals_arr, radial_vecs, radial_values):
        if float(rv_len) <= 1.0e-12 or not np.isfinite(float(rv_len)):
            continue
        radial_unit = rv / float(rv_len)
        n_axis = abs(float(np.dot(normal, axis_vec)))
        n_proj = normal - float(np.dot(normal, axis_vec)) * axis_vec
        n_norm = float(np.linalg.norm(n_proj))
        normal_axis_values.append(n_axis)
        if n_norm > 1.0e-12 and np.isfinite(n_norm):
            radial_align_values.append(abs(float(np.dot(n_proj / n_norm, radial_unit))))
    normal_axis_median = float(np.median(np.asarray(normal_axis_values, dtype=float))) if normal_axis_values else 1.0
    radial_align_median = float(np.median(np.asarray(radial_align_values, dtype=float))) if radial_align_values else 0.0
    axial_span = float(float(np.max(axial_values)) - float(np.min(axial_values))) if len(axial_values) else 0.0

    c0 = np.asarray(center0, dtype=float).reshape(3)
    c1 = np.asarray(center1, dtype=float).reshape(3)
    axial0 = float(np.dot(c0 - centerline_point, axis_vec))
    axial1 = float(np.dot(c1 - centerline_point, axis_vec))
    shape_center0 = centerline_point + axial0 * axis_vec
    shape_center1 = centerline_point + axial1 * axis_vec

    eig0 = float(eigvals[0]) if len(eigvals) else 0.0
    eig1 = float(eigvals[1]) if len(eigvals) > 1 else 0.0
    axis_ratio = float(eig0 / max(eig1, 1.0e-12))
    # Do not require an extreme PCA ratio because a short/large bore can have a
    # less dominant axial eigenvalue.  The actual guard is the cylinder residual
    # plus wall-normal relationship to the fitted axis.
    valid = bool(
        radius > 1.0e-9
        and len(pts_arr) >= 24
        and axial_span > max(2.0 * radius, 1.0e-6)
        and rel_error <= 0.06
        and radial_mad / max(abs(radius), 1.0e-9) <= 0.08
        and normal_axis_median <= 0.20
        and radial_align_median >= 0.70
    )
    reason = "owned_wall_full_cylinder_frame_valid_v173z" if valid else "owned_wall_full_cylinder_frame_quality_rejected_v173z"
    return {
        "valid": bool(valid),
        "used": bool(valid),
        "reason": reason,
        "owned_face_count": int(len(ids)),
        "sample_count": int(len(pts_arr)),
        "axis": np.asarray(axis_vec, dtype=float).reshape(3),
        "basis_u": np.asarray(basis_u, dtype=float).reshape(3),
        "basis_v": np.asarray(basis_v, dtype=float).reshape(3),
        "centerline_point": tuple(float(v) for v in centerline_point.reshape(3)),
        "shape_center0": np.asarray(shape_center0, dtype=float).reshape(3),
        "shape_center1": np.asarray(shape_center1, dtype=float).reshape(3),
        "fitted_radius": float(radius),
        "median_radius_from_owned_faces": float(radial_median),
        "radial_mad": float(radial_mad),
        "radial_error_median": float(radial_error_median),
        "radial_error_p90": float(radial_error_p90),
        "radial_error_rel_median": float(rel_error),
        "normal_axis_abs_median": float(normal_axis_median),
        "radial_normal_alignment_median": float(radial_align_median),
        "axial_span": float(axial_span),
        "pca_eigenvalues": tuple(float(v) for v in eigvals.tolist()),
        "pca_axis_ratio": float(axis_ratio),
        "fallback_axis_dot": float(abs(np.dot(axis_vec, fallback_vec))),
        "semantic_rule": "owned_BORE_WALL_faces_define_axis_centerline_basis_and_radius_as_one_rebuild_shape_frame_v173z",
        "v173z_full_axis_authority": True,
    }

def shape_frame_loop_angles_v173z_v176y(
    *,
    vertices: np.ndarray,
    loop: tuple[int, ...],
    center: np.ndarray,
    axis: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> tuple[float, ...]:
    """Return the actual angular samples of a boundary loop in a shape frame."""
    angles: list[float] = []
    center_arr = np.asarray(center, dtype=float).reshape(3)
    axis_vec = unit_vector(axis)
    u = unit_vector(basis_u)
    v = unit_vector(basis_v)
    for vid in tuple(loop or ()): 
        if int(vid) < 0 or int(vid) >= len(vertices):
            continue
        rel = vertices[int(vid), :3] - center_arr
        radial = rel - float(np.dot(rel, axis_vec)) * axis_vec
        angle = float(np.arctan2(float(np.dot(radial, v)), float(np.dot(radial, u))))
        if angle < 0.0:
            angle += float(2.0 * math.pi)
        angles.append(angle)
    return tuple(float(a) for a in angles)

