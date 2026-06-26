"""Pure geometry helpers for BoreTool rebuild.

This module is intentionally non-mutating.  It may normalize vectors, build
axis frames, project points/rings, measure loop radii, and sample closed loops,
but it must not authorize deletion, interpret CandidateData, validate final
topology, or mutate the mesh.  ``rebuild.py`` remains the single rebuild
mutation authority.
"""

from __future__ import annotations

from typing import Iterable

import math

import numpy as np

Vector3 = tuple[float, float, float]

REBUILD_GEOMETRY_EXTRACTION_CHECKPOINT_V176D = (
    "v176d_rebuild_geometry_helper_extraction_no_behavior_change"
)

REBUILD_GEOMETRY_NON_MUTATION_CONTRACT_V176D = (
    "rebuild_geometry_may_compute_frames_radii_samples_only_rebuild_py_remains_mutation_authority"
)

def unit_vector(value: object, fallback: object = (0.0, 0.0, 1.0)) -> np.ndarray:
    try:
        vec = np.asarray(value, dtype=float).reshape(3)
    except Exception:
        vec = np.asarray(fallback, dtype=float).reshape(3)
    length = float(np.linalg.norm(vec))
    if np.isfinite(length) and length > 1.0e-12:
        return vec / length
    fb = np.asarray(fallback, dtype=float).reshape(3)
    fb_len = float(np.linalg.norm(fb))
    if np.isfinite(fb_len) and fb_len > 1.0e-12:
        return fb / fb_len
    return np.array([0.0, 0.0, 1.0], dtype=float)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def to_vector3(value: object) -> Vector3:
    try:
        arr = np.asarray(value, dtype=float).reshape(3)
        return (float(arr[0]), float(arr[1]), float(arr[2]))
    except Exception:
        return (0.0, 0.0, 1.0)


def unit_normal(value: object) -> np.ndarray | None:
    try:
        vec = np.asarray(value, dtype=float).reshape(3)
    except Exception:
        return None
    length = float(np.linalg.norm(vec))
    if not np.isfinite(length) or length <= 1.0e-12:
        return None
    return vec / length


def orthonormal_basis(axis: object) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable orthonormal basis perpendicular to ``axis``.

    v173x introduced owned-wall cylinder shape fitting that projects wall
    centroids and normals into the plane normal to the BORE axis.  The original
    file already had a latent call to ``orthonormal_basis`` inside
    ``loop_angle_basis`` but no helper definition.  Most old rebuild paths did
    not hit that fallback; the v173x shape-authority audit calls it directly.

    This helper is pure geometry plumbing.  It does not choose feature identity,
    does not choose CandidateData, and does not change delete-patch authority.
    """

    axis_vec = unit_vector(axis, fallback=(0.0, 0.0, 1.0))
    candidates = (
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    )
    ref = min(candidates, key=lambda item: abs(float(np.dot(item, axis_vec))))
    basis_u = ref - float(np.dot(ref, axis_vec)) * axis_vec
    basis_u = unit_vector(basis_u, fallback=(1.0, 0.0, 0.0))
    if abs(float(np.dot(basis_u, axis_vec))) > 0.95:
        basis_u = np.array([0.0, 1.0, 0.0], dtype=float)
        basis_u = basis_u - float(np.dot(basis_u, axis_vec)) * axis_vec
        basis_u = unit_vector(basis_u, fallback=(0.0, 1.0, 0.0))
    basis_v = unit_vector(np.cross(axis_vec, basis_u), fallback=(0.0, 1.0, 0.0))
    basis_u = unit_vector(np.cross(basis_v, axis_vec), fallback=basis_u)
    return np.asarray(basis_u, dtype=float).reshape(3), np.asarray(basis_v, dtype=float).reshape(3)


def project_radial(points: np.ndarray, center: np.ndarray, axis: np.ndarray) -> np.ndarray:
    rel = np.asarray(points, dtype=float)[:, :3] - np.asarray(center, dtype=float).reshape(1, 3)
    axis = unit_vector(axis)
    axial = rel @ axis
    radial = rel - axial.reshape(-1, 1) * axis.reshape(1, 3)
    lengths = np.linalg.norm(radial, axis=1)
    out = np.zeros_like(radial)
    valid = lengths > 1.0e-12
    out[valid] = radial[valid] / lengths[valid].reshape(-1, 1)
    return out


def loop_center(vertices: np.ndarray, loop: Iterable[int]) -> np.ndarray:
    ids = [int(v) for v in tuple(loop or ()) if 0 <= int(v) < len(vertices)]
    if not ids:
        return np.zeros(3, dtype=float)
    return np.mean(vertices[np.asarray(ids, dtype=np.int64), :3], axis=0)


def minimum_loop_pair_separation(vertices: np.ndarray, loop0: tuple[int, ...], loop1: tuple[int, ...]) -> float:
    median_edge = max(median_loop_edge_length(vertices, loop0), median_loop_edge_length(vertices, loop1), 1.0e-12)
    return max(1.0e-6, 0.02 * median_edge)


def median_loop_edge_length(vertices: np.ndarray, loop: tuple[int, ...]) -> float:
    if len(loop) < 2:
        return 0.0
    lengths: list[float] = []
    for i, a in enumerate(loop):
        b = loop[(i + 1) % len(loop)]
        ia, ib = int(a), int(b)
        if 0 <= ia < len(vertices) and 0 <= ib < len(vertices):
            length = float(np.linalg.norm(vertices[ia, :3] - vertices[ib, :3]))
            if np.isfinite(length) and length > 0.0:
                lengths.append(length)
    return float(np.median(lengths)) if lengths else 0.0


def loop_median_radius(*, vertices: np.ndarray, loop: tuple[int, ...], center: np.ndarray, axis: np.ndarray) -> float:
    ids = np.asarray(tuple(int(v) for v in loop if 0 <= int(v) < len(vertices)), dtype=np.int64)
    if ids.size == 0:
        return 0.0
    pts = vertices[ids, :3]
    axis_vec = unit_vector(axis)
    rel = pts - np.asarray(center, dtype=float).reshape(1, 3)
    axial = rel @ axis_vec.reshape(3)
    radial = rel - axial.reshape(-1, 1) * axis_vec.reshape(1, 3)
    radii = np.linalg.norm(radial, axis=1)
    radii = radii[np.isfinite(radii) & (radii > 1.0e-12)]
    return float(np.median(radii)) if radii.size else 0.0


def loop_radius_spread_ratio(*, vertices: np.ndarray, loop: tuple[int, ...], center: np.ndarray, axis: np.ndarray) -> float:
    ids = np.asarray(tuple(int(v) for v in loop if 0 <= int(v) < len(vertices)), dtype=np.int64)
    if ids.size == 0:
        return 0.0
    pts = vertices[ids, :3]
    axis_vec = unit_vector(axis)
    rel = pts - np.asarray(center, dtype=float).reshape(1, 3)
    axial = rel @ axis_vec.reshape(3)
    radial = rel - axial.reshape(-1, 1) * axis_vec.reshape(1, 3)
    radii = np.linalg.norm(radial, axis=1)
    radii = radii[np.isfinite(radii) & (radii > 1.0e-12)]
    if radii.size == 0:
        return 0.0
    median = float(np.median(radii))
    if median <= 1.0e-12 or not np.isfinite(median):
        return 0.0
    return float((float(np.max(radii)) - float(np.min(radii))) / median)


def loop_angle_basis(*, vertices: np.ndarray, loop: tuple[int, ...], center: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis_vec = unit_vector(axis)
    ids = tuple(int(v) for v in loop if 0 <= int(v) < len(vertices))
    center_arr = np.asarray(center, dtype=float).reshape(3)
    basis_u: np.ndarray | None = None
    for vid in ids:
        rel = vertices[int(vid), :3] - center_arr
        radial = rel - float(np.dot(rel, axis_vec)) * axis_vec
        norm = float(np.linalg.norm(radial))
        if norm > 1.0e-12 and np.isfinite(norm):
            basis_u = radial / norm
            break
    if basis_u is None:
        basis_u, _ = orthonormal_basis(axis_vec)
    basis_v = unit_vector(np.cross(axis_vec, basis_u), fallback=(0.0, 1.0, 0.0))
    basis_u = unit_vector(np.cross(basis_v, axis_vec), fallback=basis_u)
    return np.asarray(basis_u, dtype=float).reshape(3), np.asarray(basis_v, dtype=float).reshape(3)


def sample_closed_loop_points(points: np.ndarray, count: int) -> np.ndarray:
    """Sample ``count`` points around a closed polyline by fractional index."""

    pts = np.asarray(points, dtype=float).reshape((-1, 3))
    n = int(len(pts))
    count = int(count)
    if n <= 0 or count <= 0:
        return np.zeros((0, 3), dtype=float)
    if n == count:
        return pts.copy()
    out = np.zeros((count, 3), dtype=float)
    for k in range(count):
        pos = float(k) * float(n) / float(count)
        i0 = int(math.floor(pos)) % n
        i1 = (i0 + 1) % n
        frac = float(pos - math.floor(pos))
        out[k, :] = (1.0 - frac) * pts[i0, :] + frac * pts[i1, :]
    return out


def smoothstep_v165(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return float(x * x * (3.0 - 2.0 * x))


def boundary_adapter_weight_v165(t: float, fraction: float) -> float:
    """Blend weight from locked-boundary shape to semantic target shape.

    The boundary itself remains untouched.  Near each protected loop the first
    generated rings follow the old boundary-to-boundary interpolation; only after
    the adapter band do rings fully follow the measured semantic primitive.  This
    preserves manifold seams while still making the middle of the feature a true
    semantic remesh.
    """
    t = max(0.0, min(1.0, float(t)))
    fraction = max(1.0e-6, min(0.49, float(fraction)))
    return min(smoothstep_v165(t / fraction), smoothstep_v165((1.0 - t) / fraction))


def endpoint_safe_semantic_weight_v166(t: float) -> float:
    """Symmetric BORE blend: old boundary shape at both locked endpoints.

    Old rebuild generated rings purely by linear interpolation between the two
    aligned boundary loops; that did not create lips.  v160+ direct projection
    improved the cylinder body but could create endpoint collars when the
    protected opposite loop contained old lip/chamfer offset.  v166 uses the old
    path as the baseline everywhere and applies semantic projection with a
    smooth center-only weight: 0 at both locked boundaries and 1 at the middle.
    """
    t = max(0.0, min(1.0, float(t)))
    return float(math.sin(math.pi * t) ** 2)


__all__ = [
    "REBUILD_GEOMETRY_EXTRACTION_CHECKPOINT_V176D",
    "REBUILD_GEOMETRY_NON_MUTATION_CONTRACT_V176D",
    "boundary_adapter_weight_v165",
    "endpoint_safe_semantic_weight_v166",
    "loop_angle_basis",
    "loop_center",
    "loop_median_radius",
    "loop_radius_spread_ratio",
    "median_loop_edge_length",
    "minimum_loop_pair_separation",
    "orthonormal_basis",
    "project_radial",
    "safe_float",
    "sample_closed_loop_points",
    "smoothstep_v165",
    "to_vector3",
    "unit_normal",
    "unit_vector",
]
