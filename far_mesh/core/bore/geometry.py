"""Mesh-native Bore geometry helpers.

This module is the FAR MESH mesh-native geometry engine for the Bore evidence
model.  Measurements and fits are evidence for recognition, not final rebuild
placement instructions.  The final rebuild
uses measured patch boundaries in ``rebuild.py``.  This module only provides measurement primitives. Feature classification,
local layer naming, and promotion live outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import math
import numpy as np

Vector3 = tuple[float, float, float]
Vector2 = tuple[float, float]

_EPS = 1.0e-12


@dataclass(frozen=True, slots=True)
class PlaneFit:
    """Least-squares plane fit for a set of 3D points."""

    center: Vector3
    normal: Vector3
    rms: float
    mad: float
    max_abs: float
    point_count: int
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Circle2DFit:
    """Kasa-style least-squares circle fit in 2D."""

    center: Vector2
    radius: float
    rms: float
    rel_rms: float
    mad: float
    max_abs: float
    point_count: int
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RingFit:
    """3D ring fit: plane + circle projected into that plane."""

    center: Vector3
    axis: Vector3
    radius: float
    diameter: float
    plane_rms: float
    plane_rel_rms: float
    plane_mad: float
    radius_rms: float
    radius_rel_rms: float
    radius_mad: float
    circularity: float
    point_count: int
    confidence: float
    diagnostics: dict[str, object] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Basic numeric helpers
# -----------------------------------------------------------------------------


def clamp(value: float, lo: float, hi: float) -> float:
    """Return value limited to [lo, hi]."""

    return float(max(float(lo), min(float(hi), float(value))))


def as_points3(points: Iterable[object] | np.ndarray) -> np.ndarray:
    """Coerce an iterable/array to an ``(N, 3)`` float array."""

    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.empty((0, 3), dtype=float)
    return arr[:, :3].astype(float, copy=False)


def to_vector3(value: object) -> Vector3:
    """Convert any 3-ish numeric value to a plain tuple."""

    arr = np.asarray(value, dtype=float).reshape(-1)
    out = np.zeros(3, dtype=float)
    out[: min(3, len(arr))] = arr[: min(3, len(arr))]
    return (float(out[0]), float(out[1]), float(out[2]))


def median(values: Iterable[float]) -> float:
    """Robust median with an empty-input fallback."""

    arr = np.asarray(tuple(float(v) for v in values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.median(arr))


def median_abs_deviation(values: Iterable[float]) -> float:
    """Median absolute deviation around the median."""

    arr = np.asarray(tuple(float(v) for v in values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def unit_vector(vector: object, *, fallback: object = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Return a normalized vector with a deterministic fallback."""

    vec = np.asarray(vector, dtype=float).reshape(3)
    length = float(np.linalg.norm(vec))
    if np.isfinite(length) and length > _EPS:
        return vec / length
    fb = np.asarray(fallback, dtype=float).reshape(3)
    fb_length = float(np.linalg.norm(fb))
    if np.isfinite(fb_length) and fb_length > _EPS:
        return fb / fb_length
    return np.array([0.0, 0.0, 1.0], dtype=float)


def canonical_axis(axis: object) -> np.ndarray:
    """Normalize an axis and orient it deterministically by its major component."""

    a = unit_vector(axis)
    major = int(np.argmax(np.abs(a)))
    if float(a[major]) < 0.0:
        a = -a
    return a


def axis_hint(axis: object) -> str:
    """Return X/Y/Z for mostly axis-aligned vectors, otherwise FREE."""

    a = unit_vector(axis)
    values = np.abs(a)
    best = float(values.max())
    if best < 0.90:
        return "FREE"
    return ("X", "Y", "Z")[int(np.argmax(values))]


def line_base_from_points(points: Iterable[object] | np.ndarray, axis: object) -> np.ndarray:
    """Return the mean line base perpendicular to ``axis``."""

    pts = as_points3(points)
    if len(pts) == 0:
        return np.zeros(3, dtype=float)
    a = canonical_axis(axis)
    projections = pts - np.outer(pts @ a, a)
    return projections.mean(axis=0)


def line_distance_parallel(base0: object, axis0: object, base1: object, axis1: object, *, parallel_dot: float = 0.985) -> float:
    """Distance between two near-parallel centerlines, or a large value."""

    a0 = canonical_axis(axis0)
    a1 = canonical_axis(axis1)
    if abs(float(np.dot(a0, a1))) < float(parallel_dot):
        return 999999.0
    p0 = np.asarray(base0, dtype=float).reshape(3)
    p1 = np.asarray(base1, dtype=float).reshape(3)
    d = p1 - p0
    return float(np.linalg.norm(d - a0 * float(np.dot(d, a0))))


# -----------------------------------------------------------------------------
# Plane, basis and circle fitting
# -----------------------------------------------------------------------------


def fit_plane(points: Iterable[object] | np.ndarray) -> PlaneFit | None:
    """Fit a best plane with PCA/SVD.

    The smallest-variance direction is returned as the plane normal.  This works for angled bores too, not only axis-aligned openings.
    """

    pts = as_points3(points)
    if len(pts) < 3:
        return None
    center = pts.mean(axis=0)
    centered = pts - center
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return None
    if vh.shape[0] < 3:
        return None
    normal = unit_vector(vh[-1])
    distances = centered @ normal
    abs_dist = np.abs(distances)
    rms = float(np.sqrt(np.mean(distances * distances))) if distances.size else 0.0
    mad = float(np.median(abs_dist)) if abs_dist.size else 0.0
    max_abs = float(abs_dist.max()) if abs_dist.size else 0.0
    return PlaneFit(
        center=to_vector3(center),
        normal=to_vector3(normal),
        rms=rms,
        mad=mad,
        max_abs=max_abs,
        point_count=int(len(pts)),
        diagnostics={"axis_hint": axis_hint(normal)},
    )


def plane_basis(axis: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal ``(u, v, axis)`` basis for a plane normal/axis."""

    a = unit_vector(axis)
    helper = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(helper, a))) > 0.85:
        helper = np.array([0.0, 1.0, 0.0], dtype=float)
    u = helper - a * float(np.dot(helper, a))
    u = unit_vector(u, fallback=(0.0, 1.0, 0.0))
    v = np.cross(a, u)
    v = unit_vector(v, fallback=(0.0, 0.0, 1.0))
    return u, v, a


def project_points_to_plane(points: Iterable[object] | np.ndarray, *, center: object, axis: object) -> np.ndarray:
    """Project 3D points to a 2D coordinate system on a plane."""

    pts = as_points3(points)
    if len(pts) == 0:
        return np.empty((0, 2), dtype=float)
    c = np.asarray(center, dtype=float).reshape(3)
    u, v, _axis = plane_basis(axis)
    rel = pts - c
    return np.column_stack((rel @ u, rel @ v)).astype(float, copy=False)


def fit_circle_2d(points2d: Iterable[object] | np.ndarray, *, min_points: int = 6, min_radius: float = 0.05) -> Circle2DFit | None:
    """Kasa-style least-squares circle fit.

    Fit ``x² + y² + D*x + E*y + F = 0`` and let callers enforce strict RMS
guards using numpy's solver.
    """

    arr = np.asarray(points2d, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    pts = arr[:, :2]
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < int(min_points):
        return None

    x = pts[:, 0]
    y = pts[:, 1]
    z = x * x + y * y
    mat = np.array(
        [
            [float(np.dot(x, x)), float(np.dot(x, y)), float(np.sum(x))],
            [float(np.dot(x, y)), float(np.dot(y, y)), float(np.sum(y))],
            [float(np.sum(x)), float(np.sum(y)), float(len(pts))],
        ],
        dtype=float,
    )
    rhs = -np.array([float(np.dot(x, z)), float(np.dot(y, z)), float(np.sum(z))], dtype=float)

    try:
        d, e, f = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        return None
    except Exception:
        return None

    cx = -0.5 * float(d)
    cy = -0.5 * float(e)
    r2 = cx * cx + cy * cy - float(f)
    if not np.isfinite(r2) or r2 <= _EPS:
        return None
    radius = float(np.sqrt(r2))
    if radius < float(min_radius):
        return None

    radii = np.sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy))
    residuals = radii - radius
    abs_res = np.abs(residuals)
    rms = float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else 0.0
    rel_rms = float(rms / max(radius, _EPS))
    mad = float(np.median(abs_res)) if abs_res.size else 0.0
    max_abs = float(abs_res.max()) if abs_res.size else 0.0

    return Circle2DFit(
        center=(float(cx), float(cy)),
        radius=radius,
        rms=rms,
        rel_rms=rel_rms,
        mad=mad,
        max_abs=max_abs,
        point_count=int(len(pts)),
    )


def fit_ring_points(
    points: Iterable[object] | np.ndarray,
    *,
    axis: object | None = None,
    min_points: int = 6,
    min_radius: float = 0.05,
    max_rel_rms_for_confidence: float = 0.070,
    max_plane_rel_for_confidence: float = 0.035,
) -> RingFit | None:
    """Fit a 3D circular/near-circular ring to mesh points.

    This is intentionally an analyzer, not an accept/reject gate.  It reports
    plane and radius errors so callers can decide whether a measured opening is
    strong enough for selection, wall detection or rebuild.
    """

    pts = as_points3(points)
    if len(pts) < int(min_points):
        return None

    plane = fit_plane(pts)
    if plane is None:
        return None

    plane_center = np.asarray(plane.center, dtype=float)
    plane_axis = unit_vector(axis, fallback=plane.normal) if axis is not None else np.asarray(plane.normal, dtype=float)
    plane_axis = unit_vector(plane_axis)

    pts2 = project_points_to_plane(pts, center=plane_center, axis=plane_axis)
    circle = fit_circle_2d(pts2, min_points=min_points, min_radius=min_radius)
    if circle is None:
        return None

    u, v, a = plane_basis(plane_axis)
    cx, cy = circle.center
    center3 = plane_center + u * float(cx) + v * float(cy)

    rel = pts - center3
    axial = rel @ a
    radial = rel - np.outer(axial, a)
    radii = np.linalg.norm(radial, axis=1)
    radius = float(circle.radius)
    residuals = radii - radius
    abs_res = np.abs(residuals)

    plane_abs = np.abs(axial)
    plane_rms = float(np.sqrt(np.mean(axial * axial))) if axial.size else 0.0
    plane_mad = float(np.median(plane_abs)) if plane_abs.size else 0.0
    plane_rel = float(plane_rms / max(radius, _EPS))
    radius_rms = float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else 0.0
    radius_rel = float(radius_rms / max(radius, _EPS))
    radius_mad = float(np.median(abs_res)) if abs_res.size else 0.0

    radius_score = 1.0 - clamp(radius_rel / max(float(max_rel_rms_for_confidence), _EPS), 0.0, 1.0)
    plane_score = 1.0 - clamp(plane_rel / max(float(max_plane_rel_for_confidence), _EPS), 0.0, 1.0)
    point_score = clamp((len(pts) - max(0, int(min_points) - 1)) / 24.0, 0.0, 1.0)
    circularity = clamp(0.65 * radius_score + 0.35 * plane_score, 0.0, 1.0)
    confidence = clamp(0.15 + 0.55 * circularity + 0.30 * point_score, 0.0, 1.0)

    return RingFit(
        center=to_vector3(center3),
        axis=to_vector3(canonical_axis(a)),
        radius=radius,
        diameter=float(2.0 * radius),
        plane_rms=plane_rms,
        plane_rel_rms=plane_rel,
        plane_mad=plane_mad,
        radius_rms=radius_rms,
        radius_rel_rms=radius_rel,
        radius_mad=radius_mad,
        circularity=circularity,
        point_count=int(len(pts)),
        confidence=confidence,
        diagnostics={
            "axis_hint": axis_hint(a),
            "circle_rms": float(circle.rms),
            "circle_rel_rms": float(circle.rel_rms),
            "circle_max_abs": float(circle.max_abs),
            "plane_fit_rms": float(plane.rms),
            "plane_fit_max_abs": float(plane.max_abs),
        },
    )


# -----------------------------------------------------------------------------
# Bore boundary-loop stack geometry
# -----------------------------------------------------------------------------

EdgeKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class BoundaryLoopGeometry:
    """Geometric description of one closed boundary loop of a Bore region patch."""

    index: int
    edge_count: int
    vertex_count: int
    center: Vector3
    axis: Vector3
    axial_position: float
    radius: float
    radius_mad: float
    plane_rms: float
    radius_rel_mad: float
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "index": int(self.index),
            "edge_count": int(self.edge_count),
            "vertex_count": int(self.vertex_count),
            "center": self.center,
            "axis": self.axis,
            "axial_position": float(self.axial_position),
            "radius": float(self.radius),
            "radius_mad": float(self.radius_mad),
            "plane_rms": float(self.plane_rms),
            "radius_rel_mad": float(self.radius_rel_mad),
            **dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class BoundaryLoopStackGeometry:
    """Geometry ledger for all wall-boundary loops in one Bore feature preview."""

    boundary_loops: tuple[BoundaryLoopGeometry, ...]
    boundary_loop_edge_counts: tuple[int, ...]
    suggested_core_pair_indices: tuple[int, int] | None
    suggested_core_pair_reason: str
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": "bore_boundary_loop_stack_geometry",
            "boundary_loop_count": len(self.boundary_loops),
            "boundary_loop_edge_counts": self.boundary_loop_edge_counts,
            "boundary_loops": tuple(item.to_dict() for item in self.boundary_loops),
            "suggested_core_pair_indices": self.suggested_core_pair_indices,
            "suggested_core_pair_reason": self.suggested_core_pair_reason,
            **dict(self.diagnostics),
        }


def normalize_edge(edge: object) -> EdgeKey:
    """Return a stable undirected integer edge key."""

    a_raw, b_raw = edge  # type: ignore[misc]
    a = int(a_raw)
    b = int(b_raw)
    return (a, b) if a <= b else (b, a)


def boundary_loop_vertex_ids(loop: Iterable[object], vertex_count: int | None = None) -> tuple[int, ...]:
    """Return sorted unique vertex IDs from an edge loop."""

    out: set[int] = set()
    limit = None if vertex_count is None else int(vertex_count)
    for edge in loop:
        try:
            a, b = normalize_edge(edge)
        except Exception:
            continue
        if a >= 0 and (limit is None or a < limit):
            out.add(a)
        if b >= 0 and (limit is None or b < limit):
            out.add(b)
    return tuple(sorted(out))


def describe_boundary_loop_geometry(
    *,
    index: int,
    loop: Iterable[object],
    vertices: Iterable[object] | np.ndarray,
    axis: object,
) -> BoundaryLoopGeometry:
    """Measure one wall-boundary loop in the plane perpendicular to ``axis``."""

    verts = as_points3(vertices)
    edges = tuple(normalize_edge(edge) for edge in loop)
    vertex_ids = boundary_loop_vertex_ids(edges, len(verts))
    axis_vec = canonical_axis(axis)
    if not vertex_ids or len(verts) == 0:
        return BoundaryLoopGeometry(
            index=int(index),
            edge_count=len(edges),
            vertex_count=0,
            center=(0.0, 0.0, 0.0),
            axis=to_vector3(axis_vec),
            axial_position=0.0,
            radius=0.0,
            radius_mad=0.0,
            plane_rms=0.0,
            radius_rel_mad=0.0,
        )

    pts = verts[np.asarray(vertex_ids, dtype=np.int64), :3]
    center = pts.mean(axis=0)
    rel = pts - center.reshape(1, 3)
    axial_offsets = rel @ axis_vec.reshape(3, 1)
    radial = rel - axial_offsets * axis_vec.reshape(1, 3)
    radii = np.linalg.norm(radial, axis=1)
    radius = float(np.median(radii)) if radii.size else 0.0
    radius_mad = float(np.median(np.abs(radii - radius))) if radii.size else 0.0
    plane_rms = float(np.sqrt(np.mean(np.square(axial_offsets)))) if axial_offsets.size else 0.0
    radius_rel_mad = float(radius_mad / max(radius, _EPS))

    return BoundaryLoopGeometry(
        index=int(index),
        edge_count=int(len(edges)),
        vertex_count=int(len(vertex_ids)),
        center=to_vector3(center),
        axis=to_vector3(axis_vec),
        axial_position=float(center @ axis_vec.reshape(3)),
        radius=radius,
        radius_mad=radius_mad,
        plane_rms=plane_rms,
        radius_rel_mad=radius_rel_mad,
        diagnostics={"vertex_ids_sample": tuple(vertex_ids[:8])},
    )


def describe_boundary_loop_stack_geometry(
    *,
    boundary_loops: Iterable[Iterable[object]],
    vertices: Iterable[object] | np.ndarray,
    axis: object,
    nominal_radius: float = 0.0,
    min_loop_edges: int = 12,
) -> BoundaryLoopStackGeometry:
    """Measure all boundary loops and find a conservative core pair candidate.

    The core-pair suggestion is diagnostic only.  It requires comparable loop
    sizes, similar radii, axial separation, and—critically—coaxial centers.  The
    center guard prevents unrelated pocket/chamfer loops in a feature array from
    being paired just because their radii match.
    """

    axis_vec = canonical_axis(axis)
    loops = tuple(tuple(normalize_edge(edge) for edge in loop) for loop in boundary_loops)
    loop_geometry = tuple(
        describe_boundary_loop_geometry(index=i, loop=loop, vertices=vertices, axis=axis_vec)
        for i, loop in enumerate(loops)
    )
    pair, reason, diag = suggest_core_boundary_loop_pair(
        loop_geometry=loop_geometry,
        axis=axis_vec,
        nominal_radius=float(nominal_radius),
        min_loop_edges=int(min_loop_edges),
    )
    return BoundaryLoopStackGeometry(
        boundary_loops=loop_geometry,
        boundary_loop_edge_counts=tuple(int(len(loop)) for loop in loops),
        suggested_core_pair_indices=pair,
        suggested_core_pair_reason=reason,
        diagnostics=diag,
    )


def suggest_core_boundary_loop_pair(
    *,
    loop_geometry: tuple[BoundaryLoopGeometry, ...],
    axis: object,
    nominal_radius: float,
    min_loop_edges: int,
) -> tuple[tuple[int, int] | None, str, dict[str, object]]:
    """Return a diagnostic-only likely inner/core pair from a loop stack."""

    if len(loop_geometry) < 3:
        return None, "", {"core_pair_candidate_count": 0}

    axis_vec = canonical_axis(axis)
    radius_ref = max(float(nominal_radius), 1.0e-9)
    candidates: list[tuple[float, tuple[int, int], dict[str, object]]] = []

    for i, a in enumerate(loop_geometry):
        if a.edge_count < int(min_loop_edges) or a.radius <= 1.0e-9:
            continue
        ca = np.asarray(a.center, dtype=float).reshape(3)
        for b in loop_geometry[i + 1 :]:
            if b.edge_count < int(min_loop_edges) or b.radius <= 1.0e-9:
                continue
            cb = np.asarray(b.center, dtype=float).reshape(3)
            small_edges = max(min(a.edge_count, b.edge_count), 1)
            edge_ratio = max(a.edge_count, b.edge_count) / float(small_edges)
            radius_delta_rel = abs(a.radius - b.radius) / max(max(a.radius, b.radius), 1.0e-9)
            axial_distance = abs(a.axial_position - b.axial_position)
            avg_radius = 0.5 * (a.radius + b.radius)
            center_delta = cb - ca
            center_cross = float(np.linalg.norm(center_delta - axis_vec * float(center_delta @ axis_vec)))

            max_center_cross = max(avg_radius * 0.55, radius_ref * 0.18, 0.75)
            min_axial = max(avg_radius * 0.20, 1.0e-6)
            if edge_ratio > 1.22:
                continue
            if radius_delta_rel > 0.18:
                continue
            if axial_distance < min_axial:
                continue
            if center_cross > max_center_cross:
                continue

            # Prefer smaller, coaxial rings with useful depth.  Lower score wins.
            score = (
                (avg_radius / radius_ref)
                + 0.35 * radius_delta_rel
                + 0.20 * (center_cross / max(max_center_cross, 1.0e-9))
                - 0.015 * min(axial_distance / max(radius_ref, 1.0e-9), 20.0)
            )
            candidates.append(
                (
                    float(score),
                    (int(a.index), int(b.index)),
                    {
                        "edge_ratio": float(edge_ratio),
                        "radius_delta_rel": float(radius_delta_rel),
                        "axial_distance": float(axial_distance),
                        "avg_radius": float(avg_radius),
                        "center_cross_distance": float(center_cross),
                        "max_center_cross_distance": float(max_center_cross),
                    },
                )
            )

    if not candidates:
        return None, "no comparable coaxial loop pair in multi-loop feature stack", {"core_pair_candidate_count": 0}

    candidates.sort(key=lambda item: item[0])
    score, pair, metrics = candidates[0]
    return pair, "diagnostic-only possible coaxial inner/core pair; not used for rebuild yet", {
        "core_pair_candidate_count": len(candidates),
        "suggested_core_pair_score": float(score),
        "suggested_core_pair_metrics": metrics,
    }


def boundary_loop_radius_families(
    loop_geometry: tuple[BoundaryLoopGeometry, ...],
    *,
    rel_tol: float = 0.12,
    abs_tol: float = 0.20,
) -> tuple[dict[str, object], ...]:
    """Group measured boundary loops into coarse radius families.

    This is deliberately descriptive, not an acceptance test.  It helps the
    feature layer distinguish one bore region from a plate face that contains a
    large mouth plus several neighbouring small bores/pocket loops.
    """

    loops = sorted(tuple(loop_geometry), key=lambda loop: float(loop.radius))
    families: list[list[BoundaryLoopGeometry]] = []
    for loop in loops:
        placed = False
        for family in families:
            ref = float(np.median([member.radius for member in family])) if family else 0.0
            tol = max(float(abs_tol), max(abs(ref), abs(loop.radius)) * float(rel_tol))
            if abs(float(loop.radius) - ref) <= tol:
                family.append(loop)
                placed = True
                break
        if not placed:
            families.append([loop])

    out: list[dict[str, object]] = []
    for family_index, family in enumerate(families):
        radii = [float(member.radius) for member in family]
        edge_counts = [int(member.edge_count) for member in family]
        centers = np.asarray([member.center for member in family], dtype=float) if family else np.zeros((0, 3))
        if len(centers) >= 2:
            span = np.ptp(centers, axis=0)
            center_spread = float(np.linalg.norm(span))
        else:
            center_spread = 0.0
        out.append(
            {
                "family_index": int(family_index),
                "count": int(len(family)),
                "loop_indices": tuple(int(member.index) for member in family),
                "median_radius": float(np.median(radii)) if radii else 0.0,
                "min_radius": float(min(radii)) if radii else 0.0,
                "max_radius": float(max(radii)) if radii else 0.0,
                "median_edge_count": float(np.median(edge_counts)) if edge_counts else 0.0,
                "edge_counts": tuple(edge_counts),
                "center_spread": float(center_spread),
            }
        )
    return tuple(out)


def boundary_loop_spatial_families(
    loop_geometry: tuple[BoundaryLoopGeometry, ...],
    *,
    axis: object = (0.0, 0.0, 1.0),
    rel_radius_tol: float = 0.14,
    abs_radius_tol: float = 0.20,
    center_factor: float = 1.15,
    center_abs_tol: float = 1.25,
) -> tuple[dict[str, object], ...]:
    """Group loops by radius family and approximate centerline.

    Radius-only grouping is enough to notice repeated small bores, but not
    enough to distinguish a coaxial counterbore stack from neighbouring bores
    distributed around a plate.  This helper keeps the grouping descriptive and
    bounded: it first requires similar radius, then checks cross-axis center
    distance.  The feature layer can use the resulting families as a ledger, not
    as rebuild authorization.
    """

    axis_vec = canonical_axis(axis)
    families: list[list[BoundaryLoopGeometry]] = []

    def _family_radius(family: list[BoundaryLoopGeometry]) -> float:
        return float(np.median([float(item.radius) for item in family])) if family else 0.0

    def _family_center(family: list[BoundaryLoopGeometry]) -> np.ndarray:
        if not family:
            return np.zeros(3, dtype=float)
        centers = np.asarray([item.center for item in family], dtype=float)
        return centers.mean(axis=0)

    for loop in sorted(tuple(loop_geometry), key=lambda item: (-float(item.radius), int(item.index))):
        placed = False
        center = np.asarray(loop.center, dtype=float).reshape(3)
        for family in families:
            ref_radius = _family_radius(family)
            radius_tol = max(float(abs_radius_tol), max(abs(ref_radius), abs(loop.radius)) * float(rel_radius_tol))
            if abs(float(loop.radius) - ref_radius) > radius_tol:
                continue
            ref_center = _family_center(family)
            delta = center - ref_center
            center_cross = float(np.linalg.norm(delta - axis_vec * float(delta @ axis_vec)))
            center_tol = max(float(center_abs_tol), max(abs(ref_radius), abs(loop.radius)) * float(center_factor))
            if center_cross <= center_tol:
                family.append(loop)
                placed = True
                break
        if not placed:
            families.append([loop])

    out: list[dict[str, object]] = []
    for family_index, family in enumerate(families):
        radii = [float(item.radius) for item in family]
        centers = np.asarray([item.center for item in family], dtype=float) if family else np.zeros((0, 3))
        axial_positions = [float(item.axial_position) for item in family]
        if len(centers) >= 2:
            mean_center = centers.mean(axis=0)
            cross_distances = []
            for center in centers:
                delta = center - mean_center
                cross_distances.append(float(np.linalg.norm(delta - axis_vec * float(delta @ axis_vec))))
            center_cross_spread = float(max(cross_distances)) if cross_distances else 0.0
        else:
            mean_center = centers[0] if len(centers) else np.zeros(3, dtype=float)
            center_cross_spread = 0.0
        out.append(
            {
                "family_index": int(family_index),
                "count": int(len(family)),
                "loop_indices": tuple(int(item.index) for item in family),
                "median_radius": float(np.median(radii)) if radii else 0.0,
                "min_radius": float(min(radii)) if radii else 0.0,
                "max_radius": float(max(radii)) if radii else 0.0,
                "center": to_vector3(mean_center),
                "center_cross_spread": float(center_cross_spread),
                "axial_min": float(min(axial_positions)) if axial_positions else 0.0,
                "axial_max": float(max(axial_positions)) if axial_positions else 0.0,
                "axial_span": float(max(axial_positions) - min(axial_positions)) if axial_positions else 0.0,
                "edge_counts": tuple(int(item.edge_count) for item in family),
            }
        )
    return tuple(out)


@dataclass(frozen=True, slots=True)
class CylinderFaceEvidence:
    """Measured residuals of mesh faces against one bore cylinder model.

    This is deliberately a measurement object, not a selection rule.  Callers can derive thresholds from local rim-adjacent samples instead of
    hard-coded guesses.
    """

    center: Vector3
    axis: Vector3
    radius: float
    axial_position: np.ndarray
    radial_distance: np.ndarray
    radial_error: np.ndarray
    normal_axis_abs: np.ndarray
    radial_normal_alignment: np.ndarray
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FeaturePatchMeasurement:
    """Measured geometry of an inspection-only Bore feature patch.

    This is for chamfer mouths, blind pockets, counterbore layers and other
    Bore-related feature regions that are not safe simple two-rim rebuilds yet.
    It records what was measured so callers do not have to infer feature meaning
    from a face count or a blocked rebuild button.
    """

    center: Vector3
    axis: Vector3
    radius: float
    face_count: int
    face_ids: tuple[int, ...] = ()
    axial_min: float = 0.0
    axial_max: float = 0.0
    axial_span: float = 0.0
    radial_error_median: float = 0.0
    radial_error_mad: float = 0.0
    radial_error_q85: float = 0.0
    normal_axis_abs_median: float = 0.0
    radial_normal_alignment_median: float = 0.0
    boundary_loop_count: int = 0
    mouth_loop_index: int | None = None
    mouth_loop_edge_count: int = 0
    mouth_loop_radius: float = 0.0
    mouth_loop_center: Vector3 = (0.0, 0.0, 0.0)
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": "bore_feature_patch_measurement",
            "center": self.center,
            "axis": self.axis,
            "radius": float(self.radius),
            "diameter": float(2.0 * self.radius),
            "face_count": int(self.face_count),
            "face_ids": tuple(int(fid) for fid in tuple(self.face_ids or ())),
            "face_id_count": int(len(tuple(self.face_ids or ()))),
            "axial_min": float(self.axial_min),
            "axial_max": float(self.axial_max),
            "axial_span": float(self.axial_span),
            "depth_estimate": float(self.axial_span),
            "radial_error_median": float(self.radial_error_median),
            "radial_error_mad": float(self.radial_error_mad),
            "radial_error_q85": float(self.radial_error_q85),
            "normal_axis_abs_median": float(self.normal_axis_abs_median),
            "radial_normal_alignment_median": float(self.radial_normal_alignment_median),
            "boundary_loop_count": int(self.boundary_loop_count),
            "mouth_loop_index": self.mouth_loop_index,
            "mouth_loop_edge_count": int(self.mouth_loop_edge_count),
            "mouth_loop_radius": float(self.mouth_loop_radius),
            "mouth_loop_diameter": float(2.0 * self.mouth_loop_radius),
            "mouth_loop_center": self.mouth_loop_center,
            **dict(self.diagnostics),
        }


def measure_faces_against_cylinder(
    face_centroids: Iterable[object] | np.ndarray,
    face_normals: Iterable[object] | np.ndarray,
    *,
    center: object,
    axis: object,
    radius: float,
) -> CylinderFaceEvidence:
    """Measure face centroids/normals against a bore cylinder.

    Returns per-face arrays for radial residual, axial position, normal-axis
    relation and radial-normal relation.  This ports the macro's evidence-first
    approach into the mesh Bore core: first measure the geometry, then let recognition decide feature meaning outside this module.
    """

    centroids = as_points3(face_centroids)
    normals = as_points3(face_normals)
    if len(normals) != len(centroids):
        n = min(len(normals), len(centroids))
        centroids = centroids[:n]
        normals = normals[:n]

    c = np.asarray(center, dtype=float).reshape(3)
    a = unit_vector(axis)
    r = float(max(float(radius), 0.0))

    rel = centroids - c.reshape(1, 3) if len(centroids) else np.zeros((0, 3), dtype=float)
    axial = rel @ a.reshape(3, 1) if len(rel) else np.zeros((0, 1), dtype=float)
    axial_flat = axial.reshape(-1)
    radial_vec = rel - axial * a.reshape(1, 3) if len(rel) else np.zeros((0, 3), dtype=float)
    radial_distance = np.linalg.norm(radial_vec, axis=1) if len(radial_vec) else np.zeros(0, dtype=float)
    radial_error = np.abs(radial_distance - r)

    normal_axis_abs = np.abs(normals @ a.reshape(3, 1)).reshape(-1) if len(normals) else np.zeros(0, dtype=float)
    radial_unit = np.zeros_like(radial_vec, dtype=float)
    ok = radial_distance > _EPS
    if np.any(ok):
        radial_unit[ok] = radial_vec[ok] / radial_distance[ok].reshape(-1, 1)
    radial_normal_alignment = np.abs(np.sum(normals * radial_unit, axis=1)) if len(normals) else np.zeros(0, dtype=float)

    finite_radial_error = radial_error[np.isfinite(radial_error)]
    finite_alignment = radial_normal_alignment[np.isfinite(radial_normal_alignment)]
    diagnostics = {
        "mode": "cylinder_face_evidence",
        "face_count": int(len(centroids)),
        "radius": float(r),
        "radial_error_median": float(np.median(finite_radial_error)) if finite_radial_error.size else 0.0,
        "radial_error_mad": float(np.median(np.abs(finite_radial_error - np.median(finite_radial_error)))) if finite_radial_error.size else 0.0,
        "radial_normal_alignment_median": float(np.median(finite_alignment)) if finite_alignment.size else 0.0,
    }
    return CylinderFaceEvidence(
        center=to_vector3(c),
        axis=to_vector3(a),
        radius=float(r),
        axial_position=axial_flat,
        radial_distance=radial_distance,
        radial_error=radial_error,
        normal_axis_abs=normal_axis_abs,
        radial_normal_alignment=radial_normal_alignment,
        diagnostics=diagnostics,
    )


def measure_feature_patch_geometry(
    *,
    face_ids: Iterable[int],
    face_centroids: Iterable[object] | np.ndarray,
    face_normals: Iterable[object] | np.ndarray,
    center: object,
    axis: object,
    radius: float,
    boundary_loop_geometry: Iterable[BoundaryLoopGeometry] = (),
) -> FeaturePatchMeasurement:
    """Measure a Bore-related feature patch against a cylinder frame.

    The output is descriptive only.  It is used for neutral evidence reporting,
    not for feature promotion or authorizing rebuild.  The main
    point is to keep mouth radius, depth/axial span and face residual evidence
    explicit instead of hiding it behind a generic ``not rebuild ready`` status.
    """

    centroids_all = as_points3(face_centroids)
    normals_all = as_points3(face_normals)
    ids = tuple(sorted({int(fid) for fid in face_ids if int(fid) >= 0 and int(fid) < len(centroids_all)}))
    if not ids:
        return FeaturePatchMeasurement(
            center=to_vector3(center),
            axis=to_vector3(unit_vector(axis)),
            radius=float(radius),
            face_count=0,
            axial_min=0.0,
            axial_max=0.0,
            axial_span=0.0,
            radial_error_median=0.0,
            radial_error_mad=0.0,
            radial_error_q85=0.0,
            normal_axis_abs_median=0.0,
            radial_normal_alignment_median=0.0,
            boundary_loop_count=0,
            diagnostics={"reason": "no_feature_patch_faces"},
        )

    centroids = centroids_all[np.asarray(ids, dtype=np.int64), :3]
    normals = normals_all[np.asarray(ids, dtype=np.int64), :3] if len(normals_all) >= len(centroids_all) else np.zeros_like(centroids)

    # Phase 39: measure the feature patch in the best physical mouth frame, not
    # necessarily the noisy raw selected-edge fit.  The user pick remains useful
    # evidence, but fragmented selected openings can produce an inflated radius
    # and shifted center.  When a real wall-boundary/mouth loop has been derived,
    # use that loop as the analysis frame for radial/axial surface typing.
    loops = tuple(boundary_loop_geometry)
    mouth_loop = max(loops, key=lambda item: (float(item.radius), int(item.edge_count)), default=None)
    analysis_center = center
    analysis_axis = axis
    analysis_radius = float(radius)
    measurement_frame_source = "selected_opening_fit"
    raw_center_tuple = to_vector3(center)
    raw_axis_tuple = to_vector3(unit_vector(axis))
    raw_radius_value = float(radius)
    if mouth_loop is not None and float(mouth_loop.radius) > _EPS and int(mouth_loop.edge_count) >= 8:
        analysis_center = mouth_loop.center
        analysis_axis = mouth_loop.axis
        analysis_radius = float(mouth_loop.radius)
        measurement_frame_source = "physical_mouth_boundary_loop"

    evidence = measure_faces_against_cylinder(
        centroids,
        normals,
        center=analysis_center,
        axis=analysis_axis,
        radius=float(analysis_radius),
    )

    axial = evidence.axial_position[np.isfinite(evidence.axial_position)]
    radial_error = evidence.radial_error[np.isfinite(evidence.radial_error)]
    normal_axis = evidence.normal_axis_abs[np.isfinite(evidence.normal_axis_abs)]
    alignment = evidence.radial_normal_alignment[np.isfinite(evidence.radial_normal_alignment)]

    if axial.size:
        axial_min = float(np.min(axial))
        axial_max = float(np.max(axial))
    else:
        axial_min = 0.0
        axial_max = 0.0
    radial_med = float(np.median(radial_error)) if radial_error.size else 0.0
    radial_mad = float(np.median(np.abs(radial_error - radial_med))) if radial_error.size else 0.0
    radial_q85 = float(np.percentile(radial_error, 85.0)) if radial_error.size else 0.0

    radial_distance = evidence.radial_distance[np.isfinite(evidence.radial_distance)]
    axis_profile = {
        "count": int(axial.size),
        "min": float(np.min(axial)) if axial.size else 0.0,
        "q10": float(np.percentile(axial, 10.0)) if axial.size else 0.0,
        "median": float(np.median(axial)) if axial.size else 0.0,
        "q90": float(np.percentile(axial, 90.0)) if axial.size else 0.0,
        "max": float(np.max(axial)) if axial.size else 0.0,
        "span": float((np.max(axial) - np.min(axial)) if axial.size else 0.0),
    }
    radial_profile = {
        "count": int(radial_distance.size),
        "min": float(np.min(radial_distance)) if radial_distance.size else 0.0,
        "q10": float(np.percentile(radial_distance, 10.0)) if radial_distance.size else 0.0,
        "median": float(np.median(radial_distance)) if radial_distance.size else 0.0,
        "q90": float(np.percentile(radial_distance, 90.0)) if radial_distance.size else 0.0,
        "max": float(np.max(radial_distance)) if radial_distance.size else 0.0,
        "nominal_radius": float(analysis_radius),
        "raw_selected_opening_radius": float(raw_radius_value),
    }

    mouth_radius_for_layers = float(mouth_loop.radius) if mouth_loop is not None and float(mouth_loop.radius) > _EPS else float(analysis_radius)

    # Phase 8.1 cleanup:
    # geometry.py is primitive measurement only.  It must not call recognition,
    # split feature ownership, promote high-level entities, or classify radial
    # layers as BOREHOLE/CHAMFER/etc.  Keep only neutral radial/axial evidence.
    core_radius_estimate = float(np.percentile(radial_distance, 10.0)) if radial_distance.size else float(analysis_radius)
    radial_delta = float(max(mouth_radius_for_layers - core_radius_estimate, 0.0))
    radial_delta_ratio = float(radial_delta / max(abs(mouth_radius_for_layers), _EPS))
    radial_layer_analysis = {
        "mode": "geometry_measurement_only_radial_axial_profile",
        "classification_policy": "none_geometry_does_not_classify_or_promote_features",
        "mouth_radius": float(mouth_radius_for_layers),
        "nominal_radius": float(radius),
        "core_radius_estimate": float(core_radius_estimate),
        "radius_delta": float(radial_delta),
        "radius_delta_ratio": float(radial_delta_ratio),
        "face_count": int(len(ids)),
        "axis_profile": axis_profile,
        "radial_profile": radial_profile,
        "normal_axis_abs_profile": {
            "count": int(normal_axis.size),
            "min": float(np.min(normal_axis)) if normal_axis.size else 0.0,
            "q10": float(np.percentile(normal_axis, 10.0)) if normal_axis.size else 0.0,
            "median": float(np.median(normal_axis)) if normal_axis.size else 0.0,
            "q90": float(np.percentile(normal_axis, 90.0)) if normal_axis.size else 0.0,
            "max": float(np.max(normal_axis)) if normal_axis.size else 0.0,
        },
        "radial_normal_alignment_profile": {
            "count": int(alignment.size),
            "min": float(np.min(alignment)) if alignment.size else 0.0,
            "q10": float(np.percentile(alignment, 10.0)) if alignment.size else 0.0,
            "median": float(np.median(alignment)) if alignment.size else 0.0,
            "q90": float(np.percentile(alignment, 90.0)) if alignment.size else 0.0,
            "max": float(np.max(alignment)) if alignment.size else 0.0,
        },
        "entities": (),
        "high_level_entities": (),
        "high_level_entity_count": 0,
    }
    loop_measurements = tuple(
        {
            "index": int(item.index),
            "edge_count": int(item.edge_count),
            "vertex_count": int(item.vertex_count),
            "center": item.center,
            "axial_position": float(item.axial_position),
            "radius": float(item.radius),
            "diameter": float(2.0 * item.radius),
            "radius_mad": float(item.radius_mad),
            "plane_rms": float(item.plane_rms),
        }
        for item in loops
    )

    return FeaturePatchMeasurement(
        center=to_vector3(analysis_center),
        axis=to_vector3(unit_vector(analysis_axis)),
        radius=float(analysis_radius),
        face_count=int(len(ids)),
        face_ids=tuple(int(fid) for fid in ids),
        axial_min=axial_min,
        axial_max=axial_max,
        axial_span=float(axial_max - axial_min),
        radial_error_median=radial_med,
        radial_error_mad=radial_mad,
        radial_error_q85=radial_q85,
        normal_axis_abs_median=float(np.median(normal_axis)) if normal_axis.size else 0.0,
        radial_normal_alignment_median=float(np.median(alignment)) if alignment.size else 0.0,
        boundary_loop_count=int(len(loops)),
        mouth_loop_index=(int(mouth_loop.index) if mouth_loop is not None else None),
        mouth_loop_edge_count=(int(mouth_loop.edge_count) if mouth_loop is not None else 0),
        mouth_loop_radius=(float(mouth_loop.radius) if mouth_loop is not None else 0.0),
        mouth_loop_center=(mouth_loop.center if mouth_loop is not None else (0.0, 0.0, 0.0)),
        diagnostics={
            "source": "selected_feature_patch_faces",
            "measurement_frame_source": str(measurement_frame_source),
            "raw_selected_opening_center": raw_center_tuple,
            "raw_selected_opening_axis": raw_axis_tuple,
            "raw_selected_opening_radius": float(raw_radius_value),
            "analysis_center": to_vector3(analysis_center),
            "analysis_axis": to_vector3(unit_vector(analysis_axis)),
            "analysis_radius": float(analysis_radius),
            "face_id_sample": tuple(ids[:12]),
            "cylinder_evidence": dict(evidence.diagnostics),
            "axis_profile": axis_profile,
            "radial_profile": radial_profile,
            "radial_layer_analysis": radial_layer_analysis,
            "radial_layer_entities": tuple(radial_layer_analysis.get("entities", ()) or ()),
            "boundary_loop_measurements": loop_measurements,
        },
    )



__all__ = [
    "Vector2",
    "Vector3",
    "PlaneFit",
    "Circle2DFit",
    "RingFit",
    "clamp",
    "as_points3",
    "to_vector3",
    "median",
    "median_abs_deviation",
    "unit_vector",
    "canonical_axis",
    "axis_hint",
    "line_base_from_points",
    "line_distance_parallel",
    "fit_plane",
    "plane_basis",
    "project_points_to_plane",
    "fit_circle_2d",
    "fit_ring_points",
    "EdgeKey",
    "BoundaryLoopGeometry",
    "BoundaryLoopStackGeometry",
    "CylinderFaceEvidence",
    "FeaturePatchMeasurement",
    "normalize_edge",
    "boundary_loop_vertex_ids",
    "describe_boundary_loop_geometry",
    "describe_boundary_loop_stack_geometry",
    "suggest_core_boundary_loop_pair",
    "boundary_loop_radius_families",
    "boundary_loop_spatial_families",
    "measure_faces_against_cylinder",
    "measure_feature_patch_geometry",
]