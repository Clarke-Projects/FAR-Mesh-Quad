"""Mesh-realization evidence helpers for Bore opening translation.

This module is intentionally an evidence/measurement layer.  It converts raw
mesh realization details (clean loop, sparse polygon, fragmented cloud, virtual
contour support) into canonical opening-footprint evidence that Recognition may
consume later.

Semantic boundary
-----------------
Input meaning:
    RegionData / selected edges / local mesh arrays = raw mesh evidence.

Output meaning:
    OpeningEvidenceLedgerData / OpeningFootprintAuthority = measured evidence
    about the selected opening footprint and mesh realization quality.

It does not classify a BORE/POCKET/HEX feature, does not assign surface
ownership, does not authorize CandidateData promotion, and does not construct a
rebuild target.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable, Mapping

import math
import numpy as np

from ..types import (
    MeshRealizationAssessment,
    MeshRealizationKind,
    OpeningEvidenceLedgerData,
    OpeningFootprintAuthority,
    OpeningProfileKind,
)

EdgeKey = tuple[int, int]


def build_opening_evidence_ledger_from_arrays(
    *,
    vertices: np.ndarray,
    edge_index_to_vertices: np.ndarray,
    selected_edge_ids: Iterable[int],
    center: object | None = None,
    axis: object | None = None,
    radius: float | None = None,
    source: str = "mesh_realization.opening_footprint_provider",
    support_face_ids: Iterable[int] = (),
    min_support_edges: int = 3,
) -> OpeningEvidenceLedgerData | None:
    """Build a canonical opening evidence ledger from selected mesh edges.

    This is the first mesh-realization translator used by the v120 roadmap.  It
    is deliberately conservative: its result is diagnostics/authority evidence
    only, not feature identity.  The same return contract can represent a clean
    dense loop, a sparse polygonal opening, or a contaminated cloud.
    """

    verts = _as_vertices(vertices)
    edges = _as_edges(edge_index_to_vertices)
    if verts.size == 0 or edges.size == 0:
        return None

    raw_ids = tuple(sorted({int(v) for v in tuple(selected_edge_ids or ()) if 0 <= int(v) < len(edges)}))
    if not raw_ids:
        return None

    edge_keys = tuple(_edge_key(edges[eid]) for eid in raw_ids)
    median_edge_length = _median_edge_length(verts, edge_keys)
    frame = _coerce_or_fit_frame(
        vertices=verts,
        edge_keys=edge_keys,
        center=center,
        axis=axis,
        radius=radius,
    )
    if frame is None:
        assessment = MeshRealizationAssessment(
            realization_kind=MeshRealizationKind.UNKNOWN,
            topology_quality=0.0,
            edge_fragmentation=1.0,
            closed_loop_quality=0.0,
            polygonality=0.0,
            pollution_score=1.0,
            angular_support_quality=0.0,
            diagnostics={"reason": "opening_frame_fit_failed", "raw_edge_count": len(raw_ids)},
        )
        authority = OpeningFootprintAuthority(
            source=source,
            profile_kind=OpeningProfileKind.UNKNOWN,
            confidence=0.0,
            support_edge_ids=(),
            support_face_ids=tuple(sorted({int(v) for v in tuple(support_face_ids or ()) if int(v) >= 0})),
            contamination_flags=("opening_frame_fit_failed",),
            diagnostics={"not_feature_recognition": True},
        )
        return OpeningEvidenceLedgerData(
            raw_edge_ids=raw_ids,
            selected_authority=authority,
            mesh_realization_assessment=assessment,
            rejected_edge_ids=raw_ids,
            diagnostics={"semantic_stage": "raw_edge_evidence_to_opening_mesh_realization_ledger"},
        )

    samples = _edge_support_samples(
        vertices=verts,
        edge_keys=edge_keys,
        raw_ids=raw_ids,
        frame=frame,
        median_edge_length=median_edge_length,
    )
    if not samples:
        return None

    support = [item for item in samples if item["support"]]
    if len(support) < int(min_support_edges):
        # Keep the best available samples for diagnostics, but mark the authority weak.
        support = sorted(samples, key=lambda item: float(item["score"]), reverse=True)[: max(1, int(min_support_edges))]
        insufficient_support = True
    else:
        insufficient_support = False

    support_ids = tuple(sorted({int(item["edge_id"]) for item in support if int(item["edge_id"]) >= 0}))
    rejected_ids = tuple(sorted(set(raw_ids) - set(support_ids)))
    angular = _angular_statistics([float(item["angle"]) for item in support])
    radii = _radius_statistics(frame=frame, support=support)
    projected = _projected_footprint_statistics(frame=frame, support=support)
    graph = _edge_graph_quality([edge_keys[raw_ids.index(eid)] for eid in raw_ids if eid in raw_ids])

    estimated_n = int(angular.get("estimated_segment_count", 0) or 0)
    observed_ratio = radii.get("observed_min_max_ratio")
    expected_ratio = None
    polygon_delta = None
    polygon_agreement = False
    if estimated_n >= 3 and estimated_n <= 96 and observed_ratio is not None:
        expected_ratio = float(math.cos(math.pi / float(estimated_n)))
        polygon_delta = abs(float(observed_ratio) - expected_ratio)
        tolerance = max(0.06, min(0.16, (1.0 - expected_ratio) * 1.20))
        polygon_agreement = bool(polygon_delta <= tolerance)

    raw_count = len(raw_ids)
    support_count = len(support_ids)
    rejected_count = len(rejected_ids)
    pollution_score = float(rejected_count / max(raw_count, 1))
    angular_coverage = float(angular.get("angular_coverage", 0.0) or 0.0)
    radial_spread_ratio = float(radii.get("radial_spread_ratio", 0.0) or 0.0)
    circle_rel_rms = float(projected.get("circle_rel_rms", 1.0) or 1.0)

    topology_quality = float(max(0.0, min(1.0, graph.get("degree_two_fraction", 0.0))))
    closed_quality = float(max(0.0, min(1.0, graph.get("near_closed_score", 0.0))))
    angular_quality = float(max(0.0, min(1.0, angular_coverage * (1.0 - min(float(angular.get("max_gap_fraction", 1.0)), 1.0) * 0.35))))
    polygonality = 0.0
    if estimated_n >= 3:
        if polygon_agreement:
            polygonality = 1.0
        elif observed_ratio is not None and expected_ratio is not None:
            polygonality = max(0.0, 1.0 - abs(float(observed_ratio) - expected_ratio) / 0.30)

    profile_kind = _classify_opening_profile(
        estimated_n=estimated_n,
        aspect=float(projected.get("aspect", 0.0) or 0.0),
        circle_rel_rms=circle_rel_rms,
        radial_spread_ratio=radial_spread_ratio,
        polygon_agreement=polygon_agreement,
        angular_coverage=angular_coverage,
    )

    contamination_flags: list[str] = []
    if insufficient_support:
        contamination_flags.append("insufficient_support_edges")
    if pollution_score >= 0.55 and raw_count >= 16:
        contamination_flags.append("raw_cloud_contains_many_rejected_edges")
    if observed_ratio is not None and estimated_n >= 3 and expected_ratio is not None and not polygon_agreement and estimated_n <= 12:
        contamination_flags.append("polygon_ratio_disagrees_with_estimated_segment_count")
    if angular_coverage < 0.35:
        contamination_flags.append("low_angular_coverage")
    if radial_spread_ratio > 0.35 and not polygon_agreement:
        contamination_flags.append("wide_radial_envelope_without_polygon_agreement")

    if closed_quality >= 0.80 and radial_spread_ratio <= 0.12:
        realization = MeshRealizationKind.TOPOLOGY_CLOSED_LOOP
    elif polygon_agreement and estimated_n <= 12 and angular_coverage >= 0.45:
        realization = MeshRealizationKind.SPARSE_POLYGONAL
    elif contamination_flags:
        realization = MeshRealizationKind.CONTAMINATED_EDGE_CLOUD
    elif support_count >= int(min_support_edges):
        realization = MeshRealizationKind.VIRTUAL_CONTOUR_SUPPORT
    else:
        realization = MeshRealizationKind.UNKNOWN

    confidence = (
        0.18
        + 0.30 * angular_quality
        + 0.20 * max(0.0, 1.0 - min(circle_rel_rms, 1.0))
        + 0.16 * polygonality
        + 0.10 * closed_quality
        + 0.10 * max(0.0, 1.0 - pollution_score)
    )
    if contamination_flags:
        confidence -= 0.12
    confidence = float(max(0.0, min(0.95, confidence)))

    assessment = MeshRealizationAssessment(
        realization_kind=realization,
        topology_quality=topology_quality,
        edge_fragmentation=float(graph.get("component_fragmentation", 1.0) or 1.0),
        closed_loop_quality=closed_quality,
        polygonality=float(max(0.0, min(1.0, polygonality))),
        pollution_score=float(max(0.0, min(1.0, pollution_score))),
        angular_support_quality=angular_quality,
        diagnostics={
            "raw_edge_count": int(raw_count),
            "support_edge_count": int(support_count),
            "rejected_edge_count": int(rejected_count),
            "median_edge_length": float(median_edge_length),
            "edge_graph_quality": graph,
            "not_feature_recognition": True,
        },
    )

    authority = OpeningFootprintAuthority(
        source=source,
        profile_kind=profile_kind,
        center=_to_vector3(frame["center"]),
        axis=_to_vector3(frame["axis"]),
        radius_min=float(radii["radius_min"]),
        radius_nominal=float(radii["radius_nominal"]),
        radius_max=float(radii["radius_max"]),
        diameter_min=float(2.0 * float(radii["radius_min"])),
        diameter_nominal=float(2.0 * float(radii["radius_nominal"])),
        diameter_max=float(2.0 * float(radii["radius_max"])),
        radial_spread=float(radii["radial_spread"]),
        radial_spread_ratio=float(radial_spread_ratio),
        support_edge_ids=support_ids,
        support_face_ids=tuple(sorted({int(v) for v in tuple(support_face_ids or ()) if int(v) >= 0})),
        virtual_contour_points=_virtual_contour_points(frame, support),
        angular_coverage=angular_coverage,
        max_angular_gap_degrees=float(angular.get("max_angular_gap_degrees", 360.0) or 360.0),
        estimated_segment_count=int(estimated_n),
        expected_polygon_min_max_ratio=expected_ratio,
        observed_min_max_ratio=observed_ratio,
        polygon_model_agreement=bool(polygon_agreement),
        confidence=confidence,
        contamination_flags=tuple(contamination_flags),
        diagnostics={
            "semantic_stage": "mesh_realization_provider_to_opening_footprint_authority",
            "not_feature_recognition": True,
            "radius_model": radii,
            "angular_model": angular,
            "projected_footprint": projected,
            "polygon_ratio_delta": polygon_delta,
            "support_source": "edge_radial_tangential_plane_scoring",
        },
    )

    return OpeningEvidenceLedgerData(
        raw_edge_ids=raw_ids,
        selected_authority=authority,
        mesh_realization_assessment=assessment,
        provider_observations=(
            {
                "provider": "opening_coordinate_footprint_provider",
                "source": str(source),
                "raw_edge_count": int(raw_count),
                "support_edge_count": int(support_count),
                "rejected_edge_count": int(rejected_count),
                "realization_kind": str(realization.value),
                "profile_kind": str(profile_kind.value),
                "confidence": confidence,
            },
        ),
        rejected_edge_ids=rejected_ids,
        diagnostics={
            "semantic_stage": "raw_edge_evidence_to_opening_mesh_realization_ledger",
            "contract": "evidence_only_not_feature_identity_not_rebuild_authority",
            "mesh_realization_provider_version": "v120.opening_footprint_authority.1",
        },
    )


def _as_vertices(vertices: np.ndarray) -> np.ndarray:
    arr = np.asarray(vertices, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.empty((0, 3), dtype=float)
    return arr[:, :3].astype(float, copy=False)


def _as_edges(edges: np.ndarray) -> np.ndarray:
    arr = np.asarray(edges, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.empty((0, 2), dtype=np.int64)
    return arr[:, :2].astype(np.int64, copy=False)


def _edge_key(edge: object) -> EdgeKey:
    a, b = tuple(edge)[:2]  # type: ignore[arg-type]
    ia, ib = int(a), int(b)
    return (ia, ib) if ia <= ib else (ib, ia)


def _to_vector3(value: object) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return (0.0, 0.0, 0.0)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _unit(value: object, fallback: object = (0.0, 0.0, 1.0)) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(3)
    n = float(np.linalg.norm(arr))
    if math.isfinite(n) and n > 1.0e-12:
        return arr / n
    fb = np.asarray(fallback, dtype=float).reshape(3)
    fn = float(np.linalg.norm(fb))
    return fb / fn if math.isfinite(fn) and fn > 1.0e-12 else np.array([0.0, 0.0, 1.0], dtype=float)


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(ref, axis))) > 0.90:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = _unit(np.cross(axis, ref), (0.0, 1.0, 0.0))
    v = _unit(np.cross(axis, u), (1.0, 0.0, 0.0))
    return u, v


def _median_edge_length(vertices: np.ndarray, edge_keys: Iterable[EdgeKey]) -> float:
    vals = []
    for a, b in tuple(edge_keys or ()): 
        if 0 <= int(a) < len(vertices) and 0 <= int(b) < len(vertices):
            length = float(np.linalg.norm(vertices[int(a), :3] - vertices[int(b), :3]))
            if math.isfinite(length) and length > 1.0e-12:
                vals.append(length)
    return float(np.median(np.asarray(vals, dtype=float))) if vals else 1.0


def _coerce_or_fit_frame(*, vertices: np.ndarray, edge_keys: tuple[EdgeKey, ...], center: object | None, axis: object | None, radius: float | None) -> dict[str, np.ndarray | float] | None:
    pts = []
    for a, b in edge_keys:
        if 0 <= int(a) < len(vertices):
            pts.append(vertices[int(a), :3])
        if 0 <= int(b) < len(vertices):
            pts.append(vertices[int(b), :3])
    if len(pts) < 3:
        return None
    arr = np.asarray(pts, dtype=float)
    finite = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite]
    if len(arr) < 3:
        return None

    center_ok = center is not None
    axis_ok = axis is not None
    radius_ok = radius is not None and math.isfinite(float(radius)) and float(radius) > 1.0e-12
    if center_ok and axis_ok and radius_ok:
        c = np.asarray(center, dtype=float).reshape(-1)[:3]
        a = _unit(axis)
        r = float(radius)
        if np.all(np.isfinite(c)):
            return {"center": c.astype(float), "axis": a.astype(float), "radius": float(r), "frame_source": "caller_supplied"}

    c0 = arr.mean(axis=0)
    centered = arr - c0
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        normal = _unit(np.asarray(vh[-1], dtype=float), (0.0, 0.0, 1.0))
    except Exception:
        return None
    u, v = _orthonormal_basis(normal)
    xy = np.column_stack((centered @ u, centered @ v))
    try:
        mat = np.column_stack((xy[:, 0], xy[:, 1], np.ones(len(xy))))
        rhs = -(xy[:, 0] * xy[:, 0] + xy[:, 1] * xy[:, 1])
        sol, *_ = np.linalg.lstsq(mat, rhs, rcond=None)
        cx = -0.5 * float(sol[0])
        cy = -0.5 * float(sol[1])
        radius_sq = max(0.0, cx * cx + cy * cy - float(sol[2]))
        r = float(math.sqrt(radius_sq))
        c = c0 + cx * u + cy * v
    except Exception:
        rel = centered - np.outer(centered @ normal, normal)
        radii = np.linalg.norm(rel, axis=1)
        valid = radii[np.isfinite(radii) & (radii > 1.0e-12)]
        if len(valid) == 0:
            return None
        r = float(np.median(valid))
        c = c0
    if not math.isfinite(r) or r <= 1.0e-12:
        return None
    return {"center": np.asarray(c, dtype=float).reshape(3), "axis": normal, "radius": float(r), "frame_source": "least_squares_opening_frame"}


def _edge_support_samples(*, vertices: np.ndarray, edge_keys: tuple[EdgeKey, ...], raw_ids: tuple[int, ...], frame: Mapping[str, object], median_edge_length: float) -> list[dict[str, object]]:
    center = np.asarray(frame["center"], dtype=float).reshape(3)
    axis = _unit(frame["axis"])
    radius = float(frame["radius"])
    u, v = _orthonormal_basis(axis)
    radial_tol = max(radius * 0.28, median_edge_length * 3.0, 1.0e-9)
    plane_tol = max(radius * 0.18, median_edge_length * 3.0, 1.0e-9)
    lower_segment_band = max(radius * 0.52, 1.0e-12)
    upper_mid_band = max(radius * 1.50, radius + 1.0e-12)
    out: list[dict[str, object]] = []
    for edge_id, key in zip(raw_ids, edge_keys, strict=False):
        a, b = key
        if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
            continue
        p0 = vertices[int(a), :3]
        p1 = vertices[int(b), :3]
        mid = 0.5 * (p0 + p1)
        rel = mid - center
        axial = float(np.dot(rel, axis))
        radial_vec = rel - axis * axial
        radial = float(np.linalg.norm(radial_vec))
        if not math.isfinite(radial) or radial <= 1.0e-12:
            continue
        x, y = float(np.dot(radial_vec, u)), float(np.dot(radial_vec, v))
        angle = float(math.atan2(y, x))
        edge_vec = p1 - p0
        edge_len = float(np.linalg.norm(edge_vec))
        radial_alignment = 0.0
        axial_alignment = 0.0
        tangential_score = 0.0
        if edge_len > 1.0e-12:
            tangent = edge_vec / edge_len
            radial_unit = radial_vec / max(radial, 1.0e-12)
            radial_alignment = abs(float(np.dot(tangent, radial_unit)))
            axial_alignment = abs(float(np.dot(tangent, axis)))
            tangential_score = max(0.0, 1.0 - radial_alignment) * max(0.0, 1.0 - axial_alignment * 0.50)

        q0 = p0 - center
        q1 = p1 - center
        q0 = q0 - axis * float(np.dot(q0, axis))
        q1 = q1 - axis * float(np.dot(q1, axis))
        seg = q1 - q0
        seg_len_sq = float(np.dot(seg, seg))
        seg_dist = radial
        if math.isfinite(seg_len_sq) and seg_len_sq > 1.0e-18:
            t = max(0.0, min(1.0, float(-np.dot(q0, seg) / seg_len_sq)))
            seg_dist = float(np.linalg.norm(q0 + t * seg))
        endpoint_radii = [float(np.linalg.norm(q0)), float(np.linalg.norm(q1))]
        plane_score = max(0.0, 1.0 - abs(axial) / plane_tol)
        radial_score = max(0.0, 1.0 - abs(radial - radius) / radial_tol)
        interior_chord = bool(seg_dist < lower_segment_band)
        spoke_like = bool(radial_alignment > 0.88 and axial_alignment < 0.45)
        out_of_band = bool(radial < lower_segment_band * 0.85 or radial > upper_mid_band)
        support = bool(plane_score > 0.05 and radial_score > 0.05 and not interior_chord and not spoke_like and not out_of_band)
        score = float(0.42 * radial_score + 0.25 * plane_score + 0.23 * tangential_score + 0.10 * max(0.0, min(1.0, seg_dist / max(radius, 1.0e-12))))
        if interior_chord or spoke_like or out_of_band:
            score -= 0.45
        out.append(
            {
                "edge_id": int(edge_id),
                "edge_key": (int(a), int(b)),
                "support": bool(support),
                "score": float(score),
                "angle": float(angle),
                "radial": float(radial),
                "segment_distance": float(seg_dist),
                "endpoint_radius_min": float(min(endpoint_radii)),
                "endpoint_radius_max": float(max(endpoint_radii)),
                "axial_abs": float(abs(axial)),
                "radial_alignment": float(radial_alignment),
                "axial_alignment": float(axial_alignment),
                "tangential_score": float(tangential_score),
                "rejection_flags": tuple(
                    name
                    for name, flag in (
                        ("interior_chord", interior_chord),
                        ("spoke_like", spoke_like),
                        ("out_of_band", out_of_band),
                        ("off_plane", plane_score <= 0.05),
                    )
                    if flag
                ),
            }
        )
    return out


def _angular_statistics(angles: Iterable[float]) -> dict[str, object]:
    vals = np.asarray([float(a) for a in tuple(angles or ()) if math.isfinite(float(a))], dtype=float)
    if vals.size < 2:
        return {
            "angular_coverage": 0.0,
            "max_angular_gap_degrees": 360.0,
            "max_gap_fraction": 1.0,
            "estimated_segment_count": int(vals.size),
            "gap_uniformity": 0.0,
        }
    vals = np.sort(np.mod(vals, 2.0 * math.pi))
    gaps = np.diff(np.concatenate([vals, vals[:1] + 2.0 * math.pi]))
    max_gap = float(np.max(gaps)) if gaps.size else 2.0 * math.pi
    positive = gaps[np.isfinite(gaps) & (gaps > 1.0e-6)]
    median_gap = float(np.median(positive)) if positive.size else max_gap
    estimated_n = int(round((2.0 * math.pi) / max(median_gap, 1.0e-9))) if median_gap > 1.0e-9 else int(vals.size)
    estimated_n = int(max(1, min(256, estimated_n)))
    coverage = float(max(0.0, min(1.0, (2.0 * math.pi - max_gap) / (2.0 * math.pi))))
    gap_uniformity = float(1.0 - min(np.std(positive) / max(float(np.mean(positive)), 1.0e-12), 1.0)) if positive.size >= 2 else 0.0
    return {
        "angular_sample_count": int(vals.size),
        "angular_coverage": coverage,
        "max_angular_gap_degrees": float(math.degrees(max_gap)),
        "max_gap_fraction": float(max_gap / (2.0 * math.pi)),
        "median_gap_degrees": float(math.degrees(median_gap)),
        "estimated_segment_count": int(estimated_n),
        "gap_uniformity": gap_uniformity,
    }


def _radius_statistics(*, frame: Mapping[str, object], support: list[Mapping[str, object]]) -> dict[str, object]:
    nominal = float(frame["radius"])
    lower = np.asarray([float(item.get("segment_distance", item.get("radial", nominal))) for item in support], dtype=float)
    upper = np.asarray([float(item.get("endpoint_radius_max", item.get("radial", nominal))) for item in support], dtype=float)
    mid = np.asarray([float(item.get("radial", nominal)) for item in support], dtype=float)
    lower = lower[np.isfinite(lower) & (lower > 1.0e-12)]
    upper = upper[np.isfinite(upper) & (upper > 1.0e-12)]
    mid = mid[np.isfinite(mid) & (mid > 1.0e-12)]
    radius_min = float(np.percentile(lower, 10.0 if len(lower) >= 8 else 0.0)) if len(lower) else nominal
    radius_max = float(np.percentile(upper, 90.0 if len(upper) >= 8 else 100.0)) if len(upper) else nominal
    radius_nominal = float(np.median(mid)) if len(mid) else nominal
    if radius_min > radius_max:
        radius_min, radius_max = radius_max, radius_min
    spread = float(max(0.0, radius_max - radius_min))
    ratio = float(spread / max(radius_nominal, 1.0e-12))
    observed = float(radius_min / radius_max) if radius_max > 1.0e-12 else None
    return {
        "radius_min": radius_min,
        "radius_nominal": radius_nominal,
        "radius_max": radius_max,
        "radial_spread": spread,
        "radial_spread_ratio": ratio,
        "observed_min_max_ratio": observed,
    }


def _projected_footprint_statistics(*, frame: Mapping[str, object], support: list[Mapping[str, object]]) -> dict[str, object]:
    center = np.asarray(frame["center"], dtype=float).reshape(3)
    axis = _unit(frame["axis"])
    u, v = _orthonormal_basis(axis)
    points = []
    radii = []
    for item in support:
        # Reconstruct an approximate virtual point from angle + radial midpoint.
        theta = float(item.get("angle", 0.0))
        r = float(item.get("radial", frame.get("radius", 0.0)))
        points.append((r * math.cos(theta), r * math.sin(theta)))
        radii.append(r)
    if not points:
        return {"aspect": 0.0, "circle_rel_rms": 1.0, "point_count": 0}
    arr = np.asarray(points, dtype=float)
    spans = np.max(arr, axis=0) - np.min(arr, axis=0)
    width = float(abs(spans[0]))
    height = float(abs(spans[1]))
    aspect = float(max(width, height) / max(min(width, height), 1.0e-12)) if min(width, height) > 1.0e-12 else 999999.0
    rarr = np.asarray(radii, dtype=float)
    rmed = float(np.median(rarr)) if rarr.size else float(frame.get("radius", 0.0))
    rms = float(np.sqrt(np.mean((rarr - rmed) ** 2))) if rarr.size else 0.0
    return {
        "point_count": int(len(points)),
        "width": width,
        "height": height,
        "aspect": aspect,
        "mean_radius": float(np.mean(rarr)) if rarr.size else 0.0,
        "median_radius": rmed,
        "circle_rel_rms": float(rms / max(rmed, 1.0e-12)),
    }


def _classify_opening_profile(*, estimated_n: int, aspect: float, circle_rel_rms: float, radial_spread_ratio: float, polygon_agreement: bool, angular_coverage: float) -> OpeningProfileKind:
    if aspect >= 1.65 and circle_rel_rms >= 0.08:
        return OpeningProfileKind.SLOT_LIKE
    if 5 <= int(estimated_n) <= 7 and 0.72 <= aspect <= 1.38 and polygon_agreement:
        return OpeningProfileKind.HEX_LIKE
    if int(estimated_n) >= 3 and int(estimated_n) <= 12 and polygon_agreement:
        return OpeningProfileKind.POLYGONAL
    if angular_coverage >= 0.65 and circle_rel_rms <= 0.08 and radial_spread_ratio <= 0.14:
        return OpeningProfileKind.CIRCULAR
    if aspect > 1.25 and aspect < 4.50 and circle_rel_rms < 0.20:
        return OpeningProfileKind.ELLIPSE_LIKE
    return OpeningProfileKind.UNKNOWN


def _virtual_contour_points(frame: Mapping[str, object], support: list[Mapping[str, object]], max_points: int = 96) -> tuple[tuple[float, float, float], ...]:
    center = np.asarray(frame["center"], dtype=float).reshape(3)
    axis = _unit(frame["axis"])
    u, v = _orthonormal_basis(axis)
    ordered = sorted(support, key=lambda item: float(item.get("angle", 0.0)))
    if len(ordered) > int(max_points):
        step = max(1, int(math.ceil(len(ordered) / float(max_points))))
        ordered = ordered[::step]
    out = []
    for item in ordered:
        theta = float(item.get("angle", 0.0))
        r = float(item.get("radial", frame.get("radius", 0.0)))
        p = center + u * (r * math.cos(theta)) + v * (r * math.sin(theta))
        out.append(_to_vector3(p))
    return tuple(out)


def _edge_graph_quality(edge_keys: Iterable[EdgeKey]) -> dict[str, float | int | bool]:
    keys = tuple({_edge_key(edge) for edge in tuple(edge_keys or ())})
    if not keys:
        return {"edge_count": 0, "component_count": 0, "degree_two_fraction": 0.0, "near_closed": False, "near_closed_score": 0.0, "component_fragmentation": 1.0}
    vertex_to_edges: dict[int, set[EdgeKey]] = defaultdict(set)
    for key in keys:
        a, b = key
        vertex_to_edges[int(a)].add(key)
        vertex_to_edges[int(b)].add(key)
    degrees = [len(vals) for vals in vertex_to_edges.values()]
    degree_two_fraction = float(sum(1 for d in degrees if d == 2) / max(len(degrees), 1))
    odd_or_tail_fraction = float(sum(1 for d in degrees if d != 2) / max(len(degrees), 1))
    comps = _connected_components(set(keys), vertex_to_edges)
    component_count = len(comps)
    near_closed = bool(component_count == 1 and degree_two_fraction >= 0.88 and len(keys) >= 3)
    near_closed_score = float(max(0.0, min(1.0, degree_two_fraction * (1.0 if component_count == 1 else 0.65))))
    fragmentation = float(max(0.0, min(1.0, (component_count - 1) / max(len(keys), 1))))
    return {
        "edge_count": int(len(keys)),
        "vertex_count": int(len(vertex_to_edges)),
        "component_count": int(component_count),
        "degree_two_fraction": degree_two_fraction,
        "odd_or_tail_fraction": odd_or_tail_fraction,
        "near_closed": near_closed,
        "near_closed_score": near_closed_score,
        "component_fragmentation": fragmentation,
    }


def _connected_components(edges: set[EdgeKey], vertex_to_edges: Mapping[int, set[EdgeKey]]) -> list[set[EdgeKey]]:
    remaining = set(edges)
    comps: list[set[EdgeKey]] = []
    while remaining:
        seed = remaining.pop()
        comp = {seed}
        queue = deque([seed])
        while queue:
            edge = queue.popleft()
            for vertex in edge:
                for nxt in tuple(vertex_to_edges.get(int(vertex), ())):
                    if nxt in remaining:
                        remaining.remove(nxt)
                        comp.add(nxt)
                        queue.append(nxt)
        comps.append(comp)
    return comps


# -----------------------------------------------------------------------------
# X1-style local opening probe ledger
# -----------------------------------------------------------------------------

def build_x1_style_opening_probe_ledger_from_arrays(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray | None = None,
    region_face_ids: Iterable[int] = (),
    seed_face_ids: Iterable[int] = (),
    selected_edge_ids: Iterable[int] = (),
    normalized_edge_ids: Iterable[int] = (),
    edge_index_to_vertices: np.ndarray | None = None,
    center: object | None = None,
    axis: object | None = None,
    radius: float | None = None,
    source: str = "mesh_realization.x1_style_opening_probe_ledger",
    min_layer_samples: int = 8,
) -> dict[str, object] | None:
    """Build an X1-inspired local coaxial probe ledger for a selected opening AOI.

    v133 narrows the v132 probe from the full RegionData volume to a local
    coaxial probe corridor around the resolved opening frame.  This keeps the
    useful X1 meaning transform but avoids treating the whole neutral cutout as
    one bore/pocket body:

        raw/normalized opening evidence + RegionData faces
            -> local coaxial probe corridor
            -> ring/layer/floor relation evidence

    The result is still measurement/evidence only.  It must not declare feature
    identity, assign owned faces, create CandidateData, authorize DeletePatch, or
    mutate mesh topology.
    """

    verts = _as_vertices(vertices)
    face_arr = np.asarray(faces, dtype=np.int64) if faces is not None else np.empty((0, 3), dtype=np.int64)
    if verts.size == 0 or face_arr.ndim != 2 or face_arr.shape[0] == 0 or face_arr.shape[1] < 3:
        return None
    face_arr = face_arr[:, :3]

    valid_region_faces = tuple(sorted({int(fid) for fid in tuple(region_face_ids or ()) if 0 <= int(fid) < len(face_arr)}))
    valid_seed_faces = tuple(sorted({int(fid) for fid in tuple(seed_face_ids or ()) if 0 <= int(fid) < len(face_arr)}))
    broad_probe_face_ids = valid_region_faces or valid_seed_faces
    if not broad_probe_face_ids:
        return None

    if center is None or axis is None or radius is None:
        frame = None
        if edge_index_to_vertices is not None:
            edge_arr = _as_edges(edge_index_to_vertices)
            ids = tuple(int(v) for v in tuple(normalized_edge_ids or selected_edge_ids or ()) if 0 <= int(v) < len(edge_arr))
            if ids:
                keys = tuple(_edge_key(edge_arr[eid]) for eid in ids)
                frame = _coerce_or_fit_frame(vertices=verts, edge_keys=keys, center=center, axis=axis, radius=radius)
        if frame is None:
            return None
        frame_center = np.asarray(frame["center"], dtype=float).reshape(3)
        frame_axis = _unit(frame["axis"])
        frame_radius = float(frame["radius"])
    else:
        frame_center = np.asarray(center, dtype=float).reshape(-1)[:3]
        frame_axis = _unit(axis)
        frame_radius = float(radius)

    if frame_center.size != 3 or not np.all(np.isfinite(frame_center)):
        return None
    if not math.isfinite(frame_radius) or frame_radius <= 1.0e-12:
        return None

    normals = _x1_probe_face_normals(verts, face_arr, face_normals)
    centroids = verts[face_arr[:, :3]].mean(axis=1)

    # Use a local mesh scale, but do not let the coarse-triangle edge length blow
    # the probe radius up to the full RegionData cutout.  v132 used edge_length*4
    # in places, which made the probe absorb remote walls.  v133 keeps edge
    # length as a stabilizer only.
    median_edge_length = _x1_probe_region_median_edge_length(verts, face_arr, broad_probe_face_ids)
    radial_tol = max(frame_radius * 0.34, median_edge_length * 1.25, 1.0e-6)
    sidewall_radial_tol = max(frame_radius * 0.42, median_edge_length * 1.55, 1.0e-6)
    floor_radial_tol = max(frame_radius * 0.18, median_edge_length * 0.85, 1.0e-6)
    mouth_padding = max(frame_radius * 0.18, median_edge_length * 1.25, 1.0e-6)

    local_face_ids, local_diag = _x1_probe_local_face_corridor(
        vertices=verts,
        faces=face_arr,
        centroids=centroids,
        normals=normals,
        region_face_ids=broad_probe_face_ids,
        seed_face_ids=valid_seed_faces,
        frame_center=frame_center,
        frame_axis=frame_axis,
        frame_radius=frame_radius,
        median_edge_length=median_edge_length,
        sidewall_radial_tol=sidewall_radial_tol,
        floor_radial_tol=floor_radial_tol,
        mouth_padding=mouth_padding,
    )
    probe_face_ids = local_face_ids or broad_probe_face_ids

    samples: list[dict[str, object]] = []
    sidewall_face_ids: list[int] = []
    cap_like_face_ids: list[int] = []
    transition_face_ids: list[int] = []
    floor_like_face_ids: list[int] = []
    parent_like_face_ids: list[int] = []

    # v136b fix: the role-island filter must use the exact face metrics from
    # this local probe pass.  v136 accidentally referenced a metrics_cache that
    # only existed inside _x1_probe_local_face_corridor(...), so the probe ledger
    # builder failed and Region Select silently reported the X1 ledger as
    # unavailable.  Keep a local cache here and pass it to the ownership filter.
    metrics_cache: dict[int, dict[str, object]] = {}

    for fid in probe_face_ids:
        metrics = _x1_probe_face_metrics(
            fid=int(fid),
            centroids=centroids,
            normals=normals,
            frame_center=frame_center,
            frame_axis=frame_axis,
            frame_radius=frame_radius,
        )
        if metrics is None:
            continue
        metrics_cache[int(fid)] = dict(metrics)
        radial = float(metrics["radial"])
        radius_delta = float(metrics["radius_delta"])
        normal_axis_abs = float(metrics["normal_axis_abs"])
        radial_normal_align = float(metrics["radial_normal_alignment"])

        role = "context"
        if radius_delta <= sidewall_radial_tol and radial_normal_align >= 0.50 and normal_axis_abs <= 0.78:
            role = "sidewall_like"
            sidewall_face_ids.append(int(fid))
        elif normal_axis_abs >= 0.80 and radial <= frame_radius + floor_radial_tol:
            role = "axis_cap_like"
            cap_like_face_ids.append(int(fid))
        elif radius_delta <= sidewall_radial_tol and radial_normal_align >= 0.28:
            role = "transition_like"
            transition_face_ids.append(int(fid))
        elif radial <= frame_radius + floor_radial_tol:
            role = "near_axis_context"

        metrics["role"] = role
        samples.append(metrics)

    if not samples:
        return None

    sidewall_samples = [item for item in samples if item.get("role") == "sidewall_like"]
    cap_samples = [item for item in samples if item.get("role") == "axis_cap_like"]

    inward_sign = _x1_probe_inward_axis_sign(sidewall_samples=sidewall_samples, cap_samples=cap_samples, mouth_padding=mouth_padding)
    resolved_floor_depth = 0.0
    floor_layer = _x1_probe_first_floor_layer(
        cap_samples=cap_samples,
        inward_sign=inward_sign,
        frame_radius=frame_radius,
        median_edge_length=median_edge_length,
        mouth_padding=mouth_padding,
        min_layer_samples=max(3, int(min_layer_samples) // 2),
    )
    if floor_layer is not None:
        resolved_floor_depth = float(floor_layer.get("depth", 0.0) or 0.0)

    # Now split cap-like samples using the chosen inward direction.  This is the
    # local ring-relation step missing from v132: the cap near the opening is
    # mouth/parent context; the first inward cap layer is floor/opposite support.
    floor_depth_padding = max(frame_radius * 0.18, median_edge_length * 1.25, 1.0e-6)
    for item in cap_samples:
        fid = int(item["face_id"])
        depth = float(item.get("axial", 0.0)) * float(inward_sign)
        item["inward_depth"] = float(depth)
        if depth <= mouth_padding:
            parent_like_face_ids.append(fid)
        elif floor_layer is not None and abs(depth - resolved_floor_depth) <= floor_depth_padding:
            floor_like_face_ids.append(fid)
        elif floor_layer is None and depth > max(frame_radius * 0.35, median_edge_length * 1.5):
            floor_like_face_ids.append(fid)
        else:
            parent_like_face_ids.append(fid)

    # When a floor/end layer exists, restrict side-wall evidence to the physical
    # corridor between the mouth and that first end layer.  This prevents a broad
    # RegionData column from turning into a fake 100-mm sidewall span.
    corridor_max_depth = None
    if resolved_floor_depth > 1.0e-9:
        corridor_max_depth = resolved_floor_depth + floor_depth_padding
        kept_sidewall_ids: list[int] = []
        kept_sidewall_samples: list[Mapping[str, object]] = []
        for item in sidewall_samples:
            depth = float(item.get("axial", 0.0)) * float(inward_sign)
            item["inward_depth"] = float(depth)  # type: ignore[index]
            if -mouth_padding <= depth <= corridor_max_depth:
                kept_sidewall_ids.append(int(item.get("face_id", -1)))
                kept_sidewall_samples.append(item)
        sidewall_face_ids = [fid for fid in kept_sidewall_ids if fid >= 0]
        sidewall_samples = list(kept_sidewall_samples)

    # v136: role-island ownership filter.  v135 isolated the broad probe
    # corridor, but the final sidewall/floor role lists could still contain
    # two disconnected feature-role islands when a polluted raw selection seeded
    # neighboring openings.  CandidateData must receive one physical role island
    # only: connected through sidewall/floor/transition role faces, anchored by
    # the picked seed/opening evidence where possible.
    role_keep, role_filter_diag = _x1_probe_role_island_filter(
        faces=face_arr,
        metrics_cache=metrics_cache,
        sidewall_face_ids=sidewall_face_ids,
        floor_like_face_ids=floor_like_face_ids,
        transition_face_ids=transition_face_ids,
        seed_face_ids=valid_seed_faces,
        frame_radius=frame_radius,
        median_edge_length=median_edge_length,
    )
    if role_keep:
        before_role_count = len(set(sidewall_face_ids) | set(floor_like_face_ids) | set(transition_face_ids))
        sidewall_face_ids = [int(fid) for fid in sidewall_face_ids if int(fid) in role_keep]
        floor_like_face_ids = [int(fid) for fid in floor_like_face_ids if int(fid) in role_keep]
        transition_face_ids = [int(fid) for fid in transition_face_ids if int(fid) in role_keep]
        sidewall_samples = [item for item in sidewall_samples if int(item.get("face_id", -1)) in role_keep]
        # Keep cap samples available for diagnostics, but floor ownership is now
        # restricted by floor_like_face_ids above.
        role_filter_diag["role_filter_before_count"] = int(before_role_count)
        role_filter_diag["role_filter_after_count"] = int(len(role_keep))

    if sidewall_samples:
        sw_depths = np.asarray([float(item.get("axial", 0.0)) * float(inward_sign) for item in sidewall_samples], dtype=float)
        sidewall_depth_min = float(np.min(sw_depths))
        sidewall_depth_max = float(np.max(sw_depths))
        sidewall_span = float(sidewall_depth_max - sidewall_depth_min)
        # Preserve signed raw axial diagnostics too.
        sw_axials = np.asarray([float(item["axial"]) for item in sidewall_samples], dtype=float)
        sidewall_axial_min = float(np.min(sw_axials))
        sidewall_axial_max = float(np.max(sw_axials))
    else:
        sidewall_depth_min = 0.0
        sidewall_depth_max = 0.0
        sidewall_span = 0.0
        sidewall_axial_min = 0.0
        sidewall_axial_max = 0.0

    layers = _x1_probe_sidewall_ring_layers(
        sidewall_samples=sidewall_samples,
        frame_radius=frame_radius,
        median_edge_length=median_edge_length,
        min_layer_samples=int(min_layer_samples),
    )

    axial_layer_count = int(len(layers))
    if axial_layer_count >= 2:
        layer_radii = np.asarray([float(layer["radius_nominal"]) for layer in layers], dtype=float)
        layer_centers = np.asarray([float(layer["axial_center"]) for layer in layers], dtype=float)
        radius_rel_spread = float((np.max(layer_radii) - np.min(layer_radii)) / max(float(np.median(layer_radii)), 1.0e-12))
        layer_span = float(np.max(layer_centers) - np.min(layer_centers))
    else:
        radius_rel_spread = 0.0
        layer_span = 0.0

    floor_support = bool(len(floor_like_face_ids) >= max(3, int(min_layer_samples) // 2))
    sidewall_support = bool(len(sidewall_face_ids) >= max(6, int(min_layer_samples)))
    coaxial_stack_support = bool(axial_layer_count >= 2 and radius_rel_spread <= 0.40 and layer_span > max(median_edge_length * 1.5, frame_radius * 0.10))
    single_opening_only = bool(sidewall_support and not floor_support and not coaxial_stack_support)

    if sidewall_support and floor_support:
        relation_hint = "local_coaxial_recess_or_blind_pocket_probe_support"
    elif sidewall_support and coaxial_stack_support:
        relation_hint = "local_coaxial_sidewall_ring_stack_probe_support"
    elif sidewall_support:
        relation_hint = "local_sidewall_support_without_depth_closure"
    else:
        relation_hint = "insufficient_local_probe_support"

    resolved_depth = 0.0
    if floor_support and resolved_floor_depth > 1.0e-9:
        resolved_depth = float(resolved_floor_depth)
    elif sidewall_support:
        resolved_depth = float(max(0.0, sidewall_depth_max))

    confidence = 0.10
    confidence += 0.30 if sidewall_support else 0.0
    confidence += min(0.20, 0.045 * float(axial_layer_count))
    confidence += 0.16 if coaxial_stack_support else 0.0
    confidence += 0.22 if floor_support else 0.0
    confidence += 0.04 if local_face_ids and len(local_face_ids) < len(broad_probe_face_ids) else 0.0
    confidence -= 0.10 if single_opening_only else 0.0
    confidence = float(max(0.0, min(0.95, confidence)))

    raw_ids = tuple(sorted({int(v) for v in tuple(selected_edge_ids or ()) if int(v) >= 0}))
    norm_ids = tuple(sorted({int(v) for v in tuple(normalized_edge_ids or ()) if int(v) >= 0}))
    raw_to_normalized_ratio = float(len(norm_ids)) / max(float(len(raw_ids)), 1.0) if raw_ids else 1.0

    contradictions: list[str] = []
    if raw_ids and norm_ids and raw_to_normalized_ratio < 0.25:
        contradictions.append("raw_selection_is_much_larger_than_normalized_opening_evidence")
    if len(probe_face_ids) >= len(broad_probe_face_ids) and len(broad_probe_face_ids) > max(50, len(valid_seed_faces) * 4):
        contradictions.append("local_probe_corridor_did_not_reduce_regiondata_scope")
    if sidewall_support and not floor_support and not coaxial_stack_support:
        contradictions.append("local_sidewall_probe_has_no_floor_or_opposite_ring_relation")
    if not sidewall_support:
        contradictions.append("insufficient_local_sidewall_probe_support")

    return {
        "contract_type": "x1_style_opening_probe_ledger",
        "semantic_stage": "raw_or_normalized_opening_evidence_to_local_coaxial_probe_ring_relation_ledger",
        "source": str(source),
        "authority": "measurement_evidence_only",
        "forbidden_authority": (
            "may_not_declare_feature_identity",
            "may_not_assign_surface_ownership",
            "may_not_create_CandidateData",
            "may_not_authorize_DeletePatchProposal",
            "may_not_mutate_mesh_topology",
        ),
        "not_feature_recognition": True,
        "input": {
            "raw_selected_edge_count": int(len(raw_ids)),
            "normalized_edge_count": int(len(norm_ids)),
            "normalized_to_raw_edge_ratio": float(raw_to_normalized_ratio),
            "region_face_count": int(len(valid_region_faces)),
            "seed_face_count": int(len(valid_seed_faces)),
        },
        "frame": {
            "center": _to_vector3(frame_center),
            "axis": _to_vector3(frame_axis),
            "radius": float(frame_radius),
            "diameter": float(2.0 * frame_radius),
            "median_edge_length": float(median_edge_length),
        },
        "probe_counts": {
            "sample_face_count": int(len(samples)),
            "broad_region_face_count": int(len(broad_probe_face_ids)),
            "local_probe_face_count": int(len(probe_face_ids)),
            "local_probe_reduction_count": int(max(0, len(broad_probe_face_ids) - len(probe_face_ids))),
            "sidewall_like_face_count": int(len(sidewall_face_ids)),
            "axis_cap_like_face_count": int(len(cap_like_face_ids)),
            "parent_like_face_count": int(len(parent_like_face_ids)),
            "floor_like_face_count": int(len(floor_like_face_ids)),
            "transition_like_face_count": int(len(transition_face_ids)),
            "axial_layer_count": int(axial_layer_count),
        },
        "ring_layers": tuple(layers),
        "relationships": {
            "probe_scope": "local_coaxial_single_probe_island",
            "sidewall_support": bool(sidewall_support),
            "floor_or_blind_end_support": bool(floor_support),
            "coaxial_ring_stack_support": bool(coaxial_stack_support),
            "single_opening_only": bool(single_opening_only),
            "inward_axis_sign": int(inward_sign),
            "sidewall_axial_min": float(sidewall_axial_min),
            "sidewall_axial_max": float(sidewall_axial_max),
            "sidewall_axial_span": float(sidewall_span),
            "sidewall_depth_min": float(sidewall_depth_min),
            "sidewall_depth_max": float(sidewall_depth_max),
            "sidewall_depth_span": float(sidewall_span),
            "resolved_depth": float(resolved_depth),
            "floor_layer_depth": float(resolved_floor_depth),
            "layer_axial_span": float(layer_span),
            "layer_radius_rel_spread": float(radius_rel_spread),
            "relation_hint": str(relation_hint),
        },
        "support_face_ids": {
            "probe_local": tuple(sorted(set(int(v) for v in probe_face_ids))),
            "sidewall_like": tuple(sorted(set(sidewall_face_ids))),
            "floor_like": tuple(sorted(set(floor_like_face_ids))),
            "parent_like": tuple(sorted(set(parent_like_face_ids))),
            "transition_like": tuple(sorted(set(transition_face_ids))),
        },
        "confidence": float(confidence),
        "contradictions": tuple(contradictions),
        "diagnostics": {
            "probe_version": "v136b.x1_probe_role_island_ownership_filter_runtime_fix.1",
            "classification_note": "relation_hint_is_evidence_only_recognition_must_decide_feature_family",
            "radial_tolerance": float(radial_tol),
            "sidewall_radial_tolerance": float(sidewall_radial_tol),
            "floor_radial_tolerance": float(floor_radial_tol),
            "mouth_padding": float(mouth_padding),
            "corridor": dict(local_diag),
            "role_island_filter": dict(role_filter_diag),
        },
    }


def _x1_probe_face_metrics(
    *,
    fid: int,
    centroids: np.ndarray,
    normals: np.ndarray,
    frame_center: np.ndarray,
    frame_axis: np.ndarray,
    frame_radius: float,
) -> dict[str, object] | None:
    if int(fid) < 0 or int(fid) >= len(centroids):
        return None
    p = np.asarray(centroids[int(fid)], dtype=float).reshape(3)
    n = _unit(normals[int(fid)])
    rel = p - frame_center
    axial = float(np.dot(rel, frame_axis))
    radial_vec = rel - frame_axis * axial
    radial = float(np.linalg.norm(radial_vec))
    if not math.isfinite(radial):
        return None
    radial_unit = radial_vec / max(radial, 1.0e-12) if radial > 1.0e-12 else np.zeros(3, dtype=float)
    normal_axis_abs = abs(float(np.dot(n, frame_axis)))
    radial_normal_align = abs(float(np.dot(n, radial_unit))) if radial > 1.0e-12 else 0.0
    radius_delta = float(abs(radial - frame_radius))
    return {
        "face_id": int(fid),
        "axial": float(axial),
        "radial": float(radial),
        "radius_delta": float(radius_delta),
        "normal_axis_abs": float(normal_axis_abs),
        "radial_normal_alignment": float(radial_normal_align),
    }


def _x1_probe_local_face_corridor(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    centroids: np.ndarray,
    normals: np.ndarray,
    region_face_ids: Iterable[int],
    seed_face_ids: Iterable[int],
    frame_center: np.ndarray,
    frame_axis: np.ndarray,
    frame_radius: float,
    median_edge_length: float,
    sidewall_radial_tol: float,
    floor_radial_tol: float,
    mouth_padding: float,
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Return a connected local coaxial corridor around the picked opening.

    This is deliberately a measurement helper.  It does not own surfaces.  It
    simply prevents the X1-style probe from sampling the entire neutral RegionData
    volume when the picked opening is only a small local recess/pocket.
    """

    region_set = {int(fid) for fid in tuple(region_face_ids or ()) if 0 <= int(fid) < len(faces)}
    seed_set = {int(fid) for fid in tuple(seed_face_ids or ()) if 0 <= int(fid) < len(faces)}
    if not region_set:
        return (), {"reason": "empty_region_face_ids"}

    candidate: set[int] = set()
    rim_band = float(sidewall_radial_tol)
    inner_band = float(floor_radial_tol)
    outer_radius = float(frame_radius + rim_band)

    metrics_cache: dict[int, dict[str, object]] = {}
    for fid in sorted(region_set):
        metrics = _x1_probe_face_metrics(
            fid=fid,
            centroids=centroids,
            normals=normals,
            frame_center=frame_center,
            frame_axis=frame_axis,
            frame_radius=frame_radius,
        )
        if metrics is None:
            continue
        metrics_cache[int(fid)] = metrics
        radial = float(metrics["radial"])
        radius_delta = float(metrics["radius_delta"])
        normal_axis_abs = float(metrics["normal_axis_abs"])
        radial_align = float(metrics["radial_normal_alignment"])

        near_sidewall_band = radius_delta <= rim_band and radial_align >= 0.18
        inside_opening_disk = radial <= frame_radius + inner_band
        cap_inside_opening = inside_opening_disk and normal_axis_abs >= 0.55
        near_core_context = radial <= max(frame_radius * 0.80, frame_radius - inner_band)
        if (radial <= outer_radius and (near_sidewall_band or cap_inside_opening or near_core_context)):
            candidate.add(int(fid))

    if not candidate:
        return (), {
            "reason": "no_faces_passed_radial_corridor",
            "region_face_count": int(len(region_set)),
            "sidewall_radial_tolerance": float(sidewall_radial_tol),
            "floor_radial_tolerance": float(floor_radial_tol),
        }

    adjacency = _x1_probe_face_adjacency(faces, candidate)
    seed_candidates = tuple(sorted(seed_set & candidate))
    if not seed_candidates:
        # Fall back to the faces closest to the opening rim/axis.  This covers
        # cases where the raw seed came from a polluted edge cloud and its direct
        # adjacent faces were not inside the final radial corridor.  v135 still
        # treats these as seed *hints* only; it does not allow several disconnected
        # hint components to merge into one ownership/probe island.
        scored = []
        for fid in candidate:
            m = metrics_cache.get(int(fid))
            if m is None:
                continue
            score = abs(float(m["radius_delta"])) / max(frame_radius, 1.0e-12) + abs(float(m["axial"])) / max(frame_radius, median_edge_length, 1.0e-12) * 0.35
            scored.append((float(score), int(fid)))
        seed_candidates = tuple(fid for _score, fid in sorted(scored)[: max(1, min(12, len(scored)))])

    # v135: isolate exactly one local probe island.  The v134 corridor could
    # return the union of multiple disconnected radial components when the raw
    # selection cloud touched neighboring features.  That made Recognition own
    # faces from two different pockets.  Here the radial corridor is split into
    # connected components and only the best opening-connected component may
    # continue to role measurement.
    components = _x1_probe_connected_components(adjacency=adjacency, allowed=candidate)
    if not components:
        return (), {
            "reason": "no_connected_components_in_radial_corridor",
            "region_face_count": int(len(region_set)),
            "candidate_face_count": int(len(candidate)),
            "seed_candidate_count": int(len(seed_candidates)),
        }

    seed_candidate_set = {int(v) for v in tuple(seed_candidates or ())}
    component_infos: list[dict[str, object]] = []
    for index, component in enumerate(components):
        comp_metrics = [metrics_cache[fid] for fid in sorted(component) if fid in metrics_cache]
        sidewall_like = [m for m in comp_metrics if float(m["radius_delta"]) <= rim_band and float(m["radial_normal_alignment"]) >= 0.36 and float(m["normal_axis_abs"]) <= 0.82]
        cap_like = [m for m in comp_metrics if float(m["normal_axis_abs"]) >= 0.78 and float(m["radial"]) <= frame_radius + inner_band]
        comp_inward_sign = _x1_probe_inward_axis_sign(sidewall_samples=sidewall_like, cap_samples=cap_like, mouth_padding=mouth_padding)
        comp_floor_layer = _x1_probe_first_floor_layer(
            cap_samples=cap_like,
            inward_sign=comp_inward_sign,
            frame_radius=frame_radius,
            median_edge_length=median_edge_length,
            mouth_padding=mouth_padding,
            min_layer_samples=3,
        )
        if comp_metrics:
            radial_error = float(np.median(np.asarray([abs(float(m["radius_delta"])) for m in comp_metrics], dtype=float))) / max(frame_radius, 1.0e-12)
            axial_error = float(np.median(np.asarray([abs(float(m["axial"])) for m in comp_metrics], dtype=float))) / max(frame_radius, median_edge_length, 1.0e-12)
        else:
            radial_error = 999.0
            axial_error = 999.0
        seed_hits = len(set(component) & seed_candidate_set)
        floor_count = len(tuple(comp_floor_layer.get("face_ids", ()) or ())) if comp_floor_layer is not None else 0
        score = (
            38.0 * float(seed_hits)
            + 2.25 * float(min(len(sidewall_like), 80))
            + 5.50 * float(min(floor_count, 40))
            + (30.0 if comp_floor_layer is not None else 0.0)
            + 0.08 * float(min(len(component), 220))
            - 26.0 * float(radial_error)
            - 4.0 * float(axial_error)
        )
        component_infos.append({
            "index": int(index),
            "face_ids": tuple(sorted(component)),
            "face_count": int(len(component)),
            "seed_hit_count": int(seed_hits),
            "sidewall_like_count": int(len(sidewall_like)),
            "floor_like_count": int(floor_count),
            "has_floor_layer": bool(comp_floor_layer is not None),
            "floor_depth": float(comp_floor_layer.get("depth", 0.0)) if comp_floor_layer is not None else 0.0,
            "radial_error": float(radial_error),
            "axial_error": float(axial_error),
            "score": float(score),
            "inward_axis_sign": int(comp_inward_sign),
        })

    component_infos.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    selected_info = component_infos[0]
    connected = {int(v) for v in tuple(selected_info.get("face_ids", ()) or ())}
    component_source = "v135_selected_single_probe_island"

    # Directional crop: if a floor/end cap can be found close to the seed, keep
    # only the mouth-to-first-floor part of the selected probe island.
    comp_metrics = [metrics_cache[fid] for fid in sorted(connected) if fid in metrics_cache]
    sidewall_like = [m for m in comp_metrics if float(m["radius_delta"]) <= rim_band and float(m["radial_normal_alignment"]) >= 0.36 and float(m["normal_axis_abs"]) <= 0.82]
    cap_like = [m for m in comp_metrics if float(m["normal_axis_abs"]) >= 0.78 and float(m["radial"]) <= frame_radius + inner_band]
    inward_sign = _x1_probe_inward_axis_sign(sidewall_samples=sidewall_like, cap_samples=cap_like, mouth_padding=mouth_padding)
    floor_layer = _x1_probe_first_floor_layer(
        cap_samples=cap_like,
        inward_sign=inward_sign,
        frame_radius=frame_radius,
        median_edge_length=median_edge_length,
        mouth_padding=mouth_padding,
        min_layer_samples=3,
    )
    floor_depth = float(floor_layer.get("depth", 0.0)) if floor_layer is not None else 0.0
    directional_cropped = False
    crop_component_count = 0
    crop_rejected_face_count = 0
    if floor_depth > 1.0e-9:
        max_depth = floor_depth + max(frame_radius * 0.20, median_edge_length * 1.50)
        min_depth = -max(mouth_padding, median_edge_length)
        cropped = {
            int(m["face_id"])
            for m in comp_metrics
            if min_depth <= float(m.get("axial", 0.0)) * float(inward_sign) <= max_depth
        }
        if len(cropped) >= max(8, min(len(connected), len(seed_candidates))):
            crop_adjacency = _x1_probe_face_adjacency(faces, cropped)
            crop_components = _x1_probe_connected_components(adjacency=crop_adjacency, allowed=cropped)
            crop_component_count = int(len(crop_components))
            if crop_components:
                # Prefer the crop component that overlaps the selected seed hints;
                # otherwise keep the largest crop piece.  Do not re-union split
                # crop islands.
                crop_seed_set = seed_candidate_set & cropped
                crop_components.sort(key=lambda comp: (len(set(comp) & crop_seed_set), len(comp)), reverse=True)
                best_crop = set(crop_components[0])
                crop_rejected_face_count = int(max(0, len(cropped) - len(best_crop)))
                connected = best_crop
            else:
                connected = cropped
            directional_cropped = True

    selected_component_index = int(selected_info.get("index", -1))
    rejected_component_face_count = int(max(0, len(candidate) - len(connected)))
    component_summaries = tuple(
        {
            "index": int(row.get("index", -1)),
            "face_count": int(row.get("face_count", 0)),
            "seed_hit_count": int(row.get("seed_hit_count", 0)),
            "sidewall_like_count": int(row.get("sidewall_like_count", 0)),
            "floor_like_count": int(row.get("floor_like_count", 0)),
            "has_floor_layer": bool(row.get("has_floor_layer", False)),
            "floor_depth": float(row.get("floor_depth", 0.0)),
            "score": float(row.get("score", 0.0)),
        }
        for row in component_infos[:8]
    )

    return tuple(sorted(connected)), {
        "region_face_count": int(len(region_set)),
        "candidate_face_count": int(len(candidate)),
        "seed_candidate_count": int(len(seed_candidates)),
        "connected_face_count": int(len(connected)),
        "component_source": str(component_source),
        "component_count": int(len(components)),
        "selected_component_index": int(selected_component_index),
        "selected_component_score": float(selected_info.get("score", 0.0)),
        "selected_component_seed_hit_count": int(selected_info.get("seed_hit_count", 0)),
        "selected_component_face_count_before_crop": int(selected_info.get("face_count", 0)),
        "rejected_component_face_count": int(rejected_component_face_count),
        "component_summaries": component_summaries,
        "directional_cropped_to_first_floor": bool(directional_cropped),
        "crop_component_count": int(crop_component_count),
        "crop_rejected_face_count": int(crop_rejected_face_count),
        "inward_axis_sign": int(inward_sign),
        "first_floor_depth": float(floor_depth),
        "sidewall_radial_tolerance": float(sidewall_radial_tol),
        "floor_radial_tolerance": float(floor_radial_tol),
        "mouth_padding": float(mouth_padding),
    }



def _x1_probe_role_island_filter(
    *,
    faces: np.ndarray,
    metrics_cache: Mapping[int, Mapping[str, object]],
    sidewall_face_ids: Iterable[int],
    floor_like_face_ids: Iterable[int],
    transition_face_ids: Iterable[int],
    seed_face_ids: Iterable[int],
    frame_radius: float,
    median_edge_length: float,
) -> tuple[set[int], dict[str, object]]:
    """Select one connected physical role island from probe role evidence.

    The broad/local probe corridor may be connected through parent/context faces.
    Rebuildable POCKET evidence must not be unioned through that context.  This
    helper builds connectivity only over role faces that may become part of the
    pocket candidate (sidewall, floor, transition) and keeps exactly one island.

    This is still measurement evidence.  It does not authorize rebuild by itself;
    it only prevents Recognition from receiving a multi-feature role union.
    """

    side_set = {int(fid) for fid in tuple(sidewall_face_ids or ()) if 0 <= int(fid) < len(faces)}
    floor_set = {int(fid) for fid in tuple(floor_like_face_ids or ()) if 0 <= int(fid) < len(faces)}
    trans_set = {int(fid) for fid in tuple(transition_face_ids or ()) if 0 <= int(fid) < len(faces)}
    role_set = set(side_set) | set(floor_set) | set(trans_set)
    seed_set = {int(fid) for fid in tuple(seed_face_ids or ()) if 0 <= int(fid) < len(faces)}
    if not role_set:
        return set(), {
            "role_island_filter_used": False,
            "role_island_filter_reason": "no_role_faces",
            "role_component_count": 0,
        }

    adjacency = _x1_probe_face_adjacency(faces, role_set)
    components = _x1_probe_connected_components(adjacency=adjacency, allowed=role_set)
    if not components:
        return set(role_set), {
            "role_island_filter_used": False,
            "role_island_filter_reason": "no_role_components",
            "role_component_count": 0,
            "role_face_count": int(len(role_set)),
        }
    if len(components) == 1:
        return set(components[0]), {
            "role_island_filter_used": True,
            "role_island_filter_reason": "single_role_component",
            "role_component_count": 1,
            "selected_role_component_index": 0,
            "selected_role_component_face_count": int(len(components[0])),
            "rejected_role_face_count": 0,
            "role_sidewall_count_before": int(len(side_set)),
            "role_floor_count_before": int(len(floor_set)),
            "role_transition_count_before": int(len(trans_set)),
        }

    rows: list[dict[str, object]] = []
    for idx, comp in enumerate(components):
        comp_set = set(int(v) for v in comp)
        side_count = len(comp_set & side_set)
        floor_count = len(comp_set & floor_set)
        trans_count = len(comp_set & trans_set)
        seed_hits = len(comp_set & seed_set)
        metrics = [metrics_cache[fid] for fid in sorted(comp_set) if fid in metrics_cache]
        if metrics:
            radial_error = float(np.median(np.asarray([abs(float(m.get("radius_delta", 999.0))) for m in metrics], dtype=float))) / max(float(frame_radius), 1.0e-12)
            axial_error = float(np.median(np.asarray([abs(float(m.get("axial", 0.0))) for m in metrics], dtype=float))) / max(float(frame_radius), float(median_edge_length), 1.0e-12)
            mouth_contact = sum(1 for m in metrics if abs(float(m.get("axial", 0.0))) <= max(float(frame_radius) * 0.32, float(median_edge_length) * 1.5))
        else:
            radial_error = 999.0
            axial_error = 999.0
            mouth_contact = 0
        # Prefer the role island attached to the picked opening and with both
        # wall and floor support.  Do not reward raw face-count enough to let a
        # large neighboring feature win by size alone.
        score = (
            72.0 * float(seed_hits)
            + 6.0 * float(min(side_count, 30))
            + 16.0 * float(min(floor_count, 18))
            + 2.0 * float(min(trans_count, 12))
            + 5.0 * float(min(mouth_contact, 12))
            + (34.0 if side_count >= 6 and floor_count >= 3 else 0.0)
            - 34.0 * float(radial_error)
            - 6.0 * float(axial_error)
        )
        rows.append({
            "index": int(idx),
            "face_ids": tuple(sorted(comp_set)),
            "face_count": int(len(comp_set)),
            "sidewall_count": int(side_count),
            "floor_count": int(floor_count),
            "transition_count": int(trans_count),
            "seed_hit_count": int(seed_hits),
            "mouth_contact_count": int(mouth_contact),
            "radial_error": float(radial_error),
            "axial_error": float(axial_error),
            "score": float(score),
        })

    rows.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    selected = rows[0]
    keep = {int(v) for v in tuple(selected.get("face_ids", ()) or ())}
    rejected = int(max(0, len(role_set) - len(keep)))
    return keep, {
        "role_island_filter_used": True,
        "role_island_filter_reason": "selected_best_seed_anchored_role_component",
        "role_component_count": int(len(components)),
        "selected_role_component_index": int(selected.get("index", -1)),
        "selected_role_component_score": float(selected.get("score", 0.0)),
        "selected_role_component_seed_hit_count": int(selected.get("seed_hit_count", 0)),
        "selected_role_component_face_count": int(len(keep)),
        "rejected_role_face_count": int(rejected),
        "role_sidewall_count_before": int(len(side_set)),
        "role_floor_count_before": int(len(floor_set)),
        "role_transition_count_before": int(len(trans_set)),
        "role_component_summaries": tuple(
            {
                "index": int(row.get("index", -1)),
                "face_count": int(row.get("face_count", 0)),
                "sidewall_count": int(row.get("sidewall_count", 0)),
                "floor_count": int(row.get("floor_count", 0)),
                "transition_count": int(row.get("transition_count", 0)),
                "seed_hit_count": int(row.get("seed_hit_count", 0)),
                "score": float(row.get("score", 0.0)),
            }
            for row in rows[:8]
        ),
    }


def _x1_probe_face_adjacency(faces: np.ndarray, face_ids: Iterable[int]) -> dict[int, tuple[int, ...]]:
    selected = {int(fid) for fid in tuple(face_ids or ()) if 0 <= int(fid) < len(faces)}
    edge_to_faces: dict[EdgeKey, list[int]] = defaultdict(list)
    for fid in selected:
        tri = faces[int(fid), :3]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = _edge_key((int(a), int(b)))
            edge_to_faces[key].append(int(fid))
    adjacency: dict[int, set[int]] = {int(fid): set() for fid in selected}
    for ids in edge_to_faces.values():
        if len(ids) < 2:
            continue
        for fid in ids:
            adjacency.setdefault(int(fid), set()).update(int(v) for v in ids if int(v) != int(fid))
    return {int(fid): tuple(sorted(vals)) for fid, vals in adjacency.items()}


def _x1_probe_connected_component(*, adjacency: Mapping[int, Iterable[int]], seeds: Iterable[int], allowed: set[int]) -> set[int]:
    out: set[int] = set()
    queue: deque[int] = deque()
    for seed in tuple(seeds or ()):
        sid = int(seed)
        if sid in allowed and sid not in out:
            out.add(sid)
            queue.append(sid)
    while queue:
        fid = queue.popleft()
        for nxt in tuple(adjacency.get(int(fid), ())):
            nid = int(nxt)
            if nid in allowed and nid not in out:
                out.add(nid)
                queue.append(nid)
    return out


def _x1_probe_connected_components(*, adjacency: Mapping[int, Iterable[int]], allowed: set[int]) -> list[set[int]]:
    """Return disjoint connected components within a probe candidate set.

    v135 uses this to keep X1-style probe ownership tied to one physical
    opening island.  A raw Ctrl-click cloud may touch several disconnected
    features; those components are evidence alternatives, not one candidate.
    """

    remaining = {int(fid) for fid in set(allowed)}
    components: list[set[int]] = []
    while remaining:
        seed = int(next(iter(remaining)))
        comp: set[int] = {seed}
        remaining.remove(seed)
        queue: deque[int] = deque([seed])
        while queue:
            fid = queue.popleft()
            for nxt in tuple(adjacency.get(int(fid), ())):
                nid = int(nxt)
                if nid in remaining:
                    remaining.remove(nid)
                    comp.add(nid)
                    queue.append(nid)
        components.append(comp)
    components.sort(key=lambda item: len(item), reverse=True)
    return components


def _x1_probe_inward_axis_sign(*, sidewall_samples: Iterable[Mapping[str, object]], cap_samples: Iterable[Mapping[str, object]], mouth_padding: float) -> int:
    values = [float(item.get("axial", 0.0)) for item in tuple(sidewall_samples or ()) if math.isfinite(float(item.get("axial", 0.0)))]
    if not values:
        values = [float(item.get("axial", 0.0)) for item in tuple(cap_samples or ()) if math.isfinite(float(item.get("axial", 0.0))) and abs(float(item.get("axial", 0.0))) > float(mouth_padding)]
    if not values:
        return 1
    med = float(np.median(np.asarray(values, dtype=float)))
    return 1 if med >= 0.0 else -1


def _x1_probe_first_floor_layer(
    *,
    cap_samples: Iterable[Mapping[str, object]],
    inward_sign: int,
    frame_radius: float,
    median_edge_length: float,
    mouth_padding: float,
    min_layer_samples: int,
) -> dict[str, object] | None:
    items: list[tuple[float, int]] = []
    min_depth = max(float(mouth_padding), frame_radius * 0.22, median_edge_length * 1.0)
    for item in tuple(cap_samples or ()): 
        try:
            depth = float(item.get("axial", 0.0)) * float(inward_sign)
            fid = int(item.get("face_id", -1))
        except Exception:
            continue
        if fid < 0 or not math.isfinite(depth) or depth < min_depth:
            continue
        items.append((float(depth), int(fid)))
    if len(items) < max(2, int(min_layer_samples)):
        return None
    depths = np.asarray([d for d, _fid in items], dtype=float)
    dmin = float(np.min(depths))
    dmax = float(np.max(depths))
    span = float(max(0.0, dmax - dmin))
    bin_width = max(float(median_edge_length) * 1.75, float(frame_radius) * 0.16, 1.0e-6)
    bin_count = int(max(1, min(48, math.ceil(max(span, bin_width) / bin_width))))
    hist, edges = np.histogram(depths, bins=bin_count, range=(dmin, max(dmax, dmin + bin_width)))
    best: dict[str, object] | None = None
    for idx in range(int(hist.size)):
        count = int(hist[idx])
        if count < max(2, int(min_layer_samples)):
            continue
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        layer_items = [(d, fid) for d, fid in items if lo <= d <= hi]
        if len(layer_items) < max(2, int(min_layer_samples)):
            continue
        layer_depths = np.asarray([d for d, _fid in layer_items], dtype=float)
        layer = {
            "depth": float(np.median(layer_depths)),
            "depth_min": float(np.min(layer_depths)),
            "depth_max": float(np.max(layer_depths)),
            "sample_count": int(len(layer_items)),
            "face_ids": tuple(sorted({int(fid) for _d, fid in layer_items})),
        }
        if best is None or float(layer["depth"]) < float(best["depth"]):
            best = layer
    return best


def _x1_probe_face_normals(vertices: np.ndarray, faces: np.ndarray, provided: np.ndarray | None) -> np.ndarray:
    if provided is not None:
        try:
            arr = np.asarray(provided, dtype=float)
            if arr.shape == (len(faces), 3):
                return arr.astype(float, copy=False)
        except Exception:
            pass
    tri = vertices[faces[:, :3], :3]
    raw = np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :])
    lengths = np.linalg.norm(raw, axis=1)
    out = np.zeros_like(raw)
    ok = np.isfinite(lengths) & (lengths > 1.0e-12)
    out[ok] = raw[ok] / lengths[ok].reshape(-1, 1)
    return out


def _x1_probe_region_median_edge_length(vertices: np.ndarray, faces: np.ndarray, face_ids: Iterable[int]) -> float:
    vals: list[float] = []
    for fid in tuple(face_ids or ()): 
        if int(fid) < 0 or int(fid) >= len(faces):
            continue
        tri = faces[int(fid), :3]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            ia, ib = int(a), int(b)
            if 0 <= ia < len(vertices) and 0 <= ib < len(vertices):
                length = float(np.linalg.norm(vertices[ia, :3] - vertices[ib, :3]))
                if math.isfinite(length) and length > 1.0e-12:
                    vals.append(length)
    return float(np.median(np.asarray(vals, dtype=float))) if vals else 1.0


def _x1_probe_sidewall_ring_layers(
    *,
    sidewall_samples: list[Mapping[str, object]],
    frame_radius: float,
    median_edge_length: float,
    min_layer_samples: int,
) -> tuple[dict[str, object], ...]:
    if len(sidewall_samples) < max(4, int(min_layer_samples)):
        return ()
    axials = np.asarray([float(item.get("axial", 0.0)) for item in sidewall_samples], dtype=float)
    if axials.size == 0 or not np.all(np.isfinite(axials)):
        return ()
    amin = float(np.min(axials))
    amax = float(np.max(axials))
    span = float(amax - amin)
    if span <= 1.0e-12:
        radii = np.asarray([float(item.get("radial", frame_radius)) for item in sidewall_samples], dtype=float)
        return (
            {
                "layer_index": 0,
                "axial_center": float(np.median(axials)),
                "axial_min": float(amin),
                "axial_max": float(amax),
                "radius_nominal": float(np.median(radii)),
                "radius_min": float(np.min(radii)),
                "radius_max": float(np.max(radii)),
                "sample_count": int(len(sidewall_samples)),
                "face_ids": tuple(sorted({int(item.get("face_id", -1)) for item in sidewall_samples if int(item.get("face_id", -1)) >= 0})),
            },
        )

    bin_width = max(float(median_edge_length) * 3.0, float(frame_radius) * 0.18, 1.0e-6)
    bin_count = int(max(2, min(48, math.ceil(span / bin_width))))
    hist, edges = np.histogram(axials, bins=bin_count, range=(amin, amax))
    layers: list[dict[str, object]] = []
    for bin_index in range(int(hist.size)):
        if int(hist[bin_index]) < max(4, int(min_layer_samples)):
            continue
        lo = float(edges[bin_index])
        hi = float(edges[bin_index + 1])
        items = [item for item in sidewall_samples if lo <= float(item.get("axial", 0.0)) <= hi]
        if len(items) < max(4, int(min_layer_samples)):
            continue
        radii = np.asarray([float(item.get("radial", frame_radius)) for item in items], dtype=float)
        layer_axials = np.asarray([float(item.get("axial", 0.0)) for item in items], dtype=float)
        layers.append(
            {
                "layer_index": int(len(layers)),
                "axial_center": float(np.median(layer_axials)),
                "axial_min": float(np.min(layer_axials)),
                "axial_max": float(np.max(layer_axials)),
                "radius_nominal": float(np.median(radii)),
                "radius_min": float(np.min(radii)),
                "radius_max": float(np.max(radii)),
                "radius_rel_spread": float((np.max(radii) - np.min(radii)) / max(float(np.median(radii)), 1.0e-12)),
                "sample_count": int(len(items)),
                "face_ids": tuple(sorted({int(item.get("face_id", -1)) for item in items if int(item.get("face_id", -1)) >= 0})),
            }
        )
    return tuple(layers)


__all__ = ["build_opening_evidence_ledger_from_arrays", "build_x1_style_opening_probe_ledger_from_arrays"]
