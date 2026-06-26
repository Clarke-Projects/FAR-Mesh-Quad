"""Mesh-native Bore measurement layer.

This module is non-mutating measurement code only. It does not select faces,
recognize features, build rebuild targets, mutate meshes, touch the viewport,
or own GUI state.

Naming policy
-------------
``region_*`` names mean neutral RegionData / selected mesh cutout evidence:
the WHERE evidence passed from Region Select into Recognition.

``bore_*`` names mean bore-family measurement evidence: opening fit, cylindrical
wall consistency, radius/depth estimates, and diagnostics that help Recognition
or Rebuild reason about BOREHOLE candidates.

``chamfer_*`` should be used by future chamfer-family measurement helpers. This
file currently contains bore-opening and bore-region measurement only.

The important boundary is that RegionData faces may be measured as possible
bore-family evidence, but measurement never promotes them into CandidateData and
never turns them into DeletePatchProposal or RebuildResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import trimesh

from .geometry import (
    RingFit,
    Vector3,
    axis_hint,
    canonical_axis,
    clamp,
    fit_ring_points,
    median,
    median_abs_deviation,
    to_vector3,
    unit_vector,
)
from .topology import connected_edge_components, edge_graph_stats, face_edges, normalize_edge
from .types import tuple_ints

EdgeKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class BoreOpeningRadialEnvelope:
    """Unified radial-envelope measurement for every bore opening.

    This is not a coarse-mesh special case.  Every measured opening has a
    radial envelope.  Dense circular rims naturally produce a narrow envelope
    where min/nominal/max are almost identical.  Coarse polygonal rims produce a
    wider envelope where the inward flats/segments and outward corner vertices
    are both preserved as measurement evidence.

    ``radius`` on older contracts remains the nominal fitted radius.  The
    envelope makes the lower and upper support explicit without changing the
    semantic stage: selected rim evidence -> opening measurement evidence.
    """

    center: Vector3
    axis: Vector3
    radius_min: float
    radius_nominal: float
    radius_max: float
    diameter_min: float
    diameter_nominal: float
    diameter_max: float
    radial_spread: float
    radial_spread_ratio: float
    vertex_sample_count: int
    edge_midpoint_sample_count: int
    segment_distance_sample_count: int
    coarse_polygonal: bool
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_type": "bore_opening_radial_envelope",
            "semantic_stage": "rim_evidence_to_unified_opening_radial_envelope",
            "not_coarse_mesh_special_case": True,
            "center": self.center,
            "axis": self.axis,
            "radius_min": float(self.radius_min),
            "radius_nominal": float(self.radius_nominal),
            "radius_max": float(self.radius_max),
            "diameter_min": float(self.diameter_min),
            "diameter_nominal": float(self.diameter_nominal),
            "diameter_max": float(self.diameter_max),
            "radial_spread": float(self.radial_spread),
            "radial_spread_ratio": float(self.radial_spread_ratio),
            "vertex_sample_count": int(self.vertex_sample_count),
            "edge_midpoint_sample_count": int(self.edge_midpoint_sample_count),
            "segment_distance_sample_count": int(self.segment_distance_sample_count),
            "coarse_polygonal": bool(self.coarse_polygonal),
            "diagnostics": dict(self.diagnostics or {}),
        }


@dataclass(frozen=True, slots=True)
class BoreOpeningMeasurement:
    """Measurement of one selected/near-selected bore opening."""

    edge_ids: tuple[int, ...]
    edge_count: int
    vertex_ids: tuple[int, ...]
    vertex_count: int
    center: Vector3
    axis: Vector3
    radius: float
    diameter: float
    closed: bool
    near_closed: bool
    endpoint_gap: float
    endpoint_gap_ratio: float
    branch_vertex_count: int
    open_endpoint_count: int
    component_count: int
    plane_rms: float
    plane_rel_rms: float
    radius_rms: float
    radius_rel_rms: float
    radius_mad: float
    circularity: float
    confidence: float
    radius_min: float | None = None
    radius_max: float | None = None
    diameter_min: float | None = None
    diameter_max: float | None = None
    radial_spread: float | None = None
    radial_spread_ratio: float | None = None
    coarse_polygonal: bool = False
    radial_envelope: Mapping[str, object] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BoreRegionMeasurement:
    """Bore-family measurement derived from RegionData/candidate region faces.

    The input faces are still region evidence. The bore-specific fields below
    are measurement results, not recognition ownership and not rebuild targets.
    """

    opening: BoreOpeningMeasurement
    region_face_ids: tuple[int, ...]
    region_face_count: int
    center: Vector3
    axis: Vector3
    radius: float
    diameter: float
    depth: float
    opening_center: Vector3
    opposite_center: Vector3
    axial_min: float
    axial_max: float
    normal_axis_abs_median: float
    radial_mad: float
    radial_rel_mad: float
    confidence: float
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BoreTwoOpeningMeasurement:
    """Measured BORE frame from selected opening A and opposite opening B.

    This is measurement evidence only.  It does not own wall faces, does not
    create CandidateData, and does not authorize rebuild.  Its purpose is to
    make the intended BORE order explicit:

        selected opening -> opposite opening -> refined bore frame -> wall search
    """

    selected_opening: BoreOpeningMeasurement
    opposite_opening: BoreOpeningMeasurement | None
    valid: bool
    center: Vector3
    axis: Vector3
    radius: float
    diameter: float
    depth: float
    opening_center: Vector3
    opposite_center: Vector3
    axial_min: float
    axial_max: float
    confidence: float
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": bool(self.valid),
            "center": self.center,
            "axis": self.axis,
            "radius": float(self.radius),
            "diameter": float(self.diameter),
            "depth": float(self.depth),
            "opening_center": self.opening_center,
            "opposite_center": self.opposite_center,
            "axial_min": float(self.axial_min),
            "axial_max": float(self.axial_max),
            "confidence": float(self.confidence),
            "selected_opening": _opening_measurement_to_dict(self.selected_opening),
            "opposite_opening": _opening_measurement_to_dict(self.opposite_opening) if self.opposite_opening is not None else {},
            "diagnostics": dict(self.diagnostics or {}),
        }


# -----------------------------------------------------------------------------
# Public measurement API
# -----------------------------------------------------------------------------


def _projected_segment_distance_to_center(
    *,
    vertices: np.ndarray,
    edge: EdgeKey,
    center: np.ndarray,
    axis: np.ndarray,
) -> float | None:
    """Return center-to-edge distance in the opening plane.

    For a coarse polygonal opening this distance is the important lower radius:
    the segment/flat protruding into the opening can be closer to the center
    than the corner vertices.  This is the geometric source of the min diameter.
    """

    try:
        a, b = normalize_edge(edge)
    except Exception:
        return None
    if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
        return None
    p0 = np.asarray(vertices[int(a), :3], dtype=float).reshape(3) - center
    p1 = np.asarray(vertices[int(b), :3], dtype=float).reshape(3) - center
    p0 = p0 - axis * float(np.dot(p0, axis))
    p1 = p1 - axis * float(np.dot(p1, axis))
    seg = p1 - p0
    seg_len_sq = float(np.dot(seg, seg))
    if not np.isfinite(seg_len_sq) or seg_len_sq <= 1.0e-18:
        return None
    t = float(-np.dot(p0, seg) / seg_len_sq)
    t = max(0.0, min(1.0, t))
    closest = p0 + t * seg
    dist = float(np.linalg.norm(closest))
    return dist if np.isfinite(dist) and dist > 1.0e-12 else None


def _finite_positive_values(values: Iterable[object]) -> np.ndarray:
    arr = np.asarray(tuple(float(v) for v in tuple(values or ())), dtype=float)
    if arr.size == 0:
        return np.empty((0,), dtype=float)
    return arr[np.isfinite(arr) & (arr > 1.0e-12)]


def _robust_percentile(values: Iterable[object], percentile: float, fallback: float) -> float:
    arr = _finite_positive_values(values)
    if arr.size == 0:
        return float(fallback)
    try:
        return float(np.percentile(arr, float(percentile)))
    except Exception:
        return float(fallback)


def _measure_opening_radial_envelope(
    *,
    vertices: np.ndarray,
    measurement_edges: Iterable[EdgeKey],
    selected_vertices: Iterable[int],
    ring: RingFit,
) -> BoreOpeningRadialEnvelope:
    """Measure the min/nominal/max opening envelope for any rim.

    This is the default measurement model, not a parallel coarse-mesh route.
    The nominal ring fit is established first, then the lower/upper envelope is
    measured only from samples that still behave like rim-band evidence in that
    same frame.  This prevents coarse or triangulated meshes from reporting a
    tiny ``radius_min`` merely because a spoke, floor chord, or transition edge
    crosses the opening interior.

    A dense circular opening is the low-spread case.  A coarse polygonal opening
    is the high-spread case.  Interior/chord contamination is rejected before the
    envelope becomes measurement truth.
    """

    center = np.asarray(ring.center, dtype=float).reshape(3)
    axis = unit_vector(ring.axis)
    radius_nominal = float(ring.radius)
    edge_set = {normalize_edge(edge) for edge in tuple(measurement_edges or ())}
    vertex_ids = tuple(sorted({int(v) for v in tuple(selected_vertices or ()) if 0 <= int(v) < len(vertices)}))

    # The band is deliberately expressed relative to the nominal radius rather
    # than edge count.  It is therefore the normal route for all meshes.  Hex / square
    # approximations keep their inradius evidence; interior fan spokes do not.
    if radius_nominal > 1.0e-12:
        lower_band = max(1.0e-12, float(radius_nominal) * 0.62)
        upper_band = max(float(radius_nominal) * 1.42, float(radius_nominal) + 1.0e-12)
        fallback_radius_min = float(radius_nominal) * 0.92
        fallback_radius_max = float(radius_nominal) * 1.08
    else:
        lower_band = 1.0e-12
        upper_band = float("inf")
        fallback_radius_min = float(radius_nominal)
        fallback_radius_max = float(radius_nominal)

    vertex_radii_all: list[float] = []
    vertex_radii_band: list[float] = []
    for vid in vertex_ids:
        rel = np.asarray(vertices[int(vid), :3], dtype=float).reshape(3) - center
        radial = rel - axis * float(np.dot(rel, axis))
        dist = float(np.linalg.norm(radial))
        if np.isfinite(dist) and dist > 1.0e-12:
            vertex_radii_all.append(dist)
            if lower_band <= dist <= upper_band:
                vertex_radii_band.append(dist)

    midpoint_radii_all: list[float] = []
    midpoint_radii_band: list[float] = []
    segment_distances_all: list[float] = []
    segment_distances_band: list[float] = []
    rejected_spoke_edge_count = 0
    rejected_interior_segment_count = 0
    rejected_out_of_band_sample_count = 0

    for edge in edge_set:
        try:
            a, b = normalize_edge(edge)
        except Exception:
            continue
        if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
            continue
        pa3 = np.asarray(vertices[int(a), :3], dtype=float).reshape(3)
        pb3 = np.asarray(vertices[int(b), :3], dtype=float).reshape(3)
        mid3 = 0.5 * (pa3 + pb3)
        rel_mid = mid3 - center
        axial_mid = float(np.dot(rel_mid, axis))
        radial_vec_mid = rel_mid - axis * axial_mid
        midpoint_radius = float(np.linalg.norm(radial_vec_mid))
        if np.isfinite(midpoint_radius) and midpoint_radius > 1.0e-12:
            midpoint_radii_all.append(midpoint_radius)

        edge_vec = pb3 - pa3
        edge_len = float(np.linalg.norm(edge_vec))
        radial_alignment = 0.0
        axial_alignment = 0.0
        obvious_spoke = False
        if edge_len > 1.0e-12 and midpoint_radius > 1.0e-12:
            tangent = edge_vec / edge_len
            radial_unit = radial_vec_mid / midpoint_radius
            radial_alignment = abs(float(np.dot(tangent, radial_unit)))
            axial_alignment = abs(float(np.dot(tangent, axis)))
            # A rim side is tangential to the opening.  A radial spoke points
            # across the opening interior.  Do not let spokes define r_min.
            obvious_spoke = bool(radial_alignment > 0.88 and axial_alignment < 0.45)

        midpoint_in_band = bool(lower_band <= midpoint_radius <= upper_band)
        if midpoint_in_band and not obvious_spoke:
            midpoint_radii_band.append(midpoint_radius)
        elif np.isfinite(midpoint_radius) and midpoint_radius > 1.0e-12:
            rejected_out_of_band_sample_count += 0 if midpoint_in_band else 1

        seg_dist = _projected_segment_distance_to_center(vertices=vertices, edge=edge, center=center, axis=axis)
        if seg_dist is not None:
            segment_distances_all.append(float(seg_dist))
            if obvious_spoke:
                rejected_spoke_edge_count += 1
            elif float(seg_dist) < lower_band:
                rejected_interior_segment_count += 1
            elif float(seg_dist) <= upper_band:
                segment_distances_band.append(float(seg_dist))
            else:
                rejected_out_of_band_sample_count += 1

    lower_samples = segment_distances_band or midpoint_radii_band or vertex_radii_band or [fallback_radius_min]
    upper_samples = vertex_radii_band or midpoint_radii_band or segment_distances_band or [fallback_radius_max]

    # If the rim-band filter removes almost all support, do not publish a wildly
    # wide envelope.  Fall back to a conservative nominal band and record why.
    filtered_support = int(len(vertex_radii_band) + len(midpoint_radii_band) + len(segment_distances_band))
    all_support = int(len(vertex_radii_all) + len(midpoint_radii_all) + len(segment_distances_all))
    low_support_fallback = bool(filtered_support < max(6, min(12, len(edge_set))))
    if low_support_fallback:
        lower_samples = [fallback_radius_min, radius_nominal]
        upper_samples = [radius_nominal, fallback_radius_max]

    lower_percentile = 10.0 if len(lower_samples) >= 8 else 0.0
    upper_percentile = 90.0 if len(upper_samples) >= 8 else 100.0
    radius_min = _robust_percentile(lower_samples, lower_percentile, radius_nominal)
    radius_max = _robust_percentile(upper_samples, upper_percentile, radius_nominal)

    if radius_min > radius_max:
        radius_min, radius_max = radius_max, radius_min
    if radius_nominal > 1.0e-12:
        radius_min = max(lower_band, min(float(radius_min), float(radius_nominal) * 1.12))
        radius_max = min(upper_band, max(float(radius_max), float(radius_nominal) * 0.88))
    radius_max = max(float(radius_max), float(radius_min), 1.0e-12)
    radial_spread = float(max(0.0, radius_max - radius_min))
    radial_spread_ratio = float(radial_spread / max(float(radius_nominal), 1.0e-12))

    coarse_polygonal = bool(
        radial_spread_ratio >= 0.045
        or float(ring.radius_rel_rms) >= 0.045
        or (len(edge_set) <= 16 and radial_spread_ratio >= 0.025)
    )

    def _stats(vals: Iterable[float]) -> dict[str, object]:
        arr = _finite_positive_values(vals)
        if arr.size == 0:
            return {"count": 0}
        return {
            "count": int(arr.size),
            "min": float(arr.min()),
            "p10": float(np.percentile(arr, 10.0)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90.0)),
            "max": float(arr.max()),
        }

    envelope = BoreOpeningRadialEnvelope(
        center=to_vector3(center),
        axis=to_vector3(canonical_axis(axis)),
        radius_min=float(radius_min),
        radius_nominal=float(radius_nominal),
        radius_max=float(radius_max),
        diameter_min=float(2.0 * radius_min),
        diameter_nominal=float(2.0 * radius_nominal),
        diameter_max=float(2.0 * radius_max),
        radial_spread=float(radial_spread),
        radial_spread_ratio=float(radial_spread_ratio),
        vertex_sample_count=int(len(vertex_radii_band)),
        edge_midpoint_sample_count=int(len(midpoint_radii_band)),
        segment_distance_sample_count=int(len(segment_distances_band)),
        coarse_polygonal=bool(coarse_polygonal),
        diagnostics={
            "normal_measurement_route": "all_openings_are_radial_envelopes",
            "dense_circle_is_low_spread_case": True,
            "coarse_polygon_is_high_spread_case": True,
            "v118_ring_band_filtered_envelope": True,
            "lower_radius_source": "ring_band_filtered_projected_edge_segment_distance_percentile",
            "upper_radius_source": "ring_band_filtered_rim_vertex_radius_percentile",
            "lower_band_radius": float(lower_band),
            "upper_band_radius": float(upper_band),
            "fallback_radius_min": float(fallback_radius_min),
            "fallback_radius_max": float(fallback_radius_max),
            "low_support_fallback_used": bool(low_support_fallback),
            "filtered_support_count": int(filtered_support),
            "all_support_count": int(all_support),
            "rejected_spoke_edge_count": int(rejected_spoke_edge_count),
            "rejected_interior_segment_count": int(rejected_interior_segment_count),
            "rejected_out_of_band_sample_count": int(rejected_out_of_band_sample_count),
            "lower_percentile": float(lower_percentile),
            "upper_percentile": float(upper_percentile),
            "vertex_radii_all": _stats(vertex_radii_all),
            "edge_midpoint_radii_all": _stats(midpoint_radii_all),
            "segment_distances_all": _stats(segment_distances_all),
            "vertex_radii_band": _stats(vertex_radii_band),
            "edge_midpoint_radii_band": _stats(midpoint_radii_band),
            "segment_distances_band": _stats(segment_distances_band),
            "ring_radius_rel_rms": float(ring.radius_rel_rms),
            "ring_radius_mad": float(ring.radius_mad),
            "ring_circularity": float(ring.circularity),
        },
    )
    return envelope


def measure_bore_opening(
    mesh: trimesh.Trimesh,
    edge_ids: Iterable[int],
    *,
    min_ring_points: int = 6,
    min_radius: float = 0.05,
    near_closed_gap_edge_lengths: float = 3.5,
) -> BoreOpeningMeasurement:
    """Measure the selected bore opening from stable mesh edge IDs.

    This accepts both a perfect closed degree-2 loop and a near-closed rim-ish
    selected chain.  It does not promote the measurement into a rebuild target;
    it only returns geometry, quality and diagnostics.
    """

    _validate_mesh(mesh)
    selected_edge_ids = tuple(sorted({int(v) for v in edge_ids}))
    if not selected_edge_ids:
        raise ValueError("No selected Bore opening edges to measure.")

    vertices = _mesh_vertices(mesh)
    edge_table = build_edge_table(mesh)
    unique_edges = edge_table["unique_edges"]
    if not isinstance(unique_edges, tuple):
        raise ValueError("Invalid edge table: unique_edges missing.")

    selected_edges = _edge_ids_to_keys(selected_edge_ids, unique_edges)
    if not selected_edges:
        raise ValueError("Selected Bore opening edge IDs did not resolve to mesh edges.")

    components = connected_edge_components(set(selected_edges))
    component_sizes = tuple(sorted((len(comp) for comp in components), reverse=True))
    main_component = max(components, key=len) if components else set(selected_edges)
    main_vertices = tuple(sorted({v for edge in main_component for v in edge}))
    all_selected_edges = set(selected_edges)
    all_selected_vertices = tuple(sorted({v for edge in all_selected_edges for v in edge}))

    # Clean meshes usually provide one connected degree-2 loop, so the largest
    # component is the right measurement source.  Damaged imported meshes can
    # visually select a good bore opening while the edge graph is split into many
    # tiny fragments.  In that case the largest component may be only a short
    # arc (the observed bad mesh had 114 selected edges but only 11 vertices in
    # the largest connected component).  Measure the whole selected edge cloud
    # instead of failing before the geometric fit can happen.
    largest_fraction = float(len(main_component)) / max(float(len(all_selected_edges)), 1.0)
    polluted_fragment_cloud = bool(len(components) > 12 and largest_fraction < 0.25)
    use_all_fragments = bool(
        not polluted_fragment_cloud
        and len(all_selected_vertices) >= int(min_ring_points)
        and (
            len(main_vertices) < int(min_ring_points)
            or (len(components) > 1 and largest_fraction < 0.45)
        )
    )

    if use_all_fragments:
        measurement_edges = all_selected_edges
        selected_vertices = all_selected_vertices
        component_strategy = "all_selected_edge_fragments"
    else:
        measurement_edges = main_component
        selected_vertices = main_vertices
        component_strategy = "largest_connected_component" if not polluted_fragment_cloud else "largest_component_only_polluted_fragment_cloud_rejected"

    if len(selected_vertices) < int(min_ring_points):
        raise ValueError(
            "Selected Bore opening has too few vertices to measure. "
            f"Vertices: {len(selected_vertices)}; required: {int(min_ring_points)}; "
            f"input_edges: {len(selected_edges)}; component_sizes: {component_sizes}."
        )

    points = vertices[np.asarray(selected_vertices, dtype=np.int64), :3]
    ring = fit_ring_points(points, min_points=int(min_ring_points), min_radius=float(min_radius))
    if ring is None:
        raise ValueError("Selected Bore opening could not be fit as a measurable circular/near-circular ring.")

    refinement_diagnostics: dict[str, object] = {"used": False}
    if use_all_fragments and (float(ring.circularity) < 0.28 or float(ring.radius_rel_rms) > 0.12):
        refined = _refine_fragmented_ring_band(
            vertices=vertices,
            edges=all_selected_edges,
            initial_ring=ring,
            min_ring_points=int(min_ring_points),
            min_radius=float(min_radius),
        )
        if refined is not None:
            refined_edges, refined_vertices, refined_ring, refinement_diagnostics = refined
            # Use the radial-band result only when it gives a real improvement
            # and keeps a meaningful amount of selected rim evidence.
            if (
                len(refined_edges) >= max(int(min_ring_points), 12)
                and len(refined_vertices) >= int(min_ring_points)
                and (
                    float(refined_ring.circularity) > float(ring.circularity) + 0.08
                    or float(refined_ring.radius_rel_rms) < float(ring.radius_rel_rms) * 0.72
                    or float(refined_ring.confidence) > float(ring.confidence) + 0.08
                )
            ):
                measurement_edges = refined_edges
                selected_vertices = refined_vertices
                points = vertices[np.asarray(selected_vertices, dtype=np.int64), :3]
                ring = refined_ring
                component_strategy = "fragmented_radial_band"
                refinement_diagnostics = dict(refinement_diagnostics)
                refinement_diagnostics["used"] = True

    graph = edge_graph_stats(measurement_edges, vertices=vertices)
    median_edge_length = float(graph.get("median_edge_length", 1.0))
    endpoint_gap = float(graph.get("endpoint_gap", 0.0))
    endpoint_gap_ratio = float(endpoint_gap / max(median_edge_length, 1.0e-12))
    closed = bool(graph.get("closed", False))
    open_endpoint_count = int(graph.get("open_endpoint_count", 0))
    branch_vertex_count = int(graph.get("branch_vertex_count", 0))
    near_closed = bool(
        (not closed)
        and int(graph.get("component_count", 1)) == 1
        and open_endpoint_count <= 2
        and branch_vertex_count == 0
        and endpoint_gap_ratio <= float(near_closed_gap_edge_lengths)
    )

    closure_score = 1.0 if closed else (0.72 if near_closed else 0.30)
    branch_score = 1.0 - clamp(branch_vertex_count / 4.0, 0.0, 1.0)
    size_score = clamp((len(measurement_edges) - 5) / 30.0, 0.0, 1.0)
    confidence = clamp(
        0.45 * float(ring.confidence)
        + 0.25 * closure_score
        + 0.20 * branch_score
        + 0.10 * size_score,
        0.0,
        1.0,
    )

    radial_envelope = _measure_opening_radial_envelope(
        vertices=vertices,
        measurement_edges=measurement_edges,
        selected_vertices=selected_vertices,
        ring=ring,
    )

    diagnostics: dict[str, object] = {
        "mode": "bore_opening_measurement",
        "input_edge_count": len(selected_edge_ids),
        "measured_component_edge_count": len(measurement_edges),
        "dropped_component_edge_count": int(len(selected_edges) - len(measurement_edges)),
        "component_strategy": component_strategy,
        "largest_component_edge_count": int(len(main_component)),
        "largest_component_vertex_count": int(len(main_vertices)),
        "largest_component_fraction": float(largest_fraction),
        "all_selected_vertex_count": int(len(all_selected_vertices)),
        "component_sizes": component_sizes,
        "fragmented_ring_refinement": refinement_diagnostics,
        "median_edge_length": median_edge_length,
        "axis_hint": axis_hint(ring.axis),
        "ring_fit": dict(ring.diagnostics),
        "radial_envelope": radial_envelope.to_dict(),
        "radius_min": float(radial_envelope.radius_min),
        "radius_nominal": float(radial_envelope.radius_nominal),
        "radius_max": float(radial_envelope.radius_max),
        "diameter_min": float(radial_envelope.diameter_min),
        "diameter_nominal": float(radial_envelope.diameter_nominal),
        "diameter_max": float(radial_envelope.diameter_max),
        "radial_spread": float(radial_envelope.radial_spread),
        "radial_spread_ratio": float(radial_envelope.radial_spread_ratio),
        "coarse_polygonal": bool(radial_envelope.coarse_polygonal),
        "graph": {k: v for k, v in graph.items() if k != "endpoint_vertices"},
    }

    return BoreOpeningMeasurement(
        edge_ids=tuple(sorted(_keys_to_ids(measurement_edges, unique_edges))),
        edge_count=int(len(measurement_edges)),
        vertex_ids=selected_vertices,
        vertex_count=int(len(selected_vertices)),
        center=ring.center,
        axis=ring.axis,
        radius=float(ring.radius),
        diameter=float(ring.diameter),
        closed=closed,
        near_closed=near_closed,
        endpoint_gap=endpoint_gap,
        endpoint_gap_ratio=endpoint_gap_ratio,
        branch_vertex_count=branch_vertex_count,
        open_endpoint_count=open_endpoint_count,
        component_count=int(graph.get("component_count", 1)),
        plane_rms=float(ring.plane_rms),
        plane_rel_rms=float(ring.plane_rel_rms),
        radius_rms=float(ring.radius_rms),
        radius_rel_rms=float(ring.radius_rel_rms),
        radius_mad=float(ring.radius_mad),
        circularity=float(ring.circularity),
        confidence=float(confidence),
        radius_min=float(radial_envelope.radius_min),
        radius_max=float(radial_envelope.radius_max),
        diameter_min=float(radial_envelope.diameter_min),
        diameter_max=float(radial_envelope.diameter_max),
        radial_spread=float(radial_envelope.radial_spread),
        radial_spread_ratio=float(radial_envelope.radial_spread_ratio),
        coarse_polygonal=bool(radial_envelope.coarse_polygonal),
        radial_envelope=radial_envelope.to_dict(),
        diagnostics=diagnostics,
    )


def measure_bore_opening_candidates(
    mesh: trimesh.Trimesh,
    edge_ids: Iterable[int],
    *,
    min_ring_points: int = 6,
    min_radius: float = 0.05,
    max_candidates: int = 8,
) -> tuple[BoreOpeningMeasurement, ...]:
    """Return several measured opening candidates from the same selected edge evidence.

    This is the mesh-native equivalent of an evidence-ledger approach:
    keep competing ring observations instead of immediately collapsing the
    selected fragments to one radius/center.  Chamfers, counterbores and nearby
    pockets can create several radial bands.  Recognition can then evaluate
    the measured candidates against RegionData face evidence.
    """

    _validate_mesh(mesh)
    selected_edge_ids = tuple(sorted({int(v) for v in edge_ids}))
    if not selected_edge_ids:
        return ()

    try:
        primary = measure_bore_opening(
            mesh,
            selected_edge_ids,
            min_ring_points=int(min_ring_points),
            min_radius=float(min_radius),
        )
    except Exception:
        primary = None

    vertices = _mesh_vertices(mesh)
    edge_table = build_edge_table(mesh)
    unique_edges = edge_table["unique_edges"]
    if not isinstance(unique_edges, tuple):
        return tuple(c for c in (primary,) if c is not None)

    try:
        all_edges = set(_edge_ids_to_keys(selected_edge_ids, unique_edges))
    except Exception:
        return tuple(c for c in (primary,) if c is not None)

    all_vertices = tuple(sorted({v for edge in all_edges for v in edge}))
    components_for_cloud_guard = connected_edge_components(set(all_edges))
    largest_cloud_fraction = (
        float(max((len(comp) for comp in components_for_cloud_guard), default=0)) / max(float(len(all_edges)), 1.0)
    )
    polluted_fragment_cloud = bool(len(components_for_cloud_guard) > 12 and largest_cloud_fraction < 0.25)
    candidates: list[BoreOpeningMeasurement] = []
    if primary is not None:
        candidates.append(primary)

    if (not polluted_fragment_cloud) and len(all_vertices) >= int(min_ring_points):
        points = vertices[np.asarray(all_vertices, dtype=np.int64), :3]
        initial_ring = fit_ring_points(points, min_points=int(min_ring_points), min_radius=float(min_radius))
        if initial_ring is not None:
            for cand_edges, cand_vertices, cand_ring, cand_diag in _fragmented_ring_band_candidate_list(
                vertices=vertices,
                edges=all_edges,
                initial_ring=initial_ring,
                min_ring_points=int(min_ring_points),
                min_radius=float(min_radius),
                max_candidates=int(max_candidates),
            ):
                try:
                    cand = _opening_measurement_from_ring_edges(
                        vertices=vertices,
                        unique_edges=unique_edges,
                        input_edge_count=len(selected_edge_ids),
                        measurement_edges=cand_edges,
                        selected_vertices=cand_vertices,
                        ring=cand_ring,
                        component_strategy="fragmented_radial_band_candidate",
                        extra_diagnostics=cand_diag,
                    )
                except Exception:
                    continue
                candidates.append(cand)

    # Dedupe near-identical candidates while preserving the stronger/supporting one.
    deduped: list[BoreOpeningMeasurement] = []
    for cand in candidates:
        duplicate_index = None
        cand_center = np.asarray(cand.center, dtype=float)
        cand_axis = unit_vector(cand.axis)
        for idx, old in enumerate(deduped):
            old_center = np.asarray(old.center, dtype=float)
            old_axis = unit_vector(old.axis)
            if abs(float(np.dot(cand_axis, old_axis))) < 0.985:
                continue
            radius_ref = max(float(cand.radius), float(old.radius), 1.0e-9)
            radius_close = abs(float(cand.radius) - float(old.radius)) <= max(radius_ref * 0.035, 0.08)
            center_delta = cand_center - old_center
            center_cross = float(np.linalg.norm(center_delta - cand_axis * float(np.dot(center_delta, cand_axis))))
            if radius_close and center_cross <= max(radius_ref * 0.05, 0.35):
                duplicate_index = idx
                break
        if duplicate_index is None:
            deduped.append(cand)
        else:
            old = deduped[duplicate_index]
            cand_score = _opening_candidate_preference_score(cand)
            old_score = _opening_candidate_preference_score(old)
            if cand_score > old_score:
                deduped[duplicate_index] = cand

    deduped.sort(key=_opening_candidate_preference_score, reverse=True)
    return tuple(deduped[: max(1, int(max_candidates))])



def measure_bore_opening_component_candidates(
    mesh: trimesh.Trimesh,
    edge_ids: Iterable[int],
    *,
    min_ring_points: int = 6,
    min_radius: float = 0.05,
    max_candidates: int = 12,
) -> tuple[BoreOpeningMeasurement, ...]:
    """Resolve measured opening candidates from raw selected-edge components.

    This is the damaged-bore measurement repair path.  ``measure_bore_opening``
    and ``measure_bore_opening_candidates`` can fit a whole selected edge cloud
    when the selection is fragmented.  That is useful when all fragments belong
    to one physical rim, but it is wrong when a damaged imported mesh produces a
    polluted selection containing hundreds of unrelated edge fragments.

    This helper performs the missing semantic measurement transform:

        raw selected edge evidence -> component-seeded MeasuredBoreFrame candidates

    It does not classify a BORE, does not own wall faces, and does not authorize
    rebuild.  It simply measures local ring candidates that are seeded by actual
    connected components of the raw selection, then expands each seed only to
    nearby selected fragments that agree with the seed's plane/radius frame.
    """

    _validate_mesh(mesh)
    selected_edge_ids = tuple(sorted({int(v) for v in edge_ids if int(v) >= 0}))
    if not selected_edge_ids:
        return ()

    vertices = _mesh_vertices(mesh)
    edge_table = build_edge_table(mesh)
    unique_edges = edge_table.get("unique_edges")
    if not isinstance(unique_edges, tuple):
        return ()

    try:
        all_edges = {normalize_edge(edge) for edge in _edge_ids_to_keys(selected_edge_ids, unique_edges)}
    except Exception:
        return ()
    if len(all_edges) < int(min_ring_points):
        return ()

    components = connected_edge_components(set(all_edges))
    if not components:
        return ()

    def _edge_length(edge: EdgeKey) -> float:
        a, b = edge
        if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
            return 0.0
        return float(np.linalg.norm(vertices[a, :3] - vertices[b, :3]))

    selected_lengths = [_edge_length(edge) for edge in all_edges]
    selected_lengths = [v for v in selected_lengths if np.isfinite(v) and v > 1.0e-12]
    median_edge_length = float(np.median(np.asarray(selected_lengths, dtype=float))) if selected_lengths else 1.0

    candidates: list[BoreOpeningMeasurement] = []
    # Prefer meaningful components, but keep enough small components for damaged
    # fragmented rims where the correct mouth may be represented only by arcs.
    ranked_components = sorted(
        (set(comp) for comp in components if comp),
        key=lambda comp: (-len(comp), tuple(sorted(comp))[:1]),
    )
    max_seed_components = min(80, len(ranked_components))

    for component_index, comp in enumerate(ranked_components[:max_seed_components]):
        comp_edges = {normalize_edge(edge) for edge in comp}
        comp_vertices = tuple(sorted({v for edge in comp_edges for v in edge}))
        if len(comp_edges) < max(3, int(min_ring_points) // 2) or len(comp_vertices) < int(min_ring_points):
            continue

        seed_points = vertices[np.asarray(comp_vertices, dtype=np.int64), :3]
        seed_ring = fit_ring_points(seed_points, min_points=int(min_ring_points), min_radius=float(min_radius))
        if seed_ring is None:
            continue
        if float(seed_ring.radius) <= max(float(min_radius), 1.0e-12):
            continue

        seed_center = np.asarray(seed_ring.center, dtype=float).reshape(3)
        seed_axis = unit_vector(seed_ring.axis)
        seed_radius = float(seed_ring.radius)
        # A component is only a seed.  Expand to selected fragments that live in
        # the same local opening plane and radius band.  This allows broken rim
        # fragments to join the measurement without letting the whole polluted
        # selection cloud become one ring.
        plane_tol = max(
            float(seed_ring.plane_rms) * 3.0,
            float(seed_ring.plane_mad) * 4.0,
            seed_radius * 0.14,
            median_edge_length * 3.0,
            1.0e-6,
        )
        radial_tol = max(
            float(seed_ring.radius_mad) * 4.5,
            float(seed_ring.radius_rms) * 2.75,
            seed_radius * 0.085,
            median_edge_length * 2.25,
            1.0e-6,
        )
        # v1.3.7 seed-island repair:
        # The previous raw-component resolver expanded a seed component against
        # *all* selected edges that matched the same plane/radius band. On large
        # damaged imports that can accidentally pull in a different selected
        # region that happens to share the same cylindrical frame. Measurement
        # must construct the selected opening, so expansion is now additionally
        # bounded to the spatial neighborhood of the seed ring. A true rim edge
        # lies roughly one radius from the ring center; remote geometry should
        # not be allowed to join merely because it is co-planar and radius-like.
        local_distance_limit = max(
            1.85 * max(seed_radius, 1.0e-9),
            9.0 * max(median_edge_length, 1.0e-9),
            2.5 * max(plane_tol, radial_tol, 1.0e-9),
        )
        local_distance_rejected = 0
        inlier_edges: set[EdgeKey] = set()
        for edge in all_edges:
            a, b = edge
            if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
                continue
            pa = vertices[a, :3]
            pb = vertices[b, :3]
            mid = 0.5 * (pa + pb)
            rel = mid - seed_center
            if float(np.linalg.norm(rel)) > float(local_distance_limit):
                local_distance_rejected += 1
                continue
            axial = float(np.dot(rel, seed_axis))
            radial_vec = rel - seed_axis * axial
            radial = float(np.linalg.norm(radial_vec))
            if not np.isfinite(radial):
                continue
            vec = pb - pa
            length = float(np.linalg.norm(vec))
            if not np.isfinite(length) or length <= 1.0e-12:
                continue
            tangent = vec / length
            radial_unit = radial_vec / max(radial, 1.0e-12)
            # Drop obvious spokes/cross edges where possible.
            radial_tangent = abs(float(np.dot(tangent, radial_unit))) if radial > 1.0e-12 else 0.0
            axial_tangent = abs(float(np.dot(tangent, seed_axis)))
            if radial_tangent > 0.92 and axial_tangent < 0.35:
                continue
            if abs(axial) <= plane_tol and abs(radial - seed_radius) <= radial_tol:
                inlier_edges.add(edge)

        if len(inlier_edges) < int(min_ring_points):
            continue
        inlier_vertices = tuple(sorted({v for edge in inlier_edges for v in edge}))
        if len(inlier_vertices) < int(min_ring_points):
            continue

        points = vertices[np.asarray(inlier_vertices, dtype=np.int64), :3]
        ring = fit_ring_points(points, axis=seed_axis, min_points=int(min_ring_points), min_radius=float(min_radius))
        if ring is None:
            continue

        # One bounded cleanup pass in the refit frame.
        center2 = np.asarray(ring.center, dtype=float).reshape(3)
        axis2 = unit_vector(ring.axis)
        radius2 = float(ring.radius)
        radial_tol2 = max(
            float(ring.radius_mad) * 4.0,
            float(ring.radius_rms) * 2.25,
            radius2 * 0.070,
            median_edge_length * 2.0,
            1.0e-6,
        )
        plane_tol2 = max(
            float(ring.plane_mad) * 4.0,
            float(ring.plane_rms) * 2.25,
            radius2 * 0.10,
            median_edge_length * 2.5,
            1.0e-6,
        )
        local_distance_limit2 = max(
            1.85 * max(radius2, 1.0e-9),
            9.0 * max(median_edge_length, 1.0e-9),
            2.5 * max(plane_tol2, radial_tol2, 1.0e-9),
        )
        refined_edges: set[EdgeKey] = set()
        for edge in inlier_edges:
            a, b = edge
            mid = 0.5 * (vertices[a, :3] + vertices[b, :3])
            rel = mid - center2
            if float(np.linalg.norm(rel)) > float(local_distance_limit2):
                continue
            axial = float(np.dot(rel, axis2))
            radial_vec = rel - axis2 * axial
            radial = float(np.linalg.norm(radial_vec))
            if abs(axial) <= plane_tol2 and abs(radial - radius2) <= radial_tol2:
                refined_edges.add(edge)
        if len(refined_edges) >= int(min_ring_points):
            refined_vertices = tuple(sorted({v for edge in refined_edges for v in edge}))
            if len(refined_vertices) >= int(min_ring_points):
                points2 = vertices[np.asarray(refined_vertices, dtype=np.int64), :3]
                ring2 = fit_ring_points(points2, axis=axis2, min_points=int(min_ring_points), min_radius=float(min_radius))
                if ring2 is not None:
                    inlier_edges = refined_edges
                    inlier_vertices = refined_vertices
                    ring = ring2

        support_fraction = float(len(inlier_edges)) / max(float(len(all_edges)), 1.0)
        seed_fraction = float(len(comp_edges)) / max(float(len(all_edges)), 1.0)
        try:
            measurement = _opening_measurement_from_ring_edges(
                vertices=vertices,
                unique_edges=unique_edges,
                input_edge_count=len(selected_edge_ids),
                measurement_edges=inlier_edges,
                selected_vertices=inlier_vertices,
                ring=ring,
                component_strategy="raw_selected_edge_component_opening_resolver",
                extra_diagnostics={
                    "mode": "raw_selected_edge_component_opening_resolver",
                    "component_index": int(component_index),
                    "component_count": int(len(components)),
                    "seed_component_edge_count": int(len(comp_edges)),
                    "seed_component_vertex_count": int(len(comp_vertices)),
                    "seed_component_fraction": float(seed_fraction),
                    # v1.3.7: preserve the primary raw selected-edge component
                    # separately from the expanded inlier ring. Downstream
                    # Recognition must use this seed island as locality authority;
                    # expanded fragments are measurement support only and may not
                    # become display/rebuild seed authority.
                    "seed_component_edge_ids": tuple(sorted(_keys_to_ids(comp_edges, unique_edges))),
                    "expanded_inlier_edge_ids": tuple(sorted(_keys_to_ids(inlier_edges, unique_edges))),
                    "expanded_inlier_edge_count": int(len(inlier_edges)),
                    "expanded_inlier_vertex_count": int(len(inlier_vertices)),
                    "expanded_support_fraction": float(support_fraction),
                    "seed_radius": float(seed_radius),
                    "resolved_radius": float(ring.radius),
                    "plane_tolerance": float(plane_tol),
                    "radial_tolerance": float(radial_tol),
                    "local_distance_limit": float(local_distance_limit),
                    "local_distance_rejected_edge_count": int(local_distance_rejected),
                    "median_selected_edge_length": float(median_edge_length),
                    "all_raw_component_count": int(len(components)),
                },
            )
        except Exception:
            continue
        candidates.append(measurement)

    # Dedupe near-identical component-resolved frames.
    deduped: list[BoreOpeningMeasurement] = []
    for cand in candidates:
        duplicate_index = None
        cand_center = np.asarray(cand.center, dtype=float)
        cand_axis = unit_vector(cand.axis)
        for idx, old in enumerate(deduped):
            old_center = np.asarray(old.center, dtype=float)
            old_axis = unit_vector(old.axis)
            if abs(float(np.dot(cand_axis, old_axis))) < 0.985:
                continue
            radius_ref = max(float(cand.radius), float(old.radius), 1.0e-9)
            radius_close = abs(float(cand.radius) - float(old.radius)) <= max(radius_ref * 0.04, 0.10)
            delta = cand_center - old_center
            center_cross = float(np.linalg.norm(delta - cand_axis * float(np.dot(delta, cand_axis))))
            if radius_close and center_cross <= max(radius_ref * 0.06, 0.45):
                duplicate_index = idx
                break
        if duplicate_index is None:
            deduped.append(cand)
        else:
            if _opening_candidate_preference_score(cand) > _opening_candidate_preference_score(deduped[duplicate_index]):
                deduped[duplicate_index] = cand

    def _component_resolver_score(cand: BoreOpeningMeasurement) -> float:
        diag = dict(cand.diagnostics or {})
        support = float(diag.get("expanded_support_fraction", 0.0) or 0.0)
        seed_edges = float(diag.get("seed_component_edge_count", 0.0) or 0.0)
        return float(
            1.10 * float(cand.confidence)
            + 0.90 * float(cand.circularity)
            + 0.55 * min(float(cand.edge_count) / 80.0, 1.0)
            + 0.35 * min(seed_edges / 32.0, 1.0)
            + 0.45 * min(support * 8.0, 1.0)
            - 0.90 * min(float(cand.radius_rel_rms), 1.0)
            - 0.25 * min(float(cand.plane_rel_rms), 1.0)
        )

    deduped.sort(key=_component_resolver_score, reverse=True)
    return tuple(deduped[: max(1, int(max_candidates))])


def _opening_measurement_to_dict(opening: BoreOpeningMeasurement | None) -> dict[str, object]:
    if opening is None:
        return {}
    return {
        "edge_ids": tuple(int(v) for v in opening.edge_ids),
        "edge_count": int(opening.edge_count),
        "vertex_count": int(opening.vertex_count),
        "center": opening.center,
        "axis": opening.axis,
        "radius": float(opening.radius),
        "diameter": float(opening.diameter),
        "radius_min": float(opening.radius_min if opening.radius_min is not None else opening.radius),
        "radius_nominal": float(opening.radius),
        "radius_max": float(opening.radius_max if opening.radius_max is not None else opening.radius),
        "diameter_min": float(opening.diameter_min if opening.diameter_min is not None else opening.diameter),
        "diameter_nominal": float(opening.diameter),
        "diameter_max": float(opening.diameter_max if opening.diameter_max is not None else opening.diameter),
        "radial_spread": float(opening.radial_spread if opening.radial_spread is not None else 0.0),
        "radial_spread_ratio": float(opening.radial_spread_ratio if opening.radial_spread_ratio is not None else 0.0),
        "coarse_polygonal": bool(opening.coarse_polygonal),
        "radial_envelope": dict(opening.radial_envelope or {}),
        "closed": bool(opening.closed),
        "near_closed": bool(opening.near_closed),
        "endpoint_gap_ratio": float(opening.endpoint_gap_ratio),
        "component_count": int(opening.component_count),
        "plane_rel_rms": float(opening.plane_rel_rms),
        "radius_rel_rms": float(opening.radius_rel_rms),
        "circularity": float(opening.circularity),
        "confidence": float(opening.confidence),
        "diagnostics": dict(opening.diagnostics or {}),
    }


def _median_runtime_edge_length(vertices: np.ndarray, edges: Iterable[EdgeKey]) -> float:
    vals: list[float] = []
    for edge in tuple(edges or ()):
        try:
            a, b = normalize_edge(edge)
        except Exception:
            continue
        if 0 <= a < len(vertices) and 0 <= b < len(vertices):
            length = float(np.linalg.norm(vertices[a, :3] - vertices[b, :3]))
            if np.isfinite(length) and length > 1.0e-12:
                vals.append(length)
    if not vals:
        return 1.0
    return float(np.median(np.asarray(vals, dtype=float)))


def _ring_opening_from_boundary_loop(
    *,
    vertices: np.ndarray,
    unique_edges: tuple[EdgeKey, ...],
    loop_edges: Iterable[EdgeKey],
    axis_hint_value: object,
    min_ring_points: int,
    min_radius: float,
) -> BoreOpeningMeasurement | None:
    """Measure one boundary-loop candidate as an opening ring."""

    edges = {normalize_edge(edge) for edge in tuple(loop_edges or ())}
    if len(edges) < int(min_ring_points):
        return None
    vertex_ids = tuple(sorted({v for edge in edges for v in edge if 0 <= int(v) < len(vertices)}))
    if len(vertex_ids) < int(min_ring_points):
        return None
    pts = vertices[np.asarray(vertex_ids, dtype=np.int64), :3]
    ring = fit_ring_points(pts, axis=axis_hint_value, min_points=int(min_ring_points), min_radius=float(min_radius))
    if ring is None:
        return None
    try:
        return _opening_measurement_from_ring_edges(
            vertices=vertices,
            unique_edges=unique_edges,
            input_edge_count=len(edges),
            measurement_edges=edges,
            selected_vertices=vertex_ids,
            ring=ring,
            component_strategy="opposite_boundary_loop_candidate",
            extra_diagnostics={
                "mode": "opposite_boundary_loop_candidate",
                "boundary_loop_edge_count": int(len(edges)),
                "semantic_role": "opposite_opening_evidence_candidate",
            },
        )
    except Exception:
        graph = edge_graph_stats(edges, vertices=vertices)
        return BoreOpeningMeasurement(
            edge_ids=tuple(sorted(_keys_to_ids(edges, unique_edges))),
            edge_count=int(len(edges)),
            vertex_ids=vertex_ids,
            vertex_count=int(len(vertex_ids)),
            center=ring.center,
            axis=ring.axis,
            radius=float(ring.radius),
            diameter=float(ring.diameter),
            closed=bool(graph.get("closed", False)),
            near_closed=bool(graph.get("closed", False)),
            endpoint_gap=float(graph.get("endpoint_gap", 0.0)),
            endpoint_gap_ratio=float(graph.get("endpoint_gap", 0.0)) / max(float(graph.get("median_edge_length", 1.0)), 1.0e-12),
            branch_vertex_count=int(graph.get("branch_vertex_count", 0) or 0),
            open_endpoint_count=int(graph.get("open_endpoint_count", 0) or 0),
            component_count=int(graph.get("component_count", 1) or 1),
            plane_rms=float(ring.plane_rms),
            plane_rel_rms=float(ring.plane_rel_rms),
            radius_rms=float(ring.radius_rms),
            radius_rel_rms=float(ring.radius_rel_rms),
            radius_mad=float(ring.radius_mad),
            circularity=float(ring.circularity),
            confidence=float(ring.confidence),
            diagnostics={"mode": "opposite_boundary_loop_candidate", "fallback_constructor": True},
        )



def _opposite_opening_candidates_from_region_edge_bands(
    *,
    mesh: trimesh.Trimesh,
    vertices: np.ndarray,
    unique_edges: tuple[EdgeKey, ...],
    region_face_ids: Iterable[int],
    selected_opening: BoreOpeningMeasurement,
    selected_edges: set[EdgeKey],
    selected_axis: np.ndarray,
    selected_center: np.ndarray,
    selected_radius: float,
    median_edge_length: float,
    min_ring_points: int,
    min_radius: float,
    max_candidates: int = 12,
) -> tuple[tuple[str, int, BoreOpeningMeasurement, set[EdgeKey]], ...]:
    """Find opposite-opening evidence from feature/rim-like edge bands.

    RegionData boundary loops are neutral AOI boundaries.  On clean bore meshes
    the true opposite rim may be a feature/seam loop inside that AOI rather than
    a boundary loop of the RegionData patch.  This helper searches RegionData
    edges near the selected opening cylinder, clusters them by axial station,
    and measures each cluster as a possible opposite opening.  The result is
    still measurement evidence only; Recognition must own wall faces later.
    """

    faces = np.asarray(mesh.faces, dtype=np.int64)[:, :3]
    valid_region_faces = {int(fid) for fid in tuple(region_face_ids or ()) if 0 <= int(fid) < len(faces)}
    if not valid_region_faces or not unique_edges:
        return ()

    region_edges: set[EdgeKey] = set()
    for fid in valid_region_faces:
        try:
            for edge in face_edges(faces[int(fid)]):
                region_edges.add(normalize_edge(edge))
        except Exception:
            continue

    # Do not re-measure the selected mouth as the opposite opening.
    region_edges.difference_update(selected_edges)
    if len(region_edges) < int(min_ring_points):
        return ()

    radius = float(max(selected_radius, min_radius, 1.0e-9))
    axial_skip = max(0.45 * radius, 4.0 * max(float(median_edge_length), 1.0e-9), 0.75)
    radial_tol = max(0.22 * radius, 4.0 * max(float(median_edge_length), 1.0e-9), 0.35)
    bin_width = max(0.075 * radius, 3.0 * max(float(median_edge_length), 1.0e-9), 0.45)

    band_edges_by_bin: dict[int, set[EdgeKey]] = {}
    for edge in region_edges:
        a, b = normalize_edge(edge)
        if not (0 <= a < len(vertices) and 0 <= b < len(vertices)):
            continue
        mid = 0.5 * (vertices[a, :3] + vertices[b, :3])
        rel = mid - selected_center.reshape(3)
        axial = float(np.dot(rel, selected_axis))
        if not np.isfinite(axial) or abs(axial) < axial_skip:
            continue
        radial_vec = rel - selected_axis * axial
        radial = float(np.linalg.norm(radial_vec))
        if not np.isfinite(radial):
            continue
        if abs(radial - radius) > radial_tol:
            continue
        bin_id = int(round(axial / bin_width))
        band_edges_by_bin.setdefault(bin_id, set()).add(edge)

    measured: list[tuple[str, int, BoreOpeningMeasurement, set[EdgeKey]]] = []
    for bin_id, edges in sorted(band_edges_by_bin.items(), key=lambda item: -len(item[1])):
        # Include immediate neighboring bins so tessellated rims split across a
        # few axial slices are measured as one physical opening station.
        cluster_edges: set[EdgeKey] = set(edges)
        cluster_edges.update(band_edges_by_bin.get(bin_id - 1, set()))
        cluster_edges.update(band_edges_by_bin.get(bin_id + 1, set()))
        if len(cluster_edges) < int(min_ring_points):
            continue
        candidate = _ring_opening_from_boundary_loop(
            vertices=vertices,
            unique_edges=unique_edges,
            loop_edges=cluster_edges,
            axis_hint_value=selected_axis,
            min_ring_points=int(min_ring_points),
            min_radius=float(min_radius),
        )
        if candidate is None:
            continue
        measured.append(("region_edge_band", int(bin_id), candidate, cluster_edges))
        if len(measured) >= int(max_candidates):
            break
    return tuple(measured)


def measure_two_opening_bore_frame(
    mesh: trimesh.Trimesh,
    selected_opening: BoreOpeningMeasurement,
    *,
    region_boundary_loops: Iterable[Iterable[EdgeKey]] = (),
    region_face_ids: Iterable[int] = (),
    min_ring_points: int = 6,
    min_radius: float = 0.05,
) -> BoreTwoOpeningMeasurement:
    """Measure a BORE frame by explicitly finding the opposite opening.

    This implements the intended BORE measurement order:

        selected opening A -> opposite opening B -> refined A/B bore frame.

    It never treats RegionData axial extent as an opposite opening.  RegionData
    may provide boundary-loop evidence where the opposite opening can be found;
    if no loop is measured, the result is invalid/review-only evidence.
    """

    _validate_mesh(mesh)
    vertices = _mesh_vertices(mesh)
    edge_table = build_edge_table(mesh)
    unique_edges = edge_table.get("unique_edges")
    if not isinstance(unique_edges, tuple):
        unique_edges = ()

    selected_axis = unit_vector(selected_opening.axis)
    selected_center = np.asarray(selected_opening.center, dtype=float).reshape(3)
    selected_radius = float(selected_opening.radius)
    selected_edges = {normalize_edge(edge) for edge in _edge_ids_to_keys(tuple_ints(selected_opening.edge_ids), unique_edges)} if unique_edges and selected_opening.edge_ids else set()
    median_edge_length = _median_runtime_edge_length(vertices, selected_edges) if selected_edges else 1.0

    candidates: list[tuple[float, BoreOpeningMeasurement, dict[str, object]]] = []
    all_candidate_diagnostics: list[dict[str, object]] = []
    seen_loop_count = 0

    def _consider_opposite_candidate(
        *,
        source: str,
        source_index: int,
        candidate: BoreOpeningMeasurement,
        source_edges: set[EdgeKey],
    ) -> None:
        cand_center = np.asarray(candidate.center, dtype=float).reshape(3)
        cand_axis = unit_vector(candidate.axis)
        delta = cand_center - selected_center
        axial_sep = float(np.dot(delta, selected_axis))
        abs_axial_sep = abs(axial_sep)
        cross = float(np.linalg.norm(delta - selected_axis * axial_sep))
        radius_ref = max(float(selected_radius), float(candidate.radius), 1.0e-9)
        radius_delta_rel = abs(float(candidate.radius) - float(selected_radius)) / radius_ref
        axis_dot = abs(float(np.dot(selected_axis, cand_axis)))
        overlap_edges = int(len(selected_edges & source_edges)) if selected_edges else 0
        overlap_ratio = float(overlap_edges) / max(float(min(len(selected_edges), len(source_edges))), 1.0) if selected_edges else 0.0
        min_depth = max(0.75 * radius_ref, 5.0 * max(median_edge_length, 1.0e-9), 0.50)
        centerline_limit = max(0.35 * radius_ref, 4.0 * max(median_edge_length, 1.0e-9), 0.75)

        rejection_reasons: list[str] = []
        if overlap_ratio > 0.25:
            rejection_reasons.append("overlaps_selected_opening")
        if abs_axial_sep < min_depth:
            rejection_reasons.append("axial_separation_below_min_depth")
        if axis_dot < 0.70:
            rejection_reasons.append("axis_not_parallel_to_selected_opening")
        if radius_delta_rel > 0.45:
            rejection_reasons.append("radius_mismatch_to_selected_opening")
        if cross > centerline_limit:
            rejection_reasons.append("centerline_distance_too_large")
        valid = not rejection_reasons

        depth_score = clamp(abs_axial_sep / max(1.75 * radius_ref, 1.0e-9), 0.0, 1.0)
        radius_score = 1.0 - clamp(radius_delta_rel / 0.45, 0.0, 1.0)
        center_score = 1.0 - clamp(cross / max(centerline_limit, 1.0e-9), 0.0, 1.0)
        score = (
            1.50 * float(candidate.confidence)
            + 1.20 * float(candidate.circularity)
            + 1.25 * float(axis_dot)
            + 1.50 * radius_score
            + 1.50 * center_score
            + 1.10 * depth_score
            + 0.40 * min(float(candidate.edge_count) / 96.0, 1.0)
            - 2.50 * overlap_ratio
        )
        diag = {
            "source": str(source),
            "source_index": int(source_index),
            "valid_opposite_candidate": bool(valid),
            "rejection_reasons": tuple(rejection_reasons),
            "score": float(score),
            "axial_separation": float(axial_sep),
            "abs_axial_separation": float(abs_axial_sep),
            "min_depth": float(min_depth),
            "centerline_distance": float(cross),
            "centerline_limit": float(centerline_limit),
            "axis_abs_dot": float(axis_dot),
            "radius_delta_rel": float(radius_delta_rel),
            "selected_edge_overlap_count": int(overlap_edges),
            "selected_edge_overlap_ratio": float(overlap_ratio),
            "candidate_edge_count": int(candidate.edge_count),
            "candidate_radius": float(candidate.radius),
            "candidate_confidence": float(candidate.confidence),
            "candidate_center": candidate.center,
            "candidate_axis": candidate.axis,
        }
        all_candidate_diagnostics.append(diag)
        if valid:
            candidates.append((float(score), candidate, diag))

    for loop_index, raw_loop in enumerate(tuple(region_boundary_loops or ()), start=1):
        loop_edges = {normalize_edge(edge) for edge in tuple(raw_loop or ())}
        if len(loop_edges) < int(min_ring_points):
            continue
        seen_loop_count += 1
        candidate = _ring_opening_from_boundary_loop(
            vertices=vertices,
            unique_edges=unique_edges,
            loop_edges=loop_edges,
            axis_hint_value=selected_axis,
            min_ring_points=int(min_ring_points),
            min_radius=float(min_radius),
        )
        if candidate is None:
            all_candidate_diagnostics.append({
                "source": "region_boundary_loop",
                "source_index": int(loop_index),
                "valid_opposite_candidate": False,
                "rejection_reasons": ("loop_could_not_be_fit_as_opening",),
                "candidate_edge_count": int(len(loop_edges)),
            })
            continue
        _consider_opposite_candidate(
            source="region_boundary_loop",
            source_index=int(loop_index),
            candidate=candidate,
            source_edges=loop_edges,
        )

    # v1.5.3: RegionData boundary loops are AOI boundaries, not guaranteed
    # physical bore rims.  If no opposite was accepted from the AOI boundary,
    # search internal feature/seam edge bands near the selected opening cylinder.
    edge_band_candidate_count = 0
    if not candidates:
        for source, source_index, candidate, source_edges in _opposite_opening_candidates_from_region_edge_bands(
            mesh=mesh,
            vertices=vertices,
            unique_edges=unique_edges,
            region_face_ids=region_face_ids,
            selected_opening=selected_opening,
            selected_edges=selected_edges,
            selected_axis=selected_axis,
            selected_center=selected_center,
            selected_radius=float(selected_radius),
            median_edge_length=float(median_edge_length),
            min_ring_points=int(min_ring_points),
            min_radius=float(min_radius),
            max_candidates=12,
        ):
            edge_band_candidate_count += 1
            _consider_opposite_candidate(
                source=str(source),
                source_index=int(source_index),
                candidate=candidate,
                source_edges=set(source_edges),
            )

    # v1.6.4 semantic correction: an "opposite opening" for a through-bore
    # means the far compatible opening on the same centerline, not merely the
    # highest scoring nearby rim/seam.  Damaged bores often contain internal
    # defect rings or mid-depth edge bands that fit the radius and axis well;
    # those are support/defect evidence, not the opposite mouth.  Prefer the
    # farthest valid compatible candidate first, then score.
    def _opposite_sort_key(item: tuple[float, BoreOpeningMeasurement, dict[str, object]]) -> tuple[float, float, float]:
        score, _candidate, diag = item
        abs_sep = float(diag.get("abs_axial_separation", 0.0) or 0.0)
        radius_delta = float(diag.get("radius_delta_rel", 1.0) or 1.0)
        centerline_distance = float(diag.get("centerline_distance", 1.0e9) or 1.0e9)
        # Larger separation wins; score breaks ties; smaller radius/center error
        # stabilizes the result when two rings are almost equally far.
        return (abs_sep, float(score), -radius_delta - 0.001 * centerline_distance)

    candidates.sort(key=_opposite_sort_key, reverse=True)
    all_candidate_diagnostics.sort(key=lambda item: (-float(item.get("abs_axial_separation", 0.0) or 0.0), -float(item.get("score", -1.0e9) or -1.0e9), str(item.get("source", ""))))
    candidate_diagnostics = tuple(dict(item[2]) for item in candidates[:12])
    rejected_candidate_diagnostics = tuple(dict(item) for item in all_candidate_diagnostics[:12])
    if not candidates:
        return BoreTwoOpeningMeasurement(
            selected_opening=selected_opening,
            opposite_opening=None,
            valid=False,
            center=selected_opening.center,
            axis=selected_opening.axis,
            radius=float(selected_opening.radius),
            diameter=float(2.0 * float(selected_opening.radius)),
            depth=0.0,
            opening_center=selected_opening.center,
            opposite_center=selected_opening.center,
            axial_min=0.0,
            axial_max=0.0,
            confidence=0.0,
            diagnostics={
                "mode": "two_opening_bore_frame_measurement",
                "semantic_stage": "selected_opening_to_opposite_opening_to_measured_bore_frame",
                "valid": False,
                "opposite_opening_found": False,
                "boundary_loop_count": int(seen_loop_count),
                "opposite_candidate_count": 0,
                "edge_band_candidate_count": int(edge_band_candidate_count),
                "opposite_candidate_rejection_diagnostics": rejected_candidate_diagnostics,
                "rejection_reason": "no_boundary_loop_candidate_satisfied_opposite_opening_rules",
                "forbidden_transfer": "Do not use RegionData axial extent as opposite opening; no opposite opening means no bore-wall ownership.",
            },
        )

    _score, opposite, best_diag = candidates[0]
    p0 = selected_center
    p1 = np.asarray(opposite.center, dtype=float).reshape(3)
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if not np.isfinite(length) or length <= 1.0e-12:
        axis = selected_axis
        depth = 0.0
    else:
        axis = vec / length
        depth = length
    radius = float(np.median(np.asarray([float(selected_opening.radius), float(opposite.radius)], dtype=float)))
    radius_delta_rel = abs(float(selected_opening.radius) - float(opposite.radius)) / max(radius, 1.0e-9)
    confidence = clamp(
        0.40 * float(selected_opening.confidence)
        + 0.35 * float(opposite.confidence)
        + 0.15 * (1.0 - clamp(radius_delta_rel / 0.45, 0.0, 1.0))
        + 0.10 * clamp(depth / max(1.50 * max(radius, 1.0e-9), 1.0e-9), 0.0, 1.0),
        0.0,
        1.0,
    )
    return BoreTwoOpeningMeasurement(
        selected_opening=selected_opening,
        opposite_opening=opposite,
        valid=True,
        center=to_vector3(p0),
        axis=to_vector3(axis),
        radius=radius,
        diameter=float(2.0 * radius),
        depth=float(depth),
        opening_center=to_vector3(p0),
        opposite_center=to_vector3(p1),
        axial_min=0.0,
        axial_max=float(depth),
        confidence=float(confidence),
        diagnostics={
            "mode": "two_opening_bore_frame_measurement",
            "semantic_stage": "selected_opening_to_opposite_opening_to_measured_bore_frame",
            "valid": True,
            "opposite_opening_found": True,
            "boundary_loop_count": int(seen_loop_count),
            "opposite_candidate_count": int(len(candidates)),
            "edge_band_candidate_count": int(edge_band_candidate_count),
            "best_opposite_candidate": dict(best_diag),
            "candidate_rankings": candidate_diagnostics,
            "opposite_candidate_rejection_diagnostics": rejected_candidate_diagnostics,
            "selected_opening_radius": float(selected_opening.radius),
            "opposite_opening_radius": float(opposite.radius),
            "radius_delta_rel": float(radius_delta_rel),
            "depth": float(depth),
            "axis_hint": axis_hint(axis),
            "forbidden_transfer": "Measured two-opening bore frame is evidence only; Recognition must still own wall faces before CandidateData.",
        },
    )

def measure_bore_region(
    mesh: trimesh.Trimesh,
    rim_edge_ids: Iterable[int],
    region_face_ids: Iterable[int],
) -> BoreRegionMeasurement:
    """Measure bore-family geometry from an opening and region face evidence.

    ``region_face_ids`` are neutral RegionData/candidate-region input faces. The
    returned scores are bore-family measurement evidence only; they do not
    classify the feature and do not authorize rebuild.
    """

    _validate_mesh(mesh)
    opening = measure_bore_opening(mesh, rim_edge_ids)
    face_ids = _valid_face_ids(mesh, region_face_ids)
    if not face_ids:
        raise ValueError("No valid Bore region faces to measure.")

    vertices = _mesh_vertices(mesh)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    centroids = vertices[faces[np.asarray(face_ids, dtype=np.int64), :3], :3].mean(axis=1)
    axis = canonical_axis(opening.axis)
    center = np.asarray(opening.center, dtype=float)

    axial = (centroids - center) @ axis
    axial_min = float(np.min(axial)) if axial.size else 0.0
    axial_max = float(np.max(axial)) if axial.size else 0.0

    # Use region-face centroid axial span as bore-depth evidence, but anchor the
    # reported opposite center at the farther side from the opening plane.
    far_t = axial_max if abs(axial_max) >= abs(axial_min) else axial_min
    depth = float(abs(far_t))
    opposite_center = center + axis * float(far_t)

    rel = centroids - center
    radial = rel - np.outer(rel @ axis, axis)
    radial_distances = np.linalg.norm(radial, axis=1)
    radius = float(median(radial_distances)) if len(radial_distances) else float(opening.radius)
    if radius <= 1.0e-12:
        radius = float(opening.radius)
    radial_abs = np.abs(radial_distances - radius)
    radial_mad = float(np.median(radial_abs)) if radial_abs.size else 0.0
    radial_rel_mad = float(radial_mad / max(radius, 1.0e-12))

    normals = _face_normals(mesh, vertices, faces)
    face_normals = normals[np.asarray(face_ids, dtype=np.int64), :3]
    normal_axis_abs = np.abs(face_normals @ axis)
    normal_axis_abs_median = float(np.median(normal_axis_abs)) if normal_axis_abs.size else 1.0

    bore_radius_consistency_score = 1.0 - clamp(radial_rel_mad / 0.20, 0.0, 1.0)
    bore_wall_normal_score = 1.0 - clamp(normal_axis_abs_median / 0.72, 0.0, 1.0)
    depth_score = clamp(depth / max(radius * 1.5, 1.0e-12), 0.0, 1.0)
    confidence = clamp(0.35 * opening.confidence + 0.25 * bore_radius_consistency_score + 0.25 * bore_wall_normal_score + 0.15 * depth_score, 0.0, 1.0)

    diagnostics: dict[str, object] = {
        "mode": "bore_region_measurement",
        "opening_confidence": float(opening.confidence),
        "axis_hint": axis_hint(axis),
        "region_face_count": len(face_ids),
        "axial_span": float(axial_max - axial_min),
        "bore_radius_consistency_score": float(bore_radius_consistency_score),
        "bore_wall_normal_score": float(bore_wall_normal_score),
        "depth_score": float(depth_score),
        "opening": dict(opening.diagnostics),
    }

    return BoreRegionMeasurement(
        opening=opening,
        region_face_ids=face_ids,
        region_face_count=int(len(face_ids)),
        center=to_vector3(center),
        axis=to_vector3(axis),
        radius=radius,
        diameter=float(2.0 * radius),
        depth=depth,
        opening_center=opening.center,
        opposite_center=to_vector3(opposite_center),
        axial_min=axial_min,
        axial_max=axial_max,
        normal_axis_abs_median=normal_axis_abs_median,
        radial_mad=radial_mad,
        radial_rel_mad=radial_rel_mad,
        confidence=float(confidence),
        diagnostics=diagnostics,
    )


def measure_bore_from_region_faces(
    mesh: trimesh.Trimesh,
    region_face_ids: Iterable[int],
    *,
    min_radius: float = 0.05,
) -> BoreRegionMeasurement | None:
    """Best-effort bore-family measurement from region faces only.

    This diagnostic helper receives neutral region faces and tries to infer a
    weak bore opening/radius/depth estimate. It returns ``None`` when the region
    is not stable enough to measure. The result is diagnostic evidence only, not
    recognition promotion and not rebuild authorization.
    """

    _validate_mesh(mesh)
    face_ids = _valid_face_ids(mesh, region_face_ids)
    if not face_ids:
        return None

    vertices = _mesh_vertices(mesh)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_vertices = vertices[faces[np.asarray(face_ids, dtype=np.int64), :3].reshape(-1), :3]
    if len(face_vertices) < 8:
        return None

    # A bore-family cylindrical surface usually has its longest spread along
    # the bore axis. This weak estimator should be superseded by rim-based
    # measurement whenever possible.
    centered = face_vertices - face_vertices.mean(axis=0)
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return None
    axis = canonical_axis(vh[0])
    rel = face_vertices - face_vertices.mean(axis=0)
    axial = rel @ axis
    t_min = float(np.min(axial))
    t_max = float(np.max(axial))
    depth = float(t_max - t_min)
    if depth <= 1.0e-12:
        return None

    # Estimate one boundary ring from the lower axial quartile.  This result is
    # diagnostic-only, so return None if it is not ring-like.
    q = float(np.quantile(axial, 0.12))
    rim_points = face_vertices[axial <= q]
    ring: RingFit | None = fit_ring_points(rim_points, axis=axis, min_points=6, min_radius=min_radius)
    if ring is None:
        return None

    opening_edges: tuple[int, ...] = ()
    opening = BoreOpeningMeasurement(
        edge_ids=opening_edges,
        edge_count=0,
        vertex_ids=(),
        vertex_count=int(len(rim_points)),
        center=ring.center,
        axis=ring.axis,
        radius=float(ring.radius),
        diameter=float(ring.diameter),
        closed=False,
        near_closed=False,
        endpoint_gap=0.0,
        endpoint_gap_ratio=0.0,
        branch_vertex_count=0,
        open_endpoint_count=0,
        component_count=1,
        plane_rms=float(ring.plane_rms),
        plane_rel_rms=float(ring.plane_rel_rms),
        radius_rms=float(ring.radius_rms),
        radius_rel_rms=float(ring.radius_rel_rms),
        radius_mad=float(ring.radius_mad),
        circularity=float(ring.circularity),
        confidence=float(min(ring.confidence, 0.55)),
        diagnostics={"mode": "bore_region_faces_inferred_opening", "warning": "diagnostic-only weak opening inference"},
    )

    center = np.asarray(ring.center, dtype=float)
    opposite = center + canonical_axis(ring.axis) * depth
    return BoreRegionMeasurement(
        opening=opening,
        region_face_ids=face_ids,
        region_face_count=int(len(face_ids)),
        center=ring.center,
        axis=ring.axis,
        radius=float(ring.radius),
        diameter=float(ring.diameter),
        depth=depth,
        opening_center=ring.center,
        opposite_center=to_vector3(opposite),
        axial_min=t_min,
        axial_max=t_max,
        normal_axis_abs_median=0.0,
        radial_mad=float(ring.radius_mad),
        radial_rel_mad=float(ring.radius_mad / max(ring.radius, 1.0e-12)),
        confidence=float(min(ring.confidence, 0.55)),
        diagnostics={"mode": "bore_region_measurement_from_region_faces", "diagnostic_only": True},
    )


# -----------------------------------------------------------------------------
# Mesh edge/face helpers kept local to the measurement layer
# -----------------------------------------------------------------------------


def build_edge_table(mesh: trimesh.Trimesh) -> dict[str, object]:
    """Build the same sorted unique-edge table used by Bore selection."""

    faces = np.asarray(mesh.faces, dtype=np.int64)
    edge_to_faces: dict[EdgeKey, list[int]] = {}
    for fid, tri in enumerate(faces):
        for edge in face_edges(tri):
            edge_to_faces.setdefault(edge, []).append(int(fid))
    unique_edges = tuple(sorted(edge_to_faces.keys()))
    key_to_index = {edge: int(i) for i, edge in enumerate(unique_edges)}
    return {"edge_to_faces": edge_to_faces, "unique_edges": unique_edges, "key_to_index": key_to_index}


def _refine_fragmented_ring_band(
    *,
    vertices: np.ndarray,
    edges: set[EdgeKey],
    initial_ring: RingFit,
    min_ring_points: int,
    min_radius: float,
) -> tuple[set[EdgeKey], tuple[int, ...], RingFit, dict[str, object]] | None:
    """Extract the dominant circular radial band from fragmented rim evidence.

    Damaged imported meshes can produce many short, disconnected selected edge
    fragments around the same physical bore mouth.  A single least-squares fit
    over all endpoints is easily pulled by spokes/chamfers/scraps.  This bounded
    refinement keeps the evidence workflow mesh-only:

    1. use the first fit only as a rough frame,
    2. bin selected edge midpoints by radial distance in that frame,
    3. keep the densest radial band near the opening plane,
    4. refit the ring from vertices of those inlier edges.

    No graph search, no path bridging, no flood fill.  O(N) over selected edges
    plus tiny fixed iteration count.
    """

    normalized_edges = {normalize_edge(edge) for edge in edges}
    if len(normalized_edges) < int(min_ring_points):
        return None

    center = np.asarray(initial_ring.center, dtype=float).reshape(3)
    axis = unit_vector(initial_ring.axis)
    radius0 = float(initial_ring.radius)
    if radius0 <= max(float(min_radius), 1.0e-12):
        return None

    edge_items: list[tuple[EdgeKey, float, float, float, float]] = []
    edge_lengths: list[float] = []
    radial_values: list[float] = []
    axial_values: list[float] = []

    for edge in normalized_edges:
        a, b = edge
        if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
            continue
        pa = vertices[a, :3]
        pb = vertices[b, :3]
        mid = 0.5 * (pa + pb)
        vec = pb - pa
        length = float(np.linalg.norm(vec))
        if not np.isfinite(length) or length <= 1.0e-12:
            continue
        rel = mid - center
        axial = float(np.dot(rel, axis))
        radial_vec = rel - axis * axial
        radial = float(np.linalg.norm(radial_vec))
        if not np.isfinite(radial) or radial <= 1.0e-12:
            continue
        tangent = vec / length
        radial_unit = radial_vec / max(radial, 1.0e-12)
        axial_tangent = abs(float(np.dot(tangent, axis)))
        radial_tangent = abs(float(np.dot(tangent, radial_unit)))
        edge_items.append((edge, radial, abs(axial), axial_tangent, radial_tangent))
        edge_lengths.append(length)
        radial_values.append(radial)
        axial_values.append(abs(axial))

    if len(edge_items) < int(min_ring_points):
        return None

    median_edge_length = float(median(edge_lengths)) if edge_lengths else 1.0
    axial_arr = np.asarray(axial_values, dtype=float)
    radial_arr = np.asarray(radial_values, dtype=float)

    # Keep a wide but finite plane slab first.  The slab is wide enough for
    # tessellated/chamfered rims but prevents unrelated side bands from driving
    # the radial histogram.
    axial_tol = max(
        float(initial_ring.plane_rms) * 2.75,
        float(initial_ring.plane_mad) * 4.0,
        radius0 * 0.16,
        median_edge_length * 3.0,
        1.0e-6,
    )
    plane_indices = np.nonzero(axial_arr <= axial_tol)[0]
    if len(plane_indices) < int(min_ring_points):
        plane_indices = np.arange(len(edge_items), dtype=np.int64)

    plane_radials = radial_arr[plane_indices]
    if len(plane_radials) < int(min_ring_points):
        return None

    r_min = float(np.min(plane_radials))
    r_max = float(np.max(plane_radials))
    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_max <= r_min:
        return None

    bin_width = max(median_edge_length * 1.75, radius0 * 0.035, 0.18)
    bin_count = int(max(1, min(160, np.ceil((r_max - r_min) / bin_width))))
    hist, edges_bins = np.histogram(plane_radials, bins=bin_count, range=(r_min, r_max))
    if hist.size == 0 or int(hist.max()) < max(4, int(min_ring_points) // 2):
        return None

    # Phase 18: do not blindly use the single densest radial bin.  Chamfered
    # plates can contain a large mouth ring plus many small neighbouring bore
    # rings; a histogram peak alone can lock onto the wrong feature family.
    # Evaluate several strong bands and keep the one with the best geometric
    # evidence: enough support, low radial RMS, and high circularity.
    ranked_bins = sorted(range(int(hist.size)), key=lambda idx: int(hist[idx]), reverse=True)
    max_band_trials = min(12, len(ranked_bins))
    band_candidates: list[tuple[float, set[EdgeKey], tuple[int, ...], RingFit, dict[str, object]]] = []

    for trial_rank, candidate_bin in enumerate(ranked_bins[:max_band_trials]):
        lo = float(edges_bins[max(0, candidate_bin - 1)])
        hi = float(edges_bins[min(len(edges_bins) - 1, candidate_bin + 2)])
        band_values = plane_radials[(plane_radials >= lo) & (plane_radials <= hi)]
        if len(band_values) < max(4, int(min_ring_points) // 2):
            band_values = plane_radials[
                (plane_radials >= float(edges_bins[candidate_bin]))
                & (plane_radials <= float(edges_bins[candidate_bin + 1]))
            ]
        if len(band_values) == 0:
            continue

        band_center_i = float(np.median(band_values))
        band_mad_i = float(np.median(np.abs(band_values - band_center_i))) if len(band_values) else 0.0
        radial_tol_i = max(
            bin_width * 1.65,
            band_mad_i * 3.5,
            band_center_i * 0.055,
            median_edge_length * 1.75,
            1.0e-6,
        )

        edges_i: set[EdgeKey] = set()
        for edge, radial, axial_abs, axial_tangent, radial_tangent in edge_items:
            if axial_abs > axial_tol:
                continue
            if abs(radial - band_center_i) > radial_tol_i:
                continue
            # Drop obvious radial spokes when possible.
            if radial_tangent > 0.90 and axial_tangent < 0.40:
                continue
            edges_i.add(edge)

        if len(edges_i) < int(min_ring_points):
            continue
        vertices_i = tuple(sorted({v for edge in edges_i for v in edge}))
        if len(vertices_i) < int(min_ring_points):
            continue

        points_i = vertices[np.asarray(vertices_i, dtype=np.int64), :3]
        ring_i = fit_ring_points(
            points_i,
            axis=axis,
            min_points=int(min_ring_points),
            min_radius=float(min_radius),
        )
        if ring_i is None:
            continue

        support = min(float(len(edges_i)) / max(float(len(normalized_edges)), 1.0), 1.0)
        radius_delta_rel = abs(float(ring_i.radius) - radius0) / max(radius0, 1.0e-9)
        score = (
            3.50 * float(ring_i.radius_rel_rms)
            + 0.70 * float(ring_i.plane_rel_rms)
            + 0.18 * radius_delta_rel
            - 0.55 * float(ring_i.circularity)
            - 0.30 * support
            + 0.025 * float(trial_rank)
        )
        band_candidates.append(
            (
                float(score),
                edges_i,
                vertices_i,
                ring_i,
                {
                    "trial_rank": int(trial_rank),
                    "histogram_bin": int(candidate_bin),
                    "histogram_peak_count": int(hist[candidate_bin]),
                    "band_center_radius": float(band_center_i),
                    "band_radius_tolerance": float(radial_tol_i),
                    "band_inlier_edge_count": int(len(edges_i)),
                    "band_inlier_vertex_count": int(len(vertices_i)),
                    "band_score": float(score),
                    "support_fraction": float(support),
                },
            )
        )

    if not band_candidates:
        return None

    band_candidates.sort(key=lambda item: item[0])
    _, inlier_edges, inlier_vertices, refined_ring, band_choice_diagnostics = band_candidates[0]
    band_center = float(band_choice_diagnostics.get("band_center_radius", refined_ring.radius))
    radial_tol = float(band_choice_diagnostics.get("band_radius_tolerance", 0.0))
    best_bin = int(band_choice_diagnostics.get("histogram_bin", int(np.argmax(hist))))

    # One more bounded inlier pass in the refined frame, so the returned edges
    # match the final measured ring instead of the rough initial band.
    center2 = np.asarray(refined_ring.center, dtype=float).reshape(3)
    axis2 = unit_vector(refined_ring.axis)
    radius2 = float(refined_ring.radius)
    radial_tol2 = max(
        float(refined_ring.radius_mad) * 4.0,
        float(refined_ring.radius_rms) * 2.0,
        radius2 * 0.075,
        median_edge_length * 2.5,
        1.0e-6,
    )
    axial_tol2 = max(
        float(refined_ring.plane_mad) * 4.0,
        float(refined_ring.plane_rms) * 2.5,
        radius2 * 0.12,
        median_edge_length * 3.0,
        1.0e-6,
    )
    inlier_edges2: set[EdgeKey] = set()
    for edge in inlier_edges:
        a, b = edge
        mid = 0.5 * (vertices[a, :3] + vertices[b, :3])
        rel = mid - center2
        axial = float(np.dot(rel, axis2))
        radial_vec = rel - axis2 * axial
        radial = float(np.linalg.norm(radial_vec))
        if abs(axial) <= axial_tol2 and abs(radial - radius2) <= radial_tol2:
            inlier_edges2.add(edge)

    if len(inlier_edges2) >= int(min_ring_points):
        inlier_vertices2 = tuple(sorted({v for edge in inlier_edges2 for v in edge}))
        if len(inlier_vertices2) >= int(min_ring_points):
            refined_points2 = vertices[np.asarray(inlier_vertices2, dtype=np.int64), :3]
            refined_ring2 = fit_ring_points(
                refined_points2,
                axis=axis2,
                min_points=int(min_ring_points),
                min_radius=float(min_radius),
            )
            if refined_ring2 is not None:
                inlier_edges = inlier_edges2
                inlier_vertices = inlier_vertices2
                refined_ring = refined_ring2

    diagnostics: dict[str, object] = {
        "mode": "fragmented_radial_band_refinement",
        "used": False,
        "input_edge_count": len(normalized_edges),
        "plane_candidate_edge_count": int(len(plane_indices)),
        "band_center_radius": float(band_center),
        "band_radius_tolerance": float(radial_tol),
        "band_inlier_edge_count": int(len(inlier_edges)),
        "band_inlier_vertex_count": int(len(inlier_vertices)),
        "initial_radius": float(initial_ring.radius),
        "initial_radius_rel_rms": float(initial_ring.radius_rel_rms),
        "initial_circularity": float(initial_ring.circularity),
        "refined_radius": float(refined_ring.radius),
        "refined_radius_rel_rms": float(refined_ring.radius_rel_rms),
        "refined_circularity": float(refined_ring.circularity),
        "refined_confidence": float(refined_ring.confidence),
        "axial_tolerance": float(axial_tol),
        "radial_bin_width": float(bin_width),
        "histogram_peak_count": int(hist[best_bin]),
        "multi_band_trial_count": int(len(band_candidates)),
        "selected_band_score": float(band_choice_diagnostics.get("band_score", 0.0)),
        "selected_band_support_fraction": float(band_choice_diagnostics.get("support_fraction", 0.0)),
    }
    diagnostics.update({k: v for k, v in band_choice_diagnostics.items() if k not in diagnostics})
    return inlier_edges, inlier_vertices, refined_ring, diagnostics


def _fragmented_ring_band_candidate_list(
    *,
    vertices: np.ndarray,
    edges: set[EdgeKey],
    initial_ring: RingFit,
    min_ring_points: int,
    min_radius: float,
    max_candidates: int = 8,
) -> list[tuple[set[EdgeKey], tuple[int, ...], RingFit, dict[str, object]]]:
    """Return multiple radial-band ring observations from fragmented edge evidence."""

    normalized_edges = {normalize_edge(edge) for edge in edges}
    if len(normalized_edges) < int(min_ring_points):
        return []

    center = np.asarray(initial_ring.center, dtype=float).reshape(3)
    axis = unit_vector(initial_ring.axis)
    radius0 = float(initial_ring.radius)
    if radius0 <= max(float(min_radius), 1.0e-12):
        return []

    edge_items: list[tuple[EdgeKey, float, float, float, float]] = []
    edge_lengths: list[float] = []
    radial_values: list[float] = []
    axial_values: list[float] = []
    for edge in normalized_edges:
        a, b = edge
        if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
            continue
        pa = vertices[a, :3]
        pb = vertices[b, :3]
        mid = 0.5 * (pa + pb)
        vec = pb - pa
        length = float(np.linalg.norm(vec))
        if not np.isfinite(length) or length <= 1.0e-12:
            continue
        rel = mid - center
        axial = float(np.dot(rel, axis))
        radial_vec = rel - axis * axial
        radial = float(np.linalg.norm(radial_vec))
        if not np.isfinite(radial) or radial <= 1.0e-12:
            continue
        tangent = vec / length
        radial_unit = radial_vec / max(radial, 1.0e-12)
        axial_tangent = abs(float(np.dot(tangent, axis)))
        radial_tangent = abs(float(np.dot(tangent, radial_unit)))
        edge_items.append((edge, radial, abs(axial), axial_tangent, radial_tangent))
        edge_lengths.append(length)
        radial_values.append(radial)
        axial_values.append(abs(axial))

    if len(edge_items) < int(min_ring_points):
        return []

    median_edge_length = float(median(edge_lengths)) if edge_lengths else 1.0
    axial_arr = np.asarray(axial_values, dtype=float)
    radial_arr = np.asarray(radial_values, dtype=float)
    axial_tol = max(
        float(initial_ring.plane_rms) * 2.75,
        float(initial_ring.plane_mad) * 4.0,
        radius0 * 0.16,
        median_edge_length * 3.0,
        1.0e-6,
    )
    plane_indices = np.nonzero(axial_arr <= axial_tol)[0]
    if len(plane_indices) < int(min_ring_points):
        plane_indices = np.arange(len(edge_items), dtype=np.int64)
    plane_radials = radial_arr[plane_indices]
    if len(plane_radials) < int(min_ring_points):
        return []

    r_min = float(np.min(plane_radials))
    r_max = float(np.max(plane_radials))
    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_max <= r_min:
        return []

    bin_width = max(median_edge_length * 1.45, radius0 * 0.028, 0.14)
    bin_count = int(max(1, min(180, np.ceil((r_max - r_min) / bin_width))))
    hist, edges_bins = np.histogram(plane_radials, bins=bin_count, range=(r_min, r_max))
    if hist.size == 0:
        return []

    ranked_bins = sorted(range(int(hist.size)), key=lambda idx: int(hist[idx]), reverse=True)
    candidates: list[tuple[float, set[EdgeKey], tuple[int, ...], RingFit, dict[str, object]]] = []
    for trial_rank, candidate_bin in enumerate(ranked_bins[: min(18, len(ranked_bins))]):
        if int(hist[candidate_bin]) < max(3, int(min_ring_points) // 3):
            continue
        lo = float(edges_bins[max(0, candidate_bin - 1)])
        hi = float(edges_bins[min(len(edges_bins) - 1, candidate_bin + 2)])
        band_values = plane_radials[(plane_radials >= lo) & (plane_radials <= hi)]
        if len(band_values) == 0:
            continue
        band_center = float(np.median(band_values))
        band_mad = float(np.median(np.abs(band_values - band_center))) if len(band_values) else 0.0
        radial_tol = max(
            bin_width * 1.55,
            band_mad * 3.5,
            band_center * 0.048,
            median_edge_length * 1.55,
            1.0e-6,
        )
        inlier_edges: set[EdgeKey] = set()
        for edge, radial, axial_abs, axial_tangent, radial_tangent in edge_items:
            if axial_abs > axial_tol:
                continue
            if abs(radial - band_center) > radial_tol:
                continue
            if radial_tangent > 0.92 and axial_tangent < 0.35:
                continue
            inlier_edges.add(edge)
        if len(inlier_edges) < max(4, int(min_ring_points)):
            continue
        inlier_vertices = tuple(sorted({v for edge in inlier_edges for v in edge}))
        if len(inlier_vertices) < int(min_ring_points):
            continue
        points = vertices[np.asarray(inlier_vertices, dtype=np.int64), :3]
        ring = fit_ring_points(points, axis=axis, min_points=int(min_ring_points), min_radius=float(min_radius))
        if ring is None:
            continue
        support = min(float(len(inlier_edges)) / max(float(len(normalized_edges)), 1.0), 1.0)
        # This score is for candidate ordering only.  The actual promotion will
        # happen downstream in Recognition through candidate-specific evidence.
        score = (
            0.95 * float(ring.confidence)
            + 0.55 * float(ring.circularity)
            + 0.25 * support
            - 0.80 * float(ring.radius_rel_rms)
            - 0.12 * float(trial_rank)
        )
        candidates.append(
            (
                float(score),
                inlier_edges,
                inlier_vertices,
                ring,
                {
                    "mode": "fragmented_radial_band_candidate",
                    "trial_rank": int(trial_rank),
                    "histogram_bin": int(candidate_bin),
                    "histogram_peak_count": int(hist[candidate_bin]),
                    "band_center_radius": float(band_center),
                    "band_radius_tolerance": float(radial_tol),
                    "band_inlier_edge_count": int(len(inlier_edges)),
                    "band_inlier_vertex_count": int(len(inlier_vertices)),
                    "band_score": float(score),
                    "support_fraction": float(support),
                    "initial_radius": float(initial_ring.radius),
                    "initial_radius_rel_rms": float(initial_ring.radius_rel_rms),
                    "initial_circularity": float(initial_ring.circularity),
                    "refined_radius": float(ring.radius),
                    "refined_radius_rel_rms": float(ring.radius_rel_rms),
                    "refined_circularity": float(ring.circularity),
                    "refined_confidence": float(ring.confidence),
                    "axial_tolerance": float(axial_tol),
                    "radial_bin_width": float(bin_width),
                },
            )
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    out: list[tuple[set[EdgeKey], tuple[int, ...], RingFit, dict[str, object]]] = []
    seen: list[tuple[float, np.ndarray]] = []
    for _score, cand_edges, cand_vertices, cand_ring, cand_diag in candidates:
        center_i = np.asarray(cand_ring.center, dtype=float)
        radius_i = float(cand_ring.radius)
        duplicate = False
        for radius_j, center_j in seen:
            if abs(radius_i - radius_j) <= max(max(radius_i, radius_j) * 0.035, 0.08):
                if float(np.linalg.norm(center_i - center_j)) <= max(max(radius_i, radius_j) * 0.05, 0.35):
                    duplicate = True
                    break
        if duplicate:
            continue
        seen.append((radius_i, center_i))
        out.append((cand_edges, cand_vertices, cand_ring, cand_diag))
        if len(out) >= int(max_candidates):
            break
    return out


def _opening_measurement_from_ring_edges(
    *,
    vertices: np.ndarray,
    unique_edges: tuple[EdgeKey, ...],
    input_edge_count: int,
    measurement_edges: set[EdgeKey],
    selected_vertices: tuple[int, ...],
    ring: RingFit,
    component_strategy: str,
    extra_diagnostics: Mapping[str, object] | None = None,
) -> BoreOpeningMeasurement:
    graph = edge_graph_stats(measurement_edges, vertices=vertices)
    median_edge_length = float(graph.get("median_edge_length", 1.0))
    endpoint_gap = float(graph.get("endpoint_gap", 0.0))
    endpoint_gap_ratio = float(endpoint_gap / max(median_edge_length, 1.0e-12))
    closed = bool(graph.get("closed", False))
    open_endpoint_count = int(graph.get("open_endpoint_count", 0))
    branch_vertex_count = int(graph.get("branch_vertex_count", 0))
    near_closed = bool(
        (not closed)
        and int(graph.get("component_count", 1)) == 1
        and open_endpoint_count <= 2
        and branch_vertex_count == 0
        and endpoint_gap_ratio <= 3.5
    )
    closure_score = 1.0 if closed else (0.72 if near_closed else 0.30)
    branch_score = 1.0 - clamp(branch_vertex_count / 4.0, 0.0, 1.0)
    size_score = clamp((len(measurement_edges) - 5) / 30.0, 0.0, 1.0)
    confidence = clamp(
        0.45 * float(ring.confidence)
        + 0.25 * closure_score
        + 0.20 * branch_score
        + 0.10 * size_score,
        0.0,
        1.0,
    )
    radial_envelope = _measure_opening_radial_envelope(
        vertices=vertices,
        measurement_edges=measurement_edges,
        selected_vertices=selected_vertices,
        ring=ring,
    )
    diagnostics = {
        "mode": "bore_opening_candidate_measurement",
        "input_edge_count": int(input_edge_count),
        "measured_component_edge_count": int(len(measurement_edges)),
        "component_strategy": str(component_strategy),
        "median_edge_length": float(median_edge_length),
        "axis_hint": axis_hint(ring.axis),
        "ring_fit": dict(ring.diagnostics),
        "radial_envelope": radial_envelope.to_dict(),
        "radius_min": float(radial_envelope.radius_min),
        "radius_nominal": float(radial_envelope.radius_nominal),
        "radius_max": float(radial_envelope.radius_max),
        "diameter_min": float(radial_envelope.diameter_min),
        "diameter_nominal": float(radial_envelope.diameter_nominal),
        "diameter_max": float(radial_envelope.diameter_max),
        "radial_spread": float(radial_envelope.radial_spread),
        "radial_spread_ratio": float(radial_envelope.radial_spread_ratio),
        "coarse_polygonal": bool(radial_envelope.coarse_polygonal),
        "graph": {k: v for k, v in graph.items() if k != "endpoint_vertices"},
    }
    if extra_diagnostics:
        diagnostics["candidate_source"] = dict(extra_diagnostics)
    return BoreOpeningMeasurement(
        edge_ids=tuple(sorted(_keys_to_ids(measurement_edges, unique_edges))),
        edge_count=int(len(measurement_edges)),
        vertex_ids=tuple(int(v) for v in selected_vertices),
        vertex_count=int(len(selected_vertices)),
        center=ring.center,
        axis=ring.axis,
        radius=float(ring.radius),
        diameter=float(ring.diameter),
        closed=closed,
        near_closed=near_closed,
        endpoint_gap=endpoint_gap,
        endpoint_gap_ratio=endpoint_gap_ratio,
        branch_vertex_count=branch_vertex_count,
        open_endpoint_count=open_endpoint_count,
        component_count=int(graph.get("component_count", 1)),
        plane_rms=float(ring.plane_rms),
        plane_rel_rms=float(ring.plane_rel_rms),
        radius_rms=float(ring.radius_rms),
        radius_rel_rms=float(ring.radius_rel_rms),
        radius_mad=float(ring.radius_mad),
        circularity=float(ring.circularity),
        confidence=float(confidence),
        radius_min=float(radial_envelope.radius_min),
        radius_max=float(radial_envelope.radius_max),
        diameter_min=float(radial_envelope.diameter_min),
        diameter_max=float(radial_envelope.diameter_max),
        radial_spread=float(radial_envelope.radial_spread),
        radial_spread_ratio=float(radial_envelope.radial_spread_ratio),
        coarse_polygonal=bool(radial_envelope.coarse_polygonal),
        radial_envelope=radial_envelope.to_dict(),
        diagnostics=diagnostics,
    )


def _opening_candidate_preference_score(measurement: BoreOpeningMeasurement) -> float:
    support = min(float(measurement.edge_count) / 80.0, 1.0)
    return (
        1.15 * float(measurement.confidence)
        + 0.75 * float(measurement.circularity)
        + 0.22 * support
        - 0.75 * float(measurement.radius_rel_rms)
        - 0.10 * float(measurement.plane_rel_rms)
    )




# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _validate_mesh(mesh: trimesh.Trimesh) -> None:
    if mesh is None:
        raise ValueError("No mesh provided.")
    vertices = getattr(mesh, "vertices", None)
    faces = getattr(mesh, "faces", None)
    if vertices is None or faces is None:
        raise ValueError("Mesh must provide vertices and faces.")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Mesh is empty.")


def _mesh_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError("Mesh vertices must be an (N, 3) array.")
    return vertices[:, :3].astype(float, copy=False)


def _edge_ids_to_keys(edge_ids: tuple[int, ...], unique_edges: tuple[EdgeKey, ...]) -> list[EdgeKey]:
    result: list[EdgeKey] = []
    edge_count = len(unique_edges)
    for raw in edge_ids:
        eid = int(raw)
        if eid < 0 or eid >= edge_count:
            raise ValueError(f"Edge ID {eid} out of range for {edge_count} mesh edges.")
        result.append(normalize_edge(unique_edges[eid]))
    return result


def _keys_to_ids(edges: set[EdgeKey], unique_edges: tuple[EdgeKey, ...]) -> tuple[int, ...]:
    key_to_index = {normalize_edge(edge): int(i) for i, edge in enumerate(unique_edges)}
    return tuple(sorted(int(key_to_index[normalize_edge(edge)]) for edge in edges if normalize_edge(edge) in key_to_index))


def _valid_face_ids(mesh: trimesh.Trimesh, face_ids: Iterable[int]) -> tuple[int, ...]:
    max_id = int(len(mesh.faces)) - 1
    return tuple(sorted({int(fid) for fid in face_ids if 0 <= int(fid) <= max_id}))


def _face_normals(mesh: trimesh.Trimesh, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    try:
        normals = np.asarray(mesh.face_normals, dtype=float)
        if normals.shape == (len(faces), 3):
            lengths = np.linalg.norm(normals, axis=1)
            out = np.zeros_like(normals, dtype=float)
            valid = np.isfinite(lengths) & (lengths > 1.0e-12)
            out[valid] = normals[valid] / lengths[valid, None]
            return out
    except Exception:
        pass
    tri = vertices[faces[:, :3], :3]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    out = np.zeros_like(normals, dtype=float)
    valid = np.isfinite(lengths) & (lengths > 1.0e-12)
    out[valid] = normals[valid] / lengths[valid, None]
    return out


__all__ = [
    "EdgeKey",
    "BoreOpeningRadialEnvelope",
    "BoreOpeningMeasurement",
    "BoreRegionMeasurement",
    "BoreTwoOpeningMeasurement",
    "measure_bore_opening",
    "measure_bore_opening_candidates",
    "measure_bore_opening_component_candidates",
    "measure_two_opening_bore_frame",
    "measure_bore_region",
    "measure_bore_from_region_faces",
    "build_edge_table",
    "normalize_edge",
    "face_edges",
    "connected_edge_components",
    "edge_graph_stats",
]
