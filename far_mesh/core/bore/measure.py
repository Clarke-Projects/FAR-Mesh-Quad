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

EdgeKey = tuple[int, int]


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


# -----------------------------------------------------------------------------
# Public measurement API
# -----------------------------------------------------------------------------


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
    use_all_fragments = bool(
        len(all_selected_vertices) >= int(min_ring_points)
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
        component_strategy = "largest_connected_component"

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
    candidates: list[BoreOpeningMeasurement] = []
    if primary is not None:
        candidates.append(primary)

    if len(all_vertices) >= int(min_ring_points):
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
    diagnostics = {
        "mode": "bore_opening_candidate_measurement",
        "input_edge_count": int(input_edge_count),
        "measured_component_edge_count": int(len(measurement_edges)),
        "component_strategy": str(component_strategy),
        "median_edge_length": float(median_edge_length),
        "axis_hint": axis_hint(ring.axis),
        "ring_fit": dict(ring.diagnostics),
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
    "BoreOpeningMeasurement",
    "BoreRegionMeasurement",
    "measure_bore_opening",
    "measure_bore_opening_candidates",
    "measure_bore_region",
    "measure_bore_from_region_faces",
    "build_edge_table",
    "normalize_edge",
    "face_edges",
    "connected_edge_components",
    "edge_graph_stats",
]
