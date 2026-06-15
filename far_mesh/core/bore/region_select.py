"""Clean Bore region selection.

This module is a closed, neutral selection entity for the BoreTool.

Its job is deliberately narrow:

    selected edge / rim evidence
        -> infer a circular-ish opening frame
        -> project a finite volume into the mesh
        -> collect RegionData from that volume
        -> hand RegionData to Recognition

It does not recognize features, split borehole/chamfer/pocket surfaces, choose a
"bore side", find rebuild targets, or authorize deletion.  Recognition owns
feature candidates.  Rebuild-target policy owns delete patches.  Rebuild owns
mesh mutation and watertight validation.

The public Region Select contract is ``select_region_data(...) -> RegionData``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import trimesh

from .geometry import canonical_axis
from .types import RegionData
from .measure import measure_bore_opening_candidates
from .topology import (
    boundary_edges_for_face_patch as _topology_boundary_edges_for_face_patch,
    connected_edge_components as _topology_connected_edge_components,
    face_edges as _topology_face_edges,
    normalize_edge as _topology_normalize_edge,
)

EdgeKey = tuple[int, int]
Vector3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class BoreRegionSelection:
    """Result of selecting faces inside a closed boundary loop.

    This cap/interior helper is separate from neutral RegionData volume selection.
    """

    face_ids: tuple[int, ...]
    loop_vertices: tuple[int, ...]
    loop_edges: tuple[EdgeKey, ...]
    closed: bool
    component_count: int
    selected_component_size: int
    diagnostics: dict[str, object] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def faces_inside_boundary(mesh: trimesh.Trimesh, edge_ids: Iterable[int]) -> tuple[int, ...]:
    """Return face IDs inside the closed loop formed by selected edge IDs."""

    return select_bore_region(mesh, edge_ids).face_ids



def region_faces(mesh: trimesh.Trimesh, edge_ids: Iterable[int]) -> tuple[int, ...]:
    """Return RegionData/context faces anchored to selected edges."""

    return select_region_data(mesh, edge_ids).face_ids



def select_bore_region(mesh: trimesh.Trimesh, edge_ids: Iterable[int]) -> BoreRegionSelection:
    """Select the smaller face component produced by cutting a closed loop.

    This helper is intentionally independent from neutral RegionData volume selection.
    """

    _validate_mesh(mesh)
    selected_edge_ids = tuple(int(v) for v in edge_ids)
    if not selected_edge_ids:
        raise ValueError("No selected Bore boundary edges.")

    edge_table = _build_edge_table(mesh)
    loops = _selected_edges_to_ordered_loops(selected_edge_ids, edge_table)
    closed_loops = [loop for loop in loops if bool(loop.get("closed", False))]
    if not closed_loops:
        raise ValueError("Selected edges do not form a closed boundary loop.")

    all_inside: set[int] = set()
    loop_diagnostics: list[dict[str, object]] = []
    primary_loop_edges: tuple[EdgeKey, ...] = ()
    primary_loop_vertices: tuple[int, ...] = ()
    total_components = 0

    for loop_index, loop in enumerate(closed_loops):
        selected_faces, cut_diag = _faces_inside_cut(mesh, loop, edge_table)
        all_inside.update(selected_faces)
        total_components += int(cut_diag.get("component_count", 0))

        loop_edges = tuple(_normalize_edge(edge) for edge in tuple(loop.get("edges", ())))
        loop_vertices = tuple(int(v) for v in tuple(loop.get("vertices", ())))
        if not primary_loop_edges:
            primary_loop_edges = loop_edges
            primary_loop_vertices = loop_vertices

        loop_diagnostics.append(
            {
                "loop_index": int(loop_index),
                "edge_count": int(len(loop_edges)),
                "vertex_count": int(len(loop_vertices)),
                **cut_diag,
            }
        )

    face_ids = tuple(sorted(all_inside))
    if not face_ids:
        raise ValueError("Closed boundary found, but no inside faces were selected.")

    return BoreRegionSelection(
        face_ids=face_ids,
        loop_vertices=primary_loop_vertices,
        loop_edges=primary_loop_edges,
        closed=True,
        component_count=int(total_components),
        selected_component_size=int(len(face_ids)),
        diagnostics={
            "mode": "cap_inside",
            "selected_edge_count": int(len(selected_edge_ids)),
            "loop_count": int(len(loops)),
            "closed_loop_count": int(len(closed_loops)),
            "selected_face_count": int(len(face_ids)),
            "loop_diagnostics": tuple(loop_diagnostics),
        },
    )



def select_region_data(
    mesh: trimesh.Trimesh,
    edge_ids: Iterable[int],
    *,
    # Configuration parameters control RegionData collection. They are not
    # feature-recognition gates.
    max_normal_axis_dot: float = 0.72,
    radius_tolerance_factor: float = 0.45,
    edge_scale_tolerance_factor: float = 3.0,
    min_bore_loop_edges: int = 12,
    min_bore_seed_faces: int = 3,
    volume_ring_depth: int = 96,
    max_volume_face_count: int = 0,
    distance_edge_scale_factor: float = 5.0,
    bbox_edge_scale_factor: float = 8.0,
    radial_padding_radius_factor: float = 0.28,
    radial_padding_edge_factor: float = 4.0,
    axial_depth_radius_factor: float = 12.0,
    axial_depth_edge_factor: float = 96.0,
) -> RegionData:
    """Collect RegionData from a picked rim/opening.

    Clean selection contract:

    1. Resolve stable selected edge IDs.
    2. If the user picked only a short feature edge, infer a rim-like edge
       component from local boundary/crease topology.  If the user already
       selected a full rim/ring, use those edges directly.
    3. Measure a circular-ish opening frame from the rim evidence.
    4. Project a finite cylinder along the measured axis.
    5. Collect all mesh faces whose centroid/vertices intersect that cylinder.
    6. Return this RegionData and all cutout information for Recognition.

    No normal gating, bore-side filtering, chamfer splitting, wall promotion,
    candidate identity, target construction, or rebuild authorization happens
    here.
    """

    _validate_mesh(mesh)

    selected_edge_ids = tuple(int(v) for v in edge_ids)
    if not selected_edge_ids:
        raise ValueError("No selected Bore boundary edges.")

    vertices = _mesh_vertices(mesh)
    faces = _mesh_faces(mesh)
    edge_table = _build_edge_table(mesh)
    selected_edges, invalid_edge_ids = _edge_ids_to_keys(selected_edge_ids, edge_table)
    if not selected_edges:
        raise ValueError(
            "Selected Bore edge IDs did not resolve to mesh edges. "
            f"Invalid IDs: {invalid_edge_ids[:12]}"
        )

    selected_edge_set = set(selected_edges)
    selected_vertex_ids = tuple(sorted({int(v) for edge in selected_edge_set for v in edge}))
    loops = _selected_edges_to_ordered_loops(selected_edge_ids, edge_table)
    loop_count = int(len(loops))
    closed_loop_count = int(sum(1 for loop in loops if bool(loop.get("closed", False))))

    edge_to_faces = edge_table["edge_to_faces"]  # type: ignore[index]
    if not isinstance(edge_to_faces, dict):
        raise ValueError("Invalid edge table: edge_to_faces missing.")

    face_normals = _face_normals(mesh, vertices, faces)
    face_centroids = vertices[faces[:, :3]].mean(axis=1)

    # The raw selected edge cloud remains diagnostics.  The opening evidence may
    # be the raw selection itself or a local rim-like component grown from a
    # single picked feature edge.
    rim_edges, rim_edge_ids, rim_diag = _opening_rim_edges_from_selection(
        mesh=mesh,
        selected_edges=selected_edge_set,
        selected_edge_ids=selected_edge_ids,
        edge_table=edge_table,
        vertices=vertices,
        faces=faces,
        face_normals=face_normals,
        min_loop_edges=max(3, int(min_bore_loop_edges)),
    )
    if not rim_edges:
        rim_edges = set(selected_edge_set)
        rim_edge_ids = tuple(selected_edge_ids)
        rim_diag = {**rim_diag, "fallback_used": True, "fallback_reason": "empty_inferred_rim_use_raw_selected_edges"}

    rim_vertex_ids = tuple(sorted({int(v) for edge in rim_edges for v in edge})) or selected_vertex_ids

    raw_seed_faces = _seed_faces_from_selected_edges(
        selected_edges=selected_edge_set,
        edge_to_faces=edge_to_faces,
        face_count=len(faces),
    )
    if len(raw_seed_faces) < max(1, int(min_bore_seed_faces)):
        raw_seed_faces = tuple(sorted(set(raw_seed_faces) | set(_seed_faces_from_vertices(faces, selected_vertex_ids))))

    seed_faces = _seed_faces_from_selected_edges(
        selected_edges=set(rim_edges),
        edge_to_faces=edge_to_faces,
        face_count=len(faces),
    )
    if len(seed_faces) < max(1, int(min_bore_seed_faces)):
        seed_faces = tuple(sorted(set(seed_faces) | set(raw_seed_faces) | set(_seed_faces_from_vertices(faces, rim_vertex_ids))))
    if not seed_faces:
        raise ValueError("Selected Bore/opening evidence has no adjacent mesh faces.")

    median_edge_length = _median_edge_length(vertices, set(rim_edges))
    if median_edge_length <= 0.0:
        median_edge_length = _median_edge_length(vertices, selected_edge_set)
    if median_edge_length <= 0.0:
        median_edge_length = _median_mesh_edge_length(vertices, edge_table)

    center, axis, radius = _opening_frame_from_edges(
        mesh=mesh,
        vertices=vertices,
        edge_ids=tuple(rim_edge_ids),
        rim_vertex_ids=rim_vertex_ids,
    )
    fallback_center = np.asarray(center, dtype=float).reshape(3)
    fallback_axis = _unit_vector(axis, fallback=(0.0, 0.0, 1.0))
    fallback_radius = float(radius)

    best_opening_candidate = None
    measured_frame_diag: dict[str, object]
    try:
        best_opening_candidate, measured_frame_diag = _choose_measured_opening_frame(
            mesh=mesh,
            selected_edge_ids=tuple(rim_edge_ids),
            fallback_center=fallback_center,
            fallback_axis=fallback_axis,
            fallback_radius=fallback_radius,
            max_candidates=6,
        )
    except Exception as exc:
        measured_frame_diag = {
            "used": False,
            "reason": "opening_candidate_measurement_failed",
            "error": str(exc),
        }
        best_opening_candidate = None

    if best_opening_candidate is not None:
        center_arr = np.asarray(getattr(best_opening_candidate, "center"), dtype=float).reshape(3)
        axis_arr = canonical_axis(getattr(best_opening_candidate, "axis"))
        radius_value = float(getattr(best_opening_candidate, "radius"))
        frame_source = "measured_opening_candidate"
    else:
        center_arr = fallback_center
        axis_arr = fallback_axis
        radius_value = float(fallback_radius)
        frame_source = "selected_edge_svd_fallback"

    if not np.isfinite(radius_value) or radius_value <= 1.0e-9:
        radius_value = max(float(median_edge_length), 1.0)

    face_ids_set, cylinder_diag = _project_cylindrical_aoi_faces(
        vertices=vertices,
        faces=faces,
        face_centroids=face_centroids,
        center=center_arr,
        axis=axis_arr,
        radius=float(radius_value),
        median_edge_length=float(median_edge_length),
        seed_face_ids=seed_faces,
        radial_padding_radius_factor=float(radial_padding_radius_factor),
        radial_padding_edge_factor=float(radial_padding_edge_factor),
        axial_depth_radius_factor=float(axial_depth_radius_factor),
        axial_depth_edge_factor=float(axial_depth_edge_factor),
        max_volume_face_count=int(max_volume_face_count),
    )
    if not face_ids_set:
        face_ids_set = set(seed_faces)
        cylinder_diag = {
            **cylinder_diag,
            "fallback_used": True,
            "fallback_reason": "cylindrical_projection_returned_empty_use_seed_faces",
        }

    face_ids = tuple(sorted(face_ids_set))
    region_preview_face_ids = face_ids

    vertex_ids = tuple(sorted({int(v) for fid in face_ids for v in faces[int(fid), :3]}))
    edge_keys = tuple(sorted({edge for fid in face_ids for edge in _face_edges(faces[int(fid)])}))
    boundary_edges = _patch_boundary_edges(faces, face_ids)
    boundary_loops = _boundary_loops_from_edges(boundary_edges)

    # Recognition may need local topology inside the RegionData cutout.  This adjacency is
    # neutral evidence and does not imply candidate ownership.
    full_adjacency = _build_face_adjacency(faces)
    adjacency_subset = _face_adjacency_subset(full_adjacency, face_ids)

    ignored_legacy_params = {
        "max_normal_axis_dot": float(max_normal_axis_dot),
        "radius_tolerance_factor": float(radius_tolerance_factor),
        "edge_scale_tolerance_factor": float(edge_scale_tolerance_factor),
        "volume_ring_depth": int(volume_ring_depth),
        "distance_edge_scale_factor": float(distance_edge_scale_factor),
        "bbox_edge_scale_factor": float(bbox_edge_scale_factor),
        "reason": "clean selector projects a measured volume cutout and does not use feature/normal/side gates",
    }

    diagnostics: dict[str, object] = {
        "pipeline_stage": "region_select_cylindrical_region_data_evidence",
        "mode": "selected_edge_opening_to_cylindrical_region_data_cutout",
        "selection_contract": "edge_opening_to_measured_cylindrical_region_data_no_feature_bias",
        "semantic_role": "neutral_region_data_cutout_only",
        "feature_authority": False,
        "recognition_authority": False,
        "rebuild_authority": False,
        "recognition_required": True,
        "feature_recognition_used": False,
        "normal_gating_used": False,
        "seed_side_filter_used": False,
        "bore_side_filter_used": False,
        "chamfer_split_used": False,
        "candidate_identity_used": False,
        "target_policy_used": False,
        "selected_edge_count": int(len(selected_edge_ids)),
        "resolved_selected_edge_count": int(len(selected_edge_set)),
        "invalid_selected_edge_id_count": int(len(invalid_edge_ids)),
        "loop_count": int(loop_count),
        "closed_loop_count": int(closed_loop_count),
        "primary_anchor_edge_count": int(len(rim_edges)),
        "raw_non_anchor_edge_count": int(max(0, len(selected_edge_set) - len(rim_edges))),
        "volumetric_anchor_policy": "picked_or_inferred_rim_projected_as_cylindrical_region_data",
        "raw_adjacent_face_count": int(len(raw_seed_faces)),
        "direct_selected_edge_adjacent_face_count": int(len(raw_seed_faces)),
        "direct_selected_edge_adjacent_face_ids": tuple(sorted(raw_seed_faces)),
        "seed_face_count": int(len(seed_faces)),
        "selected_face_count": int(len(face_ids)),
        "volume_face_count": int(len(face_ids)),
        "region_preview_face_count": int(len(region_preview_face_ids)),
        "region_collection_policy": "measure_opening_project_cylinder_collect_region_faces_no_feature_classification",
        "frame_source": str(frame_source),
        "frame_measurement": dict(measured_frame_diag),
        "best_opening_candidate": _opening_candidate_summary(best_opening_candidate),
        "opening_rim_inference": dict(rim_diag),
        "median_selected_edge_length": float(median_edge_length),
        "volume_selection": dict(cylinder_diag),
        "cylindrical_region_data_projection": dict(cylinder_diag),
        "projection_method": str(cylinder_diag.get("method", "measured_cylindrical_region_data_projection")),
        "projection_radial_max": float(cylinder_diag.get("outer_radius", 0.0) or 0.0),
        "projection_axial_half_depth": float(cylinder_diag.get("axial_half_depth", 0.0) or 0.0),
        "boundary_edge_count": int(len(boundary_edges)),
        "boundary_loop_count": int(len(boundary_loops)),
        "vertex_count": int(len(vertex_ids)),
        "edge_key_count": int(len(edge_keys)),
        "selected_edge_cloud": {
            "vertex_count": int(len(selected_vertex_ids)),
            "bbox_min": _to_vector3(np.min(vertices[np.asarray(selected_vertex_ids, dtype=np.int64)], axis=0)) if selected_vertex_ids else (0.0, 0.0, 0.0),
            "bbox_max": _to_vector3(np.max(vertices[np.asarray(selected_vertex_ids, dtype=np.int64)], axis=0)) if selected_vertex_ids else (0.0, 0.0, 0.0),
        },
        "cutout": {
            "face_ids": face_ids,
            "region_preview_face_ids": tuple(sorted(region_preview_face_ids)),
            "region_face_ids": tuple(sorted(region_preview_face_ids)),
            "primary_region_component_face_ids": (),
            "primary_region_face_ids": (),
            "seed_face_ids": tuple(sorted(seed_faces)),
            "direct_selected_edge_adjacent_face_ids": tuple(sorted(raw_seed_faces)),
            "opening_rim_edge_ids": tuple(int(v) for v in tuple(rim_edge_ids)),
            "opening_rim_edges": tuple(sorted(rim_edges)),
            "vertex_ids": vertex_ids,
            "edge_keys": edge_keys,
            "boundary_edges": boundary_edges,
            "face_adjacency": adjacency_subset,
        },
        "feature": {
            "class": "undecided",
            "hint": "neutral volume cutout; recognition decides whether it contains borehole/chamfer/pocket/other features",
            "rebuild_block_reason": "region_select_region_data_is_not_a_rebuild_target",
        },
        "rebuild_ready": False,
        "rebuild_block_reason": "region_select_region_data_is_not_a_rebuild_target",
        "ignored_selector_gate_parameters": ignored_legacy_params,
    }

    return RegionData(
        edge_ids=tuple(int(v) for v in selected_edge_ids),
        face_ids=face_ids,
        loop_vertices=rim_vertex_ids,
        loop_edges=tuple(sorted(rim_edges)),
        center=_to_vector3(center_arr),
        axis=_to_vector3(axis_arr),
        radius=float(radius_value),
        seed_face_ids=tuple(sorted(seed_faces)),
        region_preview_face_ids=tuple(sorted(region_preview_face_ids)),
        derived_boundary_loops=boundary_loops,
        derived_opposite_rim_edge_ids=(),
        diagnostics=diagnostics,
    )


# -----------------------------------------------------------------------------
# Neutral opening / RegionData helpers
# -----------------------------------------------------------------------------


def _opening_rim_edges_from_selection(
    *,
    mesh: trimesh.Trimesh,
    selected_edges: set[EdgeKey],
    selected_edge_ids: tuple[int, ...],
    edge_table: Mapping[str, object],
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    min_loop_edges: int,
) -> tuple[set[EdgeKey], tuple[int, ...], dict[str, object]]:
    """Return rim/opening evidence edges from the user selection.

    If the user already selected a meaningful rim cloud, use it directly.  If
    the user picked only one/few edge segments, grow a local component of
    boundary/crease edges connected to that pick.  This is opening inference, not
    feature recognition.
    """

    selected = {_normalize_edge(edge) for edge in tuple(selected_edges or ())}
    if not selected:
        return set(), (), {"used": False, "reason": "empty_selected_edges"}

    edge_key_to_id = edge_table.get("edge_key_to_id", {})
    if not isinstance(edge_key_to_id, dict):
        edge_key_to_id = {}

    if len(selected) >= max(3, int(min_loop_edges)):
        ids = tuple(sorted(int(edge_key_to_id.get(edge, -1)) for edge in selected if int(edge_key_to_id.get(edge, -1)) >= 0))
        return set(selected), ids or tuple(selected_edge_ids), {
            "used": True,
            "method": "raw_selected_edge_cloud_is_opening_rim",
            "selected_edge_count": int(len(selected)),
            "inferred_edge_count": int(len(selected)),
            "single_pick_growth_used": False,
        }

    candidate_edges = _feature_rim_candidate_edges(faces=faces, face_normals=face_normals, edge_table=edge_table)
    candidate_edges.update(selected)

    components = _topology_connected_edge_components(candidate_edges)
    selected_vertices = {int(v) for edge in selected for v in edge}
    touching: list[set[EdgeKey]] = []
    for comp in components:
        comp_vertices = {int(v) for edge in comp for v in edge}
        if comp_vertices & selected_vertices or bool(set(comp) & selected):
            touching.append(set(comp))

    if not touching:
        ids = tuple(sorted(int(edge_key_to_id.get(edge, -1)) for edge in selected if int(edge_key_to_id.get(edge, -1)) >= 0))
        return set(selected), ids or tuple(selected_edge_ids), {
            "used": False,
            "method": "single_pick_rim_growth",
            "reason": "no_touching_feature_edge_component",
            "selected_edge_count": int(len(selected)),
            "inferred_edge_count": int(len(selected)),
        }

    # Prefer the largest component that touches the picked edge.  For one-edge
    # picks this is usually the full crease/boundary rim.  Keep a cap to avoid a
    # whole model-wide crease network becoming the opening.
    touching.sort(key=lambda comp: len(comp), reverse=True)
    chosen = set(touching[0])
    max_reasonable_edges = max(24, int(min_loop_edges) * 24)
    cap_applied = False
    if len(chosen) > max_reasonable_edges:
        chosen = _local_edge_component_subset(chosen, selected, vertices, max_edges=max_reasonable_edges)
        cap_applied = True

    ids = tuple(sorted(int(edge_key_to_id.get(edge, -1)) for edge in chosen if int(edge_key_to_id.get(edge, -1)) >= 0))
    return chosen, ids or tuple(selected_edge_ids), {
        "used": True,
        "method": "selected_edge_to_local_boundary_or_crease_rim_component",
        "selected_edge_count": int(len(selected)),
        "candidate_feature_edge_count": int(len(candidate_edges)),
        "touching_component_count": int(len(touching)),
        "inferred_edge_count": int(len(chosen)),
        "cap_applied": bool(cap_applied),
        "single_pick_growth_used": True,
        "not_feature_recognition": True,
    }



def _feature_rim_candidate_edges(*, faces: np.ndarray, face_normals: np.ndarray, edge_table: Mapping[str, object]) -> set[EdgeKey]:
    edge_to_faces = edge_table.get("edge_to_faces", {})
    if not isinstance(edge_to_faces, dict):
        return set()
    normals = np.asarray(face_normals, dtype=float)
    out: set[EdgeKey] = set()
    # Boundary edges and crease/high-dihedral edges are generic rim/opening
    # evidence.  They are not classified as bore/chamfer/pocket here.
    crease_dot_threshold = float(np.cos(np.deg2rad(18.0)))
    for edge, adjacent in edge_to_faces.items():
        try:
            edge_key = _normalize_edge(edge)
            ids = tuple(int(v) for v in tuple(adjacent or ()))
        except Exception:
            continue
        if len(ids) != 2:
            out.add(edge_key)
            continue
        a, b = ids[0], ids[1]
        if a < 0 or b < 0 or a >= len(normals) or b >= len(normals):
            continue
        n0 = normals[a, :3]
        n1 = normals[b, :3]
        if not (np.all(np.isfinite(n0)) and np.all(np.isfinite(n1))):
            continue
        l0 = float(np.linalg.norm(n0))
        l1 = float(np.linalg.norm(n1))
        if l0 <= 1.0e-12 or l1 <= 1.0e-12:
            continue
        dot = abs(float(np.dot(n0 / l0, n1 / l1)))
        if dot <= crease_dot_threshold:
            out.add(edge_key)
    return out



def _local_edge_component_subset(component: set[EdgeKey], seed_edges: set[EdgeKey], vertices: np.ndarray, *, max_edges: int) -> set[EdgeKey]:
    seed_vertices = {int(v) for edge in seed_edges for v in edge}
    if not component or not seed_vertices:
        return set(component)
    adjacency: dict[int, set[int]] = defaultdict(set)
    edge_by_pair: set[EdgeKey] = set()
    for edge in component:
        a, b = _normalize_edge(edge)
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
        edge_by_pair.add((int(a), int(b)) if int(a) < int(b) else (int(b), int(a)))
    chosen_edges: set[EdgeKey] = set()
    visited_vertices: set[int] = set(seed_vertices)
    q: deque[int] = deque(sorted(seed_vertices))
    while q and len(chosen_edges) < int(max_edges):
        v = int(q.popleft())
        for nb in sorted(adjacency.get(v, ())):
            edge = (v, nb) if v < nb else (nb, v)
            if edge not in edge_by_pair or edge in chosen_edges:
                continue
            chosen_edges.add(edge)
            if nb not in visited_vertices:
                visited_vertices.add(nb)
                q.append(nb)
            if len(chosen_edges) >= int(max_edges):
                break
    return chosen_edges or set(seed_edges)



def _opening_frame_from_edges(
    *,
    mesh: trimesh.Trimesh,
    vertices: np.ndarray,
    edge_ids: tuple[int, ...],
    rim_vertex_ids: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, float]:
    del mesh, edge_ids  # measured candidates are tried separately by caller
    return _display_frame_from_selected_vertices(vertices, rim_vertex_ids)



def _project_cylindrical_aoi_faces(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_centroids: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    median_edge_length: float,
    seed_face_ids: Iterable[int],
    radial_padding_radius_factor: float,
    radial_padding_edge_factor: float,
    axial_depth_radius_factor: float,
    axial_depth_edge_factor: float,
    max_volume_face_count: int,
) -> tuple[set[int], dict[str, object]]:
    """Collect all faces intersecting a finite cylinder around the opening.

    The test uses face centroids, vertices, and edge midpoints as conservative
    mesh-native samples.  No face-normal or feature-kind filtering is applied.
    """

    face_count = int(len(faces))
    center_vec = np.asarray(center, dtype=float).reshape(3)
    axis_vec = _unit_vector(axis, fallback=(0.0, 0.0, 1.0))
    edge_scale = max(float(median_edge_length), 1.0e-9)
    radius_value = max(float(radius), edge_scale)

    radial_padding = max(
        radius_value * max(float(radial_padding_radius_factor), 0.0),
        edge_scale * max(float(radial_padding_edge_factor), 0.0),
        edge_scale,
    )
    outer_radius = float(radius_value + radial_padding)
    axial_half_depth = max(
        radius_value * max(float(axial_depth_radius_factor), 0.0),
        edge_scale * max(float(axial_depth_edge_factor), 0.0),
        edge_scale * 8.0,
    )

    tri = vertices[np.asarray(faces, dtype=np.int64)[:, :3], :3]
    cent = np.asarray(face_centroids, dtype=float)[:, :3]
    mid01 = 0.5 * (tri[:, 0, :] + tri[:, 1, :])
    mid12 = 0.5 * (tri[:, 1, :] + tri[:, 2, :])
    mid20 = 0.5 * (tri[:, 2, :] + tri[:, 0, :])
    samples = np.stack((cent, tri[:, 0, :], tri[:, 1, :], tri[:, 2, :], mid01, mid12, mid20), axis=1)

    rel = samples - center_vec.reshape(1, 1, 3)
    axial_values = np.einsum("nsp,p->ns", rel, axis_vec)
    radial_vectors = rel - axial_values[:, :, None] * axis_vec.reshape(1, 1, 3)
    radial_values = np.linalg.norm(radial_vectors, axis=2)

    finite = np.isfinite(axial_values) & np.isfinite(radial_values)
    inside_samples = finite & (np.abs(axial_values) <= axial_half_depth) & (radial_values <= outer_radius)
    inside = np.any(inside_samples, axis=1)

    selected: set[int] = {int(fid) for fid in np.where(inside)[0] if 0 <= int(fid) < face_count}
    seeds = {int(fid) for fid in tuple_ints_local(seed_face_ids) if 0 <= int(fid) < face_count}
    selected.update(seeds)

    cap_reached = False
    cap_policy = "disabled"
    if int(max_volume_face_count) > 0 and len(selected) > int(max_volume_face_count):
        # Compatibility cap only.  Keep closest faces to the cylinder surface and
        # all seeds.  Normal/feature filtering is still not used.
        cap_reached = True
        cap_policy = "closest_to_measured_cylinder_surface_keep_seeds"
        ids = np.asarray(sorted(selected), dtype=np.int64)
        cent_rel = cent[ids] - center_vec.reshape(1, 3)
        cent_ax = cent_rel @ axis_vec.reshape(3)
        cent_rad = np.linalg.norm(cent_rel - cent_ax.reshape(-1, 1) * axis_vec.reshape(1, 3), axis=1)
        score = np.abs(cent_rad - radius_value) + 0.10 * np.maximum(0.0, np.abs(cent_ax) - axial_half_depth)
        order = ids[np.argsort(score)]
        kept = set(int(v) for v in order[: max(int(max_volume_face_count), len(seeds))])
        kept.update(seeds)
        selected = kept

    return selected, {
        "used": True,
        "method": "measured_cylindrical_region_data_projection",
        "face_count": int(face_count),
        "selected_face_count": int(len(selected)),
        "seed_face_count": int(len(seeds)),
        "radius": float(radius_value),
        "radial_padding": float(radial_padding),
        "outer_radius": float(outer_radius),
        "axial_half_depth": float(axial_half_depth),
        "radial_padding_radius_factor": float(radial_padding_radius_factor),
        "radial_padding_edge_factor": float(radial_padding_edge_factor),
        "axial_depth_radius_factor": float(axial_depth_radius_factor),
        "axial_depth_edge_factor": float(axial_depth_edge_factor),
        "sample_policy": "centroid_vertices_edge_midpoints",
        "normal_gating_used": False,
        "seed_side_filter_used": False,
        "feature_recognition_used": False,
        "cap_reached": bool(cap_reached),
        "max_volume_face_count": int(max_volume_face_count),
        "cap_policy": str(cap_policy),
        "not_feature_recognition": True,
        "not_rebuild_target": True,
    }



def _choose_measured_opening_frame(
    *,
    mesh: trimesh.Trimesh,
    selected_edge_ids: tuple[int, ...],
    fallback_center: np.ndarray,
    fallback_axis: np.ndarray,
    fallback_radius: float,
    max_candidates: int = 6,
) -> tuple[object | None, dict[str, object]]:
    """Choose a measured rim/opening frame from selected or inferred rim edges."""

    candidates = tuple(
        measure_bore_opening_candidates(
            mesh,
            selected_edge_ids,
            min_ring_points=6,
            min_radius=0.05,
            max_candidates=int(max_candidates),
        )
    )
    if not candidates:
        return None, {
            "used": False,
            "method": "measured_opening_candidate_frame",
            "reason": "no_candidates",
            "candidate_count": 0,
        }

    fb_center = np.asarray(fallback_center, dtype=float).reshape(3)
    fb_axis = _unit_vector(fallback_axis, fallback=(0.0, 0.0, 1.0))
    try:
        fb_radius_float = float(fallback_radius)
    except Exception:
        fb_radius_float = 0.0
    fb_radius = fb_radius_float if np.isfinite(fb_radius_float) and fb_radius_float > 0.0 else 0.0
    max_edges = max((int(getattr(c, "edge_count", 0) or 0) for c in candidates), default=1)
    scored: list[tuple[float, object, dict[str, object]]] = []

    for idx, cand in enumerate(candidates):
        try:
            c_center = np.asarray(getattr(cand, "center", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
            c_axis = _unit_vector(getattr(cand, "axis", (0.0, 0.0, 1.0)), fallback=fb_axis)
            c_radius = float(getattr(cand, "radius", 0.0) or 0.0)
            confidence = max(0.0, min(float(getattr(cand, "confidence", 0.0) or 0.0), 1.0))
            circularity = max(0.0, min(float(getattr(cand, "circularity", 0.0) or 0.0), 1.0))
            radius_rel_rms = max(0.0, float(getattr(cand, "radius_rel_rms", 999.0) or 999.0))
            plane_rel_rms = max(0.0, float(getattr(cand, "plane_rel_rms", 999.0) or 999.0))
            edge_count = int(getattr(cand, "edge_count", 0) or 0)
        except Exception:
            continue

        support_score = min(1.0, float(edge_count) / max(float(max_edges), 1.0))
        selected_support_score = min(1.0, float(edge_count) / max(float(len(selected_edge_ids)), 1.0))
        axis_agreement = abs(float(np.dot(c_axis, fb_axis)))
        if fb_radius > 0.0 and c_radius > 0.0:
            radius_delta_ratio = abs(c_radius - fb_radius) / max(fb_radius, c_radius, 1.0e-9)
            radius_agreement = max(0.0, 1.0 - min(radius_delta_ratio / 0.42, 1.0))
        else:
            radius_delta_ratio = 0.0
            radius_agreement = 0.5
        center_delta = c_center - fb_center
        center_cross = float(np.linalg.norm(center_delta - fb_axis * float(np.dot(center_delta, fb_axis))))
        center_score = max(0.0, 1.0 - min(center_cross / max(min(c_radius, fb_radius) if fb_radius > 0.0 else c_radius, 1.0), 1.0))
        rms_score = max(0.0, 1.0 - min(radius_rel_rms / 0.24, 1.0))
        plane_score = max(0.0, 1.0 - min(plane_rel_rms / 0.20, 1.0))
        graph_bonus = 0.08 if bool(getattr(cand, "closed", False)) else (0.04 if bool(getattr(cand, "near_closed", False)) else 0.0)

        intrinsic_score = (
            0.24 * confidence
            + 0.20 * circularity
            + 0.10 * support_score
            + 0.12 * rms_score
            + 0.07 * plane_score
            + 0.06 * axis_agreement
            + graph_bonus
        )
        locality_score = 0.45 * radius_agreement + 0.30 * center_score + 0.25 * selected_support_score
        score = intrinsic_score * max(0.12, locality_score)
        diag = {
            "index": int(idx),
            "score": float(score),
            "confidence": float(confidence),
            "circularity": float(circularity),
            "radius": float(c_radius),
            "edge_count": int(edge_count),
            "radius_rel_rms": float(radius_rel_rms),
            "plane_rel_rms": float(plane_rel_rms),
            "support_score": float(support_score),
            "selected_support_score": float(selected_support_score),
            "axis_agreement": float(axis_agreement),
            "radius_agreement": float(radius_agreement),
            "center_score": float(center_score),
            "locality_score": float(locality_score),
            "intrinsic_score": float(intrinsic_score),
            "radius_delta_ratio": float(radius_delta_ratio),
            "center_cross_delta": float(center_cross),
            "closed": bool(getattr(cand, "closed", False)),
            "near_closed": bool(getattr(cand, "near_closed", False)),
            "component_strategy": str(dict(getattr(cand, "diagnostics", {}) or {}).get("component_strategy", "-")),
        }
        scored.append((float(score), cand, diag))

    if not scored:
        return None, {
            "used": False,
            "method": "measured_opening_candidate_frame",
            "reason": "no_scoreable_candidates",
            "candidate_count": int(len(candidates)),
        }

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best, best_diag = scored[0]
    best_locality = float(best_diag.get("locality_score", 0.0) or 0.0)
    best_radius_delta_ratio = float(best_diag.get("radius_delta_ratio", 0.0) or 0.0)
    accepted = bool(
        best_score >= 0.14
        and best_locality >= 0.20
        and best_radius_delta_ratio <= 0.82
        and int(best_diag.get("edge_count", 0)) >= 6
    )
    diag = {
        "used": bool(accepted),
        "method": "measured_opening_candidate_frame",
        "candidate_count": int(len(candidates)),
        "accepted_candidate_index": int(best_diag.get("index", 0)),
        "accepted_score": float(best_score),
        "accepted_locality_score": float(best_locality),
        "accepted_radius_delta_ratio": float(best_radius_delta_ratio),
        "accepted": bool(accepted),
        "fallback_used": bool(not accepted),
        "fallback_reason": "weak_or_nonlocal_measured_candidates" if not accepted else "",
        "candidate_summaries": tuple(item[2] for item in scored[:8]),
    }
    return (best if accepted else None), diag



def _opening_candidate_summary(candidate: object | None) -> dict[str, object]:
    if candidate is None:
        return {}
    try:
        radius = float(getattr(candidate, "radius", 0.0) or 0.0)
        return {
            "center": _to_vector3(getattr(candidate, "center", (0.0, 0.0, 0.0))),
            "axis": _to_vector3(getattr(candidate, "axis", (0.0, 0.0, 1.0))),
            "radius": float(radius),
            "diameter": float(getattr(candidate, "diameter", 2.0 * radius) or (2.0 * radius)),
            "circularity": float(getattr(candidate, "circularity", 0.0) or 0.0),
            "confidence": float(getattr(candidate, "confidence", 0.0) or 0.0),
            "radius_rel_rms": float(getattr(candidate, "radius_rel_rms", 0.0) or 0.0),
            "plane_rel_rms": float(getattr(candidate, "plane_rel_rms", 0.0) or 0.0),
            "edge_count": int(getattr(candidate, "edge_count", 0) or 0),
            "vertex_count": int(getattr(candidate, "vertex_count", 0) or 0),
            "closed": bool(getattr(candidate, "closed", False)),
            "near_closed": bool(getattr(candidate, "near_closed", False)),
            "component_strategy": str(dict(getattr(candidate, "diagnostics", {}) or {}).get("component_strategy", "-")),
        }
    except Exception:
        return {}


# -----------------------------------------------------------------------------
# Mesh topology / geometry helpers
# -----------------------------------------------------------------------------


def tuple_ints_local(values: Iterable[object] | object) -> tuple[int, ...]:
    try:
        return tuple(sorted({int(v) for v in tuple(values or ()) if int(v) >= 0}))
    except Exception:
        return ()



def _seed_faces_from_selected_edges(
    *,
    selected_edges: set[EdgeKey] | Iterable[EdgeKey],
    edge_to_faces: Mapping[EdgeKey, Iterable[int]],
    face_count: int,
) -> tuple[int, ...]:
    """Return faces directly adjacent to the exact selected/opening edges.

    This is neutral selection evidence: it records which mesh faces touch the
    picked rim/opening.  It does not classify those faces as borehole, chamfer,
    cap, wall, or rebuild target.
    """

    seeds: set[int] = set()
    for edge in tuple(selected_edges or ()):  # type: ignore[arg-type]
        try:
            key = _normalize_edge(edge)
        except Exception:
            continue
        for fid in tuple(edge_to_faces.get(key, ()) or ()):  # type: ignore[arg-type]
            try:
                fid_i = int(fid)
            except Exception:
                continue
            if 0 <= fid_i < int(face_count):
                seeds.add(fid_i)
    return tuple(sorted(seeds))


def _seed_faces_from_vertices(faces: np.ndarray, vertex_ids: Iterable[int]) -> tuple[int, ...]:
    """Return faces touching any selected/opening vertex.

    This is only a sparse fallback for imported meshes whose selected edge IDs
    resolve but whose edge->face table does not expose enough adjacent faces.
    It is not a feature-classification step.
    """

    selected_vertices = {int(v) for v in tuple(vertex_ids or ()) if int(v) >= 0}
    if not selected_vertices:
        return ()
    arr = np.asarray(faces, dtype=np.int64)
    out: set[int] = set()
    for fid, face in enumerate(arr):
        if any(int(v) in selected_vertices for v in face[:3]):
            out.add(int(fid))
    return tuple(sorted(out))


def _validate_mesh(mesh: trimesh.Trimesh) -> None:
    if mesh is None or not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise ValueError("A valid mesh is required for Bore selection.")
    if len(getattr(mesh, "vertices")) == 0 or len(getattr(mesh, "faces")) == 0:
        raise ValueError("Mesh has no vertices/faces for Bore selection.")



def _mesh_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError("Mesh vertices must be an Nx3 array.")
    return vertices[:, :3]



def _mesh_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] < 3:
        raise ValueError("Mesh faces must be an Nx3 array.")
    return faces[:, :3]



def _normalize_edge(edge: object) -> EdgeKey:
    return _topology_normalize_edge(edge)



def _face_edges(face: Iterable[int]) -> tuple[EdgeKey, EdgeKey, EdgeKey]:
    return _topology_face_edges(tuple(int(v) for v in tuple(face)[:3]))



def _build_edge_table(mesh: trimesh.Trimesh) -> dict[str, object]:
    faces = _mesh_faces(mesh)
    edge_to_faces: dict[EdgeKey, list[int]] = defaultdict(list)
    for fid, face in enumerate(faces):
        for edge in _face_edges(face):
            edge_to_faces[edge].append(int(fid))
    unique_edges = tuple(sorted(edge_to_faces.keys()))
    edge_key_to_id = {edge: int(i) for i, edge in enumerate(unique_edges)}
    return {
        "unique_edges": unique_edges,
        "edge_to_faces": {edge: tuple(ids) for edge, ids in edge_to_faces.items()},
        "edge_key_to_id": edge_key_to_id,
    }



def _edge_ids_to_keys(edge_ids: Iterable[int], edge_table: Mapping[str, object]) -> tuple[tuple[EdgeKey, ...], tuple[int, ...]]:
    unique_edges = edge_table.get("unique_edges", ())
    if not isinstance(unique_edges, tuple):
        raise ValueError("Invalid edge table: unique_edges missing.")
    out: list[EdgeKey] = []
    invalid: list[int] = []
    for raw_id in tuple(edge_ids or ()):
        idx = int(raw_id)
        if 0 <= idx < len(unique_edges):
            out.append(_normalize_edge(unique_edges[idx]))
        else:
            invalid.append(idx)
    return tuple(sorted(set(out))), tuple(invalid)



def _selected_edges_to_ordered_loops(selected_edge_ids: Iterable[int], edge_table: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    selected_edges, _invalid = _edge_ids_to_keys(selected_edge_ids, edge_table)
    components = _topology_connected_edge_components(set(selected_edges))
    loops: list[dict[str, object]] = []
    for comp_index, component in enumerate(components):
        degrees = _edge_degrees(component)
        closed = bool(component) and all(deg == 2 for deg in degrees.values())
        ordered_vertices = _order_component_vertices(component, closed=closed)
        loops.append(
            {
                "component_index": int(comp_index),
                "edges": tuple(sorted(component)),
                "vertices": tuple(int(v) for v in ordered_vertices),
                "edge_count": int(len(component)),
                "vertex_count": int(len({v for edge in component for v in edge})),
                "closed": bool(closed),
                "degree_histogram": {int(k): int(sum(1 for v in degrees.values() if v == k)) for k in sorted(set(degrees.values()))},
            }
        )
    return tuple(loops)



def _edge_degrees(edges: Iterable[EdgeKey]) -> dict[int, int]:
    degree: dict[int, int] = defaultdict(int)
    for a, b in edges:
        degree[int(a)] += 1
        degree[int(b)] += 1
    return dict(degree)



def _order_component_vertices(edges: Iterable[EdgeKey], *, closed: bool) -> tuple[int, ...]:
    edge_set = {_normalize_edge(edge) for edge in edges}
    if not edge_set:
        return ()
    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in edge_set:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    for values in adjacency.values():
        values.sort()
    if closed:
        start = min(adjacency)
    else:
        endpoints = sorted(v for v, nbs in adjacency.items() if len(nbs) == 1)
        start = endpoints[0] if endpoints else min(adjacency)
    ordered = [int(start)]
    prev: int | None = None
    current = int(start)
    used_edges: set[EdgeKey] = set()
    while True:
        candidates = []
        for nb in adjacency.get(current, []):
            edge = _normalize_edge((current, nb))
            if edge in used_edges:
                continue
            if prev is not None and int(nb) == int(prev) and len(adjacency.get(current, [])) > 1:
                continue
            candidates.append(int(nb))
        if not candidates:
            break
        nxt = candidates[0]
        edge = _normalize_edge((current, nxt))
        used_edges.add(edge)
        if closed and nxt == start:
            break
        ordered.append(int(nxt))
        prev, current = current, nxt
        if len(used_edges) >= len(edge_set):
            break
    return tuple(ordered)



def _faces_inside_cut(mesh: trimesh.Trimesh, loop: Mapping[str, object], edge_table: Mapping[str, object]) -> tuple[set[int], dict[str, object]]:
    faces = _mesh_faces(mesh)
    loop_edges = {_normalize_edge(edge) for edge in tuple(loop.get("edges", ())) }
    edge_to_faces = edge_table["edge_to_faces"]
    if not isinstance(edge_to_faces, dict):
        raise ValueError("Invalid edge table: edge_to_faces missing.")

    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge, adjacent_faces in edge_to_faces.items():
        edge_key = _normalize_edge(edge)
        if edge_key in loop_edges:
            continue
        ids = tuple(int(fid) for fid in tuple(adjacent_faces or ()))
        if len(ids) < 2:
            continue
        for fid in ids:
            adjacency.setdefault(fid, set()).update(other for other in ids if other != fid)

    loop_adjacent_faces = {
        int(fid)
        for edge in loop_edges
        for fid in tuple(edge_to_faces.get(edge, ()) or ())
        if 0 <= int(fid) < len(faces)
    }
    if not loop_adjacent_faces:
        return set(), {"component_count": 0, "reason": "loop_has_no_adjacent_faces"}

    visited: set[int] = set()
    components: list[set[int]] = []
    for seed in sorted(loop_adjacent_faces):
        if seed in visited:
            continue
        comp: set[int] = set()
        q: deque[int] = deque([int(seed)])
        visited.add(int(seed))
        while q:
            fid = int(q.popleft())
            comp.add(fid)
            for nb in tuple(adjacency.get(fid, ()) or ()):
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        components.append(comp)

    if not components:
        return set(), {"component_count": 0, "reason": "no_components_after_cut"}
    components.sort(key=len)
    chosen = set(components[0])
    return chosen, {
        "component_count": int(len(components)),
        "component_sizes": tuple(int(len(c)) for c in components),
        "chosen_component_size": int(len(chosen)),
        "selection_policy": "smallest_component_after_loop_cut",
    }



def _build_face_adjacency(faces: np.ndarray) -> dict[int, tuple[int, ...]]:
    edge_to_faces: dict[EdgeKey, list[int]] = defaultdict(list)
    for fid, face in enumerate(np.asarray(faces, dtype=np.int64)):
        for edge in _face_edges(face):
            edge_to_faces[edge].append(int(fid))
    adjacency: dict[int, set[int]] = {int(fid): set() for fid in range(len(faces))}
    for adjacent in edge_to_faces.values():
        if len(adjacent) < 2:
            continue
        for fid in adjacent:
            adjacency[int(fid)].update(int(other) for other in adjacent if int(other) != int(fid))
    return {int(fid): tuple(sorted(values)) for fid, values in adjacency.items()}



def _patch_boundary_edges(faces: np.ndarray, face_ids: Iterable[int]) -> tuple[EdgeKey, ...]:
    return _topology_boundary_edges_for_face_patch(faces, tuple_ints_local(face_ids))



def _face_adjacency_subset(face_adjacency: Mapping[int, tuple[int, ...]], face_ids: Iterable[int]) -> dict[int, tuple[int, ...]]:
    selected = {int(fid) for fid in tuple_ints_local(face_ids)}
    return {
        int(fid): tuple(sorted(int(nb) for nb in tuple(face_adjacency.get(int(fid), ()) or ()) if int(nb) in selected))
        for fid in sorted(selected)
    }



def _boundary_loops_from_edges(edges: Iterable[EdgeKey]) -> tuple[tuple[EdgeKey, ...], ...]:
    components = _topology_connected_edge_components({_normalize_edge(edge) for edge in tuple(edges or ())})
    return tuple(tuple(sorted(component)) for component in components)



def _median_edge_length(vertices: np.ndarray, edges: Iterable[EdgeKey]) -> float:
    values: list[float] = []
    for a, b in tuple(edges or ()):
        ia, ib = int(a), int(b)
        if 0 <= ia < len(vertices) and 0 <= ib < len(vertices):
            values.append(float(np.linalg.norm(vertices[ia, :3] - vertices[ib, :3])))
    finite = np.asarray([v for v in values if np.isfinite(v) and v > 0.0], dtype=float)
    return float(np.median(finite)) if finite.size else 0.0



def _median_mesh_edge_length(vertices: np.ndarray, edge_table: Mapping[str, object]) -> float:
    unique_edges = tuple(edge_table.get("unique_edges", ()) or ())
    if not unique_edges:
        return 1.0
    sample = unique_edges[: min(len(unique_edges), 4096)]
    value = _median_edge_length(vertices, sample)
    return float(value if value > 0.0 else 1.0)



def _unit_vector(vector: object, *, fallback: object = (0.0, 0.0, 1.0)) -> np.ndarray:
    try:
        arr = np.asarray(vector, dtype=float).reshape(3)
        length = float(np.linalg.norm(arr))
        if np.isfinite(length) and length > 1.0e-12:
            return arr / length
    except Exception:
        pass
    fb = np.asarray(fallback, dtype=float).reshape(3)
    fb_len = float(np.linalg.norm(fb))
    if np.isfinite(fb_len) and fb_len > 1.0e-12:
        return fb / fb_len
    return np.array([0.0, 0.0, 1.0], dtype=float)



def _to_vector3(value: object) -> Vector3:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if len(arr) >= 3:
            return (float(arr[0]), float(arr[1]), float(arr[2]))
    except Exception:
        pass
    return (0.0, 0.0, 0.0)



def _face_normals(mesh: trimesh.Trimesh, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    try:
        normals = np.asarray(getattr(mesh, "face_normals"), dtype=float)
        if normals.shape == (len(faces), 3):
            return normals[:, :3]
    except Exception:
        pass
    tri = vertices[np.asarray(faces, dtype=np.int64)[:, :3], :3]
    raw = np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :])
    length = np.linalg.norm(raw, axis=1)
    out = np.zeros_like(raw)
    valid = np.isfinite(length) & (length > 1.0e-12)
    out[valid] = raw[valid] / length[valid].reshape(-1, 1)
    return out



def _display_frame_from_selected_vertices(vertices: np.ndarray, vertex_ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray, float]:
    ids = tuple(sorted({int(v) for v in tuple(vertex_ids or ()) if 0 <= int(v) < len(vertices)}))
    if len(ids) < 3:
        if ids:
            pts = vertices[np.asarray(ids, dtype=np.int64), :3]
            center = pts.mean(axis=0)
        else:
            center = np.zeros(3, dtype=float)
        return center, np.array([0.0, 0.0, 1.0], dtype=float), 1.0

    pts = vertices[np.asarray(ids, dtype=np.int64), :3]
    center = pts.mean(axis=0)
    centered = pts - center.reshape(1, 3)
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        axis = _unit_vector(vh[-1], fallback=(0.0, 0.0, 1.0))
    except Exception:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    radial = centered - np.outer(centered @ axis, axis)
    radii = np.linalg.norm(radial, axis=1)
    finite = radii[np.isfinite(radii) & (radii > 0.0)]
    radius = float(np.median(finite)) if finite.size else 1.0
    return center, axis, radius
