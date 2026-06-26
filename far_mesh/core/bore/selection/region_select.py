"""Clean Bore region selection.

This module is a closed, neutral selection entity for the BoreTool.  It
translates the user's intent of selecting a feature rim/opening into a neutral
RegionData/AOI package for later stages.

Its job is deliberately narrow:

    selected edge / rim evidence
        -> infer a circular-ish opening frame for AOI construction
        -> project a finite volume into the mesh
        -> collect RegionData from that volume
        -> hand RegionData and diagnostics to Recognition

It does not recognize features, split borehole/chamfer/pocket surfaces, choose a
"bore side", find rebuild targets, or authorize deletion.  Recognition owns
feature candidates.  Rebuild-target policy owns delete patches.  Rebuild owns
mesh mutation and watertight validation.

The public Region Select contract is ``select_region_data(...) -> RegionData``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import math
import numpy as np
import trimesh

from ..geometry import canonical_axis
from ..types import RegionData
from ..measure import measure_bore_opening_candidates
from ..topology import (
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


@dataclass(frozen=True, slots=True)
class _OpeningRimFrame:
    """Neutral geometric frame for an inferred opening/rim loop.

    This is Region Select evidence only.  It is not a BORE, not a CHAMFER,
    and not a rebuild target.  It exists so a weak raw Ctrl-click edge cloud
    can be normalized into the complete circular rim the operator indicated.
    """

    center: np.ndarray
    axis: np.ndarray
    radius: float
    median_edge_length: float
    fit_edge_count: int
    fit_vertex_count: int
    plane_rel_rms: float
    radius_rel_rms: float



# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def faces_inside_boundary(mesh: trimesh.Trimesh, edge_ids: Iterable[int]) -> tuple[int, ...]:
    """Return face IDs inside the closed loop formed by selected edge IDs."""

    return select_bore_region(mesh, edge_ids).face_ids



def region_faces(mesh: trimesh.Trimesh, edge_ids: Iterable[int]) -> tuple[int, ...]:
    """Return RegionData/context faces anchored to selected edges."""

    return select_region_data(mesh, edge_ids).face_ids



def normalize_opening_rim_edge_ids_from_arrays(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    edge_index_to_vertices: np.ndarray,
    edge_to_faces: Mapping[EdgeKey, Iterable[int]] | None,
    selected_edge_ids: Iterable[int],
    min_loop_edges: int = 12,
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Normalize raw edge IDs into neutral complete opening/rim evidence.

    This is the Region Select authority used by the live Bore edge selector.
    The caller may provide only one clicked edge or a small partial arc.  This
    function does not classify a BORE or a CHAMFER; it only performs the neutral
    meaning transform:

        raw clicked edge evidence -> complete opening/rim edge IDs

    It returns stable edge indices in the caller's ``edge_index_to_vertices``
    table plus diagnostics.  If the rim cannot be inferred, it returns the raw
    selected IDs rather than inventing a feature.
    """

    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] < 3:
        return tuple(int(v) for v in tuple(selected_edge_ids or ()) if int(v) >= 0), {
            "used": False,
            "reason": "invalid_vertices",
            "authority": "region_select_neutral_opening_rim",
        }
    verts = verts[:, :3]

    face_arr = np.asarray(faces, dtype=np.int64) if faces is not None else np.empty((0, 3), dtype=np.int64)
    if face_arr.ndim != 2 or face_arr.shape[1] < 3:
        return tuple(int(v) for v in tuple(selected_edge_ids or ()) if int(v) >= 0), {
            "used": False,
            "reason": "invalid_faces",
            "authority": "region_select_neutral_opening_rim",
        }
    face_arr = face_arr[:, :3]

    edge_arr = np.asarray(edge_index_to_vertices, dtype=np.int64)
    if edge_arr.ndim != 2 or edge_arr.shape[1] < 2:
        return tuple(int(v) for v in tuple(selected_edge_ids or ()) if int(v) >= 0), {
            "used": False,
            "reason": "invalid_edge_index_to_vertices",
            "authority": "region_select_neutral_opening_rim",
        }
    edge_arr = edge_arr[:, :2]

    raw_ids = tuple(sorted({int(v) for v in tuple(selected_edge_ids or ()) if 0 <= int(v) < len(edge_arr)}))
    if not raw_ids:
        return (), {
            "used": False,
            "reason": "empty_selected_edge_ids",
            "authority": "region_select_neutral_opening_rim",
        }

    edge_key_to_id: dict[EdgeKey, int] = {}
    selected_edges: set[EdgeKey] = set()
    for idx, edge in enumerate(edge_arr):
        key = _normalize_edge((int(edge[0]), int(edge[1])))
        edge_key_to_id[key] = int(idx)
        if int(idx) in raw_ids:
            selected_edges.add(key)

    norm_edge_to_faces: dict[EdgeKey, tuple[int, ...]] = {}
    if edge_to_faces:
        for key, value in edge_to_faces.items():
            try:
                norm_key = _normalize_edge(key)
                norm_edge_to_faces[norm_key] = tuple(int(v) for v in tuple(value or ()))
            except Exception:
                continue
    else:
        tmp: dict[EdgeKey, list[int]] = defaultdict(list)
        for fid, face in enumerate(face_arr):
            for key in _face_edges(face):
                tmp[key].append(int(fid))
        norm_edge_to_faces = {key: tuple(values) for key, values in tmp.items()}

    normals = _face_normals_from_arrays(verts, face_arr)
    edge_table = {
        "edge_to_faces": norm_edge_to_faces,
        "edge_key_to_id": edge_key_to_id,
        "unique_edges": tuple(_normalize_edge(edge) for edge in edge_arr),
    }

    rim_edges, rim_ids, diag = _opening_rim_edges_from_selection(
        mesh=None,  # not interpreted by the helper
        selected_edges=selected_edges,
        selected_edge_ids=raw_ids,
        edge_table=edge_table,
        vertices=verts,
        faces=face_arr,
        face_normals=normals,
        min_loop_edges=int(min_loop_edges),
    )

    out_ids = tuple(int(v) for v in tuple(rim_ids or raw_ids) if 0 <= int(v) < len(edge_arr))
    quality = _edge_cloud_graph_quality(rim_edges or selected_edges, verts)
    opening_ledger = _opening_evidence_ledger_dict_from_arrays(
        vertices=verts,
        edge_index_to_vertices=edge_arr,
        edge_ids=out_ids,
        source="region_select.normalize_opening_rim_edge_ids_from_arrays",
    )
    return out_ids, {
        **dict(diag or {}),
        "authority": "region_select_neutral_opening_rim",
        "raw_selected_edge_count": int(len(raw_ids)),
        "normalized_edge_count": int(len(out_ids)),
        "normalized_edge_graph_quality": quality,
        "opening_evidence_ledger": opening_ledger,
        "mesh_realization_evidence_ledger": opening_ledger,
        "opening_footprint_authority": dict((opening_ledger.get("selected_authority", {}) if isinstance(opening_ledger, dict) else {}) or {}),
        "not_feature_recognition": True,
    }


def _face_normals_from_arrays(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute face normals from runtime arrays without needing a mesh object."""

    tri = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=np.int64)[:, :3], :3]
    raw = np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :])
    length = np.linalg.norm(raw, axis=1)
    out = np.zeros_like(raw)
    valid = np.isfinite(length) & (length > 1.0e-12)
    out[valid] = raw[valid] / length[valid].reshape(-1, 1)
    return out



def _opening_evidence_ledger_dict_from_arrays(
    *,
    vertices: np.ndarray,
    edge_index_to_vertices: np.ndarray,
    edge_ids: Iterable[int],
    center: object | None = None,
    axis: object | None = None,
    radius: float | None = None,
    source: str,
    support_face_ids: Iterable[int] = (),
) -> dict[str, object]:
    """Return v120 mesh-realization opening evidence without feature meaning.

    This helper is deliberately defensive.  Region Select may add its diagnostics
    when available, but RegionData creation must not fail if the optional ledger
    cannot be produced.
    """

    try:
        from .mesh_realization import build_opening_evidence_ledger_from_arrays

        ledger = build_opening_evidence_ledger_from_arrays(
            vertices=vertices,
            edge_index_to_vertices=edge_index_to_vertices,
            selected_edge_ids=tuple(int(v) for v in tuple(edge_ids or ())),
            center=center,
            axis=axis,
            radius=radius,
            source=str(source),
            support_face_ids=tuple(int(v) for v in tuple(support_face_ids or ()) if int(v) >= 0),
        )
        if ledger is None:
            return {
                "contract_type": "opening_evidence_ledger",
                "semantic_stage": "mesh_realization_evidence_ledger",
                "available": False,
                "reason": "ledger_builder_returned_none",
                "not_feature_identity": True,
                "not_surface_ownership": True,
                "not_rebuild_authority": True,
            }
        return ledger.to_dict()
    except Exception as exc:
        return {
            "contract_type": "opening_evidence_ledger",
            "semantic_stage": "mesh_realization_evidence_ledger",
            "available": False,
            "reason": "ledger_builder_failed",
            "error": str(exc),
            "not_feature_identity": True,
            "not_surface_ownership": True,
            "not_rebuild_authority": True,
        }



def _opening_probe_ledger_dict_from_region_arrays(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    region_face_ids: Iterable[int],
    seed_face_ids: Iterable[int],
    selected_edge_ids: Iterable[int],
    normalized_edge_ids: Iterable[int],
    edge_index_to_vertices: np.ndarray,
    center: object,
    axis: object,
    radius: float,
    source: str,
) -> dict[str, object]:
    """Return X1-style local probe evidence without feature authority."""

    try:
        from .mesh_realization import build_x1_style_opening_probe_ledger_from_arrays

        ledger = build_x1_style_opening_probe_ledger_from_arrays(
            vertices=vertices,
            faces=faces,
            face_normals=face_normals,
            region_face_ids=tuple(int(v) for v in tuple(region_face_ids or ()) if int(v) >= 0),
            seed_face_ids=tuple(int(v) for v in tuple(seed_face_ids or ()) if int(v) >= 0),
            selected_edge_ids=tuple(int(v) for v in tuple(selected_edge_ids or ()) if int(v) >= 0),
            normalized_edge_ids=tuple(int(v) for v in tuple(normalized_edge_ids or ()) if int(v) >= 0),
            edge_index_to_vertices=edge_index_to_vertices,
            center=center,
            axis=axis,
            radius=float(radius),
            source=str(source),
        )
        if ledger is None:
            return {
                "contract_type": "x1_style_opening_probe_ledger",
                "available": False,
                "reason": "probe_ledger_builder_returned_none",
                "not_feature_identity": True,
                "not_surface_ownership": True,
                "not_rebuild_authority": True,
            }
        return dict(ledger)
    except Exception as exc:
        return {
            "contract_type": "x1_style_opening_probe_ledger",
            "available": False,
            "reason": "probe_ledger_builder_failed",
            "error": str(exc),
            "not_feature_identity": True,
            "not_surface_ownership": True,
            "not_rebuild_authority": True,
        }


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



def _normalize_selection_seed_metadata(
    selection_metadata: Mapping[str, object] | None,
    *,
    edge_table: Mapping[str, object],
    face_count: int,
) -> dict[str, object]:
    """Normalize host/viewport raw clicked-edge metadata for Region Select.

    The raw selected edge cloud can be expanded/contaminated before BoreTool is
    called.  This metadata preserves the primitive the operator actually clicked
    so Region Select can keep rim completion local to that seed.
    """

    meta = dict(selection_metadata or {}) if isinstance(selection_metadata, Mapping) else {}
    out: dict[str, object] = {
        "available": False,
        "contract": str(meta.get("metadata_contract", meta.get("contract", "")) or ""),
    }
    unique_edges = tuple(edge_table.get("unique_edges", ()) or ())
    seed_edge_id: int | None = None
    for key in ("seed_edge_id", "clicked_edge_id", "edge_id"):
        if key in meta:
            try:
                value = int(meta.get(key))
            except Exception:
                continue
            if 0 <= value < len(unique_edges):
                seed_edge_id = value
                break
    if seed_edge_id is not None:
        edge_key = _normalize_edge(unique_edges[seed_edge_id])
        out["available"] = True
        out["seed_edge_id"] = int(seed_edge_id)
        out["seed_edge_key"] = edge_key
        out["seed_edge_vertex_ids"] = tuple(int(v) for v in edge_key)
        edge_to_faces = edge_table.get("edge_to_faces", {})
        faces_for_edge: tuple[int, ...] = ()
        if isinstance(edge_to_faces, Mapping):
            try:
                faces_for_edge = tuple(
                    int(v) for v in tuple(edge_to_faces.get(edge_key, ()) or ())
                    if 0 <= int(v) < int(face_count)
                )
            except Exception:
                faces_for_edge = ()
        out["seed_adjacent_face_ids"] = faces_for_edge

    try:
        seed_faces = tuple(int(v) for v in tuple(meta.get("seed_adjacent_face_ids", ()) or ()))
    except Exception:
        seed_faces = ()
    if seed_faces:
        valid_seed_faces = tuple(sorted({int(v) for v in seed_faces if 0 <= int(v) < int(face_count)}))
        if valid_seed_faces:
            out["available"] = True
            out["metadata_seed_adjacent_face_ids"] = valid_seed_faces
            if not out.get("seed_adjacent_face_ids"):
                out["seed_adjacent_face_ids"] = valid_seed_faces

    try:
        point_raw = tuple(float(v) for v in tuple(meta.get("seed_pick_point", ()) or ()))
        if len(point_raw) >= 3:
            out["seed_pick_point"] = (float(point_raw[0]), float(point_raw[1]), float(point_raw[2]))
    except Exception:
        pass

    for key in ("selection_origin", "edge_region_strategy", "backend"):
        if key in meta:
            out[key] = meta.get(key)
    return out


def _seed_edge_component_subset(
    *,
    selected_edges: set[EdgeKey],
    seed_edge: EdgeKey | None,
) -> set[EdgeKey]:
    if not selected_edges or seed_edge is None:
        return set()
    seed = _normalize_edge(seed_edge)
    if seed not in selected_edges:
        return {seed}
    for comp in _topology_connected_edge_components(set(selected_edges)):
        comp_set = {_normalize_edge(edge) for edge in comp}
        if seed in comp_set:
            return set(comp_set)
    return {seed}


def select_region_data(
    mesh: trimesh.Trimesh,
    edge_ids: Iterable[int],
    *,
    selection_metadata: Mapping[str, object] | None = None,
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
    seed_metadata = _normalize_selection_seed_metadata(
        selection_metadata,
        edge_table=edge_table,
        face_count=len(faces),
    )
    selected_edges, invalid_edge_ids = _edge_ids_to_keys(selected_edge_ids, edge_table)
    if not selected_edges:
        raise ValueError(
            "Selected Bore edge IDs did not resolve to mesh edges. "
            f"Invalid IDs: {invalid_edge_ids[:12]}"
        )

    selected_edge_set = set(selected_edges)
    true_seed_edge = seed_metadata.get("seed_edge_key") if bool(seed_metadata.get("available", False)) else None
    true_seed_edge_key = _normalize_edge(true_seed_edge) if isinstance(true_seed_edge, tuple) and len(true_seed_edge) >= 2 else None
    seed_component_edges = _seed_edge_component_subset(
        selected_edges=selected_edge_set,
        seed_edge=true_seed_edge_key,
    )
    rim_input_edges = set(selected_edge_set)
    rim_input_edge_ids = tuple(selected_edge_ids)
    seed_local_guard_used = False
    if true_seed_edge_key is not None and len(selected_edge_set) > max(24, int(min_bore_loop_edges) * 2):
        # For polluted automatic Ctrl-click clouds, use the actual clicked edge
        # as rim-completion authority.  The expanded cloud remains diagnostics
        # and AOI context, but it no longer chooses a remote normalized ring.
        rim_input_edges = set(seed_component_edges or {true_seed_edge_key})
        edge_key_to_id = edge_table.get("edge_key_to_id", {})
        if isinstance(edge_key_to_id, Mapping):
            ids = _edge_ids_from_keys(rim_input_edges, edge_key_to_id)
            if ids:
                rim_input_edge_ids = tuple(int(v) for v in ids)
        seed_local_guard_used = True
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
        selected_edges=rim_input_edges,
        selected_edge_ids=rim_input_edge_ids,
        edge_table=edge_table,
        vertices=vertices,
        faces=faces,
        face_normals=face_normals,
        min_loop_edges=max(3, int(min_bore_loop_edges)),
    )
    if not rim_edges:
        rim_edges = set(rim_input_edges or selected_edge_set)
        rim_edge_ids = tuple(rim_input_edge_ids or selected_edge_ids)
        rim_diag = {**rim_diag, "fallback_used": True, "fallback_reason": "empty_inferred_rim_use_raw_selected_edges"}

    rim_vertex_ids = tuple(sorted({int(v) for edge in rim_edges for v in edge})) or selected_vertex_ids

    raw_seed_faces = _seed_faces_from_selected_edges(
        selected_edges=selected_edge_set,
        edge_to_faces=edge_to_faces,
        face_count=len(faces),
    )
    true_seed_faces = tuple(int(v) for v in tuple(seed_metadata.get("seed_adjacent_face_ids", ()) or ()))
    if true_seed_faces:
        raw_seed_faces = tuple(sorted(set(true_seed_faces)))
    elif len(raw_seed_faces) < max(1, int(min_bore_seed_faces)):
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

    opening_ledger = _opening_evidence_ledger_dict_from_arrays(
        vertices=vertices,
        edge_index_to_vertices=np.asarray(tuple(edge_table.get("unique_edges", ()) or ()), dtype=np.int64),
        edge_ids=tuple(int(v) for v in tuple(rim_edge_ids)),
        center=center_arr,
        axis=axis_arr,
        radius=float(radius_value),
        source="region_select.select_region_data.opening_mesh_realization",
        support_face_ids=tuple(sorted(seed_faces)),
    )
    opening_authority = dict((opening_ledger.get("selected_authority", {}) if isinstance(opening_ledger, dict) else {}) or {})
    mesh_realization_assessment = dict((opening_ledger.get("mesh_realization_assessment", {}) if isinstance(opening_ledger, dict) else {}) or {})
    opening_probe_ledger = _opening_probe_ledger_dict_from_region_arrays(
        vertices=vertices,
        faces=faces,
        face_normals=face_normals,
        region_face_ids=tuple(sorted(face_ids)),
        seed_face_ids=tuple(sorted(seed_faces)),
        selected_edge_ids=tuple(int(v) for v in tuple(selected_edge_ids)),
        normalized_edge_ids=tuple(int(v) for v in tuple(rim_edge_ids)),
        edge_index_to_vertices=np.asarray(tuple(edge_table.get("unique_edges", ()) or ()), dtype=np.int64),
        center=center_arr,
        axis=axis_arr,
        radius=float(radius_value),
        source="region_select.select_region_data.x1_style_probe_reanalysis",
    )

    diagnostics: dict[str, object] = {
        "pipeline_stage": "region_select_selected_annular_rail_region_data_evidence",
        "mode": "selected_edge_annular_rail_to_neutral_region_data_cutout",
        "legacy_mode_alias": "selected_edge_opening_to_cylindrical_region_data_cutout",
        "selection_contract": "selected_edges_to_neutral_annular_rail_region_data_no_feature_bias",
        "selected_annular_rail_contract_v173d": "selected_edges_are_neutral_annular_rail_evidence_not_bore_opening_authority",
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
        "true_seed_metadata_available": bool(seed_metadata.get("available", False)),
        "true_seed_edge_id": seed_metadata.get("seed_edge_id", -1),
        "true_seed_adjacent_face_ids": tuple(int(v) for v in tuple(seed_metadata.get("seed_adjacent_face_ids", ()) or ())),
        "seed_local_rim_guard": {
            "used": bool(seed_local_guard_used),
            "source": "viewport_click_seed_primitive_v143" if bool(seed_metadata.get("available", False)) else "unavailable",
            "raw_selected_edge_count": int(len(selected_edge_set)),
            "rim_input_edge_count": int(len(rim_input_edges)),
            "seed_component_edge_count": int(len(seed_component_edges)),
            "rim_output_edge_count": int(len(rim_edges)),
            "rim_seed_edge_overlap": int(1 if true_seed_edge_key is not None and true_seed_edge_key in set(rim_edges) else 0),
        },
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
        "opening_evidence_ledger": opening_ledger,
        "mesh_realization_evidence_ledger": opening_ledger,
        "opening_footprint_authority": opening_authority,
        "mesh_realization_assessment": mesh_realization_assessment,
        "mesh_realization_contract": "evidence_only_between_region_select_and_recognition",
        "opening_probe_ledger": opening_probe_ledger,
        "x1_style_opening_probe_ledger": opening_probe_ledger,
        "x1_probe_reanalysis_used": bool(isinstance(opening_probe_ledger, dict) and opening_probe_ledger.get("available", True) is not False),
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
            "true_seed_edge_id": seed_metadata.get("seed_edge_id", -1),
            "true_seed_adjacent_face_ids": tuple(int(v) for v in tuple(seed_metadata.get("seed_adjacent_face_ids", ()) or ())),
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
    """Return authoritative neutral opening/rim evidence from raw picked edges.

    The viewport/SelectionController may provide only a partial arc when the
    operator Ctrl-clicks an inner chamfer seam or a plain bore rim.  That raw
    selection is useful evidence, but it is not the BoreTool semantic selection.

    Region Select owns this neutral meaning transform:

        raw selected edge IDs -> normalized complete opening/rim edge evidence

    This function therefore accepts raw selected edges only when the raw graph is
    already a coherent rim.  Otherwise it reconstructs the same circular local
    rim from boundary/crease evidence in the mesh.  The output still has no
    feature identity: it is not a BORE, not a CHAMFER, and not a rebuild target.
    """

    del mesh  # mesh object is intentionally not interpreted here; arrays/tables carry the neutral evidence.

    raw_selected = {_normalize_edge(edge) for edge in tuple(selected_edges or ())}
    if not raw_selected:
        return set(), (), {"used": False, "reason": "empty_selected_edges"}

    edge_key_to_id = edge_table.get("edge_key_to_id", {})
    if not isinstance(edge_key_to_id, dict):
        edge_key_to_id = {}

    min_edges = max(3, int(min_loop_edges))
    raw_ok, raw_quality = _edge_cloud_is_coherent_opening_rim(
        raw_selected,
        vertices=vertices,
        min_loop_edges=min_edges,
    )
    if raw_ok:
        ids = _edge_ids_from_keys(raw_selected, edge_key_to_id) or tuple(selected_edge_ids)
        return set(raw_selected), ids, {
            "used": True,
            "method": "raw_selected_edge_cloud_is_coherent_opening_rim",
            "selected_edge_count": int(len(raw_selected)),
            "inferred_edge_count": int(len(raw_selected)),
            "single_pick_growth_used": False,
            "complete_circle_recovery_used": False,
            "raw_edge_cloud_quality": raw_quality,
            "seed_local_rim_guard": {"used": False, "reason": "raw_cloud_already_coherent"},
            "not_feature_recognition": True,
        }

    # v142: if the live selector gave us a polluted multi-component cloud, do
    # not let every selected vertex become seed authority.  That was the coarse
    # mesh failure mode: the normalizer could choose a geometrically plausible
    # ring that did not belong to the user's local rim island.  Pick one raw
    # selected component as the seed-owned opening island, then restrict rim
    # completion and candidate scanning around that island.  Manual rim selection
    # and clean meshes remain unchanged because they normally arrive as a single
    # coherent component.
    selected, seed_guard_diag = _choose_seed_local_raw_edge_component(
        selected_edges=raw_selected,
        selected_edge_ids=tuple(selected_edge_ids),
        edge_table=edge_table,
        vertices=vertices,
        min_loop_edges=min_edges,
    )

    candidate_edges = _feature_rim_candidate_edges(faces=faces, face_normals=face_normals, edge_table=edge_table)
    candidate_edges.update(selected)

    # Region Select owns neutral rim completion.  The visible Ctrl-click chain can
    # be only a short arc, and crease-only candidate edges can be fragmented on
    # repaired/imported meshes.  For seed-owned polluted clouds, the final
    # circular closure pass is limited to a spatial neighborhood of the chosen
    # seed island instead of all runtime edges.  This prevents same-radius remote
    # features from becoming the selected opening frame.
    all_scan_edges = {_normalize_edge(edge) for edge in tuple(edge_table.get("unique_edges", ()) or ())}
    all_scan_edges.update(candidate_edges)
    selected_vertices = {int(v) for edge in selected for v in edge}
    selected_mid = _edge_cloud_midpoint(vertices, selected)
    all_scan_edges, scan_guard_diag = _seed_local_scan_edges(
        vertices=vertices,
        edges=all_scan_edges,
        seed_edges=selected,
        min_loop_edges=min_edges,
        enabled=bool(seed_guard_diag.get("used")),
    )

    components = tuple(_topology_connected_edge_components(candidate_edges))
    touching: list[set[EdgeKey]] = []
    nearby: list[set[EdgeKey]] = []
    median_selected_length = _median_edge_length(vertices, selected)
    search_distance = max(12.0 * max(median_selected_length, 1.0e-9), 1.0e-6)

    for comp in components:
        comp_set = {_normalize_edge(edge) for edge in comp}
        if not comp_set:
            continue
        comp_vertices = {int(v) for edge in comp_set for v in edge}
        if comp_vertices & selected_vertices or bool(comp_set & selected):
            touching.append(comp_set)
            continue
        # If topology does not connect through the exact picked vertices, still
        # consider nearby crease/boundary components.  This covers repaired or
        # imported meshes where the visible rim is split into many tiny arcs.  In
        # the polluted-cloud case, selected_mid is the seed-owned island midpoint,
        # not the midpoint of the whole 274-edge cloud.
        dist = _edge_cloud_distance_to_point(vertices, comp_set, selected_mid)
        if np.isfinite(dist) and dist <= search_distance:
            nearby.append(comp_set)

    candidate_components = touching or nearby
    if not candidate_components:
        ids = _edge_ids_from_keys(selected, edge_key_to_id) or tuple(selected_edge_ids)
        return set(selected), ids, {
            "used": False,
            "method": "selected_edge_to_complete_neutral_opening_rim",
            "reason": "no_touching_or_nearby_boundary_or_crease_component",
            "selected_edge_count": int(len(selected)),
            "inferred_edge_count": int(len(selected)),
            "raw_edge_cloud_quality": raw_quality,
            "candidate_feature_edge_count": int(len(candidate_edges)),
            "seed_local_rim_guard": {**dict(seed_guard_diag), "scan": dict(scan_guard_diag)},
            "not_feature_recognition": True,
        }

    max_reasonable_edges = max(32, int(min_edges) * 32)
    scored: list[tuple[float, int, set[EdgeKey], dict[str, object]]] = []
    for comp in candidate_components:
        completed, complete_diag = _complete_opening_rim_component_geometrically(
            component=comp,
            all_candidate_edges=all_scan_edges,
            selected_edges=selected,
            selected_vertices=selected_vertices,
            vertices=vertices,
            min_loop_edges=min_edges,
            max_reasonable_edges=max_reasonable_edges,
        )
        score, quality = _score_opening_rim_component(
            completed,
            selected_edges=selected,
            selected_vertices=selected_vertices,
            vertices=vertices,
            min_loop_edges=min_edges,
            max_reasonable_edges=max_reasonable_edges,
        )
        merged_quality = {**quality, "complete_circle_recovery": complete_diag}
        scored.append((float(score), int(len(completed)), set(completed), merged_quality))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen = set(scored[0][2])
    chosen_quality = dict(scored[0][3])

    cap_applied = False
    if len(chosen) > max_reasonable_edges:
        chosen = _local_edge_component_subset(chosen, selected, vertices, max_edges=max_reasonable_edges)
        cap_applied = True
        chosen_quality = {**chosen_quality, "post_cap_quality": _edge_cloud_graph_quality(chosen, vertices)}

    ids = _edge_ids_from_keys(chosen, edge_key_to_id) or tuple(selected_edge_ids)
    return chosen, ids, {
        "used": True,
        "method": "selected_edge_to_complete_neutral_opening_rim",
        "selected_edge_count": int(len(selected)),
        "candidate_feature_edge_count": int(len(candidate_edges)),
        "touching_component_count": int(len(touching)),
        "nearby_component_count": int(len(nearby)),
        "inferred_edge_count": int(len(chosen)),
        "cap_applied": bool(cap_applied),
        "single_pick_growth_used": True,
        "complete_circle_recovery_used": True,
        "raw_edge_cloud_quality": raw_quality,
        "chosen_edge_cloud_quality": chosen_quality,
        "chosen_component_score": float(scored[0][0]),
        "seed_local_rim_guard": {**dict(seed_guard_diag), "scan": dict(scan_guard_diag)},
        "normalized_seed_edge_overlap": int(len(chosen & selected)),
        "normalized_raw_edge_overlap": int(len(chosen & raw_selected)),
        "not_feature_recognition": True,
    }



def _edge_ids_from_keys(edges: Iterable[EdgeKey], edge_key_to_id: Mapping[object, object]) -> tuple[int, ...]:
    """Return stable mesh edge IDs for normalized edge keys."""

    ids: list[int] = []
    for edge in sorted({_normalize_edge(edge) for edge in tuple(edges or ())}):
        try:
            idx = int(edge_key_to_id.get(edge, -1))  # type: ignore[attr-defined]
        except Exception:
            idx = -1
        if idx >= 0:
            ids.append(idx)
    return tuple(sorted(set(ids)))



def _edge_cloud_midpoint(vertices: np.ndarray, edges: Iterable[EdgeKey]) -> np.ndarray:
    mids: list[np.ndarray] = []
    verts = np.asarray(vertices, dtype=float)
    for a, b in tuple(edges or ()):  # type: ignore[misc]
        ia, ib = int(a), int(b)
        if 0 <= ia < len(verts) and 0 <= ib < len(verts):
            mids.append(0.5 * (verts[ia, :3] + verts[ib, :3]))
    if not mids:
        return np.zeros(3, dtype=float)
    return np.asarray(mids, dtype=float).mean(axis=0)



def _edge_cloud_distance_to_point(vertices: np.ndarray, edges: Iterable[EdgeKey], point: np.ndarray) -> float:
    p = np.asarray(point, dtype=float).reshape(3)
    best = float("inf")
    verts = np.asarray(vertices, dtype=float)
    for a, b in tuple(edges or ()):  # type: ignore[misc]
        ia, ib = int(a), int(b)
        if 0 <= ia < len(verts) and 0 <= ib < len(verts):
            mid = 0.5 * (verts[ia, :3] + verts[ib, :3])
            best = min(best, float(np.linalg.norm(mid - p)))
    return float(best)



def _edge_cloud_graph_quality(edges: Iterable[EdgeKey], vertices: np.ndarray) -> dict[str, object]:
    """Neutral graph quality for an edge cloud.

    The result describes whether the cloud is a coherent loop/near-loop.  It is
    deliberately feature-agnostic: no BORE/CHAMFER classification is performed.
    """

    edge_set = {_normalize_edge(edge) for edge in tuple(edges or ())}
    components = tuple(_topology_connected_edge_components(edge_set))
    degree: dict[int, int] = defaultdict(int)
    lengths: list[float] = []
    verts = np.asarray(vertices, dtype=float)
    for a, b in edge_set:
        degree[int(a)] += 1
        degree[int(b)] += 1
        if 0 <= int(a) < len(verts) and 0 <= int(b) < len(verts):
            length = float(np.linalg.norm(verts[int(a), :3] - verts[int(b), :3]))
            if np.isfinite(length) and length > 1.0e-12:
                lengths.append(length)
    endpoints = tuple(sorted(int(v) for v, d in degree.items() if int(d) == 1))
    branch_vertices = tuple(sorted(int(v) for v, d in degree.items() if int(d) > 2))
    median_length = float(np.median(np.asarray(lengths, dtype=float))) if lengths else 0.0
    endpoint_gap = 0.0
    if len(endpoints) == 2:
        a, b = endpoints
        if 0 <= int(a) < len(verts) and 0 <= int(b) < len(verts):
            endpoint_gap = float(np.linalg.norm(verts[int(a), :3] - verts[int(b), :3]))
    closed = bool(edge_set) and len(components) == 1 and all(int(d) == 2 for d in degree.values())
    near_closed = bool(
        edge_set
        and len(components) == 1
        and len(branch_vertices) == 0
        and len(endpoints) in (0, 2)
        and (
            len(endpoints) == 0
            or (median_length > 0.0 and endpoint_gap <= max(3.5 * median_length, 1.0e-9))
        )
    )
    component_edge_counts = tuple(int(len(comp)) for comp in components)
    largest_component_edges = int(max(component_edge_counts) if component_edge_counts else 0)
    return {
        "edge_count": int(len(edge_set)),
        "vertex_count": int(len(degree)),
        "component_count": int(len(components)),
        "component_edge_counts": component_edge_counts,
        "largest_component_edges": int(largest_component_edges),
        "largest_component_fraction": float(largest_component_edges) / max(float(len(edge_set)), 1.0),
        "closed": bool(closed),
        "near_closed": bool(near_closed),
        "open_endpoint_count": int(len(endpoints)),
        "branch_vertex_count": int(len(branch_vertices)),
        "median_edge_length": float(median_length),
        "endpoint_gap": float(endpoint_gap),
    }



def _edge_cloud_is_coherent_opening_rim(edges: Iterable[EdgeKey], *, vertices: np.ndarray, min_loop_edges: int) -> tuple[bool, dict[str, object]]:
    """Return whether raw selected edges are already good rim evidence."""

    quality = _edge_cloud_graph_quality(edges, vertices)
    edge_count = int(quality.get("edge_count", 0))
    ok = bool(
        edge_count >= max(3, int(min_loop_edges))
        and int(quality.get("component_count", 0)) == 1
        and int(quality.get("branch_vertex_count", 0)) == 0
        and bool(quality.get("near_closed", False))
    )
    quality = {**quality, "accepted_as_raw_opening_rim": bool(ok)}
    return bool(ok), quality




def _choose_seed_local_raw_edge_component(
    *,
    selected_edges: set[EdgeKey],
    selected_edge_ids: tuple[int, ...],
    edge_table: Mapping[str, object],
    vertices: np.ndarray,
    min_loop_edges: int,
) -> tuple[set[EdgeKey], dict[str, object]]:
    """Choose one raw selected component as seed-owned opening evidence.

    This is a conservative guard for contaminated live Ctrl-click selections.
    It does not classify a feature.  It only prevents a broad raw edge cloud from
    acting as one giant seed during rim normalization.
    """

    selected = {_normalize_edge(edge) for edge in tuple(selected_edges or ())}
    components = tuple(_topology_connected_edge_components(selected))
    if len(components) <= 1:
        return set(selected), {
            "used": False,
            "reason": "raw_selection_single_component",
            "raw_component_count": int(len(components)),
            "seed_component_edge_count": int(len(selected)),
        }

    unique_edges = tuple(edge_table.get("unique_edges", ()) or ())
    first_edge_key: EdgeKey | None = None
    if selected_edge_ids:
        try:
            first_idx = int(tuple(selected_edge_ids)[0])
            if 0 <= first_idx < len(unique_edges):
                first_edge_key = _normalize_edge(unique_edges[first_idx])
        except Exception:
            first_edge_key = None

    scored: list[tuple[float, set[EdgeKey], dict[str, object]]] = []
    min_edges = max(3, int(min_loop_edges))
    for idx, comp in enumerate(components):
        comp_set = {_normalize_edge(edge) for edge in tuple(comp or ())}
        if not comp_set:
            continue
        quality = _edge_cloud_graph_quality(comp_set, vertices)
        edge_count = int(quality.get("edge_count", 0) or 0)
        branch_count = int(quality.get("branch_vertex_count", 0) or 0)
        endpoint_count = int(quality.get("open_endpoint_count", 0) or 0)
        touches_first = bool(first_edge_key is not None and first_edge_key in comp_set)
        frame = _fit_opening_rim_frame(vertices=vertices, edges=comp_set)
        frame_score = 0.0
        if frame is not None:
            frame_score += max(0.0, 1.0 - min(float(frame.radius_rel_rms) / 0.35, 1.0)) * 750.0
            frame_score += max(0.0, 1.0 - min(float(frame.plane_rel_rms) / 0.30, 1.0)) * 300.0
        score = 0.0
        if bool(quality.get("closed", False)):
            score += 6500.0
        elif bool(quality.get("near_closed", False)):
            score += 3800.0
        if edge_count >= min_edges:
            score += 700.0
        score += min(edge_count, min_edges * 4) * 35.0
        score += frame_score
        score += 1200.0 if touches_first else 0.0
        score -= branch_count * 600.0
        score -= max(0, endpoint_count - 2) * 120.0
        scored.append((float(score), set(comp_set), {
            "index": int(idx),
            "score": float(score),
            "edge_count": int(edge_count),
            "touches_first_selected_edge": bool(touches_first),
            "closed": bool(quality.get("closed", False)),
            "near_closed": bool(quality.get("near_closed", False)),
            "branch_vertex_count": int(branch_count),
            "open_endpoint_count": int(endpoint_count),
            "frame_fit_available": bool(frame is not None),
            "frame_radius_rel_rms": float(getattr(frame, "radius_rel_rms", 0.0) if frame is not None else 0.0),
            "frame_plane_rel_rms": float(getattr(frame, "plane_rel_rms", 0.0) if frame is not None else 0.0),
        }))

    if not scored:
        return set(selected), {
            "used": False,
            "reason": "no_scoreable_raw_components",
            "raw_component_count": int(len(components)),
            "seed_component_edge_count": int(len(selected)),
        }

    scored.sort(key=lambda item: item[0], reverse=True)
    chosen_score, chosen, chosen_diag = scored[0]
    return set(chosen), {
        "used": True,
        "reason": "contaminated_multi_component_raw_selection_seed_island_chosen",
        "raw_selected_edge_count": int(len(selected)),
        "raw_component_count": int(len(components)),
        "seed_component_edge_count": int(len(chosen)),
        "seed_component_fraction": float(len(chosen)) / max(float(len(selected)), 1.0),
        "selected_component_score": float(chosen_score),
        "selected_component": dict(chosen_diag),
        "candidate_component_summaries": tuple(item[2] for item in scored[:8]),
        "not_feature_recognition": True,
    }


def _seed_local_scan_edges(
    *,
    vertices: np.ndarray,
    edges: Iterable[EdgeKey],
    seed_edges: Iterable[EdgeKey],
    min_loop_edges: int,
    enabled: bool,
) -> tuple[set[EdgeKey], dict[str, object]]:
    """Restrict same-circle scanning to a seed-owned spatial neighborhood."""

    edge_set = {_normalize_edge(edge) for edge in tuple(edges or ())}
    seed_set = {_normalize_edge(edge) for edge in tuple(seed_edges or ())}
    if not enabled or not edge_set or not seed_set:
        return set(edge_set), {
            "used": False,
            "reason": "disabled_or_empty",
            "input_edge_count": int(len(edge_set)),
            "output_edge_count": int(len(edge_set)),
        }

    verts = np.asarray(vertices, dtype=float)
    seed_mid = _edge_cloud_midpoint(verts, seed_set)
    seed_len = _median_edge_length(verts, seed_set)
    frame = _fit_opening_rim_frame(vertices=verts, edges=seed_set)
    if frame is not None and float(frame.radius) > 0.0:
        max_distance = max(float(frame.radius) * 2.75, float(seed_len) * 36.0, 1.0e-6)
        reason = "seed_frame_radius_neighborhood"
    else:
        max_distance = max(float(seed_len) * 48.0, 1.0e-6)
        reason = "seed_edge_length_neighborhood"

    seed_vertices = {int(v) for edge in seed_set for v in edge}
    out: set[EdgeKey] = set(seed_set)
    for edge in edge_set:
        if edge in seed_set:
            continue
        a, b = edge
        if int(a) in seed_vertices or int(b) in seed_vertices:
            out.add(edge)
            continue
        if _edge_cloud_distance_to_point(verts, (edge,), seed_mid) <= max_distance:
            out.add(edge)

    # Never make the scan so small that a single manually selected partial rim
    # cannot be completed; fall back to the original set if the neighborhood is
    # unusably tiny.
    if len(out) < max(3, int(min_loop_edges)):
        return set(edge_set), {
            "used": False,
            "reason": "seed_local_scan_too_small_fallback_global",
            "input_edge_count": int(len(edge_set)),
            "output_edge_count": int(len(edge_set)),
            "attempted_output_edge_count": int(len(out)),
            "max_distance": float(max_distance),
        }

    return set(out), {
        "used": True,
        "reason": reason,
        "input_edge_count": int(len(edge_set)),
        "output_edge_count": int(len(out)),
        "seed_edge_count": int(len(seed_set)),
        "max_distance": float(max_distance),
        "seed_midpoint": _to_vector3(seed_mid),
    }


def _score_opening_rim_component(
    edges: Iterable[EdgeKey],
    *,
    selected_edges: set[EdgeKey],
    selected_vertices: set[int],
    vertices: np.ndarray,
    min_loop_edges: int,
    max_reasonable_edges: int,
) -> tuple[float, dict[str, object]]:
    """Score a candidate edge component as normalized rim evidence.

    Selection overlap keeps the chosen component anchored to the user's pick;
    loop quality keeps us from choosing an arbitrary large crease network.
    """

    edge_set = {_normalize_edge(edge) for edge in tuple(edges or ())}
    quality = _edge_cloud_graph_quality(edge_set, vertices)
    comp_vertices = {int(v) for edge in edge_set for v in edge}
    edge_overlap = len(edge_set & selected_edges)
    vertex_overlap = len(comp_vertices & selected_vertices)
    edge_count = int(quality.get("edge_count", 0))

    score = 0.0
    if bool(quality.get("closed", False)):
        score += 10000.0
    elif bool(quality.get("near_closed", False)):
        score += 6500.0
    if edge_count >= max(3, int(min_loop_edges)):
        score += 1500.0
    score += float(min(edge_count, max_reasonable_edges))
    score += float(edge_overlap) * 180.0
    score += float(vertex_overlap) * 32.0
    score -= float(quality.get("branch_vertex_count", 0)) * 700.0
    score -= float(max(0, int(quality.get("open_endpoint_count", 0)) - 2)) * 250.0
    if edge_count > max_reasonable_edges:
        score -= float(edge_count - max_reasonable_edges) * 4.0

    return float(score), {
        **quality,
        "selected_edge_overlap": int(edge_overlap),
        "selected_vertex_overlap": int(vertex_overlap),
        "score": float(score),
    }



def _complete_opening_rim_component_geometrically(
    *,
    component: Iterable[EdgeKey],
    all_candidate_edges: Iterable[EdgeKey],
    selected_edges: set[EdgeKey],
    selected_vertices: set[int],
    vertices: np.ndarray,
    min_loop_edges: int,
    max_reasonable_edges: int,
) -> tuple[set[EdgeKey], dict[str, object]]:
    """Complete one local boundary/crease component into a same-circle rim.

    This is Region Select's neutral rim closure transform.  It deliberately does
    not ask what feature the rim belongs to.  A raw Ctrl-click can be a small arc
    on an inner chamfer seam or a plain bore opening.  The meaning transform here
    is only:

        raw edge evidence -> complete opening/rim loop evidence

    The older implementation selected the best connected component of the
    same-circle edge band.  That made the live selection stay as a partial arc on
    repaired triangulations where the real circular rim is split into several
    edge components.  The fixed implementation first tries to recover the full
    angular 360-degree same-circle band, then falls back to component scoring only
    when the angular evidence is genuinely partial.
    """

    comp = {_normalize_edge(edge) for edge in tuple(component or ())}
    if not comp:
        return set(selected_edges), {"used": False, "reason": "empty_component"}

    # Prefer a frame from the touching/nearby component.  If that component is too
    # small, selected edges still provide the fallback seed.
    frame_seed = set(comp)
    if len(frame_seed) < max(3, int(min_loop_edges)):
        frame_seed.update(selected_edges)
    frame = _fit_opening_rim_frame(vertices=vertices, edges=frame_seed)
    if frame is None:
        return set(comp), {
            "used": False,
            "reason": "no_opening_frame_from_component",
            "component_edge_count": int(len(comp)),
        }

    same_circle = _same_circle_candidate_edges(
        vertices=vertices,
        edges=all_candidate_edges,
        frame=frame,
        selected_edges=selected_edges,
        max_reasonable_edges=max_reasonable_edges,
    )
    same_circle.update(selected_edges)
    if len(same_circle) < max(3, int(min_loop_edges)):
        return set(comp), {
            "used": False,
            "reason": "same_circle_band_too_small",
            "component_edge_count": int(len(comp)),
            "same_circle_edge_count": int(len(same_circle)),
            "frame": _opening_rim_frame_diagnostics(frame),
        }

    # First authority: angular closure around the fitted neutral circle.  This
    # can return several disconnected edge components, but that is acceptable at
    # this semantic stage: Region Select is normalizing rim evidence, not deriving
    # a topological delete patch.  The returned edge IDs are exactly the full rim
    # evidence the user indicated.
    angular_edges, angular_diag = _angularly_closed_same_circle_rim(
        vertices=vertices,
        edges=same_circle,
        frame=frame,
        selected_edges=selected_edges,
        min_loop_edges=int(min_loop_edges),
        max_reasonable_edges=int(max_reasonable_edges),
    )
    angular_coverage = float(angular_diag.get("angular_coverage", 0.0))
    angular_component_count = int(angular_diag.get("component_count", 0))
    angular_edge_count = int(len(angular_edges))
    if angular_edge_count >= max(3, int(min_loop_edges)) and angular_coverage >= 0.78:
        return set(angular_edges), {
            "used": True,
            "method": "same_circle_angular_closed_rim_from_region_select_frame",
            "component_edge_count": int(len(comp)),
            "same_circle_edge_count": int(len(same_circle)),
            "chosen_edge_count": int(len(angular_edges)),
            "component_count": int(angular_component_count),
            "angular_completion_used": True,
            "angular_completion": dict(angular_diag),
            "frame": _opening_rim_frame_diagnostics(frame),
            "not_feature_recognition": True,
        }

    # Fallback: choose the best connected component anchored to the selected
    # evidence.  This preserves conservative behavior when the mesh really only
    # provides a partial rim.
    selected_mid = _edge_cloud_midpoint(vertices, selected_edges)
    components = tuple(_topology_connected_edge_components(same_circle))
    best_component: set[EdgeKey] = set()
    best_score = -1.0e18
    best_quality: dict[str, object] = {}
    for raw_comp in components:
        comp_set = {_normalize_edge(edge) for edge in raw_comp}
        comp_vertices = {int(v) for edge in comp_set for v in edge}
        edge_overlap = len(comp_set & selected_edges)
        vertex_overlap = len(comp_vertices & selected_vertices)
        distance = _edge_cloud_distance_to_point(vertices, comp_set, selected_mid)
        quality = _edge_cloud_graph_quality(comp_set, vertices)
        comp_angular = _angular_coverage_diagnostics(vertices=vertices, edges=comp_set, frame=frame)
        score = 0.0
        if bool(quality.get("closed", False)):
            score += 10000.0
        elif bool(quality.get("near_closed", False)):
            score += 6000.0
        score += 3500.0 * float(comp_angular.get("angular_coverage", 0.0))
        score += min(len(comp_set), max_reasonable_edges)
        score += 900.0 * edge_overlap
        score += 120.0 * vertex_overlap
        score -= 2.0 * min(distance / max(frame.median_edge_length, 1.0e-9), 1000.0)
        score -= 400.0 * int(quality.get("branch_vertex_count", 0))
        if score > best_score:
            best_score = float(score)
            best_component = set(comp_set)
            best_quality = {**quality, "angular": comp_angular}

    if not best_component:
        best_component = set(same_circle)

    pruned = _prune_non_loop_tails(best_component)
    if len(pruned) >= max(3, int(min_loop_edges)) and _component_is_still_anchored(pruned, selected_edges, selected_vertices):
        chosen = pruned
        pruned_used = True
    else:
        chosen = best_component
        pruned_used = False

    return chosen, {
        "used": True,
        "method": "same_circle_edge_band_from_region_select_frame",
        "component_edge_count": int(len(comp)),
        "same_circle_edge_count": int(len(same_circle)),
        "chosen_edge_count": int(len(chosen)),
        "component_count": int(len(components)),
        "best_component_score": float(best_score),
        "best_component_quality": best_quality,
        "tail_pruning_used": bool(pruned_used),
        "angular_completion_used": False,
        "angular_completion": dict(angular_diag),
        "frame": _opening_rim_frame_diagnostics(frame),
        "not_feature_recognition": True,
    }




def _angularly_closed_same_circle_rim(
    *,
    vertices: np.ndarray,
    edges: Iterable[EdgeKey],
    frame: _OpeningRimFrame,
    selected_edges: set[EdgeKey],
    min_loop_edges: int,
    max_reasonable_edges: int,
) -> tuple[set[EdgeKey], dict[str, object]]:
    """Return the full angular same-circle rim if the edge band covers the circle.

    This is the final neutral closure pass.  The topological chain may be broken,
    but the circle is still present in the mesh as a set of same-plane,
    same-radius, tangential edges.  We use angular coverage rather than connected
    component closure so inner rims and plain bore openings are not reduced to a
    visible partial arc.
    """

    edge_set = {_normalize_edge(edge) for edge in tuple(edges or ())}
    if len(edge_set) < max(3, int(min_loop_edges)):
        return set(edge_set), {"used": False, "reason": "too_few_edges", "edge_count": int(len(edge_set))}

    scored: list[tuple[float, float, EdgeKey]] = []
    for edge in edge_set:
        score, angle = _same_circle_edge_score_and_angle(vertices=vertices, edge=edge, frame=frame)
        if not np.isfinite(score) or not np.isfinite(angle):
            continue
        # Selected edges are raw evidence; keep them even if the local tangent is
        # slightly weak.  Unselected edges must look like real rim edges.
        if edge not in selected_edges and score < 0.48:
            continue
        scored.append((float(score), float(angle), edge))

    if len(scored) < max(3, int(min_loop_edges)):
        return set(edge_set), {
            "used": False,
            "reason": "too_few_scored_same_circle_edges",
            "edge_count": int(len(edge_set)),
            "scored_edge_count": int(len(scored)),
        }

    # Collapse duplicate/competing candidates per angular cell only when the band
    # is very dense.  For normal tessellated rims this keeps nearly all actual rim
    # edges.  For fan/chord clutter it keeps the best tangential edge per sector.
    bin_count = _angular_bin_count(frame=frame, edge_count=len(scored))
    bins: dict[int, tuple[float, float, EdgeKey]] = {}
    for score, angle, edge in scored:
        idx = int(np.floor((angle % (2.0 * math.pi)) / (2.0 * math.pi) * float(bin_count))) % int(bin_count)
        current = bins.get(idx)
        # Prefer selected evidence and then the highest geometric score.
        boosted_score = float(score) + (0.20 if edge in selected_edges else 0.0)
        if current is None or boosted_score > float(current[0]) + (0.20 if current[2] in selected_edges else 0.0):
            bins[idx] = (float(score), float(angle), edge)

    chosen = {edge for _score, _angle, edge in bins.values()}
    # If binning became too lossy on a coarse rim, use all scored edges.  This is
    # common on simple OBJ test cylinders where the real rim has one edge per
    # angular step.
    if len(chosen) < max(3, int(min_loop_edges)) and len(scored) <= int(max_reasonable_edges):
        chosen = {edge for _score, _angle, edge in scored}

    if len(chosen) > int(max_reasonable_edges):
        ordered = sorted(
            ((_same_circle_edge_score_and_angle(vertices=vertices, edge=edge, frame=frame)[0], edge) for edge in chosen),
            key=lambda item: item[0],
            reverse=True,
        )
        chosen = {edge for _score, edge in ordered[: int(max_reasonable_edges)]}

    coverage = _angular_coverage_diagnostics(vertices=vertices, edges=chosen, frame=frame)
    components = tuple(_topology_connected_edge_components(chosen))
    return chosen, {
        "used": True,
        "edge_count": int(len(chosen)),
        "input_edge_count": int(len(edge_set)),
        "scored_edge_count": int(len(scored)),
        "bin_count": int(bin_count),
        "occupied_bin_count": int(len(bins)),
        "component_count": int(len(components)),
        **coverage,
    }



def _same_circle_edge_score_and_angle(*, vertices: np.ndarray, edge: EdgeKey, frame: _OpeningRimFrame) -> tuple[float, float]:
    edge = _normalize_edge(edge)
    a, b = edge
    if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
        return 0.0, float("nan")
    p0 = vertices[int(a), :3]
    p1 = vertices[int(b), :3]
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if not np.isfinite(length) or length <= 1.0e-12:
        return 0.0, float("nan")
    mid = 0.5 * (p0 + p1)
    rel = mid - frame.center.reshape(3)
    axial = float(np.dot(rel, frame.axis.reshape(3)))
    radial_vec = rel - axial * frame.axis.reshape(3)
    radial = float(np.linalg.norm(radial_vec))
    if not np.isfinite(radial) or radial <= 1.0e-12:
        return 0.0, float("nan")
    plane_error = abs(axial)
    radius_error = abs(radial - float(frame.radius))
    plane_score = 1.0 - min(plane_error / max(frame.median_edge_length * 3.5, frame.radius * 0.075, 1.0e-9), 1.0)
    radius_score = 1.0 - min(radius_error / max(frame.median_edge_length * 4.5, frame.radius * 0.090, 1.0e-9), 1.0)
    tangent = _edge_tangent_alignment_to_frame(vertices, edge, frame)
    length_score = 1.0 - min(max(length - frame.median_edge_length * 4.0, 0.0) / max(frame.radius, 1.0e-9), 1.0)
    score = 0.34 * tangent + 0.26 * plane_score + 0.30 * radius_score + 0.10 * length_score
    u_axis, v_axis = _opening_frame_plane_basis(frame.axis)
    angle = float(math.atan2(float(np.dot(radial_vec, v_axis)), float(np.dot(radial_vec, u_axis))))
    if angle < 0.0:
        angle += 2.0 * math.pi
    return float(score), float(angle)



def _angular_bin_count(*, frame: _OpeningRimFrame, edge_count: int) -> int:
    circumference_bins = int(round((2.0 * math.pi * max(float(frame.radius), 1.0e-9)) / max(float(frame.median_edge_length), 1.0e-9)))
    # Do not over-bin tiny/coarse rings; over-binning makes a complete ring look
    # artificially incomplete.  Keep enough bins for the rim shape, but bound it.
    value = max(24, min(384, max(int(edge_count), int(circumference_bins))))
    return int(value)



def _angular_coverage_diagnostics(*, vertices: np.ndarray, edges: Iterable[EdgeKey], frame: _OpeningRimFrame) -> dict[str, object]:
    angles: list[float] = []
    for edge in {_normalize_edge(e) for e in tuple(edges or ())}:
        _score, angle = _same_circle_edge_score_and_angle(vertices=vertices, edge=edge, frame=frame)
        if np.isfinite(angle):
            angles.append(float(angle) % (2.0 * math.pi))
    if len(angles) < 2:
        return {
            "angular_coverage": 0.0,
            "largest_gap_degrees": 360.0,
            "angle_sample_count": int(len(angles)),
        }
    arr = np.asarray(sorted(angles), dtype=float)
    gaps = np.diff(np.concatenate((arr, arr[:1] + 2.0 * math.pi)))
    largest_gap = float(np.max(gaps)) if gaps.size else 2.0 * math.pi
    coverage = 1.0 - min(max(largest_gap / (2.0 * math.pi), 0.0), 1.0)
    return {
        "angular_coverage": float(coverage),
        "largest_gap_degrees": float(math.degrees(largest_gap)),
        "angle_sample_count": int(len(angles)),
    }



def _opening_frame_plane_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _unit_vector(axis, fallback=(0.0, 0.0, 1.0))
    reference = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(reference, normal))) > 0.90:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    u_axis = _unit_vector(np.cross(normal, reference), fallback=(0.0, 1.0, 0.0))
    v_axis = _unit_vector(np.cross(normal, u_axis), fallback=(1.0, 0.0, 0.0))
    return u_axis, v_axis




def _fit_opening_rim_frame(*, vertices: np.ndarray, edges: Iterable[EdgeKey]) -> _OpeningRimFrame | None:
    """Fit a neutral circular rim frame from edge vertices."""

    edge_set = {_normalize_edge(edge) for edge in tuple(edges or ())}
    vertex_ids = tuple(sorted({int(v) for edge in edge_set for v in edge if 0 <= int(v) < len(vertices)}))
    if len(vertex_ids) < 4:
        return None

    pts = np.asarray(vertices, dtype=float)[np.asarray(vertex_ids, dtype=np.int64), :3]
    center0 = pts.mean(axis=0)
    centered = pts - center0.reshape(1, 3)
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        axis = _unit_vector(vh[-1], fallback=(0.0, 0.0, 1.0))
    except Exception:
        return None

    reference = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(reference, axis))) > 0.90:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    u_axis = _unit_vector(np.cross(axis, reference), fallback=(0.0, 1.0, 0.0))
    v_axis = _unit_vector(np.cross(axis, u_axis), fallback=(1.0, 0.0, 0.0))

    xy = np.column_stack((centered @ u_axis, centered @ v_axis))
    try:
        mat = np.column_stack((xy[:, 0], xy[:, 1], np.ones(len(xy))))
        rhs = -(xy[:, 0] * xy[:, 0] + xy[:, 1] * xy[:, 1])
        sol, *_ = np.linalg.lstsq(mat, rhs, rcond=None)
        cx = -0.5 * float(sol[0])
        cy = -0.5 * float(sol[1])
        radius_sq = max(0.0, cx * cx + cy * cy - float(sol[2]))
        radius = float(np.sqrt(radius_sq))
        center = center0 + cx * u_axis + cy * v_axis
    except Exception:
        rel = centered - np.outer(centered @ axis, axis)
        radii = np.linalg.norm(rel, axis=1)
        valid = radii[np.isfinite(radii) & (radii > 1.0e-12)]
        if valid.size < 3:
            return None
        radius = float(np.median(valid))
        center = center0

    median_len = _median_edge_length(vertices, edge_set)
    if median_len <= 0.0:
        median_len = 1.0
    if not np.isfinite(radius) or radius <= max(1.0e-9, 1.25 * median_len):
        return None

    rel = pts - center.reshape(1, 3)
    plane_dist = rel @ axis.reshape(3)
    radial_vec = rel - np.outer(plane_dist, axis)
    radii = np.linalg.norm(radial_vec, axis=1)
    plane_rms = float(np.sqrt(np.mean(plane_dist * plane_dist))) if plane_dist.size else 0.0
    radius_rms = float(np.sqrt(np.mean((radii - radius) * (radii - radius)))) if radii.size else 0.0
    plane_rel_rms = plane_rms / max(float(radius), 1.0e-9)
    radius_rel_rms = radius_rms / max(float(radius), 1.0e-9)

    # Reject frames clearly fit to a broad surface/cloud rather than one rim.
    if plane_rel_rms > 0.35 or radius_rel_rms > 0.35:
        return None

    return _OpeningRimFrame(
        center=np.asarray(center, dtype=float).reshape(3),
        axis=np.asarray(canonical_axis(axis), dtype=float).reshape(3),
        radius=float(radius),
        median_edge_length=float(median_len),
        fit_edge_count=int(len(edge_set)),
        fit_vertex_count=int(len(vertex_ids)),
        plane_rel_rms=float(plane_rel_rms),
        radius_rel_rms=float(radius_rel_rms),
    )



def _same_circle_candidate_edges(
    *,
    vertices: np.ndarray,
    edges: Iterable[EdgeKey],
    frame: _OpeningRimFrame,
    selected_edges: set[EdgeKey],
    max_reasonable_edges: int,
) -> set[EdgeKey]:
    """Return candidate edges lying on the same neutral circle as ``frame``."""

    edge_set = {_normalize_edge(edge) for edge in tuple(edges or ())}
    max_plane_error = max(frame.median_edge_length * 3.5, frame.radius * 0.075)
    max_radius_error = max(frame.median_edge_length * 4.5, frame.radius * 0.090)
    max_edge_length = max(frame.median_edge_length * 8.0, frame.radius * 0.35)
    out: set[EdgeKey] = set()
    for edge in edge_set:
        a, b = edge
        if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
            continue
        length = float(np.linalg.norm(vertices[int(a), :3] - vertices[int(b), :3]))
        if not np.isfinite(length) or length <= 1.0e-12 or length > max_edge_length:
            continue
        mid = 0.5 * (vertices[int(a), :3] + vertices[int(b), :3])
        rel = mid - frame.center.reshape(3)
        plane_error = abs(float(np.dot(rel, frame.axis)))
        radial_vec = rel - float(np.dot(rel, frame.axis)) * frame.axis.reshape(3)
        radial = float(np.linalg.norm(radial_vec))
        radius_error = abs(radial - frame.radius)
        if edge not in selected_edges:
            if plane_error > max_plane_error or radius_error > max_radius_error:
                continue
            tangent = _edge_tangent_alignment_to_frame(vertices, edge, frame)
            if tangent < 0.30:
                continue
        out.add(edge)
        if len(out) > max_reasonable_edges * 4:
            # This is no longer one local rim; the caller will score/cap later,
            # but do not allow a model-wide crease network to dominate runtime.
            break
    return out



def _edge_tangent_alignment_to_frame(vertices: np.ndarray, edge: EdgeKey, frame: _OpeningRimFrame) -> float:
    a, b = _normalize_edge(edge)
    if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
        return 0.0
    vec = vertices[int(b), :3] - vertices[int(a), :3]
    length = float(np.linalg.norm(vec))
    if length <= 1.0e-12:
        return 0.0
    edge_dir = vec / length
    mid = 0.5 * (vertices[int(a), :3] + vertices[int(b), :3])
    rel = mid - frame.center.reshape(3)
    radial = rel - float(np.dot(rel, frame.axis)) * frame.axis.reshape(3)
    radial_len = float(np.linalg.norm(radial))
    if radial_len <= 1.0e-12:
        return 0.0
    tangent = np.cross(frame.axis.reshape(3), radial / radial_len)
    tangent_len = float(np.linalg.norm(tangent))
    if tangent_len <= 1.0e-12:
        return 0.0
    tangent = tangent / tangent_len
    return float(abs(np.dot(edge_dir, tangent)))



def _prune_non_loop_tails(edges: Iterable[EdgeKey]) -> set[EdgeKey]:
    """Remove leaf tails from an edge set while preserving closed/near-closed loops."""

    remaining = {_normalize_edge(edge) for edge in tuple(edges or ())}
    changed = True
    while changed and remaining:
        changed = False
        degree: dict[int, int] = defaultdict(int)
        for a, b in remaining:
            degree[int(a)] += 1
            degree[int(b)] += 1
        leaves = {int(v) for v, deg in degree.items() if int(deg) <= 1}
        if not leaves:
            break
        drop = {edge for edge in remaining if int(edge[0]) in leaves or int(edge[1]) in leaves}
        if drop:
            remaining.difference_update(drop)
            changed = True
    return remaining



def _component_is_still_anchored(edges: Iterable[EdgeKey], selected_edges: set[EdgeKey], selected_vertices: set[int]) -> bool:
    edge_set = {_normalize_edge(edge) for edge in tuple(edges or ())}
    if edge_set & selected_edges:
        return True
    verts = {int(v) for edge in edge_set for v in edge}
    return bool(verts & selected_vertices)



def _opening_rim_frame_diagnostics(frame: _OpeningRimFrame) -> dict[str, object]:
    return {
        "center": _to_vector3(frame.center),
        "axis": _to_vector3(frame.axis),
        "radius": float(frame.radius),
        "median_edge_length": float(frame.median_edge_length),
        "fit_edge_count": int(frame.fit_edge_count),
        "fit_vertex_count": int(frame.fit_vertex_count),
        "plane_rel_rms": float(frame.plane_rel_rms),
        "radius_rel_rms": float(frame.radius_rel_rms),
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
