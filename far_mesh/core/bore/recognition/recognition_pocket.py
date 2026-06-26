"""POCKET recognition helpers for FAR MESH BoreTool.

This module owns POCKET-local recognition helper logic that was first split out
of ``recognition_component_engine.py`` in v174l.  It remains a recognition layer:
it may transform measured POCKET evidence into floor/side-wall ownership fields
and CandidateData contract dictionaries, but it does not build
DeletePatchProposal objects, generate replacement topology, mutate meshes, or
perform host/viewport work.

Semantic boundary
-----------------
RegionData / measured local evidence -> POCKET floor and side-wall role evidence
-> CandidateData contract fields.  Child BORE openings inside a POCKET floor are
relationship metadata/protected boundaries, not a new mixed feature family and
not BORE ownership transfer.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import math
import numpy as np

from ..geometry import canonical_axis, to_vector3
from ..topology import boundary_edges_for_face_patch, connected_face_components, edge_loop_components
from ..heuristics import POCKET_HEURISTIC_RECIPE, compact_heuristic_summary, make_heuristic_result
from ..types import (
    EvidenceKind,
    FeatureFamily,
    FeaturePrimitiveKind,
    FeatureRelationshipKind,
    RecognitionStage,
    tuple_ints,
)

POCKET_RECOGNITION_SPLIT_CHECKPOINT_V174L = (
    "v174l_first_pocket_module_split_child_bore_filter_floor_boundary_and_candidate_contract_helpers_no_behavior_change"
)

POCKET_RECOGNITION_X1_PROBE_HELPER_SPLIT_CHECKPOINT_V174M = (
    "v174m_pocket_x1_probe_ledger_evidence_helpers_no_behavior_change"
)

POCKET_RECOGNITION_AUTHORITY_LEAK_DIAGNOSTIC_CHECKPOINT_V174N = (
    "v174n_pocket_family_authority_leak_diagnostics_no_behavior_change"
)


POCKET_RECOGNITION_DETECTOR_SPLIT_CHECKPOINT_V174O = (
    "v174o_pocket_detector_orchestration_split_no_behavior_change"
)




class _PocketFrameV174O:
    """Small POCKET-local frame equivalent to the old component-engine _Frame.

    v174o keeps this private and behavior-preserving so the moved POCKET detector
    does not import from recognition_component_engine.py.
    """

    def __init__(self, *, center: object, axis: object, radius: object) -> None:
        self.center = np.asarray(center, dtype=float).reshape(3)
        self.axis = canonical_axis(axis)
        self.radius = _safe_float(radius, 0.0)


def _pocket_unit_rows_v174o(values: np.ndarray) -> np.ndarray:
    """Return unit row vectors for POCKET-local detector geometry.

    This helper is copied from the component-engine-local numeric helper during
    the v174o module split so the moved detector can remain behavior-preserving
    and independent from recognition_component_engine.py.
    """

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((0, 3), dtype=float)
    arr = arr[:, :3]
    length = np.linalg.norm(arr, axis=1)
    out = np.zeros_like(arr)
    ok = np.isfinite(length) & (length > 1.0e-12)
    out[ok] = arr[ok] / length[ok].reshape(-1, 1)
    return out


def _pocket_percentile_v174o(values: np.ndarray, q: float, default: float = 0.0) -> float:
    """Behavior-preserving percentile helper for the moved POCKET detector."""

    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, q))


def _pocket_local_arrays_v174o(
    *,
    frame: object,
    face_ids: tuple[int, ...],
    face_centroids: np.ndarray,
    face_normals: np.ndarray,
) -> dict[str, np.ndarray]:
    """Behavior-preserving local cylindrical arrays for POCKET recognition.

    The input frame is the same component-engine frame object used before v174o;
    only the ownership of this helper moved to recognition_pocket.py.
    """

    ids = np.asarray(face_ids, dtype=np.int64)
    center = np.asarray(getattr(frame, "center"), dtype=float).reshape(3)
    axis = canonical_axis(getattr(frame, "axis"))
    pts = np.asarray(face_centroids, dtype=float)[ids, :3]
    normals = _pocket_unit_rows_v174o(np.asarray(face_normals, dtype=float))
    if len(normals) <= int(np.max(ids)) if len(ids) else False:
        normals = np.zeros_like(np.asarray(face_centroids, dtype=float)[:, :3])
    n = normals[ids, :3]
    rel = pts - center.reshape(1, 3)
    axial = rel @ axis.reshape(3)
    radial_vec = rel - axial.reshape(-1, 1) * axis.reshape(1, 3)
    radial = np.linalg.norm(radial_vec, axis=1)
    radial_dir = np.zeros_like(radial_vec)
    ok_radial = radial > 1.0e-12
    radial_dir[ok_radial] = radial_vec[ok_radial] / radial[ok_radial].reshape(-1, 1)
    normal_axis_abs = np.abs(n @ axis.reshape(3))
    radial_normal_alignment = np.abs(np.sum(n * radial_dir, axis=1))
    finite = (
        np.isfinite(axial)
        & np.isfinite(radial)
        & np.isfinite(normal_axis_abs)
        & np.isfinite(radial_normal_alignment)
    )
    return {
        "ids": ids,
        "points": pts,
        "axial": axial,
        "radial": radial,
        "normal_axis_abs": normal_axis_abs,
        "radial_normal_alignment": radial_normal_alignment,
        "finite": finite,
    }

def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    return float(out) if math.isfinite(out) else float(default)

def _normalize_edge_key(edge: object) -> tuple[int, int]:
    try:
        a, b = tuple(edge)[:2]  # type: ignore[arg-type]
        ia = int(a)
        ib = int(b)
    except Exception:
        return (-1, -1)
    if ia == ib:
        return (-1, -1)
    return (ia, ib) if ia < ib else (ib, ia)

def _face_patch_boundary_semantic_report(faces: np.ndarray, face_ids: Iterable[int]) -> dict[str, object]:
    """Return a small semantic topology report for a candidate owned face patch.

    This is recognition / ownership evidence only.  It does not mutate topology and
    it does not claim a rebuild will succeed.  The purpose is to keep CandidateData
    honest: an owned BORE wall that is supposed to mean "wall surface bounded by
    the selected opening and the opposite/end opening" must not silently contain
    extra unclassified boundary loops.
    """

    ids = tuple_ints(face_ids)
    if not ids:
        return {
            "face_count": 0,
            "boundary_edge_count": 0,
            "boundary_loop_count": 0,
            "boundary_loop_edge_counts": (),
            "boundary_loop_vertex_counts": (),
            "component_count": 0,
            "component_face_counts": (),
        }
    try:
        boundary = boundary_edges_for_face_patch(faces, ids)
        loops = edge_loop_components(boundary)
        comps = connected_face_components(faces, ids)
        loop_vertex_counts = []
        for loop in loops:
            verts = set()
            for edge in loop:
                try:
                    a, b = tuple(edge)[:2]
                    verts.add(int(a))
                    verts.add(int(b))
                except Exception:
                    continue
            loop_vertex_counts.append(int(len(verts)))
        return {
            "face_count": int(len(ids)),
            "boundary_edge_count": int(len(boundary)),
            "boundary_loop_count": int(len(loops)),
            "boundary_loop_edge_counts": tuple(int(len(loop)) for loop in loops),
            "boundary_loop_vertex_counts": tuple(loop_vertex_counts),
            "component_count": int(len(comps)),
            "component_face_counts": tuple(int(len(comp)) for comp in comps),
        }
    except Exception as exc:
        return {
            "face_count": int(len(ids)),
            "boundary_edge_count": 0,
            "boundary_loop_count": 0,
            "boundary_loop_edge_counts": (),
            "boundary_loop_vertex_counts": (),
            "component_count": 0,
            "component_face_counts": (),
            "boundary_report_error": str(exc),
        }

def _face_vertices_for_ids(faces: np.ndarray, face_ids: Iterable[int]) -> set[int]:
    """Return source vertex ids touched by the given face ids."""

    out: set[int] = set()
    arr = np.asarray(faces, dtype=np.int64)
    for fid in tuple_ints(face_ids):
        if 0 <= int(fid) < len(arr):
            for v in arr[int(fid), :3]:
                out.add(int(v))
    return out

def _face_edges_for_ids(faces: np.ndarray, face_ids: Iterable[int]) -> set[tuple[int, int]]:
    """Return undirected source edges touched by the given face ids."""

    out: set[tuple[int, int]] = set()
    arr = np.asarray(faces, dtype=np.int64)
    for fid in tuple_ints(face_ids):
        if 0 <= int(fid) < len(arr):
            tri = [int(v) for v in arr[int(fid), :3]]
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                if a == b:
                    continue
                out.add((a, b) if a < b else (b, a))
    return out

def _loop_records_vertices_and_edges(records: Iterable[Mapping[str, object]]) -> tuple[set[int], set[tuple[int, int]]]:
    """Collect protected-loop vertices/edges from diagnostic loop records."""

    vertices: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for row in tuple(records or ()):  # tolerate generators and None-like inputs
        for v in tuple(row.get("vertices", ()) or ()):  # type: ignore[union-attr]
            try:
                vertices.add(int(v))
            except Exception:
                continue
        for edge in tuple(row.get("edges", ()) or ()):  # type: ignore[union-attr]
            key = _normalize_edge_key(edge)
            if key != (-1, -1):
                edges.add(key)
    return vertices, edges

def filter_pocket_child_bore_intrusion_faces_v174l(
    *,
    faces: np.ndarray,
    side_wall_face_ids: Iterable[int],
    floor_face_ids: Iterable[int],
    protected_bore_opening_loops: Iterable[Mapping[str, object]],
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, object]]:
    """Split POCKET side-wall ownership away from child-BORE wall intrusion.

    This is a semantic ownership step, not a parameter fit.  A child bore opening
    recorded in the owned pocket floor is a protected boundary.  Faces from the
    child bore wall that touch that protected loop are not pocket side-wall role
    faces, even if they are cylindrical and close to the pocket footprint.  They
    remain relationship / rejected evidence and are not shown as owned POCKET
    CandidateData.
    """

    side_ids = tuple_ints(side_wall_face_ids)
    floor_ids = tuple_ints(floor_face_ids)
    protected_rows = tuple(dict(row) for row in tuple(protected_bore_opening_loops or ()))
    protected_vertices, protected_edges = _loop_records_vertices_and_edges(protected_rows)
    if not protected_rows or not protected_vertices:
        return side_ids, (), {
            "v158_pocket_child_boundary_filter_used": False,
            "v158_pocket_child_boundary_filter_reason": "no_protected_child_bore_floor_loop",
            "v158_pocket_child_bore_intrusion_face_count": 0,
            "v158_pocket_side_wall_face_count_before_child_filter": int(len(side_ids)),
            "v158_pocket_side_wall_face_count_after_child_filter": int(len(side_ids)),
            "pocket_child_bore_boundary_rebuild_safe_v158": False,
            "pocket_child_bore_boundary_rebuild_safe_v159": False,
            "v159_pocket_reauthorization_reason": "no_protected_child_bore_floor_loop",
        }

    kept: list[int] = []
    rejected: list[int] = []
    component_reports: list[dict[str, object]] = []
    for comp in connected_face_components(np.asarray(faces, dtype=np.int64), side_ids):
        comp_ids = tuple_ints(comp)
        comp_vertices = _face_vertices_for_ids(faces, comp_ids)
        comp_edges = _face_edges_for_ids(faces, comp_ids)
        protected_edge_touch = int(len(comp_edges & protected_edges))
        protected_vertex_touch = int(len(comp_vertices & protected_vertices))
        touches_child_boundary = bool(protected_edge_touch > 0 or protected_vertex_touch > 0)
        if touches_child_boundary:
            rejected.extend(comp_ids)
        else:
            kept.extend(comp_ids)
        component_reports.append({
            "face_count": int(len(comp_ids)),
            "touches_protected_child_bore_boundary": bool(touches_child_boundary),
            "protected_edge_touch_count": int(protected_edge_touch),
            "protected_vertex_touch_count": int(protected_vertex_touch),
            "semantic_role_v158": "rejected_child_bore_wall_intrusion" if touches_child_boundary else "owned_parent_pocket_side_wall",
        })

    kept_ids = tuple_ints(kept)
    rejected_ids = tuple_ints(rejected)
    floor_report = _face_patch_boundary_semantic_report(faces, floor_ids)
    combined_report = _face_patch_boundary_semantic_report(faces, tuple_ints(kept_ids + floor_ids))
    # Rebuild supports one protected child-BORE floor opening.  After removing
    # side-wall components that touch the protected child loop, a non-empty parent
    # sidewall plus owned floor can be a rebuildable pocket cup candidate.
    # v159: this is no longer only a safety/rejection flag.  If the semantic
    # split actually removed child-bore intrusion and leaves parent sidewall +
    # floor ownership, that cleaned face set is the rebuildable POCKET target.
    safe = bool(len(protected_rows) <= 1 and kept_ids and floor_ids and rejected_ids)
    safe_reason = (
        "cleaned_parent_pocket_wall_floor_after_child_bore_intrusion_split"
        if safe
        else (
            "protected_child_loop_present_but_no_intrusion_removed"
            if not rejected_ids
            else "cleaned_parent_pocket_wall_floor_missing_required_faces"
        )
    )
    return kept_ids, rejected_ids, {
        "v158_pocket_child_boundary_filter_used": True,
        "v158_pocket_child_boundary_filter_reason": "child_bore_wall_components_removed_from_parent_pocket_ownership" if rejected_ids else "protected_child_loop_present_no_sidewall_intrusion_component_found",
        "v158_pocket_child_bore_intrusion_face_ids": rejected_ids,
        "v158_pocket_child_bore_intrusion_face_count": int(len(rejected_ids)),
        "v158_pocket_side_wall_face_count_before_child_filter": int(len(side_ids)),
        "v158_pocket_side_wall_face_count_after_child_filter": int(len(kept_ids)),
        "v158_pocket_child_boundary_component_reports": tuple(component_reports),
        "v158_pocket_floor_boundary_loop_count_after_child_filter": int(floor_report.get("boundary_loop_count", 0) or 0),
        "v158_pocket_combined_boundary_loop_count_after_child_filter": int(combined_report.get("boundary_loop_count", 0) or 0),
        "pocket_child_bore_boundary_rebuild_safe_v158": bool(safe),
        "pocket_child_bore_boundary_rebuild_safe_v159": bool(safe),
        "pocket_child_bore_boundary_rebuild_safe_v157": bool(safe),
        "v159_pocket_reauthorization_reason": safe_reason,
        "v158_pocket_ownership_split": "parent_pocket_side_wall_and_floor_owned_child_bore_wall_rejected_as_relationship_evidence",
    }

def pocket_floor_boundary_loop_records_v174l(
    *,
    faces: np.ndarray,
    vertices: np.ndarray,
    floor_face_ids: tuple[int, ...],
    center: object,
    axis: object,
) -> tuple[dict[str, object], ...]:
    """Return neutral floor boundary-loop evidence for POCKET relationships.

    This does not create a new feature family.  It only describes the boundary
    loops of the owned pocket-floor role.  Recognition may use smaller interior
    loops as protected child-BORE opening evidence while keeping POCKET and BORE
    as separate feature hypotheses.
    """

    face_arr = np.asarray(faces, dtype=np.int64)
    verts = np.asarray(vertices, dtype=float)
    if face_arr.ndim != 2 or face_arr.shape[1] < 3 or verts.ndim != 2 or verts.shape[1] < 3:
        return ()
    ids = tuple_ints(floor_face_ids)
    if not ids:
        return ()
    axis_vec = canonical_axis(axis)
    center_vec = np.asarray(center, dtype=float).reshape(3)
    loops: list[dict[str, object]] = []
    try:
        boundary = boundary_edges_for_face_patch(face_arr[:, :3], ids)
        components = edge_loop_components(boundary)
    except Exception:
        return ()
    for index, comp in enumerate(components):
        edges = tuple(sorted({_normalize_edge_key(edge) for edge in tuple(comp or ()) if _normalize_edge_key(edge) != (-1, -1)}))
        loop_vertices = tuple(sorted({int(v) for edge in edges for v in edge if 0 <= int(v) < len(verts)}))
        if len(loop_vertices) < 3:
            continue
        pts = verts[np.asarray(loop_vertices, dtype=np.int64), :3]
        rel = pts - center_vec.reshape(1, 3)
        axial = rel @ axis_vec
        radial_vec = rel - axial.reshape(-1, 1) * axis_vec.reshape(1, 3)
        radii = np.linalg.norm(radial_vec, axis=1)
        loops.append({
            "index": int(index),
            "vertices": loop_vertices,
            "edges": edges,
            "vertex_count": int(len(loop_vertices)),
            "edge_count": int(len(edges)),
            "center": to_vector3(np.mean(pts, axis=0)),
            "radial_median": float(np.median(radii)) if radii.size else 0.0,
            "radial_max": float(np.max(radii)) if radii.size else 0.0,
            "axial_median": float(np.median(axial)) if axial.size else 0.0,
        })
    loops.sort(key=lambda row: (-_safe_float(row.get("radial_median", 0.0), 0.0), -int(row.get("edge_count", 0) or 0)))
    return tuple(loops)

def pocket_floor_compound_boundary_metadata_v174l(
    *,
    faces: np.ndarray,
    vertices: np.ndarray,
    floor_face_ids: tuple[int, ...],
    center: object,
    axis: object,
) -> dict[str, object]:
    """Return relationship metadata for a POCKET floor that contains BORE openings.

    The returned rows are metadata only.  They do not create a
    `pocket_with_bore` family.  The POCKET candidate may own the floor around a
    child bore, while the child BORE remains a separate rebuildable feature
    object.  Rebuild Target later protects these loops when rebuilding the
    pocket floor.
    """

    loops = pocket_floor_boundary_loop_records_v174l(
        faces=faces,
        vertices=vertices,
        floor_face_ids=floor_face_ids,
        center=center,
        axis=axis,
    )
    if not loops:
        return {
            "pocket_floor_boundary_loop_count": 0,
            "pocket_protected_floor_bore_opening_count": 0,
            "pocket_protected_floor_bore_opening_loops": (),
            "pocket_floor_outer_boundary_loop": {},
        }
    outer = loops[0]
    outer_radius = max(_safe_float(outer.get("radial_median", 0.0), 0.0), 1.0e-9)
    protected: list[dict[str, object]] = []
    for row in loops[1:]:
        radius = _safe_float(row.get("radial_median", 0.0), 0.0)
        edge_count = int(row.get("edge_count", 0) or 0)
        # An inner boundary of the owned pocket floor is protected opening
        # evidence for a separate child BORE.  It is not pocket ownership and
        # not a new feature family.
        if edge_count >= 6 and radius <= 0.90 * outer_radius:
            protected.append({
                **dict(row),
                "relationship_meaning": "protected_child_bore_opening_boundary_in_owned_pocket_floor",
                "semantic_status": "relationship_metadata_only_not_feature_family",
            })
    return {
        "pocket_floor_boundary_loop_count": int(len(loops)),
        "pocket_floor_boundary_loops": loops,
        "pocket_floor_outer_boundary_loop": dict(outer),
        "pocket_protected_floor_bore_opening_count": int(len(protected)),
        "pocket_protected_floor_bore_opening_loops": tuple(protected),
        "pocket_compound_relationship_model": "POCKET candidate plus separate BORE candidate; protected floor opening metadata only",
    }


def pocket_family_authority_leak_diagnostics_v174n(
    *,
    candidate_family: str,
    feature_ownership_source: object,
    diagnostics: Mapping[str, object],
    primitive_sources: Iterable[object] = (),
) -> dict[str, object]:
    """Return POCKET-family authority-leak diagnostics without changing behavior.

    v174n is a recognition-side audit only.  The smoke test showed a POCKET
    rebuild log reporting a BORE owned-wall cylinder authority.  Recognition
    does not fix rebuild shape authority here; it only publishes the POCKET
    candidate's expected family authority and scans candidate metadata for
    suspicious BORE-wall authority tokens so a later rebuild/refactor patch can
    correct the actual source if needed.
    """

    family = str(candidate_family or "").strip().lower() or FeatureFamily.POCKET.value
    source = str(feature_ownership_source or "").strip()
    candidate_diag = dict(diagnostics or {})

    ignored_key_fragments = (
        "child",
        "protected",
        "relationship",
        "embedded",
        "contains_bore_opening",
    )
    inspected_key_fragments = (
        "authority",
        "source",
        "policy",
        "constraint",
        "ownership",
    )
    suspect_value_fragments = (
        "owned_bore_wall",
        "bore_wall",
        "bore_cylindrical",
        "bore_cylinder",
        "borehole_wall",
    )

    observed_tokens: list[dict[str, object]] = []
    suspect_tokens: list[dict[str, object]] = []

    def _record(key: str, value: object) -> None:
        key_text = str(key or "").strip()
        key_lower = key_text.lower()
        if any(fragment in key_lower for fragment in ignored_key_fragments):
            return
        if not any(fragment in key_lower for fragment in inspected_key_fragments):
            return
        value_text = str(value or "").strip()
        if not value_text:
            return
        row = {"key": key_text, "value": value_text[:240]}
        observed_tokens.append(row)
        value_lower = value_text.lower()
        if any(fragment in value_lower for fragment in suspect_value_fragments):
            suspect_tokens.append(row)

    _record("feature_ownership_source", source)
    for idx, primitive_source in enumerate(tuple(primitive_sources or ())):
        _record(f"feature_primitive_source_{idx}", primitive_source)
    for key, value in candidate_diag.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            _record(str(key), value)

    mismatch = bool(family in {FeatureFamily.POCKET.value, FeatureFamily.CIRCULAR_POCKET.value, "pocket", "circular_pocket"} and suspect_tokens)
    return {
        "pocket_authority_leak_audit_v174n": True,
        "pocket_authority_leak_diagnostic_only_v174n": True,
        "pocket_authority_leak_checkpoint_v174n": POCKET_RECOGNITION_AUTHORITY_LEAK_DIAGNOSTIC_CHECKPOINT_V174N,
        "pocket_expected_authority_family_v174n": "pocket",
        "pocket_candidate_family_v174n": family,
        "pocket_expected_rebuild_shape_authority_v174n": "owned_pocket_floor_side_wall_recess_authority_not_bore_wall",
        "pocket_candidate_feature_ownership_source_v174n": source,
        "pocket_observed_candidate_authority_tokens_v174n": tuple(observed_tokens[:16]),
        "pocket_authority_leak_suspect_tokens_v174n": tuple(suspect_tokens[:16]),
        "pocket_radius_authority_family_mismatch_v174n": bool(mismatch),
        "pocket_authority_leak_status_v174n": (
            "candidate_metadata_contains_bore_wall_authority_token_diagnostic_only"
            if mismatch
            else "no_recognition_side_bore_wall_authority_token_seen_rebuild_side_must_still_be_checked"
        ),
        "pocket_rebuild_authority_smoke_test_note_v174n": (
            "If a later rebuild log for family=pocket reports owned_bore_wall_cylinder_shape_authority_v173x, "
            "treat that as a rebuild-side family authority leak to fix after the recognition split roadmap."
        ),
    }

def pocket_candidate_contract_fields_v174l(
    *,
    candidate_id: str,
    side_wall_face_ids: tuple[int, ...],
    floor_face_ids: tuple[int, ...],
    transition_face_ids: tuple[int, ...],
    protected_bore_opening_loops: tuple[Mapping[str, object], ...],
    embedded_bore_wall_evidence_face_ids: tuple[int, ...],
    confidence: float,
    depth: float,
    axis: object,
    rim_center: object,
    floor_center: object,
    footprint_radius: float,
    pocket_kind: str,
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Return accepted CandidateData for a recognized POCKET.

    This is a POCKET semantic object, not a disguised BORE.  Recognition has
    transformed measured rim/floor/side-wall/depth evidence into a POCKET
    hypothesis and then into owned pocket floor and side-wall surface roles.
    CandidateData may expose those owned roles to Rebuild Target, but it does
    not itself mutate topology.  The v99 rebuild path consumes the owned floor
    + side-wall roles to rebuild a recessed pocket cup: side wall plus floor,
    with the top opening left open and transition/chamfer evidence excluded.
    """

    side_wall_face_ids = tuple_ints(side_wall_face_ids)
    floor_face_ids = tuple_ints(floor_face_ids)
    transition_face_ids = tuple_ints(transition_face_ids)
    embedded_bore_wall_evidence_face_ids = tuple_ints(embedded_bore_wall_evidence_face_ids)
    protected_bore_opening_loops = tuple(dict(row) for row in tuple(protected_bore_opening_loops or ()))
    protected_loop_vertices = tuple(
        tuple_ints(row.get("vertices", ()))
        for row in protected_bore_opening_loops
        if tuple_ints(row.get("vertices", ()))
    )
    protected_loop_edges = tuple(
        tuple(tuple(_normalize_edge_key(edge) for edge in tuple(row.get("edges", ()) or ()) if _normalize_edge_key(edge) != (-1, -1)))
        for row in protected_bore_opening_loops
    )
    has_child_bore = bool(protected_loop_vertices or embedded_bore_wall_evidence_face_ids)
    # v173b compound POCKET semantics, restored from the v172b repair and kept
    # as a role/relationship transform instead of a post-candidate gate:
    #
    #   child BORE opening in a POCKET floor = protected relationship boundary
    #   child BORE wall/interior faces inside owned POCKET target = intrusion
    #
    # Only the second case blocks the parent POCKET cup CandidateData.  A child
    # opening loop carried as relationship metadata must not erase the owned
    # POCKET floor + side-wall role that Recognition has already created.
    pocket_child_bore_boundary_rebuild_safe_v158 = bool(
        diagnostics.get("pocket_child_bore_boundary_rebuild_safe_v158", False)
    )
    pocket_child_bore_boundary_rebuild_safe_v159 = bool(
        diagnostics.get("pocket_child_bore_boundary_rebuild_safe_v159", pocket_child_bore_boundary_rebuild_safe_v158)
    )
    pocket_child_bore_legacy_safe_v157 = bool(
        diagnostics.get("pocket_child_bore_boundary_rebuild_safe_v157", False)
        or pocket_child_bore_boundary_rebuild_safe_v158
        or pocket_child_bore_boundary_rebuild_safe_v159
    )
    try:
        child_bore_intrusion_diag_count_v173b = int(
            diagnostics.get(
                "v158_pocket_child_bore_intrusion_face_count",
                diagnostics.get("pocket_child_bore_intrusion_face_count_v158", 0),
            ) or 0
        )
    except Exception:
        child_bore_intrusion_diag_count_v173b = 0
    child_bore_intrusion_face_count_v173b = int(
        len(embedded_bore_wall_evidence_face_ids)
        + max(0, child_bore_intrusion_diag_count_v173b)
    )
    child_bore_metadata_only_v173b = bool(
        protected_loop_vertices
        and child_bore_intrusion_face_count_v173b <= 0
    )
    protected_floor_opening_relationship_safe_v173b = bool(
        child_bore_metadata_only_v173b
        and side_wall_face_ids
        and floor_face_ids
    )
    pocket_child_bore_boundary_rebuild_safe_v173b = bool(
        not has_child_bore
        or pocket_child_bore_legacy_safe_v157
        or protected_floor_opening_relationship_safe_v173b
    )
    # Keep the legacy v157 key as a compatibility alias for old UI/debug text,
    # but its value now mirrors the actual compound relationship meaning.
    pocket_child_bore_boundary_rebuild_safe_v157 = bool(pocket_child_bore_boundary_rebuild_safe_v173b)
    child_bore_boundary_mode_v173b = (
        "no_child_bore_relationship"
        if not has_child_bore
        else (
            "protected_floor_opening_relationship_not_intrusion"
            if protected_floor_opening_relationship_safe_v173b
            else (
                "legacy_cleaned_child_boundary_split"
                if pocket_child_bore_legacy_safe_v157
                else "child_bore_intrusion_or_unproven_boundary"
            )
        )
    )
    semantic_face_ids = tuple_ints(side_wall_face_ids + floor_face_ids)
    display_face_ids = tuple_ints(semantic_face_ids + transition_face_ids)
    axis_tuple = to_vector3(axis)
    rim_center_tuple = to_vector3(rim_center)
    floor_center_tuple = to_vector3(floor_center)
    family = FeatureFamily.CIRCULAR_POCKET.value if str(pocket_kind).lower() == "circular" else FeatureFamily.POCKET.value
    display_kind = "CIRCULAR POCKET" if family == FeatureFamily.CIRCULAR_POCKET.value else "POCKET"
    radius_min = _safe_float(diagnostics.get("radius_min", diagnostics.get("selected_opening_radius_min", footprint_radius)), footprint_radius)
    radius_nominal = _safe_float(diagnostics.get("radius_nominal", diagnostics.get("radius", footprint_radius)), footprint_radius)
    radius_max = _safe_float(diagnostics.get("radius_max", diagnostics.get("selected_opening_radius_max", footprint_radius)), footprint_radius)
    if radius_min > radius_max:
        radius_min, radius_max = radius_max, radius_min
    diameter_min = _safe_float(diagnostics.get("diameter_min", 2.0 * radius_min), 2.0 * radius_min)
    diameter_nominal = _safe_float(diagnostics.get("diameter_nominal", 2.0 * radius_nominal), 2.0 * radius_nominal)
    diameter_max = _safe_float(diagnostics.get("diameter_max", 2.0 * radius_max), 2.0 * radius_max)
    radial_spread = _safe_float(diagnostics.get("radial_spread", max(0.0, radius_max - radius_min)), max(0.0, radius_max - radius_min))
    radial_spread_ratio = _safe_float(diagnostics.get("radial_spread_ratio", radial_spread / max(float(radius_nominal), 1.0e-12)), radial_spread / max(float(radius_nominal), 1.0e-12))
    radial_envelope = diagnostics.get("radial_envelope", {})
    if not isinstance(radial_envelope, Mapping):
        radial_envelope = {}
    coarse_polygonal = bool(diagnostics.get("coarse_polygonal", radial_spread_ratio >= 0.045))
    feature_ownership_source = "recognition_component_engine_pocket_hypothesis_floor_sidewall_role_ownership_child_bore_relationships"
    authority_leak_diagnostics_v174n = pocket_family_authority_leak_diagnostics_v174n(
        candidate_family=family,
        feature_ownership_source=feature_ownership_source,
        diagnostics=diagnostics,
        primitive_sources=(
            "recognition_component_engine.v102_pocket_recess_cup_rebuild",
            "recognition_component_engine.v102_pocket_recess_cup_rebuild",
            "recognition_component_engine.v102_pocket_recess_cup_rebuild",
        ),
    )
    candidate_diagnostics = dict(diagnostics)
    candidate_diagnostics.update(authority_leak_diagnostics_v174n)

    # v145: Rebuild is not allowed to consume a visually plausible but
    # geometrically unstable POCKET ownership set.  The coarse auto-select
    # path can produce a local-looking preview whose floor/wall roles are
    # still not a bounded rebuild target: wide radial envelope, raw-vs-normalized
    # opening disagreement, or normalized-candidate authority from a contaminated
    # edge cloud.  Keep such objects previewable for inspection, but do not expose
    # them as delete/rebuild targets until the opening/rim acquisition stage is
    # seed-local and the owned face set passes these invariants.
    audit_status = str(
        diagnostics.get(
            "audit_status",
            diagnostics.get("opening_measurement_audit_status", ""),
        )
        or ""
    )
    resolver_source = str(
        diagnostics.get(
            "resolver_source",
            diagnostics.get("selected_opening_frame_source", ""),
        )
        or ""
    )
    realization_kind = str(diagnostics.get("realization_kind", diagnostics.get("opening_realization_kind", "")) or "")
    raw_vs_normalized_agree = diagnostics.get("raw_vs_normalized_measurement_agree", None)
    normalized_to_raw_ratio = _safe_float(diagnostics.get("normalized_to_raw_edge_ratio", 1.0), 1.0)
    centerline_delta = _safe_float(diagnostics.get("raw_vs_normalized_centerline_distance", 0.0), 0.0)
    max_rebuild_radial_spread_ratio = 0.28 if coarse_polygonal else 0.12
    face_selection_rejection_reasons: list[str] = []
    face_selection_warning_reasons: list[str] = []

    if len(side_wall_face_ids) < 16:
        face_selection_rejection_reasons.append("pocket_sidewall_face_selection_too_small_for_rebuild")
    if len(floor_face_ids) < 4:
        face_selection_rejection_reasons.append("pocket_floor_face_selection_too_small_for_rebuild")
    if not math.isfinite(float(radial_spread_ratio)):
        face_selection_rejection_reasons.append("pocket_radial_spread_ratio_invalid")
    elif float(radial_spread_ratio) > float(max_rebuild_radial_spread_ratio):
        # v147: on coarse tessellated pockets the opening/radius envelope can be
        # very noisy even when the owned floor + side-wall face set is visually
        # correct and topologically rebuildable.  Treat spread as a warning, not
        # a hard rebuild blocker.  The actual topology decision belongs to the
        # rebuild target / rebuild validator (v146 now rejects bad loop plans
        # cleanly without a traceback).
        face_selection_warning_reasons.append("pocket_opening_radial_spread_too_wide_for_rebuild_review")
    if audit_status and audit_status not in {"raw_and_normalized_opening_measurements_agree", "", "-"}:
        face_selection_warning_reasons.append("pocket_opening_measurement_ambiguous_review")
    if raw_vs_normalized_agree is False:
        face_selection_warning_reasons.append("pocket_raw_normalized_opening_disagree_review")
    if normalized_to_raw_ratio > 1.25 or normalized_to_raw_ratio < 0.75:
        face_selection_warning_reasons.append("pocket_normalized_rim_edge_count_not_seed_stable_review")
    if centerline_delta > max(2.0, 0.25 * max(float(radius_nominal), 1.0)):
        face_selection_warning_reasons.append("pocket_opening_centerline_not_seed_stable_review")
    if resolver_source == "normalized_candidates" and realization_kind == "contaminated_edge_cloud":
        face_selection_warning_reasons.append("pocket_resolved_from_contaminated_normalized_candidates_review")
    if has_child_bore and not pocket_child_bore_boundary_rebuild_safe_v173b:
        face_selection_rejection_reasons.append("pocket_child_bore_boundary_not_rebuild_safe_v173b")

    face_selection_rejection_reasons = list(dict.fromkeys(face_selection_rejection_reasons))
    face_selection_warning_reasons = list(dict.fromkeys(face_selection_warning_reasons))
    pocket_rebuild_face_selection_safe = not face_selection_rejection_reasons
    pocket_status = (
        "pocket_recognition_accepted_recess_cup_rebuild_target"
        if pocket_rebuild_face_selection_safe
        else "pocket_recognition_preview_only_face_selection_not_rebuild_safe"
    )
    rebuild_block_reason_v145 = "; ".join(face_selection_rejection_reasons)

    return {
        "candidate_id": candidate_id,
        "feature_id": candidate_id,
        "entity_type": "pocket",
        "feature_kind": "pocket",
        "candidate_scope": "candidate_data",
        "candidate_variant": str(pocket_kind or "freeform"),
        "candidate_authority": "surface_component_classifier_v102_pocket_recess_hypothesis_role_ownership_compound_relationships",
        "active_candidate_authority": "surface_component_classifier_v102_pocket_recess_hypothesis_role_ownership_compound_relationships",
        "feature_family": family,
        "recognition_stage": RecognitionStage.ACCEPTED_CANDIDATE.value if pocket_rebuild_face_selection_safe else RecognitionStage.REVIEW.value,
        "display_name": f"{display_kind} — pocket recess rebuild",
        "role": "pocket_recess_rebuild_floor_and_sidewall_owned_roles" if pocket_rebuild_face_selection_safe else "pocket_recess_review_child_bore_boundary_not_proven",
        "status": pocket_status,
        "promotion_state": "promoted" if pocket_rebuild_face_selection_safe else "evidence_only",
        "candidate_action_enabled": bool(pocket_rebuild_face_selection_safe),
        "candidate_action": "delete_and_rebuild_owned_pocket_floor_and_side_wall_as_recess_cup",
        "rebuild_authorized": bool(pocket_rebuild_face_selection_safe),
        "rebuild_gate": "pocket_recess_cup_rebuild_target_contract_v102_protect_child_bore_boundaries",
        "rebuild_block_reason": rebuild_block_reason_v145,
        "face_ids": semantic_face_ids,
        "semantic_face_ids": semantic_face_ids,
        "display_face_ids": display_face_ids,
        "preview_face_ids": display_face_ids,
        "rebuild_face_ids": semantic_face_ids if pocket_rebuild_face_selection_safe else (),
        "candidate_rebuild_face_ids": semantic_face_ids if pocket_rebuild_face_selection_safe else (),
        "owned_face_ids": semantic_face_ids,
        "pocket_side_wall_face_ids": side_wall_face_ids,
        "pocket_side_wall_face_count": int(len(side_wall_face_ids)),
        "pocket_floor_face_ids": floor_face_ids,
        "pocket_floor_face_count": int(len(floor_face_ids)),
        "pocket_transition_face_ids": transition_face_ids,
        "pocket_transition_face_count": int(len(transition_face_ids)),
        "pocket_has_child_bore_opening": bool(has_child_bore),
        "pocket_protected_floor_bore_opening_count": int(len(protected_loop_vertices)),
        "pocket_protected_floor_bore_opening_loop_vertex_ids": protected_loop_vertices,
        "pocket_protected_floor_bore_opening_loop_edges": protected_loop_edges,
        "pocket_embedded_bore_wall_evidence_face_ids": embedded_bore_wall_evidence_face_ids,
        "pocket_embedded_bore_wall_evidence_face_count": int(len(embedded_bore_wall_evidence_face_ids)),
        "pocket_child_bore_boundary_rebuild_safe_v157": bool(pocket_child_bore_boundary_rebuild_safe_v157),
        "pocket_child_bore_boundary_rebuild_safe_v158": bool(pocket_child_bore_boundary_rebuild_safe_v158),
        "pocket_child_bore_boundary_rebuild_safe_v159": bool(pocket_child_bore_boundary_rebuild_safe_v159),
        "pocket_child_bore_boundary_rebuild_safe_v173b": bool(pocket_child_bore_boundary_rebuild_safe_v173b),
        "pocket_child_bore_boundary_rebuild_safe_v173c": bool(pocket_child_bore_boundary_rebuild_safe_v173b),
        "pocket_child_bore_boundary_mode_v173c": child_bore_boundary_mode_v173b,
        "pocket_child_bore_boundary_legacy_safe_v157": bool(pocket_child_bore_legacy_safe_v157),
        "pocket_child_bore_boundary_metadata_only_v173b": bool(child_bore_metadata_only_v173b),
        "pocket_child_bore_boundary_relationship_protected_v173b": bool(protected_floor_opening_relationship_safe_v173b),
        "pocket_child_bore_boundary_intrusion_face_count_v173b": int(child_bore_intrusion_face_count_v173b),
        "pocket_child_bore_boundary_mode_v173b": child_bore_boundary_mode_v173b,
        "v173b_compound_pocket_reauth_used": bool(has_child_bore and protected_floor_opening_relationship_safe_v173b and not pocket_child_bore_legacy_safe_v157),
        "v173b_compound_pocket_reauthorization_reason": (
            "child_bore_is_protected_floor_opening_relationship_not_pocket_owned_bore_wall_intrusion"
            if protected_floor_opening_relationship_safe_v173b and not pocket_child_bore_legacy_safe_v157
            else "legacy_or_no_child_bore_path"
        ),
        "v158_pocket_child_boundary_filter_used": bool(diagnostics.get("v158_pocket_child_boundary_filter_used", False)),
        "v158_pocket_child_boundary_filter_reason": str(diagnostics.get("v158_pocket_child_boundary_filter_reason", "")),
        "v158_pocket_child_bore_intrusion_face_count": int(diagnostics.get("v158_pocket_child_bore_intrusion_face_count", 0) or 0),
        "v158_pocket_side_wall_face_count_before_child_filter": int(diagnostics.get("v158_pocket_side_wall_face_count_before_child_filter", len(side_wall_face_ids)) or 0),
        "v158_pocket_side_wall_face_count_after_child_filter": int(diagnostics.get("v158_pocket_side_wall_face_count_after_child_filter", len(side_wall_face_ids)) or 0),
        "v158_pocket_combined_boundary_loop_count_after_child_filter": int(diagnostics.get("v158_pocket_combined_boundary_loop_count_after_child_filter", 0) or 0),
        "v159_pocket_reauthorization_reason": str(diagnostics.get("v159_pocket_reauthorization_reason", "")),
        "v159_pocket_reauthorized_after_cleaned_child_boundary_split": bool(has_child_bore and pocket_child_bore_boundary_rebuild_safe_v159),
        "pocket_child_bore_boundary_policy_v157": "legacy compatibility alias; superseded by v173b compound protected-floor-opening rule",
        "pocket_child_bore_boundary_policy_v159": "cleaned_parent_pocket_sidewall_plus_floor_face_set_can_rebuild_after_child_bore_intrusion_split",
        "pocket_child_bore_boundary_policy_v173b": "protected child-bore floor-opening relationship does not block parent POCKET cup rebuild; block only child-bore wall/interior intrusion into owned pocket target",
        "compound_feature_model": "separate POCKET candidate plus separate BORE candidate; protected child-bore opening relationship metadata only",
        "face_count": int(len(semantic_face_ids)),
        "display_face_count": int(len(display_face_ids)),
        "confidence": float(max(0.0, min(0.95, confidence))),
        "depth": float(depth),
        "height": float(depth),
        "axial_span": float(depth),
        "axis": axis_tuple,
        "primitive_axis": axis_tuple,
        "rim_center": rim_center_tuple,
        "floor_center": floor_center_tuple,
        "primitive_depth": float(depth),
        "primitive_radius": float(footprint_radius),
        "primitive_radius_min": float(radius_min),
        "primitive_radius_max": float(radius_max),
        "primitive_radial_spread_ratio": float(radial_spread_ratio),
        "radius": float(footprint_radius),
        "radius_min": float(radius_min),
        "radius_nominal": float(radius_nominal),
        "radius_max": float(radius_max),
        "diameter": float(2.0 * footprint_radius),
        "diameter_min": float(diameter_min),
        "diameter_nominal": float(diameter_nominal),
        "diameter_max": float(diameter_max),
        "radial_spread": float(radial_spread),
        "radial_spread_ratio": float(radial_spread_ratio),
        "coarse_polygonal": bool(coarse_polygonal),
        "radial_envelope": dict(radial_envelope),
        "pocket_rebuild_face_selection_gate_v145": bool(pocket_rebuild_face_selection_safe),
        "pocket_rebuild_face_selection_rejection_reasons_v145": tuple(face_selection_rejection_reasons),
        "pocket_rebuild_face_selection_max_radial_spread_ratio_v145": float(max_rebuild_radial_spread_ratio),
        "pocket_rebuild_face_selection_audit_status_v145": audit_status,
        "pocket_rebuild_face_selection_resolver_source_v145": resolver_source,
        "pocket_rebuild_face_selection_gate_v147": bool(pocket_rebuild_face_selection_safe),
        "pocket_rebuild_face_selection_warning_reasons_v147": tuple(face_selection_warning_reasons),
        "pocket_rebuild_gate_policy_v147": "coarse radial/opening ambiguity is review warning; topology validator owns final rebuild rejection",
        "surface_condition": "owned_pocket_floor_and_side_wall_faces_for_recess_cup_rebuild",
        "repair_strategy": "pocket_native_recess_cup_rebuild_delete_owned_floor_plus_sidewall_generate_sidewall_plus_floor",
        "evidence_kinds": tuple(v for v in (
            EvidenceKind.POCKET_RIM.value,
            EvidenceKind.POCKET_FLOOR.value,
            EvidenceKind.POCKET_SIDE_WALL.value,
            EvidenceKind.POCKET_DEPTH.value,
            EvidenceKind.POCKET_TRANSITION.value,
            EvidenceKind.POCKET_PROTECTED_BORE_OPENING.value if has_child_bore else "",
        ) if v),
        "promotion_reasons": tuple(v for v in (
            "pocket_floor_surface_role_ownership",
            "pocket_side_wall_surface_role_ownership",
            "positive_recess_depth_resolved",
            "transition_faces_separated_from_floor_wall_ownership",
            "rebuild_faces_are_owned_pocket_side_wall_plus_floor_roles",
            "floor_faces_are_owned_pocket_floor_rebuild_role_not_parent_cap",
            "child_bore_opening_is_relationship_metadata_not_pocket_family" if has_child_bore else "",
        ) if v),
        "rejection_reasons": tuple(face_selection_rejection_reasons),
        "warning_reasons": tuple(face_selection_warning_reasons),
        "feature_primitives": (
            {
                "primitive_kind": FeaturePrimitiveKind.POCKET_RECESS.value,
                "source": "recognition_component_engine.v102_pocket_recess_cup_rebuild",
                "role": "preview_descriptor_not_cad_body_not_rebuild_target",
                "center": rim_center_tuple,
                "axis": axis_tuple,
                "radius": float(footprint_radius),
                "radius_min": float(radius_min),
                "radius_nominal": float(radius_nominal),
                "radius_max": float(radius_max),
                "diameter": float(2.0 * footprint_radius),
                "diameter_min": float(diameter_min),
                "diameter_nominal": float(diameter_nominal),
                "diameter_max": float(diameter_max),
                "radial_spread": float(radial_spread),
                "radial_spread_ratio": float(radial_spread_ratio),
                "coarse_polygonal": bool(coarse_polygonal),
                "radial_envelope": dict(radial_envelope),
                "depth": float(depth),
                "confidence": float(max(0.0, min(0.95, confidence))),
                "face_ids": semantic_face_ids,
                "diagnostics": dict(candidate_diagnostics),
            },
            {
                "primitive_kind": FeaturePrimitiveKind.PLANAR_FLOOR.value,
                "source": "recognition_component_engine.v102_pocket_recess_cup_rebuild",
                "role": "owned_pocket_floor_surface_role",
                "center": floor_center_tuple,
                "axis": axis_tuple,
                "depth": float(depth),
                "confidence": float(max(0.0, min(0.95, confidence))),
                "face_ids": floor_face_ids,
                "diagnostics": dict(candidate_diagnostics),
            },
            {
                "primitive_kind": FeaturePrimitiveKind.POCKET_SIDE_WALL_SET.value,
                "source": "recognition_component_engine.v102_pocket_recess_cup_rebuild",
                "role": "owned_pocket_side_wall_surface_role",
                "center": rim_center_tuple,
                "axis": axis_tuple,
                "radius": float(footprint_radius),
                "radius_min": float(radius_min),
                "radius_nominal": float(radius_nominal),
                "radius_max": float(radius_max),
                "diameter_min": float(diameter_min),
                "diameter_nominal": float(diameter_nominal),
                "diameter_max": float(diameter_max),
                "radial_spread": float(radial_spread),
                "radial_spread_ratio": float(radial_spread_ratio),
                "coarse_polygonal": bool(coarse_polygonal),
                "radial_envelope": dict(radial_envelope),
                "depth": float(depth),
                "confidence": float(max(0.0, min(0.95, confidence))),
                "face_ids": side_wall_face_ids,
                "diagnostics": dict(candidate_diagnostics),
            },
        ),
        "feature_primitive_count": 3,
        "feature_relationships": tuple(
            {
                "relationship_kind": FeatureRelationshipKind.POCKET_CONTAINS_BORE_OPENING.value,
                "source": "recognition_component_engine.v102_compound_pocket_child_bore_relationship",
                "parent_feature_kind": "pocket",
                "child_feature_kind": "borehole",
                "role": "protected_floor_opening_relationship_metadata_only",
                "protected_loop_vertex_ids": tuple(loop),
                "relationship_semantics": "compound object: POCKET remains POCKET; BORE remains BORE; this metadata protects the floor hole during POCKET rebuild",
            }
            for loop in protected_loop_vertices
        ),
        "feature_relationship_count": int(len(protected_loop_vertices)),
        "x1_primitive_bridge_contract": "mesh-native pocket recess descriptor; rebuild target is owned pocket floor plus side-wall, not a bore wall and not a parent-surface cap",
        "delete_patch_request_allowed": bool(pocket_rebuild_face_selection_safe),
        "rebuild_target_policy_allowed": bool(pocket_rebuild_face_selection_safe),
        "rebuild_target_policy_reason": (
            "POCKET recess-cup rebuild is allowed; delete patch is owned side-wall plus owned floor, transition faces excluded"
            if pocket_rebuild_face_selection_safe
            else "POCKET preview only: owned floor/side-wall face selection is not rebuild-safe; fix rim/opening acquisition before DeletePatchProposal"
        ),
        "recognition_rule": "v102_pocket_hypothesis_floor_sidewall_ownership_with_child_bore_relationship_metadata",
        "feature_ownership_source": "recognition_component_engine_pocket_hypothesis_floor_sidewall_role_ownership_child_bore_relationships",
        "feature_ownership_split": "rebuild_face_ids_are_owned_pocket_side_wall_plus_floor_faces_transition_excluded",
        "pocket_rebuild_enable_scope": "owned_floor_plus_side_wall_recess_cup_protect_child_bore_openings",
        "pocket_rebuild_floor_policy": "floor_faces_are_owned_floor_role_and_rebuilt_as_bottom_of_recess",
        **authority_leak_diagnostics_v174n,
        "diagnostics": dict(candidate_diagnostics),
    }




def x1_probe_ledger_from_region_diagnostics_v174m(region_diagnostics: Mapping[str, object]) -> dict[str, object]:
    """Return the v133/v134 X1-style probe ledger from RegionData diagnostics.

    The ledger is POCKET-family measurement evidence only.  This helper does not
    promote the ledger by itself; callers still apply recognition/ownership
    gates and final CandidateData assembly outside this helper.
    """

    for key in ("opening_probe_ledger", "x1_style_opening_probe_ledger"):
        value = region_diagnostics.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def x1_probe_nested_mapping_v174m(value: Mapping[str, object], key: str) -> dict[str, object]:
    """Return a nested X1 probe mapping for POCKET evidence diagnostics."""

    nested = value.get(key)
    return dict(nested) if isinstance(nested, Mapping) else {}


def x1_probe_face_ids_v174m(
    ledger: Mapping[str, object],
    key: str,
    *,
    valid_face_set: set[int],
) -> tuple[int, ...]:
    """Return validated face IDs from a POCKET X1 probe support bucket."""

    support = ledger.get("support_face_ids")
    if not isinstance(support, Mapping):
        return ()
    return tuple(
        sorted({
            int(fid)
            for fid in tuple_ints(support.get(key, ()))
            if int(fid) in valid_face_set
        })
    )


def x1_probe_supports_local_pocket_candidate_v174m(
    ledger: Mapping[str, object],
    *,
    floor_face_ids: tuple[int, ...],
    side_wall_face_ids: tuple[int, ...],
    min_depth: float,
) -> tuple[bool, dict[str, object]]:
    """Gate X1-style probe evidence before POCKET CandidateData emission.

    This remains a Recognition-stage POCKET evidence gate.  It turns the v133
    probe into a candidate input only when side-wall/floor/depth evidence is
    strong enough.  It does not emit CandidateData, authorize a delete patch, or
    mutate topology.
    """

    relationships = x1_probe_nested_mapping_v174m(ledger, "relationships")
    probe_counts = x1_probe_nested_mapping_v174m(ledger, "probe_counts")
    confidence = _safe_float(ledger.get("confidence", 0.0), 0.0)
    resolved_depth = _safe_float(relationships.get("resolved_depth", 0.0), 0.0)
    relation_hint = str(relationships.get("relation_hint", "") or "")
    sidewall_support = bool(relationships.get("sidewall_support", False))
    floor_support = bool(relationships.get("floor_or_blind_end_support", False))

    reasons: list[str] = []
    if confidence < 0.58:
        reasons.append("probe_confidence_below_candidate_gate")
    if resolved_depth <= max(float(min_depth), 1.0e-9):
        reasons.append("probe_depth_below_candidate_gate")
    if not sidewall_support:
        reasons.append("probe_sidewall_support_missing")
    if not floor_support:
        reasons.append("probe_floor_support_missing")
    if relation_hint not in {
        "local_coaxial_recess_or_blind_pocket_probe_support",
        "local_coaxial_sidewall_ring_stack_probe_support",
    }:
        reasons.append("probe_relation_hint_not_pocket_like")
    if len(floor_face_ids) < 3:
        reasons.append("probe_floor_face_ids_too_few")
    if len(side_wall_face_ids) < 8:
        reasons.append("probe_sidewall_face_ids_too_few")

    diag = {
        "probe_candidate_gate_version": "v134.x1_probe_ledger_to_candidate_data.1",
        "probe_candidate_gate_passed": not reasons,
        "probe_candidate_rejection_reasons": tuple(reasons),
        "probe_confidence": float(confidence),
        "probe_relation_hint": str(relation_hint),
        "probe_resolved_depth": float(resolved_depth),
        "probe_sidewall_like_face_count": int(len(side_wall_face_ids)),
        "probe_floor_like_face_count": int(len(floor_face_ids)),
        "probe_local_face_count": int(probe_counts.get("local_probe_face_count", 0) or 0),
        "probe_broad_region_face_count": int(probe_counts.get("broad_region_face_count", 0) or 0),
        "v174m_pocket_x1_probe_helper_split": True,
        "v174m_candidate_emission_still_outside_helper": True,
    }
    return (not reasons), diag

def detect_pocket_preview_candidates_v174o(
    *,
    faces: np.ndarray,
    vertices: np.ndarray,
    face_centroids: np.ndarray,
    face_normals: np.ndarray,
    valid_face_ids: tuple[int, ...],
    region_frame: object,
    seed_face_set: set[int],
    adjacency: Mapping[int, tuple[int, ...]],
    selected_opening_frame_resolver: Mapping[str, object],
    region_diagnostics: Mapping[str, object],
    opening_probe_ledger: Mapping[str, object] | None = None,
    edge_scale: float,
    region_face_count: int,
    selected_opening_clean_radius_authority: bool = False,
    two_opening_bore_frame_valid: bool = False,
    two_opening_bore_frame_depth: float = 0.0,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Recognize rebuildable POCKET hypothesis candidates from RegionData evidence.

    This detector intentionally stays in Recognition.  Region Select still owns
    only the neutral AOI.  The detector requires a recessed floor, side-wall
    evidence, and positive depth before emitting preview CandidateData.
    """

    base_diag: dict[str, object] = {
        "pocket_recognition_version": "v102_pocket_hypothesis_floor_wall_depth_with_child_bore_relationships",
        "pocket_semantic_order": "RegionData -> pocket floor/wall/depth evidence -> optional protected child-bore opening relationship -> POCKET hypothesis -> floor/side-wall ownership -> accepted POCKET CandidateData -> pocket recess-cup target",
        "pocket_rebuild_authorized": True,
        "pocket_rebuild_block_reason": "",
        "pocket_rebuild_policy": "delete/rebuild owned pocket side-wall plus floor as recessed cup; top opening remains open; child bore floor boundaries are protected relationship metadata; transitions remain excluded",
    }
    if len(valid_face_ids) < 8 or faces.ndim != 2 or face_centroids.ndim != 2:
        return (), {**base_diag, "pocket_status": "invalid_or_too_small_region"}

    resolver_resolved = bool(selected_opening_frame_resolver.get("resolved", False))
    rim_center = np.asarray(
        selected_opening_frame_resolver.get("center", region_frame.center),
        dtype=float,
    ).reshape(3)
    axis_value = selected_opening_frame_resolver.get("axis", region_frame.axis) if resolver_resolved else region_frame.axis
    pocket_axis = canonical_axis(axis_value)
    footprint_radius = _safe_float(
        selected_opening_frame_resolver.get("radius", region_frame.radius),
        _safe_float(region_frame.radius, 0.0),
    )
    if footprint_radius <= 1.0e-9:
        footprint_radius = max(_safe_float(region_frame.radius, 0.0), 1.0)

    frame = _PocketFrameV174O(center=rim_center, axis=pocket_axis, radius=footprint_radius)
    local = _pocket_local_arrays_v174o(frame=frame, face_ids=valid_face_ids, face_centroids=face_centroids, face_normals=face_normals)
    axial = local["axial"]
    radial = local["radial"]
    normal_axis_abs = local["normal_axis_abs"]
    finite = local["finite"]
    fid_to_local = {int(fid): int(i) for i, fid in enumerate(valid_face_ids)}
    raw_edge_scale = max(float(edge_scale), 1.0e-9)
    radial_p80 = _pocket_percentile_v174o(radial[finite], 80.0, footprint_radius) if len(radial) else footprint_radius
    footprint_limit = max(1.40 * float(footprint_radius), float(radial_p80) + 2.0 * raw_edge_scale, 4.0 * raw_edge_scale)
    # Pocket depth can be smaller than the local triangle size on coarse test
    # meshes.  Use edge scale as evidence tolerance, not as a hard multi-edge
    # minimum, otherwise a legitimate shallow/coarse pocket floor is rejected.
    min_depth = max(0.25 * raw_edge_scale, 0.04 * max(float(footprint_radius), 1.0), 0.05)

    # v134: X1 probe-ledger bridge.  When Region Select/Measurement already
    # produced a local coaxial probe corridor with explicit side-wall/floor face
    # evidence, Recognition may use that *measured evidence* as the POCKET role
    # proposal instead of rediscovering roles from the broad neutral RegionData
    # volume.  This keeps the raw 274-edge cloud and the 2213-face AOI out of
    # CandidateData ownership while preserving the accepted POCKET rebuild path.
    probe_ledger = dict(opening_probe_ledger or {}) if isinstance(opening_probe_ledger, Mapping) else x1_probe_ledger_from_region_diagnostics_v174m(region_diagnostics)
    if probe_ledger:
        probe_side_wall_face_ids = x1_probe_face_ids_v174m(probe_ledger, "sidewall_like", valid_face_set=set(valid_face_ids))
        probe_floor_face_ids = x1_probe_face_ids_v174m(probe_ledger, "floor_like", valid_face_set=set(valid_face_ids))
        probe_transition_face_ids = x1_probe_face_ids_v174m(probe_ledger, "transition_like", valid_face_set=set(valid_face_ids))
        probe_relationships = x1_probe_nested_mapping_v174m(probe_ledger, "relationships")
        probe_frame = x1_probe_nested_mapping_v174m(probe_ledger, "frame")
        probe_depth = _safe_float(probe_relationships.get("resolved_depth", 0.0), 0.0)
        probe_gate_ok, probe_gate_diag = x1_probe_supports_local_pocket_candidate_v174m(
            probe_ledger,
            floor_face_ids=probe_floor_face_ids,
            side_wall_face_ids=probe_side_wall_face_ids,
            min_depth=float(min_depth),
        )
        if probe_gate_ok:
            # v150: if the selected opening is a clean closed BORE-like rim and
            # an accepted two-opening frame exists, a very deep X1 pocket probe
            # is usually the same through/side wall column being misread as a
            # parent POCKET.  Keep that evidence diagnostic instead of promoting
            # a huge pocket ownership set around the already-bound BORE.
            clean_two_opening_bore_context = bool(
                selected_opening_clean_radius_authority
                and two_opening_bore_frame_valid
                and _safe_float(two_opening_bore_frame_depth, 0.0) > 1.0e-9
            )
            deep_probe_exceeds_bore_frame = bool(
                clean_two_opening_bore_context
                and probe_depth > max(
                    1.30 * _safe_float(two_opening_bore_frame_depth, 0.0),
                    _safe_float(two_opening_bore_frame_depth, 0.0) + 0.55 * max(float(footprint_radius), 1.0),
                )
            )
            if deep_probe_exceeds_bore_frame:
                demote_diag = {
                    **base_diag,
                    **probe_gate_diag,
                    "pocket_status": "probe_pocket_demoted_clean_two_opening_bore_context_v150",
                    "pocket_probe_demoted_v150": True,
                    "pocket_probe_demote_reason_v150": "clean_selected_bore_opening_owns_feature_probe_depth_exceeds_two_opening_frame",
                    "selected_opening_clean_radius_authority_v150": bool(selected_opening_clean_radius_authority),
                    "two_opening_bore_frame_valid_v150": bool(two_opening_bore_frame_valid),
                    "two_opening_bore_frame_depth_v150": float(_safe_float(two_opening_bore_frame_depth, 0.0)),
                    "probe_depth_v150": float(probe_depth),
                    "probe_sidewall_like_face_count": int(len(probe_side_wall_face_ids)),
                    "probe_floor_like_face_count": int(len(probe_floor_face_ids)),
                    "probe_ledger_used_for_candidate_data": False,
                    "recognition_rule": "v150_clean_bore_context_demotes_deep_x1_pocket_probe",
                }
                return (), demote_diag

            probe_floor_center = (
                np.mean(face_centroids[list(probe_floor_face_ids), :3], axis=0)
                if probe_floor_face_ids else (rim_center + pocket_axis * probe_depth)
            )
            floor_boundary_metadata = pocket_floor_compound_boundary_metadata_v174l(
                faces=faces,
                vertices=vertices,
                floor_face_ids=probe_floor_face_ids,
                center=rim_center,
                axis=pocket_axis,
            )
            protected_bore_opening_loops = tuple(
                dict(row)
                for row in tuple(floor_boundary_metadata.get("pocket_protected_floor_bore_opening_loops", ()) or ())
            )
            # v158: split child-BORE wall intrusion out of parent POCKET side-wall
            # ownership before CandidateData is built.  The protected bore loop is
            # a topology boundary; faces touching it are child-BORE relationship
            # evidence, not owned pocket side-wall.
            probe_side_wall_face_ids, probe_child_intrusion_face_ids_v158, probe_child_filter_diag_v158 = filter_pocket_child_bore_intrusion_faces_v174l(
                faces=faces,
                side_wall_face_ids=probe_side_wall_face_ids,
                floor_face_ids=probe_floor_face_ids,
                protected_bore_opening_loops=protected_bore_opening_loops,
            )
            probe_confidence = _safe_float(probe_ledger.get("confidence", 0.0), 0.0)
            confidence = float(max(0.05, min(0.94, 0.62 + 0.28 * min(probe_confidence, 1.0))))
            pocket_kind = "circular" if resolver_resolved and footprint_radius > 1.0e-9 and int(selected_opening_frame_resolver.get("edge_count", selected_opening_frame_resolver.get("expanded_edge_count", 0)) or 0) >= 8 else "freeform"
            heuristic_results = (
                make_heuristic_result(
                    "X1ProbeOpeningLedgerHeuristic",
                    input_count=int((x1_probe_nested_mapping_v174m(probe_ledger, "probe_counts")).get("broad_region_face_count", region_face_count) or region_face_count),
                    proposal_count=1,
                    accepted_count=1,
                    confidence=float(probe_confidence),
                    proposal_face_ids=tuple_ints(tuple(probe_side_wall_face_ids) + tuple(probe_floor_face_ids)),
                    diagnostics={**probe_gate_diag, "semantic_output": "local measured POCKET role proposal"},
                ),
                make_heuristic_result(
                    "ProbePocketFloorRoleOwnership",
                    input_count=int(len(probe_floor_face_ids)),
                    proposal_count=1,
                    accepted_count=int(len(probe_floor_face_ids)),
                    proposal_face_ids=probe_floor_face_ids,
                    diagnostics={"pocket_floor_face_count": int(len(probe_floor_face_ids)), "probe_resolved_depth": float(probe_depth)},
                ),
                make_heuristic_result(
                    "ProbePocketSideWallRoleOwnership",
                    input_count=int(len(probe_side_wall_face_ids)),
                    proposal_count=1,
                    accepted_count=int(len(probe_side_wall_face_ids)),
                    proposal_face_ids=probe_side_wall_face_ids,
                    diagnostics={"pocket_side_wall_face_count": int(len(probe_side_wall_face_ids)), "probe_relation_hint": str(probe_relationships.get("relation_hint", ""))},
                ),
                make_heuristic_result(
                    "EmitProbeAnchoredPocketCandidateData",
                    input_count=int(len(probe_floor_face_ids) + len(probe_side_wall_face_ids)),
                    proposal_count=1,
                    accepted_count=1,
                    proposal_face_ids=tuple_ints(tuple(probe_side_wall_face_ids) + tuple(probe_floor_face_ids)),
                    diagnostics={"semantic_output": "POCKET CandidateData from X1-style local probe ledger"},
                ),
            )
            candidate_diag = {
                **base_diag,
                **probe_gate_diag,
                "pocket_recognition_version": "v134_x1_probe_anchored_pocket_recess_role_ownership",
                "pocket_semantic_order": "RegionData diagnostics -> X1 local probe ledger -> pocket floor/side-wall role proposal -> POCKET CandidateData",
                "pocket_status": "candidate_emitted_from_x1_local_probe_ledger",
                "pocket_kind": str(pocket_kind),
                "pocket_axis": to_vector3(pocket_axis),
                "pocket_rim_center": to_vector3(rim_center),
                "pocket_floor_center": to_vector3(probe_floor_center),
                "pocket_footprint_radius": float(footprint_radius),
                "radius_min": _safe_float(selected_opening_frame_resolver.get("radius_min", probe_frame.get("radius", footprint_radius)), footprint_radius),
                "radius_nominal": _safe_float(selected_opening_frame_resolver.get("radius_nominal", selected_opening_frame_resolver.get("radius", probe_frame.get("radius", footprint_radius))), footprint_radius),
                "radius_max": _safe_float(selected_opening_frame_resolver.get("radius_max", probe_frame.get("radius", footprint_radius)), footprint_radius),
                "diameter_min": _safe_float(selected_opening_frame_resolver.get("diameter_min", 2.0 * _safe_float(selected_opening_frame_resolver.get("radius_min", footprint_radius), footprint_radius)), 2.0 * footprint_radius),
                "diameter_nominal": _safe_float(selected_opening_frame_resolver.get("diameter_nominal", 2.0 * footprint_radius), 2.0 * footprint_radius),
                "diameter_max": _safe_float(selected_opening_frame_resolver.get("diameter_max", 2.0 * _safe_float(selected_opening_frame_resolver.get("radius_max", footprint_radius), footprint_radius)), 2.0 * footprint_radius),
                "radial_spread": _safe_float(selected_opening_frame_resolver.get("radial_spread", 0.0), 0.0),
                "radial_spread_ratio": _safe_float(selected_opening_frame_resolver.get("radial_spread_ratio", 0.0), 0.0),
                "coarse_polygonal": bool(selected_opening_frame_resolver.get("coarse_polygonal", True)),
                "radial_envelope": dict(selected_opening_frame_resolver.get("radial_envelope", {}) or {}) if isinstance(selected_opening_frame_resolver.get("radial_envelope", {}), Mapping) else {},
                "pocket_depth": float(probe_depth),
                "pocket_depth_sign": float((_safe_float(probe_relationships.get("inward_axis_sign", 1.0), 1.0))),
                "pocket_floor_face_count": int(len(probe_floor_face_ids)),
                "pocket_floor_face_ids": probe_floor_face_ids,
                "pocket_side_wall_face_count": int(len(probe_side_wall_face_ids)),
                "pocket_side_wall_face_ids": probe_side_wall_face_ids,
                "pocket_transition_face_count": int(len(probe_transition_face_ids)),
                "pocket_transition_face_ids": probe_transition_face_ids,
                "pocket_side_wall_coverage": float(max(0.0, min(1.0, _safe_float(probe_relationships.get("sidewall_depth_span", probe_depth), 0.0) / max(probe_depth, 1.0e-9)))),
                "probe_ledger_used_for_candidate_data": True,
                "probe_ledger_contract_type": str(probe_ledger.get("contract_type", "")),
                "probe_ledger_semantic_stage": str(probe_ledger.get("semantic_stage", "")),
                "probe_ledger_authority": str(probe_ledger.get("authority", "measurement_evidence_only")),
                "probe_ledger_forbidden_authority": tuple(probe_ledger.get("forbidden_authority", ()) or ()),
                "probe_ledger_relationships": dict(probe_relationships),
                "probe_ledger_counts": x1_probe_nested_mapping_v174m(probe_ledger, "probe_counts"),
                "probe_ledger_contradictions": tuple(probe_ledger.get("contradictions", ()) or ()),
                **dict(floor_boundary_metadata),
                **dict(probe_child_filter_diag_v158),
                "pocket_child_bore_boundary_rebuild_safe_v157": bool(probe_child_filter_diag_v158.get("pocket_child_bore_boundary_rebuild_safe_v157", False)),
                "pocket_child_bore_boundary_rebuild_safe_v158": bool(probe_child_filter_diag_v158.get("pocket_child_bore_boundary_rebuild_safe_v158", False)),
                "pocket_child_bore_boundary_rebuild_safe_v159": bool(probe_child_filter_diag_v158.get("pocket_child_bore_boundary_rebuild_safe_v159", probe_child_filter_diag_v158.get("pocket_child_bore_boundary_rebuild_safe_v158", False))),
                "pocket_child_bore_intrusion_face_ids_v158": tuple_ints(probe_child_intrusion_face_ids_v158),
                "pocket_child_bore_intrusion_face_count_v158": int(len(tuple(probe_child_intrusion_face_ids_v158 or ()))),
                "pocket_embedded_bore_wall_evidence_face_count": int(len(tuple(probe_child_intrusion_face_ids_v158 or ()))),
                "pocket_embedded_bore_wall_evidence_face_ids": tuple_ints(probe_child_intrusion_face_ids_v158),
                "compound_pocket_bore_semantics": "POCKET remains parent candidate; probe child-bore loops, if any, are relationship metadata only",
                "pocket_heuristic_recipe": ("X1ProbeOpeningLedgerHeuristic", "ProbePocketFloorRoleOwnership", "ProbePocketSideWallRoleOwnership", "EmitProbeAnchoredPocketCandidateData"),
                "pocket_heuristic_results": heuristic_results,
                "pocket_heuristic_result_summaries": compact_heuristic_summary(heuristic_results),
                "transition_policy": "probe transition faces remain excluded from floor/wall rebuild ownership unless explicitly accepted later",
                "recognition_rule": "v134_x1_probe_local_coaxial_corridor_to_pocket_candidate",
                "feature_ownership_source": "x1_local_probe_ledger_floor_sidewall_role_proposal",
                "feature_ownership_split": "rebuild_face_ids_are_probe_floor_plus_probe_sidewall_faces_transition_excluded",
                "pocket_rebuild_enable_scope": "owned_probe_floor_plus_sidewall_recess_cup_protect_child_bore_openings",
            }
            candidate = pocket_candidate_contract_fields_v174l(
                candidate_id="component_engine.v134.x1_probe.pocket_recess.1",
                side_wall_face_ids=probe_side_wall_face_ids,
                floor_face_ids=probe_floor_face_ids,
                transition_face_ids=probe_transition_face_ids,
                protected_bore_opening_loops=protected_bore_opening_loops,
                embedded_bore_wall_evidence_face_ids=(),
                confidence=float(confidence),
                depth=float(probe_depth),
                axis=pocket_axis,
                rim_center=rim_center,
                floor_center=probe_floor_center,
                footprint_radius=float(footprint_radius),
                pocket_kind=str(pocket_kind),
                diagnostics=candidate_diag,
            )
            return (candidate,), candidate_diag

    floor_like_ids = tuple(
        int(fid)
        for fid, idx in fid_to_local.items()
        if bool(finite[idx])
        and float(normal_axis_abs[idx]) >= 0.82
        and abs(float(axial[idx])) >= float(min_depth)
        and float(radial[idx]) <= float(footprint_limit)
    )
    if not floor_like_ids:
        return (), {
            **base_diag,
            "pocket_status": "no_recessed_floor_evidence",
            "pocket_floor_candidate_count": 0,
            "pocket_min_depth": float(min_depth),
            "pocket_footprint_limit": float(footprint_limit),
        }

    floor_components = connected_face_components(faces, floor_like_ids)
    floor_rows: list[dict[str, object]] = []
    min_floor_faces = max(2, int(0.0015 * max(region_face_count, 1)))
    for comp in floor_components:
        comp_ids = tuple_ints(comp)
        idx = np.asarray([fid_to_local[int(fid)] for fid in comp_ids if int(fid) in fid_to_local], dtype=np.int64)
        if idx.size == 0:
            continue
        ax = axial[idx]
        rd = radial[idx]
        na = normal_axis_abs[idx]
        face_count = int(len(comp_ids))
        floor_level = float(np.median(ax)) if ax.size else 0.0
        depth_abs = abs(float(floor_level))
        radial_max = float(np.max(rd)) if rd.size else 0.0
        radial_median = float(np.median(rd)) if rd.size else 0.0
        normal_median = float(np.median(na)) if na.size else 0.0
        if face_count < min_floor_faces or depth_abs < min_depth:
            continue
        footprint_ratio = float(radial_max / max(footprint_limit, 1.0e-9))
        score = (
            1.25 * min(depth_abs / max(2.0 * min_depth, 1.0e-9), 1.0)
            + 0.90 * min(face_count / max(16.0, float(min_floor_faces)), 1.0)
            + 0.75 * max(0.0, min((normal_median - 0.80) / 0.20, 1.0))
            + 0.45 * max(0.0, 1.0 - min(footprint_ratio, 1.0))
        )
        floor_rows.append({
            "face_ids": comp_ids,
            "face_count": face_count,
            "floor_axial_level": float(floor_level),
            "depth_abs": float(depth_abs),
            "depth_sign": 1.0 if floor_level >= 0.0 else -1.0,
            "radial_max": float(radial_max),
            "radial_median": float(radial_median),
            "normal_axis_abs_median": float(normal_median),
            "score": float(score),
        })
    floor_rows.sort(key=lambda row: (-_safe_float(row.get("score", 0.0), 0.0), -int(row.get("face_count", 0) or 0)))
    if not floor_rows:
        return (), {
            **base_diag,
            "pocket_status": "floor_evidence_rejected",
            "pocket_floor_candidate_count": int(len(floor_components)),
            "pocket_min_floor_faces": int(min_floor_faces),
            "pocket_min_depth": float(min_depth),
        }

    floor = floor_rows[0]
    primary_floor_level = _safe_float(floor.get("floor_axial_level", 0.0), 0.0)
    primary_depth = abs(float(primary_floor_level))
    floor_level_tolerance = max(2.0 * raw_edge_scale, 0.08 * max(primary_depth, min_depth), 0.05)
    merged_floor_ids: set[int] = set()
    merged_floor_rows: list[dict[str, object]] = []
    for row in floor_rows:
        level = _safe_float(row.get("floor_axial_level", 0.0), 0.0)
        if (level >= 0.0) != (primary_floor_level >= 0.0):
            continue
        if abs(float(level - primary_floor_level)) > floor_level_tolerance:
            continue
        merged_floor_ids.update(int(fid) for fid in tuple_ints(row.get("face_ids", ())))
        merged_floor_rows.append(dict(row))
    floor_face_ids = tuple(sorted(merged_floor_ids)) or tuple_ints(floor.get("face_ids", ()))
    floor_face_set = {int(fid) for fid in floor_face_ids}
    floor_level = float(primary_floor_level)
    depth_sign = 1.0 if floor_level >= 0.0 else -1.0
    depth = abs(float(floor_level))
    if len(merged_floor_rows) > 1:
        floor = {
            **dict(floor),
            "merged_floor_component_count": int(len(merged_floor_rows)),
            "merged_floor_face_count": int(len(floor_face_ids)),
            "floor_merge_level_tolerance": float(floor_level_tolerance),
        }
    depth_lo = -0.08 * depth
    depth_hi = 1.10 * depth
    floor_union_idx = np.asarray([fid_to_local[int(fid)] for fid in floor_face_ids if int(fid) in fid_to_local], dtype=np.int64)
    floor_radial_max = float(np.max(radial[floor_union_idx])) if floor_union_idx.size else _safe_float(floor.get("radial_max", footprint_radius), footprint_radius)
    wall_radial_limit = max(footprint_limit, floor_radial_max + 2.0 * raw_edge_scale, float(footprint_radius) + 4.0 * raw_edge_scale)

    wall_proposal_ids = tuple(
        int(fid)
        for fid, idx in fid_to_local.items()
        if bool(finite[idx])
        and int(fid) not in floor_face_set
        and float(normal_axis_abs[idx]) <= 0.68
        and depth_lo <= float(axial[idx]) * depth_sign <= depth_hi
        and float(radial[idx]) <= float(wall_radial_limit)
    )
    wall_components = connected_face_components(faces, wall_proposal_ids) if wall_proposal_ids else ()
    accepted_wall_ids: set[int] = set()
    embedded_bore_wall_ids: set[int] = set()
    wall_rows: list[dict[str, object]] = []
    min_wall_faces = max(2, int(0.001 * max(region_face_count, 1)))
    for comp in wall_components:
        comp_ids = tuple_ints(comp)
        idx = np.asarray([fid_to_local[int(fid)] for fid in comp_ids if int(fid) in fid_to_local], dtype=np.int64)
        if idx.size == 0:
            continue
        comp_set = {int(fid) for fid in comp_ids}
        ax_depth = axial[idx] * depth_sign
        na = normal_axis_abs[idx]
        face_count = int(len(comp_ids))
        ax_min = float(np.min(ax_depth)) if ax_depth.size else 0.0
        ax_max = float(np.max(ax_depth)) if ax_depth.size else 0.0
        coverage = max(0.0, min(ax_max, depth) - max(ax_min, 0.0)) / max(depth, 1.0e-9)
        touches_floor = any(int(nb) in floor_face_set for fid in comp_set for nb in adjacency.get(int(fid), ()))
        touches_seed = any(int(nb) in seed_face_set for fid in comp_set for nb in adjacency.get(int(fid), ())) or bool(comp_set & seed_face_set)
        comp_radial_median = float(np.median(radial[idx])) if idx.size else 0.0
        comp_radial_max = float(np.max(radial[idx])) if idx.size else 0.0
        # Compound POCKET+BORE rule: a wall-like component that touches the
        # pocket floor but does not touch the selected pocket rim/opening is
        # child-BORE wall evidence, not pocket side-wall ownership.  Keep this
        # as relationship/protected-boundary metadata; do not promote a fake
        # pocket_with_bore family and do not swallow the BORE wall into POCKET.
        embedded_bore_like = bool(
            touches_floor
            and not touches_seed
            and face_count >= min_wall_faces
            and comp_radial_median <= 0.92 * max(float(footprint_radius), 1.0e-9)
        )
        accepted = bool(
            face_count >= min_wall_faces
            and coverage >= 0.18
            and not embedded_bore_like
            and (touches_seed or coverage >= 0.38 or comp_radial_median >= 0.72 * max(float(footprint_radius), 1.0e-9))
        )
        score = (
            1.20 * min(coverage / 0.70, 1.0)
            + 0.65 * min(face_count / max(12.0, float(min_wall_faces)), 1.0)
            + (0.55 if touches_floor else 0.0)
            + (0.35 if touches_seed else 0.0)
            + 0.35 * max(0.0, 1.0 - min(float(np.median(na)) / 0.68 if na.size else 1.0, 1.0))
        )
        wall_rows.append({
            "face_ids": comp_ids,
            "face_count": face_count,
            "accepted": bool(accepted),
            "axial_coverage": float(coverage),
            "touches_floor": bool(touches_floor),
            "touches_seed": bool(touches_seed),
            "embedded_bore_like_relationship_evidence": bool(embedded_bore_like),
            "radial_median": float(comp_radial_median),
            "radial_max": float(comp_radial_max),
            "normal_axis_abs_median": float(np.median(na)) if na.size else 1.0,
            "score": float(score),
        })
        if embedded_bore_like:
            embedded_bore_wall_ids.update(int(fid) for fid in comp_ids)
        if accepted:
            accepted_wall_ids.update(int(fid) for fid in comp_ids)

    if not accepted_wall_ids:
        return (), {
            **base_diag,
            "pocket_status": "no_side_wall_evidence",
            "pocket_floor_face_count": int(len(floor_face_ids)),
            "pocket_side_wall_candidate_component_count": int(len(wall_components)),
            "pocket_side_wall_component_reports": tuple(wall_rows[:12]),
            "pocket_depth": float(depth),
        }

    side_wall_face_ids = tuple(sorted(accepted_wall_ids))
    owned_set = set(side_wall_face_ids) | floor_face_set
    neighbor_ids = {
        int(nb)
        for fid in owned_set
        for nb in adjacency.get(int(fid), ())
        if int(nb) in fid_to_local and int(nb) not in owned_set
    }
    transition_ids: list[int] = []
    for fid in sorted(neighbor_ids):
        idx = fid_to_local[int(fid)]
        local_depth = float(axial[idx]) * depth_sign
        if -0.12 * depth <= local_depth <= 1.12 * depth and float(radial[idx]) <= wall_radial_limit + 2.0 * raw_edge_scale:
            na = float(normal_axis_abs[idx])
            if 0.30 < na < 0.84:
                transition_ids.append(int(fid))
    transition_face_ids = tuple_ints(transition_ids)
    floor_boundary_metadata = pocket_floor_compound_boundary_metadata_v174l(
        faces=faces,
        vertices=vertices,
        floor_face_ids=floor_face_ids,
        center=rim_center,
        axis=pocket_axis,
    )
    protected_bore_opening_loops = tuple(dict(row) for row in tuple(floor_boundary_metadata.get("pocket_protected_floor_bore_opening_loops", ()) or ()))
    embedded_bore_wall_evidence_face_ids = tuple(sorted(embedded_bore_wall_ids))
    # v158: resolve parent-pocket ownership instead of merely rejecting unsafe
    # child-BORE cases.  Child bore wall faces that touch the protected floor
    # loop are split out as relationship evidence before CandidateData/rebuild.
    side_wall_face_ids, child_intrusion_face_ids_v158, child_filter_diag_v158 = filter_pocket_child_bore_intrusion_faces_v174l(
        faces=faces,
        side_wall_face_ids=side_wall_face_ids,
        floor_face_ids=floor_face_ids,
        protected_bore_opening_loops=protected_bore_opening_loops,
    )
    embedded_bore_wall_evidence_face_ids = tuple_ints(tuple(embedded_bore_wall_evidence_face_ids) + tuple(child_intrusion_face_ids_v158))
    if bool(child_filter_diag_v158.get("v158_pocket_child_boundary_filter_used", False)):
        owned_set = set(side_wall_face_ids) | floor_face_set
        neighbor_ids = {
            int(nb)
            for fid in owned_set
            for nb in adjacency.get(int(fid), ())
            if int(nb) in fid_to_local and int(nb) not in owned_set
        }
        transition_ids = []
        for fid in sorted(neighbor_ids):
            idx = fid_to_local[int(fid)]
            local_depth = float(axial[idx]) * depth_sign
            if -0.12 * depth <= local_depth <= 1.12 * depth and float(radial[idx]) <= wall_radial_limit + 2.0 * raw_edge_scale:
                na = float(normal_axis_abs[idx])
                if 0.30 < na < 0.84:
                    transition_ids.append(int(fid))
        transition_face_ids = tuple_ints(transition_ids)

    side_idx = np.asarray([fid_to_local[int(fid)] for fid in side_wall_face_ids if int(fid) in fid_to_local], dtype=np.int64)
    floor_idx = np.asarray([fid_to_local[int(fid)] for fid in floor_face_ids if int(fid) in fid_to_local], dtype=np.int64)
    floor_center = np.mean(face_centroids[list(floor_face_ids), :3], axis=0) if floor_face_ids else (rim_center + pocket_axis * floor_level)
    side_wall_coverage = 0.0
    if side_idx.size:
        side_depth = axial[side_idx] * depth_sign
        side_wall_coverage = max(0.0, min(float(np.max(side_depth)), depth) - max(float(np.min(side_depth)), 0.0)) / max(depth, 1.0e-9)
    floor_conf = min(1.0, len(floor_face_ids) / max(10.0, float(min_floor_faces)))
    wall_conf = min(1.0, len(side_wall_face_ids) / max(10.0, float(min_wall_faces)))
    depth_conf = min(1.0, depth / max(2.0 * min_depth, 1.0e-9))
    confidence = float(max(0.05, min(0.92, 0.20 + 0.22 * floor_conf + 0.24 * wall_conf + 0.20 * depth_conf + 0.16 * side_wall_coverage)))
    pocket_kind = "circular" if resolver_resolved and footprint_radius > 1.0e-9 and int(selected_opening_frame_resolver.get("edge_count", selected_opening_frame_resolver.get("expanded_edge_count", 0)) or 0) >= 8 else "freeform"

    heuristic_results = (
        make_heuristic_result("SelectedBoundaryToPocketRimHeuristic", input_count=int(region_diagnostics.get("selected_edge_count", 0) or 0), proposal_count=1, accepted_count=1 if resolver_resolved else 0, diagnostics={"resolved": bool(resolver_resolved), "footprint_radius": float(footprint_radius)}),
        make_heuristic_result("MeasurePocketOpening", input_count=int(region_face_count), proposal_count=1, accepted_count=1, confidence=float(confidence), diagnostics={"pocket_axis": to_vector3(pocket_axis), "pocket_footprint_radius": float(footprint_radius)}),
        make_heuristic_result("PocketFloorSurfaceSearchHeuristic", input_count=int(region_face_count), proposal_count=int(len(floor_components)), accepted_count=1, proposal_face_ids=floor_face_ids, diagnostics={"pocket_floor_face_count": int(len(floor_face_ids)), "pocket_floor_axial_level": float(floor_level)}),
        make_heuristic_result("PocketSideWallSearchHeuristic", input_count=int(region_face_count), proposal_count=int(len(wall_components)), accepted_count=int(len(side_wall_face_ids)), proposal_face_ids=side_wall_face_ids, diagnostics={"pocket_side_wall_face_count": int(len(side_wall_face_ids)), "pocket_side_wall_coverage": float(side_wall_coverage)}),
        make_heuristic_result("PocketDepthResolver", input_count=1, proposal_count=1, accepted_count=1, confidence=float(depth_conf), diagnostics={"pocket_depth": float(depth), "pocket_depth_sign": float(depth_sign), "pocket_depth_valid": True}),
        make_heuristic_result("PocketTransitionSeparation", input_count=int(len(owned_set)), proposal_count=int(len(transition_face_ids)), accepted_count=int(len(transition_face_ids)), proposal_face_ids=transition_face_ids, diagnostics={"pocket_transition_face_count": int(len(transition_face_ids))}),
        make_heuristic_result("PocketRoleOwnership", input_count=int(len(floor_face_ids) + len(side_wall_face_ids)), proposal_count=2, accepted_count=int(len(floor_face_ids) + len(side_wall_face_ids)), proposal_face_ids=tuple_ints(tuple(floor_face_ids) + tuple(side_wall_face_ids)), diagnostics={"pocket_floor_face_count": int(len(floor_face_ids)), "pocket_side_wall_face_count": int(len(side_wall_face_ids))}),
        make_heuristic_result("PocketChildBoreBoundaryRelationship", input_count=int(len(floor_face_ids)), proposal_count=int(len(protected_bore_opening_loops)), accepted_count=int(len(protected_bore_opening_loops)), proposal_face_ids=embedded_bore_wall_evidence_face_ids, diagnostics={"pocket_protected_floor_bore_opening_count": int(len(protected_bore_opening_loops)), "pocket_embedded_bore_wall_evidence_face_count": int(len(embedded_bore_wall_evidence_face_ids)), "semantic_output": "relationship metadata only; no pocket_with_bore family"}),
        make_heuristic_result("EmitPocketCandidateData", input_count=int(len(floor_face_ids) + len(side_wall_face_ids)), proposal_count=1, accepted_count=1, proposal_face_ids=tuple_ints(tuple(floor_face_ids) + tuple(side_wall_face_ids)), diagnostics={"semantic_output": "POCKET CandidateData recess-cup rebuild", "compound_relationship_count": int(len(protected_bore_opening_loops))}),
    )

    candidate_diag = {
        **base_diag,
        "pocket_status": "candidate_emitted_rebuildable_pocket_recess_cup_trial",
        "pocket_kind": str(pocket_kind),
        "pocket_axis": to_vector3(pocket_axis),
        "pocket_rim_center": to_vector3(rim_center),
        "pocket_floor_center": to_vector3(floor_center),
        "pocket_footprint_radius": float(footprint_radius),
        "radius_min": _safe_float(selected_opening_frame_resolver.get("radius_min", footprint_radius), footprint_radius),
        "radius_nominal": _safe_float(selected_opening_frame_resolver.get("radius_nominal", selected_opening_frame_resolver.get("radius", footprint_radius)), footprint_radius),
        "radius_max": _safe_float(selected_opening_frame_resolver.get("radius_max", footprint_radius), footprint_radius),
        "diameter_min": _safe_float(selected_opening_frame_resolver.get("diameter_min", 2.0 * _safe_float(selected_opening_frame_resolver.get("radius_min", footprint_radius), footprint_radius)), 2.0 * footprint_radius),
        "diameter_nominal": _safe_float(selected_opening_frame_resolver.get("diameter_nominal", 2.0 * footprint_radius), 2.0 * footprint_radius),
        "diameter_max": _safe_float(selected_opening_frame_resolver.get("diameter_max", 2.0 * _safe_float(selected_opening_frame_resolver.get("radius_max", footprint_radius), footprint_radius)), 2.0 * footprint_radius),
        "radial_spread": _safe_float(selected_opening_frame_resolver.get("radial_spread", 0.0), 0.0),
        "radial_spread_ratio": _safe_float(selected_opening_frame_resolver.get("radial_spread_ratio", 0.0), 0.0),
        "coarse_polygonal": bool(selected_opening_frame_resolver.get("coarse_polygonal", False)),
        "radial_envelope": dict(selected_opening_frame_resolver.get("radial_envelope", {}) or {}) if isinstance(selected_opening_frame_resolver.get("radial_envelope", {}), Mapping) else {},
        "pocket_footprint_limit": float(footprint_limit),
        "pocket_min_depth": float(min_depth),
        "pocket_depth": float(depth),
        "pocket_depth_sign": float(depth_sign),
        "pocket_floor_candidate_count": int(len(floor_components)),
        "pocket_floor_face_count": int(len(floor_face_ids)),
        "pocket_floor_best_report": dict(floor),
        "pocket_side_wall_candidate_component_count": int(len(wall_components)),
        "pocket_side_wall_face_count": int(len(side_wall_face_ids)),
        "pocket_side_wall_coverage": float(side_wall_coverage),
        "pocket_side_wall_component_reports": tuple(wall_rows[:12]),
        "pocket_transition_face_count": int(len(transition_face_ids)),
        **dict(floor_boundary_metadata),
        **dict(child_filter_diag_v158),
        "pocket_child_bore_boundary_rebuild_safe_v157": bool(child_filter_diag_v158.get("pocket_child_bore_boundary_rebuild_safe_v157", False)),
        "pocket_child_bore_boundary_rebuild_safe_v158": bool(child_filter_diag_v158.get("pocket_child_bore_boundary_rebuild_safe_v158", False)),
        "pocket_child_bore_boundary_rebuild_safe_v159": bool(child_filter_diag_v158.get("pocket_child_bore_boundary_rebuild_safe_v159", child_filter_diag_v158.get("pocket_child_bore_boundary_rebuild_safe_v158", False))),
        "pocket_child_bore_intrusion_face_ids_v158": tuple_ints(child_intrusion_face_ids_v158),
        "pocket_child_bore_intrusion_face_count_v158": int(len(tuple(child_intrusion_face_ids_v158 or ()))),
        "pocket_embedded_bore_wall_evidence_face_count": int(len(embedded_bore_wall_evidence_face_ids)),
        "pocket_embedded_bore_wall_evidence_face_ids": embedded_bore_wall_evidence_face_ids,
        "compound_pocket_bore_semantics": "POCKET remains parent candidate; BORE remains separate child candidate; floor bore loop is protected relationship metadata",
        "pocket_heuristic_recipe": POCKET_HEURISTIC_RECIPE,
        "pocket_heuristic_results": heuristic_results,
        "pocket_heuristic_result_summaries": compact_heuristic_summary(heuristic_results),
        "transition_policy": "exclude_chamfer_or_fillet_like_faces_from_floor_wall_ownership",
    }
    candidate = pocket_candidate_contract_fields_v174l(
        candidate_id="component_engine.v102.pocket.recess_cup.1",
        side_wall_face_ids=side_wall_face_ids,
        floor_face_ids=floor_face_ids,
        transition_face_ids=transition_face_ids,
        protected_bore_opening_loops=protected_bore_opening_loops,
        embedded_bore_wall_evidence_face_ids=embedded_bore_wall_evidence_face_ids,
        confidence=float(confidence),
        depth=float(depth),
        axis=pocket_axis,
        rim_center=rim_center,
        floor_center=floor_center,
        footprint_radius=float(footprint_radius),
        pocket_kind=str(pocket_kind),
        diagnostics=candidate_diag,
    )
    return (candidate,), candidate_diag

