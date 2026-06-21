"""Slim active Bore recognition entry point.

This is the active recognition entry point.  Removed recognition prototypes are
not imported or referenced by the active package.

Active flow:
    region_select.py -> RegionData
    recognition.py -> RegionData-to-CandidateData bridge
    recognition_component_engine.py -> physical CandidateData classifier
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .geometry import (
    canonical_axis,
    describe_boundary_loop_stack_geometry,
    measure_feature_patch_geometry,
    to_vector3,
)
from .measure import (
    BoreOpeningMeasurement,
    measure_bore_opening_candidates,
    measure_bore_opening_component_candidates,
    measure_two_opening_bore_frame,
)
from .recognition_component_engine import (
    component_engine_feature_candidates,
    recognition_result_dict_from_component_features,
)
from .types import EvidenceKind, FeatureFamily, RecognitionStage, tuple_edges, tuple_ints

ACTIVE_CANDIDATE_AUTHORITY = "surface_component_classifier_v93_heuristic_role_scale_clamp_full_depth_wall_ownership"
REGION_SELECT_FEATURE_AUTHORITY = False
ASSEMBLY_CLASSIFICATION_POLICY = "do_not_classify_assemblies_classify_physical_surface_objects"


def _mesh_faces_and_vertices(mesh: object) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(getattr(mesh, "vertices"), dtype=float)[:, :3]
    faces = np.asarray(getattr(mesh, "faces"), dtype=np.int64)[:, :3]
    return vertices, faces


def _face_normals(vertices: np.ndarray, faces: np.ndarray, mesh: object) -> np.ndarray:
    try:
        normals = np.asarray(getattr(mesh, "face_normals"), dtype=float)
        if normals.shape == (len(faces), 3):
            return normals
    except Exception:
        pass
    tri = vertices[faces[:, :3]]
    raw = np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :])
    lengths = np.linalg.norm(raw, axis=1)
    out = np.zeros_like(raw)
    valid = np.isfinite(lengths) & (lengths > 1.0e-12)
    out[valid] = raw[valid] / lengths[valid].reshape(-1, 1)
    return out



def _recognition_anchor_allows_candidates(region_diag: Mapping[str, object]) -> bool:
    """Honor old anchor rejection in recognition, not in BoreActions."""

    anchor = region_diag.get("recognition_anchor_handoff")
    if not isinstance(anchor, Mapping):
        ctx = region_diag.get("recognition_context")
        if isinstance(ctx, Mapping):
            maybe = ctx.get("anchor_handoff")
            if isinstance(maybe, Mapping):
                anchor = maybe
    if isinstance(anchor, Mapping) and bool(anchor.get("used", False)):
        return bool(anchor.get("preferred_anchor_usable", anchor.get("has_usable_anchor", False)))
    return True


def _candidate_data_face_ids(raw: Mapping[str, object]) -> tuple[int, ...]:
    """Return CandidateData-owned faces only.

    v33 semantic boundary: target/delete-patch fields are downstream meaning and
    must not be used to reconstruct CandidateData ownership.
    """

    for key in ("semantic_face_ids", "face_ids", "preview_face_ids", "display_face_ids"):
        ids = tuple_ints(raw.get(key, ()))
        if ids:
            return ids
    return ()


def _normalise_recognition_feature(raw: Mapping[str, object], *, source: str, index: int) -> dict[str, object] | None:
    """Convert compatibility rows into canonical CandidateData dictionaries."""

    item = dict(raw)
    face_ids = _candidate_data_face_ids(item)
    if not face_ids:
        return None
    entity_type = str(item.get("entity_type", item.get("feature_kind", item.get("object_type", "recognized_feature"))) or "recognized_feature").strip().lower() or "recognized_feature"
    candidate_id = str(item.get("candidate_id", item.get("feature_id", "")) or "").strip()
    if not candidate_id:
        candidate_id = f"recognition_bridge.{entity_type}.{index}"
    item.setdefault("candidate_id", candidate_id)
    item.setdefault("feature_id", candidate_id)
    item["entity_type"] = entity_type
    item["feature_kind"] = str(item.get("feature_kind", entity_type) or entity_type)
    item.setdefault("candidate_scope", "candidate_data")
    item.setdefault("candidate_authority", ACTIVE_CANDIDATE_AUTHORITY)
    item.setdefault("active_candidate_authority", ACTIVE_CANDIDATE_AUTHORITY)
    item.setdefault("feature_ownership_source", source)
    item.setdefault("display_name", entity_type.upper().replace("_", " "))
    item.setdefault("role", "diagnostic_preview_only")
    item.setdefault("status", "diagnostic_recognition_bridge_candidate")
    item.setdefault("promotion_state", "diagnostic_only")
    item.setdefault("feature_family", FeatureFamily.UNKNOWN.value)
    item.setdefault("recognition_stage", RecognitionStage.DIAGNOSTIC_ONLY.value)
    item.setdefault("evidence_kinds", ())
    stage = str(item.get("recognition_stage", "") or "").strip().lower()
    family = str(item.get("feature_family", "") or "").strip().lower()
    stage_allows_action = bool(stage == RecognitionStage.ACCEPTED_CANDIDATE.value and family in {FeatureFamily.BORE.value, FeatureFamily.CHAMFER_FORM.value, FeatureFamily.POCKET.value, FeatureFamily.CIRCULAR_POCKET.value})
    item["candidate_action_enabled"] = bool(stage_allows_action and item.get("candidate_action_enabled", item.get("rebuild_authorized", False)))
    item["rebuild_authorized"] = bool(item["candidate_action_enabled"])
    item.setdefault("rebuild_gate", "candidate_data_bridge_diagnostic_only")
    item["face_ids"] = face_ids
    item.setdefault("semantic_face_ids", face_ids)
    item.setdefault("preview_face_ids", face_ids)
    explicit_rebuild_ids = tuple_ints(item.get("rebuild_face_ids", ()))
    item["rebuild_face_ids"] = explicit_rebuild_ids if bool(stage_allows_action) else ()
    item["face_count"] = int(item.get("face_count", 0) or len(face_ids))
    item.setdefault("recognition_rule", source)
    return item


def _fallback_features_from_region_diagnostics(region_diag: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    """Return no candidates from diagnostic ledgers.

    v33 semantic boundary: diagnostics, feature-layer analysis, macro-family
    ledgers, and old radial-layer rows are evidence/support only.  They must not
    be normalized back into CandidateData when the active component engine emits
    no owned candidates.
    """

    return ()



def _opening_edge_ids_from_region_diagnostics(region_diag: Mapping[str, object], fallback: tuple[int, ...]) -> tuple[int, ...]:
    """Return normalized/measured rim edge IDs from Region Select diagnostics."""

    candidates: list[object] = []
    for key in ("opening_rim_edge_ids", "normalized_edge_ids", "normalized_rim_edge_ids", "primary_anchor_edge_ids"):
        if key in region_diag:
            candidates.append(region_diag.get(key))
    cutout = region_diag.get("cutout", {})
    if isinstance(cutout, Mapping):
        for key in ("opening_rim_edge_ids", "normalized_edge_ids", "normalized_rim_edge_ids", "primary_anchor_edge_ids"):
            if key in cutout:
                candidates.append(cutout.get(key))
    frame_measurement = region_diag.get("frame_measurement", {})
    if isinstance(frame_measurement, Mapping):
        for key in ("opening_rim_edge_ids", "measured_edge_ids", "edge_ids"):
            if key in frame_measurement:
                candidates.append(frame_measurement.get(key))
    for value in candidates:
        ids = tuple_ints(value or ())
        if ids:
            return ids
    return tuple_ints(fallback)


def _opening_measurement_row(value: object, *, source: str, rank: int, input_edge_count: int) -> dict[str, object]:
    """Serialize a BoreOpeningMeasurement as non-mutating audit data."""

    diag = dict(getattr(value, "diagnostics", {}) or {})
    candidate_source = diag.get("candidate_source", {})
    if not isinstance(candidate_source, Mapping):
        candidate_source = {}
    return {
        "source": str(source),
        "rank": int(rank),
        "input_edge_count": int(input_edge_count),
        "edge_count": int(getattr(value, "edge_count", 0) or 0),
        "edge_ids": tuple_ints(getattr(value, "edge_ids", ())),
        "vertex_ids": tuple_ints(getattr(value, "vertex_ids", ())),
        "vertex_count": int(getattr(value, "vertex_count", 0) or 0),
        "center": to_vector3(getattr(value, "center", (0.0, 0.0, 0.0))),
        "axis": to_vector3(getattr(value, "axis", (0.0, 0.0, 1.0))),
        "radius": float(getattr(value, "radius", 0.0) or 0.0),
        "diameter": float(getattr(value, "diameter", 0.0) or 0.0),
        "closed": bool(getattr(value, "closed", False)),
        "near_closed": bool(getattr(value, "near_closed", False)),
        "endpoint_gap_ratio": float(getattr(value, "endpoint_gap_ratio", 0.0) or 0.0),
        "branch_vertex_count": int(getattr(value, "branch_vertex_count", 0) or 0),
        "open_endpoint_count": int(getattr(value, "open_endpoint_count", 0) or 0),
        "component_count": int(getattr(value, "component_count", 0) or 0),
        "plane_rel_rms": float(getattr(value, "plane_rel_rms", 0.0) or 0.0),
        "radius_rel_rms": float(getattr(value, "radius_rel_rms", 0.0) or 0.0),
        "radius_mad": float(getattr(value, "radius_mad", 0.0) or 0.0),
        "circularity": float(getattr(value, "circularity", 0.0) or 0.0),
        "confidence": float(getattr(value, "confidence", 0.0) or 0.0),
        "component_strategy": str(diag.get("component_strategy", "") or ""),
        "largest_component_edge_count": int(diag.get("largest_component_edge_count", 0) or 0),
        "largest_component_vertex_count": int(diag.get("largest_component_vertex_count", 0) or 0),
        "largest_component_fraction": float(diag.get("largest_component_fraction", 0.0) or 0.0),
        "measured_component_edge_count": int(diag.get("measured_component_edge_count", getattr(value, "edge_count", 0)) or 0),
        "dropped_component_edge_count": int(diag.get("dropped_component_edge_count", 0) or 0),
        "component_sizes": tuple(int(v) for v in tuple(diag.get("component_sizes", ()) or ())[:12]),
        "fragmented_ring_refinement": dict(diag.get("fragmented_ring_refinement", {}) or {}) if isinstance(diag.get("fragmented_ring_refinement", {}), Mapping) else {},
        # v1.3.7: keep primary raw selected-edge component distinct from expanded
        # measurement support. Recognition may use the primary seed island for
        # locality, while expanded edges remain evidence only.
        "seed_component_edge_ids": tuple_ints(candidate_source.get("seed_component_edge_ids", ())),
        "expanded_inlier_edge_ids": tuple_ints(candidate_source.get("expanded_inlier_edge_ids", ())),
        "seed_component_edge_count": int(candidate_source.get("seed_component_edge_count", 0) or 0),
        "expanded_inlier_edge_count": int(candidate_source.get("expanded_inlier_edge_count", getattr(value, "edge_count", 0)) or 0),
        "local_distance_limit": float(candidate_source.get("local_distance_limit", 0.0) or 0.0),
        "local_distance_rejected_edge_count": int(candidate_source.get("local_distance_rejected_edge_count", 0) or 0),
    }


def _opening_axis_dot(a: object, b: object) -> float:
    try:
        av = canonical_axis(a)
        bv = canonical_axis(b)
        return abs(float(np.dot(av, bv)))
    except Exception:
        return 0.0


def _opening_centerline_distance(row0: Mapping[str, object], row1: Mapping[str, object]) -> float:
    """Return distance between two near-parallel opening centerlines."""

    try:
        axis = canonical_axis(row0.get("axis", (0.0, 0.0, 1.0)))
        p0 = np.asarray(row0.get("center", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
        p1 = np.asarray(row1.get("center", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
        delta = p1 - p0
        return float(np.linalg.norm(delta - axis * float(np.dot(delta, axis))))
    except Exception:
        return 999999.0


def _selected_opening_measurement_audit(mesh: object, selected_edge_ids: tuple[int, ...], normalized_edge_ids: tuple[int, ...]) -> dict[str, object]:
    """Measure raw and normalized opening evidence without creating candidates.

    This is a diagnostic/audit object only. It exists to answer whether the
    selected damaged bore mouth is being measured before Recognition attempts
    surface-role ownership. It must not create CandidateData, DeletePatchProposal,
    or rebuild authorization.
    """

    raw_ids = tuple_ints(selected_edge_ids)
    rim_ids = tuple_ints(normalized_edge_ids)
    out: dict[str, object] = {
        "audit_version": "v1.3.5",
        "semantic_stage": "selected_edge_ids_to_measured_opening_evidence_audit_only",
        "not_candidate_data": True,
        "not_rebuild_authority": True,
        "raw_selected_edge_count": int(len(raw_ids)),
        "normalized_rim_edge_count": int(len(rim_ids)),
        "normalized_to_raw_edge_ratio": float(len(rim_ids)) / max(float(len(raw_ids)), 1.0),
    }

    def measure_source(label: str, ids: tuple[int, ...]) -> tuple[tuple[dict[str, object], ...], str]:
        if not ids:
            return (), "no_edge_ids"
        try:
            rows = tuple(
                _opening_measurement_row(cand, source=label, rank=i, input_edge_count=len(ids))
                for i, cand in enumerate(measure_bore_opening_candidates(mesh, ids, max_candidates=8), start=1)
            )
            return rows, "measured" if rows else "no_candidates"
        except Exception as exc:
            return (), f"measurement_failed: {exc}"

    raw_rows, raw_status = measure_source("raw_selected_edges", raw_ids)
    rim_rows, rim_status = measure_source("normalized_rim_edges", rim_ids)

    try:
        component_rows = tuple(
            _opening_measurement_row(cand, source="raw_component_candidates", rank=i, input_edge_count=len(raw_ids))
            for i, cand in enumerate(measure_bore_opening_component_candidates(mesh, raw_ids, max_candidates=12), start=1)
        )
        component_status = "measured" if component_rows else "no_candidates"
    except Exception as exc:
        component_rows = ()
        component_status = f"measurement_failed: {exc}"

    out["raw_measurement_status"] = raw_status
    out["normalized_measurement_status"] = rim_status
    out["raw_component_measurement_status"] = component_status
    out["raw_candidate_count"] = int(len(raw_rows))
    out["normalized_candidate_count"] = int(len(rim_rows))
    out["raw_component_candidate_count"] = int(len(component_rows))
    out["raw_candidates"] = raw_rows
    out["normalized_candidates"] = rim_rows
    out["raw_component_candidates"] = component_rows

    raw_best = raw_rows[0] if raw_rows else {}
    rim_best = rim_rows[0] if rim_rows else {}
    component_best = component_rows[0] if component_rows else {}
    if raw_best:
        out["raw_best_radius"] = float(raw_best.get("radius", 0.0) or 0.0)
        out["raw_best_confidence"] = float(raw_best.get("confidence", 0.0) or 0.0)
        out["raw_best_edge_count"] = int(raw_best.get("edge_count", 0) or 0)
        out["raw_best_component_strategy"] = str(raw_best.get("component_strategy", "") or "")
        out["raw_best_component_count"] = int(raw_best.get("component_count", 0) or 0)
        out["raw_best_largest_component_fraction"] = float(raw_best.get("largest_component_fraction", 0.0) or 0.0)
    if rim_best:
        out["normalized_best_radius"] = float(rim_best.get("radius", 0.0) or 0.0)
        out["normalized_best_confidence"] = float(rim_best.get("confidence", 0.0) or 0.0)
        out["normalized_best_edge_count"] = int(rim_best.get("edge_count", 0) or 0)
        out["normalized_best_component_strategy"] = str(rim_best.get("component_strategy", "") or "")
        out["normalized_best_component_count"] = int(rim_best.get("component_count", 0) or 0)
        out["normalized_best_largest_component_fraction"] = float(rim_best.get("largest_component_fraction", 0.0) or 0.0)
    if component_best:
        out["raw_component_best_radius"] = float(component_best.get("radius", 0.0) or 0.0)
        out["raw_component_best_confidence"] = float(component_best.get("confidence", 0.0) or 0.0)
        out["raw_component_best_edge_count"] = int(component_best.get("edge_count", 0) or 0)
        out["raw_component_best_component_strategy"] = str(component_best.get("component_strategy", "") or "")
        out["raw_component_best_component_count"] = int(component_best.get("component_count", 0) or 0)
        out["raw_component_best_largest_component_fraction"] = float(component_best.get("largest_component_fraction", 0.0) or 0.0)

    if raw_best and rim_best:
        raw_radius = float(raw_best.get("radius", 0.0) or 0.0)
        rim_radius = float(rim_best.get("radius", 0.0) or 0.0)
        radius_ref = max(raw_radius, rim_radius, 1.0e-9)
        radius_delta_rel = abs(raw_radius - rim_radius) / radius_ref
        axis_dot = _opening_axis_dot(raw_best.get("axis"), rim_best.get("axis"))
        centerline_distance = _opening_centerline_distance(raw_best, rim_best)
        out["raw_vs_normalized_radius_delta_rel"] = float(radius_delta_rel)
        out["raw_vs_normalized_axis_abs_dot"] = float(axis_dot)
        out["raw_vs_normalized_centerline_distance"] = float(centerline_distance)
        out["raw_vs_normalized_measurement_agree"] = bool(radius_delta_rel <= 0.16 and axis_dot >= 0.965 and centerline_distance <= max(0.25 * radius_ref, 1.0))
    else:
        out["raw_vs_normalized_measurement_agree"] = False

    collapsed = bool(len(raw_ids) >= 300 and len(rim_ids) > 0 and (float(len(rim_ids)) / max(float(len(raw_ids)), 1.0)) <= 0.12)
    severe_fragmentation = bool(
        (raw_best and float(raw_best.get("largest_component_fraction", 1.0) or 1.0) < 0.25)
        or (raw_best and int(raw_best.get("component_count", 1) or 1) >= 12)
    )
    low_rim_support = bool(rim_best and int(rim_best.get("edge_count", 0) or 0) < max(24, int(0.10 * max(len(raw_ids), 1))))
    out["normalized_rim_collapse_suspected"] = bool(collapsed)
    out["raw_selection_severe_fragmentation_suspected"] = bool(severe_fragmentation)
    out["normalized_rim_low_support_suspected"] = bool(low_rim_support)
    if not raw_rows and not rim_rows:
        status = "no_measured_opening_candidates"
    elif collapsed and not bool(out.get("raw_vs_normalized_measurement_agree", False)):
        status = "normalized_rim_collapse_or_wrong_opening_suspected"
    elif severe_fragmentation and not bool(out.get("raw_vs_normalized_measurement_agree", False)):
        status = "raw_fragmented_selection_requires_opening_recovery"
    elif bool(out.get("raw_vs_normalized_measurement_agree", False)):
        status = "raw_and_normalized_opening_measurements_agree"
    else:
        status = "opening_measurement_ambiguous_needs_review"
    out["audit_status"] = status
    out["next_stage_permission"] = "Recognition may use measured opening frame only after audit identifies a stable selected-opening candidate."
    out["forbidden_transfer"] = "Do not promote broad RegionData or remote cylindrical fragments into damaged BORE ownership from this audit."
    return out


def _edge_id_adjacent_faces(mesh: object, edge_ids: tuple[int, ...]) -> tuple[int, ...]:
    """Return faces adjacent to measured opening edge IDs.

    This is still measurement/anchor evidence, not ownership.  It exists because
    damaged imported meshes can make RegionData.seed_face_ids come from a
    collapsed normalized rim.  Recognition needs the faces adjacent to the
    actually resolved selected-opening measurement, not whatever generic rim
    normalizer happened to preserve.
    """

    ids = tuple_ints(edge_ids)
    if not ids:
        return ()
    try:
        faces = np.asarray(getattr(mesh, "faces"), dtype=np.int64)[:, :3]
    except Exception:
        return ()
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fid, tri in enumerate(faces):
        try:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                edge_to_faces.setdefault(key, []).append(int(fid))
        except Exception:
            continue
    try:
        unique_edges = tuple(sorted(edge_to_faces.keys()))
    except Exception:
        return ()
    out: set[int] = set()
    for eid in ids:
        if 0 <= int(eid) < len(unique_edges):
            out.update(int(v) for v in edge_to_faces.get(unique_edges[int(eid)], ()))
    return tuple(sorted(out))


def _candidate_centerline_distance_to_region(row: Mapping[str, object], *, region_center: np.ndarray, region_axis: np.ndarray) -> float:
    try:
        p = np.asarray(row.get("center", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
        delta = p - region_center.reshape(3)
        return float(np.linalg.norm(delta - region_axis.reshape(3) * float(np.dot(delta, region_axis.reshape(3)))))
    except Exception:
        return 999999.0


def _finite_float_value(value: object, default: float) -> float:
    """Return finite float without treating valid 0.0 as missing."""

    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if np.isfinite(out) else float(default)


def _opening_from_resolved_frame(frame: Mapping[str, object]) -> BoreOpeningMeasurement | None:
    """Rehydrate Measurement's selected-opening frame as evidence only.

    The resolver row is not CandidateData; this just lets the two-opening
    measurement stage use the already-measured selected opening instead of
    reinterpreting RegionData or the collapsed normalized rim.
    """

    if not bool(frame.get("resolved", False)):
        return None
    candidate = frame.get("candidate", {})
    if not isinstance(candidate, Mapping):
        candidate = frame
    edge_ids = tuple_ints(frame.get("expanded_edge_ids", ())) or tuple_ints(candidate.get("edge_ids", ())) or tuple_ints(frame.get("edge_ids", ()))
    radius = _finite_float_value(frame.get("radius", candidate.get("radius", 0.0)), 0.0)
    if radius <= 1.0e-12:
        return None
    center = tuple(float(v) for v in tuple(frame.get("center", candidate.get("center", (0.0, 0.0, 0.0))))[:3])
    axis = tuple(float(v) for v in tuple(frame.get("axis", candidate.get("axis", (0.0, 0.0, 1.0))))[:3])
    edge_count = int(frame.get("expanded_edge_count", candidate.get("edge_count", len(edge_ids))) or len(edge_ids))
    return BoreOpeningMeasurement(
        edge_ids=edge_ids,
        edge_count=int(edge_count),
        vertex_ids=(),
        vertex_count=int(candidate.get("vertex_count", 0) or 0),
        center=center,  # type: ignore[arg-type]
        axis=axis,  # type: ignore[arg-type]
        radius=float(radius),
        diameter=float(2.0 * radius),
        closed=bool(candidate.get("closed", False)),
        near_closed=bool(candidate.get("near_closed", False)),
        endpoint_gap=_finite_float_value(candidate.get("endpoint_gap", 0.0), 0.0),
        endpoint_gap_ratio=_finite_float_value(candidate.get("endpoint_gap_ratio", 0.0), 0.0),
        branch_vertex_count=int(candidate.get("branch_vertex_count", 0) or 0),
        open_endpoint_count=int(candidate.get("open_endpoint_count", 0) or 0),
        component_count=int(candidate.get("component_count", 1) or 1),
        plane_rms=_finite_float_value(candidate.get("plane_rms", 0.0), 0.0),
        plane_rel_rms=_finite_float_value(candidate.get("plane_rel_rms", 0.0), 0.0),
        radius_rms=_finite_float_value(candidate.get("radius_rms", 0.0), 0.0),
        radius_rel_rms=_finite_float_value(candidate.get("radius_rel_rms", 0.0), 0.0),
        radius_mad=_finite_float_value(candidate.get("radius_mad", 0.0), 0.0),
        circularity=_finite_float_value(candidate.get("circularity", 0.0), 0.0),
        confidence=_finite_float_value(frame.get("confidence", candidate.get("confidence", 0.0)), 0.0),
        diagnostics={
            **(dict(candidate.get("diagnostics", {}) or {}) if isinstance(candidate.get("diagnostics", {}), Mapping) else {}),
            "mode": "resolved_selected_opening_frame_rehydrated_for_two_opening_measurement",
            "semantic_role": "selected_opening_measurement_evidence_only",
            "source_resolver_status": str(frame.get("resolver_status", "")),
            "primary_edge_count": int(frame.get("primary_edge_count", 0) or 0),
            "expanded_edge_count": int(frame.get("expanded_edge_count", 0) or 0),
        },
    )


def _measure_two_opening_frame_from_resolver(
    *,
    mesh: object,
    selected_opening_frame_resolver: Mapping[str, object],
    boundary_loops: tuple[tuple[object, ...], ...],
    face_ids: tuple[int, ...],
) -> dict[str, object]:
    selected_opening = _opening_from_resolved_frame(selected_opening_frame_resolver)
    if selected_opening is None:
        return {
            "used": True,
            "valid": False,
            "status": "no_resolved_selected_opening_for_two_opening_measurement",
            "semantic_stage": "selected_opening_to_opposite_opening_to_measured_bore_frame",
        }
    try:
        measured = measure_two_opening_bore_frame(  # type: ignore[arg-type]
            mesh,
            selected_opening,
            region_boundary_loops=boundary_loops,  # type: ignore[arg-type]
            region_face_ids=face_ids,
        )
        out = measured.to_dict()
        diag = dict(out.get("diagnostics", {}) or {})
        out.update({
            "used": True,
            "status": "measured_two_opening_bore_frame" if bool(out.get("valid", False)) else str(diag.get("rejection_reason", "two_opening_measurement_unresolved")),
            "semantic_stage": "selected_opening_to_opposite_opening_to_measured_bore_frame",
        })
        return out
    except Exception as exc:
        return {
            "used": True,
            "valid": False,
            "status": f"two_opening_measurement_failed: {exc}",
            "semantic_stage": "selected_opening_to_opposite_opening_to_measured_bore_frame",
        }


def _selected_opening_frame_resolver(
    *,
    audit: Mapping[str, object],
    region_center: np.ndarray,
    region_axis: np.ndarray,
    region_radius: float,
    region_edge_ids: tuple[int, ...],
    region_seed_face_ids: tuple[int, ...],
) -> dict[str, object]:
    """Resolve the measured selected-opening frame from competing evidence.

    v1.3.5 fixes the core messy-mesh failure: when the audit says the
    normalized rim has collapsed or points to the wrong opening, the resolver is
    forbidden to use normalized candidates or the RegionData frame as authority.
    It must resolve from raw selected-edge component measurements.  RegionData is
    still neutral AOI context; it is not allowed to decide the selected physical
    opening on a polluted mesh.
    """

    collapsed_or_disagrees = bool(
        audit.get("normalized_rim_collapse_suspected", False)
        and not bool(audit.get("raw_vs_normalized_measurement_agree", False))
    )

    rows: list[dict[str, object]] = []
    source_order = ("raw_component_candidates", "raw_candidates") if collapsed_or_disagrees else ("raw_component_candidates", "raw_candidates", "normalized_candidates")
    for source_key in source_order:
        source_rows = audit.get(source_key, ())
        try:
            iterator = tuple(source_rows or ())
        except Exception:
            iterator = ()
        for raw in iterator:
            if isinstance(raw, Mapping):
                row = dict(raw)
                row.setdefault("resolver_input_source", source_key)
                rows.append(row)

    try:
        region_radius = float(region_radius)
    except Exception:
        region_radius = 0.0
    region_radius = region_radius if np.isfinite(region_radius) and region_radius > 1.0e-9 else 0.0
    if not rows:
        return {
            "used": False,
            "resolved": False,
            "resolver_status": "no_measured_opening_candidates",
            "semantic_stage": "selected_edge_evidence_to_measured_bore_frame",
            "normalized_candidates_forbidden_by_collapse_audit": bool(collapsed_or_disagrees),
        }

    region_axis = canonical_axis(region_axis)
    region_center = np.asarray(region_center, dtype=float).reshape(3)
    scored: list[tuple[float, dict[str, object]]] = []
    for row in rows:
        radius = _finite_float_value(row.get("radius", 0.0), 0.0)
        if radius <= 1.0e-9:
            continue
        source = str(row.get("resolver_input_source", "") or "")
        axis_dot = _opening_axis_dot(row.get("axis", (0.0, 0.0, 1.0)), region_axis)
        centerline_distance = _candidate_centerline_distance_to_region(row, region_center=region_center, region_axis=region_axis)
        radius_delta_rel = abs(radius - region_radius) / max(radius, region_radius, 1.0e-9) if region_radius > 1.0e-9 else 0.0
        confidence = _finite_float_value(row.get("confidence", 0.0), 0.0)
        circularity = _finite_float_value(row.get("circularity", 0.0), 0.0)
        radius_rel_rms = _finite_float_value(row.get("radius_rel_rms", 1.0), 1.0)
        plane_rel_rms = _finite_float_value(row.get("plane_rel_rms", 1.0), 1.0)
        edge_count = int(row.get("edge_count", 0) or 0)
        input_edge_count = int(row.get("input_edge_count", 0) or 0)
        support = min(float(edge_count) / max(float(input_edge_count), 1.0), 1.0)
        component_count = int(row.get("component_count", 1) or 1)
        largest_fraction = _finite_float_value(row.get("largest_component_fraction", 1.0), 1.0)
        strategy = str(row.get("component_strategy", "") or "")

        if collapsed_or_disagrees:
            # RegionData/normalized rim may already be wrong.  Score the raw
            # component measurement as selected-opening evidence in its own
            # right.  Keep region metrics as diagnostics only.
            source_bonus = 1.35 if source == "raw_component_candidates" else -0.85
            whole_cloud_penalty = 1.60 if (strategy == "all_selected_edge_fragments" and component_count >= 12 and largest_fraction < 0.35) else 0.0
            normalized_penalty = 999.0 if source == "normalized_candidates" else 0.0
            score = (
                source_bonus
                + 2.15 * max(0.0, min(confidence, 1.0))
                + 1.80 * max(0.0, min(circularity, 1.0))
                + 0.90 * min(float(edge_count) / 64.0, 1.0)
                + 0.65 * min(support * 10.0, 1.0)
                - 1.35 * min(radius_rel_rms, 1.0)
                - 0.45 * min(plane_rel_rms, 1.0)
                - whole_cloud_penalty
                - normalized_penalty
            )
        else:
            whole_cloud_penalty = 1.25 if (strategy == "all_selected_edge_fragments" and component_count >= 12 and largest_fraction < 0.35) else 0.0
            collapsed_norm_penalty = 0.40 if (source == "normalized_candidates" and bool(audit.get("normalized_rim_collapse_suspected", False))) else 0.0
            score = (
                3.20 * max(0.0, 1.0 - min(radius_delta_rel / 0.32, 1.0))
                + 2.40 * max(0.0, min(axis_dot, 1.0))
                + 1.90 * max(0.0, 1.0 - min(centerline_distance / max(region_radius * 1.25, 2.0), 1.0))
                + 0.90 * confidence
                + 0.60 * circularity
                + 0.40 * support
                - 0.90 * min(radius_rel_rms, 1.0)
                - whole_cloud_penalty
                - collapsed_norm_penalty
            )
            normalized_penalty = 0.0

        scored_row = {
            **row,
            "resolver_score": float(score),
            "resolver_axis_abs_dot_to_region": float(axis_dot),
            "resolver_radius_delta_rel_to_region": float(radius_delta_rel),
            "resolver_centerline_distance_to_region": float(centerline_distance),
            "resolver_component_mode": "raw_component_required" if collapsed_or_disagrees else "normal_region_agreement_mode",
            "resolver_normalized_candidates_forbidden": bool(collapsed_or_disagrees),
            "resolver_source_bonus_or_penalty_applied": float(-999.0 if source == "normalized_candidates" and collapsed_or_disagrees else (1.35 if source == "raw_component_candidates" and collapsed_or_disagrees else 0.0)),
        }
        scored.append((float(score), scored_row))

    scored.sort(key=lambda item: item[0], reverse=True)
    best = dict(scored[0][1]) if scored else {}
    if not best:
        return {
            "used": False,
            "resolved": False,
            "resolver_status": "all_measured_candidates_invalid",
            "semantic_stage": "selected_edge_evidence_to_measured_bore_frame",
            "normalized_candidates_forbidden_by_collapse_audit": bool(collapsed_or_disagrees),
        }

    radius_delta = _finite_float_value(best.get("resolver_radius_delta_rel_to_region", 999.0), 999.0)
    axis_dot = _finite_float_value(best.get("resolver_axis_abs_dot_to_region", 0.0), 0.0)
    center_dist = _finite_float_value(best.get("resolver_centerline_distance_to_region", 999999.0), 999999.0)
    score = _finite_float_value(best.get("resolver_score", 0.0), 0.0)
    source = str(best.get("resolver_input_source", "") or "")
    confidence = _finite_float_value(best.get("confidence", 0.0), 0.0)
    edge_count = int(best.get("edge_count", 0) or 0)
    radius_rel_rms = _finite_float_value(best.get("radius_rel_rms", 1.0), 1.0)

    if collapsed_or_disagrees:
        stable = bool(
            source == "raw_component_candidates"
            and score >= 1.55
            and edge_count >= 6
            and confidence >= 0.18
            and radius_rel_rms <= 0.55
        )
    else:
        stable = bool(
            score >= 3.15
            and radius_delta <= 0.36
            and axis_dot >= 0.72
            and center_dist <= max(region_radius * 1.65, 4.0)
        )

    if stable:
        expanded_edge_ids = tuple_ints(best.get("edge_ids", ()))
        primary_edge_ids = tuple_ints(best.get("seed_component_edge_ids", ())) or expanded_edge_ids
        edge_ids = primary_edge_ids
        return {
            "used": True,
            "resolved": True,
            "resolver_status": "measured_selected_opening_frame_resolved",
            "resolver_source": source,
            "semantic_stage": "selected_edge_evidence_to_measured_bore_frame",
            "center": tuple(float(v) for v in tuple(best.get("center", (0.0, 0.0, 0.0)))[:3]),
            "axis": tuple(float(v) for v in tuple(best.get("axis", (0.0, 0.0, 1.0)))[:3]),
            "radius": float(best.get("radius", region_radius) or region_radius),
            "diameter": float(2.0 * float(best.get("radius", region_radius) or region_radius)),
            "edge_ids": edge_ids,
            "edge_count": int(len(edge_ids) or edge_count),
            "primary_edge_ids": primary_edge_ids,
            "primary_edge_count": int(len(primary_edge_ids)),
            "expanded_edge_ids": expanded_edge_ids,
            "expanded_edge_count": int(len(expanded_edge_ids)),
            "seed_island_authority": "primary_raw_component_edges",
            "confidence": float(confidence),
            "score": float(score),
            "axis_abs_dot_to_region": float(axis_dot),
            "radius_delta_rel_to_region": float(radius_delta),
            "centerline_distance_to_region": float(center_dist),
            "raw_component_resolver_used": bool(source == "raw_component_candidates"),
            "normalized_candidates_forbidden_by_collapse_audit": bool(collapsed_or_disagrees),
            "candidate": best,
            "candidate_rankings": tuple(dict(row) for _score, row in scored[:12]),
            "forbidden_transfer": "MeasuredBoreFrame is evidence for Recognition; it is not CandidateData or rebuild authority.",
        }

    return {
        "used": True,
        "resolved": False,
        "resolver_status": "measured_candidates_do_not_resolve_selected_opening_frame",
        "semantic_stage": "selected_edge_evidence_to_measured_bore_frame",
        "best_unresolved_candidate": best,
        "best_score": float(score),
        "best_axis_abs_dot_to_region": float(axis_dot),
        "best_radius_delta_rel_to_region": float(radius_delta),
        "best_centerline_distance_to_region": float(center_dist),
        "raw_component_resolver_used": bool(source == "raw_component_candidates"),
        "normalized_candidates_forbidden_by_collapse_audit": bool(collapsed_or_disagrees),
        "candidate_rankings": tuple(dict(row) for _score, row in scored[:12]),
        "region_fallback_edge_ids": tuple_ints(region_edge_ids),
        "region_fallback_seed_face_ids": tuple_ints(region_seed_face_ids),
        "forbidden_transfer": "No measured selected-opening frame means no damaged BORE wall ownership promotion.",
    }


def recognize_bore_region_selection(mesh: object, region: object) -> dict[str, object]:
    """Recognize independent physical feature objects from a RegionData."""

    try:
        vertices, faces = _mesh_faces_and_vertices(mesh)
    except Exception as exc:
        error = {"failed": True, "error": f"invalid_mesh_for_recognition: {exc}", "pipeline_stage": "recognition"}
        return {
            "bore_evidence_ledger": error,
            "bore_recognition_result": error,
            "candidate_data": (),
            "recognition_features": (),
            "recognition_engine_features": (),
            "promoted_feature_candidates": (),
            "rebuild_ready": False,
            "rebuild_block_reason": "recognition_failed_invalid_mesh",
            "active_candidate_authority": ACTIVE_CANDIDATE_AUTHORITY,
        }

    region_diag = dict(getattr(region, "diagnostics", {}) or {})
    # v1.2.4 damaged-bore handoff correction:
    # RegionData.edge_ids is the stable raw selected-edge snapshot.  Diagnostics
    # may contain only normalized/opening-rim IDs, so Recognition must prefer the
    # typed RegionData field and use diagnostics only as a fallback.
    try:
        selected_edge_ids = tuple_ints(getattr(region, "edge_ids", ()))
    except Exception:
        selected_edge_ids = ()
    if not selected_edge_ids:
        try:
            selected_edge_ids = tuple(int(v) for v in tuple(region_diag.get("selected_edge_ids", ()) or ()))
        except Exception:
            selected_edge_ids = ()
    try:
        loop_edges = tuple(getattr(region, "loop_edges", ()) or ())
    except Exception:
        loop_edges = ()
    try:
        loop_vertices = tuple_ints(getattr(region, "loop_vertices", ()))
    except Exception:
        loop_vertices = ()
    try:
        center = np.asarray(getattr(region, "center", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
    except Exception:
        center = np.zeros(3, dtype=float)
    try:
        axis = canonical_axis(getattr(region, "axis", (0.0, 0.0, 1.0)))
    except Exception:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    try:
        radius = float(getattr(region, "radius", 0.0) or 0.0)
    except Exception:
        radius = 0.0
    try:
        seed_face_ids = tuple_ints(getattr(region, "seed_face_ids", ()))
    except Exception:
        seed_face_ids = ()

    # v1.2.4: keep the raw selected-edge adjacent faces separate from the
    # normalized rim seed faces.  Damaged BORE review must be able to display
    # evidence at the operator's raw selection instead of being forced to use the
    # small normalized rim seed result.
    cutout_for_raw = region_diag.get("cutout", {})
    raw_selected_edge_adjacent_face_ids = tuple_ints(region_diag.get("direct_selected_edge_adjacent_face_ids", ()))
    if not raw_selected_edge_adjacent_face_ids and isinstance(cutout_for_raw, Mapping):
        raw_selected_edge_adjacent_face_ids = tuple_ints(cutout_for_raw.get("direct_selected_edge_adjacent_face_ids", ()))
    if not raw_selected_edge_adjacent_face_ids:
        raw_selected_edge_adjacent_face_ids = tuple_ints(region_diag.get("raw_selected_edge_adjacent_face_ids", ()))
    try:
        face_ids = tuple_ints(getattr(region, "face_ids", ()))
    except Exception:
        face_ids = ()
    try:
        boundary_loops = tuple(tuple_edges(loop) for loop in tuple(getattr(region, "derived_boundary_loops", ()) or ()))
    except Exception:
        boundary_loops = ()
    cutout = region_diag.get("cutout", {})
    if not boundary_loops and isinstance(cutout, Mapping):
        try:
            boundary_loops = tuple(tuple_edges(loop) for loop in tuple(cutout.get("boundary_loops", ()) or ()))
        except Exception:
            boundary_loops = ()

    face_normals = _face_normals(vertices, faces, mesh)
    try:
        face_centroids = vertices[faces[:, :3]].mean(axis=1)
    except Exception as exc:
        error = {"failed": True, "error": f"failed_face_centroids: {exc}", "pipeline_stage": "recognition"}
        return {
            "bore_evidence_ledger": error,
            "bore_recognition_result": error,
            "candidate_data": (),
            "recognition_features": (),
            "recognition_engine_features": (),
            "promoted_feature_candidates": (),
            "rebuild_ready": False,
            "rebuild_block_reason": "recognition_failed_face_centroids",
            "active_candidate_authority": ACTIVE_CANDIDATE_AUTHORITY,
        }

    try:
        boundary_stack = describe_boundary_loop_stack_geometry(
            boundary_loops=boundary_loops,
            vertices=vertices,
            axis=axis,
            nominal_radius=float(radius),
            min_loop_edges=8,
        )
        boundary_loop_geometry = tuple(boundary_stack.boundary_loops)
    except Exception:
        boundary_loop_geometry = ()

    try:
        feature_patch_measurement = measure_feature_patch_geometry(
            face_ids=face_ids,
            face_centroids=face_centroids,
            face_normals=face_normals,
            center=center,
            axis=axis,
            radius=float(radius),
            boundary_loop_geometry=boundary_loop_geometry,
        ).to_dict()
    except Exception as exc:
        feature_patch_measurement = {
            "failed": True,
            "error": str(exc),
            "face_count": int(len(face_ids)),
            "radius": float(radius),
            "diameter": float(2.0 * radius),
            "measurement_frame_source": "selected_edge_region_frame",
        }

    normalized_opening_edge_ids = _opening_edge_ids_from_region_diagnostics(region_diag, tuple_ints(getattr(region, "edge_ids", ())))
    selected_opening_measurement_audit = _selected_opening_measurement_audit(
        mesh,
        selected_edge_ids=selected_edge_ids,
        normalized_edge_ids=normalized_opening_edge_ids,
    )
    selected_opening_frame_resolver = _selected_opening_frame_resolver(
        audit=selected_opening_measurement_audit,
        region_center=center,
        region_axis=axis,
        region_radius=float(radius),
        region_edge_ids=normalized_opening_edge_ids,
        region_seed_face_ids=seed_face_ids,
    )
    resolved_opening_seed_face_ids = ()
    if bool(selected_opening_frame_resolver.get("resolved", False)):
        # v1.3.7: derive the Recognition seed island from the primary raw
        # selected-edge component, not from expanded measurement support.
        # Expanded support may contain extra fragments from the same fitted frame;
        # it is evidence only, not locality/display authority.
        seed_edge_ids_for_faces = tuple_ints(
            selected_opening_frame_resolver.get("primary_edge_ids", ())
            or selected_opening_frame_resolver.get("edge_ids", ())
        )
        resolved_opening_seed_face_ids = _edge_id_adjacent_faces(mesh, seed_edge_ids_for_faces)
    two_opening_bore_frame = _measure_two_opening_frame_from_resolver(
        mesh=mesh,
        selected_opening_frame_resolver=selected_opening_frame_resolver,
        boundary_loops=boundary_loops,
        face_ids=face_ids,
    )

    selected_opening_measurement_audit = {
        **selected_opening_measurement_audit,
        "selected_opening_frame_resolver": selected_opening_frame_resolver,
        "selected_opening_frame_resolved": bool(selected_opening_frame_resolver.get("resolved", False)),
        "resolved_opening_seed_face_count": int(len(resolved_opening_seed_face_ids)),
        "resolved_opening_primary_edge_count": int(selected_opening_frame_resolver.get("primary_edge_count", 0) or 0),
        "resolved_opening_expanded_edge_count": int(selected_opening_frame_resolver.get("expanded_edge_count", 0) or 0),
        "selected_opening_seed_island_source": str(selected_opening_frame_resolver.get("seed_island_authority", "")),
        "two_opening_bore_frame": two_opening_bore_frame,
        "two_opening_bore_frame_valid": bool(two_opening_bore_frame.get("valid", False)),
        "two_opening_bore_frame_status": str(two_opening_bore_frame.get("status", "")),
    }

    candidate_result = component_engine_feature_candidates(
        faces=faces,
        face_ids=face_ids,
        face_centroids=face_centroids,
        face_normals=face_normals,
        region_center=center,
        region_axis=axis,
        region_radius=float(radius),
        boundary_loop_geometry=boundary_loop_geometry,
        boundary_loops=boundary_loops,
        vertices=vertices,
        seed_face_ids=seed_face_ids,
        raw_selected_edge_adjacent_face_ids=raw_selected_edge_adjacent_face_ids,
        selected_edge_ids=selected_edge_ids,
        normalized_opening_edge_ids=normalized_opening_edge_ids,
        selected_opening_measurement_audit=selected_opening_measurement_audit,
        selected_opening_frame_resolver=selected_opening_frame_resolver,
        two_opening_bore_frame=two_opening_bore_frame,
        resolved_opening_seed_face_ids=resolved_opening_seed_face_ids,
        region_diagnostics=region_diag,
    )
    candidates = tuple(
        dict(item)
        for item in tuple(candidate_result.get("candidate_data", candidate_result.get("features", ())) or ())
        if isinstance(item, Mapping)
    )
    component_diag = dict(candidate_result.get("diagnostics", {}) or {})
    component_diag.setdefault("active_candidate_authority", ACTIVE_CANDIDATE_AUTHORITY)
    component_diag["v48_connected_opening_component_used"] = True
    component_diag["selected_edge_ids_handoff_count"] = int(len(selected_edge_ids))
    component_diag["normalized_seed_face_count"] = int(len(seed_face_ids))
    component_diag["raw_selected_edge_adjacent_face_count"] = int(len(raw_selected_edge_adjacent_face_ids))
    component_diag["normalized_opening_edge_ids_handoff_count"] = int(len(normalized_opening_edge_ids))
    component_diag["v137_selected_opening_seed_island_isolation_used"] = True
    component_diag["selected_opening_frame_resolver_status"] = str(selected_opening_frame_resolver.get("resolver_status", ""))
    component_diag["selected_opening_frame_resolved"] = bool(selected_opening_frame_resolver.get("resolved", False))
    component_diag["two_opening_bore_frame_valid"] = bool(two_opening_bore_frame.get("valid", False))
    component_diag["two_opening_bore_frame_status"] = str(two_opening_bore_frame.get("status", ""))
    component_diag["two_opening_bore_frame_depth"] = two_opening_bore_frame.get("depth", "")
    component_diag["resolved_opening_seed_face_count"] = int(len(resolved_opening_seed_face_ids))
    component_diag["v132_selected_opening_measurement_audit_used"] = True
    component_diag["opening_measurement_audit_status"] = str(selected_opening_measurement_audit.get("audit_status", ""))
    component_diag["raw_component_opening_candidate_count"] = int(selected_opening_measurement_audit.get("raw_component_candidate_count", 0) or 0)
    component_diag["raw_component_best_opening_radius"] = selected_opening_measurement_audit.get("raw_component_best_radius", "")
    component_diag["v48_connected_opening_component_preview_used"] = True
    component_diag["bore_detection_and_rebuild_trial_enabled"] = bool(component_diag.get("bore_recognition_enabled", False))
    component_diag["damaged_bore_detection_preview_enabled"] = bool(component_diag.get("damaged_bore_preview_enabled", False))
    component_diag["v33_semantic_boundary_hardening_used"] = True
    component_diag["diagnostic_to_candidate_bridge_enabled"] = False

    anchor_allows_candidates = _recognition_anchor_allows_candidates(region_diag)
    fallback_candidates = _fallback_features_from_region_diagnostics(region_diag)
    if not anchor_allows_candidates:
        candidates = ()
        component_diag["candidate_bridge_source"] = "anchor_rejected_all_candidates"
        component_diag["candidate_bridge_count"] = 0
    elif not candidates and fallback_candidates:
        candidates = fallback_candidates
        component_diag["candidate_bridge_source"] = "disabled_by_v33_semantic_boundary_hardening"
        component_diag["candidate_bridge_count"] = 0

    feature_layer_analysis = {
        "mode": ACTIVE_CANDIDATE_AUTHORITY,
        "feature_authority": "recognition",
        "candidate_authority": ACTIVE_CANDIDATE_AUTHORITY,
        "feature_entities": candidates,
        "component_diagnostics": component_diag,
        "assembly_classification_policy": ASSEMBLY_CLASSIFICATION_POLICY,
        "x1_feature_family_vocabulary": tuple(item.value for item in FeatureFamily),
        "recognition_stage_policy": tuple(item.value for item in RecognitionStage),
        "evidence_kind_vocabulary": tuple(item.value for item in EvidenceKind),
        "removed_pre_component_candidate_paths": (
            "ui_collector_candidates",
            "subtract_transition_surfaces_fallback",
            "broad_region_borehole_fallback",
            "posthoc_repair_helpers",
        ),
    }
    component_family_ledger = {
        "mode": "component_engine_feature_object_ledger",
        "candidate_authority": ACTIVE_CANDIDATE_AUTHORITY,
        "object_promotion_policy": "physical_surface_objects_only_with_x1_stage_gate",
        "stage_gate": "only accepted_candidate may be action-enabled; diagnostic/review/promotion_preview stay preview-only; damaged_bore remains review-only and must display selected-opening-local evidence only",
        "ui_candidate_count": int(len(candidates)),
        "diagnostic_candidate_count": int(len(tuple(item for item in candidates if not bool(item.get("candidate_action_enabled", item.get("rebuild_authorized", False)))))),
        "feature_object_count": int(len(candidates)),
        "assembly_classification_policy": ASSEMBLY_CLASSIFICATION_POLICY,
    }

    ledger_dict = {
        "mode": "region_data_to_candidate_data_ledger",
        "pipeline_stage": "recognition_evidence_ledger",
        "active_candidate_authority": ACTIVE_CANDIDATE_AUTHORITY,
        "region_data": {
            "edge_ids": selected_edge_ids,
            "loop_edges": tuple_edges(loop_edges),
            "loop_vertices": loop_vertices,
            "center": tuple(float(v) for v in center),
            "axis": tuple(float(v) for v in axis),
            "radius": float(radius),
            "seed_face_ids": seed_face_ids,
            "raw_selected_edge_adjacent_face_ids": raw_selected_edge_adjacent_face_ids,
            "normalized_opening_edge_ids": normalized_opening_edge_ids,
        },
        "measured_face_ids": face_ids,
        "measured_boundary_loops": boundary_loops,
        "feature_patch_measurement": feature_patch_measurement,
        "selected_opening_measurement_audit": selected_opening_measurement_audit,
        "selected_opening_frame_resolver": selected_opening_frame_resolver,
        "two_opening_bore_frame": two_opening_bore_frame,
        "resolved_opening_seed_face_ids": resolved_opening_seed_face_ids,
        "feature_layer_analysis": feature_layer_analysis,
        "component_family_ledger": component_family_ledger,
        "diagnostics": {
            "region_select_feature_authority": False,
            "selected_edge_count": int(region_diag.get("selected_edge_count", len(loop_edges)) or len(loop_edges)),
            "selected_face_count": int(len(face_ids)),
            "seed_face_count": int(len(seed_face_ids)),
            "raw_selected_edge_adjacent_face_count": int(len(raw_selected_edge_adjacent_face_ids)),
            "normalized_opening_edge_count": int(len(normalized_opening_edge_ids)),
            "opening_measurement_audit_status": str(selected_opening_measurement_audit.get("audit_status", "")),
            "selected_opening_frame_resolver_status": str(selected_opening_frame_resolver.get("resolver_status", "")),
            "selected_opening_frame_resolved": bool(selected_opening_frame_resolver.get("resolved", False)),
            "two_opening_bore_frame_valid": bool(two_opening_bore_frame.get("valid", False)),
            "two_opening_bore_frame_status": str(two_opening_bore_frame.get("status", "")),
            "resolved_opening_seed_face_count": int(len(resolved_opening_seed_face_ids)),
            "boundary_loop_count": int(len(boundary_loops)),
            "boundary_loop_edge_counts": tuple(int(len(loop)) for loop in boundary_loops),
        },
    }

    result_dict = recognition_result_dict_from_component_features(
        features=candidates,
        diagnostics=component_diag,
    )
    promoted = tuple(
        item for item in candidates
        if bool(item.get("candidate_action_enabled", item.get("rebuild_authorized", False)))
        and str(item.get("recognition_stage", "") or "") == RecognitionStage.ACCEPTED_CANDIDATE.value
        and str(item.get("feature_family", "") or "") in {FeatureFamily.BORE.value, FeatureFamily.CHAMFER_FORM.value}
    )
    promoted_borehole = next((dict(item) for item in promoted if str(item.get("entity_type", item.get("feature_kind", ""))) == "borehole"), None)
    promoted_chamfer = next((dict(item) for item in promoted if str(item.get("entity_type", item.get("feature_kind", ""))) == "chamfer"), None)

    return {
        "region_data_ledger": ledger_dict,
        "bore_evidence_ledger": ledger_dict,
        "bore_recognition_result": result_dict,
        "feature_patch_measurement": feature_patch_measurement,
        "selected_opening_measurement_audit": selected_opening_measurement_audit,
        "selected_opening_frame_resolver": selected_opening_frame_resolver,
        "two_opening_bore_frame": two_opening_bore_frame,
        "resolved_opening_seed_face_ids": resolved_opening_seed_face_ids,
        "feature_layer_analysis": feature_layer_analysis,
        "component_feature_family_ledger": component_family_ledger,
        "candidate_data": candidates,
        "candidate_result": result_dict,
        "recognition_features": candidates,
        "recognition_engine_features": candidates,
        "promoted_candidate_data": promoted,
        "promoted_feature_candidates": promoted,
        "promoted_feature_candidate_count": int(len(promoted)),
        "promoted_feature_types": tuple(str(item.get("entity_type", item.get("feature_kind", "unknown"))) for item in promoted),
        "promoted_borehole_candidate_available": promoted_borehole is not None,
        "promoted_borehole_rebuild": promoted_borehole is not None,
        "promoted_borehole_candidate": promoted_borehole or {},
        "promoted_chamfer_candidate_available": promoted_chamfer is not None,
        "promoted_chamfer_rebuild": promoted_chamfer is not None,
        "promoted_chamfer_candidate": promoted_chamfer or {},
        "rebuild_ready": bool(promoted),
        "rebuild_block_reason": "" if promoted else "component_engine_did_not_emit_actionable_rebuild_authorized_candidate_data",
        "pipeline_stage": "recognition_complete",
        "region_select_feature_authority": False,
        "active_candidate_authority": ACTIVE_CANDIDATE_AUTHORITY,
        "component_engine_diagnostics": component_diag,
        "removed_pre_component_candidate_paths": (
            "ui_collector_candidates",
            "broad_region_borehole_fallback",
            "subtract_transition_surfaces_borehole_fallback",
            "posthoc_repair_helpers",
        ),
    }


__all__ = [
    "ACTIVE_CANDIDATE_AUTHORITY",
    "ASSEMBLY_CLASSIFICATION_POLICY",
    "REGION_SELECT_FEATURE_AUTHORITY",
    "recognize_bore_region_selection",
]
