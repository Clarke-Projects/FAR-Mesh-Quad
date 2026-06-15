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
)
from .recognition_component_engine import (
    component_engine_feature_candidates,
    recognition_result_dict_from_component_features,
)
from .types import EvidenceKind, FeatureFamily, RecognitionStage, tuple_edges, tuple_ints

ACTIVE_CANDIDATE_AUTHORITY = "surface_component_classifier_v7"
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
    for key in ("rebuild_face_ids", "delete_patch_face_ids", "preview_face_ids", "semantic_face_ids", "face_ids"):
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
    stage_allows_action = bool(stage == RecognitionStage.ACCEPTED_CANDIDATE.value and family in {FeatureFamily.BORE.value, FeatureFamily.CHAMFER_FORM.value})
    item["candidate_action_enabled"] = bool(stage_allows_action and item.get("candidate_action_enabled", item.get("rebuild_authorized", False)))
    item["rebuild_authorized"] = bool(item["candidate_action_enabled"])
    item.setdefault("rebuild_gate", "candidate_data_bridge_diagnostic_only")
    item["face_ids"] = face_ids
    item.setdefault("semantic_face_ids", face_ids)
    item.setdefault("preview_face_ids", face_ids)
    item.setdefault("rebuild_face_ids", tuple_ints(item.get("rebuild_face_ids", item.get("delete_patch_face_ids", ()))) or face_ids)
    item["face_count"] = int(item.get("face_count", 0) or len(face_ids))
    item.setdefault("recognition_rule", source)
    return item


def _fallback_features_from_region_diagnostics(region_diag: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    """CandidateData-owned bridge for older diagnostic rows.

    BoreActions must not recover candidates from diagnostic ledgers. During the
    transition, recognition imports those rows and emits them as proper
    CandidateData rows only when the component engine produced no candidate rows.
    """

    if not _recognition_anchor_allows_candidates(region_diag):
        return ()

    rows: list[tuple[str, Mapping[str, object]]] = []

    for key in ("recognition_features", "recognition_engine_features"):
        existing = region_diag.get(key)
        if isinstance(existing, (list, tuple)):
            for raw in existing:
                if isinstance(raw, Mapping):
                    rows.append((f"region_diagnostics.{key}", raw))

    ledger = region_diag.get("macro_feature_family_ledger")
    if isinstance(ledger, Mapping):
        for key in ("ui_candidates", "diagnostic_candidates"):
            values = ledger.get(key)
            if isinstance(values, (list, tuple)):
                for raw in values:
                    if isinstance(raw, Mapping):
                        rows.append((f"macro_feature_family_ledger.{key}", raw))

    patch = region_diag.get("feature_patch_measurement")
    if isinstance(patch, Mapping):
        radial = patch.get("radial_layer_analysis")
        if isinstance(radial, Mapping):
            for key in ("high_level_entities", "entities"):
                values = radial.get(key)
                if isinstance(values, (list, tuple)):
                    for raw in values:
                        if isinstance(raw, Mapping):
                            rows.append((f"feature_patch_measurement.radial_layer_analysis.{key}", raw))

    layer = region_diag.get("feature_layer_analysis")
    if isinstance(layer, Mapping):
        values = layer.get("feature_entities")
        if isinstance(values, (list, tuple)):
            for raw in values:
                if isinstance(raw, Mapping):
                    rows.append(("feature_layer_analysis.feature_entities", raw))

    out: list[dict[str, object]] = []
    seen: set[tuple[str, tuple[int, ...], str]] = set()
    for index, (source, raw) in enumerate(rows, start=1):
        item = _normalise_recognition_feature(raw, source=source, index=index)
        if item is None:
            continue
        key = (str(item.get("entity_type", "")), tuple_ints(item.get("face_ids", ())), str(item.get("candidate_id", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return tuple(out)

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
    )
    candidates = tuple(
        dict(item)
        for item in tuple(candidate_result.get("candidate_data", candidate_result.get("features", ())) or ())
        if isinstance(item, Mapping)
    )
    component_diag = dict(candidate_result.get("diagnostics", {}) or {})
    component_diag.setdefault("active_candidate_authority", ACTIVE_CANDIDATE_AUTHORITY)

    anchor_allows_candidates = _recognition_anchor_allows_candidates(region_diag)
    fallback_candidates = _fallback_features_from_region_diagnostics(region_diag)
    if not anchor_allows_candidates:
        candidates = ()
        component_diag["candidate_bridge_source"] = "anchor_rejected_all_candidates"
        component_diag["candidate_bridge_count"] = 0
    elif not candidates and fallback_candidates:
        candidates = fallback_candidates
        component_diag["candidate_bridge_source"] = "region_diagnostics_compatibility_bridge"
        component_diag["candidate_bridge_count"] = int(len(candidates))

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
        "stage_gate": "only accepted_candidate may be action-enabled; diagnostic/review/promotion_preview stay preview-only",
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
        },
        "measured_face_ids": face_ids,
        "measured_boundary_loops": boundary_loops,
        "feature_patch_measurement": feature_patch_measurement,
        "feature_layer_analysis": feature_layer_analysis,
        "component_family_ledger": component_family_ledger,
        "diagnostics": {
            "region_select_feature_authority": False,
            "selected_edge_count": int(region_diag.get("selected_edge_count", len(loop_edges)) or len(loop_edges)),
            "selected_face_count": int(len(face_ids)),
            "seed_face_count": int(len(seed_face_ids)),
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
        "rebuild_block_reason": "" if promoted else "component_engine_did_not_emit_actionable_candidate_data",
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
