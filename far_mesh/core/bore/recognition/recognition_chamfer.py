"""CHAMFER surface-role ownership helpers for FAR MESH BoreTool.

This module is the first v174 split from ``recognition_component_engine.py``.
It owns CHAMFER-local rail-to-rail bounded-surface reasoning only.  It does not
emit CandidateData, decide rebuild targets, mutate topology, or perform host
state changes.

Semantic boundary
-----------------
Input meaning:
    raw selected-rail transition evidence + local measured frame

Output meaning:
    owned CHAMFER_BAND face IDs and diagnostics for the caller to emit as
    CandidateData when the family-local recognizer authorizes it.

The component engine remains the orchestration and CandidateData emission owner.
v174e isolated the v173t CHAMFER ownership authority. v174f moves the
CHAMFER-local v173c role proof and v173o selected-rail-local ownership proof
helpers here while preserving their existing diagnostic dictionaries. v174g moves
the quarantined v173p/v173q fallback evaluators here as explicit legacy
fallback helpers, still behind the same component-engine export order. v174p
moves the CHAMFER-local CandidateData contract helper here while final
candidate-list assembly remains in the component engine. v174y moves CHAMFER
candidate-row assembly here while the component engine keeps the final
multi-family candidate-list merge.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping

import math
import numpy as np

from ..topology import boundary_edges_for_face_patch, connected_face_components, edge_loop_components
from ..types import EvidenceKind, FeatureFamily, FeaturePrimitiveKind, RecognitionStage, tuple_ints

OwnedSubsetEvaluator = Callable[
    [Iterable[int]],
    tuple[tuple[int, ...], dict[str, object], dict[str, object], dict[str, object]],
]

CHAMFER_RECOGNITION_SPLIT_CHECKPOINT_V174E = (
    "v174e_first_module_split_chamfer_rail_to_rail_ownership_no_behavior_change"
)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    return float(out) if math.isfinite(out) else float(default)




def _safe_int_v174y(value: object, default: int = 0) -> int:
    try:
        out = int(value)  # type: ignore[arg-type]
    except Exception:
        return int(default)
    return int(out)


def _component_stats_v174y(
    *,
    comp: tuple[int, ...],
    fid_to_local: Mapping[int, int],
    axial: np.ndarray,
    radial: np.ndarray,
    normal_axis_abs: np.ndarray,
    radial_normal_alignment: np.ndarray,
    seed_face_set: set[int],
    adjacency: Mapping[int, Iterable[int]],
) -> dict[str, object]:
    """Return component statistics used by CHAMFER candidate assembly.

    This is a local copy of the v174x shared-stat calculation to avoid a module
    cycle: recognition_common imports CHAMFER proof helpers, so the CHAMFER
    assembly helper cannot import recognition_common back.
    """

    idx = np.asarray([fid_to_local[int(fid)] for fid in comp if int(fid) in fid_to_local], dtype=np.int64)
    if idx.size == 0:
        return {"face_count": 0}
    comp_set = {int(fid) for fid in comp}
    direct_seed = comp_set & seed_face_set
    adjacent_seed: set[int] = set()
    if seed_face_set:
        for fid in comp_set:
            for nb in adjacency.get(int(fid), ()):  # selected-face adjacency only
                if int(nb) in seed_face_set:
                    adjacent_seed.add(int(fid))
                    break
    ax = axial[idx]
    rd = radial[idx]
    na = normal_axis_abs[idx]
    al = radial_normal_alignment[idx]
    radial_span = float(np.max(rd) - np.min(rd)) if rd.size else 0.0
    axial_span = float(np.max(ax) - np.min(ax)) if ax.size else 0.0
    return {
        "face_count": int(len(comp)),
        "axial_min": float(np.min(ax)),
        "axial_max": float(np.max(ax)),
        "axial_center": float(np.median(ax)),
        "axial_span": axial_span,
        "radial_min": float(np.min(rd)),
        "radial_max": float(np.max(rd)),
        "radial_median": float(np.median(rd)),
        "radial_span": radial_span,
        "radial_mad": float(np.median(np.abs(rd - np.median(rd)))) if rd.size else 0.0,
        "radial_rel_mad": float((np.median(np.abs(rd - np.median(rd))) / max(float(np.median(rd)), 1.0e-12))) if rd.size else 999999.0,
        "normal_axis_abs_median": float(np.median(na)),
        "normal_axis_abs_q75": float(np.percentile(na, 75.0)) if na.size else 1.0,
        "radial_normal_alignment_median": float(np.median(al)),
        "radial_normal_alignment_q25": float(np.percentile(al, 25.0)) if al.size else 0.0,
        "seed_direct_face_count": int(len(direct_seed)),
        "seed_adjacent_face_count": int(len(adjacent_seed)),
        "seed_related": bool(direct_seed or adjacent_seed),
    }


def _chamfer_score_v174y(st: Mapping[str, object], *, radius_scale: float, axial_span_all: float) -> float:
    face_count = int(st.get("face_count", 0) or 0)
    if face_count <= 0:
        return -1.0e9
    axial_span = _safe_float(st.get("axial_span", 0.0), 0.0)
    radial_span = _safe_float(st.get("radial_span", 0.0), 0.0)
    normal_axis = _safe_float(st.get("normal_axis_abs_median", 0.0), 0.0)
    radial_align = _safe_float(st.get("radial_normal_alignment_median", 0.0), 0.0)

    radial_transition = min(radial_span / max(0.020 * radius_scale, 0.20), 1.0)
    axial_transition = min(axial_span / max(0.020 * radius_scale, 0.20), 1.0)
    normal_mix = max(0.0, 1.0 - abs(normal_axis - radial_align))
    slope_present = min(normal_axis / 0.35, 1.0) * min(radial_align / 0.35, 1.0)
    seed_bonus = 2.2 if bool(st.get("seed_related", False)) else 0.0
    count_score = min(face_count / 80.0, 1.0)
    too_deep_penalty = 0.0
    if axial_span_all > 1.0e-9 and axial_span > 0.55 * axial_span_all:
        too_deep_penalty = 1.5
    return float(
        1.15 * radial_transition
        + 1.00 * axial_transition
        + 0.85 * normal_mix
        + 1.25 * slope_present
        + 0.25 * count_score
        + seed_bonus
        - too_deep_penalty
    )


def _project_points_to_frame_v174e(*, points: np.ndarray, frame: object) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points into the active rail frame as axial/radial coordinates."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    rel = pts[:, :3] - frame.center.reshape(1, 3)
    ax = rel @ frame.axis.reshape(3)
    rv = rel - ax.reshape(-1, 1) * frame.axis.reshape(1, 3)
    rd = np.linalg.norm(rv, axis=1)
    return ax.astype(float), rd.astype(float)



CHAMFER_RECOGNITION_PROOF_HELPER_SPLIT_CHECKPOINT_V174F = (
    "v174f_split_chamfer_role_proof_helpers_no_behavior_change"
)

CHAMFER_FAMILY_LOCAL_OWNERSHIP_CONTRACT_V174F = (
    "family_local_candidate_ownership_v173c_no_cross_family_candidate_creation_relationships_are_metadata"
)

CHAMFER_RECOGNITION_LEGACY_FALLBACK_SPLIT_CHECKPOINT_V174G = (
    "v174g_split_quarantined_chamfer_v173p_v173q_fallback_evaluators_no_behavior_change"
)

CHAMFER_RECOGNITION_CANDIDATE_CONTRACT_SPLIT_CHECKPOINT_V174P = (
    "v174p_split_chamfer_candidate_contract_helper_no_behavior_change"
)

CHAMFER_RECOGNITION_CANDIDATE_ASSEMBLY_SPLIT_CHECKPOINT_V174Y = (
    "v174y_split_chamfer_candidate_row_assembly_no_behavior_change"
)


def chamfer_candidate_contract_fields_v174p(
    *,
    candidate_id: str,
    face_ids: tuple[int, ...],
    accepted: bool,
    confidence: float,
    radius_inner: float,
    radius_outer: float,
    axial_span: float,
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Build the CHAMFER CandidateData dictionary used by the component engine.

    This is a family-local CHAMFER contract helper only.  It preserves the
    existing v173/v174 field names and actionability behavior; it does not
    append rows to the final candidate list, authorize a DeletePatchProposal,
    or mutate topology.
    """

    stage = RecognitionStage.ACCEPTED_CANDIDATE.value if accepted else RecognitionStage.REVIEW.value
    face_ids = tuple_ints(face_ids)
    return {
        "candidate_id": candidate_id,
        "feature_id": candidate_id,
        "entity_type": "chamfer",
        "feature_kind": "chamfer",
        "candidate_scope": "candidate_data",
        "candidate_authority": str(diagnostics.get("candidate_authority_v173c", "surface_component_classifier_v101_selected_rail_transition_band_ownership")),
        "active_candidate_authority": str(diagnostics.get("active_candidate_authority_v173c", "surface_component_classifier_v101_selected_rail_transition_band_ownership")),
        "feature_family": FeatureFamily.CHAMFER_FORM.value,
        "recognition_stage": stage,
        "display_name": "CHAMFER — clean annular transition surface" if accepted else "CHAMFER — annular transition review",
        "role": "rebuildable_chamfer_operation" if accepted else "chamfer_review_only",
        "status": "promoted_chamfer_from_clean_annular_transition_ownership" if accepted else "review_chamfer_annular_transition_evidence",
        "candidate_side_role": "selected_side_chamfer" if bool(diagnostics.get("seed_related", False)) else "independent_opposite_or_secondary_chamfer",
        "selection_seed_related": bool(diagnostics.get("seed_related", False)),
        "promotion_state": "promoted" if accepted else "review",
        "candidate_action_enabled": bool(accepted),
        "candidate_action": "rebuild" if accepted else "preview_only",
        "rebuild_authorized": bool(accepted),
        "rebuild_gate": "promoted_chamfer_candidate" if accepted else "chamfer_review_candidate",
        "face_ids": face_ids,
        "semantic_face_ids": face_ids,
        "display_face_ids": face_ids,
        "preview_face_ids": face_ids,
        "rebuild_face_ids": face_ids if accepted else (),
        "face_count": int(len(face_ids)),
        "confidence": float(confidence),
        "radius": float((radius_inner + radius_outer) * 0.5) if radius_outer >= radius_inner else float(radius_inner),
        "diameter": float((radius_inner + radius_outer)),
        "inner_radius": float(radius_inner),
        "outer_radius": float(radius_outer),
        "chamfer_width": float(abs(radius_outer - radius_inner)),
        "axial_span": float(axial_span),
        "depth": float(axial_span),
        "height": float(axial_span),
        "surface_condition": "annular_transition_owned_faces_only",
        "repair_strategy": "delete_patch_from_chamfer_owned_faces",
        "evidence_kinds": (
            EvidenceKind.OPENING_RING.value,
            EvidenceKind.CHAMFER_BAND.value,
            EvidenceKind.RADIUS_CONSISTENCY.value,
        ),
        "promotion_reasons": (
            "clean_chamfer_only_recognition",
            "annular_transition_surface_role",
            "selected_opening_seed_context" if bool(diagnostics.get("seed_related", False)) else "independent_annular_transition_geometry",
            "independent_clean_chamfer_rebuild_path_supported" if accepted else "review_only_chamfer_evidence_not_promoted",
        ),
        "rejection_reasons": () if accepted else ("chamfer_evidence_not_promoted",),
        "feature_primitives": (
            {
                "primitive_kind": FeaturePrimitiveKind.ANNULAR_CHAMFER_BAND.value,
                "source": str(diagnostics.get("primitive_source_v173c", "recognition_component_engine.v100_selected_rail_anchored_annular_transition_ownership")),
                "role": "accepted_physical_primitive" if accepted else "diagnostic_primitive_descriptor",
                "radius": float((radius_inner + radius_outer) * 0.5) if radius_outer >= radius_inner else float(radius_inner),
                "inner_radius": float(radius_inner),
                "outer_radius": float(radius_outer),
                "depth": float(axial_span),
                "confidence": float(confidence),
                "face_ids": face_ids,
                "diagnostics": dict(diagnostics),
            },
        ),
        "feature_primitive_count": 1,
        "feature_relationships": (),
        "feature_relationship_count": 0,
        "x1_primitive_bridge_contract": "mesh-native primitive descriptor, not CAD body and not rebuild target",
        "delete_patch_request_allowed": bool(accepted),
        "rebuild_target_policy_allowed": bool(accepted),
        "rebuild_target_policy_reason": "accepted chamfer-form candidate may request DeletePatchProposal; rebuild.py still validates topology" if accepted else "review candidate is preview-only",
        "recognition_rule": str(diagnostics.get("recognition_rule", "v96_independent_chamfer_band_surface_role_ownership")),
        "feature_ownership_source": str(diagnostics.get("feature_ownership_source_v173c", "independent_chamfer_band_surface_role_ownership_v173c")),
        "feature_ownership_split": "chamfer_owned_face_ids_only_no_bore_or_regiondata_ownership_no_cross_family_candidate_creation_v173c",
        "diagnostics": dict(diagnostics),
    }

def _face_patch_boundary_semantic_report_v174f(faces: np.ndarray, face_ids: Iterable[int]) -> dict[str, object]:
    """Return a small semantic topology report for a CHAMFER-owned face patch.

    This is CHAMFER-local role-proof evidence only.  It does not mutate topology
    and does not claim rebuild success.  v174f moves the helper used by the
    v173c/v173o CHAMFER proof path out of the component engine without changing
    the output dictionary keys consumed by existing diagnostics.
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

def chamfer_band_role_proof_v174f(
    *,
    faces: np.ndarray,
    face_ids: Iterable[int],
    component_stats: Mapping[str, object],
) -> dict[str, object]:
    """Prove CHAMFER_BAND ownership before CandidateData may rebuild.

    This is the v173c replacement for the old BORE-mouth-transition shortcut.
    A chamfer is an independent annular transition surface.  It may share rails
    with a BORE wall or a parent surface, but those rails are relationship
    metadata.  CandidateData for CHAMFER may only be emitted from a face strip
    whose own topology and measured normals/radii prove a chamfer-band role.

    The current locked-boundary rebuild path requires two compatible rails.  A
    two-loop patch with mismatched rail vertex counts can remain review evidence,
    but it must not be exposed as rebuildable CHAMFER ownership because rebuild
    cannot map the rails without leaving boundary edges.
    """

    ids = tuple_ints(face_ids)
    report = _face_patch_boundary_semantic_report_v174f(faces, ids)
    loop_count = int(report.get("boundary_loop_count", 0) or 0)
    component_count = int(report.get("component_count", 0) or 0)
    loop_vertices = tuple(int(v) for v in tuple(report.get("boundary_loop_vertex_counts", ()) or ()))
    loop_edges = tuple(int(v) for v in tuple(report.get("boundary_loop_edge_counts", ()) or ()))
    rail_vertex_delta = int(abs(loop_vertices[0] - loop_vertices[1])) if len(loop_vertices) >= 2 else 999999
    rail_edge_delta = int(abs(loop_edges[0] - loop_edges[1])) if len(loop_edges) >= 2 else 999999

    radial_span = _safe_float(component_stats.get("radial_span", 0.0), 0.0)
    axial_span = _safe_float(component_stats.get("axial_span", 0.0), 0.0)
    normal_axis = _safe_float(component_stats.get("normal_axis_abs_median", 0.0), 0.0)
    radial_align = _safe_float(component_stats.get("radial_normal_alignment_median", 0.0), 0.0)
    face_count = int(component_stats.get("face_count", len(ids)) or 0)

    reasons: list[str] = []
    if face_count <= 0 or not ids:
        reasons.append("chamfer_band_no_faces")
    if component_count != 1:
        reasons.append("chamfer_band_must_be_one_connected_surface_role")
    if loop_count != 2:
        reasons.append("chamfer_band_must_have_exactly_two_rail_boundary_loops")
    if loop_count == 2 and rail_vertex_delta != 0:
        reasons.append("chamfer_band_rail_vertex_counts_not_rebuild_compatible")
    if loop_count == 2 and rail_edge_delta != 0:
        reasons.append("chamfer_band_rail_edge_counts_not_rebuild_compatible")
    if radial_span <= 1.0e-9:
        reasons.append("chamfer_band_missing_radial_transition")
    if axial_span <= 1.0e-9:
        reasons.append("chamfer_band_missing_axial_transition")
    # Chamfer normals must contain both axial and radial components.  Very low
    # axial or radial support means this is more likely a bore wall, cap, floor,
    # or broad context surface than a transition band.
    if normal_axis < 0.15 or radial_align < 0.15:
        reasons.append("chamfer_band_normals_do_not_prove_transition_role")

    valid = bool(not reasons)
    return {
        "chamfer_band_role_proof_contract_v173c": CHAMFER_FAMILY_LOCAL_OWNERSHIP_CONTRACT_V174F,
        "chamfer_band_role_valid_v173c": bool(valid),
        "chamfer_band_role_rejection_reasons_v173c": tuple(reasons),
        "chamfer_band_boundary_loop_count_v173c": int(loop_count),
        "chamfer_band_component_count_v173c": int(component_count),
        "chamfer_band_loop_vertex_counts_v173c": loop_vertices,
        "chamfer_band_loop_edge_counts_v173c": loop_edges,
        "chamfer_band_rail_vertex_count_delta_v173c": int(rail_vertex_delta),
        "chamfer_band_rail_edge_count_delta_v173c": int(rail_edge_delta),
        "chamfer_band_semantic_rule_v173c": (
            "CHAMFER CandidateData is emitted only from independent CHAMFER_BAND surface-role proof; "
            "BORE mouth-transition or shared rail evidence remains relationship metadata."
        ),
        "chamfer_band_boundary_report_v173c": dict(report),
    }


def chamfer_selected_rail_local_band_ownership_proof_v174f(
    *,
    faces: np.ndarray,
    face_ids: Iterable[int],
    component_stats: Mapping[str, object],
    selected_radius: float,
    selected_confidence: float,
    selected_segment_count: int,
    edge_scale: float,
    radial_scale: float,
    axial_span_all: float,
    role_proof: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Prove selected-rail-local CHAMFER_BAND ownership before CandidateData.

    v173o deliberately does *not* reason about world/object position.  A chamfer
    close to an object corner, outer side, pocket, or bore mouth is still judged
    only by local transition-band evidence.  The proof is the semantic mapping
    requested for the selected rail:

        selected rail anchor
            -> local transition band component
            -> inner/outer rail relationship
            -> band normal transition
            -> axial/radial thickness
            -> angular support around selected rail
            -> component containment
            -> CHAMFER_BAND ownership

    The output is evidence/ownership diagnostics for Recognition.  It does not
    create a rebuild target and it does not inspect global feature position.
    """

    ids = tuple_ints(face_ids)
    rp = dict(role_proof or chamfer_band_role_proof_v174f(
        faces=faces,
        face_ids=ids,
        component_stats=component_stats,
    ))
    face_count = int(component_stats.get("face_count", len(ids)) or len(ids))
    component_count = int(rp.get("chamfer_band_component_count_v173c", 0) or 0)
    loop_count = int(rp.get("chamfer_band_boundary_loop_count_v173c", 0) or 0)
    loop_edges = tuple(int(v) for v in tuple(rp.get("chamfer_band_loop_edge_counts_v173c", ()) or ()))
    loop_vertices = tuple(int(v) for v in tuple(rp.get("chamfer_band_loop_vertex_counts_v173c", ()) or ()))
    rail_edge_delta = int(rp.get("chamfer_band_rail_edge_count_delta_v173c", 999999) or 999999)
    rail_vertex_delta = int(rp.get("chamfer_band_rail_vertex_count_delta_v173c", 999999) or 999999)

    radial_span = _safe_float(component_stats.get("radial_span", 0.0), 0.0)
    axial_span = _safe_float(component_stats.get("axial_span", 0.0), 0.0)
    radial_min = _safe_float(component_stats.get("radial_min", 0.0), 0.0)
    radial_max = _safe_float(component_stats.get("radial_max", radial_min), radial_min)
    radial_median = _safe_float(component_stats.get("radial_median", 0.0), 0.0)
    radial_mad = _safe_float(component_stats.get("radial_mad", 0.0), 0.0)
    normal_axis = _safe_float(component_stats.get("normal_axis_abs_median", 0.0), 0.0)
    radial_align = _safe_float(component_stats.get("radial_normal_alignment_median", 0.0), 0.0)
    seed_related = bool(component_stats.get("seed_related", False))

    seg_count = max(int(selected_segment_count or 0), 1)
    es = max(float(edge_scale), 1.0e-9)
    rs = max(float(radial_scale), float(selected_radius), 1.0)
    span_all = max(float(axial_span_all), 1.0e-9)

    min_transition_span = max(0.35 * es, 0.006 * rs, 0.04)
    max_radial_thickness = max(4.50 * es, 0.35 * max(float(selected_radius), 1.0), 0.18)
    max_axial_thickness = max(4.50 * es, 0.35 * max(float(selected_radius), 1.0), 0.18)
    max_axial_fraction = 0.22
    local_face_max = max(96, int(10 * seg_count))
    # A coarse selected rail can have two triangles per segment, and a good mesh
    # can have a denser but still local band.  This is a containment bound, not
    # a world-position special case.
    face_count_contained = bool(4 <= face_count <= local_face_max)

    selected_rail_anchor_valid = bool(float(selected_radius) > 1.0e-9 and float(selected_confidence) >= 0.20 and seg_count >= 3)
    local_transition_component_valid = bool(component_count == 1 and face_count > 0 and seed_related)
    rail_count_tolerance = max(2, int(round(0.18 * seg_count)))
    inner_outer_rail_relationship_valid = bool(
        loop_count == 2
        and len(loop_edges) >= 2
        and len(loop_vertices) >= 2
        and rail_edge_delta <= rail_count_tolerance
        and rail_vertex_delta <= rail_count_tolerance
    )
    band_normal_transition_valid = bool(normal_axis >= 0.10 and radial_align >= 0.10 and (normal_axis + radial_align) >= 0.32)
    radial_thickness_valid = bool(radial_span >= min_transition_span and radial_span <= max_radial_thickness)
    axial_thickness_valid = bool(axial_span >= min_transition_span and axial_span <= max_axial_thickness and axial_span <= max_axial_fraction * span_all)
    band_thickness_valid = bool(radial_thickness_valid and axial_thickness_valid)
    angular_support_valid = bool(face_count >= max(4, int(0.75 * seg_count)))
    component_containment_valid = bool(face_count_contained and local_transition_component_valid)
    radius_relation_valid = bool(
        float(selected_radius) <= 1.0e-9
        or radial_min <= float(selected_radius) + max(2.5 * es, 0.12 * max(float(selected_radius), 1.0))
        or radial_max >= float(selected_radius) - max(2.5 * es, 0.12 * max(float(selected_radius), 1.0))
    )

    reasons: list[str] = []
    if not selected_rail_anchor_valid:
        reasons.append("selected_rail_anchor_not_measured")
    if not local_transition_component_valid:
        reasons.append("local_transition_band_component_not_seed_related_or_not_single")
    if not inner_outer_rail_relationship_valid:
        reasons.append("inner_outer_rail_relationship_not_proven")
    if not band_normal_transition_valid:
        reasons.append("band_normals_do_not_prove_transition")
    if not radial_thickness_valid:
        reasons.append("radial_thickness_outside_local_transition_band")
    if not axial_thickness_valid:
        reasons.append("axial_thickness_outside_local_transition_band")
    if not angular_support_valid:
        reasons.append("angular_support_around_selected_rail_too_weak")
    if not component_containment_valid:
        reasons.append("component_containment_not_local_transition_band")
    if not radius_relation_valid:
        reasons.append("selected_rail_not_related_to_band_radial_interval")

    valid = bool(not reasons)
    return {
        "chamfer_band_ownership_proof_v173o": True,
        "chamfer_band_ownership_proof_valid_v173o": bool(valid),
        "chamfer_band_ownership_rejection_reasons_v173o": tuple(reasons),
        "chamfer_selected_rail_anchor_valid_v173o": bool(selected_rail_anchor_valid),
        "chamfer_selected_rail_anchor_edges_v173o": int(seg_count),
        "chamfer_selected_rail_radius_v173o": float(selected_radius),
        "chamfer_selected_rail_confidence_v173o": float(selected_confidence),
        "chamfer_local_transition_component_valid_v173o": bool(local_transition_component_valid),
        "chamfer_local_transition_component_count_v173o": int(component_count),
        "chamfer_inner_outer_rail_relationship_valid_v173o": bool(inner_outer_rail_relationship_valid),
        "chamfer_band_loop_edge_counts_v173o": loop_edges,
        "chamfer_band_loop_vertex_counts_v173o": loop_vertices,
        "chamfer_band_rail_edge_delta_v173o": int(rail_edge_delta),
        "chamfer_band_rail_vertex_delta_v173o": int(rail_vertex_delta),
        "chamfer_band_rail_count_tolerance_v173o": int(rail_count_tolerance),
        "chamfer_band_normal_transition_valid_v173o": bool(band_normal_transition_valid),
        "chamfer_band_normal_axis_abs_median_v173o": float(normal_axis),
        "chamfer_band_radial_normal_alignment_median_v173o": float(radial_align),
        "chamfer_band_thickness_valid_v173o": bool(band_thickness_valid),
        "chamfer_radial_thickness_valid_v173o": bool(radial_thickness_valid),
        "chamfer_axial_thickness_valid_v173o": bool(axial_thickness_valid),
        "chamfer_band_radial_span_v173o": float(radial_span),
        "chamfer_band_axial_span_v173o": float(axial_span),
        "chamfer_band_min_transition_span_v173o": float(min_transition_span),
        "chamfer_band_max_radial_thickness_v173o": float(max_radial_thickness),
        "chamfer_band_max_axial_thickness_v173o": float(max_axial_thickness),
        "chamfer_band_max_axial_fraction_v173o": float(max_axial_fraction),
        "chamfer_angular_support_valid_v173o": bool(angular_support_valid),
        "chamfer_component_containment_valid_v173o": bool(component_containment_valid),
        "chamfer_component_face_count_v173o": int(face_count),
        "chamfer_component_face_count_max_v173o": int(local_face_max),
        "chamfer_band_radius_relation_valid_v173o": bool(radius_relation_valid),
        "chamfer_band_radial_interval_v173o": (float(radial_min), float(radial_max)),
        "chamfer_band_radial_median_v173o": float(radial_median),
        "chamfer_band_radial_mad_v173o": float(radial_mad),
        "chamfer_rebuild_faces_source_v173o": "owned_local_transition_band_only",
        "chamfer_position_independent_semantics_v173o": True,
        "chamfer_position_independent_semantic_rule_v173o": (
            "Feature position is not a semantic category; CHAMFER ownership is proven only by selected-rail-local transition-band evidence."
        ),
        "chamfer_band_semantic_stage_v173o": "SelectedAnnularRailEvidence -> local_transition_band_component -> CHAMFER_BAND ownership -> CandidateData",
    }


def chamfer_legacy_local_transition_band_owned_subset_mapping_v174g(
    *,
    faces: np.ndarray,
    face_ids: Iterable[int],
    fid_to_local: Mapping[int, int],
    axial: np.ndarray,
    radial: np.ndarray,
    normal_axis_abs: np.ndarray,
    radial_normal_alignment: np.ndarray,
    seed_face_set: set[int],
    adjacency: Mapping[int, tuple[int, ...]],
    selected_radius: float,
    selected_confidence: float,
    selected_segment_count: int,
    edge_scale: float,
    radial_scale: float,
    axial_span_all: float,
    pre_component_stats: Mapping[str, object] | None = None,
    pre_role_proof: Mapping[str, object] | None = None,
    pre_ownership_proof: Mapping[str, object] | None = None,
    proof_evaluator: OwnedSubsetEvaluator,
) -> dict[str, object]:
    """Map a proposed CHAMFER evidence component to its owned face subset.

    v174g moves this quarantined legacy v173p fallback evaluator out of the
    component engine.  It remains a preserved fallback only; v173t rail-to-rail
    bounded CHAMFER_BAND ownership is still the primary export authority.

    v173o correctly introduced a selected-rail-local ownership proof, but it was
    intentionally all-or-nothing.  On coarse meshes a valid annular transition
    band can contain a small number of interfering faces.  Those faces should
    not force the entire CHAMFER candidate into review if a coherent owned
    transition-band core remains.

    v173p keeps the semantic rule strict: CandidateData may expose only the
    CHAMFER_BAND owned subset.  It does not add position semantics and it does
    not make object corners meaningful.  It simply separates:

        proposed transition evidence component
            -> owned_chamfer_band_faces
            -> rejected_interfering_faces / support-context faces
            -> revalidated owned CHAMFER_BAND CandidateData

    The helper returns diagnostics and the best owned subset.  If no cleaned
    subset can pass the v173o proof, the original component remains review-only.
    """

    original_ids = tuple_ints(face_ids)
    original_set = set(int(fid) for fid in original_ids)
    seg_count = max(int(selected_segment_count or 0), 1)
    es = max(float(edge_scale), 1.0e-9)
    min_owned_faces = max(4, int(0.75 * seg_count))

    def _evaluate(ids_in: Iterable[int]) -> tuple[tuple[int, ...], dict[str, object], dict[str, object], dict[str, object]]:
        ids_eval, st_eval, role_eval, proof_eval = proof_evaluator(ids_in)
        return tuple_ints(ids_eval), dict(st_eval or {}), dict(role_eval or {}), dict(proof_eval or {})

    if not original_ids:
        ids0, st0, role0, proof0 = _evaluate(())
        return {
            "owned_face_ids_v173p": (),
            "rejected_interfering_face_ids_v173p": (),
            "owned_component_stats_v173p": st0,
            "owned_role_proof_v173p": role0,
            "owned_ownership_proof_v173p": proof0,
            "chamfer_owned_subset_mapping_v173p": True,
            "chamfer_owned_subset_mapping_valid_v173p": False,
            "chamfer_owned_subset_mapping_reason_v173p": "empty_proposed_chamfer_component",
        }

    _ids0, eval_st0, eval_role0, eval_proof0 = _evaluate(original_ids)
    st0 = dict(pre_component_stats or eval_st0)
    role0 = dict(pre_role_proof or eval_role0)
    proof0 = dict(pre_ownership_proof or eval_proof0)
    if bool(proof0.get("chamfer_band_ownership_proof_valid_v173o", False)):
        return {
            "owned_face_ids_v173p": original_ids,
            "rejected_interfering_face_ids_v173p": (),
            "owned_component_stats_v173p": st0,
            "owned_role_proof_v173p": role0,
            "owned_ownership_proof_v173p": proof0,
            "chamfer_owned_subset_mapping_v173p": True,
            "chamfer_owned_subset_mapping_valid_v173p": True,
            "chamfer_owned_subset_mapping_reason_v173p": "original_component_already_valid",
            "chamfer_owned_subset_input_face_count_v173p": int(len(original_ids)),
            "chamfer_owned_subset_face_count_v173p": int(len(original_ids)),
            "chamfer_owned_subset_rejected_face_count_v173p": 0,
            "chamfer_owned_subset_revalidated_v173p": True,
        }

    valid_idx_pairs = [(int(fid), int(fid_to_local[int(fid)])) for fid in original_ids if int(fid) in fid_to_local]
    if len(valid_idx_pairs) < min_owned_faces:
        return {
            "owned_face_ids_v173p": original_ids,
            "rejected_interfering_face_ids_v173p": (),
            "owned_component_stats_v173p": st0,
            "owned_role_proof_v173p": role0,
            "owned_ownership_proof_v173p": proof0,
            "chamfer_owned_subset_mapping_v173p": True,
            "chamfer_owned_subset_mapping_valid_v173p": False,
            "chamfer_owned_subset_mapping_reason_v173p": "not_enough_faces_for_subset_purification",
            "chamfer_owned_subset_input_face_count_v173p": int(len(original_ids)),
            "chamfer_owned_subset_face_count_v173p": int(len(original_ids)),
            "chamfer_owned_subset_rejected_face_count_v173p": 0,
            "chamfer_owned_subset_pre_rejection_reasons_v173p": tuple(proof0.get("chamfer_band_ownership_rejection_reasons_v173o", ()) or ()),
        }

    ids_arr = [fid for fid, _ in valid_idx_pairs]
    idx_arr = np.asarray([idx for _, idx in valid_idx_pairs], dtype=np.int64)
    ax = np.asarray(axial[idx_arr], dtype=float)
    rd = np.asarray(radial[idx_arr], dtype=float)
    na = np.asarray(normal_axis_abs[idx_arr], dtype=float)
    ra = np.asarray(radial_normal_alignment[idx_arr], dtype=float)
    ax_med = float(np.median(ax)) if ax.size else 0.0
    rd_med = float(np.median(rd)) if rd.size else 0.0
    ax_mad = float(np.median(np.abs(ax - ax_med))) if ax.size else 0.0
    rd_mad = float(np.median(np.abs(rd - rd_med))) if rd.size else 0.0
    ax_scale = max(1.4826 * ax_mad, 0.45 * es, 0.035)
    rd_scale = max(1.4826 * rd_mad, 0.45 * es, 0.035)
    transition_floor = 0.26
    current_set = set(ids_arr)
    degrees: dict[int, int] = {}
    for fid in current_set:
        degrees[int(fid)] = sum(1 for nb in adjacency.get(int(fid), ()) if int(nb) in current_set)

    scores: list[tuple[float, int]] = []
    for pos, fid in enumerate(ids_arr):
        axial_dev = abs(float(ax[pos]) - ax_med) / ax_scale
        radial_dev = abs(float(rd[pos]) - rd_med) / rd_scale
        transition_strength = float(na[pos]) + float(ra[pos])
        weak_transition = max(0.0, transition_floor - transition_strength) / max(transition_floor, 1.0e-9)
        degree = int(degrees.get(int(fid), 0))
        spur_penalty = 1.35 if degree <= 1 else (0.35 if degree == 2 else 0.0)
        # Faces that are both geometric outliers and weakly attached are the
        # first ownership-mapping candidates.  This is not a world-position or
        # corner test; it is a local-band coherence test.
        score = 0.85 * radial_dev + 0.85 * axial_dev + 1.20 * weak_transition + spur_penalty
        scores.append((float(score), int(fid)))
    scores.sort(reverse=True)

    max_remove = max(1, min(6, int(max(2, round(0.12 * len(original_ids))))))
    candidate_pool = [fid for score, fid in scores[: min(10, len(scores))]]
    tried: set[tuple[int, ...]] = set()

    def _try_removed(remove_ids: Iterable[int], strategy: str) -> dict[str, object] | None:
        remove_tuple = tuple(sorted(int(fid) for fid in remove_ids))
        if remove_tuple in tried:
            return None
        tried.add(remove_tuple)
        remaining = tuple(int(fid) for fid in original_ids if int(fid) not in set(remove_tuple))
        if len(remaining) < min_owned_faces:
            return None
        comps = tuple(connected_face_components(np.asarray(faces, dtype=np.int64), remaining))
        if len(comps) != 1:
            # CandidateData for CHAMFER should be one contained transition band.
            return None
        ids_eval, st_eval, role_eval, proof_eval = _evaluate(remaining)
        if bool(proof_eval.get("chamfer_band_ownership_proof_valid_v173o", False)):
            rejected = tuple(int(fid) for fid in original_ids if int(fid) not in set(ids_eval))
            return {
                "owned_face_ids_v173p": ids_eval,
                "rejected_interfering_face_ids_v173p": rejected,
                "owned_component_stats_v173p": st_eval,
                "owned_role_proof_v173p": role_eval,
                "owned_ownership_proof_v173p": proof_eval,
                "chamfer_owned_subset_mapping_v173p": True,
                "chamfer_owned_subset_mapping_valid_v173p": True,
                "chamfer_owned_subset_mapping_reason_v173p": "cleaned_subset_revalidated",
                "chamfer_owned_subset_strategy_v173p": str(strategy),
                "chamfer_owned_subset_input_face_count_v173p": int(len(original_ids)),
                "chamfer_owned_subset_face_count_v173p": int(len(ids_eval)),
                "chamfer_owned_subset_rejected_face_count_v173p": int(len(rejected)),
                "chamfer_owned_subset_revalidated_v173p": True,
                "chamfer_owned_subset_pre_rejection_reasons_v173p": tuple(proof0.get("chamfer_band_ownership_rejection_reasons_v173o", ()) or ()),
                "chamfer_owned_subset_post_rejection_reasons_v173p": tuple(proof_eval.get("chamfer_band_ownership_rejection_reasons_v173o", ()) or ()),
            }
        return None

    # Greedy prefixes catch the common case: one or two spur/outlier faces.
    for k in range(1, max_remove + 1):
        result = _try_removed(candidate_pool[:k], f"greedy_top_{k}_local_band_outliers")
        if result is not None:
            return result

    # Exhaustive small-combination search over the worst local-band outliers.
    from itertools import combinations
    for k in range(1, min(3, max_remove) + 1):
        for combo in combinations(candidate_pool, k):
            result = _try_removed(combo, f"combination_{k}_of_top_local_band_outliers")
            if result is not None:
                return result

    return {
        "owned_face_ids_v173p": original_ids,
        "rejected_interfering_face_ids_v173p": (),
        "owned_component_stats_v173p": st0,
        "owned_role_proof_v173p": role0,
        "owned_ownership_proof_v173p": proof0,
        "chamfer_owned_subset_mapping_v173p": True,
        "chamfer_owned_subset_mapping_valid_v173p": False,
        "chamfer_owned_subset_mapping_reason_v173p": "no_revalidated_owned_subset_found",
        "chamfer_owned_subset_input_face_count_v173p": int(len(original_ids)),
        "chamfer_owned_subset_face_count_v173p": int(len(original_ids)),
        "chamfer_owned_subset_rejected_face_count_v173p": 0,
        "chamfer_owned_subset_candidate_pool_v173p": tuple(int(fid) for fid in candidate_pool[:10]),
        "chamfer_owned_subset_pre_rejection_reasons_v173p": tuple(proof0.get("chamfer_band_ownership_rejection_reasons_v173o", ()) or ()),
    }





def chamfer_legacy_rail_anchored_owned_band_coordinate_mapping_v174g(
    *,
    faces: np.ndarray,
    face_ids: Iterable[int],
    fid_to_local: Mapping[int, int],
    axial: np.ndarray,
    radial: np.ndarray,
    normal_axis_abs: np.ndarray,
    radial_normal_alignment: np.ndarray,
    seed_face_set: set[int],
    adjacency: Mapping[int, tuple[int, ...]],
    selected_radius: float,
    selected_confidence: float,
    selected_segment_count: int,
    edge_scale: float,
    radial_scale: float,
    axial_span_all: float,
    pre_component_stats: Mapping[str, object] | None = None,
    pre_role_proof: Mapping[str, object] | None = None,
    pre_ownership_proof: Mapping[str, object] | None = None,
    proof_evaluator: OwnedSubsetEvaluator,
) -> dict[str, object]:
    """Rail-anchored CHAMFER_BAND coordinate ownership mapping.

    v174g moves this quarantined legacy v173q/v173s fallback evaluator out of
    the component engine.  It remains a preserved fallback only; v173t
    rail-to-rail bounded CHAMFER_BAND ownership remains primary authority.

    v173p still required the cleaned subset to satisfy the same topology-heavy
    ownership proof that rejected the raw component.  On coarse meshes this can
    leave two interfering evidence faces in the preview while the whole chamfer
    is demoted to review.  v173q adds the missing semantic step: map the local
    transition evidence into the selected-rail band coordinate frame and export
    only the coherent owned band core.

    This helper is still position-independent.  It does not reason about object
    corners, global placement, or neighboring world geometry.  It measures only
    selected-rail-local band evidence: local coordinate containment, thin band
    thickness, normal transition behavior, angular support by expected segment
    count, and component containment where topology supports it.

    v173s correction: CHAMFER ownership is defined by the bounded annular
    slope between two rails, not by the raw connected evidence component and
    not by a face-count guess.  The count model can propose that a coarse
    two-triangle-per-selected-rail band should contain 2*N faces, but the
    semantic authority is boundary-contained annular-slope membership.  Faces
    outside that bounded slope remain evidence/context, and must not be exported
    as CandidateData display or rebuild faces.
    """

    original_ids = tuple_ints(face_ids)
    original_count = int(len(original_ids))
    # v173r: For selected-rail CHAMFER ownership, the support count is the
    # measured selected rail support, not a polygon-estimated segment count.
    # A coarse 21-edge rail can have an estimated 22-sided circle, but its
    # owned two-triangle CHAMFER band is still anchored to the 21 actual rail
    # support edges.  The estimated segment count is useful measurement
    # evidence; it must not become face-ownership authority.
    selected_support_count_v173r = max(int(selected_segment_count or 0), 1)
    seg_count = int(selected_support_count_v173r)
    es = max(float(edge_scale), 1.0e-9)
    rs = max(float(radial_scale), float(selected_radius), 1.0)
    span_all = max(float(axial_span_all), 1.0e-9)
    expected_two_tri_band = max(4, int(2 * selected_support_count_v173r))
    min_owned_faces = max(4, int(0.75 * selected_support_count_v173r))

    def _evaluate(ids_in: Iterable[int]) -> tuple[tuple[int, ...], dict[str, object], dict[str, object], dict[str, object]]:
        ids_eval, st_eval, role_eval, proof_eval = proof_evaluator(ids_in)
        return tuple_ints(ids_eval), dict(st_eval or {}), dict(role_eval or {}), dict(proof_eval or {})

    _ids0, eval_st0, eval_role0, eval_proof0 = _evaluate(original_ids)
    st0 = dict(pre_component_stats or eval_st0)
    role0 = dict(pre_role_proof or eval_role0)
    proof0 = dict(pre_ownership_proof or eval_proof0)

    valid_idx_pairs = [(int(fid), int(fid_to_local[int(fid)])) for fid in original_ids if int(fid) in fid_to_local]
    if len(valid_idx_pairs) < min_owned_faces:
        return {
            "owned_face_ids_v173q": original_ids,
            "rejected_interfering_face_ids_v173q": (),
            "owned_component_stats_v173q": st0,
            "owned_role_proof_v173q": role0,
            "owned_ownership_proof_v173q": proof0,
            "chamfer_coordinate_owned_band_mapping_v173q": True,
            "chamfer_coordinate_owned_band_valid_v173q": False,
            "chamfer_coordinate_owned_band_reason_v173q": "not_enough_faces_for_coordinate_mapping",
            "chamfer_coordinate_owned_band_input_face_count_v173q": int(original_count),
            "chamfer_coordinate_owned_band_face_count_v173q": int(original_count),
            "chamfer_coordinate_owned_band_rejected_face_count_v173q": 0,
            "chamfer_selected_rail_support_count_authority_v173r": int(selected_support_count_v173r),
            "chamfer_expected_owned_faces_from_selected_rail_support_v173r": int(expected_two_tri_band),
        }

    ids_arr = [fid for fid, _ in valid_idx_pairs]
    idx_arr = np.asarray([idx for _, idx in valid_idx_pairs], dtype=np.int64)
    ax = np.asarray(axial[idx_arr], dtype=float)
    rd = np.asarray(radial[idx_arr], dtype=float)
    na = np.asarray(normal_axis_abs[idx_arr], dtype=float)
    ra = np.asarray(radial_normal_alignment[idx_arr], dtype=float)
    ax_med = float(np.median(ax)) if ax.size else 0.0
    rd_med = float(np.median(rd)) if rd.size else 0.0
    ax_mad = float(np.median(np.abs(ax - ax_med))) if ax.size else 0.0
    rd_mad = float(np.median(np.abs(rd - rd_med))) if rd.size else 0.0
    ax_scale = max(1.4826 * ax_mad, 0.40 * es, 0.030)
    rd_scale = max(1.4826 * rd_mad, 0.40 * es, 0.030)
    current_set = {int(fid) for fid in ids_arr}
    degrees: dict[int, int] = {}
    seed_touch: dict[int, bool] = {}
    for fid in current_set:
        nb = tuple(int(v) for v in adjacency.get(int(fid), ()) or ())
        degrees[int(fid)] = sum(1 for item in nb if int(item) in current_set)
        seed_touch[int(fid)] = bool(int(fid) in seed_face_set or any(int(item) in seed_face_set for item in nb))

    scored: list[tuple[float, int]] = []
    for pos, fid in enumerate(ids_arr):
        axial_dev = abs(float(ax[pos]) - ax_med) / ax_scale
        radial_dev = abs(float(rd[pos]) - rd_med) / rd_scale
        transition_strength = float(na[pos]) + float(ra[pos])
        degree = int(degrees.get(int(fid), 0))
        locality_bonus = 0.35 if bool(seed_touch.get(int(fid), False)) else 0.0
        degree_bonus = 0.18 if degree >= 2 else (-0.45 if degree <= 1 else 0.0)
        # Higher score means stronger membership in the local annular transition
        # band core.  The score intentionally uses local band coordinates and
        # normal transition evidence, not world/object position.
        score = (
            1.15 * transition_strength
            - 0.62 * radial_dev
            - 0.62 * axial_dev
            + degree_bonus
            + locality_bonus
        )
        scored.append((float(score), int(fid)))
    scored.sort(reverse=True)

    # In the common coarse triangular chamfer case, the owned band has two
    # triangles per selected rail segment.  If the raw evidence component has a
    # few extra faces, keep the strongest expected band core.  If the raw count
    # is not near that model, keep the raw count and only use coordinate proof as
    # diagnostics; broad contamination must not be converted into actionability.
    near_two_tri_count = bool(original_count >= expected_two_tri_band and original_count <= expected_two_tri_band + max(3, int(0.20 * seg_count)))
    target_count = int(expected_two_tri_band if near_two_tri_count else original_count)
    target_count = max(min_owned_faces, min(int(target_count), int(len(scored))))
    core_ids = tuple(sorted(int(fid) for _score, fid in scored[:target_count]))
    rejected = tuple(sorted(int(fid) for fid in original_ids if int(fid) not in set(core_ids)))

    ids_core, st_core, role_core, proof_core = _evaluate(core_ids)
    radial_span = _safe_float(st_core.get("radial_span", 0.0), 0.0)
    axial_span = _safe_float(st_core.get("axial_span", 0.0), 0.0)
    normal_axis = _safe_float(st_core.get("normal_axis_abs_median", 0.0), 0.0)
    radial_align = _safe_float(st_core.get("radial_normal_alignment_median", 0.0), 0.0)
    component_count = int((role_core or {}).get("chamfer_band_component_count_v173c", 0) or 0)

    min_transition_span = max(0.25 * es, 0.004 * rs, 0.025)
    max_radial_thickness = max(5.0 * es, 0.38 * max(float(selected_radius), 1.0), 0.20)
    max_axial_thickness = max(5.0 * es, 0.38 * max(float(selected_radius), 1.0), 0.20)
    selected_anchor_valid = bool(float(selected_radius) > 1.0e-9 and float(selected_confidence) >= 0.20 and seg_count >= 3)
    count_model_valid = bool((not near_two_tri_count) or len(ids_core) == expected_two_tri_band)
    coordinate_thickness_valid = bool(
        radial_span >= min_transition_span
        and radial_span <= max_radial_thickness
        and axial_span >= min_transition_span
        and axial_span <= max_axial_thickness
        and axial_span <= 0.24 * span_all
    )
    normal_transition_valid = bool(normal_axis >= 0.08 and radial_align >= 0.08 and (normal_axis + radial_align) >= 0.24)
    local_support_valid = bool(len(ids_core) >= min_owned_faces and len(ids_core) >= int(0.75 * seg_count))
    # Prefer a single connected patch, but allow a rail-coordinate band whose
    # cleaned core was split by removing a tiny number of interfering faces.  The
    # rebuild still validates exact CandidateData faces before committing.
    containment_valid = bool(component_count == 1 or (component_count <= 2 and len(rejected) <= max(2, int(0.10 * original_count))))
    small_clean_rejection_valid = bool(len(rejected) <= max(3, int(0.20 * seg_count)))

    # v173s: physical/mathematical CHAMFER ownership is bounded annular slope
    # containment.  The raw evidence component may include faces outside the
    # slope rails.  Those faces can be connected and angled, but they are not
    # owned CHAMFER_BAND faces.  Use the rail-count model only to propose a
    # contained core, then accept the core when its measured axial/radial
    # interval is a thin local transition band around the selected rail.
    boundary_contained_slope_valid_v173s = bool(
        selected_anchor_valid
        and near_two_tri_count
        and len(ids_core) == expected_two_tri_band
        and len(rejected) > 0
        and small_clean_rejection_valid
        and local_support_valid
        and radial_span >= min_transition_span
        and axial_span >= min_transition_span
        and radial_span <= max_radial_thickness
        and axial_span <= max_axial_thickness
        and axial_span <= 0.24 * span_all
        and (normal_transition_valid or (normal_axis + radial_align) >= 0.20)
        and containment_valid
    )
    coordinate_valid = bool(
        (
            selected_anchor_valid
            and count_model_valid
            and coordinate_thickness_valid
            and normal_transition_valid
            and local_support_valid
            and containment_valid
            and small_clean_rejection_valid
        )
        or boundary_contained_slope_valid_v173s
    )

    return {
        "owned_face_ids_v173q": ids_core,
        "rejected_interfering_face_ids_v173q": rejected,
        "owned_component_stats_v173q": st_core,
        "owned_role_proof_v173q": role_core,
        "owned_ownership_proof_v173q": proof_core,
        "chamfer_coordinate_owned_band_mapping_v173q": True,
        "chamfer_coordinate_owned_band_valid_v173q": bool(coordinate_valid),
        "chamfer_coordinate_owned_band_reason_v173q": "rail_anchored_coordinate_core_valid" if coordinate_valid else "rail_anchored_coordinate_core_not_valid",
        "chamfer_coordinate_owned_band_input_face_count_v173q": int(original_count),
        "chamfer_coordinate_owned_band_expected_two_triangle_face_count_v173q": int(expected_two_tri_band),
        "chamfer_coordinate_owned_band_face_count_v173q": int(len(ids_core)),
        "chamfer_coordinate_owned_band_rejected_face_count_v173q": int(len(rejected)),
        "chamfer_boundary_contained_annular_slope_ownership_v173s": True,
        "chamfer_boundary_contained_annular_slope_valid_v173s": bool(boundary_contained_slope_valid_v173s),
        "chamfer_boundary_contained_annular_slope_input_face_count_v173s": int(original_count),
        "chamfer_boundary_contained_annular_slope_owned_face_count_v173s": int(len(ids_core)),
        "chamfer_boundary_contained_annular_slope_context_face_count_v173s": int(len(rejected)),
        "chamfer_boundary_contained_annular_slope_rule_v173s": "owned CHAMFER_BAND faces must lie inside the bounded annular slope between the measured rails; raw connected evidence outside the slope is diagnostic context only",
        "chamfer_boundary_contained_annular_slope_face_export_v173s": "CandidateData display/rebuild faces use bounded annular slope owned core only when v173s validates",
        "chamfer_selected_rail_support_count_authority_v173r": int(selected_support_count_v173r),
        "chamfer_expected_owned_faces_from_selected_rail_support_v173r": int(expected_two_tri_band),
        "chamfer_count_authority_rule_v173r": "selected rail support edge count, not polygon-estimated segment count",
        "chamfer_coordinate_owned_band_rejected_face_ids_sample_v173q": tuple(int(fid) for fid in rejected[:12]),
        "chamfer_coordinate_owned_band_near_two_triangle_count_model_v173q": bool(near_two_tri_count),
        "chamfer_coordinate_owned_band_selected_anchor_valid_v173q": bool(selected_anchor_valid),
        "chamfer_coordinate_owned_band_count_model_valid_v173q": bool(count_model_valid),
        "chamfer_coordinate_owned_band_thickness_valid_v173q": bool(coordinate_thickness_valid),
        "chamfer_coordinate_owned_band_normal_transition_valid_v173q": bool(normal_transition_valid),
        "chamfer_coordinate_owned_band_local_support_valid_v173q": bool(local_support_valid),
        "chamfer_coordinate_owned_band_containment_valid_v173q": bool(containment_valid),
        "chamfer_coordinate_owned_band_small_clean_rejection_valid_v173q": bool(small_clean_rejection_valid),
        "chamfer_coordinate_owned_band_component_count_v173q": int(component_count),
        "chamfer_coordinate_owned_band_radial_span_v173q": float(radial_span),
        "chamfer_coordinate_owned_band_axial_span_v173q": float(axial_span),
        "chamfer_coordinate_owned_band_normal_axis_abs_median_v173q": float(normal_axis),
        "chamfer_coordinate_owned_band_radial_normal_alignment_median_v173q": float(radial_align),
        "chamfer_coordinate_owned_band_semantic_rule_v173q": (
            "CandidateData display/rebuild faces are the rail-anchored CHAMFER_BAND core; "
            "raw transition evidence faces rejected by coordinate ownership remain diagnostic context only."
        ),
    }




def chamfer_rail_to_rail_bounded_surface_ownership_v174e(
    *,
    faces: np.ndarray,
    vertices: np.ndarray,
    face_ids: Iterable[int],
    fid_to_local: Mapping[int, int],
    axial: np.ndarray,
    radial: np.ndarray,
    normal_axis_abs: np.ndarray,
    radial_normal_alignment: np.ndarray,
    seed_face_set: set[int],
    adjacency: Mapping[int, tuple[int, ...]],
    frame: object,
    selected_support_edge_count: int,
    edge_scale: float,
    radial_scale: float,
    axial_span_all: float,
    evaluate_owned_subset: OwnedSubsetEvaluator,
) -> dict[str, object]:
    """Return CHAMFER_BAND faces by the physical rail-to-rail definition.

    A CHAMFER is not an arbitrary angled connected component.  In the annular
    case it is the bounded sloped transition surface between two measured rails.
    This helper makes that the export authority for CandidateData:

        raw transition evidence faces
            -> boundary rails of the proposed annular strip
            -> 2D rail-to-rail slope line in (radius, axial) space
            -> per-face containment in that bounded slope
            -> owned CHAMFER_BAND faces only

    Faces outside that bounded rail-to-rail slope are diagnostic context, even
    if they touch the raw component, look angled, or live close to the selected
    rail.  This deliberately replaces the failed v173o/p/q/r/s authority cascade
    for selected-rail CHAMFER CandidateData export.
    """

    original_ids = tuple_ints(face_ids)
    original_set = {int(fid) for fid in original_ids}
    support_count = max(int(selected_support_edge_count or 0), 1)
    es = max(float(edge_scale), 1.0e-9)
    span_all = max(float(axial_span_all), 1.0e-9)
    expected_two_tri_band = max(4, int(2 * support_count))
    min_owned_faces = max(4, int(0.65 * support_count))

    if not original_ids:
        ids0, st0, role0, proof0 = evaluate_owned_subset(())
        return {
            "owned_face_ids_v173t": (),
            "context_face_ids_v173t": (),
            "owned_component_stats_v173t": st0,
            "owned_role_proof_v173t": role0,
            "owned_ownership_proof_v173t": proof0,
            "chamfer_rail_to_rail_bounded_surface_ownership_v173t": True,
            "chamfer_rail_to_rail_bounded_surface_valid_v173t": False,
            "chamfer_rail_to_rail_bounded_surface_reason_v173t": "empty_raw_transition_evidence",
        }

    try:
        boundary = boundary_edges_for_face_patch(faces, original_ids)
        loops = edge_loop_components(boundary)
    except Exception as exc:
        ids0, st0, role0, proof0 = evaluate_owned_subset(original_ids)
        return {
            "owned_face_ids_v173t": original_ids,
            "context_face_ids_v173t": (),
            "owned_component_stats_v173t": st0,
            "owned_role_proof_v173t": role0,
            "owned_ownership_proof_v173t": proof0,
            "chamfer_rail_to_rail_bounded_surface_ownership_v173t": True,
            "chamfer_rail_to_rail_bounded_surface_valid_v173t": False,
            "chamfer_rail_to_rail_bounded_surface_reason_v173t": "boundary_loop_extraction_failed",
            "chamfer_rail_to_rail_boundary_error_v173t": str(exc),
        }

    verts_arr = np.asarray(vertices, dtype=float)
    face_arr = np.asarray(faces, dtype=np.int64)

    loop_rows: list[dict[str, object]] = []
    for idx, loop in enumerate(loops):
        vertex_ids = tuple(sorted({int(v) for edge in loop for v in tuple(edge)[:2]}))
        valid_vertices = tuple(v for v in vertex_ids if 0 <= int(v) < len(verts_arr))
        if not valid_vertices:
            continue
        ax_l, rd_l = _project_points_to_frame_v174e(points=verts_arr[np.asarray(valid_vertices, dtype=np.int64), :3], frame=frame)
        if ax_l.size == 0 or rd_l.size == 0:
            continue
        loop_rows.append({
            "index": int(idx),
            "edge_count": int(len(loop)),
            "vertex_count": int(len(valid_vertices)),
            "axial_median": float(np.median(ax_l)),
            "radial_median": float(np.median(rd_l)),
            "axial_min": float(np.min(ax_l)),
            "axial_max": float(np.max(ax_l)),
            "radial_min": float(np.min(rd_l)),
            "radial_max": float(np.max(rd_l)),
            "radius_delta_to_frame": float(abs(float(np.median(rd_l)) - float(frame.radius))),
        })

    if len(loop_rows) >= 2:
        # The two rails of an annular chamfer are the two strongest boundary
        # loops of the raw transition strip.  If more loops exist, smaller loops
        # are diagnostic/context and do not become rail authority.
        loop_rows_sorted = sorted(loop_rows, key=lambda r: (-int(r.get("edge_count", 0)), float(r.get("radius_delta_to_frame", 0.0))))
        rail_a = loop_rows_sorted[0]
        rail_b = loop_rows_sorted[1]
    else:
        rail_a = None
        rail_b = None

    scored: list[tuple[float, int, dict[str, float]]] = []
    if rail_a is not None and rail_b is not None:
        p0 = np.asarray([float(rail_a["radial_median"]), float(rail_a["axial_median"])], dtype=float)
        p1 = np.asarray([float(rail_b["radial_median"]), float(rail_b["axial_median"])], dtype=float)
    else:
        # Fallback uses the median face cloud as a diagnostic frame only.  It can
        # produce a preview core, but not a rebuild-authorized result.
        idx_all = np.asarray([fid_to_local[int(fid)] for fid in original_ids if int(fid) in fid_to_local], dtype=np.int64)
        if idx_all.size:
            p0 = np.asarray([float(np.percentile(radial[idx_all], 15.0)), float(np.percentile(axial[idx_all], 15.0))], dtype=float)
            p1 = np.asarray([float(np.percentile(radial[idx_all], 85.0)), float(np.percentile(axial[idx_all], 85.0))], dtype=float)
        else:
            p0 = np.zeros((2,), dtype=float)
            p1 = np.asarray([1.0, 0.0], dtype=float)

    d = p1 - p0
    d2 = float(np.dot(d, d))
    dlen = math.sqrt(d2) if d2 > 1.0e-18 else 0.0
    if dlen <= 1.0e-12:
        d = np.asarray([1.0, 0.0], dtype=float)
        d2 = 1.0
        dlen = 1.0

    # Compute raw residual distribution to adapt to coarse remeshed chamfer bands.
    residuals: list[float] = []
    face_rows: list[tuple[int, float, float, float, float, float, int]] = []
    for fid in original_ids:
        if int(fid) not in fid_to_local or not (0 <= int(fid) < len(face_arr)):
            continue
        local_idx = int(fid_to_local[int(fid)])
        p = np.asarray([float(radial[local_idx]), float(axial[local_idx])], dtype=float)
        t = float(np.dot(p - p0, d) / d2)
        closest = p0 + t * d
        residual = float(np.linalg.norm(p - closest))
        residuals.append(residual)
        transition_strength = float(normal_axis_abs[local_idx]) + float(radial_normal_alignment[local_idx])
        degree = sum(1 for nb in adjacency.get(int(fid), ()) if int(nb) in original_set)
        face_rows.append((int(fid), float(t), float(residual), float(transition_strength), float(normal_axis_abs[local_idx]), float(radial_normal_alignment[local_idx]), int(degree)))

    if not face_rows:
        ids0, st0, role0, proof0 = evaluate_owned_subset(original_ids)
        return {
            "owned_face_ids_v173t": original_ids,
            "context_face_ids_v173t": (),
            "owned_component_stats_v173t": st0,
            "owned_role_proof_v173t": role0,
            "owned_ownership_proof_v173t": proof0,
            "chamfer_rail_to_rail_bounded_surface_ownership_v173t": True,
            "chamfer_rail_to_rail_bounded_surface_valid_v173t": False,
            "chamfer_rail_to_rail_bounded_surface_reason_v173t": "no_projectable_faces",
        }

    res_arr = np.asarray(residuals, dtype=float)
    res_med = float(np.median(res_arr)) if res_arr.size else 0.0
    res_mad = float(np.median(np.abs(res_arr - res_med))) if res_arr.size else 0.0
    residual_limit = max(res_med + 2.75 * 1.4826 * res_mad, 0.18 * max(dlen, 1.0e-9), 0.28 * es, 0.025)
    t_tol = max(0.16, min(0.34, 1.15 * es / max(dlen, 1.0e-9)))

    scored_faces: list[tuple[float, int]] = []
    contained_by_measurement: list[int] = []
    for fid, t, residual, transition_strength, na, ra, degree in face_rows:
        t_out = max(0.0, -t - t_tol, t - (1.0 + t_tol))
        residual_out = max(0.0, residual - residual_limit) / max(residual_limit, 1.0e-9)
        weak_transition = max(0.0, 0.20 - transition_strength) / 0.20
        spur = 0.35 if degree <= 1 else 0.0
        # Lower score means stronger rail-to-rail bounded-slope membership.
        score = 2.5 * t_out + 2.0 * residual_out + 0.75 * weak_transition + spur + 0.05 * residual / max(residual_limit, 1.0e-9)
        scored_faces.append((float(score), int(fid)))
        if t_out <= 1.0e-9 and residual <= residual_limit and transition_strength >= 0.16:
            contained_by_measurement.append(int(fid))

    scored_faces.sort(key=lambda item: (item[0], item[1]))

    raw_count = int(len(original_ids))
    near_expected_band = bool(
        raw_count >= expected_two_tri_band
        and raw_count <= expected_two_tri_band + max(4, int(0.25 * support_count))
    )
    # Rail-to-rail containment is primary.  The selected-support count is only a
    # coarse mesh cardinality sanity check when the raw patch has a tiny number
    # of excess faces.  It must never promote a broad raw evidence component.
    if near_expected_band and expected_two_tri_band >= min_owned_faces:
        target_count = min(int(expected_two_tri_band), len(scored_faces))
        owned_ids = tuple(sorted(int(fid) for _score, fid in scored_faces[:target_count]))
    else:
        owned_ids = tuple(sorted(contained_by_measurement))
        if len(owned_ids) < min_owned_faces:
            # Keep the best local core as review display if the physical proof is
            # incomplete.  This avoids showing raw outside-slope evidence as the
            # CHAMFER candidate while still making the diagnostic visible.
            target_count = max(min_owned_faces, min(len(scored_faces), expected_two_tri_band if expected_two_tri_band <= len(scored_faces) else len(scored_faces)))
            owned_ids = tuple(sorted(int(fid) for _score, fid in scored_faces[:target_count]))

    context_ids = tuple(sorted(int(fid) for fid in original_ids if int(fid) not in set(owned_ids)))
    ids_owned, st_owned, role_owned, proof_owned = evaluate_owned_subset(owned_ids)
    radial_span = _safe_float(st_owned.get("radial_span", 0.0), 0.0)
    axial_span = _safe_float(st_owned.get("axial_span", 0.0), 0.0)
    normal_axis = _safe_float(st_owned.get("normal_axis_abs_median", 0.0), 0.0)
    radial_align = _safe_float(st_owned.get("radial_normal_alignment_median", 0.0), 0.0)
    component_count = int((role_owned or {}).get("chamfer_band_component_count_v173c", 0) or 0)
    small_context = bool(len(context_ids) <= max(4, int(0.25 * support_count)))
    rail_pair_valid = bool(rail_a is not None and rail_b is not None)
    physical_valid = bool(
        rail_pair_valid
        and len(ids_owned) >= min_owned_faces
        and small_context
        and radial_span > max(0.02, 0.15 * es)
        and axial_span > max(0.02, 0.15 * es)
        and axial_span <= 0.28 * span_all
        and (normal_axis + radial_align) >= 0.20
        and component_count <= 2
    )

    return {
        "owned_face_ids_v173t": ids_owned,
        "context_face_ids_v173t": context_ids,
        "owned_component_stats_v173t": st_owned,
        "owned_role_proof_v173t": role_owned,
        "owned_ownership_proof_v173t": proof_owned,
        "chamfer_rail_to_rail_bounded_surface_ownership_v173t": True,
        "chamfer_rail_to_rail_bounded_surface_valid_v173t": bool(physical_valid),
        "chamfer_rail_to_rail_bounded_surface_reason_v173t": "rail_to_rail_bounded_slope_valid" if physical_valid else "rail_to_rail_bounded_slope_review_only",
        "chamfer_rail_to_rail_raw_face_count_v173t": int(raw_count),
        "chamfer_rail_to_rail_owned_face_count_v173t": int(len(ids_owned)),
        "chamfer_rail_to_rail_context_face_count_v173t": int(len(context_ids)),
        "chamfer_rail_to_rail_context_face_ids_sample_v173t": tuple(int(fid) for fid in context_ids[:12]),
        "chamfer_rail_to_rail_selected_support_edge_count_v173t": int(support_count),
        "chamfer_rail_to_rail_expected_two_triangle_face_count_v173t": int(expected_two_tri_band),
        "chamfer_rail_to_rail_boundary_loop_count_v173t": int(len(loops)),
        "chamfer_rail_to_rail_boundary_loop_summaries_v173t": tuple(loop_rows[:6]),
        "chamfer_rail_to_rail_rail_pair_valid_v173t": bool(rail_pair_valid),
        "chamfer_rail_to_rail_slope_point0_v173t": (float(p0[0]), float(p0[1])),
        "chamfer_rail_to_rail_slope_point1_v173t": (float(p1[0]), float(p1[1])),
        "chamfer_rail_to_rail_slope_length_v173t": float(dlen),
        "chamfer_rail_to_rail_residual_median_v173t": float(res_med),
        "chamfer_rail_to_rail_residual_mad_v173t": float(res_mad),
        "chamfer_rail_to_rail_residual_limit_v173t": float(residual_limit),
        "chamfer_rail_to_rail_tolerance_v173t": float(t_tol),
        "chamfer_rail_to_rail_near_expected_band_v173t": bool(near_expected_band),
        "chamfer_candidate_face_export_rule_v173t": "display_face_ids_and_rebuild_face_ids_use_rail_to_rail_owned_chamfer_band_faces_not_raw_transition_evidence",
        "chamfer_physical_definition_v173t": "annular CHAMFER = bounded sloped transition surface between two measured rails",
        "chamfer_failed_patch_cascade_quarantined_v173t": True,
    }




def _chamfer_export_status_v174y(export_authority: str) -> str:
    """Return an explicit status label for the CHAMFER export path."""

    authority = str(export_authority or "").strip()
    if authority == "v173t_rail_to_rail_bounded_surface_ownership":
        return "primary_v173t_owned_chamfer_band_export"
    if authority == "legacy_v173q_coordinate_owned_band_fallback_preserved":
        return "quarantined_legacy_v173q_fallback_export_preserved"
    if authority == "legacy_v173p_owned_subset_fallback_preserved":
        return "quarantined_legacy_v173p_fallback_export_preserved"
    return "no_chamfer_candidate_export_authority"


def select_chamfer_candidate_export_faces_v174y(
    *,
    raw_face_ids: Iterable[int],
    owned_subset_v173p: Mapping[str, object],
    owned_coordinate_v173q: Mapping[str, object],
    owned_rail_to_rail_v173t: Mapping[str, object],
    fallback_component_stats: Mapping[str, object],
    fallback_role_proof: Mapping[str, object],
    fallback_ownership_proof: Mapping[str, object],
) -> dict[str, object]:
    """Choose the CHAMFER CandidateData face export using existing v173 behavior.

    v174y moves this CHAMFER-local authority seam out of the component-engine
    orchestrator without changing the decision tree. v173t rail-to-rail
    ownership remains the preferred CHAMFER_BAND export. v173q/v173p are
    retained only in the same fallback positions they occupied before v174y.
    """

    raw_ids = tuple_ints(raw_face_ids)
    use_rail_to_rail_owned_subset_v173t = bool(tuple_ints(owned_rail_to_rail_v173t.get("owned_face_ids_v173t", ())))
    use_coordinate_owned_subset_v173q = bool(
        not use_rail_to_rail_owned_subset_v173t
        and not bool(owned_subset_v173p.get("chamfer_owned_subset_mapping_valid_v173p", False))
        and bool(owned_coordinate_v173q.get("chamfer_coordinate_owned_band_valid_v173q", False))
    )

    if use_rail_to_rail_owned_subset_v173t:
        candidate_ids = tuple_ints(owned_rail_to_rail_v173t.get("owned_face_ids_v173t", ()))
        component_stats = dict(owned_rail_to_rail_v173t.get("owned_component_stats_v173t", fallback_component_stats) or fallback_component_stats)
        role_proof = dict(owned_rail_to_rail_v173t.get("owned_role_proof_v173t", fallback_role_proof) or fallback_role_proof)
        ownership_proof = dict(owned_rail_to_rail_v173t.get("owned_ownership_proof_v173t", fallback_ownership_proof) or fallback_ownership_proof)
        export_authority = "v173t_rail_to_rail_bounded_surface_ownership"
    elif use_coordinate_owned_subset_v173q:
        candidate_ids = tuple_ints(owned_coordinate_v173q.get("owned_face_ids_v173q", ()))
        component_stats = dict(owned_coordinate_v173q.get("owned_component_stats_v173q", fallback_component_stats) or fallback_component_stats)
        role_proof = dict(owned_coordinate_v173q.get("owned_role_proof_v173q", fallback_role_proof) or fallback_role_proof)
        ownership_proof = dict(owned_coordinate_v173q.get("owned_ownership_proof_v173q", fallback_ownership_proof) or fallback_ownership_proof)
        export_authority = "legacy_v173q_coordinate_owned_band_fallback_preserved"
    else:
        candidate_ids = tuple_ints(owned_subset_v173p.get("owned_face_ids_v173p", ()))
        if not candidate_ids or set(candidate_ids) == set(raw_ids):
            # Preserve the pre-v174y behavior exactly: if no proven owned subset
            # exists, keep only any already-measured rail-to-rail subset instead
            # of exporting the raw connected component.
            candidate_ids = tuple_ints(owned_rail_to_rail_v173t.get("owned_face_ids_v173t", ())) or ()
        component_stats = dict(owned_subset_v173p.get("owned_component_stats_v173p", fallback_component_stats) or fallback_component_stats)
        role_proof = dict(owned_subset_v173p.get("owned_role_proof_v173p", fallback_role_proof) or fallback_role_proof)
        ownership_proof = dict(owned_subset_v173p.get("owned_ownership_proof_v173p", fallback_ownership_proof) or fallback_ownership_proof)
        export_authority = "legacy_v173p_owned_subset_fallback_preserved"

    return {
        "candidate_ids": candidate_ids,
        "component_stats": component_stats,
        "role_proof": role_proof,
        "ownership_proof": ownership_proof,
        "use_rail_to_rail_owned_subset_v173t": bool(use_rail_to_rail_owned_subset_v173t),
        "use_coordinate_owned_subset_v173q": bool(use_coordinate_owned_subset_v173q),
        "export_authority": export_authority,
        "v174c_chamfer_export_status": _chamfer_export_status_v174y(export_authority),
        "v174c_chamfer_primary_authority_v173t": bool(export_authority == "v173t_rail_to_rail_bounded_surface_ownership"),
        "v174c_chamfer_legacy_fallback_export_used": bool(export_authority in {
            "legacy_v173q_coordinate_owned_band_fallback_preserved",
            "legacy_v173p_owned_subset_fallback_preserved",
        }),
        "v174c_chamfer_fallback_quarantine_contract": True,
        "v174b_chamfer_export_selection_helper_used": True,
        "v174b_no_behavior_change_contract": True,
        "v174c_no_behavior_change_contract": True,
        "v174y_chamfer_export_selection_in_family_module": True,
    }


def assemble_chamfer_candidate_rows_v174y(
    *,
    faces: np.ndarray,
    vertices: np.ndarray,
    components: Iterable[Iterable[int]],
    fid_to_local: Mapping[int, int],
    axial: np.ndarray,
    radial: np.ndarray,
    normal_axis_abs: np.ndarray,
    radial_normal_alignment: np.ndarray,
    seed_face_set: Iterable[int],
    selected_seed_face_set: Iterable[int],
    adjacency: Mapping[int, Iterable[int]],
    selected_annular_rail_diagnostics: Mapping[str, object],
    selected_opening_frame_resolver: Mapping[str, object],
    selected_edge_ids: Iterable[int],
    edge_scale: float,
    radial_scale: float,
    axial_span_all: float,
    min_faces_general: int,
    min_radial_span: float,
    min_axial_span: float,
    region_face_count: int,
    bore_resolution_active: bool,
    region_frame: object,
    bore_mouth_transition_face_ids: Iterable[int],
    bore_mouth_transition_diagnostics: Mapping[str, object],
    bore_radius: float,
    bore_depth: float,
) -> dict[str, object]:
    """Assemble CHAMFER CandidateData rows using the existing v173/v174 rules.

    v174y moves the CHAMFER-local candidate row assembly out of
    ``recognition_component_engine.py``.  The component engine still owns the
    final multi-family merge.  This helper intentionally preserves the existing
    decision order: v173t rail-to-rail ownership first, then quarantined v173q
    and v173p fallback paths only when the preferred authority has no owned
    subset.
    """

    chamfer_rows: list[tuple[float, tuple[int, ...], dict[str, object]]] = []
    chamfer_rejected: list[dict[str, object]] = []
    chamfer_row_face_sets_v173d: set[frozenset[int]] = set()
    seed_face_ids = set(int(v) for v in tuple_ints(seed_face_set))
    selected_seed_face_ids = set(int(v) for v in tuple_ints(selected_seed_face_set))
    selected_ids = tuple_ints(selected_edge_ids)

    for rail_row in tuple(selected_annular_rail_diagnostics.get("selected_annular_rail_chamfer_candidate_rows_v173d", ()) or ()):  # type: ignore[arg-type]
        if not isinstance(rail_row, Mapping):
            continue
        comp_ids = tuple_ints(rail_row.get("face_ids", ()))
        if not comp_ids:
            continue
        key = frozenset(int(v) for v in comp_ids)
        if key in chamfer_row_face_sets_v173d:
            continue
        st = dict(rail_row.get("stats", {}) or {})
        score = _safe_float(rail_row.get("score", st.get("score", 0.0)), 0.0)
        chamfer_rows.append((float(score), comp_ids, st))
        chamfer_row_face_sets_v173d.add(key)

    for comp in components:
        comp_ids_for_stats = tuple_ints(comp)
        st = _component_stats_v174y(
            comp=comp_ids_for_stats,
            fid_to_local=fid_to_local,
            axial=axial,
            radial=radial,
            normal_axis_abs=normal_axis_abs,
            radial_normal_alignment=radial_normal_alignment,
            seed_face_set=seed_face_ids,
            adjacency=adjacency,
        )
        face_count = int(st.get("face_count", 0) or 0)
        radial_span = _safe_float(st.get("radial_span", 0.0), 0.0)
        axial_span = _safe_float(st.get("axial_span", 0.0), 0.0)
        normal_axis = _safe_float(st.get("normal_axis_abs_median", 0.0), 0.0)
        radial_align = _safe_float(st.get("radial_normal_alignment_median", 0.0), 0.0)
        accept_evidence = bool(
            face_count >= (4 if bool(st.get("seed_related", False)) else int(min_faces_general))
            and radial_span >= (0.55 * float(min_radial_span) if bool(st.get("seed_related", False)) else float(min_radial_span))
            and axial_span >= (0.55 * float(min_axial_span) if bool(st.get("seed_related", False)) else float(min_axial_span))
            and normal_axis >= 0.15
            and radial_align >= 0.15
        )
        score = _chamfer_score_v174y(st, radius_scale=max(float(radial_scale), 1.0), axial_span_all=max(float(axial_span_all), 1.0e-9))
        chamfer_region_fraction = float(face_count) / max(float(region_face_count), 1.0)
        chamfer_too_large_for_annular_transition = bool(chamfer_region_fraction > 0.45)
        role_proof = chamfer_band_role_proof_v174f(
            faces=faces,
            face_ids=comp_ids_for_stats,
            component_stats=st,
        )
        ownership_proof_v173o = chamfer_selected_rail_local_band_ownership_proof_v174f(
            faces=faces,
            face_ids=comp_ids_for_stats,
            component_stats=st,
            selected_radius=float(_safe_float(selected_opening_frame_resolver.get("radius", 0.0), 0.0)),
            selected_confidence=float(_safe_float(selected_opening_frame_resolver.get("confidence", 0.0), 0.0)),
            selected_segment_count=int(max(1, len(selected_ids) if selected_ids else len(seed_face_ids))),
            edge_scale=float(edge_scale),
            radial_scale=float(radial_scale),
            axial_span_all=float(axial_span_all),
            role_proof=role_proof,
        )
        st = {
            **st,
            **role_proof,
            **ownership_proof_v173o,
            "score": float(score),
            "accepted_as_annular_transition_evidence": bool(accept_evidence and not chamfer_too_large_for_annular_transition and bool(ownership_proof_v173o.get("chamfer_band_ownership_proof_valid_v173o", False))),
            "chamfer_region_fraction": float(chamfer_region_fraction),
            "chamfer_too_large_for_annular_transition": bool(chamfer_too_large_for_annular_transition),
            "chamfer_action_demoted_by_bore_resolution": False,
            "chamfer_action_independent_of_bore_resolution_v173c": True,
            "cross_family_candidate_creation_allowed_v173c": False,
        }
        comp_ids_for_chamfer = tuple_ints(comp)
        comp_key_v173d = frozenset(int(v) for v in comp_ids_for_chamfer)
        if accept_evidence:
            if comp_key_v173d not in chamfer_row_face_sets_v173d:
                chamfer_rows.append((score, comp_ids_for_chamfer, st))
                chamfer_row_face_sets_v173d.add(comp_key_v173d)
        else:
            chamfer_rejected.append(st)
    chamfer_rows.sort(key=lambda item: (-item[0], -int(item[2].get("seed_related", False)), -len(item[1]), item[1][:1]))

    features: list[dict[str, object]] = []
    selected_rail_transition_band_diag_v173h: dict[str, object] = {}
    selected_rail_transition_band_candidate_emitted_v173h = False

    if bore_mouth_transition_face_ids:
        mt = dict(bore_mouth_transition_diagnostics or {})
        mt_ids = tuple_ints(bore_mouth_transition_face_ids)
        mt_face_count = int(mt.get("face_count", len(mt_ids)) or len(mt_ids))
        mt_axial_span = _safe_float(mt.get("axial_span", 0.0), 0.0)
        mt_score = _safe_float(mt.get("score", 0.0), 0.0)
        mt_touches_selected = bool(mt.get("touches_selected_opening", False))
        mt_touches_opposite = bool(mt.get("touches_opposite_opening", False))
        mt_stats_v173o = _component_stats_v174y(
            comp=mt_ids,
            fid_to_local=fid_to_local,
            axial=axial,
            radial=radial,
            normal_axis_abs=normal_axis_abs,
            radial_normal_alignment=radial_normal_alignment,
            seed_face_set=selected_seed_face_ids or seed_face_ids,
            adjacency=adjacency,
        )
        if mt_touches_selected:
            mt_stats_v173o = {**dict(mt_stats_v173o), "seed_related": True}
        mt_role_proof_v173o = chamfer_band_role_proof_v174f(
            faces=faces,
            face_ids=mt_ids,
            component_stats=mt_stats_v173o,
        )
        mt_selected_radius_v173p = float(_safe_float(selected_opening_frame_resolver.get("radius", bore_radius), bore_radius))
        mt_selected_confidence_v173p = float(_safe_float(selected_opening_frame_resolver.get("confidence", 0.0), 0.0))
        mt_selected_support_count_v173r = _safe_int_v174y(
            selected_opening_frame_resolver.get(
                "primary_edge_count",
                selected_opening_frame_resolver.get("edge_count", 0),
            ),
            0,
        )
        if mt_selected_support_count_v173r <= 0:
            mt_selected_support_count_v173r = int(len(selected_ids) if selected_ids else len(seed_face_ids))
        mt_selected_segment_count_v173p = int(max(1, mt_selected_support_count_v173r))
        mt_ownership_proof_v173o = chamfer_selected_rail_local_band_ownership_proof_v174f(
            faces=faces,
            face_ids=mt_ids,
            component_stats=mt_stats_v173o,
            selected_radius=mt_selected_radius_v173p,
            selected_confidence=mt_selected_confidence_v173p,
            selected_segment_count=mt_selected_segment_count_v173p,
            edge_scale=float(edge_scale),
            radial_scale=float(radial_scale),
            axial_span_all=float(axial_span_all),
            role_proof=mt_role_proof_v173o,
        )

        def _mt_chamfer_owned_subset_proof_evaluator_v175c(
            ids_in: Iterable[int],
        ) -> tuple[tuple[int, ...], dict[str, object], dict[str, object], dict[str, object]]:
            """Re-evaluate a cleaned CHAMFER subset for legacy v173p/v173q fallback helpers.

            v175c fixes a split-wiring regression introduced while moving the
            CHAMFER candidate assembly into recognition_chamfer.py: the legacy
            fallback helpers require a proof_evaluator so they can test each
            proposed cleaned subset against the same v173o CHAMFER_BAND role
            proof.  This helper restores that call contract without changing the
            v173t-first export authority or CandidateData semantics.
            """

            ids_eval = tuple_ints(ids_in)
            st_eval = _component_stats_v174y(
                comp=ids_eval,
                fid_to_local=fid_to_local,
                axial=axial,
                radial=radial,
                normal_axis_abs=normal_axis_abs,
                radial_normal_alignment=radial_normal_alignment,
                seed_face_set=selected_seed_face_ids or seed_face_ids,
                adjacency=adjacency,
            )
            if mt_touches_selected:
                st_eval = {**dict(st_eval), "seed_related": True}
            role_eval = chamfer_band_role_proof_v174f(
                faces=faces,
                face_ids=ids_eval,
                component_stats=st_eval,
            )
            proof_eval = chamfer_selected_rail_local_band_ownership_proof_v174f(
                faces=faces,
                face_ids=ids_eval,
                component_stats=st_eval,
                selected_radius=mt_selected_radius_v173p,
                selected_confidence=mt_selected_confidence_v173p,
                selected_segment_count=mt_selected_segment_count_v173p,
                edge_scale=float(edge_scale),
                radial_scale=float(radial_scale),
                axial_span_all=float(axial_span_all),
                role_proof=role_eval,
            )
            return ids_eval, dict(st_eval or {}), dict(role_eval or {}), dict(proof_eval or {})

        mt_owned_subset_v173p = chamfer_legacy_local_transition_band_owned_subset_mapping_v174g(
            faces=faces,
            face_ids=mt_ids,
            fid_to_local=fid_to_local,
            axial=axial,
            radial=radial,
            normal_axis_abs=normal_axis_abs,
            radial_normal_alignment=radial_normal_alignment,
            seed_face_set=selected_seed_face_ids or seed_face_ids,
            adjacency=adjacency,
            selected_radius=mt_selected_radius_v173p,
            selected_confidence=mt_selected_confidence_v173p,
            selected_segment_count=mt_selected_segment_count_v173p,
            edge_scale=float(edge_scale),
            radial_scale=float(radial_scale),
            axial_span_all=float(axial_span_all),
            pre_component_stats=mt_stats_v173o,
            pre_role_proof=mt_role_proof_v173o,
            pre_ownership_proof=mt_ownership_proof_v173o,
            proof_evaluator=_mt_chamfer_owned_subset_proof_evaluator_v175c,
        )
        mt_owned_coordinate_v173q = chamfer_legacy_rail_anchored_owned_band_coordinate_mapping_v174g(
            faces=faces,
            face_ids=mt_ids,
            fid_to_local=fid_to_local,
            axial=axial,
            radial=radial,
            normal_axis_abs=normal_axis_abs,
            radial_normal_alignment=radial_normal_alignment,
            seed_face_set=selected_seed_face_ids or seed_face_ids,
            adjacency=adjacency,
            selected_radius=mt_selected_radius_v173p,
            selected_confidence=mt_selected_confidence_v173p,
            selected_segment_count=mt_selected_segment_count_v173p,
            edge_scale=float(edge_scale),
            radial_scale=float(radial_scale),
            axial_span_all=float(axial_span_all),
            pre_component_stats=mt_stats_v173o,
            pre_role_proof=mt_role_proof_v173o,
            pre_ownership_proof=mt_ownership_proof_v173o,
            proof_evaluator=_mt_chamfer_owned_subset_proof_evaluator_v175c,
        )
        mt_owned_rail_to_rail_v173t = chamfer_rail_to_rail_bounded_surface_ownership_v174e(
            faces=faces,
            vertices=vertices,
            face_ids=mt_ids,
            fid_to_local=fid_to_local,
            axial=axial,
            radial=radial,
            normal_axis_abs=normal_axis_abs,
            radial_normal_alignment=radial_normal_alignment,
            seed_face_set=selected_seed_face_ids or seed_face_ids,
            adjacency=adjacency,
            frame=region_frame,
            selected_support_edge_count=int(mt_selected_segment_count_v173p),
            edge_scale=float(edge_scale),
            radial_scale=float(radial_scale),
            axial_span_all=float(axial_span_all),
            evaluate_owned_subset=_mt_chamfer_owned_subset_proof_evaluator_v175c,
        )
        mt_export_selection_v174b = select_chamfer_candidate_export_faces_v174y(
            raw_face_ids=mt_ids,
            owned_subset_v173p=mt_owned_subset_v173p,
            owned_coordinate_v173q=mt_owned_coordinate_v173q,
            owned_rail_to_rail_v173t=mt_owned_rail_to_rail_v173t,
            fallback_component_stats=mt_stats_v173o,
            fallback_role_proof=mt_role_proof_v173o,
            fallback_ownership_proof=mt_ownership_proof_v173o,
        )
        mt_use_rail_to_rail_owned_subset_v173t = bool(mt_export_selection_v174b.get("use_rail_to_rail_owned_subset_v173t", False))
        mt_use_coordinate_owned_subset_v173q = bool(mt_export_selection_v174b.get("use_coordinate_owned_subset_v173q", False))
        mt_candidate_ids_v173p = tuple_ints(mt_export_selection_v174b.get("candidate_ids", ()))
        mt_stats_for_candidate_v173p = dict(mt_export_selection_v174b.get("component_stats", mt_stats_v173o) or mt_stats_v173o)
        mt_role_proof_v173o = dict(mt_export_selection_v174b.get("role_proof", mt_role_proof_v173o) or mt_role_proof_v173o)
        mt_ownership_proof_v173o = dict(mt_export_selection_v174b.get("ownership_proof", mt_ownership_proof_v173o) or mt_ownership_proof_v173o)
        mt_face_count_for_candidate_v173p = int(len(mt_candidate_ids_v173p))
        mt_axial_span_for_candidate_v173p = _safe_float(mt_stats_for_candidate_v173p.get("axial_span", mt_axial_span), mt_axial_span)
        mt_coordinate_owned_band_valid_v173q = bool(mt_owned_coordinate_v173q.get("chamfer_coordinate_owned_band_valid_v173q", False))
        mt_rail_to_rail_owned_band_valid_v173t = bool(mt_owned_rail_to_rail_v173t.get("chamfer_rail_to_rail_bounded_surface_valid_v173t", False))
        mt_accepted = bool(
            mt_face_count_for_candidate_v173p >= 12
            and mt_touches_selected
            and not mt_touches_opposite
            and mt_axial_span_for_candidate_v173p <= max(0.18 * float(bore_depth), 2.5 * float(bore_radius), 1.0)
            and (
                mt_rail_to_rail_owned_band_valid_v173t
                or bool(mt_ownership_proof_v173o.get("chamfer_band_ownership_proof_valid_v173o", False))
                or mt_coordinate_owned_band_valid_v173q
            )
        )
        selected_rail_transition_band_diag_v173h = {
            **mt,
            **mt_role_proof_v173o,
            **mt_ownership_proof_v173o,
            **{k: v for k, v in dict(mt_owned_subset_v173p).items() if k not in {"owned_component_stats_v173p", "owned_role_proof_v173p", "owned_ownership_proof_v173p"}},
            **{k: v for k, v in dict(mt_owned_coordinate_v173q).items() if k not in {"owned_component_stats_v173q", "owned_role_proof_v173q", "owned_ownership_proof_v173q", "owned_face_ids_v173q"}},
            **{k: v for k, v in dict(mt_owned_rail_to_rail_v173t).items() if k not in {"owned_component_stats_v173t", "owned_role_proof_v173t", "owned_ownership_proof_v173t", "owned_face_ids_v173t"}},
            **dict(mt_export_selection_v174b),
            "selected_rail_transition_band_extractor_v173h": True,
            "selected_rail_transition_band_source_v173h": "former_mouth_transition_extractor_rehomed_before_candidate_data",
            "selected_rail_transition_band_face_count_v173h": int(mt_face_count),
            "selected_rail_transition_band_owned_face_count_v173p": int(mt_face_count_for_candidate_v173p),
            "selected_rail_transition_band_rejected_interfering_face_count_v173p": int(len(tuple_ints(mt_owned_subset_v173p.get("rejected_interfering_face_ids_v173p", ())))),
            "selected_rail_transition_band_owned_face_count_v173q": int(mt_face_count_for_candidate_v173p),
            "selected_rail_transition_band_rejected_interfering_face_count_v173q": int(len(tuple_ints(mt_owned_coordinate_v173q.get("rejected_interfering_face_ids_v173q", ())))),
            "selected_rail_transition_band_coordinate_owned_subset_used_v173q": bool(mt_use_coordinate_owned_subset_v173q),
            "selected_rail_transition_band_boundary_contained_annular_slope_used_v173s": bool(mt_use_coordinate_owned_subset_v173q and bool(mt_owned_coordinate_v173q.get("chamfer_boundary_contained_annular_slope_valid_v173s", False))),
            "selected_rail_transition_band_boundary_contained_annular_slope_owned_face_count_v173s": int(len(tuple_ints(mt_owned_coordinate_v173q.get("owned_face_ids_v173q", ()))) if bool(mt_owned_coordinate_v173q.get("chamfer_boundary_contained_annular_slope_valid_v173s", False)) else 0),
            "selected_rail_transition_band_boundary_contained_annular_slope_context_face_count_v173s": int(len(tuple_ints(mt_owned_coordinate_v173q.get("rejected_interfering_face_ids_v173q", ()))) if bool(mt_owned_coordinate_v173q.get("chamfer_boundary_contained_annular_slope_valid_v173s", False)) else 0),
            "selected_rail_transition_band_rail_to_rail_owned_subset_used_v173t": bool(mt_use_rail_to_rail_owned_subset_v173t),
            "selected_rail_transition_band_rail_to_rail_valid_v173t": bool(mt_rail_to_rail_owned_band_valid_v173t),
            "selected_rail_transition_band_rail_to_rail_owned_face_count_v173t": int(len(tuple_ints(mt_owned_rail_to_rail_v173t.get("owned_face_ids_v173t", ())))),
            "selected_rail_transition_band_rail_to_rail_context_face_count_v173t": int(len(tuple_ints(mt_owned_rail_to_rail_v173t.get("context_face_ids_v173t", ())))),
            "selected_rail_transition_band_selected_support_count_authority_v173r": int(mt_selected_segment_count_v173p),
            "selected_rail_transition_band_expected_owned_face_count_v173r": int(2 * mt_selected_segment_count_v173p),
            "selected_rail_transition_band_count_authority_rule_v173r": "actual selected rail support edges define expected owned CHAMFER_BAND faces; polygon-estimated segment count remains measurement evidence only",
            "selected_rail_transition_band_accepted_v173h": bool(mt_accepted),
            "selected_rail_transition_band_semantic_stage_v173h": "SelectedAnnularRailEvidence -> SelectedRailTransitionBandEvidence -> CHAMFER_BAND ownership",
            "bore_mouth_transition_candidate_authority_v173c": "rehomed_selected_rail_transition_band_chamfer_ownership_v173h",
            "cross_family_candidate_creation_allowed_v173c": False,
            "cross_family_candidate_creation_allowed_v173h": False,
            "semantic_transform_v173h": "legacy mouth-transition geometry extractor rehomed as selected-rail CHAMFER_BAND ownership; no BORE ownership transfer",
            "rebuild_authority": "selected_rail_transition_band_owned_chamfer_faces_exact_delete_patch_only",
            "feature_ownership_source_v173h": "selected_rail_transition_band_chamfer_band_ownership_v173h",
            "recognition_chamfer_candidate_assembly_split_v174y": True,
            "recognition_chamfer_legacy_proof_evaluator_wireup_v175c": True,
            "recognition_chamfer_legacy_proof_evaluator_wireup_status_v175c": "runtime_split_regression_fixed_no_candidate_semantics_changed",
        }
        if mt_candidate_ids_v173p:
            features.append(
                chamfer_candidate_contract_fields_v174p(
                    candidate_id="component_engine.v107.chamfer.rail_to_rail_bounded_slope_owned_band.1" if (mt_accepted and mt_rail_to_rail_owned_band_valid_v173t) else ("component_engine.v107.chamfer.rail_to_rail_bounded_slope_review.1" if mt_use_rail_to_rail_owned_subset_v173t else ("component_engine.v101.chamfer.selected_rail_transition_band_owned_subset.1" if mt_accepted else "component_engine.v101.chamfer.selected_rail_transition_band_review.1")),
                    face_ids=mt_candidate_ids_v173p,
                    accepted=bool(mt_accepted),
                    confidence=float(max(0.05, min(0.90, 0.32 + 0.08 * mt_score + (0.04 if bool(mt_owned_subset_v173p.get("chamfer_owned_subset_mapping_valid_v173p", False)) else 0.0)))),
                    radius_inner=_safe_float(mt_stats_for_candidate_v173p.get("radial_min", mt.get("radius_inner", bore_radius)), _safe_float(mt.get("radius_inner", bore_radius), bore_radius)),
                    radius_outer=_safe_float(mt_stats_for_candidate_v173p.get("radial_max", mt.get("radius_outer", bore_radius)), _safe_float(mt.get("radius_outer", bore_radius), bore_radius)),
                    axial_span=float(mt_axial_span_for_candidate_v173p),
                    diagnostics={
                        **selected_rail_transition_band_diag_v173h,
                        "rank": 0,
                        "recognition_rule": "v107_chamfer_rail_to_rail_bounded_surface_ownership_v173t",
                        "candidate_authority_v173c": "surface_component_classifier_v101_selected_rail_transition_band_ownership",
                        "active_candidate_authority_v173c": "surface_component_classifier_v101_selected_rail_transition_band_ownership",
                        "primitive_source_v173c": "recognition_component_engine.v107_rail_to_rail_bounded_chamfer_ownership",
                    },
                )
            )
            selected_rail_transition_band_candidate_emitted_v173h = True
    bore_mouth_transition_relationship_diag_v173c: dict[str, object] = dict(selected_rail_transition_band_diag_v173h)

    for index, (score, comp, st) in enumerate(chamfer_rows[:8], start=1):
        if bool(st.get("chamfer_too_large_for_annular_transition", False)) and bool(bore_resolution_active):
            chamfer_rejected.append({
                **dict(st),
                "chamfer_display_suppressed_by_bore_resolution": True,
                "chamfer_display_suppress_reason": "chamfer_candidate_matches_full_regiondata_not_annular_surface_ownership",
            })
            continue
        comp_ids = tuple_ints(comp)
        radius_inner = _safe_float(st.get("radial_min", 0.0), 0.0)
        radius_outer = _safe_float(st.get("radial_max", radius_inner), radius_inner)
        axial_span = _safe_float(st.get("axial_span", 0.0), 0.0)
        confidence = max(0.05, min(0.92, 0.22 + 0.11 * float(score) + (0.07 if bool(st.get("seed_related", False)) else 0.0)))
        accepted = bool(st.get("accepted_as_annular_transition_evidence", False)) and (
            bool(st.get("chamfer_band_role_valid_v173c", False))
            or bool(st.get("chamfer_band_role_valid_for_candidate_v173g", False))
            or bool(st.get("selected_rail_local_chamfer_band_owned_v173g", False))
        )
        cid = f"component_engine.v100.chamfer.rail_anchored_band_ownership.{index}" if accepted else f"component_engine.v100.chamfer.review.{index}"
        features.append(
            chamfer_candidate_contract_fields_v174p(
                candidate_id=cid,
                face_ids=comp_ids,
                accepted=bool(accepted),
                confidence=float(confidence),
                radius_inner=float(radius_inner),
                radius_outer=float(radius_outer),
                axial_span=float(axial_span),
                diagnostics={
                    **st,
                    "rank": int(index),
                    "recognition_rule": "v100_selected_rail_anchored_annular_transition_ownership",
                    "candidate_authority_v173c": "surface_component_classifier_v101_selected_rail_transition_band_ownership",
                    "active_candidate_authority_v173c": "surface_component_classifier_v101_selected_rail_transition_band_ownership",
                    "feature_ownership_source_v173c": "independent_chamfer_band_surface_role_ownership_v173c",
                    "primitive_source_v173c": "recognition_component_engine.v100_selected_rail_anchored_annular_transition_ownership",
                    "recognition_chamfer_candidate_assembly_split_v174y": True,
                },
            )
        )

    return {
        "features": tuple(features),
        "chamfer_rows": tuple(chamfer_rows),
        "chamfer_rejected": tuple(chamfer_rejected),
        "selected_rail_transition_band_diag_v173h": dict(selected_rail_transition_band_diag_v173h),
        "selected_rail_transition_band_candidate_emitted_v173h": bool(selected_rail_transition_band_candidate_emitted_v173h),
        "bore_mouth_transition_relationship_diag_v173c": dict(bore_mouth_transition_relationship_diag_v173c),
        "recognition_chamfer_candidate_assembly_split_checkpoint_v174y": CHAMFER_RECOGNITION_CANDIDATE_ASSEMBLY_SPLIT_CHECKPOINT_V174Y,
        "recognition_chamfer_candidate_assembly_no_behavior_change_v174y": True,
    }
