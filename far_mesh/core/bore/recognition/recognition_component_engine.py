"""Semantic feature recognition and surface-role ownership engine for FAR MESH BoreTool.

This module consumes neutral RegionData plus measured evidence and emits
CandidateData rows for the active mesh-native feature families.  It is the
recognition/ownership boundary, not a Region Select module and not a topology
mutation module.

Current semantic meaning paths
------------------------------
RegionData faces
    -> measured annular-transition evidence
    -> independent CHAMFER feature hypotheses
    -> chamfer_owned_face_ids per hypothesis
    -> accepted or review CandidateData for the owned chamfer surface

RegionData faces
    -> selected opening / radius-layer / side-wall evidence
    -> BORE wall hypothesis
    -> bore_wall_owned_face_ids per hypothesis
    -> accepted CandidateData when wall ownership and boundary proof are valid
    -> review CandidateData when ownership or boundary proof is incomplete

RegionData faces
    -> local pocket probe / floor / side-wall evidence
    -> POCKET recess hypothesis
    -> owned floor + side-wall roles, with child-bore relationships protected
    -> accepted or review CandidateData depending on rebuild-safe ownership

Important semantic rule
-----------------------
Selected opening seed relation is context and evidence.  It is not exclusive
rebuild authority.  This module may accept a feature candidate only after
feature identity, surface-role ownership, and rebuild-safety policy agree.
CandidateData is still not a DeletePatchProposal: downstream rebuild-target
policy bounds deletion, and rebuild.py performs trial topology validation.

This preserves the BoreTool meaning boundary:
    evidence != measurement != ownership != CandidateData != DeletePatchProposal != RebuildResult

v174 cleanup/refactor checkpoint
--------------------------------
The v174 cleanup begins as a behavior-preserving authority-map pass.  This file
still contains the active family-local recognizers, but the stable authority
paths are now named explicitly so later extraction can move CHAMFER, BORE,
POCKET, emission, and diagnostics into smaller modules without reactivating old
fallback paths. v174h starts the BORE module split by moving the terminal-continuation filter out of this orchestrator while keeping CandidateData emission here. v174i moves BORE wall-subset and post-isolation validation helpers into recognition_bore.py without changing CandidateData behavior. v174j moves selected-opening locality and seed-island evidence helpers into recognition_bore.py while CandidateData emission remains in this orchestrator. v174k moves BORE CandidateData construction helpers into recognition_bore.py; final candidate-list assembly remains here. v174l starts the POCKET module split by moving child-bore intrusion filtering, floor-boundary relationship metadata, and POCKET CandidateData contract field construction into recognition_pocket.py without changing candidate behavior. v174m moves X1 probe-ledger pocket-evidence helpers into recognition_pocket.py; final POCKET detector assembly remains in this orchestrator. v174n adds POCKET family-authority leak diagnostics before moving the larger detector so the observed rebuild-side pocket/BORE authority mismatch can be tracked without changing CandidateData behavior. v174o moves the POCKET detector orchestration body into recognition_pocket.py behind the same component-engine wrapper; CandidateData emission shape and rebuild behavior remain unchanged. v174p moves CHAMFER CandidateData contract construction into recognition_chamfer.py. v174q starts the diagnostics module split by moving the read-only surface-role evidence graph into recognition_diagnostics.py while preserving the component-engine wrapper and public diagnostic keys. v174r continues that diagnostics split by moving the static v174a authority-map diagnostics into recognition_diagnostics.py while keeping compatibility wrappers and no CandidateData behavior change. v174s starts the shared emission split by moving the public CandidateResult-like packing helper into recognition_emit.py while preserving the component-engine wrapper and output shape. v174t records the orchestrator/fanout boundary explicitly so the remaining component-engine body can be thinned without changing feature behavior. v174u adds a remaining-wrapper/local-helper inventory before deleting or moving any additional call sites. v174v adds a recognition regression/smoke-test checkpoint diagnostic before any behavior-changing POCKET authority fix or rebuild split. v174w starts actual line-count reduction by migrating internal call sites from compatibility wrappers to family/diagnostics/emit modules, then removing wrapper definitions while preserving public entry points and CandidateData behavior. v174x moves shared measurement/topology helper bodies into recognition_common.py so the component engine keeps shrinking without changing CandidateData behavior. v174y moves CHAMFER candidate-row assembly into recognition_chamfer.py while keeping the final candidate-list merge here. v174z moves BORE candidate-row assembly into recognition_bore.py while keeping the final multi-family merge here. v175a moves final top-level diagnostic assembly into recognition_diagnostics.py while preserving CandidateData and public result behavior. v175b moves the remaining static authority/legacy checkpoint diagnostic body into recognition_diagnostics.py, leaving this module closer to fanout/orchestration only. v175d prunes now-unused imported helper symbols after the family/diagnostics/emit splits; this is a no-behavior import-surface cleanup that keeps public entry points unchanged. v175f adds a diagnostic-only SecondaryOpeningRimSearchHeuristic after family fanout: it proposes additional AOI rim/opening evidence but does not alter CandidateData or rebuild behavior. v175g routes that heuristic proposal into BORE as alternate opening evidence/anchor only: it may help BORE wall ownership succeed, but only through the existing BORE ownership gates; the heuristic still does not emit CandidateData or ownership by itself.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..geometry import canonical_axis
from .recognition_bore import (
    assemble_bore_candidate_rows_v174z as _assemble_bore_candidate_rows_v174z,
)
from .recognition_chamfer import (
    assemble_chamfer_candidate_rows_v174y as _assemble_chamfer_candidate_rows_v174y,
)
from .recognition_common import (
    _Frame,
    edge_median_length_v174x as _edge_median_length,
    local_arrays_v174x as _local_arrays,
    percentile_v174x as _percentile,
    safe_float_v174x as _safe_float,
    selected_annular_rail_role_resolver_v174x as _selected_annular_rail_role_resolver_v173d,
    secondary_opening_rim_search_heuristic_v175f as _secondary_opening_rim_search_heuristic_v175f,
)
from .recognition_pocket import (
    detect_pocket_preview_candidates_v174o as _detect_pocket_preview_candidates_v174o,
    x1_probe_ledger_from_region_diagnostics_v174m as _x1_probe_ledger_from_region_diagnostics_v174m,
)
from .recognition_diagnostics import (
    final_component_engine_diagnostics_v175a as _final_component_engine_diagnostics_v175a,
    component_engine_static_authority_diagnostics_v175b as _component_engine_static_authority_diagnostics_v175b,
    surface_role_evidence_graph_diagnostics_v174q as _surface_role_evidence_graph_diagnostics_v174q,
)
from .recognition_emit import (
    recognition_result_dict_from_component_features_v174s as _recognition_result_dict_from_component_features_v174s,
)
from ..topology import (
    connected_face_components,
    face_adjacency_for_patch,
)
from ..types import (
    FeatureFamily,
    tuple_ints,
)





# -----------------------------------------------------------------------------
# v173b surface-role evidence graph diagnostics
# -----------------------------------------------------------------------------



def component_engine_feature_candidates(**kwargs: object) -> dict[str, object]:
    """Recognize physical CandidateData from explicit semantic stages.

    v1.5.2 cleanup: BORE CandidateData is no longer created from one local
    cylinder-like strip, selected-edge fragment, or RegionData extent; and
    a valid two-opening frame now creates a BORE review object even when wall
    ownership is not yet accepted.  A BORE
    candidate may be promoted only after this chain exists:

        measured selected opening A
        -> measured opposite opening B
        -> valid two-opening bore frame
        -> connected bore-wall ownership between A and B
        -> CandidateData

    CHAMFER recognition remains available as an independent feature-family path.
    BORE frames may produce relationship evidence for shared mouth rails, but
    they are not allowed to create accepted CHAMFER CandidateData.  CHAMFER
    actionability must come only from CHAMFER_BAND surface-role ownership.
    """

    faces = np.asarray(kwargs.get("faces"), dtype=np.int64)
    face_centroids = np.asarray(kwargs.get("face_centroids"), dtype=float)
    face_normals = np.asarray(kwargs.get("face_normals"), dtype=float)
    vertices = np.asarray(kwargs.get("vertices"), dtype=float)
    all_face_ids = tuple_ints(kwargs.get("face_ids", ()))
    valid_face_ids = tuple(
        int(fid) for fid in all_face_ids
        if 0 <= int(fid) < len(faces) and 0 <= int(fid) < len(face_centroids)
    )
    valid_face_set = {int(fid) for fid in valid_face_ids}
    seed_face_set = {int(fid) for fid in tuple_ints(kwargs.get("seed_face_ids", ())) if int(fid) in valid_face_set}
    resolved_seed_face_set = {int(fid) for fid in tuple_ints(kwargs.get("resolved_opening_seed_face_ids", ())) if int(fid) in valid_face_set}
    selected_seed_face_set = resolved_seed_face_set or seed_face_set

    region_diagnostics = dict(kwargs.get("region_diagnostics", {}) or {})
    selected_opening_measurement_audit = dict(kwargs.get("selected_opening_measurement_audit", {}) or {})
    selected_opening_frame_resolver = dict(kwargs.get("selected_opening_frame_resolver", {}) or {})
    two_opening_bore_frame = dict(kwargs.get("two_opening_bore_frame", {}) or {})

    selected_edge_ids = tuple_ints(kwargs.get("selected_edge_ids", ()))
    selected_edge_count = int(region_diagnostics.get("selected_edge_count", len(selected_edge_ids)) or 0) or 0
    normalized_edge_count = int(region_diagnostics.get("normalized_edge_count", region_diagnostics.get("primary_anchor_edge_count", region_diagnostics.get("normalized_rim_edge_count", selected_edge_count))) or 0)
    region_face_count = int(len(valid_face_ids))
    broad_or_ambiguous_selection = bool(
        selected_edge_count >= 500
        or region_face_count >= 5000
        or bool(selected_opening_measurement_audit.get("raw_selection_severe_fragmentation_suspected", False))
        or bool(selected_opening_measurement_audit.get("normalized_rim_collapse_suspected", False))
    )

    if not valid_face_ids or faces.ndim != 2 or face_centroids.ndim != 2:
        return {
            "candidate_data": (),
            "features": (),
            "diagnostics": {
                "component_engine_version": 93,
                "active_candidate_authority": "surface_component_classifier_v105_selected_rail_support_count_chamfer_ownership",
                "recognition_cleanup": "invalid_or_empty_region_input",
                "region_face_count": int(region_face_count),
            },
            "promoted_candidate_count": 0,
        }

    # ------------------------------------------------------------------
    # Frame authority: BORE may use only a valid two-opening frame.
    # RegionData frame remains diagnostic context for CHAMFER only.
    # ------------------------------------------------------------------
    two_opening_valid = bool(two_opening_bore_frame.get("valid", False))
    bore_frame_reason = str(two_opening_bore_frame.get("status", "") or "")
    if two_opening_valid:
        # v1.5.9 semantic/numeric fix:
        # The two-opening frame axis is directed evidence from selected opening A
        # toward opposite opening B.  Do not canonicalize it independently,
        # because canonical_axis may flip -Z to +Z while bore_center remains at
        # opening A.  That puts the axial search interval outside the physical
        # bore and makes mouth/chamfer faces look like the only candidate.
        opening_center_arr = np.asarray(
            two_opening_bore_frame.get("opening_center", two_opening_bore_frame.get("center", (0.0, 0.0, 0.0))),
            dtype=float,
        ).reshape(3)
        opposite_center_arr = np.asarray(
            two_opening_bore_frame.get("opposite_center", opening_center_arr),
            dtype=float,
        ).reshape(3)
        center_delta = opposite_center_arr - opening_center_arr
        center_delta_len = float(np.linalg.norm(center_delta))
        raw_axis = np.asarray(two_opening_bore_frame.get("axis", center_delta if center_delta_len > 1.0e-12 else (0.0, 0.0, 1.0)), dtype=float).reshape(3)
        raw_axis_len = float(np.linalg.norm(raw_axis))
        if np.isfinite(center_delta_len) and center_delta_len > 1.0e-12:
            directed_axis = center_delta / center_delta_len
            if np.isfinite(raw_axis_len) and raw_axis_len > 1.0e-12:
                raw_axis_unit = raw_axis / raw_axis_len
                if float(np.dot(raw_axis_unit, directed_axis)) >= 0.0:
                    directed_axis = raw_axis_unit
                else:
                    directed_axis = -raw_axis_unit
            bore_depth_from_centers = center_delta_len
        elif np.isfinite(raw_axis_len) and raw_axis_len > 1.0e-12:
            directed_axis = raw_axis / raw_axis_len
            bore_depth_from_centers = _safe_float(two_opening_bore_frame.get("depth", 0.0), 0.0)
        else:
            directed_axis = np.array([0.0, 0.0, 1.0], dtype=float)
            bore_depth_from_centers = _safe_float(two_opening_bore_frame.get("depth", 0.0), 0.0)
        bore_center = opening_center_arr
        bore_axis = directed_axis
        bore_radius = _safe_float(two_opening_bore_frame.get("radius", 0.0), 0.0)
        bore_depth = _safe_float(two_opening_bore_frame.get("depth", bore_depth_from_centers), bore_depth_from_centers)
        if not np.isfinite(bore_depth) or bore_depth <= 1.0e-12:
            bore_depth = float(bore_depth_from_centers)
        bore_axial_min = 0.0
        bore_axial_max = float(bore_depth)
        two_opening_axis_direction_preserved = True
        two_opening_axis_dot_opening_to_opposite = float(np.dot(bore_axis, center_delta / center_delta_len)) if np.isfinite(center_delta_len) and center_delta_len > 1.0e-12 else 0.0
    else:
        bore_center = np.asarray(kwargs.get("region_center", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
        bore_axis = canonical_axis(kwargs.get("region_axis", (0.0, 0.0, 1.0)))
        bore_radius = _safe_float(kwargs.get("region_radius", 0.0), 0.0)
        bore_depth = 0.0
        bore_axial_min = 0.0
        bore_axial_max = 0.0
        two_opening_axis_direction_preserved = False
        two_opening_axis_dot_opening_to_opposite = 0.0

    region_frame = _Frame(
        center=kwargs.get("region_center", bore_center),
        axis=kwargs.get("region_axis", bore_axis),
        radius=kwargs.get("region_radius", bore_radius),
    )
    # v1.7.4 semantic scale correction:
    # The RegionData cutout is neutral AOI evidence and may include large cap,
    # panel, or previously rebuilt faces.  Its median triangle edge length must
    # not become BORE-wall ownership scale authority.  Use it only as a raw
    # diagnostic, then clamp the role scale to the selected/opening ring
    # measurement.  Otherwise a clean full-depth bore wall can be rejected as
    # "axial_span_too_small" because a broad AOI face made edge_scale larger
    # than the physical bore-wall sampling scale.
    raw_region_edge_scale = _edge_median_length(vertices, faces, valid_face_ids)
    edge_scale = float(raw_region_edge_scale)
    adjacency = face_adjacency_for_patch(faces, valid_face_ids)

    # v175g: run the SecondaryOpeningRimSearchHeuristic before the BORE pass so
    # BORE can treat discovered AOI rims as alternate opening evidence/anchor.
    # This is still a heuristic proposal stage: it does not emit CandidateData
    # and does not assign ownership.  BORE may consume the measured proposal only
    # through its existing wall-ownership validation gates.
    secondary_opening_rim_heuristic_diag_v175f = _secondary_opening_rim_search_heuristic_v175f(
        faces=faces,
        vertices=vertices,
        valid_face_ids=valid_face_ids,
        selected_opening_frame_resolver=selected_opening_frame_resolver,
        region_frame=region_frame,
        edge_scale=float(edge_scale),
        accepted_chamfer_present=False,
    )

    # ------------------------------------------------------------------
    # BORE recognition.  v174z moves BORE candidate-row assembly into
    # recognition_bore.py while preserving this function as the multi-family
    # orchestrator and final candidate-list merge point.
    # ------------------------------------------------------------------
    bore_assembly_v174z = _assemble_bore_candidate_rows_v174z(
        faces=faces,
        vertices=vertices,
        face_centroids=face_centroids,
        face_normals=face_normals,
        valid_face_ids=valid_face_ids,
        seed_face_set=seed_face_set,
        selected_seed_face_set=selected_seed_face_set,
        adjacency=adjacency,
        kwargs=kwargs,
        region_diagnostics=region_diagnostics,
        selected_opening_measurement_audit=selected_opening_measurement_audit,
        selected_opening_frame_resolver=selected_opening_frame_resolver,
        two_opening_bore_frame=two_opening_bore_frame,
        selected_edge_count=int(selected_edge_count),
        normalized_edge_count=int(normalized_edge_count),
        region_face_count=int(region_face_count),
        broad_or_ambiguous_selection=bool(broad_or_ambiguous_selection),
        bore_center=bore_center,
        bore_axis=bore_axis,
        bore_radius=float(bore_radius),
        bore_depth=float(bore_depth),
        bore_axial_min=float(bore_axial_min),
        bore_axial_max=float(bore_axial_max),
        bore_frame_reason=str(bore_frame_reason),
        two_opening_valid=bool(two_opening_valid),
        two_opening_axis_direction_preserved=bool(two_opening_axis_direction_preserved),
        two_opening_axis_dot_opening_to_opposite=float(two_opening_axis_dot_opening_to_opposite),
        region_frame=region_frame,
        raw_region_edge_scale=float(raw_region_edge_scale),
        edge_scale=float(edge_scale),
        secondary_opening_rim_heuristic_diag=secondary_opening_rim_heuristic_diag_v175f,
    )
    bore_features = list(bore_assembly_v174z.get("bore_features", ()) or ())
    bore_rejected = list(bore_assembly_v174z.get("bore_rejected", ()) or ())
    bore_wall_candidate_components = tuple(bore_assembly_v174z.get("bore_wall_candidate_components", ()) or ())
    bore_owned_face_count = int(bore_assembly_v174z.get("bore_owned_face_count", 0) or 0)
    best_bore_wall_diag = dict(bore_assembly_v174z.get("best_bore_wall_diag", {}) or {})
    bore_tube_face_ids = tuple_ints(bore_assembly_v174z.get("bore_tube_face_ids", ()))
    bore_wall_search_face_ids = tuple_ints(bore_assembly_v174z.get("bore_wall_search_face_ids", ()))
    bore_mouth_transition_face_ids = tuple_ints(bore_assembly_v174z.get("bore_mouth_transition_face_ids", ()))
    bore_mouth_transition_diag = dict(bore_assembly_v174z.get("bore_mouth_transition_diag", {}) or {})
    bore_resolution_active = bool(bore_assembly_v174z.get("bore_resolution_active", False))
    edge_scale = float(bore_assembly_v174z.get("edge_scale", edge_scale) or edge_scale)
    bore_radius = float(bore_assembly_v174z.get("bore_radius", bore_radius) or bore_radius)
    two_opening_bore_frame = dict(bore_assembly_v174z.get("two_opening_bore_frame", two_opening_bore_frame) or two_opening_bore_frame)
    selected_opening_clean_radius_authority = bool(bore_assembly_v174z.get("selected_opening_clean_radius_authority", False))
    selected_opening_support_weak = bool(bore_assembly_v174z.get("selected_opening_support_weak", False))
    # ------------------------------------------------------------------
    # CHAMFER recognition.  This is now a family-local ownership transform:
    # independent CHAMFER_BAND proof may create CHAMFER CandidateData.  BORE
    # mouth-transition evidence below remains relationship metadata only.
    # ------------------------------------------------------------------
    local = _local_arrays(frame=region_frame, face_ids=valid_face_ids, face_centroids=face_centroids, face_normals=face_normals)
    axial_all = local["axial"]
    radial_all = local["radial"]
    normal_axis_abs_all = local["normal_axis_abs"]
    radial_normal_alignment_all = local["radial_normal_alignment"]
    axial_span_all = float(np.max(axial_all) - np.min(axial_all)) if len(axial_all) else 0.0
    radial_scale = max(region_frame.radius, _percentile(radial_all, 75.0, 1.0), 1.0)
    min_faces_general = max(18, int(0.0025 * max(region_face_count, 1)))
    min_radial_span = max(0.015 * radial_scale, 2.0 * edge_scale, 0.18)
    min_axial_span = max(0.015 * radial_scale, 2.0 * edge_scale, 0.18)

    components = connected_face_components(faces, valid_face_ids)
    fid_to_local = {int(fid): int(i) for i, fid in enumerate(valid_face_ids)}

    selected_annular_rail_diag_v173d = _selected_annular_rail_role_resolver_v173d(
        faces=faces,
        valid_face_ids=valid_face_ids,
        selected_seed_face_ids=selected_seed_face_set or seed_face_set,
        seed_face_set=seed_face_set,
        fid_to_local=fid_to_local,
        axial=axial_all,
        radial=radial_all,
        normal_axis_abs=normal_axis_abs_all,
        radial_normal_alignment=radial_normal_alignment_all,
        adjacency=adjacency,
        selected_opening_frame_resolver=selected_opening_frame_resolver,
        edge_scale=edge_scale,
        radial_scale=radial_scale,
        axial_span_all=axial_span_all,
    )
    selected_annular_rail_primary_role_v173d = str(
        selected_annular_rail_diag_v173d.get("selected_annular_rail_primary_role_v173d", "unresolved_annular_rail")
    )
    selected_annular_rail_public_diag_v173d = {
        k: v
        for k, v in dict(selected_annular_rail_diag_v173d).items()
        if k != "selected_annular_rail_chamfer_candidate_rows_v173d"
    }

    pocket_features, pocket_diag = _detect_pocket_preview_candidates_v174o(
        faces=faces,
        vertices=vertices,
        face_centroids=face_centroids,
        face_normals=face_normals,
        valid_face_ids=valid_face_ids,
        region_frame=region_frame,
        seed_face_set=selected_seed_face_set or seed_face_set,
        adjacency=adjacency,
        selected_opening_frame_resolver=selected_opening_frame_resolver,
        region_diagnostics=region_diagnostics,
        opening_probe_ledger=_x1_probe_ledger_from_region_diagnostics_v174m(region_diagnostics),
        edge_scale=edge_scale,
        region_face_count=region_face_count,
        selected_opening_clean_radius_authority=bool(selected_opening_clean_radius_authority),
        two_opening_bore_frame_valid=bool(two_opening_valid),
        two_opening_bore_frame_depth=float(bore_depth),
    )

    features: list[dict[str, object]] = []
    features.extend(bore_features)
    features.extend(pocket_features)

    chamfer_assembly_v174y = _assemble_chamfer_candidate_rows_v174y(
        faces=faces,
        vertices=vertices,
        components=components,
        fid_to_local=fid_to_local,
        axial=axial_all,
        radial=radial_all,
        normal_axis_abs=normal_axis_abs_all,
        radial_normal_alignment=radial_normal_alignment_all,
        seed_face_set=seed_face_set,
        selected_seed_face_set=selected_seed_face_set,
        adjacency=adjacency,
        selected_annular_rail_diagnostics=selected_annular_rail_diag_v173d,
        selected_opening_frame_resolver=selected_opening_frame_resolver,
        selected_edge_ids=selected_edge_ids,
        edge_scale=float(edge_scale),
        radial_scale=float(radial_scale),
        axial_span_all=float(axial_span_all),
        min_faces_general=int(min_faces_general),
        min_radial_span=float(min_radial_span),
        min_axial_span=float(min_axial_span),
        region_face_count=int(region_face_count),
        bore_resolution_active=bool(bore_resolution_active),
        region_frame=region_frame,
        bore_mouth_transition_face_ids=bore_mouth_transition_face_ids,
        bore_mouth_transition_diagnostics=bore_mouth_transition_diag,
        bore_radius=float(bore_radius),
        bore_depth=float(bore_depth),
    )
    chamfer_rows = tuple(chamfer_assembly_v174y.get("chamfer_rows", ()) or ())
    chamfer_rejected = list(chamfer_assembly_v174y.get("chamfer_rejected", ()) or ())
    selected_rail_transition_band_diag_v173h = dict(chamfer_assembly_v174y.get("selected_rail_transition_band_diag_v173h", {}) or {})
    selected_rail_transition_band_candidate_emitted_v173h = bool(chamfer_assembly_v174y.get("selected_rail_transition_band_candidate_emitted_v173h", False))
    bore_mouth_transition_relationship_diag_v173c: dict[str, object] = dict(chamfer_assembly_v174y.get("bore_mouth_transition_relationship_diag_v173c", {}) or {})
    features.extend(tuple(chamfer_assembly_v174y.get("features", ()) or ()))

    # v173b: do not arbitrate after CandidateData is emitted.  Post-candidate
    # demotion is a permission filter and can reintroduce old feature-pair bugs.
    # Candidate actionability must be created by the family-specific ownership
    # transform before CandidateData.  Keep only a read-only role/relationship
    # graph for debugging BORE/POCKET/CHAMFER contact.
    selected_rail_chamfer_accepted_v173d = bool(
        any(str(item.get("feature_family", "")) == FeatureFamily.CHAMFER_FORM.value and bool(item.get("candidate_action_enabled", False)) for item in features)
        and (
            str(selected_annular_rail_diag_v173d.get("selected_annular_rail_consumed_by_family_v173d", "")) == "chamfer"
            or bool(locals().get("selected_rail_transition_band_candidate_emitted_v173h", False))
        )
    )

    # v175g: CHAMFER acceptance is discovered after the early secondary-rim
    # heuristic has already been routed into BORE.  Preserve the same proposal
    # rows and update only the context diagnostics; do not rerun or alter BORE
    # CandidateData after family fanout.
    secondary_opening_rim_heuristic_diag_v175f = {
        **dict(secondary_opening_rim_heuristic_diag_v175f or {}),
        "secondary_opening_rim_accepted_chamfer_context_present_v175g": bool(selected_rail_chamfer_accepted_v173d),
        "secondary_opening_rim_bore_consumption_stage_v175g": "heuristic_proposal_consumed_by_bore_only_through_wall_ownership_gates",
    }

    # v173i: Recognition is a multi-family fanout over the neutral AOI.
    # A CHAMFER ownership result must not terminate or suppress BORE / POCKET
    # recognition.  The selected rail can acquire CHAMFER_BAND meaning for the
    # CHAMFER candidate while still remaining neutral evidence for other family
    # recognizers.  Only exact same-surface duplicate rows may be collapsed later;
    # this stage does not remove BORE review evidence or POCKET candidates merely
    # because a CHAMFER was found.
    bore_review_suppressed_by_selected_rail_role_v173d = 0
    bore_review_suppression_removed_v173i = True
    multi_family_aoi_fanout_v173i = True
    candidate_suppression_scope_v173i = "same_surface_duplicate_only_no_family_winner_take_all"

    surface_role_graph_diag_v173b = _surface_role_evidence_graph_diagnostics_v174q(features)
    features = list(features)
    promoted = tuple(item for item in features if bool(item.get("candidate_action_enabled", False)))

    # v175a: final top-level diagnostics now live in recognition_diagnostics.py.
    # This keeps component_engine_feature_candidates() as family fanout + merge
    # while preserving the exact CandidateData and public result behavior.
    diag = _final_component_engine_diagnostics_v175a(
        component_locals=locals(),
        bore_assembly_v174z=bore_assembly_v174z,
        base_static_diagnostics=_component_engine_static_authority_diagnostics_v175b(),
    )

    return {
        "candidate_data": tuple(features),
        "features": tuple(features),
        "diagnostics": diag,
        "promoted_candidate_count": int(len(promoted)),
    }


def recognition_result_dict_from_component_features(
    *,
    features: tuple[dict[str, object], ...],
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Return a CandidateResult-like dictionary for GUI diagnostics.

    v174s moves the shared result packing body into ``recognition_emit.py``.
    This wrapper preserves the public component-engine API and the returned
    dictionary shape.
    """

    return _recognition_result_dict_from_component_features_v174s(
        features=features,
        diagnostics=diagnostics,
    )


__all__ = [
    "component_engine_feature_candidates",
    "recognition_result_dict_from_component_features",
]
