"""Semantic heuristic tool contracts for FAR MESH BoreTool.

A heuristic in BoreTool is a bounded evidence-search/proposal operation.  It is
allowed to look inside an already-known semantic context, propose evidence, and
publish diagnostics.  It is not allowed to declare feature identity, own faces,
create CandidateData, authorize delete patches, or mutate mesh topology.

This module is deliberately small and serializable.  It gives the active
recognition code a real vocabulary for the heuristic layer documented in v4.6
and v4.7:

    semantic context -> heuristic proposal -> measurement -> recognition
    -> ownership -> CandidateData -> rebuild target -> rebuild validation

The helpers below do not perform geometry themselves.  They let existing geometry
and recognition operations report which heuristic role they are fulfilling, what
kind of proposal/evidence they produced, and which authority they do *not* have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

HEURISTIC_CONTRACT_VERSION = "heuristic_tool_contract_v1_7_4"
HEURISTIC_AUTHORITY = "evidence_proposal_only"
HEURISTIC_FORBIDDEN_AUTHORITY: tuple[str, ...] = (
    "may_not_declare_feature_identity",
    "may_not_assign_surface_ownership",
    "may_not_create_CandidateData",
    "may_not_authorize_DeletePatchProposal",
    "may_not_mutate_mesh_topology",
)


@dataclass(frozen=True, slots=True)
class HeuristicToolContract:
    """Declared contract for one bounded heuristic operation."""

    name: str
    semantic_context: str
    input_semantic_object: str
    operation: str
    output_proposal_type: str
    next_semantic_stage: str
    authority: str = HEURISTIC_AUTHORITY
    forbidden_authority: tuple[str, ...] = HEURISTIC_FORBIDDEN_AUTHORITY
    diagnostic_keys: tuple[str, ...] = ()
    implementation_hint: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_type": "heuristic_tool_contract",
            "contract_version": HEURISTIC_CONTRACT_VERSION,
            "heuristic_name": str(self.name),
            "semantic_context": str(self.semantic_context),
            "input_semantic_object": str(self.input_semantic_object),
            "operation": str(self.operation),
            "output_proposal_type": str(self.output_proposal_type),
            "next_semantic_stage": str(self.next_semantic_stage),
            "authority": str(self.authority),
            "forbidden_authority": tuple(str(v) for v in self.forbidden_authority),
            "diagnostic_keys": tuple(str(v) for v in self.diagnostic_keys),
            "implementation_hint": str(self.implementation_hint),
        }


@dataclass(frozen=True, slots=True)
class HeuristicInvocationResult:
    """Runtime diagnostic row for one heuristic's proposed evidence.

    This row records what an already-running heuristic-like code path did.  It is
    still evidence-only.  It must never be consumed as ownership or rebuild input.
    """

    heuristic_name: str
    output_proposal_type: str
    semantic_context: str
    input_count: int = 0
    proposal_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    confidence: float | None = None
    proposal_face_ids: tuple[int, ...] = ()
    proposal_edge_ids: tuple[int, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    authority: str = HEURISTIC_AUTHORITY
    forbidden_authority: tuple[str, ...] = HEURISTIC_FORBIDDEN_AUTHORITY

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "contract_type": "heuristic_invocation_result",
            "contract_version": HEURISTIC_CONTRACT_VERSION,
            "heuristic_name": str(self.heuristic_name),
            "semantic_context": str(self.semantic_context),
            "output_proposal_type": str(self.output_proposal_type),
            "input_count": int(self.input_count),
            "proposal_count": int(self.proposal_count),
            "accepted_count": int(self.accepted_count),
            "rejected_count": int(self.rejected_count),
            "proposal_face_ids": tuple(int(v) for v in self.proposal_face_ids),
            "proposal_face_count": int(len(self.proposal_face_ids)),
            "proposal_edge_ids": tuple(int(v) for v in self.proposal_edge_ids),
            "proposal_edge_count": int(len(self.proposal_edge_ids)),
            "rejection_reasons": tuple(str(v) for v in self.rejection_reasons),
            "diagnostics": dict(self.diagnostics or {}),
            "authority": str(self.authority),
            "forbidden_authority": tuple(str(v) for v in self.forbidden_authority),
            "not_feature_identity": True,
            "not_surface_ownership": True,
            "not_candidate_data": True,
            "not_rebuild_authority": True,
        }
        if self.confidence is not None:
            try:
                out["confidence"] = float(self.confidence)
            except Exception:
                out["confidence"] = None
        return out


# -----------------------------------------------------------------------------
# Registry: named heuristic tools used by the current Bore/Chamfer recipes.
# -----------------------------------------------------------------------------

BORE_HEURISTIC_RECIPE: tuple[str, ...] = (
    "SelectedEdgesToOpeningRingHeuristic",
    "MeasureSelectedOpening",
    "OppositeOpeningSearchHeuristic",
    "MeasureTwoOpeningFrame",
    "BoreWallSearchHeuristic",
    "BoreWallRoleMeasurement",
    "BoreWallOwnership",
    "TrueEndpointResolver",
    "TerminalWallCompletionHeuristic",
    "TerminalWallOwnership",
    "EmitBoreCandidateData",
)

CHAMFER_HEURISTIC_RECIPE: tuple[str, ...] = (
    "OpeningContextToChamferSearch",
    "MeasureChamferBand",
    "ChamferNeighborSurfaceHeuristic",
    "ChamferSurfaceOwnership",
    "EmitChamferCandidateData",
)

POCKET_HEURISTIC_RECIPE: tuple[str, ...] = (
    "SelectedBoundaryToPocketRimHeuristic",
    "MeasurePocketOpening",
    "PocketFloorSurfaceSearchHeuristic",
    "MeasurePocketFloor",
    "PocketSideWallSearchHeuristic",
    "PocketDepthResolver",
    "PocketTransitionSeparation",
    "PocketRoleOwnership",
    "PocketChildBoreBoundaryRelationship",
    "EmitPocketCandidateData",
)

HEURISTIC_TOOL_REGISTRY: dict[str, HeuristicToolContract] = {
    "SelectedEdgesToOpeningRingHeuristic": HeuristicToolContract(
        name="SelectedEdgesToOpeningRingHeuristic",
        semantic_context="user_marked_possible_opening_or_rim",
        input_semantic_object="selected_edge_ids / normalized rim evidence",
        operation="normalize fragmented rim evidence and propose complete opening-ring evidence",
        output_proposal_type="OpeningRingProposal",
        next_semantic_stage="MeasureSelectedOpening",
        diagnostic_keys=(
            "raw_selected_edge_count",
            "normalized_rim_edge_count",
            "normalized_to_raw_edge_ratio",
            "opening_measurement_audit_status",
        ),
        implementation_hint="region_select.normalize_opening_rim_edge_ids_from_arrays + measure.measure_bore_opening_candidates",
    ),
    "MeasureSelectedOpening": HeuristicToolContract(
        name="MeasureSelectedOpening",
        semantic_context="opening_ring_proposal_exists",
        input_semantic_object="OpeningRingProposal",
        operation="fit opening ring and quantify center, radius, axis, confidence",
        output_proposal_type="MeasuredOpeningFrame",
        next_semantic_stage="OppositeOpeningSearchHeuristic",
        diagnostic_keys=(
            "selected_opening_frame_resolved",
            "selected_opening_frame_source",
            "resolved_opening_seed_face_count",
        ),
        implementation_hint="measure.BoreOpeningMeasurement / selected opening frame resolver diagnostics",
    ),
    "OppositeOpeningSearchHeuristic": HeuristicToolContract(
        name="OppositeOpeningSearchHeuristic",
        semantic_context="selected_opening_frame_resolved",
        input_semantic_object="MeasuredOpeningFrame + RegionData boundary loops",
        operation="search radius/axis-compatible opposite loops and classify provisional endpoint evidence",
        output_proposal_type="OppositeOpeningProposal / EndpointProposal / InternalDefectBoundaryProposal",
        next_semantic_stage="MeasureTwoOpeningFrame",
        diagnostic_keys=(
            "opposite_opening_found",
            "opposite_candidate_count",
            "best_opposite_candidate",
            "boundary_loop_count",
        ),
        implementation_hint="measure.measure_two_opening_bore_frame",
    ),
    "MeasureTwoOpeningFrame": HeuristicToolContract(
        name="MeasureTwoOpeningFrame",
        semantic_context="selected_opening_plus_opposite_proposal",
        input_semantic_object="MeasuredOpeningFrame + OppositeOpeningProposal",
        operation="measure provisional bore axis, centerline, radius, and depth",
        output_proposal_type="MeasuredTwoOpeningBoreFrame",
        next_semantic_stage="BoreWallSearchHeuristic",
        diagnostic_keys=(
            "two_opening_bore_frame_valid",
            "two_opening_bore_frame_depth",
            "two_opening_axis_direction_preserved",
        ),
        implementation_hint="measure.BoreTwoOpeningMeasurement",
    ),
    "BoreWallSearchHeuristic": HeuristicToolContract(
        name="BoreWallSearchHeuristic",
        semantic_context="provisional_bore_frame_exists",
        input_semantic_object="RegionData + bore axis/radius/depth frame",
        operation="project faces into the bore frame and propose wall-like face evidence",
        output_proposal_type="BoreWallFaceProposal",
        next_semantic_stage="BoreWallRoleMeasurement",
        diagnostic_keys=(
            "sidewall_normal_evidence_count",
            "frame_sidewall_normal_evidence_count",
            "bore_wall_search_face_count",
        ),
        implementation_hint="recognition_component_engine sidewall/radius masks",
    ),
    "BoreWallRoleMeasurement": HeuristicToolContract(
        name="BoreWallRoleMeasurement",
        semantic_context="bore_wall_face_proposals_exist",
        input_semantic_object="BoreWallFaceProposal",
        operation="measure radial consistency, axial span, angular coverage, and normal role evidence",
        output_proposal_type="MeasuredBoreWallEvidence",
        next_semantic_stage="BoreWallOwnership",
        diagnostic_keys=(
            "learned_wall_radius",
            "aggregate_wall_axial_coverage",
            "aggregate_wall_angular_coverage",
            "radial_normal_alignment_median",
        ),
        implementation_hint="recognition_component_engine component reports",
    ),
    "BoreWallOwnership": HeuristicToolContract(
        name="BoreWallOwnership",
        semantic_context="measured_bore_wall_evidence_exists",
        input_semantic_object="MeasuredBoreWallEvidence + bore hypothesis",
        operation="promote only faces that play the cylindrical bore-wall role",
        output_proposal_type="BoreWallOwnership",
        next_semantic_stage="TrueEndpointResolver",
        authority="surface_role_ownership_after_recognition_validation",
        diagnostic_keys=(
            "bore_wall_ownership_valid",
            "bore_wall_owned_face_count",
            "bore_wall_component_rejection_reasons",
        ),
        implementation_hint="recognition_component_engine selected_seed_related_components / aggregate wall ownership",
    ),
    "TrueEndpointResolver": HeuristicToolContract(
        name="TrueEndpointResolver",
        semantic_context="bore_wall_ownership_valid",
        input_semantic_object="BoreWallOwnership + RegionData axial endpoint evidence",
        operation="resolve true damaged-bore endpoint instead of trusting provisional opposite loops",
        output_proposal_type="EndpointAuthority",
        next_semantic_stage="TerminalWallCompletionHeuristic",
        authority="endpoint_authority_after_wall_ownership_validation",
        diagnostic_keys=(
            "region_true_endpoint_depth",
            "true_endpoint_depth_gap",
            "true_endpoint_reconciled_from_region_endpoint",
            "bore_frame_depth_source",
        ),
        implementation_hint="recognition_component_engine v88/v90 true endpoint authority block",
    ),
    "TerminalWallCompletionHeuristic": HeuristicToolContract(
        name="TerminalWallCompletionHeuristic",
        semantic_context="true_endpoint_extends_beyond_owned_wall_span",
        input_semantic_object="BoreWallOwnership + EndpointAuthority",
        operation="search the missing terminal band and propose strict bore-wall role faces only",
        output_proposal_type="TerminalWallBandProposal",
        next_semantic_stage="TerminalWallOwnership",
        diagnostic_keys=(
            "terminal_wall_candidate_face_count",
            "terminal_wall_completion_rejection_reasons",
            "endpoint_support_face_count",
        ),
        implementation_hint="recognition_component_engine v90 terminal wall completion block",
    ),
    "TerminalWallOwnership": HeuristicToolContract(
        name="TerminalWallOwnership",
        semantic_context="terminal_wall_band_proposals_exist",
        input_semantic_object="TerminalWallBandProposal + BoreWallOwnership",
        operation="merge only validated terminal bore-wall faces into full-depth wall ownership",
        output_proposal_type="FullDepthBoreWallOwnership",
        next_semantic_stage="EmitBoreCandidateData",
        authority="surface_role_ownership_after_terminal_role_validation",
        diagnostic_keys=(
            "terminal_wall_face_count",
            "terminal_wall_completion_used",
            "true_endpoint_extension_added_face_count",
        ),
        implementation_hint="recognition_component_engine terminal_wall_face_ids promotion",
    ),
    "EmitBoreCandidateData": HeuristicToolContract(
        name="EmitBoreCandidateData",
        semantic_context="full_depth_bore_wall_ownership_exists",
        input_semantic_object="FullDepthBoreWallOwnership + EndpointAuthority + primitive descriptor",
        operation="emit accepted BORE CandidateData with explicit display/rebuild face IDs",
        output_proposal_type="BORE CandidateData",
        next_semantic_stage="RebuildTarget",
        authority="accepted_CandidateData_after_surface_ownership",
        diagnostic_keys=(
            "feature_family",
            "recognition_stage",
            "display_face_ids",
            "rebuild_face_ids",
        ),
        implementation_hint="recognition_component_engine._bore_candidate_contract_fields",
    ),
    "OpeningContextToChamferSearch": HeuristicToolContract(
        name="OpeningContextToChamferSearch",
        semantic_context="opening_or_mouth_region_context",
        input_semantic_object="RegionData + opening/rim context",
        operation="search near the mouth/rim for annular sloped transition bands",
        output_proposal_type="ChamferBandProposal",
        next_semantic_stage="MeasureChamferBand",
        diagnostic_keys=("chamfer_candidate_count", "chamfer_score", "seed_related"),
        implementation_hint="recognition_component_engine chamfer component scoring",
    ),
    "MeasureChamferBand": HeuristicToolContract(
        name="MeasureChamferBand",
        semantic_context="chamfer_band_proposals_exist",
        input_semantic_object="ChamferBandProposal",
        operation="measure radial/axial transition, normal mix, slope, and annular continuity",
        output_proposal_type="MeasuredChamferBandEvidence",
        next_semantic_stage="ChamferNeighborSurfaceHeuristic",
        diagnostic_keys=("radial_span", "axial_span", "normal_axis_abs_median", "radial_normal_alignment_median"),
        implementation_hint="recognition_component_engine._chamfer_score and component stats",
    ),
    "ChamferNeighborSurfaceHeuristic": HeuristicToolContract(
        name="ChamferNeighborSurfaceHeuristic",
        semantic_context="measured_chamfer_band_exists",
        input_semantic_object="MeasuredChamferBandEvidence + RegionData adjacency",
        operation="find adjacent bore-wall side, parent-surface side, and mouth/rim boundary context",
        output_proposal_type="ChamferNeighborProposal",
        next_semantic_stage="ChamferSurfaceOwnership",
        diagnostic_keys=("bore_resolution_active", "chamfer_action_demoted_by_bore_resolution"),
        implementation_hint="recognition_component_engine chamfer demotion/relationship logic",
    ),
    "ChamferSurfaceOwnership": HeuristicToolContract(
        name="ChamferSurfaceOwnership",
        semantic_context="chamfer_measurement_and_neighbor_context_valid",
        input_semantic_object="MeasuredChamferBandEvidence + ChamferNeighborProposal",
        operation="accept faces that play the chamfer transition-band role and reject bore wall/caps/context",
        output_proposal_type="ChamferBandOwnership",
        next_semantic_stage="EmitChamferCandidateData",
        authority="surface_role_ownership_after_recognition_validation",
        diagnostic_keys=("accepted_as_annular_transition_evidence", "chamfer_too_large_for_annular_transition"),
        implementation_hint="recognition_component_engine._candidate_contract_fields",
    ),
    "EmitChamferCandidateData": HeuristicToolContract(
        name="EmitChamferCandidateData",
        semantic_context="chamfer_band_ownership_exists",
        input_semantic_object="ChamferBandOwnership + measured transition evidence",
        operation="emit accepted CHAMFER CandidateData with explicit owned transition-band faces",
        output_proposal_type="CHAMFER CandidateData",
        next_semantic_stage="RebuildTarget",
        authority="accepted_CandidateData_after_surface_ownership",
        diagnostic_keys=("feature_family", "recognition_stage", "display_face_ids", "rebuild_face_ids"),
        implementation_hint="recognition_component_engine._candidate_contract_fields",
    ),
    "SelectedBoundaryToPocketRimHeuristic": HeuristicToolContract(
        name="SelectedBoundaryToPocketRimHeuristic",
        semantic_context="user_marked_possible_pocket_opening_boundary",
        input_semantic_object="selected edge/rim evidence + neutral RegionData",
        operation="propose a bounded pocket opening footprint without declaring feature identity",
        output_proposal_type="PocketOpeningBoundaryProposal",
        next_semantic_stage="MeasurePocketOpening",
        diagnostic_keys=("selected_edge_count", "normalized_edge_count", "pocket_opening_frame_resolved"),
        implementation_hint="recognition_component_engine preview-only pocket detection uses RegionData frame and selected-opening resolver",
    ),
    "MeasurePocketOpening": HeuristicToolContract(
        name="MeasurePocketOpening",
        semantic_context="pocket_opening_boundary_proposal_exists",
        input_semantic_object="PocketOpeningBoundaryProposal",
        operation="measure local pocket axis, rim plane, footprint scale, and opening confidence",
        output_proposal_type="MeasuredPocketOpeningFrame",
        next_semantic_stage="PocketFloorSurfaceSearchHeuristic",
        diagnostic_keys=("pocket_axis", "pocket_rim_level", "pocket_footprint_radius"),
        implementation_hint="recognition_component_engine._detect_pocket_preview_candidates",
    ),
    "PocketFloorSurfaceSearchHeuristic": HeuristicToolContract(
        name="PocketFloorSurfaceSearchHeuristic",
        semantic_context="measured_pocket_opening_frame_exists",
        input_semantic_object="RegionData + MeasuredPocketOpeningFrame",
        operation="search for planar recessed floor-surface evidence offset from the rim plane",
        output_proposal_type="PocketFloorProposal",
        next_semantic_stage="MeasurePocketFloor",
        diagnostic_keys=("pocket_floor_candidate_count", "pocket_floor_face_count", "pocket_floor_offset"),
        implementation_hint="floor-like components are evidence only until PocketRoleOwnership",
    ),
    "MeasurePocketFloor": HeuristicToolContract(
        name="MeasurePocketFloor",
        semantic_context="pocket_floor_proposals_exist",
        input_semantic_object="PocketFloorProposal",
        operation="measure floor axial level, planarity proxy, footprint containment, and confidence",
        output_proposal_type="MeasuredPocketFloorEvidence",
        next_semantic_stage="PocketSideWallSearchHeuristic",
        diagnostic_keys=("pocket_floor_axial_level", "pocket_floor_normal_axis_abs", "pocket_floor_radial_extent"),
        implementation_hint="recognition_component_engine component statistics",
    ),
    "PocketSideWallSearchHeuristic": HeuristicToolContract(
        name="PocketSideWallSearchHeuristic",
        semantic_context="measured_pocket_floor_evidence_exists",
        input_semantic_object="RegionData + MeasuredPocketOpeningFrame + MeasuredPocketFloorEvidence",
        operation="search for wall-like faces connecting rim level to floor level",
        output_proposal_type="PocketSideWallProposal",
        next_semantic_stage="PocketDepthResolver",
        diagnostic_keys=("pocket_side_wall_face_count", "pocket_side_wall_component_count"),
        implementation_hint="wall-like faces remain proposal/evidence until role ownership",
    ),
    "PocketDepthResolver": HeuristicToolContract(
        name="PocketDepthResolver",
        semantic_context="pocket_floor_and_side_wall_evidence_exists",
        input_semantic_object="MeasuredPocketOpeningFrame + MeasuredPocketFloorEvidence",
        operation="resolve positive pocket depth between rim plane and floor plane",
        output_proposal_type="PocketDepthEvidence",
        next_semantic_stage="PocketTransitionSeparation",
        diagnostic_keys=("pocket_depth", "pocket_depth_sign", "pocket_depth_valid"),
        implementation_hint="depth is evidence; it is not rebuild authority",
    ),
    "PocketTransitionSeparation": HeuristicToolContract(
        name="PocketTransitionSeparation",
        semantic_context="pocket_depth_evidence_exists",
        input_semantic_object="RegionData + floor/wall proposals",
        operation="separate chamfer/fillet-like transition faces near rim/floor from floor and wall ownership",
        output_proposal_type="PocketTransitionEvidence",
        next_semantic_stage="PocketRoleOwnership",
        diagnostic_keys=("pocket_transition_face_count", "pocket_transition_policy"),
        implementation_hint="transitions are excluded from floor/wall owned role sets",
    ),
    "PocketRoleOwnership": HeuristicToolContract(
        name="PocketRoleOwnership",
        semantic_context="pocket_hypothesis_has_floor_wall_depth_evidence",
        input_semantic_object="MeasuredPocketFloorEvidence + PocketSideWallProposal + PocketDepthEvidence",
        operation="transform measured evidence into owned pocket floor and side-wall surface roles",
        output_proposal_type="PocketSurfaceRoleOwnership",
        next_semantic_stage="EmitPocketCandidateData",
        authority="surface_role_ownership_after_recognition_validation",
        diagnostic_keys=("pocket_floor_face_ids", "pocket_side_wall_face_ids", "pocket_owned_face_count"),
        implementation_hint="recognition_component_engine._pocket_candidate_contract_fields",
    ),

    "PocketChildBoreBoundaryRelationship": HeuristicToolContract(
        name="PocketChildBoreBoundaryRelationship",
        semantic_context="pocket_floor_surface_role_ownership_exists",
        input_semantic_object="OwnedPocketFloorBoundaryLoops + separate BORE opening evidence",
        operation="mark inner floor boundary loops as protected child-BORE opening relationship metadata without creating a new feature family",
        output_proposal_type="PocketContainsBoreOpeningRelationship",
        next_semantic_stage="EmitPocketCandidateData",
        authority="relationship_metadata_only_not_feature_identity",
        diagnostic_keys=("pocket_protected_floor_bore_opening_count", "pocket_embedded_bore_wall_evidence_face_count"),
        implementation_hint="POCKET remains the parent candidate; BORE remains a separate candidate/rebuild object",
    ),
    "EmitPocketCandidateData": HeuristicToolContract(
        name="EmitPocketCandidateData",
        semantic_context="pocket_surface_role_ownership_exists",
        input_semantic_object="PocketSurfaceRoleOwnership + pocket primitive descriptor",
        operation="emit accepted POCKET CandidateData with separate floor and side-wall ownership plus protected child-BORE relationship metadata when present",
        output_proposal_type="POCKET CandidateData",
        next_semantic_stage="RebuildTarget",
        authority="accepted_CandidateData_after_surface_ownership",
        diagnostic_keys=("feature_family", "recognition_stage", "display_face_ids", "rebuild_face_ids", "pocket_depth"),
        implementation_hint="POCKET rebuild target consumes owned floor+wall; protected child-BORE floor openings remain holes and BORE stays separately rebuildable",
    ),
}


def _tuple_ints(values: Iterable[object] | object) -> tuple[int, ...]:
    try:
        raw = tuple(values or ())  # type: ignore[arg-type]
    except Exception:
        return ()
    out: list[int] = []
    for value in raw:
        try:
            out.append(int(value))
        except Exception:
            continue
    return tuple(out)


def _tuple_strs(values: Iterable[object] | object) -> tuple[str, ...]:
    try:
        raw = tuple(values or ())  # type: ignore[arg-type]
    except Exception:
        return ()
    return tuple(str(v) for v in raw if str(v))


def heuristic_contract(name: str) -> dict[str, object]:
    """Return one registered heuristic contract as a dictionary."""

    item = HEURISTIC_TOOL_REGISTRY.get(str(name))
    if item is None:
        return {
            "contract_type": "heuristic_tool_contract",
            "contract_version": HEURISTIC_CONTRACT_VERSION,
            "heuristic_name": str(name),
            "authority": HEURISTIC_AUTHORITY,
            "forbidden_authority": HEURISTIC_FORBIDDEN_AUTHORITY,
            "missing_from_registry": True,
        }
    return item.to_dict()


def heuristic_registry_dict() -> dict[str, object]:
    """Return the full heuristic registry in a serializable shape."""

    return {
        "contract_type": "heuristic_tool_registry",
        "contract_version": HEURISTIC_CONTRACT_VERSION,
        "authority": HEURISTIC_AUTHORITY,
        "forbidden_authority": HEURISTIC_FORBIDDEN_AUTHORITY,
        "bore_recipe": BORE_HEURISTIC_RECIPE,
        "chamfer_recipe": CHAMFER_HEURISTIC_RECIPE,
        "pocket_recipe": POCKET_HEURISTIC_RECIPE,
        "tools": {name: contract.to_dict() for name, contract in sorted(HEURISTIC_TOOL_REGISTRY.items())},
    }


def recipe_contracts(recipe: str = "bore") -> tuple[dict[str, object], ...]:
    """Return registered contracts in the order of a semantic recipe."""

    recipe_name = str(recipe).strip().lower()
    if recipe_name.startswith("chamfer"):
        names = CHAMFER_HEURISTIC_RECIPE
    elif recipe_name.startswith("pocket"):
        names = POCKET_HEURISTIC_RECIPE
    else:
        names = BORE_HEURISTIC_RECIPE
    return tuple(heuristic_contract(name) for name in names)


def make_heuristic_result(
    heuristic_name: str,
    *,
    semantic_context: str | None = None,
    output_proposal_type: str | None = None,
    input_count: int = 0,
    proposal_count: int = 0,
    accepted_count: int = 0,
    rejected_count: int = 0,
    confidence: float | None = None,
    proposal_face_ids: Iterable[object] = (),
    proposal_edge_ids: Iterable[object] = (),
    rejection_reasons: Iterable[object] = (),
    diagnostics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create a runtime heuristic diagnostic row.

    The row remains evidence/proposal metadata.  It includes explicit negative
    authority flags so downstream code cannot accidentally treat it as ownership
    or rebuild permission.
    """

    contract = HEURISTIC_TOOL_REGISTRY.get(str(heuristic_name))
    return HeuristicInvocationResult(
        heuristic_name=str(heuristic_name),
        semantic_context=str(semantic_context if semantic_context is not None else (contract.semantic_context if contract else "unknown")),
        output_proposal_type=str(output_proposal_type if output_proposal_type is not None else (contract.output_proposal_type if contract else "EvidenceProposal")),
        input_count=int(input_count),
        proposal_count=int(proposal_count),
        accepted_count=int(accepted_count),
        rejected_count=int(rejected_count),
        confidence=confidence,
        proposal_face_ids=_tuple_ints(tuple(proposal_face_ids or ())),
        proposal_edge_ids=_tuple_ints(tuple(proposal_edge_ids or ())),
        rejection_reasons=_tuple_strs(tuple(rejection_reasons or ())),
        diagnostics=dict(diagnostics or {}),
        authority=str(contract.authority if contract else HEURISTIC_AUTHORITY),
        forbidden_authority=tuple(contract.forbidden_authority if contract else HEURISTIC_FORBIDDEN_AUTHORITY),
    ).to_dict()


def compact_heuristic_summary(results: Iterable[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    """Return small readable rows for UI/debug output."""

    out: list[dict[str, object]] = []
    for item in tuple(results or ()):  # type: ignore[arg-type]
        if not isinstance(item, Mapping):
            continue
        out.append({
            "heuristic_name": str(item.get("heuristic_name", "")),
            "output_proposal_type": str(item.get("output_proposal_type", "")),
            "authority": str(item.get("authority", HEURISTIC_AUTHORITY)),
            "proposal_count": int(item.get("proposal_count", 0) or 0),
            "accepted_count": int(item.get("accepted_count", 0) or 0),
            "rejected_count": int(item.get("rejected_count", 0) or 0),
            "proposal_face_count": int(item.get("proposal_face_count", 0) or 0),
            "proposal_edge_count": int(item.get("proposal_edge_count", 0) or 0),
            "rejection_reasons": tuple(item.get("rejection_reasons", ()) or ()),
        })
    return tuple(out)


__all__ = [
    "HEURISTIC_CONTRACT_VERSION",
    "HEURISTIC_AUTHORITY",
    "HEURISTIC_FORBIDDEN_AUTHORITY",
    "BORE_HEURISTIC_RECIPE",
    "CHAMFER_HEURISTIC_RECIPE",
    "POCKET_HEURISTIC_RECIPE",
    "HEURISTIC_TOOL_REGISTRY",
    "HeuristicToolContract",
    "HeuristicInvocationResult",
    "heuristic_contract",
    "heuristic_registry_dict",
    "recipe_contracts",
    "make_heuristic_result",
    "compact_heuristic_summary",
]
