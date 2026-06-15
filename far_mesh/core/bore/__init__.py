"""Public Bore core API.

Hard-clean Bore recognition package.

Active architecture:
    region_select.py -> RegionData only
    recognition.py -> RegionData-to-CandidateData bridge
    recognition_component_engine.py -> physical CandidateData classifier
    rebuild_target.py -> DeletePatchProposal policy
    rebuild.py -> RebuildResult / topology validation

Removed:
    removed pre-component candidate pipeline
    parent-minus-chamfer fallback
    guarded all-RegionData borehole fallback
    post-hoc recognition repair helpers
"""

from __future__ import annotations

from .types import (
    RegionData,
    RegionEvidenceLedger,
    CandidateData,
    CandidateResult,
    DeletePatchProposal,
    FeatureFamily,
    RecognitionStage,
    EvidenceKind,
    FeaturePrimitiveKind,
    FeaturePrimitiveData,
    FeatureRelationshipKind,
    FeatureRelationshipData,
    FeatureEvidenceItem,
    FeatureEvidenceLedger,
    X1_FREECAD_TO_FAR_MESH_DICTIONARY,
    EdgeKey,
    Vector3,
)
from .exceptions import (
    BoreError,
    BoreRecognitionError,
    BoreRebuildRejected,
    BoreTargetInvalid,
    BoreTopologyError,
)
from .rebuild_target import (
    build_bounded_rebuild_target_face_sets,
    build_rebuild_target_contract_for_feature,
    candidate_can_request_delete_patch,
    prepare_rebuild_target,
    target_from_candidate_dict,
)
from .rebuild import RebuildResult, delete_and_rebuild_candidate_region
from .region_select import (
    BoreRegionSelection,
    region_faces,
    faces_inside_boundary,
    select_bore_region,
    select_region_data,
)
from .geometry import (
    BoundaryLoopGeometry,
    BoundaryLoopStackGeometry,
    describe_boundary_loop_geometry,
    describe_boundary_loop_stack_geometry,
    boundary_loop_spatial_families,
    CylinderFaceEvidence,
    FeaturePatchMeasurement,
    measure_faces_against_cylinder,
    measure_feature_patch_geometry,
)
from .recognition import (
    ACTIVE_CANDIDATE_AUTHORITY,
    ASSEMBLY_CLASSIFICATION_POLICY,
    REGION_SELECT_FEATURE_AUTHORITY,
    recognize_bore_region_selection,
)
from .recognition_component_engine import (
    component_engine_feature_candidates,
    recognition_result_dict_from_component_features,
)
from .measure import (
    BoreOpeningMeasurement,
    BoreRegionMeasurement,
    measure_bore_from_region_faces,
    measure_bore_opening,
    measure_bore_opening_candidates,
    measure_bore_region,
)
from .tool import (
    BoreCandidateView,
    BoreInsideBoundaryPreview,
    BoreToolDisplayResult,
    BoreToolRuntime,
    analyze_bore_candidates,
    format_bore_diagnostics,
    preview_faces_inside_boundary,
    rebuild_bore_candidate,
)


__all__ = [
    "ACTIVE_CANDIDATE_AUTHORITY",
    "ASSEMBLY_CLASSIFICATION_POLICY",
    "REGION_SELECT_FEATURE_AUTHORITY",
    "RegionData",
    "RegionEvidenceLedger",
    "CandidateData",
    "CandidateResult",
    "DeletePatchProposal",
    "FeatureFamily",
    "RecognitionStage",
    "EvidenceKind",
    "FeaturePrimitiveKind",
    "FeaturePrimitiveData",
    "FeatureRelationshipKind",
    "FeatureRelationshipData",
    "FeatureEvidenceItem",
    "FeatureEvidenceLedger",
    "X1_FREECAD_TO_FAR_MESH_DICTIONARY",
    "BoreRegionSelection",
    "BoreOpeningMeasurement",
    "BoreRegionMeasurement",
    "faces_inside_boundary",
    "region_faces",
    "select_bore_region",
    "select_region_data",
    "measure_bore_opening",
    "measure_bore_opening_candidates",
    "measure_bore_region",
    "measure_bore_from_region_faces",
    "candidate_can_request_delete_patch",
    "delete_and_rebuild_candidate_region",
    "BoundaryLoopGeometry",
    "BoundaryLoopStackGeometry",
    "describe_boundary_loop_geometry",
    "describe_boundary_loop_stack_geometry",
    "boundary_loop_spatial_families",
    "CylinderFaceEvidence",
    "FeaturePatchMeasurement",
    "measure_faces_against_cylinder",
    "measure_feature_patch_geometry",
    "EdgeKey",
    "Vector3",
    "BoreError",
    "BoreRecognitionError",
    "BoreRebuildRejected",
    "BoreTargetInvalid",
    "BoreTopologyError",
    "recognize_bore_region_selection",
    "component_engine_feature_candidates",
    "recognition_result_dict_from_component_features",
    "build_bounded_rebuild_target_face_sets",
    "build_rebuild_target_contract_for_feature",
    "prepare_rebuild_target",
    "target_from_candidate_dict",
    "BoreCandidateView",
    "BoreInsideBoundaryPreview",
    "BoreToolDisplayResult",
    "BoreToolRuntime",
    "analyze_bore_candidates",
    "preview_faces_inside_boundary",
    "rebuild_bore_candidate",
    "format_bore_diagnostics",
    "RebuildResult",
]
