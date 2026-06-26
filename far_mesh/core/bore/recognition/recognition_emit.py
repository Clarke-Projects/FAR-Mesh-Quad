"""Candidate result emission helpers for FAR MESH BoreTool recognition.

This module owns shared CandidateData/result packaging for the recognition
stage.  It does not classify features, assign surface-role ownership, authorize
DeletePatchProposal, build rebuild targets, or mutate mesh topology.

v174s starts the emission split by moving the public CandidateResult-like
packing helper out of ``recognition_component_engine.py`` while keeping the
component engine's public wrapper and return shape unchanged.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from ..types import FeatureFamily, RecognitionStage

RECOGNITION_EMIT_SPLIT_CHECKPOINT_V174S = (
    "v174s_split_shared_candidate_result_emission_no_behavior_change"
)

_ALLOWED_PROMOTED_FAMILIES_V174S: tuple[str, ...] = (
    FeatureFamily.BORE.value,
    FeatureFamily.CHAMFER_FORM.value,
    FeatureFamily.POCKET.value,
    FeatureFamily.CIRCULAR_POCKET.value,
)


def promoted_candidate_rows_v174s(
    features: Iterable[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Return the same promoted CandidateData rows used before v174s.

    This is shared emission bookkeeping only.  It consumes CandidateData rows
    already emitted by family-local recognizers; it does not create candidates,
    alter actionability, change rebuild faces, or participate in target policy.
    """

    return tuple(
        item for item in tuple(features or ())
        if bool(item.get("candidate_action_enabled", item.get("rebuild_authorized", False)))
        and str(item.get("recognition_stage", "") or "") == RecognitionStage.ACCEPTED_CANDIDATE.value
        and str(item.get("feature_family", "") or "") in set(_ALLOWED_PROMOTED_FAMILIES_V174S)
    )


def recognition_result_dict_from_component_features_v174s(
    *,
    features: tuple[dict[str, object], ...],
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Return the preserved CandidateResult-like dictionary for GUI diagnostics.

    The field names, engine string, mode string, candidate tuple, diagnostics
    pass-through, and promoted-count policy intentionally match the pre-v174s
    component-engine implementation.
    """

    promoted = promoted_candidate_rows_v174s(tuple(features or ()))
    return {
        "contract_type": "candidate_result",
        "engine": "surface_component_classifier_v105_selected_rail_support_count_chamfer_ownership",
        "mode": "selected_annular_rail_to_family_local_ownership_candidates",
        "candidate_count": int(len(tuple(features or ()))),
        "candidate_data": tuple(features or ()),
        "features": tuple(features or ()),
        "diagnostics": dict(diagnostics or {}),
        "promoted_candidate_count": int(len(promoted)),
    }


def recognition_emit_split_status_v174s() -> dict[str, object]:
    """Return read-only diagnostics for the v174s emission-module split."""

    return {
        "recognition_emit_split_checkpoint_v174s": RECOGNITION_EMIT_SPLIT_CHECKPOINT_V174S,
        "recognition_component_engine_refactor_mode_v174s": "split_shared_candidate_result_emission_no_candidate_behavior_change",
        "recognition_result_dict_source_v174s": "recognition_emit.recognition_result_dict_from_component_features_v174s",
        "recognition_component_engine_result_wrapper_preserved_v174s": True,
        "recognition_emit_only_no_candidate_promotion_change_v174s": True,
        "v174s_no_behavior_change_contract": True,
        "v174s_not_feature_recognition": True,
        "v174s_not_surface_ownership": True,
        "v174s_not_candidate_promotion": True,
        "v174s_not_delete_patch_policy": True,
        "v174s_not_mesh_mutation": True,
    }
