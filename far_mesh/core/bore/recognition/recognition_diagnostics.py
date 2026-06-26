"""Read-only recognition diagnostics for FAR MESH BoreTool.

This module contains diagnostic/reporting helpers for the recognition stage.
It does not classify features, assign ownership, emit CandidateData, authorize
DeletePatchProposal, or mutate mesh topology.

v174q starts the diagnostics split by moving the surface-role evidence graph out
of ``recognition_component_engine.py`` while preserving the exact diagnostic keys
and semantics used by the component engine. v174r moves the static authority map
source-of-truth here. v174t adds an explicit module-role/orchestrator inventory
so the component engine can be treated as fanout/orchestration without changing
CandidateData behavior. v174u adds a read-only inventory of remaining local
helpers and compatibility wrappers before any deletion or behavior change. v174w begins actual line-count reduction by migrating internal call sites to direct family/diagnostic module helpers and removing compatibility wrappers from the component engine. v175a moves the final component-engine diagnostic assembly here so recognition_component_engine.py can keep shrinking toward orchestration/fanout only. v175b moves the remaining static authority/checkpoint diagnostic assembly here, reducing recognition_component_engine.py to orchestration/fanout plus public wrappers. v175f surfaces the diagnostic-only SecondaryOpeningRimSearchHeuristic so compound CHAMFER/BORE/POCKET evidence is proposed without becoming identity, ownership, CandidateData, or rebuild authority.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from ..types import FeatureFamily, RecognitionStage, tuple_ints
from ..heuristics import (
    HEURISTIC_CONTRACT_VERSION,
    BORE_HEURISTIC_RECIPE,
    CHAMFER_HEURISTIC_RECIPE,
    POCKET_HEURISTIC_RECIPE,
    compact_heuristic_summary,
    heuristic_registry_dict,
)

from .recognition_bore import (
    BORE_RECOGNITION_CANDIDATE_ASSEMBLY_SPLIT_CHECKPOINT_V174Z,
    BORE_RECOGNITION_CANDIDATE_CONTRACT_SPLIT_CHECKPOINT_V174K,
    BORE_RECOGNITION_OPENING_LOCALITY_HELPER_SPLIT_CHECKPOINT_V174J,
    BORE_RECOGNITION_OWNERSHIP_HELPER_SPLIT_CHECKPOINT_V174I,
    BORE_RECOGNITION_SPLIT_CHECKPOINT_V174H,
)
from .recognition_chamfer import (
    CHAMFER_RECOGNITION_CANDIDATE_ASSEMBLY_SPLIT_CHECKPOINT_V174Y,
    CHAMFER_RECOGNITION_CANDIDATE_CONTRACT_SPLIT_CHECKPOINT_V174P,
    CHAMFER_RECOGNITION_LEGACY_FALLBACK_SPLIT_CHECKPOINT_V174G,
    CHAMFER_RECOGNITION_PROOF_HELPER_SPLIT_CHECKPOINT_V174F,
    CHAMFER_RECOGNITION_SPLIT_CHECKPOINT_V174E,
)
from .recognition_common import (
    RECOGNITION_COMPONENT_ENGINE_SHARED_HELPER_SPLIT_CHECKPOINT_V174X,
    SECONDARY_OPENING_RIM_SEARCH_HEURISTIC_CHECKPOINT_V175F,
)
from .recognition_emit import recognition_emit_split_status_v174s
from .recognition_pocket import (
    POCKET_RECOGNITION_AUTHORITY_LEAK_DIAGNOSTIC_CHECKPOINT_V174N,
    POCKET_RECOGNITION_DETECTOR_SPLIT_CHECKPOINT_V174O,
    POCKET_RECOGNITION_SPLIT_CHECKPOINT_V174L,
    POCKET_RECOGNITION_X1_PROBE_HELPER_SPLIT_CHECKPOINT_V174M,
)

RECOGNITION_DIAGNOSTICS_SPLIT_CHECKPOINT_V174Q = (
    "v174q_split_surface_role_evidence_graph_diagnostics_no_behavior_change"
)

SURFACE_ROLE_EVIDENCE_GRAPH_CONTRACT_V173B = (
    "surface_role_evidence_graph_v173b_roles_before_candidate_authority_relationships_do_not_block_parent_roles"
)


COMPONENT_ENGINE_AUTHORITY_MAP_DIAGNOSTICS_SPLIT_CHECKPOINT_V174R = (
    "v174r_split_static_component_engine_authority_map_diagnostics_no_behavior_change"
)

RECOGNITION_COMPONENT_ENGINE_ORCHESTRATOR_THINNING_CHECKPOINT_V174T = (
    "v174t_component_engine_orchestrator_inventory_no_behavior_change"
)

RECOGNITION_COMPONENT_ENGINE_LOCAL_HELPER_INVENTORY_CHECKPOINT_V174U = (
    "v174u_component_engine_remaining_local_helper_inventory_no_behavior_change"
)

RECOGNITION_REGRESSION_SMOKE_CHECKPOINT_V174V = (
    "v174v_recognition_regression_smoke_checkpoint_no_behavior_change"
)

RECOGNITION_COMPONENT_ENGINE_WRAPPER_CALLSITE_MIGRATION_CHECKPOINT_V174W = (
    "v174w_migrate_internal_wrapper_call_sites_and_remove_compatibility_wrappers_no_behavior_change"
)

RECOGNITION_FINAL_DIAGNOSTICS_ASSEMBLY_SPLIT_CHECKPOINT_V175A = (
    "v175a_final_component_engine_diagnostics_assembly_split_no_behavior_change"
)

RECOGNITION_STATIC_AUTHORITY_DIAGNOSTICS_SPLIT_CHECKPOINT_V175B = (
    "v175b_static_authority_diagnostics_assembly_split_no_behavior_change"
)


RECOGNITION_SECONDARY_OPENING_RIM_HEURISTIC_DIAGNOSTICS_CHECKPOINT_V175F = (
    "v175f_secondary_opening_rim_search_heuristic_diagnostics_evidence_only"
)

RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174A = (
    "v174a_behavior_preserving_authority_map_before_module_extraction"
)

STABLE_AUTHORITY_PATHS_V174A: tuple[tuple[str, str, str], ...] = (
    (
        "selection",
        "v173l_selected_annular_rail_lineage",
        "clicked/raw edge evidence is measured into selected annular rail evidence before family ownership",
    ),
    (
        "chamfer",
        "v173t_rail_to_rail_bounded_surface_ownership",
        "CHAMFER CandidateData face export comes from owned CHAMFER_BAND faces, not raw connected angled topology",
    ),
    (
        "bore",
        "v173w_terminal_continuation_ownership_audit",
        "terminal continuation may remain diagnostic evidence but cannot enter BORE_WALL ownership unless its role audit passes",
    ),
    (
        "pocket",
        "family_local_pocket_floor_sidewall_ownership",
        "POCKET owns floor/side-wall roles while child BORE relations remain protected metadata",
    ),
    (
        "emit",
        "CandidateData_surface_role_export",
        "CandidateData is emitted only after family-local ownership and remains separate from DeletePatchProposal",
    ),
)

LEGACY_PATH_QUARANTINE_CANDIDATES_V174A: tuple[str, ...] = (
    "v173o_chamfer_local_transition_band_review_demoter",
    "v173p_chamfer_owned_subset_mapping_attempt",
    "v173q_chamfer_rail_coordinate_mapping_fallback",
    "v173r_selected_rail_support_count_authority_attempt",
    "v173s_boundary_contained_annular_slope_attempt",
    "raw_mt_ids_chamfer_fallback_export",
    "terminal_continuation_without_passed_bore_wall_audit",
    "cross_family_negative_definition_or_subtraction",
)


def component_engine_refactor_authority_map_v174r() -> dict[str, object]:
    """Return the static v174a cleanup authority map from diagnostics module.

    This is read-only diagnostic metadata.  It does not inspect RegionData, does
    not choose candidates, does not alter family-local ownership, and does not
    participate in rebuild target policy.  v174r only moves the static authority
    map out of ``recognition_component_engine.py`` so the component engine can
    continue shrinking toward orchestration/fanout only.
    """

    return {
        "recognition_component_engine_refactor_checkpoint_v174a": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174A,
        "recognition_component_engine_refactor_mode_v174a": "audit_only_no_candidate_behavior_change",
        "recognition_component_engine_next_cleanup_step_v174a": (
            "extract final diagnostics/authority bookkeeping, then isolate CHAMFER v173t candidate emission"
        ),
        "stable_authority_paths_v174a": tuple(
            {
                "semantic_area": area,
                "authority": authority,
                "meaning": meaning,
            }
            for area, authority, meaning in STABLE_AUTHORITY_PATHS_V174A
        ),
        "legacy_path_quarantine_candidates_v174a": LEGACY_PATH_QUARANTINE_CANDIDATES_V174A,
        "v174a_no_behavior_change_contract": True,
        "v174a_not_feature_recognition": True,
        "v174a_not_candidate_promotion": True,
        "v174a_not_delete_patch_policy": True,
        "v174a_not_mesh_mutation": True,
    }


def component_engine_authority_map_split_status_v174r() -> dict[str, object]:
    """Return the v174r diagnostics-split status row.

    This row is intentionally separate from the preserved v174a keys so runtime
    logs can prove the module move happened without changing what v174a meant.
    """

    return {
        "recognition_diagnostics_authority_map_split_checkpoint_v174r": COMPONENT_ENGINE_AUTHORITY_MAP_DIAGNOSTICS_SPLIT_CHECKPOINT_V174R,
        "recognition_component_engine_refactor_mode_v174r": "split_static_authority_map_diagnostics_no_candidate_behavior_change",
        "recognition_component_engine_authority_map_source_v174r": "recognition_diagnostics.component_engine_refactor_authority_map_v174r",
        "recognition_component_engine_v174a_authority_map_wrapper_preserved_v174r": True,
        "recognition_diagnostics_only_no_candidate_promotion_v174r": True,
        "v174r_no_behavior_change_contract": True,
        "v174r_not_feature_recognition": True,
        "v174r_not_candidate_promotion": True,
        "v174r_not_delete_patch_policy": True,
        "v174r_not_mesh_mutation": True,
    }



FAMILY_LOCAL_OWNERSHIP_CONTRACT_V173C = (
    "family_local_candidate_ownership_v173c_no_cross_family_candidate_creation_relationships_are_metadata"
)

SELECTED_ANNULAR_RAIL_CONTRACT_V173D = (
    "selected_annular_rail_role_resolver_v173d_selected_edges_are_neutral_rail_evidence_before_family_ownership"
)

# v174r moves the static v174a authority-map data into recognition_diagnostics.py.
# These imported names remain available in this module for compatibility and
# for existing diagnostic consumers, but the source of truth is now the
# diagnostics module.


# v174 diagnostic checkpoint labels still referenced by the static component-engine
# authority map.  They are diagnostic strings only; they do not classify features,
# assign ownership, emit CandidateData, or participate in rebuild target policy.
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174B = (
    "v174b_extract_static_diagnostics_and_chamfer_export_selection_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174C = (
    "v174c_quarantine_legacy_chamfer_fallbacks_and_prepare_terminal_audit_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174D = (
    "v174d_extract_terminal_continuation_filter_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174E = (
    "v174e_first_chamfer_module_split_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174F = (
    "v174f_chamfer_proof_helper_split_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174G = (
    "v174g_chamfer_legacy_fallback_split_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174H = (
    "v174h_first_bore_module_split_terminal_filter_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174I = (
    "v174i_bore_ownership_helper_split_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174J = (
    "v174j_bore_opening_locality_helper_split_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174K = (
    "v174k_bore_candidate_contract_helper_split_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174L = (
    "v174l_first_pocket_module_split_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174M = (
    "v174m_pocket_x1_probe_helper_split_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174N = (
    "v174n_pocket_authority_leak_diagnostics_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174O = (
    "v174o_pocket_detector_orchestration_split_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174P = (
    "v174p_chamfer_candidate_contract_helper_split_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174T = (
    "v174t_component_engine_orchestrator_inventory_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174U = (
    "v174u_component_engine_local_helper_inventory_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174V = (
    "v174v_recognition_regression_smoke_checkpoint_no_behavior_change"
)
RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174Y = (
    "v174y_chamfer_candidate_row_assembly_split_no_behavior_change"
)
LEGACY_CHAMFER_FALLBACK_STATUS_V174C: tuple[dict[str, object], ...] = (
    {"path": "v173o", "status": "diagnostic_preproof_only", "candidate_export_authority": False},
    {"path": "v173p", "status": "quarantined_fallback_export_preserved", "candidate_export_authority": "fallback_only_when_v173t_absent"},
    {"path": "v173q", "status": "quarantined_fallback_export_preserved", "candidate_export_authority": "fallback_only_when_v173t_absent"},
    {"path": "v173r", "status": "measurement_diagnostic_only", "candidate_export_authority": False},
    {"path": "v173s", "status": "coordinate_mapping_diagnostic_inside_v173q", "candidate_export_authority": "fallback_only_via_v173q"},
)


def legacy_chamfer_quarantine_diagnostics_v175b() -> dict[str, object]:
    """Return the v174c CHAMFER legacy-path quarantine map.

    The v173o/p/q/r/s helpers are intentionally not deleted in v174c because
    v174 is still a behavior-preserving cleanup.  This diagnostic map makes
    their authority explicit: v173t is the preferred CHAMFER_BAND export, while
    v173q/v173p remain named fallback paths only until regression tests prove
    they can be removed.
    """

    return {
        "recognition_component_engine_refactor_checkpoint_v174c": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174C,
        "recognition_component_engine_refactor_mode_v174c": "legacy_chamfer_quarantine_and_terminal_audit_prep_no_candidate_behavior_change",
        "chamfer_primary_candidate_export_authority_v174c": "v173t_rail_to_rail_bounded_surface_ownership",
        "legacy_chamfer_fallback_quarantine_v174c": True,
        "legacy_chamfer_fallback_status_v174c": LEGACY_CHAMFER_FALLBACK_STATUS_V174C,
        "legacy_chamfer_fallback_policy_v174c": (
            "v173o/v173r/v173s are diagnostics/preproof only; v173q/v173p are preserved fallback exports only when "
            "v173t does not provide owned CHAMFER_BAND faces. No raw connected mt_ids may become CandidateData export."
        ),
        "v174c_no_behavior_change_contract": True,
        "v174c_not_feature_recognition": True,
        "v174c_not_candidate_promotion": True,
        "v174c_not_delete_patch_policy": True,
        "v174c_not_mesh_mutation": True,
    }


def bore_terminal_continuation_audit_contract_v175b() -> dict[str, object]:
    """Return the v174c BORE terminal-continuation extraction-prep contract.

    v174c does not alter the v173w audit behavior.  It simply publishes the
    contract that the later extraction must preserve before the audit block is
    moved into a smaller BORE-local helper/module.
    """

    return {
        "bore_terminal_continuation_extraction_prep_v174c": True,
        "bore_terminal_continuation_current_authority_v174c": "v173w_terminal_continuation_ownership_audit",
        "bore_terminal_continuation_semantic_rule_v174c": (
            "Terminal continuation is evidence only until its own BORE_WALL role audit passes; failed terminal faces must not "
            "enter preview, ownership, CandidateData, DeletePatchProposal, or rebuild target faces."
        ),
        "bore_terminal_continuation_next_cleanup_step_v174c": (
            "extract the v173w terminal-continuation filter into a BORE-local helper after CHAMFER fallback quarantine is validated"
        ),
    }


def component_engine_static_authority_diagnostics_v175b() -> dict[str, object]:
    """Return stable top-level authority diagnostics for the component engine.

    v174b starts moving final diagnostic bookkeeping out of the giant
    ``component_engine_feature_candidates`` body.  This helper is intentionally
    static and read-only: it does not inspect RegionData, does not promote
    candidates, and does not alter family-local ownership.
    """

    return {
        "component_engine_version": 109,
        "recognition_static_authority_diagnostics_split_checkpoint_v175b": RECOGNITION_STATIC_AUTHORITY_DIAGNOSTICS_SPLIT_CHECKPOINT_V175B,
        "recognition_component_engine_static_authority_source_v175b": "recognition_diagnostics.component_engine_static_authority_diagnostics_v175b",
        "recognition_component_engine_line_count_after_v175b": 554,
        "v175b_no_behavior_change_contract": True,
        "v175b_not_feature_recognition": True,
        "v175b_not_surface_ownership": True,
        "v175b_not_candidate_promotion": True,
        "v175b_not_delete_patch_policy": True,
        "v175b_not_mesh_mutation": True,
        "active_candidate_authority": "surface_component_classifier_v109_bore_opening_to_exit_wall_ownership_terminal_continuation_audit",
        "family_local_ownership_contract_v173c": FAMILY_LOCAL_OWNERSHIP_CONTRACT_V173C,
        "selected_annular_rail_contract_v173d": SELECTED_ANNULAR_RAIL_CONTRACT_V173D,
        **component_engine_refactor_authority_map_v174r(),
        "recognition_component_engine_refactor_checkpoint_v174b": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174B,
        "recognition_component_engine_refactor_mode_v174b": "extract_static_diagnostics_and_chamfer_export_selection_no_candidate_behavior_change",
        "recognition_component_engine_next_cleanup_step_v174b": (
            "quarantine legacy CHAMFER v173o/p/q/r/s helpers behind explicit diagnostic-only names, then extract BORE terminal-continuation audit"
        ),
        **legacy_chamfer_quarantine_diagnostics_v175b(),
        **bore_terminal_continuation_audit_contract_v175b(),
        "recognition_component_engine_next_cleanup_step_v174c": (
            "extract BORE terminal-continuation audit/filter into a helper, then split CHAMFER/BORE ownership builders from orchestration"
        ),
        "recognition_component_engine_refactor_checkpoint_v174d": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174D,
        "recognition_component_engine_refactor_mode_v174d": "extract_bore_terminal_continuation_filter_no_candidate_behavior_change",
        "bore_terminal_continuation_filter_helper_extracted_v174d": True,
        "recognition_component_engine_next_cleanup_step_v174d": (
            "begin first module split by moving CHAMFER rail-to-rail owned-band logic into recognition_chamfer.py"
        ),
        "recognition_component_engine_refactor_checkpoint_v174e": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174E,
        "recognition_chamfer_split_checkpoint_v174e": CHAMFER_RECOGNITION_SPLIT_CHECKPOINT_V174E,
        "recognition_component_engine_refactor_mode_v174e": "first_module_split_chamfer_rail_to_rail_ownership_no_candidate_behavior_change",
        "recognition_chamfer_module_split_v174e": True,
        "recognition_chamfer_candidate_emission_still_in_component_engine_v174e": True,
        "recognition_component_engine_next_cleanup_step_v174e": (
            "move more CHAMFER-local proof helpers to recognition_chamfer.py after regression confirms v173t export is unchanged"
        ),
        "recognition_component_engine_refactor_checkpoint_v174f": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174F,
        "recognition_chamfer_proof_helper_split_checkpoint_v174f": CHAMFER_RECOGNITION_PROOF_HELPER_SPLIT_CHECKPOINT_V174F,
        "recognition_component_engine_refactor_mode_v174f": "split_chamfer_role_proof_helpers_no_candidate_behavior_change",
        "recognition_chamfer_role_proof_helpers_split_v174f": True,
        "recognition_chamfer_candidate_emission_still_in_component_engine_v174f": True,
        "recognition_component_engine_next_cleanup_step_v174f": (
            "move quarantined CHAMFER fallback evaluators v173p/v173q into recognition_chamfer.py or remove them after regression proves they are unused"
        ),
        "recognition_component_engine_refactor_checkpoint_v174g": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174G,
        "recognition_chamfer_legacy_fallback_split_checkpoint_v174g": CHAMFER_RECOGNITION_LEGACY_FALLBACK_SPLIT_CHECKPOINT_V174G,
        "recognition_component_engine_refactor_mode_v174g": "split_quarantined_chamfer_legacy_fallbacks_no_candidate_behavior_change",
        "recognition_chamfer_legacy_fallback_helpers_split_v174g": True,
        "recognition_chamfer_candidate_emission_still_in_component_engine_v174g": True,
        "recognition_component_engine_next_cleanup_step_v174g": (
            "start BORE split by moving the already-extracted v173w terminal-continuation filter into recognition_bore.py"
        ),
        "recognition_component_engine_refactor_checkpoint_v174h": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174H,
        "recognition_bore_split_checkpoint_v174h": BORE_RECOGNITION_SPLIT_CHECKPOINT_V174H,
        "recognition_component_engine_refactor_mode_v174h": "first_bore_module_split_terminal_continuation_filter_no_candidate_behavior_change",
        "recognition_bore_terminal_continuation_filter_split_v174h": True,
        "recognition_bore_candidate_emission_still_in_component_engine_v174h": True,
        "recognition_component_engine_next_cleanup_step_v174h": (
            "move more BORE-local ownership helpers into recognition_bore.py after regression confirms terminal-continuation audit is unchanged"
        ),
        "recognition_component_engine_refactor_checkpoint_v174i": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174I,
        "recognition_bore_ownership_helper_split_checkpoint_v174i": BORE_RECOGNITION_OWNERSHIP_HELPER_SPLIT_CHECKPOINT_V174I,
        "recognition_component_engine_refactor_mode_v174i": "split_bore_wall_subset_and_post_isolation_validation_helpers_no_candidate_behavior_change",
        "recognition_bore_wall_subset_helper_split_v174i": True,
        "recognition_bore_post_isolation_validation_helper_split_v174i": True,
        "recognition_bore_candidate_emission_still_in_component_engine_v174i": True,
        "recognition_component_engine_next_cleanup_step_v174i": (
            "continue BORE split by moving opening-anchor and seed-island evidence helpers into recognition_bore.py while preserving CandidateData emission"
        ),
        "recognition_component_engine_refactor_checkpoint_v174j": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174J,
        "recognition_bore_opening_locality_helper_split_checkpoint_v174j": BORE_RECOGNITION_OPENING_LOCALITY_HELPER_SPLIT_CHECKPOINT_V174J,
        "recognition_component_engine_refactor_mode_v174j": "split_selected_opening_locality_and_seed_island_helpers_no_candidate_behavior_change",
        "recognition_bore_seed_neighborhood_helper_split_v174j": True,
        "recognition_bore_seed_island_helper_split_v174j": True,
        "recognition_bore_opening_anchor_helper_split_v174j": True,
        "recognition_bore_candidate_emission_still_in_component_engine_v174j": True,
        "recognition_component_engine_next_cleanup_step_v174j": (
            "extract BORE candidate emission after regression confirms opening-locality evidence is unchanged"
        ),
        "recognition_component_engine_refactor_checkpoint_v174k": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174K,
        "recognition_bore_candidate_contract_split_checkpoint_v174k": BORE_RECOGNITION_CANDIDATE_CONTRACT_SPLIT_CHECKPOINT_V174K,
        "recognition_component_engine_refactor_mode_v174k": "split_bore_candidate_contract_helpers_no_candidate_behavior_change",
        "recognition_bore_candidate_contract_helpers_split_v174k": True,
        "recognition_bore_final_candidate_list_assembly_still_in_component_engine_v174k": True,
        "recognition_component_engine_next_cleanup_step_v174k": (
            "start POCKET split by moving child-bore intrusion filtering and POCKET CandidateData contract helpers into recognition_pocket.py"
        ),
        "recognition_component_engine_refactor_checkpoint_v174l": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174L,
        "recognition_pocket_split_checkpoint_v174l": POCKET_RECOGNITION_SPLIT_CHECKPOINT_V174L,
        "recognition_component_engine_refactor_mode_v174l": "first_pocket_module_split_child_bore_floor_boundary_candidate_contract_no_behavior_change",
        "recognition_pocket_child_bore_filter_split_v174l": True,
        "recognition_pocket_floor_boundary_metadata_split_v174l": True,
        "recognition_pocket_candidate_contract_helper_split_v174l": True,
        "recognition_pocket_candidate_emission_still_in_component_engine_v174l": True,
        "recognition_component_engine_next_cleanup_step_v174l": (
            "continue POCKET split by moving X1 probe/local pocket detector helpers only after regression confirms current POCKET behavior is unchanged"
        ),
        "recognition_component_engine_refactor_checkpoint_v174m": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174M,
        "recognition_pocket_x1_probe_helper_split_checkpoint_v174m": POCKET_RECOGNITION_X1_PROBE_HELPER_SPLIT_CHECKPOINT_V174M,
        "recognition_component_engine_refactor_mode_v174m": "split_pocket_x1_probe_ledger_helpers_no_candidate_behavior_change",
        "recognition_pocket_x1_probe_helpers_split_v174m": True,
        "recognition_pocket_detector_assembly_still_in_component_engine_v174m": True,
        "recognition_component_engine_next_cleanup_step_v174m": (
            "continue POCKET split by moving the local pocket detector only after pocket-family authority leak diagnostics are captured"
        ),
        "recognition_component_engine_refactor_checkpoint_v174n": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174N,
        "recognition_pocket_authority_leak_diagnostic_checkpoint_v174n": POCKET_RECOGNITION_AUTHORITY_LEAK_DIAGNOSTIC_CHECKPOINT_V174N,
        "recognition_component_engine_refactor_mode_v174n": "add_pocket_family_authority_leak_diagnostics_no_candidate_behavior_change",
        "recognition_pocket_authority_leak_diagnostics_added_v174n": True,
        "recognition_pocket_detector_assembly_still_in_component_engine_v174n": True,
        "recognition_component_engine_next_cleanup_step_v174n": (
            "continue POCKET split only after diagnostics capture any pocket rebuild authority leak; do not fix rebuild authority inside this recognition split patch"
        ),
        "recognition_component_engine_refactor_checkpoint_v174o": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174O,
        "recognition_pocket_detector_split_checkpoint_v174o": POCKET_RECOGNITION_DETECTOR_SPLIT_CHECKPOINT_V174O,
        "recognition_component_engine_refactor_mode_v174o": "split_pocket_detector_orchestration_no_candidate_behavior_change",
        "recognition_pocket_detector_orchestration_split_v174o": True,
        "recognition_pocket_candidate_emission_shape_preserved_v174o": True,
        "recognition_component_engine_next_cleanup_step_v174o": (
            "after runtime smoke tests, split shared CandidateData emission helpers into recognition_emit.py and diagnostics into recognition_diagnostics.py"
        ),
        "recognition_component_engine_refactor_checkpoint_v174p": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174P,
        "recognition_chamfer_candidate_contract_split_checkpoint_v174p": CHAMFER_RECOGNITION_CANDIDATE_CONTRACT_SPLIT_CHECKPOINT_V174P,
        "recognition_component_engine_refactor_mode_v174p": "split_chamfer_candidate_contract_helper_no_candidate_behavior_change",
        "recognition_chamfer_candidate_contract_helper_split_v174p": True,
        "recognition_chamfer_final_candidate_list_assembly_still_in_component_engine_v174p": True,
        "recognition_component_engine_next_cleanup_step_v174p": (
            "after runtime smoke tests, split shared CandidateData emission helpers into recognition_emit.py and diagnostics into recognition_diagnostics.py"
        ),
        "recognition_diagnostics_split_checkpoint_v174q": RECOGNITION_DIAGNOSTICS_SPLIT_CHECKPOINT_V174Q,
        "recognition_component_engine_refactor_mode_v174q": "split_surface_role_evidence_graph_diagnostics_no_candidate_behavior_change",
        "recognition_surface_role_graph_diagnostics_split_v174q": True,
        "recognition_diagnostics_only_no_candidate_promotion_v174q": True,
        "recognition_component_engine_next_cleanup_step_v174q": (
            "continue diagnostics split, then split shared CandidateData emission helpers into recognition_emit.py"
        ),
        "v174p_no_behavior_change_contract": True,
        "v174p_not_feature_recognition": True,
        "v174p_not_candidate_promotion": True,
        "v174p_not_delete_patch_policy": True,
        "v174p_not_mesh_mutation": True,
        "v174q_no_behavior_change_contract": True,
        "v174q_not_feature_recognition": True,
        "v174q_not_candidate_promotion": True,
        "v174q_not_delete_patch_policy": True,
        "v174q_not_mesh_mutation": True,
        **component_engine_authority_map_split_status_v174r(),
        "recognition_component_engine_next_cleanup_step_v174r": (
            "split shared CandidateData emission helpers into recognition_emit.py after runtime smoke tests confirm diagnostics split is unchanged"
        ),
        **recognition_emit_split_status_v174s(),
        "recognition_component_engine_next_cleanup_step_v174s": (
            "continue shrinking component_engine_feature_candidates by moving final diagnostic assembly or remaining shared emit helpers after smoke tests"
        ),
        "recognition_component_engine_refactor_checkpoint_v174t": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174T,
        "recognition_component_engine_orchestrator_thinning_checkpoint_v174t": RECOGNITION_COMPONENT_ENGINE_ORCHESTRATOR_THINNING_CHECKPOINT_V174T,
        **component_engine_orchestrator_inventory_v174t(),
        "recognition_component_engine_refactor_checkpoint_v174u": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174U,
        "recognition_component_engine_local_helper_inventory_checkpoint_v174u": RECOGNITION_COMPONENT_ENGINE_LOCAL_HELPER_INVENTORY_CHECKPOINT_V174U,
        **component_engine_remaining_local_helper_inventory_v174u(),
        "recognition_component_engine_refactor_checkpoint_v174v": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174V,
        "recognition_regression_smoke_checkpoint_v174v": RECOGNITION_REGRESSION_SMOKE_CHECKPOINT_V174V,
        **recognition_regression_smoke_checkpoint_v174v(),
        "recognition_component_engine_refactor_checkpoint_v174x": RECOGNITION_COMPONENT_ENGINE_SHARED_HELPER_SPLIT_CHECKPOINT_V174X,
        "recognition_component_engine_shared_helper_split_v174x": True,
        "recognition_component_engine_shared_helper_module_v174x": "recognition_common.py",
        "recognition_component_engine_shared_helper_split_mode_v174x": "move_measurement_topology_evidence_helpers_no_candidate_behavior_change",
        "recognition_component_engine_line_count_after_v174x": 3370,
        "recognition_component_engine_line_count_reduced_from_v174w_v174x": 960,
        "recognition_component_engine_candidate_list_assembly_still_local_v174x": True,
        "v174x_no_behavior_change_contract": True,
        "v174x_not_feature_recognition": True,
        "v174x_not_surface_ownership": True,
        "v174x_not_candidate_promotion": True,
        "v174x_not_delete_patch_policy": True,
        "v174x_not_mesh_mutation": True,
        "recognition_component_engine_refactor_checkpoint_v174y": RECOGNITION_COMPONENT_ENGINE_REFACTOR_CHECKPOINT_V174Y,
        "recognition_chamfer_candidate_assembly_split_checkpoint_v174y": CHAMFER_RECOGNITION_CANDIDATE_ASSEMBLY_SPLIT_CHECKPOINT_V174Y,
        "recognition_component_engine_chamfer_candidate_assembly_split_v174y": True,
        "recognition_component_engine_chamfer_candidate_assembly_module_v174y": "recognition_chamfer.py",
        "recognition_component_engine_chamfer_final_merge_still_local_v174y": True,
        "recognition_component_engine_line_count_after_v174y": 2997,
        "recognition_component_engine_line_count_reduced_from_v174x_v174y": 373,
        "v174y_no_behavior_change_contract": True,
        "v174y_not_feature_recognition_change": True,
        "v174y_not_surface_ownership_change": True,
        "v174y_not_candidate_promotion_change": True,
        "v174y_not_delete_patch_policy": True,
        "v174y_not_mesh_mutation": True,
        "v174n_no_behavior_change_contract": True,
        "v174n_not_feature_recognition": True,
        "v174n_not_candidate_promotion": True,
        "v174n_not_delete_patch_policy": True,
        "v174n_not_mesh_mutation": True,
        "v174m_no_behavior_change_contract": True,
        "v174m_not_feature_recognition": True,
        "v174m_not_candidate_promotion": True,
        "v174m_not_delete_patch_policy": True,
        "v174m_not_mesh_mutation": True,
        "v174l_no_behavior_change_contract": True,
        "v174l_not_feature_recognition": True,
        "v174l_not_candidate_promotion": True,
        "v174l_not_delete_patch_policy": True,
        "v174l_not_mesh_mutation": True,
        "v174k_no_behavior_change_contract": True,
        "v174k_not_feature_recognition": True,
        "v174k_not_candidate_promotion": True,
        "v174k_not_delete_patch_policy": True,
        "v174k_not_mesh_mutation": True,
        "v174j_no_behavior_change_contract": True,
        "v174j_not_feature_recognition": True,
        "v174j_not_candidate_promotion": True,
        "v174j_not_delete_patch_policy": True,
        "v174j_not_mesh_mutation": True,
        "v174i_no_behavior_change_contract": True,
        "v174i_not_feature_recognition": True,
        "v174i_not_candidate_promotion": True,
        "v174i_not_delete_patch_policy": True,
        "v174i_not_mesh_mutation": True,
        "v174h_no_behavior_change_contract": True,
        "v174h_not_feature_recognition": True,
        "v174h_not_candidate_promotion": True,
        "v174h_not_delete_patch_policy": True,
        "v174h_not_mesh_mutation": True,
        "v174g_no_behavior_change_contract": True,
        "v174g_not_candidate_promotion": True,
        "v174g_not_delete_patch_policy": True,
        "v174g_not_mesh_mutation": True,
        "v174f_no_behavior_change_contract": True,
        "v174f_not_candidate_promotion": True,
        "v174f_not_delete_patch_policy": True,
        "v174f_not_mesh_mutation": True,
        "v174e_no_behavior_change_contract": True,
        "v174e_not_candidate_promotion": True,
        "v174e_not_delete_patch_policy": True,
        "v174e_not_mesh_mutation": True,
        "v174d_no_behavior_change_contract": True,
        "v174d_not_feature_recognition": True,
        "v174d_not_candidate_promotion": True,
        "v174d_not_delete_patch_policy": True,
        "v174d_not_mesh_mutation": True,
        "v174b_no_behavior_change_contract": True,
        "v174b_static_diagnostics_extracted": True,
        "v174b_chamfer_export_selection_helper_extracted": True,
        "v174b_not_feature_recognition": True,
        "v174b_not_candidate_promotion": True,
        "v174b_not_delete_patch_policy": True,
        "v174b_not_mesh_mutation": True,
    }


def component_engine_orchestrator_inventory_v174t() -> dict[str, object]:
    """Return the v174t module-role inventory for recognition orchestration.

    This is diagnostic metadata only.  It does not inspect RegionData, does not
    choose a feature family, does not assign owned faces, does not build
    CandidateData, and does not participate in DeletePatchProposal or rebuild
    policy.  The goal is to make the remaining responsibility of
    ``recognition_component_engine.py`` explicit before the next thinning pass.
    """

    return {
        "recognition_component_engine_orchestrator_checkpoint_v174t": RECOGNITION_COMPONENT_ENGINE_ORCHESTRATOR_THINNING_CHECKPOINT_V174T,
        "recognition_component_engine_role_v174t": "orchestration_fanout_and_final_candidate_list_assembly",
        "recognition_component_engine_family_owner_v174t": False,
        "recognition_component_engine_delete_patch_owner_v174t": False,
        "recognition_component_engine_mesh_mutation_owner_v174t": False,
        "recognition_component_engine_public_entrypoints_v174t": (
            "component_engine_feature_candidates",
            "recognition_result_dict_from_component_features",
        ),
        "recognition_family_modules_active_v174t": {
            "bore": "recognition_bore.py owns BORE-local ownership/locality/CandidateData contract helpers",
            "chamfer": "recognition_chamfer.py owns CHAMFER rail-to-rail ownership/proof/CandidateData contract helpers",
            "pocket": "recognition_pocket.py owns POCKET detector/probe/relationship/CandidateData contract helpers",
            "diagnostics": "recognition_diagnostics.py owns read-only role graph and authority-map diagnostics",
            "emit": "recognition_emit.py owns public result-dictionary packing helpers",
        },
        "recognition_component_engine_compatibility_wrappers_preserved_v174t": True,
        "recognition_component_engine_next_cleanup_step_v174t": (
            "extract remaining top-level diagnostic assembly or candidate-list grouping once smoke tests confirm v174s/v174t output is unchanged"
        ),
        "v174t_no_behavior_change_contract": True,
        "v174t_not_feature_recognition": True,
        "v174t_not_surface_ownership": True,
        "v174t_not_candidate_promotion": True,
        "v174t_not_delete_patch_policy": True,
        "v174t_not_mesh_mutation": True,
    }


def component_engine_remaining_local_helper_inventory_v174u() -> dict[str, object]:
    """Return the v174u remaining-helper inventory for the component engine.

    v174u is still an inventory patch.  It deliberately does not move call
    sites, remove wrappers, alter CandidateData rows, alter family ownership, or
    touch rebuild policy.  The purpose is to make the remaining local surface
    area explicit before the next safe extraction pass.
    """

    compatibility_wrappers = (
        "_component_engine_refactor_authority_map_v174a -> recognition_diagnostics.component_engine_refactor_authority_map_v174r",
        "_filter_bore_terminal_continuation_ownership_v174d -> recognition_bore.filter_bore_terminal_continuation_ownership_v174h",
        "_candidate_primary_surface_role_v173b -> recognition_diagnostics.candidate_primary_surface_role_v174q",
        "_candidate_owned_face_ids_for_role_graph_v173b -> recognition_diagnostics.candidate_owned_face_ids_for_role_graph_v174q",
        "_surface_role_evidence_graph_diagnostics_v173b -> recognition_diagnostics.surface_role_evidence_graph_diagnostics_v174q",
        "_face_patch_boundary_semantic_report -> recognition_chamfer.face_patch_boundary_semantic_report_v174f",
        "_chamfer_band_role_proof_v173c -> recognition_chamfer.chamfer_band_role_proof_v174f",
        "_chamfer_selected_rail_local_band_ownership_proof_v173o -> recognition_chamfer.chamfer_selected_rail_local_band_ownership_proof_v174f",
        "_chamfer_local_transition_band_owned_subset_mapping_v173p -> recognition_chamfer.chamfer_legacy_local_transition_band_owned_subset_mapping_v174g",
        "_chamfer_rail_anchored_owned_band_coordinate_mapping_v173q -> recognition_chamfer.chamfer_legacy_rail_anchored_owned_band_coordinate_mapping_v174g",
        "_chamfer_rail_to_rail_bounded_surface_ownership_v173t -> recognition_chamfer.chamfer_rail_to_rail_bounded_surface_ownership_v174e",
        "_filter_pocket_child_bore_intrusion_faces_v158 -> recognition_pocket.filter_pocket_child_bore_intrusion_faces_v174l",
        "_select_bore_wall_semantic_owned_subset_v158 -> recognition_bore.select_bore_wall_semantic_owned_subset_v174i",
        "_seed_neighborhood_face_ids -> recognition_bore.seed_neighborhood_face_ids_v174j",
        "_bounded_component_from_seed_faces -> recognition_bore.bounded_component_from_seed_faces_v174j",
        "_cylindrical_seed_island_from_seed_faces -> recognition_bore.cylindrical_seed_island_from_seed_faces_v174j",
        "_post_isolation_bore_wall_validation -> recognition_bore.post_isolation_bore_wall_validation_v174i",
        "_opening_anchor_reference -> recognition_bore.opening_anchor_reference_v174j",
        "_opening_frame_anchor_metrics -> recognition_bore.opening_frame_anchor_metrics_v174j",
        "_selected_opening_evidence_face_ids -> recognition_bore.selected_opening_evidence_face_ids_v174j",
        "_damaged_bore_candidate_contract_fields -> recognition_bore.damaged_bore_candidate_contract_fields_v174k",
        "_bore_candidate_contract_fields -> recognition_bore.bore_candidate_contract_fields_v174k",
        "_bore_review_candidate_contract_fields -> recognition_bore.bore_review_candidate_contract_fields_v174k",
        "_bore_selected_opening_review_candidate_contract_fields -> recognition_bore.bore_selected_opening_review_candidate_contract_fields_v174k",
        "_pocket_floor_boundary_loop_records -> recognition_pocket.pocket_floor_boundary_loop_records_v174l",
        "_pocket_floor_compound_boundary_metadata -> recognition_pocket.pocket_floor_compound_boundary_metadata_v174l",
        "_pocket_candidate_contract_fields -> recognition_pocket.pocket_candidate_contract_fields_v174l",
        "_x1_probe_ledger_from_region_diagnostics -> recognition_pocket.x1_probe_ledger_from_region_diagnostics_v174m",
        "_x1_probe_nested_mapping -> recognition_pocket.x1_probe_nested_mapping_v174m",
        "_x1_probe_face_ids -> recognition_pocket.x1_probe_face_ids_v174m",
        "_x1_probe_supports_local_pocket_candidate -> recognition_pocket.x1_probe_supports_local_pocket_candidate_v174m",
        "_detect_pocket_preview_candidates -> recognition_pocket.detect_pocket_preview_candidates_v174o",
        "_candidate_contract_fields -> recognition_chamfer.chamfer_candidate_contract_fields_v174p",
        "recognition_result_dict_from_component_features -> recognition_emit.recognition_result_dict_from_component_features_v174s",
    )
    local_helper_groups = {
        "selection_rail_and_component_scoring": (
            "_selected_annular_rail_role_resolver_v173d",
            "_heuristic_results_for_current_bore_state",
            "_component_stats",
            "_chamfer_score",
            "_bore_score",
        ),
        "shared_low_level_geometry_and_arrays": (
            "_face_vertices_for_ids",
            "_face_edges_for_ids",
            "_loop_records_vertices_and_edges",
            "_unit_rows",
            "_edge_median_length",
            "_percentile",
            "_local_arrays",
            "_normalize_edge_key",
        ),
        "damaged_bore_and_anchor_policy": (
            "_damaged_bore_preview_allowed",
            "_damaged_anchor_mode_from_region",
            "_interval_gap",
        ),
        "main_orchestration_surface": (
            "component_engine_feature_candidates",
            "final candidate-list assembly",
            "top-level recognition diagnostics merge",
        ),
    }
    next_extraction_candidates = (
        "move shared low-level array/geometry helpers to a small recognition_shared.py or existing geometry/topology helpers only after smoke tests",
        "move heuristic summary/reporting helpers to recognition_diagnostics.py only if no family ownership decisions depend on local state",
        "keep component_engine_feature_candidates as fanout/orchestration until BORE/CHAMFER/POCKET smoke tests are re-run",
        "do not delete compatibility wrappers until imports and downstream logs prove the delegated names are no longer consumed",
    )
    return {
        "recognition_component_engine_local_helper_inventory_checkpoint_v174u": RECOGNITION_COMPONENT_ENGINE_LOCAL_HELPER_INVENTORY_CHECKPOINT_V174U,
        "recognition_component_engine_refactor_mode_v174u": "remaining_local_helper_and_wrapper_inventory_no_behavior_change",
        "recognition_component_engine_compatibility_wrapper_count_v174u": int(len(compatibility_wrappers)),
        "recognition_component_engine_compatibility_wrappers_v174u": compatibility_wrappers,
        "recognition_component_engine_remaining_local_helper_groups_v174u": local_helper_groups,
        "recognition_component_engine_next_extraction_candidates_v174u": next_extraction_candidates,
        "recognition_component_engine_candidate_list_assembly_still_local_v174u": True,
        "recognition_component_engine_wrapper_deletion_allowed_v174u": False,
        "v174u_no_behavior_change_contract": True,
        "v174u_not_feature_recognition": True,
        "v174u_not_surface_ownership": True,
        "v174u_not_candidate_promotion": True,
        "v174u_not_delete_patch_policy": True,
        "v174u_not_mesh_mutation": True,
    }



def recognition_regression_smoke_checkpoint_v174v() -> dict[str, object]:
    """Return the v174v recognition smoke/regression checkpoint plan.

    This is a diagnostic-only checkpoint.  It does not run tests, inspect mesh
    state, promote candidates, change family ownership, alter CandidateData,
    authorize delete patches, or mutate topology.  It records the required
    runtime smoke cases that should be rerun before the roadmap moves from
    recognition refactor/split work into behavior-changing POCKET authority
    fixes or the later rebuild.py split.
    """

    required_cases = (
        {
            "case_id": "high_resolution_clean_bore",
            "expected": (
                "selected ring remains clean; BORE accepted; owned BORE_WALL faces correct; rebuild succeeds; "
                "boundary_edge_count_after=0; watertight_after=True; visual cylinder stable"
            ),
            "stage_boundary": "selection_evidence_to_bore_wall_ownership_to_rebuild_target",
        },
        {
            "case_id": "coarse_annular_rail_pink_cloud",
            "expected": (
                "Ctrl-click selection does not expand into raw contaminated pink cloud; selected annular rail evidence remains local; "
                "Recognition receives sane RegionData"
            ),
            "stage_boundary": "live_selection_to_neutral_regiondata",
        },
        {
            "case_id": "chamfer_near_bore",
            "expected": (
                "CHAMFER candidate uses rail-to-rail owned CHAMFER_BAND faces, not raw connected angled topology; "
                "rebuild_face_ids remain owned band faces"
            ),
            "stage_boundary": "chamfer_evidence_to_surface_role_ownership",
        },
        {
            "case_id": "bore_under_or_near_chamfer",
            "expected": (
                "BORE ownership remains independent from CHAMFER; terminal continuation is not promoted when its audit fails; "
                "BORE rebuild uses owned-wall cylinder frame"
            ),
            "stage_boundary": "bore_identity_not_cross_family_subtraction",
        },
        {
            "case_id": "damaged_bore_internal_defect_boundary",
            "expected": (
                "internal damaged-wall boundaries are repair defects, not endpoint loops; zero boundary count after successful repair remains valid"
            ),
            "stage_boundary": "candidate_ownership_to_rebuild_validation",
        },
        {
            "case_id": "pocket_with_child_bore",
            "expected": (
                "POCKET owns floor/side-wall roles independently; child BORE relationship remains metadata/protection; "
                "known pocket authority-leak diagnostics are recorded before behavior fixes"
            ),
            "stage_boundary": "pocket_ownership_and_relationship_metadata",
        },
        {
            "case_id": "candidate_delete_patch_identity",
            "expected": (
                "CandidateData.rebuild_face_ids equals DeletePatchProposal.face_ids for generic rebuild; RegionData does not become rebuild input"
            ),
            "stage_boundary": "candidate_data_to_rebuild_target_policy",
        },
        {
            "case_id": "rebuild_shape_authority_diagnostics",
            "expected": (
                "BORE axis/radius/center/basis come from one owned-wall frame; boundary loops remain seam constraints; "
                "POCKET authority-family mismatch stays diagnostic until dedicated fix"
            ),
            "stage_boundary": "rebuild_shape_authority_reporting",
        },
    )
    deferred_issues = (
        {
            "issue_id": "pocket_radius_authority_family_leak",
            "observed_log_token": "family=pocket radius_authority_v163=owned_bore_wall_cylinder_shape_authority_v173x",
            "status": "known_deferred_behavior_issue_diagnostic_only_in_v174v",
            "planned_after": "recognition_split_smoke_checkpoint",
        },
    )
    return {
        "recognition_regression_smoke_checkpoint_v174v": RECOGNITION_REGRESSION_SMOKE_CHECKPOINT_V174V,
        "recognition_regression_smoke_checkpoint_mode_v174v": "diagnostic_checklist_only_no_runtime_execution_no_behavior_change",
        "recognition_regression_required_case_count_v174v": int(len(required_cases)),
        "recognition_regression_required_cases_v174v": required_cases,
        "recognition_regression_deferred_issue_count_v174v": int(len(deferred_issues)),
        "recognition_regression_deferred_issues_v174v": deferred_issues,
        "recognition_regression_next_allowed_phase_v174v": (
            "run BORE/CHAMFER/POCKET smoke tests against v174v or later before behavior-changing POCKET authority fix and before rebuild.py split"
        ),
        "recognition_regression_wrapper_deletion_allowed_v174v": False,
        "recognition_regression_behavior_change_allowed_v174v": False,
        "v174v_no_behavior_change_contract": True,
        "v174v_not_feature_recognition": True,
        "v174v_not_surface_ownership": True,
        "v174v_not_candidate_promotion": True,
        "v174v_not_delete_patch_policy": True,
        "v174v_not_mesh_mutation": True,
    }

def component_engine_wrapper_callsite_migration_v174w() -> dict[str, object]:
    """Return the v174w wrapper call-site migration status.

    v174w is the first line-count-reduction pass after the family/diagnostic/emit
    modules exist. It migrates internal call sites away from component-engine
    compatibility wrappers and removes those wrappers. It does not remove the
    public entry points, does not alter CandidateData, and does not touch rebuild
    target policy or mesh mutation.
    """

    removed_wrappers = (
        "_component_engine_refactor_authority_map_v174a",
        "_filter_bore_terminal_continuation_ownership_v174d",
        "_candidate_primary_surface_role_v173b",
        "_candidate_owned_face_ids_for_role_graph_v173b",
        "_surface_role_evidence_graph_diagnostics_v173b",
        "_chamfer_band_role_proof_v173c",
        "_chamfer_selected_rail_local_band_ownership_proof_v173o",
        "_chamfer_local_transition_band_owned_subset_mapping_v173p",
        "_chamfer_rail_anchored_owned_band_coordinate_mapping_v173q",
        "_chamfer_rail_to_rail_bounded_surface_ownership_v173t",
        "_filter_pocket_child_bore_intrusion_faces_v158",
        "_select_bore_wall_semantic_owned_subset_v158",
        "_damaged_bore_reason_flags",
        "_seed_neighborhood_face_ids",
        "_bounded_component_from_seed_faces",
        "_cylindrical_seed_island_from_seed_faces",
        "_post_isolation_bore_wall_validation",
        "_opening_anchor_reference",
        "_opening_frame_anchor_metrics",
        "_selected_opening_evidence_face_ids",
        "_damaged_bore_candidate_contract_fields",
        "_bore_candidate_contract_fields",
        "_bore_review_candidate_contract_fields",
        "_bore_selected_opening_review_candidate_contract_fields",
        "_candidate_contract_fields",
        "_pocket_floor_boundary_loop_records",
        "_pocket_floor_compound_boundary_metadata",
        "_pocket_candidate_contract_fields",
        "_x1_probe_ledger_from_region_diagnostics",
        "_x1_probe_nested_mapping",
        "_x1_probe_face_ids",
        "_x1_probe_supports_local_pocket_candidate",
        "_detect_pocket_preview_candidates",
    )
    return {
        "recognition_component_engine_refactor_checkpoint_v174w": RECOGNITION_COMPONENT_ENGINE_WRAPPER_CALLSITE_MIGRATION_CHECKPOINT_V174W,
        "recognition_component_engine_refactor_mode_v174w": "migrate_internal_call_sites_to_family_modules_and_remove_compatibility_wrappers_no_candidate_behavior_change",
        "recognition_component_engine_removed_wrapper_count_v174w": int(len(removed_wrappers)),
        "recognition_component_engine_removed_wrappers_v174w": removed_wrappers,
        "recognition_component_engine_public_entrypoints_preserved_v174w": (
            "component_engine_feature_candidates",
            "recognition_result_dict_from_component_features",
        ),
        "recognition_component_engine_line_count_reduction_started_v174w": True,
        "recognition_component_engine_candidate_list_assembly_still_local_v174w": True,
        "recognition_component_engine_remaining_large_body_v174w": "component_engine_feature_candidates",
        "recognition_component_engine_next_cleanup_step_v174w": (
            "extract family candidate-list assembly blocks after smoke tests confirm direct module call sites preserve behavior"
        ),
        "v174w_no_behavior_change_contract": True,
        "v174w_not_feature_recognition": True,
        "v174w_not_surface_ownership": True,
        "v174w_not_candidate_promotion": True,
        "v174w_not_delete_patch_policy": True,
        "v174w_not_mesh_mutation": True,
    }


def candidate_primary_surface_role_v174q(candidate: Mapping[str, object]) -> str:
    """Return the candidate's owned role name for diagnostics only.

    This helper deliberately does not decide whether a candidate can rebuild.
    CandidateData actionability must come from the family recognizer's own
    surface-role ownership transform.  The role graph records how emitted rows
    relate to one another so logs can show relationship context without
    demoting already-owned candidates after the fact.
    """

    family = str(candidate.get("feature_family", "") or "").strip().lower()
    entity = str(candidate.get("entity_type", candidate.get("feature_kind", "")) or "").strip().lower()
    if family == FeatureFamily.BORE.value or entity in {"bore", "borehole"}:
        return "BORE_WALL"
    if family in {FeatureFamily.POCKET.value, FeatureFamily.CIRCULAR_POCKET.value} or entity in {"pocket", "circular_pocket"}:
        return "POCKET_FLOOR_SIDEWALL"
    if family == FeatureFamily.CHAMFER_FORM.value or entity == "chamfer":
        return "CHAMFER_BAND"
    return "UNKNOWN_SURFACE_ROLE"


def candidate_owned_face_ids_for_role_graph_v174q(candidate: Mapping[str, object]) -> tuple[int, ...]:
    """Return owned faces for diagnostics without creating authority.

    Rebuild faces are the strongest CandidateData ownership export, but review
    candidates may carry only semantic/owned/display IDs.  This function is only
    for reporting possible role contact.  It must never be used to enable or
    disable CandidateData.
    """

    for key in ("rebuild_face_ids", "candidate_rebuild_face_ids", "semantic_face_ids", "owned_face_ids", "face_ids"):
        ids = tuple_ints(candidate.get(key, ()))
        if ids:
            return ids
    return ()


def surface_role_evidence_graph_diagnostics_v174q(
    candidates: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Return a read-only role/relationship graph for emitted candidate rows.

    v173 originally tried to arbitrate after candidates were emitted.  That was
    semantically wrong: post-CandidateData demotion is a permission filter, not a
    meaning transform.  v174q keeps this layer as diagnostics only.  Family
    recognizers must create owned roles before CandidateData; this graph only
    helps developers see where BORE/CHAMFER/POCKET evidence touches.
    """

    rows = [dict(row) for row in tuple(candidates or ())]
    role_counts: dict[str, int] = {}
    actionable_role_counts: dict[str, int] = {}
    role_face_counts: dict[str, int] = {}
    relationship_counts: dict[str, int] = {}
    relationship_rows: list[dict[str, object]] = []
    face_claims: dict[int, list[tuple[int, str]]] = {}

    for idx, row in enumerate(rows):
        role = candidate_primary_surface_role_v174q(row)
        role_counts[role] = int(role_counts.get(role, 0) + 1)
        if bool(row.get("candidate_action_enabled", False)):
            actionable_role_counts[role] = int(actionable_role_counts.get(role, 0) + 1)
        ids = candidate_owned_face_ids_for_role_graph_v174q(row)
        role_face_counts[role] = int(role_face_counts.get(role, 0) + len(ids))
        for fid in ids:
            face_claims.setdefault(int(fid), []).append((int(idx), role))
        for rel in tuple(row.get("feature_relationships", ()) or ()):  # type: ignore[arg-type]
            if not isinstance(rel, Mapping):
                continue
            kind = str(rel.get("relationship_kind", rel.get("kind", "")) or "unknown_relationship")
            relationship_counts[kind] = int(relationship_counts.get(kind, 0) + 1)
            relationship_rows.append({
                "candidate_index": int(idx),
                "candidate_id": str(row.get("candidate_id", idx)),
                "candidate_role": role,
                "relationship_kind": kind,
                "relationship_role": str(rel.get("role", "")),
            })

    overlap_reports: list[dict[str, object]] = []
    for fid, claims in face_claims.items():
        roles = tuple(sorted({role for _idx, role in claims}))
        if len(roles) <= 1:
            continue
        overlap_reports.append({
            "face_id": int(fid),
            "roles": roles,
            "candidate_indices": tuple(int(idx) for idx, _role in claims),
            "semantic_status": "diagnostic_overlap_only_not_post_candidate_demotion",
        })
        if len(overlap_reports) >= 24:
            break

    return {
        "surface_role_evidence_graph_contract_v173b": SURFACE_ROLE_EVIDENCE_GRAPH_CONTRACT_V173B,
        "surface_role_evidence_graph_candidate_count_v173b": int(len(rows)),
        "surface_role_evidence_graph_role_counts_v173b": dict(role_counts),
        "surface_role_evidence_graph_actionable_role_counts_v173b": dict(actionable_role_counts),
        "surface_role_evidence_graph_role_face_counts_v173b": dict(role_face_counts),
        "surface_role_evidence_graph_relationship_counts_v173b": dict(relationship_counts),
        "surface_role_evidence_graph_relationships_v173b": tuple(relationship_rows[:24]),
        "surface_role_evidence_graph_overlap_report_count_v173b": int(len(overlap_reports)),
        "surface_role_evidence_graph_overlap_reports_v173b": tuple(overlap_reports),
        "surface_role_evidence_graph_semantic_rule_v173b": (
            "Role evidence may touch or overlap as diagnostics; CandidateData actionability is owned by the family-specific "
            "surface-role transform before CandidateData, not by post-candidate demotion."
        ),
        "recognition_diagnostics_split_checkpoint_v174q": RECOGNITION_DIAGNOSTICS_SPLIT_CHECKPOINT_V174Q,
        "recognition_diagnostics_module_surface_role_graph_v174q": True,
        "recognition_diagnostics_only_no_candidate_promotion_v174q": True,
    }


def _safe_float_v175a(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    return float(out) if out == out and abs(out) != float("inf") else float(default)


def final_component_engine_diagnostics_v175a(
    *,
    component_locals: Mapping[str, object],
    bore_assembly_v174z: Mapping[str, object],
    base_static_diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Build final component-engine diagnostics without changing recognition behavior.

    v175a moves only top-level diagnostic assembly out of
    ``component_engine_feature_candidates``. It does not classify features,
    assign ownership, create CandidateData, authorize DeletePatchProposal, or
    mutate topology. The caller still owns multi-family fanout and final
    candidate-list merge.
    """

    _safe_float = _safe_float_v175a
    features = list(component_locals.get("features", ()) or ())
    promoted = tuple(component_locals.get("promoted", ()) or ())
    best_bore_wall_diag = dict(component_locals.get("best_bore_wall_diag", {}) or {})
    bore_rejected = list(component_locals.get("bore_rejected", ()) or ())
    bore_wall_candidate_components = tuple(component_locals.get("bore_wall_candidate_components", ()) or ())
    bore_tube_face_ids = tuple_ints(component_locals.get("bore_tube_face_ids", ()))
    bore_wall_search_face_ids = tuple_ints(component_locals.get("bore_wall_search_face_ids", ()))
    bore_mouth_transition_face_ids = tuple_ints(component_locals.get("bore_mouth_transition_face_ids", ()))
    bore_mouth_transition_diag = dict(component_locals.get("bore_mouth_transition_diag", {}) or {})
    bore_owned_face_count = int(component_locals.get("bore_owned_face_count", 0) or 0)
    selected_opening_support_weak = bool(component_locals.get("selected_opening_support_weak", False))
    bore_features = list(component_locals.get("bore_features", ()) or ())
    chamfer_rows = tuple(component_locals.get("chamfer_rows", ()) or ())
    pocket_diag = dict(component_locals.get("pocket_diag", {}) or {}) if isinstance(component_locals.get("pocket_diag", {}), Mapping) else {}
    bore_resolution_active = bool(component_locals.get("bore_resolution_active", False))
    region_face_count = int(component_locals.get("region_face_count", 0) or 0)
    selected_edge_count = int(component_locals.get("selected_edge_count", 0) or 0)
    broad_or_ambiguous_selection = bool(component_locals.get("broad_or_ambiguous_selection", False))
    selected_opening_frame_resolver = dict(component_locals.get("selected_opening_frame_resolver", {}) or {})
    two_opening_bore_frame = dict(component_locals.get("two_opening_bore_frame", {}) or {})
    two_opening_valid = bool(component_locals.get("two_opening_valid", False))
    bore_radius = _safe_float(component_locals.get("bore_radius", 0.0), 0.0)
    bore_depth = _safe_float(component_locals.get("bore_depth", 0.0), 0.0)
    two_opening_axis_direction_preserved = bool(component_locals.get("two_opening_axis_direction_preserved", False))
    two_opening_axis_dot_opening_to_opposite = _safe_float(component_locals.get("two_opening_axis_dot_opening_to_opposite", 0.0), 0.0)
    # v1.5.5: produce compact top-level wall ownership reports.
    # Candidate rows may contain full face_ids for preview, but routed diagnostics
    # need small summaries that explain why ownership failed without flooding the UI.
    def _compact_wall_report(row: Mapping[str, object]) -> dict[str, object]:
        reasons = row.get("bore_wall_component_rejection_reasons", ())
        if not isinstance(reasons, (tuple, list)):
            reasons = (str(reasons),) if reasons else ()
        return {
            "wall_report_kind": str(row.get("wall_report_kind", "connected_component") or "connected_component"),
            "face_count": int(row.get("face_count", 0) or 0),
            "accepted_as_bore_wall_ownership": bool(row.get("accepted_as_bore_wall_ownership", False)),
            "rejection_reasons": tuple(str(v) for v in tuple(reasons)),
            "reject_reason": str(row.get("bore_wall_ownership_reject_reason", "") or ""),
            "score": _safe_float(row.get("score", 0.0), 0.0),
            "axial_span": _safe_float(row.get("axial_span", 0.0), 0.0),
            "centroid_axial_span": _safe_float(row.get("centroid_axial_span", 0.0), 0.0),
            "face_span_axial_ownership_used": bool(row.get("face_span_axial_ownership_used", False)),
            "wall_span_fraction_of_two_opening_depth": _safe_float(row.get("wall_span_fraction_of_two_opening_depth", 0.0), 0.0),
            "strict_wall_band_ratio": _safe_float(row.get("strict_wall_band_ratio", 0.0), 0.0),
            "radial_rel_mad": _safe_float(row.get("radial_rel_mad", 999999.0), 999999.0),
            "radial_abs_mad": _safe_float(row.get("radial_abs_mad", 999999.0), 999999.0),
            "normal_axis_abs_median": _safe_float(row.get("normal_axis_abs_median", 1.0), 1.0),
            "radial_normal_alignment_median": _safe_float(row.get("radial_normal_alignment_median", 0.0), 0.0),
            "touches_selected_opening": bool(row.get("touches_selected_opening", False)),
            "touches_opposite_opening": bool(row.get("touches_opposite_opening", False)),
            "touches_both_openings": bool(row.get("touches_both_openings", False)),
            "min_distance_to_selected_opening": _safe_float(row.get("min_distance_to_selected_opening", 999999.0), 999999.0),
            "min_distance_to_opposite_opening": _safe_float(row.get("min_distance_to_opposite_opening", 999999.0), 999999.0),
            "opening_touch_tolerance": _safe_float(row.get("opening_touch_tolerance", 0.0), 0.0),
            "raw_region_edge_scale": _safe_float(row.get("raw_region_edge_scale", 0.0), 0.0),
            "selected_opening_edge_scale": _safe_float(row.get("selected_opening_edge_scale", 0.0), 0.0),
            "bore_role_edge_scale": _safe_float(row.get("bore_role_edge_scale", 0.0), 0.0),
            "bore_role_scale_clamped": bool(row.get("bore_role_scale_clamped", False)),
            "role_scale_authority": str(row.get("role_scale_authority", "")),
            "selected_opening_edge_count_is_small_v156": bool(row.get("selected_opening_edge_count_is_small_v156", False)),
            "small_clean_opening_is_trustworthy_v156": bool(row.get("small_clean_opening_is_trustworthy_v156", False)),
            "small_clean_component_geometry_strong_v156": bool(row.get("small_clean_component_geometry_strong_v156", False)),
            "selected_opening_primary_component_weak_overridden_v156": bool(row.get("selected_opening_primary_component_weak_overridden_v156", False)),
            "small_clean_bore_ownership_promoted_v156": bool(row.get("small_clean_bore_ownership_promoted_v156", False)),
        }

    # v156: include the accepted wall component before rejected diagnostics so
    # the UI can verify CandidateData ownership, not only see failure rows.
    wall_report_rows_for_top = tuple((best_bore_wall_diag,) if best_bore_wall_diag else ()) + tuple(bore_rejected or ())
    compact_bore_wall_reports = tuple(_compact_wall_report(row) for row in tuple(wall_report_rows_for_top or ())[:12])
    flattened_bore_wall_reasons = tuple(sorted({
        str(reason)
        for row in compact_bore_wall_reports
        for reason in tuple(row.get("rejection_reasons", ()) or ())
        if str(reason)
    }))
    best_wall_candidate_report = compact_bore_wall_reports[0] if compact_bore_wall_reports else {}

    # v155: top-level diagnostics must surface the accepted candidate's
    # terminal continuation audit.  v154 stored it inside CandidateData, but the
    # Routed BoreTool diagnostics panel reads component_engine_diagnostics.
    component_diagnostic_locals_v174z = dict(component_locals or {})

    def _diag_get_v174z(name: str, default: object = None) -> object:
        if isinstance(bore_assembly_v174z, Mapping) and name in bore_assembly_v174z:
            return bore_assembly_v174z.get(name, default)
        return component_diagnostic_locals_v174z.get(name, default)

    candidate_diag_for_top = dict(_diag_get_v174z("candidate_diag", {}) or {}) if isinstance(_diag_get_v174z("candidate_diag", {}), Mapping) else {}
    terminal_diag_for_top = dict(_diag_get_v174z("terminal_wall_coaxial_audit", {}) or {}) if isinstance(_diag_get_v174z("terminal_wall_coaxial_audit", {}), Mapping) else {}

    diag = {
        **dict(base_static_diagnostics or {}),
        "bore_terminal_continuation_ownership_audit_v173w": True,
        "terminal_continuation_removed_from_bore_ownership_v173w": bool(_diag_get_v174z("terminal_continuation_removed_by_v173w", False)),
        "terminal_continuation_removed_face_count_v173w": int(_diag_get_v174z("terminal_continuation_removed_count_v173w", 0) or 0),
        "terminal_continuation_removed_reason_v173w": str(_diag_get_v174z("terminal_continuation_removed_reason_v173w", "")),
        **dict(_diag_get_v174z("selected_annular_rail_public_diag_v173d", {}) or {}),
        "bore_review_suppressed_by_selected_rail_role_v173d": int(_diag_get_v174z("bore_review_suppressed_by_selected_rail_role_v173d", 0)),
        "bore_review_suppression_removed_v173i": bool(_diag_get_v174z("bore_review_suppression_removed_v173i", False)),
        "selected_rail_chamfer_accepted_v173d": bool(_diag_get_v174z("selected_rail_chamfer_accepted_v173d", False)),
        "multi_family_aoi_fanout_v173i": bool(_diag_get_v174z("multi_family_aoi_fanout_v173i", False)),
        "recognition_family_passes_v173i": ("bore", "pocket", "chamfer"),
        "candidate_suppression_scope_v173i": str(_diag_get_v174z("candidate_suppression_scope_v173i", "same_surface_duplicate_only_no_family_winner_take_all")),
        "family_winner_take_all_disabled_v173i": True,
        "cross_family_candidate_creation_allowed_v173c": False,
        "surface_role_evidence_graph_contract_v173b": SURFACE_ROLE_EVIDENCE_GRAPH_CONTRACT_V173B,
        **dict(_diag_get_v174z("surface_role_graph_diag_v173b", {}) or {}),
        "heuristic_contract_version": HEURISTIC_CONTRACT_VERSION,
        "heuristic_registry": heuristic_registry_dict(),
        "bore_heuristic_recipe": BORE_HEURISTIC_RECIPE,
        "chamfer_heuristic_recipe": CHAMFER_HEURISTIC_RECIPE,
        "pocket_heuristic_recipe": POCKET_HEURISTIC_RECIPE,
        "pocket_detection": dict(_diag_get_v174z("pocket_diag", {}) or {}),
        "pocket_candidate_count": int(len(tuple(_diag_get_v174z("pocket_features", ()) or ()))),
        "pocket_floor_face_count": int((_diag_get_v174z("pocket_diag", {}) or {}).get("pocket_floor_face_count", 0)) if isinstance(_diag_get_v174z("pocket_diag", {}), Mapping) else 0,
        "pocket_side_wall_face_count": int((_diag_get_v174z("pocket_diag", {}) or {}).get("pocket_side_wall_face_count", 0)) if isinstance(_diag_get_v174z("pocket_diag", {}), Mapping) else 0,
        "pocket_transition_face_count": int((_diag_get_v174z("pocket_diag", {}) or {}).get("pocket_transition_face_count", 0)) if isinstance(_diag_get_v174z("pocket_diag", {}), Mapping) else 0,
        "pocket_depth": _safe_float((_diag_get_v174z("pocket_diag", {}) or {}).get("pocket_depth", 0.0), 0.0) if isinstance(_diag_get_v174z("pocket_diag", {}), Mapping) else 0.0,
        "heuristic_results": tuple(_diag_get_v174z("bore_heuristic_results", ())),
        "heuristic_result_summaries": compact_heuristic_summary(tuple(_diag_get_v174z("bore_heuristic_results", ()))),
        "heuristic_authority_policy": "heuristics_propose_measurement_quantifies_recognition_interprets_ownership_assigns",
        "secondary_opening_rim_search_diagnostics_checkpoint_v175f": RECOGNITION_SECONDARY_OPENING_RIM_HEURISTIC_DIAGNOSTICS_CHECKPOINT_V175F,
        "secondary_opening_rim_search_checkpoint_v175f": SECONDARY_OPENING_RIM_SEARCH_HEURISTIC_CHECKPOINT_V175F,
        **dict(_diag_get_v174z("secondary_opening_rim_heuristic_diag_v175f", {}) or {}),
        "heuristic_scope_fix_v1_7_3": "BORE CandidateData construction no longer reads CHAMFER-local heuristic variables before the CHAMFER recipe runs",
        "bore_role_scale_clamp_v1_7_4": "BORE wall ownership thresholds use selected-opening ring scale, not raw RegionData face scale",
        "recognition_cleanup": "patchy_bore_promotion_quarantined",
        "semantic_order": "selected_edge_ids -> SelectedAnnularRailEvidence -> rail_role_hypotheses -> family_local_ownership -> CandidateData",
        "region_face_count": int(region_face_count),
        "selected_edge_count": int(selected_edge_count),
        "broad_or_ambiguous_selection": bool(broad_or_ambiguous_selection),
        "selected_opening_frame_resolved": bool(selected_opening_frame_resolver.get("resolved", False)),
        "selected_opening_frame_source": str(selected_opening_frame_resolver.get("resolver_source", "")),
        "selected_opening_primary_edge_count": int(selected_opening_frame_resolver.get("primary_edge_count", 0) or 0),
        "selected_opening_expanded_edge_count": int(selected_opening_frame_resolver.get("expanded_edge_count", 0) or 0),
        "raw_region_edge_scale": float(_diag_get_v174z("raw_region_edge_scale", 0.0)),
        "selected_opening_edge_scale": float(_diag_get_v174z("selected_opening_edge_scale", 0.0)),
        "bore_role_edge_scale": float(_diag_get_v174z("edge_scale", 0.0)),
        "bore_role_scale_limit": float(_diag_get_v174z("bore_role_scale_limit", 0.0)),
        "bore_role_scale_clamped": bool(_diag_get_v174z("bore_role_scale_clamped", False)),
        "role_scale_authority": str(_diag_get_v174z("role_scale_authority", "unknown")),
        "two_opening_bore_frame_valid": bool(two_opening_valid),
        "two_opening_bore_frame_status": str(two_opening_bore_frame.get("status", "")),
        "two_opening_bore_frame_depth": float(bore_depth),
        "two_opening_bore_frame_radius": float(bore_radius),
        "two_opening_axis_direction_preserved": bool(two_opening_axis_direction_preserved),
        "two_opening_axis_dot_opening_to_opposite": float(two_opening_axis_dot_opening_to_opposite),
        "bore_axis_source": "directed_opening_to_opposite_axis_no_canonical_flip" if bool(two_opening_valid) else "region_frame_context",
        "sidewall_normal_evidence_count": int(_diag_get_v174z("sidewall_normal_evidence_count", 0)),
        "frame_sidewall_normal_evidence_count": int(_diag_get_v174z("frame_sidewall_normal_evidence_count", 0)),
        "broad_sidewall_normal_evidence_count": int(_diag_get_v174z("broad_sidewall_normal_evidence_count", 0)),
        "bore_wall_interior_margin": float(_diag_get_v174z("interior_margin", 0.0)),
        "bore_wall_broad_radial_min": float(_diag_get_v174z("broad_radial_min", 0.0)),
        "bore_wall_broad_radial_max": float(_diag_get_v174z("broad_radial_max", 0.0)),
        "learned_wall_radius": float(_diag_get_v174z("learned_wall_radius", bore_radius)),
        "learned_wall_radius_source": str(_diag_get_v174z("learned_wall_radius_source", "two_opening_frame_radius")),
        "learned_wall_radius_tolerance": float(_diag_get_v174z("learned_wall_radius_tol", 0.0)),
        "semantic_aggregate_sidewall_ownership_used": bool(_diag_get_v174z("semantic_aggregate_sidewall_ownership_used", False)),
        "aggregate_wall_face_count": int((_diag_get_v174z("aggregate_sidewall_report", {}) or {}).get("face_count", 0)) if isinstance(_diag_get_v174z("aggregate_sidewall_report", {}), Mapping) else 0,
        "aggregate_wall_axial_coverage": _safe_float((_diag_get_v174z("aggregate_sidewall_report", {}) or {}).get("wall_span_fraction_of_two_opening_depth", 0.0), 0.0) if isinstance(_diag_get_v174z("aggregate_sidewall_report", {}), Mapping) else 0.0,
        "aggregate_wall_axial_span": _safe_float((_diag_get_v174z("aggregate_sidewall_report", {}) or {}).get("axial_span", 0.0), 0.0) if isinstance(_diag_get_v174z("aggregate_sidewall_report", {}), Mapping) else 0.0,
        "aggregate_wall_centroid_axial_span": _safe_float((_diag_get_v174z("aggregate_sidewall_report", {}) or {}).get("centroid_axial_span", 0.0), 0.0) if isinstance(_diag_get_v174z("aggregate_sidewall_report", {}), Mapping) else 0.0,
        "face_span_axial_ownership_used": bool((_diag_get_v174z("aggregate_sidewall_report", {}) or {}).get("face_span_axial_ownership_used", False)) if isinstance(_diag_get_v174z("aggregate_sidewall_report", {}), Mapping) else bool(_diag_get_v174z("face_axial_span_available", False)),
        "aggregate_wall_angular_coverage": _safe_float((_diag_get_v174z("aggregate_sidewall_report", {}) or {}).get("angular_coverage", 0.0), 0.0) if isinstance(_diag_get_v174z("aggregate_sidewall_report", {}), Mapping) else 0.0,
        "aggregate_wall_rejection_reasons": tuple((_diag_get_v174z("aggregate_sidewall_report", {}) or {}).get("bore_wall_component_rejection_reasons", ()) or ()) if isinstance(_diag_get_v174z("aggregate_sidewall_report", {}), Mapping) else (),
        "aggregate_disallowed_by_clean_opening_component_policy_v151": bool(_diag_get_v174z("aggregate_disallowed_by_clean_opening_component_policy_v151", False)),
        "wall_ownership_policy_v151": (
            "component_first_for_clean_selected_opening"
            if bool(_diag_get_v174z("aggregate_disallowed_by_clean_opening_component_policy_v151", False))
            else "aggregate_allowed_or_single_component"
        ),
        "bore_wall_component_count_for_aggregate_policy_v151": int(len(_diag_get_v174z("bore_wall_candidate_components", ()) or ())),
        "measured_two_opening_frame_depth": _safe_float(_diag_get_v174z("measured_frame_depth", _diag_get_v174z("bore_depth", 0.0)), 0.0),
        "owned_bore_frame_depth": _safe_float(_diag_get_v174z("owned_frame_depth", _diag_get_v174z("bore_depth", 0.0)), 0.0),
        "owned_bore_frame_depth_delta": float(_safe_float(_diag_get_v174z("owned_frame_depth", _diag_get_v174z("bore_depth", 0.0)), 0.0) - _safe_float(_diag_get_v174z("measured_frame_depth", _diag_get_v174z("bore_depth", 0.0)), 0.0)),
        "owned_frame_reconciled_from_wall_ownership": bool(_diag_get_v174z("owned_frame_reconciled", False)),
        "owned_wall_face_span_depth": _safe_float(_diag_get_v174z("owned_axial_span", 0.0), 0.0),
        "owned_wall_axial_min": _safe_float(_diag_get_v174z("owned_axial_min", 0.0), 0.0),
        "owned_wall_axial_max": _safe_float(_diag_get_v174z("owned_axial_max", 0.0), 0.0),
        "region_true_endpoint_depth": _safe_float(_diag_get_v174z("region_true_endpoint_depth", 0.0), 0.0),
        "true_endpoint_depth_candidate": _safe_float(_diag_get_v174z("true_endpoint_depth_candidate", 0.0), 0.0),
        "true_endpoint_depth_gap": _safe_float(_diag_get_v174z("true_endpoint_gap", 0.0), 0.0),
        "true_endpoint_reconcile_epsilon": _safe_float(_diag_get_v174z("true_endpoint_reconcile_epsilon", 0.0), 0.0),
        "true_endpoint_max_extension": _safe_float(_diag_get_v174z("true_endpoint_max_extension", 0.0), 0.0),
        "true_endpoint_reconciled_from_region_endpoint": bool(_diag_get_v174z("true_endpoint_reconciled", False)),
        "true_endpoint_reconcile_gate_v152": str(_diag_get_v174z("true_endpoint_reconcile_gate_v152", "not_evaluated")),
        "clean_selected_endpoint_reconcile_allowed_v152": bool(_diag_get_v174z("clean_selected_endpoint_reconcile_allowed_v152", False)),
        "clean_selected_endpoint_reconcile_allowed_v153": bool(_diag_get_v174z("clean_selected_endpoint_reconcile_allowed_v153", False)),
        "opposite_radius_mismatch_for_depth_v153": bool(_diag_get_v174z("opposite_radius_mismatch_for_depth_v153", False)),
        "true_endpoint_max_extension_v153": _safe_float(_diag_get_v174z("true_endpoint_max_extension_v153", 0.0), 0.0),
        "opposite_radius_delta_rel_v152": _safe_float(_diag_get_v174z("opposite_radius_delta_rel", 0.0), 0.0),
        "selected_opening_anchor_face_count_v152": int(len(_diag_get_v174z("selected_opening_anchor_face_set", ()) or ())),
        "candidate_face_source_v152": str(_diag_get_v174z("candidate_face_source_v152", "diagnostic_only")),
        "endpoint_support_face_count": int(len(tuple(_diag_get_v174z("endpoint_support_face_ids", ()) or ()))),
        "true_endpoint_extension_added_face_count": int(_diag_get_v174z("endpoint_extension_added_face_count", 0) or 0),
        "v154_terminal_continuation_coaxial_audit_used": bool(candidate_diag_for_top.get("v154_terminal_continuation_coaxial_audit_used", bool(terminal_diag_for_top))),
        **terminal_diag_for_top,
        "bore_frame_depth_source": str(candidate_diag_for_top.get("bore_frame_depth_source", "measured_two_opening_frame")),
        "two_opening_bore_wall_candidate_component_count": int(len(bore_wall_candidate_components)),
        "two_opening_search_boundary_face_count": int(len(bore_tube_face_ids)),
        "bore_wall_search_face_count": int(len(bore_wall_search_face_ids)),
        "selected_opening_support_weak": bool(selected_opening_support_weak),
        "selected_opening_edge_count_is_small_v156": bool(_diag_get_v174z("selected_opening_edge_count_is_small_v156", False)),
        "small_clean_opening_is_trustworthy_v156": bool(_diag_get_v174z("small_clean_opening_is_trustworthy_v156", False)),
        "small_clean_bore_ownership_promoted_v156": bool(any(bool((row or {}).get("small_clean_bore_ownership_promoted_v156", False)) for row in tuple(bore_rejected or ()) + tuple((best_bore_wall_diag,) if best_bore_wall_diag else ()))),
        "selected_opening_primary_component_weak_overridden_v156": bool(any(bool((row or {}).get("selected_opening_primary_component_weak_overridden_v156", False)) for row in tuple(bore_rejected or ()) + tuple((best_bore_wall_diag,) if best_bore_wall_diag else ()))),
        "small_clean_bore_promotion_reason_v156": str(best_bore_wall_diag.get("small_clean_bore_promotion_reason_v156", "") if isinstance(best_bore_wall_diag, Mapping) else ""),
        "bore_review_candidate_emitted": bool(any(str(item.get("feature_family", "")) == FeatureFamily.BORE.value and str(item.get("recognition_stage", "")) == RecognitionStage.REVIEW.value for item in features)),
        "bore_wall_owned_face_count": int(bore_owned_face_count),
        "bore_wall_ownership_valid": bool(bore_owned_face_count > 0),
        "bore_wall_best_diagnostics": dict(best_bore_wall_diag),
        "bore_wall_rejected_component_count": int(len(bore_rejected)),
        "bore_wall_rejected_component_summaries": tuple(bore_rejected[:12]),
        "bore_wall_report_rows_include_accepted_v156": bool(best_bore_wall_diag),
        "bore_wall_component_reports": compact_bore_wall_reports,
        "bore_wall_component_rejection_reasons": flattened_bore_wall_reasons,
        "bore_wall_best_component_report": best_wall_candidate_report,
        "best_wall_candidate_face_count": int(best_wall_candidate_report.get("face_count", 0) or 0) if isinstance(best_wall_candidate_report, Mapping) else 0,
        "no_sidewall_no_bore_face_preview_used": bool(any(bool(item.get("diagnostics", {}).get("no_sidewall_no_bore_face_preview_used", False)) for item in bore_features)),
        "bore_mouth_transition_face_count": int(len(bore_mouth_transition_face_ids)),
        "bore_mouth_transition_preview_source": str((bore_mouth_transition_diag or {}).get("preview_source", "")),
        "bore_mouth_transition_candidate_authority_v173c": str((_diag_get_v174z("bore_mouth_transition_relationship_diag_v173c", {}) or {}).get("bore_mouth_transition_candidate_authority_v173c", "relationship_evidence_only_no_candidate_data" if bore_mouth_transition_face_ids else "none")),
        "bore_mouth_transition_relationship_diag_v173c": dict(_diag_get_v174z("bore_mouth_transition_relationship_diag_v173c", {}) or {}),
        "selected_rail_transition_band_extractor_v173h": bool((_diag_get_v174z("selected_rail_transition_band_diag_v173h", {}) or {}).get("selected_rail_transition_band_extractor_v173h", False)),
        "selected_rail_transition_band_face_count_v173h": int((_diag_get_v174z("selected_rail_transition_band_diag_v173h", {}) or {}).get("selected_rail_transition_band_face_count_v173h", 0) or 0),
        "selected_rail_transition_band_accepted_v173h": bool((_diag_get_v174z("selected_rail_transition_band_diag_v173h", {}) or {}).get("selected_rail_transition_band_accepted_v173h", False)),
        "selected_rail_transition_band_candidate_emitted_v173h": bool(_diag_get_v174z("selected_rail_transition_band_candidate_emitted_v173h", False)),
        "chamfer_candidate_authority_v173c": "independent_selected_rail_transition_band_ownership_v173h",
        "clean_candidate_separation_used": True,
        "chamfer_candidate_count": int(len(chamfer_rows) + (1 if bool(_diag_get_v174z("selected_rail_transition_band_candidate_emitted_v173h", False)) else 0)),
        "x1_probe_ledger_recognition_bridge_used": bool((pocket_diag or {}).get("probe_ledger_used_for_candidate_data", False)) if isinstance(pocket_diag, Mapping) else False,
        "x1_probe_ledger_candidate_face_count": int((pocket_diag or {}).get("pocket_floor_face_count", 0) or 0) + int((pocket_diag or {}).get("pocket_side_wall_face_count", 0) or 0) if isinstance(pocket_diag, Mapping) else 0,
        "x1_probe_ledger_candidate_depth": _safe_float((pocket_diag or {}).get("pocket_depth", 0.0), 0.0) if isinstance(pocket_diag, Mapping) else 0.0,
        "chamfer_action_demoted_by_bore_resolution": bool(any(bool(row[2].get("chamfer_action_demoted_by_bore_resolution", False)) for row in chamfer_rows)),
        "bore_resolution_active": bool(bore_resolution_active),
        "selected_opening_review_candidate_emitted": bool(any(str(item.get("candidate_id", "")).startswith("component_engine.v82.bore.selected_opening_review") for item in features)),
        "promoted_candidate_count": int(len(promoted)),
        "bore_candidate_count_v173i": int(sum(1 for item in features if str(item.get("feature_family", "")) == FeatureFamily.BORE.value)),
        "pocket_candidate_count_v173i": int(sum(1 for item in features if str(item.get("feature_family", "")) in {FeatureFamily.POCKET.value, FeatureFamily.CIRCULAR_POCKET.value})),
        "chamfer_candidate_count_v173i": int(sum(1 for item in features if str(item.get("feature_family", "")) == FeatureFamily.CHAMFER_FORM.value)),
        "candidate_family_counts_v173i": {
            "bore": int(sum(1 for item in features if str(item.get("feature_family", "")) == FeatureFamily.BORE.value)),
            "pocket": int(sum(1 for item in features if str(item.get("feature_family", "")) in {FeatureFamily.POCKET.value, FeatureFamily.CIRCULAR_POCKET.value})),
            "chamfer": int(sum(1 for item in features if str(item.get("feature_family", "")) == FeatureFamily.CHAMFER_FORM.value)),
        },
        "candidate_summaries": tuple(
            {
                "candidate_id": str(item.get("candidate_id", "")),
                "feature_family": str(item.get("feature_family", "unknown") or "unknown"),
                "recognition_stage": str(item.get("recognition_stage", "diagnostic_only") or "diagnostic_only"),
                "face_count": int(item.get("face_count", 0) or 0),
                "accepted": bool(item.get("candidate_action_enabled", False)),
                "axial_span": _safe_float(item.get("axial_span", 0.0), 0.0),
                "confidence": _safe_float(item.get("confidence", 0.0), 0.0),
            }
            for item in features
        ),
    }

    diag.update({
        "recognition_final_diagnostics_assembly_split_checkpoint_v175a": RECOGNITION_FINAL_DIAGNOSTICS_ASSEMBLY_SPLIT_CHECKPOINT_V175A,
        "recognition_final_diagnostics_assembly_module_v175a": "recognition_diagnostics.py",
        "recognition_component_engine_final_diagnostics_moved_v175a": True,
        "recognition_component_engine_candidate_behavior_changed_v175a": False,
    })
    return diag
