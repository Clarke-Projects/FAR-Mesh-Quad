"""Rebuild roadmap and inventory metadata for BoreTool rebuild.

This module is intentionally non-mutating.  It contains the v176 refactor
roadmap metadata and inspection helpers that used to live in ``rebuild.py``.
It must not authorize deletion, generate topology, validate a trial mesh, mutate
mesh/project/host state, or emit ``RebuildResult``.

``rebuild.py`` remains the public rebuild orchestrator and single mutation
authority.
"""

from __future__ import annotations

from .rebuild_bore import (
    REBUILD_BORE_FRAME_HELPER_EXTRACTION_CHECKPOINT_V176X,
    REBUILD_BORE_FRAME_HELPER_NON_MUTATION_CONTRACT_V176X,
    REBUILD_BORE_SHAPE_AUTHORITY_EXTRACTION_CHECKPOINT_V176Y,
    REBUILD_BORE_SHAPE_AUTHORITY_NON_MUTATION_CONTRACT_V176Y,
)
from .rebuild_emit import (
    REBUILD_EMIT_EXTRACTION_CHECKPOINT_V176B,
    REBUILD_EMIT_NON_MUTATION_CONTRACT_V176B,
)
from .rebuild_geometry import (
    REBUILD_GEOMETRY_EXTRACTION_CHECKPOINT_V176D,
    REBUILD_GEOMETRY_NON_MUTATION_CONTRACT_V176D,
)
from .rebuild_loops import (
    REBUILD_GENERIC_LOOP_HELPER_EXTRACTION_CHECKPOINT_V176G,
    REBUILD_LOOPS_EXTRACTION_CHECKPOINT_V176C,
    REBUILD_LOOPS_NON_MUTATION_CONTRACT_V176C,
    REBUILD_POCKET_LOOP_ROLE_EXTRACTION_CHECKPOINT_V176E,
)
from .rebuild_semantic import (
    REBUILD_SEMANTIC_HELPER_EXTRACTION_CHECKPOINT_V176Z,
    REBUILD_SEMANTIC_HELPER_NON_MUTATION_CONTRACT_V176Z,
    REBUILD_SEMANTIC_SHAPING_POLICY_EXTRACTION_CHECKPOINT_V177A,
    REBUILD_SEMANTIC_SHAPING_POLICY_NON_MUTATION_CONTRACT_V177A,
)
from .rebuild_pocket import (
    REBUILD_POCKET_CAP_GEOMETRY_HELPER_EXTRACTION_CHECKPOINT_V176M,
    REBUILD_POCKET_CAP_GEOMETRY_HELPER_NON_MUTATION_CONTRACT_V176M,
    REBUILD_POCKET_FLOOR_GRID_HELPER_EXTRACTION_CHECKPOINT_V176U,
    REBUILD_POCKET_FLOOR_GRID_HELPER_NON_MUTATION_CONTRACT_V176U,
    REBUILD_POCKET_PATCH_PLAN_FIELDS_V176I,
    REBUILD_POCKET_PATCH_PLAN_SHELL_CHECKPOINT_V176I,
    REBUILD_POCKET_PATCH_PLAN_SHELL_NON_MUTATION_CONTRACT_V176I,
    REBUILD_POCKET_RECESS_CUP_GATE_EXTRACTION_CHECKPOINT_V176J,
    REBUILD_POCKET_RECESS_CUP_GATE_NON_MUTATION_CONTRACT_V176J,
    pocket_patch_plan_shell_inventory_v176i,
)
from .rebuild_plan import (
    REBUILD_QUAD_PLAN_EXTRACTION_CHECKPOINT_V177C,
    REBUILD_QUAD_PLAN_NON_MUTATION_CONTRACT_V177C,
    REBUILD_QUAD_PLAN_DATACLASS_HOTFIX_CHECKPOINT_V177D,
    REBUILD_QUAD_PLAN_DATACLASS_HOTFIX_NON_MUTATION_CONTRACT_V177D,
)
from .rebuild_validation import (
    REBUILD_VALIDATION_EXTRACTION_CHECKPOINT_V176F,
    REBUILD_PLAN_GEOMETRY_QUALITY_EXTRACTION_CHECKPOINT_V177F,
    REBUILD_PLAN_GEOMETRY_QUALITY_NON_MUTATION_CONTRACT_V177F,
)

REBUILD_INVENTORY_EXTRACTION_CHECKPOINT_V176T = (
    "v176t_rebuild_inventory_metadata_extracted_no_behavior_change"
)

REBUILD_INVENTORY_EXTRACTION_NON_MUTATION_CONTRACT_V176T = (
    "rebuild_inventory_contains_roadmap_metadata_only_rebuild_py_remains_mutation_authority"
)

REBUILD_ROADMAP_INVENTORY_CHECKPOINT_V176A = (
    "v176a_rebuild_inventory_no_behavior_change_monolithic_authority_modular_internals"
)

REBUILD_MONOLITHIC_AUTHORITY_CONTRACT_V176A = (
    "rebuild_py_remains_single_mutation_authority_family_helpers_may_only_return_patch_plans"
)

REBUILD_SAFE_EXTRACTION_ORDER_V176A: tuple[str, ...] = (
    "rebuild_emit_diagnostics_and_result_packing",
    "rebuild_loops_boundary_order_orientation_phase_helpers",
    "rebuild_geometry_pure_vector_frame_projection_helpers",
    "rebuild_validation_topology_and_semantic_trial_checks",
    "v176f_rebuild_validation_acceptance_helpers_extracted",
    "family_patch_generators_return_generated_patch_plan_only",
    "v176i_pocket_patch_plan_shell_created_no_behavior_change",
    "v176k_pocket_cap_request_gate_extracted_no_behavior_change",
    "v176m_pocket_cap_geometry_helper_extracted_no_behavior_change",
    "v176n_transition_ring_helpers_extracted_no_behavior_change",
    "v176u_pocket_floor_grid_helpers_extracted_no_behavior_change",
    "v176r_triangle_orientation_helpers_extracted_no_behavior_change",
    "v176x_bore_frame_helpers_extracted_no_behavior_change",
    "v176y_bore_shape_authority_helpers_extracted_no_behavior_change",
    "v176z_semantic_radius_and_ring_projection_helpers_extracted_no_behavior_change",
    "v177a_semantic_generated_vertex_shaping_policy_extracted_no_behavior_change",
    "v177c_generic_quad_plan_builders_extracted_no_behavior_change",
    "v177d_quad_plan_dataclass_hotfix_no_behavior_change",
    "v177e_rebuild_inventory_refresh_after_plan_extraction_no_behavior_change",
    "v177f_plan_geometry_quality_gate_extracted_no_behavior_change",
    "v177g_pocket_small_unequal_loop_adaptive_sew_collar_fallback",
    "v177h_pocket_linear_sidewall_safe_mode_and_explicit_floor_cap_loop_guard",
    "v177i_pocket_floor_loop_binding_hotfix_no_behavior_change",
    "v177j_post_regression_inventory_no_behavior_change",
)

REBUILD_MODULE_INVENTORY_V176A: dict[str, tuple[str, ...]] = {
    "stay_in_rebuild_py": (
        "public_delete_and_rebuild_entry_point",
        "candidate_family_dispatch",
        "trial_mesh_construction_and_replacement_boundary",
        "commit_or_reject_validation_decision",
        "final_RebuildResult_authority",
    ),
    "future_rebuild_emit_py": (
        "policy_string_construction",
        "trace_summary_compaction",
        "family_result_label_formatting",
        "RebuildResult_metadata_packing_helpers",
    ),
    "future_rebuild_loops_py": (
        "boundary_loop_extraction",
        "loop_ordering_and_orientation",
        "cyclic_shift_and_seam_alignment",
        "mouth_floor_phase_matching",
        "ring_sample_ordering_without_mesh_mutation",
    ),
    "future_rebuild_geometry_py": (
        "normalize_dot_cross_helpers",
        "axis_frame_construction",
        "plane_projection",
        "radius_sampling",
        "ring_interpolation_primitives_without_mesh_mutation",
    ),
    "future_rebuild_validation_py": (
        "boundary_edge_count_after_checks",
        "watertight_trial_checks",
        "loop_count_and_boundary_role_checks",
        "semantic_target_proof_helpers",
    ),
    "future_rebuild_bore_py": (
        "candidate_two_opening_frame_metadata_resolution",
        "bore_protected_loop_resolution",
        "owned_bore_wall_shape_authority",
        "bore_semantic_radius_and_ring_projection",
    ),
    "future_rebuild_plan_py": (
        "generic_equal_loop_quad_plan_builder",
        "generic_unequal_loop_quad_plan_builder",
        "attempt_to_quad_plan_dispatch",
        "logical_quad_and_generated_triangle_plan_data",
    ),
    "future_family_generators": (
        "build_bore_replacement_patch_plan",
        "build_chamfer_replacement_patch_plan",
        "build_pocket_recess_replacement_patch_plan",
        "v176i_pocket_patch_plan_shell_fields",
    ),
}

REBUILD_AUTHORITY_BOUNDARIES_V176A: dict[str, str] = {
    "recognition": "owns_feature_identity_measurement_and_surface_role_ownership",
    "rebuild_target": "owns_CandidateData_to_DeletePatchProposal_permission",
    "rebuild": "owns_patch_generation_trial_validation_and_RebuildResult",
    "mesh_processor": "owns_active_mesh_replacement_commit_undo_redo_and_host_state",
}

REBUILD_INVENTORY_REFRESH_CHECKPOINT_V177E = (
    "v177e_rebuild_inventory_refresh_after_v177c_v177d_no_behavior_change"
)

REBUILD_POCKET_LINEAR_SIDEWALL_AND_EXPLICIT_FLOOR_CAP_CHECKPOINT_V177H = (
    "v177h_pocket_linear_sidewall_safe_mode_and_explicit_floor_cap_loop_guard"
)

REBUILD_POCKET_LINEAR_SIDEWALL_AND_EXPLICIT_FLOOR_CAP_NON_MUTATION_CONTRACT_V177H = (
    "pocket_recess_cup_uses_explicit_owned_floor_cap_loop_and_may_use_boundary_linear_sidewall_rings_for_wide_locked_radius_envelopes_rebuild_py_still_mutates_and_validates"
)

REBUILD_POCKET_FLOOR_LOOP_BINDING_HOTFIX_CHECKPOINT_V177I = (
    "v177i_pocket_floor_loop_binding_hotfix_no_behavior_change"
)

REBUILD_POCKET_FLOOR_LOOP_BINDING_HOTFIX_NON_MUTATION_CONTRACT_V177I = (
    "pocket_recess_cup_binds_explicit_floor_loop_vertices_before_floor_cap_guard_no_delete_authorization_no_geometry_policy_change_no_mutation_change"
)

REBUILD_POST_REGRESSION_INVENTORY_CHECKPOINT_V177J = (
    "v177j_rebuild_post_regression_inventory_no_behavior_change"
)

REBUILD_POST_REGRESSION_INVENTORY_NON_MUTATION_CONTRACT_V177J = (
    "post_regression_inventory_records_v177f_to_v177i_lessons_only_no_delete_authorization_no_geometry_no_trial_no_mutation"
)

REBUILD_PYTHON_ZERO_DISCIPLINE_CONTRACT_V177J = (
    "python_zero_values_are_semantic_not_generic_false_values_preserve_valid_zero_counts_but_reject_missing_required_geometry"
)


REBUILD_ZERO_SAFETY_AUDIT_CHECKPOINT_V177K = (
    "v177k_rebuild_zero_safety_audit_validation_counts_no_behavior_change_for_valid_diagnostics"
)

REBUILD_ZERO_SAFETY_AUDIT_NON_MUTATION_CONTRACT_V177K = (
    "zero_safety_audit_preserves_valid_zero_counts_and_keeps_missing_boundary_diagnostics_as_rejection_no_geometry_no_mutation"
)

REBUILD_INVENTORY_REFRESH_NON_MUTATION_CONTRACT_V177E = (
    "inventory_refresh_documents_current_rebuild_split_state_only_no_delete_authorization_no_geometry_no_trial_no_mutation"
)


REBUILD_VALIDATION_QUALITY_GATE_REFRESH_CHECKPOINT_V177F = (
    "v177f_rebuild_plan_geometry_quality_gate_extracted_no_behavior_change"
)

REBUILD_VALIDATION_QUALITY_GATE_REFRESH_NON_MUTATION_CONTRACT_V177F = (
    "inventory_records_plan_geometry_quality_gate_extraction_only_rebuild_py_remains_mutation_authority"
)

REBUILD_POCKET_SMALL_UNEQUAL_LOOP_ADAPTIVE_COLLAR_CHECKPOINT_V177G = (
    "v177g_pocket_small_unequal_loop_adaptive_sew_collar_fallback"
)

REBUILD_POCKET_SMALL_UNEQUAL_LOOP_ADAPTIVE_COLLAR_CONTRACT_V177G = (
    "pocket_recess_cup_may_use_family_local_adaptive_sew_collar_for_small_locked_loop_count_drift_only_rebuild_py_still_trials_validates_and_emits_results"
)


REBUILD_V177E_CURRENT_MODULE_STATUS: dict[str, tuple[str, ...]] = {
    "rebuild_py_still_owns": (
        "public_delete_and_rebuild_candidate_region_entry_point",
        "candidate_context_normalization_and_family_dispatch",
        "RegionData_anchor_refresh_for_rebuild",
        "DeletePatchProposal_consumption",
        "BoundaryLoopAttempt_selection_for_trial",
        "trial_mesh_delete_replace_application",
        "trial_validation_decision",
        "final_RebuildResult_creation",
    ),
    "rebuild_plan_py_now_owns": (
        "QuadPlan_dataclass_runtime_plan_shape",
        "equal_loop_quad_plan_builder",
        "unequal_loop_quad_plan_builder",
        "attempt_to_quad_plan_dispatch",
        "generated_vertices_triangles_logical_quads_plan_data_only",
    ),
    "rebuild_semantic_py_now_owns": (
        "semantic_radius_authority_selection",
        "semantic_ring_projection",
        "generated_vertex_shaping_policy",
        "constraint_quality_diagnostics",
    ),
    "rebuild_bore_py_now_owns": (
        "accepted_candidate_two_opening_frame_reading",
        "bore_protected_loop_resolution",
        "owned_bore_wall_shape_authority",
        "locked_boundary_loop_angle_samples_for_bore_phase_guard",
    ),
    "rebuild_pocket_py_now_owns": (
        "pocket_request_gate_predicates",
        "pocket_patch_plan_shell",
        "pocket_recess_cup_patch_array_assembly",
        "pocket_cap_and_floor_grid_geometry_helpers_without_trial_application",
    ),
    "rebuild_loops_py_now_owns": (
        "loop_phase_traces",
        "loop_role_resolution",
        "cyclic_alignment",
        "unequal_loop_angle_alignment",
        "transition_count_sequences",
    ),
    "rebuild_validation_py_now_owns": (
        "boundary_edge_count",
        "zero_preserving_int_coercion",
        "trial_acceptance_predicates",
        "boundary_match_diagnostics",
        "borehole_plan_geometry_quality_gate",
        "loop_radius_wobble_quality_stats",
    ),
}

REBUILD_V177E_NEXT_SAFE_REFACTOR_STEPS: tuple[str, ...] = (
    "stop_and_regression_test_after_v177d_import_hotfix",
    "do_not_move_trial_mesh_application_out_of_rebuild_py",
    "do_not_move_final_RebuildResult_creation_out_of_rebuild_py",
    "next_split_may_only_extract_remaining_pure_loop_or_geometry_helpers",
    "family_modules_may_return_patch_plan_data_only",
    "before_each_extraction_run_import_smoke_not_compile_only",
)

REBUILD_POCKET_GENERATOR_INVENTORY_CHECKPOINT_V176H = (
    "v176h_rebuild_pocket_generator_inventory_no_behavior_change_patch_plan_boundary"
)

REBUILD_POCKET_GENERATOR_NON_MUTATION_CONTRACT_V176H = (
    "pocket_generator_inventory_is_diagnostic_only_rebuild_py_still_owns_mutation_and_validation"
)

REBUILD_POCKET_RECESS_CUP_SECTIONS_V176H: dict[str, tuple[str, ...]] = {
    "candidate_and_target_preconditions": (
        "detect_accepted_pocket_or_circular_pocket_candidate",
        "require_explicit_owned_pocket_side_wall_face_ids",
        "require_explicit_owned_pocket_floor_face_ids",
        "require_rebuild_face_ids_match_owned_floor_plus_sidewall_roles",
    ),
    "loop_role_resolution": (
        "derive_owned_floor_outer_loop",
        "derive_mouth_or_top_sidewall_loop",
        "derive_floor_or_bottom_sidewall_loop",
        "protect_child_floor_opening_loops",
        "record_v176e_loop_role_resolution_diagnostics",
    ),
    "measured_recess_frame": (
        "compute_depth_axis_from_mouth_to_floor_loop_centers",
        "preserve_selected_pocket_candidate_radius_authority_v175l",
        "lock_circumferential_segments_to_protected_boundary_loop_count",
        "reuse_locked_pocket_mouth_loop_angular_samples",
    ),
    "phase_and_density_diagnostics": (
        "record_v175j_input_loop_phase_trace",
        "record_v175j_planned_loop_phase_trace",
        "surface_v175k_compact_trace_summary",
        "keep_wave_risk_diagnostic_without_acceptance_change",
    ),
    "generated_patch_components": (
        "build_sidewall_ring_patch_between_mouth_and_floor",
        "build_quad_floor_grid_or_annular_floor_grid",
        "orient_pocket_wall_triangles_by_radial_role",
        "collect_generated_vertices_faces_and_logical_quads",
    ),
    "trial_and_result_boundary": (
        "rebuild_py_applies_generated_patch_to_trial_mesh",
        "rebuild_validation_py_evaluates_trial_topology",
        "rebuild_py_accepts_or_rejects_and_returns_RebuildResult",
        "family_generator_must_return_patch_plan_only_when_extracted",
    ),
}

REBUILD_POCKET_GENERATOR_EXTRACTION_BOUNDARY_V176H: dict[str, tuple[str, ...]] = {
    "must_stay_in_rebuild_py": (
        "delete_patch_application_to_trial_mesh",
        "candidate_family_dispatch",
        "final_trial_acceptance_decision",
        "RebuildResult_authority",
    ),
    "may_move_to_future_rebuild_pocket_py": (
        "pocket_recess_cup_patch_geometry_planning",
        "pocket_sidewall_ring_generation",
        "pocket_quad_floor_generation",
        "pocket_radius_frame_authority_diagnostics",
        "pocket_family_local_patch_plan_diagnostics",
    ),
    "future_rebuild_pocket_py_must_not": (
        "authorize_deletion",
        "interpret_CandidateData_feature_identity",
        "commit_mesh_replacement",
        "override_rebuild_validation",
        "mutate_active_project_state",
    ),
}


def rebuild_pocket_generator_inventory_v176h() -> dict[str, object]:
    """Return the no-behavior-change POCKET rebuild generator inventory.

    This is a roadmap/inspection helper only.  It documents where the current
    POCKET recess-cup generator starts and ends so a later extraction can move
    family-local patch planning into ``rebuild_pocket.py`` without moving mesh
    mutation, delete authorization, or final validation out of ``rebuild.py``.
    """

    return {
        "checkpoint": REBUILD_POCKET_GENERATOR_INVENTORY_CHECKPOINT_V176H,
        "non_mutation_contract": REBUILD_POCKET_GENERATOR_NON_MUTATION_CONTRACT_V176H,
        "sections": {
            key: tuple(values)
            for key, values in REBUILD_POCKET_RECESS_CUP_SECTIONS_V176H.items()
        },
        "extraction_boundary": {
            key: tuple(values)
            for key, values in REBUILD_POCKET_GENERATOR_EXTRACTION_BOUNDARY_V176H.items()
        },
        "active_patch_plan_shell": pocket_patch_plan_shell_inventory_v176i(),
        "depends_on_extracted_helpers": (
            REBUILD_EMIT_EXTRACTION_CHECKPOINT_V176B,
            REBUILD_LOOPS_EXTRACTION_CHECKPOINT_V176C,
            REBUILD_GEOMETRY_EXTRACTION_CHECKPOINT_V176D,
            REBUILD_POCKET_LOOP_ROLE_EXTRACTION_CHECKPOINT_V176E,
            REBUILD_VALIDATION_EXTRACTION_CHECKPOINT_V176F,
            REBUILD_GENERIC_LOOP_HELPER_EXTRACTION_CHECKPOINT_V176G,
            REBUILD_POCKET_PATCH_PLAN_SHELL_CHECKPOINT_V176I,
            REBUILD_POCKET_FLOOR_GRID_HELPER_EXTRACTION_CHECKPOINT_V176U,
        ),
    }


def rebuild_refactor_inventory_v176a() -> dict[str, object]:
    """Return the no-behavior-change rebuild refactor inventory checkpoint.

    This helper is intentionally informational.  It is not called by the active
    rebuild pipeline and must not authorize deletion, generate geometry, mutate
    a mesh, or alter validation.  It exists to pin the v176 roadmap boundary:
    keep ``rebuild.py`` as the monolithic mutation authority while extracting
    only pure helpers and patch-plan generators in later patches.
    """

    return {
        "checkpoint": REBUILD_ROADMAP_INVENTORY_CHECKPOINT_V176A,
        "authority_contract": REBUILD_MONOLITHIC_AUTHORITY_CONTRACT_V176A,
        "active_emit_extraction_checkpoint": REBUILD_EMIT_EXTRACTION_CHECKPOINT_V176B,
        "active_emit_non_mutation_contract": REBUILD_EMIT_NON_MUTATION_CONTRACT_V176B,
        "active_loop_extraction_checkpoint": REBUILD_LOOPS_EXTRACTION_CHECKPOINT_V176C,
        "active_loop_non_mutation_contract": REBUILD_LOOPS_NON_MUTATION_CONTRACT_V176C,
        "active_geometry_extraction_checkpoint": REBUILD_GEOMETRY_EXTRACTION_CHECKPOINT_V176D,
        "active_geometry_non_mutation_contract": REBUILD_GEOMETRY_NON_MUTATION_CONTRACT_V176D,
        "active_pocket_loop_role_extraction_checkpoint": REBUILD_POCKET_LOOP_ROLE_EXTRACTION_CHECKPOINT_V176E,
        "active_validation_extraction_checkpoint": REBUILD_VALIDATION_EXTRACTION_CHECKPOINT_V176F,
        "active_plan_geometry_quality_checkpoint": REBUILD_PLAN_GEOMETRY_QUALITY_EXTRACTION_CHECKPOINT_V177F,
        "active_plan_geometry_quality_contract": REBUILD_PLAN_GEOMETRY_QUALITY_NON_MUTATION_CONTRACT_V177F,
        "active_generic_loop_helper_extraction_checkpoint": REBUILD_GENERIC_LOOP_HELPER_EXTRACTION_CHECKPOINT_V176G,
        "active_pocket_generator_inventory_checkpoint": REBUILD_POCKET_GENERATOR_INVENTORY_CHECKPOINT_V176H,
        "active_pocket_patch_plan_shell_checkpoint": REBUILD_POCKET_PATCH_PLAN_SHELL_CHECKPOINT_V176I,
        "active_pocket_patch_plan_shell_contract": REBUILD_POCKET_PATCH_PLAN_SHELL_NON_MUTATION_CONTRACT_V176I,
        "active_pocket_patch_plan_shell_fields": tuple(REBUILD_POCKET_PATCH_PLAN_FIELDS_V176I),
        "active_pocket_recess_cup_gate_checkpoint": REBUILD_POCKET_RECESS_CUP_GATE_EXTRACTION_CHECKPOINT_V176J,
        "active_pocket_recess_cup_gate_contract": REBUILD_POCKET_RECESS_CUP_GATE_NON_MUTATION_CONTRACT_V176J,
        "active_pocket_recess_cup_gate_extracted": True,
        "active_pocket_cap_geometry_helper_checkpoint": REBUILD_POCKET_CAP_GEOMETRY_HELPER_EXTRACTION_CHECKPOINT_V176M,
        "active_pocket_cap_geometry_helper_contract": REBUILD_POCKET_CAP_GEOMETRY_HELPER_NON_MUTATION_CONTRACT_V176M,
        "active_pocket_cap_geometry_helper_extracted": True,
        "active_pocket_floor_grid_helper_checkpoint": REBUILD_POCKET_FLOOR_GRID_HELPER_EXTRACTION_CHECKPOINT_V176U,
        "active_pocket_floor_grid_helper_contract": REBUILD_POCKET_FLOOR_GRID_HELPER_NON_MUTATION_CONTRACT_V176U,
        "active_pocket_floor_grid_helpers_extracted": True,
        "active_bore_frame_helper_checkpoint": REBUILD_BORE_FRAME_HELPER_EXTRACTION_CHECKPOINT_V176X,
        "active_bore_frame_helper_contract": REBUILD_BORE_FRAME_HELPER_NON_MUTATION_CONTRACT_V176X,
        "active_bore_frame_helpers_extracted": True,
        "active_bore_shape_authority_checkpoint": REBUILD_BORE_SHAPE_AUTHORITY_EXTRACTION_CHECKPOINT_V176Y,
        "active_bore_shape_authority_contract": REBUILD_BORE_SHAPE_AUTHORITY_NON_MUTATION_CONTRACT_V176Y,
        "active_semantic_helper_checkpoint": REBUILD_SEMANTIC_HELPER_EXTRACTION_CHECKPOINT_V176Z,
        "active_semantic_helper_contract": REBUILD_SEMANTIC_HELPER_NON_MUTATION_CONTRACT_V176Z,
        "active_semantic_shaping_policy_checkpoint": REBUILD_SEMANTIC_SHAPING_POLICY_EXTRACTION_CHECKPOINT_V177A,
        "active_semantic_shaping_policy_contract": REBUILD_SEMANTIC_SHAPING_POLICY_NON_MUTATION_CONTRACT_V177A,
        "active_quad_plan_extraction_checkpoint": REBUILD_QUAD_PLAN_EXTRACTION_CHECKPOINT_V177C,
        "active_quad_plan_extraction_contract": REBUILD_QUAD_PLAN_NON_MUTATION_CONTRACT_V177C,
        "active_quad_plan_dataclass_hotfix_checkpoint": REBUILD_QUAD_PLAN_DATACLASS_HOTFIX_CHECKPOINT_V177D,
        "active_quad_plan_dataclass_hotfix_contract": REBUILD_QUAD_PLAN_DATACLASS_HOTFIX_NON_MUTATION_CONTRACT_V177D,
        "active_inventory_refresh_checkpoint": REBUILD_INVENTORY_REFRESH_CHECKPOINT_V177E,
        "active_inventory_refresh_contract": REBUILD_INVENTORY_REFRESH_NON_MUTATION_CONTRACT_V177E,
        "current_module_status_v177e": {
            key: tuple(values) for key, values in REBUILD_V177E_CURRENT_MODULE_STATUS.items()
        },
        "next_safe_refactor_steps_v177e": REBUILD_V177E_NEXT_SAFE_REFACTOR_STEPS,
        "active_bore_shape_authority_extracted": True,
        "pocket_generator_inventory": rebuild_pocket_generator_inventory_v176h(),
        "pocket_patch_plan_shell_inventory": pocket_patch_plan_shell_inventory_v176i(),
        "safe_extraction_order": REBUILD_SAFE_EXTRACTION_ORDER_V176A,
        "module_inventory": {
            key: tuple(values) for key, values in REBUILD_MODULE_INVENTORY_V176A.items()
        },
        "authority_boundaries": dict(REBUILD_AUTHORITY_BOUNDARIES_V176A),
    }



def rebuild_refactor_inventory_v177e() -> dict[str, object]:
    """Return the v177e rebuild inventory refresh after v177c/v177d.

    This is informational only.  It documents the current extracted-module
    responsibility map after the generic QuadPlan split and dataclass hotfix.
    It must not be used as runtime permission, geometry generation, validation,
    mutation, or result authority.
    """

    base = dict(rebuild_refactor_inventory_v176a())
    base.update({
        "checkpoint": REBUILD_INVENTORY_REFRESH_CHECKPOINT_V177E,
        "non_mutation_contract": REBUILD_INVENTORY_REFRESH_NON_MUTATION_CONTRACT_V177E,
        "v177d_hotfix_checkpoint": REBUILD_QUAD_PLAN_DATACLASS_HOTFIX_CHECKPOINT_V177D,
        "v177d_hotfix_contract": REBUILD_QUAD_PLAN_DATACLASS_HOTFIX_NON_MUTATION_CONTRACT_V177D,
        "current_module_status": {
            key: tuple(values) for key, values in REBUILD_V177E_CURRENT_MODULE_STATUS.items()
        },
        "next_safe_refactor_steps": REBUILD_V177E_NEXT_SAFE_REFACTOR_STEPS,
        "rebuild_py_remains_single_mutation_authority": True,
        "family_modules_return_patch_plan_data_only": True,
        "compileall_is_not_sufficient_for_decorator_extraction_bugs": True,
        "required_validation_after_extraction": (
            "python -m compileall far_mesh/core/bore",
            "python -c import_smoke_for_far_mesh_core_bore_rebuild_and_rebuild_plan",
            "manual_BORE_under_CHAMFER_rebuild_regression",
            "manual_POCKET_recess_cup_rebuild_regression",
            "manual_CHAMFER_rebuild_regression",
        ),
    })
    return base


def rebuild_refactor_inventory_v177f() -> dict[str, object]:
    """Return the v177f rebuild validation-quality extraction inventory.

    This is informational only.  The active behavior change is intentionally
    none: the previous private ``rebuild.py`` geometry-quality gate is imported
    back under the same private alias from ``rebuild_validation.py``.  The helper
    remains a validation-quality gate only; it does not authorize deletion,
    generate geometry, apply a trial mesh, mutate topology, or emit a
    ``RebuildResult``.
    """

    base = dict(rebuild_refactor_inventory_v177e())
    base.update({
        "checkpoint": REBUILD_VALIDATION_QUALITY_GATE_REFRESH_CHECKPOINT_V177F,
        "non_mutation_contract": REBUILD_VALIDATION_QUALITY_GATE_REFRESH_NON_MUTATION_CONTRACT_V177F,
        "active_plan_geometry_quality_checkpoint": REBUILD_PLAN_GEOMETRY_QUALITY_EXTRACTION_CHECKPOINT_V177F,
        "active_plan_geometry_quality_contract": REBUILD_PLAN_GEOMETRY_QUALITY_NON_MUTATION_CONTRACT_V177F,
        "plan_geometry_quality_gate_moved_to_rebuild_validation": True,
        "rebuild_py_private_callsite_alias_preserved": True,
        "rebuild_py_remains_single_mutation_authority": True,
        "next_safe_refactor_steps": (
            "run_import_smoke_after_v177f",
            "manual_BORE_under_CHAMFER_wave_regression",
            "manual_POCKET_recess_cup_regression",
            "manual_CHAMFER_rebuild_regression",
            "then_consider_only_small_remaining_pure_helpers_or_stop",
        ),
    })
    return base


def rebuild_refactor_inventory_v177g() -> dict[str, object]:
    """Return the v177g POCKET small unequal-loop fallback inventory.

    This documents the targeted regression fix for normal POCKET recess-cup
    rebuilds where the owned side-wall top and floor loops differ by one locked
    boundary vertex after coarse remeshes or after nearby feature rebuilds.  The
    fix does not move mutation authority: the family-local adaptive sew collar
    only produces plan triangles, while ``rebuild.py`` still applies the trial,
    validates topology, and emits ``RebuildResult``.
    """

    base = dict(rebuild_refactor_inventory_v177f())
    base.update({
        "checkpoint": REBUILD_POCKET_SMALL_UNEQUAL_LOOP_ADAPTIVE_COLLAR_CHECKPOINT_V177G,
        "non_mutation_contract": REBUILD_POCKET_SMALL_UNEQUAL_LOOP_ADAPTIVE_COLLAR_CONTRACT_V177G,
        "pocket_small_unequal_loop_fallback_enabled": True,
        "pocket_small_unequal_loop_fallback_scope": (
            "entity_type_pocket_or_circular_pocket",
            "recess_cup_sidewall_plan_only",
            "locked_loop_count_delta_1_or_2",
            "generic_all_quad_plan_rejected",
        ),
        "pocket_large_unequal_loop_rejection_preserved": True,
        "rebuild_py_remains_single_mutation_authority": True,
        "family_modules_return_patch_plan_data_only": True,
        "required_validation_after_v177g": (
            "python -m compileall far_mesh/core/bore",
            "python -m far_mesh.main",
            "manual_normal_POCKET_recess_cup_22x23_or_23x22_regression",
            "manual_previous_POCKET_17x17_regression",
            "manual_BORE_under_CHAMFER_wave_regression",
            "manual_CHAMFER_rebuild_regression",
        ),
    })
    return base


def rebuild_refactor_inventory_v177h() -> dict[str, object]:
    out = rebuild_refactor_inventory_v177g()
    out = dict(out)
    out["checkpoint"] = REBUILD_POCKET_LINEAR_SIDEWALL_AND_EXPLICIT_FLOOR_CAP_CHECKPOINT_V177H
    out["v177h_pocket_fix"] = {
        "pocket_sidewall_safe_mode": "wide locked pocket radius envelopes use boundary-linear generated side-wall rings instead of nominal cylinder projection",
        "floor_cap_authority": "floor cap loop is the explicit owned floor perimeter, not inferred from planned side-wall loop intersection",
        "mutation_authority": "unchanged: rebuild.py still applies trial mesh, validates, and emits RebuildResult",
    }
    out.setdefault("non_mutation_contracts", ())
    try:
        out["non_mutation_contracts"] = tuple(out.get("non_mutation_contracts", ())) + (
            REBUILD_POCKET_LINEAR_SIDEWALL_AND_EXPLICIT_FLOOR_CAP_NON_MUTATION_CONTRACT_V177H,
        )
    except Exception:
        pass
    return out

def rebuild_refactor_inventory_v177i() -> dict[str, object]:
    """Return the v177i POCKET floor-loop binding hotfix inventory.

    v177i is a no-behavior-change runtime hotfix for v177h.  It binds the
    explicit floor_loop_vertices returned by resolve_pocket_recess_loop_roles_v176e
    before the v177h floor-cap guard consumes it.
    """

    out = rebuild_refactor_inventory_v177h()
    out = dict(out)
    out["checkpoint"] = REBUILD_POCKET_FLOOR_LOOP_BINDING_HOTFIX_CHECKPOINT_V177I
    out["v177i_hotfix"] = {
        "fixed_symbol": "floor_loop_vertices",
        "bug_class": "refactor_transition_unbound_local_name_in_v177h_floor_cap_guard",
        "behavior_change": False,
        "geometry_policy_change": False,
        "mutation_authority": "unchanged: rebuild.py still applies trial mesh, validates, and emits RebuildResult",
    }
    try:
        out["non_mutation_contracts"] = tuple(out.get("non_mutation_contracts", ())) + (
            REBUILD_POCKET_FLOOR_LOOP_BINDING_HOTFIX_NON_MUTATION_CONTRACT_V177I,
        )
    except Exception:
        pass
    return out




def rebuild_refactor_inventory_v177j() -> dict[str, object]:
    """Return the v177j post-regression rebuild inventory.

    v177j is documentation/metadata only.  It records the regression lessons
    from the v177f/v177g/v177h/v177i POCKET and validation work before another
    extraction is attempted.  It intentionally does not move helpers, does not
    change geometry generation, does not alter validation policy, and does not
    touch mesh mutation authority.
    """

    out = dict(rebuild_refactor_inventory_v177i())
    out["checkpoint"] = REBUILD_POST_REGRESSION_INVENTORY_CHECKPOINT_V177J
    out["non_mutation_contract"] = REBUILD_POST_REGRESSION_INVENTORY_NON_MUTATION_CONTRACT_V177J
    try:
        out["non_mutation_contracts"] = tuple(out.get("non_mutation_contracts", ())) + (
            REBUILD_POST_REGRESSION_INVENTORY_NON_MUTATION_CONTRACT_V177J,
            REBUILD_PYTHON_ZERO_DISCIPLINE_CONTRACT_V177J,
        )
    except Exception:
        pass
    out["v177j_post_regression_lessons"] = {
        "v177f": (
            "Plan geometry quality gates belong in rebuild_validation.py only when they remain "
            "read-only rejection/diagnostic helpers; they must not authorize delete patches, "
            "generate geometry, apply trial meshes, or emit RebuildResult."
        ),
        "v177g": (
            "The generic all-quad loop planner may correctly reject small locked loop-count "
            "mismatches such as 22↔23; POCKET may use its family-local adaptive sew collar "
            "only as a plan fallback for small count drift, while rebuild.py still validates."
        ),
        "v177h": (
            "Watertight topology is not sufficient proof of visually valid POCKET geometry. "
            "The floor cap must be bound to explicit owned POCKET floor-loop evidence, and "
            "wide locked radius envelopes may require boundary-linear side-wall placement."
        ),
        "v177i": (
            "Refactor transition patches must bind moved semantic variables before use. "
            "The unbound floor_loop_vertices crash was a wiring error, not a Recognition, "
            "Rebuild Target, or mesh mutation policy change."
        ),
    }
    out["v177j_smoke_test_baseline"] = {
        "reported_by_user": True,
        "mesh": "champfer_and_holes_pocked_holed_coarse_remesh_obj",
        "passed_cases": (
            "POCKET_22x22_recess_cup_committed_radiusBoreLeak0_radiusSrc_pocket_candidate_radius_v163",
            "BORE_committed_radius_authority_owned_bore_wall_cylinder_shape_authority_v173x",
            "CHAMFER_committed_radius_authority_chamfer_boundary_radii_v163",
            "POCKET_17x17_recess_cup_committed_radiusBoreLeak0_radiusSrc_pocket_candidate_radius_v163",
        ),
        "remaining_pocket_wave_diagnostic_is_not_auto_failure": True,
        "remaining_pocket_wave_diagnostic_policy": (
            "Do not patch geometry solely because pocket_trace_v175k reports wave1. "
            "Treat wave1 as a quality diagnostic unless there is visible distortion, "
            "boundary mismatch, radius authority leak, crash, or failed validation."
        ),
    }
    out["python_zero_discipline_v177j"] = {
        "valid_zero_count_rule": (
            "For topology/validation counts, zero can be the success value. "
            "Use explicit None checks or int_preserve_zero; never use value-or-fallback."
        ),
        "missing_geometry_not_zero_rule": (
            "For required geometry/evidence such as radius, depth, loop vertices, floor-loop proof, "
            "or boundary authority, do not let missing/invalid data collapse into a fake valid zero. "
            "Reject or report missing evidence explicitly."
        ),
        "examples": (
            "boundary_edge_count_after=0 is valid success when watertight_after=True",
            "radius=0.0 is not a valid circular pocket radius",
            "floor_loop_vertices=() is missing floor-loop evidence, not an acceptable zero-sized floor",
        ),
    }
    out["next_safe_refactor_steps"] = (
        "stop_geometry_changes_until_a_new_visual_or_validation_failure_is_reproduced",
        "run_compileall_and_import_smoke_after_every_extraction_not_compileall_only",
        "keep_rebuild_py_as_single_mutation_trial_validation_RebuildResult_authority",
        "if continuing cleanup_choose_only_read_only_diagnostics_or_small_pure_helpers",
        "do_not_move_floor_cap_or_sidewall_geometry_again_until_pocket_wave_diagnostic_has_a_measured_threshold",
    )
    out["rebuild_py_remains_single_mutation_authority"] = True
    out["behavior_change"] = False
    out["geometry_policy_change"] = False
    return out


def rebuild_refactor_inventory_v177k() -> dict[str, object]:
    """Return the v177k zero-safety audit inventory.

    v177k is a conservative zero-discipline cleanup after the v177i/v177j smoke
    baseline.  It keeps valid zero topology counts valid, but prevents missing
    boundary-mismatch diagnostics from collapsing into fake zero-success values.
    It does not change geometry generation, target policy, CandidateData, or
    mesh mutation authority.
    """

    out = dict(rebuild_refactor_inventory_v177j())
    out["checkpoint"] = REBUILD_ZERO_SAFETY_AUDIT_CHECKPOINT_V177K
    out["non_mutation_contract"] = REBUILD_ZERO_SAFETY_AUDIT_NON_MUTATION_CONTRACT_V177K
    try:
        out["non_mutation_contracts"] = tuple(out.get("non_mutation_contracts", ())) + (
            REBUILD_ZERO_SAFETY_AUDIT_NON_MUTATION_CONTRACT_V177K,
        )
    except Exception:
        pass
    out["v177k_zero_safety_audit"] = {
        "scan_scope": "all uploaded BoreTool Python files plus sequential v177d-v177j overlays",
        "high_risk_fixed": (
            "rebuild_validation.trial_accepts_for_context missing/extra/patch boundary diagnostics",
            "rebuild.py pocket_cap local_accept missing/extra/patch boundary diagnostics",
            "rebuild.py damaged-bore residual boundary count read",
        ),
        "valid_zero_preserved": (
            "boundary_edge_count_after=0 remains a success when paired with watertight_after=True",
            "missing_patch_boundary_edge_count=0 remains exact boundary preservation",
            "extra_generated_boundary_edge_count=0 remains exact boundary preservation",
            "generated_boundary_generated_vertex_edge_count=0 remains success for locked-boundary local acceptance",
        ),
        "missing_not_promoted_to_zero": (
            "missing_patch_boundary_edge_count=None or invalid now uses rejection default 1",
            "extra_generated_boundary_edge_count=None or invalid now uses rejection default 1",
            "patch_boundary_edge_count=None or invalid now uses rejection default -1",
        ),
        "diagnostic_only_or_safe_remaining_patterns": (
            "numeric value-or-zero formatting remains in recognition/emit summaries where zero is display/reporting only",
            "geometry radii/depths that use 0.0 defaults are still guarded by positive-radius/depth checks before promotion",
        ),
    }
    out["behavior_change"] = False
    out["geometry_policy_change"] = False
    out["rebuild_py_remains_single_mutation_authority"] = True
    return out
