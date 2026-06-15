# far_mesh/core/hole_fill_preview.py
"""
Phase 2E preview-only hole filling helpers for FAR MESH Quad 3.

This module is intentionally GUI-free and viewport-free.

It builds non-destructive preview meshes from HoleCandidate objects. It does
not commit results, mutate MeshProcessor state, or perform viewport work.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Iterable, Mapping

import numpy as np
import trimesh

from far_mesh.core.open3d_tensor_bridge import fill_holes_with_open3d_tensor
from far_mesh.core.hole_curvature import (
    build_curvature_sphere_center_fan_preview_mesh,
    build_curvature_sphere_refined_preview_mesh,
    build_curvature_sphere_grid8_preview_mesh,
    build_curvature_sphere_uvgrid_preview_mesh,
    build_curvature_sphere_uvdelaunay_preview_mesh,
    build_curvature_sphere_uvdelaunay_relaxed_preview_mesh,
    build_surface_uvdelaunay_relaxed_preview_mesh,
    build_surface_uvdelaunay_sealed_relaxed_preview_mesh,
    build_surface_uvdelaunay_sealed_dense_relaxed_preview_mesh,
)

from .selection_topology import HoleCandidate
from far_mesh.core.hole_fill.seed_surface import (
    analyze_seed_surface_alignment,
    analyze_support_target_disagreement,
)


SUPPORTED_HOLE_FILL_METHODS = {"fan", "fan_triangulate", "center_fan"}

ADAPTIVE_SURFACE_METHOD = "adaptive_surface"
ADAPTIVE_SURFACE_V2_METHOD = "adaptive_surface_v2"

# Public optional methods shown by capability/UI helpers.
OPTIONAL_HOLE_FILL_METHODS = {
    "open3d",
    ADAPTIVE_SURFACE_V2_METHOD,
}
ALL_HOLE_FILL_METHODS = SUPPORTED_HOLE_FILL_METHODS | OPTIONAL_HOLE_FILL_METHODS

# Legacy/dev-only routes. These remain callable for internal comparison and
# old project/debug workflows, but they are no longer public UI methods.
_LEGACY_EXPERIMENTAL_HOLE_FILL_METHODS = {
    "curvature_sphere",
    "curvature_sphere_refined",
    "curvature_sphere_grid8",
    "curvature_sphere_uvgrid",
    "curvature_sphere_uvdelaunay",
    "curvature_sphere_uvdelaunay_relaxed",
    "surface_uvdelaunay_relaxed",
    "surface_uvdelaunay_sealed_relaxed",
    "surface_uvdelaunay_sealed_dense_relaxed",
}


def _normalize_method_key(method: object) -> str:
    method_key = str(method or "fan").strip().lower().replace("-", "_")

    if method_key == "triangulate_boundary_fan":
        return "fan"
    if method_key == "open3d_fill":
        return "open3d"
    if method_key in {
        "adaptive_v2",
        "adaptive_surface_v2",
        "adaptive_surface_fill_v2",
    }:
        return ADAPTIVE_SURFACE_V2_METHOD

    if method_key in {
        "adaptive",
        "adaptive_surface",
        "adaptive_surface_fill",
        "adaptive_uvdelaunay_relaxed",
    }:
        return ADAPTIVE_SURFACE_V2_METHOD

    return method_key


def is_open3d_import_available() -> bool:
    """Return whether the Open3D Python package can be imported.

    This is an import-spec check only. It does not mean the Open3D hole-fill
    backend is implemented or enabled.
    """

    return importlib.util.find_spec("open3d") is not None


def is_open3d_hole_fill_backend_available() -> bool:
    """Return whether the optional Open3D tensor fill_holes backend is usable."""

    try:
        import open3d as o3d  # type: ignore
    except Exception:
        return False

    try:
        return hasattr(o3d.t.geometry.TriangleMesh, "fill_holes")
    except Exception:
        return False


def _hole_fill_method_capability_builtin(method: str | None) -> dict[str, object]:
    """Describe whether a hole-fill preview method is available.

    This helper is read-only and has no side effects. It exists so the GUI and
    future routed task handlers can make capability decisions without trying to
    execute a preview first.
    """

    normalized = _normalize_method_key(method)

    if normalized in SUPPORTED_HOLE_FILL_METHODS:
        return {
            "method": normalized,
            "available": True,
            "backend": "trimesh_fan",
            "reason": "Built-in preview method.",
            "preview_only": True,
        }

    if normalized == "open3d":
        import_available = is_open3d_import_available()
        backend_available = is_open3d_hole_fill_backend_available()

        if not import_available:
            reason = "Open3D Python package is not installed."
        elif not backend_available:
            reason = "Open3D tensor TriangleMesh.fill_holes is unavailable."
        else:
            reason = "Open3D tensor TriangleMesh.fill_holes backend is available."

        return {
            "method": "open3d",
            "available": backend_available,
            "backend": "open3d",
            "open3d_import_available": import_available,
            "reason": reason,
            "preview_only": True,
        }

    raise ValueError(f"Unsupported hole fill preview method: {method!r}.")



def hole_fill_method_capability(method: str) -> dict[str, object]:
    method_key = _normalize_method_key(method)


    if method_key == ADAPTIVE_SURFACE_V2_METHOD:
        return {
            "method": ADAPTIVE_SURFACE_V2_METHOD,
            "available": True,
            "backend": "adaptive_surface_v2_curvature_normal_seed",
            "reason": (
                "Adaptive Surface Fill v2 direct GUI route. Builds the "
                "curvature-normal-aligned v2 seed candidate as the canonical "
                "public adaptive method. Old adaptive_surface names route here."
            ),
            "preview_only": True,
            "experimental": True,
            "public": True,
        }

    if method_key == ADAPTIVE_SURFACE_METHOD:
        return {
            "method": ADAPTIVE_SURFACE_METHOD,
            "available": True,
            "backend": "adaptive_surface_v2_curvature_normal_seed",
            "reason": (
                "Compatibility alias for Adaptive Surface Fill v2. The v1 "
                "public adaptive controller has been retired; old method names "
                "route into the canonical v2 path."
            ),
            "preview_only": True,
            "experimental": True,
            "public": False,
            "alias_for": ADAPTIVE_SURFACE_V2_METHOD,
        }

    if method_key in _LEGACY_EXPERIMENTAL_HOLE_FILL_METHODS:
        return {
            "method": method_key,
            "available": True,
            "backend": method_key,
            "reason": (
                "Legacy experimental hole-fill preview route. Kept for "
                "internal comparison, not exposed as a public UI method."
            ),
            "preview_only": True,
            "experimental": True,
            "public": False,
        }

    return _hole_fill_method_capability_builtin(method)


_HOLE_FILL_PREVIEW_METADATA_KEYS = (
    "topology_before",
    "topology_after",
    "quality_before",
    "quality_after",
    "relaxation",
    "dynamic_diagnostics",
    "seam_constraint_report",
    "seam_recovery_decision",
    "constrained_edge_recovery_report",
    "boundary_loop_diagnostic",
    "surface_context_decision",
    "local_density_budget_decision",
    "local_density_refinement_budget",
    "preseed_local_density_budget_decision",
    "preseed_local_density_refinement_budget",
    "seed_max_interior_points",
    "new_face_ids",
    "new_vertex_ids",
    "delaunay_triangle_count",
    "interior_uv_count",
    "biharmonic_fairing_decision",
    "biharmonic_fairing",
    "fit_report",
    "seed_surface_error",
    "seed_surface_kind",
    "adaptive_seed_context_edge_length_median",
    "adaptive_seed_generated_vertex_count",
    "adaptive_seed_support_normal_spread_degrees",
    "adaptive_seed_requested_surface_weight",
    "adaptive_seed_effective_surface_weight",
    "adaptive_seed_signed_offset_max",
    "adaptive_seed_signed_offset_min",
    "adaptive_seed_signed_offset_mean",
    "adaptive_seed_projection_max_ratio",
    "adaptive_seed_projection_mean_ratio",
    "adaptive_seed_projection_distance_max",
    "adaptive_seed_projection_distance_mean",
    "adaptive_seed_alignment_reasons",
    "adaptive_target_mls_ring3_normal_spread_degrees",
    "adaptive_target_mls_ring2_normal_spread_degrees",
    "adaptive_target_mls_ring1_normal_spread_degrees",
    "adaptive_target_signed_disagreement_max",
    "adaptive_target_signed_disagreement_min",
    "adaptive_target_signed_disagreement_mean",
    "adaptive_target_mls2_vs_mls3_max",
    "adaptive_target_mls2_vs_mls3_mean",
    "adaptive_target_mls1_vs_mls2_max",
    "adaptive_target_mls1_vs_mls2_mean",
    "adaptive_target_mls2_vs_plane_max",
    "adaptive_target_mls2_vs_plane_mean",
    "adaptive_target_mls2_vs_sphere_max",
    "adaptive_target_mls2_vs_sphere_mean",
    "adaptive_target_disagreement_max_ratio",
    "adaptive_target_disagreement_mean_ratio",
    "adaptive_target_disagreement_max",
    "adaptive_target_disagreement_mean",
    "adaptive_target_disagreement_reasons",
    "adaptive_target_disagreement_action",
    "adaptive_target_disagreement_status",
    "support_target_disagreement",
    "adaptive_seed_alignment_action",
    "adaptive_seed_alignment_status",
    "seed_surface_alignment",
    "preseed_relaxation_confidence_profile",
    "support_context",
    "surface_guidance",
    "seed_backend",
    "relaxation_iterations",
    "relaxation_strength",
    "relaxation_surface_weight",
    "relaxed_max_displacement",
    "relaxed_mean_displacement",
    "adaptive_controller",
    "adaptive_controller_schema_version",
    "adaptive_public_method",
    "adaptive_selected_strategy",
    "adaptive_attempted_strategies",
    "adaptive_fallback_used",
    "adaptive_fallback_reason",
    "adaptive_fallback_error",
    "adaptive_primary_seam_decision",
    "adaptive_fallback_seam_decision",
    "adaptive_controller_notes",
    "adaptive_selected_surface_weight",
    "adaptive_selected_relaxation_strength",
    "adaptive_selected_relaxation_iterations",
    "adaptive_conservative_g1_decision",
    "adaptive_conservative_g1_reason",
    "adaptive_conservative_g1_used",
    "adaptive_conservative_g1_attempted",
    "adaptive_primary_g1_decision",
    "adaptive_g1_policy_reasons",
    "adaptive_g1_relaxation_policy",
    "adaptive_feature_support_normal_spread_degrees",
    "adaptive_feature_boundary_vertex_count",
    "adaptive_feature_smooth_context_valid",
    "adaptive_feature_density_mode",
    "adaptive_feature_recommended_action",
    "adaptive_feature_policy_reasons",
    "adaptive_feature_preservation_mode",
    "adaptive_feature_context_kind",
    "adaptive_curvature_reasons",
    "adaptive_curvature_patch_sample_count",
    "adaptive_curvature_support_sample_count",
    "adaptive_curvature_sign_consistency",
    "adaptive_curvature_relative_delta_mean",
    "adaptive_curvature_delta_max",
    "adaptive_curvature_delta_mean",
    "adaptive_patch_curvature_std",
    "adaptive_patch_curvature_max",
    "adaptive_patch_curvature_mean",
    "adaptive_support_curvature_std",
    "adaptive_support_curvature_max",
    "adaptive_support_curvature_mean",
    "adaptive_curvature_estimator",
    "adaptive_curvature_context_kind",
    "adaptive_curvature_status",
    "adaptive_curvature_fairing_reasons",
    "adaptive_curvature_fairing_max_displacement_factor",
    "adaptive_curvature_fairing_iterations",
    "adaptive_curvature_fairing_strength",
    "adaptive_curvature_fairing_needed",
    "adaptive_curvature_fairing_eligible",
    "adaptive_curvature_fairing_action",
    "adaptive_curvature_fairing_status",
    "adaptive_curvature_fairing_trial_movement_to_context_edge_ratio",
    "adaptive_curvature_fairing_trial_mean_displacement",
    "adaptive_curvature_fairing_trial_max_displacement",
    "adaptive_curvature_fairing_trial_error",
    "adaptive_curvature_fairing_trial_notes",
    "adaptive_curvature_fairing_trial_reasons",
    "adaptive_curvature_fairing_trial_mode",
    "adaptive_curvature_fairing_trial_accepted",
    "adaptive_curvature_fairing_trial_applied",
    "adaptive_curvature_fairing_trial_attempted",
    "adaptive_curvature_fairing_trial_action",
    "adaptive_curvature_fairing_trial_status",
    "adaptive_end_layer_reasons",
    "adaptive_end_layer_selected_patch_source",
    "adaptive_end_layer_rerun_reason",
    "adaptive_end_layer_rerun_allowed",
    "adaptive_end_layer_refinement_recommended",
    "adaptive_end_layer_problem_region_count",
    "adaptive_end_layer_geometry_deviation_max",
    "adaptive_end_layer_geometry_deviation_mean",
    "adaptive_end_layer_curvature_relative_deviation_mean",
    "adaptive_end_layer_curvature_deviation_max",
    "adaptive_end_layer_curvature_deviation_mean",
    "adaptive_end_layer_patch_vertex_count",
    "adaptive_end_layer_support_vertex_count",
    "adaptive_end_layer_support_ring_count",
    "adaptive_end_layer_reference_patch_available",
    "adaptive_end_layer_patch_available",
    "adaptive_end_layer_local_region_available",
    "adaptive_end_layer_action",
    "adaptive_end_layer_status",
    "adaptive_score_delta",
    "adaptive_surface_v2_status",
    "adaptive_surface_v2_case",
    "adaptive_surface_v2_action",
    "adaptive_surface_v2_block_legacy_selection",
    "adaptive_surface_v2_allow_confidence_target_delta",
    "adaptive_surface_v2_allow_local_anisotropic_correction",
    "adaptive_surface_v2_require_new_seed",
    "adaptive_surface_v2_recommended_seed_family",
    "adaptive_surface_v2_recommended_target_policy",
    "adaptive_surface_v2_reasons",
    "adaptive_surface_v2_seed_plan_status",
    "adaptive_surface_v2_seed_plan_action",
    "adaptive_surface_v2_seed_plan_build_required",
    "adaptive_surface_v2_seed_family",
    "adaptive_surface_v2_orientation_case",
    "adaptive_surface_v2_orientation_action",
    "adaptive_surface_v2_target_policy",
    "adaptive_surface_v2_curvature_policy",
    "adaptive_surface_v2_confidence_policy",
    "adaptive_surface_v2_support_context_policy",
    "adaptive_surface_v2_boundary_normal_mean_deviation",
    "adaptive_surface_v2_boundary_normal_max_deviation",
    "adaptive_surface_v2_support_normal_spread",
    "adaptive_surface_v2_seed_projection_mean_ratio",
    "adaptive_surface_v2_seed_projection_max_ratio",
    "adaptive_surface_v2_target_confidence",
    "adaptive_surface_v2_target_low_confidence_fraction",
    "adaptive_surface_v2_curvature_relative_delta_mean",
    "adaptive_surface_v2_curvature_sign_consistency",
    "adaptive_surface_v2_seed_plan_reasons",
    "adaptive_surface_v2_seed_prototype_status",
    "adaptive_surface_v2_seed_prototype_action",
    "adaptive_surface_v2_seed_prototype_build_required",
    "adaptive_surface_v2_seed_prototype_family",
    "adaptive_surface_v2_seed_prototype_geometry_status",
    "adaptive_surface_v2_seed_prototype_orientation_status",
    "adaptive_surface_v2_seed_prototype_orientation_action",
    "adaptive_surface_v2_seed_prototype_orientation_confidence",
    "adaptive_surface_v2_seed_prototype_boundary_normal_mean_deviation",
    "adaptive_surface_v2_seed_prototype_boundary_normal_max_deviation",
    "adaptive_surface_v2_seed_prototype_side_score_mean",
    "adaptive_surface_v2_seed_prototype_side_score_max",
    "adaptive_surface_v2_seed_prototype_normal_sign_status",
    "adaptive_surface_v2_seed_prototype_support_context_policy",
    "adaptive_surface_v2_seed_prototype_target_policy",
    "adaptive_surface_v2_seed_prototype_curvature_policy",
    "adaptive_surface_v2_seed_prototype_confidence_policy",
    "adaptive_surface_v2_seed_prototype_seed_projection_mean_ratio",
    "adaptive_surface_v2_seed_prototype_seed_projection_max_ratio",
    "adaptive_surface_v2_seed_prototype_target_confidence",
    "adaptive_surface_v2_seed_prototype_target_low_confidence_fraction",
    "adaptive_surface_v2_seed_prototype_support_normal_spread",
    "adaptive_surface_v2_seed_prototype_curvature_relative_delta_mean",
    "adaptive_surface_v2_seed_prototype_curvature_sign_consistency",
    "adaptive_surface_v2_seed_prototype_reasons",
    "adaptive_surface_v2_seed_candidate_status",
    "adaptive_surface_v2_seed_candidate_action",
    "adaptive_surface_v2_seed_candidate_family",
    "adaptive_surface_v2_seed_candidate_geometry_status",
    "adaptive_surface_v2_seed_candidate_selectable",
    "adaptive_surface_v2_seed_candidate_curvature_normal_field_status",
    "adaptive_surface_v2_seed_candidate_support_filter",
    "adaptive_surface_v2_seed_candidate_frame_policy",
    "adaptive_surface_v2_seed_candidate_target_policy",
    "adaptive_surface_v2_seed_candidate_density_policy",
    "adaptive_surface_v2_seed_candidate_acceptance_policy",
    "adaptive_surface_v2_seed_candidate_boundary_vertices",
    "adaptive_surface_v2_seed_candidate_legacy_seed_vertices",
    "adaptive_surface_v2_seed_candidate_planned_seed_vertices",
    "adaptive_surface_v2_seed_candidate_planned_support_rings",
    "adaptive_surface_v2_seed_candidate_planned_interior_rings",
    "adaptive_surface_v2_seed_candidate_normal_continuity_mismatch_score",
    "adaptive_surface_v2_seed_candidate_target_confidence",
    "adaptive_surface_v2_seed_candidate_target_low_confidence_fraction",
    "adaptive_surface_v2_seed_candidate_seed_projection_max_ratio",
    "adaptive_surface_v2_seed_candidate_curvature_relative_delta_mean",
    "adaptive_surface_v2_seed_candidate_curvature_sign_consistency",
    "adaptive_surface_v2_seed_candidate_reasons",
    "adaptive_surface_v2_seed_candidate_geometry_action",
    "adaptive_surface_v2_seed_candidate_geometry_available",
    "adaptive_surface_v2_seed_candidate_geometry_applied",
    "adaptive_surface_v2_seed_candidate_geometry_selected",
    "adaptive_surface_v2_seed_candidate_geometry_family",
    "adaptive_surface_v2_seed_candidate_geometry_mode",
    "adaptive_surface_v2_seed_candidate_geometry_face_count",
    "adaptive_surface_v2_seed_candidate_geometry_vertex_count",
    "adaptive_surface_v2_seed_candidate_geometry_reoriented_face_count",
    "adaptive_surface_v2_seed_candidate_geometry_moved_vertex_count",
    "adaptive_surface_v2_seed_candidate_geometry_movement_mean",
    "adaptive_surface_v2_seed_candidate_geometry_movement_max",
    "adaptive_surface_v2_seed_candidate_geometry_movement_ratio_max",
    "adaptive_surface_v2_seed_candidate_geometry_predicted_g1_mean_deviation",
    "adaptive_surface_v2_seed_candidate_geometry_predicted_g1_max_deviation",
    "adaptive_surface_v2_seed_candidate_geometry_predicted_g1_status",
    "adaptive_surface_v2_seed_candidate_geometry_topology_status",
    "adaptive_surface_v2_seed_candidate_geometry_topology_reasons",
    "adaptive_surface_v2_seed_candidate_geometry_reasons",
    "adaptive_surface_v2_seed_candidate_geometry_evaluation_status",
    "adaptive_surface_v2_seed_candidate_geometry_evaluation_action",
    "adaptive_surface_v2_seed_candidate_geometry_evaluation_selectable",
    "adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_status",
    "adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_mean_deviation",
    "adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_max_deviation",
    "adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_mean_limit",
    "adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_max_limit",
    "adaptive_surface_v2_seed_candidate_geometry_evaluation_quality_status",
    "adaptive_surface_v2_seed_candidate_geometry_evaluation_g2_status",
    "adaptive_surface_v2_seed_candidate_geometry_evaluation_policy",
    "adaptive_surface_v2_seed_candidate_geometry_evaluation_reasons",
    "adaptive_surface_v2_selected_mesh_source",
    "adaptive_score_decision",
    "adaptive_selected_score_breakdown",
    "adaptive_selected_score",
    "adaptive_conservative_g1_score_breakdown",
    "adaptive_conservative_g1_score",
    "adaptive_primary_score_breakdown",
    "adaptive_primary_score",
)


def _attach_hole_fill_preview_metadata(preview_mesh, result):
    """Attach preview-builder metadata to the returned trimesh preview mesh."""
    if not isinstance(result, dict):
        return preview_mesh

    raw_metadata = getattr(preview_mesh, "metadata", None)
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}

    for key in _HOLE_FILL_PREVIEW_METADATA_KEYS:
        if key in result:
            metadata[key] = result[key]

    preview_mesh.metadata.update(metadata)
    return preview_mesh


def available_hole_fill_preview_methods(*, include_unavailable: bool = False) -> tuple[str, ...]:
    """Return public hole-fill preview methods in stable UI order."""

    ordered = (
        "fan",
        "fan_triangulate",
        "center_fan",
        ADAPTIVE_SURFACE_V2_METHOD,
        "open3d",
    )
    methods: list[str] = []

    for method in ordered:
        capability = hole_fill_method_capability(method)
        if include_unavailable or bool(capability.get("available")):
            methods.append(str(capability.get("method") or method))

    return tuple(methods)


def _normalize_method(method: str | None) -> str:
    normalized = _normalize_method_key(method)

    if normalized in SUPPORTED_HOLE_FILL_METHODS:
        return normalized

    if normalized == ADAPTIVE_SURFACE_V2_METHOD:
        capability = hole_fill_method_capability(normalized)
        if bool(capability.get("available")):
            return normalized
        raise ValueError(
            "Adaptive Surface Fill v2 preview is not available: "
            f"{capability.get('reason')}"
        )

    if normalized == ADAPTIVE_SURFACE_METHOD:
        return ADAPTIVE_SURFACE_V2_METHOD

    if normalized in _LEGACY_EXPERIMENTAL_HOLE_FILL_METHODS:
        capability = hole_fill_method_capability(normalized)
        if bool(capability.get("available")):
            return normalized
        raise ValueError(
            "Legacy experimental hole fill preview is not available: "
            f"{capability.get('reason')}"
        )

    if normalized == "open3d":
        capability = hole_fill_method_capability(normalized)
        if bool(capability.get("available")):
            return normalized
        raise ValueError(
            "Open3D hole fill preview is not available: "
            f"{capability.get('reason')}"
        )

    raise ValueError(
        f"Unsupported hole fill preview method: {method!r}. "
        "Available public methods: fan, fan_triangulate, center_fan, "
        "adaptive_surface, adaptive_surface_v2, open3d."
    )


def _ordered_candidate_vertices(candidate: HoleCandidate) -> tuple[int, ...]:
    vertices = tuple(int(v) for v in getattr(candidate, "boundary_vertices", ()) or ())

    # Prefer the loop order when available. It may include a duplicated closing
    # vertex, so remove that duplicate before building faces.
    loop = getattr(candidate, "loop", None)
    loop_vertices = tuple(int(v) for v in getattr(loop, "vertices", ()) or ())
    if loop_vertices:
        if len(loop_vertices) >= 2 and loop_vertices[0] == loop_vertices[-1]:
            loop_vertices = loop_vertices[:-1]
        vertices = loop_vertices

    # Preserve order but remove repeated vertices if present.
    ordered: list[int] = []
    seen: set[int] = set()
    for vertex_id in vertices:
        if vertex_id < 0 or vertex_id in seen:
            continue
        ordered.append(vertex_id)
        seen.add(vertex_id)

    return tuple(ordered)


def _validate_candidate_vertices(mesh: trimesh.Trimesh, vertices: Iterable[int]) -> tuple[int, ...]:
    vertex_ids = tuple(int(v) for v in vertices)
    if len(vertex_ids) < 3:
        raise ValueError("Hole fill preview requires at least three boundary vertices.")

    vertex_count = int(len(mesh.vertices))
    invalid = [v for v in vertex_ids if v < 0 or v >= vertex_count]
    if invalid:
        raise ValueError(f"Hole fill preview candidate contains invalid vertex ids: {invalid}")

    return vertex_ids



def _source_directed_edges(source_faces: np.ndarray) -> set[tuple[int, int]]:
    """Return directed triangle edges from the source mesh faces."""
    directed_edges: set[tuple[int, int]] = set()

    faces = np.asarray(source_faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] < 3:
        return directed_edges

    for raw_face in faces[:, :3]:
        a, b, c = (int(raw_face[0]), int(raw_face[1]), int(raw_face[2]))
        if a == b or b == c or c == a:
            continue
        directed_edges.add((a, b))
        directed_edges.add((b, c))
        directed_edges.add((c, a))

    return directed_edges


def _patch_boundary_orientation_votes(
    source_faces: np.ndarray,
    patch_faces_global: np.ndarray,
) -> tuple[int, int]:
    """Return ``(opposite_votes, same_votes)`` for patch/source boundary edges.

    A consistently wound fill patch must traverse each shared source-boundary
    edge in the opposite direction from the existing neighboring source face.
    When the patch traverses more shared edges in the same direction than in
    the opposite direction, the patch face winding should be flipped before the
    patch is appended to the source mesh.
    """
    source_edges = _source_directed_edges(source_faces)
    if not source_edges:
        return (0, 0)

    patch_faces = np.asarray(patch_faces_global, dtype=np.int64)
    if patch_faces.ndim != 2 or patch_faces.shape[0] == 0 or patch_faces.shape[1] < 3:
        return (0, 0)

    opposite_votes = 0
    same_votes = 0

    for raw_face in patch_faces[:, :3]:
        a, b, c = (int(raw_face[0]), int(raw_face[1]), int(raw_face[2]))
        for u, v in ((a, b), (b, c), (c, a)):
            if u == v:
                continue
            if (v, u) in source_edges:
                opposite_votes += 1
            if (u, v) in source_edges:
                same_votes += 1

    return opposite_votes, same_votes


def _orient_patch_faces_against_source_mesh(
    source_faces: np.ndarray,
    patch_faces_global: np.ndarray,
) -> np.ndarray:
    """Return patch faces with winding compatible with the source mesh.

    The input/output face IDs are global source-mesh vertex IDs.  This helper
    intentionally runs before the patch faces are appended, so the resulting
    preview keeps the source mesh untouched while making the new cap faces
    locally winding-consistent with the existing boundary faces.
    """
    patch_faces = np.asarray(patch_faces_global, dtype=np.int64).copy()
    if patch_faces.size == 0:
        return patch_faces.reshape((0, 3))
    if patch_faces.ndim != 2 or patch_faces.shape[1] < 3:
        return patch_faces

    opposite_votes, same_votes = _patch_boundary_orientation_votes(
        source_faces,
        patch_faces,
    )

    if same_votes > opposite_votes:
        patch_faces[:, [1, 2]] = patch_faces[:, [2, 1]]

    return patch_faces


def build_open3d_hole_fill_preview_mesh(
    mesh: trimesh.Trimesh,
    candidate: HoleCandidate,
    *,
    all_candidates: Iterable[HoleCandidate] | None = None,
) -> trimesh.Trimesh:
    """Build a non-destructive Open3D tensor fill_holes preview mesh.

    Safety note for 2J-B1:
    Open3D's tensor TriangleMesh.fill_holes() operates on the whole mesh. To
    preserve candidate-specific semantics, this helper only allows the preview
    when the caller proves there is exactly one fill candidate. Multi-hole,
    candidate-scoped Open3D fill is intentionally deferred.
    """

    if mesh is None:
        raise ValueError("mesh must not be None")
    if candidate is None:
        raise ValueError("candidate must not be None")

    candidates = tuple(all_candidates) if all_candidates is not None else (candidate,)
    if len(candidates) != 1:
        raise ValueError(
            "Open3D hole fill preview currently requires exactly one hole candidate; "
            f"got {len(candidates)}."
        )

    # Validate the requested candidate against the current mesh before invoking
    # the whole-mesh Open3D fill. This keeps errors candidate-oriented even
    # though Open3D itself fills all holes.
    _validate_candidate_vertices(mesh, _ordered_candidate_vertices(candidate))

    try:
        preview = fill_holes_with_open3d_tensor(mesh)
    except Exception as exc:
        raise ValueError(f"Open3D hole fill preview failed: {exc}") from exc

    if len(preview.faces) <= len(mesh.faces):
        raise ValueError("Open3D hole fill preview did not add any faces.")

    return preview


def _build_builtin_hole_fill_patch_geometry(
    mesh: trimesh.Trimesh,
    candidate: HoleCandidate,
    *,
    source_faces: np.ndarray,
    base_vertices: np.ndarray,
    next_vertex_id: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], dict[str, object]]:
    """Return one candidate patch as global face ids plus new vertices.

    This helper is intentionally patch-only.  It does not append the patch to
    the source mesh.  Both single-preview and batch-preview paths use it so fan
    orientation, triangle-hole handling, and metadata stay identical.
    """

    boundary_vertices = _validate_candidate_vertices(
        mesh,
        _ordered_candidate_vertices(candidate),
    )

    if len(boundary_vertices) == 3:
        new_vertices = np.empty((0, 3), dtype=float)
        patch_faces = np.asarray([boundary_vertices], dtype=np.int64)
        new_vertex_ids: tuple[int, ...] = ()
    else:
        boundary_points = base_vertices[np.asarray(boundary_vertices, dtype=np.int64)]
        centroid = np.mean(boundary_points, axis=0)
        center_index = int(next_vertex_id)

        fan_faces: list[list[int]] = []
        count = len(boundary_vertices)
        for i in range(count):
            a = int(boundary_vertices[i])
            b = int(boundary_vertices[(i + 1) % count])
            if a == b:
                continue
            fan_faces.append([center_index, a, b])

        if not fan_faces:
            raise ValueError("Hole fill preview could not build fan faces for candidate.")

        new_vertices = centroid.reshape(1, 3)
        patch_faces = np.asarray(fan_faces, dtype=np.int64)
        new_vertex_ids = (center_index,)

    patch_faces = _orient_patch_faces_against_source_mesh(source_faces, patch_faces)
    opposite_votes, same_votes = _patch_boundary_orientation_votes(
        source_faces,
        patch_faces,
    )

    summary = {
        "boundary_vertices": int(len(boundary_vertices)),
        "boundary_edges": int(len(getattr(candidate, "boundary_edges", ()) or ())),
        "patch_face_count": int(len(patch_faces)),
        "new_vertex_ids": new_vertex_ids,
        "area_hint": getattr(candidate, "area_hint", None),
        "perimeter": getattr(candidate, "perimeter", None),
        "fill_priority": getattr(candidate, "fill_priority", None),
        "orientation": {
            "policy": "source_boundary_directed_edge_votes",
            "opposite_votes": int(opposite_votes),
            "same_votes": int(same_votes),
        },
    }

    return new_vertices, patch_faces, new_vertex_ids, summary


def _build_hole_fill_preview_mesh_builtin(
    mesh: trimesh.Trimesh,
    candidate: HoleCandidate,
    *,
    method: str | None = "fan",
    all_candidates: Iterable[HoleCandidate] | None = None,
) -> trimesh.Trimesh:
    """
    Build a preview mesh with one candidate hole capped by fan triangles.

    This is a non-destructive Phase 2E helper. The input mesh is not modified.

    Method support:
    - fan / fan_triangulate / center_fan

    Behaviour:
    - triangular boundary: add one triangle using existing boundary vertices
    - n-gon boundary: add a centroid vertex and triangulate fan faces around it

    The result is intended for viewport preview first. Commit remains a separate
    safety decision at the MeshProcessor/MainWindow layer.
    """

    if mesh is None:
        raise ValueError("mesh must not be None")
    if candidate is None:
        raise ValueError("candidate must not be None")

    normalized_method = _normalize_method(method)
    if normalized_method == "open3d":
        return build_open3d_hole_fill_preview_mesh(
            mesh,
            candidate,
            all_candidates=all_candidates,
        )

    if normalized_method not in SUPPORTED_HOLE_FILL_METHODS:
        raise ValueError(
            f"Built-in fan preview does not support method: {method!r}."
        )

    base = mesh.copy()
    vertices = np.asarray(base.vertices, dtype=float)
    faces = np.asarray(base.faces, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh must contain 3D vertices")
    if faces.ndim != 2 or faces.shape[1] < 3:
        raise ValueError("mesh must contain polygonal faces")

    source_faces = faces[:, :3].astype(np.int64, copy=False)
    patch_vertices, patch_faces, patch_new_vertex_ids, patch_summary = (
        _build_builtin_hole_fill_patch_geometry(
            base,
            candidate,
            source_faces=source_faces,
            base_vertices=vertices,
            next_vertex_id=int(len(vertices)),
        )
    )

    if len(patch_vertices):
        new_vertices = np.vstack([vertices, patch_vertices])
    else:
        new_vertices = vertices.copy()

    new_faces = np.vstack([source_faces, patch_faces])
    new_face_ids = tuple(range(int(len(source_faces)), int(len(new_faces))))

    preview = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    preview.metadata.update(
        {
            "method": normalized_method,
            "backend": "trimesh_fan",
            "new_face_ids": new_face_ids,
            "new_vertex_ids": patch_new_vertex_ids,
            "hole_fill_patch_orientation": patch_summary["orientation"],
        }
    )
    return preview


def _build_batch_adaptive_surface_v2_preview_mesh(
    mesh: trimesh.Trimesh,
    candidates: Iterable[HoleCandidate],
    *,
    method: str = ADAPTIVE_SURFACE_V2_METHOD,
) -> trimesh.Trimesh:
    """Build a sequential Adaptive Surface Fill v2 batch preview.

    Adaptive Surface Fill v2 can add vertices and alter local topology.  Batch
    mode therefore must not reuse stale HoleCandidate objects after the first
    fill.  This helper keeps the original mesh read-only, fills one current
    highest-priority candidate at a time, and re-detects candidates on the
    working preview mesh after every successful step.
    """

    if mesh is None:
        raise ValueError("mesh must not be None")

    from far_mesh.core.selection_topology import find_boundary_edges, find_hole_candidates

    initial_candidates = tuple(candidates) if candidates is not None else ()
    if not initial_candidates:
        raise ValueError("Adaptive Surface Fill v2 batch preview requires at least one candidate.")

    working = mesh.copy()
    original_face_count = int(len(np.asarray(mesh.faces, dtype=np.int64)))
    original_vertex_count = int(len(np.asarray(mesh.vertices, dtype=float)))

    step_summaries: list[dict[str, object]] = []
    attempted_count = 0
    max_steps = max(1, len(initial_candidates))

    for step_index in range(max_steps):
        current_candidates = tuple(find_hole_candidates(working))
        if not current_candidates:
            break

        candidate = current_candidates[0]
        before_face_count = int(len(working.faces))
        before_vertex_count = int(len(working.vertices))
        before_boundary_edge_count = int(len(find_boundary_edges(working)))
        before_candidate_count = int(len(current_candidates))
        attempted_count += 1

        try:
            next_mesh = build_hole_fill_preview_mesh(
                working,
                candidate,
                method=method,
                all_candidates=current_candidates,
            )
        except Exception as exc:
            raise ValueError(
                "Adaptive Surface Fill v2 batch preview failed while building "
                f"candidate {step_index}: {exc}"
            ) from exc

        if not isinstance(next_mesh, trimesh.Trimesh):
            raise TypeError(
                "Adaptive Surface Fill v2 batch step did not return a trimesh.Trimesh preview mesh."
            )

        after_face_count = int(len(next_mesh.faces))
        after_vertex_count = int(len(next_mesh.vertices))
        after_boundary_edge_count = int(len(find_boundary_edges(next_mesh)))
        after_candidates = tuple(find_hole_candidates(next_mesh))
        after_candidate_count = int(len(after_candidates))

        face_delta = after_face_count - before_face_count
        vertex_delta = after_vertex_count - before_vertex_count
        boundary_edge_delta = after_boundary_edge_count - before_boundary_edge_count
        candidate_delta = after_candidate_count - before_candidate_count

        if face_delta <= 0 and boundary_edge_delta >= 0 and candidate_delta >= 0:
            raise ValueError(
                "Adaptive Surface Fill v2 batch preview made no safe progress at "
                f"candidate {step_index}: faces {before_face_count}->{after_face_count}, "
                f"boundary edges {before_boundary_edge_count}->{after_boundary_edge_count}, "
                f"candidates {before_candidate_count}->{after_candidate_count}."
            )

        metadata = dict(getattr(next_mesh, "metadata", {}) or {})
        commit_allowed = bool(metadata.get("commit_allowed", True))
        blocking_reasons = tuple(metadata.get("commit_blocking_reasons", ()) or ())
        commit_warnings = tuple(metadata.get("commit_warnings", ()) or ())

        if not commit_allowed:
            raise ValueError(
                "Adaptive Surface Fill v2 batch preview candidate "
                f"{step_index} is not commit-eligible: "
                + "; ".join(str(reason) for reason in blocking_reasons)
            )

        step_summaries.append(
            {
                "candidate_index": int(step_index),
                "method": ADAPTIVE_SURFACE_V2_METHOD,
                "backend": str(metadata.get("backend") or "adaptive_surface_v2"),
                "faces_before": int(before_face_count),
                "faces_after": int(after_face_count),
                "vertices_before": int(before_vertex_count),
                "vertices_after": int(after_vertex_count),
                "face_delta": int(face_delta),
                "vertex_delta": int(vertex_delta),
                "boundary_edges_before": int(before_boundary_edge_count),
                "boundary_edges_after": int(after_boundary_edge_count),
                "candidate_count_before": int(before_candidate_count),
                "candidate_count_after": int(after_candidate_count),
                "boundary_vertices": int(len(getattr(candidate, "boundary_vertices", ()) or ())),
                "boundary_edges": int(len(getattr(candidate, "boundary_edges", ()) or ())),
                "area_hint": getattr(candidate, "area_hint", None),
                "perimeter": getattr(candidate, "perimeter", None),
                "fill_priority": getattr(candidate, "fill_priority", None),
                "commit_warnings": tuple(str(item) for item in commit_warnings if str(item)),
            }
        )
        working = next_mesh

    if not step_summaries:
        raise ValueError("Adaptive Surface Fill v2 batch preview did not fill any candidates.")

    final_candidates = tuple(find_hole_candidates(working))
    preview = working.copy()

    preview_face_count = int(len(preview.faces))
    preview_vertex_count = int(len(preview.vertices))
    new_face_ids = tuple(range(original_face_count, preview_face_count))
    new_vertex_ids = tuple(range(original_vertex_count, preview_vertex_count))

    if not new_face_ids:
        raise ValueError("Adaptive Surface Fill v2 batch preview did not append any patch faces.")

    metadata = dict(getattr(preview, "metadata", {}) or {})
    metadata.update(
        {
            "method": ADAPTIVE_SURFACE_V2_METHOD,
            "public_method": ADAPTIVE_SURFACE_V2_METHOD,
            "backend": "adaptive_surface_v2_batch_sequential",
            "batch_mode": True,
            "batch_policy": "sequential_redetect_after_each_fill",
            "candidate_count": int(len(initial_candidates)),
            "attempted_candidate_count": int(attempted_count),
            "successful_candidate_count": int(len(step_summaries)),
            "failed_candidate_count": 0,
            "remaining_candidate_count": int(len(final_candidates)),
            "batch_original_face_count": int(original_face_count),
            "batch_original_vertex_count": int(original_vertex_count),
            "batch_patch_face_count": int(len(new_face_ids)),
            "batch_patch_vertex_count": int(max(0, preview_vertex_count - original_vertex_count)),
            "new_face_ids": tuple(int(v) for v in new_face_ids),
            "new_vertex_ids": tuple(int(v) for v in new_vertex_ids),
            "candidate_summaries": tuple(step_summaries),
            "commit_allowed": True,
            "commit_blocking_reasons": (),
            "commit_warnings": (
                f"Adaptive Surface Fill v2 batch filled {len(step_summaries)} candidate(s) sequentially.",
            ),
        }
    )
    preview.metadata.update(metadata)
    return preview


def build_batch_hole_fill_preview_mesh(
    mesh: trimesh.Trimesh,
    candidates: Iterable[HoleCandidate],
    *,
    method: str = "fan",
) -> trimesh.Trimesh:
    """Build one non-destructive preview mesh that caps every candidate.

    Built-in fan methods are batched in one deterministic pass. Adaptive
    Surface Fill v2 is batched conservatively by sequentially filling one
    current candidate at a time and re-detecting remaining holes after each
    step, because v2 can add vertices and change local topology.
    """

    if mesh is None:
        raise ValueError("mesh must not be None")

    candidate_list = tuple(candidates) if candidates is not None else ()
    if not candidate_list:
        raise ValueError("Batch hole fill preview requires at least one candidate.")

    method_key = _normalize_method_key(method)
    if method_key == ADAPTIVE_SURFACE_V2_METHOD:
        return _build_batch_adaptive_surface_v2_preview_mesh(
            mesh,
            candidate_list,
            method=method_key,
        )

    if method_key not in SUPPORTED_HOLE_FILL_METHODS:
        raise ValueError(
            "Batch hole fill preview currently supports fan, fan_triangulate, "
            "center_fan, and adaptive_surface_v2 methods. "
            f"Got: {method!r}."
        )

    base = mesh.copy()
    vertices = np.asarray(base.vertices, dtype=float)
    faces = np.asarray(base.faces, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh must contain 3D vertices")
    if faces.ndim != 2 or faces.shape[1] < 3:
        raise ValueError("mesh must contain polygonal faces")

    source_faces = faces[:, :3].astype(np.int64, copy=False)
    patch_face_arrays: list[np.ndarray] = []
    patch_vertex_arrays: list[np.ndarray] = []
    new_vertex_ids: list[int] = []
    new_face_ids: list[int] = []
    candidate_summaries: list[dict[str, object]] = []
    next_vertex_id = int(len(vertices))
    next_face_id = int(len(source_faces))
    orientation_opposite_votes = 0
    orientation_same_votes = 0

    for candidate_index, candidate in enumerate(candidate_list):
        if candidate is None:
            raise ValueError(
                f"Batch hole fill candidate {candidate_index} is None."
            )

        try:
            patch_vertices, patch_faces, candidate_new_vertex_ids, summary = (
                _build_builtin_hole_fill_patch_geometry(
                    base,
                    candidate,
                    source_faces=source_faces,
                    base_vertices=vertices,
                    next_vertex_id=next_vertex_id,
                )
            )
        except Exception as exc:
            raise ValueError(
                f"Batch hole fill failed while building candidate {candidate_index}: {exc}"
            ) from exc

        patch_face_count = int(len(patch_faces))
        candidate_new_face_ids = tuple(
            range(next_face_id, next_face_id + patch_face_count)
        )

        summary = dict(summary)
        summary.update(
            {
                "candidate_index": int(candidate_index),
                "new_face_ids": candidate_new_face_ids,
                "new_vertex_ids": tuple(candidate_new_vertex_ids),
            }
        )

        orientation = summary.get("orientation")
        if isinstance(orientation, dict):
            orientation_opposite_votes += int(orientation.get("opposite_votes", 0) or 0)
            orientation_same_votes += int(orientation.get("same_votes", 0) or 0)

        if len(patch_vertices):
            patch_vertex_arrays.append(patch_vertices)
        if patch_face_count:
            patch_face_arrays.append(patch_faces)

        new_vertex_ids.extend(int(v) for v in candidate_new_vertex_ids)
        new_face_ids.extend(int(v) for v in candidate_new_face_ids)
        candidate_summaries.append(summary)

        next_vertex_id += int(len(patch_vertices))
        next_face_id += patch_face_count

    if not patch_face_arrays:
        raise ValueError("Batch hole fill preview did not build any patch faces.")

    if patch_vertex_arrays:
        new_vertices = np.vstack([vertices, *patch_vertex_arrays])
    else:
        new_vertices = vertices.copy()

    patch_faces_all = np.vstack(patch_face_arrays)
    new_faces = np.vstack([source_faces, patch_faces_all])

    preview = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    preview.metadata.update(
        {
            "method": method_key,
            "backend": "trimesh_fan_batch",
            "batch_mode": True,
            "candidate_count": int(len(candidate_list)),
            "successful_candidate_count": int(len(candidate_summaries)),
            "failed_candidate_count": 0,
            "new_face_ids": tuple(new_face_ids),
            "new_vertex_ids": tuple(new_vertex_ids),
            "batch_patch_face_count": int(len(patch_faces_all)),
            "batch_original_face_count": int(len(source_faces)),
            "candidate_summaries": tuple(candidate_summaries),
            "hole_fill_batch_patch_orientation": {
                "policy": "source_boundary_directed_edge_votes",
                "opposite_votes": int(orientation_opposite_votes),
                "same_votes": int(orientation_same_votes),
            },
        }
    )
    return preview


def _adaptive_preview_metadata_from_result(
    result: dict[str, object],
    preview_mesh: trimesh.Trimesh,
) -> dict[str, object]:
    metadata: dict[str, object] = {}

    raw_mesh_metadata = getattr(preview_mesh, "metadata", None)
    if isinstance(raw_mesh_metadata, dict):
        metadata.update(raw_mesh_metadata)
        for nested_key in (
            "hole_fill_preview",
            "preview_builder_metadata",
            "adaptive_diagnostics",
        ):
            nested = raw_mesh_metadata.get(nested_key)
            if isinstance(nested, dict):
                metadata.update(nested)

    metadata.update(result)
    return metadata


def _adaptive_int(value: object, default: int = 0) -> int:
    try:
        if value in (None, "", "-", {}, [], ()):
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _adaptive_float(value: object, default: float | None = None) -> float | None:
    try:
        if value in (None, "", "-", {}, [], ()):
            return default
        return float(value)
    except Exception:
        return default


def _adaptive_mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
            if isinstance(result, dict):
                return dict(result)
        except Exception:
            return {}

    return {}


def _adaptive_nested_mapping(
    mapping: dict[str, object],
    key: str,
    nested_key: str,
) -> dict[str, object]:
    parent = _adaptive_mapping(mapping.get(key))
    return _adaptive_mapping(parent.get(nested_key))


def _adaptive_quality_after(metadata: dict[str, object]) -> dict[str, object]:
    quality = _adaptive_mapping(metadata.get("quality_after"))
    if quality:
        return quality

    quality = _adaptive_nested_mapping(metadata, "relaxation", "quality_after")
    if quality:
        return quality

    quality = _adaptive_nested_mapping(metadata, "biharmonic_fairing", "quality_after")
    if quality:
        return quality

    return {}


def _adaptive_support_context(metadata: dict[str, object]) -> dict[str, object]:
    support = _adaptive_mapping(metadata.get("support_context"))
    if support:
        return support

    surface_decision = _adaptive_mapping(metadata.get("surface_context_decision"))
    support = _adaptive_mapping(surface_decision.get("support_context"))
    if support:
        return support

    return {}


def _adaptive_seam_decision(metadata: dict[str, object]) -> dict[str, object]:
    seam_report = _adaptive_mapping(metadata.get("seam_constraint_report"))
    seam_decision = _adaptive_mapping(metadata.get("seam_recovery_decision"))
    topology_after = _adaptive_mapping(metadata.get("topology_after"))

    missing = _adaptive_int(
        seam_decision.get(
            "missing_seam_edge_count",
            seam_report.get(
                "missing_seam_edge_count",
                topology_after.get("missing_seam_edge_count", 0),
            ),
        )
    )
    overused = _adaptive_int(
        seam_decision.get(
            "overused_seam_edge_count",
            seam_report.get("overused_seam_edge_count", 0),
        )
    )
    weak = _adaptive_int(
        seam_decision.get(
            "weak_seam_edge_count",
            seam_report.get("weak_seam_edge_count", 0),
        )
    )
    coverage = _adaptive_float(
        seam_decision.get(
            "seam_coverage_ratio",
            seam_report.get(
                "seam_coverage_ratio",
                topology_after.get("seam_coverage_ratio", None),
            ),
        )
    )
    recovery_required = bool(
        seam_decision.get(
            "recovery_required",
            bool(missing or overused or (coverage is not None and coverage < 0.999)),
        )
    )

    needs_fallback = bool(
        recovery_required
        or missing > 0
        or overused > 0
        or (coverage is not None and coverage < 0.999)
    )

    return {
        "missing_seam_edge_count": missing,
        "overused_seam_edge_count": overused,
        "weak_seam_edge_count": weak,
        "seam_coverage_ratio": coverage,
        "recovery_required": recovery_required,
        "needs_fallback": needs_fallback,
        "strategy": str(seam_decision.get("strategy", "unknown")),
        "problem_edge_runs": seam_decision.get("problem_edge_runs", ()),
    }


def _adaptive_bool(value: object, default: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, "", "-", {}, [], ()):
        return default

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "ok", "valid", "enabled"}:
        return True
    if text in {"0", "false", "no", "n", "invalid", "disabled"}:
        return False
    return default


def _adaptive_reason_tuple(value: object) -> tuple[str, ...]:
    if value in (None, "", "-", {}, [], ()):
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    try:
        return tuple(str(item) for item in value if str(item))
    except Exception:
        text = str(value)
        return (text,) if text else ()


def _adaptive_first_mapping_value(
    mapping: dict[str, object],
    *keys: str,
    default: object = None,
) -> object:
    for key in keys:
        if key in mapping and mapping.get(key) not in (None, "", "-", {}, [], ()):
            return mapping.get(key)
    return default


def _adaptive_feature_context_decision(
    metadata: dict[str, object],
    *,
    support_spread: float | None,
) -> dict[str, object]:
    """Classify the support context for feature-aware G1 behavior."""

    surface_decision = _adaptive_mapping(metadata.get("surface_context_decision"))
    support_context = _adaptive_support_context(metadata)
    density_decision = _adaptive_mapping(
        metadata.get("local_density_budget_decision")
        or metadata.get("preseed_local_density_budget_decision")
        or metadata.get("local_density_refinement_budget")
        or metadata.get("preseed_local_density_refinement_budget")
    )
    boundary_diagnostic = _adaptive_mapping(metadata.get("boundary_loop_diagnostic"))

    recommended_action = str(
        _adaptive_first_mapping_value(
            surface_decision,
            "recommended_action",
            "action",
            "decision",
            default="",
        )
        or ""
    )
    density_mode = str(
        _adaptive_first_mapping_value(
            density_decision,
            "recommended_density_mode",
            "density_mode",
            "mode",
            default="",
        )
        or ""
    )

    smooth_valid = _adaptive_bool(
        _adaptive_first_mapping_value(
            surface_decision,
            "smooth_surface_context_valid",
            "surface_guidance_enabled",
            default=None,
        ),
        default=None,
    )

    boundary_count = _adaptive_int(
        _adaptive_first_mapping_value(
            density_decision,
            "boundary_vertex_count",
            default=_adaptive_first_mapping_value(
                boundary_diagnostic,
                "boundary_vertex_count",
                "boundary_vertices",
                default=metadata.get("boundary_vertices", 0),
            ),
        ),
        0,
    )

    reasons: list[str] = []
    for source in (
        surface_decision.get("reasons"),
        density_decision.get("reasons"),
        support_context.get("reasons"),
    ):
        reasons.extend(_adaptive_reason_tuple(source))

    lower_blob = " ".join(
        [
            recommended_action,
            density_mode,
            " ".join(reasons),
        ]
    ).lower()

    feature_like_hint = any(
        token in lower_blob
        for token in (
            "feature",
            "sharp",
            "rim",
            "bore",
            "high_normal_spread",
            "disable_mls",
            "disable_surface",
            "no_surface",
            "preserve",
        )
    )
    minimal_hint = any(
        token in lower_blob
        for token in (
            "minimal",
            "tiny",
            "zero_interior",
            "no_interior",
            "preserve_minimal_seed",
        )
    )

    if smooth_valid is False:
        feature_like_hint = True
        reasons.append("smooth surface context is not valid")

    if support_spread is not None and support_spread >= 75.0:
        feature_like_hint = True
        reasons.append(f"support normal spread is feature-like ({support_spread:.3g}°)")
    elif support_spread is not None and support_spread >= 45.0:
        reasons.append(f"support normal spread is elevated ({support_spread:.3g}°)")

    tiny_loop = bool(boundary_count > 0 and boundary_count <= 6)

    if feature_like_hint and (tiny_loop or minimal_hint):
        context_kind = "tiny_feature_like"
        preservation_mode = "preserve_minimal_topology"
        g1_policy = "feature_preserve_minimal_topology"
    elif feature_like_hint:
        context_kind = "feature_like"
        preservation_mode = "preserve_feature_no_surface_pull"
        g1_policy = "feature_preserve_no_surface_pull"
    elif support_spread is not None and support_spread >= 45.0:
        context_kind = "mixed_curved"
        preservation_mode = "conservative_no_surface_pull"
        g1_policy = "conservative_no_surface_pull"
    else:
        context_kind = "smooth"
        preservation_mode = "allow_surface_guidance"
        g1_policy = "smooth_surface_guided_relaxation"

    return {
        "feature_context_kind": context_kind,
        "feature_preservation_mode": preservation_mode,
        "feature_policy_reasons": tuple(dict.fromkeys(str(reason) for reason in reasons if str(reason))),
        "feature_recommended_action": recommended_action or "-",
        "feature_density_mode": density_mode or "-",
        "feature_smooth_context_valid": smooth_valid,
        "feature_boundary_vertex_count": boundary_count if boundary_count else "-",
        "feature_support_normal_spread_degrees": support_spread,
        "g1_policy": g1_policy,
        "try_conservative": g1_policy != "smooth_surface_guided_relaxation",
    }


def _adaptive_g1_decision(metadata: dict[str, object]) -> dict[str, object]:
    quality = _adaptive_quality_after(metadata)
    support = _adaptive_support_context(metadata)

    mean_dev = _adaptive_float(
        quality.get("boundary_normal_mean_deviation_degrees"),
        None,
    )
    max_dev = _adaptive_float(
        quality.get("boundary_normal_max_deviation_degrees"),
        None,
    )
    support_spread = _adaptive_float(
        support.get("normal_spread_degrees"),
        None,
    )

    feature = _adaptive_feature_context_decision(
        metadata,
        support_spread=support_spread,
    )

    reasons = list(_adaptive_reason_tuple(feature.get("feature_policy_reasons")))
    try_conservative = bool(feature.get("try_conservative"))
    policy = str(feature.get("g1_policy") or "smooth_surface_guided_relaxation")

    if mean_dev is not None and mean_dev > 12.0:
        try_conservative = True
        reasons.append(f"boundary normal mean deviation is elevated ({mean_dev:.3g}°)")
    if max_dev is not None and max_dev > 30.0:
        try_conservative = True
        reasons.append(f"boundary normal max deviation is elevated ({max_dev:.3g}°)")

    return {
        "g1_policy": policy,
        "try_conservative": bool(try_conservative),
        "boundary_normal_mean_deviation_degrees": mean_dev,
        "boundary_normal_max_deviation_degrees": max_dev,
        "support_normal_spread_degrees": support_spread,
        "reasons": tuple(dict.fromkeys(str(reason) for reason in reasons if str(reason))),
        "feature_context_kind": feature.get("feature_context_kind"),
        "feature_preservation_mode": feature.get("feature_preservation_mode"),
        "feature_policy_reasons": feature.get("feature_policy_reasons", ()),
        "feature_recommended_action": feature.get("feature_recommended_action", "-"),
        "feature_density_mode": feature.get("feature_density_mode", "-"),
        "feature_smooth_context_valid": feature.get("feature_smooth_context_valid"),
        "feature_boundary_vertex_count": feature.get("feature_boundary_vertex_count", "-"),
        "feature_support_normal_spread_degrees": feature.get("feature_support_normal_spread_degrees"),
    }


def _adaptive_score_breakdown(metadata: dict[str, object]) -> dict[str, float]:
    """Return named score components for adaptive candidate comparison.

    Lower is better. The score is intentionally conservative:
    - seam/topology failures dominate everything
    - degenerate faces dominate visual improvements
    - triangle quality matters before G1 cosmetics
    - G1 boundary-normal improvement matters
    - movement/displacement is penalized to avoid over-relaxation
    """

    seam = _adaptive_seam_decision(metadata)
    quality = _adaptive_quality_after(metadata)
    relaxation = _adaptive_mapping(metadata.get("relaxation"))
    g1 = _adaptive_g1_decision(metadata)

    coverage = _adaptive_float(seam.get("seam_coverage_ratio"), 0.0)
    if coverage is None:
        coverage = 0.0

    missing = float(_adaptive_int(seam.get("missing_seam_edge_count"), 0))
    overused = float(_adaptive_int(seam.get("overused_seam_edge_count"), 0))
    weak = float(_adaptive_int(seam.get("weak_seam_edge_count"), 0))

    degenerate = float(_adaptive_int(quality.get("degenerate_face_count"), 0))

    min_angle = _adaptive_float(quality.get("min_triangle_angle_degrees"), 0.0)
    if min_angle is None:
        min_angle = 0.0

    aspect = _adaptive_float(quality.get("max_triangle_aspect_ratio"), 999.0)
    if aspect is None:
        aspect = 999.0

    mean_dev = _adaptive_float(g1.get("boundary_normal_mean_deviation_degrees"), 0.0)
    if mean_dev is None:
        mean_dev = 0.0

    max_dev = _adaptive_float(g1.get("boundary_normal_max_deviation_degrees"), 0.0)
    if max_dev is None:
        max_dev = 0.0

    support_spread = _adaptive_float(g1.get("support_normal_spread_degrees"), 0.0)
    if support_spread is None:
        support_spread = 0.0

    max_move = _adaptive_float(
        relaxation.get("max_displacement", metadata.get("relaxed_max_displacement")),
        0.0,
    )
    if max_move is None:
        max_move = 0.0

    mean_move = _adaptive_float(
        relaxation.get("mean_displacement", metadata.get("relaxed_mean_displacement")),
        0.0,
    )
    if mean_move is None:
        mean_move = 0.0

    # Support spread is not directly a blocker. It only slightly biases against
    # forcing smoothness in mixed/feature-like contexts. Boundary normal quality
    # and movement carry the real G1 decision weight.
    support_penalty = 0.0
    if support_spread >= 75.0:
        support_penalty = 2.0
    elif support_spread >= 45.0:
        support_penalty = 1.0

    return {
        "seam_missing": missing * 10000.0,
        "seam_overused": overused * 10000.0,
        "seam_weak": weak * 1000.0,
        "seam_coverage": max(0.0, 1.0 - float(coverage)) * 5000.0,
        "degenerate_faces": degenerate * 10000.0,
        "min_angle": max(0.0, 10.0 - float(min_angle)) * 60.0,
        "aspect_ratio": max(0.0, float(aspect) - 10.0) * 6.0,
        "g1_boundary_max": float(max_dev) * 1.25,
        "g1_boundary_mean": float(mean_dev) * 0.75,
        "g1_support_context": support_penalty,
        "relaxation_max_displacement": float(max_move) * 2.0,
        "relaxation_mean_displacement": float(mean_move) * 1.0,
    }


def _adaptive_score_total(metadata: dict[str, object]) -> float:
    return float(sum(_adaptive_score_breakdown(metadata).values()))


def _adaptive_quality_score(metadata: dict[str, object]) -> tuple[float, ...]:
    """Comparison tuple for adaptive variants.

    Lower is better. Keep tuple ordering stable so seam/topology and hard
    quality failures dominate before softer G1/displacement preferences.
    """

    breakdown = _adaptive_score_breakdown(metadata)
    return (
        breakdown["seam_missing"],
        breakdown["seam_overused"],
        breakdown["seam_weak"],
        breakdown["seam_coverage"],
        breakdown["degenerate_faces"],
        breakdown["min_angle"],
        breakdown["aspect_ratio"],
        breakdown["g1_boundary_max"],
        breakdown["g1_boundary_mean"],
        breakdown["g1_support_context"],
        breakdown["relaxation_max_displacement"],
        breakdown["relaxation_mean_displacement"],
    )


def _adaptive_seam_score(decision: dict[str, object]) -> tuple[int, int, int, float]:
    missing = _adaptive_int(decision.get("missing_seam_edge_count"))
    overused = _adaptive_int(decision.get("overused_seam_edge_count"))
    weak = _adaptive_int(decision.get("weak_seam_edge_count"))
    coverage = _adaptive_float(decision.get("seam_coverage_ratio"), 0.0)
    if coverage is None:
        coverage = 0.0
    return (missing, overused, weak, -coverage)


def _adaptive_surface_builder_kwargs() -> dict[str, object]:
    return {
        "rings": 2,
        "relaxation_iterations": 8,
        "relaxation_strength": 0.20,
        "surface_weight": 0.20,
        "support_rings": 2,
        "smooth_dot_threshold": 0.50,
        # H-ADAPT-5C4: adaptive_surface must not silently keep a
        # biharmonic fairing result that moves farther than the adaptive
        # curvature-fairing proposal allows.
        "biharmonic_max_movement_ratio": 0.03,
    }


def _adaptive_conservative_surface_kwargs(
    decision: dict[str, object],
) -> dict[str, object]:
    policy = str(decision.get("g1_policy") or "")

    kwargs = _adaptive_surface_builder_kwargs()

    if policy == "feature_preserve_minimal_topology":
        kwargs.update(
            {
                "relaxation_iterations": 1,
                "relaxation_strength": 0.03,
                "surface_weight": 0.0,
            }
        )
    elif policy == "feature_preserve_no_surface_pull":
        kwargs.update(
            {
                "relaxation_iterations": 2,
                "relaxation_strength": 0.06,
                "surface_weight": 0.0,
            }
        )
    elif policy == "conservative_no_surface_pull":
        kwargs.update(
            {
                "relaxation_iterations": 4,
                "relaxation_strength": 0.10,
                "surface_weight": 0.0,
            }
        )
    else:
        kwargs.update(
            {
                "relaxation_iterations": 6,
                "relaxation_strength": 0.14,
                "surface_weight": 0.08,
            }
        )

    return kwargs


def _call_hole_surface_builder(
    builder: object,
    mesh: trimesh.Trimesh,
    candidate: HoleCandidate,
    **kwargs: object,
) -> dict[str, object]:
    import inspect

    try:
        signature = inspect.signature(builder)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_kwargs:
            filtered_kwargs = dict(kwargs)
        else:
            filtered_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }
    except Exception:
        filtered_kwargs = dict(kwargs)

    result = builder(mesh, candidate, **filtered_kwargs)
    if not isinstance(result, dict):
        raise ValueError("Adaptive surface builder did not return a result dictionary.")

    preview_mesh = result.get("preview_mesh")
    if not isinstance(preview_mesh, trimesh.Trimesh):
        raise ValueError("Adaptive surface builder did not return preview_mesh as trimesh.Trimesh.")

    return result


def _adaptive_candidate_boundary_ids(candidate: HoleCandidate) -> tuple[int, ...]:
    raw = tuple(int(v) for v in getattr(candidate, "boundary_vertices", ()) or ())
    if raw:
        return tuple(dict.fromkeys(v for v in raw if v >= 0))

    loop = getattr(candidate, "loop", None)
    raw_loop = tuple(int(v) for v in getattr(loop, "vertices", ()) or ())
    if len(raw_loop) >= 2 and raw_loop[0] == raw_loop[-1]:
        raw_loop = raw_loop[:-1]

    return tuple(dict.fromkeys(v for v in raw_loop if v >= 0))


def _adaptive_vertex_neighbors(mesh: trimesh.Trimesh) -> list[tuple[int, ...]]:
    try:
        neighbors = getattr(mesh, "vertex_neighbors")
        return [tuple(int(v) for v in values) for values in neighbors]
    except Exception:
        return [tuple() for _ in range(int(len(getattr(mesh, "vertices", ()))))]


def _adaptive_curvature_samples(
    mesh: trimesh.Trimesh,
    vertex_ids: object,
) -> tuple[float, ...]:
    """Estimate unsigned local 3D curvature from normal-angle / edge-length.

    This is intentionally lightweight and diagnostic-only. It is not a final
    differential-geometry estimator; it gives a stable local signal for the
    upcoming G2/C2 stage.
    """

    try:
        ids = tuple(
            int(v)
            for v in vertex_ids
            if int(v) >= 0 and int(v) < int(len(mesh.vertices))
        )
    except Exception:
        ids = ()

    if not ids:
        return ()

    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
        normals = np.asarray(mesh.vertex_normals, dtype=float)
    except Exception:
        return ()

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return ()
    if normals.ndim != 2 or normals.shape != vertices.shape:
        return ()

    neighbors = _adaptive_vertex_neighbors(mesh)
    samples: list[float] = []

    for vertex_id in ids:
        if vertex_id >= len(neighbors):
            continue

        normal = normals[vertex_id]
        normal_len = float(np.linalg.norm(normal))
        if normal_len <= 1.0e-12:
            continue
        normal = normal / normal_len

        local_values: list[float] = []
        for neighbor_id in neighbors[vertex_id]:
            if neighbor_id < 0 or neighbor_id >= len(vertices):
                continue

            other_normal = normals[neighbor_id]
            other_len = float(np.linalg.norm(other_normal))
            if other_len <= 1.0e-12:
                continue
            other_normal = other_normal / other_len

            edge_len = float(np.linalg.norm(vertices[neighbor_id] - vertices[vertex_id]))
            if edge_len <= 1.0e-12:
                continue

            dot = float(np.clip(np.dot(normal, other_normal), -1.0, 1.0))
            angle = float(np.arccos(dot))
            local_values.append(angle / edge_len)

        if local_values:
            samples.append(float(np.mean(local_values)))

    return tuple(samples)


def _adaptive_curvature_summary(samples: tuple[float, ...]) -> dict[str, object]:
    if not samples:
        return {
            "count": 0,
            "mean": None,
            "max": None,
            "std": None,
        }

    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "max": None,
            "std": None,
        }

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def _adaptive_mean_normal_for_ids(
    mesh: trimesh.Trimesh,
    vertex_ids: object,
) -> np.ndarray | None:
    try:
        ids = tuple(
            int(v)
            for v in vertex_ids
            if int(v) >= 0 and int(v) < int(len(mesh.vertices))
        )
    except Exception:
        ids = ()

    if not ids:
        return None

    try:
        normals = np.asarray(mesh.vertex_normals, dtype=float)
    except Exception:
        return None

    if normals.ndim != 2 or normals.shape[1] != 3:
        return None

    selected = normals[np.asarray(ids, dtype=np.int64)]
    mean = np.mean(selected, axis=0)
    length = float(np.linalg.norm(mean))
    if length <= 1.0e-12:
        return None
    return mean / length


def _adaptive_local_curvature_diagnostics(
    *,
    base_mesh: trimesh.Trimesh,
    candidate: HoleCandidate,
    preview_mesh: trimesh.Trimesh,
    metadata: dict[str, object],
) -> dict[str, object]:
    boundary_ids = _adaptive_candidate_boundary_ids(candidate)

    base_neighbors = _adaptive_vertex_neighbors(base_mesh)
    support_ids: set[int] = set(boundary_ids)
    for vertex_id in boundary_ids:
        if 0 <= vertex_id < len(base_neighbors):
            support_ids.update(int(v) for v in base_neighbors[vertex_id])

    support_ids = {
        int(v)
        for v in support_ids
        if int(v) >= 0 and int(v) < int(len(base_mesh.vertices))
    }

    new_vertex_ids = metadata.get("new_vertex_ids", ())
    try:
        patch_ids = {int(v) for v in new_vertex_ids}
    except Exception:
        patch_ids = set()

    if not patch_ids:
        base_vertex_count = int(len(base_mesh.vertices))
        patch_ids.update(
            int(v)
            for v in range(base_vertex_count, int(len(preview_mesh.vertices)))
        )

    patch_ids.update(
        int(v)
        for v in boundary_ids
        if int(v) >= 0 and int(v) < int(len(preview_mesh.vertices))
    )

    support_samples = _adaptive_curvature_samples(base_mesh, tuple(sorted(support_ids)))
    patch_samples = _adaptive_curvature_samples(preview_mesh, tuple(sorted(patch_ids)))

    support = _adaptive_curvature_summary(support_samples)
    patch = _adaptive_curvature_summary(patch_samples)

    reasons: list[str] = []

    support_mean = support.get("mean")
    support_max = support.get("max")
    support_std = support.get("std")
    patch_mean = patch.get("mean")
    patch_max = patch.get("max")
    patch_std = patch.get("std")

    if support.get("count", 0) == 0 or patch.get("count", 0) == 0:
        status = "not_reported"
        context_kind = "unknown"
        reasons.append("insufficient support or patch curvature samples")
        delta_mean = None
        delta_max = None
        relative_delta_mean = None
    else:
        support_mean_f = float(support_mean or 0.0)
        support_max_f = float(support_max or 0.0)
        support_std_f = float(support_std or 0.0)
        patch_mean_f = float(patch_mean or 0.0)
        patch_max_f = float(patch_max or 0.0)

        delta_mean = abs(patch_mean_f - support_mean_f)
        delta_max = abs(patch_max_f - support_max_f)
        relative_delta_mean = delta_mean / max(abs(support_mean_f), 1.0e-6)

        if support_std_f > max(0.25, support_mean_f * 0.75):
            context_kind = "mixed_curvature"
            reasons.append("support curvature spread is high")
        elif support_mean_f < 0.05 and support_max_f < 0.15:
            context_kind = "low_curvature"
        else:
            context_kind = "curved"

        if context_kind == "mixed_curvature":
            status = "feature_like"
        elif relative_delta_mean > 2.0 and delta_mean > 0.25:
            status = "warning"
            reasons.append(
                f"patch/support curvature mean delta is high ({delta_mean:.6g})"
            )
        elif delta_max > 1.0:
            status = "warning"
            reasons.append(
                f"patch/support curvature max delta is elevated ({delta_max:.6g})"
            )
        else:
            status = "ok"

    base_normal = _adaptive_mean_normal_for_ids(base_mesh, tuple(sorted(support_ids)))
    patch_normal = _adaptive_mean_normal_for_ids(preview_mesh, tuple(sorted(patch_ids)))

    if base_normal is None or patch_normal is None:
        sign_consistency: object = "-"
    else:
        sign_consistency = bool(float(np.dot(base_normal, patch_normal)) >= 0.0)

    return {
        "status": status,
        "context_kind": context_kind,
        "estimator": "normal_angle_over_edge_length_v1",
        "support_curvature_mean": support_mean,
        "support_curvature_max": support_max,
        "support_curvature_std": support_std,
        "patch_curvature_mean": patch_mean,
        "patch_curvature_max": patch_max,
        "patch_curvature_std": patch_std,
        "curvature_delta_mean": delta_mean,
        "curvature_delta_max": delta_max,
        "curvature_relative_delta_mean": relative_delta_mean,
        "curvature_sign_consistency": sign_consistency,
        "support_sample_count": support.get("count", 0),
        "patch_sample_count": patch.get("count", 0),
        "reasons": tuple(dict.fromkeys(reason for reason in reasons if reason)),
    }


def _adaptive_bounded_curvature_fairing_proposal(
    *,
    metadata: dict[str, object],
    curvature_diagnostics: dict[str, object],
    g1_decision: dict[str, object],
) -> dict[str, object]:
    """Return a conservative curvature-fairing proposal.

    H-ADAPT-5C1 is diagnostics/proposal only. It does not alter geometry.
    """

    reasons: list[str] = []

    curvature_status = str(curvature_diagnostics.get("status") or "not_reported")
    curvature_context = str(curvature_diagnostics.get("context_kind") or "unknown")

    delta_mean = _adaptive_float(curvature_diagnostics.get("curvature_delta_mean"), None)
    delta_max = _adaptive_float(curvature_diagnostics.get("curvature_delta_max"), None)
    relative_delta = _adaptive_float(
        curvature_diagnostics.get("curvature_relative_delta_mean"),
        None,
    )
    sign_consistency = curvature_diagnostics.get("curvature_sign_consistency")

    feature_context = str(g1_decision.get("feature_context_kind") or "")
    g1_policy = str(g1_decision.get("g1_policy") or "")

    quality = _adaptive_quality_after(metadata)
    degenerate_faces = _adaptive_int(quality.get("degenerate_face_count"), 0)

    seam = _adaptive_seam_decision(metadata)
    seam_ok = not bool(seam.get("needs_fallback"))

    if not seam_ok:
        reasons.append("seam diagnostics are not clean")
    if degenerate_faces > 0:
        reasons.append(f"quality has degenerate faces ({degenerate_faces})")
    if sign_consistency is False:
        reasons.append("curvature sign consistency is false")
    if curvature_status in {"not_reported", "unknown"}:
        reasons.append("curvature diagnostics are not reported")
    if curvature_status == "feature_like" or curvature_context == "mixed_curvature":
        reasons.append("mixed/feature-like curvature context should not force smooth G2")
    if feature_context in {"feature_like", "tiny_feature_like"}:
        reasons.append(f"feature context is {feature_context}; preserve feature intent")
    if g1_policy in {"feature_preserve_no_surface_pull", "feature_preserve_minimal_topology"}:
        reasons.append(f"G1 policy is {g1_policy}; do not force curvature fairing")

    eligible = (
        seam_ok
        and degenerate_faces == 0
        and sign_consistency is not False
        and curvature_status not in {"not_reported", "unknown", "feature_like"}
        and curvature_context != "mixed_curvature"
        and feature_context not in {"feature_like", "tiny_feature_like"}
        and g1_policy not in {"feature_preserve_no_surface_pull", "feature_preserve_minimal_topology"}
    )

    needed = False
    if eligible:
        # H-ADAPT-5C1B:
        # This is still proposal-only, so the threshold should be sensitive
        # enough to recommend a bounded fairing trial for visible curved-surface
        # mismatch, not only catastrophic curvature failures.
        if (
            relative_delta is not None
            and relative_delta > 0.10
            and delta_mean is not None
            and delta_mean > 0.02
        ):
            needed = True
            reasons.append(
                f"relative curvature delta is fairing-worthy ({relative_delta:.6g})"
            )
        elif delta_mean is not None and delta_mean > 0.03:
            needed = True
            reasons.append(f"curvature mean delta is fairing-worthy ({delta_mean:.6g})")
        elif delta_max is not None and delta_max > 0.08:
            needed = True
            reasons.append(f"curvature max delta is fairing-worthy ({delta_max:.6g})")

    if not eligible:
        status = "not_eligible"
        action = "do_not_apply_curvature_fairing"
        strength = 0.0
        iterations = 0
        max_displacement_factor = 0.0
    elif not needed:
        status = "not_needed"
        action = "keep_current_patch"
        strength = 0.0
        iterations = 0
        max_displacement_factor = 0.0
        reasons.append("curvature delta is already within bounded fairing tolerance")
    else:
        status = "proposal_ready"
        action = "try_bounded_curvature_fairing_variant"
        # Conservative first fairing trial. H-ADAPT-5C2 will still accept this
        # only if seam, quality, G1, G2, and displacement checks improve.
        strength = 0.025
        iterations = 2
        max_displacement_factor = 0.03

    return {
        "status": status,
        "action": action,
        "eligible": bool(eligible),
        "needed": bool(needed),
        "strength": strength,
        "iterations": iterations,
        "max_displacement_factor": max_displacement_factor,
        "reasons": tuple(dict.fromkeys(str(reason) for reason in reasons if str(reason))),
    }


def _adaptive_bounded_curvature_fairing_trial_decision(
    *,
    selected_metadata: dict[str, object],
    curvature_fairing_proposal: dict[str, object],
) -> dict[str, object]:
    """Bridge the backend biharmonic probe into adaptive curvature policy.

    H-ADAPT-5C2 does not run a second fairing solver. The surface backend
    already performs a bounded post-relaxation biharmonic probe and accepts it
    only when quality/movement gates pass. This helper reports that probe as the
    adaptive curvature-fairing trial.
    """

    proposal_status = str(curvature_fairing_proposal.get("status") or "")
    proposal_action = str(curvature_fairing_proposal.get("action") or "")

    raw_decision = selected_metadata.get("biharmonic_fairing_decision")
    decision = raw_decision if isinstance(raw_decision, dict) else {}

    raw_fairing = selected_metadata.get("biharmonic_fairing")
    fairing = raw_fairing if isinstance(raw_fairing, dict) else {}

    attempted = bool(decision.get("attempted", False))
    applied = bool(decision.get("applied", False))
    eligible = bool(decision.get("eligible", False))

    reasons: list[str] = []
    for source in (
        curvature_fairing_proposal.get("reasons"),
        decision.get("reasons"),
    ):
        reasons.extend(_adaptive_reason_tuple(source))

    notes = _adaptive_reason_tuple(decision.get("notes"))
    mode = str(decision.get("mode") or "-")
    error = str(decision.get("error") or "")

    max_displacement = fairing.get("max_displacement", "-")
    mean_displacement = fairing.get("mean_displacement", "-")
    movement_ratio = fairing.get("movement_to_context_edge_ratio", "-")

    if proposal_status != "proposal_ready":
        status = "not_requested"
        action = "keep_current_patch"
        accepted = False
        reasons.append(f"fairing proposal status is {proposal_status or 'unknown'}")
    elif not eligible:
        status = "not_eligible"
        action = "keep_current_patch"
        accepted = False
        if not reasons:
            reasons.append("backend fairing probe was not eligible")
    elif not attempted:
        status = "not_attempted"
        action = "keep_current_patch"
        accepted = False
        if not reasons:
            reasons.append("backend fairing probe was not attempted")
    elif applied:
        status = "accepted"
        action = "accepted_backend_biharmonic_fairing"
        accepted = True
        if not reasons:
            reasons.append("backend fairing probe was accepted")
    else:
        status = "rejected"
        action = "keep_current_patch"
        accepted = False
        if not reasons:
            reasons.append("backend fairing probe was rejected by quality/movement gates")

    return {
        "status": status,
        "action": action,
        "attempted": attempted,
        "applied": applied,
        "accepted": accepted,
        "eligible": eligible,
        "mode": mode,
        "reasons": tuple(dict.fromkeys(str(reason) for reason in reasons if str(reason))),
        "notes": notes,
        "error": error,
        "max_displacement": max_displacement,
        "mean_displacement": mean_displacement,
        "movement_to_context_edge_ratio": movement_ratio,
        "proposal_action": proposal_action or "-",
    }


def _adaptive_end_layer_refinement_diagnostics(
    *,
    base_mesh: trimesh.Trimesh,
    candidate: HoleCandidate,
    preview_mesh: trimesh.Trimesh,
    selected_metadata: dict[str, object],
    curvature_diagnostics: dict[str, object],
    curvature_fairing_trial: dict[str, object],
    support_rings: int = 2,
) -> dict[str, object]:
    """Fresh local-region + reference-overlay diagnostics for the end-layer pass.

    H-ADAPT-5E2 is still proposal-only:
    - pull/count a fresh support region around the hole boundary
    - pull/count the selected patch region
    - build a fresh reference overlay patch for comparison only
    - compare selected patch against the reference patch
    - recommend whether the later rerun layer should be prepared

    It does not replace the selected preview mesh.
    """

    boundary_ids = _adaptive_candidate_boundary_ids(candidate)
    base_neighbors = _adaptive_vertex_neighbors(base_mesh)

    support_ids: set[int] = set(int(v) for v in boundary_ids)
    frontier: set[int] = set(support_ids)

    for _ring in range(max(0, int(support_rings))):
        next_frontier: set[int] = set()
        for vertex_id in frontier:
            if 0 <= int(vertex_id) < len(base_neighbors):
                next_frontier.update(int(v) for v in base_neighbors[int(vertex_id)])
        next_frontier = {
            int(v)
            for v in next_frontier
            if int(v) >= 0 and int(v) < int(len(base_mesh.vertices))
        }
        support_ids.update(next_frontier)
        frontier = next_frontier

    raw_new_vertex_ids = selected_metadata.get("new_vertex_ids", ())
    try:
        selected_new_ids = tuple(
            int(v)
            for v in raw_new_vertex_ids
            if int(v) >= 0 and int(v) < int(len(preview_mesh.vertices))
        )
    except Exception:
        selected_new_ids = ()

    patch_ids = set(selected_new_ids)
    patch_ids.update(
        int(v)
        for v in boundary_ids
        if int(v) >= 0 and int(v) < int(len(preview_mesh.vertices))
    )

    support_count = int(len(support_ids))
    patch_count = int(len(patch_ids))

    local_region_available = bool(support_count > 0 and patch_count > 0)
    patch_available = bool(patch_count > 0)

    selected_support_delta_mean = _adaptive_float(
        curvature_diagnostics.get("curvature_delta_mean"),
        None,
    )
    selected_support_delta_max = _adaptive_float(
        curvature_diagnostics.get("curvature_delta_max"),
        None,
    )
    selected_support_relative_delta = _adaptive_float(
        curvature_diagnostics.get("curvature_relative_delta_mean"),
        None,
    )

    trial_reasons = _adaptive_reason_tuple(
        curvature_fairing_trial.get("reasons", ())
    )
    trial_accepted = bool(curvature_fairing_trial.get("accepted", False))
    low_displacement_used = any(
        "low_displacement" in str(reason)
        for reason in trial_reasons
    )

    if trial_accepted and low_displacement_used:
        selected_patch_source = "bounded_low_displacement_fairing"
    elif trial_accepted:
        selected_patch_source = "backend_biharmonic_fairing"
    else:
        selected_patch_source = str(selected_metadata.get("backend") or "selected_patch")

    reasons: list[str] = []

    reference_available = False
    reference_error = ""
    geometry_deviation_mean: object = "-"
    geometry_deviation_max: object = "-"
    reference_curvature_deviation_mean: object = selected_support_delta_mean if selected_support_delta_mean is not None else "-"
    reference_curvature_deviation_max: object = selected_support_delta_max if selected_support_delta_max is not None else "-"
    reference_relative_deviation_mean: object = (
        selected_support_relative_delta if selected_support_relative_delta is not None else "-"
    )

    if local_region_available:
        try:
            reference_kwargs = _adaptive_surface_builder_kwargs()
            reference_kwargs.update(
                {
                    # Fresh reference overlay:
                    # slightly more surface-guided than the conservative selected
                    # patch, still bounded to the same movement ratio.
                    "relaxation_iterations": 8,
                    "relaxation_strength": 0.12,
                    "surface_weight": 0.12,
                    "support_rings": int(support_rings),
                    "biharmonic_max_movement_ratio": 0.03,
                }
            )

            reference_result = _call_hole_surface_builder(
                build_surface_uvdelaunay_relaxed_preview_mesh,
                base_mesh,
                candidate,
                **reference_kwargs,
            )
            reference_mesh = reference_result["preview_mesh"]
            assert isinstance(reference_mesh, trimesh.Trimesh)

            reference_metadata = _adaptive_preview_metadata_from_result(
                reference_result,
                reference_mesh,
            )
            reference_new_ids_raw = reference_metadata.get("new_vertex_ids", ())
            try:
                reference_new_ids = tuple(
                    int(v)
                    for v in reference_new_ids_raw
                    if int(v) >= 0 and int(v) < int(len(reference_mesh.vertices))
                )
            except Exception:
                reference_new_ids = ()

            reference_available = bool(len(reference_new_ids) > 0)
            if not reference_available:
                reasons.append("reference overlay patch did not report generated vertices")

            if selected_new_ids and reference_new_ids and len(selected_new_ids) == len(reference_new_ids):
                selected_points = np.asarray(preview_mesh.vertices, dtype=float)[
                    list(selected_new_ids)
                ]
                reference_points = np.asarray(reference_mesh.vertices, dtype=float)[
                    list(reference_new_ids)
                ]
                deviations = np.linalg.norm(selected_points - reference_points, axis=1)
                finite = deviations[np.isfinite(deviations)]
                if finite.size:
                    geometry_deviation_mean = float(np.mean(finite))
                    geometry_deviation_max = float(np.max(finite))
            elif selected_new_ids and reference_new_ids:
                reasons.append(
                    "reference overlay vertex count differs from selected patch; geometry deviation is not sampled"
                )

            reference_curvature = _adaptive_local_curvature_diagnostics(
                base_mesh=base_mesh,
                candidate=candidate,
                preview_mesh=reference_mesh,
                metadata=reference_metadata,
            )

            selected_patch_mean = _adaptive_float(
                curvature_diagnostics.get("patch_curvature_mean"),
                None,
            )
            selected_patch_max = _adaptive_float(
                curvature_diagnostics.get("patch_curvature_max"),
                None,
            )
            reference_patch_mean = _adaptive_float(
                reference_curvature.get("patch_curvature_mean"),
                None,
            )
            reference_patch_max = _adaptive_float(
                reference_curvature.get("patch_curvature_max"),
                None,
            )

            if selected_patch_mean is not None and reference_patch_mean is not None:
                reference_curvature_deviation_mean = abs(
                    float(selected_patch_mean) - float(reference_patch_mean)
                )
                reference_relative_deviation_mean = (
                    float(reference_curvature_deviation_mean)
                    / max(abs(float(reference_patch_mean)), 1.0e-6)
                )

            if selected_patch_max is not None and reference_patch_max is not None:
                reference_curvature_deviation_max = abs(
                    float(selected_patch_max) - float(reference_patch_max)
                )

            reasons.append("reference overlay patch generated for deviation comparison")
            # H-ADAPT-5G1:
            # Diagnostic-only dense/remesh probe. The visible dimple survived
            # bounded position/fairing passes, so inspect whether a denser
            # sealed local topology would provide a better candidate. This does
            # not replace selected geometry.
            try:
                dense_result = _call_hole_surface_builder(
                    build_surface_uvdelaunay_sealed_dense_relaxed_preview_mesh,
                    base_mesh,
                    candidate,
                    rings=2,
                    collar_fraction=0.15,
                    dense_inner_rings=3,
                    relaxation_iterations=10,
                    relaxation_strength=0.16,
                    surface_weight=0.20,
                    support_rings=int(support_rings),
                    smooth_dot_threshold=0.50,
                )
                dense_mesh = dense_result["preview_mesh"]
                assert isinstance(dense_mesh, trimesh.Trimesh)

                dense_metadata = _adaptive_preview_metadata_from_result(
                    dense_result,
                    dense_mesh,
                )
                dense_new_vertex_ids = tuple(
                    int(v)
                    for v in dense_metadata.get("new_vertex_ids", ())
                    if int(v) >= 0 and int(v) < int(len(dense_mesh.vertices))
                )
                dense_new_face_ids = tuple(
                    int(f)
                    for f in dense_metadata.get("new_face_ids", ())
                    if int(f) >= 0 and int(f) < int(len(dense_mesh.faces))
                )

                dense_curvature = _adaptive_local_curvature_diagnostics(
                    base_mesh=base_mesh,
                    candidate=candidate,
                    preview_mesh=dense_mesh,
                    metadata=dense_metadata,
                )

                dense_quality = _adaptive_quality_after(dense_metadata)
                dense_seam = _adaptive_seam_decision(dense_metadata)

                dense_relative = _adaptive_float(
                    dense_curvature.get("curvature_relative_delta_mean"),
                    None,
                )
                dense_delta_mean = _adaptive_float(
                    dense_curvature.get("curvature_delta_mean"),
                    None,
                )
                dense_delta_max = _adaptive_float(
                    dense_curvature.get("curvature_delta_max"),
                    None,
                )

                reasons.append(
                    "dense remesh probe available: "
                    f"faces={len(dense_mesh.faces)} vertices={len(dense_mesh.vertices)} "
                    f"new_faces={len(dense_new_face_ids)} new_vertices={len(dense_new_vertex_ids)}"
                )
                reasons.append(
                    "dense remesh probe seam: "
                    f"coverage={dense_seam.get('seam_coverage_ratio', '-')} "
                    f"missing={dense_seam.get('missing_seam_edge_count', '-')}"
                )
                reasons.append(
                    "dense remesh probe quality: "
                    f"degenerate={dense_quality.get('degenerate_face_count', '-')} "
                    f"min_angle={dense_quality.get('min_triangle_angle_degrees', '-')}"
                )
                reasons.append(
                    "dense remesh probe curvature: "
                    f"relative={dense_relative if dense_relative is not None else '-'} "
                    f"delta_mean={dense_delta_mean if dense_delta_mean is not None else '-'} "
                    f"delta_max={dense_delta_max if dense_delta_max is not None else '-'}"
                )
                reasons.append(
                    "H-ADAPT-5G1 is diagnostic-only; dense remesh probe is not selected"
                )
            except Exception as dense_exc:
                reasons.append(f"dense remesh probe failed: {dense_exc}")
        except Exception as exc:
            reference_available = False
            reference_error = str(exc)
            reasons.append(f"reference overlay patch failed: {exc}")

    if not local_region_available:
        status = "not_available"
        action = "skip_end_layer"
        refinement_recommended = False
        problem_region_count = 0
        reasons.append("fresh patch/support local region is not available")
    else:
        status = "reference_overlay_ready" if reference_available else "diagnostic_ready"
        action = "prepare_bounded_rerun" if reference_available else "prepare_reference_overlay"
        refinement_recommended = False
        problem_region_count = 0

        support_signal = (
            selected_support_relative_delta is not None
            and selected_support_relative_delta > 0.08
            and selected_support_delta_mean is not None
            and selected_support_delta_mean > 0.015
        )

        reference_signal = False
        reference_relative = _adaptive_float(reference_relative_deviation_mean, None)
        reference_geometry_max = _adaptive_float(geometry_deviation_max, None)

        if reference_available and reference_relative is not None and reference_relative > 0.04:
            reference_signal = True
        if reference_available and reference_geometry_max is not None and reference_geometry_max > 1.0e-6:
            reference_signal = True

        if support_signal or reference_signal:
            refinement_recommended = True
            problem_region_count = 1
            if support_signal:
                reasons.append(
                    f"remaining relative support-curvature deviation is refinement-worthy ({selected_support_relative_delta:.6g})"
                )
            if reference_signal:
                reasons.append("reference overlay deviation suggests a bounded rerun should be evaluated")
        else:
            reasons.append("end-layer reference/support deviation is within diagnostic tolerance")

        reasons.append(
            "H-ADAPT-5E2 is proposal-only; selected geometry is not replaced"
        )

    return {
        "status": status,
        "action": action,
        "local_region_available": bool(local_region_available),
        "patch_available": bool(patch_available),
        "reference_patch_available": bool(reference_available),
        "support_ring_count": int(support_rings),
        "support_vertex_count": support_count,
        "patch_vertex_count": patch_count,
        "curvature_deviation_mean": reference_curvature_deviation_mean,
        "curvature_deviation_max": reference_curvature_deviation_max,
        "curvature_relative_deviation_mean": reference_relative_deviation_mean,
        "geometry_deviation_mean": geometry_deviation_mean,
        "geometry_deviation_max": geometry_deviation_max,
        "problem_region_count": int(problem_region_count),
        "refinement_recommended": bool(refinement_recommended),
        "rerun_allowed": False,
        "rerun_reason": (
            "proposal_only_reference_overlay_ready"
            if reference_available
            else "diagnostics_only_reference_overlay_not_available"
        ),
        "selected_patch_source": selected_patch_source,
        "reasons": tuple(dict.fromkeys(str(reason) for reason in reasons if str(reason))),
    }



def _adaptive_end_layer_bounded_rerun_evaluation(
    *,
    base_mesh: trimesh.Trimesh,
    candidate: HoleCandidate,
    selected_result: dict[str, object],
    selected_mesh: trimesh.Trimesh,
    selected_metadata: dict[str, object],
    curvature_diagnostics: dict[str, object],
    end_layer_diagnostics: dict[str, object],
    support_rings: int = 2,
) -> dict[str, object]:
    """Evaluate one bounded end-layer rerun candidate.

    H-ADAPT-5E4 is the first end-layer selection step:
    - build the same fresh reference overlay used by H-ADAPT-5E2
    - blend generated patch vertices toward that reference
    - keep seam/boundary vertices fixed
    - accept only if movement is bounded and curvature/G2 remains safe

    The selected mesh changes only when this helper returns accepted=True.
    """

    reasons: list[str] = []

    if str(end_layer_diagnostics.get("status") or "").lower() != "reference_overlay_ready":
        return {
            "accepted": False,
            "status": "not_requested",
            "action": "keep_selected_patch",
            "reason": "end-layer reference overlay is not ready",
            "reasons": ("end-layer reference overlay is not ready",),
        }

    if not bool(end_layer_diagnostics.get("reference_patch_available", False)):
        return {
            "accepted": False,
            "status": "not_available",
            "action": "keep_selected_patch",
            "reason": "reference overlay patch is not available",
            "reasons": ("reference overlay patch is not available",),
        }

    if not bool(end_layer_diagnostics.get("refinement_recommended", False)):
        return {
            "accepted": False,
            "status": "not_needed",
            "action": "keep_selected_patch",
            "reason": "end-layer refinement is not recommended",
            "reasons": ("end-layer refinement is not recommended",),
        }

    try:
        reference_kwargs = _adaptive_surface_builder_kwargs()
        reference_kwargs.update(
            {
                "relaxation_iterations": 8,
                "relaxation_strength": 0.12,
                "surface_weight": 0.12,
                "support_rings": int(support_rings),
                "biharmonic_max_movement_ratio": 0.03,
            }
        )

        reference_result = _call_hole_surface_builder(
            build_surface_uvdelaunay_relaxed_preview_mesh,
            base_mesh,
            candidate,
            **reference_kwargs,
        )
        reference_mesh = reference_result["preview_mesh"]
        assert isinstance(reference_mesh, trimesh.Trimesh)

        reference_metadata = _adaptive_preview_metadata_from_result(
            reference_result,
            reference_mesh,
        )

        selected_new_ids = tuple(
            int(v)
            for v in selected_metadata.get("new_vertex_ids", ())
            if int(v) >= 0 and int(v) < int(len(selected_mesh.vertices))
        )
        reference_new_ids = tuple(
            int(v)
            for v in reference_metadata.get("new_vertex_ids", ())
            if int(v) >= 0 and int(v) < int(len(reference_mesh.vertices))
        )

        if not selected_new_ids or not reference_new_ids:
            return {
                "accepted": False,
                "status": "rejected",
                "action": "keep_selected_patch",
                "reason": "selected or reference patch has no generated vertices",
                "reasons": ("selected or reference patch has no generated vertices",),
            }

        if len(selected_new_ids) != len(reference_new_ids):
            return {
                "accepted": False,
                "status": "rejected",
                "action": "keep_selected_patch",
                "reason": "reference vertex count differs from selected patch",
                "reasons": ("reference vertex count differs from selected patch",),
            }

        selected_vertices = np.asarray(selected_mesh.vertices, dtype=float)
        reference_vertices = np.asarray(reference_mesh.vertices, dtype=float)

        selected_points = selected_vertices[list(selected_new_ids)]
        reference_points = reference_vertices[list(reference_new_ids)]
        delta = reference_points - selected_points
        delta_norm = np.linalg.norm(delta, axis=1)

        finite = delta_norm[np.isfinite(delta_norm)]
        if finite.size == 0:
            return {
                "accepted": False,
                "status": "rejected",
                "action": "keep_selected_patch",
                "reason": "reference deviation contains no finite samples",
                "reasons": ("reference deviation contains no finite samples",),
            }

        reference_deviation_max = float(np.max(finite))
        reference_deviation_mean = float(np.mean(finite))

        # H-ADAPT-5F1:
        # Localized dimple targeting. The previous end-layer pass moved all
        # generated patch vertices uniformly toward the reference overlay.
        # That improved global deviation but can leave a local dimple visible.
        #
        # Here we identify the strongest per-vertex reference deviations,
        # expand one ring through generated patch connectivity, and move only
        # that local cluster. Boundary/seam vertices are not in selected_new_ids
        # and remain fixed.
        selected_new_array = np.asarray(selected_new_ids, dtype=np.int64)
        finite_delta_norm = np.asarray(delta_norm, dtype=float)
        finite_delta_norm = np.where(np.isfinite(finite_delta_norm), finite_delta_norm, 0.0)

        localized_weights = np.zeros(int(len(selected_new_array)), dtype=float)

        if finite_delta_norm.size:
            local_mean = float(np.mean(finite_delta_norm))
            local_std = float(np.std(finite_delta_norm))
            high_threshold = max(
                float(reference_deviation_max) * 0.50,
                local_mean + local_std * 0.50,
            )

            seed_indices = np.where(finite_delta_norm >= high_threshold)[0]
            min_seed_count = max(1, int(np.ceil(float(len(selected_new_array)) * 0.12)))
            max_seed_count = max(min_seed_count, int(np.ceil(float(len(selected_new_array)) * 0.28)))

            if seed_indices.size < min_seed_count:
                ranked = np.argsort(finite_delta_norm)[::-1]
                seed_indices = ranked[:min_seed_count]
            elif seed_indices.size > max_seed_count:
                ranked_seed = seed_indices[np.argsort(finite_delta_norm[seed_indices])[::-1]]
                seed_indices = ranked_seed[:max_seed_count]

            selected_index_by_vertex = {
                int(vertex_id): int(index)
                for index, vertex_id in enumerate(selected_new_array)
            }
            generated_vertex_set = set(int(v) for v in selected_new_array)
            seed_vertices = set(int(selected_new_array[int(index)]) for index in seed_indices)

            neighbor_vertices: set[int] = set()
            try:
                selected_faces = np.asarray(selected_mesh.faces, dtype=np.int64)
                for raw_face in selected_faces:
                    tri = [int(raw_face[0]), int(raw_face[1]), int(raw_face[2])]
                    if not any(vertex_id in seed_vertices for vertex_id in tri):
                        continue
                    for vertex_id in tri:
                        if vertex_id in generated_vertex_set and vertex_id not in seed_vertices:
                            neighbor_vertices.add(vertex_id)
            except Exception:
                neighbor_vertices = set()

            for vertex_id in seed_vertices:
                index = selected_index_by_vertex.get(int(vertex_id))
                if index is not None:
                    localized_weights[int(index)] = 1.0

            for vertex_id in neighbor_vertices:
                index = selected_index_by_vertex.get(int(vertex_id))
                if index is not None:
                    localized_weights[int(index)] = max(float(localized_weights[int(index)]), 0.55)

            if not np.any(localized_weights > 0.0):
                strongest = int(np.argmax(finite_delta_norm))
                localized_weights[strongest] = 1.0
                seed_indices = np.asarray([strongest], dtype=np.int64)
                seed_vertices = {int(selected_new_array[strongest])}
                neighbor_vertices = set()

            localized_dimple_seed_count = int(len(seed_vertices))
            localized_dimple_influenced_count = int(np.count_nonzero(localized_weights > 0.0))
            localized_dimple_threshold = float(high_threshold)
        else:
            localized_dimple_seed_count = 0
            localized_dimple_influenced_count = 0
            localized_dimple_threshold = 0.0

        if reference_deviation_max <= 1.0e-12:
            return {
                "accepted": False,
                "status": "not_needed",
                "action": "keep_selected_patch",
                "reason": "reference overlay is identical within tolerance",
                "reasons": ("reference overlay is identical within tolerance",),
            }

        # Estimate the same context-edge scale used by the low-displacement
        # fairing gate. If available, derive it from accepted fairing movement.
        fairing = _adaptive_mapping(selected_metadata.get("biharmonic_fairing"))
        fairing_max = _adaptive_float(fairing.get("max_displacement"), None)
        fairing_ratio = _adaptive_float(
            fairing.get("movement_to_context_edge_ratio"),
            None,
        )

        if (
            fairing_max is not None
            and fairing_ratio is not None
            and fairing_ratio > 1.0e-12
        ):
            context_edge_scale = float(fairing_max) / float(fairing_ratio)
        else:
            context_edge_scale = max(reference_deviation_max, 1.0)

        # H-ADAPT-5F2:
        # Signed localized dimple lift. Keep the correction local, but allow a
        # stronger end-layer cap than 5F1 because only the detected dimple
        # cluster and its one-ring generated neighbors are affected.
        # Seam/boundary vertices remain fixed because only selected_new_ids are
        # moved.
        max_movement_ratio = 0.18
        max_allowed_move = max(0.0, float(context_edge_scale) * max_movement_ratio)

        blend_scale = min(1.0, max_allowed_move / max(reference_deviation_max, 1.0e-12))

        if blend_scale <= 1.0e-12:
            return {
                "accepted": False,
                "status": "rejected",
                "action": "keep_selected_patch",
                "reason": "bounded blend scale is zero",
                "reasons": ("bounded blend scale is zero",),
            }

        rerun_vertices = np.array(selected_vertices, dtype=float, copy=True)

        base_weighted_delta = delta * (float(blend_scale) * localized_weights).reshape((-1, 1))

        signed_dimple_lift_applied = False
        signed_dimple_lift_amount = 0.0
        signed_dimple_lift_direction = "-"
        signed_lift_delta = np.zeros_like(base_weighted_delta)

        try:
            patch_normal = _adaptive_mean_normal_for_ids(
                selected_mesh,
                tuple(int(v) for v in selected_new_ids),
            )
            active = localized_weights > 0.0

            if patch_normal is not None and np.any(active):
                projections = np.dot(delta, patch_normal)
                active_projections = projections[active]
                active_projections = active_projections[np.isfinite(active_projections)]

                if active_projections.size:
                    projection_mean = float(np.mean(active_projections))
                    sign = 1.0 if projection_mean >= 0.0 else -1.0
                    signed_dimple_lift_direction = (
                        "positive_patch_normal"
                        if sign >= 0.0
                        else "negative_patch_normal"
                    )

                    # H-ADAPT-5F3:
                    # Disable the signed normal lift for now. F2 proved that the
                    # guessed signed lift can worsen global curvature and be
                    # rejected. Keep the center-biased reference-overlay blend,
                    # which is safer and directly targets the dimple cluster.
                    signed_dimple_lift_amount = 0.0
                    signed_lift_delta = (
                        np.asarray(patch_normal, dtype=float).reshape((1, 3))
                        * sign
                        * signed_dimple_lift_amount
                        * np.square(localized_weights).reshape((-1, 1))
                    )
                    signed_dimple_lift_applied = True
        except Exception:
            signed_dimple_lift_applied = False
            signed_dimple_lift_amount = 0.0
            signed_dimple_lift_direction = "error"

        weighted_delta = base_weighted_delta + signed_lift_delta

        # Clamp every moved generated vertex to the end-layer movement cap.
        weighted_norm = np.linalg.norm(weighted_delta, axis=1)
        too_far = weighted_norm > max_allowed_move
        if np.any(too_far):
            safe_norm = np.maximum(weighted_norm[too_far], 1.0e-12)
            weighted_delta[too_far] = (
                weighted_delta[too_far]
                * (float(max_allowed_move) / safe_norm).reshape((-1, 1))
            )

        rerun_vertices[list(selected_new_ids)] = selected_points + weighted_delta

        rerun_mesh = trimesh.Trimesh(
            vertices=rerun_vertices,
            faces=np.asarray(selected_mesh.faces, dtype=np.int64),
            process=False,
        )

        moved_points = np.asarray(rerun_mesh.vertices, dtype=float)[list(selected_new_ids)]
        rerun_move = np.linalg.norm(moved_points - selected_points, axis=1)
        finite_move = rerun_move[np.isfinite(rerun_move)]

        if finite_move.size:
            rerun_movement_max = float(np.max(finite_move))
            rerun_movement_mean = float(np.mean(finite_move))
        else:
            rerun_movement_max = 0.0
            rerun_movement_mean = 0.0

        rerun_movement_ratio = (
            float(rerun_movement_max) / max(float(context_edge_scale), 1.0e-12)
        )

        if rerun_movement_ratio > max_movement_ratio + max(1.0e-9, max_movement_ratio * 1.0e-6):
            return {
                "accepted": False,
                "status": "rejected",
                "action": "keep_selected_patch",
                "reason": "bounded rerun movement exceeded limit",
                "movement_ratio": rerun_movement_ratio,
                "reasons": ("bounded rerun movement exceeded limit",),
            }

        rerun_metadata = dict(selected_metadata)
        rerun_metadata["adaptive_end_layer_selected_patch_source"] = "end_layer_bounded_reference_blend"
        rerun_metadata["adaptive_end_layer_rerun_allowed"] = True
        rerun_metadata["adaptive_end_layer_rerun_reason"] = "bounded_end_layer_rerun_selected"

        rerun_curvature = _adaptive_local_curvature_diagnostics(
            base_mesh=base_mesh,
            candidate=candidate,
            preview_mesh=rerun_mesh,
            metadata=rerun_metadata,
        )

        selected_relative = _adaptive_float(
            curvature_diagnostics.get("curvature_relative_delta_mean"),
            None,
        )
        rerun_relative = _adaptive_float(
            rerun_curvature.get("curvature_relative_delta_mean"),
            None,
        )
        selected_delta_mean = _adaptive_float(
            curvature_diagnostics.get("curvature_delta_mean"),
            None,
        )
        rerun_delta_mean = _adaptive_float(
            rerun_curvature.get("curvature_delta_mean"),
            None,
        )

        rerun_status = str(rerun_curvature.get("status") or "not_reported")
        sign_consistency = rerun_curvature.get("curvature_sign_consistency")

        reject_reasons: list[str] = []

        if rerun_status not in {"ok", "warning"}:
            reject_reasons.append(f"rerun curvature status is {rerun_status}")

        if sign_consistency is False:
            reject_reasons.append("rerun curvature sign consistency is false")

        if (
            selected_relative is not None
            and rerun_relative is not None
            and rerun_relative > selected_relative * 1.25 + 1.0e-9
        ):
            reject_reasons.append(
                "rerun relative curvature deviation worsened beyond tolerance"
            )

        if (
            selected_delta_mean is not None
            and rerun_delta_mean is not None
            and rerun_delta_mean > selected_delta_mean * 1.25 + 1.0e-9
        ):
            reject_reasons.append(
                "rerun curvature mean deviation worsened beyond tolerance"
            )

        # The rerun is meant to reduce the visible dimple by moving toward the
        # reference overlay. It is accepted when it is bounded and does not
        # violate curvature safety.
        if reject_reasons:
            return {
                "accepted": False,
                "status": "rejected",
                "action": "keep_selected_patch",
                "reason": "; ".join(reject_reasons),
                "movement_ratio": rerun_movement_ratio,
                "movement_max": rerun_movement_max,
                "movement_mean": rerun_movement_mean,
                "blend_scale": blend_scale,
                "curvature": rerun_curvature,
                "reasons": tuple(reject_reasons),
            }

        result = dict(selected_result)
        result["preview_mesh"] = rerun_mesh
        result["adaptive_end_layer_rerun_allowed"] = True
        result["adaptive_end_layer_rerun_reason"] = "bounded_end_layer_rerun_selected"
        result["adaptive_end_layer_selected_patch_source"] = "end_layer_bounded_reference_blend"

        reasons.append("bounded end-layer rerun moved generated patch vertices toward reference overlay")
        reasons.append("seam/boundary vertices were fixed")
        reasons.append(f"bounded end-layer rerun movement ratio {rerun_movement_ratio:.6g}")
        reasons.append("H-ADAPT-5F3 used center-biased reference-overlay dimple blend cap 0.18")
        reasons.append(
            "localized dimple pass selected "
            f"{localized_dimple_seed_count} seed vertices and "
            f"{localized_dimple_influenced_count} influenced generated vertices"
        )
        reasons.append(
            f"localized dimple deviation threshold {localized_dimple_threshold:.6g}"
        )
        reasons.append(
            "signed normal lift disabled; using reference-overlay direction only"
        )

        return {
            "accepted": True,
            "status": "accepted",
            "action": "selected_bounded_rerun",
            "reason": "bounded_end_layer_rerun_selected",
            "reasons": tuple(dict.fromkeys(str(reason) for reason in reasons if str(reason))),
            "result": result,
            "mesh": rerun_mesh,
            "metadata": rerun_metadata,
            "curvature": rerun_curvature,
            "movement_ratio": rerun_movement_ratio,
            "movement_max": rerun_movement_max,
            "movement_mean": rerun_movement_mean,
            "blend_scale": blend_scale,
            "signed_dimple_lift_applied": signed_dimple_lift_applied,
            "signed_dimple_lift_amount": signed_dimple_lift_amount,
            "signed_dimple_lift_direction": signed_dimple_lift_direction,
            "localized_dimple_seed_count": localized_dimple_seed_count,
            "localized_dimple_influenced_count": localized_dimple_influenced_count,
            "localized_dimple_threshold": localized_dimple_threshold,
            "reference_deviation_mean": reference_deviation_mean,
            "reference_deviation_max": reference_deviation_max,
        }

    except Exception as exc:
        return {
            "accepted": False,
            "status": "error",
            "action": "keep_selected_patch",
            "reason": str(exc),
            "reasons": (f"bounded end-layer rerun failed: {exc}",),
        }


def _copy_v2_payload_fields(
    target: dict[str, object],
    *,
    prefix: str,
    payload: Mapping[str, object],
    field_map: Mapping[str, str],
) -> None:
    for field_name, metadata_name in field_map.items():
        if field_name in payload:
            target[f"{prefix}_{metadata_name}"] = payload[field_name]


def _adaptive_surface_v2_pipeline_metadata(
    *,
    selected_result: Mapping[str, Any],
    selected_metadata: Mapping[str, object],
    controller_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Build diagnostic Adaptive Surface Fill v2 metadata without selecting geometry.

    This keeps the v2 path ordered as:
    classifier -> seed plan -> seed prototype -> seed-candidate blueprint ->
    non-selecting geometry probe.

    The probe may build a candidate mesh internally for diagnostics, but the
    returned public preview mesh remains the selected v1/adaptive_surface mesh
    until later full G1/G2/quality gates make a candidate selectable.
    """

    try:
        from far_mesh.core.hole_fill.adaptive_surface_v2 import (
            build_adaptive_surface_v2_decision,
        )
        from far_mesh.core.hole_fill.seed_directional import (
            build_curvature_directional_seed_plan,
        )
        from far_mesh.core.hole_fill.seed_v2 import (
            build_adaptive_surface_fill_v2_seed_prototype,
        )
        from far_mesh.core.hole_fill.seed_candidate_v2 import (
            build_adaptive_surface_fill_v2_seed_candidate_blueprint,
        )
        from far_mesh.core.hole_fill.seed_candidate_geometry_v2 import (
            build_curvature_normal_aligned_seed_candidate_geometry_probe,
        )
    except Exception as exc:
        return {
            "adaptive_surface_v2_status": "unavailable",
            "adaptive_surface_v2_case": "v2_modules_unavailable",
            "adaptive_surface_v2_action": "keep_legacy_adaptive_surface_path",
            "adaptive_surface_v2_block_legacy_selection": False,
            "adaptive_surface_v2_require_new_seed": False,
            "adaptive_surface_v2_reasons": (f"Adaptive Surface Fill v2 modules unavailable: {exc}",),
        }

    out: dict[str, object] = {}
    v2_metadata: dict[str, object] = {
        **dict(selected_metadata or {}),
        **dict(controller_metadata or {}),
    }

    def _first_present(*values: object, default: object = None) -> object:
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and value.strip() in {"", "-"}:
                continue
            return value
        return default

    primary_g1 = v2_metadata.get("adaptive_primary_g1_decision")
    if not isinstance(primary_g1, Mapping):
        primary_g1 = {}
    quality_after = v2_metadata.get("quality_after")
    if not isinstance(quality_after, Mapping):
        quality_after = {}

    # The v2 modules intentionally accept stable semantic names.  The legacy
    # preview builder stores the same facts under older nested diagnostic keys,
    # so bridge them here instead of changing the solver logic.
    v2_metadata.setdefault(
        "adaptive_g1_boundary_normal_mean_deviation",
        _first_present(
            primary_g1.get("boundary_normal_mean_deviation_degrees"),
            quality_after.get("boundary_normal_mean_deviation_degrees"),
        ),
    )
    v2_metadata.setdefault(
        "adaptive_g1_boundary_normal_max_deviation",
        _first_present(
            primary_g1.get("boundary_normal_max_deviation_degrees"),
            quality_after.get("boundary_normal_max_deviation_degrees"),
        ),
    )
    v2_metadata.setdefault(
        "adaptive_g1_support_normal_spread",
        _first_present(
            primary_g1.get("support_normal_spread_degrees"),
            v2_metadata.get("adaptive_seed_support_normal_spread_degrees"),
            v2_metadata.get("adaptive_feature_support_normal_spread_degrees"),
        ),
    )
    v2_metadata.setdefault(
        "adaptive_g2_relative_delta_mean",
        _first_present(v2_metadata.get("adaptive_curvature_relative_delta_mean")),
    )
    v2_metadata.setdefault(
        "adaptive_g2_sign_consistency",
        _first_present(v2_metadata.get("adaptive_curvature_sign_consistency")),
    )
    v2_metadata.setdefault(
        "curvature_relative_delta_mean",
        _first_present(v2_metadata.get("adaptive_curvature_relative_delta_mean")),
    )
    v2_metadata.setdefault(
        "curvature_sign_consistency",
        _first_present(v2_metadata.get("adaptive_curvature_sign_consistency")),
    )

    try:
        decision = build_adaptive_surface_v2_decision(v2_metadata)
        decision_payload = decision.to_dict()
        _copy_v2_payload_fields(
            out,
            prefix="adaptive_surface_v2",
            payload=decision_payload,
            field_map={
                "status": "status",
                "case": "case",
                "action": "action",
                "block_legacy_selection": "block_legacy_selection",
                "allow_confidence_target_delta": "allow_confidence_target_delta",
                "allow_local_anisotropic_correction": "allow_local_anisotropic_correction",
                "require_new_seed": "require_new_seed",
                "recommended_seed_family": "recommended_seed_family",
                "recommended_target_policy": "recommended_target_policy",
                "reasons": "reasons",
            },
        )
        v2_metadata.update(out)

        plan = build_curvature_directional_seed_plan(v2_metadata)
        plan_payload = plan.to_dict()
        _copy_v2_payload_fields(
            out,
            prefix="adaptive_surface_v2",
            payload=plan_payload,
            field_map={
                "status": "seed_plan_status",
                "action": "seed_plan_action",
                "build_required": "seed_plan_build_required",
                "seed_family": "seed_family",
                "orientation_case": "orientation_case",
                "orientation_action": "orientation_action",
                "target_policy": "target_policy",
                "curvature_policy": "curvature_policy",
                "confidence_policy": "confidence_policy",
                "support_context_policy": "support_context_policy",
                "boundary_normal_mean_deviation": "boundary_normal_mean_deviation",
                "boundary_normal_max_deviation": "boundary_normal_max_deviation",
                "support_normal_spread": "support_normal_spread",
                "seed_projection_mean_ratio": "seed_projection_mean_ratio",
                "seed_projection_max_ratio": "seed_projection_max_ratio",
                "target_confidence": "target_confidence",
                "target_low_confidence_fraction": "target_low_confidence_fraction",
                "curvature_relative_delta_mean": "curvature_relative_delta_mean",
                "curvature_sign_consistency": "curvature_sign_consistency",
                "reasons": "seed_plan_reasons",
            },
        )
        v2_metadata.update(out)

        prototype = build_adaptive_surface_fill_v2_seed_prototype(v2_metadata)
        prototype_payload = prototype.to_dict()
        _copy_v2_payload_fields(
            out,
            prefix="adaptive_surface_v2_seed_prototype",
            payload=prototype_payload,
            field_map={
                "status": "status",
                "action": "action",
                "build_required": "build_required",
                "seed_family": "family",
                "geometry_status": "geometry_status",
                "orientation_status": "orientation_status",
                "orientation_action": "orientation_action",
                "orientation_confidence": "orientation_confidence",
                "boundary_normal_mean_deviation": "boundary_normal_mean_deviation",
                "boundary_normal_max_deviation": "boundary_normal_max_deviation",
                "boundary_normal_side_score_mean": "side_score_mean",
                "boundary_normal_side_score_max": "side_score_max",
                "normal_sign_status": "normal_sign_status",
                "support_context_policy": "support_context_policy",
                "target_policy": "target_policy",
                "curvature_policy": "curvature_policy",
                "confidence_policy": "confidence_policy",
                "seed_projection_mean_ratio": "seed_projection_mean_ratio",
                "seed_projection_max_ratio": "seed_projection_max_ratio",
                "target_confidence": "target_confidence",
                "target_low_confidence_fraction": "target_low_confidence_fraction",
                "support_normal_spread": "support_normal_spread",
                "curvature_relative_delta_mean": "curvature_relative_delta_mean",
                "curvature_sign_consistency": "curvature_sign_consistency",
                "reasons": "reasons",
            },
        )
        v2_metadata.update(out)

        blueprint = build_adaptive_surface_fill_v2_seed_candidate_blueprint(v2_metadata)
        blueprint_payload = blueprint.to_dict()
        _copy_v2_payload_fields(
            out,
            prefix="adaptive_surface_v2_seed_candidate",
            payload=blueprint_payload,
            field_map={
                "status": "status",
                "action": "action",
                "candidate_family": "family",
                "geometry_status": "geometry_status",
                "selectable": "selectable",
                "curvature_normal_field_status": "curvature_normal_field_status",
                "support_filter": "support_filter",
                "frame_policy": "frame_policy",
                "target_policy": "target_policy",
                "density_policy": "density_policy",
                "acceptance_policy": "acceptance_policy",
                "boundary_vertices": "boundary_vertices",
                "legacy_seed_vertices": "legacy_seed_vertices",
                "planned_seed_vertices": "planned_seed_vertices",
                "planned_support_rings": "planned_support_rings",
                "planned_interior_rings": "planned_interior_rings",
                "normal_continuity_mismatch_score": "normal_continuity_mismatch_score",
                "target_confidence": "target_confidence",
                "target_low_confidence_fraction": "target_low_confidence_fraction",
                "seed_projection_max_ratio": "seed_projection_max_ratio",
                "curvature_relative_delta_mean": "curvature_relative_delta_mean",
                "curvature_sign_consistency": "curvature_sign_consistency",
                "reasons": "reasons",
            },
        )
        v2_metadata.update(out)

        geometry_probe = build_curvature_normal_aligned_seed_candidate_geometry_probe(
            legacy_result=selected_result,
            metadata=v2_metadata,
        )
        geometry_payload = geometry_probe.to_dict()
        _copy_v2_payload_fields(
            out,
            prefix="adaptive_surface_v2_seed_candidate_geometry",
            payload=geometry_payload,
            field_map={
                "status": "status",
                "action": "action",
                "available": "available",
                "applied": "applied",
                "selected": "selected",
                "family": "family",
                "geometry_mode": "mode",
                "face_count": "face_count",
                "vertex_count": "vertex_count",
                "reoriented_face_count": "reoriented_face_count",
                "moved_vertex_count": "moved_vertex_count",
                "movement_mean": "movement_mean",
                "movement_max": "movement_max",
                "movement_ratio_max": "movement_ratio_max",
                "predicted_g1_mean_deviation": "predicted_g1_mean_deviation",
                "predicted_g1_max_deviation": "predicted_g1_max_deviation",
                "predicted_g1_status": "predicted_g1_status",
                "topology_status": "topology_status",
                "topology_reasons": "topology_reasons",
                "reasons": "reasons",
                "evaluation_status": "evaluation_status",
                "evaluation_action": "evaluation_action",
                "evaluation_selectable": "evaluation_selectable",
                "evaluation_g1_status": "evaluation_g1_status",
                "evaluation_g1_mean_deviation": "evaluation_g1_mean_deviation",
                "evaluation_g1_max_deviation": "evaluation_g1_max_deviation",
                "evaluation_g1_mean_limit": "evaluation_g1_mean_limit",
                "evaluation_g1_max_limit": "evaluation_g1_max_limit",
                "evaluation_quality_status": "evaluation_quality_status",
                "evaluation_g2_status": "evaluation_g2_status",
                "evaluation_policy": "evaluation_policy",
                "evaluation_reasons": "evaluation_reasons",
            },
        )

        # Private in-memory bridge for H-CORE-V2-F.  The candidate mesh is not
        # serialized into metadata; it is popped by the caller before metadata
        # is attached to the preview mesh.
        if bool(getattr(geometry_probe, "evaluation_selectable", False)):
            candidate_mesh = getattr(geometry_probe, "candidate_mesh", None)
            if isinstance(candidate_mesh, trimesh.Trimesh):
                out["_adaptive_surface_v2_selected_mesh"] = candidate_mesh.copy()

        return out
    except Exception as exc:
        return {
            **out,
            "adaptive_surface_v2_status": out.get("adaptive_surface_v2_status", "error"),
            "adaptive_surface_v2_case": out.get("adaptive_surface_v2_case", "v2_metadata_failed"),
            "adaptive_surface_v2_action": "keep_legacy_adaptive_surface_path",
            "adaptive_surface_v2_block_legacy_selection": False,
            "adaptive_surface_v2_require_new_seed": False,
            "adaptive_surface_v2_reasons": tuple(
                dict.fromkeys(
                    [
                        *tuple(out.get("adaptive_surface_v2_reasons", ()) or ()),
                        f"Adaptive Surface Fill v2 diagnostic path failed: {exc}",
                    ]
                )
            ),
        }

def build_adaptive_surface_preview_mesh(
    mesh: trimesh.Trimesh,
    candidate: HoleCandidate,
    *,
    all_candidates: object | None = None,
    force_adaptive_surface_v2_candidate: bool = False,
) -> trimesh.Trimesh:
    """Build the canonical adaptive_surface preview.

    H-ADAPT-4B:
    - natural surface_uvdelaunay_relaxed remains the first result
    - G1/support-normal diagnostics decide whether to try a conservative
      no-surface-pull relaxation variant
    - the conservative result is selected only if its seam/quality/G1 score is
      better than the primary result
    - sealed fallback remains available for explicit seam failures
    """

    del all_candidates

    primary_strategy = "surface_uvdelaunay_relaxed"
    fallback_strategy = "surface_uvdelaunay_sealed_relaxed"

    attempted_strategies: list[str] = [primary_strategy]
    fallback_used = False
    fallback_reason = "primary seam diagnostics acceptable"
    fallback_error = ""
    conservative_attempted = False
    conservative_used = False
    conservative_reason = ""

    primary_kwargs = _adaptive_surface_builder_kwargs()
    if force_adaptive_surface_v2_candidate:
        primary_kwargs.update(
            {
                "adaptive_surface_v2_direct_requested": True,
                "adaptive_surface_v2_force_candidate": True,
                "force_adaptive_surface_v2_candidate": True,
            }
        )

    primary_result = _call_hole_surface_builder(
        build_surface_uvdelaunay_relaxed_preview_mesh,
        mesh,
        candidate,
        **primary_kwargs,
    )
    primary_mesh = primary_result["preview_mesh"]
    assert isinstance(primary_mesh, trimesh.Trimesh)

    primary_metadata = _adaptive_preview_metadata_from_result(primary_result, primary_mesh)
    primary_seam = _adaptive_seam_decision(primary_metadata)
    primary_g1 = _adaptive_g1_decision(primary_metadata)

    selected_result = primary_result
    selected_mesh = primary_mesh
    selected_strategy = primary_strategy
    selected_kwargs = dict(primary_kwargs)

    conservative_result: dict[str, object] | None = None
    conservative_seam: dict[str, object] = {}
    conservative_g1: dict[str, object] = {}

    if (
        not force_adaptive_surface_v2_candidate
        and not bool(primary_seam.get("needs_fallback"))
        and bool(primary_g1.get("try_conservative"))
    ):
        conservative_attempted = True
        attempted_strategies.append("surface_uvdelaunay_relaxed:conservative_g1")
        conservative_kwargs = _adaptive_conservative_surface_kwargs(primary_g1)

        try:
            conservative_result = _call_hole_surface_builder(
                build_surface_uvdelaunay_relaxed_preview_mesh,
                mesh,
                candidate,
                **conservative_kwargs,
            )
            conservative_mesh = conservative_result["preview_mesh"]
            assert isinstance(conservative_mesh, trimesh.Trimesh)

            conservative_metadata = _adaptive_preview_metadata_from_result(
                conservative_result,
                conservative_mesh,
            )
            conservative_seam = _adaptive_seam_decision(conservative_metadata)
            conservative_g1 = _adaptive_g1_decision(conservative_metadata)

            primary_score = _adaptive_quality_score(primary_metadata)
            conservative_score = _adaptive_quality_score(conservative_metadata)
            primary_score_total = _adaptive_score_total(primary_metadata)
            conservative_score_total = _adaptive_score_total(conservative_metadata)

            if (
                not bool(conservative_seam.get("needs_fallback"))
                and conservative_score < primary_score
            ):
                selected_result = conservative_result
                selected_mesh = conservative_mesh
                selected_strategy = primary_strategy
                selected_kwargs = dict(conservative_kwargs)
                conservative_used = True
                conservative_reason = (
                    "conservative G1 relaxation improved seam/quality/G1 score "
                    f"({primary_score_total:.6g} -> {conservative_score_total:.6g})"
                )
            else:
                conservative_reason = (
                    "conservative G1 relaxation did not improve score "
                    f"({primary_score_total:.6g} -> {conservative_score_total:.6g})"
                )

        except Exception as exc:
            conservative_reason = f"conservative G1 relaxation failed: {exc}"

    selected_metadata = _adaptive_preview_metadata_from_result(selected_result, selected_mesh)
    selected_seam = _adaptive_seam_decision(selected_metadata)
    fallback_seam: dict[str, object] = {}

    if bool(selected_seam.get("needs_fallback")) and force_adaptive_surface_v2_candidate:
        fallback_reason = (
            "direct adaptive_surface_v2 route keeps surface_uvdelaunay_relaxed as the "
            "candidate base; sealed fallback is not used as the v2 seed source"
        )
    elif bool(selected_seam.get("needs_fallback")):
        fallback_reason = "selected seam diagnostics requested fallback"
        attempted_strategies.append(fallback_strategy)

        try:
            fallback_result = _call_hole_surface_builder(
                build_surface_uvdelaunay_sealed_relaxed_preview_mesh,
                mesh,
                candidate,
                **selected_kwargs,
            )
            fallback_mesh = fallback_result["preview_mesh"]
            assert isinstance(fallback_mesh, trimesh.Trimesh)

            fallback_metadata = _adaptive_preview_metadata_from_result(
                fallback_result,
                fallback_mesh,
            )
            fallback_seam = _adaptive_seam_decision(fallback_metadata)

            if _adaptive_seam_score(fallback_seam) < _adaptive_seam_score(selected_seam):
                selected_result = fallback_result
                selected_mesh = fallback_mesh
                selected_strategy = fallback_strategy
                fallback_used = True
                fallback_reason = "sealed fallback improved seam diagnostics"
            else:
                fallback_reason = "sealed fallback did not improve seam diagnostics"

        except Exception as exc:
            fallback_error = str(exc)
            fallback_reason = "sealed fallback failed"

    selected_metadata = _adaptive_preview_metadata_from_result(selected_result, selected_mesh)
    primary_score_total = _adaptive_score_total(primary_metadata)
    selected_score_total = _adaptive_score_total(selected_metadata)
    conservative_score_total_value = (
        _adaptive_score_total(
            _adaptive_preview_metadata_from_result(conservative_result, conservative_result["preview_mesh"])
        )
        if isinstance(conservative_result, dict)
        and isinstance(conservative_result.get("preview_mesh"), trimesh.Trimesh)
        else None
    )

    if conservative_attempted:
        if conservative_used:
            score_decision = "selected_conservative_g1_lower_score"
        else:
            score_decision = "kept_primary_lower_or_equal_score"
    else:
        score_decision = "no_conservative_g1_needed"

    curvature_diagnostics = _adaptive_local_curvature_diagnostics(
        base_mesh=mesh,
        candidate=candidate,
        preview_mesh=selected_mesh,
        metadata=selected_metadata,
    )

    curvature_fairing_proposal = _adaptive_bounded_curvature_fairing_proposal(
        metadata=selected_metadata,
        curvature_diagnostics=curvature_diagnostics,
        g1_decision=primary_g1,
    )

    curvature_fairing_trial = _adaptive_bounded_curvature_fairing_trial_decision(
        selected_metadata=selected_metadata,
        curvature_fairing_proposal=curvature_fairing_proposal,
    )

    end_layer_diagnostics = _adaptive_end_layer_refinement_diagnostics(
        base_mesh=mesh,
        candidate=candidate,
        preview_mesh=selected_mesh,
        selected_metadata=selected_metadata,
        curvature_diagnostics=curvature_diagnostics,
        curvature_fairing_trial=curvature_fairing_trial,
        support_rings=2,
    )

    if force_adaptive_surface_v2_candidate:
        end_layer_rerun = {
            "accepted": False,
            "reason": "direct adaptive_surface_v2 route skips legacy end-layer rerun before v2 candidate selection",
            "reasons": (
                "direct adaptive_surface_v2 route skips legacy end-layer rerun before v2 candidate selection",
            ),
        }
    else:
        end_layer_rerun = _adaptive_end_layer_bounded_rerun_evaluation(
            base_mesh=mesh,
            candidate=candidate,
            selected_result=selected_result,
            selected_mesh=selected_mesh,
            selected_metadata=selected_metadata,
            curvature_diagnostics=curvature_diagnostics,
            end_layer_diagnostics=end_layer_diagnostics,
            support_rings=2,
        )

    if bool(end_layer_rerun.get("accepted", False)):
        selected_result = end_layer_rerun["result"]
        selected_mesh = end_layer_rerun["mesh"]
        assert isinstance(selected_mesh, trimesh.Trimesh)

        selected_metadata = _adaptive_preview_metadata_from_result(
            selected_result,
            selected_mesh,
        )
        selected_score_total = _adaptive_score_total(selected_metadata)
        selected_strategy = "surface_uvdelaunay_relaxed:end_layer_bounded_rerun"
        attempted_strategies.append("end_layer_bounded_reference_blend")
        score_decision = f"{score_decision}+end_layer_bounded_rerun_selected"

        curvature_diagnostics = _adaptive_local_curvature_diagnostics(
            base_mesh=mesh,
            candidate=candidate,
            preview_mesh=selected_mesh,
            metadata=selected_metadata,
        )

        curvature_fairing_proposal = _adaptive_bounded_curvature_fairing_proposal(
            metadata=selected_metadata,
            curvature_diagnostics=curvature_diagnostics,
            g1_decision=primary_g1,
        )

        curvature_fairing_trial = _adaptive_bounded_curvature_fairing_trial_decision(
            selected_metadata=selected_metadata,
            curvature_fairing_proposal=curvature_fairing_proposal,
        )

        end_layer_diagnostics = _adaptive_end_layer_refinement_diagnostics(
            base_mesh=mesh,
            candidate=candidate,
            preview_mesh=selected_mesh,
            selected_metadata=selected_metadata,
            curvature_diagnostics=curvature_diagnostics,
            curvature_fairing_trial=curvature_fairing_trial,
            support_rings=2,
        )

        end_layer_diagnostics = {
            **end_layer_diagnostics,
            "status": "rerun_accepted",
            "action": "selected_bounded_rerun",
            "reference_patch_available": True,
            "refinement_recommended": False,
            "rerun_allowed": True,
            "rerun_reason": "bounded_end_layer_rerun_selected",
            "selected_patch_source": "end_layer_bounded_reference_blend",
            "geometry_deviation_mean": end_layer_rerun.get("movement_mean", "-"),
            "geometry_deviation_max": end_layer_rerun.get("movement_max", "-"),
            "reasons": tuple(
                dict.fromkeys(
                    [
                        *tuple(end_layer_diagnostics.get("reasons", ())),
                        *tuple(end_layer_rerun.get("reasons", ())),
                    ]
                )
            ),
        }
    else:
        end_layer_diagnostics = {
            **end_layer_diagnostics,
            "rerun_allowed": False,
            "rerun_reason": str(
                end_layer_rerun.get("reason", "bounded_end_layer_rerun_not_selected")
            ),
            "reasons": tuple(
                dict.fromkeys(
                    [
                        *tuple(end_layer_diagnostics.get("reasons", ())),
                        *tuple(end_layer_rerun.get("reasons", ())),
                    ]
                )
            ),
        }

    seed_surface_alignment = analyze_seed_surface_alignment(
        base_mesh=mesh,
        candidate=candidate,
        seed_metadata=selected_metadata,
        support_rings=2,
        smooth_dot_threshold=0.50,
    )

    support_target_disagreement = analyze_support_target_disagreement(
        base_mesh=mesh,
        candidate=candidate,
        seed_metadata=selected_metadata,
        support_rings=2,
        smooth_dot_threshold=0.50,
    )

    controller_metadata = {
        "adaptive_controller": "adaptive_surface_v4_feature_aware_g1",
        "adaptive_controller_schema_version": 4,
        "adaptive_public_method": ADAPTIVE_SURFACE_V2_METHOD,
        "adaptive_surface_v2_direct_requested": bool(force_adaptive_surface_v2_candidate),
        "adaptive_surface_v2_force_candidate": bool(force_adaptive_surface_v2_candidate),
        "force_adaptive_surface_v2_candidate": bool(force_adaptive_surface_v2_candidate),
        "adaptive_selected_strategy": selected_strategy,
        "seed_surface_alignment": seed_surface_alignment,
        "adaptive_seed_alignment_status": seed_surface_alignment.get("status"),
        "adaptive_seed_alignment_action": seed_surface_alignment.get("action"),
        "adaptive_seed_alignment_reasons": seed_surface_alignment.get("reasons", ()),
        "adaptive_seed_projection_distance_mean": seed_surface_alignment.get("seed_projection_distance_mean"),
        "adaptive_seed_projection_distance_max": seed_surface_alignment.get("seed_projection_distance_max"),
        "adaptive_seed_projection_mean_ratio": seed_surface_alignment.get("seed_projection_distance_mean_to_context_edge_ratio"),
        "adaptive_seed_projection_max_ratio": seed_surface_alignment.get("seed_projection_distance_max_to_context_edge_ratio"),
        "adaptive_seed_signed_offset_mean": seed_surface_alignment.get("seed_signed_offset_mean"),
        "adaptive_seed_signed_offset_min": seed_surface_alignment.get("seed_signed_offset_min"),
        "adaptive_seed_signed_offset_max": seed_surface_alignment.get("seed_signed_offset_max"),
        "adaptive_seed_effective_surface_weight": seed_surface_alignment.get("effective_surface_weight"),
        "adaptive_seed_requested_surface_weight": seed_surface_alignment.get("requested_surface_weight"),
        "adaptive_seed_support_normal_spread_degrees": seed_surface_alignment.get("support_normal_spread_degrees"),
        "adaptive_seed_generated_vertex_count": seed_surface_alignment.get("seed_generated_vertex_count"),
        "adaptive_seed_context_edge_length_median": seed_surface_alignment.get("context_edge_length_median"),
        "support_target_disagreement": support_target_disagreement,
        "adaptive_target_disagreement_status": support_target_disagreement.get("status"),
        "adaptive_target_disagreement_action": support_target_disagreement.get("action"),
        "adaptive_target_disagreement_reasons": support_target_disagreement.get("reasons", ()),
        "adaptive_target_disagreement_mean": support_target_disagreement.get("target_disagreement_mean"),
        "adaptive_target_disagreement_max": support_target_disagreement.get("target_disagreement_max"),
        "adaptive_target_disagreement_mean_ratio": support_target_disagreement.get("target_disagreement_mean_ratio"),
        "adaptive_target_disagreement_max_ratio": support_target_disagreement.get("target_disagreement_max_ratio"),
        "adaptive_target_mls2_vs_sphere_mean": support_target_disagreement.get("mls2_vs_sphere_mean"),
        "adaptive_target_mls2_vs_sphere_max": support_target_disagreement.get("mls2_vs_sphere_max"),
        "adaptive_target_mls2_vs_plane_mean": support_target_disagreement.get("mls2_vs_plane_mean"),
        "adaptive_target_mls2_vs_plane_max": support_target_disagreement.get("mls2_vs_plane_max"),
        "adaptive_target_mls1_vs_mls2_mean": support_target_disagreement.get("mls1_vs_mls2_mean"),
        "adaptive_target_mls1_vs_mls2_max": support_target_disagreement.get("mls1_vs_mls2_max"),
        "adaptive_target_mls2_vs_mls3_mean": support_target_disagreement.get("mls2_vs_mls3_mean"),
        "adaptive_target_mls2_vs_mls3_max": support_target_disagreement.get("mls2_vs_mls3_max"),
        "adaptive_target_signed_disagreement_mean": support_target_disagreement.get("target_signed_disagreement_mean"),
        "adaptive_target_signed_disagreement_min": support_target_disagreement.get("target_signed_disagreement_min"),
        "adaptive_target_signed_disagreement_max": support_target_disagreement.get("target_signed_disagreement_max"),
        "adaptive_target_mls_ring1_normal_spread_degrees": support_target_disagreement.get("mls_ring1_normal_spread_degrees"),
        "adaptive_target_mls_ring2_normal_spread_degrees": support_target_disagreement.get("mls_ring2_normal_spread_degrees"),
        "adaptive_target_mls_ring3_normal_spread_degrees": support_target_disagreement.get("mls_ring3_normal_spread_degrees"),
        "adaptive_attempted_strategies": tuple(attempted_strategies),
        "adaptive_fallback_used": fallback_used,
        "adaptive_fallback_reason": fallback_reason,
        "adaptive_fallback_error": fallback_error,
        "adaptive_primary_seam_decision": primary_seam,
        "adaptive_fallback_seam_decision": fallback_seam,
        "adaptive_g1_relaxation_policy": primary_g1.get("g1_policy"),
        "adaptive_g1_policy_reasons": primary_g1.get("reasons", ()),
        "adaptive_feature_context_kind": primary_g1.get("feature_context_kind"),
        "adaptive_feature_preservation_mode": primary_g1.get("feature_preservation_mode"),
        "adaptive_feature_policy_reasons": primary_g1.get("feature_policy_reasons", ()),
        "adaptive_feature_recommended_action": primary_g1.get("feature_recommended_action", "-"),
        "adaptive_feature_density_mode": primary_g1.get("feature_density_mode", "-"),
        "adaptive_feature_smooth_context_valid": primary_g1.get("feature_smooth_context_valid"),
        "adaptive_feature_boundary_vertex_count": primary_g1.get("feature_boundary_vertex_count", "-"),
        "adaptive_feature_support_normal_spread_degrees": primary_g1.get("feature_support_normal_spread_degrees"),
        "adaptive_curvature_status": curvature_diagnostics.get("status"),
        "adaptive_curvature_context_kind": curvature_diagnostics.get("context_kind"),
        "adaptive_curvature_estimator": curvature_diagnostics.get("estimator"),
        "adaptive_support_curvature_mean": curvature_diagnostics.get("support_curvature_mean"),
        "adaptive_support_curvature_max": curvature_diagnostics.get("support_curvature_max"),
        "adaptive_support_curvature_std": curvature_diagnostics.get("support_curvature_std"),
        "adaptive_patch_curvature_mean": curvature_diagnostics.get("patch_curvature_mean"),
        "adaptive_patch_curvature_max": curvature_diagnostics.get("patch_curvature_max"),
        "adaptive_patch_curvature_std": curvature_diagnostics.get("patch_curvature_std"),
        "adaptive_curvature_delta_mean": curvature_diagnostics.get("curvature_delta_mean"),
        "adaptive_curvature_delta_max": curvature_diagnostics.get("curvature_delta_max"),
        "adaptive_curvature_relative_delta_mean": curvature_diagnostics.get("curvature_relative_delta_mean"),
        "adaptive_curvature_sign_consistency": curvature_diagnostics.get("curvature_sign_consistency"),
        "adaptive_curvature_support_sample_count": curvature_diagnostics.get("support_sample_count"),
        "adaptive_curvature_patch_sample_count": curvature_diagnostics.get("patch_sample_count"),
        "adaptive_curvature_reasons": curvature_diagnostics.get("reasons", ()),
        "adaptive_curvature_fairing_status": curvature_fairing_proposal.get("status"),
        "adaptive_curvature_fairing_action": curvature_fairing_proposal.get("action"),
        "adaptive_curvature_fairing_eligible": curvature_fairing_proposal.get("eligible"),
        "adaptive_curvature_fairing_needed": curvature_fairing_proposal.get("needed"),
        "adaptive_curvature_fairing_strength": curvature_fairing_proposal.get("strength"),
        "adaptive_curvature_fairing_iterations": curvature_fairing_proposal.get("iterations"),
        "adaptive_curvature_fairing_max_displacement_factor": curvature_fairing_proposal.get("max_displacement_factor"),
        "adaptive_curvature_fairing_reasons": curvature_fairing_proposal.get("reasons", ()),
        "adaptive_curvature_fairing_trial_status": curvature_fairing_trial.get("status"),
        "adaptive_curvature_fairing_trial_action": curvature_fairing_trial.get("action"),
        "adaptive_curvature_fairing_trial_attempted": curvature_fairing_trial.get("attempted"),
        "adaptive_curvature_fairing_trial_applied": curvature_fairing_trial.get("applied"),
        "adaptive_curvature_fairing_trial_accepted": curvature_fairing_trial.get("accepted"),
        "adaptive_curvature_fairing_trial_mode": curvature_fairing_trial.get("mode"),
        "adaptive_curvature_fairing_trial_reasons": curvature_fairing_trial.get("reasons", ()),
        "adaptive_curvature_fairing_trial_notes": curvature_fairing_trial.get("notes", ()),
        "adaptive_curvature_fairing_trial_error": curvature_fairing_trial.get("error", ""),
        "adaptive_curvature_fairing_trial_max_displacement": curvature_fairing_trial.get("max_displacement", "-"),
        "adaptive_curvature_fairing_trial_mean_displacement": curvature_fairing_trial.get("mean_displacement", "-"),
        "adaptive_curvature_fairing_trial_movement_to_context_edge_ratio": curvature_fairing_trial.get("movement_to_context_edge_ratio", "-"),
        "adaptive_end_layer_status": end_layer_diagnostics.get("status"),
        "adaptive_end_layer_action": end_layer_diagnostics.get("action"),
        "adaptive_end_layer_local_region_available": end_layer_diagnostics.get("local_region_available"),
        "adaptive_end_layer_patch_available": end_layer_diagnostics.get("patch_available"),
        "adaptive_end_layer_reference_patch_available": end_layer_diagnostics.get("reference_patch_available"),
        "adaptive_end_layer_support_ring_count": end_layer_diagnostics.get("support_ring_count"),
        "adaptive_end_layer_support_vertex_count": end_layer_diagnostics.get("support_vertex_count"),
        "adaptive_end_layer_patch_vertex_count": end_layer_diagnostics.get("patch_vertex_count"),
        "adaptive_end_layer_curvature_deviation_mean": end_layer_diagnostics.get("curvature_deviation_mean"),
        "adaptive_end_layer_curvature_deviation_max": end_layer_diagnostics.get("curvature_deviation_max"),
        "adaptive_end_layer_curvature_relative_deviation_mean": end_layer_diagnostics.get("curvature_relative_deviation_mean"),
        "adaptive_end_layer_geometry_deviation_mean": end_layer_diagnostics.get("geometry_deviation_mean"),
        "adaptive_end_layer_geometry_deviation_max": end_layer_diagnostics.get("geometry_deviation_max"),
        "adaptive_end_layer_problem_region_count": end_layer_diagnostics.get("problem_region_count"),
        "adaptive_end_layer_refinement_recommended": end_layer_diagnostics.get("refinement_recommended"),
        "adaptive_end_layer_rerun_allowed": end_layer_diagnostics.get("rerun_allowed"),
        "adaptive_end_layer_rerun_reason": end_layer_diagnostics.get("rerun_reason"),
        "adaptive_end_layer_selected_patch_source": end_layer_diagnostics.get("selected_patch_source"),
        "adaptive_end_layer_reasons": end_layer_diagnostics.get("reasons", ()),
        "adaptive_primary_g1_decision": primary_g1,
        "adaptive_conservative_g1_attempted": conservative_attempted,
        "adaptive_conservative_g1_used": conservative_used,
        "adaptive_conservative_g1_reason": conservative_reason,
        "adaptive_conservative_g1_decision": conservative_g1,
        "adaptive_selected_relaxation_iterations": selected_kwargs.get("relaxation_iterations"),
        "adaptive_selected_relaxation_strength": selected_kwargs.get("relaxation_strength"),
        "adaptive_selected_surface_weight": selected_kwargs.get("surface_weight"),
        "adaptive_primary_score": primary_score_total,
        "adaptive_primary_score_breakdown": _adaptive_score_breakdown(primary_metadata),
        "adaptive_conservative_g1_score": conservative_score_total_value,
        "adaptive_conservative_g1_score_breakdown": (
            _adaptive_score_breakdown(
                _adaptive_preview_metadata_from_result(conservative_result, conservative_result["preview_mesh"])
            )
            if isinstance(conservative_result, dict)
            and isinstance(conservative_result.get("preview_mesh"), trimesh.Trimesh)
            else {}
        ),
        "adaptive_selected_score": selected_score_total,
        "adaptive_selected_score_breakdown": _adaptive_score_breakdown(selected_metadata),
        "adaptive_score_decision": score_decision,
        "adaptive_score_delta": (
            None
            if conservative_score_total_value is None
            else float(conservative_score_total_value) - float(primary_score_total)
        ),
        "adaptive_controller_notes": (
            "Adaptive surface first tried natural surface_uvdelaunay_relaxed.",
            "Elevated support normal spread can trigger a conservative no-surface-pull variant.",
            "Conservative G1 relaxation is selected only if seam/quality/G1 score improves.",
            "Feature-aware context classification distinguishes smooth, mixed-curved, feature-like, and tiny feature-like holes.",
            "Local 3D curvature diagnostics estimate unsigned support/patch curvature; no G2 fairing is applied yet.",
            "Bounded curvature fairing proposal is diagnostic-only; no geometry is changed in H-ADAPT-5C1.",
            "Bounded curvature fairing trial reports the backend biharmonic probe result; accepted probes are already reflected in the selected backend mesh.",
            "End-layer diagnostics pull a fresh local patch/support region and prepare the later reference-overlay rerun layer.",
            "End-layer reference overlay is proposal-only in H-ADAPT-5E2; selected geometry is not replaced.",
            "End-layer bounded rerun may replace selected generated patch vertices in H-ADAPT-5E4 when all gates remain safe.",
            "H-ADAPT-5E5 allows a stronger end-layer-only dimple pass while keeping seam/boundary vertices fixed.",
            "H-ADAPT-5F1 localizes the dimple pass to the strongest reference-overlay deviation cluster.",
            "H-ADAPT-5F2 adds a signed local dimple lift along the patch normal for the detected cluster.",
            "H-ADAPT-5F3 disables signed lift and uses a center-biased reference-overlay dimple blend.",
            "H-ADAPT-5G1 probes dense local remesh topology for persistent dimple artifacts without selecting it.",
            "H-CORE-R1 seed/support surface alignment diagnostics are reported from the new far_mesh.core.hole_fill package.",
            "H-CORE-R1B compares MLS/sphere/plane/ring support targets before changing seed geometry.",
            "The public adaptive_surface v1 route is retired; old adaptive_surface requests normalize to adaptive_surface_v2.",
            "Direct adaptive_surface_v2 keeps the surface_uvdelaunay_relaxed candidate base so the v2 seed is tested on the natural high-density path, not the sealed/fan fallback path.",
            "Direct adaptive_surface_v2 skips conservative G1 and legacy end-layer rerun as public selectors before v2 candidate selection.",
        ),
    }

    v2_pipeline_output = _adaptive_surface_v2_pipeline_metadata(
        selected_result=selected_result,
        selected_metadata=selected_metadata,
        controller_metadata=controller_metadata,
    )
    v2_selected_mesh = v2_pipeline_output.pop("_adaptive_surface_v2_selected_mesh", None)
    controller_metadata.update(v2_pipeline_output)

    if isinstance(v2_selected_mesh, trimesh.Trimesh):
        selected_mesh = v2_selected_mesh
        selected_result = dict(selected_result)
        selected_result["preview_mesh"] = selected_mesh
        selected_strategy = f"{selected_strategy}:adaptive_surface_v2_curvature_normal_seed"
        controller_metadata["adaptive_selected_strategy"] = selected_strategy
        attempted_with_v2 = list(attempted_strategies)
        if "adaptive_surface_v2:curvature_normal_aligned_seed_candidate" not in attempted_with_v2:
            attempted_with_v2.append("adaptive_surface_v2:curvature_normal_aligned_seed_candidate")
        controller_metadata["adaptive_attempted_strategies"] = tuple(attempted_with_v2)
        controller_metadata["adaptive_surface_v2_selected_mesh_source"] = (
            "strict_curvature_normal_aligned_seed_candidate"
        )

    selected_result.update(controller_metadata)

    selected_mesh = _attach_hole_fill_preview_metadata(selected_mesh, selected_result)
    selected_mesh.metadata.update(controller_metadata)
    return selected_mesh


def build_hole_fill_preview_mesh(
    mesh: trimesh.Trimesh,
    candidate: HoleCandidate,
    *,
    method: str = "fan",
    all_candidates: Iterable[HoleCandidate] | None = None,
) -> trimesh.Trimesh:
    """Build a hole-fill preview mesh.

    The curvature_sphere method is experimental, preview-first, and exposed
    through the same capability-gated method list used by the GUI.
    """

    method_key = _normalize_method_key(method)
    if method_key == ADAPTIVE_SURFACE_METHOD:
        method_key = ADAPTIVE_SURFACE_V2_METHOD


    if method_key == ADAPTIVE_SURFACE_V2_METHOD:
        # H-CORE-V2-G2: public GUI/task route forces the v2 candidate pipeline,
        # but keeps the natural surface_uvdelaunay_relaxed base as the v2 seed
        # source.  This avoids the degraded sealed/fan fallback condition and
        # makes v2 the only public adaptive route.
        preview_mesh = build_adaptive_surface_preview_mesh(
            mesh,
            candidate,
            all_candidates=all_candidates,
            force_adaptive_surface_v2_candidate=True,
        )
        preview_mesh.metadata["method"] = ADAPTIVE_SURFACE_V2_METHOD
        preview_mesh.metadata["adaptive_public_method"] = ADAPTIVE_SURFACE_V2_METHOD
        preview_mesh.metadata["adaptive_surface_v2_direct_requested"] = True
        preview_mesh.metadata["adaptive_surface_v2_force_candidate"] = True
        preview_mesh.metadata["force_adaptive_surface_v2_candidate"] = True
        preview_mesh.metadata["backend"] = str(
            preview_mesh.metadata.get("adaptive_selected_strategy")
            or "adaptive_surface_v2_curvature_normal_seed"
        )
        preview_mesh.metadata["adaptive_stage"] = str(
            preview_mesh.metadata.get("adaptive_stage")
            or preview_mesh.metadata.get("adaptive_selected_strategy")
            or "adaptive_surface_v2_direct_natural_base"
        )
        return preview_mesh

    if method_key == "curvature_sphere":
        result = build_curvature_sphere_center_fan_preview_mesh(
            mesh,
            candidate,
            rings=2,
        )
        preview_mesh = result.get("preview_mesh")
        if not isinstance(preview_mesh, trimesh.Trimesh):
            raise ValueError("curvature_sphere preview did not return a trimesh.Trimesh")
        return _attach_hole_fill_preview_metadata(preview_mesh, locals().get("result"))

    if method_key == "curvature_sphere_refined":
        result = build_curvature_sphere_refined_preview_mesh(
            mesh,
            candidate,
            rings=2,
            interior_rings=None,
        )
        preview_mesh = result.get("preview_mesh")
        if not isinstance(preview_mesh, trimesh.Trimesh):
            raise ValueError("curvature_sphere_refined preview did not return a trimesh.Trimesh")
        return _attach_hole_fill_preview_metadata(preview_mesh, locals().get("result"))

    if method_key == "curvature_sphere_grid8":
        result = build_curvature_sphere_grid8_preview_mesh(
            mesh,
            candidate,
            rings=2,
            interior_rings=None,
        )
        preview_mesh = result.get("preview_mesh")
        if not isinstance(preview_mesh, trimesh.Trimesh):
            raise ValueError("curvature_sphere_grid8 preview did not return a trimesh.Trimesh")
        return _attach_hole_fill_preview_metadata(preview_mesh, locals().get("result"))

    if method_key == "curvature_sphere_uvgrid":
        result = build_curvature_sphere_uvgrid_preview_mesh(
            mesh,
            candidate,
            rings=2,
            interior_rings=None,
        )
        preview_mesh = result.get("preview_mesh")
        if not isinstance(preview_mesh, trimesh.Trimesh):
            raise ValueError("curvature_sphere_uvgrid preview did not return a trimesh.Trimesh")
        return _attach_hole_fill_preview_metadata(preview_mesh, locals().get("result"))

    if method_key == "curvature_sphere_uvdelaunay":
        result = build_curvature_sphere_uvdelaunay_preview_mesh(
            mesh,
            candidate,
            rings=2,
        )
        preview_mesh = result.get("preview_mesh")
        if not isinstance(preview_mesh, trimesh.Trimesh):
            raise ValueError("curvature_sphere_uvdelaunay preview did not return a trimesh.Trimesh")
        return _attach_hole_fill_preview_metadata(preview_mesh, locals().get("result"))

    if method_key == "curvature_sphere_uvdelaunay_relaxed":
        result = build_curvature_sphere_uvdelaunay_relaxed_preview_mesh(
            mesh,
            candidate,
            rings=2,
            relaxation_iterations=8,
            relaxation_strength=0.20,
            surface_weight=0.15,
        )
        preview_mesh = result.get("preview_mesh")
        if not isinstance(preview_mesh, trimesh.Trimesh):
            raise ValueError("curvature_sphere_uvdelaunay_relaxed preview did not return a trimesh.Trimesh")
        return _attach_hole_fill_preview_metadata(preview_mesh, locals().get("result"))

    if method_key == "surface_uvdelaunay_relaxed":
        result = build_surface_uvdelaunay_relaxed_preview_mesh(
            mesh,
            candidate,
            rings=2,
            relaxation_iterations=8,
            relaxation_strength=0.20,
            surface_weight=0.20,
            support_rings=2,
            smooth_dot_threshold=0.50,
        )
        preview_mesh = result.get("preview_mesh")
        if not isinstance(preview_mesh, trimesh.Trimesh):
            raise ValueError("surface_uvdelaunay_relaxed preview did not return a trimesh.Trimesh")
        return _attach_hole_fill_preview_metadata(preview_mesh, locals().get("result"))

    if method_key == "surface_uvdelaunay_sealed_dense_relaxed":
        result = build_surface_uvdelaunay_sealed_dense_relaxed_preview_mesh(
            mesh,
            candidate,
            rings=2,
            collar_fraction=0.15,
            dense_inner_rings=3,
            relaxation_iterations=10,
            relaxation_strength=0.18,
            surface_weight=0.28,
            support_rings=2,
            smooth_dot_threshold=0.50,
        )
        preview_mesh = result.get("preview_mesh")
        if not isinstance(preview_mesh, trimesh.Trimesh):
            raise ValueError("surface_uvdelaunay_sealed_dense_relaxed preview did not return a trimesh.Trimesh")
        return _attach_hole_fill_preview_metadata(preview_mesh, locals().get("result"))

    if method_key == "surface_uvdelaunay_sealed_relaxed":
        result = build_surface_uvdelaunay_sealed_relaxed_preview_mesh(
            mesh,
            candidate,
            rings=2,
            collar_fraction=0.30,
            relaxation_iterations=8,
            relaxation_strength=0.20,
            surface_weight=0.20,
            support_rings=2,
            smooth_dot_threshold=0.50,
        )
        preview_mesh = result.get("preview_mesh")
        if not isinstance(preview_mesh, trimesh.Trimesh):
            raise ValueError("surface_uvdelaunay_sealed_relaxed preview did not return a trimesh.Trimesh")
        return _attach_hole_fill_preview_metadata(preview_mesh, locals().get("result"))

    return _build_hole_fill_preview_mesh_builtin(
        mesh,
        candidate,
        method=method,
        all_candidates=all_candidates,
    )

