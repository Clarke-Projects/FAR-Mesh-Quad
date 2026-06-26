"""Constraint-bound BoreTool feature remeshing pipeline.

This module is intentionally GUI-free and viewport-free.  It receives stable
selection/candidate data from the caller and returns either a fully validated
rebuilt mesh or a diagnostic failure.  It does not mutate the active project
state directly.

Architecture
------------
The rebuild pipeline is deliberately linear:

1. Normalize the caller/candidate request.
2. Gather RegionData anchor evidence from ``region_select``.
3. Build bounded delete-patch proposals with ``rebuild_target``.
4. Derive closed boundary-loop pairs from each exact delete patch.
5. Build measured-loop quad replacement plans.
6. Trial-build the full mesh and require topology validation.
7. Apply the first validated plan and return ``RebuildResult``.

Layer ownership
---------------
``region_select.py``
    Produces RegionData/anchor evidence only.  It does not classify features and
    does not authorize rebuilds.

``recognition.py`` / ``recognition_component_engine.py``
    Consume RegionData and provide CandidateData such as BORE, POCKET, or
    CHAMFER plus explicit owned/rebuild face IDs when the candidate is actionable.

``rebuild_target.py``
    Converts CandidateData face evidence into bounded DeletePatchProposal objects.

``rebuild.py`` / this module
    Owns replacement topology only: boundary-loop extraction, equal/unequal
    measured-loop quad planning, semantic feature remeshing, trial validation,
    and final mesh construction.

Geometry / rebuild policy
-------------------------
Rebuild is semantic feature remeshing under measured constraints.  It is not
just a delete/reconnect operation.  Recognition identifies the feature,
Measurement defines the feature constraints, CandidateData owns the surfaces,
and Rebuild reconstructs those owned surfaces into the best valid quad-style
replacement mesh it can produce while preserving protected boundaries and
relationships.

The exact delete patch still owns the boundary loops.  Boundary vertices remain
connected to the surrounding mesh unless a later explicit boundary-relax action
opts in to moving them.  Replacement topology is allowed to insert new rings and
place generated vertices on the measured semantic primitive emitted by
Recognition, so rebuild is an actual geometry/topology improvement rather than
a boundary-locked copy of the old coarse triangles.

Current locked-boundary rule
----------------------------
When feature boundaries are locked, Rebuild may add axial/detail rings but must
not increase circumferential/angular density directly at the protected boundary.
True angular refinement is deferred until a future parent-boundary split and
parent-collar relaxation stage exists.  This intent applies to all active
feature families: BORE, POCKET, CHAMFER, and future compound features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import math

import numpy as np
import trimesh

from ..exceptions import BoreRebuildRejected
from .rebuild_target import (
    RebuildTargetPatch,
    build_bounded_rebuild_target_face_sets,
    target_patches_from_result,
)
from .rebuild_emit import (
    REBUILD_EMIT_EXTRACTION_CHECKPOINT_V176B,
    REBUILD_EMIT_NON_MUTATION_CONTRACT_V176B,
    REBUILD_FAILURE_REPORTING_EXTRACTION_CHECKPOINT_V176O,
    REBUILD_FAILURE_REPORTING_NON_MUTATION_CONTRACT_V176O,
    attempt_summary_v176o as _attempt_summary,
    compact_attempt_summaries_v176o as _compact_attempt_summaries,
    format_failure_message_v176o as _format_failure_message,
    pocket_rebuild_trace_summary_v175k,
)
from .rebuild_density import (
    REBUILD_DENSITY_HELPER_EXTRACTION_CHECKPOINT_V176Q,
    REBUILD_DENSITY_HELPER_NON_MUTATION_CONTRACT_V176Q,
    QUAD_DENSITY_MODE_ALIASES,
    QUAD_DENSITY_MODE_FULL,
    QUAD_DENSITY_MODE_LEAN,
    QUAD_DENSITY_MODE_PI,
    axial_segment_count_v176q as _axial_segment_count,
    normalize_quad_density_mode_v176q as _normalize_quad_density_mode,
    resolve_quad_density_mode_v176q as _resolve_quad_density_mode,
)
from .rebuild_semantic import (
    SEMANTIC_FEATURE_REMESH_CONTRACT_V162,
    REBUILD_SEMANTIC_HELPER_EXTRACTION_CHECKPOINT_V176Z,
    REBUILD_SEMANTIC_HELPER_NON_MUTATION_CONTRACT_V176Z,
    candidate_rebuild_radius_v176z as _candidate_rebuild_radius,
    chamfer_local_rail_ring_points_v167_v176z as _chamfer_local_rail_ring_points_v167,
    metadata_first_float_v163_v176z as _metadata_first_float_v163,
    metadata_indicates_bore_opening_to_exit_wall_v173v_v176z as _metadata_indicates_bore_opening_to_exit_wall_v173v,
    semantic_constraint_quality_summary_v176z as _semantic_constraint_quality_summary,
    semantic_projected_ring_points_v176z as _semantic_projected_ring_points,
    semantic_radius_authority_v163_v176z as _semantic_radius_authority_v163,
    semantic_target_circumferential_count_v176z as _semantic_target_circumferential_count,
    semantic_generated_vertex_shaping_policy_v177a as _semantic_generated_vertex_shaping_policy,
)
from .rebuild_bore import (
    REBUILD_BORE_FRAME_HELPER_EXTRACTION_CHECKPOINT_V176X,
    REBUILD_BORE_FRAME_HELPER_NON_MUTATION_CONTRACT_V176X,
    REBUILD_BORE_SHAPE_AUTHORITY_EXTRACTION_CHECKPOINT_V176Y,
    REBUILD_BORE_SHAPE_AUTHORITY_NON_MUTATION_CONTRACT_V176Y,
    edge_loop_near_frame_opening_v176x as _edge_loop_near_frame_opening,
    full_depth_bore_wall_target_faces_from_candidate_frame_v176x as _full_depth_bore_wall_target_faces_from_candidate_frame,
    owned_bore_wall_cylinder_shape_authority_v173x_v176y as _owned_bore_wall_cylinder_shape_authority_v173x,
    owned_bore_wall_cylinder_shape_authority_v173z_v176y as _owned_bore_wall_cylinder_shape_authority_v173z,
    protected_loop_pair_from_candidate_frame_v176x as _protected_loop_pair_from_candidate_frame,
    protected_loop_pair_from_selection_v176x as _protected_loop_pair_from_selection,
    shape_frame_loop_angles_v173z_v176y as _shape_frame_loop_angles_v173z,
    two_opening_frame_from_candidate_metadata_v176x as _two_opening_frame_from_candidate_metadata,
)
from .rebuild_pocket import (
    REBUILD_POCKET_PATCH_PLAN_FIELDS_V176I,
    REBUILD_POCKET_PATCH_PLAN_SHELL_CHECKPOINT_V176I,
    REBUILD_POCKET_PATCH_PLAN_SHELL_NON_MUTATION_CONTRACT_V176I,
    REBUILD_POCKET_CAP_GATE_EXTRACTION_CHECKPOINT_V176K,
    REBUILD_POCKET_CAP_GATE_NON_MUTATION_CONTRACT_V176K,
    REBUILD_POCKET_RECESS_CUP_GATE_EXTRACTION_CHECKPOINT_V176J,
    REBUILD_POCKET_RECESS_CUP_GATE_NON_MUTATION_CONTRACT_V176J,
    REBUILD_POCKET_RECESS_CUP_PATCH_ASSEMBLY_CHECKPOINT_V176L,
    REBUILD_POCKET_RECESS_CUP_PATCH_ASSEMBLY_NON_MUTATION_CONTRACT_V176L,
    REBUILD_POCKET_CAP_GEOMETRY_HELPER_EXTRACTION_CHECKPOINT_V176M,
    REBUILD_POCKET_CAP_GEOMETRY_HELPER_NON_MUTATION_CONTRACT_V176M,
    REBUILD_POCKET_FLOOR_GRID_HELPER_EXTRACTION_CHECKPOINT_V176U,
    REBUILD_POCKET_FLOOR_GRID_HELPER_NON_MUTATION_CONTRACT_V176U,
    assemble_pocket_recess_cup_patch_plan_v176l,
    cap_triangles_for_loop_v176m as _cap_triangles_for_loop,
    pocket_band_faces_between_rings_with_adaptive_sew_collar_v176v as _band_faces_between_rings_with_adaptive_sew_collar,
    pocket_adaptive_zipper_sew_collar_faces_between_rings_v176v as _adaptive_zipper_sew_collar_faces_between_rings,
    pocket_floor_annular_quad_grid_from_loops_v176w as _pocket_floor_annular_quad_grid_from_loops,
    pocket_floor_center_fan_fallback_v176u as _pocket_floor_center_fan_fallback,
    pocket_floor_quad_grid_from_loop_v176u as _pocket_floor_quad_grid_from_loop,
    candidate_requests_pocket_cap_rebuild_v176k as _candidate_requests_pocket_cap_rebuild,
    candidate_requests_pocket_recess_cup_rebuild_v176j as _candidate_requests_pocket_recess_cup_rebuild,
    pocket_patch_plan_shell_inventory_v176i,
)
from .rebuild_loops import (
    REBUILD_GENERIC_LOOP_HELPER_EXTRACTION_CHECKPOINT_V176G,
    REBUILD_GENERIC_LOOP_HELPER_NON_MUTATION_CONTRACT_V176G,
    REBUILD_LOOPS_EXTRACTION_CHECKPOINT_V176C,
    REBUILD_LOOPS_NON_MUTATION_CONTRACT_V176C,
    REBUILD_POCKET_LOOP_ROLE_EXTRACTION_CHECKPOINT_V176E,
    REBUILD_POCKET_LOOP_ROLE_NON_MUTATION_CONTRACT_V176E,
    REBUILD_TRANSITION_RING_HELPER_EXTRACTION_CHECKPOINT_V176N,
    REBUILD_TRANSITION_RING_HELPER_NON_MUTATION_CONTRACT_V176N,
    align_second_loop_to_first_v176g as _align_second_loop_to_first,
    band_quads_between_rings_v176n as _band_quads_between_rings,
    densify_transition_count_sequence_v176n as _densify_transition_count_sequence,
    distributed_drop_positions_v176n as _distributed_drop_positions,
    loop_edges_from_vertices_v176g as _loop_edges_from_vertices,
    loop_vertices_to_edges_v176g as _loop_vertices_to_edges,
    order_closed_edge_loop_vertices_v176g as _order_closed_edge_loop_vertices,
    pocket_loop_phase_trace_v175j,
    resolve_pocket_recess_loop_roles_v176e,
    sort_loop_pair_by_angle_v176g as _sort_loop_pair_by_angle,
    align_unequal_loop_pair_to_angle_samples_v176w as _align_unequal_loop_pair_to_angle_samples,
    target_unequal_transition_band_count_v176n as _target_unequal_transition_band_count,
    transition_count_sequence_v176n as _transition_count_sequence,
)
from .rebuild_validation import (
    REBUILD_VALIDATION_EXTRACTION_CHECKPOINT_V176F,
    REBUILD_VALIDATION_NON_MUTATION_CONTRACT_V176F,
    REBUILD_BOUNDARY_DIAGNOSTIC_EXTRACTION_CHECKPOINT_V176P,
    REBUILD_BOUNDARY_DIAGNOSTIC_NON_MUTATION_CONTRACT_V176P,
    REBUILD_PLAN_GEOMETRY_QUALITY_EXTRACTION_CHECKPOINT_V177F,
    REBUILD_PLAN_GEOMETRY_QUALITY_NON_MUTATION_CONTRACT_V177F,
    boundary_edge_count as _boundary_edge_count,
    damaged_bore_internal_boundary_swallow_diagnostics_v176p as _damaged_bore_internal_boundary_swallow_diagnostics_v173n,
    generated_surface_boundary_match_diagnostics_v176p as _generated_surface_boundary_match_diagnostics,
    int_preserve_zero as _int_preserve_zero,
    pocket_recess_cup_trial_accepts as _pocket_recess_cup_trial_accepts,
    trial_accepts_for_context as _trial_accepts_for_context,
    validate_mesh as _validate_mesh,
    validate_plan_geometry_quality_v177f as _validate_plan_geometry_quality,
)
from .rebuild_geometry import (
    REBUILD_GEOMETRY_EXTRACTION_CHECKPOINT_V176D,
    REBUILD_GEOMETRY_NON_MUTATION_CONTRACT_V176D,
    boundary_adapter_weight_v165 as _boundary_adapter_weight_v165,
    endpoint_safe_semantic_weight_v166 as _endpoint_safe_semantic_weight_v166,
    loop_angle_basis as _loop_angle_basis,
    loop_center as _loop_center,
    loop_median_radius as _loop_median_radius,
    loop_radius_spread_ratio as _loop_radius_spread_ratio,
    median_loop_edge_length as _median_loop_edge_length,
    minimum_loop_pair_separation as _minimum_loop_pair_separation,
    orthonormal_basis as _orthonormal_basis,
    safe_float as _safe_float,
    sample_closed_loop_points as _sample_closed_loop_points,
    smoothstep_v165 as _smoothstep_v165,
    to_vector3 as _to_vector3,
    unit_normal as _unit_normal,
    unit_vector as _unit_vector,
)
from .rebuild_orientation import (
    REBUILD_ORIENTATION_HELPER_EXTRACTION_CHECKPOINT_V176R,
    REBUILD_ORIENTATION_HELPER_NON_MUTATION_CONTRACT_V176R,
    orient_plan_triangles_to_source_patch_v176r as _orient_plan_triangles_to_source_patch,
    orient_pocket_wall_triangles_by_radial_role_v176r as _orient_pocket_wall_triangles_by_radial_role,
    orient_triangles_to_source_role_normal_v176r as _orient_triangles_to_source_role_normal,
    source_patch_face_references_v176r as _source_patch_face_references,
)
from .rebuild_plan import (
    REBUILD_QUAD_PLAN_EXTRACTION_CHECKPOINT_V177C,
    REBUILD_QUAD_PLAN_NON_MUTATION_CONTRACT_V177C,
    QuadPlan,
    equal_loop_quad_plan_v177c as _equal_loop_quad_plan,
    loop_pair_transition_allowed_v177c as _loop_pair_transition_allowed,
    quad_plan_for_attempt_v177c as _quad_plan_for_attempt,
    unequal_loop_quad_plan_v177c as _unequal_loop_quad_plan,
)
from ..selection.region_select import select_region_data
from ..types import FeatureFamily, RecognitionStage, RegionData, tuple_ints
from ..topology import (
    boundary_edges_for_face_patch,
    build_edge_to_faces,
    connected_face_components,
    edge_loop_components,
    normalize_edge,
)

EdgeKey = tuple[int, int]
Vector3 = tuple[float, float, float]
RGBA = tuple[int, int, int, int]

DEFAULT_BASE_FACE_COLOR: RGBA = (190, 190, 190, 255)
DEFAULT_REBUILT_FACE_COLOR: RGBA = (0, 213, 255, 255)


from .rebuild_inventory import (
    REBUILD_AUTHORITY_BOUNDARIES_V176A,
    REBUILD_INVENTORY_EXTRACTION_CHECKPOINT_V176T,
    REBUILD_INVENTORY_EXTRACTION_NON_MUTATION_CONTRACT_V176T,
    REBUILD_MODULE_INVENTORY_V176A,
    REBUILD_MONOLITHIC_AUTHORITY_CONTRACT_V176A,
    REBUILD_POCKET_GENERATOR_EXTRACTION_BOUNDARY_V176H,
    REBUILD_POCKET_GENERATOR_INVENTORY_CHECKPOINT_V176H,
    REBUILD_POCKET_GENERATOR_NON_MUTATION_CONTRACT_V176H,
    REBUILD_POCKET_RECESS_CUP_SECTIONS_V176H,
    REBUILD_ROADMAP_INVENTORY_CHECKPOINT_V176A,
    REBUILD_SAFE_EXTRACTION_ORDER_V176A,
    rebuild_pocket_generator_inventory_v176h,
    rebuild_refactor_inventory_v176a,
)

def _canonical_feature_family_from_metadata(
    candidate_metadata: Mapping[str, object] | None,
    *,
    entity_type: object | None = None,
    rebuild_gate: object | None = None,
) -> str:
    """Return the semantic operation family for diagnostics and target policy.

    This helper is intentionally diagnostic/policy naming only.  It does not
    classify a feature and it does not alter generated geometry.  Recognition
    already chose the candidate; Rebuild only needs one canonical family label
    so BORE, POCKET, and CHAMFER logs do not inherit stale gate wording from
    older BoreTool paths.
    """

    meta = dict(candidate_metadata or {})
    family = str(meta.get("feature_family", "") or "").strip().lower()
    entity = str(entity_type if entity_type is not None else meta.get("entity_type", meta.get("feature_kind", "")) or "").strip().lower()
    gate = str(rebuild_gate if rebuild_gate is not None else meta.get("rebuild_gate", meta.get("candidate_action", "")) or "").strip().lower()

    if entity in {"pocket", "circular_pocket"} or family in {FeatureFamily.POCKET.value, FeatureFamily.CIRCULAR_POCKET.value}:
        return "pocket"
    if entity == "chamfer" or family == FeatureFamily.CHAMFER_FORM.value:
        return "chamfer"
    if entity in {"bore", "borehole", "core_bore_cylinder_candidate"} or family == FeatureFamily.BORE.value:
        return "bore"

    # Gate strings are fallback hints only.  They may contain legacy "bore"
    # wording even for pocket operations, so use them after explicit entity/family.
    if "pocket" in gate:
        return "pocket"
    if "chamfer" in gate:
        return "chamfer"
    if "bore" in gate:
        return "bore"
    return family or entity or "unknown"


# -----------------------------------------------------------------------------
# Public result and internal pipeline data models
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RebuildResult:
    """Result of deleting a Bore feature patch and inserting replacement faces."""

    mesh: trimesh.Trimesh
    removed_face_ids: tuple[int, ...]
    added_face_ids: tuple[int, ...]
    added_faces: tuple[tuple[int, int, int], ...]
    loop0_vertices: tuple[int, ...]
    loop1_vertices: tuple[int, ...]
    axis: Vector3
    radius: float
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def removed_face_count(self) -> int:
        return len(self.removed_face_ids)

    @property
    def added_face_count(self) -> int:
        return len(self.added_face_ids)


@dataclass(frozen=True, slots=True)
class RebuildCandidateContext:
    """Normalized caller/candidate state for one rebuild request."""

    selected_edge_ids: tuple[int, ...]
    entity_type: str
    rebuild_gate: str
    role: str
    candidate_from_component_engine: bool
    feature_ownership_source: str
    candidate_has_preview_face_patch: bool
    allows_unequal_loop_transition: bool
    quad_density_mode: str
    damaged_bore_rebuild_trial: bool = False


@dataclass(frozen=True, slots=True)
class BoundaryLoopAttempt:
    """One measured loop-pair rebuild attempt for one exact delete patch."""

    source: str
    target_source: str
    face_ids: tuple[int, ...]
    loop0: tuple[int, ...]
    loop1: tuple[int, ...]
    boundary_loop_count: int
    exact_two_loop_patch: bool
    protected_loop_pair: bool
    axial_separation: float
    min_required_axial_separation: float
    unequal_loop_transition_allowed: bool
    boundary_loop_vertex_count_delta: int
    loop_summaries: tuple[dict[str, object], ...] = ()


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def delete_and_rebuild_candidate_region(
    mesh: trimesh.Trimesh,
    edge_ids: Iterable[int],
    *,
    region_face_ids: Iterable[int] | None = None,
    feature_candidate_metadata: Mapping[str, object] | None = None,
    color_rebuilt_faces: bool = True,
    base_face_color: RGBA = DEFAULT_BASE_FACE_COLOR,
    rebuilt_face_color: RGBA = DEFAULT_REBUILT_FACE_COLOR,
    isolate_rebuilt_vertices_for_color: bool = False,
    allow_diagnostic_preview_rebuild: bool = False,
    quad_density_mode: str | None = None,
) -> RebuildResult:
    """Delete the selected recognized Bore feature patch and rebuild it.

    All candidate sources enter the same measured-patch pipeline. The public
    signature uses canonical RegionData naming; no old BoreWall wrapper remains.
    """

    # Stage 0 — source mesh normalization.
    # Trimesh may keep additional per-face columns in some import paths; the
    # rebuild runtime uses triangular faces, so all downstream topology operates
    # on the first three vertex indices.
    _validate_mesh(mesh)

    vertices = np.asarray(mesh.vertices, dtype=float)[:, :3]
    source_faces = np.asarray(mesh.faces, dtype=np.int64)
    if source_faces.ndim != 2 or source_faces.shape[1] < 3:
        raise ValueError("Bore rebuild requires triangular runtime faces.")
    source_faces = source_faces[:, :3].astype(np.int64, copy=False)

    selected_edge_ids = tuple(int(v) for v in tuple(edge_ids or ()))
    if not selected_edge_ids:
        raise ValueError("No selected Bore rim edges for rebuild.")

    # Stage 1 — normalize UI/candidate metadata into one small context object.
    # This removes branching around where the candidate came from.  A previewed
    # BOREHOLE and a previewed CHAMFER enter the same measured-patch pipeline;
    # only their topology permissions differ.
    candidate_metadata = dict(feature_candidate_metadata or {})
    context = _candidate_context(
        selected_edge_ids=selected_edge_ids,
        candidate_metadata=candidate_metadata,
        region_face_ids=tuple(region_face_ids or ()) if region_face_ids is not None else (),
        explicit_quad_density_mode=quad_density_mode,
    )
    canonical_feature_family = _canonical_feature_family_from_metadata(
        candidate_metadata,
        entity_type=context.entity_type,
        rebuild_gate=context.rebuild_gate,
    )

    # Stage 2 — collect RegionData anchor evidence.  region_select remains a RegionData collector only; the selected CandidateData
    # face IDs below still define the rebuild request.
    region_data = select_region_data(mesh, selected_edge_ids)
    region_data_diagnostics = dict(getattr(region_data, "diagnostics", {}) or {})

    if region_face_ids is None:
        raise ValueError(
            "Bore rebuild requires explicit CandidateData rebuild_face_ids. "
            "RegionData.face_ids are neutral AOI evidence and cannot be used as rebuild input."
        )
    initial_face_ids = _normalize_face_ids(region_face_ids, face_count=len(source_faces))
    if not initial_face_ids:
        raise ValueError("No CandidateData rebuild_face_ids were provided for rebuild.")

    axis = _unit_vector(getattr(region_data, "axis", (0.0, 0.0, 1.0)))
    radius = _safe_float(getattr(region_data, "radius", 0.0), 0.0)
    # RegionData may describe a broad AOI around the selected rim, especially
    # after compound POCKET -> child BORE operations.  It is anchor evidence,
    # not delete-patch or protected-loop authority.  The generic rebuild path
    # therefore starts with no protected-loop override; BORE/CHAMFER/POCKET
    # loop attempts must come from the accepted CandidateData-owned patch.
    protected_loop_pair: tuple[tuple[int, ...], tuple[int, ...]] | None = None

    # POCKET v99: keep the meaning transform explicit.  A POCKET candidate is
    # not a BORE sleeve candidate.  Recognition has already created a POCKET
    # hypothesis and owned floor/side-wall roles.  Rebuild Target may therefore
    # request a pocket-native recessed cup rebuild: delete owned floor + owned
    # side wall, regenerate side wall + floor, keep the top opening open, and
    # exclude transition/chamfer evidence.  Only an explicit cap/flatten action
    # enters the old top-cap path.
    if _candidate_requests_pocket_cap_rebuild(context=context, candidate_metadata=candidate_metadata):
        return _delete_pocket_and_fill_opening_cap(
            mesh=mesh,
            vertices=vertices,
            source_faces=source_faces,
            face_ids=initial_face_ids,
            selected_edge_ids=selected_edge_ids,
            candidate_metadata=candidate_metadata,
            context=context,
            axis=axis,
            radius=radius,
            color_rebuilt_faces=bool(color_rebuilt_faces),
            base_face_color=base_face_color,
            rebuilt_face_color=rebuilt_face_color,
        )
    if _candidate_requests_pocket_recess_cup_rebuild(context=context, candidate_metadata=candidate_metadata):
        return _delete_and_rebuild_pocket_recess_cup(
            mesh=mesh,
            vertices=vertices,
            source_faces=source_faces,
            face_ids=initial_face_ids,
            selected_edge_ids=selected_edge_ids,
            candidate_metadata=candidate_metadata,
            context=context,
            axis=axis,
            radius=radius,
            color_rebuilt_faces=bool(color_rebuilt_faces),
            base_face_color=base_face_color,
            rebuilt_face_color=rebuilt_face_color,
        )

    extra_candidate_face_sets: list[tuple[str, tuple[int, ...]]] = []
    # v33 semantic boundary: RegionData remains neutral AOI/anchor evidence for
    # loop protection and measurement only.  It is not offered as a candidate
    # face pool to rebuild_target.py.
    region_data_face_ids = _normalize_face_ids(getattr(region_data, "face_ids", ()), face_count=len(source_faces))

    # v1.6.4 damaged-BORE correction: once Recognition has emitted accepted
    # two-opening BoreWallOwnership, RebuildTarget may use the measured
    # two-opening frame as bounded target evidence.  This is not a RegionData
    # fallback: the candidate frame constrains the target, RegionData only
    # supplies the local search pool.  It fixes damaged bores where the selected
    # owned wall fragment is only a half-depth surviving strip and therefore
    # has no usable boundary-loop pair by itself.
    candidate_frame = _two_opening_frame_from_candidate_metadata(candidate_metadata)
    if candidate_frame is not None:
        axis = _unit_vector(candidate_frame.get("axis", axis), fallback=axis)
        candidate_protected = _protected_loop_pair_from_candidate_frame(
            vertices=vertices,
            source_faces=source_faces,
            frame=candidate_frame,
            # v172g: frame-derived protected loops are searched only inside
            # CandidateData-owned rebuild faces.  RegionData is deliberately
            # not a loop pool here; otherwise a parent POCKET/child BORE hard
            # transition can be mistaken for a BORE endpoint.
            preferred_pool=initial_face_ids,
        )
        if candidate_protected is not None:
            protected_loop_pair = candidate_protected

    # v172g semantic invariant: CandidateData.rebuild_face_ids already carry
    # the owned delete-patch meaning for normal accepted candidates.  Older
    # damaged-bore repair code could synthesize a larger full-depth target from
    # the measured frame and RegionData pool.  That converted relationship or
    # context geometry into BORE ownership and caused parent POCKET / child BORE
    # transitions to be swallowed.  Expanded targets may return only as a future
    # explicit RebuildTarget semantic object, not as an implicit rebuild.py
    # fallback.
    region_data_diagnostics["v172g_candidate_frame_expansion_disabled"] = True

    # Stage 3 — ask rebuild_target for bounded delete-patch proposals.  The
    # proposals are not accepted here; each proposal must still survive a full
    # measured-loop replacement trial as a watertight mesh.
    topology_seal_callback = _topology_seal_callback_for_context(context)

    target_result = build_bounded_rebuild_target_face_sets(
        source_faces=source_faces,
        initial_face_ids=initial_face_ids,
        protected_loop_pair=protected_loop_pair,
        extra_candidate_face_sets=tuple(extra_candidate_face_sets),
        preview_candidate_patch_owns_delete=bool(context.candidate_has_preview_face_patch),
        topology_seal_callback=None,
        protected_fragment_bridge_callback=None,
    )
    if not bool(target_result.get("valid", False)):
        raise ValueError(
            "Bore rebuild target construction failed. "
            f"diagnostics={dict(target_result.get('diagnostics', {}) or {})}; Geometry changed: no."
        )

    # Convert target proposals into immutable local patch objects.  Each patch is
    # tried independently; the first watertight trial wins.
    target_patches = target_patches_from_result(target_result)
    if not target_patches:
        target_patches = (RebuildTargetPatch("initial_candidate_faces", initial_face_ids),)

    # Damaged imported bores can contain holes, tiny islands, and broken wall
    # fragments.  In that case the candidate patch may have many boundary loops
    # even though the selected bore still has two valid rim loops.  Add a
    # conservative fallback target that swallows only same-cylinder defect
    # boundaries between the two protected rims.  This is still rebuild-target
    # policy: Region Select does not classify anything, and Rebuild still has to
    # pass watertight trial validation before geometry can change.
    # v50 semantic boundary: damaged-bore defect handling is now allowed only
    # when Recognition produced a damaged BORE CandidateData row with explicit
    # rebuild_face_ids.  This is still Target/Rebuild policy, not RegionData
    # fallback: the base delete patch remains the candidate-owned bore wall.
    # v172g: no implicit damaged-bore defect-swallow targets for generic
    # accepted CandidateData.  A larger delete patch must be produced upstream
    # as an explicit semantic RebuildTarget object, not inferred from AOI /
    # residual-boundary topology inside rebuild.py.
    damaged_bore_targets: tuple[RebuildTargetPatch, ...] = ()
    region_data_diagnostics["v172g_damaged_bore_defect_swallow_targets_disabled"] = True

    # Keep only the exact CandidateData-owned delete patch in the generic path.
    # This is not an actionability gate; it is the meaning-preservation mapping:
    # CandidateData-owned rebuild faces -> DeletePatchProposal faces.
    initial_set_for_v172g = set(int(v) for v in initial_face_ids)
    target_patches = tuple(
        patch for patch in tuple(target_patches or ())
        if set(int(v) for v in tuple(patch.face_ids or ())) == initial_set_for_v172g
    ) or (RebuildTargetPatch("initial_final_delete_faces", initial_face_ids),)

    # Stage 4 — derive loop-pair attempts from the exact CandidateData-owned
    # patch boundaries.  RegionData selected/AOI loops are no longer accepted
    # as fallback loop authority in the generic path because compound feature
    # relationship boundaries can otherwise be reinterpreted as BORE endpoints.
    attempts = _build_boundary_loop_attempts(
        vertices=vertices,
        source_faces=source_faces,
        target_patches=target_patches,
        axis=axis,
        protected_loop_pair=protected_loop_pair,
        allow_unequal_loop_transition=bool(context.allows_unequal_loop_transition),
    )

    # v172g: attempt face sets must preserve the CandidateData-owned delete
    # patch meaning.  Boundary-loop alternatives are allowed only when they use
    # the same face set; they may choose a different loop pair, but they may not
    # delete extra parent POCKET, CHAMFER, collar, or relationship faces.
    attempts = tuple(
        attempt for attempt in tuple(attempts or ())
        if set(int(v) for v in tuple(attempt.face_ids or ())) == initial_set_for_v172g
    )

    if not attempts:
        raise ValueError(
            "Bore measured-patch rebuild found no usable boundary-loop pair inside the accepted CandidateData delete patch. "
            f"target_sources={tuple(target.source for target in target_patches)}; "
            f"initial_face_count={len(initial_face_ids)}; Geometry changed: no."
        )

    # Stage 5 — plan, trial, and select.  Nothing is returned until the trial
    # mesh has zero boundary edges and Trimesh reports it as watertight.
    selected: tuple[BoundaryLoopAttempt, QuadPlan, dict[str, object]] | None = None
    attempt_summaries: list[dict[str, object]] = []
    best_failure: dict[str, object] = {}

    for attempt_index, attempt in enumerate(attempts):
        try:
            plan = _quad_plan_for_attempt(
                vertices=vertices,
                source_faces=source_faces,
                attempt=attempt,
                axis=axis,
                context=context,
                candidate_metadata=candidate_metadata,
                quad_density_mode=context.quad_density_mode,
            )
            plan = _orient_plan_triangles_to_source_patch(
                vertices=vertices,
                source_faces=source_faces,
                face_ids=attempt.face_ids,
                plan=plan,
            )
            boundary_match = _generated_surface_boundary_match_diagnostics(
                source_faces=source_faces,
                face_ids=attempt.face_ids,
                triangles=plan.triangles,
                source_vertex_count=int(len(vertices)),
            )
            plan_diag_with_boundary = dict(plan.diagnostics)
            plan_diag_with_boundary.update(boundary_match)
            if str(context.entity_type or "").strip().lower() in {"borehole", "bore"}:
                plan_diag_with_boundary.update(
                    _damaged_bore_internal_boundary_swallow_diagnostics_v173n(
                        source_faces=source_faces,
                        face_ids=attempt.face_ids,
                        triangles=plan.triangles,
                        source_vertex_count=int(len(vertices)),
                        loop0=attempt.loop0,
                        loop1=attempt.loop1,
                        boundary_loop_count=int(attempt.boundary_loop_count),
                    )
                )
            plan = QuadPlan(
                generated_vertices=plan.generated_vertices,
                triangles=plan.triangles,
                logical_quads=plan.logical_quads,
                loop0=plan.loop0,
                loop1=plan.loop1,
                center0=plan.center0,
                center1=plan.center1,
                axis=plan.axis,
                diagnostics=plan_diag_with_boundary,
            )
            quality = _validate_plan_geometry_quality(context=context, vertices=vertices, attempt=attempt, plan=plan)
            if not bool(quality.get("valid", False)):
                summary = _attempt_summary(
                    attempt_index,
                    attempt,
                    error="geometry_quality_rejected: " + str(quality.get("reason", "unknown")),
                )
                summary.update({f"quality_{key}": value for key, value in quality.items() if key != "valid"})
                attempt_summaries.append(summary)
                if not best_failure:
                    best_failure = dict(summary)
                continue

            trial = _trial_rebuild(
                vertices=vertices,
                source_faces=source_faces,
                face_ids=attempt.face_ids,
                generated_vertices=plan.generated_vertices,
                triangles=plan.triangles,
            )
            residual_after_v177k = _int_preserve_zero(trial.get("boundary_edge_count_after", -1), -1)
            if bool(context.damaged_bore_rebuild_trial) and 0 < residual_after_v177k <= 24:
                # v51 damaged-bore target seal: the v50 trial proved that the
                # damaged wall candidate can leave a small residual boundary
                # after the normal two-rim measured replacement.  The observed
                # failure had 10 remaining boundary edges, so the previous
                # 6-edge residual loop cap was too strict and only produced a
                # huge rejection diagnostic.  This remains guarded by the
                # damaged_bore_rebuild_trial flag and the final watertight trial.
                seal_triangles, seal_diag = _small_boundary_seal_triangles_for_trial_mesh(
                    source_faces=source_faces,
                    face_ids=attempt.face_ids,
                    generated_vertices=plan.generated_vertices,
                    triangles=plan.triangles,
                    max_boundary_edges=24,
                    max_loop_edges=16,
                )
                plan_diag_with_seal = dict(plan.diagnostics)
                plan_diag_with_seal.update({f"damaged_bore_small_boundary_seal_{key}": value for key, value in dict(seal_diag).items()})
                plan_diag_with_seal["damaged_bore_small_boundary_seal_considered"] = True
                if len(seal_triangles):
                    sealed_triangles = np.vstack([np.asarray(plan.triangles, dtype=np.int64).reshape((-1, 3)), seal_triangles])
                    sealed_trial = _trial_rebuild(
                        vertices=vertices,
                        source_faces=source_faces,
                        face_ids=attempt.face_ids,
                        generated_vertices=plan.generated_vertices,
                        triangles=sealed_triangles,
                    )
                    plan_diag_with_seal["damaged_bore_small_boundary_seal_trial_boundary_edge_count_after"] = int(sealed_trial.get("boundary_edge_count_after", -1))
                    plan_diag_with_seal["damaged_bore_small_boundary_seal_trial_watertight_after"] = bool(sealed_trial.get("watertight_after", False))
                    if int(sealed_trial.get("boundary_edge_count_after", 10**9)) <= int(trial.get("boundary_edge_count_after", 10**9)):
                        plan_diag_with_seal["damaged_bore_small_boundary_seal_used"] = True
                        plan_diag_with_seal["damaged_bore_small_boundary_seal_added_triangle_count"] = int(len(seal_triangles))
                        plan = QuadPlan(
                            generated_vertices=plan.generated_vertices,
                            triangles=sealed_triangles,
                            logical_quads=plan.logical_quads,
                            loop0=plan.loop0,
                            loop1=plan.loop1,
                            center0=plan.center0,
                            center1=plan.center1,
                            axis=plan.axis,
                            diagnostics=plan_diag_with_seal,
                        )
                        trial = sealed_trial
                else:
                    # Keep the rejection reason visible in the attempt summary so
                    # future damaged-bore target work has concrete evidence.
                    plan = QuadPlan(
                        generated_vertices=plan.generated_vertices,
                        triangles=plan.triangles,
                        logical_quads=plan.logical_quads,
                        loop0=plan.loop0,
                        loop1=plan.loop1,
                        center0=plan.center0,
                        center1=plan.center1,
                        axis=plan.axis,
                        diagnostics=plan_diag_with_seal,
                    )
            summary = _attempt_summary(attempt_index, attempt, trial=trial, plan=plan)
            attempt_summaries.append(summary)
            if not best_failure or int(summary.get("boundary_edge_count_after", 10**9)) < int(best_failure.get("boundary_edge_count_after", 10**9)):
                best_failure = dict(summary)
            if _trial_accepts_for_context(context=context, trial=trial, plan=plan):
                selected = (attempt, plan, trial)
                break
        except Exception as exc:
            summary = _attempt_summary(attempt_index, attempt, error=str(exc))
            attempt_summaries.append(summary)
            if not best_failure:
                best_failure = dict(summary)

    if selected is None:
        message = _format_failure_message(
            context=context,
            best_failure=best_failure,
            attempt_summaries=tuple(attempt_summaries),
            target_result=target_result,
        )
        raise ValueError(message)

    selected_attempt, selected_plan, selected_trial = selected
    selected_attempt_set_v172g = set(int(v) for v in tuple(selected_attempt.face_ids or ()))
    if selected_attempt_set_v172g != initial_set_for_v172g:
        raise ValueError(
            "Bore rebuild rejected before mesh mutation: DeletePatchProposal face IDs diverged from accepted CandidateData-owned rebuild_face_ids. "
            f"candidate_rebuild_face_count={len(initial_set_for_v172g)}; "
            f"delete_patch_face_count={len(selected_attempt_set_v172g)}; "
            "Geometry changed: no."
        )

    # Stage 6 — apply the already-validated plan to a fresh mesh object.
    # The active project state is still owned by the caller/UI layer.
    result = _apply_rebuild(
        mesh=mesh,
        vertices=vertices,
        source_faces=source_faces,
        face_ids=selected_attempt.face_ids,
        generated_vertices=selected_plan.generated_vertices,
        triangles=selected_plan.triangles,
        color_rebuilt_faces=bool(color_rebuilt_faces),
        base_face_color=base_face_color,
        rebuilt_face_color=rebuilt_face_color,
    )

    before_face_count = int(len(source_faces))
    result_mesh = result["mesh"]
    after_face_count = int(len(getattr(result_mesh, "faces", ())))
    before_vertex_count = int(len(vertices))
    after_vertex_count = int(len(getattr(result_mesh, "vertices", ())))

    diagnostics: dict[str, object] = {
        "mode": "coherent_measured_patch_rebuild_v9R",
        "topology_policy": "semantic_measured_primitive_generated_vertex_quad_rebuild_v160",
        "pipeline": (
            "candidate_context",
            "bounded_rebuild_target",
            "patch_boundary_loop_attempts",
            "measured_loop_quad_plan",
            "watertight_trial",
            "apply_rebuild",
        ),
        "candidate_entity_type": context.entity_type or "-",
        "candidate_rebuild_gate": context.rebuild_gate,
        "candidate_role": context.role,
        "candidate_from_component_engine": bool(context.candidate_from_component_engine),
        "candidate_feature_ownership_source": context.feature_ownership_source,
        "candidate_has_preview_face_patch": bool(context.candidate_has_preview_face_patch),
        "preview_candidate_patch_owns_delete": bool(context.candidate_has_preview_face_patch),
        "selected_edge_count": int(len(selected_edge_ids)),
        "initial_face_count": int(len(initial_face_ids)),
        "before_face_count": int(before_face_count),
        "after_face_count": int(after_face_count),
        "before_vertex_count": int(before_vertex_count),
        "after_vertex_count": int(after_vertex_count),
        "removed_face_count": int(len(selected_attempt.face_ids)),
        "candidate_rebuild_face_count_v172g": int(len(initial_face_ids)),
        "delete_patch_face_count_v172g": int(len(selected_attempt.face_ids)),
        "delete_patch_equals_candidate_rebuild_faces_v172g": bool(set(int(v) for v in tuple(selected_attempt.face_ids or ())) == set(int(v) for v in tuple(initial_face_ids or ()))),
        "delete_patch_meaning_invariant_v172g": "CandidateData.rebuild_face_ids_are_the_DeletePatchProposal_faces_for_generic_rebuild",
        "target_face_set_sources": tuple(str(target.source) for target in target_patches),
        "selected_target_source": selected_attempt.target_source,
        "selected_attempt_source": selected_attempt.source,
        "selected_attempt_index": int(next((i for i, item in enumerate(attempts) if item is selected_attempt), -1)),
        "attempt_count": int(len(attempts)),
        "attempt_summaries": tuple(attempt_summaries),
        "boundary_loop_count": int(selected_attempt.boundary_loop_count),
        "loop0_vertex_count": int(len(selected_plan.loop0)),
        "loop1_vertex_count": int(len(selected_plan.loop1)),
        "boundary_loop_vertex_count_delta": int(abs(len(selected_plan.loop0) - len(selected_plan.loop1))),
        "unequal_loop_transition_allowed": bool(selected_attempt.unequal_loop_transition_allowed),
        "unequal_loop_transition_used": bool(selected_plan.diagnostics.get("unequal_loop_transition_used", False)),
        "transition_drop_quad_count": int(selected_plan.diagnostics.get("transition_drop_quad_count", 0) or 0),
        "transition_ring_vertex_count": int(selected_plan.diagnostics.get("transition_ring_vertex_count", 0) or 0),
        "logical_quad_count": int(len(selected_plan.logical_quads)),
        "added_logical_quad_count": int(len(selected_plan.logical_quads)),
        "added_triangle_count": int(len(selected_plan.triangles)),
        "added_runtime_triangle_count": int(len(selected_plan.triangles)),
        "actual_added_face_count": int(len(selected_plan.triangles)),
        "added_face_count": int(len(selected_plan.triangles)),
        "colored_rebuilt_face_count": int(len(result["added_face_ids"])),
        "generated_vertex_count": int(len(selected_plan.generated_vertices)),
        "transition_base_count_sequence": tuple(selected_plan.diagnostics.get("transition_base_count_sequence", ()) or ()),
        "transition_count_sequence": tuple(selected_plan.diagnostics.get("transition_count_sequence", ()) or ()),
        "transition_base_band_count": int(selected_plan.diagnostics.get("transition_base_band_count", 0) or 0),
        "transition_target_band_count": int(selected_plan.diagnostics.get("transition_target_band_count", 0) or 0),
        "transition_band_count": int(selected_plan.diagnostics.get("transition_band_count", 0) or 0),
        "transition_band_count_pairs": tuple(selected_plan.diagnostics.get("transition_band_count_pairs", ()) or ()),
        "transition_band_drop_counts": tuple(selected_plan.diagnostics.get("transition_band_drop_counts", ()) or ()),
        "transition_equal_count_spacer_band_count": int(selected_plan.diagnostics.get("transition_equal_count_spacer_band_count", 0) or 0),
        "target_axial_edge_length": float(selected_plan.diagnostics.get("target_axial_edge_length", 0.0) or 0.0),
        "raw_equal_edge_axial_segments": int(selected_plan.diagnostics.get("raw_equal_edge_axial_segments", 0) or 0),
        "axial_segment_cap": int(selected_plan.diagnostics.get("axial_segment_cap", 0) or 0),
        "generated_triangle_normal_flip_count": int(selected_plan.diagnostics.get("normal_flip_count", 0) or 0),
        "generated_triangle_normal_alignment_median": float(selected_plan.diagnostics.get("normal_alignment_median", 0.0) or 0.0),
        "generated_triangle_normal_alignment_min": float(selected_plan.diagnostics.get("normal_alignment_min", 0.0) or 0.0),
        "boundary_edge_count_before": int(selected_trial.get("boundary_edge_count_before", -1)),
        "boundary_edge_count_after": int(selected_trial.get("boundary_edge_count_after", -1)),
        "boundary_edge_count_delta": int(selected_trial.get("boundary_edge_count_delta", 0)),
        "watertight_after": bool(selected_trial.get("watertight_after", False)),
        "local_topology_acceptance_used": bool(not bool(selected_trial.get("watertight_after", False)) and _trial_accepts_for_context(context=context, trial=selected_trial, plan=selected_plan)),
        "local_bore_wall_rebuild_acceptance_v85": bool(str(context.entity_type).strip().lower() == "borehole" and not bool(selected_trial.get("watertight_after", False)) and _trial_accepts_for_context(context=context, trial=selected_trial, plan=selected_plan)),
        "local_pocket_sidewall_rebuild_acceptance_v98": bool(str(context.entity_type).strip().lower() in {"pocket", "circular_pocket"} and not bool(selected_trial.get("watertight_after", False)) and _trial_accepts_for_context(context=context, trial=selected_trial, plan=selected_plan)),
        "local_pocket_sidewall_rebuild_acceptance_v97": bool(str(context.entity_type).strip().lower() in {"pocket", "circular_pocket"} and not bool(selected_trial.get("watertight_after", False)) and _trial_accepts_for_context(context=context, trial=selected_trial, plan=selected_plan)),
        "pocket_local_boundary_exact_acceptance_contract_v98": "owned_side_wall_preserve_recess_delete_patch_boundary_match_exact_no_new_boundary_edges_no_global_boundary_increase_zero_counts_preserved",
        "pocket_local_boundary_exact_acceptance_contract_v97": "owned_side_wall_preserve_recess_delete_patch_boundary_match_exact_no_new_boundary_edges_no_global_boundary_increase",
        "zero_boundary_trial_acceptance_v85": bool(int(selected_trial.get("boundary_edge_count_after", -1) if selected_trial.get("boundary_edge_count_after", -1) is not None else -1) == 0 and bool(selected_trial.get("watertight_after", False))),
        "damaged_bore_internal_defect_boundaries_swallowed_v85": bool(str(context.entity_type).strip().lower() == "borehole" and int(selected_trial.get("boundary_edge_count_after", -1) if selected_trial.get("boundary_edge_count_after", -1) is not None else -1) == 0 and bool(selected_trial.get("watertight_after", False))),
        "damaged_bore_internal_boundary_swallow_used_v173n": bool(selected_plan.diagnostics.get("damaged_bore_internal_boundary_swallow_acceptance_v173n", False)),
        "damaged_bore_boundary_match_scope_v173n": str(selected_plan.diagnostics.get("damaged_bore_boundary_match_scope_v173n", "-")),
        "damaged_bore_protected_boundary_edge_count_v173n": int(selected_plan.diagnostics.get("damaged_bore_protected_boundary_edge_count_v173n", 0) or 0),
        "damaged_bore_swallowed_defect_boundary_edge_count_v173n": int(selected_plan.diagnostics.get("damaged_bore_swallowed_defect_boundary_edge_count_v173n", 0) or 0),
        "quad_density_mode": context.quad_density_mode,
        "quad_plan": dict(selected_plan.diagnostics),
        "rebuild_target_diagnostics": dict(target_result.get("diagnostics", {}) or {}),
        "region_data_diagnostics": region_data_diagnostics,
        "v33_semantic_boundary_hardening_used": True,
        "region_data_as_rebuild_input_enabled": False,
        "topology_seal_callback_enabled": False,
        "damaged_bore_defect_swallow_targets_enabled": False,
        "parameter_fit_used": False,
        "radius_used_for_delete_expansion": False,
        "axis_used_for_delete_expansion": False,
        "radius_used_for_vertex_placement": bool(selected_plan.diagnostics.get("radius_used_for_vertex_placement", False)),
        "axis_used_for_vertex_placement": bool(selected_plan.diagnostics.get("axis_used_for_vertex_placement", False)),
        "candidate_measured_primitive_used_for_vertex_placement": bool(selected_plan.diagnostics.get("candidate_measured_primitive_used_for_vertex_placement", False)),
        "semantic_geometric_rebuild_used_v160": bool(selected_plan.diagnostics.get("semantic_geometric_rebuild_used_v160", False)),
        "semantic_geometric_rebuild_policy_v160": str(selected_plan.diagnostics.get("semantic_geometric_rebuild_policy_v160", "-")),
        "semantic_geometric_generated_vertex_count_v160": int(selected_plan.diagnostics.get("semantic_geometric_generated_vertex_count_v160", 0) or 0),
        "semantic_geometric_ring_count_sequence_v160": tuple(selected_plan.diagnostics.get("semantic_geometric_ring_count_sequence_v160", ()) or ()),
        "semantic_geometric_band_count_pairs_v160": tuple(selected_plan.diagnostics.get("semantic_geometric_band_count_pairs_v160", ()) or ()),
        "semantic_geometric_boundary_vertices_locked_v160": bool(selected_plan.diagnostics.get("semantic_geometric_boundary_vertices_locked_v160", True)),
        "constraint_bound_feature_remesh_used_v162": bool(selected_plan.diagnostics.get("constraint_bound_feature_remesh_used_v162", False)),
        "constraint_bound_feature_remesh_used": bool(selected_plan.diagnostics.get("constraint_bound_feature_remesh_used", selected_plan.diagnostics.get("constraint_bound_feature_remesh_used_v162", False))),
        "semantic_feature_remesh_contract_v162": str(selected_plan.diagnostics.get("semantic_feature_remesh_contract_v162", SEMANTIC_FEATURE_REMESH_CONTRACT_V162)),
        "semantic_feature_remesh_contract": str(selected_plan.diagnostics.get("semantic_feature_remesh_contract", selected_plan.diagnostics.get("semantic_feature_remesh_contract_v162", SEMANTIC_FEATURE_REMESH_CONTRACT_V162))),
        "semantic_feature_family_v162": str(canonical_feature_family),
        "semantic_feature_family": str(canonical_feature_family),
        "semantic_feature_family_plan_reported": str(selected_plan.diagnostics.get("semantic_feature_family", selected_plan.diagnostics.get("semantic_feature_family_v162", context.entity_type or "unknown"))),
        "semantic_constraint_model_v162": str(selected_plan.diagnostics.get("semantic_constraint_model_v162", "-")),
        "semantic_constraint_model": str(selected_plan.diagnostics.get("semantic_constraint_model", selected_plan.diagnostics.get("semantic_constraint_model_v162", "-"))),
        "semantic_radius_authority_source": str(selected_plan.diagnostics.get("semantic_radius_authority_source", selected_plan.diagnostics.get("v163_constraint_radius_authority_source", "-"))),
        "bore_wall_cylinder_shape_authority_used_v173x": bool(selected_plan.diagnostics.get("bore_wall_cylinder_shape_authority_used_v173x", False)),
        "bore_wall_shape_authority_radius_v173x": float(selected_plan.diagnostics.get("bore_wall_shape_authority_radius_v173x", 0.0) or 0.0),
        "bore_wall_shape_authority_rule_v173x": str(selected_plan.diagnostics.get("bore_wall_shape_authority_rule_v173x", "")),
        "bore_unified_owned_wall_shape_frame_v173y": bool(selected_plan.diagnostics.get("bore_unified_owned_wall_shape_frame_v173y", False)),
        "bore_unified_shape_frame_rule_v173y": str(selected_plan.diagnostics.get("bore_unified_shape_frame_rule_v173y", "")),
        "bore_owned_wall_full_axis_authority_v173z": bool(selected_plan.diagnostics.get("bore_owned_wall_full_axis_authority_v173z", False)),
        "bore_owned_wall_full_cylinder_frame_valid_v173z": bool(selected_plan.diagnostics.get("bore_owned_wall_full_cylinder_frame_valid_v173z", False)),
        "bore_owned_wall_boundary_loop_frame_reordered_v173z": bool(selected_plan.diagnostics.get("bore_owned_wall_boundary_loop_frame_reordered_v173z", False)),
        "bore_owned_wall_shape_phase_offset_v173z": float(selected_plan.diagnostics.get("bore_owned_wall_shape_phase_offset_v173z", 0.0) or 0.0),
        "bore_owned_wall_shape_authority_rule_v173z": str(selected_plan.diagnostics.get("bore_owned_wall_shape_authority_rule_v173z", "")),
        "bore_v166_endpoint_safe_blend_disabled_for_unified_frame_v173y": bool(selected_plan.diagnostics.get("bore_v166_endpoint_safe_blend_disabled_for_unified_frame_v173y", False)),
        "bore_v165_boundary_adapter_disabled_for_unified_frame_v173y": bool(selected_plan.diagnostics.get("bore_v165_boundary_adapter_disabled_for_unified_frame_v173y", False)),
        "locked_boundary_angular_density_guard_used": bool(selected_plan.diagnostics.get("locked_boundary_angular_density_guard_used", False)),
        "endpoint_safe_bore_center_blend_used": bool(selected_plan.diagnostics.get("endpoint_safe_bore_center_blend_used", selected_plan.diagnostics.get("v166_endpoint_safe_bore_blend_used", False))),
        "chamfer_local_rail_curvature_adapter_used": bool(selected_plan.diagnostics.get("chamfer_local_rail_curvature_adapter_used", selected_plan.diagnostics.get("v167_chamfer_local_rail_curvature_adapter_used", False))),
        "chamfer_locked_boundary_density_guard_used": bool(selected_plan.diagnostics.get("chamfer_locked_boundary_density_guard_used", selected_plan.diagnostics.get("v168_chamfer_boundary_locked_angular_density_deferred", False))),
        "generated_vertices_follow_measured_constraints_v162": bool(selected_plan.diagnostics.get("generated_vertices_follow_measured_constraints_v162", False)),
        "semantic_constraint_expected_radial_improvement_v162": bool(selected_plan.diagnostics.get("semantic_constraint_expected_radial_improvement_v162", False)),
        "semantic_constraint_boundary_radius_spread_before_v162": float(selected_plan.diagnostics.get("semantic_constraint_boundary_radius_spread_before_v162", 0.0) or 0.0),
        "semantic_constraint_target_radius_spread_v162": float(selected_plan.diagnostics.get("semantic_constraint_target_radius_spread_v162", 0.0) or 0.0),
        "existing_boundary_vertices_moved": int(selected_plan.diagnostics.get("existing_boundary_vertices_moved", 0) or 0),
        "isolate_rebuilt_vertices_for_color_requested": bool(isolate_rebuilt_vertices_for_color),
        "allow_diagnostic_preview_rebuild_requested": bool(allow_diagnostic_preview_rebuild),
    }

    return RebuildResult(
        mesh=result["mesh"],
        removed_face_ids=selected_attempt.face_ids,
        added_face_ids=result["added_face_ids"],
        added_faces=tuple(tuple(int(v) for v in tri) for tri in selected_plan.triangles.tolist()),
        loop0_vertices=selected_plan.loop0,
        loop1_vertices=selected_plan.loop1,
        axis=_to_vector3(selected_plan.axis),
        radius=float(radius),
        diagnostics=diagnostics,
    )



# -----------------------------------------------------------------------------
# POCKET v99: recessed cup delete/rebuild
# -----------------------------------------------------------------------------


# v176j: POCKET recess-cup request gate moved to rebuild_pocket.py as
# candidate_requests_pocket_recess_cup_rebuild_v176j and imported above under
# the legacy private name to preserve existing call sites.


def _delete_and_rebuild_pocket_recess_cup(
    *,
    mesh: trimesh.Trimesh,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    selected_edge_ids: tuple[int, ...],
    candidate_metadata: Mapping[str, object],
    context: RebuildCandidateContext,
    axis: np.ndarray,
    radius: float,
    color_rebuilt_faces: bool,
    base_face_color: RGBA,
    rebuilt_face_color: RGBA,
) -> RebuildResult:
    """Delete owned pocket floor + side wall and rebuild the recessed cup.

    Semantic contract:
        accepted POCKET CandidateData
            -> owned pocket side-wall faces + owned pocket floor faces
            -> pocket recess-cup DeletePatchProposal
            -> generated side-wall surface + generated floor surface
            -> local topology validation

    This path is neither a BORE wall sleeve nor a parent-surface cap.  The top
    opening remains open.  The bottom floor is regenerated at the measured floor
    boundary.  Transition/chamfer faces stay outside the delete patch unless
    Recognition explicitly owns them under a later transition role.
    """

    canonical_feature_family = _canonical_feature_family_from_metadata(
        candidate_metadata,
        entity_type=context.entity_type,
        rebuild_gate=context.rebuild_gate,
    )

    faces_arr = np.asarray(source_faces, dtype=np.int64)[:, :3]
    wall_ids = _normalize_face_ids(candidate_metadata.get("pocket_side_wall_face_ids", ()), face_count=len(faces_arr))
    floor_ids = _normalize_face_ids(candidate_metadata.get("pocket_floor_face_ids", ()), face_count=len(faces_arr))
    if not wall_ids or not floor_ids:
        raise ValueError(
            "POCKET recess-cup rebuild requires explicit owned pocket_side_wall_face_ids and pocket_floor_face_ids. "
            f"wall_count={len(wall_ids)}; floor_count={len(floor_ids)}; Geometry changed: no."
        )

    patch_faces = tuple(sorted({int(fid) for fid in tuple(wall_ids + floor_ids) if 0 <= int(fid) < len(faces_arr)}))
    explicit_request = _normalize_face_ids(face_ids, face_count=len(faces_arr))
    if explicit_request and not set(patch_faces).issubset(set(explicit_request)):
        # CandidateView should normally pass exactly the owned floor+wall faces.
        # Keep the operation safe by refusing a mismatched delete patch instead
        # of silently expanding from unrelated RegionData.
        raise ValueError(
            "POCKET recess-cup rebuild request does not match owned POCKET roles. "
            f"request_face_count={len(explicit_request)}; owned_floor_plus_wall_count={len(patch_faces)}; Geometry changed: no."
        )

    floor_boundary_loops = _candidate_patch_boundary_edge_loops(source_faces=faces_arr, face_ids=floor_ids)
    wall_boundary_loops = _candidate_patch_boundary_edge_loops(source_faces=faces_arr, face_ids=wall_ids)
    combined_boundary_loops = _candidate_patch_boundary_edge_loops(source_faces=faces_arr, face_ids=patch_faces)
    if not floor_boundary_loops:
        raise ValueError(
            "POCKET recess-cup rebuild could not derive an owned floor perimeter loop. "
            f"floor_face_count={len(floor_ids)}; wall_face_count={len(wall_ids)}; "
            f"wall_boundary_loop_count={len(wall_boundary_loops)}; combined_boundary_loop_count={len(combined_boundary_loops)}; "
            "Geometry changed: no."
        )

    loop_roles_v176e = resolve_pocket_recess_loop_roles_v176e(
        vertices=vertices,
        axis=axis,
        wall_boundary_loops=wall_boundary_loops,
        floor_boundary_loops=floor_boundary_loops,
        combined_boundary_loops=combined_boundary_loops,
    )
    if not bool(loop_roles_v176e.get("valid", False)):
        reason = str(loop_roles_v176e.get("reason", "loop_roles_not_resolved"))
        if reason == "missing_floor_boundary_loops":
            raise ValueError(
                "POCKET recess-cup rebuild could not derive an owned floor perimeter loop. "
                f"floor_face_count={len(floor_ids)}; wall_face_count={len(wall_ids)}; "
                f"wall_boundary_loop_count={len(wall_boundary_loops)}; combined_boundary_loop_count={len(combined_boundary_loops)}; "
                "Geometry changed: no."
            )
        raise ValueError(
            "POCKET recess-cup rebuild could not separate parent opening loop from floor loop. "
            f"wall_face_count={len(wall_ids)}; floor_face_count={len(floor_ids)}; "
            f"wall_boundary_loop_count={len(wall_boundary_loops)}; floor_boundary_loop_count={len(floor_boundary_loops)}; "
            f"combined_boundary_loop_count={len(combined_boundary_loops)}; "
            f"loop_resolution_source={loop_roles_v176e.get('loop_resolution_source', '-')}; "
            "Geometry changed: no."
        )

    floor_loop_record = loop_roles_v176e["floor_loop_record"]
    protected_floor_hole_records = tuple(loop_roles_v176e.get("protected_floor_hole_records", ()) or ())
    if len(protected_floor_hole_records) > 1:
        raise ValueError(
            "POCKET compound recess-cup rebuild currently supports one protected child-BORE opening per pocket floor. "
            f"protected_opening_count={len(protected_floor_hole_records)}; Geometry changed: no."
        )

    bottom_record = loop_roles_v176e["bottom_record"]
    top_record = loop_roles_v176e["top_record"]
    top_loop = tuple(int(v) for v in tuple(loop_roles_v176e.get("top_loop", ()) or ()))
    bottom_loop = tuple(int(v) for v in tuple(loop_roles_v176e.get("bottom_loop", ()) or ()))
    # v177i: v177h introduced an explicit owned-floor cap loop guard but
    # accidentally referenced floor_loop_vertices without binding it in this
    # scope.  The loop-role resolver already returns floor_loop_vertices as
    # explicit owned-floor perimeter evidence; fall back to the record only for
    # defensive compatibility.
    floor_loop_vertices = tuple(int(v) for v in tuple(loop_roles_v176e.get("floor_loop_vertices", ()) or ()))
    if not floor_loop_vertices and isinstance(floor_loop_record, Mapping):
        floor_loop_vertices = tuple(int(v) for v in tuple(floor_loop_record.get("vertices", ()) or ()))
    loop_resolution_source = str(loop_roles_v176e.get("loop_resolution_source", "-"))
    loop_resolution_status = str(loop_roles_v176e.get("loop_resolution_status", "-"))
    wall_loop_summaries_v176e = tuple(loop_roles_v176e.get("wall_loop_summaries", ()) or ())

    if len(top_loop) < 3 or len(bottom_loop) < 3:
        raise ValueError(
            "POCKET recess-cup rebuild found invalid top/floor loops. "
            f"top_loop_vertices={len(top_loop)}; bottom_loop_vertices={len(bottom_loop)}; Geometry changed: no."
        )

    axis_vec = _unit_vector(axis)
    center_top = _loop_center(vertices, top_loop)
    center_bottom = _loop_center(vertices, bottom_loop)
    axial_separation = abs(float(np.dot(center_bottom - center_top, axis_vec)))
    min_sep = _minimum_loop_pair_separation(vertices, top_loop, bottom_loop)
    if axial_separation <= min_sep:
        # Fall back to the center-to-center vector if RegionData axis was not the
        # useful pocket depth direction.  This remains measurement, not feature
        # classification.
        axis_vec = _unit_vector(center_bottom - center_top, fallback=axis_vec)
        axial_separation = abs(float(np.dot(center_bottom - center_top, axis_vec)))
    if axial_separation <= min_sep:
        raise ValueError(
            "POCKET recess-cup rebuild top/floor loops have no usable depth separation. "
            f"axial_separation={axial_separation}; min_required={min_sep}; Geometry changed: no."
        )

    pocket_input_loop_phase_trace_v175j = pocket_loop_phase_trace_v175j(
        vertices=vertices,
        loop0=top_loop,
        loop1=bottom_loop,
        center0=center_top,
        center1=center_bottom,
        axis=axis_vec,
        label="pocket_input_top_to_floor_loop_pair_before_quad_planning",
    )

    attempt = BoundaryLoopAttempt(
        source="pocket_recess_cup_wall_loop_pair_v99",
        target_source="pocket_recess_cup_owned_floor_plus_sidewall_target_v99",
        face_ids=wall_ids,
        loop0=top_loop,
        loop1=bottom_loop,
        boundary_loop_count=int(len(wall_boundary_loops)),
        exact_two_loop_patch=bool(len(wall_boundary_loops) == 2),
        protected_loop_pair=False,
        axial_separation=float(axial_separation),
        min_required_axial_separation=float(min_sep),
        unequal_loop_transition_allowed=bool(len(top_loop) != len(bottom_loop)),
        boundary_loop_vertex_count_delta=int(abs(len(top_loop) - len(bottom_loop))),
        loop_summaries=tuple(wall_loop_summaries_v176e) + (
            {
                "source": "v144_loop_resolution",
                "status": str(loop_resolution_status),
                "loop_resolution_source": str(loop_resolution_source),
                "wall_boundary_loop_count": int(len(wall_boundary_loops)),
                "floor_boundary_loop_count": int(len(floor_boundary_loops)),
                "combined_boundary_loop_count": int(len(combined_boundary_loops)),
                "top_loop_vertex_count": int(len(top_loop)),
                "bottom_loop_vertex_count": int(len(bottom_loop)),
            },
        ),
    )

    try:
        wall_plan = _quad_plan_for_attempt(
            vertices=vertices,
            source_faces=source_faces,
            attempt=attempt,
            axis=axis_vec,
            context=context,
            candidate_metadata=candidate_metadata,
            quad_density_mode=context.quad_density_mode,
        )
    except ValueError as exc:
        # v177g: POCKET side-wall loops may differ by one locked boundary
        # vertex after previous feature rebuilds or on coarse remeshes.  The
        # generic quad-plan topology rule correctly rejects odd count deltas for
        # an all-quad transition, but POCKET already owns a family-local adaptive
        # sew collar that preserves every locked boundary edge without splitting
        # parent geometry.  Use it only for small count drift; larger deltas still
        # remain clean no-mutation target rejections.
        delta = int(abs(len(top_loop) - len(bottom_loop)))
        max_count = int(max(len(top_loop), len(bottom_loop)))
        small_locked_loop_count_drift = bool(delta > 0 and delta <= 2 and max_count >= 3)
        if small_locked_loop_count_drift:
            sorted_top, sorted_bottom = _sort_loop_pair_by_angle(
                vertices,
                tuple(int(v) for v in top_loop),
                tuple(int(v) for v in bottom_loop),
                center_top,
                center_bottom,
                axis_vec,
            )
            aligned_top, aligned_bottom, alignment_diag = _align_unequal_loop_pair_to_angle_samples(
                vertices=vertices,
                loop0=tuple(int(v) for v in sorted_top),
                loop1=tuple(int(v) for v in sorted_bottom),
                center0=center_top,
                center1=center_bottom,
                axis=axis_vec,
            )
            wall_quads, wall_triangles, collar_diag = _band_faces_between_rings_with_adaptive_sew_collar(
                tuple(int(v) for v in aligned_top),
                tuple(int(v) for v in aligned_bottom),
            )
            if wall_triangles:
                wall_plan = QuadPlan(
                    generated_vertices=np.zeros((0, 3), dtype=float),
                    triangles=np.asarray(wall_triangles, dtype=np.int64).reshape((-1, 3)),
                    logical_quads=tuple(tuple(int(v) for v in q) for q in tuple(wall_quads or ())),
                    loop0=tuple(int(v) for v in aligned_top),
                    loop1=tuple(int(v) for v in aligned_bottom),
                    center0=np.asarray(center_top, dtype=float).reshape(3),
                    center1=np.asarray(center_bottom, dtype=float).reshape(3),
                    axis=np.asarray(axis_vec, dtype=float).reshape(3),
                    diagnostics={
                        **dict(alignment_diag or {}),
                        **dict(collar_diag or {}),
                        "plan_type": "pocket_recess_sidewall_adaptive_locked_loop_sew_collar_v177g",
                        "pocket_sidewall_adaptive_locked_loop_fallback_v177g": True,
                        "pocket_sidewall_adaptive_locked_loop_fallback_reason_v177g": "generic_all_quad_plan_rejected_small_locked_loop_count_drift",
                        "pocket_sidewall_generic_planner_error_v177g": str(exc),
                        "pocket_sidewall_top_loop_count_v177g": int(len(top_loop)),
                        "pocket_sidewall_bottom_loop_count_v177g": int(len(bottom_loop)),
                        "pocket_sidewall_locked_loop_count_delta_v177g": int(delta),
                        "pocket_sidewall_adaptive_boundary_preservation_rule_v177g": "consume_every_locked_top_and_floor_boundary_edge_once_no_boundary_vertex_split",
                        "unequal_loop_transition_used": True,
                        "loop0_vertex_count": int(len(aligned_top)),
                        "loop1_vertex_count": int(len(aligned_bottom)),
                        "circumferential_segments": int(max_count),
                        "original_boundary_circumferential_segments": int(max_count),
                        "axial_segments": 1,
                        "generated_internal_ring_count": 0,
                        "semantic_geometric_rebuild_used_v160": True,
                        "semantic_geometric_rebuild_policy_v160": "pocket_adaptive_locked_loop_sew_collar_v177g",
                        "semantic_geometric_boundary_vertices_locked_v160": True,
                        "semantic_geometric_generated_vertex_count_v160": 0,
                        "constraint_bound_feature_remesh_used_v162": True,
                        "constraint_bound_feature_remesh_used": True,
                        "semantic_feature_remesh_contract_v162": SEMANTIC_FEATURE_REMESH_CONTRACT_V162,
                        "semantic_feature_remesh_contract": SEMANTIC_FEATURE_REMESH_CONTRACT_V162,
                        "semantic_feature_family_v162": str(canonical_feature_family),
                        "semantic_feature_family": str(canonical_feature_family),
                        "semantic_constraint_model_v162": "measured_circular_recess_sidewall_between_locked_mouth_and_floor_boundaries_with_adaptive_sew_collar",
                        "semantic_constraint_model": "measured_circular_recess_sidewall_between_locked_mouth_and_floor_boundaries_with_adaptive_sew_collar",
                        "semantic_radius_authority_source": "pocket_candidate_radius_v163",
                        "v163_constraint_radius_authority_source": "pocket_candidate_radius_v163",
                        "v163_candidate_radius_rejected": False,
                        "parameter_fit_used": False,
                        "candidate_measured_primitive_used_for_vertex_placement": False,
                        "radius_used_for_vertex_placement": False,
                        "axis_used_for_vertex_placement": False,
                        "existing_boundary_vertices_moved": 0,
                        "geometry_source": "pocket_locked_loop_adaptive_zipper_sew_collar_v177g",
                        "interpolation_rule": "adaptive_zipper_between_locked_pocket_wall_boundary_loops_no_boundary_split",
                    },
                )
            else:
                wall_plan = None
        else:
            wall_plan = None
        if wall_plan is None:
            # v146: this is an expected rebuild-target rejection, not a programming
            # error.  Coarse/tessellated pocket ownership can expose top/floor loops
            # with incompatible vertex counts (for example 22 ↔ 29 or 25 ↔ 38).
            # Do not let that propagate as a worker traceback.  Rebuild must reject
            # the target cleanly with Geometry changed: no, preserving the semantic
            # boundary: Recognition may preview weak face ownership, but Rebuild only
            # mutates after loop planning + validation succeed.
            raise BoreRebuildRejected(
                "POCKET rebuild target rejected before mesh mutation: owned pocket wall/floor loops "
                "cannot form a valid quad transition. "
                f"top_loop_vertices={len(top_loop)}; bottom_loop_vertices={len(bottom_loop)}; "
                f"wall_faces={len(wall_ids)}; floor_faces={len(floor_ids)}; delete_faces={len(patch_faces)}; "
                f"loop_resolution_status={loop_resolution_status}; loop_resolution_source={loop_resolution_source}; "
                f"planner_error={exc}; Geometry changed: no."
            ) from None
    wall_plan = _orient_plan_triangles_to_source_patch(
        vertices=vertices,
        source_faces=faces_arr,
        face_ids=wall_ids,
        plan=wall_plan,
    )

    # v177h: the floor cap must be generated from the explicit owned-floor
    # perimeter exported by POCKET recognition / loop-role resolution, not from
    # whichever side-wall loop happens to intersect the current bottom-loop hint
    # after angular planning.  The old heuristic could seal the mouth loop on
    # coarse pockets and still pass watertight validation.
    cap_loop = tuple(int(v) for v in tuple(floor_loop_vertices or ()) if 0 <= int(v) < len(vertices))
    if len(cap_loop) < 3:
        raise BoreRebuildRejected(
            "POCKET rebuild target rejected before mesh mutation: explicit owned floor cap loop is invalid. "
            f"floor_cap_loop_vertices={len(cap_loop)}; floor_faces={len(floor_ids)}; wall_faces={len(wall_ids)}; "
            "Geometry changed: no."
        ) from None
    cap_loop_set = set(int(v) for v in cap_loop)
    plan_loop0_set = set(int(v) for v in tuple(wall_plan.loop0 or ()))
    plan_loop1_set = set(int(v) for v in tuple(wall_plan.loop1 or ()))
    cap_matches_wall_loop = bool(cap_loop_set == plan_loop0_set or cap_loop_set == plan_loop1_set)
    if not cap_matches_wall_loop:
        raise BoreRebuildRejected(
            "POCKET rebuild target rejected before mesh mutation: owned floor cap loop does not match either rebuilt side-wall boundary. "
            f"floor_cap_loop_vertices={len(cap_loop)}; wall_loop0_vertices={len(plan_loop0_set)}; "
            f"wall_loop1_vertices={len(plan_loop1_set)}; floor_faces={len(floor_ids)}; wall_faces={len(wall_ids)}; "
            "Geometry changed: no."
        ) from None

    if protected_floor_hole_records:
        protected_loop = tuple(int(v) for v in tuple(protected_floor_hole_records[0].get("vertices", ()) or ()))
        floor_generated_vertices, floor_triangles, floor_logical_quads, floor_diag = _pocket_floor_annular_quad_grid_from_loops(
            vertices=vertices,
            outer_loop_vertices=cap_loop,
            inner_loop_vertices=protected_loop,
            generated_vertex_offset=int(len(wall_plan.generated_vertices)),
            floor_axis=np.asarray(wall_plan.axis, dtype=float).reshape(3),
            quad_density_mode=context.quad_density_mode,
        )
    else:
        floor_generated_vertices, floor_triangles, floor_logical_quads, floor_diag = _pocket_floor_quad_grid_from_loop(
            vertices=vertices,
            loop_vertices=cap_loop,
            generated_vertex_offset=int(len(wall_plan.generated_vertices)),
            quad_density_mode=context.quad_density_mode,
        )
    floor_diag = dict(floor_diag or {})
    floor_diag.update({
        "pocket_floor_cap_uses_explicit_owned_floor_loop_v177h": True,
        "pocket_floor_cap_loop_vertex_count_v177h": int(len(cap_loop)),
        "pocket_floor_cap_loop_matches_wall_boundary_v177h": True,
        "pocket_floor_cap_loop_source_v177h": "explicit_owned_floor_loop_record",
    })

    floor_triangles = _orient_triangles_to_source_role_normal(
        vertices=vertices,
        generated_vertices=np.vstack([
            np.asarray(wall_plan.generated_vertices, dtype=float).reshape((-1, 3)),
            floor_generated_vertices,
        ]).reshape((-1, 3)),
        triangles=floor_triangles,
        source_faces=faces_arr,
        face_ids=floor_ids,
        fallback_normal=np.asarray(wall_plan.axis, dtype=float).reshape(3),
    )
    wall_plan = _orient_pocket_wall_triangles_by_radial_role(
        vertices=vertices,
        source_faces=faces_arr,
        face_ids=wall_ids,
        plan=wall_plan,
    )
    pocket_planned_loop_phase_trace_v175j = pocket_loop_phase_trace_v175j(
        vertices=vertices,
        loop0=tuple(int(v) for v in tuple(wall_plan.loop0 or ())),
        loop1=tuple(int(v) for v in tuple(wall_plan.loop1 or ())),
        center0=np.asarray(wall_plan.center0, dtype=float).reshape(3),
        center1=np.asarray(wall_plan.center1, dtype=float).reshape(3),
        axis=np.asarray(wall_plan.axis, dtype=float).reshape(3),
        label="pocket_planned_sidewall_loop_pair_after_quad_planning",
    )

    patch_plan_v176l = assemble_pocket_recess_cup_patch_plan_v176l(
        removed_face_ids=patch_faces,
        wall_generated_vertices=np.asarray(wall_plan.generated_vertices, dtype=float).reshape((-1, 3)),
        wall_triangles=np.asarray(wall_plan.triangles, dtype=np.int64).reshape((-1, 3)),
        wall_logical_quads=tuple(wall_plan.logical_quads),
        wall_loop0=tuple(int(v) for v in tuple(wall_plan.loop0 or ())),
        wall_loop1=tuple(int(v) for v in tuple(wall_plan.loop1 or ())),
        wall_center0=np.asarray(wall_plan.center0, dtype=float).reshape(3),
        wall_center1=np.asarray(wall_plan.center1, dtype=float).reshape(3),
        wall_axis=np.asarray(wall_plan.axis, dtype=float).reshape(3),
        wall_diagnostics=dict(wall_plan.diagnostics),
        floor_generated_vertices=floor_generated_vertices,
        floor_triangles=floor_triangles,
        floor_logical_quads=tuple(floor_logical_quads),
        floor_diagnostics=dict(floor_diag),
        cap_loop=cap_loop,
        wall_face_ids=wall_ids,
        floor_face_ids=floor_ids,
        wall_boundary_loop_count=int(len(wall_boundary_loops)),
        floor_boundary_loop_count=int(len(floor_boundary_loops)),
        combined_boundary_loop_count=int(len(combined_boundary_loops)),
        top_loop_source="wall_boundary" if top_record in wall_boundary_loops else "combined_patch_boundary",
        bottom_loop_source="floor_boundary" if bottom_record is floor_loop_record else "wall_boundary",
        loop_resolution_status=str(loop_resolution_status),
        loop_resolution_source=str(loop_resolution_source),
        protected_floor_hole_vertex_counts=tuple(
            int(len(tuple(record.get("vertices", ()) or ())))
            for record in protected_floor_hole_records
        ),
        pocket_input_loop_phase_trace_v175j=pocket_input_loop_phase_trace_v175j,
        pocket_planned_loop_phase_trace_v175j=pocket_planned_loop_phase_trace_v175j,
        radius=float(radius),
    )

    combined_plan = QuadPlan(
        generated_vertices=np.asarray(patch_plan_v176l.generated_vertices, dtype=float).reshape((-1, 3)),
        triangles=np.asarray(patch_plan_v176l.triangles, dtype=np.int64).reshape((-1, 3)),
        logical_quads=tuple(patch_plan_v176l.logical_quads),
        loop0=tuple(int(v) for v in tuple(patch_plan_v176l.loop0 or ())),
        loop1=tuple(int(v) for v in tuple(patch_plan_v176l.loop1 or ())),
        center0=np.asarray(patch_plan_v176l.center0, dtype=float).reshape(3),
        center1=np.asarray(patch_plan_v176l.center1, dtype=float).reshape(3),
        axis=np.asarray(patch_plan_v176l.axis, dtype=float).reshape(3),
        diagnostics=dict(patch_plan_v176l.diagnostics),
    )

    boundary_match = _generated_surface_boundary_match_diagnostics(
        source_faces=faces_arr,
        face_ids=patch_faces,
        triangles=combined_plan.triangles,
        source_vertex_count=int(len(vertices)),
    )
    combined_plan = QuadPlan(
        generated_vertices=combined_plan.generated_vertices,
        triangles=combined_plan.triangles,
        logical_quads=combined_plan.logical_quads,
        loop0=combined_plan.loop0,
        loop1=combined_plan.loop1,
        center0=combined_plan.center0,
        center1=combined_plan.center1,
        axis=combined_plan.axis,
        diagnostics={**dict(combined_plan.diagnostics), **boundary_match},
    )

    trial = _trial_rebuild(
        vertices=vertices,
        source_faces=faces_arr,
        face_ids=patch_faces,
        generated_vertices=combined_plan.generated_vertices,
        triangles=combined_plan.triangles,
    )
    local_accept = _pocket_recess_cup_trial_accepts(trial=trial, plan=combined_plan)
    if not bool(trial.get("watertight_after", False)) and not local_accept:
        raise ValueError(
            "POCKET recess-cup rebuild local validation failed. "
            f"delete_faces={len(patch_faces)}; wall_faces={len(wall_ids)}; floor_faces={len(floor_ids)}; "
            f"boundary_match_exact={combined_plan.diagnostics.get('boundary_match_exact', False)}; "
            f"missing={combined_plan.diagnostics.get('missing_patch_boundary_edge_count', '-')}; "
            f"extra={combined_plan.diagnostics.get('extra_generated_boundary_edge_count', '-')}; "
            f"generated_new_boundary={combined_plan.diagnostics.get('generated_boundary_generated_vertex_edge_count', '-')}; "
            f"boundary_before={trial.get('boundary_edge_count_before', '-')}; boundary_after={trial.get('boundary_edge_count_after', '-')}; "
            "Geometry changed: no."
        )

    result = _apply_rebuild(
        mesh=mesh,
        vertices=vertices,
        source_faces=faces_arr,
        face_ids=patch_faces,
        generated_vertices=combined_plan.generated_vertices,
        triangles=combined_plan.triangles,
        color_rebuilt_faces=bool(color_rebuilt_faces),
        base_face_color=base_face_color,
        rebuilt_face_color=rebuilt_face_color,
    )
    before_face_count = int(len(faces_arr))
    result_mesh = result["mesh"]
    after_face_count = int(len(getattr(result_mesh, "faces", ())))
    pocket_trace_summary_v175k = pocket_rebuild_trace_summary_v175k(combined_plan.diagnostics)
    pocket_base_topology_policy_v175k = "owned_pocket_floor_plus_sidewall_recess_cup_quad_floor_local_validation_v100"
    pocket_base_semantic_policy_v175k = str(combined_plan.diagnostics.get("semantic_geometric_rebuild_policy_v160", "-"))
    pocket_surfaced_topology_policy_v175k = f"{pocket_base_topology_policy_v175k};{pocket_trace_summary_v175k}"
    pocket_surfaced_semantic_policy_v175k = f"{pocket_base_semantic_policy_v175k};{pocket_trace_summary_v175k}"

    diagnostics = {
        "mode": "pocket_recess_cup_rebuild_v100_quad_floor",
        "topology_policy": pocket_surfaced_topology_policy_v175k,
        "topology_policy_base_v175k": pocket_base_topology_policy_v175k,
        "pocket_rebuild_trace_surfacing_checkpoint_v175k": True,
        "pocket_rebuild_trace_surfacing_contract_v175k": "diagnostic_only_copy_v175j_trace_into_visible_rebuild_result_fields_no_geometry_change",
        "pocket_rebuild_trace_summary_v175k": pocket_trace_summary_v175k,
        "pocket_rebuild_trace_surfaced_in_topology_policy_v175k": True,
        "pocket_rebuild_trace_surfaced_in_semantic_policy_v175k": True,
        "pipeline": (
            "accepted_pocket_candidate_data",
            "owned_floor_plus_sidewall_target",
            "sidewall_loop_pair_from_wall_ownership",
            "floor_perimeter_from_floor_ownership",
            "generate_sidewall",
            "generate_floor",
            "local_topology_validation",
            "apply_rebuild",
        ),
        "candidate_entity_type": context.entity_type or "pocket",
        "candidate_rebuild_gate": context.rebuild_gate,
        "candidate_role": context.role,
        "candidate_from_component_engine": bool(context.candidate_from_component_engine),
        "candidate_feature_ownership_source": context.feature_ownership_source,
        "selected_edge_count": int(len(selected_edge_ids)),
        "initial_face_count": int(len(face_ids)),
        "before_face_count": int(before_face_count),
        "after_face_count": int(after_face_count),
        "before_vertex_count": int(len(vertices)),
        "after_vertex_count": int(len(getattr(result_mesh, "vertices", ()))),
        "removed_face_count": int(len(patch_faces)),
        "pocket_removed_side_wall_face_count": int(len(wall_ids)),
        "pocket_removed_floor_face_count": int(len(floor_ids)),
        "pocket_rebuild_operation": "restore_recess_cup",
        "pocket_loop_resolution_status_v144": str(loop_resolution_status),
        "pocket_loop_resolution_source_v144": str(loop_resolution_source),
        "pocket_wall_boundary_loop_count_v144": int(len(wall_boundary_loops)),
        "pocket_floor_boundary_loop_count_v144": int(len(floor_boundary_loops)),
        "pocket_combined_boundary_loop_count_v144": int(len(combined_boundary_loops)),
        "pocket_top_loop_source_v144": "wall_boundary" if top_record in wall_boundary_loops else "combined_patch_boundary",
        "pocket_bottom_loop_source_v144": "floor_boundary" if bottom_record is floor_loop_record else "wall_boundary",
        "pocket_rebuild_target_loop_status": "top_and_floor_loops_resolved",
        "pocket_rebuild_floor_loop_fallback_used_v144": bool(str(loop_resolution_source).endswith("v144")),
        "pocket_rebuild_semantic_contract_v102": "POCKET hypothesis owns floor+side-wall roles; child BORE opening is protected relationship metadata; target rebuilds a recessed cup with solid or annular quad floor",
        "pocket_rebuild_semantic_contract_v100": "POCKET hypothesis owns floor+side-wall roles; target deletes those roles and rebuilds a recessed cup with a quad floor, not a bore sleeve, not a fan floor, and not a flush cap",
        "pocket_floor_role_rebuilt": True,
        "pocket_side_wall_role_rebuilt": True,
        "pocket_top_opening_preserved": True,
        "pocket_transition_faces_excluded": True,
        "quad_density_mode": context.quad_density_mode,
        "logical_quad_count": int(len(combined_plan.logical_quads)),
        "added_logical_quad_count": int(len(combined_plan.logical_quads)),
        "added_triangle_count": int(len(combined_plan.triangles)),
        "added_runtime_triangle_count": int(len(combined_plan.triangles)),
        "actual_added_face_count": int(len(combined_plan.triangles)),
        "added_face_count": int(len(combined_plan.triangles)),
        "colored_rebuilt_face_count": int(len(result["added_face_ids"])),
        "generated_vertex_count": int(len(combined_plan.generated_vertices)),
        "loop0_vertex_count": int(len(combined_plan.loop0)),
        "loop1_vertex_count": int(len(combined_plan.loop1)),
        "boundary_loop_vertex_count_delta": int(abs(len(combined_plan.loop0) - len(combined_plan.loop1))),
        "pocket_floor_cap_loop_vertex_count": int(len(cap_loop)),
        "pocket_floor_cap_added_triangle_count": int(len(floor_triangles)),
        "pocket_floor_added_triangle_count": int(len(floor_triangles)),
        "pocket_floor_logical_quad_count": int(len(floor_logical_quads)),
        "pocket_floor_fill_kind": str(floor_diag.get("pocket_floor_fill_kind", "quad_grid")),
        "pocket_floor_protected_child_bore_opening_count": int(len(protected_floor_hole_records)),
        "pocket_floor_protected_child_bore_loop_vertex_counts": tuple(int(len(tuple(record.get("vertices", ()) or ()))) for record in protected_floor_hole_records),
        "compound_pocket_bore_semantics": "POCKET rebuild protects child BORE floor opening as relationship metadata; BORE remains separate candidate",
        "pocket_wall_added_triangle_count": int(len(wall_plan.triangles)),
        "boundary_edge_count_before": int(trial.get("boundary_edge_count_before", -1)),
        "boundary_edge_count_after": int(trial.get("boundary_edge_count_after", -1)),
        "boundary_edge_count_delta": int(trial.get("boundary_edge_count_delta", 0)),
        "watertight_after": bool(trial.get("watertight_after", False)),
        "local_topology_acceptance_used": bool(local_accept and not bool(trial.get("watertight_after", False))),
        "local_pocket_recess_cup_acceptance_v99": bool(local_accept),
        "quad_plan": dict(combined_plan.diagnostics),
        "parameter_fit_used": False,
        "radius_used_for_delete_expansion": False,
        "axis_used_for_delete_expansion": False,
        "radius_used_for_vertex_placement": bool(combined_plan.diagnostics.get("radius_used_for_vertex_placement", False)),
        "axis_used_for_vertex_placement": bool(combined_plan.diagnostics.get("axis_used_for_vertex_placement", False)),
        "candidate_measured_primitive_used_for_vertex_placement": bool(combined_plan.diagnostics.get("candidate_measured_primitive_used_for_vertex_placement", False)),
        "semantic_geometric_rebuild_used_v160": bool(combined_plan.diagnostics.get("semantic_geometric_rebuild_used_v160", False)),
        "semantic_geometric_rebuild_policy_v160": pocket_surfaced_semantic_policy_v175k,
        "semantic_geometric_rebuild_policy_base_v175k": pocket_base_semantic_policy_v175k,
        "semantic_geometric_generated_vertex_count_v160": int(combined_plan.diagnostics.get("semantic_geometric_generated_vertex_count_v160", 0) or 0),
        "semantic_geometric_ring_count_sequence_v160": tuple(combined_plan.diagnostics.get("semantic_geometric_ring_count_sequence_v160", ()) or ()),
        "semantic_geometric_band_count_pairs_v160": tuple(combined_plan.diagnostics.get("semantic_geometric_band_count_pairs_v160", ()) or ()),
        "semantic_geometric_boundary_vertices_locked_v160": bool(combined_plan.diagnostics.get("semantic_geometric_boundary_vertices_locked_v160", True)),
        "constraint_bound_feature_remesh_used_v162": bool(combined_plan.diagnostics.get("constraint_bound_feature_remesh_used_v162", False)),
        "constraint_bound_feature_remesh_used": bool(combined_plan.diagnostics.get("constraint_bound_feature_remesh_used", combined_plan.diagnostics.get("constraint_bound_feature_remesh_used_v162", False))),
        "semantic_feature_remesh_contract_v162": str(combined_plan.diagnostics.get("semantic_feature_remesh_contract_v162", SEMANTIC_FEATURE_REMESH_CONTRACT_V162)),
        "semantic_feature_remesh_contract": str(combined_plan.diagnostics.get("semantic_feature_remesh_contract", combined_plan.diagnostics.get("semantic_feature_remesh_contract_v162", SEMANTIC_FEATURE_REMESH_CONTRACT_V162))),
        "semantic_feature_family_v162": str(canonical_feature_family),
        "semantic_feature_family": str(canonical_feature_family),
        "semantic_feature_family_plan_reported": str(combined_plan.diagnostics.get("semantic_feature_family", combined_plan.diagnostics.get("semantic_feature_family_v162", context.entity_type or "pocket"))),
        "semantic_constraint_model_v162": str(combined_plan.diagnostics.get("semantic_constraint_model_v162", "-")),
        "semantic_constraint_model": str(combined_plan.diagnostics.get("semantic_constraint_model", combined_plan.diagnostics.get("semantic_constraint_model_v162", "-"))),
        "semantic_radius_authority_source": str(combined_plan.diagnostics.get("semantic_radius_authority_source", combined_plan.diagnostics.get("v163_constraint_radius_authority_source", "-"))),
        "pocket_rebuild_frame_trace_checkpoint_v175j": bool(combined_plan.diagnostics.get("pocket_rebuild_frame_trace_checkpoint_v175j", False)),
        "pocket_rebuild_frame_trace_semantic_contract_v175j": str(combined_plan.diagnostics.get("pocket_rebuild_frame_trace_semantic_contract_v175j", "diagnostic_only")),
        "pocket_radius_authority_source_v175j": str(combined_plan.diagnostics.get("semantic_radius_authority_source", combined_plan.diagnostics.get("v163_constraint_radius_authority_source", "-"))),
        "pocket_radius_authority_family_mismatch_v175j": bool("bore" in str(combined_plan.diagnostics.get("semantic_radius_authority_source", combined_plan.diagnostics.get("v163_constraint_radius_authority_source", ""))).lower()),
        "pocket_input_loop_phase_wave_risk_v175j": bool(combined_plan.diagnostics.get("pocket_input_loop_phase_wave_risk_v175j", False)),
        "pocket_planned_loop_phase_wave_risk_v175j": bool(combined_plan.diagnostics.get("pocket_planned_loop_phase_wave_risk_v175j", False)),
        "pocket_input_loop_phase_trace_v175j": dict(combined_plan.diagnostics.get("pocket_input_loop_phase_trace_v175j", {}) or {}),
        "pocket_planned_loop_phase_trace_v175j": dict(combined_plan.diagnostics.get("pocket_planned_loop_phase_trace_v175j", {}) or {}),
        "pocket_input_loop0_count_v175j": int(combined_plan.diagnostics.get("pocket_input_loop0_count_v175j", 0) or 0),
        "pocket_input_loop1_count_v175j": int(combined_plan.diagnostics.get("pocket_input_loop1_count_v175j", 0) or 0),
        "pocket_planned_loop0_count_v175j": int(combined_plan.diagnostics.get("pocket_planned_loop0_count_v175j", 0) or 0),
        "pocket_planned_loop1_count_v175j": int(combined_plan.diagnostics.get("pocket_planned_loop1_count_v175j", 0) or 0),
        "pocket_planned_loop_best_shift_v175j": int(combined_plan.diagnostics.get("pocket_planned_loop_best_shift_v175j", 0) or 0),
        "pocket_planned_loop_best_shift_degrees_v175j": float(combined_plan.diagnostics.get("pocket_planned_loop_best_shift_degrees_v175j", 0.0) or 0.0),
        "pocket_planned_loop_reversed_v175j": bool(combined_plan.diagnostics.get("pocket_planned_loop_reversed_v175j", False)),
        "pocket_planned_loop_mean_chord_error_v175j": float(combined_plan.diagnostics.get("pocket_planned_loop_mean_chord_error_v175j", 0.0) or 0.0),
        "pocket_planned_loop_max_chord_error_v175j": float(combined_plan.diagnostics.get("pocket_planned_loop_max_chord_error_v175j", 0.0) or 0.0),
        "locked_boundary_angular_density_guard_used": bool(combined_plan.diagnostics.get("locked_boundary_angular_density_guard_used", False)),
        "endpoint_safe_bore_center_blend_used": bool(combined_plan.diagnostics.get("endpoint_safe_bore_center_blend_used", combined_plan.diagnostics.get("v166_endpoint_safe_bore_blend_used", False))),
        "chamfer_local_rail_curvature_adapter_used": bool(combined_plan.diagnostics.get("chamfer_local_rail_curvature_adapter_used", combined_plan.diagnostics.get("v167_chamfer_local_rail_curvature_adapter_used", False))),
        "chamfer_locked_boundary_density_guard_used": bool(combined_plan.diagnostics.get("chamfer_locked_boundary_density_guard_used", combined_plan.diagnostics.get("v168_chamfer_boundary_locked_angular_density_deferred", False))),
        "generated_vertices_follow_measured_constraints_v162": bool(combined_plan.diagnostics.get("generated_vertices_follow_measured_constraints_v162", False)),
        "semantic_constraint_expected_radial_improvement_v162": bool(combined_plan.diagnostics.get("semantic_constraint_expected_radial_improvement_v162", False)),
        "semantic_constraint_boundary_radius_spread_before_v162": float(combined_plan.diagnostics.get("semantic_constraint_boundary_radius_spread_before_v162", 0.0) or 0.0),
        "semantic_constraint_target_radius_spread_v162": float(combined_plan.diagnostics.get("semantic_constraint_target_radius_spread_v162", 0.0) or 0.0),
        "existing_boundary_vertices_moved": int(combined_plan.diagnostics.get("existing_boundary_vertices_moved", 0) or 0),
        **{str(k): v for k, v in boundary_match.items()},
    }
    return RebuildResult(
        mesh=result["mesh"],
        removed_face_ids=patch_faces,
        added_face_ids=result["added_face_ids"],
        added_faces=tuple(tuple(int(v) for v in tri) for tri in combined_plan.triangles.tolist()),
        loop0_vertices=combined_plan.loop0,
        loop1_vertices=combined_plan.loop1,
        axis=_to_vector3(combined_plan.axis),
        radius=float(radius),
        diagnostics=diagnostics,
    )



# -----------------------------------------------------------------------------
# POCKET v96: cap-style pocket delete/rebuild
# -----------------------------------------------------------------------------


def _delete_pocket_and_fill_opening_cap(
    *,
    mesh: trimesh.Trimesh,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    selected_edge_ids: tuple[int, ...],
    candidate_metadata: Mapping[str, object],
    context: RebuildCandidateContext,
    axis: np.ndarray,
    radius: float,
    color_rebuilt_faces: bool,
    base_face_color: RGBA,
    rebuilt_face_color: RGBA,
) -> RebuildResult:
    """Delete owned pocket side-wall+floor faces and fill the opening loop.

    This is intentionally separate from the measured-loop sleeve rebuild.  A
    pocket cleanup target has one surviving parent-surface opening boundary after
    the floor is included in the delete patch.  The correct local rebuild is a
    planar cap bounded by that one loop, not a two-loop wall replacement.
    """

    patch_faces = _normalize_face_ids(face_ids, face_count=len(source_faces))
    if not patch_faces:
        raise ValueError("Pocket cap rebuild received no owned floor/side-wall delete faces. Geometry changed: no.")

    loops = _candidate_patch_boundary_edge_loops(source_faces=source_faces, face_ids=patch_faces)
    if not loops:
        raise ValueError(
            "Pocket cap rebuild could not find a closed opening boundary after deleting owned floor+side-wall faces. "
            f"delete_face_count={len(patch_faces)}; Geometry changed: no."
        )

    # The valid first pocket target is one main parent-surface opening loop.  If
    # small stray components exist, keep the largest loop but report them.  This
    # keeps the trial bounded to the user-selected pocket rather than falling
    # back to RegionData.
    main_loop = max(loops, key=lambda row: int(row.get("edge_count", 0) or 0))
    loop_vertices = tuple(int(v) for v in tuple(main_loop.get("vertices", ()) or ()))
    if len(loop_vertices) < 3:
        raise ValueError(
            "Pocket cap rebuild opening loop has too few vertices. "
            f"loop_vertex_count={len(loop_vertices)}; delete_face_count={len(patch_faces)}; Geometry changed: no."
        )

    normal_hint = _unit_vector(candidate_metadata.get("axis", axis), fallback=axis)
    center = _loop_center(vertices, loop_vertices)
    generated_vertices = np.asarray([center], dtype=float).reshape((1, 3))
    center_index = int(len(vertices))
    triangles = _cap_triangles_for_loop(loop_vertices=loop_vertices, center_index=center_index, vertices=vertices, normal_hint=normal_hint)
    if triangles.size == 0:
        raise ValueError(
            "Pocket cap rebuild failed to generate cap triangles. "
            f"loop_vertex_count={len(loop_vertices)}; Geometry changed: no."
        )

    boundary_match = _generated_surface_boundary_match_diagnostics(
        source_faces=source_faces,
        face_ids=patch_faces,
        triangles=triangles,
        source_vertex_count=int(len(vertices)),
    )
    trial = _trial_rebuild(
        vertices=vertices,
        source_faces=source_faces,
        face_ids=patch_faces,
        generated_vertices=generated_vertices,
        triangles=triangles,
    )

    patch_boundary_edges = _int_preserve_zero(boundary_match.get("patch_boundary_edge_count", -1), -1)
    generated_boundary_edges = _int_preserve_zero(boundary_match.get("generated_boundary_edge_count", -1), -1)
    missing = _int_preserve_zero(boundary_match.get("missing_patch_boundary_edge_count", 1), 1)
    extra = _int_preserve_zero(boundary_match.get("extra_generated_boundary_edge_count", 1), 1)
    before = int(trial.get("boundary_edge_count_before", -1) if trial.get("boundary_edge_count_before", -1) is not None else -1)
    after = int(trial.get("boundary_edge_count_after", -1) if trial.get("boundary_edge_count_after", -1) is not None else -1)
    local_accept = bool(
        bool(boundary_match.get("boundary_match_exact", False))
        and patch_boundary_edges > 0
        and generated_boundary_edges == patch_boundary_edges
        and missing == 0
        and extra == 0
        and before >= 0
        and after <= before
    )
    if not (bool(trial.get("watertight_after", False)) or local_accept):
        raise ValueError(
            "Pocket cap rebuild local topology trial rejected. "
            f"delete_face_count={len(patch_faces)}; loop_count={len(loops)}; loop_vertex_count={len(loop_vertices)}; "
            f"boundary_edge_count_before={before}; boundary_edge_count_after={after}; "
            f"watertight_after={bool(trial.get('watertight_after', False))}; "
            f"boundary_match_exact={bool(boundary_match.get('boundary_match_exact', False))}; "
            f"patch_boundary_edge_count={patch_boundary_edges}; generated_boundary_edge_count={generated_boundary_edges}; "
            f"missing_patch_boundary_edge_count={missing}; extra_generated_boundary_edge_count={extra}; "
            "Geometry changed: no."
        )

    result = _apply_rebuild(
        mesh=mesh,
        vertices=vertices,
        source_faces=source_faces,
        face_ids=patch_faces,
        generated_vertices=generated_vertices,
        triangles=triangles,
        color_rebuilt_faces=bool(color_rebuilt_faces),
        base_face_color=base_face_color,
        rebuilt_face_color=rebuilt_face_color,
    )

    diagnostics: dict[str, object] = {
        "mode": "pocket_cap_rebuild_v96",
        "topology_policy": "owned_pocket_floor_plus_sidewall_delete_single_opening_loop_planar_cap_v96",
        "candidate_entity_type": context.entity_type or "pocket",
        "candidate_rebuild_gate": context.rebuild_gate,
        "candidate_role": context.role,
        "candidate_from_component_engine": bool(context.candidate_from_component_engine),
        "candidate_feature_ownership_source": context.feature_ownership_source,
        "candidate_has_preview_face_patch": bool(context.candidate_has_preview_face_patch),
        "preview_candidate_patch_owns_delete": bool(context.candidate_has_preview_face_patch),
        "selected_edge_count": int(len(selected_edge_ids)),
        "removed_face_count": int(len(patch_faces)),
        "pocket_floor_face_count": int(len(_normalize_face_ids(candidate_metadata.get("pocket_floor_face_ids", ()), face_count=len(source_faces)))),
        "pocket_side_wall_face_count": int(len(_normalize_face_ids(candidate_metadata.get("pocket_side_wall_face_ids", ()), face_count=len(source_faces)))),
        "pocket_transition_face_count": int(len(_normalize_face_ids(candidate_metadata.get("pocket_transition_face_ids", ()), face_count=len(source_faces)))),
        "pocket_cap_loop_count": int(len(loops)),
        "pocket_cap_loop_vertex_count": int(len(loop_vertices)),
        "pocket_cap_loop_edge_count": int(main_loop.get("edge_count", len(loop_vertices)) or len(loop_vertices)),
        "pocket_cap_generated_center_vertex_count": 1,
        "pocket_cap_added_triangle_count": int(len(triangles)),
        "pocket_cap_fill_acceptance_contract_v96": "delete_owned_floor_plus_side_wall_faces_fill_single_parent_opening_loop_no_missing_or_extra_generated_boundary_edges",
        "pocket_cap_local_topology_acceptance_used": bool(local_accept and not bool(trial.get("watertight_after", False))),
        "watertight_after": bool(trial.get("watertight_after", False)),
        "boundary_edge_count_before": before,
        "boundary_edge_count_after": after,
        "boundary_edge_count_delta": int(trial.get("boundary_edge_count_delta", after - before)),
        "quad_plan": {
            "geometry_source": "pocket_opening_boundary_loop_planar_cap",
            "cap_center": _to_vector3(center),
            "cap_normal_hint": _to_vector3(normal_hint),
            **dict(boundary_match),
        },
        "parameter_fit_used": False,
        "radius_used_for_delete_expansion": False,
        "axis_used_for_delete_expansion": False,
        "radius_used_for_vertex_placement": False,
        "axis_used_for_vertex_placement": False,
        "candidate_measured_primitive_used_for_vertex_placement": False,
        "semantic_geometric_rebuild_used_v160": False,
        "semantic_geometric_rebuild_policy_v160": "pocket_cap_planar_fill_not_primitive_sidewall_rebuild",
        "semantic_geometric_generated_vertex_count_v160": 0,
        "semantic_geometric_ring_count_sequence_v160": (),
        "semantic_geometric_band_count_pairs_v160": (),
        "semantic_geometric_boundary_vertices_locked_v160": True,
        "existing_boundary_vertices_moved": 0,
        "region_data_as_rebuild_input_enabled": False,
        "floor_rebuild_method": "floor_is_deleted_with_side_walls_then_opening_is_capped",
    }

    return RebuildResult(
        mesh=result["mesh"],
        removed_face_ids=patch_faces,
        added_face_ids=result["added_face_ids"],
        added_faces=tuple(tuple(int(v) for v in tri) for tri in triangles.tolist()),
        loop0_vertices=loop_vertices,
        loop1_vertices=(),
        axis=_to_vector3(normal_hint),
        radius=float(radius),
        diagnostics=diagnostics,
    )



# v176m: POCKET cap fan triangulation helpers moved to rebuild_pocket.py as
# cap_triangles_for_loop_v176m and imported above under the legacy private name
# to preserve existing call sites.


# -----------------------------------------------------------------------------
# Stage 1: request/candidate normalization
# -----------------------------------------------------------------------------


def _candidate_context(
    *,
    selected_edge_ids: tuple[int, ...],
    candidate_metadata: Mapping[str, object],
    region_face_ids: tuple[int, ...],
    explicit_quad_density_mode: str | None,
) -> RebuildCandidateContext:
    entity_type = str(candidate_metadata.get("entity_type", "") or "").strip().lower()
    rebuild_gate = str(candidate_metadata.get("rebuild_gate", "") or "").strip().lower()
    feature_family = str(candidate_metadata.get("feature_family", "") or "").strip().lower()
    recognition_stage = str(candidate_metadata.get("recognition_stage", "") or "").strip().lower()
    if recognition_stage and recognition_stage != RecognitionStage.ACCEPTED_CANDIDATE.value:
        raise ValueError(
            "Bore rebuild rejected before target construction: CandidateData is not an accepted candidate. "
            f"recognition_stage={recognition_stage!r}; feature_family={feature_family!r}; Geometry changed: no."
        )
    if feature_family and feature_family not in {FeatureFamily.BORE.value, FeatureFamily.CHAMFER_FORM.value, FeatureFamily.POCKET.value, FeatureFamily.CIRCULAR_POCKET.value}:
        raise ValueError(
            "Bore rebuild rejected before target construction: feature family has no rebuild implementation in the active side-wall rebuild trial. "
            f"recognition_stage={recognition_stage!r}; feature_family={feature_family!r}; Geometry changed: no."
        )
    role = str(candidate_metadata.get("role", candidate_metadata.get("rebuild_role", "")) or "").strip().lower()
    ownership = str(
        candidate_metadata.get("feature_ownership_source", candidate_metadata.get("active_candidate_authority", ""))
        or ""
    ).strip().lower()
    from_component = bool(
        ownership.startswith("surface_component_classifier")
        or str(candidate_metadata.get("candidate_id", "") or "").startswith("component_engine.")
        or str(candidate_metadata.get("recognition_rule", "") or "").startswith("connected_")
    )
    is_borehole = bool(entity_type in {"borehole", "core_bore_cylinder_candidate"} or "borehole" in rebuild_gate or "core_bore" in role)
    # Damaged-bore is not inferred from generic wording such as a boundary
    # defect or broad role labels.  Those strings are diagnostics.  A damaged
    # repair path that expands ownership must be emitted as an explicit future
    # RebuildTarget semantic object.  For the current generic rebuild path,
    # CandidateData.rebuild_face_ids remain the delete patch.
    damaged_bore_rebuild_trial = bool(
        is_borehole
        and (
            bool(candidate_metadata.get("damaged_bore_rebuild_trial_enabled", False))
            and bool(candidate_metadata.get("explicit_expanded_delete_patch_semantic_object", False))
        )
    )
    is_chamfer = bool(entity_type == "chamfer" or rebuild_gate == "promoted_chamfer_candidate")
    is_counterbore = bool(entity_type == "counterbore" or "counterbore" in rebuild_gate)
    is_pocket = bool(entity_type in {"pocket", "circular_pocket"} or "pocket" in rebuild_gate)
    allows_unequal = bool(is_borehole or is_chamfer or is_counterbore or is_pocket)
    return RebuildCandidateContext(
        selected_edge_ids=tuple(int(v) for v in selected_edge_ids),
        entity_type=entity_type,
        rebuild_gate=rebuild_gate,
        role=role,
        candidate_from_component_engine=from_component,
        feature_ownership_source=ownership,
        candidate_has_preview_face_patch=bool(region_face_ids),
        allows_unequal_loop_transition=allows_unequal,
        quad_density_mode=_resolve_quad_density_mode(explicit_quad_density_mode, candidate_metadata),
        damaged_bore_rebuild_trial=bool(damaged_bore_rebuild_trial),
    )



# -----------------------------------------------------------------------------
# Stage 2: bounded delete-patch proposals
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Stage 3: boundary-loop attempt generation
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Damaged-bore rebuild-target salvage
# -----------------------------------------------------------------------------


def _damaged_bore_defect_swallow_targets(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    target_patches: tuple[RebuildTargetPatch, ...],
    axis: np.ndarray,
    protected_loop_pair: tuple[tuple[int, ...], tuple[int, ...]] | None,
    entity_type: str,
) -> tuple[RebuildTargetPatch, ...]:
    """Return fallback targets for damaged two-rim bore patches.

    This is deliberately conservative.  It activates only for BOREHOLE rebuilds
    and only when a target patch has many boundary loops.  The selected/protected
    rim pair remains the rebuild boundary.  Extra boundary loops are treated as
    defect holes/islands only when neighboring faces lie in the same cylindrical
    band between those rims.

    The fallback does not accept a rebuild.  It only creates another exact delete
    target for the normal measured-loop trial.  If the final mesh is not
    watertight, the rebuild still rejects with Geometry changed: no.
    """

    if str(entity_type or "").strip().lower() not in {"borehole", "bore"}:
        return ()
    if len(source_faces) == 0 or len(vertices) == 0:
        return ()

    axis_vec = _unit_vector(axis)
    out: list[RebuildTargetPatch] = []
    for target in tuple(target_patches or ()):  # preserve source priority outside
        base_faces = tuple(int(fid) for fid in tuple(target.face_ids or ()) if 0 <= int(fid) < len(source_faces))
        if not base_faces:
            continue

        boundary_loops = _candidate_patch_boundary_edge_loops(source_faces=source_faces, face_ids=base_faces)
        if len(boundary_loops) <= 2:
            continue

        main_pair = _main_two_rim_edge_loop_pair(
            vertices=vertices,
            boundary_loops=boundary_loops,
            axis=axis_vec,
            protected_loop_pair=protected_loop_pair,
        )
        if main_pair is None:
            continue

        loop_a, loop_b = main_pair
        if len(loop_a["vertices"]) < 6 or len(loop_b["vertices"]) < 6:
            continue

        expanded, diag = _expand_patch_to_swallow_defect_boundaries(
            vertices=vertices,
            source_faces=source_faces,
            face_ids=base_faces,
            loop_a=loop_a,
            loop_b=loop_b,
            axis=axis_vec,
        )
        if len(expanded) <= len(base_faces):
            continue

        old_loop_count = int(len(boundary_loops))
        new_loop_count = int(len(_candidate_patch_boundary_edge_loops(source_faces=source_faces, face_ids=expanded)))
        # Keep the target only when it actually simplifies the broken topology or
        # reaches the desired two-rim boundary.  The watertight trial remains the
        # final authority.
        if new_loop_count >= old_loop_count and new_loop_count != 2:
            continue

        source = f"{target.source}_damaged_bore_defect_swallowed_two_rim_patch"
        out.append(
            RebuildTargetPatch(
                source=source,
                face_ids=tuple(sorted({int(fid) for fid in expanded})),
            )
        )
    return tuple(out)


def _candidate_patch_boundary_edge_loops(
    *,
    source_faces: np.ndarray,
    face_ids: Iterable[int],
) -> tuple[dict[str, object], ...]:
    """Return ordered boundary loops with both vertices and edge keys."""

    boundary_edges = set(boundary_edges_for_face_patch(source_faces, face_ids))
    components = edge_loop_components(boundary_edges)
    loops: list[dict[str, object]] = []
    for comp in components:
        edges = {_edge_key(edge) for edge in tuple(comp or ())}
        ordered = _order_closed_edge_loop_vertices(edges)
        vertices = tuple(int(v) for v in tuple(ordered.get("vertices", ()) or ()))
        if bool(ordered.get("closed", False)) and len(vertices) >= 3:
            loops.append({"vertices": vertices, "edges": tuple(sorted(edges)), "edge_count": int(len(edges))})
    loops.sort(key=lambda item: (-len(tuple(item.get("vertices", ()) or ())), tuple(item.get("vertices", ()) or (-1,))[0]))
    return tuple(loops)


def _main_two_rim_edge_loop_pair(
    *,
    vertices: np.ndarray,
    boundary_loops: tuple[dict[str, object], ...],
    axis: np.ndarray,
    protected_loop_pair: tuple[tuple[int, ...], tuple[int, ...]] | None,
) -> tuple[dict[str, object], dict[str, object]] | None:
    """Choose the two large bore rims from noisy patch boundary loops."""

    axis_vec = _unit_vector(axis)
    if protected_loop_pair is not None:
        protected_sets = (set(int(v) for v in protected_loop_pair[0]), set(int(v) for v in protected_loop_pair[1]))
        matches: list[dict[str, object]] = []
        for wanted in protected_sets:
            best: tuple[int, dict[str, object]] | None = None
            for loop in boundary_loops:
                verts = set(int(v) for v in tuple(loop.get("vertices", ()) or ()))
                score = int(len(verts & wanted))
                if score <= 0:
                    continue
                if best is None or score > best[0]:
                    best = (score, loop)
            if best is not None and best[0] >= max(3, int(0.35 * max(len(wanted), 1))):
                matches.append(best[1])
        if len(matches) == 2 and set(matches[0].get("vertices", ()) or ()) != set(matches[1].get("vertices", ()) or ()):
            return (matches[0], matches[1])

    # Fallback: choose the large, similarly sized pair with maximum axial
    # separation.  Tiny 3-vertex defect loops are excluded by size.
    candidates = [loop for loop in boundary_loops if len(tuple(loop.get("vertices", ()) or ())) >= 8]
    if len(candidates) < 2:
        return None
    best_pair: tuple[dict[str, object], dict[str, object]] | None = None
    best_score = -1.0
    for i, loop_a in enumerate(candidates):
        va = tuple(int(v) for v in tuple(loop_a.get("vertices", ()) or ()))
        ca = _loop_center(vertices, va)
        ta = float(np.dot(ca, axis_vec))
        for loop_b in candidates[i + 1:]:
            vb = tuple(int(v) for v in tuple(loop_b.get("vertices", ()) or ()))
            cb = _loop_center(vertices, vb)
            tb = float(np.dot(cb, axis_vec))
            sep = abs(tb - ta)
            size_ratio = min(len(va), len(vb)) / max(float(max(len(va), len(vb))), 1.0)
            if size_ratio < 0.55:
                continue
            score = float(sep) * float(size_ratio) + 0.01 * float(min(len(va), len(vb)))
            if score > best_score:
                best_score = score
                best_pair = (loop_a, loop_b)
    return best_pair


def _expand_patch_to_swallow_defect_boundaries(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    loop_a: Mapping[str, object],
    loop_b: Mapping[str, object],
    axis: np.ndarray,
    max_iterations: int = 32,
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Grow a delete patch across same-cylinder defect boundaries.

    The two main rim loops are protected.  Any other patch boundary loop is a
    potential defect hole/island boundary.  Neighboring kept faces are swallowed
    only when their centroids are inside the axial span and near the cylindrical
    radius implied by the main loops.
    """

    faces_arr = np.asarray(source_faces, dtype=np.int64)[:, :3]
    selected: set[int] = {int(fid) for fid in tuple(face_ids or ()) if 0 <= int(fid) < len(faces_arr)}
    if not selected:
        return (), {"changed": False, "reason": "empty_patch"}

    axis_vec = _unit_vector(axis)
    loop0_vertices = tuple(int(v) for v in tuple(loop_a.get("vertices", ()) or ()))
    loop1_vertices = tuple(int(v) for v in tuple(loop_b.get("vertices", ()) or ()))
    main_edges = set(_loop_edges_from_vertices(loop0_vertices)) | set(_loop_edges_from_vertices(loop1_vertices))
    if not main_edges:
        return tuple(sorted(selected)), {"changed": False, "reason": "no_main_edges"}

    c0 = _loop_center(vertices, loop0_vertices)
    c1 = _loop_center(vertices, loop1_vertices)
    base = (c0 + c1) * 0.5
    axial0 = float(np.dot(c0 - base, axis_vec))
    axial1 = float(np.dot(c1 - base, axis_vec))
    axial_min = min(axial0, axial1)
    axial_max = max(axial0, axial1)
    axial_span = max(abs(axial_max - axial_min), 1.0e-9)
    axial_pad = max(0.5, 0.08 * axial_span)

    rim_points = vertices[np.asarray(tuple(loop0_vertices + loop1_vertices), dtype=np.int64), :3]
    rel = rim_points - base.reshape(1, 3)
    axial_values = rel @ axis_vec.reshape(3)
    radial_vec = rel - axial_values.reshape(-1, 1) * axis_vec.reshape(1, 3)
    rim_radial = np.linalg.norm(radial_vec, axis=1)
    radius = float(np.median(rim_radial[np.isfinite(rim_radial)])) if np.any(np.isfinite(rim_radial)) else 0.0
    if not np.isfinite(radius) or radius <= 1.0e-9:
        return tuple(sorted(selected)), {"changed": False, "reason": "invalid_main_loop_radius"}
    radial_tolerance = max(0.75, 0.42 * max(float(radius), 1.0))

    edge_to_faces = build_edge_to_faces(faces_arr)
    face_centroids = vertices[faces_arr[:, :3]].mean(axis=1)

    def face_in_bore_band(fid: int) -> bool:
        p = face_centroids[int(fid)]
        r = p - base
        t = float(np.dot(r, axis_vec))
        if t < axial_min - axial_pad or t > axial_max + axial_pad:
            return False
        rv = r - t * axis_vec
        rr = float(np.linalg.norm(rv))
        return bool(abs(rr - radius) <= radial_tolerance)

    max_added = max(256, min(8000, int(max(len(selected) * 1.25, 1024))))
    added_total = 0
    iterations = 0
    last_extra_count = -1
    for iteration in range(int(max_iterations)):
        iterations = iteration + 1
        boundary_edges = set(boundary_edges_for_face_patch(faces_arr, selected))
        extra_edges = {edge for edge in boundary_edges if _edge_key(edge) not in main_edges}
        if not extra_edges:
            break
        if len(extra_edges) == last_extra_count and iteration > 2:
            # Avoid burning time on a stable frontier when no more same-band
            # faces can be swallowed.
            pass
        last_extra_count = len(extra_edges)
        additions: set[int] = set()
        for edge in extra_edges:
            for fid in edge_to_faces.get(_edge_key(edge), ()):
                fid = int(fid)
                if fid in selected:
                    continue
                if face_in_bore_band(fid):
                    additions.add(fid)
        if not additions:
            break
        if added_total + len(additions) > max_added:
            remaining = max(0, max_added - added_total)
            additions = set(sorted(additions)[:remaining])
        if not additions:
            break
        selected.update(additions)
        added_total += len(additions)
        if added_total >= max_added:
            break

    final_boundary_loop_count = int(len(_candidate_patch_boundary_edge_loops(source_faces=faces_arr, face_ids=selected)))
    return tuple(sorted(selected)), {
        "changed": bool(added_total > 0),
        "added_face_count": int(added_total),
        "iterations": int(iterations),
        "radius": float(radius),
        "radial_tolerance": float(radial_tolerance),
        "axial_pad": float(axial_pad),
        "final_boundary_loop_count": int(final_boundary_loop_count),
    }


def _build_boundary_loop_attempts(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    target_patches: tuple[RebuildTargetPatch, ...],
    axis: np.ndarray,
    protected_loop_pair: tuple[tuple[int, ...], tuple[int, ...]] | None,
    allow_unequal_loop_transition: bool,
) -> tuple[BoundaryLoopAttempt, ...]:
    attempts: list[BoundaryLoopAttempt] = []
    for target in target_patches:
        attempts.extend(
            _attempts_from_patch_boundary(
                vertices=vertices,
                source_faces=source_faces,
                target=target,
                axis=axis,
                allow_unequal_loop_transition=allow_unequal_loop_transition,
            )
        )
        protected = _protected_attempt_for_target(
            vertices=vertices,
            target=target,
            axis=axis,
            protected_loop_pair=protected_loop_pair,
            allow_unequal_loop_transition=allow_unequal_loop_transition,
        )
        if protected is not None:
            attempts.append(protected)

    def priority(attempt: BoundaryLoopAttempt) -> tuple[int, int, int, float, str]:
        target_rank = _target_source_priority(attempt.target_source)
        kind_rank = 1 if attempt.protected_loop_pair else 0
        exact_rank = 0 if attempt.exact_two_loop_patch else 1
        return (target_rank, kind_rank, exact_rank, -float(attempt.axial_separation), str(attempt.source))

    deduped: list[BoundaryLoopAttempt] = []
    seen: set[tuple[frozenset[int], frozenset[int], frozenset[int]]] = set()
    for attempt in sorted(attempts, key=priority):
        key = (frozenset(attempt.face_ids), frozenset(attempt.loop0), frozenset(attempt.loop1))
        rev = (frozenset(attempt.face_ids), frozenset(attempt.loop1), frozenset(attempt.loop0))
        if key in seen or rev in seen:
            continue
        seen.add(key)
        deduped.append(attempt)
    return tuple(deduped)


def _target_source_priority(source: str) -> int:
    text = str(source or "")
    if text.endswith("_damaged_bore_defect_swallowed_two_rim_patch"):
        return 0
    if text.endswith("_topology_sealed_two_rim_patch"):
        return 1
    if text == "candidate_owned_protected_fragment_bridge":
        return 2
    if text == "initial_final_delete_faces":
        return 3
    if text.startswith("measured_connected_component_"):
        return 4
    return 5


def _attempts_from_patch_boundary(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    target: RebuildTargetPatch,
    axis: np.ndarray,
    allow_unequal_loop_transition: bool,
) -> tuple[BoundaryLoopAttempt, ...]:
    loops = _candidate_patch_boundary_vertex_loops(source_faces=source_faces, face_ids=target.face_ids)
    if len(loops) < 2:
        return ()
    axis = _unit_vector(axis)
    centers = [_loop_center(vertices, loop) for loop in loops]
    axials = [float(np.dot(center, axis)) for center in centers]
    loop_summaries = tuple(
        {
            "index": int(i),
            "vertex_count": int(len(loop)),
            "axial": float(axials[i]),
        }
        for i, loop in enumerate(loops[:12])
    )
    attempts: list[BoundaryLoopAttempt] = []
    for i, loop_a in enumerate(loops):
        for j in range(i + 1, len(loops)):
            loop_b = loops[j]
            transition_ok = _loop_pair_transition_allowed(
                len(loop_a),
                len(loop_b),
                allow_unequal_loop_transition=allow_unequal_loop_transition,
            )
            if not transition_ok:
                continue
            sep = abs(float(axials[j] - axials[i]))
            min_sep = _minimum_loop_pair_separation(vertices, loop_a, loop_b)
            if sep <= min_sep:
                continue
            attempts.append(
                BoundaryLoopAttempt(
                    source=f"{target.source}_patch_boundary_loop_pair_{i}_{j}",
                    target_source=target.source,
                    face_ids=target.face_ids,
                    loop0=tuple(int(v) for v in loop_a),
                    loop1=tuple(int(v) for v in loop_b),
                    boundary_loop_count=int(len(loops)),
                    exact_two_loop_patch=bool(len(loops) == 2),
                    protected_loop_pair=False,
                    axial_separation=float(sep),
                    min_required_axial_separation=float(min_sep),
                    unequal_loop_transition_allowed=bool(len(loop_a) != len(loop_b)),
                    boundary_loop_vertex_count_delta=int(abs(len(loop_a) - len(loop_b))),
                    loop_summaries=loop_summaries,
                )
            )
    return tuple(attempts)


def _protected_attempt_for_target(
    *,
    vertices: np.ndarray,
    target: RebuildTargetPatch,
    axis: np.ndarray,
    protected_loop_pair: tuple[tuple[int, ...], tuple[int, ...]] | None,
    allow_unequal_loop_transition: bool,
) -> BoundaryLoopAttempt | None:
    if protected_loop_pair is None:
        return None
    loop_a, loop_b = protected_loop_pair
    if len(loop_a) < 3 or len(loop_b) < 3:
        return None
    if not _loop_pair_transition_allowed(len(loop_a), len(loop_b), allow_unequal_loop_transition=allow_unequal_loop_transition):
        return None
    axis = _unit_vector(axis)
    center_a = _loop_center(vertices, loop_a)
    center_b = _loop_center(vertices, loop_b)
    sep = abs(float(np.dot(center_b - center_a, axis)))
    min_sep = _minimum_loop_pair_separation(vertices, loop_a, loop_b)
    if sep <= min_sep:
        return None
    return BoundaryLoopAttempt(
        source=f"{target.source}_protected_loop_pair",
        target_source=target.source,
        face_ids=target.face_ids,
        loop0=tuple(int(v) for v in loop_a),
        loop1=tuple(int(v) for v in loop_b),
        boundary_loop_count=-1,
        exact_two_loop_patch=False,
        protected_loop_pair=True,
        axial_separation=float(sep),
        min_required_axial_separation=float(min_sep),
        unequal_loop_transition_allowed=bool(len(loop_a) != len(loop_b)),
        boundary_loop_vertex_count_delta=int(abs(len(loop_a) - len(loop_b))),
        loop_summaries=(),
    )




# -----------------------------------------------------------------------------
# Stage 4: measured-loop quad planning
# -----------------------------------------------------------------------------

















# -----------------------------------------------------------------------------
# Target helper: topology-only local sealing proposal
# -----------------------------------------------------------------------------


def _topology_seal_callback_for_context(context: RebuildCandidateContext):
    """Return the topology-only two-rim seal callback for rebuild-target policy.

    The v12 boundary diagnostics proved the previewed BOREHOLE candidate already
    has the right semantic object, but its exact 566-face patch boundary is not
    directly stitchable: 40 delete-patch boundary edges are missing from the
    generated replacement boundary and 40 wrong same-vertex edges are generated.

    The older working rebuild path did not solve that by changing recognition or
    by projecting vertices.  It allowed rebuild_target.py to create a
    topology-sealed two-rim delete-patch proposal, then accepted it only after a
    full boundary-loop/watertight trial.

    Therefore this callback must not add a second tiny 8..24-face cap for
    component-engine BOREHOLE candidates.  rebuild_target.py already owns the
    candidate-owned absorption limit and rejects broad neighboring-feature
    absorption before any trial.  This callback only performs the local
    topology-only seal inside the budget requested by rebuild_target.py.
    """

    def _callback(
        *,
        source_faces: np.ndarray,
        initial_face_ids: tuple[int, ...],
        protected_loop_pair: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
        max_added_faces: int = 0,
    ) -> dict[str, object] | None:
        # ``max_added_faces`` is accepted for compatibility but deliberately
        # ignored.  Topology sealing is governed by leak-edge closure and the
        # final watertight trial, not a face-count budget.
        return _topologically_seal_patch_to_main_boundary_loops(
            source_faces=source_faces,
            initial_face_ids=initial_face_ids,
            protected_loop_pair=protected_loop_pair,
            max_added_faces=0,
        )

    return _callback


def _topologically_seal_patch_to_main_boundary_loops(
    *,
    source_faces: np.ndarray,
    initial_face_ids: tuple[int, ...],
    protected_loop_pair: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    max_added_faces: int = 0,
) -> dict[str, object] | None:
    """Return a locally sealed two-rim patch proposal.

    Cap-free topology-only seal: absorb all faces adjacent across non-protected
    boundary leaks until the only remaining patch boundary is the protected
    two-rim boundary.  This does not use radius/axis expansion and it does not
    accept the result; the full measured-loop replacement trial still decides.
    """

    patch: set[int] = {int(fid) for fid in tuple(initial_face_ids or ()) if 0 <= int(fid) < len(source_faces)}
    if len(patch) < 3:
        return None

    initial_loops = _candidate_patch_boundary_vertex_loops(source_faces=source_faces, face_ids=tuple(sorted(patch)))
    if protected_loop_pair is not None and len(protected_loop_pair[0]) >= 3 and len(protected_loop_pair[1]) >= 3:
        protected_loops = (
            tuple(int(v) for v in protected_loop_pair[0]),
            tuple(int(v) for v in protected_loop_pair[1]),
        )
    else:
        if len(initial_loops) < 2:
            return None
        protected_loops = tuple(initial_loops[:2])

    protected_edges: set[EdgeKey] = set()
    for loop in protected_loops:
        protected_edges.update(_loop_vertices_to_edges(loop))
    if not protected_edges:
        return None

    edge_to_faces = build_edge_to_faces(source_faces)
    added: set[int] = set()
    iteration_summaries: list[dict[str, object]] = []
    iteration = 0

    while True:
        boundary = set(boundary_edges_for_face_patch(source_faces, patch))
        leak_edges = tuple(sorted(edge for edge in boundary if edge not in protected_edges))
        if not leak_edges:
            break

        candidates: set[int] = set()
        for edge in leak_edges:
            for fid in edge_to_faces.get(edge, ()):
                fid_int = int(fid)
                if fid_int not in patch and 0 <= fid_int < len(source_faces):
                    candidates.add(fid_int)

        if not candidates:
            iteration_summaries.append({
                "iteration": int(iteration),
                "leak_edge_count": int(len(leak_edges)),
                "reason": "no_adjacent_faces",
            })
            break

        chosen = tuple(sorted(candidates))
        before = int(len(patch))
        patch.update(chosen)
        added.update(chosen)
        after = int(len(patch))

        iteration_summaries.append(
            {
                "iteration": int(iteration),
                "leak_edge_count": int(len(leak_edges)),
                "candidate_face_count": int(len(candidates)),
                "added_face_count": int(after - before),
                "total_added_face_count": int(len(added)),
                "policy": "add_all_adjacent_faces_across_non_protected_leak_edges_no_face_count_cap",
            }
        )

        if after <= before:
            iteration_summaries.append({
                "iteration": int(iteration),
                "leak_edge_count": int(len(leak_edges)),
                "reason": "no_topology_progress",
            })
            break
        iteration += 1

    if not added:
        return None

    final_boundary = set(boundary_edges_for_face_patch(source_faces, patch))
    remaining_leaks = tuple(sorted(edge for edge in final_boundary if edge not in protected_edges))
    if remaining_leaks:
        return None

    final_ids = tuple(sorted(patch))
    final_loops = _candidate_patch_boundary_vertex_loops(source_faces=source_faces, face_ids=final_ids)
    return {
        "face_ids": final_ids,
        "added_face_ids": tuple(sorted(added)),
        "added_face_count": int(len(added)),
        "initial_face_count": int(len(initial_face_ids)),
        "final_face_count": int(len(final_ids)),
        "initial_boundary_loop_count": int(len(initial_loops)),
        "final_boundary_loop_count": int(len(final_loops)),
        "initial_boundary_loop_vertex_counts": tuple(int(len(loop)) for loop in initial_loops[:12]),
        "final_boundary_loop_vertex_counts": tuple(int(len(loop)) for loop in final_loops[:12]),
        "protected_boundary_loop_vertex_counts": tuple(int(len(loop)) for loop in protected_loops),
        "iteration_summaries": tuple(iteration_summaries),
        "policy": "cap_free_topology_only_absorb_non_rim_boundary_leaks_until_protected_two_rim_boundary",
        "max_added_faces_ignored": True,
        "parameter_fit_used": False,
        "radius_used_for_delete_expansion": False,
        "axis_used_for_delete_expansion": False,
    }


def _candidate_patch_boundary_vertex_loops(
    *,
    source_faces: np.ndarray,
    face_ids: Iterable[int],
) -> tuple[tuple[int, ...], ...]:
    boundary_edges = set(boundary_edges_for_face_patch(source_faces, face_ids))
    components = edge_loop_components(boundary_edges)
    loops: list[tuple[int, ...]] = []
    for comp in components:
        ordered = _order_closed_edge_loop_vertices(set(comp))
        vertices = tuple(int(v) for v in tuple(ordered.get("vertices", ()) or ()))
        if bool(ordered.get("closed", False)) and len(vertices) >= 3:
            loops.append(vertices)
    loops.sort(key=lambda loop: (-len(loop), loop[0] if loop else -1))
    return tuple(loops)












# -----------------------------------------------------------------------------
# Stage 5: trial validation and final mesh application
# -----------------------------------------------------------------------------


def _small_boundary_seal_triangles_for_trial_mesh(
    *,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    generated_vertices: np.ndarray,
    triangles: np.ndarray,
    max_boundary_edges: int = 12,
    max_loop_edges: int = 6,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return conservative closure triangles for tiny damaged-bore residual holes.

    This is not a general mesh repair and it is never used for clean candidates.
    The caller only invokes it for damaged BORE rebuild trials after the normal
    measured two-loop replacement still leaves a very small boundary.  Large or
    open residual boundaries remain a rejection.
    """

    source_arr = np.asarray(source_faces, dtype=np.int64)[:, :3]
    remove_mask = np.ones(len(source_arr), dtype=bool)
    valid_faces = tuple(int(fid) for fid in tuple(face_ids or ()) if 0 <= int(fid) < len(source_arr))
    if valid_faces:
        remove_mask[np.asarray(valid_faces, dtype=np.int64)] = False
    kept_faces = source_arr[remove_mask, :3].copy()
    tri_arr = np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    if tri_arr.size == 0:
        return np.zeros((0, 3), dtype=np.int64), {"used": False, "reason": "no_generated_triangles"}
    output_faces = np.vstack([kept_faces, tri_arr])
    boundary_edges = set(boundary_edges_for_face_patch(output_faces, range(len(output_faces))))
    if not boundary_edges:
        return np.zeros((0, 3), dtype=np.int64), {"used": False, "reason": "already_closed"}
    if len(boundary_edges) > int(max_boundary_edges):
        return np.zeros((0, 3), dtype=np.int64), {"used": False, "reason": "residual_boundary_too_large", "boundary_edge_count": int(len(boundary_edges))}

    components = edge_loop_components(boundary_edges)
    close_tris: list[tuple[int, int, int]] = []
    loop_sizes: list[int] = []
    for comp in components:
        edges = {_edge_key(edge) for edge in tuple(comp or ())}
        ordered = _order_closed_edge_loop_vertices(edges)
        verts = tuple(int(v) for v in tuple(ordered.get("vertices", ()) or ()))
        if not bool(ordered.get("closed", False)) or len(verts) < 3:
            return np.zeros((0, 3), dtype=np.int64), {"used": False, "reason": "residual_boundary_not_closed", "boundary_edge_count": int(len(boundary_edges))}
        if len(verts) > int(max_loop_edges):
            return np.zeros((0, 3), dtype=np.int64), {"used": False, "reason": "residual_boundary_loop_too_large", "boundary_edge_count": int(len(boundary_edges)), "loop_vertex_count": int(len(verts))}
        loop_sizes.append(int(len(verts)))
        if len(verts) == 3:
            close_tris.append((int(verts[0]), int(verts[1]), int(verts[2])))
        else:
            anchor = int(verts[0])
            for i in range(1, len(verts) - 1):
                close_tris.append((anchor, int(verts[i]), int(verts[i + 1])))

    if not close_tris:
        return np.zeros((0, 3), dtype=np.int64), {"used": False, "reason": "no_closure_triangles_created"}
    out = np.asarray(close_tris, dtype=np.int64).reshape((-1, 3))
    sealed_faces = np.vstack([output_faces, out])
    after_edges = int(len(boundary_edges_for_face_patch(sealed_faces, range(len(sealed_faces)))))
    return out, {
        "used": True,
        "reason": "tiny_residual_boundary_closed_for_damaged_bore_trial",
        "boundary_edge_count_before": int(len(boundary_edges)),
        "boundary_edge_count_after_seal": int(after_edges),
        "closure_triangle_count": int(len(out)),
        "closed_loop_count": int(len(loop_sizes)),
        "closed_loop_vertex_counts": tuple(int(v) for v in loop_sizes),
    }


def _trial_rebuild(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    generated_vertices: np.ndarray,
    triangles: np.ndarray,
) -> dict[str, object]:
    remove_mask = np.ones(len(source_faces), dtype=bool)
    valid_faces = tuple(int(fid) for fid in tuple(face_ids or ()) if 0 <= int(fid) < len(source_faces))
    remove_mask[np.asarray(valid_faces, dtype=np.int64)] = False
    kept_faces = source_faces[remove_mask, :3].copy()
    generated_vertices = np.asarray(generated_vertices, dtype=float).reshape((-1, 3))
    triangles = np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    if triangles.size == 0:
        return {"boundary_edge_count_after": -1, "watertight_after": False, "reason": "no_generated_faces"}
    output_vertices = np.asarray(vertices, dtype=float).copy()
    if generated_vertices.size:
        output_vertices = np.vstack([output_vertices, generated_vertices])
    output_faces = np.vstack([kept_faces, triangles])
    boundary_before = _boundary_edge_count(source_faces)
    boundary_count = _boundary_edge_count(output_faces)
    trial_mesh = trimesh.Trimesh(vertices=output_vertices, faces=output_faces, process=False)
    boundary_delta = int(boundary_count) - int(boundary_before)
    return {
        "boundary_edge_count_before": int(boundary_before),
        "boundary_edge_count_after": int(boundary_count),
        "boundary_edge_count_delta": int(boundary_delta),
        "watertight_after": bool(getattr(trial_mesh, "is_watertight", False)),
        "kept_face_count": int(len(kept_faces)),
        "after_face_count": int(len(output_faces)),
        "after_vertex_count": int(len(output_vertices)),
    }



def _apply_rebuild(
    *,
    mesh: trimesh.Trimesh,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    generated_vertices: np.ndarray,
    triangles: np.ndarray,
    color_rebuilt_faces: bool,
    base_face_color: RGBA,
    rebuilt_face_color: RGBA,
) -> dict[str, object]:
    remove_mask = np.ones(len(source_faces), dtype=bool)
    remove_mask[np.asarray(tuple(int(fid) for fid in face_ids), dtype=np.int64)] = False
    kept_faces = source_faces[remove_mask, :3].copy()

    output_vertices = np.asarray(vertices, dtype=float).copy()
    generated_vertices = np.asarray(generated_vertices, dtype=float).reshape((-1, 3))
    if generated_vertices.size:
        output_vertices = np.vstack([output_vertices, generated_vertices])
    triangles = np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    output_faces = np.vstack([kept_faces, triangles])

    rebuilt = trimesh.Trimesh(vertices=output_vertices, faces=output_faces, process=False)
    added_start = int(len(kept_faces))
    added_face_ids = tuple(range(added_start, added_start + int(len(triangles))))

    if color_rebuilt_faces:
        _assign_face_colors(
            rebuilt,
            added_face_ids=added_face_ids,
            base_face_color=base_face_color,
            rebuilt_face_color=rebuilt_face_color,
        )

    rebuilt.remove_unreferenced_vertices()
    return {"mesh": rebuilt, "added_face_ids": added_face_ids}



# -----------------------------------------------------------------------------
# Diagnostics and failure reporting
# -----------------------------------------------------------------------------



# Boundary-match diagnostics moved to rebuild_validation.py in v176p.
# Imported aliases preserve previous private helper names and call sites.

# Failure reporting helpers moved to rebuild_emit.py in v176o.
# Imported aliases preserve the previous private helper names and call sites.


# -----------------------------------------------------------------------------
# Numeric / loop ordering helpers
# -----------------------------------------------------------------------------



# Plan geometry quality validation moved to rebuild_validation.py in v177f.
# Imported alias preserves the previous private helper name and call site.



def _assign_face_colors(
    mesh: trimesh.Trimesh,
    *,
    added_face_ids: tuple[int, ...],
    base_face_color: RGBA,
    rebuilt_face_color: RGBA,
) -> None:
    face_count = int(len(mesh.faces))
    colors = np.tile(np.asarray(base_face_color, dtype=np.uint8).reshape(1, 4), (face_count, 1))
    for fid in added_face_ids:
        if 0 <= int(fid) < face_count:
            colors[int(fid), :] = np.asarray(rebuilt_face_color, dtype=np.uint8).reshape(4)
    try:
        mesh.visual.face_colors = colors
    except Exception:
        pass


def _normalize_face_ids(values: Iterable[int] | None, *, face_count: int) -> tuple[int, ...]:
    result: set[int] = set()
    for raw in tuple(values or ()):
        try:
            fid = int(raw)
        except Exception:
            continue
        if 0 <= fid < int(face_count):
            result.add(fid)
    return tuple(sorted(result))


def _edge_key(edge: object) -> EdgeKey:
    return normalize_edge(edge)



__all__ = ["RebuildResult", "delete_and_rebuild_candidate_region"]
